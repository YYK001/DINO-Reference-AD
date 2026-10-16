"""ShiftADD AeBAD-S P0 v2: Anchor-50K plus priority-reservoir factors."""
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image

from DINOv3.MADEqual.aebad_s_full_p0 import runner as baseline_runner
from DINOv3.MADEqual.aebad_s_full_p0.dataset import (
    DOMAINS,
    TestLocator,
    discover_phase_a,
    load_phase_b_records,
)
from DINOv3.relation_oracle_cache.extract import mad_equal
from DINOv3.relation_reliability.calibration import MADCalibrator, fit_calibrators
from DINOv3.relation_reliability.datasets import Record, split_source_normal
from DINOv3.relation_reliability.official_metrics import validate_official_metric_backend
from DINOv3.relation_reliability.protocol import ECDF_EPSILON
from DINOv3.relation_reliability.superadd_host import (
    construct_exact_host,
    install_import_paths,
    interpolate_score,
    mask_at_patch_grid,
)

from .memory import (
    MemoryBuildResult,
    PartitionedMemory,
    SUBTYPE_NAMES,
    build_partitioned_memory,
    load_official_banks,
    load_partitioned_memory,
)
from .protocol import (
    ANCHOR_QUOTA,
    AUDIT_REVISION,
    BRANCH_CODES,
    BRIGHTNESS2_CONTROL_PROTOCOL_ID,
    CACHE_REVISION,
    CHECKPOINT_SHA256,
    CONTROL_METHODS,
    CONTROL_NEW_METHODS,
    LAYERS,
    MAIN_METHODS,
    MAIN_NEW_METHODS,
    METRIC_NAMES,
    PROTOCOL_ID,
    SAMPLER_REVISION,
    TRANSFORM_REVISION,
    protocol_id_for_variant,
    revisions,
)
from .scoring import observe_partitioned_pil


@dataclass(frozen=True)
class BaselineState:
    output_dir: Path
    official_banks: dict[int, torch.Tensor]


@dataclass(frozen=True)
class CachedObservation:
    anchor_raw: np.ndarray
    factor_raw: np.ndarray
    top1_indices: np.ndarray
    branch_codes: np.ndarray
    output_shape: tuple[int, int]


@dataclass(frozen=True)
class MechanismResult:
    audit_rows: list[dict[str, Any]]
    hit_rows: list[dict[str, Any]]
    shifted_normal_mean: float
    defect_mean: float


def _atomic_json(path: Path, payload: Any) -> None:
    baseline_runner._atomic_json(path, payload)


def _atomic_npz(path: Path, **values: np.ndarray) -> None:
    baseline_runner._atomic_npz(path, **values)


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    baseline_runner._write_csv(path, rows)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _numeric_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    numeric = set(METRIC_NAMES) | {f"delta_vs_official_{name}" for name in METRIC_NAMES}
    numeric |= {"image_count", "label", "image_score"}
    for row in rows:
        value = dict(row)
        for key in numeric & value.keys():
            value[key] = int(value[key]) if key in {"image_count", "label"} else float(value[key])
        output.append(value)
    return output


def _load_baseline_state(output_dir: str | Path) -> BaselineState:
    """Perform the one baseline/output identity preflight."""
    root = Path(output_dir).resolve()
    required = (
        "complete.json", "official_memory.pt", "calibration_stats.csv",
        "group_metrics.csv", "summary.csv", "sample_scores.csv",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"baseline output is incomplete: {missing}")
    complete = json.loads((root / "complete.json").read_text(encoding="utf-8"))
    if (
        complete.get("protocol_id") != baseline_runner.PROTOCOL_ID
        or complete.get("status") != "complete"
        or complete.get("run_type") != "formal"
        or [int(complete.get(key, -1)) for key in (
            "train_count", "memory_count", "calibration_count", "prediction_count",
        )] != [521, 455, 66, 1639]
        or not {"SUPERADD_OFFICIAL", "MAD_EQUAL"}.issubset(complete.get("methods", ()))
    ):
        raise RuntimeError("--baseline-output is not the completed frozen AeBAD-S P0")
    calibration_count = len(list((root / "raw_cache" / "calibration").glob("*.npz")))
    test_count = len(list((root / "raw_cache" / "test").glob("*.npz")))
    if (calibration_count, test_count) != (66, 1639):
        raise RuntimeError("baseline raw cache count must be 66 calibration and 1639 test")
    return BaselineState(
        output_dir=root,
        official_banks=load_official_banks(root / "official_memory.pt"),
    )


def _load_baseline_rows(
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse GT-derived baseline tables only after this run's barrier."""
    return (
        _numeric_rows(_read_csv(output_dir / "group_metrics.csv")),
        _numeric_rows(_read_csv(output_dir / "sample_scores.csv")),
    )


def _write_or_check_run_identity(output: Path, variant: str) -> None:
    path = output / "run_identity.json"
    expected = revisions()
    expected.update({"protocol_id": protocol_id_for_variant(variant), "variant": variant})
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise RuntimeError("output directory contains a different ShiftADD v2 revision")
    else:
        _atomic_json(path, expected)


def _active_paths(main_output: Path) -> set[str]:
    complete = json.loads((main_output / "complete.json").read_text(encoding="utf-8"))
    if (
        complete.get("protocol_id") != PROTOCOL_ID
        or complete.get("status") != "complete"
        or complete.get("mechanism_pass") is not True
    ):
        raise RuntimeError(
            "brightness2 control requires a completed v2 main output with Mechanism PASS"
        )
    rows = _read_csv(main_output / "augmentation_manifest.csv")
    active = {
        str(row["relative_path"])
        for row in rows
        if str(row.get("candidate_active", "")).lower() in {"true", "1"}
    }
    if not active:
        raise RuntimeError("main augmentation manifest has no active factor paths")
    return active


def _main_reuse_rows(main_output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    keep = {"SUPERADD_OFFICIAL", "MAD_EQUAL", "ANCHOR50_EQUAL", "ANCHOR50_MAD"}
    groups = [row for row in _numeric_rows(_read_csv(main_output / "group_metrics.csv")) if row["method"] in keep]
    samples = [row for row in _numeric_rows(_read_csv(main_output / "sample_scores.csv")) if row["method"] in keep]
    return groups, samples


def _method_pass(
    main_groups: Sequence[dict[str, Any]],
    main_summary: Sequence[dict[str, Any]],
    control_groups: Sequence[dict[str, Any]],
    control_summary: Sequence[dict[str, Any]],
) -> bool:
    shifted = _row(main_summary, "SHIFTADD_MAD", "shifted_macro")
    baseline = _row(main_summary, "MAD_EQUAL", "shifted_macro")
    brightness = _row(control_summary, "BRIGHTNESS2_MAD", "shifted_macro")
    by_key = {(row["method"], row["group"]): row for row in main_groups}
    control_by_key = {(row["method"], row["group"]): row for row in control_groups}
    shifted_domains = ("background", "illumination", "view")
    improved_vs_mad = sum(
        by_key[("SHIFTADD_MAD", domain)]["image_AUROC"]
        > by_key[("MAD_EQUAL", domain)]["image_AUROC"]
        for domain in shifted_domains
    )
    improved_vs_brightness = sum(
        by_key[("SHIFTADD_MAD", domain)]["image_AUROC"]
        > control_by_key[("BRIGHTNESS2_MAD", domain)]["image_AUROC"]
        for domain in shifted_domains
    )
    same_drop = (
        by_key[("MAD_EQUAL", "same")]["image_AUROC"]
        - by_key[("SHIFTADD_MAD", "same")]["image_AUROC"]
    )
    return bool(
        shifted["image_AUROC"] - baseline["image_AUROC"] >= 0.03
        and improved_vs_mad >= 2
        and same_drop <= 0.01
        and shifted["AUPRO"] >= baseline["AUPRO"]
        and shifted["image_AUROC"] - brightness["image_AUROC"] >= 0.01
        and improved_vs_brightness >= 2
    )


def _read_cache(path: Path) -> CachedObservation:
    """Read a cache directly; identity checks were completed once at startup."""
    with np.load(path, allow_pickle=False) as payload:
        return CachedObservation(
            anchor_raw=np.asarray(payload["anchor_raw_distances"], dtype=np.float32),
            factor_raw=np.asarray(payload["factor_raw_distances"], dtype=np.float32),
            top1_indices=np.asarray(payload["combined_top1_indices"], dtype=np.int64),
            branch_codes=np.asarray(payload["top1_branch_codes"], dtype=np.int16),
            output_shape=tuple(int(value) for value in payload["output_shape"].tolist()),
        )


def _observe_or_load(
    path: Path,
    image_path: str,
    model: Any,
    memory: PartitionedMemory,
    device: torch.device,
    *,
    allow_observe: bool = True,
) -> CachedObservation:
    if path.is_file():
        return _read_cache(path)
    if not allow_observe:
        raise RuntimeError(f"prediction barrier cache is missing: {path}")
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        downscale = int(model.evaluation_downscale)
        output_shape = (image.height // downscale, image.width // downscale)
        observed = observe_partitioned_pil(model, image, memory, device=device)
    _atomic_npz(
        path,
        anchor_raw_distances=observed.anchor_raw_distances,
        factor_raw_distances=observed.shift_raw_distances,
        combined_top1_indices=observed.combined_top1_indices,
        top1_branch_codes=observed.top1_branch_codes,
        output_shape=np.asarray(output_shape, dtype=np.int32),
    )
    return CachedObservation(
        observed.anchor_raw_distances, observed.shift_raw_distances,
        observed.combined_top1_indices, observed.top1_branch_codes, output_shape,
    )


def _baseline_raw(path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    with np.load(path, allow_pickle=False) as payload:
        return (
            np.asarray(payload["raw_distances"], dtype=np.float32),
            tuple(int(value) for value in payload["output_shape"].tolist()),
        )


def _fit_mad(samples: dict[int, list[np.ndarray]]) -> tuple[MADCalibrator, dict[int, np.ndarray]]:
    pooled = {layer: np.concatenate(samples[layer]) for layer in LAYERS}
    _ecdf, mad = fit_calibrators(pooled)
    return mad, pooled


def _score_raw(raw: np.ndarray, mad: MADCalibrator, prefix: str) -> dict[str, np.ndarray]:
    value = np.asarray(raw, dtype=np.float32)
    return {
        f"{prefix}_EQUAL": value.mean(axis=0),
        f"{prefix}_MAD": mad_equal(value, mad.medians, mad.scales),
    }


def summarize_group_metrics(
    rows: Sequence[dict[str, Any]], methods: Sequence[str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in methods:
        by_domain = {str(row["group"]): row for row in rows if row["method"] == method}
        if set(by_domain) != set(DOMAINS):
            raise RuntimeError(f"incomplete domain metrics for {method}")
        for scope, domains in (
            ("four_domain_macro", DOMAINS),
            ("shifted_macro", ("background", "illumination", "view")),
        ):
            result: dict[str, Any] = {"method": method, "scope": scope}
            for metric in METRIC_NAMES:
                result[metric] = float(np.mean([float(by_domain[d][metric]) for d in domains]))
            output.append(result)
    for scope in ("four_domain_macro", "shifted_macro"):
        official = next(row for row in output if row["method"] == "SUPERADD_OFFICIAL" and row["scope"] == scope)
        for row in output:
            if row["scope"] == scope:
                for metric in METRIC_NAMES:
                    row[f"delta_vs_official_{metric}"] = float(row[metric] - official[metric])
    return output


def _add_domain_deltas(rows: Sequence[dict[str, Any]]) -> None:
    for domain in DOMAINS:
        official = next(row for row in rows if row["method"] == "SUPERADD_OFFICIAL" and row["group"] == domain)
        for row in rows:
            if row["group"] == domain:
                for metric in METRIC_NAMES:
                    row[f"delta_vs_official_{metric}"] = float(row[metric] - official[metric])


def _evaluate_new_methods(
    *,
    methods: Sequence[str],
    records: Sequence[Record],
    locators: Sequence[TestLocator],
    cache_root: Path,
    model: Any,
    anchor_mad: MADCalibrator | None,
    factor_mad: MADCalibrator,
    factor_prefix: str,
    mechanism_anchor_mad: MADCalibrator | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], MechanismResult | None]:
    groups: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    audit_totals: dict[tuple[str, int | str, str], list[float | int]] = {}
    hit_counts: dict[tuple[str, str, int, str], list[int]] = {}
    for domain in DOMAINS:
        indices = [i for i, record in enumerate(records) if record.domain == domain]
        labels = [records[i].label for i in indices]
        masks: list[np.ndarray] = []
        maps = {method: [] for method in methods}
        for position, index in enumerate(indices, 1):
            record, locator = records[index], locators[index]
            observed = _read_cache(cache_root / f"{locator.sample_id}.npz")
            scores = _score_raw(observed.factor_raw, factor_mad, factor_prefix)
            if anchor_mad is not None:
                scores.update(_score_raw(observed.anchor_raw, anchor_mad, "ANCHOR50"))
            evaluation_mask, patch_mask = _load_evaluation_and_patch_masks(
                record, observed.output_shape, observed.anchor_raw.shape[1:],
            )
            masks.append(evaluation_mask)
            if mechanism_anchor_mad is not None:
                _accumulate_mechanism(
                    audit_totals, hit_counts, record, observed,
                    patch_mask, mechanism_anchor_mad,
                )
            for method in methods:
                final = interpolate_score(scores[method], observed.output_shape).astype(np.float32)
                maps[method].append(final)
                samples.append({
                    "sample_id": locator.sample_id, "relative_path": locator.relative_path,
                    "domain": domain, "label": record.label, "method": method,
                    "image_score": float(final.max()),
                })
            print(f"[SHIFTADD-AEBAD-S-P0-V2] score {domain} {position}/{len(indices)}", flush=True)
        for method in methods:
            groups.append({
                "method": method, "group": domain, "image_count": len(indices),
                **baseline_runner._evaluate(labels, masks, maps[method]),
            })
    mechanism = (
        _finish_mechanism(audit_totals, hit_counts)
        if mechanism_anchor_mad is not None
        else None
    )
    return groups, samples, mechanism


def _load_evaluation_and_patch_masks(
    record: Record,
    output_shape: tuple[int, int],
    patch_grid: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    if record.label == 0:
        return (
            np.zeros(output_shape, dtype=np.uint8),
            np.ones(patch_grid, dtype=bool),
        )
    if not record.mask_path:
        raise RuntimeError(f"anomaly record lacks mask: {record.path}")
    with Image.open(record.mask_path) as source:
        gray = source.convert("L")
        evaluation = gray.resize(
            (output_shape[1], output_shape[0]), Image.Resampling.NEAREST,
        )
        patch = mask_at_patch_grid(gray, patch_grid)
        return (
            (np.asarray(evaluation, dtype=np.uint8) > 0).astype(np.uint8),
            patch,
        )


def _add_total(
    totals: dict[tuple[str, int | str, str], list[float | int]],
    key: tuple[str, int | str, str],
    values: np.ndarray,
) -> None:
    item = totals.setdefault(key, [0.0, 0])
    item[0] = float(item[0]) + float(np.asarray(values, dtype=np.float64).sum())
    item[1] = int(item[1]) + int(values.size)


def _accumulate_mechanism(
    totals: dict[tuple[str, int | str, str], list[float | int]],
    hit_counts: dict[tuple[str, str, int, str], list[int]],
    record: Record,
    observed: CachedObservation,
    patch_mask: np.ndarray,
    anchor_mad: MADCalibrator,
) -> None:
    shifted = {"background", "illumination", "view"}
    region = "normal" if record.label == 0 else "defect_region"
    for offset, layer in enumerate(LAYERS):
        reduction = (
            (observed.anchor_raw[offset] - observed.factor_raw[offset])
            / (float(anchor_mad.scales[offset]) + float(ECDF_EPSILON))
        )
        minimum = float(reduction.min())
        if minimum < -1e-6:
            raise RuntimeError(
                f"factor reduction violates partition minimum at L{layer}: {minimum}"
            )
        selected_reduction = reduction[patch_mask]
        if record.label == 0 and record.domain in shifted:
            _add_total(totals, ("shifted_normal", layer, record.domain), selected_reduction)
            _add_total(totals, ("shifted_normal", layer, "pooled"), selected_reduction)
            _add_total(totals, ("shifted_normal", "all", "patch_pooled"), selected_reduction)
        elif record.label == 1:
            _add_total(totals, ("defect", layer, record.domain), selected_reduction)
            if record.domain in shifted:
                _add_total(totals, ("defect", layer, "pooled"), selected_reduction)
                _add_total(totals, ("defect", "all", "patch_pooled"), selected_reduction)
        selected = observed.branch_codes[offset][patch_mask]
        for branch, code in BRANCH_CODES.items():
            item = hit_counts.setdefault((record.domain, region, layer, branch), [0, 0])
            item[0] += int(np.count_nonzero(selected == code))
            item[1] += int(selected.size)


def _finish_mechanism(
    totals: dict[tuple[str, int | str, str], list[float | int]],
    hit_counts: dict[tuple[str, str, int, str], list[int]],
) -> MechanismResult:
    audit_rows: list[dict[str, Any]] = []
    for (population, layer, domain), (total, count) in sorted(
        totals.items(), key=lambda item: tuple(str(value) for value in item[0]),
    ):
        audit_rows.append({
            "population": population, "layer": layer, "domain": domain,
            "patch_count": int(count),
            "mean_reduction": float(total) / int(count) if count else float("nan"),
        })
    normal_total, normal_count = totals[("shifted_normal", "all", "patch_pooled")]
    defect_total, defect_count = totals[("defect", "all", "patch_pooled")]
    normal_mean = float(normal_total) / int(normal_count)
    defect_mean = float(defect_total) / int(defect_count)
    hit_rows = [
        {"domain": domain, "region": region, "layer": layer, "branch": branch, "hit_count": count, "patch_count": total, "hit_rate": count / total if total else 0.0}
        for (domain, region, layer, branch), (count, total) in sorted(hit_counts.items())
    ]
    domain_regions = sorted({(row["domain"], row["region"], row["layer"]) for row in hit_rows})
    hit_rows.extend(
        {"domain": domain, "region": region, "layer": layer, "branch": branch, "hit_count": 0, "patch_count": total, "hit_rate": 0.0}
        for domain, region, layer in domain_regions
        for branch in ("POSE_NOOP", "SCENE_NOOP")
        for total in [next(row["patch_count"] for row in hit_rows if row["domain"] == domain and row["region"] == region and row["layer"] == layer)]
    )
    return MechanismResult(audit_rows, hit_rows, normal_mean, defect_mean)


def _row(rows: Sequence[dict[str, Any]], method: str, scope: str) -> dict[str, Any]:
    return next(row for row in rows if row["method"] == method and row.get("scope") == scope)


def _mechanism_pass(summary: Sequence[dict[str, Any]], normal_r: float, defect_r: float) -> bool:
    shift = _row(summary, "SHIFTADD_MAD", "shifted_macro")
    baseline = _row(summary, "MAD_EQUAL", "shifted_macro")
    anchor = _row(summary, "ANCHOR50_MAD", "shifted_macro")
    return bool(
        shift["image_AUROC"] - baseline["image_AUROC"] >= 0.01
        and shift["AUPRO"] >= baseline["AUPRO"]
        and shift["image_AUROC"] > anchor["image_AUROC"]
        and normal_r > defect_r
    )


def _objective_report(
    *, variant: str, summary: Sequence[dict[str, Any]], mechanism_pass: bool | None,
    normal_r: float | None, defect_r: float | None, method_pass: bool | None,
) -> str:
    title = "ShiftADD AeBAD-S P0 v2" if variant == "shiftadd" else "Brightness2 matched-control AeBAD-S P0 v2"
    lines = [
        f"# {title}", "", f"Protocol: `{protocol_id_for_variant(variant)}`", "",
        "All new methods use the frozen Official Anchor-50K and one partitioned exact query scan. MAD fusion calls the existing project implementation.",
        "", "AUPRO backend: project ADEval CUDA, `nstrips=200`.", "",
        "## Macro results", "",
        "| method | scope | Image AUROC | Image AUPR | Pixel AUROC | Pixel AUPR | AUPRO |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['method']} | {row['scope']} | {row['image_AUROC']:.6f} | {row['image_AUPR']:.6f} | "
            f"{row['pixel_AUROC']:.6f} | {row['pixel_AUPR']:.6f} | {row['AUPRO']:.6f} |"
        )
    if variant == "shiftadd":
        lines.extend((
            "", "## Frozen decisions", "",
            f"Patch-pooled normalized reduction: shifted normal `{normal_r:.6f}`, GT defect `{defect_r:.6f}`.",
            f"Mechanism PASS: `{'PASS' if mechanism_pass else 'FAIL'}`.",
            "Method PASS: `PENDING_BRIGHTNESS2_CONTROL`.",
        ))
    else:
        lines.extend((
            "", "This matched control does not make a Mechanism PASS decision.",
            f"Combined main-plus-control Method PASS: `{'PASS' if method_pass else 'FAIL'}`.",
        ))
    lines.extend(("", "Published MMR/MSFR results are context only and are not same-protocol direct comparisons.", ""))
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    variant = str(args.variant)
    protocol_id = protocol_id_for_variant(variant)
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("ShiftADD AeBAD-S formal P0 requires CUDA")
    validate_official_metric_backend()

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset_root, train_records, locators = discover_phase_a(args.aebad_root, formal=True)
    memory_records, calibration_records = split_source_normal(train_records, threshold_fraction=8)
    if (len(train_records), len(memory_records), len(calibration_records), len(locators)) != (521, 455, 66, 1639):
        raise RuntimeError("AeBAD-S frozen count preflight failed")
    baseline = _load_baseline_state(args.baseline_output)
    _write_or_check_run_identity(output, variant)

    main_output = Path(args.main_output).resolve() if args.main_output else None
    active_paths = None
    main_anchor_indices: dict[int, np.ndarray] | None = None
    if variant == "brightness2_control":
        if main_output is None:
            raise RuntimeError("brightness2_control requires --main-output")
        active_paths = _active_paths(main_output)
        main_memory = load_partitioned_memory(
            main_output / "shiftadd_memory.pt", variant="shiftadd", device=torch.device("cpu"),
        )
        if main_memory is None:
            raise RuntimeError("completed main output lacks its v2 memory")
        main_anchor_indices = {
            layer: main_memory.anchor_official_indices[layer].copy() for layer in LAYERS
        }
        del main_memory

    install_import_paths(Path(args.project_root), Path(args.official_root), Path(args.dinov3_root))
    model, host_config = construct_exact_host(
        Path(args.official_root), Path(args.dinov3_root), Path(args.checkpoint), CHECKPOINT_SHA256,
    )
    memory = load_partitioned_memory(output / "shiftadd_memory.pt", variant=variant, device=device)
    if memory is None:
        built = build_partitioned_memory(
            model, baseline.official_banks, memory_records, dataset_root=dataset_root,
            device=device, output_dir=output, variant=variant, active_factor_paths=active_paths,
        )
        memory = built.memory
    if variant == "brightness2_control":
        if main_anchor_indices is None or any(
            not np.array_equal(memory.anchor_official_indices[layer], main_anchor_indices[layer])
            for layer in LAYERS
        ):
            raise RuntimeError("main and brightness2 Official anchors differ")

    _atomic_json(output / "memory_manifest.json", {
        "protocol_id": protocol_id, "variant": variant, "source_count": len(memory_records),
        "anchor_entries_per_layer": ANCHOR_QUOTA, "factor_entries_per_layer": ANCHOR_QUOTA,
        "layers": list(LAYERS), "host_config": host_config,
        "transform_revision": TRANSFORM_REVISION, "sampler_revision": SAMPLER_REVISION,
        "audit_revision": AUDIT_REVISION, "cache_revision": CACHE_REVISION,
    })

    anchor_samples = {layer: [] for layer in LAYERS}
    factor_samples = {layer: [] for layer in LAYERS}
    calibration_root = output / "raw_cache" / "calibration"
    for index, record in enumerate(calibration_records, 1):
        observed = _observe_or_load(
            calibration_root / f"calibration_{index - 1:04d}.npz", record.path,
            model, memory, device,
        )
        for offset, layer in enumerate(LAYERS):
            anchor_samples[layer].append(observed.anchor_raw[offset].reshape(-1))
            factor_samples[layer].append(observed.factor_raw[offset].reshape(-1))
        print(f"[SHIFTADD-AEBAD-S-P0-V2] calibration {index}/{len(calibration_records)}", flush=True)
    anchor_mad, anchor_pooled = _fit_mad(anchor_samples)
    factor_mad, factor_pooled = _fit_mad(factor_samples)
    factor_name = "SHIFTADD" if variant == "shiftadd" else "BRIGHTNESS2"
    calibration_rows = []
    for offset, layer in enumerate(LAYERS):
        for bank, mad, pooled in (
            ("ANCHOR50", anchor_mad, anchor_pooled),
            (factor_name, factor_mad, factor_pooled),
        ):
            calibration_rows.append({
                "bank": bank, "layer": layer, "patch_count": int(pooled[layer].size),
                "median": float(mad.medians[offset]), "mad_scale_1p4826": float(mad.scales[offset]),
            })
    _write_csv(output / "calibration_stats.csv", calibration_rows)

    target_root = output / "raw_cache" / "test"
    for index, locator in enumerate(locators, 1):
        _observe_or_load(target_root / f"{locator.sample_id}.npz", locator.path, model, memory, device)
        print(f"[SHIFTADD-AEBAD-S-P0-V2] test raw {index}/{len(locators)}", flush=True)
    _atomic_json(output / "prediction_barrier.json", {
        "protocol_id": protocol_id, "status": "complete", "prediction_count": len(locators),
        "ground_truth_parsed": False,
    })

    baseline_groups, baseline_samples = _load_baseline_rows(baseline.output_dir)
    main_groups_all: list[dict[str, Any]] | None = None
    main_summary_all: list[dict[str, Any]] | None = None
    main_reuse_groups: list[dict[str, Any]] | None = None
    main_reuse_samples: list[dict[str, Any]] | None = None
    if variant == "brightness2_control":
        main_groups_all = _numeric_rows(_read_csv(main_output / "group_metrics.csv"))
        main_summary_all = _numeric_rows(_read_csv(main_output / "summary.csv"))
        main_reuse_groups, main_reuse_samples = _main_reuse_rows(main_output)

    records = load_phase_b_records(dataset_root, locators, formal=True)
    manifest_rows = []
    for locator, record in zip(locators, records, strict=True):
        row = record.as_row()
        row.update({"sample_id": locator.sample_id, "relative_path": locator.relative_path})
        manifest_rows.append(row)
    _write_csv(output / "dataset_manifest.csv", manifest_rows)

    if variant == "shiftadd":
        base_groups, base_samples = baseline_groups, baseline_samples
        new_methods = MAIN_NEW_METHODS
        methods = MAIN_METHODS
        factor_prefix = "SHIFTADD"
        eval_anchor_mad: MADCalibrator | None = anchor_mad
    else:
        if main_reuse_groups is None or main_reuse_samples is None:
            raise AssertionError("control preflight rows were not retained")
        base_groups, base_samples = main_reuse_groups, main_reuse_samples
        new_methods = CONTROL_NEW_METHODS
        methods = CONTROL_METHODS
        factor_prefix = "BRIGHTNESS2"
        eval_anchor_mad = None
    new_groups, new_samples, combined_mechanism = _evaluate_new_methods(
        methods=new_methods, records=records, locators=locators, cache_root=target_root,
        model=model, anchor_mad=eval_anchor_mad, factor_mad=factor_mad,
        factor_prefix=factor_prefix,
        mechanism_anchor_mad=anchor_mad if variant == "shiftadd" else None,
    )
    group_rows = list(base_groups) + new_groups
    _add_domain_deltas(group_rows)
    summary_rows = summarize_group_metrics(group_rows, methods)
    sample_rows = list(base_samples) + new_samples

    mechanism = None
    method = None
    normal_r = defect_r = None
    if variant == "shiftadd":
        if combined_mechanism is None:
            raise AssertionError("main evaluation did not retain mechanism audit")
        normal_r = combined_mechanism.shifted_normal_mean
        defect_r = combined_mechanism.defect_mean
        _write_csv(output / "mechanism_audit.csv", combined_mechanism.audit_rows)
        _write_csv(output / "nn_branch_hits.csv", combined_mechanism.hit_rows)
        mechanism = _mechanism_pass(summary_rows, normal_r, defect_r)
    else:
        _write_csv(output / "mechanism_audit.csv", [{"status": "not_applicable_to_brightness2_control"}])
        _write_csv(output / "nn_branch_hits.csv", [{"status": "not_applicable_to_brightness2_control"}])
        if main_groups_all is None or main_summary_all is None:
            raise AssertionError("control main metrics were not retained")
        method = _method_pass(main_groups_all, main_summary_all, group_rows, summary_rows)

    _write_csv(output / "sample_scores.csv", sample_rows)
    _write_csv(output / "group_metrics.csv", group_rows)
    _write_csv(output / "summary.csv", summary_rows)
    (output / "objective_report.md").write_text(
        _objective_report(
            variant=variant, summary=summary_rows, mechanism_pass=mechanism,
            normal_r=normal_r, defect_r=defect_r, method_pass=method,
        ),
        encoding="utf-8",
    )
    _atomic_json(output / "complete.json", {
        "protocol_id": protocol_id, "status": "complete", "run_type": "formal",
        "variant": variant, "train_count": 521, "memory_count": 455,
        "calibration_count": 66, "prediction_count": 1639, "methods": list(methods),
        "mechanism_pass": mechanism,
        "method_pass": "method_pass_pending_brightness2_control" if variant == "shiftadd" else method,
        "shared_query_raw_tensor": True, "aupro_backend": "ADEval CUDA nstrips=200",
    })
    print(f"SHIFTADD_AEBAD_S_P0_V2_COMPLETE={output}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ShiftADD AeBAD-S P0 v2")
    parser.add_argument("--aebad-root", required=True)
    parser.add_argument("--baseline-output", required=True)
    parser.add_argument("--main-output")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--dinov3-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variant", choices=("shiftadd", "brightness2_control"), default="shiftadd")
    parser.add_argument("--device", default="cuda:0")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
