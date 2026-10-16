"""AeBAD-S target-normal few-shot + Exact SuperADD + MAD-Equal Seed42."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from DINOv3.MADEqual.aebad_s_broad6_compose2 import mmr_compatible_eval as mmr_eval
from DINOv3.MADEqual.aebad_s_full_p0 import runner as aebad_runner
from DINOv3.MADEqual.aebad_s_full_p0.dataset import (
    discover_phase_a,
    load_phase_b_records,
)
from DINOv3.MADEqual.robustad_target_fewshot_p0 import runner as robust_runner
from DINOv3.MADEqual.robustad_target_fewshot_p0.scoring import observe_fewshot_pil
from DINOv3.relation_reliability.calibration import MADCalibrator, fit_calibrators
from DINOv3.relation_reliability.datasets import Record, split_source_normal
from DINOv3.relation_reliability.official_metrics import validate_official_metric_backend
from DINOv3.relation_reliability.protocol import LAYERS
from DINOv3.relation_reliability.superadd_host import (
    build_exact_official_memory,
    construct_exact_host,
    install_import_paths,
)

from .support import (
    DOMAINS,
    EXPECTED_HELDOUT,
    SHOT_COUNTS,
    SupportFeatures,
    extract_or_load_support,
    relative_path,
    select_source_extra,
    select_target_support,
    support_manifest_rows,
)


PROTOCOL_ID = "mad_equal_aebad_s_target_fewshot_seed42_v1"
CACHE_REVISION = "fewshot_raw_v1"
SEED = 42
CHECKPOINT_SHA256 = "7c1da9a54b3bdb333f5ebc42e404b7f19b1b5bed504877623c9dc87397f41488"
METHODS = (
    "SUPERADD_OFFICIAL",
    "MAD_EQUAL",
    "SOURCE_EXTRA_K1_MAD",
    "SOURCE_EXTRA_K2_MAD",
    "SOURCE_EXTRA_K4_MAD",
    "SOURCE_EXTRA_K8_MAD",
    "TARGET_MATCHED_K1_MAD",
    "TARGET_MATCHED_K2_MAD",
    "TARGET_MATCHED_K4_MAD",
    "TARGET_MATCHED_K8_MAD",
)
METRICS = ("image_AUROC", "image_AUPR", "pixel_AUROC", "pixel_AUPR", "AUPRO")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **values)
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    if not values:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in values for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)
    os.replace(temporary, path)


def _seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def _save_memory(path: Path, model: Any, *, source_count: int) -> None:
    payload: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "source_count": int(source_count),
    }
    for layer in LAYERS:
        payload[f"bank_{layer}"] = model.prototype_embeddings[layer].detach().cpu()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _load_memory(path: Path, model: Any, *, source_count: int, device: torch.device) -> bool:
    if not path.is_file():
        return False
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("protocol_id") != PROTOCOL_ID or int(payload.get("source_count", -1)) != source_count:
        raise RuntimeError("Official memory checkpoint belongs to another protocol")
    banks: dict[int, torch.Tensor] = {}
    for layer in LAYERS:
        value = payload.get(f"bank_{layer}")
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != (100_000, 1280)
            or not torch.isfinite(value).all()
        ):
            raise RuntimeError(f"invalid Official memory L{layer}")
        banks[layer] = value.to(device=device, dtype=torch.float32)
    model.prototype_embeddings = banks
    return True


def _save_mad(path: Path, mad: MADCalibrator) -> None:
    _atomic_npz(
        path,
        protocol_id=np.asarray(PROTOCOL_ID),
        medians=np.asarray(mad.medians, np.float32),
        scales=np.asarray(mad.scales, np.float32),
    )


def _load_mad(path: Path) -> MADCalibrator | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as payload:
        if str(payload["protocol_id"].item()) != PROTOCOL_ID:
            raise RuntimeError("MAD checkpoint belongs to another protocol")
        medians = np.asarray(payload["medians"], np.float32)
        scales = np.asarray(payload["scales"], np.float32)
    if medians.shape != (4,) or scales.shape != (4,) or not np.isfinite(medians).all() or np.any(scales <= 0):
        raise RuntimeError("invalid source MAD checkpoint")
    return MADCalibrator(medians, scales)


def _calibration_raw(
    path: Path,
    record: Record,
    *,
    dataset_root: Path,
    model: Any,
    device: torch.device,
) -> np.ndarray:
    relative = relative_path(record, dataset_root)
    if path.is_file():
        with np.load(path, allow_pickle=False) as payload:
            if str(payload["protocol_id"].item()) != PROTOCOL_ID:
                raise RuntimeError(f"calibration cache protocol mismatch: {path}")
            if str(payload["relative_path"].item()) != relative:
                raise RuntimeError(f"calibration cache path mismatch: {path}")
            raw = np.asarray(payload["raw_distances"], np.float32)
    else:
        with Image.open(record.path) as source:
            raw = np.asarray(
                aebad_runner.observe_pil(model, source.convert("RGB"), device=device).raw_distances,
                np.float32,
            )
        _atomic_npz(
            path,
            protocol_id=np.asarray(PROTOCOL_ID),
            relative_path=np.asarray(relative),
            raw_distances=raw,
        )
    if raw.ndim != 3 or raw.shape[0] != len(LAYERS) or not np.isfinite(raw).all():
        raise RuntimeError(f"invalid calibration raw cache: {path}")
    return raw


def _fit_source_mad(
    calibration: Sequence[Record],
    *,
    dataset_root: Path,
    model: Any,
    device: torch.device,
    output: Path,
) -> MADCalibrator:
    pooled: dict[int, list[np.ndarray]] = {layer: [] for layer in LAYERS}
    for index, record in enumerate(calibration, start=1):
        raw = _calibration_raw(
            output / "raw_cache" / "calibration" / f"{index:03d}.npz",
            record,
            dataset_root=dataset_root,
            model=model,
            device=device,
        )
        for offset, layer in enumerate(LAYERS):
            pooled[layer].append(raw[offset].reshape(-1))
        print(f"[AEBAD-TARGET-FEWSHOT] calibration {index}/{len(calibration)}", flush=True)
    samples = {layer: np.concatenate(values) for layer, values in pooled.items()}
    _ecdf, mad = fit_calibrators(samples)
    _write_csv(output / "calibration_stats.csv", [
        {
            "layer": layer,
            "patch_count": int(samples[layer].size),
            "median": float(mad.medians[offset]),
            "mad_scale_1p4826": float(mad.scales[offset]),
        }
        for offset, layer in enumerate(LAYERS)
    ])
    _save_mad(output / "source_mad_state.npz", mad)
    return mad


def _support_tensors(
    features: Sequence[SupportFeatures], *, device: torch.device,
) -> dict[int, list[torch.Tensor]]:
    return robust_runner._support_tensors(features, device=device)


def _save_observation(
    path: Path,
    *,
    relative: str,
    output_shape: tuple[int, int],
    observed: Any,
) -> None:
    values: dict[str, np.ndarray] = {
        "protocol_id": np.asarray(PROTOCOL_ID),
        "cache_revision": np.asarray(CACHE_REVISION),
        "relative_path": np.asarray(relative),
        "output_shape": np.asarray(output_shape, np.int32),
        "official_raw": np.asarray(observed.official_raw, np.float32),
    }
    for shot in SHOT_COUNTS:
        values[f"source_raw_{shot}"] = np.asarray(observed.source_raw[shot], np.float32)
        values[f"target_raw_{shot}"] = np.asarray(observed.target_raw[shot], np.float32)
        values[f"source_support_{shot}"] = np.asarray(observed.source_support_raw[shot], np.float32)
        values[f"target_support_{shot}"] = np.asarray(observed.target_support_raw[shot], np.float32)
    _atomic_npz(path, **values)


def _load_observation(path: Path, *, relative: str) -> tuple[dict[str, np.ndarray], tuple[int, int]]:
    with np.load(path, allow_pickle=False) as payload:
        if str(payload["protocol_id"].item()) != PROTOCOL_ID:
            raise RuntimeError(f"raw cache protocol mismatch: {path}")
        if str(payload["cache_revision"].item()) != CACHE_REVISION:
            raise RuntimeError(f"raw cache revision mismatch: {path}")
        if str(payload["relative_path"].item()) != relative:
            raise RuntimeError(f"raw cache path mismatch: {path}")
        output_shape = tuple(int(value) for value in payload["output_shape"].tolist())
        names = [
            "official_raw",
            *[f"source_raw_{shot}" for shot in SHOT_COUNTS],
            *[f"target_raw_{shot}" for shot in SHOT_COUNTS],
            *[f"source_support_{shot}" for shot in SHOT_COUNTS],
            *[f"target_support_{shot}" for shot in SHOT_COUNTS],
        ]
        values = {name: np.asarray(payload[name], np.float32) for name in names}
    shape = values["official_raw"].shape
    if (
        len(output_shape) != 2
        or len(shape) != 3
        or shape[0] != len(LAYERS)
        or any(value.shape != shape or not np.isfinite(value).all() for value in values.values())
    ):
        raise RuntimeError(f"invalid raw cache: {path}")
    return values, output_shape


def _delegated_method(method: str) -> str:
    return "OFFICIAL" if method == "SUPERADD_OFFICIAL" else method


def _method_map(
    values: Mapping[str, np.ndarray],
    method: str,
    mad: MADCalibrator,
    output_shape: tuple[int, int],
) -> np.ndarray:
    return robust_runner._method_map(
        values, _delegated_method(method), mad, output_shape,
    )


def summarize_metrics(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in METHODS:
        by_domain = {
            str(row["domain"]): row for row in rows if row["method"] == method
        }
        if set(by_domain) != set(DOMAINS):
            raise RuntimeError(f"incomplete domain metrics for {method}")
        for scope, domains in (
            ("four_domain_macro", DOMAINS),
            ("shifted_macro", ("background", "illumination", "view")),
        ):
            value: dict[str, Any] = {"method": method, "scope": scope}
            for metric in METRICS:
                value[metric] = mean(float(by_domain[domain][metric]) for domain in domains)
            output.append(value)
    return output


def _comparison_rows(
    domain_rows: Sequence[dict[str, Any]], summary_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    combined = [dict(row, scope=row["domain"]) for row in domain_rows] + list(summary_rows)
    output: list[dict[str, Any]] = []
    for row in combined:
        method = str(row["method"])
        if not method.startswith("TARGET_MATCHED_"):
            continue
        shot = int(method.split("_")[-2][1:])
        scope = str(row["scope"])
        baseline = next(
            item for item in combined
            if item["method"] == "MAD_EQUAL" and item["scope"] == scope
        )
        control_name = f"SOURCE_EXTRA_K{shot}_MAD"
        control = next(
            item for item in combined
            if item["method"] == control_name and item["scope"] == scope
        )
        value: dict[str, Any] = {"method": method, "scope": scope, "K": shot}
        for metric in METRICS:
            value[f"delta_vs_MAD_EQUAL_{metric}"] = float(row[metric]) - float(baseline[metric])
            value[f"delta_vs_{control_name}_{metric}"] = float(row[metric]) - float(control[metric])
        output.append(value)
    return output


def _k_change_rows(summary_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for branch in ("SOURCE_EXTRA", "TARGET_MATCHED"):
        for scope in ("four_domain_macro", "shifted_macro"):
            first = next(row for row in summary_rows if row["method"] == f"{branch}_K1_MAD" and row["scope"] == scope)
            last = next(row for row in summary_rows if row["method"] == f"{branch}_K8_MAD" and row["scope"] == scope)
            row: dict[str, Any] = {"branch": branch, "scope": scope, "comparison": "K8_minus_K1"}
            for metric in METRICS:
                row[f"delta_{metric}"] = float(last[metric]) - float(first[metric])
            output.append(row)
    return output


def _report(
    *,
    support_rows: Sequence[dict[str, Any]],
    domain_rows: Sequence[dict[str, Any]],
    summary_rows: Sequence[dict[str, Any]],
    comparison_rows: Sequence[dict[str, Any]],
    k_change_rows: Sequence[dict[str, Any]],
    mechanism_macro: Sequence[dict[str, Any]],
    mmr_domain_rows: Sequence[dict[str, Any]],
    mmr_summary: Sequence[dict[str, Any]],
    timing: Sequence[dict[str, Any]],
) -> str:
    def number(value: Any, signed: bool = False) -> str:
        return f"{float(value):+.6f}" if signed else f"{float(value):.6f}"

    lines = [
        "# AeBAD-S Target-normal Few-shot + Exact SuperADD + MAD-Equal",
        "",
        f"Protocol: `{PROTOCOL_ID}`. Seed42 formal protocol; no automatic best-K selection, PASS/FAIL, or significance claim.",
        "",
        "Official memory is rebuilt from all 521 ordered source normals with the existing Exact Official builder. All MAD methods share one source-only MAD fitted from the 66 unaugmented source calibration images.",
        "",
        "Each domain contributes a fixed nested eight-image real target-normal support pool. The same eight images are excluded from every method and every K; all 1,149 anomalies remain in the common 1,607-image held-out set.",
        "",
        "## Support pools",
        "",
        "| branch | domain | index | relative path | K1 | K2 | K4 | K8 |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in support_rows:
        lines.append(
            f"| {row['branch']} | {row['domain']} | {row['support_index']} | {row['relative_path']} | "
            f"{row['included_in_K1']} | {row['included_in_K2']} | {row['included_in_K4']} | {row['included_in_K8']} |"
        )
    lines += [
        "", "## Common held-out set", "",
        "| domain | support normals excluded | held-out normal | held-out anomaly | held-out total |",
        "|---|---:|---:|---:|---:|",
    ]
    for domain in DOMAINS:
        normal, anomaly, total = EXPECTED_HELDOUT[domain]
        lines.append(f"| {domain} | 8 | {normal} | {anomaly} | {total} |")
    lines += [
        "| total | 32 | 458 | 1,149 | 1,607 |",
        "", "## Per-domain full-protocol metrics", "",
        "| method | domain | N | Image AUROC | Image AUPR | Pixel AUROC | Pixel AUPR | AUPRO |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in domain_rows:
        lines.append(
            f"| {row['method']} | {row['domain']} | {row['image_count']} | {number(row['image_AUROC'])} | "
            f"{number(row['image_AUPR'])} | {number(row['pixel_AUROC'])} | "
            f"{number(row['pixel_AUPR'])} | {number(row['AUPRO'])} |"
        )
    lines += [
        "", "## Macro metrics", "",
        "| method | scope | Image AUROC | Image AUPR | Pixel AUROC | Pixel AUPR | AUPRO |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['method']} | {row['scope']} | {number(row['image_AUROC'])} | "
            f"{number(row['image_AUPR'])} | {number(row['pixel_AUROC'])} | "
            f"{number(row['pixel_AUPR'])} | {number(row['AUPRO'])} |"
        )
    lines += [
        "", "## Target-matched differences", "",
        "| method | scope | K | d Image AUROC vs MAD | d Image AUROC vs source-extra | d AUPRO vs MAD | d AUPRO vs source-extra |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparison_rows:
        shot = int(row["K"])
        lines.append(
            f"| {row['method']} | {row['scope']} | {shot} | "
            f"{number(row['delta_vs_MAD_EQUAL_image_AUROC'], True)} | "
            f"{number(row[f'delta_vs_SOURCE_EXTRA_K{shot}_MAD_image_AUROC'], True)} | "
            f"{number(row['delta_vs_MAD_EQUAL_AUPRO'], True)} | "
            f"{number(row[f'delta_vs_SOURCE_EXTRA_K{shot}_MAD_AUPRO'], True)} |"
        )
    lines += [
        "", "## K1 to K8 changes", "",
        "| branch | scope | d Image AUROC | d Image AUPR | d Pixel AUROC | d Pixel AUPR | d AUPRO |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in k_change_rows:
        lines.append(
            f"| {row['branch']} | {row['scope']} | {number(row['delta_image_AUROC'], True)} | "
            f"{number(row['delta_image_AUPR'], True)} | {number(row['delta_pixel_AUROC'], True)} | "
            f"{number(row['delta_pixel_AUPR'], True)} | {number(row['delta_AUPRO'], True)} |"
        )
    lines += [
        "", "## MMR-compatible I-AUROC", "",
        "This is output-side MMR-compatible post-evaluation, not a full MMR input-protocol reproduction; DINO features still come from the full original images.",
        "",
        "| method | domain | full protocol | MMR-compatible | delta |",
        "|---|---|---:|---:|---:|",
    ]
    for row in mmr_domain_rows:
        lines.append(
            f"| {row['method']} | {row['domain']} | {number(row['full_protocol_image_AUROC'])} | "
            f"{number(row['mmr_compatible_image_AUROC'])} | {number(row['delta'], True)} |"
        )
    lines += [
        "",
        "| method | scope | full protocol | MMR-compatible | delta |",
        "|---|---|---:|---:|---:|",
    ]
    for row in mmr_summary:
        lines.append(
            f"| {row['method']} | {row['scope']} | {number(row['full_protocol_image_AUROC'])} | "
            f"{number(row['mmr_compatible_image_AUROC'])} | {number(row['delta'], True)} |"
        )
    lines += [
        "", "## Signed raw-distance mechanism", "",
        "R_N and R_A are image-equal signed reductions `(d_official-d_union)/source_scale`; G=R_N-R_A. Defect regions use adaptive-max mapping to the patch grid.",
        "",
        "| method | scope | K | layer | R_N | R_A | G |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in mechanism_macro:
        lines.append(
            f"| {row['method']} | {row['scope']} | {row['K']} | {row['layer']} | "
            f"{number(row['R_N'])} | {number(row['R_A'])} | {number(row['G'])} |"
        )
    lines += [
        "", "## Timing", "",
        "| stage | seconds |", "|---|---:|",
    ]
    for row in timing:
        lines.append(f"| {row['stage']} | {float(row['seconds']):.3f} |")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    timing: list[dict[str, Any]] = []
    _seed_everything()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal AeBAD-S target few-shot requires CUDA")
    validate_official_metric_backend()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    install_import_paths(Path(args.project_root), Path(args.official_root), Path(args.dinov3_root))
    model, config = construct_exact_host(
        Path(args.official_root), Path(args.dinov3_root), Path(args.checkpoint), CHECKPOINT_SHA256,
    )

    stage = time.perf_counter()
    dataset_root, train_records, locators = discover_phase_a(args.aebad_root, formal=True)
    memory_fit, calibration = split_source_normal(train_records, threshold_fraction=8)
    if (len(train_records), len(memory_fit), len(calibration), len(locators)) != (521, 455, 66, 1639):
        raise RuntimeError("AeBAD-S frozen counts must be 521/455/66/1639")
    records = load_phase_b_records(dataset_root, locators, formal=True)
    target_pools, heldout = select_target_support(records, dataset_root=dataset_root)
    source_pool = select_source_extra(calibration, dataset_root=dataset_root)
    support_rows = support_manifest_rows(
        source_pool, target_pools, dataset_root=dataset_root,
    )
    _write_csv(output / "support_manifest.csv", support_rows)
    locator_by_path = {str(Path(locator.path).resolve()): locator for locator in locators}
    timing.append({"stage": "dataset_and_support_selection", "seconds": time.perf_counter() - stage})

    stage = time.perf_counter()
    memory_path = output / "official_memory.pt"
    resumed_memory = _load_memory(
        memory_path, model, source_count=len(train_records), device=device,
    )
    builder_metadata: dict[str, Any] = {"resumed": True}
    if not resumed_memory:
        build = build_exact_official_memory(
            model, train_records, device=device, scratch_dir=output / "memory_scratch",
        )
        builder_metadata = dict(build.metadata)
        _save_memory(memory_path, model, source_count=len(train_records))
        del build
    _atomic_json(output / "memory_manifest.json", {
        "protocol_id": PROTOCOL_ID,
        "builder_input_count": len(train_records),
        "memory_fit_count": len(memory_fit),
        "calibration_count": len(calibration),
        "threshold_fraction": 8,
        "official_brightness": [0.8, 1.2],
        "entries_per_layer": {
            str(layer): int(model.prototype_embeddings[layer].shape[0]) for layer in LAYERS
        },
        "shared_by_all_methods": True,
        "host_config": config,
        "builder_metadata": builder_metadata,
    })
    timing.append({"stage": "official_memory", "seconds": time.perf_counter() - stage})

    stage = time.perf_counter()
    mad = _load_mad(output / "source_mad_state.npz")
    if mad is None:
        mad = _fit_source_mad(
            calibration,
            dataset_root=dataset_root,
            model=model,
            device=device,
            output=output,
        )
    elif not (output / "calibration_stats.csv").is_file():
        raise RuntimeError("resumed source MAD lacks calibration_stats.csv")
    timing.append({"stage": "source_mad", "seconds": time.perf_counter() - stage})

    stage = time.perf_counter()
    support_root = output / "support_features"
    source_features = [
        extract_or_load_support(
            support_root / "source_extra" / f"{index:02d}.npz",
            record,
            dataset_root=dataset_root,
            model=model,
            device=device,
            protocol_id=PROTOCOL_ID,
        )
        for index, record in enumerate(source_pool, start=1)
    ]
    target_features: dict[str, list[SupportFeatures]] = {}
    for domain in DOMAINS:
        target_features[domain] = [
            extract_or_load_support(
                support_root / "target_matched" / domain / f"{index:02d}.npz",
                record,
                dataset_root=dataset_root,
                model=model,
                device=device,
                protocol_id=PROTOCOL_ID,
            )
            for index, record in enumerate(target_pools[domain], start=1)
        ]
    bank_rows: list[dict[str, Any]] = []
    for branch, domain, features in (
        [("source_extra", "source", source_features)]
        + [("target_matched", domain, target_features[domain]) for domain in DOMAINS]
    ):
        for shot in SHOT_COUNTS:
            for layer in LAYERS:
                bank_rows.append({
                    "branch": branch,
                    "domain": domain,
                    "K": shot,
                    "layer": layer,
                    "image_count": shot,
                    "patch_entries": sum(int(item.layers[layer].shape[0]) for item in features[:shot]),
                })
    _write_csv(output / "support_bank_stats.csv", bank_rows)
    timing.append({"stage": "support_feature_extraction", "seconds": time.perf_counter() - stage})

    stage = time.perf_counter()
    source_tensors = _support_tensors(source_features, device=device)
    raw_root = output / "raw_cache" / "heldout"
    for domain in DOMAINS:
        target_tensors = _support_tensors(target_features[domain], device=device)
        for index, record in enumerate(heldout[domain], start=1):
            locator = locator_by_path[str(Path(record.path).resolve())]
            path = raw_root / domain / f"{locator.sample_id}.npz"
            if path.is_file():
                _load_observation(path, relative=locator.relative_path)
            else:
                with Image.open(record.path) as source:
                    image = source.convert("RGB")
                    output_shape = (
                        image.height // int(model.evaluation_downscale),
                        image.width // int(model.evaluation_downscale),
                    )
                    observed = observe_fewshot_pil(
                        model,
                        image,
                        official_banks=model.prototype_embeddings,
                        source_support=source_tensors,
                        target_support=target_tensors,
                        device=device,
                    )
                _save_observation(
                    path,
                    relative=locator.relative_path,
                    output_shape=output_shape,
                    observed=observed,
                )
            print(f"[AEBAD-TARGET-FEWSHOT] raw {domain} {index}/{len(heldout[domain])}", flush=True)
        del target_tensors
        torch.cuda.empty_cache()
    _atomic_json(output / "prediction_barrier.json", {
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "prediction_count": sum(len(values) for values in heldout.values()),
        "heldout_normal_count": 458,
        "heldout_anomaly_count": 1149,
        "ground_truth_masks_opened": False,
    })
    timing.append({"stage": "heldout_exact_nn", "seconds": time.perf_counter() - stage})

    stage = time.perf_counter()
    domain_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    mmr_sample_rows: list[dict[str, Any]] = []
    mmr_domain_rows: list[dict[str, Any]] = []
    mechanism_per_image: list[dict[str, Any]] = []
    for domain in DOMAINS:
        domain_records = heldout[domain]
        loaded = []
        for record in domain_records:
            locator = locator_by_path[str(Path(record.path).resolve())]
            loaded.append(_load_observation(
                raw_root / domain / f"{locator.sample_id}.npz",
                relative=locator.relative_path,
            ))
        labels = [int(record.label) for record in domain_records]
        cropped_masks = [
            mmr_eval.resize_and_crop_mask(mmr_eval._load_original_mask(record))
            for record in domain_records
        ]
        compatible_labels = [
            mmr_eval.mmr_compatible_label(record.label, cropped)
            for record, cropped in zip(domain_records, cropped_masks, strict=True)
        ]
        for method in METHODS:
            maps: list[np.ndarray] = []
            cropped_scores: list[float] = []
            for record, (values, output_shape), cropped_mask in zip(
                domain_records, loaded, cropped_masks, strict=True,
            ):
                locator = locator_by_path[str(Path(record.path).resolve())]
                final_map = _method_map(values, method, mad, output_shape)
                maps.append(final_map)
                _cropped, crop_score = mmr_eval.cropped_image_score(final_map)
                cropped_scores.append(crop_score)
                full_score = float(final_map.max())
                sample_rows.append({
                    "sample_id": locator.sample_id,
                    "relative_path": locator.relative_path,
                    "domain": domain,
                    "label": int(record.label),
                    "method": method,
                    "image_score": full_score,
                })
                mmr_sample_rows.append({
                    "sample_id": locator.sample_id,
                    "relative_path": locator.relative_path,
                    "domain": domain,
                    "method": method,
                    "original_label": int(record.label),
                    "mmr_compatible_label": mmr_eval.mmr_compatible_label(record.label, cropped_mask),
                    "mask_empty_after_crop": bool(int(record.label) == 1 and not cropped_mask.any()),
                    "full_image_score": full_score,
                    "cropped_image_score": crop_score,
                })
            masks = [
                aebad_runner._load_mask(record, value.shape)
                for record, value in zip(domain_records, maps, strict=True)
            ]
            metrics = aebad_runner._evaluate(labels, masks, maps)
            domain_rows.append({
                "method": method,
                "domain": domain,
                "image_count": len(domain_records),
                "normal_count": sum(label == 0 for label in labels),
                "anomaly_count": sum(label == 1 for label in labels),
                **metrics,
            })
            compatible_auc = float(roc_auc_score(compatible_labels, cropped_scores))
            mmr_domain_rows.append({
                "method": method,
                "domain": domain,
                "image_count": len(domain_records),
                "original_anomaly_count": sum(labels),
                "compatible_anomaly_count": sum(compatible_labels),
                "relabelled_to_normal_count": sum(
                    original == 1 and compatible == 0
                    for original, compatible in zip(labels, compatible_labels, strict=True)
                ),
                "full_protocol_image_AUROC": metrics["image_AUROC"],
                "mmr_compatible_image_AUROC": compatible_auc,
                "delta": compatible_auc - metrics["image_AUROC"],
            })
            print(f"[AEBAD-TARGET-FEWSHOT] metrics {domain} {method} complete", flush=True)
        mechanism_per_image.extend(robust_runner._mechanism_per_image_rows(
            domain,
            domain_records,
            loaded,
            mad,
            category="AeBAD-S",
            pixel_evaluation=True,
            dataset_root=dataset_root,
        ))

    summary_rows = summarize_metrics(domain_rows)
    comparison_rows = _comparison_rows(domain_rows, summary_rows)
    k_change_rows = _k_change_rows(summary_rows)
    mmr_summary = mmr_eval.summarize_domain_metrics(mmr_domain_rows, methods=METHODS)
    mechanism_by_shift = robust_runner._mechanism_by_shift_layer(
        mechanism_per_image, pixel_evaluation=True,
    )
    mechanism_by_domain = [dict(row, domain=row["shift"]) for row in mechanism_by_shift]
    for row in mechanism_by_domain:
        row.pop("shift", None)
    four_macro = robust_runner._mechanism_macro(
        mechanism_by_shift, shifts=DOMAINS, pixel_evaluation=True,
    )
    shifted_rows = [row for row in mechanism_by_shift if row["shift"] != "same"]
    shifted_macro = robust_runner._mechanism_macro(
        shifted_rows,
        shifts=("background", "illumination", "view"),
        pixel_evaluation=True,
    )
    for row in four_macro:
        row["scope"] = "four_domain_macro"
    for row in shifted_macro:
        row["scope"] = "shifted_macro"
    mechanism_macro = [*four_macro, *shifted_macro]

    _write_csv(output / "sample_scores.csv", sample_rows)
    _write_csv(output / "domain_metrics.csv", domain_rows)
    _write_csv(output / "summary.csv", summary_rows)
    _write_csv(output / "mechanism_by_domain_layer.csv", mechanism_by_domain)
    _write_csv(output / "mechanism_macro.csv", mechanism_macro)
    _write_csv(output / "mmr_compatible_sample_scores.csv", mmr_sample_rows)
    _write_csv(output / "mmr_compatible_domain_metrics.csv", mmr_domain_rows)
    _write_csv(output / "mmr_compatible_summary.csv", mmr_summary)
    timing.append({"stage": "metrics_mmr_and_mechanism", "seconds": time.perf_counter() - stage})
    timing.append({"stage": "total", "seconds": time.perf_counter() - started})
    _write_csv(output / "timing.csv", timing)
    (output / "objective_report.md").write_text(
        _report(
            support_rows=support_rows,
            domain_rows=domain_rows,
            summary_rows=summary_rows,
            comparison_rows=comparison_rows,
            k_change_rows=k_change_rows,
            mechanism_macro=mechanism_macro,
            mmr_domain_rows=mmr_domain_rows,
            mmr_summary=mmr_summary,
            timing=timing,
        ),
        encoding="utf-8",
    )
    _atomic_json(output / "complete.json", {
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "run_type": "formal",
        "dataset": "AeBAD-S",
        "seed": SEED,
        "source_count": len(train_records),
        "memory_fit_count": len(memory_fit),
        "calibration_count": len(calibration),
        "target_count": len(records),
        "heldout_count": sum(len(values) for values in heldout.values()),
        "heldout_normal_count": 458,
        "heldout_anomaly_count": 1149,
        "support_pool_size_per_domain": 8,
        "source_extra_pool_size": 8,
        "K": list(SHOT_COUNTS),
        "methods": list(METHODS),
        "shared_official_memory": True,
        "shared_source_mad": True,
        "aupro_backend": "ADEval CUDA nstrips=200",
        "mmr_compatible_scope": "output-side post-evaluation",
    })
    print(f"AEBAD_S_TARGET_FEWSHOT_COMPLETE={output}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aebad-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--dinov3-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
