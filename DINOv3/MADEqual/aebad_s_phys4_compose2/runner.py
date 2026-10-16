from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

from DINOv3.MADEqual.aebad_s_broad6_compose2 import runner as broad6_runner
from DINOv3.MADEqual.aebad_s_full_p0 import runner as baseline_runner
from DINOv3.MADEqual.aebad_s_full_p0.dataset import (
    DOMAINS,
    discover_phase_a,
    load_phase_b_records,
)
from DINOv3.relation_reliability.datasets import split_source_normal
from DINOv3.relation_reliability.official_metrics import validate_official_metric_backend
from DINOv3.relation_reliability.protocol import LAYERS
from DINOv3.relation_reliability.superadd_host import (
    construct_exact_host,
    install_import_paths,
    interpolate_score,
    observe_pil,
    preprocessed_spatial_shape,
)

from .augmentations import PHYS4, SEED, phys4_manifest_row
from .memory import PROTOCOL_ID, build_phys4_memory, load_memory, save_memory


CHECKPOINT_SHA256 = broad6_runner.CHECKPOINT_SHA256
METHOD = "PHYS4_COMPOSE2_MAD_EQUAL"
METRIC_NAMES = baseline_runner.METRIC_NAMES
CACHE_REVISION = "phys4_raw_v1"
BROAD6_PROTOCOL_ID = "mad_equal_aebad_s_broad6_compose2_seed42_v1"


def _load_comparisons(
    baseline_path: str | Path,
    broad6_path: str | Path,
) -> tuple[Path, Path, list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_root, baseline_groups, baseline_summary = broad6_runner.load_baseline(
        baseline_path
    )
    broad_root = Path(broad6_path).resolve()
    complete_path = broad_root / "complete.json"
    if not complete_path.is_file():
        raise FileNotFoundError(complete_path)
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if complete.get("status") != "complete" or complete.get("protocol_id") != BROAD6_PROTOCOL_ID:
        raise RuntimeError("completed Broad6 formal output is required")
    for required in ("group_metrics.csv", "summary.csv", "sample_scores.csv"):
        if not (broad_root / required).is_file():
            raise FileNotFoundError(broad_root / required)
    broad_groups = broad6_runner._numeric_metrics(
        broad6_runner._read_csv(broad_root / "group_metrics.csv")
    )
    broad_summary = broad6_runner._numeric_metrics(
        broad6_runner._read_csv(broad_root / "summary.csv")
    )
    groups = [*baseline_groups, *[
        row for row in broad_groups if row["method"] == "BROAD6_COMPOSE2_MAD_EQUAL"
    ]]
    summaries = [*baseline_summary, *[
        row for row in broad_summary if row["method"] == "BROAD6_COMPOSE2_MAD_EQUAL"
    ]]
    if len(groups) != 12 or len(summaries) != 6:
        raise RuntimeError("comparison rows are incomplete")
    return baseline_root, broad_root, groups, summaries


def _observe_or_load(
    path: Path,
    *,
    relative_path: str,
    image_path: str,
    model: Any,
    device: torch.device,
    allow_observe: bool = True,
) -> tuple[np.ndarray, tuple[int, int]]:
    if path.is_file():
        with np.load(path, allow_pickle=False) as payload:
            if str(payload["protocol_id"].item()) != PROTOCOL_ID:
                raise RuntimeError(f"incompatible Phys4 raw cache: {path}")
            if str(payload["cache_revision"].item()) != CACHE_REVISION:
                raise RuntimeError(f"stale Phys4 raw cache: {path}")
            if str(payload["relative_path"].item()) != relative_path:
                raise RuntimeError(f"Phys4 raw cache path mismatch: {path}")
            raw = np.asarray(payload["raw_distances"], dtype=np.float32)
            output_shape = tuple(int(value) for value in payload["output_shape"].tolist())
        with Image.open(image_path) as source:
            expected_output = (
                source.height // int(model.evaluation_downscale),
                source.width // int(model.evaluation_downscale),
            )
            spatial = preprocessed_spatial_shape(model, source.height, source.width)
        patch = int(model.patch_exec.model_patch_size)
        expected_grid = (spatial[0] // patch, spatial[1] // patch)
        if (
            raw.shape != (len(LAYERS), *expected_grid)
            or output_shape != expected_output
            or not np.isfinite(raw).all()
        ):
            raise RuntimeError(f"invalid Phys4 raw cache: {path}")
        return raw, output_shape
    if not allow_observe:
        raise RuntimeError(f"prediction barrier cache is missing: {path}")
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        output_shape = (
            image.height // int(model.evaluation_downscale),
            image.width // int(model.evaluation_downscale),
        )
        observation = observe_pil(model, image, device=device)
    raw = np.asarray(observation.raw_distances, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[0] != len(LAYERS) or not np.isfinite(raw).all():
        raise RuntimeError(f"invalid Phys4 observation: {image_path}")
    baseline_runner._atomic_npz(
        path,
        protocol_id=np.asarray(PROTOCOL_ID),
        cache_revision=np.asarray(CACHE_REVISION),
        relative_path=np.asarray(relative_path),
        raw_distances=raw,
        output_shape=np.asarray(output_shape, dtype=np.int32),
    )
    return raw, output_shape


def summarize_phys4(domain_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_domain = {str(row["group"]): row for row in domain_rows}
    if set(by_domain) != set(DOMAINS):
        raise RuntimeError("Phys4 domain metrics are incomplete")
    output = []
    for scope, domains in (
        ("four_domain_macro", DOMAINS),
        ("shifted_macro", ("background", "illumination", "view")),
    ):
        row: dict[str, Any] = {"method": METHOD, "scope": scope}
        for metric in METRIC_NAMES:
            row[metric] = float(np.mean([by_domain[domain][metric] for domain in domains]))
        output.append(row)
    return output


def _add_reference_deltas(rows: Sequence[dict[str, Any]], *, key: str) -> None:
    for group in {str(row[key]) for row in rows}:
        subset = [row for row in rows if str(row[key]) == group]
        references = {
            "baseline_official": next(
                row for row in subset if row["method"] == "SUPERADD_OFFICIAL"
            ),
            "baseline_mad_equal": next(
                row for row in subset if row["method"] == "MAD_EQUAL"
            ),
            "broad6": next(
                row for row in subset if row["method"] == "BROAD6_COMPOSE2_MAD_EQUAL"
            ),
        }
        for row in subset:
            for name, reference in references.items():
                for metric in METRIC_NAMES:
                    row[f"delta_vs_{name}_{metric}"] = float(row[metric] - reference[metric])


def _report(
    group_rows: Sequence[dict[str, Any]],
    summary_rows: Sequence[dict[str, Any]],
    augmentation_rows: Sequence[dict[str, Any]],
) -> str:
    operation_counts = Counter(
        operation
        for row in augmentation_rows
        for operation in (str(row["op1"]), str(row["op2"]))
    )
    recipe_counts = Counter(str(row["recipe"]) for row in augmentation_rows)
    lines = [
        "# AeBAD-S Exact SuperADD Phys4 Compose2 + MAD-Equal",
        "",
        f"Protocol: `{PROTOCOL_ID}`",
        "",
        "Phys4 is the fixed pool Brightness, Gamma, WhiteBalance, and IlluminationField. Every memory-fit image contributes one ordered pair of distinct float32 transforms. Existing Official, MAD-Equal, and Broad6 results are read from completed outputs and are not rerun.",
        "",
        "AUPRO backend: project ADEval CUDA, `nstrips=200`.",
        "",
        "## Augmentation composition",
        "",
        f"Memory-fit images: {len(augmentation_rows)}; one Compose2 view per image.",
        "",
        "| operation | occurrence count |",
        "|---|---:|",
    ]
    for operation in PHYS4:
        lines.append(f"| {operation} | {operation_counts[operation]} |")
    lines += [
        "",
        "| ordered recipe | image count |",
        "|---|---:|",
    ]
    for op1 in PHYS4:
        for op2 in PHYS4:
            if op1 != op2:
                recipe = f"{op1}>{op2}"
                lines.append(f"| {recipe} | {recipe_counts[recipe]} |")
    lines += [
        "",
        "## Domain results",
        "",
        "| method | domain | Image AUROC | Image AUPR | Pixel AUROC | Pixel AUPR | AUPRO |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in group_rows:
        lines.append(
            f"| {row['method']} | {row['group']} | {row['image_AUROC']:.6f} | "
            f"{row['image_AUPR']:.6f} | {row['pixel_AUROC']:.6f} | "
            f"{row['pixel_AUPR']:.6f} | {row['AUPRO']:.6f} |"
        )
    lines += [
        "", "## Phys4 domain deltas", "",
        "| domain | d vs MAD Image AUROC | d vs MAD AUPRO | d vs Broad6 Image AUROC | d vs Broad6 AUPRO |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in group_rows:
        if row["method"] != METHOD:
            continue
        lines.append(
            f"| {row['group']} | {row['delta_vs_baseline_mad_equal_image_AUROC']:+.6f} | "
            f"{row['delta_vs_baseline_mad_equal_AUPRO']:+.6f} | "
            f"{row['delta_vs_broad6_image_AUROC']:+.6f} | "
            f"{row['delta_vs_broad6_AUPRO']:+.6f} |"
        )
    lines += [
        "", "## Macro results", "",
        "| method | scope | Image AUROC | Image AUPR | Pixel AUROC | Pixel AUPR | AUPRO |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['method']} | {row['scope']} | {row['image_AUROC']:.6f} | "
            f"{row['image_AUPR']:.6f} | {row['pixel_AUROC']:.6f} | "
            f"{row['pixel_AUPR']:.6f} | {row['AUPRO']:.6f} |"
        )
    lines += [
        "", "No automatic PASS/FAIL or stopping decision is assigned by this report.", "",
    ]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal Phys4 Compose2 requires CUDA")
    validate_official_metric_backend()
    baseline_root, broad_root, comparison_groups, comparison_summary = _load_comparisons(
        args.baseline_output, args.broad6_output
    )

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset_root, train_records, locators = discover_phase_a(args.aebad_root, formal=True)
    memory_records, calibration_records = split_source_normal(train_records, threshold_fraction=8)
    if (len(memory_records), len(calibration_records)) != (455, 66):
        raise RuntimeError("AeBAD-S split must be 455/66")

    install_import_paths(Path(args.project_root), Path(args.official_root), Path(args.dinov3_root))
    model, config = construct_exact_host(
        Path(args.official_root), Path(args.dinov3_root), Path(args.checkpoint), CHECKPOINT_SHA256
    )
    memory_path = output / "phys4_memory.pt"
    planned_manifest = [
        phys4_manifest_row(Path(record.path).resolve().relative_to(dataset_root).as_posix())
        for index, record in enumerate(train_records) if index % 8 != 0
    ]
    metadata: dict[str, Any] = {"resumed": True}
    if not load_memory(memory_path, model, source_count=len(train_records), device=device):
        banks, built_manifest, metadata = build_phys4_memory(
            model, train_records, dataset_root=dataset_root, device=device,
            scratch_dir=output / "memory_scratch",
        )
        if built_manifest != planned_manifest:
            raise RuntimeError("Phys4 augmentation plan changed during memory construction")
        save_memory(memory_path, banks, source_count=len(train_records))
    broad6_runner._write_csv(output / "augmentation_manifest.csv", planned_manifest)
    baseline_runner._atomic_json(output / "memory_manifest.json", {
        "protocol_id": PROTOCOL_ID,
        "builder_input_count": len(train_records),
        "memory_fit_count": len(memory_records),
        "calibration_count": len(calibration_records),
        "augmentation_pool": list(PHYS4),
        "augmentation": "one ordered Phys4 Compose2 float32 view per memory-fit image",
        "official_brightness_used": False,
        "official_brightness_rng_draws_discarded_for_sampler_parity": len(memory_records),
        "clean_aug_union": False,
        "entries_per_layer": {
            str(layer): int(model.prototype_embeddings[layer].shape[0]) for layer in LAYERS
        },
        "host_config": config,
        "builder_metadata": metadata,
    })

    calibration_values = {layer: [] for layer in LAYERS}
    for index, record in enumerate(calibration_records, 1):
        relative = Path(record.path).resolve().relative_to(dataset_root).as_posix()
        raw, _ = _observe_or_load(
            output / "raw_cache" / "calibration" / f"calibration_{index - 1:04d}.npz",
            relative_path=relative, image_path=record.path, model=model, device=device,
        )
        for offset, layer in enumerate(LAYERS):
            calibration_values[layer].append(raw[offset].reshape(-1))
        print(f"[PHYS4-COMPOSE2] calibration {index}/{len(calibration_records)}", flush=True)
    pooled = {layer: np.concatenate(calibration_values[layer]) for layer in LAYERS}
    mad = broad6_runner._fit_mad(pooled)
    broad6_runner._write_csv(output / "calibration_stats.csv", [
        {
            "layer": layer,
            "patch_count": int(pooled[layer].size),
            "median": float(mad.medians[offset]),
            "mad_scale_1p4826": float(mad.scales[offset]),
        }
        for offset, layer in enumerate(LAYERS)
    ])

    target_root = output / "raw_cache" / "test"
    for index, locator in enumerate(locators, 1):
        _observe_or_load(
            target_root / f"{locator.sample_id}.npz",
            relative_path=locator.relative_path, image_path=locator.path,
            model=model, device=device,
        )
        print(f"[PHYS4-COMPOSE2] test raw {index}/{len(locators)}", flush=True)
    baseline_runner._atomic_json(output / "prediction_barrier.json", {
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "prediction_count": len(locators),
        "ground_truth_parsed": False,
    })

    records = load_phase_b_records(dataset_root, locators, formal=True)
    broad6_runner._write_csv(output / "dataset_manifest.csv", [
        {
            **record.as_row(),
            "sample_id": locator.sample_id,
            "relative_path": locator.relative_path,
        }
        for locator, record in zip(locators, records, strict=True)
    ])
    phys_groups: list[dict[str, Any]] = []
    phys_samples: list[dict[str, Any]] = []
    for domain in DOMAINS:
        indices = [index for index, record in enumerate(records) if record.domain == domain]
        labels = [records[index].label for index in indices]
        masks: list[np.ndarray] = []
        maps: list[np.ndarray] = []
        for position, index in enumerate(indices, 1):
            locator, record = locators[index], records[index]
            raw, output_shape = _observe_or_load(
                target_root / f"{locator.sample_id}.npz",
                relative_path=locator.relative_path, image_path=locator.path,
                model=model, device=device, allow_observe=False,
            )
            score = broad6_runner._score_raw(raw, mad)
            final_map = interpolate_score(score, output_shape).astype(np.float32)
            masks.append(baseline_runner._load_mask(record, output_shape))
            maps.append(final_map)
            phys_samples.append({
                "sample_id": locator.sample_id, "relative_path": locator.relative_path,
                "domain": domain, "label": record.label, "method": METHOD,
                "image_score": float(final_map.max()),
            })
            print(f"[PHYS4-COMPOSE2] score {domain} {position}/{len(indices)}", flush=True)
        phys_groups.append({
            "method": METHOD, "group": domain, "image_count": len(indices),
            **baseline_runner._evaluate(labels, masks, maps),
        })

    group_rows = [*comparison_groups, *phys_groups]
    summary_rows = [*comparison_summary, *summarize_phys4(phys_groups)]
    _add_reference_deltas(group_rows, key="group")
    _add_reference_deltas(summary_rows, key="scope")
    existing_samples = broad6_runner._read_csv(broad_root / "sample_scores.csv")
    broad6_runner._write_csv(output / "sample_scores.csv", [*existing_samples, *phys_samples])
    broad6_runner._write_csv(output / "group_metrics.csv", group_rows)
    broad6_runner._write_csv(output / "summary.csv", summary_rows)
    report = _report(group_rows, summary_rows, planned_manifest)
    (output / "objective_report.md").write_text(report, encoding="utf-8")
    baseline_runner._atomic_json(output / "complete.json", {
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "run_type": "formal",
        "train_count": len(train_records),
        "memory_count": len(memory_records),
        "calibration_count": len(calibration_records),
        "prediction_count": len(records),
        "methods": [
            "SUPERADD_OFFICIAL", "MAD_EQUAL", "BROAD6_COMPOSE2_MAD_EQUAL", METHOD,
        ],
        "baseline_output": str(baseline_root),
        "broad6_output": str(broad_root),
        "aupro_backend": "ADEval CUDA nstrips=200",
    })
    print(f"PHYS4_COMPOSE2_AEBAD_S_COMPLETE={output}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AeBAD-S Phys4 Compose2 + MAD-Equal")
    parser.add_argument("--aebad-root", required=True)
    parser.add_argument("--baseline-output", required=True)
    parser.add_argument("--broad6-output", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--dinov3-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
