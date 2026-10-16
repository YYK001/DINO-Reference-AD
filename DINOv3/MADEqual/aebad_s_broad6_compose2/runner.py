from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image

from DINOv3.MADEqual.aebad_s_full_p0 import runner as baseline_runner
from DINOv3.MADEqual.aebad_s_full_p0.dataset import (
    DOMAINS,
    discover_phase_a,
    load_phase_b_records,
)
from DINOv3.relation_oracle_cache.extract import mad_equal
from DINOv3.relation_reliability.calibration import MADCalibrator, fit_calibrators
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

from .augmentations import SEED, compose2_manifest_row
from .memory import PROTOCOL_ID, build_broad6_memory, load_memory, save_memory


BASELINE_PROTOCOL_ID = "mad_equal_aebad_s_full_p0_seed42_v1"
CHECKPOINT_SHA256 = "7c1da9a54b3bdb333f5ebc42e404b7f19b1b5bed504877623c9dc87397f41488"
METHOD = "BROAD6_COMPOSE2_MAD_EQUAL"
METRIC_NAMES = baseline_runner.METRIC_NAMES
CACHE_REVISION = "broad6_raw_v1"


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    baseline_runner._write_csv(path, rows)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _numeric_metrics(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for source in rows:
        row = dict(source)
        for metric in METRIC_NAMES:
            row[metric] = float(row[metric])
        if "image_count" in row:
            row["image_count"] = int(row["image_count"])
        output.append(row)
    return output


def load_baseline(path: str | Path) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]]]:
    root = Path(path).resolve()
    complete_path = root / "complete.json"
    if not complete_path.is_file():
        raise FileNotFoundError(f"completed AeBAD-S baseline is required: {complete_path}")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    expected = {
        "status": "complete",
        "protocol_id": BASELINE_PROTOCOL_ID,
        "train_count": 521,
        "memory_count": 455,
        "calibration_count": 66,
        "prediction_count": 1639,
    }
    for key, value in expected.items():
        if complete.get(key) != value:
            raise RuntimeError(f"baseline {key} is {complete.get(key)!r}, expected {value!r}")
    if not {"SUPERADD_OFFICIAL", "MAD_EQUAL"}.issubset(set(complete.get("methods", []))):
        raise RuntimeError("baseline does not contain Official and MAD-Equal")
    for required in ("calibration_stats.csv", "group_metrics.csv", "summary.csv"):
        if not (root / required).is_file():
            raise FileNotFoundError(root / required)
    if len(list((root / "raw_cache" / "calibration").glob("*.npz"))) != 66:
        raise RuntimeError("baseline calibration raw cache is incomplete")
    if len(list((root / "raw_cache" / "test").glob("*.npz"))) != 1639:
        raise RuntimeError("baseline test raw cache is incomplete")
    return (
        root,
        _numeric_metrics(_read_csv(root / "group_metrics.csv")),
        _numeric_metrics(_read_csv(root / "summary.csv")),
    )


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
                raise RuntimeError(f"incompatible Broad6 raw cache: {path}")
            if str(payload["cache_revision"].item()) != CACHE_REVISION:
                raise RuntimeError(f"stale Broad6 raw cache: {path}")
            if str(payload["relative_path"].item()) != relative_path:
                raise RuntimeError(f"Broad6 raw cache path mismatch: {path}")
            raw = np.asarray(payload["raw_distances"], dtype=np.float32)
            output_shape = tuple(int(x) for x in payload["output_shape"].tolist())
        with Image.open(image_path) as image:
            expected_output = (
                image.height // int(model.evaluation_downscale),
                image.width // int(model.evaluation_downscale),
            )
            spatial = preprocessed_spatial_shape(model, image.height, image.width)
        patch = int(model.patch_exec.model_patch_size)
        expected_grid = (spatial[0] // patch, spatial[1] // patch)
        if (
            raw.shape != (len(LAYERS), *expected_grid)
            or output_shape != expected_output
            or not np.isfinite(raw).all()
        ):
            raise RuntimeError(f"invalid Broad6 raw cache: {path}")
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
        raise RuntimeError(f"invalid Broad6 observation: {image_path}")
    baseline_runner._atomic_npz(
        path,
        protocol_id=np.asarray(PROTOCOL_ID),
        cache_revision=np.asarray(CACHE_REVISION),
        relative_path=np.asarray(relative_path),
        raw_distances=raw,
        output_shape=np.asarray(output_shape, dtype=np.int32),
    )
    return raw, output_shape


def _fit_mad(samples: dict[int, np.ndarray]) -> MADCalibrator:
    _ecdf, mad = fit_calibrators(samples)
    return mad


def _score_raw(raw: np.ndarray, mad: MADCalibrator) -> np.ndarray:
    score = mad_equal(np.asarray(raw, dtype=np.float32), mad.medians, mad.scales)
    if score.shape != raw.shape[1:] or not np.isfinite(score).all():
        raise RuntimeError("invalid Broad6 MAD-Equal patch map")
    return score


def summarize_new_method(domain_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_domain = {str(row["group"]): row for row in domain_rows}
    if set(by_domain) != set(DOMAINS):
        raise RuntimeError("Broad6 domain metrics are incomplete")
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


def _add_comparison_deltas(
    rows: Sequence[dict[str, Any]], *, key: str,
) -> None:
    groups = {str(row[key]) for row in rows}
    for group in groups:
        subset = [row for row in rows if str(row[key]) == group]
        official = next(row for row in subset if row["method"] == "SUPERADD_OFFICIAL")
        mad = next(row for row in subset if row["method"] == "MAD_EQUAL")
        for row in subset:
            for metric in METRIC_NAMES:
                row[f"delta_vs_baseline_official_{metric}"] = float(row[metric] - official[metric])
                row[f"delta_vs_baseline_mad_equal_{metric}"] = float(row[metric] - mad[metric])


def _report(group_rows: Sequence[dict[str, Any]], summary_rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# AeBAD-S Exact SuperADD Broad6 Compose2 + MAD-Equal",
        "",
        f"Protocol: `{PROTOCOL_ID}`",
        "",
        "The existing Official + MAD-Equal baseline is read from `--baseline-output`; it is not rerun. Broad6 replaces Official brightness during construction of a new 100K-per-layer memory. Calibration and target images are unaugmented.",
        "",
        "AUPRO backend: project ADEval CUDA, `nstrips=200`.",
        "",
        "## Domain results",
        "",
        "| method | domain | Image AUROC | Image AUPR | Pixel AUROC | Pixel AUPR | AUPRO | d vs baseline MAD Image AUROC | d vs baseline MAD AUPRO |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in group_rows:
        lines.append(
            f"| {row['method']} | {row['group']} | {row['image_AUROC']:.6f} | "
            f"{row['image_AUPR']:.6f} | {row['pixel_AUROC']:.6f} | "
            f"{row['pixel_AUPR']:.6f} | {row['AUPRO']:.6f} | "
            f"{row['delta_vs_baseline_mad_equal_image_AUROC']:+.6f} | "
            f"{row['delta_vs_baseline_mad_equal_AUPRO']:+.6f} |"
        )
    lines += [
        "", "## Macro results", "",
        "| method | scope | Image AUROC | Image AUPR | Pixel AUROC | Pixel AUPR | AUPRO | d vs baseline MAD Image AUROC | d vs baseline MAD AUPRO |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['method']} | {row['scope']} | {row['image_AUROC']:.6f} | "
            f"{row['image_AUPR']:.6f} | {row['pixel_AUROC']:.6f} | "
            f"{row['pixel_AUPR']:.6f} | {row['AUPRO']:.6f} | "
            f"{row['delta_vs_baseline_mad_equal_image_AUROC']:+.6f} | "
            f"{row['delta_vs_baseline_mad_equal_AUPRO']:+.6f} |"
        )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal Broad6 Compose2 P0 requires CUDA")
    validate_official_metric_backend()
    baseline_root, baseline_groups, baseline_summary = load_baseline(args.baseline_output)

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset_root, train_records, locators = discover_phase_a(args.aebad_root, formal=True)
    memory_records, calibration_records = split_source_normal(train_records, threshold_fraction=8)
    if (len(memory_records), len(calibration_records)) != (455, 66):
        raise RuntimeError("AeBAD-S split must be 455 memory-fit and 66 calibration")

    install_import_paths(Path(args.project_root), Path(args.official_root), Path(args.dinov3_root))
    model, config = construct_exact_host(
        Path(args.official_root), Path(args.dinov3_root), Path(args.checkpoint), CHECKPOINT_SHA256
    )
    memory_path = output / "broad6_memory.pt"
    augmentation_rows = [
        compose2_manifest_row(Path(record.path).resolve().relative_to(dataset_root).as_posix())
        for index, record in enumerate(train_records)
        if index % 8 != 0
    ]
    metadata: dict[str, Any] = {"resumed": True}
    if not load_memory(memory_path, model, source_count=len(train_records), device=device):
        banks, built_rows, metadata = build_broad6_memory(
            model,
            train_records,
            dataset_root=dataset_root,
            device=device,
            scratch_dir=output / "memory_scratch",
        )
        if built_rows != augmentation_rows:
            raise RuntimeError("Broad6 augmentation plan changed during memory construction")
        save_memory(memory_path, banks, source_count=len(train_records))
    _write_csv(output / "augmentation_manifest.csv", augmentation_rows)
    baseline_runner._atomic_json(output / "memory_manifest.json", {
        "protocol_id": PROTOCOL_ID,
        "builder_input_count": len(train_records),
        "memory_fit_count": len(memory_records),
        "calibration_count": len(calibration_records),
        "augmentation": "one C12_BroadCompose2 view per memory-fit image",
        "official_brightness_used": False,
        "official_brightness_rng_draws_discarded_for_sampler_parity": len(memory_records),
        "clean_aug_union": False,
        "entries_per_layer": {str(layer): int(model.prototype_embeddings[layer].shape[0]) for layer in LAYERS},
        "host_config": config,
        "builder_metadata": metadata,
    })

    calibration_values = {layer: [] for layer in LAYERS}
    for index, record in enumerate(calibration_records, 1):
        relative = Path(record.path).resolve().relative_to(dataset_root).as_posix()
        raw, _ = _observe_or_load(
            output / "raw_cache" / "calibration" / f"calibration_{index - 1:04d}.npz",
            relative_path=relative,
            image_path=record.path,
            model=model,
            device=device,
        )
        for offset, layer in enumerate(LAYERS):
            calibration_values[layer].append(raw[offset].reshape(-1))
        print(f"[BROAD6-COMPOSE2] calibration {index}/{len(calibration_records)}", flush=True)
    pooled = {layer: np.concatenate(calibration_values[layer]) for layer in LAYERS}
    mad = _fit_mad(pooled)
    _write_csv(output / "calibration_stats.csv", [
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
            relative_path=locator.relative_path,
            image_path=locator.path,
            model=model,
            device=device,
        )
        print(f"[BROAD6-COMPOSE2] test raw {index}/{len(locators)}", flush=True)
    baseline_runner._atomic_json(output / "prediction_barrier.json", {
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "prediction_count": len(locators),
        "ground_truth_parsed": False,
    })

    records = load_phase_b_records(dataset_root, locators, formal=True)
    _write_csv(output / "dataset_manifest.csv", [
        {**record.as_row(), "sample_id": locator.sample_id, "relative_path": locator.relative_path}
        for locator, record in zip(locators, records, strict=True)
    ])
    new_groups: list[dict[str, Any]] = []
    new_samples: list[dict[str, Any]] = []
    for domain in DOMAINS:
        indices = [index for index, record in enumerate(records) if record.domain == domain]
        labels = [records[index].label for index in indices]
        masks: list[np.ndarray] = []
        maps: list[np.ndarray] = []
        for position, index in enumerate(indices, 1):
            locator, record = locators[index], records[index]
            raw, output_shape = _observe_or_load(
                target_root / f"{locator.sample_id}.npz",
                relative_path=locator.relative_path,
                image_path=locator.path,
                model=model,
                device=device,
                allow_observe=False,
            )
            final_map = interpolate_score(_score_raw(raw, mad), output_shape).astype(np.float32)
            masks.append(baseline_runner._load_mask(record, output_shape))
            maps.append(final_map)
            new_samples.append({
                "sample_id": locator.sample_id,
                "relative_path": locator.relative_path,
                "domain": domain,
                "label": record.label,
                "method": METHOD,
                "image_score": float(final_map.max()),
            })
            print(f"[BROAD6-COMPOSE2] score {domain} {position}/{len(indices)}", flush=True)
        new_groups.append({
            "method": METHOD,
            "group": domain,
            "image_count": len(indices),
            **baseline_runner._evaluate(labels, masks, maps),
        })

    group_rows = [*baseline_groups, *new_groups]
    summary_rows = [*baseline_summary, *summarize_new_method(new_groups)]
    _add_comparison_deltas(group_rows, key="group")
    _add_comparison_deltas(summary_rows, key="scope")
    baseline_samples = (
        _read_csv(baseline_root / "sample_scores.csv")
        if (baseline_root / "sample_scores.csv").is_file()
        else []
    )
    _write_csv(output / "sample_scores.csv", [*baseline_samples, *new_samples])
    _write_csv(output / "group_metrics.csv", group_rows)
    _write_csv(output / "summary.csv", summary_rows)
    (output / "objective_report.md").write_text(
        _report(group_rows, summary_rows), encoding="utf-8"
    )
    baseline_runner._atomic_json(output / "complete.json", {
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "run_type": "formal",
        "train_count": len(train_records),
        "memory_count": len(memory_records),
        "calibration_count": len(calibration_records),
        "prediction_count": len(records),
        "methods": ["SUPERADD_OFFICIAL", "MAD_EQUAL", METHOD],
        "baseline_output": str(baseline_root),
        "aupro_backend": "ADEval CUDA nstrips=200",
    })
    print(f"BROAD6_COMPOSE2_AEBAD_S_COMPLETE={output}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AeBAD-S Exact SuperADD Broad6 Compose2 + MAD-Equal")
    parser.add_argument("--aebad-root", required=True)
    parser.add_argument("--baseline-output", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--dinov3-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
