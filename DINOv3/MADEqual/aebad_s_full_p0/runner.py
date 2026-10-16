from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score

from DINOv3.relation_oracle_cache.extract import mad_equal
from DINOv3.relation_reliability.calibration import (
    MADCalibrator,
    fit_calibrators,
)
from DINOv3.relation_reliability.datasets import Record, split_source_normal
from DINOv3.relation_reliability.official_metrics import (
    official_aupro_adeval,
    validate_official_metric_backend,
)
from DINOv3.relation_reliability.protocol import LAYERS
from DINOv3.relation_reliability.superadd_host import (
    build_exact_official_memory,
    construct_exact_host,
    install_import_paths,
    interpolate_score,
    observe_pil,
    preprocessed_spatial_shape,
)

from .dataset import DOMAINS, TestLocator, discover_phase_a, load_phase_b_records


PROTOCOL_ID = "mad_equal_aebad_s_full_p0_seed42_v1"
CACHE_REVISION = "raw_distance_cache_v2_geometry_bound"
CHECKPOINT_SHA256 = "7c1da9a54b3bdb333f5ebc42e404b7f19b1b5bed504877623c9dc87397f41488"
METHODS = (
    "SUPERADD_OFFICIAL",
    "MAD_EQUAL",
)
METRIC_NAMES = (
    "image_AUROC",
    "image_AUPR",
    "pixel_AUROC",
    "pixel_AUPR",
    "AUPRO",
)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
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


def _atomic_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **values)
    os.replace(temporary, path)


def _save_memory(path: Path, model: Any, *, source_count: int) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    state: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "source_count": int(source_count),
    }
    for layer in LAYERS:
        state[f"bank_{layer}"] = model.prototype_embeddings[layer].detach().cpu()
    torch.save(state, temporary)
    os.replace(temporary, path)


def _load_memory(path: Path, model: Any, *, source_count: int, device: torch.device) -> bool:
    if not path.is_file():
        return False
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("protocol_id") != PROTOCOL_ID or int(state.get("source_count", -1)) != source_count:
        raise RuntimeError(f"incompatible Official memory cache: {path}")
    banks = {}
    for layer in LAYERS:
        value = state.get(f"bank_{layer}")
        if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[0] > 100_000:
            raise RuntimeError(f"invalid Official memory cache layer {layer}: {path}")
        banks[layer] = value.to(device=device, dtype=torch.float32)
    model.prototype_embeddings = banks
    return True


def _clone_numpy_state(state: tuple[Any, ...]) -> tuple[Any, ...]:
    return (state[0], np.asarray(state[1]).copy(), state[2], state[3], state[4])


class _IsolatedNumpyPreprocessing:
    """Replay Official brightness draws without advancing the sampler RNG."""

    def __init__(self, delegate: Any, state: tuple[Any, ...]) -> None:
        self.delegate = delegate
        self.state = _clone_numpy_state(state)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        sampler_state = _clone_numpy_state(np.random.get_state())
        np.random.set_state(self.state)
        try:
            result = self.delegate(value)
            self.state = _clone_numpy_state(np.random.get_state())
        finally:
            np.random.set_state(sampler_state)
        return result


def _post_preprocessing_rng_state(
    initial_state: tuple[Any, ...], preprocessing: Any, image_count: int,
) -> tuple[Any, ...]:
    generator = np.random.RandomState()
    generator.set_state(initial_state)
    for _ in range(int(image_count)):
        generator.uniform(
            preprocessing.min_brightness_factor,
            preprocessing.max_brightness_factor,
        )
    return _clone_numpy_state(generator.get_state())


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _build_exact_memory_layerwise(
    model: Any,
    train_records: Sequence[Record],
    *,
    device: torch.device,
    output: Path,
) -> None:
    """Build independent Official layer banks while bounding candidate storage."""
    original_model_layers = model.layers
    original_backbone_layers = model.backbone.layers
    original_preprocessing = model.augmented_preprocessing
    if tuple(int(value) for value in original_model_layers) != tuple(LAYERS):
        raise RuntimeError("Exact layerwise builder requires the frozen four-layer host")

    prototype_count = sum(
        index % int(model.threshold_fraction) != 0
        for index in range(len(train_records))
    )
    initial_brightness_state = _clone_numpy_state(np.random.get_state())
    sampler_state = _post_preprocessing_rng_state(
        initial_brightness_state, original_preprocessing, prototype_count,
    )
    checkpoint_root = output / "memory_layers"
    banks: dict[int, torch.Tensor] = {}
    try:
        for layer in LAYERS:
            checkpoint = checkpoint_root / f"layer_{layer}.pt"
            if checkpoint.is_file():
                state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                if (
                    state.get("protocol_id") != PROTOCOL_ID or
                    int(state.get("source_count", -1)) != len(train_records) or
                    int(state.get("layer", -1)) != layer
                ):
                    raise RuntimeError(f"incompatible layer memory checkpoint: {checkpoint}")
                bank = state.get("bank")
                if not isinstance(bank, torch.Tensor) or bank.ndim != 2 or bank.shape[0] > 100_000:
                    raise RuntimeError(f"invalid layer memory checkpoint: {checkpoint}")
                banks[layer] = bank.to(device=device, dtype=torch.float32)
                sampler_state = _clone_numpy_state(state["numpy_rng_after_sampler"])
                print(f"[MAD-EQUAL-AEBAD-S-P0] resume Official memory L{layer}", flush=True)
                continue

            np.random.set_state(sampler_state)
            model.layers = [int(layer)]
            model.backbone.layers = [int(layer)]
            model.augmented_preprocessing = _IsolatedNumpyPreprocessing(
                original_preprocessing, initial_brightness_state,
            )
            build = build_exact_official_memory(
                model, train_records, device=device,
                scratch_dir=output / "memory_scratch" / f"layer_{layer}",
            )
            bank = build.banks[layer]
            sampler_state = _clone_numpy_state(np.random.get_state())
            _atomic_torch_save(checkpoint, {
                "protocol_id": PROTOCOL_ID,
                "source_count": len(train_records),
                "layer": int(layer),
                "bank": bank.detach().cpu(),
                "numpy_rng_after_sampler": sampler_state,
            })
            banks[layer] = bank
            del build
            gc.collect()
            print(f"[MAD-EQUAL-AEBAD-S-P0] Official memory L{layer} complete", flush=True)
    finally:
        model.layers = original_model_layers
        model.backbone.layers = original_backbone_layers
        model.augmented_preprocessing = original_preprocessing
    np.random.set_state(sampler_state)
    model.prototype_embeddings = banks


def _observe_or_load(
    cache_path: Path,
    *,
    relative_path: str,
    dataset_root: str,
    image_path: str,
    model: Any,
    device: torch.device,
    allow_observe: bool = True,
) -> tuple[np.ndarray, tuple[int, int]]:
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as payload:
            if str(payload["protocol_id"].item()) != PROTOCOL_ID or str(payload["cache_revision"].item()) != CACHE_REVISION:
                raise RuntimeError(f"incompatible raw-distance cache: {cache_path}")
            if str(payload["relative_path"].item()) != relative_path:
                raise RuntimeError(f"raw-distance cache path mismatch: {cache_path}")
            if str(payload["dataset_root"].item()) != dataset_root:
                raise RuntimeError(f"raw-distance cache dataset root mismatch: {cache_path}")
            raw = np.asarray(payload["raw_distances"], dtype=np.float32)
            output_shape = tuple(int(value) for value in payload["output_shape"].tolist())
        with Image.open(image_path) as source:
            expected_output = (
                source.height // int(model.evaluation_downscale),
                source.width // int(model.evaluation_downscale),
            )
            preprocessed = preprocessed_spatial_shape(model, source.height, source.width)
        expected_grid = tuple(
            int(value) // int(model.patch_exec.model_patch_size) for value in preprocessed
        )
        if (
            raw.ndim != 3 or raw.shape[0] != len(LAYERS) or
            tuple(raw.shape[1:]) != expected_grid or
            len(output_shape) != 2 or output_shape != expected_output or
            not np.isfinite(raw).all()
        ):
            raise RuntimeError(f"invalid raw-distance cache shape: {cache_path}")
        return raw, output_shape

    if not allow_observe:
        raise RuntimeError(f"prediction barrier cache is missing: {cache_path}")

    with Image.open(image_path) as source:
        image = source.convert("RGB")
        downscale = int(model.evaluation_downscale)
        output_shape = (image.height // downscale, image.width // downscale)
        observation = observe_pil(model, image, device=device)
    raw = np.asarray(observation.raw_distances, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[0] != len(LAYERS) or not np.isfinite(raw).all():
        raise RuntimeError(f"invalid raw-distance observation for {image_path}")
    _atomic_npz(
        cache_path,
        protocol_id=np.asarray(PROTOCOL_ID),
        cache_revision=np.asarray(CACHE_REVISION),
        dataset_root=np.asarray(dataset_root),
        relative_path=np.asarray(relative_path),
        raw_distances=raw,
        output_shape=np.asarray(output_shape, dtype=np.int32),
    )
    return raw, output_shape


def _score_raw(
    raw: np.ndarray,
    *,
    mad: MADCalibrator,
) -> dict[str, np.ndarray]:
    value = np.asarray(raw, dtype=np.float32)
    scores = {
        "SUPERADD_OFFICIAL": value.mean(axis=0),
        "MAD_EQUAL": mad_equal(value, mad.medians, mad.scales),
    }
    for name, score in scores.items():
        if score.shape != value.shape[1:] or not np.isfinite(score).all():
            raise RuntimeError(f"invalid {name} patch map")
    return scores


def _fit_source_calibrators(
    pooled: dict[int, np.ndarray],
) -> MADCalibrator:
    _ecdf, mad = fit_calibrators(pooled)
    return mad


def _load_mask(record: Record, output_shape: tuple[int, int]) -> np.ndarray:
    if record.label == 0:
        return np.zeros(output_shape, dtype=np.uint8)
    if not record.mask_path:
        raise RuntimeError(f"anomaly record lacks mask: {record.path}")
    with Image.open(record.mask_path) as source:
        mask = source.convert("L").resize(
            (output_shape[1], output_shape[0]), Image.Resampling.NEAREST,
        )
        return (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)


def _evaluate(labels: Sequence[int], masks: Sequence[np.ndarray], maps: Sequence[np.ndarray]) -> dict[str, float]:
    image_scores = np.asarray([float(value.max()) for value in maps], dtype=np.float64)
    image_labels = np.asarray(labels, dtype=np.uint8)
    pixel_labels = np.concatenate([value.reshape(-1) for value in masks])
    pixel_scores = np.concatenate([value.reshape(-1) for value in maps])
    return {
        "image_AUROC": float(roc_auc_score(image_labels, image_scores)),
        "image_AUPR": float(average_precision_score(image_labels, image_scores)),
        "pixel_AUROC": float(roc_auc_score(pixel_labels, pixel_scores)),
        "pixel_AUPR": float(average_precision_score(pixel_labels, pixel_scores)),
        "AUPRO": official_aupro_adeval(masks, maps, nstrips=200),
    }


def summarize_group_metrics(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in METHODS:
        method_rows = {str(row["group"]): row for row in rows if row["method"] == method}
        if set(method_rows) != set(DOMAINS):
            raise RuntimeError(f"incomplete domain metrics for {method}")
        for scope, domains in (
            ("four_domain_macro", DOMAINS),
            ("shifted_macro", ("background", "illumination", "view")),
        ):
            result: dict[str, Any] = {"method": method, "scope": scope}
            for metric in METRIC_NAMES:
                result[metric] = float(np.mean([method_rows[domain][metric] for domain in domains]))
            output.append(result)
    for scope in ("four_domain_macro", "shifted_macro"):
        official = next(
            row for row in output
            if row["method"] == "SUPERADD_OFFICIAL" and row["scope"] == scope
        )
        for row in output:
            if row["scope"] != scope:
                continue
            for metric in METRIC_NAMES:
                row[f"delta_vs_official_{metric}"] = float(row[metric] - official[metric])
    return output


def _add_domain_deltas(rows: Sequence[dict[str, Any]]) -> None:
    for domain in DOMAINS:
        official = next(
            row for row in rows
            if row["method"] == "SUPERADD_OFFICIAL" and row["group"] == domain
        )
        for row in rows:
            if row["group"] != domain:
                continue
            for metric in METRIC_NAMES:
                row[f"delta_vs_official_{metric}"] = float(row[metric] - official[metric])


def _objective_report(group_rows: Sequence[dict[str, Any]], summary_rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# AeBAD-S Exact SuperADD + MAD-Equal Full P0",
        "",
        f"Protocol: `{PROTOCOL_ID}`",
        "",
        "Both primary methods use the same Exact Official memory and the same per-image four-layer raw 1-NN tensor. MAD-Equal calls the existing project implementation; it is not reimplemented here.",
        "",
        "AUPRO backend: project ADEval CUDA, `nstrips=200`.",
        "",
        "## Domain results",
        "",
        "| method | domain | Image AUROC | Image AUPR | Pixel AUROC | Pixel AUPR | AUPRO | d Image AUROC | d AUPRO |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in group_rows:
        lines.append(
            f"| {row['method']} | {row['group']} | {row['image_AUROC']:.6f} | "
            f"{row['image_AUPR']:.6f} | {row['pixel_AUROC']:.6f} | "
            f"{row['pixel_AUPR']:.6f} | {row['AUPRO']:.6f} | "
            f"{row['delta_vs_official_image_AUROC']:+.6f} | "
            f"{row['delta_vs_official_AUPRO']:+.6f} |"
        )
    lines.extend((
        "",
        "## Macro results",
        "",
        "| method | scope | Image AUROC | Image AUPR | Pixel AUROC | Pixel AUPR | AUPRO |",
        "|---|---|---:|---:|---:|---:|---:|",
    ))
    for row in summary_rows:
        lines.append(
            f"| {row['method']} | {row['scope']} | {row['image_AUROC']:.6f} | "
            f"{row['image_AUPR']:.6f} | {row['pixel_AUROC']:.6f} | "
            f"{row['pixel_AUPR']:.6f} | {row['AUPRO']:.6f} |"
        )
    mad_shifted = next(
        row for row in summary_rows
        if row["method"] == "MAD_EQUAL" and row["scope"] == "shifted_macro"
    )
    image_delta = mad_shifted["delta_vs_official_image_AUROC"]
    aupro_delta = mad_shifted["delta_vs_official_AUPRO"]
    if image_delta > 0 and aupro_delta > 0:
        conclusion = "supports transfer on both frozen primary metrics"
    elif image_delta <= 0 and aupro_delta <= 0:
        conclusion = "does not support transfer on either frozen primary metric"
    else:
        conclusion = "is mixed across the two frozen primary metrics"
    lines.extend((
        "",
        "## Frozen objective",
        "",
        f"MAD-Equal minus Official on shifted macro: Image AUROC {image_delta:+.6f}; AUPRO {aupro_delta:+.6f}.",
        f"Under the frozen two-primary-metric rule, the result {conclusion}.",
    ))
    lines.extend((
        "",
        "The shifted macro is the unweighted mean of background, illumination, and view; same is excluded. Published MMR/MSFR/MMR+++ values are not treated as direct same-implementation comparisons.",
        "",
    ))
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal Full P0 requires a CUDA device")
    validate_official_metric_backend()

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset_root, train_records, locators = discover_phase_a(args.aebad_root, formal=True)
    memory_records, calibration_records = split_source_normal(train_records, threshold_fraction=8)
    if (len(memory_records), len(calibration_records)) != (455, 66):
        raise RuntimeError("AeBAD-S source split must be memory=455/calibration=66")

    install_import_paths(Path(args.project_root), Path(args.official_root), Path(args.dinov3_root))
    model, config = construct_exact_host(
        Path(args.official_root), Path(args.dinov3_root), Path(args.checkpoint), CHECKPOINT_SHA256,
    )
    memory_path = output / "official_memory.pt"
    if not _load_memory(memory_path, model, source_count=len(train_records), device=device):
        _build_exact_memory_layerwise(
            model, train_records, device=device, output=output,
        )
        _save_memory(memory_path, model, source_count=len(train_records))
        for layer in LAYERS:
            (output / "memory_layers" / f"layer_{layer}.pt").unlink(missing_ok=True)
        (output / "memory_layers").rmdir()
    _atomic_json(output / "memory_manifest.json", {
        "protocol_id": PROTOCOL_ID,
        "builder_input_count": len(train_records),
        "memory_split_count": len(memory_records),
        "calibration_split_count": len(calibration_records),
        "threshold_fraction": 8,
        "layers": list(LAYERS),
        "entries_per_layer": {
            str(layer): int(model.prototype_embeddings[layer].shape[0]) for layer in LAYERS
        },
        "shared_by_all_methods": True,
        "memory_construction": "layerwise_calls_to_existing_exact_official_builder",
        "host_config": config,
    })

    calibration_samples = {layer: [] for layer in LAYERS}
    calibration_root = output / "raw_cache" / "calibration"
    for index, record in enumerate(calibration_records, 1):
        relative = Path(record.path).resolve().relative_to(dataset_root).as_posix()
        raw, _ = _observe_or_load(
            calibration_root / f"calibration_{index - 1:04d}.npz",
            relative_path=relative, dataset_root=str(dataset_root), image_path=record.path,
            model=model, device=device,
        )
        for offset, layer in enumerate(LAYERS):
            calibration_samples[layer].append(raw[offset].reshape(-1))
        print(f"[MAD-EQUAL-AEBAD-S-P0] calibration {index}/{len(calibration_records)}", flush=True)
    pooled = {layer: np.concatenate(calibration_samples[layer]) for layer in LAYERS}
    mad = _fit_source_calibrators(pooled)
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
            relative_path=locator.relative_path, dataset_root=str(dataset_root), image_path=locator.path,
            model=model, device=device,
        )
        print(f"[MAD-EQUAL-AEBAD-S-P0] test raw {index}/{len(locators)}", flush=True)
    _atomic_json(output / "prediction_barrier.json", {
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "prediction_count": len(locators),
        "ground_truth_parsed": False,
    })

    records = load_phase_b_records(dataset_root, locators, formal=True)
    manifest_rows = []
    for locator, record in zip(locators, records, strict=True):
        row = record.as_row()
        row.update({"sample_id": locator.sample_id, "relative_path": locator.relative_path})
        manifest_rows.append(row)
    _write_csv(output / "dataset_manifest.csv", manifest_rows)

    group_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for domain in DOMAINS:
        indices = [index for index, record in enumerate(records) if record.domain == domain]
        labels = [records[index].label for index in indices]
        masks: list[np.ndarray] = []
        method_maps = {method: [] for method in METHODS}
        for position, index in enumerate(indices, 1):
            locator, record = locators[index], records[index]
            raw, output_shape = _observe_or_load(
                target_root / f"{locator.sample_id}.npz",
                relative_path=locator.relative_path, dataset_root=str(dataset_root), image_path=locator.path,
                model=model, device=device, allow_observe=False,
            )
            scores = _score_raw(raw, mad=mad)
            masks.append(_load_mask(record, output_shape))
            for method in METHODS:
                final_map = interpolate_score(scores[method], output_shape).astype(np.float32)
                method_maps[method].append(final_map)
                sample_rows.append({
                    "sample_id": locator.sample_id,
                    "relative_path": locator.relative_path,
                    "domain": domain,
                    "label": record.label,
                    "method": method,
                    "image_score": float(final_map.max()),
                })
            print(f"[MAD-EQUAL-AEBAD-S-P0] score {domain} {position}/{len(indices)}", flush=True)
        for method in METHODS:
            metrics = _evaluate(labels, masks, method_maps[method])
            group_rows.append({
                "method": method, "group": domain, "image_count": len(indices), **metrics,
            })
        del method_maps, masks

    _add_domain_deltas(group_rows)
    summary_rows = summarize_group_metrics(group_rows)
    _write_csv(output / "sample_scores.csv", sample_rows)
    _write_csv(output / "group_metrics.csv", group_rows)
    _write_csv(output / "summary.csv", summary_rows)
    report = _objective_report(group_rows, summary_rows)
    (output / "objective_report.md").write_text(report, encoding="utf-8")
    _atomic_json(output / "complete.json", {
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "run_type": "formal",
        "train_count": len(train_records),
        "memory_count": len(memory_records),
        "calibration_count": len(calibration_records),
        "prediction_count": len(records),
        "methods": list(METHODS),
        "shared_memory": True,
        "shared_raw_distance_tensor": True,
        "aupro_backend": "ADEval CUDA nstrips=200",
    })
    print(f"MAD_EQUAL_AEBAD_S_FULL_P0_COMPLETE={output}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AeBAD-S Exact SuperADD + MAD-Equal Full P0")
    parser.add_argument("--aebad-root", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--dinov3-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
