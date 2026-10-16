"""RobustAD target-normal few-shot + MAD-Equal P0 runner."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score

from DINOv3.relation_oracle_cache.extract import (
    install_category_banks,
    mad_equal,
)
from DINOv3.relation_reliability.calibration import MADCalibrator, fit_calibrators
from DINOv3.relation_reliability.datasets import (
    Record,
    robustad_full_records,
    split_source_normal,
)
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
    mask_at_patch_grid,
    observe_pil,
)

from .scoring import FewshotObservation, observe_fewshot_pil
from .support import (
    SHOT_COUNTS,
    TARGET_SHIFTS,
    TARGET_SHIFTS_BY_CATEGORY,
    SupportFeatures,
    extract_or_load_support,
    relative_path,
    select_source_extra,
    select_target_support,
    support_manifest_rows,
)


@dataclass(frozen=True)
class CategorySpec:
    protocol_id: str
    shifts: tuple[str, ...]
    pixel_evaluation: bool


CATEGORY_SPECS = {
    "MetalParts": CategorySpec(
        "mad_equal_robustad_target_fewshot_metalparts_seed42_p0_v1",
        TARGET_SHIFTS_BY_CATEGORY["MetalParts"], True,
    ),
    "PCB": CategorySpec(
        "mad_equal_robustad_target_fewshot_pcb_seed42_p0_v1",
        TARGET_SHIFTS_BY_CATEGORY["PCB"], True,
    ),
    "PiledBags": CategorySpec(
        "mad_equal_robustad_target_fewshot_piledbags_seed42_p0_v1",
        TARGET_SHIFTS_BY_CATEGORY["PiledBags"], False,
    ),
}

PROTOCOL_ID = CATEGORY_SPECS["MetalParts"].protocol_id
CATEGORY = "MetalParts"
PIXEL_EVALUATION = True
SEED = 42
SOURCE_COUNT = 180
MEMORY_COUNT = 157
CALIBRATION_COUNT = 23
CHECKPOINT_SHA256 = "7c1da9a54b3bdb333f5ebc42e404b7f19b1b5bed504877623c9dc87397f41488"
METHODS = (
    "OFFICIAL", "MAD_EQUAL",
    "SOURCE_EXTRA_K1_MAD", "SOURCE_EXTRA_K2_MAD",
    "SOURCE_EXTRA_K4_MAD", "SOURCE_EXTRA_K8_MAD",
    "TARGET_MATCHED_K1_MAD", "TARGET_MATCHED_K2_MAD",
    "TARGET_MATCHED_K4_MAD", "TARGET_MATCHED_K8_MAD",
)
METRICS = ("image_AUROC", "image_AUPR", "pixel_AUROC", "pixel_AUPR", "AUPRO")
IMAGE_METRICS = ("image_AUROC", "image_AUPR")


def configure_category(category: str) -> CategorySpec:
    """Select one frozen category protocol for this single-process CLI run."""
    global CATEGORY, PROTOCOL_ID, TARGET_SHIFTS, PIXEL_EVALUATION
    spec = CATEGORY_SPECS[category]
    CATEGORY = category
    PROTOCOL_ID = spec.protocol_id
    TARGET_SHIFTS = spec.shifts
    PIXEL_EVALUATION = spec.pixel_evaluation
    return spec


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
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


def _sample_id(record: Record, root: Path) -> str:
    return relative_path(record, root)


def _cache_name(record: Record, root: Path) -> str:
    import hashlib
    key = _sample_id(record, root).encode("utf-8")
    return hashlib.sha256(key).hexdigest()[:20] + ".npz"


def _save_local_memory(path: Path, model: Any) -> None:
    payload: dict[str, Any] = {"protocol_id": PROTOCOL_ID, "source_count": SOURCE_COUNT}
    for layer in LAYERS:
        payload[f"bank_{layer}"] = model.prototype_embeddings[layer].detach().cpu()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _load_local_memory(path: Path, model: Any, device: torch.device) -> bool:
    if not path.is_file():
        return False
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("protocol_id") != PROTOCOL_ID or int(payload.get("source_count", -1)) != SOURCE_COUNT:
        raise RuntimeError("local Official memory belongs to another protocol")
    banks: dict[int, torch.Tensor] = {}
    for layer in LAYERS:
        value = payload.get(f"bank_{layer}")
        if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[1] != 1280:
            raise RuntimeError(f"invalid local Official bank L{layer}")
        if value.shape[0] != 100_000 or not torch.isfinite(value).all():
            raise RuntimeError(f"invalid local Official bank values L{layer}")
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
            raise RuntimeError("local MAD state belongs to another protocol")
        medians = np.asarray(payload["medians"], np.float32)
        scales = np.asarray(payload["scales"], np.float32)
    if medians.shape != (4,) or scales.shape != (4,) or not np.isfinite(medians).all() or np.any(scales <= 0):
        raise RuntimeError("invalid local MAD state")
    return MADCalibrator(medians, scales)


def _fit_source_mad(
    model: Any,
    calibration: Sequence[Record],
    *,
    output: Path,
    device: torch.device,
) -> MADCalibrator:
    pooled: dict[int, list[np.ndarray]] = {layer: [] for layer in LAYERS}
    cache_root = output / "source_calibration_raw"
    for index, record in enumerate(calibration, start=1):
        path = cache_root / f"{index:03d}.npz"
        if path.is_file():
            with np.load(path, allow_pickle=False) as payload:
                raw = np.asarray(payload["raw_distances"], np.float32)
        else:
            with Image.open(record.path) as source:
                raw = np.asarray(
                    observe_pil(model, source.convert("RGB"), device=device).raw_distances,
                    np.float32,
                )
            _atomic_npz(path, raw_distances=raw)
        if raw.ndim != 3 or raw.shape[0] != 4 or not np.isfinite(raw).all():
            raise RuntimeError(f"invalid source calibration raw cache: {path}")
        for offset, layer in enumerate(LAYERS):
            pooled[layer].append(raw[offset].reshape(-1))
        print(f"[TARGET-FEWSHOT-P0] source calibration {index}/{len(calibration)}", flush=True)
    _ecdf, mad = fit_calibrators({
        layer: np.concatenate(values) for layer, values in pooled.items()
    })
    return mad


@dataclass(frozen=True)
class ExternalBundle:
    raw_paths: Mapping[str, Path]
    medians: np.ndarray
    scales: np.ndarray


def _external_raw_index(root: Path) -> dict[str, Path]:
    category = root / "categories" / "robustad" / CATEGORY
    complete_path = category / "complete.json"
    if not complete_path.is_file():
        raise FileNotFoundError(complete_path)
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if complete.get("status") != "complete" or complete.get("category") != CATEGORY:
        raise RuntimeError(f"external relation cache is not a completed {CATEGORY} cache")
    result: dict[str, Path] = {}
    for path in sorted((category / "images").rglob("*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            image = str(Path(str(payload["image_path"].item())).resolve())
        if image in result:
            raise RuntimeError(f"duplicate external raw image: {image}")
        result[image] = path
    return result


def _try_external_bundle(
    model: Any,
    records: Sequence[Record],
    *,
    source_output: Path | None,
    relation_cache: Path | None,
    device: torch.device,
) -> ExternalBundle | None:
    if source_output is None or relation_cache is None:
        return None
    try:
        category = source_output / "categories" / "robustad" / CATEGORY
        complete_path = category / "complete.json"
        bank_path = category / "memory" / "official_banks.npz"
        calibration_path = category / "calibration" / "calibration_state.npz"
        for path in (complete_path, bank_path, calibration_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if (
            complete.get("status") != "complete"
            or complete.get("dataset") != "robustad"
            or complete.get("category") != CATEGORY
            or int(complete.get("seed", -1)) != SEED
        ):
            raise RuntimeError("external source baseline identity mismatch")
        with np.load(calibration_path, allow_pickle=False) as payload:
            medians = np.asarray(payload["mad_medians"], np.float32)
            scales = np.asarray(payload["mad_scales"], np.float32)
        if (
            medians.shape != (4,) or scales.shape != (4,)
            or not np.isfinite(medians).all() or not np.isfinite(scales).all()
            or np.any(scales <= 0)
        ):
            raise RuntimeError("external source MAD state is invalid")
        index = _external_raw_index(relation_cache)
        required = {str(Path(record.path).resolve()) for record in records}
        if not required.issubset(index):
            raise RuntimeError(f"external baseline raw cache misses {len(required - set(index))} images")
        install_category_banks(
            model, bank_path,
            expected_entries={layer: 100_000 for layer in LAYERS},
            device=device,
        )
        return ExternalBundle(index, medians, scales)
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
        print(f"[TARGET-FEWSHOT-P0] external bundle unavailable; rebuilding locally: {error}", flush=True)
        return None


def _external_raw(path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    with np.load(path, allow_pickle=False) as payload:
        raw = np.asarray(payload["raw_distances"], np.float32)
        name = "evaluation_output_shape" if "evaluation_output_shape" in payload.files else "output_shape"
        output_shape = tuple(int(value) for value in payload[name].tolist())
    if raw.ndim != 3 or raw.shape[0] != 4 or not np.isfinite(raw).all() or len(output_shape) != 2:
        raise RuntimeError(f"invalid external baseline raw cache: {path}")
    return raw, output_shape


def _support_tensors(
    features: Sequence[SupportFeatures],
    *,
    device: torch.device,
) -> dict[int, list[torch.Tensor]]:
    if len(features) != 8:
        raise RuntimeError("support tensor pool must contain eight images")
    return {
        layer: [
            torch.from_numpy(np.asarray(item.layers[layer], np.float32)).to(device=device)
            for item in features
        ]
        for layer in LAYERS
    }


def _save_observation(
    path: Path,
    record: Record,
    root: Path,
    observed: FewshotObservation,
    output_shape: tuple[int, int],
) -> None:
    values: dict[str, np.ndarray] = {
        "protocol_id": np.asarray(PROTOCOL_ID),
        "relative_path": np.asarray(_sample_id(record, root)),
        "output_shape": np.asarray(output_shape, np.int32),
        "official_raw": observed.official_raw,
    }
    for shot in SHOT_COUNTS:
        values[f"source_raw_{shot}"] = observed.source_raw[shot]
        values[f"target_raw_{shot}"] = observed.target_raw[shot]
        values[f"source_support_{shot}"] = observed.source_support_raw[shot]
        values[f"target_support_{shot}"] = observed.target_support_raw[shot]
    _atomic_npz(path, **values)


def _load_observation(path: Path, record: Record, root: Path) -> tuple[dict[str, np.ndarray], tuple[int, int]]:
    with np.load(path, allow_pickle=False) as payload:
        if str(payload["protocol_id"].item()) != PROTOCOL_ID:
            raise RuntimeError(f"raw cache protocol mismatch: {path}")
        if str(payload["relative_path"].item()) != _sample_id(record, root):
            raise RuntimeError(f"raw cache image mismatch: {path}")
        output_shape = tuple(int(value) for value in payload["output_shape"].tolist())
        names = [
            "official_raw",
            *[f"source_raw_{shot}" for shot in SHOT_COUNTS],
            *[f"target_raw_{shot}" for shot in SHOT_COUNTS],
            *[f"source_support_{shot}" for shot in SHOT_COUNTS],
            *[f"target_support_{shot}" for shot in SHOT_COUNTS],
        ]
        values = {name: np.asarray(payload[name], np.float32) for name in names if name in payload.files}
    expected = {
        "official_raw",
        *{f"source_raw_{shot}" for shot in SHOT_COUNTS},
        *{f"target_raw_{shot}" for shot in SHOT_COUNTS},
        *{f"source_support_{shot}" for shot in SHOT_COUNTS},
        *{f"target_support_{shot}" for shot in SHOT_COUNTS},
    }
    if set(values) != expected:
        raise RuntimeError(f"raw cache fields are incomplete: {path}")
    shape = values["official_raw"].shape
    if len(shape) != 3 or shape[0] != 4 or any(value.shape != shape or not np.isfinite(value).all() for value in values.values()):
        raise RuntimeError(f"raw cache shape/value failure: {path}")
    return values, output_shape


def _evaluation_mask(record: Record, shape: tuple[int, int]) -> np.ndarray:
    if record.label == 0:
        return np.zeros(shape, np.uint8)
    if not record.mask_path:
        raise FileNotFoundError(f"anomaly lacks mask: {record.path}")
    with Image.open(record.mask_path) as source:
        resized = source.convert("L").resize((shape[1], shape[0]), Image.Resampling.NEAREST)
    return (np.asarray(resized, np.uint8) > 0).astype(np.uint8)


def _patch_mask(record: Record, grid: tuple[int, int]) -> np.ndarray | None:
    if record.label == 0:
        return None
    if not record.mask_path:
        raise FileNotFoundError(f"anomaly lacks mask: {record.path}")
    with Image.open(record.mask_path) as source:
        return mask_at_patch_grid(source, grid)


def _method_raw(values: Mapping[str, np.ndarray], method: str) -> np.ndarray:
    if method in ("OFFICIAL", "MAD_EQUAL"):
        return values["official_raw"]
    parts = method.split("_")
    shot = int(parts[-2][1:])
    branch = "source" if method.startswith("SOURCE_EXTRA") else "target"
    return values[f"{branch}_raw_{shot}"]


def _method_map(values: Mapping[str, np.ndarray], method: str, mad: MADCalibrator, shape: tuple[int, int]) -> np.ndarray:
    raw = _method_raw(values, method)
    patch = raw.mean(axis=0) if method == "OFFICIAL" else mad_equal(raw, mad.medians, mad.scales)
    result = interpolate_score(patch, shape).astype(np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError(f"non-finite prediction map: {method}")
    return result


def _metric_row(
    records: Sequence[Record], maps: Sequence[np.ndarray], method: str, shift: str,
    *, pixel_evaluation: bool | None = None,
) -> dict[str, Any]:
    evaluate_pixels = PIXEL_EVALUATION if pixel_evaluation is None else pixel_evaluation
    labels = np.asarray([record.label for record in records], np.uint8)
    image_scores = np.asarray([float(value.max()) for value in maps], np.float64)
    result = {
        "dataset": "robustad", "category": CATEGORY, "domain": "target" if shift != "source" else "source",
        "shift": shift, "method": method, "image_count": len(records),
        "normal_count": int((labels == 0).sum()), "anomaly_count": int((labels == 1).sum()),
        "image_AUROC": float(roc_auc_score(labels, image_scores)),
        "image_AUPR": float(average_precision_score(labels, image_scores)),
    }
    if not evaluate_pixels:
        result.update({"pixel_AUROC": "", "pixel_AUPR": "", "AUPRO": ""})
        return result
    masks = [_evaluation_mask(record, value.shape) for record, value in zip(records, maps, strict=True)]
    truth = np.concatenate([value.reshape(-1) for value in masks])
    score = np.concatenate([value.reshape(-1) for value in maps])
    result.update({
        "pixel_AUROC": float(roc_auc_score(truth, score)),
        "pixel_AUPR": float(average_precision_score(truth, score)),
        "AUPRO": official_aupro_adeval(masks, maps, nstrips=200),
    })
    return result


def summarize_target_macro(
    rows: Sequence[dict[str, Any]],
    source_rows: Sequence[dict[str, Any]],
    *,
    shifts: Sequence[str] | None = None,
    pixel_evaluation: bool | None = None,
) -> list[dict[str, Any]]:
    selected_shifts = tuple(shifts or TARGET_SHIFTS)
    evaluate_pixels = PIXEL_EVALUATION if pixel_evaluation is None else pixel_evaluation
    source = {row["method"]: row for row in source_rows}
    output: list[dict[str, Any]] = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        if {row["shift"] for row in selected} != set(selected_shifts):
            raise RuntimeError(f"target macro is incomplete for {method}")
        source_method = "OFFICIAL" if method == "OFFICIAL" else "MAD_EQUAL"
        result: dict[str, Any] = {"method": method, "scope": f"{len(selected_shifts)}_target_shift_macro"}
        for metric in METRICS:
            if not evaluate_pixels and metric not in IMAGE_METRICS:
                result[metric] = ""
                result[f"ARD_{metric}"] = ""
                continue
            values = [float(row[metric]) for row in selected]
            source_value = float(source[source_method][metric])
            result[metric] = mean(values)
            result[f"ARD_{metric}"] = mean(min(0.0, value - source_value) for value in values)
        output.append(result)
    return output


def _difference_rows(
    rows: Sequence[dict[str, Any]], macro: Sequence[dict[str, Any]],
    *, pixel_evaluation: bool | None = None,
) -> list[dict[str, Any]]:
    evaluate_pixels = PIXEL_EVALUATION if pixel_evaluation is None else pixel_evaluation
    combined = list(rows) + [dict(row, shift=row["scope"]) for row in macro]
    output: list[dict[str, Any]] = []
    for row in combined:
        if row["method"] in ("OFFICIAL", "MAD_EQUAL"):
            continue
        scope = row["shift"]
        baseline = next(item for item in combined if item["method"] == "MAD_EQUAL" and item["shift"] == scope)
        shot = int(str(row["method"]).split("_")[-2][1:])
        control_name = f"SOURCE_EXTRA_K{shot}_MAD"
        control = next(item for item in combined if item["method"] == control_name and item["shift"] == scope)
        value: dict[str, Any] = {"method": row["method"], "shift_or_scope": scope, "K": shot}
        for metric in METRICS:
            if not evaluate_pixels and metric not in IMAGE_METRICS:
                value[f"delta_vs_MAD_EQUAL_{metric}"] = ""
                value[f"delta_vs_SOURCE_EXTRA_K{shot}_{metric}"] = ""
                continue
            value[f"delta_vs_MAD_EQUAL_{metric}"] = float(row[metric]) - float(baseline[metric])
            value[f"delta_vs_SOURCE_EXTRA_K{shot}_{metric}"] = float(row[metric]) - float(control[metric])
        output.append(value)
    return output


def _mechanism_per_image_rows(
    shift: str,
    records: Sequence[Record],
    loaded: Sequence[tuple[dict[str, np.ndarray], tuple[int, int]]],
    mad: MADCalibrator,
    *,
    category: str | None = None,
    pixel_evaluation: bool | None = None,
    dataset_root: Path | None = None,
) -> list[dict[str, Any]]:
    selected_category = category or CATEGORY
    evaluate_pixels = PIXEL_EVALUATION if pixel_evaluation is None else pixel_evaluation
    output: list[dict[str, Any]] = []
    patch_masks: list[np.ndarray | None] = []
    for record, (values, _shape) in zip(records, loaded, strict=True):
        if record.label == 1 and evaluate_pixels:
            patch_masks.append(_patch_mask(record, tuple(values["official_raw"].shape[1:])))
        else:
            patch_masks.append(None)
    for branch, prefix in (("source", "SOURCE_EXTRA"), ("target", "TARGET_MATCHED")):
        for shot in SHOT_COUNTS:
            for record, (values, _shape), patch_mask in zip(records, loaded, patch_masks, strict=True):
                base = values["official_raw"]
                new = values[f"{branch}_raw_{shot}"]
                support = values[f"{branch}_support_{shot}"]
                reduction = (base - new) / (np.asarray(mad.scales)[:, None, None] + 1e-8)
                for offset, layer in enumerate(LAYERS):
                    row: dict[str, Any] = {
                        "category": selected_category,
                        "shift": shift,
                        "sample_id": _sample_id(record, dataset_root) if dataset_root is not None else str(record.path),
                        "label": int(record.label),
                        "method": f"{prefix}_K{shot}_MAD",
                        "K": shot,
                        "layer": layer,
                        "R_value": "",
                        "normal_hit_rate": "",
                        "defect_hit_rate": "",
                        "valid_region": "unavailable_image_only" if record.label else "normal_all_patches",
                    }
                    if record.label == 0:
                        row["R_value"] = float(reduction[offset].mean())
                        # Per-image patch mean. Later aggregation gives every image equal weight.
                        row["normal_hit_rate"] = float((support[offset] < base[offset]).mean())
                    elif evaluate_pixels:
                        if patch_mask is None or not bool(patch_mask.any()):
                            raise RuntimeError(f"empty defect patch mask: {record.path}")
                        row["R_value"] = float(reduction[offset][patch_mask].mean())
                        row["defect_hit_rate"] = float(
                            (support[offset][patch_mask] < base[offset][patch_mask]).mean()
                        )
                        row["valid_region"] = "gt_defect_patches"
                    output.append(row)
    return output


def _mechanism_by_shift_layer(
    rows: Sequence[dict[str, Any]],
    *,
    pixel_evaluation: bool | None = None,
) -> list[dict[str, Any]]:
    evaluate_pixels = PIXEL_EVALUATION if pixel_evaluation is None else pixel_evaluation
    output: list[dict[str, Any]] = []
    keys = sorted({(str(row["shift"]), str(row["method"]), int(row["K"]), int(row["layer"])) for row in rows})
    for shift, method, shot, layer in keys:
        selected = [row for row in rows if (row["shift"], row["method"], row["K"], row["layer"]) == (shift, method, shot, layer)]
        normal = [row for row in selected if int(row["label"]) == 0]
        anomaly = [row for row in selected if int(row["label"]) == 1 and row["R_value"] != ""]
        if not normal:
            raise RuntimeError(f"mechanism normal support is empty: {shift}/{method}/L{layer}")
        rn = mean(float(row["R_value"]) for row in normal)
        ra: float | str = mean(float(row["R_value"]) for row in anomaly) if evaluate_pixels and anomaly else ""
        output.append({
            "shift": shift, "method": method, "K": shot, "layer": layer,
            "R_N": rn, "R_A": ra, "G": rn - float(ra) if ra != "" else "",
            "normal_hit_rate": mean(float(row["normal_hit_rate"]) for row in normal),
            "defect_hit_rate": mean(float(row["defect_hit_rate"]) for row in anomaly) if evaluate_pixels and anomaly else "",
            "normal_image_count": len(normal), "anomaly_image_count": len(anomaly),
            "hit_aggregation": "image_equal",
        })
    return output


def _mechanism_macro(
    rows: Sequence[dict[str, Any]],
    *, shifts: Sequence[str] | None = None,
    pixel_evaluation: bool | None = None,
) -> list[dict[str, Any]]:
    selected_shifts = tuple(shifts or TARGET_SHIFTS)
    evaluate_pixels = PIXEL_EVALUATION if pixel_evaluation is None else pixel_evaluation
    output: list[dict[str, Any]] = []
    keys = sorted({(str(row["method"]), int(row["K"]), int(row["layer"])) for row in rows})
    for method, shot, layer in keys:
        selected = [row for row in rows if (row["method"], row["K"], row["layer"]) == (method, shot, layer)]
        if {row["shift"] for row in selected} != set(selected_shifts):
            raise RuntimeError("mechanism macro is missing a target shift")
        value: dict[str, Any] = {
            "method": method, "K": shot, "layer": layer,
            "scope": f"{len(selected_shifts)}_target_shift_macro",
            "R_N": mean(float(row["R_N"]) for row in selected),
            "normal_hit_rate": mean(float(row["normal_hit_rate"]) for row in selected),
            "R_A": "", "G": "", "defect_hit_rate": "",
            "normal_image_count": sum(int(row["normal_image_count"]) for row in selected),
            "anomaly_image_count": sum(int(row["anomaly_image_count"]) for row in selected),
            "hit_aggregation": "image_equal",
        }
        if evaluate_pixels:
            for name in ("R_A", "G", "defect_hit_rate"):
                value[name] = mean(float(row[name]) for row in selected)
        output.append(value)
    return output


def _report(
    *,
    cache_mode: str,
    support_rows: Sequence[dict[str, Any]],
    group_rows: Sequence[dict[str, Any]],
    macro_rows: Sequence[dict[str, Any]],
    difference_rows: Sequence[dict[str, Any]],
    mechanism_macro: Sequence[dict[str, Any]],
) -> str:
    def shown(value: Any, *, signed: bool = False) -> str:
        if value in ("", None, "NA"):
            return "NA"
        return f"{float(value):+.6f}" if signed else f"{float(value):.6f}"

    macro_scope = f"{len(TARGET_SHIFTS)}_target_shift_macro"
    lines = [
        "# RobustAD Target-normal Few-shot + MAD-Equal P0",
        "",
        f"Protocol: `{PROTOCOL_ID}`",
        "",
        "Scientific status: `SCIENTIFIC_REVIEW_REQUIRED`.",
        "",
        "This is target-normal few-shot, condition-aware adaptation. It is not an unsupervised method.",
        "",
        f"Source baseline mode: `{cache_mode}`. Official100 uses the frozen 180-image source input and 157/23 split. All calibrated methods share one source-only MAD fit.",
        "",
        f"Each target shift uses a fixed nested eight-normal support pool. Those eight images are excluded from every method and every K; every anomaly remains held out for evaluation. This is {CATEGORY}, Seed42 only, and no statistical-significance claim is made.",
        "" if PIXEL_EVALUATION else "PiledBags is image-only: masks are not opened and Pixel AUROC, Pixel AUPR and AUPRO are unavailable.",
        "",
        "## Support pools",
        "",
        "| branch | shift | index | relative path |",
        "|---|---|---:|---|",
    ]
    for row in support_rows:
        lines.append(f"| {row['branch']} | {row['shift']} | {row['support_index']} | {row['relative_path']} |")
    lines.extend(("", "## Per-shift metrics", "", "| method | shift | N | Image AUROC | Image AUPR | Pixel AUROC | Pixel AUPR | AUPRO |", "|---|---|---:|---:|---:|---:|---:|---:|"))
    for row in group_rows:
        if row["shift"] == "source":
            continue
        lines.append(
            f"| {row['method']} | {row['shift']} | {row['image_count']} | {shown(row['image_AUROC'])} | "
            f"{shown(row['image_AUPR'])} | {shown(row['pixel_AUROC'])} | {shown(row['pixel_AUPR'])} | {shown(row['AUPRO'])} |"
        )
    lines.extend(("", f"## {len(TARGET_SHIFTS)}-shift macro", "", "| method | Image AUROC | Image AUPR | Pixel AUROC | Pixel AUPR | AUPRO | Image AUROC ARD | AUPRO ARD |", "|---|---:|---:|---:|---:|---:|---:|---:|"))
    for row in macro_rows:
        lines.append(f"| {row['method']} | {shown(row['image_AUROC'])} | {shown(row['image_AUPR'])} | {shown(row['pixel_AUROC'])} | {shown(row['pixel_AUPR'])} | {shown(row['AUPRO'])} | {shown(row['ARD_image_AUROC'])} | {shown(row['ARD_AUPRO'])} |")
    macro_differences = [row for row in difference_rows if row["shift_or_scope"] == macro_scope]
    lines.extend(("", "## Macro differences", "", "| method | K | d Image AUROC vs MAD | d Image AUROC vs source-extra | d AUPRO vs MAD | d AUPRO vs source-extra |", "|---|---:|---:|---:|---:|---:|"))
    for row in macro_differences:
        shot = int(row["K"])
        image_control_key = f"delta_vs_SOURCE_EXTRA_K{shot}_image_AUROC"
        aupro_control_key = f"delta_vs_SOURCE_EXTRA_K{shot}_AUPRO"
        lines.append(
            f"| {row['method']} | {shot} | {shown(row['delta_vs_MAD_EQUAL_image_AUROC'], signed=True)} | "
            f"{shown(row[image_control_key], signed=True)} | "
            f"{shown(row['delta_vs_MAD_EQUAL_AUPRO'], signed=True)} | "
            f"{shown(row[aupro_control_key], signed=True)} |"
        )
    lines.extend(("", "## Mechanism macro", "", "Signed pre-MAD reductions use `R_N`, defect-only `R_A`, and `G=R_N-R_A`.", "", "| method | K | layer | R_N | R_A | G | normal hit | defect hit |", "|---|---:|---:|---:|---:|---:|---:|---:|"))
    for row in mechanism_macro:
        lines.append(f"| {row['method']} | {row['K']} | {row['layer']} | {shown(row['R_N'])} | {shown(row['R_A'])} | {shown(row['G'])} | {shown(row['normal_hit_rate'])} | {shown(row['defect_hit_rate'])} |")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> None:
    spec = configure_category(args.category)
    started = time.perf_counter()
    timing: list[dict[str, Any]] = []
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _seed_everything()
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("formal Exact SuperADD execution requires CUDA")
    if spec.pixel_evaluation:
        validate_official_metric_backend()

    install_import_paths(args.project_root, args.official_root, args.dinov3_root)
    model, _config = construct_exact_host(
        args.official_root, args.dinov3_root, args.checkpoint,
        CHECKPOINT_SHA256,
    )
    device = torch.device(args.device)

    stage = time.perf_counter()
    train, evaluation = robustad_full_records(args.robustad_root, (CATEGORY,))
    source = train[CATEGORY]
    all_evaluation = evaluation[CATEGORY]
    if len(source) != SOURCE_COUNT or any(record.label != 0 for record in source):
        raise RuntimeError(f"{CATEGORY} source train must contain exactly 180 normals")
    memory_fit, calibration = split_source_normal(source, 8)
    if (len(memory_fit), len(calibration)) != (MEMORY_COUNT, CALIBRATION_COUNT):
        raise RuntimeError(f"{CATEGORY} source split is not 157/23")
    groups = {(record.domain, record.shift) for record in all_evaluation}
    if not {("target", shift) for shift in TARGET_SHIFTS}.issubset(groups):
        raise RuntimeError(f"{CATEGORY} target shifts are incomplete")
    capabilities = {record.mask_capability for record in all_evaluation if record.domain == "target"}
    if spec.pixel_evaluation and capabilities != {"pixel"}:
        raise RuntimeError(f"{CATEGORY} must provide pixel masks")
    if not spec.pixel_evaluation and "pixel" in capabilities:
        raise RuntimeError(f"{CATEGORY} is frozen as image-only")
    target_pools, heldout = select_target_support(
        all_evaluation, dataset_root=args.robustad_root,
        category=CATEGORY, shifts=TARGET_SHIFTS,
    )
    source_pool = select_source_extra(
        calibration, dataset_root=args.robustad_root, category=CATEGORY,
    )
    support_rows = support_manifest_rows(
        source_pool, target_pools, dataset_root=args.robustad_root, category=CATEGORY,
    )
    _write_csv(output / "support_manifest.csv", support_rows)
    timing.append({"stage": "dataset_and_support_selection", "seconds": time.perf_counter() - stage})

    stage = time.perf_counter()
    external = _try_external_bundle(
        model, all_evaluation,
        source_output=args.source_output_dir,
        relation_cache=args.relation_cache_root,
        device=device,
    )
    if external is not None:
        mad = MADCalibrator(external.medians, external.scales)
        cache_mode = "complete_external_bundle"
    else:
        memory_path = output / "source_baseline" / "official_banks.pt"
        if not _load_local_memory(memory_path, model, device):
            build_exact_official_memory(
                model, source, device=device,
                scratch_dir=output / "source_baseline" / "memory_scratch",
            )
            if any(int(model.prototype_embeddings[layer].shape[0]) != 100_000 for layer in LAYERS):
                raise RuntimeError(f"rebuilt {CATEGORY} Official memory is not 100K per layer")
            _save_local_memory(memory_path, model)
        mad_path = output / "source_baseline" / "mad_state.npz"
        mad = _load_mad(mad_path)
        if mad is None:
            mad = _fit_source_mad(model, calibration, output=output, device=device)
            _save_mad(mad_path, mad)
        cache_mode = "local_rebuild_or_resume"
    timing.append({"stage": "source_baseline", "seconds": time.perf_counter() - stage})

    stage = time.perf_counter()
    support_cache = output / "support_features"
    source_features = [
        extract_or_load_support(
            support_cache / "source_extra" / f"{index:02d}.npz", record,
            dataset_root=args.robustad_root, model=model, device=device,
            protocol_id=PROTOCOL_ID,
        )
        for index, record in enumerate(source_pool, start=1)
    ]
    target_features: dict[str, list[SupportFeatures]] = {}
    for shift in TARGET_SHIFTS:
        target_features[shift] = [
            extract_or_load_support(
                support_cache / "target_matched" / shift / f"{index:02d}.npz", record,
                dataset_root=args.robustad_root, model=model, device=device,
                protocol_id=PROTOCOL_ID,
            )
            for index, record in enumerate(target_pools[shift], start=1)
        ]
    bank_rows: list[dict[str, Any]] = []
    for branch, shift, features in (
        [("source_extra", "source", source_features)]
        + [("target_matched", shift, target_features[shift]) for shift in TARGET_SHIFTS]
    ):
        for shot in SHOT_COUNTS:
            for layer in LAYERS:
                bank_rows.append({
                    "branch": branch, "shift": shift, "K": shot, "layer": layer,
                    "image_count": shot,
                    "patch_entries": sum(int(item.layers[layer].shape[0]) for item in features[:shot]),
                })
    _write_csv(output / "support_bank_stats.csv", bank_rows)
    timing.append({"stage": "support_feature_extraction", "seconds": time.perf_counter() - stage})

    stage = time.perf_counter()
    source_tensors = _support_tensors(source_features, device=device)
    raw_root = output / "raw_cache"
    for shift in TARGET_SHIFTS:
        target_tensors = _support_tensors(target_features[shift], device=device)
        for index, record in enumerate(heldout[shift], start=1):
            path = raw_root / shift / _cache_name(record, args.robustad_root)
            if path.is_file():
                _load_observation(path, record, args.robustad_root)
            else:
                cached_raw: np.ndarray | None = None
                cached_shape: tuple[int, int] | None = None
                if external is not None:
                    cached_raw, cached_shape = _external_raw(
                        external.raw_paths[str(Path(record.path).resolve())]
                    )
                with Image.open(record.path) as source_image:
                    image = source_image.convert("RGB")
                    output_shape = cached_shape or (
                        image.height // int(model.evaluation_downscale),
                        image.width // int(model.evaluation_downscale),
                    )
                    observed = observe_fewshot_pil(
                        model, image,
                        official_banks=model.prototype_embeddings,
                        source_support=source_tensors,
                        target_support=target_tensors,
                        device=device,
                        cached_official_raw=cached_raw,
                    )
                _save_observation(path, record, args.robustad_root, observed, output_shape)
            print(f"[TARGET-FEWSHOT-P0] raw {shift} {index}/{len(heldout[shift])}", flush=True)
        del target_tensors
        torch.cuda.empty_cache()
    timing.append({"stage": "heldout_exact_nn", "seconds": time.perf_counter() - stage})

    # Source metrics are baseline-only and structurally shared by every calibrated adaptation.
    source_records = [record for record in all_evaluation if record.domain == "source" and record.shift == "source"]
    source_maps: dict[str, list[np.ndarray]] = {"OFFICIAL": [], "MAD_EQUAL": []}
    for record in source_records:
        if external is None:
            with Image.open(record.path) as source_image:
                image = source_image.convert("RGB")
                observation = observe_pil(model, image, device=device)
                raw = np.asarray(observation.raw_distances, np.float32)
                shape = (image.height // int(model.evaluation_downscale), image.width // int(model.evaluation_downscale))
        else:
            raw, shape = _external_raw(external.raw_paths[str(Path(record.path).resolve())])
        values = {"official_raw": raw}
        source_maps["OFFICIAL"].append(_method_map(values, "OFFICIAL", mad, shape))
        source_maps["MAD_EQUAL"].append(_method_map(values, "MAD_EQUAL", mad, shape))

    stage = time.perf_counter()
    source_metric_rows = [
        _metric_row(source_records, source_maps[method], method, "source")
        for method in ("OFFICIAL", "MAD_EQUAL")
    ]
    group_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    mechanism_per_image: list[dict[str, Any]] = []
    for shift in TARGET_SHIFTS:
        records = heldout[shift]
        loaded = [
            _load_observation(
                raw_root / shift / _cache_name(record, args.robustad_root),
                record, args.robustad_root,
            )
            for record in records
        ]
        for method in METHODS:
            maps: list[np.ndarray] = []
            for record, (values, shape) in zip(records, loaded, strict=True):
                score = _method_map(values, method, mad, shape)
                maps.append(score)
                sample_rows.append({
                    "category": CATEGORY,
                    "sample_id": _sample_id(record, args.robustad_root),
                    "shift": shift, "label": record.label, "method": method,
                    "image_score": float(score.max()),
                })
            group_rows.append(_metric_row(records, maps, method, shift))
        mechanism_per_image.extend(_mechanism_per_image_rows(
            shift, records, loaded, mad,
            category=CATEGORY, pixel_evaluation=spec.pixel_evaluation,
            dataset_root=args.robustad_root,
        ))
        print(f"[TARGET-FEWSHOT-P0] metrics {shift} complete", flush=True)
    all_group_rows = source_metric_rows + group_rows
    macro_rows = summarize_target_macro(
        group_rows, source_metric_rows,
        shifts=TARGET_SHIFTS, pixel_evaluation=spec.pixel_evaluation,
    )
    difference_rows = _difference_rows(
        group_rows, macro_rows, pixel_evaluation=spec.pixel_evaluation,
    )
    mechanism_rows = _mechanism_by_shift_layer(
        mechanism_per_image, pixel_evaluation=spec.pixel_evaluation,
    )
    mechanism_macro = _mechanism_macro(
        mechanism_rows, shifts=TARGET_SHIFTS,
        pixel_evaluation=spec.pixel_evaluation,
    )
    _write_csv(output / "sample_scores.csv", sample_rows)
    _write_csv(output / "group_metrics.csv", all_group_rows)
    _write_csv(output / "target_shift_macro.csv", macro_rows)
    _write_csv(output / "method_differences.csv", difference_rows)
    _write_csv(output / "mechanism_per_image.csv", mechanism_per_image)
    _write_csv(output / "mechanism_by_shift_layer.csv", mechanism_rows)
    _write_csv(output / "mechanism_macro.csv", mechanism_macro)
    timing.append({"stage": "metrics_and_mechanism", "seconds": time.perf_counter() - stage})
    timing.append({"stage": "total", "seconds": time.perf_counter() - started})
    _write_csv(output / "timing.csv", timing)
    (output / "objective_report.md").write_text(
        _report(
            cache_mode=cache_mode, support_rows=support_rows,
            group_rows=all_group_rows, macro_rows=macro_rows,
            difference_rows=difference_rows,
            mechanism_macro=mechanism_macro,
        ),
        encoding="utf-8",
    )
    _atomic_json(output / "complete.json", {
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "scientific_status": "SCIENTIFIC_REVIEW_REQUIRED",
        "dataset": "RobustAD", "category": CATEGORY, "seed": SEED,
        "source_count": SOURCE_COUNT, "memory_count": MEMORY_COUNT,
        "calibration_count": CALIBRATION_COUNT,
        "target_shifts": list(TARGET_SHIFTS), "K": list(SHOT_COUNTS),
        "methods": list(METHODS), "cache_mode": cache_mode,
        "aupro_backend": "ADEval CUDA nstrips=200" if spec.pixel_evaluation else "unavailable_image_only",
        "pixel_evaluation": "available" if spec.pixel_evaluation else "unavailable_image_only",
        "adaptation_scope": "target-normal few-shot condition-aware",
        "support_pool_size": 8,
    })
    print(f"TARGET_FEWSHOT_MAD_EQUAL_P0_COMPLETE={output}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robustad-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--dinov3-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-output-dir", type=Path)
    parser.add_argument("--relation-cache-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--category", choices=tuple(CATEGORY_SPECS), default="MetalParts")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
