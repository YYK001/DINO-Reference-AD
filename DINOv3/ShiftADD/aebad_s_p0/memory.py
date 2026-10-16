"""Anchor-50K plus streaming factor memory construction for ShiftADD v2."""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from DINOv3.MADEqual.aebad_s_full_p0 import runner as baseline_runner
from DINOv3.relation_reliability.datasets import Record
from DINOv3.relation_reliability.superadd_host import (
    deterministic_official_preprocessing,
    image_tensor,
)

from .protocol import (
    ANCHOR_QUOTA,
    AUDIT_REVISION,
    BRANCH_CODES,
    CACHE_REVISION,
    CONTROL_FACTOR_QUOTA,
    FACTOR_QUOTAS,
    FEATURE_DIM,
    LAYERS,
    MAX_DATABASE_SIZE,
    PROTOCOL_ID,
    SAMPLER_REVISION,
    TRANSFORM_REVISION,
    brightness2_factors,
    make_candidate_plans,
    protocol_id_for_variant,
)
from .sampler import (
    PriorityReservoir,
    ReservoirResult,
    anchor_indices,
    patch_priorities_from_seed,
    path_priority_seed,
)
from .transforms import TransformResult, apply_shift_transform


SUBTYPE_NAMES = (
    "brightness",
    "contrast",
    "gamma",
    "white_balance",
    "smooth_shadow",
    "affine_perspective_2d",
    "border_connected_background",
    "pose_containment_invalid",
    "background_mask_invalid",
    "official_brightness_draw_2",
)
SUBTYPE_CODES = {name: index for index, name in enumerate(SUBTYPE_NAMES)}


@dataclass(frozen=True)
class FactorProvenance:
    source_indices: np.ndarray
    branch_codes: np.ndarray
    subtype_codes: np.ndarray
    patch_indices: np.ndarray
    pose_attempts: np.ndarray
    scene_overlap: np.ndarray


@dataclass(frozen=True)
class PartitionedMemory:
    anchor_banks: dict[int, torch.Tensor]
    factor_banks: dict[int, torch.Tensor]
    anchor_official_indices: dict[int, np.ndarray]
    factor_provenance: dict[int, FactorProvenance]
    variant: str
    protocol_id: str


@dataclass(frozen=True)
class MemoryBuildResult:
    memory: PartitionedMemory
    augmentation_rows: list[dict[str, Any]]
    composition_rows: list[dict[str, Any]]
    scene_overlap_rows: list[dict[str, Any]]


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
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


def _relative_path(record: Record, dataset_root: Path) -> str:
    return Path(record.path).resolve().relative_to(dataset_root.resolve()).as_posix()


def load_official_banks(path: str | Path) -> dict[int, torch.Tensor]:
    """Load the already-completed baseline Official 100K banks once."""
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    if (
        state.get("protocol_id") != baseline_runner.PROTOCOL_ID
        or int(state.get("source_count", -1)) != 521
    ):
        raise RuntimeError("baseline official_memory.pt has the wrong protocol identity")
    banks: dict[int, torch.Tensor] = {}
    for layer in LAYERS:
        value = state.get(f"bank_{layer}")
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != (
            MAX_DATABASE_SIZE,
            FEATURE_DIM,
        ):
            raise RuntimeError(
                f"baseline Official bank L{layer} must be (100000,1280)"
            )
        banks[layer] = value.detach().cpu().to(dtype=torch.float32).contiguous()
    return banks


def select_official_anchors(
    official_banks: dict[int, torch.Tensor],
) -> tuple[dict[int, torch.Tensor], dict[int, np.ndarray]]:
    anchors: dict[int, torch.Tensor] = {}
    indices: dict[int, np.ndarray] = {}
    for layer in LAYERS:
        selected = anchor_indices(
            layer, count=MAX_DATABASE_SIZE, quota=ANCHOR_QUOTA,
        )
        anchors[layer] = official_banks[layer].index_select(
            0, torch.from_numpy(selected.astype(np.int64, copy=False)),
        ).contiguous()
        indices[layer] = selected.astype(np.int64, copy=False)
    return anchors, indices


def _validate_host(model: Any) -> None:
    if tuple(int(value) for value in model.layers) != tuple(LAYERS):
        raise RuntimeError("ShiftADD v2 requires the frozen four-layer Exact host")
    if int(model.patch_exec.patch_size) != 640 or int(model.patch_exec.patch_overlap) != 128:
        raise RuntimeError("ShiftADD v2 requires patch geometry 640/128")
    if int(model.backbone.dino.embed_dim) != FEATURE_DIM:
        raise RuntimeError("ShiftADD v2 requires feature dimension 1280")


def explicit_official_brightness_tensor(
    tensor: torch.Tensor,
    factor: float,
) -> torch.Tensor:
    """Apply the legacy Official brightness operation before normalization."""
    return torch.clip(tensor * float(factor), 0.0, 1.0)


@torch.inference_mode()
def _extract_four_layers(
    model: Any,
    image: Image.Image,
    *,
    device: torch.device,
    brightness_factor: float | None = None,
) -> tuple[dict[int, np.ndarray], tuple[int, int]]:
    tensor = image_tensor(image).to(device=device)[None]
    if brightness_factor is not None:
        tensor = explicit_official_brightness_tensor(tensor, brightness_factor)
    processed = deterministic_official_preprocessing(model, tensor)
    predictions = model.patch_exec(processed, model.backbone)
    if len(predictions) != len(LAYERS):
        raise RuntimeError("factor forward did not return all four frozen layers")
    output: dict[int, np.ndarray] = {}
    expected_grid: tuple[int, int] | None = None
    for layer, prediction in zip(LAYERS, predictions):
        value = np.asarray(prediction, dtype=np.float32)
        if value.ndim != 4 or value.shape[0] != 1 or value.shape[-1] != FEATURE_DIM:
            raise RuntimeError(f"invalid factor feature tensor for L{layer}: {value.shape}")
        grid = (int(value.shape[1]), int(value.shape[2]))
        if expected_grid is None:
            expected_grid = grid
        elif grid != expected_grid:
            raise RuntimeError("factor layers do not share the Exact stitched grid")
        flattened = np.ascontiguousarray(value.reshape(-1, FEATURE_DIM), dtype=np.float32)
        if not np.isfinite(flattened).all():
            raise RuntimeError(f"non-finite factor feature encountered at L{layer}")
        output[layer] = flattened
    del tensor, processed, predictions
    if expected_grid is None:
        raise RuntimeError("factor forward returned no feature grid")
    return output, expected_grid


def scene_overlap_at_grid(mask: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    values = torch.from_numpy(np.asarray(mask, dtype=np.float32))[None, None]
    overlap = F.adaptive_avg_pool2d(values, output_size=grid)[0, 0].numpy()
    return np.clip(overlap.reshape(-1), 0.0, 1.0).astype(np.float32, copy=False)


def _manifest_row(
    *,
    relative_path: str,
    source_index: int,
    planned_branch: str,
    result: TransformResult | None,
    brightness_factor: float | None = None,
    control_active: bool = True,
) -> dict[str, Any]:
    if result is None:
        actual = "BRIGHTNESS2" if control_active else "CONTROL_INACTIVE"
        subtype = "official_brightness_draw_2" if control_active else "not_generated"
        active = bool(control_active)
        params = (
            {"brightness_factor": float(brightness_factor)} if control_active else {}
        )
        pose_attempt = -1
        containment = False
        scene_valid = False
    else:
        actual = result.actual_branch
        subtype = result.subtype
        active = result.candidate_active
        params = result.params
        pose_attempt = result.pose_attempt
        containment = result.pose_containment_pass
        scene_valid = result.scene_valid
    return {
        "relative_path": relative_path,
        "memory_fit_source_index": int(source_index),
        "planned_branch": planned_branch,
        "actual_branch": actual,
        "candidate_active": bool(active),
        "subtype": subtype,
        "pose_attempt": int(pose_attempt),
        "pose_containment_pass": bool(containment),
        "scene_valid": bool(scene_valid),
        "params": json.dumps(params, sort_keys=True, separators=(",", ":")),
    }


def _reservoirs_for_variant(variant: str) -> dict[int, dict[str, PriorityReservoir]]:
    if variant == "shiftadd":
        quotas = FACTOR_QUOTAS
    elif variant == "brightness2_control":
        quotas = {"BRIGHTNESS2": CONTROL_FACTOR_QUOTA}
    else:
        raise ValueError(f"unsupported ShiftADD v2 variant: {variant}")
    return {
        layer: {
            branch: PriorityReservoir(quota, FEATURE_DIM)
            for branch, quota in quotas.items()
        }
        for layer in LAYERS
    }


def _subtype_code(name: str) -> int:
    if name not in SUBTYPE_CODES:
        raise RuntimeError(f"unsupported factor subtype: {name}")
    return int(SUBTYPE_CODES[name])


def _reservoir_results(
    reservoirs: dict[int, dict[str, PriorityReservoir]],
    variant: str,
) -> tuple[dict[int, torch.Tensor], dict[int, FactorProvenance]]:
    branch_order = (
        ("ILLUMINATION", "POSE", "SCENE_VALID")
        if variant == "shiftadd"
        else ("BRIGHTNESS2",)
    )
    factor_banks: dict[int, torch.Tensor] = {}
    provenance: dict[int, FactorProvenance] = {}
    for layer in LAYERS:
        parts: list[ReservoirResult] = [
            reservoirs[layer][branch].finalize() for branch in branch_order
        ]
        features = np.concatenate([part.features for part in parts], axis=0)
        source = np.concatenate([part.source_indices for part in parts])
        patch = np.concatenate([part.patch_indices for part in parts])
        subtype = np.concatenate([part.subtype_codes for part in parts])
        attempts = np.concatenate([part.pose_attempts for part in parts])
        overlap = np.concatenate([part.scene_overlap for part in parts])
        codes = np.concatenate([
            np.full(part.features.shape[0], BRANCH_CODES[branch], dtype=np.int16)
            for branch, part in zip(branch_order, parts)
        ])
        if features.shape != (CONTROL_FACTOR_QUOTA, FEATURE_DIM):
            raise RuntimeError(f"factor L{layer} must contain exactly 50000 entries")
        factor_banks[layer] = torch.from_numpy(
            np.ascontiguousarray(features, dtype=np.float32)
        )
        provenance[layer] = FactorProvenance(
            source_indices=source.astype(np.int32, copy=False),
            branch_codes=codes,
            subtype_codes=subtype.astype(np.int16, copy=False),
            patch_indices=patch.astype(np.int64, copy=False),
            pose_attempts=attempts.astype(np.int16, copy=False),
            scene_overlap=overlap.astype(np.float32, copy=False),
        )
    return factor_banks, provenance


def _composition_rows(
    reservoirs: dict[int, dict[str, PriorityReservoir]],
    provenance: dict[int, FactorProvenance],
    augmentation_rows: Sequence[dict[str, Any]],
    variant: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    branch_order = (
        ("ILLUMINATION", "POSE", "SCENE_VALID")
        if variant == "shiftadd"
        else ("BRIGHTNESS2",)
    )
    for layer in LAYERS:
        rows.append({
            "record_type": "bank",
            "layer": layer,
            "branch": "OFFICIAL_ANCHOR",
            "candidate_count": MAX_DATABASE_SIZE,
            "selected_count": ANCHOR_QUOTA,
            "selected_fraction": ANCHOR_QUOTA / MAX_DATABASE_SIZE,
            "unique_source_images": "not_available_for_official_bank",
        })
        layer_provenance = provenance[layer]
        for branch in branch_order:
            code = BRANCH_CODES[branch]
            selected_mask = layer_provenance.branch_codes == code
            rows.append({
                "record_type": "bank",
                "layer": layer,
                "branch": branch,
                "candidate_count": reservoirs[layer][branch].candidate_count,
                "selected_count": int(selected_mask.sum()),
                "selected_fraction": int(selected_mask.sum()) / float(
                    max(1, reservoirs[layer][branch].candidate_count)
                ),
                "unique_source_images": int(
                    np.unique(layer_provenance.source_indices[selected_mask]).size
                ),
            })
    if variant == "shiftadd":
        for branch in ("ILLUMINATION", "POSE", "SCENE"):
            assigned = [row for row in augmentation_rows if row["planned_branch"] == branch]
            active = [row for row in assigned if row["candidate_active"]]
            rows.append({
                "record_type": "image_audit",
                "layer": "all",
                "branch": branch,
                "assigned_image_count": len(assigned),
                "active_image_count": len(active),
                "noop_image_count": len(assigned) - len(active),
            })
        for branch in ("POSE_NOOP", "SCENE_NOOP"):
            rows.append({
                "record_type": "noop_image_audit",
                "layer": "all",
                "branch": branch,
                "image_count": sum(row["actual_branch"] == branch for row in augmentation_rows),
                "selected_count": 0,
            })
        for attempt in range(8):
            rows.append({
                "record_type": "pose_attempt",
                "layer": "all",
                "branch": "POSE",
                "pose_attempt": attempt,
                "image_count": sum(
                    row["actual_branch"] == "POSE" and int(row["pose_attempt"]) == attempt
                    for row in augmentation_rows
                ),
            })
        for subtype in ("brightness", "contrast", "gamma", "white_balance", "smooth_shadow"):
            rows.append({
                "record_type": "illumination_subtype",
                "layer": "all",
                "branch": "ILLUMINATION",
                "subtype": subtype,
                "image_count": sum(row["subtype"] == subtype for row in augmentation_rows),
            })
    return rows


def _scene_overlap_rows(provenance: dict[int, FactorProvenance]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    intervals = (
        ("r_scene == 0", lambda value: value == 0.0),
        ("0 < r_scene <= 0.25", lambda value: (value > 0.0) & (value <= 0.25)),
        ("0.25 < r_scene <= 0.50", lambda value: (value > 0.25) & (value <= 0.50)),
        ("0.50 < r_scene <= 1.00", lambda value: (value > 0.50) & (value <= 1.00)),
    )
    for layer in LAYERS:
        selected = provenance[layer].branch_codes == BRANCH_CODES["SCENE_VALID"]
        overlap = provenance[layer].scene_overlap[selected]
        if overlap.size != FACTOR_QUOTAS["SCENE_VALID"] or not np.isfinite(overlap).all():
            raise RuntimeError(f"Scene overlap audit L{layer} is incomplete")
        total = int(overlap.size)
        closed = 0
        for label, predicate in intervals:
            mask = predicate(overlap)
            count = int(mask.sum())
            closed += count
            rows.append({
                "layer": layer,
                "overlap_bin": label,
                "selected_count": count,
                "fraction": count / float(total),
                "mean_overlap": float(overlap[mask].mean()) if count else 0.0,
            })
        if closed != total:
            raise RuntimeError(f"Scene overlap bins do not close at L{layer}")
    return rows


def _save_memory(path: Path, memory: PartitionedMemory, *, source_count: int) -> None:
    state: dict[str, Any] = {
        "protocol_id": memory.protocol_id,
        "variant": memory.variant,
        "transform_revision": TRANSFORM_REVISION,
        "sampler_revision": SAMPLER_REVISION,
        "audit_revision": AUDIT_REVISION,
        "cache_revision": CACHE_REVISION,
        "source_count": int(source_count),
        "layers": list(LAYERS),
        "feature_dim": FEATURE_DIM,
        "anchor_entries": ANCHOR_QUOTA,
        "factor_entries": CONTROL_FACTOR_QUOTA,
        "logical_entries": MAX_DATABASE_SIZE,
        "subtype_names": list(SUBTYPE_NAMES),
    }
    for layer in LAYERS:
        state[f"anchor_bank_{layer}"] = memory.anchor_banks[layer].detach().cpu()
        state[f"factor_bank_{layer}"] = memory.factor_banks[layer].detach().cpu()
        state[f"anchor_official_indices_{layer}"] = memory.anchor_official_indices[layer]
        item = memory.factor_provenance[layer]
        state[f"factor_source_indices_{layer}"] = item.source_indices
        state[f"factor_branch_codes_{layer}"] = item.branch_codes
        state[f"factor_subtype_codes_{layer}"] = item.subtype_codes
        state[f"factor_patch_indices_{layer}"] = item.patch_indices
        state[f"factor_pose_attempts_{layer}"] = item.pose_attempts
        state[f"factor_scene_overlap_{layer}"] = item.scene_overlap
    _atomic_torch_save(path, state)


def load_partitioned_memory(
    path: str | Path,
    *,
    variant: str,
    device: torch.device,
) -> PartitionedMemory | None:
    location = Path(path)
    if not location.is_file():
        return None
    state = torch.load(location, map_location="cpu", weights_only=False)
    expected_protocol = protocol_id_for_variant(variant)
    identity = (
        state.get("protocol_id") == expected_protocol
        and state.get("variant") == variant
        and state.get("transform_revision") == TRANSFORM_REVISION
        and state.get("sampler_revision") == SAMPLER_REVISION
        and state.get("audit_revision") == AUDIT_REVISION
        and state.get("cache_revision") == CACHE_REVISION
        and int(state.get("source_count", -1)) == 455
        and tuple(state.get("layers", ())) == tuple(LAYERS)
        and int(state.get("feature_dim", -1)) == FEATURE_DIM
        and int(state.get("logical_entries", -1)) == MAX_DATABASE_SIZE
    )
    if not identity:
        raise RuntimeError(f"incompatible ShiftADD v2 memory: {location}")
    anchors: dict[int, torch.Tensor] = {}
    factors: dict[int, torch.Tensor] = {}
    anchor_indices_by_layer: dict[int, np.ndarray] = {}
    provenance: dict[int, FactorProvenance] = {}
    for layer in LAYERS:
        anchor = state.get(f"anchor_bank_{layer}")
        factor = state.get(f"factor_bank_{layer}")
        if not isinstance(anchor, torch.Tensor) or tuple(anchor.shape) != (ANCHOR_QUOTA, FEATURE_DIM):
            raise RuntimeError(f"invalid Anchor-50K bank at L{layer}")
        if not isinstance(factor, torch.Tensor) or tuple(factor.shape) != (CONTROL_FACTOR_QUOTA, FEATURE_DIM):
            raise RuntimeError(f"invalid Factor-50K bank at L{layer}")
        anchors[layer] = anchor.to(device=device, dtype=torch.float32).contiguous()
        factors[layer] = factor.to(device=device, dtype=torch.float32).contiguous()
        anchor_indices_by_layer[layer] = np.asarray(
            state[f"anchor_official_indices_{layer}"], dtype=np.int64,
        )
        provenance[layer] = FactorProvenance(
            source_indices=np.asarray(state[f"factor_source_indices_{layer}"], dtype=np.int32),
            branch_codes=np.asarray(state[f"factor_branch_codes_{layer}"], dtype=np.int16),
            subtype_codes=np.asarray(state[f"factor_subtype_codes_{layer}"], dtype=np.int16),
            patch_indices=np.asarray(state[f"factor_patch_indices_{layer}"], dtype=np.int64),
            pose_attempts=np.asarray(state[f"factor_pose_attempts_{layer}"], dtype=np.int16),
            scene_overlap=np.asarray(state[f"factor_scene_overlap_{layer}"], dtype=np.float32),
        )
    return PartitionedMemory(
        anchor_banks=anchors,
        factor_banks=factors,
        anchor_official_indices=anchor_indices_by_layer,
        factor_provenance=provenance,
        variant=variant,
        protocol_id=expected_protocol,
    )


def build_partitioned_memory(
    model: Any,
    official_banks: dict[int, torch.Tensor],
    memory_records: Sequence[Record],
    *,
    dataset_root: Path,
    device: torch.device,
    output_dir: Path,
    variant: str,
    active_factor_paths: set[str] | None = None,
) -> MemoryBuildResult:
    """Build one v2 factor stream; no Official candidates or feature kNN."""
    _validate_host(model)
    if len(memory_records) != 455 or any(record.label != 0 for record in memory_records):
        raise RuntimeError("ShiftADD v2 memory input must be 455 source-normal images")
    output_dir.mkdir(parents=True, exist_ok=True)
    anchors_cpu, anchor_indices_by_layer = select_official_anchors(official_banks)
    anchors = {
        layer: value.to(device=device, dtype=torch.float32).contiguous()
        for layer, value in anchors_cpu.items()
    }
    pairs = sorted(
        ((_relative_path(record, dataset_root), record) for record in memory_records),
        key=lambda item: item[0],
    )
    plans = {plan.relative_path: plan for plan in make_candidate_plans(path for path, _ in pairs)}
    reservoirs = _reservoirs_for_variant(variant)
    manifest_rows: list[dict[str, Any]] = []
    brightness = (
        brightness2_factors(relative for relative, _record in pairs)
        if variant == "brightness2_control"
        else {}
    )
    if variant == "brightness2_control" and active_factor_paths is None:
        raise RuntimeError("brightness2 control requires frozen active_factor_paths")

    for source_index, (relative, record) in enumerate(pairs):
        with Image.open(record.path) as source:
            image = source.convert("RGB")
        if variant == "shiftadd":
            result = apply_shift_transform(image, plans[relative])
            manifest_rows.append(_manifest_row(
                relative_path=relative,
                source_index=source_index,
                planned_branch=plans[relative].planned_branch,
                result=result,
            ))
            if not result.candidate_active:
                continue
            branch = result.actual_branch
            transformed = result.image
            factor = None
        else:
            active = relative in active_factor_paths
            factor = brightness.get(relative) if active else None
            manifest_rows.append(_manifest_row(
                relative_path=relative,
                source_index=source_index,
                planned_branch="BRIGHTNESS2",
                result=None,
                brightness_factor=factor,
                control_active=active,
            ))
            if not active:
                continue
            branch = "BRIGHTNESS2"
            transformed = image
            result = None

        layer_features, feature_grid = _extract_four_layers(
            model, transformed, device=device, brightness_factor=factor,
        )
        priority_seed = path_priority_seed(relative)
        scene_overlap = None
        if result is not None and branch == "SCENE_VALID":
            if result.scene_mask is None:
                raise RuntimeError("SCENE_VALID lacks its safe RGB mask")
            scene_overlap = scene_overlap_at_grid(result.scene_mask, feature_grid)
        for layer in LAYERS:
            features = layer_features[layer]
            patch_index = np.arange(features.shape[0], dtype=np.int64)
            priorities = patch_priorities_from_seed(
                priority_seed, layer, branch, features.shape[0],
            )
            overlap = None
            pose_attempt = -1
            subtype = "official_brightness_draw_2"
            if result is not None:
                pose_attempt = result.pose_attempt
                subtype = result.subtype
                if branch == "SCENE_VALID":
                    overlap = scene_overlap
                    if overlap.size != features.shape[0]:
                        raise RuntimeError("Scene overlap does not match feature patches")
            reservoirs[layer][branch].update(
                features,
                priorities,
                source_index=source_index,
                patch_indices=patch_index,
                subtype_code=_subtype_code(subtype),
                pose_attempt=pose_attempt,
                scene_overlap=overlap,
            )
        print(
            f"[SHIFTADD-AEBAD-S-P0-V2] factor {source_index + 1}/{len(pairs)} "
            f"active branch={branch}",
            flush=True,
        )

    factor_banks_cpu, provenance = _reservoir_results(reservoirs, variant)
    factor_banks = {
        layer: value.to(device=device, dtype=torch.float32).contiguous()
        for layer, value in factor_banks_cpu.items()
    }
    memory = PartitionedMemory(
        anchor_banks=anchors,
        factor_banks=factor_banks,
        anchor_official_indices=anchor_indices_by_layer,
        factor_provenance=provenance,
        variant=variant,
        protocol_id=protocol_id_for_variant(variant),
    )
    composition = _composition_rows(reservoirs, provenance, manifest_rows, variant)
    scene_rows = (
        _scene_overlap_rows(provenance)
        if variant == "shiftadd"
        else [{
            "layer": "not_applicable",
            "overlap_bin": "brightness2_control",
            "selected_count": 0,
            "fraction": 0.0,
            "mean_overlap": 0.0,
        }]
    )
    _write_csv(output_dir / "augmentation_manifest.csv", manifest_rows)
    _write_csv(output_dir / "memory_composition.csv", composition)
    _write_csv(output_dir / "scene_overlap_audit.csv", scene_rows)
    _save_memory(output_dir / "shiftadd_memory.pt", memory, source_count=len(pairs))
    return MemoryBuildResult(memory, manifest_rows, composition, scene_rows)
