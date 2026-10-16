"""Stable support selection and deterministic support-feature caching."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from DINOv3.relation_reliability.datasets import Record
from DINOv3.relation_reliability.protocol import LAYERS
from DINOv3.ShiftADD.aebad_s_p0.memory import _extract_four_layers


TARGET_SHIFTS_BY_CATEGORY = {
    "MetalParts": (
        "background_1", "background_2", "lighting",
        "position", "rotation", "scale",
    ),
    "PCB": ("lighting", "white_balancing", "rotation", "position", "shadow"),
    "PiledBags": (
        "background_box_color", "lighting", "position_rotation", "scale", "shadow",
    ),
}
# Backward-compatible alias for the completed MetalParts protocol and its tests.
TARGET_SHIFTS = TARGET_SHIFTS_BY_CATEGORY["MetalParts"]
SHOT_COUNTS = (1, 2, 4, 8)
SUPPORT_POOL_SIZE = 8


def relative_path(record: Record, dataset_root: Path) -> str:
    return Path(record.path).resolve().relative_to(dataset_root.resolve()).as_posix()


def stable_support_key(
    record: Record,
    *,
    dataset_root: Path,
    shift: str,
    seed: int = 42,
    category: str = "MetalParts",
) -> str:
    rel = relative_path(record, dataset_root)
    value = f"seed{seed}|{category}|{shift}|{rel}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def select_target_support(
    records: Sequence[Record],
    *,
    dataset_root: Path,
    category: str = "MetalParts",
    shifts: Sequence[str] | None = None,
) -> tuple[dict[str, list[Record]], dict[str, list[Record]]]:
    """Select fixed normal support pools and one common held-out set per shift."""
    selected_shifts = tuple(shifts or TARGET_SHIFTS_BY_CATEGORY[category])
    pools: dict[str, list[Record]] = {}
    heldout: dict[str, list[Record]] = {}
    for shift in selected_shifts:
        group = [
            record for record in records
            if record.domain == "target" and record.shift == shift
        ]
        normals = [record for record in group if record.label == 0]
        normals.sort(key=lambda record: stable_support_key(
            record, dataset_root=dataset_root, shift=shift, category=category,
        ))
        if len(normals) < SUPPORT_POOL_SIZE:
            raise RuntimeError(f"RobustAD/{category}/{shift} has fewer than 8 target normals")
        pool = normals[:SUPPORT_POOL_SIZE]
        selected = {str(Path(record.path).resolve()) for record in pool}
        evaluation = [
            record for record in group
            if str(Path(record.path).resolve()) not in selected
        ]
        if any(record.label != 0 for record in pool):
            raise RuntimeError("target support contains an anomaly")
        if any(record.label == 1 and str(Path(record.path).resolve()) in selected for record in group):
            raise RuntimeError("anomaly was removed from held-out evaluation")
        if {record.label for record in evaluation} != {0, 1}:
            raise RuntimeError(f"held-out shift lacks normal/anomaly support: {shift}")
        pools[shift] = pool
        heldout[shift] = evaluation
    return pools, heldout


def select_source_extra(
    calibration: Sequence[Record],
    *,
    dataset_root: Path,
    category: str = "MetalParts",
) -> list[Record]:
    if any(record.label != 0 for record in calibration):
        raise RuntimeError("source calibration contains a non-normal image")
    ordered = sorted(
        calibration,
        key=lambda record: stable_support_key(
            record, dataset_root=dataset_root, shift="source_extra", category=category,
        ),
    )
    if len(ordered) < SUPPORT_POOL_SIZE:
        raise RuntimeError("source calibration has fewer than 8 images")
    return ordered[:SUPPORT_POOL_SIZE]


def support_prefixes(pool: Sequence[Record]) -> dict[int, list[Record]]:
    if len(pool) < SUPPORT_POOL_SIZE:
        raise ValueError("support pool must contain at least 8 images")
    return {shot: list(pool[:shot]) for shot in SHOT_COUNTS}


@dataclass(frozen=True)
class SupportFeatures:
    relative_path: str
    grid: tuple[int, int]
    layers: Mapping[int, np.ndarray]


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def extract_or_load_support(
    cache_path: Path,
    record: Record,
    *,
    dataset_root: Path,
    model: Any,
    device: torch.device,
    protocol_id: str,
) -> SupportFeatures:
    rel = relative_path(record, dataset_root)
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as payload:
            if str(payload["protocol_id"].item()) != protocol_id:
                raise RuntimeError(f"support cache protocol mismatch: {cache_path}")
            if str(payload["relative_path"].item()) != rel:
                raise RuntimeError(f"support cache path mismatch: {cache_path}")
            grid = tuple(int(value) for value in payload["grid"].tolist())
            layers = {
                layer: np.asarray(payload[f"layer_{layer}"], dtype=np.float32)
                for layer in LAYERS
            }
    else:
        with Image.open(record.path) as source:
            layers, grid = _extract_four_layers(
                model, source.convert("RGB"), device=device,
                brightness_factor=None,
            )
        _atomic_npz(
            cache_path,
            protocol_id=np.asarray(protocol_id),
            relative_path=np.asarray(rel),
            grid=np.asarray(grid, dtype=np.int32),
            **{f"layer_{layer}": layers[layer] for layer in LAYERS},
        )
    expected_rows = int(grid[0]) * int(grid[1])
    for layer in LAYERS:
        value = layers[layer]
        if (
            value.ndim != 2 or value.shape != (expected_rows, 1280)
            or not np.isfinite(value).all()
        ):
            raise RuntimeError(f"invalid support feature cache L{layer}: {cache_path}")
    return SupportFeatures(rel, grid, layers)


def support_manifest_rows(
    source_pool: Sequence[Record],
    target_pools: Mapping[str, Sequence[Record]],
    *,
    dataset_root: Path,
    category: str = "MetalParts",
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for branch, shift, pool in (
        [("source_extra", "source", source_pool)]
        + [("target_matched", shift, pool) for shift, pool in target_pools.items()]
    ):
        for index, record in enumerate(pool, start=1):
            rows.append({
                "category": category,
                "branch": branch,
                "shift": shift,
                "support_index": index,
                "relative_path": relative_path(record, dataset_root),
                "label": record.label,
                "included_in_K1": int(index <= 1),
                "included_in_K2": int(index <= 2),
                "included_in_K4": int(index <= 4),
                "included_in_K8": int(index <= 8),
            })
    return rows
