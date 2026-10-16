"""Reuse the frozen SOURCE_EXTRA identities and create one Broad6 view each."""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from DINOv3.MADEqual.aebad_s_broad6_compose2.augmentations import broad6_compose2
from DINOv3.MADEqual.aebad_s_target_fewshot.support import (
    DOMAINS,
    EXPECTED_HELDOUT,
    SHOT_COUNTS,
    SupportFeatures,
    relative_path,
    support_prefixes,
)
from DINOv3.MADEqual.robustad_target_fewshot_p0.support import SUPPORT_POOL_SIZE
from DINOv3.ShiftADD.aebad_s_p0.memory import _extract_four_layers
from DINOv3.relation_reliability.datasets import Record
from DINOv3.relation_reliability.protocol import LAYERS


SEED = 42


@dataclass(frozen=True)
class SyntheticSupport:
    features: SupportFeatures
    manifest: Mapping[str, Any]


def load_base_support_paths(
    path: str | Path,
) -> tuple[list[str], dict[str, list[str]]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    source_rows = sorted(
        (row for row in rows if row["branch"] == "source_extra"),
        key=lambda row: int(row["support_index"]),
    )
    if [int(row["support_index"]) for row in source_rows] != list(range(1, 9)):
        raise RuntimeError("base SOURCE_EXTRA support indices must be exactly 1..8")
    source_paths = [row["relative_path"].replace("\\", "/") for row in source_rows]
    if len(set(source_paths)) != SUPPORT_POOL_SIZE:
        raise RuntimeError("base SOURCE_EXTRA identities are not eight distinct images")

    target_paths: dict[str, list[str]] = {}
    for domain in DOMAINS:
        selected = sorted(
            (
                row for row in rows
                if row["branch"] == "target_matched" and row["domain"] == domain
            ),
            key=lambda row: int(row["support_index"]),
        )
        if [int(row["support_index"]) for row in selected] != list(range(1, 9)):
            raise RuntimeError(f"base target support is incomplete: {domain}")
        target_paths[domain] = [row["relative_path"].replace("\\", "/") for row in selected]
    return source_paths, target_paths


def resolve_source_records(
    relative_paths: Sequence[str],
    records: Sequence[Record],
    *,
    dataset_root: Path,
) -> list[Record]:
    by_path = {relative_path(record, dataset_root): record for record in records}
    missing = [path for path in relative_paths if path not in by_path]
    if missing:
        raise RuntimeError(f"base SOURCE_EXTRA path is absent from AeBAD-S: {missing[0]}")
    selected = [by_path[path] for path in relative_paths]
    if len(selected) != 8 or len({record.path for record in selected}) != 8:
        raise RuntimeError("synthetic support must use eight distinct source images")
    if any(int(record.label) != 0 for record in selected):
        raise RuntimeError("synthetic support contains a non-normal source image")
    return selected


def heldout_from_base_manifest(
    records: Sequence[Record],
    target_paths: Mapping[str, Sequence[str]],
    *,
    dataset_root: Path,
) -> dict[str, list[Record]]:
    output: dict[str, list[Record]] = {}
    for domain in DOMAINS:
        excluded = set(target_paths[domain])
        if len(excluded) != 8:
            raise RuntimeError(f"base target support must contain eight images: {domain}")
        group = [record for record in records if record.domain == domain]
        known = {relative_path(record, dataset_root) for record in group}
        if not excluded <= known:
            raise RuntimeError(f"base target support path is absent from domain: {domain}")
        values = [
            record for record in group
            if relative_path(record, dataset_root) not in excluded
        ]
        counts = (
            sum(int(record.label) == 0 for record in values),
            sum(int(record.label) == 1 for record in values),
            len(values),
        )
        if counts != EXPECTED_HELDOUT[domain]:
            raise RuntimeError(f"held-out membership differs from base run: {domain}/{counts}")
        output[domain] = values
    if sum(len(values) for values in output.values()) != 1607:
        raise RuntimeError("synthetic experiment held-out set must contain 1607 images")
    return output


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def extract_or_load_synthetic_support(
    cache_path: Path,
    record: Record,
    *,
    support_index: int,
    dataset_root: Path,
    model: Any,
    device: torch.device,
    protocol_id: str,
) -> SyntheticSupport:
    rel = relative_path(record, dataset_root)
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as payload:
            if str(payload["protocol_id"].item()) != protocol_id:
                raise RuntimeError(f"synthetic support cache protocol mismatch: {cache_path}")
            if str(payload["relative_path"].item()) != rel:
                raise RuntimeError(f"synthetic support cache path mismatch: {cache_path}")
            if int(payload["support_index"].item()) != support_index:
                raise RuntimeError(f"synthetic support cache index mismatch: {cache_path}")
            grid = tuple(int(value) for value in payload["grid"].tolist())
            op1 = str(payload["op1"].item())
            op2 = str(payload["op2"].item())
            params1 = str(payload["op1_params"].item())
            params2 = str(payload["op2_params"].item())
            layers = {
                layer: np.asarray(payload[f"layer_{layer}"], dtype=np.float32)
                for layer in LAYERS
            }
    else:
        with Image.open(record.path) as source:
            transformed = broad6_compose2(source.convert("RGB"), rel, seed=SEED)
        layers, grid = _extract_four_layers(
            model, transformed.image, device=device, brightness_factor=None,
        )
        op1, op2 = transformed.op1, transformed.op2
        params1 = json.dumps(transformed.params1, sort_keys=True, separators=(",", ":"))
        params2 = json.dumps(transformed.params2, sort_keys=True, separators=(",", ":"))
        _atomic_npz(
            cache_path,
            protocol_id=np.asarray(protocol_id),
            relative_path=np.asarray(rel),
            support_index=np.asarray(support_index, np.int16),
            grid=np.asarray(grid, np.int32),
            op1=np.asarray(op1),
            op2=np.asarray(op2),
            op1_params=np.asarray(params1),
            op2_params=np.asarray(params2),
            **{f"layer_{layer}": layers[layer] for layer in LAYERS},
        )
    expected_rows = int(grid[0]) * int(grid[1])
    if op1 == op2:
        raise RuntimeError("Broad6 Compose2 support operations are not distinct")
    for layer in LAYERS:
        value = layers[layer]
        if value.shape != (expected_rows, 1280) or not np.isfinite(value).all():
            raise RuntimeError(f"invalid synthetic support feature L{layer}: {cache_path}")
    manifest = {
        "support_index": support_index,
        "relative_path": rel,
        "source_identity": rel,
        "synthetic_view_count": 1,
        "op1": op1,
        "op2": op2,
        "recipe": f"{op1}>{op2}",
        "op1_params": params1,
        "op2_params": params2,
        "included_in_K1": int(support_index <= 1),
        "included_in_K2": int(support_index <= 2),
        "included_in_K4": int(support_index <= 4),
        "included_in_K8": int(support_index <= 8),
        "shared_by_domains": ";".join(DOMAINS),
        "description": "GNL-inspired Broad6 Compose2 synthetic support",
    }
    return SyntheticSupport(SupportFeatures(rel, grid, layers), manifest)


__all__ = [
    "DOMAINS", "EXPECTED_HELDOUT", "SHOT_COUNTS", "SyntheticSupport",
    "extract_or_load_synthetic_support", "heldout_from_base_manifest",
    "load_base_support_paths", "resolve_source_records", "support_prefixes",
]
