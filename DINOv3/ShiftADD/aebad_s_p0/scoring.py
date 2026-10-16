"""One-query, partitioned Exact 1-NN scoring for ShiftADD v2."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image

from DINOv3.relation_reliability.superadd_host import (
    _exact_top1_chunked,
    _extract_processed_batch,
    image_tensor,
)

from .memory import PartitionedMemory
from .protocol import ANCHOR_QUOTA, BRANCH_CODES, LAYERS


@dataclass(frozen=True)
class PartitionedObservation:
    anchor_raw_distances: np.ndarray
    shift_raw_distances: np.ndarray
    combined_top1_indices: np.ndarray
    top1_branch_codes: np.ndarray
    grid: tuple[int, int]


def partitioned_exact_top1(
    queries: torch.Tensor,
    anchor: torch.Tensor,
    factor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Anchor distance plus min(Anchor,Factor) with Exact cdist semantics."""
    anchor_distance, anchor_index = _exact_top1_chunked(queries, anchor)
    factor_distance, factor_index = _exact_top1_chunked(queries, factor)
    factor_wins = ~(anchor_distance < factor_distance)
    shift_distance = torch.where(factor_wins, factor_distance, anchor_distance)
    combined_index = torch.where(
        factor_wins,
        factor_index + int(anchor.shape[0]),
        anchor_index,
    )
    return anchor_distance, shift_distance, combined_index, factor_wins


@torch.inference_mode()
def observe_partitioned_pil(
    model: Any,
    image: Image.Image,
    memory: PartitionedMemory,
    *,
    device: torch.device,
) -> PartitionedObservation:
    """Extract one four-layer query and scan the two 50K partitions."""
    tensor = image_tensor(image).to(device=device)[None]
    processed = model.preprocessing(tensor)
    pending_items = _extract_processed_batch(
        model,
        processed,
        device=device,
        query_masks=[None],
        include_identity_descriptors=False,
    )
    if len(pending_items) != 1:
        raise RuntimeError("partitioned observation expected one query image")
    pending = pending_items[0]
    grid = tuple(int(value) for value in pending.grid)
    anchor_raw = np.empty((len(LAYERS), *grid), dtype=np.float32)
    shift_raw = np.empty_like(anchor_raw)
    combined_top1 = np.empty((len(LAYERS), *grid), dtype=np.int64)
    branch_codes = np.empty((len(LAYERS), *grid), dtype=np.int16)

    for offset, layer in enumerate(LAYERS):
        query = pending.queries[layer]
        channels = int(query.shape[1])
        anchor_distance, shift_distance, combined_index, factor_wins = (
            partitioned_exact_top1(
                query,
                memory.anchor_banks[layer],
                memory.factor_banks[layer],
            )
        )
        anchor_values = anchor_distance.numpy() / float(channels)
        shift_values = shift_distance.numpy() / float(channels)
        anchor_raw[offset].reshape(-1)[pending.positions] = anchor_values
        shift_raw[offset].reshape(-1)[pending.positions] = shift_values
        combined = combined_index.numpy().astype(np.int64, copy=False)
        combined_top1[offset].reshape(-1)[pending.positions] = combined
        factor_selected = factor_wins.numpy().astype(bool, copy=False)
        selected_branch = np.full(combined.size, BRANCH_CODES["OFFICIAL_ANCHOR"], dtype=np.int16)
        if factor_selected.any():
            factor_index = combined[factor_selected] - ANCHOR_QUOTA
            selected_branch[factor_selected] = memory.factor_provenance[layer].branch_codes[
                factor_index
            ]
        branch_codes[offset].reshape(-1)[pending.positions] = selected_branch
        del anchor_distance, shift_distance, combined_index, factor_wins

    pending.queries.clear()
    del tensor, processed, pending_items
    if not np.isfinite(anchor_raw).all() or not np.isfinite(shift_raw).all():
        raise RuntimeError("partitioned Exact 1-NN produced non-finite distances")
    if np.any(shift_raw > anchor_raw + 1e-6):
        raise RuntimeError("partitioned minimum exceeds its Anchor distance")
    return PartitionedObservation(
        anchor_raw_distances=anchor_raw,
        shift_raw_distances=shift_raw,
        combined_top1_indices=combined_top1,
        top1_branch_codes=branch_codes,
        grid=grid,
    )
