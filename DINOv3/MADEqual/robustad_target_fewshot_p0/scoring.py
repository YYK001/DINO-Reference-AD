"""One-forward Exact Official/support scoring and mechanism aggregation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from DINOv3.relation_reliability.protocol import LAYERS
from DINOv3.relation_reliability.superadd_host import (
    _exact_top1_chunked,
    _extract_processed_batch,
    image_tensor,
)


PREFIXES = (1, 2, 4, 8)


def _notify_nearest(observer: Callable | None, pool: str, layer: int | None,
                    support_index: int, distances: np.ndarray, indices: Any) -> None:
    """Side-channel copies: observers cannot mutate the arrays used for scoring."""
    if observer is not None:
        distance_copy = distances.copy()
        index_copy = indices.numpy().copy()
        distance_copy.flags.writeable = index_copy.flags.writeable = False
        observer(pool, layer, support_index, distance_copy, index_copy)


def exact_prefix_minima(
    query: torch.Tensor,
    official: torch.Tensor,
    supports: Sequence[torch.Tensor],
    *,
    nearest_observer: Callable | None = None,
    layer: int | None = None,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Scan each support image once and return union and support-only prefix minima."""
    if len(supports) != 8:
        raise ValueError("exact prefix scoring requires exactly eight support images")
    channels = int(query.shape[1])
    if channels != 1280:
        raise ValueError(f"frozen feature dimension is 1280, got {channels}")
    official_distance, official_indices = _exact_top1_chunked(query, official)
    official_distance = official_distance.numpy().astype(np.float32) / float(channels)
    _notify_nearest(nearest_observer, "source", layer, 0, official_distance, official_indices)
    cumulative: np.ndarray | None = None
    union: dict[int, np.ndarray] = {}
    support_only: dict[int, np.ndarray] = {}
    for index, bank in enumerate(supports, start=1):
        distance, indices = _exact_top1_chunked(query, bank)
        value = distance.numpy().astype(np.float32) / float(channels)
        _notify_nearest(nearest_observer, "source_extra", layer, index, value, indices)
        cumulative = value if cumulative is None else np.minimum(cumulative, value)
        if index in PREFIXES:
            support_only[index] = cumulative.copy()
            union[index] = np.minimum(official_distance, cumulative)
    return official_distance, union, support_only


def _support_prefix_minima(
    query: torch.Tensor,
    supports: Sequence[torch.Tensor],
    *,
    nearest_observer: Callable | None = None,
    pool: str = "target",
    layer: int | None = None,
) -> dict[int, np.ndarray]:
    if len(supports) != 8:
        raise ValueError("support scoring requires exactly eight support images")
    channels = int(query.shape[1])
    cumulative: np.ndarray | None = None
    result: dict[int, np.ndarray] = {}
    for index, bank in enumerate(supports, start=1):
        distance, indices = _exact_top1_chunked(query, bank)
        value = distance.numpy().astype(np.float32) / float(channels)
        _notify_nearest(nearest_observer, pool, layer, index, value, indices)
        cumulative = value if cumulative is None else np.minimum(cumulative, value)
        if index in PREFIXES:
            result[index] = cumulative.copy()
    return result


@dataclass(frozen=True)
class FewshotObservation:
    official_raw: np.ndarray
    source_raw: Mapping[int, np.ndarray]
    target_raw: Mapping[int, np.ndarray]
    source_support_raw: Mapping[int, np.ndarray]
    target_support_raw: Mapping[int, np.ndarray]
    grid: tuple[int, int]


@torch.inference_mode()
def observe_fewshot_pil(
    model: Any,
    image: Image.Image,
    *,
    official_banks: Mapping[int, torch.Tensor],
    source_support: Mapping[int, Sequence[torch.Tensor]] | None,
    target_support: Mapping[int, Sequence[torch.Tensor]],
    device: torch.device,
    cached_official_raw: np.ndarray | None = None,
    nearest_observer: Callable | None = None,
    geometry_observer: Callable | None = None,
) -> FewshotObservation:
    """Extract the query once, then scan Official and both eight-image pools."""
    tensor = image_tensor(image).to(device=device)[None]
    # Query preprocessing stays on the frozen Exact SuperADD path. Only the
    # support-bank extractor is deliberately fixed to factor=1.
    processed = model.preprocessing(tensor)
    pending_items = _extract_processed_batch(
        model, processed, device=device, query_masks=[None],
        include_identity_descriptors=False,
    )
    if len(pending_items) != 1:
        raise RuntimeError("few-shot observation expected one query image")
    pending = pending_items[0]
    grid = tuple(int(value) for value in pending.grid)
    if geometry_observer is not None:
        positions = pending.positions.copy()
        positions.flags.writeable = False
        geometry_observer(grid, positions, tuple(int(v) for v in processed.shape[-2:]))
    if nearest_observer is not None and cached_official_raw is not None:
        raise ValueError("nearest logging requires the same-call Source search, not cached Source distances")
    official_raw = np.empty((len(LAYERS), *grid), dtype=np.float32)
    source_raw = {shot: np.empty_like(official_raw) for shot in PREFIXES}
    target_raw = {shot: np.empty_like(official_raw) for shot in PREFIXES}
    source_only = {shot: np.empty_like(official_raw) for shot in PREFIXES} if source_support is not None else {}
    target_only = {shot: np.empty_like(official_raw) for shot in PREFIXES}

    cached = None if cached_official_raw is None else np.asarray(cached_official_raw, np.float32)
    if cached is not None and cached.shape != official_raw.shape:
        raise RuntimeError("external baseline raw cache has different patch geometry")
    for offset, layer in enumerate(LAYERS):
        query = pending.queries[layer]
        if source_support is None:
            if cached is None:
                distance, indices = _exact_top1_chunked(query, official_banks[layer])
                off = distance.numpy().astype(np.float32) / float(int(query.shape[1]))
                _notify_nearest(nearest_observer, "source", layer, 0, off, indices)
            else:
                off = cached[offset].reshape(-1)[pending.positions]
            source_support_only = {}
        elif cached is None:
            off, _source_union, source_support_only = exact_prefix_minima(
                query, official_banks[layer], source_support[layer],
                nearest_observer=nearest_observer, layer=layer,
            )
        else:
            off = cached[offset].reshape(-1)
            source_support_only = _support_prefix_minima(
                query, source_support[layer],
                nearest_observer=nearest_observer, pool="source_extra", layer=layer,
            )
        # The Official distance is shared by both branches.
        target_support_only = _support_prefix_minima(
            query, target_support[layer], nearest_observer=nearest_observer, layer=layer,
        )
        baseline = cached[offset].reshape(-1) if cached is not None else off
        official_raw[offset].reshape(-1)[pending.positions] = baseline
        for shot in PREFIXES:
            if source_support is not None:
                source_only[shot][offset].reshape(-1)[pending.positions] = source_support_only[shot]
            target_only[shot][offset].reshape(-1)[pending.positions] = target_support_only[shot]
            source_raw[shot][offset].reshape(-1)[pending.positions] = (
                np.minimum(baseline, source_support_only[shot]) if source_support is not None else baseline
            )
            target_raw[shot][offset].reshape(-1)[pending.positions] = np.minimum(
                baseline, target_support_only[shot],
            )
    pending.queries.clear()
    del tensor, processed, pending_items
    values = [official_raw, *source_raw.values(), *target_raw.values()]
    if any(not np.isfinite(value).all() for value in values):
        raise RuntimeError("few-shot Exact 1-NN produced non-finite distances")
    return FewshotObservation(
        official_raw, source_raw, target_raw, source_only, target_only, grid,
    )


def signed_reduction_by_image(
    baseline: np.ndarray,
    adapted: np.ndarray,
    scales: np.ndarray,
    *,
    labels: Sequence[int],
    defect_masks: Sequence[np.ndarray | None],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return layer-wise R_N, R_A and G using image-equal aggregation."""
    base = np.asarray(baseline, dtype=np.float32)
    new = np.asarray(adapted, dtype=np.float32)
    scale = np.asarray(scales, dtype=np.float32)
    if base.shape != new.shape or base.ndim != 4 or base.shape[1] != len(LAYERS):
        raise ValueError("mechanism inputs must be aligned [N,4,H,W]")
    if len(labels) != base.shape[0] or len(defect_masks) != base.shape[0]:
        raise ValueError("mechanism labels/masks are not sample-aligned")
    reduction = (base - new) / (scale[None, :, None, None] + 1e-8)
    normal_values: list[np.ndarray] = []
    anomaly_values: list[np.ndarray] = []
    for index, label in enumerate(labels):
        if int(label) == 0:
            normal_values.append(reduction[index].mean(axis=(1, 2)))
        else:
            mask = defect_masks[index]
            if mask is None:
                raise ValueError("anomaly mechanism audit requires a defect patch mask")
            selected = np.asarray(mask, dtype=bool)
            if selected.shape != tuple(base.shape[2:]) or not selected.any():
                raise ValueError("invalid defect patch mask")
            anomaly_values.append(reduction[index, :, selected].mean(axis=1))
    if not normal_values or not anomaly_values:
        raise ValueError("mechanism audit needs both normal and anomaly images")
    rn = np.mean(normal_values, axis=0)
    ra = np.mean(anomaly_values, axis=0)
    return rn, ra, rn - ra
