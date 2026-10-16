"""One-forward scan of only the eight new synthetic support images."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from DINOv3.MADEqual.robustad_target_fewshot_p0.scoring import _support_prefix_minima
from DINOv3.relation_reliability.protocol import LAYERS
from DINOv3.relation_reliability.superadd_host import (
    _extract_processed_batch,
    image_tensor,
)


PREFIXES = (1, 2, 4, 8)


def union_with_official(
    official_raw: np.ndarray,
    support_only: Mapping[int, np.ndarray],
) -> dict[int, np.ndarray]:
    baseline = np.asarray(official_raw, dtype=np.float32)
    output: dict[int, np.ndarray] = {}
    for shot in PREFIXES:
        support = np.asarray(support_only[shot], dtype=np.float32)
        if support.shape != baseline.shape:
            raise ValueError("Official and synthetic raw tensors are not aligned")
        output[shot] = np.minimum(baseline, support)
    return output


@dataclass(frozen=True)
class SyntheticObservation:
    synthetic_raw: Mapping[int, np.ndarray]
    support_only: Mapping[int, np.ndarray]
    grid: tuple[int, int]


@torch.inference_mode()
def observe_synthetic_pil(
    model: Any,
    image: Image.Image,
    *,
    synthetic_support: Mapping[int, Sequence[torch.Tensor]],
    cached_official_raw: np.ndarray,
    device: torch.device,
) -> SyntheticObservation:
    """Extract the query once; never scan or load the Official 100K bank."""
    tensor = image_tensor(image).to(device=device)[None]
    processed = model.preprocessing(tensor)
    pending_items = _extract_processed_batch(
        model, processed, device=device, query_masks=[None],
        include_identity_descriptors=False,
    )
    if len(pending_items) != 1:
        raise RuntimeError("synthetic few-shot expected one query image")
    pending = pending_items[0]
    grid = tuple(int(value) for value in pending.grid)
    baseline = np.asarray(cached_official_raw, dtype=np.float32)
    expected_shape = (len(LAYERS), *grid)
    if baseline.shape != expected_shape or not np.isfinite(baseline).all():
        raise RuntimeError("base Official raw has incompatible query geometry")
    support_only = {shot: np.empty(expected_shape, np.float32) for shot in PREFIXES}
    for offset, layer in enumerate(LAYERS):
        minima = _support_prefix_minima(pending.queries[layer], synthetic_support[layer])
        for shot in PREFIXES:
            support_only[shot][offset].reshape(-1)[pending.positions] = minima[shot]
    pending.queries.clear()
    del tensor, processed, pending_items
    if any(not value.size or not np.isfinite(value).all() for value in support_only.values()):
        raise RuntimeError("synthetic support scan produced non-finite distances")
    return SyntheticObservation(
        synthetic_raw=union_with_official(baseline, support_only),
        support_only=support_only,
        grid=grid,
    )


__all__ = ["PREFIXES", "SyntheticObservation", "observe_synthetic_pil", "union_with_official"]

