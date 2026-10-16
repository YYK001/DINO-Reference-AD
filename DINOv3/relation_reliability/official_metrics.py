from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
import torch


def _install_numpy_adeval_compatibility() -> None:
    """Provide ADEval 1.1.0's legacy name on NumPy versions that removed it."""
    if hasattr(np, "trapz"):
        return
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:
        raise RuntimeError(
            "ADEval requires trapezoidal integration, but this NumPy exposes "
            "neither np.trapz nor np.trapezoid"
        )
    np.trapz = trapezoid  # type: ignore[attr-defined]


def validate_official_metric_backend() -> None:
    _install_numpy_adeval_compatibility()
    if not torch.cuda.is_available():
        raise RuntimeError("official ADEval AUPRO requires CUDA")
    try:
        from adeval import EvalAccumulatorCuda  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "official ADEval AUPRO backend is unavailable; install the frozen "
            "adeval dependency before starting the experiment"
        ) from exc


def official_aupro_adeval(
    masks: Sequence[np.ndarray],
    maps: Sequence[np.ndarray],
    *,
    nstrips: int = 200,
) -> float:
    """Compute the established CUDA ADEval AUPRO used by project baselines."""
    _install_numpy_adeval_compatibility()
    if len(masks) != len(maps) or not masks:
        raise ValueError("AUPRO masks/maps must be non-empty and aligned")
    if not torch.cuda.is_available():
        raise RuntimeError("official ADEval AUPRO requires CUDA")

    predictions = [np.asarray(value, dtype=np.float32) for value in maps]
    targets = [np.asarray(value, dtype=np.uint8) for value in masks]
    for index, (prediction, target) in enumerate(zip(predictions, targets)):
        if prediction.ndim != 2 or prediction.shape != target.shape:
            raise ValueError(
                f"AUPRO map/mask shape mismatch at image {index}: "
                f"{prediction.shape} vs {target.shape}"
            )
        if not np.isfinite(prediction).all():
            raise FloatingPointError(f"non-finite AUPRO prediction at image {index}")

    score_min = min(float(value.min()) for value in predictions)
    score_max = max(float(value.max()) for value in predictions)
    if score_max <= score_min:
        raise ValueError("AUPRO requires a non-constant score range")

    from adeval import EvalAccumulatorCuda

    deterministic_enabled = torch.are_deterministic_algorithms_enabled()
    warn_only_enabled = (
        torch.is_deterministic_algorithms_warn_only_enabled()
        if hasattr(torch, "is_deterministic_algorithms_warn_only_enabled")
        else False
    )
    grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, prediction in enumerate(predictions):
        grouped[tuple(int(value) for value in prediction.shape)].append(index)

    try:
        torch.use_deterministic_algorithms(False)
        accumulator = EvalAccumulatorCuda(
            score_min, score_max, score_min, score_max,
            skip_pixel_aupro=False, nstrips=int(nstrips),
        )
        for indices in grouped.values():
            prediction_batch = np.stack([predictions[index] for index in indices])
            target_batch = np.stack([targets[index] for index in indices])
            accumulator.add_anomap_batch(
                torch.from_numpy(prediction_batch).cuda(non_blocking=True),
                torch.from_numpy(target_batch).cuda(non_blocking=True),
            )
        value = float(accumulator.summary()["p_aupro"])
    finally:
        torch.use_deterministic_algorithms(
            deterministic_enabled, warn_only=warn_only_enabled,
        )
    if not np.isfinite(value):
        raise FloatingPointError("ADEval returned non-finite AUPRO")
    return value