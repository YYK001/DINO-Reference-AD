from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy import ndimage

from DINOv3.MADEqual.image_granularity_calibration.scoring import fit_image_mad
from DINOv3.relation_reliability.protocol import ECDF_EPSILON, LAYERS


STATISTICS = ("GLOBAL_MASS", "LOCAL_MEAN", "CONNECTED_MASS")
SHUFFLE_SEEDS = (42, 43, 44)


@dataclass(frozen=True)
class RegionResult:
    mass: float
    area: int
    row_min: int | None
    row_max: int | None
    col_min: int | None
    col_max: int | None


def fit_region_thresholds(calibration_patch_maps: Sequence[np.ndarray]) -> np.ndarray:
    """One NumPy-linear Q0.95 threshold per category/layer."""
    if not calibration_patch_maps:
        raise ValueError("normal calibration patch maps are empty")
    arrays = [np.asarray(value, dtype=np.float32) for value in calibration_patch_maps]
    if any(value.ndim != 3 or value.shape[0] != len(LAYERS) for value in arrays):
        raise ValueError("calibration patch maps must be [4,H,W]")
    if any(not np.isfinite(value).all() for value in arrays):
        raise FloatingPointError("calibration patch maps contain NaN/Inf")
    return np.asarray([
        np.quantile(
            np.concatenate([value[offset].reshape(-1) for value in arrays]),
            0.95,
            method="linear",
        )
        for offset in range(len(LAYERS))
    ], dtype=np.float32)


def fixed_permutation(shape: tuple[int, int], seed: int) -> np.ndarray:
    if len(shape) != 2 or min(shape) <= 0 or seed not in SHUFFLE_SEEDS:
        raise ValueError("shuffle requires a valid patch grid and frozen seed")
    # Stable integer namespace; no Python hash() and no call-order state.
    return np.random.default_rng(np.random.SeedSequence([seed, shape[0], shape[1]])).permutation(
        shape[0] * shape[1]
    )


def shuffle_layers(layer_maps: np.ndarray, seed: int) -> np.ndarray:
    values = np.asarray(layer_maps, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != len(LAYERS):
        raise ValueError("layer maps must be [4,H,W]")
    permutation = fixed_permutation(values.shape[1:], seed)
    return values.reshape(len(LAYERS), -1)[:, permutation].reshape(values.shape)


def global_mass(patch_map: np.ndarray, threshold: float) -> float:
    values = np.asarray(patch_map, dtype=np.float32)
    return float(np.maximum(values - np.float32(threshold), 0.0).sum(dtype=np.float64))


def local_mean(patch_map: np.ndarray) -> float:
    values = np.asarray(patch_map, dtype=np.float32)
    if values.ndim != 2 or min(values.shape) < 3 or not np.isfinite(values).all():
        raise ValueError("LOCAL_MEAN needs a finite grid of at least 3x3")
    # valid 3x3, stride 1, without padding
    sums = ndimage.uniform_filter(values, size=3, mode="constant", cval=0.0) * 9.0
    valid = sums[1:-1, 1:-1] / 9.0
    return float(valid.max())


def connected_mass(patch_map: np.ndarray, threshold: float) -> RegionResult:
    values = np.asarray(patch_map, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("CONNECTED_MASS needs one finite 2D patch grid")
    active = values > np.float32(threshold)
    labels, count = ndimage.label(active, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return RegionResult(0.0, 0, None, None, None, None)
    excess = np.maximum(values - np.float32(threshold), 0.0)
    masses = np.asarray(ndimage.sum(excess, labels, index=np.arange(1, count + 1)), dtype=np.float64)
    winner = int(np.argmax(masses)) + 1
    rows, cols = np.nonzero(labels == winner)
    return RegionResult(
        mass=float(masses[winner - 1]), area=int(rows.size),
        row_min=int(rows.min()), row_max=int(rows.max()),
        col_min=int(cols.min()), col_max=int(cols.max()),
    )


def layer_statistics(
    layer_maps: np.ndarray,
    thresholds: np.ndarray,
    statistics: Sequence[str] = STATISTICS,
) -> tuple[dict[str, np.ndarray], list[RegionResult]]:
    values = np.asarray(layer_maps, dtype=np.float32)
    thresholds = np.asarray(thresholds, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != len(LAYERS) or thresholds.shape != (len(LAYERS),):
        raise ValueError("statistics require [4,H,W] maps and four thresholds")
    unknown = set(statistics) - set(STATISTICS)
    if unknown:
        raise ValueError(f"unknown statistics: {sorted(unknown)}")
    regions = (
        [connected_mass(values[o], thresholds[o]) for o in range(len(LAYERS))]
        if "CONNECTED_MASS" in statistics else []
    )
    output: dict[str, np.ndarray] = {}
    if "GLOBAL_MASS" in statistics:
        output["GLOBAL_MASS"] = np.asarray(
            [global_mass(values[o], thresholds[o]) for o in range(len(LAYERS))], np.float32,
        )
    if "LOCAL_MEAN" in statistics:
        output["LOCAL_MEAN"] = np.asarray(
            [local_mean(values[o]) for o in range(len(LAYERS))], np.float32,
        )
    if "CONNECTED_MASS" in statistics:
        output["CONNECTED_MASS"] = np.asarray(
            [region.mass for region in regions], np.float32,
        )
    return output, regions


def fit_statistic_image_mad(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    samples = np.asarray(values, dtype=np.float32)
    medians, scales = fit_image_mad(samples)
    raw_mad = np.median(np.abs(samples - medians[None, :]), axis=0).astype(np.float32)
    floor = scales <= np.float32(ECDF_EPSILON)
    return medians, scales, raw_mad, floor


def calibrated_branch(statistics: np.ndarray, medians: np.ndarray, scales: np.ndarray) -> float:
    values = np.asarray(statistics, dtype=np.float32)
    output = (values - np.asarray(medians, np.float32)) / np.asarray(scales, np.float32)
    if output.shape != (len(LAYERS),) or not np.isfinite(output).all():
        raise FloatingPointError("image MAD branch is invalid")
    return float(output.mean(dtype=np.float32))


def fused_score(base_score: float, branch_score: float) -> float:
    result = 0.5 * float(base_score) + 0.5 * float(branch_score)
    if not np.isfinite(result):
        raise FloatingPointError("fused image score is invalid")
    return result
