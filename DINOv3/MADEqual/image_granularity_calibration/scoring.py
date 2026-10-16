from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from DINOv3.relation_oracle_cache.extract import mad_calibrated, mad_equal
from DINOv3.relation_reliability.protocol import ECDF_EPSILON, LAYERS
from DINOv3.relation_reliability.superadd_host import interpolate_score


@dataclass(frozen=True)
class ImageGranularityScores:
    current_mad: float
    order_only: float
    granularity_mad: float
    layer_maxima: np.ndarray


def interpolated_layer_maxima(
    raw_distances: np.ndarray,
    medians: np.ndarray,
    scales: np.ndarray,
    output_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-layer interpolated MAD maps and their image maxima."""
    calibrated = mad_calibrated(raw_distances, medians, scales)
    layer_maps = np.stack(
        [
            np.asarray(interpolate_score(calibrated[offset], output_shape), dtype=np.float32)
            for offset in range(len(LAYERS))
        ],
        axis=0,
    )
    if layer_maps.shape != (len(LAYERS), *output_shape):
        raise RuntimeError(f"interpolated MAD layer shape is invalid: {layer_maps.shape}")
    if not np.isfinite(layer_maps).all():
        raise FloatingPointError("interpolated MAD layers contain NaN/Inf")
    maxima = layer_maps.reshape(len(LAYERS), -1).max(axis=1)
    return layer_maps, maxima.astype(np.float32, copy=False)


def fit_image_mad(
    calibration_layer_maxima: np.ndarray,
    *,
    epsilon: float = ECDF_EPSILON,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the frozen median and 1.4826*MAD to calibration-image maxima."""
    values = np.asarray(calibration_layer_maxima, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(LAYERS) or values.shape[0] == 0:
        raise ValueError(f"calibration maxima must be [image,4], got {values.shape}")
    if not np.isfinite(values).all():
        raise FloatingPointError("calibration image maxima contain NaN/Inf")
    medians = np.median(values, axis=0).astype(np.float32)
    raw_mad = np.median(np.abs(values - medians[None, :]), axis=0)
    scales = np.maximum(1.4826 * raw_mad, float(epsilon)).astype(np.float32)
    if not np.isfinite(scales).all() or (scales <= 0).any():
        raise FloatingPointError("image-level MAD scales must be finite and positive")
    return medians, scales


def score_image(
    raw_distances: np.ndarray,
    patch_medians: np.ndarray,
    patch_scales: np.ndarray,
    output_shape: tuple[int, int],
    image_medians: Sequence[float],
    image_scales: Sequence[float],
) -> ImageGranularityScores:
    layer_maps, maxima = interpolated_layer_maxima(
        raw_distances, patch_medians, patch_scales, output_shape,
    )
    image_medians_array = np.asarray(image_medians, dtype=np.float32)
    image_scales_array = np.asarray(image_scales, dtype=np.float32)
    if image_medians_array.shape != (len(LAYERS),):
        raise ValueError("image medians must contain four layers")
    if (
        image_scales_array.shape != (len(LAYERS),)
        or not np.isfinite(image_scales_array).all()
        or (image_scales_array <= 0).any()
    ):
        raise ValueError("image scales must contain four finite positive values")

    # The existing fused-then-interpolated MAD map is retained for CURRENT_MAD.
    # Bilinear interpolation is linear, so it must match the mean of the four
    # individually interpolated calibrated maps used by ORDER_ONLY.
    existing_current = np.asarray(
        interpolate_score(
            mad_equal(raw_distances, patch_medians, patch_scales), output_shape,
        ),
        dtype=np.float32,
    )
    layer_mean = layer_maps.mean(axis=0, dtype=np.float32)
    if not np.allclose(existing_current, layer_mean, atol=2e-6, rtol=2e-6):
        raise RuntimeError("fused and layerwise-interpolated CURRENT_MAD maps differ")

    current = float(existing_current.max())
    order = float(maxima.mean(dtype=np.float32))
    granularity = float(
        ((maxima - image_medians_array) / image_scales_array).mean(dtype=np.float32)
    )
    if not np.isfinite([current, order, granularity]).all():
        raise FloatingPointError("image-granularity scores contain NaN/Inf")
    return ImageGranularityScores(current, order, granularity, maxima)
