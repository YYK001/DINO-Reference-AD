"""CPU port of image_granularity_calibration/connected_region scoring.

Keep float32 operation order, including uniform_filter * 9 / 9. Interpolation
reuses the independently validated CPU geometry implementation. Original Torch
statistics are exported during explicit recovery, and checked before evaluation.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic.numerics import (
    LAYERS, reconstruct, check,
)

SCORE_ATOL = 1e-5
DISTANCE_ATOL = 1e-7
DISTANCE_RTOL = 1e-6
BRANCHES = ('SOURCE_CURRENT', 'SOURCE_CH1', 'UNION_K8_CURRENT', 'UNION_K8_CH1')
COMPARISONS = (
    ('SOURCE_CH1', 'SOURCE_CURRENT'),
    ('UNION_K8_CH1', 'UNION_K8_CURRENT'),
    ('UNION_K8_CH1', 'SOURCE_CH1'),
)


def statistics(raw, positions, grid, shape, patch_medians, patch_scales):
    z, maps, _ = reconstruct(raw, positions, grid, shape, patch_medians, patch_scales)
    if min(grid) < 3:
        raise ValueError('valid 3x3 statistics require both native dimensions >= 3')
    peak = np.asarray([maps[layer].max() for layer in LAYERS], np.float32)
    local = np.asarray([
        ((ndimage.uniform_filter(a, size=3, mode='constant', cval=0.0) * 9.0)[1:-1, 1:-1] / 9.0).max()
        for a in z
    ], np.float32)
    # Both scoring rules share this exact localization definition. Only the
    # scalar image score changes; no second map is produced for CH1.
    return float(maps['FUSED'].max()), peak, local


def fit_image_mad(values):
    a = np.asarray(values, np.float32)
    if a.ndim != 2 or a.shape[1] != 4 or not len(a) or not np.isfinite(a).all():
        raise ValueError('image calibration must be finite [images,4]')
    median = np.median(a, axis=0).astype(np.float32)
    scale = np.maximum(1.4826 * np.median(np.abs(a - median[None, :]), axis=0), 1e-8).astype(np.float32)
    return median, scale


def ch1_score(peak, local, state):
    parts = []
    for name, value in (('peak', peak), ('local', local)):
        median = np.asarray(state[name]['medians'], np.float32)
        scale = np.asarray(state[name]['scales'], np.float32)
        a = np.asarray(value, np.float32)
        if a.shape != (4,) or median.shape != (4,) or scale.shape != (4,) or np.any(scale <= 0):
            raise ValueError('four-layer image MAD state required')
        parts.append(float(((a - median) / scale).mean(dtype=np.float32)))
    result = .5 * parts[0] + .5 * parts[1]
    if not np.isfinite(result):
        raise FloatingPointError('nonfinite CH1 score')
    return result


def pair_errors(labels, scores):
    y = np.asarray(labels)
    s = np.asarray(scores, float)
    if y.ndim != 1 or s.shape != y.shape or not np.isfinite(s).all() or set(y.tolist()) != {0, 1}:
        raise ValueError('finite binary image scores with both classes required')
    difference = s[y == 0, None] - s[None, y == 1]
    return (difference > 0).astype(float) + .5 * (difference == 0)


def image_metrics(labels, scores):
    # CPU only; average_precision_score handles tied thresholds, not trapezoids.
    from sklearn.metrics import average_precision_score
    errors = pair_errors(labels, scores)
    return dict(image_AUROC=float(1 - errors.mean()),
                image_AP=float(average_precision_score(labels, scores)))


def complete(count, checks):
    return count == 297 and bool(checks) and all(row['verified'] for row in checks)
