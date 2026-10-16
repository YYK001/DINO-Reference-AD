"""NumPy-only three-side geometry. Cache distances are ALREADY Euclidean/1280."""
from __future__ import annotations

import numpy as np

LAYERS = (7, 15, 23, 31)
BRANCHES = ("K8_UNION", "K8_MIDPOINT", "K8_SEGMENT")
DIMENSION = 1280
DIST_ATOL = 1e-7
DIST_RTOL = 1e-6
SCORE_ATOL = 1e-5
ENDPOINT_CHUNK = 1024  # Memory bound, not a reference-selection parameter.
EPS64 = np.finfo(np.float64).eps
GAMMA = (DIMENSION + 16) * EPS64 / (1 - (DIMENSION + 16) * EPS64)


class NumericalError(ValueError):
    def __init__(self, message: str, diagnostics: dict):
        super().__init__(message)
        self.diagnostics = diagnostics


def bounded_distances(a: np.ndarray, b: np.ndarray, c2: np.ndarray,
                      source: np.ndarray) -> tuple[dict, dict]:
    """Float64 arithmetic does not recover precision lost in stored FP32 a/b."""
    a, b, c2, source = [np.asarray(v, np.float64) for v in (a, b, c2, source)]
    if (a.ndim != 1 or any(v.shape != a.shape for v in (b, c2, source))
            or any(not np.isfinite(v).all() or (v < 0).any() for v in (a, b, c2, source))
            or (a > b).any()):
        raise ValueError("invalid finite nonnegative ordered sides/source")
    c = np.sqrt(c2)
    ea, eb = DIST_ATOL + DIST_RTOL * a, DIST_ATOL + DIST_RTOL * b
    ec = GAMMA * c
    tolerance = ea + eb + ec + 16 * EPS64 * (a + b + c)
    violation = np.maximum(np.abs(a - b) - c, c - a - b)
    diag = dict(patch_count=a.size, triangle_failed_count=int((violation > tolerance).sum()),
                triangle_max_excess=float(np.max(violation - tolerance, initial=0)),
                degenerate_count=int((c2 == 0).sum()), correction_count=0,
                maximum_squared_correction=0., endpoint_identity_max_error=0.)
    if diag["triangle_failed_count"]:
        raise NumericalError("incompatible cached a/b and feature endpoint length", diag)
    aa, bb = a * a, b * b
    # Algebraically a²+c²-b²; keep c² when a==b and endpoints are very close.
    numerator = (a - b) * (a + b) + c2
    alpha = np.zeros_like(a)
    active = c2 > 0
    alpha[active & (numerator >= 2 * c2)] = 1.
    interior = active & (numerator > 0) & (numerator < 2 * c2)
    np.divide(numerator, 2 * c2, out=alpha, where=interior)

    corrections = []
    def square(t):
        v1, v2, v3 = (1 - t) * aa, t * bb, t * (1 - t) * c2
        value = v1 + v2 - v3
        # Cache-error propagation at fixed t plus bounded FP64 arithmetic.
        bound = ((1 - t) * (2 * a * ea + ea * ea)
                 + t * (2 * b * eb + eb * eb) + t * (1 - t) * GAMMA * c2
                 + 16 * EPS64 * (np.abs(v1) + np.abs(v2) + np.abs(v3)))
        bad = active & (value < -bound)
        if bad.any():
            diag.update(negative_square_failed_count=int(bad.sum()),
                        minimum_squared_distance=float(value.min()),
                        maximum_negative_excess=float((-value - bound)[bad].max()))
            raise NumericalError("negative squared distance exceeds declared error bound", diag)
        corrected = active & (value < 0)
        corrections.append(np.where(corrected, -value, 0.))
        diag["correction_count"] += int(corrected.sum())
        diag["maximum_squared_correction"] = max(diag["maximum_squared_correction"],
                                                  float(np.max(-value[corrected], initial=0)))
        value = value.copy()
        value[corrected] = 0.
        # Exact duplicate endpoints fall back to UNION, never to fabricated zero.
        value[~active] = aa[~active]
        return value

    # Explicit endpoint identities; evaluated in the same expression as f(t).
    f0 = (1 - 0.) * aa + 0. * bb - 0. * c2
    f1 = (1 - 1.) * aa + 1. * bb - 0. * c2
    diag["endpoint_identity_max_error"] = float(max(np.max(abs(f0-aa), initial=0),
                                                     np.max(abs(f1-bb), initial=0)))
    union = np.minimum(source, a)
    midpoint = np.minimum(union, np.sqrt(square(np.full_like(a, .5))))
    segment = np.minimum(union, np.sqrt(square(alpha)))
    midpoint[~active] = segment[~active] = union[~active]
    monotone_tolerance = DIST_ATOL + DIST_RTOL * union
    if ((segment > midpoint + monotone_tolerance).any() or (midpoint > union).any()):
        raise NumericalError("segment <= midpoint <= union failed", diag)
    result = dict(K8_UNION=union.astype(np.float32), K8_MIDPOINT=midpoint.astype(np.float32),
                  K8_SEGMENT=segment.astype(np.float32), alpha=alpha, endpoint_length=c,
                  squared_correction=np.stack(corrections))
    if any(not np.isfinite(v).all() for v in result.values()):
        raise NumericalError("nonfinite bounded reference result", diag)
    return result, diag


def geometry_layer(target: np.ndarray, target_index: np.ndarray, source: np.ndarray,
                   features: list[np.ndarray], *, chunk: int = ENDPOINT_CHUNK) -> tuple[dict, dict]:
    """Select two different SUPPORTS' top-1s, stable distance/manifest order."""
    target, indices, source = np.asarray(target), np.asarray(target_index), np.asarray(source)
    if (target.ndim != 2 or indices.shape != target.shape or source.shape != target.shape[1:]
            or len(features) != len(target) or len(target) < 1 or chunk < 1
            or not np.issubdtype(indices.dtype, np.integer) or (indices < 0).any()
            or not np.isfinite(target).all() or (target < 0).any()
            or not np.isfinite(source).all() or (source < 0).any()):
        raise ValueError("invalid geometry inputs")
    for j, bank in enumerate(features):
        if (bank.ndim != 2 or bank.shape[1] != DIMENSION or not np.isfinite(bank).all()
                or np.any(indices[j] >= len(bank))):
            raise ValueError("invalid support feature dimensions/finiteness/index")
    count = target.shape[1]
    order = np.argsort(target, axis=0, kind="stable")  # row order = manifest 1..8
    union = np.minimum(source, target.min(axis=0)).astype(np.float32)
    result = dict(K8_UNION=union, K8_MIDPOINT=union.copy(), K8_SEGMENT=union.copy(),
                  alpha=np.zeros(count, np.float64), endpoint_length=np.zeros(count, np.float64),
                  squared_correction=np.zeros((2,count), np.float64),
                  endpoint_support=np.zeros((2, count), np.int8),
                  endpoint_index=np.zeros((2, count), np.int64))
    stats = dict(patch_count=count, triangle_failed_count=0, triangle_max_excess=0.,
                 degenerate_count=0, correction_count=0, maximum_squared_correction=0.,
                 endpoint_identity_max_error=0., single_support_fallback_count=0,
                 near_tie_count=0)
    if len(target) == 1:
        result['endpoint_support'][0] = 1
        result['endpoint_index'][0] = indices[0]
        stats['single_support_fallback_count'] = count
        return result, stats
    selected = order[:2]
    columns = np.arange(count)
    result['endpoint_support'][:] = selected + 1
    result['endpoint_index'][:] = indices[selected, columns]
    for start in range(0, count, chunk):
        stop = min(count, start + chunk)
        cols = columns[start:stop]
        s1, s2 = selected[:, start:stop]
        endpoints = [np.empty((len(cols), DIMENSION), np.float64) for _ in range(2)]
        for endpoint, chosen in zip(endpoints, (s1, s2), strict=True):
            for j, bank in enumerate(features):
                take = chosen == j
                endpoint[take] = bank[indices[j, cols[take]]].astype(np.float64)
        difference = endpoints[1] - endpoints[0]
        c2 = np.einsum('ij,ij->i', difference, difference) / (DIMENSION * DIMENSION)
        aa, bb = target[s1, cols], target[s2, cols]
        values, diag = bounded_distances(aa, bb, c2, source[cols])
        for key in BRANCHES + ('alpha', 'endpoint_length'):
            result[key][start:stop] = values[key]
        result['squared_correction'][:,start:stop] = values['squared_correction']
        stats['near_tie_count'] += int((bb-aa <= DIST_ATOL+DIST_RTOL*np.abs(aa)).sum())
        for key in ('triangle_failed_count','degenerate_count','correction_count'):
            stats[key] += diag[key]
        for key in ('triangle_max_excess','maximum_squared_correction','endpoint_identity_max_error'):
            stats[key] = max(stats[key], diag[key])
    if not np.array_equal(result['K8_UNION'], union):
        raise NumericalError('original union changed', stats)
    return result, stats


def scatter(compact: np.ndarray, positions: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    compact, positions = np.asarray(compact), np.asarray(positions)
    count = int(np.prod(grid))
    if (compact.shape != (4, count) or positions.dtype != np.int64
            or positions.shape != (count,) or not np.array_equal(np.sort(positions), np.arange(count))):
        raise ValueError('invalid complete compact/physical positions mapping')
    result = np.empty_like(compact)
    result[:, positions] = compact
    return result.reshape(4, *grid)
