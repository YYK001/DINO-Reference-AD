"""Parent multiplication order; saved solver only, no transport or refitting."""
from __future__ import annotations
import numpy as np


def project(query, constant):
    q = np.asarray(query)
    p, b = constant['prototypes'], constant['factor']
    if (q.ndim != 2 or q.dtype != np.float32 or not len(q) or p.dtype != np.float32
            or p.ndim != 2 or q.shape[1] != p.shape[1] or b.shape != p.shape
            or b.dtype != np.float64 or not all(np.isfinite(v).all() for v in (q,p,b))):
        raise ValueError('finite original float32 query/prototypes and saved float64 factor required')
    x = q.astype(np.float64)/1280
    m = p.astype(np.float64)/1280
    # Do not associate as X @ (M.T @ B): retain the parent's actual order.
    a = (x @ m.T) @ b
    raw = np.ascontiguousarray(a*1280, dtype=np.float32)
    ideal_residual = np.linalg.norm(x-a, axis=1)
    cast_residual = np.linalg.norm(q.astype(np.float64)-raw.astype(np.float64), axis=1)/1280
    normal = (a-x) @ m.T
    relative = float(np.linalg.norm(normal)/max(np.linalg.norm(x@m.T), np.finfo(float).tiny))
    if not np.isfinite(raw).all() or not np.isfinite(relative) or relative > 1e-8:
        raise ValueError('initial projection normal-equation residual exceeds parent 1e-8')
    return raw, ideal_residual, cast_residual, dict(
        OT_called=False, query_rows=len(q), projection_rows=len(raw),
        prototype_identity=constant.get('prototype_identity'),
        normal_equation_relative_residual=relative,
        cast_residual_max_abs_change=float(np.abs(cast_residual-ideal_residual).max()),
        arithmetic='float64 ((Q/1280) @ (P/1280).T) @ saved_B; float32(A*1280)')


def difference(actual, reference):
    """Signed actual minus reference; units supplied by the caller."""
    d = np.asarray(actual, np.float64)-np.asarray(reference, np.float64)
    if d.size == 0 or not np.isfinite(d).all():
        raise ValueError('nonempty finite difference required')
    result = dict(count=int(d.size), max_abs=float(np.abs(d).max()),
                  mean_abs=float(np.abs(d).mean()), mean_signed=float(d.mean()),
                  min_signed=float(d.min()), max_signed=float(d.max()))
    for percentile in (0,1,5,50,95,99,100):
        result[f'signed_p{percentile}'] = float(np.percentile(d,percentile))
        result[f'abs_p{percentile}'] = float(np.percentile(np.abs(d),percentile))
    return result
