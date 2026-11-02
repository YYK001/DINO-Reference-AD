"""CPU FP32, two streaming passes over sources; no N x 9 x 1280 residual cache."""
import numpy as np
from .protocol import A, B, C
from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import statistics, ch1_score
from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.calibration import z_values
from DINOv3.MADEqual.aebad_s_target_loo_layer_weight_p0.numerics import weighted


def order(positions, grid):
    p = np.asarray(positions)
    n = int(np.prod(grid))
    if p.dtype != np.int64 or p.shape != (n,) or not np.array_equal(np.sort(p), np.arange(n)):
        raise ValueError('one complete unique physical grid required; stitched overlap cannot be counted twice')
    return np.argsort(p)


def norm(x):
    return np.sqrt(np.sum(x*x, axis=-1, dtype=np.float32)).astype(np.float32)


def translate(q, banks, indices, distances, *, chunk=1024, delta_override=None):
    """q and indices are already in canonical physical order.

    Distances are the ORIGINAL Exact /1280 outputs, used for its original UNION
    source decision (Source first, then support order on ties). Candidate norms
    use original feature vectors. Means always precede /1280.
    """
    q = np.asarray(q)
    ids = np.asarray(indices);distances = np.asarray(distances)
    n, dim = q.shape
    k = len(banks)
    if q.dtype != np.float32 or not n or not np.isfinite(q).all() or chunk < 1:
        raise ValueError('finite nonempty FP32 query and positive chunk required')
    if ids.shape != (k,n) or ids.dtype != np.int64 or distances.shape != (k,n) or distances.dtype != np.float32 or not np.isfinite(distances).all() or (distances<0).any():
        raise ValueError('per-source original indices and scaled distances required')
    for s, bank in enumerate(banks):
        if bank.dtype != np.float32 or bank.ndim != 2 or bank.shape[1] != dim or (ids[s]<0).any() or (ids[s]>=len(bank)).any():
            raise ValueError('invalid same-identity memory/index rows')
    winner = distances.argmin(0).astype(np.int8)
    # Source-wise FP32 means have deterministic physical order. At most q plus
    # two N x D buffers; chunk size does not alter the reduction order.
    union_r = np.empty_like(q);means = np.empty((k,dim),np.float32)
    residual_means = np.empty(k,np.float32)
    for s, bank in enumerate(banks):
        r = np.empty_like(q)
        for start in range(0,n,chunk):
            end = min(n,start+chunk);r[start:end] = q[start:end]-np.asarray(bank[ids[s,start:end]],np.float32)
        if not np.isfinite(r).all():raise ValueError('nonfinite original residual')
        means[s] = r.mean(0,dtype=np.float32)
        residual_means[s] = norm(r).mean(dtype=np.float32)
        keep = winner==s;union_r[keep] = r[keep]
    delta = union_r.mean(0,dtype=np.float32)
    if delta_override is not None:
        delta,means = (np.asarray(v,np.float32) for v in delta_override)
        if delta.shape!=(dim,) or means.shape!=(k,dim):raise ValueError('invalid explicit test correction')
    original = distances.min(0)
    result = {A: norm(union_r-delta)/np.float32(1280), B: np.full(n,np.inf,np.float32), C: np.full(n,np.inf,np.float32)}
    new_winner = {B: np.zeros(n,np.int8), C: np.zeros(n,np.int8)}
    for s,bank in enumerate(banks):
        for start in range(0,n,chunk):
            end=min(n,start+chunk);r=q[start:end]-np.asarray(bank[ids[s,start:end]],np.float32)
            for b,shift in ((B,delta),(C,means[s])):
                v=norm(r-shift)/np.float32(1280);keep=v<result[b][start:end]
                result[b][start:end][keep]=v[keep];new_winner[b][start:end][keep]=s
    if np.any(result[B]>result[A]+np.float32(1e-7)):
        raise ValueError('same-candidate B <= A invariant failed')
    umean=norm(union_r).mean(dtype=np.float32)
    diagnostic=dict(delta_U_norm=float(norm(delta)),mean_union_residual_norm=float(umean),
                    delta_U_ratio=float(norm(delta)/umean) if umean else None,
                    zero_union_denominator=bool(umean==0),
                    sources=[dict(source=s,delta_norm=float(norm(means[s])),mean_residual_norm=float(residual_means[s]),
                                  ratio=float(norm(means[s])/residual_means[s]) if residual_means[s] else None,
                                  zero_denominator=bool(residual_means[s]==0)) for s in range(k)],
                    branches={b:dict(mean_distance_before=float(original.mean(dtype=np.float32)),
                                     mean_distance_after=float(v.mean(dtype=np.float32)),
                                     mean_signed_change=float((v-original).mean(dtype=np.float32)),
                                     increased_positions=int(np.sum(v>original)),
                                     source_switch_fraction=0. if b==A else float(np.mean(new_winner[b]!=winner))) for b,v in result.items()})
    return result,dict(delta_U=delta,delta_ref=means,union_source=winner,
                       selected_B=new_winner[B],selected_C=new_winner[C]),diagnostic


def score(raw, positions, grid, shape, patch, image_state, source_weights):
    # CH1 uses ORIGINAL per-layer statistics, independently of u-weighted map.
    _,p,q=statistics(raw,positions,grid,shape,np.asarray(patch['medians'],np.float32),np.asarray(patch['scales'],np.float32))
    image,_=weighted(z_values(raw,patch),positions,grid,shape,source_weights)
    return image,dict(CH1=ch1_score(p,q,image_state),WEIGHTED_MAP_MAX=float(image.max()))
