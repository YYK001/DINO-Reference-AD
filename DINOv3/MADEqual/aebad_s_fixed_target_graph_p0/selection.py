"""Only held-out NORMAL support scores select configurations or damping."""
import numpy as np
from .protocol import CONFIGS, config_id


def choose(scores,empty,complexities,*,witness,atol=1e-5):
    """Highest-two CH1 mean; 95% gain, then fold-average complexity.

    Empty is legal and wins whenever the best gain is not greater than the
    frozen CH1 numerical tolerance. No anomaly scores/GT enter this API.
    """
    empty=np.asarray(empty,np.float64)
    if empty.shape!=(8,) or not np.isfinite(empty).all(): raise ValueError('eight normal LOO scores required')
    j_empty=float(np.sort(empty)[-2:].mean());records=[]
    for c in CONFIGS:
        if (c[2] is not None)!=witness: continue
        name=config_id(c);values=np.asarray(scores[name],np.float64)
        costs=np.asarray(complexities[name],np.float64)
        if values.shape!=(8,) or costs.shape!=(8,2) or not np.isfinite(values).all() or not np.isfinite(costs).all():
            raise ValueError('incomplete normal score/complexity grid')
        j=float(np.sort(values)[-2:].mean())
        records.append(dict(config=list(c),config_id=name,J=j,mean=float(values.mean()),
            per_image_scores=values.tolist(),gain=j_empty-j,
            length_per_patch=float(costs[:,0].mean()),edges_per_patch=float(costs[:,1].mean())))
    best=max((r['gain'] for r in records),default=0.)
    eligible=[r for r in records if r['gain']>=.95*best] if best>atol else []
    selected=min(eligible,key=lambda r:(r['length_per_patch'],r['edges_per_patch'],r['config'][0],
                                      r['config'][1],r['config'][2] or 0.)) if eligible else None
    return dict(selected=selected,J_empty=j_empty,best_gain=best,numerical_atol=atol,
        empty_selected=selected is None,reason='no_gain_beyond_fixed_numerical_error' if selected is None else '95pct_gain_complexity',
        candidates=records,normal_only=True,GT_used=False)


def matched_coefficient(union,witness,paired):
    """Per-layer, per-image means summed over eight LOO images (image equal)."""
    if not len(union)==len(witness)==len(paired): raise ValueError('matched normal images required')
    numerator=denominator=0.
    for u,w,g in zip(union,witness,paired):
        u,w,g=(np.asarray(a,np.float32).astype(np.float64) for a in (u,w,g))
        if u.shape!=w.shape or w.shape!=g.shape or not all(np.isfinite(a).all() for a in (u,w,g)):
            raise ValueError('invalid damping distances')
        tolerance=1e-7+1e-6*np.abs(u)
        if np.any(u-w < -tolerance) or np.any((u-w)-(u-g)>tolerance):
            raise ValueError('G1 subset G0 reduction monotonicity failed')
        numerator+=float(np.maximum(u-w,0).mean());denominator+=float(np.maximum(u-g,0).mean())
    coefficient=numerator/denominator if denominator else 0.
    if not 0<=coefficient<=1+1e-12: raise ValueError('normal damping outside [0,1]')
    return min(coefficient,1.),dict(numerator=numerator,denominator=denominator,
        aggregation='sum of image means; distances Euclidean /1280',zero_denominator=denominator==0)


def controls(union,paired,witness,lam):
    u,g,w=(np.asarray(a,np.float32) for a in (union,paired,witness))
    if not u.shape==g.shape==w.shape or not all(np.isfinite(a).all() for a in (u,g,w)):
        raise ValueError('aligned finite control arrays required')
    delta=u.astype(np.float64)-g;main=u.astype(np.float64)-w;tol=1e-7+1e-6*np.abs(u)
    if np.any(delta < -tol) or np.any(main < -tol) or np.any(main>delta+tol):
        raise ValueError('graph subset / UNION inclusion failed')
    delta=np.maximum(delta,0);main=np.maximum(main,0)
    if not 0<=lam<=1: raise ValueError('frozen lambda outside [0,1]')
    denominator=float(delta.sum());beta=float(main.sum()/denominator) if denominator else 0.
    if not 0<=beta<=1+1e-12: raise ValueError('image matching beta outside [0,1]')
    beta=min(beta,1.)
    normal=(u.astype(np.float64)-lam*delta).astype(np.float32)
    image=(u.astype(np.float64)-beta*delta).astype(np.float32)
    actual=float((u.astype(np.float64)-image).sum());requested=float(main.sum())
    bound=float(np.spacing(u).astype(np.float64).sum())+1e-12
    if abs(actual-requested)>bound: raise FloatingPointError('image match exceeds FP32 rounding bound')
    return normal,image,dict(lambda_normal=lam,beta_image=beta,paired_total=float(delta.sum()),
        witness_total=requested,image_control_total=actual,rounding_bound=bound,
        G0_zero_reduction_positions=int((delta==0).sum()),
        G1_reduction_where_G0_zero=int(((delta==0)&(main>tol)).sum()))
