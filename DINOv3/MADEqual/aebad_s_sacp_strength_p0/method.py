"""Replay the parent's FP32 sqrt order and unchanged-distance z inheritance."""
import numpy as np
from .protocol import BRANCHES,MAIN,TARGET,UNIFORM,HALF,CONFIG,U
from DINOv3.MADEqual.aebad_s_sacp_p0.numerics import matched_uniform
from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.calibration import z_values,response


def distance(union,eta,strength):
    if union.dtype!=np.float32 or eta.dtype!=np.float32 or union.shape!=eta.shape:raise ValueError('FP32 raw_union/eta shape required')
    if not np.isfinite(union).all() or (union<0).any() or not np.isfinite(eta).all() or (eta<0).any() or (eta>1).any():
        raise ValueError('invalid raw_union/eta; stored eta must already be in [0,1], no cleaning')
    if strength not in (.5,1.):raise ValueError('only the fixed parent .5 replay and new 1.0 are implemented')
    return (union*np.sqrt(np.float32(1)-np.float32(strength)*eta)).astype(np.float32)


def values(arrays,strength):
    u=arrays['raw_union'];eta=arrays['eta'];result={MAIN:distance(u,eta[0],strength),TARGET:distance(u,eta[1],strength)};uniform=[];audit=[]
    for j in range(4):
        v,r=matched_uniform(u[j],result[MAIN][j]);uniform.append(v);audit.append(r)
    result[UNIFORM]=np.stack(uniform).astype(np.float32)
    return result,audit


def calibrated(raw,union,parent_union_z,patch):
    z=z_values(raw,patch);same=raw==union;z[same]=parent_union_z[same]
    return z


def comparison(checks,sample,kind,actual,expected,*,branch='',layer='',score=False):
    actual=np.asarray(actual);expected=np.asarray(expected)
    atol=CONFIG['score_atol'] if score else CONFIG['distance_atol'];rtol=0 if score else CONFIG['distance_rtol']
    good=actual.shape==expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all() and np.allclose(actual,expected,atol=atol,rtol=rtol)
    checks.append(dict(domain=sample['domain'],sample_id=sample['sample_id'],branch=branch,layer=layer,kind=kind,
        max_abs_error=float(np.max(np.abs(actual.astype(np.float64)-expected),initial=0)) if actual.shape==expected.shape else None,atol=atol,rtol=rtol,verified=bool(good)))
    if not good:raise ValueError('parent replay mismatch: '+kind+'; original tolerances retained')


def replay_half(arrays,metadata,state):
    raw,audit=values(arrays,.5);checks=[];sample=metadata['sample']
    for j,l in enumerate(CONFIG['layers']):comparison(checks,sample,'parent_half_raw_main',raw[MAIN][j],arrays['raw_main'][j],branch=HALF[MAIN],layer=l)
    for b in BRANCHES:
        z=calibrated(raw[b],arrays['raw_union'],arrays[U],state['patch'])
        for j,l in enumerate(CONFIG['layers']):comparison(checks,sample,'parent_half_z',z[j],arrays[HALF[b]][j],branch=HALF[b],layer=l)
        _,scores=response(z,arrays['positions'],metadata['grid'],metadata['output_shape'],state['image'])
        for rule in ('CH1','CURRENT'):comparison(checks,sample,'parent_half_'+rule,scores[rule],metadata['scores'][HALF[b]][rule],branch=HALF[b],score=True)
    return checks,raw
