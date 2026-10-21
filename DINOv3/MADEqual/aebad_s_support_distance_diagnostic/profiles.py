"""Native patch quantities only. No detector, threshold search or Target fitting."""
from __future__ import annotations
import numpy as np
from DINOv3.MADEqual.aebad_s_neighborhood_segment_p0.gates import LAYERS, ATOL, RTOL, scatter
from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic.numerics import reconstruct, adaptive_max

FEATURES = ('source_advantage', 'gap2', 'gap4', 'gap8', 'count_better_source',
            'count_tie_source', 'effective_support_count', *(f'loo_{j}' for j in range(1,9)), 'loo_max')


def native(source, target, segment, median, scale):
    """All inputs are original compact float32 rows, already Euclidean/1280."""
    s,t,v=map(np.asarray,(source,target,segment))
    if (s.ndim!=1 or t.shape!=(8,len(s)) or v.shape!=s.shape
            or any(a.dtype!=np.float32 or not np.isfinite(a).all() or np.any(a<0) for a in (s,t,v))
            or not np.isfinite([median,scale]).all() or scale<=0):
        raise ValueError('finite original float32 Source/Target/SEGMENT rows required')
    u=np.minimum(s,t.min(axis=0))
    if np.any(v>u+ATOL+RTOL*np.abs(u)):
        raise ValueError('SEGMENT exceeds full UNION')
    # Exact previous audit order: float32 scale+epsilon, float64 signed difference.
    denominator=(np.asarray([scale],np.float32)+1e-8)[0]
    d=(u.astype(np.float64)-v)/denominator
    order=np.argsort(t,axis=0,kind='stable')
    sorted_t=np.take_along_axis(t,order,axis=0)
    z=((u-np.float32(median))/np.float32(scale)).astype(np.float32)
    w=np.maximum(s.astype(np.float64)[None,:]-t,0)
    wsum=w.sum(axis=0);squares=(w*w).sum(axis=0)
    effective=np.full(len(s),np.nan)
    np.divide(wsum**2,squares,out=effective,where=squares>0)
    result=dict(source=s,union=u,segment=v,b=z,delta=d,
        effective_reduction=(u-v)>ATOL+RTOL*np.abs(u),
        source_advantage=(s.astype(np.float64)-sorted_t[0])/denominator,
        count_better_source=((s[None,:]-t)>(ATOL+RTOL*np.abs(s))[None,:]).sum(axis=0),
        count_tie_source=(np.abs(s[None,:]-t)<=(ATOL+RTOL*np.abs(s))[None,:]).sum(axis=0),
        effective_support_count=effective)
    for k in (2,4,8):result[f'gap{k}']=(sorted_t[k-1].astype(np.float64)-sorted_t[0])/denominator
    loo=[]
    for j in range(8):
        removed=np.minimum(s,np.delete(t,j,axis=0).min(axis=0))
        value=(removed.astype(np.float64)-u)/denominator
        result[f'target_{j+1}']=t[j];result[f'target_sorted_{j+1}']=sorted_t[j]
        result[f'loo_{j+1}']=value;loo.append(value)
    result['loo_max']=np.max(loo,axis=0)
    return result


def high_response(physical_fused):
    a=np.asarray(physical_fused)
    if a.ndim!=2 or not np.isfinite(a).all():raise ValueError('finite native fused grid required')
    n=a.size;k=max(1,(n+99)//100)
    # Highest response, deterministic physical flat index for all ties.
    return np.lexsort((np.arange(n),-a.ravel()))[:k]


def layer_profiles(ctx,raw,meta,segments,union,segment):
    out=[];grid=tuple(meta['grid'])
    for j,l in enumerate(LAYERS):
        p=native(raw['source_distance'][j],raw['target_distance'][:,j],segment[j],ctx['medians'][j],ctx['scales'][j])
        for name in ('alpha','endpoint_length'):
            p[name]=scatter(segments[l][name],segments[l]['positions'],grid).ravel()[raw['positions']]
        for k in (0,1):
            p[f'endpoint_support_{k+1}']=scatter(segments[l]['endpoint_support'],segments[l]['positions'],grid).reshape(2,-1)[k,raw['positions']]
            p[f'endpoint_index_{k+1}']=scatter(segments[l]['endpoint_index'],segments[l]['positions'],grid).reshape(2,-1)[k,raw['positions']]
        if not np.array_equal(p['union'],union[j]):raise ValueError('UNION reconstruction differs')
        out.append(p)
    return out


def summary_at_stencil(profile,offsets,weights):
    """Interpolate descriptions for explanation only, never interpolate before min."""
    result={}
    for key in FEATURES:
        values=np.asarray(profile[key])[offsets].astype(np.float64)
        valid=np.isfinite(values);mass=float(np.asarray(weights)[valid].sum())
        result[key]=float(np.dot(values[valid],np.asarray(weights)[valid])/mass) if mass>0 else None
        result[key+'_valid_stencil_mass']=mass
    return result
