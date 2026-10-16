"""Only SEGMENT and source-pair votes. No support-coordinate compatibility test."""
from __future__ import annotations
import numpy as np
from DINOv3.MADEqual.aebad_s_target_local_segment.geometry import geometry_layer, LAYERS, DIST_ATOL, DIST_RTOL
from DINOv3.MADEqual.aebad_s_neighborhood_segment_p0.gates import scatter

BRANCHES=('UNION_K8_CH1','SEGMENT_K8_CH1','SEGMENT_PAIR_GATE_K8_CH1')
U,S,P=BRANCHES


def pair_gate(endpoint_support,positions,grid):
    if len(grid)!=2 or min(grid)<=0:raise ValueError('positive 2D physical grid required')
    support=scatter(endpoint_support,positions,grid)
    if (support.shape!=(2,*grid) or support.dtype.kind not in 'iu'
            or np.any((support<1)|(support>8)) or np.any(support[0]==support[1])):
        raise ValueError('two different manifest support IDs required')
    pair=np.sort(support,axis=0);h,w=grid
    count=np.zeros(grid,np.uint8);votes=count.copy()
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            if dy==dx==0:continue
            i=(slice(max(0,-dy),min(h,h-dy)),slice(max(0,-dx),min(w,w-dx)))
            j=(slice(max(0,dy),min(h,h+dy)),slice(max(0,dx),min(w,w+dx)))
            count[i]+=1;votes[i]+=np.all(pair[(slice(None),*i)]==pair[(slice(None),*j)],axis=0)
    return (count>0)&(votes>=(count+1)//2),votes,count


def add_geometry(raw,features):
    """Use original feature endpoints, FP64 geometry and cache-error handling."""
    result=dict(raw);diagnostics=[];values=[]
    for off,layer in enumerate(LAYERS):
        geometry,diag=geometry_layer(raw['target_distance'][:,off],raw['target_index'][:,off],
                                    raw['source_distance'][off],features[layer])
        values.append(geometry);diagnostics.append(dict(layer=layer,**diag))
    for new,old in (('segment_distance','K8_SEGMENT'),('alpha','alpha'),('endpoint_length','endpoint_length'),
                    ('endpoint_support','endpoint_support'),('endpoint_index','endpoint_index'),
                    ('squared_correction','squared_correction')):
        result[new]=np.stack([v[old] for v in values])
    return result,diagnostics


def validate_geometry(raw):
    n=len(raw['positions']);selected=np.argsort(raw['target_distance'],axis=0,kind='stable')[:2]
    for key,shape in (('segment_distance',(4,n)),('alpha',(4,n)),('endpoint_length',(4,n)),
                      ('endpoint_support',(4,2,n)),('endpoint_index',(4,2,n))):
        if raw[key].shape!=shape or not np.isfinite(raw[key]).all():raise ValueError('invalid geometry '+key)
    if raw['segment_distance'].dtype!=np.float32 or np.any(raw['segment_distance']<0):raise ValueError('invalid SEGMENT units/dtype')
    if np.any((raw['alpha']<0)|(raw['alpha']>1)) or np.any(raw['endpoint_length']<0):raise ValueError('invalid saved alpha/length')
    if raw['endpoint_support'].dtype!=np.int8 or raw['endpoint_index'].dtype!=np.int64:raise ValueError('invalid endpoint dtype')
    for off in range(4):
        if (not np.array_equal(raw['endpoint_support'][off],selected[:,off]+1)
                or not np.array_equal(raw['endpoint_index'][off],raw['target_index'][selected[:,off],off,np.arange(n)])):
            raise ValueError('geometry endpoints differ from stable Top2')


def predict(raw,grid):
    """No sample label, GT, image, support coordinates or cross-image state input."""
    validate_geometry(raw)
    union=np.minimum(raw['source_distance'],raw['target_distance'].min(0))
    seg=raw['segment_distance'];pair=[];gates=[];votes=[];counts=[];stats=[]
    for off,layer in enumerate(LAYERS):
        gate,vote,count=pair_gate(raw['endpoint_support'][off],raw['positions'],grid)
        compact=gate.ravel()[raw['positions']];distance=np.where(compact,seg[off],union[off])
        for lower,upper in ((seg[off],distance),(distance,union[off])):
            if np.any(lower>upper+DIST_ATOL+DIST_RTOL*np.abs(upper)):
                raise ValueError('SEGMENT <= PAIR <= UNION failed; clipping forbidden')
        significant=union[off]-seg[off]>DIST_ATOL+DIST_RTOL*np.abs(union[off])
        stats.append(dict(layer=layer,patches=len(compact),accept_ratio=float(gate.mean()),
                          segment_reduced_patches=int(significant.sum()),pair_reduced_patches=int((significant&compact).sum()),
                          segment_reduced_ratio=float(significant.mean()),pair_reduced_ratio=float((significant&compact).mean()),
                          pair_reduction_given_accept=float((significant&compact).sum()/compact.sum()) if compact.any() else None,
                          boundary_patches=int((count<8).sum()),
                          **{f'{key}_{v}_patches':int((a==v).sum()) for key,a in (('neighbors',count),('votes',vote)) for v in range(9)}))
        pair.append(distance);gates.append(gate);votes.append(vote);counts.append(count)
    return dict(positions=raw['positions'].copy(),source_distance=raw['source_distance'].copy(),
                union_distance=union,segment_distance=seg.copy(),pair_distance=np.stack(pair),
                pair_gate=np.stack(gates),pair_votes=np.stack(votes),neighbor_count=np.stack(counts)),stats
