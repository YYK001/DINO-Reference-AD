"""Support-only global mutual graphs, fold-specific scales and exhaustive witness."""
import numpy as np
from .protocol import KS, QUANTILES, ETAS, CONFIGS
from .numerics import Backend


def layout(banks):
    ids=sorted(banks);counts=[len(banks[c]) for c in ids]
    offsets=dict(zip(ids,np.cumsum([0]+counts[:-1]).tolist()))
    return (np.concatenate([banks[c] for c in ids]),offsets,
            np.concatenate([np.full(len(banks[c]),c,np.int32) for c in ids]),
            np.concatenate([np.arange(len(banks[c]),dtype=np.int64) for c in ids]))


def mutual_pool(banks,pairs,max_k=100):
    """Pair tables retain min(max_k, image size); rebuild after image exclusion.

    Merging those tables is sufficient for the exact GLOBAL cross-image Top-k:
    a row below rank k in any individual image cannot enter the global Top-k.
    """
    nodes,off,support,row=layout(banks);total=len(nodes)
    scales=np.full(total,np.nan);sources=[];destinations=[];distances=[];ranks=[]
    for a in sorted(banks):
        nearest=[];best_i=np.empty((len(banks[a]),0),np.int64);best_d=np.empty_like(best_i,dtype=np.float64)
        for c in sorted(banks):
            if a==c: continue
            idx,d=pairs[a,c]
            if idx.shape!=d.shape or len(idx)!=len(banks[a]) or idx.shape[1]<min(max_k,len(banks[c])):
                raise ValueError('complete per-image Top-k tables required')
            nearest.append(d[:,0])
            joined_i=np.concatenate((best_i,idx+off[c]),axis=1)
            joined_d=np.concatenate((best_d,d),axis=1)
            choose=np.lexsort((joined_i,joined_d),axis=1)[:,:max_k]
            best_i=np.take_along_axis(joined_i,choose,axis=1)
            best_d=np.take_along_axis(joined_d,choose,axis=1)
        scales[off[a]:off[a]+len(banks[a])]=np.median(np.stack(nearest),axis=0)
        k=best_i.shape[1]
        sources.append(np.repeat(np.arange(off[a],off[a]+len(banks[a])),k))
        destinations.append(best_i.ravel());distances.append(best_d.ravel())
        ranks.append(np.tile(np.arange(1,k+1,dtype=np.int16),len(banks[a])))
    src=np.concatenate(sources);dst=np.concatenate(destinations)
    length=np.concatenate(distances);rank=np.concatenate(ranks)
    code=src.astype(np.int64)*total+dst
    order=np.argsort(code);sorted_code=code[order]
    reverse=dst.astype(np.int64)*total+src
    pos=np.searchsorted(sorted_code,reverse);safe=np.minimum(pos,len(code)-1)
    reciprocal=(pos<len(code))&(sorted_code[safe]==reverse)&(src<dst)
    take=np.flatnonzero(reciprocal);partner=order[safe[take]]
    edges=np.stack((src[take],dst[take]),axis=1)
    rank=np.maximum(rank[take],rank[partner]);length=length[take]
    canonical=np.lexsort((edges[:,1],edges[:,0]))
    edges,rank,length=edges[canonical],rank[canonical],length[canonical]
    zero=length==0
    return dict(edges=edges[~zero],rank=rank[~zero],length=length[~zero],
        scales=scales,node_support=support,node_row=row,zero_length_removed=int(zero.sum()),
        unavailable_scales=int((~np.isfinite(scales)|(scales<=0)).sum()))


def length_limits(pool,ks=KS,quantiles=QUANTILES):
    return {(k,q):(float(np.quantile(pool['length'][pool['rank']<=k],q))
                    if np.any(pool['rank']<=k) else 0.) for k in ks for q in quantiles}


def masks(pool,limits,configs=CONFIGS):
    return np.asarray([(pool['rank']<=k)&(pool['length']<=limits[k,q])&(
        True if eta is None else pool['eta']<=eta) for k,q,eta in configs],bool)


def witness_pool(banks,pool,*,device='cpu',edge_chunk=64,reference_chunk=1024,progress=None):
    """Scan ALL third-image points, not endpoint neighbors; shared by all eta.

    Invalid scale never gets epsilon. FP64 projection identities are checked
    against conservative intervals and direct vectors near alpha/eta boundaries
    and wherever the representative minimum could be numerically ambiguous.
    """
    nodes,off,support,_=layout(banks);edges=pool['edges'];n=len(edges)
    backend=Backend(device);x=backend.array(nodes)
    result=dict(eta=np.full(n,np.inf),witness_support=np.full(n,-1,np.int32),
        witness_row=np.full(n,-1,np.int64),witness_alpha=np.full(n,np.nan),
        witness_h=np.full(n,np.nan),witness_scale=np.full(n,np.nan),
        source_counts=np.zeros((n,len(ETAS)),np.int16),
        alpha_histogram=np.zeros((n,len(ETAS),6),np.int32),
        valid_internal_points=np.zeros(n,np.int64),outside_interval=np.zeros(n,np.int64),
        invalid_scale_points=np.zeros(n,np.int64))
    for start in range(0,n,edge_chunk):
        stop=min(start+edge_chunk,n);es=edges[start:stop]
        for c in sorted(banks):
            loc=np.flatnonzero((support[es[:,0]]!=c)&(support[es[:,1]]!=c))
            if not len(loc): continue
            rows=start+loc;u=x[es[loc,0]];v=x[es[loc,1]];direction=v-u
            cc=(direction*direction).sum(-1)
            source_pass=np.zeros((len(loc),len(ETAS)),bool)
            for r in range(0,len(banks[c]),reference_chunk):
                w=x[off[c]+r:off[c]+min(r+reference_chunk,len(banks[c]))]
                a,aerr=backend.squared(u,w)
                t=direction@w.T-(direction*u).sum(-1)[:,None]
                alpha=t/cc[:,None];h2=a-t*t/cc[:,None]
                un=(u*u).sum(-1);wn=(w*w).sum(-1)
                terr=32*(nodes.shape[1]+8)*np.finfo(np.float64).eps*(
                    (cc[:,None]*wn[None,:])**.5+((cc*un)**.5)[:,None])
                herr=aerr+(2*abs(t)+terr)*terr/cc[:,None]
                aa=backend.numpy(alpha); hh=backend.numpy(h2);err=backend.numpy(herr)
                ae=backend.numpy(terr/cc[:,None]);scale=pool['scales'][off[c]+r:off[c]+r+len(w)]
                if np.any(hh < -err): raise FloatingPointError('witness residual square below FP64 bound')
                usable=np.isfinite(scale)&(scale>0)
                # Only a diagnostic interval; negative/near-zero values get direct recomputation.
                ratio=np.sqrt(np.maximum(hh,0))/np.where(usable,scale,1)[None,:]
                low=np.sqrt(np.maximum(hh-err,0))/np.where(usable,scale,1)[None,:]
                high=np.sqrt(np.maximum(hh+err,0))/np.where(usable,scale,1)[None,:]
                ambiguous=(hh<=err)|((abs(aa-.2)<=ae)|(abs(aa-.8)<=ae))
                for eta in ETAS: ambiguous|=(low<=eta)&(high>=eta)&usable
                can_internal=(aa+ae>=.2)&(aa-ae<=.8)&usable
                upper=np.min(np.where(can_internal,high,np.inf),axis=1)
                ambiguous|=can_internal&(low<=upper[:,None])
                qi,wi=np.where(ambiguous)
                for z in range(0,len(qi),256):
                    ii,jj=qi[z:z+256],wi[z:z+256]
                    delta=w[jj]-u[ii];d=direction[ii]
                    al=(delta*d).sum(-1)/cc[ii]
                    residual=delta-al[:,None]*d
                    aa[ii,jj]=backend.numpy(al);hh[ii,jj]=backend.numpy((residual*residual).sum(-1))
                backend.direct_pairs+=len(qi)
                if (hh<0).any(): raise FloatingPointError('negative witness residual after direct refinement')
                inside=(aa>=.2)&(aa<=.8)
                result['outside_interval'][rows]+=(~inside).sum(1)
                result['invalid_scale_points'][rows]+=(inside&~usable).sum(1)
                valid=inside&usable
                result['valid_internal_points'][rows]+=valid.sum(1)
                ratio=np.sqrt(hh)/np.where(usable,scale,1)[None,:]
                for e,eta in enumerate(ETAS):
                    passed=valid&(ratio<=eta);source_pass[:,e]|=passed.any(1)
                    for b in range(6):
                        lower=.2+b*.1;upper=.2+(b+1)*.1
                        result['alpha_histogram'][rows,e,b]+=(passed&(aa>=lower)&(
                            aa<=upper if b==5 else aa<upper)).sum(1)
                value=np.where(valid,ratio,np.inf);col=np.argmin(value,axis=1)
                best=value[np.arange(len(rows)),col]
                # c,row scan order implements representative tie semantics.
                take=best<result['eta'][rows];rr=rows[take];jj=col[take]
                result['eta'][rr]=best[take];result['witness_support'][rr]=c
                result['witness_row'][rr]=r+jj;result['witness_alpha'][rr]=aa[take,jj]
                result['witness_h'][rr]=np.sqrt(hh[take,jj]);result['witness_scale'][rr]=scale[jj]
            result['source_counts'][rows]+=source_pass.astype(np.int16)
        if progress: progress(stop,n)
    result['direct_pairs']=np.asarray(backend.direct_pairs,np.int64)
    return result


def summary(pool,mask,scale,*,domain,layer,fold,config):
    edges=pool['edges'][mask];length=pool['length'][mask];ids=pool['node_support']
    degree=np.bincount(edges.ravel(),minlength=len(ids)) if len(edges) else np.zeros(len(ids),int)
    counts={f'{a}-{b}':int(((ids[edges[:,0]]==a)&(ids[edges[:,1]]==b)).sum())
            for a in sorted(set(ids)) for b in sorted(set(ids)) if a<b}
    return dict(domain=domain,layer=layer,fold=fold,config=config,nodes=len(ids),edges=len(edges),
        covered_nodes=int((degree>0).sum()),coverage=float((degree>0).mean()),
        max_degree=int(degree.max(initial=0)),mean_degree=float(degree.mean()),
        raw_total_length=float(length.sum()),raw_max_length=float(length.max(initial=0)),
        normalized_length_per_patch=float((length/1280/scale).sum()/len(ids)),
        normalized_length_unit='Euclidean / 1280 / frozen Source patch MAD scale',
        edges_per_patch=float(len(edges)/len(ids)),image_pair_counts=counts,
        zero_length_removed=pool['zero_length_removed'],unavailable_scales=pool['unavailable_scales'])
