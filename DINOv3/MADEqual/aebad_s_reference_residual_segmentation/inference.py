"""All head families consume the same unmodified query and Exact original banks."""
import numpy as np
from . import inputs as ip,storage as st
from .protocol import LAYERS,CONFIG,U,S
from .retrieval import References,exact_with_audit,target_nearest


def context(a,row,q,geometry,refs,*,cache_path=None):
    """Formal complete Source+K8 references; no train-fold or Target leave-one-out."""
    domain=row['domain'];origins=[r['relative_path'] for r in ip.catalog(a)['supports'][domain]]
    arrays={};audits=[];bank_ids=[]
    cached=st.get(cache_path) if cache_path is not None and (cache_path/'manifest.json').exists() else None
    for l in LAYERS:
        resident=hasattr(refs,'nearest_source')
        if resident:
            _,seal=refs.source('formal',l);_,seals,offsets=refs.parts(origins,l)
            rmeta=dict(source_seal=seal,target_seals=seals,target_offsets=offsets,target_origins=origins)
        else:source,target,rmeta=refs.tensors('formal',origins,l)
        bank_ids.append(rmeta)
        if cached:
            x,r=cached;ds=x[f'ds{l}'];si=x[f'si{l}'];dt=x[f'dt{l}'];ti=x[f'ti{l}']
        else:
            if resident:
                query=refs.search.query(q[l]);ds,si,sa,_=refs.nearest_source(q[l],'formal',l,query)
                dt,ti,ta,_,_=refs.nearest_target(q[l],origins,l,query);del query
            else:
                ds,si,sa=exact_with_audit(q[l],source,a.device)
                dt,ti,ta=target_nearest(q[l],target,rmeta['target_offsets'],a.device)
            arrays.update({f'ds{l}':ds,f'si{l}':si,f'dt{l}':dt,f'ti{l}':ti,
                f'perd{l}':ta['per_support_distances'],f'peri{l}':ta['per_support_indices']})
            audits.append(dict(layer=l,source=sa,target=ta['support_audits']))
    metadata=dict(identity=ip.identity(a),relative_path=row['relative_path'],image_sha256=row['image_sha256'],
        domain=domain,grid=geometry['grid'],positions=np.asarray(geometry['positions']).tolist(),banks=bank_ids,
        scope='formal_complete_Source_K8_no_self_calibration_origin_in_training_bank')
    if cached:
        if r['metadata']['inputs']!=metadata:raise ValueError('formal NN cache is not compatible')
        return x,r
    value=dict(inputs=metadata,audits=audits)
    if cache_path is not None:record=st.put(cache_path,arrays,value)
    else:record=dict(metadata=value,seal=st.digest(value))
    return arrays,record


def head_inputs(a,row,q,geometry,nn,refs,*,use_reference):
    import torch
    from .model import inputs
    norm,_=st.get(a.cache_root/'input_normalization');patch=st.sealed_json(a.run_root/'input_distance_mad.json')['patch'] if use_reference else None
    grid=geometry['grid'];order=np.argsort(np.asarray(geometry['positions'],np.int64));result=[]
    origins=[r['relative_path'] for r in ip.catalog(a)['supports'][row['domain']]]
    for j,l in enumerate(LAYERS):
        x=torch.from_numpy(np.ascontiguousarray(q[l][order].reshape(*grid,1280).transpose(2,0,1)))
        kwargs={}
        if use_reference:
            source,target,meta=refs.gather('formal',origins,l,nn[f'si{l}'][order],nn[f'ti{l}'][order])
            kwargs=dict(source=torch.from_numpy(np.ascontiguousarray(source.reshape(*grid,1280).transpose(2,0,1))),
                target=torch.from_numpy(np.ascontiguousarray(target.reshape(*grid,1280).transpose(2,0,1))),
                distances=torch.from_numpy(np.stack((nn[f'ds{l}'][order],nn[f'dt{l}'][order])).reshape(2,*grid)),
                patch=(float(patch['medians'][j]),float(patch['scales'][j])))
        value=inputs(x,torch.from_numpy(np.array(norm[f'mu{l}'])),float(norm[f'sigma{l}']),use_reference=use_reference,**kwargs)
        result.append(value[None].to(a.device))
    return result


def logits(net,values,positions):
    import torch
    # Keep frozen backbone flags outside this head-only scope unchanged.
    flags=dict(enabled=torch.backends.cudnn.enabled,benchmark=torch.backends.cudnn.benchmark,
        benchmark_limit=torch.backends.cudnn.benchmark_limit,deterministic=torch.backends.cudnn.deterministic,allow_tf32=False)
    with torch.inference_mode(),torch.backends.cudnn.flags(**flags):out=net(values).float()[0].cpu().numpy().reshape(4,-1)
    out=out[:,np.asarray(positions,np.int64)]
    if not np.isfinite(out).all():raise FloatingPointError('nonfinite raw head logits')
    return out.astype(np.float32)


def baseline(a,row,nn,geometry,refs):
    """Prefer exact compatible frozen baseline z; fallback uses current common bank."""
    from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.calibration import z_values
    from DINOv3.MADEqual.aebad_s_target_local_segment.geometry import geometry_layer
    state=st.sealed_json(a.run_root/'input_distance_mad.json');patch=state['patch']
    u=np.stack([np.minimum(nn[f'ds{l}'],nn[f'dt{l}']) for l in LAYERS])
    s=[];origins=[r['relative_path'] for r in ip.catalog(a)['supports'][row['domain']]]
    for l in LAYERS:
        if hasattr(refs,'parts'):parts,seals,offsets=refs.parts(origins,l)
        else:
            target,seals,offsets=refs.target(origins,l);parts=[target[lo:hi] for lo,hi in zip(offsets[:-1],offsets[1:])]
        idx=nn[f'peri{l}']-np.asarray(offsets[:-1],np.int64)[:,None]
        result,_=geometry_layer(nn[f'perd{l}'],idx,nn[f'ds{l}'],parts)
        s.append(result['K8_SEGMENT'])
    s=np.stack(s)
    parent_path=a.run_root/'baseline_parent.json'
    if parent_path.exists():
        from pathlib import Path
        parent=st.sealed_json(parent_path);p=Path(parent['cache'])/'predictions'/row['domain']/row['sample_id']
        if (p/'manifest.json').exists():
            x,r=st.get(p,(U,S,'positions'));m=r['metadata']
            recovered=x[U]*np.asarray(patch['scales'],np.float32)[:,None]+np.asarray(patch['medians'],np.float32)[:,None]
            recovered_s=x[S]*np.asarray(patch['scales'],np.float32)[:,None]+np.asarray(patch['medians'],np.float32)[:,None]
            if (m['sample']==row and m['grid']==geometry['grid'] and np.array_equal(x['positions'],geometry['positions'])
                    and np.allclose(recovered,u,atol=1e-7,rtol=1e-6) and np.allclose(recovered_s,s,atol=1e-7,rtol=1e-6)):
                return {U:np.array(x[U]),S:np.array(x[S])},dict(kind='compatible_frozen_parent',run=parent['run'],seal=r['seal'],
                    current_union_max_abs=float(np.max(np.abs(recovered-u))),current_segment_max_abs=float(np.max(np.abs(recovered_s-s))))
    return {U:z_values(u,patch),S:z_values(s,patch)},dict(kind='recomputed_current_common_bank')
