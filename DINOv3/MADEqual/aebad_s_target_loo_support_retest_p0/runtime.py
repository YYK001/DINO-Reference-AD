"""Explicit FP32 retrieval; one full query shared by all four independent maps."""
from pathlib import Path
import numpy as np
from . import inputs as ip
from .inputs import st
from .protocol import *
from DINOv3.MADEqual.aebad_s_target_loo_layer_weight_p0 import numerics as math
from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.calibration import response,z_values


def features(a,row,persist=False):
    ref=ip.feature_ref(a,row);p=ip.parent(a)
    if ref:
        x,r=st.get(ref['path'],tuple(f'q{l}' for l in LAYERS)+('positions',))
        if r['seal']!=ref['seal']:raise ValueError('feature manifest changed')
        q={l:x[f'q{l}'] for l in LAYERS};meta=dict(grid=r['metadata']['grid'],positions=x['positions']);origin=ref
    else:
        if not a.extract_missing:raise FileNotFoundError('complete query/support feature missing; explicit --extract-missing required: '+row['relative_path'])
        from DINOv3.MADEqual.aebad_s_same_origin_neighbor_p0 import feature_identity as fi
        from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.runtime import host,extraction,reliable_precision
        fi.compatible(fi.native_extraction(a),p.extraction);reliable_precision(a.device)
        if not hasattr(a,'_model'):a._model=host(a)[0]
        from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.fresh_inputs import Data
        with Data(a.data) as data:q,meta=extraction(a._model,data.image(row),a.device)
        a.feature_forwards=getattr(a,'feature_forwards',0)+1;origin=dict(kind='fresh_original_full_forward',image=row,extraction=p.extraction)
    pos=np.asarray(meta['positions'],np.int64);n=int(np.prod(meta['grid']))
    if not np.array_equal(np.sort(pos),np.arange(n)) or any(q[l].shape!=(n,1280) or q[l].dtype!=np.float32 for l in LAYERS):raise ValueError('complete original FP32 feature grid required')
    if ref is None and persist:
        dest=a.cache_root/'normal_features'/ip.layer_inputs.old.key(row)
        rec=st.put(dest,{**{f'q{l}':q[l] for l in LAYERS},'positions':pos},dict(image=row,grid=meta['grid'],extraction=p.extraction,kind='original_complete_grid'))
        origin=dict(path=str(dest),seal=rec['seal'])
    return q,dict(meta,positions=pos,output_shape=[row['image_height']//4,row['image_width']//4]),origin


class References:
    def __init__(self,a):
        from DINOv3.MADEqual.aebad_s_reference_residual_segmentation.execution import ResidentExact
        from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.runtime import reliable_precision
        reliable_precision(a.device)
        self.a=a;self.p,self.supports,_=ip.context(a);self.search=ResidentExact(a.device,6*2**30)
        a.exact_execution=self.search.stats
        ref=self.p.assets['references']['source'];self.source,r=st.get(ref['path'],tuple(f'L{l}' for l in LAYERS))
        if r['seal']!=ref['seal']:raise ValueError('Source changed; no rebuilding')
        self.source_seal=r['seal'];self.parts=[];self.support_seals=[]
        for row in self.supports:
            q,meta,origin=features(a,row,persist=True);self.parts.append(q);self.support_seals.append(st.digest(origin))

    def distances(self,q,heldout=None):
        if heldout is not None and heldout not in range(8):raise ValueError('invalid group-local heldout')
        result={}
        for l in LAYERS:
            query=self.search.query(q[l]);ds,si=self.search.exact(query,self.source[f'L{l}'],('Source',self.source_seal,l))
            dt=[];idx=[];selected=[]
            for j,part in enumerate(self.parts):
                if j==heldout:continue  # Entire support image excluded, never just a matching row.
                d,i=self.search.exact(query,part[l],('Target',self.support_seals[j],l));dt.append(d);idx.append(i);selected.append(j)
            if len(selected)!=(8 if heldout is None else 7):raise ValueError('group must contain exactly K8/K7')
            result[l]=dict(ds=ds,si=si,dt=np.stack(dt),indices=np.stack(idx),support_indices=selected)
        return result


def baseline(a,row,allow_compute=False,refs=None):
    p,_,_=ip.context(a)
    if a.support_seed==42:
        x,r=p.prediction(row)
        return x,dict(r['metadata'],origin=dict(kind='read_only_seed42_parent',manifest_seal=r['seal']))
    dest=ip.cache(a)/'baselines'/row['sample_id'];ref=ip.legacy_ref(a,row)
    if (dest/'manifest.json').exists() or ref:
        path=dest if (dest/'manifest.json').exists() else Path(ref['path']);x,r=st.get(path)
        if r['metadata'].get('dependencies')!=ip.baseline_dependencies(a,row):raise ValueError('baseline Source/support/feature identity differs')
        if ref and path!=dest and r['seal']!=ref['seal']:raise ValueError('legacy baseline changed')
        return x,dict(r['metadata'],origin=dict(kind='own_compact_baseline' if path==dest else 'compatible_legacy_reference',path=str(path),seal=r['seal']))
    if not allow_compute or not a.supplement_retrieval:raise FileNotFoundError('seed43 baseline missing; explicit --supplement-retrieval required: '+row['sample_id'])
    if refs is None:raise ValueError('current seed43 references required')
    q,m,origin=features(a,row,persist=False);nn=refs.distances(q)
    from DINOv3.MADEqual.aebad_s_target_local_segment.geometry import geometry_layer
    u=[];s=[];arrays={'positions':m['positions']};endpoint=[]
    for l in LAYERS:
        v=nn[l];parts=[x[l] for x in refs.parts]
        seg,_=geometry_layer(v['dt'],v['indices'],v['ds'],parts)
        u.append(seg['K8_UNION']);s.append(seg['K8_SEGMENT']);endpoint.append(seg['endpoint_support'])
    raw_u=np.stack(u);raw_s=np.stack(s)
    arrays.update({U:z_values(raw_u,p.mad['patch']),S:z_values(raw_s,p.mad['patch']),'raw_union':raw_u,'raw_segment':raw_s,
        'Source_distance':np.stack([nn[l]['ds'] for l in LAYERS]),'Source_index':np.stack([nn[l]['si'] for l in LAYERS]).astype(np.int32),
        'target_distance':np.stack([nn[l]['dt'] for l in LAYERS]),'target_index':np.stack([nn[l]['indices'] for l in LAYERS]).astype(np.int32),
        'endpoint_support':np.stack(endpoint)})
    rec=st.put(dest,arrays,dict(dependencies=ip.baseline_dependencies(a,row),grid=m['grid'],output_shape=m['output_shape'],query_feature_origin=origin,GT_used=False))
    return arrays,dict(rec['metadata'],origin=dict(kind='new_original_Exact_same_group',path=str(dest),seal=rec['seal']))


def weights(a):
    r=st.sealed_json(ip.setting(a)/'weights.json')
    if any(r.get(k)!=v for k,v in ip.signature(a).items()):raise ValueError('weights belong to another support set')
    return r


def maps(a,row,only=None,allow_compute=False,refs=None):
    x,m=baseline(a,row,allow_compute,refs);p=ip.parent(a);w=weights(a);images={};scores={}
    cached=None
    if a.support_seed==42 and a.domain in a.weight42_caches:
        root=a.weight42_caches[a.domain]/'predictions'
        path=root/row['sample_id'] if a.domain=='background' else root/a.domain/row['sample_id']
        if (path/'manifest.json').exists():
            cached,record=st.get(path,NEW);meta=record['metadata']
            provider=st.sealed_json(a.weight42_runs[a.domain]/'inputs.json');original_weights=st.sealed_json(a.weight42_runs[a.domain]/'weights.json')
            if meta['identity']!=provider['identity'] or meta['weights_seal']!=original_weights['seal'] or meta['sample']!=row:raise ValueError('seed42 weighted map provider differs')
            cached=(cached,meta['scores'])
    n=int(np.prod(m['grid']))
    if x['positions'].shape!=(n,) or not np.array_equal(np.sort(x['positions']),np.arange(n)):raise ValueError('baseline physical positions differ')
    for b in MAPS if only is None else (only,):
        if b in (U,S):images[b],scores[b]=response(x[b],x['positions'],m['grid'],m['output_shape'],p.mad['image'])
        elif cached is not None:images[b],scores[b]=cached[0][b],cached[1][b]
        else:images[b],scores[b]=math.weighted(x[U],x['positions'],m['grid'],m['output_shape'],w['weights'][b])
    return images,scores,m
