"""Reuse the tested continuation plumbing with isolated dependency bindings.

No changes to the old module's globals, parent records, graphs or selection.
No old FP64 predictions are imported into the independent FP32 experiment.
"""
import time
from pathlib import Path
from types import FunctionType, SimpleNamespace
import numpy as np
from DINOv3.MADEqual.aebad_s_fixed_graph_acceleration_p0 import continuation as base
from . import search

st=base.st


def bind_namespace(module,**overrides):
    values=dict(vars(module))
    for name,fn in vars(module).items():
        if isinstance(fn,FunctionType) and fn.__module__==module.__name__:
            rebound=FunctionType(fn.__code__,values,fn.__name__,fn.__defaults__,fn.__closure__)
            rebound.__kwdefaults__=None if fn.__kwdefaults__ is None else dict(fn.__kwdefaults__)
            values[name]=rebound
    values.update(overrides)
    return SimpleNamespace(**values)


def engine_files():
    roots=(Path(__file__).parent,Path(base.__file__).parent)
    source_root=Path(base.__file__).resolve().parents[3]
    return {p.relative_to(source_root).as_posix():st.sha(p) for root in roots for p in sorted(root.glob('*.py'))}


class Storage:
    def __getattr__(self,name):return getattr(st,name)

    @staticmethod
    def write_json(path,value):
        if Path(path).name=='protocol.json' and isinstance(value,dict) and 'geometry' in value:
            value=dict(value,geometry='FP32 shared-origin screen and raw-feature direct projection; fused single edge pass',
                       prediction_precision='FP32',parent_FP64_predictions_imported=False)
        return st.write_json(path,value)


class PredictionStorage(Storage):
    @staticmethod
    def seal(path,identity,arrays,metadata,**kwargs):
        seconds=sum(r.get('graph_numerics',{}).get('seconds',0.) for r in metadata.get('audits',[]))
        timing={key:sum(r.get('timing',{}).get(key,0.) for r in metadata.get('audits',[])) for key in
                ('original_discrete_seconds','original_SEGMENT_seconds','vector_diagnostics_seconds','layer_total_seconds')}
        print(f'[FP32 graph image] {metadata.get("domain")}/{metadata.get("sample_id")} '
              f'four_layer_graph_seconds={seconds:.3f} '
              +' '.join(f'{k}={v:.3f}' for k,v in timing.items()),flush=True)
        return st.seal(path,identity,arrays,dict(metadata,acceleration=dict(engine=search.ENGINE,
                       origin='new_FP32_prediction',parent_predictions_imported=False)),**kwargs)


def layer_branches(a,d,l,q,source,banks,pool,limits,choice,prepared=None,device_banks=None,nodes=None,active=None):
    """Keep old discrete/SEGMENT paths; graph and its vector diagnostics FP32."""
    from DINOv3.MADEqual.aebad_s_target_local_segment.geometry import geometry_layer
    c=get_service();p=c.protocol
    started=time.perf_counter()
    s,target,indices=c.runtime.original_nearest(q,source,device_banks if device_banks is not None else banks)
    nn_done=time.perf_counter()
    geometry,geo=geometry_layer(target,indices,s,list(banks.values()))
    geometry_done=time.perf_counter()
    union,segment=geometry['K8_UNION'],geometry['K8_SEGMENT']
    main,plain=choice['main']['selected'],choice['plain']['selected']
    configs=[tuple(choice['paired_config']) if main else None,
             tuple(main['config']) if main else None,tuple(plain['config']) if plain else None]
    if active is None:active=np.asarray([c.runtime.graph.masks(pool,limits,[cfg])[0] if cfg else np.zeros(len(pool['edges']),bool) for cfg in configs])
    if nodes is None:nodes,*_=c.runtime.graph.layout(banks)
    distances,winners,numerics=search.scan_segments(q,nodes,pool['edges'],active,union,device=a.device,
        query_chunk=a.query_chunk,edge_chunk=a.edge_chunk,reference_chunk=a.reference_chunk,prepared=prepared)
    graph_done=time.perf_counter()
    paired,witness,local=distances
    normal,image,audit=c.old.controls(union,paired,witness,choice['lambda_by_layer'][p.LAYERS.index(l)])
    values=dict(zip(p.BRANCHES,(union,segment,paired,witness,local,normal,image)))
    diagnostics={};wd=np.full(len(q),np.nan,np.float32)
    for k,b in enumerate((p.G0,p.W,p.GL)):
        win=winners[k].copy();win[distances[k]>=union]=-1
        alphas=np.full(len(q),np.nan,np.float32);take=np.flatnonzero(win>=0)
        if len(take):
            es=pool['edges'][win[take]];u=nodes[es[:,0]];direction=nodes[es[:,1]]-u;delta=q[take]-u
            alphas[take]=np.clip(np.sum(delta*direction,axis=-1,dtype=np.float32)/
                                 np.sum(direction*direction,axis=-1,dtype=np.float32),np.float32(0),np.float32(1))
            if b==p.W:
                ss=pool['witness_support'][win[take]];rr=pool['witness_row'][win[take]]
                if (ss<1).any():raise ValueError('winning witnessed edge lacks representative')
                points=np.stack([banks[int(sid)][i] for sid,i in zip(ss,rr)])
                difference=q[take]-points
                wd[take]=np.sqrt(np.sum(difference*difference,axis=-1,dtype=np.float32))/np.float32(1280)
        diagnostics['winner_'+b]=win;diagnostics['alpha_'+b]=alphas
    diagnostics.update(source_distance=s,target_distance=target.min(0),witness_distance=wd)
    audit.update(domain=d,layer=l,SEGMENT_geometry=geo,graph_numerics=numerics,
        effective_WITNESS_positions=int(((union.astype(np.float64)-witness)>1e-7+1e-6*np.abs(union)).sum()),
        raw_distance_unit='Euclidean /1280 once',graph_vector_diagnostics_dtype='FP32',
        timing=dict(original_discrete_seconds=nn_done-started,original_SEGMENT_seconds=geometry_done-nn_done,
                    vector_diagnostics_seconds=time.perf_counter()-graph_done,layer_total_seconds=time.perf_counter()-started))
    return values,diagnostics,audit


def predict(a,*,limit=None):
    a.limit=limit
    freeze=base.bind(base.old.freeze,PreparedSearch=search.PreparedSearch,
                     layer_branches=layer_branches,st=PredictionStorage())
    return freeze(a)


def smoke(a,*,query_count=128):
    c=get_service()
    from DINOv3.MADEqual.aebad_s_fixed_target_graph_p0.numerics import scan_segments as reference
    c.runtime.precision(a);model,_=c.runtime.host(a);rows=[]
    for d in c.protocol.DOMAINS:
        sample=next(s for s in c.ip.catalog(a)['samples'] if s['domain']==d)
        with c.ip.Data(Path(c.ip.catalog(a)['data_root'])) as data:
            features,_=c.old.extraction(model,data.image(sample),a.device)
        choice=c.old.selection(a,d)
        for l in c.protocol.LAYERS:
            banks=c.ip.load_banks(a,d,l);nodes,*_=c.runtime.graph.layout(banks)
            pool,limits,_=c.runtime.load_graph(a,d,l,0)
            main,plain=choice['main']['selected'],choice['plain']['selected']
            configs=[tuple(choice['paired_config']) if main else None,
                     tuple(main['config']) if main else None,tuple(plain['config']) if plain else None]
            masks=np.asarray([c.runtime.graph.masks(pool,limits,[cfg])[0] if cfg else
                              np.zeros(len(pool['edges']),bool) for cfg in configs])
            q=features[f'layer_{l}'][:query_count];union=np.full(len(q),1e8,np.float32)
            started=time.perf_counter()
            old=reference(q,nodes,pool['edges'],masks,union,device=a.device,
                          query_chunk=a.query_chunk,edge_chunk=a.edge_chunk,reference_chunk=a.reference_chunk)
            reference_seconds=time.perf_counter()-started
            started=time.perf_counter()
            new=search.scan_segments(q,nodes,pool['edges'],masks,union,device=a.device,
                          query_chunk=a.query_chunk,edge_chunk=a.edge_chunk,reference_chunk=a.reference_chunk)
            seconds=time.perf_counter()-started
            provenance_ok=True;direct_ok=True
            for k in range(3):
                take=np.flatnonzero(new[1][k]>=0);ids=new[1][k,take]
                if len(ids):
                    provenance_ok &= bool(masks[k,ids].all())
                    es=pool['edges'][ids]
                    direct=np.sqrt(search.direct_fp32(q[take],nodes[es[:,0]],nodes[es[:,1]]))/np.float32(1280)
                    direct_ok &= bool(np.allclose(new[0][k,take],direct,atol=1e-7,rtol=1e-6))
            row=dict(domain=d,layer=l,sample_id=sample['sample_id'],queries=len(q),atol=1e-7,rtol=1e-6,
                     max_abs_error=float(np.abs(new[0]-old[0]).max(initial=0)),
                     winner_mismatches=int((new[1]!=old[1]).sum()),
                     winner_policy='FP32 canonical ties; FP64 winner changes audited, not relabeled as bitwise equal',
                     direct_FP32_reference_verified=direct_ok,selected_edge_provenance_verified=provenance_ok,
                     verified=bool(np.allclose(new[0],old[0],atol=1e-7,rtol=1e-6) and direct_ok and provenance_ok),
                     reference_seconds=reference_seconds,FP32_seconds=seconds,diagnostics=new[2],
                     timing_scope='128-row sample includes preparation/JIT; not full-image speedup')
            rows.append(row);print('[FP32 graph smoke] '+__import__('json').dumps(row),flush=True)
    value=dict(identity=c.ip.identity(a),engine=search.ENGINE,checks=rows,
               verified=all(r['verified'] for r in rows),GT_used=False,query_forward_calls=3)
    value['seal']=st.digest(value)
    # Retain the shared plumbing's filename, with an explicitly FP32 engine.
    st.write_json(a.run_root/'mixed_smoke.json',value)
    if not value['verified']:raise ValueError('FP32/reference distance verification failed; evidence retained')
    return value


_SERVICE=None


def get_service():
    global _SERVICE
    if _SERVICE is None:
        _SERVICE=bind_namespace(base,search=search,st=Storage(),engine_files=engine_files,
                                PredictionStorage=PredictionStorage,smoke=smoke,predict=predict)
        original_initialize=_SERVICE.initialize
        def initialize(*args,import_completed=False,**kwargs):
            if import_completed:raise ValueError('FP32 run must not import old FP64 predictions')
            return original_initialize(*args,import_completed=False,**kwargs)
        _SERVICE.initialize=initialize
    return _SERVICE
