"""Lazy original host and fresh stages. No real computation at import time."""
import gc
from pathlib import Path
from types import FunctionType
import numpy as np
from . import inputs as ip
from .inputs import st
from .protocol import LAYERS,DOMAINS,CONFIGS,BRANCHES,U,S,G0,W,GL,ND,ID,config_id
from . import graph
from .numerics import scan_segments,direct_projection
from .selection import choose,matched_coefficient,controls


def host(a):
    from DINOv3.MADEqual.aebad_s_sparc_p0.runtime import load_host
    from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.inputs import PRECISION
    return load_host(dict(paths=dict(project=str(ip.ROOT),official=str(a.official_root),
        dinov3=str(a.dinov3_root),checkpoint=str(a.checkpoint)),
        frozen=dict(precision=PRECISION,host_config=st.read_json(a.official_root/'config.json'))),a.device)


def original_stage(a,name):
    """Bind only the host dependency; do not mutate the parent's module globals."""
    from DINOv3.MADEqual.aebad_s_support_witness_segment_p0 import fresh_runtime as old
    fn=getattr(old,name)
    return FunctionType(fn.__code__,dict(fn.__globals__,host=host),fn.__name__,fn.__defaults__)(a)


def build_source(a):
    if st.restore(a.cache_root/'source_bank.npz',ip.identity(a)): return dict(status='source_reused',model_forward_calls=0)
    ctx=ip.reuse_context(a)
    if ctx:
        _,base,identity,_=ctx
        if ip.link_reused(a,base/'source_bank.npz',a.cache_root/'source_bank.npz',identity):
            return dict(status='source_reused_explicit_fresh_origin',model_forward_calls=0)
    return original_stage(a,'source')


def calibrate(a):
    ctx=ip.reuse_context(a)
    if ctx:
        _,base,identity,_=ctx
        for i in range(1,67):
            ip.link_reused(a,base/'calibration'/f'{i:03d}.npz',a.cache_root/'calibration'/f'{i:03d}.npz',identity)
    return original_stage(a,'calibration')


def precision(a):
    import torch
    from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.inputs import PRECISION
    torch.cuda.set_device(torch.device(a.device))
    torch.set_float32_matmul_precision(PRECISION['float32_matmul_precision'])
    torch.backends.cuda.matmul.allow_tf32=PRECISION['cuda_matmul_allow_tf32']
    torch.backends.cudnn.allow_tf32=PRECISION['cudnn_allow_tf32']


def source_banks(a):
    from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.fresh_runtime import load_source
    return load_source(a,a.device)


def original_nearest(q,source,banks):
    """Exactly the parent's FP32 cdist Top1 and division order."""
    import torch
    from DINOv3.relation_reliability.superadd_host import _exact_top1_chunked
    with torch.inference_mode():
        query=torch.as_tensor(q,device=source.device)
        d,idx=_exact_top1_chunked(query,source)
        s=d.numpy().astype(np.float32)/1280
        ds=[];ids=[]
        for bank in banks.values():
            d,idx=_exact_top1_chunked(query,torch.as_tensor(bank,device=query.device))
            ds.append(d.numpy().astype(np.float32)/1280);ids.append(idx.numpy().astype(np.int64))
    return s,np.stack(ds),np.stack(ids)


def extract_support(a):
    identity=ip.identity(a);c=ip.catalog(a);ctx=ip.reuse_context(a);model=None;calls=0
    for d in getattr(a,'domains',DOMAINS):
        for row in c['supports'][d]:
            sid=row['support_index'];path=a.cache_root/'supports'/d/f'{sid:03d}.npz'
            if st.restore(path,identity): continue
            reused=False
            if ctx:
                _,base,old_identity,old=ctx
                for previous in old['supports'][d]:
                    if all(previous[k]==row[k] for k in ('relative_path','image_sha256')):
                        reused=ip.link_reused(a,base/'supports'/d/f"{previous['support_index']:03d}.npz",path,old_identity)
                        break
            if not reused:
                if model is None: model,_=host(a)
                from DINOv3.ShiftADD.aebad_s_p0.memory import _extract_four_layers
                with ip.Data(Path(c['data_root'])) as data:
                    layers,grid=_extract_four_layers(model,data.image(row),device=a.device,brightness_factor=None)
                calls+=1
                st.seal(path,identity,dict(relative_path=np.asarray(row['relative_path']),grid=np.asarray(grid,np.int32),
                    **{f'layer_{l}':layers[l] for l in LAYERS}),dict(grid=list(grid),relative_path=row['relative_path']),compressed=False)
            print(f'[fixed graph support] {d}/{sid} ready',flush=True)
    del model;gc.collect()
    # Supports are also the LOO queries. Search Source and every OTHER support
    # from these same saved features, rather than forwarding 24 images again.
    pending=[(d,h) for d in getattr(a,'domains',DOMAINS) for h in range(1,9)
             if st.restore(a.cache_root/'support_baseline'/d/f'{h:03d}.npz',identity) is None]
    if pending:
        precision(a);source=source_banks(a)
        for d in getattr(a,'domains',DOMAINS):
            result={h:dict(source=[],target=[],indices=[]) for dd,h in pending if dd==d}
            for l in LAYERS:
                banks=ip.load_banks(a,d,l)
                for h in result:
                    refs={k:v for k,v in banks.items() if k!=h}
                    s,t,idx=original_nearest(banks[h],source[l],refs)
                    result[h]['source'].append(s);result[h]['target'].append(t);result[h]['indices'].append(idx)
            for h,values in result.items():
                st.seal(a.cache_root/'support_baseline'/d/f'{h:03d}.npz',identity,
                    {k:np.stack(v) for k,v in values.items()},dict(domain=d,heldout=h,
                        reference_supports=[i for i in range(1,9) if i!=h],same_saved_support_features=True))
        del source
    ip.freeze_assets(a)
    return dict(status='supports_and_LOO_discrete_baselines_ready',support_forward_calls=calls,cost_estimate=ip.estimate(a))


def pair_path(a,d,l,u,v): return a.graph_root/'neighbors'/d/f'L{l}'/f'{u:02d}_to_{v:02d}.npz'
def graph_path(a,d,l,h): return a.graph_root/'graphs'/d/f'L{l}'/f'fold_{h}.npz'


def read_pairs(a,d,l,ids):
    pairs={};records=[]
    for u in ids:
        for v in ids:
            if u==v:continue
            x,r=st.arrays(pair_path(a,d,l,u,v),ip.identity(a));pairs[u,v]=x['indices'],x['distances']
            records.append(r['sha256'])
    return pairs,st.digest(records)


def load_graph(a,d,l,h):
    pool,record=st.arrays(graph_path(a,d,l,h),ip.identity(a))
    limits={(r['k'],r['q']):r['limit'] for r in record['metadata']['limits']}
    for key in ('zero_length_removed','unavailable_scales'):pool[key]=record['metadata'][key]
    return pool,limits,record


def prepare_graphs(a):
    from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.neighbors import topk
    from .neighbors import bidirectional
    identity=ip.identity(a);med,scale,state=ip.patch_state(a);rows=[];witness_rows=[]
    for d in getattr(a,'domains',DOMAINS):
        for l in getattr(a,'layers',LAYERS):
            j=LAYERS.index(l)
            banks=ip.load_banks(a,d,l)
            for u in banks:
                for v in banks:
                    if u>=v:continue
                    path=pair_path(a,d,l,u,v)
                    meta=dict(domain=d,layer=l,query_support=u,reference_support=v,k=100,
                        supports=[st.sha((a.cache_root/'supports'/d/f'{c:03d}.npz').with_suffix('.json')) for c in (u,v)])
                    reverse_path=pair_path(a,d,l,v,u)
                    reverse_meta=dict(meta,query_support=v,reference_support=u,supports=meta['supports'][::-1])
                    forward=st.restore(path,identity,meta);reverse=st.restore(reverse_path,identity,reverse_meta)
                    if forward is None and reverse is None:
                        one,two=bidirectional(banks[u],banks[v],device=a.device,query_chunk=a.query_chunk,
                            reference_chunk=a.reference_chunk,
                            progress=lambda n,total:print(f'[graph Top100 both ways] {d}/L{l} {u}<->{v} {n}/{total}',flush=True) if n==total else None)
                        for p,m,(idx,dist) in ((path,meta,one),(reverse_path,reverse_meta,two)):
                            st.seal(p,identity,dict(indices=idx,distances=dist),m,compressed=False)
                    elif forward is None or reverse is None:
                        aa,bb=(u,v) if forward is None else (v,u)
                        p,m=(path,meta) if forward is None else (reverse_path,reverse_meta)
                        idx,dist=topk(banks[aa],banks[bb],min(100,len(banks[bb])),device=a.device,
                            query_chunk=a.query_chunk,reference_chunk=a.reference_chunk)
                        st.seal(p,identity,dict(indices=idx,distances=dist),m,compressed=False)
            for h in sorted(getattr(a,'heldouts',range(9)),key=lambda h:(h==0,h)):
                path=graph_path(a,d,l,h);retained=[c for c in banks if c!=h]
                # Whole-image exclusion changes Top-k, scales and quantiles.
                fold={c:banks[c] for c in retained};pairs,pair_identity=read_pairs(a,d,l,retained)
                if st.restore(path,identity) is None:
                    pool=graph.mutual_pool(fold,pairs);limits=graph.length_limits(pool)
                    selected=np.zeros(len(pool['edges']),bool)
                    for k in (10,30,100):selected|=(pool['rank']<=k)&(pool['length']<=limits[k,.9])
                    for key in ('edges','rank','length'):pool[key]=pool[key][selected]
                    n=len(pool['edges']);nodes=len(pool['scales']);others=max(0,len(fold)-2)
                    estimate=dict(domain=d,layer=l,fold=h,nodes=nodes,candidate_edges=n,
                        witness_point_tests_estimate=n*others*max(map(len,fold.values())),
                        LOO_query_edge_pairs_estimate=(len(banks[h])*n if h else None),
                        graph_arrays_estimate_bytes=n*198+nodes*20,
                        geometry_block_pairs=a.witness_edge_chunk*a.reference_chunk,
                        endpoint_distance_block_bytes=a.query_chunk*nodes*8,
                        note='counts/bytes, not measured seconds; exhaustive, no resource admission gate')
                    st.write_json(path.with_name(path.stem+'_estimate.json'),estimate)
                    print('[graph cost] '+__import__('json').dumps(estimate),flush=True)
                    witness=graph.witness_pool(fold,pool,device=a.device,edge_chunk=a.witness_edge_chunk,
                        reference_chunk=a.reference_chunk,
                        progress=lambda n,total:print(f'[graph witness full search] {d}/L{l} fold={h} {n}/{total}',flush=True)
                            if n==total or n%(a.witness_edge_chunk*100)==0 else None)
                    meta=dict(domain=d,layer=l,heldout=h,retained=retained,pair_identity=pair_identity,
                        candidate_edges=len(pool['edges']),nodes=len(pool['scales']),
                        limits=[dict(k=k,q=q,limit=value) for (k,q),value in limits.items()],
                        zero_length_removed=pool.pop('zero_length_removed'),unavailable_scales=pool.pop('unavailable_scales'))
                    st.seal(path,identity,{**pool,**witness},meta,compressed=False)
                pool,limits,record=load_graph(a,d,l,h)
                if record['metadata']['pair_identity']!=pair_identity:raise ValueError('graph neighbor dependencies changed')
                masks=graph.masks(pool,limits)
                finite=np.isfinite(pool['eta'])
                witness_rows.append(dict(domain=d,layer=l,fold=h,edges=len(pool['edges']),
                    graph_file_bytes=path.stat().st_size,
                    no_valid_third_point=int((~finite).sum()),valid_internal_points=int(pool['valid_internal_points'].sum()),
                    outside_interval_points=int(pool['outside_interval'].sum()),
                    invalid_scale_points=int(pool['invalid_scale_points'].sum()),
                    eta_min=float(pool['eta'][finite].min()) if finite.any() else None,
                    eta_median=float(np.median(pool['eta'][finite])) if finite.any() else None,
                    **{f'accepted_eta_{eta:.2f}':int((pool['eta']<=eta).sum()) for eta in (.25,.5,.75,1.)}))
                for config,mask in zip(CONFIGS,masks):
                    summary=graph.summary(pool,mask,float(scale[j]),domain=d,layer=l,fold=h,config=config_id(config))
                    summary['length_limit']=limits[config[0],config[1]]
                    summary['witness_accepted_edges']=int(((pool['eta']<=config[2])&mask).sum()) if config[2] is not None else None
                    rows.append(summary)
    ip.merge_rows(a.run_root/'graph_summary.csv',rows,('domain','layer','fold','config'))
    ip.merge_rows(a.run_root/'witness_search_summary.csv',witness_rows,('domain','layer','fold'))
    return dict(status='requested_graph_jobs_ready',graphs=len(witness_rows))


def support_loo(a):
    identity=ip.identity(a);rows=[]
    for d in getattr(a,'domains',DOMAINS):
        for l in getattr(a,'layers',LAYERS):
            for h in (h for h in getattr(a,'heldouts',range(1,9)) if h):
                path=a.graph_root/'loo_distances'/d/f'L{l}'/f'{h:03d}.npz'
                pool,limits,grec=load_graph(a,d,l,h)
                baseline,brec=st.arrays(a.cache_root/'support_baseline'/d/f'{h:03d}.npz',identity)
                meta=dict(domain=d,layer=l,heldout=h,graph_sha256=grec['sha256'],baseline_sha256=brec['sha256'])
                saved=st.restore(path,identity)
                if saved and any(saved['metadata'][k]!=v for k,v in meta.items()):
                    raise ValueError('LOO dependencies differ')
                if saved is None:
                    all_banks=ip.load_banks(a,d,l);q=all_banks[h]
                    banks={c:v for c,v in all_banks.items() if c!=h};nodes,*_=graph.layout(banks)
                    off=LAYERS.index(l);u=np.minimum(baseline['source'][off],baseline['target'][off].min(0))
                    distances,winners,diag=scan_segments(q,nodes,pool['edges'],graph.masks(pool,limits),u,
                        device=a.device,query_chunk=a.query_chunk,edge_chunk=a.edge_chunk,reference_chunk=a.reference_chunk,
                        progress=lambda n,total:print(f'[graph normal LOO] {d}/L{l} heldout={h} {n}/{total}',flush=True)
                            if n==total else None)
                    st.seal(path,identity,dict(union=u,distances=distances),dict(meta,numerics=diag))
                x,r=st.arrays(path,identity)
                if any(r['metadata'][k]!=v for k,v in meta.items()):raise ValueError('LOO dependencies differ')
                rows.append(dict(domain=d,layer=l,heldout=h,images=1,graph_sha256=grec['sha256']))
    ip.merge_rows(a.run_root/'support_loo_jobs.csv',rows,('domain','layer','heldout'))
    return dict(status='requested_Source_Target7_graph_LOO_jobs_complete',jobs=len(rows))


def select(a):
    from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import statistics,ch1_score
    identity=ip.identity(a);med,scale,state=ip.patch_state(a);all_rows=[];witness_rows=[]
    for d in getattr(a,'domains',DOMAINS):
        scores={config_id(c):[] for c in CONFIGS};cost={k:[] for k in scores};empty=[]
        dependencies=[]
        for h in range(1,9):
            layer_arrays=[];complexity=[]
            row=ip.catalog(a)['supports'][d][h-1]
            with np.load(a.cache_root/'supports'/d/f'{h:03d}.npz') as z:grid=tuple(z['grid'])
            for l in LAYERS:
                path=a.graph_root/'loo_distances'/d/f'L{l}'/f'{h:03d}.npz'
                x,r=st.arrays(path,identity);layer_arrays.append(x);dependencies.append(r['sha256'])
                pool,limits,_=load_graph(a,d,l,h)
                complexity.append([graph.summary(pool,m,float(scale[LAYERS.index(l)]),domain=d,layer=l,fold=h,config=config_id(c))
                    for c,m in zip(CONFIGS,graph.masks(pool,limits))])
            n=np.prod(grid);positions=np.arange(n,dtype=np.int64);shape=(row['image_height']//4,row['image_width']//4)
            def score(raw):
                _,peak,local=statistics(raw,positions,grid,shape,med,scale)
                return ch1_score(peak,local,state)
            empty.append(score(np.stack([x['union'] for x in layer_arrays])))
            for i,c in enumerate(CONFIGS):
                key=config_id(c);s=score(np.stack([x['distances'][i] for x in layer_arrays]));scores[key].append(s)
                cost[key].append([float(np.mean([x[i]['normalized_length_per_patch'] for x in complexity])),
                                  float(np.mean([x[i]['edges_per_patch'] for x in complexity]))])
                all_rows.append(dict(domain=d,heldout=h,config=key,CH1=s,UNION_CH1=empty[-1],
                    normalized_length_per_patch=cost[key][-1][0],edges_per_patch=cost[key][-1][1],GT_used=False))
        main=choose(scores,empty,cost,witness=True);plain=choose(scores,empty,cost,witness=False)
        selected=main['selected'];paired=tuple(selected['config'][:2])+ (None,) if selected else None
        wi=CONFIGS.index(tuple(selected['config'])) if selected else None
        gi=CONFIGS.index(paired) if paired else None
        coefficients=[];audits=[]
        for l in LAYERS:
            uu=[];ww=[];gg=[]
            for h in range(1,9):
                x,_=st.arrays(a.graph_root/'loo_distances'/d/f'L{l}'/f'{h:03d}.npz',identity)
                uu.append(x['union']);ww.append(x['distances'][wi] if wi is not None else x['union'])
                gg.append(x['distances'][gi] if gi is not None else x['union'])
            coefficient,audit=matched_coefficient(uu,ww,gg);coefficients.append(coefficient);audits.append(audit)
        value=dict(domain=d,identity=identity,main=main,plain=plain,paired_config=list(paired) if paired else None,
            lambda_by_layer=coefficients,lambda_audits=audits,LOO_dependencies=st.digest(dependencies),
            GT_used=False,test_images_used=False,evaluation_scope=a.evaluation_scope,seed=a.seed)
        value['seal']=st.digest(value);path=a.run_root/'selection'/f'{d}.json'
        if path.exists() and st.read_json(path)!=value:raise ValueError('frozen normal selection changed')
        st.write_json(path,value)
        for l in LAYERS:
            pool,limits,_=load_graph(a,d,l,0)
            for label,cfg in [('WITNESS_GRAPH',selected['config'] if selected else None),
                              ('LOCAL_GRAPH_PAIRED',paired),('LOCAL_GRAPH_LOO',plain['selected']['config'] if plain['selected'] else None)]:
                mask=graph.masks(pool,limits,[tuple(cfg)])[0] if cfg is not None else np.zeros(len(pool['edges']),bool)
                result=graph.summary(pool,mask,float(scale[LAYERS.index(l)]),domain=d,layer=l,fold=0,config=label)
                result.update(eta_threshold=cfg[2] if cfg else None,
                    represented_edges=int((pool['witness_support'][mask]>0).sum()),
                    zero_acceptance=int(mask.sum())==0,
                    witness_alpha_median=float(np.nanmedian(pool['witness_alpha'][mask])) if mask.any() else None)
                witness_rows.append(result)
    ip.merge_rows(a.run_root/'support_loo_scores.csv',all_rows,('domain','heldout','config'))
    ip.merge_rows(a.run_root/'witness_summary.csv',witness_rows,('domain','layer','config'))
    return dict(status='requested_normal_only_selection_frozen',domains=len(getattr(a,'domains',DOMAINS)))
