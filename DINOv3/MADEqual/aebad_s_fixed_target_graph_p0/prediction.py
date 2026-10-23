"""One original-host forward per image; all seven branches share that Q."""
from pathlib import Path
import numpy as np
from . import inputs as ip,graph,runtime
from .inputs import st
from .protocol import LAYERS,BRANCHES,U,S,G0,W,GL,ND,ID
from .numerics import scan_segments,direct_projection,PreparedSearch
from .selection import controls


def selection(a,d):
    value=st.read_json(a.run_root/'selection'/f'{d}.json')
    if value['identity']!=ip.identity(a) or value['seal']!=st.digest({k:v for k,v in value.items() if k!='seal'}):
        raise ValueError('normal selection modified')
    return value


def extraction(model,image,device):
    """Actual observe_fewshot_pil extraction call chain, without duplicate forward."""
    import torch
    from DINOv3.relation_reliability.superadd_host import image_tensor,_extract_processed_batch
    with torch.inference_mode():
        processed=model.preprocessing(image_tensor(image).to(device)[None])
        pending=_extract_processed_batch(model,processed,device=device,query_masks=[None],include_identity_descriptors=False)
        if len(pending)!=1:raise ValueError('one complete query required')
        item=pending[0]
        return {f'layer_{l}':item.queries[l].detach().float().cpu().numpy() for l in LAYERS},dict(
            grid=list(item.grid),positions=item.positions.copy(),processed_shape=list(processed.shape[-2:]))


def layer_branches(a,d,l,q,source,banks,pool,limits,choice,prepared=None,device_banks=None,nodes=None,active=None):
    from DINOv3.MADEqual.aebad_s_target_local_segment.geometry import geometry_layer
    s,target,indices=runtime.original_nearest(q,source,device_banks if device_banks is not None else banks)
    geometry,diag=geometry_layer(target,indices,s,list(banks.values()))
    union=geometry['K8_UNION'];segment=geometry['K8_SEGMENT']
    main=choice['main']['selected'];plain=choice['plain']['selected']
    configs=[tuple(choice['paired_config']) if main else None,
             tuple(main['config']) if main else None,tuple(plain['config']) if plain else None]
    if active is None:active=np.asarray([graph.masks(pool,limits,[c])[0] if c else np.zeros(len(pool['edges']),bool) for c in configs])
    if nodes is None:nodes,*_=graph.layout(banks)
    distances,winners,numerics=scan_segments(q,nodes,pool['edges'],active,union,device=a.device,
        query_chunk=a.query_chunk,edge_chunk=a.edge_chunk,reference_chunk=a.reference_chunk,prepared=prepared)
    paired,witness,local=distances
    normal,image,audit=controls(union,paired,witness,choice['lambda_by_layer'][LAYERS.index(l)])
    values=dict(zip(BRANCHES,(union,segment,paired,witness,local,normal,image)))
    diagnostics={};witness_distance=np.full(len(q),np.nan,np.float32)
    for k,b in enumerate((G0,W,GL)):
        win=winners[k].copy();win[distances[k]>=union]=-1
        alphas=np.full(len(q),np.nan)
        take=np.flatnonzero(win>=0)
        if len(take):
            es=pool['edges'][win[take]]
            _,alpha=direct_projection(q[take],nodes[es[:,0]],nodes[es[:,1]])
            alphas[take]=alpha
            if b==W:
                ss=pool['witness_support'][win[take]];rr=pool['witness_row'][win[take]]
                if (ss<1).any(): raise ValueError('winning witnessed edge lacks representative')
                points=np.stack([banks[int(c)][i] for c,i in zip(ss,rr)])
                difference=q[take].astype(np.float64)-points.astype(np.float64)
                witness_distance[take]=(np.sqrt(np.sum(difference*difference,axis=1))/1280).astype(np.float32)
        diagnostics['winner_'+b]=win;diagnostics['alpha_'+b]=alphas
    diagnostics.update(source_distance=s,target_distance=target.min(0),witness_distance=witness_distance)
    audit.update(domain=d,layer=l,SEGMENT_geometry=diag,graph_numerics=numerics,
        effective_WITNESS_positions=int(((union.astype(np.float64)-witness)>1e-7+1e-6*np.abs(union)).sum()),
        raw_distance_unit='Euclidean /1280 once')
    return values,diagnostics,audit


def freeze(a):
    """Incomplete image sets never become frozen; disk Q survives downstream errors."""
    identity=ip.identity(a);catalog=ip.catalog(a)
    requested=getattr(a,'domains',ip.DOMAINS)
    cached=[d for d in requested if (a.run_root/'prediction_manifests'/f'{d}.json').exists()]
    if cached:verified(a,domains=cached)
    if len(cached)==len(requested):
        images=sum(s['domain'] in requested for s in catalog['samples'])
        return dict(status='all_predictions_frozen' if images==len(catalog['samples']) else 'requested_domains_frozen',
            query_forward_calls=0,new_images=0,reused_images=images,query_mode=a.query_mode,images=images)
    med,scale,state=ip.patch_state(a)
    from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import statistics,ch1_score
    model=None;source=None;calls=0;reused=0;new_images=0;records={d:[] for d in ip.DOMAINS}
    reuse=ip.reuse_context(a)
    reuse_q_root=None
    if reuse:
        previous=st.read_json(reuse[0]/'setup.json')['configuration']
        if previous.get('query_mode')=='disk':reuse_q_root=Path(previous['query_cache_root'])
    precision_ready=False
    for d in requested:
        if d in cached:
            records[d]=st.read_json(a.run_root/'prediction_manifests'/f'{d}.json')['records']
            reused+=len(records[d]);continue
        if not precision_ready:runtime.precision(a);precision_ready=True
        choice=selection(a,d);choice_sha=st.sha(a.run_root/'selection'/f'{d}.json')
        banks={l:ip.load_banks(a,d,l) for l in LAYERS}
        graphs={l:runtime.load_graph(a,d,l,0) for l in LAYERS}
        searches={};device_banks={};contexts={}
        for sample in (s for s in catalog['samples'] if s['domain']==d):
            path=a.prediction_root/d/(sample['sample_id']+'.npz')
            record=st.restore(path,identity)
            if record:
                _,meta,record=ip.verify_prediction(a,sample);reused+=1
            else:
                if getattr(a,'limit',None) is not None and new_images>=a.limit:continue
                qpath=a.query_cache_root/d/(sample['sample_id']+'.npz')
                qrecord=st.restore(qpath,identity) if a.query_mode=='disk' else None
                if qrecord is None and a.query_mode=='disk' and reuse_q_root is not None:
                    previous_sample=next((r for r in reuse[3]['samples'] if r['domain']==d and r['sample_id']==sample['sample_id']),None)
                    if previous_sample and previous_sample['image_sha256']==sample['image_sha256']:
                        ip.link_reused(a,reuse_q_root/d/(sample['sample_id']+'.npz'),qpath,reuse[2])
                        qrecord=st.restore(qpath,identity)
                if qrecord:
                    features,qr=st.arrays(qpath,identity);qmeta=qr['metadata']
                    if qmeta['image_sha256']!=sample['image_sha256']:raise ValueError('query cache identity differs')
                    positions=features.pop('positions')
                else:
                    if model is None:model,_=runtime.host(a)
                    with ip.Data(Path(catalog['data_root'])) as data:
                        features,geometry=extraction(model,data.image(sample),a.device)
                    calls+=1;positions=geometry.pop('positions')
                    qmeta=dict(sample,**geometry,query_dtype='float32',units='raw unnormalized 1280D')
                    if a.query_mode=='disk':
                        st.seal(qpath,identity,dict(features,positions=positions),qmeta,compressed=False)
                if source is None:source=runtime.source_banks(a)
                for l in LAYERS:
                    v=features[f'layer_{l}']
                    if v.dtype!=np.float32 or v.shape!=(len(positions),1280) or not np.isfinite(v).all():
                        raise ValueError('query must retain original complete FP32 1280D features')
                if not contexts:
                    for l in LAYERS:
                        pool,limits,_=graphs[l];nodes,*_=graph.layout(banks[l])
                        main=choice['main']['selected'];plain=choice['plain']['selected']
                        configs=[tuple(choice['paired_config']) if main else None,
                            tuple(main['config']) if main else None,tuple(plain['config']) if plain else None]
                        active=np.asarray([graph.masks(pool,limits,[c])[0] if c else np.zeros(len(pool['edges']),bool) for c in configs])
                        contexts[l]=(nodes,active)
                        if a.device!='cpu':
                            import torch
                            if len(pool['edges']) and active.any():searches[l]=PreparedSearch(nodes,pool['edges'],active,a.device,edge_chunk=a.edge_chunk)
                            else:searches[l]=None
                            device_banks[l]={sid:torch.as_tensor(v,device=a.device) for sid,v in banks[l].items()}
                distances={b:[] for b in BRANCHES};diagnostics={};audits=[]
                for l in LAYERS:
                    pool,limits,grec=graphs[l]
                    values,extra,audit=layer_branches(a,d,l,features[f'layer_{l}'],source[l],banks[l],pool,limits,choice,
                        prepared=searches.get(l),device_banks=device_banks.get(l),nodes=contexts[l][0],active=contexts[l][1])
                    # Context arrays are immutable for every query in this domain.
                    for b in BRANCHES:distances[b].append(values[b])
                    for key,value in extra.items():diagnostics.setdefault(key,[]).append(value)
                    audits.append(audit)
                arrays={b:np.stack(v) for b,v in distances.items()}
                arrays.update({k:np.stack(v) for k,v in diagnostics.items()});arrays['positions']=positions
                if positions.dtype!=np.int64 or not np.array_equal(np.sort(positions),np.arange(np.prod(qmeta['grid']))):
                    raise ValueError('complete original physical positions required')
                scores=[]
                shape=(sample['image_height']//4,sample['image_width']//4)
                for b in BRANCHES:
                    current,peak,local=statistics(arrays[b],positions,tuple(qmeta['grid']),shape,med,scale)
                    scores.append(dict(branch=b,CH1=ch1_score(peak,local,state),CURRENT=current))
                meta=dict(sample,grid=qmeta['grid'],output_shape=list(shape),selection_sha256=choice_sha,
                    source_mad_state_sha256=st.sha(a.cache_root/'source_patch_mad.npz'),
                    image_mad_state_sha256=st.sha(a.run_root/'image_calibration_state.json'),
                    graph_sha256={str(l):graphs[l][2]['sha256'] for l in LAYERS},scores=scores,audits=audits,
                    query_mode=a.query_mode,query_cache_sha256=st.sha(qpath) if a.query_mode=='disk' else None,
                    GT_used=False,labels_used=False)
                record=st.seal(path,identity,arrays,meta)
                new_images+=1
                del features,arrays,diagnostics
                print(f'[fixed graph query] {d}/{sample["sample_id"]} seven_branches_same_Q',flush=True)
            records[d].append(dict(sample_id=sample['sample_id'],sha256=record['sha256'],file=str(path.resolve()),
                record_sha256=st.sha(path.with_suffix('.json'))))
        value=dict(status='predictions_frozen',identity=identity,domain=d,records=records[d],
            selection_sha256=choice_sha,images=len(records[d]),seed=a.seed,evaluation_scope=a.evaluation_scope)
        manifest=a.run_root/'prediction_manifests'/f'{d}.json'
        if len(records[d])==sum(s['domain']==d for s in catalog['samples']):
            if manifest.exists() and st.read_json(manifest)!=value:raise ValueError('frozen prediction manifest differs')
            st.write_json(manifest,value)
        else:
            st.write_json(a.run_root/'prediction_progress'/f'{d}.json',dict(value,status='partial_predictions'))
        # Do not retain previous domain's graph/Target tensors during the next transfer.
        searches.clear();device_banks.clear();contexts.clear()
        import gc
        gc.collect()
    images=sum(map(len,records.values()))
    query_bytes=sum(p.stat().st_size for d in ip.DOMAINS for p in (a.query_cache_root/d).glob('*.npz')) if a.query_mode=='disk' else 0
    prediction_bytes=sum(p.stat().st_size for d in ip.DOMAINS for p in (a.prediction_root/d).glob('*.npz'))
    return dict(status='all_predictions_frozen' if images==len(catalog['samples']) else 'partial_predictions',
        query_forward_calls=calls,new_images=new_images,reused_images=reused,query_mode=a.query_mode,images=images,
        query_cache_logical_bytes=query_bytes,prediction_logical_bytes=prediction_bytes,
        byte_scope='logical retained NPZ sizes; links may share physical source storage')


def verified(a,domains=None):
    """Require every declared sample and branch before opening any GT."""
    samples=ip.catalog(a)['samples'];identity=ip.identity(a)
    selected=domains if domains is not None else ip.DOMAINS
    for d in selected:
        selection(a,d)
        manifest=st.read_json(a.run_root/'prediction_manifests'/f'{d}.json')
        wanted=[s for s in samples if s['domain']==d]
        if manifest['status']!='predictions_frozen' or manifest['identity']!=identity or manifest['images']!=len(wanted):
            raise ValueError('incomplete frozen predictions')
        if len(manifest['records'])!=len(wanted) or {s['sample_id'] for s in wanted}!={r['sample_id'] for r in manifest['records']}:
            raise ValueError('prediction population differs')
        for sample in wanted:
            record=next(r for r in manifest['records'] if r['sample_id']==sample['sample_id'])
            x,m,r=ip.verify_prediction(a,sample)
            p=Path(record['file'])
            if record['sha256']!=r['sha256'] or record['record_sha256']!=st.sha(p.with_suffix('.json')):
                raise ValueError('prediction changed after freeze')
            n=int(np.prod(m['grid']))
            for b in BRANCHES:
                if x[b].shape!=(4,n) or x[b].dtype!=np.float32 or not np.isfinite(x[b]).all() or (x[b]<0).any():
                    raise ValueError('invalid frozen branch distances')
    return [s for s in samples if s['domain'] in selected]
