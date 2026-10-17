"""Explicit original-host supplementation. All model imports are inside run()."""
from __future__ import annotations
import json
import shutil
import time
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from . import inputs as ip
from .method import LAYERS,add_geometry,scatter
from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import statistics
from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic.numerics import check
from .verification import historical_score, audit


def verify_scores(ctx,sample,source,union,positions,meta,checks):
    for branch,value,key in (('SOURCE_CURRENT',source,'source_score'),('UNION_CURRENT',union,'union_score')):
        current=statistics(value,positions,tuple(meta['grid']),tuple(meta['output_shape']),ctx['medians'],ctx['scales'])[0]
        historical_score(checks,sample,branch,current,sample[key])
    if not all(r['verified'] for r in checks[-2:]):raise ValueError('Source/Union CURRENT exceeds frozen historical compatibility tolerance 1e-4')
    from .references import verify
    verify(ctx,sample,source,union,positions,meta,checks)


def verify_current(ctx,sample,raw,meta,checks):
    union=np.minimum(raw['source_distance'],raw['target_distance'].min(0))
    verify_scores(ctx,sample,raw['source_distance'],union,raw['positions'],meta,checks)


def load_cached(ctx,root,record,sample):
    arrays,meta=ip.load_npz(Path(root)/ip.safe(record['file']),record['sha256'])
    ip.validate_raw(ctx,sample,arrays,meta)
    return arrays,meta


def query_arrays(ctx,directory,record,sample):
    arrays,meta=ip.load_npz(Path(directory)/ip.safe(record['file']),record['sha256'])
    if (meta.get('extraction_identity')!=ctx['extraction_identity'] or meta.get('image_sha256')!=sample['image_sha256']
            or meta.get('domain')!=sample['domain'] or meta.get('sample_id')!=sample['sample_id']
            or meta.get('relative_path')!=sample['relative_path'] or meta.get('layers')!=list(LAYERS)
            or meta.get('dtype')!='float32' or meta.get('feature_units')!='original_1280d_no_normalization'):
        raise ValueError('untrusted query feature provenance')
    grid=tuple(meta['grid']);positions=arrays['positions'];n=int(np.prod(grid))
    scatter(np.zeros((4,n),np.float32),positions,grid)
    for layer in LAYERS:
        a=arrays[f'layer_{layer}']
        if a.dtype!=np.float32 or a.shape!=(n,1280) or not np.isfinite(a).all():raise ValueError('invalid original Q cache')
    if meta['output_shape']!=[sample['image_height']//4,sample['image_width']//4]:raise ValueError('query output dimensions differ')
    return arrays,meta


def run(ctx,domains,*,device_name='cuda:0',query_features=None,support_features=None,reuse_root=None,capture_min_free_vram_gib=20):
    from .runner import capture_vram_bytes
    model_minimum=capture_vram_bytes(capture_min_free_vram_gib)
    root=ctx['paths']['cache_root'];started=time.perf_counter();state=ip.cache_state(ctx)
    # Validate all old hashes BEFORE creating/resetting any status.
    if not root.exists():
        minimal=dict(ctx,paths={k:v for k,v in ctx['paths'].items() if k!='cache_root'})
        ip.new_output(minimal,root,'capture')
    old_checks=ip.csv_file(root/'capture_verification.csv') if (root/'capture_verification.csv').is_file() else []
    # Existing CSV booleans must remain booleans for completion checks.
    checks=[dict(r,verified=r['verified'] in ('True','true','1')) for r in old_checks]
    if any(not r['verified'] for r in checks):raise ValueError('prior numerical verification failed; preserve cache and investigate before resuming')
    q_manifest=ip.external_manifest(ctx,query_features,'original_query_features_v1')
    s_manifest=ip.external_manifest(ctx,support_features,'original_support_features_v1')
    for role,path,manifest in (('query_features',query_features,q_manifest),('support_features',support_features,s_manifest)):
        if manifest is not None:
            checksum=ip.sha(Path(path)/'manifest.json')
            if role in state['sources'] and state['sources'][role]!=checksum:raise ValueError('previously enrolled provenance manifest changed')
            state['sources'][role]=checksum
    counters=dict(query_extraction_calls=0,support_extraction_calls=0,query_cache_reuses=0,
                  complete_capture_reuses=0,exact_source_calls=0,exact_target_calls=0)
    invocation=dict(started_utc=datetime.now(timezone.utc).isoformat(),domains=list(domains),counts=counters,timing_seconds={},
                    resource_admission=dict(model_min_free_VRAM_bytes=model_minimum,retrieval_only_min_free_VRAM_bytes=8*2**30,
                                            batch_changed=False,precision_changed=False,automatic_fallback=False))
    state.setdefault('invocations',[]).append(invocation)
    def commit():
        state.update(tool=ip.TOOL,identity=ctx['identity'],status='partial_cache_verified',last_invocation_counts=counters)
        state['historical_score_audit']=audit(checks)
        ip.write_json(root/'cache_status.json',state);ip.write_csv(root/'capture_verification.csv',checks)
    commit()
    try:
        if reuse_root is not None:
            previous=ip.cache_state(ctx,reuse_root)
            for r in previous['records']:
                if r['domain'] not in domains or any((x['domain'],x['sample_id'])==(r['domain'],r['sample_id']) for x in state['records']):continue
                sample=next(s for s in ctx['catalog']['domains'][r['domain']]['samples'] if s['sample_id']==r['sample_id'])
                arrays,meta=load_cached(ctx,reuse_root,r,sample);verify_current(ctx,sample,arrays,meta,checks)
                file=root/ip.safe(r['file']);file.parent.mkdir(parents=True,exist_ok=True)
                if file.exists():raise ValueError('unrecorded destination cache')
                shutil.copyfile(Path(reuse_root)/r['file'],file)
                state['records'].append(dict(r,action='reuse_verified_complete_geometry',source_sha256=r['sha256']))
                counters['complete_capture_reuses']+=1;commit()
        pending={domain:[s for s in ctx['catalog']['domains'][domain]['samples'] if not any(r['domain']==domain and r['sample_id']==s['sample_id'] for r in state['records'])] for domain in domains}
        # Finished domains need no model, memory file or support feature existence.
        if any(pending.values()):
            from .runner import resources
            estimate=resources(ctx)['domains'];required=2*2**30
            for d,selected in pending.items():
                if not selected:continue
                required+=int(estimate[d]['approximate_uncompressed_raw_bytes']*len(selected)/len(ctx['catalog']['domains'][d]['samples']))
                required+=int(estimate[d]['approximate_support_feature_bytes']*sum(not any(r['domain']==d and r['support_index']==i for r in state['supports']) for i in range(1,9))/8)
            if shutil.disk_usage(root).free<required:raise OSError(f'capture requires estimated {required} free disk bytes; no precision/budget reduction')
            if not ctx['paths']['memory'].is_file() or ip.sha(ctx['paths']['memory'])!=ctx['catalog']['memory_sha256']:
                raise FileNotFoundError('original Source memory missing/mismatched; rebuilding forbidden')
            q_records={(r['domain'],r['sample_id']):r for r in (q_manifest or {}).get('records',[])}
            external_support={(r['domain'],r['support_index']):r for r in (s_manifest or {}).get('records',[])}
            needs_query_model=any((d,s['sample_id']) not in q_records for d,v in pending.items() for s in v)
            needs_support_model=any(not any(r['domain']==d and r['support_index']==i for r in state['supports']) and (d,i) not in external_support
                                    for d,v in pending.items() if v for i in range(1,9))
            need_model=needs_query_model or needs_support_model
            if need_model:
                for key,file in (('checkpoint',None),('dinov3','hubconf.py'),('official','config.json')):
                    path=ctx['paths'][key] if file is None else ctx['paths'][key]/file
                    if not path.is_file():raise FileNotFoundError('original model asset missing: '+str(path))
                if ip.sha(ctx['paths']['checkpoint'])!=ctx['catalog']['checkpoint_sha256']:raise ValueError('checkpoint changed')
            # Lazy dependencies, only after concrete missing assets have been reported.
            import torch
            from DINOv3.MADEqual.aebad_s_union_support_loo.analysis import CaptureLog,FORMAT
            from DINOv3.MADEqual.aebad_s_target_fewshot import runner as original
            from DINOv3.MADEqual.robustad_target_fewshot_p0 import scoring,support
            from DINOv3.MADEqual.aebad_s_sparc_p0.runtime import load_host
            from DINOv3.relation_reliability.superadd_host import _exact_top1_chunked
            device=torch.device(device_name)
            if device.type!='cuda' or not torch.cuda.is_available():raise RuntimeError('explicit capture requires CUDA; no CPU model fallback')
            device=torch.device('cuda',device.index if device.index is not None else torch.cuda.current_device());torch.cuda.set_device(device)
            minimum=model_minimum if need_model else 8*2**30
            free_vram,total_vram=torch.cuda.mem_get_info(device)
            invocation['resource_admission'].update(required_free_VRAM_bytes=minimum,free_VRAM_before_loading_bytes=int(free_vram),
                                                    total_VRAM_bytes=int(total_vram),device=str(device),model_required=need_model)
            commit()
            print(f'[crossdomain admission] free={free_vram/2**30:.2f} GiB required={minimum/2**30:.2f} GiB; original batch/float32 unchanged',flush=True)
            if free_vram<minimum:raise MemoryError(f'insufficient free VRAM: selected admission requires {minimum/2**30:g} GiB; no batch/precision reduction or fallback')
            if ip.common.available_ram()<12*2**30:raise MemoryError('capture needs 12 GiB free host RAM for full supports and host')
            original._seed_everything();torch.set_float32_matmul_precision('highest')
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
            holder=SimpleNamespace()
            tick=time.perf_counter()
            if not original._load_memory(ctx['paths']['memory'],holder,source_count=521,device=device):raise FileNotFoundError('Source memory missing')
            model=None
            if need_model:
                host_config=ip.read_json(ctx['paths']['formal_dir']/'memory_manifest.json')['host_config']
                model,device=load_host(dict(paths={k:str(v) for k,v in ctx['paths'].items()},frozen=dict(precision=ip.PRECISION,host_config=host_config)),device_name)
                actual=dict(brightness_factor=model.preprocessing.min_brightness_factor,resize_factor=model.preprocessing.resize_factor,
                    patch_size=model.patch_exec.patch_size,patch_overlap=model.patch_exec.patch_overlap,model_patch_size=model.patch_exec.model_patch_size,
                    evaluation_downscale=model.evaluation_downscale,dino_training=model.backbone.dino.training,
                    trainable_parameters=sum(p.requires_grad for p in model.backbone.dino.parameters()))
                if actual!=ip.QUERY or model.preprocessing.max_brightness_factor!=1:raise ValueError('host differs from frozen unaugmented query protocol')
            invocation['timing_seconds']['Source_memory_and_host_loading']=time.perf_counter()-tick
            with ip.Data(ctx['paths']['data']) as dataset:
                for domain in domains:
                    if not pending[domain]:continue
                    features=[];support_meta=[]
                    for sample in ctx['catalog']['domains'][domain]['supports']:
                        tick=time.perf_counter()
                        index=sample['support_index'];record=next((r for r in state['supports'] if r['domain']==domain and r['support_index']==index),None)
                        if record is None:
                            external=external_support.get((domain,index))
                            if external:
                                path=Path(support_features)/ip.safe(external['file'])
                                grid,layers=ip.support_arrays(path,sample)
                                record=dict(domain=domain,support_index=index,file=f'support_features/{domain}/{index:03d}.npz',
                                    storage='external',external_path=str(path.resolve()),sha256=external['sha256'],action='reuse_verified_original_support')
                            else:
                                if model is None:raise RuntimeError('support extraction explicitly requires original host')
                                layers,grid=support._extract_four_layers(model,dataset.image(sample),device=device,brightness_factor=None)
                                path=root/f'support_features/{domain}/{index:03d}.npz'
                                if path.exists():raise ValueError('support orphan cannot be re-approved')
                                support._atomic_npz(path,protocol_id=np.asarray(original.PROTOCOL_ID),relative_path=np.asarray(sample['relative_path']),
                                    grid=np.asarray(grid,np.int32),**{f'layer_{l}':layers[l] for l in LAYERS})
                                record=dict(domain=domain,support_index=index,file=path.relative_to(root).as_posix(),sha256=ip.sha(path),action='original_unaugmented_support_extraction')
                                counters['support_extraction_calls']+=1
                            record.update(relative_path=sample['relative_path'],image_sha256=sample['image_sha256'],grid=list(grid))
                            state['supports'].append(record);commit()
                        else:
                            path=ip.record_path(root,record)
                            if not path.is_file():raise FileNotFoundError('recorded support missing; restore its original file before continuing capture')
                            grid,layers=ip.support_arrays(path,sample)
                        features.append(SimpleNamespace(grid=grid,layers=layers))
                        support_meta.append(dict(support_index=index,relative_path=sample['relative_path'],image_sha256=sample['image_sha256'],
                                                 feature_sha256=record['sha256'],grid=list(grid)))
                        timing=invocation['timing_seconds'];timing['support_loading_or_extraction']=timing.get('support_loading_or_extraction',0)+time.perf_counter()-tick
                    target_tensors=original._support_tensors(features,device=device)
                    geometry_features={l:[f.layers[l] for f in features] for l in LAYERS}
                    for sample in pending[domain]:
                        tick=time.perf_counter()
                        log=CaptureLog();qrecord=q_records.get((domain,sample['sample_id']));observed=None
                        if qrecord:
                            q,qmeta=query_arrays(ctx,query_features,qrecord,sample)
                            log.geometry(tuple(qmeta['grid']),q['positions'],tuple(qmeta['processed_shape']))
                            with torch.inference_mode():
                                for l in LAYERS:
                                    query=torch.from_numpy(q[f'layer_{l}']).to(device)
                                    d,idx=_exact_top1_chunked(query,holder.prototype_embeddings[l]);log.nearest('source',l,0,d.numpy().astype(np.float32)/1280,idx.numpy())
                                    for j,bank in enumerate(target_tensors[l],1):
                                        d,idx=_exact_top1_chunked(query,bank);log.nearest('target',l,j,d.numpy().astype(np.float32)/1280,idx.numpy())
                            action='verified_Q_original_Exact_retrieval';counters['query_cache_reuses']+=1
                        else:
                            if model is None:raise RuntimeError('no original host for missing Q')
                            observed=scoring.observe_fewshot_pil(model,dataset.image(sample),official_banks=holder.prototype_embeddings,
                                source_support=None,target_support=target_tensors,device=device,nearest_observer=log.nearest,geometry_observer=log.geometry)
                            action='same_call_original_full_image_capture';counters['query_extraction_calls']+=1
                        raw=log.arrays();counters['exact_source_calls']+=4;counters['exact_target_calls']+=32
                        timing=invocation['timing_seconds'];timing['query_extraction_or_Q_loading_and_exact_retrieval']=timing.get('query_extraction_or_Q_loading_and_exact_retrieval',0)+time.perf_counter()-tick
                        if observed is not None:
                            check(checks,sample['sample_id'],'same_call_Source',scatter(raw['source_distance'],raw['positions'],log.grid),observed.official_raw,atol=1e-7,rtol=1e-6,domain=domain)
                            check(checks,sample['sample_id'],'same_call_Union',scatter(np.minimum(raw['source_distance'],raw['target_distance'].min(0)),raw['positions'],log.grid),observed.target_raw[8],atol=1e-7,rtol=1e-6,domain=domain)
                            if not all(r['verified'] for r in checks[-2:]):raise ValueError('same-call logging mismatch')
                        tick=time.perf_counter();raw,diagnostics=add_geometry(raw,geometry_features)
                        timing['SEGMENT_geometry']=timing.get('SEGMENT_geometry',0)+time.perf_counter()-tick
                        meta=dict(format=FORMAT,tool=ip.TOOL,identity=ctx['identity'],domain=domain,sample_id=sample['sample_id'],relative_path=sample['relative_path'],
                            image_sha256=sample['image_sha256'],grid=list(log.grid),output_shape=[sample['image_height']//4,sample['image_width']//4],
                            geometry=dict(processed_shape=list(log.processed_shape)),supports=support_meta,source_rows=[100000]*4,
                            distance_unit=ip.PROTOCOL['distance_unit'],action=action,query_feature_sha256=qrecord['sha256'] if qrecord else None,
                            geometry_diagnostics=diagnostics,extraction_identity=ctx['extraction_identity'])
                        ip.validate_raw(ctx,sample,raw,meta)
                        file=f'raw_cache/{domain}/{sample["sample_id"]}.npz';ip.save_npz(root/file,raw,meta)
                        record=dict(domain=domain,sample_id=sample['sample_id'],file=file,sha256=ip.sha(root/file),action=action)
                        state['records'].append(record);commit() # preserve even a later failed reproduction
                        tick=time.perf_counter();verify_current(ctx,sample,raw,meta,checks);commit()
                        timing['CURRENT_and_optional_raw_reproduction']=timing.get('CURRENT_and_optional_raw_reproduction',0)+time.perf_counter()-tick
                        print(f'[crossdomain capture] {domain} {sample["sample_id"]} {action}',flush=True)
                    del target_tensors,features,geometry_features;torch.cuda.empty_cache()
            ip.write_json(root/'capture_environment.json',dict(torch=torch.__version__,CUDA=torch.version.cuda,device=torch.cuda.get_device_name(device),
                precision=ip.PRECISION,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device)))
        # Recheck reused records too; no resetting old per-image hash history.
        for domain in domains:
            for sample in ctx['catalog']['domains'][domain]['samples']:
                record=next(r for r in state['records'] if r['domain']==domain and r['sample_id']==sample['sample_id'])
                arrays,meta=load_cached(ctx,root,record,sample);verify_current(ctx,sample,arrays,meta,checks)
        if ip.context(ctx['paths'])['identity']!=ctx['identity']:raise ValueError('joint inputs/code changed during capture')
        complete=all(sum(r['domain']==d for r in state['records'])==len(ctx['catalog']['domains'][d]['samples']) for d in ip.DOMAINS)
        commit();state.update(status='capture_complete' if complete else 'partial_cache_verified',seconds=time.perf_counter()-started,
            jointly_frozen_domains=list(ip.DOMAINS),GT_decoded=False,default_query_features_persisted=False)
        from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.evaluation import peak_rss
        invocation.update(seconds=time.perf_counter()-started,process_peak_rss_bytes=peak_rss(),
            retained_cache_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()))
        ip.write_json(root/'cache_status.json',state);return root
    except Exception as error:
        commit();state.update(status='capture_failed',error=f'{type(error).__name__}: {error}');ip.write_json(root/'cache_status.json',state)
        raise
