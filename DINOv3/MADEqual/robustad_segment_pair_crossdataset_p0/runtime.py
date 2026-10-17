"""Explicit CUDA only; load original banks, never call a memory/MAD builder."""
from __future__ import annotations
import gc
import json
import time
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager
import numpy as np
from . import inputs as ip
from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.method import add_geometry,scatter
from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.verification import historical_score
from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import statistics
from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic.numerics import check


def validate_source(arrays,meta,sample):
    grid=tuple(meta['grid']);n=int(np.prod(grid))
    scatter(np.zeros((4,n),np.float32),arrays['positions'],grid)
    if meta['output_shape']!=[sample['image_height']//4,sample['image_width']//4]:raise ValueError('output geometry changed')
    a=arrays['source_distance']
    if a.dtype!=np.float32 or a.shape!=(4,n) or not np.isfinite(a).all() or np.any(a<0):raise ValueError('invalid Source distances')
    if 'source_index' in arrays:
        v=arrays['source_index']
        if v.dtype!=np.int64 or v.shape!=(4,n) or np.any(v<0) or np.any(v>=100000):raise ValueError('invalid Source indices')


def verify_target(ctx,sample,raw,meta,checks):
    from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.method import validate_geometry
    validate_source(raw,meta,sample);validate_geometry(raw)
    n=len(raw['positions'])
    for k in ('target_distance','target_index'):
        a=raw[k]
        if a.shape!=(8,4,n) or not np.isfinite(a).all() or np.any(a<0):raise ValueError('invalid per-support retrieval')
    if raw['target_distance'].dtype!=np.float32 or raw['target_index'].dtype!=np.int64:raise ValueError('target retrieval dtype differs')
    expected=ctx['cat']['shifts'][sample['shift']]['supports']
    if [(s['support_index'],s['sample_id'],s['image_sha256']) for s in meta['supports']]!=[(s['support_index'],s['sample_id'],s['image_sha256']) for s in expected]:raise ValueError('K8 identity/order differs')
    for i,s in enumerate(meta['supports']):
        if np.any(raw['target_index'][i]>=int(np.prod(s['grid']))):raise ValueError('support index out of bounds')
    union=np.minimum(raw['source_distance'],raw['target_distance'].min(0))
    if 'historical_Source_distance_audit' in meta and not any(r['sample_id']==sample['sample_id'] and r['check_kind']=='historical_Source_distance' for r in checks):
        checks.append(meta['historical_Source_distance_audit'].copy())
    for kind,value,key in (('SOURCE_CURRENT',raw['source_distance'],'source_score'),('UNION_CURRENT',union,'union_score')):
        actual=statistics(value,raw['positions'],tuple(meta['grid']),tuple(meta['output_shape']),ctx['medians'],ctx['scales'])[0]
        historical_score(checks,sample,kind,actual,sample[key]);checks[-1]['category']=ctx['category']
    if not all(r['verified'] for r in checks[-2:]):raise ValueError('historical Source/Union exceeds fixed 1e-4 compatibility bound')


def load_host(ctx,device_name,min_free_gib=20):
    if not ctx['bank'].is_file():raise FileNotFoundError('original category bank missing; rebuilding forbidden: '+str(ctx['bank']))
    if ip.sha(ctx['bank'])!=ctx['frozen']['bank_sha256']:raise ValueError('bank changed since first enrollment')
    for name,suffix in (('checkpoint',None),('dinov3','hubconf.py'),('official','config.json')):
        p=ctx['paths'][name] if suffix is None else ctx['paths'][name]/suffix
        if not p.is_file():raise FileNotFoundError(p)
    import torch
    from DINOv3.MADEqual.aebad_s_sparc_p0.runtime import load_host as original_host
    from DINOv3.relation_oracle_cache.extract import install_category_banks
    device=torch.device(device_name)
    if device.type!='cuda' or not torch.cuda.is_available():raise RuntimeError('explicit recovery/capture requires CUDA')
    device=torch.device('cuda',device.index if device.index is not None else torch.cuda.current_device());torch.cuda.set_device(device)
    if torch.cuda.mem_get_info(device)[0]<min_free_gib*2**30 or ip.available_ram()<8*2**30:raise MemoryError('insufficient CUDA/RAM for original complete host; no fallback')
    model,device=original_host(dict(paths={k:str(v) for k,v in ctx['paths'].items()},
        frozen=dict(precision=ip.PRECISION,host_config=ctx['catalog']['host_config'])),str(device))
    # Inspect dtype/shape before the original loader's float32 conversion.
    with np.load(ctx['bank'],allow_pickle=False) as f:
        for l in ip.LAYERS:
            a=f[str(l)]
            if a.dtype!=np.float32 or a.shape!=(100000,1280) or not np.isfinite(a).all():raise ValueError('invalid original Source bank')
    install_category_banks(model,ctx['bank'],expected_entries={l:100000 for l in ip.LAYERS},device=device)
    return model,device


@contextmanager
def profile_original(module,device,timing,prefix):
    """Time original extraction/Exact functions with identical arguments/results."""
    import torch
    from unittest.mock import patch
    def wrapped(original,key):
        def call(*args,**kwargs):
            if getattr(device,'type',None)=='cuda':torch.cuda.synchronize(device)
            tick=time.perf_counter()
            result=original(*args,**kwargs)
            if getattr(device,'type',None)=='cuda':torch.cuda.synchronize(device)
            timing[key]=timing.get(key,0.)+time.perf_counter()-tick
            return result
        return call
    with (patch.object(module,'_extract_processed_batch',wrapped(module._extract_processed_batch,prefix+'_feature_extraction')),
          patch.object(module,'_exact_top1_chunked',wrapped(module._exact_top1_chunked,prefix+'_Exact_retrieval'))):
        yield


def source_observation(ctx,sample,model,device,timing=None):
    from DINOv3.relation_reliability.superadd_host import observe_pil,preprocessed_spatial_shape
    from DINOv3.relation_reliability import superadd_host
    with profile_original(superadd_host,device,timing if timing is not None else {},'Source'):
        observed=observe_pil(model,ip.image(ctx,sample),device=device)
    raw=observed.raw_distances;grid=tuple(observed.grid)
    arrays=dict(positions=np.arange(int(np.prod(grid)),dtype=np.int64),source_distance=raw.reshape(4,-1).copy(),
                source_index=observed.top1_indices.reshape(4,-1).astype(np.int64))
    meta=dict(grid=list(grid),processed_shape=list(preprocessed_spatial_shape(model,sample['image_height'],sample['image_width'])),
              output_shape=[sample['image_height']//4,sample['image_width']//4],action='original_Source_observe_pil')
    validate_source(arrays,meta,sample)
    if sample.get('source_role')=='calibration':
        from DINOv3.MADEqual.image_granularity_calibration.scoring import interpolated_layer_maxima
        from DINOv3.MADEqual.connected_region.scoring import local_mean
        from DINOv3.relation_oracle_cache.extract import mad_calibrated
        _,peak=interpolated_layer_maxima(raw,ctx['medians'],ctx['scales'],tuple(meta['output_shape']))
        arrays['original_peak']=peak
        arrays['original_local']=np.asarray([local_mean(a) for a in mad_calibrated(raw,ctx['medians'],ctx['scales'])],np.float32)
    return arrays,meta


def feature_manifest(ctx,state):
    directory=ctx['paths'].get('feature_root')
    if directory is None:return {}
    root=directory/ctx['category'];path=root/'manifest.json';manifest=ip.read_json(path)
    if manifest.get('format')!='original_query_support_features_v1' or manifest.get('status')!='verified' or manifest.get('category')!=ctx['category'] or manifest.get('extraction_identity')!=ctx['extraction_identity']:raise ValueError('untrusted original Q/support feature manifest')
    previous=state.setdefault('external_manifests',{}).get('features')
    checksum=ip.sha(path)
    if previous is not None and previous!=checksum:raise ValueError('previously enrolled feature manifest changed')
    state['external_manifests']['features']=checksum
    result={}
    for r in manifest['records']:
        k=(r['kind'],r['sample_id'])
        if k in result:raise ValueError('duplicate feature provenance')
        result[k]=dict(r,path=root/ip.safe(r['file']))
    return result


def load_features(ctx,record,sample):
    path=record['path']
    if ip.sha(path)!=record['sha256']:raise ValueError('recorded external features changed')
    with np.load(path,allow_pickle=False) as f:
        meta=json.loads(str(f['metadata'].item()));a={k:f[k].copy() for k in f.files if k!='metadata'}
    if any(meta.get(k)!=v for k,v in dict(category=ctx['category'],sample_id=sample['sample_id'],image_sha256=sample['image_sha256'],extraction_identity=ctx['extraction_identity'],layers=list(ip.LAYERS),dtype='float32',feature_units='original_1280d_no_normalization').items()):raise ValueError('feature header identity differs')
    grid=tuple(meta['grid']);n=int(np.prod(grid))
    scatter(np.zeros((4,n),np.float32),a['positions'],grid)
    for l in ip.LAYERS:
        if a[f'layer_{l}'].dtype!=np.float32 or a[f'layer_{l}'].shape!=(n,1280) or not np.isfinite(a[f'layer_{l}']).all():raise ValueError('invalid original feature rows')
    if meta['output_shape']!=[sample['image_height']//4,sample['image_width']//4]:raise ValueError('feature output geometry differs')
    return a,meta


def capture(ctx,shifts,scope,device_name,min_free_gib=20):
    """A=own complete ledger, B=trusted Q/support, C=explicit original model."""
    started=time.perf_counter();state=ip.ledger(ctx);checks=[];model=None;device=None;resident_banks=None
    refs=feature_manifest(ctx,state)
    source=ctx['cat']['source_evaluation'] if scope in ('source','both') else []
    targets=[s for g in shifts for s in ctx['cat']['shifts'][g]['samples']] if scope in ('target','both') else []
    done={(r['kind'],r['sample_id']):r for r in state['records']}
    counters=dict(query_forward_calls=0,support_forward_calls=0,Source_evaluation_forward_calls=0,trusted_query_feature_reuses=0)
    source_pending=[]
    timings=dict(model_and_bank_loading=0.,query_extraction_and_exact=0.,support_loading_and_extraction=0.,geometry=0.)
    def host():
        nonlocal model,device,resident_banks
        if model is None:
            if resident_banks is not None:
                import torch
                resident_banks=None;gc.collect();torch.cuda.empty_cache()
            tick=time.perf_counter();model,device=load_host(ctx,device_name,min_free_gib);timings['model_and_bank_loading']+=time.perf_counter()-tick
        return model,device
    try:
        for s in source:
            if ('source',s['sample_id']) not in done:
                from .calibration import legacy_raw
                value=legacy_raw(ctx,s,state)
                if value is None:
                    if not (ctx['paths']['data']/s['relative_path']).is_file():
                        source_pending.append(s['sample_id']);print('ARD_PENDING: missing Source image '+s['sample_id'],flush=True);continue
                    mm,dd=host();value=source_observation(ctx,s,mm,dd,timings);counters['Source_evaluation_forward_calls']+=1
                a,meta=value;validate_source(a,meta,s);r=ip.save_cache(ctx,state,'source',s,a,meta);done[('source',s['sample_id'])]=r
            print(f'[Robust capture Source] {ctx["category"]} {s["sample_id"]}',flush=True)
        for shift in shifts if targets else []:
            pending=[s for s in targets if s['shift']==shift and ('target',s['sample_id']) not in done]
            if not pending:continue
            features=[];support_meta=[];tick=time.perf_counter()
            for s in ctx['cat']['shifts'][shift]['supports']:
                r=done.get(('support',s['sample_id']))
                if r:
                    a,meta=ip.load_cache(ctx,r,s);grid=tuple(meta['grid']);layers={l:a[f'layer_{l}'] for l in ip.LAYERS}
                else:
                    external=refs.get(('support',s['sample_id']))
                    if external:
                        a,meta=load_features(ctx,external,s);grid=tuple(meta['grid'])
                        layers={l:np.empty_like(a[f'layer_{l}']) for l in ip.LAYERS}
                        for l in ip.LAYERS:layers[l][a['positions']]=a[f'layer_{l}']
                    else:
                        from DINOv3.MADEqual.robustad_target_fewshot_p0.support import _extract_four_layers
                        mm,dd=host();layers,grid=_extract_four_layers(mm,ip.image(ctx,s),device=dd,brightness_factor=None);counters['support_forward_calls']+=1
                    r=ip.save_cache(ctx,state,'support',s,{f'layer_{l}':layers[l] for l in ip.LAYERS},dict(grid=list(grid),feature_units='original_1280d_no_normalization'))
                    done[('support',s['sample_id'])]=r
                for l in ip.LAYERS:
                    if layers[l].dtype!=np.float32 or layers[l].shape!=(int(np.prod(grid)),1280) or not np.isfinite(layers[l]).all():raise ValueError('invalid support features')
                features.append(SimpleNamespace(layers=layers,grid=grid))
                support_meta.append(dict(support_index=s['support_index'],sample_id=s['sample_id'],image_sha256=s['image_sha256'],grid=list(grid),feature_sha256=r['sha256']))
                print(f'[Robust support] {ctx["category"]} {shift} {s["support_index"]}/8',flush=True)
            timings['support_loading_and_extraction']+=time.perf_counter()-tick
            import torch
            from DINOv3.MADEqual.aebad_s_union_support_loo.analysis import CaptureLog
            from DINOv3.MADEqual.robustad_target_fewshot_p0.scoring import observe_fewshot_pil
            from DINOv3.relation_reliability.superadd_host import _exact_top1_chunked
            if model is None and resident_banks is None:
                # B still needs the original bank, but never constructs a model.
                device=torch.device(device_name);torch.cuda.set_device(device)
                if device.type!='cuda' or not torch.cuda.is_available():raise RuntimeError('Exact retrieval requires explicit CUDA')
                if torch.cuda.mem_get_info(device)[0]<8*2**30 or ip.available_ram()<8*2**30:raise MemoryError('insufficient retrieval-only RAM/VRAM')
                torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
                if ip.sha(ctx['bank'])!=ctx['frozen']['bank_sha256']:raise ValueError('bank changed')
                with np.load(ctx['bank'],allow_pickle=False) as f:
                    resident_banks={}
                    for l in ip.LAYERS:
                        a=f[str(l)]
                        if a.dtype!=np.float32 or a.shape!=(100000,1280) or not np.isfinite(a).all():raise ValueError('invalid original bank')
                        resident_banks[l]=torch.from_numpy(a.copy()).to(device)
            banks=model.prototype_embeddings if model is not None else resident_banks
            if any(('query',s['sample_id']) not in refs for s in pending) and model is None:
                banks=None;mm,dd=host();banks=mm.prototype_embeddings
            target={l:[torch.from_numpy(f.layers[l]).to(device) for f in features] for l in ip.LAYERS}
            for s in pending:
                tick=time.perf_counter();log=CaptureLog();external=refs.get(('query',s['sample_id']));observed=None
                if external:
                    q,qmeta=load_features(ctx,external,s);log.geometry(tuple(qmeta['grid']),q['positions'],tuple(qmeta['processed_shape']))
                    with torch.inference_mode():
                        for l in ip.LAYERS:
                            query=torch.from_numpy(q[f'layer_{l}']).to(device)
                            d,idx=_exact_top1_chunked(query,banks[l]);log.nearest('source',l,0,d.numpy().astype(np.float32)/1280,idx.numpy())
                            for j,bank in enumerate(target[l],1):
                                d,idx=_exact_top1_chunked(query,bank);log.nearest('target',l,j,d.numpy().astype(np.float32)/1280,idx.numpy())
                    counters['trusted_query_feature_reuses']+=1
                else:
                    mm,dd=host();banks=mm.prototype_embeddings
                    from DINOv3.MADEqual.robustad_target_fewshot_p0 import scoring
                    with profile_original(scoring,dd,timings,'Target_query'):
                        observed=observe_fewshot_pil(mm,ip.image(ctx,s),official_banks=banks,source_support=None,target_support=target,device=dd,nearest_observer=log.nearest,geometry_observer=log.geometry)
                    counters['query_forward_calls']+=1
                raw=log.arrays();timings['query_extraction_and_exact']+=time.perf_counter()-tick
                if observed is not None:
                    check(checks,s['sample_id'],'same_call_Source',scatter(raw['source_distance'],raw['positions'],log.grid),observed.official_raw,atol=1e-7,rtol=1e-6)
                    check(checks,s['sample_id'],'same_call_Union',scatter(np.minimum(raw['source_distance'],raw['target_distance'].min(0)),raw['positions'],log.grid),observed.target_raw[8],atol=1e-7,rtol=1e-6)
                    if not all(r['verified'] for r in checks[-2:]):raise ValueError('same-call observer mismatch')
                tick=time.perf_counter();raw,diag=add_geometry(raw,{l:[f.layers[l] for f in features] for l in ip.LAYERS});timings['geometry']+=time.perf_counter()-tick
                meta=dict(grid=list(log.grid),processed_shape=list(log.processed_shape),output_shape=[s['image_height']//4,s['image_width']//4],supports=support_meta,geometry_diagnostics=diag)
                from .calibration import legacy_raw
                historical=legacy_raw(ctx,s,state)
                if historical is not None:
                    prior,prior_meta=historical
                    actual=scatter(raw['source_distance'],raw['positions'],log.grid)
                    reference=scatter(prior['source_distance'],prior['positions'],tuple(prior_meta['grid']))
                    check(checks,s['sample_id'],'historical_Source_distance',actual,reference,atol=1e-7,rtol=1e-6,
                          category=ctx['category'],domain=shift,acceptance_role='historical_distance_audit_only',
                          reference_sha256=prior_meta['original_raw_sha256'])
                    if actual.shape==reference.shape:
                        diff=actual.astype(np.float64)-reference
                        checks[-1].update(mean_abs_error=float(np.abs(diff).mean()),mean_signed_error=float(diff.mean()))
                    meta['historical_Source_distance_audit']=checks[-1].copy()
                r=ip.save_cache(ctx,state,'target',s,raw,meta);done[('target',s['sample_id'])]=r
                verify_target(ctx,s,raw,meta,checks)
                ip.write_csv(ctx['root']/'capture_verification.csv',checks)
                print(f'[Robust capture Target] {ctx["category"]} {shift} {s["sample_id"]}',flush=True)
            del target,features,banks;gc.collect();torch.cuda.empty_cache()
        for s in source+targets:
            if s['sample_id'] in source_pending:continue
            kind='source' if s['shift']=='source' else 'target';a,meta=ip.load_cache(ctx,done[(kind,s['sample_id'])],s)
            if kind=='target':verify_target(ctx,s,a,meta,checks)
            else:validate_source(a,meta,s)
        state['status']='Target_capture_complete_Source_pending' if source_pending else 'selected_capture_complete'
        state['missing_Source_evaluation_ids']=source_pending
        state.setdefault('invocations',[]).append(dict(counts=counters,seconds=time.perf_counter()-started,timing=timings,scope=scope,shifts=shifts))
    except Exception as exc:
        state.update(status='capture_failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.evaluation import peak_rss
        state['last_cost']=dict(seconds=time.perf_counter()-started,timing_seconds=timings,counts=counters,peak_rss_bytes=peak_rss(),
            retained_cache_bytes=sum((ctx['root']/r['file']).stat().st_size for r in state['records']),Source_memory_increment_bytes=0,MAD_refitted=False)
        if device is not None:
            import torch
            if getattr(device,'type',None)=='cuda':
                state['last_cost'].update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
                    peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device),torch=torch.__version__,CUDA=torch.version.cuda,device=torch.cuda.get_device_name(device))
            model=None;resident_banks=None;banks=None;target=None;features=None;mm=None;query=None;bank=None
            gc.collect();torch.cuda.empty_cache()
        ip.write_json(ctx['root']/'cache_status.json',state);ip.write_csv(ctx['root']/'capture_verification.csv',checks)
    ip.ledger(ctx)
    return state
