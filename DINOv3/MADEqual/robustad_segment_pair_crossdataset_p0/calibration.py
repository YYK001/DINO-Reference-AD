"""Recover only the original 23 Source normals; never refit patch MAD."""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
from . import inputs as ip
from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import statistics,fit_image_mad
from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic.numerics import check

DEFINITION='CH1_interpolated_peak_valid3x3_local_float32_1.4826MAD_eps1e-8_half_half'


def legacy_raw(ctx,sample,state):
    """Original Source-only cache headers and Source v6 identity, no guessed Q."""
    directory=ctx['paths'].get('source_raw_root')
    if directory is None:return None
    root=directory/'categories/robustad'/ctx['category']
    if not root.exists():return None
    complete=ip.read_json(root/'complete.json')
    if complete.get('status')!='complete' or complete.get('category')!=ctx['category']:raise ValueError('incomplete original raw cache category')
    expected=ip.digest(dict(protocol='relation_reliability_exact_superadd_seed42_v6',complete=ctx['cat']['source_complete'],
        bank_size=ctx['frozen']['bank_size_bytes'],sidecars_sha256=ctx['cat']['source_sha256']['memory/anchor_sidecars.npz'],
        calibration_sha256=ctx['cat']['source_sha256']['calibration/calibration_state.npz'],
        memory_provenance_sha256=ctx['cat']['source_sha256']['memory/memory_provenance.json']))
    enrolled=state.setdefault('legacy_sources',{})
    # Header scan is explicit reuse of original observations, never image extraction.
    if '_legacy_raw_index' not in ctx:
        index={}
        for p in sorted((root/'images').rglob('*.npz')):
            with np.load(p,allow_pickle=False) as f:
                try:relative=ip.canonical(str(f['image_path'].item()),ctx['category'])
                except (KeyError,ValueError):continue
            index.setdefault(relative,[]).append(p)
        ctx['_legacy_raw_index']=index
    candidates=ctx['_legacy_raw_index'].get(sample['relative_path'],[])
    if not candidates:return None
    if len(candidates)!=1:raise ValueError('duplicate original Source raw observation')
    path=candidates[0];previous=enrolled.get(sample['sample_id']);checksum=ip.sha(path)
    if previous is not None and previous['sha256']!=checksum:raise ValueError('previously used original raw changed')
    with np.load(path,allow_pickle=False) as f:
        if (str(f['source_identity_sha256'].item())!=expected or str(f['category'].item())!=ctx['category']
                or str(f['source_protocol'].item())!='relation_reliability_exact_superadd_seed42_v6'
                or int(f['label'])!=sample['label'] or list(f['layers'])!=list(ip.LAYERS)):
            raise ValueError('original raw Source identity differs')
        # cache_identity includes original absolute path and current image bytes.
        key=dict(protocol='relation_oracle_four_layer_cache_seed42_v2',source_identity=expected,
            record=ctx['cat']['legacy_records'][sample['relative_path']],
            image_sha256=sample['image_sha256'],mask_sha256=sample.get('gt_sha256'))
        if str(f['cache_identity_sha256'].item())!=ip.digest(key):raise ValueError('original raw image/GT/source byte identity differs')
        raw=f['raw_distances'].copy();indices=f['top1_indices'].astype(np.int64)
        shape=[int(v) for v in f['evaluation_output_shape']]
    if raw.dtype!=np.float32 or raw.ndim!=3 or raw.shape[0]!=4 or shape!=[sample['image_height']//4,sample['image_width']//4]:raise ValueError('original raw geometry differs')
    enrolled[sample['sample_id']]=dict(file=str(path),sha256=checksum,source_identity=expected)
    return dict(source_distance=raw.reshape(4,-1),source_index=indices.reshape(4,-1),positions=np.arange(raw[0].size,dtype=np.int64)),dict(grid=list(raw.shape[1:]),output_shape=shape,processed_shape=None,action='verified_original_Source_raw',original_raw_sha256=checksum)


def load_state(ctx):
    path=ctx['root']/'image_calibration_state.json';status=ip.read_json(ctx['root']/'calibration_status.json')
    if status.get('status')!='image_calibration_frozen' or status.get('identity')!=ctx['identity'] or ip.sha(path)!=status['state_file_sha256']:raise ValueError('category image calibration incomplete or modified')
    state=ip.read_json(path)
    if (state.get('category')!=ctx['category'] or state.get('identity')!=ctx['identity']
            or state.get('definition')!=DEFINITION or state.get('calibration_ids')!=[s['sample_id'] for s in ctx['cat']['calibration']]
            or state.get('bank_sha256')!=ctx['frozen']['bank_sha256'] or state.get('patch_mad_sha256')!=ctx['cat']['source_sha256']['calibration/calibration_state.npz']
            or ip.digest({k:v for k,v in state.items() if k!='state_sha256'})!=state.get('state_sha256')):raise ValueError('image calibration belongs to another class/bank/split/definition')
    for kind in ('peak','local'):
        med=np.asarray(state[kind]['medians'],np.float32);scale=np.asarray(state[kind]['scales'],np.float32)
        if med.shape!=(4,) or scale.shape!=(4,) or not np.isfinite(med).all() or not np.isfinite(scale).all() or np.any(scale<=0):raise ValueError('invalid image MAD')
    return state


def recover(ctx,device_name,min_free_gib=20):
    from . import runtime as rt
    started=time.perf_counter();state=ip.ledger(ctx);samples=ctx['cat']['calibration']
    if len(samples)!=23 or any(s['label']!=0 or s['source_index']%8 for s in samples):raise ValueError('must use exactly original 23 Source normals')
    if (ctx['root']/'image_calibration_state.json').exists():return load_state(ctx)
    if (ctx['root']/'calibration_status.json').is_file():raise ValueError('orphan calibration status; preserve it')
    external=ctx['paths'].get('image_state_root')
    if external is not None:
        import shutil
        other=dict(ctx,root=external/ctx['category']);load_state(other)
        for name in ('image_calibration_state.json','calibration_status.json'):
            if (ctx['root']/name).exists():raise FileExistsError('calibration import cannot overwrite')
            shutil.copyfile(other['root']/name,ctx['root']/name)
        return load_state(ctx)
    records={(r['kind'],r['sample_id']):r for r in state['records']};model=None;device=None
    checks=[];values={'peak':[],'local':[]};raws=[];per_image=[];new_calls=0;timing={}
    try:
        for s in samples:
            r=records.get(('calibration',s['sample_id']))
            if r is None:
                value=legacy_raw(ctx,s,state)
                if value is None:
                    if model is None:
                        tick=time.perf_counter();model,device=rt.load_host(ctx,device_name,min_free_gib);timing['model_and_bank_loading']=time.perf_counter()-tick
                    value=rt.source_observation(ctx,s,model,device,timing);new_calls+=1
                a,meta=value;rt.validate_source(a,meta,s);r=ip.save_cache(ctx,state,'calibration',s,a,meta)
            a,meta=ip.load_cache(ctx,r,s);rt.validate_source(a,meta,s)
            _,peak,local=statistics(a['source_distance'],a['positions'],tuple(meta['grid']),tuple(meta['output_shape']),ctx['medians'],ctx['scales'])
            for kind,v in (('peak',peak),('local',local)):
                if 'original_'+kind in a:check(checks,s['sample_id'],'CPU_vs_original_'+kind,v,a['original_'+kind])
                values[kind].append(v)
                for j,l in enumerate(ip.LAYERS):per_image.append(dict(category=ctx['category'],sample_id=s['sample_id'],source_index=s['source_index'],calibration_index=s['calibration_index'],statistic=kind,layer=l,value=float(v[j]),cache_sha256=r['sha256']))
            raws.append(a['source_distance'])
            print(f'[Robust calibration] {ctx["category"]} {len(raws)}/23',flush=True)
        with np.load(ctx['source']/'calibration/calibration_state.npz',allow_pickle=False) as old:
            for j,l in enumerate(ip.LAYERS):
                pooled=np.concatenate([a[j] for a in raws]);median=float(np.median(pooled));scale=float(np.float32(max(1.4826*float(np.median(np.abs(pooled-median))),1e-8)))
                check(checks,'Source_calibration','original_patch_count',pooled.size,old[f'ecdf_{l}'].size,atol=0,rtol=0,layer=l)
                check(checks,'Source_calibration','original_patch_median',median,ctx['medians'][j],atol=1e-7,rtol=1e-6,layer=l)
                check(checks,'Source_calibration','original_patch_scale',scale,ctx['scales'][j],atol=1e-7,rtol=1e-6,layer=l)
        ip.write_csv(ctx['root']/'calibration_per_image.csv',per_image);ip.write_csv(ctx['root']/'calibration_verification.csv',checks)
        if not checks or not all(r['verified'] for r in checks):raise ValueError('calibration failed original patch MAD/statistic checks; original patch MAD not refitted')
        image_state=dict(category=ctx['category'],identity=ctx['identity'],definition=DEFINITION,
            bank_sha256=ctx['frozen']['bank_sha256'],patch_mad_sha256=ctx['cat']['source_sha256']['calibration/calibration_state.npz'],
            calibration_ids=[s['sample_id'] for s in samples])
        for kind,v in values.items():
            med,scale=fit_image_mad(v);image_state[kind]=dict(medians=med.tolist(),scales=scale.tolist())
        image_state['state_sha256']=ip.digest(image_state);ip.write_json(ctx['root']/'image_calibration_state.json',image_state)
        ip.write_json(ctx['root']/'calibration_status.json',dict(status='image_calibration_frozen',identity=ctx['identity'],category=ctx['category'],images=23,
            state_file_sha256=ip.sha(ctx['root']/'image_calibration_state.json'),source_forward_calls=new_calls,patch_MAD_refitted=False,
            seconds=time.perf_counter()-started,GT_decoded=False))
    finally:
        from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.evaluation import peak_rss
        cost=dict(seconds=time.perf_counter()-started,timing_seconds=timing,peak_rss_bytes=peak_rss(),source_forward_calls=new_calls,
                  original_patch_MAD_refitted=False,Source_memory_increment_bytes=0)
        model=None
        if device is not None:
            import gc,torch
            cost.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device))
            gc.collect();torch.cuda.empty_cache()
        ip.write_json(ctx['root']/'cost_calibration.json',cost)
    ip.ledger(ctx)
    return load_state(ctx)
