"""Exact full-pixel CPU evaluation; explicit original ADEval continuation."""
from __future__ import annotations

import gc
import shutil
import tempfile
import time
from pathlib import Path
import numpy as np
from . import inputs as ip, prediction
from .selection import SEEDS
from .storage import json_bytes, read_json
from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0 import analysis as original
from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.method import BRANCHES
from DINOv3.MADEqual.aebad_s_neighborhood_segment_p0.analysis import score_maps, array_sha, verify_map_record
from DINOv3.MADEqual.aebad_s_neighborhood_segment_p0.prediction import bounded_images
from DINOv3.MADEqual.aebad_s_union_diagnostics.spatial import resized_mask
from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import image_metrics
from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic.metrics import exact_binary, resource_bytes


def unit_name(domain, seed, branch): return f'{domain}/seed{seed}/{branch}'


def read_unit(ctx, stage, domain, seed, branch):
    row = ctx['ledger'].get(stage+'/'+unit_name(domain,seed,branch))
    if not row:return None
    if ip.sha(ctx['ledger'].root/row['file'])!=row['files'][row['file']]:
        raise ValueError('completed evaluation file changed')
    result=read_json(ctx['ledger'].root/row['file'])
    if result.get('identity')!=ctx['identity']:
        raise ValueError('evaluation unit identity mismatch')
    expected=[s['sample_id'] for s in ctx['plan']['samples'] if s['domain']==domain]
    if stage=='cpu':
        m=result['metric']
        if (result.get('status')!='cpu_complete_AUPRO_pending'
                or (m['domain'],m['support_seed'],m['branch'],m['image_count'])!=(domain,seed,branch,len(expected))
                or [r['sample_id'] for r in result['scores']]!=expected
                or [r['sample_id'] for r in result['map_manifest']]!=expected
                or any(not np.isfinite(m[k]) for k in ('image_AUROC','image_AP','pixel_AUROC','pixel_AP'))):
            raise ValueError('incomplete/mismatched CPU evaluation combination')
    elif (result.get('status')!='AUPRO_complete' or (result['domain'],result['support_seed'],result['branch'])!=(domain,seed,branch)
          or result['images']!=len(expected) or not np.isfinite(result['AUPRO'])):
        raise ValueError('incomplete/mismatched AUPRO combination')
    return result


def resources():
    import psutil
    from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.evaluation import peak_rss
    return dict(process_peak_rss_bytes=peak_rss(), process_current_rss_bytes=psutil.Process().memory_info().rss,
                available_ram_bytes=ip.old.common.available_ram())


def _cpu_unit(ctx, domain, seed, branch, workers, budget, work_dir):
    samples = prediction.require_frozen(ctx,domain)
    total = sum((s['image_height']//4)*(s['image_width']//4) for s in samples)
    required = resource_bytes(total)
    if required>budget or ip.old.common.available_ram()<required+(512<<20):
        raise MemoryError(f'exact full-pixel evaluation requires {required} bytes plus headroom; no approximation')
    work_dir = Path(work_dir).resolve()
    if work_dir.drive and work_dir.drive.upper()!='G:':
        raise ValueError('Windows scratch must stay on G:')
    for key in ('data','formal_dir','image_state','memory','checkpoint','dinov3','official','output_root'):
        path = ctx['paths'][key]
        if work_dir==path or path in work_dir.parents:
            raise ValueError('scratch must be separate from protected inputs/results')
    work_dir.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(work_dir).free<total*5+(512<<20):
        raise OSError('insufficient exact-sort scratch disk')
    scores = []; instances = []; mechanisms = []; maps_audit = []; started = time.perf_counter()

    def one(sample):
        arrays,meta,rows = prediction.load(ctx,sample,seed)
        scored = {b:score_maps(ctx,arrays[key],dict(meta,positions=arrays['positions']))
                  for b,key in prediction.FIELDS.items()}
        maps = {b:r[2] for b,r in scored.items()}
        for row in rows:
            actual = scored[row['branch']]
            if (abs(row['image_score']-actual[1])>1e-5 or abs(row['CURRENT_audit']-float(actual[2].max()))>1e-5):
                raise ValueError('map/image score changed after freeze')
        with ip.Data(ctx['paths']['data']) as data: mask = data.gt(sample)
        # Original spatial kernel receives all three maps so fixed Union peaks
        # and changes vs Segment are identical across method evaluations.
        inst = [r for r in original.spatial(sample,mask,maps,tuple(meta['grid'])) if r['branch']==branch]
        mech = [r for r in original.mechanisms(ctx,sample,mask,arrays,meta) if r['branch']==branch]
        response = maps[branch]; scaled_mask = resized_mask(mask, meta['output_shape']).astype(np.uint8)
        row = next(r for r in rows if r['branch']==branch)
        return sample,response,scaled_mask,row,inst,mech

    with tempfile.TemporaryDirectory(prefix='k8_sensitivity_owned_',dir=work_dir) as temporary:
        scratch = Path(temporary)
        labels = np.memmap(scratch/'labels.bin',dtype=np.uint8,mode='w+',shape=(total,))
        values = np.memmap(scratch/'maps.bin',dtype=np.float32,mode='w+',shape=(total,))
        try:
            offset = 0
            for sample,response,mask,row,inst,mech in bounded_images(samples,one,workers):
                n = response.size; labels[offset:offset+n] = mask.ravel(); values[offset:offset+n] = response.ravel(); offset+=n
                scores.append(dict(domain=domain,support_seed=seed,sample_id=sample['sample_id'],relative_path=sample['relative_path'],
                    label=sample['label'], **row))
                instances.extend(dict(r,support_seed=seed) for r in inst)
                mechanisms.extend(dict(r,support_seed=seed) for r in mech)
                maps_audit.append(dict(domain=domain,sample_id=sample['sample_id'],support_seed=seed,branch=branch,
                    output_height=response.shape[0],output_width=response.shape[1],
                    map_float32_sha256=array_sha(response),mask_uint8_sha256=array_sha(mask)))
                if len(scores)%25==0 or len(scores)==len(samples):
                    print(f'[K8 sensitivity CPU maps] {unit_name(domain,seed,branch)} {len(scores)}/{len(samples)}',flush=True)
            if offset!=total: raise ValueError('incomplete pixel population')
            maps_seconds = time.perf_counter()-started; tick = time.perf_counter()
            if ip.old.common.available_ram()<required+(512<<20):
                raise MemoryError('RAM headroom dropped before exact sort')
            print(f'[K8 sensitivity exact pixels] {unit_name(domain,seed,branch)} pixels={total}',flush=True)
            pixel = exact_binary(labels,values,memory_budget_bytes=budget)
            if not pixel['valid']: raise ValueError('required full pixel metric undefined')
            image = image_metrics([r['label'] for r in scores],[r['image_score'] for r in scores])
            metric = dict(domain=domain,support_seed=seed,branch=branch,image_count=len(samples),
                normal_count=sum(s['label']==0 for s in samples),anomaly_count=sum(s['label']==1 for s in samples),
                **image,pixel_AUROC=pixel['AUROC'],pixel_AP=pixel['AP'],AUPRO=None,pixel_count=total,
                image_score_rule='CH1',metric_origin='new_common_subset_exact_CPU',AUPRO_status='pending_explicit_ADEval')
            result = dict(identity=ctx['identity'],status='cpu_complete_AUPRO_pending',metric=metric,scores=scores,
                instances=instances,mechanisms=mechanisms,map_manifest=maps_audit,
                timing=dict(maps_GT_spatial_seconds=maps_seconds,exact_pixel_and_image_seconds=time.perf_counter()-tick,
                    whole_seconds=time.perf_counter()-started),resources=resources(),cpu_workers=workers,
                GT_used_only_after_predictions_frozen=True)
        finally:
            labels._mmap.close(); values._mmap.close(); del labels,values; gc.collect()
    return result


def run_cpu(ctx, domains, *, workers=4, memory_budget=8<<30, work_dir):
    ledger = ctx['ledger']; ledger.verify(); done = 0; reused = 0
    for domain in domains:
        prediction.require_frozen(ctx,domain)
        for seed in SEEDS:
            for branch in BRANCHES:
                key = 'cpu/'+unit_name(domain,seed,branch)
                if ledger.get(key):
                    read_unit(ctx,'cpu',domain,seed,branch)
                    reused+=1; continue
                result = _cpu_unit(ctx,domain,seed,branch,workers,memory_budget,work_dir)
                name = key+'/result.json'
                ledger.commit(key,dict(file=name,status=result['status']),{name:json_bytes(result)})
                done+=1
                print(f'[K8 sensitivity CPU] {unit_name(domain,seed,branch)} complete; AUPRO pending',flush=True)
    ledger.verify()
    return dict(new_combinations=done,reused_combinations=reused,model_inference=False,**resources())


def run_aupro(ctx, domains, *, device='cuda:0', allocator_limit_gib=1.5):
    ledger = ctx['ledger']; ledger.verify()
    pending = [(d,s,b) for d in domains for s in SEEDS for b in BRANCHES
               if not ledger.get('aupro/'+unit_name(d,s,b))]
    if not pending: return dict(new_combinations=0,model_inference=False)
    for domain,seed,branch in pending:
        prediction.require_frozen(ctx,domain)
        if not ledger.get('cpu/'+unit_name(domain,seed,branch)):
            raise ValueError('CPU map/score verification must precede AUPRO')
    from importlib.metadata import version
    if version('adeval')!='1.1.0': raise ValueError('frozen AUPRO requires ADEval1.1.0; leave pending')
    import torch
    requested = torch.device(device)
    if requested.type!='cuda' or not torch.cuda.is_available(): raise RuntimeError('explicit CUDA AUPRO unavailable; leave pending')
    requested = torch.device('cuda',torch.cuda.current_device() if requested.index is None else requested.index)
    torch.cuda.set_device(requested)
    total = torch.cuda.get_device_properties(requested).total_memory
    if not 0<allocator_limit_gib*2**30<=total or torch.cuda.mem_get_info(requested)[0]<allocator_limit_gib*2**30:
        raise MemoryError('requested AUPRO allocator cap unavailable')
    torch.cuda.set_per_process_memory_fraction(allocator_limit_gib*2**30/total,requested)
    torch.cuda.reset_peak_memory_stats(requested)
    from DINOv3.relation_reliability.official_metrics import official_aupro_adeval
    for domain,seed,branch in pending:
        samples = prediction.require_frozen(ctx,domain)
        pixels = sum(s['image_height']//4*(s['image_width']//4) for s in samples)
        if ip.old.common.available_ram()<pixels*16+(2<<30): raise MemoryError('insufficient RAM for original full-pixel AUPRO')
        cpu = read_unit(ctx,'cpu',domain,seed,branch)
        lookup = {r['sample_id']:r for r in cpu['map_manifest']}
        maps = []; masks = []; started = time.perf_counter()
        with ip.Data(ctx['paths']['data']) as data:
            for sample in samples:
                arrays,meta,_ = prediction.load(ctx,sample,seed)
                response = score_maps(ctx,arrays[prediction.FIELDS[branch]],dict(meta,positions=arrays['positions']))[2]
                mask = resized_mask(data.gt(sample),meta['output_shape']).astype(np.uint8)
                verify_map_record(lookup[sample['sample_id']],response,mask)
                maps.append(response); masks.append(mask)
                if len(maps)%25==0 or len(maps)==len(samples):
                    print(f'[K8 sensitivity AUPRO maps] {unit_name(domain,seed,branch)} {len(maps)}/{len(samples)}',flush=True)
        map_seconds = time.perf_counter()-started; tick = time.perf_counter()
        print(f'[K8 sensitivity AUPRO backend] {unit_name(domain,seed,branch)} nstrips=200 FPR=0.3',flush=True)
        value = official_aupro_adeval(masks,maps,nstrips=200)
        if not np.isfinite(value): raise FloatingPointError('nonfinite AUPRO')
        key = 'aupro/'+unit_name(domain,seed,branch); name=key+'/result.json'
        row = dict(identity=ctx['identity'],status='AUPRO_complete',domain=domain,support_seed=seed,branch=branch,
            AUPRO=float(value),images=len(samples),backend='official_aupro_adeval_ADEval1.1.0_200_FPR0.3',
            cpu_files=ledger.get('cpu/'+unit_name(domain,seed,branch))['files'],
            timing=dict(map_GT_seconds=map_seconds,AUPRO_seconds=time.perf_counter()-tick),resources=resources(),
            CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(requested),
            CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(requested),model_inference=False)
        ledger.commit(key,dict(file=name,status='AUPRO_complete'),{name:json_bytes(row)})
        print('[K8 sensitivity AUPRO] '+unit_name(domain,seed,branch),flush=True)
        del maps,masks; gc.collect()
    ledger.verify()
    return dict(new_combinations=len(pending),model_inference=False)
