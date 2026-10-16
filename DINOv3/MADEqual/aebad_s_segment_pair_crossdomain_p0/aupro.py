"""Explicit original CUDA ADEval continuation, never imported by preflight."""
from __future__ import annotations
import time
import numpy as np
from . import inputs as ip, prediction, analysis
from .method import BRANCHES
from DINOv3.MADEqual.aebad_s_neighborhood_segment_p0.prediction import protect_dependencies
from DINOv3.MADEqual.aebad_s_neighborhood_segment_p0.analysis import score_maps,verify_map_record
from DINOv3.MADEqual.aebad_s_union_diagnostics.spatial import resized_mask


def metric_rows(path):
    rows=ip.csv_file(path)
    for r in rows:
        for k in analysis.METRICS:r[k]=float(r[k]) if r.get(k) else None
    return rows


def run(ctx,prediction_dir,cpu_directory,output,device='cuda:0',allocator_limit_gib=1.5):
    started=time.perf_counter();cpu,cm=analysis.verified_analysis(ctx,prediction_dir,cpu_directory)
    pred,pm,samples=prediction.verified(ctx,prediction_dir)
    protect_dependencies(output,pred,cpu)
    # Concrete admission before initialization; unavailable backend stays pending.
    import importlib.util
    if importlib.util.find_spec('adeval') is None:raise ModuleNotFoundError('original ADEval unavailable; CPU results retain AUPRO pending')
    import torch
    requested=torch.device(device)
    if requested.type!='cuda' or not torch.cuda.is_available():raise RuntimeError('original AUPRO requires available explicit CUDA; no substitute')
    requested=torch.device('cuda',torch.cuda.current_device() if requested.index is None else requested.index)
    torch.cuda.set_device(requested)
    if allocator_limit_gib<=0:raise ValueError('positive CUDA allocator cap required')
    total=torch.cuda.get_device_properties(requested).total_memory
    if allocator_limit_gib*2**30>total or torch.cuda.mem_get_info(requested)[0]<allocator_limit_gib*2**30:
        raise MemoryError('insufficient VRAM for requested AUPRO cap; no metric/resolution change')
    torch.cuda.set_per_process_memory_fraction(allocator_limit_gib*2**30/total,requested)
    torch.cuda.reset_peak_memory_stats(requested)
    from DINOv3.relation_reliability.official_metrics import official_aupro_adeval
    out=ip.new_output(ctx,output,'evaluate-aupro')
    rows=[r for r in metric_rows(cpu/'metrics.csv') if r['domain'] in pm['domains']]
    map_records={(r['domain'],r['sample_id'],r['branch']):r for r in ip.csv_file(cpu/'map_manifest.csv')}
    records={(r['domain'],r['sample_id']):r for r in pm['records']};timings={}
    for domain in pm['domains']:
        selected=[s for s in samples if s['domain']==domain]
        pixels=sum(s['image_height']//4*(s['image_width']//4) for s in selected)
        # List plus backend stacking/copies; conservative bound, no background service.
        if ip.common.available_ram()<pixels*16+2*2**30:raise MemoryError('insufficient RAM for full-resolution original AUPRO')
        for branch in BRANCHES:
            maps=[];masks=[];tick=time.perf_counter()
            with ip.Data(ctx['paths']['data']) as dataset:
                for sample in selected:
                    arrays,meta=prediction.load(ctx,pred,records[(domain,sample['sample_id'])],sample)
                    response=score_maps(ctx,arrays[analysis.FIELDS[branch]],dict(meta,positions=arrays['positions']))[2]
                    mask=resized_mask(dataset.gt(sample),meta['output_shape']).astype(np.uint8)
                    verify_map_record(map_records[(domain,sample['sample_id'],branch)],response,mask)
                    maps.append(response);masks.append(mask)
            timings[domain+'_'+branch+'_map_GT_seconds']=time.perf_counter()-tick;tick=time.perf_counter()
            value=official_aupro_adeval(masks,maps,nstrips=200)
            if not np.isfinite(value):raise FloatingPointError('nonfinite original AUPRO')
            row=next(r for r in rows if (r['domain'],r['branch'])==(domain,branch))
            row.update(AUPRO=float(value),aupro_origin='new_original_ADEval_CUDA_200_FPR_le_0.3')
            timings[domain+'_'+branch+'_AUPRO_seconds']=time.perf_counter()-tick
            del maps,masks
    analysis.verified_analysis(ctx,pred,cpu)
    if ip.context(ctx['paths'])['identity']!=ctx['identity']:raise ValueError('joint frozen assets changed during AUPRO')
    rows=analysis.append_macro(rows)
    ip.write_csv(out/'metrics.csv',rows);ip.write_csv(out/'metric_comparisons.csv',analysis.metric_comparisons(rows))
    analysis.write_report(out,rows,ip.csv_file(cpu/'ranking_summary.csv'),
        _numeric(ip.csv_file(cpu/'mechanism_summary.csv')), _numeric(ip.csv_file(cpu/'margin_repair_summary.csv')),pm['domains'])
    from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.evaluation import peak_rss
    ip.write_json(out/'timing.json',dict(seconds=time.perf_counter()-started,stage_seconds=timings,process_peak_rss_bytes=peak_rss(),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(requested),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(requested),
        allocator_limit_gib=allocator_limit_gib,environment=dict(torch=torch.__version__,CUDA=torch.version.cuda,device=torch.cuda.get_device_name(requested)),
        new_output_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),scope='AUPRO_only_no_model_no_feature_extraction',
        CPU_results_reused_from=str(cpu)))
    manifest=dict(tool=ip.TOOL,identity=ctx['identity'],domains=pm['domains'],images=pm['images'],
        prediction_directory=str(pred),prediction_manifest_sha256=ip.sha(pred/'prediction_manifest.json'),
        cpu_directory=str(cpu),cpu_analysis_manifest_sha256=ip.sha(cpu/'analysis_manifest.json'),
        files={p.name:ip.sha(p) for p in out.iterdir() if p.is_file() and p.name!='status.json'})
    ip.write_json(out/'result_manifest.json',manifest)
    ip.write_json(out/'status.json',dict(tool=ip.TOOL,status='all_requested_metrics_complete',identity=ctx['identity'],
        domains=pm['domains'],images=len(samples),all_metrics_complete=True,joint_domains_complete=set(pm['domains'])==set(ip.DOMAINS),
        result_manifest_sha256=ip.sha(out/'result_manifest.json'),model_inference=False,feature_extraction=False))
    return out


def _numeric(rows):
    # CSV table reload for report formatting only; preserve domain/branch/group IDs.
    for r in rows:
        for k,v in list(r.items()):
            if v=='':r[k]=None
            elif k not in ('domain','branch','group','definition','reference','aggregation'):
                try:r[k]=float(v)
                except ValueError:pass
    return rows


def verified_result(ctx,directory):
    from pathlib import Path
    root=Path(directory).resolve();s=ip.read_json(root/'status.json');m=ip.read_json(root/'result_manifest.json')
    if (s['status']!='all_requested_metrics_complete' or s['identity']!=ctx['identity'] or m['identity']!=ctx['identity']
            or s['result_manifest_sha256']!=ip.sha(root/'result_manifest.json')):raise ValueError('AUPRO result incomplete/incompatible')
    cpu,cm=analysis.verified_analysis(ctx,m['prediction_directory'],m['cpu_directory'])
    if (ip.sha(cpu/'analysis_manifest.json')!=m['cpu_analysis_manifest_sha256'] or m['domains']!=cm['domains']
            or m['images']!=cm['images'] or m['prediction_manifest_sha256']!=cm['prediction_manifest_sha256']):raise ValueError('AUPRO scope/parent mismatch')
    for file,h in m['files'].items():
        if ip.sha(root/ip.safe(file))!=h:raise ValueError('AUPRO result changed')
    return root,m
