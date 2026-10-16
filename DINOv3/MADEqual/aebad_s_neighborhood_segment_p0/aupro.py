"""EXPLICIT CUDA-only continuation; never imported by CPU entry points."""
from __future__ import annotations

import time
import numpy as np

from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic.numerics import reconstruct
from DINOv3.MADEqual.aebad_s_union_diagnostics.spatial import resized_mask
from . import inputs as ip
from .analysis import Dataset, P, G, comparisons, verified_analysis, verify_map_record
from .prediction import fresh, verified_predictions, load_prediction, protect_dependencies


def evaluate_aupro(ctx,prediction_dir,cpu_directory,output,device='cuda:0'):
    # Admission precedes GPU initialization and all new metric computation.
    cpu,_=verified_analysis(ctx,prediction_dir,cpu_directory)
    pred,records=verified_predictions(ctx,prediction_dir)
    protect_dependencies(output,pred,cpu)
    out=fresh(ctx,output,'evaluate-aupro');started=time.perf_counter()
    import torch
    requested=torch.device(device)
    if requested.type!='cuda' or not torch.cuda.is_available(): raise RuntimeError('original AUPRO requires explicit available CUDA device')
    requested=torch.device('cuda',torch.cuda.current_device() if requested.index is None else requested.index)
    torch.cuda.set_device(requested);torch.cuda.reset_peak_memory_stats(requested)
    from DINOv3.relation_reliability.official_metrics import official_aupro_adeval
    metrics=ip.read_csv((cpu/'metrics.csv').read_bytes())
    map_records={(r['sample_id'],r['branch']):r for r in ip.read_csv((cpu/'map_manifest.csv').read_bytes())}
    for row in metrics:
        for field in ('image_AUROC','image_AP','pixel_AUROC','pixel_AP','AUPRO'):
            if field in row: row[field]=float(row[field]) if row[field] else None
    timings={}
    for branch,field in ((P,'pair_distance'),(G,'neighbor_distance')):
        maps=[];masks=[];tick=time.perf_counter()
        with ip.Archive(ctx['paths']['raw_archive']) as a, Dataset(ctx) as data:
            for sample in ctx['samples']:
                raw,meta,segments=ip.load_image(ctx,a,sample)
                saved=load_prediction(ctx,pred,records[sample['sample_id']],raw,meta,segments)
                _,responses,_=reconstruct(saved[field],raw['positions'],tuple(meta['grid']),tuple(meta['output_shape']),ctx['medians'],ctx['scales'])
                response=responses['FUSED'];mask=resized_mask(data.gt(sample),meta['output_shape']).astype(np.uint8)
                expected=map_records[(sample['sample_id'],branch)]
                verify_map_record(expected,response,mask)
                maps.append(response);masks.append(mask)
        timings[branch+'_map_GT_seconds']=time.perf_counter()-tick;tick=time.perf_counter()
        value=official_aupro_adeval(masks,maps,nstrips=200)
        if not np.isfinite(value): raise FloatingPointError('nonfinite AUPRO')
        row=next(r for r in metrics if r['branch']==branch and r['normal_group']=='all85')
        row.update(AUPRO=float(value),aupro_source='new_original_ADEval_CUDA_nstrips200_FPR_le_0.3')
        timings[branch+'_AUPRO_seconds']=time.perf_counter()-tick
        del maps,masks
    verified_analysis(ctx,pred,cpu)
    if ip.context(ctx['paths'])['identity']!=ctx['identity']: raise ValueError('protected inputs changed during AUPRO')
    ip.write_csv(out/'metrics.csv',metrics);ip.write_csv(out/'metric_comparisons.csv',comparisons(metrics))
    ip.write_json(out/'timing.json',dict(**timings,seconds=time.perf_counter()-started,
                                        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(requested),
                                        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(requested),
                                        scope='explicit_AUPRO_only_not_model_inference'))
    (out/'objective_report.md').write_text('Original CUDA/ADEval AUPRO continuation only.\n'
        '200 thresholds, project original FPR integration <=0.3; no CPU substitute.\n'
        'Image metrics and pixel AUROC/AP are reused from the verified CPU analysis; Union/SEGMENT pixels retain their historical origin.\n'
        'See the CPU objective report for mechanisms, ranking, instances and development-domain limits.\n',encoding='utf-8')
    ip.write_json(out/'status.json',dict(tool=ip.TOOL,stage='evaluate-aupro',status='all_requested_metrics_complete',
                                        identity=ctx['identity'],analyzed_images=297,all_metrics_complete=True,
                                        cpu_analysis_manifest_sha256=ip.sha(cpu/'analysis_manifest.json'),
                                        prediction_manifest_sha256=ip.sha(pred/'prediction_manifest.json'),
                                        model_inference=False,feature_extraction=False,
                                        output_sha256={n:ip.sha(out/n) for n in ('metrics.csv','metric_comparisons.csv','timing.json','objective_report.md')}))
    return out
