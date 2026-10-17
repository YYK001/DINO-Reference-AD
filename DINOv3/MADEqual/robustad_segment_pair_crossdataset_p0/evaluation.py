"""Exact CPU metrics and original CUDA AUPRO; GT only after group freeze."""
from __future__ import annotations
import gc
import time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from . import inputs as ip,prediction as pr,calibration
from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0 import analysis as shared
from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.verification import historical_score,audit
from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import image_metrics,pair_errors
from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic.numerics import check
from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic.metrics import exact_binary,resource_bytes
from DINOv3.MADEqual.aebad_s_neighborhood_segment_p0.analysis import score_maps,array_sha,verify_map_record
from DINOv3.MADEqual.aebad_s_neighborhood_segment_p0.prediction import bounded_images
from DINOv3.MADEqual.aebad_s_union_diagnostics.spatial import resized_mask

METRICS=('image_AUROC','image_AP','pixel_AUROC','pixel_AP','AUPRO')


def directory(ctx,group,tag,stage):return ctx['root']/'evaluations'/ip.safe(tag)/ip.safe(group)/stage


def numeric_rows(path):
    rows=ip.read_csv(path)
    for r in rows:
        for k in METRICS:r[k]=float(r[k]) if r.get(k) else None
    return rows


def seal(ctx,out,group,status,prediction_sha,extra=None):
    files={p.relative_to(out).as_posix():ip.sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ('manifest.json','status.json','failure.json')}
    ip.write_json(out/'manifest.json',dict(identity=ctx['identity'],category=ctx['category'],shift=group,prediction_manifest_sha256=prediction_sha,files=files,**(extra or {})))
    ip.write_json(out/'status.json',dict(tool=ip.TOOL,identity=ctx['identity'],category=ctx['category'],shift=group,status=status,
        manifest_sha256=ip.sha(out/'manifest.json'),all_requested_metrics_complete=False))


def verified(ctx,group,tag,stage):
    out=directory(ctx,group,tag,stage);status=ip.read_json(out/'status.json');manifest=ip.read_json(out/'manifest.json')
    pred,_,_=pr.verify(ctx,group)
    expected='group_metrics_complete' if stage=='aupro' else 'cpu_complete_aupro_pending'
    if (status['status']!=expected or status['identity']!=ctx['identity'] or manifest['identity']!=ctx['identity']
            or manifest['category']!=ctx['category'] or manifest['shift']!=group or status['manifest_sha256']!=ip.sha(out/'manifest.json')
            or manifest['prediction_manifest_sha256']!=ip.sha(pred/'manifest.json')):raise ValueError('evaluation incomplete or provenance changed')
    for name,h in manifest['files'].items():
        if ip.sha(out/ip.safe(name))!=h:raise ValueError('evaluation sidecar/case changed')
    if stage=='aupro':
        cpu,_=verified(ctx,group,tag,'cpu')
        if manifest.get('cpu_manifest_sha256')!=ip.sha(cpu/'manifest.json'):raise ValueError('AUPRO parent changed')
    return out,manifest


def draw_case(ctx,sample,mask,maps,arrays,out):
    """Common linear display limits only; scored arrays are not normalized."""
    width,height=420,320;canvas=Image.new('RGB',(3*width,2*height),(255,255,255));draw=ImageDraw.Draw(canvas)
    lo=min(float(a.min()) for a in maps.values());hi=max(float(a.max()) for a in maps.values())
    panels=[('RGB',ip.image(ctx,sample)),('GT',Image.fromarray(mask.astype(np.uint8)*255).convert('RGB')),
            ('PAIR acceptance, layer mean',Image.fromarray(np.uint8(arrays['pair_gate'].mean(0)*255)).convert('RGB'))]
    for b,a in maps.items():
        v=np.uint8(np.clip((a-lo)/max(hi-lo,1e-12),0,1)*255)
        panels.append((b,Image.fromarray(v).convert('RGB')))
    for i,(label,im) in enumerate(panels):
        im.thumbnail((width,height-45));x=(i%3)*width;y=(i//3)*height
        canvas.paste(im,(x+(width-im.width)//2,y+28));draw.text((x+5,y+5),label,fill=(0,0,0))
    draw.text((5,2*height-14),f'shared response display range [{lo:.5g}, {hi:.5g}], scores unchanged',fill=(0,0,0))
    path=out/'cases'/(ip.key(sample)+'.png');path.parent.mkdir(exist_ok=True);canvas.save(path)
    return dict(category=ctx['category'],shift=sample['shift'],sample_id=sample['sample_id'],file=path.relative_to(out).as_posix(),
                selection='lexicographically_first_normal_and_first_anomaly_per_shift',display_min=lo,display_max=hi,RGB_used_for_display_only=True)


def evaluate(ctx,group,tag,workers=4,memory_budget=8<<30):
    started=time.perf_counter();pred,manifest,samples=pr.verify(ctx,group)
    out=directory(ctx,group,tag,'cpu')
    if (out/'status.json').is_file() and ip.read_json(out/'status.json')['status']=='cpu_complete_aupro_pending':verified(ctx,group,tag,'cpu');return out
    if out.exists():raise FileExistsError('partial CPU output retained; retry with a new --evaluation-tag, no new capture needed')
    out.mkdir(parents=True)
    state=calibration.load_state(ctx);work=dict(ctx,state=state)
    branches=(pr.SOURCE,) if group=='source' else ip.BRANCHES
    fields={pr.SOURCE:'source_distance'} if group=='source' else pr.FIELDS
    pixels=sum((s['image_height']//4)*(s['image_width']//4) for s in samples)
    required=resource_bytes(pixels)
    if required>memory_budget or ip.available_ram()<required+(512<<20):
        ip.write_json(out/'failure.json',dict(error='insufficient full-pixel RAM/budget',required_bytes=required,budget_bytes=memory_budget,available_bytes=ip.available_ram()))
        raise MemoryError('insufficient full-pixel RAM/budget; use same predictions with a fresh evaluation tag after resolving resources')
    import shutil
    if shutil.disk_usage(out).free<pixels*(1+4*len(branches))+(512<<20):raise OSError('insufficient full-pixel scratch disk')
    records={r['sample_id']:r for r in manifest['records']};scores={(r['sample_id'],r['branch']):float(r['image_score']) for r in ip.read_csv(pred/'scores.csv')}
    checks=[];all_scores=[];instances=[];mechanisms=[];map_records=[];metrics=[];figures=[];ranking=[];costs={}
    fixed=set()
    if group!='source':
        for label in (0,1):fixed.add(min(s['sample_id'] for s in samples if s['label']==label))
    scratch=out/'_scratch';scratch.mkdir();label=np.memmap(scratch/'labels.bin',mode='w+',dtype=np.uint8,shape=(pixels,))
    mmaps={b:np.memmap(scratch/(b+'.bin'),mode='w+',dtype=np.float32,shape=(pixels,)) for b in branches};offset=0;tick=time.perf_counter()
    def one(sample):
        arrays,meta=pr.load(ctx,pred,records[sample['sample_id']],sample,state);local=[]
        scalar=pr.score(ctx,arrays,meta,state,source=group=='source')
        maps={b:score_maps(work,arrays[k],dict(meta,positions=arrays['positions'])) for b,k in fields.items()}
        for b,(current,ch1,response,_) in maps.items():
            check(local,sample['sample_id'],'same_run_map_CURRENT_'+b,float(response.max()),current)
            check(local,sample['sample_id'],'same_run_frozen_CH1_'+b,ch1,scores[(sample['sample_id'],b)])
        if group!='source':
            for b,key in ((pr.SOURCE,'source_score'),(ip.BRANCHES[0],'union_score')):
                actual=next(r['CURRENT_score'] for r in scalar if r['branch']==b)
                historical_score(local,sample,'SOURCE_CURRENT' if b==pr.SOURCE else 'UNION_CURRENT',actual,sample[key])
        if any(not r['verified'] for r in local):raise ValueError('internal score/backend or historical compatibility check failed')
        mask=ip.gt(ctx,sample) # first GT decode, only after complete prediction validation
        output_maps={b:v[2] for b,v in maps.items()}
        spatial=shared.spatial(sample,mask,output_maps,tuple(meta['grid'])) if group!='source' else []
        mechanism=shared.mechanisms(ctx,sample,mask,arrays,meta) if group!='source' else []
        figure=draw_case(ctx,sample,mask,output_maps,arrays,out) if sample['sample_id'] in fixed else None
        return sample,output_maps,resized_mask(mask,meta['output_shape']),scalar,local,spatial,mechanism,figure
    try:
        for sample,maps,mask,scalar,local,sp,mech,figure in bounded_images(samples,one,workers):
            n=mask.size;label[offset:offset+n]=mask.ravel()
            for b,a in maps.items():
                mmaps[b][offset:offset+n]=a.ravel();map_records.append(dict(category=ctx['category'],domain=group,shift=group,sample_id=sample['sample_id'],branch=b,
                    map_float32_sha256=array_sha(a),mask_uint8_sha256=array_sha(mask.astype(np.uint8)),output_height=a.shape[0],output_width=a.shape[1]))
            offset+=n;checks.extend(local);all_scores.extend(dict(r,label=sample['label']) for r in scalar)
            instances.extend(dict(r,category=ctx['category'],shift=group) for r in sp);mechanisms.extend(dict(r,category=ctx['category'],shift=group) for r in mech)
            if figure:figures.append(figure)
            print(f'[Robust evaluate] {ctx["category"]} {group} {offset}/{pixels} pixels',flush=True)
        if offset!=pixels:raise ValueError('incomplete full pixel population')
        costs['maps_GT_spatial_seconds']=time.perf_counter()-tick
        y=np.asarray([s['label'] for s in samples]);vectors={b:np.asarray([scores[(s['sample_id'],b)] for s in samples]) for b in branches}
        for b in branches:
            if ip.available_ram()<required+(512<<20):raise MemoryError('RAM dropped before exact pixel sort; no approximation')
            tick=time.perf_counter();pixel=exact_binary(label,mmaps[b],memory_budget_bytes=memory_budget)
            if not pixel['valid']:raise ValueError('required pixel metric unavailable')
            costs[b+'_pixel_seconds']=time.perf_counter()-tick
            metrics.append(dict(category=ctx['category'],shift=group,branch=b,scope='source_reference' if group=='source' else 'target_shift',
                images=len(samples),normal_count=int((y==0).sum()),anomaly_count=int((y==1).sum()),**image_metrics(y,vectors[b]),
                pixel_AUROC=pixel['AUROC'],pixel_AP=pixel['AP'],AUPRO=None,pixels=pixels,origin='new_CH1_exact_full_pixels_CPU'))
        if group!='source':
            for b,ref in shared.COMPARISONS:
                now,old=pair_errors(y,vectors[b]),pair_errors(y,vectors[ref])
                ranking.append(dict(category=ctx['category'],shift=group,branch=b,reference=ref,weighted_pair_errors=float(now.sum()),
                    reference_weighted_pair_errors=float(old.sum()),delta_pair_errors=float((now-old).sum()),improved_pairs=int((now<old).sum()),
                    worsened_pairs=int((now>old).sum()),unchanged_pairs=int((now==old).sum()),tie_weight=.5,interpretation='ordering_not_deployment_threshold'))
        groups=shared.group_small_instances(instances,[group]) if instances else []
        detail,repair=shared.margin_repairs(instances) if instances else ([],[])
        tables={'metrics.csv':metrics,'per_image_scores.csv':all_scores,'ranking_summary.csv':ranking,'map_manifest.csv':map_records,
            'defect_instances.csv':instances,'small_instance_groups.csv':groups,'margin_repair_per_instance.csv':detail,'margin_repair_summary.csv':repair,
            'mechanism_per_image.csv':mechanisms,'mechanism_summary.csv':shared.mechanism_summary(mechanisms,[group]) if mechanisms else [],
            'numerical_verification.csv':checks,'cases.csv':figures}
        if group!='source':tables['historical_baseline_comparison.csv']=shared.historical_comparisons(ctx,samples,checks)
        for name,rows in tables.items():
            ip.write_csv(out/name,[dict(r,category=ctx['category'],shift=group) for r in rows])
        ip.write_json(out/'historical_score_audit.json',audit(checks))
        from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.evaluation import peak_rss
        ip.write_json(out/'timing.json',dict(seconds=time.perf_counter()-started,stage_seconds=costs,peak_rss_bytes=peak_rss(),cpu_workers=workers,
            full_pixel_budget_bytes=memory_budget,model_inference=False,feature_extraction=False,GT_only_after_prediction_freeze=True))
        pr.verify(ctx,group)
    except Exception as exc:
        ip.write_json(out/'failure.json',dict(error=f'{type(exc).__name__}: {exc}',checks=checks));raise
    finally:
        label._mmap.close()
        for a in mmaps.values():a._mmap.close()
        for p in scratch.iterdir():
            if p.name not in {'labels.bin',*(b+'.bin' for b in branches)}:raise ValueError('unexpected scratch contents preserved')
            p.unlink()
        scratch.rmdir();gc.collect()
    seal(ctx,out,group,'cpu_complete_aupro_pending',ip.sha(pred/'manifest.json'))
    return out


def evaluate_aupro(ctx,group,tag,device_name='cuda:0',allocator_gib=1.5):
    cpu,_=verified(ctx,group,tag,'cpu');pred,manifest,samples=pr.verify(ctx,group);state=calibration.load_state(ctx)
    out=directory(ctx,group,tag,'aupro')
    if out.exists():verified(ctx,group,tag,'aupro');return out
    started=time.perf_counter()
    import torch
    device=torch.device(device_name)
    if device.type!='cuda' or not torch.cuda.is_available():raise RuntimeError('original ADEval requires explicit CUDA; pending otherwise')
    device=torch.device('cuda',device.index if device.index is not None else torch.cuda.current_device());torch.cuda.set_device(device)
    total=torch.cuda.get_device_properties(device).total_memory
    if allocator_gib<=0 or allocator_gib*2**30>total or torch.cuda.mem_get_info(device)[0]<allocator_gib*2**30:raise MemoryError('AUPRO VRAM cap unavailable')
    torch.cuda.set_per_process_memory_fraction(allocator_gib*2**30/total,device);torch.cuda.reset_peak_memory_stats(device)
    from DINOv3.relation_reliability.official_metrics import official_aupro_adeval
    out.mkdir(parents=True);rows=numeric_rows(cpu/'metrics.csv');costs={};work=dict(ctx,state=state)
    records={r['sample_id']:r for r in manifest['records']}
    references={(r['sample_id'],r['branch']):r for r in ip.read_csv(cpu/'map_manifest.csv')}
    fields={pr.SOURCE:'source_distance'} if group=='source' else pr.FIELDS
    pixels=sum((s['image_height']//4)*(s['image_width']//4) for s in samples)
    try:
        for b,k in fields.items():
            if ip.available_ram()<pixels*16+(2<<30):raise MemoryError('insufficient RAM for full-resolution original AUPRO')
            maps=[];masks=[];tick=time.perf_counter()
            for s in samples:
                arrays,meta=pr.load(ctx,pred,records[s['sample_id']],s,state)
                response=score_maps(work,arrays[k],dict(meta,positions=arrays['positions']))[2]
                mask=resized_mask(ip.gt(ctx,s),meta['output_shape']).astype(np.uint8)
                verify_map_record(references[(s['sample_id'],b)],response,mask);maps.append(response);masks.append(mask)
            costs[b+'_map_GT_seconds']=time.perf_counter()-tick;tick=time.perf_counter()
            value=float(official_aupro_adeval(masks,maps,nstrips=200))
            if not np.isfinite(value):raise FloatingPointError('nonfinite AUPRO')
            row=next(r for r in rows if r['branch']==b);row.update(AUPRO=value,aupro_origin='new_original_ADEval_CUDA_200_FPR0.3')
            costs[b+'_AUPRO_seconds']=time.perf_counter()-tick;del maps,masks;gc.collect()
            print(f'[Robust AUPRO] {ctx["category"]} {group} {b}',flush=True)
        ip.write_csv(out/'metrics.csv',rows)
        import shutil
        for name in ('historical_score_audit.json','historical_baseline_comparison.csv'):
            if (cpu/name).is_file():shutil.copyfile(cpu/name,out/name)
        ip.write_json(out/'timing.json',dict(seconds=time.perf_counter()-started,stage_seconds=costs,allocator_limit_gib=allocator_gib,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device),
            model_inference=False,feature_extraction=False))
        verified(ctx,group,tag,'cpu');seal(ctx,out,group,'group_metrics_complete',ip.sha(pred/'manifest.json'),dict(cpu_manifest_sha256=ip.sha(cpu/'manifest.json')))
    except Exception as exc:
        ip.write_json(out/'failure.json',dict(error=f'{type(exc).__name__}: {exc}'));raise
    return out
