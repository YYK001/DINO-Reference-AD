"""Exact CPU evaluation after global prediction freeze; original spatial kernels."""
import gc
import time
from pathlib import Path
import numpy as np
from DINOv3.MADEqual.aebad_s_neighborhood_segment_p0.analysis import (
    score_maps,array_sha,verify_map_record,instance_rows,history)
from DINOv3.MADEqual.aebad_s_neighborhood_segment_p0.prediction import bounded_images
from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.analysis import spatial
from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import image_metrics
from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic import metrics as pixel_metrics
from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic.numerics import check
from DINOv3.MADEqual.aebad_s_union_diagnostics.spatial import resized_mask
from DINOv3.MADEqual.aebad_s_support_distance_crossdomain.resource_policy import observation_only
from . import inputs as ip,prediction
from .method import BASES,NEW,BRANCHES


def number(v):return float(v) if v not in ('',None) else None


def restore_domain(path,identity,prediction_sha):
    if not (path/'status.json').exists():return None
    s=ip.read_json(path/'status.json')
    if s.get('status')!='CPU_complete_AUPRO_pending':return None
    if s['identity']!=identity or s['prediction_manifest_sha256']!=prediction_sha:
        raise ValueError('completed CPU domain belongs to a different input/prediction')
    for f,h in s['files'].items():
        if ip.sha(path/ip.bg.safe(f))!=h:raise ValueError('completed CPU domain was modified')
    return s


def instance_analysis(bundle,sample,mask,maps,grid,checks,old):
    original={ip.OLD.get(b,b):m for b,m in maps.items()}
    if sample['domain']=='background':
        rows,peaks=instance_rows(sample,mask,original,grid,old,checks)
        peaks={(r['instance_id'],r['branch'],r['region']):r for r in peaks}
        for r in rows:
            for region in ('instance','ring'):
                p=peaks[r['instance_id'],r['branch'],region]
                for field,key in [('own_'+region+'_y','own_y'),('own_'+region+'_x','own_x'),
                                  ('union_anchor_'+region+'_y','union_anchor_y'),('union_anchor_'+region+'_x','union_anchor_x')]:r[field]=p[key]
    else:rows=spatial(sample,mask,original,grid)
    for row in rows:
        row['branch']=ip.BACK.get(row['branch'],row['branch']);d,sid,iid=sample['domain'],sample['sample_id'],row['instance_id']
        row['domain']=d
        ref=bundle.instances[d,sid,iid,BASES[0]]
        row['small_group']=str(ref.get('small_original_three',ref.get('smallest_area_quartile','False'))).lower()=='true'
        row['small_group_definition']='original_background_smallest_three' if d=='background' else 'original_domain_smallest_area_quartile'
        for region in ('instance','ring'):
            a=row['own_'+region+'_y'],row['own_'+region+'_x'];b=row['union_anchor_'+region+'_y'],row['union_anchor_'+region+'_x']
            row['own_'+region+'_peak_moved']=a!=b if a[0] is not None and b[0] is not None else None
        if row['branch'] in BASES:
            expected=bundle.instances[d,sid,iid,row['branch']]
            for field in ('instance_max','ring_max','margin','fixed_union_instance_response','fixed_union_ring_response','fixed_union_margin'):
                value=row[field];old_value=number(expected[field])
                if value is None:
                    if old_value is not None:raise ValueError('parent instance NA changed')
                elif old_value is None:raise ValueError('parent instance became computable')
                else:check(checks,sid,'same_parent_instance_'+field,value,old_value,domain=d,instance_id=iid,branch=row['branch'])
    return rows


def sample_work(bundle,out,records,sample,old):
    d,sid=sample['domain'],sample['sample_id'];new,meta,record=prediction.load_record(out,records[d,sid],bundle.identity)
    raw,parent_meta,origin=bundle.load(sample)
    if meta['origin']!=origin or not np.array_equal(new['positions'],raw['positions']):raise ValueError('parent changed after prediction freeze')
    values=dict(zip(BASES,(raw['union_distance'],raw['segment_distance'],raw['pair_distance'])))|{b:new[b] for b in NEW}
    ctx=dict(medians=bundle.medians,scales=bundle.scales,state=bundle.state)
    maps={};scores=[];checks=[];frozen={r['branch']:r for r in record['scores']}
    for b in BRANCHES:
        current,ch1,response,_=score_maps(ctx,values[b],dict(meta,positions=new['positions']))
        check(checks,sid,'frozen_CH1',ch1,frozen[b]['CH1'],domain=d,branch=b)
        check(checks,sid,'frozen_CURRENT',current,frozen[b]['CURRENT'],domain=d,branch=b)
        maps[b]=response
        scores.append(dict(domain=d,sample_id=sid,relative_path=sample['relative_path'],label=sample['label'],branch=b,
            CH1=ch1,CURRENT=current,localization='same_frozen_Source_MAD_native_mean_then_float32_bilinear'))
    if not all(r['verified'] for r in checks):raise ValueError('prediction/map score mismatch')
    # Labels and GT are first used AFTER verified() accepted all 926 predictions.
    with ip.cross.Data(bundle.paths['data']) as data:mask=data.gt(sample)
    small_mask=resized_mask(mask,meta['output_shape']).astype(np.uint8)
    for b in BASES:verify_map_record(bundle.maps[d,sid,b],maps[b],small_mask)
    rows=instance_analysis(bundle,sample,mask,maps,tuple(meta['grid']),checks,old) if sample['label'] else []
    if not all(r['verified'] for r in checks):raise ValueError('same-parent instance reconstruction failed')
    mechanisms=[]
    regions={'normal_all_pixels':np.ones_like(small_mask,bool)} if not sample['label'] else {
        'anomaly_GT_pixels':small_mask.astype(bool),'anomaly_normal_pixels':~small_mask.astype(bool)}
    for b in BRANCHES:
        reduction=maps[BASES[0]].astype(np.float64)-maps[b]
        for group,region in regions.items():
            mechanisms.append(dict(domain=d,sample_id=sid,branch=b,group=group,pixels=int(region.sum()),
                mean_signed_response_reduction=float(reduction[region].mean()) if region.any() else None,
                reference=BASES[0],units='original_fused_Source_patch_MAD_response'))
    manifest=[dict(domain=d,sample_id=sid,branch=b,output_height=m.shape[0],output_width=m.shape[1],
        map_float32_sha256=array_sha(m),mask_uint8_sha256=array_sha(small_mask)) for b,m in maps.items()]
    return sample,maps,small_mask,scores,checks,rows,mechanisms,manifest


def case_figure(maps,mask,sample,out):
    from matplotlib import pyplot as plt
    low=min(float(m.min()) for m in maps.values());high=max(float(m.max()) for m in maps.values())
    fig,axes=plt.subplots(2,3,figsize=(12,8),layout='constrained')
    for ax,(b,m) in zip(axes.flat,maps.items()):
        ax.imshow(m,cmap='magma',vmin=low,vmax=high)
        if mask.any() and not mask.all():ax.contour(mask,levels=[.5],colors='cyan',linewidths=.5)
        ax.set_title(b,fontsize=8);ax.axis('off')
    fig.suptitle(sample['domain']+'/'+sample['sample_id']+' (fixed historical case)')
    fig.savefig(out,dpi=100);plt.close(fig)


def evaluate(bundle,out,workers=2):
    started=time.perf_counter();out=ip.initialize(out,bundle)
    manifest,records=prediction.verified(bundle,out);pred_sha=ip.sha(out/'prediction_manifest.json')
    old=history(bundle.bg);case_ids={s['sample_id'] for s in bundle.bg['capture']['selected_cases']}
    # Matplotlib settings live on G:, only original background cases are rendered.
    import os
    cfg=out/'_runtime/mpl';cfg.mkdir(parents=True,exist_ok=True);os.environ['MPLCONFIGDIR']=str(cfg)
    import matplotlib
    matplotlib.use('Agg')
    statuses=[]
    for d in ip.DOMAINS:
        path=out/'evaluation'/d;path.mkdir(parents=True,exist_ok=True)
        complete=restore_domain(path,bundle.identity,pred_sha)
        if complete:
            print(f'[GAP CPU] reuse verified {d}',flush=True);statuses.append(complete);continue
        tick=time.perf_counter();selected=[s for s in bundle.samples if s['domain']==d]
        pixels=sum((s['image_height']//4)*(s['image_width']//4) for s in selected)
        scratch=path/'_scratch';scratch.mkdir(exist_ok=True)
        label=np.memmap(scratch/'labels.bin',dtype=np.uint8,mode='w+',shape=(pixels,))
        mmaps={b:np.memmap(scratch/(b+'.bin'),dtype=np.float32,mode='w+',shape=(pixels,)) for b in NEW}
        offset=0;scores=[];checks=[];instances=[];mechanisms=[];map_records=[];metric=[]
        figure_dir=path/'figures';figure_dir.mkdir(exist_ok=True)
        try:
            for i,(s,maps,mask,sc,ck,inst,mech,mr) in enumerate(bounded_images(selected,lambda s:sample_work(bundle,out,records,s,old),workers),1):
                n=mask.size;label[offset:offset+n]=mask.ravel()
                for b in NEW:mmaps[b][offset:offset+n]=maps[b].ravel()
                offset+=n;scores.extend(sc);checks.extend(ck);instances.extend(inst);mechanisms.extend(mech);map_records.extend(mr)
                if d=='background' and s['sample_id'] in case_ids:case_figure(maps,mask,s,figure_dir/(s['sample_id']+'.png'))
                if i%25==0 or i==len(selected):print(f'[GAP CPU maps] {d} {i}/{len(selected)}',flush=True)
            if offset!=pixels:raise ValueError('full-pixel count incomplete')
            ip.write_csv(path/'numerical_verification.csv',checks)
            for b in BRANCHES:
                local=[r for r in scores if r['branch']==b]
                for scoring in ('CH1','CURRENT'):
                    im=image_metrics([r['label'] for r in local],[r[scoring] for r in local])
                    historical=bundle.metrics[d,b] if b in BASES else None
                    if historical and scoring=='CH1':
                        for k,v in im.items():check(checks,d,'same_parent_'+k,v,float(historical[k]),atol=1e-12,branch=b)
                    metric.append(dict(domain=d,branch=b,scoring=scoring,image_count=len(local),**im,
                        pixel_AUROC=number(historical['pixel_AUROC']) if historical else None,
                        pixel_AP=number(historical['pixel_AP']) if historical else None,
                        AUPRO=number(historical['AUPRO']) if historical else None,
                        pixel_metric_origin=('historical_verified_reuse:'+historical['parent_metric_file']) if historical else 'new_exact_all_pixels_CPU',
                        AUPRO_status='historical_verified_reuse' if historical else 'pending_explicit_original_ADEval_GPU'))
            for b in NEW:
                result_file=path/(b+'_exact_pixels.json')
                map_identity=ip.digest([r for r in map_records if r['branch']==b])
                if result_file.exists():
                    saved=ip.read_json(result_file)
                    if saved['identity']!=bundle.identity or saved['map_identity']!=map_identity:raise ValueError('partial pixel result identity changed')
                    value=saved['result'];print(f'[GAP exact pixels] reuse {d}/{b}',flush=True)
                else:
                    print(f'[GAP exact pixels] {d}/{b} pixels={pixels}',flush=True)
                    # The existing original exact metric kernel, without its
                    # estimated capacity gate; actual allocation errors propagate.
                    with observation_only():value=pixel_metrics.exact_binary(label,mmaps[b],memory_budget_bytes=0)
                    ip.write_json(result_file,dict(identity=bundle.identity,map_identity=map_identity,result=value))
                for row in metric:
                    if row['branch']==b:row.update(pixel_AUROC=value['AUROC'],pixel_AP=value['AP'],pixel_count=value['pixels'])
            if any(not r['verified'] for r in checks):raise ValueError('baseline image metric changed')
            for f,rows in [('metrics.csv',metric),('per_image_scores.csv',scores),('numerical_verification.csv',checks),
                           ('defect_instances.csv',instances),('response_changes.csv',mechanisms),('map_manifest.csv',map_records)]:ip.write_csv(path/f,rows)
            state=dict(status='CPU_complete_AUPRO_pending',identity=bundle.identity,domain=d,images=len(selected),
                prediction_manifest_sha256=pred_sha,numerical_failed_checks=0,seconds=time.perf_counter()-tick,
                files={p.relative_to(path).as_posix():ip.sha(p) for p in path.rglob('*') if p.is_file() and '_scratch' not in p.parts and p.name!='status.json'})
            ip.write_json(path/'status.json',state);statuses.append(state)
            print(f'[GAP CPU] {d} complete; new-branch AUPRO pending',flush=True)
        finally:
            ip.write_csv(path/'numerical_verification.csv',checks)
            del label,mmaps;gc.collect()
            # Delete only this run's explicitly named scratch files, after closing
            # all memmap views. User inputs and persistent predictions stay intact.
            for p in scratch.glob('*.bin'):p.unlink()
            scratch.rmdir()
    ip.write_json(out/'evaluation_status.json',dict(status='CPU_complete_AUPRO_pending',identity=bundle.identity,
        prediction_manifest_sha256=pred_sha,domains=[s['domain'] for s in statuses],images=926,
        numerical_failed_checks=0,seconds=time.perf_counter()-started,GPU_used=False,torch_imported='torch' in __import__('sys').modules))
    return statuses
