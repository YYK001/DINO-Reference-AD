"""Seven explicitly named branches; original CPU pixels / spatial / ADEval."""
import csv
import gc
import math
import time
from pathlib import Path
import numpy as np
from . import inputs as ip,prediction
from .inputs import st
from .protocol import BRANCHES,DOMAINS,LAYERS,U,S,G0,W,GL,ND,ID
from DINOv3.MADEqual.aebad_s_support_witness_segment_p0 import evaluation as original
from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.execution import bounded_images
from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.fresh_evaluation import mark_small
from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.analysis import spatial,U as OLD_U,S as OLD_S
from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic.numerics import check,adaptive_max,reconstruct
from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import image_metrics,pair_errors,statistics,ch1_score

METRICS=('image_AUROC','image_AP','pixel_AUROC','pixel_AP','AUPRO')


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def restore(a,d):
    path=a.run_root/'evaluation'/d/'status.json'
    if not path.exists():return None
    value=st.read_json(path)
    if value['identity']!=ip.identity(a) or value['prediction_sha256']!=st.sha(a.run_root/'prediction_manifests'/f'{d}.json'):
        raise ValueError('CPU result input identity changed')
    for f,sha in value['files'].items():
        if st.sha(path.parent/f)!=sha:raise ValueError('completed evaluation modified: '+f)
    if value['status']!='CPU_complete_AUPRO_pending' or value['numerical_failed_checks']:
        raise ValueError('CPU result incomplete')
    return value


def sample_work(a,sample):
    x,m,r=ip.verify_prediction(a,sample);med,scale,state=ip.patch_state(a)
    ctx=dict(medians=med,scales=scale,state=state);maps={};scores=[];checks=[];responses=[];coverage=[];peaks=[];locations=[];details={}
    frozen={v['branch']:v for v in m['scores']}
    for b in BRANCHES:
        # Same two shared calls/order as original.shared.score_maps, retaining
        # layer maps/native z for attribution rather than recomputing them.
        current,peak,local=statistics(x[b],x['positions'],tuple(m['grid']),tuple(m['output_shape']),med,scale)
        z,layer_maps,_=reconstruct(x[b],x['positions'],tuple(m['grid']),tuple(m['output_shape']),med,scale)
        ch1=ch1_score(peak,local,state);response=layer_maps['FUSED'];details[b]=(z,layer_maps,peak,local)
        for name,value in (('CH1',ch1),('CURRENT',current)):
            check(checks,sample['sample_id'],'frozen_'+name,value,frozen[b][name],domain=sample['domain'],branch=b)
        maps[b]=response
        scores.append(dict(domain=sample['domain'],sample_id=sample['sample_id'],relative_path=sample['relative_path'],
            label=sample['label'],branch=b,CH1=ch1,CURRENT=current))
    if not all(c['verified'] for c in checks):raise ValueError('same-cache score reconstruction failed')
    with ip.Data(Path(ip.catalog(a)['data_root'])) as data:gt=data.gt(sample)
    mask=original.shared.resized_mask(gt,m['output_shape']).astype(np.uint8)
    rename={U:OLD_U,S:OLD_S};back={v:k for k,v in rename.items()}
    instances=spatial(sample,gt,{rename.get(b,b):v for b,v in maps.items()},tuple(m['grid'])) if sample['label'] else []
    for row in instances:
        row['branch']=back.get(row['branch'],row['branch'])
        for region in ('instance','ring'):
            own=row[f'own_{region}_y'],row[f'own_{region}_x']
            anchor=row[f'union_anchor_{region}_y'],row[f'union_anchor_{region}_x']
            row[f'own_{region}_peak_moved']=own!=anchor if own[0] is not None and anchor[0] is not None else None
    regions={'normal_all_pixels':np.ones(mask.shape,bool)} if not sample['label'] else {
        'anomaly_GT_pixels':mask.astype(bool),'anomaly_normal_pixels':~mask.astype(bool)}
    physical=x['positions'];grid=tuple(m['grid']);native_gt=adaptive_max(gt,grid).ravel()[physical]
    if sample['label']:
        from scipy.ndimage import binary_dilation
        ring=binary_dilation(gt,structure=np.ones((3,3),bool),iterations=16)&~gt.astype(bool)
        groups={'defect_internal':native_gt,'defect_surrounding':adaptive_max(ring,grid).ravel()[physical]&~native_gt}
    else:
        native=details[U][0].mean(axis=0,dtype=np.float32).ravel()[physical]
        count=max(1,math.ceil(len(native)*.01));order=np.lexsort((physical,-native))
        high=np.zeros(len(native),bool);high[order[:count]]=True
        groups={'normal_high_response_top1pct':high,'normal_remaining':~high}
    anchor=np.unravel_index(np.argmax(maps[U]),maps[U].shape)
    for b in BRANCHES:
        for name,region in regions.items():
            delta=maps[U].astype(np.float64)-maps[b]
            responses.append(dict(domain=sample['domain'],sample_id=sample['sample_id'],branch=b,group=name,
                pixels=int(region.sum()),mean_signed_response_reduction=float(delta[region].mean()) if region.any() else None,
                units='fused_Source_patch_MAD_response',reference=U))
        z,layer_maps,peak,local=details[b]
        own=np.unravel_index(np.argmax(maps[b]),maps[b].shape)
        peaks.append(dict(domain=sample['domain'],sample_id=sample['sample_id'],branch=b,
            own_peak_y=int(own[0]),own_peak_x=int(own[1]),UNION_peak_y=int(anchor[0]),UNION_peak_x=int(anchor[1]),
            own_peak_response=float(maps[b][own]),fixed_UNION_peak_response=float(maps[b][anchor]),
            CH1_peak_component=float(((peak-np.asarray(state['peak']['medians'],np.float32))/
                np.asarray(state['peak']['scales'],np.float32)).mean(dtype=np.float32))*.5,
            CH1_local_component=float(((local-np.asarray(state['local']['medians'],np.float32))/
                np.asarray(state['local']['scales'],np.float32)).mean(dtype=np.float32))*.5))
        for j,l in enumerate(LAYERS):
            from scipy.ndimage import uniform_filter
            local_map=((uniform_filter(z[j],size=3,mode='constant',cval=0.)*9.)[1:-1,1:-1]/9.)
            py,px=np.unravel_index(np.argmax(layer_maps[l]),layer_maps[l].shape)
            ly,lx=np.unravel_index(np.argmax(local_map),local_map.shape)
            locations.append(dict(domain=sample['domain'],sample_id=sample['sample_id'],branch=b,layer=l,
                interpolated_peak_y=int(py),interpolated_peak_x=int(px),native_local_center_y=int(ly+1),
                native_local_center_x=int(lx+1),native_local_center_physical_patch=int((ly+1)*grid[1]+lx+1),
                peak_response=float(peak[j]),local3_response=float(local[j]),
                CH1_layer_contribution=float((peak[j]-np.float32(state['peak']['medians'][j]))/
                    np.float32(state['peak']['scales'][j]))/8+float((local[j]-np.float32(state['local']['medians'][j]))/
                    np.float32(state['local']['scales'][j]))/8,
                coordinate_source='derived from current frozen maps/native statistics; not historical coordinates'))
            delta=x[U][j].astype(np.float64)-x[b][j];tol=1e-7+1e-6*np.abs(x[U][j]);effective=delta>tol
            for group,region in groups.items():
                selected=region&effective
                target=x['target_distance'][j,selected]
                wd=x['witness_distance'][j,selected] if b==W else np.empty(0)
                finite=wd[np.isfinite(wd)]
                coverage.append(dict(domain=sample['domain'],sample_id=sample['sample_id'],branch=b,layer=l,group=group,
                    patches=int(region.sum()),effective_new_coverage=int(selected.sum()),
                    mean_raw_reduction=float(delta[region].mean()) if region.any() else None,
                    mean_raw_reduction_in_scale_units=float((delta[region]/float(scale[j])).mean()) if region.any() else None,
                    raw_scale_normalization='delta distance / frozen scale; actual reconstructed response reported separately',
                    distance_to_nearest_Target_mean=float(target.mean()) if len(target) else None,
                    distance_to_existing_UNION_mean=float(x[U][j,selected].mean()) if selected.any() else None,
                    distance_to_Source_mean=float(x['source_distance'][j,selected].mean()) if selected.any() else None,
                    distance_to_representative_witness_mean=float(finite.mean()) if len(finite) else None,
                    witness_distance_computable=int(len(finite)),distance_unit='Euclidean /1280 once',
                    G1_reduced_where_G0_did_not=int((region&(x[G0][j]==x[U][j])&
                        (x[W][j]<x[U][j]-tol)).sum())))
    manifests=[dict(domain=sample['domain'],sample_id=sample['sample_id'],branch=b,
        map_float32_sha256=original.shared.array_sha(v),mask_uint8_sha256=original.shared.array_sha(mask),
        output_height=v.shape[0],output_width=v.shape[1]) for b,v in maps.items()]
    return sample,maps,mask,scores,checks,instances,responses,coverage,peaks,manifests,locations


def ranking(scores,domain,samples):
    rows=[];tails=[];fixed=set(ip.CASE_IDS[:10]);normal_ids={s['sample_id'] for s in samples if not s['label']}
    groups=[('all_normal',normal_ids)]
    if domain=='background':groups += [('original_fixed10',normal_ids&fixed),('other_normal',normal_ids-fixed)]
    for scoring in ('CH1','CURRENT'):
        for group,ids in groups:
            if not ids:continue
            base=sorted((r for r in scores if r['branch']==U and (r['label'] or r['sample_id'] in ids)),key=lambda r:r['sample_id'])
            labels=[r['label'] for r in base];original_errors=pair_errors(labels,[r[scoring] for r in base])
            for b in BRANCHES:
                lookup={r['sample_id']:r for r in scores if r['branch']==b}
                value=[lookup[r['sample_id']][scoring] for r in base];errors=pair_errors(labels,value)
                normal=np.asarray([v for v,y in zip(value,labels) if not y])
                rows.append(dict(domain=domain,scoring=scoring,branch=b,normal_group=group,
                    normal_images=len(ids),ranking_errors=float(errors.sum()),improved_pairs=int((errors<original_errors).sum()),
                    worsened_pairs=int((errors>original_errors).sum()),interpretation='ordering, not deployment false alarms/misses'))
                tails.append(dict(domain=domain,scoring=scoring,branch=b,normal_group=group,images=len(normal),
                    mean=float(normal.mean()),p95=float(np.quantile(normal,.95)),p99=float(np.quantile(normal,.99)),max=float(normal.max())))
    return rows,tails


def figure(maps,mask,sample,path):
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(17,8),layout='constrained')
    low=min(float(m.min()) for m in maps.values());high=max(float(m.max()) for m in maps.values())
    for ax,(b,m) in zip(axes.flat,maps.items()):
        ax.imshow(m,cmap='magma',vmin=low,vmax=high)
        if mask.any() and not mask.all():ax.contour(mask,levels=[.5],colors='cyan',linewidths=.4)
        ax.set_title(b,fontsize=8);ax.axis('off')
    axes.flat[-1].imshow(mask,cmap='gray');axes.flat[-1].set_title('GT');axes.flat[-1].axis('off')
    fig.suptitle(sample['domain']+'/'+sample['sample_id']+' fixed case; shared map color scale')
    path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=100);plt.close(fig)


def evaluate(a,d,*,global_checked=False):
    if not global_checked:prediction.verified(a)  # Before the FIRST GT decode.
    if restore(a,d):return dict(status='CPU_reused',domain=d)
    path=a.run_root/'evaluation'/d;path.mkdir(parents=True,exist_ok=True)
    samples=[s for s in ip.catalog(a)['samples'] if s['domain']==d]
    count=sum((s['image_height']//4)*(s['image_width']//4) for s in samples)
    scratch=a.evaluation_scratch/d;scratch.mkdir(parents=True,exist_ok=True)
    labels=np.memmap(scratch/'labels.bin',dtype=np.uint8,mode='w+',shape=(count,))
    memory={b:np.memmap(scratch/(b+'.bin'),dtype=np.float32,mode='w+',shape=(count,)) for b in BRANCHES}
    scores=[];checks=[];instances=[];responses=[];coverage=[];peaks=[];manifests=[];locations=[];offset=0;tick=time.perf_counter()
    try:
        for i,(s,maps,mask,sc,ck,inst,resp,cov,pk,mr,loc) in enumerate(bounded_images(samples,lambda s:sample_work(a,s),a.cpu_workers),1):
            n=mask.size;labels[offset:offset+n]=mask.ravel()
            for b in BRANCHES:memory[b][offset:offset+n]=maps[b].ravel()
            offset+=n;scores.extend(sc);checks.extend(ck);instances.extend(inst);responses.extend(resp)
            coverage.extend(cov);peaks.extend(pk);manifests.extend(mr);locations.extend(loc)
            if d=='background' and s['sample_id'] in ip.CASE_IDS:figure(maps,mask,s,path/'figures'/(s['sample_id']+'.png'))
            if i%25==0 or i==len(samples):print(f'[fixed graph CPU] {d} {i}/{len(samples)}',flush=True)
        if offset!=count:raise ValueError('incomplete exact pixel population')
        counts=dict(pixels=count,positive_pixels=int(labels.sum(dtype=np.int64)),negative_pixels=count-int(labels.sum(dtype=np.int64)))
        mark_small(instances,d);metrics=[];pred_sha=st.sha(a.run_root/'prediction_manifests'/f'{d}.json')
        for b in BRANCHES:
            map_id=st.digest([r for r in manifests if r['branch']==b])
            pixel=original.exact_pixel_result(path/(b+'_exact_pixels.json'),ip.identity(a),map_id,labels,memory[b],
                domain=d,branch=b,prediction_sha=pred_sha,counts=counts)
            local=[r for r in scores if r['branch']==b]
            for scoring in ('CH1','CURRENT'):
                metrics.append(dict(domain=d,branch=b,scoring=scoring,image_count=len(local),
                    **image_metrics([r['label'] for r in local],[r[scoring] for r in local]),
                    pixel_AUROC=pixel['AUROC'],pixel_AP=pixel['AP'],AUPRO=None,AUPRO_status='pending',
                    pixel_metric_origin='current_run_original_exact_CPU',seed=a.seed,evaluation_scope=a.evaluation_scope))
        ranks,tails=ranking(scores,d,samples)
        tables={'metrics.csv':metrics,'per_image_scores.csv':scores,'numerical_verification.csv':checks,
            'defect_instances.csv':instances,'response_changes.csv':responses,'coverage_audit.csv':coverage,
            'image_peak_decomposition.csv':peaks,'score_locations.csv':locations,'map_manifest.csv':manifests,'ranking_summary.csv':ranks,'normal_tail.csv':tails}
        for f,rows in tables.items():st.write_csv(path/f,rows)
        value=dict(status='CPU_complete_AUPRO_pending',identity=ip.identity(a),domain=d,images=len(samples),
            prediction_sha256=pred_sha,numerical_failed_checks=sum(not r['verified'] for r in checks),
            seconds=time.perf_counter()-tick,files={p.name:st.sha(p) for p in path.iterdir() if p.is_file() and p.name!='status.json'})
        st.write_json(path/'status.json',value);return value
    finally:
        for m in (labels,*memory.values()):m.flush();m._mmap.close()
        del labels,memory;gc.collect()
        for name in ('labels.bin',*(b+'.bin' for b in BRANCHES)):(scratch/name).unlink()
        scratch.rmdir()


def aupro(a,d,b):
    from importlib.metadata import version
    import torch
    from DINOv3.relation_reliability.official_metrics import official_aupro_adeval
    if b not in BRANCHES or version('adeval')!='1.1.0':raise ValueError('original ADEval1.1.0 and declared branch required')
    if not restore(a,d):raise ValueError('completed CPU domain required')
    rows=[r for r in read_csv(a.run_root/'evaluation'/d/'map_manifest.csv') if r['branch']==b]
    map_id=st.digest(rows);path=a.run_root/'aupro'/d/(b+'.json')
    expected=dict(identity=ip.identity(a),domain=d,branch=b,map_identity=map_id,nstrips=200,FPR_max=.3)
    if path.exists():
        record=st.read_json(path)
        validate_aupro(record,expected);return dict(status='AUPRO_reused',**record)
    torch.cuda.set_device(torch.device(a.device));torch.cuda.reset_peak_memory_stats()
    med,scale,state=ip.patch_state(a);ctx=dict(medians=med,scales=scale,state=state)
    samples=[s for s in ip.catalog(a)['samples'] if s['domain']==d]
    if {r['sample_id'] for r in rows}!={s['sample_id'] for s in samples}:raise ValueError('incomplete AUPRO map population')
    byid={r['sample_id']:r for r in rows};maps=[];masks=[];tick=time.perf_counter()
    with ip.Data(Path(ip.catalog(a)['data_root'])) as data:
        for sample in samples:
            x,m,r=ip.verify_prediction(a,sample)
            response=original.shared.score_maps(ctx,x[b],dict(m,positions=x['positions']))[2]
            mask=original.shared.resized_mask(data.gt(sample),m['output_shape']).astype(np.uint8)
            original.shared.verify_map_record(byid[sample['sample_id']],response,mask)
            maps.append(response);masks.append(mask)
    value=float(official_aupro_adeval(masks,maps,nstrips=200))
    record=dict(expected,AUPRO=value,verified_images=len(maps),seconds=time.perf_counter()-tick,
        backend='original_ADEval_1.1.0_CUDA',peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    record['seal']=st.digest(record);validate_aupro(record,expected);st.write_json(path,record);return record


def validate_aupro(record,expected):
    if any(record.get(k)!=v for k,v in expected.items()) or record.get('seal')!=st.digest({k:v for k,v in record.items() if k!='seal'}):
        raise ValueError('AUPRO result identity modified')
    if not np.isfinite(record['AUPRO']) or not 0<=record['AUPRO']<=1:raise ValueError('invalid AUPRO')
    if record.get('backend')!='original_ADEval_1.1.0_CUDA':raise ValueError('different AUPRO backend')
