"""Four-branch orchestration around existing scoring/spatial/exact metric kernels."""
import gc
import math
from numbers import Real
import time
from pathlib import Path
import numpy as np
from . import inputs as ip,storage as st,prediction
from .method import BASES,NEW,BRANCHES,U,W,C
from .execution import bounded_images
from DINOv3.MADEqual.aebad_s_gap_soft_segment_p0 import evaluation as shared
from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import image_metrics,pair_errors
from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic import metrics as pixels
from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic.numerics import check
from DINOv3.MADEqual.aebad_s_support_distance_crossdomain.resource_policy import observation_only


def validate_exact_pixels(result,counts):
    """An intact JSON is insufficient: validate values AND reconstructed GT counts."""
    if not isinstance(result,dict):raise ValueError('exact pixel result must be a mapping')
    for key in ('AUROC','AP'):
        value=result.get(key)
        if isinstance(value,bool) or not isinstance(value,Real) or not math.isfinite(value) or not 0<=value<=1:
            raise ValueError('invalid exact pixel metric: '+key)
    for key,expected in counts.items():
        value=result.get(key)
        if isinstance(value,bool) or not isinstance(value,int) or value!=expected:
            raise ValueError('exact pixel count differs: '+key)
    if (result.get('valid') is not True or result.get('AUROC_valid') is not True or result.get('AP_valid') is not True
        or result.get('status')!='exact_all_pixels_cpu_tie_grouped'):
        raise ValueError('exact all-pixel ROC/AP result is incomplete or invalid')


def exact_pixel_result(path,identity,map_identity,labels,scores,*,domain,branch,prediction_sha,counts):
    """Commit each intermediate result before the whole CPU domain completes.

    The separate seal anchors the ORIGINAL file bytes. A missing/interrupted seal
    is rejected and preserved, never created from an existing untrusted result.
    """
    path=Path(path);seal=path.with_name(path.stem+'.seal.json')
    inputs=dict(identity=identity,map_identity=map_identity,domain=domain,branch=branch,
        prediction_manifest_sha256=prediction_sha,counts=counts,backend='original_exact_all_pixels_CPU_v1')
    if path.exists() or seal.exists():
        if not path.is_file() or not seal.is_file():raise ValueError('unsealed exact pixel intermediate preserved; cannot resume: '+str(path))
        record=st.read_json(seal)
        if record.get('format')!='sealed_exact_pixels_v1' or record.get('inputs')!=inputs or record.get('result_file_sha256')!=st.sha(path):
            raise ValueError('modified/incompatible exact pixel intermediate: '+str(path))
        saved=st.read_json(path)
        if saved.get('identity')!=identity or saved.get('map_identity')!=map_identity:raise ValueError('exact pixel input identity differs')
        result=saved.get('result');validate_exact_pixels(result,counts)
        return result
    with observation_only():result=pixels.exact_binary(labels,scores,memory_budget_bytes=0)
    validate_exact_pixels(result,counts)
    st.write_json(path,dict(identity=identity,map_identity=map_identity,result=result))
    st.write_json(seal,dict(format='sealed_exact_pixels_v1',inputs=inputs,result_file_sha256=st.sha(path)))
    return result


def restore(bundle,out,domain):
    p=Path(out)/'evaluation'/domain/'status.json'
    if not p.exists():return None
    s=st.read_json(p)
    if s['identity']!=bundle.identity or s['prediction_sha256']!=st.sha(Path(out)/'prediction_manifests'/(domain+'.json')):raise ValueError('CPU stage input identity changed')
    if s['status']!='CPU_complete_AUPRO_pending':return None
    if s['domain']!=domain or s['images']!=len(prediction.selected(bundle,domain)) or s['numerical_failed_checks']!=0:raise ValueError('CPU domain completeness differs')
    for f,h in s['files'].items():
        if st.sha(p.parent/f)!=h:raise ValueError('completed CPU evaluation modified: '+f)
    return s


def image_work(bundle,out,sample,old):
    d,sid=sample['domain'],sample['sample_id'];new,meta,record=prediction.load_record(bundle,out,d,sid)
    raw,parent_meta,origin=bundle.load(sample)
    if origin!=meta['origin'] or not np.array_equal(raw['positions'],new['positions']):raise ValueError('historical raw changed after freeze')
    values=dict(zip(BASES,(raw['union_distance'],raw['segment_distance'])))|{b:new[b] for b in NEW}
    ctx=dict(medians=bundle.medians,scales=bundle.scales,state=bundle.state);scores=[];checks=[];maps={}
    frozen={v['branch']:v for v in record['metadata']['scores']}
    for b in BRANCHES:
        current,ch1,response,_=shared.score_maps(ctx,values[b],dict(meta,positions=new['positions']))
        for k,v in [('CH1',ch1),('CURRENT',current)]:check(checks,sid,'frozen_'+k,v,frozen[b][k],domain=d,branch=b)
        maps[b]=response;scores.append(dict(domain=d,sample_id=sid,label=sample['label'],relative_path=sample['relative_path'],branch=b,CH1=ch1,CURRENT=current))
    if not all(x['verified'] for x in checks):raise ValueError('frozen map score mismatch')
    with ip.parent.cross.Data(bundle.paths['data']) as data:mask=data.gt(sample)
    small=shared.resized_mask(mask,meta['output_shape']).astype(np.uint8)
    for b in BASES:shared.verify_map_record(bundle.maps[d,sid,b],maps[b],small)
    instances=shared.instance_analysis(bundle,sample,mask,maps,tuple(meta['grid']),checks,old) if sample['label'] else []
    if not all(x['verified'] for x in checks):raise ValueError('parent instance reconstruction failed')
    regions={'normal_all_pixels':np.ones_like(small,bool)} if not sample['label'] else {'anomaly_GT_pixels':small.astype(bool),'anomaly_normal_pixels':~small.astype(bool)}
    response=[]
    for b in BRANCHES:
        delta=maps[U].astype(np.float64)-maps[b]
        for name,region in regions.items():response.append(dict(domain=d,sample_id=sid,branch=b,group=name,pixels=int(region.sum()),
            mean_signed_response_reduction=float(delta[region].mean()) if region.any() else None,reference=U,units='fused_Source_patch_MAD_response'))
    manifests=[dict(domain=d,sample_id=sid,branch=b,output_height=m.shape[0],output_width=m.shape[1],
        map_float32_sha256=shared.array_sha(m),mask_uint8_sha256=shared.array_sha(small)) for b,m in maps.items()]
    return sample,maps,small,scores,checks,instances,response,manifests


def figure(maps,mask,sample,path):
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    lo=min(float(v.min()) for v in maps.values());hi=max(float(v.max()) for v in maps.values())
    for ax,(b,m) in zip(axes.flat,list(maps.items())+[(W+' minus UNION',maps[W]-maps[U]),(C+' minus UNION',maps[C]-maps[U])]):
        diff=' minus ' in b;limit=float(np.abs(m).max()) if diff else None
        ax.imshow(m,cmap='coolwarm' if diff else 'magma',vmin=-limit if diff else lo,vmax=limit if diff else hi)
        if mask.any() and not mask.all():ax.contour(mask,levels=[.5],colors='cyan',linewidths=.4)
        ax.set_title(b,fontsize=8);ax.axis('off')
    fig.suptitle(sample['domain']+'/'+sample['sample_id']+' (original fixed case)');fig.savefig(path,dpi=100);plt.close(fig)


def ranking(scores,bundle,domain):
    rows=[];tails=[];fixed={s['sample_id'] for s in bundle.bg['capture']['selected_cases'] if s['label']==0} if domain=='background' else set()
    for scoring in ('CH1','CURRENT'):
        groups=[('all_normal',None)]+([('original_fixed10',fixed),('other_normal',set(s['sample_id'] for s in prediction.selected(bundle,domain) if not s['label'])-fixed)] if fixed else [])
        for name,ids in groups:
            baseline=[v for v in scores if v['branch']==U and (v['label'] or ids is None or v['sample_id'] in ids)]
            base_byid={v['sample_id']:v for v in baseline};ordered=sorted(base_byid);y=[base_byid[s]['label'] for s in ordered]
            original=pair_errors(y,[base_byid[s][scoring] for s in ordered])
            for b in BRANCHES:
                current={v['sample_id']:v for v in scores if v['branch']==b};v=[current[s][scoring] for s in ordered]
                errors=pair_errors(y,v);normal=np.asarray([z for z,label in zip(v,y) if not label],float)
                rows.append(dict(domain=domain,scoring=scoring,normal_group=name,branch=b,normal_images=int((np.asarray(y)==0).sum()),
                    anomaly_images=int(np.sum(y)),ranking_errors=float(errors.sum()),improved_pairs=int((errors<original).sum()),worsened_pairs=int((errors>original).sum()),unchanged_pairs=int((errors==original).sum()),
                    interpretation='ranking pairs, not threshold false alarms/misses'))
                tails.append(dict(domain=domain,scoring=scoring,normal_group=name,branch=b,images=len(normal),mean=float(normal.mean()),
                    p90=float(np.quantile(normal,.9)),p95=float(np.quantile(normal,.95)),p99=float(np.quantile(normal,.99)),max=float(normal.max())))
    return rows,tails


def evaluate(bundle,out,domain,workers=2):
    tick=time.perf_counter();out=st.initialize(out,bundle);prediction.verified(bundle,out,domain)
    previous=restore(bundle,out,domain)
    if previous:return previous
    path=out/'evaluation'/domain;path.mkdir(parents=True,exist_ok=True);samples=prediction.selected(bundle,domain)
    count=sum((s['image_height']//4)*(s['image_width']//4) for s in samples);scratch=path/'_scratch';scratch.mkdir(exist_ok=True)
    label=np.memmap(scratch/'labels.bin',mode='w+',dtype=np.uint8,shape=(count,))
    memory={b:np.memmap(scratch/(b+'.bin'),mode='w+',dtype=np.float32,shape=(count,)) for b in NEW}
    scores=[];checks=[];instances=[];responses=[];map_records=[];metrics=[];offset=0
    old=shared.history(bundle.bg);cases={s['sample_id'] for s in bundle.bg['capture']['selected_cases']}
    import os
    cfg=out/'_runtime/mpl';cfg.mkdir(parents=True,exist_ok=True);os.environ['MPLCONFIGDIR']=str(cfg)
    import matplotlib
    matplotlib.use('Agg');figs=path/'figures';figs.mkdir(exist_ok=True)
    try:
        for i,(s,maps,mask,sc,ck,inst,response,mr) in enumerate(bounded_images(samples,lambda s:image_work(bundle,out,s,old),workers),1):
            n=mask.size;label[offset:offset+n]=mask.ravel()
            for b in NEW:memory[b][offset:offset+n]=maps[b].ravel()
            offset+=n;scores.extend(sc);checks.extend(ck);instances.extend(inst);responses.extend(response);map_records.extend(mr)
            if domain=='background' and s['sample_id'] in cases:figure(maps,mask,s,figs/(s['sample_id']+'.png'))
            if i%25==0 or i==len(samples):print(f'[witness CPU maps] {domain} {i}/{len(samples)}',flush=True)
        if offset!=count:raise ValueError('incomplete all-pixel evaluation')
        for b in BRANCHES:
            local=[v for v in scores if v['branch']==b];historical=bundle.metrics[domain,b] if b in BASES else None
            for scoring in ('CH1','CURRENT'):
                im=image_metrics([v['label'] for v in local],[v[scoring] for v in local])
                if historical and scoring=='CH1':
                    for k,v in im.items():check(checks,domain,'parent_'+k,v,float(historical[k]),atol=1e-12,branch=b)
                metrics.append(dict(domain=domain,branch=b,scoring=scoring,image_count=len(local),**im,
                    pixel_AUROC=float(historical['pixel_AUROC']) if historical else None,pixel_AP=float(historical['pixel_AP']) if historical else None,
                    AUPRO=float(historical['AUPRO']) if historical else None,
                    pixel_metric_origin='historical_verified_reuse:'+historical['parent_metric_file'] if historical else 'new_exact_all_pixels_CPU',
                    AUPRO_status='historical_verified_reuse' if historical else 'pending_original_ADEval_GPU'))
        positive=int(label.sum(dtype=np.int64));counts=dict(pixels=count,positive_pixels=positive,negative_pixels=count-positive)
        if not 0<positive<count:raise ValueError('complete domain must contain positive and negative GT pixels')
        pred_sha=st.sha(out/'prediction_manifests'/(domain+'.json'))
        for b in NEW:
            identity=st.digest([v for v in map_records if v['branch']==b]);f=path/(b+'_exact_pixels.json')
            result=exact_pixel_result(f,bundle.identity,identity,label,memory[b],domain=domain,branch=b,
                prediction_sha=pred_sha,counts=counts)
            for v in metrics:
                if v['branch']==b:v.update(pixel_AUROC=result['AUROC'],pixel_AP=result['AP'],pixel_count=result['pixels'])
        if not all(v['verified'] for v in checks):raise ValueError('baseline numerical reproduction failed')
        ranks,tails=ranking(scores,bundle,domain)
        for f,rows in [('metrics.csv',metrics),('per_image_scores.csv',scores),('numerical_verification.csv',checks),('defect_instances.csv',instances),
            ('response_changes.csv',responses),('map_manifest.csv',map_records),('ranking_summary.csv',ranks),('normal_tail.csv',tails)]:st.write_csv(path/f,rows)
        state=dict(status='CPU_complete_AUPRO_pending',identity=bundle.identity,domain=domain,images=len(samples),
            prediction_sha256=st.sha(out/'prediction_manifests'/(domain+'.json')),numerical_failed_checks=0,seconds=time.perf_counter()-tick,
            files={p.relative_to(path).as_posix():st.sha(p) for p in path.rglob('*') if p.is_file() and '_scratch' not in p.parts and p.name!='status.json'},
            GPU_used=False,model_inference=False,feature_extraction=False)
        st.write_json(path/'status.json',state);return state
    finally:
        for mm in [label,*memory.values()]:mm.flush();mm._mmap.close()
        del label,memory;gc.collect()
        # Only this stage's known scratch files; no recursive deletion.
        for name in ['labels.bin',*(b+'.bin' for b in NEW)]: (scratch/name).unlink()
        scratch.rmdir()
