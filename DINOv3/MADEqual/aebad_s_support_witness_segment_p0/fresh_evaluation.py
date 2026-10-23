"""Fresh four-way evaluation; no historical pixel metrics or instance references."""
import gc
import math
import time
from pathlib import Path

import numpy as np

from . import storage as st, prediction, evaluation as old_eval
from . import fresh_inputs as fi
from .method import BRANCHES, BASES, NEW, U, S, W, C, LAYERS
from .execution import bounded_images
from .reporting import margin_transitions, METRICS
from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import image_metrics
from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic.numerics import check
from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.analysis import spatial, U as SPATIAL_U, S as SPATIAL_S


def values(bundle, out, sample):
    arrays, meta, record = prediction.load_record(bundle,out,sample['domain'],sample['sample_id'])
    raw, _, origin = bundle.load(sample)
    if origin != meta['origin'] or not np.array_equal(raw['positions'],arrays['positions']):
        raise ValueError('new-run capture changed after prediction freeze')
    return {U:raw['union_distance'],S:raw['segment_distance'],**{b:arrays[b] for b in NEW}}, arrays, meta, record


def sample_work(bundle, out, sample):
    d,sid = sample['domain'],sample['sample_id']
    distances, arrays, meta, record = values(bundle,out,sample)
    ctx = dict(medians=bundle.medians,scales=bundle.scales,state=bundle.state)
    maps, checks, scores = {}, [], []
    frozen = {r['branch']:r for r in record['metadata']['scores']}
    for b in BRANCHES:
        current,ch1,response,_ = old_eval.shared.score_maps(ctx,distances[b],dict(meta,positions=arrays['positions']))
        for k,v in (('CH1',ch1),('CURRENT',current)):
            check(checks,sid,'fresh_frozen_'+k,v,frozen[b][k],domain=d,branch=b)
        maps[b] = response
        scores.append(dict(domain=d,sample_id=sid,relative_path=sample['relative_path'],label=sample['label'],branch=b,CH1=ch1,CURRENT=current))
    if not all(r['verified'] for r in checks):
        raise ValueError('fresh frozen scores differ from evaluation maps')
    with fi.Data(bundle.paths['data']) as data:
        mask = data.gt(sample)
    small = old_eval.shared.resized_mask(mask,meta['output_shape']).astype(np.uint8)
    rename = {U:SPATIAL_U,S:SPATIAL_S}
    inverse = {v:k for k,v in rename.items()}
    rows = spatial(sample,mask,{rename.get(b,b):m for b,m in maps.items()},tuple(meta['grid'])) if sample['label'] else []
    for r in rows:
        r['branch'] = inverse.get(r['branch'],r['branch'])
        for region in ('instance','ring'):
            a = (r['own_'+region+'_y'],r['own_'+region+'_x'])
            b = (r['union_anchor_'+region+'_y'],r['union_anchor_'+region+'_x'])
            r['own_'+region+'_peak_moved'] = a!=b if a[0] is not None and b[0] is not None else None
    regions = {'normal_all_pixels':np.ones_like(small,bool)} if not sample['label'] else {
        'anomaly_GT_pixels':small.astype(bool),'anomaly_normal_pixels':~small.astype(bool)}
    responses = []
    for b in BRANCHES:
        delta = maps[U].astype(np.float64)-maps[b]
        for group,region in regions.items():
            responses.append(dict(domain=d,sample_id=sid,branch=b,group=group,pixels=int(region.sum()),
                mean_signed_response_reduction=float(delta[region].mean()) if region.any() else None,
                reference=U,units='fused_Source_patch_MAD_response'))
    manifests = [dict(domain=d,sample_id=sid,branch=b,output_height=m.shape[0],output_width=m.shape[1],
        map_float32_sha256=old_eval.shared.array_sha(m),mask_uint8_sha256=old_eval.shared.array_sha(small)) for b,m in maps.items()]
    return sample,maps,small,scores,checks,rows,responses,manifests


def mark_small(rows, domain):
    originals = sorted((r for r in rows if r['branch']==U),
                       key=lambda r:(r['area_original_pixels'],r['sample_id'],r['instance_id']))
    n = min(3,len(originals)) if domain=='background' else math.ceil(len(originals)/4)
    small = {(r['sample_id'],r['instance_id']) for r in originals[:n]}
    for r in rows:
        r['small_group'] = (r['sample_id'],r['instance_id']) in small
        r['small_group_definition'] = 'background_smallest_three_original_area' if domain=='background' else 'domain_smallest_area_quartile'


def evaluate(bundle,out,domain,workers):
    tick = time.perf_counter()
    # Global 926-image prediction completion precedes any GT decoding.
    for d in fi.DOMAINS: prediction.verified(bundle,out,d)
    if old_eval.restore(bundle,out,domain): return dict(status='CPU_reused',domain=domain)
    path = out/'evaluation'/domain
    path.mkdir(parents=True,exist_ok=True)
    samples = prediction.selected(bundle,domain)
    count = sum((s['image_height']//4)*(s['image_width']//4) for s in samples)
    scratch = path/'_scratch'; scratch.mkdir(exist_ok=True)
    labels = np.memmap(scratch/'labels.bin',dtype=np.uint8,mode='w+',shape=(count,))
    memory = {b:np.memmap(scratch/(b+'.bin'),dtype=np.float32,mode='w+',shape=(count,)) for b in BRANCHES}
    scores,checks,instances,responses,maps_manifest = [],[],[],[],[]
    offset = 0
    try:
        for i,(sample,maps,mask,sc,ck,inst,resp,mr) in enumerate(bounded_images(samples,lambda s:sample_work(bundle,out,s),workers),1):
            n = mask.size
            labels[offset:offset+n] = mask.ravel()
            for b in BRANCHES: memory[b][offset:offset+n] = maps[b].ravel()
            offset += n
            scores.extend(sc); checks.extend(ck); instances.extend(inst); responses.extend(resp); maps_manifest.extend(mr)
            if domain=='background' and sample['sample_id'] in fi.CASE_IDS:
                fig = path/'figures'/(sample['sample_id']+'.png'); fig.parent.mkdir(exist_ok=True)
                old_eval.figure(maps,mask,sample,fig)
            if i%25==0 or i==len(samples): print(f'[witness fresh CPU] {domain} {i}/{len(samples)}',flush=True)
        if offset != count: raise ValueError('incomplete exact pixel population')
        positive = int(labels.sum(dtype=np.int64))
        counts = dict(pixels=count,positive_pixels=positive,negative_pixels=count-positive)
        mark_small(instances,domain)
        metrics = []
        pred_sha = st.sha(out/'prediction_manifests'/(domain+'.json'))
        for b in BRANCHES:
            print(f'[witness fresh exact pixels] {domain}/{b} {count}',flush=True)
            map_id = st.digest([r for r in maps_manifest if r['branch']==b])
            pixel = old_eval.exact_pixel_result(path/(b+'_exact_pixels.json'),bundle.identity,map_id,labels,memory[b],
                domain=domain,branch=b,prediction_sha=pred_sha,counts=counts)
            local = [r for r in scores if r['branch']==b]
            for scoring in ('CH1','CURRENT'):
                metrics.append(dict(domain=domain,branch=b,scoring=scoring,image_count=len(local),
                    **image_metrics([r['label'] for r in local],[r[scoring] for r in local]),
                    pixel_AUROC=pixel['AUROC'],pixel_AP=pixel['AP'],AUPRO=None,
                    pixel_metric_origin='fresh_exact_all_pixels_CPU',AUPRO_status='pending_original_ADEval_GPU'))
        ranks,tails = old_eval.ranking(scores,bundle,domain)
        for f,rows in [('metrics.csv',metrics),('per_image_scores.csv',scores),('numerical_verification.csv',checks),
            ('defect_instances.csv',instances),('response_changes.csv',responses),('map_manifest.csv',maps_manifest),
            ('ranking_summary.csv',ranks),('normal_tail.csv',tails)]: st.write_csv(path/f,rows)
        state = dict(status='CPU_complete_AUPRO_pending',identity=bundle.identity,domain=domain,images=len(samples),
            prediction_sha256=pred_sha,numerical_failed_checks=0,seconds=time.perf_counter()-tick,
            files={p.relative_to(path).as_posix():st.sha(p) for p in path.rglob('*') if p.is_file() and '_scratch' not in p.parts and p.name!='status.json'},
            historical_metrics_reused=False,model_inference=False)
        st.write_json(path/'status.json',state)
        return state
    finally:
        for mm in [labels,*memory.values()]: mm.flush(); mm._mmap.close()
        del labels,memory; gc.collect()
        for name in ['labels.bin',*(b+'.bin' for b in BRANCHES)]: (scratch/name).unlink()
        scratch.rmdir()


def aupro(bundle,out,domain,branch,device):
    import torch
    from importlib.metadata import version
    from DINOv3.relation_reliability.official_metrics import official_aupro_adeval
    if branch not in BRANCHES or version('adeval')!='1.1.0': raise ValueError('original ADEval 1.1.0/four branches required')
    if old_eval.restore(bundle,out,domain) is None: raise ValueError('completed exact CPU evaluation required')
    rows = [r for r in old_eval.ip.parent.csv_file(out/'evaluation'/domain/'map_manifest.csv') if r['branch']==branch]
    path = out/'aupro'/domain/(branch+'.json')
    map_id = st.digest(rows)
    if path.exists():
        saved = st.read_json(path)
        if (saved['identity']!=bundle.identity or saved['map_identity']!=map_id or saved['nstrips']!=200 or saved['FPR_max']!=.3
            or saved['result_sha256']!=st.digest({k:v for k,v in saved.items() if k!='result_sha256'})
            or not np.isfinite(saved['AUPRO']) or not 0<=saved['AUPRO']<=1): raise ValueError('AUPRO result modified/different')
        return saved
    dev = torch.device(device); torch.cuda.set_device(dev); torch.cuda.reset_peak_memory_stats(dev)
    tick = time.perf_counter(); responses,masks = [],[]
    byid = {r['sample_id']:r for r in rows}
    ctx = dict(medians=bundle.medians,scales=bundle.scales,state=bundle.state)
    with fi.Data(bundle.paths['data']) as data:
        for sample in prediction.selected(bundle,domain):
            distances,arrays,meta,_ = values(bundle,out,sample)
            response = old_eval.shared.score_maps(ctx,distances[branch],dict(meta,positions=arrays['positions']))[2]
            mask = old_eval.shared.resized_mask(data.gt(sample),meta['output_shape']).astype(np.uint8)
            old_eval.shared.verify_map_record(byid[sample['sample_id']],response,mask)
            responses.append(response); masks.append(mask)
    value = official_aupro_adeval(masks,responses,nstrips=200)
    if len(responses)!=len(rows) or not np.isfinite(value) or not 0<=value<=1: raise ValueError('AUPRO incomplete/invalid')
    result = dict(identity=bundle.identity,domain=domain,branch=branch,map_identity=map_id,AUPRO=float(value),
        nstrips=200,FPR_max=.3,backend='original_ADEval_CUDA',verified_map_GT_images=len(rows),seconds=time.perf_counter()-tick,
        torch=torch.__version__,CUDA=torch.version.cuda,GPU=torch.cuda.get_device_name(dev),ADEval=version('adeval'),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(dev),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(dev))
    result['result_sha256'] = st.digest(result); st.write_json(path,result)
    return result


def summarize(bundle,out):
    from .supports import load_tables, validate_loo
    metrics,instances,scores,responses,checks,ranks,tails,loo = [],[],[],[],[],[],[],[]
    pending = []
    read_csv = old_eval.ip.parent.csv_file
    for d in fi.DOMAINS:
        if old_eval.restore(bundle,out,d) is None:
            pending.append(d+': CPU'); continue
        local = read_csv(out/'evaluation'/d/'metrics.csv')
        for r in local:
            for k in METRICS: r[k] = float(r[k]) if r[k] not in ('',None) else None
            p = out/'aupro'/d/(r['branch']+'.json')
            if not p.exists(): pending.append(d+': AUPRO '+r['branch']); continue
            result = st.read_json(p)
            manifest = [x for x in read_csv(out/'evaluation'/d/'map_manifest.csv') if x['branch']==r['branch']]
            if (result['identity']!=bundle.identity or result['map_identity']!=st.digest(manifest)
                or result['result_sha256']!=st.digest({k:v for k,v in result.items() if k!='result_sha256'})
                or result['nstrips']!=200 or result['FPR_max']!=.3 or not np.isfinite(result['AUPRO']) or not 0<=result['AUPRO']<=1):
                raise ValueError('fresh AUPRO identity/value changed')
            r.update(AUPRO=result['AUPRO'],AUPRO_status='fresh_original_ADEval_CUDA')
        metrics.extend(local)
        for f,dest in [('defect_instances.csv',instances),('per_image_scores.csv',scores),('response_changes.csv',responses),
            ('numerical_verification.csv',checks),('ranking_summary.csv',ranks),('normal_tail.csv',tails)]: dest.extend(read_csv(out/'evaluation'/d/f))
        for l in LAYERS:
            p = out/'support_loo'/d/f'L{l}.json'
            if not p.exists(): pending.append(f'{d}/L{l}: LOO'); continue
            saved = st.read_json(p); _,_,ni = load_tables(bundle,out,d,l)
            validate_loo(saved,bundle.identity,d,l,ni); loo.extend(saved['rows'])
    if len(metrics)==24:
        for b in BRANCHES:
            for scoring in ('CH1','CURRENT'):
                local = [r for r in metrics if r['branch']==b and r['scoring']==scoring and r['domain'] in fi.DOMAINS]
                metrics.append(dict(domain='macro_3domains',branch=b,scoring=scoring,aggregation='three_domains_equal',
                    **{k:float(np.mean([r[k] for r in local])) if all(r[k] is not None for r in local) else None for k in METRICS}))
    comparisons=[]; lookup={(r['domain'],r['scoring'],r['branch']):r for r in metrics}
    for d in (*fi.DOMAINS,'macro_3domains'):
        for scoring in ('CH1','CURRENT'):
            for ref in (U,S,C):
                for k in METRICS:
                    a,b=lookup.get((d,scoring,W),{}).get(k),lookup.get((d,scoring,ref),{}).get(k)
                    comparisons.append(dict(domain=d,scoring=scoring,branch=W,reference=ref,metric=k,
                        value=a,reference_value=b,delta=a-b if a is not None and b is not None else None,
                        delta_percentage_points=100*(a-b) if a is not None and b is not None else None,direction='candidate_minus_reference'))
    transfer,transfer_summary=margin_transitions(instances)
    mechanism=[]
    for d in fi.DOMAINS:
        for b in BRANCHES:
            fields={}
            for group,name in [('normal_all_pixels','R_N'),('anomaly_GT_pixels','R_A')]:
                a=[float(r['mean_signed_response_reduction']) for r in responses if (r['domain'],r['branch'],r['group'])==(d,b,group) and r['mean_signed_response_reduction'] not in ('',None)]
                fields[name]=float(np.mean(a)) if a else None
            fields['G']=fields['R_N']-fields['R_A'] if all(v is not None for v in fields.values()) else None
            mechanism.append(dict(domain=d,branch=b,reference=U,aggregation='group_image_equal',**fields))
    reductions=[];connections=[]
    for d in fi.DOMAINS:
        for f,dest in [('reduction_summary.csv',reductions),('connection_summary.csv',connections)]:
            p=out/'prediction_tables'/d/f
            if p.exists(): dest.extend(read_csv(p))
    for f,rows in [('metrics.csv',metrics),('metric_comparisons.csv',comparisons),('defect_instances.csv',instances),
        ('per_image_scores.csv',scores),('numerical_verification.csv',checks),('ranking_summary.csv',ranks),('normal_tail.csv',tails),
        ('response_changes.csv',responses),('mechanism_summary.csv',mechanism),('support_loo_metrics.csv',loo),
        ('margin_transitions.csv',transfer),('margin_transition_summary.csv',transfer_summary),
        ('reduction_summary.csv',reductions),('connection_summary.csv',connections)]: st.write_csv(out/f,rows)
    failed=sum(str(r['verified']).lower() not in ('true','1') for r in checks)
    complete=not pending and len(metrics)==32 and failed==0
    status=dict(status='all_requested_stages_complete' if complete else 'partial_evidence',all_metrics_complete=complete,
        identity=bundle.identity,images=926,pending=sorted(set(pending)),numerical_failed_checks=failed,
        mode=fi.REVISION,historical_metrics_reused=False,Source_and_MAD_rebuilt=True)
    st.write_json(out/'status.json',status)
    lines=['# WITNESS：从原图重新构建的固定配置实验','',
        '本轮从原521张Source正常图重建原100K/layer memory，index%8==0的66张只作Source校准；'
        '沿用原构库增强、100轮采样、原始Seed42 K8、926张评价集、Exact检索、MAD、CH1/CURRENT和WITNESS公式。',
        '四分支所有像素指标均在本轮重新评价；未引用历史指标，不声称新银行、MAD或距离与旧产物逐字节相同。'
        'GT仅在全部926张预测冻结后解码。三个域已参与开发，不能称为盲测。','',
        '|域|评分|分支|Image AUROC %|Image AP %|Pixel AUROC %|Pixel AP %|AUPRO %|',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for r in metrics: lines.append('|'+ '|'.join([r['domain'],r['scoring'],r['branch'],*['pending' if r[k] is None else f'{100*r[k]:.5f}' for k in METRICS]])+'|')
    lines += ['', '差值为候选减参照，比例及百分点均见metric_comparisons.csv。'
        'margin损失与部分/完全恢复见margin_transition_summary.csv；实例数量不是固定阈值漏检数。'
        '正常区域R_N、缺陷区域R_A及G采用融合Source MAD像素响应、组内图像等权。'
        '不根据本轮结果追加阈值、层选择或参数搜索。', '',str(status)]
    (out/'objective_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return status
