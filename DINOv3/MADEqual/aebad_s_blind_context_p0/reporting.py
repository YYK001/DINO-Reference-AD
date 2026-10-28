"""Background-only objective summary, explicit pending and resource records."""
import numpy as np
from . import inputs as ip
from .inputs import st
from .protocol import BRANCHES,MAIN,U,S,COMBINED,job_plan

METRICS=('image_AUROC','image_AP','pixel_AUROC','pixel_AP','AUPRO')


def summaries(instances,changes):
    margins=[]
    for definition in ('margin','fixed_union_margin'):
        for group in ('all','small'):
            for b in (S,MAIN):
                rows=[r for r in instances if r['branch']==b and (group=='all' or str(r['small_group']).lower()=='true')]
                vals=[float(r[definition+'_change_vs_'+U]) for r in rows if r[definition+'_change_vs_'+U] not in (None,'')]
                v=np.asarray(vals)
                margins.append(dict(domain='background',branch=b,group=group,definition=definition,instances=len(rows),computable=len(vals),NA=len(rows)-len(vals),
                                    damaged=int((v<-1e-6).sum()),improved=int((v>1e-6).sum()),unchanged=int((np.abs(v)<=1e-6).sum()),
                                    mean=float(v.mean()) if len(v) else None,median=float(np.median(v)) if len(v) else None,
                                    p05=float(np.quantile(v,.05)) if len(v) else None,p95=float(np.quantile(v,.95)) if len(v) else None,tolerance=1e-6))
    groups={}
    for name in ('normal_all_pixels','anomaly_GT_pixels'):
        values=[float(r['mean_signed_response_reduction']) for r in changes if r['group']==name and r['mean_signed_response_reduction'] not in ('',None)]
        groups[name]=float(np.mean(values)) if values else None
    rn,ra=groups['normal_all_pixels'],groups['anomaly_GT_pixels']
    mechanism=dict(domain='background',branch=MAIN,reference=U,R_N=rn,R_A=ra,G=rn-ra if rn is not None and ra is not None else None,
                   units='final fused original Source patch MAD response',aggregation='per image mean then image equal within group')
    return margins,mechanism


def summarize(a):
    p=ip.parent(a);pending=[];metrics=[dict(r,metric_origin='parent_readonly_not_reevaluated') for r in p.metrics]
    if (a.run_root/'predictions_frozen.json').exists():
        from .prediction import verified
        verified(a)
    if (a.run_root/'evaluation/status.json').exists():
        from .evaluation import restored
        restored(a);new=ip.read_csv(a.run_root/'evaluation/metrics.csv');apath=a.run_root/'aupro'/f'{MAIN}.json'
        if apath.exists():
            from DINOv3.MADEqual.aebad_s_fixed_target_graph_p0.evaluation import validate_aupro
            rows=[r for r in ip.read_csv(a.run_root/'evaluation/map_manifest.csv') if r['branch']==MAIN]
            au=st.read_json(apath);validate_aupro(au,dict(identity=ip.identity(a),domain='background',branch=MAIN,map_identity=st.digest(rows),nstrips=200,FPR_max=.3))
            if au['verified_images']!=297:raise ValueError('incomplete AUPRO record')
            for r in new:r['AUPRO']=au['AUPRO']
        else:pending.append('one original ADEval AUPRO')
        metrics.extend(new)
        inst=ip.read_csv(a.run_root/'evaluation/defect_instances.csv');changes=ip.read_csv(a.run_root/'evaluation/response_changes.csv')
        margins,mechanism=summaries(inst,changes);st.write_csv(a.run_root/'margin_summary.csv',margins);st.write_csv(a.run_root/'mechanism_summary.csv',[mechanism])
    else:pending.extend(['CPU exact evaluation','one original ADEval AUPRO'])
    for name in ('context_ready.json','training_ready.json','trained.json','predictions_frozen.json'):
        if not (a.run_root/name).exists():pending.append(name)
    isolation=a.run_root/'isolation.json'
    if not isolation.exists():pending.append('input isolation')
    else:
        from .training import require_isolation
        try:require_isolation(a,trained=(a.run_root/'trained.json').exists())
        except (OSError,ValueError,KeyError):pending.append('trained-head input isolation')
    st.write_csv(a.run_root/'metrics.csv',metrics);differences=[]
    lookup={(r['branch'],r['scoring']):r for r in metrics}
    for rule in ('CH1','CURRENT'):
        if (MAIN,rule) not in lookup:continue
        for ref in (U,S,COMBINED):
            for k in METRICS:
                value=lookup[MAIN,rule].get(k);old=lookup[ref,rule].get(k)
                differences.append(dict(domain='background',branch=MAIN,reference=ref,scoring=rule,metric=k,
                                   delta_pp=100*(float(value)-float(old)) if value not in ('',None) and old not in ('',None) else None))
    st.write_csv(a.run_root/'metric_comparisons.csv',differences)
    costs=[]
    for f in sorted((a.run_root/'costs').glob('*.json')):
        r=st.read_json(f);costs.append({k:r.get(k) for k in ('stage','seconds','process_peak_rss_bytes','peak_cuda_allocated_bytes','peak_cuda_reserved_bytes','full_four_layer_extractions','host_internal_crop_forwards','new_original_extractions')})
    st.write_csv(a.run_root/'stage_costs.csv',costs)
    value=dict(status='pending' if pending else 'all_five_metrics_complete',identity=ip.identity(a),pending=pending,
               primary=MAIN,evaluation_images=297,metric_rows=len(metrics),new_AUPRO_jobs=1,training_seed=42,
               numerical_failed_checks=0 if (a.run_root/'evaluation/status.json').exists() else None,
               cross_seed_stability='not evaluated',configurations=1,job_plan=job_plan(),
               statement='single background development domain; no automatic expansion')
    st.write_json(a.run_root/'status.json',value)
    lines=['# 盲点上下文正常参照合成：固定background验证', '',
           '状态：'+value['status']+'；pending：'+str(pending),
           '本表CH1为主，CURRENT为补充；同一定位图只评价一次。历史三分支只读复用。', '',
           '|评分|分支|Image AUROC|Image AP|Pixel AUROC|Pixel AP|AUPRO|', '|---|---|---:|---:|---:|---:|---:|']
    for r in metrics:
        vals=['pending' if r.get(k) in ('',None) else f'{100*float(r[k]):.5f}' for k in METRICS]
        lines.append('|'+ '|'.join([r['scoring'],r['branch']]+vals)+'|')
    lines+=['', '比较表见metric_comparisons.csv；实例自身/固定UNION峰值、小实例及NA见margin_summary.csv。',
            '损失计数不是固定阈值下漏检数。R_N/R_A/G使用相同Source MAD融合响应，先逐图平均。', '',
            '## 必须回答的固定问题',
            '1. 相对完整UNION是否有实际图像/定位收益？是否减少SEGMENT的定位代价？',
            '2. 相对SEGMENT评分＋UNION定位是否有足够幅度的优势？哪些指标仍下降？',
            '3. margin受损的数量和幅度、小实例与NA是否支持缺陷保持？',
            '4. 信息隔离与吞吐是否通过？隔离通过只证明通路，不证明检测有效。',
            '未获得完整结果前以上问题保持待回答；不按测试结果挑checkpoint、扩域或搜参。',
            '没有额外控制，不能声称上下文是收益的唯一原因。单个background开发域不能外推所有域或支持集稳定性。',
            '少量固定历史案例见evaluation/figures；前向/内部裁块和复用计数见阶段costs与context_ready记录。']
    (a.run_root/'objective_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return value
