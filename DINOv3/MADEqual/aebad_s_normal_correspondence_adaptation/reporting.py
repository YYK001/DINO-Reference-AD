"""Results are populated only from actual frozen predictions and completed jobs."""
from pathlib import Path
import math
import numpy as np
from . import storage as st,inputs as ip,evaluation as ev
from .protocol import DOMAINS,BRANCHES,MEMBERS,ENS,U,S,COMBINED

METRICS=('image_AUROC','image_AP','pixel_AUROC','pixel_AP','AUPRO')


def combine(rows):
    """Fixed engineering reference: SEGMENT image scores, UNION pixel map."""
    out=list(rows)
    for d in dict.fromkeys(r['domain'] for r in rows):
        for rule in ('CH1','CURRENT'):
            source=next((r for r in rows if (r['domain'],r['scoring'],r['branch'])==(d,rule,U)),None)
            score=next((r for r in rows if (r['domain'],r['scoring'],r['branch'])==(d,rule,S)),None)
            if source and score:out.append(dict(source,branch=COMBINED,image_AUROC=score['image_AUROC'],image_AP=score['image_AP'],
                image_metric_origin=S,pixel_metric_origin=U+' reused; not reevaluated',role='fixed_task_division_not_new_model'))
    return out


def comparisons(rows):
    out=[]
    for r in rows:
        if r['branch'] not in (*MEMBERS,ENS):continue
        for ref in (U,S,COMBINED):
            base=next(v for v in rows if (v['domain'],v['scoring'],v['branch'])==(r['domain'],r['scoring'],ref))
            for k in METRICS:
                delta=100*(r[k]-base[k]) if r[k] is not None and base[k] is not None else None
                out.append(dict(domain=r['domain'],scoring=r['scoring'],branch=r['branch'],reference=ref,
                    metric=k,delta=delta,units='percentage_points',direction='branch_minus_reference',status='complete' if delta is not None else 'pending'))
    return out


def margins(rows):
    aligned={}
    for r in rows:aligned.setdefault((r['domain'],r['sample_id'],r['instance_id']),{})[r['branch']]=r
    details=[]
    for (d,s,i),values in aligned.items():
        if set(values)!=set(BRANCHES):raise ValueError('instance branches incomplete')
        for definition in ('margin','fixed_union_margin'):
            def number(b):
                v=values[b][definition];return None if v in ('',None) else float(v)
            u=number(U)
            for b in BRANCHES:
                v=number(b);valid=u is not None and v is not None
                details.append(dict(domain=d,sample_id=s,instance_id=i,branch=b,definition=definition,
                    small_group=str(values[b]['small_group']).lower()=='true',computable=valid,
                    response_status=values[b]['response_status'],change=v-u if valid else None,
                    damaged=(v<u-1e-6) if valid else None,
                    units='difference of independently normal-calibrated responses, not original feature distance'))
    summary=[]
    for d in DOMAINS:
        for b in BRANCHES:
            for definition in ('margin','fixed_union_margin'):
                for group in ('all','small'):
                    selected=[r for r in details if (r['domain'],r['branch'],r['definition'])==(d,b,definition) and (group=='all' or r['small_group'])]
                    valid=[r for r in selected if r['computable']]
                    summary.append(dict(domain=d,branch=b,definition=definition,group=group,instances=len(selected),
                        NA=len(selected)-len(valid),damaged=sum(r['damaged'] for r in valid),
                        mean_change=float(np.mean([r['change'] for r in valid])) if valid else None))
    return details,summary


def summarize(a):
    rows=[];instances=[];pending=[];failed=0
    for d in DOMAINS:
        if not ev.restored(a,d):pending.append(d+': CPU');continue
        local=ev.read_csv(a.run_root/'evaluation'/d/'metrics.csv')
        instances.extend(ev.read_csv(a.run_root/'evaluation'/d/'defect_instances.csv'))
        failed+=sum(r['verified'].lower()!='true' for r in ev.read_csv(a.run_root/'evaluation'/d/'numerical_verification.csv'))
        for r in local:
            for k in METRICS:r[k]=float(r[k]) if r[k] not in ('',None) else None
            p=a.run_root/'aupro'/d/(r['branch']+'.json')
            if p.exists():
                from DINOv3.MADEqual.aebad_s_fixed_target_graph_p0.evaluation import validate_aupro
                mr=[v for v in ev.read_csv(a.run_root/'evaluation'/d/'map_manifest.csv') if v['branch']==r['branch']]
                record=st.read_json(p);validate_aupro(record,dict(identity=ip.identity(a),domain=d,branch=r['branch'],map_identity=st.digest(mr),nstrips=200,FPR_max=.3))
                r['AUPRO']=record['AUPRO'];r['AUPRO_status']='complete_original_ADEval'
            else:pending.append(d+'/'+r['branch']+': AUPRO')
        rows.extend(local)
    if len(rows)==len(DOMAINS)*len(BRANCHES)*2:
        for b in BRANCHES:
            for rule in ('CH1','CURRENT'):
                values=[r for r in rows if r['branch']==b and r['scoring']==rule]
                rows.append(dict(domain='macro_3domains',branch=b,scoring=rule,aggregation='three_domains_equal',
                    **{k:float(np.mean([r[k] for r in values])) if all(r[k] is not None for r in values) else None for k in METRICS}))
    rows=combine(rows);st.write_csv(a.run_root/'metrics.csv',rows);deltas=comparisons(rows) if len(rows)>=42 else []
    st.write_csv(a.run_root/'metric_comparisons.csv',deltas)
    repeats=[]
    for d in (*DOMAINS,'macro_3domains'):
        for rule in ('CH1','CURRENT'):
            values=[r for r in rows if r['domain']==d and r['scoring']==rule and r['branch'] in MEMBERS]
            if len(values)!=3:continue
            for k in METRICS:
                v=[r[k] for r in values]
                repeats.append(dict(domain=d,scoring=rule,metric=k,training_seeds='42,43,44',
                    mean=float(np.mean(v)) if all(x is not None for x in v) else None,
                    sample_std=float(np.std(v,ddof=1)) if all(x is not None for x in v) else None,
                    interpretation='three training repeats; same support set; not ensemble metric'))
    st.write_csv(a.run_root/'training_repeat_summary.csv',repeats)
    if instances:
        detail,summary=margins(instances);st.write_csv(a.run_root/'defect_instances.csv',instances)
        st.write_csv(a.run_root/'margin_changes.csv',detail);st.write_csv(a.run_root/'margin_summary.csv',summary)
    # Verify all final seed/domain/layer jobs; no complete status on partial ensembles.
    from .training import job_path
    from .protocol import SEEDS,LAYERS,CONFIG
    for seed in SEEDS:
        for d in ('source',*DOMAINS):
            for l in LAYERS:
                p=job_path(a,d,seed,l)/'complete.json'
                if not p.exists():pending.append(f'{d}/seed{seed}/L{l}: training')
                else:
                    record=st.sealed_json(p)
                    if record['updates']!=CONFIG['source_steps' if d=='source' else 'target_steps']:raise ValueError('incomplete final model')
    if failed:pending.append('numerical_failed_checks='+str(failed))
    value=dict(status='all_five_metrics_complete' if not pending and len(rows)==56 else 'pending',
        identity=ip.identity(a),pending=sorted(set(pending)),numerical_failed_checks=failed,
        training_repeats_are_not_support_repeats=True)
    st.write_json(a.run_root/'status.json',value)
    report=['# Source–Target正常对应残差适配：固定方案结果','',
        '这是CAReg思路启发的工程候选，不是官方复现；三域曾用于开发，不是盲测。',
        'CH1主表；CURRENT见metrics.csv。同一分支两评分共用定位图。表中单位为%。','',
        '|域|分支|Image AUROC|Image AP|Pixel AUROC|Pixel AP|AUPRO|',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['scoring']=='CH1':report.append('|'+r['domain']+'|'+r['branch']+'|'+
            '|'.join('pending' if r[k] is None else f'{100*r[k]:.5f}' for k in METRICS)+'|')
    report += ['','## 主候选相对各参照（百分点；候选减参照）','']
    for r in deltas:
        if r['branch']==ENS and r['scoring']=='CH1':
            report.append(f"- {r['domain']} / {r['reference']} / {r['metric']}: "+('pending' if r['delta'] is None else f"{r['delta']:+.5f}"))
    report += ['', '## 结果判断与成本', '']
    for d in (*DOMAINS,'macro_3domains'):
        for ref in (U,S,COMBINED):
            selected=[r for r in deltas if (r['domain'],r['branch'],r['reference'],r['scoring'])==(d,ENS,ref,'CH1')]
            if len(selected)!=5 or any(r['delta'] is None for r in selected):
                report.append(f'- {d} / {ref}：指标未齐，暂不作完整优势判断。');continue
            up=[r['metric'] for r in selected if r['delta']>0];down=[r['metric'] for r in selected if r['delta']<0]
            report.append(f'- {d} / {ref}：上升={up}；下降={down}；'+
                ('五项均提高，但不是统计显著性证明。' if len(up)==5 else '未形成五项一致优势，需保留上述取舍。'))
    if instances:
        for r in summary:
            if r['branch']==ENS and r['group']=='all':report.append(
                f"- {r['domain']} / {r['definition']}：集成相对UNION受损{r['damaged']}个，可计算{r['instances']-r['NA']}，NA={r['NA']}。")
    for p in sorted((a.run_root/'costs').glob('*.json')):
        cost=st.read_json(p);report.append(f"- 已记录成本 {cost['stage']}：{cost['seconds']:.2f}s；峰值RAM={cost['process_peak_rss_bytes']} bytes；显存分配峰值={cost.get('peak_cuda_allocated_bytes','未初始化CUDA')}。")
    report += ['','比较必须同时看图像、定位、实例损失与三种子波动；不按测试指标重新选权重、种子或checkpoint。',
        'SEGMENT_SCORE_UNION_MAP仅复用SEGMENT图像指标与UNION像素指标，是任务分工参照。',
        '响应变化使用各模型独立Source校准尺度，不解释为原始特征空间的R_N/R_A。margin下降不等于部署漏检。',
        '实际阶段耗时/峰值RAM/显存见costs；训练重复均值不是集成指标。',
        '', '完成状态：'+str(value)]
    (a.run_root/'objective_report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    return value
