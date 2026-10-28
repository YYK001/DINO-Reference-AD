"""Explicit historical/new tables, strength transitions and honest pending state."""
import numpy as np
from . import inputs as ip,prediction as pred,evaluation as ev
from .inputs import st
from .protocol import CONFIG,DOMAINS,BRANCHES,MAIN,TARGET,UNIFORM,HALF,U,S,HISTORY,MAIN_TABLE,ALL_TABLE,COMBINED,job_plan

METRICS=('image_AUROC','image_AP','pixel_AUROC','pixel_AP','AUPRO')


def number(value):return None if value in ('',None) else float(value)


def distribution(values,prefix):
    tol=CONFIG['margin_tolerance'];x=np.asarray(values,np.float64)
    return {prefix+k:v for k,v in dict(damaged=int(np.sum(x < -tol)),improved=int(np.sum(x>tol)),unchanged=int(np.sum(np.abs(x)<=tol)),
        mean=float(x.mean()) if len(x) else None,median=float(np.median(x)) if len(x) else None,
        p05=float(np.quantile(x,.05)) if len(x) else None,p95=float(np.quantile(x,.95)) if len(x) else None).items()}


def margins(a,instances):
    parent=ip.parent(a);details=[];tol=CONFIG['margin_tolerance']
    for r in instances:
        key=(r['domain'],r['sample_id'],str(r['instance_id']));b=r['branch'];old=parent.instances[*key,HALF[b]];u=parent.instances[*key,U]
        for definition in ('margin','fixed_union_margin'):
            x,h,base=(number(v[definition]) for v in (r,old,u));valid=all(v is not None for v in (x,h,base))
            row=dict(domain=r['domain'],sample_id=r['sample_id'],instance_id=r['instance_id'],branch=b,half_branch=HALF[b],definition=definition,
                small_group=str(r['small_group']).lower()=='true',computable=valid,response_status=r['response_status'],reference=U,
                units='shared_frozen_Source_patch_MAD_final_fused_margin')
            if valid:
                previous=h-base;current=x-base;change=x-h
                row.update(half_change_vs_UNION=previous,L1_change_vs_UNION=current,L1_change_vs_half=change,
                    half_damaged=previous < -tol,L1_damaged=current < -tol,
                    new_damage_since_half=previous>=-tol and current < -tol,
                    previous_damage_further_worsened=previous < -tol and change < -tol,
                    previous_damage_improved=previous < -tol and change > tol,
                    previous_damage_fully_restored=previous < -tol and current>=-tol)
            details.append(row)
    summary=[]
    for d in DOMAINS:
        for b in BRANCHES:
            for definition in ('margin','fixed_union_margin'):
                for group in ('all','small'):
                    selected=[r for r in details if (r['domain'],r['branch'],r['definition'])==(d,b,definition) and (group=='all' or r['small_group'])];valid=[r for r in selected if r['computable']]
                    summary.append(dict(domain=d,branch=b,half_branch=HALF[b],definition=definition,group=group,instances=len(selected),computable=len(valid),NA=len(selected)-len(valid),
                        **distribution([r['half_change_vs_UNION'] for r in valid],'L05_'),**distribution([r['L1_change_vs_UNION'] for r in valid],'L1_'),
                        **distribution([r['L1_change_vs_half'] for r in valid],'L1_minus_L05_'),
                        **{k:sum(r[k] for r in valid) for k in ('new_damage_since_half','previous_damage_further_worsened','previous_damage_improved','previous_damage_fully_restored')},tolerance=tol))
    return details,summary


def summarize(a):
    pred.verify(a);parent=ip.parent(a);new=[];instances=[];changes=[];checks=ip.read_csv(a.run_root/'replay_verification.csv');pending=[];score_rows=[];jobs_done=0
    for d in DOMAINS:
        if not ev.restored(a,d):pending.append(d+'/CPU');continue
        path=a.run_root/'evaluation'/d;local=ip.read_csv(path/'metrics.csv');maps=ip.read_csv(path/'map_manifest.csv')
        if {(r['branch'],r['scoring']) for r in local}!={(b,s) for b in BRANCHES for s in ('CH1','CURRENT')}:raise ValueError('incomplete new metric rows')
        for r in local:
            for k in METRICS:r[k]=number(r.get(k))
            dest=a.run_root/'aupro'/d/(r['branch']+'.json')
            if dest.exists():
                v=st.read_json(dest);ev.validate_aupro(v,dict(identity=ip.identity(a),domain=d,branch=r['branch'],map_identity=st.digest([m for m in maps if m['branch']==r['branch']]),nstrips=200,FPR_max=.3),CONFIG['populations'][d][0]);r['AUPRO']=v['AUPRO']
                if r['scoring']=='CH1':jobs_done+=1
            else:pending.append(d+'/'+r['branch']+'/AUPRO')
            r.update(metric_origin='this_run_new_L1',source_identity=ip.identity(a),source_run=str(a.run_root),strength=1.,table_role='main')
        new.extend(local);instances.extend(ip.read_csv(path/'defect_instances.csv'));changes.extend(ip.read_csv(path/'response_changes.csv'));checks.extend(ip.read_csv(path/'numerical_verification.csv'));score_rows.extend(ip.read_csv(path/'per_image_scores.csv'))
    if len(new)==len(DOMAINS)*len(BRANCHES)*2:
        for b in BRANCHES:
            for rule in ('CH1','CURRENT'):
                selected=[r for r in new if (r['branch'],r['scoring'])==(b,rule)]
                new.append(dict(domain='macro_3domains',branch=b,scoring=rule,aggregation='three_domains_equal',
                    **{k:float(np.mean([r[k] for r in selected])) if all(r[k] is not None for r in selected) else None for k in METRICS},
                    metric_origin='this_run_new_L1',source_identity=ip.identity(a),source_run=str(a.run_root),strength=1.,table_role='main'))
    history=[dict(r,**{k:number(r[k]) for k in METRICS},metric_origin='parent_readonly_reused_not_reevaluated',source_identity=parent.identity,source_run=str(parent.run),
        strength=.5 if r['branch'].startswith('SACP_') else None,table_role='main' if r['branch'] in MAIN_TABLE else 'appendix_only') for r in parent.metrics]
    rows=history+new;index={(r['domain'],r['branch'],r['scoring']):r for r in rows};comparison=[];strength=[]
    for r in new:
        for k in METRICS:
            old=index[r['domain'],HALF[r['branch']],r['scoring']]
            strength.append(dict(domain=r['domain'],scoring=r['scoring'],branch=r['branch'],reference=HALF[r['branch']],metric=k,delta_pp=100*(r[k]-old[k]) if r[k] is not None else None))
        if r['branch']!=MAIN:continue
        for ref in (HALF[MAIN],U,TARGET,UNIFORM,COMBINED):
            base=index.get((r['domain'],ref,r['scoring']))
            if base:
                for k in METRICS:comparison.append(dict(domain=r['domain'],scoring=r['scoring'],branch=MAIN,reference=ref,metric=k,
                    delta_pp=100*(r[k]-base[k]) if r[k] is not None and base[k] is not None else None))
    mechanism=[]
    for d in DOMAINS:
        for b in BRANCHES:
            means={}
            for group,name in (('normal_all_pixels','R_N'),('anomaly_GT_pixels','R_A')):
                values=[number(r['mean_signed_response_reduction']) for r in changes if (r['domain'],r['branch'],r['group'])==(d,b,group) and r['mean_signed_response_reduction']!=''];means[name]=float(np.mean(values)) if values else None
            mechanism.append(dict(domain=d,branch=b,reference=U,**means,G=means['R_N']-means['R_A'] if all(v is not None for v in means.values()) else None,
                units='final_fused_original_Source_patch_MAD_response',aggregation='image_equal_within_group',normal_group='all pixels of normal images',anomaly_group='resized GT pixels of abnormal images',metric_origin='this_run_new_L1'))
    details,margin_summary=margins(a,instances)
    numerical_failed=sum(str(r['verified']).lower()!='true' for r in checks)
    if numerical_failed:pending.append('numerical_failed_checks')
    plan=job_plan();requested=len(plan['AUPRO_jobs'])
    status=dict(status='all_five_metrics_complete' if not pending and jobs_done==requested and len(rows)==plan['complete_metric_rows'] else 'pending',
        identity=ip.identity(a),parent_identity=parent.identity,primary=MAIN,pending=sorted(set(pending)),evaluation_images=len(parent.catalog['samples']),
        metric_rows=len(rows),main_metric_rows=sum(r['branch'] in MAIN_TABLE for r in rows),new_AUPRO_requested=requested,new_AUPRO_complete=jobs_done,
        historical_metrics_reused=len(history),historical_pixel_or_AUPRO_recomputed=False,numerical_failed_checks=numerical_failed,
        query_forward_calls=0,exact_search_calls=0,subspace_fit_calls=0,training_updates=0)
    sources=[dict(domain=r['domain'],branch=r['branch'],scoring=r['scoring'],source_run=str(parent.run),source_identity=parent.identity,source_file='metrics.csv',origin='parent_readonly_reused_not_reevaluated') for r in history]
    full_scores=[dict(r,score_origin='parent_readonly',source_identity=parent.identity) for r in parent.scores.values()]+[dict(r,score_origin='this_run_new_L1',source_identity=ip.identity(a)) for r in score_rows]
    full_scores.extend(dict(r,branch=COMBINED,score_origin='parent_SEGMENT_score_fixed_UNION_map',source_identity=parent.identity) for r in parent.scores.values() if r['branch']==S)
    tables={'metrics.csv':rows,'main_metrics.csv':[r for r in rows if r['branch'] in MAIN_TABLE],'metric_comparisons.csv':comparison,'strength_effects.csv':strength,
        'historical_metric_sources.csv':sources,'per_image_scores.csv':full_scores,'defect_instances.csv':instances,'margin_changes.csv':details,'margin_summary.csv':margin_summary,
        'response_changes.csv':changes,'mechanism_summary.csv':[dict(r,metric_origin='parent_readonly') for r in parent.mechanisms]+mechanism,'numerical_verification.csv':checks}
    for name,data in tables.items():st.write_csv(a.run_root/name,data)
    costs=[]
    for path in sorted((a.run_root/'costs').glob('*.json')):
        r=st.read_json(path);costs.append(dict(record=path.name,stage=r['stage'],seconds=r['seconds'],process_peak_rss_bytes=r['process_peak_rss_bytes'],
            peak_cuda_allocated_bytes=r.get('peak_cuda_allocated_bytes'),peak_cuda_reserved_bytes=r.get('peak_cuda_reserved_bytes'),
            **{k:r[k] for k in ('query_forward_calls','exact_search_calls','subspace_fit_calls','training_updates')}))
    st.write_csv(a.run_root/'stage_costs.csv',costs)
    st.write_json(a.run_root/'status.json',status)
    report=['# SACP固定强度扩展：lambda=1','',str(status),'',
        '仅三个新增定位分支。原lambda=.5及全部历史行只读继承；CH1主表，CURRENT补充，不按域选lambda。完整表含Source方向历史附表，主表不含它。',
        '','|域|分支|Image AUROC|Image AP|Pixel AUROC|Pixel AP|AUPRO|来源|','|---|---|---:|---:|---:|---:|---:|---|']
    for d in (*DOMAINS,'macro_3domains'):
        for b in MAIN_TABLE:
            r=index.get((d,b,'CH1'))
            if r:report.append('|'+d+'|'+b+'|'+'|'.join('pending' if r[k] is None else f'{100*r[k]:.5f}' for k in METRICS)+'|'+r['metric_origin']+'|')
    report+=['','## 固定问题','']
    for ref in (HALF[MAIN],U,TARGET,UNIFORM,COMBINED):
        local=[r for r in comparison if (r['domain'],r['scoring'],r['reference'])==('macro_3domains','CH1',ref)]
        report.append('新主方法减'+ref+'：'+(', '.join(f"{r['metric']} {r['delta_pp']:+.5f} pp" for r in local) if len(local)==5 and all(r['delta_pp'] is not None for r in local) else 'pending；不作完整结论'))
    report+=['','提高强度是否带来足够收益，要结合上述五项、逐域结果和局部损失；不因某一个正向指标自动宣布有效。',
        '实例见margin_summary.csv：L05与L1并列受损/改善/基本不变、均值/中位数/5%/95%分位，以及新增损伤和旧损伤进一步恶化；固定UNION峰值和自身峰值均报告，小实例与原NA保留。计数不是部署阈值下漏检。',
        'R_N/R_A/G沿用父定义：UNION减候选最终Source MAD融合响应，正常图全部像素/异常图GT像素，图像等权；原生缩距及相对比例另见native_reductions.csv，不混用单位。',
        '无新MAD、前向、Exact、子空间拟合或训练。CPU结果不能代替原ADEval；9项新AUPRO全部有效才完成。强度只比较.5和1，不自动延伸其他lambda/rank/方向/层。',
        'Source对照是否值得保留必须看新主方法相对新Target方向和等量均匀缩距，不以eta附表代替效果。未形成优势时如实收束当前固定方向与两档强度；不外推所有子空间方法无效。',
        '成本记录见costs；复用历史结果不是本轮重算，也不构成与从零实验比较的加速倍数。三域为开发域，单组K8，不宣称跨支持稳定或盲测。']
    report+=['','## CURRENT补充','',
        '|域|分支|Image AUROC|Image AP|Pixel AUROC|Pixel AP|AUPRO|','|---|---|---:|---:|---:|---:|---:|']
    for d in (*DOMAINS,'macro_3domains'):
        for b in MAIN_TABLE:
            r=index.get((d,b,'CURRENT'))
            if r:report.append('|'+d+'|'+b+'|'+'|'.join('pending' if r[k] is None else f'{100*r[k]:.5f}' for k in METRICS)+'|')
    report+=['','## 实例损失与强度变化','',
        '表内损伤相对UNION，阈值为1e-6；新增损伤指.5未受损但1受损，继续恶化指.5已受损且1进一步降低margin。原NA不计作零损伤证据。全部分支/小实例/分位数详见margin_summary.csv。',
        '','|域|分支|口径|可计算|NA|.5受损|1受损|新增受损|旧损失继续恶化|.5均值|1均值|1减.5中位数|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in margin_summary:
        if r['group']=='all':report.append('|'+str(r['domain'])+'|'+r['branch']+'|'+r['definition']+'|'+'|'.join(str(r[k]) for k in ('computable','NA','L05_damaged','L1_damaged','new_damage_since_half','previous_damage_further_worsened'))+'|'+'|'.join('NA' if r[k] is None else f'{r[k]:+.7f}' for k in ('L05_mean','L1_mean','L1_minus_L05_median'))+'|')
    (a.run_root/'objective_report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    return status
