"""Fixed comparisons, full per-domain tables and equal-domain macro; no tuning."""
import math
import numpy as np
from . import inputs as ip,prediction as pred,evaluation as ev
from .inputs import st
from .protocol import *


def number(v):return None if v in ('',None) else float(v)


def macro(rows):
    by={d:{(r['branch'],r['scoring']):r for r in rows if r['domain']==d} for d in DOMAINS}
    keys=set.intersection(*(set(v) for v in by.values()))
    result=[]
    for b,sc in sorted(keys):
        result.append(dict(domain='macro_3domains',branch=b,scoring=sc,aggregation='equal_domain_no_image_pooling',
            **{k:None if any(number(by[d][b,sc].get(k)) is None for d in DOMAINS) else
               float(np.mean([float(by[d][b,sc][k]) for d in DOMAINS])) for k in METRICS}))
        for key in ('numeric_path','branch_role','metric_origin','image_reference','map_reference',
                    'inherited_image_score','primary_reference','extra_localization_jobs','extra_AUPRO_jobs'):
            values=[by[d][b,sc].get(key) for d in DOMAINS]
            if values[0] is not None and all(v==values[0] for v in values):result[-1][key]=values[0]
    return result


def comparisons(rows):
    result=[]
    for d in (*DOMAINS,'macro_3domains'):
        idx={(r['branch'],r['scoring']):r for r in rows if r['domain']==d}
        specs=[]
        for rule in READOUTS:
            ref,rr=(STRONG,'CH1') if rule=='SEGMENT_CH1' else (REFERENCE,rule)
            specs.extend((b,rule,ref,rr,'primary_matched_reference') for b in CANDIDATES)
            specs.extend((b,rule,r,rule,'operation_contribution') for b,r in ((B,A),(C,B)))
            specs.extend((b,rule,SOURCE,rule,'historical_secondary') for b in CANDIDATES)
        specs.extend((b,'CH1',STRONG,'CH1','own_detection_vs_matched_strong') for b in CANDIDATES)
        specs.extend((b,'CH1','SEGMENT_SCORE_SOURCE_WEIGHT_MAP','CH1','historical_secondary') for b in CANDIDATES)
        specs.extend([(REFERENCE,'CH1',SOURCE,'CH1','numerical_path_change'),
                      (REFERENCE,'WEIGHTED_MAP_MAX',SOURCE,'WEIGHTED_MAP_MAX','numerical_path_change'),
                      (STRONG,'CH1','SEGMENT_SCORE_SOURCE_WEIGHT_MAP','CH1','numerical_path_change')])
        for b,rule,ref,rr,role in specs:
            x,y=idx.get((b,rule)),idx.get((ref,rr))
            if x is None or y is None:continue
            for k in METRICS:
                xx,yy=number(x.get(k)),number(y.get(k))
                result.append(dict(domain=d,branch=b,reference=ref,scoring=rule,reference_scoring=rr,metric=k,
                    comparison_role=role,delta_pp=None if xx is None or yy is None else 100*(xx-yy),
                    inherited_image_score=rule=='SEGMENT_CH1' or b==STRONG))
    return result


def historical(a):
    result=[];p=ip.parent(a)
    for d in DOMAINS:
        base=p.metrics[d];idx={(r['branch'],r['scoring']):r for r in base}
        # Retain every original metric/readout, including original weighted PL.
        for r in base:result.append(dict(r,domain=d,metric_origin='read_only_parent_common_seed42'))
        u=idx[U,'CH1'];s=idx[S,'CH1'];loc=next(r for r in base if r['branch']==SOURCE)
        for rule,img in (('CH1',u),('WEIGHTED_MAP_MAX',None),('SEGMENT_CH1',s)):
            if rule=='WEIGHTED_MAP_MAX':
                value=idx[SOURCE,'WEIGHTED_MAP_MAX']
            else:
                value=dict(loc,image_AUROC=img['image_AUROC'],image_AP=img['image_AP'])
            if (SOURCE,rule) not in idx:
                result.append(dict(value,domain=d,branch=SOURCE,scoring=rule,
                    image_reference=S if rule=='SEGMENT_CH1' else U,map_reference=SOURCE,
                    metric_origin='fixed_existing_score_map_combination_no_new_prediction'))
    return [dict(r,numeric_path='legacy_MM',branch_role='historical_reference',primary_reference=False) for r in result]


def summarize(a):
    p=ip.parent(a);pending=[];rows=historical(a);diag=[];checks=[];scores=[];arithmetic=[];path_diffs=[];cpu_done=[];au_done=[]
    frozen=None
    try:frozen=pred.global_freeze(a)
    except FileNotFoundError:pending.append('global_prediction_freeze')
    for d in DOMAINS:
        for b in MAPS:
            if frozen is None or not ev.completed(a,d,b):
                pending.append(d+'/'+b+'/CPU');continue
            metric=ip.rows(ev.destination(a,d,b)/'metrics.csv');au=ev.au_completed(a,d,b)
            cpu_done.append((d,b))
            if au is None:pending.append(d+'/'+b+'/AUPRO')
            else:au_done.append((d,b))
            for r in metric:r['AUPRO']=None if au is None else au['AUPRO']
            rows.extend(metric);scores.extend(dict(r,branch=b) for r in ip.rows(ev.destination(a,d,b)/'per_image_scores.csv'))
            if b==REFERENCE:
                inherited=next(r for r in metric if r['scoring']=='SEGMENT_CH1')
                rows.append(dict(inherited,branch=STRONG,scoring='CH1',image_reference=S,map_reference=REFERENCE,
                    metric_origin='fixed_SEGMENT_CH1_direct_reference_map',branch_role='matched_task_division_reference',
                    primary_reference=True,extra_localization_jobs=0,extra_AUPRO_jobs=0))
        if frozen:
            for row in p.samples(d):
                r=st.sealed_json(ip.record_path(a,d,row))
                checks.extend(dict(v,domain=d,sample_id=row['sample_id']) for v in r['checks'])
                arithmetic.extend(r['arithmetic_audit'])
                path_diffs.append(st.read_json(a.run_root/d/'reference_path_differences'/(row['sample_id']+'.json')))
                for v in r['diagnostics']:
                    for b,z in v['branches'].items():diag.append(dict(domain=d,sample_id=row['sample_id'],layer=v['layer'],branch=b,
                        delta_U_norm=v['delta_U_norm'],mean_union_residual_norm=v['mean_union_residual_norm'],delta_U_ratio=v['delta_U_ratio'],
                        zero_union_denominator=v['zero_union_denominator'],**z))
    for r in rows:
        for k in METRICS:
            v=number(r.get(k))
            if v is not None and (not math.isfinite(v) or not 0<=v<=1):raise ValueError('invalid final metric')
    rows+=macro(rows);diff=comparisons(rows)
    st.write_csv(a.run_root/'metrics.csv',rows);st.write_csv(a.run_root/'metric_comparisons.csv',diff)
    st.write_csv(a.run_root/'per_image_scores.csv',scores);st.write_csv(a.run_root/'diagnostics.csv',diag)
    st.write_csv(a.run_root/'numerical_verification.csv',checks)
    st.write_csv(a.run_root/'arithmetic_path_comparisons.csv',arithmetic)
    st.write_csv(a.run_root/'source_selection_switches.csv',[{k:v for k,v in row.items() if k in
        ('domain','sample_id','layer','source_switch_count','source_switch_fraction','positions')} for row in arithmetic])
    st.write_csv(a.run_root/'reference_map_score_changes.csv',path_diffs)
    path_metrics=[r for r in diff if r['comparison_role']=='numerical_path_change']
    st.write_csv(a.run_root/'numerical_path_metric_changes.csv',path_metrics)
    source_diag=[]
    if frozen:
        for d in DOMAINS:
            for row in p.samples(d):
                rec=st.sealed_json(ip.record_path(a,d,row))
                for v in rec['diagnostics']:
                    source_diag.extend(dict(domain=d,sample_id=row['sample_id'],layer=v['layer'],**s) for s in v['sources'])
    st.write_csv(a.run_root/'source_mean_diagnostics.csv',source_diag)
    costs=[st.read_json(path) for path in sorted((a.run_root/'costs').glob('*.json'))]
    st.write_json(a.run_root/'resource_summary.json',dict(recorded_stage_seconds=sum(v['seconds'] for v in costs),
        peak_process_RAM_bytes=max((v['process_peak_rss_bytes'] for v in costs),default=None),
        peak_cuda_allocated_bytes=max((v['peak_cuda_allocated_bytes'] for v in costs if 'peak_cuda_allocated_bytes' in v),default=None),
        peak_cuda_reserved_bytes=max((v['peak_cuda_reserved_bytes'] for v in costs if 'peak_cuda_reserved_bytes' in v),default=None),
        new_array_payload_bytes=sum(v.get('new_array_payload_bytes',0) for v in costs),
        feature_forwards=sum(v.get('feature_forwards',0) for v in costs),
        note='stage records include failed/resumed work; disk counter is newly written array payload, not total IO; current summarize cost written after this snapshot'))
    status=dict(status='pending' if pending else 'all_five_metrics_complete',pending=sorted(set(pending)),primary=C,
                images=902,support_seed=42,new_AUPRO_requested=MAP_JOBS,new_AUPRO_complete=len(au_done),
                candidate_AUPRO_complete=sum(b in CANDIDATES for d,b in au_done),reference_AUPRO_complete=sum(b==REFERENCE for d,b in au_done),
                CPU_jobs_requested=MAP_JOBS,CPU_jobs_complete=len(cpu_done),primary_reference=REFERENCE,strong_reference=STRONG,
                extra_combination_AUPRO_jobs=0,reference_winner='direct_unscaled_norm',job_plan=job_plan(),
                training=False,parameter_search=False,numerical_failed_checks=sum(not r['verified'] for r in checks))
    if status['numerical_failed_checks']:status['status']='failed_numerical_verification'
    st.write_json(a.run_root/'status.json',status)
    lines=['# 查询图内共享残差平移：固定验证','',str(status),'',
        '三域有开发历史；仅seed42 K8；共同902张排除两支持组并集。C是预定主候选。',
        '所有位置（包括被评分位置）等权估计，非盲点、不与查询独立。小距离不是收益证据；不保证缺陷保持。',
        '不重新校准，Source权重只用于定位；CH1独立取各层统计。固定SEGMENT读出的图像指标全部继承。','']
    for title,rule in [('表A：候选自身CH1与同口径直接范数参照','CH1'),
                       ('表B：固定SEGMENT CH1＋各定位图','SEGMENT_CH1'),('补充：WEIGHTED_MAP_MAX','WEIGHTED_MAP_MAX')]:
        lines+=['## '+title,'','|域|分支|Image AUROC %|Image AP %|Pixel AUROC %|Pixel AP %|AUPRO %|','|---|---|---:|---:|---:|---:|---:|']
        for r in rows:
            selected=(r['scoring']==rule and r['branch'] in CANDIDATES) or (r['scoring']==rule and r['branch']==REFERENCE and rule!='SEGMENT_CH1') or (r['branch']==STRONG and r['scoring']=='CH1' and rule=='SEGMENT_CH1')
            if selected:
                lines.append('|'+ '|'.join([r['domain'],r['branch']]+['pending' if number(r.get(k)) is None else f'{100*float(r[k]):.5f}' for k in METRICS])+'|')
    lines+=['','## 新旧参照指标变化（数值路径，不计作方法收益）','',
            '|域|新参照|读出|指标|相对旧参照变化pp|','|---|---|---|---|---:|']
    for r in path_metrics:lines.append('|'+ '|'.join([r['domain'],r['branch'],r['scoring'],r['metric'],'pending' if r['delta_pp'] is None else f'{r["delta_pp"]:+.5f}'])+'|')
    lines+=['','## 历史MM参照（保留，不作主要差值参照）','',
            '|域|分支|读出|Image AUROC %|Image AP %|Pixel AUROC %|Pixel AP %|AUPRO %|','|---|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        if r.get('numeric_path')=='legacy_MM':lines.append('|'+ '|'.join([r['domain'],r['branch'],r['scoring']]+['pending' if number(r.get(k)) is None else f'{100*float(r[k]):.5f}' for k in METRICS])+'|')
    lines+=['','## 客观判读','',
        '逐域、等权macro及B−A、C−B见metric_comparisons.csv。未完成项保持pending，不将其填0。',
        '评估C相对同一直接范数Source加权参照的五项指标；表B比较固定SEGMENT CH1＋直接范数参照图。主要差值仅使用兼容参照，历史MM差值单列。',
        'source_selection_switches.csv逐图逐层报告直接范数与旧MM的来源切换，零修正对新参照仍要求1e-5。新增参照3项CPU＋3项AUPRO，总计12定位作业，固定评分组合不新增地图。',
        'A→B仅拆分同九候选的重选，B→C拆分分来源均值；不假设C有单调性。',
        '报告改善与下降的域及指标，不以距离下降、响应回升替代实际性能。',
        '这是单支持组、固定强度描述性验证，不证明真实采集状态、创新或普遍有效；不自动追加规则或参数扫描。']
    (a.run_root/'objective_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return status
