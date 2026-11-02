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
    return result


def comparisons(rows):
    result=[]
    for d in (*DOMAINS,'macro_3domains'):
        idx={(r['branch'],r['scoring']):r for r in rows if r['domain']==d}
        for rule in READOUTS:
            # Same localization reference, original UNION CH1 for own-detection
            # table and frozen SEGMENT CH1 for task-division table.
            for b,ref in [(b,SOURCE) for b in MAPS]+[(B,A),(C,B)]:
                x,y=idx.get((b,rule)),idx.get((ref,rule))
                if x is None or y is None:continue
                for k in METRICS:
                    xx,yy=number(x.get(k)),number(y.get(k))
                    result.append(dict(domain=d,branch=b,reference=ref,scoring=rule,metric=k,
                        delta_pp=None if xx is None or yy is None else 100*(xx-yy),
                        inherited_image_score=rule=='SEGMENT_CH1'))
        # Also expose the actual candidate's own detection tradeoff against the
        # existing strong task-division combination, not only the zero image
        # deltas inherited in table B.
        reference=idx.get(('SEGMENT_SCORE_SOURCE_WEIGHT_MAP','CH1'))
        for b in MAPS:
            candidate=idx.get((b,'CH1'))
            if candidate is None or reference is None:continue
            for k in METRICS:
                xx,yy=number(candidate.get(k)),number(reference.get(k))
                result.append(dict(domain=d,branch=b,reference='SEGMENT_SCORE_SOURCE_WEIGHT_MAP',scoring='CH1',reference_scoring='CH1',metric=k,
                    delta_pp=None if xx is None or yy is None else 100*(xx-yy),inherited_image_score=False))
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
    return result


def summarize(a):
    p=ip.parent(a);pending=[];rows=historical(a);diag=[];checks=[];scores=[]
    frozen=None
    try:frozen=pred.global_freeze(a)
    except FileNotFoundError:pending.append('global_prediction_freeze')
    for d in DOMAINS:
        for b in MAPS:
            if frozen is None or not ev.completed(a,d,b):
                pending.append(d+'/'+b+'/CPU');continue
            metric=ip.rows(ev.destination(a,d,b)/'metrics.csv');au=ev.au_completed(a,d,b)
            if au is None:pending.append(d+'/'+b+'/AUPRO')
            for r in metric:r['AUPRO']=None if au is None else au['AUPRO']
            rows.extend(metric);scores.extend(dict(r,branch=b) for r in ip.rows(ev.destination(a,d,b)/'per_image_scores.csv'))
        if frozen:
            for row in p.samples(d):
                r=st.sealed_json(ip.record_path(a,d,row))
                checks.extend(dict(v,domain=d,sample_id=row['sample_id']) for v in r['checks'])
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
                images=902,support_seed=42,new_AUPRO_requested=9,new_AUPRO_complete=9-sum(x.endswith('/AUPRO') or x.endswith('/CPU') for x in pending if '/' in x),
                training=False,parameter_search=False,numerical_failed_checks=sum(not r['verified'] for r in checks))
    if status['numerical_failed_checks']:status['status']='failed_numerical_verification'
    st.write_json(a.run_root/'status.json',status)
    lines=['# 查询图内共享残差平移：固定验证','',str(status),'',
        '三域有开发历史；仅seed42 K8；共同902张排除两支持组并集。C是预定主候选。',
        '所有位置（包括被评分位置）等权估计，非盲点、不与查询独立。小距离不是收益证据；不保证缺陷保持。',
        '不重新校准，Source权重只用于定位；CH1独立取各层统计。固定SEGMENT读出的图像指标全部继承。','']
    for title,rule in [('表A：候选自身CH1与未修正UNION CH1＋Source加权定位','CH1'),
                       ('表B：固定SEGMENT CH1＋各定位图','SEGMENT_CH1'),('补充：WEIGHTED_MAP_MAX','WEIGHTED_MAP_MAX')]:
        lines+=['## '+title,'','|域|分支|Image AUROC %|Image AP %|Pixel AUROC %|Pixel AP %|AUPRO %|','|---|---|---:|---:|---:|---:|---:|']
        for r in rows:
            if r['scoring']==rule and r['branch'] in (*MAPS,SOURCE):
                lines.append('|'+ '|'.join([r['domain'],r['branch']]+['pending' if number(r.get(k)) is None else f'{100*float(r[k]):.5f}' for k in METRICS])+'|')
    lines+=['','## 客观判读','',
        '逐域、等权macro及B−A、C−B见metric_comparisons.csv。未完成项保持pending，不将其填0。',
        '评估C相对未修正Source加权参照的五项指标；表B单独检验定位能否改善现有强组合。',
        'A→B仅拆分同九候选的重选，B→C拆分分来源均值；不假设C有单调性。',
        '报告改善与下降的域及指标，不以距离下降、响应回升替代实际性能。',
        '这是单支持组、固定强度描述性验证，不证明真实采集状态、创新或普遍有效；不自动追加规则或参数扫描。']
    (a.run_root/'objective_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return status
