"""Paired differences and two-level, unweighted support sensitivity summaries."""
from __future__ import annotations

from datetime import datetime, timezone
import numpy as np
from . import inputs as ip, evaluation, prediction
from .selection import DOMAINS, SEEDS
from .storage import csv_bytes, json_bytes
from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0 import analysis as old
from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.method import BRANCHES, U, S, P
from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import pair_errors

METRICS = ('image_AUROC','image_AP','pixel_AUROC','pixel_AP','AUPRO')
COMPARISONS = ((S,U),(P,U),(P,S))


def aggregate(metrics):
    """No image pooling; differences within seed, then domain mean, then seeds."""
    keys = [(r['domain'],int(r['support_seed']),r['branch']) for r in metrics]
    if len(set(keys))!=len(keys): raise ValueError('duplicate metric combination')
    table = dict(zip(keys,metrics)); comparisons = []; macros = []; summaries = []
    for seed in SEEDS:
        for domain in DOMAINS:
            for branch,ref in COMPARISONS:
                for metric in METRICS:
                    a,b=(table.get((domain,seed,x),{}).get(metric) for x in (branch,ref))
                    comparisons.append(dict(domain=domain,support_seed=seed,branch=branch,reference=ref,metric=metric,
                        delta=a-b if a is not None and b is not None else None,
                        direction='branch_minus_reference',units='fraction',
                        status='available' if a is not None and b is not None else 'pending'))
        for branch in BRANCHES:
            row=dict(domain='macro_three_domains',support_seed=seed,branch=branch,aggregation='domain_equal')
            for metric in METRICS:
                values=[table.get((d,seed,branch),{}).get(metric) for d in DOMAINS]
                row[metric]=float(np.mean(values)) if all(v is not None for v in values) else None
            macros.append(row)
        for branch,ref in COMPARISONS:
            for metric in METRICS:
                vals=[r['delta'] for r in comparisons if r['domain'] in DOMAINS and r['support_seed']==seed
                      and r['branch']==branch and r['reference']==ref and r['metric']==metric]
                if len(vals)!=3: raise ValueError('paired domain scope mismatch')
                comparisons.append(dict(domain='macro_three_domains',support_seed=seed,branch=branch,reference=ref,
                    metric=metric,delta=float(np.mean(vals)) if all(v is not None for v in vals) else None,
                    units='fraction',direction='branch_minus_reference',status='available' if all(v is not None for v in vals) else 'pending'))
    for domain in (*DOMAINS,'macro_three_domains'):
        for branch,ref in [(b,None) for b in BRANCHES]+list(COMPARISONS):
            for metric in METRICS:
                if ref is None:
                    rows=metrics+macros
                    vals=[next((r.get(metric) for r in rows if r['domain']==domain and int(r['support_seed'])==s
                                and r['branch']==branch),None) for s in SEEDS]
                else:
                    vals=[next(r['delta'] for r in comparisons if r['domain']==domain and r['support_seed']==s
                               and r['branch']==branch and r['reference']==ref and r['metric']==metric) for s in SEEDS]
                complete=all(v is not None for v in vals)
                row=dict(domain=domain,branch=branch,reference=ref,metric=metric,units='fraction',
                    repetitions=3,available=sum(v is not None for v in vals),status='available' if complete else 'pending',
                    interpretation='three_support_repetitions_same_shared_evaluation_not_independent_data_experiments')
                if complete:
                    row.update(mean=float(np.mean(vals)),sample_std=float(np.std(vals,ddof=1)),minimum=min(vals),maximum=max(vals),
                        positive_count=sum(v>0 for v in vals),zero_count=sum(v==0 for v in vals),negative_count=sum(v<0 for v in vals))
                summaries.append(row)
    return metrics+macros,comparisons,summaries


def ranking_rows(scores):
    output=[]
    for domain in DOMAINS:
        for seed in SEEDS:
            rows=[r for r in scores if r['domain']==domain and r['support_seed']==seed]
            grouped={b:sorted([r for r in rows if r['branch']==b],key=lambda r:r['sample_id']) for b in BRANCHES}
            if not all(grouped.values()): continue
            ids=[r['sample_id'] for r in grouped[U]]
            if any([r['sample_id'] for r in grouped[b]]!=ids for b in BRANCHES):
                raise ValueError('ranking methods use different evaluation images')
            y=[r['label'] for r in grouped[U]]
            errors={b:pair_errors(y,[r['image_score'] for r in grouped[b]]) for b in BRANCHES}
            for branch,ref in COMPARISONS:
                now,base=errors[branch],errors[ref]
                output.append(dict(domain=domain,support_seed=seed,branch=branch,reference=ref,
                    weighted_pair_errors=float(now.sum()),reference_weighted_pair_errors=float(base.sum()),
                    delta_pair_errors=float((now-base).sum()),improved_pairs=int((now<base).sum()),
                    worsened_pairs=int((now>base).sum()),unchanged_pairs=int((now==base).sum()),tie_weight=.5,
                    meaning='normal_anomaly_ranking_not_threshold_false_positives_or_negatives'))
    return output


def spatial_tables(instances, mechanisms):
    groups=[]; details=[]; repairs=[]; rsummary=[]; margin=[]; signatures={}
    for seed in SEEDS:
        rows=[r for r in instances if r['support_seed']==seed]
        # Only aligned complete domain triplets enter summaries; partial remain in raw tables.
        domains=[d for d in DOMAINS if {r['branch'] for r in rows if r['domain']==d}==set(BRANCHES)]
        rows=[r for r in rows if r['domain'] in domains]
        if not rows: continue
        g=old.group_small_instances(rows,domains)
        for d in domains:
            signature=sorted((r['sample_id'],r['instance_id'],r['area_original_pixels'],r['in_smallest_area_quartile']) for r in g if r['domain']==d)
            if d in signatures and signature!=signatures[d]:
                raise ValueError('GT instance list/small-area groups changed between support seeds')
            signatures[d]=signature
        groups.extend(dict(r,support_seed=seed) for r in g)
        det,rep=old.margin_repairs(rows)
        details.extend(dict(r,support_seed=seed) for r in det)
        repairs.extend(dict(r,support_seed=seed) for r in rep)
        mech=[r for r in mechanisms if r['support_seed']==seed and r['domain'] in domains]
        rsummary.extend(dict(r,support_seed=seed) for r in old.mechanism_summary(mech,domains))
        for domain in domains:
            for small in (False,True):
                for definition in ('margin','fixed_union_margin'):
                    selected=[r for r in det if r['domain']==domain and r['definition']==definition
                              and (not small or r['smallest_area_quartile'])]
                    valid=[r for r in selected if r['computable']]
                    margin.append(dict(domain=domain,support_seed=seed,group='smallest_area_quartile' if small else 'all',
                        definition=definition,reference=U,branch=P,total=len(selected),not_computable=len(selected)-len(valid),
                        improved=sum(r['pair_change_vs_union']>1e-6 for r in valid),
                        worsened=sum(r['pair_change_vs_union']< -1e-6 for r in valid),
                        unchanged=sum(abs(r['pair_change_vs_union'])<=1e-6 for r in valid),tolerance=1e-6))
    return {'small_instance_groups.csv':groups,'margin_repair_per_instance.csv':details,
            'margin_repair_summary.csv':repairs,'instance_margin_summary.csv':margin,'mechanism_summary.csv':rsummary}


def report(metrics,sensitivity,margin,status):
    lines=['# AeBAD-S 固定K8支持集敏感性', '', '状态：'+status+'.',
        '27个组合均在每域同一共同评价子集上计算；不复用旧完整集合的汇总指标。',
        'CH1为主评分；CURRENT只作seed42历史审计。所有差值为分支减参照，单位fraction（乘100为百分点）。',
        '', '| 域 | 支持seed | 方法 | I-AUROC | I-AP | P-AUROC | P-AP | AUPRO |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for r in metrics:
        lines.append('| '+r['domain']+' | '+str(r['support_seed'])+' | '+r['branch']+' | '+
                     ' | '.join('pending' if r.get(k) is None else f'{r[k]:.9f}' for k in METRICS)+' |')
    lines+=['','UNION的支持波动（域等权后，三个支持重复）：']
    for r in sensitivity:
        if r['domain']=='macro_three_domains' and r['branch']==U and r['reference'] is None:
            lines.append(f"- {r['metric']}: "+(f"mean={r['mean']:.9f}, sample std={r['sample_std']:.9f}, range=[{r['minimum']:.9f},{r['maximum']:.9f}]" if r['status']=='available' else 'pending'))
    lines+=['','SEGMENT/PAIR收益是否跨支持重复：']
    for r in sensitivity:
        if r['domain']=='macro_three_domains' and r['reference']==U:
            lines.append(f"- {r['branch']} − UNION / {r['metric']}: "+(
                f"mean={r['mean']:+.9f}, std={r['sample_std']:.9f}; 正/零/负={r['positive_count']}/{r['zero_count']}/{r['negative_count']}" if r['status']=='available' else 'pending'))
    lines+=['','定位收益与缺陷保持需分开判断：']
    for r in margin:
        lines.append(f"- {r['domain']}/seed{r['support_seed']}/{r['group']}/{r['definition']}: PAIR−UNION margin 改善/不变/下降={r['improved']}/{r['unchanged']}/{r['worsened']}; NA={r['not_computable']}。")
    lines+=['','R_N/R_A/G保留有符号值，图像等权后层等权，参照为同seed UNION。小缺陷按每域GT原始面积最小四分位，三seed共用分组；映射消失、空环带保持NA。',
        '固定UNION峰值仅在同一支持seed内共享；跨seed峰值可不同，不能当作同一物理位置比较。',
        '三个支持重复共享评价集，不能据此作强独立样本显著性声明，也不挑选最优支持组。',
        '像素指标改善不自动证明缺陷margin保持。排序错误不是阈值下误报/漏检；没有预先假定机制成立或达到SOTA。']
    return ('\n'.join(lines)+'\n').encode('utf-8')


def run(ctx):
    ledger=ctx['ledger']; ledger.verify()
    metrics=[]; scores=[]; instances=[]; mechanisms=[]; pending=[]; resources=[]
    for domain in DOMAINS:
        for seed in SEEDS:
            for branch in BRANCHES:
                cpu=evaluation.read_unit(ctx,'cpu',domain,seed,branch)
                if not cpu:
                    pending.append(dict(domain=domain,support_seed=seed,branch=branch,stage='CPU_and_AUPRO')); continue
                metric=dict(cpu['metric']); scores.extend(cpu['scores']); instances.extend(cpu['instances']); mechanisms.extend(cpu['mechanisms'])
                resources.append(dict(domain=domain,support_seed=seed,branch=branch,stage='CPU',**cpu['resources'],**cpu['timing']))
                gpu=evaluation.read_unit(ctx,'aupro',domain,seed,branch)
                if gpu:
                    if gpu['cpu_files']!=ledger.get('cpu/'+evaluation.unit_name(domain,seed,branch))['files']:
                        raise ValueError('AUPRO parent CPU record differs')
                    metric.update(AUPRO=gpu['AUPRO'],AUPRO_status='original_ADEval_complete')
                    resources.append(dict(domain=domain,support_seed=seed,branch=branch,stage='AUPRO',**gpu['resources'],**gpu['timing'],
                        CUDA_peak_allocated_bytes=gpu['CUDA_peak_allocated_bytes'],CUDA_peak_reserved_bytes=gpu['CUDA_peak_reserved_bytes']))
                else: pending.append(dict(domain=domain,support_seed=seed,branch=branch,stage='AUPRO'))
                metrics.append(metric)
    metric_table,comparisons,sensitivity=aggregate(metrics)
    spatial=spatial_tables(instances,mechanisms)
    checks=[dict(r) for k,v in ledger.state['records'].items() if k.startswith('capture/') for r in v.get('checks',[])]
    historical=[]
    for domain in DOMAINS:
        samples=[s for s in ctx['plan']['samples'] if s['domain']==domain]
        if all(ledger.get('capture/'+s['domain']+'/'+s['sample_id']) for s in samples):
            historical.extend(old.historical_comparisons(ctx,samples,checks))
    status='all_requested_metrics_complete' if len(metrics)==27 and not pending else 'partial_evidence'
    tables={'metrics.csv':metric_table,'metric_comparisons.csv':comparisons,'support_sensitivity_summary.csv':sensitivity,
            'per_image_scores.csv':scores,'ranking_summary.csv':ranking_rows(scores),'defect_instances.csv':instances,
            'mechanism_per_image.csv':mechanisms,'numerical_verification.csv':checks,
            'historical_common_subset_CURRENT_comparison.csv':historical,'resource_records.csv':resources,**spatial}
    timestamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'); prefix='summaries/'+timestamp
    files={prefix+'/'+n:csv_bytes(r) for n,r in tables.items()}
    completion=dict(status=status,identity=ctx['identity'],completed_CPU_combinations=len(metrics),
        completed_AUPRO_combinations=27-len(pending),pending=pending,expected_combinations=27,expected_images=878,
        all_requested_metrics_complete=status=='all_requested_metrics_complete',model_inference=False)
    files[prefix+'/status.json']=json_bytes(completion)
    files[prefix+'/objective_report.md']=report(metric_table,sensitivity,spatial['instance_margin_summary.csv'],status)
    ledger.commit(prefix,completion,files); ledger.verify()
    return dict(completion,summary_directory=str(ledger.root/prefix))
