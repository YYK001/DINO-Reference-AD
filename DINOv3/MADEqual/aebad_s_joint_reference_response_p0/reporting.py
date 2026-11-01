"""Fixed comparisons and equal-domain/seed tables. No winner selection."""
import math
import numpy as np
from . import inputs as ip, evaluation as ev, prediction as pred
from .inputs import st
from .protocol import *


def aggregate(groups, domain, seed):
    """Complete matching branch/readout keys only; missing metrics remain NA."""
    indexes = [{(r['branch'], r['scoring']): r for r in rows} for rows in groups]
    keys = set(indexes[0])
    if any(set(i) != keys for i in indexes):raise ValueError('incomplete macro comparison keys')
    result = []
    for b, rule in sorted(keys):
        values = {}
        for k in METRICS:
            x = [i[b, rule].get(k) for i in indexes]
            values[k] = None if any(v in ('', None) for v in x) else float(np.mean([float(v) for v in x], dtype=np.float64))
        result.append(dict(domain=domain, support_seed=seed, branch=b, scoring=rule,
                           aggregation='equal_domain_then_equal_support_seed', **values))
    return result


def reference_metrics(p, d, seed):
    old = {(r['branch'], r['scoring']): r for r in p.settings[d, seed]['metrics']}
    segment = old[S, 'CH1'];rows = []
    for b in (U, S, SOURCE):
        current = old[b, 'WEIGHTED_MAP_MAX' if b == SOURCE else 'CURRENT']
        for rule in READOUTS:
            image = segment if rule == 'SEGMENT_CH1' else current
            rows.append(dict(domain=d, support_seed=seed, branch=b, scoring=rule,
                **{k: float((image if k.startswith('image_') else current)[k]) for k in METRICS},
                map_origin='read_only_completed_parent_common_902', image_reference=S if rule == 'SEGMENT_CH1' else b,
                image_original_scoring='CH1' if rule == 'SEGMENT_CH1' else current['scoring'], image_metrics_inherited=True))
    return rows


def summarize(a):
    p = ip.parent(a);rows = [];history = [];pending = [];groups = {};au_done = 0;checks = [];image_scores=[]
    globally_frozen = (a.run_root/'predictions_frozen.json').exists()
    if globally_frozen:pred.global_freeze(a)
    else:pending.append('global_prediction_freeze_before_GT')
    for d in DOMAINS:
        for seed in SEEDS:
            values = reference_metrics(p, d, seed)
            history.extend(dict(r, domain=d, support_seed=seed) for r in p.settings[d, seed]['metrics'])
            for b in MAPS:
                if globally_frozen and ev.completed(a, d, seed, b):
                    item = ip.audit.parent_reporting.rows(ev.destination(a, d, seed, b)/'metrics.csv')
                    expected, _ = ev.au_expected(a, d, seed, b);path = ip.unit(a, d, seed)/'aupro'/(b+'.json')
                    if path.exists():
                        au = st.read_json(path);ev.validate_aupro(au, expected)
                        if au['verified_images'] != len(p.samples(d)):raise ValueError('partial AUPRO')
                        au_done += 1
                        for r in item:r['AUPRO'] = au['AUPRO']
                    else:pending.append(f'{d}/seed{seed}/{b}/AUPRO')
                    for r in item:
                        if r['scoring'] == 'SEGMENT_CH1':
                            baseline = next(v for v in values if v['branch'] == S and v['scoring'] == r['scoring'])
                            if any(float(r[k]) != baseline[k] for k in ('image_AUROC', 'image_AP')):
                                raise ValueError('new-map table A image metrics differ from frozen SEGMENT')
                    values.extend(item)
                else:
                    pending.append(f'{d}/seed{seed}/{b}/CPU')
                    for rule in READOUTS:
                        seg = next(v for v in values if v['branch'] == S and v['scoring'] == 'SEGMENT_CH1')
                        values.append(dict(domain=d, support_seed=seed, branch=b, scoring=rule,
                             **{k: seg[k] if rule == 'SEGMENT_CH1' and k.startswith('image_') else None for k in METRICS}))
            rows.extend(values);groups[d, seed] = values
            if (ip.unit(a, d, seed)/'predictions_frozen.json').exists():
                for row in p.samples(d):
                    rec = st.sealed_json(ip.record_path(a, d, seed, row))
                    checks.extend(dict(c, domain=d, support_seed=seed, sample_id=row['sample_id']) for c in rec['checks'])
                    image_scores.extend(dict(domain=d, support_seed=seed, sample_id=row['sample_id'], branch=b,
                                             image=row['relative_path'], **rec['scores'][b]) for b in MAPS)
    if not (a.run_root/'cases_status.json').exists():pending.append('fixed_cases_and_response_distributions')
    else:
        cases = st.sealed_json(a.run_root/'cases_status.json')
        if cases['records'] != 48 or cases['prediction_seal'] != pred.global_freeze(a)['seal']:
            raise ValueError('fixed case population/prediction identity differs')
        for name, sha in cases['files'].items():
            if st.sha(a.run_root/name) != sha:raise ValueError('fixed case artifact differs')
    for r in rows:
        for k in METRICS:
            if r.get(k) not in (None, '') and (not math.isfinite(float(r[k])) or not 0 <= float(r[k]) <= 1):
                raise ValueError('invalid formal metric')
    if checks and any(not c['verified'] for c in checks):raise ValueError('failed numerical checks')
    index = {(r['domain'], int(r['support_seed']), r['branch'], r['scoring']): r for r in rows}
    differences = []
    for d in DOMAINS:
        for seed in SEEDS:
            for rule in READOUTS:
                for reference in COMPARISONS:
                    for k in METRICS:
                        new, old = (index[d, seed, b, rule].get(k) for b in (JOINT, reference))
                        differences.append(dict(domain=d, support_seed=seed, scoring=rule, branch=JOINT, reference=reference, metric=k,
                                                delta_pp=None if new in ('', None) or old in ('', None) else 100*(float(new)-float(old))))
    per_seed = [r for seed in SEEDS for r in aggregate([groups[d, seed] for d in DOMAINS], 'three_domain_macro', seed)]
    per_domain = [r for d in DOMAINS for r in aggregate([groups[d, seed] for seed in SEEDS], d, 'two_seed_mean')]
    overall = aggregate([[r for r in per_seed if r['support_seed'] == seed] for seed in SEEDS], 'overall', 'equal_seed')
    st.write_csv(a.run_root/'metrics.csv', rows);st.write_csv(a.run_root/'table_A_SEGMENT_CH1.csv', [r for r in rows if r['scoring'] == 'SEGMENT_CH1'])
    st.write_csv(a.run_root/'per_image_scores.csv', image_scores)
    st.write_csv(a.run_root/'table_B_MAP_MAX.csv', [r for r in rows if r['scoring'] == 'MAP_MAX'])
    st.write_csv(a.run_root/'historical_metrics_read_only.csv', history);st.write_csv(a.run_root/'metric_comparisons.csv', differences)
    st.write_csv(a.run_root/'per_seed_macro.csv', per_seed);st.write_csv(a.run_root/'per_domain_two_seed_mean.csv', per_domain)
    st.write_csv(a.run_root/'overall_macro.csv', overall);st.write_csv(a.run_root/'numerical_verification.csv', checks)
    status = dict(status='pending' if pending else 'all_five_metrics_complete', pending=sorted(set(pending)),
                  unique_images=902, repeated_prediction_instances=1804, complete_AUPRO_jobs=au_done, requested_AUPRO_jobs=24,
                  numerical_failed_checks=0, numerical_checks=len(checks), training=False, Source_rebuild=False, parameter_search=False,
                  model_configurations=1, primary=JOINT, support_groups_are_not_independent_test_sets=True)
    st.write_json(a.run_root/'status.json', status)
    lines = ['# Source–Target联合正常响应：单侧联合尾部建模', '',
             '固定问题：两路信息是否优于先取min？同位置配对是否优于两路各自分布？是否超过完整K8 Source加权定位参照？', '',
             '状态：'+status['status']+'；待办 '+str(len(status['pending']))+'；新增AUPRO '+str(au_done)+'/24。',
             '仅本组8张Target正常图构造八个Source+K7分支，正常/评价查询使用相同B_j；Source提供完整锚点及原MAD，未混入66张Source校准响应。',
             '固定h=1.0、每支持最多512个物理位置、原层和提取。先平均概率再取负对数；统计量不是缺陷概率、校准p值或有限样本FPR保证。', '',
             '表A所有定位图统一搭配同组冻结SEGMENT CH1；图像指标是继承能力。表B统一最终定位图MAP_MAX，不能称为CH1；历史独立Source加权PL及原组合另保存在historical_metrics_read_only.csv。', '']
    for rule in READOUTS:
        lines += ['## '+('表A：固定SEGMENT评分＋定位图' if rule == 'SEGMENT_CH1' else '表B：定位图自身MAP_MAX'), '',
                  '|域|支持seed|地图|Image AUROC %|Image AP %|Pixel AUROC %|Pixel AP %|AUPRO %|', '|---|---|---|---:|---:|---:|---:|---:|']
        for r in rows:
            if r['scoring'] == rule:
                lines.append('|'+ '|'.join([r['domain'], str(r['support_seed']), r['branch']]+['pending' if r.get(k) in (None, '') else f'{100*float(r[k]):.5f}' for k in METRICS])+'|')
    lines += ['', '## 固定比较与结论边界', '',
              '逐设置四组固定差值见metric_comparisons.csv；每seed三域等权、每域两seed等权、先域后seed总体结果全部保留，不池化重复图像。',
              '未完成时不能回答效益；完成后也不选最佳seed/域/对照，不自动修改h、采样量或概率汇总。三个域都有开发历史，902张重复评价不构成1804张独立样本。']
    if not pending:
        for ref in COMPARISONS:
            for metric in ('pixel_AP', 'AUPRO'):
                delta = [r for r in differences if r['reference'] == ref and r['metric'] == metric and r['scoring'] == 'SEGMENT_CH1']
                lines.append('- '+JOINT+' − '+ref+' '+metric+'，六设置百分点差值：'+', '.join(f"{r['domain']}/seed{r['support_seed']} {r['delta_pp']:+.5f}" for r in delta))
        lines += ['优于U4但未优于独立对照不能归因于配对；优于独立但仍低于完整K8强参照不能宣布总体有效。定位与自身图像判断分别裁决，反向域不被小宏平均遮蔽。']
    (a.run_root/'objective_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return status
