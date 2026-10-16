"""Explicit CPU image-only evaluation; fitting uses recovered Source normals only."""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .inputs import TOOL, digest, sha, replay, load_context
from .recovery import validate_recovery, calibration_rows, read_cache, verify_patch_mad
from .scoring import (LAYERS, BRANCHES, COMPARISONS, statistics, fit_image_mad,
                      ch1_score, image_metrics, pair_errors, check, complete)


def fit_source_images(ctx, root, ledger, checks):
    values = dict(peak=[], local=[])
    rows, raws = [], []
    records = {r['file']: r for r in ledger['records']}
    for row in calibration_rows(ctx):
        name = f"raw/{row['calibration_index']:03d}.npz"
        raw = read_cache(root / name, row, ctx['identity'])
        raws.append(raw)
        _, peak, local = statistics(raw.reshape(4, -1), np.arange(raw[0].size, dtype=np.int64),
                                    raw.shape[1:], row['output_shape'], ctx['medians'], ctx['scales'])
        for kind, actual in (('peak', peak), ('local', local)):
            check(checks, row['relative_path'], 'CPU_vs_original_' + kind, actual, records[name][kind])
            values[kind].append(actual)
            rows.extend(dict(calibration_index=row['calibration_index'], source_index=row['source_index'],
                             relative_path=row['relative_path'], layer=l, statistic=kind, value=float(actual[j]),
                             original_reference=records[name][kind][j]) for j, l in enumerate(LAYERS))
    verify_patch_mad(raws, ctx, checks)
    state = dict(tool=TOOL, source='66_Source_calibration_to_original_Source_memory',
                 calibration_images=len(raws), image_mad_shared_by=['SOURCE', 'UNION_K8'],
                 patch_mad_frozen_sha256=ctx['frozen']['assets']['source_mad_state.npz'],
                 recovery_inputs_sha256=ledger['inputs_sha256'],
                 recovery_cache_identity=digest([{k: r[k] for k in ('file', 'cache_sha256')} for r in ledger['records']]),
                 layers=list(LAYERS), epsilon=1e-8, fusion_weights=[.5, .5])
    for kind in values:
        medians, scales = fit_image_mad(np.stack(values[kind]))
        state[kind] = dict(medians=medians.tolist(), scales=scales.tolist())
    state['state_sha256'] = digest(state)
    return state, rows


def historical(ctx):
    samples = ctx['original']['capture']['samples']
    formal = replay.read_csv(ctx['paths']['formal'] / 'sample_scores.csv')
    lookup = {(r['method'], r['sample_id']): r for r in formal if r['domain'] == 'background'}
    result = {}
    for branch, method in (('SOURCE_CURRENT', 'MAD_EQUAL'), ('UNION_K8_CURRENT', 'TARGET_MATCHED_K8_MAD')):
        rows = []
        for s in samples:
            r = lookup[method, s['sample_id']]
            if r['relative_path'] != s['relative_path'] or int(r['label']) != s['label']:
                raise ValueError('formal score identity differs from frozen capture')
            rows.append(float(r['image_score']))
        result[branch] = np.asarray(rows, float)
    return result


def ranking_tables(samples, scores, checks, old):
    labels = np.asarray([s['label'] for s in samples])
    normals = [s for s in samples if s['label'] == 0]
    fixed = {sid for sid, _ in replay.CASES[:10]}
    if len(normals) != 85 or not fixed.issubset({s['sample_id'] for s in normals}):
        raise ValueError('original 85/10/75 normal groups required')
    groups = dict(all85=np.ones(85, bool), fixed10=np.array([s['sample_id'] in fixed for s in normals]),
                  other75=np.array([s['sample_id'] not in fixed for s in normals]))
    errors = {b: pair_errors(labels, scores[b]) for b in BRANCHES}
    for b in old:
        check(checks, 'background', b + '_all_pair_states', errors[b], pair_errors(labels, old[b]), atol=0)
    ranking, detail = [], []
    for branch, reference in COMPARISONS:
        before, after = errors[reference], errors[branch]
        delta = after - before
        for name, mask in groups.items():
            ranking.append(dict(branch=branch, reference=reference, normal_group=name,
                                normal_count=int(mask.sum()), anomaly_count=212, pair_count=int(after[mask].size),
                                weighted_pair_errors=float(after[mask].sum()),
                                reference_weighted_pair_errors=float(before[mask].sum()),
                                delta_pair_errors=float(delta[mask].sum()), improved_pairs=int((delta[mask] < 0).sum()),
                                worsened_pairs=int((delta[mask] > 0).sum()), unchanged_pairs=int((delta[mask] == 0).sum()),
                                tie_error_weight=.5, interpretation='ordering_only_not_deployment_errors'))
        ni = ai = 0
        for s in samples:
            d = delta[ni] if s['label'] == 0 else delta[:, ai]
            detail.append(dict(sample_id=s['sample_id'], label=s['label'], branch=branch, reference=reference,
                               weighted_pair_error_change=float(d.sum()), improved_pairs=int((d < 0).sum()),
                               worsened_pairs=int((d > 0).sum()), fixed_difficult_normal=s['sample_id'] in fixed))
            ni += s['label'] == 0
            ai += s['label'] == 1
    return ranking, detail


def metric_tables(samples, scores, old, ctx, checks):
    labels = np.array([s['label'] for s in samples])
    fixed = {sid for sid, _ in replay.CASES[:10]}
    masks = {'all85': np.ones(len(samples), bool),
             'fixed10': np.array([s['label'] == 1 or s['sample_id'] in fixed for s in samples]),
             'other75': np.array([s['label'] == 1 or s['sample_id'] not in fixed for s in samples])}
    metrics = []
    formal = replay.read_csv(ctx['paths']['formal'] / 'domain_metrics.csv')
    # Admit both CURRENT baselines before any new formal image metrics. Scores
    # and ranking differences remain available even if admission fails.
    admitted = True
    for branch in old:
        for name, mask in masks.items():
            actual = image_metrics(labels[mask], scores[branch][mask])
            expected = image_metrics(labels[mask], old[branch][mask])
            for key in actual:
                admitted = check(checks, name, branch + '_' + key, actual[key], expected[key]) and admitted
            if name == 'all85':
                method = 'MAD_EQUAL' if branch == 'SOURCE_CURRENT' else 'TARGET_MATCHED_K8_MAD'
                rows = [r for r in formal if r['method'] == method and r['domain'] == 'background']
                if len(rows) != 1:
                    raise ValueError('formal baseline summary is not unique')
                for metric, field in (('image_AUROC', 'image_AUROC'), ('image_AP', 'image_AUPR')):
                    admitted = check(checks, 'formal_summary', branch + '_' + metric,
                                     actual[metric], float(rows[0][field])) and admitted
    admitted = admitted and all(r['verified'] for r in checks)
    for branch in (BRANCHES if admitted else tuple(old)):
        for name, mask in masks.items():
            current = image_metrics(labels[mask], scores[branch][mask])
            metrics.append(dict(branch=branch, normal_group=name, image_count=int(mask.sum()),
                                normal_count=int((labels[mask] == 0).sum()), anomaly_count=int((labels[mask] == 1).sum()),
                                **current, source='recomputed_CPU_image_only', units='fraction',
                                pixel_metrics='not_re_evaluated'))
    deltas = []
    if not admitted:
        return metrics, deltas
    by = {(r['branch'], r['normal_group']): r for r in metrics}
    for branch, reference in COMPARISONS:
        for group in masks:
            for key in ('image_AUROC', 'image_AP'):
                delta = by[branch, group][key] - by[reference, group][key]
                deltas.append(dict(branch=branch, reference=reference, normal_group=group, metric=key,
                                   delta_fraction=delta, delta_percentage_points=100 * delta,
                                   direction='branch_minus_reference'))
    return metrics, deltas


def evaluate(ctx):
    start = time.perf_counter()
    p = ctx['paths']
    output, root = p['output'], p['calibration']
    if output is None or root is None:
        raise ValueError('evaluate requires --source-calibration-dir and independent --output-dir')
    if output.exists():
        raise ValueError('evaluation output exists; select a new directory, never overwrite')
    ledger = validate_recovery(root, ctx, complete=True)
    if ctx['bundle'] is None:
        raise ValueError('approved local 297-distance archive is required')
    output.mkdir(parents=True)
    replay.atomic_json(output / 'inputs.json', dict(tool=TOOL, identity=ctx['identity'], frozen=ctx['frozen'],
        parent_archive_sha256=sha(p['parent']) if 'parent' in p else 'synthetic_fixture',
        recovery_status_sha256=sha(root / 'recovery_status.json'),
        paths={k: str(v) if v else None for k, v in p.items()}))
    state = dict(tool=TOOL, stage='evaluate', identity=ctx['identity'], status='computing',
                 evaluated_images=0, model_loaded=False, torch_imported='torch' in sys.modules,
                 pixel_metrics_recomputed=False, localization_definition_unchanged=True,
                 cpu_workers=ctx.get('cpu_workers', 1))
    replay.atomic_json(output / 'status.json', state)
    checks, rows = [], []
    try:
        image_state, calibration = fit_source_images(ctx, root, ledger, checks)
        replay.write_csv(output / 'calibration_per_image.csv', calibration)
        replay.atomic_json(output / 'image_calibration_state.json', image_state)
        # Reference parity and frozen patch-MAD validation are admission checks:
        # failure preserves evidence, but does not permit formal test scoring.
        if not all(r['verified'] for r in checks):
            raise ValueError('calibration/reference verification failed')
        samples = ctx['original']['capture']['samples']
        old = historical(ctx)
        scores = {b: [] for b in BRANCHES}
        with ThreadPoolExecutor(max_workers=state['cpu_workers']) as pool:
            # map preserves the frozen manifest order. Each worker owns its
            # maps; no per-image reduction or floating point order is changed.
            for sample, reconstructed in zip(samples, pool.map(lambda sample: image_statistics(ctx, sample), samples)):
                sid = sample['sample_id']
                for memory, (current, peak, local) in reconstructed.items():
                    values = {'CURRENT': current, 'CH1': ch1_score(peak, local, image_state)}
                    check(checks, sid, memory + '_CURRENT_score', current,
                          old[memory + '_CURRENT'][state['evaluated_images']])
                    for rule, score in values.items():
                        branch = memory + '_' + rule
                        scores[branch].append(score)
                        rows.append(dict(sample_id=sid, relative_path=sample['relative_path'], label=sample['label'], branch=branch,
                                         image_score=score, image_calibration_state_sha256=image_state['state_sha256'] if rule == 'CH1' else '',
                                         localization_branch=memory, localization_rule='frozen_patch_MAD_equal_interpolation',
                                         **{f'peak_L{l}': float(peak[j]) for j, l in enumerate(LAYERS)},
                                         **{f'local_L{l}': float(local[j]) for j, l in enumerate(LAYERS)}))
                state['evaluated_images'] += 1
        scores = {b: np.asarray(v) for b, v in scores.items()}
        # CURRENT identities/scores/metrics AND actual 18,020 pair states are
        # checked; a score tolerance alone cannot justify unchanged ranking.
        ranking, detail = ranking_tables(samples, scores, checks, old)
        metrics, deltas = metric_tables(samples, scores, old, ctx, checks)
        replay.write_csv(output / 'per_image_scores.csv', rows)
        replay.write_csv(output / 'ranking_summary.csv', ranking)
        replay.write_csv(output / 'ranking_per_image.csv', detail)
        replay.write_csv(output / 'image_metrics.csv', metrics)
        replay.write_csv(output / 'metric_comparisons.csv', deltas)
        final_ctx = load_context(p)
        check(checks, 'inputs', 'frozen_inputs_unchanged', [final_ctx['identity'] == ctx['identity']], [True], atol=0)
        validate_recovery(root, ctx, complete=True)
        state.update(status='evaluation_complete' if complete(state['evaluated_images'], checks) else 'partial_evidence',
                     numerical_failed_checks=sum(not r['verified'] for r in checks),
                     source_image_MAD_shared=True, protected_inputs_unchanged=True)
        lines = ['# AeBAD-S/background 第一章冻结图像评分继承对照', '',
                 '本项为已有评分的继承对照，不计作第二章新方法。', '',
                 '原 patch MAD 冻结；新拟合的两组 image MAD 仅来自恢复的66张 Source 校准图，并由 Source/完整K8共用。',
                 '原逐图校准缓存已删除：汇总统计一致不代表与旧逐图距离逐元素复现。',
                 '下表为本次 CPU 重新计算的图像指标，单位%。未重新评价任何像素指标。', '',
                 '|分支|Image AUROC|Image AP|', '|---|---:|---:|']
        lines += [f"|{r['branch']}|{100*r['image_AUROC']:.6f}|{100*r['image_AP']:.6f}|"
                  for r in metrics if r['normal_group'] == 'all85']
        lines += ['', '以下差值均为左侧分支减参照，单位为百分点；正负值均保留。']
        lines += [f"- {r['branch']} − {r['reference']}，{r['metric']}：{r['delta_percentage_points']:+.6f} pp。"
                  for r in deltas if r['normal_group'] == 'all85']
        lines += ['', '前两组差值分别回答 Source、完整Union 的评分继承变化；第三组才是同一CH1评分下的Target支持收益。',
                  '后续参照适配应以冻结 UNION_K8_CH1 为同信息主比较基线，同时保留 SOURCE_CH1 和两项 CURRENT 对照；不依据本次指标选择评分或层。',
                  'CURRENT 与 CH1 在各自 memory 分支内共享同一定义的定位图；图像评分变化不代表定位改善。',
                  '排序表按全部85、原固定困难10、其余75张正常图分别统计，对212张异常图；同分错误权重0.5，不等同于部署误报/漏检。',
                  '历史逐图分数及 domain_metrics.csv 仅用作 CURRENT 复核参照；本表的图像指标均为重算。',
                  'background 已用于多轮开发，本次不属于独立最终验证。', '',
                  f"验证状态：{state['status']}；失败检查：{state['numerical_failed_checks']}。若为partial_evidence，表格仅表示已计算，不能作为已验证结论。"]
        (output / 'objective_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    except Exception as error:
        state.update(status='partial_evidence', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        replay.write_csv(output / 'numerical_verification.csv', checks)
        state['seconds'] = time.perf_counter() - start
        state['torch_imported'] = 'torch' in sys.modules
        state['new_output_bytes'] = sum(f.stat().st_size for f in output.rglob('*') if f.is_file())
        state['process_peak_rss_bytes'] = peak_rss()
        replay.atomic_json(output / 'status.json', state)
        replay.atomic_json(output / 'timing.json', dict(stage='CPU_image_evaluation', seconds=state['seconds'],
                           model_inference=False, pixel_evaluation=False))
    return state


def peak_rss():
    """Observed process peak where supported; CPU only."""
    import os
    if os.name == 'nt':
        import psutil
        return int(psutil.Process().memory_info().peak_wset)
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def image_statistics(ctx, sample):
    raw = ctx['bundle']['raw'][f"raw/{sample['sample_id']}.npz"]
    return {memory: statistics(distances, raw['positions'], raw['grid'], raw['metadata']['output_shape'],
                               ctx['medians'], ctx['scales']) for memory, distances in
            (('SOURCE', raw['source_distance']),
             ('UNION_K8', np.minimum(raw['source_distance'], raw['target_full_distance'])))}
