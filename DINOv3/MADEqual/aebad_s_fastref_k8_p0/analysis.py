"""Four branches, signed audits, original spatial definitions and two new metrics."""
from __future__ import annotations
import json
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
from . import inputs, runtime
from DINOv3.MADEqual.aebad_s_union_support_loo.analysis import load_raw
from DINOv3.MADEqual.aebad_s_union_support_loo.runner import audit, formal_baselines
from DINOv3.MADEqual.aebad_s_union_diagnostics.spatial import components, instance_regions, MapBackend
from DINOv3.MADEqual.aebad_s_sparc_p0.analysis import spatial_rows, runtime_evaluator

COMPARISONS = (('UNION_TARGET512','UNION_K8'),('FASTREF_TARGET512_SOURCE','UNION_TARGET512'),('FASTREF_TARGET512_SOURCE','UNION_K8'))
COMPARISON_MEANINGS = ('fixed_bounded_inverse_density_Target_memory_compression',
    'refinement_of_the_same_frozen_Target512','whole_adaptation_vs_full_K8_union')


def metric_comparisons(metrics):
    """Only called after actual evaluation: signed branch minus reference."""
    groups = {r['branch']:r for r in metrics}
    if len(metrics) != 4 or set(groups) != set(inputs.BRANCHES):
        raise ValueError('four complete branch metrics required for comparisons')
    rows = []
    for code,(branch,ref),meaning in zip(('A','B','C'),COMPARISONS,COMPARISON_MEANINGS,strict=True):
        row = dict(comparison=code,branch=branch,reference=ref,interpretation=meaning,
            delta_direction='branch_minus_reference',delta_unit='metric_fraction',image_count=297)
        for key in inputs.METRIC_KEYS:
            value,baseline = float(groups[branch][key]),float(groups[ref][key])
            if not np.isfinite([value,baseline]).all(): raise ValueError('nonfinite metric comparison')
            row['delta_'+key] = value-baseline
        rows.append(row)
    return rows


def load_prediction(data, sample, record, old_record, meta, old_arrays):
    root = Path(data['paths']['output']); file = root/record['file']
    if record['file'] != f"raw/{sample['sample_id']}.npz" or inputs.replay.sha256(file) != record['sha256']: raise ValueError('new raw modified/unsafe')
    with np.load(file,allow_pickle=False) as f:
        m = json.loads(str(f['metadata'].item())); arrays = {k:f[k].copy() for k in f.files if k != 'metadata'}
    if (m['tool'] != inputs.TOOL or m['identity'] != data['identity'] or m['sample_id'] != sample['sample_id']
        or m['relative_path'] != sample['relative_path'] or m['image_sha256'] != meta['image_sha256']
        or m['historical_raw_sha256'] != old_record['sha256'] or m['geometry'] != meta['geometry']
        or m['grid'] != meta['grid'] or m['output_shape'] != meta['output_shape'] or m['distance_divisor'] != 1280
        or m['prepare_sha256'] != inputs.replay.sha256(root/'prepare_status.json') or m['query_forward_calls'] != 1
        or m['refined_rows'] != len(old_arrays['positions']) or m['query_features_persisted'] or m['refined_vectors_persisted']
        or arrays['grid'].tolist() != meta['grid'] or not np.array_equal(arrays['positions'],old_arrays['positions'])
        or arrays['processed_shape'].tolist() != [meta['geometry']['processed_height'],meta['geometry']['processed_width']]):
        raise ValueError('prediction frozen identity/geometry changed')
    n = int(np.prod(meta['grid']))
    for key in ('source_distance','target_full_distance','prototype_distance','refined_distance','prototype_union_distance','fastref_union_distance'):
        a = arrays[key]
        if a.shape != (4,n) or a.dtype != np.float32 or not np.isfinite(a).all() or np.any(a < 0): raise ValueError('invalid prediction distances')
    for key,bound in [('prototype_index',512),('refined_index',n)]:
        a = arrays[key]
        if a.shape != (4,n) or a.dtype != np.int64 or np.any(a < 0) or np.any(a >= bound): raise ValueError('invalid nearest-neighbor index')
    if not np.array_equal(arrays['prototype_union_distance'],np.minimum(arrays['source_distance'],arrays['prototype_distance'])) or not np.array_equal(arrays['fastref_union_distance'],np.minimum(arrays['source_distance'],arrays['refined_distance'])):
        raise ValueError('new branches must retain Source only, without adding full T')
    return arrays


def ranking_tables(scores):
    from sklearn.metrics import average_precision_score, roc_auc_score
    groups = {b:{r['sample_id']:r for r in scores if r['branch']==b} for b in inputs.BRANCHES}
    ids = sorted(groups['UNION_K8']); labels = np.array([groups['UNION_K8'][i]['label'] for i in ids])
    if len(ids)!=297 or any(set(g)!=set(ids) for g in groups.values()): raise ValueError('four identical full heldout sets required')
    normal = labels==0; anomaly = labels==1
    if (int(normal.sum()),int(anomaly.sum())) != (85,212): raise ValueError('original 85/212 scope required')
    values = {b:np.array([groups[b][i]['image_score'] for i in ids],np.float64) for b in inputs.BRANCHES}
    def errors(x): return (x[normal,None]>x[None,anomaly]).astype(float)+.5*(x[normal,None]==x[None,anomaly])
    rows = []; detail = []
    fixed = {sid for sid,_ in inputs.replay.CASES[:10]}
    for branch,ref in COMPARISONS:
        before = errors(values[ref]); after = errors(values[branch]); x = values[branch]
        rows.append(dict(branch=branch,reference=ref,image_count=297,normal_count=85,anomaly_count=212,
            image_AUROC=float(roc_auc_score(labels,x)),image_AP=float(average_precision_score(labels,x)),
            weighted_pair_errors=float(after.sum()),reference_weighted_pair_errors=float(before.sum()),pair_count=int(after.size),
            improved_pairs=int((after<before).sum()),worsened_pairs=int((after>before).sum()),unchanged_pairs=int((after==before).sum()),
            tie_error_weight=.5,normal_score_p90=float(np.percentile(x[normal],90)),normal_score_p95=float(np.percentile(x[normal],95)),normal_score_max=float(x[normal].max()),
            fixed_UNION_top10_mean_score_change=float(np.mean([groups[branch][sid]['image_score']-groups[ref][sid]['image_score'] for sid in fixed]))))
        for index,sid in enumerate(ids):
            difference = after-before
            region = difference[int(np.count_nonzero(normal[:index]))] if normal[index] else difference[:,int(np.count_nonzero(anomaly[:index]))]
            detail.append(dict(branch=branch,reference=ref,sample_id=sid,label=int(labels[index]),
                score_change=float(values[branch][index]-values[ref][index]),weighted_pair_error_change=float(region.sum()),
                improved_pairs=int((region<0).sum()),worsened_pairs=int((region>0).sum()),fixed_UNION_top10=sid in fixed))
    return rows,detail


def distance_rows(sample, raws, positions, patch_mask, scales):
    chosen = np.ones(len(positions),bool) if not sample['label'] else patch_mask.reshape(-1)[positions]
    if not chosen.any(): raise ValueError('anomaly GT patch region empty')
    result = []
    for branch,ref in COMPARISONS:
        delta = (raws[ref].astype(np.float64)-raws[branch].astype(np.float64))/np.asarray(scales)[:,None]
        for offset,layer in enumerate(inputs.LAYERS):
            result.append(dict(branch=branch,reference=ref,scale='frozen_original_Source_MAD_scale',sample_id=sample['sample_id'],label=sample['label'],
                layer=layer,R_value=float(delta[offset,chosen].mean()),selected_patch_count=int(chosen.sum()),region='normal_all_patches' if not sample['label'] else 'original_GT_defect_patches'))
    return result


def distance_summary(rows):
    result = []
    for branch,ref in COMPARISONS:
        layers = []
        for layer in inputs.LAYERS:
            data = [r for r in rows if (r['branch'],r['reference'],r['layer'])==(branch,ref,layer)]
            normal = [r['R_value'] for r in data if not r['label']]; anomaly = [r['R_value'] for r in data if r['label']]
            if (len(normal),len(anomaly)) != (85,212): raise ValueError('full-domain signed distance audit incomplete')
            rn,ra = float(np.mean(normal)),float(np.mean(anomaly))
            row = dict(branch=branch,reference=ref,scale='frozen_original_Source_MAD_scale',layer=layer,R_N=rn,R_A=ra,G=rn-ra,normal_image_count=85,anomaly_image_count=212,aggregation='image_equal')
            layers.append(row); result.append(row)
        result.append(dict(branch=branch,reference=ref,scale='frozen_original_Source_MAD_scale',layer='four_layer_mean',
            **{k:float(np.mean([r[k] for r in layers])) for k in ('R_N','R_A','G')},normal_image_count=85,anomaly_image_count=212,aggregation='equal_four_layer_image_equal'))
    return result


def instance_rows(sample, maps, mask, grid, backend):
    rows = spatial_rows(sample,maps,mask,grid,backend)
    if not sample['label']: return [],[]
    labels,descriptors = components(mask); peaks = []
    for desc in descriptors:
        inside,ring = instance_regions(labels==desc['instance_id'],mask,maps['UNION_K8'].shape)
        for ref in ('UNION_K8','UNION_TARGET512'):
            for role,region in [('instance',inside),('ring',ring)]:
                if not region.any(): continue
                index = int(np.argmax(np.where(region,maps[ref],-np.inf))); y,x = divmod(index,region.shape[1])
                for branch,value in maps.items():
                    peaks.append(dict(sample_id=sample['sample_id'],instance_id=desc['instance_id'],reference=ref,branch=branch,region=role,
                        reference_peak_y=y,reference_peak_x=x,score_at_fixed_reference_peak=float(value[y,x]),
                        response_change_at_fixed_reference_peak=float(value[y,x]-maps[ref][y,x]),peak_tie_rule='first_row_major'))
    return rows,peaks


def evaluate(data, prepared, device_name):
    root = Path(data['paths']['output']); frozen = runtime.verify_stage(data,prepared)
    directory = root/'evaluation'
    if directory.exists(): raise ValueError('evaluation output exists; no overwrite/re-approval')
    directory.mkdir(); started = time.perf_counter(); timing = defaultdict(float); tables = defaultdict(list)
    state = dict(tool=inputs.TOOL,stage='evaluate',status='partial_evidence',identity=data['identity'],analyzed_images=0,selected_figures=0,
        numerical_failed_checks=0,missing_gt_ids=[],pixel_metrics_complete=False,old_branches_pixel_evaluated=False,
        predictions_frozen_before_GT=True,model_inference=False,source_reference_unchanged=True)
    try:
        from DINOv3.MADEqual.aebad_s_sparc_union_p0.runtime import evaluation_device
        evaluation_device(dict(parent=dict(frozen=dict(precision=data['frozen']['precision']))),device_name)
        evaluator = runtime_evaluator(); backend = MapBackend(Path(data['paths']['mad']))
        formal = formal_baselines(Path(data['paths']['results']),data['capture']['samples'])
        historical_metrics = inputs.original_metrics(Path(data['paths']['results']))
        new = {r['sample_id']:r for r in frozen['records']}; old = {r['sample_id']:r for r in data['capture_status']['records']}
        labels = []; masks = []; collected = {b:[] for b in inputs.NEW}
        for sample in data['capture']['samples']:
            t = time.perf_counter(); sid = sample['sample_id']
            historical,meta = load_raw(Path(data['paths']['capture'])/old[sid]['file'],identity=data['capture']['identity'],sample=sample,checksum=old[sid]['sha256'])
            arrays = load_prediction(data,sample,new[sid],old[sid],meta,historical)
            raws = dict(SOURCE=historical['source_distance'],UNION_K8=np.minimum(historical['source_distance'],historical['target_distance'].min(axis=0)),
                UNION_TARGET512=arrays['prototype_union_distance'],FASTREF_TARGET512_SOURCE=arrays['fastref_union_distance'])
            maps = runtime.maps_for(raws,arrays['positions'],meta,backend)
            okay = audit(tables['numerical_verification'],sid,'offline_Source_distance',arrays['source_distance'],historical['source_distance'])
            okay = audit(tables['numerical_verification'],sid,'offline_Target8_min_distance',arrays['target_full_distance'],historical['target_distance'].min(axis=0)) and okay
            current_source = runtime.maps_for({'SOURCE':arrays['source_distance']},arrays['positions'],meta,backend)['SOURCE']
            okay = audit(tables['numerical_verification'],sid,'offline_current_Source_score',current_source.max(),formal['Source'][sid]['image_score'],score=True) and okay
            for branch,key in [('SOURCE','Source'),('UNION_K8','K8')]:
                okay = audit(tables['numerical_verification'],sid,'offline_historical_score',maps[branch].max(),formal[key][sid]['image_score'],score=True,branch=branch) and okay
            if not okay: raise ValueError('offline formal baseline reproduction failed; cannot complete')
            for branch,value in maps.items():
                y,x = np.unravel_index(np.argmax(value),value.shape)
                score = float(value.max()) if branch in inputs.NEW else formal['Source' if branch=='SOURCE' else 'K8'][sid]['image_score']
                tables['per_image_scores'].append(dict(branch=branch,sample_id=sid,relative_path=sample['relative_path'],label=sample['label'],image_score=score,
                    full_peak_y=int(y),full_peak_x=int(x),result_origin='new_calculation' if branch in inputs.NEW else 'reused_historical_result'))
            timing['cached_maps_and_baseline_checks'] += time.perf_counter()-t; t = time.perf_counter()
            gt = Path(data['paths']['data'])/inputs.shared.gt_relative(sample['relative_path']) if sample['label'] else None
            if gt is not None and not gt.is_file(): state['missing_gt_ids'].append(sid); raise ValueError('required anomaly GT missing: '+sid)
            mask = None
            if gt is not None:
                with Image.open(gt) as image: mask = np.asarray(image.convert('L'))>0
            instances,peaks = instance_rows(sample,maps,mask,tuple(meta['grid']),backend)
            for row in instances: row['calibration_scale_note']='all four branches share original frozen Source MAD; no area threshold/filter'
            tables['defect_instances'].extend(instances); tables['fixed_reference_instance_peaks'].extend(peaks)
            patch = backend.patch_mask(Image.fromarray(mask.astype(np.uint8)*255),tuple(meta['grid'])) if mask is not None else np.zeros(meta['grid'],bool)
            tables['distance_audit_per_image'].extend(distance_rows(sample,raws,arrays['positions'],patch,backend.mad.scales))
            record = SimpleNamespace(path=Path(data['paths']['data'])/sample['relative_path'],mask_path=gt,label=sample['label'])
            masks.append(evaluator._load_mask(record,tuple(meta['output_shape']))); labels.append(sample['label'])
            for branch in inputs.NEW: collected[branch].append(maps[branch])
            timing['GT_and_original_spatial_diagnostics'] += time.perf_counter()-t
            if sid in {i for i,_ in inputs.replay.CASES}:
                from .visualization import plot_case
                t = time.perf_counter()
                with Image.open(record.path) as image: plot_case(image.convert('RGB'),mask,maps,directory/'figures'/f'{sid}.png')
                timing['fixed_figures'] += time.perf_counter()-t; state['selected_figures'] += 1
            state['analyzed_images'] += 1
        tables['ranking_summary'],tables['ranking_per_image'] = ranking_tables(tables['per_image_scores'])
        tables['distance_audit_summary'] = distance_summary(tables['distance_audit_per_image'])
        unique = {(r['sample_id'],r['instance_id']):r for r in tables['defect_instances']}
        smallest = sorted(unique.values(),key=lambda r:(r['area_original_pixels'],r['relative_path'],r['instance_id']))[:3]
        small_ids = {(r['sample_id'],r['instance_id']) for r in smallest}
        for row in tables['defect_instances']: row['original_smallest_three_rule'] = (row['sample_id'],row['instance_id']) in small_ids
        tables['small_instance_summary'] = [dict(sample_id=r['sample_id'],instance_id=r['instance_id'],area_original_pixels=r['area_original_pixels'],patch_coverage=r['patch_coverage'],definition='original diagnostics: smallest three discrete GT components; no tuned threshold') for r in smallest]
        from sklearn.metrics import average_precision_score, roc_auc_score
        for metric in historical_metrics:
            branch = metric['branch']; rows = [r for r in tables['per_image_scores'] if r['branch']==branch]
            for key,actual in [('image_AUROC',roc_auc_score([r['label'] for r in rows],[r['image_score'] for r in rows])),('image_AP',average_precision_score([r['label'] for r in rows],[r['image_score'] for r in rows]))]:
                if not audit(tables['numerical_verification'],'background','historical_summary',actual,metric[key],score=True,branch=branch,metric=key): raise ValueError('formal baseline aggregate changed')
            tables['metrics'].append(dict(metric,result_origin='reused_historical_result',source_file=str(Path(data['paths']['results'])/'domain_metrics.csv'),source_sha256=inputs.replay.sha256(Path(data['paths']['results'])/'domain_metrics.csv')))
        t = time.perf_counter()
        for branch in inputs.NEW:
            values = evaluator._evaluate(labels,masks,collected[branch])
            metric = dict(branch=branch,image_count=297,image_AUROC=values['image_AUROC'],image_AP=values['image_AUPR'],pixel_AUROC=values['pixel_AUROC'],pixel_AP=values['pixel_AUPR'],AUPRO=values['AUPRO'],result_origin='new_calculation',protocol='original ADEval CUDA 200 thresholds FPR<=0.3')
            if any(not np.isfinite(metric[k]) or not 0<=metric[k]<=1 for k in inputs.METRIC_KEYS): raise ValueError('new metric invalid')
            summary = next(r for r in tables['ranking_summary'] if r['branch']==branch)
            for key in ('image_AUROC','image_AP'):
                if not audit(tables['numerical_verification'],'background','new_metric_matches_image_scores',metric[key],summary[key],score=True,branch=branch,metric=key):
                    raise ValueError('new backend image metric differs from frozen per-image scores')
            tables['metrics'].append(metric)
        timing['two_new_branch_pixel_evaluations'] += time.perf_counter()-t
        tables['metric_comparisons'] = metric_comparisons(tables['metrics'])
        runtime.verify_stage(data,prepared)
        state['numerical_failed_checks'] = sum(not r['verified'] for r in tables['numerical_verification'])
        complete = state['analyzed_images']==297 and state['selected_figures']==12 and len(tables['per_image_scores'])==1188 and len(tables['metrics'])==4 and len(tables['metric_comparisons'])==3 and not state['numerical_failed_checks'] and not state['missing_gt_ids']
        state.update(status='analysis_complete' if complete else 'partial_evidence',pixel_metrics_complete=True,protected_inputs_unchanged=True,GT_used_only_after_predictions_frozen=True)
    except (ImportError,ModuleNotFoundError) as error: state.update(status='evaluation_pending',error=str(error))
    except Exception as error:
        state.update(status='partial_evidence',error=f'{type(error).__name__}: {error}')
        state['numerical_failed_checks'] = sum(not r['verified'] for r in tables['numerical_verification'])
    for name,rows in tables.items(): inputs.replay.write_csv(directory/f'{name}.csv',rows)
    state.update(seconds=time.perf_counter()-started,timing=dict(timing)); inputs.replay.atomic_json(directory/'status.json',state)
    lines = ['# FastRef Target512 / original Source anchor','',f"Status: {state['status']}",'',
        '| branch | origin | Image AUROC | Image AP | Pixel AUROC | Pixel AP | AUPRO@0.3 |','|---|---|---:|---:|---:|---:|---:|']
    for row in tables['metrics']: lines.append('| '+row['branch']+' | '+row['result_origin']+' | '+' | '.join(f'{row[k]:.9f}' for k in inputs.METRIC_KEYS)+' |')
    if tables['metric_comparisons']:
        lines += ['','| comparison | branch minus reference | delta Image AUROC | delta Image AP | delta Pixel AUROC | delta Pixel AP | delta AUPRO |',
            '|---|---|---:|---:|---:|---:|---:|']
        for row in tables['metric_comparisons']:
            lines.append('| '+row['comparison']+' | '+row['branch']+' - '+row['reference']+' | '+
                ' | '.join(f"{row['delta_'+k]:+.9f}" for k in inputs.METRIC_KEYS)+' |')
    lines += ['','Controlled Euclidean formula adaptation, not official FastRef reproduction. Background is a development domain.',
        'A measures Target compression by the frozen bounded inverse-density adaptation. This is not exactly the original Source selection algorithm. Full K8 union did not use this compression.',
        'B measures refinement of exactly the same frozen prototypes; C measures the whole adaptation against full K8 union. Metric deltas are branch minus reference, in fractions.',
        'A decrease is not evidence of FastRef failure. A negative B applies only to these prototypes and this configuration; shared prototypes do not establish sampler-independent conclusions.',
        'Neither new branch additionally includes full T. Signed distance audits remain reference minus branch and are not clipped.',
        'GT was unavailable to prototype selection/refinement. No test-based parameter or threshold selection.',f"Error: {state.get('error','')}"]
    (directory/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); return state
