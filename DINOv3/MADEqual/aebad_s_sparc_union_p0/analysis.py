"""Explicit four-branch comparison; only the new branch runs pixel metrics."""
from __future__ import annotations
import json
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image

from .inputs import TOOL,NEW,BRANCHES,replay,parent_inputs,verify_files,history_tables
from .runtime import validate_arrays,source_checks
from DINOv3.MADEqual.aebad_s_sparc_p0 import analysis as parent_analysis
from DINOv3.MADEqual.aebad_s_sparc_p0.projection import LAYERS
from DINOv3.MADEqual.aebad_s_union_support_loo.analysis import branch_maps
from DINOv3.MADEqual.aebad_s_union_support_loo.runner import audit

REFERENCES=('UNION_K8','SPARC_SOURCE_K8')


def backends_for(data):
    from DINOv3.MADEqual.aebad_s_union_diagnostics.spatial import MapBackend
    old=MapBackend(Path(data['paths']['results'])/'source_mad_state.npz')
    corrected=MapBackend(Path(data['paths']['parent'])/'sparc_source_mad_state.npz')
    return dict(SOURCE=old,UNION_K8=old,SPARC_SOURCE_K8=corrected,SPARC_UNION_K8=corrected)


def maps_for(raws,positions,meta,backends):
    return {b:branch_maps({b:parent_analysis.scatter(raw,positions,tuple(meta['grid']))},
                         tuple(meta['output_shape']),backends[b])[b] for b,raw in raws.items()}


def load_parent(data,sample,cap,old,new):
    parent=dict(data['parent']);parent['paths']=dict(parent['paths'],output=data['paths']['parent'])
    return parent_analysis.load_triplet(parent,sample,cap,old,new)


def load_new(data,sample,record,meta,positions,parent_record):
    root=Path(data['paths']['output']);file=root/record['file']
    if record['file']!=f"raw/{sample['sample_id']}.npz" or replay.sha256(file)!=record['sha256']:
        raise ValueError('new raw missing/modified/unsafe')
    with np.load(file,allow_pickle=False) as f:
        m=json.loads(str(f['metadata'].item()))
        arrays={k:f[k].copy() for k in ('source_distance','target_distance','union_distance','source_index','target_index',
                                      'positions','grid','processed_shape')}
    validate_arrays(arrays,tuple(data['frozen']['grid']))
    if (m.get('tool')!=TOOL or m['identity']!=data['identity'] or m['parent_identity']!=data['frozen']['parent_identity']
            or m['sample_id']!=sample['sample_id'] or m['relative_path']!=sample['relative_path']
            or m['image_sha256']!=meta['image_sha256'] or m['grid']!=meta['grid']
            or m['geometry']!=meta['geometry'] or m['output_shape']!=meta['output_shape']
            or m['parent_raw_sha256']!=parent_record['sha256'] or m['branch']!=NEW or m['distance_divisor']!=1280
            or m['target_prepare_sha256']!=replay.sha256(root/'prepare_target_status.json')
            or not m['historical_source_reproduced'] or m['query_forward_calls']!=1 or m['query_corrections_per_layer']!=1
            or not np.array_equal(positions,arrays['positions'])
            or arrays['processed_shape'].tolist()!=[meta['geometry']['processed_height'],meta['geometry']['processed_width']]):
        raise ValueError('new raw provenance/full geometry does not match frozen parent')
    return arrays


def check_frozen(data,prepared):
    root=Path(data['paths']['output']);state=json.loads((root/'infer_status.json').read_text(encoding='utf-8'))
    samples=data['parent']['samples']
    if (state.get('tool')!=TOOL or state['identity']!=data['identity'] or state['status']!='predictions_frozen'
            or state['parent_identity']!=data['frozen']['parent_identity'] or state['predicted_images']!=297
            or state['inputs_sha256']!=replay.sha256(root/'inputs.json')
            or state['prepare_target_sha256']!=replay.sha256(root/'prepare_target_status.json')
            or state['verification_sha256']!=replay.sha256(root/'infer_verification.csv')
            or not state['historical_source_reproduced']
            or [r['sample_id'] for r in state['records']]!=[s['sample_id'] for s in samples]):
        raise ValueError('all 297 verified frozen predictions required before GT')
    checks=replay.read_csv(root/'infer_verification.csv')
    expected={(s['sample_id'],kind) for s in samples for kind in
              ('historical_SPARC_Source_full_grid','historical_SPARC_Source_image_score','same_query_same_MAD_map_monotonicity')}
    if (len(checks)!=len(expected) or {(r['sample_id'],r['check_kind']) for r in checks}!=expected
            or any(r['verified'].lower()!='true' for r in checks)):
        raise ValueError('complete successful historical Source reproduction checks required')
    verify_files(data['required_files']);parent_inputs.verify_artifacts(root,prepared['artifacts'])
    parent_inputs.verify_artifacts(root,state['records']);return state


def rankings(scores):
    from sklearn.metrics import roc_auc_score,average_precision_score
    by={b:{r['sample_id']:r for r in scores if r['branch']==b} for b in BRANCHES}
    ids=sorted(by[NEW]);labels=np.array([by[NEW][i]['label'] for i in ids]);normal=labels==0;anomaly=labels==1
    if any(set(v)!=set(ids) for v in by.values()):raise ValueError('four identical heldout sets required')
    values={b:np.array([by[b][i]['image_score'] for i in ids],np.float64) for b in BRANCHES}
    def errors(x):return (x[normal,None]>x[None,anomaly]).astype(float)+.5*(x[normal,None]==x[None,anomaly])
    current=values[NEW];e=errors(current);rows=[]
    for ref in REFERENCES:
        baseline=errors(values[ref])
        rows.append(dict(branch=NEW,reference=ref,image_count=len(ids),normal_count=int(normal.sum()),anomaly_count=int(anomaly.sum()),
            image_AUROC=float(roc_auc_score(labels,current)),image_AP=float(average_precision_score(labels,current)),
            weighted_pair_errors=float(e.sum()),reference_weighted_pair_errors=float(baseline.sum()),pair_count=int(e.size),tie_error_weight=.5,
            improved_pairs=int((e<baseline).sum()),worsened_pairs=int((e>baseline).sum()),unchanged_pairs=int((e==baseline).sum()),
            normal_score_p90=float(np.percentile(current[normal],90)),normal_score_p95=float(np.percentile(current[normal],95)),normal_score_max=float(current[normal].max()),
            fixed_UNION_top10_mean_score_change=float(np.mean([by[NEW][sid]['image_score']-by[ref][sid]['image_score'] for sid,_ in replay.CASES[:10]]))))
    return rows


def distance_audit(sample,raws,positions,patch_mask,old_scales,sparc_scales):
    selected=np.ones(len(positions),bool) if not sample['label'] else patch_mask.reshape(-1)[positions]
    if not selected.any():raise ValueError('required GT patch audit region empty')
    rows=[]
    for ref,scale,name in [('UNION_K8',old_scales,'old_Source_MAD_scale'),('SPARC_SOURCE_K8',sparc_scales,'SPARC_Source_MAD_scale')]:
        scales=np.asarray(scale,np.float64)
        if scales.shape!=(4,) or not np.isfinite(scales).all() or (scales<=0).any():raise ValueError('invalid fixed audit scale')
        delta=(raws[ref].astype(np.float64)-raws[NEW].astype(np.float64))/scales[:,None]
        for offset,layer in enumerate(LAYERS):
            rows.append(dict(sample_id=sample['sample_id'],label=sample['label'],branch=NEW,reference=ref,scale=name,layer=layer,
                R_value=float(delta[offset,selected].mean()),selected_patch_count=int(selected.sum()),
                region='normal_all_patches' if not sample['label'] else 'GT_defect_patches'))
    return rows


def summarize_audit(rows):
    result=[]
    for ref,name in [('UNION_K8','old_Source_MAD_scale'),('SPARC_SOURCE_K8','SPARC_Source_MAD_scale')]:
        group=[]
        for layer in LAYERS:
            values=[r for r in rows if r['reference']==ref and r['layer']==layer]
            normal=[r['R_value'] for r in values if not r['label']];anomaly=[r['R_value'] for r in values if r['label']]
            rn,ra=(float(np.mean(v)) if v else None for v in (normal,anomaly))
            row=dict(branch=NEW,reference=ref,scale=name,layer=layer,R_N=rn,R_A=ra,
                G=rn-ra if rn is not None and ra is not None else None,normal_image_count=len(normal),anomaly_image_count=len(anomaly),aggregation='image_equal')
            group.append(row);result.append(row)
        result.append(dict(branch=NEW,reference=ref,scale=name,layer='four_layer_mean',
            **{k:float(np.mean([r[k] for r in group])) if all(r[k] is not None for r in group) else None for k in ('R_N','R_A','G')},
            normal_image_count=min(r['normal_image_count'] for r in group),anomaly_image_count=min(r['anomaly_image_count'] for r in group),aggregation='equal_four_layer_image_equal'))
    return result


def evaluate(data,prepared):
    root=Path(data['paths']['output']);base=Path(data['paths']['parent'])
    if (root/'evaluation').exists():raise ValueError('evaluate refuses existing/partial results')
    infer=check_frozen(data,prepared) # GT remains unopened until this barrier passes.
    new={r['sample_id']:r for r in infer['records']};parent_state=json.loads((base/'infer_status.json').read_text(encoding='utf-8'))
    parent_raw={r['sample_id']:r for r in parent_state['records']}
    cap,cap_state=parent_inputs.read_capture(Path(data['paths']['capture']));old={r['sample_id']:r for r in cap_state['records']}
    history,metrics=history_tables(base,data['parent']['samples']);directory=root/'evaluation';directory.mkdir()
    start=time.perf_counter();timing=defaultdict(float);tables=defaultdict(list)
    state=dict(tool=TOOL,status='running',identity=data['identity'],parent_identity=data['frozen']['parent_identity'],
        analyzed_images=0,numerical_failed_checks=0,incomplete_spatial_ids=[],selected_figures=0,pixel_metrics_complete=False,
        old_branches_pixel_evaluated=False,new_branch=NEW,predictions_frozen_before_GT=True,model_inference=False)
    try:
        backends=backends_for(data);evaluator=parent_analysis.runtime_evaluator();labels=[];masks=[];new_maps=[]
        for i,s in enumerate(data['parent']['samples']):
            t=time.perf_counter();raws,positions,meta=load_parent(data,s,cap,old[s['sample_id']],parent_raw[s['sample_id']])
            arrays=load_new(data,s,new[s['sample_id']],meta,positions,parent_raw[s['sample_id']]);raws[NEW]=arrays['union_distance']
            maps=maps_for(raws,positions,meta,backends)
            repeated_source=maps_for({'SPARC_SOURCE_K8':arrays['source_distance']},positions,meta,backends)['SPARC_SOURCE_K8']
            source_checks(s,arrays,raws['SPARC_SOURCE_K8'],repeated_source,float(history['SPARC_SOURCE_K8'][s['sample_id']]['image_score']),tables['numerical_verification'])
            for branch in BRANCHES[:3]:
                audit(tables['numerical_verification'],s['sample_id'],'historical_score_reconstruction',maps[branch].max(),
                      float(history[branch][s['sample_id']]['image_score']),score=True,branch=branch)
            mono=bool(np.all(maps[NEW]<=repeated_source+1e-5))
            tables['numerical_verification'].append(dict(sample_id=s['sample_id'],check_kind='same_MAD_map_monotonicity',verified=mono,score_atol=1e-5))
            if not mono:raise ValueError('new map exceeds same-query same-MAD Source beyond tolerance')
            for branch in BRANCHES:
                score=float(maps[branch].max()) if branch==NEW else float(history[branch][s['sample_id']]['image_score'])
                y,x=np.unravel_index(np.argmax(maps[branch]),maps[branch].shape)
                tables['per_image_scores'].append(dict(sample_id=s['sample_id'],relative_path=s['relative_path'],label=s['label'],branch=branch,image_score=score,
                    result_origin='new_calculation' if branch==NEW else 'historical_parent_score',parent_identity=data['frozen']['parent_identity'],
                    full_peak_y=int(y),full_peak_x=int(x),score_change_vs_UNION=score-float(history['UNION_K8'][s['sample_id']]['image_score']),
                    score_change_vs_SPARC_SOURCE=score-float(history['SPARC_SOURCE_K8'][s['sample_id']]['image_score'])))
            timing['cache_reading_and_maps']+=time.perf_counter()-t;t=time.perf_counter()
            record=SimpleNamespace(path=str(Path(data['paths']['data'])/s['relative_path']),label=s['label'],
                mask_path=str(Path(data['paths']['data'])/parent_inputs.gt_relative(s['relative_path'])) if s['label'] else None)
            mask=patch=None
            if s['label']:
                with Image.open(record.mask_path) as image:mask=np.asarray(image.convert('L'))>0
                if mask.shape!=(s['image_height'],s['image_width']) or not mask.any():raise ValueError('missing/empty/changed original GT')
                patch=backends['SOURCE'].patch_mask(Image.fromarray(mask.astype(np.uint8)*255),tuple(meta['grid']))
                instance_rows=parent_analysis.spatial_rows(s,maps,mask,tuple(meta['grid']),backends['SOURCE'])
                indexed={(r['instance_id'],r['branch']):r for r in instance_rows}
                for row in instance_rows:
                    ref=indexed[(row['instance_id'],'SPARC_SOURCE_K8')]['margin']
                    row['margin_change_vs_SPARC_SOURCE']=row['margin']-ref if row['margin'] is not None and ref is not None else None
                    row['SPARC_margin_scale_note']='SPARC two branches share frozen MAD; comparison to UNION uses different MAD scales'
                tables['defect_instances'].extend(instance_rows)
            tables['distance_audit_per_image'].extend(distance_audit(s,raws,positions,patch,backends['SOURCE'].mad.scales,backends[NEW].mad.scales))
            labels.append(s['label']);masks.append(evaluator._load_mask(record,tuple(meta['output_shape'])));new_maps.append(maps[NEW])
            timing['GT_and_spatial_audit']+=time.perf_counter()-t
            if s['sample_id'] in {sid for sid,_ in replay.CASES}:
                from .visualization import plot_case
                t=time.perf_counter()
                with Image.open(record.path) as image:plot_case(image.convert('RGB'),mask,maps,directory/'figures'/f"{s['sample_id']}.png")
                timing['figures']+=time.perf_counter()-t;state['selected_figures']+=1
            state['analyzed_images']+=1
            if i%25==0 or i==296:print(f'[SPARC UNION evaluate] {i+1}/297',flush=True)
        tables['ranking_summary']=rankings(tables['per_image_scores']);tables['distance_audit_summary']=summarize_audit(tables['distance_audit_per_image'])
        from sklearn.metrics import roc_auc_score,average_precision_score
        for metric in metrics:
            b=metric['branch'];ids=sorted(history[b]);labs=[int(history[b][sid]['label']) for sid in ids];scores=[float(history[b][sid]['image_score']) for sid in ids]
            for key,actual in [('image_AUROC',roc_auc_score(labs,scores)),('image_AP',average_precision_score(labs,scores))]:
                audit(tables['numerical_verification'],'background','historical_summary_reproduction',actual,float(metric[key]),score=True,branch=b,metric=key)
            tables['metrics'].append(dict(metric,result_origin='reused_historical_result',parent_identity=data['frozen']['parent_identity'],
                source_file=str(base/'evaluation/metrics.csv'),source_sha256=replay.sha256(base/'evaluation/metrics.csv')))
        t=time.perf_counter();values=evaluator._evaluate(labels,masks,new_maps);timing['new_branch_pixel_evaluation']+=time.perf_counter()-t
        metric=dict(branch=NEW,image_count=297,image_AUROC=values['image_AUROC'],image_AP=values['image_AUPR'],pixel_AUROC=values['pixel_AUROC'],pixel_AP=values['pixel_AUPR'],AUPRO=values['AUPRO'],
            result_origin='new_calculation',parent_identity=data['frozen']['parent_identity'],protocol='original full-protocol; ADEval CUDA 200 thresholds FPR<=0.3')
        if any(not np.isfinite(metric[k]) or not 0<=metric[k]<=1 for k in ('image_AUROC','image_AP','pixel_AUROC','pixel_AP','AUPRO')):raise ValueError('invalid new metric')
        for key in ('image_AUROC','image_AP'):
            audit(tables['numerical_verification'],'background','new_metric_matches_scores',metric[key],tables['ranking_summary'][0][key],score=True,metric=key)
        tables['metrics'].append(metric);check_frozen(data,prepared)
        state['numerical_failed_checks']=sum(not r['verified'] for r in tables['numerical_verification'])
        complete=(state['analyzed_images']==297 and len(tables['per_image_scores'])==1188 and state['selected_figures']==12
            and state['numerical_failed_checks']==0 and all(r['normal_image_count']==85 and r['anomaly_image_count']==212 for r in tables['distance_audit_summary']))
        state.update(status='analysis_complete' if complete else 'partial_evidence',pixel_metrics_complete=True,protected_inputs_unchanged=True)
    except Exception as e:
        state.update(status='partial_evidence',error=f'{type(e).__name__}: {e}')
        state['numerical_failed_checks']=sum(not r['verified'] for r in tables['numerical_verification'])
    for name,rows in tables.items():replay.write_csv(directory/f'{name}.csv',rows)
    state.update(seconds=time.perf_counter()-start,timing=dict(timing));replay.atomic_json(directory/'status.json',state)
    lines=['# SPARC_UNION_K8 supplement','',f"Status: {state['status']}",'',
        '| branch | origin | Image AUROC | Image AP | Pixel AUROC | Pixel AP | AUPRO@0.3 |','|---|---|---:|---:|---:|---:|---:|']
    for r in tables['metrics']:lines.append('| '+r['branch']+' | '+r['result_origin']+' | '+' | '.join(f'{float(r[k]):.9f}' for k in ('image_AUROC','image_AP','pixel_AUROC','pixel_AP','AUPRO'))+' |')
    lines+=['','Within each row, adding Target is the controlled change. Cross-row comparisons also change Source correction/reconstruction and MAD.',
        'Duplicate corrected supports do not imply a unique semantic Top-1 support. Distance/map monotonicity does not guarantee metric or margin improvement.',
        'Background is a development domain; no parameter/threshold selection. This is a combination control, not a paper-table reproduction or a claimed innovation.',f"Error: {state.get('error','')}"]
    (directory/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return state
