"""One new evaluation; explicit historical origins and signed mechanism evidence."""
from __future__ import annotations
import time
from pathlib import Path
from collections import defaultdict
from types import SimpleNamespace
import numpy as np
from PIL import Image
from . import inputs, runtime
from .projection import difference
from DINOv3.MADEqual.aebad_s_fastref_k8_p0.analysis import instance_rows
from DINOv3.MADEqual.aebad_s_union_diagnostics.spatial import MapBackend
from DINOv3.MADEqual.aebad_s_sparc_p0.analysis import runtime_evaluator
from DINOv3.MADEqual.aebad_s_union_support_loo.runner import audit


def metric_comparisons(metrics):
    by={r['branch']:r for r in metrics}
    if len(metrics)!=5 or set(by)!=set(inputs.BRANCHES): raise ValueError('five complete metric rows required')
    meanings=('removing_OT_after_same_initial_projection','initial_projection_vs_same_fixed_prototypes','projection_vs_full_K8_union')
    result=[]
    for ref,meaning in zip(inputs.REFERENCES,meanings,strict=True):
        row=dict(branch=inputs.BRANCH,reference=ref,interpretation=meaning,
                 delta_direction='branch_minus_reference',delta_unit='metric_fraction',image_count=297)
        for key in inputs.parent.METRIC_KEYS:
            values=[float(by[b][key]) for b in (inputs.BRANCH,ref)]
            if not np.isfinite(values).all(): raise ValueError('nonfinite metric delta')
            row['delta_'+key]=values[0]-values[1]
        result.append(row)
    return result


def ranking_tables(scores):
    from sklearn.metrics import average_precision_score,roc_auc_score
    by={b:{r['sample_id']:r for r in scores if r['branch']==b} for b in inputs.BRANCHES}
    ids=sorted(by[inputs.BRANCH]); labels=np.array([by[inputs.BRANCH][i]['label'] for i in ids])
    if (len(scores)!=1485 or len(ids)!=297 or (int((labels==0).sum()),int((labels==1).sum()))!=(85,212)
            or any(set(rows)!=set(ids) or any(int(rows[i]['label'])!=labels[j] for j,i in enumerate(ids)) for rows in by.values())):
        raise ValueError('five identical complete 85/212 heldout sets required')
    values={b:np.array([by[b][i]['image_score'] for i in ids]) for b in inputs.BRANCHES}
    if not all(np.isfinite(v).all() for v in values.values()): raise ValueError('nonfinite image score')
    normal=labels==0; anomaly=labels==1
    def errors(v): return (v[normal,None]>v[None,anomaly]).astype(float)+.5*(v[normal,None]==v[None,anomaly])
    after=errors(values[inputs.BRANCH]); summary=[]; detail=[]
    fixed={i for i,_ in inputs.replay.CASES[:10]}; normals=[i for i in ids if by[inputs.BRANCH][i]['label']==0]
    if not fixed.issubset(normals): raise ValueError('fixed original ten difficult normals missing')
    for ref in inputs.REFERENCES:
        before=errors(values[ref]); change=after-before
        for group,chosen in [('all_normal',np.ones(85,bool)),('fixed_difficult10',np.array([i in fixed for i in normals])),
                             ('other75',np.array([i not in fixed for i in normals]))]:
            summary.append(dict(branch=inputs.BRANCH,reference=ref,normal_group=group,normal_count=int(chosen.sum()),
                anomaly_count=212,pair_count=int(change[chosen].size),weighted_pair_errors=float(after[chosen].sum()),
                reference_weighted_pair_errors=float(before[chosen].sum()),improved_pairs=int((change[chosen]<0).sum()),
                worsened_pairs=int((change[chosen]>0).sum()),unchanged_pairs=int((change[chosen]==0).sum()),tie_error_weight=.5,
                interpretation='normal_anomaly_ordering_pairs_not_deployment_threshold_errors',
                image_AUROC=float(roc_auc_score(labels,values[inputs.BRANCH])),
                image_AP=float(average_precision_score(labels,values[inputs.BRANCH]))))
        for j,i in enumerate(ids):
            region=change[int(normal[:j].sum())] if normal[j] else change[:,int(anomaly[:j].sum())]
            detail.append(dict(branch=inputs.BRANCH,reference=ref,sample_id=i,label=int(labels[j]),
                normal_group=('fixed_difficult10' if i in fixed else 'other75') if normal[j] else '',
                score_change=float(values[inputs.BRANCH][j]-values[ref][j]),weighted_pair_error_change=float(region.sum()),
                improved_pairs=int((region<0).sum()),worsened_pairs=int((region>0).sum())))
    return summary,detail


def distance_comparison(sample,new,old):
    """Keep pre-min and post-min fields separate, never infer one from the other."""
    rows=[]; deltas={}
    for kind,a,b in [('Target_only',new['projection_distance'],old['refined_distance']),
                     ('Source_union',new['projection_union_distance'],old['fastref_union_distance'])]:
        delta=a.astype(np.float64)-b.astype(np.float64); deltas[kind]=delta
        for index,layer in enumerate(inputs.LAYERS):
            rows.append(dict(sample_id=sample['sample_id'],label=sample['label'],layer=layer,kind=kind,
                branch=inputs.BRANCH,reference='FASTREF_TARGET512_SOURCE',unit='Euclidean_div_1280',
                direction='projection_minus_FastRef',**difference(delta[index],0)))
        rows.append(dict(sample_id=sample['sample_id'],label=sample['label'],layer='all_four_layers',kind=kind,
            branch=inputs.BRANCH,reference='FASTREF_TARGET512_SOURCE',unit='Euclidean_div_1280',
            direction='projection_minus_FastRef',**difference(delta,0)))
    return rows,deltas


def signed_audit(sample,raws,positions,patch_mask,scales):
    scales=np.asarray(scales,np.float64)
    if scales.shape!=(4,) or not np.isfinite(scales).all() or np.any(scales<=0): raise ValueError('original Source MAD scales invalid')
    chosen=np.ones(len(positions),bool) if not sample['label'] else patch_mask.reshape(-1)[positions]
    if not chosen.any(): raise ValueError('GT patch evidence empty; cannot complete R_A')
    rows=[]
    for ref in inputs.REFERENCES:
        delta=(raws[ref].astype(np.float64)-raws[inputs.BRANCH].astype(np.float64))/scales[:,None]
        for j,layer in enumerate(inputs.LAYERS):
            rows.append(dict(sample_id=sample['sample_id'],label=sample['label'],layer=layer,branch=inputs.BRANCH,
                reference=ref,R_value=float(delta[j,chosen].mean()),selected_patch_count=int(chosen.sum()),
                direction='reference_minus_projection',scale='frozen_Source_MAD_no_refit',
                region='normal_all_patches' if not sample['label'] else 'original_GT_defect_patches'))
    return rows


def signed_summary(rows):
    result=[]
    for ref in inputs.REFERENCES:
        layers=[]
        for layer in inputs.LAYERS:
            values=[r for r in rows if r['reference']==ref and r['layer']==layer]
            normal=[r['R_value'] for r in values if not r['label']]; anomaly=[r['R_value'] for r in values if r['label']]
            if (len(normal),len(anomaly))!=(85,212): raise ValueError('R_N/R_A domain evidence incomplete')
            rn,ra=float(np.mean(normal)),float(np.mean(anomaly))
            row=dict(branch=inputs.BRANCH,reference=ref,layer=layer,R_N=rn,R_A=ra,G=rn-ra,
                     direction='reference_minus_projection',scale='frozen_Source_MAD',aggregation='image_equal',normal_images=85,anomaly_images=212)
            layers.append(row);result.append(row)
        result.append(dict(branch=inputs.BRANCH,reference=ref,layer='four_layer_mean',
            **{k:float(np.mean([r[k] for r in layers])) for k in ('R_N','R_A','G')},
            direction='reference_minus_projection',scale='frozen_Source_MAD',aggregation='equal_four_layers_image_equal',normal_images=85,anomaly_images=212))
    return result


def instance_comparisons(rows):
    grouped=defaultdict(dict)
    for row in rows: grouped[row['sample_id'],row['instance_id']][row['branch']]=row
    result=[]
    for (sid,i),by in grouped.items():
        if set(by)!=set(inputs.BRANCHES): raise ValueError('five-branch spatial evidence incomplete')
        for ref in inputs.REFERENCES:
            row=dict(sample_id=sid,instance_id=i,branch=inputs.BRANCH,reference=ref,
                     direction='projection_minus_reference',unit='Source_MAD_response',
                     response_status=by[inputs.BRANCH]['response_status'],reference_status=by[ref]['response_status'])
            for key in ('instance_max','ring_max','margin'):
                a,b=by[inputs.BRANCH][key],by[ref][key]
                row['delta_'+key]=a-b if a is not None and b is not None else None
            result.append(row)
    return result


def complete(state,tables):
    return (state['analyzed_images']==297 and state['selected_figures']==12
            and state['pixel_metrics_complete'] and state['historical_baselines_reproduced']
            and not state['missing_gt_ids'] and not state['numerical_failed_checks']
            and len(tables['per_image_scores'])==1485 and len(tables['metrics'])==5
            and len(tables['metric_comparisons'])==3 and len(tables['signed_distance_summary'])==15
            and len(tables['distance_differences'])==297*10 and len(tables['ranking_summary'])==9
            and bool(tables['defect_instances']) and bool(tables['instance_comparisons']))


def evaluate(data,p,prepared,parent_state,device_name):
    root=Path(data['paths']['output']); frozen=runtime.verify_stage(data,p,prepared,parent_state)
    directory=root/'evaluation'
    if directory.exists(): raise ValueError('evaluation output exists; no overwrite')
    directory.mkdir(); start=time.perf_counter(); timing=defaultdict(float); tables=defaultdict(list)
    state=dict(tool=inputs.TOOL,stage='evaluate',status='partial_evidence',identity=data['identity'],
               analyzed_images=0,selected_figures=0,missing_gt_ids=[],numerical_failed_checks=0,
               pixel_metrics_complete=False,historical_baselines_reproduced=False,
               model_inference=False,OT_called=False,predictions_frozen_before_GT=True,historical_pixel_evaluations=0)
    try:
        from DINOv3.MADEqual.aebad_s_sparc_union_p0.runtime import evaluation_device
        evaluation_device(dict(parent=dict(frozen=dict(precision=p['frozen']['precision']))),device_name)
        backend=MapBackend(Path(p['paths']['mad'])); evaluator=runtime_evaluator()
        historical_scores=inputs.historical_scores(data['paths']['parent'])
        historical_instances={(r['sample_id'],int(r['instance_id']),r['branch']):r for r in
                              inputs.replay.read_csv(Path(data['paths']['parent'])/'evaluation/defect_instances.csv')}
        new={r['sample_id']:r for r in frozen['records']}; old={r['sample_id']:r for r in parent_state['records']}
        pooled=defaultdict(list); labels=[]; masks=[]; collected=[]
        for sample in p['capture']['samples']:
            sid=sample['sample_id']; t=time.perf_counter()
            cached,meta=runtime.parent_raw(p,sample,old[sid],prepared)
            arrays=runtime.load_raw_new(data,sample,new[sid],cached,meta)
            raws=dict(SOURCE=cached['source_distance'],UNION_K8=np.minimum(cached['source_distance'],cached['target_full_distance']),
                      UNION_TARGET512=cached['prototype_union_distance'],FASTREF_TARGET512_SOURCE=cached['fastref_union_distance'])
            raws[inputs.BRANCH]=arrays['projection_union_distance']
            maps=runtime.original.maps_for(raws,arrays['positions'],meta,backend)
            for key in ('source_distance','prototype_distance'):
                if not audit(tables['numerical_verification'],sid,'offline_'+key,arrays[key],cached[key]):
                    raise ValueError('offline query baseline distance changed')
            current=runtime.original.maps_for(dict(SOURCE=arrays['source_distance'],UNION_TARGET512=np.minimum(arrays['source_distance'],arrays['prototype_distance'])),arrays['positions'],meta,backend)
            for b,value in current.items():
                if not audit(tables['numerical_verification'],sid,'offline_current_score',value.max(),historical_scores[b,sid],score=True,branch=b):
                    raise ValueError('offline current Source/P score changed')
            for b,value in maps.items():
                if b!=inputs.BRANCH and not audit(tables['numerical_verification'],sid,'historical_score',value.max(),historical_scores[b,sid],score=True,branch=b):
                    raise ValueError('offline historical scoring backend changed')
                y,x=np.unravel_index(np.argmax(value),value.shape)
                tables['per_image_scores'].append(dict(branch=b,sample_id=sid,relative_path=sample['relative_path'],label=sample['label'],
                    image_score=float(value.max()) if b==inputs.BRANCH else historical_scores[b,sid],full_peak_y=int(y),full_peak_x=int(x),
                    result_origin='new_projection_calculation' if b==inputs.BRANCH else 'reused_parent_result'))
            rows,deltas=distance_comparison(sample,arrays,cached);tables['distance_differences'].extend(rows)
            for kind,delta in deltas.items():
                for j,layer in enumerate(inputs.LAYERS): pooled[kind,layer].append(delta[j])
            for ref in inputs.REFERENCES:
                tables['map_differences'].append(dict(sample_id=sid,branch=inputs.BRANCH,reference=ref,
                    unit='Source_MAD_response',direction='projection_minus_reference',**difference(maps[inputs.BRANCH],maps[ref])))
            timing['cached_maps_baseline_and_distance_comparisons']+=time.perf_counter()-t;t=time.perf_counter()
            gt=Path(p['paths']['data'])/inputs.parent.shared.gt_relative(sample['relative_path']) if sample['label'] else None
            if gt is not None and not gt.is_file():
                state['missing_gt_ids'].append(sid);raise ValueError('required anomaly GT missing: '+sid)
            mask=None
            if gt is not None:
                with Image.open(gt) as image: mask=np.asarray(image.convert('L'))>0
            instances,peaks=instance_rows(sample,maps,mask,tuple(meta['grid']),backend)
            for row in instances:
                row['calibration_scale_note']='all five branches share frozen original Source MAD'
                if row['branch']!=inputs.BRANCH:
                    old_instance=historical_instances[row['sample_id'],int(row['instance_id']),row['branch']]
                    if old_instance['response_status']!=row['response_status']: raise ValueError('historical spatial state changed')
                    for key in ('instance_max','ring_max','margin'):
                        a,b=row[key],old_instance[key]
                        if (a is None)!=(b==''): raise ValueError('historical missing spatial evidence changed')
                        if a is not None and not audit(tables['numerical_verification'],sid,'historical_instance_'+key,a,float(b),score=True,
                                                       branch=row['branch'],instance_id=row['instance_id']):
                            raise ValueError('historical instance response changed')
            tables['defect_instances'].extend(instances);tables['fixed_reference_instance_peaks'].extend(peaks)
            patch=backend.patch_mask(Image.fromarray(mask.astype(np.uint8)*255),tuple(meta['grid'])) if mask is not None else np.zeros(meta['grid'],bool)
            tables['signed_distance_per_image'].extend(signed_audit(sample,raws,arrays['positions'],patch,backend.mad.scales))
            record=SimpleNamespace(path=Path(p['paths']['data'])/sample['relative_path'],mask_path=gt,label=sample['label'])
            masks.append(evaluator._load_mask(record,tuple(meta['output_shape'])));labels.append(sample['label']);collected.append(maps[inputs.BRANCH])
            timing['GT_spatial_and_signed_diagnostics']+=time.perf_counter()-t
            if sid in {i for i,_ in inputs.replay.CASES}:
                from .visualization import plot_case
                t=time.perf_counter()
                with Image.open(record.path) as image: plot_case(image.convert('RGB'),mask,maps,directory/'figures'/f'{sid}.png')
                timing['fixed_figures']+=time.perf_counter()-t;state['selected_figures']+=1
            state['analyzed_images']+=1
        for (kind,layer),values in pooled.items():
            tables['distance_difference_summary'].append(dict(kind=kind,layer=layer,image_count=297,
                branch=inputs.BRANCH,reference='FASTREF_TARGET512_SOURCE',unit='Euclidean_div_1280',
                direction='projection_minus_FastRef',aggregation='pooled_patches_exact_quantiles',**difference(np.concatenate(values),0)))
        del pooled
        tables['ranking_summary'],tables['ranking_per_image']=ranking_tables(tables['per_image_scores'])
        tables['signed_distance_summary']=signed_summary(tables['signed_distance_per_image'])
        unique={(r['sample_id'],r['instance_id']):r for r in tables['defect_instances']}
        smallest=sorted(unique.values(),key=lambda r:(r['area_original_pixels'],r['relative_path'],r['instance_id']))[:3]
        small_ids={(r['sample_id'],r['instance_id']) for r in smallest}
        for row in tables['defect_instances']: row['original_smallest_three_rule']=(row['sample_id'],row['instance_id']) in small_ids
        tables['small_instance_summary']=[dict(sample_id=r['sample_id'],instance_id=r['instance_id'],area_original_pixels=r['area_original_pixels'],
            patch_coverage=r['patch_coverage'],definition='original three smallest GT components; no tuned threshold') for r in smallest]
        tables['instance_comparisons']=instance_comparisons(tables['defect_instances'])
        from sklearn.metrics import roc_auc_score,average_precision_score
        for row in inputs.historical_metrics(data['paths']['parent']):
            scores=[r['image_score'] for r in tables['per_image_scores'] if r['branch']==row['branch']]
            for key,actual in [('image_AUROC',roc_auc_score(labels,scores)),('image_AP',average_precision_score(labels,scores))]:
                if not audit(tables['numerical_verification'],'background','historical_summary',actual,row[key],score=True,branch=row['branch'],metric=key):
                    raise ValueError('historical aggregate score changed')
            tables['metrics'].append(row)
        state['historical_baselines_reproduced']=True;t=time.perf_counter()
        values=evaluator._evaluate(labels,masks,collected)  # Only the new branch.
        metric=dict(branch=inputs.BRANCH,image_count=297,image_AUROC=values['image_AUROC'],image_AP=values['image_AUPR'],
                    pixel_AUROC=values['pixel_AUROC'],pixel_AP=values['pixel_AUPR'],AUPRO=values['AUPRO'],
                    result_origin='new_projection_calculation',protocol='original ADEval CUDA, 200 thresholds, FPR<=0.3')
        if any(not np.isfinite(metric[k]) or not 0<=metric[k]<=1 for k in inputs.parent.METRIC_KEYS): raise ValueError('invalid new metric')
        scores=[r['image_score'] for r in tables['per_image_scores'] if r['branch']==inputs.BRANCH]
        for key,actual in [('image_AUROC',roc_auc_score(labels,scores)),('image_AP',average_precision_score(labels,scores))]:
            if not audit(tables['numerical_verification'],'background','new_metric_matches_frozen_scores',actual,metric[key],score=True,metric=key):
                raise ValueError('new evaluation backend differs from frozen image scores')
        tables['metrics'].append(metric);tables['metric_comparisons']=metric_comparisons(tables['metrics'])
        timing['one_new_branch_pixel_evaluation']+=time.perf_counter()-t
        runtime.verify_stage(data,p,prepared,parent_state)
        state.update(pixel_metrics_complete=True,numerical_failed_checks=sum(not r['verified'] for r in tables['numerical_verification']))
        if complete(state,tables): state.update(status='analysis_complete',protected_inputs_unchanged=True,GT_used_only_after_predictions_frozen=True)
    except Exception as error:
        state.update(error=f'{type(error).__name__}: {error}',numerical_failed_checks=sum(not r['verified'] for r in tables['numerical_verification']))
    for name,rows in tables.items(): inputs.replay.write_csv(directory/f'{name}.csv',rows)
    state.update(seconds=time.perf_counter()-start,timing=dict(timing));inputs.replay.atomic_json(directory/'status.json',state)
    return state
