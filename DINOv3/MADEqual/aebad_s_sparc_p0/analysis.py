"""Three fixed branches; formal maps use own MAD, distance audit uses old scale."""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image

from .inputs import replay,TOOL,gt_relative,verify_artifacts,verify_files,read_capture
from .projection import BRANCHES,LAYERS,physical_positions
from DINOv3.MADEqual.aebad_s_union_support_loo.analysis import load_raw,branch_maps
from DINOv3.MADEqual.aebad_s_union_support_loo.runner import formal_baselines,audit
from DINOv3.MADEqual.aebad_s_union_diagnostics.spatial import components,instance_regions


def scatter(compact,positions,grid):
    p=physical_positions(positions,grid)
    values=np.asarray(compact)
    if values.shape!=(4,len(p)) or values.dtype!=np.float32 or not np.isfinite(values).all() or (values<0).any():
        raise ValueError('invalid finite four-layer compact distances')
    physical=np.empty_like(values);physical[:,p]=values
    return physical.reshape(4,*grid)


def load_triplet(inputs,sample,cap,old,new):
    paths={k:Path(v) for k,v in inputs['paths'].items()};out=paths['output']
    arrays,meta=load_raw(paths['capture']/old['file'],identity=cap['identity'],sample=sample,checksum=old['sha256'])
    file=out/new['file']
    if new['file']!=f"sparc_raw/{sample['sample_id']}.npz" or replay.sha256(file)!=new['sha256']:
        raise ValueError('SPARC prediction missing/modified/unsafe')
    with np.load(file,allow_pickle=False) as f:
        m=json.loads(str(f['metadata'].item()));d=f['distance'].copy();indices=f['nearest_index'].copy()
        positions=f['positions'].copy();grid=f['grid'].tolist()
    if (m.get('tool')!=TOOL or m['identity']!=inputs['identity'] or m['sample_id']!=sample['sample_id']
            or m['relative_path']!=sample['relative_path'] or m['baseline_raw_sha256']!=old['sha256']
            or m['image_sha256']!=meta['image_sha256'] or m['branch']!='SPARC_SOURCE_K8'
            or m['distance_divisor']!=1280 or m['geometry']!=meta['geometry']
            or m['output_shape']!=meta['output_shape'] or grid!=meta['grid'] or m['grid']!=grid
            or not np.array_equal(positions,arrays['positions']) or indices.dtype!=np.int64
            or indices.shape!=d.shape or (indices<0).any() or (indices>=100000).any()):
        raise ValueError('SPARC prediction provenance/geometry/index mismatch')
    raws=dict(SOURCE=arrays['source_distance'],UNION_K8=np.minimum(arrays['source_distance'],arrays['target_distance'].min(axis=0)),
              SPARC_SOURCE_K8=d)
    for r in raws.values(): scatter(r,positions,tuple(grid))
    return raws,positions,meta


def formal_maps(raws,positions,meta,backends):
    return {b:branch_maps({b:scatter(raws[b],positions,tuple(meta['grid']))},tuple(meta['output_shape']),backends[b])[b]
            for b in BRANCHES}


def rankings(rows,fixed_top10):
    from sklearn.metrics import roc_auc_score,average_precision_score
    by={b:{r['sample_id']:r for r in rows if r['branch']==b} for b in BRANCHES}
    ids=sorted(by['UNION_K8']);labels=np.array([by['UNION_K8'][i]['label'] for i in ids])
    if any(set(g)!=set(ids) for g in by.values()): raise ValueError('three branches must use the identical heldout set')
    n,a=labels==0,labels==1
    def err(scores):
        ns,as_=scores[n,None],scores[None,a]
        return (ns>as_).astype(float)+.5*(ns==as_)
    base=err(np.array([by['UNION_K8'][i]['image_score'] for i in ids]))
    result=[]
    for b in BRANCHES:
        s=np.array([by[b][i]['image_score'] for i in ids]);e=err(s)
        result.append(dict(branch=b,reference='UNION_K8',image_count=len(ids),normal_count=int(n.sum()),anomaly_count=int(a.sum()),
            image_AUROC=float(roc_auc_score(labels,s)),image_AP=float(average_precision_score(labels,s)),
            weighted_pair_errors=float(e.sum()),pair_count=int(e.size),tie_error_weight=.5,
            improved_pairs_vs_UNION=int((e<base).sum()),worsened_pairs_vs_UNION=int((e>base).sum()),unchanged_pairs_vs_UNION=int((e==base).sum()),
            normal_score_p90=float(np.percentile(s[n],90)),normal_score_p95=float(np.percentile(s[n],95)),normal_score_max=float(s[n].max()),
            fixed_UNION_top10_mean_score_change=float(np.mean([by[b][i]['image_score']-by['UNION_K8'][i]['image_score'] for i in fixed_top10]))))
    return result


def spatial_rows(sample,maps,mask,grid,backend):
    rows=[]
    if not sample['label']: return rows
    labels,descriptors=components(mask)
    if not descriptors: raise ValueError('empty anomaly GT cannot complete spatial evidence')
    for desc in descriptors:
        component=labels==desc['instance_id'];inside,ring=instance_regions(component,mask,maps['UNION_K8'].shape)
        maxima={b:(float(m[inside].max()) if inside.any() else None,float(m[ring].max()) if ring.any() else None) for b,m in maps.items()}
        ui,ur=maxima['UNION_K8'];um=ui-ur if ui is not None and ur is not None else None
        for b,(im,rm) in maxima.items():
            margin=im-rm if im is not None and rm is not None else None
            rows.append(dict(sample_id=sample['sample_id'],relative_path=sample['relative_path'],**desc,branch=b,
                instance_area_output_pixels=int(inside.sum()),ring_area_output_pixels=int(ring.sum()),
                patch_coverage=int(backend.patch_mask(Image.fromarray(component.astype(np.uint8)*255),grid).sum()),
                instance_max=im,ring_max=rm,margin=margin,margin_change_vs_UNION=margin-um if margin is not None and um is not None else None,
                ring_radius_original_pixels=16,ring_metric='Chebyshev',ring_excludes_all_GT=True,
                response_status='mapped_instance_disappeared' if not inside.any() else ('empty_ring' if not ring.any() else 'available'),
                calibration_scale_note='own branch MAD; cross-branch margin differences are not raw-distance reductions',
                annotation_semantics='unreviewed; no size threshold'))
    return rows


def distance_audit(sample,raws,positions,patch_mask,old_scales):
    scales=np.asarray(old_scales,np.float64)
    if scales.shape!=(4,) or not np.isfinite(scales).all() or (scales<=0).any(): raise ValueError('invalid old fixed MAD scale')
    selected=np.ones(len(positions),bool) if not sample['label'] else patch_mask.reshape(-1)[positions]
    if not selected.any(): raise ValueError('required distance audit region empty')
    rows=[]
    for b in BRANCHES:
        delta=(raws['UNION_K8'].astype(np.float64)-raws[b].astype(np.float64))/scales[:,None]
        for offset,l in enumerate(LAYERS):
            rows.append(dict(sample_id=sample['sample_id'],label=sample['label'],branch=b,layer=l,
                reference='UNION_K8',scale='old_Source_MAD_scale_no_refit',R_value=float(delta[offset,selected].mean()),
                selected_patch_count=int(selected.sum()),region='normal_all_patches' if not sample['label'] else 'GT_defect_patches'))
    return rows


def summarize_audit(rows):
    result=[]
    for b in BRANCHES:
        group=[]
        for l in LAYERS:
            values=[r for r in rows if r['branch']==b and r['layer']==l]
            n=[r['R_value'] for r in values if not r['label']];a=[r['R_value'] for r in values if r['label']]
            rn,ra=(float(np.mean(v)) if v else None for v in (n,a))
            row=dict(branch=b,layer=l,reference='UNION_K8',scale='old_Source_MAD_scale',R_N=rn,R_A=ra,
                     G=rn-ra if rn is not None and ra is not None else None,normal_image_count=len(n),anomaly_image_count=len(a),aggregation='image_equal')
            result.append(row);group.append(row)
        result.append(dict(branch=b,layer='four_layer_mean',reference='UNION_K8',scale='old_Source_MAD_scale',
            **{k:float(np.mean([r[k] for r in group])) if all(r[k] is not None for r in group) else None for k in ('R_N','R_A','G')},
            normal_image_count=min(r['normal_image_count'] for r in group),anomaly_image_count=min(r['anomaly_image_count'] for r in group),aggregation='equal_four_layer_image_equal'))
    return result


def runtime_backends(inputs):
    from DINOv3.MADEqual.aebad_s_union_diagnostics.spatial import MapBackend
    paths={k:Path(v) for k,v in inputs['paths'].items()}
    old=MapBackend(paths['results']/'source_mad_state.npz');new=MapBackend(paths['output']/'sparc_source_mad_state.npz')
    return dict(SOURCE=old,UNION_K8=old,SPARC_SOURCE_K8=new)


def runtime_evaluator():
    from DINOv3.MADEqual.aebad_s_full_p0 import runner
    from DINOv3.relation_reliability.official_metrics import validate_official_metric_backend
    validate_official_metric_backend()
    return runner


def evaluate(inputs,prepared):
    out=Path(inputs['paths']['output']);paths={k:Path(v) for k,v in inputs['paths'].items()}
    if (out/'evaluation').exists(): raise ValueError('evaluation refuses existing/partial results')
    infer=json.loads((out/'infer_status.json').read_text(encoding='utf-8'))
    if (infer.get('tool')!=TOOL or infer['identity']!=inputs['identity'] or infer['status']!='predictions_frozen'
            or infer['predicted_images']!=297 or infer['inputs_sha256']!=replay.sha256(out/'inputs.json')
            or infer['prepare_status_sha256']!=replay.sha256(out/'prepare_status.json')
            or [r['sample_id'] for r in infer['records']]!=[s['sample_id'] for s in inputs['samples']]):
        raise ValueError('all 297 frozen predictions required before GT evaluation')
    verify_artifacts(out,infer['records']);verify_artifacts(out,prepared['artifacts']);verify_files(inputs['required_files'])
    cap,old_status=read_capture(paths['capture']);old={r['sample_id']:r for r in old_status['records']};new={r['sample_id']:r for r in infer['records']}
    root=out/'evaluation';root.mkdir();start=time.perf_counter();timing=defaultdict(float)
    state=dict(tool=TOOL,stage='evaluate',status='running',identity=inputs['identity'],development_domain=True,
               analyzed_images=0,incomplete_spatial_ids=[],numerical_failed_checks=0,pixel_metrics_complete=False,
               predictions_frozen_before_GT=True,model_inference=False)
    tables=defaultdict(list)
    try:
        backends=runtime_backends(inputs);evaluator=runtime_evaluator()
        formal=formal_baselines(paths['results'],inputs['samples']);fixed={sid for sid,_ in replay.CASES}
        full_masks=[];labels=[s['label'] for s in inputs['samples']]
        for i,s in enumerate(inputs['samples']):
            t=time.perf_counter();raws,p,meta=load_triplet(inputs,s,cap,old[s['sample_id']],new[s['sample_id']])
            maps=formal_maps(raws,p,meta,backends);timing['cached_reading_and_maps']+=time.perf_counter()-t
            for b,old_b in (('SOURCE','Source'),('UNION_K8','K8')):
                audit(tables['numerical_verification'],s['sample_id'],'formal_baseline_score',maps[b].max(),formal[old_b][s['sample_id']]['image_score'],score=True,branch=b)
            for b in BRANCHES:
                y,x=np.unravel_index(np.argmax(maps[b]),maps[b].shape)
                tables['per_image_scores'].append(dict(sample_id=s['sample_id'],relative_path=s['relative_path'],label=s['label'],
                    branch=b,image_score=float(maps[b].max()),score_change_vs_UNION=float(maps[b].max()-maps['UNION_K8'].max()),full_peak_y=int(y),full_peak_x=int(x),
                    MAD_state='sparc_source_mad_state.npz' if b=='SPARC_SOURCE_K8' else 'original source_mad_state.npz'))
            t=time.perf_counter();record=SimpleNamespace(path=str(paths['data']/s['relative_path']),label=s['label'],
                mask_path=str(paths['data']/gt_relative(s['relative_path'])) if s['label'] else None)
            full_masks.append(evaluator._load_mask(record,tuple(meta['output_shape'])))
            mask=patch=None
            if s['label']:
                with Image.open(record.mask_path) as image: mask=np.asarray(image.convert('L'))>0
                if mask.shape!=(s['image_height'],s['image_width']): raise ValueError('GT original geometry changed')
                if not mask.any(): raise ValueError('empty GT')
                patch=backends['SOURCE'].patch_mask(Image.fromarray(mask.astype(np.uint8)*255),tuple(meta['grid']))
                tables['defect_instances'].extend(spatial_rows(s,maps,mask,tuple(meta['grid']),backends['SOURCE']))
            tables['distance_audit_per_image'].extend(distance_audit(s,raws,p,patch,backends['SOURCE'].mad.scales))
            timing['GT_and_spatial_audit']+=time.perf_counter()-t
            if s['sample_id'] in fixed:
                from .visualization import plot_case
                t=time.perf_counter()
                with Image.open(record.path) as image: plot_case(image.convert('RGB'),mask,maps,root/'figures'/f"{s['sample_id']}.png")
                timing['figures']+=time.perf_counter()-t
            state['analyzed_images']+=1
            if i%25==0 or i==296: print(f'[SPARC evaluate] {i+1}/297',flush=True)
        ranking=rankings(tables['per_image_scores'],[sid for sid,_ in replay.CASES[:10]])
        tables['ranking_summary']=ranking
        from sklearn.metrics import roc_auc_score,average_precision_score
        for b,old_b in (('SOURCE','Source'),('UNION_K8','K8')):
            summary=next(r for r in ranking if r['branch']==b);ids=sorted(formal[old_b])
            labs=[formal[old_b][sid]['label'] for sid in ids];scores=[formal[old_b][sid]['image_score'] for sid in ids]
            for key,expected in (('image_AUROC',roc_auc_score(labs,scores)),('image_AP',average_precision_score(labs,scores))):
                audit(tables['numerical_verification'],'background','formal_baseline_summary',summary[key],expected,score=True,branch=b,metric=key)
        tables['distance_audit_summary']=summarize_audit(tables['distance_audit_per_image'])
        for b in BRANCHES:
            t=time.perf_counter();maps=[]
            for s in inputs['samples']:
                raws,p,meta=load_triplet(inputs,s,cap,old[s['sample_id']],new[s['sample_id']])
                maps.append(branch_maps({b:scatter(raws[b],p,tuple(meta['grid']))},tuple(meta['output_shape']),backends[b])[b])
            values=evaluator._evaluate(labels,full_masks,maps)
            metric=dict(branch=b,image_count=297,image_AUROC=values['image_AUROC'],image_AP=values['image_AUPR'],
                pixel_AUROC=values['pixel_AUROC'],pixel_AP=values['pixel_AUPR'],AUPRO=values['AUPRO'],
                protocol='original full-protocol; ADEval CUDA 200 thresholds FPR<=0.3')
            if not all(np.isfinite(metric[k]) and 0<=metric[k]<=1 for k in ('image_AUROC','image_AP','pixel_AUROC','pixel_AP','AUPRO')):
                raise ValueError('invalid official metric')
            summary=next(r for r in ranking if r['branch']==b)
            for key in ('image_AUROC','image_AP'):
                audit(tables['numerical_verification'],'background','full_metric_matches_scores',metric[key],summary[key],score=True,branch=b,metric=key)
            tables['metrics'].append(metric);del maps
            timing['full_pixel_evaluation_including_map_reconstruction']+=time.perf_counter()-t
        verify_files(inputs['required_files']);verify_artifacts(out,prepared['artifacts']);verify_artifacts(out,infer['records'])
        state['numerical_failed_checks']=sum(not r['verified'] for r in tables['numerical_verification'])
        state['pixel_metrics_complete']=len(tables['metrics'])==3
        state['formal_baseline_reproduced']=all(r['verified'] for r in tables['numerical_verification'] if r['check_kind'].startswith('formal_'))
        complete=(state['analyzed_images']==297 and not state['numerical_failed_checks'] and state['pixel_metrics_complete']
            and len(tables['per_image_scores'])==891 and all(r['normal_image_count']==85 and r['anomaly_image_count']==212 for r in tables['distance_audit_summary']))
        state['status']='analysis_complete' if complete else 'partial_evidence'
        state['protected_inputs_unchanged']=True;state['selected_figures']=12
    except Exception as e:
        state['status']='partial_evidence';state['error']=f'{type(e).__name__}: {e}'
        state['numerical_failed_checks']=sum(not r['verified'] for r in tables['numerical_verification'])
    for name,rows in tables.items():
        if name=='defect_instances': rows.sort(key=lambda r:(r['area_original_pixels'],r['sample_id'],r['instance_id'],r['branch']))
        replay.write_csv(root/f'{name}.csv',rows)
    state['seconds']=time.perf_counter()-start;state['timing']=dict(timing)
    replay.atomic_json(root/'status.json',state)
    lines=['# AeBAD-S/background SPARC same-host comparison','',f"Status: {state['status']}; background is a development domain.",'',
        '| branch | Image AUROC | Image AP | Pixel AUROC | Pixel AP | AUPRO@0.3 |','|---|---:|---:|---:|---:|---:|']
    for r in tables['metrics']: lines.append('| '+r['branch']+' | '+' | '.join(f'{r[k]:.9f}' for k in ('image_AUROC','image_AP','pixel_AUROC','pixel_AP','AUPRO'))+' |')
    lines += ['', 'Formal scores use branch-specific MAD. Signed distance audits use the same old Source scale.',
        'Instance margins are in different calibration scales; their changes alone do not establish defect preservation.',
        'G>0 is an aggregate and does not guarantee preservation of every defect. Semantic review is pending.',
        'This is a DINOv3 host migration of an external method, not a reproduction of the paper tables or independent final validation.',
        f"Error: {state.get('error','')}"]
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return state
