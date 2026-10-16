"""CPU evaluation AFTER prediction freeze. Historical pixels are explicitly reused."""
from __future__ import annotations

import gc
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import statistics, ch1_score, pair_errors, image_metrics
from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic.numerics import reconstruct, adaptive_max, check
from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic.metrics import exact_binary, resource_bytes
from DINOv3.MADEqual.aebad_s_union_diagnostics.spatial import components, instance_regions, resized_mask
from . import inputs as ip
from .gates import BRANCHES, LAYERS, ATOL, RTOL
from .prediction import bounded_images, fresh, verified_predictions, load_prediction, original_distances, protect_dependencies

U, S, P, G = BRANCHES
HISTORY_BRANCH = {U:'K8_UNION', S:'K8_SEGMENT'}
COMPARISONS = ((S,U), (P,U), (G,U), (P,S), (G,S), (G,P))


class Dataset:
    """Read GT only in evaluate; RGB only in visualize. No train directory needed."""
    def __init__(self, ctx):
        self.ctx = ctx;path=ctx['paths']['data']
        self.zip=zipfile.ZipFile(path) if path.is_file() else None
        self.root=None if self.zip else ip.dataset_root(path)
    def close(self):
        if self.zip:self.zip.close()
    def __enter__(self): return self
    def __exit__(self, *args): self.close()
    def image(self, relative, expected):
        data = self.zip.read('AeBAD/AeBAD_S/'+ip.safe(relative)) if self.zip else (self.root/ip.safe(relative)).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected: raise ValueError('dataset image modified: '+relative)
        with Image.open(io.BytesIO(data)) as image: return image.copy()
    def gt(self, sample):
        shape = (sample['image_height'],sample['image_width'])
        if sample['label'] == 0: return np.zeros(shape,bool)
        relative = sample['relative_path'].replace('test/', 'ground_truth/', 1)
        image = self.image(relative, self.ctx['capture']['gt_sha256'][relative])
        mask = np.asarray(image.convert('L')) > 0
        if mask.shape != shape or not mask.any(): raise ValueError('required abnormal GT missing, empty or wrong shape')
        return mask


def history(ctx):
    with ip.Archive(ctx['paths']['review_archive']) as a:
        old = {name:a.csv(ip.SROOT+'/'+name) for name in (
            'metrics.csv','per_image_scores.csv','defect_instances.csv','peak_positions.csv')}
    old['scores'] = {(r['sample_id'],r['branch']):float(r['image_score']) for r in old['per_image_scores.csv']}
    old['instances'] = {(r['sample_id'],int(r['instance_id']),r['branch']):r for r in old['defect_instances.csv']}
    old['peaks'] = {(r['sample_id'],r['region'],int(r['instance_id'] or 0)):(int(r['output_y']),int(r['output_x']))
                    for r in old['peak_positions.csv'] if r['peak_of_branch']=='K8_UNION' and r['measured_branch']=='K8_UNION'
                    and r.get('output_y') and r.get('output_x')}
    old['ch1_scores'] = {(r['sample_id'],r['branch']):float(r['image_score'])
                         for r in ip.read_csv((ctx['paths']['ch1_dir']/'per_image_scores.csv').read_bytes())}
    old['ch1_metrics'] = ip.read_csv((ctx['paths']['ch1_dir']/'image_metrics.csv').read_bytes())
    old['ch1_rank'] = ip.read_csv((ctx['paths']['ch1_dir']/'ranking_summary.csv').read_bytes())
    old['fixed10'] = {s['sample_id'] for s in ctx['capture']['selected_cases'] if s['label']==0}
    if len(old['fixed10']) != 10: raise ValueError('original fixed ten normal IDs unavailable')
    # Frozen ORIGINAL small-instance description: three smallest original-GT components,
    # tie by sample ID then original 8-connected component ID; never reselect by method.
    originals = [r for r in old['defect_instances.csv'] if r['branch']=='K8_UNION']
    small = sorted(originals,key=lambda r:(int(r['area_original_pixels']),r['sample_id'],int(r['instance_id'])))[:3]
    old['small3'] = {(r['sample_id'],int(r['instance_id'])) for r in small}
    return old


def score_maps(ctx, raw, meta):
    p = meta['positions']; grid = tuple(meta['grid']); shape = tuple(meta['output_shape'])
    current, peak, local = statistics(raw,p,grid,shape,ctx['medians'],ctx['scales'])
    _, maps, interpolation_error = reconstruct(raw,p,grid,shape,ctx['medians'],ctx['scales'])
    return current, ch1_score(peak,local,ctx['state']), maps['FUSED'], interpolation_error


def peak_position(response, region):
    if not region.any(): return None
    offset = int(np.argmax(np.where(region,response,-np.inf)))
    return tuple(int(v) for v in np.unravel_index(offset,response.shape))


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def verify_map_record(expected,response,mask):
    if (response.dtype!=np.float32 or mask.dtype!=np.uint8 or mask.shape!=response.shape
            or array_sha(response)!=expected['map_float32_sha256'] or array_sha(mask)!=expected['mask_uint8_sha256']
            or response.shape!=(int(expected['output_height']),int(expected['output_width']))):
        raise ValueError('AUPRO maps/masks differ from frozen CPU pixel evaluation; no new approval or tolerance relaxation')


def instance_rows(sample, gt, maps, grid, old, checks):
    """Same 8-connectivity and original-pixel Chebyshev ring as previous SEGMENT."""
    sid = sample['sample_id']; labelled, descriptors = components(gt)
    rows, peaks = [], []
    for desc in descriptors:
        instance_id = int(desc['instance_id']); key = (sid,instance_id)
        component = labelled == instance_id
        inside, ring = instance_regions(component,gt,maps[U].shape)
        regions = {'instance':inside,'ring':ring}
        values = {}
        fixed = {}
        for region, mask in regions.items():
            anchor = old['peaks'].get((sid,region,instance_id))
            if mask.any():
                if anchor is None or not mask[anchor]: raise ValueError('original Union peak missing/outside original region')
                check(checks,sid,'historical_union_peak_is_max',maps[U][anchor],maps[U][mask].max(),instance_id=instance_id,region=region)
            elif anchor is not None: raise ValueError('historical peak exists for empty mapped region')
            fixed[region] = anchor
        for branch, response in maps.items():
            own = {r:peak_position(response,mask) for r,mask in regions.items()}
            maxima = {r:float(response[p]) if p is not None else None for r,p in own.items()}
            margin = maxima['instance']-maxima['ring'] if all(v is not None for v in maxima.values()) else None
            fixed_values = {r:float(response[p]) if p is not None else None for r,p in fixed.items()}
            fixed_margin = fixed_values['instance']-fixed_values['ring'] if all(v is not None for v in fixed_values.values()) else None
            state = 'mapped_instance_disappeared' if not inside.any() else 'empty_ring' if not ring.any() else 'available'
            row = dict(sample_id=sid,relative_path=sample['relative_path'],**desc,branch=branch,
                       patch_coverage=int(adaptive_max(component,grid).sum()),
                       instance_area_output_pixels=int(inside.sum()), ring_area_output_pixels=int(ring.sum()),
                       ring_radius_original_pixels=16,ring_metric='Chebyshev',ring_excludes_all_GT=True,
                       response_status=state,small_original_three=key in old['small3'],
                       instance_max=maxima['instance'],ring_max=maxima['ring'],margin=margin,
                       fixed_union_instance_response=fixed_values['instance'],fixed_union_ring_response=fixed_values['ring'],
                       fixed_union_margin=fixed_margin)
            for region in regions:
                p, a = own[region], fixed[region]
                peaks.append(dict(sample_id=sid,instance_id=instance_id,region=region,branch=branch,
                                  own_y=p[0] if p else None,own_x=p[1] if p else None,
                                  union_anchor_y=a[0] if a else None,union_anchor_x=a[1] if a else None,
                                  own_response=maxima[region],fixed_union_response=fixed_values[region],
                                  own_peak_differs_from_union=p!=a,peak_tie_rule='first_row_major'))
            if branch in HISTORY_BRANCH:
                reference = old['instances'][(*key,HISTORY_BRANCH[branch])]
                for field in ('area_original_pixels','instance_area_output_pixels','ring_area_output_pixels','patch_coverage'):
                    check(checks,sid,'historical_instance_'+field,row[field],int(reference[field]),atol=0,instance_id=instance_id,branch=branch)
                for field in ('instance_max','ring_max','margin'):
                    if row[field] is None:
                        if reference[field] not in ('',None): raise ValueError('historical instance computability changed')
                    else: check(checks,sid,'historical_instance_'+field,row[field],float(reference[field]),instance_id=instance_id,branch=branch)
            values[branch] = row
        for branch,row in values.items():
            for ref in (U,S):
                for field in ('instance_max','ring_max','margin','fixed_union_margin'):
                    row[f'{field}_change_vs_{ref}'] = row[field]-values[ref][field] if row[field] is not None and values[ref][field] is not None else None
            rows.append(row)
    return rows, peaks


def mechanism_rows(sample, gt, raws, positions, grid, scales, prediction):
    # EXACT old SEGMENT normalization: float32 scales+1e-8, then float64 delta.
    denominator = scales + 1e-8
    region = np.ones(len(positions),bool) if sample['label']==0 else adaptive_max(gt,grid).ravel()[positions]
    group = 'normal_all_patch' if sample['label']==0 else 'anomaly_GT_patch'
    original = (raws[U].astype(np.float64)-raws[S])/denominator[:,None]
    rows = []
    for branch in (S,P,G):
        delta = (raws[U].astype(np.float64)-raws[branch])/denominator[:,None]
        for off, layer in enumerate(LAYERS):
            old_sum = float(original[off,region].sum())
            new_sum = float(delta[off,region].sum())
            old_count = int(np.sum((raws[U][off]-raws[S][off] > ATOL+RTOL*np.abs(raws[U][off])) & region))
            new_count = int(np.sum((raws[U][off]-raws[branch][off] > ATOL+RTOL*np.abs(raws[U][off])) & region))
            row = dict(sample_id=sample['sample_id'],branch=branch,layer=layer,group=group,selected_patches=int(region.sum()),
                       R_value=float(delta[off,region].mean()) if region.any() else None,
                       signed_scaled_reduction_sum=new_sum,segment_signed_scaled_reduction_sum=old_sum,
                       extra_reduction_amount_retained_ratio=new_sum/old_sum if old_sum!=0 else None,
                       segment_reduced_patches=old_count,branch_reduced_patches=new_count,
                       extra_reduction_positions_retained_ratio=new_count/old_count if old_count else None,
                       normalization='float32_Source_scale_plus_1e-8',reference=U)
            if branch in (P,G):
                gate = prediction['pair_gate' if branch==P else 'geom_gate'][off].ravel()[positions]
                row['gate_accept_ratio_in_group'] = float(gate[region].mean()) if region.any() else None
            rows.append(row)
    return rows


def group_indices(samples, fixed):
    y = np.asarray([s['label'] for s in samples])
    hard = np.asarray([s['sample_id'] in fixed for s in samples])
    return y, {'all85':np.ones(len(samples),bool),'fixed10':(y==1)|hard,'other75':(y==1)|~hard}


def image_tables(samples, scores, old, checks):
    y, groups = group_indices(samples,old['fixed10'])
    arrays = {b:np.asarray([scores[(s['sample_id'],b)] for s in samples]) for b in BRANCHES}
    inherited = np.asarray([old['ch1_scores'][(s['sample_id'],U)] for s in samples])
    # Pair ordering is checked separately: closeness alone does NOT imply it.
    check(checks,'ALL','CH1_union_pair_order_exact',pair_errors(y,arrays[U]),pair_errors(y,inherited),atol=0)
    metrics, rankings = [], []
    for name, use in groups.items():
        yy = y[use]
        for branch in BRANCHES:
            metric = image_metrics(yy,arrays[branch][use])
            metrics.append(dict(branch=branch,normal_group=name,image_count=int(use.sum()),
                                normal_count=int(np.sum(yy==0)),anomaly_count=int(np.sum(yy==1)),
                                **metric,image_source='new_CPU_frozen_CH1',units='fraction'))
            if branch == U:
                reference = next(r for r in old['ch1_metrics'] if r['branch']==U and r['normal_group']==name)
                for k,v in metric.items(): check(checks,'ALL','historical_CH1_'+k,v,float(reference[k]),atol=1e-12,normal_group=name)
        for branch,ref in COMPARISONS:
            errors, previous = pair_errors(yy,arrays[branch][use]),pair_errors(yy,arrays[ref][use])
            rankings.append(dict(branch=branch,reference=ref,normal_group=name,normal_count=int(np.sum(yy==0)),
                                 anomaly_count=int(np.sum(yy==1)),pair_count=int(errors.size),
                                 weighted_pair_errors=float(errors.sum()),reference_weighted_pair_errors=float(previous.sum()),
                                 delta_pair_errors=float((errors-previous).sum()),
                                 improved_pairs=int(np.sum(errors<previous)),worsened_pairs=int(np.sum(errors>previous)),
                                 unchanged_pairs=int(np.sum(errors==previous)),tie_weight=0.5,
                                 interpretation='ordering_only_not_threshold_false_positive_or_negative'))
        reference = next(r for r in old['ch1_rank'] if r['branch']==U and r['normal_group']==name)
        check(checks,'ALL','historical_CH1_weighted_pair_errors',pair_errors(yy,arrays[U][use]).sum(),float(reference['weighted_pair_errors']),atol=0,normal_group=name)
    return metrics,rankings


def comparisons(metrics):
    result = []
    for branch,ref in COMPARISONS:
        for row in [r for r in metrics if r['branch']==branch]:
            reference = next(r for r in metrics if r['branch']==ref and r['normal_group']==row['normal_group'])
            for key in ('image_AUROC','image_AP','pixel_AUROC','pixel_AP','AUPRO'):
                a,b = row.get(key),reference.get(key)
                result.append(dict(branch=branch,reference=ref,normal_group=row['normal_group'],metric=key,
                                   value=a,reference_value=b,delta=a-b if a is not None and b is not None else None,
                                   direction='branch_minus_reference',units='fraction',
                                   status='available' if a is not None and b is not None else 'pending_or_not_applicable'))
    return result


def mechanism_summary(rows):
    result=[]
    for branch in (S,P,G):
        for layer in (*LAYERS,'MEAN4'):
            values={}
            for group,name in (('normal_all_patch','R_N'),('anomaly_GT_patch','R_A')):
                chosen=[r for r in rows if r['branch']==branch and r['group']==group and (layer=='MEAN4' or r['layer']==layer)]
                # Every valid image has equal weight within a layer, then four
                # layer means have equal weight (also when some GT regions are NA).
                means=[np.mean([r['R_value'] for r in chosen if r['layer']==l and r['R_value'] is not None])
                       for l in (LAYERS if layer=='MEAN4' else (layer,))
                       if any(r['layer']==l and r['R_value'] is not None for r in chosen)]
                values[name]=float(np.mean(means)) if means else None
                ratio_means=[np.mean([r['extra_reduction_amount_retained_ratio'] for r in chosen
                                      if r['layer']==l and r['extra_reduction_amount_retained_ratio'] is not None])
                             for l in (LAYERS if layer=='MEAN4' else (layer,))
                             if any(r['layer']==l and r['extra_reduction_amount_retained_ratio'] is not None for r in chosen)]
                values[group+'_amount_retained_image_layer_equal']=float(np.mean(ratio_means)) if ratio_means else None
                values[group+'_valid_image_layers']=sum(r['R_value'] is not None for r in chosen)
                values[group+'_zero_denominators']=sum(r['extra_reduction_amount_retained_ratio'] is None for r in chosen)
            result.append(dict(branch=branch,layer=layer,reference=U,**values,
                               G=values['R_N']-values['R_A'] if all(v is not None for v in (values['R_N'],values['R_A'])) else None,
                               aggregation='image_equal_within_layer_then_layer_equal'))
    return result


def spatial_summary(rows):
    result=[]
    for branch,ref in COMPARISONS:
        subset=[r for r in rows if r['branch']==branch]
        lookup={(r['sample_id'],r['instance_id']):r for r in rows if r['branch']==ref}
        for group in ('all','small_original_three'):
            selected=[r for r in subset if group=='all' or r['small_original_three']]
            for definition in ('margin','fixed_union_margin'):
                delta=[r[definition]-lookup[(r['sample_id'],r['instance_id'])][definition] for r in selected
                       if r[definition] is not None and lookup[(r['sample_id'],r['instance_id'])][definition] is not None]
                result.append(dict(branch=branch,reference=ref,group=group,definition=definition,
                                   instances=len(selected),computable=len(delta),not_computable=len(selected)-len(delta),
                                   improved=sum(d>1e-6 for d in delta),worsened=sum(d < -1e-6 for d in delta),
                                   unchanged=sum(abs(d)<=1e-6 for d in delta),mean_signed_change=float(np.mean(delta)) if delta else None,
                                   tolerance=1e-6,units='frozen_Source_patch_MAD_response'))
    return result


def evaluate(ctx, prediction_dir, output, workers=4, memory_budget_bytes=8<<30):
    started=time.perf_counter(); pred,records=verified_predictions(ctx,prediction_dir)
    protect_dependencies(output,pred)
    out=fresh(ctx,output,'evaluate'); old=history(ctx)
    scratch=out/'_scratch'; scratch.mkdir()
    total=sum((s['image_height']//4)*(s['image_width']//4) for s in ctx['samples'])
    if resource_bytes(total)>memory_budget_bytes: raise MemoryError('exact full-pixel sort exceeds budget; no approximation')
    if ip.available_ram()<resource_bytes(total)+(512<<20):
        raise MemoryError('insufficient available RAM for exact full pixels plus working headroom; close applications or use another machine')
    import shutil
    if shutil.disk_usage(out).free < total*9+(512<<20): raise OSError('insufficient scratch disk for full pixels')
    labels=np.memmap(scratch/'labels.bin',mode='w+',dtype=np.uint8,shape=(total,))
    pixel={b:np.memmap(scratch/(b+'.bin'),mode='w+',dtype=np.float32,shape=(total,)) for b in (P,G)}
    checks=[]; score_rows=[]; instance=[]; peaks=[]; mechanism=[]; audit=[]; scores={};map_records=[]
    offset=0; timings={'read_gate_recheck_maps_GT_spatial':0.,'exact_pixel_metrics':0.}
    def one(sample):
        tick=time.perf_counter()
        with ip.Archive(ctx['paths']['raw_archive'], index=ctx['raw_index']) as archive:
            raw,meta,segments=ip.load_image(ctx,archive,sample)
        saved=load_prediction(ctx,pred,records[sample['sample_id']],raw,meta,segments)
        union,segment=original_distances(raw,meta,segments)
        raws={U:union,S:segment,P:saved['pair_distance'],G:saved['neighbor_distance']}
        local_checks=[]; maps={}; local_scores=[]; sid=sample['sample_id']
        for branch,value in raws.items():
            current,score,response,error=score_maps(ctx,value,dict(meta,positions=raw['positions']))
            maps[branch]=response
            check(local_checks,sid,'finite_CH1',score,score,branch=branch)
            local_scores.append(dict(sample_id=sid,relative_path=sample['relative_path'],label=sample['label'],branch=branch,
                                     image_score=score,current_check_score=current if branch in (U,S) else None,
                                     image_calibration_state_sha256=ctx['state']['state_sha256'],
                                     fixed_hard_normal=sid in old['fixed10'],localization='unchanged_MAD_equal4_bilinear'))
            if branch in (U,S):
                check(local_checks,sid,'historical_CURRENT',current,old['scores'][(sid,HISTORY_BRANCH[branch])],branch=branch)
                if branch==U:
                    check(local_checks,sid,'formal_UNION_CURRENT',current,sample['union_score'])
                    check(local_checks,sid,'historical_CH1_score',score,old['ch1_scores'][(sid,U)])
        with Dataset(ctx) as data: gt=data.gt(sample)
        if tuple(meta['output_shape']) != (sample['image_height']//4,sample['image_width']//4):
            raise ValueError('original evaluation output dimensions changed')
        instances,peak_rows=instance_rows(sample,gt,maps,tuple(meta['grid']),old,local_checks)
        # Global own peak and original fixed Union full-image peak.
        anchor=old['peaks'].get((sid,'full',0))
        if anchor is None:
            # Historical full-image instance ID may be blank/-1 in archived schema.
            candidates=[p for (ss,r,_),p in old['peaks'].items() if ss==sid and r=='full']
            if len(candidates)!=1: raise ValueError('original Union full-image peak unavailable')
            anchor=candidates[0]
        check(local_checks,sid,'historical_union_full_peak',maps[U][anchor],maps[U].max())
        for b,response in maps.items():
            own=peak_position(response,np.ones(response.shape,bool))
            peak_rows.append(dict(sample_id=sid,instance_id=0,region='full',branch=b,own_y=own[0],own_x=own[1],
                                  union_anchor_y=anchor[0],union_anchor_x=anchor[1],own_response=float(response[own]),
                                  fixed_union_response=float(response[anchor]),own_peak_differs_from_union=own!=anchor,
                                  peak_tie_rule='first_row_major'))
        mech=mechanism_rows(sample,gt,raws,raw['positions'],tuple(meta['grid']),ctx['scales'],saved)
        audits=[]
        for off,l in enumerate(LAYERS):
            for b in (S,P,G):
                delta=raws[b][off].astype(float)-union[off]
                audits.append(dict(sample_id=sid,layer=l,branch=b,reference=U,
                                   signed_mean_difference=float(delta.mean()),max_abs_difference=float(np.abs(delta).max()),
                                   q00=float(np.min(delta)),q50=float(np.median(delta)),q95=float(np.quantile(delta,.95)),q100=float(np.max(delta)),
                                   units='Euclidean_over_1280_once',distance_containment_verified=True,endpoint_identity_verified=True))
        return sample,maps,resized_mask(gt,maps[U].shape),local_scores,instances,peak_rows,mech,audits,local_checks,time.perf_counter()-tick
    try:
        tick=time.perf_counter()
        for sample,maps,mask,sc,inst,pk,mech,ad,ch,seconds in bounded_images(ctx['samples'],one,workers):
            count=mask.size; labels[offset:offset+count]=mask.ravel()
            for b in (P,G): pixel[b][offset:offset+count]=maps[b].ravel()
            for branch,response in maps.items():
                map_records.append(dict(sample_id=sample['sample_id'],branch=branch,
                                        output_height=response.shape[0],output_width=response.shape[1],
                                        map_float32_sha256=array_sha(response),mask_uint8_sha256=array_sha(mask.astype(np.uint8))))
            offset+=count;score_rows.extend(sc);instance.extend(inst);peaks.extend(pk);mechanism.extend(mech);audit.extend(ad);checks.extend(ch)
            scores.update({(r['sample_id'],r['branch']):r['image_score'] for r in sc})
            print(f'[neighborhood evaluate] {sample["sample_id"]} frozen predictions only',flush=True)
        timings['read_gate_recheck_maps_GT_spatial']=time.perf_counter()-tick
        if offset!=total: raise ValueError('incomplete full-pixel population')
        metrics,rankings=image_tables(ctx['samples'],scores,old,checks)
        # Establish complete original instance scope, not just internally consistent new masks.
        check(checks,'ALL','original_instance_scope',len(instance)//4,len(old['instances'])//3,atol=0)
        ip.write_csv(out/'numerical_verification.csv',checks)
        if not checks or not all(r['verified'] for r in checks): raise ValueError('baseline/geometry/numerical verification failed; metrics not published')
        # Baselines passed before publishing any new metric. Original pixel metrics
        # are valid historical references because localization/GT/inputs were checked.
        tick=time.perf_counter()
        for branch in BRANCHES:
            row=next(r for r in metrics if r['branch']==branch and r['normal_group']=='all85')
            if branch in HISTORY_BRANCH:
                historical=next(r for r in old['metrics.csv'] if r['branch']==HISTORY_BRANCH[branch])
                row.update({k:float(historical[k]) for k in ('pixel_AUROC','pixel_AP','AUPRO')})
                row.update(pixel_source=f'historical_reuse:{ip.SROOT}/metrics.csv:{HISTORY_BRANCH[branch]}',
                           pixel_status='historical_not_re_evaluated',aupro_source='historical_ADEval_CUDA_200_FPR_0.3')
            else:
                value=exact_binary(labels,pixel[branch],memory_budget_bytes=memory_budget_bytes)
                if not value['valid']: raise ValueError('full pixel metrics not computable')
                row.update(pixel_AUROC=value['AUROC'],pixel_AP=value['AP'],AUPRO=None,
                           pixel_source='new_exact_CPU_all_pixels_no_quantization',pixel_count=total,
                           pixel_status=value['status'],aupro_source='pending_explicit_CUDA_ADEval_200_FPR_0.3')
        timings['exact_pixel_metrics']=time.perf_counter()-tick
        verified_predictions(ctx,pred)
        if ip.context(ctx['paths'])['identity']!=ctx['identity']: raise ValueError('protected inputs/code changed during evaluation')
        tables={'metrics.csv':metrics,'metric_comparisons.csv':comparisons(metrics),'per_image_scores.csv':score_rows,
                'ranking_summary.csv':rankings,'gate_statistics.csv':ip.read_csv((pred/'gate_statistics.csv').read_bytes()),
                'distance_audit_summary.csv':audit,'defect_instances.csv':instance,'peak_positions.csv':peaks,
                'mechanism_per_image.csv':mechanism,'mechanism_summary.csv':mechanism_summary(mechanism),
                'instance_margin_summary.csv':spatial_summary(instance),'map_manifest.csv':map_records}
        for name,rows in tables.items(): ip.write_csv(out/name,rows)
        from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.evaluation import peak_rss
        ip.write_json(out/'timing.json',dict(**timings,whole_seconds=time.perf_counter()-started,
                                           cpu_workers=workers,process_peak_rss_bytes=peak_rss(),
                                           pixel_sort_budget_bytes=memory_budget_bytes,pixel_count=total,
                                           environment=ip.environment(),
                                           timing_scope='offline_frozen_cache_CPU_only',historical_costs_included=False))
        report(out,metrics,rankings,mechanism_summary(mechanism),spatial_summary(instance))
        files=[p.name for p in out.iterdir() if p.is_file() and p.name!='status.json']
        manifest=dict(identity=ctx['identity'],prediction_manifest_sha256=ip.sha(pred/'prediction_manifest.json'),
                      images=297,prediction_directory=str(pred),files={n:ip.sha(out/n) for n in files},
                      baseline_verified=True,GT_used_only_after_prediction_frozen=True)
        ip.write_json(out/'analysis_manifest.json',manifest)
        ip.write_json(out/'status.json',dict(tool=ip.TOOL,stage='evaluate',status='cpu_evaluation_complete_aupro_pending',
                                            identity=ctx['identity'],analyzed_images=297,baseline_verified=True,
                                            all_metrics_complete=False,pending_metrics=['new_gate_AUPRO@0.3'],
                                            image_metrics_complete=True,pixel_AUROC_AP_complete=True,spatial_complete=True,
                                            numerical_failed_checks=0,GT_used_only_after_prediction_frozen=True,
                                            analysis_manifest_sha256=ip.sha(out/'analysis_manifest.json'),torch_imported=False))
    finally:
        # Only delete this new run's known scratch files; never any protected cache.
        labels._mmap.close()
        for a in pixel.values(): a._mmap.close()
        for p in scratch.iterdir():
            if p.name not in {'labels.bin',P+'.bin',G+'.bin'}: raise ValueError('unexpected scratch file; preserving directory')
            p.unlink()
        scratch.rmdir();gc.collect()
    return out


def report(out,metrics,rankings,mechanism,spatial):
    lines=['# AeBAD-S/background: neighborhood-constrained Target segment',
           '', 'Development domain only; no cross-domain conclusion. Prediction uses no GT, labels or RGB.',
           'All four image scores use the SAME frozen CH1 state. Historical CURRENT scores are reproduction checks only.',
           'New CPU pixel AUROC/AP use every original evaluation pixel. New AUPRO is pending the explicit original CUDA/ADEval entry.',
           'Union/SEGMENT pixel metrics are historical references, not new evaluations. Scores/maps retain negative responses.',
           '', '| Branch | Image AUROC | Image AP | Pixel AUROC | Pixel AP | AUPRO |',
           '|---|---:|---:|---:|---:|---:|']
    for r in metrics:
        if r['normal_group']=='all85':
            lines.append('| '+r['branch']+' | '+' | '.join('pending' if r.get(k) is None else f'{r[k]:.9f}' for k in ('image_AUROC','image_AP','pixel_AUROC','pixel_AP','AUPRO'))+' |')
    lines+=['','Differences below are branch minus reference; pair counts concern ranking, not deployed threshold errors.']
    for r in rankings:
        if r['normal_group']=='all85':
            lines.append(f"- {r['branch']} vs {r['reference']}: weighted pair-error delta {r['delta_pair_errors']:+g}; improved {r['improved_pairs']}, worsened {r['worsened_pairs']}.")
    lines+=['','R_N/R_A: relative to complete Union, original SEGMENT (Source scale + 1e-8), image equal within layer then four-layer equal. Signed values are preserved.']
    for r in mechanism:
        if r['layer']=='MEAN4': lines.append(f"- {r['branch']}: R_N={r['R_N']}, R_A={r['R_A']}, G={r['G']}; normal expansion retained={r['normal_all_patch_amount_retained_image_layer_equal']}, defect expansion retained={r['anomaly_GT_patch_amount_retained_image_layer_equal']} (NA denominators excluded and counted in CSV).")
    lines+=['','Own-region peak margin and original Union fixed-peak responses are separate in defect_instances.csv. Original small-three mapped-disappearance/empty-ring states are retained.']
    for r in spatial:
        if r['group']=='all' and r['definition']=='margin': lines.append(f"- Margin {r['branch']} vs {r['reference']}: improved={r['improved']}, worsened={r['worsened']}, unavailable={r['not_computable']}.")
    gates=ip.read_csv((out/'gate_statistics.csv').read_bytes())
    lines+=['',f"Mean image-layer acceptance: pair={np.mean([float(r['pair_accept_ratio']) for r in gates]):.6f}, geometry={np.mean([float(r['geom_accept_ratio']) for r in gates]):.6f}.",
            'If acceptance is near zero, similarity to Union is a fallback result, not evidence of effective defect preservation.',
            'Distance containment does not imply monotone AUROC, AP or margin. Pair-vs-geometry comparisons isolate the added local spatial test in this frozen configuration; consistency is not proof of physical correspondence.',
            'See metric_comparisons.csv for metric directions, ranking_summary.csv for fixed-ten/other-normal groups, and instance_margin_summary.csv for margin-loss changes.']
    (out/'objective_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def verified_analysis(ctx,prediction_dir,directory):
    pred,_=verified_predictions(ctx,prediction_dir);root=Path(directory).resolve()
    status=ip.json_file(root/'status.json');manifest=ip.json_file(root/'analysis_manifest.json')
    if (status.get('status')!='cpu_evaluation_complete_aupro_pending' or status.get('numerical_failed_checks')!=0
            or status.get('analysis_manifest_sha256')!=ip.sha(root/'analysis_manifest.json')
            or manifest['identity']!=ctx['identity'] or manifest['images']!=297
            or manifest['prediction_manifest_sha256']!=ip.sha(pred/'prediction_manifest.json')):
        raise ValueError('CPU analysis missing, modified or incomplete')
    for name,checksum in manifest['files'].items():
        if ip.sha(root/ip.safe(name))!=checksum: raise ValueError('CPU analysis artifact modified: '+name)
    return root,manifest
