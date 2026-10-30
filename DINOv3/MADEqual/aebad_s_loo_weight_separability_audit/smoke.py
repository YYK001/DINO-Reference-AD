"""Label-independent first two before freeze; class-paired GT smoke afterwards."""
import math,time
import numpy as np
from . import inputs as ip,prediction as pred,numerics as nm,evaluation as ev
from .inputs import st,GTData
from .protocol import *


def selected(p,d,with_gt=False):
    # Before global freeze, even smoke case selection ignores test labels.
    if not with_gt:return p.samples(d)[:2]
    return [next(r for r in p.samples(d) if r['label']==label) for label in (0,1)]


def run(a,with_gt=False):
    if with_gt:pred.global_freeze(a)
    p=ip.parent(a);results=[];name='GT' if with_gt else 'response';out=a.output_root/'smoke'/name;out.mkdir(parents=True,exist_ok=True)
    for d in DOMAINS:
        for m in SEEDS:
            item=dict(domain=d,memory_seed=m,scope='real_frozen_scalar_response',images=2)
            cases=selected(p,d,with_gt);elapsed=[];layers=[];normal_seconds=[];instance_seconds=[];maps=[];masks=[]
            item['sample_ids']=[r['sample_id'] for r in cases]
            for row in cases:
                tick=time.perf_counter();images,scores,stats,z,pos,ref=pred.load_maps(a,d,m,row);elapsed.append(time.perf_counter()-tick)
                tick=time.perf_counter();lm=nm.per_layer_maps(z,pos,ref['grid'],ref['output_shape']);layers.append(time.perf_counter()-tick)
                if not with_gt or not row['label']:
                    tick=time.perf_counter();nm.normal_tail(images);normal_seconds.append(time.perf_counter()-tick)
                if with_gt:
                    with GTData(a.data) as data:
                        tick=time.perf_counter();original,mask=ev.mask_record(data,row,ref['output_shape'])
                        if row['label']:nm.instance_changes(images,lm,original,{w:p.weights[d,w] for w in WEIGHTS})
                        instance_seconds.append(time.perf_counter()-tick)
                    maps.append(images[crossed(m)]);masks.append(mask)
            item.update(read_three_weighted_maps_seconds_per_image=float(np.mean(elapsed)),four_layer_interpolation_seconds_per_image=float(np.mean(layers)))
            item['normal_top1_seconds_per_image' if with_gt else 'top1_operation_seconds_per_image']=float(np.mean(normal_seconds))
            if not with_gt:item['label_independent_operation_benchmark_only']=True
            if with_gt:
                labels=np.concatenate([x.ravel() for x in masks]);values=np.concatenate([x.ravel() for x in maps]);positive=int(labels.sum(dtype=np.int64));tick=time.perf_counter()
                pixel=ev.backend.exact_pixel_result(out/f'{d}_Z{m}_smoke_pixels.json','smoke_'+ip.signature(a,d,m)['identity'],st.digest([ev.backend.shared.array_sha(x) for x in maps]),labels,values,domain=d,branch='SMOKE_ONLY_'+branch(m,crossed(m)),prediction_sha=st.sealed_json(a.output_root/'responses_frozen.json')['seal'],counts=dict(pixels=len(labels),positive_pixels=positive,negative_pixels=len(labels)-positive))
                item.update(GT_decode_and_anomaly_instance_seconds=instance_seconds[1],sample_exact_pixel_seconds=time.perf_counter()-tick,sample_pixels=len(labels),formal_metrics=False)
            results.append(item)
    st.write_json(out/'timings.json',dict(scope='fixed_actual_assets_not_constructed_inputs',with_GT=with_gt,rows=results))
    if with_gt:
        components=dict(no_GT_prediction_seconds=0.,spatial_and_pair_analysis_seconds=0.,new_exact_CPU_seconds=0.,new_AUPRO_seconds_from_parent_reference=0.)
        for r in results:
            d=r['domain'];m=r['memory_seed'];n,normal,anomaly=COUNTS[d]
            components['no_GT_prediction_seconds']+=n*r['read_three_weighted_maps_seconds_per_image']
            components['spatial_and_pair_analysis_seconds']+=n*(r['read_three_weighted_maps_seconds_per_image']+r['four_layer_interpolation_seconds_per_image'])+normal*r['normal_top1_seconds_per_image']+anomaly*r['GT_decode_and_anomaly_instance_seconds']
            full_pixels=sum((s['image_height']//4)*(s['image_width']//4) for s in p.samples(d));ratio=full_pixels/r['sample_pixels']
            components['new_exact_CPU_seconds']+=n*r['read_three_weighted_maps_seconds_per_image']+r['sample_exact_pixel_seconds']*ratio*math.log2(max(full_pixels,2))/math.log2(max(r['sample_pixels'],2))
            old=[st.read_json(p.root/d/f'seed{m}'/'aupro'/(b+'.json'))['seconds'] for b in (SOURCE,TARGET)]
            components['new_AUPRO_seconds_from_parent_reference']+=float(np.mean(old))
        estimated=sum(components.values());estimate=dict(status='fixed_real_asset_smoke_extrapolation',full_seconds=estimated,planning_seconds_range=[estimated*.5,estimated*3],components=components,
            evidence='actual first normal/anomaly x six memory settings; exact sorting NlogN extrapolation; prior same-population Source/Target AUPRO job durations include their replay, no second addition',
            limitations=['not full-run measurement','instance heterogeneity unmeasured','new crossed AUPRO backend throughput not measured','small pair decomposition and summary overhead not separately measured; not assumed zero, included in planning range'],
            existing_predictions_reused=(a.output_root/'responses_frozen.json').exists(),old_pixel_or_AUPRO_recomputed=False,new_feature_forwards=0,new_Exact_searches=0,new_weight_fit_calls=0,new_training=0,over_12h_requires_approval=True)
        if estimate['existing_predictions_reused']:estimate['remaining_seconds']=estimated-components['no_GT_prediction_seconds']
        st.write_json(a.output_root/'time_estimate.json',estimate);pred.budget(a,6,6)
    return dict(status='fixed_asset_smoke_complete',GT_used=with_gt,rows=results)
