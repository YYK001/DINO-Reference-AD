import time
import numpy as np
from . import inputs as ip,runtime as rt
from .inputs import st
from .protocol import *
from DINOv3.MADEqual.aebad_s_target_loo_layer_weight_p0 import numerics as math
from DINOv3.MADEqual.aebad_s_target_loo_layer_weight_p0.composition import image_scores
from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.evaluation import shared


def guard(a,stage,done,total):
    previous=sum(st.read_json(p)['seconds'] for p in a.run_root.glob('*/seed*/costs/*.json'))
    elapsed=time.perf_counter()-a._started
    remaining=0
    for d,x in ip.plan(a)['domains'].items():
        if not (a.run_root/d/'seed43/predictions_frozen.json').exists():remaining+=x['common_images']
    projected=previous+elapsed*total/max(1,done)+remaining*7+1800
    if not a.allow_over_12h and (previous+elapsed>43200 or (done>=min(5,total) and projected>43200)):
        st.write_json(a.run_root/'budget_pause.json',dict(**ip.signature(a),stage=stage,done=done,total=total,projected_seconds=projected,status='paused_12h_budget'))
        raise RuntimeError('estimated full experiment >12h; paused at committed unit; explicit approval needed for --allow-over-12h')


def prepare_normal_stats(a):
    sig=ip.signature(a);dest=ip.setting(a)/'normal_statistics.json'
    if dest.exists():
        r=st.sealed_json(dest)
        if any(r.get(k)!=v for k,v in sig.items()):raise ValueError('normal statistics belong to another group')
        return r
    if a.support_seed==42:
        provider=st.sealed_json(a.weight42_runs[a.domain]/'normal_statistics.json')
        return st.sealed_json(dest,dict(**sig,records=provider['records'],origin=dict(path=str(a.weight42_runs[a.domain]),seal=provider['seal'],read_only=True),weights_refitted=False))
    p,supports,_=ip.context(a);source=st.sealed_json(a.weight42_runs['background']/'normal_statistics.json')
    records=[r for r in source['records'] if r['kind']=='Source'];refs=None
    for i,row in enumerate(supports):
        path=ip.cache(a)/'LOO'/f'{i:02d}';deps=dict(**sig,Source_seed=42,source=p.assets['references']['source']['seal'],query=row,reference_images=[x for j,x in enumerate(supports) if i!=j],extraction=p.extraction)
        if not (path/'manifest.json').exists():
            if not a.supplement_retrieval:raise FileNotFoundError('seed43 support LOO missing; explicit retrieval required')
            if refs is None:refs=rt.References(a)
            q,m,origin=rt.features(a,row,persist=True);nn=refs.distances(q,heldout=i)
            raw=np.stack([np.minimum(nn[l]['ds'],nn[l]['dt'].min(0)) for l in LAYERS])
            st.put(path,dict(raw=raw,positions=m['positions']),dict(deps,grid=m['grid'],output_shape=m['output_shape'],feature_origin=origin))
        x,r=st.get(path)
        if any(r['metadata'].get(k)!=v for k,v in deps.items()):raise ValueError('LOO reference support set differs')
        meta=r['metadata'];stats=math.normal_statistics(x['raw'],x['positions'],meta['grid'],meta['output_shape'],p.mad['patch'])
        records.append(dict(kind='Target',index=i,image=row['relative_path'],support_index=i+1,statistics=stats.tolist(),distance_seal=r['seal']))
        print(f'[two K8 LOO] {a.domain}/seed43 {i+1}/8 whole_image_excluded',flush=True);guard(a,'normal',i+1,8)
    return st.sealed_json(dest,dict(**sig,records=records,Source_origin=dict(path=str(a.weight42_runs['background']),seal=source['seal']),GT_used=False))


def fit_weights(a):
    normal=st.sealed_json(ip.setting(a)/'normal_statistics.json');sig=ip.signature(a)
    if any(normal.get(k)!=v for k,v in sig.items()):raise ValueError('normal statistics group changed')
    rows=[];concentration=[]
    if a.support_seed==42:
        frozen=st.sealed_json(a.weight42_runs[a.domain]/'weights.json');w=frozen['weights'];origin=dict(path=str(a.weight42_runs[a.domain]),seal=frozen['seal'],read_only=True,refitted=False)
    else:
        s=np.asarray([r['statistics'] for r in normal['records'] if r['kind']=='Source']);t=np.asarray([r['statistics'] for r in normal['records'] if r['kind']=='Target'])
        anchor,ss,tt=math.moments(s,t);w={b:math.weights(ss,tt,g)[0].tolist() for b,g in zip(NEW,(0.,.5))};origin=dict(kind='this_group_Target_LOO_only',Source_statistics_shared=True)
    s=np.asarray([r['statistics'] for r in normal['records'] if r['kind']=='Source']);t=np.asarray([r['statistics'] for r in normal['records'] if r['kind']=='Target']);anchor,ss,tt=math.moments(s,t)
    for b,gamma in zip(NEW,(0.,.5)):
        calculated,scale,floor=math.weights(ss,tt,gamma)
        if not np.array_equal(calculated,np.asarray(w[b])):raise ValueError('frozen/copied weights differ from unchanged formula')
        for l,layer in enumerate(LAYERS):
            for j,name in enumerate(('u','p','q')):rows.append(dict(domain=a.domain,support_seed=a.support_seed,branch=b,layer=layer,statistic=name,Source_scale=np.sqrt(ss[l,j]),Target_LOO_scale=np.sqrt(tt[l,j]),mixed_scale=scale[l,j],floor_triggered=bool(floor[l,j]),weight=calculated[l,j]))
    if w[SOURCE]!=ip.weight42(a,'background')['weights'][SOURCE]:raise ValueError('Source gamma0 weights changed across domain/support_seed')
    sq=np.maximum(t[:,:,0]-anchor[:,0],0)**2
    for j,l in enumerate(LAYERS):
        total=float(sq[:,j].sum());concentration.append(dict(domain=a.domain,support_seed=a.support_seed,layer=l,u_positive_squared_sum=total,max_single_image_fraction=float(sq[:,j].max()/total) if total else None,status='valid' if total else 'NA_no_positive_deviation',interpretation='statistic_concentration_not_performance_contribution'))
    st.write_csv(ip.setting(a)/'weights.csv',rows);st.write_csv(ip.setting(a)/'u_concentration.csv',concentration)
    return st.sealed_json(ip.setting(a)/'weights.json',dict(**sig,weights=w,normal_statistics_seal=normal['seal'],origin=origin,GT_used=False))


def predict(a):
    sig=ip.signature(a);w=rt.weights(a);_,_,samples=ip.context(a);records=[];refs=None
    if a.support_seed==42:
        parent_maps={(r['sample_id'],r['branch']):r for r in ip.parent(a).table('map_manifest.csv') if r['branch'] in (U,S)}
        weighted_maps={(r['sample_id'],r['branch']):r for r in ip.layer_inputs.old.read_csv(a.weight42_runs[a.domain]/'evaluation/map_manifest.csv')}
        old_scores={(r['sample_id'],r['branch']):r for r in ip.layer_inputs.old.read_csv(a.weight42_runs[a.domain]/'evaluation/per_image_scores.csv')}
    for i,row in enumerate(samples,1):
        dest=ip.record_path(a,row)
        if not dest.exists():
            if a.support_seed==43 and not (ip.cache(a)/'baselines'/row['sample_id']/'manifest.json').exists() and not ip.legacy_ref(a,row):
                if not a.supplement_retrieval:raise FileNotFoundError('seed43 query retrieval explicitly required')
                if refs is None:refs=rt.References(a)
            images,scores,m=rt.maps(a,row,allow_compute=True,refs=refs)
            hashes={b:shared.array_sha(image) for b,image in images.items()}
            if a.support_seed==42:
                for b in MAPS:
                    expected=(parent_maps if b in (U,S) else weighted_maps)[row['sample_id'],b]
                    if hashes[b]!=expected['map_float32_sha256']:raise ValueError('seed42 retained map differs from historical map; no tolerance change')
                    sc=m['scores'][b] if b in (U,S) else old_scores[row['sample_id'],b]
                    if any(abs(v-float(sc[k]))>1e-5 for k,v in scores[b].items()):raise ValueError('seed42 retained scores differ')
            r=st.sealed_json(dest,dict(**sig,image=row,weights_seal=w['seal'],baseline_origin=m['origin'],map_hashes=hashes,scores=scores,combination_scores=image_scores(scores[S]),GT_used=False))
        r=st.sealed_json(dest)
        if any(r.get(k)!=v for k,v in sig.items()) or r['image']!=row or r['weights_seal']!=w['seal']:raise ValueError('cached prediction group/image/weights differ')
        records.append(dict(sample_id=row['sample_id'],seal=r['seal']))
        if i%10==0 or i==len(samples):print(f'[two K8 predict] {a.domain}/seed{a.support_seed} {i}/{len(samples)} four_maps_one_query',flush=True)
        guard(a,'predict',i,len(samples))
    return st.sealed_json(ip.setting(a)/'predictions_frozen.json',dict(**sig,weights_seal=w['seal'],records=records,common_manifest=st.sealed_json(a.run_root/'common_manifest.json')['seal'],images=len(samples),GT_used=False))


def verified(a):
    r=st.sealed_json(ip.setting(a)/'predictions_frozen.json');sig=ip.signature(a);_,_,samples=ip.context(a)
    if any(r.get(k)!=v for k,v in sig.items()) or r['weights_seal']!=rt.weights(a)['seal'] or [x['sample_id'] for x in r['records']]!=[s['sample_id'] for s in samples]:raise ValueError('incomplete or wrong-group prediction freeze')
    for row,rec in zip(samples,r['records']):
        pred=st.sealed_json(ip.record_path(a,row))
        if pred['seal']!=rec['seal'] or any(pred.get(k)!=v for k,v in sig.items()):raise ValueError('frozen per-image prediction changed')
    return r
