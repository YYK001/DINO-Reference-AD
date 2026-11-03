"""Fixed requested work, independently committed pairs, normal calibration then freeze."""
import time
from pathlib import Path
import numpy as np
from . import inputs as ip,runtime as rt,numerics as nm
from .protocol import *
st=ip.st


def observations(a):
    x=ip.assets(a)
    for d in DOMAINS:
        for s in SEEDS:
            for row in x.calibration:yield d,s,'Source',row
            for row in x.samples(d):yield d,s,'query',row


def budget(a,estimate=None):
    # User requested a duration boundary, not RAM/disk admission heuristics.
    committed=sum(st.read_json(p)['seconds'] for p in (a.run_root/'costs').glob('*.json'))
    elapsed=committed+time.perf_counter()-a._started
    pending=0 if estimate is None else estimate.get('known_pending_seconds',0)
    if not a.allow_over_12h and elapsed+pending>43200:
        st.write_json(a.run_root/'budget_pause.json',dict(status='pending_12h_consent',recorded_elapsed_seconds=elapsed,
            known_pending_seconds=pending,unknown_stages=[] if estimate is None else estimate['unknown_stages']))
        raise RuntimeError('estimated/recorded work exceeds 12 hours; retained checkpoints, explicit user consent required')


def prepare(a):
    rms=rt.source_rms(a,lambda *_:budget(a));x=ip.assets(a);entries=[]
    for i,row in enumerate(x.region_rows(),1):
        _,record=rt.region(a,row,rms)
        entries.append(dict(image=ip.image(row),path=str(rt.region_path(a,row)),seal=record['seal']))
        if i%10==0:print(f'[coverage prepare] {i}/{len(x.region_rows())} unique images',flush=True);budget(a)
    result=st.sealed_json(a.run_root/'prepared.json',dict(identity=ip.signature(a),RMS=rms['seal'],regions=entries))
    return dict(status='regions_complete',images=len(entries),seal=result['seal'])


def estimate(a,smoke):
    x=ip.assets(a);region_rows=x.region_rows();present=sum((rt.region_path(a,r)/'manifest.json').exists() for r in region_rows)
    pairs=0
    for d,s,kind,row in observations(a):
        pairs+=sum(not rt.pair_path(a,row,t).is_file() for t in x.supports(d,s))
    region_rate=float(np.mean([r['seconds'] for r in smoke['regions']])) if smoke['regions'] else None
    matching=[r['seconds'] for r in smoke['matching'] if r['all_pairs_new']] if smoke.get('cpu_workers',a.cpu_workers)==a.cpu_workers else []
    rate=float(np.mean(matching))/8 if matching else None
    unknown=['CPU_image_evaluation_and_IO_pack']
    # A reused real score does not measure new LP throughput.
    if rate is None and pairs:unknown.append('new_LP_throughput_no_fresh_complete_template_group_measured')
    missing=sum(ip.feature_ref(a,row) is None for row in region_rows if not (rt.region_path(a,row)/'manifest.json').exists())
    forward_rates=smoke.get('feature_forward_seconds',[])
    forward_rate=float(np.mean(forward_rates)) if forward_rates else None
    if missing and forward_rate is None:unknown.append('missing_full_feature_forwards')
    remaining_regions=len(region_rows)-present
    result=dict(status='partial_real_asset_estimate',plan=plan(),measured_setting=smoke['setting'],
        remaining_unique_region_images=remaining_regions,remaining_LP_pairs=pairs,
        region_seconds=None if region_rate is None else remaining_regions*region_rate,
        matching_distance_LP_IO_seconds=None if rate is None else pairs*rate,
        missing_feature_seconds=None if forward_rate is None else missing*forward_rate,
        known_pending_seconds=(0 if region_rate is None else remaining_regions*region_rate)+(0 if rate is None else pairs*rate)+(0 if forward_rate is None else missing*forward_rate),
        RMS='complete_reused_not_counted_again',missing_feature_images=missing,unknown_stages=unknown,
        unknown_is_not_zero=True,scope='real original features; two full eight-template groups; one setting extrapolation, not all-domain guarantee',
        already_reused_work_excluded=True,cpu_workers=a.cpu_workers)
    st.write_json(a.run_root/'cost_estimate.json',result);return result


def smoke(a):
    path=a.run_root/'smoke.json'
    if path.exists():
        saved=st.sealed_json(path)
        if saved['identity']!=ip.signature(a):raise ValueError('smoke backend/input definition changed')
        return dict(status='smoke_complete',reused=True,estimate=estimate(a,saved))
    x=ip.assets(a);rms=rt.source_rms(a,lambda *_:budget(a));d,s=DOMAINS[0],SEEDS[0]
    queries=[x.calibration[0],x.samples(d)[0]];region_stats=[];matching=[]
    for row in [*x.supports(d,s),*queries]:
        _,r=rt.region(a,row,rms);region_stats.append(dict(image=ip.image(row),seconds=r['metadata']['seconds']))
        budget(a)
    from .regions import distance_matrix
    first,_=rt.region(a,queries[0],rms);template,_=rt.region(a,x.supports(d,s)[0],rms)
    reference=distance_matrix(template['means'],first['means'],rms['RMS'],'cpu')
    actual=distance_matrix(template['means'],first['means'],rms['RMS'],a.device)
    error=float(np.max(abs(reference-actual)))
    if error>CONFIG['distance_backend_check_atol']:raise ValueError('new distance backend differs from direct FP32 CPU reference; retained original tolerances')
    with rt.Coverage(a,rms) as engine:
        for row in queries:
            new=all(not rt.pair_path(a,row,t).exists() for t in x.supports(d,s))
            tick=time.perf_counter();score=engine.score(row,x.supports(d,s))
            matching.append(dict(image=ip.image(row),seconds=time.perf_counter()-tick,all_pairs_new=new,
                objectives=[r['result']['objective'] for r in score['templates']],free=[r['result']['free'] for r in score['templates']]))
            budget(a)
    saved=st.sealed_json(path,dict(identity=ip.signature(a),setting=f'{d}/seed{s}',regions=region_stats,matching=matching,
        actual_features=True,full_256_regions=True,LP_pairs=16,RMS=rms['seal'],GT_used=False,
        feature_forward_seconds=getattr(a,'_forward_rates',[]),distance_backend_max_abs_error=error,
        distance_backend_atol=CONFIG['distance_backend_check_atol'],cpu_workers=a.cpu_workers))
    projection=estimate(a,saved);budget(a,projection)
    return dict(status='smoke_complete',estimate=projection)


def observation(a,d,s,kind,row,engine):
    path=a.run_root/d/f'seed{s}'/'coverage'/(ip.key(row)+'.json')
    x=ip.assets(a);P=x.P(d,s,kind,row)
    coverage=engine.score(row,x.supports(d,s))  # Reused pairs verified; never silently accept a stale observation.
    value=dict(identity=ip.signature(a),kind=kind,image=ip.image(row),domain=d,support_seed=s,
        P=P['P'],P_origin=P['origin'],GT_used=False,**coverage)
    return st.sealed_json(path,value)


def predict(a):
    rms=rt.source_rms(a);x=ip.assets(a);all_records=[];blocked=[];parameters=[]
    # One metadata inventory, then decrement as new pairs commit; no rescanning
    # big payloads or the entire pair tree inside the LP loop.
    remaining=sum(not rt.pair_path(a,row,t).exists() for d,s,_,row in observations(a) for t in x.supports(d,s))
    rates=[]
    def observe(engine,domain,seed,kind,row):
        nonlocal remaining
        count=engine.new_pairs;tick=time.perf_counter();r=observation(a,domain,seed,kind,row,engine)
        made=engine.new_pairs-count
        if made:
            rates.append((time.perf_counter()-tick)/made);remaining-=made
        if rates:
            projection=dict(known_pending_seconds=max(0,remaining)*float(np.mean(rates[-20:])),unknown_stages=['CPU_evaluation_pack'])
            used=sum(st.read_json(p)['seconds'] for p in (a.run_root/'costs').glob('*.json'))+time.perf_counter()-a._started
            if not a.allow_over_12h and used+projection['known_pending_seconds']>43200:
                st.write_json(a.run_root/'budget_pause.json',dict(status='pending_12h_consent',recorded_elapsed_seconds=used,**projection))
                raise RuntimeError('observed throughput projects this run beyond 12h; committed pairs retained, explicit consent required')
        budget(a);return r
    with rt.Coverage(a,rms) as engine:
        for d in DOMAINS:
            for s in SEEDS:
                normals=[]
                for i,row in enumerate(x.calibration,1):
                    normals.append(observe(engine,d,s,'Source',row))
                    if i%10==0:print(f'[coverage calibration] {d}/seed{s} {i}/66',flush=True)
                model=nm.calibration(normals)
                parameter=st.sealed_json(a.run_root/d/f'seed{s}'/'calibration.json',dict(identity=ip.signature(a),
                    population=[ip.image(r) for r in x.calibration],observations=[r['seal'] for r in normals],parameters=model,GT_used=False))
                parameters.append(dict(domain=d,support_seed=s,seal=parameter['seal']))
                for i,row in enumerate(x.samples(d),1):
                    obs=observe(engine,d,s,'query',row)
                    values,invalid=nm.scores(obs['P'],obs['cap'],obs['free'],model)
                    record=st.sealed_json(a.run_root/d/f'seed{s}'/'scores'/(row['sample_id']+'.json'),dict(
                        identity=ip.signature(a),domain=d,support_seed=s,sample_id=row['sample_id'],image=ip.image(row),
                        scores=values,blocked=invalid,calibration=parameter['seal'],observation=obs['seal'],
                        map_hash=x.query[d,s,row['sample_id']]['map_hash'],map_reference='UNION_SOURCE_WEIGHT',GT_used=False))
                    all_records.append(dict(domain=d,support_seed=s,sample_id=row['sample_id'],seal=record['seal']))
                    blocked.extend(dict(domain=d,support_seed=s,branch=b,reason=why) for b,why in invalid.items())
                    budget(a)
                    if i%10==0:print(f'[coverage predict] {d}/seed{s} {i}/{len(x.samples(d))} eight_whole_templates',flush=True)
    frozen=st.sealed_json(a.run_root/'predictions_frozen.json',dict(identity=ip.signature(a),records=all_records,calibrations=parameters,
        pending=sorted({f"{r['domain']}/seed{r['support_seed']}/{r['branch']}" for r in blocked}),GT_used=False))
    st.write_csv(a.run_root/'blocked_branches.csv',list({(r['domain'],r['support_seed'],r['branch']):r for r in blocked}.values()))
    return dict(status='scores_frozen_with_blocked_branches' if blocked else 'all_scores_frozen',records=len(all_records),seal=frozen['seal'])


def evaluate(a):
    from sklearn.metrics import roc_auc_score,average_precision_score
    from .reporting import aggregate
    frozen=st.sealed_json(a.run_root/'predictions_frozen.json')
    if frozen['identity']!=ip.signature(a):raise ValueError('frozen scores use another configuration')
    x=ip.assets(a);index={(r['domain'],r['support_seed'],r['sample_id']):r for r in frozen['records']}
    if len(index)!=1804:raise ValueError('complete prediction population must be frozen before labels are evaluated')
    metrics=[];table=[];normal_table=[];template_table=[]
    for d in DOMAINS:
        for s in SEEDS:
            records=[]
            for row in x.samples(d):
                record=st.sealed_json(a.run_root/d/f'seed{s}'/'scores'/(row['sample_id']+'.json'))
                if record['seal']!=index[d,s,row['sample_id']]['seal'] or record['image']!=ip.image(row):raise ValueError('prediction identity changed')
                records.append(record)
                table.append(dict(domain=d,support_seed=s,sample_id=row['sample_id'],label=row['label'],**record['scores']))
                obs=st.sealed_json(a.run_root/d/f'seed{s}'/'coverage'/(ip.key(row)+'.json'))
                if record['observation']!=obs['seal']:raise ValueError('coverage observation changed')
                for ref in obs['templates']:
                    template_table.append(dict(domain=d,support_seed=s,kind='query',image=row['relative_path'],
                        template=ref['template']['relative_path'],cap=ref['result']['objective'],free=ref['result']['free'],
                        cap_winner=ref['template']==obs['cap_template'],free_winner=ref['template']==obs['free_template'],
                        solver_success=ref['result']['success'],**ref['result']['feasibility']))
            labels=[r['label'] for r in x.samples(d)]
            for branch in BRANCHES:
                if any(branch not in r['scores'] for r in records):continue
                v=[r['scores'][branch] for r in records]
                metrics.append(dict(domain=d,support_seed=s,branch=branch,image_AUROC=float(roc_auc_score(labels,v)),
                    image_AP=float(average_precision_score(labels,v)),images=len(records),localization='inherited_UNION_SOURCE_WEIGHT_unchanged'))
            model=st.sealed_json(a.run_root/d/f'seed{s}'/'calibration.json')['parameters']
            for row in x.calibration:
                obs=st.sealed_json(a.run_root/d/f'seed{s}'/'coverage'/(ip.key(row)+'.json'))
                normal_table.append(dict(domain=d,support_seed=s,image=row['relative_path'],P=obs['P'],cap=obs['cap'],free=obs['free'],
                    cap_template=obs['cap_template']['relative_path'],free_template=obs['free_template']['relative_path']))
                for ref in obs['templates']:
                    template_table.append(dict(domain=d,support_seed=s,kind='Source',image=row['relative_path'],
                        template=ref['template']['relative_path'],cap=ref['result']['objective'],free=ref['result']['free'],
                        cap_winner=ref['template']==obs['cap_template'],free_winner=ref['template']==obs['free_template'],
                        solver_success=ref['result']['success'],**ref['result']['feasibility']))
            # Positive affine P standardization must preserve ranking/metrics.
            if model['P']['status']=='valid':
                P=np.array([r['scores'][BASE] for r in records]);z=(P-model['P']['mean'])/model['P']['std']
                if (abs(roc_auc_score(labels,z)-roc_auc_score(labels,P))>1e-12 or
                    abs(average_precision_score(labels,z)-average_precision_score(labels,P))>1e-12):
                    raise ValueError('positive affine SEGMENT calibration changed ranking')
    st.write_csv(a.run_root/'setting_metrics.csv',metrics);st.write_csv(a.run_root/'metrics.csv',aggregate(metrics))
    st.write_csv(a.run_root/'per_image_scores.csv',table);st.write_csv(a.run_root/'normal_calibration.csv',normal_table)
    st.write_csv(a.run_root/'template_costs.csv',template_table)
    saved=st.sealed_json(a.run_root/'evaluation_complete.json',dict(identity=ip.signature(a),prediction_seal=frozen['seal'],
        files={name:st.sha(a.run_root/name) for name in ('setting_metrics.csv','metrics.csv','per_image_scores.csv','normal_calibration.csv','template_costs.csv')}))
    return dict(status='image_evaluation_complete',cells=len(metrics),new_pixel_jobs=0,new_AUPRO_jobs=0,seal=saved['seal'])
