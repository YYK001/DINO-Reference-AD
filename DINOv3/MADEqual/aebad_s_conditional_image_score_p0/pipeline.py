"""Normal-only fitting, label-free score freeze, then image-only evaluation."""
import time
from pathlib import Path
import numpy as np
from . import inputs as ip,numerics as nm,runtime as rt
from .inputs import st
from .protocol import *
from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import image_metrics


def units(a,kind=None):
    normal=list(ip.normal_units(a))
    # Source feature/Source search is reused across the six different K8 settings.
    normal.sort(key=lambda x:(x[2]!='Source',x[3]['relative_path'],x[0],x[1]))
    query=[(d,s,'query',r,None) for d in DOMAINS for r in ip.samples(a,d) for s in SEEDS]
    return normal if kind=='normal' else query if kind=='query' else normal+query


def costs(a):return sum(float(st.read_json(p)['seconds']) for p in (a.run_root/'costs').glob('*.json'))


def budget(a):
    elapsed=costs(a)+time.perf_counter()-a._started;known=0.;unknown=[]
    path=a.run_root/'smoke.json';measured=st.sealed_json(path) if path.exists() else {}
    # Older test-only estimates have no smoke commit; production admission uses
    # the sealed smoke, not an editable display JSON.
    if not measured and (a.run_root/'time_estimate.json').exists():
        display=st.read_json(a.run_root/'time_estimate.json')
        if display.get('status')=='asset_smoke_complete':raise ValueError('unsealed smoke estimate cannot authorize full work')
    rates={(r['domain'],r['support_seed'],r['kind'],r['execution_path']):r['seconds'] for r in measured.get('samples',[])}
    rates.update({k:float(np.mean(v)) for k,v in getattr(a,'_live_rates',{}).items()})
    if measured and measured.get('identity')!=ip.signature(a):raise ValueError('smoke identity differs')
    if not hasattr(a,'_pending_work'):
        work=[]
        for d,s,kind,row,j in units(a):
            c=ip.contract(a,d,s,kind,row,j);dest=ip.record_path(a,c)
            if dest.exists():continue
            mode='small_BP_import' if ip.export_record(a,'statistics_provider',c) else 'native_response_statistics' if ip.response(a,c) else 'cached_geometry' if ip.retrieval(a,c) else 'fresh_Exact_SEGMENT'
            work.append((dest,(d,s,kind,mode)))
        a._pending_work=work
    for dest,key in a._pending_work:
        if dest.exists():continue
        rate=rates.get(key)
        if rate is None:unknown.append(key)
        else:known+=rate
    result=dict(elapsed_seconds=elapsed,known_pending_compute_seconds=known,unknown_categories=[list(k) for k in sorted(set(unknown))],
        other_unknown_stages=['GT_area_descriptions','image_evaluation_plot_pack'],unknown_is_not_zero=True)
    if not a.allow_over_12h and elapsed+known>43200:
        st.write_json(a.run_root/'budget_pause.json',dict(status='pending_12h_consent',elapsed_seconds=elapsed,
            known_pending_compute_seconds=known,unknown_categories=sorted(set(unknown))))
        raise RuntimeError('known elapsed/projected work exceeds12h; checkpoint retained, fresh consent required')
    return result


def run_observations(a,which):
    engine=rt.Engine(a);records=[]
    for i,(d,s,kind,row,j) in enumerate(units(a,which),1):
        existed=ip.record_path(a,ip.contract(a,d,s,kind,row,j)).exists()
        r=rt.observe(a,d,s,kind,row,j,engine)
        if not existed and 'execution_path' in r:
            if not hasattr(a,'_live_rates'):a._live_rates={}
            a._live_rates.setdefault((d,s,kind,r['execution_path']),[]).append(r['seconds'])
        record=dict(domain=d,support_seed=s,kind=kind,image=ip.image(row),path=str(ip.record_path(a,r['contract'])),seal=r['seal'])
        if which=='query':
            old=ip.parent(a).record(d,s,row);expected=float(old['scores']['SEGMENT_K8']['CH1']);error=abs(r['P']-expected)
            if not np.isfinite(error) or error>1e-5:
                st.write_csv(a.run_root/'failed_numerical_checks.csv',[dict(domain=d,support_seed=s,sample_id=row['sample_id'],
                    check='SEGMENT_CH1',max_abs_error=error,atol=1e-5,verified=False)])
                raise ValueError('original SEGMENT CH1 reconstruction differs; no relaxed tolerance')
            map_ref=ip.parent(a).maps[d,s][row['sample_id'],MAP]
            if old['map_hashes'][MAP]!=map_ref['map_float32_sha256']:raise ValueError('parent localization identity differs')
            record.update(sample_id=row['sample_id'],B=r['B'],P=expected,P_reconstructed=r['P'],
                CH1_abs_error=error,atol=1e-5,map_hash=old['map_hashes'][MAP],parent_prediction=old['seal'])
        records.append(record)
        if i%20==0:print(f'[conditional {which}] {i}/{len(units(a,which))}',flush=True)
        # Project once per image-sized chunk, not once per patch/inner kernel.
        if i%20==0:budget(a)
    name='normal_statistics' if which=='normal' else 'query_statistics'
    result=st.sealed_json(a.run_root/(name+'.json'),dict(identity=ip.signature(a),records=records,GT_used=False))
    if which=='normal':
        table=[]
        for ref in records:
            r=st.sealed_json(Path(ref['path']));c=r['contract']
            table.append(dict(domain=ref['domain'],support_seed=ref['support_seed'],kind=ref['kind'],image=ref['image']['relative_path'],
                B=r['B'],P=r['P'],K=c['K'],heldout=c['heldout'],source_seal=c['source'],extraction=c['extraction'],
                reference_images='|'.join(v['relative_path'] for v in c['references']),statistics_seal=r['seal'],origin=str(r['origin'])))
        st.write_csv(a.run_root/'normal_statistics.csv',table)
    else:st.write_csv(a.run_root/'numerical_verification.csv',[{k:r[k] for k in ('domain','support_seed','sample_id','CH1_abs_error','atol')}|dict(verified=True) for r in records])
    budget(a)
    return dict(status=name+'_complete',records=len(records),seal=result['seal'])


def normal_data(a):
    r=st.sealed_json(a.run_root/'normal_statistics.json')
    if r['identity']!=ip.signature(a) or r['GT_used']:raise ValueError('normal preparation identity changed')
    expected={(d,s,k,ip.image(row)['relative_path']):ip.contract(a,d,s,k,row,j) for d,s,k,row,j in ip.normal_units(a)}
    records={}
    for ref in r['records']:
        obs=st.sealed_json(Path(ref['path']));key=(ref['domain'],ref['support_seed'],ref['kind'],ref['image']['relative_path'])
        if key not in expected or obs['seal']!=ref['seal'] or obs['contract']!=expected[key] or obs['GT_used']:raise ValueError('wrong Source K8/Target K7 normal population')
        nm.validate_observation(obs,ip.parent(a).inputs['mad'])
        if key in records:raise ValueError('duplicate normal observation')
        records[key]=obs
    if set(records)!=set(expected):raise ValueError('incomplete 396 Source/48 Target normal statistics')
    return r,records


def fit(a):
    normal,records=normal_data(a);models={};influence=[];table=[]
    for d in DOMAINS:
        for s in SEEDS:
            src=[records[d,s,'Source',r['relative_path']] for r in ip.parent(a).inputs['Source_calibration']]
            tgt=[records[d,s,'Target',r['relative_path']] for r in ip.supports(a,d,s)]
            bs,ps,bt,pt=([r[k] for r in pop] for pop,k in ((src,'B'),(src,'P'),(tgt,'B'),(tgt,'P')))
            model=nm.fit(bs,ps,bt,pt);models[f'{d}/{s}']=model
            table.append(dict(domain=d,support_seed=s,**model))
            influence.extend(dict(domain=d,support_seed=s,full_beta_ST=model['beta_ST'],beta_ST_change=r['beta_ST']-model['beta_ST'],
                Source_parameters_fixed=True,**r) for r in nm.influence(bt,pt,model))
    result=st.sealed_json(a.run_root/'coefficients.json',dict(identity=ip.signature(a),normal_statistics=normal['seal'],
        models=models,GT_used=False,test_statistics_used=False))
    st.write_csv(a.run_root/'coefficients.csv',table);st.write_csv(a.run_root/'coefficient_influence.csv',influence)
    return dict(status='six_normal_models_frozen',models=6,seal=result['seal'])


def coefficients(a):
    r=st.sealed_json(a.run_root/'coefficients.json');n=st.sealed_json(a.run_root/'normal_statistics.json')
    if r['identity']!=ip.signature(a) or r['normal_statistics']!=n['seal'] or r['GT_used'] or r['test_statistics_used']:
        raise ValueError('normal coefficient identity changed')
    return r


def predict(a):
    coef=coefficients(a);query=st.sealed_json(a.run_root/'query_statistics.json')
    if query['identity']!=ip.signature(a) or query['GT_used']:raise ValueError('query statistic identity changed')
    indexed={(r['domain'],r['support_seed'],r['sample_id']):r for r in query['records']}
    expected={(d,s,r['sample_id']) for d in DOMAINS for s in SEEDS for r in ip.samples(a,d)}
    if len(indexed)!=len(query['records']) or set(indexed)!=expected:raise ValueError('incomplete unique query settings')
    frozen=[];all_scores=[]
    for d in DOMAINS:
        for s in SEEDS:
            records=[indexed[d,s,r['sample_id']] for r in ip.samples(a,d)]
            for r in records:
                obs=st.sealed_json(Path(r['path']))
                if obs['seal']!=r['seal'] or obs['B']!=r['B'] or abs(obs['P']-r['P'])>1e-5:raise ValueError('collected B/P changed')
            values,x=nm.scores([r['B'] for r in records],[r['P'] for r in records],coef['models'][f'{d}/{s}'])
            table=[dict(domain=d,support_seed=s,sample_id=r['sample_id'],B=r['B'],P=r['P'],x=float(x[i]),
                Source_correction=r['P']-float(values[SOURCE][i]),Target_correction=r['P']-float(values[TARGET][i]),
                **{b:float(values[b][i]) for b in BRANCHES},map_reference=MAP,map_hash=r['map_hash'],parent_prediction=r['parent_prediction']) for i,r in enumerate(records)]
            path=ip.unit(a,d,s)/'scores.json'
            r=st.sealed_json(path,dict(identity=ip.signature(a),coefficients=coef['seal'],query_statistics=query['seal'],records=table,GT_used=False))
            frozen.append(dict(domain=d,support_seed=s,path=str(path),seal=r['seal']));all_scores.extend(table)
    freeze=st.sealed_json(a.run_root/'predictions_frozen.json',dict(identity=ip.signature(a),coefficients=coef['seal'],
        records=frozen,instances=1804,unique_images=902,GT_used=False))
    st.write_csv(a.run_root/'per_image_scores.csv',all_scores)
    return dict(status='all_image_scores_frozen',instances=len(all_scores),seal=freeze['seal'],new_maps=0)


def verified(a):
    r=st.sealed_json(a.run_root/'predictions_frozen.json')
    if r['identity']!=ip.signature(a) or r['coefficients']!=coefficients(a)['seal'] or r['GT_used'] or len(r['records'])!=6:
        raise ValueError('prediction freeze incomplete or wrong coefficients')
    result={}
    for ref in r['records']:
        unit=st.sealed_json(Path(ref['path']));d,s=ref['domain'],ref['support_seed']
        if unit['seal']!=ref['seal'] or unit['identity']!=ip.signature(a) or unit['coefficients']!=r['coefficients'] or unit['GT_used']:
            raise ValueError('frozen image scores changed')
        if [x['sample_id'] for x in unit['records']]!=[x['sample_id'] for x in ip.samples(a,d)]:raise ValueError('prediction membership/order')
        if (d,s) in result:raise ValueError('duplicate setting')
        result[d,s]=unit['records']
    if set(result)!={(d,s) for d in DOMAINS for s in SEEDS}:raise ValueError('six settings required')
    return r,result


def evaluate(a):
    frozen,score_sets=verified(a)  # Test labels/GT accessed only AFTER this gate.
    path=a.run_root/'evaluation_complete.json'
    if path.exists():
        old=st.sealed_json(path)
        if old['prediction_seal']!=frozen['seal']:raise ValueError('evaluation inputs changed')
        for name,digest in old['files'].items():
            if st.sha(a.run_root/name)!=digest:raise ValueError('completed small image evaluation changed')
        return dict(status='image_evaluation_complete',reused=True,settings=6)
    metric_rows=[];rank=[];changes=[];labelled=[];area={};missing_area=[];data=None
    try:
        try:data=ip.jp.GTData(a.data);data.__enter__()
        except (OSError,ValueError) as e:missing_area.append(dict(role='GT_dataset',reason=str(e)));data=None
        for (d,s),scores in score_sets.items():
            population=ip.samples(a,d);labels=[row['label'] for row in population]
            loc=ip.parent(a).known_metrics(d,s,'WS')[0]
            original=[r[BASE] for r in scores]
            for b in BRANCHES:
                values=[r[b] for r in scores];m=image_metrics(labels,values)
                if b==BASE:
                    expected=next(r for r in ip.parent(a).settings[d,s]['metrics'] if r['branch']=='SEGMENT_K8' and r['scoring']=='CH1')
                    if any(abs(m[k]-float(expected[k]))>1e-12 for k in m):raise ValueError('original SEGMENT image metrics differ')
                metric_rows.append(dict(domain=d,support_seed=s,branch=b,scoring='CH1_conditional',**m,
                    **{k:float(loc[k]) for k in ('pixel_AUROC','pixel_AP','AUPRO')},localization='inherited_unchanged',
                    localization_reference=MAP,localization_origin=str(ip.parent(a).root/d/f'seed{s}'),image_count=len(scores)))
                rank.append(dict(domain=d,support_seed=s,branch=b,**nm.ranking(labels,original,values)))
                for label,name in ((0,'normal'),(1,'anomaly')):
                    delta=np.asarray([v-p for v,p,y in zip(values,original,labels) if y==label])
                    changes.append(dict(domain=d,support_seed=s,branch=b,group=name,images=len(delta),mean_change=float(delta.mean()),
                        median_change=float(np.median(delta)),min_change=float(delta.min()),max_change=float(delta.max()),
                        decreased_images=int((delta<0).sum()),increased_images=int((delta>0).sum())))
            for row,score in zip(population,scores):
                k=(d,row['sample_id'])
                if k not in area:
                    area[k]=None
                    if not row['label']:area[k]=0.
                    elif data is not None:
                        try:area[k]=float(np.asarray(data.gt(row),bool).mean())
                        except (OSError,ValueError,KeyError) as e:missing_area.append(dict(domain=d,sample_id=row['sample_id'],reason=str(e)))
                    else:missing_area.append(dict(domain=d,sample_id=row['sample_id'],reason='GT area unavailable; core image metrics complete'))
                labelled.append(dict(score,label=row['label'],GT_area_fraction=area[k],area_definition='original_GT_pixel_fraction_not_resized'))
    finally:
        if data is not None:data.__exit__(None,None,None)
    st.write_csv(a.run_root/'setting_metrics.csv',metric_rows);st.write_csv(a.run_root/'ranking_summary.csv',rank)
    st.write_csv(a.run_root/'score_changes.csv',changes);st.write_csv(a.run_root/'evaluation_scores.csv',labelled)
    st.write_csv(a.run_root/'missing_area.csv',missing_area)
    from .reporting import diagnostics
    extra=diagnostics(a,labelled)
    files=['setting_metrics.csv','ranking_summary.csv','score_changes.csv','evaluation_scores.csv','missing_area.csv',*extra]
    st.sealed_json(path,dict(identity=ip.signature(a),prediction_seal=frozen['seal'],settings=6,image_cells=18,
        new_pixel_jobs=0,new_AUPRO_jobs=0,files={n:st.sha(a.run_root/n) for n in files}))
    return dict(status='image_evaluation_complete',settings=6,image_cells=18,new_AUPRO_jobs=0)


def smoke(a):
    destination=a.run_root/'smoke.json'
    if destination.exists():
        r=st.sealed_json(destination)
        if r['identity']!=ip.signature(a):raise ValueError('smoke belongs to other inputs/backend')
        return dict(status='asset_smoke_complete',reused=True,estimate=budget(a))
    engine=rt.Engine(a);samples=[]
    for d in DOMAINS:
        for s in SEEDS:
            for kind,row,j in (('Source',ip.parent(a).inputs['Source_calibration'][0],None),
                               ('Target',ip.supports(a,d,s)[0],0),('query',ip.samples(a,d)[0],None)):
                tick=time.perf_counter();r=rt.observe(a,d,s,kind,row,j,engine)
                if kind=='query' and abs(r['P']-ip.parent(a).record(d,s,row)['scores']['SEGMENT_K8']['CH1'])>1e-5:
                    raise ValueError('smoke SEGMENT CH1 compatibility failed; unchanged1e-5')
                samples.append(dict(domain=d,support_seed=s,kind=kind,seconds=r['seconds'],execution_path=r['execution_path'],
                    scope='committed_real_asset_complete_grid_path',smoke_read_seconds=time.perf_counter()-tick))
    r=st.sealed_json(destination,dict(identity=ip.signature(a),samples=samples,GT_used=False))
    st.write_json(a.run_root/'time_estimate.json',dict(status='asset_smoke_complete',identity=ip.signature(a),samples=samples,
        note='same first complete image reused; Source feature/search shared across settings; cold per-setting rates are approximate',
        unknown_stages=['GT_area_descriptions','image_evaluation_plot_pack'],unknown_is_not_zero=True,plan=plan()))
    estimate=budget(a)
    st.write_json(a.run_root/'budget_estimate.json',estimate)
    return dict(status='asset_smoke_complete',samples=18,seal=r['seal'],estimate=estimate)
