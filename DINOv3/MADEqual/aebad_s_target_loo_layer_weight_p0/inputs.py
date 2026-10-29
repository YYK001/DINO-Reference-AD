from pathlib import Path
import numpy as np
from DINOv3.MADEqual.aebad_s_blind_context_p0 import inputs as old
from .protocol import CONFIG,LAYERS,COUNTS,config
st=old.st
ROOT=Path(__file__).resolve().parents[3]


def domain(a):return getattr(a,'domain','background')


def count(a):return COUNTS[domain(a)][0]


def tags(a):return {} if domain(a)=='background' else dict(domain=domain(a))


def prediction_path(a,row):
    if row.get('domain',domain(a))!=domain(a):raise ValueError('query belongs to another domain')
    base=a.cache_root/'predictions'
    return base/row['sample_id'] if domain(a)=='background' else base/domain(a)/row['sample_id']


def parent(a):
    if not hasattr(a,'_parent'):
        if a.parent_run is None or a.parent_cache is None:raise FileNotFoundError('supply --parent-run and --parent-cache for one consistent completed SACP parent')
        if domain(a)=='background':
            a._parent=old.Parent(a);a._parent.domain='background'
        else:
            from .parent import Parent
            a._parent=Parent(a)
    if getattr(a._parent,'domain',domain(a))!=domain(a):raise ValueError('parent reader belongs to another domain')
    return a._parent


def identity(a):
    r=st.sealed_json(a.run_root/'inputs.json')
    if r.get('config',{}).get('domain',domain(a))!=domain(a):raise ValueError('run belongs to another domain')
    return r['identity']


def freeze(a):
    p=parent(a)
    value=dict(config=config(domain(a)),parent_identity=p.identity,parent_run=str(a.parent_run),parent_cache=str(a.parent_cache),
        raw_run=str(a.raw_run) if a.raw_run else None,raw_cache=str(a.raw_cache) if a.raw_cache else None,
        data=str(a.data),parent_manifest=p.manifest['seal'],mad=p.mad['seal'],source=p.assets['references']['source']['seal'],
        supports=p.supports,source_calibration=[r for r in p.catalog['source'] if r['calibration']],extraction=p.extraction)
    if domain(a)!='background':
        shared=source_statistics(a)
        value.update(domain=domain(a),source_statistics_provider=dict(path=str(a.source_stats_run),seal=shared['seal']))
    value['identity']=st.digest(value)
    return st.sealed_json(a.run_root/'inputs.json',value)


def units(a):
    p=parent(a)
    if domain(a)!='background' and not getattr(a,'source_stats_run',None):raise FileNotFoundError('new domains require --source-stats-run for read-only Source reuse')
    source=[] if getattr(a,'source_stats_run',None) else [('Source',i,r) for i,r in enumerate(p.catalog['source']) if r['calibration']]
    return source+[('Target',i,r) for i,r in enumerate(p.supports)]


def normal_path(a,kind,i):
    base=a.cache_root/'normal_distances'
    return base/f'{kind}_{i:03d}' if domain(a)=='background' else base/domain(a)/f'{kind}_{i:03d}'


def normal_identity(a,kind,i,row):
    p=parent(a)
    refs=[] if kind=='Source' else [r for j,r in enumerate(p.supports) if j!=i]
    return dict(**tags(a),identity=identity(a),kind=kind,image=row,source=p.assets['references']['source']['seal'],
        reference_images=[r['relative_path'] for r in refs],extraction=p.extraction)


def source_statistics(a):
    """Read-only dependency-based import; never relabel old background artifacts."""
    path=getattr(a,'source_stats_run',None)
    if path is None:raise FileNotFoundError('--source-stats-run must point to completed background run')
    p=parent(a);saved=st.sealed_json(path/'inputs.json');normal=st.sealed_json(path/'normal_statistics.json')
    status=st.read_json(path/'status.json')
    if status['identity']!=saved['identity'] or status['status']!='all_five_metrics_complete' or status['pending']:raise ValueError('completed background statistics provider required')
    if saved['config']!=CONFIG or normal['identity']!=saved['identity']:raise ValueError('shared statistics protocol/identity differs')
    expected=dict(source=p.assets['references']['source']['seal'],source_calibration=[r for r in p.catalog['source'] if r['calibration']],extraction=p.extraction,mad=p.mad['seal'])
    if any(saved.get(k)!=v for k,v in expected.items()):raise ValueError('shared Source bank/list/extraction/MAD differs')
    if normal['quantile_method']!='numpy.quantile_linear':raise ValueError('shared normal statistic implementation differs')
    selected=[r for r in normal['records'] if r['kind']=='Source']
    if [r['image'] for r in selected]!=[r['relative_path'] for r in expected['source_calibration']] or len(selected)!=66:raise ValueError('shared Source order/population differs')
    weights=st.sealed_json(path/'weights.json')
    if weights['identity']!=saved['identity'] or weights['normal_statistics_seal']!=normal['seal']:raise ValueError('shared weights have different normal statistics')
    from .protocol import SOURCE
    return dict(seal=normal['seal'],records=selected,inputs_seal=saved['seal'],weights={SOURCE:weights['weights'][SOURCE]},normal_statistics_path=str(path/'normal_statistics.json'))


def normal_statistics(a):
    r=st.sealed_json(a.run_root/'normal_statistics.json');p=parent(a)
    if r['identity']!=identity(a) or r.get('domain',domain(a))!=domain(a):raise ValueError('normal statistics belong to another run/domain')
    source=[x for x in r['records'] if x['kind']=='Source'];target=[x for x in r['records'] if x['kind']=='Target']
    if len(source)!=66 or len(target)!=8 or [x['image'] for x in source]!=[s['relative_path'] for s in p.catalog['source'] if s['calibration']] or [x['image'] for x in target]!=[s['relative_path'] for s in p.supports] or any(x.get('domain',domain(a))!=domain(a) for x in target):raise ValueError('normal statistic Source/Target population or domain differs')
    return r


def feature_ref(a,row):
    ref=old.raw_reference(a,row)
    if ref is None:raise FileNotFoundError('complete normal feature: '+row['relative_path'])
    old.inspect_raw(ref,row)
    return ref


def preflight(a):
    missing=[];reuse=[];unconfirmed=[]
    try:
        p=parent(a)
        if (a.run_root/'inputs.json').exists():identity(a)
        if getattr(a,'source_stats_run',None):
            shared=source_statistics(a)
            reuse.append(dict(kind='Source',role='normal_statistics_66',path=shared['normal_statistics_path'],seal=shared['seal']))
        tasks=units(a)
        if (a.run_root/'normal_statistics.json').exists():
            complete=normal_statistics(a);reuse.append(dict(kind='Source_and_Target',role='complete_normal_statistics',path=str(a.run_root/'normal_statistics.json'),seal=complete['seal']));tasks=[]
        for kind,i,row in tasks:
            path=normal_path(a,kind,i)
            if (path/'manifest.json').exists():
                _,r=st.get(path,('raw','positions'))
                if any(r['metadata'].get(k)!=v for k,v in normal_identity(a,kind,i,row).items()):raise ValueError('normal distance provenance differs')
                reuse.append(dict(kind=kind,path=str(path)))
            else:
                missing.append(dict(kind=kind,role='normal_distance',sample=row['relative_path'],path=str(path)))
                try:ref=feature_ref(a,row);reuse.append(dict(kind=kind,role='normal_feature',path=ref['path']))
                except FileNotFoundError as e:missing.append(dict(kind=kind,role='normal_feature',sample=row['relative_path'],path=str(e)))
        for row in p.samples:
            try:p.prediction(row,False)
            except FileNotFoundError:missing.append(dict(kind='evaluation',role='four_layer_parent_prediction',sample=row['sample_id'],path=str(p.cache/'predictions'/domain(a)/row['sample_id'])))
        bank=Path(p.assets['references']['source']['path'])
        if any(r.get('role')=='normal_distance' for r in missing) and not (bank/'manifest.json').exists():missing.append(dict(kind='Source',role='bank_for_supplement_only',path=str(bank)))
        if not a.data.exists():missing.append(dict(kind='evaluation',role='GT',path=str(a.data)))
        unconfirmed.append(dict(kind='legacy_assets',reason='old Source NPZ / evaluation leave-one-support-out / scalar LOO scores are not admitted without this parent bank/extraction/K7 image provenance'))
        state='inputs_available' if not missing else 'missing_inputs'
    except (OSError,ValueError,KeyError) as e:state='blocked';missing.append(dict(role='parent_metadata',path=str(e)))
    result=dict(status=state,domain=domain(a),missing=missing,reusable=reuse,unconfirmed=unconfirmed,config=config(domain(a)),
        estimate=dict(total_hours_range=[.15,1.5],conditional_on='existing original bank, 74 normal full features, 297 parent layer predictions and GT; no extraction or bank reconstruction',
        normal_retrieval='74 images x four layers, Source-only 66 and whole-image K7 LOO 8; 0.05-0.5h planning range, not measured',
        missing_feature_supplement='unknown until original host throughput and missing image count are known; explicit extract flag only',
        fitting_and_replay='0.02-0.2h',CPU_and_two_AUPRO='0.03-0.3h',remaining_IO='0.05-0.5h',
        evidence='same workstation prior three-map evaluation 72.5s plus AUPRO47.6s; no throughput measurement of this new replay; >12h pauses',
        unknown_if_assets_missing=True),torch_imported=__import__('sys').modules.get('torch') is not None)
    if domain(a)!='background':result['estimate']=dict(total_hours_range=[.15,1.5],scope='one new domain; two domains 0.3-3h planning range; background read-only',
        tasks=dict(shared_Source_statistics=66,new_Target_LOO_queries=8,evaluation_images=count(a),new_localization_maps=2,new_AUPRO_jobs=2),
        evidence='background successful stages302.83s (normal retrieval184.49s, replay34.15s, CPU60.07s, AUPRO23.90s); not a new-domain throughput measurement',
        missing_feature_forward='unknown, explicit --extract-missing only; count missing normal_feature rows',
        missing_evaluation_layers='unknown; list exact images; locate compatible parent cache before replay, no fused-map inversion or automatic reconstruction',
        normal_retrieval='only 8 Target full-image LOO; Source66 imported with dependency audit',
        missing_bank='blocked, no automatic Source rebuild',duration_budget_hours=12)
    a.run_root.mkdir(parents=True,exist_ok=True)
    st.write_json(a.run_root/'preflight.json',result);st.write_csv(a.run_root/'missing_inputs.csv',missing)
    st.write_csv(a.run_root/'asset_inventory.csv',reuse+unconfirmed)
    return result
