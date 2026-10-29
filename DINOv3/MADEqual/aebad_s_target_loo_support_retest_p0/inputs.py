"""Small metadata inventory and immutable common-population protocol; no Torch."""
from pathlib import Path
import numpy as np
from DINOv3.MADEqual.aebad_s_target_loo_layer_weight_p0 import inputs as layer_inputs
from DINOv3.MADEqual.aebad_s_target_loo_layer_weight_p0.parent import Parent
from DINOv3.MADEqual.aebad_s_target_loo_layer_weight_p0.aggregation import completed,source_contract
from .protocol import *
from .selection import select
st=layer_inputs.st
ROOT=layer_inputs.ROOT


def parent(a,domain=None):
    d=domain or a.domain
    if a.parent_run is None or a.parent_cache is None:raise FileNotFoundError('one current Source/seed42 SACP parent run/cache is required')
    if not hasattr(a,'_parents'):a._parents={}
    if d not in a._parents:
        from types import SimpleNamespace
        a._parents[d]=Parent(SimpleNamespace(domain=d,parent_run=a.parent_run,parent_cache=a.parent_cache))
    return a._parents[d]


def weight42(a,domain=None):
    d=domain or a.domain
    if not hasattr(a,'_weight42'):a._weight42={}
    if d not in a._weight42:
        r=completed(a.weight42_runs[d],d);p=parent(a,d)
        expected=dict(source=p.assets['references']['source']['seal'],source_calibration=[x for x in p.catalog['source'] if x['calibration']],extraction=p.extraction,mad=p.mad['seal'])
        if source_contract(r['inputs'])!=expected or r['inputs']['supports']!=p.supports:raise ValueError('seed42 weighting provider has incompatible Source/supports')
        a._weight42[d]=r
    return a._weight42[d]


def plan(a):
    if not hasattr(a,'_plan'):
        history=st.read_json(Path(__file__).with_name('historical_supports.json'))
        a._plan=select({d:parent(a,d) for d in DOMAINS},history,a.data.parent)
    return a._plan


def freeze_plan(a):
    r=st.sealed_json(a.run_root/'common_manifest.json',plan(a));supports=[];common=[];excluded=[];counts=[]
    for d,x in r['domains'].items():
        for seed in SEEDS:supports.extend(dict(row,domain=d,support_seed=seed,group_index=i+1,support_index=i+1,source_record_support_index=row.get('support_index')) for i,row in enumerate(x['supports'][str(seed)]))
        common.extend(x['common']);excluded.extend(dict(row['image'],excluded_by_seeds=row['seeds']) for row in x['excluded'])
        counts.append(dict(domain=d,full_images=x['full_images'],common_images=x['common_images'],normal_images=x['normal_images'],anomaly_images=x['anomaly_images'],overlap=x['overlap'],seed44_normals_retained=len(x['seed44_normals_retained'])))
    st.write_csv(a.run_root/'support_manifest.csv',supports);st.write_csv(a.run_root/'common_evaluation_manifest.csv',common)
    st.write_csv(a.run_root/'excluded_images.csv',excluded);st.write_csv(a.run_root/'population.csv',counts)
    return r


def freeze(a):
    manifest=freeze_plan(a);p=parent(a,'background');providers={}
    for d in DOMAINS:
        w=weight42(a,d)
        if w['weights'][SOURCE]!=weight42(a,'background')['weights'][SOURCE]:raise ValueError('Source weights differ across seed42 domains')
        providers[d]=dict(path=str(a.weight42_runs[d]),inputs=st.sealed_json(a.weight42_runs[d]/'inputs.json')['seal'],weights=st.sealed_json(a.weight42_runs[d]/'weights.json')['seal'])
    value=dict(protocol=CONFIG,common_manifest=manifest['seal'],Source_seed=42,source=p.assets['references']['source'],extraction=p.extraction,mad=p.mad,
        Source_calibration=[x for x in p.catalog['source'] if x['calibration']],seed42_providers=providers,parent_run=str(a.parent_run),parent_cache=str(a.parent_cache),
        raw_run=str(a.raw_run) if a.raw_run else None,raw_cache=str(a.raw_cache) if a.raw_cache else None,
        feature_manifest=str(a.feature_manifest) if a.feature_manifest else None,legacy_provider=str(a.legacy_provider) if a.legacy_provider else None,data=str(a.data))
    if a.feature_manifest:value['feature_provider_seal']=st.sealed_json(a.feature_manifest)['seal']
    if a.legacy_provider:value['legacy_provider_seal']=st.sealed_json(a.legacy_provider)['seal']
    value['identity']=st.digest(value)
    return st.sealed_json(a.run_root/'inputs.json',value)


def context(a):
    p=parent(a);group=plan(a)['domains'][a.domain]
    return p,group['supports'][str(a.support_seed)],group['common']


def signature(a):
    frozen=st.sealed_json(a.run_root/'inputs.json')
    manifest=st.sealed_json(a.run_root/'common_manifest.json')
    if frozen['protocol']!=CONFIG or frozen['common_manifest']!=manifest['seal']:raise ValueError('frozen protocol/common population changed')
    group=manifest['domains'][a.domain];supports=group['supports'][str(a.support_seed)];samples=group['common']
    return dict(domain=a.domain,support_seed=a.support_seed,support_id=st.digest(supports),identity=st.digest([frozen['identity'],a.domain,a.support_seed,supports,st.digest(samples)]))


def setting(a):return a.run_root/a.domain/f'seed{a.support_seed}'
def cache(a):return a.cache_root/a.domain/f'seed{a.support_seed}'
def record_path(a,row):return setting(a)/'predictions'/(row['sample_id']+'.json')


def baseline_dependencies(a,row):
    p,supports,_=context(a)
    if not hasattr(a,'_geometry_definition'):
        a._geometry_definition=dict(distance='original_Exact_FP32_euclidean_divide_1280_once',segment_geometry=st.sha(ROOT/'DINOv3/MADEqual/aebad_s_target_local_segment/geometry.py'))
    return dict(domain=a.domain,support_seed=a.support_seed,Source_seed=42,source=p.assets['references']['source']['seal'],
        supports=supports,extraction=p.extraction,patch=p.mad['patch'],image=row,geometry=a._geometry_definition)


def legacy_ref(a,row):
    if not a.legacy_provider:return None
    r=st.sealed_json(a.legacy_provider);ref=r.get('records',{}).get(f'{a.domain}/seed{a.support_seed}/'+row['sample_id'])
    if ref is None:return None
    x=st.sealed_json(Path(ref['path'])/'manifest.json')
    if x['seal']!=ref['seal'] or x['metadata'].get('dependencies')!=baseline_dependencies(a,row):raise ValueError('legacy record lacks matching actual Source/feature/support/geometry dependencies')
    return ref


def inspect_parent_query(ref,row):
    """Read the parent's query schema, distinct from normal-feature bundles.

    SACP's sealed assets bind parent_identity to the original extraction provider;
    query producers store sample/identity and do not set image or a normal kind.
    """
    path=Path(ref['path']);record=st.sealed_json(path/'manifest.json');meta=record['metadata']
    if (record['seal']!=ref['seal'] or meta.get('sample')!=row
            or not ref.get('parent_identity') or meta.get('identity')!=ref['parent_identity']):
        raise ValueError('parent query feature sample/provider identity differs')
    n=int(np.prod(meta['grid']))
    for layer in LAYERS:layer_inputs.old.header(path,f'q{layer}',(n,1280),np.float32)
    layer_inputs.old.positions(layer_inputs.old.header(path,'positions',(n,),np.int64),n)
    return ref


def feature_ref(a,row):
    p=parent(a);a._parent=p
    own=a.cache_root/'normal_features'/layer_inputs.old.key(row)
    if (own/'manifest.json').exists():
        r=st.sealed_json(own/'manifest.json')
        if r['metadata'].get('image')!=row or r['metadata'].get('extraction')!=p.extraction:raise ValueError('shared feature configuration differs')
        return dict(path=str(own),seal=r['seal'])
    if a.feature_manifest:
        r=st.sealed_json(a.feature_manifest)
        if r['extraction']!=p.extraction:raise ValueError('explicit query feature provider extraction differs')
        ref=r.get('normal_features',r.get('records',{})).get(row['relative_path'])
        if ref:
            layer_inputs.old.inspect_raw(ref,row);return ref
    try:return layer_inputs.feature_ref(a,row)
    except FileNotFoundError:
        ref=p.assets.get('queries',{}).get(row['domain']+'/'+row['sample_id'])
        if ref:return inspect_parent_query(ref,row)
        return None


def preflight(a):
    reusable=[];missing=[];unknown=[];selected=(a.domain,a.support_seed)
    try:
        m=freeze_plan(a)
        for d in DOMAINS:
            parent(a,d);weight42(a,d)
        for d,x in m['domains'].items():
            a.domain=d
            reusable.append(dict(domain=d,role='seed42_weights_normal_statistics',path=str(a.weight42_runs[d]),evaluation_metrics_reusable=False))
            for seed in SEEDS:
                a.support_seed=seed
                if seed==43:
                    normal=setting(a)/'normal_statistics.json'
                    if normal.exists():
                        r=st.sealed_json(normal)
                        if any(r.get(k)!=v for k,v in signature(a).items()):raise ValueError('existing normal statistics group differs')
                        reusable.append(dict(domain=d,support_seed=seed,role='seed43_normal_statistics',path=str(normal)))
                    else:
                        for i,row in enumerate(x['supports']['43']):
                            path=cache(a)/'LOO'/f'{i:02d}'
                            if (path/'manifest.json').exists():
                                meta=st.sealed_json(path/'manifest.json')['metadata']
                                expected=dict(domain=d,support_seed=seed,Source_seed=42,source=parent(a).assets['references']['source']['seal'],query=row,reference_images=[v for j,v in enumerate(x['supports']['43']) if j!=i],extraction=parent(a).extraction)
                                if any(meta.get(k)!=v for k,v in expected.items()):raise ValueError('cached support LOO group differs')
                                reusable.append(dict(domain=d,support_seed=seed,role='Target_LOO_distance',sample_id=row['sample_id'],path=str(path)))
                            else:missing.append(dict(domain=d,support_seed=seed,role='Target_LOO_distance',sample_id=row['sample_id'],path=str(path)))
                for row in x['common']:
                    own=cache(a)/'baselines'/row['sample_id']
                    if seed==42:
                        try:parent(a).prediction(row,False);reusable.append(dict(domain=d,support_seed=seed,role='baseline42',sample_id=row['sample_id']))
                        except FileNotFoundError:missing.append(dict(domain=d,support_seed=seed,role='baseline42_parent_layers',sample_id=row['sample_id'],path=str(parent(a).cache/'predictions'/d/row['sample_id'])))
                    elif (own/'manifest.json').exists() or legacy_ref(a,row):
                        if (own/'manifest.json').exists() and st.sealed_json(own/'manifest.json')['metadata'].get('dependencies')!=baseline_dependencies(a,row):raise ValueError('own seed43 baseline dependency mismatch')
                        reusable.append(dict(domain=d,support_seed=seed,role='baseline43',sample_id=row['sample_id']))
                    else:
                        missing.append(dict(domain=d,support_seed=seed,role='query43_retrieval',sample_id=row['sample_id']))
                        if feature_ref(a,row) is None:missing.append(dict(domain=d,support_seed=seed,role='query_feature',sample_id=row['sample_id'],path=row['relative_path']))
                if seed==43 and any(r.get('domain')==d and r['role'] in ('query43_retrieval','Target_LOO_distance') for r in missing):
                    for row in x['supports']['43']:
                        ref=feature_ref(a,row)
                        (reusable if ref else missing).append(dict(domain=d,support_seed=seed,role='support_feature',sample_id=row['sample_id'],path=ref['path'] if ref else row['relative_path']))
                status=setting(a)/'status.json'
                if status.exists() and st.read_json(status).get('status')=='common_five_metrics_complete':
                    from .reporting import read_complete
                    read_complete(a);reusable.append(dict(domain=d,support_seed=seed,role='this_common_population_evaluation',maps=4,path=str(status)))
                else:missing.append(dict(domain=d,support_seed=seed,role='common_population_evaluation',maps=4,reason='old per-domain and old 3-seed metrics cannot be reused on the new common population'))
        source=Path(parent(a).assets['references']['source']['path'])
        if any(r['role'] in ('query43_retrieval','Target_LOO_distance') for r in missing) and not (source/'manifest.json').exists():missing.append(dict(role='Source_bank',path=str(source),automatic_rebuild=False))
        if not a.data.exists():missing.append(dict(role='GT_and_RGB_dataset',path=str(a.data),reason='GT required for common evaluation; RGB only for explicit missing feature extraction'))
        else:reusable.append(dict(role='GT_and_RGB_dataset',path=str(a.data),presence_checked=True,per_image_mapping_checked_during_evaluation=True))
        history=st.read_json(Path(__file__).with_name('historical_supports.json'))
        unknown.append(dict(role='native_old_seed43_cache',status='not_admitted_Source_bank_equivalence_unconfirmed',old_memory_sha256=history['provenance']['assets']['memory_sha256'],
            current_Source_seal=parent(a).assets['references']['source']['seal'],reason='old .pt memory hash is not proof of current per-layer bank/extraction/MAD equivalence; review contains no prediction NPZ; historical support paths checked, no whole old pipeline invoked'))
        result=dict(status='inventory_complete_execution_pending',common_images=m['total_common_images'],prediction_instances=2*m['total_common_images'],independent_map_jobs=24,reusable=reusable,missing=missing,unconfirmed=unknown,
            estimate=dict(full_hours_range=[1.,8.],conditional='original Source and seed42 parents available; seed43 may require up to common_images query forwards plus 24 new support forwards; new throughput unmeasured',
            breakdown=dict(normal_features_and_LOO='0.1-0.8h if missing supports',query43_extraction_and_Exact_SEGMENT='0.5-4.5h provisional from earlier ~3-7s/image plus IO',seed42_replay_and_common_pixel_AUPRO='0.2-1.5h; all 24 maps re-evaluated',IO_summary='0.2-1.2h'),
            disk_increment_GiB_range=[8,20],SSD_scratch_GiB_range=[3,12],raw_query_features_persisted=False,over_12h_requires_approval=True),torch_imported='torch' in __import__('sys').modules)
    except (OSError,ValueError,KeyError) as e:result=dict(status='blocked',error=str(e),reusable=reusable,missing=missing,unconfirmed=unknown)
    finally:a.domain,a.support_seed=selected
    a.run_root.mkdir(parents=True,exist_ok=True);st.write_json(a.run_root/'preflight.json',result)
    st.write_csv(a.run_root/'asset_inventory.csv',reusable+unknown);st.write_csv(a.run_root/'missing_inputs.csv',missing)
    return result
