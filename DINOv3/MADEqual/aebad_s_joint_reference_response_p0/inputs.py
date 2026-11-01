"""Metadata-first inventory; no Torch import, bank rebuild, or implicit recovery."""
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
from .protocol import *
from . import numerics as nm
from DINOv3.MADEqual.aebad_s_loo_weight_separability_audit import inputs as audit
from DINOv3.MADEqual.aebad_s_target_loo_support_retest_p0 import inputs as previous

st = audit.st
GTData = audit.GTData
ROOT = audit.ROOT


def parent(a):
    if not hasattr(a, '_joint_parent'):
        a._joint_parent = audit.Parent(a.parent_run)
    return a._joint_parent


def unit(a, domain, seed): return a.run_root/domain/f'seed{seed}'
def normal_path(a, domain, seed, j): return a.cache_root/'normal_responses'/domain/f'seed{seed}'/f'{j:02d}'
def query_path(a, domain, seed, row): return a.cache_root/'split_distances'/domain/f'seed{seed}'/row['sample_id']
def source_path(a, domain, row): return a.cache_root/'shared_Source'/domain/row['sample_id']
def prediction_path(a, domain, seed, row): return a.cache_root/'predictions'/domain/f'seed{seed}'/row['sample_id']
def record_path(a, domain, seed, row): return unit(a, domain, seed)/'predictions'/(row['sample_id']+'.json')


def image_contract(row):
    return {k: row[k] for k in ('relative_path', 'image_sha256', 'image_height', 'image_width')}


def source_contract(a, row):
    p = parent(a)
    return dict(image=image_contract(row), source_seal=p.inputs['source']['seal'],
                extraction=p.inputs['extraction'], layers=list(LAYERS),
                distance='original_Exact_FP32_euclidean_divide_1280_once')


def split_contract(a, d, seed, row, heldout=None):
    supports = parent(a).common['domains'][d]['supports'][str(seed)]
    indices = list(range(8)) if heldout is None else heldout_indices(heldout)
    return dict(source=source_contract(a, row), domain=d, support_seed=seed,
                reference_images=[image_contract(supports[j]) for j in indices], support_indices=indices,
                heldout=heldout)


def signature(a, d, seed):
    r = st.sealed_json(a.run_root/'inputs.json')
    return dict(identity=r['identity'], domain=d, support_seed=seed)


def freeze(a):
    p = parent(a)
    value = dict(config=CONFIG, parent_identity=p.identity, parent_inputs_seal=p.inputs['seal'],
                 common_manifest_seal=p.common['seal'], parent_run=str(a.parent_run),
                 source_contract=dict(source=p.inputs['source']['seal'], extraction=p.inputs['extraction'], patch=p.inputs['mad']['patch']),
                 routes={k: str(getattr(a, k)) if getattr(a, k, None) else None for k in
                         ('memory42_root', 'memory43_root', 'source_bank', 'support_feature_root', 'feature_manifest', 'sacp_run', 'raw_cache', 'data')},
                 cache_root=str(a.cache_root), model_device=a.device,
                 blocks=dict(query=a.query_chunk, reference=a.reference_chunk), job_plan=work_plan())
    value['numerical_code'] = {name: st.sha(Path(__file__).with_name(name)) for name in ('protocol.py', 'numerics.py', 'runtime.py', 'prediction.py')}
    if a.feature_manifest:
        value['feature_provider_seal'] = st.sealed_json(a.feature_manifest)['seal']
    value['identity'] = st.digest(value)
    frozen = st.sealed_json(a.run_root/'inputs.json', value)
    st.sealed_json(a.run_root/'protocol.json', CONFIG)
    st.sealed_json(a.run_root/'common_manifest.json', {k: v for k, v in p.common.items() if k != 'seal'})
    return frozen


def old_args(a):
    """Adapt paths to the existing scalar reader without requiring old banks."""
    return SimpleNamespace(parent_run=a.parent_run, response_root=None, memory42_root=a.memory42_root,
                           memory43_root=a.memory43_root, _parent=parent(a))


def original_baseline(a, d, seed, row, keys=(U, 'positions')):
    ref = audit.inspect(old_args(a), d, seed, row)
    x, record = st.get(ref['path'], keys)
    return x, record, ref


def inspect_split(a, d, seed, row, load=True):
    """Full split arrays, never reconstruct two channels from UNION."""
    own = query_path(a, d, seed, row)
    if (own/'manifest.json').is_file():
        x, record = st.get(own, ('Source_distance', 'target_distance', 'positions'))
        if record['metadata']['contract'] != split_contract(a, d, seed, row):
            raise ValueError('split query Source/support/image/extraction dependency mismatch')
        meta = record['metadata']
    else:
        ref = audit.inspect(old_args(a), d, seed, row)
        record = st.sealed_json(Path(ref['path'])/'manifest.json')
        if not {'Source_distance', 'target_distance', 'positions'}.issubset(record['files']):
            raise FileNotFoundError('parent contains UNION but not separate Source/per-support Target distances: '+ref['path'])
        x, record = st.get(ref['path'], ('Source_distance', 'target_distance', 'positions'))
        meta = record['metadata']
    if load:
        nm.split_check(x['Source_distance'], x['target_distance'], x['positions'], meta['grid'])
    else:
        # Inventory only: shapes/dtypes/physical mapping, never a full distance
        # payload scan. Values are checked when actually consumed for prediction.
        nm.positions_check(x['positions'], meta['grid']);n=len(x['positions'])
        if x['Source_distance'].shape!=(4,n) or x['target_distance'].shape!=(4,8,n) or x['Source_distance'].dtype!=np.float32 or x['target_distance'].dtype!=np.float32:
            raise ValueError('separate distance header differs')
    return x, record


def inspect_source(a, d, row, load=True):
    """Source is shared only with matching actual bank, image, extraction, grid."""
    own = source_path(a, d, row)
    if (own/'manifest.json').is_file():
        x, r = st.get(own)
        if r['metadata']['contract'] != source_contract(a, row):
            raise ValueError('shared Source scientific identity differs')
        nm.positions_check(x['positions'], r['metadata']['grid'])
        if x['Source_distance'].shape != (4, len(x['positions'])) or x['Source_distance'].dtype != np.float32 or (load and not np.isfinite(x['Source_distance']).all()):
            raise ValueError('invalid shared Source distance')
        return x, r, dict(path=str(own), seal=r['seal'], support_seed_independent=True)
    # seed43 is normally the complete compact provider; original inspect binds all
    # dependencies, including Source and feature extraction. Reorder physically.
    for seed in (43, 42):
        try:
            x, r = inspect_split(a, d, seed, row, load=load)
        except FileNotFoundError:
            continue
        return dict(Source_distance=x['Source_distance'], positions=x['positions']), r, dict(seal=r['seal'], provider_seed=seed, support_seed_independent=True)
    raise FileNotFoundError('no scientifically compatible Source distance: '+d+'/'+row['sample_id'])


def feature_ref(a, row):
    from DINOv3.MADEqual.aebad_s_same_origin_neighbor_p0 import feature_identity as fi
    p = parent(a);options = []
    key = previous.layer_inputs.old.key(row)
    for root in (a.cache_root, a.support_feature_root, a.memory43_root, a.raw_cache):
        if root:
            options.extend((root/'normal_features'/key, root/key))
    if a.feature_manifest:
        export = st.sealed_json(a.feature_manifest)
        fi.compatible(export.get('extraction'), p.inputs['extraction'])
        ref = export.get('normal_features', export.get('records', {})).get(row['relative_path'])
        if ref:
            options.insert(0, Path(ref['path']))
    assets = st.sealed_json(a.sacp_run/'assets.json') if (a.sacp_run/'assets.json').exists() else None
    if assets:
        if assets['references']['extraction'] != p.inputs['extraction'] or assets['references']['source']['seal'] != p.inputs['source']['seal']:
            raise ValueError('feature authority belongs to another Source/extraction')
        ref = assets.get('normal_features', {}).get(row['relative_path']) or assets.get('queries', {}).get(row['domain']+'/'+row.get('sample_id', ''))
        if ref:
            options.append(Path(ref['path']))
    for path in dict.fromkeys(options):
        if not (path/'manifest.json').exists():
            continue
        r = st.sealed_json(path/'manifest.json');meta = r['metadata']
        image = meta.get('image', meta.get('sample', {}))
        if image_contract(image) != image_contract(row):
            raise ValueError('feature image identity differs')
        bridge = SimpleNamespace(run_root=a.cache_root/'_unused_feature_probe', parent_run=a.sacp_run,
                                 raw_cache=a.raw_cache, feature_root=a.support_feature_root, feature_manifest=a.feature_manifest)
        proof = fi.proof(bridge, path, row, r)
        if proof['extraction_identity'] != st.digest(p.inputs['extraction']):
            raise ValueError('feature provider extraction differs')
        n = int(np.prod(meta['grid']))
        for l in LAYERS:
            previous.layer_inputs.old.header(path, 'q'+str(l), (n, 1280), np.float32)
        pos = previous.layer_inputs.old.header(path, 'positions', (n,), np.int64)
        nm.positions_check(pos, meta['grid'])
        return dict(path=str(path), seal=r['seal'], grid=meta['grid'], proof=proof)
    return None


def inspect_normal(a, d, seed, j):
    row = parent(a).common['domains'][d]['supports'][str(seed)][j]
    path = normal_path(a, d, seed, j)
    if not (path/'manifest.json').is_file():
        raise FileNotFoundError(str(path))
    x, r = st.get(path)
    if r['metadata']['contract'] != split_contract(a, d, seed, row, j):
        raise ValueError('normal reference whole-image K7 identity differs')
    n = len(x['positions']);nm.positions_check(x['positions'], r['metadata']['grid'])
    for k in ('Source_distance', 'Target7_distance'):
        if x[k].shape != (4, n) or x[k].dtype != np.float32 or not np.isfinite(x[k]).all() or (x[k] < 0).any():
            raise ValueError('invalid separate normal K7 responses')
    return x, r


def preflight(a):
    reusable = [];missing = [];required = [];work = [];unknown = [];sampling_plan=[]
    try:
        p = parent(a);freeze(a)
        source_needed = False;missing_query_features = set()
        for d in DOMAINS:
            for row in p.samples(d):
                shared = None
                try:
                    _, _, shared = inspect_source(a, d, row, load=False)
                except FileNotFoundError:
                    source_needed = True
                need_query = False
                for seed in SEEDS:
                    original=p.original(d,seed,row)
                    item = dict(role='query_split_distances', domain=d, support_seed=seed, sample_id=row['sample_id'], relative_path=row['relative_path'],
                                original_path=original['path'], expected_original_manifest_seal=original['seal'],
                                files='manifest.json;Source_distance.npy;target_distance.npy;positions.npy',
                                note='seed42 original producer may lack split fields; do not assume availability')
                    required.append(item)
                    try:
                        _, r = inspect_split(a, d, seed, row, load=False);reusable.append(dict(item, seal=r['seal'], action='reuse_separate_distances'))
                    except FileNotFoundError as e:
                        need_query = True;missing.append(dict(item, reason=str(e), source_reusable=shared is not None))
                        work.append(dict(item, necessary='per_support_Target_Exact', Source_search_required=shared is None))
                    try:
                        ref = audit.inspect(old_args(a), d, seed, row)
                        reusable.append(dict(item, role='frozen_K8_baseline', path=ref['path'], action='replay_check_no_formal_reevaluation'))
                    except FileNotFoundError as e:
                        missing.append(dict(item, role='frozen_K8_baseline', reason=str(e), recovery='restore original scalar UNION/SEGMENT cache; no inversion'))
                if need_query:
                    ref = feature_ref(a, row)
                    if ref is None:
                        missing_query_features.add(d+'/'+row['sample_id'])
                        work.append(dict(domain=d, sample_id=row['sample_id'], role='query_features', necessary='one_shared_full_forward_if_explicitly_authorized', maximum_forwards=1))
                    else:
                        reusable.append(dict(role='query_features', domain=d, sample_id=row['sample_id'], **ref))
            for seed in SEEDS:
                for j, row in enumerate(p.common['domains'][d]['supports'][str(seed)]):
                    sampling_plan.append(dict(domain=d,support_seed=seed,support_index=j,sample_id=row.get('sample_id'),
                        relative_path=row['relative_path'],image_sha256=row['image_sha256'],maximum_positions=512,
                        rule=CONFIG['normal_sampling'],status='coordinates_generated_only_after_actual_grid_validation'))
                    item = dict(role='normal_separate_K7_response', domain=d, support_seed=seed, support_index=j, relative_path=row['relative_path'],
                                path=str(normal_path(a,d,seed,j)),files='manifest.json;Source_distance.npy;Target7_distance.npy;positions.npy')
                    required.append(item)
                    try:
                        _, r = inspect_normal(a, d, seed, j);reusable.append(dict(item, seal=r['seal'], action='reuse'))
                    except FileNotFoundError:
                        source_needed = True;ref = feature_ref(a, row)
                        missing.append(dict(item, feature_available=ref is not None,
                                            recovery='reuse complete normal features; supplement Source and same-group K7 Exact once'))
                        if ref:reusable.append(dict(item, role='support_features', **ref))
                        else:missing.append(dict(item, role='support_features', recovery='restore compatible features or explicitly authorize one original forward'))
        bank = a.source_bank or Path(p.inputs['source']['path'])
        if (bank/'manifest.json').exists():
            r = st.sealed_json(bank/'manifest.json')
            if r['seal'] != p.inputs['source']['seal']:raise ValueError('Source bank identity differs; no replacement/rebuild')
            for l in LAYERS:previous.layer_inputs.old.header(bank, 'L'+str(l), (100000, 1280), np.float32)
            reusable.append(dict(role='Source_bank', path=str(bank), needed_for_supplement=source_needed))
        elif source_needed:
            missing.append(dict(role='Source_bank', path=str(bank), recovery='restore exact parent source_original bundle; rebuilding is NOT implemented or authorized'))
        else:
            unknown.append(dict(role='Source_bank_entity', needed=False, reason='all separate distances already cached'))
        with GTData(a.data) as data:
            missing.extend(dict(role='GT', relative_path=x) for x in data.missing([r for d in DOMAINS for r in p.samples(d)]))
        result = dict(status='missing_inputs' if missing else 'inputs_available', plan=work_plan(), missing_records=len(missing),
                      maximum_missing_query_forwards=len(missing_query_features), torch_imported='torch' in sys.modules)
    except (OSError, ValueError, KeyError) as e:
        result = dict(status='blocked', error=str(e), torch_imported='torch' in sys.modules)
        missing.append(dict(role='input_error', reason=str(e)))
    from importlib.metadata import version, PackageNotFoundError
    versions={}
    for name in ('numpy','scipy','Pillow','psutil','torch','torchvision','adeval'):
        try:versions[name]=version(name)
        except PackageNotFoundError:versions[name]=None
    result['dependency_versions_without_import']=versions
    result['environment_note']='metadata inventory only, no CUDA/runtime probe; ADEval1.1.0 required for explicit AUPRO stage'
    st.write_csv(a.run_root/'asset_inventory.csv', reusable+unknown)
    st.write_csv(a.run_root/'missing_inputs.csv', missing)
    st.write_csv(a.run_root/'required_cache_files.csv', required)
    st.write_csv(a.run_root/'necessary_work.csv', work)
    st.write_csv(a.run_root/'normal_sampling_plan.csv',sampling_plan)
    st.write_json(a.run_root/'preflight.json', result)
    if not (a.run_root/'time_estimate.json').exists():
        st.write_json(a.run_root/'time_estimate.json', dict(status='unknown_until_fixed_normal_asset_smoke', total_seconds=None,
            stages={k: None for k in ('cache_read', 'support_supplement', 'query_supplement', 'tail_model_prediction', 'exact_CPU', 'AUPRO', 'IO_and_pack')},
            work_plan=work_plan(), necessary_repetition=work, new_disk_bytes=None,
            rule='no unknown cost is zero; full launch requires measured normal-asset smoke and <=12h or explicit approval'))
    return result
