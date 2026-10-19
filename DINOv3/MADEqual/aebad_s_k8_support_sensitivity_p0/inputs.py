"""Independent full test index, frozen subset and parameterized asset admission."""
from __future__ import annotations

import hashlib
import io
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from . import selection as sel
from .storage import sha, digest, read_json, write_json, csv_file, safe, Ledger, json_bytes, csv_bytes
from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0 import inputs as old

TOOL = 'aebad_s_k8_support_sensitivity_v1'
BACKGROUND_IDENTITY_SHA = 'f580b9fa02a874a931e7824a7e65f817bc00b1058500dc5af94cf125dd81a82b'
LAYERS = (7, 15, 23, 31)
PROTOCOL = dict(tool=TOOL, domains=list(sel.DOMAINS), support_seeds=list(sel.SEEDS), source_seed=42,
    support_images=8, layers=list(LAYERS), dimension=1280, support_compression=False,
    Source_brightness=[.8, 1.2], Source_images=521, Source_memory_images=455, Source_calibration_images=66,
    Source_rows_per_layer=100000, query=old.QUERY, precision=old.PRECISION,
    distance_unit='euclidean_over_1280_once', geometry='original_geometry_layer_bounded_distances',
    pair='original_crossdomain_predict_radius1_exclude_center_ceil_actual_neighbors_half',
    distance_atol=1e-7, distance_rtol=1e-6, internal_score_atol=1e-5,
    historical_strict_atol=1e-5, historical_compatibility_atol=1e-4, historical_rtol=0,
    score='frozen_Source_CH1_peak_local_half_half', CURRENT='historical_audit_only',
    localization='Source_patch_MAD_equal4_original_bilinear_negative_responses_retained',
    nstrips=200, FPR_max=.3, aupro='original_CUDA_ADEval_1.1.0',
    aggregation='domain_equal_per_seed_then_three_support_repetitions_mean_sample_std_range',
    small_instances='per_domain_smallest_ceil_N_over_4_original_GT_area_same_across_seeds',
    default_shutdown=False)


def defaults(root):
    root=Path(root)
    paths = old.defaults(root)
    paths.pop('cache_root')
    paths['output_root'] = Path(root)/'DINOv3/outputs/mad_equal/aebad_s_k8_support_sensitivity_v1'
    # Current local assets were found by filename inventory, not inferred from
    # old READMEs. Existence never bypasses the frozen SHA checks.
    for role,candidate in (('memory',root/'downloads/official_memory.pt'),
                           ('checkpoint',root/'_external/dinov3_checkpoints/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth'),
                           ('dinov3',root/'_external/dinov3')):
        if not paths[role].exists() and candidate.exists():paths[role]=candidate
    return paths


class Data:
    """ZIP/directory adapter: no train-folder requirement, no GT decoding before evaluation."""
    def __init__(self, path):
        self.path = Path(path)
        self.zip = zipfile.ZipFile(path) if self.path.is_file() else None
        if self.zip:
            entries = [n for n in self.zip.namelist() if n.startswith('AeBAD/AeBAD_S/') and not n.endswith('/')]
            if len(entries) != len(set(entries)):
                raise ValueError('duplicate ZIP members')
            self.names = {n[len('AeBAD/AeBAD_S/'):] for n in entries}
            self.root = None
        else:
            self.root = old.common.dataset_root(self.path)
            self.names = {p.relative_to(self.root).as_posix() for folder in ('test', 'ground_truth')
                          for p in (self.root/folder).rglob('*') if p.is_file()}

    def __enter__(self): return self
    def __exit__(self, *args):
        if self.zip: self.zip.close()

    def open(self, name):
        name = safe(name)
        return self.zip.open('AeBAD/AeBAD_S/'+name) if self.zip else (self.root/name).open('rb')

    def bytes(self, name, expected=None):
        with self.open(name) as f: content = f.read()
        if expected and hashlib.sha256(content).hexdigest() != expected:
            raise ValueError('image/GT bytes changed: '+name)
        return content

    def image(self, sample):
        with Image.open(io.BytesIO(self.bytes(sample['relative_path'], sample['image_sha256']))) as im:
            return im.convert('RGB')

    def gt(self, sample):
        if not sample['label']:
            return np.zeros((sample['image_height'], sample['image_width']), bool)
        name = sample['relative_path'].replace('test/', 'ground_truth/', 1)
        with Image.open(io.BytesIO(self.bytes(name, sample['gt_sha256']))) as im:
            mask = np.asarray(im.convert('L')) > 0
        if mask.shape != (sample['image_height'], sample['image_width']) or not mask.any():
            raise ValueError('required GT empty or wrong dimensions: '+name)
        return mask


def code_hashes():
    root = Path(__file__).resolve().parents[3]
    hashes = old.code_hashes()
    extra = list(Path(__file__).parent.glob('*.py')) + [root/p for p in (
        'DINOv3/MADEqual/aebad_s_target_fewshot/support.py',
        'DINOv3/MADEqual/aebad_s_segment_pair_crossdomain_p0/frozen_catalog.json',
        'DINOv3/ShiftADD/aebad_s_p0/memory.py')]+[Path(__file__).with_name('background_identity.json')]
    hashes.update({p.relative_to(root).as_posix(): sha(p) for p in extra})
    return hashes


def metadata(paths):
    catalog = read_json(Path(old.__file__).with_name('frozen_catalog.json'))
    if sha(Path(old.__file__).with_name('frozen_catalog.json')) != old.CATALOG_SHA:
        raise ValueError('original frozen formal identity changed')
    for name, checksum in catalog['formal_sha256'].items():
        if sha(paths['formal_dir']/name) != checksum:
            raise ValueError('frozen formal metadata changed: '+name)
    if sha(paths['image_state']) != catalog['image_state_sha256']:
        raise ValueError('Source image MAD file changed')
    state = read_json(paths['image_state'])
    if (digest({k: v for k, v in state.items() if k != 'state_sha256'}) != state['state_sha256']
            or state['patch_mad_frozen_sha256'] != catalog['formal_sha256']['source_mad_state.npz']):
        raise ValueError('Source patch/image MAD provenance mismatch')
    with np.load(paths['formal_dir']/'source_mad_state.npz', allow_pickle=False) as f:
        if str(f['protocol_id'].item()) != 'mad_equal_aebad_s_target_fewshot_seed42_v1':
            raise ValueError('patch MAD protocol mismatch')
        medians, scales = f['medians'].copy(), f['scales'].copy()
    for i, r in enumerate(csv_file(paths['formal_dir']/'calibration_stats.csv')):
        if (int(r['layer']) != LAYERS[i] or int(r['patch_count']) != 918984
                or not np.isclose(float(r['median']), medians[i], atol=1e-7, rtol=1e-6)
                or not np.isclose(float(r['mad_scale_1p4826']), scales[i], atol=1e-7, rtol=1e-6)):
            raise ValueError('original calibration statistics differ from frozen patch MAD')
    manifest = read_json(paths['formal_dir']/'memory_manifest.json')
    if (manifest['official_brightness'] != [.8, 1.2] or manifest['memory_fit_count'] != 455
            or manifest['calibration_count'] != 66
            or manifest['entries_per_layer'] != {str(l): 100000 for l in LAYERS}):
        raise ValueError('not the original Official brightness Source bank')
    return catalog, state, medians, scales


def build_plan(paths, *, verify_bytes=False):
    catalog, state, medians, scales = metadata(paths)
    formal = csv_file(paths['formal_dir']/'sample_scores.csv')
    references = {}
    for r in formal:
        if r['method'] in ('MAD_EQUAL', 'TARGET_MATCHED_K8_MAD'):
            references.setdefault(r['relative_path'], {})[r['method']] = r
    known = {s['relative_path']: s for d in catalog['domains'].values() for s in d['samples']+d['supports']}
    background_path=Path(__file__).with_name('background_identity.json')
    if sha(background_path)!=BACKGROUND_IDENTITY_SHA:
        raise ValueError('historical background raster identity manifest changed')
    known.update(read_json(background_path)['images'])
    records = []
    with Data(paths['data']) as data:
        names = sorted(n for n in data.names if n.startswith('test/') and len(n.split('/')) == 4
                       and Path(n).suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
                       and not Path(n).name.startswith('._') and '__MACOSX' not in n)
        if len(names) != 1639:  # original 689+305+273+372, not a filtered index
            raise ValueError(f'original full test index has {len(names)} images; expected 1639')
        for index, name in enumerate(names):
            domain = name.split('/')[2]
            if domain not in sel.DOMAINS: continue
            label = int(name.split('/')[1] != 'good')
            sample = dict(sample_id=f'test_{index:04d}', relative_path=name, domain=domain, label=label)
            for method, field in (('MAD_EQUAL', 'source_score'), ('TARGET_MATCHED_K8_MAD', 'union_score')):
                ref = references.get(name, {}).get(method)
                if ref:
                    if (ref['sample_id'], int(ref['label']), ref['domain']) != (sample['sample_id'], label, domain):
                        raise ValueError('original sample ID/path/label mismatch: '+name)
                    sample[field] = float(ref['image_score'])
            with data.open(name) as f, Image.open(f) as im:
                sample.update(image_height=im.height, image_width=im.width)
            if name not in known or any(sample[k]!=known[name][k] for k in ('image_height','image_width')):
                raise ValueError('image identity/header not in original frozen raster manifest: '+name)
            if verify_bytes:
                sample['image_sha256'] = hashlib.sha256(data.bytes(name)).hexdigest()
                if name in known and sample['image_sha256'] != known[name]['image_sha256']:
                    raise ValueError('original image identity changed: '+name)
            if label:
                gt = name.replace('test/', 'ground_truth/', 1)
                if gt not in data.names: raise FileNotFoundError('required GT: '+gt)
                if verify_bytes:
                    sample['gt_sha256'] = hashlib.sha256(data.bytes(gt)).hexdigest()
                    if name in known and sample['gt_sha256'] != known[name]['gt_sha256']:
                        raise ValueError('original GT identity changed: '+gt)
            records.append(sample)
    plan = sel.select(records, csv_file(paths['formal_dir']/'support_manifest.csv'), root=paths['project'])
    # Every common evaluation image has a formal historical score; supports don't need one.
    if any('source_score' not in s or 'union_score' not in s for s in plan['samples']):
        raise ValueError('common evaluation samples absent from formal historical CSV')
    assets = dict(formal_sha256=catalog['formal_sha256'], image_state_sha256=catalog['image_state_sha256'],
                  memory_sha256=catalog['memory_sha256'], checkpoint_sha256=catalog['checkpoint_sha256'])
    return dict(plan=plan, assets=assets, state=state, medians=medians, scales=scales)


def prepare(paths):
    root = paths['output_root']
    if root.exists():
        return load(paths)  # NEVER redraw or replace a previously frozen manifest
    ctx = build_plan(paths, verify_bytes=True)
    frozen = dict(protocol=PROTOCOL, plan=ctx['plan'], assets=ctx['assets'], code=code_hashes())
    identity = digest(frozen)
    # Protect all input paths, but allow the project to contain a dedicated output directory.
    for key, source in paths.items():
        if key in ('output_root', 'project'): continue
        if root == source or root in source.parents or source in root.parents:
            raise ValueError('output overlaps input: '+key)
    ledger = Ledger(root, identity)
    flat = [s for d in sel.DOMAINS for seed in sel.SEEDS for s in ctx['plan']['groups'][d][str(seed)]]
    files = {'frozen.json': json_bytes(dict(identity=identity, frozen=frozen)),
             'protocol.json': json_bytes(PROTOCOL), 'support_manifest.csv': csv_bytes(flat),
             'support_candidates.csv': csv_bytes(ctx['plan']['candidates']),
             'support_overlap.csv': csv_bytes(ctx['plan']['overlap']),
             'common_evaluation_manifest.csv': csv_bytes(ctx['plan']['samples']),
             'excluded_images.csv': csv_bytes(ctx['plan']['excluded']),
             'input_inventory.json': json_bytes(inventory(paths, ctx)),
             'implementation_audit.md': Path(__file__).with_name('implementation_audit.md').read_bytes()}
    ledger.commit('prepare', dict(status='manifests_frozen', GT_decoded=False), files)
    return load(paths)


def load(paths):
    root = paths['output_root']; snapshot = read_json(root/'frozen.json')
    frozen = snapshot['frozen']; identity = snapshot['identity']
    if digest(frozen) != identity or frozen['code'] != code_hashes() or frozen['protocol'] != PROTOCOL:
        raise ValueError('frozen code/protocol identity changed; no resume under new semantics')
    ledger = Ledger(root, identity)
    if not ledger.get('prepare'): raise ValueError('prepare not completed')
    catalog, state, medians, scales = metadata(paths)
    if frozen['assets']['memory_sha256'] != catalog['memory_sha256']:
        raise ValueError('Source identity changed')
    return dict(paths=paths, identity=identity, plan=frozen['plan'], assets=frozen['assets'],
                state=state, medians=medians, scales=scales, ledger=ledger)


def inventory(paths, ctx=None):
    if ctx is None:
        catalog = read_json(Path(old.__file__).with_name('frozen_catalog.json'))
        required = [paths['formal_dir']/n for n in catalog['formal_sha256']]+[paths['image_state'],paths['data']]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            return dict(tool=TOOL,status='missing_manifest_or_data_inputs',missing_assets=missing,
                        torch_imported='torch' in sys.modules,model_loaded=False,GT_decoded=False,
                        note='no downloading, model operation, rebuild or fitting attempted')
    ctx = ctx or build_plan(paths)
    assets = []
    for key in ('memory', 'checkpoint', 'dinov3', 'official', 'data', 'formal_dir', 'image_state'):
        path = paths[key]
        needed = path/('hubconf.py' if key == 'dinov3' else 'config.json') if key in ('dinov3', 'official') else path
        status = 'present' if needed.exists() else 'missing'
        row = dict(asset=key, path=str(path), status=status)
        if key in ('memory', 'checkpoint') and status == 'present':
            row['sha256'] = sha(path)
            row['status'] = 'identity_verified' if row['sha256'] == ctx['assets'][key+'_sha256'] else 'identity_mismatch'
        assets.append(row)
    support_patches=sum(int(s['image_height']*.625)//16*(int(s['image_width']*.625)//16)
        for d in sel.DOMAINS for s in sel.support_pool(ctx['plan'],d)[0])
    query_patches=sum(int(s['image_height']*.625)//16*(int(s['image_width']*.625)//16) for s in ctx['plan']['samples'])
    # Full support FP32 storage; raw shared 24*4*(distance+index), per-group
    # geometry, plus compact predictions. No model scratch or dense matrix.
    support_bytes=support_patches*4*1280*4
    raw_bytes=query_patches*(4*12+24*4*12+3*4*54)
    prediction_bytes=query_patches*(3*4*48)
    recorded=ctx['ledger'].state['records'] if 'ledger' in ctx else {}
    from importlib.metadata import version, PackageNotFoundError
    versions={}
    for name in ('numpy','scipy','Pillow','scikit-learn','psutil','torch','torchvision','timm','omegaconf','adeval'):
        try:versions[name]=version(name)
        except PackageNotFoundError:versions[name]=None
    capture_pending=878-sum(k.startswith('capture/') for k in recorded)
    status=('blocked_identity_mismatch' if any(r['status']=='identity_mismatch' for r in assets)
            else 'blocked_capture_assets_missing' if capture_pending and any(r['status']=='missing' for r in assets)
            else 'inventory_indexed_not_a_runtime_test')
    return dict(tool=TOOL, status=status, assets=assets,
                counts={d:dict(common=len([s for s in ctx['plan']['samples'] if s['domain']==d]),
                    normal=sum(s['label']==0 for s in ctx['plan']['samples'] if s['domain']==d),
                    anomaly=sum(s['label']==1 for s in ctx['plan']['samples'] if s['domain']==d),
                    unique_supports=len(sel.support_pool(ctx['plan'],d)[0])) for d in sel.DOMAINS},
                combinations=27, support_overlap=ctx['plan']['overlap'], Source=PROTOCOL,
                dependency_versions_without_import=versions,
                dependencies_note='metadata only; no model imports, CUDA or ADEval runtime check',
                missing_shared_query_captures=capture_pending,
                supports=[dict(domain=d,support_seed=seed,images=[s['relative_path'] for s in ctx['plan']['groups'][d][str(seed)]])
                          for d in sel.DOMAINS for seed in sel.SEEDS],
                resource_estimates=dict(full_support_float32_bytes=support_bytes,
                    uncompressed_shared_capture_bytes=raw_bytes,uncompressed_compact_prediction_bytes=prediction_bytes,
                    retained_uncompressed_total_bytes=support_bytes+raw_bytes+prediction_bytes,
                    note='geometry-based estimates, not measured peaks; NPZ compression not assumed',
                    capture_RAM_admission_bytes=12*2**30,query_model_minimum_free_VRAM_GiB_default=20,
                    expected_new_query_extractions_without_reuse=878,expected_unique_support_extractions=72),
                image_bytes_verified=all('image_sha256' in s for s in ctx['plan']['samples']),
                GT_decoded=False, torch_imported='torch' in sys.modules, model_loaded=False,
                automatic_shutdown=False)


def local_evidence(project):
    """Read current metadata/manifests, not old README claims. No archive extraction."""
    result = []
    downloads = Path(project)/'downloads'
    for name in ('aebad_segment_pair_compat_20261017_091539_501e80_EVAL_SUCCESS_SUMMARY_20261017_162654.tar.gz',
                 'aebad_s_segment_looB_raw_cache_20261013.tar', 'aebad_s_target_local_segment_review.tar.gz'):
        path = downloads/name
        if not path.is_file():
            result.append(dict(path=str(path), status='absent')); continue
        with tarfile.open(path) as archive:
            members = [m.name for m in archive.getmembers() if m.isfile()]
        result.append(dict(path=str(path), bytes=path.stat().st_size, npz_members=sum(n.endswith('.npz') for n in members),
            retrieval_members=sum('/raw_cache/' in n and n.endswith('.npz') for n in members),
            prediction_members=sum('/predictions/' in n and n.endswith('.npz') for n in members),
            reusable_as_new_support_capture=False, reason='seed42-only or summary; no new-set Top1 evidence'))
    pred = Path(project)/'DINOv3/outputs/mad_equal/aebad_s_neighborhood_segment_local_20261016_134948_predict'
    manifest = pred/'prediction_manifest.json'
    if manifest.is_file():
        result.append(dict(path=str(pred), prediction_records=len(read_json(manifest)['records']),
            role='background_seed42_CPU_regression_only_not_new_full_capture'))
    return result
