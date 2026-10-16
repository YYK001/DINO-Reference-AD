"""Read-only archive adapter and frozen-input admission. No extraction or Torch."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import platform
import tarfile
from pathlib import Path, PurePosixPath

import numpy as np

from DINOv3.MADEqual.aebad_s_union_support_loo.analysis import validate_arrays
from .gates import LAYERS, BRANCHES, ATOL, RTOL

TOOL = 'aebad_s_neighborhood_segment_p0_v1'
SROOT = 'aebad_s_target_local_segment_20261008_220118'
CROOT = 'aebad_s_union_support_loo_B_20261007_230639'
CAP_ID = '4fc53367a32843d12d60c9c7a3a6932f2af694af7dbfab63cf115de57e71a0c1'
SEG_ID = '085ccc952f484ebe09dc5c9e9237382f232ea63867fdb8c703fc826f8eb63c7a'
ARCHIVE_SHA = '6fd4086bbaeb5318a5168dd90eef57d9955ffa37da44764db926913c3e4b2e41'
REVIEW_SHA = 'f2703ee4518330d7f82a7b1dc4477a83b05e626fb104d96f2f1c38313efa7524'
DATA_SHA = 'c643b5e956acdf5949cceab0b495b86be846cda198c01c05666207b81b450080'
CH1_PINS = {
    'image_calibration_state.json': 'd185432e7c2e67c14f437426be05df05a3bc2d021973efd9aae40130fd048f74',
    'per_image_scores.csv': '9f65b6c1acf4340e147cdd86686599b003bca028f5b19ea025fd3a0d17eda2eb',
    'image_metrics.csv': '146c1eb76c7182a137fbe6cb12a3665adc711fe953bf2024d4cbed70f3b54fdb',
    'inputs.json': '033f6dae7583899f3b2576489f65f07c650fa931e1b2619b2d45c6acc5327a77',
    'numerical_verification.csv': '189485b778a0575f87b2f80ce129f7479b2c28f0a178dcce66bbd031a4e1e522',
    'ranking_summary.csv': 'edf76961abc298be483ac7402d420da841f6c59b002cdd7d3bc37bf16ca7c08e',
    'status.json': 'f9f38793e2f738ab30c61338c54eced75641e5ff1c1a8c8fd4e57d3fc0357bf2',
}
PROTOCOL = dict(tool=TOOL, branches=list(BRANCHES), layers=list(LAYERS),
                query_radius=1, exclude_center=True, vote='ceil(valid_neighbors/2)',
                spatial_tolerance_support_patches=1, no_neighbor='full_union',
                distance_unit='euclidean_over_1280_once', distance_atol=ATOL, distance_rtol=RTOL,
                score_atol=1e-5, score_rtol=0, image_score='frozen_CH1_peak_local_0.5_0.5',
                localization='frozen_Source_patch_MAD_equal4_bilinear_align_corners_False',
                image_count=297, normal_count=85, anomaly_count=212,
                evaluation_ring_original_pixels=16, evaluation_ring_metric='Chebyshev_exclude_all_GT',
                R_scale='original_SEGMENT_float32_Source_scale_plus_1e-8',
                aupro_backend='official_aupro_adeval', aupro_thresholds=200, aupro_fpr_max=0.3)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def safe(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
        raise ValueError('unsafe relative archive/manifest path')
    return str(p)


def read_csv(value):
    return list(csv.DictReader(io.StringIO(value.decode('utf-8-sig'))))


def json_file(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    # Native scalar conversion is explicit; NaN never enters provenance.
    def convert(x):
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    p = Path(path); temporary = p.with_suffix(p.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=convert), encoding='utf-8')
    temporary.replace(p)


def write_csv(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


class Archive:
    def __init__(self, path, index=None):
        self.path = Path(path)
        self.tar = tarfile.open(self.path, 'r:*')
        self.members = index if index is not None else {m.name: m for m in self.tar.getmembers() if m.isfile()}
        if index is None and len(self.members) != sum(m.isfile() for m in self.tar.getmembers()):
            self.close(); raise ValueError('duplicate archive members')
        if index is None:
            for name in self.members: safe(name)

    def close(self): self.tar.close()
    def __enter__(self): return self
    def __exit__(self, *args): self.close()

    def bytes(self, name, checksum=None):
        with self.tar.extractfile(self.members[safe(name)]) as f: data = f.read()
        if checksum and hashlib.sha256(data).hexdigest() != checksum:
            raise ValueError(f'archive member modified: {name}')
        return data

    def json(self, name): return json.loads(self.bytes(name))
    def csv(self, name): return read_csv(self.bytes(name))

    def npz(self, name, checksum):
        with np.load(io.BytesIO(self.bytes(name, checksum)), allow_pickle=False) as f:
            meta = json.loads(str(f['metadata'].item()))
            arrays = {k: f[k].copy() for k in f.files if k != 'metadata'}
        return arrays, meta


def defaults(root):
    root = Path(root)
    return dict(raw_archive=root/'downloads/aebad_s_segment_looB_raw_cache_20261013.tar',
                review_archive=root/'downloads/aebad_s_target_local_segment_review.tar.gz',
                formal_dir=root/'server_archives/aebad_s_target_fewshot_seed42_v1_formal_results/aebad_s_target_fewshot_seed42_v1',
                ch1_dir=root/'DINOv3/outputs/mad_equal/aebad_s_ch1_scoring_local_20261016_034133',
                data=root/'datasets/AeBAD.zip')


def code_hashes():
    base = Path(__file__).resolve().parents[1]
    files = list(Path(__file__).parent.glob('*.py'))
    files += [base/p for p in (
        'aebad_s_projection_layer_diagnostic/numerics.py',
        'aebad_s_projection_layer_diagnostic/metrics.py',
        'aebad_s_ch1_scoring_inheritance/scoring.py',
        'aebad_s_union_support_loo/analysis.py', 'aebad_s_union_diagnostics/spatial.py')]
    return {p.relative_to(base).as_posix(): sha(p) for p in sorted(files)}


def available_ram():
    import psutil
    available=int(psutil.virtual_memory().available)
    root=Path('/sys/fs/cgroup')
    try:
        limit=(root/'memory.max').read_text().strip()
        if limit!='max': available=min(available,max(0,int(limit)-int((root/'memory.current').read_text())))
    except (OSError,ValueError): pass
    return available


def environment():
    from importlib.metadata import version
    return dict(python=platform.python_version(),platform=platform.platform(),
                packages={name:version(name) for name in ('numpy','scipy','Pillow','scikit-learn','psutil')})


def dataset_root(path):
    """Target-only directory also supported; never require train/good."""
    path=Path(path)
    for candidate in (path,path/'AeBAD_S',path/'AeBAD/AeBAD_S'):
        if (candidate/'test').is_dir() and (candidate/'ground_truth').is_dir(): return candidate
    raise ValueError('data must be the frozen ZIP or an AeBAD_S test/ground_truth root')


def context(paths, *, check_data=False):
    """Pinned metadata admission; heavy per-image raw validation is separate."""
    paths = {k: Path(v).resolve() for k, v in paths.items()}
    checksums = {}
    for key, expected in (('raw_archive', ARCHIVE_SHA), ('review_archive', REVIEW_SHA)):
        actual = sha(paths[key])
        if actual != expected: raise ValueError(f'{key}: not the frozen approved archive')
        checksums[key] = actual
    with Archive(paths['raw_archive']) as a:
        raw_index = a.members.copy()
        capture = a.json(CROOT+'/inputs.json'); capture_status = a.json(CROOT+'/status.json')
        segment = a.json(SROOT+'/inputs.json'); prediction = a.json(SROOT+'/predictions.json')
        segment_status = a.json(SROOT+'/status.json')
        if (capture['identity'] != CAP_ID or digest(capture['frozen']) != CAP_ID
                or segment['identity'] != SEG_ID or digest(segment['frozen']) != SEG_ID
                or capture_status['status'] != 'capture_verified' or segment_status['status'] != 'analysis_complete'
                or capture_status['failed_checks'] or segment_status['numerical_failed_checks']
                or not capture_status['protected_inputs_unchanged'] or not segment_status['protected_inputs_unchanged']):
            raise ValueError('historical cache status / input identity not verified')
        if hashlib.sha256(a.bytes(CROOT+'/inputs.json')).hexdigest() != capture_status['inputs_sha256']:
            raise ValueError('capture input snapshot mismatch')
        if segment['frozen']['capture_inputs_sha256'] != capture_status['inputs_sha256']:
            raise ValueError('SEGMENT/capture parent mismatch')
        if segment['samples'] != capture['samples'] or segment['frozen']['supports'] != capture['supports']:
            raise ValueError('SEGMENT/capture samples or support order differ')
        for name in ('inputs.json', 'predictions.json', 'status.json', 'numerical_verification.csv'):
            with Archive(paths['review_archive']) as review:
                if a.bytes(SROOT+'/'+name) != review.bytes(SROOT+'/'+name):
                    raise ValueError('historical review differs from raw archive')
        raw_checks = a.csv(CROOT+'/capture_verification.csv') + a.csv(SROOT+'/numerical_verification.csv')
        if not raw_checks or any(r['verified'].lower() not in ('true', '1') for r in raw_checks):
            raise ValueError('historical numerical verification failed')
    formal = capture['frozen']['formal_sha256']
    for name, expected in formal.items():
        # Recorded keys are portable filenames in the existing capture protocol.
        file = paths['formal_dir']/safe(name)
        if sha(file) != expected: raise ValueError(f'formal input modified: {name}')
        checksums['formal/'+name] = expected
    for name, expected in CH1_PINS.items():
        if sha(paths['ch1_dir']/name) != expected: raise ValueError(f'frozen CH1 input modified: {name}')
        checksums['ch1/'+name] = expected
    state = json_file(paths['ch1_dir']/'image_calibration_state.json')
    if digest({k:v for k,v in state.items() if k != 'state_sha256'}) != state['state_sha256']:
        raise ValueError('CH1 image MAD identity invalid')
    if state['patch_mad_frozen_sha256'] != checksums['formal/source_mad_state.npz'] or state['layers'] != list(LAYERS):
        raise ValueError('CH1 patch MAD/layers mismatch')
    if json_file(paths['ch1_dir']/'status.json')['status'] != 'evaluation_complete':
        raise ValueError('CH1 calibration output not verified')
    with np.load(paths['formal_dir']/'source_mad_state.npz', allow_pickle=False) as f:
        medians, scales = f['medians'].copy(), f['scales'].copy()
    samples = capture['samples']; supports = capture['supports']
    if (len(samples) != 297 or len({s['sample_id'] for s in samples}) != 297
            or sum(s['label'] == 0 for s in samples) != 85 or sum(s['label'] == 1 for s in samples) != 212
            or len(supports) != 8 or any(s['relative_path'] in {x['relative_path'] for x in samples} for s in supports)
            or capture['distance_unit'] != PROTOCOL['distance_unit']):
        raise ValueError('fixed background scope / support exclusion / distance unit mismatch')
    formal_rows=read_csv((paths['formal_dir']/'sample_scores.csv').read_bytes())
    lookup={(r['sample_id'],r['method']):r for r in formal_rows if r['domain']=='background'}
    for sample in samples:
        for method,key in (('MAD_EQUAL','source_score'),('TARGET_MATCHED_K8_MAD','union_score')):
            row=lookup[(sample['sample_id'],method)]
            if (row['relative_path']!=sample['relative_path'] or int(row['label'])!=sample['label']
                    or float(row['image_score'])!=sample[key]):
                raise ValueError('capture formal scores/scope differ from original few-shot CSV')
    captures = {r['sample_id']: r for r in capture_status['records']}
    segments = {(r['sample_id'], r['layer']): r for r in prediction['records']}
    if (set(captures) != {s['sample_id'] for s in samples}
            or set(segments) != {(s['sample_id'], l) for s in samples for l in LAYERS}
            or len(capture_status['records']) != 297 or len(prediction['records']) != 1188):
        raise ValueError('incomplete original cache manifests')
    if check_data:
        needed=dict(capture['gt_sha256'])
        needed.update({s['relative_path']:capture['image_sha256'][s['relative_path']] for s in capture['selected_cases']})
        if paths['data'].is_file():
            if sha(paths['data']) != DATA_SHA: raise ValueError('dataset ZIP differs from frozen asset')
            import zipfile
            with zipfile.ZipFile(paths['data']) as z:
                names = set(z.namelist())
                if any('AeBAD/AeBAD_S/'+safe(n) not in names for n in needed):
                    raise ValueError('dataset missing fixed GT or case RGB')
        else:
            root=dataset_root(paths['data'])
            for name,checksum in needed.items():
                if sha(root/safe(name))!=checksum: raise ValueError('dataset raster differs: '+name)
    checksums['dataset_raster_manifest']=digest(dict(GT=capture['gt_sha256'],RGB=capture['image_sha256']))
    frozen = dict(protocol=PROTOCOL, assets=checksums, source_capture=CAP_ID,
                  source_segment=SEG_ID, CH1_state=state['state_sha256'], code=code_hashes())
    return dict(paths=paths, capture=capture, captures=captures, segments=segments,raw_index=raw_index,
                samples=samples, state=state, medians=medians, scales=scales,
                manifest=frozen, identity=digest(frozen))


def load_image(ctx, archive, sample):
    sid = sample['sample_id']; record = ctx['captures'][sid]
    raw, meta = archive.npz(CROOT+'/'+record['file'], record['sha256'])
    validate_arrays(raw, meta)
    if (meta['identity'] != CAP_ID or meta['sample_id'] != sid or meta['relative_path'] != sample['relative_path']
            or meta['supports'] != ctx['capture']['supports']
            or meta['image_sha256'] != ctx['capture']['image_sha256'][sample['relative_path']]):
        raise ValueError('capture query/support metadata mismatch')
    grid = tuple(meta['grid'])
    if min(grid) < 3 or any(tuple(s['grid']) != grid for s in meta['supports']):
        raise ValueError('unsupported query/support native grid scale')
    # Grid dimensions alone are insufficient: physical resize/patch scales must match.
    geometry = meta['geometry']
    for s in meta['supports']:
        if any(s['geometry'][k] != geometry[k] for k in
               ('processed_height', 'processed_width', 'model_patch_size')):
            raise ValueError('query/support physical patch scales differ')
    segments = {}
    for layer in LAYERS:
        r = ctx['segments'][(sid, layer)]
        a, m = archive.npz(SROOT+'/'+r['file'], r['sha256'])
        if (m['format'] != 'bounded_target_reference_layer_v1' or m['identity'] != SEG_ID
                or m['sample_id'] != sid or m['relative_path'] != sample['relative_path'] or m['layer'] != layer
                or m['capture_raw_sha256'] != record['sha256'] or tuple(m['grid']) != grid
                or m['output_shape'] != meta['output_shape']):
            raise ValueError('SEGMENT original input identity mismatch')
        n = len(raw['positions'])
        if (a['positions'].dtype != np.int64 or not np.array_equal(np.sort(a['positions']), np.arange(n))
                or a['endpoint_support'].shape != (2,n) or a['endpoint_support'].dtype != np.int8
                or a['endpoint_index'].shape != (2,n) or a['endpoint_index'].dtype != np.int64
                or a['segment_distance'].shape != (n,) or a['segment_distance'].dtype != np.float32
                or not np.isfinite(a['segment_distance']).all() or np.any(a['segment_distance'] < 0)
                or a['alpha'].shape != (n,) or not np.isfinite(a['alpha']).all()
                or np.any((a['alpha'] < 0) | (a['alpha'] > 1))
                or a['endpoint_length'].shape != (n,)
                or not np.isfinite(a['endpoint_length']).all() or np.any(a['endpoint_length'] < 0)):
            raise ValueError('invalid saved SEGMENT arrays')
        # Check endpoint identity WITHOUT computing any gate or new prediction.
        from .gates import scatter
        support = scatter(a['endpoint_support'], a['positions'], grid).reshape(2,-1)[:,raw['positions']]
        index = scatter(a['endpoint_index'], a['positions'], grid).reshape(2,-1)[:,raw['positions']]
        off = LAYERS.index(layer)
        chosen = np.argsort(raw['target_distance'][:,off], axis=0, kind='stable')[:2]
        if not np.array_equal(support, chosen+1) or not np.array_equal(index, raw['target_index'][chosen,off,np.arange(n)]):
            raise ValueError('saved endpoints differ from original stable Top2')
        u = np.minimum(raw['source_distance'][off], raw['target_distance'][:,off].min(0))
        s = scatter(a['segment_distance'], a['positions'], grid).ravel()[raw['positions']]
        if np.any(s > u+ATOL+RTOL*np.abs(u)): raise ValueError('saved SEGMENT exceeds original union')
        segments[layer] = a
    return raw, meta, segments


def protect_output(ctx, output):
    p = Path(output).resolve()
    if p.exists(): raise ValueError('fresh output directory required; no overwrite or implicit resume')
    for source in ctx['paths'].values():
        if p == source or p in source.parents or source in p.parents:
            raise ValueError('output overlaps a protected input')
    if p.drive and p.drive.upper() != 'G:': raise ValueError('local outputs/scratch must be on G:')
    return p
