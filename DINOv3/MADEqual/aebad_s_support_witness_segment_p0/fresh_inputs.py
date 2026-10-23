"""New-run inputs rebuilt from RGB, never masquerading as historical captures."""
import hashlib
import io
import zipfile
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

from . import storage as st
from .method import PROTOCOL, LAYERS, BASES
from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.inputs import Data
from DINOv3.MADEqual.aebad_s_union_support_loo.analysis import validate_arrays

REVISION = 'witness_fresh_rgb_seed42_v1'
DOMAINS = ('background', 'illumination', 'view')
CASE_IDS = ('test_0977', 'test_0910', 'test_0887', 'test_0924', 'test_0921',
            'test_0930', 'test_0893', 'test_0926', 'test_0934', 'test_0935',
            'test_0023', 'test_0545')
ROOT = Path(__file__).resolve().parents[3]


def support_key(relative, domain):
    # Exact pure-Python rule in target_fewshot.support.stable_support_key.
    return zlib.crc32(f'42|{domain}|{relative}'.encode()) & 0xffffffff, relative


def select_catalog(source, test):
    """Preserve original global test IDs, ordered train split and CRC32 K8."""
    source = sorted(source, key=lambda r: r['relative_path'])
    test = sorted(test, key=lambda r: r['relative_path'])
    train = [dict(r, source_index=i, calibration=i % 8 == 0) for i, r in enumerate(source)]
    tests = [dict(r, sample_id=f'test_{i:04d}') for i, r in enumerate(test)]
    supports, samples = {}, []
    for d in DOMAINS:
        group = [r for r in tests if r['domain'] == d]
        normals = sorted((r for r in group if not r['label']),
                         key=lambda r: support_key(r['relative_path'], d))
        if len(normals) < 8:
            raise ValueError('fewer than eight normal supports: ' + d)
        supports[d] = [dict(r, support_index=i) for i, r in enumerate(normals[:8], 1)]
        excluded = {r['relative_path'] for r in supports[d]}
        samples.extend(r for r in group if r['relative_path'] not in excluded)
    return dict(source=train, supports=supports, samples=samples)


def raster_row(path, root):
    relative = path.relative_to(root).as_posix()
    blob = path.read_bytes()
    with Image.open(io.BytesIO(blob)) as image:
        width, height = image.size
    row = dict(relative_path=relative, image_sha256=hashlib.sha256(blob).hexdigest(),
               image_height=height, image_width=width)
    if relative.startswith('test/'):
        parts = Path(relative).parts
        row.update(domain=parts[2], label=int(parts[1] != 'good'))
        if row['label']:
            gt = root / relative.replace('test/', 'ground_truth/', 1)
            row['gt_sha256'] = st.sha(gt)  # Bytes only; never decode GT here.
    return row


def validate_catalog(catalog):
    if len(catalog['source']) != 521 or sum(r['calibration'] for r in catalog['source']) != 66:
        raise ValueError('original 521 Source / 66 calibration split required')
    for d in DOMAINS:
        rows = [r for r in catalog['samples'] if r['domain'] == d]
        if [len(rows), sum(not r['label'] for r in rows), sum(r['label'] for r in rows)] != PROTOCOL['populations'][d]:
            raise ValueError('original full evaluation population differs: ' + d)
        support = catalog['supports'][d]
        if [r['support_index'] for r in support] != list(range(1, 9)) or any(r['label'] for r in support):
            raise ValueError('ordered eight normal supports required')
        if {r['relative_path'] for r in rows} & {r['relative_path'] for r in support}:
            raise ValueError('support entered evaluation')
    # Check against code-shipped original image catalogs, not old score values.
    cross = st.read_json(ROOT / 'DINOv3/MADEqual/aebad_s_segment_pair_crossdomain_p0/frozen_catalog.json')['domains']
    for d in DOMAINS[1:]:
        for key in ('samples', 'supports'):
            actual = catalog['supports'][d] if key == 'supports' else [r for r in catalog['samples'] if r['domain'] == d]
            fields = ('relative_path', 'image_sha256', 'support_index') if key == 'supports' else ('relative_path', 'image_sha256', 'sample_id', 'label')
            if [tuple(r[k] for k in fields) for r in actual] != [tuple(r[k] for k in fields) for r in cross[d][key]]:
                raise ValueError('original ' + d + ' ' + key + ' identities/order differ')


def prepare_data(run, data, cache):
    """Explicit AeBAD-S-only extraction; no model and no historical result inputs."""
    run, data, cache = map(Path, (run, data, cache))
    destination = run / 'fresh_catalog.json'
    if destination.exists():
        catalog = st.read_json(destination)
        seal = st.read_json(run / 'fresh_catalog.seal.json')
        if st.sha(destination) != seal['sha256']:
            raise ValueError('new-run catalog modified')
        return catalog
    if data.is_file():
        root = cache / 'dataset/AeBAD_S'
        root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(data) as archive:
            prefix = 'AeBAD/AeBAD_S/'
            for item in archive.infolist():
                if not item.filename.startswith(prefix) or item.is_dir():
                    continue
                rel = Path(item.filename[len(prefix):])
                if rel.is_absolute() or '..' in rel.parts:
                    raise ValueError('unsafe dataset ZIP member')
                target = root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                # Read ZIP entries with their CRC check, including on resumed extraction.
                blob = archive.read(item)
                if target.exists():
                    if target.read_bytes() != blob:
                        raise ValueError('existing extracted image differs: ' + str(target))
                else:
                    target.write_bytes(blob)
    else:
        from DINOv3.MADEqual.aebad_s_full_p0.dataset import find_aebad_s_root
        root = find_aebad_s_root(data)
    from DINOv3.MADEqual.aebad_s_full_p0.dataset import discover_phase_a
    root, source, locators = discover_phase_a(root, formal=True)
    catalog = select_catalog([raster_row(Path(r.path), root) for r in source],
                             [raster_row(Path(r.path), root) for r in locators])
    validate_catalog(catalog)
    catalog.update(revision=REVISION, data_root=str(root.resolve()),
                   source_rule='original sorted 521; index%8==0 excluded from bank and used for calibration',
                   support_rule='original CRC32(42|domain|relative_path), then path; first8')
    st.write_json(destination, catalog)
    st.write_json(run / 'fresh_catalog.seal.json', dict(sha256=st.sha(destination)))
    return catalog


def setup_identity(run):
    return st.read_json(Path(run) / 'setup.json')['identity']


def asset(path, identity):
    record = st.restore(path, identity)
    if record is None:
        raise FileNotFoundError(path)
    return record


class Bundle:
    """Adapter for unchanged WITNESS neighbor/LOO/prediction kernels."""
    def __init__(self, run):
        self.run = Path(run)
        self.protocol = dict(PROTOCOL, run_mode=REVISION, historical_metrics_reused=False)
        setup = st.read_json(self.run / 'setup.json')
        self.capture_identity = setup['identity']
        self.catalog = st.read_json(self.run / 'fresh_catalog.json')
        if st.sha(self.run / 'fresh_catalog.json') != setup['catalog_sha256']:
            raise ValueError('catalog changed after input freeze')
        validate_catalog(self.catalog)
        self.samples, self.supports = self.catalog['samples'], self.catalog['supports']
        self.paths = dict(data=Path(self.catalog['data_root']))
        base = Path(setup['cache'])
        patch, _ = st.arrays(base / 'source_patch_mad.npz', self.capture_identity)
        self.medians, self.scales = patch['medians'], patch['scales']
        self.state = st.read_json(self.run / 'image_calibration_state.json')
        if self.state['state_sha256'] != st.digest({k:v for k,v in self.state.items() if k != 'state_sha256'}):
            raise ValueError('image calibration state modified')
        cm = st.read_json(self.run / 'capture_manifest.json')
        seal = st.read_json(self.run / 'capture_manifest.seal.json')
        if cm['status'] != 'capture_complete' or cm['identity'] != self.capture_identity or st.sha(self.run / 'capture_manifest.json') != seal['sha256']:
            raise ValueError('complete sealed fresh capture required')
        self.records = {(r['domain'],r['sample_id']):r for r in cm['records']}
        expected_keys = {(s['domain'],s['sample_id']) for s in self.samples}
        if len(cm['records']) != len(expected_keys) or set(self.records) != expected_keys:
            raise ValueError('fresh capture sample coverage differs')
        self.expected, self.locations = {}, {}
        self.references = {}
        for r in cm['records']:
            for b in BASES:
                sc = r['scores'][b]
                self.references[r['domain'],r['sample_id'],b] = dict(image_score=sc['CH1'], CURRENT_check_score=sc['CURRENT'])
        for d in DOMAINS:
            self.expected[d] = []
            for row in self.supports[d]:
                path = base / 'supports' / d / f'{row["support_index"]:03d}.npz'
                record = asset(path, self.capture_identity)
                value = dict(row, feature_sha256=record['sha256'], grid=record['metadata']['grid'])
                self.expected[d].append(value)
                self.locations[d,row['support_index']] = str(path)
        self.bg = dict(capture=dict(selected_cases=[s for s in self.samples if s['domain']=='background' and s['sample_id'] in CASE_IDS]))
        self.frozen = dict(protocol=PROTOCOL, mode=REVISION, setup=setup,
            capture_manifest_sha256=seal['sha256'], image_state=self.state,
            patch_mad_sha256=asset(base/'source_patch_mad.npz',self.capture_identity)['sha256'],
            # Offline stages use the already recorded Source identity, not its
            # 2 GiB entity. The model stages validate actual bank bytes on load.
            source_bank_sha256=st.read_json((base/'source_bank.npz').with_suffix('.json'))['sha256'],
            supports=self.expected)
        self.identity = st.digest(self.frozen)
        self.protected_paths = [self.paths['data'], base/'source_bank.npz', self.run/'fresh_catalog.json', self.run/'capture_manifest.json']
        self.cache = base

    def load(self, sample):
        rec = self.records[sample['domain'],sample['sample_id']]
        path = self.cache / rec['file']
        arrays, record = st.arrays(path, self.capture_identity)
        if record['sha256'] != rec['sha256'] or st.sha(path.with_suffix('.json')) != rec['record_sha256']:
            raise ValueError('fresh capture modified after freeze')
        meta = record['metadata']
        if any(meta[k] != sample[k] for k in ('domain','sample_id','relative_path','image_sha256')):
            raise ValueError('fresh capture image identity differs')
        validate_arrays(arrays, meta)
        union = np.minimum(arrays['source_distance'], arrays['target_distance'].min(0))
        if not np.array_equal(union, arrays['union_distance']):
            raise ValueError('fresh UNION differs from full K8 min')
        return arrays, meta, dict(fresh_capture_sha256=rec['sha256'], capture_identity=self.capture_identity)
