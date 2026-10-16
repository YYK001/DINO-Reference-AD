"""Read-only, portable input mapping; recorded server paths are never followed."""
from __future__ import annotations

import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic.inputs import (
    PARENT, PARENT_ID, PINS, PREFIX, digest, sha, read_bundle, validate_raw,
)
from DINOv3.MADEqual.aebad_s_target_fewshot import replay12 as replay
from DINOv3.MADEqual.aebad_s_full_p0.dataset import IMAGE_SUFFIXES
from .scoring import LAYERS

TOOL = 'aebad_s_ch1_scoring_inheritance_v1'
FORMAL_PINS = {
    'memory_manifest.json': '30b2c331b9e8e30f37ebfb2b44cc6c3c36f170e34810b51542e257b56ead40b0',
    'calibration_stats.csv': 'd49d5a5a08174e53322db354bf09e3649ac73d0b75b9d5da6ddad5c0d7fdaf26',
    'source_mad_state.npz': PINS['mad'],
    'sample_scores.csv': '9d7a57adcbc7ef715b9f8ca29a7627e93710acd1319ea0fd1365e0cd3a993d76',
    'support_manifest.csv': 'd4889e925d8700faf4b06ddb493c096d08a44240e87adbca09e5d2d0b1ba1983',
    'domain_metrics.csv': '67aa9e573e5a58f32dceee3776e1cc073b489a7e94ac87498057d69871febc89',
}
MEMORY_SHA = '1b9e00c85fd6be35ba365d378b1da7fb2a673400319a256bb62408fe6e001717'
PARENT_INPUT_SHA = '25ae7701d6cf1045c76a128a272c3f5eeddc7b6c984d510faaeb6f305e1e732b'
# Filled from the approved ZIP's 521 ordered train/good RGB file identities;
# unrelated AppleDouble/__MACOSX entries are excluded by the original discovery rule.
SOURCE_MANIFEST_SHA = '610d0a3d328878dacdba44e0d53a888d5eb3955712fad6082a1469222b0fbe9f'
SOURCE_PATHS_SHA = '59ec76c3375eb42b783a4e3f5ad00a297f616544e25fc4cd71e575297efec978'
CODE_PATHS = (
    'DINOv3/MADEqual/aebad_s_full_p0/dataset.py',
    'DINOv3/relation_reliability/datasets.py',
    'DINOv3/relation_reliability/superadd_host.py',
    'DINOv3/relation_oracle_cache/extract.py',
    'DINOv3/MADEqual/image_granularity_calibration/scoring.py',
    'DINOv3/MADEqual/connected_region/scoring.py',
    'DINOv3/MADEqual/aebad_s_target_fewshot/runner.py',
    'DINOv3/MADEqual/aebad_s_sparc_p0/runtime.py',
    'DINOv3/MADEqual/aebad_s_projection_layer_diagnostic/numerics.py',
)


class SourceData:
    def __init__(self, path):
        self.path = Path(path)
        self.zip = zipfile.ZipFile(self.path) if self.path.is_file() else None
        if self.zip:
            names = self.zip.namelist()
            if len(names) != len(set(names)):
                raise ValueError('duplicate ZIP entries')
            self.paths = [n[len(PREFIX):] for n in names if n.startswith(PREFIX) and not n.endswith('/')]
        else:
            from DINOv3.MADEqual.aebad_s_full_p0.dataset import discover_phase_a
            self.path, train, _ = discover_phase_a(self.path, formal=False)
            self.paths = [Path(r.path).relative_to(self.path).as_posix() for r in train]

    def read(self, relative):
        replay.safe_relative(relative)
        return self.zip.read(PREFIX + relative) if self.zip else (self.path / relative).read_bytes()

    def source_paths(self):
        return sorted(p for p in self.paths if p.startswith('train/good/')
                       and Path(p).suffix.lower() in IMAGE_SUFFIXES
                       and not Path(p).name.startswith('._') and '__MACOSX' not in Path(p).parts)

    def manifest(self):
        paths = self.source_paths()
        if len(paths) != 521 or len(set(paths)) != 521:
            raise ValueError(f'expected 521 unique ordered Source-normal images, got {len(paths)}')
        return ordered_manifest(paths, self.read)

    def close(self):
        if self.zip:
            self.zip.close()


def ordered_manifest(paths, read):
    """Same lexicographic relative path order and zero-based modulo as discovery/split."""
    if paths != sorted(set(paths)):
        raise ValueError('Source paths must be unique and sorted by original relative path')
    rows = []
    for index, relative in enumerate(paths):
        blob = read(relative)
        with Image.open(io.BytesIO(blob)) as image:
            width, height = image.size
        rows.append(dict(source_index=index, relative_path=relative,
                         role='calibration' if index % 8 == 0 else 'memory_fit',
                         calibration_index=index // 8 + 1 if index % 8 == 0 else None,
                         image_sha256=hashlib.sha256(blob).hexdigest(),
                         image_height=height, image_width=width,
                         output_shape=[height // 4, width // 4]))
    return rows


def resolve(args):
    root = Path(args.project_root).resolve()
    formal = Path(args.results_dir or root / 'server_archives/aebad_s_target_fewshot_seed42_v1_formal_results/aebad_s_target_fewshot_seed42_v1').resolve()
    return dict(project=root, formal=formal,
                parent=Path(args.parent_dir or args.parent_archive or root / 'downloads' / f'{PARENT}_results_20261014.tar.gz').resolve(),
                data=Path(args.data or root / 'datasets/AeBAD.zip').resolve(),
                memory=Path(args.memory or formal / 'official_memory.pt').resolve(),
                checkpoint=Path(args.checkpoint or root / 'external_baselines/SuperADD/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth').resolve(),
                official=Path(args.official_root or root / 'external_baselines/SuperADD').resolve(),
                dinov3=Path(args.dinov3_root or root.parent / 'dinov3').resolve(),
                calibration=Path(args.source_calibration_dir).resolve() if args.source_calibration_dir else None,
                output=Path(args.output_dir).resolve() if args.output_dir else None)


def guard(paths):
    output = paths['output']
    if output is None:
        return
    for key in ('formal', 'parent', 'data', 'memory', 'checkpoint', 'official', 'dinov3', 'calibration'):
        path = paths[key]
        if path and (output == path or output.is_relative_to(path) or path.is_relative_to(output)):
            raise ValueError(f'output overlaps protected {key}: {path}')


def load_context(paths, *, test_raw=False, recovery=False):
    guard(paths)
    formal = paths['formal']
    for name, expected in FORMAL_PINS.items():
        if sha(formal / name) != expected:
            raise ValueError('formal input SHA mismatch: ' + name)
    manifest = json.loads((formal / 'memory_manifest.json').read_text(encoding='utf-8'))
    if (manifest['protocol_id'] != replay.PROTOCOL_ID or
            [manifest[k] for k in ('builder_input_count', 'memory_fit_count', 'calibration_count', 'threshold_fraction')] != [521, 455, 66, 8]):
        raise ValueError('original Source split/memory record mismatch')
    parent = paths['parent']
    bundle = None
    if parent.is_dir():
        if sha(parent / 'inputs.json') != PARENT_INPUT_SHA:
            raise ValueError('server parent inputs differ from approved local package')
        original = json.loads((parent / 'inputs.json').read_text(encoding='utf-8'))
        if test_raw:
            raise ValueError('CPU evaluation requires the approved local archive, not an unchecked directory')
    else:
        if sha(parent) != PINS['parent']:
            raise ValueError('approved 297-distance archive SHA mismatch')
        bundle = read_bundle(parent, PARENT)
        original = bundle['data']['inputs.json']
    if original['identity'] != PARENT_ID or digest(original['frozen']) != PARENT_ID:
        raise ValueError('parent identity mismatch')
    if original['frozen']['host_config'] != manifest['host_config']:
        raise ValueError('host differs from original memory manifest')
    data = SourceData(paths['data'])
    try:
        expected_paths = json.loads(Path(__file__).with_name('source_scope.json').read_text(encoding='utf-8'))
        if digest(expected_paths) != SOURCE_PATHS_SHA:
            raise ValueError('frozen Source filename scope asset modified')
        actual_paths = data.source_paths()
        if actual_paths != expected_paths:
            raise ValueError('Source scope differs; missing=' + json.dumps(sorted(set(expected_paths) - set(actual_paths)))
                             + '; extra=' + json.dumps(sorted(set(actual_paths) - set(expected_paths))))
        source = data.manifest()
        if digest(source) != SOURCE_MANIFEST_SHA:
            raise ValueError('Source 521-image order/identity differs from frozen local dataset')
        cap = original['capture']
        replay.select_cases(replay.read_csv(formal / 'sample_scores.csv'))
        supports = replay.select_support(replay.read_csv(formal / 'support_manifest.csv'),
                                         replay.select_cases(replay.read_csv(formal / 'sample_scores.csv')))
        supports = [s for s in supports if s['branch'] == 'target_matched']
        samples = cap['samples']
        ids = [s['sample_id'] for s in samples]
        if (len(ids) != 297 or len(set(ids)) != 297 or sum(s['label'] == 0 for s in samples) != 85
                or sum(s['label'] == 1 for s in samples) != 212
                or [s['relative_path'] for s in cap['supports']] != [s['relative_path'] for s in supports]
                or {s['relative_path'] for s in samples} & {s['relative_path'] for s in supports}):
            raise ValueError('original 297-image/8-support scope mismatch')
        formal_scores = replay.read_csv(formal / 'sample_scores.csv')
        for method in replay.METHODS:
            historical = {r['sample_id']: r for r in formal_scores if r['method'] == method and r['domain'] == 'background'}
            if set(historical) != set(ids) or any(
                    historical[s['sample_id']]['relative_path'] != s['relative_path']
                    or int(historical[s['sample_id']]['label']) != s['label'] for s in samples):
                raise ValueError('all 297 frozen identities/labels must match formal scores')
        # No GT reads. Source recovery need not have evaluation RGBs available.
        if not recovery:
            for s in [*samples, *cap['supports']]:
                blob = data.read(s['relative_path'])
                if hashlib.sha256(blob).hexdigest() != cap['image_sha256'][s['relative_path']]:
                    raise ValueError('evaluation image SHA mismatch: ' + s['relative_path'])
                with Image.open(io.BytesIO(blob)) as image:
                    if 'image_height' in s and list(image.size) != [s['image_width'], s['image_height']]:
                        raise ValueError('evaluation image size mismatch')
        if bundle:
            for s in samples:
                raw = bundle['raw'][f"raw/{s['sample_id']}.npz"]
                validate_raw(raw, s, PARENT_ID, cap['image_sha256'][s['relative_path']])
                if raw['metadata']['output_shape'] != [s['image_height'] // 4, s['image_width'] // 4]:
                    raise ValueError('evaluation downscale mismatch')
    finally:
        data.close()
    with np.load(formal / 'source_mad_state.npz', allow_pickle=False) as f:
        if str(f['protocol_id'].item()) != replay.PROTOCOL_ID:
            raise ValueError('patch MAD protocol mismatch')
        medians, scales = f['medians'].copy(), f['scales'].copy()
    if any(a.dtype != np.float32 or a.shape != (4,) or not np.isfinite(a).all() for a in (medians, scales)) or np.any(scales <= 0):
        raise ValueError('invalid original patch MAD')
    code = {rel: sha(paths['project'] / rel) for rel in CODE_PATHS}
    for path in sorted(Path(__file__).parent.glob('*.py')):
        code[path.relative_to(paths['project']).as_posix()] = sha(path)
    scope = Path(__file__).with_name('source_scope.json')
    code[scope.relative_to(paths['project']).as_posix()] = sha(scope)
    pins = cap['frozen']['input_code_sha256']
    for rel, key in (('DINOv3/relation_reliability/superadd_host.py', 'code_host'),
                     ('DINOv3/relation_oracle_cache/extract.py', 'code_mad'),
                     ('DINOv3/MADEqual/aebad_s_target_fewshot/runner.py', 'code_runner')):
        if code[rel] != pins[key]:
            raise ValueError('frozen host/MAD/extraction code changed: ' + rel)
    frozen = dict(tool=TOOL, protocol_id=replay.PROTOCOL_ID, source_manifest=source,
                  assets={**FORMAL_PINS, 'memory': MEMORY_SHA, 'checkpoint': replay.CHECKPOINT_SHA256},
                  host_config=manifest['host_config'], precision=original['frozen']['precision'],
                  query_protocol=cap['frozen']['query_protocol'], code=code,
                  layers=list(LAYERS), distance_divisor=1280,
                  scoring=dict(window=3, stride=1, weights=[.5, .5], epsilon=1e-8))
    return dict(paths=paths, original=original, bundle=bundle, source=source,
                frozen=frozen, identity=digest(frozen), medians=medians, scales=scales)


def recovery_assets(context):
    p = context['paths']
    pins = context['original']['capture']['frozen']['input_code_sha256']
    required = [(p['memory'], MEMORY_SHA), (p['checkpoint'], replay.CHECKPOINT_SHA256),
                (p['official'] / 'config.json', pins['official_config']),
                (p['official'] / 'tracks/industrial/src/industrial/model.py', pins['official_model']),
                (p['dinov3'] / 'hubconf.py', pins['dinov3_hub'])]
    errors = []
    for path, expected in required:
        if not path.is_file():
            errors.append('missing recovery asset: ' + str(path))
        elif sha(path) != expected:
            errors.append('recovery asset SHA mismatch: ' + str(path))
    return errors


def preflight(args):
    paths = resolve(args)
    guard(paths)
    report = dict(tool=TOOL, status='blocked', errors=[], missing_assets=[],
                  torch_imported='torch' in sys.modules, model_loaded=False,
                  recovery_executed=False, evaluation_executed=False,
                  paths={k: str(v) if v else None for k, v in paths.items()},
                  expected_source=521, memory_fit=455, calibration=66, evaluation_images=297)
    required = [paths['formal'] / n for n in FORMAL_PINS] + [paths['parent'], paths['data']]
    report['missing_assets'] = [str(p) for p in required if not p.exists()]
    if report['missing_assets']:
        return report, None
    try:
        recovery = args.for_recovery or args.stage == 'recover-calibration'
        ctx = load_context(paths, recovery=recovery)
        report['identity'] = ctx['identity']
        report['source_manifest_sha256'] = digest(ctx['source'])
        if recovery:
            report['errors'].extend(recovery_assets(ctx))
        elif paths['calibration'] is None:
            report['missing_assets'] = [f'source-calibration-dir/raw/{i:03d}.npz' for i in range(1, 67)]
            report['errors'].append('66 original caches deleted; explicit server recovery required')
        else:
            from .recovery import validate_recovery
            validate_recovery(paths['calibration'], ctx, complete=True)
        if not report['errors'] and not report['missing_assets']:
            report['status'] = 'filesystem_preflight_passed'
        ctx['cpu_workers'] = args.cpu_workers
        report['torch_imported'] = 'torch' in sys.modules
        if report['torch_imported']:
            raise RuntimeError('preflight unexpectedly imported Torch')
        return report, ctx
    except (OSError, ValueError, KeyError, RuntimeError, zipfile.BadZipFile) as error:
        report['status'] = 'blocked'
        report['errors'].append(f'{type(error).__name__}: {error}')
    report['torch_imported'] = 'torch' in sys.modules
    return report, None
