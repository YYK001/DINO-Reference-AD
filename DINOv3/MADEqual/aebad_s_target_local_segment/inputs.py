"""Read-only offline inputs. No banks, checkpoints, model or extraction fallback."""
from __future__ import annotations

import importlib.metadata
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from DINOv3.MADEqual.aebad_s_target_fewshot import replay12 as replay
from DINOv3.MADEqual.aebad_s_union_support_loo.inputs import read_capture, guard_output as guard_roots, verify_files
from DINOv3.MADEqual.aebad_s_union_support_loo.runner import formal_baselines
from .geometry import LAYERS, DIMENSION

TOOL = 'aebad_s_target_local_segment_v1'
CAPTURE_NAME = 'aebad_s_union_support_loo_B_20261007_230639'


def guard_output(paths: dict[str,Path]) -> None:
    guard_roots(paths)
    prior=paths['output']/'preflight.json'
    if prior.is_file() and json.loads(prior.read_text(encoding='utf-8')).get('tool') != TOOL:
        raise ValueError('existing preflight belongs to another tool; refusing historical directory')


def paths_for(args) -> dict[str, Path]:
    project = Path(args.project_root).resolve()
    base = project / 'DINOv3/outputs/mad_equal'
    results = Path(args.results_dir or base / 'aebad_s_target_fewshot_seed42_v1').resolve()
    return dict(project=project, results=results,
        capture=Path(args.capture_dir or base / CAPTURE_NAME).resolve(),
        support=Path(args.support_cache_dir or base / 'aebad_s_union_background_replay12_seed42_v1/support_features/target_matched/background').resolve(),
        mad=Path(args.mad or results / 'source_mad_state.npz').resolve(),
        data=Path(args.data).resolve(),
        output=Path(args.output_dir or base / 'aebad_s_target_local_segment_seed42_v1').resolve())


def data_root(supplied: Path) -> Path:
    candidates = [p for p in (supplied, supplied/'AeBAD_S', supplied/'AeBAD/AeBAD_S')
                  if (p/'ground_truth').is_dir() and (p/'test').is_dir()]
    if len(candidates) != 1:
        raise ValueError('data must resolve one test/ground_truth root; no train/good required')
    return candidates[0].resolve()


def gt_relative(relative: str) -> str:
    from DINOv3.MADEqual.aebad_s_union_diagnostics.analysis import gt_path
    return replay.safe_relative(gt_path(relative))


def support_manifest_matches(supports: list[dict], manifest: Path) -> None:
    rows = sorted((r for r in replay.read_csv(manifest)
                   if r['branch'] == 'target_matched' and r['domain'] == 'background'),
                  key=lambda r: int(r['support_index']))
    if (len(supports) != 8 or [r['support_index'] for r in supports] != list(range(1, 9))
            or len(rows) != 8 or len({r['relative_path'] for r in supports}) != 8):
        raise ValueError('fixed eight Target supports/order required')
    for row, recorded in zip(rows, supports, strict=True):
        if (int(row['support_index']) != recorded['support_index']
                or row['relative_path'] != recorded['relative_path'] or int(row['label']) != 0
                or not replay.safe_relative(row['relative_path']).startswith('test/good/background/')):
            raise ValueError('support identity/order differs from original manifest')


def load_support_layer(paths: dict, supports: list[dict], layer: int) -> list[np.ndarray]:
    """Only the NPZ read contract of SupportFeatures; never import its extractor.

    The existing support module eagerly imports Torch/extraction. Reusing its
    header checker and exact stored format keeps this read path NumPy-only.
    One layer of eight banks is resident, and endpoints are gathered in chunks.
    """
    if layer not in LAYERS:
        raise ValueError('unexpected support layer')
    banks = []
    for row in supports:
        path = paths['support'] / f"{row['support_index']:02d}.npz"
        if replay.sha256(path) != row['feature_sha256']:
            raise ValueError('support changed since capture; no new approval')
        replay.check_support_header(path, row['relative_path'])
        with np.load(path, allow_pickle=False) as payload:
            grid = tuple(int(v) for v in payload['grid'])
            bank = payload[f'layer_{layer}'].copy()
        if (list(grid) != row['grid'] or bank.dtype != np.float32
                or bank.shape != (int(np.prod(grid)), DIMENSION) or not np.isfinite(bank).all()):
            raise ValueError('support feature grid/dtype/finiteness differs')
        banks.append(bank)
    return banks


def preflight(args) -> dict:
    paths = paths_for(args)
    guard_output(paths)
    check = dict(tool=TOOL, mode=args.mode, scope='background', paths={k:str(v) for k,v in paths.items()},
                 errors=[], required_files=[], samples=[], supports=[],
                 model_inference=False, feature_extraction=False, torch_imported=False,
                 pixel_backend='original full-protocol ADEval CUDA, nstrips=200, FPR<=0.3',
                 cuda_runtime_check='deferred_to_explicit_analyze',
                 support_finiteness='deferred_to_explicit_analyze', expected_raw_images=297)
    check['dependency_distributions'] = {}
    for package in ('numpy','Pillow','scipy','torch','torchvision','scikit-learn','matplotlib','adeval'):
        try: check['dependency_distributions'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: check['dependency_distributions'][package] = None

    def require(kind: str, path: Path, expected: str | None = None):
        if not path.is_file():
            check['errors'].append(f'missing {kind}: {path}')
            return
        checksum = replay.sha256(path)
        if expected is not None and checksum != expected:
            check['errors'].append(f'modified {kind}: {path}')
            return
        check['required_files'].append(dict(kind=kind, path=str(path), sha256=checksum))

    capture_inputs = None
    try:
        if not paths['capture'].is_dir():
            raise ValueError('capture-dir must be the complete B directory, not a check/archive bundle; missing raw does not require model replay')
        capture_inputs, capture_status = read_capture(paths['capture'])
        samples, supports = capture_inputs['samples'], capture_inputs['supports']
        if (capture_inputs['scope'] != 'background' or len(samples) != 297
                or Counter(s['label'] for s in samples) != {0:85,1:212}
                or len({s['sample_id'] for s in samples}) != 297
                or len({s['relative_path'] for s in samples}) != 297
                or any(not re.fullmatch(r'test_\d+', s['sample_id']) for s in samples)
                or {s['relative_path'] for s in samples} & {s['relative_path'] for s in supports}):
            raise ValueError('full heldout 297/85/212 or original exclusions changed')
        if capture_inputs['frozen']['protocol_id'] != replay.PROTOCOL_ID:
            raise ValueError('wrong frozen protocol')
        for sample in samples:
            replay.safe_relative(sample['relative_path'])
        fixed = capture_inputs['selected_cases']
        if [(s['sample_id'],s['relative_path']) for s in fixed] != list(replay.CASES):
            raise ValueError('fixed twelve cases changed')
        for key in ('inputs.json','status.json','capture_verification.csv'):
            require('capture_'+key, paths['capture']/key)
        for record in capture_status['records']:
            require('raw_'+record['sample_id'], paths['capture']/record['file'], record['sha256'])
        check.update(capture_inputs=capture_inputs, capture_status=capture_status,
                     samples=samples, supports=supports, identity=capture_inputs['identity'])
    except (OSError,ValueError,KeyError,IndexError,RuntimeError) as error:
        check['errors'].append(f'capture rejected: {type(error).__name__}: {error}')

    # These are the only formal files needed; no official_memory.pt or weights.
    for key in ('sample_scores.csv','support_manifest.csv','complete.json','memory_manifest.json','source_mad_state.npz'):
        path = paths['mad'] if key == 'source_mad_state.npz' else paths['results']/key
        expected = capture_inputs['frozen']['formal_sha256'][key] if capture_inputs else None
        require(key, path, expected)
    try:
        if paths['mad'].is_file(): replay.check_mad(paths['mad'])
        if capture_inputs:
            formal_baselines(paths['results'], check['samples'])
            support_manifest_matches(check['supports'], paths['results']/'support_manifest.csv')
            frozen_supports = capture_inputs['frozen']['supports']
            for row, original in zip(check['supports'], frozen_supports, strict=True):
                if any(row[k] != original[k] for k in ('support_index','relative_path','feature_sha256','image_sha256')):
                    raise ValueError('support provenance differs from frozen identity')
                path = paths['support']/f"{row['support_index']:02d}.npz"
                require(f"support_{row['support_index']}", path, row['feature_sha256'])
                if path.is_file():
                    replay.check_support_header(path,row['relative_path'])
                    with np.load(path,allow_pickle=False) as payload:
                        if payload['grid'].tolist() != row['grid']: raise ValueError('support grid changed')
            # Only original map dependencies are anchored to capture code hashes.
            for key, rel in (
                ('code_runner','DINOv3/MADEqual/aebad_s_target_fewshot/runner.py'),
                ('code_host','DINOv3/relation_reliability/superadd_host.py'),
                ('code_mad','DINOv3/relation_oracle_cache/extract.py')):
                require(key, paths['project']/rel, capture_inputs['frozen']['input_code_sha256'][key])
    except (OSError,ValueError,KeyError,RuntimeError) as error:
        check['errors'].append(f'formal/support rejected: {type(error).__name__}: {error}')
    if capture_inputs is None:
        for j in range(1,9):
            require(f'support_{j}_identity_unverified_without_capture',paths['support']/f'{j:02d}.npz')

    try:
        root = data_root(paths['data'])
        paths['data'] = root
        check['paths']['data'] = str(root)
        guard_output(paths)
        if capture_inputs:
            for sample in check['samples']:
                if sample['label']:
                    relative = gt_relative(sample['relative_path'])
                    require('GT_'+sample['sample_id'],root/relative,capture_inputs['gt_sha256'][relative])
            for sample in capture_inputs['selected_cases']:
                relative = sample['relative_path']
                require('RGB_'+sample['sample_id'],root/relative,capture_inputs['image_sha256'][relative])
    except (OSError,ValueError,KeyError) as error:
        check['errors'].append(f'data rejected: {type(error).__name__}: {error}')
    for path in Path(__file__).parent.glob('*.py'):
        require('code_local_'+path.stem,path)
    # Record additional evaluation implementation without rewriting history.
    for rel in ('DINOv3/MADEqual/robustad_target_fewshot_p0/runner.py',
                'DINOv3/MADEqual/aebad_s_full_p0/runner.py',
                'DINOv3/relation_reliability/official_metrics.py',
                'DINOv3/MADEqual/aebad_s_union_diagnostics/spatial.py'):
        require('evaluation_code_'+rel,paths['project']/rel)
    check['status'] = 'blocked' if check['errors'] else 'filesystem_preflight_passed'
    return check
