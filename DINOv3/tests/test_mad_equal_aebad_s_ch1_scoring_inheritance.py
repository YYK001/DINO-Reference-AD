"""CPU production math/ledger/evaluation tests; model boundary stays unexecuted."""
from __future__ import annotations

import ast
import copy
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.metrics import average_precision_score, roc_auc_score

from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance import scoring as s, inputs, recovery, evaluation, runner
from DINOv3.MADEqual.aebad_s_union_diagnostics.spatial import interpolation_stencil
from DINOv3.relation_reliability.datasets import Record, split_source_normal
from DINOv3.MADEqual.aebad_s_full_p0.dataset import discover_phase_a

ROOT = Path(__file__).resolve().parents[2]


def original_functions():
    """Execute actual original function bodies, excluding Torch-dependent imports.

    This tests the exact NumPy/SciPy function bodies, not the real Torch resize.
    Interpolation has a separate scalar-stencil reference below and recovery must
    additionally check it against original Torch statistics on the server.
    """
    names = {'fit_image_mad', 'local_mean', 'calibrated_branch', 'fused_score'}
    ns = dict(np=np, ndimage=ndimage, LAYERS=s.LAYERS, ECDF_EPSILON=1e-8)
    for relative in ('image_granularity_calibration/scoring.py', 'connected_region/scoring.py'):
        tree = ast.parse((ROOT / 'DINOv3/MADEqual' / relative).read_text(encoding='utf-8'))
        body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        exec(compile(ast.Module(body=body, type_ignores=[]), relative, 'exec'), ns)
    return ns


def resize(a, shape):
    out = np.empty(shape, np.float32)
    for y in range(shape[0]):
        for x in range(shape[1]):
            out[y, x] = sum(float(a[i, j]) * w for i, j, w in interpolation_stencil((y, x), a.shape, shape))
    return out


def save_cache(root, ctx, row, raw):
    path = root / f"raw/{row['calibration_index']:03d}.npz"
    path.parent.mkdir(exist_ok=True)
    np.savez(path, raw_distances=raw, protocol_id=np.asarray(inputs.replay.PROTOCOL_ID),
             relative_path=np.asarray(row['relative_path']), layers=np.array(s.LAYERS, np.int32),
             grid=np.array(raw.shape[1:], np.int32), output_shape=np.array(row['output_shape'], np.int32),
             metadata=np.asarray(json.dumps(dict(input_identity=ctx['identity'], image=row, distance_divisor=1280,
                                                 raw_already_divided=True, physical_row_major=True))))
    return path


def fixture(root):
    formal, recovered = root / 'formal', root / 'recovered'
    formal.mkdir(); recovered.mkdir()
    ctx = dict(source=[], identity=inputs.digest({'fixture': 1}), frozen={'fixture': 1},
               paths=dict(output=root / 'evaluation', calibration=recovered, formal=formal),
               medians=np.array([.1, .2, .3, .4], np.float32), scales=np.array([.5, 1, 2, 4], np.float32))
    ctx['frozen']['assets'] = {'source_mad_state.npz': 'fixture_patch_state'}
    ctx['identity'] = inputs.digest(ctx['frozen'])
    records, raws = [], []
    original = original_functions()
    rng = np.random.RandomState(42)
    for i in range(66):
        row = dict(source_index=i * 8, calibration_index=i + 1, role='calibration', relative_path=f'train/good/n{i:03d}.png',
                   output_shape=[2, 3], image_height=8, image_width=12, image_sha256=f'image{i}')
        ctx['source'].append(row)
        raw = (rng.uniform(.5, 3, (4, 3, 4)) + i / 100).astype(np.float32)
        raws.append(raw)
        path = save_cache(recovered, ctx, row, raw)
        z = (raw - ctx['medians'][:, None, None]) / ctx['scales'][:, None, None]
        peak = [float(resize(a, (2, 3)).max()) for a in z]
        local = [original['local_mean'](a) for a in z]
        records.append(dict(file=path.relative_to(recovered).as_posix(), cache_sha256=inputs.sha(path),
                            relative_path=row['relative_path'], peak=peak, local=local))
    patch_stats = []
    for j, layer in enumerate(s.LAYERS):
        v = np.concatenate([a[j].ravel() for a in raws]); median = float(np.median(v))
        scale = float(np.float32(1.4826 * float(np.median(abs(v - median)))))
        # Frozen patch stats are allowed to differ from fixture transform states;
        # here set them equal and regenerate original reference statistics.
        ctx['medians'][j] = median; ctx['scales'][j] = scale
        patch_stats.append(dict(layer=layer, patch_count=len(v), median=median, mad_scale_1p4826=scale))
    for record, a in zip(records, raws):
        z = (a - ctx['medians'][:, None, None]) / ctx['scales'][:, None, None]
        record['peak'] = [float(resize(v, (2, 3)).max()) for v in z]
        record['local'] = [original['local_mean'](v) for v in z]
    inputs.replay.write_csv(formal / 'calibration_stats.csv', patch_stats)
    inputs.replay.atomic_json(recovered / 'inputs.json', dict(identity=ctx['identity'], frozen=ctx['frozen']))
    ledger = dict(tool=inputs.TOOL, identity=ctx['identity'], status='calibration_recovered', images=66, failed_checks=0,
                  records=records, inputs_sha256=inputs.sha(recovered / 'inputs.json'), output_sha256={})
    for name in ('source_ordered_manifest.csv', 'calibration_reference.csv', 'patch_mad_validation.csv', 'numerical_verification.csv'):
        (recovered / name).write_text('synthetic_reference\n', encoding='utf-8')
        ledger['output_sha256'][name] = inputs.sha(recovered / name)
    ledger['manifest_sha256'] = inputs.sha(recovered / 'source_ordered_manifest.csv')
    inputs.replay.atomic_json(recovered / 'recovery_status.json', ledger)
    cases = [dict(sample_id=sid, relative_path=p, label=int(i >= 10)) for i, (sid, p) in enumerate(inputs.replay.CASES)]
    samples = cases + [dict(sample_id=f'n{i}', relative_path=f'test/good/background/n{i}.png', label=0) for i in range(75)]
    samples += [dict(sample_id=f'a{i}', relative_path=f'test/defect/background/a{i}.png', label=1) for i in range(210)]
    cache, old_rows = {}, []
    permutation = rng.permutation(12).astype(np.int64)
    for i, row in enumerate(samples):
        raw = (rng.uniform(.6, 4, (4, 3, 4)) + row['label'] + i / 10000).astype(np.float32)
        target = raw * np.float32(.93)
        cache[f"raw/{row['sample_id']}.npz"] = dict(source_distance=raw.reshape(4, -1)[:, permutation],
            target_full_distance=target.reshape(4, -1)[:, permutation], positions=permutation, grid=np.array([3, 4]),
            metadata=dict(output_shape=[2, 3]))
        for method, a in (('MAD_EQUAL', raw), ('TARGET_MATCHED_K8_MAD', target)):
            z = (a - ctx['medians'][:, None, None]) / ctx['scales'][:, None, None]
            old_rows.append(dict(**row, method=method, domain='background', image_score=float(resize(z.mean(0), (2, 3)).max())))
    inputs.replay.write_csv(formal / 'sample_scores.csv', old_rows)
    domain = []
    for method in ('MAD_EQUAL', 'TARGET_MATCHED_K8_MAD'):
        chosen = [r for r in old_rows if r['method'] == method]
        y = [r['label'] for r in chosen]; score = [r['image_score'] for r in chosen]
        domain.append(dict(method=method, domain='background', image_AUROC=roc_auc_score(y, score),
                           image_AUPR=average_precision_score(y, score)))
    inputs.replay.write_csv(formal / 'domain_metrics.csv', domain)
    ctx.update(original=dict(capture=dict(samples=samples)), bundle=dict(raw=cache))
    return ctx, recovered


class ScoringTests(unittest.TestCase):
    def test_original_function_bodies_float32(self):
        original = original_functions()
        rng = np.random.RandomState(3)
        v = rng.normal(size=(66, 4)).astype(np.float32)
        for got, expected in zip(s.fit_image_mad(v), original['fit_image_mad'](v)):
            np.testing.assert_array_equal(got, expected)
        raw = rng.uniform(0, 5, (4, 5, 6)).astype(np.float32)
        _, peak, local = s.statistics(raw.reshape(4, -1), np.arange(30, dtype=np.int64), (5, 6), (9, 11), np.zeros(4, np.float32), np.ones(4, np.float32))
        np.testing.assert_array_equal(local, [original['local_mean'](a) for a in raw])
        m, c = s.fit_image_mad(v)
        state = {k: dict(medians=m.tolist(), scales=c.tolist()) for k in ('peak', 'local')}
        expected = original['fused_score'](original['calibrated_branch'](peak, m, c), original['calibrated_branch'](local, m, c))
        self.assertEqual(s.ch1_score(peak, local, state), expected)

    def test_positions_and_interpolated_peak_not_native_max(self):
        raw = np.ones((4, 3, 3), np.float32) * 2
        raw[:, 0, 0] = 20
        p = np.array([8, 1, 6, 3, 2, 5, 7, 0, 4], np.int64)
        a = s.statistics(raw.reshape(4, -1)[:, p], p, (3, 3), (1, 1), np.zeros(4, np.float32), np.ones(4, np.float32))
        self.assertEqual(a[0], 2)
        np.testing.assert_array_equal(a[1], [2] * 4)
        np.testing.assert_allclose(a[2], [4] * 4, atol=1e-6)
        with self.assertRaises(ValueError):
            s.statistics(raw.reshape(4, -1), np.zeros(9, np.int64), (3, 3), (1, 1), np.zeros(4, np.float32), np.ones(4, np.float32))

    def test_negative_valid_local_and_image_mad_floor(self):
        a = np.full((4, 3, 3), -8, np.float32)
        # Use positive raw, larger median to preserve negative calibrated maps.
        _, peak, local = s.statistics((a + 10).reshape(4, -1), np.arange(9, dtype=np.int64), (3, 3), (9, 9), np.full(4, 10, np.float32), np.ones(4, np.float32))
        np.testing.assert_array_equal(peak, [-8] * 4); np.testing.assert_array_equal(local, [-8] * 4)
        np.testing.assert_array_equal(s.fit_image_mad(np.ones((66, 4), np.float32))[1], np.full(4, 1e-8, np.float32))

    def test_ties_and_average_precision(self):
        y, score = np.array([0, 0, 1, 1]), np.array([1., 2., 2., 3.])
        result = s.image_metrics(y, score)
        self.assertEqual(result['image_AUROC'], roc_auc_score(y, score))
        self.assertEqual(result['image_AP'], average_precision_score(y, score))
        self.assertEqual(s.pair_errors(y, score).sum(), .5)

    def test_completion_requires_all_checks(self):
        self.assertFalse(s.complete(296, [dict(verified=True)]))
        self.assertFalse(s.complete(297, [dict(verified=False)]))
        self.assertFalse(s.complete(297, []))


class InputLedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'downloads', prefix='ch1_test_')
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_split_matches_actual_discovery_521(self):
        data = self.root / 'AeBAD_S'; (data / 'test').mkdir(parents=True)
        train = data / 'train/good'; train.mkdir(parents=True)
        for i in reversed(range(521)):
            Image.new('RGB', (8, 12), (i % 256, 0, 0)).save(train / f'{i:04d}.png')
        (train / '._0000.png').write_bytes(b'ignored')
        _, records, _ = discover_phase_a(data, formal=False)
        memory, calibration = split_source_normal(records, threshold_fraction=8)
        source = inputs.SourceData(data)
        try: rows = source.manifest()
        finally: source.close()
        self.assertEqual((len(records), len(memory), len(calibration)), (521, 455, 66))
        selected = [r for r in rows if r['role'] == 'calibration']
        self.assertEqual([r['relative_path'] for r in selected], [Path(r.path).relative_to(data).as_posix() for r in calibration])
        self.assertEqual([r['source_index'] for r in selected], list(range(0, 521, 8)))
        self.assertEqual(selected[0]['output_shape'], [3, 2])
        with self.assertRaises(ValueError): inputs.ordered_manifest(['b', 'a'], lambda _: b'')

    def test_resume_rejects_modified_cache_without_rewriting_history(self):
        ctx, root = fixture(self.root)
        recovery.validate_recovery(root, ctx, complete=True)
        history = (root / 'recovery_status.json').read_bytes()
        row = ctx['source'][0]
        raw = recovery.read_cache(root / 'raw/001.npz', row, ctx['identity'])
        save_cache(root, ctx, row, raw + np.float32(.01))
        with self.assertRaisesRegex(ValueError, 'changed since previous record'):
            recovery.validate_recovery(root, ctx, complete=False)
        self.assertEqual(history, (root / 'recovery_status.json').read_bytes())

    def test_resume_rejects_orphan_missing_and_changed_inputs(self):
        ctx, root = fixture(self.root)
        missing = root / 'raw/066.npz'; blob = missing.read_bytes(); missing.unlink()
        with self.assertRaisesRegex(ValueError, 'missing, duplicate, orphan'): recovery.validate_recovery(root, ctx, complete=False)
        missing.write_bytes(blob)
        (root / 'raw/067.npz').write_bytes(blob)
        with self.assertRaisesRegex(ValueError, 'missing, duplicate, orphan'): recovery.validate_recovery(root, ctx, complete=False)
        (root / 'raw/067.npz').unlink()
        wrong = copy.deepcopy(ctx); wrong['identity'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'identity'): recovery.validate_recovery(root, wrong, complete=False)

    def test_no_torch_default_preflight_even_missing_inputs(self):
        script = f"""import sys
from pathlib import Path
from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.runner import main
assert main(['--project-root', {str(self.root)!r}]) == 2
assert 'torch' not in sys.modules
assert not list(Path({str(self.root)!r}).iterdir())
"""
        result = subprocess.run([sys.executable, '-B', '-c', script], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_protected_output(self):
        p = runner.parser().parse_args(['--output-dir', str(ROOT / 'datasets')])
        with self.assertRaisesRegex(ValueError, 'overlaps'): inputs.guard(inputs.resolve(p))

    def test_full_synthetic_cpu_evaluation_and_shared_calibration(self):
        ctx, root = fixture(self.root)
        ctx['cpu_workers'] = 4
        with patch.object(evaluation, 'load_context', return_value=ctx):
            state = evaluation.evaluate(ctx)
        self.assertEqual(state['status'], 'evaluation_complete')
        self.assertFalse(state['torch_imported'])
        rows = inputs.replay.read_csv(ctx['paths']['output'] / 'per_image_scores.csv')
        self.assertEqual(len(rows), 1188)
        self.assertEqual(len({r['image_calibration_state_sha256'] for r in rows if r['branch'].endswith('CH1')}), 1)
        metrics = inputs.replay.read_csv(ctx['paths']['output'] / 'metric_comparisons.csv')
        self.assertEqual({(r['branch'], r['reference']) for r in metrics}, set(s.COMPARISONS))
        self.assertEqual(len(inputs.replay.read_csv(ctx['paths']['output'] / 'ranking_summary.csv')), 9)
        with self.assertRaisesRegex(ValueError, 'output exists'): evaluation.evaluate(ctx)

    def test_backend_shift_cannot_complete(self):
        ctx, _ = fixture(self.root)
        original_statistics = evaluation.statistics
        def shifted(*args, **kwargs):
            current, peak, local = original_statistics(*args, **kwargs)
            return current + 1, peak, local
        with patch.object(evaluation, 'load_context', return_value=ctx), patch.object(evaluation, 'statistics', side_effect=shifted):
            state = evaluation.evaluate(ctx)
        self.assertEqual(state['status'], 'partial_evidence')
        self.assertGreater(state['numerical_failed_checks'], 0)

    def test_complete_recovery_reuses_without_model_import(self):
        ctx, root = fixture(self.root)
        ctx['paths']['output'] = root
        with patch.object(recovery, 'recovery_assets', return_value=[]):
            self.assertEqual(recovery.recover(ctx, 'cuda:0')['status'], 'calibration_recovered')
        self.assertNotIn('torch', sys.modules)

    def test_parallel_reconstruction_equals_serial(self):
        from concurrent.futures import ThreadPoolExecutor
        ctx, _ = fixture(self.root)
        chosen = ctx['original']['capture']['samples'][:12]
        serial = [evaluation.image_statistics(ctx, row) for row in chosen]
        with ThreadPoolExecutor(max_workers=4) as pool:
            parallel = list(pool.map(lambda row: evaluation.image_statistics(ctx, row), chosen))
        for before, after in zip(serial, parallel):
            for memory in before:
                for a, b in zip(before[memory], after[memory]): np.testing.assert_array_equal(a, b)

    def test_near_equal_scores_still_check_pair_states(self):
        samples = [dict(sample_id=sid, label=0) for sid, _ in inputs.replay.CASES[:10]]
        samples += [dict(sample_id=f'n{i}', label=0) for i in range(75)]
        samples += [dict(sample_id=f'a{i}', label=1) for i in range(212)]
        old = np.arange(297, dtype=float); old[84:86] = [2, 2]
        new = old.copy(); new[84] -= 1e-7
        scores = {b: new for b in s.BRANCHES}; checks = []
        evaluation.ranking_tables(samples, scores, checks, {'SOURCE_CURRENT': old})
        self.assertFalse(checks[0]['verified'])


if __name__ == '__main__':
    unittest.main()
