"""Lightweight tests: no Torch import, model, GPU, or real feature extraction."""
from __future__ import annotations

import ast
import contextlib
import copy
import io
import json
import math
import shutil
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence
from unittest.mock import patch

import numpy as np
from PIL import Image

from DINOv3.MADEqual.aebad_s_union_support_loo import runner, inputs
from DINOv3.MADEqual.aebad_s_union_support_loo.analysis import (
    ALL_BRANCHES, BRANCHES, FORMAT, CaptureLog, branch_maps, branch_raws, correspondence_rows,
    image_rows, load_raw, ranking_rows, save_raw, spatial_rows, support_layer_rows, validate_arrays,
)
from DINOv3.MADEqual.aebad_s_target_support_trace.analysis import comparison
from DINOv3.MADEqual.aebad_s_union_diagnostics.spatial import instance_regions, interpolation_stencil
from DINOv3.tests.test_mad_equal_aebad_s_target_support_trace import actual_calibrated, actual_patch_exec

ROOT = Path(__file__).resolve().parents[2]
FORMAL = ROOT / "server_archives/aebad_s_target_fewshot_seed42_v1_formal_results/aebad_s_target_fewshot_seed42_v1"


def raw_fixture():
    source = np.broadcast_to(np.array([6, 7, 8, 9], np.float32), (4, 4)).copy()
    target = np.stack([np.full((4, 4), 3+j, np.float32) for j in range(8)])
    arrays = dict(positions=np.array([2, 0, 3, 1], np.int64), source_distance=source,
                  target_distance=target, source_index=np.zeros((4, 4), np.int64),
                  target_index=np.zeros((8, 4, 4), np.int64))
    meta = dict(format=FORMAT, identity="fixture", sample_id="q", relative_path="test/good/background/q.png",
                grid=[2, 2], output_shape=[2, 2], source_rows=[10]*4,
                supports=[dict(support_index=j, grid=[1, 1]) for j in range(1, 9)])
    return arrays, meta


def actual_map_backend():
    """Execute actual MAD and original _method_map; identity-size interpolation
    is an instrumented test dependency, not a substitute production backend."""
    calibrated = actual_calibrated()
    trace = []
    def equal(raw, medians, scales):
        trace.append("layer_MAD_then_mean")
        return calibrated(raw, medians, scales).mean(axis=0)
    def interpolate(value, shape):
        trace.append("interpolate_after_mean")
        assert value.shape == shape
        return value.copy()
    path = ROOT / "DINOv3/MADEqual/robustad_target_fewshot_p0/runner.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_method_map")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    env = dict(np=np, _method_raw=lambda values, method: values["official_raw"], mad_equal=equal, interpolate_score=interpolate)
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), env)
    mad = SimpleNamespace(medians=np.array([1, 2, 3, 4], np.float32), scales=np.array([1, 2, 4, 8], np.float32))
    return SimpleNamespace(runner=SimpleNamespace(_method_map=env["_method_map"]), mad=mad,
                           calibrated=calibrated, patch_mask=lambda mask, grid: np.asarray(mask)>0), trace


class Tensor:
    """Minimal NumPy test double; actual CUDA cdist is deliberately not tested."""
    def __init__(self, value): self.value = np.asarray(value)
    @property
    def shape(self): return self.value.shape
    def numpy(self): return self.value
    def to(self, **kwargs): return self
    def __getitem__(self, key): return Tensor(self.value[key])


def scoring_fixture():
    path = ROOT / "DINOv3/MADEqual/robustad_target_fewshot_p0/scoring.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {"_notify_nearest", "exact_prefix_minima", "_support_prefix_minima", "observe_fewshot_pil", "FewshotObservation"}
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    for node in nodes:
        if isinstance(node, ast.FunctionDef): node.decorator_list = []
    calls, forwards = [], []
    def top1(query, bank):
        calls.append((query.shape, bank.shape))
        matrix = np.linalg.norm(query.value[:, None]-bank.value[None], axis=2)
        indices = np.argmin(matrix, axis=1)
        return Tensor(matrix[np.arange(len(indices)), indices].astype(np.float32)), Tensor(indices.astype(np.int64))
    positions = np.array([2, 0, 3, 1], np.int64)
    query = np.zeros((4, 1280), np.float32)
    query[:, 0] = [0, 3, 6, 9]
    def extract(model, processed, **kwargs):
        forwards.append(kwargs)
        return [SimpleNamespace(grid=(2, 2), positions=positions, queries={l: Tensor(query.copy()) for l in (7, 15, 23, 31)})]
    env = dict(np=np, dataclass=dataclass, PREFIXES=(1, 2, 4, 8), LAYERS=(7, 15, 23, 31),
               Any=Any, Mapping=Mapping, Sequence=Sequence, Callable=Callable,
               _exact_top1_chunked=top1, _extract_processed_batch=extract,
               image_tensor=lambda image: Tensor(np.zeros((3, 32, 32), np.float32)))
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), env)
    banks = []
    for j in range(8):
        b = np.zeros((2, 1280), np.float32); b[:, 0] = [j, j+10]
        banks.append(Tensor(b))
    official = Tensor(np.zeros((1, 1280), np.float32))
    kwargs = dict(model=SimpleNamespace(preprocessing=lambda t: t), image=None,
                  official_banks={l: official for l in env["LAYERS"]},
                  source_support={l: banks for l in env["LAYERS"]}, target_support={l: banks for l in env["LAYERS"]}, device="fixture")
    return env, kwargs, calls, forwards


def preflight_fixture(root: Path):
    """Synthetic headers/identities with real formal CSV; no tensor loading."""
    from DINOv3.tests.test_mad_equal_aebad_s_target_support_trace import minimal_trace_fixture
    minimal_trace_fixture(root)
    results, cache, diagnostic = root/"formal", root/"replay", root/"diagnostic"
    replay = runner.replay
    full = replay.select_cases(replay.read_csv(results/"sample_scores.csv"))
    supports = [r for r in replay.select_support(replay.read_csv(results/"support_manifest.csv"), full)
                if r["branch"] == "target_matched"]
    data = root/"data/AeBAD/AeBAD_S"
    original_history = json.loads((cache/"replay_status.json").read_text())
    hist_inputs = json.loads((cache/"replay_inputs.json").read_text())
    for sample in full:
        image_path = data/sample["relative_path"]
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 24), "gray").save(image_path)
        hist_inputs["identity"]["image_sha256"][sample["relative_path"]] = replay.sha256(image_path)
        raw = cache/f"raw_cache/heldout/background/{sample['sample_id']}.npz"
        raw.parent.mkdir(parents=True, exist_ok=True)
        arrays = {n: np.full((4, 2, 3), .5, np.float32) for n in replay.RAW_ARRAY_NAMES}
        np.savez(raw, protocol_id=np.asarray(replay.PROTOCOL_ID), cache_revision=np.asarray(replay.CACHE_REVISION),
                 relative_path=np.asarray(sample["relative_path"]), output_shape=np.asarray([6, 8], np.int32), **arrays)
        next(r for r in original_history["cases"] if r["sample_id"] == sample["sample_id"])["cache_sha256"] = replay.sha256(raw)
    support_root = cache/"support_features/target_matched/background"
    support_root.mkdir(parents=True)
    provenance = []
    for support in supports:
        path = support_root/f"{int(support['support_index']):02d}.npz"
        np.savez(path, protocol_id=np.asarray(replay.PROTOCOL_ID), relative_path=np.asarray(support["relative_path"]), grid=np.array([1, 1], np.int32),
                 **{f"layer_{l}": np.zeros((1, 1280), np.float32) for l in replay.LAYERS})
        provenance.append(dict(relative_path=support["relative_path"], sha256=replay.sha256(path)))
    replay.write_csv(cache/"support_provenance.csv", provenance)
    replay.write_csv(diagnostic/"selected_cases.csv", [dict(r, image_height=24, image_width=32) for r in full])
    (results/"official_memory.pt").write_bytes(b"synthetic header fixture; never loaded")
    replay.atomic_json(cache/"replay_status.json", original_history)
    replay.atomic_json(cache/"replay_inputs.json", hist_inputs)
    return runner.build_parser().parse_args(["preflight", "--project-root", str(ROOT), "--results-dir", str(results),
        "--replay-dir", str(cache), "--diagnostic-dir", str(diagnostic), "--official-root", str(root/"official"),
        "--checkpoint", str(root/"checkpoint.pth"), "--dinov3-root", str(root/"dino"), "--data", str(root/"data"),
        "--output-dir", str(root/"output"), "--no-write"])


def offline_capture_fixture(root: Path):
    """A tiny synthetic completed log; never claims a real capture validation."""
    args = preflight_fixture(root)
    cases = runner.replay.select_cases(runner.replay.read_csv(root/"formal/sample_scores.csv"))
    capture_dir = root/"captured"
    def geometry(grid):
        return dict(grid=list(grid), model_patch_size=16, original_height=24, original_width=32,
                    processed_height=grid[0]*16, processed_width=grid[1]*16,
                    axes={axis: dict(input=[[0, size*16]], prediction=[[0, size]], result=[[0, size]]) for axis, size in zip(('y', 'x'), grid)},
                    coordinate_meaning='projected_input_token_cell_not_complete_receptive_field')
    supports = [dict(support_index=int(r['support_index']), relative_path=r['relative_path'], grid=[1, 1], geometry=geometry((1, 1)),
                     image_sha256=runner.replay.sha256(root/'data/AeBAD/AeBAD_S'/r['relative_path']), feature_sha256='synthetic')
                for r in runner.replay.select_support(runner.replay.read_csv(root/'formal/support_manifest.csv'), cases) if r['branch']=='target_matched']
    # Install known toy reference scores in the TEMPORARY formal CSV. Leave
    # embedded capture sample scores untouched, so success requires reading CSV.
    fixture_arrays, _ = raw_fixture()
    fixture_backend, _ = actual_map_backend()
    fixture_maps = branch_maps(branch_raws(fixture_arrays, (2, 2)), (2, 2), fixture_backend)
    formal_rows = runner.replay.read_csv(root/'formal/sample_scores.csv')
    selected_ids = {c['sample_id'] for c in cases}
    for row in formal_rows:
        if row['sample_id'] in selected_ids and row['method'] in runner.replay.METHODS:
            branch = 'Source' if row['method'] == runner.replay.METHODS[0] else 'K8'
            row['image_score'] = float(fixture_maps[branch].max())
    runner.replay.write_csv(root/'formal/sample_scores.csv', formal_rows)
    frozen = dict(formal_sha256={name: runner.replay.sha256(root/'formal'/name) for name in
                  ('sample_scores.csv', 'support_manifest.csv', 'complete.json', 'memory_manifest.json', 'source_mad_state.npz')})
    identity = inputs.digest(frozen)
    gt_hashes = {}
    from DINOv3.MADEqual.aebad_s_union_diagnostics.analysis import gt_path
    records = []
    for case in cases:
        case.update(image_height=24, image_width=32)
        image_hash = runner.replay.sha256(root/'data/AeBAD/AeBAD_S'/case['relative_path'])
        if case['label']:
            relative = gt_path(case['relative_path'])
            gt_hashes[relative] = runner.replay.sha256(root/'data/AeBAD/AeBAD_S'/relative)
        arrays, meta = raw_fixture()
        meta.update(identity=identity, sample_id=case['sample_id'], relative_path=case['relative_path'], supports=supports,
                    geometry=geometry((2, 2)), image_sha256=image_hash)
        path = capture_dir/f"raw_cache/{case['sample_id']}.npz"
        save_raw(path, arrays, meta)
        records.append(dict(sample_id=case['sample_id'], file=f"raw_cache/{case['sample_id']}.npz", sha256=runner.replay.sha256(path)))
    capture_inputs = dict(tool=inputs.TOOL, identity=identity, frozen=frozen, scope='twelve', samples=cases, selected_cases=cases,
                         supports=supports, image_sha256={r['relative_path']: runner.replay.sha256(root/'data/AeBAD/AeBAD_S'/r['relative_path']) for r in cases}, gt_sha256=gt_hashes)
    runner.replay.atomic_json(capture_dir/'inputs.json', capture_inputs)
    runner.replay.write_csv(capture_dir/'capture_verification.csv', [dict(sample_id=r['sample_id'], check_kind='synthetic_fixture', verified=True) for r in cases])
    runner.replay.atomic_json(capture_dir/'status.json', dict(tool=inputs.TOOL, scope='twelve', identity=identity, status='capture_verified', records=records,
        inputs_sha256=runner.replay.sha256(capture_dir/'inputs.json'), verification_sha256=runner.replay.sha256(capture_dir/'capture_verification.csv')))
    args.mode, args.capture_dir, args.output_dir = 'analyze', str(capture_dir), str(root/'analysis')
    args.no_write = False
    return args


class LoggingTests(unittest.TestCase):
    def test_logging_preserves_default_result_interface_and_search_calls(self):
        env, kwargs, calls, forwards = scoring_fixture()
        plain = env["observe_fewshot_pil"](**kwargs)
        self.assertEqual(len(calls), 68)
        calls.clear()
        logged_events = []
        logged = env["observe_fewshot_pil"](**kwargs, nearest_observer=lambda *e: logged_events.append(e))
        self.assertEqual(len(calls), 68)
        self.assertEqual(len(logged_events), 68)
        self.assertEqual(type(plain), type(logged))
        np.testing.assert_array_equal(plain.official_raw, logged.official_raw)
        for field in ("source_raw", "target_raw", "source_support_raw", "target_support_raw"):
            for k, value in getattr(plain, field).items():
                np.testing.assert_array_equal(value, getattr(logged, field)[k])
        self.assertTrue(all(c[0] == (4, 1280) for c in calls))
        self.assertEqual(len(forwards), 2)
        self.assertTrue(all(f["query_masks"] == [None] for f in forwards))

    def test_target_only_capture_same_call_distance_indices_and_physical_mapping(self):
        env, kwargs, calls, forwards = scoring_fixture()
        kwargs["source_support"] = None
        log = CaptureLog()
        result = env["observe_fewshot_pil"](**kwargs, nearest_observer=log.nearest, geometry_observer=log.geometry)
        self.assertEqual(len(calls), 36)
        self.assertEqual(len(forwards), 1)
        arrays = log.arrays()
        physical = branch_raws(arrays, log.grid)
        np.testing.assert_array_equal(physical["Source"], result.official_raw)
        np.testing.assert_array_equal(physical["K8"], result.target_raw[8])
        np.testing.assert_array_equal(arrays["source_distance"][0], np.array([0, 3, 6, 9], np.float32)/1280)
        self.assertEqual(arrays["source_index"].dtype, np.int64)

    def test_callbacks_receive_readonly_copies(self):
        env, kwargs, _, _ = scoring_fixture()
        def observer(pool, layer, index, distance, indices):
            self.assertFalse(distance.flags.writeable)
            self.assertFalse(indices.flags.writeable)
            with self.assertRaises(ValueError): distance[0] = 999
        env["observe_fewshot_pil"](**kwargs, nearest_observer=observer)

    def test_cached_source_cannot_claim_same_call_index_log(self):
        env, kwargs, _, _ = scoring_fixture()
        with self.assertRaisesRegex(ValueError, "same-call"):
            env["observe_fewshot_pil"](**kwargs, cached_official_raw=np.zeros((4, 2, 2), np.float32), nearest_observer=lambda *a: None)


class OfflineTests(unittest.TestCase):
    def test_eight_removals_match_manual_values_and_nonidentity_positions(self):
        arrays, meta = raw_fixture()
        raws = branch_raws(arrays, (2, 2))
        self.assertEqual(len(raws), 10)
        np.testing.assert_array_equal(raws["Source"][0], [[7, 9], [6, 8]])
        np.testing.assert_array_equal(raws["K8"], np.full((4, 2, 2), 3, np.float32))
        np.testing.assert_array_equal(raws["minus_1"], np.full((4, 2, 2), 4, np.float32))
        for j in range(2, 9): np.testing.assert_array_equal(raws[f"minus_{j}"], raws["K8"])
        for branch in BRANCHES: self.assertTrue(np.all(raws[branch] >= raws["K8"]-1e-7))
        validate_arrays(arrays, meta)

    def test_redundant_support_source_win_and_cross_image_ties(self):
        arrays, _ = raw_fixture()
        arrays["target_distance"][:] = 3
        rows = support_layer_rows(dict(sample_id="q", label=0), arrays)
        self.assertTrue(all(r["lost_distance_drop_fraction"] == 0 and r["own_unique_target_win_fraction"] == 0 for r in rows))
        arrays["source_distance"][:] = 1
        rows = support_layer_rows(dict(sample_id="q", label=0), arrays)
        self.assertTrue(all(r["source_strict_win_fraction"] == 1 and r["lost_distance_drop_mean"] == 0 for r in rows))
        arrays["source_distance"][:] = 3
        self.assertTrue(all(r["source_target_tie_fraction"] == 1 for r in support_layer_rows(dict(sample_id="q", label=0), arrays)))

    def test_original_map_delegation_layer_min_before_mad_mean_interpolation(self):
        arrays, _ = raw_fixture()
        arrays["target_distance"][0, :, :] = np.array([2, 12, 6, 12], np.float32)[:, None]
        arrays["target_distance"][1, :, :] = np.array([12, 3, 12, 5], np.float32)[:, None]
        backend, trace = actual_map_backend()
        raws = branch_raws(arrays, (2, 2))
        maps = branch_maps(raws, (2, 2), backend)
        expected = ((raws["K8"]-np.array([1, 2, 3, 4], np.float32)[:, None, None]) /
                    np.array([1, 2, 4, 8], np.float32)[:, None, None]).mean(axis=0)
        np.testing.assert_array_equal(maps["K8"], expected)
        self.assertEqual(trace, ["layer_MAD_then_mean", "interpolate_after_mean"]*10)
        # min after layer mean produces a different detector; never use it.
        bad = min(backend.calibrated(arrays["target_distance"][j].reshape(4, 2, 2), backend.mad.medians, backend.mad.scales).mean()
                  for j in range(8))
        self.assertNotAlmostEqual(float(maps["K8"].mean()), float(bad))

    def test_instance_margin_can_decrease_despite_pointwise_response_increase(self):
        mask = np.zeros((40, 40), bool); mask[18:22, 18:22] = True
        inside, ring = instance_regions(mask, mask, mask.shape)
        base = np.zeros(mask.shape, np.float32); base[inside] = 3; base[ring] = 1
        removed = base.copy(); removed[inside] += .1; removed[ring] += 1
        maps = {name: base.copy() for name in ALL_BRANCHES}; maps["minus_1"] = removed
        backend = SimpleNamespace(patch_mask=lambda image, grid: np.asarray(image)>0)
        _, rows = spatial_rows(dict(sample_id="q", relative_path="q", label=1), maps, mask, mask.shape, backend)
        row = next(r for r in rows if r["branch"] == "minus_1")
        self.assertLess(row["margin_change_vs_K8"], 0)
        self.assertTrue(np.all(removed >= base))
        # Image max actually is monotone for positive MAD scales/interpolation;
        # do not infer normal/anomaly ranking or margin monotonicity from it.
        self.assertGreater(float(removed.max()), float(base.max()))

    def test_original_ring_excludes_all_gt_and_preserves_fragment(self):
        mask = np.zeros((40, 40), bool); mask[20, 20] = True; mask[22:24, 22:24] = True
        component = np.zeros_like(mask); component[20, 20] = True
        inside, ring = instance_regions(component, mask, mask.shape)
        self.assertEqual(int(inside.sum()), 1)
        self.assertFalse((ring & mask).any())
        self.assertFalse(ring[3, 20]); self.assertTrue(ring[4, 20])

    def test_disappeared_instance_is_recorded(self):
        mask = np.zeros((40, 40), bool); mask[1, 1] = True
        maps = {name: np.zeros((2, 2), np.float32) for name in ALL_BRANCHES}
        backend = SimpleNamespace(patch_mask=lambda image, grid: np.ones(grid, bool))
        _, rows = spatial_rows(dict(sample_id="q", relative_path="q", label=1), maps, mask, (2, 2), backend)
        self.assertEqual(len(rows), 10)
        self.assertTrue(all(r["response_status"] == "mapped_instance_disappeared" and r["margin"] is None for r in rows))
        _, empty = spatial_rows(dict(sample_id="q", relative_path="q", label=1), maps, np.zeros_like(mask), (2, 2), backend)
        self.assertEqual(len(empty), 10)
        self.assertTrue(all(r["response_status"] == "empty_GT_annotation" and r["instance_id"] == "" for r in empty))

    def test_ranking_ties_and_fixed_top10_no_support_reinsertion(self):
        rows = []
        for sid, label, before in (("n", 0, 1.), ("a", 1, 1.)):
            for branch in ALL_BRANCHES:
                score = 2. if sid == "a" and branch == "minus_1" else before
                rows.append(dict(sample_id=sid, label=label, branch=branch, image_score=score, score_change_vs_K8=score-before))
        summaries, per_image = ranking_rows(rows, ["n"])
        baseline = next(r for r in summaries if r["branch"] == "K8")
        removed = next(r for r in summaries if r["branch"] == "minus_1")
        self.assertEqual(baseline["image_AUROC"], .5)
        self.assertEqual(baseline["image_AUPR"], .5)
        self.assertEqual(removed["improved_pairs_vs_K8"], 1)
        self.assertEqual(removed["image_AUROC"], 1)
        with self.assertRaisesRegex(ValueError, "held-out"):
            ranking_rows(rows[:-1], ["n"])

    def test_stencil_maps_to_explicit_compact_rows(self):
        arrays, _ = raw_fixture()
        inverse = {int(p): i for i, p in enumerate(arrays["positions"])}
        stencil = interpolation_stencil((1, 1), (2, 2), (4, 4))
        self.assertAlmostEqual(sum(w for _, _, w in stencil), 1)
        response = sum(w*float(arrays["source_distance"][0, inverse[y*2+x]]) for y, x, w in stencil)
        self.assertAlmostEqual(response, .5625*7+.1875*9+.1875*6+.0625*8)


class IntegrityTests(unittest.TestCase):
    def test_consistent_backend_offset_blocks_analyze_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = offline_capture_fixture(root)
            check = inputs.preflight(args)
            self.assertEqual(check['errors'], [])
            Path(args.output_dir).mkdir()
            backend, _ = actual_map_backend()
            backend.patch_mask = lambda image, grid: np.ones(grid, bool)
            original_map = backend.runner._method_map
            original_calibrated = backend.calibrated
            backend.runner._method_map = lambda *a: original_map(*a) + np.float32(1)
            backend.calibrated = lambda *a: original_calibrated(*a) + np.float32(1)
            raw_before = {p.name: runner.replay.sha256(p) for p in (Path(args.capture_dir)/'raw_cache').glob('*.npz')}
            with patch('DINOv3.MADEqual.aebad_s_union_diagnostics.spatial.MapBackend', return_value=backend), \
                 patch('DINOv3.MADEqual.aebad_s_union_support_loo.visualization.plot_case'), \
                 patch('DINOv3.MADEqual.aebad_s_union_support_loo.visualization.plot_correspondences'), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.analyze(args, check), 3)
            status = json.loads((Path(args.output_dir)/'status.json').read_text())
            self.assertEqual(status['status'], 'partial_evidence')
            self.assertFalse(status['formal_baselines_reproduced'])
            rows = runner.replay.read_csv(Path(args.output_dir)/'analysis_verification.csv')
            baseline = [r for r in rows if r.get('check_kind') == 'formal_baseline_image_score']
            self.assertEqual(len(baseline), 24)
            self.assertTrue(all(r['verified'] == 'False' and float(r['max_abs_error']) == 1 for r in baseline))
            stencil = [r for r in rows if not r.get('check_kind')]
            self.assertTrue(stencil)
            self.assertTrue(all(r['verified'] == 'True' for r in stencil))
            self.assertEqual(raw_before, {p.name: runner.replay.sha256(p) for p in (Path(args.capture_dir)/'raw_cache').glob('*.npz')})

    def test_full_domain_baseline_summary_checks_use_formal_csv(self):
        from sklearn.metrics import roc_auc_score, average_precision_score
        from DINOv3.MADEqual.aebad_s_union_support_loo.runner import formal_baselines, verify_formal_ranking
        formal = formal_baselines(FORMAL, [])
        summaries = []
        for branch in ('Source', 'K8'):
            values = list(formal[branch].values())
            labels = [r['label'] for r in values]
            scores = [r['image_score'] for r in values]
            summaries.append(dict(branch=branch, image_count=297, normal_count=85, anomaly_count=212,
                image_AUROC=float(roc_auc_score(labels, scores)), image_AUPR=float(average_precision_score(labels, scores))))
        rows = []
        verify_formal_ranking(summaries, formal, rows)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r['verified'] for r in rows))
        for summary in summaries:
            summary['image_AUROC'] += 2e-5
            summary['image_AUPR'] -= 2e-5
        failed = []
        verify_formal_ranking(summaries, formal, failed)
        self.assertTrue(all(not r['verified'] for r in failed))

    def test_offline_analysis_outputs_and_missing_gt_completion_gate(self):
        from DINOv3.MADEqual.aebad_s_union_diagnostics.analysis import gt_path
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = offline_capture_fixture(root)
            check = inputs.preflight(args)
            self.assertEqual(check['errors'], [])
            Path(args.output_dir).mkdir()
            backend, _ = actual_map_backend()
            backend.patch_mask = lambda image, grid: np.ones(grid, bool)
            with patch('DINOv3.MADEqual.aebad_s_union_diagnostics.spatial.MapBackend', return_value=backend), \
                 patch('DINOv3.MADEqual.aebad_s_union_support_loo.visualization.plot_case'), \
                 patch('DINOv3.MADEqual.aebad_s_union_support_loo.visualization.plot_correspondences'), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.analyze(args, check), 0)
                status = json.loads((Path(args.output_dir)/'status.json').read_text())
                self.assertEqual(status['status'], 'analysis_complete')
                self.assertTrue(status['formal_baselines_reproduced'])
                images = runner.replay.read_csv(Path(args.output_dir)/'per_image_loo.csv')
                self.assertEqual(len(images), 120)
                self.assertEqual(len(runner.replay.read_csv(Path(args.output_dir)/'per_support_effects.csv')), 8)
                anomaly = next(c for c in check['samples'] if c['label'])
                (root/'data/AeBAD/AeBAD_S'/gt_path(anomaly['relative_path'])).unlink()
                args.output_dir = str(root/'partial_analysis')
                partial = inputs.preflight(args)
                Path(args.output_dir).mkdir()
                self.assertEqual(runner.analyze(args, partial), 3)
                status = json.loads((Path(args.output_dir)/'status.json').read_text())
                self.assertEqual(status['analyzed_images'], 12)
                self.assertEqual(status['status'], 'partial_evidence')
                self.assertIn(anomaly['sample_id'], status['missing_gt_ids'])

    def test_cache_modified_or_half_written_is_rejected_without_overwrite(self):
        arrays, meta = raw_fixture()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"q.npz"
            save_raw(path, arrays, meta)
            checksum = runner.replay.sha256(path)
            load_raw(path, identity="fixture", sample=meta, checksum=checksum)
            original = path.read_bytes()
            with self.assertRaises(ValueError): save_raw(path, arrays, meta)
            self.assertEqual(path.read_bytes(), original)
            changed = copy.deepcopy(arrays); changed["target_distance"][1, 1, 0] -= .1
            with path.open("wb") as handle: np.savez(handle, metadata=np.asarray(json.dumps(meta)), **changed)
            with self.assertRaisesRegex(ValueError, "modified"):
                load_raw(path, identity="fixture", sample=meta, checksum=checksum)
            half = Path(temp)/"half.npz.tmp"; half.write_bytes(b"partial")
            with self.assertRaises(ValueError): save_raw(Path(temp)/"half.npz", arrays, meta)
            self.assertEqual(half.read_bytes(), b"partial")

    def test_invalid_mapping_dtype_and_bounds_rejected(self):
        arrays, meta = raw_fixture()
        for name, value in (("positions", np.array([0, 0, 2, 3], np.int64)),
                            ("target_distance", arrays["target_distance"].astype(np.float64)),
                            ("target_index", np.ones_like(arrays["target_index"]))):
            modified = dict(arrays); modified[name] = value
            with self.assertRaises(ValueError): validate_arrays(modified, meta)

    def test_help_and_default_mode_do_not_import_torch(self):
        self.assertEqual(runner.build_parser().parse_args([]).mode, "preflight")
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as error:
            runner.main(["--help"])
        self.assertEqual(error.exception.code, 0)
        self.assertNotIn("torch", sys.modules)

    def test_real_formal_csv_has_fixed_85_212_and_original_support_exclusions(self):
        replay = runner.replay
        rows = replay.read_csv(FORMAL/"sample_scores.csv")
        cases = replay.select_cases(rows)
        supports = [r for r in replay.select_support(replay.read_csv(FORMAL/"support_manifest.csv"), cases) if r["branch"] == "target_matched"]
        for method in replay.METHODS:
            evaluated = [r for r in rows if r["domain"] == "background" and r["method"] == method]
            self.assertEqual((len(evaluated), sum(int(r["label"]) == 0 for r in evaluated)), (297, 85))
            self.assertFalse({r["relative_path"] for r in supports} & {r["relative_path"] for r in evaluated})

    def test_preflight_test_only_root_and_blocked_stage_b_never_load_models(self):
        with tempfile.TemporaryDirectory() as temp:
            args = preflight_fixture(Path(temp))
            with patch.object(runner.replay, "CHECKPOINT_SHA256", runner.replay.sha256(Path(args.checkpoint))):
                checked = inputs.preflight(args)
                self.assertEqual(checked["errors"], [])
                self.assertEqual(len(checked["samples"]), 12)
                self.assertFalse((Path(checked["paths"]["data"])/"train/good").exists())
                self.assertNotIn("torch", sys.modules)
                # Stage B is blocked by a missing/failed A; no inference import.
                args.scope = "background"
                checked = inputs.preflight(args)
                self.assertTrue(checked["errors"])
                self.assertFalse(Path(args.output_dir).exists())

    def test_preflight_rejects_changed_support_bytes_and_changed_historical_raw(self):
        with tempfile.TemporaryDirectory() as temp:
            args = preflight_fixture(Path(temp))
            checkpoint_sha = runner.replay.sha256(Path(args.checkpoint))
            path = Path(temp)/"replay/support_features/target_matched/background/01.npz"
            with path.open("ab") as handle: handle.write(b"changed storage")
            with patch.object(runner.replay, "CHECKPOINT_SHA256", checkpoint_sha):
                check = inputs.preflight(args)
            self.assertTrue(any("provenance" in e for e in check["errors"]))

    def test_analysis_cannot_accept_failed_capture_or_overwrite_formal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runner.replay.atomic_json(root/"inputs.json", dict(tool=inputs.TOOL, identity=inputs.digest({}), frozen={}, scope="twelve"))
            runner.replay.atomic_json(root/"status.json", dict(status="capture_not_reproduced"))
            with self.assertRaises(ValueError): inputs.read_capture(root, stage_a=True)
            with self.assertRaises(ValueError): inputs.guard_output(dict(project=ROOT, output=FORMAL, results=FORMAL))


if __name__ == "__main__":
    unittest.main()
