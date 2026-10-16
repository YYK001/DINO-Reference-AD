"""CPU-only trace bookkeeping tests; no model or feature extraction is executed."""
from __future__ import annotations

import ast
import contextlib
import io
import json
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from DINOv3.MADEqual.aebad_s_target_support_trace import runner
from DINOv3.MADEqual.aebad_s_target_support_trace.analysis import (
    CASES, LAYERS, METHODS, comparison, flat_coordinates, image_geometry, patch_projection,
    per_support_nearest, position_patches, target_check, trace_positions, weighted_layers,
)
from DINOv3.MADEqual.aebad_s_union_diagnostics.spatial import instance_regions, interpolation_stencil

WORKSPACE = Path(__file__).resolve().parents[2]
RESULTS = WORKSPACE / "server_archives/aebad_s_target_fewshot_seed42_v1_formal_results/aebad_s_target_fewshot_seed42_v1"


def actual_patch_exec():
    # Execute only the actual pure-Python constructor/axis splitter, without
    # importing the upstream model module or replacing its geometry.
    path = WORKSPACE / "external_baselines/SuperADD/tracks/industrial/src/industrial/model.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PatchedExecution")
    original.body = [node for node in original.body if isinstance(node, ast.FunctionDef) and node.name in ("__init__", "axis_patch_split")]
    module = ast.fix_missing_locations(ast.Module(body=[original], type_ignores=[]))
    namespace = dict(ceil=math.ceil, floor=math.floor)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["PatchedExecution"](640, 128, 16)


def actual_calibrated():
    path = WORKSPACE / "DINOv3/relation_oracle_cache/extract.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    module = ast.fix_missing_locations(ast.Module(body=[
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "mad_calibrated"
    ], type_ignores=[]))
    namespace = dict(np=np, LAYERS=LAYERS)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["mad_calibrated"]


def numpy_reference_top1(query, bank):
    distances = np.linalg.norm(query[:, None, :] - bank[None, :, :], axis=2)
    indices = np.argmin(distances, axis=1)
    return distances[np.arange(len(query)), indices], indices


def minimal_trace_fixture(root):
    """Real formal metadata, tiny synthetic pixels/raw; no backbone or inference."""
    from PIL import Image
    from DINOv3.MADEqual.aebad_s_union_diagnostics.analysis import gt_path
    from DINOv3.MADEqual.aebad_s_union_diagnostics.spatial import components
    replay = runner.replay
    results, cache, diagnostic = (root / name for name in ("formal", "replay", "diagnostic"))
    official, dino, data = root / "official", root / "dino", root / "data/AeBAD/AeBAD_S"
    results.mkdir()
    for name in ("sample_scores.csv", "support_manifest.csv", "memory_manifest.json", "complete.json", "source_mad_state.npz"):
        shutil.copyfile(RESULTS / name, results / name)
    for path in (official / "config.json", official / "tracks/industrial/src/industrial/model.py", dino / "hubconf.py"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic read-only preflight code fixture")
    checkpoint = root / "checkpoint.pth"
    checkpoint.write_bytes(b"synthetic preflight checkpoint; never loaded")
    full = replay.select_cases(replay.read_csv(results / "sample_scores.csv"))
    supports = [row for row in replay.select_support(replay.read_csv(results / "support_manifest.csv"), full)
                if row["branch"] == "target_matched"]
    cases = [next(row for row in full if row["sample_id"] == sample_id) for sample_id, _ in CASES]
    image_hashes = {}
    for row in [*cases, *supports]:
        path = data / row["relative_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 24), "gray").save(path)
        image_hashes[row["relative_path"]] = replay.sha256(path)
    instances = []
    for row in cases:
        if row["label"]:
            mask = np.zeros((24, 32), np.uint8)
            mask[6:10, 9:13] = 255
            path = data / gt_path(row["relative_path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask).save(path)
            instances.append(dict(sample_id=row["sample_id"], **components(mask > 0)[1][0]))
    history = []
    for row in full:
        checksum = "unused-outside-four-case-scope"
        if row in cases:
            path = cache / "raw_cache/heldout/background" / (row["sample_id"] + ".npz")
            path.parent.mkdir(parents=True, exist_ok=True)
            values = {name: np.full((4, 2, 3), .5, np.float32) for name in replay.RAW_ARRAY_NAMES}
            np.savez(path, protocol_id=np.asarray(replay.PROTOCOL_ID), cache_revision=np.asarray(replay.CACHE_REVISION),
                     relative_path=np.asarray(row["relative_path"]), output_shape=np.asarray([6, 8], np.int32), **values)
            checksum = replay.sha256(path)
        history.append(dict(sample_id=row["sample_id"], relative_path=row["relative_path"], status="verified",
                            scores_reproduced=True, max_abs_error=0, cache_sha256=checksum))
    protocol = dict(brightness_factor=1, resize_factor=.625, patch_size=640, patch_overlap=128,
                    model_patch_size=16, evaluation_downscale=4, dino_training=False, trainable_parameters=0)
    replay.atomic_json(cache / "replay_status.json", dict(tool="aebad_s_background_replay12", protocol_id=replay.PROTOCOL_ID,
        cache_revision=replay.CACHE_REVISION, status="verified", precise_original_explanation_allowed=True,
        verified_queries=12, case_ids=[row["sample_id"] for row in full], cases=history, query_protocol=protocol))
    hashes = {name: replay.sha256(results / name) for name in
              ("sample_scores.csv", "support_manifest.csv", "memory_manifest.json", "complete.json")}
    hashes.update({key: replay.sha256(WORKSPACE / relative) for key, relative in runner.SOURCE_CODES.items()})
    hashes.update(mad=replay.sha256(results / "source_mad_state.npz"),
                  official_config=replay.sha256(official / "config.json"),
                  official_model=replay.sha256(official / "tracks/industrial/src/industrial/model.py"),
                  dinov3_hub=replay.sha256(dino / "hubconf.py"))
    replay.atomic_json(cache / "replay_inputs.json", {"identity": {"input_sha256": hashes, "image_sha256": image_hashes}})
    replay.atomic_json(cache / "environment_fixture.json", dict(float32_matmul_precision="highest",
        cuda_matmul_allow_tf32=False, cudnn_allow_tf32=True, software={}))
    replay.write_csv(cache / "support_provenance.csv", [dict(relative_path=row["relative_path"], sha256="old-replay-bytes") for row in supports])
    replay.atomic_json(diagnostic / "diagnostic_status.json", {"status": "planned_subset_complete"})
    replay.atomic_json(diagnostic / "verification.json", dict(formal_results_unchanged=True,
        planned_anomaly_spatial_complete=True, max_score_error=0))
    replay.write_csv(diagnostic / "selected_cases.csv", [dict(row, image_height=24, image_width=32) for row in cases])
    replay.write_csv(diagnostic / "defect_instances.csv", instances)
    for name in ("case_peaks.csv", "case_peak_layers.csv"):
        replay.write_csv(diagnostic / name, [{"unused": "preflight fixture"}])
    args = runner.build_parser().parse_args(["preflight", "--results-dir", str(results), "--cache-dir", str(cache),
        "--diagnostic-dir", str(diagnostic), "--data", str(root / "data"), "--official-root", str(official),
        "--dinov3-root", str(dino), "--checkpoint", str(checkpoint), "--output-dir", str(root / "new_trace")])
    args.started = runner.time.perf_counter()
    with patch.object(replay, "CHECKPOINT_SHA256", replay.sha256(checkpoint)), patch.object(runner, "dependencies", return_value={}):
        check = runner.preflight(args)
    return args, check


class TargetSupportTraceTests(unittest.TestCase):
    def test_fixed_four_cases_and_eight_original_supports(self):
        expected = (("test_0887", "test/good/background/IMG_7728.png"),
                    ("test_0921", "test/good/background/IMG_7765.png"),
                    ("test_0545", "test/fracture/background/IMG_9385.png"),
                    ("test_0023", "test/ablation/background/IMG_8863.png"))
        self.assertEqual(CASES, expected)
        self.assertEqual(runner.CASES, expected)
        full = runner.replay.select_cases(runner.replay.read_csv(RESULTS / "sample_scores.csv"))
        support = runner.replay.select_support(runner.replay.read_csv(RESULTS / "support_manifest.csv"), full)[8:]
        self.assertEqual([Path(row["relative_path"]).name for row in support], list(runner.replay.TARGET_NAMES))
        self.assertTrue(all(row["relative_path"].startswith("test/good/background/") for row in support))

    def test_default_mode_is_preflight(self):
        self.assertEqual(runner.build_parser().parse_args([]).mode, "preflight")

    def test_coincident_normal_peaks_keep_both_roles_and_row_major_ties(self):
        source = np.ones((9, 11), np.float32)
        source[1, 8] = source[3, 0] = 5
        positions = trace_positions(dict(sample_id="n", relative_path="normal", label=0),
                                    {method: source for method in METHODS}, None)
        self.assertEqual(len(positions), 1)
        self.assertEqual((positions[0]["output_y"], positions[0]["output_x"]), (1, 8))
        self.assertEqual(positions[0]["roles"], "source_full_peak;union_full_peak")
        self.assertEqual(positions[0]["query_region"], "unreviewed")

    def test_anomaly_roles_deduplicate_without_assuming_full_peak_is_defect(self):
        mask = np.zeros((64, 64), bool)
        mask[32:34, 32:34] = True
        source = np.zeros((64, 64), np.float32)
        union = np.zeros_like(source)
        source[32, 32], source[31, 33] = 3, 2
        union[0, 0], union[33, 33], union[31, 33] = 4, 2, 3
        positions = trace_positions(dict(sample_id="a", relative_path="anomaly", label=1),
                                    dict(zip(METHODS, (source, union))), mask)
        self.assertEqual(len(positions), 4)
        roles = [role for position in positions for role in position["roles"].split(";")]
        self.assertEqual(len(roles), 6)
        full = next(row for row in positions if "union_full_peak" in row["roles"])
        self.assertFalse(full["inside_gt_instance"])
        self.assertFalse(full["inside_nondefect_ring"])
        self.assertEqual(full["query_region"], "unreviewed")

    def test_fixed_ring_excludes_all_gt_and_keeps_16_original_pixels(self):
        component = np.zeros((80, 80), bool)
        component[40, 40] = True
        full = component.copy()
        full[40, 45] = True
        inside, ring = instance_regions(component, full, (80, 80))
        self.assertTrue(inside[40, 40])
        self.assertFalse(ring[40, 40])
        self.assertFalse(ring[40, 45])
        self.assertTrue(ring[40, 56])
        self.assertFalse(ring[40, 57])

    def test_stencil_uses_actual_grid_and_fixed_dominant_patch(self):
        position = dict(sample_id="n", position_id="p", roles="source_full_peak",
                        output_y=4, output_x=8, output_height=9, output_width=17)
        rows = position_patches([position], (2, 2))
        self.assertEqual(len(rows), 16)
        for layer in LAYERS:
            group = [row for row in rows if row["layer"] == layer]
            self.assertAlmostEqual(sum(row["interpolation_weight"] for row in group), 1)
            self.assertEqual(next(row for row in group if row["display_patch"])["query_flat_index"], 0)
        border = dict(position, output_y=0, output_x=0)
        self.assertEqual(len(position_patches([border], (2, 2))), 4)

    def test_flat_index_roundtrip_and_bounds(self):
        for index in (0, 4, 26, 34):
            y, x = flat_coordinates(index, (5, 7))
            self.assertEqual(y * 7 + x, index)
        with self.assertRaisesRegex(ValueError, "outside"):
            flat_coordinates(35, (5, 7))

    def test_per_image_top1_equals_explicit_concatenation(self):
        # Tests the actual candidate/ranking orchestration with a NumPy exact
        # reference callback. The production torch kernel is not available locally.
        generator = np.random.default_rng(7)
        query = np.zeros((3, 1280), np.float32)
        query[:, :4] = generator.normal(size=(3, 4))
        grids = [(1, 2), (1, 3)] * 4
        banks = []
        for grid in grids:
            bank = np.zeros((grid[0] * grid[1], 1280), np.float32)
            bank[:, :4] = generator.normal(size=(len(bank), 4))
            banks.append(bank)
        results = per_support_nearest(query, banks, grids, numpy_reference_top1)
        combined = np.concatenate(banks)
        distances, indices = numpy_reference_top1(query, combined)
        offsets = np.cumsum([0] + [len(bank) for bank in banks])
        for query_index, candidates in enumerate(results):
            self.assertEqual(len(candidates), 8)
            winner = candidates[0]
            global_index = offsets[winner["support_index"] - 1] + winner["support_flat_index"]
            self.assertEqual(global_index, indices[query_index])
            self.assertAlmostEqual(winner["distance_euclidean_over_1280"], float(distances[query_index] / 1280), places=8)
            self.assertEqual([row["support_rank"] for row in candidates], list(range(1, 9)))
            self.assertEqual(sum(row["winner"] for row in candidates), 1)

    def test_nn_ties_use_first_flat_then_first_manifest_image(self):
        query = np.zeros((1, 1280), np.float32)
        banks = [np.zeros((2, 1280), np.float32) for _ in range(8)]
        rows = per_support_nearest(query, banks, [(1, 2)] * 8, numpy_reference_top1)[0]
        self.assertEqual([row["support_index"] for row in rows], list(range(1, 9)))
        self.assertTrue(all(row["support_flat_index"] == 0 for row in rows))
        self.assertTrue(rows[0]["winner"])

    def test_invalid_support_geometry_or_returned_index_rejected(self):
        query = np.zeros((1, 1280), np.float32)
        banks = [np.zeros((2, 1280), np.float32) for _ in range(8)]
        with self.assertRaisesRegex(ValueError, "stored grid"):
            per_support_nearest(query, banks, [(2, 2)] * 8, numpy_reference_top1)
        with self.assertRaisesRegex(ValueError, "outside"):
            per_support_nearest(query, banks, [(1, 2)] * 8,
                                lambda query, bank: (np.zeros(1), np.asarray([2])))

    def test_mismatch_records_real_errors_and_cannot_pass(self):
        result = target_check(np.asarray([.2]), np.asarray([.1001]), np.asarray([.1]), np.asarray([.1]))
        self.assertFalse(result["verified"])
        self.assertAlmostEqual(result["target_abs_error"], .0001)
        self.assertAlmostEqual(result["target_rel_error"], .001)
        self.assertFalse(comparison([np.nan], [.1])["verified"])
        self.assertTrue(target_check(.2, .1, .1, .1)["verified"])
        self.assertFalse(comparison(1.00002, 1, score=True)["verified"])

    def test_actual_stitch_geometry_projection_covers_nonhardcoded_grids(self):
        model = SimpleNamespace(patch_exec=actual_patch_exec())
        for processed in ((640, 640), (688, 720), (1890, 1890)):
            grid = tuple(size // 16 for size in processed)
            geometry = image_geometry(model, 3024, 3024, grid, processed)
            for index in (0, grid[1] + 1, grid[0] * grid[1] - 1):
                row = patch_projection(index, geometry)
                self.assertGreaterEqual(row["original_y0"], 0)
                self.assertLessEqual(row["original_y1_exclusive"], 3024)
                self.assertEqual(row["coordinate_meaning"], "projected_input_token_cell_not_complete_receptive_field")
                self.assertAlmostEqual(row["original_x1_exclusive"] - row["original_x0"], 16 * 3024 / processed[1])
        with self.assertRaisesRegex(ValueError, "geometry"):
            image_geometry(model, 3024, 3024, (10, 10), (688, 720))

    def test_position_contributions_use_original_mad_and_preserve_negative_values(self):
        base = np.full((4, 2, 3), .1, np.float32)
        values = dict(official_raw=base, target_support_8=base / 2, target_raw_8=base / 2)
        position = dict(sample_id="n", position_id="p", roles="source_full_peak", output_y=3, output_x=4,
                        output_height=7, output_width=11)
        patches = position_patches([position], (2, 3))
        medians, scales = np.full(4, .2, np.float32), np.ones(4, np.float32)
        result = weighted_layers(values, patches, actual_calibrated(), medians, scales)
        self.assertAlmostEqual(sum(result["official_raw"]), -.1, places=6)
        self.assertAlmostEqual(sum(result["target_raw_8"]), -.15, places=6)

    def test_query_cache_contract_and_sparse_physical_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "query.npz"
            model = SimpleNamespace(patch_exec=actual_patch_exec())
            geometry = image_geometry(model, 640, 640, (40, 40), (640, 640))
            relative, contract = CASES[0][1], {"proof": "synthetic cache metadata test"}
            arrays = {f"layer_{layer}": np.full((3, 1280), layer, np.float32) for layer in LAYERS}
            np.savez(path, protocol_id=np.asarray(runner.replay.PROTOCOL_ID), relative_path=np.asarray(relative),
                     contract=np.asarray(json.dumps(contract)), grid=np.asarray([40, 40], np.int32),
                     geometry=np.asarray(json.dumps(geometry)), flat_positions=np.asarray([1, 4, 7], np.int64), **arrays)
            loaded = runner.load_query(path, relative, contract, geometry, [1, 7])
            self.assertEqual(loaded[7].shape, (2, 1280))
            with self.assertRaisesRegex(ValueError, "all selected"):
                runner.load_query(path, relative, contract, geometry, [2])
            with self.assertRaisesRegex(ValueError, "contract"):
                runner.load_query(path, relative, {"proof": "changed"}, geometry, [1])

    def test_help_and_preflight_import_do_not_require_torch(self):
        code = """
import importlib.abc, sys
class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise ImportError('Torch intentionally unavailable')
sys.meta_path.insert(0, BlockTorch())
from DINOv3.MADEqual.aebad_s_target_support_trace import runner
assert 'torch' not in sys.modules
assert runner.trace(None, {'errors': ['blocked']}) == 2
assert not runner.dependencies()['torch']['discoverable_without_import']
assert 'torch' not in sys.modules
runner.build_parser().parse_args(['--help'])
"""
        result = subprocess.run([sys.executable, "-c", code], cwd=WORKSPACE, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("preflight", result.stdout)

    def test_existing_query_feature_keeps_prior_hash_constraint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "query_features" / "test_0887.npz"
            path.parent.mkdir()
            path.write_bytes(b"synthetic feature checksum fixture")
            with self.assertRaisesRegex(ValueError, "requires its prior"):
                runner.query_cache_provenance(path, CASES[0][1])
            runner.replay.write_csv(root / "feature_provenance.csv", [
                {"kind": "query", "relative_path": CASES[0][1], "sha256": runner.replay.sha256(path)}])
            self.assertEqual(runner.query_cache_provenance(path, CASES[0][1])["feature_sha256"], runner.replay.sha256(path))
            path.write_bytes(b"modified features with the same declared metadata")
            with self.assertRaisesRegex(ValueError, "changed after"):
                runner.query_cache_provenance(path, CASES[0][1])

    def test_four_raw_target_only_preflight_resolves_root_without_training(self):
        with tempfile.TemporaryDirectory() as directory:
            args, check = minimal_trace_fixture(Path(directory))
            self.assertEqual(check["errors"], [])
            self.assertEqual(check["status"], "filesystem_preflight_passed")
            self.assertEqual((len(check["cases"]), len(check["supports"]), len(check["images"])), (4, 8, 14))
            self.assertEqual(check["resolved_paths"]["data"], str(Path(directory) / "data/AeBAD/AeBAD_S"))
            self.assertFalse((Path(check["resolved_paths"]["data"]) / "train/good").exists())
            self.assertFalse(check["backbone_loaded"])
            self.assertFalse(check["feature_extraction_started"])
            self.assertEqual(check["dataset_index_status"], "original_csv_ids_preserved_no_subset_renumbering")
            index = runner.trace_data_index(Path(args.data))
            try:
                for row in check["images"]:
                    self.assertEqual(index.describe(row["relative_path"])["sha256"], row["sha256"])
                    self.assertEqual(runner.trace_image(index, row["relative_path"], "RGB").size, (32, 24))
            finally:
                index.close()

    def test_four_raw_scope_keeps_history_hashes_and_default_twelve_requirement(self):
        with tempfile.TemporaryDirectory() as directory:
            _args, check = minimal_trace_fixture(Path(directory))
            cache, results = (Path(check["resolved_paths"][key]) for key in ("cache", "results"))
            self.assertEqual(runner.replay.verified_replay_scope(cache, results, required_cases=CASES), {row[0] for row in CASES})
            with self.assertRaisesRegex(ValueError, "exactly 12"):
                runner.replay.verified_replay_scope(cache, results)
            # A missing or modified outside-scope file has no bearing on these four.
            outside = cache / "raw_cache/heldout/background/test_0977.npz"
            outside.write_bytes(b"outside-scope data")
            self.assertEqual(len(runner.replay.verified_replay_scope(cache, results, required_cases=CASES)), 4)
            selected = cache / "raw_cache/heldout/background/test_0887.npz"
            original = selected.read_bytes()
            selected.write_bytes(b"modified selected distance arrays")
            with self.assertRaisesRegex(ValueError, "changed after"):
                runner.replay.verified_replay_scope(cache, results, required_cases=CASES)
            selected.unlink()
            with self.assertRaisesRegex(ValueError, "cache missing"):
                runner.replay.verified_replay_scope(cache, results, required_cases=CASES)
            selected.write_bytes(original)
            marker = cache / "replay_status.json"
            history = json.loads(marker.read_text(encoding="utf-8"))
            changed = json.loads(json.dumps(history))
            changed["cases"][0]["scores_reproduced"] = False
            runner.replay.atomic_json(marker, changed)
            with self.assertRaisesRegex(ValueError, "incomplete/unverified"):
                runner.replay.verified_replay_scope(cache, results, required_cases=CASES)
            runner.replay.atomic_json(marker, history)
            for invalid in ((), (CASES[0], CASES[0]), (("unknown", CASES[0][1]),)):
                with self.assertRaisesRegex(ValueError, "subset of the frozen cases"):
                    runner.replay.verified_replay_scope(cache, results, required_cases=invalid)
            (results / "sample_scores.csv").write_bytes(b"modified formal scores")
            with self.assertRaisesRegex(ValueError, "formal input differs"):
                runner.replay.verified_replay_scope(cache, results, required_cases=CASES)

    def test_trace_checks_four_case_scope_at_start_and_end(self):
        tree = ast.parse(Path(runner.__file__).read_text(encoding="utf-8"))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "verified_replay_scope"]
        self.assertEqual(len(calls), 2)
        for call in calls:
            scope = next(item.value for item in call.keywords if item.arg == "required_cases")
            self.assertEqual(scope.id, "CASES")

    def test_prior_trace_support_reuse_binds_frozen_inputs_and_file_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, check = minimal_trace_fixture(root)
            self.assertFalse(check["errors"])
            relative = check["supports"][0]["relative_path"]
            original_path = Path(check["resolved_paths"]["cache"]) / "support_features/target_matched/background/01.npz"
            original_path.parent.mkdir(parents=True)
            values = dict(protocol_id=np.asarray(runner.replay.PROTOCOL_ID), relative_path=np.asarray(relative),
                          grid=np.asarray([2, 3], np.int32),
                          **{f"layer_{layer}": np.zeros((6, 1280), np.float32) for layer in LAYERS})
            np.savez(original_path, **values)
            original_hash = runner.replay.sha256(original_path)
            prior = root / "prior_trace"
            path = prior / "support_features/target_matched/background/01.npz"
            path.parent.mkdir(parents=True)
            np.savez_compressed(path, **values)
            checksum = runner.replay.sha256(path)
            self.assertNotEqual(checksum, original_hash)
            with self.assertRaisesRegex(ValueError, "requires prior trace provenance"):
                runner.support_cache_provenance(path, relative, check, original_path, original_hash)
            geometry = {"grid": [2, 3], "synthetic_fixture": True}
            # Old trace CSVs without a contract column are bound by their saved inputs/environment.
            runner.replay.write_csv(prior / "feature_provenance.csv", [dict(kind="target_support", relative_path=relative,
                sha256=checksum, geometry=json.dumps(geometry))])
            runner.replay.atomic_json(prior / "trace_inputs.json", check)
            runner.replay.atomic_json(prior / "trace_environment.json", {"precision": check["expected_precision"]})
            proof = runner.support_cache_provenance(path, relative, check, original_path, original_hash)
            self.assertEqual(proof["origin"], "verified_prior_trace_target_support")
            self.assertEqual(proof["feature_sha256"], checksum)
            self.assertEqual(len(proof["evidence_files"]), 3)
            provenance_path = Path(check["resolved_paths"]["cache"]) / "support_provenance.csv"
            originals = runner.replay.read_csv(provenance_path)
            next(row for row in originals if row["relative_path"] == relative)["sha256"] = original_hash
            runner.replay.write_csv(provenance_path, originals)
            args.support_cache_dir = str(path.parent)
            args.started = runner.time.perf_counter()
            with patch.object(runner.replay, "CHECKPOINT_SHA256", runner.replay.sha256(Path(args.checkpoint))), \
                    patch.object(runner, "dependencies", return_value={}):
                reused = runner.preflight(args)
            self.assertEqual(reused["errors"], [])
            feature = next(row for row in reused["feature_inputs"] if row["relative_path"] == relative)
            self.assertEqual(feature["origin"], "verified_prior_trace_target_support")
            self.assertEqual(feature["prior_geometry"], geometry)
            self.assertEqual(len([row for row in reused["required_files"] if row["kind"].startswith("support_feature_prior_")]), 3)
            changed = json.loads(json.dumps(check))
            changed["image_sha256"][relative] = "different source pixels"
            with self.assertRaisesRegex(ValueError, "frozen extraction inputs"):
                runner.support_cache_provenance(path, relative, changed, original_path, original_hash)
            changed = json.loads(json.dumps(check))
            changed["expected_precision"]["cuda_matmul_allow_tf32"] = True
            with self.assertRaisesRegex(ValueError, "frozen extraction inputs"):
                runner.support_cache_provenance(path, relative, changed, original_path, original_hash)
            # A modified original replay cache cannot opt in to the new provenance path.
            original_path.write_bytes(path.read_bytes())
            with self.assertRaisesRegex(ValueError, "original Target support"):
                runner.support_cache_provenance(original_path, relative, check, original_path, original_hash)
            values["layer_7"][0, 0] = 1
            np.savez_compressed(path, **values)
            with self.assertRaisesRegex(ValueError, "changed or lacks"):
                runner.support_cache_provenance(path, relative, check, original_path, original_hash)

    def test_synthetic_display_is_fixed_and_marks_unverified(self):
        from PIL import Image
        from DINOv3.MADEqual.aebad_s_target_support_trace.visualization import correspondence_figure
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            geometry = image_geometry(SimpleNamespace(patch_exec=actual_patch_exec()), 640, 640, (40, 40), (640, 640))
            cell = patch_projection(205, geometry)
            image = Image.new("RGB", (640, 640), "lightgray")
            image.save(root / "support.png")
            winners = [dict(layer=layer, support_index=1, support_relative_path="support.png",
                support_flat_index=205, support_patch_y=5, support_patch_x=5,
                distance_euclidean_over_1280=.11, cached_d_T=.1, correspondence_verified=False, **cell) for layer in LAYERS]
            position = dict(position_id="synthetic_p01", correspondence_verified=False,
                original_x_center=90, original_y_center=90, roles="source_full_peak;union_full_peak",
                source_score=.2, union_score=.1, display_query_flat_index=205)
            path = root / "figure.png"
            correspondence_figure(image, position, cell, winners, root, path)
            with Image.open(path) as result:
                self.assertEqual(result.size, (1500, 850))
                pixels = np.asarray(result)
                self.assertTrue(((pixels[:30, :, 0] > 150) & (pixels[:30, :, 1] < 100)).any())

    def test_failed_preflight_never_enters_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(runner, "trace", side_effect=AssertionError("must not infer")), contextlib.redirect_stdout(io.StringIO()):
                result = runner.main(["trace", "--results-dir", str(root / "missing_results"),
                    "--cache-dir", str(root / "missing_cache"), "--diagnostic-dir", str(root / "missing_diag"),
                    "--data", str(root / "missing_data"), "--checkpoint", str(root / "missing_checkpoint"),
                    "--dinov3-root", str(root / "missing_dino"), "--output-dir", str(root / "output")])
            self.assertEqual(result, 2)
            report = json.loads((root / "output/preflight.json").read_text(encoding="utf-8"))
            self.assertFalse(report["backbone_loaded"])
            self.assertFalse(report["feature_extraction_started"])
            self.assertFalse((root / "output/trace_status.json").exists())

    def test_trace_has_no_source_memory_full_run_or_full_observe_call(self):
        tree = ast.parse(Path(runner.__file__).read_text(encoding="utf-8"))
        forbidden = {"run", "observe_fewshot_pil", "observe_pil", "_load_memory", "build_exact_official_memory",
                     "_fit_source_mad", "fit_calibrators", "load_official_banks", "exact_prefix_minima"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
                self.assertNotIn(name, forbidden)

    def test_existing_output_or_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {name: root / name for name in ("project", "results", "cache", "diagnostic", "data", "support", "query")}
            paths["output"] = paths["cache"] / "trace"
            with self.assertRaisesRegex(ValueError, "protected"):
                runner.guard_output(paths)
            paths["output"] = root / "trace"
            paths["output"].mkdir()
            (paths["output"] / "trace_status.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "no overwrite"):
                runner.guard_output(paths)


if __name__ == "__main__":
    unittest.main()
