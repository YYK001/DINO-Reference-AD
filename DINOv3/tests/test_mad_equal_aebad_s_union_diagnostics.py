"""Small cache/CSV tests runnable with unittest and no PyTorch."""
from __future__ import annotations

import importlib.abc
import contextlib
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

from DINOv3.MADEqual.aebad_s_union_diagnostics.analysis import (
    CACHE_REVISION, PROTOCOL_ID, ranking, validate_inputs,
)
from DINOv3.MADEqual.aebad_s_union_diagnostics.spatial import (
    components, crop_diagnostics, instance_response, interpolation_stencil, load_cache,
)
from DINOv3.MADEqual.aebad_s_union_diagnostics import runner
from DINOv3.MADEqual.aebad_s_target_fewshot import replay12


class UnionDiagnosticTests(unittest.TestCase):
    def test_replay12_completion_checks_gt_through_diagnostic_run(self):
        # Run the real CSV/GT/instance/crop/status pipeline with a synthetic
        # dataset and map backend. No model, Torch or real distance inference.
        workspace = Path(__file__).resolve().parents[2]
        results = workspace / "server_archives/aebad_s_target_fewshot_seed42_v1_formal_results/aebad_s_target_fewshot_seed42_v1"
        rows = runner.read_csv(results / "sample_scores.csv")
        background = {row["relative_path"]: row for row in rows if row["domain"] == "background" and row["method"] == runner.SOURCE}
        scores = {(row["relative_path"], row["method"]): float(row["image_score"]) for row in rows}
        support_paths = {row["relative_path"] for row in runner.read_csv(results / "support_manifest.csv")
                         if row["branch"] == "target_matched" and row["domain"] == "background"}
        scope = {sample_id for sample_id, _ in replay12.CASES}
        outside = next(row["sample_id"] for row in background.values() if int(row["label"]) and row["sample_id"] not in scope)
        for missing_ids, expected in ((set(), "planned_subset_complete"), ({"test_0023"}, "partial_evidence"),
                                      ({"test_0545"}, "partial_evidence"), ({"test_0023", "test_0545"}, "partial_evidence"),
                                      ({outside}, "planned_subset_complete")):
            with self.subTest(missing=missing_ids), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                mask = np.zeros((64, 64), np.uint8)
                mask[20:30, 20:30] = 255
                dataset = SimpleNamespace(
                    ids=dict({relative: row["sample_id"] for relative, row in background.items()},
                             **{relative: "synthetic_support_id" for relative in support_paths}),
                    names=set(background) | support_paths | {runner.gt_path(relative) for relative, row in background.items()
                           if int(row["label"]) and row["sample_id"] not in missing_ids},
                    image=lambda relative, mode="RGB": Image.fromarray(mask).convert(mode) if relative.startswith("ground_truth/") else Image.new(mode, (64, 64)),
                    close=lambda: None)
                values = dict(official_raw=np.full((4, 2, 3), 2, np.float32), target_raw_8=np.ones((4, 2, 3), np.float32))
                backend = SimpleNamespace(
                    mad=SimpleNamespace(scales=np.ones(4)),
                    patch_mask=lambda image, grid: np.ones(grid, bool),
                    maps=lambda path, relative: (values, {method: np.full((64, 64), scores[(relative, method)]) for method in runner.METHODS}, (64, 64)))
                for sample_id in scope:
                    path = root / "cache/raw_cache/heldout/background" / f"{sample_id}.npz"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"synthetic backend marker")
                args = SimpleNamespace(results_dir=results, cache_dir=root / "cache", data=root / "data", output_dir=root / "output",
                                       score_tolerance=1e-5, skip_archive_inventory=True)
                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch.object(replay12, "verified_replay_scope", return_value=scope))
                    stack.enter_context(patch.object(runner, "Dataset", return_value=dataset))
                    stack.enter_context(patch.object(runner, "load_cache", return_value=(values, (64, 64), 0)))
                    stack.enter_context(patch.object(runner, "MapBackend", return_value=backend))
                    stack.enter_context(patch.object(runner, "montage"))
                    stack.enter_context(patch.object(runner, "peak_rows", return_value=([], [])))
                    stack.enter_context(patch.object(runner, "plot_maps", return_value={}))
                    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    runner.run(args)
                status = json.loads((root / "output/diagnostic_status.json").read_text(encoding="utf-8"))
                verification = json.loads((root / "output/verification.json").read_text(encoding="utf-8"))
                self.assertEqual(status["status"], expected)
                self.assertEqual(status["reconstructed_images"], 12)
                self.assertEqual(status["required_anomaly_gt_images"], 2)
                self.assertEqual(set(status["missing_required_gt_ids"]), missing_ids & scope)
                self.assertEqual(status["planned_anomaly_spatial_complete"], expected == "planned_subset_complete")
                self.assertEqual(verification["outside_scope_images"], 285)

    def test_anomaly_diagnostics_must_finish_before_completion(self):
        mask = np.zeros((64, 64), bool)
        mask[20:30, 20:30] = True
        backend = SimpleNamespace(mad=SimpleNamespace(scales=np.ones(4)), patch_mask=lambda image, grid: np.ones(grid, bool))
        maps = {method: np.ones((64, 64)) for method in runner.METHODS}
        values = dict(official_raw=np.ones((4, 2, 3)), target_raw_8=np.zeros((4, 2, 3)))
        availability = [dict(sample_id="a", label=1, gt_status="available")]
        instances = [dict(sample_id="a", **instance_response(mask, mask, maps, values, backend))]
        crops = [dict(sample_id="a", method=method, **crop_diagnostics(maps[method], mask, 1)) for method in runner.METHODS]
        reductions = [dict(sample_id="a", layer=layer, region="GT_defect_patches") for layer in runner.LAYERS]
        base = dict(reconstructed_images=12, planned_cache_images=12, scope_kind="verified_replay12")
        for responses, resize_crop, defect_reduction in (([], crops, reductions), (instances, [], reductions), (instances, crops, [])):
            checks = runner.spatial_completion(availability, {"a"}, responses, resize_crop, defect_reduction)
            self.assertEqual(checks["incomplete_anomaly_spatial_ids"], ["a"])
            self.assertEqual(runner.diagnostic_status(dict(base, **checks)), "partial_evidence")
        checks = runner.spatial_completion(availability, {"a"}, instances, crops, reductions)
        self.assertEqual(runner.diagnostic_status(dict(base, **checks)), "planned_subset_complete")

    def _input_fixture(self):
        supports = [dict(branch="target_matched", domain="background", support_index=str(i),
                         relative_path=f"test/good/background/support{i}.png", label="0",
                         **{f"included_in_K{k}": str(int(i <= k)) for k in (1, 2, 4, 8)}) for i in range(1, 9)]
        paths = [f"test/good/background/n{i}.png" for i in range(85)]
        paths += [f"test/fracture/background/a{i}.png" for i in range(212)]
        ids = {path: f"test_{i:04d}" for i, path in enumerate(sorted(paths + [r["relative_path"] for r in supports]))}
        rows = [dict(sample_id=ids[path], relative_path=path, domain="background",
                     label=str(int("/good/" not in path)), method=method, image_score=score)
                for path in paths for method, score in (("MAD_EQUAL", "2"), ("TARGET_MATCHED_K8_MAD", "1"))]
        return rows, supports, SimpleNamespace(ids=ids)

    def test_all_methods_share_heldout_and_preserve_sample_id(self):
        rows, supports, dataset = self._input_fixture()
        samples, pool = validate_inputs(rows, supports, dataset)
        self.assertEqual((len(samples), len(pool)), (297, 8))
        rows[0]["sample_id"] = "test_9999"
        with self.assertRaisesRegex(ValueError, "sample_id/path"):
            validate_inputs(rows, supports, dataset)

    def test_support_leakage_is_rejected(self):
        rows, supports, dataset = self._input_fixture()
        path = supports[0]["relative_path"]
        rows[0].update(relative_path=path, sample_id=dataset.ids[path])
        with self.assertRaisesRegex(ValueError, "support leakage"):
            validate_inputs(rows, supports, dataset)

    def test_incomplete_method_is_rejected(self):
        rows, supports, dataset = self._input_fixture()
        with self.assertRaisesRegex(ValueError, "complete background"):
            validate_inputs(rows[:-1], supports, dataset)

    def test_wrong_support_prefix_is_rejected(self):
        rows, supports, dataset = self._input_fixture()
        supports[0]["included_in_K1"] = "0"
        with self.assertRaisesRegex(ValueError, "not nested"):
            validate_inputs(rows, supports, dataset)

    def test_ties_and_pair_transitions(self):
        samples = [
            dict(sample_id="n0", relative_path="n0", label=0, source_score=2.0, union_score=1.0),
            dict(sample_id="n1", relative_path="n1", label=0, source_score=1.0, union_score=1.0),
            dict(sample_id="a0", relative_path="a0", label=1, source_score=1.0, union_score=1.0),
            dict(sample_id="a1", relative_path="a1", label=1, source_score=3.0, union_score=0.0),
        ]
        summary, pairs, images = ranking(samples)
        self.assertEqual(summary["source_weighted_errors"], 1.5)
        self.assertEqual(summary["source_image_AUROC"], 0.625)
        self.assertEqual(summary["union_weighted_errors"], 3.0)
        self.assertEqual(summary["union_image_AUROC"], 0.25)
        self.assertEqual((summary["source_ties"], summary["union_ties"]), (1, 2))
        self.assertEqual((summary["improved_pairs"], summary["worsened_pairs"]), (1, 2))
        self.assertEqual(sum(row["union_errors"] for row in images if row["label"] == 0), 3.0)
        self.assertEqual(sum(row["union_errors"] for row in images if row["label"] == 1), 3.0)
        self.assertEqual(len(pairs), 4)

    def test_top_normal_sort_uses_path_for_ties(self):
        samples = [dict(sample_id="n2", relative_path="b", label=0, source_score=2, union_score=2),
                   dict(sample_id="n1", relative_path="a", label=0, source_score=2, union_score=2),
                   dict(sample_id="a", relative_path="c", label=1, source_score=1, union_score=1)]
        self.assertEqual(ranking(samples)[0]["top10"], ["n1", "n2"])

    def test_cache_semantics_and_shape_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test_0001.npz"
            base = np.full((4, 2, 3), 2, np.float32)
            target = np.full_like(base, 1)
            source_extra = np.full_like(base, 3)
            values = dict(protocol_id=np.asarray(PROTOCOL_ID), cache_revision=np.asarray(CACHE_REVISION),
                          relative_path=np.asarray("test/good/background/example.png"),
                          output_shape=np.asarray([8, 12], np.int32), official_raw=base)
            for shot in (1, 2, 4, 8):
                values.update({f"target_support_{shot}": target, f"target_raw_{shot}": np.minimum(base, target),
                               f"source_support_{shot}": source_extra, f"source_raw_{shot}": np.minimum(base, source_extra)})
            np.savez(path, **values)
            loaded, shape, error = load_cache(path, "test/good/background/example.png")
            self.assertEqual(shape, (8, 12))
            self.assertEqual(error, 0)
            self.assertTrue(np.array_equal(loaded["official_raw"], base))
            with self.assertRaisesRegex(ValueError, "relative_path"):
                load_cache(path, "test/good/background/wrong.png")
            values["target_raw_8"] = np.full_like(base, 0.5)
            np.savez(path, **values)
            with self.assertRaisesRegex(ValueError, "layer-wise min"):
                load_cache(path, "test/good/background/example.png")
            values["target_raw_8"] = target
            values["target_support_8"] = np.full((4, 3, 2), 1, np.float32)
            np.savez(path, **values)
            with self.assertRaisesRegex(ValueError, "distance arrays"):
                load_cache(path, "test/good/background/example.png")

    def test_resize_and_boundary_removal_are_separate(self):
        image = np.zeros((512, 512), np.float32)
        image[0, 0] = 16
        image[200:260, 200:260] = 2
        mask = np.zeros(image.shape, bool)
        mask[0:10, 0:10] = True
        row = crop_diagnostics(image, mask, 1)
        self.assertGreater(row["resize_drop"], 0)
        self.assertGreater(row["boundary_removal_drop"], 0)
        self.assertAlmostEqual(row["total_drop"], row["resize_drop"] + row["boundary_removal_drop"])
        self.assertTrue(row["all_resize_peaks_cropped"])
        self.assertEqual(row["original_label"], 1)
        self.assertEqual(row["archived_mmr_label_rule"], 0)

    def test_negative_maps_are_not_clipped(self):
        row = crop_diagnostics(np.full((256, 256), -3, np.float32), None, 0)
        self.assertEqual(row["original_max"], -3)
        self.assertEqual(row["crop_224_max"], -3)
        self.assertFalse(row["all_resize_peaks_cropped"])

    def test_half_pixel_stencil_clamps_edges_and_preserves_weights(self):
        self.assertEqual(interpolation_stencil((0, 0), (2, 2), (4, 4)), [(0, 0, 1.0)])
        self.assertEqual(interpolation_stencil((3, 3), (2, 2), (4, 4)), [(1, 1, 1.0)])
        stencil = interpolation_stencil((1, 1), (2, 2), (4, 4))
        self.assertEqual(stencil, [(0, 0, 0.5625), (0, 1, 0.1875), (1, 0, 0.1875), (1, 1, 0.0625)])
        self.assertEqual(sum(weight for _, _, weight in stencil), 1)

    def test_min_must_precede_interpolation(self):
        base = np.array([[0, 8], [0, 8]], np.float32)
        target = 8 - base
        stencil = interpolation_stencil((1, 1), (2, 2), (3, 3))
        interp = lambda a: sum(float(a[y, x]) * weight for y, x, weight in stencil)
        self.assertEqual(interp(np.minimum(base, target)), 0)
        self.assertEqual(min(interp(base), interp(target)), 4)

    def test_instances_use_8_connectivity(self):
        mask = np.zeros((8, 8), bool)
        mask[1, 1] = mask[2, 2] = mask[6, 6] = True
        _labels, rows = components(mask)
        self.assertEqual([row["area_original_pixels"] for row in rows], [2, 1])
        self.assertEqual(rows[0]["bbox_y1_exclusive"], 3)

    def test_local_margin_excludes_other_defects(self):
        instance = np.zeros((64, 64), bool)
        instance[25:30, 25:30] = True
        gt = instance.copy()
        gt[32:36, 25:30] = True
        source = np.full((64, 64), 2, np.float32)
        source[instance] = 5
        source[gt & ~instance] = 100
        union = source.copy()
        union[instance] = 4
        backend = SimpleNamespace(mad=SimpleNamespace(scales=np.ones(4, np.float32)),
                                  patch_mask=lambda image, grid: np.asarray(image) > 0)
        values = dict(official_raw=np.ones((4, 64, 64), np.float32),
                      target_raw_8=np.zeros((4, 64, 64), np.float32))
        row = instance_response(instance, gt, {"MAD_EQUAL": source, "TARGET_MATCHED_K8_MAD": union}, values, backend)
        self.assertEqual(row["source_margin"], 3)
        self.assertEqual(row["union_margin"], 2)
        self.assertEqual(row["margin_change"], -1)
        self.assertEqual(row["patch_coverage"], 25)
        row = instance_response(instance, np.ones((64, 64), bool), {"MAD_EQUAL": source, "TARGET_MATCHED_K8_MAD": union}, values, backend)
        self.assertEqual(row["source_margin"], "")
        self.assertEqual(row["union_margin"], "")

    def test_diagnostic_import_does_not_require_torch(self):
        code = """
import importlib.abc, sys
class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise ImportError('PyTorch intentionally unavailable')
sys.meta_path.insert(0, BlockTorch())
from DINOv3.MADEqual.aebad_s_union_diagnostics import runner
assert 'torch' not in sys.modules
runner.build_parser().parse_args(['--results-dir', 'r', '--data', 'd', '--output-dir', 'o'])
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                cwd=Path(__file__).resolve().parents[2])
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
