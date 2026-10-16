"""Lightweight replay planning, pure original cache I/O and fail-closed tests."""
from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from DINOv3.MADEqual.aebad_s_target_fewshot import replay12 as replay

WORKSPACE = Path(__file__).resolve().parents[2]
RESULTS = WORKSPACE / "server_archives/aebad_s_target_fewshot_seed42_v1_formal_results/aebad_s_target_fewshot_seed42_v1"
SELECTED = WORKSPACE / "DINOv3/outputs/mad_equal/aebad_s_union_background_diagnostic_seed42_v1_review1/selected_cases.csv"


def pure_original_io() -> SimpleNamespace:
    # Execute the actual original NumPy-only save/load functions, without
    # importing the heavyweight runner or substituting their implementations.
    tree = ast.parse((WORKSPACE / "DINOv3/MADEqual/aebad_s_target_fewshot/runner.py").read_text(encoding="utf-8"))
    names = {"_atomic_npz", "_save_observation", "_load_observation"}
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    body += [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    namespace = dict(np=np, os=os, Path=Path, PROTOCOL_ID=replay.PROTOCOL_ID,
                     CACHE_REVISION=replay.CACHE_REVISION, SHOT_COUNTS=replay.SHOTS, LAYERS=replay.LAYERS)
    exec(compile(module, "original_cache_functions", "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


def synthetic_cache(path: Path) -> tuple[SimpleNamespace, dict]:
    io = pure_original_io()
    base = np.full((4, 2, 3), 2, np.float32)
    source = {k: np.full_like(base, 4 / k) for k in replay.SHOTS}
    target = {k: np.full_like(base, 3 / k) for k in replay.SHOTS}
    observation = SimpleNamespace(official_raw=base,
                                  source_raw={k: np.minimum(base, source[k]) for k in replay.SHOTS},
                                  target_raw={k: np.minimum(base, target[k]) for k in replay.SHOTS},
                                  source_support_raw=source, target_support_raw=target)
    case = dict(sample_id=replay.CASES[0][0], relative_path=replay.CASES[0][1], source_score=2, union_score=0.375)
    io._save_observation(path, relative=case["relative_path"], output_shape=(2, 3), observed=observation)
    return io, case


class Replay12Tests(unittest.TestCase):
    def test_exact_fixed_cases_exclude_three_smallest_instances(self):
        cases = replay.select_cases(replay.read_csv(RESULTS / "sample_scores.csv"), replay.read_csv(SELECTED))
        self.assertEqual([(row["sample_id"], row["relative_path"]) for row in cases], list(replay.CASES))
        self.assertEqual([row["label"] for row in cases], [0] * 10 + [1] * 2)
        self.assertTrue({"test_0002", "test_0004", "test_0015"}.isdisjoint(row["sample_id"] for row in cases))

    def test_subsample_renumbering_is_rejected(self):
        rows = replay.read_csv(RESULTS / "sample_scores.csv")
        for row in rows:
            if row["relative_path"] == replay.CASES[-1][1]:
                row["sample_id"] = "test_0000"
        with self.assertRaisesRegex(ValueError, "fixed sample_id"):
            replay.select_cases(rows)

    def test_target_and_source_extra_order_and_prefixes(self):
        cases = replay.select_cases(replay.read_csv(RESULTS / "sample_scores.csv"))
        supports = replay.select_support(replay.read_csv(RESULTS / "support_manifest.csv"), cases)
        self.assertEqual(len(supports), 16)
        self.assertEqual([Path(row["relative_path"]).name for row in supports[8:]], list(replay.TARGET_NAMES))
        self.assertEqual([row["cache_relative"] for row in supports[:2]], ["source_extra/01.npz", "source_extra/02.npz"])
        rows = replay.read_csv(RESULTS / "support_manifest.csv")
        selected = [row for row in rows if row["branch"] == "target_matched" and row["domain"] == "background"]
        selected[0]["relative_path"], selected[1]["relative_path"] = selected[1]["relative_path"], selected[0]["relative_path"]
        with self.assertRaisesRegex(ValueError, "manifest order"):
            replay.select_support(rows, cases)

    def test_preflight_and_help_import_without_torch(self):
        code = """
import importlib.abc, sys
class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise ImportError('PyTorch deliberately unavailable')
sys.meta_path.insert(0, BlockTorch())
from DINOv3.MADEqual.aebad_s_target_fewshot import replay12
assert 'torch' not in sys.modules
assert replay12.replay(None, {'errors': ['missing memory', 'missing MAD']}) == 2
replay12.build_parser().parse_args(['--help'])
"""
        result = subprocess.run([sys.executable, "-c", code], cwd=WORKSPACE, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("preflight", result.stdout)
        self.assertIn("replay", result.stdout)

    def test_missing_memory_or_mad_blocks_before_replay(self):
        # No fake checkpoint/model is loaded. Use a filename-index stand-in to
        # keep this policy test small; real image/zip checks run separately.
        class Index:
            root = WORKSPACE / "datasets" / "not_used_by_policy_test"
            source_count = 521
            ids = {path: sample_id for sample_id, path in replay.CASES}
            ids.update({f"unopened_test_{i}": f"unused_{i}" for i in range(1639 - 12)})
            def __init__(self, _path):
                pass
            def describe(self, relative):
                return dict(relative_path=relative, height=3024, width=3024, sha256="policy-test-only")
            def close(self):
                pass
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = ["replay", "--project-root", str(WORKSPACE), "--results-dir", str(RESULTS),
                    "--memory", str(root / "missing_memory.pt"), "--mad", str(root / "missing_mad.npz"),
                    "--aebad-root", str(Index.root), "--dinov3-root", str(root / "missing_dino"),
                    "--output-dir", str(root / "output")]
            with patch.object(replay, "DataIndex", Index), patch.object(replay, "replay", side_effect=AssertionError("must not start replay")), contextlib.redirect_stdout(io.StringIO()):
                code = replay.main(argv)
            self.assertEqual(code, 2)
            report = json.loads((root / "output/preflight.json").read_text(encoding="utf-8"))
            self.assertEqual((len(report["cases"]), len(report["supports"])), (12, 16))
            self.assertTrue(any("missing required memory" in error for error in report["errors"]))
            self.assertTrue(any("missing required mad" in error for error in report["errors"]))
            self.assertFalse(report["backbone_loaded"])
            self.assertFalse(report["query_inference_started"])

    def test_no_rebuild_refit_full_run_or_gt_load_call_sites(self):
        tree = ast.parse(Path(replay.__file__).read_text(encoding="utf-8"))
        forbidden = {"build_exact_official_memory", "fit_calibrators", "_fit_source_mad", "_save_memory", "load_phase_b_records", "run"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
                self.assertNotIn(name, forbidden)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "observe_fewshot_pil"]
        self.assertEqual(len(calls), 1)
        self.assertIn("source_count=521", Path(replay.__file__).read_text(encoding="utf-8"))

    def test_original_full_cache_format_loads_in_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.npz"
            io, case = synthetic_cache(path)
            values, shape = io._load_observation(path, relative=case["relative_path"])
            diagnostic_values, diagnostic_shape, error = replay.validate_raw(path, case)
            self.assertEqual(set(values), set(replay.RAW_ARRAY_NAMES))
            self.assertEqual((shape, diagnostic_shape, error), ((2, 3), (2, 3), 0))
            self.assertEqual(set(diagnostic_values), set(values))
            self.assertFalse(np.array_equal(values["target_raw_1"], values["target_raw_8"]))
            with np.load(path, allow_pickle=False) as payload:
                self.assertEqual(set(payload.files), set(replay.RAW_ARRAY_NAMES) | {"protocol_id", "cache_revision", "relative_path", "output_shape"})

    def test_support_header_checks_original_grid_dimension_and_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "support.npz"
            relative = "test/good/background/IMG_7785.png"
            values = dict(protocol_id=np.asarray(replay.PROTOCOL_ID), relative_path=np.asarray(relative), grid=np.asarray([2, 3], np.int32),
                          **{f"layer_{layer}": np.zeros((6, 1280), np.float32) for layer in replay.LAYERS})
            np.savez(path, **values)
            replay.check_support_header(path, relative)
            with self.assertRaisesRegex(ValueError, "protocol/path"):
                replay.check_support_header(path, "test/good/background/wrong.png")
            values["layer_31"] = np.zeros((5, 1280), np.float32)
            np.savez(path, **values)
            with self.assertRaisesRegex(ValueError, "L31 shape/dtype"):
                replay.check_support_header(path, relative)

    def test_partial_dataset_files_are_not_renumbered(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "subset.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("AeBAD/AeBAD_S/" + replay.CASES[0][1], b"filename-only fixture")
                archive.writestr("AeBAD/AeBAD_S/train/good/background/IMG_7295.png", b"filename-only fixture")
            index = replay.DataIndex(path)
            try:
                self.assertEqual(index.test_count, 1)
                self.assertEqual(index.ids, {})
                self.assertEqual(index.source_count, 1)
            finally:
                index.close()

    def test_mismatch_preserves_cache_and_reports_real_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.npz"
            io, case = synthetic_cache(path)
            io._method_map = lambda values, method, mad, shape: values["official_raw" if method == "MAD_EQUAL" else "target_raw_8"].mean(axis=0)
            before = replay.sha256(path)
            changed = dict(case, union_score=1)
            row = replay.verify_case(path, changed, io, None, 1e-5, (2, 3), (2, 3))
            self.assertFalse(row["scores_reproduced"])
            self.assertEqual(row["union_abs_error"], 0.625)
            self.assertEqual(replay.sha256(path), before)
            with self.assertRaisesRegex(ValueError, "geometry"):
                replay.verify_case(path, case, io, None, 1e-5, (4, 6), (2, 3))

    def test_invalid_existing_cache_is_rejected_before_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.npz"
            io, case = synthetic_cache(path)
            with np.load(path, allow_pickle=False) as payload:
                values = {key: payload[key] for key in payload.files}
            values["target_raw_2"] = np.zeros((4, 2, 3), np.float32)
            np.savez(path, **values)
            with self.assertRaisesRegex(ValueError, "K2 union"):
                replay.validate_raw(path, case)

    def _scope_fixture(self, root: Path):
        cache, results = root / "cache", root / "formal"
        results.mkdir()
        files = {}
        for name in ("sample_scores.csv", "support_manifest.csv", "memory_manifest.json", "complete.json", "source_mad_state.npz"):
            path = results / name
            path.write_text("immutable synthetic input", encoding="utf-8")
            files["mad" if name == "source_mad_state.npz" else name] = replay.sha256(path)
        rows = []
        for sample_id, relative in replay.CASES:
            path = cache / "raw_cache/heldout/background" / f"{sample_id}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic cache for hash-gate test only")
            rows.append(dict(sample_id=sample_id, relative_path=relative, status="verified",
                             scores_reproduced=True, max_abs_error=0.0, cache_sha256=replay.sha256(path)))
        status = dict(tool="aebad_s_background_replay12", protocol_id=replay.PROTOCOL_ID,
                      cache_revision=replay.CACHE_REVISION, status="verified", precise_original_explanation_allowed=True,
                      verified_queries=12, case_ids=[sample_id for sample_id, _ in replay.CASES], cases=rows)
        replay.atomic_json(cache / "replay_inputs.json", {"identity": {"input_sha256": files}})
        replay.atomic_json(cache / "replay_status.json", status)
        return cache, results, status

    def test_diagnostics_accept_exactly_verified_12_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            cache, results, status = self._scope_fixture(Path(directory))
            self.assertEqual(replay.verified_replay_scope(cache, results), {sample_id for sample_id, _ in replay.CASES})
            status["status"] = "not_reproduced"
            replay.atomic_json(cache / "replay_status.json", status)
            with self.assertRaisesRegex(ValueError, "incomplete/unverified"):
                replay.verified_replay_scope(cache, results)

    def test_tampered_cache_cannot_be_used_as_exact_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            cache, results, _status = self._scope_fixture(Path(directory))
            (cache / "raw_cache/heldout/background/test_0977.npz").write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "changed after"):
                replay.verified_replay_scope(cache, results)

    def _resume_fixture(self, root: Path):
        cache, results, status = self._scope_fixture(root)
        check = dict(errors=[], resolved_paths={"output": str(cache)}, required_files=[], images=[], supports=[],
                     cases=[dict(sample_id=sample_id, relative_path=relative) for sample_id, relative in replay.CASES])
        replay.atomic_json(cache / "replay_inputs.json", {"identity": replay.input_identity(check)})
        replay.write_csv(cache / "replay_verification.csv", status["cases"])
        return cache, results, status, check, SimpleNamespace(score_tolerance=1e-5)

    def test_resume_rejects_score_invisible_source_extra_change_before_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            cache, _results, status, check, args = self._resume_fixture(Path(directory))
            path = cache / "raw_cache/heldout/background" / f"{replay.CASES[0][0]}.npz"
            original, case = synthetic_cache(path)
            original._method_map = lambda values, method, mad, shape: values["official_raw" if method == "MAD_EQUAL" else "target_raw_8"].mean(axis=0)
            status["cases"][0]["cache_sha256"] = replay.sha256(path)
            replay.atomic_json(cache / "replay_status.json", status)
            replay.write_csv(cache / "replay_verification.csv", status["cases"])
            with np.load(path, allow_pickle=False) as payload:
                values = {key: payload[key] for key in payload.files}
            for key in ("source_support_8", "source_raw_8"):
                values[key] = np.full_like(values[key], 0.4)
            np.savez(path, **values)
            # Reproduce the review: structure and both scores still pass.
            verified = replay.verify_case(path, case, original, None, 1e-5, (2, 3), (2, 3))
            self.assertTrue(verified["scores_reproduced"])
            self.assertNotEqual(verified["cache_sha256"], status["cases"][0]["cache_sha256"])
            preserved = {name: (cache / name).read_bytes() for name in ("replay_status.json", "replay_inputs.json", "replay_verification.csv")}
            modified_bytes = path.read_bytes()
            # Real replay entry must fail before numerical verification or any
            # state reset, and a second retry must retain the same constraint.
            for _ in range(2):
                with patch.object(replay, "verify_case", side_effect=AssertionError("must reject before scoring")):
                    with self.assertRaisesRegex(ValueError, "recorded replay cache changed"):
                        replay.replay(args, check)
                self.assertEqual(path.read_bytes(), modified_bytes)
                for name, content in preserved.items():
                    self.assertEqual((cache / name).read_bytes(), content)

    def test_resume_keeps_all_old_hashes_when_dependency_fails(self):
        import builtins
        original_import = builtins.__import__
        def without_torch(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise ImportError("PyTorch intentionally unavailable")
            return original_import(name, *args, **kwargs)
        for prior_state, count in (("verified", 12), ("running", 3), ("failed", 3)):
            with self.subTest(prior_state=prior_state), tempfile.TemporaryDirectory() as directory:
                cache, _results, status, check, args = self._resume_fixture(Path(directory))
                status.update(status=prior_state, cases=status["cases"][:count])
                replay.atomic_json(cache / "replay_status.json", status)
                with patch("builtins.__import__", side_effect=without_torch):
                    with self.assertRaisesRegex(ImportError, "intentionally unavailable"):
                        replay.replay(args, check)
                failed = json.loads((cache / "replay_status.json").read_text(encoding="utf-8"))
                self.assertEqual(failed["status"], "failed")
                self.assertEqual(failed["cases"], status["cases"])
                # A later retry must still reject changes to an unvisited case.
                (cache / "raw_cache/heldout/background" / f"{status['cases'][-1]['sample_id']}.npz").write_bytes(b"modified")
                with self.assertRaisesRegex(ValueError, "recorded replay cache changed"):
                    replay.replay(args, check)
                self.assertEqual(json.loads((cache / "replay_status.json").read_text(encoding="utf-8")), failed)

    def test_resume_rejects_missing_recorded_cache_without_regenerating(self):
        with tempfile.TemporaryDirectory() as directory:
            cache, _results, _status, check, args = self._resume_fixture(Path(directory))
            path = cache / "raw_cache/heldout/background" / f"{replay.CASES[0][0]}.npz"
            path.unlink()
            with self.assertRaisesRegex(ValueError, "changed or missing"):
                replay.replay(args, check)
            self.assertFalse(path.exists())

    def test_output_cannot_overlap_formal_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {name: root / name for name in ("results", "data", "official", "dinov3", "checkpoint", "memory", "mad", "support")}
            paths["output"] = paths["results"] / "replay"
            with self.assertRaisesRegex(ValueError, "protected results"):
                replay.guard_output(paths)


if __name__ == "__main__":
    unittest.main()
