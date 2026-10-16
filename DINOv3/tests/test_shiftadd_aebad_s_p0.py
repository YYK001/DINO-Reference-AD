from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from DINOv3.MADEqual.aebad_s_full_p0 import runner as baseline_runner
from DINOv3.MADEqual.aebad_s_full_p0.dataset import OFFICIAL_TEST_COUNTS, validate_official_counts
from DINOv3.ShiftADD.aebad_s_p0 import memory, protocol, runner, sampler, scoring, transforms
from DINOv3.relation_reliability.calibration import MADCalibrator
from DINOv3.relation_reliability.datasets import Record, split_source_normal
from DINOv3.relation_reliability.superadd_host import _MinimumSpatialPreprocessing


def _plans() -> list[protocol.CandidatePlan]:
    return protocol.make_candidate_plans(f"train/good/{index:04d}.png" for index in range(455))


def _plan(branch: str) -> protocol.CandidatePlan:
    return next(item for item in _plans() if item.planned_branch == branch)


def test_v2_identity_stable_assignment_and_path_order_independence() -> None:
    assert protocol.PROTOCOL_ID == "shiftadd_aebad_s_seed42_p0_v2_anchor_priority"
    assert "v2_anchor_priority" in protocol.BRIGHTNESS2_CONTROL_PROTOCOL_ID
    assert sum(protocol.FACTOR_QUOTAS.values()) == 50_000
    assert protocol.ANCHOR_QUOTA + sum(protocol.FACTOR_QUOTAS.values()) == 100_000
    assert protocol.MAIN_METHODS == (
        "SUPERADD_OFFICIAL", "MAD_EQUAL", "ANCHOR50_EQUAL", "ANCHOR50_MAD",
        "SHIFTADD_EQUAL", "SHIFTADD_MAD",
    )
    paths = [f"train/good/{index:04d}.png" for index in range(455)]
    first = {item.relative_path: item for item in protocol.make_candidate_plans(paths)}
    second = {item.relative_path: item for item in protocol.make_candidate_plans(reversed(paths))}
    assert first == second
    assert {
        branch: sum(item.planned_branch == branch for item in first.values())
        for branch in protocol.SHIFT_BRANCHES
    } == {"ILLUMINATION": 152, "POSE": 152, "SCENE": 151}


def test_source_split_and_official_counts() -> None:
    records = [
        Record(
            path=f"/source/{index:04d}.png", label=0, mask_path=None,
            category="AeBAD-S", dataset="aebad_s", domain="source",
            shift="train_good", role="train", mask_capability="not_evaluation",
        )
        for index in range(521)
    ]
    memory_fit, calibration = split_source_normal(records, threshold_fraction=8)
    assert (len(memory_fit), len(calibration)) == (455, 66)
    assert {item.path for item in memory_fit}.isdisjoint(item.path for item in calibration)
    validate_official_counts(521, dict(OFFICIAL_TEST_COUNTS))


def test_priority_reservoir_is_bounded_order_independent_and_maps_provenance() -> None:
    features = np.arange(6 * 3, dtype=np.float32).reshape(6, 3)
    priorities = np.asarray([9, 2, 8, 1, 7, 3], dtype=np.uint64)
    expected = sampler.priority_topk_indices(
        priorities, np.asarray([0, 0, 0, 1, 1, 1]), np.asarray([0, 1, 2, 0, 1, 2]), 3,
    )
    reservoir = sampler.PriorityReservoir(3, feature_dim=3)
    reservoir.update(
        features[:3], priorities[:3], source_index=0, patch_indices=np.arange(3), subtype_code=4,
    )
    reservoir.update(
        features[3:], priorities[3:], source_index=1, patch_indices=np.arange(3), subtype_code=7,
    )
    result = reservoir.finalize()
    np.testing.assert_array_equal(result.features, features[expected])
    np.testing.assert_array_equal(result.source_indices, np.asarray([1, 0, 1], dtype=np.int32))
    assert result.features.shape == (3, 3)
    assert not hasattr(memory, "_MatrixSpool")
    np.testing.assert_array_equal(sampler.anchor_indices(7), sampler.anchor_indices(7))
    reordered = sampler.PriorityReservoir(3, feature_dim=3)
    reordered.update(
        features[3:], priorities[3:], source_index=1, patch_indices=np.arange(3), subtype_code=7,
    )
    reordered.update(
        features[:3], priorities[:3], source_index=0, patch_indices=np.arange(3), subtype_code=4,
    )
    reordered_result = reordered.finalize()
    np.testing.assert_array_equal(reordered_result.features, result.features)
    np.testing.assert_array_equal(reordered_result.source_indices, result.source_indices)


def test_official_rng_isolation_and_brightness2_variant_isolation() -> None:
    image = Image.fromarray(np.full((96, 112, 3), 128, dtype=np.uint8), mode="RGB")
    np.random.seed(913)
    expected = np.random.random(8)
    np.random.seed(913)
    protocol.make_candidate_plans(reversed([f"train/good/{index:04d}.png" for index in range(455)]))
    transforms.apply_shift_transform(image, _plan("ILLUMINATION"))
    np.testing.assert_array_equal(np.random.random(8), expected)

    paths = ["train/good/b.png", "train/good/a.png"]
    first = protocol.brightness2_factors(paths)
    second = protocol.brightness2_factors(reversed(paths))
    assert first == second
    assert all(0.8 <= value <= 1.2 for value in first.values())
    assert protocol.protocol_id_for_variant("shiftadd") != protocol.protocol_id_for_variant("brightness2_control")
    full_paths = [f"train/good/{index:04d}.png" for index in range(455)]
    generator = np.random.RandomState(42)
    generator.uniform(0.8, 1.2, size=455)
    expected_second_round = generator.uniform(0.8, 1.2, size=455)
    actual_second_round = protocol.brightness2_factors(full_paths)
    np.testing.assert_array_equal(
        [actual_second_round[path] for path in sorted(full_paths)], expected_second_round,
    )


def test_explicit_brightness_matches_existing_official_operation(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = SimpleNamespace(
        min_brightness_factor=0.8, max_brightness_factor=1.2, resize_factor=1.0,
        mean=torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1),
        std=torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1),
    )
    host = SimpleNamespace(
        preprocessing=processor, augmented_preprocessing=processor,
        patch_exec=SimpleNamespace(patch_size=640, model_patch_size=16),
    )
    wrapped = _MinimumSpatialPreprocessing(processor, host)
    value = torch.linspace(0, 1, 3 * 640 * 640).reshape(1, 3, 640, 640)
    factor = 1.073
    monkeypatch.setattr(np.random, "uniform", lambda _low, _high: factor)
    expected = wrapped(value.clone())
    explicit = memory.explicit_official_brightness_tensor(value.clone(), factor)
    actual = memory.deterministic_official_preprocessing(host, explicit)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_transform_ranges_shapes_pose_bound_and_scene_foreground_protection() -> None:
    source = np.full((128, 128, 3), 120, dtype=np.uint8)
    source[42:86, 42:86] = np.asarray([245, 8, 8], dtype=np.uint8)
    image = Image.fromarray(source, mode="RGB")
    for branch in protocol.SHIFT_BRANCHES:
        result = transforms.apply_shift_transform(image, _plan(branch))
        assert result.image.size == image.size
        assert np.asarray(result.image).dtype == np.uint8
        if branch == "POSE":
            assert result.pose_attempt in range(-1, 8)
    for attempt in range(8):
        params = transforms.pose_attempt_parameters(_plan("POSE").relative_path, attempt)
        assert -12 <= params["rotation_degrees"] <= 12
        assert -0.08 <= params["translate_x_fraction"] <= 0.08
        assert -0.08 <= params["translate_y_fraction"] <= 0.08
        assert 0.9 <= params["scale"] <= 1.1
        assert abs(params["perspective_x_fraction"]) <= 0.03
        assert abs(params["perspective_y_fraction"]) <= 0.03
    pose_plan = _plan("POSE")
    attempt_zero = transforms.pose_attempt_parameters(pose_plan.relative_path, 0)
    legacy_rng = protocol.stable_rng(pose_plan.relative_path, "shift_transform")
    legacy = {
        "rotation_degrees": float(legacy_rng.uniform(-12.0, 12.0)),
        "translate_x_fraction": float(legacy_rng.uniform(-0.08, 0.08)),
        "translate_y_fraction": float(legacy_rng.uniform(-0.08, 0.08)),
        "scale": float(legacy_rng.uniform(0.9, 1.1)),
        "perspective_x_fraction": float(legacy_rng.uniform(-0.03, 0.03)),
        "perspective_y_fraction": float(legacy_rng.uniform(-0.03, 0.03)),
    }
    assert attempt_zero == legacy
    segmentation = transforms.source_color_segmentation(image)
    if segmentation.trusted_foreground_bbox is not None:
        pose = transforms.apply_shift_transform(image, pose_plan)
        if pose.candidate_active:
            assert transforms.pose_contains_bbox(
                image.width, image.height, segmentation.trusted_foreground_bbox, pose.params,
            )
    impossible = dict(attempt_zero)
    impossible["translate_x_fraction"] = 10.0
    assert not transforms.pose_contains_bbox(128, 128, (0, 0, 32, 32), impossible)
    scene = transforms.apply_shift_transform(image, _plan("SCENE"))
    if scene.scene_valid:
        assert scene.scene_mask is not None and not scene.scene_mask[64, 64]
        np.testing.assert_array_equal(np.asarray(scene.image)[~scene.scene_mask], source[~scene.scene_mask])
    else:
        assert scene.actual_branch == "SCENE_NOOP"
        np.testing.assert_array_equal(np.asarray(scene.image), source)


def test_scene_overlap_is_bounded_and_bins_close() -> None:
    safe = np.zeros((64, 80), dtype=bool)
    safe[:20, :30] = True
    overlap = memory.scene_overlap_at_grid(safe, (4, 5))
    assert overlap.shape == (20,)
    assert np.all((overlap >= 0) & (overlap <= 1))
    values = np.resize(overlap, protocol.FACTOR_QUOTAS["SCENE_VALID"])
    provenance = {}
    for layer in protocol.LAYERS:
        branch = np.full(50_000, protocol.BRANCH_CODES["ILLUMINATION"], dtype=np.int16)
        branch[: values.size] = protocol.BRANCH_CODES["SCENE_VALID"]
        scene_values = np.full(50_000, np.nan, dtype=np.float32)
        scene_values[: values.size] = values
        provenance[layer] = memory.FactorProvenance(
            source_indices=np.zeros(50_000, dtype=np.int32), branch_codes=branch,
            subtype_codes=np.zeros(50_000, dtype=np.int16), patch_indices=np.arange(50_000),
            pose_attempts=np.full(50_000, -1, dtype=np.int16), scene_overlap=scene_values,
        )
    rows = memory._scene_overlap_rows(provenance)
    for layer in protocol.LAYERS:
        assert sum(row["selected_count"] for row in rows if row["layer"] == layer) == 16_666


def test_partitioned_exact_scan_tie_goes_to_factor() -> None:
    query = torch.tensor([[0.0, 0.0], [2.0, 2.0]])
    anchor = torch.tensor([[1.0, 1.0], [4.0, 4.0]])
    factor = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    anchor_distance, combined_distance, indices, factor_wins = scoring.partitioned_exact_top1(query, anchor, factor)
    assert factor_wins.tolist() == [True, True]
    assert indices.tolist() == [2, 3]
    assert torch.all(combined_distance <= anchor_distance)
    random_query = torch.tensor([[0.2, 0.7], [2.1, 1.7]], dtype=torch.float32)
    random_anchor = torch.tensor([[0.0, 0.0], [3.0, 3.0]], dtype=torch.float32)
    random_factor = torch.tensor([[1.0, 0.0], [1.0, 2.0]], dtype=torch.float32)
    _anchor_d, split_d, split_i, _wins = scoring.partitioned_exact_top1(
        random_query, random_anchor, random_factor,
    )
    joined = torch.cat((random_anchor, random_factor), dim=0)
    expected_d, expected_i = torch.cdist(random_query, joined).min(dim=1)
    torch.testing.assert_close(split_d, expected_d, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(split_i, expected_i, rtol=0, atol=0)
    source = Path(scoring.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    observe = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "observe_partitioned_pil")
    calls = [
        node.func.id if isinstance(node.func, ast.Name) else node.func.attr
        for node in ast.walk(observe) if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))
    ]
    assert calls.count("_extract_processed_batch") == 1


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_baseline_preflight_reads_completed_result_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "baseline"
    root.mkdir()
    (root / "complete.json").write_text(json.dumps({
        "protocol_id": baseline_runner.PROTOCOL_ID, "status": "complete", "run_type": "formal",
        "train_count": 521, "memory_count": 455, "calibration_count": 66,
        "prediction_count": 1639, "methods": ["SUPERADD_OFFICIAL", "MAD_EQUAL"],
    }), encoding="utf-8")
    for name in ("official_memory.pt", "calibration_stats.csv"):
        (root / name).touch()
    group_rows = [
        {"method": method, "group": domain, "image_count": 1, **{metric: 0.5 for metric in protocol.METRIC_NAMES}}
        for method in ("SUPERADD_OFFICIAL", "MAD_EQUAL") for domain in ("same", "background", "illumination", "view")
    ]
    _write_csv(root / "group_metrics.csv", group_rows)
    _write_csv(root / "summary.csv", [{"method": "SUPERADD_OFFICIAL", "scope": "shifted_macro", **{metric: 0.5 for metric in protocol.METRIC_NAMES}}])
    _write_csv(root / "sample_scores.csv", [{"method": "SUPERADD_OFFICIAL", "label": 0, "image_score": 0.1}])
    for section, count in (("calibration", 66), ("test", 1639)):
        folder = root / "raw_cache" / section
        folder.mkdir(parents=True)
        for index in range(count):
            (folder / f"{index:04d}.npz").touch()
    sentinel = {layer: torch.empty(0) for layer in protocol.LAYERS}
    monkeypatch.setattr(runner, "load_official_banks", lambda _path: sentinel)
    monkeypatch.setattr(
        runner, "_read_csv",
        lambda _path: (_ for _ in ()).throw(AssertionError("GT-derived CSV parsed before barrier")),
    )
    state = runner._load_baseline_state(root)
    assert state.official_banks is sentinel


def test_control_requires_mechanism_pass(tmp_path: Path) -> None:
    root = tmp_path / "main"
    root.mkdir()
    (root / "complete.json").write_text(json.dumps({
        "protocol_id": protocol.PROTOCOL_ID, "status": "complete", "mechanism_pass": False,
    }), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Mechanism PASS"):
        runner._active_paths(root)


def test_mad_reuse_raw_cache_and_macro(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_mad = MADCalibrator(np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32))
    calls: dict[str, object] = {}
    monkeypatch.setattr(runner, "fit_calibrators", lambda values: ("ecdf", fake_mad))
    samples = {layer: [np.arange(8, dtype=np.float32)] for layer in protocol.LAYERS}
    assert runner._fit_mad(samples)[0] is fake_mad

    raw = np.arange(16, dtype=np.float32).reshape(4, 2, 2)
    def spy(value: np.ndarray, medians: np.ndarray, scales: np.ndarray) -> np.ndarray:
        calls["raw"] = value
        return np.full(value.shape[1:], -2.0, dtype=np.float32)
    monkeypatch.setattr(runner, "mad_equal", spy)
    scores = runner._score_raw(raw, fake_mad, "SHIFTADD")
    assert calls["raw"] is raw
    assert np.all(scores["SHIFTADD_MAD"] == -2.0)

    cache = tmp_path / "cache.npz"
    runner._atomic_npz(
        cache, anchor_raw_distances=raw, factor_raw_distances=raw - 1,
        combined_top1_indices=np.zeros_like(raw, dtype=np.int64),
        top1_branch_codes=np.zeros_like(raw, dtype=np.int16),
        output_shape=np.asarray([32, 32], dtype=np.int32),
    )
    loaded = runner._read_cache(cache)
    np.testing.assert_array_equal(loaded.factor_raw, raw - 1)
    assert loaded.output_shape == (32, 32)

    rows = [
        {"method": method, "group": domain, **{metric: value for metric in protocol.METRIC_NAMES}}
        for method, value in (("SUPERADD_OFFICIAL", 0.5), ("MAD_EQUAL", 0.7))
        for domain in ("same", "background", "illumination", "view")
    ]
    summary = runner.summarize_group_metrics(rows, ("SUPERADD_OFFICIAL", "MAD_EQUAL"))
    assert runner._row(summary, "MAD_EQUAL", "shifted_macro")["image_AUROC"] == pytest.approx(0.7)


def test_mechanism_uses_shift_domains_epsilon_and_unclipped_reduction() -> None:
    totals: dict[tuple[str, int | str, str], list[float | int]] = {}
    hits: dict[tuple[str, str, int, str], list[int]] = {}
    anchor = np.ones((4, 1, 2), dtype=np.float32)
    shift = anchor + np.float32(5e-7)
    observed = runner.CachedObservation(
        anchor, shift, np.zeros_like(anchor, dtype=np.int64),
        np.zeros_like(anchor, dtype=np.int16), (1, 2),
    )
    mad = MADCalibrator(np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32))
    shifted_record = Record(
        path="shift.png", label=1, mask_path="mask.png", category="AeBAD-S",
        dataset="aebad_s", domain="background", shift="background", role="test",
        mask_capability="available",
    )
    same_record = Record(
        path="same.png", label=1, mask_path="mask.png", category="AeBAD-S",
        dataset="aebad_s", domain="same", shift="same", role="test",
        mask_capability="available",
    )
    mask = np.ones((1, 2), dtype=bool)
    runner._accumulate_mechanism(totals, hits, shifted_record, observed, mask, mad)
    primary_before = list(totals[("defect", "all", "patch_pooled")])
    runner._accumulate_mechanism(totals, hits, same_record, observed, mask, mad)
    assert totals[("defect", "all", "patch_pooled")] == primary_before
    expected = ((anchor - shift) / (1.0 + runner.ECDF_EPSILON)).sum()
    assert totals[("defect", "all", "patch_pooled")][0] == pytest.approx(float(expected))
    assert any(float(value[0]) < 0 for key, value in totals.items() if key[0] == "defect")


def test_gt_loading_occurs_only_after_prediction_barrier_and_no_v1_sampler() -> None:
    path = Path(runner.__file__)
    source = path.read_text(encoding="utf-8")
    assert source.index('output / "prediction_barrier.json"') < source.index("records = load_phase_b_records(")
    assert source.index('output / "prediction_barrier.json"') < source.index("baseline_groups, baseline_samples = _load_baseline_rows(")
    load_baseline = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "_load_baseline_state")
    assert not any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_read_csv"
        for node in ast.walk(load_baseline)
    )
    tree = ast.parse(source)
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
    assert not any(isinstance(node, ast.While) for node in ast.walk(run))
    memory_source = Path(memory.__file__).read_text(encoding="utf-8")
    assert "_MatrixSpool" not in memory_source
    assert "_official_subsample_values_with_indices" not in memory_source
    assert "if not result.candidate_active:\n                continue" in memory_source
    memory_tree = ast.parse(memory_source)
    build = next(node for node in memory_tree.body if isinstance(node, ast.FunctionDef) and node.name == "build_partitioned_memory")
    assert sum(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "path_priority_seed"
        for node in ast.walk(build)
    ) == 1
    assert "patch_priorities_from_seed" in memory_source
