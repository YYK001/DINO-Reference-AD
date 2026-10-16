from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from DINOv3.MADEqual.robustad_target_fewshot_p0 import runner
from DINOv3.MADEqual.robustad_target_fewshot_p0.scoring import (
    exact_prefix_minima,
    signed_reduction_by_image,
)
from DINOv3.MADEqual.robustad_target_fewshot_p0.support import (
    SHOT_COUNTS,
    TARGET_SHIFTS,
    TARGET_SHIFTS_BY_CATEGORY,
    select_source_extra,
    select_target_support,
    support_prefixes,
)
from DINOv3.MADEqual.robustad_target_fewshot_p0.summarize_three_categories import (
    CATEGORIES,
    aggregate_category_macros,
)
from DINOv3.relation_reliability.calibration import MADCalibrator
from DINOv3.relation_reliability.datasets import Record, split_source_normal


def _record(root: Path, name: str, *, shift: str, label: int) -> Record:
    path = root / "MetalParts" / shift / name
    return Record(
        str(path), label, None, "MetalParts", "robustad",
        "target" if shift != "source" else "source", shift,
        "evaluation", "pixel",
    )


def _target_records(root: Path) -> list[Record]:
    output: list[Record] = []
    for shift in TARGET_SHIFTS:
        output.extend(_record(root, f"normal_{index:02d}.png", shift=shift, label=0) for index in range(12))
        output.extend(_record(root, f"bad_{index:02d}.png", shift=shift, label=1) for index in range(5))
    return output


def test_source_count_and_frozen_157_23_split(tmp_path: Path) -> None:
    records = [_record(tmp_path, f"source_{index:03d}.png", shift="source", label=0) for index in range(180)]
    memory, calibration = split_source_normal(records, 8)
    assert (len(records), len(memory), len(calibration)) == (180, 157, 23)
    assert set(memory).isdisjoint(calibration)


@pytest.mark.parametrize("category", ("MetalParts", "PCB", "PiledBags"))
def test_each_category_keeps_frozen_180_157_23_split(tmp_path: Path, category: str) -> None:
    records = [
        Record(
            str(tmp_path / category / "source" / f"normal_{index:03d}.png"),
            0, None, category, "robustad", "source", "source_train", "train", "not_evaluation",
        )
        for index in range(180)
    ]
    memory, calibration = split_source_normal(records, 8)
    assert (len(memory), len(calibration)) == (157, 23)
    assert set(memory).isdisjoint(calibration)


def test_six_target_shifts_are_frozen() -> None:
    assert TARGET_SHIFTS == (
        "background_1", "background_2", "lighting",
        "position", "rotation", "scale",
    )


def test_three_category_target_shift_sets_are_frozen() -> None:
    assert TARGET_SHIFTS_BY_CATEGORY["PCB"] == (
        "lighting", "white_balancing", "rotation", "position", "shadow",
    )
    assert TARGET_SHIFTS_BY_CATEGORY["PiledBags"] == (
        "background_box_color", "lighting", "position_rotation", "scale", "shadow",
    )
    assert runner.CATEGORY_SPECS["PCB"].pixel_evaluation is True
    assert runner.CATEGORY_SPECS["PiledBags"].pixel_evaluation is False


def test_support_pool_is_normal_and_fixed_eight_are_removed(tmp_path: Path) -> None:
    records = _target_records(tmp_path)
    pools, heldout = select_target_support(records, dataset_root=tmp_path)
    for shift in TARGET_SHIFTS:
        assert len(pools[shift]) == 8
        assert {item.label for item in pools[shift]} == {0}
        selected = {item.path for item in pools[shift]}
        assert selected.isdisjoint(item.path for item in heldout[shift])


def test_support_prefixes_are_nested() -> None:
    pool = list(range(8))
    prefixes = support_prefixes(pool)  # type: ignore[arg-type]
    assert tuple(prefixes) == SHOT_COUNTS
    assert prefixes[1] == pool[:1]
    assert prefixes[2] == pool[:2]
    assert prefixes[4] == pool[:4]
    assert prefixes[8] == pool


def test_all_methods_share_same_heldout_ids_for_every_k(tmp_path: Path) -> None:
    _, heldout = select_target_support(_target_records(tmp_path), dataset_root=tmp_path)
    for shift in TARGET_SHIFTS:
        expected = tuple(item.path for item in heldout[shift])
        evaluated = {method: tuple(item.path for item in heldout[shift]) for method in runner.METHODS}
        assert all(ids == expected for ids in evaluated.values())


def test_anomalies_never_enter_support_and_all_remain_heldout(tmp_path: Path) -> None:
    records = _target_records(tmp_path)
    pools, heldout = select_target_support(records, dataset_root=tmp_path)
    for shift in TARGET_SHIFTS:
        original_bad = {item.path for item in records if item.shift == shift and item.label == 1}
        assert not any(item.label for item in pools[shift])
        assert {item.path for item in heldout[shift] if item.label == 1} == original_bad


def _toy_exact() -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray], torch.Tensor, list[torch.Tensor]]:
    query = torch.zeros((2, 1280), dtype=torch.float32)
    query[0, 0] = 3.0
    query[1, 0] = 9.0
    official = torch.zeros((1, 1280), dtype=torch.float32)
    supports = []
    for value in (8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0):
        bank = torch.zeros((1, 1280), dtype=torch.float32)
        bank[0, 0] = value
        supports.append(bank)
    off, union, support = exact_prefix_minima(query, official, supports)
    return off, union, support, query, supports


def test_cumulative_prefix_matches_explicit_union_exact_top1() -> None:
    off, union, _support, query, supports = _toy_exact()
    official = torch.zeros((1, 1280), dtype=torch.float32)
    for shot in SHOT_COUNTS:
        explicit = torch.cdist(query, torch.cat([official, *supports[:shot]])).min(dim=1).values.numpy() / 1280.0
        np.testing.assert_allclose(union[shot], explicit, rtol=0, atol=1e-7)
    assert off.shape == (2,)


def test_fewshot_distance_never_exceeds_official() -> None:
    off, union, _support, _query, _supports = _toy_exact()
    for shot in SHOT_COUNTS:
        assert np.all(union[shot] <= off)


def test_support_prefix_distance_is_monotonic_in_k() -> None:
    _off, _union, support, _query, _supports = _toy_exact()
    for earlier, later in zip(SHOT_COUNTS, SHOT_COUNTS[1:]):
        assert np.all(support[later] <= support[earlier])


def test_exact_distance_is_euclidean_divided_by_1280_not_squared() -> None:
    off, _union, _support, _query, _supports = _toy_exact()
    assert off[0] == pytest.approx(3.0 / 1280.0)
    assert off[0] != pytest.approx(9.0 / 1280.0)


def test_all_calibrated_methods_call_same_existing_mad(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[np.ndarray, np.ndarray]] = []

    def fake_mad(raw: np.ndarray, medians: np.ndarray, scales: np.ndarray) -> np.ndarray:
        seen.append((medians, scales))
        return raw.mean(axis=0)

    monkeypatch.setattr(runner, "mad_equal", fake_mad)
    raw = np.ones((4, 2, 2), np.float32)
    values = {"official_raw": raw}
    for shot in SHOT_COUNTS:
        values[f"source_raw_{shot}"] = raw
        values[f"target_raw_{shot}"] = raw
    mad = MADCalibrator(np.arange(4, dtype=np.float32), np.arange(1, 5, dtype=np.float32))
    for method in runner.METHODS[1:]:
        runner._method_map(values, method, mad, (2, 2))
    assert len(seen) == len(runner.METHODS) - 1
    assert all(medians is mad.medians and scales is mad.scales for medians, scales in seen)


def test_runner_has_one_source_fit_and_no_target_mad_refit() -> None:
    tree = ast.parse(Path(runner.__file__).read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "fit_calibrators"]
    assert len(calls) == 1
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_fit_source_mad")
    assert calls[0] in list(ast.walk(function))


def test_signed_mechanism_uses_normal_all_patches_and_anomaly_defect_only() -> None:
    baseline = np.full((2, 4, 2, 2), 10.0, np.float32)
    adapted = baseline.copy()
    adapted[0] -= 2.0
    adapted[1] -= 1.0
    adapted[1, :, 0, 0] = 6.0
    mask = np.array([[True, False], [False, False]])
    rn, ra, gap = signed_reduction_by_image(
        baseline, adapted, np.ones(4, np.float32),
        labels=[0, 1], defect_masks=[None, mask],
    )
    np.testing.assert_allclose(rn, 2.0)
    np.testing.assert_allclose(ra, 4.0)
    np.testing.assert_allclose(gap, -2.0)


def test_source_extra_uses_only_source_calibration_normals(tmp_path: Path) -> None:
    calibration = [_record(tmp_path, f"source_{index:02d}.png", shift="source", label=0) for index in range(23)]
    pool = select_source_extra(calibration, dataset_root=tmp_path)
    assert len(pool) == 8
    assert all(item.label == 0 and item.shift == "source" for item in pool)


def test_target_macro_requires_all_six_aligned_shifts() -> None:
    source = [{"method": method, **{metric: 0.5 for metric in runner.METRICS}} for method in ("OFFICIAL", "MAD_EQUAL")]
    rows = []
    for method in runner.METHODS:
        for shift in TARGET_SHIFTS:
            rows.append({"method": method, "shift": shift, **{metric: 0.6 for metric in runner.METRICS}})
    result = runner.summarize_target_macro(rows, source)
    assert len(result) == len(runner.METHODS)
    assert all(row["image_AUROC"] == pytest.approx(0.6) for row in result)
    assert all(row["ARD_image_AUROC"] == pytest.approx(0.0) for row in result)


def test_category_bound_support_is_stable_and_excludes_same_eight(tmp_path: Path) -> None:
    shifts = TARGET_SHIFTS_BY_CATEGORY["PCB"]
    records: list[Record] = []
    for shift in shifts:
        records.extend(
            Record(
                str(tmp_path / "PCB" / shift / f"normal_{index:02d}.png"),
                0, None, "PCB", "robustad", "target", shift, "evaluation", "pixel",
            )
            for index in range(12)
        )
        records.extend(
            Record(
                str(tmp_path / "PCB" / shift / f"bad_{index:02d}.png"),
                1, str(tmp_path / f"mask_{shift}_{index}.png"), "PCB", "robustad",
                "target", shift, "evaluation", "pixel",
            )
            for index in range(3)
        )
    first, heldout = select_target_support(
        records, dataset_root=tmp_path, category="PCB", shifts=shifts,
    )
    second, _ = select_target_support(
        list(reversed(records)), dataset_root=tmp_path, category="PCB", shifts=shifts,
    )
    for shift in shifts:
        assert [row.path for row in first[shift]] == [row.path for row in second[shift]]
        assert len(first[shift]) == 8
        assert all(row.label == 0 for row in first[shift])
        assert sum(row.label == 1 for row in heldout[shift]) == 3


def test_piledbags_metric_path_never_opens_mask_or_calls_aupro(monkeypatch: pytest.MonkeyPatch) -> None:
    records = [
        Record("normal.png", 0, None, "PiledBags", "robustad", "target", "lighting", "evaluation", "image"),
        Record("bad.png", 1, None, "PiledBags", "robustad", "target", "lighting", "evaluation", "image"),
    ]
    monkeypatch.setattr(runner, "_evaluation_mask", lambda *_args, **_kwargs: pytest.fail("mask opened"))
    monkeypatch.setattr(runner, "official_aupro_adeval", lambda *_args, **_kwargs: pytest.fail("AUPRO called"))
    row = runner._metric_row(
        records,
        [np.zeros((2, 2), np.float32), np.ones((2, 2), np.float32)],
        "MAD_EQUAL", "lighting", pixel_evaluation=False,
    )
    assert row["image_AUROC"] == pytest.approx(1.0)
    assert row["pixel_AUROC"] == row["pixel_AUPR"] == row["AUPRO"] == ""


def test_hit_rate_is_image_equal_not_patch_pooled() -> None:
    rows = [
        {"shift": "lighting", "method": "TARGET_MATCHED_K1_MAD", "K": 1, "layer": 7,
         "label": 0, "R_value": 1.0, "normal_hit_rate": 1.0, "defect_hit_rate": ""},
        {"shift": "lighting", "method": "TARGET_MATCHED_K1_MAD", "K": 1, "layer": 7,
         "label": 0, "R_value": 0.0, "normal_hit_rate": 0.0, "defect_hit_rate": ""},
    ]
    summary = runner._mechanism_by_shift_layer(rows, pixel_evaluation=False)
    assert summary[0]["normal_hit_rate"] == pytest.approx(0.5)
    assert summary[0]["hit_aggregation"] == "image_equal"


def test_three_category_macro_uses_three_image_and_two_pixel_categories() -> None:
    grouped = {}
    for category_index, category in enumerate(CATEGORIES, start=1):
        rows = []
        for method in runner.METHODS:
            row = {"method": method}
            for metric in runner.METRICS:
                if category == "PiledBags" and not metric.startswith("image_"):
                    row[metric] = row[f"ARD_{metric}"] = ""
                else:
                    row[metric] = float(category_index)
                    row[f"ARD_{metric}"] = -float(category_index)
            rows.append(row)
        grouped[category] = rows
    result = aggregate_category_macros(grouped)
    for row in result:
        assert row["image_AUROC"] == pytest.approx(2.0)
        assert row["image_AUROC_category_count"] == 3
        assert row["pixel_AUROC"] == pytest.approx(1.5)
        assert row["pixel_AUROC_category_count"] == 2
        assert row["pixel_AUROC_categories"] == "MetalParts;PCB"


def test_query_forward_remains_single_extraction_call() -> None:
    scoring_path = Path(runner.__file__).with_name("scoring.py")
    tree = ast.parse(scoring_path.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "observe_fewshot_pil"
    )
    calls = [
        node for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "_extract_processed_batch"
    ]
    assert len(calls) == 1
