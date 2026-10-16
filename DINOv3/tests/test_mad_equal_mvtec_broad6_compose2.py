from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from DINOv3.MADEqual.aebad_s_broad6_compose2.augmentations import (
    BROAD6,
    broad6_compose2,
)
from DINOv3.MADEqual.mvtec_broad6_compose2 import memory, metrics, runner
from DINOv3.relation_reliability.datasets import mvtec_ad_records, split_source_normal
from DINOv3.relation_reliability.protocol import LAYERS, MVTEC_AD_CATEGORIES


def _png(path: Path, value: int = 128) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((12, 16, 3), value, dtype=np.uint8)).save(path)


def test_mvtec_parser_and_index_modulo_split(tmp_path: Path) -> None:
    category = tmp_path / "bottle"
    for index in range(17):
        _png(category / "train" / "good" / f"{index:03d}.png", index)
    _png(category / "test" / "good" / "000.png")
    _png(category / "test" / "broken" / "001.png")
    _png(category / "ground_truth" / "broken" / "001_mask.png", 255)
    train, evaluation = mvtec_ad_records(tmp_path, categories=("bottle",))
    memory_records, calibration_records = split_source_normal(train["bottle"], 8)
    assert [Path(item.path).stem for item in calibration_records] == ["000", "008", "016"]
    assert [Path(item.path).stem for item in memory_records] == [
        f"{index:03d}" for index in range(17) if index % 8 != 0
    ]
    assert {record.label for record in evaluation["bottle"]} == {0, 1}
    assert not ({record.path for record in memory_records} & {record.path for record in calibration_records})


def test_broad6_reuses_existing_compose2_and_is_deterministic() -> None:
    assert len(BROAD6) == 6
    image = Image.fromarray(np.full((24, 32, 3), 120, dtype=np.uint8))
    first = broad6_compose2(image, "bottle/train/good/000.png", seed=42)
    second = broad6_compose2(image, "bottle/train/good/000.png", seed=42)
    assert first.op1 != first.op2
    assert (first.op1, first.op2, first.params1, first.params2) == (
        second.op1, second.op2, second.params1, second.params2
    )
    source = Path(memory.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "DINOv3.MADEqual.aebad_s_broad6_compose2.augmentations"
        for alias in node.names
    }
    assert "broad6_compose2" in imports


def test_one_compose2_view_per_memory_fit_image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    records = []
    from DINOv3.relation_reliability.datasets import Record

    for index in range(9):
        path = tmp_path / "bottle" / "train" / "good" / f"{index:03d}.png"
        _png(path, index)
        records.append(Record(
            path=str(path), label=0, mask_path=None, category="bottle",
            dataset="mvtec_ad", domain="source", shift="train_good",
            role="train", mask_capability="not_evaluation",
        ))
    calls: list[str] = []
    real = broad6_compose2

    def wrapped(image: Image.Image, relative_path: str, *, seed: int):
        calls.append(relative_path)
        return real(image, relative_path, seed=seed)

    monkeypatch.setattr(memory, "broad6_compose2", wrapped)
    monkeypatch.setattr(memory, "image_tensor", lambda image: torch.zeros((3, 2, 2)))
    monkeypatch.setattr(memory, "_official_candidate_resource_estimate", lambda model, tensors: (7, 1))
    monkeypatch.setattr(
        memory,
        "_validate_bank_shapes",
        lambda banks, **kwargs: {layer: int(value.shape[0]) for layer, value in banks.items()},
    )

    def fake_builder(model, tensors, **kwargs):
        assert sum(int(tensor.numel() == 0) for tensor in tensors) == 2
        return SimpleNamespace(
            banks={layer: torch.empty((1, 1)) for layer in LAYERS},
            metadata={"fake": True},
        )

    monkeypatch.setattr(memory, "build_official_memory_spooled", fake_builder)
    model = SimpleNamespace(
        threshold_fraction=8,
        max_database_size=100_000,
        augmented_preprocessing=SimpleNamespace(
            min_brightness_factor=0.8, max_brightness_factor=1.2,
        ),
    )
    _banks, manifest, _metadata = memory.build_broad6_memory(
        model,
        records,
        dataset_root=tmp_path,
        device=torch.device("cpu"),
        scratch_dir=tmp_path / "scratch",
    )
    assert len(calls) == len(manifest) == 7
    assert len(set(calls)) == 7


def test_broad6_preprocessing_consumes_but_does_not_apply_brightness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from DINOv3.MADEqual.aebad_s_broad6_compose2 import memory as existing_memory

    draws: list[tuple[float, float]] = []
    monkeypatch.setattr(
        existing_memory.np.random,
        "uniform",
        lambda low, high: draws.append((low, high)) or 1.19,
    )
    monkeypatch.setattr(
        existing_memory,
        "deterministic_official_preprocessing",
        lambda model, image: image + 3.0,
    )
    delegate = SimpleNamespace(min_brightness_factor=0.8, max_brightness_factor=1.2)
    preprocessing = existing_memory._Broad6Preprocessing(SimpleNamespace(), delegate)
    image = torch.ones((1, 3, 2, 2))
    output = preprocessing(image)
    assert draws == [(0.8, 1.2)]
    assert torch.equal(output, image + 3.0)


def test_bank_shape_contract_treats_100k_as_cap_without_allocation() -> None:
    banks = {
        layer: torch.empty((100_000, 1280), device="meta") for layer in LAYERS
    }
    assert memory._validate_bank_shapes(banks) == {layer: 100_000 for layer in LAYERS}
    toothbrush_banks = {
        layer: torch.empty((83_200, 1280), device="meta") for layer in LAYERS
    }
    assert memory._validate_bank_shapes(toothbrush_banks) == {
        layer: 83_200 for layer in LAYERS
    }
    memory._validate_bank_shapes(toothbrush_banks, expected_entries=83_200)


def test_two_method_mad_fits_are_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[int, np.ndarray]] = []

    def fake(samples):
        calls.append(samples)
        offset = float(len(calls))
        return None, SimpleNamespace(
            medians=np.full(4, offset, dtype=np.float32),
            scales=np.ones(4, dtype=np.float32),
        )

    monkeypatch.setattr(runner, "fit_calibrators", fake)
    official = {layer: np.full(256, layer, dtype=np.float32) for layer in LAYERS}
    broad6 = {layer: np.full(256, layer + 1, dtype=np.float32) for layer in LAYERS}
    first = runner._fit_method_mad(official)
    second = runner._fit_method_mad(broad6)
    assert len(calls) == 2
    assert not np.array_equal(first.medians, second.medians)


def _reference_aupro(masks, maps, thresholds, max_fpr=0.3):
    from scipy import ndimage

    background_count = sum(int((~mask).sum()) for mask in masks)
    components = []
    for mask, score in zip(masks, maps):
        labels, count = ndimage.label(mask)
        for component_id in range(1, count + 1):
            components.append(score[labels == component_id])
    fprs, pros = [], []
    for threshold in thresholds:
        background_positive = sum(
            int((score[~mask] > threshold).sum()) for mask, score in zip(masks, maps)
        )
        fprs.append(background_positive / background_count)
        pros.append(np.mean([(region > threshold).mean() for region in components]))
    points = [(0.0, 0.0), *zip(fprs, pros), (1.0, 1.0)]
    x = np.asarray([item[0] for item in points], dtype=np.float64)
    y = np.asarray([item[1] for item in points], dtype=np.float64)
    order = np.argsort(x)
    x, y = x[order], y[order]
    unique = np.unique(x)
    upper = np.asarray([y[x == value].max() for value in unique])
    if max_fpr not in unique:
        value = np.interp(max_fpr, unique, upper)
        keep = unique < max_fpr
        unique, upper = np.append(unique[keep], max_fpr), np.append(upper[keep], value)
    else:
        keep = unique <= max_fpr
        unique, upper = unique[keep], upper[keep]
    return float(np.trapezoid(upper, unique) / max_fpr)


def test_fast_aupro_matches_small_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    masks = [
        np.asarray([[0, 0], [0, 0]], dtype=bool),
        np.asarray([[1, 0], [0, 0]], dtype=bool),
        np.asarray([[0, 0], [0, 1]], dtype=bool),
    ]
    maps = [
        np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        np.asarray([[0.9, 0.1], [0.2, 0.3]], dtype=np.float32),
        np.asarray([[0.1, 0.2], [0.4, 0.8]], dtype=np.float32),
    ]
    labels = [0, 1, 1]
    monkeypatch.setattr(
        metrics,
        "pixel_ranking_metrics_and_thresholds",
        lambda *args, **kwargs: (0.5, 0.5, np.asarray([123.0]), "test_ranking"),
    )
    actual = metrics.evaluate_fast(
        labels, masks, maps, device=torch.device("cpu"), allow_cpu_fallback=True,
    )["AUPRO_at_0p3"]
    flat_scores = np.concatenate([score.reshape(-1) for score in maps])
    thresholds = np.linspace(
        float(flat_scores.max()), float(flat_scores.min()), 200, dtype=np.float64,
    )
    expected = _reference_aupro(masks, maps, thresholds)
    assert actual == pytest.approx(expected, abs=1e-12)


def test_aupro_thresholds_are_fixed_global_range_not_quantiles() -> None:
    scores = np.concatenate([
        np.zeros(997, dtype=np.float32),
        np.asarray([0.25, 0.5, 1.0], dtype=np.float32),
    ])
    thresholds = metrics.global_score_range_thresholds(scores)
    assert thresholds.shape == (200,)
    assert np.array_equal(thresholds, np.linspace(1.0, 0.0, 200, dtype=np.float64))
    assert len(np.unique(thresholds)) == 200


def test_runner_has_no_adeval_or_official_aupro_call() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "official_aupro_adeval" not in source
    assert "EvalAccumulatorCuda" not in source


def test_runner_has_global_prediction_barrier_before_phase_b() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    body = source[source.index("def run("):]
    phase_a = body.index("_phase_a_category(")
    root_barrier = body.index('_atomic_json(output / "prediction_barrier.json"')
    phase_b = body.index("_phase_b_category(")
    assert phase_a < root_barrier < phase_b
    assert '"ground_truth_masks_opened": False' in body[phase_a:phase_b]


def test_metric_metadata_records_four_connected_components() -> None:
    assert metrics.COMPONENT_CONNECTIVITY == "4-connected_scipy_ndimage_default"


def test_macro_is_unweighted_over_15_categories() -> None:
    rows = []
    for method_offset, method in enumerate(runner.METHODS):
        for index, category in enumerate(MVTEC_AD_CATEGORIES):
            rows.append({
                "method": method,
                "category": category,
                **{metric: float(index + method_offset) for metric in metrics.METRIC_NAMES},
            })
    output = metrics.category_macro(rows)
    assert output[0]["image_AUROC"] == pytest.approx(np.mean(np.arange(15)))
    assert output[1]["image_AUROC"] == pytest.approx(np.mean(np.arange(15) + 1))


def test_baseline_assets_reads_completed_category(tmp_path: Path) -> None:
    root = tmp_path / "categories" / "mvtec_ad" / "bottle"
    root.mkdir(parents=True)
    (root / "complete.json").write_text(
        json.dumps({"status": "complete", "category": "bottle"}), encoding="utf-8",
    )
    (root / "calibration").mkdir()
    np.savez(
        root / "calibration" / "calibration_state.npz",
        mad_medians=np.arange(4, dtype=np.float32),
        mad_scales=np.ones(4, dtype=np.float32),
        **{f"ecdf_{layer}": np.ones(256, dtype=np.float32) for layer in LAYERS},
    )
    assets = runner.load_baseline_assets(tmp_path, "bottle")
    assert assets.medians.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert assets.patch_counts == (256, 256, 256, 256)
