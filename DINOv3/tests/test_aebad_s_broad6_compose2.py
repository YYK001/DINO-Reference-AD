from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from DINOv3.MADEqual.aebad_s_broad6_compose2 import augmentations as aug
from DINOv3.MADEqual.aebad_s_broad6_compose2 import memory, runner
from DINOv3.relation_reliability.calibration import MADCalibrator
from DINOv3.relation_reliability.datasets import Record, split_source_normal
from DINOv3.relation_reliability.protocol import LAYERS


def _record(path: Path) -> Record:
    return Record(
        path=str(path), label=0, mask_path=None, category="AeBAD-S",
        dataset="aebad_s", domain="source", shift="train_good",
        role="train", mask_capability="not_evaluation",
    )


def test_broad6_pool_and_compose2_are_frozen_and_distinct() -> None:
    assert aug.BROAD6 == (
        "AutoContrast", "Gamma", "Contrast", "WhiteBalance", "JPEG", "IlluminationField"
    )
    for index in range(200):
        op1, op2 = aug.choose_broad_compose2(f"train/good/{index:03d}.png")
        assert op1 in aug.BROAD6 and op2 in aug.BROAD6 and op1 != op2


def test_assignment_and_atomic_parameters_are_path_deterministic() -> None:
    path = "train\\good\\sample.png"
    assert aug.choose_broad_compose2(path) == aug.choose_broad_compose2(path.replace("\\", "/"))
    for op in aug.BROAD6:
        assert aug.atomic_params(op, path) == aug.atomic_params(op, path.replace("\\", "/"))


def test_parameter_ranges_and_white_balance_normalization() -> None:
    for index in range(50):
        path = f"train/good/{index}.png"
        assert 0.8 <= aug.atomic_params("Gamma", path)["gamma"] < 1.25
        assert 0.8 <= aug.atomic_params("Contrast", path)["factor"] < 1.2
        assert 70 <= aug.atomic_params("JPEG", path)["quality"] <= 95
        field = aug.atomic_params("IlluminationField", path)
        assert 0.1 <= field["amplitude"] < 0.2
        assert 0.0 <= field["angle"] < 2.0 * np.pi
        wb = aug.atomic_params("WhiteBalance", path)
        assert np.prod([wb["r"], wb["g"], wb["b"]]) == pytest.approx(1.0)


def test_all_transforms_preserve_rgb_shape() -> None:
    rgb = np.arange(18 * 24 * 3, dtype=np.uint8).reshape(18, 24, 3)
    for op in aug.BROAD6:
        result = aug.apply_atomic(rgb, op, aug.atomic_params(op, "train/good/x.png"))
        assert result.shape == rgb.shape
        assert result.dtype == np.uint8


def test_existing_split_is_455_66_without_overlap(tmp_path: Path) -> None:
    records = [_record(tmp_path / f"{index:04d}.png") for index in range(521)]
    memory_records, calibration_records = split_source_normal(records, threshold_fraction=8)
    assert (len(memory_records), len(calibration_records)) == (455, 66)
    assert not ({record.path for record in memory_records} & {record.path for record in calibration_records})


def test_memory_builder_replaces_official_brightness_and_checks_bank_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "train.png"
    Image.new("RGB", (4, 4), (20, 40, 60)).save(image_path)
    records = [_record(image_path) for _ in range(521)]
    original_preprocessing = SimpleNamespace(
        min_brightness_factor=0.8, max_brightness_factor=1.2,
    )
    model = SimpleNamespace(threshold_fraction=8, augmented_preprocessing=original_preprocessing)
    captured = {}

    monkeypatch.setattr(memory, "_official_candidate_resource_estimate", lambda model, values: (7, 9))

    def fake_build(model, tensors, **kwargs):
        captured["preprocessing"] = model.augmented_preprocessing
        captured["nonempty"] = sum(tensor.numel() > 0 for tensor in tensors)
        banks = {layer: torch.empty((100_000, 1280), device="meta") for layer in LAYERS}
        return SimpleNamespace(banks=banks, metadata={"fake": True})

    monkeypatch.setattr(memory, "build_official_memory_spooled", fake_build)
    banks, manifest, _ = memory.build_broad6_memory(
        model, records, dataset_root=tmp_path, device=torch.device("cpu"), scratch_dir=tmp_path / "scratch"
    )
    assert captured["nonempty"] == 455
    assert isinstance(captured["preprocessing"], memory._Broad6Preprocessing)
    assert model.augmented_preprocessing is original_preprocessing
    assert len(manifest) == 455
    assert all(tuple(banks[layer].shape) == (100_000, 1280) for layer in LAYERS)


def test_broad6_preprocessing_does_not_call_brightness_and_preserves_sampler_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrightnessMustNotRun:
        min_brightness_factor = 0.8
        max_brightness_factor = 1.2

        def __call__(self, value):
            raise AssertionError("Official brightness preprocessing was called")

    monkeypatch.setattr(
        memory, "deterministic_official_preprocessing", lambda model, value: value
    )
    preprocessing = memory._Broad6Preprocessing(object(), BrightnessMustNotRun())
    value = torch.zeros((1, 3, 2, 2))
    np.random.seed(42)
    for _ in range(455):
        assert preprocessing(value) is value
    actual_next = np.random.random()
    np.random.seed(42)
    for _ in range(455):
        np.random.uniform(0.8, 1.2)
    assert actual_next == np.random.random()


def test_runner_calls_existing_fit_calibrators_and_mad_equal(monkeypatch: pytest.MonkeyPatch) -> None:
    fit_called = {}
    mad_called = {}
    calibrator = MADCalibrator(np.zeros(4, np.float32), np.ones(4, np.float32))

    def fake_fit(samples):
        fit_called["samples"] = samples
        return object(), calibrator

    def fake_mad(raw, medians, scales):
        mad_called["raw"] = raw
        return np.asarray(raw).mean(axis=0)

    monkeypatch.setattr(runner, "fit_calibrators", fake_fit)
    monkeypatch.setattr(runner, "mad_equal", fake_mad)
    samples = {layer: np.arange(5, dtype=np.float32) for layer in LAYERS}
    assert runner._fit_mad(samples) is calibrator
    raw = np.ones((4, 2, 3), np.float32)
    assert runner._score_raw(raw, calibrator).shape == (2, 3)
    assert fit_called["samples"] is samples
    assert np.array_equal(mad_called["raw"], raw)


def test_domain_and_shifted_macros() -> None:
    rows = []
    for index, domain in enumerate(("same", "background", "illumination", "view"), 1):
        rows.append({"method": runner.METHOD, "group": domain, **{metric: float(index) for metric in runner.METRIC_NAMES}})
    summary = {row["scope"]: row for row in runner.summarize_new_method(rows)}
    assert summary["four_domain_macro"]["image_AUROC"] == pytest.approx(2.5)
    assert summary["shifted_macro"]["image_AUROC"] == pytest.approx(3.0)


def test_baseline_path_is_explicit_and_validated(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        runner.load_baseline(tmp_path)
    complete = {
        "protocol_id": runner.BASELINE_PROTOCOL_ID,
        "status": "complete",
        "train_count": 521,
        "memory_count": 455,
        "calibration_count": 66,
        "prediction_count": 1639,
        "methods": ["SUPERADD_OFFICIAL", "MAD_EQUAL"],
    }
    (tmp_path / "complete.json").write_text(json.dumps(complete), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        runner.load_baseline(tmp_path)
