from __future__ import annotations

import ast
import itertools
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from DINOv3.MADEqual.aebad_s_broad6_compose2 import mmr_compatible_eval
from DINOv3.MADEqual.aebad_s_phys4_compose2 import augmentations as aug
from DINOv3.MADEqual.aebad_s_phys4_compose2 import memory, runner
from DINOv3.relation_reliability.datasets import Record
from DINOv3.relation_reliability.protocol import LAYERS


def _record(path: Path) -> Record:
    return Record(
        path=str(path), label=0, mask_path=None, category="AeBAD-S",
        dataset="aebad_s", domain="source", shift="train_good",
        role="train", mask_capability="not_evaluation",
    )


def test_phys4_pool_is_exact() -> None:
    assert aug.PHYS4 == (
        "Brightness", "Gamma", "WhiteBalance", "IlluminationField",
    )


def test_compose2_is_deterministic_without_replacement_and_recipes_are_legal() -> None:
    observed = set()
    for index in range(1000):
        path = f"train/good/{index:04d}.png"
        first = aug.choose_phys4_compose2(path)
        assert first == aug.choose_phys4_compose2(path)
        assert first[0] != first[1]
        observed.add(first)
    expected = set(itertools.permutations(aug.PHYS4, 2))
    assert observed == expected


def test_assignment_seed_uses_frozen_crc32_namespace() -> None:
    relative = "train\\good\\background\\example.png"
    normalized = "train/good/background/example.png"
    expected = zlib.crc32(
        f"42|ASSIGN::{aug.POLICY_NAME}|{normalized}".encode("utf-8")
    ) & 0xFFFFFFFF
    assert aug.stable_seed(f"ASSIGN::{aug.POLICY_NAME}", relative) == expected


def test_parameter_ranges_and_white_balance_geometric_mean() -> None:
    for index in range(100):
        path = f"train/good/{index}.png"
        assert 0.8 <= aug.atomic_params("Brightness", path)["factor"] < 1.2
        assert 0.8 <= aug.atomic_params("Gamma", path)["gamma"] < 1.25
        field = aug.atomic_params("IlluminationField", path)
        assert 0.1 <= field["amplitude"] < 0.2
        assert 0.0 <= field["angle"] < 2.0 * np.pi
        wb = aug.atomic_params("WhiteBalance", path)
        assert np.prod([wb["r"], wb["g"], wb["b"]]) == pytest.approx(1.0)


def test_float32_brightness_is_multiply_and_clamp_without_uint8_roundtrip() -> None:
    value = np.asarray([[[0.12345, 0.50123, 0.99991]]], dtype=np.float32)
    factor = 1.137
    actual = aug.apply_atomic_float(value, "Brightness", {"factor": factor})
    expected = np.clip(value * np.float32(factor), 0.0, 1.0).astype(np.float32)
    assert actual.dtype == np.float32
    assert np.array_equal(actual, expected)
    quantized = (actual * 255.0).astype(np.uint8).astype(np.float32) / 255.0
    assert not np.array_equal(actual, quantized)


def test_compose2_returns_one_float_chw_view() -> None:
    image = Image.new("RGB", (7, 5), (31, 127, 251))
    result = aug.phys4_compose2(image, "train/good/x.png")
    assert tuple(result.tensor.shape) == (3, 5, 7)
    assert result.tensor.dtype == torch.float32
    assert float(result.tensor.min()) >= 0.0
    assert float(result.tensor.max()) <= 1.0
    assert result.op1 != result.op2


def test_memory_builder_generates_455_views_and_skips_66_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "train.png"
    Image.new("RGB", (4, 4), (20, 40, 60)).save(image_path)
    records = [_record(image_path) for _ in range(521)]
    original = SimpleNamespace(min_brightness_factor=0.8, max_brightness_factor=1.2)
    model = SimpleNamespace(threshold_fraction=8, augmented_preprocessing=original)
    captured = {}
    monkeypatch.setattr(memory, "_official_candidate_resource_estimate", lambda m, t: (7, 9))

    def fake_builder(model, tensors, **kwargs):
        captured["nonempty"] = sum(tensor.numel() > 0 for tensor in tensors)
        captured["empty_indices"] = [
            index for index, tensor in enumerate(tensors) if tensor.numel() == 0
        ]
        captured["preprocessing"] = model.augmented_preprocessing
        return SimpleNamespace(
            banks={layer: torch.empty((100_000, 1280), device="meta") for layer in LAYERS},
            metadata={"fake": True},
        )

    monkeypatch.setattr(memory, "build_official_memory_spooled", fake_builder)
    banks, manifest, _ = memory.build_phys4_memory(
        model, records, dataset_root=tmp_path, device=torch.device("cpu"),
        scratch_dir=tmp_path / "scratch",
    )
    assert captured["nonempty"] == 455
    assert captured["empty_indices"] == list(range(0, 521, 8))
    assert len(manifest) == 455
    assert model.augmented_preprocessing is original
    assert all(tuple(banks[layer].shape) == (100_000, 1280) for layer in LAYERS)


def test_runner_reuses_existing_scoring_nn_and_aupro() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
    assert "observe_pil" in calls
    assert "interpolate_score" in calls
    assert "_score_raw" in calls
    assert "_evaluate" in calls
    assert "fit_calibrators" not in calls  # delegated to existing Broad6 _fit_mad
    assert "mad_equal" not in calls  # delegated to existing Broad6 _score_raw


def test_mmr_compatible_evaluator_accepts_phys4_and_summarizes_four_methods() -> None:
    args = mmr_compatible_eval.build_parser().parse_args([
        "--aebad-root", "dataset", "--baseline-output", "baseline",
        "--broad6-output", "broad6", "--phys4-output", "phys4",
        "--output-dir", "output", "--report-path", "report.md",
    ])
    assert args.phys4_output == "phys4"
    methods = (*mmr_compatible_eval.METHODS, mmr_compatible_eval.PHYS4_METHOD)
    rows = [
        {
            "method": method, "domain": domain,
            "full_protocol_image_AUROC": float(index),
            "mmr_compatible_image_AUROC": float(index + 1),
        }
        for method in methods
        for index, domain in enumerate(mmr_compatible_eval.DOMAINS, 1)
    ]
    summary = mmr_compatible_eval.summarize_domain_metrics(rows, methods=methods)
    assert len(summary) == 8
    assert {row["method"] for row in summary} == set(methods)
