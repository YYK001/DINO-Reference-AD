from __future__ import annotations

from pathlib import Path
import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from DINOv3.MADEqual.aebad_s_full_p0 import dataset, runner
from DINOv3.relation_reliability.calibration import MADCalibrator
from DINOv3.relation_reliability.datasets import split_source_normal
from DINOv3.relation_reliability.protocol import LAYERS


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


@pytest.mark.parametrize("layout", ("direct", "short", "nested"))
def test_root_discovery_and_phase_b_paths(tmp_path: Path, layout: str) -> None:
    if layout == "direct":
        root = supplied = tmp_path / "AeBAD_S"
    elif layout == "short":
        supplied, root = tmp_path, tmp_path / "AeBAD_S"
    else:
        supplied, root = tmp_path, tmp_path / "AeBAD" / "AeBAD_S"
    _touch(root / "train" / "good" / "source.png")
    _touch(root / "train" / "good" / "nested" / "source_nested.png")
    _touch(root / "train" / "good" / "nested" / "._source_nested.png")
    _touch(root / "train" / "good" / "__MACOSX" / "metadata.png")
    _touch(root / "test" / "good" / "same" / "normal.png")
    anomaly = _touch(root / "test" / "scratch" / "view" / "bad.jpg")
    expected_mask = _touch(root / "ground_truth" / "scratch" / "view" / "bad.png")

    found, train, locators = dataset.discover_phase_a(supplied, formal=False)
    assert found == root.resolve()
    assert len(train) == 2 and all(record.label == 0 for record in train)
    assert [Path(record.path).name for record in train] == ["source_nested.png", "source.png"]
    assert [(item.domain, Path(item.path).name) for item in locators] == [
        ("same", "normal.png"), ("view", "bad.jpg"),
    ]

    records = dataset.load_phase_b_records(found, locators, formal=False)
    normal, defect = records
    assert normal.label == 0 and normal.mask_path is None
    assert defect.path == str(anomaly)
    assert defect.label == 1 and defect.domain == "view"
    assert defect.mask_path == str(expected_mask)


def test_official_count_validation() -> None:
    dataset.validate_official_counts(521, dict(dataset.OFFICIAL_TEST_COUNTS))
    with pytest.raises(RuntimeError, match="train count mismatch"):
        dataset.validate_official_counts(520, dict(dataset.OFFICIAL_TEST_COUNTS))
    wrong = dict(dataset.OFFICIAL_TEST_COUNTS)
    wrong["same"] -= 1
    with pytest.raises(RuntimeError, match="domain counts mismatch"):
        dataset.validate_official_counts(521, wrong)


def test_frozen_source_split_is_455_and_66_without_overlap(tmp_path: Path) -> None:
    records = [
        dataset.Record(
            path=str(tmp_path / f"{index:04d}.png"), label=0, mask_path=None,
            category="AeBAD-S", dataset="aebad_s", domain="source",
            shift="train_good", role="train", mask_capability="not_evaluation",
        )
        for index in range(521)
    ]
    memory, calibration = split_source_normal(records, threshold_fraction=8)
    assert (len(memory), len(calibration)) == (455, 66)
    assert {item.path for item in memory}.isdisjoint(item.path for item in calibration)
    assert [records.index(item) for item in calibration] == list(range(0, 521, 8))


def test_runner_calls_existing_fit_and_mad_equal(monkeypatch: pytest.MonkeyPatch) -> None:
    pooled = {
        layer: np.arange(10, dtype=np.float32) + offset
        for offset, layer in enumerate(LAYERS)
    }
    sentinel_mad = object()
    fit_calls = []

    def fake_fit(values):
        fit_calls.append(values)
        return object(), sentinel_mad

    monkeypatch.setattr(runner, "fit_calibrators", fake_fit)
    mad = runner._fit_source_calibrators(pooled)
    assert fit_calls == [pooled]
    assert mad is sentinel_mad

    raw = np.arange(4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3)
    existing_mad = MADCalibrator(
        np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32),
    )
    mad_calls = []

    def fake_mad_equal(values, medians, scales):
        mad_calls.append((values, medians, scales))
        return np.full(values.shape[1:], -7.0, dtype=np.float32)

    monkeypatch.setattr(runner, "mad_equal", fake_mad_equal)
    scores = runner._score_raw(raw, mad=existing_mad)
    assert mad_calls[0][0] is raw
    assert np.all(scores["MAD_EQUAL"] == -7.0)
    assert np.array_equal(scores["SUPERADD_OFFICIAL"], raw.mean(axis=0))


def test_runner_installs_official_import_paths_before_construct() -> None:
    source = inspect.getsource(runner.run)
    assert source.index("install_import_paths") < source.index("construct_exact_host")


def test_raw_cache_rejects_nonfinite_or_wrong_geometry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image_path = tmp_path / "image.png"
    Image.fromarray(np.zeros((64, 80, 3), dtype=np.uint8)).save(image_path)
    cache = tmp_path / "raw.npz"
    common = {
        "protocol_id": np.asarray(runner.PROTOCOL_ID),
        "cache_revision": np.asarray(runner.CACHE_REVISION),
        "dataset_root": np.asarray(str(tmp_path)),
        "relative_path": np.asarray("image.png"),
        "output_shape": np.asarray((16, 20), dtype=np.int32),
    }
    monkeypatch.setattr(runner, "preprocessed_spatial_shape", lambda model, h, w: (64, 80))
    model = type("Model", (), {
        "evaluation_downscale": 4,
        "patch_exec": type("Patch", (), {"model_patch_size": 16})(),
    })()
    _atomic = runner._atomic_npz
    _atomic(cache, raw_distances=np.zeros((4, 4, 5), dtype=np.float32), **common)
    raw, shape = runner._observe_or_load(
        cache, relative_path="image.png", dataset_root=str(tmp_path),
        image_path=str(image_path), model=model, device=torch.device("cpu"), allow_observe=False,
    )
    assert raw.shape == (4, 4, 5) and shape == (16, 20)
    _atomic(cache, raw_distances=np.full((4, 4, 5), np.nan, dtype=np.float32), **common)
    with pytest.raises(RuntimeError, match="invalid raw-distance cache"):
        runner._observe_or_load(
            cache, relative_path="image.png", dataset_root=str(tmp_path),
            image_path=str(image_path), model=model, device=torch.device("cpu"), allow_observe=False,
        )


def test_layerwise_memory_preserves_brightness_and_sampler_rng(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Preprocessing:
        min_brightness_factor = 0.8
        max_brightness_factor = 1.2

        def __init__(self) -> None:
            self.draws = []

        def __call__(self, value):
            self.draws.append(float(np.random.uniform(0.8, 1.2)))
            return value

    preprocessing = Preprocessing()
    model = SimpleNamespace(
        layers=list(LAYERS),
        backbone=SimpleNamespace(layers=list(LAYERS)),
        augmented_preprocessing=preprocessing,
        threshold_fraction=2,
        prototype_embeddings=None,
    )
    records = [object() for _ in range(5)]
    sampler_draws = {}

    def fake_builder(model, train_records, *, device, scratch_dir):
        layer = model.layers[0]
        for index in range(len(train_records)):
            if index % model.threshold_fraction != 0:
                model.augmented_preprocessing(torch.zeros(1))
        sampler_draws[layer] = float(np.random.rand())
        return SimpleNamespace(banks={layer: torch.full((2, 3), float(layer))})

    monkeypatch.setattr(runner, "build_exact_official_memory", fake_builder)
    np.random.seed(42)
    expected = np.random.RandomState(42)
    expected_brightness = [float(expected.uniform(0.8, 1.2)) for _ in range(2)]
    expected_sampler = {layer: float(expected.rand()) for layer in LAYERS}

    np.random.seed(42)
    runner._build_exact_memory_layerwise(
        model, records, device=torch.device("cpu"), output=tmp_path,
    )

    for offset in range(0, len(preprocessing.draws), 2):
        assert preprocessing.draws[offset:offset + 2] == pytest.approx(expected_brightness)
    assert sampler_draws == pytest.approx(expected_sampler)
    assert model.layers == list(LAYERS)
    assert model.backbone.layers == list(LAYERS)
    assert tuple(model.prototype_embeddings) == tuple(LAYERS)


def test_domain_and_shifted_macros_are_unweighted() -> None:
    rows = []
    values = {"same": 1.0, "background": 2.0, "illumination": 4.0, "view": 8.0}
    for method in runner.METHODS:
        for domain, value in values.items():
            rows.append({
                "method": method,
                "group": domain,
                **{metric: value for metric in runner.METRIC_NAMES},
            })
    summary = runner.summarize_group_metrics(rows)
    first = [row for row in summary if row["method"] == runner.METHODS[0]]
    four_domain = next(row for row in first if row["scope"] == "four_domain_macro")
    shifted = next(row for row in first if row["scope"] == "shifted_macro")
    assert four_domain["image_AUROC"] == pytest.approx(3.75)
    assert shifted["image_AUROC"] == pytest.approx(14.0 / 3.0)
    assert shifted["delta_vs_official_image_AUROC"] == pytest.approx(0.0)
