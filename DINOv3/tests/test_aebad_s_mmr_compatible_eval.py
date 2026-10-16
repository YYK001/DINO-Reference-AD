from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from DINOv3.MADEqual.aebad_s_broad6_compose2 import mmr_compatible_eval as evaluation


def test_center_crop_coordinates_are_16_to_240() -> None:
    value = np.arange(256 * 256, dtype=np.int64).reshape(256, 256)
    cropped = evaluation.center_crop_224(value)
    assert cropped.shape == (224, 224)
    assert cropped[0, 0] == value[16, 16]
    assert cropped[-1, -1] == value[239, 239]


def test_empty_anomaly_mask_is_relabelled_normal() -> None:
    assert evaluation.mmr_compatible_label(1, np.zeros((224, 224), bool)) == 0


def test_nonempty_anomaly_mask_remains_anomaly() -> None:
    mask = np.zeros((224, 224), bool)
    mask[20, 30] = True
    assert evaluation.mmr_compatible_label(1, mask) == 1


def test_original_normal_always_remains_normal() -> None:
    mask = np.ones((224, 224), bool)
    assert evaluation.mmr_compatible_label(0, mask) == 0


def test_image_score_is_cropped_map_max() -> None:
    full = np.zeros((256, 256), np.float32)
    full[0, 0] = 100.0
    full[100, 100] = 7.0
    cropped, score = evaluation.cropped_image_score(full)
    assert cropped.shape == (224, 224)
    assert score == pytest.approx(float(cropped.max()))
    assert score == pytest.approx(7.0)


def test_domain_macros_are_unweighted() -> None:
    rows = []
    for method in evaluation.METHODS:
        for index, domain in enumerate(evaluation.DOMAINS, 1):
            rows.append({
                "method": method,
                "domain": domain,
                "full_protocol_image_AUROC": float(index),
                "mmr_compatible_image_AUROC": float(index + 1),
            })
    summary = {
        (row["method"], row["scope"]): row
        for row in evaluation.summarize_domain_metrics(rows)
    }
    for method in evaluation.METHODS:
        assert summary[(method, "four_domain_macro")]["full_protocol_image_AUROC"] == 2.5
        assert summary[(method, "shifted_macro")]["full_protocol_image_AUROC"] == 3.0
        assert summary[(method, "four_domain_macro")]["delta"] == 1.0


def test_module_never_calls_model_or_exact_nn_entrypoints() -> None:
    source_path = Path(evaluation.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    forbidden = {
        "construct_exact_host", "build_exact_official_memory", "observe_pil",
        "build_official_memory_spooled", "record_tensors", "patch_exec",
    }
    calls = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
    assert calls.isdisjoint(forbidden)
