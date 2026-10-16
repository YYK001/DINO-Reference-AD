from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from DINOv3.MADEqual.aebad_s_target_fewshot import runner, support
from DINOv3.MADEqual.robustad_target_fewshot_p0.scoring import exact_prefix_minima
from DINOv3.relation_reliability.datasets import Record, split_source_normal


def _record(path: Path, *, domain: str, label: int, role: str = "evaluation") -> Record:
    return Record(
        path=str(path),
        label=label,
        mask_path=str(path.with_suffix(".png")) if label else None,
        category="AeBAD-S",
        dataset="aebad_s",
        domain=domain,
        shift="defect" if label else "good",
        role=role,
        mask_capability="pixel" if role == "evaluation" else "not_evaluation",
    )


def _evaluation_records(root: Path) -> list[Record]:
    rows: list[Record] = []
    counts = {
        "same": (230, 459),
        "background": (93, 212),
        "illumination": (75, 198),
        "view": (92, 280),
    }
    for domain, (normal_count, anomaly_count) in counts.items():
        rows.extend(
            _record(root / "test" / "good" / domain / f"n_{index:04d}.png", domain=domain, label=0)
            for index in range(normal_count)
        )
        rows.extend(
            _record(root / "test" / "bad" / domain / f"a_{index:04d}.png", domain=domain, label=1)
            for index in range(anomaly_count)
        )
    return rows


def test_source_split_is_455_66_without_overlap(tmp_path: Path) -> None:
    records = [
        _record(tmp_path / "train" / "good" / f"{index:04d}.png", domain="source", label=0, role="train")
        for index in range(521)
    ]
    memory, calibration = split_source_normal(records, threshold_fraction=8)
    assert (len(memory), len(calibration)) == (455, 66)
    assert {record.path for record in memory}.isdisjoint(record.path for record in calibration)


def test_four_domains_select_eight_normal_support_and_common_heldout(tmp_path: Path) -> None:
    records = _evaluation_records(tmp_path)
    pools, heldout = support.select_target_support(records, dataset_root=tmp_path)
    assert set(pools) == set(support.DOMAINS)
    assert all(len(pool) == 8 for pool in pools.values())
    assert all(record.label == 0 for pool in pools.values() for record in pool)
    assert {
        domain: (
            sum(record.label == 0 for record in rows),
            sum(record.label == 1 for record in rows),
            len(rows),
        )
        for domain, rows in heldout.items()
    } == support.EXPECTED_HELDOUT
    assert sum(len(rows) for rows in heldout.values()) == 1607
    assert sum(record.label == 1 for rows in heldout.values() for record in rows) == 1149


def test_support_prefixes_are_nested() -> None:
    pool = list(range(8))
    prefixes = support.support_prefixes(pool)
    assert prefixes == {1: [0], 2: [0, 1], 4: [0, 1, 2, 3], 8: pool}


def test_every_method_uses_the_same_heldout_membership(tmp_path: Path) -> None:
    _pools, heldout = support.select_target_support(
        _evaluation_records(tmp_path), dataset_root=tmp_path,
    )
    identities = {
        method: tuple(
            record.path for domain in support.DOMAINS for record in heldout[domain]
        )
        for method in runner.METHODS
    }
    assert len(set(identities.values())) == 1


def test_source_extra_comes_only_from_source_calibration_normals(tmp_path: Path) -> None:
    calibration = [
        _record(tmp_path / "train" / "good" / f"{index:04d}.png", domain="source", label=0, role="train")
        for index in range(66)
    ]
    selected = support.select_source_extra(calibration, dataset_root=tmp_path)
    assert len(selected) == 8
    assert all(record in calibration and record.label == 0 for record in selected)


def test_exact_union_matches_explicit_concatenation_and_prefix_is_monotone() -> None:
    generator = torch.Generator().manual_seed(42)
    query = torch.randn((5, 1280), generator=generator)
    official = torch.randn((7, 1280), generator=generator)
    supports = [torch.randn((3, 1280), generator=generator) for _ in range(8)]
    official_distance, union, _support_only = exact_prefix_minima(query, official, supports)
    for shot in (1, 2, 4, 8):
        explicit = torch.cdist(query, torch.cat([official, *supports[:shot]])).min(dim=1).values.numpy() / 1280.0
        assert np.allclose(union[shot], explicit, atol=1e-6)
        assert np.all(union[shot] <= official_distance + 1e-7)
    assert np.all(union[2] <= union[1] + 1e-7)
    assert np.all(union[4] <= union[2] + 1e-7)
    assert np.all(union[8] <= union[4] + 1e-7)


def test_exact_distance_is_euclidean_divided_by_1280_not_squared() -> None:
    query = torch.zeros((1, 1280))
    official = torch.zeros((1, 1280))
    official[0, 0] = 3.0
    official[0, 1] = 4.0
    far = torch.full((1, 1280), 100.0)
    distance, _union, _support = exact_prefix_minima(query, official, [far] * 8)
    assert distance[0] == pytest.approx(5.0 / 1280.0)
    assert distance[0] != pytest.approx(25.0 / 1280.0)


def test_runner_has_one_source_mad_fit_and_no_target_refit() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    fit_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "fit_calibrators"
    ]
    assert len(fit_calls) == 1
    assert "target_mad" not in source.lower()
    assert "domain_mad" not in source.lower()
    assert "k_mad" not in source.lower()


def test_query_forward_has_one_observe_call_site_and_methods_do_not_reobserve() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "observe_fewshot_pil"
    ]
    assert len(calls) == 1
    method_map = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_method_map"
    )
    assert not any(
        isinstance(node, ast.Name) and node.id in {"observe_fewshot_pil", "observe_pil"}
        for node in ast.walk(method_map)
    )


def test_mmr_summary_accepts_all_ten_methods() -> None:
    rows = [
        {
            "method": method,
            "domain": domain,
            "full_protocol_image_AUROC": 0.5 + index * 0.01,
            "mmr_compatible_image_AUROC": 0.6 + index * 0.01,
        }
        for method in runner.METHODS
        for index, domain in enumerate(support.DOMAINS)
    ]
    summary = runner.mmr_eval.summarize_domain_metrics(rows, methods=runner.METHODS)
    assert len(summary) == 20
    assert {row["method"] for row in summary} == set(runner.METHODS)


def test_support_path_does_not_import_synthetic_augmentation_modules() -> None:
    texts = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for path in (support.__file__, runner.__file__)
    ).lower()
    assert "broad6_compose2.augmentations" not in texts
    assert "phys4_compose2.augmentations" not in texts
    assert "gnl" not in texts
