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

from DINOv3.MADEqual.aebad_s_synthetic_fewshot import runner, scoring, support
from DINOv3.MADEqual.aebad_s_target_fewshot import runner as base_runner
from DINOv3.relation_reliability.datasets import Record
from DINOv3.relation_reliability.protocol import LAYERS


def _record(path: Path, *, domain: str, label: int, role: str = "evaluation") -> Record:
    return Record(
        path=str(path), label=label,
        mask_path=str(path.with_suffix(".mask.png")) if label else None,
        category="AeBAD-S", dataset="aebad_s", domain=domain,
        shift="bad" if label else "good", role=role,
        mask_capability="pixel" if role == "evaluation" else "not_evaluation",
    )


def _write_manifest(path: Path) -> list[str]:
    source_paths = [f"train/good/source_{index}.png" for index in range(1, 9)]
    rows = []
    for index, relative in enumerate(source_paths, 1):
        rows.append({
            "branch": "source_extra", "domain": "source",
            "support_index": index, "relative_path": relative,
        })
    for domain in support.DOMAINS:
        for index in range(1, 9):
            rows.append({
                "branch": "target_matched", "domain": domain,
                "support_index": index,
                "relative_path": f"test/good/{domain}/support_{index}.png",
            })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    return source_paths


def test_source_identities_and_order_come_from_base_manifest(tmp_path: Path) -> None:
    expected = _write_manifest(tmp_path / "support_manifest.csv")
    actual, targets = support.load_base_support_paths(tmp_path / "support_manifest.csv")
    assert actual == expected
    assert all(len(targets[domain]) == 8 for domain in support.DOMAINS)


def test_nested_prefixes_use_eight_distinct_source_images() -> None:
    pool = list(range(8))
    prefixes = support.support_prefixes(pool)
    assert prefixes == {1: [0], 2: [0, 1], 4: [0, 1, 2, 3], 8: pool}
    assert len(set(prefixes[8])) == 8


def test_all_domains_share_one_synthetic_pool() -> None:
    assert runner.SYNTHETIC_METHODS == (
        "SYNTH_BROAD6_K1_MAD", "SYNTH_BROAD6_K2_MAD",
        "SYNTH_BROAD6_K4_MAD", "SYNTH_BROAD6_K8_MAD",
    )
    assert support.DOMAINS == ("same", "background", "illumination", "view")


def test_support_extractor_calls_existing_broad6_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "train" / "good" / "x.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), (10, 20, 30)).save(image_path)
    record = _record(image_path, domain="source", label=0, role="train")
    calls = []

    def fake_broad6(image, relative_path, *, seed):
        calls.append((relative_path, seed))
        return SimpleNamespace(
            image=image, op1="Gamma", op2="JPEG",
            params1={"gamma": 1.1}, params2={"quality": 90},
        )

    monkeypatch.setattr(support, "broad6_compose2", fake_broad6)
    monkeypatch.setattr(
        support, "_extract_four_layers",
        lambda model, image, **kwargs: (
            {layer: np.zeros((4, 1280), np.float32) for layer in LAYERS}, (2, 2),
        ),
    )
    result = support.extract_or_load_synthetic_support(
        tmp_path / "cache.npz", record, support_index=1,
        dataset_root=tmp_path, model=object(), device=torch.device("cpu"),
        protocol_id=runner.PROTOCOL_ID,
    )
    assert calls == [("train/good/x.png", 42)]
    assert result.manifest["synthetic_view_count"] == 1
    assert result.manifest["shared_by_domains"] == ";".join(support.DOMAINS)


def test_cumulative_minimum_and_union_match_direct_values() -> None:
    official = np.full((4, 1, 3), 10.0, np.float32)
    individual = [
        np.full_like(official, value) for value in (9.0, 8.0, 12.0, 7.0, 6.0, 11.0, 5.0, 4.0)
    ]
    cumulative = None
    support_only = {}
    for index, value in enumerate(individual, 1):
        cumulative = value if cumulative is None else np.minimum(cumulative, value)
        if index in scoring.PREFIXES:
            support_only[index] = cumulative.copy()
    union = scoring.union_with_official(official, support_only)
    for shot in scoring.PREFIXES:
        direct = np.minimum.reduce([official, *individual[:shot]])
        assert np.array_equal(union[shot], direct)


def test_heldout_membership_is_exactly_1607(tmp_path: Path) -> None:
    records = []
    counts = {
        "same": (230, 459), "background": (93, 212),
        "illumination": (75, 198), "view": (92, 280),
    }
    targets = {}
    for domain, (normal_count, anomaly_count) in counts.items():
        domain_records = [
            _record(tmp_path / "test" / "good" / domain / f"n_{i}.png", domain=domain, label=0)
            for i in range(normal_count)
        ] + [
            _record(tmp_path / "test" / "bad" / domain / f"a_{i}.png", domain=domain, label=1)
            for i in range(anomaly_count)
        ]
        records.extend(domain_records)
        targets[domain] = [
            Path(record.path).resolve().relative_to(tmp_path.resolve()).as_posix()
            for record in domain_records[:8]
        ]
    heldout = support.heldout_from_base_manifest(records, targets, dataset_root=tmp_path)
    assert sum(len(values) for values in heldout.values()) == 1607
    assert sum(record.label == 0 for values in heldout.values() for record in values) == 458
    assert sum(record.label == 1 for values in heldout.values() for record in values) == 1149


def test_runner_reads_base_mad_and_never_refits() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    called_names = {
        node.func.id for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "fit_calibrators" not in called_names
    assert "_fit_source_mad" not in called_names
    assert "_read_base_mad" in called_names
    assert "mad_equal" in called_names


def test_runner_never_builds_or_scans_official_memory() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    scoring_source = Path(scoring.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {"build_exact_official_memory", "observe_fewshot_pil", "observe_pil", "_exact_top1_chunked"}
    called = {
        node.func.id for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not (called & forbidden)
    assert "official_banks" not in scoring_source
    assert "_support_prefix_minima" in scoring_source


def test_base_run_is_explicit_and_validated(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        runner.load_base_run(tmp_path)
    complete = {
        "protocol_id": base_runner.PROTOCOL_ID, "status": "complete", "run_type": "formal",
        "heldout_count": 1607, "heldout_normal_count": 458, "heldout_anomaly_count": 1149,
        "methods": list(base_runner.METHODS),
    }
    (tmp_path / "complete.json").write_text(json.dumps(complete), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        runner.load_base_run(tmp_path)


def test_comparison_gain_ratio_and_zero_denominator() -> None:
    synthetic_domain = []
    base_domain = []
    for domain in support.DOMAINS:
        for method, value in (("MAD_EQUAL", 0.5), ("SOURCE_EXTRA_K1_MAD", 0.6), ("TARGET_MATCHED_K1_MAD", 0.8)):
            base_domain.append({"method": method, "domain": domain, **{metric: value for metric in runner.METRICS}})
        synthetic_domain.append({
            "method": "SYNTH_BROAD6_K1_MAD", "domain": domain,
            **{metric: 0.7 for metric in runner.METRICS},
        })
    synth_summary = [{
        "method": "SYNTH_BROAD6_K1_MAD", "scope": scope,
        **{metric: 0.7 for metric in runner.METRICS},
    } for scope in ("four_domain_macro", "shifted_macro")]
    base_summary = [
        {"method": method, "scope": scope, **{metric: value for metric in runner.METRICS}}
        for scope in ("four_domain_macro", "shifted_macro")
        for method, value in (("MAD_EQUAL", 0.5), ("SOURCE_EXTRA_K1_MAD", 0.6), ("TARGET_MATCHED_K1_MAD", 0.8))
    ]
    rows = runner.comparison_rows(synthetic_domain, synth_summary, base_domain, base_summary)
    assert rows[0]["gain_ratio_image_AUROC"] == pytest.approx(0.5)

