"""AeBAD-S GNL-inspired Broad6 Compose2 synthetic few-shot control."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from DINOv3.MADEqual.aebad_s_broad6_compose2 import mmr_compatible_eval as mmr_eval
from DINOv3.MADEqual.aebad_s_full_p0 import runner as aebad_runner
from DINOv3.MADEqual.aebad_s_full_p0.dataset import discover_phase_a, load_phase_b_records
from DINOv3.MADEqual.aebad_s_target_fewshot import runner as base_runner
from DINOv3.MADEqual.aebad_s_target_fewshot.support import relative_path
from DINOv3.MADEqual.robustad_target_fewshot_p0 import runner as robust_runner
from DINOv3.MADEqual.robustad_target_fewshot_p0.scoring import PREFIXES
from DINOv3.relation_oracle_cache.extract import mad_equal
from DINOv3.relation_reliability.calibration import MADCalibrator
from DINOv3.relation_reliability.datasets import Record
from DINOv3.relation_reliability.official_metrics import validate_official_metric_backend
from DINOv3.relation_reliability.protocol import LAYERS
from DINOv3.relation_reliability.superadd_host import (
    construct_exact_host,
    install_import_paths,
    interpolate_score,
)

from .scoring import observe_synthetic_pil
from .support import (
    DOMAINS,
    extract_or_load_synthetic_support,
    heldout_from_base_manifest,
    load_base_support_paths,
    resolve_source_records,
)


PROTOCOL_ID = "mad_equal_aebad_s_synthetic_fewshot_seed42_v1"
CACHE_REVISION = "synthetic_broad6_raw_v1"
BASE_PROTOCOL_ID = base_runner.PROTOCOL_ID
SEED = 42
CHECKPOINT_SHA256 = "7c1da9a54b3bdb333f5ebc42e404b7f19b1b5bed504877623c9dc87397f41488"
SYNTHETIC_METHODS = tuple(f"SYNTH_BROAD6_K{shot}_MAD" for shot in PREFIXES)
REFERENCE_METHODS = (
    "MAD_EQUAL",
    *(f"SOURCE_EXTRA_K{shot}_MAD" for shot in PREFIXES),
    *(f"TARGET_MATCHED_K{shot}_MAD" for shot in PREFIXES),
)
DISPLAY_METHODS = (*REFERENCE_METHODS, *SYNTHETIC_METHODS)
METRICS = ("image_AUROC", "image_AUPR", "pixel_AUROC", "pixel_AUPR", "AUPRO")


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    base_runner._write_csv(path, rows)


def _atomic_json(path: Path, value: Any) -> None:
    base_runner._atomic_json(path, value)


def _atomic_npz(path: Path, **values: np.ndarray) -> None:
    base_runner._atomic_npz(path, **values)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_base_run(path: str | Path) -> dict[str, Any]:
    root = Path(path).resolve()
    with (root / "complete.json").open(encoding="utf-8") as handle:
        complete = json.load(handle)
    expected = {
        "protocol_id": BASE_PROTOCOL_ID,
        "status": "complete",
        "run_type": "formal",
        "heldout_count": 1607,
        "heldout_normal_count": 458,
        "heldout_anomaly_count": 1149,
    }
    for key, value in expected.items():
        if complete.get(key) != value:
            raise RuntimeError(f"base run has incompatible {key}: {complete.get(key)!r}")
    if set(base_runner.METHODS) - set(complete.get("methods", [])):
        raise RuntimeError("base run is missing frozen few-shot methods")
    required = (
        "support_manifest.csv", "calibration_stats.csv", "domain_metrics.csv",
        "summary.csv", "mechanism_macro.csv", "mmr_compatible_domain_metrics.csv",
        "mmr_compatible_summary.csv", "prediction_barrier.json",
    )
    for name in required:
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    raw_root = root / "raw_cache" / "heldout"
    if not raw_root.is_dir():
        raise FileNotFoundError(raw_root)
    return {
        "root": root,
        "complete": complete,
        "domain_metrics": _read_csv(root / "domain_metrics.csv"),
        "summary": _read_csv(root / "summary.csv"),
        "mechanism_macro": _read_csv(root / "mechanism_macro.csv"),
        "mmr_domain": _read_csv(root / "mmr_compatible_domain_metrics.csv"),
        "mmr_summary": _read_csv(root / "mmr_compatible_summary.csv"),
    }


def _read_base_mad(path: Path) -> MADCalibrator:
    return mmr_eval._read_calibration(path / "calibration_stats.csv")


def _load_base_official_raw(
    path: Path, *, relative: str,
) -> tuple[np.ndarray, tuple[int, int]]:
    with np.load(path, allow_pickle=False) as payload:
        if str(payload["protocol_id"].item()) != BASE_PROTOCOL_ID:
            raise RuntimeError(f"base raw protocol mismatch: {path}")
        if str(payload["cache_revision"].item()) != base_runner.CACHE_REVISION:
            raise RuntimeError(f"base raw revision mismatch: {path}")
        if str(payload["relative_path"].item()) != relative:
            raise RuntimeError(f"base raw path mismatch: {path}")
        raw = np.asarray(payload["official_raw"], np.float32)
        output_shape = tuple(int(value) for value in payload["output_shape"].tolist())
    if (
        raw.ndim != 3 or raw.shape[0] != len(LAYERS)
        or len(output_shape) != 2 or not np.isfinite(raw).all()
    ):
        raise RuntimeError(f"invalid base Official raw: {path}")
    return raw, output_shape


def _save_synthetic_raw(
    path: Path,
    *,
    relative: str,
    output_shape: tuple[int, int],
    observation: Any,
) -> None:
    values: dict[str, np.ndarray] = {
        "protocol_id": np.asarray(PROTOCOL_ID),
        "cache_revision": np.asarray(CACHE_REVISION),
        "relative_path": np.asarray(relative),
        "output_shape": np.asarray(output_shape, np.int32),
    }
    for shot in PREFIXES:
        values[f"synthetic_raw_{shot}"] = np.asarray(
            observation.synthetic_raw[shot], np.float32,
        )
        values[f"synthetic_support_{shot}"] = np.asarray(
            observation.support_only[shot], np.float32,
        )
    _atomic_npz(path, **values)


def _load_synthetic_raw(
    path: Path, *, relative: str,
) -> tuple[dict[str, np.ndarray], tuple[int, int]]:
    with np.load(path, allow_pickle=False) as payload:
        if str(payload["protocol_id"].item()) != PROTOCOL_ID:
            raise RuntimeError(f"synthetic raw protocol mismatch: {path}")
        if str(payload["cache_revision"].item()) != CACHE_REVISION:
            raise RuntimeError(f"synthetic raw revision mismatch: {path}")
        if str(payload["relative_path"].item()) != relative:
            raise RuntimeError(f"synthetic raw path mismatch: {path}")
        output_shape = tuple(int(value) for value in payload["output_shape"].tolist())
        names = [
            *[f"synthetic_raw_{shot}" for shot in PREFIXES],
            *[f"synthetic_support_{shot}" for shot in PREFIXES],
        ]
        values = {name: np.asarray(payload[name], np.float32) for name in names}
    shape = values["synthetic_raw_1"].shape
    if (
        len(output_shape) != 2 or len(shape) != 3 or shape[0] != len(LAYERS)
        or any(value.shape != shape or not np.isfinite(value).all() for value in values.values())
    ):
        raise RuntimeError(f"invalid synthetic raw cache: {path}")
    return values, output_shape


def _support_tensors(features: Sequence[Any], *, device: torch.device) -> dict[int, list[torch.Tensor]]:
    return base_runner._support_tensors(features, device=device)


def _synthetic_map(
    values: Mapping[str, np.ndarray],
    shot: int,
    mad: MADCalibrator,
    output_shape: tuple[int, int],
) -> np.ndarray:
    patch = mad_equal(values[f"synthetic_raw_{shot}"], mad.medians, mad.scales)
    result = np.asarray(interpolate_score(patch, output_shape), np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError(f"non-finite synthetic prediction map: K{shot}")
    return result


def summarize_synthetic(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in SYNTHETIC_METHODS:
        by_domain = {row["domain"]: row for row in rows if row["method"] == method}
        if set(by_domain) != set(DOMAINS):
            raise RuntimeError(f"incomplete synthetic metrics: {method}")
        for scope, domains in (
            ("four_domain_macro", DOMAINS),
            ("shifted_macro", ("background", "illumination", "view")),
        ):
            item: dict[str, Any] = {"method": method, "scope": scope}
            for metric in METRICS:
                item[metric] = mean(float(by_domain[domain][metric]) for domain in domains)
            output.append(item)
    return output


def comparison_rows(
    synthetic_domain: Sequence[dict[str, Any]],
    synthetic_summary: Sequence[dict[str, Any]],
    base_domain: Sequence[Mapping[str, Any]],
    base_summary: Sequence[Mapping[str, Any]],
    *,
    epsilon: float = 1e-8,
) -> list[dict[str, Any]]:
    new = [dict(row, scope=row["domain"]) for row in synthetic_domain] + list(synthetic_summary)
    old = [dict(row, scope=row["domain"]) for row in base_domain] + list(base_summary)
    output: list[dict[str, Any]] = []
    for row in new:
        method = str(row["method"])
        shot = int(method.split("_K", 1)[1].split("_", 1)[0])
        scope = str(row["scope"])
        mad = next(item for item in old if item["method"] == "MAD_EQUAL" and item["scope"] == scope)
        source = next(
            item for item in old
            if item["method"] == f"SOURCE_EXTRA_K{shot}_MAD" and item["scope"] == scope
        )
        target = next(
            item for item in old
            if item["method"] == f"TARGET_MATCHED_K{shot}_MAD" and item["scope"] == scope
        )
        value: dict[str, Any] = {"method": method, "scope": scope, "K": shot}
        for metric in METRICS:
            synthetic_value = float(row[metric])
            mad_value = float(mad[metric])
            source_value = float(source[metric])
            target_value = float(target[metric])
            denominator = target_value - source_value
            value[f"synthetic_minus_MAD_EQUAL_{metric}"] = synthetic_value - mad_value
            value[f"synthetic_minus_SOURCE_EXTRA_K{shot}_{metric}"] = synthetic_value - source_value
            value[f"TARGET_MATCHED_K{shot}_minus_synthetic_{metric}"] = target_value - synthetic_value
            value[f"gain_ratio_{metric}"] = (
                "" if abs(denominator) < epsilon
                else (synthetic_value - source_value) / denominator
            )
        output.append(value)
    return output


def _synthetic_mechanism_rows(
    domain: str,
    records: Sequence[Record],
    loaded: Sequence[tuple[np.ndarray, dict[str, np.ndarray]]],
    mad: MADCalibrator,
) -> list[dict[str, Any]]:
    normal: dict[tuple[int, int], list[float]] = {}
    anomaly: dict[tuple[int, int], list[float]] = {}
    for record, (official, values) in zip(records, loaded, strict=True):
        patch_mask = (
            robust_runner._patch_mask(record, tuple(official.shape[1:]))
            if int(record.label) == 1 else None
        )
        for shot in PREFIXES:
            reduction = (
                official - values[f"synthetic_raw_{shot}"]
            ) / (np.asarray(mad.scales)[:, None, None] + 1e-8)
            for offset, layer in enumerate(LAYERS):
                key = (shot, layer)
                if int(record.label) == 0:
                    normal.setdefault(key, []).append(float(reduction[offset].mean()))
                else:
                    if patch_mask is None or not patch_mask.any():
                        raise RuntimeError(f"invalid defect patch mask: {record.path}")
                    anomaly.setdefault(key, []).append(float(reduction[offset][patch_mask].mean()))
    rows: list[dict[str, Any]] = []
    for shot in PREFIXES:
        for layer in LAYERS:
            rn = mean(normal[(shot, layer)])
            ra = mean(anomaly[(shot, layer)])
            rows.append({
                "method": f"SYNTH_BROAD6_K{shot}_MAD", "domain": domain,
                "K": shot, "layer": layer, "R_N": rn, "R_A": ra, "G": rn - ra,
                "normal_image_count": len(normal[(shot, layer)]),
                "anomaly_image_count": len(anomaly[(shot, layer)]),
                "aggregation": "image_equal",
            })
    return rows


def _synthetic_mechanism_macro(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in SYNTHETIC_METHODS:
        shot = int(method.split("_K", 1)[1].split("_", 1)[0])
        for scope, domains in (
            ("four_domain_macro", DOMAINS),
            ("shifted_macro", ("background", "illumination", "view")),
        ):
            for layer in LAYERS:
                selected = [
                    row for row in rows
                    if row["method"] == method and row["domain"] in domains
                    and int(row["layer"]) == layer
                ]
                value = {
                    "method": method, "K": shot, "layer": layer, "scope": scope,
                    "R_N": mean(float(row["R_N"]) for row in selected),
                    "R_A": mean(float(row["R_A"]) for row in selected),
                    "G": mean(float(row["G"]) for row in selected),
                    "aggregation": "domain_equal_then_layer_preserved",
                }
                output.append(value)
            selected_layers = [row for row in output if row["method"] == method and row["scope"] == scope]
            output.append({
                "method": method, "K": shot, "layer": "four_layer_macro",
                "scope": f"{scope}_four_layer_macro",
                "R_N": mean(float(row["R_N"]) for row in selected_layers),
                "R_A": mean(float(row["R_A"]) for row in selected_layers),
                "G": mean(float(row["G"]) for row in selected_layers),
                "aggregation": "domain_equal_layer_equal",
            })
    return output


def _combined_mechanism_macro(
    base_rows: Sequence[Mapping[str, Any]],
    synthetic_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected = [
        dict(row) for row in base_rows
        if str(row["method"]).startswith(("SOURCE_EXTRA_", "TARGET_MATCHED_"))
    ]
    output = [*selected, *synthetic_rows]
    for method in (*[f"SOURCE_EXTRA_K{k}_MAD" for k in PREFIXES], *[f"TARGET_MATCHED_K{k}_MAD" for k in PREFIXES]):
        for scope in ("four_domain_macro", "shifted_macro"):
            layers = [row for row in selected if row["method"] == method and row["scope"] == scope]
            if len(layers) != 4:
                raise RuntimeError(f"base mechanism macro is incomplete: {method}/{scope}")
            output.append({
                "method": method, "K": layers[0]["K"], "layer": "four_layer_macro",
                "scope": f"{scope}_four_layer_macro",
                "R_N": mean(float(row["R_N"]) for row in layers),
                "R_A": mean(float(row["R_A"]) for row in layers),
                "G": mean(float(row["G"]) for row in layers),
                "aggregation": "domain_equal_layer_equal",
            })
    return output


def _mmr_rows(
    domain_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    summary = mmr_eval.summarize_domain_metrics(domain_rows, methods=SYNTHETIC_METHODS)
    return [
        *[dict(row, scope=row["domain"]) for row in domain_rows],
        *summary,
    ]


def _report(
    support_rows: Sequence[dict[str, Any]],
    synthetic_domain: Sequence[dict[str, Any]],
    synthetic_summary: Sequence[dict[str, Any]],
    comparison: Sequence[dict[str, Any]],
    base_summary: Sequence[Mapping[str, Any]],
    mmr_rows: Sequence[dict[str, Any]],
    timing: Sequence[dict[str, Any]],
) -> str:
    def shown(value: Any, signed: bool = False) -> str:
        if value in (None, "", "NA"):
            return "NA"
        return f"{float(value):+.6f}" if signed else f"{float(value):.6f}"

    lines = [
        "# AeBAD-S GNL-inspired Broad6 Compose2 Synthetic Few-shot 客观报告", "",
        f"Protocol: `{PROTOCOL_ID}`。本实验只新增八张 synthetic normal support；不重建或扫描 Official 100K，不重新拟合 MAD，也不改变 base-run 的 1,607 张共同 held-out 集。", "",
        "## Synthetic support", "",
        "| index | source identity | op1 | op2 | K1 | K2 | K4 | K8 |",
        "|---:|---|---|---|---:|---:|---:|---:|",
    ]
    for row in support_rows:
        lines.append(
            f"| {row['support_index']} | {row['relative_path']} | {row['op1']} | {row['op2']} | "
            f"{row['included_in_K1']} | {row['included_in_K2']} | {row['included_in_K4']} | {row['included_in_K8']} |"
        )
    lines += ["", "## Synthetic 逐域指标", "",
              "| method | domain | Image AUROC | Image AUPR | Pixel AUROC | Pixel AUPR | AUPRO |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in synthetic_domain:
        lines.append(
            f"| {row['method']} | {row['domain']} | "
            + " | ".join(shown(row[name]) for name in METRICS) + " |"
        )
    combined = [row for row in base_summary if row["method"] in REFERENCE_METHODS] + list(synthetic_summary)
    lines += ["", "## Base 与 Synthetic macro 并表", "",
              "| method | scope | Image AUROC | Image AUPR | Pixel AUROC | Pixel AUPR | AUPRO |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in combined:
        lines.append(
            f"| {row['method']} | {row['scope']} | "
            + " | ".join(shown(row[name]) for name in METRICS) + " |"
        )
    lines += ["", "## 直接差值与增益比例（Image AUROC）", "",
              "| method | scope | Synth−MAD | Synth−Source | Target−Synth | gain ratio |",
              "|---|---|---:|---:|---:|---:|"]
    for row in comparison:
        shot = int(row["K"])
        source_key = f"synthetic_minus_SOURCE_EXTRA_K{shot}_image_AUROC"
        target_key = f"TARGET_MATCHED_K{shot}_minus_synthetic_image_AUROC"
        lines.append(
            f"| {row['method']} | {row['scope']} | "
            f"{shown(row['synthetic_minus_MAD_EQUAL_image_AUROC'], True)} | "
            f"{shown(row[source_key], True)} | "
            f"{shown(row[target_key], True)} | "
            f"{shown(row['gain_ratio_image_AUROC'])} |"
        )
    lines += ["", "## MMR-compatible I-AUROC", "",
              "该部分是输出侧后评测，不是完整 MMR 输入协议复现。", "",
              "| method | scope | full I-AUROC | compatible I-AUROC | delta |",
              "|---|---|---:|---:|---:|"]
    for row in mmr_rows:
        if row["scope"] not in ("four_domain_macro", "shifted_macro"):
            continue
        lines.append(
            f"| {row['method']} | {row['scope']} | "
            f"{shown(row['full_protocol_image_AUROC'])} | "
            f"{shown(row['mmr_compatible_image_AUROC'])} | {shown(row['delta'], True)} |"
        )
    lines += ["", "## 客观回答边界", ""]
    for shot in PREFIXES:
        rows = [row for row in comparison if int(row["K"]) == shot and row["scope"] in DOMAINS]
        above = [row["scope"] for row in rows if float(row[f"synthetic_minus_SOURCE_EXTRA_K{shot}_image_AUROC"]) > 0]
        not_above = [row["scope"] for row in rows if row["scope"] not in above]
        macro = next(row for row in comparison if int(row["K"]) == shot and row["scope"] == "four_domain_macro")
        lines.append(
            f"- K{shot}：Image AUROC 高于同 K clean Source-extra 的 domains 为 "
            f"{', '.join(above) if above else '无'}；未超过的 domains 为 "
            f"{', '.join(not_above) if not_above else '无'}；四域 macro gain ratio="
            f"{shown(macro['gain_ratio_image_AUROC'])}。"
        )
    lines += [
        "", "### 四域 macro 的五指标增益恢复比例", "",
        "| K | Image AUROC | Image AUPR | Pixel AUROC | Pixel AUPR | AUPRO | Image/Pixel direction |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for shot in PREFIXES:
        macro = next(
            row for row in comparison
            if int(row["K"]) == shot and row["scope"] == "four_domain_macro"
        )
        source_deltas = [
            float(macro[f"synthetic_minus_SOURCE_EXTRA_K{shot}_{metric}"])
            for metric in METRICS
        ]
        image_sign = np.sign(source_deltas[0])
        pixel_signs = [np.sign(value) for value in source_deltas[2:]]
        direction = "同向" if all(value == image_sign for value in pixel_signs) else "不完全同向"
        lines.append(
            f"| {shot} | "
            + " | ".join(shown(macro[f"gain_ratio_{metric}"]) for metric in METRICS)
            + f" | {direction} |"
        )
    lines += [
        "", "比例的分子为 Synthetic−同 K Source-extra，分母为 Real Target-matched−同 K Source-extra；接近零的分母记为 NA。Image/Pixel direction 按 Synthetic−Source-extra 的 Image AUROC 与三项 Pixel 指标符号是否一致记录。",
        "", "## Timing", "",
        "| stage | seconds |", "|---|---:|",
    ]
    for row in timing:
        lines.append(f"| {row['stage']} | {float(row['seconds']):.3f} |")
    lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    timing: list[dict[str, Any]] = []
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal synthetic few-shot requires CUDA")
    validate_official_metric_backend()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    base = load_base_run(args.base_run)
    base_root: Path = base["root"]
    mad = _read_base_mad(base_root)

    install_import_paths(Path(args.project_root), Path(args.official_root), Path(args.dinov3_root))
    model, _config = construct_exact_host(
        Path(args.official_root), Path(args.dinov3_root), Path(args.checkpoint), CHECKPOINT_SHA256,
    )

    stage = time.perf_counter()
    dataset_root, train_records, locators = discover_phase_a(args.aebad_root, formal=True)
    records = load_phase_b_records(dataset_root, locators, formal=True)
    source_paths, target_paths = load_base_support_paths(base_root / "support_manifest.csv")
    source_records = resolve_source_records(source_paths, train_records, dataset_root=dataset_root)
    heldout = heldout_from_base_manifest(records, target_paths, dataset_root=dataset_root)
    locator_by_path = {str(Path(locator.path).resolve()): locator for locator in locators}
    timing.append({"stage": "base_and_membership", "seconds": time.perf_counter() - stage})

    stage = time.perf_counter()
    synthetic = [
        extract_or_load_synthetic_support(
            output / "synthetic_support_features" / f"{index:02d}.npz",
            record,
            support_index=index,
            dataset_root=dataset_root,
            model=model,
            device=device,
            protocol_id=PROTOCOL_ID,
        )
        for index, record in enumerate(source_records, start=1)
    ]
    support_rows = [dict(item.manifest) for item in synthetic]
    _write_csv(output / "synthetic_support_manifest.csv", support_rows)
    support_tensors = _support_tensors([item.features for item in synthetic], device=device)
    timing.append({"stage": "synthetic_support_features", "seconds": time.perf_counter() - stage})

    stage = time.perf_counter()
    raw_root = output / "synthetic_raw_cache"
    base_raw_root = base_root / "raw_cache" / "heldout"
    for domain in DOMAINS:
        for index, record in enumerate(heldout[domain], start=1):
            locator = locator_by_path[str(Path(record.path).resolve())]
            base_raw, output_shape = _load_base_official_raw(
                base_raw_root / domain / f"{locator.sample_id}.npz",
                relative=locator.relative_path,
            )
            path = raw_root / domain / f"{locator.sample_id}.npz"
            if path.is_file():
                _load_synthetic_raw(path, relative=locator.relative_path)
            else:
                with Image.open(record.path) as source:
                    observation = observe_synthetic_pil(
                        model, source.convert("RGB"), synthetic_support=support_tensors,
                        cached_official_raw=base_raw, device=device,
                    )
                _save_synthetic_raw(
                    path, relative=locator.relative_path,
                    output_shape=output_shape, observation=observation,
                )
            print(f"[AEBAD-SYNTH-FEWSHOT] raw {domain} {index}/{len(heldout[domain])}", flush=True)
    timing.append({"stage": "synthetic_support_scan", "seconds": time.perf_counter() - stage})

    stage = time.perf_counter()
    synthetic_domain: list[dict[str, Any]] = []
    mmr_domain: list[dict[str, Any]] = []
    mechanism_domain: list[dict[str, Any]] = []
    for domain in DOMAINS:
        domain_records = heldout[domain]
        loaded: list[tuple[np.ndarray, dict[str, np.ndarray], tuple[int, int]]] = []
        for record in domain_records:
            locator = locator_by_path[str(Path(record.path).resolve())]
            official, base_shape = _load_base_official_raw(
                base_raw_root / domain / f"{locator.sample_id}.npz", relative=locator.relative_path,
            )
            values, shape = _load_synthetic_raw(
                raw_root / domain / f"{locator.sample_id}.npz", relative=locator.relative_path,
            )
            if shape != base_shape:
                raise RuntimeError("synthetic cache output shape differs from base run")
            loaded.append((official, values, shape))
        labels = [int(record.label) for record in domain_records]
        original_masks = [mmr_eval._load_original_mask(record) for record in domain_records]
        cropped_masks = [mmr_eval.resize_and_crop_mask(mask) for mask in original_masks]
        compatible_labels = [
            mmr_eval.mmr_compatible_label(label, mask)
            for label, mask in zip(labels, cropped_masks, strict=True)
        ]
        mechanism_domain.extend(_synthetic_mechanism_rows(
            domain, domain_records,
            [(official, values) for official, values, _shape in loaded], mad,
        ))
        for shot in PREFIXES:
            method = f"SYNTH_BROAD6_K{shot}_MAD"
            maps = [_synthetic_map(values, shot, mad, shape) for _official, values, shape in loaded]
            masks = [
                aebad_runner._load_mask(record, value.shape)
                for record, value in zip(domain_records, maps, strict=True)
            ]
            metrics = aebad_runner._evaluate(labels, masks, maps)
            synthetic_domain.append({
                "method": method, "domain": domain, "image_count": len(domain_records),
                "normal_count": sum(label == 0 for label in labels),
                "anomaly_count": sum(label == 1 for label in labels), **metrics,
            })
            full_scores = [float(value.max()) for value in maps]
            cropped_scores = [mmr_eval.cropped_image_score(value)[1] for value in maps]
            full_auc = float(roc_auc_score(labels, full_scores))
            compatible_auc = float(roc_auc_score(compatible_labels, cropped_scores))
            mmr_domain.append({
                "method": method, "domain": domain, "image_count": len(domain_records),
                "original_anomaly_count": sum(labels),
                "compatible_anomaly_count": sum(compatible_labels),
                "relabelled_to_normal_count": sum(
                    old == 1 and new == 0
                    for old, new in zip(labels, compatible_labels, strict=True)
                ),
                "full_protocol_image_AUROC": full_auc,
                "mmr_compatible_image_AUROC": compatible_auc,
                "delta": compatible_auc - full_auc,
                "scope": domain,
            })
            print(f"[AEBAD-SYNTH-FEWSHOT] metrics {domain} {method} complete", flush=True)

    synthetic_summary = summarize_synthetic(synthetic_domain)
    comparison = comparison_rows(
        synthetic_domain, synthetic_summary, base["domain_metrics"], base["summary"],
    )
    synthetic_mechanism_macro = _synthetic_mechanism_macro(mechanism_domain)
    combined_mechanism = _combined_mechanism_macro(
        base["mechanism_macro"], synthetic_mechanism_macro,
    )
    mmr_rows = _mmr_rows(mmr_domain)
    timing.append({"stage": "metrics_mmr_mechanism", "seconds": time.perf_counter() - stage})
    timing.append({"stage": "total", "seconds": time.perf_counter() - started})

    _write_csv(output / "synthetic_domain_metrics.csv", synthetic_domain)
    _write_csv(output / "synthetic_summary.csv", synthetic_summary)
    _write_csv(output / "comparison_to_source_and_target.csv", comparison)
    _write_csv(output / "mechanism_by_domain_layer.csv", mechanism_domain)
    _write_csv(output / "mechanism_macro.csv", combined_mechanism)
    _write_csv(output / "mmr_compatible_metrics.csv", mmr_rows)
    _write_csv(output / "timing.csv", timing)
    (output / "objective_report.md").write_text(
        _report(
            support_rows, synthetic_domain, synthetic_summary, comparison,
            base["summary"], mmr_rows, timing,
        ),
        encoding="utf-8",
    )
    _atomic_json(output / "complete.json", {
        "protocol_id": PROTOCOL_ID, "status": "complete", "run_type": "formal",
        "dataset": "AeBAD-S", "seed": SEED, "base_run": str(base_root),
        "heldout_count": 1607, "heldout_normal_count": 458,
        "heldout_anomaly_count": 1149, "synthetic_support_count": 8,
        "K": list(PREFIXES), "methods": list(SYNTHETIC_METHODS),
        "official_memory_rebuilt": False, "official_memory_scanned": False,
        "source_mad_refitted": False,
        "support_description": "GNL-inspired Broad6 Compose2 synthetic support",
        "aupro_backend": "ADEval CUDA nstrips=200",
    })
    print(f"AEBAD_S_SYNTHETIC_FEWSHOT_COMPLETE={output}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aebad-root", type=Path, required=True)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--dinov3-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
