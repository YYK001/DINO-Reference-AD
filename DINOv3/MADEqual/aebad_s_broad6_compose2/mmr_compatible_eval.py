from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image
from DINOv3.MADEqual.aebad_s_full_p0.dataset import (
    DOMAINS,
    discover_phase_a,
    load_phase_b_records,
)
from DINOv3.relation_reliability.calibration import MADCalibrator
from DINOv3.relation_reliability.protocol import LAYERS


METHODS = (
    "SUPERADD_OFFICIAL",
    "MAD_EQUAL",
    "BROAD6_COMPOSE2_MAD_EQUAL",
)
PHYS4_METHOD = "PHYS4_COMPOSE2_MAD_EQUAL"
CROP_START = 16
CROP_STOP = 240
RESIZE_SIZE = 256
PUBLIC_CONTEXT = {
    "MMR": 84.7,
    "MST": 87.5,
    "MMR+++": 88.2,
}


def center_crop_224(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (RESIZE_SIZE, RESIZE_SIZE):
        raise ValueError(f"center crop expects 256x256, got {array.shape}")
    return array[CROP_START:CROP_STOP, CROP_START:CROP_STOP]


def resize_and_crop_map(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("anomaly map must be rank 2")
    resized = Image.fromarray(array, mode="F").resize(
        (RESIZE_SIZE, RESIZE_SIZE), Image.Resampling.BILINEAR,
    )
    return center_crop_224(np.asarray(resized, dtype=np.float32))


def resize_and_crop_mask(mask: np.ndarray) -> np.ndarray:
    array = (np.asarray(mask) > 0).astype(np.uint8)
    if array.ndim != 2:
        raise ValueError("mask must be rank 2")
    resized = Image.fromarray(array * 255, mode="L").resize(
        (RESIZE_SIZE, RESIZE_SIZE), Image.Resampling.NEAREST,
    )
    return center_crop_224(np.asarray(resized, dtype=np.uint8) > 0)


def mmr_compatible_label(original_label: int, cropped_mask: np.ndarray) -> int:
    if int(original_label) == 0:
        return 0
    return int(np.asarray(cropped_mask, dtype=bool).any())


def cropped_image_score(full_map: np.ndarray) -> tuple[np.ndarray, float]:
    cropped = resize_and_crop_map(full_map)
    return cropped, float(cropped.max())


def summarize_domain_metrics(
    rows: Sequence[dict[str, Any]],
    *,
    methods: Sequence[str] = METHODS,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in methods:
        by_domain = {
            str(row["domain"]): row for row in rows if row["method"] == method
        }
        if set(by_domain) != set(DOMAINS):
            raise RuntimeError(f"incomplete MMR-compatible domains for {method}")
        for scope, domains in (
            ("four_domain_macro", DOMAINS),
            ("shifted_macro", ("background", "illumination", "view")),
        ):
            full = float(np.mean([
                float(by_domain[domain]["full_protocol_image_AUROC"])
                for domain in domains
            ]))
            compatible = float(np.mean([
                float(by_domain[domain]["mmr_compatible_image_AUROC"])
                for domain in domains
            ]))
            output.append({
                "method": method,
                "scope": scope,
                "full_protocol_image_AUROC": full,
                "mmr_compatible_image_AUROC": compatible,
                "delta": compatible - full,
            })
    return output


def _read_calibration(path: Path) -> MADCalibrator:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_layer = {int(row["layer"]): row for row in rows}
    if set(by_layer) != set(LAYERS):
        raise RuntimeError(f"calibration layers are incomplete: {path}")
    return MADCalibrator(
        medians=np.asarray([float(by_layer[layer]["median"]) for layer in LAYERS], np.float32),
        scales=np.asarray([
            float(by_layer[layer]["mad_scale_1p4826"]) for layer in LAYERS
        ], np.float32),
    )


def _load_raw(path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        raw = np.asarray(payload["raw_distances"], dtype=np.float32)
        output_shape = tuple(int(value) for value in payload["output_shape"].tolist())
    if raw.ndim != 3 or raw.shape[0] != len(LAYERS):
        raise RuntimeError(f"invalid four-layer raw cache: {path}")
    return raw, output_shape


def _load_original_mask(record: Any) -> np.ndarray:
    if int(record.label) == 0:
        return np.zeros((RESIZE_SIZE, RESIZE_SIZE), dtype=np.uint8)
    if not record.mask_path:
        raise RuntimeError(f"anomaly lacks mask: {record.path}")
    with Image.open(record.mask_path) as source:
        return np.asarray(source.convert("L"), dtype=np.uint8)


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    from DINOv3.MADEqual.aebad_s_full_p0 import runner as baseline_runner

    baseline_runner._write_csv(path, rows)


def _report(
    domain_rows: Sequence[dict[str, Any]],
    summary_rows: Sequence[dict[str, Any]],
) -> str:
    lines = [
        "# AeBAD-S MMR-compatible I-AUROC 后评测",
        "",
        "本实验是 MMR-compatible post-evaluation，不是完整 MMR 输入协议复现。SuperADD 的 DINO 特征仍来自完整原图；仅在已有输出异常图上执行 256×256 双线性 resize、224×224 center crop，并依据裁剪后 GT mask 是否为空重新定义异常标签。",
        "",
        "因此，本实验用于量化评测裁剪与标签重定义对现有 I-AUROC 的影响，不能声称 SuperADD 已在完全相同的 MMR 输入协议下完成评测。",
        "",
        "## 逐域结果",
        "",
        "| method | domain | images | original anomalies | compatible anomalies | relabelled normal | full I-AUROC | compatible I-AUROC | delta |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in domain_rows:
        lines.append(
            f"| {row['method']} | {row['domain']} | {row['image_count']} | "
            f"{row['original_anomaly_count']} | {row['compatible_anomaly_count']} | "
            f"{row['relabelled_to_normal_count']} | "
            f"{row['full_protocol_image_AUROC']:.6f} | "
            f"{row['mmr_compatible_image_AUROC']:.6f} | {row['delta']:+.6f} |"
        )
    lines += [
        "", "## Macro 结果", "",
        "| method | scope | full I-AUROC | compatible I-AUROC | delta |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['method']} | {row['scope']} | "
            f"{row['full_protocol_image_AUROC']:.6f} | "
            f"{row['mmr_compatible_image_AUROC']:.6f} | {row['delta']:+.6f} |"
        )
    context_methods = ["BROAD6_COMPOSE2_MAD_EQUAL"]
    if any(row["method"] == PHYS4_METHOD for row in summary_rows):
        context_methods.append(PHYS4_METHOD)
    lines += [
        "", "## 与公开结果的表面差距", "",
        "下表只计算数值表面差距，不构成同协议直接比较。", "",
        "| evaluated method | public result | reported I-AUROC | compatible minus reported |",
        "|---|---|---:|---:|",
    ]
    for method in context_methods:
        result = next(
            row for row in summary_rows
            if row["method"] == method and row["scope"] == "four_domain_macro"
        )
        score_percent = float(result["mmr_compatible_image_AUROC"]) * 100.0
        for name, public_score in PUBLIC_CONTEXT.items():
            lines.append(
                f"| {method} | {name} | {public_score:.1f}% | "
                f"{score_percent - public_score:+.2f} pp |"
            )
    lines += [
        "", "## 协议边界", "",
        "- 该后评测没有重新运行 DINO、构建 memory、执行 Exact 1-NN 或重新拟合 MAD。",
        "- 异常图由既有 raw distance、既有 calibration statistics、既有 `mad_equal` 和 `interpolate_score` 重建。",
        "- MMR-compatible 标签只由原始标签和中心裁剪后的 GT mask 是否为空决定。",
        "- 公开 MMR、MST 与 MMR+++ 数字仅作为上下文，不能视为完全相同输入与实现协议下的直接竞争结果。",
        "",
    ]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    # Keep the output-only resize/crop helpers usable without the inference stack.
    from sklearn.metrics import roc_auc_score
    from DINOv3.MADEqual.aebad_s_broad6_compose2 import runner as broad6_runner
    from DINOv3.MADEqual.aebad_s_full_p0 import runner as baseline_runner
    from DINOv3.relation_reliability.superadd_host import interpolate_score

    baseline_output = Path(args.baseline_output).resolve()
    broad6_output = Path(args.broad6_output).resolve()
    phys4_output = Path(args.phys4_output).resolve() if args.phys4_output else None
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    for root in (baseline_output, broad6_output):
        for required in (
            "calibration_stats.csv", "dataset_manifest.csv", "prediction_barrier.json",
        ):
            if not (root / required).is_file():
                raise FileNotFoundError(root / required)
    if phys4_output is not None:
        for required in (
            "calibration_stats.csv", "dataset_manifest.csv", "prediction_barrier.json",
        ):
            if not (phys4_output / required).is_file():
                raise FileNotFoundError(phys4_output / required)

    dataset_root, _train, locators = discover_phase_a(args.aebad_root, formal=True)
    records = load_phase_b_records(dataset_root, locators, formal=True)
    baseline_mad = _read_calibration(baseline_output / "calibration_stats.csv")
    broad6_mad = _read_calibration(broad6_output / "calibration_stats.csv")
    phys4_mad = (
        _read_calibration(phys4_output / "calibration_stats.csv")
        if phys4_output is not None else None
    )
    methods = (*METHODS, PHYS4_METHOD) if phys4_output is not None else METHODS

    sample_rows: list[dict[str, Any]] = []
    domain_accumulator: dict[tuple[str, str], dict[str, list[Any]]] = {
        (method, domain): {
            "original_labels": [], "compatible_labels": [],
            "full_scores": [], "cropped_scores": [],
        }
        for method in methods for domain in DOMAINS
    }
    for index, (locator, record) in enumerate(zip(locators, records, strict=True), 1):
        baseline_raw, baseline_shape = _load_raw(
            baseline_output / "raw_cache" / "test" / f"{locator.sample_id}.npz"
        )
        broad6_raw, broad6_shape = _load_raw(
            broad6_output / "raw_cache" / "test" / f"{locator.sample_id}.npz"
        )
        baseline_scores = baseline_runner._score_raw(baseline_raw, mad=baseline_mad)
        broad6_score = broad6_runner._score_raw(broad6_raw, broad6_mad)
        patch_maps = {
            "SUPERADD_OFFICIAL": (baseline_scores["SUPERADD_OFFICIAL"], baseline_shape),
            "MAD_EQUAL": (baseline_scores["MAD_EQUAL"], baseline_shape),
            "BROAD6_COMPOSE2_MAD_EQUAL": (broad6_score, broad6_shape),
        }
        if phys4_output is not None:
            phys4_raw, phys4_shape = _load_raw(
                phys4_output / "raw_cache" / "test" / f"{locator.sample_id}.npz"
            )
            if phys4_mad is None:
                raise RuntimeError("Phys4 calibration is unavailable")
            patch_maps[PHYS4_METHOD] = (
                broad6_runner._score_raw(phys4_raw, phys4_mad), phys4_shape,
            )

        cropped_mask = resize_and_crop_mask(_load_original_mask(record))
        compatible_label = mmr_compatible_label(record.label, cropped_mask)
        mask_empty = bool(int(record.label) == 1 and not cropped_mask.any())
        for method, (patch_map, output_shape) in patch_maps.items():
            full_map = np.asarray(interpolate_score(patch_map, output_shape), dtype=np.float32)
            full_score = float(full_map.max())
            _cropped_map, crop_score = cropped_image_score(full_map)
            sample_rows.append({
                "sample_id": locator.sample_id,
                "relative_path": locator.relative_path,
                "domain": locator.domain,
                "method": method,
                "original_label": int(record.label),
                "mmr_compatible_label": compatible_label,
                "mask_empty_after_crop": mask_empty,
                "full_image_score": full_score,
                "cropped_image_score": crop_score,
            })
            accumulator = domain_accumulator[(method, locator.domain)]
            accumulator["original_labels"].append(int(record.label))
            accumulator["compatible_labels"].append(compatible_label)
            accumulator["full_scores"].append(full_score)
            accumulator["cropped_scores"].append(crop_score)
        print(f"[MMR-COMPATIBLE-EVAL] sample {index}/{len(locators)}", flush=True)

    domain_rows: list[dict[str, Any]] = []
    for method in methods:
        for domain in DOMAINS:
            values = domain_accumulator[(method, domain)]
            original = np.asarray(values["original_labels"], dtype=np.uint8)
            compatible = np.asarray(values["compatible_labels"], dtype=np.uint8)
            full_auc = float(roc_auc_score(original, values["full_scores"]))
            compatible_auc = float(roc_auc_score(compatible, values["cropped_scores"]))
            domain_rows.append({
                "method": method,
                "domain": domain,
                "image_count": int(original.size),
                "original_anomaly_count": int(original.sum()),
                "compatible_anomaly_count": int(compatible.sum()),
                "relabelled_to_normal_count": int(((original == 1) & (compatible == 0)).sum()),
                "full_protocol_image_AUROC": full_auc,
                "mmr_compatible_image_AUROC": compatible_auc,
                "delta": compatible_auc - full_auc,
            })

    summary_rows = summarize_domain_metrics(domain_rows, methods=methods)
    _write_csv(output / "mmr_compatible_sample_scores.csv", sample_rows)
    _write_csv(output / "mmr_compatible_domain_metrics.csv", domain_rows)
    _write_csv(output / "mmr_compatible_summary.csv", summary_rows)
    report = _report(domain_rows, summary_rows)
    objective_path = output / "mmr_compatible_objective_report.md"
    objective_path.write_text(report, encoding="utf-8")
    report_path = Path(args.report_path).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path != objective_path:
        report_path.write_text(report, encoding="utf-8")
    print(f"MMR_COMPATIBLE_EVAL_COMPLETE={output}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AeBAD-S MMR-compatible I-AUROC post-evaluation")
    parser.add_argument("--aebad-root", required=True)
    parser.add_argument("--baseline-output", required=True)
    parser.add_argument("--broad6-output", required=True)
    parser.add_argument("--phys4-output")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-path", required=True)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
