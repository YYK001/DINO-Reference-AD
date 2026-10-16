"""Category-equal summary for the frozen three-category RobustAD few-shot P0."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


CATEGORIES = ("MetalParts", "PCB", "PiledBags")
METRICS = ("image_AUROC", "image_AUPR", "pixel_AUROC", "pixel_AUPR", "AUPRO")
PIXEL_CATEGORIES = {"MetalParts", "PCB"}
METHODS = (
    "OFFICIAL", "MAD_EQUAL",
    "SOURCE_EXTRA_K1_MAD", "SOURCE_EXTRA_K2_MAD",
    "SOURCE_EXTRA_K4_MAD", "SOURCE_EXTRA_K8_MAD",
    "TARGET_MATCHED_K1_MAD", "TARGET_MATCHED_K2_MAD",
    "TARGET_MATCHED_K4_MAD", "TARGET_MATCHED_K8_MAD",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = [dict(row) for row in rows]
    if not values:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in values for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _number(row: Mapping[str, Any], field: str) -> float | None:
    value = row.get(field)
    if value in (None, "", "NA", "N/A"):
        return None
    return float(value)


def load_category_macro(result_dir: Path, expected_category: str) -> list[dict[str, Any]]:
    complete_path = result_dir / "complete.json"
    macro_path = result_dir / "target_shift_macro.csv"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if complete.get("status") != "complete" or complete.get("category") != expected_category:
        raise RuntimeError(f"incomplete or wrong category result: {result_dir}")
    rows = _read_csv(macro_path)
    if {row.get("method") for row in rows} != set(METHODS):
        raise RuntimeError(f"method set mismatch: {macro_path}")
    output: list[dict[str, Any]] = []
    for row in rows:
        value: dict[str, Any] = {"category": expected_category, **row}
        for metric in METRICS:
            number = _number(row, metric)
            ard = _number(row, f"ARD_{metric}")
            value[metric] = "" if number is None else number
            value[f"ARD_{metric}"] = "" if ard is None else ard
        output.append(value)
    return output


def aggregate_category_macros(
    category_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    if set(category_rows) != set(CATEGORIES):
        raise RuntimeError("three-category input must contain MetalParts, PCB and PiledBags")
    indexed = {
        category: {str(row["method"]): row for row in rows}
        for category, rows in category_rows.items()
    }
    output: list[dict[str, Any]] = []
    for method in METHODS:
        row: dict[str, Any] = {"method": method, "scope": "three_category_equal_macro"}
        for metric in METRICS:
            eligible = CATEGORIES if metric.startswith("image_") else tuple(
                category for category in CATEGORIES if category in PIXEL_CATEGORIES
            )
            values = [
                value for category in eligible
                if (value := _number(indexed[category][method], metric)) is not None
            ]
            ards = [
                value for category in eligible
                if (value := _number(indexed[category][method], f"ARD_{metric}")) is not None
            ]
            if len(values) != len(eligible) or len(ards) != len(eligible):
                raise RuntimeError(f"category support mismatch for {method}/{metric}")
            row[metric] = mean(values)
            row[f"ARD_{metric}"] = mean(ards)
            row[f"{metric}_category_count"] = len(eligible)
            row[f"{metric}_categories"] = ";".join(eligible)
        output.append(row)
    return output


def method_differences(macros: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    indexed = {str(row["method"]): row for row in macros}
    output: list[dict[str, Any]] = []
    for shot in (1, 2, 4, 8):
        target = indexed[f"TARGET_MATCHED_K{shot}_MAD"]
        source = indexed[f"SOURCE_EXTRA_K{shot}_MAD"]
        baseline = indexed["MAD_EQUAL"]
        row: dict[str, Any] = {"method": target["method"], "K": shot}
        for metric in METRICS:
            row[f"delta_vs_MAD_EQUAL_{metric}"] = float(target[metric]) - float(baseline[metric])
            row[f"delta_vs_SOURCE_EXTRA_K{shot}_{metric}"] = float(target[metric]) - float(source[metric])
        output.append(row)
    return output


def _report(
    category_rows: Sequence[Mapping[str, Any]],
    macros: Sequence[Mapping[str, Any]],
    differences: Sequence[Mapping[str, Any]],
    *,
    per_shift_differences: Mapping[str, Sequence[Mapping[str, Any]]],
    mechanism_macros: Mapping[str, Sequence[Mapping[str, Any]]],
) -> str:
    lines = [
        "# RobustAD Target-normal Few-shot: three-category objective summary",
        "",
        "This report is a category-equal aggregation of one Seed42 support pool per category. It does not select a best K and makes no statistical-significance claim.",
        "",
        "MetalParts legacy hit-rate fields are patch-pooled and are not used in this summary (`legacy_patch_pooled_not_used`).",
        "",
        "## Category target-shift macros",
        "",
        "| category | method | Image AUROC | Image AUPR | Pixel AUROC | Pixel AUPR | AUPRO |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in category_rows:
        shown = lambda name: "NA" if row[name] in ("", None) else f"{float(row[name]):.6f}"
        lines.append(f"| {row['category']} | {row['method']} | {shown('image_AUROC')} | {shown('image_AUPR')} | {shown('pixel_AUROC')} | {shown('pixel_AUPR')} | {shown('AUPRO')} |")
    lines.extend((
        "", "## Category-equal RobustAD macros", "",
        "Image metrics average MetalParts, PCB and PiledBags. Pixel metrics and AUPRO average MetalParts and PCB only.", "",
        "| method | Image AUROC | Image AUPR | Pixel AUROC | Pixel AUPR | AUPRO |",
        "|---|---:|---:|---:|---:|---:|",
    ))
    for row in macros:
        lines.append(f"| {row['method']} | {float(row['image_AUROC']):.6f} | {float(row['image_AUPR']):.6f} | {float(row['pixel_AUROC']):.6f} | {float(row['pixel_AUPR']):.6f} | {float(row['AUPRO']):.6f} |")
    lines.extend((
        "", "## Frozen-K differences", "",
        "| K | d Image AUROC vs MAD | d Image AUROC vs source-extra | d AUPRO vs MAD | d AUPRO vs source-extra |",
        "|---:|---:|---:|---:|---:|",
    ))
    for row in differences:
        shot = int(row["K"])
        lines.append(
            f"| {shot} | {float(row['delta_vs_MAD_EQUAL_image_AUROC']):+.6f} | "
            f"{float(row[f'delta_vs_SOURCE_EXTRA_K{shot}_image_AUROC']):+.6f} | "
            f"{float(row['delta_vs_MAD_EQUAL_AUPRO']):+.6f} | "
            f"{float(row[f'delta_vs_SOURCE_EXTRA_K{shot}_AUPRO']):+.6f} |"
        )
    lines.extend(("", "## Per-shift target-matched differences", "", "This table records whether gains are concentrated in particular shifts and compares every frozen K with both MAD_EQUAL and its equal-capacity source-extra control.", "", "| category | shift | K | d Image AUROC vs MAD | d Image AUROC vs source-extra |", "|---|---|---:|---:|---:|"))
    for category in CATEGORIES:
        for row in per_shift_differences[category]:
            method = str(row.get("method", ""))
            scope = str(row.get("shift_or_scope", ""))
            if not method.startswith("TARGET_MATCHED_") or scope.endswith("macro"):
                continue
            shot = int(row["K"])
            lines.append(
                f"| {category} | {scope} | {shot} | {float(row['delta_vs_MAD_EQUAL_image_AUROC']):+.6f} | "
                f"{float(row[f'delta_vs_SOURCE_EXTRA_K{shot}_image_AUROC']):+.6f} |"
            )
    lines.extend(("", "## PCB signed mechanism", "", "R_N, defect-only R_A, G, and defect-hit rates below use the corrected image-equal aggregation.", "", "| method | K | layer | R_N | R_A | G | defect hit |", "|---|---:|---:|---:|---:|---:|---:|"))
    for row in mechanism_macros["PCB"]:
        lines.append(f"| {row['method']} | {row['K']} | {row['layer']} | {float(row['R_N']):.6f} | {float(row['R_A']):.6f} | {float(row['G']):.6f} | {float(row['defect_hit_rate']):.6f} |")
    lines.extend(("", "## PiledBags normal-side mechanism", "", "PiledBags has no pixel GT; only held-out-normal R_N and image-equal normal hit rate are reported.", "", "| method | K | layer | R_N | normal hit |", "|---|---:|---:|---:|---:|"))
    for row in mechanism_macros["PiledBags"]:
        lines.append(f"| {row['method']} | {row['K']} | {row['layer']} | {float(row['R_N']):.6f} | {float(row['normal_hit_rate']):.6f} |")
    lines.extend((
        "", "## Evidence limits", "",
        "The category, frozen-K, per-shift and mechanism tables above provide the numerical record for the ten protocol questions: PCB/PiledBags versus MAD_EQUAL and source-extra; K1 versus later prefixes; shift concentration; PCB R_N versus R_A and any defect-hit/G<0 layers; PiledBags image changes; three-category image macros; two-category pixel/AUPRO macros; and evidence scope.",
        "",
        "Current evidence remains Seed42 with one deterministic support pool per category. No automatic best-K, innovation, or significance conclusion is produced.",
    ))
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> None:
    inputs = {
        "MetalParts": args.metalparts.resolve(),
        "PCB": args.pcb.resolve(),
        "PiledBags": args.piledbags.resolve(),
    }
    grouped = {category: load_category_macro(path, category) for category, path in inputs.items()}
    per_shift = {category: _read_csv(path / "method_differences.csv") for category, path in inputs.items()}
    mechanisms = {
        category: _read_csv(path / "mechanism_macro.csv")
        for category, path in inputs.items() if category != "MetalParts"
    }
    category_output = [dict(row) for category in CATEGORIES for row in grouped[category]]
    macros = aggregate_category_macros(grouped)
    differences = method_differences(macros)
    output = args.output_dir.resolve()
    _write_csv(output / "category_target_macros.csv", category_output)
    _write_csv(output / "robustad_method_macros.csv", macros)
    _write_csv(output / "robustad_method_differences.csv", differences)
    (output / "three_category_objective_report.md").write_text(
        _report(
            category_output, macros, differences,
            per_shift_differences=per_shift,
            mechanism_macros=mechanisms,
        ), encoding="utf-8",
    )
    _atomic_json(output / "complete.json", {
        "status": "complete",
        "scientific_status": "SCIENTIFIC_REVIEW_REQUIRED",
        "dataset": "RobustAD",
        "categories": list(CATEGORIES),
        "seed": 42,
        "K": [1, 2, 4, 8],
        "aggregation": "target_shift_equal_then_category_equal",
        "image_metric_category_count": 3,
        "pixel_metric_category_count": 2,
        "metalparts_hit_rate": "legacy_patch_pooled_not_used",
        "inputs": {category: str(path) for category, path in inputs.items()},
    })


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metalparts", type=Path, required=True)
    parser.add_argument("--pcb", type=Path, required=True)
    parser.add_argument("--piledbags", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
