"""CSV ranking and immutable dataset access, without PyTorch."""
from __future__ import annotations

import csv
import hashlib
import io
import math
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image

from DINOv3.MADEqual.aebad_s_full_p0.dataset import IMAGE_SUFFIXES, find_aebad_s_root

PROTOCOL_ID = "mad_equal_aebad_s_target_fewshot_seed42_v1"
CACHE_REVISION = "fewshot_raw_v1"
SOURCE = "MAD_EQUAL"
UNION = "TARGET_MATCHED_K8_MAD"
METHODS = (SOURCE, UNION)
FOCUS_PATHS = (
    "test/fracture/background/IMG_9385.png",
    "test/ablation/background/IMG_8863.png",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return  # Unavailable evidence does not get an empty placeholder CSV.
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def snapshot(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            result[path.relative_to(root).as_posix()] = digest.hexdigest()
    return result


def relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or ":" in value:
        raise ValueError(f"unsafe relative path: {value}")
    return path.as_posix()


class Dataset:
    """Read selected images directly from the zip; never extract the dataset."""

    def __init__(self, path: Path):
        self.zip = zipfile.ZipFile(path) if path.is_file() else None
        if self.zip:
            names = self.zip.namelist()
            roots = [prefix for prefix in ("AeBAD/AeBAD_S/", "AeBAD_S/", "")
                     if any(name.startswith(prefix + "test/") for name in names)]
            if len(roots) != 1:
                self.zip.close()
                raise ValueError("zip must contain one AeBAD-S test root")
            self.prefix = roots[0]
            self.names = {name[len(self.prefix):] for name in names
                          if name.startswith(self.prefix) and not name.endswith("/")}
        else:
            self.root = find_aebad_s_root(path)
            self.names = {p.relative_to(self.root).as_posix() for folder in ("test", "ground_truth")
                          for p in (self.root / folder).rglob("*") if p.is_file()}
        test_paths = sorted(name for name in self.names
                            if name.startswith("test/") and PurePosixPath(name).suffix.lower() in IMAGE_SUFFIXES
                            and not PurePosixPath(name).name.startswith("._") and "__MACOSX" not in name)
        # Same full-test lexical indexing as discover_phase_a; no subset renumbering.
        self.ids = {name: f"test_{index:04d}" for index, name in enumerate(test_paths)}

    def image(self, relative: str, mode: str = "RGB") -> Image.Image:
        name = relative_path(relative)
        if self.zip:
            with self.zip.open(self.prefix + name) as handle:
                with Image.open(io.BytesIO(handle.read())) as image:
                    return image.convert(mode)
        with Image.open(self.root / name) as image:
            return image.convert(mode)

    def close(self) -> None:
        if self.zip:
            self.zip.close()


def gt_path(path: str) -> str:
    parts = PurePosixPath(path).parts
    return f"ground_truth/{parts[1]}/{parts[2]}/{PurePosixPath(path).stem}.png"


def validate_inputs(rows: list[dict[str, str]], support_rows: list[dict[str, str]],
                    dataset: Dataset) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    supports = sorted((row for row in support_rows
                       if row["branch"] == "target_matched" and row["domain"] == "background"),
                      key=lambda row: int(row["support_index"]))
    if [int(row["support_index"]) for row in supports] != list(range(1, 9)):
        raise ValueError("background support must contain ordered indices 1..8")
    support_paths = {relative_path(row["relative_path"]) for row in supports}
    if len(support_paths) != 8:
        raise ValueError("duplicate support image")
    for row in supports:
        if int(row["label"]) != 0 or not row["relative_path"].startswith("test/good/background/"):
            raise ValueError("support is not a background normal")
        for k in (1, 2, 4, 8):
            if int(row[f"included_in_K{k}"]) != int(int(row["support_index"]) <= k):
                raise ValueError("support manifest prefixes are not nested")
        if row["relative_path"] not in dataset.ids:
            raise ValueError("support image missing from dataset")
    by_method: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        if row["domain"] != "background":
            continue
        path = relative_path(row["relative_path"])
        label = int(row["label"])
        parts = PurePosixPath(path).parts
        if (len(parts) != 4 or parts[0] != "test" or parts[2] != "background"
                or label not in (0, 1) or label != int(parts[1] != "good")):
            raise ValueError(f"invalid label/path: {path}")
        if path in support_paths or dataset.ids.get(path) != row["sample_id"]:
            raise ValueError(f"support leakage or sample_id/path mismatch: {path}")
        if not math.isfinite(float(row["image_score"])):
            raise ValueError("non-finite CSV score")
        method_rows = by_method.setdefault(row["method"], {})
        if path in method_rows:
            raise ValueError("duplicate method/path in sample_scores.csv")
        method_rows[path] = row
    if not set(METHODS).issubset(by_method):
        raise ValueError("required baseline methods missing")
    expected_paths = {path for path in dataset.ids if "/background/" in path} - support_paths
    if any(set(values) != expected_paths for values in by_method.values()):
        raise ValueError("methods do not share the complete background held-out set")
    samples = []
    for path in sorted(expected_paths):
        original = by_method[SOURCE][path]
        identity = (original["sample_id"], original["label"])
        if any((values[path]["sample_id"], values[path]["label"]) != identity
               for values in by_method.values()):
            raise ValueError("method sample identities do not agree")
        sample = {"sample_id": original["sample_id"], "relative_path": path,
                  "label": int(original["label"]),
                  "source_score": float(original["image_score"]),
                  "union_score": float(by_method[UNION][path]["image_score"])}
        if sample["union_score"] > sample["source_score"] + 1e-5:
            raise ValueError("union image score exceeds Source; check protocol")
        samples.append(sample)
    if Counter(sample["label"] for sample in samples) != {0: 85, 1: 212}:
        raise ValueError("background must have 85 normals and 212 anomalies")
    return samples, supports


def pair_error(normal: float, anomaly: float) -> float:
    return 1.0 if normal > anomaly else (0.5 if normal == anomaly else 0.0)


def ranking(samples: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    normals = [sample for sample in samples if sample["label"] == 0]
    anomalies = [sample for sample in samples if sample["label"] == 1]
    pairs = []
    totals = {sample["sample_id"]: {"source_errors": 0.0, "union_errors": 0.0,
                                      "source_ties": 0, "union_ties": 0} for sample in samples}
    for normal in normals:
        for anomaly in anomalies:
            old = pair_error(normal["source_score"], anomaly["source_score"])
            new = pair_error(normal["union_score"], anomaly["union_score"])
            pairs.append({"normal_id": normal["sample_id"], "normal_path": normal["relative_path"],
                          "anomaly_id": anomaly["sample_id"], "anomaly_path": anomaly["relative_path"],
                          "source_error": old, "union_error": new, "error_change": new - old,
                          "source_gap_anomaly_minus_normal": anomaly["source_score"] - normal["source_score"],
                          "union_gap_anomaly_minus_normal": anomaly["union_score"] - normal["union_score"]})
            for sample in (normal, anomaly):
                for prefix, error in (("source", old), ("union", new)):
                    totals[sample["sample_id"]][prefix + "_errors"] += error
                    totals[sample["sample_id"]][prefix + "_ties"] += int(error == 0.5)
    per_image = [dict(sample, **totals[sample["sample_id"]],
                      error_change=totals[sample["sample_id"]]["union_errors"] - totals[sample["sample_id"]]["source_errors"],
                      score_drop=sample["source_score"] - sample["union_score"]) for sample in samples]
    hard = sorted((sample for sample in per_image if sample["label"] == 0),
                  key=lambda sample: (-sample["union_score"], sample["relative_path"]))[:10]
    union_error = sum(pair["union_error"] for pair in pairs)
    summary = {"normal_count": len(normals), "anomaly_count": len(anomalies), "pair_count": len(pairs),
               "tie_error_weight": 0.5,
               "improved_pairs": sum(pair["error_change"] < 0 for pair in pairs),
               "worsened_pairs": sum(pair["error_change"] > 0 for pair in pairs),
               "unchanged_pairs": sum(pair["error_change"] == 0 for pair in pairs),
               "top10_remaining_error_fraction": sum(sample["union_errors"] for sample in hard) / union_error if union_error else None}
    for prefix in ("source", "union"):
        errors = sum(pair[prefix + "_error"] for pair in pairs)
        summary.update({prefix + "_weighted_errors": errors,
                        prefix + "_strict_reversals": sum(pair[prefix + "_error"] == 1 for pair in pairs),
                        prefix + "_ties": sum(pair[prefix + "_error"] == 0.5 for pair in pairs),
                        prefix + "_image_AUROC": 1 - errors / len(pairs)})
    summary["top10"] = [sample["sample_id"] for sample in hard]
    return summary, pairs, per_image
