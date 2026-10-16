"""Preflight or explicitly replay exactly 12 frozen AeBAD-S/background cases.

The help/preflight paths never import PyTorch. No memory builder, MAD fitter,
download, full runner or GT loader is called by this module.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import sys
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROTOCOL_ID = "mad_equal_aebad_s_target_fewshot_seed42_v1"
CACHE_REVISION = "fewshot_raw_v1"
CHECKPOINT_SHA256 = "7c1da9a54b3bdb333f5ebc42e404b7f19b1b5bed504877623c9dc87397f41488"
LAYERS = (7, 15, 23, 31)
SHOTS = (1, 2, 4, 8)
METHODS = ("MAD_EQUAL", "TARGET_MATCHED_K8_MAD")
CASES = (
    ("test_0977", "test/good/background/IMG_8147.png"),
    ("test_0910", "test/good/background/IMG_7753.png"),
    ("test_0887", "test/good/background/IMG_7728.png"),
    ("test_0924", "test/good/background/IMG_7768.png"),
    ("test_0921", "test/good/background/IMG_7765.png"),
    ("test_0930", "test/good/background/IMG_7775.png"),
    ("test_0893", "test/good/background/IMG_7734.png"),
    ("test_0926", "test/good/background/IMG_7770.png"),
    ("test_0934", "test/good/background/IMG_7780.png"),
    ("test_0935", "test/good/background/IMG_7781.png"),
    ("test_0023", "test/ablation/background/IMG_8863.png"),
    ("test_0545", "test/fracture/background/IMG_9385.png"),
)
TARGET_NAMES = ("IMG_7785.png", "IMG_7764.png", "IMG_7771.png", "IMG_8149.png",
                "IMG_7778.png", "IMG_7752.png", "IMG_7799.png", "IMG_7747.png")
RAW_ARRAY_NAMES = ("official_raw", *(f"{branch}_{kind}_{k}" for branch in ("source", "target")
                                    for kind in ("raw", "support") for k in SHOTS))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def safe_relative(value: str) -> str:
    parts = value.split("/")
    if value.startswith("/") or "\\" in value or ":" in value or any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"unsafe relative path: {value}")
    return value


def select_cases(rows: list[dict[str, str]], selected: list[dict[str, str]] | None = None) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, dict[str, str]]] = {method: {} for method in METHODS}
    for row in rows:
        if row["domain"] != "background" or row["method"] not in METHODS:
            continue
        path = safe_relative(row["relative_path"])
        group = groups[row["method"]]
        if path in group or not math.isfinite(float(row["image_score"])):
            raise ValueError("duplicate/non-finite background sample score")
        if len(path.split("/")) != 4 or path.split("/")[2] != "background" or int(row["label"]) != int("/good/" not in path):
            raise ValueError("background CSV label/path mismatch")
        group[path] = row
    base, union = (groups[method] for method in METHODS)
    if set(base) != set(union) or len(base) != 297 or sum(int(row["label"]) == 0 for row in base.values()) != 85:
        raise ValueError("CSV must contain the original common 85/212 background held-out set")
    for path in base:
        if (base[path]["sample_id"], base[path]["label"]) != (union[path]["sample_id"], union[path]["label"]):
            raise ValueError("Source/union sample identity mismatch")
    top10 = sorted((row for row in union.values() if int(row["label"]) == 0),
                   key=lambda row: (-float(row["image_score"]), row["relative_path"]))[:10]
    if [(row["sample_id"], row["relative_path"]) for row in top10] != list(CASES[:10]):
        raise ValueError("CSV top10 differs from the fixed replay12 protocol; no automatic reselection")
    if selected is not None:
        chosen = [row for row in selected if set(row.get("selection_reason", "").split(";")) & {"top10_union_normal", "focus_anomaly"}]
        if len(chosen) != 12 or {(row["sample_id"], row["relative_path"]) for row in chosen} != set(CASES):
            raise ValueError("selected_cases.csv does not resolve to exactly the fixed 12 cases")
    cases = []
    for index, (sample_id, path) in enumerate(CASES):
        if path not in base or base[path]["sample_id"] != sample_id:
            raise ValueError(f"fixed sample_id/path not found in original CSV: {sample_id}")
        cases.append({"sample_id": sample_id, "relative_path": path, "label": int(index >= 10),
                      "selection_reason": "top10_union_normal" if index < 10 else "focus_anomaly",
                      "source_score": float(base[path]["image_score"]), "union_score": float(union[path]["image_score"])})
    return cases


def select_support(rows: list[dict[str, str]], cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for branch, domain, prefix in (("source_extra", "source", "source_extra"),
                                    ("target_matched", "background", "target_matched/background")):
        pool = sorted((row for row in rows if row["branch"] == branch and row["domain"] == domain),
                      key=lambda row: int(row["support_index"]))
        if [int(row["support_index"]) for row in pool] != list(range(1, 9)):
            raise ValueError(f"support indices must be 1..8: {branch}")
        for row in pool:
            path = safe_relative(row["relative_path"])
            if int(row["label"]) != 0 or not path.startswith("train/good/" if branch == "source_extra" else "test/good/background/"):
                raise ValueError("support must be the original normal-only pool")
            if any(int(row[f"included_in_K{k}"]) != int(int(row["support_index"]) <= k) for k in SHOTS):
                raise ValueError("support prefixes are not nested")
            output.append(dict(row, cache_relative=f"{prefix}/{int(row['support_index']):02d}.npz"))
        if branch == "target_matched" and [Path(row["relative_path"]).name for row in pool] != list(TARGET_NAMES):
            raise ValueError("Target K8 manifest order differs from the fixed protocol")
    paths = {row["relative_path"] for row in output}
    if len(paths) != 16 or paths & {row["relative_path"] for row in cases}:
        raise ValueError("duplicate support or support/query overlap")
    return output


class DataIndex:
    """Filename/header checks for 28 images; never open GT or extract a zip."""

    def __init__(self, supplied: Path):
        self.zip = zipfile.ZipFile(supplied) if supplied.is_file() else None
        if self.zip:
            names = self.zip.namelist()
            prefixes = [prefix for prefix in ("AeBAD/AeBAD_S/", "AeBAD_S/", "")
                        if any(name.startswith(prefix + "test/") for name in names)]
            if len(prefixes) != 1:
                self.zip.close()
                raise ValueError("could not resolve a unique AeBAD-S zip root")
            self.prefix = prefixes[0]
            self.names = {name[len(self.prefix):] for name in names if name.startswith(self.prefix) and not name.endswith("/")}
            self.root = supplied
        else:
            roots = [supplied, supplied / "AeBAD_S", supplied / "AeBAD" / "AeBAD_S"]
            self.root = next((root for root in roots if (root / "train/good").is_dir() and (root / "test").is_dir()), supplied)
            self.names = {path.relative_to(self.root).as_posix() for folder in ("train/good", "test")
                          for path in (self.root / folder).rglob("*") if path.is_file()}
        images = sorted(name for name in self.names if name.startswith("test/") and self.is_image(name))
        # Never assign new IDs to a partial file set; the original CSV remains
        # authoritative. A complete filename index permits an extra cross-check.
        self.test_count = len(images)
        self.ids = {path: f"test_{index:04d}" for index, path in enumerate(images)} if len(images) == 1639 else {}
        self.source_count = sum(name.startswith("train/good/") and self.is_image(name) for name in self.names)

    @staticmethod
    def is_image(name: str) -> bool:
        return Path(name).suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"} and not Path(name).name.startswith("._") and "__MACOSX" not in name

    def describe(self, relative: str) -> dict[str, Any]:
        from PIL import Image
        if relative not in self.names:
            raise FileNotFoundError(relative)
        path = self.root / relative
        handle = self.zip.open(self.prefix + relative) if self.zip else path.open("rb")
        with handle:
            with Image.open(handle) as image:
                width, height = image.size
        if self.zip:
            with self.zip.open(self.prefix + relative) as handle:
                digest = hashlib.sha256()
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
                checksum = digest.hexdigest()
        else:
            checksum = sha256(path)
        return {"relative_path": relative, "height": height, "width": width, "sha256": checksum}

    def close(self) -> None:
        if self.zip:
            self.zip.close()


def check_mad(path: Path) -> None:
    import numpy as np
    with np.load(path, allow_pickle=False) as payload:
        if str(payload["protocol_id"].item()) != PROTOCOL_ID:
            raise ValueError("Source MAD protocol mismatch")
        medians, scales = (payload[key] for key in ("medians", "scales"))
        if medians.shape != (4,) or scales.shape != (4,) or not np.isfinite(medians).all() or not np.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError("Source MAD must be finite with four positive scales")


def check_support_header(path: Path, relative: str) -> None:
    """Validate small metadata and feature headers without materializing huge banks."""
    import numpy as np
    with np.load(path, allow_pickle=False) as payload:
        if str(payload["protocol_id"].item()) != PROTOCOL_ID or str(payload["relative_path"].item()) != relative:
            raise ValueError("support cache protocol/path mismatch")
        grid = payload["grid"]
        if grid.shape != (2,) or not np.issubdtype(grid.dtype, np.integer) or (grid <= 0).any():
            raise ValueError("invalid support grid")
    with zipfile.ZipFile(path) as archive:
        for layer in LAYERS:
            with archive.open(f"layer_{layer}.npy") as handle:
                version = np.lib.format.read_magic(handle)
                if version == (1, 0):
                    shape, _fortran, dtype = np.lib.format.read_array_header_1_0(handle)
                elif version == (2, 0):
                    shape, _fortran, dtype = np.lib.format.read_array_header_2_0(handle)
                else:
                    raise ValueError("unsupported support array header version")
            if shape != (int(grid[0]) * int(grid[1]), 1280) or dtype != np.dtype("float32"):
                raise ValueError(f"support L{layer} shape/dtype mismatch")


def resolved_paths(args: argparse.Namespace) -> dict[str, Path]:
    project = Path(args.project_root).resolve()
    results = Path(args.results_dir or project / "DINOv3/outputs/mad_equal/aebad_s_target_fewshot_seed42_v1").resolve()
    official = Path(args.official_root or project / "external_baselines/SuperADD").resolve()
    return {"project": project, "results": results, "official": official,
            "data": Path(args.aebad_root).resolve(), "dinov3": Path(args.dinov3_root).resolve(),
            "checkpoint": Path(args.checkpoint or official / "weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth").resolve(),
            "memory": Path(args.memory or results / "official_memory.pt").resolve(),
            "mad": Path(args.mad or results / "source_mad_state.npz").resolve(),
            "support": Path(args.support_cache_dir or results / "support_features").resolve(),
            "output": Path(args.output_dir or project / "DINOv3/outputs/mad_equal/aebad_s_union_background_replay12_seed42_v1").resolve()}


def guard_output(paths: dict[str, Path]) -> None:
    output = paths["output"]
    for key in ("results", "data", "official", "dinov3", "checkpoint", "memory", "mad", "support"):
        root = paths[key]
        if output == root or output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError(f"replay output overlaps protected {key}: {root}")


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    paths = resolved_paths(args)
    guard_output(paths)
    files, errors, cases, supports, images = [], [], [], [], []
    required = ([(name, paths[name]) for name in ("checkpoint", "memory", "mad")]
            + [(name, paths["results"] / name) for name in ("sample_scores.csv", "support_manifest.csv", "memory_manifest.json", "complete.json")]
            + [("official_config", paths["official"] / "config.json"), ("dinov3_hub", paths["dinov3"] / "hubconf.py"),
               ("official_model", paths["official"] / "tracks/industrial/src/industrial/model.py")]
            + [("code_" + name, paths["project"] / relative) for name, relative in (
                ("runner", "DINOv3/MADEqual/aebad_s_target_fewshot/runner.py"),
                ("scoring", "DINOv3/MADEqual/robustad_target_fewshot_p0/scoring.py"),
                ("support", "DINOv3/MADEqual/robustad_target_fewshot_p0/support.py"),
                ("host", "DINOv3/relation_reliability/superadd_host.py"),
                ("replay", "DINOv3/MADEqual/aebad_s_target_fewshot/replay12.py"))])
    for key, path in required:
        row = {"kind": key, "path": str(path), "required": True, "exists": path.is_file()}
        if row["exists"]:
            row.update(sha256=sha256(path), bytes=path.stat().st_size)
        else:
            errors.append(f"missing required {key}: {path}")
        files.append(row)
    available = {row["kind"] for row in files if row["exists"]}
    try:
        if "checkpoint" in available and next(row["sha256"] for row in files if row["kind"] == "checkpoint") != CHECKPOINT_SHA256:
            errors.append("checkpoint SHA256 differs from original approved ViT-H+/16 weights")
        if "mad" in available:
            check_mad(paths["mad"])
            archived_mad = paths["results"] / "source_mad_state.npz"
            if archived_mad.is_file() and sha256(archived_mad) != next(row["sha256"] for row in files if row["kind"] == "mad"):
                errors.append("specified MAD differs from the original formal MAD file")
        for name in ("complete.json", "memory_manifest.json"):
            if name in available:
                value = json.loads((paths["results"] / name).read_text(encoding="utf-8"))
                if value.get("protocol_id") != PROTOCOL_ID:
                    raise ValueError(f"protocol mismatch in {name}")
                count_key = "source_count" if name == "complete.json" else "builder_input_count"
                if value.get(count_key) != 521:
                    raise ValueError(f"{name}: original memory builder/source_count must be 521")
                if name == "complete.json" and value.get("status") != "complete":
                    raise ValueError("formal completion marker is not complete")
        if "official_config" in available:
            from DINOv3.relation_reliability.protocol import validate_superadd_config
            config = json.loads((paths["official"] / "config.json").read_text(encoding="utf-8"))
            validate_superadd_config(config)
            if "memory_manifest.json" in available:
                original = json.loads((paths["results"] / "memory_manifest.json").read_text(encoding="utf-8"))["host_config"]
                keys = ("backbone", "layers", "patch_size", "patch_overlap", "max_database_size", "threshold_fraction", "evaluation_downscale", "brightness_augmentation")
                if any(config.get(key) != original.get(key) for key in keys):
                    raise ValueError("current host config differs from original preprocessing/bank geometry")
    except (ValueError, KeyError, OSError, ImportError, RuntimeError) as error:
        errors.append(str(error))
    try:
        if {"sample_scores.csv", "support_manifest.csv"}.issubset(available):
            selected = read_csv(Path(args.cases_csv)) if args.cases_csv else None
            cases = select_cases(read_csv(paths["results"] / "sample_scores.csv"), selected)
            supports = select_support(read_csv(paths["results"] / "support_manifest.csv"), cases)
    except (ValueError, KeyError, OSError, ImportError, RuntimeError) as error:
        errors.append(str(error))
    index = None
    dataset_index_status = "unavailable"
    try:
        index = DataIndex(paths["data"])
        paths["data"] = index.root.resolve()
        dataset_index_status = "full_1639_test_filename_index_checked" if len(index.ids) == 1639 else "original_csv_ids_preserved_partial_files_not_renumbered"
        if len(index.ids) == 1639:
            for case in cases:
                if index.ids.get(case["relative_path"]) != case["sample_id"]:
                    errors.append(f"dataset full-index sample_id mismatch: {case['sample_id']}")
        for relative in [row["relative_path"] for row in cases + supports]:
            try:
                images.append(index.describe(relative))
            except (OSError, ValueError) as error:
                errors.append(f"required image {relative}: {error}")
    except (OSError, ValueError, zipfile.BadZipFile, ImportError) as error:
        errors.append(f"dataset check: {error}")
    finally:
        if index is not None:
            index.close()
    support_status = []
    for row in supports:
        old, new = paths["support"] / row["cache_relative"], paths["output"] / "support_features" / row["cache_relative"]
        status = dict(row, original_cache=str(old), replay_cache=str(new), status="can_extract_frozen_at_replay")
        for kind, path in (("original", old), ("replay", new)):
            if path.is_file():
                try:
                    check_support_header(path, row["relative_path"])
                    status.update({kind + "_sha256": sha256(path), kind + "_header_valid": True})
                    status["status"] = "existing_replay_cache" if kind == "replay" else "original_cache_available"
                except (OSError, KeyError, ValueError, ImportError, zipfile.BadZipFile) as error:
                    errors.append(f"invalid {kind} support cache {path}: {error}")
        support_status.append(status)
    if args.mode == "replay" and paths["data"].is_file():
        errors.append("replay requires an extracted AeBAD-S directory; zip is accepted only for preflight")
    return {"tool": "aebad_s_background_replay12", "protocol_id": PROTOCOL_ID, "cache_revision": CACHE_REVISION,
            "status": "filesystem_preflight_passed" if not errors else "blocked",
            "resolved_paths": {name: str(path) for name, path in paths.items()}, "required_files": files,
            "cases": cases, "supports": support_status, "images": images, "errors": errors,
            "dataset_index_status": dataset_index_status, "source_memory_builder_count": 521,
            "memory_payload_validation": "deferred_to_original_load_memory_in_replay_no_torch_in_preflight",
            "support_finite_validation": "deferred_to_original_extract_or_load_support_in_replay",
            "backbone_loaded": False, "query_inference_started": False,
            "missing_memory_policy": "stop_no_rebuild_original_bank_required",
            "missing_mad_policy": "stop_no_fit_original_state_required"}


def validate_raw(path: Path, case: dict[str, Any]) -> tuple[dict[str, Any], tuple[int, int], float]:
    import numpy as np
    from DINOv3.MADEqual.aebad_s_union_diagnostics.spatial import load_cache
    with np.load(path, allow_pickle=False) as payload:
        if any(payload[name].dtype != np.dtype("float32") for name in RAW_ARRAY_NAMES):
            raise ValueError("original distance cache arrays must be float32")
    values, shape, error = load_cache(path, case["relative_path"])
    # Check every prefix, rather than filling K1/K2/K4 with K8 placeholders.
    for branch in ("source", "target"):
        previous = None
        for k in SHOTS:
            support = values[f"{branch}_support_{k}"]
            union = values[f"{branch}_raw_{k}"]
            difference = float(np.max(np.abs(union - np.minimum(values["official_raw"], support))))
            error = max(error, difference)
            if not np.array_equal(union, np.minimum(values["official_raw"], support)):
                raise ValueError(f"{branch} K{k} union differs from layer-wise min")
            if previous is not None and np.any(support > previous):
                raise ValueError(f"{branch} support prefix is not monotone at K{k}")
            previous = support
    return values, shape, error


def verify_case(path: Path, case: dict[str, Any], runner: Any, mad: Any,
                tolerance: float, expected_shape: tuple[int, int], expected_grid: tuple[int, int]) -> dict[str, Any]:
    values, shape, min_error = validate_raw(path, case)
    original_values, original_shape = runner._load_observation(path, relative=case["relative_path"])
    if shape != expected_shape or original_shape != shape or values["official_raw"].shape != (4, *expected_grid):
        raise ValueError("cache grid/output shape differs from original query geometry")
    scores = {method: float(runner._method_map(original_values, method, mad, shape).max()) for method in METHODS}
    source_error = abs(scores[METHODS[0]] - case["source_score"])
    union_error = abs(scores[METHODS[1]] - case["union_score"])
    return {"source_replay_score": scores[METHODS[0]], "union_replay_score": scores[METHODS[1]],
            "source_abs_error": source_error, "union_abs_error": union_error,
            "max_abs_error": max(source_error, union_error), "union_min_max_error": min_error,
            "scores_reproduced": max(source_error, union_error) <= tolerance,
            "cache_sha256": sha256(path)}


def input_identity(check: dict[str, Any]) -> dict[str, Any]:
    return {"protocol_id": PROTOCOL_ID, "cache_revision": CACHE_REVISION, "cases": check["cases"],
            "input_sha256": {row["kind"]: row["sha256"] for row in check["required_files"] if row["exists"]},
            "image_sha256": {row["relative_path"]: row["sha256"] for row in check["images"]},
            "supports": [{key: row[key] for key in ("branch", "domain", "support_index", "relative_path", "cache_relative")}
                         for row in check["supports"]]}


def check_recorded_cache(path: Path, row: dict[str, Any]) -> None:
    expected = row.get("cache_sha256")
    if expected and (not path.is_file() or sha256(path) != expected):
        raise ValueError(f"recorded replay cache changed or missing: {row['sample_id']}; preserve existing products")


def resume_case_records(output: Path) -> dict[str, dict[str, Any]]:
    """Check old hashes before replacing any replay state or importing inference."""
    status_path = output / "replay_status.json"
    if not status_path.is_file():
        return {}
    state = json.loads(status_path.read_text(encoding="utf-8"))
    expected, records = dict(CASES), {}
    for row in state.get("cases", []):
        sample_id = row.get("sample_id")
        if sample_id in records or sample_id not in expected or row.get("relative_path") != expected[sample_id]:
            raise ValueError("recorded replay case identity changed")
        if row.get("status") == "verified" and not row.get("cache_sha256"):
            raise ValueError(f"recorded verified replay cache has no checksum: {sample_id}")
        check_recorded_cache(output / "raw_cache/heldout/background" / f"{sample_id}.npz", row)
        records[sample_id] = row
    return records


def replay(args: argparse.Namespace, check: dict[str, Any]) -> int:
    if check["errors"]:
        return 2  # No inference imports, no fallback, even when caller bypasses main().
    paths = {key: Path(value) for key, value in check["resolved_paths"].items()}
    output, identity = paths["output"], input_identity(check)
    state_path = output / "replay_inputs.json"
    if state_path.is_file():
        if json.loads(state_path.read_text(encoding="utf-8"))["identity"] != identity:
            raise ValueError("resume inputs differ from the existing replay; use a new output directory")
    elif any(path.name != "preflight.json" for path in output.iterdir()):
        raise ValueError("refusing an unowned nonempty replay directory")
    prior_cases = resume_case_records(output)
    atomic_json(state_path, {"identity": identity, "preflight": check, "score_tolerance": args.score_tolerance})
    status_path = output / "replay_status.json"
    progress: dict[str, Any] = {"tool": "aebad_s_background_replay12", "protocol_id": PROTOCOL_ID,
                                "cache_revision": CACHE_REVISION, "status": "running", "case_ids": [case[0] for case in CASES],
                                "planned_queries": 12, "query_forward_calls_this_attempt": 0,
                                "cases": [prior_cases[sample_id] for sample_id, _ in CASES if sample_id in prior_cases],
                                "score_tolerance": args.score_tolerance, "precise_original_explanation_allowed": False}
    atomic_json(status_path, progress)
    try:
        import numpy as np
        import torch
        from PIL import Image
        from DINOv3.MADEqual.aebad_s_target_fewshot import runner
        from DINOv3.MADEqual.robustad_target_fewshot_p0.scoring import observe_fewshot_pil
        from DINOv3.MADEqual.robustad_target_fewshot_p0.support import extract_or_load_support
        from DINOv3.relation_reliability.datasets import Record
        from DINOv3.relation_reliability.superadd_host import construct_exact_host, install_import_paths, preprocessed_spatial_shape
        if (runner.PROTOCOL_ID, runner.CACHE_REVISION, runner.CHECKPOINT_SHA256) != (PROTOCOL_ID, CACHE_REVISION, CHECKPOINT_SHA256):
            raise ValueError("current original runner differs from the replay protocol")
        if tuple(runner.LAYERS) != LAYERS or tuple(runner.SHOT_COUNTS) != SHOTS:
            raise ValueError("original layer order or shot prefixes differ from replay12")
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("replay requires an available CUDA device; no CPU/model fallback")
        device = torch.device("cuda", device.index if device.index is not None else 0)
        torch.cuda.set_device(device)
        runner._seed_everything()
        software = {}
        for name in ("numpy", "torch", "torchvision", "Pillow", "scipy"):
            try:
                software[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                software[name] = None
        environment = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                       "software": software, "device": str(device), "device_name": torch.cuda.get_device_name(device),
                       "cuda": torch.version.cuda, "float32_matmul_precision": torch.get_float32_matmul_precision(),
                       "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                       "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}
        atomic_json(output / f"environment_{time.time_ns()}.json", environment)
        # Validate original artifacts before allocating/loading the backbone.
        # The original loader only assigns prototype_embeddings, so no model
        # construction is needed to check protocol/count/shape/finiteness.
        bank_holder = SimpleNamespace()
        if not runner._load_memory(paths["memory"], bank_holder, source_count=521, device=device):
            raise FileNotFoundError("original official_memory.pt missing; replay stops without rebuilding")
        mad = runner._load_mad(paths["mad"])
        if mad is None:
            raise FileNotFoundError("original Source MAD missing; replay stops without fitting")
        progress["memory_validation"] = {"source_count": 521, "loader": "original_runner._load_memory",
                                         "layers": {str(layer): {"shape": list(bank_holder.prototype_embeddings[layer].shape),
                                                                 "dtype": str(bank_holder.prototype_embeddings[layer].dtype)} for layer in LAYERS}}
        progress["mad_validation"] = {"loader": "original_runner._load_mad", "medians": mad.medians.tolist(), "scales": mad.scales.tolist()}
        install_import_paths(paths["project"], paths["official"], paths["dinov3"])
        from industrial import model as official_model
        from unittest.mock import patch
        if Path(official_model.__file__).resolve() != (paths["official"] / "tracks/industrial/src/industrial/model.py").resolve():
            raise ValueError("imported industrial.model is not from the specified official root")
        # The upstream helper otherwise globs config's weights directory. Bind
        # the CLI's SHA-checked local checkpoint for this construction only.
        def checkpoint_path(name: str) -> Path:
            if name != "dinov3_vith16plus":
                raise ValueError("unexpected backbone weight request")
            return paths["checkpoint"]
        with patch.object(official_model, "get_weights_path", checkpoint_path):
            model, config = construct_exact_host(paths["official"], paths["dinov3"], paths["checkpoint"], CHECKPOINT_SHA256)
        model.prototype_embeddings = bank_holder.prototype_embeddings
        del bank_holder
        model.backbone.dino.eval()
        model.backbone.dino.requires_grad_(False)
        if model.preprocessing.min_brightness_factor != 1 or model.preprocessing.max_brightness_factor != 1:
            raise ValueError("original query preprocessing must have factor=1")
        progress["query_protocol"] = {"brightness_factor": 1, "resize_factor": model.preprocessing.resize_factor,
                                      "patch_size": model.patch_exec.patch_size, "patch_overlap": model.patch_exec.patch_overlap,
                                      "model_patch_size": model.patch_exec.model_patch_size, "evaluation_downscale": model.evaluation_downscale,
                                      "dino_training": model.backbone.dino.training,
                                      "trainable_parameters": sum(parameter.requires_grad for parameter in model.backbone.dino.parameters())}
        atomic_json(status_path, progress)
        provenance = []
        prior_provenance = {row["relative_path"]: row for row in read_csv(output / "support_provenance.csv")} if (output / "support_provenance.csv").is_file() else {}
        feature_groups: dict[str, list[Any]] = {"source_extra": [], "target_matched": []}
        for row in check["supports"]:
            started = time.perf_counter()
            old, new = Path(row["original_cache"]), Path(row["replay_cache"])
            previous = prior_provenance.get(row["relative_path"], {})
            origin = previous.get("origin", "existing_replay_cache_origin_unrecorded")
            action = "load_existing_replay_cache"
            if not new.is_file():
                new.parent.mkdir(parents=True, exist_ok=True)
                if old.is_file():
                    temporary = new.with_suffix(".copy.tmp")
                    shutil.copyfile(old, temporary)
                    os.replace(temporary, new)
                    origin = "copied_original_support_cache"
                    action = "copy_original"
                else:
                    origin = "frozen_support_extracted_this_attempt"
                    action = "extract_frozen"
            record = Record(path=str(paths["data"] / row["relative_path"]), label=0, mask_path=None,
                            category="AeBAD-S", dataset="aebad_s", domain=row["domain"], shift="support", role="support", mask_capability="not_evaluation")
            with torch.inference_mode():
                features = extract_or_load_support(new, record, dataset_root=paths["data"], model=model,
                                                   device=device, protocol_id=PROTOCOL_ID)
            checksum = sha256(new)
            if previous.get("sha256") and checksum != previous["sha256"]:
                raise ValueError("resumed support features changed; preserve products and use a new replay directory")
            if row.get("original_sha256") and origin == "copied_original_support_cache" and checksum != row["original_sha256"]:
                raise ValueError("original support cache changed after preflight")
            feature_groups[row["branch"]].append(features)
            provenance.append(dict(branch=row["branch"], domain=row["domain"], support_index=row["support_index"],
                                   relative_path=row["relative_path"], origin=previous.get("origin", origin), action=action, original_cache=str(old),
                                   replay_cache=str(new), sha256=checksum, grid=json.dumps(list(features.grid)),
                                   rows_per_layer=int(features.grid[0]) * int(features.grid[1]), seconds=time.perf_counter() - started))
            write_csv(output / "support_provenance.csv", provenance)
        source_tensors = runner._support_tensors(feature_groups["source_extra"], device=device)
        target_tensors = runner._support_tensors(feature_groups["target_matched"], device=device)
        feature_groups.clear()
        for case in check["cases"]:
            started = time.perf_counter()
            path = output / "raw_cache/heldout/background" / f"{case['sample_id']}.npz"
            previous = prior_cases.get(case["sample_id"], {})
            row = dict(case, cache_path=str(path), status="failed", action="validate_existing" if path.is_file() else "frozen_query_replay")
            if previous.get("cache_sha256"):
                row["cache_sha256"] = previous["cache_sha256"]
            try:
                check_recorded_cache(path, previous)
                with Image.open(paths["data"] / case["relative_path"]) as source:
                    image = source.convert("RGB")
                expected_shape = (image.height // model.evaluation_downscale, image.width // model.evaluation_downscale)
                processed_shape = preprocessed_spatial_shape(model, image.height, image.width)
                expected_grid = tuple(size // model.patch_exec.model_patch_size for size in processed_shape)
                row.update(grid_height=expected_grid[0], grid_width=expected_grid[1],
                           output_height=expected_shape[0], output_width=expected_shape[1])
                if not path.is_file():
                    progress["query_forward_calls_this_attempt"] += 1
                    observed = observe_fewshot_pil(model, image, official_banks=model.prototype_embeddings,
                                                  source_support=source_tensors, target_support=target_tensors, device=device)
                    runner._save_observation(path, relative=case["relative_path"], output_shape=expected_shape, observed=observed)
                    del observed
                verified = verify_case(path, case, runner, mad, args.score_tolerance, expected_shape, expected_grid)
                if previous.get("cache_sha256") and verified["cache_sha256"] != previous["cache_sha256"]:
                    raise ValueError("recorded replay cache changed during score verification")
                row.update(verified)
                row["status"] = "verified" if row["scores_reproduced"] else "not_reproduced"
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
            row["seconds"] = time.perf_counter() - started
            prior_cases[case["sample_id"]] = row
            progress["cases"] = [prior_cases[sample_id] for sample_id, _ in CASES if sample_id in prior_cases]
            atomic_json(status_path, progress)
            write_csv(output / "replay_verification.csv", progress["cases"])
            print(f"[REPLAY12] {case['sample_id']} {row['status']} error={row.get('max_abs_error', 'unavailable')}", flush=True)
        valid = sum(row["status"] == "verified" for row in progress["cases"])
        changed = [row["kind"] for row in check["required_files"] if sha256(Path(row["path"])) != row["sha256"]]
        if changed:
            raise ValueError(f"original inputs changed during replay: {changed}")
        cache_ids = {path.stem for path in (output / "raw_cache/heldout/background").glob("*.npz")}
        if cache_ids != {sample_id for sample_id, _ in CASES}:
            raise ValueError("replay raw directory must contain exactly the fixed 12 cache files")
        progress.update(status="verified" if valid == 12 else "not_reproduced", verified_queries=valid,
                        precise_original_explanation_allowed=valid == 12,
                        cases_visited=len(progress["cases"]), support_images=16)
        atomic_json(status_path, progress)
        write_csv(output / "replay_cases.csv", check["cases"])
        return 0 if valid == 12 else 3
    except Exception as error:
        progress.update(status="failed", error=f"{type(error).__name__}: {error}", precise_original_explanation_allowed=False)
        atomic_json(status_path, progress)
        raise


def verified_replay_scope(cache: Path, results: Path, *,
                          required_cases: tuple[tuple[str, str], ...] | None = None) -> set[str] | None:
    """Keep verified history/formal inputs; optionally require only a frozen subset's files."""
    marker, inputs = cache / "replay_status.json", cache / "replay_inputs.json"
    if not marker.exists() and not inputs.exists():
        return None
    state = json.loads(marker.read_text(encoding="utf-8"))
    identity = json.loads(inputs.read_text(encoding="utf-8"))["identity"]
    if (state.get("tool") != "aebad_s_background_replay12" or state.get("protocol_id") != PROTOCOL_ID
            or state.get("cache_revision") != CACHE_REVISION or state.get("status") != "verified"
            or not state.get("precise_original_explanation_allowed") or state.get("verified_queries") != 12
            or set(state.get("case_ids", [])) != {sample_id for sample_id, _ in CASES}
            or [(row.get("sample_id"), row.get("relative_path")) for row in state.get("cases", [])] != list(CASES)
            or any(row.get("status") != "verified" or row.get("scores_reproduced") is not True
                   or not math.isfinite(row.get("max_abs_error", math.inf)) or row["max_abs_error"] > 1e-5 for row in state["cases"])):
        raise ValueError("replay12 is incomplete/unverified; diagnostics cannot use it as original spatial evidence")
    for name in ("sample_scores.csv", "support_manifest.csv", "memory_manifest.json", "complete.json"):
        if sha256(results / name) != identity["input_sha256"][name]:
            raise ValueError(f"diagnostic formal input differs from replay: {name}")
    if sha256(results / "source_mad_state.npz") != identity["input_sha256"]["mad"]:
        raise ValueError("diagnostic Source MAD differs from replay")
    expected = dict(CASES)
    if required_cases is not None:
        requested = dict(required_cases)
        if (not requested or len(requested) != len(required_cases)
                or any(expected.get(sample_id) != relative for sample_id, relative in required_cases)):
            raise ValueError("requested replay scope must be a nonempty subset of the frozen cases")
        expected = requested
    elif {path.stem for path in (cache / "raw_cache/heldout/background").glob("*.npz")} != set(expected):
        raise ValueError("verified replay scope is restricted to exactly 12 cache files")
    for row in state["cases"]:
        if row["sample_id"] not in expected:
            continue
        if expected.get(row["sample_id"]) != row["relative_path"]:
            raise ValueError("replay case identity changed")
        path = cache / "raw_cache/heldout/background" / f"{row['sample_id']}.npz"
        if not path.is_file():
            raise ValueError(f"required replay cache missing: {row['sample_id']}")
        if sha256(path) != row["cache_sha256"]:
            raise ValueError("replay cache changed after score verification")
    return set(expected)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "replay"))
    parser.add_argument("--project-root", default=os.environ.get("PROJECT_ROOT", str(Path(__file__).resolve().parents[3])))
    parser.add_argument("--results-dir", default=os.environ.get("RESULTS_DIR"))
    parser.add_argument("--aebad-root", default=os.environ.get("AEBAD_ROOT", "/root/autodl-tmp/datasets/AeBAD/AeBAD_S"))
    parser.add_argument("--official-root", default=os.environ.get("OFFICIAL_ROOT"))
    parser.add_argument("--dinov3-root", default=os.environ.get("DINOV3_ROOT", "/root/autodl-tmp/dinov3"))
    parser.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT"))
    parser.add_argument("--memory", default=os.environ.get("SOURCE_MEMORY"))
    parser.add_argument("--mad", default=os.environ.get("SOURCE_MAD"))
    parser.add_argument("--support-cache-dir", default=os.environ.get("SUPPORT_CACHE_DIR"))
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR"))
    parser.add_argument("--cases-csv", default=os.environ.get("CASES_CSV"), help="Optional prior selected_cases.csv; ignores smallest-instance rows")
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda:0"))
    # Fixed to the existing diagnostic's tolerance; no pass-by-widening flag.
    parser.set_defaults(score_tolerance=1e-5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    check = preflight(args)
    output = Path(check["resolved_paths"]["output"])
    atomic_json(output / "preflight.json", check)
    print(json.dumps(check, ensure_ascii=False, indent=2), flush=True)
    if check["errors"]:
        return 2
    return replay(args, check) if args.mode == "replay" else 0


if __name__ == "__main__":
    raise SystemExit(main())
