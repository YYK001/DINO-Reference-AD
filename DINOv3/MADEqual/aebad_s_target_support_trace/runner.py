"""Preflight (default) or explicitly trace four frozen Target support correspondences."""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import sys
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from DINOv3.MADEqual.aebad_s_target_fewshot import replay12 as replay

TOOL = "aebad_s_target_support_trace4_v1"
CASES = (replay.CASES[2], replay.CASES[4], replay.CASES[11], replay.CASES[10])
LAYERS = replay.LAYERS
QUERY_FORMAT = "full_image_selected_query_features_v1"
DIAGNOSTIC_NAME = "aebad_s_union_background_replay12_seed42_v1_diagnostic_20261003_150831"
SOURCE_CODES = {
    "code_runner": "DINOv3/MADEqual/aebad_s_target_fewshot/runner.py",
    "code_scoring": "DINOv3/MADEqual/robustad_target_fewshot_p0/scoring.py",
    "code_support": "DINOv3/MADEqual/robustad_target_fewshot_p0/support.py",
    "code_host": "DINOv3/relation_reliability/superadd_host.py",
}


def paths_for(args: argparse.Namespace) -> dict[str, Path]:
    project = Path(args.project_root).resolve()
    base = project / "DINOv3/outputs/mad_equal"
    results = Path(args.results_dir or base / "aebad_s_target_fewshot_seed42_v1").resolve()
    cache = Path(args.cache_dir or base / "aebad_s_union_background_replay12_seed42_v1").resolve()
    official = Path(args.official_root or project / "external_baselines/SuperADD").resolve()
    return {"project": project, "results": results, "cache": cache, "official": official,
            "diagnostic": Path(args.diagnostic_dir or base / DIAGNOSTIC_NAME).resolve(),
            "data": Path(args.data).resolve(), "dinov3": Path(args.dinov3_root).resolve(),
            "checkpoint": Path(args.checkpoint or official / "weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth").resolve(),
            "mad": Path(args.mad or results / "source_mad_state.npz").resolve(),
            "support": Path(args.support_cache_dir or cache / "support_features/target_matched/background").resolve(),
            "query": Path(args.query_cache_dir or cache / "query_features/heldout/background").resolve(),
            "output": Path(args.output_dir or base / "aebad_s_background_support_trace4_v1").resolve()}


def guard_output(paths: dict[str, Path]) -> None:
    output = paths["output"]
    for key, root in paths.items():
        if key in ("project", "output"):
            continue
        if output == root or output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError(f"trace output overlaps protected {key}: {root}")
    if output.exists() and any(path.name != "preflight.json" for path in output.iterdir()):
        raise ValueError("trace output must be new (or contain only its preflight.json); no overwrite/resume")


def trace_data_index(supplied: Path) -> replay.DataIndex:
    """Resolve a test-only root; preserve the shared full-index/ZIP checks."""
    if not supplied.is_file():
        roots = [root for root in (supplied, supplied / "AeBAD_S", supplied / "AeBAD/AeBAD_S")
                 if (root / "test").is_dir()]
        if len(roots) != 1:
            raise ValueError("trace data must contain one AeBAD-S test root; train/good is not required")
        supplied = roots[0].resolve()
    index = replay.DataIndex(supplied)
    if not index.zip:
        index.names.update(path.relative_to(index.root).as_posix()
                           for path in (index.root / "ground_truth").rglob("*") if path.is_file())
    return index


def trace_image(index: replay.DataIndex, relative: str, mode: str) -> Any:
    from PIL import Image
    handle = index.zip.open(index.prefix + relative) if index.zip else (index.root / relative).open("rb")
    with handle, Image.open(handle) as image:
        return image.convert(mode)


def dependencies() -> dict[str, Any]:
    results = {}
    for module, package in (("numpy", "numpy"), ("PIL", "Pillow"), ("scipy", "scipy"),
                            ("torch", "torch"), ("torchvision", "torchvision"),
                            ("sklearn", "scikit-learn"), ("tqdm", "tqdm")):
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            found = False
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = None
        results[module] = {"discoverable_without_import": found, "distribution_version": version}
    return results


def query_contract(check: dict[str, Any], relative: str) -> dict[str, Any]:
    return {"protocol_id": replay.PROTOCOL_ID, "feature_format": QUERY_FORMAT, "relative_path": relative,
            "image_sha256": check["image_sha256"][relative], "checkpoint_sha256": replay.CHECKPOINT_SHA256,
            "layers": list(LAYERS), "dimension": 1280, "query_protocol": check["query_protocol"],
            "source_code_sha256": check["source_code_sha256"], "precision": check["expected_precision"],
            "extraction": "image_tensor_then_original_model_preprocessing_then_extract_processed_batch_full_image"}


def query_header(path: Path, relative: str, contract: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    import numpy as np
    with np.load(path, allow_pickle=False) as payload:
        if str(payload["protocol_id"].item()) != replay.PROTOCOL_ID or str(payload["relative_path"].item()) != relative:
            raise ValueError("query feature cache protocol/path mismatch")
        if json.loads(str(payload["contract"].item())) != contract:
            raise ValueError("query feature provenance contract differs from frozen inputs")
        grid, positions = payload["grid"], payload["flat_positions"]
        geometry = json.loads(str(payload["geometry"].item()))
        if (grid.shape != (2,) or not np.issubdtype(grid.dtype, np.integer) or (grid <= 0).any()
                or positions.ndim != 1 or positions.size == 0 or not np.issubdtype(positions.dtype, np.integer)
                or (positions < 0).any() or (positions >= int(grid.prod())).any()
                or (np.diff(positions) <= 0).any() or geometry.get("grid") != grid.tolist()):
            raise ValueError("invalid stored query grid/physical flat positions")
    with zipfile.ZipFile(path) as archive:
        for layer in LAYERS:
            with archive.open(f"layer_{layer}.npy") as handle:
                version = np.lib.format.read_magic(handle)
                if version == (1, 0):
                    shape, _fortran, dtype = np.lib.format.read_array_header_1_0(handle)
                elif version == (2, 0):
                    shape, _fortran, dtype = np.lib.format.read_array_header_2_0(handle)
                else:
                    raise ValueError("unsupported query feature NPY header")
            if shape != (len(positions), 1280) or dtype != np.dtype("float32"):
                raise ValueError(f"query feature L{layer} shape/dtype mismatch")
    return tuple(int(value) for value in grid), positions, geometry


def query_cache_provenance(path: Path, relative: str) -> dict[str, str]:
    """An existing cache must retain its prior recorded SHA, not acquire a new one."""
    owners = [path.parent, *list(path.parent.parents)[:4]]
    for owner in owners:
        provenance = owner / "feature_provenance.csv"
        if not provenance.is_file():
            continue
        rows = [row for row in replay.read_csv(provenance)
                if row.get("kind") == "query" and row.get("relative_path") == relative]
        if len(rows) != 1 or not rows[0].get("sha256"):
            raise ValueError("existing query feature has no unique prior provenance/checksum")
        checksum = replay.sha256(path)
        if checksum != rows[0]["sha256"]:
            raise ValueError("existing query feature changed after its prior provenance record")
        return {"path": str(provenance), "sha256": replay.sha256(provenance),
                "feature_sha256": checksum}
    raise ValueError("existing query feature requires its prior feature_provenance.csv")


def support_contract(check: dict[str, Any], relative: str) -> dict[str, Any]:
    hashes = {row["kind"]: row["sha256"] for row in check["required_files"] if row["exists"]}
    supports = [row for row in check["supports"] if row["relative_path"] == relative]
    if len(supports) != 1:
        raise ValueError("Target support contract requires one frozen manifest identity")
    frozen_keys = (*SOURCE_CODES, "official_config", "official_model", "dinov3_hub",
                   "code_stitch", "code_support_extractor", "sample_scores.csv", "support_manifest.csv",
                   "memory_manifest.json", "complete.json", "mad")
    return {"tool": TOOL, "protocol_id": replay.PROTOCOL_ID, "relative_path": relative,
            "image_sha256": check["image_sha256"][relative], "checkpoint_sha256": hashes["checkpoint"],
            "support_index": int(supports[0]["support_index"]),
            "layers": list(LAYERS), "dimension": 1280, "query_protocol": check["query_protocol"],
            "precision": check["expected_precision"],
            "frozen_input_sha256": {key: hashes[key] for key in frozen_keys},
            "extraction": "original_extract_or_load_support_deterministic_full_image"}


def support_cache_provenance(path: Path, relative: str, check: dict[str, Any],
                             original_path: Path, original_sha256: str) -> dict[str, Any]:
    """Accept original bytes or a prior trace's frozen, hash-checked extraction."""
    replay.check_support_header(path, relative)
    checksum = replay.sha256(path)
    if checksum == original_sha256:
        return {"feature_sha256": checksum, "origin": "recorded_existing_target_support", "evidence_files": []}
    if path.resolve() == original_path.resolve():
        raise ValueError("original Target support feature differs from recorded replay checksum")
    for owner in [path.parent, *list(path.parent.parents)[:4]]:
        provenance = owner / "feature_provenance.csv"
        if not provenance.is_file():
            continue
        rows = [row for row in replay.read_csv(provenance)
                if row.get("kind") == "target_support" and row.get("relative_path") == relative]
        if len(rows) != 1 or rows[0].get("sha256") != checksum:
            raise ValueError("existing Target support feature changed or lacks unique prior trace provenance")
        inputs, environment = owner / "trace_inputs.json", owner / "trace_environment.json"
        prior = json.loads(inputs.read_text(encoding="utf-8"))
        runtime = json.loads(environment.read_text(encoding="utf-8"))
        if prior.get("tool") != TOOL or prior.get("errors"):
            raise ValueError("Target support provenance requires valid prior trace inputs")
        contract = support_contract(check, relative)
        if support_contract(prior, relative) != contract or runtime.get("precision") != contract["precision"]:
            raise ValueError("Target support frozen extraction inputs/precision differ from prior trace")
        if rows[0].get("contract") and json.loads(rows[0]["contract"]) != contract:
            raise ValueError("Target support recorded extraction contract differs")
        geometry = json.loads(rows[0]["geometry"])
        import numpy as np
        with np.load(path, allow_pickle=False) as payload:
            if payload["grid"].tolist() != geometry.get("grid"):
                raise ValueError("Target support grid differs from prior trace geometry")
        return {"feature_sha256": checksum, "origin": "verified_prior_trace_target_support",
                "geometry": geometry, "evidence_files": [
                    {"kind": "support_feature_prior_" + item.stem, "path": str(item),
                     "exists": True, "sha256": replay.sha256(item)} for item in (provenance, inputs, environment)]}
    raise ValueError("existing Target support feature requires prior trace provenance or original replay checksum")


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    paths = paths_for(args)
    guard_output(paths)
    deps = dependencies()
    errors = [f"missing dependency (not imported): {name}" for name, value in deps.items() if not value["discoverable_without_import"]]
    check: dict[str, Any] = {
        "tool": TOOL, "mode": args.mode, "resolved_paths": {key: str(value) for key, value in paths.items()},
        "dependencies": deps, "errors": errors, "required_files": [], "cases": [], "supports": [],
        "images": [], "image_sha256": {}, "feature_inputs": [], "source_memory_required": False,
        "source_extra_required": False, "backbone_loaded": False, "feature_extraction_started": False,
        "cuda_runtime_check": "deferred_to_explicit_trace", "feature_finiteness": "deferred_to_explicit_trace",
    }
    required = [(name, paths[name]) for name in ("checkpoint", "mad")]
    required += [(name, paths["results"] / name) for name in ("sample_scores.csv", "support_manifest.csv", "complete.json", "memory_manifest.json")]
    required += [("replay_" + name, paths["cache"] / name) for name in ("replay_status.json", "replay_inputs.json", "support_provenance.csv")]
    required += [("diagnostic_" + name, paths["diagnostic"] / name) for name in
                 ("diagnostic_status.json", "verification.json", "selected_cases.csv", "case_peaks.csv", "case_peak_layers.csv", "defect_instances.csv")]
    required += [(key, paths["project"] / value) for key, value in SOURCE_CODES.items()]
    required += [("official_config", paths["official"] / "config.json"),
                 ("official_model", paths["official"] / "tracks/industrial/src/industrial/model.py"),
                 ("dinov3_hub", paths["dinov3"] / "hubconf.py"),
                 ("code_trace", Path(__file__).resolve()),
                 ("code_trace_analysis", Path(__file__).with_name("analysis.py")),
                 ("code_trace_visualization", Path(__file__).with_name("visualization.py")),
                 ("code_spatial", paths["project"] / "DINOv3/MADEqual/aebad_s_union_diagnostics/spatial.py"),
                 ("code_stitch", paths["project"] / "external_baselines/superadd_external/memory_variants.py"),
                 ("code_support_extractor", paths["project"] / "DINOv3/ShiftADD/aebad_s_p0/memory.py")]
    for kind, path in required:
        row = {"kind": kind, "path": str(path), "exists": path.is_file()}
        if row["exists"]:
            row.update(sha256=replay.sha256(path), bytes=path.stat().st_size)
        else:
            errors.append(f"missing required {kind}: {path}")
        check["required_files"].append(row)
    hashes = {row["kind"]: row["sha256"] for row in check["required_files"] if row["exists"]}
    if "checkpoint" in hashes and hashes["checkpoint"] != replay.CHECKPOINT_SHA256:
        errors.append("checkpoint SHA256 differs from original approved weights")
    try:
        from .analysis import CASES as fixed_cases, SCORE_ATOL
        from DINOv3.MADEqual.aebad_s_union_diagnostics.spatial import components
        from DINOv3.MADEqual.aebad_s_union_diagnostics.analysis import gt_path
        if fixed_cases != CASES:
            raise ValueError("trace module fixed-case order differs")
        scope = replay.verified_replay_scope(paths["cache"], paths["results"], required_cases=CASES)
        if scope is None or not {sample_id for sample_id, _ in CASES}.issubset(scope):
            raise ValueError("a verified replay12 cache is required")
        state = json.loads((paths["cache"] / "replay_status.json").read_text(encoding="utf-8"))
        identity = json.loads((paths["cache"] / "replay_inputs.json").read_text(encoding="utf-8"))["identity"]
        check["query_protocol"] = state["query_protocol"]
        protocol = check["query_protocol"]
        if any(protocol.get(key) != value for key, value in
               (("brightness_factor", 1), ("patch_size", 640), ("patch_overlap", 128),
                ("model_patch_size", 16), ("evaluation_downscale", 4), ("dino_training", False), ("trainable_parameters", 0))):
            raise ValueError("replayed preprocessing/patch protocol differs")
        check["source_code_sha256"] = {key: hashes[key] for key in SOURCE_CODES}
        for key in (*SOURCE_CODES, "official_config", "official_model", "dinov3_hub"):
            if key in hashes and hashes[key] != identity["input_sha256"][key]:
                errors.append(f"original extraction/host code or config changed: {key}")
        replay.check_mad(paths["mad"])
        if hashes["mad"] != identity["input_sha256"]["mad"]:
            errors.append("Source MAD differs from verified replay")
        environments = sorted(paths["cache"].glob("environment_*.json"))
        if not environments:
            raise ValueError("original replay environment/precision record missing")
        environment = json.loads(environments[-1].read_text(encoding="utf-8"))
        check["expected_precision"] = {key: environment[key] for key in
                                       ("float32_matmul_precision", "cuda_matmul_allow_tf32", "cudnn_allow_tf32")}
        check["replay_environment"] = environment
        check["required_files"].append({"kind": "replay_environment", "path": str(environments[-1]),
                                         "exists": True, "sha256": replay.sha256(environments[-1])})
        diag_status = json.loads((paths["diagnostic"] / "diagnostic_status.json").read_text(encoding="utf-8"))
        verification = json.loads((paths["diagnostic"] / "verification.json").read_text(encoding="utf-8"))
        if (diag_status.get("status") != "planned_subset_complete" or not verification.get("formal_results_unchanged")
                or not verification.get("planned_anomaly_spatial_complete")
                or not math.isfinite(verification.get("max_score_error", math.inf))
                or verification["max_score_error"] > SCORE_ATOL):
            raise ValueError("completed original replay12 diagnostic verification is required")
        full_cases = replay.select_cases(replay.read_csv(paths["results"] / "sample_scores.csv"))
        by_id = {row["sample_id"]: row for row in full_cases}
        selected = {row["sample_id"]: row for row in replay.read_csv(paths["diagnostic"] / "selected_cases.csv")}
        for sample_id, relative in CASES:
            row = by_id[sample_id]
            if row["relative_path"] != relative or selected[sample_id]["relative_path"] != relative:
                raise ValueError("fixed case identity differs from original CSV/diagnostic")
            if any(abs(float(selected[sample_id][key]) - row[key]) > SCORE_ATOL for key in ("source_score", "union_score")):
                raise ValueError("selected diagnostic case scores differ from original CSV")
            path = paths["cache"] / "raw_cache/heldout/background" / f"{sample_id}.npz"
            values, shape, _error = replay.validate_raw(path, row)
            check["cases"].append(dict(row, raw_path=str(path), grid=list(values["official_raw"].shape[1:]),
                                       output_shape=list(shape), image_height=int(selected[sample_id]["image_height"]),
                                       image_width=int(selected[sample_id]["image_width"])))
            check["required_files"].append({"kind": f"raw_{sample_id}", "path": str(path), "exists": True,
                                             "sha256": replay.sha256(path)})
        all_support = replay.select_support(replay.read_csv(paths["results"] / "support_manifest.csv"), full_cases)
        check["supports"] = [row for row in all_support if row["branch"] == "target_matched"]
        prior_support = {row["relative_path"]: row for row in replay.read_csv(paths["cache"] / "support_provenance.csv")}
        index = trace_data_index(paths["data"])
        try:
            paths["data"] = index.root
            check["resolved_paths"]["data"] = str(index.root)
            guard_output(paths)
            for row in [*check["cases"], *check["supports"]]:
                image = index.describe(row["relative_path"])
                if image["sha256"] != identity["image_sha256"][row["relative_path"]]:
                    raise ValueError(f"original query/support image changed: {row['relative_path']}")
                check["images"].append(image)
                check["image_sha256"][row["relative_path"]] = image["sha256"]
                row.update(image_height=image["height"], image_width=image["width"])
                if row in check["cases"] and index.ids and index.ids[row["relative_path"]] != row["sample_id"]:
                    raise ValueError("full test filename index disagrees with fixed sample ID")
            instances = replay.read_csv(paths["diagnostic"] / "defect_instances.csv")
            for row in check["cases"]:
                if not row["label"]:
                    continue
                relative = gt_path(row["relative_path"])
                mask = __import__("numpy").asarray(trace_image(index, relative, "L")) > 0
                _labels, components_rows = components(mask)
                old = [item for item in instances if item["sample_id"] == row["sample_id"]]
                if mask.shape != (row["image_height"], row["image_width"]) or len(old) != 1 or len(components_rows) != 1:
                    raise ValueError("fixed anomaly GT geometry/instance count differs")
                if any(int(old[0][key]) != components_rows[0][key] for key in
                       ("instance_id", "area_original_pixels", "bbox_y0", "bbox_x0", "bbox_y1_exclusive", "bbox_x1_exclusive")):
                    raise ValueError("fixed anomaly GT component differs from original diagnostic")
                image = index.describe(relative)
                check["images"].append(image)
                check["image_sha256"][relative] = image["sha256"]
        finally:
            index.close()
        for row in check["supports"]:
            path = paths["support"] / f"{int(row['support_index']):02d}.npz"
            row["feature_path"] = str(path)
            feature = {"kind": "target_support", "relative_path": row["relative_path"], "path": str(path),
                       "exists": path.is_file(), "missing_action": "extract_only_this_original_target_image_at_trace"}
            if path.is_file():
                original_path = paths["cache"] / "support_features/target_matched/background" / path.name
                proof = support_cache_provenance(path, row["relative_path"], check, original_path,
                                                 prior_support[row["relative_path"]]["sha256"])
                feature.update(sha256=proof["feature_sha256"], origin=proof["origin"])
                if "geometry" in proof:
                    feature["prior_geometry"] = proof["geometry"]
                for evidence in proof["evidence_files"]:
                    if not any(item["path"] == evidence["path"] for item in check["required_files"]):
                        check["required_files"].append(evidence)
            check["feature_inputs"].append(feature)
        for row in check["cases"]:
            path = paths["query"] / f"{row['sample_id']}.npz"
            feature = {"kind": "query", "relative_path": row["relative_path"], "path": str(path),
                       "exists": path.is_file(), "missing_action": "extract_only_this_full_original_query_at_trace"}
            if path.is_file():
                grid, _positions, _geometry = query_header(path, row["relative_path"], query_contract(check, row["relative_path"]))
                if list(grid) != row["grid"]:
                    raise ValueError("query feature grid differs from verified distance cache")
                proof = query_cache_provenance(path, row["relative_path"])
                feature["sha256"] = proof["feature_sha256"]
                if not any(item["path"] == proof["path"] for item in check["required_files"]):
                    check["required_files"].append({"kind": "query_feature_prior_provenance",
                        "path": proof["path"], "sha256": proof["sha256"], "exists": True})
            check["feature_inputs"].append(feature)
        check["dataset_index_status"] = "full_index_crosschecked" if index.ids else "original_csv_ids_preserved_no_subset_renumbering"
    except (OSError, ValueError, KeyError, ImportError, RuntimeError, zipfile.BadZipFile) as error:
        errors.append(f"frozen input validation: {type(error).__name__}: {error}")
    if args.mode == "trace" and paths["data"].is_file():
        errors.append("trace requires extracted AeBAD-S data; ZIP is preflight-only")
    check["status"] = "filesystem_preflight_passed" if not errors else "blocked"
    check["seconds"] = time.perf_counter() - args.started
    return check


def load_query(path: Path, relative: str, contract: dict[str, Any], geometry: dict[str, Any],
               selected: list[int]) -> dict[int, Any]:
    import numpy as np
    grid, positions, stored_geometry = query_header(path, relative, contract)
    if list(grid) != geometry["grid"] or stored_geometry != geometry:
        raise ValueError("cached query token/tile geometry differs from actual host")
    offsets = np.searchsorted(positions, selected)
    if np.any(offsets >= len(positions)) or not np.array_equal(positions[offsets], selected):
        raise ValueError("cached query features do not contain all selected physical patches")
    with np.load(path, allow_pickle=False) as payload:
        layers = {}
        for layer in LAYERS:
            value = payload[f"layer_{layer}"]
            if not np.isfinite(value).all():
                raise ValueError("nonfinite existing query features")
            layers[layer] = np.ascontiguousarray(value[offsets], dtype=np.float32)
    return layers


def trace_report(output: Path, state: dict[str, Any]) -> None:
    lines = [
        "# AeBAD-S/background four-case Target correspondence trace", "",
        f"Status: {state['status']}. Selected cases: 4; Target support images: 8.",
        f"Positions: {state.get('positions', 0)}; verified positions: {state.get('verified_positions', 0)}.",
        f"New full-image query forwards: {state.get('query_forward_calls', 0)}; new support forwards: {state.get('support_forward_calls', 0)}.",
        f"Elapsed trace seconds: {state.get('seconds', 0):.3f}.",
        f"Error: {state.get('error', '')}", "",
        "Source distances come only from verified official_raw; Target distances and indices are traced here.",
        "Only selected interpolation-stencil patches are searched; this does not reproduce full-domain inference.",
        "No Source memory or Source-extra bank is loaded. Source MAD and detector/interpolation remain unchanged.",
        "GT selects post-hoc explanation positions only. Its complement is not a background annotation.", "",
        "Distance tolerance is fixed at atol=1e-7, rtol=1e-6; score tolerance is absolute 1e-5.",
        "Tie rules: first row-major regional maximum; first flat patch minimum per support; lower manifest index between supports.",
        "Same output coordinates are deduplicated, with every role retained. No interpolated-feature or crop feature query is used.",
        "Figures show only the largest-weight physical query patch (flat-index tie break), and its four layer winners.",
        "All stencil patches and all eight candidates remain in CSVs. Display crops are never inference inputs.",
        "Projected token cells and centers use actual preprocessing and stitch ROIs; they are not complete receptive fields.", "",
        "query_region and support_correspondence start as unreviewed. Human interpretation remains pending.",
        "Unverified figures are marked UNVERIFIED and cannot be used as exact correspondence evidence for the original experiment.",
        "Read trace_positions.csv, query_patches.csv, nearest_support.csv, trace_verification.csv, feature_provenance.csv and figures/.",
        "Input/file hashes and precision/version records are in trace_inputs.json and trace_environment.json.", "",
    ]
    (output / "trace_report.md").write_text("\n".join(lines), encoding="utf-8")


def trace(args: argparse.Namespace, check: dict[str, Any]) -> int:
    if check["errors"]:
        return 2
    paths = {key: Path(value) for key, value in check["resolved_paths"].items()}
    guard_output(paths)
    output = paths["output"]
    output.mkdir(parents=True, exist_ok=True)
    replay.atomic_json(output / "trace_inputs.json", check)
    started = time.perf_counter()
    state: dict[str, Any] = {"tool": TOOL, "status": "running", "case_ids": [key for key, _ in CASES],
                            "query_forward_calls": 0, "support_forward_calls": 0,
                            "precise_original_correspondence_allowed": False}
    replay.atomic_json(output / "trace_status.json", state)
    tables: dict[str, list[dict[str, Any]]] = {key: [] for key in
        ("trace_positions", "query_patches", "nearest_support", "trace_verification", "feature_provenance")}

    def flush() -> None:
        for name, rows in tables.items():
            replay.write_csv(output / (name + ".csv"), rows)

    try:
        import numpy as np
        import torch
        from PIL import Image
        from .analysis import (
            METHODS, DIST_ATOL, DIST_RTOL, SCORE_ATOL, comparison, image_geometry,
            patch_projection, per_support_nearest, position_patches, target_check,
            trace_positions, weighted_layers,
        )
        from .visualization import correspondence_figure
        from DINOv3.MADEqual.aebad_s_union_diagnostics.spatial import MapBackend, instance_response
        from DINOv3.MADEqual.aebad_s_union_diagnostics.analysis import gt_path
        from DINOv3.MADEqual.robustad_target_fewshot_p0.support import extract_or_load_support, _atomic_npz
        from DINOv3.relation_reliability.datasets import Record
        from DINOv3.relation_reliability.superadd_host import (
            _exact_top1_chunked, _extract_processed_batch, construct_exact_host, image_tensor,
            install_import_paths, preprocessed_spatial_shape,
        )
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("explicit trace requires CUDA; no model/nearest-neighbor CPU fallback")
        device = torch.device("cuda", device.index if device.index is not None else 0)
        torch.cuda.set_device(device)
        precision = {"float32_matmul_precision": torch.get_float32_matmul_precision(),
                     "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                     "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}
        if precision != check["expected_precision"]:
            raise ValueError("runtime precision differs from replay; parameters are not changed automatically")
        software = {}
        for name in ("numpy", "torch", "torchvision", "Pillow", "scipy"):
            software[name] = importlib.metadata.version(name)
        replay.atomic_json(output / "trace_environment.json", {
            "python": sys.version, "platform": platform.platform(), "software": software,
            "cuda": torch.version.cuda, "device": str(device), "device_name": torch.cuda.get_device_name(device),
            "precision": precision, "replay_software": check["replay_environment"]["software"],
            "version_differences": {key: [check["replay_environment"]["software"].get(key), value]
                                    for key, value in software.items() if check["replay_environment"]["software"].get(key) != value},
        })
        install_import_paths(paths["project"], paths["official"], paths["dinov3"])
        from industrial import model as original_model
        if Path(original_model.__file__).resolve() != (paths["official"] / "tracks/industrial/src/industrial/model.py").resolve():
            raise ValueError("industrial.model is not from the specified official root")
        protocol = check["query_protocol"]
        geometry_host = SimpleNamespace(
            preprocessing=SimpleNamespace(resize_factor=protocol["resize_factor"]),
            patch_exec=original_model.PatchedExecution(protocol["patch_size"], protocol["patch_overlap"], protocol["model_patch_size"]),
        )
        backend = MapBackend(paths["mad"])
        backend.runner._seed_everything()
        model_holder = []

        def model_for_missing_features() -> Any:
            if not model_holder:
                from unittest.mock import patch
                def checkpoint_path(name: str) -> Path:
                    if name != "dinov3_vith16plus":
                        raise ValueError("unexpected backbone request")
                    return paths["checkpoint"]
                with patch.object(original_model, "get_weights_path", checkpoint_path):
                    model, _config = construct_exact_host(paths["official"], paths["dinov3"],
                                                          paths["checkpoint"], replay.CHECKPOINT_SHA256)
                actual = {"brightness_factor": model.preprocessing.min_brightness_factor,
                          "resize_factor": model.preprocessing.resize_factor, "patch_size": model.patch_exec.patch_size,
                          "patch_overlap": model.patch_exec.patch_overlap, "model_patch_size": model.patch_exec.model_patch_size,
                          "evaluation_downscale": model.evaluation_downscale,
                          "dino_training": model.backbone.dino.training,
                          "trainable_parameters": sum(parameter.requires_grad for parameter in model.backbone.dino.parameters())}
                model.backbone.dino.eval()
                model.backbone.dino.requires_grad_(False)
                actual["dino_training"], actual["trainable_parameters"] = False, 0
                if actual != protocol or model.preprocessing.max_brightness_factor != 1:
                    raise ValueError("constructed host differs from replayed query protocol")
                model_holder.append(model)
                state["backbone_loaded"] = True
            return model_holder[0]

        input_features = {(row["kind"], row["relative_path"]): row for row in check["feature_inputs"]}
        old_peaks = replay.read_csv(paths["diagnostic"] / "case_peaks.csv")
        old_layers = replay.read_csv(paths["diagnostic"] / "case_peak_layers.csv")
        old_instances = replay.read_csv(paths["diagnostic"] / "defect_instances.csv")
        prepared = []

        def audit(kind: str, actual: Any, expected: Any, sample_id: str,
                  position_id: str = "", **extra: Any) -> bool:
            result = comparison(actual, expected, score=True)
            tables["trace_verification"].append(dict(sample_id=sample_id, position_id=position_id,
                check_kind=kind, actual=json.dumps(np.asarray(actual).tolist()),
                expected=json.dumps(np.asarray(expected).tolist()), score_atol=SCORE_ATOL, **result, **extra))
            return result["verified"]

        # Verify every original location before any missing-feature extraction.
        for sample in check["cases"]:
            case_started = time.perf_counter()
            with Image.open(paths["data"] / sample["relative_path"]) as handle:
                image = handle.convert("RGB")
            mask = None
            if sample["label"]:
                with Image.open(paths["data"] / gt_path(sample["relative_path"])) as handle:
                    mask = np.asarray(handle.convert("L")) > 0
            values, maps, shape = backend.maps(Path(sample["raw_path"]), sample["relative_path"])
            grid = tuple(values["official_raw"].shape[1:])
            geometry = image_geometry(geometry_host, image.height, image.width, grid,
                                      preprocessed_spatial_shape(geometry_host, image.height, image.width))
            if shape != (image.height // protocol["evaluation_downscale"], image.width // protocol["evaluation_downscale"]):
                raise ValueError("original map/output geometry differs")
            positions = trace_positions(sample, maps, mask)
            patches = position_patches(positions, grid)
            score_ok = audit("original_image_scores", [maps[method].max() for method in METHODS],
                             [sample["source_score"], sample["union_score"]], sample["sample_id"])
            instance_ok = True
            if sample["label"]:
                response = instance_response(mask, mask, maps, values, backend)
                old = next(row for row in old_instances if row["sample_id"] == sample["sample_id"])
                keys = ("source_instance_max", "union_instance_max", "source_instance_mean", "union_instance_mean",
                        "source_margin", "union_margin", "margin_change")
                instance_ok = audit("original_instance_and_margin", [response[key] for key in keys],
                                    [float(old[key]) for key in keys], sample["sample_id"], fields=";".join(keys))
                ring_scores = [response[prefix + "_instance_max"] - response[prefix + "_margin"] for prefix in ("source", "union")]
            for position in positions:
                pid = position["position_id"]
                group = [row for row in patches if row["position_id"] == pid]
                contributions = weighted_layers(values, group, backend.calibrated, backend.mad.medians, backend.mad.scales)
                weighted_ok = audit("original_stencil_response", [sum(contributions[name]) for name in
                                    ("official_raw", "target_support_8", "target_raw_8")],
                                    [maps[key][position["output_y"], position["output_x"]] for key in
                                     (*METHODS[:1], "TARGET_ONLY_DIAGNOSTIC", METHODS[1])], sample["sample_id"], pid)
                original_ok = score_ok and instance_ok and weighted_ok
                roles = position["roles"].split(";")
                for prefix in ("source", "union"):
                    if prefix + "_full_peak" in roles:
                        old = next(row for row in old_peaks if row["sample_id"] == sample["sample_id"] and row["peak"] == prefix + "_peak")
                        coordinates_ok = (position["output_y"], position["output_x"]) == (int(old["pixel_y"]), int(old["pixel_x"]))
                        value_ok = audit("archived_full_peak", [position["source_score"], position["union_score"]],
                                         [float(old["source_score_at_pixel"]), float(old["union_score_at_pixel"])],
                                         sample["sample_id"], pid, archived_coordinates_match=coordinates_ok)
                        original_ok &= coordinates_ok and value_ok
                        for offset, layer in enumerate(LAYERS):
                            old_layer = next(row for row in old_layers if row["sample_id"] == sample["sample_id"]
                                             and row["peak"] == prefix + "_peak" and int(row["layer"]) == layer)
                            actual = [contributions[name][offset] for name in ("official_raw", "target_support_8", "target_raw_8")]
                            expected = [float(old_layer[key]) for key in
                                        ("source_mad_contribution", "target_only_mad_contribution", "union_mad_contribution")]
                            original_ok &= audit("archived_peak_layer", actual, expected, sample["sample_id"], pid, layer=layer)
                    if sample["label"] and prefix + "_ring_peak" in roles:
                        original_ok &= audit("original_ring_max", position[prefix + "_score"],
                                             ring_scores[0 if prefix == "source" else 1], sample["sample_id"], pid)
                    if sample["label"] and prefix + "_instance_peak" in roles:
                        original_ok &= audit("original_instance_max", position[prefix + "_score"],
                                             response[prefix + "_instance_max"], sample["sample_id"], pid)
                position.update(original_location_verified=original_ok,
                                original_y_center=(position["output_y"] + .5) * image.height / shape[0] - .5,
                                original_x_center=(position["output_x"] + .5) * image.width / shape[1] - .5,
                                correspondence_verified=False)
                tables["trace_positions"].append(position)
            tables["query_patches"].extend(patches)
            prepared.append((sample, image, mask, values, maps, positions, patches, geometry,
                             time.perf_counter() - case_started))
        flush()
        state.update(positions=len(tables["trace_positions"]), verified_positions=0)
        replay.atomic_json(output / "trace_status.json", state)

        support_features, support_geometry = [], []
        for support in check["supports"]:
            feature_started = time.perf_counter()
            input_row = input_features[("target_support", support["relative_path"])]
            original_path = Path(input_row["path"])
            if input_row["exists"]:
                if replay.sha256(original_path) != input_row["sha256"]:
                    raise ValueError("Target support feature changed after preflight")
                path, origin, model = original_path, input_row["origin"], None
            else:
                path = output / "support_features/target_matched/background" / f"{int(support['support_index']):02d}.npz"
                origin = "new_original_deterministic_target_support"
                model = model_for_missing_features()
                state["support_forward_calls"] += 1
            record = Record(path=str(paths["data"] / support["relative_path"]), label=0, mask_path=None,
                            category="AeBAD-S", dataset="aebad_s", domain="background", shift="support",
                            role="support", mask_capability="not_evaluation")
            with torch.inference_mode():
                features = extract_or_load_support(path, record, dataset_root=paths["data"], model=model,
                                                   device=device, protocol_id=replay.PROTOCOL_ID)
            geometry = image_geometry(geometry_host, support["image_height"], support["image_width"], features.grid,
                preprocessed_spatial_shape(geometry_host, support["image_height"], support["image_width"]))
            if "prior_geometry" in input_row and input_row["prior_geometry"] != geometry:
                raise ValueError("Target support actual tile geometry differs from prior trace")
            support_features.append(features)
            support_geometry.append(geometry)
            tables["feature_provenance"].append({"kind": "target_support", "relative_path": support["relative_path"],
                "feature_path": str(path), "origin": origin, "sha256": replay.sha256(path),
                "grid": json.dumps(list(features.grid)), "selected_query_rows": "", "geometry": json.dumps(geometry),
                "contract": json.dumps(support_contract(check, support["relative_path"]), sort_keys=True),
                "seconds": time.perf_counter() - feature_started})
            print(f"[TRACE4] support {support['support_index']} {origin}", flush=True)
            flush()

        for sample, image, mask, values, maps, positions, patches, geometry, preparation_seconds in prepared:
            case_started = time.perf_counter()
            feature_started = time.perf_counter()
            selected = sorted({row["query_flat_index"] for row in patches})
            contract = query_contract(check, sample["relative_path"])
            input_row = input_features[("query", sample["relative_path"])]
            if input_row["exists"]:
                path = Path(input_row["path"])
                if replay.sha256(path) != input_row["sha256"]:
                    raise ValueError("query features changed after preflight")
                layers = load_query(path, sample["relative_path"], contract, geometry, selected)
                origin = "verified_contract_existing_full_image_query"
            else:
                path = output / "query_features" / f"{sample['sample_id']}.npz"
                model = model_for_missing_features()
                with torch.inference_mode():
                    # Same full-image extraction lines as observe_fewshot_pil;
                    # deliberately no Official or Source-extra distance scan.
                    tensor = image_tensor(image).to(device=device)[None]
                    processed = model.preprocessing(tensor)
                    pending_items = _extract_processed_batch(model, processed, device=device,
                        query_masks=[None], include_identity_descriptors=False)
                    if len(pending_items) != 1:
                        raise ValueError("one full query observation required")
                    pending = pending_items[0]
                    if (list(pending.grid) != geometry["grid"]
                            or not np.array_equal(pending.positions, np.arange(np.prod(pending.grid)))):
                        raise ValueError("original query extraction did not return the full row-major physical grid")
                    actual_geometry = image_geometry(model, image.height, image.width, pending.grid,
                                                      tuple(int(size) for size in processed.shape[-2:]))
                    if actual_geometry != geometry:
                        raise ValueError("query extraction tile geometry differs")
                    index = torch.as_tensor(selected, dtype=torch.long, device=device)
                    layers = {layer: pending.queries[layer].index_select(0, index).detach().float().cpu().numpy()
                              for layer in LAYERS}
                    pending.queries.clear()
                    del tensor, processed, pending_items, pending
                state["query_forward_calls"] += 1
                _atomic_npz(path, protocol_id=np.asarray(replay.PROTOCOL_ID),
                            relative_path=np.asarray(sample["relative_path"]), grid=np.asarray(geometry["grid"], np.int32),
                            flat_positions=np.asarray(selected, np.int64), contract=np.asarray(json.dumps(contract, sort_keys=True)),
                            geometry=np.asarray(json.dumps(geometry, sort_keys=True)),
                            **{f"layer_{layer}": layers[layer] for layer in LAYERS})
                layers = load_query(path, sample["relative_path"], contract, geometry, selected)
                origin = "new_full_image_original_query_path_selected_rows_saved"
            tables["feature_provenance"].append({"kind": "query", "relative_path": sample["relative_path"],
                "feature_path": str(path), "origin": origin, "sha256": replay.sha256(path),
                "grid": json.dumps(geometry["grid"]), "selected_query_rows": len(selected),
                "geometry": json.dumps(geometry), "seconds": time.perf_counter() - feature_started})
            winners = {}
            revised = {key: value.copy() for key, value in values.items()
                       if key in ("official_raw", "target_support_8", "target_raw_8")}
            patch_checks = {}
            for offset, layer in enumerate(LAYERS):
                query = torch.from_numpy(layers[layer]).to(device=device)
                banks = [torch.from_numpy(features.layers[layer]).to(device=device) for features in support_features]
                def exact_top1(query: Any, bank: Any) -> tuple[Any, Any]:
                    with torch.inference_mode():
                        distances, indices = _exact_top1_chunked(query, bank)
                    return distances.numpy(), indices.numpy()
                candidates = per_support_nearest(query, banks, [features.grid for features in support_features], exact_top1)
                for flat, choices in zip(selected, candidates, strict=True):
                    y, x = divmod(flat, geometry["grid"][1])
                    source = float(values["official_raw"][offset, y, x])
                    target = choices[0]["distance_euclidean_over_1280"]
                    cached_target, cached_union = (float(values[key][offset, y, x]) for key in ("target_support_8", "target_raw_8"))
                    result = target_check(source, target, cached_target, cached_union)
                    patch_checks[(flat, layer)] = result
                    revised["target_support_8"][offset, y, x] = target
                    revised["target_raw_8"][offset, y, x] = min(source, target)
                    winners[(flat, layer)] = choices[0]
                    position_ids = sorted({row["position_id"] for row in patches if row["query_flat_index"] == flat})
                    for choice in choices:
                        support_index = choice["support_index"] - 1
                        choice.update(sample_id=sample["sample_id"], query_flat_index=flat, query_patch_y=y, query_patch_x=x,
                            layer=layer, position_ids=";".join(position_ids),
                            support_relative_path=check["supports"][support_index]["relative_path"],
                            cached_d_S=source, cached_d_T=cached_target, cached_d_ST=cached_union,
                            recomputed_d_T=target, recomputed_d_ST=min(source, target),
                            d_S_origin="verified_official_raw", d_T_origin="selected_patch_target_exact_1NN",
                            correspondence_verified=result["verified"],
                            query_region="unreviewed", support_correspondence="unreviewed",
                            **patch_projection(choice["support_flat_index"], support_geometry[support_index]))
                        tables["nearest_support"].append(choice)
                del query, banks
            for patch_row in patches:
                flat, layer = patch_row["query_flat_index"], patch_row["layer"]
                result = patch_checks[(flat, layer)]
                patch_row.update(correspondence_verified=result["verified"], **patch_projection(flat, geometry))
                tables["trace_verification"].append(dict(sample_id=sample["sample_id"],
                    position_id=patch_row["position_id"], check_kind="patch_target_and_union", layer=layer,
                    query_flat_index=flat, distance_atol=DIST_ATOL, distance_rtol=DIST_RTOL, **result))
            for position in positions:
                group = [row for row in patches if row["position_id"] == position["position_id"]]
                original = weighted_layers(values, group, backend.calibrated, backend.mad.medians, backend.mad.scales)
                updated = weighted_layers(revised, group, backend.calibrated, backend.mad.medians, backend.mad.scales)
                per_layer_ok = True
                for offset, layer in enumerate(LAYERS):
                    per_layer_ok &= audit("position_layer_mad_contribution",
                        [updated[name][offset] for name in ("official_raw", "target_support_8", "target_raw_8")],
                        [original[name][offset] for name in ("official_raw", "target_support_8", "target_raw_8")],
                        sample["sample_id"], position["position_id"], layer=layer)
                totals = [sum(updated[name]) for name in ("official_raw", "target_support_8", "target_raw_8")]
                score_ok = audit("traced_stencil_response", totals,
                    [maps[key][position["output_y"], position["output_x"]] for key in
                     (METHODS[0], "TARGET_ONLY_DIAGNOSTIC", METHODS[1])], sample["sample_id"], position["position_id"])
                verified = (position["original_location_verified"] and per_layer_ok and score_ok
                            and all(row["correspondence_verified"] for row in group))
                position.update(correspondence_verified=verified,
                    traced_source_score=totals[0], traced_target_only_score=totals[1], traced_union_score=totals[2])
                for choice in tables["nearest_support"]:
                    if choice["sample_id"] == sample["sample_id"] and position["position_id"] in choice["position_ids"].split(";"):
                        # A patch can serve several positions; retain their status separately.
                        choice.setdefault("unverified_position_ids", "")
                        if not verified:
                            choice["unverified_position_ids"] += position["position_id"] + ";"
                display = next(row for row in group if row["display_patch"])
                position["display_query_flat_index"] = display["query_flat_index"]
                figure = output / "figures" / (position["position_id"] + ("_verified.png" if verified else "_unverified.png"))
                correspondence_figure(image, position, patch_projection(display["query_flat_index"], geometry),
                                      [winners[(display["query_flat_index"], layer)] for layer in LAYERS],
                                      paths["data"], figure)
            print(f"[TRACE4] {sample['sample_id']} positions={len(positions)} selected_patches={len(selected)} "
                  f"verified={sum(row['correspondence_verified'] for row in positions)}", flush=True)
            state.setdefault("cases", []).append({"sample_id": sample["sample_id"], "positions": len(positions),
                "selected_patch_queries_per_layer": len(selected), "preparation_seconds": preparation_seconds,
                "seconds": preparation_seconds + time.perf_counter() - case_started})
            state.update(verified_positions=sum(row["correspondence_verified"] for row in tables["trace_positions"]),
                         nearest_support_rows=len(tables["nearest_support"]))
            flush()
            replay.atomic_json(output / "trace_status.json", state)
        protected = [row for row in check["required_files"] if row["exists"]]
        protected += [row for row in check["feature_inputs"] if row["exists"]]
        changed = [row["path"] for row in protected if not Path(row["path"]).is_file() or replay.sha256(Path(row["path"])) != row["sha256"]]
        if changed:
            raise ValueError(f"protected inputs changed during trace: {changed}")
        # Rehash only the four required caches against their original history.
        replay.verified_replay_scope(paths["cache"], paths["results"], required_cases=CASES)
        data_index = trace_data_index(paths["data"])
        try:
            for row in check["images"]:
                path = paths["data"] / row["relative_path"]
                checksum = data_index.describe(row["relative_path"])["sha256"] if data_index.zip else replay.sha256(path)
                if checksum != row["sha256"]:
                    raise ValueError("original query/support/GT image changed during trace")
        finally:
            data_index.close()
        verified = sum(row["correspondence_verified"] for row in tables["trace_positions"])
        all_ok = verified == len(tables["trace_positions"]) and verified > 0
        state.update(status="verified_selected_correspondence" if all_ok else "unverified_correspondence",
            positions=len(tables["trace_positions"]), verified_positions=verified,
            protected_inputs_unchanged=True, source_memory_loaded=False, source_extra_loaded=False,
            nearest_support_rows=len(tables["nearest_support"]), selected_case_count=4, target_support_count=8,
            precise_original_correspondence_allowed=all_ok,
            max_distance_abs_error=max((row.get(key, 0) for row in tables["trace_verification"]
                                       for key in ("target_abs_error", "union_abs_error") if row.get(key) is not None), default=0),
            max_distance_rel_error=max((row.get(key, 0) for row in tables["trace_verification"]
                                       for key in ("target_rel_error", "union_rel_error") if row.get(key) is not None), default=0),
            max_score_abs_error=max((row["max_abs_error"] for row in tables["trace_verification"]
                                     if row.get("max_abs_error") is not None), default=0),
            seconds=time.perf_counter() - started)
        replay.atomic_json(output / "trace_status.json", state)
        trace_report(output, state)
        return 0 if all_ok else 3
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}",
                     precise_original_correspondence_allowed=False, seconds=time.perf_counter() - started)
        flush()
        replay.atomic_json(output / "trace_status.json", state)
        trace_report(output, state)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("preflight", "trace"), default="preflight")
    parser.add_argument("--project-root", default=os.environ.get("PROJECT_ROOT", str(Path(__file__).resolve().parents[3])))
    parser.add_argument("--results-dir", default=os.environ.get("RESULTS_DIR"))
    parser.add_argument("--cache-dir", default=os.environ.get("REPLAY_DIR"))
    parser.add_argument("--diagnostic-dir", default=os.environ.get("DIAGNOSTIC_DIR"))
    parser.add_argument("--data", default=os.environ.get("AEBAD_ROOT", "/root/autodl-tmp/datasets/AeBAD/AeBAD_S"))
    parser.add_argument("--official-root", default=os.environ.get("OFFICIAL_ROOT"))
    parser.add_argument("--dinov3-root", default=os.environ.get("DINOV3_ROOT", "/root/autodl-tmp/dinov3"))
    parser.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT"))
    parser.add_argument("--mad", default=os.environ.get("SOURCE_MAD"))
    parser.add_argument("--support-cache-dir", help="Read-only directory containing original Target 01.npz..08.npz")
    parser.add_argument("--query-cache-dir", help="Read-only directory of verifiable full-image query feature NPZs")
    parser.add_argument("--output-dir", default=os.environ.get("TRACE_OUTPUT_DIR"))
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda:0"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.started = time.perf_counter()
    check = preflight(args)
    replay.atomic_json(Path(check["resolved_paths"]["output"]) / "preflight.json", check)
    print(json.dumps(check, ensure_ascii=False, indent=2), flush=True)
    if check["errors"]:
        return 2
    return trace(args, check) if args.mode == "trace" else 0


if __name__ == "__main__":
    raise SystemExit(main())
