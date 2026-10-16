"""Read-only filesystem preflight and compact, immutable capture identities."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from collections import Counter
from pathlib import Path
from typing import Any

from DINOv3.MADEqual.aebad_s_target_fewshot import replay12 as replay
from DINOv3.MADEqual.aebad_s_target_support_trace.runner import (
    SOURCE_CODES, DIAGNOSTIC_NAME, trace_data_index,
)

TOOL = "aebad_s_union_support_loo_v1"


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def paths_for(args: Any) -> dict[str, Path]:
    project = Path(args.project_root).resolve()
    base = project / "DINOv3/outputs/mad_equal"
    results = Path(args.results_dir or base / "aebad_s_target_fewshot_seed42_v1").resolve()
    replay_dir = Path(args.replay_dir or base / "aebad_s_union_background_replay12_seed42_v1").resolve()
    official = Path(args.official_root or project / "external_baselines/SuperADD").resolve()
    paths = dict(project=project, results=results, replay=replay_dir, official=official,
        diagnostic=Path(args.diagnostic_dir or base / DIAGNOSTIC_NAME).resolve(),
        data=Path(args.data).resolve(), dinov3=Path(args.dinov3_root).resolve(),
        checkpoint=Path(args.checkpoint or official / "weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth").resolve(),
        memory=Path(args.memory or results / "official_memory.pt").resolve(),
        mad=results / "source_mad_state.npz",
        support=Path(args.support_cache_dir or replay_dir / "support_features/target_matched/background").resolve(),
        output=Path(args.output_dir or base / f"aebad_s_union_support_loo_{args.scope}_{args.mode}_v1").resolve())
    if args.stage_a_dir:
        paths["stage_a"] = Path(args.stage_a_dir).resolve()
    if args.capture_dir:
        paths["capture"] = Path(args.capture_dir).resolve()
    return paths


def guard_output(paths: dict[str, Path]) -> None:
    output = paths["output"]
    for key, root in paths.items():
        if key in ("project", "output"):
            continue
        if output == root or output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError(f"output overlaps protected {key}: {root}")
    if output.exists() and (not output.is_dir() or any(p.name != "preflight.json" for p in output.iterdir())):
        raise ValueError("use a new output directory; no resume/overwrite of existing results or half-written files")


def verify_files(records: list[dict]) -> None:
    for record in records:
        path = Path(record["path"])
        if not path.is_file() or replay.sha256(path) != record["sha256"]:
            raise ValueError(f"protected input missing/modified: {record['kind']}")


def read_capture(root: Path, *, stage_a: bool = False) -> tuple[dict, dict]:
    from .analysis import load_raw
    inputs = json.loads((root / "inputs.json").read_text(encoding="utf-8"))
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    if (inputs.get("tool") != TOOL or inputs.get("identity") != digest(inputs["frozen"])
            or status.get("tool") != TOOL or status.get("status") != "capture_verified"
            or status.get("identity") != inputs["identity"]
            or status.get("scope") != inputs["scope"]
            or (stage_a and inputs["scope"] != "twelve")
            or status.get("inputs_sha256") != replay.sha256(root / "inputs.json")
            or status.get("verification_sha256") != replay.sha256(root / "capture_verification.csv")):
        raise ValueError("capture is incomplete/unverified or its validation record changed")
    expected = inputs["samples"]
    if (len(expected) != (12 if inputs["scope"] == "twelve" else 297)
            or len(status["records"]) != len(expected)
            or [r["sample_id"] for r in status["records"]] != [r["sample_id"] for r in expected]):
        raise ValueError("capture record scope/order mismatch")
    checks = replay.read_csv(root / "capture_verification.csv")
    if not checks or any(r["verified"].lower() != "true" for r in checks):
        raise ValueError("failed numerical capture check")
    for record, sample in zip(status["records"], expected, strict=True):
        if record["file"] != f"raw_cache/{sample['sample_id']}.npz":
            raise ValueError("unsafe/noncanonical raw record path")
        arrays, metadata = load_raw(root / record["file"], identity=inputs["identity"], sample=sample,
                                    checksum=record["sha256"])
        if metadata["image_sha256"] != inputs["image_sha256"][sample["relative_path"]]:
            raise ValueError("query image identity mismatch")
        if metadata["supports"] != inputs["supports"]:
            raise ValueError("support identity/geometry changed")
        del arrays
    return inputs, status


def preflight(args: Any) -> dict:
    paths = paths_for(args)
    guard_output(paths)
    errors: list[str] = []
    check = dict(tool=TOOL, mode=args.mode, scope=args.scope, errors=errors,
                 paths={k: str(v) for k, v in paths.items()}, required_files=[], samples=[], supports=[],
                 backbone_loaded=False, feature_extraction_started=False, torch_imported=False)
    check["dependency_distributions"] = {}
    for name in ("numpy", "Pillow", "scipy", "torch", "torchvision", "scikit-learn", "matplotlib"):
        try:
            check["dependency_distributions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            check["dependency_distributions"][name] = None
    try:
        if args.mode == "analyze":
            if "capture" not in paths:
                raise ValueError("analyze requires --capture-dir")
            inputs, state = read_capture(paths["capture"])
            if inputs["scope"] != args.scope:
                raise ValueError("analyze --scope must match capture")
            # Relocation is allowed only for byte-identical formal inputs/MAD;
            # heavy banks and checkpoint are not required by offline analysis.
            for key in ("sample_scores.csv", "support_manifest.csv", "complete.json", "memory_manifest.json", "source_mad_state.npz"):
                path = paths["results"] / key
                if replay.sha256(path) != inputs["frozen"]["formal_sha256"][key]:
                    raise ValueError(f"offline formal input differs: {key}")
                check["required_files"].append(dict(kind=key, path=str(path), sha256=replay.sha256(path)))
            check.update(capture_inputs=inputs, capture_status=state, samples=inputs["samples"], supports=inputs["supports"])
            check["status"] = "filesystem_preflight_passed"
            return check
        if args.scope == "background":
            if "stage_a" not in paths:
                raise ValueError("background requires --stage-a-dir; twelve-case capture must pass first")
            a_inputs, a_state = read_capture(paths["stage_a"], stage_a=True)
        required = [(name, paths["results"] / name) for name in
                    ("sample_scores.csv", "support_manifest.csv", "complete.json", "memory_manifest.json", "source_mad_state.npz")]
        required += [(name, paths[name]) for name in ("memory", "checkpoint")]
        required += [("replay_"+name, paths["replay"] / name) for name in
                     ("replay_status.json", "replay_inputs.json", "support_provenance.csv")]
        required += [("diagnostic_"+name, paths["diagnostic"] / name) for name in
                     ("selected_cases.csv", "case_peaks.csv", "case_peak_layers.csv", "defect_instances.csv", "diagnostic_status.json", "verification.json")]
        required += [(k, paths["project"] / v) for k, v in SOURCE_CODES.items()]
        required += [("official_config", paths["official"] / "config.json"),
                     ("official_model", paths["official"] / "tracks/industrial/src/industrial/model.py"),
                     ("dinov3_hub", paths["dinov3"] / "hubconf.py"),
                     ("code_mad", paths["project"] / "DINOv3/relation_oracle_cache/extract.py")]
        required += [("code_loo_"+p.stem, p) for p in Path(__file__).parent.glob("*.py")]
        for kind, path in required:
            if not path.is_file():
                errors.append(f"missing {kind}: {path}")
            else:
                check["required_files"].append(dict(kind=kind, path=str(path), sha256=replay.sha256(path)))
        if errors:
            raise ValueError("required files incomplete; no automatic memory/MAD/support rebuild")
        hashes = {r["kind"]: r["sha256"] for r in check["required_files"]}
        if hashes["checkpoint"] != replay.CHECKPOINT_SHA256:
            raise ValueError("checkpoint differs from approved frozen weights")
        selected = replay.read_csv(paths["diagnostic"] / "selected_cases.csv")
        selected_cases = replay.select_cases(replay.read_csv(paths["results"] / "sample_scores.csv"), selected)
        by_id = {r["sample_id"]: r for r in selected_cases}
        # The existing selected_cases.csv is the authority, including its order.
        cases = [dict(by_id[r["sample_id"]]) for r in selected
                 if set(r.get("selection_reason", "").split(";")) & {"top10_union_normal", "focus_anomaly"}]
        required_scope = tuple((r["sample_id"], r["relative_path"]) for r in cases)
        if replay.verified_replay_scope(paths["replay"], paths["results"], required_cases=required_scope) is None:
            raise ValueError("verified replay12 history required")
        for case in cases:
            raw = paths["replay"] / f"raw_cache/heldout/background/{case['sample_id']}.npz"
            check["required_files"].append(dict(kind="historical_raw_"+case["sample_id"], path=str(raw), sha256=replay.sha256(raw)))
        historical = json.loads((paths["replay"] / "replay_inputs.json").read_text(encoding="utf-8"))["identity"]
        state = json.loads((paths["replay"] / "replay_status.json").read_text(encoding="utf-8"))
        complete = json.loads((paths["results"] / "complete.json").read_text(encoding="utf-8"))
        manifest = json.loads((paths["results"] / "memory_manifest.json").read_text(encoding="utf-8"))
        if (complete.get("status") != "complete" or complete.get("protocol_id") != replay.PROTOCOL_ID
                or manifest.get("protocol_id") != replay.PROTOCOL_ID or manifest["builder_input_count"] != 521
                or manifest["host_config"]["layers"] != list(replay.LAYERS)):
            raise ValueError("formal protocol/memory manifest differs")
        replay.check_mad(paths["mad"])
        protocol = state["query_protocol"]
        expected = dict(brightness_factor=1, resize_factor=.625, patch_size=640, patch_overlap=128,
                        model_patch_size=16, evaluation_downscale=4, dino_training=False, trainable_parameters=0)
        if protocol != expected:
            raise ValueError("frozen query protocol mismatch")
        # Logging changes scoring.py. Keep both identities, never update history.
        for key in (*[k for k in SOURCE_CODES if k != "code_scoring"], "official_config", "official_model", "dinov3_hub"):
            if hashes[key] != historical["input_sha256"][key]:
                raise ValueError(f"frozen extraction/host input changed: {key}")
        if historical["input_sha256"].get("memory") and hashes["memory"] != historical["input_sha256"]["memory"]:
            raise ValueError("original Source memory changed")
        diagnosis = json.loads((paths["diagnostic"] / "diagnostic_status.json").read_text(encoding="utf-8"))
        verification = json.loads((paths["diagnostic"] / "verification.json").read_text(encoding="utf-8"))
        if (diagnosis["status"] != "planned_subset_complete" or not verification["formal_results_unchanged"]
                or not verification["planned_anomaly_spatial_complete"]):
            raise ValueError("original spatial diagnostic incomplete")
        all_support = replay.select_support(replay.read_csv(paths["results"] / "support_manifest.csv"), cases)
        supports = [dict(r) for r in all_support if r["branch"] == "target_matched"]
        prior = {r["relative_path"]: r for r in replay.read_csv(paths["replay"] / "support_provenance.csv")}
        for row in supports:
            path = paths["support"] / f"{int(row['support_index']):02d}.npz"
            replay.check_support_header(path, row["relative_path"])
            checksum = replay.sha256(path)
            if checksum != prior[row["relative_path"]]["sha256"]:
                raise ValueError("Target support cache differs from original replay provenance")
            row.update(support_index=int(row["support_index"]), feature_sha256=checksum, feature_path=str(path))
            check["required_files"].append(dict(kind=f"support_{row['support_index']}", path=str(path), sha256=checksum))
        # Full evaluation membership is fixed by formal CSV, not file subset IDs.
        score_rows = replay.read_csv(paths["results"] / "sample_scores.csv")
        source = {r["relative_path"]: r for r in score_rows if r["domain"] == "background" and r["method"] == replay.METHODS[0]}
        union = {r["relative_path"]: r for r in score_rows if r["domain"] == "background" and r["method"] == replay.METHODS[1]}
        samples = [dict(sample_id=r["sample_id"], relative_path=path, label=int(r["label"]),
                        source_score=float(r["image_score"]), union_score=float(union[path]["image_score"])) for path, r in sorted(source.items())]
        if (Counter(r["label"] for r in samples) != {0: 85, 1: 212} or len({r["sample_id"] for r in samples}) != 297
                or {r["relative_path"] for r in supports} & set(source)):
            raise ValueError("held-out 85/212 scope or original eight exclusions changed")
        check["samples"] = cases if args.scope == "twelve" else samples
        check["selected_cases"] = cases
        index = trace_data_index(paths["data"])
        check["paths"]["data"] = str(index.root)
        paths["data"] = index.root
        guard_output(paths)
        image_hashes = {}
        gt_hashes = {}
        try:
            for row in [*check["samples"], *supports]:
                image = index.describe(row["relative_path"])
                old = historical["image_sha256"].get(row["relative_path"])
                if old and old != image["sha256"]:
                    raise ValueError("original query/support image changed")
                if index.ids and row.get("sample_id") and index.ids[row["relative_path"]] != row["sample_id"]:
                    raise ValueError("full dataset index disagrees with formal sample ID")
                row.update(image_height=image["height"], image_width=image["width"])
                image_hashes[row["relative_path"]] = image["sha256"]
            # Only presence here: GT pixels are decoded after all query capture.
            from DINOv3.MADEqual.aebad_s_union_diagnostics.analysis import gt_path
            missing = [gt_path(r["relative_path"]) for r in check["samples"] if r["label"] and gt_path(r["relative_path"]) not in index.names]
            if missing:
                raise ValueError(f"required GT missing for later verification/analysis: {missing[:3]}")
            import hashlib
            for row in check["samples"]:
                if not row["label"]:
                    continue
                relative = gt_path(row["relative_path"])
                handle = index.zip.open(index.prefix+relative) if index.zip else (index.root/relative).open("rb")
                with handle:
                    h = hashlib.sha256()
                    for chunk in iter(lambda: handle.read(4*1024*1024), b""):
                        h.update(chunk)
                gt_hashes[relative] = h.hexdigest()  # bytes only; never decode GT before predictions
        finally:
            index.close()
        environments = sorted(paths["replay"].glob("environment_*.json"))
        environment_path = environments[-1]
        environment = json.loads(environment_path.read_text(encoding="utf-8"))
        check["required_files"].append(dict(kind="replay_environment", path=str(environment_path), sha256=replay.sha256(environment_path)))
        frozen = dict(protocol_id=replay.PROTOCOL_ID, query_protocol=protocol,
                      formal_sha256={k: hashes[k] for k in ("sample_scores.csv", "support_manifest.csv", "complete.json", "memory_manifest.json", "source_mad_state.npz")},
                      input_code_sha256={r["kind"]: r["sha256"] for r in check["required_files"] if r["kind"].startswith("code_") or r["kind"] in ("official_config", "official_model", "dinov3_hub")},
                      historical_scoring_sha256=historical["input_sha256"]["code_scoring"],
                      memory_sha256=hashes["memory"], checkpoint_sha256=hashes["checkpoint"],
                      selected_cases_sha256=hashes["diagnostic_selected_cases.csv"],
                      replay_inputs_sha256=hashes["replay_replay_inputs.json"],
                      precision={k: environment[k] for k in ("float32_matmul_precision", "cuda_matmul_allow_tf32", "cudnn_allow_tf32")},
                      supports=[dict(support_index=r["support_index"], relative_path=r["relative_path"],
                                     feature_sha256=r["feature_sha256"], image_sha256=image_hashes[r["relative_path"]]) for r in supports])
        check.update(frozen=frozen, identity=digest(frozen), image_sha256=image_hashes, supports=supports,
                     gt_sha256=gt_hashes, query_protocol=protocol, replay_environment=environment)
        if args.scope == "background":
            if a_inputs["identity"] != check["identity"]:
                raise ValueError("stage A frozen inputs/code differ; use a new verified stage A")
            if any(image_hashes[r["relative_path"]] != a_inputs["image_sha256"][r["relative_path"]] for r in cases):
                raise ValueError("stage A query images changed")
            if any(gt_hashes[k] != v for k, v in a_inputs["gt_sha256"].items()):
                raise ValueError("stage A GT input identity changed")
            check.update(stage_a_inputs=a_inputs, stage_a_status=a_state)
        if args.mode == "capture" and paths["data"].is_file():
            raise ValueError("capture requires extracted data; ZIP is read-only preflight/analyze input")
    except (OSError, ValueError, KeyError, IndexError, RuntimeError, ImportError) as error:
        errors.append(f"{type(error).__name__}: {error}")
    check["status"] = "filesystem_preflight_passed" if not errors else "blocked"
    return check
