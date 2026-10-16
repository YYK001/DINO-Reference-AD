"""Default preflight; explicit capture and offline analyze. No eager Torch imports."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from DINOv3.MADEqual.aebad_s_target_fewshot import replay12 as replay
from .inputs import TOOL, digest, guard_output, paths_for, preflight, read_capture, verify_files


def verify_image_files(data: Path, image_hashes: dict[str, str]) -> None:
    from DINOv3.MADEqual.aebad_s_target_support_trace.runner import trace_data_index
    index = trace_data_index(data)
    try:
        for relative, expected in image_hashes.items():
            if index.describe(relative)["sha256"] != expected:
                raise ValueError(f"query/support image changed: {relative}")
    finally:
        index.close()


def audit(rows: list[dict], sample_id: str, kind: str, actual: Any, expected: Any,
          *, score: bool = False, **extra: Any) -> bool:
    from .analysis import comparison
    result = comparison(actual, expected, score=score)
    rows.append(dict(sample_id=sample_id, check_kind=kind, distance_atol=1e-7, distance_rtol=1e-6,
                     score_atol=1e-5, **result, **extra))
    return result["verified"]


def formal_baselines(results: Path, samples: list[dict]) -> dict[str, dict[str, dict]]:
    """Read the protected formal CSV, not scores embedded in capture metadata."""
    from collections import Counter
    import math
    groups: dict[str, dict[str, dict]] = {"Source": {}, "K8": {}}
    methods = dict(zip(replay.METHODS, groups, strict=True))
    for row in replay.read_csv(results / "sample_scores.csv"):
        if row["domain"] != "background" or row["method"] not in methods:
            continue
        group = groups[methods[row["method"]]]
        sid = row["sample_id"]
        if sid in group or not math.isfinite(float(row["image_score"])):
            raise ValueError("duplicate/nonfinite formal baseline image score")
        group[sid] = dict(relative_path=row["relative_path"], label=int(row["label"]),
                          image_score=float(row["image_score"]))
    if (set(groups["Source"]) != set(groups["K8"])
            or any(Counter(r["label"] for r in g.values()) != {0: 85, 1: 212} for g in groups.values())):
        raise ValueError("formal baselines require the identical 85/212 background set")
    for sid in groups["Source"]:
        source, union = groups["Source"][sid], groups["K8"][sid]
        if (source["relative_path"], source["label"]) != (union["relative_path"], union["label"]):
            raise ValueError("formal Source/K8 image identity mismatch")
    for sample in samples:
        for group in groups.values():
            row = group.get(sample["sample_id"])
            if row is None or (row["relative_path"], row["label"]) != (sample["relative_path"], sample["label"]):
                raise ValueError("analyze sample identity differs from formal CSV")
    return groups


def verify_formal_ranking(summaries: list[dict], formal: dict, rows: list[dict]) -> None:
    from sklearn.metrics import average_precision_score, roc_auc_score
    for branch in ("Source", "K8"):
        summary = next(r for r in summaries if r["branch"] == branch)
        reference = formal[branch]
        ids = sorted(reference)
        labels = [reference[i]["label"] for i in ids]
        scores = [reference[i]["image_score"] for i in ids]
        if (summary["image_count"], summary["normal_count"], summary["anomaly_count"]) != (297, 85, 212):
            raise ValueError("full-domain baseline summary requires all 297 held-out images")
        for metric, expected in (("image_AUROC", roc_auc_score(labels, scores)),
                                 ("image_AUPR", average_precision_score(labels, scores))):
            actual = float(summary[metric])
            audit(rows, "background", "formal_baseline_summary", actual, float(expected), score=True,
                  branch=branch, metric=metric, actual_score=actual, expected_score=float(expected))


def verify_historical(sample: dict, arrays: dict, metadata: dict, maps: dict,
                      backend: Any, paths: dict, rows: list[dict]) -> None:
    from .analysis import branch_raws
    from DINOv3.MADEqual.aebad_s_union_diagnostics.spatial import interpolation_stencil
    raws = branch_raws(arrays, tuple(metadata["grid"]))
    sid = sample["sample_id"]
    audit(rows, sid, "source_image_score", maps["Source"].max(), sample["source_score"], score=True)
    audit(rows, sid, "K8_image_score", maps["K8"].max(), sample["union_score"], score=True)
    selected = {r["sample_id"] for r in replay.read_csv(paths["diagnostic"] / "selected_cases.csv")
                if set(r.get("selection_reason", "").split(";")) & {"top10_union_normal", "focus_anomaly"}}
    if sid not in selected:
        return
    case = dict(sample)
    historical, shape, _ = replay.validate_raw(paths["replay"] / f"raw_cache/heldout/background/{sid}.npz", case)
    if list(shape) != metadata["output_shape"]:
        raise ValueError("historical output geometry mismatch")
    target = arrays["target_distance"].min(axis=0)
    physical = __import__("numpy").empty_like(raws["Source"])
    physical.reshape(4, -1)[:, arrays["positions"]] = target
    for name, actual, expected in (("Source_distance", raws["Source"], historical["official_raw"]),
                                   ("Target_K8_distance", physical, historical["target_support_8"]),
                                   ("union_layer_min", raws["K8"], historical["target_raw_8"])):
        audit(rows, sid, name, actual, expected)
    old_maps = {"Source": backend.runner._method_map(historical, replay.METHODS[0], backend.mad, shape),
                "K8": backend.runner._method_map(historical, replay.METHODS[1], backend.mad, shape)}
    old_peaks = [r for r in replay.read_csv(paths["diagnostic"] / "case_peaks.csv") if r["sample_id"] == sid]
    if {r["peak"] for r in old_peaks} != {"source_peak", "union_peak"}:
        raise ValueError("archived full peaks missing")
    old_layers = replay.read_csv(paths["diagnostic"] / "case_peak_layers.csv")
    from DINOv3.MADEqual.aebad_s_target_support_trace.analysis import weighted_layers, position_patches
    values = dict(official_raw=raws["Source"], target_support_8=physical, target_raw_8=raws["K8"])
    for old in old_peaks:
        y, x = int(old["pixel_y"]), int(old["pixel_x"])
        for branch, key in (("Source", "source_score_at_pixel"), ("K8", "union_score_at_pixel")):
            audit(rows, sid, "original_peak_response", maps[branch][y, x], float(old[key]), score=True,
                  peak=old["peak"], branch=branch)
        position = dict(output_y=y, output_x=x, output_height=shape[0], output_width=shape[1],
                        sample_id=sid, position_id=old["peak"], roles=old["peak"])
        patches = position_patches([position], tuple(metadata["grid"]))
        contributions = weighted_layers(values, patches, backend.calibrated, backend.mad.medians, backend.mad.scales)
        for offset, layer in enumerate(replay.LAYERS):
            record = [r for r in old_layers if r["sample_id"] == sid and r["peak"] == old["peak"] and int(r["layer"]) == layer]
            if len(record) != 1:
                raise ValueError("archived layer contributions incomplete")
            audit(rows, sid, "original_peak_layer_contribution",
                  [contributions[k][offset] for k in ("official_raw", "target_support_8", "target_raw_8")],
                  [float(record[0][k]) for k in ("source_mad_contribution", "target_only_mad_contribution", "union_mad_contribution")],
                  score=True, peak=old["peak"], layer=layer)
        stencil = interpolation_stencil((y, x), tuple(metadata["grid"]), shape)
        for branch in ("Source", "K8"):
            calibrated = backend.calibrated(raws[branch], backend.mad.medians, backend.mad.scales)
            response = sum(w*float(calibrated[:, py, px].mean()) for py, px, w in stencil)
            audit(rows, sid, "same_call_stencil_response", response, maps[branch][y, x], score=True,
                  peak=old["peak"], branch=branch)
        # Compare new maps at the old position; a near-tied new maximum may move.
        # This does not assert recovery of an unrecorded historical NN index.
        for branch in ("Source", "K8"):
            audit(rows, sid, "original_map_max", maps[branch].max(), old_maps[branch].max(), score=True, branch=branch)


def verify_instances(check: dict, records: list[dict], output: Path, backend: Any,
                     rows: list[dict]) -> None:
    """Decode GT only AFTER prediction records for the entire scope are written."""
    from .analysis import load_raw, branch_raws, branch_maps, spatial_rows
    from DINOv3.MADEqual.aebad_s_target_support_trace.runner import trace_data_index, trace_image
    from DINOv3.MADEqual.aebad_s_union_diagnostics.analysis import gt_path
    import numpy as np
    paths = {k: Path(v) for k, v in check["paths"].items()}
    selected_ids = {s["sample_id"] for s in check["selected_cases"] if s["label"]}
    index = trace_data_index(paths["data"])
    old = replay.read_csv(paths["diagnostic"] / "defect_instances.csv")
    by_id = {r["sample_id"]: r for r in records}
    try:
        for sample in check["samples"]:
            if sample["sample_id"] not in selected_ids:
                continue
            record = by_id[sample["sample_id"]]
            arrays, meta = load_raw(output / record["file"], identity=check["identity"], sample=sample, checksum=record["sha256"])
            maps = branch_maps(branch_raws(arrays, tuple(meta["grid"])), tuple(meta["output_shape"]), backend)
            mask = np.asarray(trace_image(index, gt_path(sample["relative_path"]), "L")) > 0
            if mask.shape != (sample["image_height"], sample["image_width"]):
                raise ValueError("GT/query image geometry differs")
            _, instances = spatial_rows(sample, maps, mask, tuple(meta["grid"]), backend)
            historical = [r for r in old if r["sample_id"] == sample["sample_id"]]
            current = [r for r in instances if r["branch"] == "K8"]
            if {int(r["instance_id"]) for r in historical} != {int(r["instance_id"]) for r in current}:
                raise ValueError("historical GT instance scope differs")
            for r in instances:
                if r["branch"] not in ("Source", "K8"):
                    continue
                reference = next(h for h in historical if int(h["instance_id"]) == r["instance_id"])
                if any(int(reference[k]) != r[k] for k in ("area_original_pixels", "bbox_y0", "bbox_x0", "bbox_y1_exclusive", "bbox_x1_exclusive")):
                    raise ValueError("historical GT component geometry differs")
                prefix = "source" if r["branch"] == "Source" else "union"
                for field in ("instance_max", "margin"):
                    if r[field] is None or reference.get(prefix+"_"+field, "") == "":
                        raise ValueError("required instance/margin evidence unavailable")
                    audit(rows, sample["sample_id"], "original_instance_"+field, r[field],
                          float(reference[prefix+"_"+field]), score=True, branch=r["branch"], instance_id=r["instance_id"])
    finally:
        index.close()


def capture(args: Any, check: dict) -> int:
    if check["errors"]:
        return 2
    import numpy as np
    import torch
    from .analysis import CaptureLog, FORMAT, save_raw, load_raw, branch_raws, branch_maps
    from DINOv3.MADEqual.aebad_s_target_fewshot import runner as original
    from DINOv3.MADEqual.robustad_target_fewshot_p0.scoring import observe_fewshot_pil
    from DINOv3.MADEqual.robustad_target_fewshot_p0.support import extract_or_load_support
    from DINOv3.MADEqual.aebad_s_union_diagnostics.spatial import MapBackend
    from DINOv3.MADEqual.aebad_s_target_support_trace.analysis import image_geometry
    from DINOv3.MADEqual.aebad_s_target_support_trace.runner import trace_data_index, trace_image
    from DINOv3.relation_reliability.datasets import Record
    from DINOv3.relation_reliability.superadd_host import construct_exact_host, install_import_paths, preprocessed_spatial_shape
    from unittest.mock import patch
    paths = {k: Path(v) for k, v in check["paths"].items()}
    output = paths["output"]
    status = dict(tool=TOOL, scope=args.scope, stage="capture", identity=check["identity"], status="running",
                  records=[], query_forward_calls=0, support_forward_calls=0, historical_unique_indices_recovered=False,
                  source_provenance="unavailable_memory_row_only", within_support_patch_uniqueness="unknown")
    verification: list[dict] = []
    started = time.perf_counter()
    replay.atomic_json(output / "status.json", status)
    try:
        verify_files(check["required_files"])
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("explicit capture requires CUDA; no model CPU fallback")
        device = torch.device("cuda", device.index if device.index is not None else 0)
        torch.cuda.set_device(device)
        original._seed_everything()
        precision = dict(float32_matmul_precision=torch.get_float32_matmul_precision(),
                         cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                         cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
        if precision != check["frozen"]["precision"]:
            raise ValueError("runtime precision differs; not modified automatically")
        software = {name: importlib.metadata.version(name) for name in ("numpy", "torch", "torchvision", "Pillow", "scipy")}
        if args.scope == "background":
            previous_environment = json.loads((paths["stage_a"] / "environment.json").read_text(encoding="utf-8"))
            if previous_environment["software"] != software or previous_environment["precision"] != precision:
                raise ValueError("stage A reuse requires identical software/precision")
        replay.atomic_json(output / "environment.json", dict(python=sys.version, platform=platform.platform(), software=software,
            precision=precision, cuda=torch.version.cuda, device_name=torch.cuda.get_device_name(device),
            replay_software=check["replay_environment"]["software"],
            version_differences={k: [check["replay_environment"]["software"].get(k), v] for k, v in software.items()
                                 if check["replay_environment"]["software"].get(k) != v}))
        holder = SimpleNamespace()
        if not original._load_memory(paths["memory"], holder, source_count=521, device=device):
            raise FileNotFoundError("Source memory missing; rebuild forbidden")
        backend = MapBackend(paths["mad"])
        install_import_paths(paths["project"], paths["official"], paths["dinov3"])
        from industrial import model as official_model
        if Path(official_model.__file__).resolve() != (paths["official"] / "tracks/industrial/src/industrial/model.py").resolve():
            raise ValueError("unexpected industrial.model import")
        def checkpoint_path(name: str) -> Path:
            if name != "dinov3_vith16plus":
                raise ValueError("unexpected backbone")
            return paths["checkpoint"]
        with patch.object(official_model, "get_weights_path", checkpoint_path):
            model, _ = construct_exact_host(paths["official"], paths["dinov3"], paths["checkpoint"], replay.CHECKPOINT_SHA256)
        model.backbone.dino.eval()
        model.backbone.dino.requires_grad_(False)
        actual = dict(brightness_factor=model.preprocessing.min_brightness_factor, resize_factor=model.preprocessing.resize_factor,
                      patch_size=model.patch_exec.patch_size, patch_overlap=model.patch_exec.patch_overlap,
                      model_patch_size=model.patch_exec.model_patch_size, evaluation_downscale=model.evaluation_downscale,
                      dino_training=model.backbone.dino.training,
                      trainable_parameters=sum(p.requires_grad for p in model.backbone.dino.parameters()))
        if actual != check["query_protocol"] or model.preprocessing.max_brightness_factor != 1:
            raise ValueError("constructed query host differs from frozen protocol")
        model.prototype_embeddings = holder.prototype_embeddings
        del holder
        features, supports = [], []
        for row in check["supports"]:
            # Missing features already block preflight. Passing model=None also
            # prevents using this loader as an implicit extraction fallback.
            record = Record(path=str(paths["data"] / row["relative_path"]), label=0, mask_path=None, category="AeBAD-S",
                            dataset="aebad_s", domain="background", shift="support", role="support", mask_capability="not_evaluation")
            if replay.sha256(Path(row["feature_path"])) != row["feature_sha256"]:
                raise ValueError("support feature changed after preflight")
            feature = extract_or_load_support(Path(row["feature_path"]), record, dataset_root=paths["data"],
                                             model=None, device=device, protocol_id=replay.PROTOCOL_ID)
            processed = preprocessed_spatial_shape(model, row["image_height"], row["image_width"])
            geometry = image_geometry(model, row["image_height"], row["image_width"], feature.grid, processed)
            supports.append(dict(support_index=row["support_index"], relative_path=row["relative_path"],
                grid=list(feature.grid), geometry=geometry, image_sha256=check["image_sha256"][row["relative_path"]],
                feature_sha256=row["feature_sha256"]))
            features.append(feature)
        target_tensors = original._support_tensors(features, device=device)
        features.clear()
        inputs = dict(tool=TOOL, scope=args.scope, frozen=check["frozen"], identity=check["identity"],
            samples=check["samples"], selected_cases=check["selected_cases"], supports=supports,
            image_sha256=check["image_sha256"], gt_sha256=check["gt_sha256"], distance_unit="euclidean_over_1280_once", dtype="float32",
            axes=dict(source_distance="layer,compact_row", target_distance="support,layer,compact_row",
                      source_index="layer,compact_row", target_index="support,layer,compact_row", positions="compact_row_to_physical_flat"),
            historical_code_sha256=check["frozen"]["historical_scoring_sha256"],
            current_logging_code_sha256=check["frozen"]["input_code_sha256"], protected_files=check["required_files"])
        replay.atomic_json(output / "inputs.json", inputs)
        prior = {r["sample_id"]: r for r in check.get("stage_a_status", {}).get("records", [])}
        index = trace_data_index(paths["data"])
        try:
            for sample in check["samples"]:
                sid = sample["sample_id"]
                path = output / "raw_cache" / f"{sid}.npz"
                if sid in prior:
                    old = paths["stage_a"] / prior[sid]["file"]
                    arrays, meta = load_raw(old, identity=check["identity"], sample=sample, checksum=prior[sid]["sha256"])
                    if meta["supports"] != supports:
                        raise ValueError("stage A support geometry differs")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with old.open("rb") as source, path.open("xb") as destination:
                        shutil.copyfileobj(source, destination)
                    action = "reuse_verified_stage_A"
                else:
                    if index.describe(sample["relative_path"])["sha256"] != check["image_sha256"][sample["relative_path"]]:
                        raise ValueError("query image changed after preflight")
                    image = trace_image(index, sample["relative_path"], "RGB")
                    log = CaptureLog()
                    status["query_forward_calls"] += 1
                    observed = observe_fewshot_pil(model, image, official_banks=model.prototype_embeddings,
                        source_support=None, target_support=target_tensors, device=device,
                        nearest_observer=log.nearest, geometry_observer=log.geometry)
                    arrays = log.arrays()
                    geometry = image_geometry(model, image.height, image.width, log.grid, log.processed_shape)
                    meta = dict(format=FORMAT, identity=check["identity"], sample_id=sid, relative_path=sample["relative_path"],
                        image_sha256=check["image_sha256"][sample["relative_path"]], grid=list(log.grid),
                        output_shape=[image.height//model.evaluation_downscale, image.width//model.evaluation_downscale],
                        geometry=geometry, supports=supports, source_rows=[int(model.prototype_embeddings[l].shape[0]) for l in replay.LAYERS],
                        source_provenance="unavailable_memory_row_only", within_support_patch_uniqueness="unknown")
                    save_raw(path, arrays, meta)
                    raws = branch_raws(arrays, log.grid)
                    # This proves the log is from the calls that produced scoring,
                    # rather than a later selected-patch search.
                    audit(verification, sid, "same_call_source", raws["Source"], observed.official_raw)
                    audit(verification, sid, "same_call_K8", raws["K8"], observed.target_raw[8])
                    del observed, log, image
                    action = "same_call_full_grid_capture"
                status["records"].append(dict(sample_id=sid, file=f"raw_cache/{sid}.npz", sha256=replay.sha256(path), action=action))
                maps = branch_maps(branch_raws(arrays, tuple(meta["grid"])), tuple(meta["output_shape"]), backend)
                verify_historical(sample, arrays, meta, maps, backend, paths, verification)
                del arrays, maps
                replay.write_csv(output / "capture_verification.csv", verification)
                replay.atomic_json(output / "status.json", status)
                print(f"[LOO capture] {sid} {action}", flush=True)
        finally:
            index.close()
        # All prediction records are now on disk. Release GPU banks/backbone
        # before the first GT decoding. The original CPU map backend remains.
        del model, target_tensors
        torch.cuda.empty_cache()
        verify_instances(check, status["records"], output, backend, verification)
        if args.scope == "background":
            from .analysis import load_raw, branch_raws, branch_maps, image_rows, ranking_rows
            current = []
            for sample, record in zip(check["samples"], status["records"], strict=True):
                arrays, meta = load_raw(output / record["file"], identity=check["identity"], sample=sample, checksum=record["sha256"])
                maps = branch_maps(branch_raws(arrays, tuple(meta["grid"])), tuple(meta["output_shape"]), backend)
                current.extend(image_rows(sample, maps))
            top10 = [s["sample_id"] for s in check["selected_cases"] if s["label"] == 0]
            ranking, _ = ranking_rows(current, top10)
            from sklearn.metrics import roc_auc_score, average_precision_score
            labels = [s["label"] for s in check["samples"]]
            for branch, score_key in (("Source", "source_score"), ("K8", "union_score")):
                row = next(r for r in ranking if r["branch"] == branch)
                archived_scores = [s[score_key] for s in check["samples"]]
                audit(verification, "background", "formal_image_ranking", [row["image_AUROC"], row["image_AUPR"]],
                      [roc_auc_score(labels, archived_scores), average_precision_score(labels, archived_scores)], score=True, branch=branch)
            del current, arrays, maps
        verify_files(check["required_files"])
        verify_image_files(paths["data"], check["image_sha256"])
        verify_image_files(paths["data"], check["gt_sha256"])
        if replay.verified_replay_scope(paths["replay"], paths["results"],
            required_cases=tuple((r["sample_id"], r["relative_path"]) for r in check["selected_cases"])) is None:
            raise ValueError("replay history disappeared")
        replay.write_csv(output / "capture_verification.csv", verification)
        passed = bool(verification) and all(r["verified"] for r in verification)
        status.update(status="capture_verified" if passed else "capture_not_reproduced", predicted_images=len(status["records"]),
                      numerical_checks=len(verification), failed_checks=sum(not r["verified"] for r in verification),
                      protected_inputs_unchanged=True, inputs_sha256=replay.sha256(output / "inputs.json"),
                      verification_sha256=replay.sha256(output / "capture_verification.csv"), seconds=time.perf_counter()-started)
        replay.atomic_json(output / "status.json", status)
        return 0 if passed else 3
    except Exception as error:
        if verification:
            replay.write_csv(output / "capture_verification.csv", verification)
        status.update(status="capture_failed", error=f"{type(error).__name__}: {error}", seconds=time.perf_counter()-started)
        replay.atomic_json(output / "status.json", status)
        raise


def write_report(output: Path, status: dict, summaries: list[dict], supports: list[dict]) -> None:
    lines = ["# 冻结 Target-normal K8 union：逐支持移除诊断", "", "## 已确认的数值事实", "",
        f"范围：{status['scope']}；已处理 {status.get('analyzed_images', 0)} 张；状态：{status['status']}。",
        f"本次离线重建的正式基线复核通过：{status.get('formal_baselines_reproduced', False)}。复核失败时下列统计仅保留现场，不作为已验证干预结果。",
        "原八张支持始终排除于评价集。九个干预分支为 K8 和 minus_1..8，Source 另作参照。",
        "本次对应来自生成评分的同一次完整网格 Exact 检索；没有声称恢复未记录的历史唯一索引。", "",
        "## 干预后发生的变化", "",
        "| 分支 | Image AUROC | Image AUPR（AP） | 排序对改善/变差/不变 | 正常 p95 |", "| --- | ---: | ---: | --- | ---: |"]
    for row in summaries:
        lines.append(f"| {row['branch']} | {row['image_AUROC']:.9f} | {row['image_AUPR']:.9f} | "
                     f"{row['improved_pairs_vs_K8']}/{row['worsened_pairs_vs_K8']}/{row['unchanged_pairs_vs_K8']} | {row['normal_score_p95']:.9f} |")
    lines += ["", "完整全域结论只适用于 background 的 85 张正常与 212 张异常；十二案例的子集排序不能代替全域结果。",
              "逐层接管/距离下降先在每图内除以 patch 数，再跨图等权平均；大图不隐式占更大权重。",
              "实例 margin 数量统计按实例计数，并保留标注碎片；不设置新的小缺陷阈值。", "",
              "## 人工判读", "", "所有语义字段为 unreviewed。GT 补集没有自动标为背景，没有自动生成错误语义匹配结论。", "",
              "## 待验证解释", "", "移除后距离升高是集合缩小的必然结果；缺陷 margin 恢复不代表删除该支持整体更好。",
              "移除无影响可能来自其他支持冗余。这是固定集合下的条件性移除效应，不是支持图的全部价值。",
              "不按测试 GT 选择最佳七张支持，不将诊断改善包装成新方法结果，也不预设统一机制。", "",
              "## 未验证或不可获得的信息", "", "Source 仅有 memory 行索引，图像/patch provenance 不可获得。",
              "图内第二近 patch 距离未记录，patch 唯一性为 unknown；跨支持图间隔不能证明图内唯一性。",
              "近似并列仅是数值不确定性标记。投影 token 单元不是完整感受野。",
              f"缺失 GT：{status.get('missing_gt_ids', [])}；未完成空间分析：{status.get('incomplete_spatial_ids', [])}。", "",
              "## 文件索引", "", "raw_cache/*.npz 位于 capture 目录；本 analyze 目录不重复保存特征或全域高分辨率地图。",
              "inputs.json/status.json：身份、范围及状态；capture_verification.csv：capture 原数值复核记录副本。",
              "per_image_loo.csv：每图九分支和 Source 参照；per_support_effects.csv：逐支持汇总。",
              "support_layer_per_image.csv/support_layer_summary.csv：逐层逐图和等权跨图距离/接管统计。",
              "defect_instance_loo.csv：实例、环带、margin；peak_positions.csv：参考位置和重新取峰值后的响应。",
              "peak_patch_correspondence.csv/position_support_effects.csv：十二案例 stencil 对应和同位置逐层移除作用。",
              "analysis_verification.csv：stencil 响应复核；figures/：十二案例共用色标及独立差分色标。", "",
              "| manifest 编号 | support 路径 |", "| ---: | --- |"]
    lines.extend(f"| {r['support_index']} | {r['relative_path']} |" for r in supports)
    (output / "diagnostic_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def analyze(args: Any, check: dict) -> int:
    import numpy as np
    from .analysis import (ALL_BRANCHES, BRANCHES, load_raw, branch_raws, branch_maps, image_rows,
        spatial_rows, correspondence_rows, support_layer_rows, ranking_rows)
    from DINOv3.MADEqual.aebad_s_union_diagnostics.spatial import MapBackend
    from DINOv3.MADEqual.aebad_s_union_diagnostics.analysis import gt_path
    from DINOv3.MADEqual.aebad_s_target_support_trace.runner import trace_data_index, trace_image
    paths = {k: Path(v) for k, v in check["paths"].items()}
    output, root = paths["output"], paths["capture"]
    inputs, captured = read_capture(root)
    formal = formal_baselines(paths["results"], inputs["samples"])
    backend = MapBackend(paths["mad"])
    index = trace_data_index(paths["data"])
    status = dict(tool=TOOL, stage="analyze", scope=inputs["scope"], status="running", identity=inputs["identity"],
                  analyzed_images=0, missing_gt_ids=[], incomplete_spatial_ids=[], model_inference=False)
    replay.atomic_json(output / "inputs.json", dict(tool=TOOL, capture_dir=str(root), capture_inputs_sha256=replay.sha256(root / "inputs.json"),
                                                 capture_status_sha256=replay.sha256(root / "status.json"), identity=inputs["identity"]))
    shutil.copyfile(root / "capture_verification.csv", output / "capture_verification.csv")
    tables: dict[str, list] = {k: [] for k in ("per_image_loo", "defect_instance_loo", "peak_positions", "peak_patch_correspondence",
                                              "support_layer_per_image", "analysis_verification", "position_support_effects")}
    selected = {s["sample_id"] for s in inputs["selected_cases"]}
    try:
        for sample, record in zip(inputs["samples"], captured["records"], strict=True):
            arrays, meta = load_raw(root / record["file"], identity=inputs["identity"], sample=sample, checksum=record["sha256"])
            raws = branch_raws(arrays, tuple(meta["grid"]))
            maps = branch_maps(raws, tuple(meta["output_shape"]), backend)
            for branch in ("Source", "K8"):
                actual = float(maps[branch].max())
                expected = formal[branch][sample["sample_id"]]["image_score"]
                audit(tables["analysis_verification"], sample["sample_id"], "formal_baseline_image_score",
                      actual, expected, score=True, branch=branch, actual_score=actual, expected_score=expected)
            sample_image_rows = image_rows(sample, maps)
            tables["per_image_loo"].extend(sample_image_rows)
            tables["support_layer_per_image"].extend(support_layer_rows(sample, arrays))
            mask = None
            spatial_ok = True
            if sample["label"]:
                try:
                    gt_relative = gt_path(sample["relative_path"])
                    if index.describe(gt_relative)["sha256"] != inputs["gt_sha256"][gt_relative]:
                        raise ValueError("GT differs from capture validation identity")
                    mask = np.asarray(trace_image(index, gt_relative, "L")) > 0
                    if mask.shape != (sample["image_height"], sample["image_width"]):
                        raise ValueError("GT/query size mismatch")
                except (OSError, ValueError, KeyError) as error:
                    status["missing_gt_ids"].append(sample["sample_id"])
                    status.setdefault("spatial_errors", []).append(dict(sample_id=sample["sample_id"], error=str(error)))
                    spatial_ok = False
            if spatial_ok:
                peaks, instances = spatial_rows(sample, maps, mask, tuple(meta["grid"]), backend)
                tables["peak_positions"].extend(peaks)
                tables["defect_instance_loo"].extend(instances)
            else:
                peaks, instances = [], []
            spatial_status = "normal_no_defect_GT" if not sample["label"] else (
                "missing_or_invalid_GT" if not spatial_ok else ("empty_GT_annotation" if not mask.any() else "GT_available"))
            for image_row in sample_image_rows:
                image_row["spatial_evidence_status"] = spatial_status
            if spatial_status == "empty_GT_annotation":
                status["incomplete_spatial_ids"].append(sample["sample_id"])
            if sample["sample_id"] in selected:
                if index.describe(sample["relative_path"])["sha256"] != inputs["image_sha256"][sample["relative_path"]]:
                    raise ValueError("figure query image differs from capture")
                image = trace_image(index, sample["relative_path"], "RGB")
                from .visualization import plot_case
                plot_case(image, mask, maps, output / "figures" / f"{sample['sample_id']}_loo.png")
                correspondence, verification = correspondence_rows(sample, peaks, arrays, meta, maps, raws, backend)
                tables["peak_patch_correspondence"].extend(correspondence)
                tables["analysis_verification"].extend(verification)
                from .visualization import plot_correspondences
                for support in inputs["supports"]:
                    if index.describe(support["relative_path"])["sha256"] != support["image_sha256"]:
                        raise ValueError("support image differs from capture")
                plot_correspondences(image, sample, peaks, correspondence, index, meta, output / "figures")
                from DINOv3.MADEqual.aebad_s_union_diagnostics.spatial import interpolation_stencil
                for y, x in sorted({(p["output_y"], p["output_x"]) for p in peaks}):
                    stencil = interpolation_stencil((y, x), tuple(meta["grid"]), tuple(meta["output_shape"]))
                    for j in range(1, 9):
                        for offset, layer in enumerate(replay.LAYERS):
                            baseline = sum(w*float(raws["K8"][offset, py, px]) for py, px, w in stencil)
                            removed = sum(w*float(raws[f"minus_{j}"][offset, py, px]) for py, px, w in stencil)
                            tables["position_support_effects"].append(dict(sample_id=sample["sample_id"], label=sample["label"],
                                output_y=y, output_x=x, support_index=j, layer=layer, K8_distance=baseline,
                                removed_distance=removed, lost_distance_drop=removed-baseline,
                                mad_response_change=(removed-baseline)/float(backend.mad.scales[offset]), query_region="unreviewed"))
            if not spatial_ok:
                status["incomplete_spatial_ids"].append(sample["sample_id"])
            status["analyzed_images"] += 1
            print(f"[LOO analyze] {sample['sample_id']}", flush=True)
        top10 = [s["sample_id"] for s in inputs["selected_cases"] if s["label"] == 0]
        summaries, pair_rows = ranking_rows(tables["per_image_loo"], top10)
        if inputs["scope"] == "background":
            verify_formal_ranking(summaries, formal, tables["analysis_verification"])
        per_pairs = {(r["sample_id"], r["branch"]): r for r in pair_rows}
        for row in tables["per_image_loo"]:
            row.update(per_pairs[(row["sample_id"], row["branch"])])
        effects = []
        for j in range(1, 9):
            row = dict(next(r for r in summaries if r["branch"] == f"minus_{j}"), support_index=j,
                       support_relative_path=inputs["supports"][j-1]["relative_path"])
            margins = [r["margin_change_vs_K8"] for r in tables["defect_instance_loo"]
                       if r["branch"] == f"minus_{j}" and r["margin_change_vs_K8"] is not None]
            row.update(instances_with_available_margin=len(margins), margin_improved=sum(m>1e-5 for m in margins),
                       margin_worsened=sum(m<-1e-5 for m in margins), margin_unchanged=sum(abs(m)<=1e-5 for m in margins),
                       margin_change_min=min(margins) if margins else None, margin_change_max=max(margins) if margins else None,
                       margin_change_p50=float(np.percentile(margins, 50)) if margins else None,
                       margin_change_p10=float(np.percentile(margins, 10)) if margins else None,
                       margin_change_p90=float(np.percentile(margins, 90)) if margins else None,
                       margin_aggregation="instance_counts_keep_annotation_fragments", margin_zero_tolerance=1e-5)
            effects.append(row)
        summary_layers = []
        for j in range(1, 9):
            for layer in replay.LAYERS:
                for label in (0, 1):
                    group = [r for r in tables["support_layer_per_image"] if r["support_index"] == j and r["layer"] == layer and r["label"] == label]
                    row = dict(support_index=j, layer=layer, label=label, image_count=len(group), aggregation="mean_of_per_image_patch_fractions_equal_image_weight")
                    for key in ("target_strict_win_fraction", "source_strict_win_fraction", "source_target_tie_fraction",
                                "own_target_min_fraction", "own_unique_target_win_fraction", "lost_distance_drop_fraction",
                                "lost_distance_drop_mean", "cross_support_near_tie_fraction"):
                        row[key] = float(np.mean([r[key] for r in group]))
                    summary_layers.append(row)
        for name, rows in tables.items():
            replay.write_csv(output / f"{name}.csv", rows)
        replay.write_csv(output / "per_support_effects.csv", effects)
        replay.write_csv(output / "support_layer_summary.csv", summary_layers)
        replay.write_csv(output / "image_ranking_summary.csv", summaries)
        # Re-read hashes; changes are not adopted as the new trusted cache.
        read_capture(root)
        verify_files(check["required_files"])
        passed = not status["incomplete_spatial_ids"] and all(r["verified"] for r in tables["analysis_verification"])
        status.update(status="analysis_complete" if passed else "partial_evidence",
                      selected_figures=len(selected), branch_count=9, source_reference_separate=True,
                      numerical_failed_checks=sum(not r["verified"] for r in tables["analysis_verification"]),
                      formal_baselines_reproduced=all(r["verified"] for r in tables["analysis_verification"]
                                                      if r.get("check_kind", "").startswith("formal_baseline_")),
                      capture_inputs_unchanged=True, GT_used_only_after_capture=True)
        replay.atomic_json(output / "status.json", status)
        write_report(output, status, summaries, inputs["supports"])
        return 0 if passed else 3
    except Exception as error:
        status.update(status="analysis_failed", error=f"{type(error).__name__}: {error}")
        replay.atomic_json(output / "status.json", status)
        raise
    finally:
        index.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "capture", "analyze"), nargs="?", default="preflight")
    parser.add_argument("--scope", choices=("twelve", "background"), default="twelve")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[3]))
    for name in ("results-dir", "replay-dir", "diagnostic-dir", "official-root", "checkpoint", "memory",
                 "support-cache-dir", "output-dir", "stage-a-dir", "capture-dir"):
        parser.add_argument("--"+name)
    parser.add_argument("--data", default="/root/autodl-tmp/datasets/AeBAD/AeBAD_S")
    parser.add_argument("--dinov3-root", default="/root/autodl-tmp/dinov3")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-write", action="store_true", help="preflight only: print checks without creating files")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.no_write and args.mode != "preflight":
        raise ValueError("--no-write applies only to preflight")
    try:
        check = preflight(args)
    except (OSError, ValueError) as error:
        print(f"PREFLIGHT_BLOCKED: {error}", file=sys.stderr)
        return 2
    output = Path(check["paths"]["output"])
    if not args.no_write:
        output.mkdir(parents=True, exist_ok=True)
        replay.atomic_json(output / "preflight.json", check)
    print(json.dumps(check, ensure_ascii=False, indent=2), flush=True)
    if check["errors"]:
        return 2
    if args.mode == "preflight":
        return 0
    try:
        return capture(args, check) if args.mode == "capture" else analyze(args, check)
    except Exception as error:
        # Also record a missing production backend (e.g. Torch import failure)
        # before that phase has been able to initialize its own progress file.
        status_path = output / "status.json"
        state = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else dict(
            tool=TOOL, scope=args.scope, stage=args.mode)
        state.update(status=args.mode+"_failed", error=f"{type(error).__name__}: {error}")
        replay.atomic_json(status_path, state)
        print(state["error"], file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
