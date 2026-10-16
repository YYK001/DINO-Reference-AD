"""Diagnose AeBAD-S/background using existing CSVs, selected images and raw caches."""
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from DINOv3.relation_reliability.protocol import LAYERS
from .analysis import (
    CACHE_REVISION, FOCUS_PATHS, METHODS, PROTOCOL_ID, SOURCE, UNION, Dataset,
    gt_path, ranking, read_csv, snapshot, validate_inputs, write_csv,
)
from .spatial import (
    MapBackend, components, crop_diagnostics, instance_response, load_cache, mmr,
    montage, peak_rows, plot_maps,
)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def spatial_completion(availability: list[dict[str, Any]], scope: set[str] | None,
                       instance_rows: list[dict[str, Any]], crop_rows: list[dict[str, Any]],
                       mechanism_rows: list[dict[str, Any]]) -> dict[str, Any]:
    required = [row for row in availability if row["label"] and (scope is None or row["sample_id"] in scope)]
    missing_gt, incomplete = [], []
    for sample in required:
        sample_id = sample["sample_id"]
        if sample.get("gt_status") != "available":
            missing_gt.append(sample_id)
            continue
        instances = [row for row in instance_rows if row["sample_id"] == sample_id]
        responses_complete = bool(instances) and all(
            row.get("patch_coverage_status") == "adaptive_max_original_gt"
            and row.get("response_status") in {"available", "instance_empty_at_evaluation_grid"}
            and "ring_area_output_pixels" in row
            and (row["response_status"] == "instance_empty_at_evaluation_grid"
                 or all(key in row for key in ("source_margin", "union_margin", "margin_change")))
            for row in instances)
        crop_methods = {row["method"] for row in crop_rows if row["sample_id"] == sample_id and row.get("gt_status") == "verified"}
        defect_layers = {row["layer"] for row in mechanism_rows if row["sample_id"] == sample_id and row.get("region") == "GT_defect_patches"}
        if not responses_complete or crop_methods != set(METHODS) or defect_layers != set(LAYERS):
            incomplete.append(sample_id)
    return {"required_anomaly_gt_images": len(required), "missing_required_gt_ids": missing_gt,
            "incomplete_anomaly_spatial_ids": incomplete,
            "planned_anomaly_spatial_complete": not missing_gt and not incomplete}


def diagnostic_status(verification: dict[str, Any]) -> str:
    if (verification["reconstructed_images"] < verification["planned_cache_images"]
            or not verification["planned_anomaly_spatial_complete"]):
        return "partial_evidence"
    return "planned_subset_complete" if verification["scope_kind"] == "verified_replay12" else "cache_diagnostics_complete"


def inventory(workspace: Path) -> list[dict[str, Any]]:
    """Inspect filenames and relevant archive headers only; no extraction or network."""
    findings = []
    needles = ("raw_cache/heldout/background/", "support_features/target_matched/background/", "official_memory.pt")
    for path in workspace.rglob("*"):
        if not path.is_file():
            continue
        name = path.as_posix()
        if any(needle in name for needle in needles):
            findings.append({"kind": "local_file", "path": str(path)})
        if "aebad" in path.name.lower() and path.name.endswith((".tar.gz", ".tar")):
            try:
                with tarfile.open(path, "r:*") as archive:
                    members = [member.name for member in archive if member.isfile()]
                findings.append({"kind": "archive", "path": str(path), "file_count": len(members),
                                 "cache_members": [name for name in members if any(needle in name for needle in needles)]})
            except (tarfile.TarError, OSError) as error:
                findings.append({"kind": "archive", "path": str(path), "error": str(error)})
    return findings


def check_protocol(results: Path) -> tuple[np.ndarray, np.ndarray]:
    for name in ("complete.json", "memory_manifest.json"):
        value = json.loads((results / name).read_text(encoding="utf-8"))
        if value.get("protocol_id") != PROTOCOL_ID:
            raise ValueError(f"wrong protocol in {name}")
        if name == "complete.json" and value.get("status") != "complete":
            raise ValueError("formal results are not marked complete")
        if name == "memory_manifest.json" and value["host_config"]["layers"] != list(LAYERS):
            raise ValueError("unexpected layer order")
    with np.load(results / "source_mad_state.npz", allow_pickle=False) as payload:
        if str(payload["protocol_id"].item()) != PROTOCOL_ID:
            raise ValueError("MAD protocol mismatch")
        medians, scales = (np.asarray(payload[name], np.float32) for name in ("medians", "scales"))
    if (medians.shape != (4,) or scales.shape != (4,) or not np.isfinite(medians).all()
            or not np.isfinite(scales).all() or np.any(scales <= 0)):
        raise ValueError("invalid saved Source MAD")
    return medians, scales


def archived_mmr(results: Path, per_image: list[dict[str, Any]], gt_rows: list[dict[str, Any]],
                 output: Path) -> dict[str, Any]:
    path = results / "mmr_compatible_sample_scores.csv"
    if not path.is_file():
        return {"status": "missing_csv"}
    lookup = {sample["sample_id"]: sample for sample in per_image}
    truth = {row["sample_id"]: row for row in gt_rows}
    by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    rows, seen = [], set()
    for row in read_csv(path):
        if row["domain"] != "background" or row["method"] not in METHODS:
            continue
        sample = lookup[row["sample_id"]]
        key = row["sample_id"], row["method"]
        if key in seen or row["relative_path"] != sample["relative_path"] or int(row["original_label"]) != sample["label"]:
            raise ValueError("MMR CSV identity/duplicate mismatch")
        seen.add(key)
        prefix = "source" if row["method"] == SOURCE else "union"
        if abs(float(row["full_image_score"]) - sample[prefix + "_score"]) > 1e-5:
            raise ValueError("MMR full score differs from original sample CSV")
        label = int(row["mmr_compatible_label"])
        expected = 0 if sample["label"] == 0 else truth[sample["sample_id"]].get("mmr_compatible_label")
        if expected is not None and label != expected:
            raise ValueError("MMR labels differ from original resize/crop GT rule")
        rows.append(dict(row, full_to_crop_drop=float(row["full_image_score"]) - float(row["cropped_image_score"]),
                         resize_256_max="", boundary_removal_drop="", decomposition_status="requires_raw_map"))
        by_method[row["method"]].append(dict(sample, label=label,
                                            source_score=float(row["cropped_image_score"]),
                                            union_score=float(row["cropped_image_score"])))
    if len(seen) != 2 * len(per_image):
        raise ValueError("MMR CSV held-out set incomplete")
    write_csv(output / "archived_mmr_samples.csv", rows)
    summary = {"status": "recomputed_from_archived_csv", "label_rule": "anomaly iff cropped GT nonempty",
               "scope": "output_side_only_not_full_MMR_input_protocol", "resize_crop_split": "unavailable_without_maps"}
    for method, samples in by_method.items():
        stats, _pairs, _per_image = ranking(samples)
        summary[method] = {"image_AUROC": stats["source_image_AUROC"],
                           "normal_count_after_relabelling": stats["normal_count"],
                           "anomaly_count_after_relabelling": stats["anomaly_count"],
                           "relabelled_anomalies": 212 - stats["anomaly_count"], "ties": stats["source_ties"]}
    return summary


def report(output: Path, summary: dict[str, Any], per_image: list[dict[str, Any]],
           availability: list[dict[str, Any]], cases: list[dict[str, Any]],
           instance_rows: list[dict[str, Any]], mechanism: list[dict[str, Any]],
           verification: dict[str, Any], mmr_summary: dict[str, Any]) -> None:
    focus = [sample for sample in per_image if sample["relative_path"] in FOCUS_PATHS]
    is_replay12 = verification.get("scope_kind") == "verified_replay12"
    lines = ["# AeBAD-S/background Source + Target K8 union 诊断", "",
             f"协议 `{PROTOCOL_ID}`；原始 sample_id 与 support_manifest 的 K8 顺序保持不变。",
             "本工具只读取既有结果、原图/GT 与缓存；不提取特征、不检索模型、不训练、不重建 memory、不拟合 MAD。", "",
             "## 已确认的数值事实", "",
             f"共同 held-out：{summary['normal_count']} 张正常、{summary['anomaly_count']} 张异常；{summary['pair_count']} 对正常—异常图像。",
             f"Source Image AUROC = {summary['source_image_AUROC']:.9f}；union = {summary['union_image_AUROC']:.9f}；差值 = {summary['union_image_AUROC'] - summary['source_image_AUROC']:+.9f}。",
             f"排序错误（ties 各计 0.5）：Source {summary['source_weighted_errors']:g} 对，union {summary['union_weighted_errors']:g} 对；ties 分别 {summary['source_ties']}、{summary['union_ties']}。",
             f"改善 {summary['improved_pairs']} 对，变差 {summary['worsened_pairs']} 对，不变 {summary['unchanged_pairs']} 对。这里的排序错误不是 FPR、漏检率或固定阈值下新增漏检。",
             "按 union 分数降序、路径打破并列所选前 10 张正常图，占 union 剩余加权排序错误的 "
             + (f"{summary['top10_remaining_error_fraction']:.2%}。" if summary['top10_remaining_error_fraction'] is not None else "不可定义（无剩余排序错误）。"), "",
             "| sample | Source score | union score | Source 排序错误 | union 排序错误 | 变化 |",
             "|---|---:|---:|---:|---:|---:|"]
    for sample in focus:
        lines.append(f"| {sample['sample_id']} / {Path(sample['relative_path']).name} | {sample['source_score']:.6f} | {sample['union_score']:.6f} | {sample['source_errors']:g}/85 | {sample['union_errors']:g}/85 | {sample['error_change']:+g} |")
    lines += ["", "这些案例的排序变化已对全部 85 张正常图重新计算。分数下降本身不能证明缺陷被吸收。", "",
              "## 空间证据与当前限制", "",
              f"背景域原始距离缓存：{sum(row['raw_status'] == 'valid' for row in availability)}/297；成功重建分数：{verification['reconstructed_images']}/297。",
              f"读取异常 GT 得到 {len(instance_rows)} 个非空 8 连通缺陷实例。" + ("本轮仅展示固定 12 个案例，不加入三个 1 像素最小实例案例。" if is_replay12 else "最小 3 个实例按面积、路径和实例编号排序，仅用于事后定向展示。"),
              "缺陷 GT 不是产品前景 mask。预览只显示原图和缺陷标注，无法自动把高响应分成背景、阴影、产品边缘或产品内部。",
              "所选案例与支持总览见 `selected_cases.csv`、`case_previews/`、`support_overview.png`；缺失文件见 `missing_files.csv`。"]
    if is_replay12:
        lines += ["该缓存目录已通过 replay12 原分数复核与文件校验。计划空间范围为 10 正常 + 2 异常；其余 285 张无 raw cache 属于范围限制，不是补跑失败。",
                  "排序继续使用全 297 张原 CSV；热图、峰值、实例响应和重算距离下降仅覆盖这 12 张。子集 reduction/gap 不代表全 background R_N/R_A/G。"]
    if not verification["planned_anomaly_spatial_complete"]:
        lines.append(f"范围内异常空间证据未完成：缺失 GT {verification['missing_required_gt_ids']}；实例响应/margin、GT resize/crop 或缺陷 patch 距离下降未完成 {verification['incomplete_anomaly_spatial_ids']}。地图重建数量不代表这些步骤已完成。")
    if verification["reconstructed_images"]:
        lines += [f"重建 image max 与正式 CSV 的最大误差：{verification['max_score_error']:.9g}；逐层 min 检查最大误差：{verification['max_union_min_error']:.9g}。",
                  "`case_peaks.csv` 与 `case_peak_layers.csv` 用半像素双线性插值 stencil 追踪原输出峰值；Target 胜出按参与插值的 patch 权重报告，不从像素坐标直接索引 patch。",
                  "评分图使用同一案例共同数值色标，保留负值。Source−union 距离图为四层原始 Euclidean/1280 距离下降均值，使用独立色标。",
                  "`resize_crop.csv` 分别记录原图 max、resize max、同一张 resize 图的 crop max，以及峰值和 GT 是否被裁掉。",
                  "实例 margin 固定为实例内 max − 相邻非缺陷环带 max；环带是原 GT 像素上 Chebyshev 半径 16 的扩张，排除全部 GT 缺陷后映射到原输出网格。空实例/空环带不产生 margin。"]
    else:
        lines += ["当前距离缓存或原数值后处理依赖不足，因此本次没有生成异常热图、案例峰值/分层贡献、实例响应/margin 或 resize/crop 数值分解。实例 patch 覆盖也标记为不可用。",
                  "原图、support 和 GT 预览不是高响应位置或最近邻来源的证据。"]
    smallest = sorted(instance_rows, key=lambda row: (row["area_original_pixels"], row["relative_path"], row["instance_id"]))[:3]
    lines += ["最小三个离散 GT 连通域面积（原像素）：" + ", ".join(str(row["area_original_pixels"]) for row in smallest)
              + "。这些是标注连通域；极小域可能是标注碎片，不能直接代表真实可见缺陷尺寸。"]
    lines += ["", "## 已归档的同一 union 机制统计", "",
              "下表直接读取本次正式结果的 mechanism_by_domain_layer.csv；不是根据选定案例重新计算，也不是早期其他方法的证据。raw cache 缺失时不能独立复核这些值。", "",
              "| layer | R_N | R_A | G |", "|---|---:|---:|---:|"]
    for row in mechanism:
        lines.append(f"| {row['layer']} | {float(row['R_N']):.6f} | {float(row['R_A']):.6f} | {float(row['G']):.6f} |")
    lines += ["", "R_N 是正常图全部 patch，R_A 是异常图 GT 缺陷 patch，均先图像等权。它们不能定位剩余高响应，也不能证明实例区分 margin 的变化。", "",
              "## MMR 输出端交叉检查", "",
              "`archived_mmr_samples.csv` 保留既有 full 与 crop 分数，仅能计算总变化。拆开 resize 与裁边必须使用真实异常图。未把裁剪后的标签静默用于原 held-out 排序。"]
    if mmr_summary.get("status") == "recomputed_from_archived_csv":
        for method in METHODS:
            row = mmr_summary[method]
            lines.append(f"沿用旧规则（裁剪后 GT 非空才算异常），{method} 的 MMR-compatible Image AUROC = {row['image_AUROC']:.9f}；{row['relabelled_anomalies']} 张原异常被重定义为正常。")
    lines += ["", "## 待核验的机制假设", "",
              "1. 困难正常图的剩余峰值位于哪类结构：需要既有 raw cache 重建的热图及人工判读。",
              "2. Target 在峰值邻域是否降低距离、哪些层仍有高贡献：需要 d_S / d_T / d_ST 及同位置 MAD 贡献，不能仅从最终图像分数判断。",
              "3. 重点异常和最小实例的局部区分是否受损：排序变化已有证据，局部 margin 还需真实距离图；两者不能互相代替。",
              "4. 最近邻对应哪张 support、哪个 patch：原缓存没有索引，未实现/执行可选特征追踪；不能从距离反推来源。", "",
              "## 当前最小下一步", "",
              ("本轮优先人工复核这 12 张真实热图、峰值分层和两个重点异常的固定环带 margin。不会扩充其他案例、域或 seed；空间类别仍需人工判读，不自动开展新方法实验。" if is_replay12 else
               "优先补齐 selected_cases.csv 对应的既有 raw_cache，完成峰值分层与固定环带 margin 复核。这是最小的证据补齐步骤；现有证据不足以选定背景抑制或其他新方法改动方向。本工具不会自动开展新方法实验。"), "",
              "## 验证边界", "",
              f"正式结果 SHA256 前后完全一致：{verification['formal_results_unchanged']}。",
              f"地图后端：{verification['map_backend_status']}。本诊断自身未执行模型推理；补跑的 GPU/分数验证另见 replay_status.json。",
              "输出为诊断覆盖范围记录；缺失的空间证据不标记为完成。", ""]
    (output / "diagnostic_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    results, cache, data, output = (Path(value).resolve() for value in
                                    (args.results_dir, args.cache_dir or args.results_dir, args.data, args.output_dir))
    for root in (results, cache, data):
        if output == root or output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError("output must be disjoint from results/cache/data")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing diagnostic directory: {output}")
    from DINOv3.MADEqual.aebad_s_target_fewshot.replay12 import verified_replay_scope
    replay_scope = verified_replay_scope(cache, results)
    before = snapshot(results)
    _medians, scales = check_protocol(results)
    dataset = Dataset(data)
    try:
        samples, supports = validate_inputs(read_csv(results / "sample_scores.csv"),
                                            read_csv(results / "support_manifest.csv"), dataset)
        summary, pairs, per_image = ranking(samples)
        domain = {row["method"]: row for row in read_csv(results / "domain_metrics.csv") if row["domain"] == "background"}
        for method, prefix in zip(METHODS, ("source", "union"), strict=True):
            if abs(float(domain[method]["image_AUROC"]) - summary[prefix + "_image_AUROC"]) > 1e-12:
                raise ValueError("CSV pair AUROC differs from archived domain metric")
        output.mkdir(parents=True, exist_ok=False)
        previews = output / "case_previews"
        previews.mkdir()
        write_csv(output / "pair_rankings.csv", pairs)
        write_csv(output / "per_image_rankings.csv", per_image)
        write_json(output / "ranking_summary.json", summary)
        print(f"CSV: N=85 A=212 Source AUROC={summary['source_image_AUROC']:.9f} union={summary['union_image_AUROC']:.9f}", flush=True)
        availability, missing, gt_rows, instance_rows = [], [], [], []
        masks: dict[str, np.ndarray] = {}
        raw_paths: dict[str, Path] = {}
        for sample in samples:
            sample_id, relative = sample["sample_id"], sample["relative_path"]
            raw_path = cache / "raw_cache" / "heldout" / "background" / f"{sample_id}.npz"
            row = dict(sample, image_status="available", raw_path=str(raw_path), raw_status="missing")
            if raw_path.is_file():
                values, shape, error = load_cache(raw_path, relative)
                row.update(raw_status="valid", grid_height=values["official_raw"].shape[1],
                           grid_width=values["official_raw"].shape[2], output_height=shape[0], output_width=shape[1],
                           union_min_max_error=error)
                raw_paths[sample_id] = raw_path
                del values
            else:
                outside_scope = replay_scope is not None and sample_id not in replay_scope
                if outside_scope:
                    row["raw_status"] = "outside_replay12_scope"
                missing.append({"kind": "raw_cache", "sample_id": sample_id, "expected_path": str(raw_path),
                                "blocks": "maps_peaks_layers_instances_resize_crop",
                                "classification": "outside_replay12_scope_not_replay_failure" if outside_scope else "missing_evidence"})
            if sample["label"]:
                gt = gt_path(relative)
                row["gt_path"] = gt
                row["gt_status"] = "available" if gt in dataset.names else "missing"
                if gt in dataset.names:
                    mask = np.asarray(dataset.image(gt, "L")) > 0
                    masks[sample_id] = mask
                    labels, instances = components(mask)
                    cropped = mmr.resize_and_crop_mask(mask)
                    resized = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((256, 256), Image.Resampling.NEAREST)) > 0
                    gt_rows.append({"sample_id": sample_id, "relative_path": relative,
                                    "gt_original_area": int(mask.sum()), "gt_resize_area": int(resized.sum()),
                                    "gt_crop_area": int(cropped.sum()), "gt_removed_by_crop": int(resized.sum() - cropped.sum()),
                                    "gt_empty_after_crop": not bool(cropped.any()),
                                    "mmr_compatible_label": mmr.mmr_compatible_label(1, cropped), "original_label": 1})
                    for instance in instances:
                        instance_rows.append(dict(instance, sample_id=sample_id, relative_path=relative,
                                                  patch_coverage="", patch_coverage_status="missing_raw_cache_geometry",
                                                  response_status="outside_replay12_scope" if replay_scope is not None and sample_id not in replay_scope else "missing_raw_cache"))
                else:
                    gt_rows.append({"sample_id": sample_id, "relative_path": relative, "gt_status": "missing"})
                    missing.append({"kind": "GT", "sample_id": sample_id, "expected_path": gt, "blocks": "instance_and_defect_region_analysis"})
            else:
                row["gt_status"] = "normal_no_defect"
            availability.append(row)
        by_id = {sample["sample_id"]: sample for sample in per_image}
        reasons: dict[str, list[str]] = {sample_id: ["top10_union_normal"] for sample_id in summary["top10"]}
        for sample in per_image:
            if sample["relative_path"] in FOCUS_PATHS:
                reasons.setdefault(sample["sample_id"], []).append("focus_anomaly")
        smallest = sorted(instance_rows, key=lambda row: (row["area_original_pixels"], row["relative_path"], row["instance_id"]))[:3] if replay_scope is None else []
        for row in smallest:
            reasons.setdefault(row["sample_id"], []).append(f"smallest_instance_{row['instance_id']}_area_{row['area_original_pixels']}")
        cases = [dict(by_id[sample_id], selection_reason=";".join(reason),
                      spatial_category="unreviewed") for sample_id, reason in reasons.items()]
        normal_preview = []
        for sample in cases:
            image = dataset.image(sample["relative_path"])
            sample.update(image_height=image.height, image_width=image.width)
            items = [(image, f"{sample['sample_id']} {Path(sample['relative_path']).name}\nSource={sample['source_score']:.6f} union={sample['union_score']:.6f}")]
            mask = masks.get(sample["sample_id"])
            if mask is not None:
                if mask.shape != (image.height, image.width):
                    raise ValueError("GT/image original geometry mismatch")
                items.append((Image.fromarray(mask.astype(np.uint8) * 255).convert("RGB"), "Original defect GT; not product foreground"))
                labels, _instances = components(mask)
                for instance in smallest:
                    if instance["sample_id"] == sample["sample_id"]:
                        y0, y1 = max(0, instance["bbox_y0"] - 16), min(image.height, instance["bbox_y1_exclusive"] + 16)
                        x0, x1 = max(0, instance["bbox_x0"] - 16), min(image.width, instance["bbox_x1_exclusive"] + 16)
                        rgb_zoom = image.crop((x0, y0, x1, y1)).resize((220, 220), Image.Resampling.BILINEAR)
                        items.append((rgb_zoom, f"Instance {instance['instance_id']}, area={instance['area_original_pixels']}\nRGB zoom + 16px context; no feature extraction"))
                        instance_gt = (labels[y0:y1, x0:x1] == instance["instance_id"]).astype(np.uint8) * 255
                        zoom = Image.fromarray(instance_gt).convert("RGB").resize((220, 220), Image.Resampling.NEAREST)
                        items.append((zoom, f"Instance {instance['instance_id']} discrete GT (nearest zoom)\nOriginal crop x=[{x0},{x1}), y=[{y0},{y1})"))
            montage(items, previews / f"{sample['sample_id']}_original_gt.png", columns=len(items))
            if sample["label"] == 0:
                normal_preview.append(items[0])
        montage(normal_preview, output / "hard_normals_overview.png", columns=5)
        support_items = []
        for row in supports:
            support_items.append((dataset.image(row["relative_path"]), f"K8 support #{row['support_index']}\n{Path(row['relative_path']).name} (manifest order)"))
            feature_path = cache / "support_features" / "target_matched" / "background" / f"{int(row['support_index']):02d}.npz"
            if not feature_path.is_file():
                missing.append({"kind": "support_features", "sample_id": "", "expected_path": str(feature_path), "blocks": "optional_nearest_support_tracking_only"})
        montage(support_items, output / "support_overview.png")
        if not (cache / "official_memory.pt").is_file():
            missing.append({"kind": "official_memory", "sample_id": "", "expected_path": str(cache / "official_memory.pt"),
                            "blocks": "optional_source_bank_tracking_only_not_required_for_raw_diagnostics"})
        mmr_summary = archived_mmr(results, per_image, gt_rows, output)
        write_json(output / "archived_mmr_summary.json", mmr_summary)
        mechanism_path = results / "mechanism_by_domain_layer.csv"
        mechanism = [dict(row, provenance="archived_formal_csv_not_recomputed") for row in read_csv(mechanism_path)
                     if row["domain"] == "background" and row["method"] == UNION] if mechanism_path.is_file() else []
        write_csv(output / "archived_mechanism.csv", mechanism)
        verification = {"map_backend_status": "not_imported_no_raw_cache", "reconstructed_images": 0,
                        "max_score_error": None, "max_union_min_error": max((row.get("union_min_max_error", 0) for row in availability), default=0) if raw_paths else None,
                        "score_tolerance": args.score_tolerance, "gpu_inference": "not_executed_by_diagnostics",
                        "planned_cache_images": 12 if replay_scope is not None else 297,
                        "scope_kind": "verified_replay12" if replay_scope is not None else "full_background",
                        "outside_scope_images": 285 if replay_scope is not None else 0}
        backend = None
        if raw_paths:
            try:
                backend = MapBackend(results / "source_mad_state.npz")
                verification["map_backend_status"] = "original_functions_cpu_map_reconstruction"
            except ImportError as error:
                verification["map_backend_status"] = f"dependency_unavailable: {error}"
                missing.append({"kind": "map_backend_dependency", "sample_id": "", "expected_path": str(error), "blocks": "raw_map_reconstruction"})
                for row in instance_rows:
                    if row["sample_id"] in raw_paths:
                        row.update(response_status="missing_map_backend_dependency", patch_coverage_status="missing_original_adaptive_max_backend")
        peaks, layer_rows, crop_rows, mechanism_rows, map_checks, color_rows = [], [], [], [], [], []
        for sample in samples if backend is not None else []:
            sample_id = sample["sample_id"]
            if sample_id not in raw_paths:
                continue
            values, maps, shape = backend.maps(raw_paths[sample_id], sample["relative_path"])
            error = max(abs(float(maps[method].max()) - sample[prefix + "_score"])
                        for method, prefix in zip(METHODS, ("source", "union"), strict=True))
            map_checks.append({"sample_id": sample_id, "max_score_error": error, "matches_csv": error <= args.score_tolerance})
            if error > args.score_tolerance:
                write_csv(output / "map_verification.csv", map_checks)
                raise ValueError(f"reconstructed score differs from CSV: {sample_id}, error={error}")
            for method in METHODS:
                crop_rows.append(dict(sample_id=sample_id, method=method,
                                      **crop_diagnostics(maps[method], masks.get(sample_id), sample["label"])))
            reduction = (values["official_raw"] - values["target_raw_8"]) / (scales[:, None, None] + 1e-8)
            patch_mask = None
            if sample["label"] and sample_id in masks and masks[sample_id].any():
                patch_mask = backend.patch_mask(Image.fromarray(masks[sample_id].astype(np.uint8) * 255), values["official_raw"].shape[1:])
            if sample["label"] == 0 or patch_mask is not None:
                for offset, layer in enumerate(LAYERS):
                    region = np.ones(reduction.shape[1:], bool) if sample["label"] == 0 else patch_mask
                    mechanism_rows.append({"sample_id": sample_id, "label": sample["label"], "layer": layer,
                                           "region": "normal_all_patches" if sample["label"] == 0 else "GT_defect_patches",
                                           "R_value": float(reduction[offset][region].mean())})
            if sample_id in masks:
                labels, _instances = components(masks[sample_id])
                for row in instance_rows:
                    if row["sample_id"] == sample_id:
                        row.update(instance_response(labels == row["instance_id"], masks[sample_id], maps, values, backend))
            if sample_id in reasons:
                case = next(case for case in cases if case["sample_id"] == sample_id)
                p, layers = peak_rows(case, values, maps, backend)
                peaks.extend(p)
                layer_rows.extend(layers)
                colors = plot_maps(dataset.image(case["relative_path"]), masks.get(sample_id), maps, values, backend, previews / f"{sample_id}_maps.png")
                color_rows.append(dict(sample_id=sample_id, **colors))
            del values, maps
        recomputed_mechanism = []
        for layer in LAYERS:
            normal = [row["R_value"] for row in mechanism_rows if row["layer"] == layer and row["label"] == 0]
            anomaly = [row["R_value"] for row in mechanism_rows if row["layer"] == layer and row["label"] == 1]
            if normal or anomaly:
                full = len(normal) == 85 and len(anomaly) == 212
                rn = float(np.mean(normal)) if normal else None
                ra = float(np.mean(anomaly)) if anomaly else None
                recomputed_mechanism.append({"layer": layer, "scope": "full_background" if full else "cached_subset_only",
                                             "normal_images": len(normal), "anomaly_images": len(anomaly),
                                             "R_N" if full else "subset_normal_reduction": rn,
                                             "R_A" if full else "subset_defect_reduction": ra,
                                             "G" if full else "subset_gap": rn - ra if rn is not None and ra is not None else None,
                                             "aggregation": "image_equal"})
        verification.update(reconstructed_images=len(map_checks), max_score_error=max((row["max_score_error"] for row in map_checks), default=None),
                            formal_results_unchanged=before == snapshot(results),
                            **spatial_completion(availability, replay_scope, instance_rows, crop_rows, mechanism_rows))
        if not verification["formal_results_unchanged"]:
            raise RuntimeError("formal inputs changed during diagnosis")
        write_csv(output / "sample_availability.csv", availability)
        write_csv(output / "selected_cases.csv", cases)
        write_csv(output / "defect_instances.csv", instance_rows)
        write_csv(output / "gt_crop_diagnostics.csv", gt_rows)
        for name, rows in (("missing_files", missing), ("case_peaks", peaks), ("case_peak_layers", layer_rows),
                           ("resize_crop", crop_rows), ("map_verification", map_checks), ("case_color_scales", color_rows),
                           ("recomputed_reduction_per_image", mechanism_rows), ("recomputed_reduction_summary", recomputed_mechanism)):
            write_csv(output / f"{name}.csv", rows)
        findings = inventory(Path(__file__).resolve().parents[3]) if not args.skip_archive_inventory else []
        write_json(output / "local_cache_inventory.json", findings)
        write_json(output / "verification.json", verification)
        write_json(output / "formal_input_sha256.json", before)
        report(output, summary, per_image, availability, cases, instance_rows, mechanism, verification, mmr_summary)
        planned = verification["planned_cache_images"]
        write_json(output / "diagnostic_status.json", {"status": diagnostic_status(verification),
                                                       "protocol_id": PROTOCOL_ID, "cache_revision": CACHE_REVISION,
                                                       "selected_cases": len(cases), "defect_instances": len(instance_rows),
                                                       "raw_cache_images": len(raw_paths), "reconstructed_images": len(map_checks),
                                                       "planned_cache_images": planned, "outside_scope_images": verification["outside_scope_images"],
                                                       "required_anomaly_gt_images": verification["required_anomaly_gt_images"],
                                                       "missing_required_gt_ids": verification["missing_required_gt_ids"],
                                                       "incomplete_anomaly_spatial_ids": verification["incomplete_anomaly_spatial_ids"],
                                                       "planned_anomaly_spatial_complete": verification["planned_anomaly_spatial_complete"],
                                                       "missing_file_count": len(missing), "optional_tracking": "not_implemented_not_executed"})
        print(f"DIAGNOSTIC_OUTPUT={output}; reconstructed={len(map_checks)}/297; no inference", flush=True)
    finally:
        dataset.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, help="Existing original experiment root, with raw_cache/ and support_features/; defaults to results-dir")
    parser.add_argument("--data", type=Path, required=True, help="AeBAD.zip or extracted AeBAD-S root")
    parser.add_argument("--output-dir", type=Path, required=True, help="New, non-existing directory; no overwrite")
    parser.add_argument("--score-tolerance", type=float, default=1e-5)
    parser.add_argument("--skip-archive-inventory", action="store_true", help="Skip workspace filenames/relevant tar header inventory")
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    if not np.isfinite(arguments.score_tolerance) or arguments.score_tolerance <= 0:
        raise ValueError("score tolerance must be finite and positive")
    run(arguments)
