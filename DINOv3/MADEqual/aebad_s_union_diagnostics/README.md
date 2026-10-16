# AeBAD-S/background Source + Target K8 union diagnostics

仅诊断 `mad_equal_aebad_s_target_fewshot_seed42_v1` 的共同 held-out，85 正常 + 212 异常。
默认读取既有 CSV、Source MAD、选定原图和全部 background 异常 GT；zip 内原图按需读取，不全量解压。
不调用原 `runner.run()`，不训练、推理、重建 memory、重拟合 MAD 或选择新的 support。
输出目录必须不存在，且与结果、缓存、数据目录互不包含；拒绝覆盖已有目录。

## 本地运行

从工程根目录运行；解释器自行替换成具有下述轻量依赖的 Python。以下命令只运行诊断。

```powershell
Set-Location 'G:\codex\workspace\experiment'
python -m DINOv3.MADEqual.aebad_s_union_diagnostics.runner `
  --results-dir 'G:\codex\workspace\experiment\server_archives\aebad_s_target_fewshot_seed42_v1_formal_results\aebad_s_target_fewshot_seed42_v1' `
  --data 'G:\codex\workspace\experiment\datasets\AeBAD.zip' `
  --output-dir 'G:\codex\workspace\experiment\DINOv3\outputs\mad_equal\aebad_s_union_background_diagnostic_seed42_v1'
```

如果上述输出已存在，明确改用新的目录名；本工具没有 overwrite/resume 选项。
`--cache-dir` 指向原实验根目录（包含 `raw_cache/` 和 `support_features/`），默认与 results-dir 相同。
也支持 `--data` 指向已解压的 AeBAD-S 根目录，不自动下载数据或读取远端。
默认盘点工程内文件名和 AeBAD 相关 tar 包头，不解压归档；`--skip-archive-inventory` 可跳过该盘点。

轻量诊断需要 NumPy、Pillow、SciPy；没有 PyTorch 时仍能完成 CSV、原图/support 预览和 GT 实例清单。
只在发现既有 raw cache 时才尝试导入原 `_load_observation`、`_load_mad`、`_method_map`、
`mad_calibrated`、`mad_equal`、`interpolate_score`、`mask_at_patch_grid`。
这些原入口要求原工程的 PyTorch/torchvision 等依赖；缓存地图使用 CPU 数值后处理，不加载模型。
地图可视化另需 Matplotlib。依赖缺失会在报告中标记，绝不降级为自行重写的评分或推理。
MMR 模块仅延迟导入推理/评测 runner，其 resize/crop 函数和数值实现保持原样。

## 数值与统计

- `official_raw = d_S`，`target_support_8 = d_T`，`target_raw_8 = d_ST`，验证逐层 `d_ST = min(d_S,d_T)`。
  距离是 Euclidean / 1280；不二次缩放、不平方。Source-extra union 不是 Source 原基线。
- 原处理顺序：逐层 min → 固定 Source MAD → 四层等权平均 → 原双线性插值。
  Source 和 union 调原 `_method_map`；Target-only 是明确标注的新增诊断视图，使用同一 MAD/插值。
  image score 是最终插值图 max，保留负值。重建与原 CSV 最大误差需 ≤ `--score-tolerance`（默认 1e-5），超限停止。
- 全域正常—异常对的加权错误：正常分数 > 异常分数计 1，相等计 0.5，正常分数较低计 0；
  AUROC = 1 − 加权错误 / (85×212)。分别报告严格逆序和 ties；这些不是 FPR/漏检率。
- 困难正常按 union 分数降序、路径升序选前 10；sample_id 由完整 test 索引验证，永不对诊断子集重新编号。
  support 使用 manifest 顺序，K1/K2/K4/K8 为嵌套前缀，验证所有 CSV 方法排除同一个 K8 pool。
- 峰值通过 align_corners=False 的半像素双线性 stencil 映射到最多 4 个 patch；
  分层距离和 MAD 贡献按该权重记录。Target 胜出/平局也按 patch 权重记录，不能先插值再取 min。
- `Delta_l = (d_S,l-d_ST,l)/(scale_l+1e-8)`。正常区域为全部 patch，异常区域仅为原 adaptive-max GT 缺陷 patch。
  全 85/212 图均有数据才称全域 R_N/R_A/G；子集只叫 cached-subset reduction/gap，先图像等权。
  `archived_mechanism.csv` 单独记录已有正式同一 union 的汇总，不冒充本地重算。
- GT 使用 8 连通域；按原像素面积、路径、实例编号选最小 3 个非空实例。GT 只用于事后诊断，定向案例不代表总体性能。
  实例 patch 覆盖调用原 adaptive-max；缺少 cache geometry 时留空并标明 unavailable。
- 局部 margin = 实例内最大响应 − 相邻非缺陷环带最大响应。
  固定环带为原始 GT 像素 Chebyshev 半径 16 扩张，排除全部 GT 缺陷后用 nearest 映射至原输出网格；
  环带为空或实例在评测网格消失时不产生 margin。不按测试标签选择阈值或制造召回率。
- 原 max → Pillow 原 MMR bilinear resize 256 max → 同一 resize 图 center crop `[16:240,16:240]` max。
  分开记录 resize_drop 和 boundary_removal_drop；如峰值并列，分别报告部分/全部 resize 峰值被裁。
  GT 用原 nearest resize/crop，原始标签保持不变。旧 MMR AUROC 单独沿用“crop GT 非空”标签规则。
- 三张评分热图共用同一案例的实际色标。距离下降图是逐层原始距离下降的均值插值，使用独立色标；没有逐图 min-max。
  缺陷 GT 的补集不能叫背景，所有空间类别默认 unreviewed，等待人工判读。

## 输出和缺失证据

必有 `ranking_summary.json`、`pair_rankings.csv`（18,020 对）、`per_image_rankings.csv`、
`sample_availability.csv`、`selected_cases.csv`、`missing_files.csv`（存在缺失才写）、
`defect_instances.csv`、`gt_crop_diagnostics.csv`、原图/GT/support 预览和 `diagnostic_report.md`。
另有已有 MMR CSV 的交叉检查、当前 union 的归档机制表，以及正式目录 SHA256 前后检查。

有可用 raw cache 和原依赖时，增加 `map_verification.csv`、`case_peaks.csv`、`case_peak_layers.csv`、
`case_color_scales.csv`、`resize_crop.csv`、重算的分图/汇总距离下降，以及真实热图；不会写空壳空间结果。
仅部分 cache 可用时覆盖范围显式记录。没有 cache 时无法拆开 resize/crop，也无法确认高响应区域或实例 margin。
`diagnostic_status.json` 的 partial_evidence 是证据缺失状态，不等于诊断失败，也不声称空间诊断已完成。

本轮未实现可选最近邻追踪。距离缓存不含索引，不能据距离反推 support 图或 patch。
缺失 support_features/official_memory.pt 只阻塞可选追踪，不阻塞 raw cache 地图诊断。

补跑丢失缓存可使用独立的 `aebad_s_target_fewshot.replay12`，说明见
`../aebad_s_target_fewshot/README_replay12.md`。它只恢复固定 10 正常 + 2 重点异常，且本地不启动推理。
本诊断自动识别 replay12 标记，要求所有案例通过原 CSV 分数复核并校验文件 SHA；失败/未完成目录会被拒绝。
使用 verified replay12 时只选择这 12 张展示，不加入三个 smallest-instance 案例；
其余 285 张缓存标为范围外，不要求恢复它们。只有 12 张地图及两个范围内异常的 GT、
实例响应/margin 检查、GT resize/crop 和四层缺陷 patch 距离下降均完成，才记为 planned_subset_complete。
范围内 GT 缺失或相关步骤未完成时保持 partial_evidence；verification.json / diagnostic_status.json
列出 missing_required_gt_ids 和 incomplete_anomaly_spatial_ids，报告同步说明缺口。
空环带、输出网格上消失的实例沿用原规则记录不可定义结果；子集距离统计不充作全域结果。

## 必要小型验证

```powershell
python -m unittest DINOv3.tests.test_mad_equal_aebad_s_union_diagnostics -v
```

覆盖 ties/排序转移、固定并列排序、cache 协议/路径/形状/union-min、插值前 min、半像素 stencil、
8 连通域、保留负值，以及 resize 与去边界的独立下降。另在隔离子进程强制禁止导入 PyTorch，验证默认诊断入口可导入。
这些合成验证不代表真实 raw map 或 GPU 已验证；真实分数对齐仅在实际缓存可用后执行。

## 本轮已执行结果

本地使用既有 Python 和 NumPy/Pillow/SciPy，在强制禁止导入 PyTorch 的进程中运行成功，退出码 0；未安装或修改环境。
14 项 unittest 通过，包括支持集排除/共同 held-out、sample_id、局部环带排除其他 GT 缺陷。
最终复核输出为 `DINOv3/outputs/mad_equal/aebad_s_union_background_diagnostic_seed42_v1_review1/`；
初次输出目录保留，未覆盖它或任何旧实验。正式结果全文件 SHA256 前后相同。

实际 CSV 分析：Source AUROC 0.669034406 → union 0.796226415；排序错误 5,964 → 3,672，
2,392 对改善、100 对变差、15,528 对不变，ties 均为 0。
前 10 正常图占剩余排序错误 40.36%；两个重点异常的错误对数分别为 IMG_9385：57→59，IMG_8863：72→73（均对全部 85 正常图）。
读取 212 张 GT 得到 388 个 8 连通实例，最小三个面积均为 1 原像素；这可能是标注碎片，不能直接代表真实可见缺陷尺寸。

本地/相关 AeBAD tar 归档未找到这条协议的 297 份 raw cache、8 份 Target support_features 或 official_memory.pt。
306 项缺失路径单独列出；输出状态为 partial_evidence。
原图、GT 和 support 预览已实际生成并检查；真实缓存地图、峰值/分层贡献、实例 margin、resize/crop 分解和 GPU 均未验证。
