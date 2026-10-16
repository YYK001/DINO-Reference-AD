# AeBAD-S/background 冻结 K8 union 逐支持移除诊断

仅实现失败归因工具。唯一干预是逐张移除原八张 Target support 之一；不选择“最佳七张”，不训练、不抽样支持、不重建 memory、不重拟合 MAD、不增加部署阈值或检测方法。默认模式为 `preflight`。本轮未在服务器运行。

## 计算路径与边界

正式入口：`python -m DINOv3.MADEqual.aebad_s_union_support_loo.runner`。

复用原 `observe_fewshot_pil` → `_extract_processed_batch` → `_exact_top1_chunked`/`_support_prefix_minima`。在原评分函数增加默认关闭的 `nearest_observer` 和 `geometry_observer`；开启时接收旁路副本，保留原 min 和返回结构。未修改 Exact kernel、query grid、chunk 上限、dtype、原 Top-1 tie-breaking 或支持扫描顺序。`exact_prefix_minima` 的同类回调用于既有 Source-extra 路径，默认行为不变。本工具显式传 `source_support=None`，只检索 Source memory 与原 Target K8；该选项下 `source_support_raw` 为空，`source_raw` 为 Source 参照，本工具不将其解释为 Source-extra 实验。

每张 query 仅一次完整视野特征提取。所有逐支持距离和最近邻索引来自生成该次原评分的同一次检索，绝不选择少数 query patch 再检索。支持特征必须已存在且符合 replay provenance；缺失即停止，不调用提取 fallback。原 memory/MAD 缺失同样停止。

保存 compact NPZ：

| 数组 | 维度 | dtype/含义 |
| --- | --- | --- |
| positions | M | int64，compact 行 → 物理 patch flat index |
| source_distance/source_index | 4×M | float32 欧氏/1280；int64 memory 行 |
| target_distance/target_index | 8×4×M | float32 欧氏/1280；int64 support patch flat index |
| metadata | 标量 JSON | 输入身份、sample/path、网格、输出形状、支持 manifest 顺序和预处理几何 |

层顺序 L7/L15/L23/L31，支持顺序严格从原 `support_manifest.csv` 读取。capture 要求完整物理网格，显式检查 `positions` 为完整排列，不假定其顺序等于行号。缓存不含 query 特征，不保存 query×memory 距离矩阵。以 118×118 网格估计，未压缩距离/索引约 5.9 MiB/图，297 张约 1.7 GiB；NPZ 实际体积取决于压缩与索引分布，另需表格/十二案例图的空间。该估算不是峰值 GPU/主机内存估算。

离线对每个 j 计算 `min(d_Source, min(d_Target[k != j]))`，随后通过原 `_method_map` 逐层 Source MAD → 四层均值 → 原双线性插值 → 最终地图 max。八个分支共享一次 capture，不做八次 forward/检索。调用 `MapBackend` 取得原 MAD、评分、插值和 GT patch pooling；不调用全量 `run()`。图像分数 AP 使用 `average_precision_score`，排序对并列误差权重沿用 `pair_error` 的 0.5。

GT 像素在整个预测范围的距离/索引落盘后才解码用于验证或 analyze；preflight 仅检查 GT 文件存在并记录字节身份。连通域、GT nearest resize、原像素 Chebyshev 半径16环带且排除全部 GT，均复用 `components`/`instance_regions`。不增加小缺陷面积阈值，保留1像素碎片、空环带和输出网格消失的记录。

Source memory 只有 bank 行索引，原图/patch provenance 为 unavailable；不重建 memory，不猜来源。八张支持各自的最小距离及跨图 top-2 间隔可查，图内第二近 patch 未记录，唯一性为 unknown；近似并列是数值不确定性标记。保留 kernel 原 Top-1；同距离 Target support 显示按 manifest 编号，Source/Target 相等明确记 tie。不能据跨图间隔推断图内对应唯一，不能宣称恢复历史未记录的唯一 NN 索引。

原八张支持始终从全部分支评价集中排除；background 为85正常/212异常，共297张。Source 是单独参照；九个干预分支为 K8 加 minus_1..8。K8 固定困难正常前十直接来自原 selected_cases/正式分数，分支之间不重选。正常尾部分位数属于分支自身分布，固定前十响应变化另记。

## 两阶段门槛与完整性

阶段 A 的案例及顺序直接读取既有 `selected_cases.csv`，校验为原十二案例，不重新选择。复核 Source raw、Target K8 min、union layer min、正式图像分数、原峰值/逐层贡献、两个重点异常的实例响应和 margin。距离 `atol=1e-7, rtol=1e-6`，分数/贡献绝对容差 `1e-5`，不可调大。失败保存 raw、真实误差及 `capture_not_reproduced`/`capture_failed` 状态；阶段 B 和 analyze 不接受失败的 capture。

阶段 B 需要显式 `--scope background --stage-a-dir ...`。只接受完整、通过验证且冻结输入/本轮代码身份完全一致的 A。十二张已验证 raw 复制到新 B 目录并继续按旧哈希复核；只有其余285张再次 forward。软件版本或精度与 A 不同也停止复用。全域逐图 Source/K8 分数对照正式 `sample_scores.csv`，AUROC/AP 汇总再次对照该 CSV 的完整 background 结果。

`inputs.json` 分别记录历史 scoring SHA 和本轮日志代码 SHA；不会把新代码身份写回 replay/正式历史。原 backbone/host/support 提取代码仍核对历史身份。添加日志后，旧四案例 trace 自己的严格 scoring SHA 校验可能拒绝该文件变化；本工具不修改那条历史保护，也不把旧 trace 的未验证对应当作输入事实。

本工具不提供续跑：输出必须为新目录，或仅有它的 preflight.json。已有产物、半写入文件、身份不匹配均停止并保留现场。read_capture 在 analyze/阶段 B 前后检查可信记录、CSV 和每图 raw 原哈希，当前修改不会被重新认可。capture 和 analyze 分别写到独立目录，不覆盖 formal/replay/trace/历史报告。

## 输入清单

所有以下路径运行前由用户自行确认实际存在。本地结果归档不包含完整 memory/权重/支持特征时，preflight 会诚实报告缺失。

- 正式目录：`sample_scores.csv`、`support_manifest.csv`、`complete.json`、`memory_manifest.json`、`source_mad_state.npz`；capture 另需原 `official_memory.pt`。
- 原 replay12 目录：`replay_inputs.json`、`replay_status.json`、`environment_*.json`、`support_provenance.csv`、十二张原 `raw_cache/heldout/background/*.npz`。
- 原诊断目录：`selected_cases.csv`、`case_peaks.csv`、`case_peak_layers.csv`、`defect_instances.csv`、`diagnostic_status.json`、`verification.json`。
- capture 原八张支持特征：默认 `replay/support_features/target_matched/background/01.npz` 到 `08.npz`。`--support-cache-dir` 可定位字节一致的副本；仍核对原 provenance SHA。
- capture 需本地 SuperADD 官方代码、DINOv3 官方代码和 SHA 核准的 DINOv3 checkpoint。
- A 数据：十二张 query、八张 support、两个重点异常 GT；B 数据：297张 query、八张 support、所有范围内异常 GT。数据根仅需 `test`/相关 `ground_truth`，无需 `train/good`。
- analyze 不需要 backbone、memory 或支持特征；需要通过验证的 capture 目录、字节一致的正式 CSV/MAD、异常 GT、原十二案例和八张支持 RGB（画对应图）。可读取目录或 ZIP，不解压 ZIP。

## 用户后续执行命令

以下为供用户后续明确授权后执行的服务器命令，本轮没有执行、上传、连接或自动关机。默认脚本只做 preflight。

```bash
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export PROJECT_ROOT=/root/autodl-tmp/DINO-NVS
export PYTHON=/root/miniconda3/bin/python
export BASE="$PROJECT_ROOT/DINOv3/outputs/mad_equal"
export RESULTS_DIR="$BASE/aebad_s_target_fewshot_seed42_v1"
export REPLAY_DIR="$BASE/aebad_s_union_background_replay12_seed42_v1"
export DIAGNOSTIC_DIR="$BASE/aebad_s_union_background_replay12_seed42_v1_diagnostic_20261003_150831"
export DATA_DIR=/root/autodl-tmp/datasets/AeBAD/AeBAD_S
export STAGE_A_DIR="$BASE/aebad_s_union_support_loo_A_$(date +%Y%m%d_%H%M%S)"
cd "$PROJECT_ROOT" || exit 1

# 无 GPU 也可运行；只检查，不导入 Torch/加载模型。
bash DINOv3/MADEqual/aebad_s_union_support_loo/scripts/run_aebad_s_union_support_loo.sh \
  --results-dir "$RESULTS_DIR" --replay-dir "$REPLAY_DIR" \
  --diagnostic-dir "$DIAGNOSTIC_DIR" --data "$DATA_DIR" --output-dir "$STAGE_A_DIR"

# 显式阶段 A（需要 CUDA）；同一个目录仅有 preflight.json 时允许继续。
"$PYTHON" -u -m DINOv3.MADEqual.aebad_s_union_support_loo.runner capture \
  --scope twelve --results-dir "$RESULTS_DIR" --replay-dir "$REPLAY_DIR" \
  --diagnostic-dir "$DIAGNOSTIC_DIR" --data "$DATA_DIR" \
  --dinov3-root /root/autodl-tmp/dinov3 --output-dir "$STAGE_A_DIR" --device cuda:0

# 检查 A；必须 status=capture_verified，数值检查无失败，才执行后面的 B。
cat "$STAGE_A_DIR/status.json"

# A 的离线分析（无需 GPU；原 CPU 地图后端仍需可用的 Torch 等依赖）。
"$PYTHON" -u -m DINOv3.MADEqual.aebad_s_union_support_loo.runner analyze \
  --scope twelve --capture-dir "$STAGE_A_DIR" --results-dir "$RESULTS_DIR" \
  --data "$DATA_DIR" --output-dir "${STAGE_A_DIR}_analysis"

# 仅在 A 验证通过后，显式启动全域 capture（CUDA）。
export STAGE_B_DIR="$BASE/aebad_s_union_support_loo_B_$(date +%Y%m%d_%H%M%S)"
"$PYTHON" -u -m DINOv3.MADEqual.aebad_s_union_support_loo.runner capture \
  --scope background --stage-a-dir "$STAGE_A_DIR" \
  --results-dir "$RESULTS_DIR" --replay-dir "$REPLAY_DIR" --diagnostic-dir "$DIAGNOSTIC_DIR" \
  --data "$DATA_DIR" --dinov3-root /root/autodl-tmp/dinov3 \
  --output-dir "$STAGE_B_DIR" --device cuda:0

# 全域离线 analyze（无需 GPU；不重复 backbone/检索）。
"$PYTHON" -u -m DINOv3.MADEqual.aebad_s_union_support_loo.runner analyze \
  --scope background --capture-dir "$STAGE_B_DIR" --results-dir "$RESULTS_DIR" \
  --data "$DATA_DIR" --output-dir "${STAGE_B_DIR}_analysis"
```

命令不应整段无条件串行执行；A 若失败，保留目录并检查 `capture_verification.csv`，不要继续全域。入口自身也执行该门槛。默认 checkpoint 在 `external_baselines/SuperADD/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth`，Source memory 在正式结果目录；实际位置不同用 `--checkpoint`/`--memory` 指定原文件。数据/代码路径亦可通过 CLI 覆盖。

只读本地检查可使用 `preflight --no-write`，不会创建输出目录。`--help` 不导入 Torch。输出退出码：0=该阶段完成，2=preflight阻止，3=数值不复现或空间证据不完整，未处理异常为非0并保存已进入阶段的状态。不要将非0目录当作通过验证的结果。

## 输出和解释

capture：`inputs.json`、`environment.json`、`status.json`、`raw_cache/`、`capture_verification.csv`。

analyze：`per_image_loo.csv`、`image_ranking_summary.csv`、`per_support_effects.csv`、`support_layer_per_image.csv`、`support_layer_summary.csv`、`defect_instance_loo.csv`、`peak_positions.csv`、`position_support_effects.csv`、`peak_patch_correspondence.csv`、`analysis_verification.csv`、十二案例 `figures/`、`diagnostic_report.md`，及身份/状态和 capture 复核副本。每次 analyze 都重新读取已核对身份的正式 `sample_scores.csv`，逐图复核本次 Source/K8 地图 max，绝对容差仍为 `1e-5`，不以 capture 元数据内的分数代替正式 CSV。background analyze 另核对 Source/K8 的 Image AUROC 与 Image AUPR（AP）完整汇总，仍采用 `1e-5`。原分数/汇总、GT 或 stencil 任一复核失败，保持 `partial_evidence`、退出码3，不宣称完成；`analysis_verification.csv` 保留实际值、正式参照值、误差和验证状态，`status.json` 记录 `formal_baselines_reproduced`。

正常图和异常图的各分支重新取峰值，也在 Source/K8及各分支固定峰值位置交叉读响应。异常另记录每实例内/非缺陷环带峰值；不能默认全图峰值在缺陷内。重点位置的 stencil 逐层记录所有参与 patch 和权重，保留八个候选及保留/移除状态。对应图展示最大插值权重 token，全部参与者在 CSV；图注明确该 token 单元不是完整感受野。

每图共同 score 色标，差分独立共同色标；对应图包含 support manifest 编号、路径、patch flat/grid、原图投影和 Source memory 行号。语义字段默认 unreviewed，可人工使用“产品内部/产品边缘/图像截断处/背景/无法判断”，工具不自动赋值。

距离移除效应按层先算每图内 patch 比例再跨图等权。实例 margin 数量按实例计数，分布保存 min/max/p10/p50/p90，保留标注碎片；不以碎片包装小缺陷结果。`margin_unchanged` 使用既定1e-5数值容差，原值及变化均保留。正常高分/困难异常是排序诊断，不称为固定阈值误报/漏检。

集合缩小必然使 patch 距离不下降；原正尺度 MAD、等权和非负双线性权重也使对应响应/图像 max 不下降。正常—异常排序和实例 margin 可以改善或恶化，不具有同样单调性。单一 margin 恢复不说明整体删除更好；无变化可能是支持冗余。报告分别陈述数值事实、干预变化、人工判读、待验证解释和未验证/不可获得信息，不自动形成研究结论。

## 同步文件与覆盖范围

只需要同步以下代码到服务器项目根的同名相对位置，由用户手动上传覆盖：

1. 新增整个 `DINOv3/MADEqual/aebad_s_union_support_loo/`（含 README 和默认 preflight 脚本）。
2. 修改 `DINOv3/MADEqual/robustad_target_fewshot_p0/scoring.py`（完整覆盖该单文件）。
3. 新增测试 `DINOv3/tests/test_mad_equal_aebad_s_union_support_loo.py`。

其余宿主、MAD、support、replay、诊断/追踪模块需服务器现有版本与本地一致；不复制这些系统实现，也不覆盖任何实验输出/旧报告。测试复用既有专项测试的纯 Python 辅助函数，因此服务器若运行测试需保留 `test_mad_equal_aebad_s_target_support_trace.py`。运行工具不依赖测试模块。

本地 PowerShell 如需打代码包，用户可执行（本轮没有自动创建这个包）：

```powershell
Set-Location 'G:\codex\workspace\experiment'
tar -czf downloads/aebad_s_union_support_loo_code_20261007.tar.gz DINOv3/MADEqual/aebad_s_union_support_loo DINOv3/MADEqual/robustad_target_fewshot_p0/scoring.py DINOv3/tests/test_mad_equal_aebad_s_union_support_loo.py
```

在服务器手动上传这个包到 `/root/autodl-tmp/` 后，从项目根覆盖代码：

```bash
cd /root/autodl-tmp/DINO-NVS || exit 1
tar -xzf /root/autodl-tmp/aebad_s_union_support_loo_code_20261007.tar.gz
```

不在工具里加入 SSH/SCP、上传、自动关机或 Git 操作。

## 本地验证边界

测试命令：

```powershell
& 'D:\anaconda\miniconda\envs\chinamoblie\python.exe' -B -m unittest DINOv3.tests.test_mad_equal_aebad_s_union_support_loo DINOv3.tests.test_mad_equal_aebad_s_target_support_trace DINOv3.tests.test_mad_equal_aebad_s_replay12 DINOv3.tests.test_mad_equal_aebad_s_union_diagnostics -v
```

轻量测试用真实 scoring 函数 AST 和小型 NumPy检索依赖验证日志开关、一次完整行检索和 positions 映射；这不是 CUDA kernel 测试。MAD/评分顺序使用原代码的纯函数 AST，地图分析生产路径始终调用 MapBackend。原 Torch 双线性插值、真实 CUDA、完整 backbone、100K Source bank、支持特征/真实 NN 对应及297张真实结果都必须在用户后续执行时验证。本地缺可用地图后端时，不实现另一种插值来冒充正式分析。

2026-10-07 本地检查记录：Python 3.11.12；本轮20项专项加既有57项轻量回归，共77项通过。新增离线流程测试使用合成 capture 和显式地图测试依赖，验证12图/10含参照分支的输出，以及GT缺失时仍为 partial_evidence，不能作为真实地图后端验证。本地真实归档的 `preflight --no-write` 返回2/blocked，报告原 memory、DINO checkpoint、DINOv3官方代码路径缺失；确认未导入Torch、未创建实验输出。语法检查在内存中执行，不写 pyc。另用临时合成输入检查实际绘图函数，11张PNG均可读取，不保留测试图。启动脚本为LF行尾；本地未发现可用的独立Bash解析器，未执行Bash脚本，请服务器运行前用 `bash -n DINOv3/MADEqual/aebad_s_union_support_loo/scripts/run_aebad_s_union_support_loo.sh` 检查。

后续 P2 修复验证：专项增至22项，全部通过。地图和校准响应同时加1的模拟中，stencil仍内部一致，但24项逐图Source/K8正式分数复核失败，analyze返回3/partial_evidence；capture raw哈希保持不变。全域基线汇总测试直接读取原完整background CSV，Source/K8的AUROC、AP均复核，超出1e-5时不通过。该验证使用合成后端，没有执行真实模型、原Torch地图后端或CUDA。
