# AeBAD-S illumination / view：冻结 SEGMENT / PAIR 验证

本工具迁移 background 已确定的规则，使用每个新域原先的八张正常支持，检验图像收益与定位损失能否重复。不重新选支持、不选参数，不把新域称为完全盲测，也不把接近 Union 写成超过 Union 或 SOTA。

## 2026-10-17：Kaggle 跨环境历史分数复核

本轮用户选择从头运行，历史 CURRENT 分数的准入改为固定 `atol=1e-4, rtol=0`，在新运行前写入协议和 joint identity。它仅用于 Source/Union 与历史 CSV 的分数比较，不改变三分支算法、提取、原 Exact 检索、MAD、插值、CH1 或评价；不根据指标选择该值，也不在运行中自动提高。新运行必须使用新目录，旧失败缓存与结果保持原样，不能混用旧 identity。

原因：用户提供的 T4 运行在第42张 `illumination/test_0240` 上出现 Source `1.95503235e-5`、Union `1.62124634e-5` 的历史分数绝对偏差。CPU 对照中使用原 Torch 插值仍超出原 `1e-5`，所以没有更改插值实现。此次改动是明确的跨环境验收规则调整，不是已经定位或消除了数值差异；不能把兼容通过写成原严格复现通过。

`capture / predict / evaluate` 共用这项规则。每次历史复核记录实际分数、历史分数、有符号误差、原 `1e-5` 的 `strict_verified` 和新门槛的 `verified`。`1e-5 < error <= 1e-4` 标为 `compatibility_warning` 并继续；超过 `1e-4`、NaN/Inf、同次计算不一致、文件身份/哈希变化、缺图/GT等仍阻断。所有同运行内部数值检查保留原容差。

输出 `historical_score_audit.json`（capture 中放在 `cache_status.json.historical_score_audit`），区分严格失败与兼容拒绝。CPU评价额外输出 `historical_baseline_comparison.csv`，对 Source/Union CURRENT 分别计算本次与历史的 Image AUROC/AP、有符号分数偏差及正常—异常排序变化；它不是 CH1 指标表，也不是历史像素指标复现。AUPRO和分域汇总继续携带这两份审计结果。是否影响结论由这些实际结果判断，不预先宣称差异无影响。

本轮仅完成实现、真实输入只读盘点、CPU 轻量验证。**没有执行两域正式提取、推理或评价。** 本地已有 CUDA 评价能力不意味着能运行完整 DINOv3。盘点见 `input_inventory_local.json`，复用依据见 `implementation_audit.md`。

## 冻结范围

| 域 | 正常 | 异常 | 评价图 | 支持图 |
|---|---:|---:|---:|---:|
| illumination | 67 | 198 | 265 | 原 target_matched 顺序 1–8 |
| view | 84 | 280 | 364 | 原 target_matched 顺序 1–8 |

`frozen_catalog.json` 来自正式 `support_manifest.csv / sample_scores.csv`，保存原 ID、路径、标签、支持顺序、历史 Source/Union CURRENT 分数、原图尺寸与图像/GT 字节哈希。启动时重新对照正式 CSV 以及原图集合，排除八张支持，不依靠计数猜测集合。两个域与代码、数学定义、Source 身份一起产生 joint identity，分批运行也不能修改其中一个域的算法。

主分支只有：

* `UNION_K8_CH1`：Source 与完整八张 Target 支持的逐支持 Exact Top1 取 min。
* `SEGMENT_K8_CH1`：原 `geometry_layer / bounded_distances`，稳定 Top2 不同支持，真实端点特征计算边长。
* `SEGMENT_PAIR_GATE_K8_CH1`：物理原生 3×3 邻域、排除中心、真实邻居中相同无序支持对票数至少 `ceil(n/2)`；n=0 拒绝。接受用 SEGMENT，拒绝回完整 Union。没有支持坐标条件，也不要求查询与支持网格同形。

维持 L7/L15/L23/L31、原 1280 维特征、640/128 窗口、原 NoAug 提取与精度、Exact NN、Euclidean/1280 一次、原 Source patch MAD、四层等权定位图和原插值。图像分数调用已验证的 `statistics / ch1_score`：插值后层峰值与原生有效 3×3 局部最大均值，各自用冻结 Source image MAD，四层平均，再以 0.5/0.5 融合。负响应不裁剪。

## 必需资产和复用优先级

本地默认根目录 `G:\codex\workspace\experiment`：

| 输入 | 默认位置 / 当前状态 |
|---|---|
| 正式六项清单及 patch MAD | `server_archives/aebad_s_target_fewshot_seed42_v1_formal_results/aebad_s_target_fewshot_seed42_v1/`；已核对 |
| 冻结 Source image MAD | `DINOv3/outputs/mad_equal/aebad_s_ch1_scoring_local_20261016_034133/image_calibration_state.json`；已核对 |
| 原图与 GT | `datasets/AeBAD.zip`；相关两域图像与 GT 字节身份已核对 |
| 完整检索与端点几何 | 默认 `DINOv3/outputs/mad_equal/aebad_s_segment_pair_crossdomain_cache_v1/`；当前无 |
| 两域可信 Q / 全量支持特征 | 当前未找到，不用 background 或其他数据集缓存替代 |
| 原 Source memory | `DINOv3/outputs/mad_equal/aebad_s_target_fewshot_seed42_v1/official_memory.pt`；当前缺失 |
| 权重 | `external_baselines/SuperADD/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth`；当前缺失 |
| DINOv3 官方代码 | 默认 `external_baselines/dinov3`；当前缺失，可显式 `--dinov3` |

不需要 Source 66 张校准缓存，不加载 GT 做预测，不重建 Source memory 或拟合 MAD。

补缓存优先级：

1. A：本工具同一 joint identity 下的完整检索/真实端点几何，直接复用。即使完整支持特征文件不在，也能 CPU 预测。
2. B：可信完整 Q 和支持特征，使用原 Exact 检索补齐记录，不运行模型。
3. C：缺 Q 或支持特征，用户显式调用 capture 后才运行原宿主，每张查询完整提取一次、每张缺失支持提取一次。

旧 Few-shot 的 `target_raw_8` 已是 Source/Target union，不能从它恢复各支持 Top1 索引。旧 background LOO/neighborhood 封装不能改域名冒充新域输入。`--reuse-cache-root` 接收本工具已验证的缓存根；没有匹配身份的旧产物会明确拒绝。

如另有原始 Q/支持缓存，使用 `--query-feature-dir / --support-feature-dir`，目录须有来源历史 `manifest.json`，包含 `format=original_query_features_v1 / original_support_features_v1`、`status=verified`、预检的 `extraction_identity`，以及 records 中的域、sample_id 或 support_index、原路径、图像 SHA、相对文件名和历史文件 SHA。**不得仅为通过检查给无来源文件现算哈希并自称 verified。** Q 使用 `layer_7/15/23/31` 原 float32 [N,1280]、完整 int64 positions；metadata 记完整提取身份、原始单位、grid、processed_shape、output_shape。支持使用原 `protocol_id/relative_path/grid/layer_*` 格式。预检只读取，不执行补提取。

可选 `--distance-reference-root` 接收原 Few-shot observation 缓存的独立历史来源清单：`format=formal_fewshot_distance_reference_v1`，`status=verified`，原 protocol/cache_revision，原 memory SHA、正式 sample_scores SHA、冻结 query/precision 和逐图 records（domain/sample_id/relative_path/image_sha256/file/sha256）。NPZ 直接保留原 `official_raw / target_raw_8 / output_shape` 等字段。程序核对 Source/Union **物理网格距离**，容差 atol=1e-7、rtol=1e-6。没有这个输入时只核验 CURRENT 分数兼容性；不能宣称历史距离逐元素一致。

## 阶段和输出

默认 `preflight` 始终只读：不创建实验目录，不导入 Torch、不初始化 CUDA、不解码 GT；可读取 GT 字节核对身份。当前本地预检预计返回 2，原因是补缓存必需资产缺失，不是轻量数学测试失败。

显式阶段：

| 阶段 | 内容 |
|---|---|
| capture | 按域保存原 Source/八支持 Top1 距离及索引、positions、真实端点、alpha、边长、原数值修正和几何诊断；保存完整支持特征及来源，不保存全部 Q 或完整距离矩阵 |
| predict | 从完整已验证缓存计算三分支并冻结；没有 GT 输入；记录 CURRENT 复核及可选原距离复核、PAIR 票数和门控 |
| evaluate | 冻结预测核验后才解码 GT；重算三分支 CH1 图像指标、完整精确 CPU pixel ROC/AP、排序、逐层 R 和实例诊断 |
| evaluate-aupro | 单独调用原 CUDA/ADEval、200 阈值、FPR≤0.3；三分支全部评价，核对地图和 mask 与 CPU 阶段逐字节相同 |
| summarize | 合并分批完成的两个域，只接受相同 joint identity、无重复域的完整结果；二域等权 macro，不混入 background |

capture 支持续跑，**先核对旧逐图 SHA，再复核数值**；旧记录、历史调用计数保留。修改过、漏失、有未登记 NPZ/临时文件的 raw 拒绝继续，不重置后重新认可。已登记但缺失的支持文件不阻塞完整几何缓存的 CPU 路径；若仍要补查询，则应恢复原支持文件，或用新 cache root 显式复用已完成 raw，再按受检来源准备支持。不会静默换哈希。

CURRENT：Source 与完整 Union 逐图对照正式 CSV，绝对容差 1e-5；这只是原宿主复现，不作主实验图像评分。predict 和 evaluate 都复核；evaluate 还复核实际地图峰值、CH1 与冻结预测，后端漂移会失败并保留差异。`SEGMENT <= PAIR <= UNION` 使用既有距离容差，不通过时不裁剪。

主要输出：

* 缓存：`cache_status.json`、`capture_verification.csv`、`capture_environment.json`、按域 `raw_cache/` 和 `support_features/`。
* 预测：`prediction_manifest.json`、`prediction_scores.csv`、`gate_statistics.csv`、按域 `predictions/`、`numerical_verification.csv`。
* 评价：`metrics.csv`、`metric_comparisons.csv`、`per_image_scores.csv`、`ranking_summary.csv`、`mechanism_per_image.csv / mechanism_summary.csv`、`defect_instances.csv`、`small_instance_groups.csv`、`margin_repair_per_instance.csv / margin_repair_summary.csv`、`map_manifest.csv`、`objective_report.md`、`timing.json`、`status.json`。

AUPRO 不可用时保持 `cpu_evaluation_complete_aupro_pending`，不换 CPU 定义填值。独立 AUPRO 输出补齐指标，不改 CPU 结果。缺图、缺范围内异常 GT、复核失败或预测未完整冻结，均不能标记全指标完成。

## 诊断解释

R 相对完整 Union，以原 float32 Source patch scale 加 1e-8 归一化，正常图全部 patch、异常图 adaptive-max 映射 GT patch，先图像等权、再四层等权，同时保留逐层 R_N/R_A/G、降距保留比例和零分母 NA。

实例为原图 GT 八连通分量；原图 16 像素 Chebyshev 环带排除所有 GT。保留映射消失和空环带，报告自身峰值及同一冻结 Union 峰值的响应/margin。每个新域按原面积、sample_id、instance_id 稳定取最小 ceil(N/4) 作为小实例组，保留 NA；不混为 background 三个最小实例的统计。

margin 逐实例对齐，区分 SEGMENT 原损失完全/部分修复、剩余/加重以及 PAIR 新增损失；单列环带恢复多于缺陷、环带恢复而缺陷未恢复。排序使用正常—异常错误对，并列计 0.5，不解释成固定阈值误报/漏检。报告分别回答两个域图像收益、定位损失与正常降距保留、跨域重复情况及环带问题；尚无真实结果时不预填结论。

## 本地只读命令和轻量测试

在工作区 PowerShell：

```powershell
Set-Location 'G:\codex\workspace\experiment'
$env:PYTHONUTF8='1'
$env:TEMP='G:\codex\workspace\experiment\downloads'
$env:TMP=$env:TEMP
$py='D:\anaconda\miniconda\envs\chinamoblie\python.exe'
& $py -B -m DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.runner preflight --no-write
& $py -B -m unittest DINOv3.tests.test_aebad_s_segment_pair_crossdomain_p0 -v
```

轻量测试包含原 background 固定 12 张真实缓存的 PAIR/CH1 等价，以及小型合成端到端 CPU 流程。模型、提取、Exact Torch 边界和 AUPRO 集成另用明确替身测试；这些测试不是新域 DINO/CUDA/真实指标验证。

## 后续获准后手动运行：完整命令

以下仅为命令说明，本轮未执行。服务器先由用户确认资产仍在；同步时需包含本新模块、正式六项元数据和上述冻结 Source image MAD 文件，保持原字节哈希。可将 `--formal-dir` 指向具有相同六个文件的原正式结果目录。数据也接受仅含 `test/ground_truth` 的解压根，不要求 `train/good`。模型阶段不支持 ZIP 外的猜测目录或自动下载。

```bash
export PROJECT_ROOT=/root/autodl-tmp/DINO-NVS
export PYTHON=/root/miniconda3/bin/python
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
cd "$PROJECT_ROOT" || exit 1

module=DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.runner
run_root="$PROJECT_ROOT/DINOv3/outputs/mad_equal/segment_pair_crossdomain_$(date +%Y%m%d_%H%M%S)"
args=(--project-root "$PROJECT_ROOT"
  --formal-dir "$PROJECT_ROOT/DINOv3/outputs/mad_equal/aebad_s_target_fewshot_seed42_v1"
  --image-state "$PROJECT_ROOT/DINOv3/outputs/mad_equal/aebad_s_ch1_scoring_local_20261016_034133/image_calibration_state.json"
  --data /root/autodl-tmp/datasets/AeBAD/AeBAD_S
  --memory "$PROJECT_ROOT/DINOv3/outputs/mad_equal/aebad_s_target_fewshot_seed42_v1/official_memory.pt"
  --checkpoint "$PROJECT_ROOT/external_baselines/SuperADD/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"
  --dinov3 /root/autodl-tmp/dinov3
  --cache-root "${run_root}_cache" --cpu-workers 8)

"$PYTHON" -B -u -m "$module" preflight "${args[@]}" --no-write
```

预检通过后，以下命令只在前一步成功时继续；失败保留现场。不会自动关机。

```bash
"$PYTHON" -B -u -m "$module" capture "${args[@]}" --domain both --device cuda:0 &&
"$PYTHON" -B -u -m "$module" predict "${args[@]}" --domain both --output-dir "${run_root}_predict" &&
"$PYTHON" -B -u -m "$module" evaluate "${args[@]}" --prediction-dir "${run_root}_predict" \
  --output-dir "${run_root}_cpu" --memory-budget-gib 8 &&
"$PYTHON" -B -u -m "$module" evaluate-aupro "${args[@]}" --prediction-dir "${run_root}_predict" \
  --cpu-dir "${run_root}_cpu" --output-dir "${run_root}_aupro" --device cuda:0 --aupro-allocator-limit-gib 1.5
```

如 ADEval 原后端不在现有环境，停止在 CPU 完成状态并保留 AUPRO pending，由用户处理环境，本工具不安装。已验证缓存重跑 capture 使用原 cache root；**预测/评价必须新目录**。分域执行时两次 capture 分别加 `--domain illumination / view`，共同使用冻结协议；predict 分域同理。两个域分别评价后可调用：

```bash
"$PYTHON" -B -u -m "$module" summarize "${args[@]}" \
  --result-dir /absolute/path/to/illumination_aupro_or_cpu \
  --result-dir /absolute/path/to/view_aupro_or_cpu \
  --output-dir "${run_root}_two_domain_summary"
```

## 资源界限

未测真实两域峰值。当前尺寸盘点的保守估计：两域完整支持约 4.32 GiB；raw 几何与三分支预测加支持合计约 11.89 GiB（未压缩估计，实际压缩不同）。逐域像素 scratch 上限 view 约 2.52 GiB；建议至少留 **20 GiB 新工作空间**，不含已有模型和 Source memory。临时文件位于本次输出目录；Windows 强制新输出在 G 盘。

精确 CPU sort 预算 illumination 约 5.01 GiB、view 约 6.70 GiB；另需解释器、地图及并行缓冲，建议可用 RAM≥12 GiB，资源不足直接失败，不降采样。默认 4 个、最多 8 个 CPU worker，内存有界；两域排序顺序进行。

完整宿主补缓存默认准入门槛为空闲显存≥20 GiB、空闲 RAM≥12 GiB；仅 Q 检索要求空闲显存≥8 GiB。当前本地 GTX1650 4GiB 不可运行完整模型路径。原 AUPRO 独立阶段可使用较小显卡，默认 allocator 上限 1.5 GiB，OOM 明确失败不替换指标。当前没有新域真实耗时数据，不据旧 FastRef OT 耗时估算本工具。

### Kaggle T4 显式尝试

`--capture-min-free-vram-gib 14` 仅将模型加载前的空闲显存准入值改为14 GiB。默认仍为20；参数须有限且≥8。该值不是显存分配上限、实测峰值或能够运行的保证。完整 batch、640切窗、float32、TF32设置、Exact NN、支持集、memory、MAD 和复现容差全部不变，不分割原提取 batch、不自动混合精度、不自动切换后端或降级。仅 Q 检索的8 GiB门槛不受此参数影响。

本次用户选择跳过额外单图测试，直接尝试完整补缓存；尚无本工具在T4上的真实运行结果。资源不足或实际OOM会停止本次调用，原错误和已登记逐图缓存保留，不自动重试。门槛、加载前空闲/总显存及无降级设置记录在 `cache_status.json` 的各次 `invocations[].resource_admission`；成功后环境文件记录实际显存峰值。续跑仍先核对所有旧哈希。

```bash
python -B -u -m DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.runner capture \
  --project-root /kaggle/working/DINO-Reference-AD \
  --formal-dir /kaggle/input/datasets/yky2026/frozen/formal \
  --image-state /kaggle/input/datasets/yky2026/frozen/image_calibration_state.json \
  --data /kaggle/input/datasets/yky2026/aebad888/AeBAD/AeBAD_S \
  --memory /kaggle/input/datasets/yky2026/temp1-pt/official_memory.pt \
  --checkpoint /kaggle/input/datasets/yky2026/checkpoint-dinov3/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth \
  --dinov3 /kaggle/working/DINO-Reference-AD/external_baselines/dinov3 \
  --cache-root /kaggle/working/aebad_segment_pair_t4_cache_v1 \
  --domain both --device cuda:0 --capture-min-free-vram-gib 14
```

执行前确认目标 cache root 属于本次运行。此次 runner/capture 代码变更会产生新的 joint code identity，不更改 catalog、提取设置或正式资产哈希；旧版本 identity 下的产物不能混入。该参数本身是运行准入设置，每次调用分别记录；不依据异常标签或指标选择数值。双T4显存不会合并，当前命令为单进程、两域顺序补缓存。
