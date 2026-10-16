# AeBAD-S background：邻域对应约束的 Target 局部参照扩展 P0

本工具检验邻域支持来源及空间对应约束能否保留 SEGMENT 的正常收益、减少缺陷损失。
background 仍是开发域，不预设收益，不把局部相容性解释为物理同源。
只读取已经冻结的 LOO B / SEGMENT 原缓存，预测不使用 RGB、label、GT、实例分组或跨图统计。
CPU 预测及评价无需 Torch；整套流程无需 checkpoint、Source memory 实体、完整支持特征或 66 张 Source 校准缓存。
实现阶段只做合成数据测试和真实输入只读预检。**用户随后追加授权，本地完整 297 张 CPU 预测、评价、固定案例绘图及两项 CUDA AUPRO 均已完成。**详见末尾本地运行记录。

## 固定方法

每层 L7/L15/L23/L31 独立处理。按 `positions` 将 compact 行还原到 query 物理网格；
支持 NN 索引用对应支持图的网格宽度解码。必须具有相同原生网格和处理尺度，不自行加入坐标变换。
原 SEGMENT 端点与八张支持 Top-1 距离的稳定 Top-2 核对，距离并列按 manifest 顺序；
近似并列仅记录，不声称索引唯一。**不重选端点，不重算 alpha 或线段参照。**

`N(i)` 是原生网格 Chebyshev 半径 1 的真实邻居，排除中心，无补零、边界复制。
支持对视为无序对。来源门控 `g_pair` 要求同支持对票数 `>= ceil(valid_neighbors/2)`。
空间门控 `g_geom` 进一步要求同一张邻居票的两个端点均满足
`max(abs((pi_k(j)-pi_k(i))-(x_j-x_i))) <= 1 support patch`。
无邻居时两门控均拒绝。规则在原始缓存上一次计算，不传播已接受结果。
重复对应可在容差内通过，单独记录重复次数，绝不宣称对应唯一。

| 分支 | 每层距离 |
|---|---|
| UNION_K8_CH1 | `U = min(source_distance, target_distance.min(axis=0))` |
| SEGMENT_K8_CH1 | 原缓存 `segment_distance`，已含完整 Union 回退 |
| SEGMENT_PAIR_GATE_K8_CH1 | `where(g_pair, segment_distance, U)` |
| SEGMENT_NEIGHBOR_GATE_K8_CH1 | `where(g_geom, segment_distance, U)` |

距离已经是 Euclidean / 1280，绝不再除一次。逐项检查
`SEGMENT <= PAIR <= NEIGHBOR <= UNION`，`atol=1e-7, rtol=1e-6`；
不通过即失败，不裁剪掩盖错误。拒绝扩展退回**完整 K8 Union**，不能退回 Source-only。
半径 1、半数票、空间容差 1 均为预设 P0 参数，没有参数搜索。

四分支共享原 Source patch MAD、等权四层定位、原双线性插值以及已完成 CH1 的
`image_calibration_state.json`。直接调用原 CPU `statistics` / `ch1_score` / `reconstruct`：
每层插值后峰值、原生有效 3×3 局部最大均值分别经过冻结 Source image MAD，
各自四层均值后 0.5/0.5 融合。保留负响应，不按图归一化，不重新拟合任何 MAD。
旧 SEGMENT CURRENT 仅用于复现检查；本轮重新计算其 CH1 图像分数。

## 实际复用入口与独立代码

| 本模块 | 作用 / 复用 |
|---|---|
| `inputs.py` | 只读 tar/ZIP 适配；复用 `union_support_loo.analysis.validate_arrays`；原清单、逐文件 SHA256、冻结 CH1 资产校验 |
| `gates.py` | 独立无标签物理网格投票；候选选择核对沿用 `target_local_segment.geometry.geometry_layer` 的稳定排序语义 |
| `prediction.py` | 有界 CPU 线程、逐图预测、冻结清单、哈希及实际内核复核 |
| `analysis.py` | 复用 CH1 `statistics/ch1_score/image_metrics/pair_errors`、CPU `numerics.reconstruct/adaptive_max`、`metrics.exact_binary`、原 `spatial.components/instance_regions` |
| `aupro.py` | 仅显式调用时导入 Torch 和 `relation_reliability.official_metrics.official_aupro_adeval` |
| `visualize.py` | 原固定 12 案例、四分支共同响应色标及逐层门控图；RGB 仅展示 |
| `runner.py` | 默认只读 `preflight`；显式 `predict/evaluate/evaluate-aupro/visualize` |
| `DINOv3/tests/test_aebad_s_neighborhood_segment_p0.py` | 原生 NumPy 门控、缓存交接、篡改、精确像素、小型评价及禁止 Torch 的测试 |

不运行旧完整 runner，不进入任何缺失补算或模型路径，不修改旧方法、结果和实验报告。

## 冻结输入

默认根目录自动定位到工程，以下相对路径在本地均已存在：

| 输入 | 默认路径 / 身份 |
|---|---|
| 完整原缓存 | `downloads/aebad_s_segment_looB_raw_cache_20261013.tar`；SHA256 `6fd4086bbaeb5318a5168dd90eef57d9955ffa37da44764db926913c3e4b2e41` |
| SEGMENT 历史结果 | `downloads/aebad_s_target_local_segment_review.tar.gz`；SHA256 `f2703ee4518330d7f82a7b1dc4477a83b05e626fb104d96f2f1c38313efa7524` |
| 原正式 CSV / MAD | `server_archives/aebad_s_target_fewshot_seed42_v1_formal_results/aebad_s_target_fewshot_seed42_v1`；按 capture 冻结的五项哈希逐一核验 |
| 冻结 CH1 | `DINOv3/outputs/mad_equal/aebad_s_ch1_scoring_local_20261016_034133`；核验状态、输入、校准状态、逐图分数、指标、排序及数值记录 |
| GT / 固定案例 RGB | `datasets/AeBAD.zip`；也支持 `--data` 指向原 `AeBAD_S` 的 `test/ground_truth` 根目录，不要求 `train/good` |

目录形式数据按 capture 中逐张图像哈希验证；ZIP 形式使用原 ZIP 哈希，并在读取图像时再核对图像哈希。
预测阶段不读取或要求数据 ZIP；预检会清点评价 GT 和案例 RGB，评价才解码 GT，可视化才解码 RGB。
297 张（85 正常 / 212 异常）、原八张支持顺序及排除规则全部冻结。
固定十张困难正常来自旧 `selected_cases`，不按新分数重选。
原三张最小实例按历史原图面积、sample_id、instance_id 排序继承，不根据新结果重选。

全 tar 只索引一次，线程各自持有只读文件句柄，按原成员 offset 读取和逐文件校验；
不解压、复制整套原始缓存，不把所有距离常驻内存。
新输出必须是不存在的独立目录，禁止与父输入、冻结预测或 CPU 评价目录重叠，不隐式续跑。
失败保留现场，退出码非 0；不会将修改后的预测重新认可为可信记录。
代码和辅助数学入口哈希进入身份；实现改变后需要独立新预测目录，不能混用旧预测。

## 阶段与完成状态

1. **preflight（默认）**：不写文件，不采样、不计算新门控/预测、不解码 GT、不导入 Torch。
   核对 297 + 1188 个缓存、清单/身份/哈希、端点、物理位置、支持几何、正式 CSV、CH1 状态及评价数据。
2. **predict**：只读取距离与索引，保存两个新距离分支、门控及票数；全部 297 张成功后才写 `prediction_manifest.json` 和 `predictions_frozen`。
3. **evaluate**：必须先通过完整冻结预测校验；GT 此后才进入评价。
   复现原 Union/SEGMENT CURRENT；复现 Union CH1 逐图分数、图像指标和实际正常—异常排序。
   分数容差 `1e-5, rtol=0`，排序独立检查，不能用分数接近替代排序核验。
   原实例/环带及 Union 固定峰值也要复现。任何失败阻止发布新指标。
4. 新两门控的完整像素 AUROC/AP 使用**全 169,746,192 像素**精确排序及 AP，不量化、不降采样。
   Union/SEGMENT 的像素指标仅在输入、定位定义和实例响应核对后引用历史值，表中标明来源。
   此阶段为 `cpu_evaluation_complete_aupro_pending`，`all_metrics_complete=false`。
5. **evaluate-aupro**：独立显式 CUDA/ADEval 原后端，200 阈值、项目原 FPR≤0.3 积分定义。
   新两分支才计算 AUPRO，CPU 指标及历史像素指标保留来源；独立输出合并表，不改写 CPU 结果。
   CPU 阶段保存地图/GT 栅格哈希；AUPRO 重建后逐图核对，后端或环境导致地图变化时明确失败，不放宽容差重新认可。
   不可用时退出失败或保留 pending，不提供慢 CPU 替代，不用空值冒充完成。
6. **visualize**：冻结预测的原 12 案例，展示原 RGB、GT、Union/SEGMENT/两门控响应及逐层门控。

## 评价及解释

- Image AUROC/AP；全部正常、原固定十张及其余正常三个分组。排序错误为正常—异常对数，严格并列权重 0.5，绝不称为部署阈值误报/漏检。
- 差值均为 `branch - reference`，指标单位 fraction；新分支分别对 Union、SEGMENT，并比较空间门控与仅来源门控。
- 实例为原图 8 连通 GT；环带为原图 Chebyshev 半径 **16 像素**且排除全部缺陷，与推理半径 **1 patch** 无关。
  各分支自身实例/环带峰值及原 Union 固定峰值响应分别保存。映射消失、空环带、零分母保留 NA。
- R_N/R_A/G **沿用旧 SEGMENT** 相对完整 Union 的口径：
  `(U - branch) / (float32(Source_patch_scale) + 1e-8)`，保留有符号值。
  正常图使用全部 patch，异常图只用原 GT adaptive-max 映射 patch；先图像等权、再四层等权。
  `G=R_N-R_A`。降距保留量为该图该层新降距总量 / 原 SEGMENT 降距总量，
  逐图逐层比值后作图像/层等权汇总；零分母为 NA，另计数量。位置保留比例另行记录。
- `gate_statistics.csv` 为无 GT 逐图逐层结果，包含接受率、空间额外拒绝率、真实降距比例、
  邻居/票数直方图、边界计数、局部重复票以及选定支持 patch 的全图重复对应次数。
  降距位置计数使用冻结数值容差：`U - branch > 1e-7 + 1e-6*abs(U)`；这只定义统计计数，
  不改变门控或距离。连续降距量、R 和保留量比值仍使用原有有符号差值，不裁去小差异。
- `mechanism_per_image.csv` 在冻结后再分正常/缺陷 patch，区分正常收益保留与缺陷降距保留。
  缺陷降距减少不等于实例 margin 必然恢复；同时看实例 margin 相对 Union 和 SEGMENT 的变化。
- 接受集合接近空时接近 Union 只是回退；距离包含关系也不保证 AUROC/AP/margin 改善。

## 手动运行命令（每次使用新目录）

本地 PowerShell。建议先释放足够 RAM；独立运行每一步，失败即停止，不要跳过冻结检查：

```powershell
Set-Location 'G:\codex\workspace\experiment'
$Py = 'D:\anaconda\miniconda\envs\chinamoblie\python.exe'
$env:PYTHONUTF8 = '1'
$env:TEMP = 'G:\codex\workspace\experiment\downloads'
$env:TMP = $env:TEMP
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$Base = 'G:\codex\workspace\experiment\DINOv3\outputs\mad_equal\aebad_s_neighborhood_segment_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
$Pred = "${Base}_predict"
$Eval = "${Base}_cpu"
$Fig = "${Base}_figures"

# 只读；默认即此阶段，--no-write 明确表示只读，不创建 OUTPUT。
& $Py -B -m DINOv3.MADEqual.aebad_s_neighborhood_segment_p0.runner preflight --no-write --cpu-workers 4
if ($LASTEXITCODE -ne 0) { throw '预检失败，停止' }

# 以下为完整 CPU 流程的手动复现命令；每次均新建独立结果。
& $Py -B -m DINOv3.MADEqual.aebad_s_neighborhood_segment_p0.runner predict --output-dir $Pred --cpu-workers 4
if ($LASTEXITCODE -ne 0) { throw '预测失败，保留现场' }
& $Py -B -m DINOv3.MADEqual.aebad_s_neighborhood_segment_p0.runner evaluate --prediction-dir $Pred --output-dir $Eval --cpu-workers 4 --pixel-sort-budget-gib 8
if ($LASTEXITCODE -ne 0) { throw 'CPU评价失败，保留现场' }
& $Py -B -m DINOv3.MADEqual.aebad_s_neighborhood_segment_p0.runner visualize --prediction-dir $Pred --output-dir $Fig
if ($LASTEXITCODE -ne 0) { throw '案例绘制失败，保留现场' }
```

每个输入可独立覆盖：`--raw-archive --review-archive --formal-dir --ch1-dir --data`。
后续获准的 CUDA 机器手动补 AUPRO（此处无连接命令，不自动运行、不自动关机）：

```bash
export PROJECT_ROOT=/root/autodl-tmp/DINO-NVS
export PYTHON=/root/miniconda3/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$PROJECT_ROOT" || exit 1

# 先把同一套缓存包、少量 formal/CH1 文件、冻结预测目录及 CPU 结果放到已确认的路径。
# 不要求上传 Source memory、支持特征或模型。这里使用环境变量接收真实路径，避免猜测旧目录还在。
: "${PREDICTION_DIR:?请填写实际冻结预测目录}"
: "${CPU_ANALYSIS_DIR:?请填写实际CPU评价目录}"
test -f "$PREDICTION_DIR/prediction_manifest.json" || exit 1
test -f "$CPU_ANALYSIS_DIR/analysis_manifest.json" || exit 1
export AUPRO_OUTPUT_DIR="${CPU_ANALYSIS_DIR}_aupro_$(date +%Y%m%d_%H%M%S)"

"$PYTHON" -B -u -m DINOv3.MADEqual.aebad_s_neighborhood_segment_p0.runner evaluate-aupro \
  --prediction-dir "$PREDICTION_DIR" --cpu-analysis-dir "$CPU_ANALYSIS_DIR" \
  --data /root/autodl-tmp/datasets/AeBAD/AeBAD_S \
  --output-dir "$AUPRO_OUTPUT_DIR" --device cuda:0
rc=$?
printf '\nAUPRO_EXIT_CODE=%s\nOUTPUT=%s\n' "$rc" "$AUPRO_OUTPUT_DIR"
```

默认其余输入相对工程位置与本地一致；若服务器不同，用上述输入参数明确覆盖。
CUDA 后端和 ADEval 的可用性需用户在获准机器验证。任何失败均保留现场，不关机。

## 资源与输出

逐图计算，有界默认 4 CPU worker；不把所有地图保留在队列，完整像素按两个新分支依次精确排序。
新距离及门控未压缩约 **0.3 GiB**，压缩后体积取决于门控分布；CSV/固定案例再预留约 0.1 GiB。
像素 scratch 为两份 float32 图和一份 uint8 标签，约 **1.42 GiB**，只在新 CPU 输出目录 `_scratch`，结束后关闭并清除。
精确排序保守工作预算约 **5.56 GiB**，加工作余量至少约 **6.06 GiB 可用 RAM**；
建议实际预留 **8–10 GiB 可用 RAM**，系统总 RAM 不等于可用 RAM。默认排序预算 8 GiB，内存不足明确失败，不切近似指标。
这些是估计，不是本轮实测正式峰值；建议新输出盘可用空间至少 **3 GiB**，另留系统余量。
本地新输出、scratch、绘图缓存都限制在 G 盘；不复制 1.17 GiB 原 tar。

预测目录：`protocol.json`、`input_manifest.json`、`prediction_manifest.json`、
`raw_cache/*.npz`、`gate_statistics.csv`、`numerical_verification.csv`、`timing.json`、`status.json`。
距离在 compact 顺序，门控/票数为四层物理网格，metadata 明确布局。
缓存含 pair/neighbor 距离、门控、有效邻居数、两种票数及重复对应票数。

CPU 评价目录：`protocol.json`、`input_manifest.json`、`analysis_manifest.json`、
`metrics.csv`、`metric_comparisons.csv`、`per_image_scores.csv`、`ranking_summary.csv`、
`gate_statistics.csv`、`distance_audit_summary.csv`、`defect_instances.csv`、`peak_positions.csv`、
`mechanism_per_image.csv`、`mechanism_summary.csv`、`instance_margin_summary.csv`、
`map_manifest.csv`、`numerical_verification.csv`、`timing.json`、`status.json`、`objective_report.md`。
CPU 结果清楚区分新计算、历史像素引用、待完成的两项 AUPRO。
CUDA 延续目录独立保存合并指标、差值、成本和来源哈希，不改写 CPU 目录。

## 本轮验证边界

测试命令：

```powershell
& 'D:\anaconda\miniconda\envs\chinamoblie\python.exe' -B -m unittest DINOv3.tests.test_aebad_s_neighborhood_segment_p0 -v
```

实际生产 NumPy 内核覆盖物理重排、无序对、平移相容、空间跳变、中心/边界、重复对应、
集合/距离包含、全接受/全拒绝、标签不影响预测、旧预测篡改拒绝、缺失预测拒绝。
小型评价集成使用合成四图：替换文件 admission/GT IO/缓存 handoff，真实执行评分、
GT 区域、排序、精确像素及报告代码；不冒充真实 297 张验证。
另验证评分后端整体偏移时禁止发布新指标，默认预检及小型 CPU 评价禁止导入 Torch。

2026-10-16 已执行：本模块 **24 项**专项测试通过（3.97 秒），复用评分/插值/精确像素数学的 **10 项**既有测试通过；模块及测试内存编译通过，未写 pyc。
真实只读预检核验 297 份 LOO + 1188 份 SEGMENT 缓存，正式分数/支持身份/CH1 状态齐全，缺项为 0。
最终代码预检 19.49 秒，在禁止 Torch 导入的进程中通过；未创建实验结果。
预检时可用 RAM 约 3.49 GiB，低于正式精确像素评价最低约 6.06 GiB；这不阻塞只读预检，但正式评价会明确阻断，需先释放内存。
该预检没有计算新门控或进行全域评价；输入齐全不代表正式数值复现、效果和资源峰值已经验证。

实现交付时上述真实全域环节尚未运行；用户后续授权的 CPU 实际运行情况如下。
CUDA/ADEval AUPRO 的后续实际记录见下文；跨域验证未开展，不能据本开发域结果宣称方法普遍有效。

## 后续用户授权的本地实际运行（2026-10-16）

结果均在工程 `DINOv3/outputs/mad_equal/` 下，公共前缀
`aebad_s_neighborhood_segment_local_20261016_134948`：

| 目录后缀 | 内容 | 实际结果 |
|---|---|---|
| `_predict` | 冻结的 297 张新距离、门控、票数及清单 | 退出码 0，53.25 秒 |
| `_cpu` | 图像指标、新两分支全像素 AUROC/AP、实例/机制、报告 | 退出码 0，359.63 秒 |
| `_figures` | 原固定 12 案例图 | 退出码 0，50.95 秒 |

含预检的总墙钟时间 483.63 秒（约 8 分 4 秒）。8842 项真实复现/数值检查全部通过。
CPU 评价进程峰值 RSS 3,918,909,440 字节（约 3.65 GiB）；该次运行开始前可用 RAM 约 8.65 GiB。
完整 169,746,192 像素精确评价实际执行，像素排序共 100.96 秒，scratch 已清除。
该 CPU 目录中 Union/SEGMENT 的像素值仍明确引用历史；新两分支 AUPRO 留空，状态
`cpu_evaluation_complete_aupro_pending`，不是全指标完成。
全程未导入 Torch、调用 GPU、模型或特征提取，未连接服务器。
具体数值见 `_cpu/metrics.csv`、`metric_comparisons.csv`、`objective_report.md`，旧实验报告未修改。

### 本地 CUDA AUPRO 延续（用户另行授权）

结果目录：`DINOv3/outputs/mad_equal/aebad_s_neighborhood_segment_local_20261016_134948_aupro_20261016_171444`。
297 张两项新增 AUPRO 均成功，退出码 0，完整指标表见该目录 `metrics.csv`。
状态为 `all_requested_metrics_complete`。原 CPU 目录及父资产均未改写。

- 本地 GTX 1650 4 GiB；实际已有可用 PyTorch 2.5.1+cu121，CUDA 12.1、OpenCV 4.10.0。
- 仅将官方 ADEval 1.1.0 安装到 `downloads/adeval_1_1_0_runtime`，使用 `--no-deps --no-compile --target`，未改动现有 Python 环境。
  Wheel SHA256：`125caf991037a668a1f072f7e99c81b6bdf86dc8c30cf1699c731d4b79e0bd6e`。
- 小型合成 CUDA 验证和原尺寸固定 12 案例验证均先通过；后者地图/GT 与冻结 CPU 哈希一致。
- 正式阶段使用原 `official_aupro_adeval`，未改源码、分批逻辑、200 阈值或 FPR≤0.3 定义。
  PyTorch 分配上限设为显存的 0.375（约 1.5 GiB）；独立子进程可用 RAM 连续低于 1.5 GiB 时终止该子进程。
- 总墙钟 224.07 秒；两项地图/GT 重建 101.05/97.57 秒，实际 AUPRO 累计分别 3.45/3.14 秒。
  GPU 峰值分配 871,661,568 字节（约 0.81 GiB），峰值保留 874,512,384 字节；进程峰值 RSS 2.32 GiB。
  系统最低可用 RAM 5.83 GiB，资源保护未触发。进程退出后 GPU 占用回到约 43 MiB。
- 来源门控 AUPRO：`0.896267534444162`；空间门控 AUPRO：`0.8990751751380255`。
  图像指标、像素 AUROC/AP 继承已核验 CPU 评价；Union/SEGMENT 像素指标仍保留历史引用标记。
- `runtime_environment.json`、`timing.json`、旁侧 `.resource_guard.json` 记录实际环境、成本和保护参数。
  ADEval 对未提供的 image/sample 累计项会输出除零警告；本次仅使用其 pixel AUPRO，两项正式值均有限并成功返回。

未连接服务器，未重跑模型或特征提取，旧实验报告保持不变。
