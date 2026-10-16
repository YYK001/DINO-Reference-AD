# AeBAD-S/background 初始投影逐层区分与融合贡献诊断

独立 CPU 离线诊断。研究投影前后的单层区分、局部实例 margin、同位置层贡献、峰值移动及响应尺度；不增加检测方法，不选择最佳层、权重、预算或阈值。父实验与投影工具、历史结果和报告保持只读。

默认 `preflight` 只验证本地输入、缓存结构/历史哈希和预计资源，不创建结果、不进行全域分析。`analyze` 必须显式执行，供独立 review 后运行。两个阶段均不导入 Torch、不调用 GPU、提取特征、加载模型、重新检索、执行 OT 或采样。

## 输入映射与身份

默认路径相对于项目根目录；可以通过 CLI 指定这四个本地输入，但内容必须保持冻结身份。只读流式读取归档，不解压。

| 输入 | 默认本地位置 | 使用内容 |
|---|---|---|
| FastRef v3 父包 | `downloads/aebad_s_fastref_target512_source_bounded_density_v3_20261013_211140_results_20261014.tar.gz` | inputs/infer/prepare 状态、正式 raw 297 份、历史分数/指标/实例/固定峰值 |
| 投影包 | `downloads/aebad_s_fastref_projection_target512_source_20261014_121828_results_20261014.tar.gz` | inputs/infer 状态、正式 raw 297 份、历史分数/指标/实例 |
| Source MAD | `server_archives/aebad_s_target_fewshot_seed42_v1_formal_results/aebad_s_target_fewshot_seed42_v1/source_mad_state.npz` | 原 float32 medians/scales，SHA256 `65df0a748d918f20e5c85f2dbb73845a2fac2801d8d62a91e3bd0b7bb37e1a13` |
| 图片与 GT | `datasets/AeBAD.zip` | ZIP 前缀 `AeBAD/AeBAD_S/`，按 capture.samples 和固定八支持 manifest 读取 |

`inputs.py` 固定了两份已确认结果包的 SHA256。包内 inputs 身份摘要、infer 状态、逐图 raw 哈希、297/85/212 集合、固定 12 案例和八支持顺序/排除、P 原型身份记录、图像身份、几何/层数量/float32、positions 的物理位置双射全部检查。配对 Source 采用原 `atol=1e-7, rtol=1e-6`，P union 必须与缓存 `min(Source, projection)` 完全一致。只使用 `raw/`，不混入 `smoke_raw/`。

服务器绝对路径仅作为历史 provenance，不尝试打开。原 memory、checkpoint、支持特征、Q 或 P/B 矩阵均不加载；prepare JSON 只用于核对已冻结原型身份。MAD 映射文件与两个输入清单中的历史 SHA 匹配。ZIP 只检查准确 manifest 中的 305 张 RGB（297 查询+8支持）和 212 份 GT，不扫描文件夹重新生成评价集。GT 路径为 `test/` 开头替换成 `ground_truth/`。

## 冻结数学定义与复用依据

```text
U_l = min(parent.source_distance_l, parent.target_full_distance_l)
P_l = projection.projection_union_distance_l
z_l = (d_l - Source_median_l) / Source_scale_l
FUSED = original_bilinear(mean_l(z_l), original_output_shape)
image_score = max(corresponding_output_map)
```

先通过 `physical[:, positions] = cached_columns` 恢复 patch 网格。MAD 来源于 `relation_oracle_cache/extract.py:mad_calibrated/mad_equal`，使用原 float32 计算，不加 epsilon、不截负值、不用 ReLU。图像分数不是四层最大值的均值。

原完整地图入口会导入 Torch。这里在 `numerics.py` 单独封装必要的 NumPy 计算，参照 `superadd_host.interpolate_score/mask_at_patch_grid` 和原 `spatial.interpolation_stencil`：bilinear、align_corners=False、半像素坐标、边界夹取；异常图插值不使用 PIL。GT 评价尺寸按原 PIL NEAREST，patch GT 按 adaptive-max 的 floor/ceil 重叠区间计算，不替换成最近邻缩放。

直接复用原 `aebad_s_union_diagnostics.spatial.components/instance_regions`：8 连通、原图像素半径 16 Chebyshev 环带、排除全部 GT、原 NEAREST 映射。这些复用函数不需要 Torch。

NumPy 与原 ATen bilinear 不宣称逐位相同。逐图融合分数、历史原峰值响应、可计算实例响应、逐层图均值与先融合再插值、固定位置 margin 分解沿用既定 `abs<=1e-5`。保留实际误差。历史融合正常—异常排序对要求完全一致，不能只检查 AUROC。融合 Image/Pixel AUROC/AP 的历史复核沿用原汇总 `abs<=1e-5`。不能通过这些门禁时保留 `partial_evidence`，报告禁止确定性归因；没有容差 CLI 或失败后放宽。

## 精确 CPU 指标和资源

U/P 每个分支分析 L7/L15/L23/L31 和 FUSED，共十种响应。每种输出 Image AUROC/AP、全评价像素 Pixel AUROC/AP、每张异常图内部 Pixel AUROC/AP 及各自有效数量。原 GT 映射后无正像素时不填零，保留不可计算；全正 GT 的 AP=1，但 AUROC 未定义，两项有效数量分开记录。

`metrics.exact_binary` 按全部原 float32 分数精确 argsort，把每个相同分数的正/负像素计数完整聚合；跨处理块的并列组继续合并。AUROC 用正像素胜过负像素的数量（并列半权），AP 按整组加入时的 precision×recall 增量计算。与 sklearn 的小型独立参考测试对照。**处理块仅限制工作内存，不抽样、不直方图量化或近似**。全域逐响应排序，不同时做十次大排序。不计算 AUPRO，也不调用 ADEval。

图像重建/GT实例步骤以有界线程队列并行（默认4，最多8），严格按 manifest 顺序写数据；不会让 297 张地图同时积压在 RAM。十种响应和一份 GT 写入新输出目录的 memmap 临时文件，关闭映射后逐响应精确排序。

实际输入共有 **169,746,192 个评价像素、4,135,428 个 patch**。资源公式是保守估计，非真实峰值：

- 精确排序 RAM 估计 `32×pixel_count + 512 MiB`，约 5.56 GiB。
- 输入缓存约 157.8 MiB；每个并行图像 worker 预留 256 MiB。4 worker 总可用 RAM 准入约 **6.71 GiB**。
- 十个 float32 响应和一份 uint8 GT 临时文件约 **6.48 GiB**，加 1 GiB 表格/安全余量，磁盘准入约 **7.48 GiB**。
- 默认排序预算8 GiB。预算/系统可用 RAM 或磁盘不足则明确阻断，不能偷偷抽样/缩图/换指标。`--memory-budget-gib` 是资源上限，不是方法参数；`--cpu-workers` 只影响执行方式。
- 成功后删除本次新输出中的临时 memmap；失败保留现场，不覆盖已有目录。

不预估全域耗时；没有实际运行这十个全像素排序。全域排序是 CPU 阶段，Python/NumPy 的这条 argsort 路径本身不会因为增加图像 worker 而多线程加速。

## 实例分解与响应尺度

原小实例标识从历史实例表读入并检查 U/P 一致，不按本轮效果选面积阈值。映射后消失或空环带保留空响应与状态。各层/融合分别取自己的实例、环带最大值及 margin，不能用单层 margin 均值代替融合 margin。

固定位置直接复用父历史表中 U 融合的实例峰值 `x_in` 和环带峰值 `x_ring`，并核对这些位置仍在原区域、响应与原峰值一致。对每层计算：

```text
c_l = (P_l(x_in)-U_l(x_in)-P_l(x_ring)+U_l(x_ring)) / 4
e_in = max_inside(P_FUSED)-P_FUSED(x_in)
e_ring = max_ring(P_FUSED)-P_FUSED(x_ring)
margin_P-margin_U ≈ sum(c_l)+e_in-e_ring
```

P 的峰值移动与固定位置贡献分别保存，并校验上述恒等式。自身峰值和历史峰值坐标的近并列变化也保留，不把四个不同位置的单层峰值混为一个位置。

响应尺度、距离变化、相关性在正常全 patch、异常非缺陷 patch、原 adaptive-max GT patch 三个区域分别统计。`pooled_patch` 为所有 patch 合并后的精确均值/标准差/分位数/Pearson；`image_equal` 为有效图像等权的图内统计均值。图内分位数的均值不是池化分位数；空/常量区域保留不可计算状态。统计 raw/MAD 并列值数量，单层固定正仿射 MAD 不作为第二种方法收益。

R_N/R_A/G 仍用原 Source MAD，`R=(U-P)/scale`，正常全 patch、异常 GT patch，先图像等权再四层等权。平均距离降低不等于排序能力降低。

## 本地命令：先 preflight，review 后再 analyze

在本地 PowerShell 执行；无服务器操作、无安装依赖、无需启动脚本。

```powershell
Set-Location -LiteralPath 'G:\codex\workspace\experiment'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'

& 'D:\anaconda\miniconda\envs\chinamoblie\python.exe' -B -m DINOv3.MADEqual.aebad_s_projection_layer_diagnostic.runner preflight --no-write
```

review 通过、资源准入通过后，手动运行完整域：

```powershell
$layerDiagnosticOutput = Join-Path 'G:\codex\workspace\experiment\DINOv3\outputs\mad_equal' ('aebad_s_projection_layer_diagnostic_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
& 'D:\anaconda\miniconda\envs\chinamoblie\python.exe' -B -u -m DINOv3.MADEqual.aebad_s_projection_layer_diagnostic.runner analyze `
  --output-dir $layerDiagnosticOutput `
  --cpu-workers 4 `
  --memory-budget-gib 8
if ($LASTEXITCODE -ne 0) { Write-Host '分析未完成，请查看输出中的 status.json 和 numerical_verification.csv；保留现场。' }
```

可选本地映射参数：`--project-root`、`--parent-archive`、`--projection-archive`、`--mad`、`--data-zip`。映射不重写包内服务器路径；不得以新内容冒充冻结输入。现有归档全部保留。

## 输出

| 文件 | 内容 |
|---|---|
| `inputs.json`, `status.json` | 本地映射、冻结输入/代码哈希、完成门禁、耗时/观察到的RSS |
| `layer_metrics.csv`, `layer_metric_differences.csv` | 两分支各五种响应的全域指标，P−U，0–1单位 |
| `per_image_scores.csv` | 对应地图最大值、各自峰值坐标、历史融合峰值/近并列变化 |
| `within_image_pixel_metrics.csv`, `within_image_pixel_differences.csv` | 212张异常图各十种响应的图内Pixel指标、AP/AUROC各自有效状态及P−U差值 |
| `ranking_summary.csv`, `ranking_per_image.csv` | 同层/融合 P−U 的排序变化，固定十张、其余75、全部85正常；非部署阈值错误 |
| `fixed_case_scores.csv` | 原固定12张案例的各层/融合响应分数，固定集合而非按本轮效果筛选 |
| `defect_instances.csv`, `instance_margin_summary.csv` | 全部原实例×两分支×五响应，自身峰值/margin、原小实例标识、不可计算状态 |
| `fixed_position_layer_contributions.csv`, `instance_peak_movement.csv` | 固定原 U 两峰值的 c_l、e_in/e_ring、margin分解及残差 |
| `layer_response_summary.csv`, `layer_response_per_image.csv` | 三种区域的响应分布、raw距离变化、MAD变化及明确池化/等权口径 |
| `layer_correlation_summary.csv`, `layer_correlations_per_image.csv` | 层间Pearson、常量/不足区域及有效数量 |
| `signed_distance_summary.csv` | 逐层/四层平均的 R_N、R_A、G，无额外epsilon |
| `numerical_verification.csv` | 逐图、排序对、实例、固定位置、分解和历史指标复核的实际误差 |
| `diagnostic_report.md` | 最后回答四个归因问题；失败/不完整时不输出确定性结论 |

报告以冻结 L7/L15/L23/L31 顺序呈现。优先方向的描述依据同时出现单层两项全像素指标下降超出既定汇总数值容差、实例 margin 均值超出数值容差下降且负变化实例多于正变化；明确这是描述性证据汇总，不是因果检验、显著性检验或调参规则。也允许证据不足或两种现象并存，不能据“某层好于融合”认定MAD唯一有问题。

## 轻量验证与本轮边界

```powershell
& 'D:\anaconda\miniconda\envs\chinamoblie\python.exe' -B -m unittest DINOv3.tests.test_mad_equal_aebad_s_projection_layer_diagnostic -v
```

测试包括独立半像素插值参考、重叠 adaptive-max、非恒等 positions、负 MAD、融合最大值规则、精确ROC/AP与 sklearn 对照及跨块并列、不可计算像素状态、固定位置/双峰移动分解、来源/union语义、无Torch默认预检、有界并行顺序、输入保护、合成297图完整流程及失败门禁。

另有一张真实缓存 `test_0023` 的 U 融合地图数值复核，仅读取现有距离与 MAD，没有特征提取、检索或全域指标。其余完整流程使用小型构造数据，不冒充真实全域分析。

实现交付时的本地输入预检：所有输入身份/594个正式raw/305个RGB与212个GT验证通过；未解压归档、未导入Torch、未加载模型或矩阵。当时资源预检因可用RAM约2.86 GiB、低于预计6.71 GiB而阻断，真实全域分析尚未执行。后续实际运行记录见下。

本轮18项专项测试通过（18.486秒），8个Python文件AST检查通过，README中的PowerShell命令语法检查通过。最后补充复用源文件哈希记录后，合成全流程测试再次单独通过；这不构成真实297张全域评价已运行。

## 2026-10-15：获准后的本地全域运行

用户释放内存后，可用RAM约7.7 GiB，输入和资源预检通过。随后实际执行297张图、两分支各五种响应、每种169,746,192个原评价像素的完整CPU分析，退出码0，状态`analysis_complete`，5970项数值复核全部通过，输入及代码保护通过。

输出位于项目内 `DINOv3/outputs/mad_equal/aebad_s_projection_layer_diagnostic_local_20261014_235403/`（目录时间为开始时间）。总进程耗时653.8秒，分析函数639.442秒；进程峰值RAM约7.30 GiB，高于此前6.71 GiB准入估计，因此资源公式仍不能当作实测峰值或硬上限。新输出约4.27 MiB，临时排序文件成功后已清理。输出、旁侧`.log`/`.exitcode`及程序临时数据均放在G盘。

历史融合分数/可计算实例响应复核的最大绝对误差约6.36e-6；固定位置层贡献加峰值移动的margin恒等式最大误差约1.35e-6，均未放宽1e-5容差。没有连接服务器、导入Torch、调用GPU、模型/原型加载、特征提取或重新检索；正式历史结果未修改。具体诊断数据及有边界的结论见该输出目录的`diagnostic_report.md`和全部CSV。
