# AeBAD-S/background 固定 K8 有界局部参照

本工具只使用已验证的全域 B capture、原八张 Target-normal 支持特征和原 Source MAD，离线比较 `K8_UNION`、`K8_MIDPOINT`、`K8_SEGMENT`。不运行 backbone、提取特征、训练、重新检索、重建 memory、重拟合 MAD 或选择支持。默认入口仅 `preflight`；正式计算须显式 `analyze`。本轮只完成本地代码和轻量验证，没有连接或执行服务器。

这是固定候选下的方法筛选原型，不是全库 Exact 线段检索。两张支持图各自的 Top-1 候选不是同一物理 patch；正常端点的组合没有正常性或缺陷保持保证。本轮不开展 K1/K2/K4 实验，不用 GT 选择分支或“最佳支持”。

## 输入与复用路径

服务器历史记录位置（未远程检查，使用前由用户确认）：

| 输入 | 默认位置 |
| --- | --- |
| 可信 B capture | `/root/autodl-tmp/DINO-NVS/DINOv3/outputs/mad_equal/aebad_s_union_support_loo_B_20261007_230639` |
| 原正式结果 | `/root/autodl-tmp/DINO-NVS/DINOv3/outputs/mad_equal/aebad_s_target_fewshot_seed42_v1` |
| 八张原支持特征 | `/root/autodl-tmp/DINO-NVS/DINOv3/outputs/mad_equal/aebad_s_union_background_replay12_seed42_v1/support_features/target_matched/background` |
| 数据根 | `/root/autodl-tmp/datasets/AeBAD/AeBAD_S` |

真正需要的文件：

- capture 的 `inputs.json`、`status.json`、`capture_verification.csv` 和全部 297 张 `raw_cache/test_*.npz`。十二案例和小型核查包不能代替完整集合。
- 支持目录的 `01.npz` 到 `08.npz`，内容 SHA、protocol、relative_path、支持顺序、grid 必须与 capture 的 frozen/input/raw metadata 及正式 manifest 一致。不是只检查文件名。
- 正式目录的 `sample_scores.csv`、`support_manifest.csv`、`complete.json`、`memory_manifest.json`、`source_mad_state.npz`。允许 `--mad` 指向字节一致的迁移副本。
- 212 张异常的原 GT；原固定十二案例的 RGB。正常的其余 RGB、训练图和八张支持 RGB不参与本工具计算。数据目录只需相关 `test`/`ground_truth`，不要求 `train/good`，不要求全部正常 query RGB。数据根可以是 AeBAD_S 或其 AeBAD/外层父目录，本工具读目录，不解压或接受 ZIP 数据。
- 服务器现有原地图、MAD、full-protocol 评价和 GT patch pooling 代码及其依赖。

不要求 `official_memory.pt`、DINOv3 checkpoint、官方模型目录、query 特征或 Source-extra 支持。本地两个已下载结果包不含完整 raw，缺这些 raw 时应迁移完整 capture，而不是重新启动模型。只读预检不会补建任何输入。

复用 `aebad_s_union_support_loo.inputs.read_capture`、`load_raw`/原 raw 校验、`formal_baselines`、原 `comparison`、`MapBackend` 和通用 `branch_maps`，再使用原 `_method_map`。旧 LOO 的十分支排名/空间函数没有直接套用；本模块单独编排三个分支，未 monkeypatch `BRANCHES` 或修改历史模块。

支持读取复用原 `replay12.check_support_header`，读取 `SupportFeatures` 所使用的原 NPZ 格式 `[grid_h*grid_w,1280]`。原 support 模块顶层导入 Torch/提取代码，所以本工具只实现只读 NPZ 读取，绝不调用可能进入缺失补提分支的 `extract_or_load_support`。每次只保留八张支持的一层，端点按 1024 行分块转换 float64；不构造全域 query/support/feature 巨型数组。

## 三个分支及运算顺序

每层对八张支持各自已记录的 Top-1 距离，按距离升序、manifest support_index 升序稳定排序，取两张不同支持图的候选。对应特征的物理 flat row 来自该层 `target_index`，不是 query 的 compact 行。三个分支共享同一组端点，逐层独立选取，不以标签或 GT 选端点。

设缓存距离 `a<=b`，`c=||t2-t1||/1280`，Source 距离 `dS`：

```text
f(alpha) = (1-alpha)*a² + alpha*b² - alpha*(1-alpha)*c²
dU       = min(dS, min_j dT_j)
dMid     = min(dU, sqrt(f(0.5)))
alpha*   = clip((a²+c²-b²)/(2*c²), 0, 1)
dSeg     = min(dU, sqrt(f(alpha*)))
```

`dU`、`a`、`b` 已经是欧氏距离/1280，不再缩放。只有端点原特征直接差分得到的 `c²` 除以 `1280²`；不归一化原特征。中点及线段都保留原 union 和全部原端点，不替换或移除它们。

距离、平方、alpha 和公式运算采用 float64。通过检查后，新距离转 float32，按：

`compact 逐层距离 → positions scatter → 原 Source MAD → 四层等权均值 → 原双线性插值 → 全图最大值`。

不在最终地图之间建立参照，不重写插值。所有新增 raw 完成并写入 `predictions.json` 后才解码 GT。评价顺序不会让 GT 参与几何计算。

## 固定数值规则

FP64 不能恢复历史 FP32 检索丢失的精度。容差基于已有距离复核 `atol=1e-7, rtol=1e-6`，分数/汇总 `abs_error<=1e-5`，不提供改变容差的参数。

令 `e_a=1e-7+1e-6*a`、`e_b=1e-7+1e-6*b`，`eps64` 为 IEEE float64 epsilon，`gamma=(1280+16)*eps64/(1-(1280+16)*eps64)`。端点来自 FP32 特征，先转 FP64 再相减、平方求和；误差界保守覆盖该长度运算。

三边检查：

```text
abs(a-b) <= c + triangle_tolerance
c <= a+b + triangle_tolerance
triangle_tolerance = e_a+e_b+gamma*c+16*eps64*(a+b+c)
```

对固定 alpha=t 的小负平方距离，允许截零的界为：

```text
bound(t) = (1-t)*(2*a*e_a+e_a²) + t*(2*b*e_b+e_b²)
         + t*(1-t)*gamma*c²
         + 16*eps64*(abs((1-t)*a²)+abs(t*b²)+abs(t*(1-t)*c²))
```

只有 `-bound<=f(t)<0` 才截零；逐 patch 保存平方距离修正量，计数以“中点/线段两次求值的修正事件”为单位。`f(t)<-bound`、严重三边不相容或非有限值立即失败，保留现场，不能无条件 `maximum(f,0)`。

完全相同端点，即直接差分 `c²==0`，中点和线段都退回 union，alpha 留为 0 并单独计退化数。近但非相同端点不新增长度门控；numerator 按代数等价的 `(a-b)*(a+b)+c²` 计算，避免 `a==b` 时丢掉很小的 c²。用 numerator 与 0、2*c² 的比较先判断边界，仅对内部投影作除法，避免小 c² 导致不必要的溢出。单支持回退只存在于数学单元测试；正式预检始终要求八张。

检查 `f(0)=a²`、`f(1)=b²`、`dSeg<=dMid<=dU`（已有距离容差），以及重新保存/读入后的有限性、dtype、原 positions 和输入哈希。新增 raw 完成后和评价结束时再次核对保护文件；修改过的缓存不会被重新认可。原 K8 逐图分数、全域 AUROC/AP重新对照正式 CSV；额外复核 Source 分数，但不把 Source作为第四方法分支。

## 指标与空间口径

所有分支都是同一组 85 正常、212 异常；原八张支持始终排除。输出 Image AUROC、Image AP、Pixel AUROC、Pixel AP、AUPRO，AP 使用 `average_precision_score`。

像素指标直接调用 `aebad_s_full_p0.runner._load_mask` 和 `_evaluate`，使用原完整视野、原地图尺寸、nearest GT resize、ADEval CUDA 200 阈值及既有 FPR 0.3 口径。没有采用 MMR 的 256 resize/224 crop。**完整像素评价需要 CUDA 和可用 adeval；不跑 backbone 不等于所有指标都无需 GPU。** 无后端时保存可用的图像/空间诊断，像素项留空，状态 `partial_evidence`，退出码 3，不切换替代 AUPRO。

实例复用 `components`/`instance_regions`：8 连通域，原图 Chebyshev 半径 16，环带排除全部 GT，原输出网格映射。保留原面积、输出面积、patch coverage、实例/环带 max、margin 及变化。空环带、映射后消失不填零；标注碎片不自动称为真实小缺陷。`peak_positions.csv` 同时记录每分支自己的峰值和各分支在同一峰值位置的读数。

相对原 K8 的机制量：

```text
Delta_l(q) = (dU_l(q)-dNew_l(q))/(Source_MAD_scale_l+1e-8)
R_N = 正常图全部 patch 的每图均值，再图像等权
R_A = 异常图 GT 覆盖 patch 的每图均值，再图像等权
G = R_N-R_A
```

GT patch 覆盖用原 adaptive-max pooling；不是异常图的全部 patch。逐层及四层等权汇总都明确 `reference=K8_UNION`，不能与历史相对 Source 的量混用。缺 GT/空 GT/缺机制区域会阻止完成。

排名另记录正常 p90/p95/max、原 K8 固定前十正常的分数变化及正常—异常排序对改善/变差/不变，并列错误权重仍为 0.5。不重新挑前十或阈值，不把高分案例叫作固定阈值误报。

## 输出及完成条件

输出必须为新目录，或仅含本工具 `preflight.json` 的目录。重跑用另一个新目录，无自动续跑框架；历史输入、结果、报告及哈希记录不修改。

| 文件 | 含义 |
| --- | --- |
| `preflight.json`、`inputs.json`、`status.json` | 路径检查、本轮与历史身份、阶段/完成状态 |
| `predictions.json` | 全部新距离冻结后的逐文件历史哈希清单 |
| `raw_cache/test_*/L7.npz` 等四层文件 | 两新增分支 float32 compact 距离；FP64 alpha/端点长度/平方修正；端点支持编号/物理 flat index、positions 与 capture raw 哈希 |
| `numerical_verification.csv` | 几何、原 K8距离、Source/K8分数与正式汇总复核 |
| `metrics.csv`、`per_image_scores.csv`、`ranking_summary.csv` | 三个分支的五项指标及图像/排序统计 |
| `defect_instances.csv`、`peak_positions.csv` | 实例/环带及同位置与重新取峰值的响应 |
| `mechanism_per_image.csv`、`mechanism_summary.csv` | 相对 K8 的逐层 R_N、R_A、G与四层汇总 |
| `interpolation_summary.csv` | 每图/每层的端点有序编号对频率、alpha分布、退化/修正数、新距离严格降低比例，及跨图汇总 |
| `figures/` | 原固定十二案例：RGB/GT、三个分支和两个差分，共同色标，标明每分支峰值 |
| `timing.csv`、`report.md` | 缓存读取、支持读取、几何、地图、指标、GT/绘图等耗时与解释边界 |

新增 raw 每图四个小 NPZ，而不是复制旧 capture 或支持特征。重做地图/评价所需的两分支距离及最小追溯信息已保存；本入口不提供 resume。失败期间尚未产生的表不会填造。原 capture、MAD、GT仍需保留。

alpha 每图输出 p10/p50/p90；全域汇总输出 pooled min/max/mean及固定 `[0,0.1,...,1]` 的直方图（最后一格含 1）。直方图仅用于统计，不是 alpha 搜索网格。端点有序对的频率保留“距离更近端点/第二端点”的顺序；汇总同时区分 patch 数加权和图像等权。距离降低比例按存储 FP32严格 `<dU` 定义，避免仅以公式 FP64 的不可见变化当成实际降低。正常/GT分组只在预测冻结后的评价阶段生成。

一次仅保留一层八张支持，几何临时端点块上限 1024×1280。full-protocol 评价一次收集一个分支的全域地图及 GT，会比纯几何需要更多 RAM/评价显存；本轮没有测得真实峰值。可用时记录进程历史 peak RSS 和评价 CUDA allocated/reserved 峰值；不可用为 null。地图会为统计和逐分支评价重复生成，仍无模型/检索。所有耗时注明 `offline_cached_analysis_only`，不能写成端到端检测速度。

完成必须同时满足：297张预测/分析，所有几何与正式基线复核通过，三分支 full-protocol 五项指标完整，所需 GT/空间与机制范围完整，输入/新缓存未修改。完成是 `analysis_complete`/退出码0；preflight阻止为2；数值/依赖/空间证据不完整为3；未处理异常为1并保留现场。

## 用户后续手动运行命令

本轮不执行以下服务器命令，不上传，不自动关机。环境不安装或自动修改。

```bash
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export PROJECT_ROOT=/root/autodl-tmp/DINO-NVS
export PYTHON=/root/miniconda3/bin/python
export BASE="$PROJECT_ROOT/DINOv3/outputs/mad_equal"
export CAPTURE_DIR="$BASE/aebad_s_union_support_loo_B_20261007_230639"
export RESULTS_DIR="$BASE/aebad_s_target_fewshot_seed42_v1"
export SUPPORT_CACHE_DIR="$BASE/aebad_s_union_background_replay12_seed42_v1/support_features/target_matched/background"
export DATA_DIR=/root/autodl-tmp/datasets/AeBAD/AeBAD_S
export OUTPUT_DIR="$BASE/aebad_s_target_local_segment_$(date +%Y%m%d_%H%M%S)"
cd "$PROJECT_ROOT" || exit 1

# 完全只读预检：不创建目录/报告，不导入Torch。
"$PYTHON" -B -m DINOv3.MADEqual.aebad_s_target_local_segment.runner preflight \
  --capture-dir "$CAPTURE_DIR" --results-dir "$RESULTS_DIR" \
  --support-cache-dir "$SUPPORT_CACHE_DIR" --data "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" --no-write

# 上面退出码0才执行。无需 backbone，但完整AUPRO需要CUDA。
"$PYTHON" -u -m DINOv3.MADEqual.aebad_s_target_local_segment.runner analyze \
  --capture-dir "$CAPTURE_DIR" --results-dir "$RESULTS_DIR" \
  --support-cache-dir "$SUPPORT_CACHE_DIR" --data "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR"

cat "$OUTPUT_DIR/status.json"
```

没有新增启动脚本。用户需要同步时只同步整个新目录 `DINOv3/MADEqual/aebad_s_target_local_segment/` 和新增测试 `DINOv3/tests/test_mad_equal_aebad_s_target_local_segment.py`，不需要覆盖旧 scoring/LOO/full runner。

## 本地验证与未执行部分

本地解释器：`D:/anaconda/miniconda/envs/chinamoblie/python.exe`，Python 3.11.12，NumPy 1.26.3；无可用于本轮生产路径的 PyTorch 环境。依赖元数据可发现不等于运行可用，未导入真实 Torch。测试命令：

```powershell
& 'D:\anaconda\miniconda\envs\chinamoblie\python.exe' -B -m unittest DINOv3.tests.test_mad_equal_aebad_s_target_local_segment -v
```

2026-10-08 最终版专项 **22 项全部通过**，本地耗时118.529秒。覆盖直接向量中点/线段投影、内部/外部投影、相同/极近端点、保留union、跨支持候选与并列、单支持回退、chunk不影响结果、/1280不重复缩放、positions、数值修正与不相容拒绝、输入哈希/支持顺序/历史输出保护、默认只读入口及完整范围编排。

合成全域流程确认1188个逐层新raw完成后才调用GT读取；只输出三个分支和891行逐图记录。后端分数整体加1时594项正式分数复核失败，退出3/partial_evidence；缺CUDA像素后端时AUPRO留空，同样不能完成。输出仅含其他工具preflight.json时也拒绝覆盖。

测试的合成向量验证三边公式与直接向量投影；原 `_method_map` 和 `_load_mask`/`_evaluate` 的轻量 AST 调用只验证编排，插值使用明确标记的测试依赖，ADEval以测试替身表示，不能当作真实地图/像素指标验证。合成 fixture 使用297个微型raw及原集合成员/manifest，既不是模型推理也不是真实297张全域实验。

本地真实路径的 `preflight --no-write` 返回2/blocked：完整B capture目录缺失、八个原支持NPZ缺失、数据根缺失；正式CSV/MAD等本地归档可读。确认没有导入Torch、没有创建探测输出目录，也没有解压两个旧下载包。用内存AST完成语法检查，不写pyc；核对原LOO四个核心文件及共用scoring.py与历史capture代码SHA一致，没有修改旧模块。

本地没有完整 B raw及原八张支持特征；真实缓存三边相容性、全域 K8复现、中点/线段几何结果、原Torch双线性地图、真实CUDA ADEval、实际资源峰值、十二真实案例图全部未执行。没有用fixture结果写性能结论，没有修改历史实验报告。
