# AeBAD-S/background SPARC 同宿主对照

本工具是外部 SPARC 方法向当前 DINOv3/Exact/MAD 宿主的迁移，不是本章新方法，不声称复现论文表格或达到 SOTA。background 已用于多轮开发，属于开发域，不能作为独立最终验证。本轮只实现代码、必要的轻量验证和运行说明；没有连接服务器，没有真实模型提取、重建或正式评价。

## 方法与宿主适配

核对原文 [3.2](https://arxiv.org/html/2608.18585v1#S3.SS2)、[3.3.1](https://arxiv.org/html/2608.18585v1#S3.SS3.SSS1) 和 [B.1/B.2](https://arxiv.org/html/2608.18585v1#A2)：按原生网格逐位置中心化支持特征，薄 SVD 保留经验非零秩，最多 K−1；保存均值和低秩基；先校正训练特征再按宿主原程序重建参照，对查询使用相同校正。

| 项目 | 实现 |
|---|---|
| 输入 | 每层、每个完整拼接位置的同一组八张 Target-normal，8×1280 |
| 均值 | float64 `mu_p=mean(X_p)` |
| 基 | `svd(X_p-mu_p, full_matrices=False)` 的右奇异向量；列正交，最多7列 |
| 固定数值秩规则 | `s > max(8,1280)*eps(float64)*s_max`，最多7；零秩为恒等变换 |
| 校正 | `C_p(f)=f-V_p[V_p.T(f-mu_p)]`，保留均值；不形成1280×1280矩阵 |
| 空间 | 完整原生拼接网格；compact 查询通过 `positions` 定位；Source 完整 embedding 使用 row-major 物理位置 |
| 查询检索 | 校正后仍对完整 Source bank 作原 Exact 1-NN，不限制同位置匹配 |

数值秩规则、float64 投影状态/运算及回写宿主 float32 是本工程明确的数值实现选择，不作为论文指定的软件细节。位置块为64，只控制内存，不改变投影定义。不做全局 PCA、均值平移、白化、query-conditioned 校正、参数搜索、门控、额外支持或 SPARC+union。不同位置有不同投影，不能假定全部最近邻距离只下降。

本文实现的宿主是冻结 DINOv3 ViT-H+/16、L7/15/23/31、resize_factor=0.625、640窗口/128 overlap、patch16、完整视野、evaluation_downscale=4、Exact Euclidean/1280一次、原 MAD、四层等权、原插值、最终地图最大值。原论文还有其他宿主和多 seed 实验；这里固定 Seed42、原K8、单一开发域及当前 MAD 评分，不能直接比较论文表格数值。

## 三个正式分支

| 分支 | 正常参照 | MAD |
|---|---|---|
| SOURCE | 原 Source | 原 `source_mad_state.npz` |
| UNION_K8 | `min(d_Source,min8 d_Target)` | 原 Source MAD |
| SPARC_SOURCE_K8 | 逐位置校正后重新采样的独立 Source bank，不追加 Target memory | 同一批66张 Source 校准图，经校正/新检索重新拟合 |

前两个分支读取可信完整 B capture，重新对照正式 CSV 逐图分数及 AUROC/AP 汇总；不重复 backbone。距离容差仍为 `atol=1e-7,rtol=1e-6`，分数/汇总为 `abs_error<=1e-5`，没有放宽参数。改变评分后端而使原分数变化，不能标记完成。

## Source 重建与复用入口

`aebad_s_full_p0.dataset._images` 的原有相对路径排序生成完整521列表，`split_source_normal(...,8)` 为455 memory-fit/66 calibration。调用 builder 的仍是完整521列表，不能先传455再划分。新路径用惰性 RGB Sequence，builder 内部只读取其选择的455张，不把所有 Source RGB/特征装入 RAM。

共享 builder `external_baselines/superadd_external/memory_variants.py::build_official_memory_spooled` 保留默认关闭的 `feature_transform(source_index,layer,full_embedding)`。关闭时保留原迭代；开启时在原亮度预处理、原 patch_exec 之后、flatten/spool/采样之前校正。传入索引是原521列表索引。新增显式 `storage_backend='mixed',ram_layers=(7,15)`，复用原RAM矩阵和磁盘spool；共享builder默认仍为全磁盘，其他调用不自动改变。仍使用原 indexed Official 距离采样、100 iterations、100000/层和原随机过程；支持处理之后恢复构造宿主时的 Source NumPy RNG 状态。校正可以改变采样结果，不强求选中同样的行；固定同一校正时，混合存储不改变候选值和采样规则。

保留实际采样 candidate_index → Source image_index → 物理 patch_position，以及每图候选区间/网格/路径。Source calibration 的 index%8==0 永不进入候选。原 bank 没有可靠逐行空间来源，本工具不读取它来做位置校正，也不修改它。

查询和校准直接复用 `_extract_processed_batch` 的完整确定性宿主路径，用同一个 `exact_corrected_layer` 校正后调用 `_exact_top1_chunked`。Source候选、支持、查询的网格、processed shape 和拼接轴必须一致；不一致失败，不插值投影基。

校准复用原 `fit_calibrators`，只汇集66张校正 Source 查询对新 bank 的新距离。新MAD独立保存。各分支有相同 Source 校准数据和拟合规则，SPARC 数值统计随空间改变。不能使用旧 calibration raw、Target 自匹配或旧 MAD 代替。

## 输入资产与默认服务器参数

以下是历史路径，本轮未远程检查。

| 参数 | 位置/要求 |
|---|---|
| `--project-root` | `/root/autodl-tmp/DINO-NVS` |
| `--results-dir` | `.../DINOv3/outputs/mad_equal/aebad_s_target_fewshot_seed42_v1` |
| `--capture-dir` | `.../DINOv3/outputs/mad_equal/aebad_s_union_support_loo_B_20261007_230639`，原inputs/status/verification及全部297 raw |
| `--support-cache-dir` | `.../DINOv3/outputs/mad_equal/aebad_s_union_background_replay12_seed42_v1/support_features/target_matched/background`，01–08.npz |
| `--data` | `/root/autodl-tmp/datasets/AeBAD/AeBAD_S`，521 Source、固定8支持、297 query、212 GT |
| `--official-root` | `/root/autodl-tmp/DINO-NVS/external_baselines/SuperADD` |
| `--dinov3-root` | `/root/autodl-tmp/dinov3` |
| `--checkpoint` | `/root/autodl-tmp/DINO-NVS/external_baselines/SuperADD/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth`，固定SHA |
| `--output-dir` | 新的独立目录，四个阶段持续使用同一个路径 |
| `--candidate-storage` | preflight/prepare默认 `mixed`：L7/L15放RAM，L23/L31放磁盘；`disk`保留原全磁盘模式。后续阶段读取冻结记录 |

正式目录需要原 `sample_scores.csv`、`support_manifest.csv`、`complete.json`、`memory_manifest.json`、`source_mad_state.npz`，字节身份与 B capture 一致。原八张支持/顺序固定排除，始终297张，85正常/212异常。已有支持文件必须匹配原 feature SHA、header、grid，修改文件失败；缺失支持只有显式 prepare 可补提到**新输出**的 `supplemented_support`，保存原图/权重/冻结输入/当前文件SHA provenance，不写回旧路径。补提NPZ字节不被冒充为历史NPZ。

Source没有现成可信完整建库特征缓存。本工具选择原预处理重新提取455张；不会把未知缓存、原压缩 bank 或四案例 selected-query 特征当成完整 Source 特征。历史正式 manifest 保存了 Source计数/划分规则，未提供全部521张历史逐图哈希；本轮按原排序冻结当前 Source 路径与字节身份，不声称已经证明历史每个 Source 文件字节未变化。

复用清单：完整B raw、正式五文件、原八支持NPZ、原数据/宿主/权重。补提清单：仅缺失的固定支持，及455张 Source建库图、66张 Source校准图、297张query。这里521张Source两部分互斥，各自走一次对应路径。重建清单：四层独立SPARC Source bank和独立MAD。原 official_memory.pt 不需要加载，因为基线已缓存且新bank必须由Source图像重建。

## 阶段与资源

- `preflight` 默认：只读身份、数据/Source划分、已验证缓存、支持header、RGB尺寸/字节哈希及GT字节哈希。无模型/Torch导入，无隐式提取。脚本默认 `--no-write`。
- `prepare` 显式：支持复用/缺失补提 → 逐位置SVD → Source采样前校正 → 原规则建库 → 校正66张校准查询 → 新MAD。GT不解码。
- `infer` 显式：校正297张完整查询 → 全库Exact → raw/geometry/hash → `predictions_frozen` barrier。GT不解码。
- `evaluate` 显式：先核对全部冻结预测、准备产物和输入哈希，再读GT并评价/诊断。无 backbone 提取。

--help/preflight 可无 PyTorch；其余阶段需要原Torch宿主依赖，prepare/infer需要CUDA，evaluate的原ADEval也需要CUDA。不安装依赖，不提供CPU AUPRO替代路径。源码/资产变化、旧缓存变更、半写输出均拒绝。prepare要求新目录或只含本工具preflight.json的目录；infer/evaluate要求此前阶段完整，拒绝重复或半成品。失败留存本工具已生成文件及失败记录，不自动续跑，也不重置已完成历史状态。

在实际源/支持均为118×118时，原455图四层候选合计仍为 **120.839 GiB**，FP64投影约 **4.249 GiB**，新bank约 **1.907 GiB**。默认混合模式候选RAM/磁盘各 **60.419 GiB**；无支持补提时prepare要求可用磁盘约 **70.805 GiB**（含新raw估算和4GiB余量），可用RAM门槛约 **80.669 GiB**（RAM候选+完整投影估算+16GiB工作余量）。全磁盘模式仍要求约 **131.225 GiB** 可用磁盘。这些是估算，真实大小按Source header重新计算；不包含旧输入文件和模型文件。空间不足停止，不能自动改100K、精度或视野，也不自动切换存储计划。

候选按冻结的两层RAM/两层磁盘计划保存，全部保持float32、C顺序；不改变逐图提取、亮度随机过程、逐层校正或采样顺序。每次支持拟合只保留八张支持的一层；投影为磁盘memmap，按64位置块运算；RGB逐图。全磁盘模式最低可用RAM余量仍为5GiB。混合模式在预检、prepare开始及建库前检查RAM与磁盘；RAM预算未知时拒绝。门槛不是峰值保证；Torch/操作系统/采样仍有开销。评价每次只收集一个分支全域地图。真实峰值尚未测量，OOM失败保留现场，不调整实验协议。

## 输出与诊断

准备目录：`inputs.json`、`source_manifest.csv`、`support_provenance.json`、`projection/L*/mu.npy,basis.npy,rank.npy`、`projection_state.json`、`projection_verification.csv`、`sparc_source_memory.pt`、`bank_provenance/L*.npz`、`candidate_sources.json`、`source_calibration/`、`sparc_source_mad_state.npz`、`calibration_stats.csv`、`prepare_status.json`。

预测目录：`sparc_raw/test_*.npz`，四层compact距离/nearest bank index/positions/grid/processed_shape及完整几何身份；`infer_status.json` 是逐文件历史SHA记录；不保存query×memory矩阵或全部query特征。`timing_per_image.csv`分别记录每张query backbone、校正、检索以及含读图/写raw的耗时。

评价目录：`evaluation/metrics.csv`、`per_image_scores.csv`、`ranking_summary.csv`、`defect_instances.csv`、`distance_audit_per_image.csv`、`distance_audit_summary.csv`、`numerical_verification.csv`、`figures/`、`status.json`、`report.md`。正式像素路径是原 `_load_mask`/`_evaluate`，完整视野、nearest GT resize、AP=`average_precision_score`、原ADEval CUDA200阈值/FPR≤0.3，没有 MMR crop。

排名使用正常—异常对及并列错误0.5，报告相对UNION改善/变差/不变、正常p90/p95/max、原固定前十的变化，不选择部署阈值，不称固定阈值误报/漏检。实例复用原8连通、原图Chebyshev16环带、排除全部GT、原输出映射；按原面积排列，保留碎片、空环带、映射后消失，无新增面积阈值。十二案例沿用原名单，包含IMG_8863/IMG_9385，不按新分数挑图。

正式响应使用各自MAD。距离审计单独使用：

`Delta_l=(d_UNION_l-d_SPARC_l)/old_source_MAD_scale_l`。

正常图用全部patch；异常图仅用原adaptive-max pooling的GT覆盖patch。每图/每层先均值，再图像等权，四层等权，输出有符号R_N/R_A/G。SOURCE也可作为相对UNION参照记录。不能把不同MAD的响应之差称为原始距离下降；G>0不保证每个缺陷保持。margin是各分支正式地图的实例max−环带max，其尺度不同，须结合正式像素指标与逐实例变化评估。

成本：`cost_prepare/infer/evaluate.json`、`timing_prepare/infer/evaluate.csv`，分别为一次性适配、query成本、离线评价；记录阶段进程峰值RAM/CUDA allocated/reserved、新bank/投影/raw/总磁盘量，prepare临时候选峰值来自builder实际metadata。Source backbone计时包住原patch_exec；建库剩余项包括预处理、spool、采样、拷贝等，不能声称仅是采样时间。统计整阶段峰值，不冒充某分支独立显存；离线evaluate耗时不等于端到端检测速度。

完成：prepare=`prepare_complete`；infer=`predictions_frozen`；evaluate仅在297张、全部正式基线/数值复核、三个完整五指标及区域证据/输入完整时=`analysis_complete`。正常退出0；blocked预检2；评价不完整3；未处理异常1。缺真实GT/像素后端不标记完成。

## 后续手动运行命令

本轮不执行下列服务器命令，不连接/上传，不安装环境，不自动关机。默认脚本已按本任务明确要求添加。

```bash
export PROJECT_ROOT=/root/autodl-tmp/DINO-NVS
export PYTHON=/root/miniconda3/bin/python
export OUTPUT_DIR="$PROJECT_ROOT/DINOv3/outputs/mad_equal/aebad_s_sparc_background_$(date +%Y%m%d_%H%M%S)"
cd "$PROJECT_ROOT" || exit 1

# 第一步，默认仅只读preflight，必须返回0。
bash DINOv3/MADEqual/aebad_s_sparc_p0/scripts/run_aebad_s_sparc_background.sh

# 以下为后续实验安排时才手动执行的三个明确阶段，保持同一OUTPUT_DIR。
bash DINOv3/MADEqual/aebad_s_sparc_p0/scripts/run_aebad_s_sparc_background.sh prepare &&
bash DINOv3/MADEqual/aebad_s_sparc_p0/scripts/run_aebad_s_sparc_background.sh infer &&
bash DINOv3/MADEqual/aebad_s_sparc_p0/scripts/run_aebad_s_sparc_background.sh evaluate
```

路径迁移通过 `RESULTS_DIR/CAPTURE_DIR/SUPPORT_CACHE_DIR/DATA_DIR/OFFICIAL_ROOT/DINOV3_ROOT/CHECKPOINT` 环境变量或同名CLI参数，所有阶段一致。直接模块入口：`$PYTHON -B -m DINOv3.MADEqual.aebad_s_sparc_p0.runner preflight --no-write ...`。重连后需恢复同一OUTPUT_DIR；换路径不会自动认可为原身份。

后续同步只需整个新模块、`external_baselines/superadd_external/memory_variants.py`、新增专项测试；不需要覆盖旧few-shot runner/scoring或旧结果。同步/执行由用户安排，本轮不自行操作。

## 本地验证及未执行部分

解释器 `D:/anaconda/miniconda/envs/chinamoblie/python.exe`，Python3.11.12/NumPy1.26.3。本轮不导入真实Torch，未改变环境。

```powershell
& 'D:\anaconda\miniconda\envs\chinamoblie\python.exe' -B -m unittest DINOv3.tests.test_mad_equal_aebad_s_sparc_p0 -v
```

2026-10-08最终版20项专项全部通过，42.250秒。覆盖均值保留/零秩、正交/幂等、八支持对应位置收敛均值、符号等价、独立位置/重排、网格拒绝、磁盘状态、原builder默认/恒等回调特征及采样来源/RNG等价、原indexed sampler与上游规则等价、双侧校正、Source划分/来源、一次/1280、恒等地图链路、固定旧尺度有符号距离审计、默认无Torch预检、哈希/半成品拒绝、三分支评价编排及整体评分偏移拒绝完成。

2026-10-09修复宿主初始化：将CUDA设备规范化为明确索引，在种子设置、显存统计重置和共享宿主构造前调用 `torch.cuda.set_device`；构造后显式执行DINO的 `eval()` 和 `requires_grad_(False)`，再检查冻结协议。`--device cuda:1` 会选择设备1；不带索引的 `cuda` 使用当前设备。prepare/infer共用此入口。

本次25项专项全部通过，48.896秒；模块7个Python文件和专项测试文件的内存AST语法检查通过，未写pyc。新增5项初始化测试未加载真实Torch、模型或CUDA。

同日混合存储修订：28项专项全部通过，41.993秒；模块7个Python文件、专项测试及共享builder共9文件的内存AST检查通过。新增合成fixture直接执行实际builder，比较全磁盘、两层RAM/两层磁盘、全RAM三种后端：逐层候选float32数组、最终bank、选中候选索引及NumPy随机状态完全一致；同时检查实际memmap分层、临时磁盘统计/清理、非法分层与容量拒绝、RAM不足或未知时在宿主加载前拒绝。该fixture的采样/张量转移依然使用明确替身，不等于真实CUDA采样或正式指标验证。

builder测试执行实际函数AST和真实NumPy spool，Torch转移、特征执行、NN距离均是明确的测试替身；indexed采样测试使用相同NumPy距离替身比较实际上游函数与实际indexed函数。地图测试执行原MAD/评分AST，identity-size插值为测试依赖；297 tiny fixture的AUPRO为替身。新增宿主初始化测试直接调用实际 `load_host`，仅Torch和宿主依赖使用隔离替身，检查冻结顺序、非默认设备、无索引设备、不可用CUDA拒绝及冻结失败拒绝。它们不等于真实模型、Torch cdist、原双线性插值或CUDA ADEval验证。

真实本地资产预检为blocked：正式CSV/manifest/MAD归档可读；完整B capture、八张原支持NPZ、521 Source及297 query/GT数据、DINOv3官方代码/权重路径未齐全。原下载核查包/局部参照review包不能代替完整B raw。不自动提取、解包、重建或创建预检输出。

只读预检Python入口实际返回2，Torch未导入，默认输出目录未创建。Python语法以内存AST检查，不写pyc。Windows没有可用的Git Bash，启动脚本未实际进行bash执行/语法检查；后续默认preflight也需由用户在服务器确认。

真实CUDA支持/Source/query提取、Source校正重建/采样、校准查询与MAD、完整精度对齐、峰值资源、三分支正式指标和十二真实图均未执行。历史报告、正式memory/MAD/支持和结果包未修改。结果必须允许优于、持平或弱于UNION；本轮没有性能结论。
