# AeBAD-S/background 固定 K8：FastRef Target512 / Source anchor

这是 FastRef 论文公式在 DINOv3、原Source参照和固定Target K8上的受控欧氏适配，不是官方实现复现，也不是Source32旧实验续跑。实现及量纲审计见 [implementation_audit.md](implementation_audit.md)。本轮只实现工具、NumPy/替身测试和只读预检，没有运行真实prepare/smoke/infer/evaluate，没有连接服务器、安装Torch或改变环境。

**v3已解决采样控制流程的兼容阻断，真实实验仍未验证。**采用有界反密度精确预算适配：保留原raw欧氏局部kNN统计和反密度偏好；候选整层随机均衡分组、固定原起始半径、全局指数race无放回选择512个不同索引。它改变原型选择行为，不声称与原Source采样完全等价。历史v1/v2产物拒绝加载；Source/MAD、FastRef数学目标、宿主及评价不变。默认仍只读preflight，真实prepare/smoke/infer/evaluate由用户后续明确执行。

## 固定对照

background Seed42，原八张支持始终排除，评价297张（85正常/212异常）；冻结DINOv3 ViT-H+/16、L7/15/23/31、1280维、原完整视野/640窗口/128 overlap、brightness=1、resize=.625、batch=1、原提取精度、原Source bank和Source MAD。原网格118×118，评价地图756×756。无增强、SPARC校正、SEGMENT、门控、额外pooling/平滑或新评分。

| 分支 | 每层进入原MAD的距离 |
|---|---|
| SOURCE | 原dS，正式五指标沿用历史 |
| UNION_K8 | min(dS, NN(Q,T))，正式五指标沿用历史 |
| UNION_TARGET512 | min(dS, NN(Q,P)) |
| FASTREF_TARGET512_SOURCE | min(dS, NN(Q,R(Q,P))) |

两个新增分支读取同一组已准备的P、相同行顺序及prototype身份，来自原八份支持的整层候选池（111392个原始1280维patch），**全池共512，不是每支持512**。支持顺序/身份不重选，新分支不另合入完整T。

采样配置固定：N=512、最多100个随机均衡且互不相交的组，每组 `k=min(100,组大小)`（包括自身），raw欧氏、无归一化或投影。半径固定为该组kNN距离均值×.1（原规则的起始尺度），权重 `w=1/(邻居距离严格小于半径的个数+1)`。局部RandomState42创建一次，L7/15/23/31连续。整层每个候选只分组一次、生成一个指数时钟 `-log(1-u)/w`，按时钟及原始索引排序选择恰好512个不同索引，再按物理索引升序保存原始特征行；没有每轮名额、余数补足或无限循环。

这是**保留Source局部反密度偏好的受控算法适配**。改变了原随机重叠子集、随预算增长半径以及固定uniform保留/随机补足的具体规则，不能称作完全沿用原算法。指数race是明确的加权无放回选择，不是随意截断候选。全零/重复特征可完成索引选择，但矩阵秩不足仍失败；不重复索引凑数、重试种子、改N、加ridge或用伪逆。若所有权重相同，实际为均匀无放回选择，诊断明确记录，不宣称密度选择有额外效果。

**为什么保留512：**新流程可对M≥512严格生成512个不同索引，不再受旧每轮5个限制；512<1280满足行Gram可逆的必要维度条件，保持已预设内存/OT预算及对照名称。仍需真实P满行秩、cond≤1e12、solve残差≤1e-10；没有按测试标签/指标选择N，不增加多预算或多采样器。完整K8 union使用全部T，不等于压缩Target512。

v3的TOOL、sampling revision、输入及原型metadata/索引/字节身份、三份采样相关代码SHA和默认输出目录均区分。旧v1投影贪心原型及v2原规则产物不可改名复用；混合配置拒绝继续，必须使用新输出重新prepare，不能覆盖旧结果。精确退出原因、选择分布变化及适配依据见 [implementation_audit.md](implementation_audit.md)。

细化采用审计中给出的**平方欧氏求和目标**、概率OT总质量1、行1/m/列1/512、lambda=.3、外层2；epsilon=.1×正常P非对角平方距离中位数，float64求解及log Sinkhorn。CPU并行最多四层，每层OT仍覆盖完整query；无m倍传输重缩放、无跨chunk独立OT。最终R有m行，每个query到全部R做原Exact欧氏NN。内部Q/P统一除1280改变单位，输出R恢复raw单位；原Exact返回欧氏再除1280**一次**。不混入余弦距离或配对残差。

Source/Target均不逐行L2归一化；关闭细化直接使用P，不以lambda=0代替。refinement.py没有跨查询可变状态；永久正常库仅含所选P，不含Q或R。全部分支复用原MapBackend：原Source patch MAD、四层等权、原插值、全图max；保留负MAD响应。

## 必需输入

相对服务器工程根 `/root/autodl-tmp/DINO-NVS`：

- `DINOv3/outputs/mad_equal/aebad_s_target_fewshot_seed42_v1/`：`official_memory.pt`、`source_mad_state.npz`、`memory_manifest.json`、`support_manifest.csv`、`sample_scores.csv`、`domain_metrics.csv`、`complete.json`。
- `DINOv3/outputs/mad_equal/aebad_s_union_support_loo_B_20261007_230639/`：完整297份 `raw_cache/test_*.npz`、`inputs.json`、`status.json`、`capture_verification.csv`。固定capture身份 `4fc53367a32843d12d60c9c7a3a6932f2af694af7dbfab63cf115de57e71a0c1` 和本地已下载记录的三个SHA，拒绝把改过的缓存新认可。
- `DINOv3/outputs/mad_equal/aebad_s_union_background_replay12_seed42_v1/support_features/target_matched/background/01.npz`—`08.npz`，严格原身份/顺序/历史SHA及float32 `[13924,1280]` 四层格式。
- `/root/autodl-tmp/datasets/AeBAD/AeBAD_S`：297查询、八支持RGB、212异常GT。根目录复用Target-only识别，不要求 `train/good`；原Source训练/校准RGB不参与本轮。
- `external_baselines/SuperADD`配置/宿主代码，`/root/autodl-tmp/dinov3`官方代码，`external_baselines/SuperADD/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth`原权重。
- 本模块 `sampling.py` 是v3生产采样控制逻辑；懒加载已有 `external_baselines/superadd_external/memory_variants.py:_chunked_exact_nearest_neighbors` 原精确距离后端；`external_baselines/SuperADD/tracks/industrial/src/industrial/nearest_neighbor.py` 保存原规则依据。三份代码SHA均冻结。不执行原无界采样while，不安装PatchCore；默认预检不导入Torch。

原图及GT的预检只做字节身份核对，不读取GT内容以选参；GT解码和诊断必须在完整预测冻结后。完整Q不存在于历史raw距离/index缓存，不能从其反推Q或使用SPARC特征。服务器smoke/infer各自每张图一次完整原宿主提取，共享给原Source/八支持基线复核及两个新增分支；smoke12与infer297是分别执行的显式阶段，总计309次提取，不复用不同阶段的查询状态。

本地 `downloads/aebad_s_segment_looB_raw_cache_20261013.tar` 只读检查已批准B capture记录、297/1188文件清单，标明不包含Q、不验证全部内部数组/逐raw哈希，不解压，不当成完整模型运行资产。SEGMENT不是本轮依赖。缺Source memory/MAD/支持/权重或正式CSV，预检列出缺失项，绝不补提支持、建Source bank、拟合MAD。

历史SOURCE/UNION指标来自原 `domain_metrics.csv`；两行全精度值交叉核对用户已批准审阅包中相同原基线值，记录源文件/SHA。只复用数字，不引入SPARC投影、特征或评价结果作为新分支。

## 五阶段与失败行为

1. `preflight`：默认始终只读，无Torch/模型/GPU，不执行采样、不创建结果或日志。列出sampling入口、完整配置、缺失代码/资产、依赖版本、容器可用RAM和磁盘。
2. `prepare`：只读取原八份支持，显式CUDA精确kNN、实际生产NumPy分组/密度/无放回选择、CPU float64常量求解。无backbone、Source重建/MAD重拟合或新特征提取；缺候选、非有限距离或非法索引时明确失败。采样诊断/来源在求解前保存，秩不足等失败保留现场、不抽样重试；严格满行秩、cond≤1e12、solve≤1e-10，没有其他回退。
3. `smoke`：仅原十二案例，无GT解码和正式指标；检查四层P与来源支持原始行/metadata完全一致，再复核两项基线、全图OT/求解、耗时和资源。不得按烟测指标/标签选择预算、参数或原型。
4. `infer`：必须先有同身份/代码/配置的成功smoke；全部297张重新完整提取，Source及八份Target全网格距离与历史比较atol=1e-7、rtol=1e-6，两项正式图像分数≤1e-5。并列NN索引可不同，距离仍必须复现。校验/数值失败保留证据，没有 `predictions_frozen`。
5. `evaluate`：冻结完整预测、原输入/原型/逐raw SHA、数值CSV和solver诊断通过后才解码GT；再次复核当前缓存Source及原Source/K8分数/全域图像汇总，防止离线评分变化被误认为完成。旧两分支五指标复用，只有两个新分支调用原 `_evaluate` 和ADEval CUDA（200阈值、FPR≤.3）；AP为average precision。缺Torch/CUDA/ADEval保持未完成，无替代AUPRO。

Sinkhorn每5步检查、最多1000步，行/列最大相对误差≤1e-6、总质量误差≤1e-8，更新一阶残差≤1e-8。失败保存实际迭代、误差、非有限项和Gram诊断，不扩大预算/改变阈值后自动续跑。旧输出、Source/MAD、支持和capture完全只读；新输出必须为空，半成品/重跑阶段拒绝。代码改变后使用新输出并重新prepare，不重写旧可信哈希。

## 输出、成本和解释边界

- `inputs.json`、`prepare_status.json`；`prototypes/L*.npz` 保存真实P、原Target索引/顺序、factor/epsilon、采样metadata及prototype身份；`prototype_provenance.csv` 保存支持图ID、物理patch索引、prototype顺序、入口/完整配置/seed和输入身份；`prototype_numerical_diagnostics.json`、`sampling_diagnostics.json`。采样诊断记录每组大小/k/radius、密度和权重范围、权重SHA/是否均匀/有效权重样本量、选中项group/weight/priority，来源CSV也记录这些证据；即使后续求解失败仍保留。
- `smoke_raw/` 与 `raw/`：紧凑四层Source/完整Target最小距离、P距离、R距离、两个新联合距离、P/R的NN索引、完整网格/位置/几何、历史raw与输入身份。没有Q、完整U、R或query×R矩阵落盘。
- 各阶段状态、`*_numerical_verification.csv`、`*_solver_diagnostics.json`；数值记录及诊断均绑定历史SHA。`cost_prepare/smoke/infer/evaluate.json` 分别记录支持/原采样/求解、backbone、原Source+Target基线检索、CPU FastRef、query到全部R检索、两项新评价、GT诊断、图表、峰值RAM/显存和新增存储。`whole_stage_seconds`包含入口预检/哈希、模型和输入加载开销；`seconds`及细分timing的计算函数范围另行标注，不把加载开销隐去。
- `evaluation/metrics.csv`、`metric_comparisons.csv`、`per_image_scores.csv`、`ranking_summary.csv`、`ranking_per_image.csv`、`distance_audit_per_image/summary.csv`、`defect_instances.csv`、`fixed_reference_instance_peaks.csv`、`small_instance_summary.csv`、原十二案例图、`status.json`及客观 `report.md`。

`metric_comparisons.csv`在真实评价完成后计算五指标差值，不预填结果：A=`UNION_TARGET512 − UNION_K8`表示按固定v3反密度适配压缩Target的影响；B=`FASTREF_TARGET512_SOURCE − UNION_TARGET512`表示同一份P上的细化影响；C=`FASTREF_TARGET512_SOURCE − UNION_K8`表示整个适配相对原强基线的变化。delta方向是branch减reference，单位是指标数值比例（乘100才为百分点）。

压缩下降不能写成FastRef失败；B下降仅适用于这份原型和当前配置。共用P控制本组比较中的采样差异，不能证明所有采样方式结论相同。排名沿用上述三项参照关系；正常尾部分位数和原固定十张困难正常不按本次结果改选。距离审计以原UNION和UNION_TARGET512为参照、共用原Source MAD尺度，**R值仍是reference减branch**，保留符号，正常全patch/异常原GT覆盖patch，每图等权再四层等权的R_N/R_A/G。细化不保证距离下降。

实例复用8连通、原面积、原adaptive-max patch覆盖和原图Chebyshev16环带（排除所有GT）；保留碎片、空环带和映射消失状态。保留每分支自己的instance/ring最大值及margin，同时另表比较两个固定参照峰值。沿用原诊断“最小三个离散GT实例”的描述规则，不引入测试驱动面积门槛或额外案例。响应差不等于原始距离差，不用测试标签设部署阈值。

当前N=512的预计新原型/求解常量约30MiB，297份raw未压缩估算约0.65GiB，含2GiB余量的磁盘门槛约2.68GiB；这些是估算，不是实测。支持/Source/模型只读复用。最低可用RAM16GiB是门槛，不是峰值保证。一次全图OT约54MiB/层（float64），R约136MiB/层；最多四层CPU并行，原及新NN均复用GPU Exact，新NN临时矩阵限制64MiB。Source/八支持基线按原默认检索chunk复现，不擅改历史检索形状。

主要成本为支持SHA/读取与v3密度采样、完整backbone、四层完整OT、Source/八支持复核、query到m个refined vectors的NN及两个新像素评价。不能可靠预估真实总耗时；先据固定smoke的实际每图耗时/内存外推297张，另加prepare/evaluate。1000步Sinkhorn预算不代表论文实时速度。CPU_WORKERS默认4，对应层并行；脚本未设置的BLAS线程默认3，最多约12个BLAS工作线程，避免四层各启15线程争抢。用户已有线程设置保留；真实提速未验证。

## 用户后续服务器手动命令（本轮未执行）

```bash
export PROJECT_ROOT=/root/autodl-tmp/DINO-NVS
export PYTHON=/root/miniconda3/bin/python
export RESULTS_DIR="$PROJECT_ROOT/DINOv3/outputs/mad_equal/aebad_s_target_fewshot_seed42_v1"
export CAPTURE_DIR="$PROJECT_ROOT/DINOv3/outputs/mad_equal/aebad_s_union_support_loo_B_20261007_230639"
export SUPPORT_CACHE_DIR="$PROJECT_ROOT/DINOv3/outputs/mad_equal/aebad_s_union_background_replay12_seed42_v1/support_features/target_matched/background"
export DATA_DIR=/root/autodl-tmp/datasets/AeBAD/AeBAD_S
export OUTPUT_DIR="$PROJECT_ROOT/DINOv3/outputs/mad_equal/aebad_s_fastref_target512_source_bounded_density_v3_$(date +%Y%m%d_%H%M%S)"
export DEVICE=cuda:0 CPU_WORKERS=4
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
cd "$PROJECT_ROOT" || exit 1

# 无Torch/GPU的只读预检；不创建输出。
bash DINOv3/MADEqual/aebad_s_fastref_k8_p0/scripts/run_aebad_s_fastref_background.sh preflight --no-write

# 以下由用户在独立review/预检通过且明确安排计算后执行；本轮未执行。
# 同一OUTPUT_DIR/配置/代码；任一步非0即停止，不重试、不放宽数值标准。
bash DINOv3/MADEqual/aebad_s_fastref_k8_p0/scripts/run_aebad_s_fastref_background.sh prepare &&
bash DINOv3/MADEqual/aebad_s_fastref_k8_p0/scripts/run_aebad_s_fastref_background.sh smoke &&
bash DINOv3/MADEqual/aebad_s_fastref_k8_p0/scripts/run_aebad_s_fastref_background.sh infer &&
bash DINOv3/MADEqual/aebad_s_fastref_k8_p0/scripts/run_aebad_s_fastref_background.sh evaluate
```

直接入口为 `"$PYTHON" -B -u -m DINOv3.MADEqual.aebad_s_fastref_k8_p0.runner preflight --project-root "$PROJECT_ROOT" --results-dir "$RESULTS_DIR" --capture-dir "$CAPTURE_DIR" --support-cache-dir "$SUPPORT_CACHE_DIR" --data "$DATA_DIR" --output-dir "$OUTPUT_DIR" --cpu-workers 4 --device cuda:0 --no-write`；其他阶段替换mode并删除`--no-write`。可显式传`--memory/--mad/--checkpoint/--dinov3-root`，后续必须与prepare冻结的路径一致。CLI不提供测试效果驱动的lambda/epsilon/iterations搜索。

成功退出0、preflight blocked退出2、评价未完成退出3、异常退出1。默认脚本没有后台管理、远程连接或自动关机；后续安排超过一小时/用户主动要求的正式命令时，按既有约定只在全部成功后另行加入AutoDL关机。

## 本地检查和尚未验证

Python `D:/anaconda/miniconda/envs/chinamoblie/python.exe` 为3.11.12，NumPy1.26.3；未导入真实Torch，未安装依赖。测试命令：

```powershell
& 'D:\anaconda\miniconda\envs\chinamoblie\python.exe' -B -m unittest DINOv3.tests.test_mad_equal_aebad_s_fastref_k8_p0 -v
```

数学测试用独立普通Sinkhorn、显式逐行W求解和一阶条件作参考；原数学内核和历史评分/缓存完整性回归保留。v3采样测试实际执行生产分组、密度、指数race、runtime接入和prepare代码，而非用另一种NumPy采样算法替换生产采样；仅缺失Torch距离/Tensor/CUDA接口被明确替身替换。小型成功prepare还执行真实缓存reader、四层512×1280普通Gram求解和小query数学细化，并验证同一P与原始行/索引身份。退化特征prepare验证秩失败时保留采样诊断/来源、不生成成功状态。

上述小构造验证**不代表真实Torch/CUDA、真实DINOv3特征、原插值、全域Sinkhorn或ADEval通过**。真实P的条件性、正常特征权重分布/是否均匀、epsilon尺度、297张基线复核、峰值资源、耗时和指标仍待服务器prepare/smoke/infer/evaluate。本轮不安装依赖、不提取真实特征、不执行模型/正式评价、不连接服务器或修改旧代码/正式结果/已有报告。

历史v1有22项本地测试通过；历史v2有32项通过但因原循环适用性保守阻断。v3已解除该采样预算阻断；本地仍缺完整B运行目录、Source memory/MAD/CSV、原八支持、原图/GT、权重/hub代码及足够RAM，不能在此执行真实阶段。只读tar清单不含完整Q、不替代这些资产，不解压或认可全部内部raw内容。最终本地检查结果见下方记录。

2026-10-13 v3最终专项 **38项通过，77.174秒**。包含原实际AST的两类不退出反例、有界生产采样的小预算/余数/小池/重复/非有限输入、局部RNG复现、来源/数量/顺序、v1/v2及混合原型拒绝、两新增分支共同P、成功prepare及秩失败保留现场；原12项数学核及全域合成评价/历史分数复核回归通过。成功prepare仅用小型构造缓存及明确距离/Tensor/CUDA替身，真实普通矩阵求解和小query细化通过，不是实际DINOv3或GPU实验。

9份Python文件AST检查通过，默认脚本LF（CRLF=0）；没有在本地执行Bash。实际默认只读预检返回2/blocked，采样尺寸检查为 `static_budget_check_passed_runtime_unverified`（M=111392、N=512、100组、最大组1114）；列出22项缺失资产/根目录问题及RAM不足，**无旧采样兼容错误**。未导入Torch、未创建输出、未执行采样或模型操作。采样最大组距离矩阵估算4963984字节（约4.73MiB），不是实测峰值；其他Torch工作区/CPU/求解开销另计。
