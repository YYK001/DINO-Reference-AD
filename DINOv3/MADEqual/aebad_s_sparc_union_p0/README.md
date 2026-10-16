# AeBAD-S/background：SPARC_UNION_K8 补充对照

仅新增 `SPARC_UNION_K8`；background、Seed42、原K8、297张（85正常/212异常）和原十二案例固定。组合外部方法与memory union，不是论文表格复现或提前认定的本章创新。本轮只完成工具、合成测试和说明，没有连接服务器或执行真实推理/正式评价。

## 定义与父实验

只读父目录：`aebad_s_sparc_background_mixed_20261009_095419`，身份 `a5d86dcae44db59652170e54304d0d11e91e5fd4013ecf84129e63c936734a88`。

每层读取既有 `mu/basis/rank`，使用原 `Projection.correct`：

`C_p(f)=f−V_p[V_pᵀ(f−mu_p)]`。

原八张支持各自按自己的物理位置校正，原顺序完整拼接为Target bank；不以mu替代，不去重/压缩/重采样。查询按查询自身位置校正，一份校正查询分别调用原 `_exact_top1_chunked` 检索既有校正Source bank和新Target bank；两个距离各自只除1280一次，逐层patch取min。检索跨完整bank，不限制位置相同。

新分支与 `SPARC_SOURCE_K8` 共用父实验 `sparc_source_mad_state.npz`：逐层MAD→四层等权→原插值→全图max。保留负响应，无额外融合或逐图归一化。

| 空间 | Source | 加入Target |
|---|---|---|
| 未校正 | SOURCE（历史） | UNION_K8（历史） |
| 原SPARC校正 | SPARC_SOURCE_K8（历史） | SPARC_UNION_K8（新增） |

同一行只改变是否追加Target；跨行还改变了校正、Source重建和对应MAD，不能声称只隔离投影。旧三个分支的正式五指标直接复用，并标注来源文件、SHA及父身份；仅新增分支调用一次原全域像素评价。

## 必需资产与完整性

`--sparc-base-dir` 必须指向完整父实验目录，不能填审阅tar.gz或把解压的审阅包当完整运行资产。使用：

- 原inputs/prepare/infer记录、support/projection provenance、evaluation状态/指标/逐图分数/验证CSV；
- 四层投影的12个npy文件，原 `sparc_source_memory.pt` 和 `sparc_source_mad_state.npz`；
- 全部297张 `sparc_raw`；
- 原八张支持NPZ和身份、顺序；原查询RGB/GT、宿主源码和权重；
- 原完整B capture及正式CSV/Source MAD，用于重建未校正地图和比较，不重新计算旧像素指标。

`inputs.py` 内固定父身份和九个批准记录的SHA，来自用户已下载的审阅包（包SHA：`d048a096906f68051eb746d613cd7357199b40fe759fc6b6de2574dd7aad12df`）。投影/Source bank/MAD和旧raw分别核对原prepare/infer历史SHA；外部输入核对原required_files记录。新模块及复用依赖另行冻结，修改文件不重新认可。旧Source RGB及Source calibration raw不成为本轮计算前提。

禁止重新提取Source、拟合投影/MAD、重建Source bank，缺资产不回退prepare；父目录、旧结果和旧memory完全只读。新输出路径重叠、半成品或重复阶段被拒绝，没有自动续跑。

## 阶段与资源

1. `preflight`（默认）：始终只读，无Torch/模型、无文件创建，明确列缺失项；父资产未加载时资源只是固定118×118协议估算。
2. `prepare-target`：仅NumPy加载投影并校正原支持，保存四层独立Target bank、完整支持编号/原patch坐标映射及float32实际坍缩误差。无Torch/backbone、无Source建库、无SVD/MAD拟合；无需GPU，但需要足够RAM。
3. `infer`：每张图一次完整backbone，每层一次校正，两侧Exact检索。全patch四层Source距离对照历史（`atol=1e-7,rtol=1e-6`），同一冻结MAD下原图像分数误差≤`1e-5`，完整grid/positions/processed shape/拼接轴及RGB字节身份一致后保存新raw。Source索引并列可不同，验证看距离而非索引。失败保留CSV/已完成的新raw，没有 `predictions_frozen`。
4. `evaluate`：冻结297张预测、历史复核和全部SHA通过后才打开GT；重建旧地图核对原三分支分数及图像指标，保留历史数值。仅新分支跑Image AUROC/AP、Pixel AUROC/AP、ADEval CUDA AUPRO（200阈值/FPR≤0.3）。AP仍为average precision，无CPU近似替代。

预计新增Target约 **2.125 GiB**，新raw约 **0.462 GiB**，磁盘预检要求约 **4.587 GiB**（含2GiB余量）；最低可用RAM门槛 **8 GiB**，不是峰值保证。不会生成Source候选临时文件，也不复制旧Source bank和投影。真实RAM/CUDA峰值及速度尚未测量。

`--cpu-workers` 默认8，范围1–15，必须在后续阶段保持与prepare-target相同。支持最多8项并发，查询层最多4项并发；工作线程只调用原NumPy `Projection.correct`，回收结果按原顺序，GPU检索仍串行。没有GPU投影迁移、精度改变或随机采样。启动脚本给未设置的OMP/MKL/OPENBLAS线程数默认1，避免内外层线程争抢；不会覆盖用户已设的值。实际提速待测量，线程数不代表一定占满所有核心。

## 输出与解释边界

- 准备：`target_memory/L*.npy`、`support_mapping.npz`、`target_provenance.json`、`support_collapse_verification.csv`、`prepare_target_status.json`。
- 推理：`raw/test_*.npz` 保存本次四层Source/Target/union距离、两侧索引、完整位置与几何；`infer_verification.csv`、`timing_per_image.csv`、`infer_status.json` 冻结历史SHA。没有query×memory矩阵。
- 评价：四分支 `metrics.csv/per_image_scores.csv`、两条比较的 `ranking_summary.csv`、原定义的 `defect_instances.csv`、两套距离审计、十二案例图（包含IMG_8863/IMG_9385）、`status.json/report.md`。
- 成本：`cost_prepare_target/infer/evaluate.json`，分别记录支持处理、backbone、查询校正、Source验证检索、Target检索、评价和新增磁盘/峰值RAM/CUDA；不重报历史Source适配时间。

主要比较仅对原UNION及SPARC_SOURCE。实例复用8连通、原图Chebyshev半径16环带、排除全部GT及原映射；小实例、碎片、空环带和映射消失状态全部保留。

距离审计分别为 `(d_UNION−d_new)/old_Source_MAD_scale` 和 `(d_SPARC_SOURCE−d_new)/SPARC_Source_MAD_scale`，明确标reference/scale。正常全patch、异常原GT覆盖patch；每图/层均值，跨图等权、四层等权，`G=R_N−R_A`。

同一查询下union距离及同MAD地图不增加（地图允许已有 `1e-5` 数值误差），不保证AUROC、AP、AUPRO、margin或G改善。两条SPARC分支可直接比较同尺度margin；与未校正UNION比较时MAD尺度不同。响应差不是原始距离下降。坍缩后的重复/近似重复支持中，Top-1编号不证明该支持有独有语义作用。不使用测试标签选参数或部署阈值，不把排序错误称固定阈值误报/漏检。

## 后续服务器命令（本轮未执行）

```bash
export PROJECT_ROOT=/root/autodl-tmp/DINO-NVS
export PYTHON=/root/miniconda3/bin/python
export SPARC_BASE_DIR="$PROJECT_ROOT/DINOv3/outputs/mad_equal/aebad_s_sparc_background_mixed_20261009_095419"
export OUTPUT_DIR="$PROJECT_ROOT/DINOv3/outputs/mad_equal/aebad_s_sparc_union_k8_$(date +%Y%m%d_%H%M%S)"
export CPU_WORKERS=8
export DEVICE=cuda:0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$PROJECT_ROOT" || exit 1

# 默认只读预检，退出码必须为0；直接入口也始终只读。
bash DINOv3/MADEqual/aebad_s_sparc_union_p0/scripts/run_aebad_s_sparc_union_background.sh

# 后续明确安排实验时才执行，三个阶段共用同一OUTPUT_DIR。
bash DINOv3/MADEqual/aebad_s_sparc_union_p0/scripts/run_aebad_s_sparc_union_background.sh prepare-target &&
bash DINOv3/MADEqual/aebad_s_sparc_union_p0/scripts/run_aebad_s_sparc_union_background.sh infer &&
bash DINOv3/MADEqual/aebad_s_sparc_union_p0/scripts/run_aebad_s_sparc_union_background.sh evaluate
```

直接入口：`"$PYTHON" -B -u -m DINOv3.MADEqual.aebad_s_sparc_union_p0.runner preflight --project-root "$PROJECT_ROOT" --sparc-base-dir "$SPARC_BASE_DIR" --output-dir "$OUTPUT_DIR" --cpu-workers 8 --no-write`。其他阶段替换mode并去掉 `--no-write`。成功退出0、预检blocked为2、评价证据不完整为3、异常为1。脚本没有后台管理或自动关机；正式运行安排时按用户约定，在所有阶段成功后另行加入AutoDL关机。

## 本地验证与未验证

Python：`D:/anaconda/miniconda/envs/chinamoblie/python.exe`；本轮不导入真实Torch，不改变环境。

```powershell
& 'D:\anaconda\miniconda\envs\chinamoblie\python.exe' -B -m unittest DINOv3.tests.test_mad_equal_aebad_s_sparc_union_p0 DINOv3.tests.test_mad_equal_aebad_s_sparc_p0 -v
```

2026-10-09：**47项通过，84.506秒**（新增19项、原三分支28项回归）。覆盖只读批准记录、实际原投影并行/串行一致、支持坍缩与重排、跨位置检索、重复/并列索引、一次提取/一次校正/两侧检索、/1280一次、patch min→MAD、单调性、历史距离/分数偏移拒绝、哈希/半成品拒绝、无Torch默认预检、非默认评价CUDA初始化、四分支编排仅一次新像素评价、评分后端一致偏移拒绝完成。

距离测试执行实际 `_exact_top1_chunked` 函数AST，但张量/CDist是明确的NumPy替身；地图测试使用原评分AST及identity-size插值；完整297 tiny fixture的AUPRO是替身。这不是CUDA cdist、原真实双线性插值或ADEval GPU验证。真实完整查询复核、支持坍缩误差、正式指标、峰值资源和速度均未执行。

实际本地只读预检返回2/blocked，Torch未导入、输出目录未创建，列出24项缺失：完整父目录的九个记录/结果文件、12个投影npy、Source bank/MAD及297 raw目录。本地只有审阅包可读，原支持/查询、宿主权重等服务器运行资产也未在本地核实为齐全。脚本为LF；Windows的bash为WSL入口，本轮未执行Bash语法/运行验证。

旧SPARC模块及共享代码未修改，默认仍为三分支；本轮新增独立模块和一份专项测试，没有解压审阅包或修改历史正式结果/报告。原依赖源码13项与父记录字节一致；官方nearest_neighbor本地为CRLF，规范化LF后匹配父SHA，未改写文件，服务器仍严格检查原历史字节SHA。

2026-10-12 序列化修正：服务器 `prepare-target` 已成功，`infer` 因几何元数据含 NumPy `int32` 而报 JSON 序列化错误。新入口现与原SPARC入口一致，在调用 `image_geometry` 前将grid和processed shape的标量显式转为Python `int`；NPZ中的原 `int32` 数组、距离、索引和数值校验不变。新增回归用例执行实际infer缓存写入路径（Torch/模型为替身），在修复前复现同一TypeError，修复后验证元数据及全部数组正确保存。修复后新旧专项合计 **48项通过，90.835秒**（新增模块20项、原三分支28项），两份修改Python文件AST检查通过；真实CUDA重试仍待用户执行。

修复改变了已冻结的运行代码SHA。保留失败目录及其原准备产物，使用新的 `OUTPUT_DIR` 从 `prepare-target` 重新启动；不要重写旧inputs/哈希、覆盖失败输出或绕过校验。原三阶段命令仍适用，支持准备在该次服务器运行约一分钟；实际重试时间可能不同。
