# SACP：Source 内部差异对照下的条件化残差子空间修正

独立、免训练的固定配置实现。默认仅 `preflight`，不导入 Torch、不创建运行目录、不初始化 CUDA。此交付只有本地 CPU 验证，尚无真实 SACP 检测结果；不连接服务器、不安装依赖、不启动提取或实验。

主要问题是：在同一完整 K8、宿主和冻结评分下，SACP 是否胜过 UNION、SEGMENT、固定任务分工参照？Source 内部差异对照是否胜过普通 Target 方向及等量均匀缩距？六个定位分支全部评价，不先用描述性 AUC 决定是否运行方法。

## 冻结范围与实际任务

`protocol.py` 是唯一固定配置；`job_plan()` 派生任务，prepare 后分别封存 `protocol.json`、`job_plan.json`。参数没有按域、层或测试结果搜索的入口。

|项目|固定安排|
|---|---|
|Source|455 张构库图，66 张校准图完全不参与方向估计|
|支持|seed42，每域原顺序 K8，24 张均排除评价|
|评价|background 297（85/212）、illumination 265（67/198）、view 364（84/280）；共 926（正常/异常）|
|宿主|原 DINOv3 ViT-H+/16，L7/15/23/31、1280D、完整视野/原切窗与拼接、原预处理|
|memory|正式 Source 每层 100K，完整 Target K8；原增强/采样器不变|
|分组|仅正式 Source；每层 32 中心；抽样最多 16384 行，单次 PCG64 初始化、最多 30 轮 Lloyd|
|拟合|16 张 Source 对照 +24 张支持；12 个域×层作业，每个含 32 组|
|patch|每图最多 4096 个物理位置；同图四层/两折一致；每图每组最多 64 差分|
|子空间|共同表示最多 32 维；最终最多 2 方向；Source 收缩 .1，lambda=.5|
|定位图|UNION_K8、SEGMENT_K8、SACP_CONTRAST、SACP_TARGET_DIR、SACP_SOURCE_DIR、SACP_MATCHED_UNIFORM|
|评价|3 域×6 图=18 个逻辑 CPU 像素/AUPRO 作业；兼容基线通过逐图地图、GT、人口核对后可复用 6 个，新增 12 个|
|训练|0；无合成、分割头、adapter、重新校准或正式 Source 构库|

CH1 是主评分，CURRENT 是补充，两者共享像素指标。另有 `SEGMENT_SCORE_UNION_MAP`：取 SEGMENT 的图像指标、UNION 的定位指标，不生成第七张图或额外 AUPRO。各域分别报告，再三域等权；完整指标表含 56 行（7 参照×2 评分×4 域/宏平均）。

## 数学和数值顺序

所有正式距离为父实现 FP32 Exact 欧氏距离除以 1280 **一次**；不使用平方距离替代检测输出、不额外 L2 归一化。32 类 K-means 内部仅用欧氏平方距离求最近中心，不压缩或替换 memory。初始化为 seed42 的 PCG64 无放回行抽样；空簇保持原中心，无重试。

在原 455 张构库图的既定 A/B 划分内，按 CRC32(`42|SACP_SOURCE_CONTROL|fold|relative_path`) 及路径打破并列，固定选各 8 张。A 查询检索 B 库，B 查询检索 A 库；各域 K8 同时检索对应 B/A 库。两折都不含对应 Source 查询及增强派生，66 张校准图不参与选择。正式检测始终使用完整 Source＋K8。

`delta=x-a(x)`，结构组由折库锚点最近的 Source 中心决定。两折各 1/2，折内有效图像等权，图内差分等权；共同表示中 Source/Target 各 1/2。每组每类每折至少 4 张不同图像，否则共同回退。相同 Target 在两折出现仍是同一张图，诊断同时记录每折及去重图像数。

对加权差分**不减均值**：FP32 随机化 SVD（oversampling=8，QR power iterations=2）取公共欧氏正交基 P，最多 32 维。秩阈值为奇异值/特征值大于最大值的 1e-6，并以 FP32 tiny 为绝对下限。大矩阵保持 FP32，不构造 1280×1280 全量协方差。共同基仅表示统计，不投影/重构查询。

在共同 P 内使用小型 FP64 二阶矩与分解：

```text
C_S = E[(P^T delta_S)(P^T delta_S)^T]
C_T = E[(P^T delta_T)(P^T delta_T)^T]
tau_S = trace(C_S)/p
tau_0 = trace(C_S+C_T)/(2p)
B = .9 C_S + (.1 tau_S + 1e-6 tau_0) I
C_T v = gamma B v
```

Cholesky/对称白化求解，不显式求大逆。选 gamma>1 的前两方向，W=P V 后做欧氏 QR 得 U。无能量、无 gamma>1、秩不足均回退 UNION。TARGET_DIR 和 SOURCE_DIR 在同一 P 内分别取 C_T/C_S 最大特征方向，使用与主方法相同有效组、方向数、lambda、锚点。任一方向对照秩不足则三个分支**共同回退**；SOURCE_DIR 仍是 K8 UNION 上的方向对照。

正式查询先求原 Exact d_S、d_T 及索引。Source/Target 并列选 Source；Target 间并列选支持 manifest 原顺序。保留原 UNION 锚点 m*，不在修正后重搜。查询组由正式 Source 最近邻行决定，残差 r=q-m*：

```text
eta = ||U^T r||^2 / ||r||^2
d_SACP = d_U * sqrt(1 - .5 eta)
```

d_U 保留原 Exact 数值，不改成重算向量范数。零残差、无有效 U、lambda=0 严格返回原 UNION。eta 仅容许 1e-5 内的舍入越界，记录裁剪数及裁剪前最大值；明显越界报错。lambda=0 是单元测试入口，不是正式扫参选项。

逐图逐层均匀对照：FP64 汇总 Delta=sum(d_U-d_main)、a=Delta/sum(d_U)，FP32 输出 `FP32(1-a)*d_U`；零总距时 a=0。记录实际下降量、差值及 `eps32*sum(d_U)` 舍入上界，超界拒绝。它匹配原生总缩距，不能声称匹配空间分布、校准后响应或实例 margin；不通过调整位置/GT补齐舍入差。

全部距离分支共享父 Source patch MAD 与冻结 CH1 peak/local image MAD，四层原生 FP32 等权均值、原 FP32 插值。无新 MAD。兼容父 UNION/SEGMENT 通过原容差核对后直接保留其 z 数组；新分支未发生距离变化的位置也保留其原 UNION z，保证共同回退精确。

## 近期资产与重复工作

默认父运行是迁移后仍在的：

```text
/home/ubun/YYK/runs/aebad_reference_residual_compact_v1_20261027_002907
/mnt/mechanical_disk/YYK/reference_residual_cache/aebad_reference_residual_compact_v1_20261027_002907
```

不假设旧 AutoDL/Kaggle 缓存存在。`--recent-run` 可重复指定兼容的**原参照残差工具**父目录，不读取其训练模型。

|资产|处理|
|---|---|
|source_original，四层 100K|只读直接引用，不重构/复制 bank|
|source_folds/fold_A、fold_B|必须已有；缺失阻断，不自动构库|
|normal_features 原完整正常网格|只读引用选中的 Source16+Target24，共40图；不遍历其余505图内容|
|input_distance_mad.json|直接继承同一正式 Source 对应的 patch/image MAD|
|training_queries 正常图 source_nn|原完整特征 seal、对折 bank seal、位置相同才复用 Source 控制图索引；否则对选中4096位置做 Exact|
|queries 原完整 Q|存在且原身份/位置/四层 shape 正确则复用；合成 tile、适配后特征、logit 不可替代|
|predictions 父 U/S、ds/dt/si/ti|同一查询/银行/支持顺序及原容差复核后继承；bank 不匹配拒绝|
|父像素指标/AUPRO|逐图地图、GT、完整名单身份一致及小结果封存有效后复用；不把报告存在视为 payload 齐全|

上轮正式预测为 stream；父 review 包无完整 Q，且预测里只保存 aggregate ds/dt/si/ti，没有八张支持的 perd/peri。服务器预检会查实际目录，若无另存 Q，需要**926 张重新前向**，每图一次供六分支共用。不能由距离图恢复残差向量。

在该情形仍须重新完成每查询四层的正式 Source 与逐支持 Exact（926×4×9=33336 次按 bank 的搜索）；已有 aggregate NN 用于严格复现和保持原锚点，不省略重新提取 Q 的兼容性验证。SEGMENT 直接复用这一批逐支持结果及既有几何。拟合最多 64 次 Source 控制图×层搜索（兼容旧正常 NN 可省）和192次 Target×折×层搜索，无新增正常图前向。

本地资产检查仅针对当前 Windows 文件系统：父 archive 元数据能读取，不代表 Linux 绝对路径在本机可读。交付预检目录 `local_preflight/` 仅保留于完整工作区，不上传GitHub；其缺失清单不证明服务器资产缺失。尚未连接服务器确认目录内容。实际预检会分别列可复用、需新增、需重复、无法确认项，使用显式 `--inventory-dir` 可保存清单。

## 阶段、续跑与资源

|阶段|实际行为|
|---|---|
|preflight（默认）|读元数据、小 NPY 头/位置；输出资产清单，不导入 Torch、不扫描大数组 SHA|
|prepare|冻结输入、配置和任务；4 层 Source-only K-means及 bank 行分组，不提取、不构库|
|fit|Source/Target 对折 Exact、加权差分和12个域层的32组子空间；0梯度训练|
|predict|一次 Q/共享检索，六分支；全部926图冻结后才可评价|
|evaluate|原全像素精确 CPU AUROC/AP、两种图像评分、最终地图实例及 eta 附表|
|evaluate-aupro|单独 GPU 阶段，原 ADEval1.1.0、200阈值、FPR≤.3；不需要模型前向|
|summarize|三域/等权宏平均、全部对照和完成状态；有缺项保持 pending|
|pack|报告、日志及小型分组/子空间；排除 raw Q、预测大数组、bank、scratch和checkpoint|
|all|只执行以上固定阶段；成功/失败保留现场并打review包；不安装、不关机|

每个对应缓存、域×层子空间、逐图预测、域CPU结果及域×分支 AUPRO 独立提交。续跑用**同一 run/cache/参数/代码**；完成的对应、拟合、预测和评价不重算。未提交的当前图/组可以重算。输入/代码冲突明确拒绝，不改写旧 manifest。CPU中间精确指标沿用原独立 seal 与范围/像素数验证。

正式 Exact、残差/输出 FP32；骨干沿用原宿主 precision，不另启 BF16/TF32。只有32维二阶矩/分解及细长 QR 参考允许FP64。**没有自定义 Triton**。单 GPU；CPU最多8工人，prepare/fit BLAS不超过所给cpu-workers，像素评价按图有界并行。参照 LRU 默认6GiB，按 bank/层组织统计，记录实际上传次数/字节，不为分支重复上传和检索。

大缓存默认放机械盘 `storage/sacp_cache`，SSD `runs` 只放报告与小元数据；SSD私有 scratch 在每域精确像素评价后清理，仅删除本次创建文件。拟合差分是最多4096位置的子集，正式Q默认stream不落盘。OS页缓存与进程RSS不同。未做实机峰值测速，不承诺内存绝不超限或几小时必完成，也不以预设可用RAM/磁盘门槛拦截。

CPU数学fixture的时间不能外推真实提取、100K Exact或ADEval。实际 `costs/*.json`、`stage_costs.csv` 分开记录各阶段墙钟、进程峰值RAM、Torch显存峰值、前向/搜索和参照上传；复用部分不计作新执行，父实验成本不冒充SACP算法耗时。汇总阶段自身成本在返回后写入costs，`stage_costs.csv`对应其进入时的记录。

## 手动运行（本轮未执行）

代码须先经review、后由用户授权发布/同步到工作站。所有读写都在YYK自己的目录或其storage链接内；不改base环境，不自动安装包。

先运行已有yyk环境下的预检，默认GPU选择仅作为未来任务的可见设备限制，不在preflight调用GPU：

```bash
cd /home/ubun/YYK/DINO-Reference-AD
export GPU_INDEX=1 CPU_WORKERS=8
export RUN_ROOT=/home/ubun/YYK/runs/aebad_s_sacp_seed42_v1
bash DINOv3/MADEqual/aebad_s_sacp_p0/scripts/run_workstation.sh \
  preflight --inventory-dir "$RUN_ROOT/inventory"
```

确认原完整资产及环境与父运行一致后，用户可以显式启动tmux后台实验：

```bash
cd /home/ubun/YYK/DINO-Reference-AD
export GPU_INDEX=1 CPU_WORKERS=8
export RUN_ROOT=/home/ubun/YYK/runs/aebad_s_sacp_seed42_v1
mkdir -p /home/ubun/YYK/runs
tmux new-session -d -s sacp_seed42 \
  "cd /home/ubun/YYK/DINO-Reference-AD && env GPU_INDEX=1 CPU_WORKERS=8 RUN_ROOT='$RUN_ROOT' bash DINOv3/MADEqual/aebad_s_sacp_p0/scripts/run_workstation.sh all > '${RUN_ROOT}_launcher.log' 2>&1"
tail -n 40 -f "${RUN_ROOT}_launcher.log"
```

停止后续跑同一目录用同一all命令；也可将stage改为 `fit`、`predict`、`evaluate`、`evaluate-aupro`、`summarize` 或 `pack`，不得省略上游冻结。新一轮请改RUN_ROOT并保持父资产只读。无自动关机。

本地CPU测试命令：

```powershell
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'
& 'D:\anaconda\miniconda\envs\chinamoblie\python.exe' -B -m unittest DINOv3.tests.test_aebad_s_sacp -v
```

本轮 **26项通过，27.974秒**；11份Python源码编译及工作站脚本`bash -n`通过。详见`validation_results.json`；原始测试日志`validation/cpu_tests.log`仅保留完整工作区，不上传GitHub。集成只使用64行小bank、6张人工特征/GT的CPU夹具；原真实926张基线复现和CUDA/ADEval仍未验证，汇总状态测试的AUPRO记录不属于实验结果。

## 继承检查的完整范围

1. 小JSON/config/任务/目录封存及原人口、支持顺序、Source折划分，原校准不参与拟合。
2. 输入manifest身份、NPY实际shape/dtype、小positions及NN索引/银行身份；可信本地大数组默认不重算内容SHA。
3. 有限性、有效图像数、秩、欧氏正交、eta舍入、均匀总下降误差和共同回退。
4. 父距离复现保持原 `atol=1e-7,rtol=1e-6`；同缓存评分 `atol=1e-5,rtol=0`，不使用更宽阈值继续。
5. 评价时一次计算最终地图/GT SHA，对应父基线和AUPRO地图身份；小结果表哈希、精确像素结果独立seal/有限范围/像素数、18项AUPRO完整性。
6. 输出/scratch与父资产目录隔离；只清理本轮私有临时文件，打包排除scratch。默认无容量准入门槛。

这些检查已全部在此列明；不隐藏全盘/大缓存递归校验。可信本地快速模式不能发现所有未改变manifest的同shape大数组内容篡改，审计不声称全量内容哈希验证。

## 输出和结论边界

`metrics.csv`、`metric_comparisons.csv`、`evaluation/*/per_image_scores.csv`、`defect_instances.csv`、`margin_changes.csv`、`margin_summary.csv`、`mechanism_summary.csv`、`eta_regions.csv`、`subspace_statistics.csv`、`fit_image_contributions.csv`、`numerical_verification.csv`、`input_distance_mad.json`、`source_control_images.csv`、`costs/`、`stage_costs.csv`、`objective_report.md`、`status.json`。

最终地图同时算各自峰值/固定UNION峰值margin；小实例、映射消失、空环带等原NA保留。R_N/R_A采用正常图所有像素/异常图GT像素的UNION减候选**最终融合Source MAD响应**，先每图平均再图像等权，G=R_N-R_A；另存逐图有符号量，不混用原生距离。

eta附表是困难正常原生UNION top1%物理位置、GT覆盖及环带，不代替margin或部署门控；所有拟合/预测均不读取测试GT，只有全926预测冻结后评价读取。gamma>1不证明真实采集扰动；普通Exact对应不宣称物理同源。三个域都用于过开发，不能称盲测。

原 `新建文件夹/code_artifact.md` 仅为思路来源：本实现修正平方距离、中心化漏均值、广义方向非欧氏正交、诊断AUC替代方法效果等问题；不引用其未经核实的文献/创新结论。若无优势如实结束当前固定配置，不自动扫lambda/rank/层/域，不把失败说成所有子空间方法或整个任务不可能。
