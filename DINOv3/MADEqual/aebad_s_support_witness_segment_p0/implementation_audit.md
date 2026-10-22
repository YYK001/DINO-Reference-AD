# 实现审计：支持端第三图佐证 P0

本轮不执行真实实验。研究依据已读：窗口报告第13/18/22/27/28节、background连接级复核与端点长度分析、原SEGMENT/GAP/crossdomain/replay实现。端点长度不足以支持统一删长连接；PAIR和GAP结果存在定位/实例损失，旧损失“有所恢复”不等于完全修复。支持敏感性878张与本轮原K8的926张协议不同，不能混用。

## 复用与新增

| 内容 | 实际入口 | 本轮行为 |
|---|---|---|
| 三域父身份、支持顺序、Source两级MAD、父像素来源 | gap_soft_segment_p0.inputs.Bundle；support_distance_crossdomain.live_parent.LiveParent | 只读复用；自己的code/protocol身份另建，不改父identity |
| background距离及端点对齐 | neighborhood_segment_p0.inputs.context/load_image | 按物理positions映射，原target_index来自每支持Top1 |
| illumination/view距离、原几何身份 | support_distance_crossdomain.inputs.decode | 保留新父自身原身份与SHA；不用union-only缓存 |
| 支持NPZ格式/协议 | target_fewshot.replay12.check_support_header的只读契约 | NumPy+zipfile读头；不导入support extractor，不调用旧runner |
| 稳定Top2和Target-only LOO线段 | target_local_segment.geometry.geometry_layer/bounded_distances | LOO只传剩余7图；正式query直接读原segment_distance，不改变历史检索后端 |
| 微小负缩距与均匀总量 | gap_soft_segment_p0.method.compute的运算次序/ULP界 | 复刻这一小段数学，不调用GAP权重；gate为支持端布尔佐证 |
| CH1/CURRENT、地图 | gap.prediction.score；neighborhood.analysis.score_maps；projection_layer_diagnostic.numerics | 同Source MAD、原生FP32四层均值、原FP32插值；不改pooling |
| 实例/环带/小缺陷与峰值 | gap.evaluation.instance_analysis | 参数化地图字典传四分支；不改旧全局BRANCHES，不伪装成PAIR |
| 图像、全像素CPU/AP | ch1_scoring_inheritance.scoring.image_metrics/pair_errors；projection_layer_diagnostic.metrics.exact_binary | 精确全像素，同分组；原observation_only取消估算准入，不近似、不放宽容差 |
| AUPRO | relation_reliability.official_metrics.official_aupro_adeval | 独立显式CUDA；ADEval1.1.0、200阈值、FPR≤0.3，地图/GT SHA先对照CPU |
| 新增选择机制 | neighbors.topk；method.witness；supports.Connections/candidate_gates | 每第三图独立k5、内部alpha/rho、支持定义惰性连接；非查询空间投票 |
| 新增编排 | prediction/evaluation/reporting/runner | 四分支、逐层/逐图/逐域续跑、完整对照与pending报告 |

## 冻结首轮规则

k_w=5、alpha∈[0.25,0.75]、rho≤0.5、至少一张第三图。没有长度阈值、额外epsilon、Source同源门控、空间投票、共享alpha、gap、OT、adapter或训练。原始1280维FP32支持完整使用，不压缩/采样。证明记录交换端点前先规范化，故接受结果不随query访问顺序变化。

GPU支持表采用FP64直接cdist而非norm/dot矩阵恒等式；这是**新增支持间检索的明确数值实现**，不是声称与历史FP32查询检索逐位相同。生产CPU路径是真实SciPy直接欧氏检索，测试不以不同的NumPy选择器冒充Torch链路。GPU块内stable argsort、块间(distance,row)合并使原行号并列规则明确。不同后端不可在同运行目录自动切换。

## 身份与信息条件

支持特征文件SHA来自原background capture，以及原crossdomain cache_status.supports。同时核对support_index、图片路径/图片SHA和grid；NPZ内校验协议、图片路径、grid、四层形状与dtype。NPZ没有自带图片SHA字段时，使用原capture记录与冻结支持manifest之间的图片SHA关联，而非编造该字段。

连接没有GT/标签参数。整图留一不访问heldout银行作为端点/佐证，缓存键包含excluded集合；heldout只通过该图到剩余7图的Top1作为正常待解释query。正式query的原八图支持保持不变。其身份信息与GT是在预测冻结后的CPU评价阶段才使用。

SQLite每条唯一连接包含端点行号、有限第三图候选ID、全部合格点和代表拒绝点的alpha/rho、原因计数；不是每个query重复保存向量。预测层还保存所用唯一证明集合摘要。支持邻居完成清单先验证旧hash再续跑；预测冻结清单封存NPZ和record SHA。修改旧缓存不会在续跑中重新登记。

## 当前输入与成本边界

准确本地预检在local_validation/。目录搜索和归档实际成员检查与“存在结果包”区分；完整支持特征缺失时prepare/predict阻断，不自动恢复。查询raw与已有CPU/像素父结果可以复用；支持间56个有向图对/域/层与连接几何是新增计算。无需Source bank实体、模型checkpoint或Q特征。

只读预检为全路径存在性、父元数据/校准身份和每域一张完整raw SHA/字段检查；其余raw在正式使用时逐图核验，不声称预检已经重新读取全部926份。完整支持检查只读NPZ头和文件SHA；加载某层才做完整有限性检查。

没有真实资源测量或运行时间承诺。两GPU按独立作业分配，未实现共享显存。没有估算容量前置准入，也没有分配失败自动换方法、降采样、改单卡或放宽门限。

## 尚未验证

原始24份完整支持特征尚未在本地验证到位。真实CUDA FP64 cdist、双卡调度、实际支持间近邻/接受率、全域预测/CPU像素与六项AUPRO均未执行。合成数据通过只支持实现路径的检查，不能当作方法收益或真实链路成功。旧报告与正式结果保持不变；本轮不修改窗口记录。

本地验收为34项合成测试通过、12份Python源文件内存编译通过，默认preflight返回`missing_inputs`。31个实际归档成员目录已检查，三张分别属于各域的raw示例通过SHA/字段/端点契约核查；原629份cross raw路径均存在。具体24份缺失支持的原历史SHA与路径已导出，无真实邻居表或正式预测产生。测试还实际触发并修正了输出父目录创建问题，并覆盖冻结审计表修改后续跑必须拒绝、整图留一残缺记录不能冒充完成。

## P2复核修复：CPU中间指标不能重新认可

原CPU续跑对`*_exact_pixels.json`只核对输入与地图身份，整域提交时才为该文件记录SHA；因此修改过的指标可在中断续跑时被重新封存。现由`evaluation.exact_pixel_result`在每个分支计算完成后立即写独立seal，续跑校验既有文件SHA及完整输入绑定，再由`validate_exact_pixels`检查AUROC/AP有限性、[0,1]范围、GT正/负/总像素数及有效状态。GT计数来自已冻结预测之后重建的同一完整mask，不改变GT映射或像素评价口径。

旧的未封存中间结果与孤立seal直接拒绝，保留现场；不根据当前文件补写可信seal。无完整CPU域状态也可复用已独立封存的合法分支结果。完整CPU状态继续封存结果与seal两者的SHA。新增测试使用少量合成像素执行真实原exact_binary，再模拟篡改（含合法范围内修改）、无封存旧文件、孤立seal、输入/地图/分支/像素数不符、非有限或越界数值、无效状态和两次提交之间中断；这些不是完整真实评价或CUDA验证。

修复后42项测试通过（10.070秒），12份Python源文件内存编译通过。本轮没有更改neighbors、witness、LOO、评分/地图内核或原exact_binary后端，没有改写正式结果、历史代码及窗口报告。

## 2026-10-22 用户授权的支持恢复和AutoDL正式入口

原缺失项是24份完整支持特征；926张原query的逐支持距离、索引和端点长度已存在，无需重跑query模型。新增`recovery.py`显式调用原宿主和原`_extract_four_layers`，不触及Source bank/MAD，不自动缺失补算。`inputs.Bundle(..., recovered_supports=...)`仅在新恢复manifest及封存链完整时接入新特征，其身份独立于历史SHA。

每文件记录原支持图片SHA/顺序/grid/历史特征SHA与当前文件SHA。恢复配置冻结原QUERY/PRECISION、checkpoint、宿主配置与源码。`verify-support`调用原`geometry_layer`，逐图逐层核对所有926张原SEGMENT距离和原endpoint_length，沿用距离atol1e-7/rtol1e-6；错误保留、不放宽。验证与文件seal绑定，不在续跑时重新认可修改。原特征SHA、历史capture和结果均不改。

这不是原NPZ字节等价或原八支持检索全等证明；没有原Q，无法独立核验重新提取银行到原Q的全部Top1。只将历史距离/索引与恢复端点几何的兼容性作为可执行接入证据，该限制写入manifest和README。正式query仍读取历史距离，不能把恢复描述为已经重跑了完整宿主查询验证。

新增`portable.py`按742个必要文件导出3.47GiB冻结CPU输入包，保留所有原相对布局、元数据和字节哈希，不含Source bank/checkpoint/图像/支持。流式打包时同步计算一次SHA；restore禁止路径越界、重复成员及不同内容覆盖。输入包不上传GitHub。

新增`autodl.py`是用户手动启动的完整流程，支持明确单/双GPU，30秒心跳、分阶段日志与退出码、成功或失败review打包、同目录续跑；仅完整指标/LOO/数值状态和打包均成功才可触发显式`--shutdown-on-success`。无自动SSH、无安装依赖、无容量估算准入、无失败后关机。新增14项轻量CPU测试通过；GPU提取、CUDA支持检索、多卡调度和真实评价仍未执行。

## 2026-10-23：显式历史地图/评分兼容政策

此前严格恢复要求endpoint_length和SEGMENT逐点均满足1e-7/1e-6，首个失败层即停止。用户真实CPU审计表明background全层完全一致，两个Kaggle来源域在全部层存在较小差异：融合图最大4.57e-5/1.70e-5，CH1/CURRENT最大均<2.4e-6；view有2张峰值位置变化。不能称定位完全一致，也未验证原Top1或WITNESS门控不变。

新增compatibility.py，仅显式historical-maps-v1使用。原strict仍默认，不改原geometry_layer/DIST_ATOL/DIST_RTOL。新数据接入要求完整地图最大差<=1e-4 Source MAD响应，各层对原四层等权融合的最大贡献差<=1e-4，CH1/CURRENT变化<=原1e-5；地图数值限值属于新工程政策，不冒充原历史严格检查。支持身份/顺序、QUERY/PRECISION/checkpoint/host config、原NPZ/record SHA继续严格验证。

旧提取目录只读；新的运行目录生成独立compatibility_inputs、逐域CPU审计+seal、兼容manifest+seal，并纳入新的运行身份。旧未封存审计不能直接认可，重新执行CPU几何/原地图/评分，未读取GT或查询标签。原几何有数学错误、任一图/分数/分层贡献超限、身份变更或文件/封存异常仍失败。兼容模式不加载模型，不提取支持或query，不构建Source，不拟合MAD。

原逐点距离失败保留在审计、manifest、正式status与报告中，不用numerical_failed_checks=0误称历史全通过；此字段范围明确为正式冻结预测/地图/分数核查。原历史定位图和指标继续按原缓存SHA核查后复用；恢复几何只用于兼容审计，正式U/S仍读取原距离。方法alpha/rho/k5、均匀缩距、Source MAD、评价器没有变化。

新增10项CPU合成测试覆盖真实小型原geometry与地图/分数接入、旧文件不改、不复制银行的外部来源加载、严格失败继续保留、分层抵消保护、图/分数超限/非有限/数量/单位错误、文件或seal篡改、旧未封存中间结果拒绝、协议不符、默认strict及复用模式不进入提取阶段。GPU启动用替身隔离，不代表真实支持检索或全域实验通过。
