# 参照残差分割学习：实现审计

## 2026-10-26 compact_v1 修订（当前默认）

本轮仅本地实现、CPU测试、文档和只读预检；未连接服务器、未安装依赖、未初始化CUDA、未启动真实前向/训练/评价，也未推送Git。下方原full审计保留历史背景，其中33模型/120行/42 AUPRO只适用于显式full_v1，不是当前默认。

### 唯一运行配置与任务

`protocol.resolve → inputs.freeze → resolved_config.json/job_plan.json → inputs.config/plan`。两个配置/清单seal进入inputs；拒绝固定profile内容冲突，任何阶段都不靠模块导入时的SEEDS/FAMILIES/BRANCHES选择运行。保留旧数学常量只是兼容旧full封装。compact默认1个Source REF预训练1000步、三域REF_ST各4000步、共4任务13000步；seed42；无控制/集成；主候选REF_ST_SEED42；三图×三域9个逻辑AUPRO，兼容基线实际复用数另报。固定组合不新增图。32行含CH1/CURRENT、macro及固定组合；单种子sample_std为NA，关闭分支not_requested不阻断完成。

新建full显式选择，原6+27任务/三种子/集成保持。旧5091600full目录只有显式full才能打开；旧inputs/config/代码指纹精确识别后派生只读完整NN视图，不修改原身份/配置文件。不识别的旧代码或profile冲突拒绝。旧长日程中间checkpoint不能当作compact最终模型，当前目录恢复仍需池/日程/精度/输入和优化器/RNG一致。

### 合成与采样变化的边界

改变的是预算/固定子集，不改RGB生成算法。`SYNTHESIS_VERSION`沿用旧RNG前缀，运行profile另有版本。Source从0–7选一个Poisson及一个texture/scratch，按稳定清单交替228/227；Poisson尺度轮换。Target从0–95按8/4/4选16个，尺度6/5/5轮换。原variant、item、供体ID不重编号，生成种子/规则/标签定义保持。计划1294，实际valid/invalid及类型/尺度、normal_only按事实输出，未做隐藏补生成。

无有效合成的个别原图保留全部正常监督；合成批从对应人口的有效原图均衡抽取，再选有效版本，保留Source/Target和正常/合成计划比例。任一所需人口整个无有效合成时，在准备两折库前明确失败。全有效时保留原full抽样随机调用和定义；full过去会被单原图无正样本阻断，现按本轮要求放宽为人口级条件，这项行为调整已明确列出。

### 原始资产与只读兼容

`reuse.py`按实际依赖判断，不使用预算identity一票否决：原始类型/完整网格、提取身份、原图/供体、生成规则/前景/RGB/标签、bank内容、几何、排除policy和NN物理行。新producer保存依赖签名；已登记5091600 producer通过旧代码指纹+固定依赖+内容记录证明。链接/新引用保留origin及原seal；Source/MAD/旧目录不改写。父MAD缺失仍可本轮校准，损坏父MAD拒绝。

支持正式Source、完整normal/K8/calibration、两折bank、选中RGB/标签、tile及NN、原始Q和严格一致的基线地图/评价。未完成大池的已封存per-item RGB可独立导入。NN不全时，可以从独立已封存的原完整前向tile恢复消费位置的原始行；重叠必须逐位相同，不填造未消费位置，再按当前合法bank查NN。旧完整NN可按物理位置取行。缺依赖记录unconfirmed并准备缺项；损坏已存在记录拒绝，不重签。

### 稀疏训练检索

`tiles.physical_union/rows_for → training_data.prepare_one → source_nn/target_nn.positions → Dataset.load`。新RGB仍完整原宿主前向后裁tile，或复用已证明的旧完整前向tile；NN只查tile与halo去重并集，四层同位置。meta区分完整网格与实际NN数，显式映射物理ID到NN行；源/目标数组各自映射，不按全网格长度索引稀疏向量。正常图覆盖全网格时全查；正式926及66校准保持完整网格。稀疏查询改变GEMM行形状，真实CUDA仍待按原距离容差复核，不能由CPU一致性宣称GPU已验证。

### 成本与smoke

任务数/更新/计划槽位/分支/人口从同一plan派生，去除旧大配置硬编码。constructed固定输入只测计算/传输下界，不再把每步随机数生成等待混入；assets是实际父检索/前向，prepared按请求阶段/族读取真实Dataset。compact不展开NOREF、REF_SOURCE_QUERY、seed43/44或集成。按六阶段输出，当前有效池/实际稀疏几何、冷读/构库/评价未测时保留unknown，总时长不编造；已确认提交的工作从待办扣除。

### 本轮新增检查（无容量准入）

- 固定profile与resolved_config/job_plan封存、真实请求的模型/任务/完成行数；旧full默认compact拒绝，不自动迁移。
- 合成原ID/供体/规则、实际前景/RGB/标签、原始类型与完整网格；人口级有效合成检查，保留normal-only原图。
- 只读provider身份、旧producer代码依赖、数组SHA、折/排除/引用bank内容、tile重叠一致；原产物封存不变。
- NN位置必须为整数、唯一、范围合法并覆盖消费位置；数组行数和冻结物理集合一致；不放宽数值容差。
- 模型日程/实际池/初始化/精度/优化器/RNG检查；不把旧1000/2000更新状态当新1000更新完成。

沿用原路径隔离、scratch私有清理、有限性、预测冻结后GT、像素中间结果封存及AUPRO身份检查。无RAM/VRAM/磁盘容量前置门槛、自动fallback、参数搜索、依赖安装、自定义Triton、服务器连接或关机。

本轮实测结果见validation_results.json。CPU接入使用构造完整特征和小头，RGB宿主/训练模型来源有明确替身；AUPRO状态测试只验证报告逻辑，不执行替代AUPRO。实际服务器旧资产导入、CUDA稀疏NN、BF16、小池有效性、完整4模型训练及9逻辑AUPRO尚待review后人工验证。旧full smoke的用户日志不能当作新compact已通过。

### 2026-10-27 最终验证记录

- 三份测试文件合并执行：73/73通过，171.423秒，见`.tmp/reference_residual_tests/compact_final_validation.log`；含原full回归、compact CPU校准/预测/精确评价接入、稀疏NN与全量取行一致、正常监督保留、复用篡改识别、日程续跑及完成状态。
- 23份Python文件编译通过；默认启动脚本`bash -n`通过；交付`compact_v1_plan.json`与实际resolver输出一致。
- 默认只读预检：4训练任务/13000 updates/1294槽位/9定位任务；Torch未导入，实验目录未创建。缺权重、官方DINOv3源码及ADEval，未补装或运行真实资产。
- 登记旧生成源码与新实现对固定构造Poisson/texture/scratch各一例的RGB、标签及诊断逐位一致。不是1294张真实池有效性或真实模型链路验证。
- 新增检查限于配置/产物身份、合法映射、数据人口和请求完成性；完整列表见上方“本轮新增检查”。未增加容量准入门槛。

实现与本地验证已具备独立review条件；未发布、未执行服务器smoke、未启动compact正式实验。

---

## 原 full_v1 实现审计与历史验证记录


日期：2026-10-26。范围：独立本地实现、CPU构造数据验证、输入预检、服务器手动入口。没有连接服务器、运行真实DINO/构库/训练/CUDA/ADEval，未安装依赖。Git发布仅包含代码、测试、说明和固定身份元数据。历史方法和报告未修改。

## 方法边界及论文依据

已阅读本地`实验报告/文献精读/候选一_参照残差与合成异常_20261026/`的paper、source_map、translation_notes，以及窗口记录最新对应学习收束和现有实际宿主调用链。

|依据|实际借鉴|本工具的改编/不继承部分|
|---|---|---|
|[ResAD](https://arxiv.org/abs/2410.20047)|正常参照残差有助于区分正常变化与损伤的假设|本轮无辅助真实异常；不复制判别目标。当前作者仓库为改进代码，未称2024官方实现|
|[MemSeg](https://arxiv.org/abs/2205.00908)|正常memory与合成监督结合|使用真实域K8和独立Source/Target残差，不搬用其完整架构。常见TooTouch实现为非官方，本轮未复制|
|[NSA](https://arxiv.org/abs/2109.15222)|正常内容局部移植、Poisson融合|正常供体、三类固定合成池、实际变化软标签均为本项目工程定义，不宣称官方复现|
|[InCTRL](https://arxiv.org/abs/2403.06495)|利用少量正常参照做残差条件判别的启发|不引入其辅助真实异常监督或复制其原协议|
|[AnomalyDINO](https://arxiv.org/abs/2405.14529)|冻结DINO正常参照框架|继承本工程原宿主与真实扰动协议，不将其标准场景结果当成本方案证据|

未下载或引入上述项目的新代码、权重、纹理库和训练环境。网络、三种子日程、两折排除、三类合成细节与独立校准是本工具首版定义。不是论文创新或有效性结论。

上轮正常对应方法训练Adapter吸引正常特征，检索前变换q与memory；本轮**不更新表示、不变换任何memory**，训练新的局部logit头。仅复用其读取/Exact/评分/评价/封存checkpoint基础设施。导入旧模块的定义不等于构造Adapter；新调用链没有Adapter、旧paired_loss、matching图、transformed目录或旧方法重新训练。

## 实际调用链

|模块|入口/作用|关键冻结点|
|---|---|---|
|`inputs.py`|`recent/raw_manifest/preflight/freeze/policy`|真实correspondence v1、迁移后WITNESS/fixed-graph与显式原始资产schema；Source排序/index/mod8；frozen_origins固定Source与background图像哈希，illumination/view继承原frozen_catalog；preflight无Torch|
|`runtime.py`|`prepare/import_normal/existing_query`|调用原`host/extraction`，只取原始Source、完整normal/Q；2048抽样、距离/索引不冒充完整Q|
|`retrieval.py`|`build_source_folds`|实际`augmented_preprocessing → patch_exec → _official_subsample_values_with_indices(iterations=100)`；normalize=False底层规则，100K/layer，保存candidate/origin/patch|
|`retrieval.py`|`sampling_matrix`|HDD顺序spool，单层私有SSD副本，原值/顺序不变；不改变RNG或采样逻辑|
|`retrieval.py`|`exact_with_audit/target_nearest`|原`_exact_top1_chunked`，FP32全库Exact，逐支持操作及manifest并列次序保持；Euclidean/1280一次|
|`execution.py/retrieval.py`|`ResidentExact/References.parts/gather`|seal+layer有界张量LRU；同查询层一次上传；训练按索引取各支持行，不拼接整套K8；原cdist和审核保留|
|`synthesis.py/runtime.py`|`foreground/manifest/generate/synthesize`|仅455 Source和各域8支持，固定5944槽、有限4次重试，实际RGB/变化/软标签与无效原因封存|
|`training_data.py`|`prepare_one/prepare_training/Dataset`|每个有效合成整图实际前向一次；物理位置映射到带halo的tile；保存NN索引，动态取原始参照行，不落完整距离矩阵或重复残差|
|`model.py/tiles.py`|`inputs/Head/loss/interpolate_center`|128通道、dilation1/2、逐位置LayerNorm；halo4；原输出窗口soft标签监督，逐图正负归一化|
|`training.py`|`job/train`|Source6+目标27个最终头；固定步数/种子与AdamW；原子checkpoint封存optimizer/scheduler/RNG/update|
|`training.py`|`prepare_batch/backward_batch`|只合并输入形状相同的tile；每图插值和loss独立，原每图权重；不padding、不改变抽样或更新数|
|`smoke.py/runner.py`|`smoke/head_benchmark/assets_benchmark/estimate`|构造、真实父资产、完整训练缓存三种显式模式；独立记录，绝不提交正式训练/评价完成；未测阶段保留pending|
|`calibration.py`|`calibrate/fit_logits/mean_member_z`|基线Source-only距离MAD与新域条件raw-logit MAD分离；ensemble只平均member z，再拟合image MAD|
|`inference.py/prediction.py`|`context/head_inputs/logits/baseline/predict/verify`|正式Source+完整K8，所有头同Q/NN；14个分支全部冻结后才进入GT评价|
|`evaluation.py`|`sample_work/evaluate_domain/aupro`|实际旧CPU精确像素、原GT/实例/插值、ADEval1.1.0/200阈值/.3；不修改旧全局BRANCHES，不伪装分支名|
|`reporting.py`|`summarize/comparisons/training_completion`|三域等权、单种子/实际集成分开、NA与pending保留；完成状态要求完整33训练和42逻辑AUPRO记录|

旧通用函数需要分支上下文时，通过独立`FunctionType`全局绑定复用；原模块字典和分支列表不改写。新模型分支按真实名称进入通用map/实例/metric核查，不冒充历史分支。已用完整CPU接入夹具验证这些绑定。

## 自排除和信息隔离

1. 正式原库无法通过“只删除一个Top1”可靠排除同源增强，因此新增A/B训练参照，每折原100K预算，**不再从折内额外删除index%8项**。455已经排除了全部66张校准图。
2. 固定交替A/B（228/227）；对折库涵盖的原图列表、原source_index、候选区间/增强网格与输出patch行可追溯。原图所有合成/增强继承同一折。
3. Source预训练模拟支持由对折正常图固定选7或8张；目标阶段Source查询用本域K8，Target查询及所有派生用K7。供体也只能来自455 Source或同域其他Target支持。
4. NoRef独立数据分支不打开NN文件或参照行，不读取距离MAD，仅保留相同Source向量均值/RMS标准化的查询。输入其余通道为零，不宣称有效输入维度完全相同。
5. Source/Target训练查询混合、原图均衡和正常/合成1:1由每update的稳定PCG选择决定，与遍历、预取或训练族无关。Source-query控制共享相应REF预训练，目标更新数相同，仍接受K8条件。
6. 训练函数没有GT读取入口。原冻结目录建立时可读取GT文件字节校验身份，但不解码GT或把测试信息用于合成/训练。预测中只写样本元数据；926全部提交后，评价入口才允许GT解码。

mu/RMS按用户要求来自完整原Source库，包含训练正常分布；这不是训练NN自身匹配。Source查询的实际NN仍只在对折库上搜索。

## 数值与监督核查

- 原Exact结果不替换成直接残差算法。保留索引，记录直接原始行`norm(q-ref)/1280`的差异；严格1e-7+1e-6观察计数与预声明FP32平方误差界分别记录。超过解析界或非有限结果才失败，不悄悄扩大历史容差。
- 骨干采用原提取精度，FP32原始向量/残差/距离。头训练BF16，loss FP32；头自身关闭TF32，同时显式保留CuDNN enabled/benchmark/benchmark_limit/deterministic，退出后恢复骨干配置，避免`cudnn.flags`默认enabled=False造成退化。没有自定义Triton。
- Point-channel LayerNorm没有整幅空间统计。卷积半径3，加一个双线性插值邻居圈，halo=4。中心像素窗口用全局半像素坐标确定，原图边界采用与旧插值一致的border处理。
- 训练标签用真实RGB变化与声明区域交集，精确面积覆盖重采样；小划痕保留非零质量。Poisson提交区域外恢复原RGB，避免OpenCV边界ROI位移留下未标注变化。
- 小型8px Poisson槽可能没有有效变化，测试未强迫它成功；生产保留原参数并记录无效。operator复现测试使用有效较大局部，同时另测零变化/无前景失败。有效合成比例可能偏离名义比例，必须报告，不隐藏无效槽。
- 每原图都要保留正常监督。若整张原图没有任何有效合成槽，准备阶段明确失败并提示检查前景与预览；不静默删图/换供体协议/放开全背景。
- 新raw logits独立MAD，零/近常数输出失败，不用sigmoid饱和分数或旧距离MAD。相应阈值预声明于CONFIG。集成patch=None，不拟合第二次patch MAD。

## 资产复用、重复计算与成本

本地最新correspondence review包的真实inputs/catalog/prepared schema已读取；它是stream模式，包中没有原始向量实体。只确认元数据与人口，**没有假称服务器实体已核验或实际已复用**。

可复用候选：该运行原始Source bank、24 Target完整特征、66 Source完整校准特征及严格相容的UNION/SEGMENT结果。原455 Source每图2048行不足，需实际完整特征；若有显式兼容全网格则不重复。原stream查询没有Q时，926正式查询还需真实原宿主前向。

correspondence提取兼容性核对原runtime/关键宿主实现、原图/支持清单、checkpoint、宿主配置、外部骨干代码及版本；不是只比较缓存名字。legacy按原实际提取链核验。显式raw manifest还需原始类型与payload seal。原身份不改写，新链接目录有自己的身份和原seal来源。

同一原库的旧Source-only距离校准可复用；原库不相容时按原66图重建**共同基线**校准。新网络输出始终另行校准。复用父UNION/SEGMENT z前同时核查本轮两套原始距离，不能只验证UNION而漏过SEGMENT近邻/端点变化。在正式基线指标复用前逐图比较map/GT SHA、完整名单及父结果封存，不能因为AUROC接近而视为等价。

本轮 review 修正 `runtime.prepare`：同一 correspondence 原库可复用、但父 `calibration/source/UNION_K8.json` 缺失时，不再无条件读取父 MAD；等完整正常特征准备后进入已有 `fit_input_mad`，用当前共同原库和原66张 Source 校准图生成本轮距离校准。父记录存在时仍执行 `sealed_json`，校验失败、不可读或无效链接不回退。仅成功复用父校准时创建 `baseline_parent.json`；不更改父库、父校准或已封存的本轮校准。没有更换校准公式、数据划分或方法设置。

新计算：两折增强/原采样参照；5944合成槽的实际RGB前向和共享NN/tile；6+27头训练；域条件新logit校准；正式头推理；最多36个新AUPRO作业及必要基线作业。没有运行旧失败方法。

最大候选spool逐折约61GiB，SSD只暂存一层约15GiB（13924网格估计），大小由`source_fold_resources.csv`按真实候选数记录。若正式原库确实缺失，其原上游builder仍以HDD spool重建，不冒称已改写旧正式构库器。全部合成全网格留存约1.67TiB，本工具只保存选定带halo的合成tiles；545正常完整网格逻辑约145GiB。训练NN、RGB/变化标签/输出软标签/预测另计，因此实际缓存可能数百GB或更多。

通用原始查询/索引供全部模型共享。训练阶段共用按需验证的mmap会话及有限LRU，CPU下一batch预取与GPU重叠；训练日志每100更新一次，checkpoint每200。每步有限性与梯度合并一次判断，不逐分项loss同步。未测真实GPU吞吐，不给预填时长或加速比。

已提交prepare/synthesize/training manifest续跑优先复用，只查身份/代码及小manifest；大payload在真正消费时按会话验证，不每个seed/patch重新扫描。分块未提交部分可重算；成功更新不重做。冻结预测后的CPU/AUPRO无需模型、bank或骨干实体。实际存储逻辑大小、硬链接与symlink区别由`storage_usage.csv`输出，不将逻辑共享大小当作新增磁盘。

## 全部检查及失败行为（按用户要求列明）

|检查|执行时间|失败处理|
|---|---|---|
|必需输入与包版本存在、ADEval1.1.0|只读preflight/对应显式阶段|列缺项，不安装、不补模型任务|
|原521排序/mod8、三域926名单、K8身份顺序与排除|prepare冻结、输入/预测核查|失败保留现场，不凑数/重选支持|
|原始/抽样/适配特征类型，网格/positions/dtype/有限性|导入或使用时|拒绝不相容资产；缺少完整原始Q才新增原前向|
|输入/代码/配置/模型/校准/NN参照seal与payload SHA|提交、续跑、首次消费|拒绝损坏，不给旧内容重新改名认证；会话内不重复大文件哈希|
|同一原库的可选父距离MAD|prepare选择原库后|缺失则在完整正常特征就绪后执行本轮原66图校准；存在但校验/读取失败仍拒绝，续跑保留已封存本轮状态|
|Source对折和Target留一policy、正常供体范围|生成/准备/训练读入|拒绝校准/test/跨域/同源进入训练参照|
|无前景/无有效RGB变化、无有效合成原图|有限重试与准备训练|逐槽无效记录；整原图缺正槽明确阻断，不无界重试|
|Exact对应原始行与距离、梯度/输出/校准有限性|相应计算阶段|数值失败保留checkpoint/日志，不换算法或极小分母救场|
|全部926预测提交、同缓存评分、map与GT身份|GT评价前及逐图重建|人口未齐不评价；复现失败不改容差|
|CPU精确像素中间结果独立封存、范围与计数|续跑像素评价|拒绝未封存/修改/越界结果，不重新给它记录“成功SHA”|
|父资产/输入、cache、评价和采样scratch隔离|规范化与直接评价入口|拒绝重叠；包括从冻结provider自动恢复的父路径，避免续跑漏传recent参数绕过|
|私有临时目录/独占文件创建|候选暂存、像素scratch|仅删除本轮创建文件，分配失败也不glob删除父文件|
|单卡、最多8 CPU workers|参数解析|不自动选择/占用其他GPU|
|驻留参照键/布局、tile原始行索引|复用张量与读取tile|身份不符或索引越界报错；不换库、不自动降低精度；缓存预算不是容量门槛|
|合批对照梯度/loss、Exact原路径数值、临时checkpoint写读|显式smoke|记录实际误差；超声明工程界报错；不修改历史距离/地图容差；scratch只清理私有临时文件|
|33最终模型、完整CPU/42逻辑AUPRO记录|summarize|任何缺项保留pending，不填0，不写五项完成|
|review允许类型与目录剪枝|pack|排除配置cache、两类scratch、NPY/BIN/PT，轻量打包，保留大缓存原件|

**没有RAM/VRAM/磁盘容量前置门槛、没有自动降级、超时切算法、参数搜索、自动安装、关机或服务器连接。** 路径保护与数据/数学身份检查不是容量准入判断。真实分配/OOM/I/O错误正常上报并保留现场。

## 验证记录与未验证项

最终本地结果见`validation_results.json`。测试目录均在G盘；不修改现有环境。旧CH1测试会检查进程未导入Torch，因此它与真实Torch CPU测试在**不同进程**运行，不能合并后把已加载Torch的共享进程误判为实现回归。

实测：新增45项全部通过（108.628s）；旧correspondence回归22通过、1因Windows目录软链接权限跳过（23项，22.862s）；旧CH1回归15项全部通过（5.935s）。最后元数据/合成专项复核9项通过。18份Python编译和手动启动脚本`bash -n`通过。CuDNN标志保留检查未初始化CUDA，不等于验证了CUDA卷积性能。

父 MAD 修复专项4项通过（0.694s）：缺失父状态时真实 `prepare` 导入同一 Source bundle 并调用本轮校准；已封存本轮校准续跑不重算；完整父 patch/image 状态按原 seal 复用；篡改父记录仍报错且不调用校准。使用三行1280维构造 Source 库，目录/导入/封存/准备分支执行生产实现；目录准入、空图像供应及校准计算为隔离替身，不代表原66张真实校准或DINO链路已执行。验证父文件字节不变、没有伪造父复用记录。

修复后完整模块49项全部通过（111.260s），包含原45项及新增4项；18份Python重新编译通过。没有重新执行此前通过的旧模块回归，也没有启动真实实验。

执行优化后，本模块及当时新增执行测试共55项通过（111.454s）；随后执行专项7项通过，覆盖驻留预算/淘汰/0预算、原Exact值与索引及并列、无整库拼接的行映射、四种训练条件的实际准备路径、原图policy记忆化、合批梯度及隔离CPU smoke。合计56项不同测试（有重复运行）；原49项全部包含。21份Python编译及启动脚本`bash -n`通过。实际GPU smoke按用户要求交由服务器手动执行，本地只探测设备，不运行CUDA计算。

优化属于执行变化：Exact依旧原FP32全库路径；参照常驻不改变输入值。头合批保留数学损失与每图权重，但改变浮点归约次序，不宣称旧训练权重逐位相同。优化版连续/恢复仍通过原逐位权重检查。旧执行代码的冻结运行不能混入本版checkpoint；重新prepare新正式目录，原始资产可按现有协议复用。

服务器smoke将测完整Source/K8检索与前三张Source训练正常图原宿主前向；头构造输入测速不包含真实训练I/O。缓存就绪后的prepared模式使用真实Dataset及CPU预取，分别短测11种阶段/分支并按6+27模型外推。合批工程阈值FP32相对梯度L2 1e-4/loss绝对1e-5，BF16为0.03/0.01，仅用于此执行方式的梯度对照，不用于放宽既有复现。实际整轮时长、真实BF16和大缓存首读仍待服务器记录；本地小矩阵不外推4090整轮时间。新增检查详列上表；没有增加任何容量准入、自动fallback、Triton或关机。

只读本地预检：识别了真实父包correspondence元数据，Source/background/全部GT身份核对无错误；该包是stream且本地未有完整raw实体。当前CPU环境缺正式checkpoint、DINOv3源码根和ADEval，状态为blocked，Torch未导入、实验目录未创建。这不表示服务器也缺这些资产；服务器没有连接或核查。

实际CPU验证包括：Source原图/派生排除和mod8；Target K7与域隔离；固定池与正常供体；实际Poisson/纹理/划痕；面积覆盖和零变化；1280D Exact及残差单位；物理重排/halo/整图中心；真实小头前向反向；连续/中断恢复权重逐位相同；目标两个REF族共用Source初始化；NoRef毒化字段/文件不可见；零尺度拒绝；z集成；42逻辑AUPRO未齐保持pending；未冻结人口禁止GT；scratch失败保留哨兵；中间指标篡改；SSD工作副本与候选字节一致；打包排除自定义cache。

新CPU接入夹具使用真实1280D向量、原Exact、66张**构造**Source校准特征、小尺寸4层头、全14分支校准/预测/实例/CPU排序/续跑。只有RGB宿主和已训练头来源被替身替换，评价人口在夹具内缩成每域2张；未把它称为真实926图、真实训练或DINO复现。AUPRO没有用替代指标填充。

仍待服务器人工验证：原始资产实体和提取兼容性；真实前景/合成预览、无效槽分布；原增强两折100K构库及真实选中索引；5944槽有效RGB前向/NN/tile吞吐与空间；CUDA/BF16与FP32头误差；完整33模型训练；原66域条件校准稳定性；926预测和正式CPU/ADEval评价；真实RAM/VRAM、SSD/HDD和耗时。模型效果、参照信息价值、Target训练作用与三种子波动均未得结论。

交付可供独立代码review；不是已通过真实实验。到此停止，不自动执行服务器任务或继续搜索公式。
