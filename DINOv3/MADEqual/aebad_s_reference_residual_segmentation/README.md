# 真实 Target-normal 支持下的参照残差分割学习

本工具保持同一套方法实现，新增运行预算 `compact_v1`，作为新建实验默认值。原完整预算保留为显式 `full_v1`。本轮只完成本地代码、CPU测试与说明，不连接服务器、不提取真实特征、不启动训练或GPU评价。新compact的效果和CUDA稀疏检索尚未验证。

## 两种固定预算

执行策略现默认 `--audit-mode fast`（脚本`AUDIT_MODE=fast`）。这是受信任本地磁盘的执行选项，不改变compact/full方法配置、训练日程或冻结输入身份。

- 默认不重算已有特征、NN、RGB、checkpoint和评价结果的内容SHA；完整正常特征不再额外全数组扫描有限性。读NPY仍检查header形状/dtype、位置和必要索引。
- prepare续跑沿用当前冻结的权重SHA，有可信父身份时沿用父权重SHA，不再重读3GB权重只为校验；没有任何既有身份的首次新权重仍生成一次SHA。fast不声称识别已信任权重在磁盘上的内容变动，真实模型加载仍要读取权重。
- Exact仍返回原FP32距离/索引，默认取消每次检索后的CPU参照向量距离复算。
- 训练、校准、预测按实际需要读数据；预测完成性检查改为三份域清单和原926人口，不在预测后、CPU评价前和AUPRO前重复遍历全部预测数组。实际消费时检查该图的manifest seal与冻结域清单一致。
- AUPRO仍用原后端、原地图生成和GT resize，但默认不重复重算地图/GT内容SHA；检查几何和dtype。CPU评价首次生成的地图身份记录保留。
- 新产物首次写入仍生成原有身份，已有产物身份不重签。小JSON seal、配置/日程/排除规则、索引/形状、loss/梯度有限性、MAD非退化、指标范围及完成数量检查保留；小型代码与环境身份检查保留。没有容量准入门槛。
- `--audit-mode full`为显式内容审计：启用数组/地图SHA和CPU NN复算；同进程未变化的同一物理文件只hash一次，共用记忆记录，不再随`st.get`重新校验。默认fast不保证识别形状相同的磁盘内容改写，这是本地信任模式的明确取舍。

已停止的`2c721a6` compact可以在原run/cache续跑。只读登记其旧代码身份，允许本次执行优化；不修改旧inputs、catalog、resolved_config、job_plan，不重提取已接入的正常特征。执行方式另记在`execution_sessions/`。未知方法代码修改、profile/数据/提取配置变化仍拒绝。

之后手动续跑示例（先同步修订代码；本轮不连接或启动服务器）：

```bash
run=/home/ubun/YYK/runs/aebad_reference_residual_compact_v1_20261027_002907
session=residual_fast_resume_$(date +%Y%m%d_%H%M%S)
tmux new-session -d -s "$session" "AUDIT_MODE=fast bash '$run/launch.sh' '$run' >> '${run}_launcher.log' 2>&1"
tail -n 30 -F "${run}_launcher.log"
```

|配置|compact_v1（默认）|full_v1（显式）|
|---|---:|---:|
|训练种子|42|42、43、44|
|训练族|REF_ST|REF_ST、NOREF_ST、REF_SOURCE_QUERY|
|每张Source合成槽|2|8|
|每张Target合成槽|16|96|
|计划合成总槽|1294=455×2+24×16|5944=455×8+24×96|
|Source预训练|1个×1000更新|6个×2000更新|
|目标域训练|3个×4000更新|27个×12000更新|
|计划更新总数|13000|336000|
|集成|不请求|3个族各三成员z集成|
|主候选|REF_ST_SEED42|REF_ST_ENSEMBLE|
|定位图分支数|3|14|
|三域定位图/AUPRO逻辑作业|9|42|

compact定位图只有 `UNION_K8`、`SEGMENT_K8`、`REF_ST_SEED42`。`SEGMENT_SCORE_UNION_MAP`保持既定SEGMENT图像评分与UNION定位指标的组合，不生成新图，不增加AUPRO。关闭的控制、种子和集成为 `not_requested`，不会成为pending或完成条件。CH1为主图像评分，CURRENT为补充，二者共享像素评价。compact完整表为32行（含三域等权macro及固定组合），full为120行。

`protocol.resolve(profile)`是唯一运行配置入口，`protocol.job_plan(config)`派生实际任务；`inputs.config/plan`加载已冻结版本。新prepare分别封存 `resolved_config.json`、`job_plan.json`，并将两者seal纳入inputs身份。各阶段按同一清单选择模型/分支/更新数。`protocol.CONFIG`及旧SEEDS等名称仅保留作旧full记录和共同数学常量的兼容模板，运行不能靠修改这些全局量选择分支。静态 `compact_v1_plan.json`由同一解析器生成，供review，不是另一份执行配置。

## 保持的方法定义

- Source原521张按路径排序，index%8==0的66张只用于校准，其余455张训练/构库。三域仍为background 297（85正常/212异常）、illumination 265（67/198）、view 364（84/280），共926张，固定seed42 K8及顺序、支持排除均不变。
- 原DINOv3 ViT-H+/16完整视野/切窗/拼接、原预处理与提取精度不变。L7/L15/L23/L31、1280维FP32原始特征；正式Source每层100K、原亮度增强与原100轮距离采样；Exact FP32 Euclidean/1280一次。
- Source训练原图及派生均只查对折训练参照：稳定交替A/B为228/227张，每折按原采样器构成100K/layer。Source预训练模拟7/8张对折正常支持；目标阶段Source查询采用域K8。Target训练原图及派生使用正式Source与其余K7，正式测试及66张校准使用完整K8。
- 每层输入是 `(q-mu)/sigma`、`(q-s*)/sigma`、`(q-t*)/sigma`和Source距离MAD标准化的ds/dt/min。mu为正式原库通道均值，sigma为全通道中心RMS，无每patch L2归一化。NoRef只有显式full请求时存在，只读查询并置零其他通道，不访问NN或距离MAD。
- 原四层头，1×1到128通道、逐位置LayerNorm/GELU、两个dilation1/2局部残差块、1×1输出logit；无全图空间归一化。中心16×16、halo4，不改变卷积/插值依赖。
- batch4、正常/合成1:1；REF_ST/NOREF_ST目标阶段Source/Target各半。四层逐图正负平衡BCE均值＋融合logit同类BCE＋0.5 soft Dice；正常图Dice为0。AdamW 3e-4、wd1e-4、梯度裁剪1，5%warmup后cosine到1e-6；按本次1000/4000或2000/12000总更新数计算，不截取旧长日程中间模型冒充完成。
- 头训练BF16、损失FP32；头推理FP32并保持CuDNN启用状态；骨干精度保持原宿主配置，FP64只用于小规模统计/误差核查。无自定义Triton。
- 新logit仍用每域K8条件下未训练的66张Source独立校准patch/image MAD，不套旧距离MAD；常数/近零尺度仍失败。Source-only距离MAD与新logit校准分离。原融合、插值、CH1/CURRENT、GT映射、精确CPU像素排序和原ADEval1.1.0/200阈值/FPR≤0.3保持。

## compact合成选择与无有效版本的原图

合成算法版本独立于运行profile：`SYNTHESIS_VERSION`保留原 `reference_residual_segmentation_v1` RNG前缀；原生成种子、供体SHA排序、variant ID和item ID不变。compact从旧Source 0–7、Target 0–95选分层子集，不重编号。

Source每张1个Poisson＋1个texture/scratch，后者按稳定455图清单交替（228/227）。Poisson目标尺度按清单轮换，保留小/中/大覆盖。Target每张8 Poisson、4 texture、4 scratch，按原尺度分层，单张总尺度数为轮换的6/5/5。供体只来自训练Source或同域其他正常Target，排除原图、校准、测试和跨域Target。选择不读取GT或指标。

前景、Poisson/texture/scratch、尺寸范围、4次有限重试、实际RGB变化和soft标签规则均保留：Source PC1＋中心定向/Otsu/最大连通域，面积5%–85%，放置至少80%落在辅助前景；变化至少2/255，soft变化/.1截断并用面积覆盖映射。没有新增补生成或把无变化位置标为异常。

1294仅为计划数。`synthetic_type_scale_counts.csv`记录域×类型×尺度×实际状态，`synthetic_origin_summary.csv`逐图记录planned/valid/normal_only。某图没有有效合成时保留其正常版本；合成batch从对应人口的有效合成原图中均衡选原图，再选有效版本。正常batch仍覆盖全部原图，Source/Target与正常/合成的计划比例不变。若Source或某域整个所需人口无有效合成，prepare-training在构库前停止，保留说明，不硬造标签、不无限换种子补齐。

## 训练NN只检索消费的位置

新建compact和full均采用 `physical_tile_union_v1`：实际合成RGB仍先做一次完整原宿主前向，随后由已冻结标签确定tile/halo，将所有所需物理patch取去重并集，四层按同一物理位置查Exact。`source_nn/target_nn`显式保存positions，complete记录完整网格大小、NN位置与数量。Dataset通过物理ID到NN行的查表取值，拒绝缺位置/重复/越界/行数不符，不把稀疏数组当全网格。

正常图训练tile覆盖完整网格时照实全查，不宣称缩减。正式926张和66张校准仍使用完整网格。`training_query_resources.csv`记录每项完整patch数、实际NN数及复用来源，位置减少不当作已测加速比。

## 只读资产复用

通过可重复 `--recent-run`或原始 `raw_assets.json`指定来源；不扫描已丢失AutoDL。已停止的本工具full运行也可作为只读父资产，允许在全合成池尚未完成时导入已封存的选中项。

下表SHA字段在默认fast下用于匹配既有manifest中的声明，不重新扫描磁盘内容；只有显式full才检查内容。旧审计验证结果不能写成本轮fast重新验证通过。

|资产|相容判断与处理|
|---|---|
|正式原Source bank|原始类型、Source图像/提取身份、prepared source seal及数组SHA；链接原始L层，不读adapted bank|
|两折训练bank|相同训练原图/顺序/折、增强、采样器/预算/种子、提取设置及内容；训练步数/seed数量不影响它|
|完整正常/支持/校准特征|原始完整网格、图像身份、层/dtype/positions和内容；2048抽样不能替代|
|选中RGB/标签|原ID、供体、图像、固定生成规则/种子、实际前景及RGB/标签SHA；不改旧manifest|
|原始训练tile|RGB/提取/标签/几何依赖一致；直接链接。NN缺失时可从已封存原完整前向的tile恢复所需原始行并只补NN，重叠行必须逐位一致；不伪造未消费位置|
|NN|同时核对实际Source/Target内容、K7/K8/对折排除、位置/标签/几何；旧完整NN直接按物理位置取行，不能只凭run ID相同|
|原始Q|原完整q层与positions，不能从距离/索引恢复；父stream未保存Q时需再前向|
|UNION/SEGMENT|与本轮同库/MAD下距离复核，像素指标还须逐图map/GT SHA和完整名单一致；不相容则本轮计算|
|模型/优化器|不自动跨profile导入。当前目录的池、日程、精度、输入身份及完整checkpoint必须完全匹配；旧2000步日程的第1000步不是新1000步日程最终模型|

新产物保存 `origin/origin_seal`等只读引用，不给旧产物重签为新身份。新版依赖签名忽略与该资产无关的训练预算；5091600旧full只在已记录实现、固定依赖和实际内容通过时接受。缺少证明记录为unconfirmed/重复项，不用名称猜相容。详见 `reuse_audit.json`、`reuse/*.csv`及各manifest。

父距离MAD缺失时，用当前共同原Source与原66图校准；存在但封存损坏时仍拒绝。两个阶段不改父校准。新建compact绝不覆盖旧full。

## 阶段、默认命令与旧full

默认stage仍为只读 `preflight`，不导入Torch/初始化CUDA/生成结果目录；展示解析配置、实际任务、复用候选/新增/确需重复/无法确认项。只读候选存在不等于内容已通过校验。实际操作由显式prepare开始。

默认顺序：`prepare → synthesize → prepare-training → train-source → train-target → calibrate → predict → evaluate → evaluate-aupro → summarize`。`all`只执行当前profile清单，不自动追加消融、其他seed或集成。末尾review按允许类型打包，剪除大cache、scratch、NPY/PT/BIN，无关机或依赖安装。

新建compact（以下为之后人工使用的命令，本轮不执行，也未推送）：

```bash
cd /home/ubun/YYK/DINO-Reference-AD
PROFILE=compact_v1 GPU_INDEX=1 CPU_WORKERS=8 \
RECENT_RUN=/home/ubun/YYK/runs/aebad_reference_residual_20261026_204836 \
SECOND_RECENT_RUN=/home/ubun/YYK/runs/aebad_correspondence_20261025_201937 \
bash DINOv3/MADEqual/aebad_s_reference_residual_segmentation/scripts/run_workstation.sh preflight
```

之后人工启动compact并固定run/cache以便续跑：

```bash
work=/home/ubun/YYK
repo=$work/DINO-Reference-AD
tag=aebad_reference_residual_compact_v1_$(date +%Y%m%d_%H%M%S)
run=$work/runs/$tag
cache=$work/storage/reference_residual_cache/$tag
mkdir -p "$work/runs"
tmux new-session -d -s "$tag" "cd '$repo' && env PROFILE=compact_v1 GPU_INDEX=1 CPU_WORKERS=8 REFERENCE_CACHE_GIB=6 QUERY_MODE=stream RUN_ROOT='$run' CACHE_ROOT='$cache' RECENT_RUN='$work/runs/aebad_reference_residual_20261026_204836' SECOND_RECENT_RUN='$work/runs/aebad_correspondence_20261025_201937' bash DINOv3/MADEqual/aebad_s_reference_residual_segmentation/scripts/run_workstation.sh all > '${run}_launcher.log' 2>&1"
printf 'SESSION=%s\nRUN=%s\nCACHE=%s\nLOG=%s_launcher.log\n' "$tag" "$run" "$cache" "$run"
```

新建full必须显式 `PROFILE=full_v1`并使用新目录。继续现有5091600旧full时也必须显式full，保持旧run/cache和父来源，绝不以compact启动旧目录：

```bash
cd /home/ubun/YYK/DINO-Reference-AD
PROFILE=full_v1 GPU_INDEX=1 CPU_WORKERS=8 REFERENCE_CACHE_GIB=6 \
RUN_ROOT=/home/ubun/YYK/runs/aebad_reference_residual_20261026_204836 \
CACHE_ROOT=/home/ubun/YYK/storage/reference_residual_cache/aebad_reference_residual_20261026_204836 \
RECENT_RUN=/home/ubun/YYK/runs/aebad_correspondence_20261025_201937 \
bash DINOv3/MADEqual/aebad_s_reference_residual_segmentation/scripts/run_workstation.sh all
```

旧full无profile字段时，已封存旧CONFIG派生只读full视图，保持旧完整NN布局，不在旧目录写入新版resolved_config/job_plan。只允许已登记5091600旧实现或未变的依赖；未知旧实现、输入/配置冲突拒绝。不是任意代码版本的自动迁移。新目录都必须冻结新版配置/清单，不能切换profile续用原模型。上面旧full例子仍会执行原大预算，不用于compact验证。

## 执行优化、smoke与资源

保留有界原始参照GPU LRU（默认4GiB，`REFERENCE_CACHE_GIB`可显式调整）、同查询层共享上传、CPU预取、同尺寸tile合批。合批不padding，不改逐图插值与损失权重；浮点归约顺序不同，不承诺旧执行方式训练权重逐位相同。新执行的checkpoint仍封存优化器、RNG、scheduler/update并支持续跑。没有自动容量准入、自动降级或第二张卡。

`smoke`三种模式均只测请求路径：constructed使用固定构造输入重复，隔离计算/传输，不再每步生成大随机输入；assets增加真实父完整检索及3张Source正常前向；prepared使用真实Dataset和预取。compact prepared只有Source REF与三个域REF_ST四种短作业。报告列出profile任务数/updates/计划槽位/926人口，未请求控制及无集成。临时checkpoint写读在私有scratch，smoke不提交正式训练或评价状态。本轮只测试CPU构造输入，不执行服务器smoke。

耗时按构库/补特征、合成/训练缓存、头训练、校准/预测、CPU评价和AUPRO分项。训练来自实际任务清单，已经提交的工作不计待执行；有效合成/稀疏几何未测不把完整K8时间按旧5944池硬套。构造输入只报告计算/传输下界，真实输入路径吞吐另测。未知项保持null/pending，不填0，不承诺几小时必完，也不把先前full的25小时当compact估计。

大缓存使用机械盘，代码/日志/模型和单层采样/精确评价scratch使用指定SSD私有路径。正常完整网格逻辑大小仍约145GiB，能链接则不新增；两折bank约3.8GiB，候选spool逐折约61GiB、SSD暂存每次一层约15GiB（13924网格估计）。compact不保存全部合成全网格，tile/RGB/标签/NN大小按实际记录；不能保证总量只需几十GB。stream不默认保存926张Q。

## 输出与结论边界

主要输出：resolved_config/job_plan、prepared/raw_assets、synthetic_manifest及类型/尺度/normal-only清单、training_manifest及NN位置资源、models、独立calibration、冻结predictions、metrics/metric_comparisons、training_repeat_summary、实例margin、costs、reuse审计、objective_report和status。像素分支只算一次，CH1/CURRENT引用同一结果；兼容基线指标与新增计算分别标明来源。

compact只检验单种子、小预算主方法在三域完整评价上是否值得继续投入。单种子标准差为NA；不写跨种子稳定。控制未请求则不据此回答参照/Target查询的独立作用。效果差只报告当前预算，不否定充分训练的全部可能；不看测试GT延长训练、调阈值/权重或挑checkpoint。GT仍只在全部正式预测冻结后用于评价。

## 本地验证

2026-10-27收尾验证：73项CPU测试全部通过（171.423秒），23份Python文件编译通过，默认脚本`bash -n`通过。默认预检正确列出4个训练任务、13000 updates、1294计划槽位及9个定位评价任务；未导入Torch、未创建实验目录。当前本地预检仍缺DINOv3权重、官方源码及ADEval；父资产仅有本地元数据，未验证服务器实际缓存。另对登记旧版本的Poisson、texture、scratch各一例执行原生成函数比较，RGB/标签及诊断逐位一致；这只验证固定构造输入下的生成兼容性。

```bash
python -B -m unittest DINOv3.tests.test_aebad_s_reference_residual_segmentation DINOv3.tests.test_reference_residual_execution DINOv3.tests.test_reference_residual_compact DINOv3.tests.test_reference_residual_fast_reads -v
```

范围和实测记录见 `implementation_audit.md`、`validation_results.json`。包含原full回归、compact三分支构造CPU接入、真实CPU Exact/梯度/续跑/只读兼容导入/稀疏位置映射；RGB宿主及训练模型来源在接入夹具中替代，评价人口缩为每域2张，不等于真实926图。AUPRO完成状态测试使用明确的报告夹具，不冒充实际后端运行。真实CUDA/BF16、稀疏GPU数值、大缓存复用及compact完整训练/评价均待review后的人工验证。本轮到此停止。

fast执行修订：87项CPU回归通过（198.424秒），另4项最终续跑专项复核通过（0.806秒）；24份Python编译及脚本语法检查通过。新增测试直接禁止已有数组读路径调用SHA、禁止fast NN调用CPU norm，检查fast/full CPU Exact结果逐位相同、完整compact CPU接入、显式full检测内容改写及只读原目录续跑。真实服务器提速、CUDA及ADEval没有在本地执行。
