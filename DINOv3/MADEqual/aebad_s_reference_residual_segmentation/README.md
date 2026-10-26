# 真实 Target-normal 支持下的参照残差分割学习

这是待验证的独立候选，借鉴参照残差、正常内容局部合成和少样本正常参照机制。不是 ResAD、MemSeg、NSA 或 InCTRL 的官方复现。当前交付限于本地实现、小型 CPU 验证与手动入口；尚无真实训练或新检测结果。

## 冻结配置与主要对照

`protocol.py:CONFIG` 是第一版固定配置，CLI 不提供训练步数、损失权重或网络结构搜索。原 DINOv3 ViT-H+/16、完整视野切窗/拼接和提取精度不变。四层 L7/L15/L23/L31、1280维 FP32 特征，Exact FP32 Euclidean/1280 **一次**。正式 Source 每层100K、完整 K8 Target、原 Source 距离 MAD、原插值和 CH1 基线不变。

原521张 Source 按原路径排序，index%8==0 的66张只校准，另外455张训练/构库。原 seed42 支持名单及顺序不重采样。评价为 background 297（85/212）、illumination 265（67/198）、view 364（84/280），总926张，排除各域原8张支持，不用878共同子集。

`frozen_origins.json`从已封存correspondence目录清单提取原521张Source及background的图像身份、尺寸、名单和K8顺序，并固定全部926张的GT文件身份，仅含元数据/哈希，不解码GT；illumination/view图像清单继续核对原共享`frozen_catalog.json`。不是只检查数量，也不把用户换过的同名图片或GT当作原数据。

|分支族|训练查询|参照输入|Source初始化|
|---|---|---|---|
|REF_ST|目标阶段Source/Target各半|查询、Source/Target残差向量与3个距离标量|对应种子REF预训练|
|NOREF_ST|与REF_ST相同样本和日程|只读查询；其他输入置零，不访问NN缓存|独立NOREF预训练|
|REF_SOURCE_QUERY|目标阶段全部Source|仍有该域K8条件，不是严格source-only|共用REF_ST对应种子预训练|

每族训练42/43/44三个种子，均使用最终checkpoint。三个成员独立logit校准后逐层平均z，另拟合集成image MAD，不再拟合patch MAD。主候选为 `REF_ST_ENSEMBLE`。正式保存14套定位图：UNION/SEGMENT、9单模型、3集成。额外 `SEGMENT_SCORE_UNION_MAP` 只组合既定SEGMENT图像指标与UNION像素指标，不能算新预测器。

## 自排除、输入与网络

- Target原图或同源合成查询：正式Source + 其余7张Target。正式测试与66张校准查询：正式Source + 全K8。
- Source训练查询：455张按稳定列表交替A/B，228/227张；查询A只用B构成的训练专用100K/layer库，反之亦然。所有派生样本继承原图组。两折库重新调用原亮度增强和原100轮距离采样，保存候选行、原图索引、patch索引。它们不替换正式Source。
- Source预训练模拟7或8张支持：通过固定SHA/PCG规则，从查询的**对折**Source正常图选取，排除原图和所有派生版本；进入目标阶段后改用该域固定K8。

检索使用原始向量；每层输入为 `(q-mu)/sigma`、`(q-s*)/sigma`、`(q-t*)/sigma`，以及按原Source距离MAD标准化的 `ds,dt,min(ds,dt)`。mu为正式原Source库通道均值，sigma为全通道中心RMS。没有每patch L2归一化。最近邻索引保留，并用对应原始行的残差范数核查距离；不把向量输入标准化当作检索变换。

四层各自1×1→128，逐位置通道LayerNorm+GELU，两个局部残差块（3×3，dilation1/2），1×1→1 raw logit。无整图空间归一化，不更新DINO/memory，不引入Adapter、OT或SEGMENT门控。

## 固定合成池与tile监督

455×8 + 24×96 = **5944个固定合成槽位**，加479张完整正常版本。名义类型比例1/2正常内容Poisson移植、1/4正常纹理/结构扰动、1/4窄划痕。小/中/大尺寸分别为原RGB短边的[.01,.025]、[.04,.10]、[.12,.25]，最小局部边8px；划痕宽1–3px。具体参数、供体、位置、重试与无效原因均记录。没有DTD或真实异常供体。

前景辅助由原Source库L31的固定PC1近似、中心定向、Otsu及最大连通区域产生，面积仅接受5%–85%；不是GT。合成位置要求至少80%落在该估计中，不能获取时明确无效，不退回全图。每槽最多4次尝试。Poisson仅提交声明局部区域内的RGB修改；保存生成区域、实际RGB最大变化与软标签。最大变化至少2/255才有效，软标签按RGB平均变化/.1截到1，再用精确面积覆盖重采样到原评价尺寸。没有中值滤波或小区域清除。极小Poisson槽可能无有效变化，保留无效记录；若某原图完全没有有效合成版本，准备阶段明确失败，不静默删去该图。

每个有效合成RGB必须真实完整视野前向一次，随后才从原生特征裁tile。中心16×16，卷积半径3，再为插值增加一圈，**halo4**、最大24×24。保留覆盖所有正区域的中心块及至多4个固定正常中心块；完整正常图的中心块覆盖全部网格。合成tile以1:1正/正常块概率选择（某类不存在时使用存在的类别），原图均衡。损失在全局半像素映射对应的原输出像素窗口计算，小区域不会被硬下采样为零。标签处理独立于正式GT映射。

训练：有效batch4图，正常/合成1:1；REF_ST/NOREF_ST目标阶段Source/Target1:1。Source预训练2000更新×6模型，目标12000×27模型，共336000更新。AdamW3e-4、wd1e-4、5%warmup+cosine至1e-6、clip1。损失为四层分别正负面积归一化BCE的均值 + 四层平均logit的同类BCE + .5融合soft Dice。正常图Dice=0。每图独立归一化，避免大缺陷图支配batch。

原始特征、残差、NN、损失FP32；头训练CUDA BF16，头推理FP32且关闭其TF32，显式保留CuDNN的启用/benchmark/deterministic设置，退出后恢复骨干设置。骨干沿用父实验精度/设置，不因新头改变提取配置。FP64仅小型RMS统计和误差界统计。没有自定义Triton、编译器安装或额外CUDA扩展。

## 资产复用与重复工作

只读取显式 `--recent-run`（可重复）或新工具导出的 `--asset-manifest raw_assets.json`。不扫描旧AutoDL目录。

|资产|兼容后处理|不能复用/新增的原因|
|---|---|---|
|近期correspondence `source_original`|冻结Source seal、软件、宿主代码、权重和图像清单一致后链接|不读transformed memory或Adapter|
|24 Target、66 Source校准完整特征|只链接`full_L*`和`full_positions`|每图2048抽样行不是完整网格|
|455 Source完整正常特征|存在完整原始资产则复用，否则新增完整前向|旧correspondence只保留2048抽样，需要补提取|
|原始正式Q|correspondence disk-Q或显式raw manifest验证后读全向量|stream仅距离/索引不能反推Q；926张可能要再提取|
|UNION/SEGMENT z、原距离MAD|必须与同一正式原Source/支持/提取设置相容|不相容则在当前共同原库重算基线，历史结果不拼接|

|基线像素/AUPRO指标|重建的逐图map、GT SHA和完整名单与封存父评价完全相同才复用|校验不一致则计算本轮指标，不放宽容差|
|两折训练参照|新增，原增强/采样规则|用于严格同源排除，正式Source未被替换|
|5944合成槽位及NN/tile|新增一次，全族全种子共享|不能修改正常特征假装合成RGB前向|

复用 correspondence 原 Source 库不要求父运行已完成校准。父 `calibration/source/UNION_K8.json` 缺失时，准备完整正常特征后，用本轮共同原库及原66张 Source 校准图执行已有距离校准，保存为本轮 `input_distance_mad.json`，不写入父目录，也不创建父校准复用记录。父文件存在时仍须通过原封存校验，损坏、不可读或无效链接直接报错，不以重新校准掩盖。已封存的本轮校准在续跑时继续复用；网络 logit 校准仍独立进行。

correspondence、迁移后的WITNESS/fixed-graph schema分别识别；legacy距离/端点文件不冒充完整Q。兼容性包括提取依赖而不是要求旧图算法代码身份等于新分析代码。原始产物seal不修改，新工具自己的输入身份单独冻结。历史小review包只有报告，不能当成上述大缓存。

`preflight`只读文件/元数据，列出缺项、元数据可用与待prepare验证的状态，不把“存在”写成“已通过运行”。实际兼容、SHA和数值核查在显式计算阶段按需完成。

## 存储和计算成本

不能沿用旧方法100GB估计。以118×118、1280维、FP32为例：

- 单图完整四层约272MiB；6423张训练查询全部留全网格约1.67TiB（本实现**不**保存全部合成全网格）。
- 545张原始正常完整网格逻辑大小约145GiB，其中兼容的90张可链接，实际新增455张约121GiB。
- 单个最大24×24四层tile约11.25MiB；每个合成图保留的tile数取决于实际软标签，规模按实记录，不能预填总容量。
- 两折库合计约3.8GiB；原增强候选临时spool逐折约61GiB。100轮采样只将当前层（估计15GiB）顺序复制到`--sampling-scratch`的SSD私有目录，用完即清除；不反复随机读取HDD大候选。真实字节见`source_fold_resources.csv`。
- 还需合成RGB、原尺寸生成区域/变化图、输出软标签、NN索引、预测与checkpoint。实际总缓存可能达**数百GB甚至更高**，不能保证100GB够用。
- stream模式不保存926张完整Q；disk模式另约246GiB。查询/候选/memory完整距离矩阵不落盘。
- 单头3,152,388参数，纯FP32参数约12MiB；训练checkpoint还含Adam/RNG状态。Source6+目标27个最终模型，不按layer拆成132个训练作业。

大缓存放机械盘指定根，代码/日志/模型放SSD；采样与精确评价的scratch分别独立指定SSD，脚本默认分别为`/home/ubun/YYK/tmp/${tag}_sampling`和`/home/ubun/YYK/tmp/${tag}`，不同时堆积四层SSD候选。若正式原库也缺失，其旧上游构库器仍使用HDD spool。每个域仅处理该域定位图，最多8个CPU图像任务。一次训练缓存预取与GPU重叠；不逐loss读取GPU标量，日志每100更新一次，checkpoint每200。必要的非有限损失/梯度检查仍有同步。未在真实GPU计时，不承诺训练总小时数或加速比。

## 执行优化与独立 smoke

- `execution.ResidentExact` 为带 seal/layer 键的有界参照张量 LRU，默认4GiB，可通过 `--reference-cache-gib` 调整，0表示不留存参照张量。预算仅限制本进程缓存，不检查机器是否有足够空闲容量，不自动降级或占用第二张卡。矩阵、骨干及分配器缓存另占显存，4GiB不是进程显存上限。
- Exact仍调用原 `_exact_top1_chunked`，保留原FP32 `torch.cdist`、分块、逐支持顺序和单位。每张训练查询逐层上传Q一次，共享Source与四种条件的检索；正式预测/校准同样复用查询和常驻参照。没有近似NN、TF32或自定义Triton。
- 训练按索引直接从各支持mmap取tile需要的行，不为每个随机支持组合拼接整套完整K7/K8。排除policy按原图和阶段记忆化；每个读取的NN/参照仍核对封存身份。训练形状相同的tile合批，不补零或换tile；逐图插值、逐图loss、每图1/4权重和固定抽样保持。合批改变浮点归约次序，不承诺旧执行方式训练权重逐位相同；新执行方式续跑仍须保持优化器/RNG一致。

`smoke`是显式工程阶段，不属于`all`训练日程，不选择方法参数。报告写入独立 `run-root/smoke/<时间>/smoke_report.json`，不产生正式模型、预测或五项指标完成状态。临时checkpoint在私有scratch中写读核对后删除。默认`preflight`行为不变。

三种模式：

|模式|实际执行|耗时外推边界|
|---|---|---|
|`constructed`|真实Exact/完整CUDA头的小构造输入；CPU时小头；合批/顺序梯度对照及checkpoint写读|检查执行链路；小NN不外推全网格，纯头测速不包含真实机械盘读取|
|`assets`|上述头检查，加显式correspondence父资产的原100K Source、三域完整K8、固定第一张Source校准完整Q的真实检索；前三张Source训练正常RGB的原宿主前向|检索/前向按上限槽数粗估；校准图仅检索，不用于训练头；不生成合成池或两折库|
|`prepared`|已完成本工具`prepare-training`后，实际Dataset/policy/NoRef/预取/完整头；Source两族、三域三族，共11个短作业|按6个Source预训练和27个目标模型外推训练；真实冷缓存首读、构库及全域评价仍另计|

CUDA支持BF16时使用正式头训练精度；不支持时只报告FP32诊断，不冒充正式BF16已验证。合批数值核查是新执行方式的工程检查：FP32梯度相对L2≤1e-4/loss绝对差≤1e-5；BF16分别≤0.03/0.01，并记录实际差值。它不修改任何既有距离、地图或历史复核容差。报告列出均值/P50/P95、数据等待、checkpoint读写、Exact传输次数/字节、RAM/显存及已测环节小时外推。未测项留`pending`，`total_hours`保持空值，不拼造整轮实测耗时。首次计时包含库/驱动热身影响，稳定部分单列。

服务器执行（同步代码后，单物理GPU1、最多8 CPU；无关机）：

```bash
cd /home/ubun/YYK/DINO-Reference-AD
tag=aebad_reference_residual_smoke_$(date +%Y%m%d_%H%M%S)
run=/home/ubun/YYK/runs/$tag
cache=/home/ubun/YYK/storage/reference_residual_cache/$tag
mkdir -p /home/ubun/YYK/runs
tmux new-session -d -s "$tag" "cd /home/ubun/YYK/DINO-Reference-AD; env GPU_INDEX=1 CPU_WORKERS=8 RUN_ROOT='$run' CACHE_ROOT='$cache' RECENT_RUN=/home/ubun/YYK/runs/aebad_correspondence_20261025_201937 SMOKE_MODE=assets SMOKE_UPDATES=30 REFERENCE_CACHE_GIB=6 bash DINOv3/MADEqual/aebad_s_reference_residual_segmentation/scripts/run_workstation.sh smoke > '${run}_launcher.log' 2>&1"
printf 'SESSION=%s\nRUN=%s\nLOG=%s_launcher.log\n' "$tag" "$run" "$run"
tail -n 40 -f "${run}_launcher.log"
```

该命令不重建memory、不执行全域推理、不写父资产。会读取真实父Source/支持/校准特征、检查所用payload并加载骨干，前向3张正常训练图。父完整资产不可用则明确失败，不悄悄换构造输入。

本工具训练缓存准备完成后，在同一正式`RUN_ROOT/CACHE_ROOT`和其原父参数下运行脚本，设置`SMOKE_MODE=prepared`即可只测速实际训练路径。`smoke`的短步数不改变正式2000/12000步配置；不要把旧代码已冻结的运行改名冒充本版，可复用兼容的原始资产，但需新建正式输出目录。

## 手动命令

仅在代码review并同步后使用已有yyk环境，不修改base或安装依赖。本地验证没有执行以下服务器命令。

只读预检（physical GPU 1只作为后续计算选择，preflight不初始化CUDA）：

```bash
cd /home/ubun/YYK/DINO-Reference-AD
GPU_INDEX=1 CPU_WORKERS=8 \
RECENT_RUN=/home/ubun/YYK/runs/aebad_correspondence_20261025_201937 \
bash DINOv3/MADEqual/aebad_s_reference_residual_segmentation/scripts/run_workstation.sh preflight
```

完整手动tmux启动，固定RUN/CACHE以便续跑。**本次交付不执行此命令。**

```bash
cd /home/ubun/YYK/DINO-Reference-AD
tag=aebad_reference_residual_$(date +%Y%m%d_%H%M%S)
run=/home/ubun/YYK/runs/$tag
cache=/home/ubun/YYK/storage/reference_residual_cache/$tag
mkdir -p /home/ubun/YYK/runs
tmux new-session -d -s "$tag" "cd /home/ubun/YYK/DINO-Reference-AD; env GPU_INDEX=1 CPU_WORKERS=8 RUN_ROOT='$run' CACHE_ROOT='$cache' RECENT_RUN=/home/ubun/YYK/runs/aebad_correspondence_20261025_201937 bash DINOv3/MADEqual/aebad_s_reference_residual_segmentation/scripts/run_workstation.sh all > '${run}_launcher.log' 2>&1"
printf 'SESSION=%s\nRUN=%s\nCACHE=%s\nLOG=%s_launcher.log\n' "$tag" "$run" "$cache" "$run"
tail -n 40 -f "${run}_launcher.log"
```

可单独执行 `prepare → synthesize` 查看`previews/`后，再执行`prepare-training → train-source → train-target → calibrate → predict → evaluate → evaluate-aupro → summarize → pack`。把同一RUN_ROOT/CACHE_ROOT和RECENT_RUN环境变量传给脚本即可。续跑执行同样参数的原阶段或`all`；已完成准备阶段优先检查小manifest，不再全量扫描RGB/完整特征，payload在真正消费时核查。中断前未提交的特征块可能重提取，已封存模型从保存update继续；更改实现/配置要新建运行目录。`evaluate/evaluate-aupro`使用冻结预测、校准和GT，不要求Source bank或骨干实体；重建地图的软件版本仍需与冻结配置一致。

## 输出、校准和完成状态

- `inputs.json/catalog.json/prepared.json/raw_assets.json/reuse_audit.json`：输入、原图/支持顺序、原始资产来源、提取配置与代码身份。
- `foreground_audit.csv/previews/`：全部455 Source与24 Target辅助前景；合成每原图前4槽预览，不能据真实异常指标选合成规则。
- `synthetic_manifest.json`：所有5944槽，含无效原因；大cache保存RGB、生成区、实际变化、soft_output、固定供体/参数。
- `training_manifest.json`：每原图/合成的完整前向来源、物理位置、tile/halo、Source/Target索引及自排除policy。残差训练时按原始行计算，不重复保存。
- `models/`：optimizer、scheduler位置、Torch/CUDA RNG、PCG采样方案、完成update、固定最终模型与日志。
- `calibration/`：新raw logits的patch MAD、域K8条件下66 Source图的image MAD；基线保留原Source-only距离MAD。零/近常数输出明确失败，不能用小分母救场。
- `predictions_frozen.json`：926张×14分支全部提交后才能解码GT。未冻结完整人口不能评价。
- `evaluation/`：两种图像评分、共享像素评价、同缓存数值复核、逐图地图/GT SHA、连通实例及固定UNION峰值口径，保留NA/小缺陷。
- `aupro/<domain>/<branch>.json`：14×3个逻辑作业，原ADEval1.1.0/200阈值/FPR≤.3。6个基线作业可在严格相同地图证明后复用，新增模型最多36个作业；不是CH1/CURRENT两倍。
- `metrics.csv/metric_comparisons.csv/training_repeat_summary.csv/margin_*.csv/objective_report.md/status.json`：分域和三域等权macro，单模型波动与实际集成分开。CH1/CURRENT和历史来源保留；AUPRO未齐保持pending，不填0。
- `costs/`：阶段时间、峰值RSS/显存、新增数组字节；`storage_usage.csv`：实际存储量；`training_jobs.csv`：参数/模型大小和训练时间。
- `_review.tar.gz`：报告/元数据/预览/日志，排除cache、scratch、`.bin/.npy/.pt`。大缓存和可恢复checkpoint应另行备份，review包本身不保证可重训。

margin在各模型独立校准尺度下仅为辅助，不能当成真实信息损失或部署漏检。raw NN距离不改变，因此不把新logit变化命名为原距离R_N/R_A/G。收益需同时看图像、定位、实例和种子波动；不能据训练loss或热图宣布有效。

## 本地验证

```powershell
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'
$env:MPLBACKEND='Agg'
& 'D:/anaconda/miniconda/envs/chinamoblie/python.exe' -X utf8 -B -m unittest DINOv3.tests.test_aebad_s_reference_residual_segmentation -v
```

实际结果和检查清单见`implementation_audit.md`。测试使用G盘临时目录，不安装Torch/ADEval。真实DINO、原增强100轮训练fold构建、CUDA/BF16、全量缓存I/O、正式训练与ADEval尚未验证。CPU小张量训练通过不能称为真实实验完成。
