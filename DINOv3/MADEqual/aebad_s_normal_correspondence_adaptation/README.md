# Source–Target正常对应残差特征适配

本模块是待真实实验验证的候选，不是CAReg官方复现。只提供本地实现、CPU验证及手动服务器入口；不自动连接服务器、安装环境或启动GPU。默认只读`preflight`，不导入Torch、不初始化CUDA、不创建输出。

## 信息条件与可复用资产

以用户2026-10-25补充为准：旧AutoDL资产已经不存在。仅考虑用户通过`--recent-run`明确指定的迁移后两轮：

- `aebad_s_witness_fresh_20261023_160224`：原图fresh WITNESS。
- `aebad_fixed_graph_fp32_20261024_235345`：三域七分支固定图；其冻结Source/支持资产来自同机父目录。若传入此子目录，只读取其记录中指定的原提取setup来核查身份，不运行图代码、Triton或读取图payload。

不是扫描整台机器寻找历史资产。原报告和review包通常不含完整特征；本地有包不能代替服务器实体。两个recent参数都是可选的；省略即可fresh。兼容性失败会记录具体原因，不把不同环境资产静默混合；缺失资产按原宿主补提取。

|输入|复用/重复规则|
|---|---|
|Source原始100K×1280×4 bank|兼容则只读一次并写入新独立FP32 NPY库；不存在则复用原455图亮度[0.8,1.2]、100轮正式builder构建一次。所有模型逐行变换，绝不重新采样。|
|24张Target完整特征|按图像SHA、路径、支持顺序及提取实现核查后复用；缺失才提取。|
|455张Source训练特征|需要图像身份及上下文，原bank不能替代；完整网格算教师后只保存最多2048训练位置及描述子。|
|66张Source校准特征|旧距离不能恢复特征，新增提取并保存完整四层。只用于校准。|
|926张查询|默认stream，同一图前向一次供六分支共享，不保存Q；disk模式保存本轮完整Q，续跑可免前向。旧距离/索引不能恢复Q。|
|原UNION/SEGMENT及原MAD|本轮共同bank上重新检索、校准、评价。这样不要求旧预测实体，也不把旧汇总指标与新表示混成对照。即使recent兼容，也不会复用旧MAD到新模型。|

Source原排序521、index%8==0的66张校准、其余455张训练/构库。seed42每域8正常支持保持身份和顺序。评价background297(85/212)、illumination265(67/198)、view364(84/280)，总926；不使用878共同子集。三域适配互不读取其他域支持；GT首次解码在全部预测封存之后。

## 固定方法

四层L7/L15/L23/L31，1280维。每图最多2048个位置使用路径稳定PCG64无放回采样，四层相同；先计算完整网格单位特征e及包含中心、边界有效邻居3×3均值的单位上下文c。教师相似度为`sum_l(dot(e,e')+dot(c,c'))/8`，只用于正常配对，不用于最终检索。

图像配对：SS每图最近2+随机2，ST每个Target最近16个Source+随机4，TT全部28对。随机候选排除最近集合；无向对去重，训练均匀选图像对后均匀选其非空方向及anchor。对应为双向mutual Top3，精确并列按物理索引，正例softmax(sim/.07)，每anchor从另一图相似度最低floor(N/2)集合排除正例后固定无放回选四负例。空对应对留在清单；某类全空则明确失败。

每层适配器：`h(f)=f+sigma*Linear2(GELU(Linear1((f-mu)/sigma)))`，1280→512→1280，含bias，最后层零初始化。mu及全通道RMS sigma只由原Source bank拟合；小向量累计用FP64，网络/特征FP32。Source尺度为零或非有限时失败。

训练距离`d=norm(h(a)-h(b))/(sqrt(1280)*sigma)`。`L=Lpos+Lneg+0.1*Lid`；Lpos是anchor内多正例加权平方距离，Lneg是`relu(.8*d0-d)^2`均值，Lid是配对行和额外正常行的标准化位移均值。Source每步额外512个Source行；Target阶段额外256 Source+256完整Target行（按行均匀）。域阶段三个配对类别分别求平均再等权；256 anchor按86/85/85轮换余数，保持总数不变。

AdamW lr1e-4/wd1e-4/clip1；各阶段前5%线性warmup，再cosine至1e-6。每层Source预训练2000步；各域从该种子Source最终权重独立开始10000步，重置优化器。种子42/43/44只改变训练，不改变支持/配对/评价人口。使用最终更新，不看测试指标选checkpoint。

训练仅Linear在CUDA允许BF16；归一化/GELU/残差/loss/主参数/优化器为FP32。匹配及正式网络推理/NN禁用TF32及autocast，骨干保留最近原FP32提取配置。**无自定义Triton、编译器依赖或全流程FP64。**原SEGMENT几何内部的小规模FP64保留原定义。

每个域模型同时变换query、原Source全部行及Target全部行。完整Exact检索欧氏距离/1280一次；不复用旧NN索引，不与原UNION再取min。每模型用原66张到h(Source)的Source-only距离独立拟合patch MAD和CH1图像MAD。

集成先对三个模型独立patch MAD后的z逐层FP32平均，再四层等权及原float32插值；image MAD在66张Source的集成z上重新拟合。无第二次patch MAD，不平均成员image MAD或指标。CH1主表、CURRENT补充，同图像评分共用定位图。

## 阶段和续跑

`preflight / prepare / match / train-source / train-target / transform-memory / calibrate / predict / evaluate / evaluate-aupro / summarize / pack / all`。

- prepare只有正常训练/支持/校准图，不读查询进行匹配。
- 每100步及最终更新封存checkpoint，包含网络、优化器、调度器位置、PCG64和Torch/CUDA RNG、更新次数。中断丢失尚未提交的最多99步，续跑从已提交次数恢复；不会在计数上重复已提交更新。每层仅保留当前已提交checkpoint，写新文件并原子更新pointer后才删除上一份。最终权重与optimizer在同一封存文件内。
- 数组bundle逐文件SHA并记录依赖；训练Store每阶段每bundle只核验一次，不逐batch重新哈希。GT不用于生成配对。未提交bundle只重写本轮未提交NPY；已提交文件被修改则失败，不重新登记SHA。
- query每图独立提交；完整926张清单封存前拒绝解码GT。CPU域/精确像素中间结果和18个AUPRO作业均可续跑。两种图像评分只计算一次像素AUPRO。
- 无RAM/显存/磁盘容量门槛，无自动安装、自动换卡、自动降级、自动关机。实际分配/数值错误保留现场。
- 路径隔离同时覆盖run、cache和evaluation-scratch：不得与数据、权重、宿主/骨干源码、指定recent及其记录的父运行/cache相互包含。路径先resolve，包含软链接目标。scratch还必须与本轮cache分离，不能等于或包含run；默认run/pixel_scratch仍合法。直接调用CPU评价入口也会核对。
- 每次域评价在scratch根下创建唯一空工作目录，以独占方式创建数组文件；分配失败也只删除本次已创建的文件。既有域目录、遗留`.bin`和新工作目录内非本次创建的文件均保留，不按通配符清理，不截断旧同名文件。

## 新工作站手动命令

通过GitHub同步发布代码，服务器先git pull。以下命令只在用户手动执行时运行。

```bash
work=/home/ubun/YYK
repo="$work/DINO-Reference-AD"
export WORK="$work" REPO="$repo"
export GPU=1                         # 用户指定的单张物理GPU；程序内部cuda:0
export RUN="$work/runs/aebad_correspondence_$(date +%Y%m%d_%H%M%S)"
export CACHE="$work/storage/correspondence_cache/$(basename "$RUN")"
export RECENT_WITNESS="$work/runs/aebad_s_witness_fresh_20261023_160224"
export RECENT_GRAPH="$work/runs/aebad_fixed_graph_fp32_20261024_235345"
export QUERY_MODE=stream
cd "$repo" || exit 1
bash DINOv3/MADEqual/aebad_s_normal_correspondence_adaptation/scripts/run_workstation.sh preflight
```

fresh运行不设置这两个recent变量：`unset RECENT_WITNESS RECENT_GRAPH`。数据/权重默认当前工作站已知路径，可分别用DATA/CHECKPOINT覆盖。没有任何旧AutoDL绝对路径。只使用yyk环境，脚本不改base或安装依赖。

显式完整运行（tmux、8 CPU workers、无关机）：

```bash
mkdir -p "$WORK/runs" "$WORK/tmp"
session="correspondence_$(date +%Y%m%d_%H%M%S)"
"$WORK/conda_envs/yyk/bin/tmux" new-session -d -s "$session" -c "$REPO" \
  "env WORK='$WORK' REPO='$REPO' GPU='$GPU' RUN='$RUN' CACHE='$CACHE' QUERY_MODE='$QUERY_MODE' RECENT_WITNESS='${RECENT_WITNESS:-}' RECENT_GRAPH='${RECENT_GRAPH:-}' bash DINOv3/MADEqual/aebad_s_normal_correspondence_adaptation/scripts/run_workstation.sh all > '${RUN}_launcher.log' 2>&1"
printf 'SESSION=%s\nRUN=%s\nLOG=%s\n' "$session" "$RUN" "${RUN}_launcher.log"
tail -n 60 -f "${RUN}_launcher.log"
```

Ctrl+C退出tail，tmux继续。续跑设置**同一**RUN/CACHE、配置及输入，再执行同一all命令。也可把all换成某阶段。`evaluate-aupro`是显式GPU评价阶段，不调用骨干。默认脚本不开始all。没有未经实测的总时间承诺；训练日志每100步记录累计有效更新时间，已完成更新直接保留。

## 存储与资源

代码、模型checkpoint、运行状态、指标和精确像素排序scratch放SSD；Source临时候选、训练样本/教师、完整校准及支持、变换memory、冻结预测放机械盘CACHE。不会把121GiB Source候选写入SSD。配对使用256行GEMM块，完整相似矩阵不落盘。

按13924 patch典型网格估计：stream持久原特征、教师、变换memory约110–140GiB，另加匹配、小checkpoint及预测；非实测峰值。fresh Source构库另有约121GiB临时候选，结束后由原builder释放。disk模式926张完整Q额外约246GiB，不能按距离缓存大小估算Q。父实验缓存另计；没有检查容量后自动拒绝的逻辑。

训练按层/图像对访问mmap，不把全Source完整网格持久保存在内存。正式预测每域把三模型全部变换库约12GiB一次载入CPU RAM，避免每图重读机械盘；GPU只接收当前模型/层库，不要求三套全部库同时驻留GPU。原Source和骨干也占显存。具体峰值取决于实际host及NN块，需服务器实测。

资源记录：costs逐阶段时间/峰值RSS/显存/新增数组字节；每个NPY manifest记录实际文件大小，模型complete记录参数量及checkpoint大小。不为了报告输出大小逐batch递归扫描cache。

## 输出与解释

catalog/inputs/prepared、reuse_audit、matches清单、models及training_log、memory_mapping、各模型独立calibration、predictions_frozen、evaluation、aupro、metrics、metric_comparisons、training_repeat_summary、defect_instances/margin_changes/margin_summary、固定首两张正常/异常地图案例、costs、status和objective_report。

六种正式预测：UNION、SEGMENT、三个ADAPT成员及固定三模型集成。额外SEGMENT_SCORE_UNION_MAP只组合SEGMENT图像指标和UNION像素指标，不再评价像素。成员均值/样本标准差单列，绝不冒充集成指标。

原精确CPU全像素ROC/AP、原ADEval1.1.0/200阈值/FPR≤.3；NA保留，小缺陷口径复用旧核：background最小三个、另外两域最小面积四分位。响应和margin变化是各自正常校准后值，不复用原空间物理距离解释；下降不等于部署漏检。未齐36个域模型层作业、12个Source预训练层作业或18个AUPRO，不标记完成。

review包只含报告类型文件，排除scratch、大数组及PT；模型和预测仍在原运行/cache中，review不是全量恢复包。未自动上传、打包服务器旧实验或清理其资产。

## 本地验证边界

```powershell
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'; $env:MPLBACKEND='Agg'
& 'D:\anaconda\miniconda\envs\chinamoblie\python.exe' -X utf8 -B -m unittest DINOv3.tests.test_aebad_s_normal_correspondence_adaptation -v
```

测试执行实际Torch CPU配对/适配器/loss/优化器checkpoint恢复/Exact NN，以及小型1280维完整库变换、独立与集成校准、六分支预测、原地图/空间实例/精确CPU像素评价、哈希和续跑。集成测试替换了训练后网络、图像读取和骨干输出，使用缩小的合成目录，不是实际训练/真实DINO/正式926张评价。真实CUDA、BF16训练、全量匹配与ADEval仍未执行，不因NumPy/CPU测试通过而宣称有效。
