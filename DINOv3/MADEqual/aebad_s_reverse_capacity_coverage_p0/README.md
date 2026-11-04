# 面积容量约束的反向覆盖：固定配置

**当前新建运行默认使用工作站脚本的 `LP_PROFILE=mixed_background_v2`，只执行background/seed42。** 下方原六设置CPU协议与早期GPU试验作为历史实现说明保留；本轮运行以本节新版协议为准。直接Python CLI默认仍保留旧 `cpu_v1`，新阶段须显式传 `--lp-profile mixed_background_v2`，旧目录不会被迁移或覆盖。

## 当前阶段：background / seed42，capacity_LP_mixed_v2

固定共同名单289图（77正常、212异常），继续排除两组支持并集；原Source66校准、固定K8顺序。评价2312 LP、校准528 LP，共2840 LP，363张独立区域图（66+8+289）。只输出六分支Image AUROC/AP，primary仍为CAPACITY_FUSION。原特征、RMS、256区域、面积、距离、每完整模板LP和8模板min、Source校准及等权融合定义不变。seed43/illumination/view均不请求，无像素/AUPRO、训练、memory重建或参数搜索。

### 新数值验收

| 项目 | 新协议 |
|---|---|
| max行和误差 | ≤1e-5 |
| max逐列容量相对误差 | max(abs(a@W−b)/b)≤1e-5 |
| 最大行代价与返回t的一致性 | ≤1e-5×max(1,abs(t)) |
| 有CPU抽查时的t差 | ≤1e-5×max(1,abs(t_CPU)) |
| W非负性 | min W≥−1e-10，另记录负质量sum(max(−W,0)) |
| 原始—对偶归一化间隙 | abs(p−d)/max(1,abs(p)+abs(d))≤1e-5 |
| 对偶可行性 | 下述两项均≤1e-5 |

本LP写为min c@x，等式E、≤约束L、x=(W,t)≥0。对偶y的E项自由、L项≤0，要求s=c−A.T@y≥0。分别独立重算 `max(max(−s,0))/max(1,||c||∞)` 和 `max(max(y_L,0))/max(1,||c||∞)`；本模型||c||∞=1。原始目标p=c@x=t，对偶目标d=rhs@y。对偶约束符号按官方min-LP约定，绝不自动翻号以挑选能过的解。每个LP单独核对，不能用8题总间隙掩盖个别失败。

记录原始等式L∞误差、不等式违反量、A/B/C/D、最小W/负质量、对偶两个违反量、p/d与归一化gap。cuOpt返回的批级L2原始/对偶残差、原始/对偶目标、原gap、迭代数、停止原因和引擎时间另存，不能把它们冒充逐LP独立口径。原始W与y均存入新cache证明包，后续读取重新验收；不裁剪负值、不归一化或修复解。覆盖分数继续使用返回t，不能改用实际最大行代价。

三类状态：Optimal且独立验收通过；TimeLimit但所有独立验收通过（明确标为近似解）；其他状态/缺证据/不通过均为未完成。旧失败小表不能仅换协议标识就通过。旧CPU严格成功记录允许只读复用，复核原输入、原HiGHS Optimal、原严格可行性/目标检查，记录来源；其原记录未存的对偶值/gap/负质量写NA，不编造为0或“本轮独立重算”。本次新GPU问题都须有完整的原始/对偶证据。

历史SEGMENT P、map/图像/支持身份以及原1e-5检查、距离定义、面积输入和校准退化规则保持原要求。

### 固定求解配置和环境

cuOpt PDLP，显式mixed=2，Stable3、单GPU、crossover=false、每批30秒、per_constraint_residual=true；绝对/相对primal/dual/gap六个停止参数均固定为1e-7，独立验收按上表1e-5，不继承旧HiGHS的1e-10停止要求。没有自动改变参数、加CPU/FP64精修、切Concurrent或更换优化目标。设置常量来自安装版本注册表，不识别时明确失败。

主进程继续在原yyk中做必要的原前向/分区/直接FP32距离，常驻子进程在独立yyk_lp中求解；默认主进程1个CPU槽位、solver最多7线程。cuOpt参数、实际包版本、实际solved_by、error status和日志保留。混合不是全FP32。无需修改或重装原环境。

### 资产、16题复核和时间预算

`--resume-from` 指向原覆盖运行，复用其RMS与区域；原16题导出通过 `--distance-export` 只读接入。其他已有兼容完整区域/距离直接使用，缺少区域时仅对本阶段必要人口生成；缺失完整特征仅在显式 `--extract-missing` 时补提前向，不重提已有图像。Source455不重读、不重建memory。

`preflight`清点当前363图与2840项，严格CPU结果先列候选，在消费时按输入/证据复核；预算使用尚未确认项的上界，不把候选全算作已经复用。新run/cache与旧资产隔离，小记录和原解全部只读保留。没有大数组全量SHA重扫，没有RAM/磁盘/显存容量门槛，没有自动关机。

旧GPU试验仅保存了16题的数值摘要、批级残差等，未保存原始/对偶向量。入口先读取这些小表与可获得的完整日志，输出 `legacy_LP_reaudit.csv/legacy_log_inventory.json`；证据不足仍标未完成，不从batch gap推断每题最优性。只对原16题补齐新协议的证据，不生成新LP输入、不新增全量CPU对照。`LP_review16.csv`固定16题，历史失败尝试另存 `LP_review16_attempts.csv`。

16题全部验收后，最多补准备两张本阶段确实缺失的正式查询区域，测冷/暖前向和分区吞吐，再测最多两组必要新距离的计算/落盘；这些资产后续直接复用，不扩展名单。用全部实际批次端到端耗时（包括失败重试/超时通过、建模、传输、验收、落盘和worker初始化）外推剩余LP；区分冷启动与暖前向、区域、距离、其他CPU/IO未知项，未测不填0。超过12小时则保存budget_pause，停止并等待用户明确同意；不会因此自行扩大范围或重试精度。

`all`在没有当前smoke时先完成上述16题复核与成本估算。未通过或超过预算就停止；通过且预算在约定范围内才依次prepare/predict/evaluate/summarize/pack。只处理本阶段。逐LP提交；中断后复核并复用成功解，完整worker解尚未写pair指针时也可恢复。失败尝试保留，不能以FREE代替。

默认preflight只写暂存的preflight_inputs，不会因为尚未选择GPU就冻结错误的执行身份；显式smoke/all才冻结inputs。工作站脚本将CUDA/CuPy/Numba缓存也限定在YYK/.cache内，不修改base或其他人的目录。

### 服务器手动运行

代码同步后，先显式选择GPU1、设置新目录。原运行指针不覆盖，新运行另记为latest_reverse_capacity_phase1_run.txt。

```bash
work=/home/ubun/YYK
cd "$work/DINO-Reference-AD"
tag=aebad_reverse_capacity_background_v2_$(date +%Y%m%d_%H%M%S)
export RUN_ROOT="$work/runs/$tag"
export CACHE_ROOT="$work/storage/reverse_capacity_cache/$tag"
export COVERAGE_PARENT=$(cat "$work/runs/latest_reverse_capacity_run.txt")
export LP_PROFILE=mixed_background_v2 GPU_INDEX=1 CPU_WORKERS=8
script=DINOv3/MADEqual/aebad_s_reverse_capacity_coverage_p0/scripts/run_workstation.sh
bash "$script" preflight
# 可先单独复核16题；all会复用它。
bash "$script" smoke --extract-missing
# 或显式all：从16题复核开始，经预算检查后只完成background/seed42。
bash "$script" all --extract-missing
```

正式执行使用tmux。若超过12小时，贴cost_estimate与budget_pause后再决定；现有allow-over-12h只供用户明确同意后手动指定，默认关闭。

后台入口为同目录 `scripts/run_phase1_tmux.sh`，默认仍只preflight；显式 `GPU_INDEX=1 ... all` 才在tmux中运行本阶段。它生成启动脚本/日志/退出码，写独立phase1指针，保护旧run的启动日志，所有新文件仍在YYK。Bash启动夹具使用假的tmux/runner核对了参数、退出码和旧目录保护，不等于验证过真实Linux GPU求解。

### 交付与完成状态

输出新 `LP_protocol.json/inputs.json/job_plan.json`、16题复核与原始向量、各LP的完成/未完成/失败尝试/停止原因、Source P/cap/free均值std、289张逐图六分支、六分支AUROC/AP、比较表和objective_report。`LP_error_in_calibration_units.csv`在本轮Source尺度可用后报告16题t误差/σcap。8模板min不会放大最大绝对输入误差，但标准化可放大，16题不能保证全域排序不变。

LP_completion含全部2840项（缺项明确未完成），只有全部LP与六分支完成才记完成。原始计划与证明矩阵存机械盘cache，review只含小表/说明/日志。时间/进程树RSS来自阶段记录，GPU主进程torch峰值与worker批后驻留单列；批后采样不是瞬时显存峰值，未测写NA。

以下为原实现与早期试验说明；本阶段不自动运行其中其他域/支持组。

本模块只增加图像评分。预定主候选为 `CAPACITY_FUSION`。没有网络训练、Source重建、参数扫描、Target-LOO校准、实例分析或新增像素/AUPRO作业。三个域均有开发历史。

## 固定任务

| 项目 | 数量 |
|---|---:|
| background / illumination / view共同评价图 | 289 / 257 / 356 |
| 唯一评价图 / 图像×支持组实例 | 902 / 1804 |
| 支持 | 原seed42、43，每域每组K8，共48张 |
| Source构库图RMS / 独立校准图 | 455 / 66 |
| 需分区的唯一图像 | 66+48+902=1016 |
| 区域数 | 每图256 |
| 正式 / 校准容量LP | 14432 / 3168，共17600 |
| 新定位 / AUPRO / 训练 / Source构库 | 全部0 |

区域始于唯一物理patch，四邻接、贪心合并当前距离最小的相邻区域；并列按区域最小物理索引对确定。区域面积为patch占比。距离是四层RMS缩放后的欧氏距离除以sqrt(1280)再平均，合并和匹配同一公式，不加入坐标、前景或层权重。

RMS只流式累加455张未增强原图的完整特征，分母为1280×总patch数。不能使用66张校准图、Target、采样100K memory或距离图代替。原Source memory增强规则不受本模块影响。

主覆盖使用 `W_ij=T_ij/a_i` 的稀疏LP：行和1、加权列和b、每行平均代价≤t，最小化t。`scipy.optimize.linprog(method='highs-ds')` 求解，256×256时65537变量、768个约束，约19.7万非零项；不建立768×65537稠密矩阵，不用普通最小总成本/Sinkhorn替代。无容量对照为 `max_i min_j d_ij`。两者各自从8张完整模板中取最小值，保留全部模板结果和胜出ID。

六个图像分支：

1. `SEGMENT_CH1`: 原P，直接引用冻结分数。
2. `FREE_SHARED_SCALE`: `Z_P(P)+Z_cap(C_free)`，与主候选共享容量标准化参数。
3. `FREE_OWN_SCALE`: `Z_P(P)+Z_free(C_free)`。
4. `CAPACITY_FUSION`: `Z_P(P)+Z_cap(C_cap)`，预定主候选。
5. `CAPACITY_ONLY`：原C_cap，辅助。
6. `FREE_ONLY`：原C_free，辅助。

每个域×支持组仅用本设置66张Source校准图拟合均值/std（ddof=0）。退化阈值预设为 `64*eps64*max(1,maxabs)`；依赖退化尺度的融合分支记为blocked/pending，不加epsilon，不选其他系数。原P正仿射标准化应保持排序，用AUROC/AP核对。

## 资产：自动检查当前机器上的已知索引

入口读取现有两K8父结果、最新条件评分结果、残差分割的catalog/prepared，以及当前机器上SACP/correspondence/中断残差运行的已知索引。最新条件评分的正常观察还保存了全部48张支持和66张校准图的完整特征来源，可直接追溯。

当前服务器上的历史文件按用户说明保留；本轮没有连接服务器。因此，本地清单中的“服务器待确认”不等于服务器丢失，也不要求用户先手工打包或整理缓存。

本地已核实1804条P、396条Source66面对各组完整K8的P、共同名单和支持顺序。455张Source及66张校准完整特征有prepared索引；48张支持完整特征有封存的观察来源。大数组主要仍在Linux路径，未下载到本地。查询完整特征须由服务器preflight查实：旧运行默认stream，生成过前向不代表保存过完整Q，很多raw_cache只有4×118×118距离图。

查找只走已知provider、manifest和候选目录；不扫描整块机械盘，不加载模型检查存在性，不默认重建memory。支持q7/q15/q23/q31、原查询L7/L15/L23/L31（metadata.sample/geometry），以及correspondence原始full_L*正常特征。拒绝采样、适配、增强和无提取身份的数组。可用 `--asset-run`、`--feature-root` 或原有封存 `--feature-manifest` 补充已知来源，无需改写旧manifest。

`preflight`输出 `asset_inventory.csv`、`missing_inputs.csv`、`necessary_work.csv`、`numerical_verification.csv`、`preflight.json`、冻结配置与任务清单。它区分本地实物、历史来源、当前待确认。已生成兼容区域可直接复用。

区域manifest统一保存既有特征/新补提前向的来源、图像、提取身份、特征seal和物理grid。预检与续跑使用同一个验证函数；原特征目录暂时不可访问时，无需原始Q或单独的feature_origins文件即可复用区域。若原特征仍可访问，继续核对其身份；RMS、提取配置或区域定义变化仍拒绝。旧区域格式只读核对已封存依赖、grid和数组头，不重写旧manifest。

P齐全时绝不重新运行Source检索或SEGMENT。本实现没有P缺失时的隐式重算：缺失会明确阻断接入，先报告范围。只在明确传入 `--extract-missing` 时补提确实缺少的完整图特征，复用原host、完整视野与精度；补提只生成RMS或区域，不默认保存完整查询Q。

## 执行与成本

默认stage为 `preflight`。运行脚本不安装依赖、不修改base、无服务器连接、无关机命令。CPU使用最多8个槽位；例如CPU_WORKERS=8时主进程+最多7个单线程HiGHS进程。区域合并和LP在CPU；可选一张GPU只做stock PyTorch直接FP32 cdist和显式缺失前向。无自定义Triton、无全流水FP64。RMS累加、标量校准及LP使用FP64。

现成特征以只读mmap逐图处理；支持区域仅约240MiB，可在匹配阶段常驻。新区域均值约5MiB/图，1016张约5GiB，加标签和小记录；不是新增数百GiB查询特征。若全部现成完整特征都需读取，RMS455和分区1016是各自真实的读盘工作，不能当零成本；RMS只对训练455，分区只对其余1016，没有对同一人群隐藏重复扫描。具体耗时须实测。

smoke先完成或复用精确定义的Source RMS；**RMS缺失时，这一步需读完455张Source特征，不是只用几张估计RMS**。随后用background/seed42第一张Source校准、第一张正式查询、完整8模板测两组完整路径（16个256×256 LP），并核查GPU距离对CPU直接FP32参考的误差。它只用于数值与吞吐，不读取测试标签/GT来选输入。

`cost_estimate.json`区分缺失前向、剩余区域划分、距离+LP+小记录IO、未测评价与打包。使用真实同规模测量，复用部分不重复计入；复用旧pair的读盘时间不会被当作新LP测速。只有一个设置的外推不是全域耗时保证。没有对应测量时保留未知，不承诺几小时完成。

预计或实测投影超过12小时会留下 `budget_pause.json` 和续跑产物，等待用户同意，不缩数据或改目标。`--allow-over-12h`仅用于用户另行明确批准以后。本轮未获得该授权。

## 服务器手动命令（代码发布后，用户自行执行）

先激活现有yyk；代码同步单独执行，不藏进启动脚本：

```bash
source /home/ubun/miniconda3/etc/profile.d/conda.sh
conda activate /home/ubun/YYK/conda_envs/yyk
cd /home/ubun/YYK/DINO-Reference-AD

export RUN_TAG=aebad_reverse_capacity_$(date +%Y%m%d_%H%M%S)
export RUN_ROOT=/home/ubun/YYK/runs/$RUN_TAG
export CACHE_ROOT=/home/ubun/YYK/storage/reverse_capacity_cache/$RUN_TAG
export CPU_WORKERS=8
export GPU_INDEX=1   # 仅在用户确认使用此卡后；不设则纯CPU。
script=DINOv3/MADEqual/aebad_s_reverse_capacity_coverage_p0/scripts/run_workstation.sh

bash "$script" preflight
```

预检退出2表示缺项清单，先看是否是服务器真的没有完整特征。不得把本地待确认清单当作服务器缺失结论。

用户允许测速后：

```bash
# 不允许补提时去掉 --extract-missing，缺项即停。
bash "$script" smoke --extract-missing
cat "$RUN_ROOT/cost_estimate.json"
```

用户批准正式计算，且预算允许后，继续同一输出目录：

```bash
mkdir -p /home/ubun/YYK/runs
printf '%s\n' "$RUN_ROOT" > /home/ubun/YYK/runs/latest_reverse_capacity_run.txt
tmux new-session -d -s "$RUN_TAG" \
  "cd /home/ubun/YYK/DINO-Reference-AD && env RUN_TAG='$RUN_TAG' RUN_ROOT='$RUN_ROOT' CACHE_ROOT='$CACHE_ROOT' CPU_WORKERS='$CPU_WORKERS' GPU_INDEX='${GPU_INDEX:-}' bash '$script' all --extract-missing > '${RUN_ROOT}_launcher.log' 2>&1; rc=\$?; printf '%s\\n' \"\$rc\" > '${RUN_ROOT}.exitcode'"
tail -n 40 -F "${RUN_ROOT}_launcher.log"
```

Ctrl+C退出tail不停止tmux。中断后保留相同RUN_ROOT/CACHE_ROOT、同距离后端及固定定义重跑 `all`；只复用已封存、输入一致的RMS、区域、pair和分数。已保存LP失败原样保留并阻断，不能当成FREE成功。CPU/GPU计算后端切换必须新目录和新smoke，不能混用。

默认所有新cache在YYK/storage机械盘，结果和小日志在YYK/runs。没有SSD scratch、全像素排序或删除父目录的代码。

## 输出、检查及结论边界

- `inputs.json/protocol.json/job_plan.json`：定义、提取身份、支持/共同名单来源、实际依赖和后端。
- `source_RMS.json`：455图的归约及RMS；cache保存区域labels/counts/area/四层mean/物理roots和manifest。
- `pairs/<query>/<template>.json`：每模板目标、FREE、求解状态/耗时、行和误差、相对容量误差、非负性、目标一致性及来源。
- 六设置各 `calibration.json`、coverage记录、scores；全局 `predictions_frozen.json` 之后才评价标签。
- `normal_calibration.csv/template_costs.csv/per_image_scores.csv/setting_metrics.csv/metrics.csv/metric_comparisons.csv`。
- `objective_report.md/status.json/resource_summary.json` 和轻量review包。报告保留每域/支持组下降和两种FREE对照，不池化1804条。

报告中的“六设置”改善/下降计数只包含原始seed42/43行，分别统计AUROC和AP，不包含三个域的两seed均值。

检查清单见implementation_audit。没有重复的大数组SHA校验：读取检查现有小manifest/身份、npy头和物理positions；原数组内容可信本地读取。新区域数组写入时按现有存储记录一次身份；小结果封存、已完成小表在汇总时检查。LP相对误差和原P的1e-5复核是科学计算正确性要求，不是可用RAM/磁盘门槛。

区域均值可能稀释小缺陷，面积不是部件数；本轮不证明拓扑或结构完整，也不保证真实缺陷保持。不根据结果选择辅助分支、参数或域。

## 本地验证

```powershell
& 'D:\anaconda\miniconda\envs\chinamoblie\python.exe' -B -m unittest DINOv3.tests.test_aebad_s_reverse_capacity_coverage -v
```

只使用小型CPU夹具和已完成的本地小表。真实256×256全规模LP吞吐、真实118×118分区吞吐、GPU直接距离/缺失前向、Linux多进程及完整实验仍待服务器smoke。本轮没有执行它们。

## FP32 GPU LP：先验证精度，不自动恢复全量实验

用户服务器的原smoke已保存16个CPU解，外推约60.5小时并暂停。新增同模块的 `gpu_lp` 手动入口，仅用这16题检验现成NVIDIA cuOpt PDLP的FP32模式。正式 `runner`、CONFIG、HiGHS求解器、评分和12小时约定保持不变；没有自定义Triton。此入口不调用旧运行的freeze，不重写旧inputs、smoke、RMS、区域、pair或预测。

NVIDIA说明：`pdlp_precision=0` 在内部使用FP32，输入和输出接口为FP64。本入口保留原始面积和距离系数，回传后按原系数用FP64复核；这部分小规模检查不能改为先量化面积。设置强制PDLP、Stable3、单GPU、关闭crossover，避免默认并发模式用CPU/FP64解代替FP32结果。详见[官方精度设置](https://docs.nvidia.com/cuopt/user-guide/latest/convex-settings.html)。

每个问题分别核对原有行和1e-7、每列相对容量1e-6、W≥−1e-10、目标一致性及FREE下界1e-7尺度；另外与已封存CPU最优值比较，容差为1e-7×max(1,abs(reference))。不要求多解LP的W逐元素一致。求解器必须实际报告PDLP和Optimal；超时即使返回可行解也单列为未通过。内部停止设置保留primal 1e-10、dual/gap 1e-9，不使用cuOpt默认1e-4。两种算法的内部残差范数不同，不能仅因停止参数数字相同就声称精度等价，以上逐问题复核才是验收依据。纯FP32可能无法达到这些标准，届时保留失败，不放宽容差、不自动改混合精度或回退CPU。

默认8题组成块对角LP，以各独立t之和为目标；约束没有跨题连接，仍逐题验收。预热1批，正式2遍×16题。单批30秒求解上限，总计20分钟软上限（在批次之间判断，cuOpt自身可能略超时）。这只是有界测速，不是自动参数搜索。完成批次可按相同配置续用，吞吐引用原批次耗时，不把缓存读取速度当成求解速度。

手动步骤，代码同步后由用户在服务器执行。新环境只为求解器使用，不修改base或原yyk。下面安装是用户手动动作，本地实现过程没有执行安装。

```bash
source /home/ubun/miniconda3/etc/profile.d/conda.sh
conda create -p /home/ubun/YYK/conda_envs/yyk_lp python=3.12 pip -y
/home/ubun/YYK/conda_envs/yyk_lp/bin/python -m pip install \
  --extra-index-url https://pypi.nvidia.com \
  'cuopt-cu13==26.10.0' numpy scipy Pillow
```

先在原yyk中导出原16题（从10张已封存区域重建小距离矩阵，不读原特征、不重分区、不重新跑CPU LP）。GPU距离也只使用物理GPU1。默认阶段仅核查小记录，无Torch/cuOpt导入。

```bash
cd /home/ubun/YYK/DINO-Reference-AD
parent=$(cat /home/ubun/YYK/runs/latest_reverse_capacity_run.txt)
trial="${parent}_gpu_lp_fp32"
module=DINOv3.MADEqual.aebad_s_reverse_capacity_coverage_p0.gpu_lp
py=/home/ubun/YYK/conda_envs/yyk/bin/python

"$py" -B -u -m "$module" preflight --parent-run "$parent" --output-root "$trial"
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$py" -B -u -m "$module" export --parent-run "$parent" --output-root "$trial"
```

再用隔离环境仅运行GPU LP，单卡、最多8个CPU线程。可直接前台短测；不要由此启动原 `all`。

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/ubun/YYK/conda_envs/yyk_lp/bin/python -B -u -m "$module" benchmark \
  --output-root "$trial" --cpu-workers 8 --batch-size 8 --repeats 2 \
  --time-limit 30 --max-seconds 1200
cat "$trial/GPU_LP_summary.json"
```

输出为 `GPU_LP_checks.csv`、`GPU_LP_summary.json`、逐批小记录和约4MiB的16题距离/面积。仅全部32次检查通过，才给LP部分的待办耗时外推；不把未测的补特征、分区或最终评价当0，也不承诺全轮12小时内完成。导出距离未在旧pair中保存，不能声称有旧距离矩阵逐元素比对；复用原区域、原RMS、同实现/环境/后端重建，并核对原FREE和CPU目标。只覆盖background/seed42两张图，跨域数值与吞吐仍未验证。

检查仅包括原小记录封存与来源/环境一致、独立输出路径、单卡配置、稀疏LP和原数值容差；不扫描完整特征SHA，不新增RAM/显存/磁盘容量门槛。添加测速文件会改变正式入口的代码身份，旧输出不能直接用新版 `runner all` 续跑；本测速明确只读接入旧输出，正式后端接入及新运行复用方案待精度/速度结果review后确定。

新增CPU回归命令：

```powershell
& 'D:\anaconda\miniconda\envs\chinamoblie\python.exe' -B -m unittest DINOv3.tests.test_aebad_s_reverse_capacity_gpu_lp DINOv3.tests.test_aebad_s_reverse_capacity_coverage -v
```

### 后续固定混合精度对照

显式增加 `--precision mixed`，对应cuOpt `pdlp_precision=2`。矩阵/转置的SpMV存储使用FP32，向量、计算类型及收敛检查使用FP64；这不是纯FP32，也不保证比纯FP32更快。默认仍为 `--precision fp32`，不会自动尝试其他模式。除精度参数外，两种模式传给cuOpt的设置、原始系数、变量边界、LP及验收容差完全相同。

上次已导出的16题可通过 `--problem-root` 只读引用，不重新导出距离、RMS或区域。混合模式结果使用独立 `--output-root`，不能覆盖FP32的失败记录；同一结果目录切换精度时拒绝混用。原 `yyk_lp` 已装好时，无需安装/升级任何依赖。

代码同步后，仅运行下面这一项固定对照，仍为物理GPU1、最多8 CPU线程、预热1批和正式16题、单批30秒时限：

```bash
cd /home/ubun/YYK/DINO-Reference-AD
parent=$(cat /home/ubun/YYK/runs/latest_reverse_capacity_run.txt)
module=DINOv3.MADEqual.aebad_s_reverse_capacity_coverage_p0.gpu_lp
trial="${parent}_gpu_lp_mixed"

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/ubun/YYK/conda_envs/yyk_lp/bin/python -B -u -m "$module" benchmark \
  --problem-root "${parent}_gpu_lp_fp32" --output-root "$trial" \
  --precision mixed --batch-size 8 --repeats 1 --cpu-workers 8 \
  --time-limit 30 --max-seconds 300
cat "$trial/GPU_LP_summary.json"
```

汇总现在直接列出精度、纯数值检查通过数、行和/相对容量/目标一致性/CPU参考目标差的最大误差，区分数值通过与Optimal状态。仅全部验收通过才给LP吞吐；失败仍不外推全量。新增检查只有“只读问题目录与新输出隔离、精度绑定”，无大数组重复SHA或机器容量门槛。46项本地CPU测试通过；混合模式的实际GPU结果待用户执行，不属于已验证精度/性能。
