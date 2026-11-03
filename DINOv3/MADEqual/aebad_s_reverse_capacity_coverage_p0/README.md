# 面积容量约束的反向覆盖：固定配置

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
