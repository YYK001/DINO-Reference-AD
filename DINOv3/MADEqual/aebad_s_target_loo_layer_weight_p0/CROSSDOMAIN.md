# illumination / view 固定跨域扩展

本轮沿用已成功的`numerics.py`，未修改正偏离、gamma、u/p/q、精度或插值。旧background入口和既有输入身份保留，原结果只读。新增域参数贯穿整个父资产链：`parent.py`显式按域读取SACP catalog、支持、预测manifest、CPU状态/指标、原AUPRO和实际NPY路径；没有修改盲点工具或切换模块全局变量。

## 固定任务

- illumination：265（67正常/198异常），view：364（84/280），名单来自原父manifest和有序K8，不按数量重抽。
- 原Source和每域完整Target8不变，Target统计为支持自身整图留一，Source+本域其余7支持。只补本域8折；不启动SACP方向拟合、盲点训练、邻域恢复或局部图。
- Source66统计从已完成background只读导入。核对Source bank seal、66图顺序、原提取配置、MAD seal、固定统计协议和线性分位数实现；`source_reuse.json`记录原path和seal。导入的记录不改写，绝不导入background Target统计或其gamma=.5权重。重新归约的gamma0权重必须与background精确相同。
- 每域独立run/cache/scratch、正常LOO/预测路径含域，输入/权重/预测/评分/地图/AUPRO/状态均绑定域；指标键至少domain+branch+scoring。
- 每域七分支：UNION、SEGMENT、SEGMENT_SCORE_UNION_MAP、两档独立加权、SEGMENT_SCORE_SOURCE_WEIGHT_MAP、SEGMENT_SCORE_TARGET_LOO_MAP。
- 两组合的CH1/CURRENT分数固定引用对应SEGMENT分数，定位引用gamma0/.5已有UNION加权图。不能归因于层加权的继承图像收益如实标明。组合不新建地图、不增加像素或AUPRO作业。独立加权仍用PL主读出和map-max补充。
- 两域只有四项新增AUPRO。每域14行指标、60项固定差值；缺强参照明确拒绝，不静默跳过。

## 本地盘点及时间

本地真实SACP小元数据读取成功，265/364和K8均确认；真实background正常统计、权重、已封存小结果回归通过。新的大数组未在本地持有。每域预检仍列出8份LOO距离、8份支持特征、265/364份父逐层预测和用于补检索的Source库实体不可用。这是本地未挂载服务器资产，不是已确认服务器丢失；GT压缩包本地存在。Source66统计已列为复用项，不列为待补检索。

保留原库、支持特征和父预测时，两个新域全流程暂估0.3–3小时（约18–180分钟），包含16折LOO、权重拟合、629图重放、两域完整CPU像素评价、4项原ADEval和汇总打包。background的302.83秒仅作吞吐参照；两域不是必然十分钟，磁盘/缓存命中/尺寸均影响时间。缺正常特征时补原前向时间另列未知和实际图数；缺逐层评价响应先寻找兼容父缓存，默认停止，不能从融合图反推或自动重建。缺Source库先报告，不自动重建。

协调脚本将两域顺序放在用户指定单卡上执行，不超8个CPU工人。两域通过`--budget-peer-run`累计彼此成本，12小时预算针对本轮新增两域合计，不给每域各12小时。预计明显超预算在已提交图/作业暂停，仅用户明确同意后加`--allow-over-12h`。无容量前置准入。

## 手动入口

本地实现和预检已完成；没有连接服务器、执行GPU或推送此次扩展。下面是交付后的手动命令，需先同步代码。

```bash
cd /home/ubun/YYK/DINO-Reference-AD
export RUN_ROOT=/home/ubun/YYK/runs/aebad_target_loo_crossdomain_seed42_$(date +%Y%m%d_%H%M%S)
export CACHE_ROOT=/home/ubun/YYK/storage/layer_weight_cache/$(basename "$RUN_ROOT")
export SOURCE_STATS_RUN=/home/ubun/YYK/runs/aebad_target_loo_weight_seed42_20261029_182735
script=DINOv3/MADEqual/aebad_s_target_loo_layer_weight_p0/scripts/run_crossdomain.sh
bash "$script" preflight
```

默认只预检。缺正常距离返回2并保留两域清单；阅读角色判断是否只需补16折LOO。没有此模块的正常距离时这是预期缺口，不代表缺模型或需要构库。

```bash
export GPU_INDEX=1  # 手动运行前由用户确认单卡
# 原支持特征、bank和逐层父预测均在时：
bash "$script" all --supplement-retrieval
# 实验日志可由tmux外层保存；每域独立logs/<stage>.log、stage_costs.csv。
```

需要后台时在tmux里执行同一脚本；既有tmux服务应显式传WORK_ROOT/PROJECT_ROOT/PYTHON/GPU_INDEX/RUN_ROOT/CACHE_ROOT/SOURCE_STATS_RUN。不自动关机或安装环境。只有明确要补缺失完整正常特征时再额外加`--extract-missing`，原host配置不兼容则拒绝，不迁就旧缓存。

单域入口（沿用协调脚本产生的目录才可续跑）：

```bash
DOMAIN=illumination RUN_ROOT="$RUN_ROOT/illumination" CACHE_ROOT="$CACHE_ROOT/illumination" \
  bash DINOv3/MADEqual/aebad_s_target_loo_layer_weight_p0/scripts/run_workstation.sh \
  all --supplement-retrieval
```

单域也可直接用`runner <stage> --domain illumination|view --source-stats-run <background>`，其他父路径/GT/cache同README。`--domain background`继续走原入口；不要对已完成background执行all。

已有正常统计会直接复用，即使原正常距离已离线；部分正常距离/逐图预测/AUPRO只补未完成项。Source提供者始终只读，不因新域run identity而改写旧manifest。

两域完成后仅重做汇总/打包：

```bash
bash "$script" aggregate
bash "$script" pack
```

顶层`summary/`只读取三个完成运行，不调用模型、检索或像素评价。background原10行独立结果不改，两个组合在顶层只读派生，明确标为“根据background开发结果提出”。两新域组合规则已进入各自冻结CONFIG，先于GT评价。

## 输出与完成条件

每域：input/provider、正常统计及Source_reuse、尺度/权重及下限、两个物理定位图与预测冻结、逐图分数、地图/GT身份、组合引用manifest、14行五项指标、60项差值、两份AUPRO、资源和前向次数。组合manifest仅引用现有地图哈希。

顶层：`domains_summary.csv`、`new_domains_macro.csv`、`three_domains_macro.csv`、`domain_weights.csv`、`metric_comparisons.csv`、`input_providers.json`、报告和status；包排除NPY/scratch/软链接。宏平均是各域指标等权，绝不合并629/926分数重算AUROC。

只有新629图及4项AUPRO有效、三个域共同Source条件/gamma0权重对齐、全部固定比较齐全，顶层才完成。缺项保留pending。全部域已有方法开发历史，不称盲测；未验证跨支持集稳定性。不按域择优，不扫参数；实例margin本轮不新增。

## 检查与验证边界

新增检查：域贯穿父读取、人数和支持排除、跨域身份及缓存隔离；共享Source依赖与gamma0一致；组合分数/地图引用一致；完整差值；共同Source可汇总；域等权；双域合计12小时。继承原有限性、FP32舍入界、地图/GT和评分1e-5、CPU封存、原ADEval协议及目录隔离。不增加大文件重复哈希或资源准入。

测试命令：

```bash
python -B -m unittest DINOv3.tests.test_aebad_s_target_loo_layer_weight DINOv3.tests.test_aebad_s_target_loo_layer_weight_domains -v
```

32项CPU测试通过（原20+新增12）；包括真实background小结果只读回归（本地持有时执行）、人工父元数据全链、两个域8份正常距离的统计复用和gamma0一致、预测中断续跑、七分支组合/差值、顶层pending与宏平均。没有真实新域大数组或CUDA/ADEval验证。真实background回归在代码仓库缺下载结果时明确skip，不以替身冒充。

最终本地验证记录：32项CPU测试通过，5.291秒；14份Python语法检查、两份Bash脚本语法检查通过。最新两域预检与顶层pending报告保存在DINOv3/outputs/mad_equal/aebad_target_loo_crossdomain_preflight_20261029。真实新增629图和4项AUPRO未执行。
