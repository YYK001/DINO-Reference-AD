# 查询图内共享残差平移：固定验证

状态：本地实现和CPU验证；等待review。没有授权或执行服务器连接、本地CUDA、真实特征提取、训练或正式评价。此目录是一套薄适配工具，不修改任何历史实验。

## 固定问题和公式

检验查询图内的参照残差均值能否改善真实扰动下的五项指标。不预设它是真实采集状态，不承诺缺陷保持或创新。

Source为原seed42、455训练/66校准、实际构库增强、每层100K原始bank。只用seed42每域完整K8；四层L7/15/23/31、1280维、原完整视野和Exact FP32。共同名单来自 `aebad_target_loo_two_K8_20261030_013433`：

|域|正常|异常|总数|
|---|---:|---:|---:|
|background|77|212|289|
|illumination|59|198|257|
|view|76|280|356|
|合计|212|690|902|

共同名单排除seed42/43支持的并集；seed43仅用于确认排除名单，不读取其特征、统计、权重或检索结果。这三个域有开发历史，不是盲测。

对每图、每层的每个唯一物理patch，从Source和8张支持分别取原始Exact Top1锚点。记残差 `r_s = q - m_s`、原UNION胜出来源 `s*`，`r_U=r_s*`，`delta_U=mean_i(r_U)`、`delta_s=mean_i(r_s)`。同距优先Source，再依支持顺序。

|分支|最终原始距离|
|---|---|
|MEAN_U_FIXED（A）|norm(r_U−delta_U)/1280|
|MEAN_U_RESELECT（B）|min_s norm(r_s−delta_U)/1280|
|MEAN_REF_RESELECT（C，固定主候选）|min_s norm(r_s−delta_s)/1280|

均值在除1280前以FP32求取。全图位置等权，包含当前被评分位置；不是盲点或独立估计。没有筛正常patch、GT、前景mask、跨图信息、截尾、聚类、门控、迭代或强度扫描。B/C始终只有9个原锚点，不对平移后的查询重搜memory；不与旧距离取min，允许距离升高。A→B拆分候选重选，B→C拆分分来源均值，不能假设C比B更小。

## 评分和评价

所有新分支继承原Source patch/image MAD，完全不重新拟合。不截负z，不做逐图min-max，不使用Target-LOO权重。

定位调用原 `target_loo_layer_weight_p0.numerics.weighted`：原生四层FP32、冻结Source u权重、再原FP32插值。CH1通过原 `statistics` 和 `ch1_score` 从各层候选响应独立取得峰值及原生valid3×3统计，绝不将PL评分当作CH1。

- 表A：候选自己的CH1与候选Source加权图；统一未修正参照为UNION CH1＋UNION_SOURCE_WEIGHT图。
- 表B：所有新图配同组SEGMENT CH1，比较现有SEGMENT CH1＋UNION_SOURCE_WEIGHT强组合。图像指标是继承值，不能算新收益。
- 补充：候选图最大值 `WEIGHTED_MAP_MAX`，不称CURRENT/CH1，不按域选读出。

保留历史UNION、SEGMENT、SEGMENT评分＋UNION图、Source加权图及其固定组合的原指标和来源。三种图像读出共享像素评价；3域×3张新图＝9个CPU定位作业、9个AUPRO作业。原五项指标、完整GT/resize、ADEval1.1.0/200阈值/FPR≤0.3不变。全部新预测无GT冻结后，才允许评价读GT。各域等权macro，不池化902张图。缺项保留pending/空值，不填0。

## 输入分级和复用

1. **完整query特征**：q7/q15/q23/q31、positions、grid及完整提取/图像/provider身份；只有logit、距离或地图不够。复用原 `joint_reference_response.runtime.features` 和 `feature_identity.proof`；支持 `metadata.image` 与 `metadata.sample` 两类既有schema。
2. **memory向量**：原Source `source_original` bundle，完整seed42支持特征。Source缺失只能恢复原库，工具没有重建入口；不需要Source锚点原图坐标/邻域。
3. **逐来源NN**：同一完整query/Source/支持顺序绑定的9×N距离和9×N索引，四层各一份。齐全时直接取回锚点，无需模型和Exact。只有Source/聚合Target索引不是9来源索引；有索引但不能证明对应查询向量和bank，不自动接纳。
4. **旧标量图**：只用于必要的原UNION/SEGMENT、评分及地图身份复核；不能倒推残差。

默认读取已信任本地NPY的manifest/header和实际使用的数值，不逐阶段重扫大文件SHA。新数组写入时记录一次SHA；小manifest/指标和逐图地图身份继续核对。旧manifest保持只读。检查全表见 `implementation_audit.md`。

原Exact的FP32矩阵求距与直接残差范数存在消减误差差异，逐图记录在arithmetic_audit，最终汇入arithmetic_path_comparisons.csv，不将该差异自动当作索引错误或“校验通过”。CPU/GPU同路径直接范数仍以atol1e-7、rtol1e-6核对；实际零修正后固定/重选图与父Source加权图、原UNION CH1仍以原atol1e-5核对。若真实地图/评分不兼容会停止，不能靠本修复强行复用父指标。三个候选公式和强度保持不变。

可用 `--feature-manifest` 指定现有封存完整特征导出。`--nn-provider` 支持本工具NN bundle的只读导出，格式如下（通过共享st.sealed_json封存）：

```python
{'records': {'background/test_0000': {'path': '/.../nearest/background/test_0000', 'seal': '...'}}}
```

bundle中有 `positions`、`distance7/index7` 等四层字段及metadata.contract。contract绑定query provider、完整extraction、Source seal、seed42有序K8和支持特征seals。不能只填路径伪造来源。现有其他格式若不满足这项证明，预检将其列为未确认，而非靠距离接近接纳。

查询缺失时，后续需显式 `--extract-missing` 才能原完整前向，每图最多一次，供三候选与基线共享；默认不落盘全部query特征。支持缺失同样显式补取、持久保存24份。索引缺失需显式 `--supplement-retrieval`，每图4×9＝36个原Exact，共用结果。不重新提取Source455/66，不训练，不重新校准。

## 本地盘点

最终本地预检及缺失清单位于：

`DINOv3/outputs/mad_equal/aebad_shared_residual_preflight_20261102_v5/`

该目录包括preflight、asset_inventory、missing_inputs、necessary_repetition、local_archive_inventory、server_confirmation和time_estimate。v5仅将cuda:0记为计划后端，用于验证元数据预检无需导入Torch或初始化CUDA；没有运行CUDA。原4份相关review包只盘点归档目录，没有重复解压大包。报告在≠向量在。

本地已确认：共同名单、seed42原分数/指标/GT身份、Source MAD/权重、小metadata，以及AeBAD.zip中的GT路径。当前指定供应路径缺902份完整query、24份完整支持特征、1份Source向量库、902份9来源NN及902份原标量预测bundle。服务器实体是否在不能靠本地报告断定，需用户手动运行preflight确认，不要求取回权重或Source大库到本地。

## 阶段命令（仅准备，尚未执行）

本地默认只预检，不导入Torch：

```powershell
python -B -m DINOv3.MADEqual.aebad_s_shared_residual_translation_p0.runner preflight --run-root DINOv3/outputs/mad_equal/shared_residual_inventory_new
```

使用项目已有Python≥3.10环境，不安装或升级环境。以下为已有工作站的人工命令；代码发布/服务器执行需后续授权。本脚本不自行git同步、挂载磁盘或安装依赖。

预测后端在首次预检时冻结：包括NumPy CPU／现成Torch CUDA、设备选择、FP32策略和软件版本。若计划用GPU，应在同一新运行的preflight前设置GPU_INDEX（或直接传 `--device cuda:0`）；这只记录计划，不导入Torch或初始化CUDA。CPU→CUDA等后端变更必须使用新的run/cache并重新做该后端smoke，不能混合续跑。旧版未记录后端的目录明确拒绝复用，不自动补写身份。完整特征和原memory仍可按各自身份只读复用。

CPU评价、CUDA AUPRO和汇总使用已冻结预测，不因其自身执行设备改变而改写预测后端；这些后处理阶段可继续引用原生产后端。

```bash
source /home/ubun/miniconda3/etc/profile.d/conda.sh
conda activate /home/ubun/YYK/conda_envs/yyk
cd /home/ubun/YYK/DINO-Reference-AD

export RUN_TAG=aebad_shared_residual_seed42_$(date +%Y%m%d_%H%M%S)
export RUN_ROOT=/home/ubun/YYK/runs/$RUN_TAG
export CACHE_ROOT=/home/ubun/YYK/storage/shared_residual_cache/$RUN_TAG
export GPU_INDEX=1  # 必须由用户确认的一张物理卡；程序内cuda:0
script=DINOv3/MADEqual/aebad_s_shared_residual_translation_p0/scripts/run_workstation.sh

bash "$script" preflight
# 缺失清单允许列出可以显式补算的query/NN；Source/旧标量baseline必须恢复。
# 真实少量完整图测速：每域固定首张，共3张；包含真实特征/原Exact、取向量、修正及写盘。
bash "$script" smoke --supplement-retrieval --extract-missing --benchmark-metrics
cat "$RUN_ROOT/time_estimate.json"
```

smoke原本已存在的query/NN会复用，三张正式候选产物直接供predict续用。中断后复用已提交的逐域完整图测速；若某图先前已完成但没有完整测速记录，按名单选择首张尚未完成图，避免重复前向。全图均值不能用少量patch均值代替；CPU与现成Torch核对因此使用完整实际网格。测速无需GT，CPU/AUPRO可选代理mask只测耗时并丢弃指标；不冒充正式评价。缓存齐全时不加载checkpoint或骨干。新提取需复用原环境和checkpoint。

通过review、测速和必要授权后，人工后台执行：

```bash
tmux new-session -d -s "$RUN_TAG" \
  "cd /home/ubun/YYK/DINO-Reference-AD && env RUN_TAG='$RUN_TAG' RUN_ROOT='$RUN_ROOT' CACHE_ROOT='$CACHE_ROOT' GPU_INDEX='$GPU_INDEX' bash '$script' all --supplement-retrieval --extract-missing > '${RUN_ROOT}_launcher.log' 2>&1"
tail -n 40 -F "${RUN_ROOT}_launcher.log"
```

`all`只执行predict→evaluate→aupro→summarize→pack，不隐藏运行smoke、重新拟合或扩域。各阶段也可单独调用同一脚本。相同run/cache/profile续跑只补未完成图/评价，已完成结果验证小封存后复用。已有GPU数值smoke可复用；默认preflight无需GPU。任何12h超限需用户显式同意后才传 `--allow-over-12h`；该flag不代表本轮授权。没有关机或第二卡逻辑。

## 成本与资源：尚未做真实测速

|阶段|完整query＋NN齐全|缺query/NN|
|---|---|---|
|补特征|0前向|最多902查询＋缺失的24支持，原完整前向|
|原Exact|0|最多32472次查询检索；无Source重建|
|向量取回、均值、三修正、写盘|902张全量工作|同左，共用9候选，不三倍检索|
|CPU精确定位|9作业|9作业|
|原ADEval|9作业|9作业|
|汇总打包|小报告|小报告|

真实前向/检索/IO和残差修正时间当前未知，time_estimate的未知项是null，不当0。最近联合响应运行902前向＋Target检索记录约51分钟，仅作不同工作的历史参照；本轮还可能需要完整Source检索，不能套用该时间。父24定位作业CPU约610秒、AUPRO约204秒，本轮9作业粗比例约229/76秒，仅是参考，分数分布、连通区域及磁盘会改变时间，不是实测承诺。全轮时间必须按上述实际完整图smoke更新。

按13924物理patch估算新增紧凑缓存约5–8GiB（主要为四层9来源int64索引与FP32距离）；不会保存所有1280维残差。完整query全部持久化约240GiB，默认禁用。已有Source/支持向量额外保留，不重复复制。RAM规划6–18GiB，VRAM规划6–20GiB，CPU临时盘规划3–12GiB，均未验证且不是资源准入门槛。参照GPU常驻缓存8GiB有界；Source/Target分层取向量，原Exact自身分块。不新增全局FP64、BF16、TF32、自定义Triton或混合精度配置。CPU上限8、BLAS线程1。

smoke将冷启动、真实完整图处理和额外一致性验证分别记录；不把CPU构造数据当真实模型吞吐。计划>12h先停等用户同意，运行中超限在已提交单位后停并保留断点，不减少图/支持/指标。

## 交付与结果边界

正式产物：metrics.csv、metric_comparisons.csv、per_image_scores.csv、diagnostics.csv、source_mean_diagnostics.csv、numerical_verification.csv、每图timings和各阶段costs、每域9项CPU/AUPRO独立完成记录、objective_report.md、status.json及小review包。delta向量、来源切换和9来源索引在紧凑NPY缓存中；零分母独立标记，ratio为NA。review包排除scratch、大数组、checkpoint和软链接。

判断以C及全部A/B五项指标和三域取舍为准；同时给B−A、C−B及相对Source强参照差值。候选自身检测与固定任务分工明确分表。无结果前保持pending；不因距离下降就宣称有效，不根据结果自动延伸方法或否定其他充分训练方案。

CPU测试可在已有环境执行，不初始化CUDA：

```bash
python -B -m unittest DINOv3.tests.test_aebad_s_shared_residual_translation -v
python -B -m DINOv3.MADEqual.aebad_s_shared_residual_translation_p0.verify_stock_cpu
```

第二条仅使用已安装Torch的CPU设备。源码仓库不包含父review包时，真实metadata测试类明确skip；数值、续跑、后端绑定及小夹具评价仍执行。本地验证日志和validation_results.json不上传GitHub。完整结果与尚未验证项见implementation_audit.md；正式服务器执行需要用户授权。
