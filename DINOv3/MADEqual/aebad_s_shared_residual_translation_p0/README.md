# 查询图内共享残差平移：同口径直接范数参照 v2

用户已授权三域直接范数参照及其3项CPU＋3项AUPRO，并继续既定实验。服务器命令由用户手动执行；本地没有连接服务器、初始化CUDA或执行真实提取/正式评价。历史结果只读保留。

## 固定数据与候选

原seed42增强Source、455训练/66校准、每层100K、完整K8、L7/15/23/31、1280维及原完整视野保持不变。共同名单来自两K8实验，排除seed42/43支持并集。seed43仅用于确认排除名单，不读取其特征或统计。

|域|正常|异常|合计|
|---|---:|---:|---:|
|background|77|212|289|
|illumination|59|198|257|
|view|76|280|356|
|合计|212|690|902|

三域有开发历史，不是盲测集。每图每层使用全部唯一物理patch，包括正在评分的位置，不是盲点估计。

原Exact FP32从Source和8张支持各取一个Top1锚点，9个候选保持不变。记 `r_s=q-m_s`。在**除1280前**用FP32直接范数选择 `s*=argmin_s norm(r_s)`，同距按Source、原支持顺序优先。令 `r_U=r_s*`、`delta_U=mean_i(r_U)`、`delta_s=mean_i(r_s)`。

|分支|原始距离|
|---|---|
|UNION_DIRECT_SOURCE_WEIGHT（新未修正参照）|min_s norm(r_s)/1280|
|MEAN_U_FIXED（A）|norm(r_U-delta_U)/1280|
|MEAN_U_RESELECT（B）|min_s norm(r_s-delta_U)/1280|
|MEAN_REF_RESELECT（C，预定主候选）|min_s norm(r_s-delta_s)/1280|

均值在原特征单位以FP32求取，最终距离只除一次1280。A固定锚点及A/B共享均值采用新直接范数胜出来源。候选公式和强度不变，但新旧胜出来源不同时A/B数值可变化；不声称与旧路径逐元素不变。B/C只重选这9个原锚点，不对平移query全库重搜。没有GT筛点、截尾、门控、参数搜索、与旧距离取min或重新校准；允许距离升高，不假设C≤B。

## 评分、评价与数值审计

原Source patch/image MAD及Source层权重不变。定位复用原native四层FP32加权及FP32插值；CH1独立复用statistics/ch1_score，不把地图最大值当CH1。不截负z、不逐图归一化。

- 表A：候选自身CH1＋自身Source加权图，主要对照新直接范数参照的CH1＋Source加权图。
- 表B：原SEGMENT CH1＋候选图，主要对照 `SEGMENT_SCORE_DIRECT_SOURCE_WEIGHT_MAP`（同一SEGMENT CH1＋新直接范数参照图）。该组合直接引用参照图，额外地图/AUPRO为0；图像指标全部继承。
- 补充统一报告WEIGHTED_MAP_MAX，不改称CURRENT、不按域挑读出。

原UNION、SEGMENT、Source加权及固定组合原指标保留，标为历史MM路径。新增参照独立计算五项指标；新旧变化只计入数值路径审计，不计作方法收益。

定位与像素评价共 **3域×4张图=12项CPU＋12项AUPRO**（候选9、参照3），三种读出共享像素/AUPRO。保留原完整GT/resize、精确CPU像素排序、ADEval1.1.0、200阈值、FPR≤0.3，不改后端或采样。所有预测无GT冻结后才允许正式评价读GT；逐域报告及三域等权macro，不池化902张图。缺项保持pending，不填0。

数值核查保持：旧Exact UNION→旧z/地图/评分仍按原检查；**零修正固定/重选图及CH1只对照新直接范数参照，atol仍1e-5**。CPU/现成Torch直接范数对照仍为atol1e-7、rtol1e-6。CPU验证在原raw容差内核对生产后端的直接范数胜出锚点，再使用同一集合复核全图均值；正式路径自行按直接范数选择，不读取CPU选择。这样避免CPU/CUDA近同距选择不同集合后误比较不同均值。

逐图逐层保存原MM与直接范数距离差、同旧锚点误差、胜出来源切换、固定/重选零修正结果和新旧地图/评分差。它们是描述性差异，不能称跨路径兼容通过。原容差下真实同路径错误仍停止并保留现场。

## 资产复用与检查边界

需要完整query特征、原Source向量、24张seed42完整支持、四层有序9来源NN及旧标量预测。仅有距离/地图不能倒推残差。支持metadata.image和metadata.sample两种既有完整特征格式；feature_identity.proof绑定图像、checkpoint、预处理、精度和provider。NN contract绑定完整query、Source seal、有序支持及物理位置。

完整兼容输入可只读复用，无需模型；缺query/支持需显式 `--extract-missing`，缺NN需 `--supplement-retrieval`。每图最多一次完整query前向、36次原Exact，四张图共享，不四倍检索。不重建Source、不重新提取455/66、不训练。Source/旧标量缺失时需恢复，工具不自动重建。

默认trusted-local fast：大数组消费时读manifest/header及所需值，不逐阶段重新扫描SHA；新数组写入时记一次SHA。小封存、预测/地图身份和范围检查保留。完整检查清单见implementation_audit.md。无RAM/显存/磁盘容量前置阻断、自动安装、关机、第二卡或自定义Triton。

新v2使用**新run/cache**，旧失败目录不迁移、不重写。原始memory/完整特征按自身依赖复用。失败首图未持久化完整q时可能需再前向/检索；不伪称旧NN已被证明对应新q。预测后端在首次preflight冻结；CPU/CUDA切换需新目录及新smoke。CPU评价/AUPRO执行设备与冻结预测生产后端分开记录。

本地v2盘点：`DINOv3/outputs/mad_equal/aebad_shared_residual_preflight_20261102_v7/`。当前本地指定路径缺Source向量1、支持24、完整query902、旧标量902、9来源NN902。父小metadata/名单/分数/MAD/Source权重及本地GT路径索引可用；4份review包仅复用归档索引，没有重复解压。服务器实体是否齐全需用户手动preflight，本地缺失不等于服务器已丢失。

## 手动工作站入口

只用现有yyk环境，物理GPU1、最多8 CPU。脚本默认preflight，不自行git同步、挂载或安装。GPU_INDEX在新目录首次preflight前指定，仅记录计划，不初始化CUDA。

```bash
source /home/ubun/miniconda3/etc/profile.d/conda.sh
conda activate /home/ubun/YYK/conda_envs/yyk
cd /home/ubun/YYK/DINO-Reference-AD
export RUN_TAG=aebad_shared_residual_directref_$(date +%Y%m%d_%H%M%S)
export RUN_ROOT=/home/ubun/YYK/runs/$RUN_TAG
export CACHE_ROOT=/home/ubun/YYK/storage/shared_residual_cache/$RUN_TAG
export GPU_INDEX=1
script=DINOv3/MADEqual/aebad_s_shared_residual_translation_p0/scripts/run_workstation.sh
bash "$script" preflight
# exit2可表示尚待显式补算的query/NN；先看缺项分类，Source/旧标量必须在。
bash "$script" smoke --supplement-retrieval --extract-missing --benchmark-metrics
cat "$RUN_ROOT/time_estimate.json"
bash "$script" all --supplement-retrieval --extract-missing
```

正式命令应置于tmux；相同run/cache可续跑。all执行predict→evaluate→aupro→summarize→pack，不隐藏扩域、拟合或参数扫描。smoke每域按固定名单首张完整图，共3张，已提交预测供predict复用；可按逐域smoke记录续跑。没有重复的模型前向用于测速。CPU核对用完整网格均值，不能用小patch子集均值替代。

预算必须含新增参照3项CPU＋3项AUPRO。真实完整图smoke外推前向/原Exact/向量修正/写盘，额外CPU数值验证单列、不乘902。可选代理mask用原CPU/AUPRO仅测速，结果丢弃，不当正式指标；CPU按NlogN、AUPRO按像素外推，真实连通区域仍不确定。未知项为null，不当0。已提交资产不计入待办预算。预计或累计超过12h停止保留断点，须新获同意才可传 `--allow-over-12h`；用户本次未授权超限。

## 成本及输出

原始query/NN齐全时零模型前向/检索；否则最多902个query＋缺失支持前向和32472次原Exact。902图统一处理三候选与一个直接范数参照。评价12 CPU＋12 AUPRO；组合0额外任务。

未做真实v2测速，不承诺总时长。父24定位CPU约610秒、AUPRO约204秒，12作业粗比例约305/102秒，只是不同运行历史参考。总时间以真实smoke为准。按13924 patch规划新增紧凑缓存8–11GiB；不保存9×N×1280残差。若持久化902完整query约240GiB，默认关闭。RAM6–18GiB、VRAM6–20GiB、评价scratch3–12GiB仅规划未实测，不是容量准入条件。参照GPU缓存8GiB有界，Exact原分块，FP32现成Torch，不引入TF32/BF16/全流程FP64。

输出metrics.csv、metric_comparisons.csv、objective_report.md、逐图分数、诊断、timings/costs、资源记录及状态。另输出：

- arithmetic_path_comparisons.csv：旧MM/直接范数及同锚点距离差。
- source_selection_switches.csv：domain＋sample_id＋layer的新旧胜出来源切换数量/比例。
- reference_map_score_changes.csv：逐图新旧参照地图/CH1差。
- numerical_path_metric_changes.csv：逐域及macro新旧参照五项指标变化。
- 每域每图独立预测身份，12项CPU/AUPRO完成记录；组合不新增记录。

review包排除大数组、scratch、checkpoint和软链接。主判断看C及A/B的完整五项指标、B−A/C−B、三域取舍与同口径强参照。小距离不等于检测收益，不保证缺陷保持，不自动追加参数或规则。

本地CPU验证，不初始化CUDA：

```bash
python -B -m unittest DINOv3.tests.test_aebad_s_shared_residual_translation -v
python -B -m DINOv3.MADEqual.aebad_s_shared_residual_translation_p0.verify_stock_cpu
```

Torch CPU核验独立进程执行。源码仓库不含父小metadata时该测试类明确skip；夹具评价和模拟AUPRO提交不代表真实ADEval实验。本地日志、validation_results和preflight不上传Git。真实CUDA、3张完整smoke、902张预测及12项正式评价尚未验证。
