# Source–Target联合正常响应：单侧联合尾部建模

独立、固定配置验证，不再调整Target-LOO权重。主候选`JOINT_ST8_CF7`；对照`INDEPENDENT_ST8_CF7`、`JOINT_U4_CF7`、`UNION_CF7_SOURCE_WEIGHT`。三域×两组支持共24张新增定位图、最多24项AUPRO；两种图像读出不重复像素评价。默认只做preflight。

本轮只进行了本地CPU实现、夹具评价和输入盘点。没有连接服务器、初始化CUDA、提取真实特征、补真实检索、运行训练或正式评价；没有修改历史报告/结果。Git发布另依据用户后续授权；实际检测效果尚未知。

## 固定协议

沿用父`aebad_target_loo_two_K8_20261030_013433`的902张共同评价图：background 289（77/212）、illumination 257（59/198）、view 356（76/280）。Source seed42，Target支持seed42/43分别独立K8；两组支持并集均排除于评价，保留原sample_id。902张在两组下形成1804个预测实例，不是独立1804张。三个域均有开发历史。

每组第j支使用原Source完整100K/层和本组其余K7。正常支持t_j整图排除自身，与该支评价查询使用完全相同的B_j。八支均在原物理网格保留同一位置四层Source/Target响应，用同一原Source patch MAD标准化；原Exact欧氏距离已经除1280，读取时不再除；不截断负z，不重新拟合Target尺度或混入Source66响应。

每张支持按SHA256（版本、路径、物理y/x）哈希排名，从完整原生网格均匀无放回取512点，不足用全部，最终各图等权。四层和两路共用位置，不按响应/标签筛选。

固定h=1，以`logsigmoid(r-v)`和`logsumexp`稳定计算：

* 联合：每图`mean_r(A*B)`；
* 独立：每图`mean_r(A)*mean_r(B)`，四维Target向量块整体失配的解析全排列平均；
* U4：先逐层min，再计算四维单侧核；
* 三者都先对八张正常参照的概率等权平均，再负log，分别除8/8/4；不是平均八个负log；不使用概率epsilon截断。

CF7容量对照先对八支逐层z_UNION取FP32原生算术均值，再融合原Source u权重，最后原插值。新图均不再套旧MAD。

普通NumPy FP32是CPU路径；GPU路径只使用现成PyTorch `logsigmoid/logsumexp/logaddexp`，没有自定义Triton。查询/参照双分块，默认1024/512；八张正常参照批量处理，不逐张重复发射小算子，短图用log空间mask并保留各自点数，仍是八图等权。A/B同时服务联合和独立对照；逐图存四张原生地图，评价时只插值所需分支。小型FP64只用于测试参考。

## 先盘点实体

本地命令（无需Torch导入，缺失退出2）：

```powershell
$py = 'D:/anaconda/miniconda/envs/chinamoblie/python.exe'
$module = 'DINOv3.MADEqual.aebad_s_joint_reference_response_p0.runner'
& $py -B -u -m $module preflight `
  --run-root G:/codex/workspace/experiment/DINOv3/outputs/mad_equal/aebad_joint_reference_preflight_20261101_v5 `
  --cache-root G:/codex/workspace/experiment/DINOv3/outputs/mad_equal/aebad_joint_reference_preflight_20261101_v5_cache
```

实际本地盘点：两份review及902名单、48支持记录、MAD、Source权重、已验证历史指标/评分可读；GT zip索引可读。**1804份分路查询距离、1804份父定位层缓存、48份分路正常K7响应、48份支持完整特征和1份Source bank实体缺失，共3705条缺失记录。** 这些是本地状态，不能据此声称工作站的文件也已丢失。

本轮最终预检输出：`DINOv3/outputs/mad_equal/aebad_joint_reference_preflight_20261101_v5/`，包含`asset_inventory.csv`、`missing_inputs.csv`、`required_cache_files.csv`、`necessary_work.csv`、`normal_sampling_plan.csv`、`time_estimate.json`和pending报告/指标模板。v1–v4为开发中已生成的盘点，保留不覆盖。48张正常支持与采样规则已列出；实际物理坐标名单`normal_sampling.csv`在真实网格通过检查后由prepare生成，本地缺数组时不伪造坐标。对单设置重新prepare不会截断已完成的全局采样表。

接入顺序：

1. 优先读取父seed43紧凑baseline的`Source_distance[4,M]`、`target_distance[4,8,M]`、`positions`和原manifest；绑定原baseline_origin seal及Source/图像/提取/支持顺序。对seed42也检查这些字段；当前SACP生产者只保存融合响应，不能据此恢复两路。
2. 对同一查询，Source距离跨seed复用之前验证实际Source内容身份、图像SHA、提取配置及完整物理网格；positions变序时按物理索引重排。已有Source距离不重复检索。目标逐支持距离一次得到，全部K7复用。
3. 旧LOO若只保存UNION则列为不足。优先用已落盘支持特征补Source/K7检索。正常响应保存完整两路，后续仅取固定512位置；没有提取Source66的阶段。
4. query特征优先显式完整feature_manifest、two_K8/原始特征缓存、SACP资产指针；缺少时只有`--extract-missing`才允许原骨干完整前向。每张查询一次前向供两组共享，不持久化查询1280维特征；中断在尚未提交的图像中可能需要重复该图前向，并在成本中记录。
5. 如需新Source检索而原bank缺失，列出精确路径，恢复原`source_original`及manifest（四层合计约1.91GiB）后再继续；没有重建或替换Source的入口。两路距离已齐时不要求checkpoint或Source实体；只有Target补检索时References延迟加载Source，未使用的bank不强制打开。

迁移特征可用`--feature-manifest`：封存JSON有完整原`extraction`及`normal_features`或`records`字典，键为relative_path，值`{path,seal}`。bundle包含FP32 q7/q15/q23/q31、int64 positions和原image/sample、grid。复用原feature_identity.proof区分正常/查询格式及提取来源，不改写旧身份。

## 工作站手动阶段

下面仅是准备好的手动入口，须先发布并同步代码、获得本轮GPU执行授权后使用。现有yyk环境和YYK目录，无安装、base修改、远程连接或关机命令。一次只指定一张物理GPU，CPU最多8。

```bash
source /home/ubun/miniconda3/etc/profile.d/conda.sh
conda activate /home/ubun/YYK/conda_envs/yyk
cd /home/ubun/YYK/DINO-Reference-AD

export RUN_ROOT=/home/ubun/YYK/runs/aebad_joint_reference_seed42_43_v1
export CACHE_ROOT=/home/ubun/YYK/storage/joint_response_cache/aebad_joint_reference_seed42_43_v1
script=DINOv3/MADEqual/aebad_s_joint_reference_response_p0/scripts/run_workstation.sh

# 不导入Torch，只检查metadata、NPY头/位置和数据索引；按missing_inputs处理。
bash "$script" preflight
```

脚本默认父two_K8、SACP、两组cache、原始特征、AeBAD.zip及权重均指向本窗口YYK工作站的具体路径，不需要已消失的AutoDL缓存。可覆盖`PARENT_RUN/SACP_RUN/MEMORY42_ROOT/MEMORY43_ROOT/RAW_CACHE/SOURCE_BANK/SUPPORT_FEATURE_ROOT/FEATURE_MANIFEST/DATA/CHECKPOINT/PYTHON/EVALUATION_SCRATCH`。

授权后，先按盘点进行**有界正常资产准备和测速**，不要先启动全部查询：

```bash
# GPU_INDEX由用户确定；这里只展示变量，不能用逗号指定两张卡。
export GPU_INDEX=1

# 最多48张本组正常支持；已有完整分路响应直接复用。
# --extract-missing仅在确认完整特征缺失且允许补前向时添加。
bash "$script" capture-normals --supplement-retrieval --extract-missing
bash "$script" prepare

# 固定每设置前两张正常支持。缺少待办项的测速参数可省略。
# benchmark-forward/retrieval是明确计费的少量重复测速，不是正式预测。
bash "$script" smoke --benchmark-retrieval --benchmark-forward --benchmark-aupro
cat "$RUN_ROOT/time_estimate.json"
```

smoke的低维输入来自真实正常K7缓存；为测八支张量规模，将该正常图的Target-K7向量重复八次，仅为吞吐代理，不能当作正式八支预测。每设置还会做小规模CPU/CUDA核对，atol/rtol均1e-6，失败不放宽。Exact/骨干测速用固定正常支持和原实现。CPU排序使用周期构造标签、AUPRO使用构造矩形mask，**均不读测试GT、不输出正式指标**。记录代理性质，不能把这些当成正式后端一致性证明。对全部测试图的正式几何/地图身份仍在预测冻结和原GT评价中核对。

估时拆分缓存/核运算、待补Source/Target检索、最多一次共享查询前向、原生/分路距离存盘、图像解码、CPU NlogN排序、原ADEval和12张案例/打包。检索测速分开记录首轮参照上传和驻留后的逐查询吞吐，上传按bank/支持组计一次，不乘902。缓存已复用的任务不再计待办；无测量的阶段保持未知，不填0。AUPRO组件分布、测试图异质性及IO变动仍有不确定性，报告0.5–3倍规划区间，不承诺几小时完成。`all`、完整`capture`、`predict`在相应待执行范围估时未知时拒绝自动铺开；预计累计超过12h先停，只有用户明确同意后才加`--allow-over-12h`。这项时间边界来自本轮提示词，不是RAM/显存准入检查。

确认资产齐全、估时及执行权限后：

```bash
# 在tmux中执行。脚本不自行新建会话、同步Git或关闭机器。
# 若没有补算，去掉两个补算flag；已有有效距离自动复用。
bash "$script" all --supplement-retrieval --extract-missing
```

`all`：可选capture → preflight → prepare → predict → evaluate → aupro → cases → summarize → pack。`all-cpu`仅跳过AUPRO，GPU脚本中的预测仍需单卡；直接Python `--device cpu`可纯CPU低维计算，但不承诺可行速度。也可显式逐阶段执行，GT评价要求全部六设置预测已冻结。`--domains/--support-seeds`支持补某设置，不能改变902共同名单。

## 续跑、评价及输出

正常响应逐支持提交；查询分路距离、Source共享响应、四原生预测逐图提交；全像素CPU与AUPRO逐设置×地图提交。单次前向计数保存在costs，不放进不可变科学身份。半途已有共享Source不重写来源。旧manifest和父结果全程只读。

完整K8 min距离经原MAD后须复现父UNION，保持原1e-5标准化检查；原UNION、SEGMENT、Source加权的地图SHA及图像分数同时核对。失败保留现场，不放宽阈值或拼接新旧基线。每张新图最终地图/自身MAP_MAX先无GT冻结，正式评价从原生缓存复建并核对SHA。

* 表A：所有地图均配同组完整K8 SEGMENT冻结CH1，继承图像指标不算新模型贡献。
* 表B：所有地图均取最终地图最大值；原UNION CURRENT、Source加权WM是同类比较。原独立PL、CH1及两种历史组合另外原样保留于historical_metrics_read_only.csv。
* 每定位图只做一次原全像素精确CPU评价、一次原ADEval1.1.0/200/FPR≤0.3；不存在48项定位作业。
* 固定前两类案例为每设置原名单第一张正常和异常；GT仅作评价/轮廓，不选择困难样本，不展开margin归因。

主要输出：`inputs/protocol/common_manifest.json`、`normal_model.json`、`normal_sampling.csv`、缓存来源manifest、`predictions_frozen.json`、逐图冻结记录、`per_image_scores.csv`、两张指标表、`metrics.csv`、240项固定差值、每seed/每域/总体宏平均、只读历史指标、12张冻结案例、响应分布、`numerical_verification.csv`、`costs.csv`、`time_estimate.json`、`status.json`及`objective_report.md`。只在24项CPU/AUPRO、预测和案例核验全部完成时标记五指标完整；缺项pending/NA，不填0。

review仅打包小表、JSON、报告、日志、案例PNG。原生地图、距离、特征、模型、smoke大载荷、scratch不入包。新距离/native缓存留机械盘；SSD只作单设置×分支的独占评价scratch，每域最大约1GiB原始labels+scores，另需原精确排序RAM。按常见M=13924，若补齐两组全部距离和positions约3.56GiB、1804组四原生图约0.37GiB；实际补算应减去可复用部分，另有共享Source小缓存和manifest。若需补支持完整特征，48份最多另约12.75GiB，查询完整特征不持久化。这是按形状计算的规划量，不是已实测峰值或准入门槛。

## CPU验证及未验证边界

```powershell
& 'D:/anaconda/miniconda/envs/chinamoblie/python.exe' -B -m unittest `
  DINOv3.tests.test_aebad_s_joint_reference_response

# 可选：另一个独立进程，用已安装Torch执行CPU张量实现核对；不会初始化CUDA。
& 'D:/anaconda/miniconda/envs/chinamoblie/python.exe' -B -m `
  DINOv3.MADEqual.aebad_s_joint_reference_response_p0.verify_stock_cpu
```

覆盖固定配置、全图排除、两个支持组隔离、Source跨seed科学身份、min/MAD顺序、物理对应与采样、逐项单调性、极端log尾部、FP64小参考、块置换枚举、概率先汇总、不同图像patch数等权、CF7融合、小夹具24项CPU评价、GT冻结边界、读取旧父902元数据、续跑/部分capture、特征配置变化拒绝、scratch隔离、打包排除、部分prepare不截断采样表及上传/逐查询估时分离。96项CPU测试通过（其中36项为本模块），另3组现成Torch批量算子在CPU上核对通过，CUDA始终未初始化。最终计数和耗时见`validation_results.json`、`cpu_test_results.txt`与`torch_cpu_validation.json`。

真实CUDA低维算子、完整补检索/骨干、24项ADEval及完整1804预测均未在本轮执行。当前没有新效果结论；实现完成后交review，不自动调h、点数、组合或门控。
