# SACP固定强度扩展：lambda=1

这是父SACP固定方向上的一次响应重放，新增三个分支，不重跑旧`all`。本轮交付仅本地代码、CPU夹具验证、缓存预检和手动命令；未连接服务器、发布代码或执行真实评价。

## 固定问题与任务

只比较父lambda=.5和新增lambda=1：强度能否扩大实际收益、定位代价是否放大、Source对照是否比普通Target方向及等量均匀缩距更有价值？不新增lambda、rank、层或门控，不按域选更好结果。

人口保持926：background 297（85正常/212异常）、illumination 265（67/198）、view 364（84/280）；父seed42每域原顺序K8，排除支持。全部层为L7/15/23/31。

|新增定位分支|唯一输入和计算|
|---|---|
|SACP_CONTRAST_L1|父raw_union与eta[0]|
|SACP_TARGET_DIR_L1|父raw_union与eta[1]|
|SACP_MATCHED_UNIFORM_L1|逐图逐层匹配本轮主方法实际总缩距|

**仅9项新CPU像素评价＋9项新AUPRO**。历史六分支及固定任务分工只读继承，旧像素排序/AUPRO不重算。Source方向没有新增lambda=1分支。

`protocol.py`独立定义任务表，不修改父`CONFIG`或全局分支列表。主表9分支，按3域＋等权宏平均、CH1/CURRENT两评分共72行；完整`metrics.csv`附父Source方向，共80行，其中56行历史、24行新增。行数从展示/任务清单派生，不套父56行。

## 父缓存接口和单位

显式`--parent-run`与`--parent-cache`，即便父`inputs.json`中旧cache、bank、权重路径已离线也不跟随它们。支持重定位后在新输出目录执行。

当前父资产位置：

```text
/home/ubun/YYK/runs/aebad_s_sacp_seed42_20261028_011454
/mnt/mechanical_disk/YYK/sacp_cache/aebad_s_sacp_seed42_20261028_011454
```

每图父目录：`<parent-cache>/predictions/<domain>/<sample_id>/`，需要原`manifest.json`与下列数组：

|数组|形状/类型|用途|
|---|---|---|
|raw_union|[4,N] FP32|原生欧氏/1280距离，不再除1280|
|raw_main|[4,N] FP32|重放父lambda=.5主方法核对|
|eta|[3,4,N] FP32|只消费前两轴，第三轴不参与新计算|
|positions|[N] int64|compact顺序到完整物理网格的排列|
|UNION_K8、SEGMENT_K8|[4,N] FP32|父z；精确无变化回退、UNION固定峰值及GT身份参照|
|SACP_CONTRAST、SACP_TARGET_DIR、SACP_MATCHED_UNIFORM|[4,N] FP32|父lambda=.5的z，仅作重放核对|

这些分支名数组是MAD后的z，不能当原生距离。原`raw_segment`、groups、NN索引、anchor标记、Source方向z/eta值不被本工具消费，不将它们添加为无用实体依赖。

eta轴顺序明确为CONTRAST、TARGET_DIR、SOURCE_DIR；支持的父预测源码SHA为`dcd4d2a3eb5867181b84e56c87abe3346572538eac27565f93b5e6175c7063d0`，与父`inputs.code['prediction.py']`核对。这是已读源码确认的模式标识，不重写父身份，不凭shape猜轴。

run元数据需要原inputs/protocol/catalog、predictions_frozen及三域predictions清单、input_distance_mad、完成status、指标，以及evaluation下逐图分数/实例/map_manifest/状态/指标和18份历史AUPRO小记录。`assets.json`只读取被封存的提取软件版本；**不访问其中Source/Q/特征/折库路径**。不需要checkpoint、骨干源码、原始Q、Source/Target实体、子空间向量或训练模型。

GT可用原AeBAD.zip，或只含`ground_truth/`的AeBAD_S目录（也可直接指向ground_truth）。不要求RGB原图或train目录。ZIP沿用原前缀`AeBAD/AeBAD_S/`。预检检查GT文件映射，评价才解码GT并用原字节身份/resize。

本地review包没有926份父预测数组；`local_preflight/required_cache_files.csv`和`missing_inputs.csv`保留完整工作区，不随GitHub发布。Windows无法访问服务器绝对路径不证明远程文件丢失。若raw_union/eta/所需manifest缺失，只报告缺项，不以区域均值、z逆变换或新推理补齐。

## 精确重放

沿用父FP32计算顺序：

```text
d_L1 = raw_union * sqrt(FP32(1) - FP32(1) * eta)
```

eta已经由父规则处理舍入，本轮检查前两轴有限且在[0,1]，不再裁剪或清洗。eta=0距离严格不变；eta=1产生0合法，不加下限。不使用“旧缩距乘2”。

均匀对照直接复用原`numerics.matched_uniform`：逐图逐层FP64求Delta、a，再`FP32(1-a)*raw_union`。a由lambda=1实际总下降新算，不能沿用/翻倍旧a。按原`eps32*sum(raw_union)`舍入界核对实际总缩距；零总距a=0。

新距离先用原`z_values`；凡`new_distance == raw_union`，复制父UNION z，保留父继承的精确回退，不通过z逆变换恢复距离。四层原FP32融合、位置映射、插值和CH1 peak/local image MAD完全继承；CURRENT仅补充。不拟合MAD、归一化、平滑或CLCR。

正式新增评价前，每图重放lambda=.5：4层raw_main、三个分支各4层z、三分支CH1/CURRENT分数，共22项记录。原生距离/z沿用父数组容差`atol=1e-7,rtol=1e-6`；逐图分数`atol=1e-5,rtol=0`。不放宽阈值。父.5重放只是验证，不进行旧分支像素重评。

## 阶段和边界

|阶段|行为|
|---|---|
|preflight（默认）|只读元数据/所需NPY头/位置/GT路径；不导入Torch、不创建运行目录，显式inventory-dir除外|
|replay|逐图核对.5后生成三张新z图；原始缓存不复制，全部926冻结后才允许GT评价|
|evaluate|原精确全像素CPU ROC/AP、图像AUROC/AP、实例及响应变化，仅给三个新分支分配排序scratch|
|evaluate-aupro|显式单GPU，原official_aupro_adeval，ADEval1.1.0、200阈值、FPR≤.3；9作业独立续跑|
|summarize|合并历史/新增及来源、差值、实例分布/损失转移、R_N/R_A/G、CURRENT补充和完成状态|
|pack|报告及小记录，排除父cache、新大预测数组和所有BIN/scratch|
|all|上述固定顺序，心跳/日志，成功或失败保留现场并打review包，不关机|

CPU命令安装Torch导入拦截器；不调用host/extraction、Exact、clustering/fit、训练。四个计数均为0，由API不接收这些资产、导入边界及夹具完整路径调用监测支持。AUPRO只加载指标后端，不构建检测模型。CPU可用不代表AUPRO环境可用，预检单独列出原后端可用性，缺依赖不自动安装或近似替代。

缓存按图提交、CPU按域提交、AUPRO按域分支提交；有效结果续跑不重算。全局重放提交同时绑定926清单及重放核对/原生缩距表，缺提交不读取GT。使用同一新run/cache及配置/父路径续跑，另行迁移则在新run显式传父位置。小结果已完成表与中间精确指标独立封存，缺seal/修改/范围或像素数不符拒绝。父/输入与新输出、scratch隔离，只清理本轮创建的临时目录/文件。

没有大缓存全量SHA、容量准入门槛或自定义Triton。默认最多8个CPU工人；大输出放机械盘，SSD私有scratch按域释放。新预测仅保存三分支z与positions及小审计信息，不复制全部父cache，不保存新的Q/NN/子空间。

## 运行命令（准备好后由用户手动执行，本轮未运行）

先review并另行授权发布代码，再在已有yyk环境的工作站执行。脚本默认只preflight，不安装/升级环境、不修改base、不自动关机。

```bash
cd /home/ubun/YYK/DINO-Reference-AD
export RUN_ROOT=/home/ubun/YYK/runs/aebad_s_sacp_strength_l1_v1
bash DINOv3/MADEqual/aebad_s_sacp_strength_p0/scripts/run_workstation.sh \
  preflight --inventory-dir "$RUN_ROOT/inventory"
```

CPU分阶段：

```bash
bash DINOv3/MADEqual/aebad_s_sacp_strength_p0/scripts/run_workstation.sh replay
bash DINOv3/MADEqual/aebad_s_sacp_strength_p0/scripts/run_workstation.sh evaluate
bash DINOv3/MADEqual/aebad_s_sacp_strength_p0/scripts/run_workstation.sh summarize
```

CPU完成后仍有9项AUPRO pending，显式授权GPU时执行：

```bash
GPU_INDEX=1 bash DINOv3/MADEqual/aebad_s_sacp_strength_p0/scripts/run_workstation.sh evaluate-aupro
bash DINOv3/MADEqual/aebad_s_sacp_strength_p0/scripts/run_workstation.sh summarize
bash DINOv3/MADEqual/aebad_s_sacp_strength_p0/scripts/run_workstation.sh pack
```

或授权后用tmux跑固定all（单GPU仅在AUPRO阶段使用）：

```bash
cd /home/ubun/YYK/DINO-Reference-AD
tag=aebad_s_sacp_strength_l1_$(date +%Y%m%d_%H%M%S)
run=/home/ubun/YYK/runs/$tag
mkdir -p /home/ubun/YYK/runs /home/ubun/YYK/tmp
touch "${run}_launcher.log"
tmux new-session -d -s "$tag" \
  "cd /home/ubun/YYK/DINO-Reference-AD && env GPU_INDEX=1 CPU_WORKERS=8 RUN_ROOT='$run' bash DINOv3/MADEqual/aebad_s_sacp_strength_p0/scripts/run_workstation.sh all > '${run}_launcher.log' 2>&1"
tail -n 40 -f "${run}_launcher.log"
```

## 输出与解释

- `protocol.json`、`job_plan.json`、`inputs.json`、共同`input_distance_mad.json`和新预测冻结清单。
- `replay_verification.csv`、`native_reductions.csv`：逐图逐层总原生下降，半强度总下降、相对下降均值及0/5/50/95/100%分位、均匀a/匹配误差。相对下降以raw_union为分母，零分母设0并计数。
- `metrics.csv`完整80行；`main_metrics.csv`72行；`historical_metric_sources.csv`明确56行复用来源；`metric_comparisons.csv`重点A—E；`strength_effects.csv`三新分支各自减对应半强度。
- `per_image_scores.csv`合并历史/新增及固定任务分工的来源；像素指标对两评分只算一次。
- `defect_instances.csv`、`margin_changes.csv`、`margin_summary.csv`：自身/固定UNION峰值、小实例与NA；两强度相对UNION和1减.5，受损/改善/基本不变计数与均值/中位/5%/95%，新增受损和原损伤继续恶化。
- `response_changes.csv`、`mechanism_summary.csv`：共享Source MAD最终响应尺度，正常图所有像素/异常图GT像素，先每图均值再图像等权，UNION减候选为R_N/R_A，G=R_N-R_A。历史值只读引用，不混用原生缩距。
- `costs/*.json`、`stage_costs.csv`、逐域分支`aupro/*.json`、`objective_report.md`、`status.json`与review包。报告汇总时的cost表不包含随后写出的本次summarize耗时，完整记录以costs为准。

所有9项新AUPRO有效、人口与重放/地图核对通过，才标all_five_metrics_complete。缺项保留pending，不填0；父Source方向保留附表，不新增Source方向GPU作业。父实例表仅在全局重放冻结之后解析用于评价，预检只核对其小文件封存。报告不因大强度更好就宣称所有子空间有效，也不因无优势就否定整个方向。

## 本地验证与全部检查清单

专项测试：`python -B -m unittest DINOv3.tests.test_aebad_s_sacp_strength -v`。实际记录见validation_results.json；测试原始日志与local_preflight只留完整工作区，不随GitHub上传。测试使用六张人工特征/GT夹具，覆盖原CPU评分、原精确像素、原实例内核和续跑；测试AUPRO元数据仅供pending/汇总/续跑验证，未执行真实GPU后端。

本轮16项CPU测试通过，13.631秒；10份Python源码编译及Bash语法通过。原父SACP生产Python与已完成父运行的记录逐份一致，未改动。真实926份缓存仍缺于本机，真实重放和9项新AUPRO均尚未执行。

保留且仅执行以下类别检查：
1. 父小JSON封存、完成状态、完整人口/支持排除、ETA源码模式和校准身份；assets仅作软件版本元数据。
2. 926份父bundle的manifest身份、所需NPY实际shape/dtype与物理positions；不读取未使用Source轴的数值、不重算已有父数组内容SHA。新产物首次写出记录自身数组SHA一次，续跑不反复重扫父内容。
3. 消费时raw_union/eta有限及合法，.5重放按原容差，L1均匀匹配沿用原舍入界。
4. 全部新预测冻结后评价；父UNION/SEGMENT地图/GT登记一致，新增同缓存评分与AUPRO地图身份一致。
5. 精确像素中间记录独立seal、有限范围/像素数，完成小表哈希及9项AUPRO完整性；没有历史全像素重评。
6. 输出/scratch与父资产隔离、续跑冻结配置及代码一致、私有临时文件清理、打包排除BIN。

本地目前只有父review元数据，未取回926份预测bundle；因此没有执行真实.5重放或lambda=1指标。服务器文件存在性仍待用户手动预检，不能据本机缺失清单宣称服务器丢失。不同本地NumPy/SciPy/Pillow也不冒充父环境真实复现。
