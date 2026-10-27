# SACP 实现审计（本地交付，2026-10-27）

状态：代码/CPU验证与本地资产预检；未执行真实926张提取、子空间拟合、CUDA Exact或ADEval。无服务器连接、依赖安装、Git提交/推送。本轮不修改历史代码/结果。父review中仅提取原有四份根元数据到server_results供只读盘点，已有文件不覆盖。

## 实现分工

|文件|职责|
|---|---|
|protocol.py|唯一固定配置与派生任务：926图、12拟合作业、6图、18逻辑AUPRO、0训练|
|inputs.py|近期原始资产只读身份/shape/位置接入，16图对折控制，缺失与重复计算清单|
|clustering.py|正式Source32类、固定抽样/单次初始化/有界空簇策略，不修改bank|
|numerics.py|图像均衡未中心化二阶矩、随机化公共表示、广义特征白化与欧氏QR、三方向共同回退及均匀对照|
|runtime.py|按bank/层的已有Exact/常驻缓存适配、兼容对折索引复用、12作业提交|
|prediction.py|六分支共享单次Q及Exact/锚点/Source分组；共享冻结MAD，父基线原容差复核，926冻结|
|evaluation.py|既有精确CPU/AP与原ADEval、同一最终地图身份、实例/eta附表、私有scratch|
|reporting.py|CH1/CURRENT、三域等权、固定任务分工、所有对照/NA、真实margin和成本/完成状态|
|runner.py / scripts/run_workstation.sh|默认只预检，分阶段/续跑/tmux手动入口和排除大临时文件的打包|

直接复用的真实接口：

- `reference_residual_segmentation.execution.ResidentExact` → 原 `superadd_host._exact_top1_chunked`，原FP32 cdist/chunk/tie及除1280；`retrieval.target_nearest`逐支持同一路径。
- `reference_residual_segmentation.runtime.host/extraction`、`normal_correspondence_adaptation.calibration.z_values/response`：原宿主/四层FP32融合/插值/CH1，不调用训练流水线。
- `target_local_segment.geometry.geometry_layer`：原Top2支持几何，不复刻近似SEGMENT。
- `support_witness_segment_p0.evaluation.exact_pixel_result/validate_exact_pixels`：原全像素精确CPU及独立中间seal、有限范围/像素数。
- `normal_correspondence_adaptation.evaluation.aupro`：仅用独立命名空间绑定本工具输入/分支，不改旧全局列表、不把SACP伪装成PAIR；继续调用`official_aupro_adeval(nstrips=200)`。
- `segment_pair_crossdomain_p0.analysis.spatial`、原`mark_small`和`reporting.margins`：实际自身/固定UNION峰值与NA；这里只修正文字单位为共享Source MAD尺度。

## 原方案修正

`新建文件夹/code_artifact.md`仅作思路来源。本版使用正式欧氏/1280输出；不以平方距离替换，不把eta区域差当margin，不用诊断AUC阈值证伪方法。未中心化统计保留整体均值；广义向量映射后做欧氏QR。Source对照来自构库A/B各8图，原66校准独立；无GT筛选或互近邻覆盖过滤。文献/创新陈述不照搬未经核实内容。

统一有效组和秩：每类每折至少4图，gamma>1方向≤2，任一普通方向对照数值秩不足则全部共同UNION回退。共享P仅表示残差统计，不投影查询，不在修正后重新挑锚点。Source方向对照仍有K8 Target信息。

均匀对照使用FP64总量和FP32(1-a)乘原UNION，匹配误差单独记录并按预先固定eps32×sum_UNION舍入界检查，不调门控/强度以追求位相或GT匹配。

## 资产与效率边界

仅使用迁移后reference-residual父运行。正式Source、A/B折库、Source16及Target24完整网格、原MAD均直接只读引用；不复制百GB特征、不修改旧manifest。缺折库阻断，无自动重建。正常训练NN仅在同查询特征seal、同bank seal和物理位置映射下可复用；分割模型/优化器/合成样本均不加载。

父stream运行未在review中保存926完整Q；如服务器亦无另存，需926次原宿主前向。每图共用四层Source+逐支持Target检索，SEGMENT与SACP六分支共用。旧aggregate索引只有通过原bank/支持/几何及距离复现才保留；缺逐支持结果不能谎称免检索。拟合最多256次bank搜索（64 Source可兼容复用，192 Target），没有正常图重新前向。

统计按层/折库组织；Source对应在三域共用。每层最多暂存40×4096×1280 FP32选中查询，group cap后的矩阵小于全网格；无1280²大协方差。GPU原Exact常驻LRU默认6GiB；大矩阵FP32，小型统计FP64，没有自定义Triton，单GPU，CPU≤8。正式Q默认stream不保存，真实吞吐/峰值尚未知。

## 检查、续跑、保护

README列出全部检查。大数组读取为可信本地快速模式：manifest、原身份、NPY头和小位置数组；不做全量SHA/容量准入。小JSON/config/结果表封存、一次地图/GT哈希和关键数值核对仍保留；不宣称能查出所有未改manifest的同shape大数组内容篡改。

阶段提交粒度为对应文件/域层拟合/图像预测/域CPU/域分支AUPRO。全部926预测冻结前拒绝读GT。原距离容差1e-7+1e-6相对、同缓存评分1e-5绝对，没有放宽。父基线地图、GT和人口如冲突则报错，不悄悄替换历史结果；旧指标只有身份与小结果封存通过才复用。

scratch纳入父资产和新缓存目录隔离，临时目录专属，只删本轮创建文件；review打包排除scratch/BIN、原Q、bank、预测大数组。未完成AUPRO为pending不填0。两个图像评分只评价一份像素图，固定任务分工不新增图。统计单次固定K8，不称跨种子稳定。

## CPU验证边界

测试文件：`DINOv3/tests/test_aebad_s_sacp.py`。实际结果和环境见`validation_results.json`；`validation/cpu_tests.log`及`local_preflight/`仅保留于完整工作区，不随GitHub发布。数学测试含未中心化常量偏移、广义解独立参考、QR/rank/eta/共同回退、严格UNION、均匀量和原物理坐标/对折身份。

最终26项CPU测试通过（27.974秒），11份Python编译通过，Bash脚本语法通过。包含AUPRO真实地图/GT身份适配、续跑结果表修改拒绝、scratch保护和打包排除回归；没有调用真实ADEval后端。

集成使用原Torch **CPU** Exact、真实小型子空间拟合、六分支打分、原CPU全像素指标和实例内核。替代范围是资产清单、64行而非100K的零Source bank、16patch网格、6张合成特征查询和人工GT；全926数量检查在该集成fixture内明确替为每域2图。另生成仅供汇总状态测试的18份AUPRO记录，不执行ADEval、不将其保存为实验结果。真实父UNION/SEGMENT复现仍需服务器真实缓存，CPU夹具不是实机复现声明。

本地preflight使用真实父catalog/输入元数据；服务器路径在本机不可访问，依赖/版本也不完全相同。不能据此判断远程文件丢失，或宣布CUDA/完整DINOv3/真实SACP/原ADEval已通过。交付到此停在review，实际优势待完整五项指标及实例结果回答。
