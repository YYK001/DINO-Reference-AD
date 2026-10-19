# 实现审计：AeBAD-S 固定K8支持集敏感性

2026-10-19。仅新增独立模块、专项测试和输入盘点；旧模块、历史结果及已有报告不修改。本轮不连接服务器，不执行真实模型、特征提取、全域实验或GPU评价，不安装依赖。

新增文件：`__init__.py`、`selection.py`、`inputs.py`、`storage.py`、`scheduler.py`、`capture.py`、`prediction.py`、`evaluation.py`、`reporting.py`、`runner.py`；`background_identity.json`、`input_inventory_local.json`、`README.md`和本文。专项测试为`DINOv3/tests/test_aebad_s_k8_support_sensitivity.py`。

## 调用链与边界

| 环节 | 实际复用入口 | 新工具职责 |
|---|---|---|
| 支持选择 | aebad_s_target_fewshot/support.py: relative_path、stable_support_key | AST原样编译纯函数避开顶层Torch；seed42读取原manifest，43/44显式seed；不调用旧select_target_support |
| Source身份 | 正式memory_manifest、source_mad_state、calibration_stats及原冻结catalog中的SHA | 保持Official brightness=[0.8,1.2]，455+66，四层各100000；不构库/拟合 |
| 支持特征 | robustad_target_fewshot_p0.support._extract_four_layers → ShiftADD/aebad_s_p0/memory.py | brightness_factor=None原完整支持提取；唯一图像一次，按层存原始FP32 |
| 查询提取 | superadd_host.image_tensor、_extract_processed_batch | 与observe_fewshot_pil同顺序：preprocessing、query_masks=[None]、include_identity_descriptors=False；一次完整网格供三组 |
| Exact | superadd_host._exact_top1_chunked | 每层Source一次，唯一支持逐图一次；不调用恰好8张的旧扫描器来处理24张，不改cdist配置 |
| 端点 | aebad_s_target_local_segment.geometry.geometry_layer | 按组原序切8张；稳定Top2不同图，FP64边长、alpha、退化与修正沿用 |
| PAIR | aebad_s_segment_pair_crossdomain_p0.method.predict/pair_gate | 物理3×3、排中心、ceil(n/2)、无邻居拒绝、局部编号1—8；不加坐标门控 |
| 评分/地图 | ch1_scoring_inheritance.statistics/ch1_score；neighborhood.analysis.score_maps | 所有组同一patch/image MAD，CH1主评分，CURRENT仅seed42审计 |
| CPU指标 | projection_layer_diagnostic.metrics.exact_binary、CH1.image_metrics | 共同子集完整像素、包含正常图零mask，精确排序AP，无量化/采样 |
| AUPRO | relation_reliability.official_metrics.official_aupro_adeval | 独立懒加载CUDA，ADEval1.1.0、200、FPR0.3，CPU地图hash核查 |
| 空间分析 | crossdomain.analysis.spatial/mechanisms/group_small_instances/margin_repairs/mechanism_summary | 按seed包装；同一GT实例清单、每域小四分位；不用旧二域macro/固定catalog |

原来“恰好8张”限制在scoring的两个支持扫描函数；原observe_fewshot_pil提取的pending.queries本身不依赖支持组数。因此只扩展提取之后的外层扫描与切组，不改宿主、切窗、拼接、精度或数学。不同支持网格只影响该支持bank行数；PAIR只使用查询物理positions与局部支持图对，不要求支持网格同形。

## 身份和可复用证据

正式六项文件及Source image MAD沿用已有固定SHA。完整测试路径排序先生成原ID再筛三域，与正式逐图CSV核对；42支持完全继承原序。当前ZIP重新列举得到281/249/348共同评价图，三组支持自然两两不重合。

`background_identity.json`由已验证的1,254,584,320字节LOO B/SEGMENT档案派生，父archive SHA为`6fd4086bbaeb5318a5168dd90eef57d9955ffa37da44764db926913c3e4b2e41`，capture inputs SHA为`dbc394cc5e5e09ceb1cb4b1e37cbafbb5a4361319bcf55c0176ec5e99496ea84`。只提取原305张RGB与212张GT身份，不迁移旧结果或修改旧输入。另两域从已冻结raster catalog读身份，**不调用它的固定范围执行包装器**。

prepare逐图核对RGB/GT字节并冻结。默认preflight只读名字/图像头、正式状态及资产哈希，字节验证范围在inventory中明确记录。每个cache/预测/评价文件在首次完成时记录SHA；续跑先校验历史，禁止先重置状态再认可新hash。孤立/临时文件保留并拒绝。每张query及评价组合提交为不可覆盖单元，CPU并行有界、主线程按输入序提交。

旧background完整距离/线段及本地PAIR预测实际存在，作为CPU回归；跨域成功review实际是汇总，不是629张完整raw。旧target_raw_8已经合入Source，不能生成新支持集合的Top1。新工具不自动用未知外部Q/支持目录补值；当前仅同一新运行的冻结缓存可续跑，无来源文件不会登记为可信。

历史Source/Union CURRENT仅seed42复核strict1e-5、compatibility1e-4，分开保留和有符号误差；内部容差不变。共同子集重新算所有27组合指标，并额外对相同子集历史CURRENT分数计算图像指标与排序差，不比较旧完整集合汇总值。

## 轻量验证及未验证项

专项测试执行真实生产CPU选择封装、几何、PAIR、评分、地图、精确像素指标、空间统计与缓存ledger。小构造中的Torch Exact和模型提取边界用明确替身，执行原八张扫描函数体作等价对照；不声称真实Torch/cdist或DINO链路验证。

真实background固定12案例CPU回归覆盖668352个层×patch：PAIR距离、gate、votes和neighbor_count与原本地结果逐元素一致，36个CH1分数最大绝对误差0。没有重算模型特征或全域评价。其他测试覆盖原42顺序、43/44确定性、878共同集合、Top2并列、positions重排与边界、不同支持尺寸、Source/校准不随seed改变、缓存篡改拒绝、不可覆盖/孤立文件、prepare不重抽、逐图及逐组合续跑、无GT预测、后端评分漂移拒绝、精确小型CPU评价、两级macro、NA及跨seed小实例分组。

测试数量、耗时及本地实际资产哈希见本轮交付记录和`input_inventory_local.json`。真实prepare/capture、模型/权重加载、CUDA Exact、资源峰值、27组合完整CPU评价及原GPU AUPRO均未执行。不能由这些CPU测试推断支持敏感性结论。

新专项29项通过（24.494秒）；此前与既有跨域专项合跑72项通过（36.483秒，其中新28项、既有44项），随后增加“完整capture不需要模型/bank”的检查并通过新29项。组合测试中的GPU/模型/AUPRO均为旧测试显式边界替身，未使用真实GPU。旧非有限输入拒绝测试出现预期NumPy RuntimeWarning，不影响通过结果。另11个Python文件内存AST解析通过。

实际盘点补充：`downloads/official_memory.pt`与`_external/dinov3_checkpoints/`权重的SHA均通过，`_external/dinov3`官方入口存在；本次不是资产全部缺失。相关950张RGB及690张GT已读字节核验，未解码GT做正式评价。新三组共享capture尚缺878张、唯一支持特征尚需72张，服务器资产当前状态未检查。完整支持FP32与未压缩raw/紧凑预测合计估计46.72GiB，另需余量/排序临时空间；没有假定压缩收益。

默认不关机；没有启动脚本。后续运行、分域/续跑和独立review/完整缓存打包命令在README。可交独立review；进入正式实验仍需用户确认运行环境并显式执行。
