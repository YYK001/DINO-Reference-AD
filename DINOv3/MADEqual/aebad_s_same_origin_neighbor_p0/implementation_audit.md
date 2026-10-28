# 实现审计（2026-10-29）

本轮仅新增本独立模块与其CPU测试。没有改原宿主、memory builder、采样器、历史方法或历史结果/报告；没有服务器连接、模型提取、训练、GPU评价或环境安装。没有自定义Triton。

## 对应要求

|要求|实现位置与处理|
|---|---|
|background297，Source455/66，原K8|inputs.validate_catalog；原seed42 CRC32支持规则；原Source顺序mod8；不采用敏感性共同子集|
|原图完整特征/裁剪拼接|runtime.features → 原reference_residual/runtime.extraction → fixed_target_graph/prediction.extraction → 原host；完整positions映射|
|优先索引恢复|inputs.index_path沿只读导入链找原source_bank.npz，避免先假定索引丢失；不读另一份大bank重复证明|
|同源增强与四邻域|provenance.map_indices/gather_selected，实际增强factor＋原图/Source索引/网格/row/column/候选索引/bank行号；不以memory行邻接当空间邻接|
|已有真实增强网格|优先读实际增强metadata；全部齐全时不加载模型/权重；中心精确对正式bank核对|
|缺增强记录|复用同宿主初始化后的原随机流按455图顺序重放，只在全部中心精确吻合时承认恢复；逐图选中邻域检查点可续跑|
|必要时重建|仅--rebuild-source。inspect.unwrap原inference_mode装饰器后局部绑定原函数globals，原100轮采样器不变，在临时spool仍在时观察选中行；不修改共享模块，不增加NumPy随机调用|
|差分/绝对残差|numerics.residuals与Search.components；只存delta_m，d_nb从中心残差和差分重建；欧氏/1280一次|
|Top16|blocked torch.cdist完整FP32，固定大小cutoff-tie修复，GPU端按行号稳定排序；不以TF32/BF16/FP64改变检索；无需全图×全库×维度张量|
|同一绝对1NN与边界|三个分支共享Top16第1名；query外圈/Top1无完整邻域共同回退UNION z；C16只过滤原16，不补位|
|分量/图像MAD|pipeline.calibrate两遍；原fit_calibrators下限；abs全量、nb/rel有效位置；B/C同分量MAD；各自完整新分支拟合CH1；不再套patchMAD或截负值|
|基线复用|同Source seal、完整父预测、小表身份、必要响应/分数与本轮地图/GT匹配才复用；不兼容报错。独立库全部重算；--baseline-mode fresh可在不重建库的情况下重算基线|
|CPU与AUPRO|复用原exact_pixel_result/原exact_binary与原ADEval1.1.0（200/.3）；共享CH1/CURRENT定位；组合行不产生地图；中间指标采用原独立封存、范围和像素数校验|
|实例|直接调用原spatial内核，仅绑定明确U/S分支名；自身与固定UNION峰值、原环带、NA保留。新标度margin只作补充|
|续跑|不可变记录不含本次调用次数；完整预测重入不分配GPU；66图分量缓存与第二遍可重放；独立AUPRO作业；失败不填0、不改容差|
|预算|先输出五阶段历史估计和未知项；显式smoke仅两正常图，结果复用于校准；按已提交图像/作业实测外推，超过12小时暂停，exit75；原Source重建中断只重做该阶段|
|存储与打包|新缓存/构库scratch在机械盘，评价scratch独立SSD；只清理本次mkdtemp创建文件；保护显式和父metadata发现的Source路径；review按小报告类型白名单，排除bin/npy/npz/模型|

## 实际采用的检查（全部列出）

1. 原521名单顺序、66校准划分、正常K8顺序、297评价名单及支持排除。
2. Source选中索引范围/唯一性、原图实际增强与完整网格映射；正式中心精确相等；同图四方向；完整1280维FP32数组和positions。
3. 分量/独立图像校准、Source实体/预测记录的实际依赖一致性；必要的小型JSON/CSV封存与首次写入身份；显式复用记录。
4. 同批缓存评分与原父基线的必要复核，原地图/GT一致性；CPU中间结果沿用原范围、有限性、像素数与封存检查；AUPRO定义与输入地图一致性。
5. 新输出、cache、构库scratch、评价scratch与父输入隔离；只清理本轮创建文件；小文件白名单打包。
6. 用户要求的12小时估时暂停。**没有RAM、显存或磁盘可用量前置门槛，没有默认全大文件SHA扫描。** 原host权重身份验证未绕过；模型只有明确需要提取时才加载。

## 本地预检解释

本地已有SACP父review元数据与297图名单。原builder源码明确返回并保存selected_7/15/23/31；当前本地未找到其NPZ实体，不能推断服务器丢失。preflight在当前文件系统未找到正式Source实体与371份完整网格（66校准＋8支持＋297评价）；还缺配置路径上的checkpoint与DINOv3入口。名单与历史指标不是完整特征。服务器现存资产必须由用户手动preflight确认后再决定来源恢复还是显式重建。

## 未验证

真实455图增强重放、正式100K构库、当前工作站CUDA Top16吞吐/峰值、真实66图校准和297图完整五项评价均未执行。CPU人工/小型替身测试不等于这些运行已验证，也不产生方法效果结论。预估1.4–6.6小时恢复路径、1.75–8.3小时重建路径仅参考同工作站历史，未知项明确保留。

## P2修复：特征提供者及提取配置的续跑身份

修复前feature_ref仅查图像/头，feature_root未入冻结记录，确实可能混用提取配置。新增feature_identity.py只处理提供者元数据证明；没有改数学、采样、MAD、评分或评价参数。

- 冻结feature_root/feature_manifest与完整提取依赖；逐图冻结实际来源、图像manifest seal和提供者manifest身份。None明确表示原计划fresh，不能在续跑时静默换成后出现的外部缓存。
- 接入要求完整配置匹配；兼容原reference-residual raw_assets/inputs记录、父资产承诺，以及图像manifest内联的完整提取记录。不以单独checkpoint字符串代替预处理/精度/代码证明。
- 内部持久特征增加运行身份/提取身份/来源绑定检查；实际新前向也核对当前提取配置，避免外部缓存与新前向来自不同配置。
- 老运行缺少绑定时拒绝并要求新目录，不就地补写身份；后处理继续支持特征/Source实体离线。
- 新增7项CPU回归，覆盖同图同形状但不同checkpoint/预处理、续跑换目录、同路径图像/提供者manifest变化、同配置新运行的身份区分、原导出/近期运行兼容接入、无证明缓存拒绝、旧身份不改写及内部缓存拒绝。

新增检查仅涉及小型声明、配置及NPY头；阶段内缓存小型声明，不增加大特征文件重复哈希、资源门槛、服务器操作或GPU执行。

## 真实准备阶段接入修复

服务器首次prepare-source暴露KeyError:image_sha256：空间layout只复制了路径和索引，丢失原读图接口需要的图像身份。修复为复制原完整Source记录后附加grid/start/stop；不改变候选顺序、增强调用、中心/邻域定义或原图读取策略，没有新增哈希检查。新增CPU测试用真实PNG同时走原ZIP/展开目录Data.image接口。

另修复子进程输出提前关闭时的心跳忙循环：等待子进程本身，30秒超时才打印心跳，不以已结束的输出线程作为等待对象。已有冻结输入和配置未改变，首次读图前失败的原目录可以继续；不需Source重建。新增CPU子进程测试覆盖stdout先关闭、进程稍后退出的情况。
