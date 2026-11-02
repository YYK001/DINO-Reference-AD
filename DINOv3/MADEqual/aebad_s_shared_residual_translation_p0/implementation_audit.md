# 实现审计：共享残差平移 v1

本轮仅本地代码、CPU测试与资产盘点。没有服务器连接、下载、CUDA初始化、真实提取/检索、训练或正式检测评价。原实现/历史结果/旧manifest未修改；本模块新建独立目录。没有发布Git或正式运行授权。

## 数值与依赖复用

- 原ResidentExact执行Source100K和8张支持各一次原Top1，距离已除1280；新方法根据原来源距离选UNION来源，最终向量残差距离只除一次1280。
- q先按positions规范为唯一完整物理网格，然后在FP32原始特征单位求全图均值。分块只控制gather，不分别求块均值；同一均值参与整图每点，包括自身。
- CPU NumPy是可复核实现；授权后的单GPU路径只用现成Torch FP32 gather、mean、vector_norm、minimum，沿用原precision设置。没有自定义Triton、TF32/BF16加速或全流程FP64。FP64只用于小CPU参考测试和标量汇总。
- 两遍逐来源处理，不保存9×N×1280残差；同一N×1280缓冲逐来源重用。Source原增强以同bank seal绑定，不能悄悄换NoAug、容量或采样器。
- 评分调用原statistics、ch1_score、weighted、bilinear，不把PEAK_LOCAL_WEIGHT当CH1。不拟合MAD、不clip z，不全库重搜平移query。C不作单调性假定。
- 原CPU exact_pixel_result和原official_aupro_adeval直接复用。9张定位图共享三读出，AUPRO只有9作业。

## 新增与沿用检查：完整披露

1. **父小metadata**：读用户指定的6种名单/输入/汇总文件，核对902人群、seed42 K8顺序和两组支持并集排除。只读seed42结果目录；不打开seed43特征、统计或检索结果。小CPU结果CSV/AUPRO来源和Source权重被原完成记录绑定。
2. **来源兼容性**：复用feature_identity.proof核对完整checkpoint/预处理/精度/软件/provider manifest。仅在实际补前向时要求checkpoint与骨干，完整缓存后处理不额外要求它们。9来源NN contract绑定query、Source、8个支持以及四层位置和Exact定义；距离近不等于来源相同。
3. **位置、形状与数值**：完整物理位置一一映射，拒绝缺失和重叠。消费时检查FP32/有限值、9来源索引范围与非负距离；不预扫所有特征数值。
4. **基线复核**：原Exact UNION经MAD与父z atol1e-5；重建原UNION/SEGMENT/Source权重地图SHA及原评分atol1e-5；零修正评分链路atol1e-5；原向量零修正与Exact距离采用原单位atol1e-7、rtol1e-6。失败保留failed_baseline_checks.csv并停止，不放宽、不拼接新旧指标。
5. **预测续跑**：不可变预测身份不含本次前向计数，但绑定NumPy CPU／现成Torch CUDA、设备/可见卡选择、FP32策略及软件版本；逐图预测JSON/数组bundle和smoke整轮/逐域记录一起绑定。切换后端在任何预测或smoke复用前拒绝，12h许可不会绕过该检查。旧未绑定目录拒绝而不补写身份。原始特征、memory和可靠的原检索输入按自身依赖复用，不因后续残差计算后端改变而无意义地重算。复用前核对query/provider及支持seals、Source seal；完整图预测按domain+sample_id存放。CPU评价、CUDA AUPRO及汇总保持已冻结的预测生产后端，不把后处理设备误作后端变更。评价后处理只需冻结紧凑预测，不强制原bank仍挂载。所有图先冻结，无GT。
6. **评价续跑**：沿用已修复的exact_pixel_result独立seal、结果哈希、[0,1]/有限性、完整像素正负数量；被修改或未封存的中间指标不会重新盖章。AUPRO核对原后端1.1.0、200、0.3、输入图名单与map identity。任何缺项保持pending。
7. **目录与清理**：新run/cache/scratch互相及父资产不重叠，scratch只删除本轮TemporaryDirectory创建文件。打包类型白名单排除bin/npy/npz/软链接和scratch，不递归吞掉失败评价的大临时文件。
8. **默认trusted-local fast**：原始大数组和checkpoint不反复SHA扫描。原NPY通过既有manifest/header及消费时检查；新数组只在写入时一次记录SHA。小JSON/CSV封存与最终逐图map SHA保留。fast模式不是每次重读所有字节的防篡改证明，不将其写成全量payload已复核。
9. **用户明确12h边界**：正式启动要求真实少量完整图smoke，已知预计累计>12h暂停，未知项保留；运行中每个已提交单位检查累计时间。没有RAM、VRAM或可用磁盘量前置阻断，没有自动第二卡、环境安装、关机或服务器连接。

## 测试范围与限制

测试覆盖固定902/9作业、A/B/C小FP64参考、零修正恢复、原单位均值/只除一次、B≤A、C可能高于或低于B、单来源相同、距离可升高、位置重排、重复网格拒绝、分块一致、零分母NA、负z保留、正确CH1/Source图、无标签公式、现成Torch CPU张量一致且CUDA未初始化。

小型三域夹具贯穿预测→全局冻结→9项精确CPU评价→三读出→续跑；检查GT前置拒绝、修改指标拒绝、基线错误保留、provider变化拒绝、完整/缺失AUPRO的完成逻辑、等权macro、scratch异常清理和打包排除。模拟AUPRO完成文件只用于编排测试，不代表运行真实ADEval。

另外用本地真实父小metadata核对共同名单和seed42来源；捕获读文件调用确认未打开seed43目录；独立新进程preflight确认未导入Torch。共享原评价/两K8/层权重代码一起回归。具体项数、时间见validation_results.json和cpu_test_results.txt。

**尚未验证**：真实CUDA向量规约与原Exact、实际checkpoint前向、真实NN复用或补算、三张完整图smoke、902图预测、真实GT精确指标与原ADEval本轮9项，以及真实GPU/RAM/磁盘峰值。资源和历史耗时仅规划参照，不能冒充新实测。保持等待review。

## 2026-11-02：P2后端续跑修复

修复CPU/CUDA切换身份不变、旧smoke被当作新后端验证的问题。新增隔离CPU测试覆盖：仅改变后端即可改变身份；同目录切换拒绝且原记录字节不变；完成/部分smoke拒绝异后端或未绑定记录；部分预测不能混用；相同后端仍正常续跑；旧未绑定目录不迁移；预算override不绕过后端检查；后处理设备与预测生产后端分离。测试仅改变设备字符串和元数据，不调用CUDA。更新后的实际测试记录见validation_results.json及backend_fix_tests.txt。
