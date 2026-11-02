# 实现审计：共享残差平移与直接范数参照 v2

本轮执行本地代码、CPU测试与资产盘点。用户已授权新增三域同口径直接范数参照、3项CPU＋3项AUPRO，并继续既定实验；预计超过12h仍需新同意。服务器由用户手动运行。本机没有连接服务器、下载、CUDA初始化、真实提取/检索、训练或正式检测评价。历史结果/旧manifest未修改。

## 数值与依赖复用

- 原ResidentExact执行Source100K和8张支持各一次原Top1，距离已除1280；新参照与候选A根据9个原锚点的未缩放FP32直接残差范数选择UNION来源（旧MM来源仅审计），最终向量残差距离只除一次1280。
- q先按positions规范为唯一完整物理网格，然后在FP32原始特征单位求全图均值。分块只控制gather，不分别求块均值；同一均值参与整图每点，包括自身。
- CPU NumPy是可复核实现；授权后的单GPU路径只用现成Torch FP32 gather、mean、vector_norm、minimum，沿用原precision设置。没有自定义Triton、TF32/BF16加速或全流程FP64。FP64只用于小CPU参考测试和标量汇总。
- 两遍逐来源处理，不保存9×N×1280残差；同一N×1280缓冲逐来源重用。Source原增强以同bank seal绑定，不能悄悄换NoAug、容量或采样器。
- 评分调用原statistics、ch1_score、weighted、bilinear，不把PEAK_LOCAL_WEIGHT当CH1。不拟合MAD、不clip z，不全库重搜平移query。C不作单调性假定。
- 原CPU exact_pixel_result和原official_aupro_adeval直接复用。12张定位图共享三读出，CPU/AUPRO各12作业，候选9＋参照3；强组合仅引用新参照地图，额外作业0。

## 新增与沿用检查：完整披露

1. **父小metadata**：读用户指定的6种名单/输入/汇总文件，核对902人群、seed42 K8顺序和两组支持并集排除。只读seed42结果目录；不打开seed43特征、统计或检索结果。小CPU结果CSV/AUPRO来源和Source权重被原完成记录绑定。
2. **来源兼容性**：复用feature_identity.proof核对完整checkpoint/预处理/精度/软件/provider manifest。仅在实际补前向时要求checkpoint与骨干，完整缓存后处理不额外要求它们。9来源NN contract绑定query、Source、8个支持以及四层位置和Exact定义；距离近不等于来源相同。
3. **位置、形状与数值**：完整物理位置一一映射，拒绝缺失和重叠。消费时检查FP32/有限值、9来源索引范围与非负距离；不预扫所有特征数值。
4. **基线复核**：原Exact UNION经MAD与父z atol1e-5；重建原UNION/SEGMENT/Source权重地图SHA及原评分atol1e-5；原Exact评分链路保持旧参照atol1e-5；实际直接向量零修正（固定/重选）的CH1与Source加权图改为只对新直接范数参照检查，仍为atol1e-5。CPU/现成Torch的直接范数（含零修正）仍使用原单位atol1e-7、rtol1e-6。不同FP32路径的MM距离/直接范数差异单列arithmetic_audit，只描述差值和超原raw容差数量，不能误称为兼容性通过。真实地图/评分不兼容仍保留failed_baseline_checks.csv并停止，不放宽、不拼接新旧指标。
5. **预测续跑**：不可变预测身份不含本次前向计数，但绑定NumPy CPU／现成Torch CUDA、设备/可见卡选择、FP32策略及软件版本；逐图预测JSON/数组bundle和smoke整轮/逐域记录一起绑定。切换后端在任何预测或smoke复用前拒绝，12h许可不会绕过该检查。旧未绑定目录拒绝而不补写身份。原始特征、memory和可靠的原检索输入按自身依赖复用，不因后续残差计算后端改变而无意义地重算。复用前核对query/provider及支持seals、Source seal；完整图预测按domain+sample_id存放。CPU评价、CUDA AUPRO及汇总保持已冻结的预测生产后端，不把后处理设备误作后端变更。评价后处理只需冻结紧凑预测，不强制原bank仍挂载。所有图先冻结，无GT。
6. **评价续跑**：沿用已修复的exact_pixel_result独立seal、结果哈希、[0,1]/有限性、完整像素正负数量；被修改或未封存的中间指标不会重新盖章。AUPRO核对原后端1.1.0、200、0.3、输入图名单与map identity。任何缺项保持pending。
7. **目录与清理**：新run/cache/scratch互相及父资产不重叠，scratch只删除本轮TemporaryDirectory创建文件。打包类型白名单排除bin/npy/npz/软链接和scratch，不递归吞掉失败评价的大临时文件。
8. **默认trusted-local fast**：原始大数组和checkpoint不反复SHA扫描。原NPY通过既有manifest/header及消费时检查；新数组只在写入时一次记录SHA。小JSON/CSV封存与最终逐图map SHA保留。fast模式不是每次重读所有字节的防篡改证明，不将其写成全量payload已复核。
9. **用户明确12h边界**：正式启动要求真实少量完整图smoke，已知预计累计>12h暂停，未知项保留；运行中每个已提交单位检查累计时间。没有RAM、VRAM或可用磁盘量前置阻断，没有自动第二卡、环境安装、关机或服务器连接。

## 测试范围与限制

测试覆盖固定902/12作业及新旧胜出来源切换、A/B/C小FP64参考、零修正恢复、原单位均值/只除一次、B≤A、C可能高于或低于B、单来源相同、距离可升高、位置重排、重复网格拒绝、分块一致、零分母NA、负z保留、正确CH1/Source图、无标签公式、现成Torch CPU张量一致且CUDA未初始化。

小型三域夹具贯穿预测→全局冻结→12项精确CPU评价→三读出→续跑；检查GT前置拒绝、修改指标拒绝、基线错误保留、provider变化拒绝、完整/缺失AUPRO的完成逻辑、等权macro、scratch异常清理和打包排除。模拟AUPRO完成文件只用于编排测试，不代表运行真实ADEval。

另外用本地真实父小metadata核对共同名单和seed42来源；捕获读文件调用确认未打开seed43目录；独立新进程preflight确认未导入Torch。共享原评价/两K8/层权重代码一起回归。具体项数、时间见validation_results.json和cpu_test_results.txt。

**尚未验证**：真实CUDA向量规约与原Exact、实际checkpoint前向、真实NN复用或补算、三张完整图smoke、902图预测、真实GT精确指标与原ADEval本轮12项，以及真实GPU/RAM/磁盘峰值。资源和历史耗时仅规划参照，不能冒充新实测。尚无正式v2效果结论。

## 2026-11-02：P2后端续跑修复

修复CPU/CUDA切换身份不变、旧smoke被当作新后端验证的问题。新增隔离CPU测试覆盖：仅改变后端即可改变身份；同目录切换拒绝且原记录字节不变；完成/部分smoke拒绝异后端或未绑定记录；部分预测不能混用；相同后端仍正常续跑；旧未绑定目录不迁移；预算override不绕过后端检查；后处理设备与预测生产后端分离。测试仅改变设备字符串和元数据，不调用CUDA。更新后的实际测试记录见validation_results.json及backend_fix_tests.txt。

## 历史修复记录：分离FP32路径（123677d，后续v2替代其跨路径零修正参照）

用户服务器首图background/test_0000的L7直接残差范数与原MM Exact距离最大差为6.034970283508301e-7；此前四层CPU/GPU直接候选核对已通过。不能把不同浮点求距路径的数学等价写成该raw容差下必然相同。独立CPU调用原_exact_top1_chunked已复现：相同query、bank及正确索引，仅改变公共特征偏置，MM消减误差即可超过旧跨路径门槛。

本修复不修改三候选计算、不放宽任何评分/地图或同路径CPU/GPU容差。跨路径差异保留为描述性记录；同时补上此前缺少的真实零向量修正CH1/地图检查（原函数此前用的仍是原Exact raw，属于自比较）。实际零修正的固定/重选图不满足1e-5时继续停止，不能据此自动启用重评基线、近似口径或混合指标。

每图紧凑缓存增加zero_fixed/zero_reselect距离，约增加0.38GiB（13924 patch、902图），仍在原5–8GiB规划内。没有保存全部残差。算术差异在每图CSV及最终arithmetic_path_comparisons.csv报告，不记作verified成功。CPU小夹具覆盖误拒复现、真实兼容性超差仍拒绝、零修正数组与公式一致；并对照0eb9529验证三个候选NumPy和现成Torch CPU输出逐元素不变。真实CUDA新复核仍需用户执行。

改动后的代码身份需要新run/cache，旧失败目录只读保留。原bank与支持特征继续复用。失败首图若未持久化完整q，则需要再次前向及对应检索；本次启动不把该首图旧NN导入新目录，不为省少量首图计算而改写旧NN身份。本轮没有重建Source、重校准或自行连接服务器。


## 2026-11-02：用户批准的直接范数参照 v2

旧Exact与旧地图/评分检查已经通过；实际直接向量零修正不能满足历史MM地图/评分1e-5。经用户明确同意，增加三域直接范数参照独立评价，而不是放宽容差或替换历史指标。

1. 原各来源Exact只负责原9个锚点检索，memory、锚点、MAD、Source权重、候选公式与强度不变。直接范数在除1280前决定新UNION胜出来源，A固定锚点和A/B共享均值据此生成；保留Source优先及支持顺序的同距规则。来源切换时A/B值可改变，这是明确授权的参照统一，不声称旧新候选逐元素相同。
2. 零修正固定/重选地图与CH1只对新参照以原1e-5检查。CPU/现成Torch同路径仍为atol1e-7、rtol1e-6。CPU核验生产后端胜出锚点在原raw容差内确为直接范数最小值，再用相同锚点集合影子复核均值；该验证选项不暴露CLI、不参与正式来源选择。远离最小值的override仍拒绝。
3. 原SEGMENT CH1＋新直接范数Source图为兼容强参照，直接引用新参照像素指标/AUPRO；主候选差值按新参照计算，旧MM仅历史对照。新旧五项指标、地图/评分差及来源切换单列CSV，宏平均保留数值路径和历史来源标签。
4. 三域新增参照各3项CPU与AUPRO，总计各12项；指标读出和强组合不会额外生成AUPRO。完成状态要求全部12项提交。预算以12项待办重算，新增参照单独pending也会计入12h边界。完整网格前向及9来源检索仍只共享一次。
5. 新协议身份为shared_residual_translation_direct_reference_v2，默认新目录；旧v1目录拒绝混用、不补写旧身份。没有新的大文件SHA扫描或资源容量门槛。完整检查清单仍是上面的9类。
6. 同时修复smoke中代理mask CPU测速导入覆盖预测backend变量的问题；小CPU夹具验证报告可JSON序列化且仍保留原backend身份。

轻量测试覆盖：未缩放来源选择（除1280后舍入同距也不能改变赢家）、Source优先、刻意构造旧MM不同赢家时A与均值同步改变、远源override拒绝、强组合无额外作业、12项独立提交/续跑、旧profile拒绝、数值路径报告及macro历史标签、参照AUPRO待办计时。独立现成Torch CPU验证三组张量及来源切换，无CUDA初始化。夹具模拟的AUPRO文件只测编排，不冒充正式ADEval。

历史段落中关于123677d与0eb9529逐元素相同的记录仅适用于那一次修复，不适用于本次批准的winner统一。旧日志及测试记录保留；当前项数、时间、预检和未验证项记在validation_results.json。真实v2 CUDA、三张完整smoke、902张预测和12项正式评价仍由用户运行。

当前v2验证：139项CPU回归通过（116.927秒，含本模块59项），独立现成Torch CPU核验通过且CUDA未初始化；12份Python AST与Bash语法通过。本地v7预检缺2731项指定向量/标量资产，未导入Torch，任务清单为候选9＋参照3。正式运行仍需服务器真实smoke，当前无正式v2指标。
