# 实现审计：正常统计驱动的SEGMENT图像条件校正

本轮只在本地新增独立模块、CPU测试、资产预检和说明。未连接服务器、初始化CUDA、提取真实特征、执行真实Exact或正式实验；未安装依赖、修改base、改写历史结果或推送GitHub。共享残差A/B/C保持搁置。

## 阅读与复用

已读窗口记录47/48节、两K8 README/selection/runtime/pipeline/inputs及父封存结果；原geometry、CH1 scoring、calibration.from_z/z_values、旧层加权normal_statistics定义，以及最近共享残差身份记录。

|复用接口|本轮用途/必要适配|
|---|---|
|两K8/audit.Parent及其read_complete|原共同名单、K8顺序、Source/MAD、原CH1、定位指标和逐图地图身份，只读metadata；不运行其层加权/像素流水线|
|feature_identity.compatible/proof/small_record、原header/positions|完整提取及provider绑定；缓存小authority，兼容image/sample；Source66无需假造测试sample_id|
|joint_reference_response.features|确实缺失且显式许可后，原完整前向与提取设置；已有特征优先本地薄读取，不升级精度|
|ResidentExact|原FP32 MM Top1、原/1280；Source搜索分量跨六设置共享，不换最近直接范数路径|
|geometry_layer|同来源距离/索引/特征一致的K7/K8原SEGMENT；原局部FP64保持|
|statistics/from_z/ch1_score/z_values|raw只做一次patch MAD；z不重复MAD；查询CH1严格1e-5复现后引用原P|
|image_metrics/pair_errors|AUROC及average precision、ties半错误和排序变化|
|现有cost/pack|阶段RAM/CUDA/前向记录及小文件白名单打包；不新增独立通用结果核对工具|

旧Source-only和Target K7 UNION的u99%统计不是本轮B/P；旧测试专用baseline也不接Source校准/Target LOO正常图。新Source66必须分别通过各设置K8 SEGMENT；只可共享其特征和Source检索分量，不可共享新增回归。

## 方法与信息边界

B＝四层原生SEGMENT z各自全位置中位数再等权平均；B/回归紧凑矩用FP64。特征、memory、原Exact、z、峰值/局部统计与CH1保留原路径。Source总体标准化(ddof0)，Source非负OLS，Target中心化协方差/方差＋固定lambda1向Source斜率收缩；不加额外截距/截断/单图选择。

Source尺度退化阈值固定64eps64×max(1,max|B_S|)，两校正直接返回原P。有限性错误不吞掉。Target零方差自然回到beta_S；beta0严格复制P；分数可负、斜率不额外限幅。Source关系含当前Target K8，不称无Target信息。Target K7作为K8代理而非同分布保证。

444个正常设置记录来自最多114个独立正常图；同一个Source66特征/Source距离跨设置共享。测试902图按图服务两seed，三个分数只做标量运算。数据顺序和统计不依赖标签/GT；fit只读取normal_statistics及其引用记录。测试原CH1/图身份在collect核对，预测冻结后evaluate才读取标签/GT。

定位固定父同域/同seed UNION_SOURCE_WEIGHT，包括seed43；不借用seed42新直接范数图。只新增18个图像评价单元，没有新Pixel/AUPRO作业。继承指标标注来源，不算新增收益。删除LOO统计影响分析保持Source标准化及beta_S不变，不重新删除支持参照、不挑系数。

## 检查代码完整披露

1. **父小metadata**：沿用已完成两K8父解析器，确认Source42、66校准、两组有序K8、共同902名单、支持并集排除、原CH1及完整定位引用。只读小记录，不重放旧评价或完整历史实验。
2. **实际缓存依赖**：小表/原生响应/NN合同绑定图、域/seed、kind、Source、提取/MAD、K、heldout、有序引用和几何。旧dependencies格式有只读桥，实际字段和原geometry/Exact定义匹配才接纳；不改旧manifest/身份。未知旧格式明确不接纳。
3. **特征与原始数组**：原图/完整extraction/provider manifest、FP32形状、位置完整双射、索引范围、消费时有限性检查。Source精确bank仍100K×1280；不重建或替换。K7距离/索引/特征列表同步筛选。
4. **统计与系数**：有限性、数量66/8、四层中位数/峰值/局部值、原CH1、ddof0/固定退化/约束；正常记录引用seal及明确Source/Target参照，防止旧UNION/Source-only冒充。
5. **查询复核**：重建原SEGMENT CH1与封存值最大误差≤1e-5；超差保存failed_numerical_checks并停止。通过后用原P；定位只核对引用身份，不重新算图或借指标。
6. **续跑隔离**：新run/cache与父输入路径不重叠，无删除代码。冻结配置/代码/provider/计划后端；每图B/P、系数、query统计、六组分数和全局预测封存，小记录变化拒绝复用。fit/predict/evaluate纯CPU后处理继承生产后端。
7. **评价完成**：全局冻结后才能标签/GT评价；三分支引用同图；原SEGMENT图像指标也对父结果核对。小评价文件单独完成记录与哈希，修改后不重新盖章。宏平均按设置等权，缺项pending，不填0。
8. **用户12h边界**：真实资产smoke按对应执行路径外推。保留原已提交单元处理时间，不能用续跑小表读取代替原检索吞吐；未知类别不当0，已完成不计待办；持续按实测更新。预计/累计超12h保留断点并等待新同意。无RAM/显存/磁盘量准入、自动安装/第二卡/关机或SSH。
9. **打包**：复用现成小文件白名单，排除bin/npy/npz/checkpoint、scratch和软链接。无全像素临时盘，无新AUPRO。

原始大特征/checkpoint不重复payload SHA扫描：现有trusted-local fast读manifest/header，消费时查值；新数组写入一次记录SHA。小封存和指标/map身份保留。Source/extraction/MAD/geometry指纹在阶段内缓存，不每图重新计算。fast不冒充完整大文件防篡改审计。

## 本地验证与范围

CPU测试区分数值内核、带替身的六设置全链及真实父小metadata。覆盖闭式约束目标、ddof0、Source退化/Target零方差/负协方差/非有限、beta0与负分数、偶数中位数/位置重排/重复缺失、z不重复MAD、K7特征顺序、Source K8/Target K7与旧统计拒绝、六设置独立系数、标签不影响拟合/分数、1e-5、定位继承/AP ties/macro、续跑/修改文件拒绝、缓存原Source共享、两seed共享query、只读旧schema桥、无Source时前向前停止、预算及smoke续跑。没有用重复测试凑数量。

真实父metadata预检在独立CPU进程执行，未导入Torch。没有可用本地原生SEGMENT实体/完整特征/Source向量，所以没有将夹具替身写成真实缓存数值验证。已有Source/Target各(B,P)还未获得，系数与实际检测效果均pending。实际测试项数和时间见本地validation_results.json及测试日志。

尚未验证：服务器Source66/支持完整特征和NN真实接入；真实checkpoint前向/单卡Exact；6设置正常SEGMENT/原CH1兼容性；真实资产smoke与12h估时；真实1804个图像×支持组评分/检测指标/面积描述；真实资源峰值。没有声称固定公式已有效，没有自动追加方法或超限授权。

## 本地预检交付

目录：DINOv3/outputs/mad_equal/aebad_conditional_image_score_preflight_20261102/。

有父共同名单/原图像分数/指标/MAD/定位身份、原Source校准名单和提取metadata。本次显式路径缺2248份SEGMENT响应或(B,P)（444正常＋1804查询）、1016份唯一完整特征、1份Source实体；checkpoint与骨干代码为需要补前向时的条件缺项。预检只说明本机指定路径，不代表现服务器文件丢失，也不要求全部下载到本地。

asset_inventory/missing_inputs/necessary_work和time_estimate说明已确认、缺失、可能重复及未知成本。服务器可能已具备全部查询SEGMENT和正常特征，仅需补正常响应；必须由用户在保留原资产的机器手动preflight判断。本轮不连接服务器确认。

最终本地验证：34项模块CPU测试通过（27.627秒）；相关94项CPU回归通过（170.668秒，包含其中33项模块测试及61项原几何/CH1/两K8测试）；9份Python AST及Bash语法通过。没有实际CUDA/模型/完整缓存或新检测结果。上述项数有重叠，不能相加冒充独立测试数量。

## 2026-11-02：P2报告六设置计数修复

报告按AUROC/AP分行计数时遗漏metric筛选，曾将两项各六设置合并为十二项。现补上metric==当前指标，仅统计同一比较、同一指标的六个正式设置，宏平均行不计入。分数、系数、指标表、支持组反转判断均不改变。新增报告级测试实际生成objective_report，覆盖AUROC/AP各3升3降、两指标方向不同含平局（AUROC3升3降，AP2升2降2平），以及SOURCE相对基线6项全不变；对三组固定比较逐一断言。测试是合成报告夹具，不是正式实验。

本修复后35项模块CPU测试通过（27.458秒），报告专用测试单独通过（1.096秒）。未运行GPU、服务器或正式实验，未推送。

## 发布授权

用户已明确要求推送并提供实验命令。本次仅发布代码、测试和说明；不包含本地预检/结果、验证日志、数据、权重或缓存，不代为连接服务器。报告计数P2修复一并发布。
