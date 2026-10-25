# 实现审计：正常对应残差适配

2026-10-25；实现与本地CPU验证，未连接服务器、执行GPU、安装依赖或修改历史产物。用户补充优先于设计正文：迁移后两轮可复用，旧AutoDL不在依赖闭包。

## 已读依据与复用

|依据/实际入口|本实现处理|
|---|---|
|文献精读RegAD_CAReg/paper.md与窗口记录29–31节|借鉴正常间对应学习，明确不是STN配准或官方CAReg复现；现有局部图没有额外收益不作为本方法有效性证据。|
|旧ReConPatch bottle报告、reconpatch_metric/model/scoring|旧线性bank关系训练已有尝试；新方法是有图像/位置身份的SS/ST/TT伪对应、非线性残差及真实K8三域独立适配。未调用旧runner或依赖旧bottle产物。|
|aebad_s_target_fewshot/runner.py:_calibration_raw,_fit_source_mad,run|核实Source亮度增强及66张Source-only MAD；没有直接导入/运行旧runner的缺失补推理路径。|
|fresh_inputs.prepare_data/select_catalog/validate_catalog|保持521排序、66划分、seed42八支持、完整926名单；支持ZIP和目录。GT字节身份可记录，GT解码严格在预测后。|
|fresh_runtime.source + memory_variants.build_official_memory_spooled|仅fresh缺bank时绑定host依赖调用原生产builder；455增强图、100K/层、100轮normalize=False、不改采样；为455训练图另提brightness1完整网格。|
|ShiftADD.memory._extract_four_layers、fixed_target_graph.prediction.extraction|核实支持提取的确定性预处理；新增原图使用原同次完整视野patchExec拼接接口；查询一次供六分支。|
|superadd_host._exact_top1_chunked、fixed_target_graph.runtime.original_nearest|沿用FP32 cdist/use_mm_for_euclid_dist、分块、稳定首NN及/1280一次。适配后同时查新Source和完整新Target，不沿用旧index。|
|aebad_s_target_local_segment.geometry.geometry_layer|原UNION/SEGMENT基线，不重新实现线段算法；小规模原FP64几何保留。|
|relation_reliability.calibration.fit_calibrators、ch1_scoring_inheritance.scoring|分别模型Source-only patch MAD和PEAK/LOCAL image MAD；本轮新增from_z避免集成再做patch MAD。|
|projection_layer_diagnostic.numerics.bilinear；旧空间spatial/mark_small|原native四层float32平均、float32插值；原实例/环带/固定UNION位置及小缺陷分组。|
|support_witness.evaluation.exact_pixel_result|精确全像素CPU排序；直接复用已修复的中间指标封存、范围、像素数核查；不伪装分支名。|
|official_metrics.official_aupro_adeval；fixed_graph.validate_aupro|原ADEval1.1.0、200阈值、FPR≤.3；18个域×分支独立作业，同图两种评分复用像素指标。|

旧模块未修改，全局BRANCHES不变，没有猴补共享模块全局变量。fresh Source仅在独立FunctionType命名空间绑定host，不改旧函数全局。

## 算法冻结

protocol.py完整记录预设参数，运行不能传步数、窗口、loss权重、seed搜索或支持选择覆盖参数。测试私有函数缩小步数/宽度仅用于CPU验证，正式CLI没有该接口。匹配块256作为数值执行配置固定；不同GEMM块可能导致softmax权重末位差异，不承诺任意块形状逐位相同。

教师每层单位e、单位有效3×3上下文c；concat(e,c)/sqrt(8)的点积即规定相似度。全局描述子来自完整原网格均值，非采样均值。位置采样按相对路径稳定PCG64、物理序排序，四层一致。图像配对、mutual Top3、固定低半区四负例不读GT或测试图。精确并列stable sort。每类均匀选无向图像对，再选有正例方向及anchor；空对显式记录，不强造对应。

2000步Source预训练×3种子×4层=12个作业；10000步×3域×3种子×4层=36个域作业。域不继承其他域；Source最终权重同种子共用。所有有效batch256，类别余数轮转；每类先均值再等权。Lid包含该类配对行和同一步共享的512额外正常行，域阶段Source/Target各256，Source阶段全512为Source。批内重复抽到合法anchor是训练抽样，不重复填充memory。

最终网络逐行变换原100K Source和完整Target；原row i仍是row i，不在新空间采样，不压缩Target，不加虚拟点。新距离可升可降。每成员校准独立。集成平均成员z，然后另拟合image MAD；不平均AUROC/AP。

## 精度、性能和存储

无自定义Triton。使用PyTorch现成Linear/GEMM/cdist/stable sort。模型/optimizer/保存特征和memory/正式距离FP32；CUDA训练仅Linear BF16；推理不autocast，GEMM TF32关闭，骨干设置沿原宿主。Source均值/RMS的小统计及原SEGMENT/评价必要旧统计允许FP64，不扩大至整个检索。

配对按256×≤2048块精确排序，分两方向计算，整幅相似度不持久化。训练正常features和teacher使用独立mmap，阶段内首次读取校验一次；不每batch重哈希。每域正式预测将三个变换库一次载入CPU RAM，GPU轮换当前成员层库；不反复从机械盘读取同一模型。没有新图/WITNESS构建、没有旧服务器扫描。

原实现的Source candidate spool仍在HDD，临时约121GiB；本轮没有宣称已经改为逐层SSD构库。配对训练样本与新库也是HDD，SSD保留code/env/model/report/精确像素scratch。stream未来新增模型要重提Q；disk额外缓存原Q成本明确。无未经实测的加速比或总时间承诺。

## 身份与所有停止检查清单

1. 输入：完整521/66、三域固定8支持及926评价人口，原支持顺序/SHA；只在指定recent读取，提取软件/关键宿主代码/官方checkpoint身份兼容后复用。
2. 方法：本模块与实际共享代码、外部骨干/宿主源码、配置/软件身份封存；恢复时不同配置/文件拒绝，不重写旧身份。
3. 数值：Source RMS有限且正、足够候选供四负例、训练类别非空、loss/gradient/output有限、probe变换非零方差。probe只发现明确坍缩，不保证所有空间没有退化。
4. 续跑：NPY及manifest、checkpoint SHA/最终checkpoint与complete一致；optimizer、RNG、已完成更新和调度器位置共同提交。中间像素指标使用原独立封存并查范围和像素数。没有凭新SHA收编被改中间结果。
5. 预测/评价：全926清单冻结前拒绝GT；模型/校准/物理位置绑定；同缓存CH1/CURRENT原1e-5复核；AUPRO地图及GT SHA对应CPU记录，200/.3/后端固定；未齐模型或指标保持pending。
6. 输出：recent只读，独立run/cache；scratch也与输入、recent、记录的父运行及外部cache隔离，resolve后双向检查包含关系；本轮scratch与cache分离且不能等于/包含run。直接CPU评价入口同样检查。域评价用唯一工作目录、独占创建NPY后端BIN文件，只清理明确记录的本次文件；不glob删除BIN，失败不截断或删除旧文件，未知文件保留。review白名单排除scratch、BIN、NPY和PT。无服务器连接、环境安装、自动关机、容量准入门槛、自动降精度或换算法。

上述检查用于可复现与发现错误，不宣称模型有效、伪对应物理正确或缺陷必然保持。真实环境OOM/磁盘不足由实际分配报错，而非加入预先资源阻断。

## 验证边界

本地测试使用已有Python/NumPy/SciPy/Torch CPU，不安装依赖。小矩阵验证教师、mutual规则、负例/并列、零初始化、完整1NN和有限梯度；实际AdamW小预算跑通Source及域适配，中断恢复与一次完成的网络权重逐位相同。集成测试执行生产transform/calibrate/predict/CPU评价和续跑，但替换了已训练网络、图像读取及骨干，使用合成小目录；不是实际DINO或全域实验。

CUDA/BF16训练、真实全量对应质量、三种子完整训练、真实特征几何、ADEval及峰值仍待服务器手动验证。报告结果不预填；未进行测试异常指标调参或新消融。提供独立review入口，真实有效性尚未验证。

2026-10-25本地记录：18项专项全部通过（21.670秒）；随后收窄GPU RNG操作到指定单卡，真实CPU优化器/断点恢复专项再次通过（1项，5.481秒）。13份模块Python及1份测试内存编译通过，Git Bash `bash -n`通过。测试临时文件后续使用工程G盘`.tmp/normal_correspondence_tests`，未安装依赖或调用CUDA。

默认本地预检已执行：`torch_imported=false`；数据ZIP和宿主config存在；默认路径下checkpoint、官方dinov3/hubconf缺失，本地测试解释器缺ADEval。状态为blocked而非真实运行通过。新工作站上的两轮大资产未远程访问；已用各自实际setup格式的小型合成记录核对fresh及FP32子目录身份读取，未把本地review包当作完整缓存。

独立review的P2修复：旧评价曾在用户指定域目录中按`*.bin`清理，可能误删父文件。现在CLI和直接评价均执行上述路径隔离，评价数组使用随机唯一子目录，创建成功后即登记所有权（含随后memmap分配失败的空文件），只关闭/删除本次资源。修复不改变训练、预测、地图或指标数学口径。

修复后回归：共23项，22项通过、1项跳过，22.161秒；跳过的是实际目录软链接夹具，原因是本地Windows缺少创建软链接权限。其余新增测试覆盖父运行及记录的外部cache、输入目录/祖先路径拒绝，默认独立scratch允许，直接评价入口提前拒绝，分配失败后旧BIN及未拥有文件保持原字节，成功清理及独占创建不截断旧同名文件；18项原CPU测试保留。目录resolve逻辑已实现，真实Linux软链接场景仍待验证。未连接服务器或启动CUDA。
