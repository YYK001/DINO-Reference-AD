# 固定SACP强度扩展实现审计

日期2026-10-28。状态：本地实现、CPU夹具和真实父元数据预检；未连接服务器、发布代码、执行真实缓存重放/像素评价或GPU。

## 独立边界

新模块aebad_s_sacp_strength_p0，不改父SACP protocol/config/分支列表，不调用旧all。仅三分支L1；Source方向留父.5附表。协议/任务清单显式派生9CPU、9AUPRO、72主表/80完整表，历史56行原样继承并注明来源。

Parent只接受显式parent-run/cache；读取父inputs/catalog/protocol/预测冻结/MAD/状态/结果表，以及assets中软件版本，不检查或跟随其Source、折库、Q、子空间、权重等实体路径。ETA模式用父预测源码SHA确认原三轴顺序，消费0/1轴。禁止eta_regions均值替代、z逆变换、缺失时自动推理。

## 实际复用函数

- 原SACP `numerics.matched_uniform`：FP64求和、FP32(1-a)输出、原舍入界。
- 原`normal_correspondence_adaptation.calibration.z_values/response`：原四层等权、物理散射、FP32插值和冻结CH1；不拟合MAD。
- 原`support_witness_segment_p0.evaluation.exact_pixel_result/validate_exact_pixels`：全像素精确CPU AUROC/AP，独立中间seal和像素数/有效性验证。
- 原`segment_pair_crossdomain_p0.analysis.spatial`：独立函数globals绑定真实UNION/SEGMENT名称，计算新分支自身与固定UNION峰值；不将分支伪装成PAIR。
- 原Data.gt及字节校验/原resize，允许只提供GT根目录；新评价前全926重放冻结。
- 显式GPU阶段直接调用原`official_aupro_adeval(nstrips=200)`，FPR≤.3、ADEval1.1.0；不另写近似后端。

## 数值继承

L1依原顺序raw_union*sqrt(FP32(1)-FP32(1)*eta)，不翻倍旧距离下降。eta严格使用父保存值，0不变、1为零合法；无新裁剪或距离下限。z生成后凡距离完全没变则复制父UNION z。

每图.5重放22项：4个raw_main、12个父z、6个分数。数组1e-7绝对/1e-6相对，分数1e-5绝对/0相对，保持父协议。均匀系数按L1重新计算并保存逐图逐层a、原生总缩距、相对变化分位和匹配误差。

实例半强度与全强度相对UNION并列统计，同时计算全减半变化；计数阈值1e-6、两峰值定义、小实例和NA继承。原受损继续恶化和原未受损新增损伤分别记录。不能把计数当漏检或只凭eta解释margin。

## 执行和审计范围

CPU入口用NoTorch导入拦截，且API仅接收缓存数组/GT。不存在模型、Exact、Source构库、聚类/子空间fit、训练入口；专项完整路径另用函数调用监测验证这些边界。四计数为0有调用路径支持，不以硬填0替代集成验证。

预检只读小元数据、NPY头和positions；数值/半强度在消费时一次验证，不重复全量SHA。小结果/地图/GT身份、像素封存和任务完整性保持；没有容量准入。输出/scratch与显式父run/cache/data隔离；只删本轮临时文件。打包只报告类型，排除BIN、大预测和父cache。

CPU不需要Torch可导入，更不初始化CUDA。AUPRO可用性在预检单列；CPU完成仍pending9项，显式GPU成功才全完成。续跑按图、域CPU、域分支AUPRO。历史UNION/SEGMENT虽可重建作实际margin/响应参照，绝不进入新排序/AUPRO作业。

## 验证证据与限制

测试`DINOv3/tests/test_aebad_s_sacp_strength.py`；实际计数/耗时见validation_results.json及完整工作区validation日志。夹具使用原父numerics.correct构造独立半强度缓存、非平凡positions、刻意带一ULP差的父UNION z，六图人工GT含映射消失的小实例。原CPU像素与空间内核真实执行，四计算边界监测，Torch被禁止。

测试parent的历史指标/AUPRO以及新汇总完成状态所需AUPRO元数据是明确标注的人工记录，不是ADEval运行。全926只做真实元数据预检；缺926 bundle且本地map软件版本不同，因此未验证真实.5重放、L1效果、CUDA/ADEval、实机吞吐或显存。当前没有可报告的新指标或性能结论。

本次新增代码、测试、文档和本机预检材料，不修改任何父结果、报告或预测。交付到review止，不自动发GitHub或执行服务器实验。
