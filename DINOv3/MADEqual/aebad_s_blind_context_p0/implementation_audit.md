# 实现审计：blind_context_block5_v1

日期：2026-10-28。本轮未连接远程、初始化CUDA、加载真实DINO、提取真实特征、训练或运行GPU指标；未改历史模块算法、正式结果或窗口记录。

已阅读窗口33—36节，SACP README/runtime/prediction、固定图原提取、superadd_host、参照残差评价/存储/常驻Exact、正常对应校准及近期两轮SACP状态。SACP两轮均已完成，不采用旧README“尚未运行”文字代替真实状态。

|要求|实现证据|
|---|---|
|真输入盲点|context.masked_input；runtime.masked_features先mask再forward_processed，后者才进入原_extract_processed_batch/patch_exec|
|4相/4层|完整网格物理块相位，单次宿主提取四层，按自己被遮挡相取位置；全局遮挡避免跨crop泄漏|
|生成器无原q输入|ContextIndex.select(context)、Synthesizer.reference(context)；prediction先计算refs四层，再加载original和parent raw_union|
|有界正常字典|64张Source路径哈希，排除66；每图512位置；值取原正式bank的Exact最近行。K8完整键和值|
|检索只看遮挡键|固定1280→128投影/L2；Source4+每支持1；没有原图NN候选、CLS或UNION置信度|
|整图排除|训练候选按image_id屏蔽所有该图条目；Target本图列无效，K7；Source仍有全部K8|
|凸组合|独立Wq/Wk归一化/0.1+两类bias；掩码softmax；值为原始1280D正常值，参数32776|
|固定训练|单seed42、1500更新；每步Source/Target64、图像等权；四层同位置、原q仅监督、Source RMS固定|
|恢复|每100步分代checkpoint，最后提交小指针；模型/优化器/step/实际样本/尺度身份一致；batch由step派生|
|距离/评分|FP32直接欧氏/1280一次，min原UNION；actual raw_union必需，不用逆MAD；共用原MAD和response|
|指标不重复|旧6行背景基线只读，新2行评分共享一像素图；原exact_pixel_result和official_aupro_adeval仅新分支|
|实例|原spatial适配仅绑定真实U/S名，不伪装为PAIR；原环带/固定峰值、小实例/NA，核对历史U/S标量|
|真正隔离验证|改变预处理输入同一遮挡块内容，实际host和生产Synthesizer；Source/Target、边缘/边界/内部；最终训练后再查|
|资源|masked投影在GPU；context键/值常驻；训练候选预计算且数据常驻；正式原q流式；每层构造字典；不保存Q×字典全矩阵|
|授权边界|默认preflight不import Torch；GPU只在显式阶段延迟导入；无安装/下载/SSH/关机/搜索入口|

CPU测试不调用真实宿主：其中一项测试替代底层DINO输出，只验证生产适配器的位置/投影传递；不能将它表述为真实ViT信息隔离通过。CPU介入测试证明遮挡后输入确实与块内容无关；未来check-isolation必须再在真实宿主上通过。

当前资源布局是估算，不是实测峰值或完成时间。少量真实正常图吞吐接口为prepare-context --limit-normal 2，未执行。只外推测得的上下文部分，其他阶段保留null。原Source bank若缺失始终阻止，不重建。原始特征允许未来按明确许可补提；缺少父预测的U/S/raw_union不擅自重跑旧检测器。

全部保留检查及其范围见README末节。没有资源容量前置拒绝、旧大负载重复SHA或自定义Triton。新缓存按依赖冻结，不写改旧manifest。校验失败不放宽容差、换指标或把缺失填零。

交付以validation_results.json记录实测CPU测试/语法与local_preflight的本机库存；delivery_status.json保持真实实验pending。固定297名单和64+8正常名单附local_preflight，未来准备阶段再次冻结为运行输入。

## 2026-10-28：P2预测冻结续跑修复

原predict把本次masked_full_extractions/new_original_extractions写进不可变冻结记录，完整复用时计数归零导致冲突。现将本次计数放回predict返回结果，由runner写独立会话cost；冻结仅绑定预测内容。已有冻结通过原verified逐图核对后直接返回，文件及seal原样保留（兼容旧计数字段），不会破坏下游CPU/AUPRO引用。缺全局提交的逐图缓存续跑通过延迟创建Data/precision/Synthesizer，仅在确需新图时初始化GPU。

新增CPU夹具覆盖297份微型数组的首次预测后重复predict/runner阶段分派、部分缓存补一张、旧计数记录不改写以及记录seal不一致仍拒绝。测试替代昂贵提取器与生成器，不运行真实DINO、CUDA或AUPRO。未修改方法、预测数值、正式结果、历史输入身份或窗口记录。
