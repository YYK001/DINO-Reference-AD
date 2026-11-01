# 实现审计：固定单侧联合正常响应

本轮权限：本地代码、必要CPU测试、资产预检及运行说明。没有服务器连接、GPU初始化、真实提取/检索/训练或全域评价。Git发布另依据用户后续授权，原正式结果和窗口报告只读。

## 数学与信息范围

协议在protocol.CONFIG冻结，h=1、512点、四层、三个核模型+CF7容量对照。唯一主方法JOINT_ST8_CF7。Source仍完整100K/层，Target每设置仅同组K8；正常t_j整图排除，查询和正常同B_j。八正常图等权，层/路共享位置。min前两路在同一Source MAD尺度；已缩放距离不再除1280。负z不截断。

生产FP32 logits使用NumPy logaddexp/logsumexp或现成PyTorch logsigmoid/logsumexp/logaddexp。八图作为独立批量轴同时运算，padding仅mask无效log概率，各图仍按自身点数归一；A/B在联合/独立间共享，独立模型保留单路四维关系，只移除两路同位置配对。先等权平均图内概率，再等权平均八图概率，最后负log并按8/4归一；无概率epsilon，低响应不会变成双侧离群。CF7按原生逐层均值→原Source权重→原插值，不拟合任何权重。小型FP64仅做参考验证，未改变原Exact和MAD后端精度。

## 接入与必要重复

父metadata使用已有audit.Parent，复用其共同名单、Source/提取/支持/MAD/完整评价来源检查；没有改旧分支全局列表，也没有把新分支冒充旧分支。

seed43实体必须真正包含Source_distance、target_distance和positions，形状4×M/4×8×M；普通UNION响应和normal_statistics不够。seed42当前SACP生产者不保存逐支持距离，报告缺项。跨seedSource复用以实际bank seal、图像SHA、提取记录和物理位置绑定，不以seed名称匹配。共享Source缓存的原始provenance保持不变，防止续跑变成自引用冲突。

normal支持完整特征接入复用feature_identity.proof，核对image/sample和提取完整记录；旧schema没有内联extraction时，必须通过已封存父assets/raw_assets/inputs或显式feature_manifest证明来源，不更改旧manifest。没有Source实体时仅报告恢复原bank及成本；若所有分路Source响应已齐，后处理和Target-only检索不会强制打开无用Source。绝不自动重建memory或提取66校准图。

补算以显式flag开启。normal响应复用OriginalReferences.distances，新增adapter只延迟加载Source并提供本组支持；原全图排除、Original ResidentExact FP32 Top1及除1280未重写。Target-only查询每层上传一次，在8张支持间复用。两组缺查询距离时，单次完整q和一次Source距离供两组共享，只补各组缺失Target。Source、Target索引数值仅用于旧基线/identity，不新建近似检索。

## 检查清单及成本

以下是全部新增/沿用检查类别，明确披露，未添加资源容量准入门槛：

1. **小metadata与已有完成状态**：原Source seed42、三域902名单、两K8各8且排除并集、原权重/MAD/Source/提取和24父AUPRO来源。沿用父read_complete验证小CSV，不重做历史像素评价；加载父时一次完成。
2. **科学缓存身份**：原manifest seal、Source内容指针、图像与支持顺序、距离单位、提取记录，完整NPY头/有限性/positions映射。默认trusted-local fast读取；没有逐阶段重复扫checkpoint、Source bank或特征payload SHA。新数组只在写入时记录一次SHA。
3. **新运行隔离与数值配置**：新output/cache/scratch彼此不重叠，不位于父资产内。固定config、数值模块小文件、设备/块参数写入新inputs，冲突拒绝，不迁移旧目录。
4. **正常参照与baseline**：K7必须是本组整图排除。固定采样只取物理坐标。新分路min后MAD复现父UNION（原atol1e-5），原UNION/SEGMENT/Source权重的地图SHA与评分atol1e-5核对；无修补或放宽。
5. **GT边界**：全六设置先无GT冻结。GT只能进入evaluate/aupro/cases；GT resize SHA继承父共同名单。每原生新图插值后核对预先冻结SHA/MAP_MAX。
6. **评价中间结果**：沿用已修复的exact_pixel_result，检查独立封存、小结果哈希、有限性、[0,1]、完整像素正负数量。半成品/被改指标不重新盖章。AUPRO原ADEval1.1.0、200阈值、0.3边界、名单、地图identity及有限性核对。
7. **临时目录与打包**：只清理TemporaryDirectory本轮创建文件，异常时不扫父目录*.bin；review按类型白名单，显式排除smoke/scratch/大数组/符号链接。记录stage RSS、CUDA allocated/reserved、时间和新目录字节增量，版本仅metadata查询，不自动安装。
8. **12h执行边界**：本轮用户提示词明确要求。full capture/predict/all前必须固定正常资产smoke，未知项不当0；预计累计>12h须先明确同意。运行中在已提交单位后观察累计时间。没有RAM/VRAM/磁盘前置可用量阻断、没有双卡、没有关机。

## 评价和报告

表A SEGMENT_CH1固定引用同组完整K8 SEGMENT；表B仅最终MAP_MAX，绝不称作CH1。像素/PRO每地图只评一次。旧原UNION/S/Source自评分及固定组合完整保留只读表；新指标表里的UNION/S/Source表A行明确为SEGMENT评分组合，不能把图像继承算作新贡献。

固定比较只有主候选−U4/独立/CF7容量/完整K8 Source权重，逐六设置及双读出保留，240行；另给逐seed三域、逐域两seed和域后seed等权总体。任何缺失项都NA/pending，聚合不删除失败域。只做12张固定案例和分布检查，不铺开新margin归因，不按测试GT调整参照/阈值。

## 验证范围

新CPU测试包括数学参考、全排列、单调性、位置与名单、真实父metadata读取、纯CPU小夹具24项精确评价、相同地图双读出、GT全局冻结、预测与部分capture续跑、Source原始来源保持、配置变化拒绝、异常scratch清理、打包排除、单设置prepare不截断全局采样表及检索首次上传不重复乘查询数。此前共享评价、两K8、权重审计测试一起回归，共96项通过。另一个独立进程导入已安装Torch，只在CPU核对3种批量张量场景（不等图长、极端尾部和降低查询），没有初始化CUDA。脚本只做Bash语法检查，不执行。

固定正常smoke尚未在真实服务器运行。其正常K7张量重复八支仅用来测算相同运算规模，不冒充有效八支正式测试；CPU构造标签和AUPRO构造mask的结果均丢弃，只保留时长和代理性质。正式完整输入核查和地图核对仍不可省。没有拿构造数据吞吐直接宣称完成时间或新效果。真实GPUkernel、骨干/Exact补算、完整902张两seed实验均未执行；完整时间仍未知。

本轮已有本地预检与pending报告不构成新方法有效性证据。最后结果和测试计数见validation_results.json。交review后停止，不继续扫描参数。
