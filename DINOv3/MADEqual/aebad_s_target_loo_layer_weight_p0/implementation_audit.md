# 实现审计：固定Target-LOO层加权

读取了窗口记录23/38节、原scoring.statistics、calibration.from_z/response、父SACP输入与Exact、same-origin原基线修复及共享CPU/ADEval后端。只新建独立模块，不改旧模块全局分支、不覆盖历史正式缓存或报告。

父身份选SACP原始Source＋background K8；最近same-origin结果提供该基线复核线索，不把same-origin的残差MAP或Top16分量缓存当UNION原距离。旧66份NPZ有旧input_identity但未证明当前bank兼容，保留unconfirmed。旧evaluation leave-one-support-out和LOO合成评分不充当支持自身整图LOO。未连接服务器。

统计只处理66 Source calibration和8确认正常支持。Source检索原库；Target跳过整个heldout支持，剩7个完整特征库。复用原Exact完整bank块，不用截断/近似索引。原raw距离仅除1280一次；原Source MAD不重拟合。必要缺特征补原完整RGB前向只有显式双开关允许，不改host求兼容。

u线性分位数；p/q原statistics顺序；float64只归约小统计表。正部先截断再平方，以Source median为共同锚；Target低于锚为0。仅γ0/.5，每类统计四层逆尺度归一化，下限1e-8。γ0不读Target。地图FP32原生加权后原插值；均匀u权重精确走原float32 mean。PL从未加权层响应计算p/q，不重复使用u权重、不套image MAD。

权重先冻结、297地图和分数再冻结，然后才读取GT评价。CPU/AUPRO直接复用既有精确像素/封存函数和official_aupro_adeval，2项像素图作业，不把两种读出重复成4项。CPU阶段导入不包含Torch；GPU只在显式补检索/缺特征或AUPRO路径导入。原两基线按地图/GT/人群/两读出核对后复用，保留CH1/CURRENT名称。完整性只接受10行有效指标及2份有效完整AUPRO。

资源不设前置门槛；12h暂停是本次用户明示要求。记录实际阶段时间与峰值，预估明确区分有完整资产与缺特征未知。输出、cache和scratch不得与父资产交叠，scratch仅清理本次临时目录。review按小文件类型打包。

19项CPU测试通过，2.779秒：单侧公式、低Target、正权重/下限/归一化、γ0独立、同二阶矩、全图排除、Source455/66分离与支持排除、原u/p/q、均匀图复现、γ0不等同原CH1、显式检索保护、74份人工缓存拟合、预测冻结重入、旧评分不匹配拒绝、参照变更身份与目录隔离。没有初始化CUDA或安装环境。实际补检索、提取、完整CPU评价和ADEval仍待服务器人工执行。本地资产预检为missing_inputs，不能宣布实验完成。


FP32归一化检查修复（2026-10-29）：原1e-7阈值会误拒绝正常舍入后的1.0000001192092896。检查改用四项正权重的FP32误差界 gamma_4=4u/(1-4u)，u=eps32/2，约2.3841864e-7，覆盖权重转换和三次加法。有限性、正值、形状和归一化检查仍保留。不重新归一化、不改权重公式/地图/评分精度，不更改旧基线容差。新增真实weights()输出的舍入回归及异常权重拒绝测试；回归同时验证地图和PL读出仍使用原FP32权重、数值不变。
