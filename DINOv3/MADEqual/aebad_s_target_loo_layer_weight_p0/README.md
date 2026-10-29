# Target-LOO正常统计驱动的层加权

固定AeBAD-S background / seed42 / K8，297张（85正常、212异常），455 Source构库与66独立校准。只改变UNION层融合与图像读出，不改距离、不训练、不搜索。默认`preflight`，不会启动模型、检索、CUDA或评价。

## 身份与已有资产

选择迁移后已完成的SACP父身份 `becf76da2ddd6f8669c077c44449976cc3bf6d82aaf98f2d02d9104f32dccf1e`，原Source bank seal为 `31d9bdb01d3576820a0fb8701b8989fc6a895a95e06db48373522694c9a8d75e`。父SACP继承的UNION/SEGMENT与最新same-origin实验经完整地图、GT、人群及评分复核一致。以SACP为只读基线提供者，**不执行SACP拟合或旧邻域恢复**。

- 原UNION/SEGMENT四层标准化响应、positions、原MAD、提取配置、支持名单、指标表可直接重放。所有输入以父manifest为依据，不按指标相近拼接。
- 服务器reference-residual完整正常特征若还在，直接复用，不再前向；正常距离缺失时显式补66张Source-only和8折支持自身Source+K7检索。每折排除整张自身及其全部patch，不是评价图移除一个支持。
- 同源邻域工具的Top16分量距离不是本轮原Exact正常统计缓存，不直接换成原距离。
- 本地旧66份Source NPZ具有旧input_identity、四层距离及尺寸，但没有已验证的当前bank对应证明。本轮不自动接受；不为迁就它改宿主。旧 `aebad_s_segment_looB_raw_cache_20261013.tar` 是评价图移除支持，不能作为支持自身LOO。局部图review只有合成LOO分数，不足以恢复u/p/q。
- 本地review只有元数据/结果，不能替代父预测NPY、Source实体或正常完整特征。缺必需bank只报告，不重建；缺特征默认停止，可显式授权补前向。

## 固定公式和数值

原Source patch MAD标准化：`z=(d-median)/scale`，距离已除1280一次。66 Source校准只检索Source。8 Target查询为自身整图留一，参照Source+其余7张。

逐图逐层：u为原生网格99%线性分位数（NumPy `method=linear`），p为原插值单层图最大值，q为原有效3×3窗口均值最大值。p/q复用原`statistics`，不从已加权地图重新提取。正常统计小表归约float64；层响应、融合、插值为FP32。

`mu=median(Source)`，`sS²=mean(max(Source-mu,0)²)`，`sT²=mean(max(TargetLOO-mu,0)²)`，**先截断正部再平方**。两档固定γ=0/.5，`s=max(sqrt((1-γ)*sS²+γ*sT²),1e-8)`；每类u/p/q独立归一化`w=(1/s)/sum(1/s)`。

新增两分支：`UNION_SOURCE_WEIGHT`、`UNION_TARGET_LOO_WEIGHT`。地图为原生`sum(w_u*z)`再原插值；无额外平滑、截断或MAD。

- 主读出`PEAK_LOCAL_WEIGHT`：`.5*sum(w_p*p)+.5*sum(w_q*q)`，不套旧image MAD。
- 补充`WEIGHTED_MAP_MAX`：新加权图最大值。
- 原UNION、SEGMENT及SEGMENT评分+UNION定位保留CH1/CURRENT。不称所有读出相同；γ=0也不是原CH1或原等权UNION。
- 两张新地图，各一次全像素CPU评价、一次原ADEval1.1.0 AUPRO（200阈值，FPR≤.3）；总共2项新增AUPRO，不是4项。原基线地图/GT哈希和评分复核通过后复用原五项指标，未放宽1e-5评分容差。

## 手动执行

不在本地执行下面GPU命令。真实执行由用户手动启动；本工具不会连接服务器。

服务器脚本默认沿用 `/home/ubun/YYK`、yyk环境及只读父路径；用环境变量覆盖真实路径。不会安装依赖、改base、改VPN或关机。CPU最多8 worker（当前地图CPU循环逐图执行）。只用用户指定的一张卡，无自定义Triton。

```bash
cd /home/ubun/YYK/DINO-Reference-AD
export RUN_ROOT=/home/ubun/YYK/runs/aebad_target_loo_weight_seed42_$(date +%Y%m%d_%H%M%S)
export CACHE_ROOT=/home/ubun/YYK/storage/layer_weight_cache/$(basename "$RUN_ROOT")
script=DINOv3/MADEqual/aebad_s_target_loo_layer_weight_p0/scripts/run_workstation.sh
bash "$script" preflight
```

预检缺新正常距离时返回2很正常；阅读`preflight.json`、`missing_inputs.csv`。只缺74份正常距离、已有完整特征及bank时：

```bash
# 显式补检索；没有该开关不会因缺文件自动跑GPU。
export GPU_INDEX=1  # 仅为手动示例，执行前由用户确认物理卡
bash "$script" prepare-normal-stats --supplement-retrieval
bash "$script" fit-weights
bash "$script" predict
bash "$script" evaluate
bash "$script" aupro
bash "$script" summarize
bash "$script" pack
```

若明确授权补缺失完整正常特征，第一步额外加`--extract-missing`，仅补缺项、保持原host/checkpoint/完整配置。已有特征配置冲突时拒绝，不能静默换新提取。没有权重/原骨干代码或环境不兼容就保留缺口。检索逐层使用原`_exact_top1_chunked`，FP32、原矩阵块规则；不启动Source构库。

所有依赖均在时可显式`all --supplement-retrieval`，按独立子进程执行上述阶段，释放阶段间GPU/RAM；日志在`logs/<stage>.log`。预检不自动执行all。需要后台时由tmux运行该脚本，沿用同一RUN_ROOT/CACHE_ROOT续跑，不生成新tag。已经成功的正常距离、权重、逐图地图、CPU中间封存和AUPRO作业分别复用；预测冻结中不放本次执行次数，避免续跑冲突。

## 总耗时与资源

已有原bank、74张完整正常特征、297份父层预测和GT时，暂估**0.15–1.5小时**：正常补检索0.05–0.5h、统计拟合与297图重放0.02–0.2h、两张图CPU/AUPRO0.03–0.3h、IO0.05–0.5h。历史同机三地图CPU72.5秒、三AUPRO47.6秒仅作参考，当前新流程尚无真实吞吐。此范围不是保证。

若原正常距离也已生成，省去其检索；两项新地图融合/像素排序及AUPRO不能用旧等权地图的指标替代。缺完整特征时补前向时间**未知**，取决于实际缺失图数、原host及IO；不能加0。缺bank需要另行决策，不自动构建。运行按已提交正常图/预测图/AUPRO作业记录耗时与外推，明显预计超过12h则暂停，只有用户明确同意后使用`--allow-over-12h`。

新地图约1.26GiB（依实际尺寸），正常距离约16MiB；CPU scratch按总像素×9字节约1.42GiB另加原精确排序工作空间。完整特征若需补提取可能额外约20GiB，存机械盘cache；scratch独立放SSD。RAM/显存未实测，不设资源准入门槛。每阶段costs记录时间、进程峰值RSS、已初始化CUDA的allocated/reserved、新数组写入字节和实际前向次数；不是全机/临时排序峰值的完整监测。

## 输出与完成边界

`preflight.json` / `asset_inventory.csv` / `missing_inputs.csv`；`normal_statistics.json/csv`（66×4×3和8×4×3）；`weights.json/csv`（u/p/q、S/T尺度、两档权重和下限触发）；逐图预测冻结；`evaluation/per_image_scores.csv`、`metrics.csv`、`metric_comparisons.csv`、两份AUPRO记录；`objective_report.md`、`status.json`、`costs/*.json`。review打包只含小型报告文件，排除scratch/大数组/软链接。

仅10行评分指标齐全、两项AUPRO有效并通过身份/地图复核后标记完成；缺项保留pending，不填0。实例分析是可选项，本轮不额外展开。最终须比较γ0与原UNION、γ.5与γ0、主候选与固定任务分工参照；保留两种读出，不择优。相近只能写“本配置未观察到额外收益”；不证明Target信息无关、不称γ.5统计最优、不用旧双侧方差分解或44%标准误解释。

## 检查清单与验证

检查仅有：父完成身份/Source与K8和完整297名单；正常缓存提取和参照排除身份；形状/位置映射；有限正权重；原基线评分1e-5与地图/GT一致性；预测和权重冻结；原CPU中间结果封存及AUPRO身份；输出/scratch与父资产隔离；用户指定12h预算。没有重复大数组SHA扫描、显存/RAM/磁盘准入门槛。新产物首次写入记录哈希，小型评价封存核对沿用原工具。仅清理由本次scratch临时目录拥有的文件。

测试：`python -B -m unittest DINOv3.tests.test_aebad_s_target_loo_layer_weight -v`。本地19项CPU测试通过（2.779秒），包括74份人工缓存统计拟合、逐图预测冻结续跑和原分数不匹配拒绝。无Torch的CPU导入由新解释器断言。未验证真实8折检索、真实特征前向、CUDA/ADEval性能或完整实验，不声称已验证效果。完成本地交付后停止，不搜索其他γ或层。


FP32归一化检查修复（2026-10-29）：原1e-7阈值会误拒绝正常舍入后的1.0000001192092896。检查改用四项正权重的FP32误差界 gamma_4=4u/(1-4u)，u=eps32/2，约2.3841864e-7，覆盖权重转换和三次加法。有限性、正值、形状和归一化检查仍保留。不重新归一化、不改权重公式/地图/评分精度，不更改旧基线容差。新增真实weights()输出的舍入回归及异常权重拒绝测试；回归同时验证地图和PL读出仍使用原FP32权重、数值不变。
