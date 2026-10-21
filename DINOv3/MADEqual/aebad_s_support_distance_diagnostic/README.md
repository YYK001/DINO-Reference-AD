# Source＋八张Target逐支持距离的条件诊断

只检验：在原UNION响应`b`和原SEGMENT候选缩距`delta`相近时，正常高响应patch与GT覆盖patch的Source/逐支持距离描述是否仍有差异。
本轮交付实现、轻量测试和只读预检；没有自动运行297张正式分析。没有新检测器、门控、分类器、Target MAD或阈值搜索。

## 冻结范围及输入

固定AeBAD-S/background原seed42、原八张Target-normal顺序、85正常＋212异常＝297张评价图。
不使用三支持组敏感性实验的281张子集，不加入其他域。所有支持继续排除在评价集合外。
原距离已经是Euclidean/1280，代码不再除1280。

默认输入（都在`G:\codex\workspace\experiment`下）：

| 输入 | 用途 |
|---|---|
| `downloads/aebad_s_segment_looB_raw_cache_20261013.tar` | 297份LOO capture及1188份逐层SEGMENT；直接读tar，逐图加载 |
| `downloads/aebad_s_target_local_segment_review.tar.gz` | 原状态、manifest、逐图CURRENT、全部实例及UNION峰值 |
| `server_archives/aebad_s_target_fewshot_seed42_v1_formal_results/aebad_s_target_fewshot_seed42_v1/` | 原Source patch MAD、K8支持和正式身份小文件，不要求memory实体 |
| `datasets/AeBAD.zip` | 原GT；RGB仅用于原固定案例图和字节身份核验 |
| `DINOv3/outputs/mad_equal/aebad_s_neighborhood_segment_local_20261016_134948_cpu/` | 可选的已有实例复核，目录不存在时仍可用原review完整运行 |

可通过`--raw-archive`、`--review-archive`、`--formal-dir`、`--data`、`--history-dir`覆盖。
`--data`可为冻结ZIP或包含test/ground_truth的Target-only目录，不要求train/good。
不存在的可选history目录不阻断；存在却哈希错误的参考不被静默忽略。

原始tar SHA256固定为
`6fd4086bbaeb5318a5168dd90eef57d9955ffa37da44764db926913c3e4b2e41`，
review SHA256固定为
`f2703ee4518330d7f82a7b1dc4477a83b05e626fb104d96f2f1c38313efa7524`。
逐文件继续使用历史SHA、capture/SEGMENT身份和capture父SHA校验，不给改动后的缓存重新认可。

## 实际复用入口

- `aebad_s_neighborhood_segment_p0.inputs.Archive/load_image`：原NPZ哈希、元数据、support顺序、完整positions双射、Top-2端点身份和SEGMENT包含关系。
- `aebad_s_neighborhood_segment_p0.prediction.original_distances`：按物理positions对齐旧SEGMENT，再由Source及全部八张支持恢复完整UNION。
- `aebad_s_projection_layer_diagnostic.numerics.reconstruct/adaptive_max/check`：原float32 patch MAD、CPU插值、GT到patch映射和1e-5响应复核。
- `aebad_s_union_diagnostics.spatial.components/instance_regions/interpolation_stencil`：原8连通实例、原图16像素Chebyshev环带（排除全部GT）、全部非零插值权重。
- `aebad_s_neighborhood_segment_p0.analysis.Dataset/peak_position`：已验证RGB/GT字节读取、峰值位置。
- `aebad_s_neighborhood_segment_p0.prediction.bounded_images`：固定顺序、有限待处理图像的CPU并行调度。

不调用旧`context`（它强制要求CH1），不实例化MapBackend，不执行PAIR或空间门控，不调用旧完整评价runner。
上述部分模块包含未执行的门控/模型函数定义；本工具只调用列出的纯CPU读取/映射函数。
不依赖CH1状态、checkpoint、Source bank实体、DINOv3或ADEval；不导入Torch。

## 分组与原生描述

正常图按UNION四层Source patch MAD融合后的原生响应，选最高`ceil(1% * patch_count)`；同分按物理flat index。
每张正常图的这组位置供四层共用，不逐层重选峰值。原固定十张仅增加案例标记，不替代85张正常图。
GT组取原GT的adaptive_max覆盖patch；同一patch只计一次，并独立保存它与全部连通实例的关联。
GT覆盖patch可以混有正常内容，不能称为纯缺陷特征；正常高响应也不是固定阈值误报。

每层保存：`s,t1..t8,u,v,b,delta`、原顺序及排序后的八个Target距离、Source/Target索引、alpha、endpoint_length和两个端点编号。
`u=min(s,min(t))`，`v`直接读取旧SEGMENT。`b`严格用原float32 `(u-median)/scale`，没有改成image MAD。
审计尺度沿用原SEGMENT：先做float32 `scale+1e-8`，再将距离转float64计算有符号差；`delta=(u-v)/audit_scale`。
有效缩距按原float32条件`u-v > 1e-7+1e-6*abs(u)`，未超过容差的负微差也保留，不裁成0。

固定16项汇总描述（不按结果增加变体）：

- `source_advantage=(s-t_(1))/audit_scale`，正值表示最近Target更近，保留符号。
- `gap2/gap4/gap8=(t_(k)-t_(1))/audit_scale`。
- `count_better_source`，比较`s-t_j`与`1e-7+1e-6*abs(s)`；`count_tie_source`独立记录近似相等。
- `effective_support_count=(sum w)^2/sum(w^2)`，`w=max(s-t,0)`；无正改善为NA及明确状态，不填0。
- 八个`loo_j=(min(s,min_{k!=j}t_k)-u)/audit_scale`及`loo_max`。

八张支持距离是query到各支持图全部patch的最小距离，不是同物理位置，也不是八次独立正常性投票。
LOO主要由Source和前两个Target决定，不额外宣称独立八维证据。有效支持数也不是独立证据数量。

## 条件加权与200次图像bootstrap

`profile_summary.csv`分别保留全部选定patch、有效缩距、无有效缩距三个范围。
主未控制/条件比较使用同一批有效缩距位置，避免样本变化与匹配混淆；各层同时列被排除patch、图像和质量。

1. 每个组先图像等权，再图内patch等权；有效子集中只对仍有有效patch的图像均衡。没有有效patch的图像数量明确报告。
2. 每层在有效样本中，正常/GT两组各占一半先验质量，按上述图像权重生成b和delta的0/20/40/60/80/100%分位边界。
3. 使用加权CDF左逆；重复边界合并，常数控制轴仅一个格。内部边界右侧入格，外边界包含，保存原始边界。
4. 每格两组原概率分别为`p0,p1`。共同覆盖质量为`min(p0,p1)`，仅共同格参与；共同质量归一化后给两组相同格权重，格内保留原图像/patch相对权重。
5. 保存每格patch及不同图像数、未覆盖概率、最大单图份额和全部图像匹配前后权重。不把失配部分隐藏。
6. 输出b/delta匹配后的均值差和SMD。SMD分母为两组加权总体方差平均值的平方根；方差全零且差值非零时SMD为NA。
7. 每个描述量还报告NA后实际有效质量，以及该有限值子集上的b/delta SMD。`abs(SMD)>0.1`仅是预先声明的残余混杂提示，不筛选patch或调参。

加权均值、中位数/IQR、GT−正常有符号均值差、SMD与描述性AUROC分别输出。
GT覆盖固定为正类，AUC<0.5保持原值。它不是工业检测的Image/Pixel AUROC。
每个图像作为一个cluster，在两个组内分别有放回重采样；同图全部patch使用同一重复次数。
固定PCG64、seed42、200次；按协议/层的SHA生成稳定RNG，与多线程调度无关。
bootstrap固定原分位边界，每次重新计算共同格概率与匹配权重；输出2.5/97.5百分位区间及有效重复数。
无共同覆盖、无有效缩距、无有效描述量保持NA。完成诊断不代表已找到可分性。

AUROC使用排序及同值权重归并，不建立全patch两两比较矩阵；排序缓存复用于200次重采样。
条件加权改变样本组成；层间结果不是跨域重复，多描述量区间不是多重检验后的独立发现。
即使gap4/8或集中度保留差异，也没有控制所有Top-2几何，不能宣称额外独立增益或构成可部署方法。

## 实例与像素位置

保留全部388个原始GT实例，包括碎片、原固定最小三实例及不可计算状态。
使用原连通域及16原图像素环带，逐项复核历史面积、patch覆盖、实例/环带峰值与margin，不放宽`atol=1e-5,rtol=0`。
固定位置取历史UNION实例峰值和环带峰值，两分支都在相同像素测量；自身重新取峰值的口径另列。

```text
inside_drop = U(u_in) - S(u_in)
ring_drop   = U(u_ring) - S(u_ring)
fixed_margin_change = ring_drop - inside_drop
```

复核恒等式；固定/自身margin下降的描述性分类沿原1e-6响应口径，和历史复现容差分开。
实例消失或空环带不伪造margin；它们的原生GT patch仍保留在主分布和实例关联中。
在UNION及SEGMENT两套实例/环带峰值处，回溯全部非零插值参与patch，逐patch保留四层描述与权重。
先在原生patch计算min和全部描述，再作权重汇总；有限描述量的有效插值质量单独报告。
像素处描述汇总仅用于解释，不称为单个真实query特征的精确最近邻，也不把最大权重patch当成唯一来源。

`instance_profile_summary.csv`按自身及固定位置两种margin口径，将下降/未下降/不可计算分开；
支持模式的inside−ring均值先图内实例等权，再图像等权。相关模式不直接等于造成margin损失的原因。

## 本地命令

只读预检（默认入口；不创建实验目录，不做正式条件分析）：

```powershell
Set-Location 'G:\codex\workspace\experiment'
& 'D:\anaconda\miniconda\envs\chinamoblie\python.exe' -B -u -m DINOv3.MADEqual.aebad_s_support_distance_diagnostic.runner preflight
if ($LASTEXITCODE -ne 0) { throw '预检失败，先查看缺项或身份错误。' }
```

以下完整分析命令留待独立review后由用户显式启动，本轮未执行：

```powershell
Set-Location 'G:\codex\workspace\experiment'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$diagnosticOut = 'G:\codex\workspace\experiment\DINOv3\outputs\mad_equal\aebad_s_support_distance_diagnostic_' + (Get-Date -Format 'yyyyMMdd_HHmmss')

& 'D:\anaconda\miniconda\envs\chinamoblie\python.exe' -B -u -m DINOv3.MADEqual.aebad_s_support_distance_diagnostic.runner analyze --cpu-workers 2 --output-dir $diagnosticOut

if ($LASTEXITCODE -ne 0) { throw "分析失败，现场保留：$diagnosticOut" }
Get-Content -LiteralPath (Join-Path $diagnosticOut 'status.json')
Write-Host "输出目录：$diagnosticOut"
```

新目录须不存在；没有隐式续跑、覆盖或旧结果重新背书。失败保留partial输出与错误，不标完成。
每25张图打印进度，bootstrap每层每50次打印进度。
依赖NumPy/SciPy/Pillow/psutil/Matplotlib；本轮没有安装依赖。Matplotlib及测试临时目录均明确放G盘并清理。
预估2个CPU worker峰值RAM约1.5 GiB，建议可用RAM至少2 GiB及磁盘余量2 GiB（均非正式实测）。
不保存全部原始缓存副本或全像素排序临时文件。正式运行时间尚未测量，主要由实例原图环带计算、CSV写入及200次bootstrap决定，不能套用上一轮22秒评分诊断耗时。

## 输出

| 文件 | 内容 |
|---|---|
| `protocol.json`、`input_manifest.json` | 事前固定规则、代码和输入身份、Source尺度、支持/评价名单及GT身份 |
| `patch_profiles.csv.gz` | 两组所有选定patch，含无有效缩距位置、全部支持原序/排序距离与描述 |
| `patch_instance_links.csv.gz` | patch与所有GT连通实例的多对多关联，主分布不重复计patch |
| `profile_summary.csv` | 全部/有效/无效缩距范围的图像均衡分布、NA质量 |
| `conditional_comparisons.csv` | 未控制与匹配后均值、中位/IQR、有符号差、SMD、AUC和图像bootstrap区间 |
| `matching_bins.json`、`matching_coverage.csv`、`image_index.csv` | 冻结分箱、覆盖/排除/格/每图权重及图像ID映射 |
| `instance_profile_analysis.csv`、`instance_profile_summary.csv` | 每个实例的两套margin、固定分解、原面积/NA和模式分组 |
| `peak_stencil_profiles.csv.gz` | 全部实际插值patch、权重、四层逐支持描述；角色重复保留 |
| `numerical_verification.csv` | 原响应、峰值、实例、固定margin分解及stencil复现 |
| `status.json`、`timing.json` | 完成或失败、耗时、峰值RSS、完成输出文件SHA |
| `objective_report.md` | 对照控制前后、覆盖、模式来源、分布重叠、实例联系和结论边界 |
| `figures/` | 固定gap8/有效支持数的逐层加权CDF及原3张案例示意，不按结果挑案例 |

没有重新计算Image/Pixel五项指标或AUPRO，也没有修改旧报告。正/负结果均不能外推为跨域泛化或整个任务的可能/不可能。

## 本轮验证边界

- 19项专项测试通过（本地末次9.065秒），包含真实1张图抽查与小型合成CDF/报告输出测试；9个Python文件AST检查通过。
- 本地末次preflight约29.79秒，状态`inputs_verified_not_a_formal_analysis`，错误/缺项为空，`torch_imported=false`。
- 本地输入齐全：297份capture、1188份SEGMENT全部逐文件核验；224个必需raster（212 GT＋原12案例RGB）字节/尺寸通过。没有创建正式分析输出。
- 轻量测试覆盖实际生产数学/读取/绘图路径：positions重排、支持顺序/并列、零改善、微小GT/空环带、先min再插值、原MAD/audit一致性、图像均衡、图像bootstrap、条件混杂合成例、无共同覆盖及NA。
- 真实数据只执行1张异常图的profile、历史响应、实例及stencil抽查；全部297张地图/388实例和完整条件比较尚未运行。
- 合成图表/报告测试不冒充真实297张诊断完成。全程无Torch/GPU/服务器/特征提取。

```powershell
& 'D:\anaconda\miniconda\envs\chinamoblie\python.exe' -B -m unittest DINOv3.tests.test_aebad_s_support_distance_diagnostic -v
```
