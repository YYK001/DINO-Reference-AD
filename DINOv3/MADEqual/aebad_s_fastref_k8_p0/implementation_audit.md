# FastRef 固定 Target K8 实现审计

审计日期：2026-10-13。定位为 FastRef 论文公式在 DINOv3 / 原 Source anchor / 真实 Target K8 上的受控欧氏适配，不是官方实现复现。background 是已反复分析的开发域；不以本次结果宣称 FastRef 整体有效或无效。

## 证据及版本

- 正式版本：[CVPR 2026 主文](https://openaccess.thecvf.com/content/CVPR2026/papers/Li_FastRef_Fast_Prototype_Refinement_for_Few-shot_Industrial_Anomaly_Detection_CVPR_2026_paper.pdf)，§3、公式3–6、Algorithm 1；[正式补充材料](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Li_FastRef_Fast_Prototype_CVPR_2026_supplemental.pdf)，附录B。直接PDF访问返回403；通过公开搜索索引读取了主文目标/更新/初始化和补充材料推导，不能声称直接通读正式PDF。
- 交叉核对：[arXiv:2506.21398v1](https://arxiv.org/html/2506.21398v1)，§4、§6.1及附录A/C。URL版本为v1，页头提交日期为2025-06-26，但HTML另显示2026-08-24；不将该显示日期视为新的arXiv版本号。正式版初始化使用正确的行Gram；该HTML初始化式出现维度不合的列Gram。
- [作者仓库](https://github.com/liyufei25/FastRef)：本次公开只读核查显示空仓库，未取得官方代码，没有clone或执行仓库代码。
- 本地两份用户指定报告、`SuperAD_formal_results.zip`内六个指定结果、`SuperAD_main_code.zip`内四个指定脚本及`src/backbones.py`均只读检查，没有解压或执行归档脚本。当前B capture和原Source/MAD/支持身份沿用既有验证记录；raw距离/index不能恢复完整Q。

## 原循环的确切原因及 v3 采样适配（2026-10-13）

代码依据：`external_baselines/SuperADD/tracks/industrial/src/industrial/nearest_neighbor.py:subsampling_distance_based_fast`；索引封装 `external_baselines/superadd_external/memory_variants.py:_official_subsample_values_with_indices`（约480–530行）。正式AeBAD入口 `aebad_s_target_fewshot/runner.py:run` → `superadd_host.build_exact_official_memory` → `build_official_memory_spooled(return_selected_indices=True)` 调用该封装，`iterations=100`。基础NoAug检测器 `mvtec_noaug_mad_equal/memory.py:build_noaug_memory` 也进入同一链；上游 `industrial/model.py:SuperADD.train` 使用相同原规则。原规则与v1投影贪心coreset实际不同。

| 入口 | 候选池及距离 | 归一化/投影 | 初始化/RNG | 预算/输出 |
|---|---|---|---|---|
| 上述Source索引封装及原宿主 `subsampling_distance_based_fast` | 每层正常memory图完整patch拼接，校准图排除；raw欧氏100-NN，包含自身；封装用原精确分块后端 | 无归一化/投影 | 全局NumPy RNG；100轮随机未选子集，半径增长及固定uniform keep-mask，之后随机补足。正式Seed42在提取前初始化，提取/预处理消耗RNG | 每层最多100K；每轮 `quota//100`；原始行及升序物理索引 |
| 第一章NoAug `memory.py:build_noaug_memory` → 同一链 | 同上，brightness固定1.0且保留原uniform调用 | 同上 | Seed42在建库前设一次，保留提取及四层RNG消耗 | 同上，原始patch而非中心 |
| 本工具历史v1 `runtime.select_prototypes` → PatchCore ApproximateGreedy | 八支持整层池；128维随机投影空间欧氏贪心覆盖 | 无L2归一化；有随机投影 | 各层重置NumPy/Torch Seed42，10随机起点 | 512贪心步，真实行索引、贪心顺序 |
| 本工具历史v2 `runtime.select_prototypes` → 原Source封装 | 八支持整层111392行；预算接口512，其余100轮/100-NN沿用 | 无归一化/投影 | prepare开始Seed42，四层连续；存在下述兼容阻断 | 目标每轮5；v2在预检阻断，没有生成真实原型 |
| 本工具v3 `sampling.sample_indices` → 原 `_chunked_exact_nearest_neighbors` 距离后端 | 冻结八支持顺序/各自row-major原始1280维行；所有候选一次随机均衡分组；每组raw欧氏 `min(100,组大小)`-NN | 无归一化/投影 | 局部RandomState42一次创建，L7/15/23/31连续；每候选一个指数race key，有限无放回排序 | **整层全局512**，无每组quota；输出512个不同物理索引并按原候选顺序保存 |

八支持的图像级抽样是 `aebad_s_target_fewshot/support.py:select_target_support` 的Seed42/域/路径CRC32稳定排序，本轮只核对冻结manifest与历史SHA，不重新调用它。本轮调整的是patch级采样，不触及支持身份或排除规则。

### 对原循环六个问题的回答

1. 每轮候选数为 `int(count / iterations)`，从尚未选中的候选中随机取该大小；每轮目标是 `quota // iterations`，余数不分轮处理，而在所有轮结束后随机补足。111392/100得到1113候选；512//100得到5，余数12。预算小于分轮数时每轮目标为0；小候选子集还可能先因不足100-NN而报错。
2. 内循环条件是 `number_of_samples > target_to_keep_subset`。初始计数人工设为目标+1。距离表、候选索引和uniform随机数只在进入while前生成一次。
3. while内只重算半径下的邻居计数、keep-mask及数量，半径乘1.1；不更新随机数、距离或候选。半径增加时保留数量单调不增，但并不保证能降到目标。
4. 对有限非负距离、正初始半径r0，半径最终超过最大距离，所有k个邻居都计入，此时最小保留数 `L=count(u < 1/(k+1))`。**L≤每轮目标时可保证退出；L>目标时无法退出**。零均值距离导致r0=0，半径一直为0，严格 `<0` 的邻居数恒为0，u∈[0,1)会留下全部候选；目标小于子集大小时也无法退出。非有限距离未被原入口明确拒绝，是另一项不可靠条件。
5. 之前独立RNG算例确实对应v2提议的“局部起点Seed42、首轮111392未选候选、取1113个、再生成1113个uniform”的调用顺序；NumPy1.26.3算得L=9>5。这是条件反例，不证明所有Source/服务器调用都如此。历史Source提取前设种子且预处理消耗RNG，其实际状态不能由该算例代表；其他种子/调用顺序也可能L≤5。旧v2预检以期望1113/101≈11.02判阻断是保守筛查，并非充分必要退出判据，v3不再使用它。
6. 原固定uniform/有限k的选择规则存在不可达到预算的适用限制；实现没有饱和/零半径退出检测是既有终止缺陷。本次把每层100K预算直接改为512仍保留100轮，暴露并放大了这些问题；不能归因于FastRef数学内核。仅改变分轮数量仍不能保证消除固定mask下限，也不能用增大N迁就循环而突破FastRef秩条件。

本地回归提取并执行原索引函数的实际AST，用明确距离/Tensor/RNG替身构造“正半径但最低mask超预算”和“零半径”两类情况，观察四次数量不变后由测试计数器中断；绝不让测试本身无限循环。加上以上单调/饱和推导定位原因，不用真实大任务验证。没有执行真实Torch采样、模型或用户归档脚本。

### v3：有界反密度精确预算采样

最终固定 **N=512、分组上限100、k上限100、Seed42、radius_factor=.1**。一次局部RNG随机排列整层全部M个候选，`array_split` 成min(100,M)个非空均衡、互不相交的组；全部候选恰好访问一次。分组只控制局部统计和kNN工作量，不分配原型名额，不把支持图分开各选N。

每组沿用原初始半径统计 `r_g = mean(raw Euclidean kNN distances)/10`，包括自身；组小于100时k取组大小。保留原反密度表达式：

`c_i = count(d_ij < r_g)`，`w_i = 1/(c_i+1)`。

**r固定，不再随预算增长。**每个候选生成一个uniform u∈[0,1)，生成指数时钟 `key_i = -log(1-u_i)/w_i`；整层按(key,物理索引)排序，取最先N个，再按物理索引升序存储。指数时钟速率为w，其最先被选概率为w_i/sum(w)，移除已选项后同样成立，所以这是明确的反密度加权无放回抽样，非随意截断候选。相同时钟由原索引打破并列。不因求解失败再抽样。

工程调整：使用局部RNG、防止污染全局状态；均衡互斥组覆盖全候选；全局精确预算解决每轮floor/余数/零名额；小组k=min(100,n_g)；不保留无界while。**改变选择行为的算法调整**：原随机重叠子集变为随机分区；原随预算增长的半径固定在原起点；固定uniform Bernoulli保留+随机补足变为精确N的指数race。这些不与原Source采样完全等价，只保留其raw欧氏局部反密度偏好。v3不是切换回PatchCore，也不是将原Source memory重建为新规则。

终止依据：G≤100个有限组；每组一次有限kNN及求和；每个候选仅一个有限权重/时钟；一次有限排序输出N个不同索引，**无while、重试或另一路后备算法**。w∈[1/(k+1),1]严格正，u<1保证时钟有限。零半径沿用严格比较得到c=0/w=1，退化为有记录的均匀无放回采样，能够退出；重复特征可返回不同物理patch，但不保证向量独立，后续Gram照常拒绝奇异原型。

原型预算条件：M≥512且512<1280，普通行Gram有可能可逆；512不是最优性能值，不按异常标签/指标选择。保持该预算可保留原对照名称和固定约30MiB原型/常量、约54MiB/层全图OT成本；无须把N增加到1280以上来适应旧采样循环。真实P仍必须满行秩、cond≤1e12、solve残差≤1e-10，保留原epsilon、lambda和全部数学失败规则；无ridge/pinv/改N回退。真实条件性只能之后prepare/smoke验证。

`sampling_diagnostics.json`记录各组大小/k/radius、密度/权重范围、全候选权重SHA、权重是否均匀、有效权重样本量以及选中项权重/priority/组编号。`prototype_provenance.csv`记录支持ID、物理索引、prototype顺序及这些选择证据。采样诊断和来源在求解前落盘；秩/求解失败仍保留证据，不生成成功prepare状态。

v3 TOOL、sampling revision、代码SHA、局部RNG、预算、八支持身份及prototype字节/索引身份全部冻结；v1/v2旧原型和预测不能改名认可。两个新增分支加载同一组P与顺序，每张query不重新选P，不合入完整T。smoke/infer再次将原型与原始支持行核对。旧Source/MAD、四层宿主、Exact单位和评价不改。

## 采用的目标、量纲和独立推导

以下是本工具明确固定的欧氏目标；不是从余弦代码直接换一个距离函数。令Q为m×d原始查询特征，P为n×d真实Target原型，d=1280、n=512。两者均不做逐行L2归一化。内部统一换算X=Q/1280、M=P/1280，仅改变全局单位，不改变范数相对尺度。

U为m×n的概率传输计划，总质量1，行边缘1/m、列边缘1/n。优化目标为：

\[
J(W,U)=\sum_i\|X_i-(WM)_i\|^2
+\lambda\sum_{ij}U_{ij}\|(WM)_i-M_j\|^2
+\lambda\epsilon\sum_{ij}U_{ij}\log U_{ij}.
\]

重建项是**求和**，不是均值；OT也是概率质量加权求和。两项几何量均是宿主距离单位的平方；epsilon同样具有距离平方单位，lambda无量纲。raw单位下等价地将epsilon乘1280²，并将整个目标乘1280²；不改变最优W/U。优化使用平方欧氏，最终评分使用平方根后的欧氏距离，这属于宿主适配。

记G=MMᵀ（n×n）。固定U，逐行求导得到：

\[
W_iG={X_iM^T+\lambda U_iG\over1+\lambda\sum_jU_{ij}},\quad
W_0=(XM^T)G^{-1}.
\]

右除G通过Cholesky和solve完成，不显式求逆。设B=solve(G,M)，则初始重建A=(XMᵀ)B。因为UM本身处于M的行空间，更新可等价实现为：

\[
R_{l+1}={A+\lambda U_{l+1}M\over1+\lambda\sum_jU_{l+1,ij}}.
\]

本实现验证该省内存更新与独立逐行线性求解/一阶最优条件一致，不保存完整W或1280×1280投影。实际行质量用于分母；不靠强行置1/m隐藏边缘误差。完整查询共用一个OT问题；不把query chunks当作独立分布。

论文的refined集合是**m×d**，并非仍有512行；512只限制初始P的数量。最终取每个原始query到全部m个refined vectors的最近邻。`_exact_top1_chunked`限制新分支临时距离矩阵64MiB；不落盘Q、U、R或query×R矩阵。

G可逆要求P满行秩且n≤d；n=512<1280只是必要条件，并不保证满秩。严格检查秩、Gram条件数、Cholesky和求解残差；不静默pinv/ridge/降秩/改变原型数量。正式主文提到伪逆，但本任务要求不作失败回退，本工具只接受满行秩ordinary solve。

论文以n=alpha×k×h×w确定原型数量，并非固定512；正文给出的压缩比例随骨干/数据集不同。若这个数量超过特征维度，行Gram必然奇异，即使不超过也可能因重复/相关原型奇异；不能把原文的逆矩阵或强凸性论述无条件移植。本次512的资源约束及满行秩门槛明确属于适配边界。

## T_cond=mU 的系数核对

令V=mU，则V行质量1、列质量m/n、总质量m。原目标重写为：

\[
J=\sum_i\|X_i-R_i\|^2+{\lambda\over m}
[\sum_{ij}V_{ij}C_{ij}+\epsilon\sum_{ij}V_{ij}\log V_{ij}]
-\lambda\epsilon\log m.
\]

最后一项是常数。因此**只有同时把系数改为lambda/m**，完整目标及更新才等价。若直接在同一求和目标中用mU、保留lambda，则几何正则化强度增大m倍；它也可解释为另一种“均值重建项”约定，但不能宣称与论文求和目标的同一lambda等价。本工具直接使用概率U，固定lambda，不做m倍缩放。m增大时lambda/m变弱是该固定目标本身的性质，不能据测试结果另行加强。

## 旧实现的核查结果

| 项目 | 旧FastRef-Adapt-512 | 本次 |
|---|---|---|
| 提取 | DINOv2 ViT-L/14，672，48×48，L6/12/18/24，1024维 | 原DINOv3 H+/16，640/128切窗，118×118，L7/15/23/31，1280维 |
| 初始原型 | 固定Source32（73728行）可信2048-step ApproxGreedy的前512行 | 仅原八张Target-normal完整patch中的512个真实行 |
| Feature/scoring | 原始Source原型；余弦OT和1−cosine最终NN | Q/P保持同一raw欧氏尺度；最终euclidean/1280与原Source取min |
| 更新 | raw M、row-space projector；T_cond=mT；lambda=.1、epsilon=.05、外层2、Sinkhorn10 | 本文求和目标及概率U；固定lambda=.3、外层2；support-derived epsilon、严格收敛门槛 |
| Source | Source32替换参照 | 原冻结Source bank和MAD原样保留 |
| 五seed | 只随机划分正常评价子集；支持/原型固定，异常全集不变 | Seed42支持固定；全297 heldout；不再抽评价子集 |
| 评分聚合 | 双线性图、top1%均值、四层融合 | 原Source MAD、四层等权、原插值、全图max |

旧background fusion Image AUROC：Source32 70.8992±2.7185%，FR-Base-512 67.0012±3.0448%，FastRef-Adapt-512 52.9326±3.8354%。这些是不同信息/视野/评价集下的记录，不是当前K8的五次支持抽样，不与当前79.6226%直接相减解释算法贡献。

额外静态风险：`frbase_distance`用`np.asarray(...,float32)`再`np.ascontiguousarray`后调用FAISS原地归一化。若输入已经是float32连续数组，两步均可共享原feature内存；`score_image_joint`随后把同一feature传入FastRef，而P仍是raw Source特征。NumPy合成别名用例可证明这一路径会修改query，但旧记录没有逐层原始query的dtype/contiguity/调用前后范数，不能确认全部旧真实调用是否发生该别名，更不能把整体退化完全归因于它。本工具不对Q或P做原地归一化。

旧component诊断只覆盖Cholesky成功和投影对称性；首张query记录了更新/重建范数和行边缘误差，L12最大约0.00988、L18约0.00379，超过旧L6预检采用的1e-3。没有完整全查询/列边缘、质量、条件数、solve残差和固定目标检查；不能据这些记录宣称旧求解全域已数值验证，也不能先验断言某个缺失检查导致退化。

## 一次性数值约定及宿主适配

- lambda=.3：选择论文欧氏PatchCore路径的系数（arXiv §6.1；正式主文完整实现细节未直接通读），不继承旧余弦版本的.1；不称当前DINOv3最优值。重建明确sum。
- 外层2：主文Algorithm及实验说明。每张query从自身Q投影重新初始化；没有跨query状态。
- epsilon=.1×每层所选正常P在统一单位下的非对角成对平方距离中位数。这是本工具的支持集尺度约定，不声称论文明确给出；系数.1固定，不搜索候选或使用GT/评价分数。零尺度直接失败。
- float64求解和log-domain balanced Sinkhorn；全图统一边缘；每5步检查，最多1000步，行/列最大相对误差≤1e-6，总质量误差≤1e-8；不收敛失败。
- Gram条件数≤1e12，标准float64秩判据，规范化solve残差≤1e-10；数值失败保存诊断，不放宽阈值。
- 原型选择采用上表v3有界反密度适配：raw欧氏、无归一化/随机投影、最多100个均衡组，整层精确512个原始行；局部RandomState42一次创建、四层连续。保存实际索引/支持编号/物理坐标、配置、密度权重和priority、输入/原型身份及采样/距离代码SHA。明确不与原Source选择完全等价。
- 关闭细化直接检索P；lambda=0仍生成Q在P行空间的投影，所以不是关闭细化。
- 只改Target使用方式；SOURCE、UNION_K8为历史正式行，新分支共享同一P且不重新合入完整T。Source/八份Target历史全网格距离及两种图像分数仍按atol=1e-7、rtol=1e-6、score≤1e-5复核。

上述欧氏平方目标、行Gram和最近邻定义没有剩余核心数学歧义。仍未知：作者完整代码细节、正式PDF未直接通读的参数说明、原型真实条件性、全域Sinkhorn收敛/耗时/显存及正式评价。论文关于余弦与平方欧氏共同最优的论述不足以证明任意混合目标等价；本轮只实现补充材料支持的欧氏推导，不扩展余弦更新。真实prepare/smoke/infer/evaluate由用户后续在服务器明确执行；本轮不进行这些验证。

## 比较解释及验证边界

真实评价成功后程序生成 `evaluation/metric_comparisons.csv`，五指标差值均为分支减参照（数值比例单位）：A=`UNION_TARGET512 − UNION_K8`，衡量按固定v3反密度适配规则压缩Target的影响；B=`FASTREF_TARGET512_SOURCE − UNION_TARGET512`，衡量同一份P上的细化；C=`FASTREF_TARGET512_SOURCE − UNION_K8`，衡量整个适配相对完整强基线的变化。不预填真实结果。原signed距离审计仍为reference减branch，不能混淆两种方向。

A下降不能归因于FastRef；B下降仅说明这份原型/配置下细化有负作用。共享P控制本组比较的采样差异，不能推广到全部采样方式。没有增加多采样器或预算搜索，FastRef目标/更新、lambda/epsilon/迭代、Source/MAD、Exact NN和原评价后端均未改变。

本地缺Torch，真实CUDA距离后端未执行。v3测试执行实际生产分组、密度权重、精确无放回选择、runtime适配及prepare/缓存reader/普通求解代码；只替换缺失的距离/Tensor/CUDA接口，另执行上游实际AST定位旧循环。采样小预算/余数/重复/退化特征、RNG可复现、原始行及身份、共同P、旧产物拒绝、数学内核和差值方向均有覆盖，不声称真实Torch/模型链路通过。后续服务器prepare/smoke仍须核对原型、真实秩/条件性、CUDA成本、全查询收敛及原Source/K8复现；失败保留证据。本轮不连接服务器、不启动实验。

最终本地记录：38项专项通过（77.174秒），9份Python AST检查通过。实际只读preflight没有Torch/输出创建，采样尺寸条件通过；本地真实资产缺失及RAM不足仍为blocked。精确数量/循环终止/构造数据上的合法求解已验证；真实正常特征权重分布、Gram条件性、CUDA性能与完整297图数值/指标没有验证。当前具备独立review条件，本地修复不等于真实实验通过。
