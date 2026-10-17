# 实现审计：illumination / view 冻结 SEGMENT 与 PAIR

日期：2026-10-16。工程根目录：`G:\codex\workspace\experiment`。仅新增独立模块与专项测试，未改旧 Python、历史报告、memory、MAD 或正式结果。本轮未连接服务器、未运行真实模型或 CUDA。

## 2026-10-17 更新：固定跨环境分数验收规则

以下更新替代本文早期“历史分数绝对容差1e-5立即阻断”的行为；内部数值复核与可选原距离复核不变。用户要求从头运行，本轮固定历史 Source/Union CURRENT 准入 `atol=1e-4, rtol=0`，保留原1e-5检查作为严格复现审计。`verification.historical_score` 在 capture、predict、evaluate 统一使用；不改历史CSV，不修改旧失败记录，不抬高全局数值容差。

用户提供的实测：T4/Torch2.11 在 illumination/test_0240 的 Source/Union 分数偏差分别1.95503235e-5和1.62124634e-5；原Torch CPU插值对照仍偏离历史2.00271606e-5和1.57356262e-5。无法仅归因于NumPy插值，尚未确认上游具体原因。1e-4是此轮重启前明确冻结的兼容验收界限，不是模型参数，不代表所有误差已被证明无影响，也不自动随失败增大。

协议新字段纳入joint identity；同一图像/特征提取身份保持不变，新旧运行不可混用。逐条保留actual/historical/signed_error、strict_verified和compatibility_warning；状态分别报告历史严格复现及兼容拒绝数量。正式指标完成状态只表明按本轮验收规则完整运行，不能替代strict字段。

CPU评价新增Source/Union CURRENT本次对历史的图像AUROC/AP、排序和有符号误差表；三分支正式比较仍全部使用同一当前环境重新计算。AUPRO及分域汇总携带审计文件。未执行真实GPU或两域全量实验；已提供实测的两项误差仅用于测试验收分支，不冒充模型复测。

本地现有环境专项44项通过（11.914秒）：实测分数对的兼容/严格双记录；小偏差贯穿完整缓存复核、预测和CPU评价；大偏差仍拒绝；NaN/Inf拒绝；内部地图2e-5漂移仍被原容差拒绝；身份隔离；AUPRO审计文件传递；CPU导入不进入Torch。原background12张668352层×patch的距离/门控逐元素一致，36项CH1分数最大误差0。模型、CUDA、Exact边界及AUPRO后端仍是明确替身，未作真实环境验证。

## 调用链与解除 background 限定

| 环节 | 实际复用入口（相对上述根目录） | 新封装的责任 |
|---|---|---|
| 支持/集合 | 正式 Few-shot `support_manifest.csv / sample_scores.csv / complete.json` | `frozen_catalog.json` 固定两域原 ID/顺序/标签/图像字节身份；`inputs.context` 再对照 CSV 和数据全集，不沿用旧 297/案例筛选 |
| 原 Source | `aebad_s_target_fewshot.runner._load_memory` | 显式 capture 内懒加载；固定 SHA、source_count=521、四层各 100000×1280；不调用 builder |
| 宿主 | `aebad_s_sparc_p0.runtime.load_host` → `superadd_host.construct_exact_host` | 仅复用原宿主构造/冻结/设备修正，不调用任何 SPARC 校正；原 host_config 和 checkpoint 校验 |
| 单次完整查询 | `robustad_target_fewshot_p0.scoring.observe_fewshot_pil` | 新域原图一次完整提取，source_support=None；原 4 Source+32 Target Exact 调用，回调拿同次副本 |
| 同次日志 | `aebad_s_union_support_loo.analysis.CaptureLog / validate_arrays` | 只复用纯数组日志及格式验证，不调用 LOO/background runner；新 tool/domain/identity 记录，不伪造旧背景身份 |
| 全支持特征 | `robustad_target_fewshot_p0.support._extract_four_layers / _atomic_npz` → `ShiftADD.aebad_s_p0.memory._extract_four_layers` | ZIP/目录读图适配；不增强、不压缩、无 coreset，原 row-major patch 顺序 |
| SEGMENT | `aebad_s_target_local_segment.geometry.geometry_layer / bounded_distances` | 八支持 Top1 稳定选不同支持的 Top2；真实特征 FP64 边长；保留 alpha、重复端点、三边失败、原修正规则。原内核内部计算的 MIDPOINT 未作为本工具分支或结果使用 |
| PAIR | 原 `aebad_s_neighborhood_segment_p0.gates.scatter`；新 `method.pair_gate` | 仅抽出无序支持对投票，保留物理 positions/真实邻居/ceil(n/2)。不调用旧 `neighborhood` 的同网格和位移验证，不引入 NEIGHBOR 规则 |
| CH1 | `aebad_s_ch1_scoring_inheritance.scoring.statistics / ch1_score` | 全部分支共用同一冻结 Source image MAD；CURRENT 只作原宿主审计 |
| 定位 | `aebad_s_neighborhood_segment_p0.analysis.score_maps` → `projection_layer_diagnostic.numerics.reconstruct` | 原 patch MAD、四层平均、原 CPU 插值，不逐图归一化、不裁负响应 |
| 像素 ROC/AP | `projection_layer_diagnostic.metrics.exact_binary` | 三分支全部重算完整像素、tie 分组 AP，无量化、无降采样。未据只有 CURRENT 分数就引用历史 Union 像素值 |
| 空间 | `aebad_s_union_diagnostics.spatial.components / instance_regions / resized_mask`、原 `adaptive_max` | 原图 GT 八连通、16 像素 Chebyshev 环带、排除全部 GT；冻结预测后才读 mask |
| AUPRO | `relation_reliability.official_metrics.official_aupro_adeval` | 独立显式 CUDA 入口，200/FPR≤0.3；重建地图/mask SHA 与 CPU 完全相同，不可用则 pending |
| 并发/资源 | 原 `bounded_images / protect_dependencies / available_ram` | 有界 CPU worker、逐域全像素排序、显存和磁盘保护；无后台监控或自动安装 |

以上生产复用文件的 SHA 纳入 joint identity。旧工具的 `inputs.context`、固定12图筛选、297图完成条件及背景归档哈希未用于新域身份判断。校验函数接受原通用数组 schema，不等于把新域伪装成旧实验。

## 冻结身份与输入盘点

`frozen_catalog.json` 的 SHA 固定为 `3a0082fa6c75e4b04b4c0c4b7be27287afb7422157b3357c79eeafbb133af77c`。正式清单重新核对为 illumination 265=67+198，view 364=84+280，各自八支持按原 support_index。另对 ZIP 中原完整 test 排序得到的原 sample_id 作过生成期检查；运行期再核对所有记录、支持排除、字节 SHA 和域内图片全集。

原 Source memory SHA：`1b9e00c85fd6be35ba365d378b1da7fb2a673400319a256bb62408fe6e001717`。

checkpoint SHA：`7c1da9a54b3bdb333f5ebc42e404b7f19b1b5bed504877623c9dc87397f41488`。

patch MAD 文件 SHA：`65df0a748d918f20e5c85f2dbb73845a2fac2801d8d62a91e3bd0b7bb37e1a13`。

Source image MAD 文件 SHA：`d185432e7c2e67c14f437426be05df05a3bc2d021973efd9aae40130fd048f74`；内部 state SHA：`8784052402f4ff389c1e8bd80ffeb1eeff1c66b58d2fda76f842a680cb58f558`。不再恢复66张校准图或重新拟合。

本地已找到 formal 清单、原图 ZIP、两级 Source MAD；未找到两域完整 raw、可信 Q、支持特征、原 memory、模型权重及官方 DINOv3 root。正式轻量归档内只有清单和 MAD，不把它当完整缓存。具体搜索范围、检查过的归档成员和路径见 `input_inventory_local.json`；不声称已检查未知归档中所有可能位置。

## 方法不变项和验证边界

* L7/L15/L23/L31、1280d、640窗口/128overlap、resize=.625、downscale=4、NoAug、seed42；最高 float32 matmul 精度、matmul TF32=False、cudnn TF32=True。
* Euclidean/1280 只在原检索日志生成时除一次；几何沿原尺度，真实端点平方距离除 1280²；最终距离不重复除。
* PAIR 只依赖本图本层端点支持来源。交换无序端点、重排 compact 行不改变物理门控；不传播门控，不看支持坐标或 GT。
* `SEGMENT<=PAIR<=Union` 为原 atol=1e-7 / rtol=1e-6。失败不裁剪。原分数绝对容差1e-5，预测与离线评价均复核；地图后端漂移不能靠内部自洽冒充复现。
* 可选历史 observation 对照直接读取原 `official_raw / target_raw_8`；后者明确是联合距离。无参照时只声明 CURRENT 分数复现，不声明距离逐元素复现。
* A/B/C 优先复用，默认不落盘完整 Q。cache_status 中逐图 SHA 在任何状态写入前校验，保留历次调用次数；被改/孤立/丢失 raw 拒绝，不重新认可。
* 小实例是每域 ceil(N/4) 面积分层，NA 留在分母和计数中；与 background 三个最小实例不是同口径。R 按图像等权再层等权，零分母 NA，不裁负差。
* 三分支全域预测冻结后才评价。域可以分批，但输入和方法 code identity 必须相同；单域完成不冒充二域完成，二域等权 macro 排除 background。

## 实际本地验证

专项测试使用现有 Python 3.11.12 / NumPy 1.26.3 / SciPy 1.14.1，不安装依赖。涵盖物理 positions、无序对、中心与边界、n=0、接受/拒绝回退、原真实端点几何与三边失败、支持不同网格、缓存防篡改续跑、完整冻结和小型 CPU 端到端评价、实际地图后端漂移、实例修复/新增损失、NA 小实例、macro 与精确像素 tie AP。

最终新专项 **30 项通过（9.704 秒）**；旧 neighborhood 专项 **24 项通过**（此前与28项新测试合并52项，12.030秒）。新模块10个Python文件及新测试1个文件均通过内存AST解析。修改评价标签不改变预测、缺范围内GT不能发布完成指标也已覆盖。

**真实 background 固定12张回归**：读取已有 LOO B+SEGMENT raw 及本地冻结 PAIR 预测。逐元素 PAIR 距离、gate、votes、neighbor_count 完全一致；共 **668352 个层×patch**；三分支共36个 CH1分数的最大绝对误差 **0**。没有重跑 background 模型或297张全域评价。

capture 的 C 路径与 B 路径另有集成替身检查：原 CaptureLog、几何、缓存、身份和分数验证执行真实代码，Torch、模型、特征提取、Exact 检索边界用明确替身。AUPRO 续跑测试使用后端替身，检查六次原接口调用、200阈值和地图身份。**这些不证明真实 Torch/cdist/CUDA/DINO 或两域指标已通过。**

默认预检读取真实正式清单及相关图像/GT字节，Torch 不导入，模型不构造，不创建实验结果。由于真实必要资产缺失，预检输出 blocked；缺项不影响交付工具与轻量验证。

## 仍待真实环境验证

用户恢复/确认原资产后，手动 capture 验证原宿主、Torch Exact 精度、一次完整提取和实际资源峰值；逐图 Source/Union CURRENT 复现失败即停止。然后显式预测、CPU评价、AUPRO，不能以本地替身成功替代。

尚不能回答 SEGMENT 在两个新域是否提高图像区分、PAIR 是否在保留正常收益时减少定位损失、效果是否主要适合 background，以及新域是否仍有环带恢复但缺陷未恢复。工具会在真实运行后根据逐域图像/像素指标、R、逐实例修复与新增损失自动生成客观报告；当前不预填研究结论。

可交给独立窗口 review 本地实现；进入真实实验仍须通过资产及正式分数复现检查。

## 2026-10-16：显式 T4 显存准入

Kaggle 已通过实际输入预检（原 memory/权重、正式六项、Source image MAD、两域图片/GT 字节及支持排除），以及 SuperADD、ADEval 1.1.0 和官方 DINOv3 hub 导入。两张T4各空闲约14.46 GiB；这些检查尚未加载模型。用户选择不增加单图验证，直接尝试完整补缓存。

仅增加 `--capture-min-free-vram-gib`（默认20，T4显式14，须有限且≥8），由 runner 验证并传给 capture，preflight资源报告反映所选门槛；仅Q检索仍固定8。门槛和实际加载前显存逐调用登记。参数不改变 batch、dtype、提取/距离/几何/评分内核或容差，不限制CUDA allocator，不自动重试/降级。失败沿用原保护逻辑保留逐图记录及错误。新代码产生新 joint identity，旧产物不重新认可；冻结 catalog/资产和提取设置保持原定义。

新增轻量验证覆盖14.46/14准入、原20门槛拒绝、仅Q的8 GiB边界、非法数值、参数报告和模拟加载OOM后保留旧缓存。本轮完整专项 **38项通过（11.016秒）**；3个改动Python文件通过内存AST解析，固定12张旧缓存的668352层×patch回归仍逐元素一致、36个CH1分数最大误差0。方法、输入、预测、评价、参照模块与冻结catalog保持修改前字节。Torch、CUDA和模型边界为明确替身，不代表T4实测成功；本窗口未执行真实模型或连接Kaggle/AutoDL。
