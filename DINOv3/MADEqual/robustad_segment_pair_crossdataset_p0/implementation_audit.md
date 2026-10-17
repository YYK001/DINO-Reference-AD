# 实现审计：RobustAD 冻结 SEGMENT / PAIR

本文件描述实现和验证边界，不是实验结果。仅新增本模块、对应专项测试及本地输入盘点；不改旧代码、正式结果或报告。

## 实际复用调用链

| 环节 | 已读取并实际复用的入口 | 本模块适配与限制 |
|---|---|---|
| 数据与原划分 | `relation_reliability.datasets.robustad_full_records / split_source_normal`；`superadd_external.dataset_adapter._robust_split` | 开发时核对原完整发现；运行预检逐 split 发现，让缺失 Source 测试输入可独立标记 ARD pending。180 张 metadata 顺序、157/23 子序列与历史 manifest 逐项相同，不按文件名重排 |
| 固定 K8 | 原 `robustad_target_fewshot_p0/support.py` 的 manifest 语义 | 直接冻结 target_matched、support_index=1..8；不调用重新选择支持的函数；排除相同八张，异常一张不减 |
| 原宿主 | `relation_reliability.superadd_host.construct_exact_host`，经 `aebad_s_sparc_p0.runtime.load_host` 的冻结宿主加载包装 | 只复用加载器，**不调用 SPARC 校正/投影/构库**。原 config、checkpoint SHA、冻结参数、完整预处理和 CUDA device 设置原样检查 |
| Source bank | `relation_oracle_cache.extract.install_category_banks` | 先验原 NPZ float32、各层 100000×1280，再调用安装；严格从原 v6 类别目录取 bank，无 builder fallback |
| 支持特征 | `robustad_target_fewshot_p0.support._extract_four_layers`，实际来源为 `DINOv3/ShiftADD/aebad_s_p0/memory.py` | 原始 1280 维行，支持自身的真实网格；无抽样、归一化或统一网格要求 |
| Target 捕获 | `robustad_target_fewshot_p0.scoring.observe_fewshot_pil`；`aebad_s_union_support_loo.analysis.CaptureLog` | 单张完整 query 一次原提取，nearest/geometry 回调复制同次完整检索，source_support=None；逐层 Source+八支持，没有 Source-extra |
| Source 观测 | `relation_reliability.superadd_host.observe_pil` | Source 校准与 Source 测试独立沿用原 Source 路径（其原单 crop batch=1）；Target 沿用原 Few-shot 完整切窗 batch。没有为省显存改变任一路径 |
| Exact | `superadd_host._exact_top1_chunked` | 原 cdist 设置和 chunk；原欧氏距离 /1280 一次。仅额外计时包装，不改变输入/返回值 |
| 线段几何 | `aebad_s_segment_pair_crossdomain_p0.method.add_geometry` → `aebad_s_target_local_segment.geometry.geometry_layer / bounded_distances` | 原 float64 三边几何、舍入界与失败处理；保存完整端点索引、来源、alpha、长度和校正诊断。不改变误差界或裁零规则 |
| PAIR | `aebad_s_segment_pair_crossdomain_p0.method.predict / pair_gate` | 原 radius1、实际邻居票数、稳定 Top2；按完整物理 positions 还原，MetalParts 非方形网格不被当成方形 |
| CH1 | `aebad_s_ch1_scoring_inheritance.scoring.statistics / fit_image_mad / ch1_score` | 各类别自己的 23 张 Source；原 patch MAD 只读。原插值后峰值/原生有效 3×3 LOCAL；同类三个主分支共享状态 |
| 地图与全像素评价 | 已验证的 `aebad_s_projection_layer_diagnostic.numerics`、`metrics.exact_binary`、neighborhood `analysis.score_maps` | CPU 精确全像素排序，AP=average precision；同次地图/分数核验仍 1e-5。memmap 只作为临时排序输入，不修改完整像素集合 |
| AUPRO | `relation_reliability.official_metrics.official_aupro_adeval` | 独立显式 CUDA 阶段、200 阈值、FPR≤0.3，复核 CPU 地图/GT 哈希；无 CPU 后端替代 |
| 实例与机制 | crossdomain `analysis.spatial / mechanisms / mechanism_summary / group_small_instances / margin_repairs / historical_comparisons` | 只调用独立数学/统计函数；按 category×shift 分组，不调用其 AeBAD context、run 或双域 macro。原图 8 连通、16px Chebyshev 环带、own/fixed-UNION 峰值、小实例 ceil(N/4)、NA 均保持 |
| CPU 并行 | neighborhood `prediction.bounded_images` | 有界线程队列；主线程顺序保存进度，线程仅处理图像计算。类别/shift 及 GPU 宿主不复制 |

## 冻结身份与旧回退的处理

原 Few-shot 的 `runner._try_external_bundle` 在外部输入不全时可能返回 None，并使旧 runner 进入 `build_exact_official_memory` / `_fit_source_mad`。本工具不运行旧 runner，也不调用这些构建入口。缺原 bank 或 patch MAD 明确失败。

`frozen_catalog.json` 包含两类原正式文件 SHA、原 Source 完成/校准/sidecar/provenance SHA、完整原训练顺序、校准子集、Source 测试名单、各 shift 支持/评价名单与图片/GT SHA。目录名不作为可信依据。局部或全域都使用相同冻结清单。

历史 Source 归档没有 bank 字节 SHA，其旧 Source identity 只包含 bank 的文件大小及 Source 元数据哈希。新工具首次显式执行从指定原树登记 bank SHA，续跑严格核对；不声称首次登记可以证明历史字节身份。初次必须由用户提供原 bank，不能重建替代。恢复 23 张原校准距离后核对层顺序、原 ECDF patch 数量、原 patch median/scale；只拟合 CH1 image MAD。

输入/协议/代码身份发生变化要求新输出目录。已有逐图缓存先对旧账本 SHA，后做结构与数值核验；不重置或重新授信。未登记孤立/临时文件拒绝。已有完整缓存优先于外部特征和显式补算。预测 progress 同样保留首轮 SHA；已完成预测的完整集合及地图/评价 sidecar 均封存。

`capture` 先按选中 scope/shift 和已验 SHA 账本确定待捕获 Target，只有存在缺项时才加载外部特征清单（每次调用最多一次）。完整缓存、Source-only 不读取无关 feature-root，也不重置已登记外部清单 SHA；真正需要该清单时，仍拒绝缺失、身份不符或历史哈希变化。

`source_raw_root` 的旧缓存按原 Source identity 与 `_record_identity` 实际定义验证，并递归兼容真实 `images/source__source/*.npz` 布局。它们保存 Source-only 观测，不包含本次完整 Target 几何/Q，不能直接充当 Target raw。首次合法复用后登记旧 NPZ SHA，之后拒绝修改。

有 Target 的历史 Source raw 时额外比较 Source 距离，保留最大/平均绝对及平均有符号误差和原 SHA；这项明确标记 `historical_distance_audit_only`，不是同次回调一致性。历史距离是否通过原严格距离界与历史 CURRENT 的跨环境兼容验收分别记录。没有历史 raw 则只审计历史 CURRENT 分数及排序/图像指标变化，不能宣称逐距离复现。续跑从缓存元数据保留该历史距离审计，不丢弃旧记录。

## 统计、完成状态和资源

Source 测试 MetalParts=171、PCB=195，均独立于 23 张校准、且只查 Source bank。三个 Target 适配分支共用该类别 SOURCE_CH1 测试参照。ARD 对每 shift/metric 先 min(0,Target−Source)，再类别内 shift 等权、两类别等权。Source 缺项、GT 缺项、AUPRO pending 不会标全部完成。

历史 CURRENT 的 1e-5 严格审计和预先冻结的 1e-4 兼容验收（均 rtol0）单列，带有符号误差；兼容通过不叫严格复现通过。同次距离仍 1e-7/1e-6，同次内部评分仍 1e-5。不会失败后自动修改这些常量。

Source 恢复/模型加载、特征提取、Exact、支持处理、几何、CPU 图像/像素评价、AUPRO 分别记录计时；记录 RSS、显存峰值和缓存体积，Source bank 增量为 0。预检内存和磁盘是估计，不是实测峰值或耗时预测。按原图尺寸推导预算；全部 11 组的新 Target raw/支持/预测未压缩估计约 19.47 GiB，不含 Source 观测、CPU 临时排序和备份副本。

原始图片：MetalParts Target 1500×2681（预估原生58×104），PCB Target 2048×2048（预估80×80）；实际网格以捕获的 positions/几何为准。PCB Source 校准原 ECDF 每层 230000 个值，与 Target 80×80 不同，不把其网格强制改成 Target 网格。

少量案例固定每 shift 按 sample_id 排序取首张正常和首张异常。原图/GT/门控和三个定位图共显示范围只用于绘图，定位分数没有逐图归一化。全部实例表而非案例决定总体诊断。

## 本地输入盘点（2026-10-17）

逐项路径和身份预检见 `input_inventory_local.json`。

- 两类原 Few-shot 正式三个必需文件、Source 原完成/来源/patch MAD/sidecar、dataset_manifest/protocol、原 RobustAD 图片与 GT：本地已存在并与冻结身份匹配。
- 指定 no_banks 目录中两类 official_banks.npz：本地缺失。AutoDL 原树尚未检查，不据本地缺失推断服务器缺失。
- 原23张逐图 Source 校准距离与本工具 CH1 image MAD：尚未到位，后续显式恢复；已有原 patch MAD不能冒充imageMAD。
- `server_results/relation_oracle_seed42_cache_and_audit/mvtec_robustad_seed42_v2`：发现 MetalParts 171/171、PCB 195/195 Source 测试缓存，并覆盖两类 Target 的 Source-only 距离。该旧归档不含 23 张训练校准缓存。已检查头部和名单覆盖，未用缺失 bank 猜算 Source identity，尚需运行端原 bank-dependent 身份核对后才复用。
- 默认 `external_baselines/dinov3` 入口本地缺失；发布副本 `.sync/DINO-Reference-AD/external_baselines/dinov3/hubconf.py` 存在，但不是默认运行路径，也未改动该副本。AutoDL 默认 `/root/autodl-tmp/dinov3` 待用户检查。
- 默认 checkpoint 本地缺失；AutoDL 历史路径待用户确认。新 Target Q/几何/支持缓存不存在，不能用历史 Source-only 距离恢复。

## 验证边界

本地只运行 Python/NumPy/SciPy CPU、小型合成数据和只读真实输入检查，临时文件在 G 盘并由测试清理。不安装依赖、不访问服务器、不调用真实模型/GPU。AUPRO 集成测试替身只覆盖阶段接口、200 阈值传参及 CPU 地图身份，**不覆盖真实 Torch/CUDA/ADEval**。

复用了既有非方形 positions、边界票数、稳定 Top2/退化端点数学测试；新增类别独立 23 图校准、原 MAD 不被替换、缓存/状态修改拒绝、无模型完整缓存续跑、预测中断续跑、内部评分漂移阻断、完整 CPU Source/Target 评价、Source/AUPRO pending、两层宏平均/ARD先截断、review包与原缓存区分、默认禁导入 Torch 等验证。

真实原 bank、DINOv3 CUDA 加载与完整提取、23图补算、全域历史兼容验收、真实五指标/ARD、真实耗时/RAM/显存和原始缓存备份仍须 AutoDL 验证。没有真实实验结果、收益预判或 SOTA 结论。

已执行：23 项专项测试全部通过（7.910s）；相关既有90项回归测试通过（与当时本模块22项合跑为112项，155.777s）。本模块与测试共9个 Python 文件 AST 通过，手动 Bash 入口语法检查通过。只读真实输入预检的身份错误为空、未导入Torch；本地缺原 bank、默认 checkpoint/源码路径和 ADEval，因此返回 blocked/2，未启动实验。

同日外部特征按需加载修复后：26 项专项测试通过（8.692s）。新增三项测试覆盖完整缓存与离线 feature-root、缺 Target 缓存时的 Source-only、Target 缺项时外部清单缺失及历史 SHA 修改仍被拒绝；未运行模型/CUDA。
