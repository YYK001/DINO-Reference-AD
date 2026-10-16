# AeBAD-S/background 四案例 Target support 最近邻追踪

独立入口：python -m DINOv3.MADEqual.aebad_s_target_support_trace.runner。
默认 preflight；显式 trace 才进入 CUDA、特征提取与最近邻检索。
本窗口仅完成代码与本地轻量验证，未连接服务器或执行真实特征提取/GPU 检索。
下面命令供用户后续另行授权执行。

## 固定范围与复用

| 案例 | 原路径 |
|---|---|
| test_0887 | test/good/background/IMG_7728.png |
| test_0921 | test/good/background/IMG_7765.png |
| test_0545 | test/fracture/background/IMG_9385.png |
| test_0023 | test/ablation/background/IMG_8863.png |

Target 顺序为 IMG_7785、IMG_7764、IMG_7771、IMG_8149、IMG_7778、IMG_7752、
IMG_7799、IMG_7747（均为 .png，位于 test/good/background/），直接读取原 manifest。
两个正常图是残余高响应案例；两个异常分别为已有 margin 下降与上升对照。
这些是事后机制案例，不用于调参或宣称全域效果。

正常图取 Source/union 整图峰值；异常图分别取两种图的整图、GT 实例内及非缺陷环带峰值。
复用 spatial.instance_regions：原 GT 上 Chebyshev 半径 16 扩张，排除全部 GT，
按原 nearest resize 映射到输出网格。异常必须各有一个实例。
区域峰值并列取 row-major 首个；坐标去重并保留全部角色。
GT 只选事后解释位置；整图峰值不默认是缺陷，GT 补集不叫背景。

复用 replay12 的身份/MAD/raw 校验，verified_replay_scope 的 required_cases 固定为本次四案例。
启动和结束均只要求这四张 raw 存在并符合历史 SHA；其余八张可缺失，不参与本次校验。
仍要求原 replay12 的完整 verified 历史记录及正式输入身份不变。
原 union 诊断未传 required_cases，继续要求完整 12 张缓存。
MapBackend 恢复四案例原地图，先对照原 CSV、case_peaks、case_peak_layers、
实例 max/mean/margin 和环带 max。interpolation_stencil 取每位置至多四个实际 patch，
保留实际 grid、flat index、权重和 L7/L15/L23/L31。

缺失 query 特征时，完整原图调用 image_tensor → model.preprocessing →
_extract_processed_batch(query_masks=[None], include_identity_descriptors=False)，
与 observe_fewshot_pil 的提取路径相同；保存完整前向后选中的物理 patch 行。
不插值特征、不对小裁剪图前向、不执行其 Source/Source-extra 检索。
支持优先复用 replay 记录 SHA 一致的 Target 缓存；
缺失时仅原八张调用 extract_or_load_support，写入新输出。

每层每个独立 query patch 分别搜索八个完整支持 bank：
_exact_top1_chunked 返回 distances/indices，欧氏距离仅除以 1280 一次。
保留全部八张的最近距离、patch 索引和排名。
图内并列取原 min 返回的首个 flat index；跨图并列按较小 manifest 编号。
d_S 来自已验证 official_raw；d_T 本次重算；d_ST 同时核验原 target_raw_8 和 min(d_S,d_T)。
再经原 Source MAD 和插值权重汇总位置响应。

固定距离容差 atol=1e-7、rtol=1e-6，沿用缓存 min 校验；分数/贡献绝对容差 1e-5。
最大相对误差分母为 max(|reference|,1e-12)，不提供放宽容差选项。
matmul precision/TF32 必须与 replay 环境相同，不自动修改；
软件版本差异另行记录。未通过的位置保存真实误差/候选并标 UNVERIFIED。
不加载 Source memory、不需要 Source-extra 特征、不拟合 MAD、不调用全量 run 或重评检测指标。

## 输入与 preflight

需要原 sample_scores.csv、support_manifest.csv、Source MAD、complete.json、memory_manifest.json；
verified replay12 的四案例 raw_cache、replay_status.json、replay_inputs.json、support_provenance.csv、
environment_*.json；已完成诊断的 selected_cases、case_peaks、case_peak_layers、defect_instances、
verification.json、diagnostic_status.json；四 query、八支持原图和两个 GT。
还需原官方 host/config、DINOv3 代码、SHA 已批准的 ViT-H+/16 权重及可用 CUDA 环境。

默认支持缓存为 replay12/support_features/target_matched/background/01.npz…08.npz。
默认可选 query 缓存为 replay12/query_features/heldout/background/{sample_id}.npz。
支持使用现有 SupportFeatures 格式。可验证 query 格式包含 protocol_id、relative_path、
grid、flat_positions、contract、geometry、layer_7/15/23/31。
四层均为 float32 [selected_rows,1280]；物理索引严格递增、覆盖所需位置；
contract 绑定原图/权重/原代码 SHA、层序、预处理与精度，geometry 记录实际拼接 ROIs。
可通过 --query-cache-dir 指向之前 trace 输出的 query_features。
同时保留其父 trace 目录的 feature_provenance.csv；已有 query 特征必须与历史记录 SHA 对齐，
不能在新 preflight 中把被修改的特征重新认可为新 checksum。
不接受仅 shape 相同而无原提取 provenance 的 query 特征。
非法已有缓存不覆盖、不静默重新提取。

--support-cache-dir 也可指向先前 trace 的 support_features/target_matched/background。
原 replay 路径仍严格匹配 replay 的旧字节 SHA；其他路径若与旧 SHA 不同，
必须保留父 trace 目录的 feature_provenance.csv、trace_inputs.json、trace_environment.json。
检查该支持图的历史 SHA、原图/权重、manifest 编号、层序、原提取代码、
正式输入/MAD、预处理、精度及拼接几何；来源不同或记录后修改均拒绝。
新 trace 将提取合同写入 provenance；旧 trace 无合同列时从已有 inputs/environment 复核。
通过输入检查后仍执行本次 Target/min 距离和位置响应复核，来源认可不等于对应已验证。

preflight 检查四 query、八 support、两个 GT（14 个图像文件）、原缓存/MAD/身份、
代码/config SHA、特征头及可发现的依赖，不导入 Torch/加载 backbone/提取特征。
CUDA、依赖运行时与特征有限值在 trace 验证。
ZIP 只供本地 preflight；trace 要求解压数据。部分数据保留原 CSV ID，不重新编号。
解压目录只需四 query、八 Target support 和两个 GT，不要求 train/good。
--data 可指向实际 AeBAD_S 根，或包含 AeBAD_S / AeBAD/AeBAD_S 的上级目录；
按唯一 test 根识别，解析出的真实根统一用于 preflight、trace 读图及支持提取。
输出与所有输入互不包含；只能为空或仅含本工具 preflight.json。
trace 不续写已完成/失败目录；重做使用新输出并只读复用可验证特征。

## 服务器命令（由用户后续授权执行）

以下路径沿用已有服务器结果，preflight 会检查实际存在性。未附加自动关机。

~~~bash
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export PROJECT_ROOT=/root/autodl-tmp/DINO-NVS
export PYTHON=/root/miniconda3/bin/python
export RESULTS_DIR="$PROJECT_ROOT/DINOv3/outputs/mad_equal/aebad_s_target_fewshot_seed42_v1"
export REPLAY_DIR="$PROJECT_ROOT/DINOv3/outputs/mad_equal/aebad_s_union_background_replay12_seed42_v1"
export DIAGNOSTIC_DIR="$PROJECT_ROOT/DINOv3/outputs/mad_equal/aebad_s_union_background_replay12_seed42_v1_diagnostic_20261003_150831"
export AEBAD_ROOT=/root/autodl-tmp/datasets/AeBAD/AeBAD_S
export OFFICIAL_ROOT="$PROJECT_ROOT/external_baselines/SuperADD"
export DINOV3_ROOT=/root/autodl-tmp/dinov3
export CHECKPOINT="$OFFICIAL_ROOT/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"
export SOURCE_MAD="$RESULTS_DIR/source_mad_state.npz"
export TRACE_OUTPUT_DIR="$PROJECT_ROOT/DINOv3/outputs/mad_equal/aebad_s_background_support_trace4_$(date +%Y%m%d_%H%M%S)"
export DEVICE=cuda:0
cd "$PROJECT_ROOT" || exit 1

"$PYTHON" -u -m DINOv3.MADEqual.aebad_s_target_support_trace.runner preflight
preflight_rc=$?
printf 'PREFLIGHT_EXIT_CODE=%s\n' "$preflight_rc"
~~~

仅在用户另行授权后，在同一终端、同一 preflight 输出目录执行 trace。
日志写在目录外，避免新日志触发非空目录保护。

~~~bash
trace_started=$(date +%s)
"$PYTHON" -u -m DINOv3.MADEqual.aebad_s_target_support_trace.runner trace \
  2>&1 | tee "$TRACE_OUTPUT_DIR.log"
trace_rc=${PIPESTATUS[0]}
trace_elapsed=$(( $(date +%s) - trace_started ))
printf 'TRACE_EXIT_CODE=%s\nELAPSED_SECONDS=%s\nOUTPUT=%s\n' \
  "$trace_rc" "$trace_elapsed" "$TRACE_OUTPUT_DIR"
if [ -f "$TRACE_OUTPUT_DIR/trace_status.json" ]; then
  cat "$TRACE_OUTPUT_DIR/trace_status.json"
fi
~~~

--help 提供 --project-root、--results-dir、--cache-dir、--diagnostic-dir、--data、
--official-root、--dinov3-root、--checkpoint、--mad、--support-cache-dir、
--query-cache-dir、--output-dir、--device。support-cache-dir 直接包含 01…08.npz。

## 输出与人工边界

| 文件 | 内容 |
|---|---|
| trace_positions.csv | 全部位置角色、原坐标、GT/环带身份、原/追踪响应与位置核验状态 |
| query_patches.csv | 每位置 × stencil patch × layer 的 flat index、权重及 tile/token 投影 |
| nearest_support.csv | 每个物理 query patch × layer × 8 张支持的完整候选、索引、排名与缓存参照 |
| trace_verification.csv | 原响应/实例/ring、逐 patch 距离、逐层 MAD 贡献与位置复核 |
| feature_provenance.csv | 实际特征来源、SHA、grid/拼接几何及耗时 |
| trace_status.json / trace_report.md | 实际范围、前向次数、失败/未验证状态、误差与耗时 |
| trace_inputs.json / trace_environment.json | 输入 SHA、原精度、当前版本与 CUDA 记录 |
| figures/ | 每位置一张，固定显示最大权重 query patch 的四层最优 Target 对应 |

最大权重并列按较小 flat index；不挑选匹配。全部参与 patch/八张候选保留在 CSV。
红十字为中心，橙框是通过实际 preprocessing 和 axis_patch_split 的
input/prediction/result ROI 恢复的投影输入 token 单元，不是完整感受野。
全图与固定 ±64 原像素邻域同时展示；裁剪只观察、不参与特征计算。

nearest_support 的 correspondence_verified 仅是该 patch 的 Target/min 距离核验；
具体位置原响应还需 trace_positions 和 unverified_position_ids。
只有 trace_status=verified_selected_correspondence 且
precise_original_correspondence_allowed=true，才允许作为选定位置的精确对应证据。
这不是全域推理复现。

query_region / support_correspondence 初始 unreviewed。
查询区域人工选产品内部、产品边缘、图像截断处、背景、无法判断；
支持对应人工选视觉上相似正常结构、不同部位、背景、无法判断。
不自动生成因果判断，不增加分割、前景阈值、约束或新评分。

退出码：0 为对应 mode 检查通过；2 为 preflight 阻塞；3 为存在未通过对应；
其他非零为运行失败，保存已有实际表格/状态/错误，不覆盖原输入。

## 本地验证

已执行：三个测试模块合计 57 项通过；新增 Python 源码及测试通过编译检查；
两段服务器命令通过 Bash `-n` 语法检查（未执行命令）。

~~~powershell
Set-Location 'G:\codex\workspace\experiment'
& D:\anaconda\miniconda\envs\chinamoblie\python.exe -m unittest DINOv3.tests.test_mad_equal_aebad_s_target_support_trace DINOv3.tests.test_mad_equal_aebad_s_replay12 DINOv3.tests.test_mad_equal_aebad_s_union_diagnostics
~~~

覆盖固定身份/顺序、峰值去重与并列、共享16px环带、原stencil、
动态grid与flat index、原axis_patch_split几何、逐图候选与显式拼接一致、
误差拒绝、原MAD负响应、query合同/稀疏行、无Torch帮助/preflight、
无Source/full-run调用、目录保护及合成展示。
另覆盖只有四 raw/14 图像且无 train/good 的完整 preflight、嵌套根统一读取、
范围外缓存缺失不阻塞、范围内缓存缺失/修改及历史/正式身份变更拒绝、
原诊断保留12张限制、历史 trace 支持复用与文件/冻结输入变更拒绝。
逐图/拼接检索使用 NumPy 精确参考回调测试实际聚合逻辑；
生产 PyTorch/CUDA _exact_top1_chunked、真实特征提取与真实对应图未执行。

本地真实文件 preflight 已检查四案例、八支持、14图像文件与原缓存；
显式阻断Torch导入，原正式/补跑/诊断输入SHA不变。
报告 blocked：缺少Torch、checkpoint、官方DINO hub；本地官方config/model.py
SHA与下载replay记录不同。未修改这些文件或现有环境。
服务器文件与GPU条件须以后在服务器验证。
本地报告：DINOv3/outputs/mad_equal/aebad_s_background_support_trace4_local_preflight_v1/preflight.json。
