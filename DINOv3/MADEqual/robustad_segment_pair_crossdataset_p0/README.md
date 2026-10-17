# RobustAD MetalParts / PCB：冻结 SEGMENT / PAIR 跨数据集验证

本工具只迁移已冻结的 AeBAD SEGMENT、PAIR 和 CH1 评分，不训练、不调参数。实验在用户手动启动的 AutoDL 上执行。开发阶段只做本地实现、输入核对和合成 CPU 测试，未运行真实模型或实验。

## 固定范围与方法

| 类别 | shift | 正常 | 异常 | 评价图 | 支持图 |
|---|---|---:|---:|---:|---:|
| MetalParts | background_1 | 52 | 40 | 92 | 8 |
| MetalParts | background_2 | 52 | 40 | 92 | 8 |
| MetalParts | lighting / position / rotation / scale（各） | 112 | 80 | 192 | 8 |
| PCB | lighting | 112 | 90 | 202 | 8 |
| PCB | white_balancing | 112 | 60 | 172 | 8 |
| PCB | rotation / position / shadow（各） | 72 | 60 | 132 | 8 |

总计 11 个 shift、1722 张 Target 评价图、88 张固定支持图。支持身份和顺序来自原 `support_manifest.csv`，所有分支排除相同八张支持，保留全部异常。没有 PiledBags、Source-extra 或 K1/K2/K4。

三个主分支：

- `UNION_K8_CH1`：原 Source 距离与完整八张 Target memory 距离取 min。
- `SEGMENT_K8_CH1`：沿用原稳定 Top2、原始端点特征和 `geometry_layer` 有界线段距离，不压缩 Target memory。
- `SEGMENT_PAIR_GATE_K8_CH1`：逐层原生物理网格上，半径 1、排除中心；无序端点支持对相同才投票，票数至少为实际邻居数的一半向上取整。通过用 SEGMENT，否则退回 UNION；无邻居退回 UNION。边界不补邻居，不检查支持坐标相邻。

沿用冻结 DINOv3 L7/L15/L23/L31、1280 维、原 640/128 切窗、resize=.625、evaluation_downscale=4、float32、最高 matmul 精度、禁用 matmul TF32、原 Exact NN；欧氏距离仅除 1280 一次。不添加 RGB、归一化或新融合。各支持和 query 独立保存真实网格，不要求尺寸相同。

Source patch MAD 直接读取各类别原 `calibration_state.npz`，不重拟合。按原 metadata 顺序恢复 180 张 Source 正常训练图：index%8==0 的 23 张校准，157 张用于原 bank。只用这 23 张到原 Source bank 的距离拟合该类别 CH1 图像 MAD：逐层插值后峰值、原生有效 3×3 均值最大值，分别 median / max(1.4826×MAD,1e-8)；四层等权，两项各 0.5。同类别三个分支共用该状态，MetalParts 与 PCB 不共用状态。

Source 评价参照单独运行 `SOURCE_CH1`，只检索原 Source bank，不加入 Target 支持：MetalParts 120 正常+51 异常=171；PCB 120 正常+75 异常=195。这 366 张不等于 46 张校准图。

## 必需输入与只读边界

默认本地位置和逐类检查结果见 `input_inventory_local.json`。正式名单及图片/GT 字节身份冻结在 `frozen_catalog.json`；不是按目录名或文件名重抽样。

运行端需要：

1. 原 RobustAD 数据及 metadata，含 Source train、Source test0、所有 Target shift 和 GT。
2. 两类原 Few-shot 正式目录中的 `complete.json`、`sample_scores.csv`、`support_manifest.csv`。
3. 原 Source v6 根目录的 `dataset_manifest.csv`、`protocol.json`；每类 `complete.json`、`memory/memory_provenance.json`、`memory/anchor_sidecars.npz`、`calibration/calibration_state.npz`。
4. **同一原 Source v6 树中**两类 `memory/official_banks.npz`，每层 float32 100000×1280。缺失直接报错，没有自动构库或补拟合路径。
5. 仅需补算时才加载原 DINOv3 官方源码、指定 ViT-H+/16 checkpoint 与 SuperADD 宿主。Checkpoint 由原宿主按冻结 SHA 校验。CPU 预测/评价不加载这些文件。
6. 显式 AUPRO 阶段需要可用 CUDA 和原 ADEval；使用已有后端、200 阈值、FPR 上限 0.3，不切换 CPU 近似实现。

本地 `no_banks` 归档只有 Source 元数据及校准参数，不能据此判断 AutoDL 上原 bank 是否还在。历史归档没有 bank 的字节 SHA：首次显式阶段从指定原 Source 树登记当前 bank SHA，之后加载严格核对。该登记**不能证明字节级历史 bank 身份**，不能把新 bank 放进旧路径冒充原库。还会核对冻结来源、23 张 patch 统计及历史 Target CURRENT 分数，但这也不等于历史距离逐元素复现。

父结果和数据始终只读。新输出独立成目录；不改历史报告、Source bank 或原 patch MAD。

## 阶段与缓存优先级

默认命令是 `preflight`：不导入 Torch、不初始化 CUDA、不创建实验输出，不执行采样或模型。检查图片/GT 字节身份时不解码 GT。

预检还通过包元数据列出依赖版本及缺项，不导入 Torch 来探测 CUDA。版本存在不等于运行链路已验证；缺 ADEval 时保留 pending，不安装依赖或改用其他 AUPRO。

| 阶段 | 行为 |
|---|---|
| `recover-calibration` | 逐类恢复原 23 张 Source 距离，验证原 patch 数量/MAD，再冻结独立 CH1 状态 |
| `capture` | 按类别、shift 顺序，取得 Source-only 和 Target 的原始检索/几何；及时释放支持与模型 |
| `predict` | CPU 并行生成三个距离分支和 CH1 分数，完整分组后冻结，不读 GT |
| `evaluate` | 冻结后读 GT；精确全像素 AUROC/AP、图像指标及实例诊断；AUPRO 标记 pending |
| `evaluate-aupro` | 不载模型，复核 CPU 地图/GT 哈希，调用原 CUDA ADEval |
| `summarize` | 逐 shift、类别 macro、两类别等权 macro、Source 参照和 ARD；保留缺项 |
| `package` | 小型 review 包，或指定类别的独立原始缓存包；不删除原件 |

优先级 A：本工具已有且历史 SHA 完整的缓存（无需模型）；B：可信原始 Q/支持特征，原 bank 上重新 Exact 检索（不构建模型）；C：显式补算，调用原完整宿主，Target query 仅一次完整提取，回调获取同次 Source/八支持检索副本。不会从距离恢复 Q，不接受 SPARC 特征。

外部特征清单只在选中范围仍有待捕获 Target 时按需读取。完整缓存续跑、Source-only 捕获不依赖旧 `--feature-root` 是否在线；一旦需要该外部输入，仍严格检查清单历史 SHA 和实际复用文件 SHA，不因目录迁移或离线而自动降级跳过验证。

可选输入：

- `--source-raw-root`：原 `relation_oracle_four_layer_cache_seed42_v2` 输出根目录，含 `categories/robustad/<category>/complete.json` 和 `images/*.npz`。验证 Source v6 身份、记录身份、图片/GT SHA、层和网格；首次复用登记字节哈希，续跑不能重新认可修改文件。指定不存在的类别缓存时可显式补算；存在但不完整/身份不符则拒绝。
- `--feature-root`：每类 `manifest.json`，format=`original_query_support_features_v1`、status=`verified`，必须包含本工具预检显示的 extraction_identity、category，以及 kind=query/support、sample_id、file、sha256 的 records。NPZ 包含 metadata、完整双射 positions、layer_7/15/23/31 原始 float32 [patch,1280]。metadata 核对类别、样本和图片 SHA、extraction_identity、层、dtype、feature_units=`original_1280d_no_normalization`、grid、processed_shape、output_shape。manifest 和文件哈希在首次复用后锁定；没有可信来源不得自行标记 verified。
- `--image-state-root`：复用本工具**相同冻结身份**的逐类 `image_calibration_state.json` 与 `calibration_status.json`，不接受外域或只有数值没有来源的 MAD。

初次登记要求原 bank 在指定树中存在。登记后仅 CPU 处理已有距离可不挂载 bank；GPU 加载仍必须按已登记 SHA 核对。

已发现本地旧归档 `server_results/relation_oracle_seed42_cache_and_audit/mvtec_robustad_seed42_v2`，实际为递归 `images/<domain>__<shift>/*.npz` 布局；包含两类完整 Source 测试缓存及 Target 的 Source-only 距离，不含训练校准的 23 张。预检只报告候选，显式指定 SOURCE_RAW_ROOT 后才进行原 bank-dependent 身份验证/复用。AutoDL 是否保留该旧归档尚未确认。有历史 Target Source raw 时另存严格距离比较审计；跨环境验收仍按冻结 CURRENT 分数兼容规则，同次回调距离核验不改变。

## AutoDL 手动命令

下列服务器路径是模板，不代表已确认文件仍在；先在租到卡的实例执行预检。代码通过 GitHub 独立 clone / pull 同步；旧 DINO-NVS 继续存放资产，不移动或覆盖历史结果。代码上传不代表已连接 AutoDL 或执行实验。

```bash
export PROJECT_ROOT=/root/autodl-tmp/DINO-Reference-AD
export ASSET_ROOT=/root/autodl-tmp/DINO-NVS
export PYTHON=/root/miniconda3/bin/python
export ROBUSTAD_ROOT=/root/autodl-tmp/datasets/RobustAD
export SOURCE_ROOT="$ASSET_ROOT/outputs/relation_reliability/seed42_v6"
export FORMAL_METALPARTS="$ASSET_ROOT/DINOv3/outputs/mad_equal/robustad_target_fewshot_metalparts_seed42_p0_v1"
export FORMAL_PCB="$ASSET_ROOT/DINOv3/outputs/mad_equal/robustad_target_fewshot_pcb_seed42_p0_v1"
export DINOV3_ROOT=/root/autodl-tmp/dinov3
export CHECKPOINT="$ASSET_ROOT/external_baselines/SuperADD/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"
export OUTPUT_ROOT="$ASSET_ROOT/DINOv3/outputs/mad_equal/robustad_segment_pair_$(date +%Y%m%d_%H%M%S)"
export CPU_WORKERS=8
export DEVICE=cuda:0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$PROJECT_ROOT" || exit 1
bash DINOv3/MADEqual/robustad_segment_pair_crossdataset_p0/scripts/run_autodl.sh preflight --no-write
```

确认真实路径、原 bank、CUDA 与依赖后，一次执行余下全流程；建议在 tmux 中执行并保留同一个 OUTPUT_ROOT：

```bash
export AUTO_SHUTDOWN=1
bash DINOv3/MADEqual/robustad_segment_pair_crossdataset_p0/scripts/run_autodl.sh all
```

**这条完整运行命令开启自动关机。仅所有阶段退出码为 0、两类别全部五指标及 ARD 完成、小型 review 包成功后才执行 `shutdown -h now`。任何失败保留现场，不关机。** 不需要关机时设 AUTO_SHUTDOWN=0。脚本没有远程连接、下载、清理或依赖安装行为。

不想一次跑完，可用同一个环境与 OUTPUT_ROOT，依次手动执行：

```bash
entry=DINOv3/MADEqual/robustad_segment_pair_crossdataset_p0/scripts/run_autodl.sh
bash "$entry" recover-calibration
bash "$entry" capture
bash "$entry" predict
bash "$entry" evaluate
bash "$entry" evaluate-aupro
export SUMMARY_DIR="$OUTPUT_ROOT/summary_v1"
bash "$entry" summarize --summary-dir "$SUMMARY_DIR"
bash "$entry" package --summary-dir "$SUMMARY_DIR" --archive "${OUTPUT_ROOT}_review.tar.gz"
```

可先逐类、逐 shift 运行，例如 `bash "$entry" capture --category MetalParts --shift lighting --scope target`，之后用同 OUTPUT_ROOT 扩展。`--scope source` 单独恢复 Source 参照。缺 Source 评价图片/GT 时保留 ARD pending，Target 可以独立运行；身份不符不是“缺失”，仍失败。

续跑重复同一阶段即可：先核对旧逐图 SHA，不重置历史；只补缺失。预测逐图写 progress，保留已验证前缀。未登记孤立文件、临时文件或修改文件均报错，不自动覆盖。CPU 评价中断时保留错误记录，用 `export EVALUATION_TAG=v2` 后从 `evaluate` 开始；不需重复 capture/predict，需重新完成本 tag 的 AUPRO、汇总。AUPRO 中断也可新建 tag 重做 CPU/AUPRO，避免覆盖旧结果。

默认 `--memory-budget-gib 8`，只按完整像素人数验收；不足明确停止，不采样、不缩图或降精度。RAM 判定计入可回收的干净 inactive 文件缓存，排除 shmem/dirty/writeback/mapped 部分。原完整宿主默认要求空闲显存至少 20 GiB，优先 24GB 卡。实际 RobustAD 资源估计见预检，未用 AeBAD 耗时推定本轮时间。CPU 地图处理默认 4 worker，脚本默认 8，BLAS 每进程 1 线程；模型和类别/shift 顺序处理，避免复制 GPU 宿主。全像素排序及单次 CUDA 后端由其原实现执行，不声称所有步骤都能线性并行。

## 输出与解释

每类独立 inputs/protocol/缓存账本、23 张 Source 距离、CH1 状态和逐图校准值；Target raw 保存四层 Source/八支持距离、NN 索引、positions、Top2 端点、alpha、长度、SEGMENT 距离及几何诊断。支持特征、预测与全部身份/顺序保留。

每 category×shift 输出三分支五项指标、逐图分数、排序改善/变差对（同分 0.5）、R_N/R_A/G、全部实例 own/fixed-UNION 峰值 margin、修复和新增损失、小实例分组、门控统计和少量案例。小实例严格按原图 GT 面积稳定选最小 ceil(N/4)，不因映射消失/空环带更改分组；不可计算保留 NA。案例固定取每 shift 按 sample_id 排序的首张正常和首张异常，不据指标挑选。

历史 CURRENT 仅用于宿主审计：严格 atol=1e-5 与兼容 atol=1e-4、rtol=0 同时保留及有符号误差；超兼容界限停止。同次内部距离仍 atol=1e-7/rtol=1e-6，内部分数仍 1e-5。评价另列历史 CURRENT 排序、AUROC/AP 的实际变化，没有历史 raw 时不宣称距离已逐项复现。

汇总 `metric_comparisons.csv` 明确 branch−reference、fraction 单位，分别列 SEGMENT−UNION、PAIR−SEGMENT、PAIR−UNION，改善和下降均保留。先按各类别所有 shift 等权，再对两个类别等权，不直接平均 11 个 shift。ARD 每 shift、每指标先算 min(0,Target−该类别 SOURCE_CH1)，再按同样层级平均。Source 参照、GT 或 AUPRO 缺失时不输出全部完成。

review 包只含完整表格、协议、输入与校准状态、检查、日志、案例和原输入元数据。`cache_index.csv` 给原检索缓存路径和 SHA；预测 manifest 给数组 SHA；`large_cache_inventory.json` 给大文件数量/体积/路径。大缓存保持原位。按类别另存原缓存（tar 不再重复压缩已有 NPZ）：

```bash
bash "$entry" package --summary-dir "$SUMMARY_DIR" --archive "${OUTPUT_ROOT}_MetalParts_raw.tar" --raw-category MetalParts
bash "$entry" package --summary-dir "$SUMMARY_DIR" --archive "${OUTPUT_ROOT}_PCB_raw.tar" --raw-category PCB
```

缓存打包会额外占用与原文件接近的空间，预检 persistent 估计不包含备份副本。下载摘要不代表大缓存已经备份，不自动删除任何原缓存。

## 本地验证

```powershell
$env:CUDA_VISIBLE_DEVICES=''
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'
& 'D:\anaconda\miniconda\envs\chinamoblie\python.exe' -B -m unittest DINOv3.tests.test_robustad_segment_pair_crossdataset_p0 -q
& 'D:\anaconda\miniconda\envs\chinamoblie\python.exe' -B -m DINOv3.MADEqual.robustad_segment_pair_crossdataset_p0.runner preflight --no-write
```

覆盖 CPU 合成校准→预测→全像素评价→汇总→打包、哈希续跑、类别/支持身份、非方形 positions、边界投票、稳定 Top2、macro/ARD 与 Torch 禁导入。AUPRO 集成测试只替代 Torch/CUDA/ADEval 接口，验证地图身份和阶段衔接，不代表真实后端已执行。真实 DINOv3/CUDA、原 bank 加载、23 张补算、历史 CURRENT 复核、全域指标和实际资源峰值仍待 AutoDL 验证。实现可交独立 review；没有预写实验结论。

2026-10-17 验证记录：本模块 23 项专项测试通过（7.910s）；另有 90 项相关既有测试通过（与本模块当时的 22 项合跑，共112项、155.777s）。新增 Python 与测试共9个文件 AST 检查通过；Bash `-n` 通过。本地真实输入只读预检：两类名单/图片/GT字节匹配、Torch 未导入；退出码2来自缺少原 bank、默认模型文件/源码入口及 ADEval，不是正式实验失败。

同日审查修复后：26 项专项测试通过（8.692s），新增覆盖完整缓存＋不可用 feature-root、Source-only＋缺 Target 缓存、待捕获 Target＋外部清单缺失/历史 SHA 不符。未调用 GPU 或真实模型。
