# AeBAD-S/background 冻结 replay12

本轮交付代码、服务器启动脚本和本地轻量验证。**补推理尚未执行，也没有连接服务器。**
本工具不调用原全量 `runner.run()`，不运行其他域、RobustAD、seed 或新增方法。
只恢复固定 10 正常 + 2 异常；之前三个 1 原像素 smallest-instance 案例不纳入。

## 文件与复用入口

- `replay12.py`：独立 `preflight` / `replay` 入口，固定案例、manifest 顺序、独立输出、缓存续跑和逐图复核。
- `scripts/run_aebad_s_background_replay12.sh`：用户手动执行的启动脚本，默认 preflight；不连接、上传、下载或自动启动后续诊断。
- `../aebad_s_union_diagnostics/runner.py`：识别并校验 replay12 成功记录，拒绝未复现/被修改的缓存，区分 285 张范围外缓存。
- `../../tests/test_mad_equal_aebad_s_replay12.py`：无 PyTorch 的案例/支持/完整缓存 I/O/失败策略与诊断接入测试。

Replay 复用原 `_load_memory(..., source_count=521)`、`_load_mad`、`_support_tensors`、
`observe_fewshot_pil`、`extract_or_load_support`、`_save_observation`、`_load_observation`、`_method_map`、
`construct_exact_host` 和 `preprocessed_spatial_shape`。
按原 query preprocessing 和 tile 拼接，每个缺缓存 query 在本次尝试中只调用一次 `observe_fewshot_pil`。
Source-extra K8、Target K8 都从原 manifest 读取；K1/K2/K4/K8 为原嵌套前缀。
完整缓存包含 17 个距离数组和 4 个协议/路径/输出几何字段，绝不以 K8 复制或占位填充其他 K。
层 7/15/23/31、1280 维、Exact Euclidean / 1280、Source MAD、原插值图 max 均保持原实现。
不启用 autocast，不更改 TF32/矩阵乘精度设置；实际设置和软件版本另行记录。
GT 在 replay 中不读取，仅在后续诊断使用。

## 服务器所需文件

所有路径都需用户执行 preflight 后核实，以下历史路径不证明文件仍在。

| 必需项 | 默认/要求 |
|---|---|
| 工程源码 | `/root/autodl-tmp/DINO-NVS`，含原 few-shot/host/共享代码和本次新文件 |
| 数据 | `/root/autodl-tmp/datasets/AeBAD/AeBAD_S`；必须有 12 张固定 query 和 16 张原支持图；完整 test 文件索引可额外交叉检查，推理仅读取这 28 张图 |
| DINOv3 官方代码 | `/root/autodl-tmp/dinov3`，含 `hubconf.py` |
| 官方 SuperADD | 工程下 `external_baselines/SuperADD`，原 `config.json`、`tracks/industrial/src`、`utils` |
| Checkpoint | `external_baselines/SuperADD/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth`，SHA256 必须为原 approved 值 |
| 原 Source memory | 原 `official_memory.pt`；protocol 一致、`source_count=521`、四层各 `[100000,1280]`，绝非新的近似 bank |
| 原 MAD | 原 `source_mad_state.npz`，有限的四层 medians/正 scales；不得重新拟合 |
| 正式结果 | `sample_scores.csv`、`support_manifest.csv`、`memory_manifest.json`、`complete.json`，与原 MAD 一致 |
| 后续诊断附加文件 | 原 `domain_metrics.csv`；保留 `mmr_compatible_sample_scores.csv` 和 `mechanism_by_domain_layer.csv` 可完成既有交叉检查 |
| CUDA 环境 | 原 Python/PyTorch/torchvision 等工程依赖，实际可用 CUDA；不安装、不下载、不自动降级 |

可补齐项：原 `support_features/source_extra/01..08.npz` 和
`support_features/target_matched/background/01..08.npz`。
有旧缓存则先验证元数据/特征 header，再复制到独立补跑目录并用原读取器验证有限值；
缺失时只为 manifest 中这 16 张原支持图调用原确定性冻结特征提取函数。
原图必须存在；不会重新选择支持图或写回原 support_features。

## 用户手动执行的服务器命令

按实际服务器路径调整环境变量。此处仅提供命令，本窗口没有执行它们。

```bash
export PROJECT_ROOT=/root/autodl-tmp/DINO-NVS
export PYTHON=/root/miniconda3/bin/python
export AEBAD_ROOT=/root/autodl-tmp/datasets/AeBAD/AeBAD_S
export RESULTS_DIR="$PROJECT_ROOT/DINOv3/outputs/mad_equal/aebad_s_target_fewshot_seed42_v1"
export OFFICIAL_ROOT="$PROJECT_ROOT/external_baselines/SuperADD"
export DINOV3_ROOT=/root/autodl-tmp/dinov3
export CHECKPOINT="$OFFICIAL_ROOT/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"
export SOURCE_MEMORY="$RESULTS_DIR/official_memory.pt"
export SOURCE_MAD="$RESULTS_DIR/source_mad_state.npz"
export SUPPORT_CACHE_DIR="$RESULTS_DIR/support_features"
export OUTPUT_DIR="$PROJECT_ROOT/DINOv3/outputs/mad_equal/aebad_s_union_background_replay12_seed42_v1"
export DEVICE=cuda:0
cd "$PROJECT_ROOT"
```

1. 先执行不加载 backbone 的检查：

```bash
bash DINOv3/MADEqual/aebad_s_target_fewshot/scripts/run_aebad_s_background_replay12.sh preflight
```

2. 用户核对 preflight 并决定执行后，明确调用 replay：

```bash
bash DINOv3/MADEqual/aebad_s_target_fewshot/scripts/run_aebad_s_background_replay12.sh replay
```

3. **仅 replay 退出码 0 且 `replay_status.json` 为 verified 后**，运行独立诊断：

```bash
"$PYTHON" -u -m DINOv3.MADEqual.aebad_s_union_diagnostics.runner \
  --results-dir "$RESULTS_DIR" \
  --cache-dir "$OUTPUT_DIR" \
  --data "$AEBAD_ROOT" \
  --output-dir "$PROJECT_ROOT/DINOv3/outputs/mad_equal/aebad_s_union_background_replay12_diagnostic_seed42_v1" \
  --skip-archive-inventory
```

诊断输出必须是新目录；已存在时由用户明确换新名称。脚本不会自动执行这一步。
CLI 参数可覆盖上述环境变量，例如 `--memory`、`--mad`、`--checkpoint`、`--support-cache-dir`。
`--cases-csv` 可指定旧 `selected_cases.csv`，只取 top10_union_normal / focus_anomaly，仍必须精确匹配固定 12 张。
Checkpoint 可存于自定义路径：仅模型构造期间将官方权重路径函数绑定到已校验的 CLI 路径；仍调用原 SHA 检查与本地 DINO loader，不改模型/预处理。
OMP/MKL/OpenBLAS 在启动脚本中设为合法整数 8，不沿用历史非法值。

## preflight、复核和续跑

Preflight 不导入 PyTorch，不加载 backbone；检查实际文件路径、文件 SHA256、原协议/计数/config、
12 张案例的原 CSV 身份、16 张支持图与原 K 顺序、图像头及内容 checksum；完整 1639 张 test 索引存在时额外交叉检查。
如果仅保留所需原图，sample_id 仍直接沿用原 CSV，不对部分文件重新编号；Source memory 的加载计数始终是原 521，不由当前 Source 文件数量改变。
大 checkpoint/memory 的文件 hash 会产生磁盘读取，但不反序列化模型。
Memory tensor 内容/shape 校验延后由 replay 中原 `_load_memory` 执行；
支持 cache 有限值延后由原 `extract_or_load_support` 执行，不把文件存在宣称为内容已通过。
本地 preflight 也接受 `AeBAD.zip`，按需读取 28 张图片的 header/字节，不解压、不读取 GT。
Replay 要求实际已解压数据目录。

独立目录含 `preflight.json`、`replay_inputs.json`、分次 `environment_*.json`、
`support_provenance.csv`、`replay_cases.csv`、`replay_verification.csv`、`replay_status.json`，以及完整缓存和支持特征。
关键输入/图片 SHA256 和原实现代码 checksum 都记录；输入变化时拒绝混用旧 replay 目录。
旧/新支持来源及 checksum 保留。续跑在重写 replay 状态或加载模型前读取历史逐图记录，
对已记录的 query cache 先比对原 `cache_sha256`；文件被修改或缺失时保留缓存、旧状态和验证记录，非零退出。
历史逐图哈希在续跑状态中保留，即使依赖/模型加载失败，下一次续跑仍受原哈希约束。
通过历史哈希检查后，已有 query cache 再核验完整数组、协议/路径、几何、各 K 的逐层 min、
support-prefix 单调性，再通过原两张 MAD 评分图与 CSV 的分数复核，合格才跳过，绝不覆盖坏缓存。

绝对误差容忍值固定为 `1e-5`，不提供扩大容忍值以通过的入口。
逐图记录实际分数、误差、耗时、复用/补推理动作及失败原因。
分数不齐时保留 raw cache 和误差，将案例标为 not_reproduced，继续处理其余固定案例后非零退出；
不得作为原实验的精确空间解释。先核对 checkpoint/memory/支持顺序/预处理/精度/版本，不按测试标签调参。
被修改、不完整或未通过分数复核的 replay12，后续诊断会拒绝。
合法续跑重用原命令；损坏缓存不自动删除/覆盖，由用户保留现场并决定隔离处理或新建目录。
不要同时启动两个 replay 写同一目录。

| 退出码 | 含义 |
|---|---|
| 0 | preflight 文件检查通过（不等于 CUDA 已验证），或 replay 的固定 12 张均通过复核 |
| 2 | 必需项缺失/协议等 preflight 检查未通过，未加载模型 |
| 3 | replay 已保留结果，但部分案例未复现；不准接入精确诊断 |
| 其他非零 | 模型/依赖/支持等运行失败，保留 status/error 和已有产物 |

Source memory 缺失：停止，报告需要恢复原 bank；重新构建将是另一个需用户决定的工作，工具不会做。
MAD 缺失：停止，恢复原 MAD；不会从新的正常图/Target/测试标签重新拟合。
这些文件的备用路径只接受用户提供的原产物，不能用新 bank 冒充原实验。

## 诊断的范围

排序仍读取原 297 张 CSV；热图、峰值和实例响应只覆盖已复核的 12 张。
其余 285 张 raw-cache 缺失标记为 outside_replay12_scope，不是补跑失败。
12 张地图重建及范围内两个异常的 GT、实例响应/margin 检查、GT resize/crop 和四层缺陷 patch 距离下降均完成后，才标记 planned_subset_complete。
缺失范围内 GT 或相关步骤时保持 partial_evidence；verification.json 和 diagnostic_status.json 列出缺失 GT、步骤未完成的案例 ID。
空环带或输出网格上消失的实例沿用原规则，记录明确的不可定义状态，不虚构 margin。子集 reduction/gap 不叫全域 R_N/R_A/G。
全域 GT 面积清单是事后标注统计，不代表全域距离/实例响应已恢复；三个 1 像素定向案例不展示。
本轮不实现最近邻来源追踪，不自动扩展补推理范围或开展新方法。

## 本地检查

```powershell
python -m unittest DINOv3.tests.test_mad_equal_aebad_s_replay12 DINOv3.tests.test_mad_equal_aebad_s_union_diagnostics -v
python -m py_compile DINOv3/MADEqual/aebad_s_target_fewshot/replay12.py DINOv3/MADEqual/aebad_s_union_diagnostics/runner.py
```

完整缓存兼容性测试在隔离的 namespace 中执行原源码内 NumPy-only 保存/读取函数，
不导入原 runner 的 PyTorch 依赖；它验证真实格式，不代表 DINO/CUDA 推理经过验证。
本地不执行 replay；真实 memory tensor 加载、支持特征提取、12 图 GPU 距离及分数对齐待用户服务器执行后验证。

本轮实际检查：28 项 unittest 通过；上述 Python 文件编译检查通过；新脚本 `bash -n` 通过。
强制禁止导入 PyTorch 的本地 preflight 已核对 12 案例、16 支持和 28 图片，原正式文件 SHA256 未变。
报告保存在 `DINOv3/outputs/mad_equal/aebad_s_background_replay12_local_preflight_v1/preflight.json`；
因指定本地路径没有原 Source memory、checkpoint、官方 DINOv3 hub 代码，状态为 blocked。
这是本地文件检查结果，不是对服务器文件存在性、GPU 或补推理完成的确认。
