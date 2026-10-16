# FastRef v3 初始投影机制对照

研究问题：父实验的距离、图像排序和局部定位变化中，初始线性投影和后续 OT 更新分别起什么作用？本工具仅新增 `PROJECTION_TARGET512_SOURCE`，不预设 OT 有效或无效，不将开发域对照包装为新方法。

## 计算与冻结条件

每层原始 `Q: m×1280`，直接读取父实验的 `P: 512×1280` 和 float64 求解常量 B。父实验计算 B 的方式为普通 Cholesky 求解，等价于 `solve(M @ M.T, M)`，本工具不重新求解。

```text
X = float64(Q) / 1280
M = float64(P) / 1280
A_scaled = (X @ M.T) @ saved_B
A_raw = contiguous_float32(A_scaled * 1280)
d_projection = original_ExactNN(Q, all_A_raw) / 1280
d_new = min(original_Source_distance, d_projection)
```

乘法结合顺序与父 `refinement.refine` 的初始投影一致。投影后直接检索，不调用 `refine`、Sinkhorn 或 OT，不通过 lambda=0 模拟。父 `enabled=False` 返回 P，与这里每张查询生成的 m 行 A 不同。完整 Target K8 不加入新分支。

复用原 Source MAD、逐层 patch MAD 后四层等权融合、原插值和地图最大值评分。保持原八张正常支持及其顺序、各层 P 顺序、原 297 张评价图（85 正常/212 异常）和支持排除规则。没有重新采样、Source 构造、MAD 拟合或参数搜索。

在精确算术下，到全部投影的最近距离等于到自身正交投影的残差。正式路径仍调用父 `runtime.host_nn(compact=True)`，即原 Euclidean Exact NN、64 MiB 距离分块和既定 Torch cdist 设置。独立 float64 减法分别计算 float64 A 的自身残差和 float32 A 的自身残差，并记录 Exact 距离减这两项的有符号差异。它们是数值核查，不替换正式路径，不要求逐位相同，也不把合法浮点差异当成方法复现失败。

## 父实验与依赖

唯一批准的父目录：

```text
.../DINOv3/outputs/mad_equal/aebad_s_fastref_target512_source_bounded_density_v3_20261013_211140
identity: 53fb970afcf50b7fd821bf438fe2eb9ddeaafcc5efec82a68314d57d121e0390
```

`inputs.py` 固定了下载包中 inputs、prepare/infer 状态、历史分数/指标/实例和各 cost 文件的 SHA256。进一步复用父 `read_prepared`、`load_constants`、`verify_stage` 检查原型来源、选择索引/顺序、求解诊断、原始输入及所有 297 张父 raw。被修改的父文件不会重新获得信任。旧配置或未完成父实验不能接入；本版要求原冻结路径可用，不隐式搬移或重写父路径。

父四个历史参照是 SOURCE、UNION_K8、UNION_TARGET512、FASTREF_TARGET512_SOURCE。只有新投影分支执行正式五指标评价。历史四行标记 `reused_parent_result`，带父指标文件及其哈希；历史地图重建只用于机制与复核，不调用它们的正式像素评价。

本地已只读检查 `downloads/aebad_s_fastref_target512_source_bounded_density_v3_20261013_211140_results_20261014.tar.gz`，未解压或修改。包内没有完整 Q、A、R。原 LOO B/SEGMENT 距离包也不能恢复 Q；在本地对应输出目录及 query 文件名范围未找到可信原始 Q，不能声称做过全磁盘穷举。因此真实 smoke/infer 需要原宿主重新完整提取每张图一次；同一 Q 同时用于 Source/P 基线复核和新增投影。不会读取 SPARC 校正特征或重跑父 OT。

真实运行还需要父冻结文件清单中的原始依赖：Source memory、Source MAD、原 LOO B capture 与正式分数、支持身份记录、DINOv3 代码/权重、原图片和评价 GT。父目录和上述输入保持只读。GT 仅在预测冻结后解码用于评价；预检/推理可以检查其存在性和文件哈希。现有校验清单还保护原 Source/支持图片等输入，即使本次不重新提取这些特征。

## 阶段和后续手动命令

默认 `preflight`：不导入 Torch、不初始化 CUDA、不采样、不创建输出；列出父结果、四层原型及 MAD、模型、图片等缺失输入。成功只表示文件/身份与估计资源预检通过，不代表 CUDA 或模型可运行。

`smoke`：固定原 12 案例，独立目录冻结 inputs 和父原型身份；原始完整提取一次、复核 Source/P 距离和分数、投影/检索、记录数值和成本。距离采用父 `atol=1e-7, rtol=1e-6`，图像分数采用 `abs<=1e-5`。投影法向方程残差沿用父 `1e-8`。不会用 12 张异常指标调参。

`infer`：要求完整成功 smoke，计算全部 297 张新预测。`evaluate`：要求全部预测、基线检查、数值诊断、哈希和身份完整，才读取 GT 并评价；离线仍逐图复核四个历史分数和当前 Source/P 分数、历史图像汇总与实例响应。缓存缺失/修改、范围不完整、GT 缺失或后端改变均阻止完成。

以下命令仅供之后获准的服务器手动执行，本次没有执行。使用原环境，不安装依赖；四层 CPU 投影并行，BLAS 各一线程以避免过量线程。无需 prepare 或新原型。

```bash
export PROJECT_ROOT=/root/autodl-tmp/DINO-NVS
export PYTHON=/root/miniconda3/bin/python
export PARENT_DIR="$PROJECT_ROOT/DINOv3/outputs/mad_equal/aebad_s_fastref_target512_source_bounded_density_v3_20261013_211140"
export OUTPUT_DIR="$PROJECT_ROOT/DINOv3/outputs/mad_equal/aebad_s_fastref_projection_target512_source_$(date +%Y%m%d_%H%M%S)"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$PROJECT_ROOT" || exit 1

"$PYTHON" -B -u -m DINOv3.MADEqual.aebad_s_fastref_projection_p0.runner preflight \
  --parent-dir "$PARENT_DIR" --output-dir "$OUTPUT_DIR" --cpu-workers 4 --no-write
```

预检通过并确认有 GPU 后，在同一 shell 执行完整流水线；断开连接也继续运行。每一步检查实际状态和退出码，失败停止并保留现场。**这段命令没有自动关机。**

```bash
nohup bash -c '
  pipeline_rc=0
  for stage in smoke infer evaluate; do
    "$PYTHON" -B -u -m DINOv3.MADEqual.aebad_s_fastref_projection_p0.runner "$stage" \
      --parent-dir "$PARENT_DIR" --output-dir "$OUTPUT_DIR" --device cuda:0 --cpu-workers 4
    stage_rc=$?
    printf "STAGE=%s EXIT_CODE=%s\n" "$stage" "$stage_rc"
    if [ "$stage_rc" -ne 0 ]; then pipeline_rc=$stage_rc; break; fi
    "$PYTHON" -B -c '\''import json,os,sys; from pathlib import Path
mode=sys.argv[1]; root=Path(os.environ["OUTPUT_DIR"])
file=root/("evaluation/status.json" if mode=="evaluate" else mode+"_status.json")
expected={"smoke":"smoke_verified","infer":"predictions_frozen","evaluate":"analysis_complete"}[mode]
assert json.loads(file.read_text())["status"]==expected'\'' "$stage"
    check_rc=$?
    if [ "$check_rc" -ne 0 ]; then pipeline_rc=$check_rc; break; fi
  done
  printf "%s\n" "$pipeline_rc" > "${OUTPUT_DIR}.exitcode"
  exit "$pipeline_rc"
' > "${OUTPUT_DIR}.log" 2>&1 < /dev/null &
printf 'PID=%s\nLOG=%s\n' "$!" "${OUTPUT_DIR}.log"

# 新终端变量不存在时，先将 OUTPUT_DIR 设置为上面实际打印的目录。
tail -n 40 -f "${OUTPUT_DIR}.log"
# 完成后：cat "${OUTPUT_DIR}.exitcode" "$OUTPUT_DIR/evaluation/status.json"
```

不提供实际耗时承诺；没有测量新分支的服务器性能。预检采用新增磁盘 2 GiB、可用 RAM 16 GiB 的准入估计，仍须看真实峰值。显存至少需容纳原 ViT-H+ 宿主与 Source bank；未实测新峰值。

## 输出与解释

| 文件 | 含义 |
|---|---|
| `inputs.json`, `smoke_status.json`, `infer_status.json` | 独立身份、冻结父依赖、逐图 raw 哈希及阶段门禁 |
| `smoke_raw/`, `raw/` | Source/P 复核距离、投影 Target/union 距离、全 A NN 索引、两种 float64 自身残差、几何与身份；不保存 Q/A 或完整距离矩阵 |
| `*_verification.csv`, `*_projection_diagnostics.json` | 原容差基线复核、无 OT、法向方程残差及 Exact/自身残差差异 |
| `evaluation/metrics.csv`, `metric_comparisons.csv` | 五行指标及三个 Projection−参照的差值；单位为 0–1 指标差，乘 100 为百分点 |
| `distance_differences.csv`, `distance_difference_summary.csv` | 投影−FastRef 的 Target 单独距离和 Source min 后联合距离，逐图/层/四层及逐层池化；max/mean abs、有符号均值及 0/1/5/50/95/99/100 分位数；单位 Euclidean/1280 |
| `map_differences.csv` | Projection−三个参照的地图差，单位 Source MAD 响应，无负值裁剪 |
| `ranking_summary.csv`, `ranking_per_image.csv` | 正常—异常错误排序对，平局权重 0.5；固定十张困难正常、其余 75 正常、全 85 正常；不是部署阈值误报/漏检 |
| `defect_instances.csv`, `instance_comparisons.csv`, `fixed_reference_instance_peaks.csv` | 原 GT 连通实例、原环带、各分支自身峰值与 margin；固定原 UNION 峰值（同时保留 Target512 参照峰值），Projection−参照的局部变化；消失/空环带状态保留，不填造响应 |
| `small_instance_summary.csv`, `figures/` | 原最小三个离散 GT 实例描述口径和固定原 12 张案例图；不调小区域阈值 |
| `signed_distance_per_image.csv`, `signed_distance_summary.csv` | `R=(d_reference-d_projection_union)/original_Source_MAD_scale`，正常全 patch、异常原 GT patch；逐图等权，再四层等权；R_N、R_A、G=R_N−R_A，有符号不裁剪，以 FastRef/Target512/K8 分别为参照 |
| `cost_smoke.json`, `cost_infer.json`, `cost_evaluate.json`, `cost_comparison.json` | 新提取、基线复核、投影及残差核查、全 A 检索、评价等耗时，RAM/显存/输出大小；父阶段成本单列 |
| `evaluation/status.json` | 只有完整范围、GT 空间证据、五行指标、数值复核和输入保护通过才是 `analysis_complete` |

先比较 Target 单独距离，再比较 Source min 后变化：Source 锚点可能掩盖 Target 的实际差别。Image AUROC 接近不能证明等价，须结合地图、AP、像素指标、排序和局部实例。差异如存在，按层/样本/实例追踪，不预先判断由哪一阶段主导。

父 `full_query_FastRef_CPU` 包含初始投影、OT、更新及诊断，不是纯 OT 时间。新成本还包含重新提取和控制复核；避免重做历史阶段不构成算法加速比。结论仅适用于当前冻结原型、配置和开发域。

## 本地验证范围

```powershell
& 'D:\anaconda\miniconda\envs\chinamoblie\python.exe' -B -m unittest DINOv3.tests.test_mad_equal_aebad_s_fastref_projection_p0 -v
```

专项测试执行实际 NumPy 投影，独立 QR/全投影 NN 参考、幂等与正交、单位/顺序、无 OT、enabled=False 区别、无查询状态积累、父原型身份/顺序、包内记录与求解常量、Target/union 区分、三组差值、默认无 Torch/无写入、缺失/不完整输入拒绝、评分后端变化拒绝。

合成集成测试执行实际新 smoke 编排与完整 297 图离线分析。宿主、查询提取、Source/P 搜索、设备、图绘制使用明确替身；smoke 中投影和全 A NumPy 检索真实执行。评价调用原 mask/metric 函数的 AST，AUPRO 明确替身。它们不能证明真实 Torch cdist、CUDA、DINOv3、GT 或正式 ADEval 已通过。本地没有 Torch，未执行真实特征提取或实验，未安装任何依赖。真实环境需要上述资产，后续必须手动 smoke 通过后再 infer/evaluate。

本轮实际验证：新专项 23 项通过（70.473 秒）；另已运行父数学内核 12 项并通过；8 个 Python 文件的内存 AST 检查通过。只读对照下载包，父模块 11 个归档文件与本地一致。未执行真实服务器命令；本地 Git Bash 不可用，README 命令未执行 Bash `-n` 校验。
