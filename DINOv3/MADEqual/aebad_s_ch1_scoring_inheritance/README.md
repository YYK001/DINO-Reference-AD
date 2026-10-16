# AeBAD-S background：Source 校准恢复与第一章评分继承

本工具只衔接已有方法，不作为第二章新方法。比较同一 Source、完整 K8 memory 距离上的旧图像评分与第一章冻结评分。不重新采样支持、不改变 memory、patch MAD、检索或定位定义，不运行 FastRef/OT/投影。

## 实现依据与冻结定义

|内容|实际复用/核对入口|
|---|---|
|Source 顺序|`aebad_s_full_p0.dataset.discover_phase_a/_images`：相对路径字典序，排除 `._*`、`__MACOSX`；`relation_reliability.datasets.split_source_normal`：零基 index % 8 == 0|
|原校准链|`aebad_s_target_fewshot.runner._calibration_raw/_fit_source_mad`；新工具直接调用其使用的 `superadd_host.observe_pil`，不调用写入正式 MAD 的旧流程|
|宿主加载|`aebad_s_sparc_p0.runtime.load_host`：仅复用加载、冻结和精度校验，不调用任何 SPARC 投影；原 `_load_memory` 只加载，缺失即失败|
|原距离单位|`superadd_host._resolve_pending_observations` 已将 Exact 欧氏距离除以1280，并恢复物理位置；新工具不再次除法|
|第一章评分|`image_granularity_calibration.scoring.interpolated_layer_maxima/fit_image_mad`，`connected_region.scoring.local_mean/calibrated_branch/fused_score`，`connected_region.runner` 的 FULL_LOCAL_MEAN|
|CPU 地图重建|直接复用 `aebad_s_projection_layer_diagnostic.numerics.reconstruct`；float32、bilinear、align_corners=False，按 positions 还原|
|研究边界|2026-09-11 第一章整体验证报告；AeBAD-S few-shot 客观报告；2026-10-13 窗口报告第16节。历史报告不修改|

Source ZIP 中521张训练正常图按原排序冻结，455 memory-fit / 66 calibration；恢复输出保存完整521行有序清单。清单指纹 `610d0a3d328878dacdba44e0d53a888d5eb3955712fad6082a1469222b0fbe9f` 包含每张图的字节哈希、尺寸、角色和原索引。服务器必须与本地 ZIP 的这份清单一致。历史 memory_manifest 只保存计数和构库配置，没有521张逐图历史哈希；这里是核对原发现/划分逻辑、原 memory 字节身份及当前数据身份，不声称原归档曾保存该逐图清单。

随模块同步 `source_scope.json`：它是冻结的521个原始文件名，用于逐项报告缺失/多余图片，不是恢复的距离缓存。其内容另有固定指纹，实际图片身份仍由上面的完整清单指纹约束。

原 memory_manifest 记录构库 brightness=[0.8,1.2]，查询协议 brightness=1；工具严格继承两者的历史记录，不将本次 Source memory 重新解释或重建成 NoAug。查询保持 resize=.625、640窗口/128 overlap、patch16、L7/15/23/31、batch1、evaluation_downscale=4，原始图尺寸分别计算 `output_shape=(height//4,width//4)`。

每层 `z=(d-patch_median)/patch_scale`，保留负数。峰值 `m=max(U(z))`；局部量 `q=max(valid 3×3 mean(z))` 来自原生网格、stride1、无padding窗口。CPU local 完整保留原 SciPy uniform_filter 的 float32 `*9 /9` 运算顺序。

仅66张 Source 校准图到原 Source memory 的 m、q 分别拟合四层 image MAD：`median=median(v)`，`scale=max(1.4826*median(abs(v-median)),1e-8)`。Source 与 Union 共用同一份状态；原 patch MAD 文件只读。

`S_CH1 = .5*mean((m-mu_peak)/sigma_peak) + .5*mean((q-mu_local)/sigma_local)`。没有层/窗口/权重搜索，无 Target 自匹配或正常测试图拟合。

|分支|距离|图像分数|
|---|---|---|
|SOURCE_CURRENT|source_distance|原等权融合定位图最大值|
|SOURCE_CH1|同上|第一章冻结评分|
|UNION_K8_CURRENT|min(source_distance,target_full_distance)|原等权融合定位图最大值|
|UNION_K8_CH1|同上|第一章冻结评分|

只读取父包 `raw/test_*.npz` 的297份正式缓存，不读取 `smoke_raw`、512原型、细化/投影距离。沿用 capture.samples 的85正常/212异常、八张支持及其顺序/排除规则。父包、原正式CSV/MAD、模型/memory和代码均校验身份，记录的服务器绝对路径不会在本地自动访问。

## 阶段与失败行为

- 默认 `preflight`：只读校验输入和范围；不创建结果、不导入 Torch、不初始化模型/CUDA、不采样或评分。CPU 模式无需 checkpoint、memory 文件。`--for-recovery` 切换为服务器资产预检，仍不导入 Torch。
- `recover-calibration`：唯一允许导入模型依赖的显式阶段，仅66张 Source-normal 各调用一次完整 `observe_pil`。保存已按物理网格排列的 float32 `[4,H,W]` raw，不保存 query-memory 大矩阵。没有 memory/MAD 自动补建路径。
- `evaluate`：CPU 拟合 Source image MAD，再重建297张四分支图像分数。先验证两项 CURRENT 的身份、逐图分数、历史 AUROC/AP及全部18020个正常—异常对的状态，再计算 CH1 正式图像指标。若 CURRENT 失败，只保留已计算分数/排序差异及 CURRENT 指标，不发布 CH1 指标差值。

续跑保留 `recovery_status.json` 的逐图旧哈希，先比哈希再检查内容。修改、缺失、额外孤立缓存、输入/代码变化均拒绝复用，不覆盖先前记录。中断后继续使用同一个 `RECOVERY_DIR`；文件写入与状态提交间若异常中断形成孤立文件，会明确拒绝，需人工核对，不能自动把它重新认定可信。完整恢复后再次调用恢复入口只复核产物，不重复提取。

恢复汇总 patch median/scale 只是与原 patch MAD、calibration_stats 的一致性检查，不替换原参数。原66张逐图距离已删除，**不能据此宣称与旧逐图数组逐元素复现**。容差预先固定：patch统计沿用距离口径 atol=1e-7、rtol=1e-6；图像分数和CPU/原函数统计 atol=1e-5、rtol=0；计数、身份和排序状态严格相同，不因运行失败放宽。尺度一致也不自动代表排序一致。

服务器同时用原第一章函数导出66张图每层 m/q，CPU在恢复结束和正式评价前分别核对。任何数值失败为 `partial_evidence`，不作为验证完成。CPU不计算 Pixel AUROC/AP/AUPRO、不解码GT、不新增案例图；每个 memory 分支内两种图像评分共享定位定义。

## 本地轻量测试与默认预检（PowerShell）

```powershell
Set-Location 'G:\codex\workspace\experiment'
$py = 'D:\anaconda\miniconda\envs\chinamoblie\python.exe'
$env:PYTHONUTF8 = '1'
& $py -B -m unittest DINOv3.tests.test_mad_equal_aebad_s_ch1_scoring_inheritance -v
& $py -B -m DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.runner preflight --no-write
```

当前本地输入：父正式结果包、AeBAD.zip、正式 Source patch MAD及CSV均到位。原66份缓存已经删除；尚未恢复，默认预检应返回 blocked，逐项列出001—066，**不是完整实验可运行的声明**。不搜索旧缓存、不自动补推理。

测试使用实际生产 CPU 数学/ledger/评价路径。原 NumPy/SciPy 函数通过 AST 仅加载函数体，避开 Torch 顶层导入；插值另外用独立标量 stencil 核对。合成297张测试只替换资产接纳/最终重读上下文，使用微型地图，不是实际297张评价或真实模型验证。真实 CPU/Torch 插值统计一致性仍须由恢复输出验证。

## 用户手动在服务器执行

先同步本模块及本模块复用的既有代码，保持历史代码不改动；以下命令**由用户执行**，本轮未连接服务器。预检会核对路径、文件哈希，CUDA实际可用性留待显式恢复阶段。

```bash
export PROJECT_ROOT=/root/autodl-tmp/DINO-NVS
export PYTHON=/root/miniconda3/bin/python
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
cd "$PROJECT_ROOT" || exit 1

export RECOVERY_DIR="$PROJECT_ROOT/DINOv3/outputs/mad_equal/aebad_s_ch1_source_calibration_$(date +%Y%m%d_%H%M%S)"
recovery_args=(
  --project-root "$PROJECT_ROOT"
  --results-dir "$PROJECT_ROOT/DINOv3/outputs/mad_equal/aebad_s_target_fewshot_seed42_v1"
  --parent-dir "$PROJECT_ROOT/DINOv3/outputs/mad_equal/aebad_s_fastref_target512_source_bounded_density_v3_20261013_211140"
  --data /root/autodl-tmp/datasets/AeBAD/AeBAD_S
  --memory "$PROJECT_ROOT/DINOv3/outputs/mad_equal/aebad_s_target_fewshot_seed42_v1/official_memory.pt"
  --checkpoint "$PROJECT_ROOT/external_baselines/SuperADD/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"
  --official-root "$PROJECT_ROOT/external_baselines/SuperADD"
  --dinov3-root /root/autodl-tmp/dinov3
  --output-dir "$RECOVERY_DIR"
)

"$PYTHON" -B -u -m DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.runner \
  preflight --for-recovery --no-write "${recovery_args[@]}"
preflight_rc=$?
printf 'PREFLIGHT_EXIT_CODE=%s\nRECOVERY_DIR=%s\n' "$preflight_rc" "$RECOVERY_DIR"
```

只有预检通过才在同一终端显式恢复。若另开终端，先重新设置以上变量；续跑将 RECOVERY_DIR 设为原目录，不能生成新的时间戳冒充续跑。

```bash
if [ "${preflight_rc:-1}" -eq 0 ]; then
  recovery_started=$(date +%s)
  "$PYTHON" -B -u -m DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.runner \
    recover-calibration "${recovery_args[@]}" --device cuda:0 \
    2>&1 | tee -a "${RECOVERY_DIR}.log"
  recovery_rc=${PIPESTATUS[0]}
  printf '%s\n' "$recovery_rc" > "${RECOVERY_DIR}.exitcode"
  printf 'RECOVERY_EXIT_CODE=%s\nELAPSED_SECONDS=%s\nOUTPUT=%s\n' \
    "$recovery_rc" "$(( $(date +%s) - recovery_started ))" "$RECOVERY_DIR"
else
  printf '预检未通过，未启动恢复。\n'
fi
```

命令未设置自动关机。恢复有完成状态和逐图日志；失败保留现场。无多图并行模型提取，保持原batch1精度；统计和检索沿用线程配置。后续 CPU 评价默认 `--cpu-workers 4`，独立图像并行重建、按冻结清单顺序收集结果，不改变单图float32运算顺序。可显式设为1—32；没有自动搜线程数或改算法。

## 新恢复结果打包与下载

仅打包新恢复结果，不带模型、memory、原297张测试缓存。先在服务器确认退出码和状态，然后执行：

```bash
if [ "$(cat "${RECOVERY_DIR}.exitcode" 2>/dev/null)" = 0 ]; then
  "$PYTHON" -B - "$RECOVERY_DIR" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
s = json.loads((p / 'recovery_status.json').read_text())
assert s['status'] == 'calibration_recovered' and s['images'] == 66 and s['failed_checks'] == 0
assert len(list((p / 'raw').glob('*.npz'))) == 66
print('RECOVERY_VERIFIED:', p)
PY
  verify_rc=$?
  if [ "$verify_rc" -eq 0 ]; then
    PACKAGE="${RECOVERY_DIR}_results_$(date +%Y%m%d_%H%M%S).tar.gz"
    tar -czf "$PACKAGE" -C "$(dirname "$RECOVERY_DIR")" \
      "$(basename "$RECOVERY_DIR")" "$(basename "$RECOVERY_DIR").log" "$(basename "$RECOVERY_DIR").exitcode"
    package_rc=$?
    if [ "$package_rc" -eq 0 ]; then
      sha256sum "$PACKAGE"
      du -h "$PACKAGE"
      printf 'DOWNLOAD_PACKAGE=%s\n' "$PACKAGE"
    fi
  fi
else
  printf '恢复没有成功退出，不打包为完成结果。\n'
fi
```

包含：`raw/001.npz`—`066.npz`、`inputs.json`（含完整521张身份）、`source_ordered_manifest.csv`、`calibration_reference.csv`、`patch_mad_validation.csv`、`numerical_verification.csv`、`recovery_status.json`及运行日志/退出码。没有重新拟合的 patch MAD。

本地 PowerShell 下载（连接地址和端口由用户确认，不假定历史实例仍有效）：

```powershell
$serverAddress = Read-Host '当前AutoDL SSH地址（不含root@）'
$serverPort = Read-Host '当前SSH端口'
$remotePackage = Read-Host '服务器DOWNLOAD_PACKAGE完整路径'
scp -P $serverPort "root@${serverAddress}:$remotePackage" 'G:\codex\workspace\experiment\downloads\'
```

## 下载后的本地 CPU 评价

解包只读取下载包内容并写入 G 盘 downloads；原正式资产和父包不动。将下例文件名、恢复目录名替换成实际值。

```powershell
Set-Location 'G:\codex\workspace\experiment'
$py = 'D:\anaconda\miniconda\envs\chinamoblie\python.exe'
$env:PYTHONUTF8 = '1'
tar -xzf 'G:\codex\workspace\experiment\downloads\实际恢复结果包.tar.gz' -C 'G:\codex\workspace\experiment\downloads'
$cal = 'G:\codex\workspace\experiment\downloads\实际恢复目录名'
$out = 'G:\codex\workspace\experiment\DINOv3\outputs\mad_equal\aebad_s_ch1_scoring_' + (Get-Date -Format yyyyMMdd_HHmmss)
& $py -B -m DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.runner preflight --source-calibration-dir $cal --output-dir $out --no-write
if ($LASTEXITCODE -eq 0) {
    & $py -B -u -m DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.runner evaluate --source-calibration-dir $cal --output-dir $out
}
```

CPU 输出：`calibration_per_image.csv`、`image_calibration_state.json`、`per_image_scores.csv`、`image_metrics.csv`、`ranking_summary.csv`、`ranking_per_image.csv`、`metric_comparisons.csv`、`numerical_verification.csv`、`objective_report.md`、`inputs.json`、`timing.json`、`status.json`。输出目录必须新建；不覆盖旧实验。记录服务器原绝对路径以便追溯，实际只读取用户指定本地目录，不要求本地存在服务器路径。

三组差值依次为 SOURCE_CH1−SOURCE_CURRENT、UNION_K8_CH1−UNION_K8_CURRENT、UNION_K8_CH1−SOURCE_CH1。指标保存 fraction及百分点，排序保存有符号变化，同分按0.5计。全部85/固定10/其余75张正常图分别对相同212异常图；不能把它们解释为部署阈值下误报、漏检。后续参照适配应在冻结的相同 CH1 评分和同信息完整 K8 基线上比较，不把跨评分差值全归于 Target memory。

background 已用于多轮开发；此对照不属于独立最终验证。收益与下降均由实际评价后报告，本轮没有预填结果。

## 本轮本地验证记录

2026-10-15：使用现有 Python3.11.12、NumPy1.26.3、SciPy/Pillow/scikit-learn，未安装依赖。15项专项测试通过（最终一轮5.504秒）；7个Python文件内存编译通过；README的3段PowerShell命令通过语法解析。真实输入只读预检核对了521个Source身份、原66张划分、297份正式raw及85/212和8支持范围，7.623秒；唯一阻断是尚未恢复的66份校准缓存。没有加载Torch/模型、连接服务器、补算校准或执行真实297张评分。Bash服务器命令尚未执行。
