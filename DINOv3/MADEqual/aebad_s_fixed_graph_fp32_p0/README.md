# 固定Target图：单遍FP32融合预测

本工具优化正式query图搜索，复用原固定图父实验已封存的Source memory、两级MAD、24张支持特征、12个最终完整图及三域正常LOO选择。原固定图和先前混合精度模块不修改。

## 去掉哪些重复工作

- 不再两次遍历完整边池。每个query/边块只访问一次候选边，在块内求上界并从原始特征做FP32直接投影；最后仅归约小规模块最小值。
- 将投影、区间筛选、候选复核及分支最小值合并在一个Triton GPU程序中。边循环里没有Python逐块启动、`.item()`、`nonzero()`或CPU回传。仍有query块边界的检查、传输和最终归约，不宣称完全无同步。
- 原先每个query块分多次2048节点GEMM，改为对实际使用端点的一次宽FP32 GEMM。缓存节点范数及边长，合并完全相同的分支掩码。
- 只移除冻结掩码未使用的边和端点；实际启用的合法线段全部保留。没有query Top-k限制，没有阈值/预算搜索，也不依据GT删候选。

GPU融合方式参考[Triton官方融合教程](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)，本工具的距离、筛选和投影为独立实现，不采用教程softmax算法。CUDA路径需已有Triton，不自动安装或静默退回旧慢路径。

## 精度与数值定义

节点、query、端点点积、区间标量、投影和归约均为FP32，没有大规模FP64张量计算。此版本固定禁用TF32/autocast以验证完整FP32，未宣称FP16/BF16/Tensor Core；原骨干、离散Source/Target NN和评价继续执行父协议精度。

上述FP32指新增图检索及其alpha/代表佐证距离诊断。**原Top2 SEGMENT对照仍沿用冻结实现，其内部有FP64计算**，没有把这项对照也一起改掉。lambda/beta及原评分、评价的统计精度保留。日志逐图分别列出原离散检索、原SEGMENT、FP32图检索、向量诊断和四层总耗时，便于用实测定位剩余瓶颈。

为减轻大公共偏移引起的抵消，**仅初筛**统一减去一个固定支持端点作为公共原点，误差界计入FP32减法及点积误差。最终距离始终使用原始q/u/v做FP32直接投影；不把平移后的特征写入memory或改变距离单位。不是新的归一化或方法参数。

保留小于Source/完整Target锚点的距离，除以1280一次，原四层融合、插值、CH1/CURRENT不变。FP32的并列按父边编号，UNION精确并列优先。FP32与FP64可能有距离及胜边差异，不声称逐位等价；沿用距离`atol=1e-7,rtol=1e-6`核查并审计胜边。完整图像bench还报告七分支评分差异；未改变的UNION/SEGMENT沿用原评分`atol=1e-5`核查。

原始特征仍是FP32。若输入超出FP32平方数的可表示范围，明确报错，不悄悄切回FP64、填零或改方法。原MAD、评分及评价后端的统计精度保持原样。

## 已有结果如何使用

**复用准备产物，重算全部FP32 query预测。** 不将旧FP64或此前混合精度的逐图结果拼进本轮表格；这些旧结果保留，用作数值参考。图结构和正常LOO选择是父实验冻结资产，不重新选择。

使用新run/cache目录；父资产通过独立来源记录硬/软链接复用，不覆盖父setup、记录或图。stream模式不保存完整Q，原父包也没有完整Q，因此全量FP32预测需要每张图按原宿主路径再前向一次。无须重做Source、108次构图或96项正常LOO。

## 本地验证边界

本地已有Torch2.5.1＋NumPy，可执行实际FP32数学参考和小型文件集成测试；没有Triton，未安装任何依赖，没有调用CUDA或模型。CPU路径验证的是算法数学与接入，不是GPU融合内核的编译、执行和性能。

CUDA内核已写入`kernels.py`并做源码编译检查，**真实Triton编译、4090数值与速度均待服务器smoke/bench**。不能把少量CPU正确性测试当成加速成功，也不承诺倍率或完工时间。

## 手动运行

仅在用户明确启动后执行。当前约定单张物理GPU1、最多8个CPU worker；不影响base、不自动关机、不连接服务器安装依赖。先停止旧预测，保留其目录。部署本模块后：

```bash
work=/home/ubun/YYK
repo="$work/DINO-Reference-AD"
py="$work/conda_envs/yyk/bin/python"
parent="$work/runs/aebad_fixed_graph_gpu_seed42_20261023_222738"
stamp=$(date +%Y%m%d_%H%M%S)
out="$work/runs/aebad_fixed_graph_fp32_$stamp"
cache="$work/storage/fixed_graph_cache/$(basename "$out")"
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg XDG_CACHE_HOME="$work/.cache" TMPDIR="$work/tmp"
export TRITON_CACHE_DIR="$work/.cache/triton"
cd "$repo" || exit 1
module=DINOv3.MADEqual.aebad_s_fixed_graph_fp32_p0.runner
common=(--parent-run "$parent" --output-root "$out" --cache-root "$cache" --device cuda:0 --cpu-workers 8)

"$py" -B -u -m "$module" preflight "${common[@]}"
"$py" -B -u -m "$module" prepare "${common[@]}"
"$py" -B -u -m "$module" smoke "${common[@]}"
"$py" -B -u -m "$module" bench "${common[@]}" --limit 3
```

默认preflight只列文件与Triton安装元信息，不导入Torch/Triton、不初始化CUDA、不做采样/推理或建立输出；没有RAM/显存/磁盘容量门槛。缺依赖明确列出，不自动安装。

smoke：每域一张完整宿主前向，同Q前128行四层对照旧FP64参考，另检查所选边是否属于冻结分支及其原始FP32直接投影。计时含JIT/准备，仅是局部验证。保存`mixed_smoke.json`（沿用共享入口文件名，engine字段明确标FP32），FP64胜边改变审计而不假称一致。距离核对超限保留证据，不调整容差。

bench：运行最多3张**新增完整图像**，保留其预测以供后续正式运行复用。记录预测总时长、含启动的每张平均、RAM/显存、以及可获得父预测的七分支距离与完整CH1/CURRENT差异。没有父预测时明确标缺参考，不补跑旧推理。GT未读取。重点看连续图完成间隔及`four_layer_graph_seconds`，不要将局部smoke计时当整图提速。

确认实际数值与速度后，再手动启动完整流程；不因写了融合代码就直接承诺性能：

```bash
printf -v launch '%q ' env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg \
  XDG_CACHE_HOME="$work/.cache" TMPDIR="$work/tmp" TRITON_CACHE_DIR="$work/.cache/triton" \
  "$py" -B -u -m "$module" all "${common[@]}"
printf -v logfile '%q' "${out}_launcher.log"
tmux new-session -d -s "fixed_graph_fp32_$stamp" -c "$repo" "$launch > $logfile 2>&1"
tail -n 50 -F "${out}_launcher.log"
```

`all`包括prepare/smoke/bench/predict/evaluate/evaluate-aupro/summarize及最终review打包。已完成FP32预测可续跑；旧FP64预测始终不导入。bench已做过时，all中的bench会计算下一批缺失图像，这些同样计入完整预测，没有丢弃重跑。CPU像素评价与21项ADEval AUPRO沿用原实现，全部预测冻结后才打开GT。review继承scratch剪枝与报告类型白名单，不把BIN/NPZ临时大文件打包。

## 检查清单

保留：父历史/代码/软件/资产身份，名单/支持顺序/MAD与选择冻结，输出目录隔离；FP32输入有限性、索引和正长度；smoke原距离容差与直接投影来源；bench未改基线的原评分容差；全预测冻结后才读取GT；原评价及指标封存。没有容量前置阻断、计时超限切换或自动安装。所有旧目录保持只读。
