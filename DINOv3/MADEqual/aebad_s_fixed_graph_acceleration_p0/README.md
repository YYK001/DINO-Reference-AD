# 固定Target图预测加速：独立续跑入口

针对已完成固定图构建、正常支持LOO和配置选择的父运行。只优化正式预测中的图检索，不修改方法、支持集、配置选择、Source memory、MAD、骨干前向、原FP32离散NN、距离单位、七分支、评分或评价。旧模块和父结果只读。

## 实际优化

1. 将三个图分支掩码的并集压紧，移除未启用边；再移除这些边未引用的节点。顺序保持父图的原始边顺序，输出胜边仍是父图编号。没有按查询Top-k截断。
2. 大规模端点点积使用原始FP32特征与FP32 GEMM，节点范数及区间标量用FP64。FP32点积误差使用维度相关gamma及Cauchy绝对和上界；溢出时区间扩大，强制直接复核，不能把非有限结果当距离。
3. 第一遍扫描求每个query/分支的全局距离上界，第二遍只对区间仍可能获胜的所有线段做原式FP64直接向量投影。不再在每个边块复核暂时最优解。最终仍按原式恢复 `/1280` 一次并转FP32，UNION平局优先，边平局按父图编号。
4. 复核批量从旧256扩大到2048，限制向量临时数组；不持久保存完整query、方向向量或全距离矩阵。

这是**FP32初筛＋FP64复核**，不是FP16/BF16自动混合精度，也不把骨干改为半精度。当前误差界要求初筛禁用TF32及外层autocast，仅在图检索局部临时设置，退出/异常均恢复原状态；不改变宿主前向。PyTorch的`highest`与FP32内部计算关系见[官方说明](https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html)。没有宣称使用Tensor Core或已经获得某个加速倍数。

## 复用与身份

父目录必须已有：setup/run_config/fresh_catalog/seal、frozen_assets、两级Source MAD、Source bank、24张支持特征、12个完整8图的最终图及三域selection。无需重新构造100轮Source、108个图或96项LOO。

新结果和新缓存必须使用独立目录。父文件不改名、不改哈希、不覆盖状态。大NPZ优先硬链接，同一文件系统不增加一份实体；跨文件系统改用软链接，不把大资产复制到固态。链接依赖原文件持续可用，不要清理父资产。

prepare检查父冻结历史与实际资产、原实现/宿主源码、软件版本和选择记录，然后生成独立加速身份。已完成且校验通过的父预测可以原样复用，新记录注明`reused_original_FP64_prediction`及来源；新计算注明`new_mixed_precision_prediction`。默认只在第一次prepare时导入当时已完成图像；不会把后来产生的父文件自动加入已经冻结的新输入。`--no-import-completed`可显式关闭导入。

旧CPU指标、旧AUPRO不会当作新评价填入。本工具调用原七分支评价/汇总接口，CH1与CURRENT共享同图像素指标。仍要求全部预测冻结才读取GT。

## 阶段及手动命令

默认`preflight`只检查文件是否存在，不导入Torch、不初始化CUDA、不建立结果、不计算大文件SHA。没有内存/显存/磁盘剩余容量门槛。

以下命令仅作手动运行说明；本地实现没有连接服务器、调用GPU或启动真实实验。切换前先由用户停止父预测，保留父目录；不要在同一张GPU上同时启动两轮。当前约定单张物理GPU1、最多8个CPU工作线程，后续换卡由用户决定。

```bash
work=/home/ubun/YYK
repo="$work/DINO-Reference-AD"
py="$work/conda_envs/yyk/bin/python"

# 改为当前固定图父运行目录；必须是已经完成select的那一轮。
parent="$work/runs/<当前固定图父运行目录>"
stamp=$(date +%Y%m%d_%H%M%S)
out="$work/runs/aebad_fixed_graph_mixed_$stamp"
cache="$work/storage/fixed_graph_cache/$(basename "$out")"
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg XDG_CACHE_HOME="$work/.cache" TMPDIR="$work/tmp"
cd "$repo" || exit 1
module=DINOv3.MADEqual.aebad_s_fixed_graph_acceleration_p0.runner
common=(--parent-run "$parent" --output-root "$out" --cache-root "$cache" --device cuda:0 --cpu-workers 8)

"$py" -B -u -m "$module" preflight "${common[@]}"
"$py" -B -u -m "$module" prepare "${common[@]}"
"$py" -B -u -m "$module" smoke "${common[@]}"
```

smoke对每个域第一张图实际前向一次，同一份Q的前128行、四层分别比较旧FP64路径和新路径：沿用距离`atol=1e-7, rtol=1e-6`，另检查胜边编号。没有GT，没有标签选参数。保存各项误差、原/新耗时、压紧前后规模和直接复核数量。该时长包含各检索器准备，只是局部对照，不能当作全图预测加速比。失败保留记录，不放宽容差；已通过smoke可续跑复用。

通过后可先做少量新增完整图像预测，看**实际全图速度**，不要再用理论FLOPS承诺时间：

```bash
"$py" -B -u -m "$module" predict "${common[@]}" --limit 3
# limit只控制本次新增图像数量；旧已完成图像照常复用。
```

正式续跑（含缺失预测、CPU评价、AUPRO、汇总、打包）：

```bash
printf -v cmd '%q ' "$py" -B -u -m "$module" all "${common[@]}"
printf -v logfile '%q' "${out}_launcher.log"
tmux new-session -d -s "fixed_graph_mixed_$stamp" -c "$repo" "$cmd > $logfile 2>&1"
tail -n 50 -F "${out}_launcher.log"
```

同一参数及目录再次运行即可续跑。`all`依次prepare/smoke/predict/evaluate/evaluate-aupro/summarize，最后打包；已完成记录按身份复用。也可分别调用这些阶段或`pack`。默认21项AUPRO，原ADEval1.1.0、200阈值/FPR≤0.3。没有自动关机。

## 输出、资源与检查清单

- `acceleration_inputs.json/prepare_plan.json/setup.json`：独立身份、父历史记录和本地冻结输入。
- `mixed_smoke.json`：12项局部检索对照及原/新耗时。不是926张正式实验已通过。
- 新缓存：小的独立来源JSON、原资产链接、新增图像的七分支距离及诊断；stream不保存完整Q。
- `acceleration_costs/`：本轮各阶段耗时与进程RAM/CUDA峰值；复用的父阶段成本不计成算法加速。
- 原七分支metrics、实例、机制及汇总报告；`*_review.tar.gz`只打包新输出的说明/评价/记录，大NPZ仍留在独立缓存目录。

保留或新增的检查全部列在这里：①父历史/输入哈希与版本身份；②选择、MAD和样本记录冻结；③新旧输出不重叠；④FP32输入有限性、索引、正长度及区间误差；⑤同Q的smoke距离容差及胜边；⑥全域预测冻结后才读取GT；⑦沿用原评价完整性/地图SHA/指标封存。**没有资源容量前置阻断**。检查不在每个query重新读取大权重。

内存：当前域固定资产＋当前图像＋有界query/edge/refine块，不随926图累积完整Q。额外FP64直接向量临时数组按2048对处理。CPU评价仍沿用原逐域七地图临时文件及单分支精确排序，AUPRO逐域逐分支。真实RAM/显存峰值和4090端到端速度需手动smoke与少量完整预测实测；本地CPU结果不能替代它们。
