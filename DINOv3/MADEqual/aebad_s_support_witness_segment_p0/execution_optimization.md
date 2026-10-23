# 批量GPU执行与检查清单（2026-10-23）

本版取代旧CPU串行佐证/生产SQLite路径。旧predict曾单核运行7.5小时，cpu-workers未用于该阶段，属于执行缺陷。方法公式、k5、alpha/rho、整图排除、query端点、MAD和评价不变。

| 阶段 | 当前执行 |
|---|---|
| prepare-support | FP64直接cdist，该层八支持特征一次上传并驻留，块内选择与块间合并留在GPU，按query块回传最终k5 |
| support-loo / predict | 域×层独立进程；GPU驻留该层八支持FP64特征，规范化/去重连接，批量计算第三图候选 |
| 显式CPU模式 | --devices cpu --cpu-workers N，启动最多N个独立域×层NumPy/SciPy进程 |
| 预测冻结 | 层作业结束后，CPU按图多线程重建地图、核对分数及提交预测，使用cpu-workers |
| evaluate | 保留精确CPU像素评价；地图和实例处理使用cpu-workers，没有更换排序器 |
| evaluate-aupro | 保留原ADEval，按域×新增分支分配GPU作业 |

显式--devices cuda:0为一条GPU进程通道，cuda:0 cuda:1为两条，动态领取域×层任务；每条通道内部为批量张量运算。不自动启用第二张卡、不将显存相加。CPU工作和磁盘I/O仍存在，不保证所有阶段GPU满载。CPU worker不再限制最多8个，可显式设15。

batch_witness.py为生产路径，method.witness保留为标量参考。默认每批256条唯一连接，检索块256×2048；这些是执行大小，不改变候选或方法参数。13924行时八支持FP64特征约1.06GiB/域/层，另有运行库/临时张量；该说明不是资源准入门槛。

connections_batched/<domain>/L*/batch_*.npz取代逐边JSON/SQLite，按唯一连接及留出命名空间保存端点、第三图候选ID、FP64 alpha/rho、逐候选原因和8位来源掩码。重复候选位置为-1，未定义alpha/rho为NaN，原因编码在config.json。不重复保存特征向量。RAM缓存复用连接；续跑按已提交数组批文件恢复。旧SQLite不转换、不覆盖、不作为新格式使用。

## 所有本轮新增/调整检查及保留规则

没有新增RAM、显存、磁盘容量门槛，没有超时切后端、失败改单卡或隐式GPU探测。

| 检查 | 行为/依据 |
|---|---|
| 执行参数 | 非正块大小、worker或非法设备报错；无容量判断 |
| 原输入身份 | 保留协议、名单、顺序、SHA及校准身份核查；缺失/改变失败，历史容差不变 |
| 新批缓存 | 提交及续跑读取时按批核验原封存SHA；孤立record、配置/身份不符或冲突结果失败并保留；没有每点/每边JSON哈希 |
| 舍入边界 | alpha接近.25/.75或rho接近.5时，以8*dimension*float64_eps*max(1,abs(value))识别数值歧义，并用原method.witness标量复算该连接；仅记录数量，不停止，不放宽接受条件 |
| 非法几何/距离 | 零长度、非有限、零分母继续记录原因；显著SEGMENT>UNION失败，原1e-7+1e-6*abs(U)不变 |
| 正式结果 | 保留评分1e-5、父地图/GT SHA、完整性、均匀缩距FP32 ULP界、CPU中间指标封存/范围检查 |
| 恢复支持 | strict或显式historical-maps-v1沿原政策，历史距离失败不改写为通过 |
| 实际错误 | CUDA、分配、I/O、数学错误原样失败；已提交缓存保留；停止领取新作业，其他通道在途作业可完成 |

原输入文件SHA仍保留，不宣称完全无哈希；没有新增全域反复预扫描。默认preflight仍不导入Torch、不初始化CUDA、不创建实验目录。代码身份/格式变化必须新建运行目录，不在旧正式目录混用。

## 手动执行

本轮只做本地实现/轻量验证，未连接服务器、推送或启动任务。同步新代码后，用新目录只读复用旧恢复源：

```bash
cd /root/autodl-tmp/DINO-Reference-AD
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
RUN=/root/autodl-tmp/aebad_s_witness_gpu_batch_$(date +%Y%m%d_%H%M%S)
/root/miniconda3/bin/python -B -u -m DINOv3.MADEqual.aebad_s_support_witness_segment_p0.autodl \
  --run-root "$RUN" \
  --reuse-recovery /root/autodl-tmp/aebad_s_witness_20261022_193510/recovered_supports \
  --recovery-policy historical-maps-v1 \
  --data /root/autodl-tmp/datasets/AeBAD/AeBAD_S \
  --checkpoint /root/autodl-tmp/DINO-NVS/external_baselines/SuperADD/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth \
  --devices cuda:0 --cpu-workers 8 --witness-batch 256
```

双卡显式改--devices cuda:0 cuda:1；15核CPU可显式--cpu-workers 15。上例不自动关机，需要时明确追加--shutdown-on-success，仍仅全部成功并打包后关机。单阶段support-loo/predict也必须传devices，不能只在prepare-support传GPU。

日志每10张输出device、进度、唯一边数、缓存命中和耗时；子任务输出实时汇入主日志，不必另找日志文件。层状态另记特征传输、缓存恢复、批量几何（含CPU准备、回传和边界复算）、证明写入时间，不把它们全称为纯GPU时间。

## 验证边界

本地测试执行实际NumPy与Torch CPU内核，对照原标量计算的候选、来源、接受结果、alpha/rho及最终距离，另验证NPZ续跑和真实并行冻结。GPU调度使用替身，真实CUDA、双卡计算、926张正式结果和GPU加速比尚未验证。

小型CPU计时：959条唯一连接、1280维，标量1.594秒，批量256为1.375秒，约1.16倍，来源掩码一致。不含磁盘、GPU和正式全流程，不能推算正式GPU加速比。先前更小的159条冷运行批量版本曾较慢，不宣称所有CPU输入均加速。

最终本地回归：93项通过（20.143秒），23份Python源文件内存编译通过；含实际Torch CPU内核、真实NPZ并行冻结、15-worker有界队列及CPU子进程日志转发。只读preflight在显式支持搜索位置缺24份特征，退出2；未导入Torch、未调用GPU、未创建实验目录。该受限搜索不冒充全盘支持特征不存在的证明。记录为local_validation/batch_execution_validation.json及batch_preflight_summary.json。
