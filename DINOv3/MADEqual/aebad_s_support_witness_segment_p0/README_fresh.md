# WITNESS：从AeBAD原图完整重跑

这是独立的`fresh`入口，不要求历史输入tar、Source bank、MAD、查询距离、支持特征或评价结果。旧runner/历史结果不覆盖。默认preflight不导入Torch、不创建目录、不检查资源容量。

只需要原AeBAD数据（ZIP或AeBAD_S目录）、官方ViT-H+/16权重、随代码发布的宿主/骨干源码、已配置的Python环境。单GPU由用户选择；CPU workers为1至8。不会连接服务器、安装环境或自动关机。

## 原方法及本轮重建边界

- Source：复用`build_official_memory_spooled`，完整521张原排序入口，index%8==0的66张排除于候选，455张进入构库；原每层100K、100轮距离采样及随机顺序不变。磁盘spool与懒加载RGB只是存储/执行调整，使用identity feature callback，不修改特征。
- 必须区分Source构库与查询预处理：此AeBAD正式父路径`aebad_s_target_fewshot.runner.run`调用原`augmented_preprocessing`，配置亮度为[0.8,1.2]；查询和Target支持固定亮度1。本入口按这个实际调用链重建，并记录配置，不能将其写成Source NoAug。
- 66张Source-only距离重建patch MAD；同一66张计算四层插值峰值和原生有效3×3局部均值，复用`fit_image_mad`拟合CH1。后续所有分支共享这一份本轮状态，不使用测试图拟合。
- 原支持选择使用`CRC32(42|domain|relative_path)`再按路径排序，每域前8张，排除相同支持图。保留全测试排序的原sample ID；两域名单/顺序/图片身份还核对随代码冻结catalog。三域仍为297/265/364，共926张。
- 使用原`observe_fewshot_pil`及同次回调生成Source和8支持逐图NN距离/索引，复用原`add_geometry`生成SEGMENT，不保存完整Q。Source/Target原始欧氏距离只除1280一次。
- 后续直接复用现有GPU批量邻居、WITNESS、整图LOO、缩距匹配和预测冻结；没有新门控或参数搜索。
- 四分支UNION、SEGMENT、WITNESS、匹配均匀对照均重新评价，CH1/CURRENT同地图，像素AUROC/AP只算一次，AUPRO共3×4=12作业。原精确CPU像素后端、ADEval1.1.0/200阈值/FPR≤0.3不变。
- 新memory/MAD与新检索形成独立复现运行；没有历史银行及查询就不声称旧字节、历史距离或旧指标已经复现。重建冻结的是规则，文件允许重新生成。

## 阶段

`preflight / prepare-data / build-source / calibrate / capture / prepare-support / support-loo / predict / evaluate / evaluate-aupro / summarize`；`all`按顺序运行，30秒心跳、实时日志、退出码，最后打包成功或失败review。

`prepare-data`只解压ZIP中的AeBAD_S，按原排序建目录清单；图片与GT字节做身份记录，GT不解码。GT首次解码在全部926张预测冻结之后。完成Source构库后临时候选spool由原builder释放；本轮持久Source bank、66校准缓存、24支持特征、926检索缓存和连接证明保留。

SSD：代码、环境、运行状态/预测、精确像素排序临时文件。HDD：数据、权重、Source候选spool及新bank、校准/支持/检索/连接缓存。`--cache-root`必须使用本次运行独立目录。连接目录由入口自动软链接，不需要提前手动创建experiment目录。

下面是用户工作站的正式命令，只有用户手动执行才会调用GPU。物理GPU1通过CUDA_VISIBLE_DEVICES限制；程序内cuda:0指该可见单卡。

```bash
work=/home/ubun/YYK
repo="$work/DINO-Reference-AD"
py="$work/conda_envs/yyk/bin/python"
tmux="$work/conda_envs/yyk/bin/tmux"
run="$work/runs/aebad_s_witness_fresh_$(date +%Y%m%d_%H%M%S)"
cache="$work/storage/witness_cache/$(basename "$run")"
session="witness_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$work/runs" "$work/tmp" "$work/.cache" "$cache"
cd "$repo" || exit 1

"$tmux" new-session -d -s "$session" -c "$repo" \
  "env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg XDG_CACHE_HOME='$work/.cache' TMPDIR='$work/tmp' '$py' -B -u -m DINOv3.MADEqual.aebad_s_support_witness_segment_p0.fresh all --run-root '$run' --cache-root '$cache' --data '$work/storage/datasets/AeBAD.zip' --checkpoint '$work/storage/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth' --device cuda:0 --cpu-workers 8 > '${run}_launcher.log' 2>&1"

printf '%s\n' "$run" > "$work/runs/latest_witness_run.txt"
printf 'SESSION=%s\nRUN=%s\nLOG=%s\n' "$session" "$run" "${run}_launcher.log"
tail -n 60 -f "${run}_launcher.log"
```

Ctrl+C只退出tail，tmux中的任务继续。查看状态与退出码：

```bash
run=$(cat /home/ubun/YYK/runs/latest_witness_run.txt)
tail -n 40 "${run}_launcher.log"
cat "$run/pipeline_status.json"
cat "${run}.exitcode"
```

续跑用同一run/cache/data/权重和配置执行`fresh all`。已封存阶段复用，未提交或被修改文件拒绝；Source构库在整库提交前中断则该阶段需重建。失败review已存在时可先手动改名该review，再续跑，避免覆盖上次包。不要git pull后在旧目录混跑变化后的源码。review只打包状态、表格、日志、图和说明，不含HDD大缓存。

## 全部检查与行为

没有RAM/显存/磁盘容量门槛，没有估算不足就停止，没有失败切CPU或切另一张卡。

1. 参数与目录：单个显式CUDA设备、1..8 workers、正块大小；输出不覆盖数据/权重/代码。
2. 输入：原521/66和三域人口、CRC32支持/顺序/排除、sample ID、官方权重身份及原宿主设置。必要的数据字节身份在准备/实际读图时记录或核验，不反复预扫描大权重/全域特征。
3. 续跑：原目录的配置/代码、bank/校准/支持/逐图封存SHA、完整capture manifest保持一致。离线阶段不反复读取或要求Source bank实体；模型阶段实际加载时核验。
4. 数学：原同次Source/K8回调与正式检索数组一致；原SEGMENT三角/范围、WITNESS边界、均匀缩距ULP和冻结图像评分检查不变。
5. 评价：全926张预测冻结后才读GT；原图/GT地图身份、CPU中间指标seal/范围/像素数、AUPRO身份/范围、全部12作业和LOO完成后才能标记完成。
6. 真正CUDA、分配、I/O和数学异常停止后续阶段、保留现场并尝试失败打包；没有自动关机。

## 验证边界

本地测试实际执行纯CPU名单逻辑、Source校准评分数学、fresh输入适配/封存、原NN同次回调及SEGMENT（Torch CPU，骨干提取由替身替代）、真实NPZ预测冻结、精确CPU像素评价/实例/排序和pending汇总。没有本地GPU、DINOv3模型提取、真实Source采样或ADEval CUDA运行。完整新入口仍需工作站首次运行验证，不能将这些合成测试写成真实实验已通过。

输出：fresh_catalog/setup/protocol、image_calibration_state、capture_manifest、逐阶段cost_records及cost.json、pipeline_status、experiment下四分支指标/差值/实例/margin/机制/留一/报告。首次完整耗时和GPU加速尚无实测，原先6–12小时仅为规划预算。
