# AeBAD-S：固定Target局部图与第三方正常佐证 P0

独立工具；不修改旧WITNESS、历史输出或报告。默认是只读`preflight`，不导入Torch、不初始化CUDA、不创建实验目录。没有RAM/显存/磁盘容量准入判断，没有自动安装、下载、关机、换卡或CPU回退。真实执行必须显式指定阶段。

本轮问题分别是：固定图能否在完整Source+Target UNION之外增加正常覆盖；佐证能否优于普通局部图和同等缩距控制；检测与定位是否仍有取舍。新增参照只能降低UNION距离，不能抬高已被UNION压低的缺陷响应，也不保证margin保持。三个域已参与研究，不能称盲测或SOTA。

## 冻结宿主与fresh

直接复用原fresh Source构库/校准函数及真实宿主：ViT-H+/16、L7/15/23/31、1280D原始特征、原预处理和完整网格、每层100K/100轮采样。521张原排序Source中66张index%8==0只校准，455张构库；此AeBAD路径构库含brightness=[.8,1.2]，不是NoAug。Source patch/image MAD冻结到全部分支，不重拟合Target MAD。距离只除1280一次，原FP32四层均值/插值，CH1主评分、CURRENT补充。

默认seed42主表：background=297(85/212)，illumination=265(67/198)，view=364(84/280)，总926；原CRC32支持选择/顺序和全局样本编号不变，每域排除同一K8。Source/MAD和所有主要指标在同一运行配对，不拼不同银行的历史指标，不要求新文件等于历史字节。

`prepare-data → build-source → calibrate → extract-support → prepare-graphs → support-loo → select → predict → evaluate → evaluate-aupro → summarize`；`all`先做现有依赖实际导入、ADEval版本和CUDA功能检查，再串联这些显式阶段，30秒心跳、子阶段日志和退出码，成功/失败均尝试review打包。这里是功能检查，不是资源剩余量判断。无自动关机。`runner`本身就是fresh入口，不需要任何旧AutoDL路径、tar、memory、MAD或query缓存。

## 固定图和完整检索

1. 支持图对各保存Top100（或该图全部可用点）。每次按当前保留图重新合并为**整个跨图池**Top-k，按距离、原support_id、原patch_row排序；禁止同图点。只有双向进入Top-k才形成规范化无向边。每图Top100表仅是精确全局Top100的可复用充分候选，不是每图各连k条边。
2. 在当折互近邻正长度边内，按raw Euclidean长度的线性分位数裁边。零边移除并计数，不当作扩展。
3. `s(w)=median_j min_z ||w-z||`，j是除w来源图之外的当前训练图。它不是已知噪声sigma；零/非有限尺度不可作佐证，不加epsilon。
4. **穷尽全部第三方图的全部点**。alpha=[.2,.8]，`eta=h/s(w)`，取最小eta，保存代表图/行/alpha/h/s/eta、不同通过来源数及alpha六分箱。没有旧端点Top5限制，无额外角度或空间门控。空/拒绝原因在witness_search_summary中保留。
5. 查询检索分支的**整个固定图**，Source和全部离散Target始终保留：

   `A=||q-u||², B=||q-v||², C=||u-v||²`

   `alpha=clip((A+C-B)/(2C),0,1)`

   `D²=(1-alpha)A+alpha B-alpha(1-alpha)C`

   `d=min(d_Source,d_Target,min_e sqrt(D²)/1280)`。

支持/尺度邻居使用原直接FP64 cdist；新图几何使用分块FP64 GEMM恒等式，记录保守误差区间。可能获胜的边、临界alpha/eta以及不可靠消减项使用直接向量投影复核；明显超出误差界的负平方失败，不截零掩盖。Torch查询热循环的端点表、边公式、区间筛选和直接复核均在显式设备上；同域的支持/图张量跨查询复用。边长预处理同样分块，不一次展开全部E×1280方向向量。CPU统计与控制归约仍有成本，不声称所有阶段GPU满载。

## 只用正常支持选择

候选固定：k={10,30,100}，长度分位={.50,.75,.90}，eta={.25,.50,.75,1.00}。佐证36配置，普通图9配置；同域四层共用一配置。不能按测试指标追加范围。

整张支持图留一，以其余七图重建互近邻、正常尺度、边长阈值和佐证；被留出图完全不作参照。留一查询使用已经提取一次的支持特征，检测器是**Source+Target7+图**，不是旧Target-only LOO。Source和两级MAD冻结。

每配置由八张图的完整四层CH1分数组成。J=最高两分数均值；另报八图均值和逐图值。空图合法。若最佳J改善不超过固定1e-5（继承CH1数值容差），选空图；否则在达到95%最佳降幅的配置内，先最小化折/层等权的每patch归一化总边长，再按每patch边数、较小k、q/eta字典序。边长归一化=`raw_length/1280/Source_patch_scale`，不再除一次1280。95%不是置信度，八图留一不是独立泛化或低FPR认证。

佐证图与普通图分别正常选择。主候选选空时，配对G0也定义为空；独立普通图仍按自己的选择运行。最终八图重新构图，`graph_transfer.csv`报告七图折均值与八图实际值，不假设密度比例。

## 七个分支

|分支|定义|
|---|---|
|UNION_K8|原完整Source+K8离散点|
|SEGMENT_K8|原查询跨支持Top2 SEGMENT，复用原geometry_layer|
|LOCAL_GRAPH_PAIRED|主候选k/长度分位对应、没有佐证过滤的G0|
|WITNESS_GRAPH|正常选择的G1，G1⊆G0|
|LOCAL_GRAPH_LOO|在自己的9配置正常留一中选择的普通图|
|NORMAL_MATCHED_DAMP|每域每层lambda由八图LOO的图等权平均降距之比冻结|
|IMAGE_MATCHED_DAMP|每个当前query×层以beta匹配W相对G0的总原生缩距|

`lambda=sum_h mean(U-W)/sum_h mean(U-G0)`；`beta=sum_q(U-W)/sum_q(U-G0)`；零分母取0，验证[0,1]及`0≤U-W≤U-G0`。两个控制均为`U-coefficient*(U-G0)`。只容许原距离容差内的负微差处理；逐图总量匹配保留FP32 ULP审计。它们不是把最终图像分数乘常数，也不保证测试lambda实际总缩距等于W。

## 成本与可复用计算

Source构库/校准一轮一次，支持RGB一轮一次，正式query一轮一次前向后处理所有分支。支持图对同一个距离块同时更新两个方向的Top100（每层28对计算，保存56张有向表），并在k、阈值、折之间复用；不能复用含留出图的全局排序、尺度和边长阈值。原宿主FP32离散检索另算并缓存，不用图的FP64表偷换原评分后端。每折整图检索一次，同时归约45配置，不分别提特征或重复做45次端点距离。

GPU执行优化：完整佐证的alpha/eta判定、四个阈值的来源计数、alpha直方图及代表点更新均留在设备上，最后只回传11组逐边摘要，不逐块回传点×边数组。直方图使用整型scatter，代表并列顺序不变；临界值仍执行原FP64直接向量复核。仍有数值检查和动态复核索引所需同步，不声称完全异步。正常LOO的45配置每次最多8个批量归约，限制临时内存并减少小kernel启动。每域/层Top100表只验证加载一次，九折共享该只读表；七图全局关系与尺度照常重算，额外常驻RAM约1.2GiB（13924点/图）。

佐证进度新增`elapsed`与`edges_per_second`，第一块完成后及之后约30秒报告一次；每折结束记录真实`witness_seconds`。完整点—边比较数量不因这些优化减少，实际CUDA加速倍数尚未测量。新运行可显式使用`--witness-edge-chunk 512`加大执行块；这只是分块参数，不是方法候选或精度调整，也不承诺它在所有GPU上最优。

全新运行仍有约120.8GiB Source临时候选（构库完成后原builder释放）。13924patch/图假设下，三域四层支持图对Top100表约14GiB；持久图本体约198B/候选边另加节点字段，稠密互近邻上界可达百GiB量级。完整Q的`disk`模式约246GiB（926×4×13924×1280×4）；`stream`为0持久Q，但仍保存足够重评的距离/胜边数组。两者都需要预测、图、支持和评价空间，不能只按Q计算总磁盘。

`cost_estimate.json`在支持提取后按实际网格输出表大小和矩阵块字节；每图折`fold_*_estimate.json`在完整佐证计算前输出实际候选边数、边×第三方点/查询点组合数、图字节估算。**完整第三方搜索、96个LOO图检索可能显著超过旧WITNESS的约3小时；当前没有真实耗时测量，不能按旧实验外推。** 估算不阻止运行、不触发近似算法。

## 查询缓存与当前新服务器资产复用

- `--query-mode stream`：不保存完整Q，当前图像同次处理所有分支；中断未提交图像需重新前向，已提交图像不再前向。
- `--query-mode disk`：保存原始FP32四层Q、positions、几何和图像身份；下游失败后可直接续算，不再前向。相同模式/配置显式复用的新工具运行，还可在seed43/44中复用相同query的Q。
- `--reuse-fresh-run PATH`是可选的**当前机器显式资产来源**，支持已完成旧WITNESS fresh运行或本工具seed42运行。核对Source图像、代码、配置、软件身份和原封存记录后，在新输出中链接原Source bank、66校准缓存、同身份支持特征；Source MAD由同66缓存重新核对拟合，无新增模型前向。原文件和记录不改名、不覆盖。来源路径记入frozen_assets；链接依赖来源文件继续存在。
- 旧WITNESS没有完整Q，不能从其Top1距离恢复Q。复用其Source/支持可省掉构库与支持提取，但926张正式query仍需一次新前向。旧AutoDL任何资产都不是前提。
- 更新执行代码后必须用**新的run/cache目录**。先停止旧作业，再同步；不要在旧run里修改`run_config.json`或哈希来绕过版本身份。本轮按用户要求重跑，不实现旧Top100/旧图的跨版本迁移。原有Source/校准/支持复用仍可显式选择；不传`--reuse-fresh-run`则完整fresh。

## 手动运行示例（部署本模块后）

源码通过DINO-Reference-AD仓库同步；下列示例只在用户手动执行后才运行。所有存储位置、源码、权重和数据由参数提供；示例沿用当前工作站路径，默认单张物理GPU1和最多8个CPU工作进程，不影响base环境。

```bash
work=/home/ubun/YYK
repo="$work/DINO-Reference-AD"
py="$work/conda_envs/yyk/bin/python"
tmux="$work/conda_envs/yyk/bin/tmux"
stamp=$(date +%Y%m%d_%H%M%S)
run="$work/runs/aebad_fixed_graph_seed42_$stamp"
cache="$work/storage/fixed_graph_cache/$(basename "$run")"
common=(
  --run-root "$run" --cache-root "$cache"
  --data "$work/storage/datasets/AeBAD.zip"
  --checkpoint "$work/storage/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"
  --official-root "$repo/external_baselines/SuperADD"
  --dinov3-root "$repo/external_baselines/dinov3"
  --graph-root "$cache/graphs"
  --query-cache-root "$cache/queries"
  --prediction-root "$cache/predictions"
  --evaluation-scratch "$work/tmp/$(basename "$run")"
  --query-mode stream --seed 42 --evaluation-scope primary
  --device cuda:0 --cpu-workers 8
  --query-chunk 128 --reference-chunk 2048 --edge-chunk 2048 --witness-edge-chunk 512
)
cd "$repo" || exit 1
module=DINOv3.MADEqual.aebad_s_fixed_target_graph_p0.runner
"$py" -B -u -m "$module" preflight "${common[@]}"
```

可选：若要复用当前机器的成功WITNESS资产，在执行任何正式阶段**之前**追加：

```bash
common+=(--reuse-fresh-run "$work/runs/aebad_s_witness_fresh_20261023_160224")
```

首次运行可只检查部分构图作业（不是完整实验）：

```bash
# 先按顺序显式完成prepare-data、build-source、calibrate、extract-support。
# 然后可单独处理background/L7的留一折1：
"$py" -B -u -m "$module" prepare-graphs "${common[@]}" --domains background --layers 7 --heldouts 1
```

正式全流程tmux启动：

```bash
mkdir -p "$work/runs" "$work/tmp" "$work/.cache"
session="fixed_graph_$stamp"
printf -v launch '%q ' env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  MPLBACKEND=Agg XDG_CACHE_HOME="$work/.cache" TMPDIR="$work/tmp" \
  "$py" -B -u -m "$module" all "${common[@]}"
printf -v log_path '%q' "${run}_launcher.log"
"$tmux" new-session -d -s "$session" -c "$repo" "$launch > $log_path 2>&1"
printf 'SESSION=%s\nRUN=%s\nCACHE=%s\nLOG=%s\n' "$session" "$run" "$cache" "${run}_launcher.log"
tail -n 60 -F "${run}_launcher.log"
```

Ctrl+C只停止tail。程序内cuda:0是CUDA_VISIBLE_DEVICES限制后的物理GPU1，不使用另一卡。结束生成review包并保留所有大缓存；没有自动关机。

## 续跑与额外支持种子

同一run、cache及冻结配置再次执行原`all`或单阶段。完成的图对、图折、LOO距离、逐图预测、CPU指标与AUPRO作业按记录复用。支持`--domains`、构图/LOO的`--layers/--heldouts`、AUPRO的`--branch`。这些仅控制执行作业，不改变完整评价协议。`predict --limit 12`只新增最多12张，不保证是固定案例，也不允许未完成预测进入GT评价；正式全量时去掉limit即可续跑。

所有926/878张预测冻结后才允许GT解码。一个AUPRO作业示例：

```bash
"$py" -B -u -m "$module" evaluate-aupro "${common[@]}" --domains view --branch WITNESS_GRAPH
"$py" -B -u -m "$module" summarize "${common[@]}"
```

额外seed43/44用**独立run/cache**、`--seed 43`或44、`--evaluation-scope common`，可显式复用本工具seed42运行的Source/MAD；若前轮为disk模式，还可复用同身份Q。算法、候选集合与选择规则不变。公平跨种子比较还需独立执行seed42/common；三组共同名单为281+249+348=878，不能与seed42/primary的926张混表。不同seed的结果各有scope/seed列，按domain/sample_id连接比较。

## 输出与评价

- `run_config/protocol/setup/fresh_catalog/frozen_assets`、支持及评价名单：方法、软件/代码、Source/MAD/资产来源身份。
- `graph_summary/witness_search_summary/witness_summary/graph_transfer`：边长、度数、覆盖、图间边数、拒绝与eta统计、七图到八图变化。
- `support_loo_scores.csv`、`selection/<domain>.json`：八图逐配置CH1、J、复杂度、空图理由、lambda及冻结记录。
- 各图预测NPZ：七分支距离、positions、G0/W/独立普通图的获胜边和alpha、Source/Target距离、到代表佐证点距离；胜边号关联对应图NPZ中的端点图/行及代表佐证图/行。支持grid可将patch_row还原坐标。失败/UNION获胜用-1，缺佐证距离保留NaN，不填造。
- `metrics/metric_comparisons`：七分支CH1/CURRENT、五项指标和三域等权宏平均；差值方向明确。像素图同一分支只评一次，两种图像评分引用同值。
- `coverage_audit`：相对完整UNION的有效新增覆盖、正常高响应top1%/缺陷内部/周边、到已有UNION/Source/Target/代表佐证点距离、原始距离与MAD单位、集合单调检查。
- `image_peak_decomposition/score_locations`：CH1两部分、逐层插值峰值位置和原生3×3局部窗口中心，均标明本轮派生位置。另有实例/两种margin/小缺陷/NA、正常尾部/排序、R_N/R_A/G、固定背景12案例（同一色标）。不把排序错误或margin损失改叫部署漏检。
- `cost_estimate/cost_records/cost/status/objective_report/delivery_manifest`及日志。阶段末记录RAM/显存以及各配置目录的逻辑字节快照，磁盘只统计文件大小；链接和嵌套目录可能重复，不作物理占用或临时磁盘峰值解释。真实异常保留现场，pending不填0。

CPU像素AUROC/AP复用原全像素精确后端；AUPRO独立显式GPU作业，ADEval1.1.0、200阈值、FPR≤.3，重建图/GT SHA必须与CPU记录一致。默认21个作业（三域×七分支），CH1/CURRENT不重复。历史指标不冒充本轮结果。Source、checkpoint实体和完整Q在已冻结预测的离线评价中不参与计算；评价仍需图像清单、两级MAD、选择记录、预测和GT。

## 检查清单与验证边界

保留：原图/Source划分/支持顺序与排除、官方权重及宿主精度、显式复用来源身份、冻结代码配置、各阶段文件封存与续跑、有限性/合法索引/几何误差界、G1⊆G0与UNION锚点、lambda/beta与FP32匹配误差、全预测冻结、GT/地图及评价完整性。没有新增资源剩余容量门槛。官方宿主构造时仍执行其原权重SHA核验，不伪称完全不读权重；不会在每个query前重新核验大权重。

本地验证运行小型真实CPU数学与NPZ流程；设备热循环用Torch CPU执行同一代码，不调用CUDA。端到端夹具只替换骨干RGB输出和Source银行加载，空图流水线覆盖完整阶段衔接；非空全搜索由独立几何测试覆盖。没有真实DINOv3提取、正式Source重建、真实规模图构建、CUDA、多卡或ADEval GPU实测，不把合成成功写成正式实验已通过。

本地测试命令：`python -B -m unittest DINOv3.tests.test_aebad_s_fixed_target_graph -v`。最终测试结果与只读输入预检另见implementation_audit.md。
