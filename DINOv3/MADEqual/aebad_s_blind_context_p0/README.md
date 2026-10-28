# 5×5块盲点上下文正常参照合成：background固定验证

本轮交付实现、CPU夹具测试、资产预检和未来手动命令。代码发布不代表真实实验完成；未连接服务器、未启动CUDA或真实模型。实际状态以 `delivery_status.json`、`validation_results.json` 和运行目录状态为准。

只研究 background 原seed42 K8和297张评价图（85正常、212异常）。不扩展illumination/view、其他支持、种子或参数。此前SACP两档强度的正式结果只作为已有背景，不修改其代码、产物或窗口记录。

## 固定配置与任务

`protocol.py` 为唯一固定定义，prepare-context时冻结 `protocol.json`、`job_plan.json`、`training_images.csv`、`evaluation_images.csv` 和输入身份。

|项目|固定安排|
|---|---|
|基础检测器|原ViT-H+/16，L7/15/23/31、1280D，正式Source每层100K，完整K8|
|辅助Source图|455构库图中按SHA256(`42\|blind_source\|relative_path`)和路径排序取64；66校准图完全排除|
|输入盲点|全局预处理后的归一化输入，5×5不重叠patch块，块坐标奇偶划为四相，所选相像素置零后才切窗|
|字典Source|每图最多512物理位置；值为这些正常位置到正式Source库Exact FP32最近邻向量|
|字典Target|原K8每图全部位置；值为该位置原始正常1280维特征|
|上下文键|每层独立固定seed随机1280→128投影、L2归一化；不改变正式memory与距离|
|候选|只用遮挡上下文：Source Top4，每张可用Target Top1；测试12、Target训练11|
|权重|四层各Wq/Wk 128→32，无线性偏置；映射后L2归一化，温度0.1，Source/Target类型偏置2项|
|输出|softmax非负归一化权重，原始正常值的凸组合；总32776参数|
|训练|seed42，AdamW lr=1e-3、wd=1e-4，1500个联合更新，四层等权|
|batch|每步Source/Target各64位置；人口内图像等权再图内位置等权，四层共用物理位置|
|监督|原始q只作平方误差目标；按64张Source训练位置原始特征固定标量RMS归一化|
|校准|只复用原Source patch和CH1 image MAD，不重拟合；CURRENT补充|
|新增评价|一张BLIND_CONTEXT_UNION定位图、一项CPU全像素精确评价、一项原ADEval AUPRO|
|历史参照|UNION、SEGMENT、SEGMENT_SCORE_UNION_MAP只读复用；组合没有额外地图/AUPRO|

Source选定物理位置使用PCG64与协议/seed/用途/路径稳定派生。按物理位置排序，与compact行顺序区分。Source训练使用同一512位置；Target训练从完整字典位置稳定抽最多1024位置。原variant/合成池不参与本轮。

这是分块盲点，**不是每个位置居中的5×5窗口**。118×118网格时块边缘按图像边缘裁剪；每个位置只收集自己所属块被遮挡的那一相。`query_masks`仍仅用于宿主提取后的传输筛选，不能创建盲点。真正遮挡发生在 `_extract_processed_batch` / 宿主重叠裁块之前。

原预处理不保证输入边长为16的倍数：3024×0.625得到1890，原axis_patch_split仍只取到1888，对应118×118输出网格。遮挡保持完整1890输入尺寸，仅映射实际完整patch覆盖范围；末尾未被原切窗读取的2行/2列原样保留。没有额外缩放、补零或裁掉输入来迎合遮挡。缓存与隔离记录另存effective_pixel_shape和ignored_trailing_pixels。最右下角的隔离干预也只覆盖实际完整patch对应的像素范围。

禁止原图q、原图CLS、UNION距离/最近邻、原图候选或置信门控进入生成器API。预测中先完成全部四层上下文参照，再加载原q和父UNION。字典正常值与训练目标是允许的未遮挡正常信息，查询字典图时排除该图全部条目，不能从同图其他patch绕过K7。

## 数学与精度

每层：`a = softmax(dot(normalize(cWq),normalize(kWk)) / .1 + type_bias)`，`m_hat=sum(a*v)`。损失是 `mean((m_hat-q)**2)/sigma_Source**2`，四层均值。Source和Target各64，学习率固定，不按测试结果挑checkpoint；保存最后1500步。

正式距离：`d_pred=sqrt(sum((q-m_hat)**2, FP32))/1280`，`d_new=min(raw_UNION,d_pred)`。使用实际父`raw_union`，**不逆变换z恢复原生距离**。零改变位置复制父UNION z；其他位置使用相同Source patch MAD。原生四层FP32均值→原bilinear，原CH1峰值/有效3×3局部统计与CURRENT完全复用。

骨干沿用兼容父提取配置（本机近期父配置为原FP32，最高matmul精度、TF32关闭），不默认改为BF16。上下文投影、候选检索、注意力和距离为FP32；仅Source RMS小型统计复用原FP64累加。没有全流程FP64、自定义Triton、近似检索、替代AUPRO或环境安装。

## 资产接入与本地预检

显式父运行是近期完成的SACP λ=.5：

```text
/home/ubun/YYK/runs/aebad_s_sacp_seed42_20261028_011454
/home/ubun/YYK/storage/sacp_cache/aebad_s_sacp_seed42_20261028_011454
```

原始特征提供者默认为：

```text
/home/ubun/YYK/runs/aebad_reference_residual_compact_v1_20261027_002907
/home/ubun/YYK/storage/reference_residual_cache/aebad_reference_residual_compact_v1_20261027_002907
```

这几份是迁移后近期资产，不引用旧AutoDL实体。允许 `--raw-run` / `--raw-cache` / `--source-bank` 显式重定位；保留父原始manifest和seal，不重写旧身份。原始Source库必须与父声明seal一致，四层实际NPY头为100000×1280 FP32。缺库或MAD直接列缺项，不提供重建入口。不存在的非必需折库、子空间、SACP eta或旧训练模型不成为依赖。

父297份预测只需要 `UNION_K8.npy`、`SEGMENT_K8.npy`、`raw_union.npy`、`positions.npy` 及对应manifest；不用SACP方向缓存。需原CPU map/scores/instances表、MAD、Source与提取元数据和历史指标/AUPRO小记录。历史基线只有人口、支持、地图、GT与评分核对通过才复用。

64+8张原始正常特征优先只读引用近期完整正常图（不复制20GB）。原始查询若缓存存在也复用，否则未来授权后再前向一次并按图流式释放。原始特征缺失是“新增提取工作”，不是用旧距离替代向量。被遮挡上下文没有旧等价物，最多72×4+297×4=1476次完整四层提取，不能宣称本轮只离线读距离。

开发工作区的 `local_preflight/` 已生成固定64+8名单、297评价名单、配置/任务表、完整缺失CSV和预检JSON，不随GitHub上传；服务器可用下方preflight命令生成自己的清单。服务器绝对路径在Windows不可访问不表示远端丢失。未运行真实提取、训练、五指标或AUPRO。

## 阶段、隔离及续跑

```text
preflight -> prepare-context -> check-isolation -> prepare-training
          -> train -> check-isolation(最终模型) -> predict
          -> evaluate -> aupro -> summarize
```

- 默认preflight不导入Torch、不初始化CUDA、不写运行目录；显式inventory-dir才写清单。
- prepare-context最多四相一次各得四层；Source只把512选择位置传回，Target全位置。GPU投影到128维后传CPU；每层字典单独落盘。
- prepare-training预计算整图排除的候选索引及128维查询上下文，目标q独立保存。训练期间不执行DINO/Exact，也不重复上传bank。
- train将有界字典和训练张量常驻单GPU；只每100步同步日志/写checkpoint。更新数相同但采样或loss_scale变化不兼容旧模型。
- 每100步checkpoint分代保存、最后提交小指针；中断读最后完整代。batch随机数由step稳定派生，与续跑会话无关。训练只生成一个最后模型，不选择中间效果。
- check-isolation选择一张Source与一张支持，覆盖图像角、块边界、块内部，常数/固定噪声两种干预。使用实际原宿主和与predict相同的GPU候选/权重/凸组合，未训练与最终模型各检查一次；不读上下文缓存来伪造相等。一次宿主前向同时得到四层。输入必须逐位相同；描述、权重和参照沿用atol=1e-7/rtol=1e-6，候选索引严格相同，失败不放宽。
- 对非16倍数输入，隔离额外覆盖右下最后一个物理patch；边界取原宿主有效像素范围，而非把未参与切窗的尾部误算入某个patch块。
- 297新预测全部冻结后才读取GT。CPU重建原U/S并对照历史map/GT SHA及同缓存评分；只排序新分支全像素。实例复用原环带、自身峰值和固定UNION峰值，继承最小3实例及NA，少量案例沿用原固定名单。
- 预测冻结文件只绑定身份、逐图记录、图像数和GT未参与声明；本次前向/新增图像计数返回到独立会话cost记录。已完成预测再次predict（包括all进入predict阶段）时核对后保留冻结文件原始字节与seal，不打开RGB、初始化GPU或重建合成器。旧格式包含执行计数时也保留，不改写为新身份；预测记录不一致仍拒绝。尚未提交全局冻结但逐图缓存已齐时，也可无GPU补提交。
- aupro独立且仅一作业，直接原`official_aupro_adeval`、ADEval1.1.0、200阈值/FPR≤.3，同CPU地图与GT字节身份，不生成第二份评分地图。
- summarize有任何未完成项均pending；不填0、不自动扩域/补控制。all只执行固定清单，结束打review包，无关机。

当前单GPU政策沿用工作站约定。脚本仅限制一张可见卡（默认物理GPU1），最多8个CPU工人，不自动用双卡。

## 未来用户明确授权后的手动命令

GitHub只发布代码、测试和说明，不包含数据、权重、缓存、本机预检或实验结果。真实GPU执行仍需用户明确授权；之后在服务器已有yyk环境执行下列命令。不安装或升级依赖，不修改base。

只预检：

```bash
cd /home/ubun/YYK/DINO-Reference-AD
export RUN_ROOT=/home/ubun/YYK/runs/aebad_s_blind_context_seed42_v1
bash DINOv3/MADEqual/aebad_s_blind_context_p0/scripts/run_workstation.sh \
  preflight --inventory-dir "$RUN_ROOT/inventory"
```

先测真实正常RGB路径的少量吞吐（2张含Source和Target各一张，最多8次遮挡提取，另计缺失原图提取）：

```bash
bash DINOv3/MADEqual/aebad_s_blind_context_p0/scripts/run_workstation.sh \
  prepare-context --limit-normal 2
```

`throughput.json`分开上下文与原始特征耗时，只外推剩余上下文；候选准备、头训练、原始查询补提、CPU/AUPRO尚未测的项为null，不报0或承诺总小时数。内部裁块前向与完整四层提取分别计数。未请求的其他域不进入估算。两张产物可原目录续用；部分测速不生成训练完成标志。

确认吞吐和资产后，未来授权固定全程后台：

```bash
bash <<'BASH'
set -euo pipefail
source /home/ubun/miniconda3/etc/profile.d/conda.sh
conda activate /home/ubun/YYK/conda_envs/yyk
export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
work=/home/ubun/YYK
repo="$work/DINO-Reference-AD"
tag=aebad_s_blind_context_$(date +%Y%m%d_%H%M%S)
run="$work/runs/$tag"
mkdir -p "$work/runs" "$work/tmp" "$work/.cache"
touch "${run}_launcher.log"
tmux new-session -d -s "$tag" \
  "cd '$repo' && env GPU_INDEX=1 CPU_WORKERS=8 RUN_ROOT='$run' \
  bash DINOv3/MADEqual/aebad_s_blind_context_p0/scripts/run_workstation.sh all \
  > '${run}_launcher.log' 2>&1; printf '%s\n' \$? > '${run}.exitcode'"
printf '%s\n' "$run" > "$work/runs/latest_blind_context_run.txt"
printf 'SESSION=%s\nRUN=%s\n' "$tag" "$run"
tail -n 40 -F "${run}_launcher.log"
BASH
```

分阶段执行用上述相同RUN_ROOT分别传stage。train后须再次执行check-isolation，否则predict拒绝。续跑既有目录需设置原RUN_ROOT、CACHE_ROOT、EVALUATION_SCRATCH及相同输入，不重命名或覆盖历史目录。若先测速，可在相同已冻结目录执行all，避免重做那两张正常图；示例后台命令默认新建独立目录。

## 输出、资源和检查清单

输出protocol/job_plan/输入、固定名单、asset_inventory、context_ready、training_ready、每100步日志/恢复指针、最终小头、isolation_checks、逐图预测清单、metrics、逐图评分、numerical_verification、完整实例/NA及margin_summary、R_N/R_A/G、固定案例图、stage_costs、objective_report和完成状态。训练权重、字典与大数组、BIN不进review包。

典型118×118网格，Source最多32768字典条目、Target111392条目；四层1280D值约2.75GiB，键约0.275GiB，40960个训练位置的四层目标约0.78GiB，其他小索引另计。这是布局量级，不是已测峰值。新增全正常特征若全部缺失可能另占约19.1GiB；已有特征只读引用。原始查询默认流式不新增约79GiB缓存。SSD只存代码/报告和一个域的评价私有scratch；大字典、上下文、训练数组放storage机械盘。

只保留以下检查，不添加容量准入门槛或旧大文件反复SHA扫描：
1. 原始人口/排序/mod8校准/K8排除、Source和MAD小manifest身份及所需实际NPY头。
2. 原始提取配置/软件版本及小源码；宿主原本的权重身份检查；RGB/GT只在真正消费时按原reader核对。
3. 全局输入遮挡及真实信息隔离；整图排除、层/物理位置、候选数、训练输入和最后步模型一致性。
4. 同缓存评分原atol=1e-5；U/S历史地图/GT和实例标量；新增CPU/AUPRO地图身份。
5. 私有scratch与父运行/输入隔离，只清理本轮创建文件；精确像素中间结果独立封存/范围/像素数，完成小表与恢复checkpoint核对。
6. 新数组首次写出记录自身SHA一次；可信本地读取不反复扫描旧feature/bank负载。启用全量审计不作为本工具默认行为。

测试命令：`python -B -m unittest DINOv3.tests.test_aebad_s_blind_context -v`。CPU使用合成夹具，含实际Torch CPU投影、搜索/注意力适配器、梯度一步及原地图/评分/像素/实例内核。实际CUDA、原骨干遮挡数值隔离、吞吐、1500步真实数据训练和ADEval均待未来授权验证，不能以CPU夹具通过替代。

1890尺寸错误修复后的复测使用真正的原axis_patch_split、裁块、拼接和_extract_processed_batch（仅DINO骨干输出为CPU替身），覆盖标准1890、矩形余数、所有四相和右下角输入干预。不要在已用旧源码冻结inputs.json的失败目录中覆盖身份；保留失败目录，新建运行目录重试，原Source库与正常特征仍按原依赖只读复用。

结论限于单个background开发域和一版固定配置。信息通路正确不等于缺陷保持；没有额外控制不声称上下文是唯一收益来源。无效时也不否定所有上下文条件化方法。

本轮实测：35项CPU测试通过，40.730秒；12份Python源码编译及Bash语法通过。历史SACP的10份生产Python与正式父运行记录一致。本地预检退出2、逻辑错误0：缺正式Source库1、父预测bundle 297；MAD及固定64+8/297名单元数据可读。已定位本地_external/dinov3_checkpoints下的权重及_external/dinov3源码，仅检查存在性，未扫描权重内容SHA、未加载模型。正常原始特征72张和原始查询297张在本机未找到，列为未来补提/远端复用待确认。ADEval本机未安装，提取软件与父环境不同，不安装或升级。详见validation_results.json与local_preflight。
