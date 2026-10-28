# 同源邻域差分检索，固定background最小验证

仅 AeBAD-S background / seed42 / 原K8，297图（85正常、212异常），521张Source按原顺序 index%8==0 的66张校准，455张构库，每层100K。无训练、无参数搜索。默认 `preflight`，不会加载Torch、模型或读取GT像素。本轮实现交付不授权远程连接或真实GPU执行。

## 固定方法

六行：UNION_K8、SEGMENT_K8、SEGMENT_IMAGE_UNION_MAP、ABS_NEIGHBOR_C1（A）、REL_NEIGHBOR_C1（B）、REL_NEIGHBOR_C16（C）。固定组合只引用SEGMENT图像指标、UNION像素指标/AUPRO，不产生地图。A不是旧TOPK-CTX-J0的严格复现。

每层1280维FP32，不归一化原特征。对上、下、左、右方向：

```
d_abs = norm(q - m) / 1280
d_nb  = mean_u norm(q_neighbor[u] - m_neighbor[u]) / 1280
d_rel = mean_u norm((q_neighbor[u]-q) - (m_neighbor[u]-m)) / 1280
A = .5*z_abs(absolute Top1) + .5*z_nb(same Top1)
B = .5*z_abs(absolute Top1) + .5*z_rel(same Top1)
C = min_{valid j in original absolute Top16} .5*z_abs(j) + .5*z_rel(j)
```

邻居必须同一正常原图、同一次实际亮度增强、同一完整拼接网格；裁块接缝不是边界。query最外圈，**或绝对Top1中心不具完整四邻域**，三个新分支一律回退到该层UNION标准化响应。C16只过滤原Top16，不从第17名补位。输出边界数量、Top1回退数量、有效位置和被剔除候选数。

Top16按距离、全库行号稳定排序：Source bank行顺序在前，之后K8 manifest支持顺序和物理patch顺序。联合代价相同则保留该Top16顺序中的首个。Top16只是工程召回约束；不是全库联合最优或防错匹配保证。A→B检验表达，B→C检验候选内选参照，不声称两因素交互协同。

## 校准、评分、评价

第一遍66图仅对Source库检索：每层绝对Top1，d_abs统计所有patch；d_nb/d_rel仅统计双方完整四邻域位置。直接复用原 `fit_calibrators` 的float32输出与MAD下限。B/C共享完全相同的分量MAD，不以联合最优重新拟合。第二遍从已保存Top16分量重放，各新分支分别按实际候选/边界规则拟合CH1图像MAD。联合响应已经标准化，绝不再套旧patch MAD，也不截负值。

复用原 `from_z/response`：native四层FP32均值、原FP32双线性、逐层峰值、valid3x3局部统计、CH1；CURRENT补充，二者共用地图。CPU使用原全像素精确ROC/AP；独立 `aupro` 使用原ADEval1.1.0、200阈值、FPR≤.3、完整GT resize。没有近似替代。原实例/环带/固定UNION峰值内核直接复用，NA保留。新响应标度不同，margin只是补充；不能叫原生距离缩减或漏检数量。

同库、父链路完整且通过本轮基线响应/评分、地图/GT复核才复用历史UNION/SEGMENT指标。独立重建一律使用新共同Source库，重做其66图基线MAD、UNION/SEGMENT预测和评价，不拼接旧指标。六分支×两评分共12行；定位图逻辑5个，实际新增3或5个CPU/AUPRO作业。

## 资产先恢复，不自动重建

`preflight.json`、`missing_inputs.csv`、`feature_inventory.csv`区分可复用、待恢复、缺失和不能本地确认的资产。原builder已有 `return_selected_indices`；会检查显式NPZ、原cache的`source_bank.npz`，并沿Source导入manifest来源链找保留的索引。**索引不是完整来源**。

1. `--source-neighborhood` 指向本工具已完成的邻域bundle，可直接只读复用，保留原provenance。
2. 有索引时，优先读 `--augmented-grids/source_<原Source索引三位>/manifest.json`（L7/L15/L23/L31完整原增强网格，metadata含relative_path、source_index、grid、brightness_factor）。否则按同宿主初始化后的原NumPy状态、455图调用顺序恢复原亮度增强。每个被选中心必须与正式bank对应行**精确相等**，否则保存差异并停止，不近似认领向量，不自动重建。恢复逐图保存选中行邻域检查点，断点可复用。
3. 确实无法恢复时，显式 `--rebuild-source`：局部绑定原builder的采样观察函数，在原候选spool尚在时提取最终选中行四邻域。保留原增强调用、候选顺序、100轮采样和随机调用顺序；不改原builder源码，不套NoAug。记录增强因子、原图/原Source索引、网格、选中候选索引、bank行号，以及physical position（行=position//width，列=position%width）。独立重建与旧库的差异单独记录。原builder重建若中断需重做该阶段，其他完成阶段不重做。

Source只为最终100K条目保存delta_m；不会把全部候选扩成五倍。Target完整K8保存同源delta。CPU和GPU都显式使用positions映射。完整四层特征可从近期reference-residual缓存复用；普通未增强Source网格不能替代增强构库网格。查询不默认永久保存，缺失时一次完整前向，五张地图共享。66图只保存小型检索/分量结果；第二遍无重复前向或NN。

## 时间、内存与磁盘：正式启动前先看预检

暂按**当前工作站历史日志**估计，尚未测新Top16内核：

|阶段|已有索引需恢复邻域|确需独立重建|
|---|---:|---:|
|Source来源/邻域|0.4–1.3小时|0.75–3小时|
|Target及补完整特征|0.05–0.3小时|相同|
|66图两遍校准|0.25–1.5小时|相同|
|297图提取/检索|0.6–3小时|相同|
|CPU/AUPRO/汇总|0.1–0.5小时|相同|
|总计，全部必须工作|**1.4–6.6小时**|**1.75–8.3小时**|

依据：此前455完整提取约24分钟；SACP926图预测4506.7秒（每图约4.9秒，非本轮Top16）；background单地图CPU36.7秒、AUPRO15.8秒。不同任务吞吐不能直接当本轮实测。Source来源失败、机械盘争用和新Top16吞吐仍未知，区间不是承诺；不能只拿前向时间当总时间。

`smoke`显式测前两张正常校准图Source-only Top16，结果直接复用于校准；不声称其等于Source+K8正式吞吐。首批正式图会继续以实测外推。**明显预计超过12小时或已耗时达到12小时，在已保存图像/作业处暂停并写budget_pause.json，退出75；须用户明确同意后才用 `--allow-over-12h`。** 不通过减少memory/图像或近似评价降成本。

单层Source delta约1.91GiB，四层7.63GiB；K8四层delta约8.50GiB；中心约4.03GiB。四层参照驻留GPU约20.2GiB，另加原骨干/提取临时量，暂估峰值30–40GiB显存、进程RAM约20–40GiB，均未实测。只用一张用户选定GPU，不自动启用第二张。

新增持久资产及恢复检查点约25–35GiB，评价/排序临时量另预留10–15GiB。独立重建另外有**原始候选约120.84GiB临时文件**，总峰值需按约140–170GiB规划，放机械盘 `BUILD_SCRATCH`；不会塞进当前仅约84GiB空闲的SSD。SSD只放代码、环境和独立评价scratch。没有资源前置拒绝门槛，仍会记录实际分阶段时间、进程峰值RSS、CUDA峰值分配/保留、新写数组字节数；空间/内存异常原样保留。

## 手动工作站命令（本轮不连接、不推送、不执行）

先在已有yyk环境、同步本模块后设置路径；不是过期AutoDL路径：

```bash
source /home/ubun/miniconda3/etc/profile.d/conda.sh
conda activate /home/ubun/YYK/conda_envs/yyk
export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
cd /home/ubun/YYK/DINO-Reference-AD
export PARENT_RUN=/home/ubun/YYK/runs/aebad_s_sacp_seed42_20261028_011454
export PARENT_CACHE=/home/ubun/YYK/storage/sacp_cache/aebad_s_sacp_seed42_20261028_011454
export RAW_CACHE=/home/ubun/YYK/storage/reference_residual_cache/aebad_reference_residual_compact_v1_20261027_002907
export RUN_ROOT=/home/ubun/YYK/runs/aebad_same_origin_neighbor_seed42_$(date +%Y%m%d_%H%M%S)
export CACHE_ROOT=/home/ubun/YYK/storage/same_origin_cache/$(basename "$RUN_ROOT")
export GPU_INDEX=1
script=DINOv3/MADEqual/aebad_s_same_origin_neighbor_p0/scripts/run_workstation.sh
bash "$script" preflight
```

检查清单中原索引的准确位置。若已发现，正常恢复/正式路径：

```bash
bash "$script" prepare-source
bash "$script" prepare-target
bash "$script" smoke
# 阅读smoke_report与耗时估计；确认后再正式all；已完成两图不会重算。
mkdir -p /home/ubun/YYK/runs
tag=$(basename "$RUN_ROOT")
tmux new-session -d -s "$tag" "cd /home/ubun/YYK/DINO-Reference-AD && env GPU_INDEX='$GPU_INDEX' RUN_ROOT='$RUN_ROOT' CACHE_ROOT='$CACHE_ROOT' PARENT_RUN='$PARENT_RUN' PARENT_CACHE='$PARENT_CACHE' RAW_CACHE='$RAW_CACHE' bash '$script' all > '${RUN_ROOT}_launcher.log' 2>&1; printf '%s\n' \$? > '${RUN_ROOT}.exitcode'"
tail -n 40 -F "${RUN_ROOT}_launcher.log"
```

显式向tmux传入路径和GPU，避免已有tmux服务保存的环境陈旧。只启动一次。同一run使用相同参数重入 `all`，跳过已封存项。不要为续跑生成新RUN_ROOT。

若确认需要独立重建，用**新目录**，其所有阶段保持同一 `--rebuild-source` 参数：

```bash
export RUN_ROOT=/home/ubun/YYK/runs/aebad_same_origin_rebuild_seed42_$(date +%Y%m%d_%H%M%S)
export CACHE_ROOT=/home/ubun/YYK/storage/same_origin_cache/$(basename "$RUN_ROOT")
bash "$script" preflight --rebuild-source
bash "$script" all --rebuild-source
```

无父缓存时可显式提供 `--catalog` 元数据目录里的catalog.json、`--source-bank`、`--selected-indices`和`--feature-root`；历史预测不是算法必需输入。只有距离预测而没有完整1280维网格，不能替代本轮输入。

### 特征提供者绑定（P2续跑修复）

新运行的 `inputs.json` 同时冻结 `feature_root`、可选 `feature_manifest`、完整提取依赖记录，以及逐图提供者路径/manifest seal/数组名/网格和提供者声明身份。提取记录包括checkpoint、原预处理host_config、精度、软件及提取代码身份。只凭图像和1280维形状不再接纳特征。

现有reference-residual缓存可通过父资产中记录的近期原运行 `raw_assets.json` 或 `inputs.json` 证明配置；不会因为旧图像manifest没有内联extraction就默认拒绝所有旧缓存。独立导入可设置 `FEATURE_MANIFEST=/path/to/original_run/raw_assets.json`（脚本对应 `--feature-manifest`）；该文件需保留原封存记录，图像路径/seal必须对应实际bundle。也支持完整提取记录内联在图像manifest的 `metadata.extraction` 中。未知或不匹配的提供者明确拒绝，不自动改身份或偷偷重提取。

续跑时更换目录、checkpoint/预处理、图像manifest或提供者manifest均拒绝；原本计划fresh的图像也不能悄悄改用新出现的缓存。内部持久特征同样保存运行身份、提取身份和来源绑定。真正需要新前向时，当前宿主提取配置必须与冻结配置一致。

这些检查只读小型配置/manifest及NPY头，不重复计算大特征文件SHA。同一阶段的小型声明按文件状态缓存。`evaluate/aupro/summarize`仍可在原特征目录离线时使用已冻结预测和配置，不额外要求恢复Source或checkpoint。旧版未记录feature_inputs的运行不能直接续算，须新建运行目录；不改写旧manifest。兼容原始特征与Source邻域仍可只读提供给新运行。

若同库基线复核不通过，保存差异并停止，不放宽容差。可在**新run**显式用 `--baseline-mode fresh` 重新校准、评价UNION/SEGMENT，仍复用这一份Source库与已完成Source邻域（`--source-neighborhood <已完成bundle>`），不需要重建memory。所有阶段保持相同参数。

## 检查清单与未验证部分

检查仅围绕本方法：521/66划分与K8排除；Source真实增强中心和坐标；完整positions；稳定Top16；校准/预测身份；同批缓存分数/地图；原CPU中间结果封存/范围/像素数量；原ADEval地图/GT一致性；输出和scratch不覆盖只读父资产，清理只删除本次创建的scratch；以及用户要求的12小时暂停。没有RAM/磁盘/显存前置门槛，没有重复全缓存SHA扫描。新产物首次写入保留哈希，小型不可变记录继续核对。原宿主构造函数自带权重身份检查，未改其定义。

本地只做必要CPU测试；真实CUDA性能、455图增强来源恢复、100K重建、66图校准和完整五项指标尚未运行。无自定义Triton。代码交付后停止，交review，不自动上传GitHub或执行服务器操作。

## 本地验证记录

2026-10-29：**33项CPU测试通过，13.977秒；13份Python文件语法检查通过**。可运行 `python -B -m unittest DINOv3.tests.test_aebad_s_same_origin_neighbor -v` 复核；本机validation和资产预检不上传。包含真实原builder/100轮采样器在7个小型替身输入上的前后对照（4维、每层10K行，仅CPU），验证bank值、选中索引及NumPy后续随机状态不变；不等于完成了正式455图/1280维/100K构库。另有66份人工4×4网格的两遍校准、独立图像MAD、分量缓存重放测试，以及预测冻结重入不加载模型、AUPRO后处理不依赖Source向量实体的测试。

新增检索与残差为FP32。既有SEGMENT小块端点几何继续使用其原FP64实现，以保留正式定义；没有将大规模NN改成FP64。
