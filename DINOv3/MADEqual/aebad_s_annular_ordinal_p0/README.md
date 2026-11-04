# 外围几何与中心原图次序证据（固定起点）

本工具检验：外围确定正常几何解释并冻结后，中心原图的次序差异能否改善 SEGMENT CH1 图像排序。原始配置只有 background / seed42 / 原 K8，沿用两组支持并集排除后的 **289 张（77 正常、212 异常）**。Source455 原始正常图构成像素参照池；独立 Source66 仅用于校准。

本次交付只做本地 CPU 逻辑测试和只读资产预检。未连接服务器、未启动真实索引、提取、训练、推理或指标实验，未安装依赖。容量 LP 已终止，不导入或恢复其求解工具；不做产品裁剪。原代码、历史结果和报告不改。

## 复用接口与输入

| 项目 | 现有接口/来源 | 本轮处理 |
|---|---|---|
| 289 共同名单、K8 顺序、支持并集排除 | `aebad_s_target_loo_support_retest_p0` 的共同 manifest | 原样读取，不重划分 |
| Source455/66 | residual compact 的 `catalog.json`、`validate_source` | 像素 Source 只用455张原图，无构库增强 |
| 原 SEGMENT CH1 P、定位身份 | two-K8 background/seed42 已封存预测 | 原 P 逐值引用，定位引用 `UNION_SOURCE_WEIGHT`，无新定位计算 |
| Source66 的 Source＋当前K8 P | conditional 的 `contract` / `bp` | 核对 Source、提取/MAD/几何、当前K8；不接受 Source-only 替代 |
| 原始解码 | segment-pair 的 `Data` | RGB、编码栅格方向，不做 EXIF transpose、缩图或平滑 |
| 缓存/小记录 | residual 的快速可信本地存储 | 新像素缓存，不复用DINO数组；小记录身份＋数组头，不反复扫大数组SHA |
| Image AUROC/AP、排序对分析 | 原 CH1 `image_metrics`、conditional `ranking` | 标签只在全部分数封存后参与评价 |
| review 归档、资源成本记录、路径隔离 | joint/shared-residual runner 的既有接口 | 白名单打包，仅报告/固定案例；独立cache不进包 |

本地盘点报告在 `DINOv3/outputs/mad_equal/aebad_annular_ordinal_preflight_v1/`。当前目录项与身份记录齐全，兼容P为289＋66条；服务器现状未远程核查，不能据此保证挂载和文件仍可访问。预检只查zip目录/文件存在性，不预先解码818张原图。

## 固定计算

`protocol.py` 是唯一方法配置。运行前生成不可变 `fixed_config.json`，包括两组固定像素对及全部24方向偏移。正式 inputs 同时绑定原图、名单、P来源、定位引用、解码版本和本模块源码身份。无窗口、分支、检索量或阈值搜索参数。

- 原分辨率，逐像素 FP32 灰度 `(0.299R+0.587G+0.114B)/255`。W64、步长32、末端贴边去重；外框128、保护带8，边界只用真实像素。
- 查询外围8像素网格，按四个方位各取最多4个正3×3范围对比度锚点，稳定并列按y/x。整17×17范围均须在真实外围。
- 查询描述128固定严格比较。正常侧全图8像素网格，固定9个旋转/尺度版本（−15/0/15度，0.9/1/1.1）；FP32双线性采样，明确正权重支撑，不补边。
- 现成 **OpenCV FLANN LSH**，近似检索；algorithm=6、table_number=4、key_size=20、multi_probe_level=1、seed42。无自定义Triton、无Python全库暴力距离扫描、无训练索引。实际OpenCV版本随索引记录，跨不兼容版本拒绝直接加载。
- Source统一索引，每锚点最多16个物理匹配；8个Target各自独立索引，每张最多4个。先请求至多预算×9＋1条原始条目，按汉明距离/原图/坐标/版本稳定排序、合并同图同坐标版本，再应用预算。近似后端可能返回不足；记录检索预算饱和，但不据此自动删除图像或宣称检索完备。
- 只允许同原图对应参与拟合。Source按不同锚点数、平均汉明距离、图像身份选最多8张，Target最多8张分别处理。每图128固定随机三点假设，方向为 **查询→正常参照**。重投影≤2，至少6个不同查询锚点内点、至少3方位，正行列式、奇异值[0.75,1.33]。
- 生成三点及验证内点的正常描述支撑经逆T回到查询外围；未参与生成的对应也须支持该解释。固定外围像素对验证，翻转率≤0.25。逐方位留出重拟合的角点变化经 M逆转换回查询像素，≤1；缺失/退化拟合明确拒绝。
- 同图角点映射差≤0.5才合并变换；不同空间位置保留。外围代价≤最佳＋0.05全部保留；超过12个近优非等价解释，主/Source多候选记搜索未决，不截断。单候选对照仍可选外围最优一项。
- 随机种子使用42、窗口坐标、正常原图身份，**不使用查询整图SHA、P、中心或标签**。正常图的全图索引预建不随查询中心变化；候选拟合/验证不使用其映射中心的差异；冻结候选后才计算正常中心预期，再计算查询中心证据。这是依赖隔离，不是说解码文件时没有读入中心字节。
- 各比较要求所有保留解释有效且一致。Source/ST多候选至少两张不同原图；同SHA副本/同图多坐标不冒充独立实例。无多数票。固定24分母；保留缺席mask、翻转数、有效比例、分歧比例和参照数。W内有效3×3均值取最大，再对全图固定窗口取最大。

未指定的接口细节已在测试指标读取前固定：查询网格用可见外围左上起的8像素**单元中心**（＋4偏移），避免描述范围恰贴边造成零插值余量；正常网格原点为整图0。8方向是整数八邻域乘1/2/4，斜向为L∞半径；“有效3×3”要求9个像素各至少一项一致比较；外围代价为固定验证对翻转率，至少存在一对真实有效验证对。原规格没有要求32对等额外门槛，因此不隐藏加入。

插值场景不具有严格光度不变性。测试只证明固定整数对应、无量化变化的严格单调变换保持位比较；不声称锚点对比度排序、近似索引或插值也必然不变。有限检索不是完整搜索证明。以上没有更换研究公式；网格相位、代价和有效块口径是原文未完全指定之处的固定实现选择。

## 分支与校准

固定输出7项 Image AUROC/AP：SEGMENT_CH1；SOURCE_MULTI_FUSION；ST_SINGLE_FUSION；预定主候选 ST_MULTI_FUSION；E_SOURCE_MULTI、E_ST_SINGLE、E_ST_MULTI 三项辅助。

同一外围匹配池服务三个E分支，不重复检索。Source66均使用同一455＋K8像素参照，P必须来自旧的Source＋K8 SEGMENT。

`S_base=(P-mu_P)/sigma_P`，`S_branch=S_base+E_branch/sigma_E`。统计ddof=0，三个融合共享主候选E的sigma_E，不中心化E。sigma_P数值退化报错；sigma_E≤1e-6固定用1并标记，不搜索尺度。SEGMENT_CH1单列保留原始P，与父指标核对至原规则1e-12，不能以标准化副本冒充原始逐图分数。

所有图保留。证据缺席时E=0并另标状态，不能写成确定正常。少量有效窗口不是缺陷保持证据。主候选无收益不能换用辅助分支宣称成功；单域单K8没有跨域稳定结论。

## 成本、存储与续跑

本地预检按真实3024×3024尺寸推导：

| 新工作 | 计划数量/未压缩规模 |
|---|---:|
| 正常原图 | 455＋8＝463 |
| 二进制描述 | 586,679,801 条，约8.74 GiB |
| 正常灰度缓存 | 约15.77 GiB |
| Source连续索引向量副本 | 约8.59 GiB |
| 66＋289查询灰度缓存 | 约12.09 GiB |
| 校准/评价窗口 | 583,176 / 2,553,604，共3,136,780 |
| 三分支逐比较缺席mask | 打包位串未压缩约107.69 GiB |
| 逐像素翻转数 | 未压缩约35.90 GiB |
| 候选变换 | 未压缩约5.05 GiB |

mask、翻转数、候选用固定64窗口NPZ块压缩；大量缺席块可压缩，但**不承诺压缩率**，也不漏存缺席mask以掩盖存储成本。OpenCV索引文件/哈希表/原生向量副本的真实RAM与磁盘为额外未知项。缓存应放机械盘；不将这些大产物放SSD或review包。峰值RAM需实测，不能依据描述矩阵大小直接保证小于60GiB。

固定4张按ID排序Source校准图做整图smoke，不能只挑好窗口。先计全部正常索引成本，再测整图检索、几何、中心读取和压缩落盘。冷索引加载与逐图灰度解码/缓存读取分开计时。`cost_estimate.json`外推剩余窗口，已完成资产不再算待办；CPU评价/IO和跨图变化未知项保留未知。若投影已知工作超过12h，写 `budget_pause.json` 并停；`--allow-over-12h` 只能在用户明确同意后使用。四图均无多候选有效比较即停，不要求正常E必须非零。

数值主要为FP32、uint8、位运算；128个小仿射假设批量求解/重投影，少量合格假设才进入支撑和留方位验证；小型几何原生FP64，66张统计FP64。这里没有适合TF32/BF16的大规模矩阵乘法，不强行调用GPU。程序最大CPU workers为8，不改base、驱动或全局代理。

复用单位：正常图灰度/描述、完整正常索引、已提交的64窗口块和已完成图。中断图仅重算未提交块；完成图不再读灰度或检索。数值输入/config/code不一致拒绝覆写旧目录，使用新run；图像身份相同的原图缓存不靠修改旧manifest来伪装来源。损坏/缺失的已提交mask块明确报错，不静默丢掉图像。未提交临时文件只在本轮独立缓存原位覆写，不删除历史资产。

完成图复用和未完成图续跑共用窗口块检查：记录字节数、NPZ解压可读性（含ZIP成员CRC）、全部必要数组的形状/dtype及窗口坐标。每进程首次复用该块读取一次；文件大小/时间/文件标识未变时只核对stat，不重复解压；变化则重新检查。完整图内存缓存也不绕过此检查。未新增全量SHA扫描；这是缓存可读性/结构检查，不承诺密码学完整性。

## 手动入口（此处未执行服务器命令）

本地默认只读盘点：

```powershell
& 'D:/anaconda/miniconda/envs/chinamoblie/python.exe' -B -m DINOv3.MADEqual.aebad_s_annular_ordinal_p0.runner preflight
& 'D:/anaconda/miniconda/envs/chinamoblie/python.exe' -B -m unittest DINOv3.tests.test_aebad_s_annular_ordinal -v
```

服务器同步且用户授权后，使用现有yyk解释器，不要求安装/修改环境；已有cv2的FLANN-LSH仍需在该机器真实验证。先运行预检，核对原盘挂载：

```bash
cd /home/ubun/YYK/DINO-Reference-AD
export RUN_ROOT=/home/ubun/YYK/runs/aebad_annular_ordinal_v1
export CACHE_ROOT=/home/ubun/YYK/storage/annular_ordinal_cache/aebad_annular_ordinal_v1
export CPU_WORKERS=8
bash DINOv3/MADEqual/aebad_s_annular_ordinal_p0/scripts/run_workstation.sh preflight
```

获准做正常索引和4张smoke后，在一个tmux会话内执行以下两步，避免后台服务丢失自定义环境变量（在该会话内重复上述export）：

```bash
tmux new-session -s annular_ordinal
# 在tmux内设置上述RUN_ROOT、CACHE_ROOT后：
bash DINOv3/MADEqual/aebad_s_annular_ordinal_p0/scripts/run_workstation.sh build-index
bash DINOv3/MADEqual/aebad_s_annular_ordinal_p0/scripts/run_workstation.sh smoke
# Ctrl+b 后按 d，回到SSH终端；无关机命令。
```

**先审阅smoke、证据覆盖、RAM、索引/块大小及成本。** 只有用户批准继续且成本门槛通过，才在同目录/同环境执行 `run_workstation.sh all`；它依次复用build-index、smoke，再calibrate、infer、evaluate、report、pack，不扩域。各阶段也可显式单独运行。未另行同意超预算时不要设置 `ALLOW_OVER_12H=1`。续跑用完全相同RUN_ROOT/CACHE_ROOT和输入参数，不新建目录伪装已完成。

日志直接显示进度，另在 `$RUN_ROOT/logs/`；可 `tail -n 30 -F "$RUN_ROOT/logs/smoke.log"`。最终报告包括固定7分支、指标差、事后覆盖和排序对；`fixed_cases/`只选第一校准图和前两评价图的首/中间窗口，不按得分选例。review包只含小表/报告/固定案例，灰度、索引及全部窗口mask留在独立cache。

## 本轮核对代码（全部列出）

1. 原有小JSON封存、父名单/身份、Source划分、支持排除及Source＋K8 P合同；基线P和指标一致性。
2. 新图首次真正解码时沿用原Data的编码图像身份检查；新数组沿用一次写出身份。已有大数组不反复SHA重读，快速读取只查manifest和数组头。
3. 方法要求的边界/插值支撑、几何、候选数、不同原图、一致性mask及有限值检查；校准退化处理。
4. 新run/cache与父资产路径分开、不可变已完成记录；复用NPZ检查大小、可读性、必要数组结构与窗口坐标，未变文件不重复解压；不清理父目录。
5. 用户要求的12h投影暂停与4张正常smoke证据门槛。无额外磁盘/RAM准入门槛、无测试指标门槛、无自动安装/模型提取/LP回退。
