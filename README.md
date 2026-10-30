# DINO-Reference-AD

AeBAD-S / RobustAD 第二章真实扰动实验的代码同步仓库。

本仓库包含 Source/Target-normal few-shot、Union、SEGMENT、PAIR、SPARC、FastRef、初始投影及其诊断工具，并保留它们实际调用的共享宿主、评分、评价和测试代码。依赖闭包内少量文件沿用旧目录名；它们是共享依赖，不代表将对应独立实验整套纳入本仓库。

## 范围

逐文件列表见 [UPLOAD_MANIFEST.md](UPLOAD_MANIFEST.md)。只包含第二章实现、必要依赖、专项测试及各工具的运行说明。

数据集、图片、权重、Source memory、MAD 结果文件、支持特征、距离缓存、实验结果包、实验报告和本机环境盘点不在 Git 中。工具内的冻结 catalog/source_scope 是代码所需的小型协议清单：包含样本 ID、支持顺序、哈希和历史基线分数，不含图像或特征。

历史输入仍由用户保存在本地或服务器上。新的 clone 不等于已经获得完整实验资产，不会自动提取特征、训练模型或重建 memory。

## 当前两域验证工具

实现及完整阶段说明：

* [illumination/view 冻结 SEGMENT 与 PAIR](DINOv3/MADEqual/aebad_s_segment_pair_crossdomain_p0/README.md)
* [实现审计](DINOv3/MADEqual/aebad_s_segment_pair_crossdomain_p0/implementation_audit.md)

默认入口只读预检，不导入 Torch、不初始化 CUDA：

```bash
python -B -m DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.runner preflight --no-write
```

DINOv3 官方源码随仓库放在 `external_baselines/dinov3`，是本工具默认查找位置；不需要另行 clone 或安装该源码包。checkpoint、正式六项元数据、冻结 Source image MAD、原 Source memory 和数据集仍由用户提供。使用对应的 `--data / --formal-dir / --image-state / --memory / --checkpoint / --dinov3` 参数指向已有资产。固定预检明确报告缺项；不能因缺项而重建冻结资产或调整参数。

Kaggle 更新代码后，官方源码路径为 `/kaggle/working/DINO-Reference-AD/external_baselines/dinov3`。此次同步仅提供源码，没有安装依赖、加载模型或执行实验；T4 显存适配和其他输入缺项仍需另行处理。

后续顺序为：显式补缓存 → 冻结预测 → CPU 评价 → 独立原 CUDA/ADEval AUPRO。补缓存需要完整宿主所需的显存；本地能够做 CUDA 评价不等于能够运行完整模型。各步骤命令以工具 README 为准。

## RobustAD MetalParts / PCB 冻结跨数据集验证

新增工具：[运行说明](DINOv3/MADEqual/robustad_segment_pair_crossdataset_p0/README.md)、[实现审计](DINOv3/MADEqual/robustad_segment_pair_crossdataset_p0/implementation_audit.md)。冻结 UNION / SEGMENT / PAIR 三分支，包含23张Source校准恢复、逐类别/shift捕获、预测、CPU全像素评价、独立CUDA AUPRO、Source参照/ARD和分开打包。

总计11个shift、1722张Target评价图、88张固定支持；不包含PiledBags。默认preflight不导入Torch、不初始化CUDA、不生成实验输出。原Source元数据、patch MAD、图片与权重由用户在运行端提供；没有自动构库回退。原bank丢失时，经用户授权可用显式rebuild-source按原入口重建到独立根目录，新身份及23张校准/历史分数检查与旧资产分开，详见模块README。已有366张Source测试缓存的本地候选发现不代表服务器已挂载或通过原bank身份检查。

本轮仅完成本地实现、CPU专项与相关回归检查；真实AutoDL模型/CUDA/指标尚未执行。完整运行示例可启用成功后关机，仅全部指标、ARD和review包成功后关机，任何失败保留现场。

## AeBAD-S 固定 K8 支持集敏感性

新增独立工具：[运行与续跑说明](DINOv3/MADEqual/aebad_s_k8_support_sensitivity_p0/README.md)、[实现审计](DINOv3/MADEqual/aebad_s_k8_support_sensitivity_p0/implementation_audit.md)。固定 background / illumination / view 三域、支持 seed42/43/44，比较 UNION / SEGMENT / PAIR 共27个组合。所有支持组在每域使用同一共同评价子集，共878张；Source memory 和校准保持冻结。

提供 preflight / prepare / capture / predict / evaluate / evaluate-aupro / summarize。默认预检只读、不导入Torch；共享查询提取和Source检索，支持逐图及评价组合续跑。新增29项本地CPU专项通过，真实模型、CUDA及本次27组合正式实验尚未执行。本机资产盘点和所有实验产物不上传。

## 代码同步

### AeBAD-S illumination / view 完整重跑与逐支持诊断

入口与阶段说明：[跨域诊断 README](DINOv3/MADEqual/aebad_s_support_distance_crossdomain/README.md)。Kaggle 可直接 clone 后调用 `DINOv3.MADEqual.aebad_s_support_distance_crossdomain.kaggle.main(PROJECT)`：环境/Input核对 → 629张capture → 预测 → 精确CPU评价（12GiB预算） → 原CUDA AUPRO → 逐支持距离诊断 → 汇总与raw两个自动结果包。每30秒心跳，失败保留现场；仍需用户显式启动，无自动服务器连接。

该工具的 `background_discovery_catalog.json` 是汇总需要的小型冻结参照，只含预设主描述量和覆盖/实例分组统计及原文件SHA，不含完整background结果、报告、图片或距离缓存。统计内核版本与catalog SHA严格核验，不把其内容当作新域验证结果。其余已完成产物、数据和模型仍不随代码上传。

单卡/双卡显式指定`capture_gpus=1/2`，已发布默认值不改。按用户最新要求，Kaggle/resume入口的资源数值仅观测，不按RAM、VRAM、磁盘剩余量或估算预算阻断；保留真实分配异常、数值与缓存身份校验。已完成629张捕获/预测时可用独立resume入口从保存版本Output继续CPU评价、原AUPRO及跨域诊断，不重跑模型；命令见跨域诊断README。

在服务器建立独立代码目录，由用户执行：

```bash
git clone https://github.com/YYK001/DINO-Reference-AD.git /root/autodl-tmp/DINO-Reference-AD
cd /root/autodl-tmp/DINO-Reference-AD
```

以后更新：

```bash
cd /root/autodl-tmp/DINO-Reference-AD
git pull --ff-only
```

已有实验资产继续保留在原目录，用显式路径传入，不移动或覆盖正式结果。此仓库由实验工作区筛选同步产生；本地发布副本与完整实验工作区分开。Git 属性禁止自动转换源码换行，保留原字节哈希和冻结身份。

AutoDL 后续统一用上述独立 clone / `git pull --ff-only` 更新代码，不在旧的 DINO-NVS 资产目录强行初始化或覆盖仓库。RobustAD 的 PROJECT_ROOT 指向代码目录；SOURCE_ROOT、FORMAL_METALPARTS、FORMAL_PCB、CHECKPOINT 仍显式指向原 DINO-NVS 资产路径，OUTPUT_ROOT 使用新的独立实验目录，完整示例见该工具 README。

## 测试与依赖

所有生产源码可做不加载模型的 AST 解析。相关专项测试位于 `DINOv3/tests/`；其中使用历史归档、原图、正式清单的集成测试需要另行提供对应资产，不能将缺失资产的全测试发现当成纯单元测试命令。

本次实现已在原工作区通过30项新专项及24项旧 neighborhood 回归，固定12张 background 的 PAIR 距离和门控逐元素一致，CH1 分数最大误差为0。模型、提取、Exact 和 AUPRO 集成替身不代表真实两域实验已完成。

运行环境需自行准备 NumPy/SciPy/Pillow/scikit-learn/psutil；显式模型阶段还需匹配的 Torch/torchvision 和原宿主依赖。原 AUPRO 使用 [adeval==1.1.0](DINOv3/relation_reliability/requirements-lock.txt)，不可用时保留 pending，不自动安装或更换指标后端。

## 第三方源码

`external_baselines/SuperADD` 为必要宿主源码，其 [LICENSE](external_baselines/SuperADD/LICENSE)、[NOTICE](external_baselines/SuperADD/NOTICE)、[CITATION.cff](external_baselines/SuperADD/CITATION.cff) 和第三方声明保留原内容；适用原 CC BY-NC 4.0 等声明。

`external_baselines/dinov3` 来自 [Meta 官方仓库](https://github.com/facebookresearch/dinov3)，保留原 Python 包、配置和附属源文件、hub 入口、依赖/构建清单、README、[LICENSE.md](external_baselines/dinov3/LICENSE.md) 与 [MODEL_CARD.md](external_baselines/dinov3/MODEL_CARD.md)。这些文件未经修改，适用 DINOv3 License Agreement；没有打包权重、上游 Git 历史、Notebook 和 CI。上游 commit 与逐文件 SHA-256 记录于 [UPSTREAM.json](external_baselines/dinov3/UPSTREAM.json)。该 commit 标识本次本地源码来源，尚未证明它与历史 AutoDL 提取时使用的源码版本完全相同；正式提取仍须通过冻结基线复核。

## 跨支持图佐证的局部参照扩展：固定P0

[运行说明](DINOv3/MADEqual/aebad_s_support_witness_segment_p0/README.md)与[实现审计](DINOv3/MADEqual/aebad_s_support_witness_segment_p0/implementation_audit.md)。冻结三域原K8与926张名单，比较UNION、SEGMENT、第三图佐证连接与等量均匀缩距。独立支持间k5、整图留一、缓存预测、CPU精确评价与6项原CUDA AUPRO，不修改旧分支列表。

显式`recover-support`只恢复丢失的24张原正常支持特征；`verify-support`核对所有原缓存SEGMENT和端点长度，再允许带新来源身份的支持接入。历史特征SHA保持不变，不能声称新银行与缺失Q的原检索逐位相同。`portable`传输冻结父输入，`autodl`按用户指令串联完整流程，明确单/双卡、30秒心跳、中断续跑及成功后可选关机；失败保留现场。代码、CPU合成测试及输入可迁移性已验证，真实模型/CUDA/完整实验仍须用户显式启动。冻结资产和本机验证日志不在GitHub。

### 支持恢复的跨环境误差审计

若WITNESS恢复在旧缓存兼容检查停止，可显式运行独立CPU审计：

```bash
python -B -u -m DINOv3.MADEqual.aebad_s_witness_recovery_audit --run-root /actual/witness_run --data /actual/AeBAD_S --cpu-workers 8
```

默认完整illumination/view两域629张、四层，不重提特征、不读取GT、不调用Torch或GPU。只观察原始距离、端点长度、Source MAD尺度误差、原定位图及CH1/CURRENT分数变化；严格失败继续保留，不把审计完成写成实验通过。相同旧缓存的地图SHA和分数仍严格复核。结果写到独立drift_audit目录，冻结WITNESS源码和原恢复seal保持不变。六项合成CPU测试通过，不代表真实跨环境审计已通过。

### 复用已封存支持的历史恢复兼容运行

WITNESS默认strict不变；用户显式选择`--reuse-recovery <旧支持提取目录> --recovery-policy historical-maps-v1`可在新的运行目录生成独立CPU兼容证书，复用24份特征而不重提。支持/图片/精度/配置及文件SHA严格核验，原缓存地图SHA与评分严格复核；恢复几何的各层等权MAD贡献和融合地图差<=1e-4，CH1/CURRENT差<=原1e-5。原逐点距离失败继续保留，不改写为严格复现，不更改检测公式或历史地图。完整复核后才执行正式支持检索/留一/预测/评价。详见[历史兼容政策与完整命令](DINOv3/MADEqual/aebad_s_support_witness_segment_p0/README.md)。本地72项CPU回归通过，真实兼容证书、GPU支持检索和新分支指标仍由用户执行。

## WITNESS批量GPU执行与设备迁移

[当前执行和检查清单](DINOv3/MADEqual/aebad_s_support_witness_segment_p0/execution_optimization.md)：近邻检索、整图留一和预测支持显式单/双GPU，CPU冻结/评价支持指定worker；公式和评价不变。93项本地CPU测试通过，包含实际Torch CPU内核；真实CUDA、多卡计算与GPU加速比尚待验证。

代码可在其他Linux/Windows设备克隆，AutoDL入口只是可选编排工具。数据、权重、恢复支持和冻结输入包不在GitHub中，需要单独迁移；在新设备准备环境并显式传入路径，历史正式结果不覆盖。


## WITNESS从原图完整重跑

没有旧缓存、Source bank或MAD的新设备使用[独立fresh入口](DINOv3/MADEqual/aebad_s_support_witness_segment_p0/README_fresh.md)。从AeBAD原图及官方权重重建521/66 Source划分、100K memory、两级MAD、24支持和926查询；GPU批量WITNESS及四分支完整评价。单卡、最多8个CPU worker、tmux、HDD大缓存、成功/失败review打包；不自动关机。真实DINOv3/Source采样/CUDA仍待运行，CPU轻量验证不能冒充正式通过。

## 固定Target局部图与第三方正常佐证

[运行说明](DINOv3/MADEqual/aebad_s_fixed_target_graph_p0/README.md)与[实现审计](DINOv3/MADEqual/aebad_s_fixed_target_graph_p0/implementation_audit.md)。独立fresh入口保留原Source/两级MAD、seed42三域926张，正常整图留一选择固定互近邻图与完整第三方佐证，比较七分支及两种缩距控制。显式支持复用当前机器已封存fresh资产，不依赖旧AutoDL缓存；单卡、最多8个CPU worker、stream/disk查询缓存、tmux和分阶段续跑，无容量准入门槛或自动关机。

38项新专项与42项原实现回归共80项本地CPU测试通过；真实DINOv3提取、CUDA、真实规模图与ADEval GPU尚未验证。测试含骨干/银行替身，不代表正式实验成功。额外seed43/44须显式使用878张共同名单，不与926主表混用。

优化版将完整佐证的筛选、计数和直方图保留在GPU，LOO配置分批归约，同层Top100表九折只读一次；公式、FP64和完整搜索不变。新增逐折边速/耗时日志，CUDA加速倍数尚未实测。更新后使用新目录重跑，命令见工具README，不迁移旧图缓存。

## 固定图预测的混合精度续跑

[独立加速入口](DINOv3/MADEqual/aebad_s_fixed_graph_acceleration_p0/README.md)复用上述父运行已冻结的Source、支持、最终图、正常LOO选择及已完成逐图预测，不重做构图。FP32端点初筛配合保守误差区间、全局两遍筛选与FP64直接复核；压紧实际启用边及端点，胜边仍对应父图原编号。原骨干、离散NN、七分支公式与评价不变。使用独立结果/缓存目录，父产物只读。

18项加速专项与101项相关旧回归通过，实际检索在Torch CPU上验证；真实4090速度、CUDA数值核对和完整预测尚待显式smoke及少量完整图像测速。review按报告文件类型打包，并剪除评价scratch目录，失败残留BIN不会进入结果包。不宣称FP16骨干或已测得加速倍数，无容量前置门槛，不自动关机。

## 固定图的单遍FP32融合预测

[FP32独立入口与运行说明](DINOv3/MADEqual/aebad_s_fixed_graph_fp32_p0/README.md)复用原Source、MAD、支持、最终图和正常LOO选择，单遍访问合法边，使用宽FP32 GEMM和Triton融合块内筛选/投影/归约，减少Python逐块派发。FP32逐图预测独立生成，旧FP64预测仅作数值参考；原SEGMENT对照和评价精度不变。smoke核对后用`bench --limit 3`测完整图像，不默认据局部测试宣称提速。

18项新专项＋56项相关回归共74项CPU测试通过。本地没有Triton，CUDA融合路径尚未实际编译、执行或测速；需要服务器验证。默认预检不导入GPU库，不自动安装依赖，无容量门槛和自动关机。

## Source–Target正常对应残差适配

[独立模块及工作站命令](DINOv3/MADEqual/aebad_s_normal_correspondence_adaptation/README.md)：三域固定K8与926张，SS/ST/TT正常伪对应、三种子残差网络、完整库逐行变换、每模型Source-only校准及固定z集成。仅接入迁移后fresh WITNESS和固定图两轮指定资产；无旧AutoDL依赖，缺失正常特征按原宿主补提取，不重跑图/WITNESS。单GPU、8 CPU worker上限、tmux、分阶段续跑及精确CPU/ADEval评价，不使用自定义Triton，不自动关机。

包含评价scratch隔离修复：保护输入/父运行/cache，临时目录独立，分配失败只清理本次创建的文件。23项CPU测试中22项通过、1项因Windows软链接权限跳过；14份Python编译和Bash语法通过，真实DINO/CUDA/BF16训练及ADEval仍需正式运行验证。默认入口只读preflight，不安装环境或进行资源容量准入。


## Target-normal参照残差分割学习与工程smoke

[独立模块、完整公式与单卡工作站命令](DINOv3/MADEqual/aebad_s_reference_residual_segmentation/README.md)：固定三域K8与926张，正常RGB固定合成池、严格同源排除、Source预训练及三族分割头、三种子z集成、独立logit校准和原精确评价。复用兼容的原始Source库；父距离MAD缺失可走本轮原66张校准，损坏父记录仍拒绝。

执行优化包含有界参照显存驻留、每查询层共享上传、训练按索引直接取参照行和同尺寸tile合批，保留原Exact及逐图损失。56项不同CPU测试通过；真实CUDA/BF16及全量结果尚未验证。提供constructed/assets/prepared独立smoke，记录数值、传输、训练/IO及成本外推；不把smoke写成正式完成。单GPU、最多8 CPU，无容量准入、自动安装、自定义Triton或关机。

## SACP：Source内部差异对照的条件化残差子空间修正

[固定公式、资产要求及单卡工作站命令](DINOv3/MADEqual/aebad_s_sacp_p0/README.md)与[实现审计](DINOv3/MADEqual/aebad_s_sacp_p0/implementation_audit.md)。复用迁移后原Source及A/B折库、固定K8和926张名单，图像均衡的未中心化二阶矩、共同低秩表示、广义特征对比与欧氏QR方向；比较UNION、SEGMENT、SACP及Target/Source方向、等量均匀缩距六张定位图。冻结同一Source patch/image MAD、CH1主评分和CURRENT补充，18项逻辑AUPRO，兼容基线可复用。

默认仅preflight；缺折库阻断，不自动构库。若父stream查询未另存完整Q，926张须重新前向，每图一次供全部分支共享。无合成、梯度训练、自定义Triton、全量缓存SHA扫描、容量准入或自动关机。26项CPU测试通过，真实CUDA/原宿主926张复现及ADEval尚未验证；本机盘点与正式结果不在仓库中。

## SACP固定强度扩展：lambda=1

[缓存重放、输入清单及工作站命令](DINOv3/MADEqual/aebad_s_sacp_strength_p0/README.md)与[实现审计](DINOv3/MADEqual/aebad_s_sacp_strength_p0/implementation_audit.md)。父lambda=.5运行已在工作站完成；本扩展只消费父raw_union和eta前两轴，新增主方法、Target方向及等量均匀缩距三个L1分支。先按原容差重放核对.5，原MAD/位置/插值/CH1和CURRENT不变，仅9项新CPU像素评价与9项新AUPRO，历史结果只读注明来源。主表72行，完整附表80行，无Source方向L1或其他参数搜索。

默认只读preflight，CPU路径禁止Torch导入；不需要Q、bank、折库、子空间向量、checkpoint或模型。单GPU仅用于显式原ADEval阶段，最多8个CPU工人，逐图/域/分支续跑，私有scratch和review打包排除大数组，不自动安装或关机。16项人工缓存/GT的CPU测试通过；真实缓存重放及L1指标仍待执行。本机预检、日志、数据和父缓存不上传。

## 5×5块盲点上下文正常参照合成

[固定background工具、资产要求和工作站命令](DINOv3/MADEqual/aebad_s_blind_context_p0/README.md)及[实现审计](DINOv3/MADEqual/aebad_s_blind_context_p0/implementation_audit.md)。原seed42 K8、297张；在预处理后、切窗前做四相5×5块输入遮挡，64张Source辅助字典的值来自正式100K库，Target保留完整K8正常键和值。仅遮挡上下文选Source Top4及每支持Top1，32776参数注意力生成真实正常值凸组合，整图排除训练1500步，最终min原UNION距离；不改原MAD/CH1/CURRENT与精确评价。

默认preflight不导入Torch；仅新增一个定位图和一项原ADEval AUPRO。完整预测重复执行保留原冻结记录及seal，会话计数独立记录，部分续跑只补缺失图。29项CPU夹具测试通过，包含续跑修复；真实骨干隔离、CUDA、吞吐、训练与五指标仍pending。单GPU、最多8 CPU，不安装依赖、自定义Triton、容量准入或自动关机；数据、权重、预检、日志与结果不上传。

修复原宿主1890像素预处理输入的遮挡接入：保持原输入和floor16切窗，不再要求完整输入为16倍数；有效1888像素范围内构造遮挡，原宿主未消费的尾部像素原样保留。隔离覆盖右下角，原始裁块/拼接与CPU骨干替身共同回归验证；已有失败目录保留，新建目录重试，不重建Source或改变冻结检测器。

## 同源邻域差分检索：固定background验证

[方法、资源估计和工作站命令](DINOv3/MADEqual/aebad_s_same_origin_neighbor_p0/README.md)及[实现审计](DINOv3/MADEqual/aebad_s_same_origin_neighbor_p0/implementation_audit.md)。原seed42 K8与297图，455/66 Source划分及每层100K；比较UNION、SEGMENT、固定评分/定位组合、绝对邻居C1、相对差分C1及绝对Top16候选内的相对差分联合选择。无训练、参数搜索或扩域。

优先沿原导入来源链恢复选中索引、实际增强及完整物理邻域；中心不吻合明确停止，仅显式选择才独立重建原Source并重评基线。新检索FP32、单GPU、最多8 CPU，原精确CPU/ADEval评价、分阶段续跑、12小时实测预算暂停与小报告打包。特征目录、提取配置和图像/提供者manifest绑定到运行身份，拒绝续跑混用；不添加容量准入、重复大文件SHA扫描、自定义Triton或关机。33项CPU测试通过，真实来源恢复、CUDA性能与完整指标尚待用户执行。


## Target-LOO正常统计层加权

[固定公式、资产与工作站入口](DINOv3/MADEqual/aebad_s_target_loo_layer_weight_p0/README.md)，[实现审计](DINOv3/MADEqual/aebad_s_target_loo_layer_weight_p0/implementation_audit.md)。background seed42 K8、297张，冻结原UNION检索与Source MAD，比较Source单侧正偏离新加权和Source/Target正常整图留一混合加权（gamma=0/.5）。两种新图像读出、两项新增原ADEval AUPRO，兼容原基线只读复用。默认preflight，无训练或搜索，不自动构库或提取；显式补正常检索、阶段续跑、单GPU、12h预算暂停。19项CPU测试通过，含FP32权重求和舍入修复；真实检索与完整GPU评价尚待运行。本机预检、结果、缓存和权重不上传。


## Target-LOO固定跨域验证

[illumination/view扩展、资源估计与单卡手动入口](DINOv3/MADEqual/aebad_s_target_loo_layer_weight_p0/CROSSDOMAIN.md)。完整域参数贯穿父资产读取，265/364图、每域原seed42 K8；只读导入background Source66统计，检查共同依赖及gamma0权重一致，不重跑background。每域七分支、两张新地图和两项AUPRO，固定SEGMENT评分/加权定位组合不增加推理或像素作业。汇总含逐域、两域及三域等权指标；background组合标为开发后派生。保留原20项CPU测试，共32项通过（代码仓库无本地background包时相关真实数据回归明确skip）。默认preflight、显式补检索、两域合计12小时预算、无自动构库/安装/关机。本机预检和实验数据不上传。


## Target-LOO层加权：两组K8支持复验

[协议、资产复用与单卡运行说明](DINOv3/MADEqual/aebad_s_target_loo_support_retest_p0/README.md)及[实现审计](DINOv3/MADEqual/aebad_s_target_loo_support_retest_p0/implementation_audit.md)。固定三域、support_seed42/43，Source seed42与455/66划分不变；共同名单289/257/356，共902张唯一图像。seed42复用统计、权重和预测，seed43仅显式补缺失完整特征/Exact/原SEGMENT；四图共享同一查询，不训练或搜索。

共同名单上的五指标全部重评，六设置最多24项独立地图/AUPRO；七分支保留两种读出，同seed组合不增加地图。逐设置、每seed三域及两seed等权汇总，保留支持敏感性比较。默认CPU preflight、按阶段续跑、12h预算暂停、单GPU与最多8 CPU；不重建Source、不重复扫描大数组SHA、无自动安装/关机或自定义Triton。56项本地CPU测试通过，包含父查询sample/identity格式的零前向缓存复用修复；真实CUDA和完整复验待用户执行。


## Target-LOO权重交叉与可分性审计

[离线输入、缓存导出及运行说明](DINOv3/MADEqual/aebad_s_loo_weight_separability_audit/README.md)与[实现审计](DINOv3/MADEqual/aebad_s_loo_weight_separability_audit/implementation_audit.md)。冻结三域902张及两套K8 memory，用WS/W42/W43构造2×3诊断矩阵；只新增六个交叉地图和六项原ADEval AUPRO，原12格指标只读复用。交叉引入另一K8统计，明确标记16张Target正常信息，不作为同信息K8方法成绩。

复用原FP32加权、插值、精确CPU评价与空间实例/环带规则，全部无GT响应冻结后才开展图像对精确并列、p/q间隔贡献、固定WS位置margin和正常Top1%审计。保留峰值移动、NA及连续面积，不拟合新权重、训练、提特征、做Exact或搜索参数。默认CPU preflight；仅需四层标量UNION与positions，不要求模型/Source bank实体；提供最小只读导出、smoke、按阶段续跑和12h预算暂停。72项本地CPU测试通过（新增16、回归56），真实交叉缓存重放及GPU AUPRO尚待执行。无自动安装、容量准入、自定义Triton或关机。
