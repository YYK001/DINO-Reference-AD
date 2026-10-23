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
