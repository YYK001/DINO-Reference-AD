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

## 代码同步

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
