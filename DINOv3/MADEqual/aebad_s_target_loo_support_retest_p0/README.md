# Target-LOO层加权：固定两组K8复验

本工具仅验证 background、illumination、view 的支持seed42/43。复用已验证的层加权数学内核、Exact、原SEGMENT、CH1、全像素CPU评价和ADEval1.1.0，不执行训练或参数搜索。默认 `preflight`，不会导入Torch、初始化CUDA、提特征或检索。

## 冻结名单与计算任务

Source随机种子始终42，Source455训练/66校准、正式每层100K bank及MAD不变。support_seed只改变本组8张Target支持；LOO整图排除本组一张，仅检索本组其余7张和Source。正式查询使用本组K8，不合并两组支持。

seed42严格继承当前名单；seed43在完整正常池上调用原 `stable_support_key(seed=43)`，已与历史三支持组review的路径、sample_id、图像身份和顺序核对。当前两组恰好无重合，代码允许自然重合。

|域|完整测试图|共同评价|正常|异常|42/43支持重合|旧44支持回归评价|
|---|---:|---:|---:|---:|---:|---:|
|background|305|289|77|212|0|8|
|illumination|273|257|59|198|0|8|
|view|372|356|76|280|0|8|

共同评价集为完整测试集减去T42与T43的并集，共902张唯一图像、1804个重复预测实例。保留原sample_id；两seed使用相同顺序和GT。不能将其称为1804张独立图像，也不能沿用历史926/878张指标。

每设置四张独立地图：UNION、SEGMENT、Source加权γ=0、Target-LOO加权γ=0.5。三域×两seed共24项定位图CPU/AUPRO评价。原/组合分支CH1和CURRENT；独立加权分支PEAK_LOCAL_WEIGHT与WEIGHTED_MAP_MAX。三个固定组合使用同seed SEGMENT图像分数与指定地图，无额外地图或AUPRO。

共享 `aebad_s_target_loo_layer_weight_p0.numerics` 的u/p/q、Source中位数锚点、正偏离二阶矩、1e-8下限、FP32加权/插值及数值容差，不另写数学近似。Sourceγ=0权重跨六设置一致，地图仍依赖各自K8。

## 当前资产盘点

本地预检输出：`DINOv3/outputs/mad_equal/aebad_target_loo_two_K8_preflight_20261029/`。

- `common_manifest.json`、`support_manifest.csv`、`common_evaluation_manifest.csv`、`excluded_images.csv`、`population.csv`：已实际冻结的名单。
- `preflight.json`、`asset_inventory.csv`、`missing_inputs.csv`：逐图资产盘点及计划估时。
- 本地有三域seed42完成结果、小型正常统计/权重和原父元数据，可只读复用。
- 本地未找到902份seed42父响应实体、902份兼容seed43基线/查询特征、24份seed43支持特征、24份seed43 K7 LOO距离及原Source bank实体。`datasets/AeBAD.zip` 存在；逐图GT映射在实际评价时复核。评价缺项按六设置列出，每设置四地图。
- 此为本机盘点，不代表服务器上的这些文件不存在；没有进行SSH检查。

旧 `aebad_s_k8_support_sensitivity_20261020_132812_review.tar.gz` 有42个归档条目、零NPZ。可核对42/43支持名单，但没有可直接读取的查询预测数组。旧`.pt` memory SHA不能单独证明与迁移后的分层Source bank等价，因此其原生seed43缓存列为“Source/提取/MAD兼容性未确认”，未冒充当前结果，也未运行旧三seed/PAIR流程。

seed42：复用66+8正常统计、原权重、共同图上的UNION/SEGMENT响应、图像分数；加权完整图缓存可选，缺失时从层响应离线重建。逐图地图SHA与分数必须对应原冻结记录，不重新检索、拟合或校准。

seed43：优先使用经证明兼容的紧凑基线和完整特征；缺项才通过显式参数补算24张支持LOO、最多902张查询的Exact/原SEGMENT。同一查询最多新增一次完整RGB前向供四图共享，不默认持久化全部查询1280维特征。Source不存在时报告缺失，工具没有重建入口。

旧指标不能复用：共同评价名单变化后，图像排序、像素排序、FPR及AUPRO均须重评。只有**本轮相同名单/地图/GT/配置**的已完成记录才能续用。

## 阶段和手动执行

现有服务器路径是脚本默认值，使用现有yyk环境；不安装依赖、不修改base、不关机。单物理卡，CPU最多8。运行根与cache及scratch三者隔离；cache位于机械盘，评价scratch位于YYK固态临时目录。

以下为人工命令，本轮未执行服务器操作。运行前先从GitHub同步代码。命令中的数据、权重、父运行保持现有路径；实际是否齐全以服务器preflight为准。

```bash
cd /home/ubun/YYK/DINO-Reference-AD
script=DINOv3/MADEqual/aebad_s_target_loo_support_retest_p0/scripts/run_workstation.sh
export RUN_ROOT=/home/ubun/YYK/runs/aebad_target_loo_two_K8_v1
bash "$script" preflight
```

预检退出0只表示盘点完成，不表示全部实体可用。查看 `$RUN_ROOT/missing_inputs.csv`。`baseline42_parent_layers` 必须先定位原父cache，不能改用seed43或重新拼接旧指标。原Source和seed42父cache完整、其余缺项符合计划后，用户可执行：

```bash
cd /home/ubun/YYK/DINO-Reference-AD
export RUN_ROOT=/home/ubun/YYK/runs/aebad_target_loo_two_K8_v1
export GPU_INDEX=1
tmux new-session -d -s two_K8 \
  "cd /home/ubun/YYK/DINO-Reference-AD && env GPU_INDEX=1 RUN_ROOT='$RUN_ROOT' bash DINOv3/MADEqual/aebad_s_target_loo_support_retest_p0/scripts/run_workstation.sh all-settings --supplement-retrieval --extract-missing > '${RUN_ROOT}_launcher.log' 2>&1; printf '%s\\n' \$? > '${RUN_ROOT}.exitcode'"
tail -n 40 -F "${RUN_ROOT}_launcher.log"
```

`--supplement-retrieval` 授权缺失Exact/SEGMENT补算，`--extract-missing` 另授权缺完整特征时用原骨干补前向。完整缓存会跳过这些计算。没有两参数时，缺相应缓存会明确停止。六设置按域/seed逐一运行；每阶段独立进程释放模型和参照。单设置/单阶段续跑：

```bash
export RUN_ROOT=/home/ubun/YYK/runs/aebad_target_loo_two_K8_v1
export GPU_INDEX=1 DOMAIN=view SUPPORT_SEED=43
script=DINOv3/MADEqual/aebad_s_target_loo_support_retest_p0/scripts/run_workstation.sh
bash "$script" prepare-normal-stats --supplement-retrieval --extract-missing
bash "$script" fit-weights
bash "$script" predict --supplement-retrieval --extract-missing
bash "$script" evaluate
bash "$script" aupro
bash "$script" summarize
bash "$script" aggregate
bash "$script" pack
```

也可在原RUN_ROOT重新执行 `all-settings`，复用已提交的逐图预测、LOO、CPU和独立AUPRO完成记录。AUPRO全完成的重放无需导入Torch；真实计算保持ADEval1.1.0、200阈值、FPR≤0.3。中断未提交的scratch会重算；不会接受无封存的中间像素指标。不要对同一RUN_ROOT并发启动两个写入进程。

脚本覆盖变量：`PARENT_RUN/PARENT_CACHE`（当前SACP）、`RAW_RUN/RAW_CACHE`（原始完整特征）、`WEIGHT42_BACKGROUND_RUN/CACHE`、`WEIGHT42_CROSS_RUN/CACHE`、`CACHE_ROOT`、`EVALUATION_SCRATCH`、`DATA`、`CHECKPOINT`、`PYTHON`。所有路径绑定记录，续跑不静默切换提供者。

## 兼容资产接入

`FEATURE_MANIFEST` 可指向封存JSON，包含原完整 `extraction` 记录和 `normal_features`（或 `records`）字典，键为原relative_path，值为 `{path,seal}`。每个bundle必须包含FP32 `q7/q15/q23/q31`、int64 `positions`，metadata有原image/grid/kind；提取配置、图像身份和完整物理网格必须一致。不可用一个最终融合图逆推出八支持检索。

父 `assets['queries']` 使用生产查询格式：`metadata.sample/identity/grid`，引用中的 `parent_identity` 绑定原特征提供者。本工具用独立的查询读取分支核对sample、提供者、seal及四层/positions，不要求查询缓存具有正常图专用的 `metadata.image` 或 `kind=original_complete_grid`；不重写旧记录，不因这一格式差异重复前向。正常图缓存仍使用原读取规则。

`LEGACY_PROVIDER` 可指向**已做独立兼容转换/核对后的导出描述**，`records`键为 `domain/seed43/sample_id`，值 `{path,seal}`。引用bundle包含本工具所需UNION_K8、SEGMENT_K8的四层标准化响应与positions/grid/output_shape，metadata.dependencies必须与 `inputs.baseline_dependencies(a,row)` 完全相同（Source42内容、原提取、MAD、支持顺序、图像、距离/SEGMENT定义）。封存使用原项目 `storage.sealed_json/put`。

该接口不是给旧原生NPZ换身份：未经Source/特征/MAD/几何等价核对的旧记录不能直接套描述导入。若转换，另存派生产物并保留原文件和来源证明，不能改写旧manifest/SHA。本轮未实现任何自动旧库等价认证，也没有因旧memory缺失重建它。

## 时间、磁盘和12小时上限

计划范围约 **1–8小时**，条件是原Source bank及seed42父响应仍在当前机器。它是依据同机此前约3–7秒/查询的粗估，不是本轮实测或完成承诺：

|部分|待办规模|暂估|
|---|---|---|
|正常特征/LOO|最多24支持前向、24 K7查询；Source66只读复用|0.1–0.8h|
|seed43查询|最多902完整查询前向及四层Exact/SEGMENT|0.5–4.5h|
|seed42重放与共同评价|1804查询实例、24张图类的精确CPU/AUPRO|0.2–1.5h|
|其他IO及汇总|依机器与缓存状态|0.2–1.2h|

若兼容seed43缓存齐全，按已复用单位减少实际工作；缺件或吞吐未测时保留不确定性。预计超过12小时须先取得用户同意。运行中按已记录耗时及当前阶段吞吐在提交单位后检查预算，超限保留断点并写 `budget_pause.json`；不得自动缩小评价名单。`--allow-over-12h` 只用于用户明确批准后。

新增cache约8–20GiB（24支持完整特征、紧凑距离/索引/几何、正常统计），不复制Source或五份查询特征。评价scratch原始标签/四地图17字节/像素，单域另有精确排序工作区，暂估3–12GiB；顺序处理六设置并只删除本轮独占临时目录。此为增量规划，最终记录实际新数组字节、耗时、进程峰值RSS、CUDA分配/保留峰值；未测资源写未知。未增加RAM/VRAM/磁盘可用量准入门槛。

## 输出与结论边界

每 `domain/seedN`：`normal_statistics.json`、`weights.json/csv`、`u_concentration.csv`、逐图预测摘要、冻结记录、CPU五表/像素审计、四个独立AUPRO记录、七分支两读出的14行指标、60项固定比较、报告与costs。

`summary/`：84行逐域seed指标、28行每seed三域宏平均、42行每域两seed平均、14行最终宏平均、720行固定比较、30行Target额外信息跨seed差异、逐层权重及变化/统计集中度、成本和完成状态。汇总先域等权再seed等权，绝不池化重复图像。只有六设置、24项AUPRO、共同GT/名单和数值检查均完成才报完成。

报告回答background/illumination定位收益是否在43重复、view是否仍退化，以及Target统计相对Source统计的支持敏感程度。保留所有图像读出与定位取舍；两支持组不足以证明普遍稳定，也不能据变化认定唯一失败机制。历史926/878指标只能作背景，不能直接相减。本轮尚未正式执行，三个研究问题均待结果。

## 本地验证和检查清单

```powershell
& 'D:/anaconda/miniconda/envs/chinamoblie/python.exe' -B -m unittest `
  DINOv3.tests.test_aebad_s_target_loo_support_retest `
  DINOv3.tests.test_aebad_s_target_loo_layer_weight `
  DINOv3.tests.test_aebad_s_target_loo_layer_weight_domains
```

检查包括：名单/支持顺序与原身份；Source与support_seed分离；K8/K7自身排除；提取配置和物理positions；源权重共享与Target分组；42保留图预测一致性；地图/GT重放身份；中间指标封存、范围和像素数；同seed组合；有限指标、共同GT及宏平均；输出与父资产隔离；12h预算。

大数组读取沿用fast本地模式（元数据封存、存在/shape/dtype检查），不逐次重算大数组SHA；新增数组首次写出保留原项目身份，CPU已生成地图/GT身份用于AUPRO数值复核，小型完成表使用封存验证。真实加载权重时原host的固定权重检查不改。本轮无新增环境安装、资源准入、自动下载或自定义Triton。

CPU测试与本地盘点不代表真实CUDA、完整特征加载、实际检索吞吐或ADEval已通过；见 `implementation_audit.md` 和 `validation_results.json`。

本轮实际结果：25项新增测试＋31项原内核/跨域回归，共56项；包含父查询生产格式的预检/零前向复用、正常缓存兼容性及错误身份/布局拒绝测试。最新通过状态与耗时见 `validation_results.json`。10份Python源码AST检查、Bash语法检查通过，测试及预检进程未导入Torch。
