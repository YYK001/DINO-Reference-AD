# Target-LOO层加权：权重交叉与可分性审计

固定的离线机制审计，不是新检测器。Source、两组K8、四层标准化UNION响应、全部u/p/q权重和902张共同名单不变，不训练、不拟合新权重、不提特征、不做Exact、不搜索参数。

## 固定干预与输入

每域分别构造以下矩阵；所有权重交换整套u/p/q三列，正式重加权只调用原 `aebad_s_target_loo_layer_weight_p0.numerics.weighted`：

|底层memory响应|WS|W42|W43|
|---|---|---|---|
|Z42|复用|复用|新增诊断交叉|
|Z43|复用|新增诊断交叉|复用|

三域六项新定位图、六项新精确CPU/AUPRO。其余12个矩阵格的完整指标只读复用，既有UNION、SEGMENT和组合保留为背景参照。两种图像读出共享地图，固定PL主读出；不按结果切换。

交叉分支仍使用一个K8检索memory，但另一组K8提供统计；当前两组不重合，涉及16张Target正常图的信息。`diagnostic_only`、`memory_seed`、`weight_source/seed`与这一信息量写入冻结计划；不能将最好交叉格当作同信息K8方法成绩。

父运行：`aebad_target_loo_two_K8_20261030_013433`。共同名单保持background289（77/212）、illumination257（59/198）、view356（76/280），共902张唯一图像、1804份memory响应。只排除42/43支持并集，不新增44或重新抽样。Source种子42，66张Source正常统计共享。

主要输入仅为每张图、每套memory的：

```text
manifest.json       # 原始完整封存记录，原样保留
UNION_K8.npy        # 已经完成Source patch MAD的FP32四层Z，4×M
positions.npy       # int64 compact→物理patch索引
```

旧manifest列出但本轮不使用的NN/SEGMENT/其他分支数组不要求携带。读取明确只取Z与positions，不再做MAD、除1280、图像MAD或特征提取。最终融合图不能逆推出Z。

本轮不需要Source bank实体、原始1280维特征、checkpoint、骨干目录、Target特征库或原RGB。GT可由已有AeBAD.zip或仅有ground_truth的目录提供，采用原GT字节身份与resize规则。

## 本地预检结论

本地小型父review完整，可复用全部冻结权重、正常统计、对角逐图分数/地图身份、24项原AUPRO及原指标。已读取窗口记录39—41节与原空间/评分实现。

本地没有1804份原四层Z/positions实体。服务器JSON中的路径只是查找线索，没有当作“数组存在”。`datasets/AeBAD.zip`存在，预检仅检查GT索引；未读取GT进行新诊断。

预检结果位于：

```text
DINOv3/outputs/mad_equal/aebad_loo_weight_separability_preflight_20261030_v1/
  asset_inventory.csv
  missing_inputs.csv
  required_response_files.csv
  time_estimate.json
  cross_plan.csv
  crossed_metrics.csv
  objective_report.md
  status.json
```

新交叉格与全部新诊断均为pending，不能从已有对角结果认定是权重问题或memory问题。预检不导入Torch、不初始化CUDA；不会自动构库或补旧实验。

## 取回最小响应缓存

代码目前只完成本地实现，下面是准备好的人工命令，须先把本模块同步到当前工作站。本轮没有连接服务器或执行这些命令。

在保留原缓存的工作站上，默认脚本通过父 `baseline_origin` 寻找真实记录，也提供迁移后两套缓存根。仅导出所需三文件，不改写旧manifest，不拷贝NN、Q或memory：

```bash
cd /home/ubun/YYK/DINO-Reference-AD
script=DINOv3/MADEqual/aebad_s_loo_weight_separability_audit/scripts/run_workstation.sh
export RUN_ROOT=/home/ubun/YYK/runs/aebad_loo_weight_cross_audit_v1
export EXPORT_ROOT=/home/ubun/YYK/storage/audit_exports/two_K8_UNION_Z_v1
bash "$script" preflight
# 预检列出不存在的原数组时先定位；不能自动重跑。
bash "$script" export-responses &&
tar -czf "${EXPORT_ROOT}.tar.gz" -C "$(dirname "$EXPORT_ROOT")" "$(basename "$EXPORT_ROOT")"
```

导出约0.5–1GiB的规划范围，确切大小由实际M决定；当前常见118×118物理网格下约0.6GiB。此数不是来自本地不存在的NPY实测。若原响应已删除，工具按图列出缺失，恢复需要重复旧memory查询/提取的成本另议，本工具不执行。

本地解压导出包后，`--response-root` 指向含 `background/memory42/...` 的目录。原manifest和原seal保持不变，迁移路径不会被伪装成旧服务器路径。父review仍指向本地 `server_results/...`。

## 阶段与手动执行

默认preflight，缺实际输入返回2。CPU与AUPRO分开保存。首次运行显式使用新输出目录；冻结后变更输入根或父依赖须新建输出，不自动覆盖旧目录。

本地CPU示例（CPU路径不会导入Torch）：

```powershell
$py = 'D:/anaconda/miniconda/envs/chinamoblie/python.exe'
$module = 'DINOv3.MADEqual.aebad_s_loo_weight_separability_audit.runner'
$out = 'G:/codex/workspace/experiment/DINOv3/outputs/mad_equal/loo_cross_audit_local_v1'
$responses = 'G:/codex/workspace/experiment/server_results/two_K8_UNION_Z_v1'
& $py -B -u -m $module preflight --output-root $out --response-root $responses
& $py -B -u -m $module all-cpu --output-root $out --response-root $responses
```

`all-cpu` 顺序执行：preflight → smoke（无GT）→ predict（冻结全部响应）→ smoke-gt → evaluate → analyze → summarize → pack。完成CPU不等于五指标完整，六项AUPRO保持pending。

在当前工作站直接计算也不需要模型/训练。使用既有yyk环境，最多8 CPU，cache读机械盘、独占pixel scratch位于YYK固态。默认不安装环境、不修改base、不关机。单GPU仅用于显式原ADEval阶段：

```bash
cd /home/ubun/YYK/DINO-Reference-AD
script=DINOv3/MADEqual/aebad_s_loo_weight_separability_audit/scripts/run_workstation.sh
export RUN_ROOT=/home/ubun/YYK/runs/aebad_loo_weight_cross_audit_v1
bash "$script" preflight
bash "$script" smoke
bash "$script" predict
bash "$script" smoke-gt
bash "$script" evaluate
bash "$script" analyze
# 以下需用户授权后在原ADEval环境执行；只用事先指定的一张卡。
GPU_INDEX=1 bash "$script" aupro
bash "$script" summarize
bash "$script" pack
```

`all` 为包含AUPRO的显式完整流程，须指定GPU_INDEX；本轮没有执行。可以通过 `--domains view --memory-seeds 43` 续跑单设置；GT阶段仍要求全部六组无GT响应已冻结。预测按图提交，CPU/分析按域×memory提交，AUPRO按六项独立提交。相同输入重复执行复用有效完成记录；损坏/未封存的中间指标拒绝续用，不为其重新盖章。

`PARENT_RUN`、`RESPONSE_ROOT`、`MEMORY42_ROOT`（SACP predictions根）、`MEMORY43_ROOT`（two_K8_cache根）、`DATA`、`EVALUATION_SCRATCH`、`PYTHON` 可覆盖脚本默认路径。没有模型/检索/训练参数。

## 评分、排序和空间审计

1. 三权重重放原FP32地图/评分；Source/对角地图SHA必须一致、分数按原1e-5核对。排序继续使用父实际冻结Source/对角标量，避免把CPU微小重算舍入当成并列变化。新交叉分数先无GT冻结。
2. 枚举本域全部正常—异常对，状态精确为负/零/正；记录全部9类迁移、新旧异常减正常间隔。`analysis/pair_records/<comparison>/` 保存紧凑NPY记录、所有八项p/q贡献和FP64分解残差。独立图像对正确率核对原Image AUROC；贡献不是AUROC/AP可加贡献。WEIGHTED_MAP_MAX不套线性分解。
3. 正常图固定同memory WS的峰值与Top1%集合。并列按响应降序、物理像素flat索引升序确定，不能随分支重选。逐图输出，再图像等权汇总，不解释为阈值FPR。
4. 原8连通GT实例、原16像素Chebyshev环带、排除全部缺陷、原resize直接复用。固定WS内部/环带峰值，核对四层u贡献和FP32融合/插值残差；自身峰值另列，margin分类沿用±1e-6。消失实例/空环带保留NA；受损不是漏检，重加权响应不是原检索距离R_N/R_A。
5. 保留连续原始面积及实例ID，生成每设置 `area_margin.png`，不搜索小缺陷分界。正常统计尺度、权重、实际正常u/p/q分布、p/q单层AUROC、修复/破坏对贡献和固定margin贡献在同一层表输出。Source尺度从已封存正常表作小型核对，没有拟合新权重。

## 输出与完成条件

根目录：`crossed_metrics.csv`（18格×两读出=36行）、`fixed_comparisons.csv`（270项）、`per_image_scores.csv`（5412条）、`image_pair_transitions.csv`、`layer_pair_contributions.csv`、`normal_tail_changes/summary.csv`、`defect_instance_changes.csv`、`margin_summary.csv`、`normal_proxy_vs_discrimination.csv`、`verification.csv`、`costs.csv`、`time_estimate.json`、报告、状态与背景指标。

每域/每memory保存原始子表、响应冻结、一个新CPU像素作业、一个新AUPRO、紧凑全图像对数组和面积图。review包含紧凑图像对数组，不含原Z、完整地图、模型、GT或pixel scratch。

只有六新CPU、六新AUPRO、全部空间/排序分析及数值核查完成，整体才报 `weight_cross_and_separability_audit_complete`。否则明确pending。跨域不池化图像，不选择最好交叉，不把诊断格回填成方法成绩。

## 耗时、磁盘与检查清单

无GT smoke按每域原名单前两张固定测速，不使用测试标签选样，仅测运算。全部交叉响应冻结后，GT smoke固定第一张正常和异常图，测实际读取/插值、实例与小样本精确排序，之后才外推完整CPU。NlogN排序按实际样本量估计；AUPRO只引用父同名单/后端的历史作业时长作参考，不当作本轮实测，也不重复累计其地图重放。图像对/汇总等未单独测速的开销保留未知并反映在规划区间，不填0。

当前本地Z缺失，**完整耗时仍未知，不能承诺几分钟完成**。`time_estimate.json` 有分阶段待测项、历史AUPRO参考、零模型计算计划。新审计输出暂估0.05–0.5GiB、SSD scratch约1–8GiB，原缓存只读；无RAM/显存/磁盘准入门槛。实际记录每阶段时间、进程RSS峰值与显式AUPRO显存峰值。预计超12h须先征求同意，运行中超限在提交单位保留断点暂停；只有明确授权后使用 `--allow-over-12h`。

检查类别完整列于实现审计：父完成/依赖与共同名单；Source/支持/权重来源；原数组manifest与FP32布局/positions；原对角复现；无GT全响应冻结；新地图/GT身份；分解残差与精确并列；NA；原全像素结果封存/范围/数量；隔离目录与独占scratch；12h预算。fast读取不重复扫描大数组SHA，不放宽原容差，不安装依赖或修改历史产物。

本地CPU测试、夹具端到端评价不等于真实902张重放、真实分解残差或原ADEval GPU已通过。当前尚未连接服务器，尚未完成六项新交叉的实际分析，见 `validation_results.json` 与 `implementation_audit.md`。

本轮实际验证：16项新增＋56项回归，共72项CPU测试全部通过（25.49秒，无跳过）；11份Python源码AST及Bash语法检查通过。测试及预检未导入Torch。新交叉实际指标与真实缓存smoke仍pending。
