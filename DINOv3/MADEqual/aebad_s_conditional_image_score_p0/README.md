# 正常统计驱动的SEGMENT整体响应条件校正

状态：独立本地实现、CPU轻量验证和资产盘点。未连接服务器，未初始化CUDA，未执行真实提取、检索或正式实验；未安装环境或修改历史结果。本轮不恢复共享残差A/B/C。

## 固定问题和协议

只改变图像分数，检验原SEGMENT CH1之外，全图多数位置的响应水平能否改善正常/异常排序。仅一个固定配置，不搜索分位数、正则、层或支持子集。

三域有开发历史，不称盲测。Source_seed42，Source455构库/66独立校准、原增强100K/层memory、L7/L15/L23/L31、1280维、原Exact/MAD不变。support_seed42/43各自固定K8，不合并支持。

|域|正常|异常|共同评价|
|---|---:|---:|---:|
|background|77|212|289|
|illumination|59|198|257|
|view|76|280|356|
|合计|212|690|902|

名单沿用父两K8实验，排除两组支持并集。902张唯一图、1804个图像×支持组实例，不能称为1804张独立图。

## 固定统计与闭式解

P为原SEGMENT CH1：原各层插值峰值、原生valid3×3局部均值峰值、原Source image MAD、0.5/0.5汇总。B为四层SEGMENT原生z分别取全网格中位数后等权平均。z使用原Source patch MAD，不重复标准化，不截负值，不取插值像素中位数，不筛前景/Top百分比。完整物理位置必须一一对应，padding、重复/缺失位置拒绝。

每设置准备66条Source校准图→Source＋当前K8的SEGMENT统计，以及8条当前支持整图留一→Source＋本组K7的SEGMENT统计，共396＋48＝444条。这不是444张独立正常图。Source检索分量可以共享，不代表Source-only的旧(B,P)能共用；各设置的Target参照不同，新增Source回归不能共用。

所有矩使用ddof=0。令mu_BS=mean(B_S)，sigma_BS=sqrt(mean((B_S-mu_BS)^2))，x=(B-mu_BS)/sigma_BS：

```text
beta_S  = max(0, Cov_S(x,P) / Var_S(x))
c_T     = mean((x_T-mean(x_T))*(P_T-mean(P_T)))
v_T     = mean((x_T-mean(x_T))^2)
beta_ST = max(0, (c_T+beta_S)/(v_T+1))
```

Source实际方差参与计算，不假设浮点值逐位等于1；正则固定1，不加额外截距。若sigma_BS≤64·eps64·max(1,max|B_S|)，两校正均返回原P；不换B、不转为Target单独拟合。非有限输入/矩报错，不按正常退化吞掉；v_T=0自然返回beta_S。beta=0严格保留原P；不限制大斜率、不截负分数、不按单图择分支。

|正式分支|分数|
|---|---|
|SEGMENT_CH1|P|
|SOURCE_CONDITIONAL|P-beta_S*x|
|TARGET_CONDITIONAL（预定主候选）|P-beta_ST*x|

Source拟合使用“Source校准图＋当前Target K8”，不能称无Target信息的检测器。Target K7关系只是正式K8关系的代理；不假设同分布、正常上界或未知缺陷安全，不补K7→K8映射。

## 定位与原分数兼容性

三分支统一引用父两K8实验同域、同seed的UNION_SOURCE_WEIGHT地图及身份。只新算Image AUROC和average precision；Pixel AUROC/AP/AUPRO继承并明确标注未改变。本轮新增像素任务和AUPRO均为0，不用seed42共享残差实验的新直接范数地图。

查询原生响应重建CH1必须通过原1e-5；通过后P引用已封存的原SEGMENT逐图分数，避免微小重建舍入改变主基线排序。原距离使用MM Exact，不能静默切到直接向量范数。原geometry内部小块FP64维持原样；新增B及紧凑矩可用FP64，完整特征/检索仍原FP32。TF32/BF16不改。

## 复用接口及实际依赖

优先级是小(B,P)表→四层原生SEGMENT→逐支持距离/索引＋Target向量重建原几何→显式补检索/特征。旧Source-only、UNION LOO、u99%统计不兼容；聚合Target7距离不能恢复两个端点的原SEGMENT。

- normal和test分开接入。Source66/Target留一图不进入父测试baseline适配器。
- K7时距离、索引、特征列表和支持顺序一起筛为7项，排除整张heldout图全部patch。旧geometry的K8_SEGMENT只是历史键名，不代表仍用8项。
- 复用ResidentExact、geometry_layer、statistics/from_z/ch1_score、image_metrics/pair_errors。raw_distance只做一次MAD，standardized_z走from_z。
- 完整特征绑定原图、完整提取配置和provider manifest，兼容metadata.image/sample。小authority在阶段内缓存，不每图反复解析大assets.json。
- Source66按图外循环服务六设置；Source距离一次缓存后共享。测试按图外循环服务两支持组，同图最多一次缺失前向，不按三个评分分支重复提取。
- 缺Source实体时停止需要重新检索的阶段，不提供重建入口。完整小统计/原生响应/兼容几何输入的离线处理不要求模型或Source实体。
- 正常特征补取后保存，供跨设置/续跑使用；902张测试高维特征默认流式，不全部落盘。

可选`--statistics-provider`、`--response-provider`、`--retrieval-provider`使用封存JSON：

```python
# 使用现有storage.sealed_json创建；records键为完整contract的digest。
{'schema': 'conditional_exports_v1', 'records': {digest(contract): record}}
```

contract由inputs.contract生成，包含kind、域/seed、完整图身份、Source、提取/MAD身份、有序K7/K8、heldout、层及几何定义。BP record保存B/P、layer_medians、peak/local、grid、positions_count、GT_used=False；原生response引用有path/seal/field/encoding/contract；NN引用的contract使用retrieval_contract，包含Source_distance[4,N]、target_distance/index[4,K,N]及positions。

原生response实际metadata需有对应contract/encoding。另支持旧dependencies格式的只读桥：实际image/source/extraction/完整有序支持、原Exact及geometry SHA必须匹配，字段限定raw_segment或SEGMENT_K8；标准化响应还核对原patch MAD。仅给旧文件换名或附一份新声明不能通过。旧manifest不改写，导入另存来源及原seal。

FEATURE_MANIFEST可使用原项目封存完整特征导出；SOURCE_BANK等路径均可显式覆盖。只信任已知路径及实际字段，不自动SSH、搜索远端或重写旧身份。

## 阶段与续跑

默认preflight只读盘点，无Torch/CUDA。依次为prepare-normal→fit→collect-query→predict→evaluate→summarize/pack；可单独重放，all执行同一固定链。fit只读正常统计；先冻结六设置预测，evaluate才读标签/可选GT。GT面积缺失不阻塞核心两项图像指标。

smoke额外提供18个真实完整网格样本路径：六设置各一条Source、Target LOO和查询，结果直接复用。保留原处理耗时，断点重放不把小表读取毫秒数冒充原缺失检索吞吐。仅同域/seed/类型/执行路径的实测时间用于对应待办外推；没有对应测量的类别保留unknown。

输入配置、支持、父身份、provider及计划后端冻结。CPU/CUDA提取/检索后端改变使用新目录；fit/predict/evaluate等纯CPU后处理继承原生产后端身份。旧目录不静默迁移，不改变原数据/公式恢复断点。

下列是review及授权后由用户手动执行的命令，本轮没有运行。使用已有yyk环境，不安装依赖、不修改base、不关机。缓存放YYK/storage机械盘，小报告放YYK/runs，最多8 CPU。GPU_INDEX只选用户指定的一张物理卡。

```bash
source /home/ubun/miniconda3/etc/profile.d/conda.sh
conda activate /home/ubun/YYK/conda_envs/yyk
cd /home/ubun/YYK/DINO-Reference-AD
export RUN_TAG=aebad_conditional_image_score_$(date +%Y%m%d_%H%M%S)
export RUN_ROOT=/home/ubun/YYK/runs/$RUN_TAG
export CACHE_ROOT=/home/ubun/YYK/storage/conditional_image_score_cache/$RUN_TAG
export GPU_INDEX=1  # 仅在用户指定GPU1时；首次preflight前设置，预检不初始化CUDA
script=DINOv3/MADEqual/aebad_s_conditional_image_score_p0/scripts/run_workstation.sh
bash "$script" preflight
# 查看asset_inventory.csv、missing_inputs.csv、necessary_work.csv。
# 原始实体恢复且获授权后，才显式补缺：
bash "$script" smoke --supplement-retrieval --extract-missing
cat "$RUN_ROOT/budget_estimate.json"
# 测速通过且预计不超12h，才进行整轮：
tmux new-session -d -s "$RUN_TAG" \
  "cd /home/ubun/YYK/DINO-Reference-AD && env RUN_TAG='$RUN_TAG' RUN_ROOT='$RUN_ROOT' CACHE_ROOT='$CACHE_ROOT' GPU_INDEX='$GPU_INDEX' bash '$script' all --supplement-retrieval --extract-missing > '${RUN_ROOT}_launcher.log' 2>&1"
tail -n 40 -F "${RUN_ROOT}_launcher.log"
```

未显式打开补算开关，缺失输入只报错，不自动加载模型。已有完整BP/原生响应的CPU阶段不需补算开关；全部缓存完整可直接all。整轮缺失补算需先smoke。预计或已累计超过12h保留断点，必须新获同意才传`--allow-over-12h`；本轮没有超限授权。没有RAM/显存/磁盘容量准入门槛。

服务器路径覆盖变量：PARENT_RUN、MEMORY42_ROOT、MEMORY43_ROOT、RAW_CACHE、SACP_RUN、SUPPORT_FEATURE_ROOT、SOURCE_BANK、FEATURE_MANIFEST、STATISTICS_PROVIDER、RESPONSE_PROVIDER、RETRIEVAL_PROVIDER、DATA、CHECKPOINT、PYTHON。旧AutoDL路径不用。

本地CPU验证入口：

```powershell
& 'D:\anaconda\miniconda\envs\chinamoblie\python.exe' -B -m unittest DINOv3.tests.test_aebad_s_conditional_image_score -v
& 'D:\anaconda\miniconda\envs\chinamoblie\python.exe' -B -m DINOv3.MADEqual.aebad_s_conditional_image_score_p0.runner preflight --run-root DINOv3/outputs/mad_equal/aebad_conditional_image_score_preflight_20261102 --sacp-run server_results/aebad_s_sacp_seed42_20261028_011454
```

## 成本与交付

本轮未做服务器真实测速，总时长保持未知，不承诺十分钟。待办依据分为小表读取、原生响应统计、正常/查询前向、Exact＋原SEGMENT、CH1插值、写盘评价与打包。preflight及smoke按实际路径报告，不把缺项当0；已完成小表不计待办。

最坏新增正常前向最多66＋48张，不是444次；测试最多902次，不是1804×3。正常响应需求仍为444条，Target不同不能强行共用。若全部正常响应需重算，四层Source搜索最多(66＋48)×4=456次，Target搜索396×8×4＋48×7×4=14016次；测试缺失时Target最多1804×8×4=57728次，Source尽量共享/复用。只允许原Exact，不换近似。

当前未测资源规划：进程RAM约10–30GiB、CUDA约8–20GiB，实际取决于现有特征和参照驻留；不是容量检查或保证。新小表/Source距离为紧凑产物；若正常高维特征全缺并补存，Source66及最多48支持约30GiB量级（按13924 patch），机械盘保存，旧特征不复制。测试902张完整特征不默认落盘，无全像素排序scratch和新AUPRO。

输出normal_statistics、coefficients及48条coefficient_influence、query_statistics、per_image_scores、metrics/comparisons、ranking_summary、正常/异常score_changes、面积关系、六张散点、数值复核、costs/resource_summary、objective_report、status及轻量review包。汇总包含六设置、每seed三域等权、每域两seed等权、三域再两seed等权，不池化重复图。排序对不是独立样本或固定阈值误报/漏检。

报告分别回答检测是否超过SEGMENT、Target是否超过Source关系、域/支持组反转及可能过补偿；系数/散点拟合不等于检测增益或物理机制。删除LOO统计只做影响分析，Source标准化和beta_S固定，不重新排除支持、不挑系数。

只读大数组采用现有trusted-local fast：不反复重扫payload SHA；小封存/结果、header、位置和消费时数值核对保留。检查全表及尚未验证部分见implementation_audit.md。本地预检/测试日志/validation_results不上传Git；代码按用户后续授权发布；本机不连接服务器或启动正式实验，服务器运行由用户手动执行。
