# 跨支持图佐证的局部参照扩展：固定 P0

**当前执行版本（2026-10-23）**：支持间检索、整图留一和正式缓存预测均支持显式单/双GPU；佐证改为批量FP64，连接缓存改为数组批文件，CPU冻结/评价使用多核。下文旧串行/SQLite说明仅代表先前版本。当前执行、检查清单及新目录命令见[执行优化说明](execution_optimization.md)。公式与评价未改，真实CUDA提速尚未验证。

这是独立实验工具，研究原 Top2 连接是否能由第三张正常支持图的实际 patch 佐证。不是新检测器的有效性结论，不保证整条线段正常、物理同源或缺陷保持。不得把互近邻、图结构、插值或留一本身写为首创性。

本地只执行合成CPU测试、语法检查和只读预检，没有执行真实模型、支持检索或正式实验。2026-10-22用户已授权在AutoDL恢复缺失支持特征并启动正式实验，新增显式恢复与整轮运行入口；本工具不自行连接服务器。历史代码、结果和窗口报告不改。

## 冻结定义

原 seed42 K8，background 297（85/212）、illumination 265（67/198）、view 364（84/280），共926张。各域保留原支持顺序和排除规则，不使用878张敏感性共同子集。原Source、完整K8、L7/L15/L23/L31、1280维原始特征、patch/image MAD、CH1和CURRENT、原四层FP32均值和插值均冻结。三个域已用于开发，不是盲测。

每层、每个支持patch，分别在每张其他图检索 `k_w=5`。采用原始欧氏距离、不归一化、投影或采样；保存原特征行号。CPU为SciPy FP64直接`cdist`；显式CUDA为Torch FP64 `cdist(compute_mode='donot_use_mm_for_euclid_dist')`，GPU稳定排序、块间合并后按query块回传最终5个邻居。所有并列按距离、原始行号排序，跨块同样处理，不常驻全库两两矩阵。CPU/CUDA是明确的执行配置，正式开始后`neighbor_config.json`冻结，不能续跑时切换数值后端。历史查询距离不重新计算、不换后端。

无向连接按 `(support_id, patch_row)` 规范化。对第三张图c，候选仅为 `N_c^5(u) ∪ N_c^5(v)`，不是整库穷尽线段搜索。FP64直接差分计算：

```text
alpha = dot(w-u,v-u) / dot(v-u,v-u)
e = norm(w-u-alpha*(v-u))
b = min(norm(w-u), norm(w-v))
accept_point = 0.25 <= alpha <= 0.75 and b > 0 and e/b <= 0.5
accept_edge = at least ONE different support image has an accepted point
```

零长度、零分母和非有限原因明确记录，不使用epsilon制造通过。一张图多个点只计一个来源。端点交换得到相同结果。连接只依赖支持集合，按需计算、按唯一连接缓存；不依赖query身份、响应、顺序、GT或标签，不声称已物化完整图。

查询仍按原八支持各自Top1的稳定Top2，使用已冻结端点索引，不搜索其他连接。设u为完整Source+K8 UNION、v为原SEGMENT、D=max(float64(u)-v,0)：

```text
UNION_K8                  = u
SEGMENT_K8                = v
WITNESS_SEGMENT_K8         = float32(u - g*D)
beta                      = sum_FP64(g*D)/sum_FP64(D), zero denominator => 0
MATCHED_UNIFORM_WITNESS    = float32(u - beta*D)
```

原距离已除1280，不再除。`v>u+1e-7+1e-6*abs(u)`立即失败；容差内微小负缩距按原GAP规则记录后置零。主候选与均匀对照的实际FP32总缩距用原ULP求和界审计，不改变位置凑相等。只匹配每图每层原生总量，不匹配地图、图像分数、幅度分布或实例margin。拒绝回完整UNION，Source和所有真实Target离散参照始终存在。

整图留一单独标为 `TARGET_ONLY_LOO_COVERAGE`。留出h不作为端点或第三方佐证，其所有patch仅作为待解释正常查询。复用保留图之间的邻居表，连接缓存有独立的heldout命名空间。比较Target7离散、无约束线段、佐证线段、均匀对照；不加载Source，不将覆盖诊断当检测收益或用于调参。

## 输入、迁移与本地缺项

默认复用 `aebad_s_gap_soft_segment_p0.inputs.Bundle` 的只读父身份适配。background从原SEGMENT/LOO tar按positions对齐，illumination/view从当前 `LiveParent` 自身的原629份raw读取。没有重新套用更早Kaggle运行的身份。`target_distance[8,4,M]`、`target_index[8,4,M]`、Source距离、原SEGMENT、positions和metadata不可替换为union-only缓存。

另需要24份原支持NPZ：每域8张，每份含`protocol_id`、`relative_path`、`grid`及四个`layer_<l>`，每层 `[13924,1280]` FP32（实际按各原grid验证）。需与原capture的支持图片身份、顺序、grid、文件SHA一致，不接受距离、少量案例向量、Target512或SPARC特征代替。

支持查找使用目录递归及实际归档成员清单，允许直接读 `archive.tar::member`，不自动解压。默认预检还检查已展开父元数据、所有cross raw路径及每域一张缓存的完整SHA/字段/端点契约；**不是在预检中解码全部926张**。正式使用每张缓存时独立核验历史SHA。支持预检只读NPZ头和哈希，有限性在显式加载一层时验证。没有缺失自动提取路径；显式恢复见下文。

本地核查发现：历史父结果、Source两级MAD、背景原始归档和两域629份raw均已存在；所检查目录/归档中未找到与原记录匹配的完整支持特征。准确24份清单见 `local_validation/required_support_files.csv`，历史哈希、图片路径、支持编号、grid均列出；不能把“存在结果包”解释为特征已经可用。按13924行计，24份四层未压缩特征约6.37GiB，NPZ压缩实际大小未知。

支持可放到一个独立根目录，例如 `G:/.../witness_supports/background/01.npz`、`illumination/001.npz`、`view/001.npz`；路径和文件名可迁移，内容及历史哈希不可改。用`--support-root`指定这个根目录，可重复指定。prepare/support-loo/predict仅要求本次选择域的特征。evaluate/AUPRO/summarize不要求支持特征实体、checkpoint、Source bank或骨干代码，只读冻结身份和预测依赖。

若原文件已丢失，默认路径仍明确阻断。现在可经用户授权显式使用 `recover-support` / `verify-support` 和 `--recovered-supports`：新文件保留自己的哈希及来源，原历史哈希保持不变，不会把重新提取的文件冒充历史文件。

## 显式阶段与本地手动命令

以下本地命令仅为分阶段示例，正式阶段在本地均**未执行**。现有独立AutoDL编排入口可以显式串联；缺原支持时先使用新恢复协议。首次单域可用background，后续相同目录、相同配置换域继续。`--layers`可分层准备/留一/预测；只有一个域的四层全部完成、逐图分数核查通过才冻结该域预测，才允许其GT评价。不会因只完成background便生成三域macro。

```powershell
$py = 'D:\anaconda\miniconda\envs\chinamoblie\python.exe'
Set-Location 'G:\codex\workspace\experiment'
$module = 'DINOv3.MADEqual.aebad_s_support_witness_segment_p0.runner'
$out = 'G:\codex\workspace\experiment\DINOv3\outputs\mad_equal\aebad_s_support_witness_fixed_p0_v1'
$features = 'G:\codex\workspace\experiment\downloads\witness_supports'
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'

# 默认只读，不创建实验目录，不导入Torch；缺项退出2。
& $py -B -u -m $module preflight --output-dir $out --support-root $features --no-write

# 新增大规模支持间检索；显式GPU入口，需另获授权。双卡把值改为cuda:0 cuda:1。
& $py -B -u -m $module prepare-support --output-dir $out --support-root $features --domains background --devices cuda:0
# 同一算法也可显式 --devices cpu，属于真实SciPy计算，预计较慢；不是自动降级。

# GPU批量佐证，整张支持图留一。不能把h当端点或佐证，不含Source检索。
& $py -B -u -m $module support-loo --output-dir $out --support-root $features --domains background --devices cuda:0 --cpu-workers 8

# GPU批量处理原query距离/索引与支持关系；CPU并行冻结，不提取query特征。
& $py -B -u -m $module predict --output-dir $out --support-root $features --domains background --devices cuda:0 --cpu-workers 8 --limit 3
# 上条是少量缓存smoke，状态partial，不能冒充全域完成。正式冻结同命令去掉 --limit 3。
& $py -B -u -m $module predict --output-dir $out --support-root $features --domains background --devices cuda:0 --cpu-workers 8

# CPU，冻结后读GT；两新分支全像素精确ROC/AP，原两分支复核后引用。
& $py -B -u -m $module evaluate --output-dir $out --domains background --cpu-workers 2

# 显式GPU，仅新增两分支的原ADEval AUPRO，不加载模型。
# 本地沿用现成portable ADEval时，在本进程设置搜索路径，不安装依赖。
$env:PYTHONPATH='G:\codex\workspace\experiment\downloads\adeval_1_1_0_runtime;G:\codex\workspace\experiment'
& $py -B -u -m $module evaluate-aupro --output-dir $out --domains background --devices cuda:0

# CPU，汇总已完成结果，保留缺域/缺AUPRO/缺LOO清单。
& $py -B -u -m $module summarize --output-dir $out
```

对illumination/view运行同样各阶段，修改`--domains`，不改变首轮规则；也可一次指定三个域。GPU阶段均支持`--devices cuda:0 cuda:1`，以域×层（prepare）或域×分支（AUPRO）作业分配两卡独立进程，不合并显存。CPU阶段使用原有有界多线程地图处理；不以估算内存/显存/磁盘前置准入，真实分配/IO错误保留。

## AutoDL完整运行（用户授权后手动启动）

先更新GitHub代码，再将本地`downloads/aebad_s_witness_frozen_inputs_20261022.tar`上传到服务器`/root/autodl-tmp/`。输入包约3.47GiB，742个原始文件，含背景归档、629份原始两域capture、冻结校准与父评价元数据；没有Source bank、checkpoint、图片或支持特征。该输入包不进GitHub。`portable.restore`按照已封存的清单恢复到代码根目录的原相对布局，不修改身份，不覆盖不同内容的现有输入。已有相同文件直接校验后复用。

```bash
repo=/root/autodl-tmp/DINO-Reference-AD
if [ -d "$repo/.git" ]; then
  git -C "$repo" pull --ff-only || exit 1
else
  git clone https://github.com/YYK001/DINO-Reference-AD.git "$repo" || exit 1
fi
cd "$repo" || exit 1
PY=/root/miniconda3/bin/python
"$PY" -B -u -m DINOv3.MADEqual.aebad_s_support_witness_segment_p0.portable restore   --archive /root/autodl-tmp/aebad_s_witness_frozen_inputs_20261022.tar || exit 1

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
RUN=/root/autodl-tmp/aebad_s_witness_$(date +%Y%m%d_%H%M%S)
nohup "$PY" -B -u -m DINOv3.MADEqual.aebad_s_support_witness_segment_p0.autodl   --run-root "$RUN"   --data /root/autodl-tmp/datasets/AeBAD/AeBAD_S   --checkpoint /root/autodl-tmp/DINO-NVS/external_baselines/SuperADD/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth   --devices cuda:0 --cpu-workers 8 --shutdown-on-success   > "${RUN}_launcher.log" 2>&1 < /dev/null &
printf 'PID=%s\nRUN=%s\nLOG=%s\n' "$!" "$RUN" "${RUN}_launcher.log"
tail -n 40 -f "${RUN}_launcher.log"
```

明确加入了**全部阶段与打包成功后关机**。中途报错不关机，保存失败review与全部现场；没有按可用RAM、显存或磁盘估算阻断。实际分配/IO错误仍失败。代码不安装环境；启动最先检查现有依赖和原ADEval1.1.0。缺依赖先明确失败，不会跑数小时后才发现评价器缺失。支持恢复需要checkpoint与原DINOv3宿主；默认官方源码随GitHub放在`external_baselines/dinov3`，可用`--dinov3`显式改路径。

默认上例单卡。确实有两卡时显式改为 `--devices cuda:0 cuda:1`，恢复按域、支持间检索/留一/预测按域×层、AUPRO按域×分支分配独立进程；不能把两卡显存相加。CPU评价使用有界地图处理和原精确全像素排序。没有CPU容量前置判断，没有错误后自动切换后端。

完整阶段：环境 → 恢复输入只读核查 → 24张固定正常支持提取 → CPU历史几何复核 → 正常preflight → 新支持间k5检索 → 整图留一 → 原926张缓存预测 → CPU评价 → 6项原AUPRO → 完整汇总 → review打包。**不重新提取926张query，不重建Source memory，不重新拟合MAD。**

运行目录包含`recovered_supports/`、`experiment/`、`logs/`、`launcher_config.json`与`pipeline_status.json`。每阶段日志及退出码单独保存，外层日志每30秒心跳。连接断开不会结束nohup进程。续跑使用同一`--run-root`、相同代码与参数，已封存成功作业校验后复用；改变数值后端/块配置/方法代码必须使用新目录。未提交或被修改的文件保留并拒绝，不重新认可。

成功包为`${RUN}_review.tar.gz`，失败为`${RUN}_FAILED_review.tar.gz`；原始支持、邻居NPZ、SQLite连接和预测缓存**不在review包**，仍留服务器，review不能独立续跑。不要在下载小包后删除大缓存，除非另行完整备份并明确决定不再复核。

## 支持恢复的证据与边界

`recover-support --no-write`只读核对原图片SHA、支持顺序、grid、checkpoint身份及提取源码，不导入Torch或创建恢复目录。显式`recover-support --devices ...`才懒加载原`aebad_s_sparc_p0.runtime.load_host`与`ShiftADD.aebad_s_p0.memory._extract_four_layers`，保持原640/128窗口、0.625预处理、四层FP32原始行、原精度和batch。只提取原24张，不抽样、不校正、不归一化。

恢复文件独立封存于`recovered_supports/features/`，每个文件的record包含原图片SHA、历史特征SHA及“显式重新提取”标记。`recovery_inputs.json`冻结宿主配置、checkpoint、官方源码、提取入口、精度和原capture记录。中断后核对先前文件SHA；孤立/未提交NPZ不会被重新提取覆盖或补认可。

默认严格模式的`verify-support`逐域逐层加载新特征，对926份原缓存的稳定Top2端点，调用原`geometry_layer`，核对**原SEGMENT距离与原endpoint_length**，继续使用`1e-7 + 1e-6*abs(reference)`。长度复核避免Source的min把支持变化掩盖。失败不放宽容差，不进入邻居准备。各域层检查独立seal，可续跑；全部通过才生成`verified_manifest.json`及独立seal。

后续`--recovered-supports`要求上述封存链完整，保留`historical_feature_sha256`与当前`feature_sha256`两个字段，把新manifest纳入本轮身份；不修改原capture/inputs或历史SHA。未显式传此参数的路径仍严格只接受历史NPZ SHA。

必须如实区分：这些检查证明新支持在所有已保存Top2连接的端点长度和原SEGMENT计算上兼容，**不能证明原NPZ字节相同，也不能在缺少原Q时证明八支持Top1检索逐元素完全等价**。本轮正式query继续使用原距离和原索引，不偷换为新查询检索。该边界与全部数值误差保存在恢复manifest中，后续报告不得省略。

## 手工取回支持特征

准确文件身份在required_support_files.csv。背景历史位置为原 `aebad_s_union_background_replay12_seed42_v1/support_features/target_matched/background/01.npz`—`08.npz`；两域历史相对位置为原capture根目录下 `support_features/<domain>/001.npz`—`008.npz`，并非当前2.45GB RAW归档中已经包含。

用户在**实际保存这些原文件的机器**执行以下标准库命令，先逐个核对清单，再仅打包24份；不包含Source bank或checkpoint。先将required_support_files.csv复制到该机器，设置真实`BACKGROUND_FEATURES`和`CROSS_CACHE`。不存在时脚本失败，不补算、不删除源文件。

```bash
export REQUIRED=/root/autodl-tmp/required_support_files.csv
export BACKGROUND_FEATURES=/root/autodl-tmp/DINO-NVS/DINOv3/outputs/mad_equal/aebad_s_union_background_replay12_seed42_v1/support_features/target_matched/background
export CROSS_CACHE=/actual/saved/crossdomain/cache
export SUPPORT_PACKAGE=/root/autodl-tmp/witness_original_supports.tar
/root/miniconda3/bin/python - <<'PY'
import csv,hashlib,os,tarfile
from pathlib import Path
rows=list(csv.DictReader(Path(os.environ['REQUIRED']).open(encoding='utf-8',newline='')))
assert len(rows)==24
files=[]
for r in rows:
    d=r['domain'];i=int(r['support_index'])
    p=Path(os.environ['BACKGROUND_FEATURES'])/f'{i:02d}.npz' if d=='background' else Path(os.environ['CROSS_CACHE'])/f'support_features/{d}/{i:03d}.npz'
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(1<<20),b''):h.update(x)
    assert h.hexdigest()==r['feature_sha256'],f'原始支持哈希不匹配: {p}'
    files.append((p,f'witness_supports/{d}/{i:03d}.npz'))
destination=Path(os.environ['SUPPORT_PACKAGE'])
with tarfile.open(destination,'x') as t:
    for p,name in files:t.add(p,arcname=name,recursive=False)
print(destination,destination.stat().st_size)
PY
```

本工具预检可直接检查tar内的支持NPZ。正式阶段默认从展开目录读取（显式目录`--support-root`）；在用户选择的输入目录手动解包后运行，不覆盖历史路径。

## 输出、续跑与成本

`neighbors/<domain>/L*/`按56个有向图对独立提交NPZ及原支持/配置身份；完整层status封存文件及record哈希。`connections/<domain>/L*.sqlite`按唯一连接与整图留出命名空间保存原索引、第三方候选、合格点alpha/rho、代表拒绝点和原因计数。拒绝点完整特征不重复保存，候选ID与原特征足够复核。

预测按层/逐图持久化，四层齐全后写`predictions/`及`prediction_manifests/<domain>.json`；未提交文件保留并报错，不重新认可已改文件。`--limit`只限制每层本次新增图数；续跑复用已验证提交。CPU评价按域提交并核对原地图/GT SHA、分数与实例。AUPRO各域×新分支独立作业，三域总共6项；CH1/CURRENT共用一次像素计算。

CPU中间像素结果`*_exact_pixels.json`在整域状态提交之前，先通过独立的`*_exact_pixels.seal.json`封存。封存记录绑定原结果文件SHA、输入/地图身份、域和分支、冻结预测清单SHA及实际GT正/负/总像素数。续跑先核对原封存和文件SHA，再检查AUROC/AP有限且在[0,1]、像素计数为整数且一致，以及原评价器的完整有效状态。未封存旧文件、孤立seal、篡改或非法结果均拒绝并保留，不重新登记当前哈希，也不自动覆盖或补造seal。中断若恰在结果与seal两次提交之间，也按未封存处理；需人工检查或选择新的输出目录。合法独立封存结果可在尚无完整CPU状态时复用，不重复像素排序。该检查不改变原评价算法、方法或参数。

完整表为metrics、metric_comparisons、support_loo_metrics、connection_summary、reduction_summary、mechanism_summary、normal_tail、ranking_summary、defect_instances、margin_transitions、margin_transition_summary、numerical_verification、objective_report和status。旧PAIR/GAP可作为研究背景，不在本工具运行或伪装成新分支。

支持间检索不是零成本。按每图13924个patch，单域单层有56个有向图对，共约108.6亿对距离；三域四层总约1303亿对。邻居表未压缩约0.697GiB，不常驻全部距离。一次只加载一层八个支持特征；GPU检索块默认256×2048 FP64。连接唯一数/接受率未知，当前连接数组证明可能成为主要新增磁盘成本，**不能承诺固定小磁盘足够**。测试query最多约5157万次逐层连接请求，按唯一键去重；不会预先穷举所有端点对。

cost.json及cost_records分别记录支持间检索、连接判定、留一、查询缓存处理、地图评价的墙钟与进程峰值RAM；显式GPU worker另记Torch/CUDA与显存峰值，独立worker记录与调度器开销不能相加冒充推理延迟。实际真实峰值和时间尚未测量。原先已成功CPU/ADEval环境只是可用线索，本轮没有调用GPU验证。

## 本地轻量验证

```powershell
& $py -B -m unittest DINOv3.tests.test_aebad_s_support_witness_segment -v
```

合成测试执行真实SciPy分块近邻、完整格式NPZ读取/原行来源、原geometry_layer、八次整图留一、SQLite唯一连接/续跑、四层缓存预测/CH1及CURRENT接入和冻结身份检查。另验证并列、端点交换、远点/近端点拒绝、整图排除、微小负缩距与FP32总量、篡改拒绝、未完整预测不能进入评价。CPU导入/检索子进程确认不导入Torch。

这些不是实际支持特征、CUDA、多卡、完整像素评价或真实全域实验通过。真实运行必须先通过恢复几何审计，冻结完整域后才能进入评价；本地只做轻量验证。

2026-10-22本地验收：34项合成测试通过（8.752秒）；11份模块Python文件与专项测试共12份内存编译通过。默认只读preflight退出2，原因是24份完整支持特征缺失（各域8份），不是Python/父缓存初始化失败。实际检查31个归档的成员目录，各域一张原raw完整SHA/字段复核通过；未导入Torch，未创建默认实验目录。完整记录见`local_validation/validation_result.json`。

同日P2修复后的最新回归为42项通过（10.070秒），新增8项覆盖CPU中间指标独立封存和中断续跑。合法结果无需重做排序；未封存、修改过或语义无效的结果会明确拒绝，不自动重新认可。12份Python源文件内存编译仍通过。没有真实支持检索、GPU或全域评价；方法和评价口径未变。代码身份随修复更新，已有其他代码身份的运行目录仍按原规则拒绝混用。

2026-10-22恢复与编排新增14项CPU测试通过（2.097秒），与原42项回归合计56项。新增测试执行真实小型恢复NPZ/封存、原geometry_layer、整批几何审核、portable打包恢复；宿主提取和流水线GPU/关机调用由替身隔离，不能写作真实CUDA或服务器运行通过。

## 2026-10-23：跨环境恢复兼容入口（显式选择）

用户服务器的独立CPU审计覆盖illumination265张和view364张、全部四层。原缓存自身的SEGMENT地图SHA及CH1/CURRENT分数复现通过；恢复几何相对原缓存的分数最大差分别约2.12e-6/2.38e-6和2.16e-6/1.91e-6，融合图最大差分别约4.57e-5与1.70e-5，view有2张峰值位置改变。此为用户提供的真实服务器审计，不是本地GPU实验。原距离严格检查并未全部通过，不能将这些结果改写成逐位复现或图像/像素完全等价。

默认strict不变。需要继续时显式指定`--reuse-recovery <原提取目录> --recovery-policy historical-maps-v1`，另建全新运行目录。跳过模型加载和支持提取，从原24份独立封存的NPZ读取完整特征，**不修改原recovery_inputs、NPZ/record、旧几何失败和旧audit**。新的兼容证书与新的方法运行身份分开保存，不能对旧目录原地改协议。

新政策是恢复数据接入的工程兼容规则，**不是原逐点距离容差被判定通过**，不是WITNESS检测规则：

| 核查 | 固定要求 |
|---|---|
| 图片/支持顺序、原QUERY/PRECISION、checkpoint、宿主配置、文件及record SHA | 与已冻结提取记录及父输入严格一致 |
| 原缓存自身重建 | 原地图SHA相同，CH1/CURRENT仍按1e-5复核 |
| 恢复几何的各层最大响应贡献 | `max(abs(S_recovered-S_old))/Source_patch_scale/4 <= 1e-4` |
| 恢复几何的四层融合定位图 | 完整地图最大绝对差<=1e-4，单位为冻结Source MAD响应 |
| 恢复几何的CH1和CURRENT | 每张图两项分数变化均<=1e-5，沿用原评分容差 |
| 原SEGMENT距离和端点长度严格检验 | 继续记录1e-7/1e-6下的真实失败数量，不重标为通过 |

地图1e-4是该**新兼容政策**的明确工程限值，不宣称它原本就是历史地图逐位验证口径。逐层除4来自原等权融合，避免大层误差经抵消后只凭最终图过关。它不进入检测分数、佐证alpha/rho/k5或缩距公式，也不根据GT、AUROC/AP选择。几何内部原三角形/有限性/距离修正规则保持不变，真正数学错误继续失败。

新证书必须重新计算全部926张CPU几何/地图/分数并逐域独立封存，不能从之前未封存的审计汇总补写可信SHA。原提取文件按先前记录SHA复核，禁止重新认可已修改文件；证书、逐域结果或seal被改、缺失或不完整均失败。只复用完整已封存作业。没有checkpoint、memory加载、Source MAD拟合或GPU前向。

```bash
cd /root/autodl-tmp/DINO-Reference-AD
git pull --ff-only
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
RUN=/root/autodl-tmp/aebad_s_witness_compat_$(date +%Y%m%d_%H%M%S)
nohup /root/miniconda3/bin/python -B -u -m DINOv3.MADEqual.aebad_s_support_witness_segment_p0.autodl \
  --run-root "$RUN" \
  --reuse-recovery /root/autodl-tmp/aebad_s_witness_20261022_193510/recovered_supports \
  --recovery-policy historical-maps-v1 \
  --data /root/autodl-tmp/datasets/AeBAD/AeBAD_S \
  --checkpoint /root/autodl-tmp/DINO-NVS/external_baselines/SuperADD/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth \
  --devices cuda:0 --cpu-workers 8 --shutdown-on-success \
  > "${RUN}_launcher.log" 2>&1 < /dev/null &
printf 'PID=%s\nRUN=%s\nLOG=%s\n' "$!" "$RUN" "${RUN}_launcher.log"
tail -n 40 -f "${RUN}_launcher.log"
```

checkpoint参数保留旧启动接口，复用模式不读取权重或构建模型。环境检查仍验证正式近邻/AUPRO依赖。成功后关机，失败不关机；单/双GPU仍显式选择。先生成新证书，然后继续邻居准备、留一、预测、CPU指标、原AUPRO与完整汇总。

最终status单独保留`historical_strict_distance_compatible`及失败数；`numerical_failed_checks`只指正式冻结预测/地图/分数，不把历史恢复差异隐去。objective_report必须说明使用恢复银行，历史分数差小不代表所有像素排序、原八支持Top1或WITNESS离散门控完全不变。没有原Q/旧完整支持，后二者仍未验证；这不是原特征逐位复现的结论。
