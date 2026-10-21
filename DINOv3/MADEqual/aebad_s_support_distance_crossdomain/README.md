# AeBAD-S：逐支持距离诊断的跨域验证

只检验background中发现的gap2/gap4/gap8与GT覆盖、实例margin的关系能否在illumination/view重复。它是描述性诊断，不是检测器，也不根据结果设计门控、阈值、层选择或融合。

固定原seed42、各域原有顺序的8张正常支持：illumination 265张（67正常、198异常），view 364张（84正常、280异常），共629张。所有计算按domain＋sample_id区分，实例再加instance_id。没有改用后来的支持敏感性共同子集。

## 接入及复用

- `inputs.py`核对冻结汇总包、原629张catalog、支持顺序、Source patch MAD、Source图像MAD、CPU分数和实例参考。原产物身份保持`e21f243fba926ea04c4593e1c3472c02f228c1c47d5bccbee01f75b46a45e6bb`；新分析程序及共享代码另记SHA，不重写旧身份。
- `discovery.py`查找展开缓存及tar/tar.gz索引，不解压大包。索引复用只是提示；每份实际raw仍核对原SHA和schema。可读归档中的raw，也可直接读原机器的cache目录。
- `analysis.py`直接调用background的`profiles.native/high_response`、`spatial.instance_analysis`、`analysis.instance_summaries`。CPU地图和CH1统计复用现有`numerics.reconstruct/adaptive_max`及`scoring.statistics/ch1_score`。
- `runner.py`直接调用background的`conditional.describe`；每域每层独立生成边界并做200次图像bootstrap，没有复制或近似统计内核。
- `reporting.py`保留逐域结果，并按domain×layer×feature汇总。background只读作发现参照，不重跑、不池化patch或AUC。

距离已是Euclidean/1280，不再除1280。输入必须包含`positions`、`source_distance[4,M]`、`target_distance[8,4,M]`、`segment_distance[4,M]`及原metadata。只有union_distance的预测缓存不能替代逐支持capture。端点几何若保存则核验；未保存的辅助几何明确NA，不重新取特征计算。

不需要Source bank实体、checkpoint、DINOv3官方代码、支持特征或原RGB图片。评价仅需GT；GT目录无需训练集或RGB文件。GT字节、尺寸及映射按原catalog核对。原AeBAD.zip可直接读取。

## 冻结分析规则

gap_k=(排序后第k张支持距离−最小支持距离)/原Source MAD尺度，epsilon及浮点次序完全沿用`profiles.native`。gap2/4/8和L7/15/23/31全部报告，其余原16维体系中的描述量作为辅助保留。

正常组：每图原生融合UNION响应最高ceil(1%×物理patch数)；异常组：原adaptive_max映射的GT覆盖物理patch并去重。同图四层使用相同物理位置。条件比较仅保留原有效SEGMENT缩距位置。

组内图像等权、图内patch等权；按原规则对b和delta生成5×5分箱、重复边界合并，按两组共同最小质量匹配。每个域独立计算数值边界，不复用background数值。200次分组图像bootstrap保持原PCG64 seed42及调度无关的调用，固定原分箱、重算每次的共同质量。GT覆盖组始终为AUC正类，不翻转小于0.5的AUC。

共同覆盖、未覆盖质量、最大单图权重、b/delta残余SMD、特征有效质量及patch/图像数、NA和有效bootstrap次数均保留。没有共同支持、没有有效缩距或特征全NA时保留不可计算行，不静默删项。

实例沿用8连通、原16像素Chebyshev环带并排除所有GT、原插值回溯和不可计算状态；记录自身峰值与固定UNION峰值的margin及分解、峰值移动、inside/ring逐支持描述。父CPU结果中已有峰值坐标则复用，缺少坐标列才按原首个row-major最大值派生并标明来源。小实例使用各域**原面积最小四分位**，不沿用background最小三实例或固定案例名单。

## 两类数值复核

1. **同批缓存**：Source CURRENT、UNION/SEGMENT CURRENT及CH1分数、实例标量、固定位置响应和插值重建沿用1e-5；整数面积/覆盖检查为精确一致。失败阻止完成。
2. **更早正式分数**：沿用原历史兼容政策1e-4，同时写出1e-5的`strict_verified`，不得混称严格复现。

真实父审计：capture有2516条历史检查，其中24条未过1e-5、0条被1e-4拒绝；父CPU有1887条，其中19条未过1e-5、0条被拒绝。最大历史误差1.9550323486328125e-5。重复旧检查只作为原审计记录保留，不充当新增独立检验。

## 本地预检及已知缺项

默认仅预检，写输入清单，不执行诊断。始终不导入Torch，不运行GPU、模型或评价指标。

```powershell
Set-Location G:\codex\workspace\experiment
$python = 'D:\anaconda\miniconda\envs\chinamoblie\python.exe'
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'
& $python -B -m DINOv3.MADEqual.aebad_s_support_distance_crossdomain.runner preflight
```

2026-10-21实际预检目录：
`G:\codex\workspace\experiment\DINOv3\outputs\mad_equal\aebad_s_support_distance_crossdomain_preflight_20261021_070926_594109`

已读downloads、server_archives、server_results、DINOv3/outputs的本地目录，并核查30份相关tar归档索引：未找到任何所需capture。`required_cache_files.csv`列出原629份路径及SHA；`missing_inputs.csv`缺629份（illumination265、view364）。478份异常GT已验证，父冻结汇总元数据可用。**真实跨域分析尚未执行。**

预检生成：required_cache_files.csv、found_cache_files.csv、missing_inputs.csv、archive_index.json、parent_historical_audit.json、input_manifest.json及status.json。`--search-index`可复用此前归档索引，但不会绕过raw SHA校验。`--search-root`可多次指定其他已有目录；`--cache-root`应指向含raw_cache/的cache根目录；`--raw-archive`可多次指定取回的tar。

## 人工取回629份capture

父记录中的旧位置是：
`/kaggle/working/aebad_segment_pair_compat_20261017_091539_501e80/cache`

这只是历史路径，本工具没有连接或确认它仍存在。如已迁移，下面的CACHE改为**真实保留原capture的cache目录**。若raw已删除，应报告缺失，不能拿预测union数组代替，也不能自动补推理。

在保留原缓存的机器上手动执行下列代码，按原manifest只打包629份raw及四份小元数据，保留raw_cache/domain/sample.npz相对结构。NPZ已压缩，因此外层使用tar以减少CPU开销。不包含memory、模型、特征或其余预测缓存。打包前会一次列出缺项并失败；不覆盖已有包。

```bash
python - <<'PY'
import json, tarfile
from collections import Counter
from pathlib import Path

CACHE = Path('/kaggle/working/aebad_segment_pair_compat_20261017_091539_501e80/cache')
OUT = CACHE.parent / 'aebad_illumination_view_support_distance_raw629.tar'
status = json.loads((CACHE / 'cache_status.json').read_text())
assert status['status'] == 'capture_complete'
assert status['identity'] == 'e21f243fba926ea04c4593e1c3472c02f228c1c47d5bccbee01f75b46a45e6bb'
records = status['records']
assert len(records) == 629 and len({(r['domain'], r['sample_id']) for r in records}) == 629
assert Counter(r['domain'] for r in records) == {'illumination': 265, 'view': 364}
names = []
for r in records:
    expected = f"raw_cache/{r['domain']}/{r['sample_id']}.npz"
    assert r['file'] == expected
    names.append(expected)
names += ['cache_status.json', 'input_manifest.json', 'protocol.json', 'capture_verification.csv']
missing = [name for name in names if not (CACHE / name).is_file()]
if missing:
    print('\n'.join(missing))
    raise SystemExit(f'Missing {len(missing)} files; no inference/rebuild attempted')
with tarfile.open(OUT, 'x') as archive:
    for name in names:
        archive.add(CACHE / name, arcname=name, recursive=False)
print('ARCHIVE:', OUT)
print('GiB:', round(OUT.stat().st_size / 2**30, 3))
PY
```

下载该包到本地downloads后，预检和分析直接读包内raw，无需另外展开。原有汇总包仍须保留，因为它提供被冻结的分数、实例和校准状态。也可在原机器上直接指定`--cache-root`进行CPU诊断；另带父汇总、GT、background的小参照文件即可，不需要原服务器绝对路径有效。

## 完整输入到位后的显式CPU运行

下面为手动命令，本轮未执行analyze。示例取回包名对应上面的打包命令。

```powershell
& $python -B -m DINOv3.MADEqual.aebad_s_support_distance_crossdomain.runner preflight `
  --raw-archive G:\codex\workspace\experiment\downloads\aebad_illumination_view_support_distance_raw629.tar

& $python -B -m DINOv3.MADEqual.aebad_s_support_distance_crossdomain.runner analyze `
  --raw-archive G:\codex\workspace\experiment\downloads\aebad_illumination_view_support_distance_raw629.tar `
  --cpu-workers 2
```

每次新建独立输出目录，不覆盖旧结果，不支持将部分分析冒充完成；raw已存在可重复读取，不涉及推理。RAM和输出磁盘分别检查2GiB和3GiB余量，这是运行门槛估计，不是观测峰值。逐图处理，最多保留`--cpu-workers`张图的工作；统计只保留选中patch属性，不保留全域地图。

纯CPU机器也可用同一入口：传`--summary`、`--data`、`--cache-root`或`--raw-archive`、`--background`和`--output-dir`指定本机路径。background汇总只需其status.json、input_manifest.json、conditional_comparisons.csv、matching_coverage.csv、instance_profile_summary.csv，保留原内容与SHA。background缺失不阻断两新域计算，但最终汇总保留缺项。

独立汇总（RESULTS替换为analyze实际输出根目录；不会再跑统计）：

```powershell
$results = 'G:\codex\workspace\experiment\DINOv3\outputs\mad_equal\替换为实际跨域analyze输出目录'
& $python -B -m DINOv3.MADEqual.aebad_s_support_distance_crossdomain.runner summarize --results-root $results
```

## 输出及解释

每域输出objective_report.md、conditional_comparisons.csv、profile_summary.csv、matching_coverage.csv、matching_bins.json、逐patch/实例连接/峰值插值回溯gzip表、instance_profile_analysis.csv、instance_profile_summary.csv、image_index.csv、numerical_verification.csv、input_manifest.json、parent_historical_audit.json、protocol.json、status.json，以及gap2/4/8的加权CDF。NA保留，主描述量和辅助量分别标注。

根目录cross_domain_summary.csv固定36行（3域×4层×3主量），列出方向、区间、bootstrap有效次数、共同覆盖、有效质量、图像权重、b/delta残差及margin分组的inside−ring均值与方向。未完成的新域保留pending；状态不会成为cross_domain_summary_complete。background发现参照不重复分析。

结论需同时回答方向是否重复、是否局限部分域/层、残余混杂、margin关联是否重复及是否足够支持下一步设计。小gap不代表八张绝对距离都小；区分GT覆盖位置不等于识别危险线段；CI不跨0.5不等于完全控制混杂或独立检测收益。illumination/view已用于过方法研究，不能称盲测。没有跨域总平均掩盖反向结果，也不会自动搜索门限。

## 轻量验证

```powershell
& $python -B -m unittest DINOv3.tests.test_aebad_s_support_distance_crossdomain DINOv3.tests.test_aebad_s_support_distance_diagnostic -v
```

新增测试26项，连同background原19项共45项通过（14.816秒，最终联合回归）；7份Python文件AST检查通过。测试实际调用生产NumPy地图、评分、空间和统计接入，覆盖域/支持/层/物理位置、同名样本、SHA篡改、缺八支持、两类容差、坐标派生、独立边界、NA/无共同质量、部分结果不能完成及禁止Torch导入。还用每域一张4×4网格合成缓存跑通入口、分域输出、分布图和pending汇总；该小fixture只替换测试人口数量，不代表真实域验证。真实跨域距离、地图/分数/实例复现、200次全域bootstrap、耗时和峰值资源仍未验证。

本轮仅从已有background表生成了缺项汇总（没有分析新域）：
`G:\codex\workspace\experiment\DINOv3\outputs\mad_equal\aebad_s_support_distance_crossdomain_summarize_20261021_070927_373423`
其中cross_domain_summary.csv为36行，24行新域结果标pending，status为partial_evidence。

## 用户授权后的Kaggle完整重跑

原Kaggle缓存已丢失，用户随后选择**在Kaggle重跑整轮**。为此新增`live_parent.py`和`kaggle.py`，不改变旧capture/预测/评价模块。历史包入口仍严格使用原SHA；显式`--parent-run`读取一轮新完成的cache、prediction、cpu、frozen_inputs和run_config，以新记录的raw SHA和新同批CPU分数/实例进行复核，保留原protocol/catalog、Source资产和提取条件核验。不会改写旧身份，也不要求新NPZ与旧丢失文件有同一字节哈希。两个域不能来自不同父运行。

已生成可直接导入Kaggle的Notebook：
`G:\codex\workspace\experiment\downloads\aebad_illumination_view_full_rerun_diagnostic_20261021.ipynb`

Notebook包含一个执行Cell：新目录git clone → 嵌入的当前代码/少量background参照 → 依赖与Input检查 → capture → predict → 原全像素CPU评价 → 原ADEval AUPRO → 新两域逐支持诊断 → 自动打包。嵌入代码用于本次代码同步，不需要先推送GitHub或另建代码Input。没有内嵌图片、memory或模型。Git commit和代码包SHA另记，既有结果不覆盖。

启用GPU/Internet，挂载原四个Input（AeBAD888、checkPoint-DINOv3、frozen、temp1-pt），在新建或重启后的会话运行全部Cell。保持现有Torch/NumPy；只安装缺少的辅助依赖及adeval==1.1.0，并在子进程检查真实导入。单模型进程使用GPU0，空闲显存准入14GiB保持原batch和float32；双T4显存不合并。CPU四worker、精确像素排序预算显式12GiB，避免此前预算不足的失败。没有改变指标或近似像素计算。

所有模型/GPU阶段都在独立子进程，退出后释放它们的资源；Notebook内核不构造模型。每30秒输出阶段与已产生raw数量；阶段日志和退出码持久化。失败停止后续计算、保留现场并尝试打包已完成缓存，不做自动降级、重采样、重试模型或关机。

结束下载两个包：小的`*_SUCCESS_SUMMARY.tar.gz`与保留629份逐支持raw及复核参照的`*_RAW_AND_REFERENCES.tar`。后者不包括支持特征、Source bank、模型或预测NPZ，包含后续`--parent-run`所需CPU/校准/清单，因此可用于纯CPU重做诊断。若打包空间不足，会明确记录、保留原目录，不能输出EXPERIMENT_SUCCESS=True；失败时小包名为FAILED_SUMMARY。没有默认删除缓存。

新接口的示例（Notebook自动执行，无需手动拆阶段）：

```bash
python -B -u -m DINOv3.MADEqual.aebad_s_support_distance_crossdomain.runner analyze \
  --parent-run /kaggle/working/本次完整运行目录 \
  --data /kaggle/input/datasets/yky2026/aebad888/AeBAD/AeBAD_S \
  --background /kaggle/working/本次完整运行目录/background_reference \
  --output-dir /kaggle/working/本次完整运行目录/diagnostic \
  --cpu-workers 4
```

新增10项Kaggle接入轻量测试覆盖可移植父产物、SHA/MAD篡改、未完成产物、跨父运行混用、输出隔离、子进程日志及失败缓存打包。它们没有执行Kaggle、CUDA或真实模型。此前35项跨域/编排测试与原19项回归共54项通过，随后新增输出隔离用例也通过（Kaggle接入共10项）；本轮真实Kaggle任务尚未启动。

### GitHub直接同步（当前推荐）

用户随后选择GitHub同步，不再要求导入内嵌代码Notebook。`kaggle.main(PROJECT)`现可直接使用clone中的代码，无需overlay或额外代码Input。`background_discovery_catalog.json`只含汇总必需的24行主量比较、8行共同覆盖、54行实例分组及原出处SHA，用作固定的发现阶段参照；不含完整background结果、报告、图片或缓存。catalog本身的SHA与统计内核版本均严格校验，不能随意改数字或混用其他运行。illumination/view仍需真实重跑，不能从该catalog生成新域结果。

在Kaggle新建/重启会话，启用GPU和Internet，挂载原四项Input后运行一个Cell：

```python
import subprocess, sys, uuid
from pathlib import Path

PROJECT = Path('/kaggle/working') / ('DINO-Reference-AD_' + uuid.uuid4().hex[:8])
subprocess.run(['git', '-c', 'http.version=HTTP/1.1', 'clone',
                'https://github.com/YYK001/DINO-Reference-AD.git', str(PROJECT)], check=True)
if any(n == 'DINOv3' or n.startswith('DINOv3.') for n in sys.modules):
    raise RuntimeError('请先重启Kaggle会话，避免复用上一次导入的旧代码。')
sys.path.insert(0, str(PROJECT))
from DINOv3.MADEqual.aebad_s_support_distance_crossdomain.kaggle import main
RUN = main(PROJECT)
```

正式执行时建议在clone后再checkout本次发布commit，以冻结源码。运行顺序、12GiB CPU预算、14GiB模型准入、原精度、心跳、失败保留和两个自动结果包均保持上述定义。没有本地或服务器实际启动行为。
