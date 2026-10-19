# AeBAD-S：冻结K8支持集敏感性

更换真实Target-normal支持，比较完整UNION、原SEGMENT和原PAIR。固定三个域、三个支持seed、三个方法，共27个组合；不增加方法、不调参、不根据GT或指标选支持。五次随机门控实验不能代替本实验。

## 冻结范围

| 域 | 完整正常池 | 三组支持并集 | 共同正常 | 异常 | 共同评价 |
|---|---:|---:|---:|---:|---:|
| background | 93 | 24 | 69 | 212 | 281 |
| illumination | 75 | 24 | 51 | 198 | 249 |
| view | 92 | 24 | 68 | 280 | 348 |

seed42严格继承正式`target_matched`顺序。seed43/44执行原`stable_support_key`的seed参数：CRC32键加路径排序，取八张。完整原测试索引先建立、后过滤，保留原sample_id。每域排除三组支持的并集；不为消除重合而重抽。当前实际清单三组自然不重合，共72张支持、878张共同评价图。

Source保持原Official brightness=[0.8,1.2]、521=455+66、四层各100000条。支持seed不进入Source种子、memory或MAD。查询和Target支持无增强，保留全部patch。特征1280维、640/128切窗、原Exact欧氏距离/1280一次，原几何、PAIR、Source patch MAD、等权四层地图及负响应均不变。主表使用同一冻结Source image MAD的CH1；CURRENT仅作seed42历史审计。

内部距离容差1e-7+1e-6*abs(reference)，内部图像分数1e-5；历史CURRENT分别记录strict1e-5和compatibility1e-4、rtol0、有符号误差。不能把兼容通过写成严格复现。seed43/44的UNION不会拿来对照旧seed42的分数。

## 复用和输入

生产数学直接调用旧`geometry_layer`、跨域`method.predict/pair_gate`、`statistics/ch1_score`、CPU地图重建、精确全像素`exact_binary`、原空间分析和显式`official_aupro_adeval`。旧固定catalog、297/265/364计数及二域macro封装不调用。

原支持模块顶层间接导入Torch。为保持preflight无Torch且不改旧代码，`selection.original_key`只从其AST编译**原样的relative_path/stable_support_key两个纯函数**，显式传seed；不是另写CRC算法，也不执行模块导入。源文件SHA进入身份；若纯函数API改变则阻断。测试另算CRC键核查，生产没有替代采样器。

| 输入 | 本地实际位置/检查 |
|---|---|
| 原正式CSV、memory manifest、patch MAD | `server_archives/aebad_s_target_fewshot_seed42_v1_formal_results/aebad_s_target_fewshot_seed42_v1/` |
| Source image MAD | `DINOv3/outputs/mad_equal/aebad_s_ch1_scoring_local_20261016_034133/image_calibration_state.json` |
| 图片及GT | `datasets/AeBAD.zip`，或解压的AeBAD_S根目录，无train目录前提 |
| 原Source memory | 当前发现`downloads/official_memory.pt`；校验固定SHA，不仅检查存在性 |
| 原权重 | 当前发现`_external/dinov3_checkpoints/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth` |
| 官方代码 | 当前发现`_external/dinov3`；SuperADD为`external_baselines/SuperADD` |
| 新支持/新全域缓存 | 尚未生成；不能把旧seed42缓存或汇总包当成新三组完整输入 |

本次实际只读盘点核对了950张相关RGB和690张异常GT的字节身份，bank与权重SHA均通过。原始支持全量FP32约19.12GiB，未压缩共享capture约21.04GiB、紧凑预测保守估计约6.56GiB，合计约46.72GiB，另留4GiB及CPU排序临时空间。NPZ实际可压缩，不能在准入时预先假定压缩比；不含完整备份包副本。均为存储估计，不是实测资源峰值。

所有路径均可用CLI覆盖。默认路径会优先已有正式路径，再发现上述本地候选；bank/权重必须通过固定SHA。缺少资产只报告，不下载、不重建、不拟合。`input_inventory_local.json`记录本次实际盘点，路径存在不证明CUDA或模型初始化可用。

两域EVAL_SUCCESS_SUMMARY包不是完整检索缓存。已有background完整LOO/SEGMENT档案和本地PAIR预测可用于CPU回归。本工具不自动接纳旧seed42几何/支持特征为新三组捕获；这些资产只能解释旧seed42，不能补出其他支持的Top1。新运行内的完整、同身份逐图缓存优先复用；完整capture之后的predict/evaluate不要求模型、bank或支持特征参与运算。暂不提供任意外部Q缓存导入，避免无来源特征被重新授信。

`background_identity.json`来自已校验的旧background输入快照，固定305张RGB及212张GT的历史字节身份；另外两个域使用已有冻结raster清单，独立于其旧固定计数。preflight读全测试文件名和相关图像头，prepare逐图校验RGB/GT字节并冻结，不解码GT。GT仅在预测冻结后的评价读取。

## 阶段、复用和续跑

默认`preflight`只读，不建实验目录、不导入Torch或初始化CUDA。只有显式`--inventory-output`会写盘点文件，且拒绝覆盖。

1. `prepare`：冻结三组支持、共同评价名单、图像身份、Source/MAD/代码身份，不提特征。已有目录只核对原记录，不重抽。
2. `capture`：首次加载一个原宿主/Source bank。每个唯一支持提取一次，存分层原始FP32 NPY；按图、按层懒载。每张查询通过原完整提取入口一次，Source Exact每层一次，域内24支持各扫描一次。按组原顺序切回8张，原几何/PAIR使用局部1—8编号。不同支持网格允许；不产生完整Q×bank矩阵，不默认持久化Q。
3. `predict`：三个支持组共享距离，CPU原内核生成9份紧凑分支响应，按图完成并冻结域；支持1—8与全局图像映射写入元数据。
4. `evaluate`：按域×seed×方法逐组合重算CH1图像指标和全像素精确CPU指标，正常图零mask包含在内。每组合可续跑。逐图重建地图与冻结分数复核，GT在冻结后进入。
5. `evaluate-aupro`：独立、显式CUDA阶段，ADEval必须1.1.0、200阈值、FPR0.3，无模型。核对CPU地图/GT字节哈希；未运行保持pending，不改口径。
6. `summarize`：生成带时间戳的独立汇总快照，不覆盖旧快照；缺组合/缺AUPRO保持partial_evidence。

`ledger.json`保存原文件SHA及逐图/逐组合记录。启动先校验旧记录，修改过、缺失、孤立或半写文件拒绝重新认可。并行仅用于CPU逐图计算（有界1—8 workers），提交由主线程有序执行。正常中断后重跑同一命令补未完成单元；文件写完但ledger未提交的极窄中断窗口会留下孤立文件，程序保留并明确失败，需人工保留现场后处理，绝不现算哈希接纳。

失败不会覆盖有效输出。历史分数复核失败的当张raw另存`rejected_capture/`，不能续跑授信；不得提高容差后复用。默认不关机。CPU排序临时文件仅在指定work-dir中本工具创建的临时子目录里，退出只清理该临时目录，历史文件不动。

## 手动运行（本轮未执行）

以下供用户在确认资产后的AutoDL终端执行；本轮没有连接服务器。修改路径环境变量即可；同一运行续跑务必继续使用同一OUT，不能每次生成新时间戳。

```bash
export PROJECT=/root/autodl-tmp/DINO-Reference-AD
export ASSETS=/root/autodl-tmp/DINO-NVS
export PYTHON=/root/miniconda3/bin/python
export OUT="$ASSETS/DINOv3/outputs/mad_equal/aebad_s_k8_support_sensitivity_$(date +%Y%m%d_%H%M%S)"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$PROJECT" || exit 1
module=DINOv3.MADEqual.aebad_s_k8_support_sensitivity_p0.runner
common=(--project-root "$PROJECT" --output-root "$OUT"
  --formal-dir "$ASSETS/DINOv3/outputs/mad_equal/aebad_s_target_fewshot_seed42_v1"
  --image-state "$ASSETS/DINOv3/outputs/mad_equal/aebad_s_ch1_scoring_local_20261016_034133/image_calibration_state.json"
  --data /root/autodl-tmp/datasets/AeBAD/AeBAD_S
  --memory "$ASSETS/DINOv3/outputs/mad_equal/aebad_s_target_fewshot_seed42_v1/official_memory.pt"
  --checkpoint "$ASSETS/external_baselines/SuperADD/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"
  --dinov3 /root/autodl-tmp/dinov3 --official "$PROJECT/external_baselines/SuperADD"
  --cpu-workers 8 --device cuda:0 --memory-budget-gib 8
  --work-dir /root/autodl-tmp/k8_support_sensitivity_scratch)

"$PYTHON" -B -u -m "$module" preflight "${common[@]}"
```

预检不是环境/CUDA成功证明。确认输出清单、存储估计和原资产SHA后，手动运行：

```bash
set -o pipefail
for stage in prepare capture predict evaluate evaluate-aupro summarize; do
  printf '\nSTAGE=%s\n' "$stage"
  "$PYTHON" -B -u -m "$module" "$stage" "${common[@]}" \
    2>&1 | tee "${OUT}_${stage}.log"
  rc=${PIPESTATUS[0]}
  printf 'STAGE=%s EXIT_CODE=%s\n' "$stage" "$rc"
  if [ "$rc" -ne 0 ]; then break; fi
done
```

此命令不自动关机。capture默认需空闲20GiB显存；参数只控制准入，不改batch/dtype。更低门槛必须显式指定，仍可能OOM，不自动降级。资源估计是完整存储估算，不是运行时承诺。

分域及续跑示例（组/共同评价集已由prepare联合冻结，不会因分批重选）：

```bash
"$PYTHON" -B -u -m "$module" capture "${common[@]}" --domains background
"$PYTHON" -B -u -m "$module" predict "${common[@]}" --domains background
"$PYTHON" -B -u -m "$module" evaluate "${common[@]}" --domains background
# 以后同一OUT对illumination/view重复这三个阶段；可重跑完整阶段，自动补未完成单元。
"$PYTHON" -B -u -m "$module" evaluate-aupro "${common[@]}"
"$PYTHON" -B -u -m "$module" summarize "${common[@]}"
```

打包命令（执行前自行确认空间；不删除原件）：

```bash
# 汇总、冻结清单和ledger；此review包不含完整raw，不能冒充可续跑缓存。
tar -czf "${OUT}_review.tar.gz" -C "$OUT" \
  summaries frozen.json protocol.json input_inventory.json implementation_audit.md \
  support_manifest.csv support_candidates.csv support_overlap.csv common_evaluation_manifest.csv excluded_images.csv ledger.json
# 如需完整可续跑备份，单独打包本实验全部文件；不要混入其他实验。
tar -czf "${OUT}_full.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
```

## 输出与解释

根目录为冻结清单、ledger、逐图capture/预测、每组合CPU/AUPRO结果与调用成本。`summaries/<时间>/`包含metrics、三组配对差值、两级macro/支持波动（均值、ddof=1样本标准差、范围、正零负次数）、逐图分数/排序、R_N/R_A/G、实例margin/修复与新增损失、小缺陷及NA数量、历史共同子集CURRENT指标/排序审计、客观报告。

先每seed对三域等权，再汇总三个支持重复；也保留逐域统计。不能池化图像代替域macro。小缺陷为每域原GT面积最小ceil(N/4)，跨三seed核对同一实例分组。固定UNION峰值按同seed定义，不能说跨seed固定物理位置相同。定位收益与margin损失并列报告，不预填结论。

本地测试命令：

```powershell
& 'D:\anaconda\miniconda\envs\chinamoblie\python.exe' -B -m unittest DINOv3.tests.test_aebad_s_k8_support_sensitivity -v
```

验证范围和实际结果见implementation_audit.md；NumPy小构造数据中的Exact/Torch边界替身不代表真实Torch/cdist、CUDA、DINOv3或27组合指标已通过。
