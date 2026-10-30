# 权重交叉与可分性实现审计

2026-10-30。只新增独立工具和CPU测试，不修改原层加权数学、两组K8实现或历史报告。未连接服务器、未初始化CUDA、无真实特征前向/Exact/训练/权重拟合。代码发布已获用户随后明确授权，不包含本地预检、数据、权重、缓存或结果。

## 阅读及复用证据

- 原902张共同结果、summary报告/权重/变化表、common manifest、各组统计与权重、逐图预测/地图身份、local_review，以及窗口39—41节已读。
- 公式：原 `numerics.weighted`、`from_z`、FP32 `bilinear`；不重复MAD、除1280或image MAD，不重归一化已冻结权重。
- 空间：直接导入原 `union_diagnostics.spatial.components/instance_regions`，8连通、16原始像素Chebyshev环带、去全部缺陷与原resize。margin分类1e-6，评分/分解原1e-5核查不放宽。
- 指标：原 `image_metrics`/精确并列、`exact_pixel_result`（范围/像素数及独立中间封存）、原ADEval1.1.0/200/.3。原12格完整指标只读复用；六个新交叉各一项CPU/AUPRO。

## 关键实现

1. Source seed42与memory seed42/43及weight source WS/W42/W43分别标识。每份权重三列整体交换；metadata保留另一K8统计导致的16张Target正常信息，诊断格不当作新K8方法。
2. 只读父baseline_origin定位标准化Z和positions。Source bank/Q/checkpoint不接入。导出只复制原manifest和两个标量NPY；迁移路径独立，旧seal和metadata不改。
3. 全部无GT三权重响应在六个设置冻结后才开放GT smoke/evaluate/analyze。Source/对角地图和评分先复现；原对角排序采用实际冻结分数，不以数值容差重定义并列。
4. 所有正常—异常对使用精确符号枚举九种迁移，紧凑记录保留新旧间隔、八项有效FP32权重下的FP64 p/q贡献与舍入残差。原AUROC由对正确率独立核对；不是把贡献称作AUROC/AP分解。地图最大值无此线性分解。
5. 正常响应固定WS峰值/Top1%，确定性处理并列，图像等权汇总。实例/环带固定WS位置与各自峰值分别保存，原始面积连续散点，不调分界。不可计算状态与无有效组均保留NA，空组贡献均值不填0。
6. 正常代理表72行，四层全保留：冻结尺度/权重、实际正常u/p/q分布、p/q单层AUROC、排序修复/破坏贡献、固定margin贡献。用已封存统计小型核对尺度，未重新学习或拟合权重。
7. 所有新状态/缓存在独立输出根；同输入按图/设置/作业续跑。成本计数不进不可变预测身份；原全像素中间结果未封存或被修改时拒绝，不重新盖章。review保留紧凑图像对数组、排除大Z/地图/scratch。

## 全部检查类别及资源策略

父完成/依赖/共同902名单与支持并集排除；Source66统计与Source权重共享；Target统计、权重来源、manifest seal和原始图像身份；FP32四层响应/完整int64 positions/原输出几何；原对角复现；无GT全响应冻结；地图与GT重放身份；精确并列与AUROC；1e-5分解残差/1e-6 margin计数/NA；中间指标封存及范围/像素数；输出、导出、scratch隔离；用户规定的12h预算。

上述为本工具的检查，不含容量准入、全盘搜索、重复大数组SHA扫描、新精度后端或自定义Triton。新增小数组写出仍使用原项目身份；fast复用检查小型封存记录及数组布局。旧Source、权重、名单、指标、报告、完成状态未修改。

## 当前边界

本地父review元数据可用，GT档案索引可用，但1804份四层Z/positions缺失。精确清单和pending矩阵在本地预检目录，未进行实际六项交叉评价或GT归因。本地测试采用小型标量特征/GT夹具及真实父元数据；不把夹具算作真实实验或GPU验证。

真实缓存齐全后才可做固定样本smoke并外推。已知父AUPRO时长只作参考，新AUPRO硬件/后端吞吐未实测；未知耗时不填0、不承诺固定几分钟。预计超过12h须先获同意，运行中可在提交单位暂停。完成后只解释冻结交叉结果，不扫描新权重、层、阈值或新方法。

测试与语法检查的实际数量/耗时见 `validation_results.json`，日志见 `cpu_test_results.txt`。

最终本地验证：72项CPU测试通过（新增16、回归56，25.49秒，无跳过），11份Python源码AST及Bash语法检查通过；Torch未导入。真实六个交叉、GT归因与原GPU AUPRO尚未运行。
