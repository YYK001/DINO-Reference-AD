"""Image-only conclusions; auxiliary controls never replace the primary."""
from . import inputs as ip
from .protocol import BASE,SOURCE,SINGLE,PRIMARY
st=ip.st

def report(a):
    result=st.sealed_json(a.run_root/'evaluation.json')
    if result['identity']!=ip.identity(a):raise ValueError('evaluation identity differs')
    values={r['branch']:r for r in result['metrics']}
    execution=ip.execution(a)
    lines=['# 固定外围几何与中心次序证据',
        'background / seed42 / 原K8；289张共同评价（77正常、212异常）。只新增Image AUROC/AP，定位能力继承。',
        f"计算后端：{execution['backend']}；检索：{execution['retrieval']['backend']}。",
        'CUDA桶索引是新的近似后端，候选不承诺与历史OpenCV一致；所有新增E和校准统一使用本轮后端。' if execution['backend']=='cuda' else '显式CPU参考路径。',
        '', '| 分支 | Image AUROC | Image AP |', '|---|---:|---:|']
    for r in result['metrics']:lines.append(f'| {r["branch"]} | {r["image_AUROC"]:.8f} | {r["image_AP"]:.8f} |')
    lines += ['',f'预定主候选始终为 {PRIMARY}。以下差值按 AUROC、AP 分别判断，不合并为单一成功标签。']
    for label,first,second in (('主候选相对SEGMENT',PRIMARY,BASE),('多候选相对单候选',PRIMARY,SINGLE),('Source+K8相对Source',PRIMARY,SOURCE)):
        lines.append(f'- {label}：AUROC差 {values[first]["image_AUROC"]-values[second]["image_AUROC"]:+.8f}；AP差 {values[first]["image_AP"]-values[second]["image_AP"]:+.8f}。')
    lines += ['', '证据覆盖（只在全部分数冻结后按标签汇总）：']
    for r in result['coverage']:lines.append(f'- {r["group"]}：{r["images"]}张；有主分支证据{r["multi_images_with_evidence"]}张；有效窗口{r["multi_valid_windows"]}/{r["windows"]}。')
    lines += ['', '缺席比较不表示已判断正常；稀少的有效窗口不能证明缺陷保持。近似LSH未告警不能证明检索完备。',
        '若主候选未改善SEGMENT，不能用辅助分支较好改称主方法成功；多候选或Target对照没有额外收益时也不作相应有效性声明。',
        '本轮无训练、无参数搜索，不按测试结果改变窗口、几何门槛、校准尺度或主分支。',
        '排序对修复/破坏见 ranking_pair_changes.csv；这些对共享图像，不是独立观测。',
        '单域、单支持集结果不能支持跨域稳定或SOTA结论。无新增定位收益。',
        '工程未完成、证据覆盖不足和正式指标无收益须分别解释；不由一项外推另一项。',
        '', '数值：灰度、采样和证据以FP32/整数运算为主；小型仿射求解及66张统计采用库原生FP64。',
        '严格单调不变性仅限固定整数对应且无量化变化；双线性采样不保证该性质。',
        '成本、冷启动与不确定项见 cost_estimate.json、gpu_image_timings.json（CUDA路径）和 costs/；重叠阶段不能相加充当总时长。索引、逐窗口mask保存在独立cache，不进入review包。']
    (a.run_root/'objective_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    status=dict(status='image_metrics_complete',success=True,pending=[],evaluation_images=289,Source_calibration=66,
        primary=PRIMARY,branches=7,image_metrics=2,new_pixel_jobs=0,new_AUPRO_jobs=0,training=False,
        backend=execution['backend'],retrieval_backend=execution['retrieval']['backend'],
        parameter_search=False,cross_domain_stability='not evaluated',localization='inherited original strong reference')
    st.write_json(a.run_root/'status.json',status);return status
