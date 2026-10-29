"""Seven fixed branches, two new maps; no inference or GT-driven selection."""
import math
from . import inputs as ip
from .inputs import st
from .protocol import NEW
from .composition import fixed_combinations,comparisons,METRICS


def summarize(a):
    from .evaluation import restored
    pending=[];rows=[]
    if restored(a):rows=ip.old.read_csv(a.run_root/'evaluation/metrics.csv')
    else:pending.append('CPU_evaluation')
    for r in rows:
        if r['domain']!=ip.domain(a):raise ValueError('another domain in metrics')
        if r['branch'] in NEW:
            path=a.run_root/'aupro'/(r['branch']+'.json')
            if not path.exists():pending.append('AUPRO:'+r['branch']);continue
            from DINOv3.MADEqual.aebad_s_fixed_target_graph_p0.evaluation import validate_aupro
            maps=[v for v in ip.old.read_csv(a.run_root/'evaluation/map_manifest.csv') if v['branch']==r['branch']]
            v=st.read_json(path)
            validate_aupro(v,dict(identity=ip.identity(a),domain=ip.domain(a),branch=r['branch'],map_identity=st.digest(maps),nstrips=200,FPR_max=.3))
            if v['verified_images']!=ip.count(a):raise ValueError('incomplete domain AUPRO')
            r['AUPRO']=v['AUPRO']
    if rows:rows=fixed_combinations(rows,ip.domain(a))
    if len(rows)!=14:pending.append('fourteen_metric_rows')
    for r in rows:
        for k in METRICS:
            if r.get(k) in ('',None):pending.append(r['branch']+':'+k)
            elif not math.isfinite(float(r[k])) or not 0<=float(r[k])<=1:raise ValueError('invalid domain metric')
    diff=comparisons(rows,ip.domain(a)) if rows else []
    if len(diff)!=60:pending.append('sixty_fixed_comparisons')
    st.write_csv(a.run_root/'metrics.csv',rows);st.write_csv(a.run_root/'metric_comparisons.csv',diff)
    result=dict(status='pending' if pending else 'all_five_metrics_complete',identity=ip.identity(a),domain=ip.domain(a),pending=sorted(set(pending)),
        evaluation_images=ip.count(a),metric_rows=len(rows),new_AUPRO_jobs=2,combinations_new_map_jobs=0,Source_statistics='read_only_background_import',training=False,parameter_search=False)
    st.write_json(a.run_root/'status.json',result)
    lines=[f'# Target-LOO固定跨域验证：{ip.domain(a)}','',str(result),'',
        '原参照与组合：CH1主、CURRENT补充；独立加权分支：PEAK_LOCAL_WEIGHT主、WEIGHTED_MAP_MAX补充。读出不可混称。',
        '组合图像指标完全继承SEGMENT；两组合各引用一张已有加权图，不产生额外像素/AUPRO作业。不能把继承的图像收益归因于层加权。','',
        '|分支|读出|Image AUROC %|Image AP %|Pixel AUROC %|Pixel AP %|AUPRO %|','|---|---|---:|---:|---:|---:|---:|']
    for r in rows:lines.append('|'+ '|'.join([r['branch'],r['scoring']]+['pending' if r.get(k) in ('',None) else f'{100*float(r[k]):.5f}' for k in METRICS])+'|')
    lines+=['','## 固定比较（百分点）','','|候选|读出|参照|参照读出|指标|差值|','|---|---|---|---|---|---:|']
    for r in diff:lines.append('|'+ '|'.join([r['branch'],r['scoring'],r['reference'],r['reference_scoring'],r['metric'],'pending' if r['delta_pp'] is None else f'{r["delta_pp"]:+.5f}'])+'|')
    lines+=['','实际尺度/权重和下限见weights.csv；Source导入见source_reuse.json。统计仅来自Source66和本域8支持，未调参数。',
        '全部域均有开发历史，不是严格盲测；未验证跨支持集稳定性。单域正向不能替代宏平均和反向结果。',
        '本轮不新增实例分析，不承诺每实例零损失。保持所有下降，不按域选择地图或评分。']
    (a.run_root/'objective_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result
