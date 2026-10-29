import math
from . import inputs as ip
from .inputs import st
from .protocol import NEW,TARGET,SOURCE,U,S,COMBINED


def summarize(a):
    from .evaluation import restored
    pending=[];rows=[]
    if restored(a):rows=ip.old.read_csv(a.run_root/'evaluation/metrics.csv')
    else:pending.append('CPU_evaluation')
    for r in rows:
        if r['branch'] in NEW:
            path=a.run_root/'aupro'/(r['branch']+'.json')
            if not path.exists():pending.append('AUPRO:'+r['branch']);continue
            v=st.read_json(path)
            from DINOv3.MADEqual.aebad_s_fixed_target_graph_p0.evaluation import validate_aupro
            maps=[x for x in ip.old.read_csv(a.run_root/'evaluation/map_manifest.csv') if x['branch']==r['branch']]
            validate_aupro(v,dict(identity=ip.identity(a),domain='background',branch=r['branch'],map_identity=st.digest(maps),nstrips=200,FPR_max=.3))
            if v['verified_images']!=297:raise ValueError('incomplete AUPRO')
            r['AUPRO']=v['AUPRO']
    if len(rows)!=10:pending.append('ten_metric_rows')
    keys=('image_AUROC','image_AP','pixel_AUROC','pixel_AP','AUPRO')
    for r in rows:
        for k in keys:
            if r.get(k) in ('',None):pending.append(r['branch']+':'+k)
            elif not math.isfinite(float(r[k])) or not 0<=float(r[k])<=1:raise ValueError('invalid metric')
    st.write_csv(a.run_root/'metrics.csv',rows)
    lookup={(r['branch'],r['scoring']):r for r in rows};comparisons=[]
    for b,rule,ref,rr in [(SOURCE,'PEAK_LOCAL_WEIGHT',U,'CH1'),(SOURCE,'WEIGHTED_MAP_MAX',U,'CURRENT'),(TARGET,'PEAK_LOCAL_WEIGHT',SOURCE,'PEAK_LOCAL_WEIGHT'),(TARGET,'WEIGHTED_MAP_MAX',SOURCE,'WEIGHTED_MAP_MAX'),(TARGET,'PEAK_LOCAL_WEIGHT',U,'CH1'),(TARGET,'PEAK_LOCAL_WEIGHT',COMBINED,'CH1')]:
        if (b,rule) not in lookup or (ref,rr) not in lookup:continue
        for k in keys:
            x,y=lookup[b,rule].get(k),lookup[ref,rr].get(k)
            comparisons.append(dict(branch=b,scoring=rule,reference=ref,reference_scoring=rr,metric=k,delta_pp=None if x in ('',None) or y in ('',None) else 100*(float(x)-float(y))))
    st.write_csv(a.run_root/'metric_comparisons.csv',comparisons)
    value=dict(status='pending' if pending else 'all_five_metrics_complete',identity=ip.identity(a),pending=sorted(set(pending)),evaluation_images=297,new_AUPRO_jobs=2,training=False,parameter_search=False)
    st.write_json(a.run_root/'status.json',value)
    lines=['# Target-LOO正常统计层加权固定验证','',str(value),'','原参照：CH1/CURRENT；新分支：PEAK_LOCAL_WEIGHT/WEIGHTED_MAP_MAX。新PL读出不套image MAD；两读出共享一张定位图。',
        'γ=0是Source新加权，不是原UNION/CH1；γ=0.5为预先固定经验选择，不声称最优。','', '|分支|读出|Image AUROC|Image AP|Pixel AUROC|Pixel AP|AUPRO|','|---|---|---:|---:|---:|---:|---:|']
    for r in rows:lines.append('|'+ '|'.join([r['branch'],r['scoring']]+['pending' if r.get(k) in ('',None) else f'{100*float(r[k]):.5f}' for k in keys])+'|')
    lines+=['','固定差值见metric_comparisons.csv；三种统计的四层实际尺度、权重及下限触发见weights.csv。',
        '仅按完整五项指标判断，不因正常分数下降或单项上升宣称有效。γ=0.5与γ=0接近只能说明本配置未观察到Target统计额外收益。',
        '单background开发域、单支持集，不证明跨域/跨种子稳定；Target7尺度不保证大于Target8。不进行参数搜索。','实例分析本轮未请求；不以跨标度margin替代检测指标。']
    (a.run_root/'objective_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return value
