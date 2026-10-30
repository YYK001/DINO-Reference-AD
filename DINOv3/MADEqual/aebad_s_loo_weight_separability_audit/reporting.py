"""Six-cell per-domain diagnostic matrices; no winner selection or method claim."""
import math
from . import inputs as ip,prediction as pred,evaluation as ev
from .inputs import st
from .protocol import *
from DINOv3.MADEqual.aebad_s_target_loo_layer_weight_p0.composition import METRICS
from DINOv3.MADEqual.aebad_s_fixed_target_graph_p0.evaluation import validate_aupro


def summarize(a):
    p=ip.parent(a);metrics=[];pending=[];completed_AUPRO=0;tables={n:[] for n in ('per_image_scores.csv','image_pair_transitions.csv','layer_pair_contributions.csv','normal_tail_changes.csv','normal_tail_summary.csv','defect_instance_changes.csv','normal_proxy_vs_discrimination.csv','verification.csv','margin_summary.csv')}
    if (a.output_root/'responses_frozen.json').exists():pred.global_freeze(a)
    references=[]
    if not (a.output_root/'responses_frozen.json').exists():pending.append('global_cross_response_freeze')
    for d in DOMAINS:
        for m in SEEDS:
            sig=ip.signature(a,d,m);w=crossed(m);out=ip.unit(a,d,m)
            for source in ('WS',diagonal(m)):
                for r in p.known_metrics(d,m,source):metrics.append(dict(r,memory_seed=m,weight_source=source,weight_seed=None if source=='WS' else m,branch=branch(m,source),diagnostic_only=False,metric_origin='read_only_parent_same_population_map_GT_readout'))
            references.extend(dict(r,memory_seed=m,role='historical_context_not_cross_method') for r in p.settings[d,m]['metrics'])
            cpu=ev.completed(a,d,m,'evaluation') if (a.output_root/'responses_frozen.json').exists() else None
            if cpu:
                rs=ip.parent_reporting.rows(out/'evaluation/metrics.csv');au=out/'aupro.json'
                if au.exists():
                    record=st.read_json(au);maps=ip.parent_reporting.rows(out/'evaluation/map_manifest.csv')
                    validate_aupro(record,dict(**sig,branch=branch(m,w),map_identity=st.digest(maps),nstrips=200,FPR_max=.3))
                    if record['verified_images']!=len(p.samples(d)):raise ValueError('partial crossed AUPRO')
                    completed_AUPRO+=1
                    for r in rs:r['AUPRO']=record['AUPRO']
                else:pending.append(f'{d}/Z{m}/cross_AUPRO')
                metrics.extend(rs)
            else:
                pending.append(f'{d}/Z{m}/cross_CPU')
                metrics.extend(dict(**sig,branch=branch(m,w),scoring=rule,**{k:None for k in METRICS},metric_origin='pending_new_cross') for rule in READOUTS)
            analysis=ev.completed(a,d,m,'analysis') if (a.output_root/'responses_frozen.json').exists() else None
            if analysis:
                for name in tables:
                    if name=='per_image_scores.csv':continue
                    tables[name].extend(dict(r,domain=d,memory_seed=m,compact_base=(out/'analysis').relative_to(a.output_root).as_posix()) for r in ip.parent_reporting.rows(out/'analysis'/name))
            else:pending.append(f'{d}/Z{m}/separability_analysis')
            for row in p.samples(d):
                f=ip.record_path(a,d,m,row)
                if not f.exists():continue
                r=st.sealed_json(f)
                tables['verification.csv'].extend(r['diagonal_verification'])
                for source in WEIGHTS:tables['per_image_scores.csv'].append(dict(domain=d,memory_seed=m,sample_id=row['sample_id'],label=row['label'],weight_source=source,diagnostic_only=source==w,**r['scores'][source]))
    for r in metrics:
        for key in METRICS:
            if r.get(key) in ('',None):continue
            if not math.isfinite(float(r[key])) or not 0<=float(r[key])<=1:raise ValueError('invalid cross matrix metric')
    lookup={(r['domain'],int(r['memory_seed']),r['weight_source'],r['scoring']):r for r in metrics};differences=[]
    for d in DOMAINS:
        for rule in READOUTS:
            for m in SEEDS:
                for new,old in COMPARISONS:
                    for k in METRICS:
                        x,y=(lookup[d,m,w,rule].get(k) for w in (new,old))
                        differences.append(dict(domain=d,fixed='memory',memory_seed=m,scoring=rule,comparison=new+'_minus_'+old,metric=k,delta_pp=None if x in ('',None) or y in ('',None) else 100*(float(x)-float(y))))
            for w in WEIGHTS:
                for k in METRICS:
                    x,y=(lookup[d,m,w,rule].get(k) for m in (43,42))
                    differences.append(dict(domain=d,fixed='weight',weight_source=w,scoring=rule,comparison='Z43_minus_Z42',metric=k,delta_pp=None if x in ('',None) or y in ('',None) else 100*(float(x)-float(y))))
    for name,rows in dict(tables,**{'crossed_metrics.csv':metrics,'fixed_comparisons.csv':differences,'parent_context_metrics.csv':references}).items():st.write_csv(a.output_root/name,rows)
    status=dict(status='pending' if pending else 'weight_cross_and_separability_audit_complete',pending=sorted(set(pending)),unique_common_images=902,new_cross_maps=6,new_AUPRO_complete=completed_AUPRO,metric_rows=len(metrics),new_feature_forwards=0,new_Exact_searches=0,new_weight_fit_calls=0,new_training=0,parameter_search=False,diagnostic_only=True)
    st.write_json(a.output_root/'status.json',status)
    lines=['# Target-LOO：冻结权重交叉与可分性审计','',str(status),'',
        '问题：正常信息何时改变正常—缺陷区分，而不只是改变分数大小？缺项时保持pending；不得从交叉表选择最佳新方法。',
        '两套memory各K8。交叉引入另一组K8正常统计，在本父实验中涉及16张正常图信息，仅作诊断。全部权重三列整体交换，Source/对角指标只读复用。',
        '下面同时保留两图像读出；主读出PL不按域更换。','']
    for d in DOMAINS:
        lines+=['## '+d,'','|memory|权重|读出|Image AUROC %|Image AP %|Pixel AUROC %|Pixel AP %|AUPRO %|','|---|---|---|---:|---:|---:|---:|---:|']
        for r in metrics:
            if r['domain']==d:lines.append('|'+ '|'.join([str(r['memory_seed']),r['weight_source'],r['scoring']]+['pending' if r.get(k) in ('',None) else f'{100*float(r[k]):.5f}' for k in METRICS])+'|')
    lines+=['','## 解释与固定问题','',
        '固定memory的W42/WS、W43/WS、W43/W42，以及固定W的Z43/Z42见fixed_comparisons.csv；保持每域矩阵，不池化重复图像。',
        'image_pair_transitions保留全部正确/错误/精确并列迁移，pair_records存每个图像对的新旧间隔与p/q层贡献；normal/anomaly分数变化分开。贡献解释评分代数，不是AUROC/AP可加贡献；地图最大值不作线性分解。',
        'normal_tail_changes使用同memory的WS固定峰值与确定性Top1%集合。defect_instance_changes使用WS固定实例/环带位置，保留自身峰值与移动；margin改善/下降阈值1e-6，NA不删除，面积保持连续，不搜小缺陷阈值。',
        'normal_proxy_vs_discrimination并列展示正常尺度、u/p/q权重、正常分布、p/q单层AUROC、修复/破坏图像对贡献与固定margin贡献。不能仅凭四层做显著相关或按测试表现挑层。','']
    if not pending:
        lines+=['## 正常降分、排序修复及固定位置缺陷证据','',
            '下列为全部图像对的精确迁移；并列进入/退出另保留在CSV，不用容差合并。间隔贡献不是AUROC/AP的可加贡献。']
        for d in DOMAINS:
            for m in SEEDS:
                for w in ('W42','W43'):
                    comp=w+'_minus_WS';pair=[r for r in tables['image_pair_transitions.csv'] if r['domain']==d and int(r['memory_seed'])==m and r['comparison']==comp]
                    means={r['population']:r for r in pair if r.get('population')}
                    transition={(int(r['old_state']),int(r['new_state'])):int(r['pair_count']) for r in pair if r.get('old_state') not in ('',None)}
                    delta=100*(float(lookup[d,m,w,READOUTS[0]]['image_AUROC'])-float(lookup[d,m,'WS',READOUTS[0]]['image_AUROC']))
                    normal=float(means['normal']['mean_score_change']);anomaly=float(means['anomaly']['mean_score_change'])
                    margins=next(r for r in tables['margin_summary.csv'] if r['domain']==d and int(r['memory_seed'])==m and r['comparison']==comp and r['margin_kind']=='fixed')
                    finding='正常降分同时排序改善' if normal<0 and delta>0 else '正常降分但排序未改善' if normal<0 else '正常平均分数未下降；仍需直接看排序'
                    lines.append(f"- {d}/Z{m}，{comp}：正常/异常平均分数变化 {normal:+.6g}/{anomaly:+.6g}；错误→正确 {transition[-1,1]} 对，正确→错误 {transition[1,-1]} 对；Image AUROC {delta:+.5f} pp。{finding}。固定margin改善/下降/基本不变/NA = {margins['improved']}/{margins['decreased']}/{margins['unchanged']}/{margins['NA']}。")
        for d in DOMAINS:
            lines.append('### '+d+'：固定干预差值（主读出，百分点）')
            for r in differences:
                if r['domain']==d and r['scoring']==READOUTS[0] and r['metric'] in ('image_AUROC','pixel_AP','AUPRO'):
                    lines.append(f"- 固定{r['fixed']} {r.get('memory_seed',r.get('weight_source'))}，{r['comparison']}，{r['metric']}：{r['delta_pp']:+.5f}。")
        lines+=['','## 各层固定margin贡献与跨memory方向','', '|域|权重|层|Z42均值贡献|Z43均值贡献|方向相同（仅描述）|','|---|---|---|---:|---:|---|']
        proxies={(r['domain'],int(r['memory_seed']),r['weight_source'],int(r['layer'])):r for r in tables['normal_proxy_vs_discrimination.csv']}
        for d in DOMAINS:
            for w in ('W42','W43'):
                for layer in LAYERS:
                    values=[proxies[d,m,w,layer]['fixed_margin_layer_contribution_mean'] for m in SEEDS]
                    valid=all(x not in ('',None) for x in values);values=[float(x) for x in values] if valid else values
                    lines.append('|'+ '|'.join([d,w,str(layer),*[f'{x:+.6g}' if x not in ('',None) else 'NA' for x in values],str((values[0]>0)==(values[1]>0) and (values[0]<0)==(values[1]<0)) if valid else 'NA'])+'|')
    lines+=['','若正常降分未伴随更多排序修复或更好定位，只能说明降低正常响应本身不足够。某W跨Z均较差支持其在这两套响应上较差；同W随Z变化则表明作用依赖底层响应。层贡献不证明物理扰动原因。',
        '已观测指标、计算归因与未知原因必须区分；不将失败单独归因为8张支持不足。混合结果保留，不指定唯一机制；不提出新方法、不选最佳交叉、不追加参数或seed。']
    (a.output_root/'objective_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return status
