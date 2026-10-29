import math
import numpy as np
from . import inputs as ip,pipeline as pp
from .inputs import st
from .protocol import *
from .evaluation import metric_branches
from DINOv3.MADEqual.aebad_s_target_loo_layer_weight_p0.composition import comparisons,METRICS
from DINOv3.MADEqual.aebad_s_fixed_target_graph_p0.evaluation import validate_aupro


def rows(path):return ip.layer_inputs.old.read_csv(path)


def final_metrics(a):
    sig=ip.signature(a);base=rows(ip.setting(a)/'evaluation/metrics.csv');maps=rows(ip.setting(a)/'evaluation/map_manifest.csv');n=len(st.sealed_json(a.run_root/'common_manifest.json')['domains'][a.domain]['common']);pending=[]
    for b in MAPS:
        path=ip.setting(a)/'aupro'/(b+'.json')
        if not path.exists():pending.append(b);continue
        r=st.read_json(path);validate_aupro(r,dict(**sig,branch=b,map_identity=st.digest([x for x in maps if x['branch']==b]),nstrips=200,FPR_max=.3))
        if r['verified_images']!=n:raise ValueError('AUPRO common image population differs')
        for x in base:
            if x['branch']==b:x['AUPRO']=r['AUPRO']
    result=metric_branches(base,a.domain,a.support_seed)
    for x in result:
        if any(x.get(k) in ('',None) for k in METRICS):pending.append('metrics:'+x['branch'])
        elif any(not math.isfinite(float(x[k])) or not 0<=float(x[k])<=1 for k in METRICS):raise ValueError('invalid metric')
    return result,pending


def summarize(a):
    from .evaluation import restored
    sig=ip.signature(a);pending=[];metrics=[]
    if restored(a):metrics,pending=final_metrics(a)
    else:pending=['common_CPU_evaluation']
    diff=comparisons(metrics,a.domain) if metrics else []
    diff=[dict(x,support_seed=a.support_seed) for x in diff]
    if len(diff)!=60:pending.append('complete_fixed_comparisons')
    st.write_csv(ip.setting(a)/'metrics.csv',metrics);st.write_csv(ip.setting(a)/'metric_comparisons.csv',diff)
    value=dict(**sig,status='pending' if pending else 'common_five_metrics_complete',pending=sorted(set(pending)),images=len(ip.plan(a)['domains'][a.domain]['common']),independent_map_jobs=4,Source_seed=42)
    st.write_json(ip.setting(a)/'status.json',value)
    lines=[f'# 两组K8共同评价：{a.domain}/seed{a.support_seed}','',str(value),'',
        '旧seed42/43指标不导入本表；四张独立地图在共同名单上重评。原/组合CH1、CURRENT，新加权PL、MAP_MAX；组合图像来自同seed SEGMENT。','',
        '|分支|读出|Image AUROC %|Image AP %|Pixel AUROC %|Pixel AP %|AUPRO %|','|---|---|---:|---:|---:|---:|---:|']
    for x in metrics:lines.append('|'+ '|'.join([x['branch'],x['scoring']]+['pending' if x.get(k) in ('',None) else f'{100*float(x[k]):.5f}' for k in METRICS])+'|')
    lines+=['','正常统计与权重见同目录；固定比较见metric_comparisons.csv。两组支持不是两组独立图像实验。不选较好seed或修改规则。']
    (ip.setting(a)/'objective_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return value


def mean_groups(groups,label):
    indexes=[{(r['branch'],r['scoring']):r for r in rs} for rs in groups]
    keys=set(indexes[0])
    if any(set(x)!=keys for x in indexes):raise ValueError('macro keys incomplete')
    return [dict(domain=label,branch=b,scoring=sc,aggregation='equal_domain_then_equal_support_seed',**{k:float(np.mean([float(x[b,sc][k]) for x in indexes],dtype=np.float64)) for k in METRICS}) for b,sc in sorted(keys)]


def read_complete(a):
    sig=ip.signature(a);path=ip.setting(a);state=st.read_json(path/'status.json');cpu=st.sealed_json(path/'evaluation/status.json');pred=st.sealed_json(path/'predictions_frozen.json');weights=st.sealed_json(path/'weights.json')
    n=len(st.sealed_json(a.run_root/'common_manifest.json')['domains'][a.domain]['common'])
    if state['status']!='common_five_metrics_complete' or state['pending'] or any(x.get(k)!=v for x in (state,cpu,pred,weights) for k,v in sig.items()) or cpu['prediction_seal']!=pred['seal'] or pred['weights_seal']!=weights['seal'] or cpu['images']!=n or pred['images']!=n:raise ValueError('setting incomplete or mismatched')
    expected=[r['sample_id'] for r in st.sealed_json(a.run_root/'common_manifest.json')['domains'][a.domain]['common']]
    if [r['sample_id'] for r in pred['records']]!=expected:raise ValueError('wrong common sample order')
    for name,h in cpu['files'].items():
        if st.sha(path/'evaluation'/name)!=h:raise ValueError('CPU result modified')
    actual,pending=final_metrics(a)
    if pending:raise ValueError('AUPRO/metric pending')
    frozen=rows(path/'metrics.csv');index={(r['branch'],r['scoring']):r for r in actual}
    if len(frozen)!=14 or {(r['branch'],r['scoring']) for r in frozen}!=set(index) or any(float(r[k])!=float(index[r['branch'],r['scoring']][k]) for r in frozen for k in METRICS):raise ValueError('summarized metrics differ')
    return frozen,weights


def aggregate(a):
    old=(a.domain,a.support_seed);groups={};pending=[];allrows=[];weight_rows=[];concentration=[];common_gt={};source_weights=None
    for d in DOMAINS:
        for seed in SEEDS:
            a.domain=d;a.support_seed=seed;key=f'{d}/seed{seed}'
            state=ip.setting(a)/'status.json'
            if not state.exists() or st.read_json(state).get('status')=='pending':pending.append(key);continue
            metric,w=read_complete(a)
            if source_weights is None:source_weights=w['weights'][SOURCE]
            elif w['weights'][SOURCE]!=source_weights:raise ValueError('Source gamma0 weights differ across six settings')
            maps=rows(ip.setting(a)/'evaluation/map_manifest.csv')
            gt=[(r['sample_id'],r['mask_uint8_sha256'],r['output_height'],r['output_width']) for r in maps if r['branch']==U]
            if d in common_gt and common_gt[d]!=gt:raise ValueError('two support sets do not share identical GT/population/order')
            common_gt[d]=gt;groups[d,seed]=metric;allrows.extend(metric)
            weight_rows.extend(rows(ip.setting(a)/'weights.csv'));concentration.extend(rows(ip.setting(a)/'u_concentration.csv'))
    a.domain,a.support_seed=old
    seed_macro=[];domain_means=[];overall=[];diff=[];sensitivity=[]
    for (d,seed),r in groups.items():diff.extend(dict(x,support_seed=seed) for x in comparisons(r,d))
    for seed in SEEDS:
        if all((d,seed) in groups for d in DOMAINS):seed_macro.extend(dict(r,support_seed=seed) for r in mean_groups([groups[d,seed] for d in DOMAINS],f'seed{seed}_domain_macro'))
    for d in DOMAINS:
        if all((d,s) in groups for s in SEEDS):domain_means.extend(mean_groups([groups[d,s] for s in SEEDS],d+'_two_seed_mean'))
    if len(groups)==6:
        overall=mean_groups([[r for r in seed_macro if r['support_seed']==s] for s in SEEDS],'overall_equal_domain_then_seed')
        for d in DOMAINS:
            for rule in READOUTS:
                indexes={s:{(r['branch'],r['scoring']):r for r in groups[d,s]} for s in SEEDS}
                for k in METRICS:
                    delta={s:100*(float(indexes[s][TARGET,rule][k])-float(indexes[s][SOURCE,rule][k])) for s in SEEDS}
                    sensitivity.append(dict(domain=d,scoring=rule,metric=k,seed42_delta_Target_pp=delta[42],seed43_delta_Target_pp=delta[43],seed43_minus_seed42_delta_pp=delta[43]-delta[42]))
    for seed in SEEDS:
        selected=[r for r in seed_macro if r['support_seed']==seed]
        if selected:diff.extend(dict(r,support_seed=seed) for r in comparisons(selected,selected[0]['domain']))
    for d in DOMAINS:
        selected=[r for r in domain_means if r['domain']==d+'_two_seed_mean']
        if selected:diff.extend(dict(r,support_seed='mean') for r in comparisons(selected,selected[0]['domain']))
    if overall:diff.extend(dict(r,support_seed='mean') for r in comparisons(overall,'overall_equal_domain_then_seed'))
    out=a.run_root/'summary';out.mkdir(parents=True,exist_ok=True)
    weight_changes=[]
    wi={(r['domain'],int(r['support_seed']),r['branch'],r['layer'],r['statistic']):r for r in weight_rows}
    for (d,seed,b,l,s),r in wi.items():
        other=wi.get((d,43,b,l,s))
        if seed==42 and other:
            weight_changes.append(dict(domain=d,branch=b,layer=l,statistic=s,**{k+'_seed43_minus_seed42':float(other[k])-float(r[k]) for k in ('Source_scale','Target_LOO_scale','mixed_scale','weight')}))
    costs=[dict(st.read_json(p),record=p.relative_to(a.run_root).as_posix()) for p in sorted(a.run_root.glob('*/seed*/costs/*.json'))]
    cost_summary=dict(recorded_stage_seconds=sum(r['seconds'] for r in costs),feature_forwards=sum(r.get('feature_forwards',0) for r in costs),Exact_searches=sum(r.get('exact_execution',{}).get('searches',0) for r in costs),
        peak_RAM_bytes=max([r['process_peak_rss_bytes'] for r in costs if 'process_peak_rss_bytes' in r],default=None),peak_cuda_allocated_bytes=max([r['peak_cuda_allocated_bytes'] for r in costs if 'peak_cuda_allocated_bytes' in r],default=None),
        peak_cuda_reserved_bytes=max([r['peak_cuda_reserved_bytes'] for r in costs if 'peak_cuda_reserved_bytes' in r],default=None),includes_failed_and_resumed_stage_work=True,unmeasured_resources_are_not_zero=True)
    st.write_json(out/'cost_summary.json',cost_summary)
    tables={'domain_seed_metrics.csv':allrows,'per_seed_domain_macro.csv':seed_macro,'per_domain_two_seed_mean.csv':domain_means,'overall_macro.csv':overall,'metric_comparisons.csv':diff,'Target_information_seed_sensitivity.csv':sensitivity,'weights.csv':weight_rows,'weight_seed_changes.csv':weight_changes,'u_concentration.csv':concentration,'costs.csv':costs}
    for name,r in tables.items():st.write_csv(out/name,r)
    manifest=st.sealed_json(a.run_root/'common_manifest.json')
    result=dict(status='pending' if pending else 'two_K8_common_validation_complete',pending=pending,unique_common_images=manifest['total_common_images'],repeated_prediction_instances=2*manifest['total_common_images'],independent_data_repetitions=False,complete_settings=len(groups),complete_AUPRO_jobs=len(groups)*4,numerical_failed_checks=0 if not pending else None,Source_seed=42,support_seeds=[42,43],parameter_search=False)
    st.write_json(out/'status.json',result)
    lines=['# Target-LOO两组K8共同评价复验','',str(result),'',
        '相同共同图像分别评价两组支持；不是双倍独立图像。历史926/878结果不能直接作本轮因果差值。Source种子固定42，支持seed仅改变K8。',
        '每个seed先对三域等权，再对两seed等权；未池化图像。完整14行/设置与两读出保留，组合只继承同seed SEGMENT评分。','']
    for label,r in [('domain_seed',allrows),('per_seed_macro',seed_macro),('domain_means',domain_means),('overall',overall)]:
        lines+=['## '+label,'','|域|支持seed|分支|读出|Image AUROC %|Image AP %|Pixel AUROC %|Pixel AP %|AUPRO %|','|---|---|---|---|---:|---:|---:|---:|---:|']
        for x in r:lines.append('|'+ '|'.join([x['domain'],str(x.get('support_seed','mean')),x['branch'],x['scoring']]+[f'{100*float(x[k]):.5f}' for k in METRICS])+'|')
    lines+=['','## 固定问题：Target信息相对Source信息的支持集敏感性','', '|域|读出|指标|seed42差值pp|seed43差值pp|两差值之差pp|','|---|---|---|---:|---:|---:|']
    for r in sensitivity:lines.append('|'+ '|'.join([r['domain'],r['scoring'],r['metric']]+[f'{r[k]:+.5f}' for k in ('seed42_delta_Target_pp','seed43_delta_Target_pp','seed43_minus_seed42_delta_pp')])+'|')
    if len(groups)==6:
        lines+=['','## 固定问题的描述性回答','', '下列均为Target统计相对Source统计的百分点差；收益/退化按每项指标分别判断。']
        for d in DOMAINS:
            for seed in SEEDS:
                idx={(r['branch'],r['scoring']):r for r in groups[d,seed]};rule=READOUTS[0]
                changes=[100*(float(idx[TARGET,rule][k])-float(idx[SOURCE,rule][k])) for k in METRICS]
                lines.append(f"- {d}/seed{seed}：主读出Image AUROC/AP {changes[0]:+.5f}/{changes[1]:+.5f}；Pixel AUROC/AP/AUPRO {changes[2]:+.5f}/{changes[3]:+.5f}/{changes[4]:+.5f}。")
        lines+=['','跨seed敏感程度见上表的两差值之差；方向相同也不等于普遍稳定，方向变化也不能锁定唯一失败机制。']
    lines+=['','缺项时上述结论pending。两支持组不足以证明普遍稳定；统计集中度不是性能贡献，不据此删支持。不挑选seed，不增加44，不搜索参数。']
    (out/'objective_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result
