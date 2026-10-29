"""Read-only completed runs; compute metrics per domain then equal-domain means."""
from pathlib import Path
import numpy as np
from .inputs import st,old
from .protocol import COUNTS,NEW,SOURCE,config
from .composition import METRICS,fixed_combinations,comparisons


def source_contract(inputs):
    return {k:inputs[k] for k in ('source','source_calibration','extraction','mad')}


def completed(path,domain):
    path=Path(path);status=st.read_json(path/'status.json');inputs=st.sealed_json(path/'inputs.json')
    if status['status']!='all_five_metrics_complete' or status['pending'] or status['identity']!=inputs['identity'] or inputs['config']!=config(domain):raise ValueError('completed run with exact domain/config required: '+domain)
    cpu=st.sealed_json(path/'evaluation/status.json');pred=st.sealed_json(path/'predictions_frozen.json');weights=st.sealed_json(path/'weights.json');normal=st.sealed_json(path/'normal_statistics.json')
    n=COUNTS[domain][0]
    if cpu['identity']!=inputs['identity'] or cpu['images']!=n or pred['images']!=n or cpu['prediction_seal']!=pred['seal'] or pred['identity']!=inputs['identity'] or pred['weights_seal']!=weights['seal'] or len(pred['records'])!=n or weights['normal_statistics_seal']!=normal['seal'] or normal['identity']!=inputs['identity'] or weights['identity']!=inputs['identity']:raise ValueError('completed dependency chain differs')
    if len({r['sample_id'] for r in pred['records']})!=n or any(r.get('domain',domain)!=domain for r in pred['records']):raise ValueError('prediction domain/population differs')
    for name,sha in cpu['files'].items():
        if st.sha(path/'evaluation'/name)!=sha:raise ValueError('modified small evaluation result: '+name)
    rows=old.read_csv(path/'evaluation/metrics.csv');maps=old.read_csv(path/'evaluation/map_manifest.csv')
    from DINOv3.MADEqual.aebad_s_fixed_target_graph_p0.evaluation import validate_aupro
    for branch in NEW:
        selected=[r for r in maps if r['branch']==branch]
        if len(selected)!=n or any(r['domain']!=domain for r in selected):raise ValueError('wrong domain map population')
        v=st.read_json(path/'aupro'/(branch+'.json'))
        validate_aupro(v,dict(identity=inputs['identity'],domain=domain,branch=branch,map_identity=st.digest(selected),nstrips=200,FPR_max=.3))
        if v['verified_images']!=n:raise ValueError('incomplete domain AUPRO')
        for row in rows:
            if row['branch']==branch:row['AUPRO']=v['AUPRO']
    derived=fixed_combinations(rows,domain,development=domain=='background')
    root=old.read_csv(path/'metrics.csv');index={(r['branch'],r['scoring']):r for r in derived}
    for row in root:
        if row.get('domain')!=domain or (row['branch'],row['scoring']) not in index:raise ValueError('unexpected root metric')
        ref=index[row['branch'],row['scoring']]
        if any(float(row[k])!=float(ref[k]) for k in METRICS):raise ValueError('root/completed CPU metric differs')
    if len(root)!=(10 if domain=='background' else 14):raise ValueError('incomplete root metric population')
    expected={(r['branch'],r['scoring']) for r in (rows if domain=='background' else derived)}
    if {(r['branch'],r['scoring']) for r in root}!=expected:raise ValueError('missing/duplicate root metric keys')
    from .numerics import moments,weights as fit
    s=[r for r in normal['records'] if r['kind']=='Source'];t=[r for r in normal['records'] if r['kind']=='Target']
    if [r['image'] for r in s]!=[r['relative_path'] for r in inputs['source_calibration']] or [r['image'] for r in t]!=[r['relative_path'] for r in inputs['supports']]:raise ValueError('normal statistics list differs')
    _,ss,tt=moments([r['statistics'] for r in s],[r['statistics'] for r in t])
    for b,gamma in zip(NEW,(0.,.5)):
        if not np.array_equal(np.asarray(weights['weights'][b]),fit(ss,tt,gamma)[0]):raise ValueError('frozen weights differ from fixed normal formula')
    for r in derived:r.update(result_origin='historical_background_read_only' if domain=='background' else 'new_domain',provider_run=str(path))
    return dict(rows=derived,inputs=inputs,weights=weights['weights'],weights_table=old.read_csv(path/'weights.csv'),path=str(path))


def macro(runs,domains,label):
    indexes=[{(r['branch'],r['scoring']):r for r in runs[d]['rows']} for d in domains]
    keys=set(indexes[0])
    if any(set(x)!=keys for x in indexes):raise ValueError('macro metric keys differ')
    return [dict(domain=label,branch=b,scoring=sc,aggregation='equal_domain_mean_of_per_domain_metrics',domains=list(domains),
        **{k:float(np.mean([float(idx[b,sc][k]) for idx in indexes],dtype=np.float64)) for k in METRICS}) for b,sc in sorted(keys)]


def aggregate(a):
    paths=dict(background=a.background_run,illumination=a.illumination_run,view=a.view_run);runs={};pending=[];errors=[]
    for d,path in paths.items():
        if path is None or not (path/'status.json').exists():pending.append(d);continue
        try:runs[d]=completed(path,d)
        except (ValueError,OSError,KeyError) as e:pending.append(d);errors.append(dict(domain=d,error=str(e)))
    if runs:
        first=next(iter(runs.values()))
        for d,r in runs.items():
            if source_contract(r['inputs'])!=source_contract(first['inputs']) or not np.array_equal(r['weights'][SOURCE],first['weights'][SOURCE]):raise ValueError('shared Source conditions/weights do not align; macro refused')
    rows=[r for d in paths if d in runs for r in runs[d]['rows']];two=[];three=[]
    if all(d in runs for d in ('illumination','view')):two=macro(runs,('illumination','view'),'new_domains_equal_macro')
    if len(runs)==3:three=macro(runs,('background','illumination','view'),'three_domains_equal_macro')
    a.run_root.mkdir(parents=True,exist_ok=True)
    st.write_csv(a.run_root/'domains_summary.csv',rows)
    st.write_csv(a.run_root/'new_domains_macro.csv',two);st.write_csv(a.run_root/'three_domains_macro.csv',three)
    diffs=[]
    for group in [*[[r for r in rows if r['domain']==d] for d in paths],two,three]:
        if group:diffs.extend(comparisons(group,group[0]['domain']))
    st.write_csv(a.run_root/'metric_comparisons.csv',diffs)
    st.write_csv(a.run_root/'domain_weights.csv',[dict(r,domain=d) for d,x in runs.items() for r in x['weights_table']])
    st.write_json(a.run_root/'input_providers.json',{d:dict(path=x['path'],identity=x['inputs']['identity']) for d,x in runs.items()})
    result=dict(status='pending' if pending else 'crossdomain_validation_complete',pending=pending,errors=errors,new_evaluation_images=629 if two else None,new_AUPRO_complete=4 if two else 0,
        background_recomputed=False,background_combinations='derived_from_development_results',parameter_search=False)
    st.write_json(a.run_root/'status.json',result)
    lines=['# Target-LOO固定跨域验证汇总','',str(result),'',
        'background只读导入；其两个新组合基于开发结果提出，不冒充先验独立实验。两个新域也都有研究历史，不称严格盲测。',
        '原分支/组合用CH1及CURRENT；独立加权用PEAK_LOCAL_WEIGHT及WEIGHTED_MAP_MAX。组合图像完全继承SEGMENT，定位引用指定加权图；继承收益不归因于层加权。',
        '宏平均先逐域算指标再域等权，未合并图像分数。全部逐域/两域/三域差值见metric_comparisons.csv。','',
        '|域/宏平均|分支|读出|Image AUROC %|Image AP %|Pixel AUROC %|Pixel AP %|AUPRO %|','|---|---|---|---:|---:|---:|---:|---:|']
    for r in rows+two+three:lines.append('|'+ '|'.join([r['domain'],r['branch'],r['scoring']]+[f'{100*float(r[k]):.5f}' for k in METRICS])+'|')
    from .protocol import TARGET,TARGET_COMBO,SOURCE_COMBO,COMBINED
    lines+=['','## 固定问题的直接证据（差值为百分点）']
    for d in list(paths)+['new_domains_equal_macro','three_domains_equal_macro']:
        ds=[r for r in diffs if r['domain']==d]
        if not ds:lines.append(d+': pending，尚不能判断。');continue
        for b,ref,rule in ((TARGET,SOURCE,'PEAK_LOCAL_WEIGHT'),(TARGET_COMBO,COMBINED,'CH1'),(TARGET_COMBO,SOURCE_COMBO,'CH1')):
            selected=[r for r in ds if r['branch']==b and r['reference']==ref and r['scoring']==rule]
            lines.append(d+' '+b+' vs '+ref+': '+'; '.join(r['metric']+f' {r["delta_pp"]:+.5f}' for r in selected))
    lines+=['','逐域结果与宏平均同时解释；有反向就保留，不能用按域择优修补。未验证跨支持集稳定性，不继续搜索参数。']
    (a.run_root/'objective_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result
