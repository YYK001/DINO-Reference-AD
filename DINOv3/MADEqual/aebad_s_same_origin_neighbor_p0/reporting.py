"""Honest pending, fixed contrasts, and separate measured/unmeasured costs."""
from pathlib import Path
import numpy as np
from . import inputs as ip,pipeline as pp
from .inputs import st,read_csv
from .protocol import CONFIG,BRANCHES,MAPS,NEW,U,S,A,B,C,COMBINED,job_plan
METRICS=('image_AUROC','image_AP','pixel_AUROC','pixel_AP','AUPRO')


def estimate(inv):
    """Ranges are prior planning estimates, never pretend to be a new benchmark."""
    rebuild=inv.get('source_case') in ('bank_unavailable','origin_unresolved_explicit_rebuild_needed','explicit_independent_rebuild')
    ready=inv.get('source_case')=='exact_recorded_neighborhood'
    stage=[dict(stage='Source provenance or explicit reconstruction',hours_range=[0,0] if ready else ([.75,3.] if rebuild else [.4,1.3]),
                evidence='same workstation earlier 455 forwards about 24min; disk/sampler variable; replay/new Top16 NOT benchmarked'),
        dict(stage='Target neighbors and complete feature preparation',hours_range=[.05,.3],evidence='8 supports; original complete grids reused when present'),
        dict(stage='66 Source calibration images, two passes',hours_range=[.25,1.5],evidence='Top16/neighbor kernel unmeasured; second pass replays small distances, no forward/NN'),
        dict(stage='297 full evaluation queries and shared searches',hours_range=[.6,3.],evidence='previous query full forward about 3s; SACP926 images in4506.65s (~4.87s/image); new Top16 overhead unknown'),
        dict(stage='exact CPU pixels, original AUPRO and summary',hours_range=[.1,.5],evidence='previous background one-map CPU36.7s and AUPRO15.8s; 3 or 5 maps plus IO')]
    lo=sum(x['hours_range'][0] for x in stage);hi=sum(x['hours_range'][1] for x in stage)
    # 4 directions, 4 layers, retained bank rows only. No fivefold all-candidate bank.
    source_delta=100000*4*1280*4*4;target_delta=8*13924*4*1280*4*4
    return dict(kind='provisional_historical_range_not_new_GPU_measurement',stages=stage,total_hours_range=[round(lo,2),round(hi,2)],
        excludes_nothing_by_design=True,unknown=['actual Top16 throughput','actual source replay compatibility','new-device CPU/scratch contention'],
        formal_budget_hours=12,policy='pause on completed image/job if measured extrapolation clearly exceeds 12h; require explicit approval before override',
        disk=dict(Source_delta_GiB=round(source_delta/2**30,2),Target_delta_GiB=round(target_delta/2**30,2),
            Source_centers_GiB=round(100000*1280*4*4/2**30,2),recovery_checkpoint_GiB=round(source_delta/2**30,2),
            official_rebuild_candidate_temporary_GiB=120.839 if rebuild else 0,
            note='additional calibration caches, output, labels/pixel sort scratch about 5-15GiB; existing assets excluded; no default full-query retention; candidate spool only in explicit rebuild; SSD scratch and HDD persistent paths configurable'),
        measured_current_run=False)


def summarize(a):
    pending=[];rows=[];cal=pp.states(a) if (a.run_root/'calibration.json').exists() else None
    from .evaluation import restored
    cpu=restored(a) if (a.run_root/'evaluation/status.json').exists() else None
    if cpu:
        rows=read_csv(a.run_root/'evaluation/metrics.csv')
        for r in rows:
            path=a.run_root/'aupro'/(r['branch']+'.json')
            if path.exists():
                from DINOv3.MADEqual.aebad_s_fixed_target_graph_p0.evaluation import validate_aupro
                maps=[x for x in read_csv(a.run_root/'evaluation/map_manifest.csv') if x['branch']==r['branch']]
                v=st.read_json(path);validate_aupro(v,dict(identity=ip.identity(a),domain='background',branch=r['branch'],map_identity=st.digest(maps),nstrips=200,FPR_max=.3))
                if v['verified_images']!=297:raise ValueError('partial AUPRO')
                r['AUPRO']=v['AUPRO']
            if r.get('AUPRO') in (None,''):pending.append('AUPRO:'+r['branch'])
        lookup={(r['branch'],r['scoring']):r for r in rows}
        for rule in ('CH1','CURRENT'):
            if (U,rule) in lookup and (S,rule) in lookup:
                combo=dict(lookup[U,rule],branch=COMBINED,image_AUROC=lookup[S,rule]['image_AUROC'],image_AP=lookup[S,rule]['image_AP'],
                    metric_origin='SEGMENT image scores + UNION localization; no new map or AUPRO');rows.append(combo)
    else:pending.append('exact CPU evaluation')
    for name in ('target_ready.json','calibration.json','predictions_frozen.json'):
        if not (a.run_root/name).exists():pending.append(name)
    if not rows or {(r['branch'],r['scoring']) for r in rows}!={(b,s) for b in BRANCHES for s in ('CH1','CURRENT')}:pending.append('complete six-branch/two-score metric table')
    st.write_csv(a.run_root/'metrics.csv',rows);lookup={(r['branch'],r['scoring']):r for r in rows};comparisons=[]
    for new,ref in ((B,A),(C,B),(C,U),(C,COMBINED)):
        for rule in ('CH1','CURRENT'):
            if (new,rule) not in lookup or (ref,rule) not in lookup:continue
            for k in METRICS:
                v=lookup[new,rule].get(k);r=lookup[ref,rule].get(k)
                comparisons.append(dict(domain='background',branch=new,reference=ref,scoring=rule,metric=k,
                    delta_pp=100*(float(v)-float(r)) if v not in (None,'') and r not in (None,'') else None))
    st.write_csv(a.run_root/'metric_comparisons.csv',comparisons)
    margins=[]
    if cpu:
        inst=read_csv(a.run_root/'evaluation/defect_instances.csv')
        for b in (S,*NEW):
            for definition in ('margin','fixed_union_margin'):
                selected=[r for r in inst if r['branch']==b]
                v=np.asarray([float(r[definition+'_change_vs_'+U]) for r in selected if r[definition+'_change_vs_'+U] not in ('',None)])
                margins.append(dict(branch=b,definition=definition,instances=len(selected),computable=len(v),NA=len(selected)-len(v),
                    damaged=int((v<-1e-6).sum()),improved=int((v>1e-6).sum()),unchanged=int((abs(v)<=1e-6).sum()),
                    mean=float(v.mean()) if len(v) else None,median=float(np.median(v)) if len(v) else None,
                    p05=float(np.quantile(v,.05)) if len(v) else None,p95=float(np.quantile(v,.95)) if len(v) else None,
                    interpretation='supplemental; response calibration/scale differs; not native distance shrinkage or missed-detection counts'))
        st.write_csv(a.run_root/'margin_summary.csv',margins)
        stats=[]
        for s in ip.catalog(a)['samples']:
            _,r=pp.load(a,s);stats.extend(dict(domain='background',sample_id=s['sample_id'],**v) for v in r['metadata']['boundaries'])
        st.write_csv(a.run_root/'boundary_statistics.csv',stats)
    costs=[st.read_json(p) for p in sorted((a.run_root/'costs').glob('*.json'))]
    st.write_csv(a.run_root/'stage_costs.csv',costs)
    result=dict(status='pending' if pending else 'all_five_metrics_complete',identity=ip.identity(a),pending=sorted(set(pending)),
        primary=C,metric_rows=len(rows),evaluation_images=297,training_jobs=0,seed=42,cross_seed_stability='not evaluated',
        numerical_failed_checks=cpu['numerical_failed_checks'] if cpu else None,
        job_plan=job_plan(cal['reuse_baselines'] if cal else True),parameter_search=False)
    st.write_json(a.run_root/'status.json',result)
    lines=['# 同源邻域差分检索：background固定验证','',f"状态：{result['status']}；pending：{result['pending']}",
        '单域、单seed42固定K8，297张评价；无训练，无参数搜索。CH1主评分，CURRENT补充。',
        '','|评分|分支|Image AUROC|Image AP|Pixel AUROC|Pixel AP|AUPRO|','|---|---|---:|---:|---:|---:|---:|']
    for rule in ('CH1','CURRENT'):
        for b in BRANCHES:
            r=lookup.get((b,rule))
            if r:lines.append('|'+ '|'.join([rule,b]+['pending' if r.get(k) in ('',None) else f'{100*float(r[k]):.5f}' for k in METRICS])+'|')
    lines+=['','A→B检验绝对邻居残差与相对差分；B→C检验绝对Top16内的联合选择。不是完整两因素交互实验。',
        'Top16是候选召回约束，只有候选内联合最优，不是全库联合最优或防错匹配保证。',
        '边界：query最外圈或绝对Top1缺完整四邻域时，三个新分支均返回UNION标准化响应；C16不从第17名以后补位。',
        'Source增强来源恢复/独立重建及原库是否共用见source_neighbors manifest和calibration.json；历史复用必须通过本轮地图/评分复核。',
        'margin仅作补充：标度已改变，受损实例数不是固定阈值漏检数。NA保留。',
        '','## 固定结论问题']
    for new,ref,label in ((B,A,'相对差分表达'),(C,B,'候选内联合选择'),(C,U,'相对UNION'),(C,COMBINED,'相对固定评分/定位组合')):
        ds=[r for r in comparisons if r['branch']==new and r['reference']==ref and r['scoring']=='CH1']
        lines.append(label+'：'+('；'.join(f"{r['metric']} {r['delta_pp']:+.5f} pp" if r['delta_pp'] is not None else r['metric']+' pending' for r in ds) if ds else 'pending，尚不能判断。'))
    lines+=['','按完整五项指标和实例代价判断，禁止只挑正向结果；单开发域结果不代表其他域或支持集稳定性。',
        '不因指标不好自动改C、权重、邻域、层或增加训练。']
    (a.run_root/'objective_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result
