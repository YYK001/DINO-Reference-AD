"""All six responses and fixed task division, no outcome-driven model choice."""
from pathlib import Path
from types import FunctionType
import numpy as np
from . import inputs as ip,prediction as pred,evaluation as ev
from .inputs import st
from .protocol import DOMAINS,LAYERS,BRANCHES,NEW,U,S,MAIN,TARGET,SOURCE,UNIFORM,COMBINED
from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation import reporting as previous
METRICS=previous.METRICS


def summarize(a):
    pred.verify(a);rows=[];instances=[];changes=[];etas=[];checks=[];pending=[];aupro_new=0;aupro_reused=0
    from DINOv3.MADEqual.aebad_s_fixed_target_graph_p0.evaluation import validate_aupro
    for d in DOMAINS:
        if not ev.restored(a,d):pending.append(d+'/CPU');continue
        path=a.run_root/'evaluation'/d;local=ev.read_csv(path/'metrics.csv')
        if {(r['branch'],r['scoring']) for r in local}!={(b,s) for b in BRANCHES for s in ('CH1','CURRENT')}:raise ValueError('incomplete metric rows')
        maps=ev.read_csv(path/'map_manifest.csv')
        for r in local:
            for key in METRICS:r[key]=float(r[key]) if r.get(key) not in ('',None) else None
            p=a.run_root/'aupro'/d/(r['branch']+'.json')
            if p.exists():
                value=st.read_json(p);validate_aupro(value,dict(identity=ip.identity(a),domain=d,branch=r['branch'],map_identity=st.digest([m for m in maps if m['branch']==r['branch']]),nstrips=200,FPR_max=.3))
                r.update(AUPRO=value['AUPRO'],AUPRO_origin=value.get('origin','this_run_original_ADEval'))
                if r['scoring']=='CH1':
                    if value.get('origin')=='parent_exact_map_GT_population':aupro_reused+=1
                    else:aupro_new+=1
            else:pending.append(d+'/'+r['branch']+'/AUPRO')
        rows.extend(local);instances.extend(ev.read_csv(path/'defect_instances.csv'));changes.extend(ev.read_csv(path/'response_changes.csv'));etas.extend(ev.read_csv(path/'eta_regions.csv'));checks.extend(ev.read_csv(path/'numerical_verification.csv'))
    if len(rows)==3*6*2:
        for b in BRANCHES:
            for rule in ('CH1','CURRENT'):
                group=[r for r in rows if (r['branch'],r['scoring'])==(b,rule)]
                rows.append(dict(domain='macro_3domains',branch=b,scoring=rule,aggregation='three_domains_equal',
                    **{k:float(np.mean([r[k] for r in group])) if all(r[k] is not None for r in group) else None for k in METRICS}))
    rows=previous.combine(rows);comparison=[];index={(r['domain'],r['branch'],r['scoring']):r for r in rows}
    for d in (*DOMAINS,'macro_3domains'):
        for rule in ('CH1','CURRENT'):
            candidate=index.get((d,MAIN,rule))
            if candidate:
                for reference in (U,S,COMBINED,TARGET,SOURCE,UNIFORM):
                    ref=index[d,reference,rule]
                    for k in METRICS:comparison.append(dict(domain=d,branch=MAIN,reference=reference,scoring=rule,metric=k,
                        delta=100*(candidate[k]-ref[k]) if candidate[k] is not None and ref[k] is not None else None,units='percentage_points',direction='MAIN_minus_reference'))
    mechanism=[]
    for d in DOMAINS:
        for b in BRANCHES:
            means={}
            for region,key in (('normal_all_pixels','R_N'),('anomaly_GT_pixels','R_A')):
                values=[float(r['mean_signed_response_reduction']) for r in changes if (r['domain'],r['branch'],r['group'])==(d,b,region) and r['mean_signed_response_reduction']!='']
                means[key]=float(np.mean(values)) if values else None
            mechanism.append(dict(domain=d,branch=b,reference=U,**means,G=means['R_N']-means['R_A'] if all(v is not None for v in means.values()) else None,
                units='final_fused_original_Source_patch_MAD_response',aggregation='image_equal_within_group',normal_group='all pixels of normal images',anomaly_group='resized GT pixels of abnormal images'))
    fn=previous.margins;detail,margin=FunctionType(fn.__code__,dict(fn.__globals__,BRANCHES=BRANCHES,DOMAINS=DOMAINS,U=U),fn.__name__)(instances) if instances else ([],[])
    for row in detail:row['units']='shared_frozen_Source_patch_MAD_final_fused_response_margin_change'
    subspaces=[]
    for d in DOMAINS:
        for l in LAYERS:
            p=a.run_root/'fit'/d/f'L{l}.json'
            if not p.exists():pending.append(f'{d}/L{l}/fit')
            else:subspaces.extend(st.sealed_json(p)['diagnostics'])
    numeric_failed=sum(str(r['verified']).lower()!='true' for r in checks)
    if numeric_failed:pending.append('numerical_failed_checks')
    contributions=[dict(domain=r['domain'],layer=r['layer'],group=r['group'],**v) for r in subspaces for v in r.get('image_contributions',[])]
    for name,data in {'metrics.csv':rows,'metric_comparisons.csv':comparison,'defect_instances.csv':instances,'margin_changes.csv':detail,'margin_summary.csv':margin,'fit_image_contributions.csv':contributions,
        'response_changes.csv':changes,'mechanism_summary.csv':mechanism,'eta_regions.csv':etas,'subspace_statistics.csv':subspaces,'numerical_verification.csv':checks}.items():st.write_csv(a.run_root/name,data)
    costs=[]
    for path in sorted((a.run_root/'costs').glob('*.json')):
        v=st.read_json(path);result=v.get('result',{});result=result if isinstance(result,dict) else {}
        costs.append(dict(record=path.name,stage=v['stage'],seconds=v['seconds'],process_peak_rss_bytes=v.get('process_peak_rss_bytes'),
            peak_cuda_allocated_bytes=v.get('peak_cuda_allocated_bytes'),peak_cuda_reserved_bytes=v.get('peak_cuda_reserved_bytes'),
            query_forward_calls=result.get('query_forward_calls',result.get('query_image_forward_calls')),
            Exact_searches=result.get('searches',result.get('Exact_searches')),reference_execution=result.get('execution')))
    st.write_csv(a.run_root/'stage_costs.csv',costs)
    status=dict(status='all_five_metrics_complete' if not pending and len(rows)==56 else 'pending',identity=ip.identity(a),primary=MAIN,pending=sorted(set(pending)),
        evaluation_images=sum(1 for s in ip.catalog(a)['samples']),metric_rows=len(rows),numerical_failed_checks=numeric_failed,
        logical_AUPRO_jobs=18,new_AUPRO_jobs=aupro_new,reused_AUPRO_jobs=aupro_reused,training_jobs=0,parameter_search=False)
    st.write_json(a.run_root/'status.json',status)
    report=['# SACP 固定配置实验','',str(status),'','三域为开发域；单组固定K8，未训练适配器或分割头。CH1主表，CURRENT补充。全部分支共享原距离MAD和CH1参数。',
        '','|域|分支|Image AUROC|Image AP|Pixel AUROC|Pixel AP|AUPRO|','|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['scoring']=='CH1':report.append('|'+r['domain']+'|'+r['branch']+'|'+'|'.join('pending' if r[k] is None else f'{100*r[k]:.5f}' for k in METRICS)+'|')
    report+=['','差值见metric_comparisons.csv：主方法减参照，单位百分点。全部域及方向/均匀对照一并报告，不按域挑最佳分支。',
        '固定SEGMENT_SCORE_UNION_MAP仅为既有任务分工，不生成定位图或额外AUPRO。',
        '实际最终地图计算自身及固定UNION峰值margin；原NA保留，小实例口径沿用原工具。eta差异不代替margin，也不等于漏检。',
        'R_N/R_A是正常图全部像素/异常图GT像素上的UNION减候选响应，先每图平均再图像等权，G=R_N-R_A；尺度为共享Source patch MAD的最终融合响应。',
        'gamma>1只表示当前收缩二阶矩下的相对能量；不证明物理同源、扰动纯化或缺陷保持。共同基保留均值且只表示统计，查询不做子空间重构。',
        '独立Source/Target方向使用与主方法完全相同的有效组和秩；均匀对照匹配原生距离总下降，幅度/位置/实例响应不因此相同。',
        'eta_regions.csv只是附表，不能用诊断AUC或阈值代替实际检测效果；不据结果改lambda、rank、分组或层。',
        '','## 固定问题的结果','']
    for reference in (U,S,COMBINED,TARGET,SOURCE,UNIFORM):
        v=[r for r in comparison if (r['domain'],r['scoring'],r['reference'])==('macro_3domains','CH1',reference)]
        report.append(reference+'：'+(', '.join(f"{r['metric']} {r['delta']:+.5f} pp" for r in v) if len(v)==5 and all(r['delta'] is not None for r in v) else 'pending，不作完整结论'))
    report+=['','若无优势，当前固定配置收束，不自动扩大规模、挑参数或将其解释为整个任务不可解决。统计自洽不等于方法有效。','阶段成本见costs；原始特征/参照上传按实际计数，不把免训练等同于零成本。']
    (a.run_root/'objective_report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    return status
