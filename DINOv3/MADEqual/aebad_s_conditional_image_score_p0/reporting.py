"""Equal-setting summaries; localization metrics are inherited, never rerun."""
import csv
import json
from pathlib import Path
import numpy as np
from . import inputs as ip
from .inputs import st
from .protocol import *


def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def diagnostics(a,labelled):
    from scipy.stats import spearmanr
    correlations=[];extra=['area_correlations.csv','scatter_status.json'];normal=st.sealed_json(a.run_root/'normal_statistics.json')
    for d in DOMAINS:
        for s in SEEDS:
            group=[r for r in labelled if r['domain']==d and r['support_seed']==s and r['label']==1 and r['GT_area_fraction'] is not None]
            for key in ('P','Source_correction','Target_correction'):
                x=[r['GT_area_fraction'] for r in group];y=[r[key] for r in group]
                valid=len(x)>=3 and np.ptp(x)>0 and np.ptp(y)>0
                correlations.append(dict(domain=d,support_seed=s,variable=key,anomaly_images=len(group),
                    Spearman_rho=float(spearmanr(x,y).statistic) if valid else None,
                    status='valid_descriptive' if valid else 'NA_insufficient_or_constant',used_for_fit=False))
    st.write_csv(a.run_root/'area_correlations.csv',correlations)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as e:
        st.write_json(a.run_root/'scatter_status.json',dict(status='pending_missing_optional_plot_dependency',reason=str(e)))
        return extra
    for d in DOMAINS:
        for s in SEEDS:
            fig,ax=plt.subplots(figsize=(6,4),constrained_layout=True)
            for label,name,color in ((0,'evaluation normal','tab:blue'),(1,'evaluation anomaly','tab:red')):
                g=[r for r in labelled if r['domain']==d and r['support_seed']==s and r['label']==label]
                ax.scatter([r['B'] for r in g],[r['P'] for r in g],s=14,alpha=.6,label=name,c=color)
            for kind,name,marker in (('Source','Source calibration','x'),('Target','Target K7 LOO','*')):
                g=[st.sealed_json(Path(r['path'])) for r in normal['records'] if r['domain']==d and r['support_seed']==s and r['kind']==kind]
                ax.scatter([r['B'] for r in g],[r['P'] for r in g],s=25 if kind=='Source' else 65,marker=marker,label=name)
            ax.set(xlabel='B: native SEGMENT median response',ylabel='P: original SEGMENT CH1',title=f'{d}, support seed {s}')
            ax.legend(fontsize=8);name=f'scatters/{d}_seed{s}.png';(a.run_root/'scatters').mkdir(exist_ok=True)
            fig.savefig(a.run_root/name,dpi=140);plt.close(fig);extra.append(name)
    st.write_json(a.run_root/'scatter_status.json',dict(status='six_descriptive_scatter_plots_complete',normal_upper_bound=False))
    return extra


def aggregate(data):
    result=[dict(r) for r in data];metric_keys=('image_AUROC','image_AP','pixel_AUROC','pixel_AP','AUPRO')
    populations=[('seed_macro',None,s,[r for r in data if int(r['support_seed'])==s]) for s in SEEDS]
    populations += [('domain_two_seeds',d,None,[r for r in data if r['domain']==d]) for d in DOMAINS]
    populations += [('macro_3domains_2seeds',None,None,data)]
    for scope,d,s,g in populations:
        for b in BRANCHES:
            selected=[r for r in g if r['branch']==b]
            expected=3 if scope=='seed_macro' else 2 if scope=='domain_two_seeds' else 6
            if len(selected)!=expected:raise ValueError('incomplete equal-setting macro')
            result.append(dict(domain=d or scope,support_seed=s if s is not None else 'both',branch=b,aggregation=scope,
                **{k:float(np.mean([float(r[k]) for r in selected])) for k in metric_keys},
                localization='inherited_unchanged',independent_repetitions=False))
    return result


def comparisons(data):
    result=[];index={(r['domain'],str(r['support_seed']),r['branch']):r for r in data}
    for (d,s,b),r in index.items():
        for left,right in ((TARGET,BASE),(TARGET,SOURCE),(SOURCE,BASE)):
            if b!=left:continue
            ref=index[d,s,right]
            for k in ('image_AUROC','image_AP'):
                result.append(dict(domain=d,support_seed=s,branch=left,reference=right,metric=k,
                    delta_pp=100*(float(r[k])-float(ref[k]))))
    return result


def summarize(a):
    path=a.run_root/'evaluation_complete.json';pending=[]
    if not path.exists():pending.append('six_setting_image_evaluation')
    else:
        from .pipeline import evaluate
        evaluate(a)  # Completed replay checks small commits only, no GT/pixel work.
    status=dict(status='pending' if pending else 'six_settings_image_evaluation_complete',pending=pending,primary=TARGET,
        plan=plan(),new_pixel_jobs=0,new_AUPRO_jobs=0,localization='inherited_unchanged',parameter_search=False,
        K7_K8_proxy_distribution_equivalence_claimed=False)
    lines=['# 正常统计驱动的SEGMENT整体响应条件校正','',
        '固定三域、support_seed42/43、Source_seed42；902张唯一图像，1804个图像×支持组实例，不是1804张独立图。',
        '三个域有开发历史；K7正常关系作为K8代理，未假设同分布、正常上界或未知缺陷安全。',
        '只改变图像分数；定位引用同组父UNION_SOURCE_WEIGHT地图，三项定位指标继承、未改变。','']
    if not pending:
        data=aggregate(rows(a.run_root/'setting_metrics.csv'));diff=comparisons(data)
        st.write_csv(a.run_root/'metrics.csv',data);st.write_csv(a.run_root/'metric_comparisons.csv',diff)
        lines+=['|域/汇总|支持seed|分支|Image AUROC %|Image AP %|Pixel AUROC %（继承）|Pixel AP %（继承）|AUPRO %（继承）|',
                '|---|---|---|---:|---:|---:|---:|---:|']
        for r in data:lines.append('|'+ '|'.join([r['domain'],str(r['support_seed']),r['branch']]+[f'{100*float(r[k]):.6f}' for k in ('image_AUROC','image_AP','pixel_AUROC','pixel_AP','AUPRO')])+'|')
        lines+=['','## 固定比较与判读','']
        for left,right in ((TARGET,BASE),(TARGET,SOURCE),(SOURCE,BASE)):
            g=[r for r in diff if r['branch']==left and r['reference']==right]
            grand=[r for r in g if r['domain']=='macro_3domains_2seeds']
            lines.append(f'{left} − {right}：等权宏平均 '+ '；'.join(f'{r["metric"]} {r["delta_pp"]:+.6f} pp' for r in grand)+'。')
            for k in ('image_AUROC','image_AP'):
                v=[r['delta_pp'] for r in g if r['domain'] in DOMAINS and r['support_seed'] in ('42','43') and r['metric']==k]
                lines.append(f'{k}六设置：改善{sum(x>0 for x in v)}，下降{sum(x<0 for x in v)}，不变{sum(x==0 for x in v)}。')
        model=st.sealed_json(a.run_root/'coefficients.json')['models']
        lines.append('退化设置：'+str([k for k,v in model.items() if v['degenerate']])+'；零Target斜率：'+str([k for k,v in model.items() if v['beta_ST']==0])+'。退化/零斜率不记为机制成功。')
        reversals=[]
        for d in DOMAINS:
            for k in ('image_AUROC','image_AP'):
                changes=[r['delta_pp'] for r in diff if r['domain']==d and r['support_seed'] in ('42','43') and r['branch']==TARGET and r['reference']==BASE and r['metric']==k]
                if len(changes)==2 and min(changes)<0<max(changes):reversals.append(f'{d}/{k}')
        lines.append('主候选相对SEGMENT出现两支持组正负反转的域/指标：'+str(reversals)+'。未反转不等于独立重复稳定性。')
        changes=rows(a.run_root/'score_changes.csv');ranks=rows(a.run_root/'ranking_summary.csv');possible=[]
        for d in DOMAINS:
            for s in SEEDS:
                g={r['group']:float(r['mean_change']) for r in changes if r['domain']==d and int(r['support_seed'])==s and r['branch']==TARGET}
                rank=next(r for r in ranks if r['domain']==d and int(r['support_seed'])==s and r['branch']==TARGET)
                if g['anomaly']<g['normal'] and float(rank['net_error_change'])>0:possible.append(f'{d}/seed{s}')
        lines.append('异常均值降分多于正常、同时排序错误净增加的设置：'+str(possible)+'。这是可能过补偿的描述性信号，不是物理因果证明。')
        lines+=['',
            '正常关系拟合、检测收益、物理解释必须分开：系数和散点只能描述拟合；检测收益由固定AUROC/AP差值判断；本轮不能证明整体响应代表真实扰动强度。',
            'score_changes及ranking_summary分别报告正常/异常分数变化和排序改善/恶化。排序对不是独立样本，净错误不是固定阈值误报/漏检。',
            '结合六设置差值检查支持组/域反转；异常分数大幅降低但指标下降只能提示过补偿，不能仅凭降分证明因果机制。',
            'GT面积相关仅评价后描述；缺失见missing_area，不参与拟合、系数选择或预测。',
            'coefficient_influence是删除一条既定LOO统计后的系数影响分析，保持Source标准化/beta_S不变，不是重新排除支持、支持集复验或选系数。',
            '若只改善正常拟合或散点而未改善检测指标，结论是无检测收益。K7/K8差异既不能被称为已解决，也不能成为失败的唯一归因。',
            '不按域选择分支，不调整B/正则/支持，不扩展已搁置的共享残差A/B/C。']
    else:lines.append('核心图像评价尚未完成；不填0、不声称有效，也没有新增定位任务。')
    costs=[st.read_json(p) for p in (a.run_root/'costs').glob('*.json')]
    st.write_json(a.run_root/'resource_summary.json',dict(stage_seconds=sum(r['seconds'] for r in costs),
        peak_process_RAM_bytes=max((r['process_peak_rss_bytes'] for r in costs),default=None),
        peak_cuda_allocated_bytes=max((r['peak_cuda_allocated_bytes'] for r in costs if 'peak_cuda_allocated_bytes' in r),default=None),
        feature_forwards=sum(r.get('feature_forwards',0) for r in costs),
        note='committed subprocess records; excludes current summarize/pack tail; not total machine RAM or exact wall time'))
    st.write_json(a.run_root/'status.json',status)
    (a.run_root/'objective_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return status
