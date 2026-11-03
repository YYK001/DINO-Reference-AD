"""Equal-domain/equal-support summaries, all losses retained; no new localization."""
import os
import tarfile
from pathlib import Path
import numpy as np
from . import inputs as ip
from .protocol import *
st=ip.st


def aggregate(data):
    result=[dict(r,aggregation='setting') for r in data]
    groups=[('domain_two_seeds',d,'both',[r for r in data if r['domain']==d],2) for d in DOMAINS]
    groups += [('seed_macro','seed_macro',s,[r for r in data if int(r['support_seed'])==s],3) for s in SEEDS]
    groups += [('macro_3domains_2seeds','macro_3domains_2seeds','both',data,6)]
    for scope,d,s,pop,n in groups:
        for b in BRANCHES:
            group=[r for r in pop if r['branch']==b]
            if len(group)!=n:continue  # Blocked cells stay absent, not a partial macro.
            result.append(dict(domain=d,support_seed=s,branch=b,aggregation=scope,
                **{k:float(np.mean([float(r[k]) for r in group])) for k in ('image_AUROC','image_AP')},
                localization='inherited_UNION_SOURCE_WEIGHT_unchanged'))
    return result


def differences(data):
    idx={(r['domain'],str(r['support_seed']),r['branch']):r for r in data};out=[]
    for (d,s,b),row in idx.items():
        if b!=PRIMARY:continue
        for baseline in (BASE,SHARED,OWN):
            if (d,s,baseline) not in idx:continue
            for metric in ('image_AUROC','image_AP'):
                out.append(dict(domain=d,support_seed=s,branch=b,reference=baseline,metric=metric,
                    delta_pp=100*(float(row[metric])-float(idx[d,s,baseline][metric]))))
    return out


def summarize(a):
    pending=[];data=[]
    path=a.run_root/'evaluation_complete.json'
    if not path.is_file():pending=['frozen_predictions_and_image_evaluation']
    else:
        r=st.sealed_json(path)
        if r['identity']!=ip.signature(a):raise ValueError('evaluation uses other inputs')
        for name,expected in r['files'].items():
            if st.sha(a.run_root/name)!=expected:raise ValueError('completed compact evaluation file changed: '+name)
        frozen=st.sealed_json(a.run_root/'predictions_frozen.json')
        if frozen['seal']!=r['prediction_seal']:raise ValueError('evaluation prediction freeze changed')
        pending=list(frozen['pending']);data=ip.csv_rows(a.run_root/'metrics.csv')
        if len(ip.csv_rows(a.run_root/'setting_metrics.csv'))!=36 and not pending:raise ValueError('six settings x six branches incomplete')
    diff=differences(data);st.write_csv(a.run_root/'metric_comparisons.csv',diff)
    status=dict(status='pending' if pending else 'six_settings_image_evaluation_complete',pending=pending,primary=PRIMARY,
        plan=plan(),new_pixel_jobs=0,new_AUPRO_jobs=0,training=False,Source_rebuild=False,
        parameter_search=False,shared_evaluation_images_not_independent_repetitions=True)
    st.write_json(a.run_root/'status.json',status)
    lines=['# 面积容量约束的反向覆盖：固定实验','',
        '三域均有开发历史；902张唯一评价图、1804个图像×支持组实例，两组支持共享评价图。',
        '原Source455、校准66、固定两组K8保持。只改变图像读出；定位引用UNION_SOURCE_WEIGHT，新增像素评价/AUPRO为0。',
        '256个空间连通区域按patch占比计质量；贪心合并和匹配同一距离；八张模板整图选优，不逐区域拼接模板。',
        '主覆盖忠实求解最差模板区域平均代价的稀疏LP；反向最近邻为取消列容量的对照。',
        '主候选为CAPACITY_FUSION，两个FREE融合是固定对照，CAPACITY_ONLY/FREE_ONLY为辅助，不据结果改主候选。','']
    if pending:lines.append('尚未完成：'+', '.join(pending)+'。不填0，不宣称有效。')
    else:
        lines += ['| 域/汇总 | 支持seed | 分支 | Image AUROC % | Image AP % |','|---|---|---|---:|---:|']
        for r in data:lines.append(f"|{r['domain']}|{r['support_seed']}|{r['branch']}|{100*float(r['image_AUROC']):.6f}|{100*float(r['image_AP']):.6f}|")
        lines+=['','固定比较（百分点）：']
        for baseline in (BASE,SHARED,OWN):
            grand=[r for r in diff if r['domain']=='macro_3domains_2seeds' and r['reference']==baseline]
            lines.append(PRIMARY+' − '+baseline+'：'+'；'.join(f"{r['metric']} {r['delta_pp']:+.6f}" for r in grand)+'。')
            for metric in ('image_AUROC','image_AP'):
                values=[r['delta_pp'] for r in diff if r['domain'] in DOMAINS and str(r['support_seed']) in ('42','43')
                    and r['reference']==baseline and r['metric']==metric]
                lines.append(f'{metric} 六设置：改善{sum(v>0 for v in values)}，下降{sum(v<0 for v in values)}，不变{sum(v==0 for v in values)}。')
        lines += ['',
            '只有主候选超过SEGMENT，才有当前新读出的检测收益；同时超过两种FREE融合，才支持容量约束的额外作用。',
            '若FREE有效而容量未额外改善，应归因于反向信息；若没有收益，保留固定负向结果，不继续调参或按域选分支。']
    lines += ['',
        '区域均值可能稀释小缺陷。面积占比不是部件数量；容量约束不构成拓扑、结构完整性或真实缺陷保持保证。',
        '正常校准只使用各设置66张Source图，无Target自检索校准；测试标签仅在冻结后评价。',
        '运输匹配、面积约束与标准化等权相加不分别宣称首创。继承定位指标不是新增定位收益。',
        '成本见resource_summary与各pair求解记录；未知项不作零成本，CPU/单GPU和真实完整资产吞吐仍须区分。']
    (a.run_root/'objective_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    costs=[st.read_json(p) for p in (a.run_root/'costs').glob('*.json')]
    st.write_json(a.run_root/'resource_summary.json',dict(recorded_stage_seconds=sum(r['seconds'] for r in costs),
        peak_process_tree_RSS_bytes=max((r.get('peak_process_tree_RSS_bytes',0) for r in costs),default=0),
        peak_cuda_allocated_bytes=max((r.get('peak_cuda_allocated_bytes',0) for r in costs),default=0),
        peak_cuda_reserved_bytes=max((r.get('peak_cuda_reserved_bytes',0) for r in costs),default=0),
        feature_forwards=sum(r.get('feature_forwards',0) for r in costs),
        new_array_payload_bytes=sum(r.get('new_array_payload_bytes',0) for r in costs),
        note='stage records exclude current summarize/pack tail; RSS sum may double-count shared pages; not machine RAM'))
    return status


def pack(a):
    dest=a.run_root.with_name(a.run_root.name+'_review.tar.gz')
    with tarfile.open(dest,'w:gz',compresslevel=1) as archive:
        for root,dirs,files in os.walk(a.run_root,followlinks=False):
            dirs[:]=[d for d in dirs if d not in ('scratch','pixel_scratch','__pycache__') and not (Path(root)/d).is_symlink()]
            for name in files:
                p=Path(root)/name
                if p.is_symlink() or p.suffix not in ('.json','.csv','.md','.txt','.log','.png'):continue
                archive.add(p,arcname=a.run_root.name+'/'+p.relative_to(a.run_root).as_posix(),recursive=False)
    return dict(archive=str(dest),bytes=dest.stat().st_size,raw_features_regions_transport_matrices_and_scratch_excluded=True)
