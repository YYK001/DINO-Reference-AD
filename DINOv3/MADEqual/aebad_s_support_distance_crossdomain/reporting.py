"""Separate domain conclusions and pending-aware cross-domain comparison."""
from __future__ import annotations
import csv
import numpy as np
from . import inputs as ip

CATALOG_SHA='9b6567d5de1c107a921f2d9a6a44e136a0e8bb79c8a4feef445f48cbf4f1cf3c'


def csv_file(path):return ip.base.read_csv(path.read_bytes())


def plots(values,directory,domain):
    from matplotlib import pyplot as plt
    for feature in ip.MAIN_FEATURES:
        fig,axes=plt.subplots(2,4,figsize=(15,6),layout='constrained')
        for j,l in enumerate(ip.LAYERS):
            p=values.get(l)
            for k,wkey in enumerate(('base_weights','matched_weights')):
                ax=axes[k,j];ax.set_title(f'{domain} L{l} '+('uncontrolled' if k==0 else 'matched b/delta'))
                if p:
                    a=p['data'][feature];y=p['data']['group'];w=p[wkey]
                    for g,label in ((0,'normal high response'),(1,'GT covered')):
                        keep=(y==g)&np.isfinite(a)&(w>0)
                        if keep.any():
                            order=np.argsort(a[keep],kind='stable');v=a[keep][order];ww=w[keep][order]
                            ax.step(v,np.cumsum(ww)/ww.sum(),where='post',label=label)
                ax.set_xlabel(feature);ax.set_ylim(0,1);ax.grid(alpha=.2)
                if ax.lines:ax.legend(fontsize=7)
        fig.savefig(directory/(feature+'_weighted_CDF.png'),dpi=120);plt.close(fig)


def domain_report(out,domain,comparisons,coverage,instances,audit):
    def f(v):return 'NA' if v in (None,'') else f'{float(v):.5g}'
    text=[f'# {domain}：逐支持距离条件诊断','',
        f'原seed42 K8、{sum(ip.COUNTS[domain])}张图；正常{ip.COUNTS[domain][0]}，异常{ip.COUNTS[domain][1]}。',
        'gap2/gap4/gap8为预设主描述量，四层全部报告；其他原内核描述量只是辅助。',
        '每域独立生成5×5分位边界，沿用background的图像均衡、共同质量和200次图像bootstrap。没有重跑检测指标。','',
        '| 层 | 主描述量 | 未控制 AUC | 匹配 AUC [95% CI] | b SMD | delta SMD | 残余混杂标记 |',
        '|---|---|---|---|---|---|---|']
    lookup={(r['layer'],r.get('mode'),r['feature']):r for r in comparisons}
    for l in ip.LAYERS:
        for key in ip.MAIN_FEATURES:
            a=lookup.get((l,'uncontrolled_effective',key),{});b=lookup.get((l,'matched_b_delta',key),{})
            text.append(f"| {l} | {key} | {f(a.get('diagnostic_AUROC'))} | {f(b.get('diagnostic_AUROC'))} [{f(b.get('diagnostic_AUROC_CI_low'))}, {f(b.get('diagnostic_AUROC_CI_high'))}] | {f(b.get('feature_valid_b_standardized_difference'))} | {f(b.get('feature_valid_delta_standardized_difference'))} | {b.get('residual_confounding_flag','NA')} |")
    text+=['','## 覆盖及实例','']
    for r in coverage:
        if r['row_kind']=='matched_group':text.append(f"- L{r['layer']} group{r['group']}：覆盖{r['images_covered']}/{r['images_effective']}张有效图，共同质量{r['effective_overlap_coefficient']:.5g}，未覆盖质量{r['uncovered_probability_mass']:.5g}，最大单图份额{r['max_matched_image_share']:.5g}。")
    for field in ('own_margin_change','fixed_margin_change'):
        available=[r for r in instances if r.get(field) is not None]
        text.append(f"- {field}：可计算{len(available)}，下降{sum(r[field]<-1e-6 for r in available)}，未下降{sum(r[field]>=-1e-6 for r in available)}，NA{len(instances)-len(available)}。")
    text+=['','## 数值复核与结论边界','',
        '同批缓存地图/分数/实例使用1e-5严格复核；更早正式分数另外使用原审计的1e-4兼容性口径，同时保留1e-5结果。',
        'parent_historical_audit.json保留原审计次数，不能声称所有历史比较都通过1e-5。',
        '峰值列标记父CPU保存坐标或本轮派生坐标；没有将派生坐标冒称原capture保存的坐标。小实例为各域原面积最小四分位，不用background名单。',
        'instance_profile_summary.csv将margin下降与未下降分开列出inside−ring描述；分组差异不是危险线段分类能力。',
        '小gap不等于八张绝对距离都小；GT覆盖的区分度不等于能识别危险线段。',
        '区间不跨0.5不是完全控制混杂或独立检测增益。两域此前参与过方法研究，不是完全未触碰的盲测。',
        '本轮不设置阈值/门控、不选层、不拟合Target MAD、不宣称泛化或逐实例零损失；联合方向见跨域汇总。']
    (out/'objective_report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')


def verified_result(path,domain):
    if domain=='background' and path.is_file():return verified_catalog(path)
    status=ip.old.read_json(path/'status.json')
    if status['status']!='diagnostic_complete' or status['images']!=(297 if domain=='background' else sum(ip.COUNTS[domain])):
        raise ValueError('incomplete result: '+domain)
    names=('conditional_comparisons.csv','matching_coverage.csv','instance_profile_summary.csv','input_manifest.json')
    for n in names:
        if ip.sha(path/n)!=status['files'][n]:raise ValueError('modified result: '+str(path/n))
    manifest=ip.old.read_json(path/'input_manifest.json')
    frozen=manifest['frozen']
    if ip.digest(frozen)!=manifest['identity']:raise ValueError('result manifest identity differs')
    kernels=frozen.get('statistical_kernels') or {k:frozen['code'][k] for k in ('profiles.py','conditional.py')}
    base=ip.ROOT/'DINOv3/MADEqual/aebad_s_support_distance_diagnostic'
    if any(ip.sha(base/k)!=v for k,v in kernels.items()):raise ValueError('background statistical kernel version differs')
    if domain!='background':
        if frozen['domain']!=domain:raise ValueError('result domain differs')
        if frozen.get('parent_origin','historical_summary_archive')=='historical_summary_archive' and frozen['summary_sha256']!=ip.SUMMARY_SHA:
            raise ValueError('historical parent identity differs')
    return dict(status=status,parent_sha256=frozen.get('summary_sha256'),parent_origin=frozen.get('parent_origin','historical_summary_archive'),
                comparisons=csv_file(path/'conditional_comparisons.csv'),coverage=csv_file(path/'matching_coverage.csv'),
                instance_summary=csv_file(path/'instance_profile_summary.csv'))


def verified_catalog(path):
    """Small frozen reference for code-only clones; original output stays private."""
    if ip.sha(path)!=CATALOG_SHA:raise ValueError('modified background discovery catalog')
    data=ip.old.read_json(path)
    if data['format']!='support_distance_background_catalog_v1' or data['domain']!='background' or data['images']!=297:
        raise ValueError('wrong background discovery scope')
    base=ip.ROOT/'DINOv3/MADEqual/aebad_s_support_distance_diagnostic'
    if any(ip.sha(base/n)!=h for n,h in data['statistical_kernels'].items()):raise ValueError('catalog statistical kernel differs')
    expected={(l,k,m) for l in ip.LAYERS for k in ip.MAIN_FEATURES for m in ('uncontrolled_effective','matched_b_delta')}
    if {(int(r['layer']),r['feature'],r['mode']) for r in data['comparisons']}!=expected:
        raise ValueError('incomplete frozen primary descriptors')
    return dict(status=dict(status='frozen_discovery_catalog',images=297),parent_sha256=CATALOG_SHA,
        parent_origin='frozen_discovery_catalog',comparisons=data['comparisons'],coverage=data['coverage'],
        instance_summary=data['instance_summary'])


def summarize(background,domain_results,out):
    rows=[];available={};notes=[]
    for domain,path in [('background',background),*[(d,domain_results.get(d)) for d in ip.DOMAINS]]:
        if path is None or not (path.is_file() or (path/'status.json').exists()):
            available[domain]=None;notes.append(f'{domain}: pending (no completed analysis)');continue
        available[domain]=verified_result(path,domain)
    completed=[available[d] for d in ip.DOMAINS if available.get(d) is not None]
    if len(completed)==2 and completed[0].get('parent_sha256')!=completed[1].get('parent_sha256'):
        raise ValueError('cannot mix domains from different parent runs')
    for domain,result in available.items():
        lookup={} if result is None else {(int(r['layer']),r.get('mode'),r['feature']):r for r in result['comparisons']}
        coverage=[] if result is None else result['coverage']
        for l in ip.LAYERS:
            for key in ip.MAIN_FEATURES:
                r=lookup.get((l,'matched_b_delta',key));raw=lookup.get((l,'uncontrolled_effective',key),{})
                row=dict(domain=domain,layer=l,feature=key,role='discovery_reference' if domain=='background' else 'cross_condition_validation',
                    status='pending' if result is None else r.get('status','not_computable') if r else 'not_computable')
                row['parent_origin']='pending' if result is None else result.get('parent_origin','historical_summary_archive')
                row['parent_source_sha256']='' if result is None else result.get('parent_sha256','')
                if result is not None and not r:
                    reason=next((z.get('status') for z in result['comparisons'] if int(z['layer'])==l and z['feature']==key),None)
                    row['not_computable_reason']=reason or 'required_feature_row_missing'
                if r:
                    for k in ('diagnostic_AUROC','diagnostic_AUROC_CI_low','diagnostic_AUROC_CI_high','diagnostic_AUROC_valid_bootstraps',
                        'mean_difference_GT_minus_normal','standardized_difference','b_mean_difference','delta_mean_difference',
                        'b_standardized_difference','delta_standardized_difference',
                        'feature_valid_b_standardized_difference','feature_valid_delta_standardized_difference','residual_confounding_flag',
                        'group0_finite_images','group1_finite_images','group0_finite_patches','group1_finite_patches',
                        'group0_valid_mass','group1_valid_mass','group0_max_finite_image_share','group1_max_finite_image_share'):
                        row[k]=r.get(k,'')
                    row['uncontrolled_AUROC']=raw.get('diagnostic_AUROC','')
                    for field in ('b_mean_difference','delta_mean_difference','b_standardized_difference','delta_standardized_difference'):
                        row['uncontrolled_'+field]=raw.get(field,'')
                    for g in (0,1):
                        mass=r.get(f'group{g}_valid_mass','')
                        row[f'group{g}_NA_or_uncovered_feature_mass']=1-float(mass) if mass!='' else None
                    auc=r.get('diagnostic_AUROC','')
                    row['direction']='NA' if auc=='' else 'GT_smaller' if float(auc)<.5 else 'GT_larger' if float(auc)>.5 else 'no_ordering_direction'
                    low,high=r.get('diagnostic_AUROC_CI_low',''),r.get('diagnostic_AUROC_CI_high','')
                    row['CI_excludes_half']='NA' if '' in (low,high) else float(high)<.5 or float(low)>.5
                    row['signed_mean_change_GT_minus_normal']=r.get('mean_difference_GT_minus_normal','')
                    # Same image-balanced instance kernel as the discovery analysis.
                    for mode,source_mode in (('own','own'),('fixed','fixed_UNION')):
                        for group in ('decreased','not_decreased'):
                            z=next((z for z in result['instance_summary'] if z['mode']==source_mode and z['margin_group']==group
                                    and z.get('layer','')!='' and int(z['layer'])==l and z['feature']==key),{})
                            row[f'{mode}_{group}_inside_minus_ring_mean']=z.get('image_balanced_mean','')
                            row[f'{mode}_{group}_valid_images']=z.get('valid_images','')
                for g in (0,1):
                    c=next((x for x in coverage if int(x['layer'])==l and x['row_kind']=='matched_group' and int(x['group'])==g),{})
                    for k in ('images_covered','images_effective','uncovered_probability_mass','effective_overlap_coefficient','max_matched_image_share'):
                        row[f'group{g}_{k}']=c.get(k,'')
                rows.append(row)
    reference={(r['layer'],r['feature']):r for r in rows if r['domain']=='background'}
    for r in rows:
        ref=reference[r['layer'],r['feature']]
        r['same_direction_as_background']=('NA' if 'direction' not in r or r['direction']=='NA' or ref.get('direction','NA')=='NA'
                                             else r['direction']==ref['direction'])
        for mode in ('own','fixed'):
            a=r.get(f'{mode}_decreased_inside_minus_ring_mean','');b=r.get(f'{mode}_not_decreased_inside_minus_ring_mean','')
            ra=ref.get(f'{mode}_decreased_inside_minus_ring_mean','');rb=ref.get(f'{mode}_not_decreased_inside_minus_ring_mean','')
            r[mode+'_margin_group_mean_difference']=float(a)-float(b) if '' not in (a,b) else None
            r[mode+'_margin_association_same_sign_as_background']=('NA' if '' in (a,b,ra,rb) else
                np.sign(float(a)-float(b))==np.sign(float(ra)-float(rb)))
    ip.write_csv(out/'cross_domain_summary.csv',rows)
    text=['# 逐支持距离诊断：跨条件汇总','',
        'background仅作发现阶段参照，不重跑、不混合patch、不计算跨域总AUC。',
        'gap2、gap4、gap8四层全部列出。illumination/view曾用于方法研究，不是完全未触碰的盲测。','']
    for l in ip.LAYERS:
        for key in ip.MAIN_FEATURES:
            selected=[r for r in rows if r['layer']==l and r['feature']==key]
            text.append(f"- L{l}/{key}："+'；'.join(f"{r['domain']}={r.get('diagnostic_AUROC','pending')} ({r.get('direction',r['status'])})" for r in selected))
    all_complete=all(v is not None for v in available.values())
    for d in ip.DOMAINS:
        rr=[r for r in rows if r['domain']==d]
        computable=[r for r in rr if r.get('same_direction_as_background')!='NA']
        flagged=[r for r in rr if str(r.get('residual_confounding_flag','')).lower()=='true']
        text+=['',f"{d}：主描述量可比较{len(computable)}/12项，点估计方向重复{sum(r['same_direction_as_background'] for r in computable) if computable else 0}项；残余混杂标记{len(flagged)}项。未完成项不计作方向重复。"]
        for mode in ('own','fixed'):
            usable=[r for r in rr if r.get(mode+'_margin_association_same_sign_as_background')!='NA']
            text.append(f"{mode} margin分组的inside−ring均值差：可比较{len(usable)}/12项，同号{sum(r[mode+'_margin_association_same_sign_as_background'] for r in usable) if usable else 0}项；这只是分组关联，不是margin损失预测能力。")
    text+=['','完整分析是否齐全：'+str(all_complete),
        '方向重复不等于独立增益。请同时检查区间、共同质量、图像权重和b/delta残余差异。',
        'margin关联另见各域实例分组：能区分正常/GT覆盖位置，不等于能识别危险线段或保护margin。',
        '小gap不是八张支持的绝对距离都小；不能据诊断AUC设计门限、选择层或组合模型。',
        '在两域未完成或残余混杂明显时，当前证据不足以据此直接进入门控方法设计；先如实保留缺项或不确定性。',*notes]
    (out/'objective_report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    ip.write_json(out/'status.json',dict(status='cross_domain_summary_complete' if all_complete else 'partial_evidence',
        domains_complete=[d for d,v in available.items() if v is not None],pending=[d for d,v in available.items() if v is None],
        rows=len(rows),aggregate_AUROC_computed=False))
    return rows
