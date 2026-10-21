"""Per-image frozen evidence, then conditional group descriptions. No new method."""
from __future__ import annotations
import csv
import gzip
import math
from pathlib import Path
import numpy as np
from . import inputs as ip
from .profiles import (FEATURES,LAYERS,layer_profiles,high_response,reconstruct,adaptive_max)
from .spatial import instance_analysis,scalar_profile
from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic.numerics import check

STAT_FIELDS=(*FEATURES,'b','delta','effective_reduction','group','image_id')


class GzipRows:
    def __init__(self,path):
        self.file=gzip.open(path,'wt',encoding='utf-8',newline='');self.writer=None
    def write(self,row):
        clean={k:('' if v is None or isinstance(v,(float,np.floating)) and not np.isfinite(v) else v) for k,v in row.items()}
        if self.writer is None:
            self.writer=csv.DictWriter(self.file,list(clean));self.writer.writeheader()
        self.writer.writerow(clean)
    def close(self):
        if self.writer is None:self.file.write('status\nno_rows\n')
        self.file.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()


def one_image(ctx,sample,image_id):
    checks=[]
    with ip.Archive(ctx['paths']['raw_archive'],ctx['raw_index']) as archive:
        raw,meta,segments=ip.load_image(ctx,archive,sample)
    union,segment=ip.original_distances(raw,meta,segments)
    profiles=layer_profiles(ctx,raw,meta,segments,union,segment)
    z,m,_=reconstruct(union,raw['positions'],tuple(meta['grid']),tuple(meta['output_shape']),ctx['medians'],ctx['scales'])
    native_union=z.mean(axis=0);maps={'UNION':m['FUSED']};m=None;z=None
    z,m,_=reconstruct(segment,raw['positions'],tuple(meta['grid']),tuple(meta['output_shape']),ctx['medians'],ctx['scales'])
    maps['SEGMENT']=m['FUSED'];m=None;z=None
    sid=sample['sample_id']
    for branch,historical in (('UNION','K8_UNION'),('SEGMENT','K8_SEGMENT')):
        check(checks,sid,'historical_'+branch+'_image_peak',float(maps[branch].max()),ctx['history']['scores'][sid,historical])
    check(checks,sid,'formal_UNION_image_peak',float(maps['UNION'].max()),sample['union_score'])
    with ip.Dataset(ctx) as data:gt=data.gt(sample)
    physical=high_response(native_union) if sample['label']==0 else np.flatnonzero(adaptive_max(gt,tuple(meta['grid'])))
    inverse=np.empty(len(raw['positions']),np.int64);inverse[raw['positions']]=np.arange(len(inverse))
    offsets=inverse[physical]
    selected=[];columns=[]
    group=int(sample['label'])
    for j,l in enumerate(LAYERS):
        p=profiles[j];n=len(physical)
        col={k:np.asarray(p[k])[offsets].copy() for k in (*FEATURES,'b','delta','effective_reduction')}
        col.update(group=np.full(n,group,np.int8),image_id=np.full(n,image_id,np.int32))
        columns.append(col)
        for physical_index,offset in zip(physical,offsets):
            selected.append(dict(sample_id=sid,image_id=image_id,relative_path=sample['relative_path'],
                group='normal_high_response' if group==0 else 'GT_covered',layer=l,
                physical_patch=int(physical_index),compact_offset=int(offset),patch_y=int(physical_index//meta['grid'][1]),
                patch_x=int(physical_index%meta['grid'][1]),native_height=meta['grid'][0],native_width=meta['grid'][1],
                fixed10_normal=sid in ctx['history']['fixed10'],selected_patch_count_in_image=n,
                unconditional_image_balanced_weight=1/((85 if group==0 else 212)*n),
                effective_support_status='available' if np.isfinite(p['effective_support_count'][offset]) else 'no_positive_target_improvement',
                source_index=int(raw['source_index'][j,offset]),
                **{f'target_index_{k+1}':int(raw['target_index'][k,j,offset]) for k in range(8)},
                **scalar_profile(p,offset)))
    links=[];stencils=[];instances=[]
    if group:
        instances=instance_analysis(ctx,sample,gt,maps,raw,meta,profiles,checks,links.append,stencils.append)
    if not all(c['verified'] for c in checks):
        return dict(sample_id=sid,checks=checks,failed=True)
    figure=None
    if sid in ('test_0977','test_0023','test_0545'):
        figure=dict(sample=sample,maps=maps,gt=gt,native=native_union,selected_physical=physical,
                    anchors=[r for r in instances])
    return dict(sample_id=sid,checks=checks,failed=False,selected=selected,columns=columns,
                links=links,stencils=stencils,instances=instances,figure=figure)


def instance_summaries(rows):
    output=[]
    for mode,field in (('own','own_margin_change'),('fixed_UNION','fixed_margin_change')):
        for group in ('decreased','not_decreased','not_computable'):
            selected=[r for r in rows if ('not_computable' if r.get(field) is None else
                       'decreased' if r[field]<-1e-6 else 'not_decreased')==group]
            base=dict(mode=mode,margin_group=group,instances=len(selected),images=len({r['sample_id'] for r in selected}),
                      small_instances=sum(r['small_original_three'] for r in selected))
            output.append(dict(**base,feature='COUNT_ONLY'))
            # Fixed-UNION stencil patterns are descriptive even when stratifying own margins.
            for l in LAYERS:
                for feature in FEATURES:
                    values=[];image_ids=[]
                    for r in selected:
                        a=r.get(f'fixed_instance_L{l}_{feature}');b=r.get(f'fixed_ring_L{l}_{feature}')
                        if a is not None and b is not None:values.append(a-b);image_ids.append(r['sample_id'])
                    if values:
                        image_means=[float(np.mean([v for v,i in zip(values,image_ids) if i==sid])) for sid in sorted(set(image_ids))]
                        output.append(dict(**base,layer=l,feature=feature,descriptor_direction='fixed_inside_minus_ring',
                            valid_instances=len(values),valid_images=len(image_means),
                            image_balanced_mean=float(np.mean(image_means)),image_median=float(np.median(image_means))))
    return output


def make_report(out,comparisons,coverage,instances,instance_summary):
    text=['# Source＋八张Target逐支持距离：条件诊断','',
        '仅AeBAD-S/background开发域，原85正常＋212异常、原seed42 K8；没有新检测器、门控、Target MAD或模型推理。',
        '支持距离为query到某张支持图全部patch的最小值，不是同一物理位置，也不是独立正常性投票。',
        '正常组是每图UNION原生融合响应最高ceil(1%×patch数)；GT组为adaptive_max覆盖patch，可能混合缺陷与正常。',
        '图像均衡权重；条件分析只使用两组的有效SEGMENT缩距位置，固定每层5×5加权分位格及共同质量最小值权重。',
        '200次PCG64 seed42的分组图像级bootstrap，固定分位边界、每次重新计算共同覆盖权重。区间不是多重检验后独立发现。','',
        '| 层 | 描述量 | 未控制AUROC | 条件AUROC [95% CI] | 条件均值差 GT−正常 | 条件控制量仍不平衡 |',
        '|---|---|---:|---|---:|---|']
    by={(r['layer'],r.get('mode'),r['feature']):r for r in comparisons}
    for l in LAYERS:
        for feature in FEATURES:
            raw=by.get((l,'uncontrolled_effective',feature),{})
            r=by.get((l,'matched_b_delta',feature),{})
            def f(v):return 'NA' if v is None else f'{float(v):.5g}'
            text.append(f"| {l} | {feature} | {f(raw.get('diagnostic_AUROC'))} | {f(r.get('diagnostic_AUROC'))} [{f(r.get('diagnostic_AUROC_CI_low'))}, {f(r.get('diagnostic_AUROC_CI_high'))}] | {f(r.get('mean_difference_GT_minus_normal'))} | {r.get('residual_confounding_flag','NA')} |")
    text+=['','## 共同覆盖与案例支配','']
    for r in coverage:
        if r['row_kind']=='matched_group':
            text.append(f"- L{r['layer']} group{r['group']}：有效图{r['images_effective']}、覆盖图{r['images_covered']}；未覆盖质量{r['uncovered_probability_mass']:.4%}；共同质量{r['effective_overlap_coefficient']:.4g}；最大单图匹配权重{r['max_matched_image_share']:.4%}。")
    text+=['','## 未控制差异是否保留','']
    for l in LAYERS:
        supported=[];uncertain=[]
        for k in FEATURES:
            r=by.get((l,'matched_b_delta',k),{})
            low,high=r.get('diagnostic_AUROC_CI_low'),r.get('diagnostic_AUROC_CI_high')
            if low is not None and (low>.5 or high<.5):supported.append(k)
            else:uncertain.append(k)
        text.append(f"- L{l}：条件AUROC区间不跨0.5的描述量：{', '.join(supported) or '无'}；仍含0.5或不可计算：{', '.join(uncertain) or '无'}。这只是预设属性的描述性区间，若覆盖不足或残余混杂存在，不能据此定为可靠独立线索。")
    text+=['','## Source/Top-2、八张分布形状与重叠','']
    for l in LAYERS:
        for label,keys in (('Source/Top-2',('source_advantage','gap2','loo_max')),
                            ('八张分布形状',('gap4','gap8','effective_support_count'))):
            descriptions=[]
            for k in keys:
                r=by.get((l,'matched_b_delta',k),{})
                auc=r.get('diagnostic_AUROC')
                q=[r.get('group0_p25'),r.get('group0_p75'),r.get('group1_p25'),r.get('group1_p75')]
                overlap=('NA' if any(v is None for v in q) else
                         'IQR交叠' if max(q[0],q[2])<=min(q[1],q[3]) else 'IQR不交叠')
                descriptions.append(k+': '+('NA' if auc is None else f'AUC={auc:.4f}')+' / '+overlap)
            text.append(f"- L{l}/{label}："+'；'.join(descriptions)+'。')
    text+=['IQR交叠只描述中间50%范围，不是完整分布相同；AUROC区间包含0.5也不是等价性证明。原值、加权CDF及有效NA质量必须一起查看，不能挑出方向有利的单项代替总体判断。']
    available=[r for r in instances if r['response_status']=='available']
    text+=['','## 实例margin与支持模式','',
        f"保留{len(instances)}个原始实例，其中{len(available)}个可计算，{len(instances)-len(available)}个保持NA；可计算实例来自{len({r['sample_id'] for r in available})}张图。",
        '固定位置变化=环带降响应−内部降响应；自身重新取峰值另列，不能相互替代。',
        'instance_profile_summary.csv按margin下降/未下降分别给出固定峰值处inside−ring的支持描述，并先图内实例等权，再图像等权；它不是语义匹配的因果证明。']
    for r in instance_summary:
        if r['feature']=='COUNT_ONLY':text.append(f"- {r['mode']}/{r['margin_group']}：{r['instances']}实例，{r['images']}图，原最小三实例中{r['small_instances']}个。")
    text+=['固定UNION位置的inside−ring模式，按自身margin下降与未下降分组（先图内实例等权，再图像等权）：']
    for l in LAYERS:
        for k in ('source_advantage','gap2','gap8','effective_support_count'):
            values={r['margin_group']:r for r in instance_summary if r['mode']=='own' and r.get('layer')==l and r['feature']==k}
            descriptions=[]
            for g in ('decreased','not_decreased'):
                r=values.get(g)
                descriptions.append(g+': '+(f"mean={r['image_balanced_mean']:+.5g}, images={r['valid_images']}" if r else 'NA'))
            text.append(f"- L{l}/{k}："+'；'.join(descriptions)+'。')
    text+=['这些分组差异是事后关联，不能断言某一支持模式造成margin损失，更不能据此选择部署阈值。']
    text+=['','## 结论边界及下一步可检验假设','',
        'Source优势、gap2、LOO主要反映Source/Top-2；gap4/gap8与有效支持集中度描述八张距离的形状。即使后者在条件比较中保留差异，也没有控制全部Top-2几何，不能声称额外独立增益。',
        'LOO八项在精确min检索下主要由Source和前两名Target决定，不是额外独立八维证据。',
        'AUROC保持GT覆盖为正类，低于0.5不翻转；它不是Image/Pixel AUROC，也没有形成可部署检测器。',
        '若多个层的预设分布形状在足够覆盖、较小残余混杂及图像级区间下保持方向，下一项仅是检验该形状是否能在未参与开发的域重复；本轮不设计规则。',
        '若匹配后消失、区间重叠或共同覆盖不足，当前标量距离尚不足以给出可靠区分依据；这不证明研究任务不可能。',
        '详细覆盖、残余b/delta标准化差异和NA质量见CSV。不能从标量缓存恢复Q、Source锚点语义、扰动方向或互近邻图。',
        '没有改变旧结果或总记录，不宣称跨域泛化、逐实例无损、解决真实扰动或达到SOTA。']
    (out/'objective_report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
