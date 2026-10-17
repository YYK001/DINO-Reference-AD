"""Three-branch validation and aligned instance accounting AFTER prediction freeze."""
from __future__ import annotations
import math
import time
from pathlib import Path
import numpy as np
from . import inputs as ip
from . import prediction
from .verification import historical_score,audit
from .method import BRANCHES,LAYERS,U,S,P
from DINOv3.MADEqual.aebad_s_neighborhood_segment_p0.analysis import score_maps,peak_position,array_sha,verify_map_record
from DINOv3.MADEqual.aebad_s_neighborhood_segment_p0.prediction import bounded_images,protect_dependencies
from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import image_metrics,pair_errors
from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic.numerics import adaptive_max
from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic.numerics import check
from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic.metrics import exact_binary,resource_bytes
from DINOv3.MADEqual.aebad_s_union_diagnostics.spatial import components,instance_regions,resized_mask

FIELDS={U:'union_distance',S:'segment_distance',P:'pair_distance'}
COMPARISONS=((S,U),(P,U),(P,S))
METRICS=('image_AUROC','image_AP','pixel_AUROC','pixel_AP','AUPRO')


def spatial(sample,mask,maps,grid):
    labelled,descriptors=components(mask);rows=[]
    for d in descriptors:
        inside,ring=instance_regions(labelled==d['instance_id'],mask,maps[U].shape)
        coverage=int(adaptive_max(labelled==d['instance_id'],grid).sum())
        anchors={name:peak_position(maps[U],r) for name,r in (('instance',inside),('ring',ring))}
        for branch,response in maps.items():
            own={name:peak_position(response,r) for name,r in (('instance',inside),('ring',ring))}
            values={name:float(response[pos]) if pos is not None else None for name,pos in own.items()}
            fixed={name:float(response[pos]) if pos is not None else None for name,pos in anchors.items()}
            row=dict(domain=sample['domain'],sample_id=sample['sample_id'],relative_path=sample['relative_path'],**d,branch=branch,
                response_status='mapped_instance_disappeared' if not inside.any() else 'empty_ring' if not ring.any() else 'available',
                patch_coverage=coverage,
                instance_area_output_pixels=int(inside.sum()),ring_area_output_pixels=int(ring.sum()),
                ring_radius_original_pixels=16,ring_metric='Chebyshev_excludes_all_GT',
                instance_max=values['instance'],ring_max=values['ring'],
                margin=values['instance']-values['ring'] if all(v is not None for v in values.values()) else None,
                fixed_union_instance_response=fixed['instance'],fixed_union_ring_response=fixed['ring'],
                fixed_union_margin=fixed['instance']-fixed['ring'] if all(v is not None for v in fixed.values()) else None,
                fixed_peak_source='this_domain_frozen_Union_first_row_major_not_historical_peak')
            for name in own:
                for kind,positions in (('own',own),('union_anchor',anchors)):
                    pos=positions[name];row[f'{kind}_{name}_y']=pos[0] if pos else None;row[f'{kind}_{name}_x']=pos[1] if pos else None
            rows.append(row)
    for row in rows:
        for ref in (U,S):
            baseline=next(r for r in rows if r['instance_id']==row['instance_id'] and r['branch']==ref)
            for field in ('instance_max','ring_max','margin','fixed_union_instance_response','fixed_union_ring_response','fixed_union_margin'):
                row[field+'_change_vs_'+ref]=row[field]-baseline[field] if row[field] is not None and baseline[field] is not None else None
    return rows


def mechanisms(ctx,sample,mask,arrays,meta):
    region=np.ones(len(arrays['positions']),bool) if sample['label']==0 else adaptive_max(mask,tuple(meta['grid'])).ravel()[arrays['positions']]
    group='normal_all_patch' if sample['label']==0 else 'anomaly_GT_patch'
    base=(arrays['union_distance'].astype(float)-arrays['segment_distance'])/(ctx['scales'][:,None]+1e-8)
    rows=[]
    for b in (S,P):
        delta=(arrays['union_distance'].astype(float)-arrays[FIELDS[b]])/(ctx['scales'][:,None]+1e-8)
        for off,l in enumerate(LAYERS):
            old=float(base[off,region].sum());new=float(delta[off,region].sum())
            old_count=int(((arrays['union_distance'][off]-arrays['segment_distance'][off]>1e-7+1e-6*np.abs(arrays['union_distance'][off]))&region).sum())
            new_count=int(((arrays['union_distance'][off]-arrays[FIELDS[b]][off]>1e-7+1e-6*np.abs(arrays['union_distance'][off]))&region).sum())
            rows.append(dict(domain=sample['domain'],sample_id=sample['sample_id'],branch=b,layer=l,group=group,
                selected_patches=int(region.sum()),R_value=float(delta[off,region].mean()) if region.any() else None,
                original_SEGMENT_reduction=old,signed_scaled_reduction=new,extra_reduction_retained_ratio=new/old if old!=0 else None,
                original_SEGMENT_reduced_patches=old_count,branch_reduced_patches=new_count,
                reduction_positions_retained_ratio=new_count/old_count if old_count else None,
                pair_accept_ratio=float(arrays['pair_gate'][off].mean()),
                accepted_reduction_in_group=float(arrays['pair_gate'][off].ravel()[arrays['positions']][region].mean()) if region.any() else None,
                reference=U,normalization='float32_Source_patch_scale_plus_1e-8'))
    return rows


def mechanism_summary(rows,domains):
    result=[]
    for domain in domains:
        for b in (S,P):
            for layer in (*LAYERS,'MEAN4'):
                fields={}
                for group,name in (('normal_all_patch','R_N'),('anomaly_GT_patch','R_A')):
                    selected=[r for r in rows if r['domain']==domain and r['branch']==b and r['group']==group]
                    for field,key in (('R_value',name),('extra_reduction_retained_ratio',group+'_retained_amount')):
                        means=[]
                        for l in LAYERS if layer=='MEAN4' else (layer,):
                            values=[r[field] for r in selected if r['layer']==l and r[field] is not None]
                            if values:means.append(float(np.mean(values)))
                        fields[key]=float(np.mean(means)) if means else None
                    fields[group+'_zero_denominators']=sum(r['extra_reduction_retained_ratio'] is None for r in selected if layer=='MEAN4' or r['layer']==layer)
                result.append(dict(domain=domain,branch=b,layer=layer,**fields,G=fields['R_N']-fields['R_A'] if fields['R_N'] is not None and fields['R_A'] is not None else None,
                                   aggregation='image_equal_within_layer_then_layer_equal',reference=U))
    return result


def group_small_instances(rows,domains):
    groups=[];small=set()
    for domain in domains:
        originals=[r for r in rows if r['domain']==domain and r['branch']==U]
        chosen=sorted(originals,key=lambda r:(r['area_original_pixels'],r['sample_id'],r['instance_id']))[:math.ceil(len(originals)/4)]
        small|={(r['domain'],r['sample_id'],r['instance_id']) for r in chosen}
        for r in originals:
            groups.append(dict(domain=domain,sample_id=r['sample_id'],instance_id=r['instance_id'],area_original_pixels=r['area_original_pixels'],
                in_smallest_area_quartile=(r['domain'],r['sample_id'],r['instance_id']) in small,
                total_instances=len(originals),small_instance_count=math.ceil(len(originals)/4),response_status=r['response_status'],
                group_definition='smallest_ceil_N_over_4_original_area_tie_sample_id_instance_id'))
    for r in rows:r['smallest_area_quartile']=(r['domain'],r['sample_id'],r['instance_id']) in small
    return groups


def margin_repairs(rows):
    aligned={}
    for r in rows:aligned.setdefault((r['domain'],r['sample_id'],r['instance_id']),{})[r['branch']]=r
    details=[];summary=[];tol=1e-6
    for (domain,sid,iid),branches in aligned.items():
        if set(branches)!=set(BRANCHES):raise ValueError('instance branch alignment incomplete')
        for field in ('margin','fixed_union_margin'):
            u,s,p=(branches[b][field] for b in BRANCHES)
            record=dict(domain=domain,sample_id=sid,instance_id=iid,definition=field,smallest_area_quartile=branches[U]['smallest_area_quartile'],
                computable=all(v is not None for v in (u,s,p)),response_status=branches[U]['response_status'])
            if record['computable']:
                ds,dp=s-u,p-u;prior=ds < -tol
                record.update(Union=u,SEGMENT=s,PAIR=p,segment_change_vs_union=ds,pair_change_vs_union=dp,pair_change_vs_segment=p-s,
                    original_segment_loss=prior,old_loss_improved=prior and p>s+tol,old_loss_fully_repaired=prior and dp>=-tol,
                    old_loss_partially_repaired=prior and dp < -tol and p>s+tol,remaining_old_loss=prior and dp < -tol,
                    worsened_old_loss=prior and p<s-tol,new_PAIR_loss=(not prior) and dp < -tol)
                if field=='margin':
                    di=branches[P]['instance_max']-branches[S]['instance_max'];dr=branches[P]['ring_max']-branches[S]['ring_max']
                    record.update(instance_recovery_vs_segment=di,ring_recovery_vs_segment=dr,
                        ring_recovers_more_and_margin_worsens=dr>di+tol and p<s-tol,
                        ring_recovers_but_defect_does_not=dr>tol and abs(di)<=tol and p<s-tol)
            details.append(record)
    for domain in dict.fromkeys(r['domain'] for r in rows):
        for group in ('all','smallest_area_quartile'):
            for field in ('margin','fixed_union_margin'):
                selected=[r for r in details if r['domain']==domain and r['definition']==field and (group=='all' or r['smallest_area_quartile'])]
                valid=[r for r in selected if r['computable']]
                summary.append(dict(domain=domain,group=group,definition=field,instances=len(selected),computable=len(valid),not_computable=len(selected)-len(valid),
                    **{k:sum(bool(r.get(k)) for r in valid) for k in ('original_segment_loss','old_loss_improved','old_loss_fully_repaired','old_loss_partially_repaired',
                        'remaining_old_loss','worsened_old_loss','new_PAIR_loss','ring_recovers_more_and_margin_worsens','ring_recovers_but_defect_does_not')},tolerance=tol))
    return details,summary


def metric_comparisons(metrics):
    rows=[]
    for domain in dict.fromkeys(r['domain'] for r in metrics):
        lookup={r['branch']:r for r in metrics if r['domain']==domain}
        for branch,ref in COMPARISONS:
            for key in METRICS:
                a,b=lookup[branch].get(key),lookup[ref].get(key)
                rows.append(dict(domain=domain,branch=branch,reference=ref,metric=key,value=a,reference_value=b,
                    delta=a-b if a is not None and b is not None else None,units='fraction',direction='branch_minus_reference',
                    status='available' if a is not None and b is not None else 'pending_AUPRO'))
    return rows


def append_macro(metrics):
    if {r['domain'] for r in metrics}!=set(ip.DOMAINS):return metrics
    result=list(metrics)
    for branch in BRANCHES:
        values=[next(r for r in metrics if r['domain']==d and r['branch']==branch) for d in ip.DOMAINS]
        result.append(dict(domain='macro_illumination_view',branch=branch,
            **{key:float(np.mean([r[key] for r in values])) if all(r.get(key) is not None for r in values) else None for key in METRICS},
            aggregation='two_validation_domains_unweighted_background_excluded',metric_origin='domain_macro'))
    return result


def run(ctx,prediction_dir,output,workers=4,memory_budget=8<<30):
    started=time.perf_counter();pred,manifest,samples=prediction.verified(ctx,prediction_dir)
    protect_dependencies(output,pred);domains=manifest['domains'];out=ip.new_output(ctx,output,'evaluate')
    records={(r['domain'],r['sample_id']):r for r in manifest['records']}
    frozen_scores={(r['domain'],r['sample_id'],r['branch']):float(r['image_score']) for r in ip.csv_file(pred/'prediction_scores.csv')}
    checks=[];score_rows=[];instances=[];mechanism=[];map_records=[];rankings=[];metrics=[];costs={}
    for domain in domains:
        selected=[s for s in samples if s['domain']==domain]
        pixels=sum((s['image_height']//4)*(s['image_width']//4) for s in selected)
        if resource_bytes(pixels)>memory_budget or ip.common.available_ram()<resource_bytes(pixels)+(512<<20):raise MemoryError('insufficient RAM/budget for exact full pixels; no approximation')
        import shutil
        if shutil.disk_usage(out).free < pixels*13+(512<<20):raise OSError('insufficient G-drive pixel scratch space')
        scratch=out/('_scratch_'+domain);scratch.mkdir()
        label=np.memmap(scratch/'labels.bin',mode='w+',dtype=np.uint8,shape=(pixels,))
        maps_mmap={b:np.memmap(scratch/(b+'.bin'),mode='w+',dtype=np.float32,shape=(pixels,)) for b in BRANCHES}
        offset=0;tick=time.perf_counter();domain_scores=[]
        def one(sample):
            arrays,meta=prediction.load(ctx,pred,records[(domain,sample['sample_id'])],sample)
            local=[]
            try:sc=prediction.scores_for(ctx,arrays,meta,sample,local)
            except Exception:
                failure=out/'failures';failure.mkdir(exist_ok=True)
                ip.write_json(failure/f'{sample["domain"]}_{sample["sample_id"]}.json',dict(checks=local))
                raise
            scored={b:score_maps(ctx,arrays[key],dict(meta,positions=arrays['positions'])) for b,key in FIELDS.items()}
            maps={b:v[2] for b,v in scored.items()}
            for b,(current,ch1,response,error) in scored.items():
                check(local,sample['sample_id'],'actual_map_CURRENT_'+b,float(response.max()),current,domain=domain)
                check(local,sample['sample_id'],'frozen_prediction_CH1_'+b,ch1,frozen_scores[(domain,sample['sample_id'],b)],domain=domain)
            historical_score(local,sample,'actual_Union_map_historical_CURRENT',float(maps[U].max()),sample['union_score'])
            if not all(r['verified'] for r in local):
                failure=out/'failures';failure.mkdir(exist_ok=True)
                ip.write_json(failure/f'{sample["domain"]}_{sample["sample_id"]}.json',dict(checks=local))
                raise ValueError('actual response/backend changed; no completion')
            # Only now, after every prediction in scope was frozen, may GT enter.
            with ip.Data(ctx['paths']['data']) as dataset:mask=dataset.gt(sample)
            return sample,maps,resized_mask(mask,meta['output_shape']),sc,local,spatial(sample,mask,maps,tuple(meta['grid'])),mechanisms(ctx,sample,mask,arrays,meta)
        try:
            for sample,maps,mask,sc,local,sp,mech in bounded_images(selected,one,workers):
                n=mask.size;label[offset:offset+n]=mask.ravel()
                for b,a in maps.items():
                    maps_mmap[b][offset:offset+n]=a.ravel()
                    map_records.append(dict(domain=domain,sample_id=sample['sample_id'],branch=b,output_height=a.shape[0],output_width=a.shape[1],
                        map_float32_sha256=array_sha(a),mask_uint8_sha256=array_sha(mask.astype(np.uint8))))
                offset+=n;checks.extend(local);score_rows.extend(dict(r,label=sample['label']) for r in sc);domain_scores.extend(sc);instances.extend(sp);mechanism.extend(mech)
                print(f'[crossdomain evaluate] {domain} {sample["sample_id"]}',flush=True)
            if offset!=pixels:raise ValueError('incomplete exact pixel population')
            ip.write_csv(out/'numerical_verification.csv',checks)
            if not checks or not all(r['verified'] for r in checks):raise ValueError('CURRENT baseline changed; metrics cannot be published')
            costs[domain+'_maps_GT_spatial_seconds']=time.perf_counter()-tick
            labels=np.asarray([s['label'] for s in selected]);lookup={(r['sample_id'],r['branch']):r['image_score'] for r in domain_scores}
            vectors={b:np.asarray([lookup[(s['sample_id'],b)] for s in selected]) for b in BRANCHES}
            for b in BRANCHES:
                metric=image_metrics(labels,vectors[b]);tick=time.perf_counter()
                if ip.common.available_ram()<resource_bytes(pixels)+(512<<20):raise MemoryError('RAM headroom dropped before exact pixel sort; preserving failure without approximation')
                pixel=exact_binary(label,maps_mmap[b],memory_budget_bytes=memory_budget)
                if not pixel['valid']:raise ValueError('required full pixel metric unavailable')
                costs[domain+'_'+b+'_pixel_seconds']=time.perf_counter()-tick
                metrics.append(dict(domain=domain,branch=b,image_count=len(selected),normal_count=int((labels==0).sum()),anomaly_count=int((labels==1).sum()),
                    **metric,pixel_AUROC=pixel['AUROC'],pixel_AP=pixel['AP'],AUPRO=None,pixel_count=pixels,
                    metric_origin='new_CH1_and_exact_all_pixels_CPU',aupro_origin='pending_explicit_original_CUDA_ADEval'))
            for b,ref in COMPARISONS:
                now,old=pair_errors(labels,vectors[b]),pair_errors(labels,vectors[ref])
                rankings.append(dict(domain=domain,branch=b,reference=ref,weighted_pair_errors=float(now.sum()),reference_weighted_pair_errors=float(old.sum()),
                    delta_pair_errors=float((now-old).sum()),improved_pairs=int((now<old).sum()),worsened_pairs=int((now>old).sum()),unchanged_pairs=int((now==old).sum()),
                    tie_weight=.5,interpretation='ordering_not_fixed_threshold_false_positives_or_negatives'))
        finally:
            label._mmap.close()
            for a in maps_mmap.values():a._mmap.close()
            for p in scratch.iterdir():
                if p.name not in {'labels.bin',*(b+'.bin' for b in BRANCHES)}:raise ValueError('unexpected scratch contents; preserving')
                p.unlink()
            scratch.rmdir()
    groups=group_small_instances(instances,domains);repairs,repair_summary=margin_repairs(instances)
    metrics=append_macro(metrics)
    prediction.verified(ctx,pred)
    if ip.context(ctx['paths'])['identity']!=ctx['identity']:raise ValueError('frozen joint inputs changed during evaluation')
    tables={'metrics.csv':metrics,'metric_comparisons.csv':metric_comparisons(metrics),'per_image_scores.csv':score_rows,'ranking_summary.csv':rankings,
        'defect_instances.csv':instances,'small_instance_groups.csv':groups,'margin_repair_per_instance.csv':repairs,'margin_repair_summary.csv':repair_summary,
        'mechanism_per_image.csv':mechanism,'mechanism_summary.csv':mechanism_summary(mechanism,domains),'map_manifest.csv':map_records,
        'gate_statistics.csv':ip.csv_file(pred/'gate_statistics.csv'),
        'background_development_reference.csv':[dict(r,domain='background',role='development_reference_not_in_macro') for r in ctx['catalog']['background_reference']['rows']]}
    for name,rows in tables.items():ip.write_csv(out/name,rows)
    ip.write_json(out/'historical_score_audit.json',audit(checks))
    ip.write_csv(out/'historical_baseline_comparison.csv',historical_comparisons(ctx,samples,checks))
    from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.evaluation import peak_rss
    ip.write_json(out/'timing.json',dict(seconds=time.perf_counter()-started,stage_seconds=costs,process_peak_rss_bytes=peak_rss(),
        exact_sort_budget_bytes=memory_budget,cpu_workers=workers,new_output_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),
        environment=ip.common.environment(),model_inference=False,feature_extraction=False))
    write_report(out,metrics,rankings,mechanism_summary(mechanism,domains),repair_summary,domains)
    files={p.name:ip.sha(p) for p in out.iterdir() if p.is_file() and p.name!='status.json'}
    ip.write_json(out/'analysis_manifest.json',dict(tool=ip.TOOL,identity=ctx['identity'],domains=domains,images=len(samples),
        prediction_directory=str(pred),prediction_manifest_sha256=ip.sha(pred/'prediction_manifest.json'),files=files))
    ip.write_json(out/'status.json',dict(tool=ip.TOOL,identity=ctx['identity'],status='cpu_evaluation_complete_aupro_pending',domains=domains,
        images=len(samples),numerical_failed_checks=0,GT_used_only_after_prediction_frozen=True,all_metrics_complete=False,
        joint_domains_complete=set(domains)==set(ip.DOMAINS),analysis_manifest_sha256=ip.sha(out/'analysis_manifest.json'),
        historical_distance_reproduction='not_claimed_without_original_reference_arrays',historical_score_audit=audit(checks)))
    return out


def write_report(out,metrics,rankings,mechanisms,repairs,domains):
    lines=['# 冻结 SEGMENT / PAIR 两域验证','',f'已评价域：{", ".join(domains)}。两域方法参数在任何预测之前共同冻结。',
        '两个域未用于本轮门控参数选择；不称为完全盲测。background 仅为开发域参照，不进入验证域 macro。',
        '三分支全部重新计算 CH1 图像指标和完整像素 AUROC/AP。AUPRO 未运行的单元格保留 NA，不能称全指标完成。','',
        '| 域 | 分支 | Image AUROC | Image AP | Pixel AUROC | Pixel AP | AUPRO |','|---|---|---:|---:|---:|---:|---:|']
    for r in metrics:lines.append('| '+r['domain']+' | '+r['branch']+' | '+' | '.join('NA / 待完成' if r.get(k) is None else f'{r[k]:.9f}' for k in METRICS)+' |')
    lines+=['','差值均为分支减参照，指标单位 fraction；排序错误为正常—异常对数，并列计 0.5，不代表部署阈值误报/漏检。']
    for domain in domains:
        lookup={r['branch']:r for r in metrics if r['domain']==domain}
        lines+=['',f'## {domain}',f"SEGMENT 相对 Union 的 Image AUROC / AP 变化：{lookup[S]['image_AUROC']-lookup[U]['image_AUROC']:+.9f} / {lookup[S]['image_AP']-lookup[U]['image_AP']:+.9f}。",
            f"PAIR 相对 Union 的 Image AUROC / AP 变化：{lookup[P]['image_AUROC']-lookup[U]['image_AUROC']:+.9f} / {lookup[P]['image_AP']-lookup[U]['image_AP']:+.9f}。",
            f"PAIR 相对 SEGMENT 的 Pixel AUROC / AP 变化：{lookup[P]['pixel_AUROC']-lookup[S]['pixel_AUROC']:+.9f} / {lookup[P]['pixel_AP']-lookup[S]['pixel_AP']:+.9f}。"]
        for b in (S,P):
            signs=[lookup[b][k]-lookup[U][k] for k in ('image_AUROC','image_AP')]
            description='两项图像指标均提高' if all(v>0 for v in signs) else '两项图像指标均未提高' if all(v<=0 for v in signs) else '两项图像指标变化方向不一致'
            lines.append(f'{b}：{description}，只描述本域冻结配置。')
        for metric in ('pixel_AUROC','pixel_AP','AUPRO'):
            u,s,p=(lookup[b].get(metric) for b in BRANCHES)
            if any(v is None for v in (u,s,p)):lines.append(f'{metric}：待原后端完成，暂不判断定位损失是否缩小。');continue
            description=('PAIR 缩小 SEGMENT 相对 Union 的损失' if p>s else 'PAIR 未缩小 SEGMENT 相对 Union 的损失') if s<u else 'SEGMENT 在该指标未低于 Union，不以修复原损失表述'
            lines.append(f'{metric}：{description}；PAIR−Union={p-u:+.9f}。')
        r=next(r for r in mechanisms if r['domain']==domain and r['branch']==P and r['layer']=='MEAN4')
        lines.append(f"PAIR 原 SEGMENT 正常/缺陷降距保留量：{r['normal_all_patch_retained_amount']} / {r['anomaly_GT_patch_retained_amount']}；R_N={r['R_N']}，R_A={r['R_A']}，G={r['G']}。零分母保持 NA。")
        r=next(r for r in repairs if r['domain']==domain and r['group']=='all' and r['definition']=='margin')
        lines.append(f"按实例对齐：SEGMENT 原有损失 {r['original_segment_loss']}；完全修复 {r['old_loss_fully_repaired']}；部分修复 {r['old_loss_partially_repaired']}；PAIR 新增损失 {r['new_PAIR_loss']}；不可计算 {r['not_computable']}。")
        lines.append(f"环带恢复多于缺陷且 margin 更差：{r['ring_recovers_more_and_margin_worsens']}；环带恢复而缺陷未恢复：{r['ring_recovers_but_defect_does_not']}。详见逐实例表，不据均值推断全部区域。")
        lines.append('仍出现恢复环带却未恢复缺陷的 margin 损失。' if r['ring_recovers_but_defect_does_not'] else '在本次可计算实例及冻结容差下，未检出“环带恢复而缺陷未恢复”模式；NA 实例不作无损失解释。')
    if set(domains)==set(ip.DOMAINS):
        for b in (S,P):
            successes=[d for d in domains if all(next(r for r in metrics if r['domain']==d and r['branch']==b)[k]>
                next(r for r in metrics if r['domain']==d and r['branch']==U)[k] for k in ('image_AUROC','image_AP'))]
            lines.append(f'{b}：两个图像指标同时改善的验证域为 {successes}，共 {len(successes)}/2；是否支持跨域重复的图像收益据此描述，不能据 macro 隐去单域损失。')
    lines+=['','只有两域全部完成，才可据逐域及二域等权 macro 判断是否重复有效；单域结果不能替代另一域。',
        '是否主要适合 background，应分别对照上表的逐域收益/损失；接近 Union 不等于超过 Union，也不表示超过 SOTA。',
        '小实例为各新域原图面积最小 ceil(N/4)，包含映射消失/空环带；不与 background 三个最小实例混口径，不把标注碎片断言为独立物理缺陷。',
        '历史 CURRENT 按固定跨环境绝对容差 1e-4 核验；原 1e-5 检查结果单独保留。兼容通过不等于严格历史复现或指标完全相同。',
        '历史分数偏差见 historical_score_audit.json，CURRENT 指标与排序变化见 historical_baseline_comparison.csv。',
        '若无原始距离参照，只核验分数，不能宣称原距离逐元素复现。']
    (out/'objective_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')


def historical_comparisons(ctx,samples,checks):
    """Evaluation-only audit: current vs historical CURRENT, not CH1 metrics."""
    lookup={(r['domain'],r['sample_id'],r['check_kind']):r for r in checks
            if r['check_kind'] in ('SOURCE_CURRENT','UNION_CURRENT')}
    rows=[]
    for domain in dict.fromkeys(s['domain'] for s in samples):
        selected=[s for s in samples if s['domain']==domain]
        labels=np.asarray([s['label'] for s in selected])
        for branch in ('SOURCE_CURRENT','UNION_CURRENT'):
            records=[lookup[(domain,s['sample_id'],branch)] for s in selected]
            actual=np.asarray([r['actual_score'] for r in records],float)
            old=np.asarray([r['historical_score'] for r in records],float)
            now_metric,old_metric=image_metrics(labels,actual),image_metrics(labels,old)
            now_pairs,old_pairs=pair_errors(labels,actual),pair_errors(labels,old)
            delta=actual-old
            rows.append(dict(domain=domain,branch=branch,images=len(selected),
                max_abs_score_error=float(np.abs(delta).max()),mean_abs_score_error=float(np.abs(delta).mean()),
                mean_signed_score_error=float(delta.mean()),
                strict_failed_images=sum(not r['strict_verified'] for r in records),
                **{k:now_metric[k] for k in now_metric},
                **{'historical_'+k:old_metric[k] for k in old_metric},
                **{'delta_'+k:now_metric[k]-old_metric[k] for k in now_metric},
                changed_pairs=int((now_pairs!=old_pairs).sum()),improved_pairs=int((now_pairs<old_pairs).sum()),
                worsened_pairs=int((now_pairs>old_pairs).sum()),delta_weighted_pair_errors=float((now_pairs-old_pairs).sum()),
                direction='current_run_minus_historical',metric_units='fraction',score_rule='CURRENT_not_CH1'))
    return rows


def verified_analysis(ctx,prediction_dir,directory):
    pred,pm,_=prediction.verified(ctx,prediction_dir);root=Path(directory).resolve()
    status=ip.read_json(root/'status.json');m=ip.read_json(root/'analysis_manifest.json')
    if (status['status']!='cpu_evaluation_complete_aupro_pending' or status['numerical_failed_checks']!=0
            or status['identity']!=ctx['identity'] or status['analysis_manifest_sha256']!=ip.sha(root/'analysis_manifest.json')
            or m['prediction_manifest_sha256']!=ip.sha(pred/'prediction_manifest.json') or m['domains']!=pm['domains'] or m['images']!=pm['images']):
        raise ValueError('CPU analysis incomplete or mismatched')
    for n,h in m['files'].items():
        if ip.sha(root/ip.safe(n))!=h:raise ValueError('CPU analysis artifact modified')
    return root,m
