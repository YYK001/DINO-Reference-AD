"""Frozen full-domain CPU diagnostics, with reconstruction before conclusions."""
from __future__ import annotations
import json
import time
from collections import defaultdict,deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
from . import inputs
from .numerics import LAYERS,RESPONSES,SCORE_ATOL,reconstruct,adaptive_max,check
from .metrics import exact_binary
from .spatial import analyze_instances


def statistics(values):
    a=np.asarray(values,np.float64)
    if not a.size:return dict(count=0,mean=None,std=None,min=None,max=None,p05=None,p50=None,p95=None)
    return dict(count=int(a.size),mean=float(a.mean()),std=float(a.std()),min=float(a.min()),max=float(a.max()),
                p05=float(np.percentile(a,5)),p50=float(np.percentile(a,50)),p95=float(np.percentile(a,95)))


def correlation(a,i,j):
    if a.shape[1]<2 or np.std(a[i])==0 or np.std(a[j])==0:return None
    return float(np.corrcoef(a[i].astype(float),a[j].astype(float))[0,1])


def historical(context):
    rows={};scores={};metrics={};anchors=defaultdict(dict);small=set()
    for b,archive,name in [('U',context['parent'],'UNION_K8'),('P',context['projection'],'PROJECTION_TARGET512_SOURCE')]:
        selected=inputs.historical_rows(archive,name,'defect_instances')
        rows[b]={(r['sample_id'],int(r['instance_id'])):r for r in selected}
        if len(rows[b])!=len(selected):raise ValueError('duplicate historical instance')
        chosen=inputs.historical_rows(archive,name,'per_image_scores')
        scores[b]={r['sample_id']:r for r in chosen}
        if len(chosen)!=297 or set(scores[b])!={s['sample_id'] for s in context['samples']}:
            raise ValueError('297 historical score rows required')
        for sample in context['samples']:
            r=scores[b][sample['sample_id']]
            if int(r['label'])!=sample['label'] or r['relative_path']!=sample['relative_path']:raise ValueError('score scope mismatch')
        chosen=inputs.historical_rows(archive,name,'metrics')
        if len(chosen)!=1:raise ValueError('unique historical metric row required')
        metrics[b]=chosen[0]
    if set(rows['U'])!=set(rows['P']):raise ValueError('historical instance sets differ')
    for key,r in rows['U'].items():
        flag=r['original_smallest_three_rule'].lower()=='true'
        if flag!=(rows['P'][key]['original_smallest_three_rule'].lower()=='true'):raise ValueError('original small-instance identity changed')
        if flag:small.add(key)
    if len(small)!=3:raise ValueError('original smallest-three identities required')
    for r in context['parent']['data']['evaluation/fixed_reference_instance_peaks.csv']:
        if r['reference']=='UNION_K8' and r['branch']=='UNION_K8':
            key=(r['sample_id'],int(r['instance_id']));role=r['region']
            if role in anchors[key]:raise ValueError('duplicate fixed original peak')
            anchors[key][role]=r
    return rows,scores,metrics,anchors,small


def prepare_image(sample,context,dataset,old,memory_budget):
    history,scores,_,anchors,small=old;sid=sample['sample_id'];cache=context['cache'][sid]
    meta=cache['metadata'];grid=tuple(meta['grid']);shape=tuple(meta['output_shape'])
    maps={};z={};checks=[];score_rows=[]
    for branch in ('U','P'):
        z[branch],maps[branch],error=reconstruct(cache[branch],cache['positions'],grid,shape,context['medians'],context['scales'])
        check(checks,sid,'layer_map_mean_vs_patch_fusion_then_interpolation',error,0,branch=branch)
        check(checks,sid,'historical_fused_image_score',maps[branch]['FUSED'].max(),float(scores[branch][sid]['image_score']),branch=branch)
        for response,value in maps[branch].items():
            y,x=np.unravel_index(np.argmax(value),value.shape)
            row=dict(sample_id=sid,relative_path=sample['relative_path'],label=sample['label'],branch=branch,
                                   response=response,image_score=float(value.max()),peak_y=int(y),peak_x=int(x),
                                   score_rule='corresponding_full_map_max_not_mean_layer_max')
            if response=='FUSED':
                old_peak=(int(scores[branch][sid]['full_peak_y']),int(scores[branch][sid]['full_peak_x']))
                check(checks,sid,'historical_global_peak_response',value[old_peak],float(scores[branch][sid]['image_score']),branch=branch)
                row.update(historical_peak_y=old_peak[0],historical_peak_x=old_peak[1],peak_coordinate_matches_history=(int(y),int(x))==old_peak,
                           peak_location_note='near_tie_locations_recorded; contribution_anchors_use_original_UNION_positions')
            score_rows.append(row)
    mask=dataset.mask(sample)
    evaluation_gt=np.zeros(shape,np.uint8) if mask is None else np.asarray(Image.fromarray(mask.astype(np.uint8)*255).resize((shape[1],shape[0]),Image.Resampling.NEAREST))>0
    instances,contributions,movement,spatial_checks=analyze_instances(sample,maps,mask,grid,history,anchors,small)
    checks.extend(spatial_checks);pixel_rows=[]
    if sample['label']:
        for branch in ('U','P'):
            for response,value in maps[branch].items():
                metric=exact_binary(evaluation_gt.reshape(-1),value.reshape(-1),memory_budget_bytes=memory_budget)
                pixel_rows.append(dict(sample_id=sid,branch=branch,response=response,pixel_AUROC=metric['AUROC'],pixel_AP=metric['AP'],
                                       valid=metric['valid'],AUROC_valid=metric['AUROC_valid'],AP_valid=metric['AP_valid'],
                                       status=metric['status'],positive_pixels=metric['positive_pixels'],negative_pixels=metric['negative_pixels']))
    patch=np.zeros(grid,bool) if mask is None else adaptive_max(mask,grid)
    regions={'normal_all_patches':np.ones(grid,bool)} if mask is None else {'anomaly_nondefect_patches':~patch,'GT_defect_patches':patch}
    distributions=[];correlations=[];selected={}
    physical={b:np.empty_like(cache[b]) for b in ('U','P')}
    for b in ('U','P'):physical[b][:,cache['positions']]=cache[b]
    delta=physical['P'].astype(float)-physical['U'].astype(float)
    for region,chosen in regions.items():
        for b in ('U','P'):
            selected[b,region]=dict(z=z[b][:,chosen].copy(),raw=physical[b].reshape(4,*grid)[:,chosen].copy())
            values=selected[b,region]['z'];raw=selected[b,region]['raw']
            for j,l in enumerate(LAYERS):
                distributions.append(dict(sample_id=sid,label=sample['label'],branch=b,layer=l,region=region,
                    aggregation='within_image_patches',**statistics(values[j]),raw_distance_mean=float(raw[j].mean()) if raw[j].size else None,
                    P_minus_U_distance_mean=float(delta[j,chosen.reshape(-1)].mean()) if chosen.any() else None,
                    P_minus_U_Source_MAD_mean=float((delta[j,chosen.reshape(-1)]/context['scales'][j]).mean()) if chosen.any() else None,
                    raw_unique_patch_values=len(np.unique(raw[j])),MAD_unique_patch_values=len(np.unique(values[j])),
                    note='fixed_positive_affine_MAD_not_an_independent_method_gain'))
            for i,l in enumerate(LAYERS):
                for j in range(i+1,4):
                    c=correlation(values,i,j)
                    correlations.append(dict(sample_id=sid,branch=b,region=region,layer_a=l,layer_b=LAYERS[j],
                        correlation=c,patch_count=values.shape[1],status='available' if c is not None else 'constant_or_insufficient_region'))
    return dict(sample=sample,maps=maps,gt=np.asarray(evaluation_gt,np.uint8),scores=score_rows,
        instances=instances,contributions=contributions,movement=movement,checks=checks,
        pixel_rows=pixel_rows,distributions=distributions,correlations=correlations,selected=selected)


def bounded_images(samples,fn,workers):
    """Bound retained maps by workers; executor.map eagerly queues all 297 images."""
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending=deque();iterator=iter(samples)
        for _ in range(workers):
            try:pending.append(executor.submit(fn,next(iterator)))
            except StopIteration:break
        while pending:
            yield pending.popleft().result()
            try:pending.append(executor.submit(fn,next(iterator)))
            except StopIteration:pass


def ranking_tables(scores,historical_scores,checks):
    result=[];detail=[];by={(b,r):{} for b in ('U','P') for r in RESPONSES}
    for row in scores:by[row['branch'],row['response']][row['sample_id']]=row
    ids=sorted(by['U','FUSED']);labels=np.array([by['U','FUSED'][i]['label'] for i in ids]);normal=labels==0;anomaly=labels==1
    if len(scores)!=2970 or len(ids)!=297 or (int(normal.sum()),int(anomaly.sum()))!=(85,212):raise ValueError('complete two by five score scope required')
    fixed={i for i,_ in inputs.replay.CASES[:10]};normals=[i for i in ids if by['U','FUSED'][i]['label']==0]
    if not fixed.issubset(normals):raise ValueError('fixed ten normal cases missing')
    def errors(v):return (v[normal,None]>v[None,anomaly]).astype(float)+.5*(v[normal,None]==v[None,anomaly])
    for response in RESPONSES:
        values={b:np.array([by[b,response][i]['image_score'] for i in ids]) for b in ('U','P')}
        before,after=errors(values['U']),errors(values['P']);delta=after-before
        if response=='FUSED':
            for b in ('U','P'):
                frozen=errors(np.array([float(historical_scores[b][i]['image_score']) for i in ids]))
                check(checks,'background','historical_all_normal_anomaly_pair_states',errors(values[b]),frozen,atol=0,branch=b)
        for name,chosen in [('all85',np.ones(85,bool)),('fixed10',np.array([i in fixed for i in normals])),('other75',np.array([i not in fixed for i in normals]))]:
            result.append(dict(response=response,branch='P',reference='U',normal_group=name,normal_count=int(chosen.sum()),anomaly_count=212,
                weighted_pair_errors=float(after[chosen].sum()),reference_weighted_pair_errors=float(before[chosen].sum()),
                improved_pairs=int((delta[chosen]<0).sum()),worsened_pairs=int((delta[chosen]>0).sum()),unchanged_pairs=int((delta[chosen]==0).sum()),
                pair_count=int(delta[chosen].size),tie_error_weight=.5,note='ordering_pairs_not_threshold_false_alarms'))
        for j,sid in enumerate(ids):
            region=delta[int(normal[:j].sum())] if normal[j] else delta[:,int(anomaly[:j].sum())]
            detail.append(dict(sample_id=sid,label=int(labels[j]),response=response,branch='P',reference='U',
                score_change=float(values['P'][j]-values['U'][j]),weighted_pair_error_change=float(region.sum()),
                improved_pairs=int((region<0).sum()),worsened_pairs=int((region>0).sum()),fixed_difficult_normal=sid in fixed))
    return result,detail


def summarize_responses(rows,pooled):
    result=[];correlations=[];signed=[]
    for (branch,region),values in pooled.items():
        z=np.concatenate([v['z'] for v in values],axis=1);raw=np.concatenate([v['raw'] for v in values],axis=1)
        for j,l in enumerate(LAYERS):
            group=[r for r in rows if r['branch']==branch and r['layer']==l and r['region']==region and r['count']]
            stats=statistics(z[j]);result.append(dict(branch=branch,layer=l,region=region,aggregation='pooled_patch',image_count=len(group),
                **stats,raw_distance_mean=float(raw[j].mean()) if raw[j].size else None))
            means={key:float(np.mean([r[key] for r in group])) if group else None for key in ('mean','std','p05','p50','p95','raw_distance_mean','P_minus_U_distance_mean','P_minus_U_Source_MAD_mean')}
            result.append(dict(branch=branch,layer=l,region=region,aggregation='image_equal',image_count=len(group),
                **{'mean_of_within_image_'+k:v for k,v in means.items()},note='mean_image_quantile_is_not_pooled_quantile'))
        for i,l in enumerate(LAYERS):
            for j in range(i+1,4):correlations.append(dict(branch=branch,region=region,layer_a=l,layer_b=LAYERS[j],aggregation='pooled_patch',
                                                       correlation=correlation(z,i,j),patch_count=z.shape[1]))
    for l in LAYERS:
        n=[r['P_minus_U_Source_MAD_mean'] for r in rows if r['branch']=='P' and r['layer']==l and r['region']=='normal_all_patches' and r['count']]
        a=[r['P_minus_U_Source_MAD_mean'] for r in rows if r['branch']=='P' and r['layer']==l and r['region']=='GT_defect_patches' and r['count']]
        if (len(n),len(a))!=(85,212):raise ValueError('full original patch R_N/R_A evidence missing')
        rn,ra=-float(np.mean(n)),-float(np.mean(a))
        signed.append(dict(layer=l,branch='P',reference='U',R_N=rn,R_A=ra,G=rn-ra,normal_images=85,anomaly_images=212,
                           aggregation='image_equal',scale='original_Source_MAD_no_epsilon',direction='U_minus_P'))
    signed.append(dict(layer='FUSED',branch='P',reference='U',**{k:float(np.mean([r[k] for r in signed])) for k in ('R_N','R_A','G')},
                       normal_images=85,anomaly_images=212,aggregation='equal_four_layers_image_equal',scale='original_Source_MAD_no_epsilon',direction='U_minus_P'))
    return result,correlations,signed


def layer_margin_summary(instances,contributions,movement):
    result=[]
    for response in RESPONSES:
        rows=[r for r in instances if r['branch']=='P' and r['response']==response];valid=[r for r in rows if r['margin_change_vs_U'] is not None]
        for group,chosen in [('all',valid),('original_smallest_three',[r for r in valid if r['original_smallest_three_rule']])]:
            delta=np.array([r['margin_change_vs_U'] for r in chosen],float)
            result.append(dict(response=response,group=group,total_instances=len(rows),valid_instances=len(chosen),
                negative_beyond_numerical_atol=int((delta < -SCORE_ATOL).sum()),positive_beyond_numerical_atol=int((delta > SCORE_ATOL).sum()),
                numerical_atol=SCORE_ATOL,mean_delta=float(delta.mean()) if delta.size else None,
                min_delta=float(delta.min()) if delta.size else None,max_delta=float(delta.max()) if delta.size else None,
                note='numerical_tolerance_only_not_a_detector_threshold'))
    return result


def complete(state,tables):
    return (state['analyzed_images']==297 and state['exact_pixel_responses']==10 and state['reconstruction_passed']
            and state['failed_checks']==0 and len(tables['layer_metrics'])==10 and len(tables['per_image_scores'])==2970
            and len(tables['ranking_summary'])==15 and len(tables['within_image_pixel_metrics'])==2120
            and bool(tables['defect_instances']) and bool(tables['fixed_position_layer_contributions'])
            and state['protected_inputs_unchanged'])


def analyze(context,args):
    inputs.guard(context['paths'])
    root=context['paths']['output'];root.mkdir(parents=True,exist_ok=True)
    records=[dict(kind=k,path=str(context['paths'][k]),sha256=v) for k,v in context['checksums'].items()]
    for f in sorted(Path(__file__).parent.glob('*.py')):records.append(dict(kind='code',path=str(f),sha256=inputs.sha(f)))
    project=Path(__file__).resolve().parents[3]
    for relative in ('DINOv3/MADEqual/aebad_s_union_diagnostics/spatial.py','DINOv3/relation_oracle_cache/extract.py',
                     'DINOv3/relation_reliability/superadd_host.py','DINOv3/MADEqual/aebad_s_full_p0/runner.py'):
        file=project/relative
        records.append(dict(kind='reused_definition_or_numerical_source',path=str(file),sha256=inputs.sha(file)))
    frozen=dict(tool=inputs.TOOL,inputs=records,layers=list(LAYERS),branches=['U','P'],scope=297,
        tolerances=dict(score_atol=SCORE_ATOL,distance_atol=1e-7,distance_rtol=1e-6),
        interpolation='NumPy_float32_half_pixel_bilinear_align_corners_false',GT_patch='adaptive_max',
        pixel_metrics='exact_all_pixels_cpu_no_AUPRO',memory_budget_gib=args.memory_budget_gib,cpu_workers=args.cpu_workers)
    with (root/'inputs.json').open('x',encoding='utf8') as file:
        file.write(json.dumps(dict(identity=inputs.digest(frozen),frozen=frozen),ensure_ascii=False,indent=2))
    tables=defaultdict(list);timing={};started=time.perf_counter();pooled=defaultdict(list);scratch=root/'scratch';scratch.mkdir()
    count=context['pixel_count'];scoremaps={};maskmap=None;dataset=None
    state=dict(tool=inputs.TOOL,status='partial_evidence',analyzed_images=0,exact_pixel_responses=0,reconstruction_passed=False,
        failed_checks=0,protected_inputs_unchanged=False,model_loaded=False,feature_extraction=False,torch_imported=False,
        retrieval=False,OT=False,full_domain_analysis_executed=True)
    try:
        old=historical(context);history,old_scores,old_metrics,_,_=old
        maskmap=np.memmap(scratch/'gt.dat',dtype=np.uint8,mode='w+',shape=(count,))
        for b in ('U','P'):
            for r in RESPONSES:scoremaps[b,r]=np.memmap(scratch/f'{b}_{r}.dat',dtype=np.float32,mode='w+',shape=(count,))
        dataset=inputs.DatasetZip(context['paths']['data']);budget=int(args.memory_budget_gib*2**30);offset=0;t=time.perf_counter()
        fn=lambda s:prepare_image(s,context,dataset,old,budget)
        for result in bounded_images(context['samples'],fn,args.cpu_workers):
            sample=result['sample'];n=result['gt'].size;maskmap[offset:offset+n]=result['gt'].reshape(-1)
            for b in ('U','P'):
                for r in RESPONSES:scoremaps[b,r][offset:offset+n]=result['maps'][b][r].reshape(-1)
            offset+=n
            for field,target in [('scores','per_image_scores'),('instances','defect_instances'),('contributions','fixed_position_layer_contributions'),
                                  ('movement','instance_peak_movement'),('checks','numerical_verification'),('pixel_rows','within_image_pixel_metrics'),
                                  ('distributions','layer_response_per_image'),('correlations','layer_correlations_per_image')]:tables[target].extend(result[field])
            for key,value in result['selected'].items():pooled[key].append(value)
            state['analyzed_images']+=1
            print(f"[Layer diagnostic] {state['analyzed_images']}/297 {sample['sample_id']}",flush=True)
        timing['reconstruction_instances_within_image_metrics']=time.perf_counter()-t
        dataset.close();dataset=None
        if offset!=count:raise ValueError('full pixel count mismatch')
        tables['ranking_summary'],tables['ranking_per_image']=ranking_tables(tables['per_image_scores'],old_scores,tables['numerical_verification'])
        if any(not r['verified'] for r in tables['numerical_verification']):raise ValueError('reconstruction/order/spatial checks failed; no definite attribution')
        state['reconstruction_passed']=True
        tables['layer_response_summary'],tables['layer_correlation_summary'],tables['signed_distance_summary']=summarize_responses(tables['layer_response_per_image'],pooled)
        del pooled
        for branch in ('U','P'):
            for region in ('normal_all_patches','anomaly_nondefect_patches','GT_defect_patches'):
                for i,l in enumerate(LAYERS):
                    for other in LAYERS[i+1:]:
                        vals=[r['correlation'] for r in tables['layer_correlations_per_image'] if r['branch']==branch and r['region']==region and r['layer_a']==l and r['layer_b']==other and r['correlation'] is not None]
                        tables['layer_correlation_summary'].append(dict(branch=branch,region=region,layer_a=l,layer_b=other,
                            aggregation='image_equal',valid_images=len(vals),correlation=float(np.mean(vals)) if vals else None))
        tables['instance_margin_summary']=layer_margin_summary(tables['defect_instances'],tables['fixed_position_layer_contributions'],tables['instance_peak_movement'])
        pixel_by={(r['sample_id'],r['branch'],r['response']):r for r in tables['within_image_pixel_metrics']}
        for sample in context['samples']:
            if not sample['label']:continue
            for response in RESPONSES:
                u,p=(pixel_by[sample['sample_id'],b,response] for b in ('U','P'))
                tables['within_image_pixel_differences'].append(dict(sample_id=sample['sample_id'],response=response,direction='P_minus_U',
                    delta_pixel_AUROC=p['pixel_AUROC']-u['pixel_AUROC'] if p['AUROC_valid'] and u['AUROC_valid'] else None,
                    delta_pixel_AP=p['pixel_AP']-u['pixel_AP'] if p['AP_valid'] and u['AP_valid'] else None,
                    U_status=u['status'],P_status=p['status']))
        tables['fixed_case_scores']=[r for r in tables['per_image_scores'] if r['sample_id'] in {i for i,_ in inputs.replay.CASES}]
        for values in scoremaps.values():values.flush();values._mmap.close()
        scoremaps.clear();maskmap.flush();maskmap._mmap.close();maskmap=None
        y=np.memmap(scratch/'gt.dat',dtype=np.uint8,mode='r',shape=(count,));t=time.perf_counter()
        labels=np.array([s['label'] for s in context['samples']],np.uint8)
        for branch in ('U','P'):
            for response in RESPONSES:
                file=scratch/f'{branch}_{response}.dat';value=np.memmap(file,dtype=np.float32,mode='r',shape=(count,))
                pixel=exact_binary(y,value,memory_budget_bytes=budget)
                image_rows=[r for r in tables['per_image_scores'] if r['branch']==branch and r['response']==response]
                image=exact_binary(labels,np.array([r['image_score'] for r in image_rows]),memory_budget_bytes=budget)
                per_image=[r for r in tables['within_image_pixel_metrics'] if r['branch']==branch and r['response']==response and r['valid']]
                ap_images=[r for r in tables['within_image_pixel_metrics'] if r['branch']==branch and r['response']==response and r['AP_valid']]
                if not pixel['valid'] or not image['valid']:raise ValueError('formal pixel/image metrics not computable')
                row=dict(branch=branch,response=response,image_count=297,image_AUROC=image['AUROC'],image_AP=image['AP'],
                    pixel_AUROC=pixel['AUROC'],pixel_AP=pixel['AP'],evaluation_pixels=count,
                    anomaly_within_image_valid_count=len(per_image),anomaly_images=212,
                    anomaly_within_image_AUROC_valid_count=len(per_image),anomaly_within_image_AP_valid_count=len(ap_images),
                    mean_within_anomaly_pixel_AUROC=float(np.mean([r['pixel_AUROC'] for r in per_image])) if per_image else None,
                    mean_within_anomaly_pixel_AP=float(np.mean([r['pixel_AP'] for r in ap_images])) if ap_images else None,
                    metric_backend='CPU_exact_sort_complete_pixels_no_histogram_no_sampling',AUPRO='not_requested_not_computed')
                tables['layer_metrics'].append(row);state['exact_pixel_responses']+=1;value._mmap.close();del value
                if response=='FUSED':
                    for key in ('image_AUROC','image_AP','pixel_AUROC','pixel_AP'):
                        check(tables['numerical_verification'],'background','historical_fused_metric',row[key],float(old_metrics[branch][key]),branch=branch,metric=key)
                print(f'[Exact CPU metrics] {branch} {response}',flush=True)
        y._mmap.close();del y;timing['ten_exact_full_pixel_metrics']=time.perf_counter()-t
        by={(r['branch'],r['response']):r for r in tables['layer_metrics']}
        for response in RESPONSES:
            u,p=by['U',response],by['P',response]
            tables['layer_metric_differences'].append(dict(response=response,branch='P',reference='U',direction='P_minus_U',unit='metric_fraction',
                **{'delta_'+key:p[key]-u[key] for key in ('image_AUROC','image_AP','pixel_AUROC','pixel_AP')}))
        for record in records:
            if inputs.sha(record['path'])!=record['sha256']:raise ValueError('protected local input/code changed during analysis')
        state.update(protected_inputs_unchanged=True,failed_checks=sum(not r['verified'] for r in tables['numerical_verification']))
        if complete(state,tables):
            state['status']='analysis_complete'
            # Only remove explicitly generated scratch files within this new output.
            for file in scratch.iterdir():
                if file.resolve().parent!=scratch.resolve() or not file.is_file():raise ValueError('unexpected scratch contents')
                file.unlink()
            scratch.rmdir()
    except Exception as error:state.update(status='partial_evidence',error=f'{type(error).__name__}: {error}',failed_checks=sum(not r['verified'] for r in tables['numerical_verification']))
    finally:
        if dataset is not None:dataset.close()
        for value in scoremaps.values():value.flush();value._mmap.close()
        if maskmap is not None:maskmap.flush();maskmap._mmap.close()
    for name,rows in tables.items():inputs.replay.write_csv(root/f'{name}.csv',rows)
    state.update(seconds=time.perf_counter()-started,timing=timing)
    try:
        import psutil
        state['observed_current_rss_bytes']=psutil.Process().memory_info().rss
        state['process_peak_rss_bytes']=getattr(psutil.Process().memory_info(),'peak_wset',None)
    except ImportError:pass
    if state.get('process_peak_rss_bytes') is None:
        try:
            import resource,sys
            state['process_peak_rss_bytes']=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024)
        except ImportError:pass
    write_report(root,state,tables)
    (root/'status.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf8')
    return state


def write_report(root,state,tables):
    lines=['# AeBAD-S/background：初始投影逐层诊断','',f"状态：{state['status']}",'',
        'U=完整K8 UNION；P=投影与Source联合。固定Source MAD，四层等权融合；仅诊断，不选层/权重/阈值。',
        'Pixel AUROC/AP 使用全部评价像素的CPU精确排序；不计算AUPRO。',
        '单层MAD为固定正仿射变换，不把raw与MAD算作两个方法；保留float32舍入的并列差异。','']
    if state['status']!='analysis_complete':
        lines+=['输入/重建/资源/全域评价未全部通过，不输出确定性归因。',f"原因：{state.get('error','incomplete evidence')}"]
    else:
        lines+=['| 响应 | ΔImage AUROC | ΔImage AP | ΔPixel AUROC | ΔPixel AP |','|---|---:|---:|---:|---:|']
        for r in tables['layer_metric_differences']:
            lines.append('| '+str(r['response'])+' | '+' | '.join(f"{r['delta_'+k]:+.9f}" for k in ('image_AUROC','image_AP','pixel_AUROC','pixel_AP'))+' |')
        lines+=['','差值P−U，单位为0–1指标差；不是百分点。','',
            '## 1. 哪些层、哪些实例出现区分损失？','']
        loss=[str(r['response']) for r in tables['layer_metric_differences'] if r['response']!='FUSED' and (r['delta_pixel_AUROC']<0 or r['delta_pixel_AP']<0)]
        lines+=['至少一项全像素指标下降的单层：'+(', '.join(loss) if loss else '无')+'。这是差值描述，不是独立统计显著性检验。',
            '全部实例及映射后消失/空环带保留在defect_instances.csv；反复下降、小实例情况见instance_margin_summary.csv；不挑选有利案例。','',
            '以下按响应列出margin下降最大的最多10个实例，仅便于定位；全体实例均保留：']
        for response in RESPONSES:
            bad=sorted([r for r in tables['defect_instances'] if r['branch']=='P' and r['response']==response and r['margin_change_vs_U'] is not None
                and r['margin_change_vs_U'] < -SCORE_ATOL],key=lambda r:(r['margin_change_vs_U'],r['sample_id'],r['instance_id']))
            lines.append(str(response)+'：下降实例'+str(len(bad))+'；'+('; '.join(f"{r['sample_id']}#{r['instance_id']} Δ={r['margin_change_vs_U']:+.6g}" for r in bad[:10]) if bad else '无超出数值容差的下降'))
        small=[r for r in tables['defect_instances'] if r['branch']=='P' and r['response']=='FUSED' and r['original_smallest_three_rule']]
        for r in small:lines.append(f"原最小GT实例 {r['sample_id']}#{r['instance_id']}：面积{r['area_original_pixels']}，patch覆盖{r['patch_coverage']}，状态{r['response_status']}，Δmargin={r['margin_change_vs_U']}。")
        lines+=['','## 2. 固定位置变化和峰值移动各起什么作用？','']
        available=[r for r in tables['instance_peak_movement'] if r['response_status']=='available']
        for key in ('fixed_position_change','e_in','e_ring','delta_margin','identity_error'):
            vals=[r[key] for r in available]
            lines.append(f"{key}：可计算实例{len(vals)}，均值{np.mean(vals):+.9g}，最大绝对值{max(abs(v) for v in vals):.9g}。" if vals else f'{key}：无可计算实例，不填造统计。')
        for l in LAYERS:
            vals=[r['c_l'] for r in tables['fixed_position_layer_contributions'] if r['layer']==l and r['c_l'] is not None]
            lines.append(f"L{l}：固定位置贡献c_l均值{np.mean(vals):+.9g}，超出数值容差的负贡献实例{sum(v < -SCORE_ATOL for v in vals)}/{len(vals)}。" if vals else f'L{l}：无可计算贡献，保留不可计算状态。')
        lines+=['','## 3. 当前证据支持优先改投影还是研究融合？','']
        repeated={r['response'] for r in tables['instance_margin_summary'] if r['group']=='all' and r['response']!='FUSED'
                  and r['mean_delta'] is not None and r['mean_delta'] < -SCORE_ATOL and r['negative_beyond_numerical_atol']>r['positive_beyond_numerical_atol']}
        before_fusion={r['response'] for r in tables['layer_metric_differences'] if r['response']!='FUSED' and r['delta_pixel_AUROC'] < -SCORE_ATOL and r['delta_pixel_AP'] < -SCORE_ATOL}
        supported=sorted(repeated&before_fusion)
        if supported:lines.append('以下层同时出现两项全像素指标下降及实例margin反复下降：'+', '.join(map(str,supported))+'。存在融合前损失证据，优先研究投影约束有依据；这不排除尺度/融合和峰值移动共同参与。')
        else:lines.append('尚未形成“两项单层全像素指标下降且实例margin反复下降”的一致证据，不能据此唯一排序两条研究方向。结合逐图指标、层贡献、尺度和相关性继续判断；证据不足也是允许结果。')
        lines+=['这是一项透明的描述性证据汇总，不是因果检验或调参规则；没有输出最佳层或融合权重。','',
            '## 4. 哪些结论仍然不能成立？','',
            '不能凭某单层优于融合证明MAD是唯一原因；不能将平均距离下降等同于区分能力下降。',
            '不能将错误排序对称为部署阈值误报/漏检；不能把不同层自身峰值当成同位置层贡献。',
            '不能外推所有采样、投影或数据域。没有GT驱动的新方法选择，没有Torch/CUDA链路验证。']
    (root/'diagnostic_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
