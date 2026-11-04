"""Fixed background/seed42 phase in the existing tool; no automatic domain expansion."""
import json
import os
import sys
import time
import subprocess
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from . import inputs as ip,runtime as rt,pipeline,numerics,reporting,gpu_lp,lp_v2,mixed_runtime as mr
from .protocol import CONFIG,BRANCHES,PRIMARY,BASE,SHARED,OWN
st=ip.st
PLAN=dict(domain='background',support_seed=42,images=289,normal=77,anomaly=212,Source_calibration=66,
    Target_templates=8,evaluation_LPs=2312,calibration_LPs=528,LPs=2840,unique_region_images=363,
    branches=list(BRANCHES),new_pixel_jobs=0,new_AUPRO_jobs=0,training=False,Source_rebuild=False,
    not_requested=['seed43','illumination','view'])


def configure(a):
    if a.resume_from is None:raise ValueError('--resume-from must identify the existing coverage run')
    old=st.sealed_json(a.resume_from/'inputs.json');cache=Path(old['routes']['cache_root']).resolve()
    if a.device!=old['backend']['distance_device']:
        if a.stage=='preflight':a.device=old['backend']['distance_device']
        else:raise ValueError('distance backend must match original coverage; select the agreed GPU explicitly')
    for protected in (a.resume_from,cache,*a.region_caches,*a.distance_exports,*a.legacy_trials):
        if any(w==protected or w.is_relative_to(protected) or protected.is_relative_to(w) for w in (a.run_root,a.cache_root)):
            raise ValueError('new run/cache must be separate from read-only parents and exports')
    if a.rms_stat is None:a.rms_stat=cache/'source_RMS.json'
    a.region_caches=list(dict.fromkeys([cache,*a.region_caches]));a._capacity_parent=old
    return a


def freeze(a,commit=True):
    x=ip.assets(a);samples=x.samples('background');supports=x.supports('background',42)
    if (len(samples)!=289 or sum(int(r['label'])==0 for r in samples)!=77 or len(x.calibration)!=66 or len(supports)!=8
        or {r['relative_path'] for r in samples}&{r['relative_path'] for r in supports}):
        raise ValueError('background common 289 (77/212), Source66 and original K8 required')
    # The original exclusion population is validated by the two-K8 parent reader.
    versions=None
    if a.lp_python.is_file():
        p=subprocess.run([str(a.lp_python),'-B','-m',__package__+'.lp_worker','--versions'],capture_output=True,text=True,check=True)
        versions=json.loads(p.stdout)
    value=dict(protocol=lp_v2.PROTOCOL,plan=PLAN,asset_identity=x.identity,parent=a._capacity_parent['seal'],
        method={k:v for k,v in CONFIG.items() if k not in ('domains','support_seeds','solver','solver_primal_tolerance',
            'solver_dual_tolerance','row_relative_tolerance','capacity_relative_tolerance','objective_tolerance')},
        images=[dict(sample_id=r['sample_id'],**ip.image(r)) for r in samples],calibration=[ip.image(r) for r in x.calibration],
        supports=[ip.image(r) for r in supports],distance_backend=a.device,physical_device=os.environ.get('CUDA_VISIBLE_DEVICES'),solver_settings=lp_v2.settings(max(1,a.cpu_workers-1)),
        lp_python=str(a.lp_python),solver_environment=versions,cache_root=str(a.cache_root),RMS=str(a.rms_stat),RMS_seal=(ip.rms_record(a) or {}).get('seal'),region_caches=[str(p) for p in a.region_caches],
        distance_exports=[str(p) for p in a.distance_exports],code={p.name:st.sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))})
    value['identity']=st.digest(value)
    if commit:r=st.sealed_json(a.run_root/'inputs.json',value)
    else:
        r=dict(value,seal=st.digest(value));st.write_json(a.run_root/'preflight_inputs.json',r)
    st.sealed_json(a.run_root/'LP_protocol.json',lp_v2.PROTOCOL)
    st.write_json(a.run_root/'job_plan.json',PLAN);return r


def old_audit(a):
    rows=[];logs=[]
    for root in a.legacy_trials:
        for path in sorted((root/'trial_results').glob('repeat0_batch*.json')):
            r=st.sealed_json(path);solver=r.get('solver') or {}
            for c in r.get('checks',[]):
                f=c.get('feasibility',{});scale=max(1.,abs(c.get('objective',0.)))
                numeric=(f.get('row_sum_max_error',float('inf'))<=1e-5 and f.get('capacity_max_relative_error',float('inf'))<=1e-5
                    and f.get('objective_consistency_error',float('inf'))<=1e-5*scale
                    and c.get('reference_abs_error',float('inf'))<=1e-5*max(1.,abs(c.get('reference_objective',0.))))
                rows.append(dict(source=str(path),problem_id=c['problem_id'],stop_reason=solver.get('status'),
                    partial_ABCD_pass=bool(numeric),acceptance='incomplete_missing_primal_and_per_LP_dual_evidence',
                    objective=c.get('objective'),reference_objective=c.get('reference_objective'),
                    solver_batch_primal_residual=(solver.get('convergence') or {}).get('primal_residual'),
                    solver_batch_dual_residual=(solver.get('convergence') or {}).get('dual_residual'),
                    solver_batch_gap=(solver.get('convergence') or {}).get('gap'),**f))
    st.write_csv(a.run_root/'legacy_LP_reaudit.csv',rows)
    for root in a.legacy_trials:
        for p in [*root.glob('*.log'),*(root/'logs').glob('*.log')]:
            dest=a.run_root/'legacy_logs'/(root.name+'_'+p.relative_to(root).as_posix().replace('/','_'))
            if not dest.exists():
                text=p.read_text(encoding='utf-8',errors='replace');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(text,encoding='utf-8')
            logs.append(dict(path=str(p),read_complete=True,copy=str(dest)))
    st.write_json(a.run_root/'legacy_log_inventory.json',dict(logs=logs,no_available_log_files=not logs,
        evidence='older trial retained summaries, not primal/dual vectors; global solver residual is not a per-LP certificate'))
    return rows


def inventory(a):
    x=ip.assets(a);rms=ip.rms_record(a);rows=pipeline.region_rows(a);items=[];reusable=0;strict=0;missing_distances=0
    exported=mr.exports(a)
    for row in rows:
        existing=rt.cached_region_path(a,row,rms) if rms else None
        feature=None if existing else ip.region_feature_ref(a,row)
        items.append(dict(image=row['relative_path'],region_available=bool(existing),region_path=str(existing or ''),
            complete_feature_available=bool(feature),action='reuse_region' if existing else 'partition_existing_features' if feature else 'explicit_missing_feature_forward'))
    for _,_,_,row in pipeline.observations(a):
        for t in x.supports('background',42):
            path=rt.pair_path(a,row,t)
            if path.is_file():
                record=st.sealed_json(path)
                if record['protocol']!=lp_v2.PROTOCOL:raise ValueError('incompatible completed LP')
                reusable+=1
            else:
                if ((ip.key(row),ip.key(t)) not in exported and
                    not (a.cache_root/'distances'/ip.key(row)/ip.key(t)/'manifest.json').exists()):missing_distances+=1
                old=a.resume_from/'pairs'/ip.key(row)/(ip.key(t)+'.json')
                if old.exists() and st.sealed_json(old)['result']['success']:strict+=1
    result=dict(plan=PLAN,regions_present=sum(r['region_available'] for r in items),remaining_regions=sum(not r['region_available'] for r in items),
        missing_feature_images=sum(not r['region_available'] and not r['complete_feature_available'] for r in items),
        committed_v2_LPs=reusable,strict_CPU_candidates=strict,remaining_LPs_upper_bound=2840-reusable,
        missing_distance_matrices=missing_distances,
        RMS_available=bool(rms),solver_environment_available=a.lp_python.is_file())
    st.write_csv(a.run_root/'asset_inventory.csv',items);st.write_json(a.run_root/'inventory.json',result)
    missing=[dict(role='feature_to_generate',image=r['image'],action=r['action']) for r in items if not r['region_available'] and not r['complete_feature_available']]
    if not rms:missing.append(dict(role='existing_Source_RMS',path=str(a.rms_stat),action='locate existing RMS; no automatic rebuild'))
    if not a.lp_python.is_file():missing.append(dict(role='isolated_solver_environment',path=str(a.lp_python),action='locate existing yyk_lp'))
    st.write_csv(a.run_root/'missing_inputs.csv',missing)
    return result


def smoke(a):
    frozen=freeze(a)
    if (a.run_root/'smoke.json').exists():
        saved=st.sealed_json(a.run_root/'smoke.json')
        if saved['identity']!=frozen['identity']:raise ValueError('review identity differs')
        return dict(status='v2_smoke_complete',reused=True,estimate=estimate(a,saved))
    if not a.distance_exports:raise ValueError('existing 16-LP --distance-export required')
    old_audit(a);rms=rt.source_rms(a)
    manifest,problems=gpu_lp.load_problems(SimpleNamespace(problem_root=a.distance_exports[0],output_root=a.run_root))
    if manifest['parent_inputs']!=a._capacity_parent['seal']:raise ValueError('16-LP export belongs to another coverage parent')
    refs=[];x=ip.assets(a);templates=x.supports('background',42);allowed={ip.key(r) for r in [*x.calibration,*x.samples('background')]}
    for ref,problem in zip(manifest['problems'],problems):
        path=a.distance_exports[0]/'problems'/ref['id'];meta=st.sealed_json(path/'manifest.json')['metadata']
        if ip.key(meta['query']) not in allowed or ip.key(meta['template']) not in {ip.key(t) for t in templates}:raise ValueError('16-LP cohort outside fixed scope')
        old=meta['dependencies']
        if old['RMS']!=rms['seal'] or old['distance_device']!=a.device:raise ValueError('review RMS/backend differs')
        original=st.sealed_json(a.resume_from/'pairs'/ip.key(meta['query'])/(ip.key(meta['template'])+'.json'))
        if original['seal']!=meta['original_pair_seal'] or not original['result']['success'] or original['result']['objective']!=ref['reference_objective']:
            raise ValueError('16-LP CPU reference differs from original committed solution')
        dep=dict(query=old['query'],template=old['template'],RMS=rms['seal'],distance_device=a.device,
            definition=CONFIG['distance'],implementation=st.sha(Path(regions_file())))
        refs.append(dict(id=ref['id'],bundle=str(path),bundle_seal=ref['seal'],reference_objective=ref['reference_objective'],
            dependencies=dep,query=meta['query'],template=meta['template']))
    review=[];timings=[]
    with mr.Worker(a) as worker:
        for start in range(0,16,8):
            group=refs[start:start+8];todo=[]
            for item in group:
                saved=a.run_root/'review16'/(item['id']+'.json')
                if saved.exists():
                    r=st.sealed_json(saved);check,proof=mr.verify_proof(item,r['proof'])
                    if r['proof_seal']!=proof['seal'] or not check['accepted']:raise ValueError('retained review evidence changed')
                    review.append(r)
                else:todo.append(item|dict(proof=str(a.cache_root/'review_proofs'/item['id']/str(time.time_ns()))))
            if not todo:continue
            response,elapsed=worker.solve(todo);timings.append(dict(seconds=elapsed,pairs=len(todo)))
            for item,proof in zip(todo,response['records']):
                check,record=mr.verify_proof(item,proof['proof'])
                r=dict(id=item['id'],proof=proof['proof'],proof_seal=record['seal'],audit=check,batch_job=response['job'],
                    bundle=item['bundle'],bundle_seal=item['bundle_seal'],CPU_reference=item['reference_objective'],
                    solver=record['metadata']['solver'],end_to_end_batch_seconds=elapsed,batch_pairs=len(todo))
                st.sealed_json(a.run_root/'review_attempts'/(item['id']+'_'+Path(proof['proof']).name+'.json'),r)
                if check['accepted']:
                    committed=st.sealed_json(a.run_root/'review16'/(item['id']+'.json'),r);review.append(committed)
                    # These are actual formal Source/query pairs, not extra data.
                    result=dict(success=True,objective=check['objective'],free=check['free'],feasibility=check,
                        iterations=(response['solver'].get('convergence') or {}).get('nb_iterations'),
                        solver_engine_seconds=response['solver'].get('engine_seconds'),shared_batch_end_to_end_seconds=elapsed,batch_pairs=len(todo),
                        **{k:check[k] for k in ('acceptance','stop_reason','backend','negative_mass','normalized_gap')})
                    pair=rt.pair_path(a,item['query'],item['template'])
                    if not pair.exists():st.sealed_json(pair,dict(protocol=lp_v2.PROTOCOL,dependencies=item['dependencies'],
                        query=item['query'],template=item['template'],result=result,proof=proof['proof'],proof_seal=record['seal'],distance_seconds=0.,GT_used=False))
    attempts=[st.sealed_json(p) for p in sorted((a.run_root/'review_attempts').glob('*.json'))]
    def table(r):return dict(id=r['id'],**{k:v for k,v in r['audit'].items() if k!='checks'},
        solver_primal_residual=(r['solver'].get('convergence') or {}).get('primal_residual'),
        solver_dual_residual=(r['solver'].get('convergence') or {}).get('dual_residual'),
        iterations=(r['solver'].get('convergence') or {}).get('nb_iterations'),engine_seconds=r['solver'].get('engine_seconds'))
    st.write_csv(a.run_root/'LP_review16_attempts.csv',[table(r) for r in attempts])
    selected={r['id']:r for r in attempts};selected.update({r['id']:r for r in review})
    st.write_csv(a.run_root/'LP_review16.csv',[table(selected[f'{i:02d}']) if f'{i:02d}' in selected else dict(id=f'{i:02d}',acceptance='uncompleted') for i in range(16)])
    if len(review)!=16:
        return dict(status='pending',reason='16-LP v2 certificates not all accepted',accepted=len(review),required=16)
    # Two necessary, previously absent formal query assets measure cold/warm
    # extraction and region preparation without rebuilding any existing input.
    prep=[]
    for row in x.samples('background'):
        if rt.cached_region_path(a,row,rms) is not None:continue
        tick=time.perf_counter();_,r=rt.region(a,row,rms)
        prep.append(dict(image=ip.image(row),seconds=time.perf_counter()-tick,region_seconds=r['metadata']['seconds']))
        if len(prep)==2:break
    distance_probe=[]
    if inventory(a).get('missing_distance_matrices',0):
        references=[(t,*rt.region(a,t,rms)) for t in templates];queries_measured=0
        for row in x.samples('background'):
            if all((ip.key(row),ip.key(t)) in mr.exports(a) or (a.cache_root/'distances'/ip.key(row)/ip.key(t)/'manifest.json').exists() for t in templates):continue
            qdata,qmeta=rt.region(a,row,rms)
            for template,tdata,tmeta in references:
                item=mr.distance(a,rms,row,template,qdata,qmeta,tdata,tmeta)
                if not item['reused_distance']:distance_probe.append(item['distance_seconds'])
            queries_measured+=1
            if queries_measured==2:break
    saved=st.sealed_json(a.run_root/'smoke.json',dict(identity=frozen['identity'],protocol=lp_v2.PROTOCOL,
        review=[dict(id=r['id'],seal=r['seal']) for r in review],preparation=prep,forward_seconds=getattr(a,'_forward_rates',[]),
        distance_probe_seconds=distance_probe,
        review_seconds=sum({r['batch_job']:r['end_to_end_batch_seconds'] for r in attempts}.values()),accepted=16,GT_used=False))
    return dict(status='v2_smoke_complete',accepted=16,estimate=estimate(a,saved))


def regions_file():
    from . import regions
    return regions.__file__


def estimate(a,smoke_record=None):
    inv=inventory(a);r=smoke_record or st.sealed_json(a.run_root/'smoke.json')
    lp_seconds=inv['remaining_LPs_upper_bound']*r['review_seconds']/16
    prep=r.get('preparation',[]);unknown=['CPU_image_evaluation_and_final_IO']
    region_rate=float(np.mean([p['region_seconds'] for p in prep])) if prep else None
    forwards=r.get('forward_seconds',[]);warm=float(np.mean(forwards[1:])) if len(forwards)>1 else None
    d=r.get('distance_probe_seconds',[]);distance_rate=float(np.mean(d[1:])) if len(d)>1 else None
    if inv['remaining_regions'] and region_rate is None:unknown.append('unmeasured_region_throughput')
    if inv['missing_feature_images'] and warm is None:unknown.append('unmeasured_warm_feature_throughput')
    if inv.get('missing_distance_matrices',0) and distance_rate is None:unknown.append('unmeasured_distance_compute_commit')
    parts=dict(LP_build_transfer_solve_audit_commit=lp_seconds,
        region_preparation=0. if inv['remaining_regions']==0 else None if region_rate is None else inv['remaining_regions']*region_rate,
        feature_forward=0. if inv['missing_feature_images']==0 else None if warm is None else inv['missing_feature_images']*warm,
        distance_compute_commit=0. if inv.get('missing_distance_matrices',0)==0 else None if distance_rate is None else inv['missing_distance_matrices']*distance_rate)
    result=dict(status='phase1_real_asset_estimate',plan=PLAN,inventory=inv,parts_seconds=parts,
        known_pending_seconds=sum(v for v in parts.values() if v is not None),unknown_stages=unknown,unknown_is_not_zero=True,
        includes_all_review_batches_and_worker_setup=True,already_completed_work_excluded=True,scope='one domain/one K8; 16 real reference LPs, not all-image runtime guarantee')
    st.write_json(a.run_root/'cost_estimate.json',result)
    text=['# 本阶段成本构成','',*[f'- {k}: {v/3600:.4f}小时' if v is not None else f'- {k}: 未测，不能当0' for k,v in parts.items()],
        '',f"已知待执行：{result['known_pending_seconds']/3600:.4f}小时；另有未知项：{', '.join(unknown)}。",
        '超过12小时则等待用户明确同意，不启动后续全量工作。',
        '优化方向：核实未接入的既有特征/区域/距离来源以减少补算；保持固定数值协议优化批处理及IO，重新测真实端到端吞吐。',
        '不减少289/66名单、不放宽本协议、不添加全量CPU精修、不根据AUROC/AP选择参数。']
    (a.run_root/'budget_analysis.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    pipeline.budget(a,result);return result


def summarize(a):
    pending=[];metrics=[]
    path=a.run_root/'evaluation_complete.json'
    if path.exists():
        e=st.sealed_json(path)
        if e['identity']!=ip.signature(a):raise ValueError('evaluation identity differs')
        for name,expected in e['files'].items():
            if st.sha(a.run_root/name)!=expected:raise ValueError('completed metric table modified')
        frozen=st.sealed_json(a.run_root/'predictions_frozen.json')
        if frozen['seal']!=e['prediction_seal']:raise ValueError('evaluation prediction freeze differs')
        metrics=ip.csv_rows(a.run_root/'setting_metrics.csv')
        if len(metrics)!=6 or {r['branch'] for r in metrics}!=set(BRANCHES):pending.append('six requested image branches')
    else:pending.append('frozen background scores and image evaluation')
    pairs=[];uncompleted=[]
    for _,_,kind,row in pipeline.observations(a):
        for template in ip.assets(a).supports('background',42):
            path=rt.pair_path(a,row,template)
            if not path.exists():
                uncompleted.append(dict(query=row['relative_path'],template=template['relative_path'],kind=kind,status='uncompleted'))
                continue
            r=st.sealed_json(path)
            if r['protocol']!=lp_v2.PROTOCOL or r['query']!=ip.image(row) or r['template']!=ip.image(template) or not r['result']['success']:
                raise ValueError('completed LP metadata is incompatible')
            pairs.append(r)
    if len(pairs)!=2840:pending.append(f'LPs {len(pairs)}/2840')
    summary=[]
    for r in pairs:
        result=r['result'];f=result['feasibility']
        summary.append(dict(query=r['query']['relative_path'],template=r['template']['relative_path'],
            backend=result['backend'],acceptance=result['acceptance'],stop_reason=result['stop_reason'],
            solver_batch_iterations=result.get('iterations'),solver_batch_engine_seconds=result.get('solver_engine_seconds'),
            shared_batch_end_to_end_seconds=result.get('shared_batch_end_to_end_seconds'),batch_pairs=result.get('batch_pairs'),
            **{k:v for k,v in f.items() if not isinstance(v,(dict,list))}))
    st.write_csv(a.run_root/'LP_completion.csv',[dict(status='accepted',**r) for r in summary]+uncompleted)
    failures=[]
    for p in (a.run_root/'attempts').glob('*/*.json'):
        r=st.sealed_json(p)
        if not r['result']['success']:failures.append(dict(query=r['query']['relative_path'],template=r['template']['relative_path'],
            source=str(p),**{k:v for k,v in r['result']['feasibility'].items() if not isinstance(v,(dict,list))}))
    for p in (a.run_root/'review_attempts').glob('*.json'):
        r=st.sealed_json(p)
        if not r['audit']['accepted']:failures.append(dict(problem_id=r['id'],source=str(p),
            **{k:v for k,v in r['audit'].items() if not isinstance(v,(dict,list))}))
    st.write_csv(a.run_root/'LP_failed_attempts.csv',failures)
    diff=reporting.differences(metrics);st.write_csv(a.run_root/'metric_comparisons.csv',diff)
    status=dict(status='pending' if pending else 'background_seed42_image_evaluation_complete',pending=pending,plan=PLAN,
        completed_LPs=len(pairs),CPU_reused=sum(r['result']['backend']=='HiGHS-DS' for r in pairs),
        GPU_certified=sum(r['result']['backend']=='PDLP' for r in pairs),
        timeout_accepted=sum(r['result']['acceptance']=='timeout_numerically_verified' for r in pairs),
        failed_attempts=len(failures),uncompleted_LPs=2840-len(pairs),protocol=lp_v2.PROTOCOL['version'],no_automatic_expansion=True)
    st.write_json(a.run_root/'status.json',status)
    lines=['# 面积容量约束的反向覆盖：background / seed42','',
        '289张共同评价图（77正常、212异常），Source66，固定K8；仅六分支Image AUROC/AP。',
        '使用capacity_LP_mixed_v2。混合迭代不是全FP32；保留返回t作为覆盖分数，无裁剪或重新归一化。',
        '逐LP独立检查原始约束、对偶可行性及归一化间隙。超时数值通过不称为已证明最优。',
        '严格CPU复用另列来源，其对偶依据为原HiGHS最优状态与严格停止设置，未伪称独立重算对偶。','']
    if pending:lines.append('尚未完成：'+', '.join(pending)+'。没有完整效果结论。')
    else:
        lines+=['| 分支 | Image AUROC % | Image AP % |','|---|---:|---:|']
        for r in metrics:lines.append(f"|{r['branch']}|{100*float(r['image_AUROC']):.6f}|{100*float(r['image_AP']):.6f}|")
        for baseline in (BASE,SHARED,OWN):
            changes=[r for r in diff if r['reference']==baseline]
            lines.append(PRIMARY+' − '+baseline+'：'+'；'.join(f"{r['metric']} {r['delta_pp']:+.6f}百分点" for r in changes)+'。')
        lines+=['主候选相对SEGMENT检验反向覆盖读出收益；相对两个FREE融合才检验容量的额外价值。',
            '如果仅FREE改善，应如实归因为反向信息；本设置不足以证明跨域或跨支持稳定，不自动扩展。']
        idx={r['branch']:r for r in metrics}
        metrics_keys=('image_AUROC','image_AP')
        base_delta=[float(idx[PRIMARY][k])-float(idx[BASE][k]) for k in metrics_keys]
        extra=[float(idx[PRIMARY][k])-float(idx[b][k]) for b in (SHARED,OWN) for k in metrics_keys]
        if all(v>0 for v in base_delta) and all(v>0 for v in extra):
            lines.append('本设置两项图像指标均改善SEGMENT并超过两个FREE融合，值得讨论独立的下一阶段复验；当前仅是单域/单支持证据，不自动启动。')
        elif all(v<=0 for v in base_delta):
            lines.append('本设置主候选未改善SEGMENT的两项图像指标，当前证据不支持直接扩大此固定公式的投入。')
        else:
            lines.append('本设置仍有指标取舍或未同时超过两个FREE对照，尚不能认定容量约束形成明确实际优势。')
        if any(float(idx[b][k])>float(idx[BASE][k]) for b in (SHARED,OWN) for k in metrics_keys) and not all(v>0 for v in extra):
            lines.append('反向信息有正向指标，但容量并未在两项指标上同时超过全部FREE对照；不能把反向信息收益全部归因于容量。')
    lines+=['','八模板min对最大绝对输入误差是非扩张的：|min u−min v|≤max|u−v|。',
        '标准化可按1/std放大误差，16题抽查不能保证全部图像排序不变。',
        '时间预算包含建模、传输、验收、落盘和必要补特征/分区，不以GPU利用率或仅内核时间代替吞吐。']
    cal=a.run_root/'background/seed42/calibration.json'
    if cal.exists():
        model=st.sealed_json(cal)['parameters'];st.write_json(a.run_root/'Source_calibration_scales.json',model)
        for name in ('P','cap','free'):lines.append(f"Source {name}: mean={model[name]['mean']:.12g}, std={model[name]['std']:.12g}。")
        review=ip.csv_rows(a.run_root/'LP_review16.csv');sigma=model['cap']['std']
        impact=[dict(problem_id=r['id'],raw_t_error=float(r['reference_abs_error']),cap_std=sigma,
            error_over_cap_std=float(r['reference_abs_error'])/sigma if model['cap']['status']=='valid' else None) for r in review if r.get('reference_abs_error')]
        st.write_csv(a.run_root/'LP_error_in_calibration_units.csv',impact)
        if impact and sigma>0:lines.append(f"16题最大原始t误差 / cap std = {max(r['error_over_cap_std'] for r in impact):.12g}；不是全量排序保证。")
        lines.append('误差相对std若不可忽略，应保留数值敏感性限制，不把全部排序变化归因于方法；本轮不会据测试指标调整精度或容差。')
    (a.run_root/'objective_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    costs=[st.read_json(p) for p in (a.run_root/'costs').glob('*.json')]
    responses=[st.sealed_json(p) for p in (a.run_root/'solver_jobs').glob('*_response.json')]
    def peak(values):
        values=[v for v in values if v is not None];return max(values) if values else None
    st.write_json(a.run_root/'resource_summary.json',dict(recorded_stage_seconds=sum(r['seconds'] for r in costs),
        process_tree_peak_RSS_bytes=peak(r.get('peak_process_tree_RSS_bytes') for r in costs),
        worker_peak_RSS_bytes=peak(r.get('resource',{}).get('peak_RSS_bytes') for r in responses),
        worker_GPU_bytes_after_batch_max=peak(r.get('resource',{}).get('GPU_memory_bytes_after_batch') for r in responses),
        main_peak_cuda_allocated_bytes=peak(r.get('peak_cuda_allocated_bytes') for r in costs),
        main_peak_cuda_reserved_bytes=peak(r.get('peak_cuda_reserved_bytes') for r in costs),
        GPU_measurement='after-batch sampled residency, not guaranteed transient peak',LPs=status,stage_tail_excluded='current summarize/pack'))
    return status


def execute(a):
    try:configure(a)
    except (OSError,KeyError,ValueError) as e:
        if a.stage!='preflight':raise
        result=dict(status='missing_inputs',plan=PLAN,reason=f'{type(e).__name__}: {e}',torch_imported='torch' in sys.modules)
        st.write_json(a.run_root/'preflight.json',result)
        st.write_csv(a.run_root/'missing_inputs.csv',[dict(role='existing_coverage_run_or_readonly_providers',path=str(a.resume_from),reason=result['reason'])])
        return result
    if a.stage=='pack':return reporting.pack(a)
    if a.stage=='preflight':
        try:
            frozen=freeze(a,commit=False);inv=inventory(a);old_audit(a)
            ready=inv['RMS_available'] and inv['solver_environment_available'] and (not inv['missing_feature_images'] or a.extract_missing)
            result=dict(status='phase1_preflight_complete' if ready else 'missing_inputs',
                **inv,identity=frozen['identity'],torch_imported='torch' in sys.modules,new_GPU_work=False)
        except (OSError,KeyError,ValueError) as e:
            result=dict(status='missing_inputs' if isinstance(e,OSError) else 'blocked',plan=PLAN,
                reason=f'{type(e).__name__}: {e}',torch_imported='torch' in sys.modules,new_GPU_work=False)
            st.write_csv(a.run_root/'missing_inputs.csv',[dict(role='phase1_input_or_solver_metadata',reason=result['reason'])])
        st.write_json(a.run_root/'preflight.json',result);return result
    if ip.rms_record(a) is None:raise FileNotFoundError('existing Source RMS metadata required for phase1; no automatic Source455 rebuild: '+str(a.rms_stat))
    if a.stage=='smoke':return smoke(a)
    freeze(a)
    if a.stage in ('prepare','predict'):
        r=st.sealed_json(a.run_root/'smoke.json')
        if r['identity']!=ip.signature(a) or r['accepted']!=16:raise ValueError('complete current v2 16-LP review required')
        estimate(a,r)
        return pipeline.prepare(a) if a.stage=='prepare' else pipeline.predict(a)
    if a.stage=='evaluate':return pipeline.evaluate(a)
    return summarize(a)
