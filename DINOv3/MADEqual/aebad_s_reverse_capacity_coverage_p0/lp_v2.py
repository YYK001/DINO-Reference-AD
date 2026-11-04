"""Shared original-coefficient LP acceptance and bounded backend checks."""
import time
_IMPORT_STARTED=time.perf_counter()
import numpy as np
from . import transport,gpu_lp as gpu

PROTOCOL=dict(version='capacity_LP_mixed_v2',row_atol=1e-5,capacity_rtol=1e-5,objective_rtol=1e-5,
    CPU_reference_rtol=1e-5,dual_feasibility=1e-5,normalized_gap=1e-5,min_W=-1e-10,
    negative_mass='sum(max(-W,0)); recorded, no clipping',score='returned_t',
    primal_objective='c@x=t',dual_objective='rhs@y',
    dual_convention='min c@x; E free, L multipliers<=0; c-A.T@y>=0; x>=0',
    dual_normalization='max positive(A.T@y-c)/max(1,||c||inf); L sign violation/max(1,||c||inf)',
    gap_normalization='abs(p-d)/max(1,abs(p)+abs(d))',repair=False,
    scope=dict(domain='background',support_seed=42,evaluation=289,normal=77,anomaly=212,calibration=66,templates=8,LPs=2840))


def settings(cpu_workers=7):
    # Different stopping criteria from the former CPU solver; independent checks
    # below, including relative small-region capacity, are authoritative.
    s=gpu.settings_values(30.,cpu_workers,'mixed')
    for k in ('absolute_primal_tolerance','relative_primal_tolerance','absolute_dual_tolerance',
              'relative_dual_tolerance','absolute_gap_tolerance','relative_gap_tolerance'):s[k]=1e-7
    s['per_constraint_residual']=True
    return s


PROTOCOL['solver']={k:v for k,v in settings().items() if k!='num_cpu_threads'}
PROTOCOL['strict_CPU_reuse']='original sealed HiGHS Optimal, original strict residual checks revalidated; unavailable dual/gap remain NA'


def audit(problem,primal,dual,status,backend='PDLP',requested_backend='PDLP'):
    """A per-problem dual certificate, never an eight-problem aggregate gap."""
    d,a,b=transport.validate_inputs(problem['distance'],problem['a'],problem['b'])
    x=np.asarray(primal,np.float64);y=np.asarray(dual,np.float64);n,m=d.shape
    result=dict(accepted=False,acceptance='incomplete',stop_reason=str(status),backend=backend,
        requested_backend=requested_backend,protocol=PROTOCOL['version'],repaired=False)
    if x.shape!=(d.size+1,) or y.shape!=(2*n+m,) or not all(np.isfinite(z).all() for z in (x,y)):
        return result|dict(reason='missing/nonfinite/wrong-shape primal or dual evidence')
    eq,rhs,ub,upper=transport.constraints(d,a,b);c=np.zeros(len(x));c[-1]=1
    W=x[:-1].reshape(d.shape);t=float(x[-1]);cost=np.sum(W*d,axis=1);capacity=a@W
    reduced=c-eq.T@y[:n+m]-ub.T@y[n+m:]
    scale=max(1.,float(abs(c).max()));p=float(c@x);dual_objective=float(rhs@y[:n+m]+upper@y[n+m:])
    metrics=dict(row_sum_max_error=float(abs(W.sum(1)-1).max()),
        capacity_max_relative_error=float((abs(capacity-b)/b).max()),
        min_W=float(W.min()),negative_mass=float(np.maximum(-W,0).sum()),min_t=t,
        primal_equality_Linf=float(abs(eq@x-rhs).max()),primal_inequality_violation=float(np.maximum(ub@x-upper,0).max()),
        objective_consistency_error=float(abs(cost.max()-t)),objective_max_violation=float(max(0.,cost.max()-t)),
        free_lower_bound_gap=t-transport.free_cost(d),primal_objective=p,dual_objective=dual_objective,
        dual_reduced_cost_violation=float(np.maximum(-reduced,0).max())/scale,
        dual_inequality_sign_violation=float(np.maximum(y[n+m:],0).max())/scale,
        normalized_gap=float(abs(p-dual_objective)/max(1.,abs(p)+abs(dual_objective))))
    checks=dict(row=metrics['row_sum_max_error']<=1e-5,capacity=metrics['capacity_max_relative_error']<=1e-5,
        nonnegative=metrics['min_W']>=-1e-10 and t>=-1e-10,
        objective=metrics['objective_consistency_error']<=1e-5*max(1.,abs(t)),
        free_lower_bound=metrics['free_lower_bound_gap']>=-1e-5*max(1.,abs(t)),
        dual=metrics['dual_reduced_cost_violation']<=1e-5 and metrics['dual_inequality_sign_violation']<=1e-5,
        gap=metrics['normalized_gap']<=1e-5)
    if problem.get('reference_objective') is not None:
        reference=float(problem['reference_objective']);metrics['reference_abs_error']=abs(t-reference)
        checks['CPU_reference']=metrics['reference_abs_error']<=1e-5*max(1.,abs(reference))
    normalized=str(status).replace('_','').replace(' ','').lower()
    category='solver_optimal_independently_verified' if normalized=='optimal' else 'timeout_numerically_verified' if normalized=='timelimit' else None
    accepted=bool(category and requested_backend in ('PDLP','Barrier') and backend==requested_backend and all(checks.values()))
    return result|metrics|dict(accepted=accepted,acceptance=category if accepted else 'incomplete',checks=checks,
        objective=t,free=transport.free_cost(d),reason=None if accepted else 'stop state or independent certificate failed')


class Solver(gpu.CuOptPDLP):
    def __init__(self,cpu_workers=7,time_limit=30.):
        super().__init__(time_limit,cpu_workers,'mixed')
        from cuopt.linear_programming.solver import solver_parameters as keys
        self.parameters=settings(cpu_workers)
        self.parameters['time_limit']=float(time_limit)
        for name,value in self.parameters.items():
            if name in ('method','pdlp_solver_mode'):continue
            self.settings.set_parameter(getattr(keys,'CUOPT_'+name.upper()),value)

    def solve_certified(self,problems,coordinate_system='original'):
        # Reuse the exact same DataModel construction and solution extraction;
        # add the dual vector through a small extension in the shared adapter.
        started=time.perf_counter()
        if coordinate_system=='original':
            batch=gpu.sparse_batch(problems);variable_scale=row_scale=None
        elif coordinate_system=='mass_flow_v1':
            batch,variable_scale,row_scale=mass_flow_batch(problems)
        elif coordinate_system=='mass_flow_basis_v1':
            batch,variable_scale,row_scale=mass_flow_basis_batch(problems)
        else:raise ValueError('unknown LP coordinate system')
        solution=self.solve(batch)
        output=[];row=0;full_row=0
        dual=solution.get('dual',np.array([]));primal=solution['primal']
        evidence_shape_ok=primal.shape==(batch['matrix'].shape[1],) and dual.shape==(batch['matrix'].shape[0],)
        for index,(problem,(start,end)) in enumerate(zip(problems,batch['slices'])):
            n,m=problem['distance'].shape;count=2*n+m
            basis=coordinate_system=='mass_flow_basis_v1';raw_count=count-int(basis)
            raw_x=primal[start:end];raw_y=dual[row:row+raw_count]
            x=raw_x if variable_scale is None or raw_x.shape!=(end-start,) else raw_x/variable_scale[start:end]
            if basis and raw_y.shape==(raw_count,):
                # Zero is the dual multiplier of the omitted dependent row;
                # this fixes its free gauge, not the primal transport plan.
                expanded=np.zeros(count);keep=np.arange(count)!=n+batch['omitted_capacity'][index]
                expanded[keep]=raw_y;y=expanded*row_scale[full_row:full_row+count]
            else:
                y=raw_y if row_scale is None or raw_y.shape!=(count,) else raw_y*row_scale[full_row:full_row+count]
            row+=raw_count;full_row+=count
            certificate=audit(problem,x,y,solution['status'],solution['solved_by'])
            if not evidence_shape_ok:
                certificate.update(accepted=False,acceptance='incomplete',reason='wrong-shape batch primal or dual evidence')
            if solution.get('error_status')!=0:
                certificate.update(accepted=False,acceptance='incomplete',reason='solver error status missing or nonzero')
            record=dict(primal=x,dual=y,audit=certificate)
            if variable_scale is not None:record.update(solver_primal=raw_x,solver_dual=raw_y)
            if basis:record['audit'].update(omitted_capacity_region=batch['omitted_capacity'][index],
                mass_balance_defect=batch['mass_balance_defect'][index],all_original_constraints_checked='checks' in certificate)
            output.append(record)
        return output,dict(seconds=time.perf_counter()-started,solver={k:v for k,v in solution.items() if k not in ('primal','dual')})


def mass_flow_batch(problems):
    """Positive diagonal change of coordinates, not a different transport LP.

    k=n, z_ij=k*a_i*W_ij; t is unchanged. Row/capacity/cost rows
    are multiplied by k*a_i, k, k*a_i, respectively. Thus the
    recovered original dual is y=R*y_scaled and W=z/(k*a_i).
    No row is removed, no coefficient is clipped, no plan is repaired.
    """
    batch=gpu.sparse_batch(problems);variables=[];rows=[]
    for problem in problems:
        d,a,b=transport.validate_inputs(problem['distance'],problem['a'],problem['b']);n,m=d.shape
        mass=n*a
        variables.append(np.r_[np.repeat(mass,m),1.])
        rows.append(np.r_[mass,np.full(m,float(n)),mass])
    variable_scale=np.concatenate(variables);row_scale=np.concatenate(rows)
    batch['matrix']=batch['matrix'].multiply(row_scale[:,None]).multiply(1./variable_scale[None,:]).tocsr()
    batch['rhs']=batch['rhs']*row_scale
    batch['objective']=batch['objective']/variable_scale
    return batch,variable_scale,row_scale


def mass_flow_basis_batch(problems):
    """Remove one algebraically dependent capacity row per LP, then audit all.

    The largest b_j is selected using input areas only. No mass is changed.
    Original row and column mass sums must agree to FP64 summation roundoff.
    The omitted constraint remains part of the original-coefficient audit.
    """
    batch,S,R=mass_flow_batch(problems);keep=np.ones(batch['matrix'].shape[0],bool)
    offset=0;omitted=[];defects=[]
    for problem in problems:
        d,a,b=transport.validate_inputs(problem['distance'],problem['a'],problem['b']);n,m=d.shape
        defect=float(a.sum()-b.sum());roundoff=8*np.finfo(np.float64).eps*max(1.,float(abs(a).sum()+abs(b).sum()))
        if abs(defect)>roundoff:raise ValueError('dependent-row reduction requires mass sums equal to FP64 summation roundoff; no renormalization')
        anchor=int(np.argmax(b));keep[offset+n+anchor]=False
        omitted.append(anchor);defects.append(defect);offset+=2*n+m
    batch['matrix']=batch['matrix'][keep].tocsr();batch['rhs']=batch['rhs'][keep];batch['senses']=batch['senses'][keep]
    batch.update(omitted_capacity=omitted,mass_balance_defect=defects)
    return batch,S,R


def mass_flow_trial(a,backend_factory=Solver):
    """Fixed 16-case diagnostic only; never switches the formal LP backend."""
    import os
    from pathlib import Path
    manifest,problems=gpu.load_problems(a)
    coordinate_system=getattr(a,'coordinate_system','mass_flow_v1')
    if coordinate_system not in ('mass_flow_v1','mass_flow_basis_v1'):raise ValueError('fixed mass-flow trial only')
    time_limit=300. if coordinate_system=='mass_flow_basis_v1' else 30.
    actual_settings=dict(settings(a.cpu_workers),time_limit=time_limit)
    binding=dict(kind='capacity_mass_flow_trial_v1',coordinate_system=coordinate_system,
        problems=manifest['seal'],problem_root=str(a.problem_root),acceptance_protocol=PROTOCOL,actual_settings=actual_settings,
        versions=gpu.packages(),visible_device=os.environ.get('CUDA_VISIBLE_DEVICES'),
        implementation=gpu.st.sha(Path(__file__)),batch_size=8,repeats=1,formal_backend_changed=False)
    gpu.st.sealed_json(a.output_root/'trial_inputs.json',binding)
    started=time.perf_counter()
    solver=backend_factory(a.cpu_workers,time_limit=time_limit) if time_limit!=30. else backend_factory(a.cpu_workers)
    checks=[];batch_seconds=[]
    for start in (0,8):
        tick=time.perf_counter();group=problems[start:start+8]
        results,timing=solver.solve_certified(group,coordinate_system=coordinate_system)
        if len(results)!=len(group):raise ValueError('missing diagnostic LP solutions')
        for problem,result in zip(group,results):
            check=dict(id=problem['id'],**result['audit']);checks.append(check)
            gpu.st.put(a.output_root/'proofs'/problem['id'],
                {k:result[k] for k in ('primal','dual','solver_primal','solver_dual')},
                gpu.finite_record(dict(trial=binding,original_input_id=problem['id'],
                    audit=check,solver=timing['solver'],coordinate_mapping='x_original=S^-1*x_solver; y_original=R*y_solver; omitted dual row=0 when basis reduction requested')))
            print('[mass-flow original-coefficient audit]',problem['id'],
                {k:check.get(k) for k in ('accepted','stop_reason','reference_abs_error',
                    'dual_reduced_cost_violation','normalized_gap')},flush=True)
        elapsed=time.perf_counter()-tick;batch_seconds.append(elapsed)
        gpu.st.sealed_json(a.output_root/f'batch{start//8}.json',gpu.finite_record(
            dict(seconds=elapsed,solver=timing['solver'],checks=[dict(id=p['id'],**r['audit']) for p,r in zip(group,results)])))
    accepted=sum(r['accepted'] for r in checks)
    summary=dict(status='mass_flow_trial_passed' if accepted==16 else 'mass_flow_trial_pending',
        coordinate_system=coordinate_system,actual_settings=actual_settings,
        accepted=accepted,required=16,seconds=time.perf_counter()-started,batch_end_to_end_seconds=batch_seconds,
        formal_backend_changed=False,feature_forwards=0,CPU_reference_recomputed=False,
        maximum_errors={k:max((r[k] for r in checks if r.get(k) is not None),default=None) for k in
            ('row_sum_max_error','capacity_max_relative_error','objective_consistency_error',
             'reference_abs_error','dual_reduced_cost_violation','dual_inequality_sign_violation','normalized_gap')})
    gpu.st.write_csv(a.output_root/'LP_checks.csv',gpu.finite_record(checks))
    gpu.st.write_json(a.output_root/'summary.json',gpu.finite_record(summary))
    return summary


def final_settings(backend,cpu_workers=7):
    """The last, fixed comparison; Barrier receives no PDLP-only options."""
    if backend=='PDLP':
        return dict(settings(cpu_workers),time_limit=120.,presolve=2,save_best_primal_solution=True)
    if backend!='Barrier':raise ValueError('only the two requested backends are allowed')
    return dict(method='Barrier',time_limit=120.,num_cpu_threads=cpu_workers,num_gpus=1,crossover=False,presolve=2,
        **{k:1e-8 for k in ('absolute_primal_tolerance','relative_primal_tolerance','absolute_dual_tolerance',
            'relative_dual_tolerance','absolute_gap_tolerance','relative_gap_tolerance')})


def final_single(a,index,problem,backend,adapters,backend_factory,resource_reader):
    """One reserved call. Completed failures and interrupted calls are never retried."""
    from pathlib import Path
    import traceback
    root=a.output_root/'cases'/f'{backend}_{index:02d}';record_path=root/'result.json'
    request=dict(identity=a._final_identity,index=index,id=problem['id'],requested_backend=backend,
        settings=final_settings(backend,a.cpu_workers),
        precision_policy='mixed=2' if backend=='PDLP' else 'native Barrier precision, FP64 input; PDLP-only options absent')
    def checked(solution):
        result=audit(problem,solution.get('primal',[]),solution.get('dual',[]),solution.get('status','missing'),
            solution.get('solved_by','missing'),requested_backend=backend)
        if solution.get('error_status')!=0:result.update(accepted=False,acceptance='incomplete',reason='solver error status missing or nonzero')
        for name,metric in (('objective','primal_objective'),('reported_dual_objective','dual_objective')):
            reported=solution.get(name);actual=result.get(metric)
            if reported is None or actual is None or not np.isfinite(reported) or abs(reported-actual)>1e-5*max(1.,abs(actual)):
                result.update(accepted=False,acceptance='incomplete',reason='solver-reported objective and returned vector do not agree')
        return result
    if record_path.exists():
        record=gpu.st.sealed_json(record_path)
        if record['request']!=request:raise ValueError('final single-case configuration differs')
        arrays,proof=gpu.st.get(root/'vectors');solution=dict(record['solver'],**arrays)
        check=checked(solution)
        if check['accepted']!=record['audit']['accepted']:raise ValueError('saved final certificate no longer verifies')
        record['end_to_end_seconds']=gpu.st.sealed_json(root/'cost.json')['end_to_end_seconds'] if (root/'cost.json').exists() else None
        return record
    if (root/'attempt.json').exists():
        # A crash may occur after Solve but before commit. Consuming the slot
        # rather than retrying enforces the user's maximum number of calls.
        old=gpu.st.sealed_json(root/'attempt.json')
        if old!=dict(request,seal=gpu.st.digest(request)):raise ValueError('reserved final call differs')
        return dict(request=request,audit=dict(accepted=False,backend='unknown',stop_reason='interrupted',reason='reserved call has no committed result; no retry'),
            end_to_end_seconds=None,error='interrupted call',timing={},resource={},solver={})
    started=time.perf_counter();gpu.st.sealed_json(root/'attempt.json',request)
    print('[final single START]',problem['id'],backend,'original W coordinates; time_limit=120s',flush=True)
    init=0.;sparse_seconds=0.;solution=dict(primal=np.array([]),dual=np.array([]),status='not_returned',solved_by='missing',error_status=None)
    error=None;solve_invoked=False
    try:
        if backend not in adapters:
            tick=time.perf_counter()
            try:adapters[backend]=(backend_factory(120.,a.cpu_workers,'mixed',backend=backend,parameters=request['settings']),None)
            except Exception:adapters[backend]=(None,traceback.format_exc())
            init=time.perf_counter()-tick
        adapter,initialization_error=adapters[backend]
        if initialization_error:raise RuntimeError(initialization_error)
        from cuopt.linear_programming.solver import solver_parameters as keys
        log=root/'solver.log';adapter.settings.set_parameter(keys.CUOPT_LOG_FILE,str(log))
        tick=time.perf_counter();batch=gpu.sparse_batch([problem]);sparse_seconds=time.perf_counter()-tick
        solve_invoked=True;solution=adapter.solve(batch)
        effective=dict(adapter.effective_settings,log_file=str(log))
    except Exception:
        error=traceback.format_exc();effective={}
    tick=time.perf_counter();check=checked(solution)
    gpu.st.put(root/'vectors',dict(primal=np.asarray(solution['primal']),dual=np.asarray(solution['dual'])),
        gpu.finite_record(dict(request=request,effective_settings=effective,audit=check)))
    resources=resource_reader()
    timing=dict(backend_initialization_seconds=init,sparse_construction_seconds=sparse_seconds,
        model_seconds=solution.get('model_seconds'),solve_call_seconds=solution.get('solve_call_seconds'),
        engine_seconds=solution.get('engine_seconds'),audit_commit_resource_seconds=time.perf_counter()-tick)
    call=solution.get('solve_call_seconds');engine=solution.get('engine_seconds')
    timing['transfer_and_API_overhead_seconds']=max(0.,call-engine) if call is not None and engine is not None else None
    record=dict(request=request,audit=check,solver={k:v for k,v in solution.items() if k not in ('primal','dual')},
        solve_invoked=solve_invoked,
        effective_settings=effective,timing=timing,end_to_end_seconds=time.perf_counter()-started,resource=resources,error=error,
        cold_backend_initialization=bool(init),time_note='initial CUDA/context setup can also occur inside first Solve; API overhead is not a pure H2D measurement')
    gpu.st.sealed_json(record_path,gpu.finite_record(record))
    gpu.st.sealed_json(root/'cost.json',dict(end_to_end_seconds=time.perf_counter()-started,
        note='includes result/proof commit; excludes only writing this final timing metadata'))
    print('[final single]',problem['id'],backend,check['stop_reason'],'accepted='+str(check['accepted']),flush=True)
    saved=gpu.st.sealed_json(record_path)
    saved['end_to_end_seconds']=gpu.st.sealed_json(root/'cost.json')['end_to_end_seconds']
    return saved


def final_cost(a,manifest,problems,records,backend):
    """Reuse the existing phase1 inventory and historical measured prep rates."""
    from pathlib import Path
    from types import SimpleNamespace
    from . import mixed_runtime as mr
    inventory=getattr(a,'_final_inventory',None)
    if inventory is None:inventory=gpu.st.read_json(a.inventory_run/'inventory.json')
    if not getattr(a,'_final_inventory_checked',False):
        frozen=gpu.st.read_json(a.inventory_run/'preflight_inputs.json')
        if frozen['parent']!=manifest['parent_inputs'] or inventory['plan']['LPs']!=2840:
            raise ValueError('cost inventory belongs to another parent or scope')
        a._final_inventory=inventory;a._final_inventory_checked=True
    cpu_reuse=getattr(a,'_final_cpu_reuse',None)
    if cpu_reuse is None:
        cpu_reuse=set()
        for i,ref in enumerate(manifest['problems']):
            meta=gpu.st.sealed_json(a.problem_root/'problems'/ref['id']/'manifest.json')['metadata']
            dep=meta['dependencies']
            result=mr.strict_cpu(SimpleNamespace(resume_from=a.coverage_parent),{},meta['query'],meta['template'],
                {k:dep[k] for k in ('query','template','RMS','distance_device')})
            if result:cpu_reuse.add(i)
        a._final_cpu_reuse=cpu_reuse
    accepted={r['request']['index'] for r in records if r['request']['requested_backend']==backend and r['audit']['accepted']}
    observed=[r for r in records if r['request']['requested_backend']==backend]
    durations=[r['end_to_end_seconds'] for r in observed]
    rate=float(np.mean(durations)) if durations and all(s is not None for s in durations) else None
    explicit_init=sum(r.get('timing',{}).get('backend_initialization_seconds',0.) for r in observed)
    warm_rate=max(0.,rate-explicit_init/len(observed)) if rate is not None else None
    next_process_init=max((r.get('timing',{}).get('backend_initialization_seconds',0.) for r in observed),default=0.)
    pending=2840-len(cpu_reuse|accepted);old=manifest.get('original_estimate',{})
    def measured_rate(seconds,count):
        return old[seconds]/old[count] if old.get(seconds) is not None and old.get(count,0)>0 else None
    region_rate=measured_rate('region_seconds','remaining_unique_region_images')
    feature_rate=measured_rate('missing_feature_seconds','missing_feature_images')
    distance_times=getattr(a,'_final_distance_times',None)
    if distance_times is None:
        distance_times=[]
        for ref in manifest['problems']:
            meta=gpu.st.sealed_json(a.problem_root/'problems'/ref['id']/'manifest.json')['metadata']
            if meta.get('distance_seconds') is not None:distance_times.append(meta['distance_seconds'])
        a._final_distance_times=distance_times
    distance_rate=float(np.mean(distance_times)) if distance_times else None
    def part(count,throughput):return 0. if count==0 else None if throughput is None else count*throughput
    parts=dict(single_LP_build_transfer_solve_audit_commit=part(pending,warm_rate),
        next_process_backend_initialization=next_process_init,
        region_preparation=part(inventory['remaining_regions'],region_rate),
        missing_feature_forward=part(inventory['missing_feature_images'],feature_rate),
        missing_distance_compute_commit=part(inventory['missing_distance_matrices'],distance_rate))
    known=sum(v for v in parts.values() if v is not None)
    result=dict(backend=backend,parts_seconds=parts,known_pending_seconds=known,
        exceeds_12h=known>12*3600,unknown_stages=[k for k,v in parts.items() if v is None]+['final_image_evaluation_and_IO'],
        remaining_LPs=pending,strict_CPU_reused_indices=sorted(cpu_reuse),verified_GPU_indices=sorted(accepted),
        LP_seconds_per_case_including_failures_and_initialization=rate,measured_case_count=len(observed),inventory=inventory,
        LP_seconds_per_case_excluding_explicit_initialization=warm_rate,
        numerical_sample_all_passed=bool(observed) and all(r['audit']['accepted'] for r in observed),
        historical_preparation_estimate=old,region_seconds_per_image=region_rate,feature_seconds_per_image=feature_rate,
        distance_seconds_per_matrix=distance_rate,
        note='No RMS rebuild; existing regions/distances excluded by scoped inventory. All LP cases, including failures/timeouts, enter the mean; explicitly timed object initialization is projected once, not once per LP. Hidden first-Solve CUDA setup remains in the conservative mean. Prep rates are historical real-asset extrapolations, not new feature work. Failed samples are not reliable completion throughput. Unknown is not zero.')
    return gpu.finite_record(result)


def final_report(a,records,status,backend,costs,manifest):
    rows=[]
    for r in records:
        check=r['audit'];checks=check.get('checks',{});request=r['request']
        rows.append(dict(id=request['id'],manifest_index=request['index'],requested_backend=request['requested_backend'],
            solve_invoked=r.get('solve_invoked'),
            actual_backend=check.get('backend'),stop_reason=check.get('stop_reason'),
            primal_feasible=all(checks.get(k,False) for k in ('row','capacity','nonnegative','objective','free_lower_bound')),
            dual_feasible=checks.get('dual',False),normalized_gap=check.get('normalized_gap'),
            CPU_reference_abs_error=check.get('reference_abs_error'),accepted=check['accepted'],
            iterations=r.get('solver',{}).get('convergence',{}).get('nb_iterations'),
            process_peak_RSS_bytes=r.get('resource',{}).get('peak_RSS_bytes'),
            GPU_residency_bytes_after_call=r.get('resource',{}).get('GPU_memory_bytes_after_batch'),
            complete_seconds=r['end_to_end_seconds'],**r.get('timing',{}),
            **{k:check.get(k) for k in ('row_sum_max_error','capacity_max_relative_error','objective_consistency_error','min_W','negative_mass',
                'dual_reduced_cost_violation','dual_inequality_sign_violation')},error=r.get('error')))
    gpu.st.write_csv(a.output_root/'final_checks.csv',rows)
    gpu.st.write_csv(a.output_root/'stage1_four_cases.csv',rows[:4])
    gpu.st.write_csv(a.output_root/'remaining14.csv',rows[4:])
    gpu.st.write_csv(a.output_root/'cost_estimate.csv',[dict(backend=c['backend'],component=k,seconds=value,
        remaining_LPs=c.get('remaining_LPs'),unknown=value is None) for c in costs for k,value in c.get('parts_seconds',{}).items()])
    complete=backend is not None and sum(r['request']['requested_backend']==backend and r['audit']['accepted'] for r in records)==16
    text='当前求解路径未能在可接受成本下可靠完成，尚未取得正式检测结果。'
    conclusion=(('16题精度核查通过，但已知成本超过12小时；暂停等待用户决定，未启动正式实验。' if any(c['exceeds_12h'] for c in costs if c['backend']==backend) else
                 '16题精度核查通过；成本仍有未知项，先汇报并等待决定，未启动正式实验。' if any(c['unknown_stages'] for c in costs if c['backend']==backend)
                 else '16题精度与成本核查通过，可讨论继续正式单域实验；未自动启动。') if complete else text)
    summary=dict(status=status,selected_backend=backend,complete_16=complete,attempt_slots=len(records),maximum_calls=18,
        remaining14_status='not_requested' if len(records)==4 else 'checked_once_no_retries',
        recorded_Solve_calls=sum(r.get('solve_invoked') is True for r in records),
        interrupted_call_slots=sum(r['audit'].get('stop_reason')=='interrupted' for r in records),
        explicit_over_12h_extension_consent=bool(getattr(a,'allow_over_12h',False)),
        accepted_cases=sum(r['audit']['accepted'] for r in records),formal_LPs_started=0,feature_forwards=0,
        selected_backend_accepted_cases=sum(r['request']['requested_backend']==backend and r['audit']['accepted'] for r in records),
        input_manifest=manifest['seal'],costs=costs,conclusion=conclusion,process_import_seconds=getattr(a,'process_import_seconds',None),
        interpreter_startup_seconds=None,shared_input_setup_seconds=getattr(a,'_final_setup_seconds',None),
        current_invocation_seconds=time.perf_counter()-getattr(a,'_final_started',time.perf_counter()),
        recorded_case_seconds=sum(r['end_to_end_seconds'] for r in records if r['end_to_end_seconds'] is not None),
        process_peak_RSS_bytes=max((r.get('resource',{}).get('peak_RSS_bytes') or 0 for r in records),default=0) or None,
        max_sampled_GPU_residency_bytes=max((r.get('resource',{}).get('GPU_memory_bytes_after_batch') or 0 for r in records),default=0) or None,
        stage1_backend_complete_seconds={b:sum(r['end_to_end_seconds'] for r in records[:4] if r['request']['requested_backend']==b and r['end_to_end_seconds'] is not None) for b in ('PDLP','Barrier')},
        timing_note='Fixed order 00/PDLP,00/Barrier,08/PDLP,08/Barrier. PDLP pays initial shared imports/context; per-backend first-call timing is retained. No extra warmup. Transfer/API overhead is combined, GPU residency is sampled after calls, not a transient peak.')
    gpu.st.write_json(a.output_root/'final_summary.json',summary)
    lines=['# 最后一轮单题LP核查','',f'状态：{status}；选定后端：{backend or "无"}。',
        '', '|编号|实际后端|终止状态|原始可行|对偶可行|归一化间隙|CPU目标差|通过|完整秒数|',
        '|---|---|---|---|---|---:|---:|---|---:|']
    for r in rows:lines.append('|'+ '|'.join(str(r[k]) if r[k] is not None else 'NA' for k in
        ('id','actual_backend','stop_reason','primal_feasible','dual_feasible','normalized_gap','CPU_reference_abs_error','accepted','complete_seconds'))+'|')
    lines+=['',conclusion,'','## 成本（包含失败、超时和初始化）','']
    for c in costs:
        lines.append(f"- {c['backend']}：数值样本均通过 {c.get('numerical_sample_all_passed')}；已知待执行 {c.get('known_pending_seconds')} 秒；未知项 {c.get('unknown_stages')}。")
        lines.append(f"  剩余LP {c.get('remaining_LPs')}；完整单题均值 {c.get('LP_seconds_per_case_including_failures_and_initialization')} 秒；"
            f"原CPU复用编号 {c.get('strict_CPU_reused_indices')}；资产数量、历史实测率及分项公式见final_summary.json/cost_estimate.csv。")
    lines+=['','未执行检测评价；不能写成AUROC/AP失败，不能据此否定容量约束思想或第二章研究任务。',
        '不追加其他后端、模式、精度、缩放或求解器挽救试验。成本明细见final_summary.json；未测项不是0。']
    (a.output_root/'objective_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (a.output_root/'section51_append.md').write_text('第51节可追加：本轮固定核查原16项中的第1/9题，比较单题mixed PDLP与GPU Barrier，最多18次调用。'
        f'当前状态为{status}，实际记录{len(records)}项，选定后端为{backend or "无"}。'+conclusion+
        '没有启动2840项正式LP、模型前向或检测指标评价；CPU测试与真实GPU记录分别报告。\n',encoding='utf-8')
    return summary


def final_check(a,backend_factory=gpu.CuOptPDLP,resource_reader=None):
    """Last fixed 4+14 sequential check; no formal pipeline is reachable."""
    import os
    from pathlib import Path
    a._final_started=time.perf_counter()
    if resource_reader is None:
        from .lp_worker import memory
        resource_reader=memory
    manifest,problems=gpu.load_problems(a)
    a._final_inventory=gpu.st.read_json(a.inventory_run/'inventory.json')
    ids={p['id']:i for i,p in enumerate(problems)}
    selected=[ids['00'],ids['08']] if '00' in ids and '08' in ids else [0,8]
    binding=dict(version='capacity_last_single_check_v1',problems=manifest['seal'],selected_indices=selected,
        coordinate_system='original_all_constraints',settings={b:final_settings(b,a.cpu_workers) for b in ('PDLP','Barrier')},
        versions=gpu.packages(),visible_device=os.environ.get('CUDA_VISIBLE_DEVICES'),maximum_calls=18,
        coverage_parent=str(a.coverage_parent),inventory=gpu.st.digest(a._final_inventory),
        implementation={p.name:gpu.st.sha(p) for p in (Path(__file__),Path(gpu.__file__))})
    frozen=gpu.st.sealed_json(a.output_root/'final_inputs.json',binding);a._final_identity=frozen['seal']
    # Validate the reused inventory and CPU provenance before any GPU call.
    final_cost(a,manifest,problems,[],'PDLP')
    a._final_setup_seconds=time.perf_counter()-a._final_started
    records=[];adapters={}
    for index in selected:
        for backend in ('PDLP','Barrier'):
            records.append(final_single(a,index,problems[index],backend,adapters,backend_factory,resource_reader))
    eligible=[b for b in ('PDLP','Barrier') if sum(r['request']['requested_backend']==b and r['audit']['accepted'] and r['end_to_end_seconds'] is not None for r in records)==2]
    costs=[final_cost(a,manifest,problems,records,b) for b in ('PDLP','Barrier')]
    if not eligible:return final_report(a,records,'closed_unreliable',None,costs,manifest)
    backend=min(eligible,key=lambda b:sum(r['end_to_end_seconds'] for r in records if r['request']['requested_backend']==b))
    cost=next(c for c in costs if c['backend']==backend)
    if cost['exceeds_12h'] and not getattr(a,'allow_over_12h',False):
        return final_report(a,records,'stopped_cost_over_12h',backend,costs,manifest)
    final_report(a,records,'stage1_passed_extension_permitted',backend,costs,manifest)
    for index in range(16):
        if index not in selected:
            records.append(final_single(a,index,problems[index],backend,adapters,backend_factory,resource_reader))
    costs=[final_cost(a,manifest,problems,records,backend)]
    passed=sum(r['request']['requested_backend']==backend and r['audit']['accepted'] for r in records)==16
    status='sixteen_verified_awaiting_decision' if passed else 'closed_unreliable'
    if passed and costs[0]['exceeds_12h']:status='sixteen_verified_cost_over_12h'
    return final_report(a,records,status,backend,costs,manifest)


def main(argv=None):
    import argparse,json,os
    from pathlib import Path
    p=argparse.ArgumentParser(description='Existing 16-LP checks, including the last fixed 4+14 single PDLP/Barrier check; no formal experiment')
    p.add_argument('--problem-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--cpu-workers',type=int,choices=range(1,8),default=7)
    p.add_argument('--coordinate-system',choices=('mass_flow_v1','mass_flow_basis_v1'),default='mass_flow_v1')
    p.add_argument('--final-check',action='store_true',help='last fixed 4+14 single-problem PDLP/Barrier check; no formal experiment')
    p.add_argument('--inventory-run',type=Path)
    p.add_argument('--coverage-parent',type=Path)
    p.add_argument('--allow-over-12h',action='store_true',help='only after explicit consent to the reported cost; never starts the formal experiment')
    a=p.parse_args(argv);a.problem_root=a.problem_root.resolve();a.output_root=a.output_root.resolve()
    if a.problem_root==a.output_root or a.problem_root in a.output_root.parents or a.output_root in a.problem_root.parents:
        raise ValueError('diagnostic output overlaps the read-only input export')
    if a.output_root.exists() and not a.final_check:raise ValueError('select a new diagnostic output; previous trials are read-only')
    device=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if not (device.isdigit() or device.startswith('GPU-')) or ',' in device:raise ValueError('one explicitly selected physical GPU required')
    if a.final_check:
        if a.coordinate_system!='mass_flow_v1':raise ValueError('final check always uses original coordinates; do not request a coordinate trial')
        if a.inventory_run is None or a.coverage_parent is None:raise ValueError('final check requires the existing scoped --inventory-run and --coverage-parent')
        a.inventory_run=a.inventory_run.resolve();a.coverage_parent=a.coverage_parent.resolve()
        for root in (a.inventory_run,a.coverage_parent):
            if a.output_root==root or root in a.output_root.parents or a.output_root in root.parents:raise ValueError('final output overlaps historical inputs/inventory')
        if a.output_root.exists() and any(a.output_root.iterdir()) and not (a.output_root/'final_inputs.json').exists():
            raise ValueError('select a new final-check directory; unrelated historical outputs are read-only')
        a.process_import_seconds=time.perf_counter()-_IMPORT_STARTED
        result=final_check(a)
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
        return 0 if result['complete_16'] else 2
    result=mass_flow_trial(a);print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return 0 if result['accepted']==16 else 2


if __name__=='__main__':raise SystemExit(main())
