"""Independent mixed-PDLP protocol. All checks use the original, unrepaired LP."""
import time
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


def audit(problem,primal,dual,status,backend='PDLP'):
    """A per-problem dual certificate, never an eight-problem aggregate gap."""
    d,a,b=transport.validate_inputs(problem['distance'],problem['a'],problem['b'])
    x=np.asarray(primal,np.float64);y=np.asarray(dual,np.float64);n,m=d.shape
    result=dict(accepted=False,acceptance='incomplete',stop_reason=str(status),backend=backend,
        protocol=PROTOCOL['version'],repaired=False)
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
    accepted=bool(category and backend=='PDLP' and all(checks.values()))
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


def main(argv=None):
    import argparse,json,os
    from pathlib import Path
    p=argparse.ArgumentParser(description='Fixed mixed-PDLP mass-flow coordinate trial on the existing 16 LPs only')
    p.add_argument('--problem-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--cpu-workers',type=int,choices=range(1,8),default=7)
    p.add_argument('--coordinate-system',choices=('mass_flow_v1','mass_flow_basis_v1'),default='mass_flow_v1')
    a=p.parse_args(argv);a.problem_root=a.problem_root.resolve();a.output_root=a.output_root.resolve()
    if a.problem_root==a.output_root or a.problem_root in a.output_root.parents or a.output_root in a.problem_root.parents:
        raise ValueError('diagnostic output overlaps the read-only input export')
    if a.output_root.exists():raise ValueError('select a new diagnostic output; previous trials are read-only')
    device=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if not (device.isdigit() or device.startswith('GPU-')) or ',' in device:raise ValueError('one explicitly selected physical GPU required')
    result=mass_flow_trial(a);print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return 0 if result['accepted']==16 else 2


if __name__=='__main__':raise SystemExit(main())
