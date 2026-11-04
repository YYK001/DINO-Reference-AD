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
    def __init__(self,cpu_workers=7):
        super().__init__(30.,cpu_workers,'mixed')
        from cuopt.linear_programming.solver import solver_parameters as keys
        self.parameters=settings(cpu_workers)
        for name,value in self.parameters.items():
            if name in ('method','pdlp_solver_mode'):continue
            self.settings.set_parameter(getattr(keys,'CUOPT_'+name.upper()),value)

    def solve_certified(self,problems):
        # Reuse the exact same DataModel construction and solution extraction;
        # add the dual vector through a small extension in the shared adapter.
        started=time.perf_counter();batch=gpu.sparse_batch(problems);solution=self.solve(batch)
        output=[];row=0
        dual=solution.get('dual',np.array([]));primal=solution['primal']
        for problem,(start,end) in zip(problems,batch['slices']):
            n,m=problem['distance'].shape;count=2*n+m
            x=primal[start:end];y=dual[row:row+count];row+=count
            certificate=audit(problem,x,y,solution['status'],solution['solved_by'])
            if solution.get('error_status')!=0:
                certificate.update(accepted=False,acceptance='incomplete',reason='solver error status missing or nonzero')
            output.append(dict(primal=x,dual=y,audit=certificate))
        return output,dict(seconds=time.perf_counter()-started,solver={k:v for k,v in solution.items() if k not in ('primal','dual')})
