"""Sparse minimax transport LP; never a total-cost or Sinkhorn substitute."""
import warnings
import numpy as np
from scipy.optimize import linprog,OptimizeWarning
from scipy.sparse import coo_matrix
from .protocol import CONFIG


def validate_inputs(distance,a,b):
    d=np.asarray(distance,np.float64);a=np.asarray(a,np.float64);b=np.asarray(b,np.float64)
    if d.shape!=(len(a),len(b)) or d.ndim!=2 or not all(np.isfinite(x).all() for x in (d,a,b)) or np.any(d<0) or min(a.min(),b.min())<=0:
        raise ValueError('nonnegative finite distances and positive area masses required')
    if max(abs(a.sum()-1),abs(b.sum()-1))>CONFIG['mass_sum_tolerance']:raise ValueError('each image area mass must sum to one')
    return d,a,b


def free_cost(distance):return float(np.max(np.min(distance,axis=1)))


def constraints(d,a,b):
    n,m=d.shape;k=n*m;idx=np.arange(k);rows=np.repeat(np.arange(n),m);cols=np.tile(np.arange(m),n)
    eq=coo_matrix((np.r_[np.ones(k),np.repeat(a,m)],(np.r_[rows,n+cols],np.r_[idx,idx])),shape=(n+m,k+1)).tocsr()
    ub=coo_matrix((np.r_[d.ravel(),-np.ones(n)],(np.r_[rows,np.arange(n)],np.r_[idx,np.full(n,k)])),shape=(n,k+1)).tocsr()
    return eq,np.r_[np.ones(n),b],ub,np.zeros(n)


def feasibility(d,a,b,W,t):
    if W.shape!=d.shape or not np.isfinite(W).all() or not np.isfinite(t):raise ValueError('nonfinite or wrong LP solution')
    costs=np.sum(W*d,axis=1);capacity=a@W
    result=dict(row_sum_max_error=float(np.max(abs(W.sum(1)-1))),
        capacity_max_relative_error=float(np.max(abs(capacity-b)/b)),
        min_W=float(W.min()),objective_max_violation=float(max(0.,costs.max()-t)),
        objective_consistency_error=float(abs(costs.max()-t)),free_lower_bound_gap=float(t-free_cost(d)))
    ok=(result['row_sum_max_error']<=CONFIG['row_relative_tolerance'] and
        result['capacity_max_relative_error']<=CONFIG['capacity_relative_tolerance'] and
        result['min_W']>=-CONFIG['negative_W_tolerance'] and
        result['objective_consistency_error']<=CONFIG['objective_tolerance']*max(1.,abs(t)) and
        result['free_lower_bound_gap']>=-CONFIG['objective_tolerance']*max(1.,abs(t)))
    return result,bool(ok)


def solve(distance,a,b,save_plan=False):
    import time
    tick=time.perf_counter();d,a,b=validate_inputs(distance,a,b);eq,rhs,ub,upper=constraints(d,a,b)
    c=np.zeros(d.size+1);c[-1]=1
    # threads/parallel are passed through to HiGHS, not a machine-wide setting.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',category=OptimizeWarning,message='Unrecognized options detected.*')
        r=linprog(c,A_ub=ub,b_ub=upper,A_eq=eq,b_eq=rhs,bounds=(0,None),method='highs-ds',
            options=dict(primal_feasibility_tolerance=CONFIG['solver_primal_tolerance'],
                dual_feasibility_tolerance=CONFIG['solver_dual_tolerance'],threads=1,parallel=False))
    result=dict(success=bool(r.success),solver_status=int(r.status),message=str(r.message),
        iterations=int(r.nit),seconds=time.perf_counter()-tick,constraint_nnz=int(eq.nnz+ub.nnz),
        objective=None,free=free_cost(d),feasibility=None)
    if r.success:
        W=r.x[:-1].reshape(d.shape);checks,ok=feasibility(d,a,b,W,float(r.fun))
        result.update(success=ok,objective=float(r.fun),feasibility=checks)
        if not ok:result['message']='solver success but normalized feasibility/objective check failed'
        if save_plan:result['W']=W.tolist()
    return result
