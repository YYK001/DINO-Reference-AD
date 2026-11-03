"""Explicit cuOpt precision trial on the 16 committed CPU-smoke LPs, never a formal run.

Export runs in the original yyk environment; benchmark runs in an isolated cuOpt
environment. No model, raw features, labels or new Source RMS are needed.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from importlib.metadata import version,PackageNotFoundError
import numpy as np
from scipy.sparse import block_diag,vstack
from . import inputs as ip,transport,regions
from .protocol import CONFIG
st=ip.st
SCHEMA='reverse_capacity_GPU_precision_trial_v1'


def settings_values(time_limit,cpu_workers,precision='fp32'):
    if precision not in ('fp32','mixed'):raise ValueError('explicit fp32 or mixed precision required')
    return dict(method='PDLP',pdlp_precision={'fp32':0,'mixed':2}[precision],pdlp_solver_mode='Stable3',num_gpus=1,crossover=False,
        num_cpu_threads=cpu_workers,time_limit=time_limit,
        absolute_primal_tolerance=CONFIG['solver_primal_tolerance'],relative_primal_tolerance=CONFIG['solver_primal_tolerance'],
        absolute_dual_tolerance=CONFIG['solver_dual_tolerance'],relative_dual_tolerance=CONFIG['solver_dual_tolerance'],
        absolute_gap_tolerance=CONFIG['solver_dual_tolerance'],relative_gap_tolerance=CONFIG['solver_dual_tolerance'])


def sparse_batch(problems):
    matrices=[];rhs=[];senses=[];objectives=[];slices=[];offset=0
    for problem in problems:
        d,a,b=transport.validate_inputs(problem['distance'],problem['a'],problem['b'])
        eq,e,ub,u=transport.constraints(d,a,b);matrices.append(vstack((eq,ub),format='csr'))
        rhs.append(np.r_[e,u]);senses.extend(['E']*len(e)+['L']*len(u))
        objective=np.zeros(d.size+1);objective[-1]=1.;objectives.append(objective)
        slices.append((offset,offset+d.size+1));offset+=d.size+1
    return dict(matrix=block_diag(matrices,format='csr'),rhs=np.concatenate(rhs),
        senses=np.array(senses),objective=np.concatenate(objectives),slices=slices)


class CuOptPDLP:
    """Stock sparse cuOpt API. Explicit precision, no implicit crossover/fallback."""
    def __init__(self,time_limit=30.,cpu_workers=8,precision='fp32'):
        # Official mode 0 uses FP32 iterations; mode 2 uses FP32 SpMV matrices
        # with FP64 vectors and convergence checks. Original inputs stay FP64.
        from cuopt import linear_programming as lp
        from cuopt.linear_programming.solver_settings.solver_settings import SolverMethod,PDLPSolverMode
        from cuopt.linear_programming.solver import solver_parameters as keys
        self.lp=lp;self.method=SolverMethod.PDLP;self.settings=lp.SolverSettings()
        self.parameters=settings_values(time_limit,cpu_workers,precision)
        for name,value in self.parameters.items():
            value=self.method if name=='method' else PDLPSolverMode.Stable3 if name=='pdlp_solver_mode' else value
            self.settings.set_parameter(getattr(keys,'CUOPT_'+name.upper()),value)

    def solve(self,batch):
        tick=time.perf_counter();model=self.lp.DataModel();matrix=batch['matrix']
        # Preserve original coefficients/area masses; do NOT quantize the input
        # problem to disguise residuals. cuOpt handles the requested internal precision.
        model.set_csr_constraint_matrix(matrix.data.astype(np.float64,copy=False),
            matrix.indices.astype(np.int32,copy=False),matrix.indptr.astype(np.int32,copy=False))
        model.set_constraint_bounds(batch['rhs']);model.set_row_types(batch['senses'])
        model.set_objective_coefficients(batch['objective'])
        model.set_variable_lower_bounds(np.zeros(matrix.shape[1],np.float64))
        model.set_variable_upper_bounds(np.full(matrix.shape[1],np.inf,np.float64))
        build=time.perf_counter()-tick;tick=time.perf_counter()
        solution=self.lp.Solve(model,self.settings)
        solved_by=solution.get_solved_by();status=solution.get_termination_reason()
        return dict(primal=np.asarray(solution.get_primal_solution(),np.float64),
            objective=float(solution.get_primal_objective()),status=status,
            solver_optimal=status.lower()=='optimal',GPU_PDLP=solved_by==self.method,
            solved_by=getattr(solved_by,'name',str(solved_by)),model_seconds=build,
            solve_call_seconds=time.perf_counter()-tick,engine_seconds=float(solution.get_solve_time()),
            convergence={k:float(v) for k,v in solution.get_lp_stats().items()})


def check_solution(problems,batch,solution):
    vector=np.asarray(solution['primal'],np.float64);out=[]
    if vector.shape!=(batch['matrix'].shape[1],) or not np.isfinite(vector).all():
        return [dict(problem_id=p['id'],passed=False,reason='missing/nonfinite/wrong-shape primal') for p in problems]
    reported=solution['objective'];total=float(batch['objective']@vector)
    aggregate_ok=bool(np.isfinite(reported) and abs(reported-total)<=CONFIG['objective_tolerance']*max(1.,abs(total)))
    for problem,(start,end) in zip(problems,batch['slices']):
        d,a,b=transport.validate_inputs(problem['distance'],problem['a'],problem['b'])
        W=vector[start:end-1].reshape(d.shape);t=float(vector[end-1])
        checks,feasible=transport.feasibility(d,a,b,W,t)
        error=abs(t-problem['reference_objective']);tol=CONFIG['objective_tolerance']*max(1.,abs(problem['reference_objective']))
        out.append(dict(problem_id=problem['id'],objective=t,reference_objective=problem['reference_objective'],
            reference_abs_error=error,reference_atol=tol,feasibility=checks,numerical_checks_passed=bool(feasible and error<=tol and aggregate_ok),
            passed=bool(feasible and error<=tol and aggregate_ok and solution['solver_optimal'] and solution['GPU_PDLP'])))
    return out


def source_case(a):
    saved=st.sealed_json(a.parent_run/'inputs.json');protocol=st.sealed_json(a.parent_run/'protocol.json')
    smoke=st.sealed_json(a.parent_run/'smoke.json');rms=st.sealed_json(a.parent_run/'source_RMS.json')
    if {k:v for k,v in protocol.items() if k!='seal'}!=CONFIG or smoke['identity']!=saved['identity']:
        raise ValueError('expected the original fixed capacity experiment and committed smoke')
    if not smoke['actual_features'] or not smoke['full_256_regions'] or smoke['LP_pairs']!=16 or smoke['RMS']!=rms['provider_seal']:
        raise ValueError('exact original 16 real full-size smoke LPs required')
    cache=(a.cache_root or Path(saved['routes']['cache_root'])).resolve()
    templates=[r['image'] for r in smoke['regions'][:8]];queries=[r['image'] for r in smoke['matching']]
    if len(templates)!=8 or len(queries)!=2:raise ValueError('two queries and eight whole templates required')
    cases=[]
    for query in queries:
        for template in templates:
            pair=a.parent_run/'pairs'/ip.key(query)/(ip.key(template)+'.json');r=st.sealed_json(pair)
            dep=r['dependencies']
            if (r['query']!=query or r['template']!=template or not r['result']['success']
                or not np.isfinite(r['result']['objective']) or dep['RMS']!=rms['provider_seal']
                or dep['distance_device']!=saved['backend']['distance_device'] or dep['protocol']!=st.digest(CONFIG)
                or dep['solver_implementation']!=st.sha(Path(transport.__file__))):
                raise ValueError('committed original CPU reference inputs differ')
            cases.append(dict(query=query,template=template,record=r,path=pair))
    return dict(inputs=saved,smoke=smoke,RMS=rms,cache=cache,cases=cases,
        distance_device=saved['backend']['distance_device'])


def region_input(context,row,expected):
    path=context['cache']/'regions'/ip.key(row);record=st.sealed_json(path/'manifest.json');dep=record['metadata']['dependencies']
    if (record['seal']!=expected or dep['image']!=row or dep['RMS']!=context['RMS']['provider_seal']
        or dep['extraction']!=st.digest(context['RMS']['contract']['extraction'])
        or dep['implementation']!=st.sha(Path(regions.__file__))):raise ValueError('region provenance changed')
    data,r=st.get(path,('means','area'))
    if data['means'].shape!=(4,256,1280) or data['means'].dtype!=np.float32 or data['area'].shape!=(256,):
        raise ValueError('original full-size regions required')
    return data


def export(a):
    context=source_case(a);problems=[];loaded={}
    # Reconstruction stays in the original environment, not the solver's new one.
    for name in ('numpy','scipy','torch'):
        wanted=context['inputs']['backend'].get(name)
        if wanted is not None and version(name)!=wanted:raise ValueError('export must use the original distance environment: '+name)
    binding=dict(schema=SCHEMA,parent_inputs=context['inputs']['seal'],parent_smoke=context['smoke']['seal'],
        RMS=context['RMS']['seal'],pairs=[c['record']['seal'] for c in context['cases']],
        regions_implementation=st.sha(Path(regions.__file__)))
    export_seal=st.sealed_json(a.output_root/'export_inputs.json',binding)['seal']
    def get(row,seal):
        if seal not in loaded:loaded[seal]=region_input(context,row,seal)
        return loaded[seal]
    for i,case in enumerate(context['cases']):
        old=case['record'];ident=f'{i:02d}';bundle=a.output_root/'problems'/ident
        meta=dict(schema=SCHEMA,export_seal=export_seal,original_pair_seal=old['seal'],original_pair=str(case['path']),
            query=case['query'],template=case['template'],dependencies=old['dependencies'],reference_objective=old['result']['objective'])
        if (bundle/'manifest.json').exists():
            _,r=st.get(bundle)
            if {k:v for k,v in r['metadata'].items() if k!='distance_seconds'}!=meta:raise ValueError('export inputs changed')
        else:
            q=get(case['query'],old['dependencies']['query']);t=get(case['template'],old['dependencies']['template'])
            tick=time.perf_counter();d=regions.distance_matrix(t['means'],q['means'],context['RMS']['RMS'],context['distance_device'])
            free=transport.free_cost(d)
            if abs(free-old['result']['free'])>CONFIG['objective_tolerance']*max(1.,abs(free)):
                raise ValueError('reconstructed distance path differs from original CPU reference problem')
            r=st.put(bundle,dict(distance=d,a=np.array(t['area']),b=np.array(q['area'])),
                dict(**meta,distance_seconds=time.perf_counter()-tick))
        problems.append(dict(id=ident,seal=r['seal'],reference_objective=old['result']['objective']))
        print(f'[GPU LP export] {i+1}/16 cached_regions_only',flush=True)
    estimate_path=a.parent_run/'cost_estimate.json'
    estimate=st.read_json(estimate_path) if estimate_path.exists() else {}
    manifest=st.sealed_json(a.output_root/'problems.json',dict(schema=SCHEMA,problems=problems,
        parent_inputs=context['inputs']['seal'],parent_smoke=context['smoke']['seal'],RMS=context['RMS']['seal'],
        protocol=CONFIG,export_seal=export_seal,distance_device=context['distance_device'],original_estimate=estimate,
        feature_forwards=0,LP_reference_recomputed=False,GT_used=False))
    return dict(status='16_original_LPs_exported',seal=manifest['seal'],new_features=0,new_Source_RMS=0)


def load_problems(a):
    root=getattr(a,'problem_root',None) or a.output_root
    manifest=st.sealed_json(root/'problems.json');out=[]
    if manifest['schema']!=SCHEMA or manifest['protocol']!=CONFIG:raise ValueError('benchmark input protocol differs')
    for ref in manifest['problems']:
        x,r=st.get(root/'problems'/ref['id'])
        if r['seal']!=ref['seal'] or r['metadata']['reference_objective']!=ref['reference_objective']:raise ValueError('exported LP commitment changed')
        transport.validate_inputs(x['distance'],x['a'],x['b'])
        if x['distance'].shape!=(256,256) or x['distance'].dtype!=np.float32:raise ValueError('real full-size FP32 distances required')
        out.append(dict(id=ref['id'],reference_objective=ref['reference_objective'],**x))
    if len(out)!=16 or len({p['id'] for p in out})!=16:raise ValueError('all 16 distinct real reference problems required')
    return manifest,out


def trial(a,backend_factory=CuOptPDLP):
    manifest,problems=load_problems(a)
    precision=getattr(a,'precision','fp32');tag=precision.upper()
    binding=dict(schema=SCHEMA,problems=manifest['seal'],settings=settings_values(a.time_limit,a.cpu_workers,precision),
        batch_size=a.batch_size,repeats=a.repeats,precision=precision,
        problem_root=str(getattr(a,'problem_root',None) or a.output_root),
        packages=packages(),visible_device=os.environ.get('CUDA_VISIBLE_DEVICES'),
        implementation=st.sha(Path(__file__)),checking_implementation=st.sha(Path(transport.__file__)))
    identity=st.sealed_json(a.output_root/'trial_inputs.json',binding)['seal']
    tick=time.perf_counter()
    try:backend=backend_factory(a.time_limit,a.cpu_workers,precision)
    except Exception as e:
        summary=dict(status='GPU_backend_unavailable',identity=identity,complete=False,passed=False,
            precision=precision,error=f'{type(e).__name__}: {e}',formal_backend_changed=False,effective_seconds_per_LP=None)
        st.write_json(a.output_root/'GPU_LP_summary.json',summary)
        return summary
    init=time.perf_counter()-tick
    groups=[problems[i:i+a.batch_size] for i in range(0,len(problems),a.batch_size)];results=[];started=time.perf_counter()
    # One persistent process/context. Warmup is explicit and excluded from steady throughput.
    for repeat in range(-1,a.repeats):
        for j,group in enumerate(groups if repeat>=0 else groups[:1]):
            if time.perf_counter()-started>a.max_seconds:break
            path=a.output_root/'trial_results'/f'repeat{repeat}_batch{j}.json'
            if path.exists():
                record=st.sealed_json(path)
                if record['identity']!=identity:raise ValueError('GPU trial configuration changed; new output root required')
            else:
                begin=time.perf_counter();batch=sparse_batch(group)
                try:
                    solved=backend.solve(batch);checks=check_solution(group,batch,solved)
                    details={k:v for k,v in solved.items() if k!='primal'}
                    value=finite_record(dict(checks=checks,solver=details,error=None))
                except Exception as e:value=dict(checks=[],solver=None,error=f'{type(e).__name__}: {e}')
                record=st.sealed_json(path,dict(identity=identity,repeat=repeat,batch=j,problem_ids=[p['id'] for p in group],
                    end_to_end_seconds=time.perf_counter()-begin,**value))
            results.append(record);print(f'[GPU LP {tag}] repeat={repeat} batch={j} passed={sum(r["passed"] for r in record["checks"])}/{len(group)} seconds={record["end_to_end_seconds"]:.3f}',flush=True)
        if time.perf_counter()-started>a.max_seconds:break
    measured=[r for r in results if r['repeat']>=0];checks=[c for r in measured for c in r['checks']]
    complete=len(measured)==len(groups)*a.repeats and len(checks)==16*a.repeats
    passed=complete and all(c['passed'] for c in checks)
    seconds=sum(r['end_to_end_seconds'] for r in measured);rate=seconds/len(checks) if passed else None
    remaining=manifest['original_estimate'].get('remaining_LP_pairs')
    maxima={}
    for name in ('row_sum_max_error','capacity_max_relative_error','objective_consistency_error','reference_abs_error'):
        values=[c[name] if name in c else c.get('feasibility',{}).get(name) for c in checks]
        values=[v for v in values if v is not None]
        maxima[name]=max(values) if values else None
    summary=dict(status=f'{tag}_precision_trial_passed' if passed else f'{tag}_precision_trial_incomplete_or_failed',
        identity=identity,precision=precision,complete=complete,passed=passed,
        successful_checks=sum(c['passed'] for c in checks),numerical_checks_passed=sum(c.get('numerical_checks_passed',False) for c in checks),
        requested_checks=16*a.repeats,max_errors=maxima,
        backend_initialization_seconds=init,warmup_seconds=sum(r['end_to_end_seconds'] for r in results if r['repeat']<0),
        measured_seconds=seconds,effective_seconds_per_LP=rate,remaining_LP_seconds=None if rate is None or remaining is None else rate*remaining,
        whole_experiment_hours=None,other_work='Source RMS already retained; feature warm throughput and region preparation must be measured separately',
        timing_includes='sparse block construction, model transfer, solve, original-coefficient checks',
        timing_excludes='one-off export, backend initialization, warmup, checkpoint IO, whole-run feature/region work',
        formal_backend_changed=False,scope='background/seed42 original two-image smoke only; no all-domain accuracy or speed guarantee')
    st.write_json(a.output_root/'GPU_LP_summary.json',summary)
    st.write_csv(a.output_root/'GPU_LP_checks.csv',[dict(repeat=r['repeat'],batch=r['batch'],**{k:v for k,v in c.items() if k!='feasibility'},**c.get('feasibility',{})) for r in measured for c in r['checks']])
    return summary


def finite_record(value):
    if isinstance(value,dict):return {k:finite_record(v) for k,v in value.items()}
    if isinstance(value,list):return [finite_record(v) for v in value]
    return None if isinstance(value,float) and not np.isfinite(value) else value


def packages():
    out={}
    for name in ('numpy','scipy','cuopt','cuopt-cu13','cuopt-cu12','libcuopt-cu13'):
        try:out[name]=version(name)
        except PackageNotFoundError:out[name]=None
    return out


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',nargs='?',default='preflight',choices=('preflight','export','benchmark'))
    p.add_argument('--parent-run',type=Path);p.add_argument('--cache-root',type=Path)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--problem-root',type=Path,help='read-only existing 16-problem export; benchmark only')
    p.add_argument('--precision',choices=('fp32','mixed'),default='fp32')
    p.add_argument('--batch-size',type=int,choices=(1,2,4,8,16),default=8)
    p.add_argument('--repeats',type=int,choices=(1,2,3),default=2)
    p.add_argument('--cpu-workers',type=int,choices=range(1,9),default=8)
    p.add_argument('--time-limit',type=float,default=30.)
    p.add_argument('--max-seconds',type=float,default=1200.)
    a=p.parse_args(argv);a.output_root=a.output_root.resolve()
    if a.problem_root:
        a.problem_root=a.problem_root.resolve()
        if a.stage!='benchmark':raise ValueError('--problem-root is only used by benchmark')
        if a.problem_root==a.output_root or a.problem_root in a.output_root.parents or a.output_root in a.problem_root.parents:
            raise ValueError('new precision output must be separate from the read-only problem provider')
    if a.stage=='benchmark':
        device=os.environ.get('CUDA_VISIBLE_DEVICES','')
        if not (device.isdigit() or device.startswith('GPU-')) or ',' in device:
            raise ValueError('set CUDA_VISIBLE_DEVICES to the one explicitly selected physical GPU')
    if a.parent_run:a.parent_run=a.parent_run.resolve()
    if a.cache_root:a.cache_root=a.cache_root.resolve()
    if not 0<a.time_limit<=120 or not 0<a.max_seconds<=3600:raise ValueError('bounded precision trial only; not a formal multi-day job')
    if a.parent_run and (a.output_root==a.parent_run or a.parent_run in a.output_root.parents or a.output_root in a.parent_run.parents):
        raise ValueError('benchmark output must be separate from the original run')
    if a.stage in ('preflight','export'):
        if a.parent_run is None:raise ValueError('--parent-run is required to locate the original smoke')
        context=source_case(a)
        if a.output_root==context['cache'] or context['cache'] in a.output_root.parents or a.output_root in context['cache'].parents:
            raise ValueError('benchmark output must be separate from the original cache')
    a.output_root.mkdir(parents=True,exist_ok=True)
    if a.stage=='preflight':
        result=dict(status='original_16_reference_records_found',cases=len(context['cases']),cache=str(context['cache']),
            distance_device=context['distance_device'],packages=packages(),Torch_imported='torch' in sys.modules,
            CUDA_initialized=False,model_or_feature_work=False,solver_required_only_for_benchmark=True)
        st.write_json(a.output_root/'GPU_LP_preflight.json',result)
    elif a.stage=='export':result=export(a)
    else:result=trial(a)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return 2 if result.get('passed') is False else 0


if __name__=='__main__':raise SystemExit(main())
