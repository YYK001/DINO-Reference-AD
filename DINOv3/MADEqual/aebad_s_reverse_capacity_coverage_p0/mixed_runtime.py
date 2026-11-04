"""One isolated persistent GPU worker, reusable distances, item-level commits."""
import os
import sys
import time
import json
import subprocess
from pathlib import Path
import numpy as np
from . import inputs as ip,runtime as rt,transport,regions,gpu_lp,lp_v2
from .protocol import CONFIG
st=ip.st


class Worker:
    def __init__(self,a):self.a=a;self.process=None;self.log=None
    def __enter__(self):return self
    def __exit__(self,*_):
        if self.process:
            self.process.stdin.close()
            try:self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:self.process.terminate();self.process.wait(timeout=10)
        if self.log:self.log.close()
    def solve(self,items):
        a=self.a;token=str(time.time_ns());root=a.run_root/'solver_jobs';root.mkdir(parents=True,exist_ok=True)
        path=root/(token+'.json');response=root/(token+'_response.json')
        job=st.sealed_json(path,dict(items=items,response=str(response),cpu_workers=max(1,a.cpu_workers-1),protocol=lp_v2.PROTOCOL))
        started=time.perf_counter()
        if self.process is None:
            log=a.run_root/'logs/LP_worker.log';log.parent.mkdir(parents=True,exist_ok=True);self.log=log.open('a',encoding='utf-8')
            env=dict(os.environ,OMP_NUM_THREADS=str(max(1,a.cpu_workers-1)),MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            self.process=subprocess.Popen([str(a.lp_python),'-B','-u','-m',__package__+'.lp_worker'],
                stdin=subprocess.PIPE,stdout=self.log,stderr=subprocess.STDOUT,text=True,env=env)
        self.process.stdin.write(json.dumps(dict(job=str(path)))+'\n');self.process.stdin.flush()
        next_heartbeat=time.perf_counter()+30
        while not response.exists():
            if self.process.poll() is not None:raise RuntimeError('GPU worker exited; see '+str(a.run_root/'logs/LP_worker.log'))
            if time.perf_counter()>next_heartbeat:
                print('[heartbeat] mixed LP solving '+token,flush=True);next_heartbeat+=30
            time.sleep(.1)
        result=st.sealed_json(response)
        if result['job']!=job['seal']:raise ValueError('worker response belongs to another job')
        elapsed=time.perf_counter()-started
        st.write_json(root/(token+'_cost.json'),dict(end_to_end_seconds=elapsed,pairs=len(items),
            includes='queue IO, worker setup/reads/build/transfer/solve/audit/proof writes and response',status=result['status']))
        if result['error']:raise RuntimeError(result['error'])
        return result,elapsed


def exports(a):
    if hasattr(a,'_LP_exports'):return a._LP_exports
    result={}
    for root in getattr(a,'distance_exports',[]):
        manifest=st.sealed_json(Path(root)/'problems.json')
        if manifest['protocol']!=CONFIG:raise ValueError('exported distance definition differs')
        for ref in manifest['problems']:
            path=Path(root)/'problems'/ref['id'];record=st.sealed_json(path/'manifest.json');meta=record['metadata']
            if record['seal']!=ref['seal']:raise ValueError('export manifest changed')
            result[ip.key(meta['query']),ip.key(meta['template'])]=dict(path=path,record=record,
                reference_objective=ref['reference_objective'])
    a._LP_exports=result;return result


def distance(a,rms,query,template,qdata,qmeta,tdata,tmeta):
    dep=dict(query=qmeta['seal'],template=tmeta['seal'],RMS=rms['seal'],distance_device=a.device,
        definition=CONFIG['distance'],implementation=st.sha(Path(regions.__file__)))
    ref=exports(a).get((ip.key(query),ip.key(template)))
    if ref:
        old=ref['record']['metadata']['dependencies']
        if any(old[k]!=dep[k] for k in ('query','template','RMS','distance_device')):raise ValueError('distance export region/RMS/backend differs')
        return dict(bundle=str(ref['path']),bundle_seal=ref['record']['seal'],dependencies=dep,
            reference_objective=ref['reference_objective'],distance_seconds=0.,reused_distance=True)
    path=a.cache_root/'distances'/ip.key(query)/ip.key(template)
    if (path/'manifest.json').exists():
        record=st.sealed_json(path/'manifest.json')
        if record['metadata']['dependencies']!=dep:raise ValueError('distance inputs changed')
        return dict(bundle=str(path),bundle_seal=record['seal'],dependencies=dep,reference_objective=None,distance_seconds=0.,reused_distance=True)
    started=time.perf_counter();d=regions.distance_matrix(tdata['means'],qdata['means'],rms['RMS'],a.device)
    record=st.put(path,dict(distance=d,a=np.array(tdata['area']),b=np.array(qdata['area'])),dict(dependencies=dep))
    return dict(bundle=str(path),bundle_seal=record['seal'],dependencies=dep,reference_objective=None,
        distance_seconds=time.perf_counter()-started,reused_distance=False)


def verify_proof(item,proof):
    arrays,record=st.get(proof)
    if record['metadata']['protocol']!=lp_v2.PROTOCOL or record['metadata']['dependencies']!=item['dependencies']:
        raise ValueError('LP proof has incompatible inputs or numerical protocol')
    settings=record['metadata'].get('settings',{})
    if any(settings.get(k)!=v for k,v in lp_v2.PROTOCOL['solver'].items()):raise ValueError('proof was not produced by the frozen mixed PDLP configuration')
    data,bundle=st.get(item['bundle'])
    if bundle['seal']!=item['bundle_seal']:raise ValueError('distance manifest differs')
    solver=record['metadata']['solver']
    result=lp_v2.audit(dict(**data,reference_objective=item.get('reference_objective')),arrays['primal'],arrays['dual'],
        solver['status'],solver['solved_by'])
    if solver.get('error_status')!=0:result.update(accepted=False,acceptance='incomplete',reason='solver error status missing or nonzero')
    committed=record['metadata']['audit']
    # Two isolated NumPy/SciPy environments need not reproduce residual sums bit
    # for bit. Recompute every threshold, and require the accepted state, each
    # predicate and the actual scalar used for scoring to remain identical.
    for key in ('accepted','acceptance','checks','objective','free','stop_reason','backend'):
        if result.get(key)!=committed.get(key):raise ValueError('LP evidence no longer reproduces the committed audit')
    return result,record


def strict_cpu(a,rms,query,template,dep):
    if not getattr(a,'resume_from',None):return None
    path=a.resume_from/'pairs'/ip.key(query)/(ip.key(template)+'.json')
    if not path.is_file():return None
    record=st.sealed_json(path);old=record['dependencies'];r=record['result'];f=r.get('feasibility') or {}
    if any(old[k]!=dep[k] for k in ('query','template','RMS','distance_device')):raise ValueError('old CPU pair inputs differ')
    if old['protocol']!=st.digest(CONFIG) or old['solver_implementation']!=st.sha(Path(transport.__file__)):
        raise ValueError('CPU provenance is not the original strict solver')
    required=('row_sum_max_error','capacity_max_relative_error','min_W','objective_consistency_error','free_lower_bound_gap')
    if (not r['success'] or r['solver_status']!=0 or not all(k in f and np.isfinite(f[k]) for k in required)
        or not np.isfinite(r['objective'])):return None
    t=r['objective']
    if (f['row_sum_max_error']>1e-7 or f['capacity_max_relative_error']>1e-6 or f['min_W']< -1e-10
        or f['objective_consistency_error']>1e-7*max(1.,abs(t)) or f['free_lower_bound_gap']< -1e-7*max(1.,abs(t))):return None
    return dict(success=True,objective=t,free=r['free'],feasibility=f,acceptance='strict_CPU_optimal_revalidated_summary',
        protocol=lp_v2.PROTOCOL['version'],backend='HiGHS-DS',source=str(path),source_seal=record['seal'],
        dual_evidence='original sealed HiGHS Optimal status at primal 1e-10/dual 1e-9; not independently recomputed',
        normalized_gap=None,negative_mass=0. if f['min_W']>=0 else None,stop_reason=r['message'],
        iterations=r['iterations'],historical_seconds=r['seconds'],
        stopping=dict(primal=CONFIG['solver_primal_tolerance'],dual=CONFIG['solver_dual_tolerance']))


class Coverage(rt.Coverage):
    def __init__(self,a,rms):super().__init__(a,rms);self.worker=Worker(a)
    def __exit__(self,*args):self.worker.__exit__(*args)
    def score(self,query,templates):
        qdata,qmeta=rt.region(self.a,query,self.rms);pending=[];entries=[]
        for template in templates:
            tdata,tmeta=self.reference(template);item=distance(self.a,self.rms,query,template,qdata,qmeta,tdata,tmeta)
            path=rt.pair_path(self.a,query,template);dep=item['dependencies']
            if path.exists():
                record=st.sealed_json(path)
                if record['dependencies']!=dep or record['protocol']!=lp_v2.PROTOCOL:raise ValueError('completed pair dependency differs')
                if record.get('proof'):
                    check,proof=verify_proof(item,record['proof'])
                    if not check['accepted'] or proof['seal']!=record['proof_seal']:raise ValueError('completed proof failed revalidation')
                elif strict_cpu(self.a,self.rms,query,template,dep)!=record['result']:raise ValueError('CPU reuse no longer verifies')
            else:
                result=strict_cpu(self.a,self.rms,query,template,dep)
                record=None
                if result:
                    record=st.sealed_json(path,dict(dependencies=dep,protocol=lp_v2.PROTOCOL,query=ip.image(query),
                        template=ip.image(template),result=result,distance_seconds=item['distance_seconds'],GT_used=False))
                else:
                    # A terminated coordinator may leave a complete worker proof
                    # without the small pair commit. Re-audit it before new GPU work.
                    for proof_path in sorted((self.a.cache_root/'LP_proofs'/ip.key(query)/ip.key(template)).glob('*/manifest.json')):
                        check,proof=verify_proof(item,proof_path.parent)
                        if not check['accepted']:continue
                        result=dict(success=True,objective=check['objective'],free=check['free'],feasibility=check,
                            **{k:check[k] for k in ('acceptance','stop_reason','backend','negative_mass','normalized_gap')})
                        record=st.sealed_json(path,dict(dependencies=dep,protocol=lp_v2.PROTOCOL,query=ip.image(query),template=ip.image(template),
                            result=result,proof=str(proof_path.parent),proof_seal=proof['seal'],distance_seconds=item['distance_seconds'],GT_used=False))
                        break
                    if record is not None:
                        entries.append(dict(template=template,item=item,path=path,record=record));continue
                    attempt=str(time.time_ns());proof=self.a.cache_root/'LP_proofs'/ip.key(query)/ip.key(template)/attempt
                    pending.append(dict(**item,id=ip.key(template),proof=str(proof),path=path,template=template))
            entries.append(dict(template=template,item=item,path=path,record=record))
        if pending:
            response,elapsed=self.worker.solve([{k:v for k,v in p.items() if k not in ('path','template')} for p in pending])
            failed=[]
            for item,ref in zip(pending,response['records']):
                check,proof=verify_proof(item,ref['proof'])
                attempt=dict(protocol=lp_v2.PROTOCOL,dependencies=item['dependencies'],query=ip.image(query),template=ip.image(item['template']),
                    result=dict(success=check['accepted'],objective=check.get('objective'),free=check.get('free'),
                        feasibility=check,iterations=(response['solver'].get('convergence') or {}).get('nb_iterations'),
                        solver_engine_seconds=response['solver'].get('engine_seconds'),shared_batch_end_to_end_seconds=elapsed,
                        batch_pairs=len(pending),**{k:check.get(k) for k in ('acceptance','stop_reason','backend','negative_mass','normalized_gap')}),
                    proof=ref['proof'],proof_seal=proof['seal'],distance_seconds=item['distance_seconds'],GT_used=False)
                st.sealed_json(self.a.run_root/'attempts'/ip.key(query)/(ip.key(item['template'])+'_'+Path(ref['proof']).name+'.json'),attempt)
                if check['accepted']:st.sealed_json(item['path'],attempt);self.new_pairs+=1
                else:failed.append(item['id'])
            if failed:raise ValueError('LPs pending under v2; proofs retained, no alternate cost: '+','.join(failed))
        records=[]
        for entry in entries:
            r=entry['record'] or st.sealed_json(entry['path'])
            records.append(dict(path=str(entry['path']),seal=r['seal'],template=r['template'],result=r['result'],distance_seconds=r['distance_seconds']))
        cap=min(records,key=lambda r:r['result']['objective']);free=min(records,key=lambda r:r['result']['free'])
        return dict(cap=cap['result']['objective'],free=free['result']['free'],cap_template=cap['template'],free_template=free['template'],
            templates=records,query_region=qmeta['seal'])
