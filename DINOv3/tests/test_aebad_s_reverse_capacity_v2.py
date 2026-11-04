"""CPU scientific checks for phase1/mixed-v2. No CUDA or real cuOpt import."""
import copy
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scipy.optimize import linprog
from DINOv3.MADEqual.aebad_s_reverse_capacity_coverage_p0 import lp_v2 as v,transport as old
from DINOv3.MADEqual.aebad_s_reverse_capacity_coverage_p0 import phase1 as ph,pipeline,inputs as ip,runtime as rt,mixed_runtime as mr,runner
from DINOv3.MADEqual.aebad_s_reverse_capacity_coverage_p0.protocol import CONFIG,BRANCHES


def exact(d=None,a=None,b=None):
    d=np.array([[0.,1.],[0.,10.]]) if d is None else np.array(d,float)
    a=np.array([.9,.1]) if a is None else np.array(a,float);b=a.copy() if b is None else np.array(b,float)
    eq,rhs,ub,u=old.constraints(d,a,b);c=np.r_[np.zeros(d.size),1.]
    r=linprog(c,A_eq=eq,b_eq=rhs,A_ub=ub,b_ub=u,bounds=(0,None),method='highs-ds',
        options=dict(primal_feasibility_tolerance=1e-10,dual_feasibility_tolerance=1e-9))
    assert r.success
    return dict(distance=d,a=a,b=b,reference_objective=float(r.fun)),r.x,np.r_[r.eqlin.marginals,r.ineqlin.marginals]


def image(name):return dict(relative_path=name,image_sha256=name,image_height=256,image_width=256)


class CertificateTests(unittest.TestCase):
    def test_optimal_and_timeout_use_same_independent_certificate(self):
        p,x,y=exact()
        a=v.audit(p,x,y,'Optimal');b=v.audit(p,x,y,'TimeLimit')
        self.assertTrue(a['accepted'],a);self.assertTrue(b['accepted'],b)
        self.assertEqual(a['acceptance'],'solver_optimal_independently_verified')
        self.assertEqual(b['acceptance'],'timeout_numerically_verified')
        self.assertAlmostEqual(a['primal_objective'],10/91)
        self.assertAlmostEqual(a['dual_objective'],10/91)
        self.assertLess(a['normalized_gap'],1e-10)

    def test_gap_alone_cannot_accept_dual_infeasibility(self):
        p,x,y=exact();y[-2:]*=-1
        r=v.audit(p,x,y,'TimeLimit')
        self.assertLess(r['normalized_gap'],1e-10)
        self.assertFalse(r['checks']['dual']);self.assertFalse(r['accepted'])

    def test_feasible_dual_with_large_gap_not_accepted(self):
        p,x,y=exact();y[:]=0
        r=v.audit(p,x,y,'Optimal')
        self.assertTrue(r['checks']['dual']);self.assertFalse(r['checks']['gap']);self.assertFalse(r['accepted'])

    def test_missing_dual_or_wrong_backend_cannot_be_relabelled(self):
        p,x,y=exact()
        self.assertFalse(v.audit(p,x,[],'Optimal')['accepted'])
        self.assertFalse(v.audit(p,x,y,'IterationLimit')['accepted'])
        self.assertFalse(v.audit(p,x,y,'Optimal','DualSimplex')['accepted'])

    def test_small_capacity_uses_each_b_not_global_absolute_error(self):
        p,x,y=exact([[0.,0.],[0.,0.]],[.5,.5],[1-1e-5,1e-5])
        W=x[:-1].reshape(2,2);W[0,1]+=1e-8;W[0,0]-=1e-8
        r=v.audit(p,x,y,'TimeLimit')
        self.assertLess(r['primal_equality_Linf'],1e-5)
        self.assertGreater(r['capacity_max_relative_error'],1e-5);self.assertFalse(r['accepted'])

    def test_negative_values_are_recorded_not_repaired(self):
        p,x,y=exact([[0.,0.],[0.,0.]],[.5,.5],[.5,.5]);W=x[:-1].reshape(2,2)
        W[:]=[[1+2e-10,-2e-10],[-2e-10,1+2e-10]];before=x.copy()
        r=v.audit(p,x,y,'TimeLimit')
        self.assertFalse(r['accepted']);self.assertAlmostEqual(r['negative_mass'],4e-10)
        np.testing.assert_array_equal(x,before);self.assertFalse(r['repaired'])

    def test_new_tolerance_not_applied_to_original_protocol(self):
        self.assertEqual(v.PROTOCOL['row_atol'],1e-5);self.assertEqual(CONFIG['row_relative_tolerance'],1e-7)
        self.assertEqual(CONFIG['distance_backend_check_atol'],1e-5)
        self.assertEqual(v.settings()['pdlp_precision'],2);self.assertEqual(v.settings()['method'],'PDLP')
        self.assertFalse(v.settings()['crossover']);self.assertEqual(v.settings()['absolute_primal_tolerance'],1e-7)

    def test_CPU_reference_error_and_returned_t_remain_explicit(self):
        p,x,y=exact();r=v.audit(p,x,y,'TimeLimit')
        self.assertEqual(r['objective'],x[-1]);p['reference_objective']+=2e-5
        r=v.audit(p,x,y,'TimeLimit');self.assertFalse(r['checks']['CPU_reference'])

    def test_normalized_gap_definition(self):
        p,x,y=exact();y*=.9;r=v.audit(p,x,y,'TimeLimit')
        self.assertAlmostEqual(r['normalized_gap'],abs(r['primal_objective']-r['dual_objective'])/max(1.,abs(r['primal_objective'])+abs(r['dual_objective'])))


class MassFlowCoordinateTests(unittest.TestCase):
    def test_operator_and_rhs_are_positive_diagonal_transforms(self):
        p,x,y=exact();original=v.gpu.sparse_batch([p]);flow,S,R=v.mass_flow_batch([p])
        np.testing.assert_allclose(flow['matrix'].toarray(),R[:,None]*original['matrix'].toarray()/S[None,:],rtol=1e-14,atol=1e-14)
        np.testing.assert_allclose(flow['rhs'],R*original['rhs'],rtol=1e-14,atol=1e-14)
        np.testing.assert_array_equal(flow['senses'],original['senses'])
        self.assertEqual(flow['matrix'].shape,original['matrix'].shape)
        self.assertTrue((S>0).all() and (R>0).all());self.assertEqual(S[-1],1.)
        np.testing.assert_allclose(flow['matrix']@(S*x),R*(original['matrix']@x),rtol=1e-13,atol=1e-13)
        self.assertEqual(float(flow['objective']@(S*x)),p['reference_objective'])

    def test_rectangular_and_unbalanced_areas_preserve_original_optimum_and_dual(self):
        rng=np.random.default_rng(42)
        for n,m in ((2,3),(4,3),(3,5)):
            a=np.r_[1e-4,rng.uniform(.1,1,n-1)];a/=a.sum()
            b=rng.uniform(.1,1,m);b/=b.sum()
            p,x,y=exact(rng.uniform(.6,.9,(n,m)),a,b)
            batch,S,R=v.mass_flow_batch([p]);E=batch['senses']=='E';L=~E
            r=linprog(batch['objective'],A_eq=batch['matrix'][E],b_eq=batch['rhs'][E],
                A_ub=batch['matrix'][L],b_ub=batch['rhs'][L],bounds=(0,None),method='highs-ds',
                options=dict(primal_feasibility_tolerance=1e-10,dual_feasibility_tolerance=1e-9))
            self.assertTrue(r.success,r.message)
            original_y=R*np.r_[r.eqlin.marginals,r.ineqlin.marginals]
            check=v.audit(p,r.x/S,original_y,'Optimal')
            self.assertTrue(check['accepted'],check)
            self.assertAlmostEqual(r.fun,p['reference_objective'],places=10)

    def test_mapped_dual_is_certified_in_original_coefficients_and_blocks(self):
        p,x,y=exact();q,xx,yy=exact([[.8,.9,.7],[.9,.7,.8]],[.2,.8],[.2,.3,.5])
        solver=v.Solver.__new__(v.Solver);batch,S,R=v.mass_flow_batch([p,q])
        primal=np.r_[x,xx]*S;dual=np.r_[y,yy]/R
        solver.solve=lambda b:dict(primal=primal,dual=dual,status='TimeLimit',solved_by='PDLP',error_status=0)
        results,_=solver.solve_certified([p,q],coordinate_system='mass_flow_v1')
        self.assertTrue(all(r['audit']['accepted'] for r in results))
        np.testing.assert_allclose(results[0]['primal'],x);np.testing.assert_allclose(results[1]['dual'],yy)
        np.testing.assert_array_equal(results[0]['solver_primal'],primal[:len(x)])
        self.assertEqual(results[1]['primal'][-1],xx[-1])
        with self.assertRaises(ValueError):solver.solve_certified([p],coordinate_system='unrequested')

    def test_missing_dual_and_bad_original_certificate_still_fail(self):
        p,x,y=exact();batch,S,R=v.mass_flow_batch([p]);solver=v.Solver.__new__(v.Solver)
        for dual in (np.array([]),np.zeros_like(y)):
            solver.solve=lambda b:dict(primal=x*S,dual=dual,status='TimeLimit',solved_by='PDLP',error_status=0)
            results,_=solver.solve_certified([p],coordinate_system='mass_flow_v1')
            self.assertFalse(results[0]['audit']['accepted'])
        self.assertEqual(v.PROTOCOL['solver']['time_limit'],30.)

    def test_trial_is_16_only_saves_raw_and_mapped_evidence_without_formal_switch(self):
        p,x,y=exact();problems=[dict(p,id=f'{i:02d}') for i in range(16)]
        class FakeSolver:
            calls=[]
            def __init__(self,workers):pass
            def solve_certified(self,group,coordinate_system='original'):
                FakeSolver.calls.append((len(group),coordinate_system))
                return [dict(primal=x,dual=y,solver_primal=x,solver_dual=y,audit=v.audit(p,x,y,'TimeLimit')) for _ in group],dict(solver={'status':'TimeLimit'})
        with tempfile.TemporaryDirectory() as tmp:
            a=SimpleNamespace(problem_root=Path(tmp)/'old',output_root=Path(tmp)/'new',cpu_workers=7)
            with patch.object(v.gpu,'load_problems',return_value=(dict(seal='original_inputs'),problems)):
                r=v.mass_flow_trial(a,FakeSolver)
            self.assertEqual(r['accepted'],16);self.assertFalse(r['formal_backend_changed'])
            self.assertEqual(FakeSolver.calls,[(8,'mass_flow_v1'),(8,'mass_flow_v1')])
            arrays,record=ph.st.get(a.output_root/'proofs/00')
            self.assertEqual(set(arrays),{'primal','dual','solver_primal','solver_dual'})
            self.assertEqual(record['metadata']['trial']['actual_settings'],v.settings())
            for array in arrays.values():array._mmap.close()
            self.assertFalse((a.output_root/'pairs').exists());self.assertFalse((a.output_root/'predictions_frozen.json').exists())
            self.assertEqual(len(ip.csv_rows(a.output_root/'LP_checks.csv')),16)


class ScopeAndReuseTests(unittest.TestCase):
    def test_plan_and_observations_do_not_expand_settings(self):
        samples=[dict(image(str(i)),sample_id=str(i),label=int(i>=77)) for i in range(289)]
        normal=[image('source'+str(i)) for i in range(66)];supports=[image('target'+str(i)) for i in range(8)]
        def only(d):self.assertEqual(d,'background');return samples
        x=SimpleNamespace(samples=only,calibration=normal,supports=lambda d,s:supports)
        a=SimpleNamespace(lp_profile='mixed_background_v2')
        with patch.object(ip,'assets',return_value=x):
            self.assertEqual(len(list(pipeline.observations(a))),355)
            self.assertEqual(len(pipeline.region_rows(a)),363)
        self.assertEqual(ph.PLAN['LPs'],2840);self.assertEqual(ph.PLAN['evaluation_LPs'],2312);self.assertEqual(ph.PLAN['calibration_LPs'],528)

    def test_legacy_small_summaries_are_not_promoted(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=SimpleNamespace(run_root=Path(tmp)/'new',legacy_trials=[Path(tmp)/'old'])
            ph.st.sealed_json(a.legacy_trials[0]/'trial_results/repeat0_batch0.json',dict(solver=dict(status='TimeLimit',
                convergence=dict(primal_residual=1e-7,dual_residual=1e-7,gap=1e-7)),checks=[dict(problem_id='00',objective=.7,
                reference_objective=.7,reference_abs_error=0.,feasibility=dict(row_sum_max_error=1e-6,capacity_max_relative_error=1e-6,objective_consistency_error=1e-8))]))
            rows=ph.old_audit(a)
            self.assertTrue(rows[0]['partial_ABCD_pass']);self.assertIn('missing',rows[0]['acceptance'])

    def test_strict_CPU_reference_revalidated_and_backend_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=SimpleNamespace(resume_from=Path(tmp));q=image('q');t=image('t')
            dep=dict(query='qr',template='tr',RMS='rms',distance_device='cuda:0')
            p,x,y=exact();result=old.solve(p['distance'],p['a'],p['b'])
            path=a.resume_from/'pairs'/ip.key(q)/(ip.key(t)+'.json')
            ph.st.sealed_json(path,dict(dependencies=dict(**dep,protocol=ph.st.digest(CONFIG),solver_implementation=ph.st.sha(Path(old.__file__))),result=result))
            r=mr.strict_cpu(a,{},q,t,dep)
            self.assertEqual(r['backend'],'HiGHS-DS');self.assertEqual(r['objective'],result['objective'])
            self.assertIsNone(r['normalized_gap'])
            with self.assertRaises(ValueError):mr.strict_cpu(a,{},q,t,dict(dep,RMS='changed'))

    def test_proof_read_rechecks_saved_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p,x,y=exact();bundle=ph.st.put(root/'distance',{k:p[k] for k in ('distance','a','b')},{})
            item=dict(bundle=str(root/'distance'),bundle_seal=bundle['seal'],dependencies={'input':'fixed'},reference_objective=p['reference_objective'])
            audit=v.audit(p,x,y,'TimeLimit')
            ph.st.put(root/'proof',dict(primal=x,dual=y),dict(protocol=v.PROTOCOL,settings=v.settings(),dependencies=item['dependencies'],audit=audit,solver=dict(status='TimeLimit',solved_by='PDLP',error_status=0)))
            check,r=mr.verify_proof(item,root/'proof');self.assertTrue(check['accepted'])
            ph.st.put(root/'fp32',dict(primal=x,dual=y),dict(protocol=v.PROTOCOL,settings=dict(v.settings(),pdlp_precision=0),
                dependencies=item['dependencies'],audit=audit,solver=dict(status='TimeLimit',solved_by='PDLP',error_status=0)))
            with self.assertRaises(ValueError):mr.verify_proof(item,root/'fp32')
            x=x.copy();x[-1]+=.1
            with (root/'proof/primal.npy').open('wb') as f:np.save(f,x)
            with self.assertRaises(ValueError):mr.verify_proof(item,root/'proof')

    def test_mixed_engine_commits_good_pairs_and_retries_only_failed_item(self):
        p,xx,yy=exact();q=image('q');templates=[image('t'+str(i)) for i in range(8)]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=SimpleNamespace(run_root=root/'run',cache_root=root/'cache',resume_from=None)
            dep=dict(query='qregion',template='tregion',RMS='rms',distance_device='cuda:0')
            bundle=ph.st.put(root/'distance',{k:p[k] for k in ('distance','a','b')},{})
            item=dict(bundle=str(root/'distance'),bundle_seal=bundle['seal'],dependencies=dep,reference_objective=p['reference_objective'],distance_seconds=0.)
            class FakeWorker:
                total=0
                fail=True
                def __init__(self,a):pass
                def __exit__(self,*_):pass
                def solve(self,items):
                    records=[];solver=dict(status='TimeLimit',solved_by='PDLP',error_status=0,engine_seconds=.01,convergence=dict(nb_iterations=1))
                    for i in items:
                        x=xx.copy()
                        if FakeWorker.fail:FakeWorker.fail=False;x[0]+=.1
                        audit=v.audit(p,x,yy,'TimeLimit');record=ph.st.put(i['proof'],dict(primal=x,dual=yy),
                            dict(protocol=v.PROTOCOL,settings=v.settings(),dependencies=dep,audit=audit,solver=solver))
                        records.append(dict(id=i['id'],proof=i['proof'],seal=record['seal']));FakeWorker.total+=1
                    return dict(records=records,solver=solver),.02
            with patch.object(rt,'region',return_value=({},dict(seal='region'))),patch.object(mr,'distance',return_value=item),patch.object(mr,'Worker',FakeWorker):
                with mr.Coverage(a,{'seal':'rms'}) as engine:
                    with self.assertRaises(ValueError):engine.score(q,templates)
                    self.assertEqual(len(list((a.run_root/'pairs').glob('*/*.json'))),7)
                    r=engine.score(q,templates)
                    self.assertEqual(FakeWorker.total,9);self.assertEqual(len(r['templates']),8)
                    self.assertAlmostEqual(r['cap'],p['reference_objective'])
                    engine.score(q,templates);self.assertEqual(FakeWorker.total,9)
                    (a.run_root/'pairs'/ip.key(q)/(ip.key(templates[1])+'.json')).unlink()
                    engine.score(q,templates);self.assertEqual(FakeWorker.total,9)

    def test_import_and_worker_metadata_do_not_import_Torch_or_CUDA(self):
        code='from DINOv3.MADEqual.aebad_s_reverse_capacity_coverage_p0 import phase1; import sys; assert "torch" not in sys.modules; assert "cuopt" not in sys.modules'
        subprocess.run([sys.executable,'-B','-c',code],check=True,capture_output=True)
        r=subprocess.run([sys.executable,'-B','-m','DINOv3.MADEqual.aebad_s_reverse_capacity_coverage_p0.lp_worker','--versions'],check=True,capture_output=True,text=True)
        self.assertIn('packages',json.loads(r.stdout))

    def test_review16_retries_only_missing_certificate_and_counts_all_attempt_time(self):
        p,x,y=exact();templates=[image('support'+str(i)) for i in range(8)];queries=[image('source0'),dict(image('query0'),sample_id='q0')]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parent=root/'old';export=root/'export';out=root/'new'
            saved=ph.st.sealed_json(parent/'inputs.json',{'fixture':'old'})
            a=SimpleNamespace(run_root=out,cache_root=root/'cache',resume_from=parent,distance_exports=[export],legacy_trials=[],
                _capacity_parent=saved,device='cuda:0',_started=time.perf_counter(),allow_over_12h=False)
            refs=[];problems=[]
            for index,(q,t) in enumerate((q,t) for q in queries for t in templates):
                dep=dict(query='qr',template='tr',RMS='rms',distance_device='cuda:0')
                original=ph.st.sealed_json(parent/'pairs'/ip.key(q)/(ip.key(t)+'.json'),dict(result=dict(success=True,objective=p['reference_objective'])))
                r=ph.st.put(export/'problems'/f'{index:02d}',{k:p[k] for k in ('distance','a','b')},
                    dict(query=ip.image(q),template=ip.image(t),dependencies=dep,original_pair_seal=original['seal']))
                refs.append(dict(id=f'{index:02d}',seal=r['seal'],reference_objective=p['reference_objective']));problems.append(dict(p,id=f'{index:02d}'))
            manifest=dict(parent_inputs=saved['seal'],problems=refs)
            assets=SimpleNamespace(calibration=[queries[0]],samples=lambda d:[queries[1]],supports=lambda d,s:templates)
            class FakeWorker:
                calls=0
                pairs=0
                fail=True
                def __init__(self,a):pass
                def __enter__(self):return self
                def __exit__(self,*_):pass
                def solve(self,items):
                    FakeWorker.calls+=1;FakeWorker.pairs+=len(items);records=[]
                    solver=dict(status='TimeLimit',solved_by='PDLP',error_status=0,engine_seconds=2.,convergence=dict(nb_iterations=1))
                    for item in items:
                        dual=y.copy()
                        if FakeWorker.fail:FakeWorker.fail=False;dual[:]=0
                        audit=v.audit(p,x,dual,'TimeLimit')
                        r=ph.st.put(item['proof'],dict(primal=x,dual=dual),dict(protocol=v.PROTOCOL,settings=v.settings(),
                            dependencies=item['dependencies'],audit=audit,solver=solver))
                        records.append(dict(proof=item['proof'],seal=r['seal']))
                    return dict(job=str(FakeWorker.calls),records=records,solver=solver),3.
            inv=dict(remaining_LPs_upper_bound=2824,remaining_regions=0,missing_feature_images=0)
            with patch.object(ph,'freeze',return_value=dict(identity='review')),patch.object(ip,'assets',return_value=assets),\
                patch.object(rt,'source_rms',return_value=dict(seal='rms')),patch.object(ph.gpu_lp,'load_problems',return_value=(manifest,problems)),\
                patch.object(mr,'Worker',FakeWorker),patch.object(rt,'cached_region_path',return_value=root/'region'),\
                patch.object(ph,'inventory',return_value=inv),patch.object(pipeline,'budget'):
                first=ph.smoke(a);self.assertEqual(first['accepted'],15)
                self.assertFalse((out/'smoke.json').exists());self.assertEqual(FakeWorker.pairs,16)
                second=ph.smoke(a);self.assertEqual(second['accepted'],16);self.assertEqual(FakeWorker.pairs,17)
                self.assertEqual(ph.st.sealed_json(out/'smoke.json')['review_seconds'],9.)
                self.assertEqual(len(ip.csv_rows(out/'LP_review16.csv')),16)
                self.assertEqual(len(ip.csv_rows(out/'LP_review16_attempts.csv')),17)
                ph.smoke(a);self.assertEqual(FakeWorker.pairs,17)

    def test_real_parent_background_population_and_no_server_contact(self):
        a=runner.normalize(runner.parser().parse_args(['--lp-profile','mixed_background_v2']))
        if not (a.parent_run/'inputs.json').is_file():self.skipTest('completed parent metadata intentionally excluded from code-only repository')
        x=ip.assets(a)
        self.assertEqual(len(x.samples('background')),289);self.assertEqual(sum(r['label']==0 for r in x.samples('background')),77)
        self.assertEqual(len(x.supports('background',42)),8);self.assertEqual(len(x.calibration),66)

    def test_common_pipeline_runs_only_289_and_66_then_completes_six_branches(self):
        samples=[dict(image('query_'+str(i)),sample_id=f'test_{i:04d}',label=int(i>=77)) for i in range(289)]
        normals=[image('source_'+str(i)) for i in range(66)];supports=[image('support_'+str(i)) for i in range(8)]
        class Assets:
            def __init__(self):
                self.calibration=normals
                self.query={('background',42,r['sample_id']):dict(map_hash='original_map') for r in samples}
            def samples(self,d):
                if d!='background':raise AssertionError('other domain expanded')
                return samples
            def supports(self,d,s):
                if (d,s)!=('background',42):raise AssertionError('other support seed expanded')
                return supports
            def P(self,d,s,kind,row):return dict(P=1+int(row['relative_path'].split('_')[-1])*.01,origin='original_SEGMENT')
        class FakeCoverage:
            calls=0
            def __init__(self,a,rms):self.a=a;self.new_pairs=0
            def __enter__(self):return self
            def __exit__(self,*_):pass
            def score(self,row,templates):
                i=int(row['relative_path'].split('_')[-1]);cap=.7+i*.003;free=.6+i*.002;refs=[]
                for j,t in enumerate(templates):
                    path=rt.pair_path(self.a,row,t);result=dict(success=True,objective=cap+j*.01,free=free+j*.01,
                        feasibility={},backend='PDLP',acceptance='timeout_numerically_verified',stop_reason='TimeLimit')
                    value=dict(protocol=v.PROTOCOL,query=ip.image(row),template=ip.image(t),result=result,distance_seconds=0.)
                    if not path.exists():self.new_pairs+=1;FakeCoverage.calls+=1
                    r=ph.st.sealed_json(path,value);refs.append(dict(path=str(path),seal=r['seal'],template=ip.image(t),result=result,distance_seconds=0.))
                return dict(cap=cap,free=free,cap_template=ip.image(templates[0]),free_template=ip.image(templates[0]),templates=refs,query_region='fixture')
        with tempfile.TemporaryDirectory() as tmp:
            a=SimpleNamespace(run_root=Path(tmp),cache_root=Path(tmp)/'cache',lp_profile='mixed_background_v2',
                _started=time.perf_counter(),allow_over_12h=False)
            with patch.object(ip,'assets',return_value=Assets()),patch.object(ip,'signature',return_value='fixed_v2'),\
                 patch.object(rt,'source_rms',return_value={}),patch.object(pipeline,'engine',FakeCoverage):
                r=pipeline.predict(a);self.assertEqual(r['records'],289)
                snapshot={p:p.read_bytes() for p in (a.run_root/'background/seed42/scores').glob('*.json')}
                pipeline.predict(a);self.assertEqual(FakeCoverage.calls,2840)
                self.assertEqual(snapshot,{p:p.read_bytes() for p in snapshot})
                pipeline.evaluate(a)
                ph.st.write_csv(a.run_root/'LP_review16.csv',[dict(id='00',reference_abs_error=8e-7)])
                status=ph.summarize(a)
                self.assertFalse(status['pending']);self.assertEqual(status['completed_LPs'],2840)
                self.assertEqual(status['timeout_accepted'],2840)
                self.assertEqual(len(ip.csv_rows(a.run_root/'setting_metrics.csv')),6)
                self.assertEqual(len(ip.csv_rows(a.run_root/'per_image_scores.csv')),289)
                self.assertEqual(len(ip.csv_rows(a.run_root/'normal_calibration.csv')),66)
                self.assertEqual(len(ip.csv_rows(a.run_root/'LP_completion.csv')),2840)
                self.assertFalse((a.run_root/'illumination').exists());self.assertFalse((a.run_root/'background/seed43').exists())
                (a.run_root/'pairs'/ip.key(samples[0])/(ip.key(supports[0])+'.json')).unlink()
                status=ph.summarize(a)
                self.assertTrue(status['pending']);self.assertEqual(status['uncompleted_LPs'],1)


if __name__=='__main__':unittest.main()
