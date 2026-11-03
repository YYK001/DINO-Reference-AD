"""CPU fixtures for the optional cuOpt adapter; never initializes a GPU."""
from enum import IntEnum
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scipy.optimize import linprog
from DINOv3.MADEqual.aebad_s_reverse_capacity_coverage_p0 import gpu_lp as g


def problem(i=0):
    d=np.array([[.2,1.3],[.8,.1]],np.float32);a=np.array([.6,.4]);b=np.array([.7,.3])
    reference=g.transport.solve(d,a,b,save_plan=True)
    assert reference['success']
    return dict(id=f'{i:02d}',distance=d,a=a,b=b,reference_objective=reference['objective']),reference


def cpu_fixture(batch):
    """Test double only: solve the same block matrix on CPU, no GPU speed claim."""
    mask=batch['senses']=='E';matrix=batch['matrix'];rhs=batch['rhs']
    r=linprog(batch['objective'],A_eq=matrix[mask],b_eq=rhs[mask],A_ub=matrix[~mask],b_ub=rhs[~mask],
        bounds=(0,None),method='highs-ds',options=dict(primal_feasibility_tolerance=1e-10,dual_feasibility_tolerance=1e-9))
    assert r.success
    return dict(primal=r.x,objective=float(r.fun),solver_optimal=True,GPU_PDLP=True,status='Optimal')


class NumericalTests(unittest.TestCase):
    def test_block_diagonal_keeps_individual_constraints_and_minimax(self):
        first,_=problem();second,_=problem(1);second['distance']=second['distance']*2
        second['reference_objective']*=2
        batch=g.sparse_batch([first,second]);self.assertEqual(batch['matrix'].shape,(12,10))
        result=cpu_fixture(batch);checks=g.check_solution([first,second],batch,result)
        self.assertTrue(all(c['passed'] for c in checks),checks)
        self.assertAlmostEqual(result['objective'],sum(p['reference_objective'] for p in (first,second)))

    def test_per_problem_error_not_hidden_by_global_objective(self):
        ps=[problem(i)[0] for i in range(2)];batch=g.sparse_batch(ps);r=cpu_fixture(batch)
        # Keep the aggregate objective exactly unchanged; each t is now wrong.
        r['primal'][4]+=1e-4;r['primal'][9]-=1e-4
        self.assertTrue(all(not c['passed'] for c in g.check_solution(ps,batch,r)))

    def test_fp32_cast_output_is_checked_on_original_small_masses(self):
        d=np.zeros((2,2),np.float32);a=np.array([.5,.5]);b=np.array([1-1e-5,1e-5])
        p=dict(id='small',distance=d,a=a,b=b,reference_objective=0.)
        W=np.tile(b,(2,1)).astype(np.float32);W[0,1]+=2e-8
        r=dict(primal=np.r_[W.ravel(),0.],objective=0.,solver_optimal=True,GPU_PDLP=True)
        check=g.check_solution([p],g.sparse_batch([p]),r)[0]
        self.assertFalse(check['passed']);self.assertGreater(check['feasibility']['capacity_max_relative_error'],1e-6)

    def test_no_plan_equality_requirement_for_degenerate_optima(self):
        p=dict(id='zero',distance=np.zeros((2,2),np.float32),a=np.array([.5,.5]),b=np.array([.5,.5]),reference_objective=0.)
        for W in (np.eye(2),np.ones((2,2))/2):
            r=dict(primal=np.r_[W.ravel(),0.],objective=0.,solver_optimal=True,GPU_PDLP=True)
            self.assertTrue(g.check_solution([p],g.sparse_batch([p]),r)[0]['passed'])

    def test_nonoptimal_or_wrong_solver_not_accepted_even_when_feasible(self):
        p,_=problem();b=g.sparse_batch([p]);good=cpu_fixture(b)
        for change in (dict(solver_optimal=False),dict(GPU_PDLP=False)):
            r=dict(good,**change);c=g.check_solution([p],b,r)[0]
            self.assertTrue(c['numerical_checks_passed']);self.assertFalse(c['passed'])

    def test_bad_primal_and_objective_fail(self):
        p,_=problem();b=g.sparse_batch([p]);good=cpu_fixture(b)
        for change in (dict(primal=[1.]),dict(primal=np.full(5,np.nan)),dict(objective=float('inf'))):
            self.assertFalse(g.check_solution([p],b,dict(good,**change))[0]['passed'])
        self.assertIsNone(g.finite_record(dict(convergence=dict(gap=float('nan'))))['convergence']['gap'])

    def test_settings_are_explicit_fp32_single_gpu_and_original_tolerances(self):
        s=g.settings_values(30,8)
        self.assertEqual(s['pdlp_precision'],0);self.assertEqual(s['method'],'PDLP')
        self.assertEqual(s['num_gpus'],1);self.assertFalse(s['crossover'])
        self.assertEqual(s['absolute_primal_tolerance'],1e-10)
        self.assertEqual(s['absolute_dual_tolerance'],1e-9)
        self.assertEqual(s['relative_gap_tolerance'],1e-9)

    def test_mixed_changes_only_the_requested_precision_setting(self):
        single=g.settings_values(30,8,'fp32');mixed=g.settings_values(30,8,'mixed')
        self.assertEqual(mixed['pdlp_precision'],2)
        self.assertEqual({k:(single[k],mixed[k]) for k in single if single[k]!=mixed[k]},dict(pdlp_precision=(0,2)))
        with self.assertRaises(ValueError):g.settings_values(30,8,'automatic')

    def test_stock_api_adapter_sends_original_fp64_coefficients(self):
        class Method(IntEnum):PDLP=1
        class Mode(IntEnum):Stable3=4
        class Settings:
            def __init__(self):self.values={}
            def set_parameter(self,key,value):self.values[key]=value
        class Model:
            def __getattr__(self,name):
                def save(*args):setattr(self,name+'_args',args)
                return save
        p,_=problem();batch=g.sparse_batch([p]);result=cpu_fixture(batch);models=[]
        def solve(model,settings):
            models.append(model)
            return SimpleNamespace(get_solved_by=lambda:Method.PDLP,get_termination_reason=lambda:'Optimal',
                get_primal_solution=lambda:result['primal'],get_primal_objective=lambda:result['objective'],
                get_solve_time=lambda:.01,get_lp_stats=lambda:dict(primal_residual=0.,dual_residual=0.,gap=0.,nb_iterations=1))
        fake=SimpleNamespace(DataModel=Model,SolverSettings=Settings,Solve=solve)
        keys=SimpleNamespace(**{'CUOPT_'+n.upper():n for n in g.settings_values(30,8)})
        modules={'cuopt':SimpleNamespace(linear_programming=fake),
            'cuopt.linear_programming.solver_settings.solver_settings':SimpleNamespace(SolverMethod=Method,PDLPSolverMode=Mode),
            'cuopt.linear_programming.solver':SimpleNamespace(solver_parameters=keys)}
        with patch.dict(sys.modules,modules):
            adapter=g.CuOptPDLP();r=adapter.solve(batch)
            mixed=g.CuOptPDLP(30,8,'mixed');other=mixed.solve(batch)
        self.assertTrue(g.check_solution([p],batch,r)[0]['passed'])
        self.assertTrue(g.check_solution([p],batch,other)[0]['passed'])
        np.testing.assert_array_equal(models[0].set_csr_constraint_matrix_args[0],batch['matrix'].data)
        self.assertEqual(models[0].set_csr_constraint_matrix_args[0].dtype,np.float64)
        self.assertEqual(models[0].set_csr_constraint_matrix_args[1].dtype,np.int32)
        self.assertEqual(adapter.settings.values['pdlp_precision'],0)
        self.assertEqual(adapter.settings.values['method'],Method.PDLP)
        self.assertEqual(mixed.settings.values['pdlp_precision'],2)
        np.testing.assert_array_equal(models[0].set_csr_constraint_matrix_args[0],models[1].set_csr_constraint_matrix_args[0])


class TrialTests(unittest.TestCase):
    def test_import_has_no_gpu_side_effects(self):
        code='from DINOv3.MADEqual.aebad_s_reverse_capacity_coverage_p0 import gpu_lp; import sys; assert "torch" not in sys.modules; assert "cuopt" not in sys.modules'
        subprocess.run([sys.executable,'-B','-c',code],check=True,capture_output=True)

    def test_manual_batch_trial_resume_and_binding(self):
        p,_=problem();ps=[dict(p,id=f'{i:02d}') for i in range(16)]
        manifest=dict(seal='fixture',original_estimate=dict(remaining_LP_pairs=17584))
        class FakeBackend:
            calls=0
            def __init__(self,*_):pass
            def solve(self,batch):
                FakeBackend.calls+=1
                return cpu_fixture(batch)
        with tempfile.TemporaryDirectory() as tmp,patch.object(g,'load_problems',return_value=(manifest,ps)):
            a=SimpleNamespace(output_root=Path(tmp),time_limit=30.,cpu_workers=8,batch_size=8,repeats=2,max_seconds=1200.)
            first=g.trial(a,FakeBackend);second=g.trial(a,FakeBackend)
            self.assertTrue(first['passed']);self.assertEqual(first['successful_checks'],32)
            self.assertEqual(FakeBackend.calls,5) # One warmup and four measured batches, none on resume.
            self.assertEqual(first['measured_seconds'],second['measured_seconds'])
            self.assertEqual(first['effective_seconds_per_LP'],second['effective_seconds_per_LP'])
            self.assertIsNone(first['whole_experiment_hours']);self.assertFalse(first['formal_backend_changed'])
            a.batch_size=4
            with self.assertRaises(ValueError):g.trial(a,FakeBackend)
            self.assertEqual(FakeBackend.calls,5)

    def test_backend_error_preserved_no_success_or_cpu_fallback(self):
        p,_=problem();ps=[dict(p,id=f'{i:02d}') for i in range(16)]
        class Failed:
            def __init__(self,*_):pass
            def solve(self,_):raise RuntimeError('fixture solver failure')
        with tempfile.TemporaryDirectory() as tmp,patch.object(g,'load_problems',return_value=(dict(seal='fixture',original_estimate={}),ps)):
            a=SimpleNamespace(output_root=Path(tmp),time_limit=30.,cpu_workers=8,batch_size=8,repeats=1,max_seconds=1200.)
            r=g.trial(a,Failed)
            self.assertFalse(r['passed']);self.assertIsNone(r['effective_seconds_per_LP'])
            self.assertIn('fixture solver failure',g.st.sealed_json(Path(tmp)/'trial_results/repeat0_batch0.json')['error'])

    def test_missing_gpu_environment_is_recorded_without_fallback(self):
        p,_=problem();ps=[dict(p,id=f'{i:02d}') for i in range(16)]
        with tempfile.TemporaryDirectory() as tmp,patch.object(g,'load_problems',return_value=(dict(seal='fixture',original_estimate={}),ps)):
            a=SimpleNamespace(output_root=Path(tmp),time_limit=30.,cpu_workers=8,batch_size=8,repeats=2,max_seconds=1200.)
            with patch.object(g,'CuOptPDLP') as backend:
                backend.side_effect=ImportError('fixture absent cuOpt')
                r=g.trial(a,backend)
            self.assertFalse(r['passed']);self.assertEqual(r['status'],'GPU_backend_unavailable')
            self.assertFalse((Path(tmp)/'trial_results').exists())

    def test_mixed_trial_is_explicit_and_cannot_relabel_old_results(self):
        p,_=problem();ps=[dict(p,id=f'{i:02d}') for i in range(16)];seen=[]
        class FakeBackend:
            def __init__(self,time_limit,cpu_workers,precision):seen.append(precision)
            def solve(self,batch):return cpu_fixture(batch)
        with tempfile.TemporaryDirectory() as tmp,patch.object(g,'load_problems',return_value=(dict(seal='fixture',original_estimate={}),ps)):
            a=SimpleNamespace(output_root=Path(tmp),time_limit=30.,cpu_workers=8,batch_size=8,repeats=1,max_seconds=300.,precision='fp32')
            g.trial(a,FakeBackend)
            snapshot={f:f.read_bytes() for f in Path(tmp).rglob('*.json')}
            a.precision='mixed'
            with self.assertRaises(ValueError):g.trial(a,FakeBackend)
            self.assertEqual(seen,['fp32']);self.assertEqual(snapshot,{f:f.read_bytes() for f in snapshot})
            a.output_root=Path(tmp)/'new_mixed'
            result=g.trial(a,FakeBackend)
            self.assertEqual(result['status'],'MIXED_precision_trial_passed');self.assertEqual(result['precision'],'mixed')
            self.assertEqual(result['numerical_checks_passed'],16);self.assertLess(result['max_errors']['reference_abs_error'],1e-7)
            self.assertEqual(seen,['fp32','mixed'])
            self.assertEqual(g.st.sealed_json(a.output_root/'trial_inputs.json')['settings']['pdlp_precision'],2)
            self.assertEqual(snapshot,{f:f.read_bytes() for f in snapshot})

    def test_explicit_problem_provider_must_be_read_only_and_disjoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for out in (root,root/'inside',root.parent):
                with self.assertRaises(ValueError):
                    g.main(['benchmark','--precision','mixed','--problem-root',str(root),'--output-root',str(out)])

    def test_export_uses_only_cached_regions_and_resume_does_not_recompute(self):
        def image(name):return dict(relative_path=name,image_sha256=name,image_height=256,image_width=256)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parent=root/'old';cache=root/'old_cache';out=root/'trial'
            templates=[image(f'template{i}') for i in range(8)];queries=[image(f'query{i}') for i in range(2)]
            inputs=g.st.sealed_json(parent/'inputs.json',dict(identity='old',routes=dict(cache_root=str(cache)),
                backend=dict(distance_device='cpu',numpy=g.version('numpy'),scipy=g.version('scipy'),torch=None)))
            g.st.sealed_json(parent/'protocol.json',g.CONFIG)
            rms=g.st.sealed_json(parent/'source_RMS.json',dict(provider_seal='rms',RMS=[1.]*4,contract=dict(extraction={})))
            g.st.sealed_json(parent/'smoke.json',dict(identity='old',actual_features=True,full_256_regions=True,LP_pairs=16,RMS='rms',
                regions=[dict(image=r) for r in templates],matching=[dict(image=r) for r in queries]))
            refs={}
            for row in templates+queries:
                record=g.st.put(cache/'regions'/g.ip.key(row),dict(means=np.zeros((4,256,1280),np.float32),area=np.ones(256)/256),
                    dict(dependencies=dict(image=row,RMS='rms',extraction=g.st.digest({}),implementation=g.st.sha(Path(g.regions.__file__)))))
                refs[g.ip.key(row)]=record['seal']
            for q in queries:
                for t in templates:
                    g.st.sealed_json(parent/'pairs'/g.ip.key(q)/(g.ip.key(t)+'.json'),dict(query=q,template=t,
                        dependencies=dict(query=refs[g.ip.key(q)],template=refs[g.ip.key(t)],RMS='rms',distance_device='cpu',
                            protocol=g.st.digest(g.CONFIG),solver_implementation=g.st.sha(Path(g.transport.__file__))),
                        result=dict(success=True,objective=1.,free=1.)))
            snapshot={p:p.read_bytes() for tree in (parent,cache) for p in tree.rglob('*.json')}
            a=SimpleNamespace(parent_run=parent,cache_root=None,output_root=out)
            with patch.object(g.regions,'distance_matrix',return_value=np.ones((256,256),np.float32)) as distance:
                self.assertEqual(g.export(a)['new_features'],0);g.export(a)
                self.assertEqual(distance.call_count,16)
                distance.reset_mock() # Release mmap views retained only by the test spy on Windows.
            manifest,ps=g.load_problems(a);self.assertEqual(len(ps),16)
            self.assertFalse(manifest['LP_reference_recomputed'])
            del ps
            a.problem_root=out;a.output_root=root/'separate_mixed'
            other,ps=g.load_problems(a)
            self.assertEqual(manifest,other);self.assertEqual(len(ps),16)
            self.assertFalse(a.output_root.exists())
            del ps
            self.assertEqual(snapshot,{p:p.read_bytes() for p in snapshot})
            self.assertEqual(inputs,g.st.sealed_json(parent/'inputs.json'))
            # Same shapes, changed region provenance must not be re-exported as the original problem.
            r=g.st.read_json(parent/'source_RMS.json');r.pop('seal');r['provider_seal']='different'
            replacement=dict(r,seal=g.st.digest(r));g.st.write_json(parent/'source_RMS.json',replacement)
            with self.assertRaises(ValueError):g.source_case(a)


if __name__=='__main__':unittest.main()
