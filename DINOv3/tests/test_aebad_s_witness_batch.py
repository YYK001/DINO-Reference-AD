"""Actual vector/array-cache CPU paths; Torch exercised on CPU in a child only."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import numpy as np
from DINOv3.MADEqual.aebad_s_support_witness_segment_p0 import batch_witness as bw
from DINOv3.MADEqual.aebad_s_support_witness_segment_p0 import method, neighbors, supports, runner, autodl, storage as st

ROOT=Path(__file__).resolve().parents[2]
SCRATCH=ROOT/'DINOv3/MADEqual/aebad_s_support_witness_segment_p0/local_validation'


def fixture(seed=42,dimension=13,rows=13):
    rng=np.random.default_rng(seed)
    banks={c:rng.normal(size=(rows,dimension)).astype(np.float32) for c in range(1,9)}
    tables=neighbors.tables(banks,query_chunk=7,reference_chunk=6)
    nb={k:v[0] for k,v in tables.items()}
    edges=np.asarray([(a,i,b,j) for a,i,b,j in zip(rng.integers(1,5,160),rng.integers(0,rows,160),
        rng.integers(5,9,160),rng.integers(0,rows,160))],np.int64)
    return banks,nb,edges


class BatchGeometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.banks,cls.nb,cls.edges=fixture()

    def reference(self,engine,edges,excluded=()):
        proof=engine.compute(edges,excluded)
        for i,e in enumerate(bw.canonical_edges(edges)):
            scalar=method.witness(engine.banks,engine.neighbors,tuple(e),excluded)
            self.assertEqual(int(proof['source_mask'][i]),sum(1<<(c-1) for c in scalar['sources']))
            self.assertEqual(bw.EDGE_REASONS[int(proof['edge_reason'][i])],scalar['reason'])
            for third in scalar['third_images']:
                c=third['support_index']-1;ids=proof['candidates'][i,c];actual=ids[ids>=0]
                self.assertEqual(actual.tolist(),third['candidates'])
                for point in third['passing_points']:
                    k=int(np.flatnonzero(ids==point['patch_index'])[0])
                    self.assertAlmostEqual(float(proof['alpha'][i,c,k]),point['alpha'],places=12)
                    self.assertAlmostEqual(float(proof['rho'][i,c,k]),point['rho'],places=12)
        return proof

    def test_vector_against_original_scalar_candidates_sources_and_geometry(self):
        self.reference(bw.Engine(self.banks,self.nb),self.edges)

    def test_actual_1280_dimensional_geometry_against_scalar(self):
        b,n,e=fixture(dimension=1280,rows=7)
        self.reference(bw.Engine(b,n),e[:12])

    def test_whole_image_exclusion(self):
        engine=bw.Engine(self.banks,self.nb)
        p=self.reference(engine,self.edges,excluded=(4,)) if not (self.edges[:,0]==4).any() else self.reference(engine,self.edges[self.edges[:,0]!=4],excluded=(4,))
        self.assertTrue((p['candidates'][:,3]==-1).all())
        self.assertTrue(((p['source_mask'] & (1<<3))==0).all())
        with self.assertRaises(ValueError):engine.get_many([[1,0,2,0]],excluded=(1,))

    def test_swap_reorder_chunks_and_duplicate_requests(self):
        first=bw.Engine(self.banks,self.nb,batch_size=7);a,reason,_=first.get_many(self.edges)
        second=bw.Engine(self.banks,self.nb,batch_size=23);b,r,_=second.get_many(self.edges[::-1, [2,3,0,1]])
        np.testing.assert_array_equal(a,b[::-1]);np.testing.assert_array_equal(reason,r[::-1])
        count=first.new;again,_,_=first.get_many(self.edges);np.testing.assert_array_equal(a,again);self.assertEqual(count,first.new)

    def test_inclusive_boundaries_use_original_scalar_without_relaxation(self):
        for point in ((.5,0.),(1.5,0.),(1.,1/np.sqrt(3)),(.1,0.),(1.,3.)):
            banks={1:np.tile([0.,0.],(5,1)),2:np.tile([2.,0.],(5,1)),3:np.tile(point,(5,1))}
            nb={(a,c):np.tile(np.arange(5),(5,1)) for a in banks for c in banks if a!=c}
            engine=bw.Engine(banks,nb);self.reference(engine,[[1,0,2,0]])
            if point[0] in (.5,1.5):self.assertEqual(engine.scalar_fallback_edges,1)

    def test_degenerate_and_nonfinite_reasons(self):
        for which,expected in (('zero','zero_length_connection'),('nonfinite','nonfinite_endpoint')):
            b={1:np.zeros((5,3)),2:np.zeros((5,3)),3:np.ones((5,3))}
            if which=='nonfinite':b[1][0,0]=np.nan
            n={(a,c):np.tile(np.arange(5),(5,1)) for a in b for c in b if a!=c}
            p=bw.Engine(b,n).compute([[1,0,2,0]])
            self.assertEqual(bw.EDGE_REASONS[int(p['edge_reason'][0])],expected)

    def test_gate_counts_and_reduction_match_actual_legacy_path(self):
        rng=np.random.default_rng(3);target=rng.uniform(.1,2,(8,90)).astype(np.float32)
        idx=rng.integers(0,13,(8,90),dtype=np.int64)
        class Scalar:
            def get(_,e,excluded):return method.witness(self.banks,self.nb,e,excluded)
        a,old=supports.candidate_gates(target,idx,Scalar())
        b,new=supports.candidate_gates(target,idx,bw.Engine(self.banks,self.nb,batch_size=9))
        np.testing.assert_array_equal(a,b)
        for k in ('unique_candidate_connections','unique_accepted_connections','effective_witness_sources','rejection_reasons'):
            self.assertEqual(old[k],new[k])
        u=target.min(0);s=u*.8
        r1,_=method.reduction(u,s,a);r2,_=method.reduction(u,s,b)
        for k in r1:np.testing.assert_array_equal(r1[k],r2[k])

    def test_empty_batch(self):
        p=bw.Engine(self.banks,self.nb).get_many(np.empty((0,4),np.int64));self.assertEqual(len(p[0]),0)


class ArrayCache(unittest.TestCase):
    def setUp(self):
        SCRATCH.mkdir(exist_ok=True);self.temp=tempfile.TemporaryDirectory(dir=SCRATCH);self.root=Path(self.temp.name)
        self.b,self.n,self.e=fixture()
    def tearDown(self):self.temp.cleanup()

    def engine(self,**kwargs):return bw.Engine(self.b,self.n,path=self.root/'proofs',identity='fixed',neighbor_identity='nb',batch_size=19,**kwargs)

    def test_actual_array_proofs_resume_and_deduplicate(self):
        a=self.engine();masks,_,_=a.get_many(self.e);count=a.new
        b=self.engine();again,_,_=b.get_many(self.e[::-1]);np.testing.assert_array_equal(again,masks[::-1]);self.assertEqual(b.new,0)
        self.assertEqual(len(b.memory),count)
        p=next((self.root/'proofs').glob('*.npz'))
        with np.load(p) as z:self.assertEqual(z['alpha'].shape,(19,8,10));self.assertIn('rho',z)

    def test_tampered_proof_is_not_enrolled(self):
        self.engine().get_many(self.e[:2]);p=next((self.root/'proofs').glob('*.npz'));p.write_bytes(p.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'modified'):self.engine()

    def test_orphaned_proof_and_old_config_rejected(self):
        self.engine().get_many(self.e[:2]);p=next((self.root/'proofs').glob('*.npz'));p.unlink()
        with self.assertRaisesRegex(ValueError,'incomplete'):self.engine()

    def test_exclusion_namespace_isolated(self):
        engine=self.engine();a=engine.get_many([[1,0,2,0]])[0]
        engine.get_many([[1,0,2,0]],excluded=(3,));self.assertEqual(engine.new,2)
        self.assertEqual(self.engine().get_many([[1,0,2,0]])[0][0],a[0])

    def test_parallel_image_freeze_matches_serial_real_maps_scores_and_npz(self):
        from types import SimpleNamespace
        from DINOv3.MADEqual.aebad_s_support_witness_segment_p0 import prediction
        b=SimpleNamespace(identity='frozen',frozen={'synthetic':True},protocol=method.PROTOCOL,protected_paths=[],
            medians=np.zeros(4,np.float32),scales=np.ones(4,np.float32),
            state={k:dict(medians=[0.]*4,scales=[1.]*4) for k in ('peak','local')},references={})
        b.samples=[dict(domain='background',sample_id=f'test_{i:04d}',relative_path=f'test/good/background/{i}.png') for i in range(6)]
        positions=np.array([8,0,4,7,2,1,5,3,6],np.int64)
        u=np.linspace(.3,1.2,36,dtype=np.float32).reshape(4,9);s=u*.8
        raw=dict(positions=positions,union_distance=u,segment_distance=s)
        meta=dict(grid=[3,3],output_shape=[6,6]);origin={'synthetic_parent':'unchanged'}
        b.load=lambda sample:(raw,meta,origin)
        for sample in b.samples:
            for branch,value in zip(method.BASES,(u,s)):
                sc=prediction.score(b,value,positions,meta)
                b.references['background',sample['sample_id'],branch]=dict(image_score=sc['CH1'],current_check_score=sc['CURRENT'])
        outputs=[self.root/'serial',self.root/'parallel']
        for out in outputs:
            st.initialize(out,b)
            for sample in b.samples:
                for i,l in enumerate(method.LAYERS):
                    arrays,_=method.reduction(u[i],s[i],np.arange(9)%2==0)
                    st.seal(prediction.layer_path(out,'background',l,sample['sample_id']),b.identity,
                        dict(positions=positions,**arrays),dict(origin=origin,reduction={},counts={}))
        a=prediction.freeze_domain(b,outputs[0],'background',workers=1)
        c=prediction.freeze_domain(b,outputs[1],'background',workers=3)
        self.assertEqual(a,c)
        for sample in b.samples:
            x,m,_=prediction.load_record(b,outputs[0],'background',sample['sample_id'])
            y,n,_=prediction.load_record(b,outputs[1],'background',sample['sample_id'])
            self.assertEqual(m,n)
            for key in x:np.testing.assert_array_equal(x[key],y[key])


class TorchAndScheduling(unittest.TestCase):
    def test_cpu_image_queue_accepts_15_workers_and_preserves_order(self):
        from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.execution import bounded_images
        lock=threading.Lock();active=0;peak=0
        def work(x):
            nonlocal active,peak
            with lock:active+=1;peak=max(peak,active)
            time.sleep(.03)
            with lock:active-=1
            return x*x
        result=list(bounded_images(range(30),work,15))
        self.assertEqual(result,[x*x for x in range(30)]);self.assertGreater(peak,8);self.assertLessEqual(peak,15)

    def test_actual_torch_cpu_kernels_match_reference_no_cuda_called(self):
        code=r'''
import numpy as np, torch
from unittest.mock import patch
from DINOv3.tests.test_aebad_s_witness_batch import fixture
from DINOv3.MADEqual.aebad_s_support_witness_segment_p0 import batch_witness as b, neighbors
torch.set_num_threads(1)
with patch.object(torch.cuda,'set_device',side_effect=AssertionError('no GPU')):
    banks,nb,e=fixture()
    a=b.Engine(banks,nb).get_many(e)
    c=b.Engine(banks,nb,device='torch:cpu',batch_size=11).get_many(e)
    np.testing.assert_array_equal(a[0],c[0]);np.testing.assert_array_equal(a[1],c[1])
    rng=np.random.default_rng(4);q=rng.normal(size=(7,11));r=rng.normal(size=(19,11));r[5]=r[1]
    for qc,rc in ((1,1),(3,4),(20,40)):
        got,d=neighbors.torch_topk(q,r,k=5,query_chunk=qc,reference_chunk=rc,device='cpu')
        ref,rd=neighbors.topk(q,r,k=5,query_chunk=qc,reference_chunk=rc)
        np.testing.assert_array_equal(got,ref);np.testing.assert_allclose(d,rd,atol=1e-14,rtol=1e-14)
    idx,d=neighbors.torch_topk(np.zeros((2,3)),np.ones((9,3)),query_chunk=2,reference_chunk=2,device='cpu')
    np.testing.assert_array_equal(idx,np.tile(np.arange(5),(2,1)))
    for point in ((.5,0.),(1.5,0.),(1.,1/np.sqrt(3))):
        banks={1:np.tile([0.,0.],(5,1)),2:np.tile([2.,0.],(5,1)),3:np.tile(point,(5,1))}
        nb={(a,c):np.tile(np.arange(5),(5,1)) for a in banks for c in banks if a!=c}
        x=b.Engine(banks,nb).get_many([[1,0,2,0]])
        y=b.Engine(banks,nb,device='torch:cpu').get_many([[1,0,2,0]])
        np.testing.assert_array_equal(x[0],y[0]);np.testing.assert_array_equal(x[1],y[1])
print('REAL_TORCH_CPU_PATH_PASSED; CUDA_NOT_EXECUTED',flush=True)
'''
        result=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_CPU_preflight_import_still_torch_free(self):
        code='import sys; from DINOv3.MADEqual.aebad_s_support_witness_segment_p0 import runner,batch_witness; assert "torch" not in sys.modules; assert runner.parser().parse_args([]).stage=="preflight"'
        r=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)

    def test_real_worker_progress_tee_and_error_code(self):
        SCRATCH.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=SCRATCH) as root:
            log=Path(root)/'worker.log';capture=io.StringIO()
            with contextlib.redirect_stdout(capture):
                rc=runner.run_worker([sys.executable,'-u','-c','import sys; print("WORKER_PROGRESS",flush=True); sys.exit(3)'],log)
            self.assertEqual(rc,3);self.assertIn('WORKER_PROGRESS',capture.getvalue());self.assertIn('WORKER_PROGRESS',log.read_text())

    def dispatch(self,devices,workers=3,stage='predict'):
        SCRATCH.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=SCRATCH) as root:
            a=runner.parser().parse_args([stage,'--devices',*devices,'--cpu-workers',str(workers)])
            calls=[];lock=threading.Lock();active=0;peak=0
            def fake(cmd,log):
                nonlocal active,peak
                with lock:active+=1;peak=max(peak,active);calls.append(cmd)
                time.sleep(.02)
                with lock:active-=1
                return 0
            with patch.object(runner,'run_worker',side_effect=fake),contextlib.redirect_stdout(io.StringIO()):
                result=runner.parallel_jobs(a,None,Path(root),[(d,l) for d in ('background','illumination','view') for l in (7,15,23,31)])
            return calls,peak,result

    def test_explicit_dual_gpu_reaches_all_layer_workers(self):
        for stage in ('support-loo','predict'):
            calls,peak,r=self.dispatch(['cuda:0','cuda:1'],stage=stage)
            self.assertEqual(len(calls),12);self.assertEqual(peak,2)
            self.assertEqual({c[c.index('--devices')+1] for c in calls},{'cuda:0','cuda:1'})
            self.assertTrue(all('--worker' in c and '--layers' in c and '--witness-batch' in c for c in calls))

    def test_single_gpu_is_one_lane_and_cpu_really_uses_workers(self):
        _,peak,_=self.dispatch(['cuda:0']);self.assertEqual(peak,1)
        _,peak,result=self.dispatch(['cpu'],workers=3);self.assertEqual(peak,3);self.assertEqual(result['process_lanes'],3)

    def test_failed_worker_stops_picking_more_jobs(self):
        SCRATCH.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=SCRATCH) as root:
            a=runner.parser().parse_args(['predict','--devices','cuda:0'])
            for kwargs,error in ((dict(return_value=2),RuntimeError),(dict(side_effect=OSError('launch failed')),OSError)):
                with patch.object(runner,'run_worker',**kwargs) as worker,contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(error):
                        runner.parallel_jobs(a,None,Path(root),[('background',7),('background',15)])
                self.assertEqual(worker.call_count,1)

    def test_launcher_passes_devices_and_workers_to_previously_serial_stages(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as root:
            calls=[]
            def fake(stage,cmd,path):
                calls.append((stage,cmd))
                if stage=='summarize':st.write_json(path/'experiment/status.json',dict(status='all_requested_stages_complete',all_metrics_complete=True,pending=[],numerical_failed_checks=0))
            with patch.object(autodl,'run',side_effect=fake),contextlib.redirect_stdout(io.StringIO()):
                rc=autodl.main(['--run-root',str(Path(root)/'run'),'--data',str(Path(root)/'data'),
                    '--checkpoint',str(Path(root)/'weights'),'--devices','cuda:0','cuda:1','--cpu-workers','15'])
            self.assertEqual(rc,0)
            for name in ('support-loo','predict'):
                cmd=next(c for s,c in calls if s==name)
                self.assertIn('cuda:1',cmd);self.assertEqual(cmd[cmd.index('--cpu-workers')+1],'15')


if __name__=='__main__':unittest.main()
