"""CPU regression for actual resident Exact, row gather and grouped training."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from DINOv3.MADEqual.aebad_s_reference_residual_segmentation import storage as st,inputs as ip


class Execution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch=torch;torch.set_num_threads(1)
        (ip.ROOT/'.tmp/reference_residual_tests').mkdir(parents=True,exist_ok=True)
    def test_resident_exact_preserves_values_indices_ties_and_query_upload_count(self):
        from DINOv3.MADEqual.aebad_s_reference_residual_segmentation.execution import ResidentExact
        from DINOv3.MADEqual.aebad_s_reference_residual_segmentation.retrieval import exact_with_audit,target_nearest
        rng=np.random.default_rng(3);q=rng.normal(size=(7,1280)).astype(np.float32);bank=rng.normal(size=(11,1280)).astype(np.float32)
        bank[6]=bank[0];q[0]=bank[0]
        old,idx,audit=exact_with_audit(q,bank,'cpu');engine=ResidentExact('cpu',bank.nbytes*2);qt=engine.query(q)
        for _ in range(2):
            new,ii,diag=exact_with_audit(q,bank,'cpu',search=engine,key='sealed-bank',query=qt)
            np.testing.assert_array_equal(new,old);np.testing.assert_array_equal(ii,idx);self.assertEqual(diag,audit)
        self.assertEqual(ii[0],0);self.assertEqual(engine.stats['reference_uploads'],1)
        self.assertEqual(engine.stats['reference_hits'],1);self.assertEqual(engine.stats['query_uploads'],1)
        old=target_nearest(q,bank,[0,5,11],'cpu')
        new=target_nearest(q,None,[0,5,11],'cpu',parts=[bank[:5],bank[5:]],search=engine,keys=['a','b'],query=qt)
        for a,b in zip(old[:2],new[:2]):np.testing.assert_array_equal(a,b)
        self.assertEqual(old[2]['support_audits'],new[2]['support_audits'])
    def test_resident_budget_eviction_zero_and_layout_identity(self):
        from DINOv3.MADEqual.aebad_s_reference_residual_segmentation.execution import ResidentExact
        b=np.ones((3,5),np.float32);engine=ResidentExact('cpu',b.nbytes)
        engine.reference(b,'a');engine.reference(b,'b');self.assertEqual(list(engine.cache),['b']);self.assertEqual(engine.bytes,b.nbytes)
        with self.assertRaisesRegex(ValueError,'layout changed'):engine.reference(b[:1],'b')
        engine=ResidentExact('cpu',0);engine.reference(b,'a');self.assertEqual(engine.bytes,0);self.assertFalse(engine.cache)
    def test_selected_row_gather_matches_compact_concatenation_without_allocating_it(self):
        from DINOv3.MADEqual.aebad_s_reference_residual_segmentation import retrieval
        base=ip.ROOT/'.tmp/reference_residual_tests';base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as tmp:
            root=Path(tmp);a=SimpleNamespace(run_root=root/'run',cache_root=root/'cache',device='cpu')
            rows=[dict(relative_path='support/'+str(i)) for i in range(2)]
            st.sealed_json(a.run_root/'catalog.json',dict(catalog=dict(source=[],supports=dict(background=rows))))
            bank=np.arange(6*1280,dtype=np.float32).reshape(6,1280);st.put(a.cache_root/'source_original',dict(L7=bank),{})
            targets=[bank[:3]+100,bank[:5]+200]
            for row,x in zip(rows,targets):st.put(ip.normal_path(a,row),dict(q7=x),{})
            refs=retrieval.References(a);si=np.array([5,0,2,1]);ti=np.array([7,0,3,2]);origins=[r['relative_path'] for r in rows]
            expected=np.concatenate(targets)[ti]
            with patch.object(retrieval.np,'concatenate',side_effect=AssertionError('no full K8 concat')):
                ss,tt,meta=refs.gather('formal',origins,7,si,ti)
            np.testing.assert_array_equal(ss,bank[si]);np.testing.assert_array_equal(tt,expected)
            self.assertEqual(meta['target_offsets'],[0,3,8]);self.assertEqual(meta['target_origins'],origins)
            del refs
    def test_shape_groups_preserve_per_image_losses_and_gradients(self):
        from DINOv3.MADEqual.aebad_s_reference_residual_segmentation.model import Head
        from DINOv3.MADEqual.aebad_s_reference_residual_segmentation.training import prepare_batch,backward_batch
        t=self.torch;t.manual_seed(6);old=Head(4,8);new=copy.deepcopy(old);samples=[]
        for i,(h,w) in enumerate(((6,6),(5,7),(6,6),(5,7))):
            soft=np.zeros((h*2,w*2),np.float32)
            if i%2:soft[3:5,3:5]=.3
            samples.append(dict(values=[t.randn(1,15,h,w) for _ in range(4)],soft=soft,origin=[0,0],grid=[h,w],shape=list(soft.shape),center=[0,0,h,w]))
        groups=prepare_batch(samples);self.assertEqual([len(g['rows']) for g in groups],[2,2])
        a=backward_batch(old,prepare_batch(samples,grouped=False),'cpu');b=backward_batch(new,groups,'cpu')
        t.testing.assert_close(a,b,rtol=1e-6,atol=1e-6)
        for x,y in zip(old.parameters(),new.parameters()):t.testing.assert_close(x.grad,y.grad,rtol=1e-4,atol=2e-6)
    def test_training_prepare_uses_same_four_contexts_indices_and_cached_origin_policy(self):
        from DINOv3.tests.test_aebad_s_reference_residual_segmentation import catalog
        from DINOv3.MADEqual.aebad_s_reference_residual_segmentation import retrieval,training_data
        base=ip.ROOT/'.tmp/reference_residual_tests'
        with tempfile.TemporaryDirectory(dir=base) as tmp:
            root=Path(tmp);a=SimpleNamespace(run_root=root/'run',cache_root=root/'cache',device='cpu',reference_cache_gib=.01)
            c=catalog();c['source']=c['source'][:25];row=c['source'][1];position=np.array([2,0,3,1]);rng=np.random.default_rng(12)
            st.sealed_json(a.run_root/'catalog.json',dict(catalog=c));st.sealed_json(a.run_root/'inputs.json',dict(identity='tiny'))
            for origin in ip.train_source(c)+sum(c['supports'].values(),[]):
                st.put(ip.normal_path(a,origin),{**{f'q{l}':rng.normal(size=(4,1280)).astype(np.float32) for l in (7,15,23,31)},'positions':position},dict(grid=[2,2]))
            st.put(a.cache_root/'source_folds/fold_B',{f'L{l}':rng.normal(size=(7,1280)).astype(np.float32) for l in (7,15,23,31)},{})
            q={l:np.array(st.get(ip.normal_path(a,row),(f'q{l}',))[0][f'q{l}']) for l in (7,15,23,31)}
            refs=retrieval.References(a);item=dict(origin=row['relative_path'],variant=-1,item_id='clean-source',row=row,status='normal');policies={}
            with patch.object(training_data,'query_policy',wraps=training_data.query_policy) as policy:
                for _ in range(2):
                    record=training_data.prepare_one(a,item,q,dict(grid=[2,2],positions=position),np.zeros((16,16),np.float32),refs,catalogue=c,policies=policies)
                self.assertEqual(policy.call_count,4)
            for phase in ('source','background','illumination','view'):
                actual,_=st.get(ip.item_path(a,item)/'target_nn'/phase);origins=record['contexts'][phase]['policy']['target_origins']
                bank,_,offsets=refs.target(origins,7);d,i,_=retrieval.target_nearest(q[7],bank,offsets,'cpu')
                np.testing.assert_array_equal(actual['d7'],d);np.testing.assert_array_equal(actual['i7'],i)
                del actual,bank
            self.assertEqual(refs.search.stats['query_uploads'],8)  # four layers per query, not per support/context
            del refs
    def test_smoke_projection_is_explicitly_partial(self):
        from DINOv3.MADEqual.aebad_s_reference_residual_segmentation.smoke import estimate
        report=estimate(dict(train_seconds_per_update=.1),{})
        self.assertAlmostEqual(report['components_hours']['head_training'],336000*.1/3600)
        self.assertIsNone(report['total_hours']);self.assertTrue(report['pending'])
        report=estimate(dict(prepared_jobs=[dict(seconds_per_update=.1,phase='source',formal_BF16_verified=False)]),{})
        self.assertNotIn('head_training',report['components_hours'])
    def test_constructed_CPU_smoke_is_isolated_and_never_completes_formal_run(self):
        from DINOv3.MADEqual.aebad_s_reference_residual_segmentation import runner
        base=ip.ROOT/'.tmp/reference_residual_tests'
        with tempfile.TemporaryDirectory(dir=base) as tmp:
            root=Path(tmp);run=root/'run';cache=root/'cache'
            code=runner.main(['smoke','--device','cpu','--run-root',str(run),'--cache-root',str(cache),'--smoke-updates','3'])
            self.assertEqual(code,0);report=st.read_json(next((run/'smoke').glob('*/smoke_report.json')))
            self.assertEqual(report['status'],'smoke_complete');self.assertFalse(report['GT_used'])
            self.assertIsNone(report['time_estimate']['total_hours'])
            self.assertFalse((run/'inputs.json').exists());self.assertFalse((run/'status.json').exists())
            self.assertFalse((run/'models').exists());self.assertFalse(cache.exists())
            self.assertFalse(self.torch.cuda.is_initialized())
