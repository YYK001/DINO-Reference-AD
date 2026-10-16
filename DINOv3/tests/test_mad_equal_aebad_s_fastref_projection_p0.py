"""NumPy math and explicit host/evaluator doubles; never real Torch or CUDA."""
from __future__ import annotations
import contextlib
import io
import json
import sys
import tarfile
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PIL import Image
from DINOv3.MADEqual.aebad_s_fastref_projection_p0 import projection,inputs,runtime,analysis,runner
from DINOv3.MADEqual.aebad_s_fastref_k8_p0 import refinement as math
from DINOv3.tests.test_mad_equal_aebad_s_fastref_k8_p0 import flow_fixture,prototype_fixture
from DINOv3.tests.test_mad_equal_aebad_s_target_local_segment import full_protocol_fixture
from DINOv3.tests.test_mad_equal_aebad_s_sparc_union_p0 import torch_fixture


class ProjectionMathTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.RandomState(42)
        self.q=rng.normal(size=(9,7)).astype(np.float32)
        self.p=rng.normal(size=(3,7)).astype(np.float32)
        m=self.p.astype(np.float64)/1280
        self.constant=dict(prototypes=self.p,factor=np.linalg.solve(m@m.T,m),prototype_identity='synthetic')

    def test_rectangular_projection_matches_independent_QR(self):
        basis,_=np.linalg.qr(self.p.astype(float).T,mode='reduced')
        expected=(self.q.astype(float)@basis)@basis.T
        actual,residual,cast,diag=projection.project(self.q,self.constant)
        np.testing.assert_allclose(actual,expected,atol=2e-7,rtol=2e-7)
        np.testing.assert_allclose(residual,np.linalg.norm(self.q.astype(float)-expected,axis=1)/1280,atol=1e-16)
        self.assertLess(diag['normal_equation_relative_residual'],1e-12)
        self.assertFalse(diag['OT_called']);self.assertEqual(cast.dtype,np.float64)

    def test_idempotence_and_residual_orthogonality(self):
        m=self.p.astype(float)/1280;x=self.q.astype(float)/1280
        a=(x@m.T)@self.constant['factor'];again=(a@m.T)@self.constant['factor']
        np.testing.assert_allclose(a,again,atol=1e-16)
        np.testing.assert_allclose((x-a)@m.T,0,atol=1e-20)

    def test_exact_reference_all_projection_NN_equals_self_residual(self):
        basis,_=np.linalg.qr(self.p.astype(float).T,mode='reduced');q=self.q.astype(float)
        a=(q@basis)@basis.T
        all_nn=np.linalg.norm(q[:,None]-a[None],axis=2).min(axis=1)
        np.testing.assert_allclose(all_nn,np.linalg.norm(q-a,axis=1),atol=1e-14,rtol=1e-14)

    def test_actual_order_cast_and_unit_conversion(self):
        actual,_,_,_=projection.project(self.q,self.constant)
        expected=(((self.q.astype(float)/1280)@(self.p.astype(float)/1280).T)@self.constant['factor'])*1280
        np.testing.assert_array_equal(actual,expected.astype(np.float32));self.assertTrue(actual.flags.c_contiguous)
        p=np.array([[1280.,0,0]],np.float32);m=p.astype(float)/1280
        q=np.array([[1280.,1280.,0]],np.float32)
        a,ideal,cast,_=projection.project(q,dict(prototypes=p,factor=np.linalg.solve(m@m.T,m)))
        np.testing.assert_array_equal(a,[[1280,0,0]]);self.assertEqual(ideal[0],1);self.assertEqual(cast[0],1)

    def test_no_OT_and_disabled_parent_is_P_not_projection(self):
        p=np.array([[1.,0,0],[0,1.,0]],np.float32);q=np.array([[.2,.3,5]],np.float32)
        constant=math.make_constants(p)
        with patch.object(math,'balanced_transport',side_effect=AssertionError('OT forbidden')),patch.object(math,'refine',wraps=math.refine) as refine:
            a,_,_,_=projection.project(q,constant);refine.assert_not_called()
            disabled,_=math.refine(q,constant,enabled=False)
        np.testing.assert_array_equal(disabled,p);np.testing.assert_allclose(a,[[.2,.3,0]],atol=1e-7)
        self.assertNotEqual(a.shape,disabled.shape)

    def test_query_independence_and_no_mutation(self):
        q=self.q.copy();p=self.p.copy();b=self.constant['factor'].copy()
        first=projection.project(q,self.constant)[0];projection.project(q*3,self.constant)
        np.testing.assert_array_equal(first,projection.project(q,self.constant)[0])
        np.testing.assert_array_equal(q,self.q);np.testing.assert_array_equal(p,self.p)
        np.testing.assert_array_equal(b,self.constant['factor'])

    def test_invalid_or_wrong_factor_rejected_without_fallback(self):
        for q in (self.q.astype(float),np.empty((0,7),np.float32),self.q*np.nan):
            with self.assertRaises(ValueError):projection.project(q,self.constant)
        with self.assertRaisesRegex(ValueError,'residual'):projection.project(self.q,dict(self.constant,factor=self.constant['factor']*2))

    def test_target_before_min_distinct_from_union_after_min(self):
        s=np.ones((4,3),np.float32)
        old=dict(refined_distance=s*4,fastref_union_distance=s)
        new=dict(projection_distance=s*2,projection_union_distance=s)
        rows,_=analysis.distance_comparison(dict(sample_id='toy',label=0),new,old)
        self.assertTrue(all(r['mean_signed']==-2 for r in rows if r['kind']=='Target_only'))
        self.assertTrue(all(r['max_abs']==0 for r in rows if r['kind']=='Source_union'))
        self.assertEqual(len(rows),10)

    def test_three_metric_references_and_direction(self):
        metrics=[dict(branch=b,**{k:v for k in inputs.parent.METRIC_KEYS}) for b,v in zip(inputs.BRANCHES,(.1,.7,.5,.6,.4))]
        rows=analysis.metric_comparisons(metrics)
        self.assertEqual([r['reference'] for r in rows],list(inputs.REFERENCES))
        for row,delta in zip(rows,(-.2,-.1,-.3)):
            for key in inputs.parent.METRIC_KEYS:self.assertAlmostEqual(row['delta_'+key],delta)
        with self.assertRaises(ValueError):analysis.metric_comparisons(metrics[:-1])


class IdentityTests(unittest.TestCase):
    def test_default_preflight_no_torch_no_output(self):
        self.assertNotIn('torch',sys.modules)
        with tempfile.TemporaryDirectory() as td,contextlib.redirect_stdout(io.StringIO()) as out:
            destination=Path(td)/'new'
            rc=runner.main(['--parent-dir',str(Path(td)/'missing'),'--output-dir',str(destination)])
            self.assertEqual(rc,2);self.assertFalse(destination.exists())
            self.assertFalse(json.loads(out.getvalue())['torch_imported'])
        self.assertNotIn('torch',sys.modules)

    def test_parent_prototypes_and_order_loaded_from_same_files(self):
        with tempfile.TemporaryDirectory() as td:
            p,prepared=prototype_fixture(Path(td))
            first=runtime.original.load_constants(p,prepared);second=runtime.original.load_constants(p,prepared)
            for layer in inputs.LAYERS:
                np.testing.assert_array_equal(first[layer]['prototypes'],second[layer]['prototypes'])
                np.testing.assert_array_equal(first[layer]['indices'],second[layer]['indices'])
                self.assertEqual(first[layer]['prototype_identity'],prepared['prototype_ids'][str(layer)])
            file=Path(p['paths']['output'])/'prototypes/L7.npz'
            with np.load(file) as f: arrays={k:f[k].copy() for k in f.files}
            arrays['prototypes']=arrays['prototypes'][::-1].copy()
            with file.open('wb') as f:np.savez_compressed(f,**arrays)
            with self.assertRaises(ValueError):runtime.original.load_constants(p,prepared)

    def test_approved_archive_records_and_saved_solver_arrays_read_only(self):
        file=Path(__file__).resolve().parents[2]/'downloads'/f'{inputs.PARENT_NAME}_results_20261014.tar.gz'
        if not file.is_file():self.skipTest('local approved parent package absent')
        wanted=set(inputs.APPROVED);prefix=inputs.PARENT_NAME+'/';found=set()
        with tarfile.open(file,'r|gz') as archive:
            for item in archive:
                name=item.name.removeprefix(prefix)
                if name in wanted:
                    blob=archive.extractfile(item).read()
                    self.assertEqual(hashlib.sha256(blob).hexdigest(),inputs.APPROVED[name],name);found.add(name)
                elif name.startswith('prototypes/') and name.endswith('.npz'):
                    with np.load(io.BytesIO(archive.extractfile(item).read()),allow_pickle=False) as f:
                        p=f['prototypes'];factor=f['factor'];m=p.astype(float)/1280
                        self.assertEqual(p.shape,(512,1280));self.assertEqual(factor.dtype,np.float64)
                        relative=np.linalg.norm((m@m.T)@factor-m)/(np.linalg.norm(m@m.T)*np.linalg.norm(factor)+np.linalg.norm(m))
                        self.assertLess(relative,1e-10)
        self.assertEqual(found,wanted);self.assertNotIn('torch',sys.modules)

    def test_output_protects_parent_and_other_inputs(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'parent'
            for out in (p,p/'child',Path(td)):
                with self.assertRaises(ValueError):inputs.guard(p,out)

    def test_old_or_modified_parent_not_reapproved(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'parent';root.mkdir();(root/'inputs.json').write_text('{}',encoding='utf8')
            out=Path(td)/'new'
            report=inputs.preflight(SimpleNamespace(parent_dir=str(root),output_dir=str(out)))
            self.assertIn('modified parent: inputs.json',report['errors'])
            self.assertEqual(report['status'],'blocked');self.assertFalse(out.exists());self.assertNotIn('torch',sys.modules)

    def test_source_scope_and_support_exclusion_rejected_if_changed(self):
        cases=[dict(sample_id=s,relative_path=p,label=int(i>=10)) for i,(s,p) in enumerate(inputs.replay.CASES)]
        extra=[dict(sample_id=f'x{i}',relative_path=f'query{i}',label=int(i>=75)) for i in range(285)]
        supports=[dict(support_index=i,relative_path=f'support{i}') for i in range(1,9)]
        p=dict(capture=dict(samples=cases+extra,selected_cases=cases,supports=supports))
        inputs.check_scope(p)
        supports[0]['relative_path']=cases[0]['relative_path']
        with self.assertRaises(ValueError):inputs.check_scope(p)


class IntegrationTests(unittest.TestCase):
    """297 tiny caches, real new validation/analysis, fake pixel backend and plots."""
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        cls.p,cls.prepared,cls.backend,_=flow_fixture(cls.root)
        cls.parent_root=Path(cls.p['paths']['output'])
        cls.parent_state=inputs.read_json(cls.parent_root/'infer_status.json')
        cls.prepared['prototype_ids']={str(l):f'synthetic_{l}' for l in inputs.LAYERS}
        cls.cached={};scores=[];instances=[];labels=[]
        for sample,record in zip(cls.p['capture']['samples'],cls.parent_state['records'],strict=True):
            arrays,meta=runtime.parent_raw(cls.p,sample,record,cls.prepared)
            cls.cached[sample['sample_id']]=(arrays,meta)
            raws=dict(SOURCE=arrays['source_distance'],UNION_K8=np.minimum(arrays['source_distance'],arrays['target_full_distance']),
                      UNION_TARGET512=arrays['prototype_union_distance'],FASTREF_TARGET512_SOURCE=arrays['fastref_union_distance'])
            maps=runtime.original.maps_for(raws,arrays['positions'],meta,cls.backend)
            for b,v in maps.items():scores.append(dict(branch=b,sample_id=sample['sample_id'],image_score=float(v.max()),label=sample['label']))
            mask=None
            if sample['label']:
                with Image.open(Path(cls.p['paths']['data'])/inputs.parent.shared.gt_relative(sample['relative_path'])) as image:mask=np.asarray(image)>0
            instances.extend(analysis.instance_rows(sample,maps,mask,tuple(meta['grid']),cls.backend)[0]);labels.append(sample['label'])
        directory=cls.parent_root/'evaluation';directory.mkdir()
        inputs.replay.write_csv(directory/'per_image_scores.csv',scores)
        inputs.replay.write_csv(directory/'defect_instances.csv',instances)
        from sklearn.metrics import roc_auc_score,average_precision_score
        cls.metrics=[dict(branch=b,image_count=297,image_AUROC=roc_auc_score(labels,[r['image_score'] for r in scores if r['branch']==b]),
                         image_AP=average_precision_score(labels,[r['image_score'] for r in scores if r['branch']==b]),pixel_AUROC=.5,pixel_AP=.5,AUPRO=.5,
                         result_origin='reused_parent_result') for b in inputs.parent.BRANCHES]

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def fixture(self,name):
        root=self.root/name;root.mkdir()
        data=dict(tool=inputs.TOOL,identity='synthetic-new-identity',paths=dict(self.p['paths'],parent=str(self.parent_root),output=str(root)),
                  frozen=dict(prototype_ids=self.prepared['prototype_ids'],cpu_workers=1),required_files=[])
        inputs.replay.atomic_json(root/'inputs.json',data)
        records=[];checks=[];diag=[]
        for sample in self.p['capture']['samples']:
            sid=sample['sample_id'];old,meta=self.cached[sid]
            target=old['refined_distance']*.95
            arrays=dict(source_distance=old['source_distance'],prototype_distance=old['prototype_distance'],projection_distance=target,
                        projection_union_distance=np.minimum(old['source_distance'],target),projection_index=np.zeros_like(target,np.int64),
                        positions=old['positions'],grid=old['grid'],processed_shape=old['processed_shape'],
                        self_projection_residual=target.astype(float),self_cast_projection_residual=target.astype(float))
            m=dict(tool=inputs.TOOL,identity=data['identity'],parent_identity=inputs.PARENT_ID,prototype_ids=self.prepared['prototype_ids'],
                   sample_id=sid,relative_path=sample['relative_path'],image_sha256=meta['image_sha256'],geometry=meta['geometry'],grid=meta['grid'],
                   output_shape=meta['output_shape'],distance_divisor=1280,query_forward_calls=1,OT_called=False,GT_decoded=False,
                   query_features_persisted=False,projection_vectors_persisted=False,parent_raw_sha256=inputs.replay.sha256(self.parent_root/'raw'/f'{sid}.npz'))
            file=root/'raw'/f'{sid}.npz';runtime.original.host_runtime.save_npz(file,metadata=np.asarray(json.dumps(m)),**arrays)
            records.append(dict(sample_id=sid,**inputs.parent.file_records(root,[file])[0]))
            for l in inputs.LAYERS:diag.append(dict(sample_id=sid,layer=l,OT_called=False,query_rows=4,projection_rows=4,
                prototype_identity=self.prepared['prototype_ids'][str(l)],normal_equation_relative_residual=0.))
        for sid,kind,layer in sorted(runtime.expected_checks(self.p['capture']['samples'])):
            checks.append(dict(sample_id=sid,check_kind=kind,layer=layer,verified=True))
        inputs.replay.write_csv(root/'infer_verification.csv',checks);inputs.replay.atomic_json(root/'infer_projection_diagnostics.json',diag)
        state=dict(tool=inputs.TOOL,identity=data['identity'],status='predictions_frozen',images=297,records=records,query_forward_calls=297,
                   OT_called=False,GT_decoded=False,inputs_sha256=inputs.replay.sha256(root/'inputs.json'),
                   verification_sha256=inputs.replay.sha256(root/'infer_verification.csv'),diagnostics_sha256=inputs.replay.sha256(root/'infer_projection_diagnostics.json'))
        inputs.replay.atomic_json(root/'infer_status.json',state)
        return data,state

    def test_full_analysis_only_one_metric_evaluation_and_twelve_figures(self):
        data,_=self.fixture('complete');evaluator=full_protocol_fixture()
        with patch('DINOv3.MADEqual.aebad_s_sparc_union_p0.runtime.evaluation_device'),\
             patch.object(analysis,'MapBackend',return_value=self.backend),patch.object(analysis,'runtime_evaluator',return_value=evaluator),\
             patch.object(inputs,'historical_metrics',return_value=self.metrics),patch.object(evaluator,'_evaluate',wraps=evaluator._evaluate) as metric,\
             patch('DINOv3.MADEqual.aebad_s_fastref_projection_p0.visualization.plot_case') as figures:
            state=analysis.evaluate(data,self.p,self.prepared,self.parent_state,'explicit-fake-device')
        self.assertEqual(state['status'],'analysis_complete',state);self.assertEqual(metric.call_count,1);self.assertEqual(figures.call_count,12)
        self.assertEqual(state['historical_pixel_evaluations'],0);self.assertFalse(state['OT_called'])
        root=Path(data['paths']['output'])/'evaluation'
        metrics=inputs.replay.read_csv(root/'metrics.csv')
        self.assertEqual([r['result_origin'] for r in metrics],['reused_parent_result']*4+['new_projection_calculation'])
        self.assertEqual(len(inputs.replay.read_csv(root/'distance_differences.csv')),2970)
        rankings=inputs.replay.read_csv(root/'ranking_summary.csv')
        self.assertEqual({int(r['normal_count']) for r in rankings},{10,75,85})

    def test_offline_consistent_backend_shift_cannot_complete(self):
        data,_=self.fixture('backend_shift');method=self.backend.runner._method_map
        with patch.object(self.backend.runner,'_method_map',side_effect=lambda *a,**kw:method(*a,**kw)+1),\
             patch('DINOv3.MADEqual.aebad_s_sparc_union_p0.runtime.evaluation_device'),\
             patch.object(analysis,'MapBackend',return_value=self.backend),patch.object(analysis,'runtime_evaluator',return_value=full_protocol_fixture()):
            state=analysis.evaluate(data,self.p,self.prepared,self.parent_state,'fake')
        self.assertEqual(state['status'],'partial_evidence');self.assertFalse(state['pixel_metrics_complete'])
        self.assertGreater(state['numerical_failed_checks'],0)

    def test_deleted_cache_stops_before_GT_or_evaluator(self):
        data,state=self.fixture('missing_raw');root=Path(data['paths']['output'])
        (root/state['records'][0]['file']).unlink()
        with patch.object(analysis,'runtime_evaluator') as metric,self.assertRaises((ValueError,FileNotFoundError)):
            analysis.evaluate(data,self.p,self.prepared,self.parent_state,'fake')
        metric.assert_not_called();self.assertFalse((root/'evaluation').exists())

    def test_rehashed_incomplete_prediction_cannot_pass(self):
        data,state=self.fixture('incomplete');state['records'].pop()
        inputs.replay.atomic_json(Path(data['paths']['output'])/'infer_status.json',state)
        with self.assertRaisesRegex(ValueError,'complete frozen'):runtime.verify_stage(data,self.p,self.prepared,self.parent_state)

    def test_rehashed_wrong_diagnostic_row_count_rejected(self):
        data,state=self.fixture('wrong_rows');root=Path(data['paths']['output'])
        file=root/'infer_projection_diagnostics.json';diagnostics=inputs.read_json(file)
        diagnostics[0].update(query_rows=5,projection_rows=5);inputs.replay.atomic_json(file,diagnostics)
        state['diagnostics_sha256']=inputs.replay.sha256(file);inputs.replay.atomic_json(root/'infer_status.json',state)
        with self.assertRaisesRegex(ValueError,'query row count'):runtime.verify_stage(data,self.p,self.prepared,self.parent_state)

    def test_rehashed_target_union_confusion_rejected(self):
        data,state=self.fixture('union_confusion');sample=self.p['capture']['samples'][0]
        file=Path(data['paths']['output'])/state['records'][0]['file']
        with np.load(file) as f: arrays={k:f[k].copy() for k in f.files}
        arrays['projection_union_distance']=arrays['source_distance']*2
        with file.open('wb') as f:np.savez_compressed(f,**arrays)
        state['records'][0]['sha256']=inputs.replay.sha256(file)
        inputs.replay.atomic_json(Path(data['paths']['output'])/'infer_status.json',state)
        with self.assertRaisesRegex(ValueError,'Target-only'):runtime.verify_stage(data,self.p,self.prepared,self.parent_state)

    def test_actual_smoke_control_one_forward_no_OT_and_all_projection_search(self):
        # Explicit doubles replace model, extraction, Source/P searches and CUDA.
        # Production project(), orchestration, serialization and stage verification run.
        root=self.root/'smoke_control';root.mkdir()
        data=dict(tool=inputs.TOOL,identity='synthetic-smoke',paths=dict(self.p['paths'],parent=str(self.parent_root),output=str(root)),
                  frozen=dict(prototype_ids=self.prepared['prototype_ids'],cpu_workers=4),required_files=[])
        inputs.replay.atomic_json(root/'inputs.json',data)
        fake,_=torch_fixture();current={};calls=[];searches=[]
        p=np.zeros((3,1280),np.float32);p[:,:3]=np.eye(3,dtype=np.float32)*1280
        m=p.astype(float)/1280
        constants={l:dict(prototypes=p.copy(),factor=np.linalg.solve(m@m.T,m),prototype_identity=self.prepared['prototype_ids'][str(l)]) for l in inputs.LAYERS}
        model=SimpleNamespace(prototype_embeddings={l:('source',l) for l in inputs.LAYERS})
        def query(model,image,device,timing):
            sid=self.p['capture']['selected_cases'][len(calls)]['sample_id'];current['sid']=sid;calls.append(sid)
            arrays,meta=self.cached[sid]
            q=np.zeros((4,1280),np.float32);q[:,0]=np.arange(4)*1280;q[:,3]=1280
            queries={}
            for l in inputs.LAYERS:
                queries[l]=q.copy();queries[l][:,4]=l
            return queries,arrays['positions'],tuple(meta['grid']),(32,32)
        def nn(q,bank,*,compact=False):
            old=self.cached[current['sid']][0]
            if isinstance(bank,tuple):return old['source_distance'][inputs.LAYERS.index(bank[1])].copy(),np.zeros(4,np.int64)
            if bank.shape[0]==3:
                return old['prototype_distance'][inputs.LAYERS.index(int(q.numpy()[0,4]))].copy(),np.zeros(4,np.int64)
            searches.append((q.shape,bank.shape,compact))
            distance=np.linalg.norm(q.numpy().astype(float)[:,None]-bank.numpy().astype(float)[None],axis=2)
            return distance.min(1).astype(np.float32)/1280,distance.argmin(1).astype(np.int64)
        # Tiny fixture has identical layers, so all baseline arrays are reproduced.
        geometry=self.cached[self.p['capture']['selected_cases'][0]['sample_id']][1]['geometry']
        p_data=dict(self.p,paths=dict(self.p['paths'],memory='unused-fixture'),frozen=dict(self.p['frozen'],host_config='fake'))
        with patch.dict(sys.modules,{'torch':fake,'DINOv3.MADEqual.aebad_s_target_fewshot.runner':SimpleNamespace(_load_memory=lambda *a,**kw:True)}),\
             patch.object(runtime.original.host_runtime,'load_host',return_value=(model,'fake')),\
             patch.object(runtime.original,'load_constants',return_value=constants),\
             patch.object(runtime.original,'query_forward',side_effect=query),patch.object(runtime.original,'host_nn',side_effect=nn),\
             patch('DINOv3.MADEqual.aebad_s_union_diagnostics.spatial.MapBackend',return_value=self.backend),\
             patch('DINOv3.MADEqual.aebad_s_target_support_trace.analysis.image_geometry',return_value=geometry),\
             patch.object(math,'refine',side_effect=AssertionError('OT path forbidden')),\
             patch.object(math,'balanced_transport',side_effect=AssertionError('OT forbidden')),contextlib.redirect_stdout(io.StringIO()):
            state=runtime.inference(data,p_data,self.prepared,self.parent_state,'fake',smoke=True)
            self.assertEqual(len(calls),12);self.assertEqual(len(searches),48)
            self.assertTrue(all(shape==(4,1280) and bank==(4,1280) and compact for shape,bank,compact in searches))
            calls.clear();searches.clear()
            failed_root=self.root/'smoke_late_failure';failed_root.mkdir()
            failed_data=dict(data,paths=dict(data['paths'],output=str(failed_root)))
            inputs.replay.atomic_json(failed_root/'inputs.json',failed_data)
            with patch.object(runtime,'verify_stage',side_effect=ValueError('synthetic late integrity failure')),self.assertRaises(ValueError):
                runtime.inference(failed_data,p_data,self.prepared,self.parent_state,'fake',smoke=True)
            self.assertEqual(inputs.read_json(failed_root/'smoke_status.json')['status'],'stage_failed')
        self.assertEqual(state['status'],'smoke_verified');self.assertEqual(len(calls),12);self.assertEqual(len(searches),48)
        self.assertTrue(all(shape==(4,1280) and bank==(4,1280) and compact for shape,bank,compact in searches))
        self.assertFalse(state['OT_called']);self.assertNotIn('torch',sys.modules)

    def test_missing_GT_remains_partial(self):
        data,_=self.fixture('missing_GT');sample=next(s for s in self.p['capture']['samples'] if s['label'])
        gt=Path(self.p['paths']['data'])/inputs.parent.shared.gt_relative(sample['relative_path']);original=gt.read_bytes();gt.unlink()
        try:
            evaluator=full_protocol_fixture()
            with patch('DINOv3.MADEqual.aebad_s_sparc_union_p0.runtime.evaluation_device'),\
                 patch.object(analysis,'MapBackend',return_value=self.backend),patch.object(analysis,'runtime_evaluator',return_value=evaluator),\
                 patch.object(evaluator,'_evaluate') as metric:
                state=analysis.evaluate(data,self.p,self.prepared,self.parent_state,'fake')
            self.assertEqual(state['status'],'partial_evidence');self.assertIn(sample['sample_id'],state['missing_gt_ids']);metric.assert_not_called()
        finally:gt.write_bytes(original)


if __name__=='__main__':unittest.main()
