"""Small CPU fixtures; no environment installs, backbone, CUDA or GT fitting."""
import unittest,tempfile,subprocess,sys,json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from DINOv3.MADEqual.aebad_s_target_loo_layer_weight_p0 import numerics as nm,inputs as ip,pipeline as pp
from DINOv3.MADEqual.aebad_s_target_loo_layer_weight_p0.runner import parser,normalize
from DINOv3.MADEqual.aebad_s_target_loo_layer_weight_p0.protocol import NEW


class WeightsTests(unittest.TestCase):
    def test_positive_part_before_square_and_lower_target_not_penalized(self):
        s=np.broadcast_to(np.arange(66,dtype=float)[:,None,None],(66,4,3)).copy()
        t=np.full((8,4,3),-100.)
        anchor,ss,tt=nm.moments(s,t)
        np.testing.assert_array_equal(anchor,np.full((4,3),32.5));np.testing.assert_array_equal(tt,0)
        np.testing.assert_allclose(ss,np.mean(np.maximum(np.arange(66)-32.5,0)**2))
        self.assertLess(ss[0,0],np.mean((np.arange(66)-32.5)**2))

    def test_gamma_zero_independent_target_and_equal_moments(self):
        ss=np.arange(1,13,dtype=float).reshape(4,3)
        a=nm.weights(ss,np.full((4,3),np.nan),0)[0]
        b=nm.weights(ss,ss,.5)[0]
        np.testing.assert_array_equal(a,b)

    def test_floors_positive_finite_normalized(self):
        for g in (0.,.5):
            w,s,f=nm.weights(np.zeros((4,3)),np.zeros((4,3)),g)
            self.assertTrue(f.all());self.assertTrue((s==1e-8).all());self.assertTrue(np.isfinite(w).all());self.assertTrue((w>0).all())
            np.testing.assert_array_equal(w.sum(0),np.ones(3))

    def test_no_extra_gammas_or_invalid_shapes(self):
        with self.assertRaises(ValueError):nm.weights(np.ones((4,3)),np.ones((4,3)),.25)
        with self.assertRaises(ValueError):nm.moments(np.zeros((65,4,3)),np.zeros((8,4,3)))

    def test_valid_FP64_weights_survive_FP32_sum_roundoff_without_changing_map(self):
        from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic.numerics import bilinear
        from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.calibration import from_z
        ss=np.asarray([[2165.5613109638693,.5152877890722317,94.18822497111822],
            [8.86874655713608,3.1855036211631167,.003420679972303465],
            [1256.8844220763392,.04164733575990459,3.484328883975659],
            [24.444741309576116,10.88875367793702,1152.2946200486836]])
        z=np.arange(140,dtype=np.float32).reshape(4,35)/70;pos=np.arange(35,dtype=np.int64)
        for gamma in (0.,.5):
            w=nm.weights(ss,ss,gamma)[0];rounded=w.astype(np.float32)
            self.assertEqual(float(rounded.sum(0)[0]),1.0000001192092896)
            self.assertGreater(abs(float(rounded.sum(0)[0])-1),1e-7)
            image,scores=nm.weighted(z,pos,(5,7),(17,23),w)
            expected=bilinear(np.sum(z.reshape(4,5,7)*rounded[:,0,None,None],axis=0,dtype=np.float32),(17,23))
            np.testing.assert_array_equal(image,expected)
            _,p,q=from_z(z,pos,(5,7),(17,23))
            self.assertEqual(scores['PEAK_LOCAL_WEIGHT'],float(.5*np.sum(rounded[:,1]*p,dtype=np.float32)+.5*np.sum(rounded[:,2]*q,dtype=np.float32)))

    def test_invalid_weights_still_rejected_by_roundoff_bound(self):
        z=np.ones((4,35),np.float32);pos=np.arange(35,dtype=np.int64)
        cases=[]
        for bad in (0.,-.1,np.nan,np.inf):
            w=np.full((4,3),.25);w[0,0]=bad;cases.append(w)
        w=np.full((4,3),.25);w[:,0]*=1.00001;cases.append(w)
        cases.append(np.full((3,3),1/3))
        for w in cases:
            with self.subTest(weights=w.tolist()),self.assertRaisesRegex(ValueError,'four positive normalized'):
                nm.weighted(z,pos,(5,7),(17,23),w)

    def test_summary_includes_fixed_parent_task_division_comparison(self):
        from DINOv3.MADEqual.aebad_s_target_loo_layer_weight_p0 import reporting
        from DINOv3.MADEqual.aebad_s_target_loo_layer_weight_p0.protocol import U,S,TARGET
        from DINOv3.MADEqual.aebad_s_blind_context_p0.protocol import COMBINED
        with tempfile.TemporaryDirectory() as tmp:
            a=SimpleNamespace(run_root=Path(tmp));out=a.run_root/'evaluation';out.mkdir()
            rows=[]
            for b in (U,S,COMBINED,*NEW):
                for scoring in (('PEAK_LOCAL_WEIGHT','WEIGHTED_MAP_MAX') if b in NEW else ('CH1','CURRENT')):
                    rows.append(dict(branch=b,scoring=scoring,**{k:.5 for k in ('image_AUROC','image_AP','pixel_AUROC','pixel_AP','AUPRO')}))
            ip.st.write_csv(out/'metrics.csv',rows);ip.st.write_csv(out/'map_manifest.csv',[])
            for b in NEW:
                v=dict(identity='fixture',domain='background',branch=b,map_identity=ip.st.digest([]),nstrips=200,FPR_max=.3,AUPRO=.5,verified_images=297,backend='original_ADEval_1.1.0_CUDA')
                v['seal']=ip.st.digest(v);ip.st.write_json(a.run_root/'aupro'/(b+'.json'),v)
            with patch('DINOv3.MADEqual.aebad_s_target_loo_layer_weight_p0.evaluation.restored',return_value=dict(images=297)),patch.object(ip,'identity',return_value='fixture'):
                reporting.summarize(a)
            comparisons=ip.old.read_csv(a.run_root/'metric_comparisons.csv')
            strong=[r for r in comparisons if r['branch']==TARGET and r['reference']==COMBINED]
            self.assertEqual(len(comparisons),30);self.assertEqual(len(strong),5)

    def test_whole_support_exclusion_all_patch_rows(self):
        supports=[dict(relative_path=f'support_{i}',rows=31+i) for i in range(8)]
        for held in range(8):
            chosen=nm.reference_rows(supports,held)
            self.assertEqual(len(chosen),7);self.assertNotIn(supports[held],chosen)
            self.assertEqual(sum(r['rows'] for r in chosen),sum(r['rows'] for r in supports)-supports[held]['rows'])

    def test_source_calibration_queries_exclude_bank_population(self):
        c=[dict(relative_path=f'{i:04d}',calibration=i%8==0) for i in range(521)]
        p=SimpleNamespace(catalog=dict(source=c),supports=[dict(relative_path=f't{i}') for i in range(8)])
        with patch.object(ip,'parent',return_value=p):r=ip.units(None)
        s={row['relative_path'] for k,i,row in r if k=='Source'}
        bank={x['relative_path'] for x in c if not x['calibration']}
        self.assertEqual(len(s),66);self.assertEqual(len(bank),455);self.assertFalse(s&bank)

    def test_evaluation_support_exclusion_uses_existing_actual_catalog_validator(self):
        from DINOv3.MADEqual.aebad_s_same_origin_neighbor_p0.inputs import validate_catalog
        f=ip.st.read_json(Path('DINOv3/MADEqual/aebad_s_reference_residual_segmentation/frozen_origins.json'))
        p=dict(source=[dict(r,source_index=i,calibration=i%8==0) for i,r in enumerate(f['source'])],supports={'background':[dict(r,domain='background') for r in f['background_supports']]},samples=[dict(r,domain='background') for r in f['background_samples']])
        q=json.loads(json.dumps(p));q['supports']['background'][0]['relative_path']=next(r['relative_path'] for r in q['samples'] if r['domain']=='background')
        with self.assertRaises(ValueError):validate_catalog(q)

    def test_uniform_weight_map_and_original_pq_not_reweighted(self):
        from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.calibration import from_z
        rng=np.random.default_rng(5);z=rng.normal(size=(4,35)).astype(np.float32);pos=rng.permutation(35).astype(np.int64)
        original,p,q=from_z(z,pos,(5,7),(17,23));w=np.full((4,3),.25)
        image,score=nm.weighted(z,pos,(5,7),(17,23),w)
        np.testing.assert_array_equal(image,original)
        self.assertEqual(score['WEIGHTED_MAP_MAX'],float(original.max()))
        w[:,0]=[.1,.2,.3,.4];w[:,1]=[.4,.3,.2,.1];w[:,2]=[.1,.4,.2,.3]
        _,score=nm.weighted(z,pos,(5,7),(17,23),w)
        self.assertAlmostEqual(score['PEAK_LOCAL_WEIGHT'],float(.5*np.sum(p*w[:,1].astype(np.float32),dtype=np.float32)+.5*np.sum(q*w[:,2].astype(np.float32),dtype=np.float32)))

    def test_statistics_linear_quantile_original_statistics(self):
        from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import statistics
        raw=np.arange(4*35,dtype=np.float32).reshape(4,35)/70
        patch_state=dict(medians=[.1]*4,scales=[.3]*4);pos=np.arange(35,dtype=np.int64)
        got=nm.normal_statistics(raw,pos,(5,7),(17,23),patch_state)
        _,p,q=statistics(raw,pos,(5,7),(17,23),np.asarray(patch_state['medians'],np.float32),np.asarray(patch_state['scales'],np.float32))
        np.testing.assert_array_equal(got[:,1],p);np.testing.assert_array_equal(got[:,2],q)
        z=(raw-np.float32(.1))/np.float32(.3)
        np.testing.assert_array_equal(got[:,0],np.quantile(z,.99,axis=1,method='linear'))

    def test_source_weight_not_forced_to_equal_or_CH1(self):
        w,_,_=nm.weights(np.arange(1,13).reshape(4,3),None,0)
        self.assertFalse(np.array_equal(w,np.full((4,3),.25)))

    def test_no_implicit_retrieval_and_only_requested_normal_population(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=SimpleNamespace(run_root=Path(tmp),cache_root=Path(tmp)/'cache',supplement_retrieval=False)
            p=SimpleNamespace()
            with patch.object(ip,'freeze'),patch.object(ip,'parent',return_value=p),patch.object(ip,'units',return_value=[('Source',0,dict(relative_path='normal'))]),patch.object(ip,'normal_identity',return_value={}):
                with self.assertRaisesRegex(FileNotFoundError,'explicit'):pp.prepare_normal_stats(a)

    def test_cpu_import_and_default_preflight_no_torch(self):
        code='import sys; from DINOv3.MADEqual.aebad_s_target_loo_layer_weight_p0 import pipeline,evaluation; from DINOv3.MADEqual.aebad_s_target_loo_layer_weight_p0.runner import parser; assert parser().parse_args([]).stage=="preflight"; assert "torch" not in sys.modules'
        r=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)

    def test_default_inventory_without_parent_reports_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=normalize(parser().parse_args(['--run-root',str(Path(tmp)/'run')]))
            result=ip.preflight(a);self.assertEqual(result['status'],'blocked')

    def test_scratch_cannot_overlap_parent_or_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):normalize(parser().parse_args(['--run-root',str(Path(tmp)/'run'),'--evaluation-scratch',str(Path(tmp)/'run'/'pixels')]))

    def test_actual_cached_74_statistics_fit_and_resume_without_retrieval(self):
        import time,contextlib,io
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=SimpleNamespace(run_root=root/'run',cache_root=root/'cache',supplement_retrieval=False,allow_over_12h=True,_started=time.perf_counter())
            a.run_root.mkdir();ip.st.sealed_json(a.run_root/'inputs.json',dict(identity='fixture'))
            source=[dict(relative_path=f's{i}',calibration=True,image_height=32,image_width=32) for i in range(66)]
            supports=[dict(relative_path=f't{i}',image_height=32,image_width=32) for i in range(8)]
            parent=SimpleNamespace(catalog=dict(source=source),supports=supports,assets=dict(references=dict(source=dict(seal='bank'))),extraction={},mad=dict(patch=dict(medians=[0]*4,scales=[1]*4)))
            a._parent=parent;rng=np.random.default_rng(42)
            for kind,i,row in ip.units(a):
                expected=ip.normal_identity(a,kind,i,row)
                ip.st.put(ip.normal_path(a,kind,i),dict(raw=rng.random((4,35),dtype=np.float32),positions=np.arange(35,dtype=np.int64)),dict(expected,grid=[5,7],output_shape=[8,8]))
            with patch.object(ip,'freeze'),patch.object(pp,'normal_features',side_effect=AssertionError('must not retrieve or extract')),contextlib.redirect_stdout(io.StringIO()):r=pp.prepare_normal_stats(a)
            self.assertEqual(len(r['records']),74)
            w=pp.fit_weights(a);first=(a.run_root/'weights.json').read_bytes();pp.fit_weights(a)
            self.assertEqual(first,(a.run_root/'weights.json').read_bytes());self.assertEqual(set(w['weights']),set(NEW))
            self.assertFalse(w['GT_used'])

    def test_modified_normal_reference_list_is_rejected(self):
        # Different leave-one-image-out is a different normal-distance identity.
        p=SimpleNamespace(assets=dict(references=dict(source=dict(seal='bank'))),extraction={},supports=[dict(relative_path=f't{i}') for i in range(8)])
        with patch.object(ip,'parent',return_value=p),patch.object(ip,'identity',return_value='id'):
            x=ip.normal_identity(None,'Target',2,p.supports[2]);y=ip.normal_identity(None,'Target',3,p.supports[2])
        self.assertNotEqual(x,y);self.assertNotIn('t2',x['reference_images'])

    def test_cached_prediction_freeze_resume_and_baseline_score_rejection(self):
        import time,contextlib,io
        from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.calibration import response
        from DINOv3.MADEqual.aebad_s_target_loo_layer_weight_p0.protocol import U,S
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=SimpleNamespace(run_root=root/'run',cache_root=root/'cache',allow_over_12h=True,_started=time.perf_counter())
            a.run_root.mkdir();ip.st.sealed_json(a.run_root/'inputs.json',dict(identity='fixture'))
            ip.st.sealed_json(a.run_root/'weights.json',dict(identity='fixture',weights={b:np.full((4,3),.25).tolist() for b in NEW}))
            rng=np.random.default_rng(8);z=rng.random((4,35),dtype=np.float32);pos=np.arange(35,dtype=np.int64)
            image={k:dict(medians=[0]*4,scales=[1]*4) for k in ('peak','local')}
            sc=response(z,pos,(5,7),(8,8),image)[1]
            samples=[dict(sample_id=f'test{i}') for i in range(2)]
            def parent_prediction(row):return {U:z,S:z, 'positions':pos},dict(seal=row['sample_id'],metadata=dict(grid=[5,7],output_shape=[8,8],scores={U:dict(sc),S:dict(sc)}))
            parent=SimpleNamespace(samples=samples,mad=dict(image=image),seals={r['sample_id']:r['sample_id'] for r in samples},prediction=parent_prediction)
            a._parent=parent
            with contextlib.redirect_stdout(io.StringIO()):pp.predict(a)
            before=(a.run_root/'predictions_frozen.json').read_bytes()
            with patch.object(parent,'prediction',side_effect=AssertionError('completed replay should be reused')),contextlib.redirect_stdout(io.StringIO()):pp.predict(a)
            self.assertEqual(before,(a.run_root/'predictions_frozen.json').read_bytes())
            pp.verified(a)
            x,r=pp.prediction(a,samples[0]);self.assertEqual(set(r['metadata']['scores'][NEW[0]]),{'PEAK_LOCAL_WEIGHT','WEIGHTED_MAP_MAX'})
            for v in x.values():v._mmap.close()
            a.cache_root=root/'newcache'
            def wrong(row):
                x,r=parent_prediction(row);r['metadata']['scores'][U]['CH1']+=.01;return x,r
            with patch.object(parent,'prediction',side_effect=wrong),self.assertRaisesRegex(ValueError,'baseline scores'):pp.predict(a)


if __name__=='__main__':unittest.main()
