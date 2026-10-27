"""Fixed SACP CPU math and integration; no real RGB, CUDA, model fitting or ADEval."""
import copy,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from DINOv3.MADEqual.aebad_s_sacp_p0 import numerics as n,clustering,protocol as p,inputs as ip,runtime as rt,prediction as pred,runner
from DINOv3.MADEqual.aebad_s_sacp_p0.inputs import st


def setUpModule():
    (ip.ROOT/'.tmp').mkdir(exist_ok=True)


class Math(unittest.TestCase):
    def test_frozen_plan_and_no_training(self):
        plan=p.job_plan();self.assertEqual(len(plan['localization_jobs']),18);self.assertEqual(plan['training_jobs'],0)
        self.assertEqual((p.CONFIG['clusters'],p.CONFIG['common_rank'],p.CONFIG['directions'],p.CONFIG['strength']),(32,32,2,.5))
        self.assertEqual(plan['prediction_images'],926);self.assertEqual(plan['expected_metric_rows'],56)
    def test_zero_strength_and_no_space_strict_parent_output(self):
        d=np.array([1.00001,3.25,0],np.float32);r=np.ones((3,4),np.float32);b=np.zeros((2,4,2),np.float32);g=np.array([0,1,0])
        for strength in (0,.5):
            value,eta,_=n.correct(d,r,g,b,np.zeros(2,np.int16),strength);np.testing.assert_array_equal(value,d);np.testing.assert_array_equal(eta,0)
    def test_parallel_orthogonal_and_zero_residual(self):
        r=np.array([[2,0,0],[0,3,0],[0,0,0]],np.float32);b=np.zeros((1,3,2),np.float32);b[0,0,0]=1
        d=np.array([7,8,.001],np.float32);value,eta,_=n.correct(d,r,np.zeros(3,int),b,np.ones(1,int))
        np.testing.assert_allclose(value,[7*np.sqrt(.5),8,.001],atol=1e-7,rtol=1e-7);np.testing.assert_array_equal(eta,[1,0,0])
        self.assertTrue(np.isfinite(value).all())
    def test_eta_only_roundoff_clip_obvious_violation_rejected(self):
        d=np.ones(1,np.float32);r=np.ones((1,1),np.float32);g=np.zeros(1,int);b=np.ones((1,1,1),np.float32)
        value,eta,audit=n.correct(d,r,g,b*np.float32(1+1e-7),np.ones(1,int));self.assertEqual(audit['roundoff_clipped'],1);self.assertEqual(eta[0],1)
        with self.assertRaisesRegex(FloatingPointError,'eta'):n.correct(d,r,g,b*2,np.ones(1,int))
    def test_uncentered_moment_retains_constant_shift(self):
        def blocks(scale):return [dict(image=f'{fold}_{i}',fold=fold,delta=np.tile([scale,0,0],(7,1)).astype(np.float32)) for fold in ('A','B') for i in range(4)]
        bases,diag=n.fit_group(blocks(1),blocks(2),42)
        self.assertEqual(diag['directions'],1);self.assertAlmostEqual(diag['Target_energy']['mean_energy'],4,places=5)
        self.assertLess(diag['Target_energy']['centered_energy'],1e-6);self.assertGreater(bases[0,0,0]**2,.999)
    def test_generalized_solve_matches_small_direct_reference(self):
        rng=np.random.default_rng(6);a=rng.normal(size=(5,5));b=rng.normal(size=(5,5));cs=a@a.T;ct=b@b.T
        gamma,v,diag=n.generalized(cs,ct);B=.9*cs+(.1*np.trace(cs)/5+1e-6*np.trace(cs+ct)/10)*np.eye(5)
        np.testing.assert_allclose(ct@v,(B@v)*gamma,atol=1e-11,rtol=1e-11)
        ref=np.sort(np.linalg.eigvals(np.linalg.solve(B,ct)))[::-1];np.testing.assert_allclose(gamma,ref,rtol=1e-11,atol=1e-11)
    def test_class_fold_image_weighting_not_patch_count(self):
        blocks=[]
        for fold in ('A','B'):
            blocks += [dict(image=fold+'many',fold=fold,delta=np.ones((100,2),np.float32)),dict(image=fold+'few',fold=fold,delta=np.ones((1,2),np.float32)*3)]
        rows,counts=n.weighted(blocks);self.assertEqual(counts,dict(A=2,B=2))
        self.assertAlmostEqual(sum(len(a)*w for a,w in rows),1);self.assertAlmostEqual(sum(float(a[:,0].sum())*w for a,w in rows),2)
    def test_group_coverage_and_control_rank_joint_fallback(self):
        source=[dict(image=f'{fold}{i}',fold=fold,delta=np.tile([1,0,0],(4,1)).astype(np.float32)) for fold in ('A','B') for i in range(4)]
        target=[dict(image=f't{i}',fold=fold,delta=np.array([[0,4,0],[0,0,4]],np.float32)) for fold in ('A','B') for i in range(4)]
        basis,diag=n.fit_group(source,target,42);self.assertEqual(diag['directions'],0);self.assertIn('control_rank',diag['reason']);self.assertFalse(basis.any())
        _,diag=n.fit_group(source[:-1],target,42);self.assertIn('fewer_than',diag['reason'])
        self.assertEqual(diag['Target_unique_images'],4)  # Not eight independent Target images.
    def test_QR_bases_shared_rank_and_orthogonality(self):
        rng=np.random.default_rng(7)
        def blocks(scale):return [dict(image=f'{fold}{i}',fold=fold,delta=(rng.normal(size=(12,6))*scale).astype(np.float32)) for fold in ('A','B') for i in range(4)]
        basis,diag=n.fit_group(blocks(1),blocks(3),42);k=diag['directions'];self.assertGreater(k,0)
        for u in basis:np.testing.assert_allclose(u[:,:k].T@u[:,:k],np.eye(k),atol=1e-6,rtol=1e-6)
    def test_uniform_matches_native_total_not_positions(self):
        rng=np.random.default_rng(12);u=rng.uniform(.01,2,91).astype(np.float32);main=u*np.linspace(.75,1,len(u)).astype(np.float32)
        value,diag=n.matched_uniform(u,main)
        self.assertLessEqual(diag['absolute_error'],diag['FP32_sum_roundoff_bound']);self.assertFalse(np.array_equal(value,main))
        np.testing.assert_array_equal(n.matched_uniform(np.zeros(3,np.float32),np.zeros(3,np.float32))[0],0)
    def test_kmeans_is_bounded_fixed_seed_and_handles_degenerate_features(self):
        rng=np.random.default_rng(11);x=rng.normal(size=(53,6)).astype(np.float32)
        a=clustering.kmeans(x,groups=4,cap=30,iterations=5);b=clustering.kmeans(x,groups=4,cap=30,iterations=5)
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1]);self.assertLessEqual(a[2]['iterations'],5)
        deg=clustering.kmeans(np.ones((40,6),np.float32),groups=32);self.assertTrue(np.isfinite(deg[0]).all());self.assertTrue((deg[1]==0).all())
    def test_controls_exclude_calibration_and_query_fold(self):
        from DINOv3.tests.test_aebad_s_reference_residual_segmentation import catalog
        c=catalog();chosen=ip.controls(c);groups=ip.old.folds(c)
        self.assertEqual([len(v) for v in chosen.values()],[8,8]);self.assertEqual(ip.controls(c),chosen)
        for fold,rows in chosen.items():
            opposite={r['relative_path'] for r in ip.old.train_source(c) if groups[r['relative_path']]!=fold}
            self.assertTrue(all(not r['calibration'] and groups[r['relative_path']]==fold and r['relative_path'] not in opposite for r in rows))
    def test_same_physical_fit_sample_for_all_layers_and_folds(self):
        positions=np.random.default_rng(3).permutation(6000)
        a=rt.fit_positions(positions,'image');b=rt.fit_positions(positions[::-1],'image')
        np.testing.assert_array_equal(a,b);self.assertEqual(len(a),4096);self.assertEqual(len(np.unique(a)),4096)
    def test_union_anchor_ties_and_source_group_identity(self):
        q=np.array([[5,6],[7,8]],np.float32);source=np.array([[1,2],[3,4]],np.float32);parts=[np.array([[9,10]],np.float32)]
        nn=dict(si=np.array([0,1]),ti=np.array([0,0]),ds=np.array([1,2],np.float32),dt=np.array([1,1],np.float32))
        residual,side=pred.residual_anchor(q,source,parts,nn)
        np.testing.assert_array_equal(residual,[[4,4],[-2,-2]]);np.testing.assert_array_equal(side,[False,True])


class Interface(unittest.TestCase):
    def test_default_preflight_no_torch_no_output_or_automatic_rebuild(self):
        with tempfile.TemporaryDirectory(dir=ip.ROOT/'.tmp') as folder:
            path=Path(folder)/'not_created';script=f'import sys; from DINOv3.MADEqual.aebad_s_sacp_p0.runner import main; rc=main(["--run-root",{str(path)!r}]); assert rc==2; assert "torch" not in sys.modules'
            result=subprocess.run([sys.executable,'-B','-c',script],cwd=ip.ROOT,env=dict(os.environ,PYTHONUTF8='1'),capture_output=True,text=True,encoding='utf-8')
            self.assertEqual(result.returncode,0,result.stderr);self.assertFalse(path.exists());self.assertIn('"training": false',result.stdout)
    def test_missing_fold_blocks_instead_of_building(self):
        a=runner.normalize(runner.parser().parse_args(['--run-root',str(ip.ROOT/'.tmp/sacp_missing_run')]))
        with patch.object(ip,'inventory',return_value=dict(status='blocked',missing=[dict(asset='fold_A')])):
            with self.assertRaisesRegex(ValueError,'missing raw assets'):rt.prepare(a)
        self.assertFalse(a.run_root.exists())
    def test_baseline_tolerances_are_unchanged(self):
        checks=[];s=dict(domain='view',sample_id='same_name')
        pred.compare(np.array([1],np.float32),np.array([1],np.float32),'UNION',s,7,checks)
        with self.assertRaisesRegex(ValueError,'no tolerance relaxation'):pred.compare(np.array([1.01]),np.array([1]),'UNION',s,7,checks)

    def test_asset_headers_and_small_manifest_seal(self):
        with tempfile.TemporaryDirectory(dir=ip.ROOT/'.tmp') as folder:
            path=Path(folder);st.put(path,dict(L7=np.ones((5,3),np.float32)),dict(kind='original_source'))
            ip.record(path,dict(L7=(5,3)),'original_source')
            np.save(path/'L7.npy',np.ones((5,3),np.float64))
            with self.assertRaisesRegex(ValueError,'NPY header'):ip.record(path,dict(L7=(5,3)),'original_source')
            value=st.read_json(path/'manifest.json');value['metadata']['kind']='adapted'
            st.write_json(path/'manifest.json',value)
            with self.assertRaises(ValueError):ip.record(path,dict(L7=(5,3)),'original_source')

    def test_review_pack_excludes_uncommitted_pixel_scratch(self):
        import tarfile
        with tempfile.TemporaryDirectory(dir=ip.ROOT/'.tmp') as folder:
            a=runner.normalize(runner.parser().parse_args(['--run-root',str(Path(folder)/'run')]))
            for path in (a.evaluation_scratch/'leftover.bin',a.cache_root/'large.npy',a.run_root/'other.bin'):
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'partial scratch fixture')
            st.write_json(a.run_root/'status.json',dict(status='pending'))
            result=runner.pack(a)
            with tarfile.open(result['archive']) as t:self.assertEqual(t.getnames(),['run/status.json'])

    def test_aupro_adapter_checks_actual_map_and_GT_identity(self):
        from DINOv3.MADEqual.aebad_s_sacp_p0 import evaluation
        fn=evaluation.bind(evaluation.previous.aupro).__globals__['original'].shared.verify_map_record
        array=np.zeros((3,4),np.float32);mask=np.zeros((3,4),np.uint8)
        row=dict(output_height=3,output_width=4,map_float32_sha256=evaluation.original.shared.array_sha(array),
            mask_uint8_sha256=evaluation.original.shared.array_sha(mask))
        fn(row,array,mask)
        changed=array.copy();changed[0,0]=1
        with self.assertRaises(ValueError):fn(row,changed,mask)
        mask[0,0]=1
        with self.assertRaises(ValueError):fn(row,array,mask)

    def test_AUPRO_stage_peak_excludes_historical_reused_job_cost(self):
        jobs=[dict(runtime_reused=True,peak_cuda_allocated_bytes=9999),dict(runtime_reused=False,peak_cuda_allocated_bytes=300)]
        peak=runner.job_gpu_peaks(jobs,dict(peak_cuda_allocated_bytes=100,peak_cuda_reserved_bytes=400))
        self.assertEqual(peak,dict(peak_cuda_allocated_bytes=300,peak_cuda_reserved_bytes=400))


class CPUIntegration(unittest.TestCase):
    """Real CPU Exact + fitting + six-map evaluation; only asset inventory/RGB/GT are fixtures."""
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch=torch;torch.set_num_threads(1)
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ip.ROOT/'.tmp');self.root=Path(self.temp.name);st.configure('fast')
    def tearDown(self):self.temp.cleanup()
    def fixture(self):
        from importlib.metadata import version
        from DINOv3.tests.test_aebad_s_reference_residual_segmentation import catalog
        a=runner.normalize(runner.parser().parse_args(['--device','cpu','--cpu-workers','1','--run-root',str(self.root/'run'),
            '--cache-root',str(self.root/'cache'),'--evaluation-scratch',str(self.root/'scratch'),'--data',str(self.root/'input.zip')]))
        c=catalog();c['samples']=[dict(domain=d,sample_id=f'test_{i}',relative_path=f'test/{d}/{i}.png',image_height=64,image_width=64,image_sha256=d+str(i),label=i) for d in p.DOMAINS for i in (0,1)]
        selected=ip.controls(c);parent=self.root/'parent';cache=self.root/'parent_cache';refs={};normals={};queries={}
        def saved(path,arrays,meta):
            r=st.put(path,arrays,meta);return dict(path=str(path),seal=r['seal'],files=r['files'],metadata=meta)
        for name in ('source','fold_A','fold_B'):refs[name]=saved(cache/name,{f'L{l}':np.zeros((64,1280),np.float32) for l in p.LAYERS},dict(kind='original_source' if name=='source' else 'training_only_opposite_fold'))
        image={k:dict(medians=[0]*4,scales=[1]*4) for k in ('peak','local')}
        refs.update(parent_run=str(parent),parent_cache=str(cache),parent_identity='fixture-parent',
            mad=dict(patch=dict(medians=[.03]*4,scales=[.02]*4),image=image,scope='original_Source_only_distances'),
            extraction=dict(software={k:version(k) for k in ('numpy','scipy','Pillow')}))
        for row in sum(selected.values(),[])+sum(c['supports'].values(),[])+c['samples']:
            rng=st.rng('SACP-CPU-fixture',row['relative_path']);pos=rng.permutation(16).astype(np.int64)
            q={f'q{l}':rng.normal(size=(16,1280)).astype(np.float32) for l in p.LAYERS}
            if row.get('support_index'):
                for v in q.values():v[:,0]+=3
            meta=dict(identity='fixture-parent',image=row,grid=[4,4],output_shape=[16,16],kind='original_complete_grid')
            if row.get('sample_id','').startswith('test_'):meta['sample']=row
            ref=saved(cache/'features'/ip.key(row),dict(q,positions=pos),meta)
            if row in c['samples']:ref['parent_identity']='fixture-parent';queries[row['domain']+'/'+row['sample_id']]=ref
            else:normals[row['relative_path']]=ref
        report=dict(status='inventory_ready_not_runtime_verified',catalog=c,controls=selected,references=refs,normal_features=normals,queries=queries,prediction_parents={})
        return a,c,report

    def test_parent_metric_proof_and_changed_small_results(self):
        from DINOv3.MADEqual.aebad_s_sacp_p0 import evaluation
        a,c,r=self.fixture();ip.freeze(a,r)
        root=Path(r['references']['parent_run'])/'evaluation'/'background';root.mkdir(parents=True)
        row=dict(domain='background',sample_id='test_0',branch=p.U,output_height=16,output_width=16,
            map_float32_sha256='frozen-map',mask_uint8_sha256='frozen-GT')
        st.write_csv(root/'map_manifest.csv',[row]);st.write_json(root/(p.U+'_exact_pixels.json'),dict(result=dict(AUROC=.7,AP=.5)))
        files={name:st.sha(root/name) for name in ('map_manifest.csv',p.U+'_exact_pixels.json')}
        st.sealed_json(root/'status.json',dict(identity='fixture-parent',images=1,numerical_failed_checks=0,files=files))
        self.assertIsNotNone(evaluation.parent_proof(a,'background',[row],p.U))
        changed=dict(row,mask_uint8_sha256='other-GT')
        with self.assertRaisesRegex(ValueError,'map/GT'):evaluation.parent_proof(a,'background',[changed],p.U)
        st.write_json(root/(p.U+'_exact_pixels.json'),dict(result=dict(AUROC=1.5,AP=.5)))
        with self.assertRaisesRegex(ValueError,'table changed'):evaluation.parent_proof(a,'background',[row],p.U)
    def test_real_CPU_fit_prediction_evaluation_and_resume(self):
        from DINOv3.MADEqual.aebad_s_sacp_p0 import evaluation,reporting
        from DINOv3.MADEqual.aebad_s_support_witness_segment_p0 import fresh_inputs
        a,c,r=self.fixture();calls={'GT':0};toy={d:[2,1,1] for d in p.DOMAINS}
        class Data:
            def __init__(self,*args):pass
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def gt(self,s):
                calls['GT']+=1;x=np.zeros((64,64),np.uint8)
                if s['label']:x[24:30,24:30]=1
                return x
        with patch.dict(p.CONFIG,{'populations':toy}),patch.object(ip,'inventory',return_value=r),patch.object(fresh_inputs,'Data',Data):
            rt.prepare(a);self.assertEqual(calls['GT'],0);fit=rt.fit(a);self.assertEqual(calls['GT'],0)
            self.assertEqual(len(fit['jobs']),12);self.assertEqual(fit['query_image_forward_calls'],0)
            with patch.object(rt.References,'exact',side_effect=AssertionError('completed fit searched again')):self.assertEqual(rt.fit(a)['new_correspondences'],0)
            with patch.object(rt,'extractor',side_effect=AssertionError('raw Q already cached')):
                result=pred.predict(a);self.assertEqual(result['query_forward_calls'],0);self.assertEqual(result['new_images'],6)
                self.assertEqual(pred.predict(a)['new_images'],0)
            self.assertEqual(calls['GT'],0);evaluation.evaluate(a);used=calls['GT'];evaluation.evaluate(a);self.assertEqual(used,calls['GT'])
            value=reporting.summarize(a);self.assertEqual(value['status'],'pending');self.assertEqual(value['metric_rows'],56);self.assertEqual(len(value['pending']),18)
            for d in p.DOMAINS:
                rows=evaluation.read_csv(a.run_root/'evaluation'/d/'metrics.csv')
                for b in p.BRANCHES:
                    v=[x for x in rows if x['branch']==b];self.assertEqual(v[0]['pixel_AP'],v[1]['pixel_AP'])
            # Reporting-only fixture: verifies requested completion, NOT actual ADEval.
            for d in p.DOMAINS:
                maps=evaluation.read_csv(a.run_root/'evaluation'/d/'map_manifest.csv')
                for b in p.BRANCHES:
                    st.sealed_json(a.run_root/'aupro'/d/(b+'.json'),dict(identity=ip.identity(a),domain=d,branch=b,map_identity=st.digest([v for v in maps if v['branch']==b]),
                        nstrips=200,FPR_max=.3,AUPRO=.5,verified_images=2,backend='original_ADEval_1.1.0_CUDA',scope='reporting_fixture_not_backend_execution'))
            value=reporting.summarize(a);self.assertEqual(value['status'],'all_five_metrics_complete');self.assertEqual(value['new_AUPRO_jobs'],18)
            self.assertFalse(self.torch.cuda.is_initialized())
    def test_opposite_fold_indices_reused_only_with_same_query_and_bank(self):
        a,c,r=self.fixture();ip.freeze(a,r);refs=rt.References(a);row=r['controls']['B'][0];layer=7;ref=r['normal_features'][row['relative_path']]
        path=Path(r['references']['parent_cache'])/'training_queries'/st.digest([row['relative_path'],'normal'])[:28]
        pos=np.arange(16,dtype=np.int64)[::-1]
        nn=st.put(path/'source_nn',dict(positions=pos,i7=pos*2,d7=np.zeros(16,np.float32)),dict(source='fold_A',source_seals=[r['references']['fold_A']['seal']]*4))
        st.sealed_json(path/'complete.json',dict(normal_feature_seal=ref['seal'],source_seal=nn['seal'],positions=pos.tolist()))
        value=rt.old_source_indices(a,row,'fold_A',7,np.array([0,3,7]),ref['seal']);np.testing.assert_array_equal(value,[0,6,14])
        self.assertIsNone(rt.old_source_indices(a,row,'fold_B',7,np.array([0,3,7]),ref['seal']))
        self.assertIsNone(rt.old_source_indices(a,row,'fold_A',7,np.array([0]),'wrong-query-seal'))
    def test_parent_Exact_baselines_and_all_fallback_maps_are_preserved(self):
        from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.model import exact
        from DINOv3.MADEqual.aebad_s_reference_residual_segmentation.retrieval import target_nearest
        from DINOv3.MADEqual.aebad_s_target_local_segment.geometry import geometry_layer
        from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.calibration import z_values,response
        from DINOv3.MADEqual.aebad_s_support_witness_segment_p0 import fresh_inputs
        a,c,r=self.fixture();parent_values={};patch_state=r['references']['mad']['patch']
        for sample in c['samples']:
            x,_=st.get(r['queries'][sample['domain']+'/'+sample['sample_id']]['path']);raw_u=[];raw_s=[];arrays={};banks=[]
            for l in p.LAYERS:
                source,_=st.get(r['references']['source']['path'],(f'L{l}',));parts=[];seals=[];offsets=[0]
                for row in c['supports'][sample['domain']]:
                    ref=r['normal_features'][row['relative_path']];part,_=st.get(ref['path'],(f'q{l}',));parts.append(part[f'q{l}']);seals.append(ref['seal']);offsets.append(offsets[-1]+16)
                ds,si=exact(x[f'q{l}'],source[f'L{l}'],'cpu');dt,ti,diag=target_nearest(x[f'q{l}'],np.concatenate(parts),offsets,'cpu')
                segment,_=geometry_layer(diag['per_support_distances'],diag['per_support_indices']-np.asarray(offsets[:-1])[:,None],ds,parts)
                raw_u.append(np.minimum(ds,dt));raw_s.append(segment['K8_SEGMENT']);arrays.update({f'ds{l}':ds,f'si{l}':si,f'dt{l}':dt,f'ti{l}':ti})
                banks.append(dict(source_seal=r['references']['source']['seal'],target_seals=seals,target_offsets=offsets,target_origins=[v['relative_path'] for v in c['supports'][sample['domain']]]))
            arrays.update({p.U:z_values(np.stack(raw_u),patch_state),p.S:z_values(np.stack(raw_s),patch_state),'positions':x['positions']})
            scores={b:response(arrays[b],x['positions'],[4,4],[16,16],r['references']['mad']['image'])[1] for b in (p.U,p.S)}
            path=self.root/'parent_cache/predictions'/sample['domain']/sample['sample_id']
            st.put(path,arrays,dict(identity='fixture-parent',sample=sample,grid=[4,4],scores=scores,NN_record=dict(metadata=dict(inputs=dict(banks=banks)))))
            r['prediction_parents'][sample['domain']+'/'+sample['sample_id']]=dict(path=str(path),parent=str(self.root/'parent'),identity='fixture-parent')
            parent_values[sample['domain'],sample['sample_id']]=arrays
        toy={d:[2,1,1] for d in p.DOMAINS}
        class Data:
            def __init__(self,*args):pass
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def gt(self,*args):raise AssertionError('GT entered prediction')
        with patch.dict(p.CONFIG,{'populations':toy}),patch.object(ip,'inventory',return_value=r),patch.object(fresh_inputs,'Data',Data):
            rt.prepare(a)
            # Explicit no-valid-subspace fixture checks strict full-prediction fallback.
            for d in p.DOMAINS:
                for l in p.LAYERS:
                    model=st.put(a.cache_root/'subspaces'/d/f'L{l}',dict(bases=np.zeros((3,32,1280,2),np.float32),ranks=np.zeros(32,np.int16)),dict(identity=ip.identity(a)))
                    st.sealed_json(a.run_root/'fit'/d/f'L{l}.json',dict(identity=ip.identity(a),subspace_seal=model['seal'],diagnostics=[]))
            result=pred.predict(a);self.assertEqual(result['new_images'],6)
            for sample in c['samples']:
                x,record=st.get(pred.prediction_path(a,sample));old=parent_values[sample['domain'],sample['sample_id']]
                for b in (p.U,*p.NEW):np.testing.assert_array_equal(x[b],old[p.U])
                np.testing.assert_array_equal(x[p.S],old[p.S]);self.assertTrue(all(v['verified'] for v in record['metadata']['checks']))
    def test_scratch_isolation_preserves_parent_files(self):
        a,c,r=self.fixture();a.recent_run=[self.root/'parent'];a.recent_run[0].mkdir();a.evaluation_scratch=a.recent_run[0]/'inside'
        with self.assertRaisesRegex(ValueError,'overlaps'):runner.isolation(a)
