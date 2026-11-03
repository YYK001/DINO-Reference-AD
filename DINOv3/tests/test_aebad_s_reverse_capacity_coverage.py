"""CPU-only risks: geometry, exact LP objective, provenance and frozen readout."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from DINOv3.MADEqual.aebad_s_reverse_capacity_coverage_p0 import protocol as p
from DINOv3.MADEqual.aebad_s_reverse_capacity_coverage_p0 import regions as rg,transport as lp,numerics as nm
from DINOv3.MADEqual.aebad_s_reverse_capacity_coverage_p0 import inputs as ip,runtime as rt,pipeline,reporting,runner


class SpatialTests(unittest.TestCase):
    def test_partition_physical_connectivity_area_and_reordering(self):
        rng=np.random.default_rng(42);q=[rng.normal(size=(30,5)).astype(np.float32) for _ in range(4)]
        pos=np.arange(30);a=rg.partition(q,pos,(5,6),np.ones(4),7)
        shuffle=rng.permutation(30);b=rg.partition([x[shuffle] for x in q],pos[shuffle],(5,6),np.ones(4),7)
        np.testing.assert_array_equal(a['labels'],b['labels']);np.testing.assert_array_equal(a['means'],b['means'])
        self.assertEqual(len(a['counts']),7);self.assertEqual(a['counts'].sum(),30);self.assertAlmostEqual(a['area'].sum(),1)
        for i in range(7):
            ids=set(np.flatnonzero(a['labels']==i));seen={min(ids)};todo=list(seen)
            while todo:
                node=todo.pop();y,x=divmod(node,6)
                for k in ([node-1] if x else [])+([node+1] if x<5 else [])+([node-6] if y else [])+([node+6] if y<4 else []):
                    if k in ids and k not in seen:seen.add(k);todo.append(k)
            self.assertEqual(ids,seen);self.assertEqual(len(ids),a['counts'][i])

    def test_ties_reproducible_and_min_physical_root(self):
        q=[np.zeros((16,3),np.float32) for _ in range(4)]
        a=rg.partition(q,np.arange(16),(4,4),np.ones(4),4)
        b=rg.partition(q,np.arange(16),(4,4),np.ones(4),4)
        np.testing.assert_array_equal(a['labels'],b['labels'])
        for i,root in enumerate(a['roots']):self.assertEqual(root,np.flatnonzero(a['labels']==i).min())

    def test_fixed_256_and_original_dimension(self):
        q=[np.ones((256,1280),np.float32) for _ in range(4)]
        r=rg.partition(q,np.arange(256),(16,16),np.ones(4))
        self.assertEqual(r['means'].shape,(4,256,1280));self.assertEqual(len(np.unique(r['labels'])),256)

    def test_no_duplicate_crop_tokens_or_memory_indices(self):
        q=[np.ones((4,3),np.float32) for _ in range(4)]
        for positions in ([0,1,1,3],[1,2,3,4]):
            with self.assertRaises(ValueError):rg.partition(q,np.array(positions),(2,2),np.ones(4),2)

    def test_descriptor_distance_four_layers_and_rms(self):
        a=np.zeros((4,2,3),np.float32);b=np.ones((4,3,3),np.float32)
        d=rg.distance_matrix(a,b,[1,2,4,8]);np.testing.assert_allclose(d,(1+1/2+1/4+1/8)/4,rtol=1e-7)


class LPTests(unittest.TestCase):
    def test_hand_minimax_not_minimum_total_transport(self):
        d=np.array([[0.,1.],[0.,10.]]);a=np.array([.9,.1]);b=a.copy()
        r=lp.solve(d,a,b,save_plan=True)
        self.assertTrue(r['success'],r);self.assertAlmostEqual(r['objective'],10/91,places=8)
        self.assertLess(r['objective'],1/9);self.assertEqual(r['free'],0)
        np.testing.assert_allclose(a@np.array(r['W']),b,atol=1e-10)

    def test_capacity_cannot_reuse_query_area_without_bound(self):
        d=np.array([[0.,1.],[0.,1.]])
        r=lp.solve(d,[.5,.5],[.5,.5]);self.assertTrue(r['success'],r)
        self.assertAlmostEqual(r['objective'],.5);self.assertEqual(lp.free_cost(d),0.)

    def test_identical_descriptors_mass_split_invariance(self):
        d=np.array([[.2,1.3],[.8,.1]]);a=[.6,.4];b=[.7,.3]
        r=lp.solve(d,a,b)
        split_row=lp.solve(d[[0,0,1]], [.2,.4,.4],b)
        split_col=lp.solve(d[:,[0,0,1]],a,[.3,.4,.3])
        self.assertTrue(all(x['success'] for x in (r,split_row,split_col)))
        self.assertAlmostEqual(r['objective'],split_row['objective'],places=8)
        self.assertAlmostEqual(r['objective'],split_col['objective'],places=8)

    def test_relative_capacity_check_detects_small_mass_distortion(self):
        d=np.zeros((2,2));a=np.array([.5,.5]);b=np.array([1-1e-5,1e-5]);W=np.tile(b,(2,1))
        W[0,1]+=2e-8;W[0,0]-=2e-8
        check,valid=lp.feasibility(d,a,b,W,0.)
        self.assertFalse(valid);self.assertGreater(check['capacity_max_relative_error'],1e-6)

    def test_small_random_problems_free_lower_bound_and_sparse_constraints(self):
        rng=np.random.default_rng(7)
        for _ in range(3):
            d=rng.random((5,6));a=rng.random(5);a/=a.sum();b=rng.random(6);b/=b.sum()
            r=lp.solve(d,a,b);self.assertTrue(r['success'],r);self.assertGreaterEqual(r['objective']+1e-7,r['free'])
        eq,_,ub,_=lp.constraints(np.ones((256,256)),np.ones(256)/256,np.ones(256)/256)
        self.assertEqual(eq.shape,(512,65537));self.assertLess(eq.nnz+ub.nnz,200000)

    def test_solver_failure_retained_no_free_substitution(self):
        fake=SimpleNamespace(success=False,status=4,message='fixture failure',nit=0)
        with patch.object(lp,'linprog',return_value=fake):r=lp.solve(np.ones((2,2)),[.5,.5],[.5,.5])
        self.assertFalse(r['success']);self.assertIsNone(r['objective']);self.assertEqual(r['free'],1)


class CalibrationTests(unittest.TestCase):
    def records(self):return [dict(kind='Source',P=i+1.,cap=2*i+1.,free=i/2+1.) for i in range(66)]
    def test_calibration_normal_only_and_no_label_influence(self):
        rows=self.records();model=nm.calibration(rows);after=copy.deepcopy(rows)
        for r in after:r['test_label']=1;r['GT']=123
        self.assertEqual(model,nm.calibration(after))
        rows[0]['kind']='query'
        with self.assertRaises(ValueError):nm.calibration(rows)

    def test_fixed_shared_and_own_scales_with_original_P_unchanged(self):
        model=nm.calibration(self.records());out,blocked=nm.scores(10,8,5,model)
        self.assertFalse(blocked);self.assertEqual(set(out),set(p.BRANCHES));self.assertEqual(out[p.BASE],10)
        zP=(10-model['P']['mean'])/model['P']['std']
        self.assertEqual(out[p.SHARED],zP+(5-model['cap']['mean'])/model['cap']['std'])
        self.assertEqual(out[p.OWN],zP+(5-model['free']['mean'])/model['free']['std'])

    def test_degenerate_scales_block_dependents_no_epsilon(self):
        rows=self.records()
        for r in rows:r['cap']=1
        model=nm.calibration(rows);out,blocked=nm.scores(1,1,2,model)
        self.assertEqual(set(blocked),{p.SHARED,p.PRIMARY});self.assertIn(p.OWN,out);self.assertNotIn(p.PRIMARY,out)

    def test_RMS_complete_feature_definition_no_centering_or_patch_L2(self):
        q=[np.full((3,1280),2,np.float32) for _ in range(4)]
        sums,n=nm.squared_sums(q);r=nm.rms_from_sums(sums,n)
        np.testing.assert_array_equal(r,np.full(4,2.))
        with self.assertRaises(ValueError):nm.squared_sums([np.ones((2,3)),*q[1:]])


class IntegrationTests(unittest.TestCase):
    def test_default_preflight_and_plan(self):
        a=runner.parser().parse_args([]);self.assertEqual(a.stage,'preflight');self.assertEqual(a.device,'cpu');self.assertFalse(a.extract_missing)
        self.assertEqual(p.plan()['LP_jobs'],17600);self.assertEqual(p.plan()['region_images'],1016)
        self.assertEqual(p.plan()['new_AUPRO_jobs'],0)

    def test_equal_setting_macro_and_report_count_separation(self):
        data=[]
        for d in p.DOMAINS:
            for seed in p.SEEDS:
                for b in p.BRANCHES:data.append(dict(domain=d,support_seed=seed,branch=b,image_AUROC=.7,image_AP=.8))
        a=reporting.aggregate(data);grand=[r for r in a if r['domain']=='macro_3domains_2seeds']
        self.assertEqual(len(grand),6);self.assertAlmostEqual(grand[0]['image_AUROC'],.7)
        removed=[r for r in data if not (r['domain']=='view' and r['support_seed']==43 and r['branch']==p.PRIMARY)]
        self.assertFalse(any(r['domain']=='macro_3domains_2seeds' and r['branch']==p.PRIMARY for r in reporting.aggregate(removed)))

    def test_actual_report_counts_only_six_settings_and_each_metric(self):
        for mixed_AP in (False,True):
            with self.subTest(mixed_AP=mixed_AP),tempfile.TemporaryDirectory() as tmp:
                a=SimpleNamespace(run_root=Path(tmp));data=[]
                for d in p.DOMAINS:
                    for s in p.SEEDS:
                        for branch in p.BRANCHES:
                            data.append(dict(domain=d,support_seed=s,branch=branch,
                                image_AUROC=.8 if branch==p.PRIMARY else .7,
                                image_AP=(.9 if not mixed_AP or s==42 else .7) if branch==p.PRIMARY else .8))
                ip.st.write_csv(a.run_root/'setting_metrics.csv',data)
                ip.st.write_csv(a.run_root/'metrics.csv',reporting.aggregate(data))
                frozen=ip.st.sealed_json(a.run_root/'predictions_frozen.json',dict(pending=[]))
                ip.st.sealed_json(a.run_root/'evaluation_complete.json',dict(identity='fixture',prediction_seal=frozen['seal'],
                    files={n:ip.st.sha(a.run_root/n) for n in ('setting_metrics.csv','metrics.csv')}))
                with patch.object(ip,'signature',return_value='fixture'):reporting.summarize(a)
                report=(a.run_root/'objective_report.md').read_text(encoding='utf-8')
                self.assertEqual(report.count('image_AUROC 六设置：改善6，下降0，不变0。'),3)
                expected='image_AP 六设置：改善3，下降3，不变0。' if mixed_AP else 'image_AP 六设置：改善6，下降0，不变0。'
                self.assertEqual(report.count(expected),3);self.assertNotIn('改善9',report)

    def test_region_resume_and_preflight_with_detached_original_features(self):
        row=dict(relative_path='train/normal.png',image_sha256='original',image_height=4,image_width=6)
        ex=dict(checkpoint='original',preprocessing='original');rms=dict(seal='Source_RMS',RMS=[1]*4)
        q=[np.repeat(np.arange(6,dtype=np.float32)[:,None],1280,axis=1) for _ in p.LAYERS]
        x=SimpleNamespace(extraction=ex,train=[],calibration=[row],supports=lambda *_:[],samples=lambda *_:[],region_rows=lambda:[row],verification=[])
        for streamed in (False,True):
            with self.subTest(streamed=streamed),tempfile.TemporaryDirectory() as tmp,patch.dict(p.CONFIG,regions=2),patch.object(ip,'assets',return_value=x):
                a=SimpleNamespace(cache_root=Path(tmp)/'cache',run_root=Path(tmp)/'run',parent_run=Path(tmp)/'parent',
                    conditional_run=Path(tmp)/'conditional',rms_stat=None,data=Path(tmp)/'data',checkpoint=Path(tmp)/'weights',dinov3_root=Path(tmp)/'backbone')
                ref=dict(path=None if streamed else 'detached_original',seal='original_feature',extraction=ip.st.digest(ex),grid=[2,3],streamed=streamed)
                with patch.object(ip,'feature_ref',return_value=None if streamed else ref),patch.object(ip,'load_features',return_value=(q,np.arange(6),ref)),patch.object(rt,'features',return_value=(q,np.arange(6),ref)),patch('builtins.print'):
                    first,meta=rt.region(a,row,rms)
                self.assertEqual(meta['metadata']['origin']['kind'],'original_forward_streamed' if streamed else 'existing_complete_features')
                before=(rt.region_path(a,row)/'manifest.json').read_bytes()
                # No separate feature_origins file is needed for either source.
                self.assertFalse((a.cache_root/'feature_origins').exists())
                with patch.object(ip,'feature_ref',side_effect=FileNotFoundError('original storage detached')),patch.object(ip,'load_features',side_effect=AssertionError('must not read original Q')),patch.object(rt,'features',side_effect=AssertionError('must not extract')):
                    after,record=rt.region(a,row,rms)
                    self.assertEqual(meta['seal'],record['seal'])
                    for name in first:np.testing.assert_array_equal(first[name],after[name])
                    for array in after.values():array._mmap.close()
                    with patch.object(ip,'freeze'),patch.object(ip,'rms_record',return_value=rms),patch.object(ip,'candidates',return_value=[]):
                        preflight=ip.preflight(a)
                    self.assertEqual(preflight['status'],'inputs_available')
                    self.assertTrue(any(r['status']=='available_derived_regions' for r in ip.csv_rows(a.run_root/'asset_inventory.csv')))
                    with self.assertRaisesRegex(ValueError,'actual inputs'):rt.region(a,row,dict(rms,seal='different_RMS'))
                    with patch.object(ip,'assets',return_value=SimpleNamespace(extraction=dict(ex,checkpoint='other'))):
                        with self.assertRaisesRegex(ValueError,'actual inputs'):rt.region(a,row,rms)
                with patch.object(ip,'feature_ref',side_effect=PermissionError('raw volume inaccessible')):
                    after,record=rt.region(a,row,rms)
                    self.assertEqual(record['seal'],meta['seal'])
                    for array in after.values():array._mmap.close()
                with patch.object(ip,'feature_ref',return_value=dict(ref,seal='replaced_feature')):
                    with self.assertRaisesRegex(ValueError,'actual inputs'):rt.region(a,row,rms)
                self.assertEqual(before,(rt.region_path(a,row)/'manifest.json').read_bytes())

    def test_legacy_region_resume_checks_sealed_dependencies_without_rewriting(self):
        row=dict(relative_path='legacy/query.png',image_sha256='q',image_height=4,image_width=4)
        ex=dict(checkpoint='original');rms=dict(seal='RMS',RMS=[1]*4)
        q=[np.ones((4,1280),np.float32) for _ in p.LAYERS]
        with tempfile.TemporaryDirectory() as tmp,patch.dict(p.CONFIG,regions=2),patch.object(ip,'assets',return_value=SimpleNamespace(extraction=ex)):
            a=SimpleNamespace(cache_root=Path(tmp));arrays=rg.partition(q,np.arange(4),(2,2),[1]*4,target=2)
            meta=ip.st.put(rt.region_path(a,row),arrays,dict(dependencies=ip.region_dependencies(a,row,rms,'old_feature_seal'),seconds=0.,GT_used=False))
            before=(rt.region_path(a,row)/'manifest.json').read_bytes()
            with patch.object(ip,'feature_ref',return_value=None),patch.object(rt,'features',side_effect=AssertionError('must not extract')):
                found=ip.cached_region(a,row,rms);self.assertEqual(found['seal'],meta['seal'])
                after,found=rt.region(a,row,rms)
                for array in after.values():array._mmap.close()
            self.assertEqual(before,(rt.region_path(a,row)/'manifest.json').read_bytes())

    def test_pair_resume_does_not_resolve_or_refit_and_rejects_backend_change(self):
        rng=np.random.default_rng(4);a=SimpleNamespace(cpu_workers=1,device='cpu')
        feature=dict(means=rng.random((4,2,3),dtype=np.float32),area=np.array([.5,.5]))
        meta=dict(seal='original_region');rms=dict(seal='rms',RMS=[1]*4)
        row=dict(relative_path='q.png',image_sha256='q',image_height=2,image_width=2)
        template=dict(relative_path='r.png',image_sha256='r',image_height=2,image_width=2)
        with tempfile.TemporaryDirectory() as tmp,patch.object(rt,'region',return_value=(feature,meta)):
            a.run_root=Path(tmp)
            with rt.Coverage(a,rms) as engine:first=engine.score(row,[template])
            with patch.object(lp,'solve',side_effect=AssertionError('should reuse')):
                with rt.Coverage(a,rms) as engine:second=engine.score(dict(row,label=1,gt_sha256='changed_GT'),[template])
            self.assertEqual(first,second);a.device='cuda:0'
            with rt.Coverage(a,rms) as engine:
                with self.assertRaisesRegex(ValueError,'dependencies'):engine.score(row,[template])
            path=rt.pair_path(a,row,template);value=json.loads(path.read_text());value['result']['objective']=99;path.write_text(json.dumps(value))
            a.device='cpu'
            with rt.Coverage(a,rms) as engine:
                with self.assertRaisesRegex(ValueError,'modified'):engine.score(row,[template])

    def test_two_domain_same_sample_id_never_collides(self):
        a=SimpleNamespace(run_root=Path('test_run'))
        q=dict(relative_path='test/a/x.png',image_sha256='x',image_height=2,image_width=2);other=dict(q,relative_path='test/b/x.png')
        self.assertNotEqual(rt.pair_path(a,q,q),rt.pair_path(a,other,q))

    def test_query_sample_schema_and_extraction_binding(self):
        row=dict(relative_path='test/a.png',image_sha256='a',image_height=2,image_width=2,domain='background',sample_id='test_0000')
        ex=dict(checkpoint='pinned',preprocess='original')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'queries/background/test_0000'
            record=ip.st.put(path,{f'L{l}':np.ones((4,1280),np.float32) for l in p.LAYERS},
                dict(sample=row,identity='old_producer',geometry=dict(grid=[2,2],positions=[2,0,3,1])))
            auth=dict(run=root,cache=root,identity='old_producer',index={},seal='authority')
            a=SimpleNamespace(cache_root=root/'new',feature_roots=[])
            with patch.object(ip,'assets',return_value=SimpleNamespace(extraction=ex)),patch.object(ip,'authorities',return_value=[auth]):
                ref=ip.feature_ref(a,row);self.assertEqual(ref['seal'],record['seal'])
                q,pos,r=ip.load_features(a,row);np.testing.assert_array_equal(pos,[2,0,3,1]);self.assertEqual(q[0].shape,(4,1280))
                for array in q:array._mmap.close()
                del q,pos
            # Inline declarations from another preprocessing/checkpoint are rejected.
            bad=root/'bad';ip.st.put(bad,{**{f'q{l}':np.ones((4,1280),np.float32) for l in p.LAYERS},'positions':np.arange(4,dtype=np.int64)},
                dict(image=row,extraction=dict(ex,checkpoint='other'),grid=[2,2]))
            with patch.object(ip,'assets',return_value=SimpleNamespace(extraction=ex)),patch.object(ip,'candidates',return_value=iter([(bad,None,None)])):
                with self.assertRaisesRegex(ValueError,'extraction'):ip.feature_ref(a,row)

    def test_RMS_record_rejects_calibration_population_even_if_count_matches(self):
        contract=dict(images=[dict(relative_path=f'train/{i}',image_sha256=str(i),image_height=1,image_width=1) for i in range(455)],
            extraction=dict(checkpoint='fixed'),definition=p.CONFIG['RMS'])
        record=dict(contract=contract,records=[dict(dependencies=dict(image=im,extraction=ip.st.digest(contract['extraction']),definition=p.CONFIG['RMS']),
            feature_seal='raw',patches=1,squared_sums=[1280]*4) for im in contract['images']],RMS=[1]*4)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'RMS.json';path.write_text('{}');a=SimpleNamespace(rms_stat=path)
            with patch.object(ip,'small',return_value=record),patch.object(ip,'rms_contract',return_value=contract):
                self.assertEqual(ip.rms_record(a)['RMS'],[1]*4)
                record['records'][0]['dependencies']['image']=dict(contract['images'][0],relative_path='calibration/not_allowed')
                with self.assertRaisesRegex(ValueError,'mixed'):ip.rms_record(a)

    def test_backend_freeze_cannot_reuse_CPU_smoke_for_CUDA(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=SimpleNamespace(run_root=Path(tmp),parent_run=Path(tmp)/'parent',conditional_run=Path(tmp)/'conditional',raw_run=Path(tmp)/'raw',
                cache_root=Path(tmp)/'cache',asset_runs=[],feature_roots=[],feature_manifest=None,rms_stat=None,device='cpu',stage='smoke')
            with patch.object(ip,'assets',return_value=SimpleNamespace(identity='fixed_assets')):
                saved=ip.freeze(a);a.device='cuda:0'
                with self.assertRaisesRegex(ValueError,'immutable'):ip.freeze(a)

    def test_duration_boundary_is_not_a_resource_admission_gate(self):
        import time
        with tempfile.TemporaryDirectory() as tmp:
            a=SimpleNamespace(run_root=Path(tmp),_started=time.perf_counter(),allow_over_12h=False)
            with self.assertRaisesRegex(RuntimeError,'12 hours'):pipeline.budget(a,dict(known_pending_seconds=50000,unknown_stages=[]))
            self.assertTrue((a.run_root/'budget_pause.json').is_file())
            ip.st.write_json(a.run_root/'costs/completed.json',dict(seconds=10000))
            with self.assertRaisesRegex(RuntimeError,'12 hours'):pipeline.budget(a,dict(known_pending_seconds=35000,unknown_stages=[]))

    def test_prediction_calibration_freeze_and_label_change_replay(self):
        import time
        def row(name):return dict(relative_path=name,image_sha256=name,image_height=4,image_width=4)
        calibrators=[dict(row(f'train/{i}'),calibration=True,index=i) for i in range(66)]
        queries={d:[dict(row(f'test/{d}/q'),sample_id='test_0000',label=0)] for d in p.DOMAINS}
        templates=[row(f'support/{i}') for i in range(8)]
        assets=SimpleNamespace(calibration=calibrators,samples=lambda d:queries[d],supports=lambda *_:templates,
            query={(d,s,'test_0000'):dict(map_hash='old_map') for d in p.DOMAINS for s in p.SEEDS})
        def observation(a,d,s,kind,r,engine):
            i=r.get('index',70)
            value=dict(kind=kind,P=.1*i,cap=1+.01*i*i,free=.1+.005*i,
                image=ip.image(r),domain=d,support_seed=s,templates=[],GT_used=False)
            return dict(value,seal=ip.st.digest(value))
        class Engine:
            def __init__(self,*_):self.new_pairs=0
            def __enter__(self):return self
            def __exit__(self,*_):pass
        with tempfile.TemporaryDirectory() as tmp:
            a=SimpleNamespace(run_root=Path(tmp),_started=time.perf_counter(),allow_over_12h=False)
            with patch.object(ip,'assets',return_value=assets),patch.object(ip,'signature',return_value='frozen_input'),patch.object(rt,'source_rms',return_value={}),patch.object(rt,'Coverage',Engine),patch.object(pipeline,'observation',side_effect=observation) as call,patch('builtins.print'):
                first=pipeline.predict(a);self.assertEqual(call.call_count,6*67)
                frozen=ip.st.sealed_json(a.run_root/'predictions_frozen.json')
                self.assertEqual(len(frozen['records']),6);self.assertFalse(frozen['pending'])
                for d in p.DOMAINS:queries[d][0]['label']=1
                second=pipeline.predict(a);self.assertEqual(first,second)
                self.assertEqual(frozen,ip.st.sealed_json(a.run_root/'predictions_frozen.json'))

    def test_review_excludes_arrays_and_scratch(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=SimpleNamespace(run_root=Path(tmp)/'new');a.run_root.mkdir()
            (a.run_root/'status.json').write_text('{}');(a.run_root/'pixel_scratch').mkdir()
            (a.run_root/'pixel_scratch/a.bin').write_bytes(b'123');(a.run_root/'large.npy').write_bytes(b'123')
            import tarfile
            r=reporting.pack(a)
            with tarfile.open(r['archive']) as t:self.assertEqual(t.getnames(),['new/status.json'])

    def test_no_Torch_import_in_preflight_process(self):
        import subprocess,sys
        r=subprocess.run([sys.executable,'-B','-c','import sys; from DINOv3.MADEqual.aebad_s_reverse_capacity_coverage_p0 import runner; assert "torch" not in sys.modules'],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)

    def test_real_compact_parent_P_scores_and_RMS_source_membership(self):
        root=ip.ROOT/'server_results/aebad_conditional_image_score_20261102_224335'
        if not root.exists():self.skipTest('read-only completed parent metadata not included in code checkout')
        a=runner.parser().parse_args([]);a.run_root=ip.ROOT/'DINOv3/outputs/mad_equal/_unused_coverage_probe';a=runner.normalize(a)
        x=ip.assets(a);self.assertEqual(len(x.train),455);self.assertEqual(len(x.query),1804);self.assertEqual(len(x.normal),396)
        self.assertEqual(len(x.region_rows()),1016);self.assertTrue(all(not r['calibration'] for r in x.train))
        self.assertLessEqual(max(r['CH1_abs_error'] for r in x.verification),1e-5)


if __name__=='__main__':unittest.main()
