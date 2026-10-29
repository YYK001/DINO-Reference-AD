"""Fixed-plan, cache and metric orchestration CPU tests; no actual model/CUDA."""
import copy,json,tempfile,subprocess,sys,unittest,io
from contextlib import redirect_stderr,redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from DINOv3.MADEqual.aebad_s_target_loo_support_retest_p0 import selection,inputs as ip,runtime as rt
from DINOv3.MADEqual.aebad_s_target_loo_support_retest_p0.protocol import *
from DINOv3.MADEqual.aebad_s_target_loo_support_retest_p0.evaluation import metric_branches
from DINOv3.MADEqual.aebad_s_target_loo_support_retest_p0.reporting import mean_groups
from DINOv3.MADEqual.aebad_s_target_loo_support_retest_p0.runner import parser,normalize


def catalog_fixture():
    f=ip.st.read_json(ip.ROOT/'DINOv3/MADEqual/aebad_s_reference_residual_segmentation/frozen_origins.json')
    other=ip.st.read_json(ip.ROOT/'DINOv3/MADEqual/aebad_s_segment_pair_crossdomain_p0/frozen_catalog.json')['domains']
    h=ip.st.read_json(Path(selection.__file__).with_name('historical_supports.json'));ids={(r['domain'],r['relative_path']):r['sample_id'] for r in h['records']}
    parents={'background':SimpleNamespace(samples=[dict(r,domain='background') for r in f['background_samples']],supports=[dict(r,label=0,domain='background',sample_id=ids['background',r['relative_path']]) for r in f['background_supports']])}
    for d in ('illumination','view'):parents[d]=SimpleNamespace(samples=other[d]['samples'],supports=[dict(r,label=0,domain=d,sample_id=ids[d,r['relative_path']]) for r in other[d]['supports']])
    return parents


def query_feature_fixture(root,*,sample=None,identity='raw-provider',positions=None,dtype=np.float32):
    row=sample or dict(domain='view',sample_id='query',relative_path='test/good/view/query.png',image_sha256='original-image',label=0,image_height=8,image_width=12)
    arrays={f'q{l}':np.full((6,1280),l,dtype=dtype) for l in LAYERS}
    arrays['positions']=np.asarray([4,1,5,0,3,2] if positions is None else positions,dtype=np.int64)
    path=root/'query';record=ip.st.put(path,arrays,dict(sample=row,identity=identity,grid=[2,3]))
    ref=dict(path=str(path),seal=record['seal'],parent_identity='raw-provider',metadata=record['metadata'],files=record['files'])
    return row,arrays,ref


class TwoK8Tests(unittest.TestCase):
    def test_parent_query_sample_schema_reused_without_normal_kind_or_forward(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);row,arrays,ref=query_feature_fixture(root)
            p=SimpleNamespace(assets={'queries':{'view/query':ref}},extraction={'checkpoint':'frozen'})
            a=SimpleNamespace(cache_root=root/'cache',feature_manifest=None,extract_missing=False)
            before=(Path(ref['path'])/'manifest.json').read_bytes()
            with patch.object(ip,'parent',return_value=p),patch.object(ip.layer_inputs,'feature_ref',side_effect=FileNotFoundError('no normal cache')),patch.object(ip.layer_inputs.old,'inspect_raw',side_effect=AssertionError('query is not a normal bundle')):
                self.assertEqual(ip.feature_ref(a,row),ref)
                q,meta,origin=rt.features(a,row)
            try:
                for l in LAYERS:np.testing.assert_array_equal(q[l],arrays[f'q{l}'])
                np.testing.assert_array_equal(meta['positions'],arrays['positions']);self.assertEqual(meta['grid'],[2,3]);self.assertEqual(origin,ref)
                self.assertFalse(hasattr(a,'_model'));self.assertEqual(getattr(a,'feature_forwards',0),0)
                self.assertEqual(before,(Path(ref['path'])/'manifest.json').read_bytes())
            finally:
                for v in q.values():v._mmap.close()
                meta['positions'].base._mmap.close()
            self.assertNotIn('torch',sys.modules)

    def test_parent_query_wrong_sample_provider_seal_or_layout_is_rejected(self):
        for case in ('sample','provider','seal','positions','dtype'):
            with self.subTest(case=case),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);row,arrays,ref=query_feature_fixture(root,positions=[0]*6 if case=='positions' else None,dtype=np.float64 if case=='dtype' else np.float32)
                actual=dict(row,domain='illumination') if case=='sample' else row
                if case=='provider':ref['parent_identity']='other-provider'
                if case=='seal':ref['seal']='different-seal'
                with self.assertRaises(ValueError):ip.inspect_parent_query(ref,actual)

    def test_original_normal_feature_schema_still_uses_normal_inspector(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);row,arrays,_=query_feature_fixture(root)
            path=root/'normal';record=ip.st.put(path,arrays,dict(image=row,kind='original_complete_grid',grid=[2,3]))
            ref=dict(path=str(path),seal=record['seal']);p=SimpleNamespace(assets={'queries':{}},extraction={})
            a=SimpleNamespace(cache_root=root/'cache',feature_manifest=None)
            with patch.object(ip,'parent',return_value=p),patch.object(ip.layer_inputs,'feature_ref',side_effect=lambda a,r:ip.layer_inputs.old.inspect_raw(ref,r)),patch.object(ip,'inspect_parent_query',side_effect=AssertionError('normal cache is not a query bundle')):
                self.assertEqual(ip.feature_ref(a,row),ref)

    def test_preflight_recognizes_parent_query_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);row,_,ref=query_feature_fixture(root);a=normalize(parser().parse_args(['--run-root',str(root/'run')]))
            a.weight42_runs={d:root/d for d in DOMAINS};groups={}
            for d in DOMAINS:
                supports=catalog_fixture()[d].supports
                groups[d]=dict(supports={'42':supports,'43':supports},common=[row] if d=='view' else [],excluded=[],full_images=9 if d=='view' else 8,common_images=1 if d=='view' else 0,normal_images=1 if d=='view' else 0,anomaly_images=0,overlap=8,seed44_normals_retained=[])
            plan=dict(domains=groups,total_common_images=1)
            p=SimpleNamespace(assets={'queries':{'view/query':ref},'references':{'source':{'path':str(root/'source'),'seal':'original_Source42'}}},extraction={},cache=root/'parent',prediction=lambda row,load:True)
            with patch.object(ip,'parent',return_value=p),patch.object(ip,'weight42',return_value={}),patch.object(ip,'plan',return_value=plan),patch.object(ip.layer_inputs,'feature_ref',side_effect=FileNotFoundError('normal cache absent')):
                result=ip.preflight(a)
            self.assertNotEqual(result['status'],'blocked',result.get('error'))
            self.assertTrue(any(r['role']=='query43_retrieval' and r['sample_id']=='query' for r in result['missing']))
            self.assertFalse(any(r['role']=='query_feature' and r.get('sample_id')=='query' for r in result['missing']))
            self.assertFalse(result['torch_imported'])

    def test_full_pool_history_order_common_population_and_seed44_return(self):
        parents=catalog_fixture();history=ip.st.read_json(Path(selection.__file__).with_name('historical_supports.json'))
        p=selection.select(parents,history,Path('datasets'))
        self.assertEqual(p['total_common_images'],902);self.assertEqual(p['support_seeds'],[42,43]);self.assertEqual(p['Source_seed'],42)
        for d,expected in [('background',(289,77,212)),('illumination',(257,59,198)),('view',(356,76,280))]:
            x=p['domains'][d];self.assertEqual((len(x['common']),x['normal_images'],x['anomaly_images']),expected);self.assertEqual(x['overlap'],0)
            self.assertEqual(x['supports']['42'],parents[d].supports);self.assertEqual(len(x['seed44_normals_retained']),8)
            s={r['relative_path'] for seed in ('42','43') for r in x['supports'][seed]}
            self.assertFalse(any(r['relative_path'] in s for r in x['common']));self.assertTrue(set(x['seed44_normals_retained']).issubset({r['relative_path'] for r in x['common']}))
            self.assertEqual([r['sample_id'] for r in x['common']],[r['sample_id'] for r in parents[d].samples if r['relative_path'] not in s])

    def test_support_overlap_is_allowed_and_not_resampled(self):
        parents={};history={'records':[]};root=Path('datasets')
        for d in DOMAINS:
            rows=[dict(domain=d,sample_id=f'test_{i:04}',relative_path=f'test/good/{d}/{i}.png',label=0,image_sha256=str(i)) for i in range(10)]
            key=lambda r,seed:selection.original_key()(SimpleNamespace(path=root/r['relative_path']),dataset_root=root,domain=d,seed=seed)
            groups={s:sorted(rows,key=lambda r:key(r,s))[:8] for s in SEEDS};support=groups[42]
            parents[d]=SimpleNamespace(supports=support,samples=[r for r in rows if r not in support])
            for seed in SEEDS:history['records'].extend(dict(r,support_seed=seed,support_index=i+1) for i,r in enumerate(groups[seed]))
        p=selection.select(parents,history,root)
        self.assertTrue(all(x['overlap']>=6 for x in p['domains'].values()))

    def test_history_mismatch_is_not_silently_ignored(self):
        h=ip.st.read_json(Path(selection.__file__).with_name('historical_supports.json'));h['records'][0]['image_sha256']='different'
        with self.assertRaises(ValueError):selection.select(catalog_fixture(),h,Path('datasets'))

    def test_setting_keys_include_domain_seed_support_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parents=catalog_fixture();history=ip.st.read_json(Path(selection.__file__).with_name('historical_supports.json'));plan=selection.select(parents,history,Path('datasets'))
            m=ip.st.sealed_json(root/'common_manifest.json',plan);ip.st.sealed_json(root/'inputs.json',dict(identity='run',protocol=CONFIG,common_manifest=m['seal'],Source_seed=42))
            a=SimpleNamespace(run_root=root,cache_root=root/'cache',domain='background',support_seed=42)
            s42=ip.signature(a);p42=ip.cache(a);a.support_seed=43;s43=ip.signature(a)
            self.assertNotEqual(s42['identity'],s43['identity']);self.assertNotEqual(p42,ip.cache(a));self.assertNotEqual(s42['support_id'],s43['support_id'])
            a.domain='view';self.assertNotEqual(ip.signature(a)['identity'],s43['identity'])
            self.assertEqual(ip.st.sealed_json(root/'inputs.json')['Source_seed'],42)

    def test_whole_image_LOO_uses_only_seven_own_group_supports(self):
        class Exact:
            def __init__(self):self.keys=[]
            def query(self,q):return q
            def exact(self,q,bank,key):
                self.keys.append(key);d=np.linalg.norm(q[:,None]-bank[None],axis=-1)/1280;i=d.argmin(1);return d[np.arange(len(q)),i].astype(np.float32),i
        refs=rt.References.__new__(rt.References);refs.search=Exact();refs.source={f'L{l}':np.full((2,1280),100,np.float32) for l in LAYERS};refs.source_seal='original_seed42_bank';refs.support_seals=['own43_'+str(i) for i in range(8)]
        refs.parts=[{l:np.full((1,1280),i,np.float32) for l in LAYERS} for i in range(8)]
        q={l:np.full((1,1280),3,np.float32) for l in LAYERS}
        out=refs.distances(q,heldout=3)
        for v in out.values():self.assertEqual(v['dt'].shape,(7,1));self.assertNotIn(3,v['support_indices']);self.assertGreater(float(v['dt'].min()),0)
        self.assertFalse(any(k[1]=='own43_3' for k in refs.search.keys));self.assertTrue(all(k[0]=='Source' or k[1].startswith('own43_') for k in refs.search.keys))

    def test_seed43_never_falls_back_to_seed42_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=SimpleNamespace(domain='view',support_seed=43,cache_root=Path(tmp),supplement_retrieval=False)
            p=SimpleNamespace(prediction=lambda row:(_ for _ in ()).throw(AssertionError('seed42 must not be used')))
            with patch.object(ip,'context',return_value=(p,[],[])),patch.object(ip,'legacy_ref',return_value=None):
                with self.assertRaises(FileNotFoundError):rt.baseline(a,dict(sample_id='same'))

    def test_four_map_jobs_seven_branches_and_same_seed_combinations(self):
        fields=('image_AUROC','image_AP','pixel_AUROC','pixel_AP','AUPRO')
        base=[dict(domain='view',support_seed=43,branch=b,scoring=sc,**{k:.5 for k in fields}) for b in MAPS for sc in (('CH1','CURRENT') if b in (U,S) else READOUTS)]
        for r in base:
            if r['branch']==S:r['image_AUROC']=.8
            if r['branch']==TARGET:r['pixel_AP']=.6
        rows=metric_branches(base,'view',43);self.assertEqual(len(rows),14);self.assertEqual(len(MAPS)*len(DOMAINS)*len(SEEDS),24)
        for r in rows:
            if r['branch'] in COMBINATIONS or r['branch']==COMBINED:self.assertEqual(r['image_AUROC'],.8);self.assertEqual(r['support_seed'],43)
            if r['branch']=='SEGMENT_SCORE_TARGET_LOO_MAP':self.assertEqual(r['pixel_AP'],.6)

    def test_macro_order_is_domains_then_support_seeds_not_image_pool(self):
        def row(v):return [dict(branch=U,scoring='CH1',**{k:v for k in ('image_AUROC','image_AP','pixel_AUROC','pixel_AP','AUPRO')})]
        per42=mean_groups([row(.1),row(.2),row(.6)],'42');per43=mean_groups([row(.9),row(.4),row(.5)],'43')
        result=mean_groups([per42,per43],'overall');self.assertAlmostEqual(result[0]['image_AUROC'],2.7/6)

    def test_default_preflight_and_all_CPU_imports_no_torch(self):
        code='import sys; from DINOv3.MADEqual.aebad_s_target_loo_support_retest_p0 import inputs,runtime,pipeline,evaluation,reporting; from DINOv3.MADEqual.aebad_s_target_loo_support_retest_p0.runner import parser; assert parser().parse_args([]).stage=="preflight"; assert "torch" not in sys.modules'
        r=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr)

    def test_no_seed44_and_outputs_do_not_overlap_parent(self):
        with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):parser().parse_args(['--support-seed','44'])
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):normalize(parser().parse_args(['--run-root',str(Path(tmp)/'run'),'--evaluation-scratch',str(Path(tmp)/'run'/'pixels')]))

    def test_missing_parents_reported_without_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=normalize(parser().parse_args(['--run-root',str(Path(tmp)/'run')]))
            self.assertEqual(ip.preflight(a)['status'],'blocked')

    def test_seed42_weights_are_copied_and_seed43_Target_weights_are_group_specific(self):
        import time
        from DINOv3.MADEqual.aebad_s_target_loo_support_retest_p0 import pipeline as pp
        from DINOv3.MADEqual.aebad_s_target_loo_layer_weight_p0 import numerics as math
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parents=catalog_fixture();history=ip.st.read_json(Path(selection.__file__).with_name('historical_supports.json'));plan=selection.select(parents,history,Path('datasets'))
            cm=ip.st.sealed_json(root/'common_manifest.json',plan);ip.st.sealed_json(root/'inputs.json',dict(identity='root',protocol=CONFIG,common_manifest=cm['seal']))
            rng=np.random.default_rng(9);ss=rng.random((66,4,3));tt=rng.random((8,4,3));_,s,t=math.moments(ss,tt)
            frozen={b:math.weights(s,t,g)[0].tolist() for b,g in zip(NEW,(0.,.5))}
            provider=root/'provider';records=[dict(kind='Source',index=i,image=f'S{i}',statistics=v.tolist()) for i,v in enumerate(ss)]+[dict(kind='Target',index=i,image=f'oldT{i}',statistics=v.tolist()) for i,v in enumerate(tt)]
            ip.st.sealed_json(provider/'normal_statistics.json',dict(identity='old',records=records))
            ip.st.sealed_json(provider/'weights.json',dict(identity='old',weights=frozen))
            before=(provider/'weights.json').read_bytes()
            a=SimpleNamespace(run_root=root,cache_root=root/'cache',domain='background',support_seed=42,weight42_runs={'background':provider},supplement_retrieval=False)
            p=SimpleNamespace(assets={'references':{'source':{'seal':'original_Source42'}}},extraction={},mad={'patch':dict(medians=[0.]*4,scales=[1.]*4)})
            with patch.object(ip,'weight42',return_value={'weights':frozen}):
                out=pp.prepare_normal_stats(a);w42=pp.fit_weights(a)
                self.assertTrue(out['origin']['read_only']);self.assertFalse(w42['origin']['refitted']);self.assertEqual(w42['weights'],frozen)
                a.support_seed=43;supports=plan['domains']['background']['supports']['43']
                for i,row in enumerate(supports):
                    deps=dict(**ip.signature(a),Source_seed=42,source='original_Source42',query=row,reference_images=[x for j,x in enumerate(supports) if j!=i],extraction={})
                    ip.st.put(ip.cache(a)/'LOO'/f'{i:02d}',dict(raw=rng.random((4,25),dtype=np.float32)*3,positions=np.arange(25,dtype=np.int64)),dict(deps,grid=[5,5],output_shape=[8,8]))
                with patch.object(ip,'context',return_value=(p,supports,[])),patch.object(rt,'References',side_effect=AssertionError('must reuse LOO distances')),patch.object(pp,'guard'):
                    pp.prepare_normal_stats(a);w43=pp.fit_weights(a)
            self.assertEqual(w43['weights'][SOURCE],w42['weights'][SOURCE]);self.assertNotEqual(w43['weights'][TARGET],w42['weights'][TARGET]);self.assertEqual(before,(provider/'weights.json').read_bytes())

    def test_real_CPU_common_pixel_evaluation_does_not_import_old_metrics(self):
        from DINOv3.MADEqual.aebad_s_target_loo_support_retest_p0 import evaluation as ev,pipeline as pp
        from DINOv3.MADEqual.aebad_s_target_loo_layer_weight_p0.composition import METRICS
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=SimpleNamespace(run_root=root,cache_root=root/'cache',evaluation_scratch=root/'pixels',domain='view',support_seed=43)
            images=[dict(domain='view',sample_id='normal',label=0,image_height=8,image_width=8),dict(domain='view',sample_id='anomaly',label=1,image_height=8,image_width=8)]
            group=dict(supports={'42':[],'43':[]},common=images)
            cm=ip.st.sealed_json(root/'common_manifest.json',dict(domains={'view':group}));ip.st.sealed_json(root/'inputs.json',dict(identity='root',protocol=CONFIG,common_manifest=cm['seal']))
            sig=ip.signature(a);frozen=dict(sig,seal='frozen',records=[])
            def output(a,row):
                v=np.full((2,2),.9 if row['label'] else .1,np.float32);maps={b:v.copy() for b in MAPS}
                scores={b:{k:float(v.max()) for k in (('CH1','CURRENT') if b in (U,S) else READOUTS)} for b in MAPS}
                return maps,scores,dict(output_shape=[2,2])
            for row in images:
                maps,scores,_=output(a,row);ip.st.sealed_json(ip.record_path(a,row),dict(map_hashes={b:ev.backend.shared.array_sha(x) for b,x in maps.items()},scores=scores))
            with patch.object(pp,'verified',return_value=frozen),patch.object(pp,'guard'),patch.object(ip,'context',return_value=(None,[],images)),patch.object(rt,'maps',side_effect=output),patch.object(ev,'gt',side_effect=lambda a,row,shape:np.full(shape,row['label'],np.uint8)):
                cpu=ev.evaluate(a);again=ev.evaluate(a)
            self.assertEqual(cpu['independent_map_jobs'],4);self.assertEqual(again['status'],'common_CPU_reused')
            metrics=ip.layer_inputs.old.read_csv(ip.setting(a)/'evaluation/metrics.csv');self.assertEqual(len(metrics),8)
            self.assertTrue(all(float(r['image_AUROC'])==1 and float(r['pixel_AUROC'])==1 for r in metrics))
            self.assertTrue(all(r['metric_origin']=='new_common_population_exact_CPU' for r in metrics))

    def test_legacy_seed43_without_actual_support_source_dependencies_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'legacy';record=ip.st.put(path,dict(U=np.zeros((4,1),np.float32)),dict(dependencies={'Source_seed':43}))
            desc=ip.st.sealed_json(root/'provider.json',dict(records={'view/seed43/test':dict(path=str(path),seal=record['seal'])}))
            a=SimpleNamespace(domain='view',support_seed=43,legacy_provider=root/'provider.json')
            with patch.object(ip,'baseline_dependencies',return_value={'Source_seed':42}):
                with self.assertRaisesRegex(ValueError,'dependencies'):ip.legacy_ref(a,dict(sample_id='test'))

    def test_prediction_partial_resume_and_completed_repeat_are_immutable(self):
        from DINOv3.MADEqual.aebad_s_target_loo_support_retest_p0 import pipeline as pp
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);images=[dict(sample_id='first'),dict(sample_id='second')]
            cm=ip.st.sealed_json(root/'common_manifest.json',dict(domains={'view':dict(supports={'43':[]},common=images)}))
            ip.st.sealed_json(root/'inputs.json',dict(identity='root',protocol=CONFIG,common_manifest=cm['seal']))
            a=SimpleNamespace(run_root=root,cache_root=root/'cache',domain='view',support_seed=43,supplement_retrieval=False)
            w=ip.st.sealed_json(ip.setting(a)/'weights.json',dict(**ip.signature(a),weights={}))
            calls=[]
            def maps(a,row,**kw):
                calls.append(row['sample_id']);x=np.zeros((2,2),np.float32)
                return {b:x for b in MAPS},{b:{r:0. for r in (('CH1','CURRENT') if b in (U,S) else READOUTS)} for b in MAPS},dict(origin='same-group-fake')
            with patch.object(ip,'context',return_value=(None,[],images)),patch.object(ip,'legacy_ref',return_value={'path':'already_compatible'}),patch.object(rt,'maps',side_effect=maps):
                with patch.object(pp,'guard',side_effect=RuntimeError('interrupt after commit')):
                    with self.assertRaisesRegex(RuntimeError,'interrupt'):pp.predict(a)
                self.assertTrue(ip.record_path(a,images[0]).exists());self.assertFalse(ip.record_path(a,images[1]).exists())
                with patch.object(pp,'guard'):
                    result=pp.predict(a);before=(ip.setting(a)/'predictions_frozen.json').read_bytes();repeat=pp.predict(a)
                pp.verified(a)
            self.assertEqual(calls,['first','second']);self.assertEqual(result,repeat);self.assertEqual(before,(ip.setting(a)/'predictions_frozen.json').read_bytes())
            a.support_seed=42
            with self.assertRaises(FileNotFoundError):rt.weights(a)

    def test_seed43_baseline_reuses_original_geometry_and_query_once(self):
        from DINOv3.MADEqual.aebad_s_target_local_segment.geometry import geometry_layer
        from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.calibration import z_values
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=SimpleNamespace(cache_root=root,domain='view',support_seed=43,supplement_retrieval=True)
            p=SimpleNamespace(mad={'patch':dict(medians=[0.]*4,scales=[1.]*4)})
            parts=[{l:np.full((2,1280),j+1,np.float32) for l in LAYERS} for j in range(8)]
            q={l:np.full((3,1280),2.5,np.float32) for l in LAYERS};nn={}
            for l in LAYERS:
                dt=np.stack([np.linalg.norm(q[l]-x[l][0],axis=1)/1280 for x in parts]).astype(np.float32)
                nn[l]=dict(ds=np.ones(3,np.float32),si=np.zeros(3,np.int64),dt=dt,indices=np.zeros((8,3),np.int64))
            refs=SimpleNamespace(parts=parts,distances=lambda q:nn);row=dict(sample_id='test')
            meta=dict(grid=[1,3],output_shape=[2,6],positions=np.array([2,0,1],np.int64))
            with patch.object(ip,'context',return_value=(p,[],[])),patch.object(ip,'legacy_ref',return_value=None),patch.object(ip,'baseline_dependencies',return_value={'Source_seed':42}),patch.object(rt,'features',return_value=(q,meta,{'kind':'fixture'})) as feature:
                x,m=rt.baseline(a,row,True,refs);again,_=rt.baseline(a,row,True,refs)
            expected=[geometry_layer(nn[l]['dt'],nn[l]['indices'],nn[l]['ds'],[v[l] for v in parts])[0]['K8_SEGMENT'] for l in LAYERS]
            np.testing.assert_array_equal(x[S],z_values(np.stack(expected),p.mad['patch']));np.testing.assert_array_equal(x['positions'],meta['positions'])
            self.assertEqual(feature.call_count,1);np.testing.assert_array_equal(again[S],x[S]);self.assertEqual(x['target_distance'].shape,(4,8,3))
            for value in again.values():
                if isinstance(value,np.memmap):value._mmap.close()

    def test_completed_aupro_reuse_and_pending_summary_without_torch(self):
        from DINOv3.MADEqual.aebad_s_target_loo_support_retest_p0 import evaluation as ev,reporting as rp,pipeline as pp
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);images=[dict(sample_id='x')];group=dict(supports={'43':[]},common=images)
            cm=ip.st.sealed_json(root/'common_manifest.json',dict(domains={'view':group}));ip.st.sealed_json(root/'inputs.json',dict(identity='root',protocol=CONFIG,common_manifest=cm['seal']))
            a=SimpleNamespace(run_root=root,domain='view',support_seed=43);sig=ip.signature(a);directory=ip.setting(a)
            maps=[dict(sample_id='x',branch=b) for b in MAPS];ip.st.write_csv(directory/'evaluation/map_manifest.csv',maps)
            fields=('image_AUROC','image_AP','pixel_AUROC','pixel_AP')
            base=[dict(**sig,branch=b,scoring=s,**{k:.7 for k in fields},AUPRO=None) for b in MAPS for s in (('CH1','CURRENT') if b in (U,S) else READOUTS)]
            ip.st.write_csv(directory/'evaluation/metrics.csv',base)
            with patch.object(ev,'restored',return_value={'valid':True}),patch.object(ip,'plan',return_value={'domains':{'view':group}}):
                r=rp.summarize(a);self.assertEqual(r['status'],'pending');self.assertTrue(r['pending'])
                for b in MAPS:
                    value=dict(**sig,branch=b,map_identity=ip.st.digest([r for r in maps if r['branch']==b]),nstrips=200,FPR_max=.3,AUPRO=.6,verified_images=1,backend='original_ADEval_1.1.0_CUDA')
                    ip.st.sealed_json(directory/'aupro'/(b+'.json'),value)
                with patch.object(pp,'verified',return_value={'seal':'frozen'}),patch.object(pp,'guard'),patch.object(ip,'context',return_value=(None,[],images)),patch.object(rt,'maps',side_effect=AssertionError('no replay for completed AUPRO')):
                    self.assertEqual(ev.aupro(a)['jobs'],4)
                r=rp.summarize(a);self.assertEqual(r['status'],'common_five_metrics_complete');self.assertFalse(r['pending'])
            self.assertNotIn('torch',sys.modules)

    def test_six_setting_aggregation_counts_and_GT_identity(self):
        from DINOv3.MADEqual.aebad_s_target_loo_support_retest_p0 import reporting as rp
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=SimpleNamespace(run_root=root,domain='background',support_seed=42)
            ip.st.sealed_json(root/'common_manifest.json',dict(total_common_images=902))
            fields=('image_AUROC','image_AP','pixel_AUROC','pixel_AP','AUPRO')
            def complete(a):
                v=.4+.01*DOMAINS.index(a.domain)+.001*(a.support_seed-42)
                base=[dict(domain=a.domain,branch=b,scoring=s,**{k:v for k in fields}) for b in MAPS for s in (('CH1','CURRENT') if b in (U,S) else READOUTS)]
                return metric_branches(base,a.domain,a.support_seed),{'weights':{SOURCE:[]}}
            for d in DOMAINS:
                for seed in SEEDS:
                    path=root/d/f'seed{seed}';ip.st.write_json(path/'status.json',{'status':'common_five_metrics_complete'})
                    ip.st.write_csv(path/'evaluation/map_manifest.csv',[dict(branch=U,sample_id='same',mask_uint8_sha256='same_GT',output_height=2,output_width=2)])
                    for name in ('weights.csv','u_concentration.csv'):ip.st.write_csv(path/name,[])
            with patch.object(rp,'read_complete',side_effect=complete):
                r=rp.aggregate(a);self.assertEqual(r['complete_settings'],6);self.assertEqual(r['complete_AUPRO_jobs'],24)
                self.assertEqual(len(rp.rows(root/'summary/domain_seed_metrics.csv')),84)
                self.assertEqual(len(rp.rows(root/'summary/Target_information_seed_sensitivity.csv')),30)
                self.assertEqual(len(rp.rows(root/'summary/metric_comparisons.csv')),720)
                overall=rp.rows(root/'summary/overall_macro.csv');self.assertEqual(len(overall),14);self.assertAlmostEqual(float(overall[0]['image_AUROC']),.4105)
                path=root/'view/seed43/evaluation/map_manifest.csv';ip.st.write_csv(path,[dict(branch=U,sample_id='same',mask_uint8_sha256='DIFFERENT',output_height=2,output_width=2)])
                with self.assertRaisesRegex(ValueError,'GT/population'):rp.aggregate(a)

    def test_fast_array_reads_do_not_repeat_payload_hashes_and_pack_excludes_scratch(self):
        from DINOv3.MADEqual.aebad_s_target_loo_support_retest_p0.runner import pack
        import tarfile
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=SimpleNamespace(run_root=root/'run');bundle=root/'array'
            ip.st.put(bundle,dict(x=np.zeros(3,np.float32)),{})
            with patch.object(ip.st,'sha',side_effect=AssertionError('no repeated array hashing')):
                ip.st.get(bundle);ip.st.get(bundle)
            ip.st.write_json(a.run_root/'status.json',{'pending':True});large=a.run_root/'pixel_scratch/left.bin';large.parent.mkdir();large.write_bytes(b'fixture')
            r=pack(a)
            with tarfile.open(r['archive']) as t:self.assertTrue(t.getnames());self.assertFalse(any('scratch' in p or p.endswith('.bin') for p in t.getnames()))

    def test_12h_projection_pauses_without_changing_requested_population(self):
        import time
        from DINOv3.MADEqual.aebad_s_target_loo_support_retest_p0 import pipeline as pp
        with tempfile.TemporaryDirectory() as tmp:
            a=SimpleNamespace(run_root=Path(tmp),domain='view',support_seed=43,_started=time.perf_counter()-120,allow_over_12h=False)
            population={'domains':{'view':{'common_images':356}}};before=copy.deepcopy(population)
            with patch.object(ip,'plan',return_value=population),patch.object(ip,'signature',return_value={'domain':'view','support_seed':43}):
                with self.assertRaisesRegex(RuntimeError,'12h'):pp.guard(a,'predict',5,3560)
                self.assertEqual(population,before);self.assertEqual(ip.st.read_json(a.run_root/'budget_pause.json')['status'],'paused_12h_budget')
                a.allow_over_12h=True;pp.guard(a,'predict',5,3560)


if __name__=='__main__':unittest.main()
