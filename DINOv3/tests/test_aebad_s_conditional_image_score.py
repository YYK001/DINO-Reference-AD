"""CPU mathematical, provenance and six-setting fixture tests. No CUDA/model."""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from DINOv3.MADEqual.aebad_s_conditional_image_score_p0 import (
    protocol as cfg,numerics as nm,inputs as ip,runtime as rt,pipeline as pipe,reporting as report,runner)
from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import image_metrics
st=ip.st
RealEngine=rt.Engine
RealResponse=ip.response


class Math(unittest.TestCase):
    def model(self):
        bs=np.linspace(-2,3,66);ps=2*bs+1;bt=np.linspace(-1,2,8);pt=bt+4
        return nm.fit(bs,ps,bt,pt)

    def test_fixed_scope(self):
        p=cfg.plan();self.assertEqual((p['normal_statistics'],p['Source_statistics'],p['Target_LOO_statistics']),(444,396,48))
        self.assertEqual((p['unique_evaluation_images'],p['image_support_instances'],p['image_evaluation_cells']),(902,1804,18))
        self.assertEqual(p['new_AUPRO_jobs'],0)

    def test_ddof0_and_actual_variance(self):
        m=self.model();b=np.linspace(-2,3,66)
        self.assertEqual(m['sigma_BS'],float(np.std(b,ddof=0)))
        self.assertAlmostEqual(m['beta_S'],2*np.std(b,ddof=0));self.assertAlmostEqual(m['v_S'],1.)
        self.assertNotEqual(m['sigma_BS'],float(np.std(b,ddof=1)))

    def test_closed_form_minimizes_both_constrained_objectives(self):
        bs=np.linspace(-2,3,66);ps=2*bs+np.sin(bs);bt=np.linspace(-1,2,8);pt=bt+4
        m=nm.fit(bs,ps,bt,pt);xs=(bs-m['mu_BS'])/m['sigma_BS'];xt=(bt-m['mu_BS'])/m['sigma_BS']
        fs=lambda beta:np.mean(((ps-ps.mean())-beta*(xs-xs.mean()))**2)
        ft=lambda beta:np.mean(((pt-pt.mean())-beta*(xt-xt.mean()))**2)+(beta-m['beta_S'])**2
        for key,f in (('beta_S',fs),('beta_ST',ft)):
            for value in (0.,m[key]*.5,m[key]+.1,m[key]+1.):self.assertLessEqual(f(m[key]),f(value)+1e-12)

    def test_negative_covariance_and_constraint(self):
        b=np.linspace(-1,1,66);bt=np.linspace(-3,3,8);m=nm.fit(b,-b,bt,-100*bt)
        self.assertEqual(m['beta_S'],0);self.assertEqual(m['beta_ST'],0)
        self.assertTrue(m['Source_constraint_triggered'] and m['Target_constraint_triggered'])

    def test_Target_zero_variance_returns_Source_slope(self):
        b=np.linspace(0,1,66);m=nm.fit(b,b,np.ones(8),np.arange(8))
        self.assertEqual(m['v_T'],0);self.assertEqual(m['beta_ST'],m['beta_S'])

    def test_Source_scale_degenerate_never_divides_and_returns_P(self):
        b=np.ones(66)+np.arange(66)*1e-17;m=nm.fit(b,np.arange(66),np.ones(8),np.arange(8))
        self.assertTrue(m['degenerate']);self.assertIsNone(m['v_T'])
        p=np.array([-0.,1.,-2.]);sc,x=nm.scores(np.array([1.,2.,3.]),p,m)
        for s in sc.values():np.testing.assert_array_equal(s,p)
        np.testing.assert_array_equal(x,np.zeros(3))

    def test_nonfinite_inputs_are_errors_even_when_degenerate(self):
        for source,target in ((np.full(66,np.nan),np.ones(8)),(np.ones(66),np.full(8,np.inf))):
            with self.assertRaises(ValueError):nm.fit(source,np.ones(66),target,np.ones(8))

    def test_beta0_exact_and_negative_scores_not_clipped(self):
        m=self.model();p=np.array([1.,2.,3.]);m['beta_S']=m['beta_ST']=0
        out,_=nm.scores(np.arange(3),p,m)
        for s in out.values():np.testing.assert_array_equal(s,p)
        m['beta_ST']=20;out,_=nm.scores(np.array([5.,6.,7.]),p,m)
        self.assertTrue((out[cfg.TARGET]<0).all())

    def test_influence_holds_Source_fixed_and_only_drops_statistic(self):
        m=self.model();before=copy.deepcopy(m);bt=np.linspace(-1,2,8);pt=bt+4
        out=nm.influence(bt,pt,m);self.assertEqual(len(out),8);self.assertEqual(m,before)
        for j,r in enumerate(out):
            expected=nm.target_moments(np.delete(bt,j),np.delete(pt,j),m)
            self.assertEqual(r['beta_ST'],expected['beta_ST']);self.assertEqual(r['target_count'],7)

    def test_native_even_median_and_no_second_MAD(self):
        z=np.tile(np.arange(-8,8,dtype=np.float32),(4,1));mad=dict(patch=dict(medians=[20]*4,scales=[2]*4),image={k:dict(medians=[0]*4,scales=[1]*4) for k in ('peak','local')})
        pos=np.arange(16,dtype=np.int64);raw=z*2+20
        first=nm.observation(z,pos,[4,4],[7,9],mad);second=nm.observation(raw,pos,[4,4],[7,9],mad,'raw_distance')
        self.assertEqual(first['B'],-.5);self.assertEqual(first['P'],second['P']);self.assertEqual(first['B'],second['B'])
        perm=np.arange(16)[::-1];third=nm.observation(z[:,perm],pos[perm],[4,4],[7,9],mad)
        self.assertEqual(first,third)
        for bad in (np.array([0]*16,np.int64),np.arange(15,dtype=np.int64)):
            with self.assertRaises(ValueError):nm.observation(z,bad,[4,4],[7,9],mad)

    def test_AP_ties_and_ranking_error_mass(self):
        y=[0,0,1,1];p=[1.,2.,1.,3.];new=[0.,2.,1.,3.]
        m=image_metrics(y,p);self.assertAlmostEqual(m['image_AUROC'],.625)
        self.assertAlmostEqual(m['image_AP'],.75)
        rank=nm.ranking(y,p,new)
        self.assertEqual(rank['pairs'],4);self.assertEqual(rank['net_error_change'],-.5)
        self.assertAlmostEqual(image_metrics(y,new)['image_AUROC']-m['image_AUROC'],-rank['net_error_change']/4)

    def test_hierarchical_macro_does_not_pool_repeated_images(self):
        data=[dict(domain=d,support_seed=s,branch=b,**{k:(i+1)/10+s/10000 for k in ('image_AUROC','image_AP','pixel_AUROC','pixel_AP','AUPRO')})
            for i,d in enumerate(cfg.DOMAINS) for s in cfg.SEEDS for b in cfg.BRANCHES]
        result=report.aggregate(data);self.assertEqual(len(result),36)
        g=next(r for r in result if r['domain']=='macro_3domains_2seeds' and r['branch']==cfg.TARGET)
        self.assertAlmostEqual(g['image_AUROC'],np.mean([r['image_AUROC'] for r in data]))
        self.assertEqual(len(report.comparisons(result)),72)


class Geometry(unittest.TestCase):
    def test_K7_filters_feature_list_together_with_distances(self):
        n=9;q=np.arange(n*1280,dtype=np.float32).reshape(n,1280)/10000
        parts=[{l:np.full((1,1280),j/10,np.float32) for l in cfg.LAYERS} for j in range(8)]
        selected=rt.selected_indices(3);nn={l:dict(ds=np.full(n,.01,np.float32),
            dt=np.stack([np.linalg.norm(q-parts[j][l][0],axis=1)/1280 for j in selected]),
            indices=np.zeros((7,n),np.int64),support_indices=selected) for l in cfg.LAYERS}
        from DINOv3.MADEqual.aebad_s_target_local_segment.geometry import geometry_layer
        with patch.object(rt,'geometry_layer',wraps=geometry_layer) as f:
            result=rt.segment(nn,parts,3)
        self.assertEqual(result.shape,(4,n))
        for call in f.call_args_list:
            self.assertEqual(len(call.args[3]),7)
            for i,j in enumerate(selected):np.testing.assert_array_equal(call.args[3][i],parts[j][7])
        nn[7]['support_indices']=list(range(7))
        with self.assertRaisesRegex(ValueError,'identical retained support order'):rt.segment(nn,parts,3)


class Fixture:
    def __init__(self,root):
        self.root=Path(root);a=runner.parser().parse_args(['--run-root',str(self.root/'run'),'--cache-root',str(self.root/'cache'),'--parent-run',str(self.root/'parent'),'--data',str(self.root/'no_GT')])
        self.a=a;self.rows={};self.supports={};self.old={};self.maps={};self.settings={}
        source=[dict(relative_path=f'train/cal_{i:02d}.png',image_sha256=f's{i}',image_height=16,image_width=16,calibration=True,source_index=i) for i in range(66)]
        mad=dict(patch=dict(medians=[0]*4,scales=[1]*4),image={k:dict(medians=[0]*4,scales=[1]*4) for k in ('peak','local')})
        for d in cfg.DOMAINS:
            self.rows[d]=[dict(domain=d,sample_id=f'test_{j}',relative_path=f'test/{d}/{j}.png',image_sha256=f'{d}{j}',image_height=16,image_width=16,label=j) for j in range(2)]
            self.supports[d]={str(s):[dict(domain=d,sample_id=f'T{s}_{j}',relative_path=f'test/{d}/T{s}_{j}.png',image_sha256=f'T{d}{s}_{j}',image_height=16,image_width=16,label=0) for j in range(8)] for s in cfg.SEEDS}
        self.p=SimpleNamespace(root=self.root/'parent',identity='parent',inputs=dict(source=dict(path=str(self.root/'original_Source'),seal='Source'),
            raw_cache=str(self.root/'raw'),parent_run=str(self.root/'sacp'),parent_cache=str(self.root/'oldcache'),
            Source_calibration=source,extraction={'checkpoint':'frozen'},mad=mad),
            common=dict(seal='common',domains={d:dict(supports=self.supports[d]) for d in cfg.DOMAINS}),samples=lambda d:self.rows[d])
        self.p.record=lambda d,s,row:self.old[d,s,row['sample_id']];self.p.maps=self.maps;self.p.settings=self.settings
        self.p.known_metrics=lambda d,s,w:[dict(pixel_AUROC=.9,pixel_AP=.3,AUPRO=.8)]
        a._joint_parent=self.p;runner.normalize(a);ip.freeze(a)
        for d in cfg.DOMAINS:
            for s in cfg.SEEDS:
                group=[]
                for row in self.rows[d]:
                    c=ip.contract(a,d,s,'query',row);value=self.value(c,row);score=value['P'];group.append(score)
                    self.old[d,s,row['sample_id']]=dict(scores={'SEGMENT_K8':{'CH1':score}},map_hashes={cfg.MAP:f'map/{d}/{s}/{row["sample_id"]}'},seal=f'parent/{d}/{s}/{row["sample_id"]}')
                self.maps[d,s]={(r['sample_id'],cfg.MAP):dict(map_float32_sha256=self.old[d,s,r['sample_id']]['map_hashes'][cfg.MAP]) for r in self.rows[d]}
                self.settings[d,s]=dict(metrics=[dict(branch='SEGMENT_K8',scoring='CH1',**image_metrics([0,1],group))])

    def value(self,c,row):
        i=int(''.join(x for x in row['relative_path'].split('/')[-1] if x.isdigit()) or '0')
        base=(i+1)/66 if c['kind']=='Source' else (i%10+1)/8
        z=np.full((4,16),base,np.float32);z[:,0]+=base*(.2+cfg.DOMAINS.index(c['domain'])+.01*c['support_seed'])
        return nm.observation(z,np.arange(16,dtype=np.int64),[4,4],[4,4],self.p.inputs['mad'])

    def context(self):
        f=self
        class Engine:
            def __init__(self,a):pass
            def acquire(self,c,row):
                value=f.value(c,row);b=value['layer_medians'][0];z=np.full((4,16),b,np.float32)
                z[:,0]=value['peak'];return z,np.arange(16,dtype=np.int64),[4,4],'standardized_z',{'kind':'fixture_not_real_retrieval'}
        stack=ExitStack();stack.enter_context(patch.object(rt,'Engine',Engine))
        stack.enter_context(patch.object(ip,'response',return_value=None));return stack


class Flow(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.f=Fixture(self.tmp.name);self.a=self.f.a;self.stack=self.f.context()
    def tearDown(self):self.stack.close();self.tmp.cleanup()

    def normals(self):pipe.run_observations(self.a,'normal');pipe.fit(self.a)

    def test_normal_models_are_setting_specific_no_query_access_or_labels(self):
        pipe.run_observations(self.a,'normal')
        with patch.object(ip,'samples',side_effect=AssertionError('fit must not inspect test population')):pipe.fit(self.a)
        model=st.sealed_json(self.a.run_root/'coefficients.json')['models'];self.assertEqual(len(model),6)
        self.assertGreater(len({r['beta_S'] for r in model.values()}),1)
        influence=report.rows(self.a.run_root/'coefficient_influence.csv');self.assertEqual(len(influence),48)
        normal=st.sealed_json(self.a.run_root/'normal_statistics.json');self.assertEqual(len(normal['records']),444)
        for ref in normal['records']:
            c=st.sealed_json(Path(ref['path']))['contract'];self.assertEqual(c['K'],8 if c['kind']=='Source' else 7)

    def test_full_image_flow_resume_and_same_localization_no_pixel_backend(self):
        self.normals();pipe.run_observations(self.a,'query');pipe.predict(self.a)
        frozen=(self.a.run_root/'predictions_frozen.json').read_bytes();before=(self.a.run_root/'coefficients.json').read_bytes()
        pipe.predict(self.a);self.assertEqual(frozen,(self.a.run_root/'predictions_frozen.json').read_bytes())
        first=pipe.evaluate(self.a);self.assertEqual(first['new_AUPRO_jobs'],0)
        self.assertTrue(pipe.evaluate(self.a)['reused'])
        status=report.summarize(self.a);self.assertEqual(status['status'],'six_settings_image_evaluation_complete')
        self.assertEqual(len(report.rows(self.a.run_root/'metrics.csv')),36)
        self.assertEqual(before,(self.a.run_root/'coefficients.json').read_bytes())
        m=report.rows(self.a.run_root/'setting_metrics.csv')
        for row in m:self.assertEqual((row['pixel_AUROC'],row['pixel_AP'],row['AUPRO']),('0.9','0.3','0.8'))
        missing=report.rows(self.a.run_root/'missing_area.csv');self.assertTrue(missing)
        with patch.object(rt.Engine,'acquire',side_effect=AssertionError('must reuse committed observations')):
            pipe.run_observations(self.a,'query')
        files=list(self.a.run_root.glob('scatters/*.png'));self.assertEqual(len(files),6)
        metric=self.a.run_root/'setting_metrics.csv';metric.write_text(metric.read_text().replace('0.9','1.5',1))
        with self.assertRaisesRegex(ValueError,'completed small image evaluation changed'):pipe.evaluate(self.a)

    def test_query_label_change_does_not_change_coefficients_or_scores(self):
        self.normals();pipe.run_observations(self.a,'query');pipe.predict(self.a)
        before=(self.a.run_root/'coefficients.json').read_bytes();scores=(self.a.run_root/'per_image_scores.csv').read_bytes()
        for pop in self.f.rows.values():
            for row in pop:row['label']=1-row['label']
        pipe.fit(self.a);pipe.predict(self.a)
        self.assertEqual(before,(self.a.run_root/'coefficients.json').read_bytes());self.assertEqual(scores,(self.a.run_root/'per_image_scores.csv').read_bytes())

    def test_evaluation_rejects_before_global_score_freeze(self):
        with patch.object(ip.jp,'GTData',side_effect=AssertionError('GT must not be accessed')):
            with self.assertRaises(FileNotFoundError):pipe.evaluate(self.a)

    def test_baseline_mismatch_stops_without_relaxation(self):
        self.f.old['background',42,'test_0']['scores']['SEGMENT_K8']['CH1']+=1e-4
        with self.assertRaisesRegex(ValueError,'no relaxed tolerance'):pipe.run_observations(self.a,'query')
        checks=report.rows(self.a.run_root/'failed_numerical_checks.csv');self.assertEqual(float(checks[0]['atol']),1e-5)
        self.assertFalse((self.a.run_root/'query_statistics.json').exists())

    def test_old_Source_only_or_UNION_observation_not_admitted(self):
        d,s,kind,row,j=next(ip.normal_units(self.a));c=ip.contract(self.a,d,s,kind,row,j)
        old=dict(self.f.value(c,row),contract=dict(c,references=[],K=0,branch='UNION_K8'),GT_used=False)
        path=self.f.root/'bad_export.json';st.sealed_json(path,dict(schema='conditional_exports_v1',records={st.digest(c):old}))
        self.a.statistics_provider=path
        with self.assertRaisesRegex(ValueError,'contract mismatch'):ip.bp(self.a,c)

    def test_normal_paths_never_call_test_baseline_adapter(self):
        with patch.object(ip.jp.audit,'inspect',side_effect=AssertionError('test-only adapter')):
            for d,s,kind,row,j in list(ip.normal_units(self.a))[:2]:self.assertIsNone(ip.response(self.a,ip.contract(self.a,d,s,kind,row,j)))

    def test_read_only_legacy_native_bridge_checks_actual_reference_dependencies(self):
        d,s,k,row,j=next(ip.normal_units(self.a));c=ip.contract(self.a,d,s,k,row,j)
        dep=dict(image=row,source=c['source'],extraction=self.f.p.inputs['extraction'],supports=ip.supports(self.a,d,s),
            geometry=dict(distance='original_Exact_FP32_euclidean_divide_1280_once',segment_geometry=c['geometry']))
        path=self.f.root/'old_native';rec=st.put(path,dict(raw_segment=np.ones((4,16),np.float32),positions=np.arange(16,dtype=np.int64)),
            dict(dependencies=dep,grid=[4,4]))
        export=self.f.root/'native_export';ref=dict(contract=c,path=str(path),seal=rec['seal'],field='raw_segment',encoding='raw_distance')
        st.sealed_json(export,dict(schema='conditional_exports_v1',records={st.digest(c):ref}));self.a.response_provider=export
        before=(path/'manifest.json').read_bytes()
        self.assertEqual(RealResponse(self.a,c)['seal'],rec['seal'])
        self.assertEqual((path/'manifest.json').read_bytes(),before)
        wrong=copy.deepcopy(c);wrong['references']=wrong['references'][:-1];wrong['K']=7
        self.a._exports={};export2=self.f.root/'wrong_native_export';ref['contract']=wrong
        st.sealed_json(export2,dict(schema='conditional_exports_v1',records={st.digest(wrong):ref}));self.a.response_provider=export2
        with self.assertRaisesRegex(ValueError,'actual dependencies differ'):RealResponse(self.a,wrong)

    def test_backend_or_input_export_change_refuses_same_run(self):
        before=(self.a.run_root/'inputs.json').read_bytes();self.a.device='cuda:0'
        with self.assertRaisesRegex(ValueError,'immutable record differs'):ip.freeze(self.a)
        self.assertEqual(before,(self.a.run_root/'inputs.json').read_bytes())

    def test_unsealed_or_modified_small_statistics_are_not_reenrolled(self):
        d,s,k,row,j=next(ip.normal_units(self.a));r=rt.observe(self.a,d,s,k,row,j,rt.Engine(self.a))
        path=ip.record_path(self.a,r['contract']);value=json.loads(path.read_text());value['B']+=1;path.write_text(json.dumps(value))
        with self.assertRaises(ValueError):ip.bp(self.a,r['contract'])

    def test_inline_complete_features_source_row_without_domain_and_bad_extraction(self):
        row=self.f.p.inputs['Source_calibration'][0];path=self.a.raw_cache/'normal_features'/ip.jp.previous.layer_inputs.old.key(row)
        arrays={f'q{l}':np.zeros((16,1280),np.float32) for l in cfg.LAYERS};arrays['positions']=np.arange(16,dtype=np.int64)
        record=st.put(path,arrays,dict(image=row,grid=[4,4],extraction=self.f.p.inputs['extraction'],kind='original_complete_grid'))
        ref=ip.feature_ref(self.a,row);self.assertEqual(ref['seal'],record['seal'])
        manifest=path/'manifest.json';r=st.read_json(manifest);r['metadata']['extraction']={'checkpoint':'different'}
        r['seal']=st.digest({k:v for k,v in r.items() if k!='seal'});st.write_json(manifest,r)
        with self.assertRaisesRegex(ValueError,'extraction configuration differs'):ip.feature_ref(self.a,row)

    def test_Source_search_shared_across_six_settings_with_stable_feature_binding(self):
        engine=RealEngine(self.a);row=self.f.p.inputs['Source_calibration'][0];q={l:np.ones((16,1280),np.float32) for l in cfg.LAYERS}
        source={f'L{l}':np.zeros((1,1280),np.float32) for l in cfg.LAYERS};calls=[]
        class Search:
            def query(self,q):return q
            def exact(self,q,bank,key):calls.append(key);return np.ones(16,np.float32),np.zeros(16,np.int64)
        engine.search=Search();engine.store=SimpleNamespace(load=lambda *args:(source,{'seal':'Source'}))
        m=dict(grid=[4,4],positions=np.arange(16,dtype=np.int64));origin=dict(path='same_features',seal='feature')
        with patch.object(ip,'bank',return_value=dict(path='Source',seal='Source')):
            for d in cfg.DOMAINS:
                for s in cfg.SEEDS:
                    c=ip.contract(self.a,d,s,'Source',row)
                    engine.source_distances(c,row,q,m,dict(origin,proof={'metadata_only':'extra'}),None)
        self.assertEqual(len(calls),4)

    def test_one_query_feature_load_serves_both_support_settings(self):
        engine=RealEngine(self.a);row=self.f.rows['background'][0]
        q={l:np.zeros((16,1280),np.float32) for l in cfg.LAYERS}
        with patch.object(rt,'features',return_value=(q,dict(grid=[4,4],positions=np.arange(16,dtype=np.int64)),{'kind':'fixture'})) as load:
            engine.features(row,False);engine.features(dict(row),False)
        self.assertEqual(load.call_count,1)

    def test_per_support_reconstruction_is_CPU_only_and_needs_no_Source_entity(self):
        d,s,kind,row,j=next(ip.normal_units(self.a));c=ip.contract(self.a,d,s,kind,row,j)
        n=16;q=np.full((n,1280),.5,np.float32)
        parts=[{l:np.full((1,1280),i/10,np.float32) for l in cfg.LAYERS} for i in range(8)]
        distance=np.stack([np.linalg.norm(q-parts[i][7][0],axis=1)/1280 for i in range(8)])
        data=dict(positions=np.arange(n,dtype=np.int64),Source_distance=np.full((4,n),.01,np.float32),
            target_distance=np.stack([distance]*4),target_index=np.zeros((4,8,n),np.int64))
        path=self.f.root/'retrieval';r=st.put(path,data,dict(contract=ip.retrieval_contract(c),grid=[4,4]))
        export=self.f.root/'retrieval_export.json';rc=ip.retrieval_contract(c)
        st.sealed_json(export,dict(schema='conditional_exports_v1',records={st.digest(rc):dict(contract=rc,path=str(path),seal=r['seal'])}))
        self.a.retrieval_provider=export;engine=RealEngine(self.a)
        with patch.object(ip,'response',return_value=None),patch.object(engine,'refs',return_value=SimpleNamespace(parts=parts)),\
             patch.object(ip,'bank',side_effect=AssertionError('cached Source distances are sufficient')),\
             patch.object(engine,'ensure_search',side_effect=AssertionError('no Torch/Exact for cached geometry')):
            raw,pos,grid,encoding,origin=engine.acquire(c,row)
        self.assertEqual(raw.shape,(4,n));self.assertEqual(encoding,'raw_distance')

    def test_twelve_hour_projection_retains_pending_and_no_compute(self):
        samples=[dict(domain=d,support_seed=s,kind=k,execution_path='fresh_Exact_SEGMENT',seconds=100000.) for d in cfg.DOMAINS for s in cfg.SEEDS for k in ('Source','Target','query')]
        st.sealed_json(self.a.run_root/'smoke.json',dict(identity=ip.signature(self.a),samples=samples))
        st.write_json(self.a.run_root/'time_estimate.json',dict(status='asset_smoke_complete',samples=[]))
        with self.assertRaisesRegex(RuntimeError,'fresh consent required'):pipe.budget(self.a)
        self.assertTrue((self.a.run_root/'budget_pause.json').exists())
        self.a.allow_over_12h=True;pipe.budget(self.a)

    def test_smoke_commits_current_paths_and_resume_does_not_repeat_them(self):
        r=pipe.smoke(self.a);self.assertEqual(r['samples'],18)
        measured=st.sealed_json(self.a.run_root/'smoke.json')
        self.assertTrue(all(v['scope']=='committed_real_asset_complete_grid_path' for v in measured['samples']))
        with patch.object(rt.Engine,'acquire',side_effect=AssertionError('no repeated acquisition')):
            self.assertTrue(pipe.smoke(self.a)['reused'])
        self.assertTrue(r['estimate']['unknown_is_not_zero'])

    def test_source_bank_absence_blocks_before_new_feature_forward(self):
        d,s,k,row,j=next(ip.normal_units(self.a));c=ip.contract(self.a,d,s,k,row,j);engine=RealEngine(self.a)
        with patch.object(ip,'response',return_value=None),patch.object(ip,'retrieval',return_value=None),\
             patch.object(ip,'bank',side_effect=FileNotFoundError('original bank absent')) as bank,\
             patch.object(engine,'features',side_effect=AssertionError('no model/forward')):
            with self.assertRaisesRegex(FileNotFoundError,'explicit'):engine.acquire(c,row)
            bank.assert_not_called()
            self.a.supplement_retrieval=True
            with self.assertRaisesRegex(FileNotFoundError,'original bank absent'):engine.acquire(c,row)
            bank.assert_called_once()

    def test_pending_is_not_complete_and_pack_excludes_arrays(self):
        r=report.summarize(self.a);self.assertEqual(r['status'],'pending')
        self.a.run_root.mkdir(exist_ok=True);(self.a.run_root/'bad.bin').write_bytes(b'fixture')
        packed=runner.pack(self.a)
        import tarfile
        with tarfile.open(packed['archive']) as t:self.assertTrue(all(not m.name.endswith('.bin') for m in t.getmembers()))

    def test_report_six_setting_counts_are_separate_for_AUROC_AP_and_ties(self):
        # Report-only fixture: no score generation, fitting or real evaluation.
        st.write_json(self.a.run_root/'evaluation_complete.json',{'fixture':True})
        st.sealed_json(self.a.run_root/'coefficients.json',dict(models={f'{d}/{s}':dict(degenerate=False,beta_ST=1.) for d in cfg.DOMAINS for s in cfg.SEEDS}))
        st.write_csv(self.a.run_root/'score_changes.csv',[dict(domain=d,support_seed=s,branch=cfg.TARGET,group=g,mean_change=0.)
            for d in cfg.DOMAINS for s in cfg.SEEDS for g in ('normal','anomaly')])
        st.write_csv(self.a.run_root/'ranking_summary.csv',[dict(domain=d,support_seed=s,branch=cfg.TARGET,net_error_change=0.)
            for d in cfg.DOMAINS for s in cfg.SEEDS])
        auroc=(.1,.1,.1,-.1,-.1,-.1)
        for ap,expected_ap in ((auroc,'改善3，下降3，不变0'),((.1,.1,-.1,-.1,0.,0.),'改善2，下降2，不变2')):
            data=[]
            for i,(d,s) in enumerate((d,s) for d in cfg.DOMAINS for s in cfg.SEEDS):
                for b in cfg.BRANCHES:
                    data.append(dict(domain=d,support_seed=s,branch=b,image_AUROC=.5+(auroc[i] if b==cfg.TARGET else 0.),
                        image_AP=.5+(ap[i] if b==cfg.TARGET else 0.),pixel_AUROC=.9,pixel_AP=.3,AUPRO=.8))
            st.write_csv(self.a.run_root/'setting_metrics.csv',data)
            with patch.object(pipe,'evaluate',return_value={'status':'fixture_complete'}):report.summarize(self.a)
            lines=(self.a.run_root/'objective_report.md').read_text(encoding='utf-8').splitlines()
            self.assertEqual([x for x in lines if x.startswith('image_AUROC六设置')],
                ['image_AUROC六设置：改善3，下降3，不变0。']*2+['image_AUROC六设置：改善0，下降0，不变6。'])
            self.assertEqual([x for x in lines if x.startswith('image_AP六设置')],
                [f'image_AP六设置：{expected_ap}。']*2+['image_AP六设置：改善0，下降0，不变6。'])

    def test_output_cannot_be_inside_parent_Source_bank(self):
        self.a.run_root=Path(self.f.p.inputs['source']['path'])/'nested_run'
        with self.assertRaisesRegex(ValueError,'separate from parent assets'):runner.normalize(self.a)


class RealMetadata(unittest.TestCase):
    def test_missing_parent_has_a_clear_inventory_without_model_import(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            rc=runner.main(['preflight','--parent-run',str(root/'absent_parent'),'--run-root',str(root/'inventory')])
            self.assertEqual(rc,2);self.assertEqual(report.rows(root/'inventory/missing_inputs.csv')[0]['role'],'parent_metadata')

    def test_parent_counts_and_no_Torch_preflight_import(self):
        p=ip.ROOT/'server_results/aebad_target_loo_two_K8_20261030_013433'
        if not p.exists():self.skipTest('real small parent review is not in source-only checkout')
        import subprocess,sys
        with tempfile.TemporaryDirectory() as temp:
            cmd=[sys.executable,'-B','-m','DINOv3.MADEqual.aebad_s_conditional_image_score_p0.runner','preflight','--parent-run',str(p),'--run-root',str(Path(temp)/'inventory')]
            r=subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(r.returncode,2,r.stdout+r.stderr)
            result=st.read_json(Path(temp)/'inventory/preflight.json')
            self.assertEqual(result['status'],'missing_inputs');self.assertFalse(result['torch_imported'])
            self.assertEqual(result['plan']['normal_statistics'],444)


if __name__=='__main__':unittest.main()
