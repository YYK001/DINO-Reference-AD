"""CPU metadata/small-statistic fixtures, not real inference or pixel evaluation."""
import copy,json,tempfile,unittest,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from DINOv3.MADEqual.aebad_s_target_loo_layer_weight_p0 import inputs as ip,pipeline as pp
from DINOv3.MADEqual.aebad_s_target_loo_layer_weight_p0.protocol import *
from DINOv3.MADEqual.aebad_s_target_loo_layer_weight_p0.parent import Parent
from DINOv3.MADEqual.aebad_s_target_loo_layer_weight_p0.composition import fixed_combinations,comparisons,image_scores,METRICS
from DINOv3.MADEqual.aebad_s_target_loo_layer_weight_p0.aggregation import macro
st=ip.st


def fixture(root):
    f=st.read_json(ip.ROOT/'DINOv3/MADEqual/aebad_s_reference_residual_segmentation/frozen_origins.json')
    domains=st.read_json(ip.ROOT/'DINOv3/MADEqual/aebad_s_segment_pair_crossdomain_p0/frozen_catalog.json')['domains']
    catalog=dict(source=[dict(r,source_index=i,calibration=i%8==0) for i,r in enumerate(f['source'])],supports={},samples=[])
    for d in ('illumination','view'):
        catalog['supports'][d]=[dict(r,label=0,domain=d) for r in domains[d]['supports']];catalog['samples'].extend(domains[d]['samples'])
    patch_state=dict(medians=[0.]*4,scales=[1.]*4);image={k:patch_state for k in ('peak','local')}
    mad=st.sealed_json(root/'input_distance_mad.json',dict(identity='parent',patch=patch_state,image=image))
    assets=st.sealed_json(root/'assets.json',dict(references=dict(extraction=dict(checkpoint='fixture'),source=dict(path=str(root/'unmounted_bank'),seal='Source'),mad=dict(patch=patch_state,image=image)),normal_features={}))
    st.sealed_json(root/'catalog.json',dict(catalog=catalog))
    st.sealed_json(root/'inputs.json',dict(identity='parent',config=dict(layers=list(LAYERS),dimension=1280,Source_rows=100000,seed=42),assets_seal=assets['seal'],catalog=st.digest(catalog)))
    allrows=[]
    for d in ('illumination','view'):
        rows=[r for r in catalog['samples'] if r['domain']==d];n=COUNTS[d][0]
        manifest=st.sealed_json(root/'predictions'/(d+'.json'),dict(identity='parent',records=[dict(domain=d,sample_id=r['sample_id'],seal=d+r['sample_id']) for r in rows]))
        base=[dict(domain=d,branch=b,scoring=sc,**{k:.5 for k in METRICS}) for b in (U,S) for sc in ('CH1','CURRENT')]
        st.write_csv(root/'evaluation'/d/'metrics.csv',base)
        st.sealed_json(root/'evaluation'/d/'status.json',dict(identity='parent',prediction_seal=manifest['seal'],images=n,numerical_failed_checks=0,files={'metrics.csv':st.sha(root/'evaluation'/d/'metrics.csv')}))
        allrows.extend(base+[dict(r,branch=COMBINED) for r in base if r['branch']==U])
        for b in (U,S):
            v=dict(identity='parent',verified_images=n,backend='original_ADEval_1.1.0_CUDA',nstrips=200,FPR_max=.3,AUPRO=.5)
            v['seal']=st.digest(v);st.write_json(root/'aupro'/d/(b+'.json'),v)
    st.write_csv(root/'metrics.csv',allrows)
    st.write_json(root/'status.json',dict(identity='parent',status='all_five_metrics_complete',pending=[],numerical_failed_checks=0))
    return catalog


def provider(path,p):
    from DINOv3.MADEqual.aebad_s_target_loo_layer_weight_p0.numerics import moments,weights
    source=[r for r in p.catalog['source'] if r['calibration']];rng=np.random.default_rng(7)
    ss=rng.normal(size=(66,4,3));tt=rng.normal(size=(8,4,3))
    records=[dict(kind='Source',index=r['source_index'],image=r['relative_path'],distance_seal='fixture',statistics=v.tolist()) for r,v in zip(source,ss)]
    records.extend(dict(kind='Target',index=i,image='background'+str(i),distance_seal='fixture',statistics=v.tolist()) for i,v in enumerate(tt))
    inp=st.sealed_json(path/'inputs.json',dict(identity='background',config=CONFIG,source='Source',source_calibration=source,extraction=p.extraction,mad=p.mad['seal']))
    normal=st.sealed_json(path/'normal_statistics.json',dict(identity='background',records=records,quantile_method='numpy.quantile_linear'))
    _,s,t=moments(ss,tt)
    st.sealed_json(path/'weights.json',dict(identity='background',normal_statistics_seal=normal['seal'],weights={b:weights(s,t,g)[0].tolist() for b,g in zip(NEW,(0.,.5))}))
    st.write_json(path/'status.json',dict(identity='background',status='all_five_metrics_complete',pending=[]))


def metric_rows(d,value=.5):
    return [dict(domain=d,branch=b,scoring=sc,**{k:value for k in METRICS}) for b in (U,S,COMBINED,*NEW) for sc in (READOUTS if b in NEW else ('CH1','CURRENT'))]


class DomainTests(unittest.TestCase):
    def test_actual_parent_reader_all_metadata_paths_counts_and_support_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root)
            for d in ('illumination','view'):
                p=Parent(SimpleNamespace(domain=d,parent_run=root,parent_cache=root/'cache'))
                self.assertEqual((len(p.samples),sum(r['label']==0 for r in p.samples),sum(r['label']==1 for r in p.samples)),COUNTS[d])
                self.assertEqual(len(p.supports),8);self.assertTrue(all(r['domain']==d for r in p.samples+p.supports))
                self.assertFalse({r['relative_path'] for r in p.supports}&{r['relative_path'] for r in p.samples})
                self.assertEqual({r['domain'] for r in p.metrics},{d});self.assertEqual(p.cpu['images'],COUNTS[d][0])
                with self.assertRaises(FileNotFoundError) as err:p.prediction(p.samples[0],False)
                self.assertIn(d,Path(err.exception.filename).parts)
                st.write_json(root/'evaluation'/d/'status.json',dict(p.cpu,images=297,seal='invalid'))
                with self.assertRaises(ValueError):Parent(SimpleNamespace(domain=d,parent_run=root,parent_cache=root/'cache'))
                # Restore only this fixture's originally valid record.
                st.write_json(root/'evaluation'/d/'status.json',p.cpu)

    def test_same_named_samples_cannot_overwrite_across_domains(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=SimpleNamespace(cache_root=Path(tmp),domain='illumination');b=SimpleNamespace(cache_root=Path(tmp),domain='view')
            self.assertNotEqual(ip.prediction_path(a,dict(sample_id='test_0')),ip.prediction_path(b,dict(sample_id='test_0')))
            self.assertNotEqual(ip.normal_path(a,'Target',0),ip.normal_path(b,'Target',0))
            with self.assertRaises(ValueError):ip.prediction_path(a,dict(domain='view',sample_id='test_0'))

    def test_source_import_is_dependency_based_read_only_and_excludes_background_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root/'parent');p=Parent(SimpleNamespace(domain='illumination',parent_run=root/'parent',parent_cache=root/'cache'))
            provider(root/'background',p)
            a=SimpleNamespace(domain='illumination',_parent=p,source_stats_run=root/'background')
            before={x.name:x.read_bytes() for x in (root/'background').glob('*.json')}
            shared=ip.source_statistics(a)
            self.assertEqual(len(shared['records']),66);self.assertTrue(all(r['kind']=='Source' for r in shared['records']))
            self.assertEqual(len(ip.units(a)),8);self.assertTrue(all(k=='Target' and r['domain']=='illumination' for k,i,r in ip.units(a)))
            self.assertEqual(before,{x.name:x.read_bytes() for x in (root/'background').glob('*.json')})
            for key in ('source','extraction','mad','source_calibration'):
                path=root/'background/inputs.json';r=st.sealed_json(path);bad=copy.deepcopy(r);bad[key]='different';bad['seal']=st.digest({k:v for k,v in bad.items() if k!='seal'});st.write_json(path,bad)
                with self.assertRaisesRegex(ValueError,'bank/list/extraction/MAD'):ip.source_statistics(a)
                st.write_json(path,r)

    def test_domain_is_in_run_identity_and_cross_domain_state_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root/'parent');p=Parent(SimpleNamespace(domain='illumination',parent_run=root/'parent',parent_cache=root/'cache'));provider(root/'background',p)
            a=SimpleNamespace(domain='illumination',parent_run=root/'parent',parent_cache=root/'cache',raw_run=None,raw_cache=None,data=root/'data',source_stats_run=root/'background',run_root=root/'run')
            ip.freeze(a);self.assertEqual(st.sealed_json(a.run_root/'inputs.json')['config']['domain'],'illumination')
            a.domain='view'
            with self.assertRaises(ValueError):ip.identity(a)
            with self.assertRaises(ValueError):ip.freeze(a)

    def test_new_domain_preparation_imports_source_and_replays_only_eight_cached_targets(self):
        import contextlib,io
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root/'parent');p=Parent(SimpleNamespace(domain='illumination',parent_run=root/'parent',parent_cache=root/'cache'));provider(root/'background',p)
            original=(root/'background/normal_statistics.json').read_bytes();source_weights=[]
            for d in ('illumination','view'):
                a=SimpleNamespace(domain=d,parent_run=root/'parent',parent_cache=root/'cache',raw_run=None,raw_cache=None,data=root/'data',source_stats_run=root/'background',run_root=root/d,cache_root=root/(d+'_cache'),supplement_retrieval=False,allow_over_12h=True,_started=time.perf_counter())
                ip.freeze(a);rng=np.random.default_rng(8 if d=='illumination' else 9)
                for kind,i,row in ip.units(a):
                    st.put(ip.normal_path(a,kind,i),dict(raw=rng.random((4,35),dtype=np.float32),positions=np.arange(35,dtype=np.int64)),dict(ip.normal_identity(a,kind,i,row),grid=[5,7],output_shape=[8,8]))
                with patch.object(pp,'normal_features',side_effect=AssertionError('cached distances must not invoke features/GPU')),contextlib.redirect_stdout(io.StringIO()):
                    normal=pp.prepare_normal_stats(a)
                with patch.object(st,'get',side_effect=AssertionError('complete statistics do not require raw distance arrays')):again=pp.prepare_normal_stats(a)
                self.assertEqual(normal,again);self.assertEqual(len(normal['records']),74)
                self.assertEqual(sum(r['kind']=='Target' and r['domain']==d for r in normal['records']),8)
                self.assertEqual(len(st.sealed_json(a.run_root/'source_reuse.json')['provider_statistics_seal']),64)
                w=pp.fit_weights(a);source_weights.append(w['weights'][SOURCE])
            self.assertEqual(source_weights[0],source_weights[1]);self.assertEqual(original,(root/'background/normal_statistics.json').read_bytes())

    def test_top_completion_waits_for_both_domains_and_background(self):
        from DINOv3.MADEqual.aebad_s_target_loo_layer_weight_p0.aggregation import aggregate
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=SimpleNamespace(run_root=root/'out',background_run=root/'background',illumination_run=root/'illumination',view_run=root/'view')
            runs={}
            for d in COUNTS:
                p=root/d;p.mkdir();st.write_json(p/'status.json',{})
                runs[d]=dict(rows=fixed_combinations(metric_rows(d),d),inputs=dict(identity=d,source='bank',source_calibration=[],extraction={},mad='mad'),weights={SOURCE:np.full((4,3),.25).tolist()},weights_table=[],path=str(p))
            with patch('DINOv3.MADEqual.aebad_s_target_loo_layer_weight_p0.aggregation.completed',side_effect=lambda path,d:runs[d]):
                result=aggregate(a);self.assertEqual(result['status'],'crossdomain_validation_complete');self.assertEqual(result['new_evaluation_images'],629);self.assertEqual(result['new_AUPRO_complete'],4)
                (root/'view/status.json').unlink();result=aggregate(a);self.assertEqual(result['status'],'pending');self.assertIn('view',result['pending'])

    def test_combinations_exactly_reuse_scores_maps_and_have_no_extra_jobs(self):
        rows=metric_rows('view');index={(r['branch'],r['scoring']):r for r in rows}
        for sc in ('CH1','CURRENT'):index[S,sc]['image_AUROC']=.9
        result=fixed_combinations(rows,'view');idx={(r['branch'],r['scoring']):r for r in result}
        self.assertEqual(len(result),14);self.assertEqual(len(NEW),2)
        for b,ref in COMBINATIONS.items():
            for sc in ('CH1','CURRENT'):
                self.assertEqual(idx[b,sc]['image_AUROC'],index[S,sc]['image_AUROC'])
                self.assertEqual([idx[b,sc][k] for k in METRICS[2:]],[index[ref,READOUTS[0]][k] for k in METRICS[2:]])
        self.assertEqual(image_scores({'CH1':8.,'CURRENT':9.})[TARGET_COMBO],{'CH1':8.,'CURRENT':9.})
        self.assertEqual(config('view')['combinations'][TARGET_COMBO]['map_reference'],TARGET)

    def test_partial_new_domain_replay_only_computes_missing_images(self):
        import contextlib,io
        from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.calibration import response
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=SimpleNamespace(domain='view',run_root=root/'run',cache_root=root/'cache')
            a.run_root.mkdir();st.sealed_json(a.run_root/'inputs.json',dict(identity='view-run',config=dict(domain='view')))
            st.sealed_json(a.run_root/'weights.json',dict(identity='view-run',domain='view',weights={b:np.full((4,3),.25).tolist() for b in NEW}))
            rows=[dict(domain='view',sample_id=f'test_{i}') for i in range(2)];z=np.arange(140,dtype=np.float32).reshape(4,35)/70;pos=np.arange(35,dtype=np.int64)
            image={k:dict(medians=[0.]*4,scales=[1.]*4) for k in ('peak','local')};score=response(z,pos,(5,7),(8,8),image)[1];calls=[]
            def baseline(row):
                calls.append(row['sample_id'])
                return {U:z,S:z, 'positions':pos},dict(seal=row['sample_id'],metadata=dict(grid=[5,7],output_shape=[8,8],scores={U:score,S:score}))
            a._parent=SimpleNamespace(domain='view',samples=rows,mad=dict(image=image),seals={r['sample_id']:r['sample_id'] for r in rows},prediction=baseline)
            with patch.object(ip,'count',return_value=2),patch.object(pp,'guard',side_effect=RuntimeError('fixture interruption')),contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(RuntimeError):pp.predict(a)
            first=(ip.prediction_path(a,rows[0])/'manifest.json').read_bytes()
            with patch.object(ip,'count',return_value=2),patch.object(pp,'guard'),contextlib.redirect_stdout(io.StringIO()):
                pp.predict(a);pp.verified(a)
            self.assertEqual(calls,['test_0','test_1']);self.assertEqual(first,(ip.prediction_path(a,rows[0])/'manifest.json').read_bytes())
            saved=st.sealed_json(ip.prediction_path(a,rows[0])/'manifest.json')
            self.assertEqual(saved['metadata']['combination_image_scores'],image_scores(score));self.assertEqual(set(saved['files']),set(NEW))

    def test_all_required_comparisons_and_missing_strong_reference_fail(self):
        r=fixed_combinations(metric_rows('illumination'),'illumination');diff=comparisons(r,'illumination')
        self.assertEqual(len(diff),60)
        combined=[x for x in diff if x['branch']==TARGET_COMBO and x['metric'].startswith('image_')]
        self.assertTrue(all(x['delta_pp']==0 for x in combined))
        with self.assertRaises(ValueError):fixed_combinations([x for x in metric_rows('illumination') if x['branch']!=COMBINED],'illumination')
        with self.assertRaises(ValueError):comparisons([x for x in r if x['branch']!=SOURCE_COMBO],'illumination')

    def test_macros_use_equal_domain_metrics_not_image_count_pooling(self):
        runs={d:dict(rows=fixed_combinations(metric_rows(d,v),d)) for d,v in [('background',.3),('illumination',.2),('view',.8)]}
        two=macro(runs,('illumination','view'),'two');three=macro(runs,('background','illumination','view'),'three')
        self.assertAlmostEqual(two[0]['image_AUROC'],.5);self.assertAlmostEqual(three[0]['image_AUROC'],1.3/3)
        self.assertNotAlmostEqual(two[0]['image_AUROC'],(.2*265+.8*364)/629)

    def test_cpu_entry_and_parent_aggregation_import_no_torch(self):
        code='import sys; from DINOv3.MADEqual.aebad_s_target_loo_layer_weight_p0 import parent,aggregation,reporting_domain; assert "torch" not in sys.modules'
        r=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr)

    def test_background_formula_regression_real_cache_when_available(self):
        path=ip.ROOT/'server_results/aebad_target_loo_weight_seed42_20261029_182735'
        if not path.exists():self.skipTest('downloaded real background review is local-only')
        from DINOv3.MADEqual.aebad_s_target_loo_layer_weight_p0.aggregation import completed
        r=completed(path,'background');self.assertEqual(len(r['rows']),14)
        self.assertTrue(all(x['result_origin']=='historical_background_read_only' for x in r['rows']))
        self.assertEqual(next(x for x in r['rows'] if x['branch']==TARGET_COMBO)['proposal_role'],'background_development_derived_combination')
