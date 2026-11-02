"""CPU-only numerical, provenance, resume and evaluation fixtures; no real run."""
import ast
from contextlib import ExitStack
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from DINOv3.MADEqual.aebad_s_shared_residual_translation_p0 import (
    protocol as cfg,numerics as nm,inputs as ip,prediction as pred,runtime as rt,
    evaluation as ev,reporting as report,runner,smoke)
from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.calibration import z_values,response
st=ip.st


def problem(seed=0,k=9,n=17,dim=12):
    rng=np.random.default_rng(seed);q=rng.normal(size=(n,dim)).astype(np.float32)
    banks=[rng.normal(size=(11+s,dim)).astype(np.float32) for s in range(k)]
    dist=[];ids=[]
    for bank in banks:
        d=nm.norm(q[:,None]-bank[None]);i=d.argmin(1);ids.append(i);dist.append(d[np.arange(n),i]/1280)
    return q,banks,np.asarray(ids,np.int64),np.asarray(dist,np.float32)


def direct64(q,banks,ids,dist):
    r=np.stack([q.astype(float)-b[ii].astype(float) for b,ii in zip(banks,ids)])
    win=np.linalg.norm(r,axis=2).argmin(0);u=r[win,np.arange(len(q))];du=u.mean(0);ds=r.mean(1)
    return {cfg.REFERENCE:np.linalg.norm(r,axis=2).min(0)/1280,cfg.A:np.linalg.norm(u-du,axis=1)/1280,cfg.B:np.linalg.norm(r-du,axis=2).min(0)/1280,
            cfg.C:np.linalg.norm(r-ds[:,None],axis=2).min(0)/1280}


class FormulaTests(unittest.TestCase):
    def test_fixed_scope(self):
        p=cfg.job_plan();self.assertEqual((p['images'],p['localization_jobs'],p['AUPRO_jobs']),(902,12,12))
        self.assertEqual(cfg.CONFIG['primary'],cfg.C);self.assertEqual(cfg.CONFIG['support_seed'],42)

    def test_zero_correction_recovers_union(self):
        q,b,i,d=problem();actual,_,_=nm.translate(q,b,i,d,delta_override=(np.zeros(q.shape[1]),np.zeros((9,q.shape[1]))))
        for x in actual.values():np.testing.assert_allclose(x,d.min(0),atol=1e-7,rtol=1e-6)

    def test_zero_arrays_match_actual_zero_formula(self):
        q,b,i,d=problem();_,small,_=nm.translate(q,b,i,d)
        zero,_,_=nm.translate(q,b,i,d,delta_override=(np.zeros(q.shape[1]),np.zeros((9,q.shape[1]))))
        np.testing.assert_array_equal(small['zero_fixed'],zero[cfg.A])
        for key in (cfg.B,cfg.C):np.testing.assert_array_equal(small['zero_reselect'],zero[key])

    def test_direct_winner_replaces_old_MM_for_A_and_shared_mean(self):
        q=np.array([[0.],[10.],[20.]],np.float32)
        banks=[np.array([[0.]],np.float32),np.array([[1.]],np.float32)]
        ids=np.zeros((2,3),np.int64);old=np.array([[0.,0.,0.],[1.,1.,1.]],np.float32)
        result,small,diag=nm.translate(q,banks,ids,old)
        np.testing.assert_array_equal(small['union_source'],[0,1,1])
        np.testing.assert_array_equal(small['legacy_union_source'],[0,0,0])
        np.testing.assert_allclose(small['delta_U'],[28/3],atol=1e-6)
        np.testing.assert_allclose(result[cfg.REFERENCE],np.array([0.,9.,19.])/1280,atol=1e-7)
        np.testing.assert_allclose(result[cfg.A],np.abs(np.array([0.,9.,19.])-28/3)/1280,atol=1e-7)
        self.assertEqual(diag['legacy_source_switch_count'],2)
        zero,_,_=nm.translate(q,banks,ids,old,delta_override=(np.zeros(1),np.zeros((2,1))))
        for value in zero.values():np.testing.assert_array_equal(value,result[cfg.REFERENCE])

    def test_direct_tie_prefers_source_then_support_order(self):
        q=np.zeros((3,1),np.float32);banks=[np.array([[-1.]],np.float32),np.array([[1.]],np.float32)]
        result,saved,diag=nm.translate(q,banks,np.zeros((2,3),np.int64),np.array([[2.]*3,[1.]*3],np.float32))
        np.testing.assert_array_equal(saved['union_source'],[0,0,0]);np.testing.assert_array_equal(saved['delta_U'],[1.])
        self.assertEqual(diag['legacy_source_switch_count'],3)

    def test_winner_comparison_precedes_final_division(self):
        bits=np.array([1.5],np.float32).view(np.uint32)[0]
        values=np.arange(bits,bits+1000,dtype=np.uint32).view(np.float32)
        scaled=values/np.float32(1280);j=int(np.flatnonzero(scaled[:-1]==scaled[1:])[0])
        q=np.zeros((1,1),np.float32);banks=[np.array([[values[j+1]]],np.float32),np.array([[values[j]]],np.float32)]
        _,small,_=nm.translate(q,banks,np.zeros((2,1),np.int64),np.zeros((2,1),np.float32))
        self.assertEqual(int(small['union_source'][0]),1)

    def test_validator_cannot_override_to_a_distant_source(self):
        q=np.zeros((3,1),np.float32);banks=[np.array([[1.]],np.float32),np.array([[10.]],np.float32)]
        with self.assertRaisesRegex(ValueError,'not a direct-norm minimum'):
            nm.translate(q,banks,np.zeros((2,3),np.int64),np.zeros((2,3),np.float32),union_source_override=np.ones(3,np.int8))

    def test_against_small_fp64_reference(self):
        for seed in range(5):
            q,b,i,d=problem(seed);actual,_,_=nm.translate(q,b,i,d)
            for k,v in direct64(q,b,i,d).items():np.testing.assert_allclose(actual[k],v,atol=1e-7,rtol=1e-6)

    def test_same_candidates_B_le_A(self):
        q,b,i,d=problem();r,_,_=nm.translate(q,b,i,d);self.assertTrue(np.all(r[cfg.B]<=r[cfg.A]+1e-7))

    def test_C_has_no_assumed_order(self):
        greater=less=False
        for seed in range(20):
            q,b,i,d=problem(seed);r,_,_=nm.translate(q,b,i,d)
            greater|=bool((r[cfg.C]>r[cfg.B]).any());less|=bool((r[cfg.C]<r[cfg.B]).any())
        self.assertTrue(greater and less)

    def test_single_source_all_equal(self):
        q,b,i,d=problem(k=1);r,_,_=nm.translate(q,b,i,d)
        np.testing.assert_array_equal(r[cfg.A],r[cfg.B]);np.testing.assert_array_equal(r[cfg.A],r[cfg.C])

    def test_original_units_and_one_division(self):
        q=np.array([[2.],[4.]],np.float32);b=[np.zeros((1,1),np.float32)]
        r,s,_=nm.translate(q,b,np.zeros((1,2),np.int64),q.T/1280)
        np.testing.assert_array_equal(s['delta_U'],[3.]);np.testing.assert_array_equal(r[cfg.C],np.full(2,1/1280,np.float32))

    def test_can_increase_distance(self):
        q=np.array([[0.],[10.],[10.]],np.float32);b=[np.zeros((1,1),np.float32)]
        r,_,_=nm.translate(q,b,np.zeros((1,3),np.int64),q.T/1280)
        self.assertGreater(r[cfg.C][0],0)

    def test_chunk_does_not_change_means_or_results(self):
        args=problem();x,s,_=nm.translate(*args,chunk=1);y,t,_=nm.translate(*args,chunk=7)
        for k in x:np.testing.assert_array_equal(x[k],y[k])
        np.testing.assert_array_equal(s['delta_ref'],t['delta_ref'])

    def test_physical_reorder_is_canonical(self):
        q,b,i,d=problem(n=9);perm=np.array([8,0,3,2,1,7,6,5,4],np.int64)
        ix=nm.order(perm,(3,3));r,s,_=nm.translate(q,b,i,d)
        rr,ss,_=nm.translate(q[perm][ix],b,i[:,perm][:,ix],d[:,perm][:,ix])
        for k in r:np.testing.assert_array_equal(r[k],rr[k])
        np.testing.assert_array_equal(s['delta_U'],ss['delta_U'])

    def test_overlap_or_incomplete_grid_rejected(self):
        for p in (np.array([0,1,1,3],np.int64),np.array([0,1,2],np.int64)):
            with self.assertRaises(ValueError):nm.order(p,(2,2))

    def test_zero_denominator_explicit_NA(self):
        r,s,d=nm.translate(np.zeros((4,2),np.float32),[np.zeros((1,2),np.float32)],np.zeros((1,4),np.int64),np.zeros((1,4),np.float32))
        self.assertIsNone(d['delta_U_ratio']);self.assertTrue(d['sources'][0]['zero_denominator'])

    def test_invalid_indices_and_nonfinite(self):
        q,b,i,d=problem();i[0,0]=-1
        with self.assertRaises(ValueError):nm.translate(q,b,i,d)
        q,b,i,d=problem();q[0,0]=np.nan
        with self.assertRaises(ValueError):nm.translate(q,b,i,d)

    def test_zero_scoring_chain_and_weighted_map(self):
        rng=np.random.default_rng(4);raw=rng.uniform(0,.2,(4,16)).astype(np.float32)
        pos=np.roll(np.arange(16,dtype=np.int64),3);mad=dict(medians=[.03]*4,scales=[.1]*4)
        state={s:dict(medians=[.2]*4,scales=[.5]*4) for s in ('peak','local')}
        w=np.array([[.1]*3,[.2]*3,[.3]*3,[.4]*3])
        im,sc=nm.score(raw,pos,[4,4],[7,8],mad,state,w)
        _,original=response(z_values(raw,mad),pos,[4,4],[7,8],state)
        weighted,_=nm.weighted(z_values(raw,mad),pos,[4,4],[7,8],w)
        self.assertEqual(sc['CH1'],original['CH1']);np.testing.assert_array_equal(im,weighted)

    def test_negative_z_is_retained(self):
        raw=np.zeros((4,9),np.float32);mad=dict(medians=[1]*4,scales=[1]*4)
        state={s:dict(medians=[0]*4,scales=[1]*4) for s in ('peak','local')}
        im,sc=nm.score(raw,np.arange(9,dtype=np.int64),[3,3],[5,5],mad,state,np.full((4,3),.25))
        self.assertLess(im.max(),0);self.assertLess(sc['CH1'],0)

    def test_prediction_formula_has_no_label_input(self):
        tree=ast.parse(Path(nm.__file__).read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='translate')
        text=ast.unparse(node)
        self.assertNotIn('label',text);self.assertNotIn('gt(',text)


class Fixture:
    def __init__(self,root):
        self.root=Path(root);self.a=runner.normalize(runner.parser().parse_args([
            '--run-root',str(self.root/'run'),'--cache-root',str(self.root/'cache'),
            '--evaluation-scratch',str(self.root/'scratch'),'--device','cpu']))
        a=self.a;st.sealed_json(a.run_root/'inputs.json',dict(identity='tiny_fixture',prediction_backend=ip.prediction_backend(a)))
        self.patch=dict(medians=[.0]*4,scales=[.1]*4)
        self.state={k:dict(medians=[0]*4,scales=[1]*4) for k in ('peak','local')}
        self.rng=np.random.default_rng(12);self.rows={};self.q={};self.bb={};self.old={};self.records={};self.maps={}
        self.weights={d:np.full((4,3),.25).tolist() for d in cfg.DOMAINS}
        self.refs=[dict(path='fixture_target',seal='target')]*8
        for d in cfg.DOMAINS:
            self.rows[d]=[]
            self.bb[d]={l:[self.rng.normal(size=(5,1280)).astype(np.float32) for _ in range(9)] for l in cfg.LAYERS}
            for j in range(2):
                row=dict(domain=d,sample_id='test_'+str(j),relative_path=d+'/'+str(j)+'.png',image_sha256='image'+str(j),image_height=16,image_width=16,label=j)
                self.rows[d].append(row);qq={l:self.rng.normal(size=(9,1280)).astype(np.float32) for l in cfg.LAYERS};self.q[d,row['sample_id']]=qq
                nn=self.nearest(d,qq);raw=np.stack([nn[f'distance{l}'].min(0) for l in cfg.LAYERS]);z=z_values(raw,self.patch)
                self.old[d,row['sample_id']]=dict(UNION_K8=z,SEGMENT_K8=z-.01,positions=np.arange(9,dtype=np.int64))
                hashes={};scores={}
                for b in (cfg.U,cfg.S,cfg.SOURCE):
                    im,sc=(nm.weighted(z,np.arange(9,dtype=np.int64),[3,3],[4,4],self.weights[d]) if b==cfg.SOURCE else response(z if b==cfg.U else z-.01,np.arange(9,dtype=np.int64),[3,3],[4,4],self.state))
                    scores[b]=sc;hashes[b]=pred.shared.array_sha(im)
                self.records[d,row['sample_id']]=dict(scores=scores,map_hashes=hashes)
                self.maps[d]=(self.maps.get(d) or {})|{(row['sample_id'],cfg.U):dict(mask_uint8_sha256=pred.shared.array_sha(self.gt(row)))}
        parent=SimpleNamespace(inputs=dict(seal='parent-seal',mad=dict(patch=self.patch,image=self.state),source=dict(seal='Source'),extraction={}),common=dict(seal='common-seal'),
            identity='parent',weights=self.weights,maps=self.maps,samples=lambda d:self.rows[d],supports=lambda d:[],record=lambda d,r:self.records[d,r['sample_id']])
        self.parent=parent;a._joint_parent=parent
        parent.metrics={d:[dict(domain=d,branch=b,scoring=sc,**{k:.6 for k in cfg.METRICS}) for b in (cfg.U,cfg.S,'SEGMENT_SCORE_UNION_MAP',cfg.SOURCE,'SEGMENT_SCORE_SOURCE_WEIGHT_MAP')
                           for sc in (('PEAK_LOCAL_WEIGHT','WEIGHTED_MAP_MAX') if b==cfg.SOURCE else ('CH1','CURRENT'))] for d in cfg.DOMAINS}

    @staticmethod
    def gt(row):
        y=np.zeros((4,4),np.uint8)
        if row['label']:y[1:3,1:3]=1
        return y

    def nearest(self,d,q):
        x=dict(positions=np.arange(9,dtype=np.int64))
        for l in cfg.LAYERS:
            ids=[];dist=[]
            for bank in self.bb[d][l]:
                dd=nm.norm(q[l][:,None]-bank[None]);ii=dd.argmin(1);ids.append(ii);dist.append(dd[np.arange(9),ii]/1280)
            x[f'index{l}']=np.asarray(ids,np.int64);x[f'distance{l}']=np.asarray(dist,np.float32)
        return x

    def context(self):
        stack=ExitStack();f=self
        class Refs:
            def __init__(self,a,d):self.d=d;self.targets=f.refs;self.search=SimpleNamespace(stats=dict(searches=0))
            def nearest(self,q,geometry,feature,row):return f.nearest(self.d,q),dict(seal='NN'),True
            def banks(self,l):return f.bb[self.d][l],list(range(9))
        def feature(a,row,persist=False):return f.q[row['domain'],row['sample_id']],dict(grid=[3,3],positions=np.arange(9,dtype=np.int64)),dict(kind='original_full_forward',extraction={},image=ip.image(row))
        def baseline(a,d,row):return f.old[d,row['sample_id']],dict(metadata=dict(grid=[3,3])),dict(path='original',seal='base')
        class Data:
            def __init__(self,path):
                if not (f.a.run_root/'predictions_frozen.json').exists():raise AssertionError('GT accessed before freeze')
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def gt(self,row):return f.gt(row)
        stack.enter_context(patch.object(rt,'References',Refs));stack.enter_context(patch.object(rt,'features',feature))
        stack.enter_context(patch.object(ip,'baseline',baseline));stack.enter_context(patch.object(ev,'GTData',Data))
        stack.enter_context(patch.object(ip,'feature_ref',return_value=None));stack.enter_context(patch.object(ip,'source_ref',return_value=dict(seal='Source')))
        # Fixture Target manifests stand in for immutable original support assets.
        for ref in self.refs:
            path=f.root/'target';r=st.put(path,dict(q7=np.zeros((1,1280),np.float32)),dict(fixture=True));ref.update(path=str(path),seal=r['seal'])
        return stack


class PipelineTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.f=Fixture(self.tmp.name);self.stack=self.f.context();self.a=self.f.a
    def tearDown(self):self.stack.close();self.tmp.cleanup()

    def test_predict_reuses_all_without_loading_references(self):
        first=pred.predict(self.a);self.assertEqual(first['new_images'],6)
        with patch.object(rt,'References',side_effect=AssertionError('no second retrieval')):
            second=pred.predict(self.a)
        self.assertEqual(second['reused_images'],6)

    def test_freeze_contains_domain_key_no_overwrite(self):
        pred.predict(self.a);r=pred.global_freeze(self.a)
        self.assertEqual(len({(v['domain'],v['sample_id']) for v in r['records']}),6)

    def test_GT_refused_before_all_domains_frozen(self):
        with self.assertRaises(FileNotFoundError):ev.evaluate(self.a)

    def test_CPU_twelve_jobs_three_scores_share_pixels_and_resume(self):
        pred.predict(self.a);r=ev.evaluate(self.a);self.assertEqual(r['jobs'],12)
        for d in cfg.DOMAINS:
            for b in cfg.MAPS:
                r=ip.rows(ev.destination(self.a,d,b)/'metrics.csv');self.assertEqual(len(r),3)
                self.assertEqual(len({x['pixel_AP'] for x in r}),1)
                inherited=next(x for x in r if x['scoring']=='SEGMENT_CH1')
                self.assertEqual(inherited['image_metrics_inherited'],'True')
        with patch.object(ev,'GTData',side_effect=AssertionError('completed pixels must resume')):
            self.assertEqual(ev.evaluate(self.a)['jobs'],12)

    def test_intermediate_metric_modification_rejected(self):
        pred.predict(self.a);ev.evaluate(self.a);out=ev.destination(self.a,cfg.DOMAINS[0],cfg.A)
        (out/'status.json').unlink();r=st.read_json(out/'exact_pixels.json');r['result']['AUROC']=1.5;st.write_json(out/'exact_pixels.json',r)
        with self.assertRaisesRegex(ValueError,'modified/incompatible'):ev.evaluate(self.a)

    def test_changed_prediction_identity_rejected(self):
        pred.predict(self.a);p=ip.record_path(self.a,cfg.DOMAINS[0],self.f.rows[cfg.DOMAINS[0]][0])
        r=st.read_json(p);r['scores'][cfg.A]['CH1']+=1;st.write_json(p,r)
        with self.assertRaisesRegex(ValueError,'modified record'):pred.verify(self.a,cfg.DOMAINS[0],self.f.rows[cfg.DOMAINS[0]][0])

    def test_pending_is_not_zero_or_completion(self):
        pred.predict(self.a);ev.evaluate(self.a);r=report.summarize(self.a)
        self.assertEqual(r['status'],'pending');self.assertEqual(len(r['pending']),12)
        ms=ip.rows(self.a.run_root/'metrics.csv');new=[v for v in ms if v['branch'] in cfg.MAPS]
        self.assertTrue(all(v['AUPRO']=='' for v in new))

    def test_twelve_AUPRO_commits_finish_and_do_not_touch_GPU_on_resume(self):
        pred.predict(self.a);ev.evaluate(self.a)
        # Synthetic sealed result fixtures exercise orchestration only, never
        # claim an actual ADEval computation or a scientific result.
        for d in cfg.DOMAINS:
            for b in cfg.MAPS:
                expected,_=ev.au_expected(self.a,d,b)
                r=dict(expected,AUPRO=.55,verified_images=2,backend='original_ADEval_1.1.0_CUDA',seconds=1.,peak_cuda_allocated_bytes=0,peak_cuda_reserved_bytes=0)
                r['seal']=st.digest(r);st.write_json(self.a.run_root/d/'aupro'/(b+'.json'),r)
        with patch.object(ev,'GTData',side_effect=AssertionError('no second GT load')):
            self.assertEqual(ev.aupro(self.a)['jobs'],12)
        r=report.summarize(self.a);self.assertEqual(r['status'],'all_five_metrics_complete');self.assertEqual(r['pending'],[])
        self.assertEqual(r['new_AUPRO_complete'],12)
        rows=ip.rows(self.a.run_root/'metrics.csv')
        self.assertEqual(len([r for r in rows if r['domain']=='macro_3domains' and r['branch'] in cfg.MAPS]),12)
        self.assertEqual(r['candidate_AUPRO_complete'],9);self.assertEqual(r['reference_AUPRO_complete'],3)
        self.assertEqual(len(list(self.a.run_root.glob('*/aupro/*.json'))),12)
        self.assertNotIn(cfg.STRONG,cfg.MAPS)
        index={(x['domain'],x['branch'],x['scoring']):x for x in rows}
        for d in (*cfg.DOMAINS,'macro_3domains'):
            first=index[d,cfg.STRONG,'CH1'];second=index[d,cfg.REFERENCE,'SEGMENT_CH1']
            for k in cfg.METRICS:self.assertEqual(float(first[k]),float(second[k]))
        changes=ip.rows(self.a.run_root/'metric_comparisons.csv')
        primary=[x for x in changes if x['comparison_role']=='primary_matched_reference']
        self.assertEqual({x['reference'] for x in primary},{cfg.REFERENCE,cfg.STRONG})
        path_changes=ip.rows(self.a.run_root/'numerical_path_metric_changes.csv')
        self.assertEqual(len(path_changes),3*5*4)
        self.assertTrue(all(x['comparison_role']=='numerical_path_change' for x in path_changes))
        self.assertTrue(all(x['numeric_path']=='legacy_MM' for x in report.historical(self.a)))
        self.assertTrue(all(x['numeric_path']=='legacy_MM' for x in rows if x['domain']=='macro_3domains' and x['branch']==cfg.U))
        switches=ip.rows(self.a.run_root/'source_selection_switches.csv')
        self.assertEqual(len(switches),6*4)
        self.assertEqual({r['domain'] for r in switches},set(cfg.DOMAINS))
        # A stale large first-run estimate must not charge already committed work.
        st.write_json(self.a.run_root/'time_estimate.json',dict(status='full_image_smoke_estimate',identity='tiny_fixture',
            prediction_backend=ip.prediction_backend(self.a),samples=[dict(domain=d,seconds=100000.) for d in cfg.DOMAINS],stages=dict(CPU_exact=100000.,AUPRO=100000.)))
        smoke.launch_check(self.a)

    def test_CPU_metric_benchmark_keeps_backend_serializable(self):
        pred.predict(self.a);self.a.benchmark_metrics=True
        expected_backend=ip.prediction_backend(self.a)
        r=smoke.run(self.a)
        self.assertEqual(r['prediction_backend'],expected_backend)
        self.assertEqual(r['job_plan']['localization_jobs'],12)
        self.assertEqual(st.read_json(self.a.run_root/'time_estimate.json')['prediction_backend'],expected_backend)
        self.assertIsNotNone(r['stages']['CPU_exact'])
        self.assertIsNone(r['stages']['AUPRO'])

    def test_pending_reference_AUPRO_jobs_are_in_whole_run_budget(self):
        # Cost-only fixtures: candidate jobs exist, three REFERENCE jobs pending.
        for d in cfg.DOMAINS:
            for b in cfg.CANDIDATES:st.write_json(self.a.run_root/d/'aupro'/(b+'.json'),dict(cost_fixture=True))
        st.write_json(self.a.run_root/'time_estimate.json',dict(status='full_image_smoke_estimate',identity='tiny_fixture',
            prediction_backend=ip.prediction_backend(self.a),samples=[dict(domain=d,seconds=0.) for d in cfg.DOMAINS],
            stages=dict(CPU_exact=0.,AUPRO=200000.)))
        self.a.stage='all'
        with patch.object(ev,'completed',return_value={'completed':True}) as completed:
            with self.assertRaisesRegex(RuntimeError,'exceeds12h'):smoke.launch_check(self.a)
        self.assertEqual(completed.call_count,12)

    def test_parent_mismatch_preserves_error_and_stops(self):
        d=cfg.DOMAINS[0];row=self.f.rows[d][0];self.f.old[d,row['sample_id']][cfg.U][0,0]+=1.
        with self.assertRaisesRegex(ValueError,'no relaxation'):pred.one(self.a,d,row,rt.References(self.a,d))
        self.assertTrue((self.a.run_root/'failed_baseline_checks.csv').exists())
        self.assertFalse(ip.record_path(self.a,d,row).exists())

    def apply_MM_offset_fixture(self,bias):
        # Controlled cross-path difference; only this tiny fixture is changed.
        d=cfg.DOMAINS[0];row=self.f.rows[d][0];original=self.f.nearest
        def shifted(domain,q):
            result=original(domain,q)
            result['distance7']=result['distance7']+np.float32(bias)
            return result
        self.f.nearest=shifted
        x=self.f.old[d,row['sample_id']];x[cfg.U][0]+=np.float32(bias/self.f.patch['scales'][0])
        record=self.f.records[d,row['sample_id']]
        for branch in (cfg.U,cfg.SOURCE):
            im,sc=(nm.weighted(x[cfg.U],x['positions'],[3,3],[4,4],self.f.weights[d]) if branch==cfg.SOURCE else
                   response(x[cfg.U],x['positions'],[3,3],[4,4],self.f.state))
            record['scores'][branch]=sc;record['map_hashes'][branch]=pred.shared.array_sha(im)
        return d,row

    def test_cross_path_drift_is_audit_but_true_zero_maps_still_pass(self):
        d,row=self.apply_MM_offset_fixture(3e-7)
        self.assertTrue(pred.one(self.a,d,row,rt.References(self.a,d)))
        rec,_,_=pred.verify(self.a,d,row)
        self.assertGreater(rec['arithmetic_audit'][0]['positions_over_original_raw_tolerance'],0)
        checks=[r for r in rec['checks'] if r['check']=='zero_direct_vector_score_map']
        self.assertEqual(len(checks),2);self.assertTrue(all(r['verified'] for r in checks))
        self.assertTrue(all(r['atol']==1e-5 for r in checks))

    def test_large_legacy_drift_is_reported_not_used_as_matched_reference(self):
        d,row=self.apply_MM_offset_fixture(3e-5)
        self.assertTrue(pred.one(self.a,d,row,rt.References(self.a,d)))
        rec,_,_=pred.verify(self.a,d,row)
        self.assertTrue(all(x['verified'] for x in rec['checks']))
        drift=st.read_json(self.a.run_root/d/'reference_path_differences'/(row['sample_id']+'.json'))
        self.assertGreater(drift['map_max_abs_difference'],1e-5)
        self.assertFalse(drift['historical_compatibility_required'])

    def test_old_9_map_profile_cannot_silently_resume_as_new_reference_profile(self):
        self.a.run_root=self.f.root/'legacy_profile';value=dict(identity='old',prediction_backend=ip.prediction_backend(self.a),config={'version':'shared_residual_translation_v1'})
        st.sealed_json(self.a.run_root/'inputs.json',value);before=(self.a.run_root/'inputs.json').read_bytes()
        with self.assertRaisesRegex(ValueError,'immutable record differs'):ip.freeze(self.a)
        self.assertEqual((self.a.run_root/'inputs.json').read_bytes(),before)

    def test_invalid_zero_against_matched_reference_still_stops(self):
        d=cfg.DOMAINS[0];row=self.f.rows[d][0];original=rt.translate
        def corrupt(*args,**kwargs):
            values,small,diagnostic=original(*args,**kwargs)
            small['zero_fixed']=small['zero_fixed']+np.float32(3e-5)
            return values,small,diagnostic
        with patch.object(rt,'translate',side_effect=corrupt):
            with self.assertRaisesRegex(ValueError,'zero_direct_vector_score_map'):
                pred.one(self.a,d,row,rt.References(self.a,d))
        self.assertFalse(ip.record_path(self.a,d,row).exists())
        zeros=[r for r in ip.rows(self.a.run_root/'failed_baseline_checks.csv') if r['check']=='zero_direct_vector_score_map']
        self.assertTrue(any(r['verified']=='False' for r in zeros))
        self.assertTrue(all(float(r['atol'])==1e-5 and r['reference']==cfg.REFERENCE for r in zeros))

    def test_changed_query_provider_rejected_on_predict_resume(self):
        pred.predict(self.a);row=self.f.rows[cfg.DOMAINS[0]][0]
        with patch.object(ip,'feature_ref',return_value=dict(path='other_checkpoint',seal='other')):
            with self.assertRaisesRegex(ValueError,'provider changed'):pred.verify(self.a,cfg.DOMAINS[0],row,check_inputs=True)

    def test_GT_identity_mismatch_rejected(self):
        pred.predict(self.a);self.f.maps[cfg.DOMAINS[0]][self.f.rows[cfg.DOMAINS[0]][0]['sample_id'],cfg.U]['mask_uint8_sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'GT/resize differs'):ev.evaluate(self.a)

    def test_macro_uses_domain_means_not_pooled_images(self):
        rows=[dict(domain=d,branch=cfg.C,scoring='CH1',**{k:v for k in cfg.METRICS}) for d,v in zip(cfg.DOMAINS,(.1,.3,.8))]
        r=report.macro(rows)[0];self.assertAlmostEqual(r['pixel_AP'],.4)

    def test_only_new_scratch_files_removed_on_failure(self):
        root=self.a.evaluation_scratch/cfg.DOMAINS[0]/cfg.A;root.mkdir(parents=True);other=root/'keep.bin';other.write_bytes(b'parent')
        with self.assertRaises(RuntimeError):
            with ev.scratch(self.a,cfg.DOMAINS[0],cfg.A,16):raise RuntimeError('fixture')
        self.assertEqual(other.read_bytes(),b'parent');self.assertEqual(list(root.iterdir()),[other])

    def test_pack_excludes_large_temporary_files(self):
        (self.a.run_root/'pixel_scratch').mkdir();(self.a.run_root/'pixel_scratch'/'huge.bin').write_bytes(b'bad')
        (self.a.run_root/'other.bin').write_bytes(b'bad');r=runner.pack(self.a)
        with tarfile.open(r['archive']) as f:self.assertTrue(all(not x.name.endswith('.bin') for x in f))

    def test_scratch_parent_overlap_rejected(self):
        args=runner.parser().parse_args(['--evaluation-scratch',str(ip.ROOT/'server_results/aebad_target_loo_two_K8_20261030_013433/pixels')])
        with self.assertRaises(ValueError):runner.normalize(args)

    def test_inherited_Source_scratch_overlap_rejected(self):
        parent=self.f.root/'parent';source=self.f.root/'old_source'
        st.sealed_json(parent/'inputs.json',dict(source=dict(path=str(source))))
        args=runner.parser().parse_args(['--parent-run',str(parent),'--evaluation-scratch',str(source/'pixels')])
        with self.assertRaises(ValueError):runner.normalize(args)

    def test_reliable_indices_require_actual_dependency_match(self):
        row=self.f.rows[cfg.DOMAINS[0]][0];path=ip.nn_path(self.a,cfg.DOMAINS[0],row)
        st.put(path,dict(positions=np.arange(9,dtype=np.int64)),dict(contract=dict(extraction='other'),grid=[3,3]))
        with self.assertRaisesRegex(ValueError,'same query'):ip.nn_ref(self.a,cfg.DOMAINS[0],row,dict(extraction='original'))


class BackendBindingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.f=Fixture(self.tmp.name);self.a=self.f.a
    def tearDown(self):self.tmp.cleanup()

    def real_freeze(self):
        # Replace only the test fixture's synthetic inputs with the actual freeze
        # function. No production or historical artifact is changed.
        (self.a.run_root/'inputs.json').unlink();return ip.freeze(self.a)

    def test_backend_identity_differs_without_importing_torch(self):
        before=set(sys.modules);cpu=self.real_freeze()
        self.a.run_root=self.f.root/'cuda_run';self.a.device='cuda:0';cuda=ip.freeze(self.a)
        self.assertNotEqual(cpu['identity'],cuda['identity'])
        self.assertNotEqual(cpu['prediction_backend'],cuda['prediction_backend'])
        self.assertEqual({k:v for k,v in cpu.items() if k not in ('prediction_backend','identity','seal')},
                         {k:v for k,v in cuda.items() if k not in ('prediction_backend','identity','seal')})
        self.assertNotIn('torch',set(sys.modules)-before)

    def test_original_retrieval_input_contract_is_not_correction_backend(self):
        row=self.f.rows['background'][0];feature=dict(path='frozen_query',seal='same_query')
        cpu=ip.contract(self.a,'background',row,feature,self.f.refs)
        self.a.device='cuda:0';cuda=ip.contract(self.a,'background',row,feature,self.f.refs)
        self.assertEqual(cpu,cuda)

    def test_changed_backend_rejects_existing_run_without_rewriting(self):
        self.real_freeze();path=self.a.run_root/'inputs.json';old=path.read_bytes();self.a.device='cuda:0'
        with self.assertRaisesRegex(ValueError,'prediction backend differs'):ip.freeze(self.a)
        self.assertEqual(path.read_bytes(),old)

    def test_changed_backend_never_reuses_completed_smoke(self):
        st.write_json(self.a.run_root/'time_estimate.json',dict(status='full_image_smoke_estimate',identity='tiny_fixture',
            prediction_backend=ip.prediction_backend(self.a),samples=[]))
        self.a.device='cuda:0'
        with patch.object(rt,'References',side_effect=AssertionError('must reject before CUDA')),patch.object(pred,'verify',side_effect=AssertionError('must reject before smoke reuse')):
            with self.assertRaisesRegex(ValueError,'prediction backend differs'):smoke.run(self.a)

    def test_same_backend_reuses_verified_smoke(self):
        st.write_json(self.a.run_root/'time_estimate.json',dict(status='full_image_smoke_estimate',identity='tiny_fixture',
            prediction_backend=ip.prediction_backend(self.a),samples=[]))
        with patch.object(rt,'References',side_effect=AssertionError('no fresh computation')):
            self.assertTrue(smoke.run(self.a)['reused_smoke'])

    def test_unbound_or_other_backend_smoke_is_rejected(self):
        for backend in (None,ip.prediction_backend(SimpleNamespace(device='cuda:0'))):
            r=dict(status='full_image_smoke_estimate',identity='tiny_fixture',samples=[])
            if backend is not None:r['prediction_backend']=backend
            st.write_json(self.a.run_root/'time_estimate.json',r)
            with self.assertRaisesRegex(ValueError,'smoke prediction backend'):smoke.run(self.a)

    def test_partial_smoke_unit_is_backend_bound(self):
        st.sealed_json(self.a.run_root/'smoke_units/background.json',dict(identity='tiny_fixture',
            prediction_backend=ip.prediction_backend(SimpleNamespace(device='cuda:0'))))
        with patch.object(rt,'References',side_effect=AssertionError('no CUDA')):
            with self.assertRaisesRegex(ValueError,'smoke inputs/backend differ'):smoke.run(self.a)

    def test_partial_prediction_cannot_mix_backends(self):
        with self.f.context():
            row=self.f.rows['background'][0];pred.one(self.a,'background',row,rt.References(self.a,'background'))
            original=ip.record_path(self.a,'background',row).read_bytes();self.a.device='cuda:0'
            with patch.object(rt,'References',side_effect=AssertionError('no CUDA or new retrieval')):
                with self.assertRaisesRegex(ValueError,'prediction backend differs'):pred.predict(self.a)
            self.assertEqual(ip.record_path(self.a,'background',row).read_bytes(),original)
            self.assertFalse(ip.record_path(self.a,'background',self.f.rows['background'][1]).exists())

    def test_unbound_legacy_run_rejected_not_backfilled(self):
        path=self.a.run_root/'inputs.json';path.unlink();st.sealed_json(path,dict(identity='legacy'))
        old=path.read_bytes()
        with self.assertRaisesRegex(ValueError,'lacks prediction backend binding'):ip.freeze(self.a)
        self.assertEqual(path.read_bytes(),old)

    def test_time_budget_override_does_not_bypass_backend_binding(self):
        self.a.allow_over_12h=True;self.a.device='cuda:0'
        with self.assertRaisesRegex(ValueError,'prediction backend differs'):smoke.launch_check(self.a)

    def test_postprocessing_device_does_not_change_producer_backend(self):
        self.a.device='cuda:0';cuda=self.real_freeze()
        self.a.device='cpu';self.a.stage='evaluate';cpu_evaluation=ip.freeze(self.a)
        self.assertEqual(cuda,cpu_evaluation)
        self.assertEqual(ip.signature(self.a,'background')['prediction_backend'],cuda['prediction_backend'])


class MetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=ip.ROOT/'server_results/aebad_target_loo_two_K8_20261030_013433'
        if not path.exists():raise unittest.SkipTest('real parent metadata absent in release checkout')
        cls.parent=ip.Parent(path)

    def test_real_population_and_supports(self):
        self.assertEqual(sum(len(self.parent.samples(d)) for d in cfg.DOMAINS),902)
        self.assertTrue(all(len(self.parent.supports(d))==8 for d in cfg.DOMAINS))

    def test_no_seed43_statistics_or_features_opened(self):
        actual=st.sealed_json;opened=[]
        def read(path,*args,**kwargs):opened.append(str(path).replace('\\','/'));return actual(path,*args,**kwargs)
        with patch.object(st,'sealed_json',side_effect=read):ip.Parent(self.parent.root)
        self.assertFalse(any('/seed43/' in p for p in opened))

    def test_default_preflight_no_torch_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            code='import sys; from DINOv3.MADEqual.aebad_s_shared_residual_translation_p0.runner import main; main(["preflight","--run-root",'+repr(str(Path(tmp)/'run'))+']); assert "torch" not in sys.modules'
            r=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)
            value=st.read_json(Path(tmp)/'run/preflight.json');self.assertFalse(value['torch_imported'])


class TensorCPUTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('torch'),'already installed Torch CPU validation only')
    def test_stock_tensor_CPU_matches_numpy(self):
        r=subprocess.run([sys.executable,'-B','-m','DINOv3.MADEqual.aebad_s_shared_residual_translation_p0.verify_stock_cpu'],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)
        value=json.loads(r.stdout);self.assertFalse(value['cuda_initialized']);self.assertEqual(len(value['cases']),3)


if __name__=='__main__':unittest.main()
