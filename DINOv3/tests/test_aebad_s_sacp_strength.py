"""Torch-free CPU tests; synthetic cached arrays/GT, no actual parent replay."""
import copy,io,json,os,sys,tempfile,unittest,subprocess
from pathlib import Path
from unittest.mock import patch
from types import FunctionType
import numpy as np
from PIL import Image
from DINOv3.MADEqual.aebad_s_sacp_strength_p0 import protocol as p,method,inputs as ip,prediction as pred,runner
from DINOv3.MADEqual.aebad_s_sacp_strength_p0.boundary import cpu_only
from DINOv3.MADEqual.aebad_s_sacp_p0.numerics import correct,matched_uniform

st=ip.st


def setUpModule():(ip.ROOT/'.tmp').mkdir(exist_ok=True)


class Math(unittest.TestCase):
    def test_fixed_plan_only_three_new_branches(self):
        q=p.job_plan();self.assertEqual(q['prediction_images'],926);self.assertEqual(len(q['CPU_pixel_jobs']),9);self.assertEqual(len(q['AUPRO_jobs']),9)
        self.assertEqual((q['main_metric_rows'],q['complete_metric_rows'],q['new_metric_rows']),(72,80,24))
        self.assertFalse(any('SOURCE_DIR_L1' in r['branch'] for r in q['AUPRO_jobs']))
    def test_zero_eta_preserves_distance_and_inherited_union_z(self):
        u=np.ones((4,3),np.float32);eta=np.zeros_like(u);raw=method.distance(u,eta,1)
        inherited=np.nextafter(np.ones_like(u),np.full_like(u,np.inf));z=method.calibrated(raw,u,inherited,dict(medians=[0]*4,scales=[1]*4))
        np.testing.assert_array_equal(raw,u);np.testing.assert_array_equal(z,inherited)
    def test_eta_one_zero_distance_and_zero_union(self):
        u=np.array([[0,1]],np.float32);result=method.distance(u,np.ones_like(u),1)
        np.testing.assert_array_equal(result,0);self.assertTrue(np.isfinite(result).all())
    def test_fixed_eta_L1_monotone_and_not_linear_doubling(self):
        u=np.ones((4,2),np.float32);eta=np.full_like(u,.8);half=method.distance(u,eta,.5);one=method.distance(u,eta,1)
        self.assertTrue((one<=half).all());self.assertFalse(np.allclose(one,u-2*(u-half)))
        with self.assertRaises(ValueError):method.distance(u,eta,.75)
    def test_invalid_eta_is_not_clipped_or_cleaned(self):
        u=np.ones((4,1),np.float32)
        for x in (-1e-7,1+1e-6,np.nan,np.inf):
            with self.assertRaisesRegex(ValueError,'no cleaning'):method.distance(u,np.full_like(u,x),1)
    def test_axes_and_layers_are_distinct(self):
        u=np.ones((4,3),np.float32);eta=np.zeros((3,4,3),np.float32);eta[0,:,0]=[.1,.2,.3,.4];eta[1,:,1]=[.5,.6,.7,.8];eta[2]=np.nan
        raw,_=method.values(dict(raw_union=u,eta=eta),1)
        np.testing.assert_allclose(raw[p.MAIN][:,0],np.sqrt(1-np.array([.1,.2,.3,.4],np.float32)),atol=1e-7)
        np.testing.assert_array_equal(raw[p.MAIN][:,1],1);np.testing.assert_array_equal(raw[p.TARGET][:,0],1)
    def test_uniform_matches_new_total_not_half_coefficient(self):
        rng=np.random.default_rng(2);u=rng.uniform(.01,2,(4,31)).astype(np.float32);eta=rng.uniform(0,1,(3,4,31)).astype(np.float32)
        one,oa=method.values(dict(raw_union=u,eta=eta),1);half,ha=method.values(dict(raw_union=u,eta=eta),.5)
        for j in range(4):
            self.assertLessEqual(oa[j]['absolute_error'],oa[j]['FP32_sum_roundoff_bound'])
            self.assertNotAlmostEqual(oa[j]['coefficient'],2*ha[j]['coefficient'],places=5)
            self.assertNotEqual(oa[j]['coefficient'],ha[j]['coefficient'])


class Integration(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ip.ROOT/'.tmp');self.root=Path(self.temp.name)
        self.toy={d:[2,1,1] for d in p.DOMAINS};self.pop=patch.dict(p.CONFIG,{'populations':self.toy});self.pop.start();self.addCleanup(self.pop.stop);self.addCleanup(self.temp.cleanup)
        self.a,self.samples=self.fixture()
    def fixture(self):
        from importlib.metadata import version
        from DINOv3.MADEqual.aebad_s_sacp_strength_p0 import evaluation as ev
        from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.analysis import spatial
        from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.fresh_evaluation import mark_small
        from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import image_metrics
        parent=self.root/'parent';cache=self.root/'relocated_predictions';data=self.root/'GT_only';data.mkdir()
        samples=[];supports={};source=[dict(relative_path=f'train/good/fixture/{i}.png',source_index=i,calibration=i%8==0) for i in range(521)];id='synthetic_sacp_parent'
        state=dict(patch=dict(medians=[.001]*4,scales=[.001]*4),image={name:dict(medians=[0]*4,scales=[1]*4) for name in ('peak','local')},scope='original_Source_only_distances')
        mad=st.sealed_json(parent/'input_distance_mad.json',dict(identity=id,**state));protocol={**copy.deepcopy(p.PARENT_CONFIG),'populations':self.toy};st.sealed_json(parent/'protocol.json',protocol)
        assets=st.sealed_json(parent/'assets.json',dict(references=dict(extraction=dict(software={n:version(n) for n in ('numpy','scipy','Pillow')}),source=dict(path='/offline/no_Source_bank')),queries={}))
        all_metrics=[];mechs=[];all_instances=[];manifest_seals=[]
        for d in p.DOMAINS:
            supports[d]=[dict(relative_path=f'test/good/{d}/support{i}.png',label=0,support_index=i) for i in range(1,9)]
            sc=[];maps_rows=[];instances=[];records=[]
            for label in (0,1):
                s=dict(domain=d,sample_id=f'test_{label}',label=label,relative_path=f'test/'+('ablation' if label else 'good')+f'/{d}/image{label}.png',image_height=64,image_width=64,image_sha256='unused_RGB')
                gt=np.zeros((64,64),np.uint8)
                if label:
                    gt[24:40,24:40]=255;gt[1,1]=255;rel=s['relative_path'].replace('test/','ground_truth/',1);dest=data/rel;dest.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(gt).save(dest);s['gt_sha256']=st.sha(dest)
                samples.append(s);rng=st.rng('strength-fixture',d,label);pos=rng.permutation(16).astype(np.int64);u=rng.uniform(.001,.01,(4,16)).astype(np.float32);eta=[];raw={p.U:u,p.S:u*np.float32(.9)}
                for j in range(4):
                    r=rng.normal(size=(16,6)).astype(np.float32);r[0]=[1,0,0,0,0,0];r[1]=0
                    groups=pos%2;ranks=np.array([2,0]);layers=[]
                    for off,b in enumerate((p.HALF_MAIN,p.HALF_TARGET,p.HALF_SOURCE)):
                        basis=np.zeros((2,6,2),np.float32);basis[0,off*2:off*2+2]=np.eye(2,dtype=np.float32)
                        v,e,_=correct(u[j],r,groups,basis,ranks,.5);raw.setdefault(b,[]).append(v);layers.append(e)
                    eta.append(layers)
                for b in (p.HALF_MAIN,p.HALF_TARGET,p.HALF_SOURCE):raw[b]=np.stack(raw[b])
                raw[p.HALF_UNIFORM]=np.stack([matched_uniform(u[j],raw[p.HALF_MAIN][j])[0] for j in range(4)])
                z={b:method.z_values(raw[b],state['patch']) for b in raw};z[p.U]=np.nextafter(z[p.U],np.full_like(z[p.U],np.inf))
                for b in raw:
                    if b!=p.U:z[b][raw[b]==u]=z[p.U][raw[b]==u]
                maps={b:method.response(z[b],pos,[4,4],[16,16],state['image'])[0] for b in raw}
                scores={b:method.response(z[b],pos,[4,4],[16,16],state['image'])[1] for b in raw}
                record=st.put(cache/'predictions'/d/s['sample_id'],dict(z,raw_union=u,raw_main=raw[p.HALF_MAIN],eta=np.stack(eta,axis=1).astype(np.float32),positions=pos),
                    dict(identity=id,sample=s,grid=[4,4],output_shape=[16,16],scores=scores,calibration_seal=mad['seal']))
                records.append(dict(domain=d,sample_id=s['sample_id'],seal=record['seal']))
                sc.extend(dict(domain=d,sample_id=s['sample_id'],relative_path=s['relative_path'],label=label,branch=b,**scores[b]) for b in raw)
                mask=ev.original.shared.resized_mask(gt>0,[16,16]).astype(np.uint8)
                maps_rows.extend(dict(domain=d,sample_id=s['sample_id'],branch=b,output_height=16,output_width=16,map_float32_sha256=ev.original.shared.array_sha(m),mask_uint8_sha256=ev.original.shared.array_sha(mask)) for b,m in maps.items())
                if label:instances.extend(FunctionType(spatial.__code__,dict(spatial.__globals__,U=p.U,S=p.S),spatial.__name__)(s,gt>0,maps,(4,4)))
            mark_small(instances,d);all_instances.extend(instances)
            m=st.sealed_json(parent/'predictions'/(d+'.json'),dict(identity=id,records=records));manifest_seals.append(m['seal'])
            metric=[]
            for b in raw:
                local=[x for x in sc if x['branch']==b]
                for rule in ('CH1','CURRENT'):metric.append(dict(domain=d,branch=b,scoring=rule,**image_metrics([x['label'] for x in local],[x[rule] for x in local]),pixel_AUROC=.7,pixel_AP=.4,AUPRO=.5))
                av=dict(identity=id,domain=d,branch=b,map_identity=st.digest([x for x in maps_rows if x['branch']==b]),nstrips=200,FPR_max=.3,AUPRO=.5,verified_images=2,backend='original_ADEval_1.1.0_CUDA',scope='synthetic_parent_fixture_not_backend_execution');av['seal']=st.digest(av);st.write_json(parent/'aupro'/d/(b+'.json'),av)
                mechs.append(dict(domain=d,branch=b,reference=p.U,R_N=.01,R_A=.02,G=-.01))
            all_metrics.extend(metric);path=parent/'evaluation'/d
            tables={'metrics.csv':metric,'per_image_scores.csv':sc,'defect_instances.csv':instances,'map_manifest.csv':maps_rows}
            for name,rs in tables.items():st.write_csv(path/name,rs)
            st.sealed_json(path/'status.json',dict(identity=id,prediction_seal=m['seal'],images=2,numerical_failed_checks=0,files={n:st.sha(path/n) for n in tables}))
        for b in raw:
            for rule in ('CH1','CURRENT'):all_metrics.append(dict(domain='macro_3domains',branch=b,scoring=rule,**{k:float(np.mean([x[k] for x in all_metrics if x['branch']==b and x['scoring']==rule and x['domain'] in p.DOMAINS])) for k in ('image_AUROC','image_AP','pixel_AUROC','pixel_AP','AUPRO')}))
        from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.reporting import combine
        all_metrics=combine(all_metrics);st.write_csv(parent/'metrics.csv',all_metrics);st.write_csv(parent/'mechanism_summary.csv',mechs);st.write_csv(parent/'margin_summary.csv',[dict(scope='fixture')]);st.write_csv(parent/'defect_instances.csv',all_instances)
        catalog=dict(source=source,supports=supports,samples=samples,data_root='/offline/old_location');st.sealed_json(parent/'catalog.json',dict(catalog=catalog))
        st.sealed_json(parent/'inputs.json',dict(identity=id,config=protocol,catalog=st.digest(catalog),assets_seal=assets['seal'],cache='/offline/old_prediction_location',code={'prediction.py':p.PARENT_PREDICTION_SHA}))
        st.sealed_json(parent/'predictions_frozen.json',dict(identity=id,images=6,seal_of_manifests=st.digest(manifest_seals),GT_used=False));st.write_json(parent/'status.json',dict(identity=id,status='all_five_metrics_complete',pending=[],numerical_failed_checks=0))
        args=runner.normalize(runner.parser().parse_args(['--parent-run',str(parent),'--parent-cache',str(cache),'--run-root',str(self.root/'new_run'),'--cache-root',str(self.root/'new_cache'),'--evaluation-scratch',str(self.root/'scratch'),'--data',str(data),'--cpu-workers','1']))
        return args,samples

    def test_cpu_replay_evaluation_resume_and_completion(self):
        from DINOv3.MADEqual.aebad_s_sacp_strength_p0 import evaluation as ev,reporting
        forbidden=[];searched=[];old_profile=sys.getprofile()
        def profile(frame,event,arg):
            if event=='call':
                file=frame.f_code.co_filename.replace('\\','/');name=frame.f_code.co_name
                if ((any(s in file for s in ('aebad_s_sacp_p0/runtime.py','aebad_s_sacp_p0/clustering.py','aebad_s_sacp_p0/numerics.py','superadd_host.py')) and name in ('host','extraction','extractor','exact','formal','fit','fit_group','kmeans','_exact_top1_chunked'))
                        or ('/training.py' in file and name!='<module>')):
                    forbidden.append((file,name));raise AssertionError('forbidden computation invoked')
            return profile
        def exact(*args,**kwargs):searched.append(kwargs['branch']);return original(*args,**kwargs)
        original=ev.original.exact_pixel_result
        with cpu_only(),patch.object(ev.original,'exact_pixel_result',side_effect=exact):
            sys.setprofile(profile)
            try:
                self.assertEqual(ip.preflight(self.a)['status'],'inventory_ready_not_runtime_verified')
                result=pred.replay(self.a);self.assertEqual(result['new_images'],6)
                self.assertNotIn('defect_instances.csv',ip.parent(self.a)._tables)
                with patch.object(method,'values',side_effect=AssertionError('completed replay recomputed')):self.assertEqual(pred.replay(self.a)['reused_images'],6)
                ev.evaluate(self.a);self.assertEqual(len(searched),9);self.assertEqual(set(searched),set(p.BRANCHES))
                ev.evaluate(self.a);self.assertEqual(len(searched),9)
                status=reporting.summarize(self.a);self.assertEqual(status['status'],'pending');self.assertEqual(len(status['pending']),9)
                self.assertEqual((status['metric_rows'],status['main_metric_rows']),(80,72))
                for d in p.DOMAINS:
                    maps=ip.read_csv(self.a.run_root/'evaluation'/d/'map_manifest.csv')
                    for b in p.BRANCHES:
                        r=dict(identity=ip.identity(self.a),domain=d,branch=b,map_identity=st.digest([x for x in maps if x['branch']==b]),nstrips=200,FPR_max=.3,AUPRO=.5,verified_images=2,
                            backend='original_ADEval_1.1.0_CUDA',scope='reporting_fixture_only_not_GPU_execution');r['seal']=st.digest(r);st.write_json(self.a.run_root/'aupro'/d/(b+'.json'),r)
                complete=reporting.summarize(self.a);self.assertEqual(complete['status'],'all_five_metrics_complete');self.assertEqual(complete['new_AUPRO_complete'],9)
                self.assertTrue(all(v['runtime_reused'] for v in ev.evaluate(self.a,only_aupro=True)))
                with patch.object(ev.original,'exact_pixel_result',side_effect=AssertionError('CPU reevaluated after completion')):ev.evaluate(self.a)
                for d in p.DOMAINS:
                    with self.assertRaises(ValueError):ev.aupro(self.a,d,p.U)
                self.assertEqual(forbidden,[]);self.assertNotIn('torch',sys.modules)
            finally:sys.setprofile(old_profile)
        self.assertTrue(all(result[k]==0 for k in ('query_forward_calls','exact_search_calls','subspace_fit_calls','training_updates')))
        ms=ip.read_csv(self.a.run_root/'margin_summary.csv');self.assertTrue(any(int(r['NA'])>0 for r in ms))
        for r in ms:self.assertEqual(int(r['L1_damaged'])+int(r['L1_improved'])+int(r['L1_unchanged']),int(r['computable']))

    def test_half_replay_identity_and_axes_rejected_on_conflict(self):
        parent=ip.Parent(self.a);s=self.samples[0];x,r=parent.load(s);checks,_=method.replay_half(x,r['metadata'],parent.mad);self.assertEqual(len(checks),22);self.assertTrue(all(v['verified'] for v in checks))
        x=dict(x);x['eta']=np.array(x['eta']);x['eta'][[0,1]]=x['eta'][[1,0]]
        with self.assertRaisesRegex(ValueError,'parent replay mismatch'):method.replay_half(x,r['metadata'],parent.mad)
    def test_no_cache_fallback_and_relocated_paths(self):
        self.assertEqual(ip.preflight(self.a)['status'],'inventory_ready_not_runtime_verified');s=self.samples[0]
        path=self.a.parent_cache/'predictions'/s['domain']/s['sample_id']/'raw_union.npy';path.unlink()
        result=ip.preflight(self.a);self.assertEqual(result['status'],'blocked');self.assertTrue(any(r['asset']=='raw_union' for r in result['missing']))
        with self.assertRaises(OSError):pred.replay(self.a)
    def test_unsupported_parent_eta_schema_rejected(self):
        path=self.a.parent_run/'inputs.json';v=st.sealed_json(path);v.pop('seal');v['code']['prediction.py']='other-schema';st.write_json(path,dict(v,seal=st.digest(v)))
        with self.assertRaisesRegex(ValueError,'eta-axis'):ip.Parent(self.a)
    def test_positions_permutation_preserves_maps_and_half_replay(self):
        parent=ip.Parent(self.a);x,r=parent.load(self.samples[0]);order=np.arange(len(x['positions']))[::-1]
        reordered={k:np.asarray(v)[...,order].copy() for k,v in x.items()}
        checks,_=method.replay_half(reordered,r['metadata'],parent.mad);self.assertTrue(all(v['verified'] for v in checks))
        raw,_=method.values(x,1.);other,_=method.values(reordered,1.)
        for b in p.BRANCHES:
            z=method.calibrated(raw[b],x['raw_union'],x[p.U],parent.mad['patch']);z2=method.calibrated(other[b],reordered['raw_union'],reordered[p.U],parent.mad['patch'])
            first=method.response(z,x['positions'],[4,4],[16,16],parent.mad['image'])[0];second=method.response(z2,reordered['positions'],[4,4],[16,16],parent.mad['image'])[0]
            np.testing.assert_array_equal(first,second)
    def test_parent_manifest_identity_change_is_rejected(self):
        parent=ip.Parent(self.a);s=self.samples[0];path=parent.path(s)/'manifest.json';r=st.sealed_json(path);r.pop('seal');r['metadata']['sample']['sample_id']='wrong_sample';st.write_json(path,dict(r,seal=st.digest(r)))
        with self.assertRaisesRegex(ValueError,'identity differs'):parent.load(s)
    def test_incomplete_replay_blocks_GT_and_changed_CPU_tables_rejected(self):
        from DINOv3.MADEqual.aebad_s_sacp_strength_p0 import evaluation as ev
        with cpu_only():
            pred.replay(self.a);path=self.a.run_root/'predictions/view.json';saved=path.read_bytes();path.unlink()
            with patch.object(ev.GTData,'gt',side_effect=AssertionError('incomplete replay accessed GT')):
                with self.assertRaises(OSError):ev.evaluate(self.a)
            path.write_bytes(saved);commit=self.a.run_root/'predictions_frozen.json';body=commit.read_bytes();commit.unlink()
            with patch.object(ev.GTData,'gt',side_effect=AssertionError('uncommitted replay accessed GT')):
                with self.assertRaises(OSError):ev.evaluate(self.a)
            commit.write_bytes(body);ev.evaluate(self.a)
            table=self.a.run_root/'evaluation/background/metrics.csv';table.write_bytes(table.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'table changed'):ev.evaluate(self.a)
    def test_scratch_protects_parent_and_pack_excludes_BIN(self):
        from DINOv3.MADEqual.aebad_s_sacp_strength_p0 import evaluation as ev
        import tarfile
        untouched=self.a.parent_run/'existing.bin';untouched.write_bytes(b'parent data');wrong=copy.copy(self.a);wrong.evaluation_scratch=self.a.parent_run/'pixels'
        with self.assertRaises(ValueError):
            with ev.scratch(wrong,'background'):pass
        self.assertEqual(untouched.read_bytes(),b'parent data')
        self.a.run_root.mkdir();st.write_json(self.a.run_root/'status.json',dict(status='pending'));self.a.evaluation_scratch.mkdir(parents=True);(self.a.evaluation_scratch/'left.bin').write_bytes(b'partial')
        archive=runner.pack(self.a)
        with tarfile.open(archive['archive']) as t:self.assertFalse(any(m.name.endswith('.bin') or 'pixel_scratch/' in m.name for m in t.getmembers()))


class CLI(unittest.TestCase):
    def test_default_preflight_no_torch_or_output(self):
        with tempfile.TemporaryDirectory(dir=ip.ROOT/'.tmp') as folder:
            out=Path(folder)/'not_created';script=f'from DINOv3.MADEqual.aebad_s_sacp_strength_p0.runner import main; import sys; assert main(["--run-root",{str(out)!r}])==2; assert "torch" not in sys.modules'
            r=subprocess.run([sys.executable,'-B','-c',script],cwd=ip.ROOT,env=dict(os.environ,PYTHONUTF8='1'),text=True,capture_output=True,encoding='utf-8')
            self.assertEqual(r.returncode,0,r.stderr);self.assertFalse(out.exists())


if __name__=='__main__':unittest.main()
