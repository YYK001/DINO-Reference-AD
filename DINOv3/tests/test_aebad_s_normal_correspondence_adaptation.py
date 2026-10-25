"""Small CPU production-path tests. No real DINO, CUDA, installation or experiment."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation import storage as st,inputs as ip,matching as mt
from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation import calibration as cal,reporting
from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.protocol import *


class Mathematics(unittest.TestCase):
    def test_zero_residual_and_full_union(self):
        import torch
        from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.model import Adapter,exact,transform
        r=np.random.default_rng(1);q=r.normal(size=(7,1280)).astype('f');s=r.normal(size=(13,1280)).astype('f');t=r.normal(size=(9,1280)).astype('f')
        net=Adapter(s.mean(0),np.float32(1.2),hidden=8)
        np.testing.assert_array_equal(transform(net,q,'cpu'),q)
        original=np.minimum(exact(q,s,'cpu')[0],exact(q,t,'cpu')[0])
        adapted=np.minimum(exact(transform(net,q,'cpu'),transform(net,s,'cpu'),'cpu')[0],exact(transform(net,q,'cpu'),transform(net,t,'cpu'),'cpu')[0])
        np.testing.assert_array_equal(adapted,original)
        brute=np.linalg.norm(q[:,None]-np.concatenate((s,t))[None],axis=2).min(1)/1280
        np.testing.assert_allclose(original,brute,atol=1e-8,rtol=1e-6)

    def test_both_sides_must_transform(self):
        import torch
        from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.model import Adapter,exact,transform
        net=Adapter(np.zeros(8,'f'),1.,4)
        with torch.no_grad():net.last.bias.fill_(3.)
        q=np.ones((3,8),'f');bank=np.zeros((10,8),'f')
        same=exact(transform(net,q,'cpu'),transform(net,bank,'cpu'),'cpu')[0]
        wrong=exact(transform(net,q,'cpu'),bank,'cpu')[0]
        self.assertTrue(np.all(wrong>same))

    def test_context_is_computed_before_sampling_and_reorder(self):
        r=np.random.default_rng(3);x={l:r.normal(size=(12,6)).astype('f') for l in LAYERS}
        order=r.permutation(12);a=mt.teacher(x,np.arange(12,dtype=np.int64),(3,4),'train/good/a.png')
        b=mt.teacher({l:v[order] for l,v in x.items()},order.astype(np.int64),(3,4),'train/good/a.png')
        for key in a:np.testing.assert_array_equal(a[key],b[key])
        # Corner context has four actual neighbors, not nine zero-padded entries.
        e=mt.normalize(x[7]).reshape(3,4,6);expected=mt.normalize(e[:2,:2].mean((0,1)))
        np.testing.assert_allclose(a['teacher'][0,6:12]*np.sqrt(8),expected,atol=2e-7)

    def test_production_mutual_top3_weights_and_negative_low_half(self):
        r=np.random.default_rng(4);a=mt.normalize(r.normal(size=(18,8)).astype('f'));b=mt.normalize(r.normal(size=(19,8)).astype('f'))
        result=mt.correspond(a,b,('SS','a','b'),'cpu',block=5)
        expected=a@b.T;ab=np.argsort(-expected,axis=1,kind='stable')[:,:3];ba=np.argsort(-expected.T,axis=1,kind='stable')[:,:3]
        for i,(x,y,sim,forward,reverse) in enumerate(((a,b,expected,ab,ba),(b,a,expected.T,ba,ab))):
            found=result[i]
            for row,anchor in enumerate(found['anchor']):
                pos=found['positive'][row];pos=pos[pos>=0]
                self.assertEqual(list(pos),[v for v in forward[anchor] if anchor in reverse[v]])
                self.assertAlmostEqual(float(found['weight'][row].sum()),1.,places=6)
                self.assertEqual(len(set(found['negative'][row])),4)
                low=set(np.argsort(sim[anchor],kind='stable')[:len(y)//2])
                self.assertTrue(set(found['negative'][row])<=low-set(pos))
        repeated=mt.correspond(a,b,('SS','a','b'),'cpu',block=7)
        for i in (0,1):
            for key in result[i]:
                if key=='weight':np.testing.assert_allclose(result[i][key],repeated[i][key],atol=6e-7,rtol=2e-6)
                else:np.testing.assert_array_equal(result[i][key],repeated[i][key])
        identical=mt.correspond(a,b,('SS','a','b'),'cpu',block=5)
        for i in (0,1):
            for key in result[i]:np.testing.assert_array_equal(result[i][key],identical[i][key])

    def test_ties_and_empty_pairs(self):
        a=np.ones((16,8),'f');b=a.copy();found=mt.correspond(a,b,('tie',),'cpu')
        np.testing.assert_array_equal(found[0]['anchor'],[0,1,2])
        np.testing.assert_array_equal(found[0]['positive'],np.tile([0,1,2],(3,1)))
        with self.assertRaisesRegex(ValueError,'four'):mt.correspond(a[:8],b[:8],('insufficient',),'cpu')

    def test_pair_graph_exclusion_and_no_gt(self):
        rows=[dict(relative_path=f'S{i:02}',source_index=i) for i in range(24)]
        rows += [dict(relative_path=f'T{i:02}',support_index=i+1) for i in range(8)]
        desc={r['relative_path']:mt.normalize(np.random.default_rng(i).normal(size=8).astype('f')) for i,r in enumerate(rows)}
        self.assertEqual(len(mt.image_pairs(rows,desc,'TT')),28)
        stpairs=mt.image_pairs(rows,desc,'ST')
        self.assertEqual(len(stpairs),8*20)
        ss=mt.image_pairs(rows,desc,'SS');self.assertTrue(all(a!=b and a.startswith('S') and b.startswith('S') for a,b in ss))
        for kind in ('SS','ST','TT'):self.assertEqual(mt.image_pairs(rows,desc,kind),mt.image_pairs(rows[::-1],desc,kind))

    def test_population_training_separation(self):
        source=[dict(relative_path='train',calibration=False),dict(relative_path='calibration',calibration=True)]
        c=dict(source=source,supports={d:[dict(relative_path=d+'/support')] for d in DOMAINS})
        ip.validate_training(c,'view',ip.training_rows(c,'view'))
        for illegal in ('calibration','illumination/support','view/test'):
            with self.assertRaises(ValueError):ip.validate_training(c,'view',[dict(relative_path=illegal)])

    def test_loss_finite_gradients_and_source_scale(self):
        import torch
        from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.model import Adapter,paired_loss,lr_at
        from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.runtime import source_scales
        r=np.random.default_rng(8);x=r.normal(size=(30,8)).astype('f');mu,sigma=source_scales(x,chunk=7)
        np.testing.assert_allclose(mu,x.mean(0),atol=1e-7)
        self.assertAlmostEqual(float(sigma),float(np.sqrt(((x-x.mean(0))**2).mean())),places=6)
        net=Adapter(mu,sigma,hidden=5);anchor=torch.tensor(x[:4]);positive=torch.tensor(x[4:16].reshape(4,3,8));neg=torch.tensor(x[8:24].reshape(4,4,8))
        w=torch.full((4,3),1/3);p,n,i=paired_loss(net,anchor,positive,w,neg,torch.tensor(x))
        self.assertEqual(float(n),0.);self.assertEqual(float(i),0.)
        (p+n+.1*i).backward();self.assertTrue(all(torch.isfinite(v.grad).all() for v in net.parameters()))
        self.assertEqual(lr_at(1999,2000),1e-6)
        with self.assertRaises(ValueError):source_scales(np.ones((4,8),'f'))

    def test_ensemble_calibration_not_average_image_mad(self):
        r=np.random.default_rng(19);states=[];raw=[]
        meta=dict(positions=list(range(12)),grid=[3,4],output_shape=[7,8])
        for j in range(3):
            values=[r.uniform(.1,2.,size=(4,12)).astype('f') for _ in range(24)]
            patch=cal.fit_patch(values);states.append(patch);raw.append(values)
        records=[]
        for i in range(24):records.append((np.stack([cal.z_values(raw[j][i],states[j]) for j in range(3)]).mean(0,dtype=np.float32),meta))
        fitted=cal.fit_image(records)
        from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import fit_image_mad
        peaks=[cal.from_z(z,np.arange(12),[3,4],[7,8])[1] for z,_ in records]
        med,scale=fit_image_mad(peaks)
        np.testing.assert_array_equal(med,np.array(fitted['peak']['medians'],'f'))
        separate=[cal.fit_image([(x,meta) for x in raw[j]],states[j]) for j in range(3)]
        averaged=np.mean([s['peak']['medians'] for s in separate],axis=0)
        self.assertFalse(np.allclose(averaged,med))
        from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import statistics
        current,peak,local=statistics(raw[0][0],np.arange(12,dtype=np.int64),(3,4),(7,8),
            np.array(states[0]['medians'],'f'),np.array(states[0]['scales'],'f'))
        response,p,l=cal.from_z(cal.z_values(raw[0][0],states[0]),np.arange(12),[3,4],[7,8])
        self.assertEqual(current,float(response.max()));np.testing.assert_array_equal(p,peak);np.testing.assert_array_equal(l,local)


class Artifacts(unittest.TestCase):
    def test_mmap_store_bounded_and_manifest_change_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);store=st.Store(max_open=2)
            for i in range(5):
                p=root/str(i);st.put(p,dict(x=np.ones((2,2),'f')),dict(index=i));store.load(p)
            self.assertEqual(len(store.records),2);self.assertEqual(len(store.verified),5)
            p=root/'0';record=st.read_json(p/'manifest.json');record['metadata']['index']=9
            record['seal']=st.digest({k:v for k,v in record.items() if k!='seal'});st.write_json(p/'manifest.json',record)
            with self.assertRaises(ValueError):store.load(p)
            store.records.clear();__import__('gc').collect()

    def test_recent_fresh_and_fp32_child_schemas_read_without_old_graphs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fresh=root/'fresh';child=root/'fp32';fresh.mkdir();child.mkdir()
            cat=dict(source=[],supports={d:[] for d in DOMAINS},samples=[])
            for p in (fresh,child):
                st.write_json(p/'fresh_catalog.json',cat)
                st.write_json(p/'fresh_catalog.seal.json',dict(sha256=st.sha(p/'fresh_catalog.json')))
            conf=dict(revision='witness_fresh_rgb_seed42_v1',software={},code={})
            setup=dict(configuration=conf,catalog_sha256=st.sha(fresh/'fresh_catalog.json'),cache=str(root/'fresh_cache'))
            setup['identity']=st.digest(setup);st.write_json(fresh/'setup.json',setup)
            childconf=dict(engine='fp32_single_edge_pass_fused_refinement_v1',parent=str(fresh),
                parent_records={str(fresh/'setup.json'):st.sha(fresh/'setup.json')})
            st.write_json(child/'setup.json',dict(configuration=childconf,identity=st.digest(childconf),cache=str(root/'child_cache')))
            with patch.object(ip.fi,'validate_catalog'):
                self.assertEqual(ip.recent(fresh)['configuration'],conf)
                recent=ip.recent(child);self.assertEqual(recent['configuration'],conf)
                self.assertEqual(recent['cache'],root/'child_cache')
                self.assertFalse((root/'child_cache').exists())

    def test_hash_modified_array_and_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'a';st.put(p,dict(x=np.ones((3,4),'f')),dict(identity='i'))
            st.get(p)
            with (p/'x.npy').open('r+b') as f:f.seek(-1,2);f.write(b'\xff')
            with self.assertRaisesRegex(ValueError,'modified array'):st.get(p)

    def test_preflight_no_torch_no_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'no_create'
            code='''import sys,json\nfrom DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.runner import main\nrc=main(['preflight','--run-root',sys.argv[1],'--data',sys.argv[1]+'_missing','--checkpoint',sys.argv[1]+'_weight'])\nassert 'torch' not in sys.modules\nfrom pathlib import Path\nassert not Path(sys.argv[1]).exists()\nassert rc==2\n'''
            result=subprocess.run([sys.executable,'-B','-c',code,str(out)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_unfrozen_predictions_refuse_gt(self):
        from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation import prediction as pr
        with tempfile.TemporaryDirectory() as tmp:
            a=SimpleNamespace(run_root=Path(tmp),cache_root=Path(tmp)/'cache')
            st.sealed_json(a.run_root/'catalog.json',dict(catalog=dict(samples=[])))
            with self.assertRaises((FileNotFoundError,ValueError)):pr.verify(a)

    def test_reference_and_comparison_direction(self):
        rows=[]
        for b,v in zip((U,S,*MEMBERS,ENS),(.6,.7,.71,.72,.73,.74)):
            for scoring in ('CH1','CURRENT'):rows.append(dict(domain='d',branch=b,scoring=scoring,**{k:v for k in reporting.METRICS}))
        out=reporting.combine(rows);combined=next(r for r in out if r['branch']==COMBINED)
        self.assertEqual(combined['image_AUROC'],.7);self.assertEqual(combined['pixel_AP'],.6)
        difference=next(r for r in reporting.comparisons(out) if r['branch']==ENS and r['reference']==COMBINED and r['metric']=='image_AP')
        self.assertAlmostEqual(difference['delta'],4.)

    def test_pack_excludes_large_scratch_checkpoints(self):
        import tarfile
        from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.runner import pack
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)/'run';(run/'pixel_scratch').mkdir(parents=True);(run/'models').mkdir()
            (run/'pixel_scratch'/'large.bin').write_bytes(b'0');(run/'models'/'x.pt').write_bytes(b'0')
            st.write_json(run/'status.json',dict(status='pending'))
            p=pack(SimpleNamespace(run_root=run))['archive']
            with tarfile.open(p) as t:self.assertEqual(t.getnames(),['run/status.json'])


class ScratchIsolation(unittest.TestCase):
    def args(self,root,scratch,recent=None):
        from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.runner import parser
        argv=['preflight','--run-root',str(root/'new_run'),'--cache-root',str(root/'new_cache'),
              '--evaluation-scratch',str(scratch),'--data',str(root/'inputs'/'AeBAD.zip'),
              '--checkpoint',str(root/'inputs'/'checkpoint.pth'),
              '--official-root',str(root/'host'),'--dinov3-root',str(root/'dinov3')]
        if recent is not None:argv += ['--recent-run',str(recent)]
        return parser().parse_args(argv)

    def test_cli_rejects_scratch_in_parent_inputs_cache_and_ancestor(self):
        from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.runner import normalize
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);recent=root/'parent';original=root/'original_parent';oldcache=root/'external_parent_cache'
            st.write_json(recent/'setup.json',dict(cache=str(root/'recent_cache'),configuration=dict(parent=str(original))))
            st.write_json(original/'setup.json',dict(cache=str(oldcache),configuration={}))
            protected=[recent,recent/'scratch',root,oldcache/'scratch',root/'recent_cache'/'scratch',
                root/'host'/'scratch',root/'dinov3',root/'inputs',root/'new_cache'/'scratch',root/'new_run']
            for scratch in protected:
                with self.subTest(scratch=scratch),self.assertRaises(ValueError):normalize(self.args(root,scratch,recent))
            normalize(self.args(root,root/'safe_scratch',recent))
            normalize(self.args(root,root/'new_run'/'pixel_scratch',recent))
            self.assertFalse((root/'new_run').exists());self.assertFalse((root/'safe_scratch').exists())

    def test_direct_evaluation_rejects_parent_before_any_result_write(self):
        from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation import evaluation as ev
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parent=root/'parent';parent.mkdir()
            marker=parent/'background'/'keep.bin';marker.parent.mkdir();marker.write_bytes(b'parent bytes')
            a=self.args(root,parent,parent)
            with self.assertRaises(ValueError),patch.object(ev,'restored') as restored:
                ev.evaluate_domain(a,'background')
            restored.assert_not_called()
            self.assertEqual(marker.read_bytes(),b'parent bytes');self.assertFalse(a.run_root.exists())

    def test_allocation_failure_keeps_old_and_unowned_bins(self):
        from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation import evaluation as ev
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);scratch=root/'scratch';scratch.mkdir();a=self.args(root,scratch)
            parent_bin=scratch/'background'/'labels.bin';parent_bin.parent.mkdir();parent_bin.write_bytes(b'old')
            own=[];unowned=[];real_memmap=np.memmap
            def fail_second(file,*args,**kwargs):
                own.append(Path(file.name))
                if len(own)==2:
                    extra=Path(file.name).parent/'do_not_delete.bin';extra.write_bytes(b'unowned');unowned.append(extra)
                    raise MemoryError('allocation fixture')
                return real_memmap(file,*args,**kwargs)
            with patch.object(ev.np,'memmap',side_effect=fail_second):
                with self.assertRaisesRegex(MemoryError,'allocation fixture'):
                    with ev.scratch_arrays(a,'background') as allocate:
                        allocate('labels',np.uint8,4);allocate(U,np.float32,4)
            self.assertEqual(parent_bin.read_bytes(),b'old')
            self.assertTrue(own);self.assertTrue(all(not p.exists() for p in own))
            self.assertEqual(unowned[0].read_bytes(),b'unowned')

    def test_success_cleanup_owns_only_unique_workspace_and_refuses_existing_name(self):
        from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation import evaluation as ev
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=self.args(root,root/'scratch');owned=[]
            with ev.scratch_arrays(a,'view') as allocate:
                value=allocate('labels',np.uint8,4);value[:]=[0,1,0,1]
                job=Path(value.filename).parent;owned.append(job)
                # Exclusive creation cannot truncate even a name in our own workspace.
                with self.assertRaises(FileExistsError):allocate('labels',np.uint8,8)
                np.testing.assert_array_equal(value,[0,1,0,1])
            self.assertFalse(job.exists())
            with ev.scratch_arrays(a,'view') as allocate:
                again=allocate('labels',np.uint8,4);second=Path(again.filename).parent
                self.assertNotEqual(second,owned[0])
            self.assertFalse(second.exists())

    def test_symlink_to_parent_scratch_is_rejected(self):
        from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.runner import normalize
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parent=root/'parent';parent.mkdir();alias=root/'alias'
            try:alias.symlink_to(parent,target_is_directory=True)
            except OSError as e:self.skipTest('directory symlink unavailable: '+str(e))
            with self.assertRaises(ValueError):normalize(self.args(root,alias/'scratch',parent))


class ActualTraining(unittest.TestCase):
    def fixture(self,root):
        a=SimpleNamespace(run_root=root/'run',cache_root=root/'cache',device='cpu')
        st.sealed_json(a.run_root/'inputs.json',dict(identity='tiny'))
        rows=[dict(relative_path='source/'+str(i),source_index=i,calibration=False) for i in range(2)]
        target=[dict(relative_path='background/'+str(i),support_index=i+1,domain='background') for i in range(8)]
        c=dict(source=rows,supports={'background':target})
        st.sealed_json(a.run_root/'catalog.json',dict(catalog=c));r=np.random.default_rng(2)
        st.put(a.cache_root/'source_original',{f'L{l}':r.normal(size=(40,8)).astype('f') for l in LAYERS},dict(identity='tiny'))
        for row in rows+target:
            st.put(ip.feature_path(a,row),{**{f'L{l}':r.normal(size=(16,8)).astype('f') for l in LAYERS},
                **{f'full_L{l}':r.normal(size=(16,8)).astype('f') for l in LAYERS}},dict(identity='tiny',image=row))
        for domain,kind,left,right in [('source','SS',rows[0],rows[1]),('background','ST',rows[0],target[0]),('background','TT',target[0],target[1])]:
            lx,lr=st.get(ip.feature_path(a,left));rx,rr=st.get(ip.feature_path(a,right))
            teacher_left=mt.normalize(lx['L7']);teacher_right=mt.normalize(rx['L7'])
            values=mt.correspond(teacher_left,teacher_right,(kind,),'cpu')
            p=a.cache_root/'matches'/domain/kind/'pair'
            rec=st.put(p,{f'{k}_{i}':v for i,d in enumerate(values) for k,v in d.items()},dict(
                left_seal=lr['seal'],right_seal=rr['seal'],left=left['relative_path'],right=right['relative_path']))
            st.sealed_json(a.run_root/'matches'/domain/(kind+'.json'),dict(identity='tiny',records=[dict(path=str(p),seal=rec['seal'],
                left=left['relative_path'],right=right['relative_path'],anchors=[len(v['anchor']) for v in values])]))
        return a

    def test_actual_optimizer_resume_same_final_weights_and_target_initialization(self):
        import torch
        from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation import training as tr
        with tempfile.TemporaryDirectory() as tmp:
            a=self.fixture(Path(tmp)/'a');b=self.fixture(Path(tmp)/'b')
            with patch.dict(CONFIG,source_steps=4,target_steps=5,hidden=6,batch=6):
                tr.train_job(a,'source',42,7,steps=4,hidden=6,batch_size=6,stop_after=2)
                self.assertFalse((tr.job_path(a,'source',42,7)/'complete.json').exists())
                tr.train_job(a,'source',42,7,steps=4,hidden=6,batch_size=6)
                tr.train_job(b,'source',42,7,steps=4,hidden=6,batch_size=6)
                ca=st.sealed_json(tr.job_path(a,'source',42,7)/'complete.json');cb=st.sealed_json(tr.job_path(b,'source',42,7)/'complete.json')
                xa=tr.load_checkpoint(tr.job_path(a,'source',42,7),ca['inputs'],'cpu')
                xb=tr.load_checkpoint(tr.job_path(b,'source',42,7),cb['inputs'],'cpu')
                self.assertEqual(xa['updates'],4)
                for k in xa['network']:self.assertTrue(torch.equal(xa['network'][k],xb['network'][k]),k)
                tr.train_job(a,'background',42,7,steps=5,hidden=6,batch_size=6)
                done=st.sealed_json(tr.job_path(a,'background',42,7)/'complete.json')
                self.assertEqual(done['inputs']['Source_initialization'],ca['checkpoint_sha256'])
                self.assertEqual(done['updates'],5)
                tr.trained(a,'background',42,7,'cpu')
                latest=st.sealed_json(tr.job_path(a,'background',42,7)/'latest.json')
                checkpoint=tr.job_path(a,'background',42,7)/latest['file']
                with checkpoint.open('ab') as f:f.write(b'modified')
                with self.assertRaises(ValueError):tr.trained(a,'background',42,7,'cpu')


class OfflineIntegration(unittest.TestCase):
    def test_transform_calibrate_predict_cpu_evaluate_resume_without_backbone_or_gt_training(self):
        """Actual 1280D NN, calibration, maps, CPU pixels, seals and resume.

        Substitutes only trained networks, image provider and backbone feature
        extraction; 24 synthetic calibration images, two queries per domain.
        This does NOT certify real training/CUDA/DINO/ADEval.
        """
        import torch
        from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation import prediction as pr,evaluation as ev
        from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.model import Adapter
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=SimpleNamespace(run_root=root/'run',cache_root=root/'cache',device='cpu',query_mode='stream',cpu_workers=2,evaluation_scratch=root/'scratch')
            st.sealed_json(a.run_root/'inputs.json',dict(identity='integration'))
            r=np.random.default_rng(23);source=[dict(relative_path='train/'+str(i),source_index=i,calibration=True) for i in range(24)]
            supports={d:[dict(relative_path=d+'/support/'+str(i),support_index=i+1,domain=d) for i in range(8)] for d in DOMAINS}
            samples=[dict(domain=d,sample_id='test_'+str(y),relative_path=d+'/query/'+str(y),label=y,image_height=48,image_width=64,image_sha256='synthetic') for d in DOMAINS for y in (0,1)]
            c=dict(source=source,supports=supports,samples=samples,data_root=str(root/'data'))
            st.sealed_json(a.run_root/'catalog.json',dict(catalog=c))
            st.put(a.cache_root/'source_original',{f'L{l}':r.normal(size=(32,1280)).astype('f') for l in LAYERS},dict(identity='integration'))
            for row in source+sum(supports.values(),[]):
                arrays={f'full_L{l}':r.normal(size=(12,1280)).astype('f') for l in LAYERS};arrays['full_positions']=np.arange(12,dtype=np.int64)
                st.put(ip.feature_path(a,row),arrays,dict(identity='integration',image=row,grid=[3,4],output_shape=[12,16]))
            q={s['relative_path']:{l:r.normal(size=(12,1280)).astype('f') for l in LAYERS} for s in samples}
            calls=[];gtcalls=[]
            class Data:
                def __init__(self,path):pass
                def __enter__(self):return self
                def __exit__(self,*args):pass
                def image(self,s):return s['relative_path']
                def gt(self,s):
                    self_test.assertTrue((a.run_root/'predictions_frozen.json').exists())
                    gtcalls.append((s['domain'],s['sample_id']))
                    m=np.zeros((48,64),np.uint8)
                    if s['label']:m[10:24,20:35]=1
                    return m
            self_test=self
            def extraction(model,image,device):
                calls.append(image);return q[image],dict(grid=[3,4],positions=np.arange(12,dtype=np.int64))
            def trained(a,domain,seed,layer,device):
                torch.manual_seed(seed+layer);net=Adapter(np.zeros(1280,'f'),2.,8)
                with torch.no_grad():net.last.weight.fill_(float(seed-40)*1e-4)
                return net,dict(checkpoint_sha256=f'{domain}/{seed}/{layer}')
            with patch('DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.training.trained',trained),\
                 patch('DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.runtime.host',return_value=(object(),'cpu')),\
                 patch('DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.runtime.extraction',extraction),\
                 patch('DINOv3.MADEqual.aebad_s_support_witness_segment_p0.fresh_inputs.Data',Data),\
                 patch.dict(CONFIG,populations={d:[2,1,1] for d in DOMAINS}):
                cal.transform_memory(a);cal.calibrate(a)
                self.assertEqual(gtcalls,[])
                done=pr.predict(a);self.assertEqual(done['new_images'],6);self.assertEqual(len(calls),6)
                original_hashes={s['relative_path']:st.sha(pr.prediction_path(a,s)/'manifest.json') for s in samples}
                for d in DOMAINS:
                    saved=ev.evaluate_domain(a,d);self.assertEqual(saved['images'],2)
                    metrics=ev.read_csv(a.run_root/'evaluation'/d/'metrics.csv');self.assertEqual(len(metrics),12)
                    self.assertTrue(all(m['AUPRO']=='' for m in metrics))
                    self.assertEqual(ev.evaluate_domain(a,d)['status'],'CPU_reused')
                before=len(calls);again=pr.predict(a);self.assertEqual(again['new_images'],0);self.assertEqual(len(calls),before)
                self.assertEqual(original_hashes,{s['relative_path']:st.sha(pr.prediction_path(a,s)/'manifest.json') for s in samples})
                for s in samples:
                    x,_=st.get(pr.prediction_path(a,s));np.testing.assert_array_equal(x[ENS],np.stack([x[b] for b in MEMBERS]).mean(0,dtype=np.float32))
                del x
                __import__('gc').collect()


if __name__=='__main__':unittest.main()
