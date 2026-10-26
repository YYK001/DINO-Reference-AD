"""Small real CPU tensors; no DINO forward, real training data, CUDA or ADEval."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
import numpy as np

from DINOv3.MADEqual.aebad_s_reference_residual_segmentation import (
    protocol as p,storage as st,inputs as ip,synthesis,tiles,runner,calibration,reporting)

BASE=ip.ROOT/'.tmp/reference_residual_tests'
BASE.mkdir(parents=True,exist_ok=True)


def catalog():
    source=[dict(relative_path=f'train/{i:04d}.png',source_index=i,calibration=i%8==0,label=0,
        image_height=64,image_width=64,image_sha256=str(i)) for i in range(521)]
    supports={d:[dict(relative_path=f'{d}/{i}.png',domain=d,support_index=i,label=0,
        image_height=64,image_width=64,image_sha256=d+str(i)) for i in range(1,9)] for d in p.DOMAINS}
    return dict(source=source,supports=supports,samples=[],data_root='unused')


class Temporary(unittest.TestCase):
    def setUp(self):
        st.configure('full')  # Existing corruption tests exercise explicit audit mode.
        self.temp=tempfile.TemporaryDirectory(dir=BASE);self.root=Path(self.temp.name)
    def tearDown(self):
        st.configure('fast');self.temp.cleanup()
    def args(self,name='run'):
        return runner.normalize(runner.parser().parse_args(['--run-root',str(self.root/name),'--cache-root',str(self.root/(name+'_cache')),
            '--evaluation-scratch',str(self.root/(name+'_scratch')),'--data',str(self.root/'input.zip'),'--checkpoint',str(self.root/'weights.pth'),
            '--official-root',str(self.root/'host'),'--dinov3-root',str(self.root/'dino'),'--device','cpu','--profile','full_v1']))


class Origins(unittest.TestCase):
    def setUp(self):self.c=catalog()
    def test_455_split_and_fold_derivatives(self):
        rows=ip.train_source(self.c);fold=ip.folds(self.c)
        self.assertEqual(len(rows),455);self.assertEqual([list(fold.values()).count(k) for k in 'AB'],[228,227])
        for row in rows:
            policy=ip.policy(self.c,row,'source')
            self.assertNotIn(row['relative_path'],policy['source_origins']);self.assertNotIn(row['relative_path'],policy['target_origins'])
            self.assertTrue(set(policy['target_origins'])<=set(policy['source_origins']))
            self.assertIn(len(policy['target_origins']),(7,8));self.assertNotEqual(policy['source'],'formal')
            self.assertEqual(policy,ip.policy(self.c,dict(row,synthetic_variant=4),'source'))
    def test_target_leave_one_and_domain(self):
        for domain in p.DOMAINS:
            for row in self.c['supports'][domain]:
                v=ip.policy(self.c,row,domain);self.assertEqual(v['source'],'formal')
                self.assertEqual(len(v['target_origins']),7);self.assertNotIn(row['relative_path'],v['target_origins'])
                for other in p.DOMAINS:
                    if other!=domain:
                        with self.assertRaises(ValueError):ip.policy(self.c,row,other)
    def test_source_domain_uses_frozen_K8(self):
        row=ip.train_source(self.c)[0]
        for d in p.DOMAINS:self.assertEqual(ip.policy(self.c,row,d)['target_origins'],[r['relative_path'] for r in self.c['supports'][d]])
    def test_calibration_and_foreign_forbidden(self):
        with self.assertRaises(ValueError):ip.policy(self.c,self.c['source'][0],'source')
        with self.assertRaises(ValueError):ip.policy(self.c,dict(relative_path='test/unknown',calibration=False),'background')
    def test_original_Source_order_and_mod8_not_just_counts(self):
        ip.validate_source(self.c)
        self.c['source'][1]['calibration']=True;self.c['source'][0]['calibration']=False
        with self.assertRaisesRegex(ValueError,'split differs'):ip.validate_source(self.c)
    def test_original_image_hashes_not_only_population_counts(self):
        frozen=st.read_json(Path(ip.__file__).with_name('frozen_origins.json'))
        c=dict(source=[dict(r,source_index=i,calibration=i%8==0) for i,r in enumerate(frozen['source'])],
            supports={'background':frozen['background_supports']},samples=[dict(r) for r in frozen['evaluation_GT']])
        back={r['sample_id']:r for r in frozen['background_samples']}
        for r in c['samples']:
            if r['domain']=='background':r.update(back[r['sample_id']])
        ip.validate_frozen_origins(c);c['source'][1]['image_sha256']='changed'
        with self.assertRaisesRegex(ValueError,'Source image identities'):ip.validate_frozen_origins(c)
        c['source'][1]['image_sha256']=frozen['source'][1]['image_sha256']
        next(r for r in c['samples'] if r['label'])['gt_sha256']='changed'
        with self.assertRaisesRegex(ValueError,'GT identities'):ip.validate_frozen_origins(c)
    def test_fixed_pool_and_donors(self):
        # Small population exercises the production manifest without 5944 RGBs.
        self.c['source']=self.c['source'][:17]
        first=synthesis.manifest(self.c,p.resolve('full_v1'));second=synthesis.manifest(self.c,p.resolve('full_v1'))
        self.assertEqual(first,second);self.assertEqual(len(first),14*8+24*96)
        valid={r['relative_path'] for r in ip.train_source(self.c)}
        lookup={r['relative_path']:d for d in p.DOMAINS for r in self.c['supports'][d]}
        for r in first:
            self.assertNotEqual(r['origin'],r['donor'])
            if r['origin'] in valid:self.assertIn(r['donor'],valid)
            else:self.assertEqual(lookup[r['origin']],lookup[r['donor']])
        counts={k:sum(r['kind']==k for r in first) for k in ('poisson','texture','scratch')}
        self.assertEqual(counts['poisson'],2*counts['scratch']);self.assertEqual(counts['texture'],counts['scratch'])
    def test_predeclared_populations(self):
        self.assertEqual(sum(v[0] for v in p.CONFIG['populations'].values()),926)
        self.assertEqual(len(p.BRANCHES),14);self.assertEqual(p.SEEDS,(42,43,44))


class Synthesis(unittest.TestCase):
    def test_area_preserves_single_pixel(self):
        m=np.zeros((31,37),np.float32);m[15,18]=1
        small=synthesis.area_resize(m,(7,9));self.assertGreater(float(small.max()),0)
        self.assertAlmostEqual(float(small.sum()*31*37/63),1,places=5)
    def test_unchanged_not_positive(self):
        a=np.full((32,32,3),128,np.uint8);mask=np.ones((32,32),np.uint8)
        soft,change=synthesis.changed_label(a,a,mask);self.assertEqual(float(soft.sum()),0);self.assertEqual(float(change.max()),0)
    def test_requested_and_actual_required(self):
        a=np.zeros((12,12,3),np.uint8);b=a.copy();b[3,4]=255;b[5,6]=255;m=np.zeros((12,12),bool);m[3,4]=1
        soft,delta=synthesis.changed_label(a,b,m);self.assertEqual(soft[3,4],1);self.assertEqual(soft[5,6],0);self.assertEqual(delta[5,6],1)
    def test_all_three_RGB_families_and_determinism(self):
        rng=np.random.default_rng(4);a=rng.integers(0,256,(128,128,3),dtype=np.uint8);b=a[::-1].copy();fg=np.ones((8,8),bool)
        for variant in (8,2,3):
            x,r=synthesis.generate(a,b,fg,'normal-origin',variant);y,t=synthesis.generate(a,b,fg,'normal-origin',variant)
            self.assertIsNotNone(x);self.assertEqual(r,t);np.testing.assert_array_equal(x['rgb'],y['rgb'])
            self.assertGreater(np.count_nonzero(x['soft']),0);self.assertLessEqual(r['attempts'],p.CONFIG['retry_limit'])
            self.assertTrue(np.all(x['soft'][x['requested']==0]==0))
    def test_invalid_foreground_no_fallback(self):
        x=np.zeros((8,8,4),np.float32);mask,r=synthesis.foreground(x,np.ones(4),np.zeros(4))
        self.assertFalse(mask.any());self.assertIn('invalid',r['status'])
        image=np.zeros((32,32,3),np.uint8);value,r=synthesis.generate(image,image,mask,'origin',0)
        self.assertIsNone(value);self.assertEqual(r['attempts'],0)
    def test_synthetic_tile_coverage_and_all_normal(self):
        m=np.zeros((81,91),np.float32);m[40,:]=.1
        plans=tiles.plan(m,(37,41),'fixed');coverage=np.zeros((37,41),bool)
        for t in plans:
            y,x,yy,xx=t['center'];coverage[y:yy,x:xx]=True
        self.assertTrue(coverage[synthesis.area_resize(m,(37,41))>0].all())
        normal=tiles.plan(np.zeros_like(m),(37,41),'normal');self.assertEqual(len(normal),9)
    def test_output_windows_partition_every_pixel(self):
        count=np.zeros((73,91),int)
        for t in tiles.plan(np.zeros((73,91),np.float32),(37,41),'normal'):
            y,x,yy,xx=tiles.output_window(t['center'],(37,41),count.shape);count[y:yy,x:xx]+=1
        np.testing.assert_array_equal(count,1)


class Bundles(Temporary):
    def test_modified_payload_rejected(self):
        dest=self.root/'bundle';st.put(dest,dict(x=np.arange(6,dtype=np.float32)),dict(identity='tiny'))
        with (dest/'x.npy').open('wb') as f:np.save(f,np.full(6,9,np.float32))
        with self.assertRaisesRegex(ValueError,'modified array'):st.Store().load(dest)
    def test_no_reference_loader_does_not_read_poisoned_NN(self):
        from DINOv3.MADEqual.aebad_s_reference_residual_segmentation.training_data import Dataset
        a=self.args();c=catalog();row=c['source'][1];c['source']=[row];c['supports']={d:[] for d in p.DOMAINS}
        st.sealed_json(a.run_root/'catalog.json',dict(catalog=c));st.sealed_json(a.run_root/'inputs.json',dict(identity='tiny'))
        item=dict(origin=row['relative_path'],variant=-1,item_id='normal',row=row,status='normal');dest=ip.item_path(a,item)
        features={f'q{l}':np.ones((16,1280),np.float32) for l in p.LAYERS};features['positions']=np.arange(16,dtype=np.int64)
        fr=st.put(ip.normal_path(a,row),features,dict(grid=[4,4]));lr=st.put(dest/'labels',dict(soft=np.zeros((16,16),np.float32)),{})
        t=dict(tile=0,center=[0,0,4,4],bounds=[0,0,4,4],positive=False)
        rec=st.sealed_json(dest/'complete.json',dict(grid=[4,4],positions=list(range(16)),tiles=[t],output_shape=[16,16],
            normal_feature_seal=fr['seal'],labels_seal=lr['seal'],contexts='POISON'))
        synthetic=dict(item,variant=0,item_id='dummy')
        st.sealed_json(a.run_root/'training_manifest.json',dict(identity='tiny',records=[dict(item=item,path=str(dest),seal=rec['seal']),
            dict(item=synthetic,path=str(dest),seal=rec['seal'])]))
        st.put(a.cache_root/'input_normalization',{**{f'mu{l}':np.zeros(1280,np.float32) for l in p.LAYERS},**{f'sigma{l}':np.asarray(1.,np.float32) for l in p.LAYERS}}, {})
        st.sealed_json(a.run_root/'input_distance_mad.json',dict(patch=dict(medians=[0]*4,scales=[1]*4)))
        dataset=Dataset(a,'source',False)
        with patch('DINOv3.MADEqual.aebad_s_reference_residual_segmentation.training_data.References',side_effect=AssertionError('reference read')):
            sample=dataset.load((item,t))
        self.assertEqual(len(sample['values']),4);self.assertEqual(sample['values'][0][:,1280:].count_nonzero().item(),0)
    def test_partial_population_cannot_freeze(self):
        from DINOv3.MADEqual.aebad_s_reference_residual_segmentation.prediction import verify
        a=self.args();st.sealed_json(a.run_root/'catalog.json',dict(catalog=catalog()))
        with self.assertRaisesRegex(ValueError,'population'):verify(a)
    def test_output_isolation_includes_recent_inputs_cache(self):
        a=self.args();parent=self.root/'parent';oldcache=self.root/'parent_cache'
        st.write_json(parent/'inputs.json',dict(cache_root=str(oldcache)))
        a.recent_run=[parent];a.evaluation_scratch=oldcache/'nested'
        with self.assertRaisesRegex(ValueError,'overlaps'):runner.check_isolation(a)
    def test_saved_providers_protected_when_resume_omits_recent_argument(self):
        a=self.args();parent=self.root/'parent';oldcache=self.root/'oldcache'
        st.sealed_json(a.run_root/'eligible_raw_providers.json',dict(providers=[dict(kind='correspondence',run=str(parent),cache=str(oldcache))]))
        a.evaluation_scratch=oldcache/'scratch'
        with self.assertRaisesRegex(ValueError,'overlaps'):runner.check_isolation(a)
    def test_explicit_raw_assets_protected(self):
        a=self.args();bank=self.root/'old_bank';manifest=self.root/'raw.json'
        st.write_json(manifest,dict(source_bank=dict(path=str(bank)),normal_features={}))
        a.asset_manifest=[manifest];a.cache_root=bank/'child'
        with self.assertRaisesRegex(ValueError,'overlaps'):runner.check_isolation(a)
    def test_failed_scratch_allocation_preserves_existing(self):
        from DINOv3.MADEqual.aebad_s_reference_residual_segmentation.evaluation import scratch_arrays
        a=self.args();a.evaluation_scratch.mkdir(parents=True);sentinel=a.evaluation_scratch/'existing.bin';sentinel.write_bytes(b'parent-data')
        with patch('numpy.memmap',side_effect=MemoryError('fixture')):
            with self.assertRaises(MemoryError):
                with scratch_arrays(a,'tiny') as allocate:allocate('labels',np.uint8,4)
        self.assertEqual(sentinel.read_bytes(),b'parent-data');self.assertEqual(list(a.evaluation_scratch.iterdir()),[sentinel])
    def test_review_excludes_scratch_and_model_payload(self):
        a=self.args();st.write_json(a.run_root/'status.json',dict(status='pending'))
        a.cache_root=a.run_root/'custom_big_arrays'
        for path in (a.run_root/'pixel_scratch/leak.bin',a.run_root/'models/final.pt',a.run_root/'cache/secret.json',a.cache_root/'synthetic_rgb/image.png'):
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'tiny')
        result=runner.pack(a)
        with tarfile.open(result['archive']) as f:names=f.getnames()
        self.assertEqual(names,[a.run_root.name+'/status.json'])
    def test_pixel_intermediate_result_seal_and_range(self):
        from DINOv3.MADEqual.aebad_s_reference_residual_segmentation.evaluation import original
        counts=dict(pixels=4,positive_pixels=2,negative_pixels=2)
        result=dict(counts,AUROC=.7,AP=.6,valid=True,AUROC_valid=True,AP_valid=True,status='exact_all_pixels_cpu_tie_grouped')
        dest=self.root/'pixels.json';labels=np.array([0,1,0,1],np.uint8);scores=np.array([0,.5,.2,1],np.float32)
        actual=original.exact_pixel_result(dest,'id','maps',labels,scores,domain='background',branch=p.PRIMARY,prediction_sha='p',counts=counts)
        with patch.object(original.pixels,'exact_binary',side_effect=AssertionError('already sealed')):
            reused=original.exact_pixel_result(dest,'id','maps',labels,scores,domain='background',branch=p.PRIMARY,prediction_sha='p',counts=counts)
        self.assertEqual(actual,reused)
        changed=st.read_json(dest);changed['result']['AUROC']=1.5;st.write_json(dest,changed)
        with self.assertRaisesRegex(ValueError,'modified/incompatible'):original.exact_pixel_result(dest,'id','maps',labels,scores,domain='background',branch=p.PRIMARY,prediction_sha='p',counts=counts)
        with self.assertRaisesRegex(ValueError,'invalid exact pixel'):original.validate_exact_pixels(dict(result,AP=float('nan')),counts)
    def test_cached_formal_NN_preserves_support_and_bank_identity(self):
        from DINOv3.MADEqual.aebad_s_reference_residual_segmentation import inference
        a=self.args();c=catalog();row=dict(c['supports']['background'][0],relative_path='test/fake.png')
        st.sealed_json(a.run_root/'catalog.json',dict(catalog=c));st.sealed_json(a.run_root/'inputs.json',dict(identity='tiny'))
        class Refs:
            def tensors(self,name,origins,layer):
                assert name=='formal' and origins==[r['relative_path'] for r in c['supports']['background']]
                return np.zeros((2,3),np.float32),np.ones((8,3),np.float32),dict(source_seal='formal',target_seals=['k8'],target_offsets=list(range(9)),target_origins=origins)
        q={l:np.full((4,3),2.,np.float32) for l in p.LAYERS};geometry=dict(grid=[2,2],positions=np.array([2,0,3,1]))
        def ex(q,b,device):return np.ones(4,np.float32),np.zeros(4,np.int64),dict(audit='fixture')
        def target(q,b,offsets,device):return *ex(q,b,device)[:2],dict(support_audits=[],per_support_distances=np.ones((8,4),np.float32),per_support_indices=np.zeros((8,4),np.int64))
        path=self.root/'formal_NN'
        with patch.object(inference,'exact_with_audit',side_effect=ex),patch.object(inference,'target_nearest',side_effect=target):
            x,r=inference.context(a,row,q,geometry,Refs(),cache_path=path)
        with patch.object(inference,'exact_with_audit',side_effect=AssertionError('must reuse')):
            xx,rr=inference.context(a,row,q,geometry,Refs(),cache_path=path)
        self.assertEqual(r['seal'],rr['seal']);np.testing.assert_array_equal(xx['ds7'],x['ds7'])
        # NN search is substituted ONLY in this cache plumbing test.
        with self.assertRaisesRegex(ValueError,'not compatible'):
            inference.context(a,row,q,dict(grid=[2,2],positions=np.arange(4)),Refs(),cache_path=path)
    def test_hot_sampling_layer_exact_bytes_and_owned_cleanup(self):
        from DINOv3.MADEqual.aebad_s_reference_residual_segmentation.retrieval import sampling_matrix
        matrix=np.arange(18,dtype=np.float32).reshape(6,3);path=self.root/'HDD.f32';matrix.tofile(path)
        spool=SimpleNamespace(path=path,rows=6,columns=3);ssd=self.root/'SSD';ssd.mkdir();keep=ssd/'unowned.bin';keep.write_bytes(b'keep')
        with self.assertRaisesRegex(RuntimeError,'fixture'):
            with sampling_matrix(spool,ssd,'A',7) as staged:
                self.assertEqual(Path(staged.filename).read_bytes(),path.read_bytes());np.testing.assert_array_equal(staged,matrix)
                raise RuntimeError('fixture')
        self.assertEqual(list(ssd.iterdir()),[keep]);self.assertEqual(keep.read_bytes(),b'keep')
    def test_old_SEGMENT_reuse_requires_more_than_matching_UNION(self):
        from DINOv3.MADEqual.aebad_s_reference_residual_segmentation import inference
        from DINOv3.MADEqual.aebad_s_target_local_segment import geometry as original_geometry
        a=self.args();c=catalog();row=dict(domain='background',sample_id='test',relative_path='test/example')
        st.sealed_json(a.run_root/'catalog.json',dict(catalog=c))
        st.sealed_json(a.run_root/'input_distance_mad.json',dict(patch=dict(medians=[0]*4,scales=[1]*4)))
        parent=self.root/'parent';oldcache=self.root/'parent_cache';g=dict(grid=[2,2],positions=np.arange(4,dtype=np.int64))
        st.sealed_json(a.run_root/'baseline_parent.json',dict(run=str(parent),cache=str(oldcache)))
        st.put(oldcache/'predictions/background/test',{p.U:np.full((4,4),.2,np.float32),p.S:np.full((4,4),.15,np.float32),
            'positions':g['positions']},dict(sample=row,grid=[2,2]))
        nn={}
        for l in p.LAYERS:
            nn.update({f'ds{l}':np.full(4,.2,np.float32),f'dt{l}':np.full(4,.2,np.float32),
                f'perd{l}':np.full((8,4),.2,np.float32),f'peri{l}':np.repeat(np.arange(8)[:,None],4,1)})
        refs=SimpleNamespace(target=lambda origins,l:(np.zeros((8,1280),np.float32),[],list(range(9))))
        with patch.object(original_geometry,'geometry_layer',return_value=({'K8_SEGMENT':np.full(4,.1,np.float32)},{})):
            z,origin=inference.baseline(a,row,nn,g,refs)
        self.assertEqual(origin['kind'],'recomputed_current_common_bank');np.testing.assert_array_equal(z[p.S],np.full((4,4),.1,np.float32))


class ParentMAD(Temporary):
    """Real prepare/bundle path; only catalogue admission and calibration are fixtures."""
    def fixture(self):
        from DINOv3.MADEqual.aebad_s_reference_residual_segmentation import runtime
        a=self.args();parent=self.root/'parent';cache=self.root/'parent_cache';a.recent_run=[parent]
        provider=dict(kind='correspondence',run=parent,cache=cache,identity='parent')
        arrays={f'L{l}':np.random.default_rng(l).normal(size=(3,1280)).astype(np.float32) for l in p.LAYERS}
        record=st.put(cache/'source_original',arrays,dict(identity='parent',kind='original_source'))
        st.sealed_json(parent/'prepared.json',dict(source_seal=record['seal']))
        c=dict(source=[],supports={},data_root='unused')
        state=dict(patch=dict(medians=[0.1]*4,scales=[0.2]*4),image=dict(fixture='image_MAD'))
        config=dict(runtime.CONFIG,bank_rows=3,foreground=dict(runtime.CONFIG['foreground'],bank_stride=1,iterations=1))
        return runtime,a,provider,c,state,config

    def prepare(self,fixture,calibrator):
        from DINOv3.MADEqual.aebad_s_support_witness_segment_p0 import fresh_inputs as fi
        runtime,a,provider,c,_,config=fixture
        # No images/GT/backbone in this fixture; real Source bundle import,
        # sealed parent read, fallback dispatch and current state commit run.
        with patch.object(fi,'prepare_data',return_value=c),patch.object(fi,'Data'),\
                patch.object(runtime,'extraction_record',return_value={}),patch.object(ip,'freeze'),\
                patch.object(ip,'identity',return_value='tiny'),patch.object(ip,'recent',return_value=provider),\
                patch.object(runtime,'compatible'),patch.object(runtime,'CONFIG',config),\
                patch.object(runtime,'fit_input_mad',side_effect=calibrator) as fit,\
                patch.object(runtime,'host',side_effect=AssertionError('no model required')):
            result=runtime.prepare(a)
        return result,fit

    def test_missing_parent_MAD_calibrates_current_common_bank(self):
        fixture=self.fixture();_,a,provider,c,state,_=fixture
        before=st.sha(provider['cache']/'source_original/L7.npy')
        def calibrate(args,catalogue):
            self.assertIs(args,a);self.assertIs(catalogue,c)
            bank,record=st.get(a.cache_root/'source_original')
            self.assertEqual(record['metadata']['origin'],str(provider['cache']/'source_original'))
            np.testing.assert_array_equal(bank['L7'],st.get(provider['cache']/'source_original')[0]['L7'])
            return dict(state,scope='original_Source_only_distances',origin='current_common_original_bank')
        result,fit=self.prepare(fixture,calibrate)
        fit.assert_called_once_with(a,c)
        self.assertEqual(result['normal_forward_calls'],0)
        self.assertEqual(st.sealed_json(a.run_root/'input_distance_mad.json')['origin'],'current_common_original_bank')
        self.assertFalse((a.run_root/'baseline_parent.json').exists())
        self.assertFalse((provider['run']/'calibration/source/UNION_K8.json').exists())
        self.assertEqual(st.sha(provider['cache']/'source_original/L7.npy'),before)

    def test_missing_parent_MAD_resume_keeps_current_calibration(self):
        fixture=self.fixture();_,a,_,_,state,_=fixture
        self.prepare(fixture,lambda *_:dict(state,origin='current_common_original_bank'))
        before=st.sha(a.run_root/'input_distance_mad.json')
        _,fit=self.prepare(fixture,AssertionError('committed current MAD must be reused'))
        fit.assert_not_called();self.assertEqual(st.sha(a.run_root/'input_distance_mad.json'),before)
        self.assertFalse((a.run_root/'baseline_parent.json').exists())

    def test_valid_parent_MAD_reused_without_calibration(self):
        fixture=self.fixture();_,a,provider,_,state,_=fixture
        path=provider['run']/'calibration/source/UNION_K8.json';old=st.sealed_json(path,state);before=st.sha(path)
        _,fit=self.prepare(fixture,AssertionError('valid parent MAD must be reused'))
        fit.assert_not_called();saved=st.sealed_json(a.run_root/'input_distance_mad.json')
        self.assertEqual(saved['patch'],old['patch']);self.assertEqual(saved['image'],old['image'])
        self.assertEqual(saved['old_state_seal'],old['seal'])
        self.assertEqual(st.sealed_json(a.run_root/'baseline_parent.json')['state'],old)
        self.assertEqual(st.sha(path),before)

    def test_modified_parent_MAD_rejected_without_fallback(self):
        fixture=self.fixture();runtime,a,provider,_,state,_=fixture
        path=provider['run']/'calibration/source/UNION_K8.json';old=st.sealed_json(path,state)
        old['patch']['medians'][0]=9.;st.write_json(path,old);before=st.sha(path)
        fit=Mock(side_effect=AssertionError('invalid parent must not recalibrate'))
        with self.assertRaisesRegex(ValueError,'modified record'):
            self.prepare(fixture,fit)
        fit.assert_not_called()
        self.assertFalse((a.run_root/'input_distance_mad.json').exists())
        self.assertFalse((a.run_root/'baseline_parent.json').exists())
        self.assertEqual(st.sha(path),before)


class CPU(Temporary):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1);cls.torch=torch
    def test_residual_distance_exact_and_single_divisor(self):
        from DINOv3.MADEqual.aebad_s_reference_residual_segmentation.retrieval import exact_with_audit,target_nearest
        rng=np.random.default_rng(3);q=rng.normal(size=(5,16)).astype(np.float32);b=rng.normal(size=(9,16)).astype(np.float32)
        d,i,audit=exact_with_audit(q,b,'cpu');direct=np.linalg.norm(q-b[i],axis=1)/1280
        np.testing.assert_allclose(d,direct,atol=1e-8,rtol=1e-6);self.assertEqual(audit['analytic_bound_failed'],0)
        d2,i2,_=target_nearest(q,b,[0,4,9],'cpu');np.testing.assert_allclose(d2,d,atol=1e-8);np.testing.assert_array_equal(i2,i)
    def test_reference_input_vectors_and_distance_units(self):
        from DINOv3.MADEqual.aebad_s_reference_residual_segmentation.model import inputs
        t=self.torch;q=t.full((2,3,3),4.);ref=t.full_like(q,2.)
        x=inputs(q,t.tensor([1.,1.]),2.,source=ref,target=ref+1,distances=t.full((2,3,3),.5),patch=(.1,.2))
        t.testing.assert_close(x[:2],t.full_like(q,1.5));t.testing.assert_close(x[2:4],t.ones_like(q))
        t.testing.assert_close(x[4:6],t.full_like(q,.5));t.testing.assert_close(x[6:],t.full((3,3,3),2.))
    def test_CPU_checks_do_not_initialize_CUDA(self):self.assertFalse(self.torch.cuda.is_initialized())
    def test_noref_ignores_poison_arguments(self):
        from DINOv3.MADEqual.aebad_s_reference_residual_segmentation.model import inputs
        x=inputs(self.torch.ones(2,3,3),self.torch.zeros(2),1.,source=object(),target=object(),distances=object(),patch=object(),use_reference=False)
        self.assertEqual(x[2:].count_nonzero().item(),0)
    def test_head_disables_TF32_without_disabling_CuDNN(self):
        from DINOv3.MADEqual.aebad_s_reference_residual_segmentation.inference import logits
        t=self.torch;before=(t.backends.cudnn.enabled,t.backends.cudnn.allow_tf32,t.backends.cudnn.benchmark,t.backends.cudnn.deterministic)
        class Net(t.nn.Module):
            def forward(self,values):
                self.assert_flags=(t.backends.cudnn.enabled,t.backends.cudnn.allow_tf32,t.backends.cudnn.benchmark,t.backends.cudnn.deterministic)
                return values[0]
        net=Net();logits(net,[t.ones(1,4,2,2)],np.arange(4))
        self.assertEqual(net.assert_flags,(before[0],False,before[2],before[3]))
        self.assertEqual((t.backends.cudnn.enabled,t.backends.cudnn.allow_tf32,t.backends.cudnn.benchmark,t.backends.cudnn.deterministic),before)
        self.assertFalse(t.cuda.is_initialized())
    def test_full_head_and_halo_native_and_interpolated(self):
        from DINOv3.MADEqual.aebad_s_reference_residual_segmentation.model import Head
        t=self.torch;t.manual_seed(4);net=Head(4,8).eval();q=[t.randn(1,15,19,23) for _ in range(4)]
        whole=net(q);center=[5,6,14,17];y,x,yy,xx=tiles.bounds(center,(19,23));small=net([v[:,:,y:yy,x:xx] for v in q])
        t.testing.assert_close(small[:,:,center[0]-y:center[2]-y,center[1]-x:center[3]-x],whole[:,:,5:14,6:17],atol=2e-6,rtol=2e-5)
        full=t.nn.functional.interpolate(whole,size=(73,89),mode='bilinear',align_corners=False)
        crop,win=tiles.interpolate_center(small,[y,x],(19,23),(73,89),center);a,b,c,d=win
        t.testing.assert_close(crop,full[:,:,a:c,b:d],atol=1e-5,rtol=5e-5)
    def test_balanced_losses_normal_soft_and_per_image(self):
        from DINOv3.MADEqual.aebad_s_reference_residual_segmentation.model import loss
        t=self.torch;x=t.randn(2,4,5,7,requires_grad=True);m=t.zeros(2,1,5,7);m[1,0,2,3]=.02
        value=loss(x,m);average=(loss(x[:1],m[:1])+loss(x[1:],m[1:]))/2
        t.testing.assert_close(value,average);value.backward();self.assertTrue(t.isfinite(x.grad).all());self.assertNotEqual(x.grad[1,0,2,3].item(),0)
    def test_calibration_raw_logit_not_distance_or_sigmoid(self):
        values=np.arange(-30,30,dtype=np.float32).reshape(4,15);state,audit=calibration.fit_logits([values])
        self.assertEqual(len(state['medians']),4);self.assertLess(state['medians'][0],0);self.assertGreater(state['scales'][0],1)
        with self.assertRaisesRegex(ValueError,'near-constant'):calibration.fit_logits([np.ones((4,15),np.float32)])
    def test_ensemble_z_no_second_patch_mad(self):
        source=ast.parse(Path(calibration.__file__).read_text())
        text=ast.get_source_segment(Path(calibration.__file__).read_text(),next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='calibrate'))
        self.assertIn('patch=None',text);self.assertIn('mean_member_z(',text)
        z=np.arange(48,dtype=np.float32).reshape(3,4,4)
        actual=calibration.mean_member_z({p.member('REF_ST',s):v for s,v in zip(p.SEEDS,z)},'REF_ST')
        np.testing.assert_array_equal(actual,sum(z)/3)
    def test_ch1_current_share_one_response(self):
        z=np.arange(4*25,dtype=np.float32).reshape(4,25)/10;state={k:dict(medians=[0]*4,scales=[1]*4) for k in ('peak','local')}
        m,s=calibration.response(z,np.arange(25),(5,5),(19,21),state)
        self.assertEqual(s['CURRENT'],float(m.max()));self.assertEqual(set(s),{'CH1','CURRENT'})
    def test_real_CPU_optimizer_resume_matches_continuous(self):
        from DINOv3.MADEqual.aebad_s_reference_residual_segmentation.model import Head
        from DINOv3.MADEqual.aebad_s_reference_residual_segmentation import training
        t=self.torch
        class Tiny:
            use_reference=True;manifest_seal='tiny-fixed-training-pool'
            def select(self,seed,update,source_queries_only=False):return [update]
            def load(self,choice):
                gen=t.Generator().manual_seed(100+choice)
                values=[t.randn(1,15,6,6,generator=gen) for _ in range(4)];soft=np.zeros((6,6),np.float32);soft[2,2]=1
                return dict(values=values,soft=soft,origin=[0,0],grid=[6,6],shape=[6,6],center=[0,0,6,6])
        aa=self.args('continuous');bb=self.args('resumed')
        for a in (aa,bb):st.sealed_json(a.run_root/'inputs.json',dict(identity='fixture'))
        t.manual_seed(5);init=Head(4,8).state_dict()
        def head():
            n=Head(4,8);n.load_state_dict(init);return n
        data=Tiny();training.job(aa,'source','REF_ST',42,dataset=data,head=head(),total=4)
        training.job(bb,'source','REF_ST',42,dataset=data,head=head(),total=4,stop_after=2)
        training.job(bb,'source','REF_ST',42,dataset=data,head=head(),total=4)
        def load(a):
            path=training.path(a,'source','REF_ST',42);inputs=st.sealed_json(path/'latest.json')['inputs'];return training.load_checkpoint(path,inputs,'cpu')
        left,right=load(aa),load(bb);self.assertEqual(right['updates'],4);self.assertEqual(right['scheduler']['completed_updates'],4)
        for k in left['network']:t.testing.assert_close(left['network'][k],right['network'][k],atol=0,rtol=0)
        before=st.sha(training.path(bb,'source','REF_ST',42)/'latest.json')
        done=training.job(bb,'source','REF_ST',42,dataset=data,head=head(),total=4)
        self.assertEqual(done['status'],'training_reused');self.assertEqual(before,st.sha(training.path(bb,'source','REF_ST',42)/'latest.json'))
        with patch.dict(p.CONFIG,{'source_updates':4}):
            for family in ('REF_ST','REF_SOURCE_QUERY'):training.job(bb,'background',family,42,dataset=data,head=head(),total=2)
        controls=[]
        for family in ('REF_ST','REF_SOURCE_QUERY'):
            dest=training.path(bb,'background',family,42);meta=st.sealed_json(dest/'latest.json')
            self.assertEqual(meta['inputs']['Source_initialization'],st.sealed_json(training.path(bb,'source','REF_ST',42)/'complete.json')['checkpoint_sha256'])
            controls.append(training.load_checkpoint(dest,meta['inputs'],'cpu')['network'])
        for k in controls[0]:t.testing.assert_close(controls[0][k],controls[1][k],atol=0,rtol=0)
        self.assertEqual(before,st.sha(training.path(bb,'source','REF_ST',42)/'latest.json'))
    def test_actual_new_calibration_prediction_and_CPU_metrics_on_constructed_features(self):
        self.integration('full_v1')
    def test_compact_actual_calibration_prediction_and_CPU_metrics(self):
        self.integration('compact_v1')
    def integration(self,profile):
        """Real 1280D Exact, four-layer heads/maps/metrics. Fake RGB host and heads' training only."""
        from DINOv3.MADEqual.aebad_s_reference_residual_segmentation import prediction,evaluation,training,runtime
        from DINOv3.MADEqual.aebad_s_reference_residual_segmentation.model import Head
        from DINOv3.MADEqual.aebad_s_support_witness_segment_p0 import fresh_inputs
        a=self.args();a.profile=profile;a.cpu_workers=1;c=catalog();samples=[];rng=np.random.default_rng(18)
        requested=ip.plan(a);model_calls=[]
        for d in p.DOMAINS:
            for i in range(2):samples.append(dict(domain=d,sample_id=f'test_{i}',relative_path=f'test/{d}/{i}.png',
                image_height=64,image_width=64,image_sha256=d+str(i),label=i))
        c['samples']=samples;st.sealed_json(a.run_root/'catalog.json',dict(catalog=c));st.sealed_json(a.run_root/'inputs.json',dict(identity='toy-integration'))
        st.put(a.cache_root/'source_original',{f'L{l}':rng.normal(size=(32,1280)).astype(np.float32) for l in p.LAYERS},{})
        st.put(a.cache_root/'input_normalization',{**{f'mu{l}':np.zeros(1280,np.float32) for l in p.LAYERS},
            **{f'sigma{l}':np.asarray(1.,np.float32) for l in p.LAYERS}}, {})
        image_state={k:dict(medians=[0]*4,scales=[1]*4) for k in ('peak','local')}
        st.sealed_json(a.run_root/'input_distance_mad.json',dict(patch=dict(medians=[.03]*4,scales=[.02]*4),image=image_state,scope='fixture_Source_distances'))
        def features(row):
            local=st.rng('raw-CPU-fixture',row['relative_path']);positions=local.permutation(16).astype(np.int64)
            return {l:local.normal(size=(16,1280)).astype(np.float32)[positions] for l in p.LAYERS},dict(grid=[4,4],positions=positions)
        for row in [r for r in c['source'] if r['calibration']]+sum(c['supports'].values(),[]):
            q,g=features(row);st.put(ip.normal_path(a,row),{**{f'q{l}':q[l] for l in p.LAYERS},'positions':g['positions']},
                dict(grid=g['grid'],output_shape=[16,16],image=row))
        def trained(args,domain,family,seed):
            self.assertIn(p.member(family,seed),requested['members']);model_calls.append((domain,family,seed))
            self.torch.manual_seed(st.seed('CPU-head-fixture',domain,family,seed)%2**31)
            return Head(1280,8).eval(),dict(checkpoint_sha256=f'{domain}_{family}_{seed}_fixture')
        calls={'GT':0,'forward':0}
        class Data:
            def __init__(self,*_):pass
            def __enter__(self):return self
            def __exit__(self,*_):pass
            def image(self,row):return row
            def gt(self,row):
                calls['GT']+=1;mask=np.zeros((64,64),np.uint8)
                if row['label']:mask[24:30,24:30]=1
                return mask
        def extraction(net,row,device):calls['forward']+=1;return features(row)
        toy_counts={d:[2,1,1] for d in p.DOMAINS}
        with patch.dict(p.CONFIG,{'populations':toy_counts}),patch.object(training,'trained',side_effect=trained),\
                patch.object(runtime,'host',return_value=(object(),'cpu')),patch.object(runtime,'extraction',side_effect=extraction),\
                patch.object(fresh_inputs,'Data',Data),patch.object(evaluation,'figure'):
            calibration.calibrate(a);self.assertEqual(calls['GT'],0)
            result=prediction.predict(a);self.assertEqual(result['query_forward_calls'],6);self.assertEqual(calls['GT'],0)
            seals=[st.sealed_json(prediction.prediction_path(a,s)/'manifest.json')['seal'] for s in samples]
            result=prediction.predict(a);self.assertEqual(result['query_forward_calls'],0);self.assertEqual(calls['forward'],6)
            evaluation.evaluate(a);before=calls['GT'];evaluation.evaluate(a);self.assertEqual(calls['GT'],before)
            status=reporting.summarize(a);self.assertEqual(status['status'],'pending');self.assertEqual(status['metric_rows'],requested['expected_metric_rows'])
            rows=evaluation.read_csv(a.run_root/'metrics.csv');self.assertTrue(all(r['AUPRO']=='' for r in rows))
            self.assertEqual(len(rows),requested['expected_metric_rows']);self.assertEqual(set(r['scoring'] for r in rows),{'CH1','CURRENT'})
            self.assertEqual(len(model_calls),len(p.DOMAINS)*len(requested['members'])*2)
            for d in p.DOMAINS:
                local=evaluation.read_csv(a.run_root/'evaluation'/d/'metrics.csv')
                for b in requested['branches']:
                    pair=[r for r in local if r['branch']==b]
                    self.assertEqual(pair[0]['pixel_AP'],pair[1]['pixel_AP'])
            self.assertEqual(seals,[st.sealed_json(prediction.prediction_path(a,s)/'manifest.json')['seal'] for s in samples])
            self.assertFalse(self.torch.cuda.is_initialized())


class Interface(Temporary):
    def test_resume_committed_preparation_without_rescanning_payloads(self):
        from DINOv3.MADEqual.aebad_s_reference_residual_segmentation import runtime,training_data
        a=self.args();st.sealed_json(a.run_root/'inputs.json',dict(identity='tiny'));st.write_json(a.run_root/'prepared.json',{})
        st.sealed_json(a.run_root/'synthetic_manifest.json',dict(identity='tiny',items=[],invalid=0))
        st.sealed_json(a.run_root/'training_manifest.json',dict(identity='tiny',records=[],status='all_fixed_training_queries_prepared'))
        with patch.object(runner,'validate_runtime_inputs'),patch.object(runtime,'prepare',side_effect=AssertionError('prepare repeated')),\
                patch.object(runtime,'synthesize',side_effect=AssertionError('RGB repeated')),patch.object(training_data,'prepare_training',side_effect=AssertionError('NN repeated')):
            for stage in ('prepare','synthesize','prepare-training'):
                a.stage=stage;self.assertIn('reused',runner.run_stage(a)['status'])
    def test_frozen_CPU_evaluation_does_not_require_backbone_or_banks(self):
        from importlib.metadata import version
        a=self.args();c=catalog();a.stage='evaluate'
        st.sealed_json(a.run_root/'catalog.json',dict(catalog=c))
        known=st.read_json(Path(p.__file__).with_name('legacy_full_v1_compatibility.json'))['code']
        st.sealed_json(a.run_root/'inputs.json',dict(config=p.CONFIG,cache=str(a.cache_root),query_mode=a.query_mode,
            catalog=st.digest(c),code={'DINOv3/MADEqual/aebad_s_reference_residual_segmentation/'+k:v for k,v in known.items()},extraction=dict(software={n:version(n) for n in ('numpy','scipy','Pillow')})))
        runner.validate_runtime_inputs(a)
        self.assertFalse((a.cache_root/'source_original').exists());self.assertFalse(a.dinov3_root.exists())
    def test_preflight_no_torch_no_output(self):
        dest=self.root/'untouched';module='DINOv3.MADEqual.aebad_s_reference_residual_segmentation.runner'
        script=f'import sys; from {module} import main; rc=main(["--run-root",{str(dest)!r},"--no-write"]); assert "torch" not in sys.modules; assert rc in (0,2)'
        env=dict(os.environ,PYTHONUTF8='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        result=subprocess.run([sys.executable,'-B','-c',script],cwd=ip.ROOT,env=env,capture_output=True,text=True,encoding='utf-8')
        self.assertEqual(result.returncode,0,result.stderr);self.assertFalse(dest.exists());self.assertIn('"torch_imported": false',result.stdout)
    def test_prediction_has_no_GT_decode(self):
        from DINOv3.MADEqual.aebad_s_reference_residual_segmentation import prediction
        tree=ast.parse(Path(prediction.__file__).read_text())
        self.assertFalse(any(isinstance(n,ast.Attribute) and n.attr in ('gt','ground_truth') for n in ast.walk(tree)))
    def test_partial_report_keeps_pending(self):
        a=self.args();st.sealed_json(a.run_root/'inputs.json',dict(identity='fixture'))
        result=reporting.summarize(a);self.assertEqual(result['status'],'pending');self.assertTrue(result['pending'])
    def test_no_global_old_branch_mutation(self):
        from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation import evaluation as old
        self.assertEqual(len(old.BRANCHES),6);self.assertEqual(len(p.BRANCHES),14)


if __name__=='__main__':unittest.main()
