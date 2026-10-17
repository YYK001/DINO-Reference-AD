"""Actual NumPy/CPU stages; no real model, CUDA or dataset experiment."""
from __future__ import annotations
import copy
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import numpy as np
from PIL import Image
from DINOv3.MADEqual.robustad_segment_pair_crossdataset_p0 import inputs as ip,calibration as cal,prediction as pr,evaluation as ev,reporting as rep,runner, runtime as rt
from DINOv3.MADEqual.robustad_segment_pair_crossdataset_p0 import rebuild
from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0 import method
from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.verification import historical_score
from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import statistics
from DINOv3.tests.test_aebad_s_segment_pair_crossdomain_p0 import raw_fixture

ROOT=Path(__file__).resolve().parents[2]


class Reconstruction(unittest.TestCase):
    """Real CPU provenance/geometry/validation; GPU builder explicitly mocked."""
    def setUp(self):
        (ROOT/'downloads').mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix='robust_rebuild_',dir=ROOT/'downloads')
        self.root=Path(self.temp.name).resolve()
    def tearDown(self):self.temp.cleanup()

    def context(self):
        catalog=ip.read_json(Path(ip.__file__).with_name('frozen_catalog.json'))
        cat=copy.deepcopy(catalog['categories']['MetalParts'])
        paths={k:self.root/k for k in ('source_root','data','official','dinov3','checkpoint','formal_metalparts','formal_pcb','output_root','rebuilt_source_root')}
        source=paths['source_root']/'categories/robustad/MetalParts'
        count=sum(int(np.prod(runner.estimated_grid(s))) for s in cat['train'] if s['source_index']%8)
        old=dict(sampler='upstream_subsampling_distance_based_fast',sampler_iterations=100,memory_budget_per_layer=100000,
                 sampler_distance_backend='exact_query_chunked_torch_cdist_topk',sampler_distance_matrix_max_bytes=1<<30,
                 layer_audits={str(l):dict(candidate_count=count,memory_entries=100000) for l in ip.LAYERS})
        ip.write_json(source/'memory/memory_provenance.json',dict(official_builder=old))
        return dict(paths=paths,source=source,category='MetalParts',cat=cat,catalog=catalog)

    def test_plan_preserves_all180_order_and_only157_candidate_blocks(self):
        ctx=self.context();p=rebuild.plan(ctx)
        self.assertEqual(len(p['train']),180);self.assertEqual(len(p['candidate_blocks']),157)
        self.assertEqual(p['candidate_rows'],947024)
        self.assertEqual([s['source_index'] for s in p['train']],list(range(180)))
        self.assertTrue(all(s['source_index']%8 for s in p['candidate_blocks']))
        self.assertEqual(p['sampler_iterations'],100)
        self.assertEqual(p['entries_per_layer'],100000)
        self.assertGreater(rebuild.resources(p)['minimum_available_ram_bytes'],p['candidate_rows']*4*1280*4)

    def test_plan_rejects_pre_excluded_or_reordered_training(self):
        ctx=self.context();ctx['cat']['train']=ctx['cat']['train'][1:]
        with self.assertRaisesRegex(ValueError,'split'):rebuild.plan(ctx)
        ctx=self.context();ctx['cat']['train'][1:3]=ctx['cat']['train'][2:0:-1]
        with self.assertRaisesRegex(ValueError,'split'):rebuild.plan(ctx)

    def test_reconstruction_must_not_overlap_original_or_prediction_tree(self):
        ctx=self.context()
        for key in ('source_root','data','output_root'):
            paths=dict(ctx['paths'],rebuilt_source_root=ctx['paths'][key]/'child')
            with self.assertRaisesRegex(ValueError,'overlaps'):rebuild.directory(paths,'MetalParts')
        paths=dict(ctx['paths']);del paths['rebuilt_source_root']
        with self.assertRaisesRegex(ValueError,'explicit'):rebuild.directory(paths,'MetalParts')

    def test_original_record_class_uses_current_paths_without_sort_or_exclusion(self):
        ctx=self.context()
        for s in ctx['cat']['train']:
            s.update(image_height=3,image_width=2)
            path=ctx['paths']['data']/s['relative_path'];path.parent.mkdir(parents=True,exist_ok=True)
            Image.new('RGB',(2,3)).save(path);s['image_sha256']=ip.sha(path)
        records=rebuild.ordered_records(ctx)
        self.assertEqual(len(records),180)
        self.assertEqual([r.path for r in records],[str(ctx['paths']['data']/s['relative_path']) for s in ctx['cat']['train']])
        path=Path(records[-1].path);path.write_bytes(path.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'bytes changed'):rebuild.ordered_records(ctx)

    def test_validation_rejects_wrong_count_repetition_range_dtype_nonfinite(self):
        banks={l:np.ones((3,2),np.float32) for l in ip.LAYERS}
        indices={l:np.array([0,2,5],np.int64) for l in ip.LAYERS}
        rebuild.validate_arrays(banks,indices,6,entries=3,dimension=2)
        for broken in (np.array([0,0,5]),np.array([0,2,6]),np.array([0,2]),np.array([0,2,5],np.int32)):
            with self.assertRaises(ValueError):rebuild.validate_arrays(banks,indices|{7:broken},6,entries=3,dimension=2)
        for broken in (np.ones((2,2),np.float32),np.full((3,2),np.nan,np.float32),np.ones((3,2),np.float64)):
            with self.assertRaises(ValueError):rebuild.validate_arrays(banks|{7:broken},indices,6,entries=3,dimension=2)

    def test_rebuilt_mode_never_adopts_old_distances_even_with_same_bank_size(self):
        ctx=self.context();ctx['paths']['source_raw_root']=self.root/'unavailable'
        ctx['frozen']=dict(bank_size_bytes=2048000974)
        # No old cache discovery or SHA enrollment may run for this bank origin.
        with mock.patch.object(ip,'read_json',side_effect=AssertionError('old cache read')):
            self.assertIsNone(cal.legacy_raw(ctx,ctx['cat']['calibration'][0],{}))

    def test_production_wrapper_passes_all_records_to_original_builder_without_reseed(self):
        from types import SimpleNamespace
        from contextlib import nullcontext
        ctx=self.context();records=list(range(180));events=[]
        class Tensor:
            def detach(self):return self
            def float(self):return self
            def cpu(self):return self
            def numpy(self):return np.ones((3,2),np.float32)
        model=SimpleNamespace(threshold_fraction=8,max_database_size=100000,layers=ip.LAYERS)
        build=SimpleNamespace(banks={l:Tensor() for l in ip.LAYERS},
             selected_candidate_indices={l:np.arange(3,dtype=np.int64) for l in ip.LAYERS},metadata={})
        def load(*a):events.append('load_original_seeded_model');return model,'cuda:0'
        def original(mm,rr,**kwargs):
            events.append('original_builder');self.assertIs(mm,model);self.assertIs(rr,records)
            self.assertEqual(kwargs,dict(device='cuda:0',scratch_dir=self.root/'scratch'));return build
        cuda=SimpleNamespace(max_memory_allocated=lambda d:1,max_memory_reserved=lambda d:2,
            get_device_name=lambda d:'fixture',empty_cache=lambda:None)
        torch=SimpleNamespace(inference_mode=nullcontext,cuda=cuda,version=SimpleNamespace(cuda='fixture'))
        variants=SimpleNamespace(choose_candidate_storage_backend=lambda *a,**k:{'backend':'ram'})
        with (mock.patch.dict(sys.modules,{'torch':torch,'DINOv3.relation_reliability.superadd_host':SimpleNamespace(build_exact_official_memory=original),
                                         'external_baselines.superadd_external.memory_variants':variants}),
              mock.patch('external_baselines.superadd_external.memory_variants',variants,create=True),
              mock.patch.object(rt,'load_model',side_effect=load)):
            arrays,ix,_,_=rebuild.original_build(ctx,records,'cuda:0',20,self.root/'scratch')
        self.assertEqual(events,['load_original_seeded_model','original_builder'])
        np.testing.assert_array_equal(ix[7],np.arange(3));self.assertEqual(arrays[7].shape,(3,2))

    def test_original_auto_decision_preserved_and_unsafe_disk_fallback_rejected(self):
        for backend in ('ram','disk'):
            decision={'backend':backend,'reason':'original_choice'}
            chooser=mock.Mock(return_value=decision)
            with mock.patch.object(rebuild.shutil,'disk_usage',return_value=type('Disk',(),{'free':1})()):
                if backend=='disk':
                    with self.assertRaisesRegex(MemoryError,'scratch'):rebuild.storage_admission(chooser,self.root,100)
                else:self.assertIs(rebuild.storage_admission(chooser,self.root,100,headroom_bytes=123),decision)
            self.assertEqual(chooser.call_count,1)

    def mock_build(self,ctx):
        banks={l:np.arange(6,dtype=np.float32).reshape(3,2) for l in ip.LAYERS}
        indices={l:np.array([0,2,5],np.int64) for l in ip.LAYERS}
        source=ctx['source']/'memory/anchor_sidecars.npz'
        np.savez(source,**{f'physical_{l}':a for l,a in indices.items()})
        return banks,indices,dict(layer_audits={str(l):dict(candidate_count=947024) for l in ip.LAYERS}),{}

    def test_build_orchestration_records_new_identity_resume_and_tamper(self):
        ctx=self.context();value=self.mock_build(ctx);validate=rebuild.validate_arrays
        with (mock.patch.object(rebuild,'ordered_records',return_value=list(range(180))) as records,
              mock.patch.object(rebuild,'original_build',return_value=value) as build,
              mock.patch.object(ip,'available_ram',return_value=120<<30),
              mock.patch.object(rebuild.shutil,'disk_usage',return_value=type('Disk',(),{'free':100<<30})()),
              mock.patch.object(rebuild,'validate_arrays',side_effect=lambda a,b,c:validate(a,b,c,entries=3,dimension=2))):
            result=rebuild.run(ctx,'cuda:0')
            self.assertEqual(build.call_count,1);self.assertEqual(build.call_args.args[1],list(range(180)))
            self.assertEqual(result['status'],'bank_reconstructed')
            self.assertFalse(result['historical_bank_byte_identity_proven'])
            self.assertTrue(all(v['exact_order_equal'] for v in result['historical_selected_index_audit'].values()))
            self.assertEqual(result,rebuild.run(ctx,'cuda:0'));self.assertEqual(build.call_count,1)
        path=rebuild.directory(ctx['paths'],'MetalParts')/'memory/official_banks.npz'
        path.write_bytes(path.read_bytes()+b'modified')
        with self.assertRaisesRegex(ValueError,'changed'):rebuild.run(ctx,'cuda:0')

    def test_partial_or_failed_rebuild_preserved_not_approved_on_retry(self):
        ctx=self.context();self.mock_build(ctx)
        with (mock.patch.object(rebuild,'ordered_records',return_value=list(range(180))),
              mock.patch.object(rebuild,'original_build',side_effect=RuntimeError('synthetic failure')),
              mock.patch.object(ip,'available_ram',return_value=120<<30),
              mock.patch.object(rebuild.shutil,'disk_usage',return_value=type('Disk',(),{'free':100<<30})())):
            with self.assertRaisesRegex(RuntimeError,'synthetic'):rebuild.run(ctx,'cuda:0')
        root=rebuild.directory(ctx['paths'],'MetalParts')
        before=ip.sha(root/'rebuild_status.json')
        self.assertEqual(ip.read_json(root/'rebuild_failure.json')['status'],'rebuild_failed')
        with self.assertRaisesRegex(ValueError,'incomplete'):rebuild.run(ctx,'cuda:0')
        self.assertEqual(before,ip.sha(root/'rebuild_status.json'))

    def test_context_enrolls_reconstruction_provenance_and_rejects_changed_bank(self):
        ctx=self.context();value=self.mock_build(ctx);validate=rebuild.validate_arrays
        state=ctx['source']/'calibration/calibration_state.npz';state.parent.mkdir()
        np.savez(state,mad_medians=np.zeros(4,np.float32),mad_scales=np.ones(4,np.float32))
        ctx['cat']['source_sha256']={'calibration/calibration_state.npz':ip.sha(state)}
        ctx['cat']['formal_sha256']={};ctx['catalog']['shared_source_files']={}
        ctx['catalog']['categories']['MetalParts']=ctx['cat']
        read=ip.read_json;sha=ip.sha
        def catalog(p):return ctx['catalog'] if Path(p).name=='frozen_catalog.json' else read(p)
        def checksum(p):return ip.CATALOG_SHA if Path(p).name=='frozen_catalog.json' else sha(p)
        with mock.patch.object(ip,'read_json',side_effect=catalog),mock.patch.object(ip,'sha',side_effect=checksum):
            pending=ip.context(ctx['paths'],'MetalParts')
            self.assertEqual(pending['frozen']['bank_provenance']['bank_origin'],'explicit_reconstruction_pending')
            with self.assertRaisesRegex(FileNotFoundError,'must complete'):ip.context(ctx['paths'],'MetalParts',enroll=True)
            with (mock.patch.object(rebuild,'ordered_records',return_value=list(range(180))),
                  mock.patch.object(rebuild,'original_build',return_value=value),
                  mock.patch.object(ip,'available_ram',return_value=120<<30),
                  mock.patch.object(rebuild.shutil,'disk_usage',return_value=type('Disk',(),{'free':100<<30})()),
                  mock.patch.object(rebuild,'validate_arrays',side_effect=lambda a,b,c:validate(a,b,c,entries=3,dimension=2))):
                rebuild.run(ctx,'cuda:0')
            enrolled=ip.context(ctx['paths'],'MetalParts',enroll=True)
            self.assertEqual(enrolled['frozen']['bank_provenance']['bank_origin'],'explicit_reconstruction_original_builder')
            self.assertEqual(enrolled['frozen']['bank_sha256'],sha(enrolled['bank']))
            before=sha(enrolled['root']/'inputs.json')
            enrolled['bank'].write_bytes(enrolled['bank'].read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'changed'):ip.context(ctx['paths'],'MetalParts',enroll=True)
            self.assertEqual(before,sha(enrolled['root']/'inputs.json'))

    def test_rebuild_dispatch_does_not_enroll_missing_bank_or_start_other_stages(self):
        ctx=self.context()
        with mock.patch.object(ip,'context',return_value=ctx) as context,mock.patch.object(rebuild,'run') as run:
            self.assertEqual(runner.main(['rebuild-source','--rebuilt-source-root',str(self.root/'rebuilt'),'--category','MetalParts']),0)
        self.assertEqual(context.call_args.kwargs,{})
        self.assertEqual(run.call_count,1)

    def test_model_and_builder_imports_stay_out_of_default_preflight(self):
        script='''
import importlib.abc,sys,json
from unittest.mock import patch
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch','torchvision'):raise RuntimeError('unexpected Torch import')
sys.meta_path.insert(0,Deny())
from DINOv3.MADEqual.robustad_segment_pair_crossdataset_p0 import runner,inputs
from importlib.metadata import PackageNotFoundError
def version(n):
 if n in ('timm','omegaconf'):raise PackageNotFoundError(n)
 return 'fixture'
with patch.object(inputs,'context',side_effect=FileNotFoundError('fixture absent metadata')),patch('importlib.metadata.version',side_effect=version):
 assert runner.main(['--rebuilt-source-root','G:/fixture_rebuild','--no-write'])==2
assert 'torch' not in sys.modules
'''
        result=subprocess.run([sys.executable,'-B','-c',script],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        data=json.loads(result.stdout)
        self.assertNotIn('timm',data['missing_dependencies']);self.assertFalse(data['torch_imported'])


class Cases(unittest.TestCase):
    def setUp(self):
        (ROOT/'downloads').mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix='robust_segment_pair_',dir=ROOT/'downloads')
        self.root=Path(self.temp.name).resolve();self.assertTrue(self.root.is_relative_to((ROOT/'downloads').resolve()))
    def tearDown(self):self.temp.cleanup()

    def fixture(self,category='MetalParts',historical_distance_audit=False):
        root=self.root/category;root.mkdir();source=self.root/(category+'_original');source.mkdir();(source/'calibration').mkdir()
        data=self.root/'data';data.mkdir(exist_ok=True)
        ctx=dict(root=root,category=category,source=source,paths=dict(data=data,output_root=self.root,source_root=self.root),
            identity='fixture_'+category,frozen=dict(bank_sha256='original_'+category,bank_size_bytes=1),extraction_identity='original_features_'+category)
        ctx['paths']['formal_'+category.lower()]=self.root/'missing_formal_fixture'
        def sample(rel,label=0,shift='lighting'):
            p=data/rel;p.parent.mkdir(parents=True,exist_ok=True);Image.new('RGB',(64,48),(10,20,30)).save(p)
            s=dict(category=category,domain=shift,shift=shift,sample_id=rel,relative_path=rel,label=label,image_sha256=ip.sha(p),image_height=48,image_width=64)
            if label:
                mask=np.zeros((48,64),np.uint8);mask[10:22,20:33]=255;p=p.with_name(p.stem+'_mask.png');Image.fromarray(mask).save(p)
                s.update(mask_path=p.relative_to(data).as_posix(),gt_sha256=ip.sha(p))
            return s
        rng=np.random.default_rng(42 if category=='MetalParts' else 43)
        values=[rng.uniform(.01,.03,(4,12)).astype(np.float32) for _ in range(23)]
        pooled=np.concatenate(values,axis=1);med=np.median(pooled,axis=1).astype(np.float32)
        scale=np.asarray([max(1.4826*float(np.median(np.abs(a-float(m)))),1e-8) for a,m in zip(pooled,med)],np.float32)
        ctx.update(medians=med,scales=scale)
        p=source/'calibration/calibration_state.npz';np.savez(p,mad_medians=med,mad_scales=scale,**{f'ecdf_{l}':np.sort(pooled[j]) for j,l in enumerate(ip.LAYERS)})
        calibration=[]
        for i in range(23):
            s=sample(f'{category}/train/cal_{i:03d}.png',shift='source_train');s.update(source_role='calibration',source_index=i*8,calibration_index=i+1);calibration.append(s)
        source_eval=[sample(f'{category}/source/test_{i}.png',i,'source') for i in (0,1)]
        raw,banks=raw_fixture((3,4));support=[]
        for i in range(8):
            s=sample(f'{category}/lighting/support_{i}.png');s.update(support_index=i+1,grid=[1,len(banks[i])]);support.append(s)
        targets=[sample(f'{category}/lighting/test_{i}.png',i) for i in (0,1)]
        for s in targets:
            s['source_score']=statistics(raw['source_distance'],raw['positions'],(3,4),(12,16),med,scale)[0]
            s['union_score']=statistics(np.minimum(raw['source_distance'],raw['target_distance'].min(0)),raw['positions'],(3,4),(12,16),med,scale)[0]
        ctx['cat']=dict(calibration=calibration,source_evaluation=source_eval,shifts={'lighting':dict(samples=targets,supports=support)},
            source_sha256={'calibration/calibration_state.npz':ip.sha(p)})
        ctx['catalog']={'categories':{category:ctx['cat']}}
        state=ip.ledger(ctx)
        for s,a in zip(calibration,values):
            ip.save_cache(ctx,state,'calibration',s,dict(positions=np.arange(12,dtype=np.int64),source_distance=a,source_index=np.zeros((4,12),np.int64)),dict(grid=[3,4],output_shape=[12,16]))
        for s in source_eval:ip.save_cache(ctx,state,'source',s,dict(positions=raw['positions'],source_distance=raw['source_distance'],source_index=raw['source_index']),dict(grid=[3,4],output_shape=[12,16]))
        for s in targets:
            meta=dict(grid=[3,4],output_shape=[12,16],supports=support)
            if historical_distance_audit:meta['historical_Source_distance_audit']=dict(sample_id=s['sample_id'],check_kind='historical_Source_distance',
                verified=False,max_abs_error=2e-6,atol=1e-7,rtol=1e-6,acceptance_role='historical_distance_audit_only')
            ip.save_cache(ctx,state,'target',s,raw,meta)
        return ctx,raw

    def test_calibration_reuses_23_raw_without_host_and_is_category_specific(self):
        a,_=self.fixture();b,_=self.fixture('PCB')
        with mock.patch.object(rt,'load_host',side_effect=AssertionError('model loaded')):
            sa=cal.recover(a,'cuda:0');sb=cal.recover(b,'cuda:0')
        self.assertNotEqual(sa['state_sha256'],sb['state_sha256']);self.assertEqual(len(sa['calibration_ids']),23)
        self.assertEqual(ip.read_json(a['root']/'calibration_status.json')['source_forward_calls'],0)
        self.assertFalse(ip.read_json(a['root']/'calibration_status.json')['patch_MAD_refitted'])
        self.assertEqual(sa,cal.recover(a,'cuda:0'))

    def test_calibration_wrong_category_or_changed_state_rejected(self):
        ctx,_=self.fixture();cal.recover(ctx,'cuda:0');path=ctx['root']/'image_calibration_state.json'
        state=ip.read_json(path);state['category']='PCB';ip.write_json(path,state)
        with self.assertRaises(ValueError):cal.load_state(ctx)

    def test_original_patch_mad_not_refit_to_accept_changed_recovery(self):
        ctx,_=self.fixture();ctx['medians']=ctx['medians']+1
        before=ip.sha(ctx['source']/'calibration/calibration_state.npz')
        with self.assertRaisesRegex(ValueError,'patch MAD'):cal.recover(ctx,'cuda:0')
        self.assertEqual(ip.sha(ctx['source']/'calibration/calibration_state.npz'),before)
        self.assertFalse((ctx['root']/'image_calibration_state.json').exists())

    def test_changed_recorded_raw_is_rejected_before_recovery(self):
        ctx,_=self.fixture();state=ip.ledger(ctx);path=ctx['root']/state['records'][0]['file'];path.write_bytes(path.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'changed'):cal.recover(ctx,'cuda:0')

    def test_orphan_raw_is_not_enrolled(self):
        ctx,_=self.fixture();np.savez(ctx['root']/'source_calibration_raw/orphan.npz',a=np.zeros(1))
        with self.assertRaisesRegex(ValueError,'orphan'):ip.ledger(ctx)

    def test_cpu_prediction_evaluation_source_and_target_end_to_end(self):
        ctx,_=self.fixture();cal.recover(ctx,'cuda:0')
        with mock.patch.object(ip,'gt',side_effect=AssertionError('GT entered prediction')):
            for group in ('source','lighting'):pr.run(ctx,group,2)
        for group in ('source','lighting'):
            out=ev.evaluate(ctx,group,'v1',2,1<<30);ev.verified(ctx,group,'v1','cpu')
            self.assertEqual(ip.read_json(out/'status.json')['status'],'cpu_complete_aupro_pending')
            rows=ev.numeric_rows(out/'metrics.csv');self.assertEqual(len(rows),1 if group=='source' else 3)
            self.assertTrue(all(r['AUPRO'] is None for r in rows))
        target=ip.read_csv(pr.verify(ctx,'lighting')[0]/'scores.csv')
        self.assertEqual(len(target),8);self.assertEqual(sum(r['branch']==pr.SOURCE for r in target),2)
        self.assertEqual(len(ip.read_csv(ev.directory(ctx,'lighting','v1','cpu')/'cases.csv')),2)

    def test_incomplete_predictions_cannot_enter_GT(self):
        ctx,_=self.fixture();cal.recover(ctx,'cuda:0');pr.run(ctx,'lighting',1)
        root,m,_=pr.verify(ctx,'lighting');(root/m['records'][0]['file']).unlink()
        with mock.patch.object(ip,'gt',side_effect=AssertionError('GT entered')):
            with self.assertRaises((OSError,ValueError)):ev.evaluate(ctx,'lighting','v1',1,1<<30)

    def test_prediction_resume_without_resampling_or_reapproving_hashes(self):
        ctx,_=self.fixture();cal.recover(ctx,'cuda:0');original=pr.score;calls=[0]
        def stop_second(*args,**kwargs):
            calls[0]+=1
            if calls[0]==2:raise RuntimeError('simulated interruption')
            return original(*args,**kwargs)
        with mock.patch.object(pr,'score',side_effect=stop_second):
            with self.assertRaises(RuntimeError):pr.run(ctx,'lighting',1)
        progress=ip.read_json(ctx['root']/'predictions/lighting/progress.json');self.assertEqual(len(progress['records']),1)
        first=progress['records'][0];pr.run(ctx,'lighting',1);root,m,_=pr.verify(ctx,'lighting')
        self.assertEqual(first,m['records'][0]);self.assertEqual(len(m['records']),2)

    def test_internal_score_drift_still_blocks_before_GT(self):
        ctx,_=self.fixture();cal.recover(ctx,'cuda:0');pr.run(ctx,'lighting',1);original=ev.score_maps
        def changed(*a,**kw):
            current,ch1,response,error=original(*a,**kw);return current,ch1,response+2e-5,error
        with mock.patch.object(ev,'score_maps',side_effect=changed),mock.patch.object(ip,'gt',side_effect=AssertionError('GT entered')):
            with self.assertRaisesRegex(ValueError,'internal'):ev.evaluate(ctx,'lighting','v1',1,1<<30)

    def test_historical_distance_audit_is_retained_separately_from_admission(self):
        ctx,_=self.fixture(historical_distance_audit=True);cal.recover(ctx,'cuda:0')
        rt.capture(ctx,['lighting'],'target','cuda:0');pr.run(ctx,'lighting',1)
        root,_,_=pr.verify(ctx,'lighting');rows=ip.read_csv(root/'numerical_verification.csv')
        historic=[r for r in rows if r['check_kind']=='historical_Source_distance']
        self.assertEqual(len(historic),2);self.assertTrue(all(r['verified']=='False' for r in historic))
        self.assertTrue(all(r['verified']=='True' for r in rows if r['check_kind'] in ('SOURCE_CURRENT','UNION_CURRENT')))

    def test_source_ARD_missing_stays_pending_and_does_not_use_current(self):
        row=dict(category='MetalParts',shift='lighting',branch=ip.BRANCHES[0],**{k:.8 for k in ev.METRICS})
        a=rep.ard_rows([row],[]);self.assertTrue(all(r['ARD'] is None for r in a))
        source=dict(category='MetalParts',shift='source',branch=pr.SOURCE,**{k:.9 for k in ev.METRICS})
        a=rep.ard_rows([row],[source]);self.assertTrue(all(abs(r['ARD']+.1)<1e-12 for r in a))
        row.update({k:.95 for k in ev.METRICS});self.assertTrue(all(r['ARD']==0 for r in rep.ard_rows([row],[source])))

    def test_complete_capture_reuse_never_loads_model_or_torch(self):
        ctx,_=self.fixture()
        with mock.patch.object(rt,'load_host',side_effect=AssertionError('model loaded')):
            state=rt.capture(ctx,['lighting'],'both','cuda:0')
        self.assertEqual(state['status'],'selected_capture_complete')
        self.assertTrue(all(v==0 for v in state['last_cost']['counts'].values()))
        path=ctx['root']/state['records'][-1]['file'];path.write_bytes(path.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'changed'):rt.capture(ctx,['lighting'],'both','cuda:0')

    def test_missing_original_bank_blocks_before_torch_or_builder(self):
        ctx,_=self.fixture();ctx['bank']=self.root/'absent_original_banks.npz'
        with self.assertRaisesRegex(FileNotFoundError,'no implicit reconstruction'):rt.load_host(ctx,'cuda:0')

    def test_complete_capture_ignores_unavailable_external_feature_root(self):
        ctx,_=self.fixture();ctx['paths']['feature_root']=self.root/'offline_features'
        state=ip.ledger(ctx);state['external_manifests']={'features':'previously_enrolled_hash'}
        before=copy.deepcopy(state['records']);ip.write_json(ctx['root']/'cache_status.json',state)
        with mock.patch.object(rt,'feature_manifest',wraps=rt.feature_manifest) as manifest, mock.patch.object(rt,'load_host',side_effect=AssertionError('model loaded')):
            result=rt.capture(ctx,['lighting'],'both','cuda:0')
        manifest.assert_not_called();self.assertEqual(result['status'],'selected_capture_complete')
        self.assertEqual(result['records'],before)
        self.assertEqual(result['external_manifests']['features'],'previously_enrolled_hash')
        self.assertTrue(all(v==0 for v in result['last_cost']['counts'].values()))

    def test_source_only_ignores_feature_root_even_with_missing_Target_caches(self):
        ctx,_=self.fixture();ctx['paths']['feature_root']=self.root/'offline_features';state=ip.ledger(ctx)
        for record in state['records']:
            if record['kind']=='target':(ctx['root']/record['file']).unlink()
        state['records']=[r for r in state['records'] if r['kind']!='target']
        ip.write_json(ctx['root']/'cache_status.json',state)
        with mock.patch.object(rt,'feature_manifest',wraps=rt.feature_manifest) as manifest, mock.patch.object(rt,'load_host',side_effect=AssertionError('model loaded')):
            result=rt.capture(ctx,['lighting'],'source','cuda:0')
        manifest.assert_not_called();self.assertEqual(result['status'],'selected_capture_complete')
        self.assertTrue(all(v==0 for v in result['last_cost']['counts'].values()))

    def test_pending_Target_still_requires_external_manifest_and_old_hash(self):
        ctx,_=self.fixture();state=ip.ledger(ctx);r=next(r for r in state['records'] if r['kind']=='target')
        (ctx['root']/r['file']).unlink();state['records'].remove(r)
        directory=self.root/'external';ctx['paths']['feature_root']=directory
        manifest=dict(format='original_query_support_features_v1',status='verified',category=ctx['category'],
                      extraction_identity=ctx['extraction_identity'],records=[])
        path=directory/ctx['category']/'manifest.json';ip.write_json(path,manifest)
        state['external_manifests']={'features':ip.sha(path)};ip.write_json(ctx['root']/'cache_status.json',state)
        before=copy.deepcopy(state['records']);path.unlink()
        with mock.patch.object(rt,'load_host',side_effect=AssertionError('model loaded')):
            with self.assertRaises(FileNotFoundError):rt.capture(ctx,['lighting'],'target','cuda:0')
            manifest['records']=[dict(kind='query',sample_id=r['sample_id'],file='changed.npz',sha256='changed')]
            ip.write_json(path,manifest)
            with self.assertRaisesRegex(ValueError,'enrolled feature manifest changed'):rt.capture(ctx,['lighting'],'target','cuda:0')
        self.assertEqual(ip.ledger(ctx)['records'],before)

    def test_legacy_Source_raw_nested_layout_identity_and_hash_history(self):
        ctx,raw=self.fixture();s=ctx['cat']['source_evaluation'][1]
        ctx['cat'].update(source_complete={'category':ctx['category'],'status':'complete'},legacy_records={s['relative_path']:{'path':'original_server_path'}})
        ctx['cat']['source_sha256'].update({'memory/anchor_sidecars.npz':'sidecar','memory/memory_provenance.json':'provenance'})
        source_id=ip.digest(dict(protocol='relation_reliability_exact_superadd_seed42_v6',complete=ctx['cat']['source_complete'],bank_size=1,
            sidecars_sha256='sidecar',calibration_sha256=ctx['cat']['source_sha256']['calibration/calibration_state.npz'],memory_provenance_sha256='provenance'))
        identity=ip.digest(dict(protocol='relation_oracle_four_layer_cache_seed42_v2',source_identity=source_id,
            record=ctx['cat']['legacy_records'][s['relative_path']],image_sha256=s['image_sha256'],mask_sha256=s['gt_sha256']))
        directory=self.root/'legacy';ctx['paths']['source_raw_root']=directory
        root=directory/'categories/robustad'/ctx['category'];ip.write_json(root/'complete.json',dict(status='complete',category=ctx['category']))
        p=root/'images/source__source/original.npz';p.parent.mkdir(parents=True)
        np.savez(p,image_path=np.asarray('/old_server/'+s['relative_path']),source_identity_sha256=np.asarray(source_id),
            source_protocol=np.asarray('relation_reliability_exact_superadd_seed42_v6'),category=np.asarray(ctx['category']),label=np.asarray(s['label']),
            layers=np.asarray(ip.LAYERS),cache_identity_sha256=np.asarray(identity),raw_distances=raw['source_distance'].reshape(4,3,4),
            top1_indices=raw['source_index'].reshape(4,3,4),evaluation_output_shape=np.asarray([12,16]))
        history={};arrays,meta=cal.legacy_raw(ctx,s,history);rt.validate_source(arrays,meta,s)
        np.testing.assert_array_equal(arrays['source_distance'],raw['source_distance'])
        p.write_bytes(p.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'changed'):cal.legacy_raw(ctx,s,history)

    def test_cpu_summary_and_review_archive_remain_pending_without_aupro(self):
        ctx,_=self.fixture();cal.recover(ctx,'cuda:0');pr.run(ctx,'lighting',1);ev.evaluate(ctx,'lighting','v1',1,1<<30)
        out=rep.summarize([ctx],'v1',self.root/'summary')
        self.assertFalse(ip.read_json(out/'status.json')['all_requested_metrics_complete'])
        archive=self.root/'review.tar.gz';rep.package([ctx],out,archive)
        with tarfile.open(archive,'r:gz') as tar:
            names=tar.getnames();self.assertIn('summary/metrics.csv',names)
            self.assertFalse(any('/raw_cache/' in n or n.endswith('.bin') for n in names))
        cpu=ev.directory(ctx,'lighting','v1','cpu');path=cpu/'cases.csv';path.write_bytes(path.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'changed'):rep.package([ctx],out,self.root/'another.tar.gz')

    def test_clean_inactive_cache_credit_does_not_include_shmem(self):
        from types import SimpleNamespace
        def text(path,*args,**kwargs):
            return {'memory.max':str(30<<30),'memory.current':str(29<<30),'memory.stat':
                    '\n'.join(f'{k} {v<<30}' for k,v in dict(file=27,inactive_file=26,shmem=2,file_dirty=1,file_writeback=1,file_mapped=2).items())}[path.name]
        with mock.patch.object(Path,'read_text',text),mock.patch('psutil.virtual_memory',return_value=SimpleNamespace(available=100<<30)):
            self.assertEqual(ip.available_ram(),22<<30)

    def test_aupro_orchestration_with_fake_backend_verifies_cpu_maps(self):
        # Only orchestration is real here. No Torch, CUDA or ADEval executes.
        from types import SimpleNamespace,ModuleType
        ctx,_=self.fixture();cal.recover(ctx,'cuda:0');pr.run(ctx,'lighting',1);ev.evaluate(ctx,'lighting','v1',1,1<<30)
        fake=ModuleType('torch');fake.device=lambda *a:SimpleNamespace(type='cuda',index=0)
        fake.cuda=SimpleNamespace(is_available=lambda:True,set_device=lambda d:None,
            get_device_properties=lambda d:SimpleNamespace(total_memory=24<<30),mem_get_info=lambda d:(24<<30,24<<30),
            set_per_process_memory_fraction=lambda *a:None,reset_peak_memory_stats=lambda d:None,
            max_memory_allocated=lambda d:0,max_memory_reserved=lambda d:0)
        backend=ModuleType('DINOv3.relation_reliability.official_metrics');calls=[]
        def metric(masks,maps,*,nstrips):
            self.assertEqual(nstrips,200);self.assertEqual(len(maps),2);calls.append(1);return .75
        backend.official_aupro_adeval=metric
        with mock.patch.dict(sys.modules,{'torch':fake,backend.__name__:backend}),mock.patch.object(ip,'available_ram',return_value=32<<30):
            out=ev.evaluate_aupro(ctx,'lighting','v1')
        ev.verified(ctx,'lighting','v1','aupro');self.assertEqual(len(calls),3)
        self.assertTrue(all(r['AUPRO']==.75 for r in ev.numeric_rows(out/'metrics.csv')))
        self.assertFalse(ip.read_json(out/'status.json')['all_requested_metrics_complete'])


class MathAndIdentity(unittest.TestCase):
    def test_non_square_physical_grid_permutation_and_edge_votes(self):
        raw,_=raw_fixture((3,5));expected,_=method.predict(raw,(3,5));perm=np.random.default_rng(17).permutation(15)
        shuffled={k:v[...,perm] for k,v in raw.items()};actual,_=method.predict(shuffled,(3,5))
        inverse=np.argsort(perm)
        np.testing.assert_array_equal(actual['pair_distance'][...,inverse],expected['pair_distance'])
        np.testing.assert_array_equal(actual['pair_gate'],expected['pair_gate'])
        self.assertEqual(expected['neighbor_count'][0,0,0],3);self.assertEqual(expected['neighbor_count'][0,0,1],5)
        self.assertEqual(expected['neighbor_count'][0,1,1],8)

    def test_stable_different_support_top2_and_degenerate_endpoints(self):
        from DINOv3.MADEqual.aebad_s_target_local_segment.geometry import geometry_layer
        bank=np.zeros((2,1280),np.float32)
        r,_=geometry_layer(np.ones((8,15),np.float32),np.zeros((8,15),np.int64),np.full(15,2,np.float32),[bank]*8)
        np.testing.assert_array_equal(r['endpoint_support'],np.tile(np.array([[1],[2]],np.int8),(1,15)))
        np.testing.assert_array_equal(r['K8_SEGMENT'],r['K8_UNION'])

    def test_two_level_macro_not_eleven_shift_equal_and_ARD_clip_before_mean(self):
        rows=[]
        for c,n,v in [('MetalParts',6,.2),('PCB',5,.8)]:
            for i in range(n):
                for b in ip.BRANCHES:rows.append(dict(category=c,shift=str(i),branch=b,**{k:v for k in ev.METRICS}))
        expected={'MetalParts':tuple(str(i) for i in range(6)),'PCB':tuple(str(i) for i in range(5))}
        result=rep.hierarchical(rows,expected)
        macro=[r for r in result if r.get('scope')=='category_equal_macro'];self.assertEqual(len(macro),3)
        self.assertTrue(all(abs(r['image_AUROC']-.5)<1e-12 for r in macro))
        source=[dict(category=c,shift='source',branch=pr.SOURCE,**{k:.5 for k in ev.METRICS}) for c in ip.CATEGORIES]
        ard=rep.ard_macros(rep.ard_rows(rows,source),expected)
        self.assertTrue(all(abs(r['ARD']+.15)<1e-12 for r in ard if r.get('scope')=='category_equal_macro'))

    def test_historical_tolerance_preserves_strict_result_and_signed_error(self):
        rows=[];s=dict(domain='lighting',sample_id='toy')
        self.assertTrue(historical_score(rows,s,'SOURCE_CURRENT',5.,5.00002));self.assertFalse(rows[-1]['strict_verified'])
        self.assertLess(rows[-1]['signed_error'],0)
        self.assertFalse(historical_score(rows,s,'SOURCE_CURRENT',5.,5.001))

    def test_real_catalog_counts_support_order_source_split_and_geometry(self):
        catalog=ip.read_json(Path(ip.__file__).with_name('frozen_catalog.json'))
        self.assertEqual(ip.sha(Path(ip.__file__).with_name('frozen_catalog.json')),ip.CATALOG_SHA)
        total=0;supports=0
        for c,v in catalog['categories'].items():
            self.assertEqual(len(v['calibration']),23);self.assertEqual(len(v['train']),180)
            self.assertEqual([s['source_index'] for s in v['calibration']],list(range(0,180,8)))
            for shift,d in v['shifts'].items():
                self.assertEqual([s['support_index'] for s in d['supports']],list(range(1,9)))
                self.assertFalse({s['sample_id'] for s in d['supports']}&{s['sample_id'] for s in d['samples']})
                self.assertEqual((sum(s['label']==0 for s in d['samples']),sum(s['label']==1 for s in d['samples'])),ip.COUNTS[c][shift])
                total+=len(d['samples']);supports+=len(d['supports'])
        self.assertEqual(total,1722);self.assertEqual(supports,88)
        self.assertEqual(len(catalog['categories']['MetalParts']['source_evaluation']),171)
        self.assertEqual(len(catalog['categories']['PCB']['source_evaluation']),195)

    def test_default_import_does_not_import_torch(self):
        code='''
import sys,importlib.abc
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,*a):
  if fullname=="torch" or fullname.startswith("torch."):raise AssertionError("Torch imported")
sys.meta_path.insert(0,Deny())
from DINOv3.MADEqual.robustad_segment_pair_crossdataset_p0 import runner
assert runner.parser().parse_args([]).stage=="preflight"
assert "torch" not in sys.modules
print("CPU_IMPORT_OK")
'''
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
        p=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,env=env,capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr);self.assertIn('CPU_IMPORT_OK',p.stdout)


if __name__=='__main__':unittest.main()
