"""Real NumPy math / actual builder AST fixtures; never import real Torch/CUDA."""
from __future__ import annotations
import ast
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from dataclasses import dataclass
from types import ModuleType,SimpleNamespace
from unittest.mock import patch
from collections import defaultdict
import numpy as np
from PIL import Image

from DINOv3.MADEqual.aebad_s_sparc_p0 import projection as geom,inputs,runtime,analysis,runner
from DINOv3.tests.test_mad_equal_aebad_s_target_local_segment import frozen_fixture,full_protocol_fixture
from DINOv3.tests.test_mad_equal_aebad_s_union_support_loo import actual_map_backend

ROOT=Path(__file__).resolve().parents[2]


def actual_functions(path,names,env):
    nodes=[n for n in ast.parse(path.read_text(encoding='utf-8')).body
           if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    for n in nodes:
        if isinstance(n,ast.FunctionDef): n.decorator_list=[]
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*nodes],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),env)
    return env


class FakeTensor:
    def __init__(self,value): self.value=np.asarray(value)
    def to(self,*args,**kwargs): return self
    def cpu(self): return self
    def contiguous(self): return self
    @property
    def shape(self): return self.value.shape
    def numpy(self): return self.value
    def __getitem__(self,key): return FakeTensor(self.value[key])


def builder_fixture():
    """Execute actual spools, split, preprocessing/callback ordering and sidecar path.

    Feature executor, tensor transfer and sampler are explicit NumPy doubles.
    """
    import tempfile as temp
    samplers=[]
    def indexed(values,quota,device,**kwargs):
        samplers.append(dict(kwargs,shape=values.shape,candidates=np.array(values,copy=True),
                             is_memmap=isinstance(values,np.memmap)))
        idx=np.sort(np.random.choice(len(values),quota,replace=False)).astype(np.int64)
        return np.asarray(values[idx]),idx
    env=dict(np=np,Path=Path,tempfile=temp,dataclass=dataclass,
             torch=SimpleNamespace(from_numpy=FakeTensor,float32=np.float32),
             RAM_BACKEND_HEADROOM_BYTES=16<<30,MEMORY_VARIANT_PROTOCOL='fixture',SAMPLER_DISTANCE_MATRIX_MAX_BYTES=1<<30,
             _official_subsample_values_with_indices=indexed)
    actual_functions(ROOT/'external_baselines/superadd_external/memory_variants.py',
                     {'VariantBuild','_MatrixSpool','_RamMatrix','build_official_memory_spooled'},env)
    rgb=[FakeTensor([i]) for i in range(17)]
    calls=[]
    def preprocess(x):
        factor=np.random.uniform(.8,1.2);return (int(x.value.reshape(-1)[0]),factor)
    def execute(processed,backbone):
        i,factor=processed;calls.append(i)
        a=np.zeros((1,2,2,1280),np.float32);a[...,0]=i*factor+np.arange(4).reshape(2,2)
        return [a.copy() for _ in geom.LAYERS]
    model=SimpleNamespace(threshold_fraction=8,layers=geom.LAYERS,max_database_size=3,
                          augmented_preprocessing=preprocess,patch_exec=execute,backbone=None)
    return env,model,rgb,calls,samplers


def evaluation_fixture(root):
    """297 tiny synthetic cached inputs, mocked prepared bank, no production stages."""
    args,arrays,banks=frozen_fixture(root)
    paths=dict(project=str(ROOT),results=args.results_dir,capture=args.capture_dir,support=args.support_cache_dir,
               data=args.data,official=str(ROOT/'external_baselines/SuperADD'),dinov3=str(root/'dino'),
               checkpoint=str(root/'weights.pth'),output=args.output_dir)
    out=Path(paths['output']);out.mkdir()
    cap=json.loads((Path(paths['capture'])/'inputs.json').read_text())
    state=json.loads((Path(paths['capture'])/'status.json').read_text())
    records=[]
    for record,s in zip(state['records'],cap['samples'],strict=True):
        file=Path(paths['capture'])/record['file']
        with np.load(file,allow_pickle=False) as f: meta=json.loads(str(f['metadata'].item()))
        meta['geometry']={'grid':[2,2],'synthetic':True}
        with file.open('wb') as f: np.savez_compressed(f,metadata=np.asarray(json.dumps(meta)),**arrays)
        record['sha256']=inputs.replay.sha256(file)
    inputs.replay.atomic_json(Path(paths['capture'])/'status.json',state)
    frozen=dict(grid=[2,2],synthetic_mock_prepare=True)
    identity=inputs.digest(frozen)
    data=dict(tool=inputs.TOOL,identity=identity,frozen=frozen,paths=paths,samples=cap['samples'],required_files=[])
    inputs.replay.atomic_json(out/'inputs.json',data)
    np.savez(out/'sparc_source_mad_state.npz',protocol_id=np.asarray(inputs.replay.PROTOCOL_ID),
             medians=np.zeros(4,np.float32),scales=np.ones(4,np.float32))
    prepared=dict(tool=inputs.TOOL,status='prepare_complete',identity=identity,inputs_sha256=inputs.replay.sha256(out/'inputs.json'),
                  artifacts=inputs.file_records(out,[out/'sparc_source_mad_state.npz']))
    inputs.replay.atomic_json(out/'prepare_status.json',prepared)
    for s,old in zip(cap['samples'],state['records'],strict=True):
        m=dict(tool=inputs.TOOL,identity=identity,sample_id=s['sample_id'],relative_path=s['relative_path'],
            baseline_raw_sha256=old['sha256'],image_sha256=cap['image_sha256'][s['relative_path']],branch='SPARC_SOURCE_K8',
            distance_divisor=1280,geometry={'grid':[2,2],'synthetic':True},grid=[2,2],output_shape=[2,2])
        file=out/'sparc_raw'/f"{s['sample_id']}.npz"
        runtime.save_npz(file,metadata=np.asarray(json.dumps(m)),distance=np.minimum(arrays['source_distance'],arrays['target_distance'].min(axis=0)),
            positions=arrays['positions'],nearest_index=np.zeros((4,4),np.int64),grid=np.array([2,2],np.int32))
        records.append(dict(sample_id=s['sample_id'],**inputs.file_records(out,[file])[0]))
    inputs.replay.atomic_json(out/'infer_status.json',dict(tool=inputs.TOOL,identity=identity,status='predictions_frozen',
        predicted_images=297,inputs_sha256=inputs.replay.sha256(out/'inputs.json'),
        prepare_status_sha256=inputs.replay.sha256(out/'prepare_status.json'),records=records))
    backend,trace=actual_map_backend();backend.mad.medians[:]=0;backend.mad.scales[:]=1
    backend.patch_mask=lambda mask,grid:np.asarray(mask)[::4,::4]>0
    backends={b:backend for b in geom.BRANCHES}
    return data,prepared,backends


class HostInitializationTests(unittest.TestCase):
    """Call the actual lazy load_host with isolated Torch/host import doubles."""
    def fixture(self,current=0,available=True,freeze_works=True):
        events=[]
        cuda=SimpleNamespace(selected=current,is_available=lambda:available,current_device=lambda:cuda.selected)
        def device(name,index=None):
            kind,_,suffix=str(name).partition(':')
            return SimpleNamespace(type=kind,index=index if index is not None else (int(suffix) if suffix else None))
        def set_device(selected):
            cuda.selected=selected.index;events.append(('select',selected.index))
        cuda.set_device=set_device
        cuda.reset_peak_memory_stats=lambda selected:events.append(('reset',selected.index,cuda.selected))
        torch=ModuleType('torch');torch.device=device;torch.cuda=cuda
        torch.backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                                       cudnn=SimpleNamespace(allow_tf32=False))
        torch.set_float32_matmul_precision=lambda value:events.append(('precision',value))
        params=[SimpleNamespace(requires_grad=True)]
        dino=SimpleNamespace(training=True,parameters=lambda:iter(params))
        def eval_model():
            dino.training=False;events.append(('eval',));return dino
        def freeze(value):
            events.append(('freeze',value))
            if freeze_works:
                for p in params:p.requires_grad=value
            return dino
        dino.eval=eval_model;dino.requires_grad_=freeze
        model=SimpleNamespace(layers=geom.LAYERS,backbone=SimpleNamespace(dino=dino),
            preprocessing=SimpleNamespace(resize_factor=.625,min_brightness_factor=1,max_brightness_factor=1),
            evaluation_downscale=4,patch_exec=SimpleNamespace(model_patch_size=16))
        data={'paths':{key:f'/fixture/{key}' for key in ('project','official','dinov3','checkpoint')},
              'frozen':{'host_config':{'fixed':True},'precision':{
                  'float32_matmul_precision':'highest','cuda_matmul_allow_tf32':False,'cudnn_allow_tf32':True}}}
        official=ModuleType('industrial.model');official.get_weights_path=lambda name:'original'
        industrial=ModuleType('industrial');industrial.__path__=[];industrial.model=official
        host=ModuleType('DINOv3.relation_reliability.superadd_host')
        host.install_import_paths=lambda *args:events.append(('paths',))
        def construct(*args):
            events.append(('construct',cuda.selected))
            self.assertEqual(official.get_weights_path('dino'),Path(data['paths']['checkpoint']))
            model.preprocessing.device=cuda.selected
            for p in params:p.device=cuda.selected
            return model,dict(data['frozen']['host_config'])
        host.construct_exact_host=construct
        seed=ModuleType('DINOv3.MADEqual.aebad_s_target_fewshot.runner')
        seed._seed_everything=lambda:events.append(('seed',cuda.selected))
        modules={'torch':torch,'industrial':industrial,'industrial.model':official,
                 host.__name__:host,seed.__name__:seed}
        return data,modules,model,cuda,events

    def test_normal_host_is_frozen_before_validation(self):
        data,modules,model,cuda,events=self.fixture()
        with patch.dict(sys.modules,modules):
            loaded,selected=runtime.load_host(data,'cuda:0')
        self.assertIs(loaded,model);self.assertEqual(selected.index,0)
        self.assertFalse(model.backbone.dino.training)
        self.assertTrue(all(not p.requires_grad for p in model.backbone.dino.parameters()))
        self.assertLess(events.index(('construct',0)),events.index(('eval',)))
        self.assertLess(events.index(('eval',)),events.index(('freeze',False)))
        self.assertEqual(modules['industrial.model'].get_weights_path('dino'),'original')
        self.assertNotIn('torch',sys.modules)

    def test_explicit_nondefault_device_selected_before_seed_reset_and_construct(self):
        data,modules,model,cuda,events=self.fixture(current=0)
        with patch.dict(sys.modules,modules):loaded,selected=runtime.load_host(data,'cuda:1')
        self.assertEqual(selected.index,1);self.assertEqual(cuda.selected,1)
        self.assertEqual(events[0],('select',1))
        self.assertIn(('seed',1),events);self.assertIn(('reset',1,1),events);self.assertIn(('construct',1),events)
        self.assertEqual(model.preprocessing.device,1)
        self.assertTrue(all(p.device==1 for p in model.backbone.dino.parameters()))
        self.assertFalse(modules['torch'].backends.cuda.matmul.allow_tf32)
        self.assertTrue(modules['torch'].backends.cudnn.allow_tf32)

    def test_bare_cuda_resolves_current_index(self):
        data,modules,model,cuda,events=self.fixture(current=1)
        with patch.dict(sys.modules,modules):loaded,selected=runtime.load_host(data,'cuda')
        self.assertEqual(selected.index,1);self.assertEqual(events[0],('select',1))
        self.assertIn(('reset',1,1),events);self.assertIn(('construct',1),events)

    def test_cpu_or_unavailable_cuda_rejected_before_construction(self):
        for name,available in [('cpu',True),('cuda:0',False)]:
            with self.subTest(device=name,available=available):
                data,modules,model,cuda,events=self.fixture(available=available)
                with patch.dict(sys.modules,modules),self.assertRaisesRegex(RuntimeError,'requires CUDA'):
                    runtime.load_host(data,name)
                self.assertEqual(events,[])

    def test_freeze_validation_remains_enforced(self):
        data,modules,model,cuda,events=self.fixture(freeze_works=False)
        with patch.dict(sys.modules,modules),self.assertRaisesRegex(ValueError,'constructed host differs'):
            runtime.load_host(data,'cuda:0')
        self.assertIn(('freeze',False),events)


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(42)
        x=rng.normal(size=(4,8,1280)).astype(np.float32)
        mu,v,rank=geom.fit_cells(x)
        self.x=x;self.p=geom.Projection((2,2),mu,v,rank);self.ids=np.arange(4,dtype=np.int64)

    def test_mean_preserved_and_zero_rank_identity(self):
        mu=np.arange(5120,dtype=np.float32).reshape(4,1280)
        means,basis,rank=geom.fit_cells(np.repeat(mu[:,None,:],8,axis=1))
        p=geom.Projection((2,2),means,basis,rank)
        np.testing.assert_array_equal(rank,0)
        np.testing.assert_array_equal(p.correct(self.x[:,0],self.ids,(2,2)),self.x[:,0])
        np.testing.assert_allclose(self.p.correct(self.p.mu.astype(np.float32),self.ids,(2,2)),self.p.mu,atol=1e-6)

    def test_basis_orthogonal_and_correction_idempotent(self):
        gram=np.einsum('ndr,nds->nrs',self.p.basis,self.p.basis)
        np.testing.assert_allclose(gram,np.broadcast_to(np.eye(7),gram.shape),atol=1e-12)
        once=self.p.correct(self.x[:,0],self.ids,(2,2))
        twice=self.p.correct(once,self.ids,(2,2))
        np.testing.assert_allclose(twice,once,atol=1e-7)

    def test_all_eight_supports_collapse_to_position_mean(self):
        np.testing.assert_array_equal(self.p.rank,7)
        for j in range(8): np.testing.assert_allclose(self.p.correct(self.x[:,j],self.ids,(2,2)),self.p.mu,atol=1e-6)

    def test_basis_sign_changes_leave_result_unchanged(self):
        p=geom.Projection((2,2),self.p.mu,self.p.basis*np.array([1,-1,1,-1,1,-1,1]),self.p.rank)
        np.testing.assert_array_equal(p.correct(self.x[:,0],self.ids,(2,2)),self.p.correct(self.x[:,0],self.ids,(2,2)))

    def test_positions_reordering_and_independent_cells(self):
        ids=np.array([2,0,3,1],np.int64)
        expected=self.p.correct(self.x[:,0],self.ids,(2,2))
        np.testing.assert_array_equal(self.p.correct(self.x[ids,0],ids,(2,2)),expected[ids])
        wrong=self.p.correct(self.x[ids,0],self.ids,(2,2))
        self.assertGreater(float(np.max(np.abs(wrong-expected[ids]))),.01)
        np.testing.assert_array_equal(self.p.correct(self.x[:,0],self.ids,(2,2),chunk=1),expected)

    def test_incompatible_grid_and_nonbijective_positions_rejected(self):
        with self.assertRaises(ValueError): self.p.correct(self.x[:,0],self.ids,(1,4))
        with self.assertRaises(ValueError): self.p.correct(self.x[:,0],np.array([0,0,2,3],np.int64),(2,2))

    def test_disk_projection_state_and_rank_statistics(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'state'
            stat=geom.fit_layer([self.x[:,j] for j in range(8)],(2,2),root)
            p=geom.load_projection(root,(2,2))
            self.assertEqual(stat['rank_counts'][7],4)
            np.testing.assert_array_equal(p.correct(self.x[:,0],self.ids,(2,2)),self.p.correct(self.x[:,0],self.ids,(2,2)))
            p.close()


class IntegrationTests(unittest.TestCase):
    def test_actual_builder_default_vs_identity_callback_and_lazy_split(self):
        env,model,tensors,calls,samplers=builder_fixture()
        with tempfile.TemporaryDirectory() as td:
            kwargs=dict(device=SimpleNamespace(type='cpu'),scratch_dir=td,return_selected_indices=True,storage_backend='disk')
            np.random.seed(42);old=env['build_official_memory_spooled'](model,tensors,**kwargs)
            before=np.random.get_state();old_candidates=calls.copy();calls.clear();seen=[]
            p=geom.Projection((2,2),np.zeros((4,1280)),np.zeros((4,1280,7)),np.zeros(4,np.int8))
            class Lazy:
                def __len__(self):return len(tensors)
                def __getitem__(self,i):
                    self_outer.assertNotEqual(i%8,0);return tensors[i]
            self_outer=self
            def identity(i,l,e):seen.append((i,l));return runtime.correct_embedding(p,e)
            np.random.seed(42);new=env['build_official_memory_spooled'](model,Lazy(),feature_transform=identity,**kwargs)
            after=np.random.get_state()
            self.assertEqual(calls,old_candidates)
            self.assertEqual(calls,[i for i in range(17) if i%8])
            self.assertEqual(len(seen),14*4)
            np.testing.assert_array_equal(before[1],after[1]);self.assertEqual(before[2:],after[2:])
            for l in geom.LAYERS:
                np.testing.assert_array_equal(old.banks[l].numpy(),new.banks[l].numpy())
                np.testing.assert_array_equal(old.selected_candidate_indices[l],new.selected_candidate_indices[l])
            self.assertTrue(all(r['iterations']==100 and r['seed'] is None for r in samplers))

    def test_original_indexed_sampler_matches_actual_upstream_rule(self):
        def nearest(a,b,*args,**kwargs):
            d=np.linalg.norm(a.value[:,None]-b.value[None],axis=2)
            return FakeTensor(np.sort(d,axis=1)),None
        env=dict(np=np,torch=SimpleNamespace(from_numpy=FakeTensor),SAMPLER_DISTANCE_MATRIX_MAX_BYTES=1<<30,
                 _chunked_exact_nearest_neighbors=nearest,nearest_neighbors=nearest)
        actual_functions(ROOT/'external_baselines/superadd_external/memory_variants.py',{'_official_subsample_values_with_indices'},env)
        actual_functions(ROOT/'external_baselines/SuperADD/tracks/industrial/src/industrial/nearest_neighbor.py',{'subsampling_distance_based_fast'},env)
        values=np.arange(2000,dtype=np.float32).reshape(1000,2)
        np.random.seed(42);old=env['subsampling_distance_based_fast'](values,100,'fixture',iterations=10)
        state=np.random.get_state()
        np.random.seed(42);new,idx=env['_official_subsample_values_with_indices'](values,100,'fixture',seed=None,iterations=10)
        np.testing.assert_array_equal(old,new);np.testing.assert_array_equal(new,values[idx])
        current=np.random.get_state();np.testing.assert_array_equal(state[1],current[1]);self.assertEqual(state[2:],current[2:])

    def test_mixed_ram_and_disk_match_all_disk_candidates_indices_banks_and_rng(self):
        env,model,tensors,calls,samplers=builder_fixture()
        mu=np.full((4,1280),3.,np.float64);v=np.zeros((4,1280,7));v[:,0,0]=1
        projection=geom.Projection((2,2),mu,v,np.ones(4,np.int8))
        seen=[]
        def transform(i,l,e):
            seen.append((i,l));e=e.copy();e[...,1]=i+l
            return runtime.correct_embedding(projection,e)
        results=[]
        with tempfile.TemporaryDirectory() as td:
            for storage in ('disk','mixed','ram'):
                calls.clear();samplers.clear();seen.clear();np.random.seed(42)
                build=env['build_official_memory_spooled'](model,tensors,
                    device=SimpleNamespace(type='cpu'),scratch_dir=td,return_selected_indices=True,
                    storage_backend=storage,ram_layers=geom.LAYERS[:2] if storage=='mixed' else None,
                    expected_candidate_rows=56,feature_transform=transform)
                results.append((build,calls.copy(),samplers.copy(),seen.copy(),np.random.get_state()))
                self.assertEqual(list(Path(td).iterdir()),[])
        old,old_calls,old_samplers,old_seen,old_rng=results[0]
        for new,new_calls,new_samplers,new_seen,new_rng in results[1:]:
            self.assertEqual(new_calls,old_calls);self.assertEqual(new_seen,old_seen)
            np.testing.assert_array_equal(old_rng[1],new_rng[1]);self.assertEqual(old_rng[2:],new_rng[2:])
            for j,l in enumerate(geom.LAYERS):
                np.testing.assert_array_equal(new_samplers[j]['candidates'],old_samplers[j]['candidates'])
                self.assertEqual(new_samplers[j]['candidates'].dtype,np.float32)
                np.testing.assert_array_equal(new.banks[l].numpy(),old.banks[l].numpy())
                np.testing.assert_array_equal(new.selected_candidate_indices[l],old.selected_candidate_indices[l])
        mixed=results[1][0];meta=mixed.metadata
        self.assertEqual([r['is_memmap'] for r in results[1][2]],[False,False,True,True])
        self.assertEqual(meta['candidate_resident_bytes'],56*1280*4*2)
        self.assertEqual(meta['candidate_scratch_peak_bytes'],56*1280*4*2)
        self.assertEqual(meta['candidate_storage_decision']['ram_layers'],[7,15])
        self.assertEqual(meta['candidate_storage_decision']['disk_layers'],[23,31])

    def test_mixed_storage_rejects_invalid_layer_plan_or_unknown_capacity(self):
        env,model,tensors,calls,samplers=builder_fixture()
        with tempfile.TemporaryDirectory() as td:
            for layers,rows in [((),56),((7,7),56),((999,),56),(geom.LAYERS,56),((7,15),None)]:
                with self.subTest(layers=layers,rows=rows),self.assertRaises(ValueError):
                    env['build_official_memory_spooled'](model,tensors,device=SimpleNamespace(type='cpu'),
                        scratch_dir=td,storage_backend='mixed',ram_layers=layers,expected_candidate_rows=rows)
            self.assertEqual(calls,[]);self.assertEqual(list(Path(td).iterdir()),[])

    def test_source_and_query_both_corrected_no_target_append_once_scaled(self):
        # Explicit tensor/NN double; tests the production correction/search orchestration.
        def top1(q,b):
            d=np.linalg.norm(q.value[:,None]-b.value[None],axis=2);idx=np.argmin(d,axis=1)
            return FakeTensor(d[np.arange(len(idx)),idx].astype(np.float32)),FakeTensor(idx.astype(np.int64))
        host=SimpleNamespace(_exact_top1_chunked=top1)
        fake=SimpleNamespace(from_numpy=FakeTensor)
        mu=np.zeros((4,1280));v=np.zeros((4,1280,7));v[:,0,0]=1
        p=geom.Projection((2,2),mu,v,np.ones(4,np.int8))
        source=np.zeros((1,2,2,1280),np.float32);source[...,0]=12800
        corrected=runtime.correct_embedding(p,source)
        query=np.zeros((4,1280),np.float32);query[:,0]=25600;query[:,1]=1280
        with patch.dict(sys.modules,{'torch':fake,'DINOv3.relation_reliability.superadd_host':host}):
            d,idx=runtime.exact_corrected_layer(query,np.arange(4,dtype=np.int64),(2,2),p,FakeTensor(corrected.reshape(4,1280)),'fixture')
            np.testing.assert_array_equal(d,1)
            wrong,_=runtime.exact_corrected_layer(query,np.arange(4,dtype=np.int64),(2,2),p,FakeTensor(source.reshape(4,1280)),'fixture')
            self.assertTrue((wrong>10).all())
        self.assertEqual(geom.BRANCHES,('SOURCE','UNION_K8','SPARC_SOURCE_K8'))
        self.assertEqual(corrected.shape,source.shape)

    def test_bank_selected_candidate_mapping_excludes_calibration(self):
        entries=[dict(source_index=1,candidate_start=0,candidate_stop=4),dict(source_index=2,candidate_start=4,candidate_stop=8)]
        result=runtime.bank_provenance(np.array([1,4,7]),entries)
        np.testing.assert_array_equal(result['source_index'],[1,2,2]);np.testing.assert_array_equal(result['physical_position'],[1,0,3])
        with self.assertRaises(ValueError): runtime.bank_provenance(np.array([8]),entries)
        entries[1]['source_index']=8
        with self.assertRaises(ValueError): runtime.bank_provenance(np.array([4]),entries)

    def test_source_count_order_and_split(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);train=root/'train/good';train.mkdir(parents=True)
            for i in range(521): (train/f'{i:04d}.png').touch()
            records=inputs.source_records(root)
            m,c=inputs.split_source_normal(records,8)
            self.assertEqual((len(records),len(m),len(c)),(521,455,66))
            self.assertEqual(Path(c[1].path).name,'0008.png')
            with self.assertRaises(ValueError):
                (train/'0520.png').unlink();inputs.source_records(root)

    def test_identity_projection_matches_original_formal_map_chain(self):
        backend,trace=actual_map_backend()
        source=np.arange(16,dtype=np.float32).reshape(4,4)+1
        p=np.array([2,0,3,1],np.int64)
        meta=dict(grid=[2,2],output_shape=[2,2]);raws={b:source.copy() for b in geom.BRANCHES}
        result=analysis.formal_maps(raws,p,meta,{b:backend for b in geom.BRANCHES})
        expected=backend.runner._method_map({'official_raw':analysis.scatter(source,p,(2,2))},'MAD_EQUAL',backend.mad,(2,2))
        for m in result.values(): np.testing.assert_array_equal(m,expected)
        self.assertEqual(trace[:2],['layer_MAD_then_mean','interpolate_after_mean'])

    def test_signed_distance_audit_uses_old_scale_not_changed_MAD(self):
        raws=dict(UNION_K8=np.full((4,4),1,np.float32),SOURCE=np.full((4,4),2,np.float32),SPARC_SOURCE_K8=np.full((4,4),3,np.float32))
        rows=analysis.distance_audit(dict(sample_id='q',label=0),raws,np.arange(4,dtype=np.int64),None,np.full(4,2))
        self.assertTrue(all(r['R_value']==-1 for r in rows if r['branch']=='SPARC_SOURCE_K8'))
        self.assertTrue(all(r['R_value']==0 for r in rows if r['branch']=='UNION_K8'))


class FlowTests(unittest.TestCase):
    def test_default_help_and_missing_asset_preflight_no_torch_no_files(self):
        self.assertNotIn('torch',sys.modules)
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as exc: runner.main(['--help'])
        self.assertEqual(exc.exception.code,0)
        self.assertEqual(runner.build_parser().parse_args([]).mode,'preflight')
        with tempfile.TemporaryDirectory() as td:
            dest=Path(td)/'output'
            with contextlib.redirect_stdout(io.StringIO()):
                rc=runner.main(['--project-root',str(ROOT),'--capture-dir',str(Path(td)/'missing'),
                    '--data',str(Path(td)/'data'),'--checkpoint',str(Path(td)/'weights'),'--output-dir',str(dest),'--no-write'])
            self.assertEqual(rc,2);self.assertFalse(dest.exists());self.assertNotIn('torch',sys.modules)

    def test_modified_prepared_state_and_half_output_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);out=root/'out';out.mkdir();file=out/'mad.npz';file.write_bytes(b'fixture')
            frozen={'fixed':True};identity=inputs.digest(frozen)
            inputs.replay.atomic_json(out/'inputs.json',dict(tool=inputs.TOOL,identity=identity,frozen=frozen,required_files=[]))
            inputs.replay.atomic_json(out/'prepare_status.json',dict(tool=inputs.TOOL,status='prepare_complete',identity=identity,
                inputs_sha256=inputs.replay.sha256(out/'inputs.json'),artifacts=inputs.file_records(out,[file])))
            inputs.read_prepared(out);file.write_bytes(b'changed')
            with self.assertRaises(ValueError):inputs.read_prepared(out)
            with self.assertRaises(ValueError):inputs.guard_output(dict(project=ROOT,output=out,data=root/'data'))

    def test_resource_estimate_never_reduces_original_budget(self):
        r=inputs.resource_estimate(455*118*118,(118,118))
        disk=inputs.resource_estimate(455*118*118,(118,118),candidate_storage='disk')
        self.assertEqual(r['candidate_float32_bytes'],455*118*118*1280*4*4)
        self.assertEqual(r['candidate_float32_bytes'],disk['candidate_float32_bytes'])
        self.assertEqual(r['candidate_scratch_bytes'],r['candidate_resident_bytes'])
        self.assertEqual(r['ram_layers'],[7,15]);self.assertEqual(r['disk_layers'],[23,31])
        self.assertEqual(disk['candidate_resident_bytes'],0)
        self.assertGreater(disk['prepare_disk_required_bytes'],120*(1<<30))
        self.assertEqual(disk['prepare_disk_required_bytes']-r['prepare_disk_required_bytes'],r['candidate_resident_bytes'])
        self.assertLess(r['prepare_disk_required_bytes'],127001509888)
        self.assertLess(r['minimum_ram_headroom_bytes'],127427170304)
        self.assertEqual(r['new_memory_bytes'],100000*1280*4*4)

    def test_mixed_prepare_rejects_low_or_unknown_ram_before_host(self):
        modules={'torch':SimpleNamespace(),
            'DINOv3.MADEqual.robustad_target_fewshot_p0.support':SimpleNamespace(extract_or_load_support=None),
            'external_baselines.superadd_external.memory_variants':SimpleNamespace(build_official_memory_spooled=None),
            'DINOv3.relation_reliability.calibration':SimpleNamespace(fit_calibrators=None),
            'DINOv3.MADEqual.aebad_s_target_support_trace.analysis':SimpleNamespace(image_geometry=None),
            'DINOv3.relation_reliability.superadd_host':SimpleNamespace(preprocessed_spatial_shape=None)}
        resources=inputs.resource_estimate(455*118*118,(118,118))
        data=dict(paths={'output':'/fixture/output'},identity='fixture',resource_estimate=resources)
        for budget in (None,resources['minimum_ram_headroom_bytes']-1):
            with self.subTest(budget=budget),patch.dict(sys.modules,modules),patch.object(runtime,'load_host') as host:
                with patch.object(runtime,'free_space',return_value=1<<40),patch.object(runtime,'available_ram',return_value=budget):
                    with self.assertRaisesRegex(RuntimeError,'RAM budget|RAM no longer sufficient'):
                        runtime.prepare(data,'cuda:0')
                host.assert_not_called()

    def test_full_evaluation_fixture_three_branches_and_baseline_verification(self):
        with tempfile.TemporaryDirectory() as td:
            data,prepared,backends=evaluation_fixture(Path(td))
            with (patch.object(analysis,'runtime_backends',return_value=backends),patch.object(analysis,'runtime_evaluator',return_value=full_protocol_fixture()),
                  patch('DINOv3.MADEqual.aebad_s_sparc_p0.visualization.plot_case')):
                with contextlib.redirect_stdout(io.StringIO()):state=analysis.evaluate(data,prepared)
            self.assertEqual(state['status'],'analysis_complete');self.assertEqual(state['analyzed_images'],297)
            out=Path(data['paths']['output'])/'evaluation'
            self.assertEqual(len(inputs.replay.read_csv(out/'per_image_scores.csv')),891)
            self.assertEqual([r['branch'] for r in inputs.replay.read_csv(out/'metrics.csv')],list(geom.BRANCHES))
            self.assertEqual(state['numerical_failed_checks'],0)
            self.assertTrue(state['formal_baseline_reproduced'])

    def test_consistent_score_backend_shift_blocks_completion(self):
        with tempfile.TemporaryDirectory() as td:
            data,prepared,backends=evaluation_fixture(Path(td));backend=backends['SOURCE']
            original=backend.runner._method_map
            backend.runner._method_map=lambda *a,**kw:original(*a,**kw)+1
            with (patch.object(analysis,'runtime_backends',return_value=backends),patch.object(analysis,'runtime_evaluator',return_value=full_protocol_fixture()),
                  patch('DINOv3.MADEqual.aebad_s_sparc_p0.visualization.plot_case')):
                with contextlib.redirect_stdout(io.StringIO()):state=analysis.evaluate(data,prepared)
            self.assertEqual(state['status'],'partial_evidence');self.assertEqual(state['numerical_failed_checks'],594)

    def test_modified_prediction_refused_before_GT(self):
        with tempfile.TemporaryDirectory() as td:
            data,prepared,backends=evaluation_fixture(Path(td));out=Path(data['paths']['output'])
            file=next((out/'sparc_raw').glob('*.npz'));file.write_bytes(b'changed')
            with patch.object(analysis,'runtime_evaluator') as backend:
                with self.assertRaises(ValueError):analysis.evaluate(data,prepared)
                backend.assert_not_called()
            self.assertFalse((out/'evaluation').exists())


if __name__=='__main__':unittest.main()
