"""Actual tensor implementation executed on CPU; explicitly NOT CUDA validation."""
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from collections import OrderedDict
import numpy as np
import torch
from DINOv3.MADEqual.aebad_s_annular_ordinal_p0 import pixels as cp,geometry as cg,evidence as ce,protocol as p,inputs as ip,runtime as rt
from DINOv3.MADEqual.aebad_s_annular_ordinal_p0 import pixels_cuda as px,geometry_cuda as gg,evidence_cuda as ee,index_cuda as ix,compact_storage as codec
from DINOv3.MADEqual.aebad_s_annular_ordinal_p0.cuda_support import Timers,TensorCache,require_device
from DINOv3.MADEqual.aebad_s_annular_ordinal_p0.runner import parser,normalize
from DINOv3.MADEqual.aebad_s_annular_ordinal_p0.verify_cuda import small_checks,fingerprint
from DINOv3.MADEqual.aebad_s_annular_ordinal_p0 import pipeline as pl

class TestAnnularCUDAImplementationOnCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.path=Path(self.temp.name)
        self.gray=np.random.default_rng(128).random((192,192),dtype=np.float32);self.t=torch.from_numpy(self.gray)
    def refs(self,images=None):
        images=images or [self.gray,self.gray.copy()]
        return [dict(index=i,instance_id=str(i),kind='Source',shape=image.shape,read_gray=lambda image=image:image) for i,image in enumerate(images)]
    def pool(self):
        T=[[1.,0,0],[0,1.,0]]
        return dict(candidates=[dict(ref=i,instance_id=str(i),T=T) for i in range(2)],search_unresolved=False)

    def script_probe(self,stages,selection):
        bash=shutil.which('bash')
        if os.name=='nt':bash='D:/Git/Git/bin/bash.exe' if Path('D:/Git/Git/bin/bash.exe').exists() else bash
        if not bash:self.skipTest('bash not available')
        root=Path(__file__).resolve().parents[2];probe=self.path/'probe.py';shim=self.path/'python-shim'
        probe.write_text('''import json,sys
from types import SimpleNamespace
from DINOv3.MADEqual.aebad_s_annular_ordinal_p0.runner import parser
from DINOv3.MADEqual.aebad_s_annular_ordinal_p0 import inputs as ip
a=parser().parse_args(sys.argv[5:])
a.run_root.mkdir(parents=True,exist_ok=True)
a._assets=SimpleNamespace(provenance=lambda:dict(fixture='frozen_inputs'))
r=ip.freeze(a)
print(json.dumps(dict(identity=r['identity'],visibility=r['execution']['visible_devices'],torch_imported='torch' in sys.modules)))
''',encoding='utf-8')
        shim.write_text('#!/usr/bin/env bash\nexec "$ANNULAR_TEST_PYTHON" "$ANNULAR_TEST_PROBE" "$@"\n',encoding='utf-8')
        shim.chmod(0o755)
        env={k:v for k,v in os.environ.items() if k not in ('GPU_INDEX','CUDA_VISIBLE_DEVICES')}
        env.update(selection,REPO_ROOT=root.as_posix(),PYTHON=shim.as_posix(),ANNULAR_TEST_PYTHON=Path(sys.executable).as_posix(),
            ANNULAR_TEST_PROBE=probe.as_posix(),PYTHONPATH=str(root),RUN_ROOT=(self.path/'run').as_posix(),CACHE_ROOT=(self.path/'cache').as_posix(),BACKEND='cuda')
        results=[]
        for stage in stages:
            run=subprocess.run([bash,(root/'DINOv3/MADEqual/aebad_s_annular_ordinal_p0/scripts/run_workstation.sh').as_posix(),stage],
                env=env,cwd=root,capture_output=True,text=True,timeout=20)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr);results.append(json.loads(run.stdout))
        return results

    def test_script_all_stages_freeze_same_visibility_without_importing_torch(self):
        stages=('preflight','check-cuda','build-index','smoke','calibrate','infer','evaluate','report','pack','all')
        results=self.script_probe(stages,dict(GPU_INDEX='1',CUDA_VISIBLE_DEVICES='0'))
        assert len({r['identity'] for r in results})==1
        assert all(r['visibility']=='1' and not r['torch_imported'] for r in results)

    def test_script_explicit_visibility_preserved_in_standalone_evaluation(self):
        results=self.script_probe(('infer','evaluate','report'),dict(CUDA_VISIBLE_DEVICES='1'))
        assert len({r['identity'] for r in results})==1
        assert all(r['visibility']=='1' and not r['torch_imported'] for r in results)

    def test_fixed_scope_default_cuda_and_no_backend_identity_collision(self):
        a=normalize(parser().parse_args(['--run-root',str(self.path/'run'),'--cache-root',str(self.path/'cache')]))
        assert a.backend=='cuda' and a.stage=='preflight'
        assert (p.CONFIG['Source_images'],p.CONFIG['Source_calibration'],p.CONFIG['evaluation_images'])==(455,66,289)
        gpu=p.execution(a);a.backend='cpu';cpu=p.execution(a)
        assert ip.st.digest(gpu)!=ip.st.digest(cpu)
        assert gpu['retrieval']['backend']!='OpenCV FLANN LSH' and gpu['retrieval']['approximate']
        assert not gpu['retrieval']['bucket_truncation'] and not gpu['custom_Triton']
        a._identity='id';row=dict(relative_path='x',image_sha256='a',image_height=64,image_width=64)
        old=rt.result_path(a,row);a.backend='cuda';assert rt.result_path(a,row)!=old
        assert len(p.cuda_index_config()['bit_positions'])==4
        assert all(len(set(bits))==20 for bits in p.cuda_index_config()['bit_positions'])

    def test_torch_pixels_bits_boundaries_and_no_center_dependency(self):
        boxes=[(0,0),(64,64),(128,128)];batch=px.outer_batch(self.t,boxes);host=px.outer_to_host(batch)
        for box,actual in zip(boxes,host):
            expected=cg.prepare_outer(self.gray,box)
            for key in ('points','bits','directions','contrast','validation_pairs','validation_bits'):np.testing.assert_array_equal(expected[key],actual[key])
        changed=self.t.clone();changed[64:128,64:128]=1-changed[64:128,64:128]
        new=px.outer_to_host(px.outer_batch(changed,[(64,64)]))[0]
        for key in ('points','bits','validation_bits'):np.testing.assert_array_equal(host[1][key],new[key])
        for version in range(9):
            spec=cp.grid_spec(self.gray.shape,version)
            np.testing.assert_array_equal(cp.describe_grid(self.gray,spec),px.describe_grid(self.t,spec,chunk=43).numpy())
        points=np.array([[-.1,0],[0,0],[191,191],[192,0],[63.5,63.5]],np.float32)
        a,av=cp.sample(self.gray,points);b,bv=px.sample(self.t,torch.from_numpy(points))
        np.testing.assert_array_equal(a,b.numpy());np.testing.assert_array_equal(av,bv.numpy())

    def test_cpu_scalar_cuda_division_shortcut_regression(self):
        # Emulate the documented CUDA CPU-scalar branch while running only CPU.
        original=torch.Tensor.__truediv__
        def cuda_scalar_shortcut(value,divisor):
            if isinstance(divisor,(int,float)):return value*float(np.float32(1./divisor))
            assert divisor.device==value.device and divisor.dtype==value.dtype
            return original(value,divisor)
        rng=np.random.default_rng(128);rng.random((192,192),dtype=np.float32)
        rgb=rng.integers(0,256,(24,32,3),np.uint8);a=rgb.astype(np.float32)
        numerator=a[...,0]*np.float32(.299)+a[...,1]*np.float32(.587)+a[...,2]*np.float32(.114)
        assert np.count_nonzero(numerator/np.float32(255)!=numerator*np.float32(1./255))==531
        with patch.object(torch.Tensor,'__truediv__',new=cuda_scalar_shortcut):
            np.testing.assert_array_equal(cp.grayscale(rgb),px.grayscale(torch.from_numpy(rgb)).numpy())
            for divisor in (9.,24.,255.):
                values=rng.random(256,dtype=np.float32)*np.float32(divisor)
                np.testing.assert_array_equal(values/np.float32(divisor),px.divide_by_constant(torch.from_numpy(values),divisor).numpy())
            # Exercise the actual evidence path under the emulated scalar shortcut.
            refs=self.refs();pool=self.pool();item=dict(source=pool,multi=pool,single=dict(candidates=pool['candidates'][:1],search_unresolved=False))
            normal=ee.normal_expectations([item],refs,[(64,64)],lambda i:self.t,'cpu')
            changed=self.gray.copy();changed[64:128:3,64:128:5]=1-changed[64:128:3,64:128:5]
            result=ee.query_evidence(torch.from_numpy(changed),normal,[(64,64)])
            for k,name in enumerate(('source','single','multi')):
                expected=ce.query_evidence(changed,ce.normal_expectation(item[name],refs,(64,64),multi=name!='single'),(64,64))
                np.testing.assert_array_equal(expected['evidence'],result['evidence'][0,k].numpy())
                np.testing.assert_array_equal(expected['flip_counts'],result['flip_counts'][0,k].numpy())
                assert expected['score']==float(result['scores'][0,k])

    def test_gpu_sampler_never_reads_rejected_query_center_support(self):
        original=px.sample
        def checked(gray,points):
            coords,weights,valid=px.support(points,gray.shape)
            box=torch.tensor([64,64])
            assert (px.annular(coords,box,gray.shape)|(weights==0)).all()
            assert valid.all()
            return original(gray,points)
        with patch.object(px,'sample',side_effect=checked):px.outer_batch(self.t,[(64,64)])

    def test_integer_hamming_all_pairs_and_tail_of_large_bucket(self):
        rng=np.random.default_rng(2);values=rng.integers(0,256,(63,16),np.uint8);v=torch.from_numpy(values)
        expected=np.unpackbits(values[:,None]^values[None],axis=-1).sum(-1)
        np.testing.assert_array_equal(expected,ix.hamming(v[:,None],v[None]).numpy())
        positions=p.cuda_index_config()['bit_positions'];free=next(i for i in range(128) if i not in {b for bits in positions for b in bits})
        descriptors=torch.zeros((64,16),dtype=torch.uint8);descriptors[:63,free//8]=1<<(free%8)
        offsets,order=ix.build_csr(descriptors,positions)
        result,visits=ix.probe(torch.zeros((1,16),dtype=torch.uint8),descriptors,offsets,order,torch.arange(64),positions,budget=4,tile=5)
        assert visits==64*4 and int(result[2][0])==63 and int(result[1][0])==0

    def test_bucket_rerank_matches_independent_reference_and_global_shards(self):
        positions=p.cuda_index_config()['bit_positions'];rng=np.random.default_rng(12)
        values=rng.integers(0,256,(54,16),np.uint8);values[30:]=values[:24]
        query=values[[0,4,8]];physical=np.arange(len(values))%30
        prior=None
        for begin,stop in ((0,27),(27,54)):
            tensor=torch.from_numpy(values[begin:stop]);offsets,order=ix.build_csr(tensor,positions)
            prior,_=ix.probe(torch.from_numpy(query),tensor,offsets,order,torch.from_numpy(physical[begin:stop]),positions,row_start=begin,budget=4,tile=7,prior=prior)
        qi,dist,rows,phys=[x.numpy() for x in prior]
        for j,q in enumerate(query):
            found=set()
            for table in positions:
                def keys(x):return (((x[:,np.array(table)//8]>>(np.array(table)%8))&1)*(1<<np.arange(20))).sum(1)
                reference=keys(values);key=keys(q[None])[0]
                probes={int(key),*[int(key)^(1<<b) for b in range(20)]}
                found.update(i for i,k in enumerate(reference) if int(k) in probes)
            all_rows=sorted(found,key=lambda i:(int(np.unpackbits(values[i]^q).sum()),i));unique=[];seen=set()
            for row in all_rows:
                if physical[row] not in seen:seen.add(physical[row]);unique.append(row)
            assert rows[qi==j].tolist()==unique[:4]

    def test_physical_mapping_for_partial_version_blocks(self):
        store=ix.Store.__new__(ix.Store)
        store.rows=[dict(instance_id='same'),dict(instance_id='same'),dict(instance_id='other')]
        store.record=dict(canonical={'same':0,'other':1},physical_height=32,physical_stride=32)
        blocks=[dict(start=0,end=6,x0=8,y0=8,nx=3,ref=0),dict(start=6,end=12,x0=8,y0=8,nx=3,ref=1),
            dict(start=12,end=18,x0=8,y0=8,nx=3,ref=2)]
        actual=store.physical(dict(start=3,rows=13,blocks=blocks),'cpu').numpy()
        expected=[]
        for row in range(3,16):
            b=next(b for b in blocks if b['start']<=row<b['end']);n=row-b['start']
            expected.append(((b['ref']==2)*32+(b['y0']+n//3*8)//8)*32+(b['x0']+n%3*8)//8)
        np.testing.assert_array_equal(actual,expected)
        assert actual[0]==actual[6] and actual[3]!=actual[9]

    def test_batched_geometry_states_and_cached_original_rng(self):
        refs=self.refs();box=(64,64);info=cg.prepare_outer(self.gray,box)
        matches=[dict(anchor=j,ref=i,point=point.tolist(),version=4,hamming=0) for i in range(2) for j,point in enumerate(info['points'])]
        old=cg.freeze(info,matches,refs,box);cache=OrderedDict()
        actual=gg.freeze_batch([info],[matches],refs,[box],lambda i:self.t,'cpu',group_batch=2,hypothesis_batch=7,triple_cache=cache)[0]
        assert actual['stats']==old['stats'] and len(actual['multi']['candidates'])==2
        again=gg.freeze_batch([info],[matches],refs,[box],lambda i:self.t,'cpu',triple_cache=cache)[0]
        for name in ('source','single','multi'):
            assert len(old[name]['candidates'])==len(actual[name]['candidates'])==len(again[name]['candidates'])
            for first,second in zip(old[name]['candidates'],actual[name]['candidates']):np.testing.assert_allclose(first['T'],second['T'],rtol=0,atol=1e-10)
        for key,triples in cache.items():
            box,identity,n=key;rng=ip.st.rng('annular_hypotheses_v1',42,list(box),identity)
            np.testing.assert_array_equal(triples,np.array([rng.choice(n,3,replace=False) for _ in range(128)]))

    def test_same_candidates_evidence_ambiguity_duplicate_and_empty(self):
        opposite=self.gray.copy();opposite[64:128,64:128]=1-opposite[64:128,64:128]
        for images in ([self.gray,self.gray.copy()],[self.gray,opposite]):
            refs=self.refs(images);item=dict(source=self.pool(),single=dict(candidates=self.pool()['candidates'][:1],search_unresolved=False),multi=self.pool())
            normal=ee.normal_expectations([item],refs,[(64,64)],lambda i:torch.from_numpy(images[i]),'cpu',candidate_batch=1)
            result=ee.query_evidence(torch.from_numpy(opposite),normal,[(64,64)])
            for k,name in enumerate(('source','single','multi')):
                expected=ce.normal_expectation(item[name],refs,(64,64),multi=name!='single');value=ce.query_evidence(opposite,expected,(64,64))
                np.testing.assert_array_equal(expected['agreed'],normal['agreed'][0,k].numpy())
                np.testing.assert_array_equal(value['flip_counts'],result['flip_counts'][0,k].numpy())
                assert value['score']==float(result['scores'][0,k])
        refs=self.refs();refs[1]['instance_id']='0';pool=self.pool();pool['candidates'][1]['instance_id']='0'
        item=dict(source=pool,multi=pool,single=dict(candidates=[],search_unresolved=False))
        normal=ee.normal_expectations([item],refs,[(64,64)],lambda i:self.t,'cpu')
        result=ee.query_evidence(self.t,normal,[(64,64)]);assert not result['evidence_present'].any() and not result['scores'].any()

    def test_noninteger_affine_geometry_and_same_candidate_sampling(self):
        yy,xx=np.mgrid[:192,:192];gray=((xx+yy*.125)/256).astype(np.float32);tensor=torch.from_numpy(gray)
        refs=self.refs([gray,gray.copy()]);box=(64,64);info=cg.prepare_outer(gray,box)
        T=np.array([[1.02,.01,.25],[.005,.99,.5]])
        matches=[dict(anchor=j,ref=i,point=cg.apply(T,point).tolist(),version=4,hamming=0)
            for i in range(2) for j,point in enumerate(info['points'])]
        old=cg.freeze(info,matches,refs,box);new=gg.freeze_batch([info],[matches],refs,[box],lambda i:tensor,'cpu')[0]
        assert old['stats']==new['stats'] and old['qualified']==new['qualified']
        assert len(new['multi']['candidates'])==2
        for first,second in zip(old['multi']['candidates'],new['multi']['candidates']):
            np.testing.assert_allclose(first['T'],second['T'],atol=1e-10,rtol=0)
        # Separate fixed-candidate parity from tiny differences between LAPACK solvers.
        normal=ee.normal_expectations([old],refs,[box],lambda i:tensor,'cpu')
        result=ee.query_evidence(tensor,normal,[box])
        for k,name in enumerate(('source','single','multi')):
            expectation=ce.normal_expectation(old[name],refs,box,multi=name!='single');expected=ce.query_evidence(gray,expectation,box)
            np.testing.assert_array_equal(expectation['agreed'],normal['agreed'][0,k].numpy())
            np.testing.assert_array_equal(expected['flip_counts'],result['flip_counts'][0,k].numpy())
            assert expected['score']==float(result['scores'][0,k])

    def test_original_batch_integration_and_sparse_decode(self):
        from DINOv3.MADEqual.aebad_s_annular_ordinal_p0.runtime_cuda import Engine as TensorEngine
        refs=self.refs();positions=[(64,64)];info=cg.prepare_outer(self.gray,positions[0])
        # Retrieval is deliberately fixed: only identical-candidate batch semantics are tested.
        answers=[([(0,i,*point.tolist(),4) for i in range(2)],False) for point in info['points']]
        store=SimpleNamespace(references=refs,rows=[dict(index=i) for i in range(2)],device='cpu',timers=Timers('cpu'),
            lookup=lambda bits:answers,lookup_many=lambda bits:answers,image=lambda row:self.t,
            matches=lambda info,values:ix.CPUStore.matches(None,info,values))
        a=SimpleNamespace(window_batch=1,geometry_groups=2,hypothesis_batch=16,prefetch_depth=2)
        cpu=rt.Engine(a,store);gpu=TensorEngine.__new__(TensorEngine);gpu.a=a;gpu.store=store;gpu.query=self.t;gpu.triples=OrderedDict()
        dense,summary=cpu.compute_batch(self.gray,positions,{},False,0,positions)
        encoded,actual=gpu.compute_batch(self.gray,positions,{},False,0,positions)
        decoded=codec.decode({k:v.numpy() for k,v in encoded.items()})
        for key in dense:
            if key=='transforms':np.testing.assert_allclose(dense[key],decoded[key],atol=1e-10,rtol=0)
            else:np.testing.assert_array_equal(dense[key],decoded[key],err_msg=key)
        assert actual==summary

    def test_cost_uses_current_index_work_warm_rate_and_one_completion_read(self):
        a=SimpleNamespace(run_root=self.path,backend='cuda')
        measured=[dict(windows=10,seconds=s,bytes=100) for s in (8.,2.,2.,2.)]
        ip.st.sealed_json(self.path/'smoke.json',dict(images=measured,cold_index_load_seconds=4.))
        (self.path/'index').mkdir();ip.st.sealed_json(self.path/'index/manifest.json',dict(seconds=1000.))
        ip.st.write_json(self.path/'index_preparation_attempts.json',[dict(seconds=3.)])
        ip.st.sealed_json(self.path/'gpu_check.json',dict(seconds=1.,observed_engine_startup_seconds=1.))
        rows=[dict(sample_id=str(i),image_height=64,image_width=64) for i in range(3)]
        assets=SimpleNamespace(rows=lambda kind:rows[:2] if kind=='calibration' else rows[2:])
        def completed(a,row):return dict(seconds=5.) if row['sample_id']=='0' else None
        with patch.object(ip,'assets',return_value=assets),patch.object(pl,'index_root',return_value=self.path/'index'),patch.object(rt,'completed',side_effect=completed) as calls:
            result=pl.estimate(a)
        assert calls.call_count==3 and result['index_build_seconds']==3. and result['index_historical_build_seconds']==1000.
        assert result['seconds_per_window']==.2 and result['measured_compressed_bytes_per_window']==10.
        assert result['projected_pixel_total_seconds']==14.4 and result['pending_images']==2

    def dense_arrays(self,n=2):
        return dict(boxes=torch.tensor([[0,0],[32,0]][:n],dtype=torch.int32),scores=torch.zeros((n,3)),evidence_present=torch.zeros((n,3),dtype=torch.bool),
            missing_mask_packed=torch.full((n,3,12288),255,dtype=torch.uint8),flip_counts=torch.zeros((n,3,64,64),dtype=torch.uint8),
            candidate_references=torch.full((n,3,12),-1,dtype=torch.int16),transforms=torch.zeros((n,3,12,2,3),dtype=torch.float64),
            agreement_fraction=torch.zeros((n,3)),disagreement_fraction=torch.zeros((n,3)),distinct_references=torch.zeros((n,3),dtype=torch.int16))

    def test_lossless_sparse_roundtrip_defaults_and_nondefault_rows(self):
        dense=self.dense_arrays();dense['missing_mask_packed'][1,2,7]=10;dense['flip_counts'][0,1,3,4]=23
        dense['candidate_references'][1,0,0]=4;dense['transforms'][1,0,0]=torch.tensor([[1.,0.,3.],[0.,1.,4.]])
        encoded={k:v.numpy() for k,v in codec.encode_tensors(dense).items()}
        codec.check_layout(encoded,dense['boxes'].tolist());decoded=codec.decode(encoded)
        for key,value in dense.items():np.testing.assert_array_equal(value.numpy(),decoded[key],err_msg=key)
        assert len(encoded['mask_rows'])==1 and len(encoded['flip_rows'])==1 and len(encoded['candidate_rows'])==1

    def test_async_commit_persisted_validation_and_corruption(self):
        dense=self.dense_arrays();arrays=codec.encode_tensors(dense);a=SimpleNamespace();writer=codec.Writer(2)
        summary=dict(identity='x',index='i',boxes=dense['boxes'].tolist())
        record=writer.submit(a,self.path,'chunk_0.json',arrays,summary,time.perf_counter()).result();writer.close()
        with patch.object(np,'load',side_effect=AssertionError('unchanged payload unnecessarily reread')):rt.validate_chunk(SimpleNamespace(),self.path,record)
        file=self.path/record['arrays'];file.write_bytes(b'')
        with self.assertRaises(ValueError):rt.validate_chunk(a,self.path,record)

    def test_bounded_writer_failure_is_not_a_completion(self):
        writer=codec.Writer(1);arrays=codec.encode_tensors(self.dense_arrays());summary=dict(identity='x',index='i',boxes=arrays['boxes'].tolist())
        with patch.object(codec,'save',side_effect=OSError('synthetic disk failure')):
            future=writer.submit(SimpleNamespace(),self.path,'chunk_0.json',arrays,summary,time.perf_counter())
            with self.assertRaises(OSError):future.result()
        with self.assertRaises(OSError):writer.close()
        assert not (self.path/'chunk_0.json').exists()

    def test_tensor_cache_prevents_per_candidate_image_upload(self):
        cache=TensorCache('cpu',1024,Timers('cpu'));loader=lambda:np.ones((8,8),np.float32)
        first=cache.get('same',loader)
        with patch.object(np,'asarray',side_effect=AssertionError('image reloaded')):
            assert cache.get('same',loader) is first
        assert cache.uploads==1 and cache.hits==1

    def test_no_cuda_fallback_and_small_checks_are_labeled_cpu(self):
        with patch.object(torch.cuda,'is_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'no CPU fallback'):require_device('cuda:0')
        assert fingerprint('cpu')['gpu_executed'] is False
        checks=small_checks('cpu');assert checks['Hamming']=='integer exact' and checks['masks_and_flips']=='exact'
        assert not torch.cuda.is_initialized()

if __name__=='__main__':unittest.main()
