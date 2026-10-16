"""Small independent numerical references and explicit model/metric doubles."""
from __future__ import annotations
import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
import numpy as np
from DINOv3.MADEqual.aebad_s_fastref_k8_p0 import refinement as math, inputs, runtime, analysis, runner, sampling
from DINOv3.tests.test_mad_equal_aebad_s_sparc_union_p0 import analysis_fixture, Tensor, torch_fixture
from DINOv3.tests.test_mad_equal_aebad_s_target_local_segment import full_protocol_fixture
from DINOv3.MADEqual.aebad_s_sparc_p0 import runtime as io_runtime


def ordinary_sinkhorn_reference(cost, epsilon):
    kernel=np.exp(-cost/epsilon); m,n=cost.shape; u=np.ones(m);v=np.ones(n)
    for _ in range(20000):
        u=(1/m)/(kernel@v);v=(1/n)/(kernel.T@u)
        t=u[:,None]*kernel*v[None,:]
        if max(np.abs(t.sum(1)*m-1).max(),np.abs(t.sum(0)*n-1).max())<1e-11:return t
    raise AssertionError('independent reference failed to converge')


def flow_fixture(root):
    old,_,backends=analysis_fixture(root)
    parent=Path(old['paths']['parent']);cap_dir=Path(old['paths']['capture'])
    cap=json.loads((cap_dir/'inputs.json').read_text());captured=json.loads((cap_dir/'status.json').read_text())
    out=root/'fastref';out.mkdir()
    frozen=dict(config=dict(math.CONFIG,cpu_workers=1),sampling=dict(inputs.SAMPLING),capture_identity=cap['identity'],
        precision=dict(float32_matmul_precision='highest',cuda_matmul_allow_tf32=False,cudnn_allow_tf32=True))
    data=dict(tool=inputs.TOOL,identity=inputs.digest(frozen),frozen=frozen,paths=dict(old['paths'],output=str(out),mad=str(Path(old['paths']['results'])/'source_mad_state.npz')),
        capture=cap,capture_status=captured,required_files=[])
    inputs.replay.atomic_json(out/'inputs.json',data)
    prepared=dict(status='prototypes_prepared',identity=data['identity'],artifacts=[],sampling=dict(inputs.SAMPLING))
    inputs.replay.atomic_json(out/'prepare_status.json',prepared)
    records=[];checks=[]
    for sample,record in zip(cap['samples'],captured['records'],strict=True):
        with np.load(cap_dir/record['file']) as f:
            meta=json.loads(str(f['metadata'].item()));s=f['source_distance'].copy();t=f['target_distance'].min(0);p=f['positions'].copy()
        arrays=dict(source_distance=s,target_full_distance=t,prototype_distance=t*1.2,refined_distance=t*.9,
            prototype_union_distance=np.minimum(s,t*1.2),fastref_union_distance=np.minimum(s,t*.9),
            prototype_index=np.zeros_like(s,np.int64),refined_index=np.zeros_like(s,np.int64),positions=p,
            grid=np.array(meta['grid'],np.int32),processed_shape=np.array([meta['geometry']['processed_height'],meta['geometry']['processed_width']],np.int32))
        m=dict(tool=inputs.TOOL,identity=data['identity'],sample_id=sample['sample_id'],relative_path=sample['relative_path'],
            image_sha256=meta['image_sha256'],historical_raw_sha256=record['sha256'],geometry=meta['geometry'],grid=meta['grid'],output_shape=meta['output_shape'],
            distance_divisor=1280,prepare_sha256=inputs.replay.sha256(out/'prepare_status.json'),query_forward_calls=1,refined_rows=len(p),query_features_persisted=False,refined_vectors_persisted=False)
        file=out/'raw'/f"{sample['sample_id']}.npz";io_runtime.save_npz(file,metadata=np.asarray(json.dumps(m)),**arrays)
        records.append(dict(sample_id=sample['sample_id'],**inputs.file_records(out,[file])[0]))
    for sid,kind,layer,outer in sorted(runtime.expected_checks(cap['samples'])):
        checks.append(dict(sample_id=sid,check_kind=kind,layer=layer,outer=outer,verified=True))
    inputs.replay.write_csv(out/'infer_numerical_verification.csv',checks)
    diagnostic = dict(gram_rank=512,gram_condition=1.,solve_relative_residual=0.,outer=[dict(
        outer_iteration=i,row_max_relative_error=0.,column_max_relative_error=0.,total_mass_error=0.,
        update_normal_equation_relative_residual=0.,nonfinite_entries=0,transport_rescaled_by_m=False,lambda_value=.3) for i in (1,2)])
    inputs.replay.atomic_json(out/'infer_solver_diagnostics.json',[
        dict(sample_id=s['sample_id'],layer=l,**diagnostic) for s in cap['samples'] for l in inputs.LAYERS])
    inputs.replay.atomic_json(out/'infer_status.json',dict(tool=inputs.TOOL,identity=data['identity'],status='predictions_frozen',images=297,records=records,
        inputs_sha256=inputs.replay.sha256(out/'inputs.json'),prepare_sha256=inputs.replay.sha256(out/'prepare_status.json'),verification_sha256=inputs.replay.sha256(out/'infer_numerical_verification.csv'),
        solver_diagnostics_sha256=inputs.replay.sha256(out/'infer_solver_diagnostics.json')))
    # Frozen original SOURCE/K8 toy values; pixels are explicitly synthetic.
    rows=inputs.replay.read_csv(parent/'evaluation/metrics.csv')
    metrics=[dict(r,branch='SOURCE' if r['branch']=='SOURCE' else 'UNION_K8',**{k:float(r[k]) for k in inputs.METRIC_KEYS}) for r in rows[:2]]
    (Path(data['paths']['results'])/'domain_metrics.csv').write_text('fixture',encoding='utf-8')
    return data,prepared,backends['SOURCE'],metrics


def prototype_fixture(root):
    """Only synthetic artifact/header fixtures; never run a real sampler."""
    capture=root/'capture';capture.mkdir();output=root/'prepared';output.mkdir()
    paths=dict(output=str(output),capture=str(capture),selector=str(root/'sampler.py'),selector_backend=str(root/'backend.py'),selector_upstream=str(root/'upstream.py'))
    for key in ('selector','selector_backend','selector_upstream'):Path(paths[key]).write_text('fixture',encoding='utf-8')
    supports=[dict(support_index=i+1,relative_path=f'test/good/background/s{i}.png',grid=[10,10],
        feature_sha256=f'feature{i}',image_sha256=f'image{i}') for i in range(8)]
    cap=dict(identity=inputs.CAPTURE_ID,supports=supports);status=dict(status='captured')
    inputs.replay.atomic_json(capture/'inputs.json',cap);inputs.replay.atomic_json(capture/'status.json',status)
    frozen=dict(config=dict(math.CONFIG,cpu_workers=1),sampling=dict(inputs.SAMPLING),capture_identity=inputs.CAPTURE_ID)
    data=dict(tool=inputs.TOOL,identity=inputs.digest(frozen),frozen=frozen,paths=paths,
        required_files=[],capture=cap,capture_status=status)
    inputs.replay.atomic_json(output/'inputs.json',data)
    state=dict(status='prototypes_prepared',identity=data['identity'],sampling=dict(inputs.SAMPLING),prototype_ids={},
        inputs_sha256=inputs.replay.sha256(output/'inputs.json'))
    indices=np.arange(512,dtype=np.int64);p=np.zeros((512,1280),np.float32);p[:,0]=indices
    files=[]
    for layer in inputs.LAYERS:
        metadata=inputs.prototype_metadata(data,layer);identity=inputs.prototype_identity(metadata,p,indices)
        state['prototype_ids'][str(layer)]=identity;file=output/'prototypes'/f'L{layer}.npz'
        io_runtime.save_npz(file,metadata=np.asarray(json.dumps(metadata)),prototype_identity=np.asarray(identity),
            prototypes=p,factor=np.zeros_like(p,np.float64),indices=indices,epsilon=np.asarray(1.),diagnostics=np.asarray('{}'))
        files.append(file)
    state['artifacts']=inputs.file_records(output,files);inputs.replay.atomic_json(output/'prepare_status.json',state)
    return data,state


def numpy_knn_backend(features, neighbors, device):
    """Explicit test distance backend, not another production sampler."""
    x=features.astype(np.float64)
    distances=np.sqrt(np.sum((x[:,None]-x[None])**2,axis=2)).astype(np.float32)
    return np.sort(distances,axis=1)[:,:neighbors]


def prepare_torch_double(distance_backend):
    """Explicit fake CUDA/tensor bridge; never import real Torch."""
    fake,_=torch_fixture();fake.device=lambda *a:SimpleNamespace(type='cuda',index=0)
    fake.cuda=SimpleNamespace(is_available=lambda:True,set_device=lambda d:None,current_device=lambda:0,reset_peak_memory_stats=lambda d:None)
    fake.set_float32_matmul_precision=lambda value:None
    fake.backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),cudnn=SimpleNamespace(allow_tf32=True))
    def distances(q,k,neighbors,normalize,**kw):return Tensor(distance_backend(q.numpy(),neighbors,'double')),None
    return {'torch':fake,'external_baselines.superadd_external.memory_variants':SimpleNamespace(_chunked_exact_nearest_neighbors=distances)}


class SamplingTests(unittest.TestCase):
    def test_actual_production_control_and_lazy_distance_bridge_select_original_rows(self):
        bank=np.zeros((600,1280),np.float32);bank[:,0]=np.arange(600);before=bank.copy()
        calls=[];fake,_=torch_fixture()
        def original_distance(q,k,neighbors,normalize,**kwargs):
            self.assertIs(q,k);calls.append((neighbors,normalize,kwargs))
            return Tensor(numpy_knn_backend(q.numpy(),neighbors,'double')),None
        module=SimpleNamespace(_chunked_exact_nearest_neighbors=original_distance)
        with patch.dict(sys.modules,{'torch':fake,'external_baselines.superadd_external.memory_variants':module}):
            indices,report=runtime.select_prototypes(bank,'explicit-device-double',rng=np.random.RandomState(42))
        self.assertEqual(len(indices),512);self.assertEqual(len(np.unique(indices)),512)
        self.assertEqual(len(calls),100);self.assertTrue(all(c[0]==6 and c[1] is False for c in calls))
        self.assertTrue(all(c[2]['max_distance_matrix_bytes']==1<<30 for c in calls))
        np.testing.assert_array_equal(bank,before);np.testing.assert_array_equal(bank[indices,0],indices)
        self.assertTrue(report['all_candidates_visited_once']);self.assertFalse(report['per_group_quotas'])
        self.assertEqual(report['radius_iterations'],0)

    def test_budget_smaller_than_rounds_and_indivisible_budget_visit_entire_pool_once(self):
        for count,quota,rounds in ((37,3,100),(57,11,10),(8,8,100),(1,1,100)):
            features=np.column_stack((np.arange(count),np.arange(count)*2)).astype(np.float32);seen=[]
            def backend(values,k,device):seen.extend(values[:,0].astype(int));return numpy_knn_backend(values,k,device)
            ids,report=sampling.sample_indices(features,quota,'double',rng=np.random.RandomState(42),rounds=rounds,distance_backend=backend)
            self.assertEqual(len(ids),quota);self.assertEqual(len(np.unique(ids)),quota)
            self.assertEqual(sorted(seen),list(range(count)))
            self.assertEqual(sum(g['candidates'] for g in report['groups']),count)
            self.assertLessEqual(max(g['candidates'] for g in report['groups'])-min(g['candidates'] for g in report['groups']),1)
            self.assertTrue(all(g['neighbors']==min(100,g['candidates']) for g in report['groups']))
            self.assertEqual([r['candidate_index'] for r in report['selected']],ids.tolist())

    def test_insufficient_candidates_and_invalid_data_fail_before_sampling_or_rng_change(self):
        for features,quota in ((np.zeros((4,6),np.float32),5),(np.zeros((4,6),np.float32),0),
                               (np.full((4,6),np.nan,np.float32),2),(np.zeros((4,6),np.float64),2)):
            rng=np.random.RandomState(42);before=rng.get_state()
            with self.assertRaises(ValueError):sampling.sample_indices(features,quota,'double',rng=rng,distance_backend=lambda *a:self.fail('backend called'))
            np.testing.assert_array_equal(before[1],rng.get_state()[1]);self.assertEqual(before[2:],rng.get_state()[2:])
        self.assertEqual(sampling.admission(511)['status'],'blocked')
        self.assertEqual(sampling.admission(2000,1281)['status'],'blocked')

    def test_repeated_zero_features_terminate_with_distinct_indices_but_solver_rejects_rank(self):
        features=np.zeros((30,6),np.float32)
        ids,report=sampling.sample_indices(features,4,'double',rng=np.random.RandomState(42),rounds=5,distance_backend=numpy_knn_backend)
        self.assertEqual(np.unique(ids).size,4);self.assertTrue(report['uniform_weights'])
        self.assertTrue(all(g['radius']==0 for g in report['groups']))
        self.assertEqual(report['retries'],0)
        with self.assertRaises(math.NumericalFailure) as error:math.make_constants(features[ids])
        self.assertFalse(error.exception.diagnostics['fallback_used'])

    def test_local_rng_reproducibility_continuous_layers_and_global_state_unchanged(self):
        features=np.random.default_rng(12).normal(size=(83,7)).astype(np.float32);before=np.random.get_state()
        def sequence():
            rng=np.random.RandomState(42)
            return [sampling.sample_indices(features,17,'double',rng=rng,rounds=9,distance_backend=numpy_knn_backend)[0] for _ in inputs.LAYERS]
        first,second=sequence(),sequence()
        for a,b in zip(first,second):np.testing.assert_array_equal(a,b)
        self.assertFalse(np.array_equal(first[0],first[1]));after=np.random.get_state()
        np.testing.assert_array_equal(before[1],after[1]);self.assertEqual(before[2:],after[2:])

    def test_density_preference_and_exact_weighted_order_are_explicit(self):
        weights,density,radius=sampling.density_weights(np.array([[0,0,1],[0,1,4]],np.float32))
        np.testing.assert_array_equal(density,[2,1]);np.testing.assert_allclose(weights,[1/3,1/2])
        self.assertGreater(radius,0)
        ids,keys=sampling.exact_weighted_indices(weights,np.array([.4,.4]),1)
        np.testing.assert_array_equal(ids,[1]);self.assertLess(keys[1],keys[0])
        ties,_=sampling.exact_weighted_indices(np.ones(8),np.zeros(8),3)
        np.testing.assert_array_equal(ties,[0,1,2])
        for weights,uniforms in (([0,1],[.2,.2]),([1,1],[1,.2]),([1,np.nan],[.2,.2])):
            with self.assertRaises(ValueError):sampling.exact_weighted_indices(weights,uniforms,1)

    def test_K8_dimensions_pass_without_sampler_or_global_rng(self):
        state=np.random.get_state();result=inputs.sampling_compatibility(8*118*118)
        self.assertEqual(result['status'],'static_budget_check_passed_runtime_unverified')
        self.assertEqual(result['quota'],512);self.assertEqual(result['effective_rounds'],100)
        after=np.random.get_state();np.testing.assert_array_equal(state[1],after[1]);self.assertEqual(state[2:],after[2:])
        self.assertNotIn('torch',sys.modules)

    def test_actual_upstream_loop_fails_to_progress_on_zero_radius_or_unreachable_mask(self):
        from DINOv3.tests.test_mad_equal_aebad_s_sparc_p0 import actual_functions
        source=Path(__file__).resolve().parents[2]/'external_baselines/superadd_external/memory_variants.py'
        class ObservedNonProgress(Exception):pass
        for zero_radius in (True,False):
            counts=[];calls=[];rng_calls=[]
            def sum_guard(value,*a,**kw):
                result=np.sum(value,*a,**kw)
                if np.asarray(value).ndim==1 and np.asarray(value).dtype==bool:
                    counts.append(int(result))
                    if len(counts)==4:raise ObservedNonProgress()
                return result
            proxy=SimpleNamespace(**{name:getattr(np,name) for name in
                ('asarray','float32','int64','float64','empty','arange','full','where','invert','mean','flatnonzero')})
            proxy.sum=sum_guard
            proxy.random=SimpleNamespace(choice=lambda values,**kw:np.asarray(values)[:kw['size']],
                rand=lambda n:(rng_calls.append(n) or np.full(n,.5 if zero_radius else 0.)),seed=lambda *a:self.fail('unexpected reseed'))
            features=np.zeros((100,100),np.float32) if zero_radius else np.eye(100,dtype=np.float32)
            def distance(q,k,neighbors,normalize,**kw):
                calls.append(1);return Tensor(numpy_knn_backend(q.numpy(),neighbors,'double')),None
            fake,_=torch_fixture();env=dict(np=proxy,torch=fake,_chunked_exact_nearest_neighbors=distance,SAMPLER_DISTANCE_MATRIX_MAX_BYTES=1<<30)
            actual_functions(source,{'_official_subsample_values_with_indices'},env)
            with self.assertRaises(ObservedNonProgress):env['_official_subsample_values_with_indices'](features,1,'double',seed=None,iterations=1)
            self.assertEqual(counts,[100]*4);self.assertEqual(len(calls),1);self.assertEqual(rng_calls,[100])

    def test_fixed_mask_can_never_reach_five_even_when_all_100_neighbors_are_inside(self):
        # Synthetic mathematical bound, not execution of the real sampler.
        draws=np.full(1113,.5);draws[:11]=.001
        minimum_count=int(np.count_nonzero(draws < 1/101))
        self.assertEqual(minimum_count,11);self.assertGreater(minimum_count,512//100)
        for neighbors_inside in (0,1,50,100):
            self.assertGreater(int(np.count_nonzero(draws < 1/(neighbors_inside+1))),5)
        # Independent RNG arithmetic for frozen K8 dimensions; no features,
        # kNN, real sampler call or prototype construction is performed.
        rng=np.random.RandomState(42);rng.choice(111392,size=1113,replace=False)
        first_mask_floor=int(np.count_nonzero(rng.rand(1113) < 1/101))
        self.assertEqual(first_mask_floor,9);self.assertGreater(first_mask_floor,5)

    def test_prepared_common_prototypes_keep_source_indices_order_and_identity(self):
        with tempfile.TemporaryDirectory() as td:
            data,state=prototype_fixture(Path(td));loaded,prepared=inputs.read_prepared(data['paths']['output'])
            constants=runtime.load_constants(loaded,prepared)
            fake,host=torch_fixture();actual=host._exact_top1_chunked
            host._exact_top1_chunked=lambda q,k,**kw:actual(q,k)
            for layer in inputs.LAYERS:
                constant=constants[layer];p=constant['prototypes'];q=p[[5,20]].copy()
                with patch.dict(sys.modules,{'torch':fake,'DINOv3.relation_reliability.superadd_host':host}):
                    base,index=runtime.host_nn(Tensor(q),Tensor(p),compact=True)
                    disabled,_=math.refine(q,constant,enabled=False)
                    same,same_index=runtime.host_nn(Tensor(q),Tensor(disabled),compact=True)
                np.testing.assert_array_equal(base,same);np.testing.assert_array_equal(index,same_index)
                np.testing.assert_array_equal(constant['indices'],np.arange(512));np.testing.assert_array_equal(p[:,0],constant['indices'])
                self.assertEqual(constant['prototype_identity'],prepared['prototype_ids'][str(layer)])
                self.assertEqual(inputs.prototype_metadata(data,layer)['shared_branches'],list(inputs.NEW))

    def test_old_sampling_artifacts_rejected_even_if_their_new_hashes_are_recorded(self):
        for old_tool in ('aebad_s_fastref_target512_source_k8_v1','aebad_s_fastref_target512_source_k8_v2_original_sampler'):
            with tempfile.TemporaryDirectory() as td:
                data,state=prototype_fixture(Path(td));root=Path(data['paths']['output'])
                data['tool']=old_tool;data['frozen']['sampling']['revision']='old_sampling'
                data['identity']=inputs.digest(data['frozen']);inputs.replay.atomic_json(root/'inputs.json',data)
                state['identity']=data['identity'];state['inputs_sha256']=inputs.replay.sha256(root/'inputs.json')
                inputs.replay.atomic_json(root/'prepare_status.json',state)
                with self.assertRaisesRegex(ValueError,'old PatchCore/v2'):inputs.read_prepared(root)

    def test_nonfinite_distance_backend_fails_without_retry_or_alternative(self):
        calls=[]
        def backend(values,k,device):calls.append(1);return np.full((len(values),k),np.nan)
        with self.assertRaisesRegex(ValueError,'finite nonnegative'):
            sampling.sample_indices(np.ones((30,4),np.float32),5,'double',rng=np.random.RandomState(42),distance_backend=backend)
        self.assertEqual(calls,[1])

    def test_successful_prepare_uses_actual_cached_reader_sampler_and_solver_and_one_frozen_P(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);data,_=prototype_fixture(root);output=root/'fresh';output.mkdir();support=root/'supports';support.mkdir()
            data['paths'].update(output=str(output),support=str(support))
            pool=np.zeros((800,1280),np.float32);pool[np.arange(800),np.arange(800)]=1280
            for row in data['capture']['supports']:
                index=row['support_index'];file=support/f'{index:02d}.npz';bank=pool[(index-1)*100:index*100]
                io_runtime.save_npz(file,protocol_id=np.asarray(inputs.replay.PROTOCOL_ID),relative_path=np.asarray(row['relative_path']),
                    grid=np.array([10,10],np.int32),**{f'layer_{l}':bank for l in inputs.LAYERS})
                row['feature_sha256']=inputs.replay.sha256(file)
            inputs.replay.atomic_json(Path(data['paths']['capture'])/'inputs.json',data['capture'])
            data['frozen']['precision']=dict(float32_matmul_precision='highest',cuda_matmul_allow_tf32=False,cudnn_allow_tf32=True)
            data['identity']=inputs.digest(data['frozen']);inputs.replay.atomic_json(output/'inputs.json',data)
            with patch.dict(sys.modules,prepare_torch_double(numpy_knn_backend)),contextlib.redirect_stdout(io.StringIO()):
                state=runtime.prepare(data,'cuda:0')
            self.assertEqual(state['status'],'prototypes_prepared');self.assertEqual(len(state['artifacts']),7)
            loaded,prepared=inputs.read_prepared(output);constants=runtime.load_constants(loaded,prepared)
            records=inputs.replay.read_csv(output/'prototype_provenance.csv');self.assertEqual(len(records),2048)
            for layer,c in constants.items():
                np.testing.assert_array_equal(c['prototypes'],pool[c['indices']])
                self.assertEqual(c['diagnostics']['gram_rank'],512);self.assertFalse(c['diagnostics']['fallback_used'])
                before=c['prototypes'].copy();q=before[[0,1]].copy()
                refined,diag=math.refine(q,c)
                self.assertEqual(refined.shape,(2,1280));self.assertEqual(diag['initial_prototype_rows'],512)
                np.testing.assert_array_equal(c['prototypes'],before)
                for row in [r for r in records if int(r['layer'])==layer]:
                    i=int(row['prototype_index']);flat=(int(row['support_index'])-1)*100+int(row['physical_patch_index'])
                    np.testing.assert_array_equal(before[i],pool[flat])

    def test_real_prepare_control_keeps_sampling_evidence_on_rank_failure(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);output=root/'output';output.mkdir()
            cfg=dict(inputs.SAMPLING);supports=[dict(support_index=i+1,relative_path=f'test/good/background/s{i}.png',
                grid=[5,15],feature_sha256=str(i),image_sha256=str(i)) for i in range(8)]
            paths=dict(output=str(output),selector=str(root/'sampler.py'),selector_backend=str(root/'backend.py'),selector_upstream=str(root/'upstream.py'))
            for key in ('selector','selector_backend','selector_upstream'):Path(paths[key]).write_text('fixture')
            data=dict(tool=inputs.TOOL,identity='fixture',paths=paths,required_files=[],capture=dict(identity=inputs.CAPTURE_ID,supports=supports),
                frozen=dict(sampling=cfg,config=dict(math.CONFIG,cpu_workers=1),precision=dict(float32_matmul_precision='highest',cuda_matmul_allow_tf32=False,cudnn_allow_tf32=True)))
            inputs.replay.atomic_json(output/'inputs.json',data)
            fake,_=torch_fixture();fake.device=lambda *a:SimpleNamespace(type='cuda',index=0)
            fake.cuda=SimpleNamespace(is_available=lambda:True,set_device=lambda d:None,current_device=lambda:0,reset_peak_memory_stats=lambda d:None)
            fake.set_float32_matmul_precision=lambda value:None
            fake.backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),cudnn=SimpleNamespace(allow_tf32=True))
            def distances(q,k,neighbors,normalize,**kw):return Tensor(np.zeros((len(q.numpy()),neighbors),np.float32)),None
            module=SimpleNamespace(_chunked_exact_nearest_neighbors=distances)
            with patch.dict(sys.modules,{'torch':fake,'external_baselines.superadd_external.memory_variants':module}),\
                 patch.object(runtime,'load_target_layer',return_value=[np.zeros((75,1280),np.float32) for _ in range(8)]),\
                 contextlib.redirect_stdout(io.StringIO()),self.assertRaises(math.NumericalFailure):
                runtime.prepare(data,'cuda:0')
            diagnostics=json.loads((output/'sampling_diagnostics.json').read_text())
            self.assertEqual(len(diagnostics),4);self.assertTrue(all(d['selected_rows']==512 and d['retries']==0 for d in diagnostics))
            self.assertEqual(len(inputs.replay.read_csv(output/'prototype_provenance.csv')),2048)
            self.assertFalse((output/'prepare_status.json').exists());self.assertFalse((output/'prototypes').exists())

    def test_mixed_prototype_metadata_is_rejected_after_artifact_rehash(self):
        with tempfile.TemporaryDirectory() as td:
            data,state=prototype_fixture(Path(td));root=Path(data['paths']['output']);file=root/'prototypes/L7.npz'
            with np.load(file) as f:values={k:f[k].copy() for k in f.files}
            metadata=json.loads(str(values['metadata'].item()));metadata['sampling']['random_projection']=True
            values['metadata']=np.asarray(json.dumps(metadata));np.savez_compressed(file,**values)
            state['artifacts']=inputs.file_records(root,list((root/'prototypes').glob('*.npz')))
            inputs.replay.atomic_json(root/'prepare_status.json',state)
            with self.assertRaisesRegex(ValueError,'sampling/input identity'):inputs.read_prepared(root)

    def test_comparison_directions_and_references_are_computed_not_prefilled(self):
        metrics=[dict(branch=branch,**{k:value for k in inputs.METRIC_KEYS}) for branch,value in zip(inputs.BRANCHES,(.1,.7,.5,.6))]
        rows=analysis.metric_comparisons(metrics)
        self.assertEqual([(r['branch'],r['reference']) for r in rows],list(analysis.COMPARISONS))
        self.assertEqual([r['comparison'] for r in rows],['A','B','C'])
        for r,expected in zip(rows,(-.2,.1,-.1)):
            for k in inputs.METRIC_KEYS:self.assertAlmostEqual(r['delta_'+k],expected)
        with self.assertRaises(ValueError):analysis.metric_comparisons(metrics[:3])


class MathTests(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(42);p=rng.normal(size=(3,5)).astype(np.float32);q=rng.normal(size=(6,5)).astype(np.float32)
        config=dict(math.CONFIG,epsilon_support_fraction=1.)
        return q,p,config,math.make_constants(p,config)

    def test_update_matches_independent_explicit_W_and_stationarity(self):
        q,p,c,constants=self.fixture();actual,diag=math.refine(q,constants,config=c)
        x=q.astype(np.float64)/1280;m=p.astype(np.float64)/1280;g=m@m.T
        w0=np.linalg.solve(g,(x@m.T).T).T;r=w0@m
        for _ in range(c['outer_iterations']):
            cost=np.sum((r[:,None]-m[None])**2,axis=2)
            t=ordinary_sinkhorn_reference(cost,constants['epsilon'])
            w=np.empty((len(q),len(p)))
            for i in range(len(q)):
                rhs=x[i]@m.T+c['lambda_value']*t[i]@g
                w[i]=np.linalg.solve(g,rhs)/(1+c['lambda_value']*t[i].sum())
                gradient=2*((1+c['lambda_value']*t[i].sum())*w[i]@g-rhs)
                np.testing.assert_allclose(gradient,0,atol=1e-15)
            r=w@m
        np.testing.assert_allclose(actual,r*1280,atol=1e-6,rtol=1e-6)
        self.assertEqual(diag['refined_rows'],len(q));self.assertTrue(all(not d['transport_rescaled_by_m'] for d in diag['outer']))

    def test_global_OT_probability_margins_and_cost_unit(self):
        cost=np.array([[0.,1.,2.],[.2,.7,1.3],[1.,.2,.1],[2.,1.,0.]])
        t,d=math.balanced_transport(cost,.4)
        np.testing.assert_allclose(t.sum(1),np.full(4,.25),atol=1e-7)
        np.testing.assert_allclose(t.sum(0),np.full(3,1/3),atol=1e-10)
        self.assertAlmostEqual(t.sum(),1.,places=9)
        same,_=math.balanced_transport(cost*1280**2,.4*1280**2)
        np.testing.assert_allclose(t,same,atol=1e-14)
        self.assertEqual(d['global_query_rows'],4)

    def test_conditional_transport_requires_lambda_over_m_for_full_objective(self):
        t=np.full((4,3),1/12);v=4*t;cost=np.arange(12).reshape(4,3)/10;eps=.2;lam=.3;reconstruction=2.5
        original=reconstruction+lam*(np.sum(t*cost)+eps*np.sum(t*np.log(t)))
        equivalent=reconstruction+(lam/4)*(np.sum(v*cost)+eps*np.sum(v*np.log(v)))-lam*eps*np.log(4)
        self.assertAlmostEqual(original,equivalent,places=12)
        wrong=reconstruction+lam*(np.sum(v*cost)+eps*np.sum(v*np.log(v)))
        self.assertNotAlmostEqual(original,wrong,places=5)

    def test_separate_chunk_OT_would_change_the_global_problem(self):
        cost=np.array([[0.,5.],[.1,5.],[5.,0.],[5.,.1]])
        full,_=math.balanced_transport(cost,1.)
        chunks=np.vstack([math.balanced_transport(cost[:2],1.)[0],math.balanced_transport(cost[2:],1.)[0]])/2
        self.assertGreater(np.linalg.norm(full-chunks),.1)
        np.testing.assert_allclose(full.sum(0),[.5,.5],atol=1e-8)

    def test_disabled_returns_P_but_lambda_zero_still_projects_query(self):
        p=np.array([[1.,0.,0.],[0.,1.,0.]],np.float32);q=np.array([[.4,.2,2.]],np.float32)
        constants=math.make_constants(p)
        disabled,d=math.refine(q,constants,enabled=False)
        zero,_=math.refine(q,constants,config=dict(math.CONFIG,lambda_value=0.))
        np.testing.assert_array_equal(disabled,p);np.testing.assert_allclose(zero,[[.4,.2,0]],atol=1e-6)
        self.assertFalse(d['refinement_enabled']);self.assertNotEqual(disabled.shape,zero.shape)

    def test_final_all_refined_NN_is_not_paired_residual_and_divides_once(self):
        fake,host=torch_fixture();actual=host._exact_top1_chunked
        host._exact_top1_chunked=lambda q,k,**kw:actual(q,k)
        q=np.array([[0.,0.],[1280.,0.]],np.float32);r=q[::-1].copy()
        with patch.dict(sys.modules,{'torch':fake,'DINOv3.relation_reliability.superadd_host':host}):
            distances,index=runtime.host_nn(Tensor(q),Tensor(r),compact=True)
            unit,_=runtime.host_nn(Tensor(np.array([[1280.,0.]],np.float32)),Tensor(np.zeros((1,2),np.float32)))
        np.testing.assert_array_equal(distances,[0,0]);np.testing.assert_array_equal(index,[1,0]);self.assertEqual(unit[0],1.)
        self.assertTrue(np.all(np.linalg.norm(q-r,axis=1)==1280))

    def test_query_forward_is_one_complete_original_call_and_keeps_raw_norms(self):
        from types import SimpleNamespace
        from collections import defaultdict
        fake,host=torch_fixture();features=np.ones((4,1280),np.float32)*5
        item=SimpleNamespace(grid=(np.int32(2),np.int32(2)),positions=np.arange(4,dtype=np.int64),queries={l:Tensor(features) for l in inputs.LAYERS})
        calls=[]
        def extract(*a,**kw):calls.append(kw);return [item]
        host._extract_processed_batch=extract;host.image_tensor=lambda image:Tensor(np.zeros((3,32,32),np.float32))
        with patch.dict(sys.modules,{'torch':fake,'DINOv3.relation_reliability.superadd_host':host}):
            q,p,grid,processed=runtime.query_forward(SimpleNamespace(preprocessing=lambda x:x),None,'fixture',defaultdict(float))
        self.assertEqual(len(calls),1);self.assertEqual(grid,(2,2));self.assertTrue(all(type(v) is int for v in (*grid,*processed)))
        for value in q.values():np.testing.assert_array_equal(value,features);self.assertGreater(np.linalg.norm(value[0]),1.)

    def test_per_query_state_and_inputs_are_independent(self):
        q,p,c,k=self.fixture();copies={name:value.copy() for name,value in k.items() if isinstance(value,np.ndarray)}
        before=q.copy();a,_=math.refine(q,k,config=c);math.refine(q*2,k,config=c);b,_=math.refine(q,k,config=c)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(q,before)
        for name,value in copies.items():np.testing.assert_array_equal(k[name],value)

    def test_singular_and_nonfinite_inputs_fail_without_fallback(self):
        for p in (np.ones((2,3),np.float32),np.array([[1,np.nan],[0,1]],np.float32)):
            with self.assertRaises(math.NumericalFailure):math.make_constants(p)
        with self.assertRaises(math.NumericalFailure):math.balanced_transport(np.ones((2,3)),0.)

    def test_nonconverged_OT_preserves_numeric_failure(self):
        with self.assertRaises(math.NumericalFailure) as error:
            math.balanced_transport(np.array([[0.,10,20],[10,0,2.]]),.01,dict(math.CONFIG,sinkhorn_max_iterations=1,sinkhorn_check_every=1))
        self.assertEqual(error.exception.diagnostics['sinkhorn_iterations'],1)
        self.assertGreater(error.exception.diagnostics['row_max_relative_error'],1e-6)

    def test_old_numpy_alias_path_can_modify_query_without_proving_real_failure(self):
        raw=np.array([[3.,4.],[6.,8.]],np.float32);before=raw.copy()
        query=np.ascontiguousarray(np.asarray(raw,dtype=np.float32))
        self.assertTrue(np.shares_memory(query,raw))
        query/=np.linalg.norm(query,axis=1,keepdims=True)
        self.assertFalse(np.array_equal(raw,before))

    def test_prototype_identity_and_physical_index_mapping(self):
        supports=[dict(support_index=i+1,relative_path=f's{i}',grid=[2,2],feature_sha256=str(i)) for i in range(8)]
        ids=np.array([0,7,12,31],np.int64);rows=runtime.provenance_rows(7,ids,supports)
        self.assertEqual([r['support_index'] for r in rows],[1,2,4,8])
        self.assertEqual([r['physical_patch_index'] for r in rows],[0,3,0,3])
        bank=np.arange(32*5,dtype=np.float32).reshape(32,5);p=bank[ids].copy()
        for row in rows:np.testing.assert_array_equal(p[row['prototype_index']],bank[row['selected_target_flat_index']])


class StageTests(unittest.TestCase):
    def test_serialized_support_paths_use_actual_cached_reader_without_torch(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);supports=[];features=np.arange(4*1280,dtype=np.float32).reshape(4,1280)
            for index in range(1,9):
                relative=f'test/good/background/s{index}.png';path=root/f'{index:02d}.npz'
                np.savez_compressed(path,protocol_id=np.asarray(inputs.replay.PROTOCOL_ID),relative_path=np.asarray(relative),grid=np.array([2,2],np.int32),
                    **{f'layer_{l}':features+index for l in inputs.LAYERS})
                supports.append(dict(support_index=index,relative_path=relative,grid=[2,2],feature_sha256=inputs.replay.sha256(path)))
            data=dict(paths={'support':str(root)},capture=dict(supports=supports))
            banks=runtime.load_target_layer(data,7)
            self.assertEqual(len(banks),8);self.assertNotIn('torch',sys.modules)
            for index,bank in enumerate(banks,1):np.testing.assert_array_equal(bank,features+index)

    def test_default_preflight_and_help_no_torch_no_output(self):
        self.assertNotIn('torch',sys.modules)
        with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(SystemExit) as e:runner.main(['--help'])
        self.assertEqual(e.exception.code,0)
        with tempfile.TemporaryDirectory() as td:
            dest=Path(td)/'new'
            output=io.StringIO()
            with contextlib.redirect_stdout(output):code=runner.main(['--project-root',td,'--output-dir',str(dest)])
            self.assertEqual(code,2);self.assertFalse(dest.exists());self.assertNotIn('torch',sys.modules)
            check=json.loads(output.getvalue());self.assertEqual(check['sampling'],inputs.SAMPLING)
            self.assertIn('prototype_selector_upstream',' '.join(check['missing_assets']))
            self.assertNotIn('patchcore_official',json.dumps(check['paths']))

    def test_original_scope_support_exclusion_and_order_checker_is_reused(self):
        supports=[dict(support_index=i+1,relative_path=f'test/good/background/s{i}.png') for i in range(8)]
        rows=[dict(branch='target_matched',domain='background',support_index=i+1,relative_path=s['relative_path'],label=0) for i,s in enumerate(supports)]
        with patch.object(inputs.replay,'read_csv',return_value=rows):
            inputs.shared.support_manifest_matches(supports,Path('fixture'))
            changed=list(supports);changed[0]=dict(changed[0],relative_path='query.png')
            with self.assertRaises(ValueError):inputs.shared.support_manifest_matches(changed,Path('fixture'))
            with self.assertRaises(ValueError):inputs.shared.support_manifest_matches(supports[::-1],Path('fixture'))

    def test_partial_stage_and_wrong_case_scope_cannot_freeze_or_evaluate(self):
        with tempfile.TemporaryDirectory() as td:
            data=dict(tool=inputs.TOOL,frozen=dict(sampling=dict(inputs.SAMPLING)),paths={'output':td},capture=dict(samples=[],selected_cases=[]))
            with self.assertRaises(FileNotFoundError):runtime.verify_stage(data,dict(artifacts=[],sampling=dict(inputs.SAMPLING)))
            self.assertFalse((Path(td)/'evaluation').exists())

    def test_signed_distance_audit_can_be_negative_and_preserves_reference(self):
        old=np.ones((4,4),np.float32);raws=dict(UNION_K8=old,UNION_TARGET512=old*2,FASTREF_TARGET512_SOURCE=old*3)
        rows=analysis.distance_rows(dict(sample_id='x',label=0),raws,np.arange(4),np.zeros((2,2),bool),np.ones(4))
        self.assertEqual(len(rows),12);self.assertTrue(all(r['R_value']<0 for r in rows))
        self.assertEqual({r['reference'] for r in rows},{'UNION_K8','UNION_TARGET512'})

    def test_full_flow_evaluates_only_two_new_branches_and_reuses_history(self):
        with tempfile.TemporaryDirectory() as td:
            data,prepared,backend,metrics=flow_fixture(Path(td));evaluator=full_protocol_fixture()
            with patch('DINOv3.MADEqual.aebad_s_sparc_union_p0.runtime.evaluation_device'),\
                 patch.object(analysis,'runtime_evaluator',return_value=evaluator),patch.object(analysis,'MapBackend',return_value=backend),\
                 patch.object(inputs,'original_metrics',return_value=metrics),patch.object(evaluator,'_evaluate',wraps=evaluator._evaluate) as evaluate,\
                 patch('DINOv3.MADEqual.aebad_s_fastref_k8_p0.visualization.plot_case') as figure:
                state=analysis.evaluate(data,prepared,'fixture')
            self.assertEqual(state['status'],'analysis_complete',state);self.assertEqual(evaluate.call_count,2);self.assertEqual(figure.call_count,12)
            rows=inputs.replay.read_csv(Path(data['paths']['output'])/'evaluation/metrics.csv')
            self.assertEqual([r['branch'] for r in rows],list(inputs.BRANCHES))
            self.assertEqual([r['result_origin'] for r in rows],['reused_historical_result']*2+['new_calculation']*2)
            self.assertFalse(state['old_branches_pixel_evaluated']);self.assertTrue(state['predictions_frozen_before_GT'])
            comparisons=inputs.replay.read_csv(Path(data['paths']['output'])/'evaluation/metric_comparisons.csv')
            self.assertEqual(len(comparisons),3)
            by_branch={r['branch']:r for r in rows}
            for r in comparisons:
                for key in inputs.METRIC_KEYS:
                    self.assertAlmostEqual(float(r['delta_'+key]),float(by_branch[r['branch']][key])-float(by_branch[r['reference']][key]))

    def test_changed_cache_blocks_before_GT_or_metric_backend(self):
        with tempfile.TemporaryDirectory() as td:
            data,prepared,_,_=flow_fixture(Path(td));root=Path(data['paths']['output']);next((root/'raw').glob('*.npz')).write_bytes(b'changed')
            with patch.object(analysis,'runtime_evaluator') as evaluator,self.assertRaises(ValueError):analysis.evaluate(data,prepared,'fixture')
            evaluator.assert_not_called();self.assertFalse((root/'evaluation').exists())

    def test_missing_required_GT_stays_partial_without_pixel_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            data,prepared,backend,metrics=flow_fixture(Path(td));evaluator=full_protocol_fixture()
            sample=next(s for s in data['capture']['samples'] if s['label'])
            (Path(data['paths']['data'])/inputs.shared.gt_relative(sample['relative_path'])).unlink()
            with patch('DINOv3.MADEqual.aebad_s_sparc_union_p0.runtime.evaluation_device'),\
                 patch.object(analysis,'runtime_evaluator',return_value=evaluator),patch.object(analysis,'MapBackend',return_value=backend),\
                 patch.object(inputs,'original_metrics',return_value=metrics),patch.object(evaluator,'_evaluate') as pixel:
                state=analysis.evaluate(data,prepared,'fixture')
            self.assertEqual(state['status'],'partial_evidence');self.assertIn(sample['sample_id'],state['missing_gt_ids'])
            self.assertFalse(state['pixel_metrics_complete']);pixel.assert_not_called()

    def test_even_rehashed_failed_solver_diagnostics_cannot_complete(self):
        with tempfile.TemporaryDirectory() as td:
            data,prepared,_,_=flow_fixture(Path(td));root=Path(data['paths']['output'])
            file=root/'infer_solver_diagnostics.json';diag=json.loads(file.read_text());diag[0]['outer'][0]['column_max_relative_error']=.01
            inputs.replay.atomic_json(file,diag)
            state=json.loads((root/'infer_status.json').read_text());state['solver_diagnostics_sha256']=inputs.replay.sha256(file)
            inputs.replay.atomic_json(root/'infer_status.json',state)
            with self.assertRaisesRegex(ValueError,'OT/update'):runtime.verify_stage(data,prepared)

    def test_consistent_map_backend_shift_cannot_reapprove_original_scores(self):
        with tempfile.TemporaryDirectory() as td:
            data,prepared,backend,metrics=flow_fixture(Path(td));method=backend.runner._method_map
            backend.runner._method_map=lambda *a,**kw:method(*a,**kw)+1.
            with patch('DINOv3.MADEqual.aebad_s_sparc_union_p0.runtime.evaluation_device'),\
                 patch.object(analysis,'runtime_evaluator',return_value=full_protocol_fixture()),patch.object(analysis,'MapBackend',return_value=backend),\
                 patch.object(inputs,'original_metrics',return_value=metrics),patch('DINOv3.MADEqual.aebad_s_fastref_k8_p0.visualization.plot_case'):
                state=analysis.evaluate(data,prepared,'fixture')
            self.assertEqual(state['status'],'partial_evidence');self.assertFalse(state['pixel_metrics_complete']);self.assertGreater(state['numerical_failed_checks'],0)


if __name__=='__main__':unittest.main()
