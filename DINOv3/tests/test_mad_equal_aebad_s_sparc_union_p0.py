"""NumPy fixtures/actual host kernel AST; no real Torch, model or CUDA."""
from __future__ import annotations
import ast
import contextlib
import io
import json
import sys
import tarfile
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from DINOv3.MADEqual.aebad_s_sparc_union_p0 import inputs,runtime,analysis,runner
from DINOv3.MADEqual.aebad_s_sparc_p0 import projection as geom,analysis as old_analysis,runtime as old_runtime
from DINOv3.tests.test_mad_equal_aebad_s_sparc_p0 import evaluation_fixture,actual_functions,FakeTensor
from DINOv3.tests.test_mad_equal_aebad_s_target_local_segment import full_protocol_fixture

ROOT=Path(__file__).resolve().parents[2]


class Tensor(FakeTensor):
    def detach(self):return self
    def float(self):return Tensor(self.value.astype(np.float32))
    @property
    def ndim(self):return self.value.ndim
    def element_size(self):return self.value.dtype.itemsize
    def __getitem__(self,key):return Tensor(self.value[key])
    def __setitem__(self,key,value):self.value[key]=value.value
    def min(self,dim):
        idx=np.argmin(self.value,axis=dim)
        return Tensor(np.take_along_axis(self.value,idx[:,None],axis=dim).reshape(-1)),Tensor(idx.astype(np.int64))


def torch_fixture():
    fake=SimpleNamespace(from_numpy=Tensor,float32=np.float32,int64=np.int64,
        empty=lambda shape,device,dtype:Tensor(np.empty(shape,dtype=dtype)),
        cdist=lambda a,b,compute_mode:Tensor(np.linalg.norm(a.value[:,None]-b.value[None],axis=2).astype(np.float32)),
        inference_mode=contextlib.nullcontext,cuda=SimpleNamespace(synchronize=lambda device:None))
    env={'torch':fake}
    actual_functions(ROOT/'DINOv3/relation_reliability/superadd_host.py',{'_exact_top1_chunked'},env)
    host=SimpleNamespace(_exact_top1_chunked=lambda q,k:env['_exact_top1_chunked'](q,k,max_matrix_bytes=32))
    return fake,host


def target_fixture(root):
    base=root/'parent';base.mkdir();out=root/'union';out.mkdir();grid=(2,2);n=4
    supports=[]
    for layer in geom.LAYERS:
        d=base/'projection'/f'L{layer}';d.mkdir(parents=True)
        mu=np.zeros((n,1280),np.float64);mu[:,0]=np.arange(n)*1280;mu[:,1]=np.arange(n)*12800
        basis=np.zeros((n,1280,7),np.float64);basis[:,0,0]=1
        np.save(d/'mu.npy',mu);np.save(d/'basis.npy',basis);np.save(d/'rank.npy',np.ones(n,np.int8))
    for j in range(8):
        value=mu.astype(np.float32).copy();value[:,0]+=(j-3.5)*128
        path=root/f'support_{j+1}.npz'
        np.savez(path,grid=np.array(grid,np.int32),**{f'layer_{l}':value for l in geom.LAYERS})
        supports.append(dict(support_index=j+1,path=str(path),relative_path=f'test/good/background/s{j+1}.png',feature_sha256=inputs.replay.sha256(path)))
    frozen=dict(parent_identity='fixture-parent',grid=list(grid),supports=supports,cpu_workers=4)
    data=dict(tool=inputs.TOOL,identity=inputs.digest(frozen),frozen=frozen,paths={'parent':str(base),'output':str(out)},
        required_files=[dict(kind='support',path=s['path'],sha256=s['feature_sha256']) for s in supports],resources=inputs.resources(grid,4))
    inputs.replay.atomic_json(out/'inputs.json',data)
    return data,mu


def analysis_fixture(root):
    parent,prepared,backends=evaluation_fixture(root)
    base=Path(parent['paths']['output']);directory=base/'evaluation';directory.mkdir()
    cap=json.loads((Path(parent['paths']['capture'])/'inputs.json').read_text())
    old_state=json.loads((Path(parent['paths']['capture'])/'status.json').read_text())
    infer=json.loads((base/'infer_status.json').read_text());scores=[];maps_by_branch={b:[] for b in geom.BRANCHES}
    # Tiny synthetic parent geometry is made internally complete for load_new.
    for s,old,new in zip(parent['samples'],old_state['records'],infer['records'],strict=True):
        file=Path(parent['paths']['capture'])/old['file']
        with np.load(file) as f:arrays={k:f[k].copy() for k in f.files if k!='metadata'};meta=json.loads(str(f['metadata'].item()))
        meta['geometry'].update(processed_height=32,processed_width=32)
        with file.open('wb') as f:np.savez_compressed(f,metadata=np.asarray(json.dumps(meta)),**arrays)
        old['sha256']=inputs.replay.sha256(file)
        file=base/new['file']
        with np.load(file) as f:values={k:f[k].copy() for k in f.files if k!='metadata'};m=json.loads(str(f['metadata'].item()))
        m.update(geometry=meta['geometry'],baseline_raw_sha256=old['sha256'])
        with file.open('wb') as f:np.savez_compressed(f,metadata=np.asarray(json.dumps(m)),**values)
        new['sha256']=inputs.replay.sha256(file)
        raws=dict(SOURCE=arrays['source_distance'],UNION_K8=np.minimum(arrays['source_distance'],arrays['target_distance'].min(axis=0)),SPARC_SOURCE_K8=values['distance'])
        for b,value in analysis.maps_for(raws,arrays['positions'],meta,backends).items():
            scores.append(dict(branch=b,sample_id=s['sample_id'],relative_path=s['relative_path'],label=s['label'],image_score=float(value.max())))
            maps_by_branch[b].append(value)
    inputs.replay.atomic_json(Path(parent['paths']['capture'])/'status.json',old_state)
    inputs.replay.atomic_json(base/'infer_status.json',infer)
    inputs.replay.write_csv(directory/'per_image_scores.csv',scores)
    labels=[s['label'] for s in parent['samples']];from sklearn.metrics import roc_auc_score,average_precision_score
    metrics=[]
    for b in geom.BRANCHES:
        values=[r['image_score'] for r in scores if r['branch']==b]
        metrics.append(dict(branch=b,image_count=297,image_AUROC=roc_auc_score(labels,values),image_AP=average_precision_score(labels,values),pixel_AUROC=.5,pixel_AP=.5,AUPRO=.5))
    inputs.replay.write_csv(directory/'metrics.csv',metrics)
    out=root/'union';out.mkdir();frozen=dict(parent_identity=parent['identity'],grid=[2,2],cpu_workers=1,supports=[])
    data=dict(tool=inputs.TOOL,identity=inputs.digest(frozen),frozen=frozen,parent=parent,
        paths=dict(parent['paths'],parent=str(base),output=str(out)),required_files=[],resources=inputs.resources((2,2),1))
    inputs.replay.atomic_json(out/'inputs.json',data)
    ready=dict(tool=inputs.TOOL,status='target_prepared',identity=data['identity'],inputs_sha256=inputs.replay.sha256(out/'inputs.json'),artifacts=[])
    inputs.replay.atomic_json(out/'prepare_target_status.json',ready)
    records=[];checks=[]
    for s,old,new in zip(parent['samples'],old_state['records'],infer['records'],strict=True):
        with np.load(base/new['file']) as f:d=f['distance'].copy();p=f['positions'].copy();m=json.loads(str(f['metadata'].item()))
        arrays=dict(source_distance=d,target_distance=d*.8,union_distance=d*.8,source_index=np.zeros_like(d,np.int64),target_index=np.zeros_like(d,np.int64),
                    positions=p,grid=np.array([2,2],np.int32),processed_shape=np.array([32,32],np.int32))
        meta=dict(tool=inputs.TOOL,identity=data['identity'],parent_identity=parent['identity'],sample_id=s['sample_id'],relative_path=s['relative_path'],
            image_sha256=m['image_sha256'],grid=m['grid'],geometry=m['geometry'],output_shape=m['output_shape'],parent_raw_sha256=new['sha256'],branch=inputs.NEW,distance_divisor=1280,
            target_prepare_sha256=inputs.replay.sha256(out/'prepare_target_status.json'),historical_source_reproduced=True,query_forward_calls=1,query_corrections_per_layer=1)
        file=out/'raw'/f"{s['sample_id']}.npz";old_runtime.save_npz(file,metadata=np.asarray(json.dumps(meta)),**arrays)
        records.append(dict(sample_id=s['sample_id'],**inputs.parent_inputs.file_records(out,[file])[0]))
        for kind in ('historical_SPARC_Source_full_grid','historical_SPARC_Source_image_score','same_query_same_MAD_map_monotonicity'):
            checks.append(dict(sample_id=s['sample_id'],check_kind=kind,verified=True))
    inputs.replay.write_csv(out/'infer_verification.csv',checks)
    inputs.replay.atomic_json(out/'infer_status.json',dict(tool=inputs.TOOL,identity=data['identity'],parent_identity=parent['identity'],status='predictions_frozen',predicted_images=297,
        inputs_sha256=inputs.replay.sha256(out/'inputs.json'),prepare_target_sha256=inputs.replay.sha256(out/'prepare_target_status.json'),records=records,historical_source_reproduced=True,
        verification_sha256=inputs.replay.sha256(out/'infer_verification.csv')))
    return data,ready,dict(backends,SPARC_UNION_K8=backends['SPARC_SOURCE_K8'])


class ProjectionAndExactTests(unittest.TestCase):
    def test_prepare_target_reuses_states_without_torch_fitting_or_source_builder(self):
        with tempfile.TemporaryDirectory() as td:
            data,mu=target_fixture(Path(td))
            with patch.object(geom,'fit_layer',side_effect=AssertionError('no fitting')),patch.object(old_runtime,'load_host',side_effect=AssertionError('no host')),patch.object(old_runtime,'prepare',side_effect=AssertionError('no Source prepare')),patch.object(inputs.parent_inputs,'available_ram',return_value=1<<40):
                state=runtime.prepare_target(data)
            self.assertEqual(state['status'],'target_prepared');self.assertNotIn('torch',sys.modules)
            for l in geom.LAYERS:
                values=np.load(Path(data['paths']['output'])/'target_memory'/f'L{l}.npy')
                self.assertEqual(values.shape,(32,1280))
                for j in range(8):np.testing.assert_array_equal(values[j*4:(j+1)*4],mu.astype(np.float32))
            rows=inputs.replay.read_csv(Path(data['paths']['output'])/'support_collapse_verification.csv')
            self.assertEqual(len(rows),32);self.assertTrue(all(r['verified']=='True' for r in rows))
            self.assertEqual(state['cpu_workers'],4)

    def test_parallel_and_serial_original_projection_are_identical(self):
        with tempfile.TemporaryDirectory() as td:
            data,mu=target_fixture(Path(td));projections=runtime.projections_for(data)
            try:
                serial=runtime.ordered_parallel(lambda s:runtime.corrected_support(Path(s['path']),7,projections[7],(2,2)),data['frozen']['supports'],1)
                parallel=runtime.ordered_parallel(lambda s:runtime.corrected_support(Path(s['path']),7,projections[7],(2,2)),data['frozen']['supports'],8)
                for a,b in zip(serial,parallel,strict=True):np.testing.assert_array_equal(a[0],b[0]);self.assertEqual(a[1],b[1])
            finally:
                for p in projections.values():p.close()

    def test_query_reorder_and_target_own_positions_with_global_cross_position_search(self):
        fake,host=torch_fixture()
        with tempfile.TemporaryDirectory() as td:
            data,mu=target_fixture(Path(td));p=runtime.projections_for(data)
            try:
                target,_=runtime.corrected_support(Path(data['frozen']['supports'][0]['path']),7,p[7],(2,2))
                features=mu.astype(np.float32).copy();features[0,1]=12800;positions=np.array([2,0,3,1],np.int64)
                corrected=p[7].correct(features[positions],positions,(2,2))
                np.testing.assert_array_equal(corrected,p[7].correct(features,np.arange(4,dtype=np.int64),(2,2))[positions])
                with patch.dict(sys.modules,{'torch':fake,'DINOv3.relation_reliability.superadd_host':host}):
                    ds,dt,du,si,ti=runtime.exact_pair(corrected,Tensor(np.zeros((1,1280),np.float32)),Tensor(np.tile(target,(8,1))),'fixture')
                self.assertEqual(int(ti[1])%4,1) # query physical position 0 matches Target position 1.
                brute=np.linalg.norm(corrected[:,None]-np.tile(target,(8,1))[None],axis=2).min(axis=1)/1280
                np.testing.assert_array_equal(dt,brute);np.testing.assert_array_equal(du,np.minimum(ds,dt));self.assertEqual(dt[1],1.)
            finally:
                for value in p.values():value.close()

    def test_one_query_forward_one_correction_per_layer_two_exact_searches(self):
        fake,host=torch_fixture();positions=np.array([2,0,3,1],np.int64);counts=defaultdict(int)
        class CountingProjection:
            def correct(self,q,p,grid):counts['correction']+=1;return q.copy()
        query=np.zeros((4,1280),np.float32);query[:,1]=1280
        item=SimpleNamespace(grid=(2,2),positions=positions,queries={l:Tensor(query) for l in geom.LAYERS})
        def extract(*args,**kwargs):counts['forward']+=1;return [item]
        actual_nearest=host._exact_top1_chunked
        def nearest(q,k):counts['nearest']+=1;return actual_nearest(q,k)
        host._exact_top1_chunked=nearest;host._extract_processed_batch=extract;host.image_tensor=lambda image:Tensor(np.zeros((3,32,32),np.float32))
        banks={l:Tensor(np.zeros((32,1280),np.float32)) for l in geom.LAYERS};model=SimpleNamespace(preprocessing=lambda x:x,prototype_embeddings=banks)
        timing=defaultdict(float)
        with patch.dict(sys.modules,{'torch':fake,'DINOv3.relation_reliability.superadd_host':host}):
            result=runtime.observe(model,None,'fixture',{l:CountingProjection() for l in geom.LAYERS},banks,4,timing)
        self.assertEqual(counts,{'correction':4,'forward':1,'nearest':8});np.testing.assert_array_equal(result['union_distance'],np.ones((4,4),np.float32))

    def test_infer_saves_numpy_int32_query_geometry_as_native_json_without_changing_arrays(self):
        from DINOv3.MADEqual.aebad_s_target_support_trace.analysis import image_geometry
        fake,_=torch_fixture();axis_sizes=[]
        def split(size):
            axis_sizes.append(size)
            return [(0,size)],[(0,size//16)],[(0,size//16)]
        model=SimpleNamespace(patch_exec=SimpleNamespace(model_patch_size=16,axis_patch_split=split))
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);base=root/'parent';base.mkdir();out=root/'union';out.mkdir();images=root/'images';images.mkdir()
            image_path=images/'query.png';runtime.Image.new('RGB',(32,32)).save(image_path)
            sample=dict(sample_id='case',relative_path='query.png',label=0)
            data=dict(identity='fixture',device='fixture',frozen=dict(parent_identity='parent',grid=[2,2],cpu_workers=1),
                parent=dict(samples=[sample]),paths=dict(parent=str(base),output=str(out),data=str(images),capture=str(root/'capture')),
                required_files=[])
            inputs.replay.atomic_json(out/'inputs.json',data)
            inputs.replay.atomic_json(out/'prepare_target_status.json',dict(status='target_prepared'))
            inputs.replay.atomic_json(base/'infer_status.json',dict(records=[dict(sample_id='case',file='parent.npz',sha256='parent-sha')]))
            bank_dir=out/'target_memory';bank_dir.mkdir()
            for layer in geom.LAYERS:np.save(bank_dir/f'L{layer}.npy',np.zeros((32,1280),np.float32))
            observed=dict(source_distance=np.ones((4,4),np.float32),target_distance=np.full((4,4),.5,np.float32),
                union_distance=np.full((4,4),.5,np.float32),source_index=np.zeros((4,4),np.int64),target_index=np.zeros((4,4),np.int64),
                positions=np.arange(4,dtype=np.int64),grid=np.array([2,2],np.int32),processed_shape=np.array([32,32],np.int32))
            geometry=image_geometry(model,32,32,(2,2),(32,32));axis_sizes.clear()
            meta=dict(image_sha256=inputs.replay.sha256(image_path),geometry=geometry,grid=[2,2],output_shape=[8,8])
            projections={layer:SimpleNamespace(close=lambda:None) for layer in geom.LAYERS}
            with patch.dict(sys.modules,{'torch':fake}),patch.object(old_runtime,'load_host',return_value=(model,'fixture')),\
                 patch.object(old_runtime,'load_memory'),patch.object(runtime,'projections_for',return_value=projections),\
                 patch.object(inputs.parent_inputs,'read_capture',return_value=({},dict(records=[dict(sample_id='case')]))),\
                 patch.object(inputs,'history_tables',return_value=({'SPARC_SOURCE_K8':{'case':{'image_score':1.}}},[])),\
                 patch.object(analysis,'backends_for',return_value={}),\
                 patch.object(analysis,'load_parent',return_value=({'SPARC_SOURCE_K8':observed['source_distance']},observed['positions'],meta)),\
                 patch.object(analysis,'maps_for',return_value={'SPARC_SOURCE_K8':np.ones((8,8)),inputs.NEW:np.full((8,8),.5)}),\
                 patch.object(runtime,'observe',return_value=observed),contextlib.redirect_stdout(io.StringIO()):
                state=runtime.infer(data,dict(artifacts=[]))
            self.assertEqual(state['status'],'predictions_frozen')
            self.assertTrue(all(type(size) is int for size in axis_sizes))
            with np.load(out/'raw/case.npz',allow_pickle=False) as cache:
                metadata=json.loads(str(cache['metadata'].item()))
                self.assertEqual(metadata['geometry'],geometry)
                self.assertEqual(cache['grid'].dtype,np.dtype(np.int32))
                self.assertEqual(cache['processed_shape'].dtype,np.dtype(np.int32))
                for key,value in observed.items():np.testing.assert_array_equal(cache[key],value)

    def test_min_before_mad_same_scale_map_monotonicity(self):
        from DINOv3.tests.test_mad_equal_aebad_s_union_support_loo import actual_map_backend
        backend,trace=actual_map_backend();backend.mad.medians[:]=1;backend.mad.scales[:]=2
        source=np.full((4,4),6,np.float32);target=np.full_like(source,4);p=np.array([2,0,3,1],np.int64)
        maps=analysis.maps_for({'SPARC_SOURCE_K8':source,inputs.NEW:np.minimum(source,target)},p,{'grid':[2,2],'output_shape':[2,2]},
            {'SPARC_SOURCE_K8':backend,inputs.NEW:backend})
        self.assertTrue(np.all(maps[inputs.NEW]<=maps['SPARC_SOURCE_K8']))
        self.assertEqual(float(maps[inputs.NEW].max()),1.5)

    def test_history_distance_or_score_shift_rejected(self):
        sample={'sample_id':'case'};base=np.ones((4,4),np.float32)
        for distance,score in [(base+.01,1.),(base,1.001)]:
            with self.subTest(score=score),self.assertRaisesRegex(ValueError,'reproduction failed'):
                runtime.source_checks(sample,{'source_distance':distance},base,np.full((2,2),score),1.,[])

    def test_raw_union_cannot_mix_legacy_or_unrelated_distances(self):
        source=np.ones((4,4),np.float32);arrays=dict(source_distance=source,target_distance=source*.5,union_distance=source,
            source_index=np.zeros((4,4),np.int64),target_index=np.zeros((4,4),np.int64),positions=np.arange(4,dtype=np.int64),grid=np.array([2,2],np.int32),processed_shape=np.array([32,32],np.int32))
        with self.assertRaisesRegex(ValueError,'same query'):runtime.validate_arrays(arrays,(2,2))

    def test_duplicate_support_and_different_tied_indices_do_not_change_distance_verification(self):
        fake,host=torch_fixture();q=np.zeros((4,1280),np.float32);q[:,0]=1280
        source=Tensor(np.zeros((2,1280),np.float32));target=Tensor(np.zeros((32,1280),np.float32))
        with patch.dict(sys.modules,{'torch':fake,'DINOv3.relation_reliability.superadd_host':host}):
            ds,dt,du,si,ti=runtime.exact_pair(q,source,target,'fixture')
        np.testing.assert_array_equal(dt,np.ones(4,np.float32));np.testing.assert_array_equal(du,ds)
        rows=[]
        runtime.source_checks({'sample_id':'tie'},{'source_distance':np.tile(ds,(4,1)),
            'source_index':np.ones((4,4),np.int64)},np.tile(ds,(4,1)),np.ones((2,2)),1.,rows)
        self.assertTrue(all(r['verified'] for r in rows))

    def test_wrong_feature_dimension_or_dtype_rejected_before_search(self):
        fake,host=torch_fixture()
        with patch.dict(sys.modules,{'torch':fake,'DINOv3.relation_reliability.superadd_host':host}):
            for value in (np.zeros((4,1280),np.float64),np.zeros((4,12),np.float32)):
                with self.assertRaisesRegex(ValueError,'already-corrected float32'):runtime.exact_pair(value,None,None,'fixture')

    def test_evaluation_selects_nondefault_cuda_and_frozen_precision_without_host(self):
        from DINOv3.tests.test_mad_equal_aebad_s_sparc_p0 import HostInitializationTests
        helper=HostInitializationTests();parent,modules,model,cuda,events=helper.fixture(current=0)
        with patch.dict(sys.modules,modules),patch.object(old_runtime,'load_host',side_effect=AssertionError('no host')):
            device=runtime.evaluation_device({'parent':parent},'cuda:1')
        self.assertEqual(device.index,1);self.assertIn(('select',1),events);self.assertIn(('reset',1,1),events)
        self.assertFalse(any(e[0]=='construct' for e in events))


class IntegrityAndEvaluationTests(unittest.TestCase):
    def test_parent_approved_records_match_actual_download_without_extraction(self):
        with tarfile.open(ROOT/'downloads/aebad_s_sparc_background_mixed_20261009_095419_review.tar.gz') as archive:
            import hashlib
            for rel,sha in inputs.PARENT_RECORDS.items():
                self.assertEqual(hashlib.sha256(archive.extractfile(inputs.BASE_NAME+'/'+rel).read()).hexdigest(),sha)
            parent=json.load(archive.extractfile(inputs.BASE_NAME+'/inputs.json'));inputs.validate_scope(parent)
            self.assertEqual(parent['identity'],inputs.PARENT_IDENTITY)
            parent['samples'][0]['relative_path']=parent['frozen']['supports'][0]['relative_path']
            with self.assertRaises(ValueError):inputs.validate_scope(parent)

    def test_help_default_preflight_no_torch_no_output_and_missing_asset_list(self):
        with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(SystemExit) as exc:runner.main(['--help'])
        self.assertEqual(exc.exception.code,0)
        with tempfile.TemporaryDirectory() as td:
            dest=Path(td)/'not-created'
            args=runner.parser().parse_args(['--sparc-base-dir',str(Path(td)/'missing'),'--output-dir',str(dest)])
            check=inputs.preflight(args)
            self.assertEqual(check['status'],'blocked');self.assertGreater(len(check['missing_assets']),20)
            self.assertFalse(dest.exists());self.assertNotIn('torch',sys.modules)
            with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(runner.main(['--sparc-base-dir',str(Path(td)/'missing'),'--output-dir',str(dest)]),2)
            self.assertFalse(dest.exists())

    def test_parent_output_overlap_and_half_written_target_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for out in (root,root/'child'):
                with self.assertRaises(ValueError):inputs.guard_output({'parent':root,'output':out})
            dest=root/'separate';dest.mkdir();(dest/'inputs.json').write_text('{}')
            with self.assertRaises(ValueError):inputs.guard_output({'parent':root/'parent','output':dest})

    def test_modified_target_memory_refused_without_reapproval(self):
        with tempfile.TemporaryDirectory() as td:
            data,mu=target_fixture(Path(td))
            with patch.object(inputs.parent_inputs,'available_ram',return_value=1<<40):runtime.prepare_target(data)
            out=Path(data['paths']['output']);inputs.read_prepared(out)
            with (out/'target_memory/L7.npy').open('ab') as f:f.write(b'modified')
            with self.assertRaises(ValueError):inputs.read_prepared(out)

    def test_distance_audits_use_two_fixed_scales_and_correct_regions(self):
        p=np.array([2,0,3,1],np.int64);raws={'UNION_K8':np.full((4,4),5,np.float32),'SPARC_SOURCE_K8':np.full((4,4),7,np.float32),inputs.NEW:np.full((4,4),3,np.float32)}
        sample=dict(sample_id='case',label=1);mask=np.array([[1,0],[0,0]],bool)
        rows=analysis.distance_audit(sample,raws,p,mask,np.full(4,2),np.full(4,4))
        self.assertEqual(len(rows),8);self.assertTrue(all(r['R_value']==1 and r['selected_patch_count']==1 for r in rows))
        self.assertEqual({r['scale'] for r in rows},{'old_Source_MAD_scale','SPARC_Source_MAD_scale'})

    def test_full_four_branch_flow_old_metrics_reused_and_only_new_pixel_evaluated(self):
        with tempfile.TemporaryDirectory() as td:
            data,prepared,backends=analysis_fixture(Path(td));evaluator=full_protocol_fixture()
            with patch.object(analysis,'backends_for',return_value=backends),patch.object(old_analysis,'runtime_evaluator',return_value=evaluator),\
                 patch.object(evaluator,'_evaluate',wraps=evaluator._evaluate) as evaluate,patch('DINOv3.MADEqual.aebad_s_sparc_union_p0.visualization.plot_case'):
                with contextlib.redirect_stdout(io.StringIO()):state=analysis.evaluate(data,prepared)
            self.assertEqual(state['status'],'analysis_complete');evaluate.assert_called_once()
            root=Path(data['paths']['output'])/'evaluation'
            rows=inputs.replay.read_csv(root/'metrics.csv');self.assertEqual([r['branch'] for r in rows],list(inputs.BRANCHES))
            self.assertEqual([r['result_origin'] for r in rows],['reused_historical_result']*3+['new_calculation'])
            self.assertEqual(len(inputs.replay.read_csv(root/'per_image_scores.csv')),1188)
            self.assertEqual(state['selected_figures'],12);self.assertFalse(state['old_branches_pixel_evaluated'])

    def test_modified_new_raw_rejected_before_gt_or_evaluation(self):
        with tempfile.TemporaryDirectory() as td:
            data,prepared,backends=analysis_fixture(Path(td));root=Path(data['paths']['output']);next((root/'raw').glob('*.npz')).write_bytes(b'changed')
            with patch.object(old_analysis,'runtime_evaluator') as evaluator:
                with self.assertRaises(ValueError):analysis.evaluate(data,prepared)
                evaluator.assert_not_called()
            self.assertFalse((root/'evaluation').exists())

    def test_consistent_scoring_backend_shift_blocks_analysis_complete(self):
        with tempfile.TemporaryDirectory() as td:
            data,prepared,backends=analysis_fixture(Path(td));backend=backends['SOURCE'];old=backend.runner._method_map
            backend.runner._method_map=lambda *a,**kw:old(*a,**kw)+.01
            with patch.object(analysis,'backends_for',return_value=backends),patch.object(old_analysis,'runtime_evaluator',return_value=full_protocol_fixture()),\
                 patch('DINOv3.MADEqual.aebad_s_sparc_union_p0.visualization.plot_case'):
                with contextlib.redirect_stdout(io.StringIO()):state=analysis.evaluate(data,prepared)
            self.assertEqual(state['status'],'partial_evidence');self.assertGreater(state['numerical_failed_checks'],0)

    def test_legacy_branches_and_default_builder_unchanged(self):
        self.assertEqual(geom.BRANCHES,('SOURCE','UNION_K8','SPARC_SOURCE_K8'))
        module=ast.parse((ROOT/'external_baselines/superadd_external/memory_variants.py').read_text(encoding='utf-8'))
        f=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='build_official_memory_spooled')
        defaults=dict(zip([a.arg for a in f.args.kwonlyargs],f.args.kw_defaults))
        self.assertEqual(ast.literal_eval(defaults['storage_backend']),'disk')
        self.assertIsNone(ast.literal_eval(defaults['feature_transform']))


if __name__=='__main__':unittest.main()
