"""CPU production-path tests. Synthetic integration is not DINO/CUDA validation."""
from __future__ import annotations
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from contextlib import nullcontext
import unittest
from unittest import mock
import numpy as np
from PIL import Image
from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0 import method as m,inputs as ip,capture,prediction as pr,analysis as an,runner
from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import statistics,ch1_score

ROOT=Path(__file__).resolve().parents[2]


def raw_fixture(grid=(3,3),seed=42):
    rng=np.random.default_rng(seed);n=int(np.prod(grid));q=rng.normal(size=(n,1280)).astype(np.float32)
    banks=[rng.normal(size=(j+2,1280)).astype(np.float32) for j in range(8)]
    target=[];indices=[]
    for bank in banks:
        # Only fixture construction uses independent NumPy distances. Production
        # SEGMENT below is the actual original geometry_layer, not a substitute.
        d=np.linalg.norm(q[:,None].astype(float)-bank[None].astype(float),axis=2)/1280
        idx=d.argmin(1);target.append(d[np.arange(n),idx].astype(np.float32));indices.append(idx.astype(np.int64))
    raw=dict(positions=np.arange(n,dtype=np.int64),source_distance=np.full((4,n),.1,np.float32),
        source_index=np.zeros((4,n),np.int64),target_distance=np.repeat(np.stack(target)[:,None],4,axis=1),
        target_index=np.repeat(np.stack(indices)[:,None],4,axis=1))
    return m.add_geometry(raw,{l:banks for l in m.LAYERS})[0],banks


class TemporaryCase(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='crossdomain_test_',dir=ROOT/'downloads')
        self.root=Path(self.temp.name).resolve()
        self.assertTrue(self.root.is_relative_to((ROOT/'downloads').resolve()))
    def tearDown(self):self.temp.cleanup()

    def fixture(self):
        raw,banks=raw_fixture();domain_rows={};data=self.root/'data';(data/'test').mkdir(parents=True);(data/'ground_truth').mkdir()
        ctx=dict(paths=dict(project=self.root,data=data,cache_root=self.root/'cache',memory=self.root/'missing.pt'),
            identity='synthetic_joint_identity',frozen={'both_domains':'one_protocol'},extraction_identity='synthetic_extraction',
            medians=np.zeros(4,np.float32),scales=np.ones(4,np.float32),state=dict(peak=dict(medians=[0]*4,scales=[1]*4),local=dict(medians=[0]*4,scales=[1]*4),state_sha256='fixture'))
        for domain in ip.DOMAINS:
            samples=[];supports=[]
            for i in range(8):supports.append(dict(support_index=i+1,relative_path=f'test/good/{domain}/support{i}.png',image_sha256='fixture',image_height=48,image_width=48,grid=[1,len(banks[i])]))
            for label in (0,1):
                sid=f'test_{domain}_{label}';rel=f'test/{"bad" if label else "good"}/{domain}/{sid}.png'
                p=data/rel;p.parent.mkdir(parents=True,exist_ok=True);Image.new('RGB',(48,48),(10,20,30)).save(p)
                s=dict(domain=domain,sample_id=sid,relative_path=rel,label=label,image_height=48,image_width=48,image_sha256=ip.sha(p))
                if label:
                    mask=np.zeros((48,48),np.uint8);mask[17:27,18:28]=255;p=data/rel.replace('test/','ground_truth/',1);p.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(mask).save(p);s['gt_sha256']=ip.sha(p)
                union=np.minimum(raw['source_distance'],raw['target_distance'].min(0))
                s.update(source_score=statistics(raw['source_distance'],raw['positions'],(3,3),(12,12),ctx['medians'],ctx['scales'])[0],
                    union_score=statistics(union,raw['positions'],(3,3),(12,12),ctx['medians'],ctx['scales'])[0]);samples.append(s)
            domain_rows[domain]=dict(samples=samples,supports=supports,normal_count=1,anomaly_count=1)
        ctx['catalog']=dict(domains=domain_rows,memory_sha256='missing',checkpoint_sha256='missing',background_reference={'rows':[]})
        cache=ctx['paths']['cache_root'];cache.mkdir();ip.write_json(cache/'input_manifest.json',dict(identity=ctx['identity'],frozen=ctx['frozen']))
        records=[]
        for domain,d in domain_rows.items():
            for s in d['samples']:
                meta=dict(format=ip.FORMAT,tool=ip.TOOL,identity=ctx['identity'],domain=domain,sample_id=s['sample_id'],relative_path=s['relative_path'],
                    image_sha256=s['image_sha256'],grid=[3,3],output_shape=[12,12],source_rows=[100000]*4,supports=d['supports'],distance_unit=ip.PROTOCOL['distance_unit'])
                file=f'raw_cache/{domain}/{s["sample_id"]}.npz';ip.save_npz(cache/file,raw,meta)
                records.append(dict(domain=domain,sample_id=s['sample_id'],file=file,sha256=ip.sha(cache/file)))
        ip.write_json(cache/'cache_status.json',dict(tool=ip.TOOL,identity=ctx['identity'],records=records,supports=[],sources={}))
        return ctx,raw


class PairTests(unittest.TestCase):
    def test_physical_permutation_and_unordered_pairs(self):
        rng=np.random.default_rng(7);pairs=np.stack([rng.integers(1,4,25),rng.integers(4,9,25)]).astype(np.int8);positions=np.arange(25,dtype=np.int64)
        expected=m.pair_gate(pairs,positions,(5,5));perm=rng.permutation(25)
        swapped=pairs.copy();swapped[:,::2]=swapped[::-1,::2]
        for a,b in zip(expected,m.pair_gate(swapped[:,perm],positions[perm],(5,5))):np.testing.assert_array_equal(a,b)

    def test_center_exclusion_boundaries_ceil(self):
        pairs=np.tile(np.array([[3],[4]],np.int8),(1,9));pairs[:,[0,1,3]]=np.array([[1],[2]])
        gate,votes,n=m.pair_gate(pairs,np.arange(9,dtype=np.int64),(3,3))
        self.assertEqual(n[0,0],3);self.assertEqual(n[0,1],5);self.assertEqual(n[1,1],8)
        self.assertEqual(votes[0,0],2);self.assertTrue(gate[0,0])
        pairs[:,3]=[3,4];self.assertFalse(m.pair_gate(pairs,np.arange(9,dtype=np.int64),(3,3))[0][0,0])
        pairs[:]=np.array([[3],[4]]);pairs[:,4]=[1,2]
        gate,votes,_=m.pair_gate(pairs,np.arange(9,dtype=np.int64),(3,3));self.assertEqual(votes[1,1],0);self.assertFalse(gate[1,1])

    def test_no_neighbors_reject(self):
        gate,votes,n=m.pair_gate(np.array([[1],[2]],np.int8),np.array([0],np.int64),(1,1))
        self.assertFalse(gate.item());self.assertEqual(votes.item(),0);self.assertEqual(n.item(),0)

    def test_all_accept_and_all_reject_fallback(self):
        raw,_=raw_fixture();raw['target_distance'][:]=np.arange(1,9,dtype=np.float32)[:,None,None]
        raw['source_distance'][:]=3;raw['segment_distance'][:]=.5
        raw['endpoint_support'][:]=np.array([1,2],np.int8)[None,:,None]
        for off in range(4):raw['endpoint_index'][off]=raw['target_index'][[0,1],off]
        out,_=m.predict(raw,(3,3));self.assertTrue(out['pair_gate'].all());np.testing.assert_array_equal(out['pair_distance'],raw['segment_distance'])
        # Nine different unordered support pairs: every real neighbor disagrees.
        pairs=np.array([(a,b) for a in range(1,9) for b in range(a+1,9)][:9],np.int8).T
        for i,(a,b) in enumerate(pairs.T):
            raw['target_distance'][:,:,i]=10;raw['target_distance'][a-1,:,i]=1;raw['target_distance'][b-1,:,i]=1.5
            raw['endpoint_support'][:,:,i]=pairs[:,i]
            for off in range(4):raw['endpoint_index'][off,:,i]=raw['target_index'][pairs[:,i]-1,off,i]
        out,_=m.predict(raw,(3,3));self.assertFalse(out['pair_gate'].any());np.testing.assert_array_equal(out['pair_distance'],out['union_distance'])

    def test_actual_geometry_real_endpoints_and_ties(self):
        raw,banks=raw_fixture();m.validate_geometry(raw)
        for off in range(4):
            ids=raw['endpoint_support'][off]-1;idx=raw['endpoint_index'][off]
            a=np.stack([banks[j][k] for j,k in zip(ids[0],idx[0])]).astype(float)
            b=np.stack([banks[j][k] for j,k in zip(ids[1],idx[1])]).astype(float)
            np.testing.assert_allclose(raw['endpoint_length'][off],np.linalg.norm(a-b,axis=1)/1280,rtol=1e-12)
        duplicate=np.zeros((2,1280),np.float32)
        from DINOv3.MADEqual.aebad_s_target_local_segment.geometry import geometry_layer
        value,diag=geometry_layer(np.ones((8,3),np.float32),np.zeros((8,3),np.int64),np.full(3,2,np.float32),[duplicate]*8)
        np.testing.assert_array_equal(value['endpoint_support'],np.tile(np.array([[1],[2]],np.int8),(1,3)))
        self.assertEqual(diag['degenerate_count'],3);np.testing.assert_array_equal(value['K8_SEGMENT'],value['K8_UNION'])

    def test_triangle_failure_is_not_silently_clipped(self):
        from DINOv3.MADEqual.aebad_s_target_local_segment.geometry import bounded_distances,NumericalError
        with self.assertRaises(NumericalError):bounded_distances(np.array([1]),np.array([1]),np.array([100]),np.array([2]))

    def test_support_grid_dimensions_and_labels_not_method_inputs(self):
        raw,_=raw_fixture();out,_=m.predict(raw,(3,3))
        self.assertEqual(out['pair_gate'].shape,(4,3,3))
        self.assertNotIn('label',__import__('inspect').signature(m.predict).parameters)
        np.testing.assert_array_equal(m.predict(copy.deepcopy(raw),(3,3))[0]['pair_distance'],out['pair_distance'])


class IntegrationTests(TemporaryCase):
    def test_original_observer_capture_integration_with_explicit_model_boundary_mocks(self):
        self._capture_boundary(False)

    def test_trusted_Q_retrieval_only_with_explicit_Exact_boundary_mock(self):
        self._capture_boundary(True)

    def _capture_boundary(self,reuse_Q):
        # These mocks replace Torch/host/extraction/Exact only. CaptureLog,
        # identity, geometry_layer, NPZ provenance and CURRENT checks are real.
        ctx,raw=self.fixture();root=ctx['paths']['cache_root'];state=ip.read_json(root/'cache_status.json')
        kept=[]
        for r in state['records']:
            if r['domain']=='illumination':(root/r['file']).unlink()
            else:kept.append(r)
        state['records']=kept;_,banks=raw_fixture();domain=ctx['catalog']['domains']['illumination']
        for s,bank in zip(domain['supports'],banks):
            file=ctx['paths']['data']/s['relative_path'];file.parent.mkdir(parents=True,exist_ok=True);Image.new('RGB',(48,48),(s['support_index'],1,1)).save(file);s['image_sha256']=ip.sha(file)
            if reuse_Q:
                path=root/f'support_features/illumination/{s["support_index"]:03d}.npz';path.parent.mkdir(parents=True,exist_ok=True)
                np.savez(path,protocol_id='mad_equal_aebad_s_target_fewshot_seed42_v1',relative_path=s['relative_path'],grid=[1,len(bank)],**{f'layer_{l}':bank for l in m.LAYERS})
                state['supports'].append(dict(domain='illumination',support_index=s['support_index'],file=path.relative_to(root).as_posix(),sha256=ip.sha(path),relative_path=s['relative_path'],image_sha256=s['image_sha256']))
        ip.write_json(root/'cache_status.json',state)
        memory=ctx['paths']['memory'];memory.write_bytes(b'mocked trusted memory');ctx['catalog']['memory_sha256']=ip.sha(memory)
        ctx['paths'].update(checkpoint=self.root/'weights',dinov3=self.root/'dinov3',official=self.root/'official',formal_dir=self.root/'formal')
        ctx['paths']['checkpoint'].write_bytes(b'mocked checkpoint');ctx['catalog']['checkpoint_sha256']=ip.sha(ctx['paths']['checkpoint'])
        for key,name in (('dinov3','hubconf.py'),('official','config.json'),('formal_dir','memory_manifest.json')):
            ctx['paths'][key].mkdir();(ctx['paths'][key]/name).write_text('{"host_config":{}}',encoding='utf8')
        class Tensor:
            def __init__(self,a):self.a=np.asarray(a)
            def to(self,*a,**k):return self
            def numpy(self):return self.a
        torch=SimpleNamespace(__version__='MOCK_NOT_TORCH',version=SimpleNamespace(cuda='MOCK'),device=lambda *a:SimpleNamespace(type='cuda',index=0),
            set_float32_matmul_precision=lambda *a:None,backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),cudnn=SimpleNamespace(allow_tf32=True)),
            inference_mode=nullcontext,from_numpy=Tensor,cuda=SimpleNamespace(is_available=lambda:True,current_device=lambda:0,set_device=lambda *a:None,
                mem_get_info=lambda *a:(100*2**30,100*2**30),empty_cache=lambda:None,get_device_name=lambda *a:'MOCK_NO_GPU',max_memory_allocated=lambda *a:0,max_memory_reserved=lambda *a:0))
        def load_memory(path,holder,**kwargs):holder.prototype_embeddings={l:('source',l,0) for l in m.LAYERS};return True
        original=SimpleNamespace(_seed_everything=lambda:None,_load_memory=load_memory,PROTOCOL_ID='mad_equal_aebad_s_target_fewshot_seed42_v1',
            _support_tensors=lambda f,device:{l:[('target',l,j) for j in range(1,9)] for l in m.LAYERS})
        def extract(model,image,**kwargs):
            bank=banks[image.getpixel((0,0))[0]-1];return {l:bank for l in m.LAYERS},(1,len(bank))
        def write_support(path,**arrays):path.parent.mkdir(parents=True,exist_ok=True);np.savez(path,**arrays)
        support=SimpleNamespace(_extract_four_layers=mock.Mock(side_effect=extract),_atomic_npz=write_support)
        def observe(model,image,**kwargs):
            self.assertIsNone(kwargs['source_support']);kwargs['geometry_observer']((3,3),raw['positions'],(48,48))
            for off,l in enumerate(m.LAYERS):
                kwargs['nearest_observer']('source',l,0,raw['source_distance'][off],raw['source_index'][off])
                for j in range(8):kwargs['nearest_observer']('target',l,j+1,raw['target_distance'][j,off],raw['target_index'][j,off])
            return SimpleNamespace(official_raw=m.scatter(raw['source_distance'],raw['positions'],(3,3)),target_raw={8:m.scatter(np.minimum(raw['source_distance'],raw['target_distance'].min(0)),raw['positions'],(3,3))})
        scoring=SimpleNamespace(observe_fewshot_pil=mock.Mock(side_effect=observe))
        model=SimpleNamespace(preprocessing=SimpleNamespace(min_brightness_factor=1,max_brightness_factor=1,resize_factor=.625),
            patch_exec=SimpleNamespace(patch_size=640,patch_overlap=128,model_patch_size=16),evaluation_downscale=4,
            backbone=SimpleNamespace(dino=SimpleNamespace(training=False,parameters=lambda:[])))
        host=SimpleNamespace(load_host=mock.Mock(return_value=(model,torch.device('cuda'))))
        def exact(q,bank):
            pool,l,j=bank;off=m.LAYERS.index(l)
            d,idx=(raw['source_distance'][off],raw['source_index'][off]) if pool=='source' else (raw['target_distance'][j-1,off],raw['target_index'][j-1,off])
            return Tensor(d*1280),Tensor(idx)
        exact_mock=mock.Mock(side_effect=exact)
        modules={'torch':torch,'DINOv3.MADEqual.aebad_s_target_fewshot.runner':original,
            'DINOv3.MADEqual.robustad_target_fewshot_p0.scoring':scoring,'DINOv3.MADEqual.robustad_target_fewshot_p0.support':support,
            'DINOv3.MADEqual.aebad_s_sparc_p0.runtime':host,'DINOv3.relation_reliability.superadd_host':SimpleNamespace(_exact_top1_chunked=exact_mock)}
        query_dir=None
        if reuse_Q:
            query_dir=self.root/'query';query_dir.mkdir();records=[]
            for s in domain['samples']:
                file=s['sample_id']+'.npz';meta=dict(domain='illumination',sample_id=s['sample_id'],relative_path=s['relative_path'],image_sha256=s['image_sha256'],
                    extraction_identity=ctx['extraction_identity'],layers=list(m.LAYERS),dtype='float32',feature_units='original_1280d_no_normalization',grid=[3,3],processed_shape=[48,48],output_shape=[12,12])
                ip.save_npz(query_dir/file,dict(positions=raw['positions'],**{f'layer_{l}':np.zeros((9,1280),np.float32) for l in m.LAYERS}),meta)
                records.append(dict(domain='illumination',sample_id=s['sample_id'],relative_path=s['relative_path'],image_sha256=s['image_sha256'],file=file,sha256=ip.sha(query_dir/file)))
            ip.write_json(query_dir/'manifest.json',dict(format='original_query_features_v1',status='verified',extraction_identity=ctx['extraction_identity'],records=records))
        with mock.patch.dict(sys.modules,modules),mock.patch.object(ip,'context',return_value=ctx),mock.patch.object(ip.common,'available_ram',return_value=100*2**30):
            capture.run(ctx,['illumination'],query_features=query_dir)
        state=ip.cache_state(ctx);counts=state['last_invocation_counts']
        self.assertEqual(counts['exact_source_calls'],8);self.assertEqual(counts['exact_target_calls'],64)
        self.assertEqual(counts['query_extraction_calls'],0 if reuse_Q else 2);self.assertEqual(counts['support_extraction_calls'],0 if reuse_Q else 8)
        self.assertEqual(counts['query_cache_reuses'],2 if reuse_Q else 0)
        self.assertEqual(scoring.observe_fewshot_pil.call_count,0 if reuse_Q else 2);self.assertEqual(host.load_host.call_count,0 if reuse_Q else 1)
        self.assertEqual(exact_mock.call_count,72 if reuse_Q else 0)

    def test_complete_capture_resume_without_memory_model_or_features(self):
        ctx,_=self.fixture()
        with mock.patch.object(ip,'context',return_value=ctx),mock.patch.dict(sys.modules,{'torch':None}):
            capture.run(ctx,list(ip.DOMAINS))
            capture.run(ctx,list(ip.DOMAINS))
        state=ip.cache_state(ctx);self.assertEqual(len(state['invocations']),2)
        self.assertEqual(state['last_invocation_counts']['query_extraction_calls'],0)
        self.assertEqual(state['status'],'capture_complete')

    def test_modified_cache_rejected_before_state_reset(self):
        ctx,_=self.fixture();root=ctx['paths']['cache_root'];before=(root/'cache_status.json').read_bytes();r=ip.read_json(root/'cache_status.json')['records'][0]
        with (root/r['file']).open('ab') as f:f.write(b'changed')
        with self.assertRaisesRegex(ValueError,'modified'):capture.run(ctx,['illumination'])
        self.assertEqual((root/'cache_status.json').read_bytes(),before)

    def test_orphan_duplicate_missing_and_wrong_joint_identity_rejected(self):
        ctx,_=self.fixture();root=ctx['paths']['cache_root'];state=ip.read_json(root/'cache_status.json')
        ip.write_json(root/'cache_status.json',dict(state,records=state['records']*2))
        with self.assertRaisesRegex(ValueError,'duplicate'):ip.cache_state(ctx)
        ip.write_json(root/'cache_status.json',state);np.savez(root/'raw_cache/orphan.npz',x=[1])
        with self.assertRaisesRegex(ValueError,'orphan'):ip.cache_state(ctx)
        (root/'raw_cache/orphan.npz').unlink();(root/state['records'][0]['file']).unlink()
        with self.assertRaises((ValueError,FileNotFoundError)):ip.cache_state(ctx)
        ctx['identity']='other';self.assertRaises(ValueError,ip.cache_state,ctx)

    def test_predict_then_CPU_evaluate_actual_small_pipeline(self):
        ctx,_=self.fixture();pred=self.root/'pred';cpu=self.root/'cpu'
        with mock.patch.object(ip,'context',return_value=ctx),mock.patch.object(ip.Data,'gt',side_effect=AssertionError('GT entered prediction')):
            pr.run(ctx,pred,list(ip.DOMAINS),2)
        with mock.patch.object(ip,'context',return_value=ctx):an.run(ctx,pred,cpu,2,1<<30)
        status=ip.read_json(cpu/'status.json');self.assertEqual(status['status'],'cpu_evaluation_complete_aupro_pending')
        self.assertFalse(status['all_metrics_complete']);self.assertTrue(status['joint_domains_complete'])
        self.assertEqual(len(ip.csv_file(cpu/'metrics.csv')),9);self.assertTrue(all(not r['AUPRO'] for r in ip.csv_file(cpu/'metrics.csv')))
        an.verified_analysis(ctx,pred,cpu)

    def test_incomplete_predictions_cannot_enter_GT_evaluation(self):
        ctx,_=self.fixture();pred=self.root/'pred'
        with mock.patch.object(ip,'context',return_value=ctx):pr.run(ctx,pred,['illumination'],1)
        record=ip.read_json(pred/'prediction_manifest.json')['records'][0];(pred/record['file']).unlink()
        with mock.patch.object(ip.Data,'gt',side_effect=AssertionError('must not reach GT')):
            with self.assertRaises(ValueError):an.run(ctx,pred,self.root/'failed',1,1<<30)
        self.assertFalse((self.root/'failed').exists())

    def test_evaluation_label_changes_do_not_change_prediction(self):
        ctx,_=self.fixture()
        with mock.patch.object(ip,'context',return_value=ctx),mock.patch.object(ip.Data,'gt',side_effect=AssertionError('GT entered prediction')):
            pr.run(ctx,self.root/'before',['illumination'],1)
            for s in ctx['catalog']['domains']['illumination']['samples']:s['label']=1-s['label']
            pr.run(ctx,self.root/'after',['illumination'],1)
        self.assertEqual((self.root/'before/prediction_scores.csv').read_bytes(),(self.root/'after/prediction_scores.csv').read_bytes())
        before=ip.read_json(self.root/'before/prediction_manifest.json');after=ip.read_json(self.root/'after/prediction_manifest.json')
        for a,b in zip(before['records'],after['records']):
            aa,_=ip.load_npz(self.root/'before'/a['file'],a['sha256']);bb,_=ip.load_npz(self.root/'after'/b['file'],b['sha256'])
            for key in aa:np.testing.assert_array_equal(aa[key],bb[key])

    def test_missing_required_GT_cannot_publish_complete_metrics(self):
        ctx,_=self.fixture();pred=self.root/'pred'
        with mock.patch.object(ip,'context',return_value=ctx):pr.run(ctx,pred,['illumination'],1)
        s=ctx['catalog']['domains']['illumination']['samples'][1]
        (ctx['paths']['data']/s['relative_path'].replace('test/','ground_truth/',1)).unlink()
        with self.assertRaises(FileNotFoundError):an.run(ctx,pred,self.root/'no_GT',1,1<<30)
        self.assertFalse((self.root/'no_GT/analysis_manifest.json').exists());self.assertFalse((self.root/'no_GT/metrics.csv').exists())

    def test_formal_score_change_blocks_prediction(self):
        ctx,_=self.fixture();ctx['catalog']['domains']['illumination']['samples'][0]['union_score']+=1
        with self.assertRaisesRegex(ValueError,'CURRENT'):pr.run(ctx,self.root/'fail',['illumination'],1)
        self.assertFalse((self.root/'fail/prediction_manifest.json').exists())
        self.assertTrue(list((self.root/'fail/failures').glob('*.json')))

    def test_map_backend_drift_blocks_completion_before_GT(self):
        ctx,_=self.fixture();pred=self.root/'pred'
        with mock.patch.object(ip,'context',return_value=ctx):pr.run(ctx,pred,['illumination'],1)
        original=an.score_maps
        def changed(*a,**k):
            current,ch1,response,error=original(*a,**k)
            return current,ch1,response+1,error
        with mock.patch.object(an,'score_maps',side_effect=changed),mock.patch.object(ip.Data,'gt',side_effect=AssertionError('must not reach GT')):
            with self.assertRaisesRegex(ValueError,'backend changed'):an.run(ctx,pred,self.root/'drift',1,1<<30)
        self.assertFalse((self.root/'drift/metrics.csv').exists())

    def test_separate_domains_summary_same_joint_freeze(self):
        from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0 import summary
        ctx,_=self.fixture();directories=[]
        with mock.patch.object(ip,'context',return_value=ctx):
            for d in ip.DOMAINS:
                p=self.root/(d+'_pred');a=self.root/(d+'_cpu');pr.run(ctx,p,[d],1);an.run(ctx,p,a,1,1<<30);directories.append(a)
            summary.run(ctx,directories,self.root/'summary')
            with self.assertRaisesRegex(ValueError,'duplicate'):summary.run(ctx,[directories[0]]*2,self.root/'duplicate')
        self.assertEqual(len(ip.csv_file(self.root/'summary/metrics.csv')),9)
        self.assertFalse(ip.read_json(self.root/'summary/status.json')['all_metrics_complete'])

    def test_AUPRO_continuation_uses_verified_CPU_maps_with_backend_mock(self):
        from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0 import aupro
        import importlib.util
        ctx,_=self.fixture();pred=self.root/'pred';cpu=self.root/'cpu'
        with mock.patch.object(ip,'context',return_value=ctx):pr.run(ctx,pred,list(ip.DOMAINS),1);an.run(ctx,pred,cpu,1,1<<30)
        cuda=SimpleNamespace(is_available=lambda:True,current_device=lambda:0,set_device=lambda *a:None,
            get_device_properties=lambda *a:SimpleNamespace(total_memory=8*2**30),mem_get_info=lambda *a:(8*2**30,8*2**30),
            set_per_process_memory_fraction=lambda *a:None,reset_peak_memory_stats=lambda *a:None,
            max_memory_allocated=lambda *a:0,max_memory_reserved=lambda *a:0,get_device_name=lambda *a:'MOCK_NO_GPU')
        torch=SimpleNamespace(__version__='MOCK',version=SimpleNamespace(cuda='MOCK'),cuda=cuda,device=lambda *a:SimpleNamespace(type='cuda',index=0))
        backend=mock.Mock(return_value=.5);original_spec=importlib.util.find_spec
        with mock.patch.dict(sys.modules,{'torch':torch,'DINOv3.relation_reliability.official_metrics':SimpleNamespace(official_aupro_adeval=backend)}),\
                mock.patch('importlib.util.find_spec',side_effect=lambda name,*a:object() if name=='adeval' else original_spec(name,*a)),\
                mock.patch.object(ip,'context',return_value=ctx):
            out=aupro.run(ctx,pred,cpu,self.root/'aupro');aupro.verified_result(ctx,out)
        self.assertEqual(backend.call_count,6)
        self.assertTrue(all(c.kwargs['nstrips']==200 for c in backend.call_args_list))
        self.assertTrue(ip.read_json(out/'status.json')['all_metrics_complete'])
        self.assertEqual(ip.read_json(cpu/'status.json')['status'],'cpu_evaluation_complete_aupro_pending')

    def test_optional_original_distance_semantics_and_hash(self):
        ctx,raw=self.fixture();s=ctx['catalog']['domains']['illumination']['samples'][0];root=self.root/'references';root.mkdir()
        ctx['paths']['distance_reference_root']=root;ctx['catalog']['formal_sha256']={'sample_scores.csv':'scores'}
        union=np.minimum(raw['source_distance'],raw['target_distance'].min(0));path=root/'reference.npz'
        np.savez(path,protocol_id='mad_equal_aebad_s_target_fewshot_seed42_v1',cache_revision='fewshot_raw_v1',relative_path=s['relative_path'],
            output_shape=[12,12],official_raw=m.scatter(raw['source_distance'],raw['positions'],(3,3)),target_raw_8=m.scatter(union,raw['positions'],(3,3)))
        record=dict(domain=s['domain'],sample_id=s['sample_id'],relative_path=s['relative_path'],image_sha256=s['image_sha256'],file=path.name,sha256=ip.sha(path))
        value=dict(format='formal_fewshot_distance_reference_v1',status='verified',protocol_id='mad_equal_aebad_s_target_fewshot_seed42_v1',cache_revision='fewshot_raw_v1',
            memory_sha256='missing',sample_scores_sha256='scores',query=ip.QUERY,precision=ip.PRECISION,records=[record]);ip.write_json(root/'manifest.json',value)
        from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0 import references
        ctx['distance_references']=references.manifest(ctx,root);checks=[]
        self.assertTrue(references.verify(ctx,s,raw['source_distance'],union,raw['positions'],dict(grid=[3,3],output_shape=[12,12]),checks))
        self.assertEqual(len(checks),2)
        with self.assertRaisesRegex(ValueError,'reproduction'):references.verify(ctx,s,raw['source_distance'],union+.01,raw['positions'],dict(grid=[3,3],output_shape=[12,12]),[])
        with path.open('ab') as f:f.write(b'changed')
        with self.assertRaisesRegex(ValueError,'changed'):references.verify(ctx,s,raw['source_distance'],union,raw['positions'],dict(grid=[3,3],output_shape=[12,12]),[])

    def test_query_provenance_rejects_normalized_or_other_domain(self):
        ctx,_=self.fixture();s=ctx['catalog']['domains']['illumination']['samples'][0]
        path=self.root/'q.npz';arrays=dict(positions=np.arange(9,dtype=np.int64),**{f'layer_{l}':np.zeros((9,1280),np.float32) for l in m.LAYERS})
        meta=dict(extraction_identity=ctx['extraction_identity'],image_sha256=s['image_sha256'],domain=s['domain'],sample_id=s['sample_id'],relative_path=s['relative_path'],
            layers=list(m.LAYERS),dtype='float32',feature_units='normalized',grid=[3,3],output_shape=[12,12])
        ip.save_npz(path,arrays,meta)
        with self.assertRaisesRegex(ValueError,'provenance'):capture.query_arrays(ctx,self.root,dict(file=path.name,sha256=ip.sha(path)),s)


class AccountingTests(unittest.TestCase):
    def test_quartile_stable_ties_includes_NA(self):
        rows=[]
        for sid,area in [('b',1),('a',1),('c',2),('d',3),('e',4)]:
            for branch in m.BRANCHES:rows.append(dict(domain='view',sample_id=sid,instance_id=1,area_original_pixels=area,branch=branch,response_status='empty_ring'))
        groups=an.group_small_instances(rows,['view'])
        self.assertEqual({r['sample_id'] for r in groups if r['in_smallest_area_quartile']},{'a','b'})
        self.assertTrue(all(r['response_status']=='empty_ring' for r in groups))

    def test_old_margin_repair_and_new_loss_are_separate(self):
        rows=[]
        for iid,values in enumerate(((3,1,3),(3,1,2),(3,4,2),(3,2,1),(None,None,None)),1):
            for b,margin in zip(m.BRANCHES,values):rows.append(dict(domain='view',sample_id='q',instance_id=iid,branch=b,margin=margin,fixed_union_margin=margin,
                instance_max=5 if margin is not None else None,ring_max=5-margin if margin is not None else None,
                smallest_area_quartile=iid==1,response_status='available' if margin is not None else 'empty_ring'))
        details,summary=an.margin_repairs(rows);r=next(r for r in summary if r['group']=='all' and r['definition']=='margin')
        self.assertEqual(r['original_segment_loss'],3);self.assertEqual(r['old_loss_fully_repaired'],1);self.assertEqual(r['old_loss_partially_repaired'],1)
        self.assertEqual(r['new_PAIR_loss'],1);self.assertEqual(r['not_computable'],1);self.assertEqual(r['ring_recovers_but_defect_does_not'],2)

    def test_macro_unweighted_two_domains_background_excluded(self):
        rows=[dict(domain=d,branch=b,image_count=n,**{k:v for k in an.METRICS}) for d,n,v in [('illumination',265,.2),('view',364,.8)] for b in m.BRANCHES]
        result=an.append_macro(rows)
        self.assertEqual(len(result),9);self.assertTrue(all(r['image_AUROC']==.5 for r in result if r['domain'].startswith('macro')))
        self.assertEqual(len(an.append_macro(rows[:3])),3)
        self.assertEqual(len(an.append_macro(rows+[dict(domain='background',branch=m.U)])),7)
        deltas=an.metric_comparisons(result);self.assertTrue(all(r['direction']=='branch_minus_reference' for r in deltas))

    def test_region_R_signed_and_zero_retained_denominator(self):
        ctx=dict(scales=np.array([1,2,3,4],np.float32));a=dict(positions=np.arange(9,dtype=np.int64),union_distance=np.ones((4,9),np.float32),
            segment_distance=np.ones((4,9),np.float32),pair_distance=np.ones((4,9),np.float32),pair_gate=np.ones((4,3,3),bool))
        rows=an.mechanisms(ctx,dict(domain='view',sample_id='x',label=0),np.zeros((12,12),bool),a,dict(grid=[3,3]))
        self.assertTrue(all(r['extra_reduction_retained_ratio'] is None for r in rows))
        a['segment_distance']+=1e-8 # preserve signed mathematical diagnostics, no max(0).
        a['pair_distance'][:]=1.1
        rows=an.mechanisms(ctx,dict(domain='view',sample_id='x',label=0),np.zeros((12,12),bool),a,dict(grid=[3,3]))
        self.assertTrue(all(r['R_value']<0 for r in rows if r['branch']==m.P))

    def test_exact_pixels_ties_match_sklearn(self):
        from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic.metrics import exact_binary
        from sklearn.metrics import roc_auc_score,average_precision_score
        y=np.array([0,1,0,1,0,1,1,0]);s=np.array([1,1,2,2,0,1,3,3],np.float32)
        r=exact_binary(y,s,memory_budget_bytes=1<<30,chunk_size=2)
        self.assertAlmostEqual(r['AUROC'],roc_auc_score(y,s));self.assertAlmostEqual(r['AP'],average_precision_score(y,s))


class ActualInputsTests(unittest.TestCase):
    def test_formal_domain_sets_support_order_exclusion_and_no_torch(self):
        ctx=ip.context(ip.defaults(ROOT),verify_data=False)
        self.assertEqual([(d,len(v['samples']),v['normal_count'],v['anomaly_count']) for d,v in ctx['catalog']['domains'].items()],
            [('illumination',265,67,198),('view',364,84,280)])
        for d in ctx['catalog']['domains'].values():
            self.assertEqual([s['support_index'] for s in d['supports']],list(range(1,9)))
            self.assertFalse({s['relative_path'] for s in d['supports']} & {s['relative_path'] for s in d['samples']})
        self.assertNotIn('torch',sys.modules)

    def test_preflight_imports_with_Torch_explicitly_forbidden(self):
        code='''import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,*args):
  if fullname=="torch" or fullname.startswith("torch."):raise AssertionError("Torch entered CPU import")
sys.meta_path.insert(0,Block())
from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0 import runner,analysis,summary
assert runner.parser().parse_args([]).stage=="preflight"
print("CPU_IMPORT_OK")
'''
        p=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr);self.assertIn('CPU_IMPORT_OK',p.stdout)

    def test_real_background_fixed12_PAIR_gates_distances_CH1_equivalent(self):
        from DINOv3.MADEqual.aebad_s_neighborhood_segment_p0 import inputs as old
        from DINOv3.MADEqual.aebad_s_target_fewshot.replay12 import CASES
        paths=old.defaults(ROOT);ctx=ip.context(ip.defaults(ROOT),verify_data=False)
        pred=ROOT/'DINOv3/outputs/mad_equal/aebad_s_neighborhood_segment_local_20261016_134948_predict'
        cpu=ROOT/'DINOv3/outputs/mad_equal/aebad_s_neighborhood_segment_local_20261016_134948_cpu'
        records={r['sample_id']:r for r in ip.read_json(pred/'prediction_manifest.json')['records']}
        scores={(r['sample_id'],r['branch']):float(r['image_score']) for r in ip.csv_file(cpu/'per_image_scores.csv')}
        maximum=0.;patches=0
        with old.Archive(paths['raw_archive']) as a:
            capture_records={r['sample_id']:r for r in a.json(old.CROOT+'/status.json')['records']}
            segments={(r['sample_id'],r['layer']):r for r in a.json(old.SROOT+'/predictions.json')['records']}
            for case in CASES:
                sid=case['sample_id'] if isinstance(case,dict) else case[0]
                r=capture_records[sid];raw,meta=a.npz(old.CROOT+'/'+r['file'],r['sha256']);geometry=[]
                for l in m.LAYERS:
                    r=segments[(sid,l)];v,_=a.npz(old.SROOT+'/'+r['file'],r['sha256']);order=np.argsort(v['positions'])[raw['positions']]
                    geometry.append({k:v[k][...,order] for k in ('segment_distance','endpoint_support','endpoint_index','alpha','endpoint_length')})
                for key in geometry[0]:raw[key]=np.stack([v[key] for v in geometry])
                actual,_=m.predict(raw,tuple(meta['grid']));saved,_=ip.load_npz(pred/records[sid]['file'],records[sid]['sha256'])
                for key in ('pair_distance','pair_gate','pair_votes','neighbor_count'):np.testing.assert_array_equal(actual[key],saved[key],err_msg=sid)
                for b,key in ((m.U,'union_distance'),(m.S,'segment_distance'),(m.P,'pair_distance')):
                    _,peak,local=statistics(actual[key],raw['positions'],tuple(meta['grid']),tuple(meta['output_shape']),ctx['medians'],ctx['scales'])
                    error=abs(ch1_score(peak,local,ctx['state'])-scores[(sid,b)]);maximum=max(maximum,error);self.assertLessEqual(error,1e-5)
                patches+=actual['pair_distance'].size
        print(f'BACKGROUND12_PAIR_EQUIVALENCE: images=12 layer_patches={patches} max_CH1_abs_error={maximum:.9g}')


if __name__=='__main__':unittest.main()
