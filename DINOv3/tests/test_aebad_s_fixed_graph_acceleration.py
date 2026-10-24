"""Actual production mixed search on Torch CPU; no CUDA/model/real data."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tarfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from DINOv3.MADEqual.aebad_s_fixed_graph_acceleration_p0 import search, continuation as c
from DINOv3.MADEqual.aebad_s_fixed_target_graph_p0 import numerics, graph, protocol

ROOT = Path(__file__).resolve().parents[2]


class MixedSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        cls.torch.set_num_threads(cls.threads)

    def compare(self, q, nodes, edges, masks, union=None, **blocks):
        if union is None:
            union = np.full(len(q), 100., np.float32)
        ref = numerics.scan_segments(q, nodes, edges, masks, union, device='torch:cpu',
                                     query_chunk=3, edge_chunk=5, reference_chunk=4)
        got = search.scan_segments(q, nodes, edges, masks, union, device='torch:cpu', **blocks)
        np.testing.assert_array_equal(got[0], ref[0])
        np.testing.assert_array_equal(got[1], ref[1])
        return got

    def test_random_original_dimension_and_different_blocks(self):
        for seed in range(5):
            rng = np.random.default_rng(seed)
            q = rng.normal(size=(8,1280)).astype(np.float32)
            nodes = rng.normal(size=(17,1280)).astype(np.float32)
            edges = np.array([(i,j) for i in range(8) for j in range(8,17)])
            masks = rng.random((3,len(edges))) < .4
            self.compare(q, nodes, edges, masks, query_chunk=4, edge_chunk=11, reference_chunk=6)

    def test_active_compaction_preserves_original_edge_ids_and_ties(self):
        nodes = np.array([[-1,0],[1,0],[-2,0],[2,0],[0,5],[0,6]],np.float32)
        q = np.array([[0,1],[0,0]],np.float32)
        edges = np.array([[4,5],[2,3],[0,1],[0,1]])
        masks = np.array([[0,0,1,1],[0,1,0,0],[0,0,0,0]],bool)
        got = self.compare(q,nodes,edges,masks,edge_chunk=2,reference_chunk=2)
        self.assertEqual(got[1][0,0],2)
        self.assertEqual(got[2]['active_edges'],3)
        self.assertEqual(got[2]['active_nodes'],4)

    def test_Source_anchor_exact_zero_and_all_false_mask(self):
        nodes=np.array([[-1,0],[1,0],[3,0]],np.float32)
        q=np.array([[0,0],[0,1]],np.float32)
        self.compare(q,nodes,[[0,1],[1,2]],[[1,1],[0,0]],np.zeros(2,np.float32))

    def test_large_common_offset_cancellation_and_short_segment(self):
        for offset in (1e3,1e5,1e8):
            spacing = np.spacing(np.float32(offset))
            nodes=np.array([[offset,offset],[offset+8*spacing,offset]],np.float32)
            q=np.array([[offset+4*spacing,offset+spacing],[offset+spacing,offset]],np.float32)
            self.compare(q,nodes,[[0,1]],[[1]],edge_chunk=1,query_chunk=1,reference_chunk=1)

    def test_subnormal_and_FP32_dot_overflow_use_conservative_candidates(self):
        for scale in (1e-22,1e20):
            nodes=np.array([[-scale,0],[scale,0],[scale,scale]],np.float32)
            q=np.array([[0,scale/2]],np.float32)
            self.compare(q,nodes,[[0,1],[1,2]],[[1,1]],np.array([1e25],np.float32))

    def test_global_screen_retains_all_near_ties(self):
        nodes=np.array([[-1,0],[1,0],[-1,1e-7],[1,1e-7],[-1,-1e-7],[1,-1e-7]],np.float32)
        q=np.array([[0,0],[0,5e-8],[0,-5e-8]],np.float32)
        for block in (1,2,9):
            self.compare(q,nodes,[[0,1],[2,3],[4,5]],[[1,1,1],[0,1,1]],edge_chunk=block)

    def test_global_bound_avoids_block_local_provisional_winners(self):
        q=np.array([[0,0]],np.float32)
        nodes=np.array([[x,y] for y in range(1,25) for x in (-1,1)],np.float32)
        edges=np.arange(len(nodes)).reshape(-1,2)[::-1].copy()
        got=self.compare(q,nodes,edges,np.ones((1,len(edges)),bool),edge_chunk=1)
        old=numerics.scan_segments(q,nodes,edges,np.ones((1,len(edges)),bool),np.ones(1,np.float32),
                                  device='torch:cpu',edge_chunk=1)
        self.assertLess(got[2]['direct_pairs'],old[2]['direct_pairs'])

    def test_prepared_reuse_reordering_no_state_accumulation(self):
        rng=np.random.default_rng(8);nodes=rng.normal(size=(13,9)).astype(np.float32)
        q=rng.normal(size=(7,9)).astype(np.float32)
        edges=np.array([[i,i+1] for i in range(12)]);masks=np.ones((2,12),bool)
        union=np.ones(len(q),np.float32)
        p=search.PreparedSearch(nodes,edges,masks,'torch:cpu')
        first=search.scan_segments(q,nodes,edges,masks,union,device='torch:cpu',prepared=p)
        second=search.scan_segments(q[::-1].copy(),nodes,edges,masks,union,device='torch:cpu',prepared=p)
        np.testing.assert_array_equal(first[0],second[0][:,::-1])
        np.testing.assert_array_equal(first[1],second[1][:,::-1])
        with self.assertRaises(ValueError):
            search.scan_segments(q,nodes,edges,masks[::-1].copy()^True,union,device='torch:cpu',prepared=p)

    def test_inactive_invalid_length_is_not_searched_and_empty_pool(self):
        nodes=np.array([[0,0],[1,0]],np.float32);q=np.array([[0,1]],np.float32)
        got=search.scan_segments(q,nodes,[[0,0],[0,1]],[[0,1]],np.ones(1,np.float32),device='torch:cpu')
        self.assertEqual(got[1][0,0],1)
        result=search.scan_segments(q,nodes,np.empty((0,2),int),np.zeros((3,0),bool),
                                   np.ones(1,np.float32),device='torch:cpu')
        self.assertTrue((result[0]==1).all())
        self.assertEqual(result[2]['endpoint_pairs'],0)

    def test_screen_interval_contains_independent_FP64_endpoint_distances(self):
        rng=np.random.default_rng(29)
        for scale,offset in ((1,0),(1,1e5),(1e-18,0),(1e18,0)):
            nodes=(rng.normal(size=(11,1280))*scale+offset).astype(np.float32)
            query=(rng.normal(size=(6,1280))*scale+offset).astype(np.float32)
            p=search.PreparedSearch(nodes,[[i,i+1] for i in range(10)],np.ones((1,10),bool),'torch:cpu')
            q=self.torch.tensor(query);qnorm=(q.double()**2).sum(-1)
            with search.strict_screen(self.torch,p.dev):
                d,e=p.endpoint(q,qnorm,0,len(p.x))
            ref=((q.double()[:,None]-p.x.double()[None])**2).sum(-1)
            self.assertTrue(((d-e<=ref)&(ref<=d+e)).all().item())

    def test_precision_context_restores_settings_even_on_error(self):
        old=self.torch.get_float32_matmul_precision();tf32=self.torch.backends.cuda.matmul.allow_tf32
        try:
            self.torch.set_float32_matmul_precision('high')
            saved=self.torch.get_float32_matmul_precision();flag=self.torch.backends.cuda.matmul.allow_tf32
            with self.assertRaises(RuntimeError):
                with search.strict_screen(self.torch,self.torch.device('cpu')):
                    self.assertEqual(self.torch.get_float32_matmul_precision(),'highest')
                    raise RuntimeError('injected')
            self.assertEqual(self.torch.get_float32_matmul_precision(),saved)
            self.assertEqual(self.torch.backends.cuda.matmul.allow_tf32,flag)
        finally:
            self.torch.set_float32_matmul_precision(old)
            self.torch.backends.cuda.matmul.allow_tf32=tf32

    def test_actual_seven_branch_binding_matches_original_and_keeps_globals(self):
        rng=np.random.default_rng(1);q=rng.normal(size=(7,1280)).astype(np.float32)
        banks={i:rng.normal(size=(5,1280)).astype(np.float32) for i in range(1,9)}
        nodes,off,*_=graph.layout(banks)
        pool=dict(edges=np.array([[0,5],[1,6],[4,14]]),rank=np.ones(3),length=np.full(3,10.),
                  eta=np.full(3,.5),witness_support=np.full(3,4),witness_row=np.zeros(3,int))
        limits={(k,z):20. for k in protocol.KS for z in protocol.QUANTILES}
        choice=dict(main=dict(selected=dict(config=[10,.5,.75])),paired_config=[10,.5,None],
                    plain=dict(selected=dict(config=[10,.5,None])),lambda_by_layer=[.25]*4)
        a=SimpleNamespace(device='torch:cpu',query_chunk=3,edge_chunk=2,reference_chunk=4)
        source=self.torch.tensor(rng.normal(size=(9,1280)).astype(np.float32))
        oldscan=c.old.layer_branches.__globals__['scan_segments']
        bound=c.bind(c.old.layer_branches,scan_segments=search.scan_segments)
        original=c.old.layer_branches(a,'background',7,q,source,banks,pool,limits,choice)
        new=bound(a,'background',7,q,source,banks,pool,limits,choice)
        for b in protocol.BRANCHES:
            np.testing.assert_array_equal(new[0][b],original[0][b])
        self.assertIs(c.old.layer_branches.__globals__['scan_segments'],oldscan)


class Contracts(unittest.TestCase):
    def test_review_pack_excludes_scratch_and_raw_payloads_but_keeps_reports(self):
        from DINOv3.MADEqual.aebad_s_fixed_graph_acceleration_p0 import runner
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            out=Path(temp)/'run';out.mkdir()
            reports={'objective_report.md':b'report','metrics.csv':b'branch,value\nU,1\n',
                     'status.json':b'{"status":"failed"}', 'logs/evaluate.log':b'error',
                     'evaluation/view/map_manifest.csv':b'sample,sha\n',
                     'evaluation/view/metric.seal.json':b'{}',
                     'figures/case.png':b'png', 'logs/evaluate.exitcode':b'2\n'}
            for name,value in reports.items():
                p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(value)
            excluded=('pixel_scratch/view/large.bin','pixel_scratch/view/diagnostic.json',
                      'nested/_scratch/debug.log','nested/other_scratch/data.csv',
                      'scratch_more/state.json','nested/scratch/plot.png',
                      'accidental_raw.npz','leftover.bin','stage.pending')
            for name in excluded:
                p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'not a review input')
            # Large logical mmap file; sparse construction avoids allocating
            # a real experiment-sized fixture. Its contents must not be read.
            large=out/'pixel_scratch/view/large.bin'
            with large.open('wb') as f:f.truncate(64<<20)
            with contextlib.redirect_stdout(io.StringIO()):result=runner.pack(SimpleNamespace(output_root=out))
            with tarfile.open(result['archive'],'r:gz') as archive:
                self.assertEqual(set(archive.getnames()),{'run/'+name for name in reports})
                for name,value in reports.items():
                    self.assertEqual(archive.extractfile('run/'+name).read(),value)
            self.assertEqual(large.stat().st_size,64<<20)
            self.assertTrue(all((out/name).exists() for name in excluded))

    def test_failed_evaluation_launcher_review_omits_abandoned_mmaps(self):
        from DINOv3.MADEqual.aebad_s_fixed_graph_acceleration_p0 import runner
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root=Path(temp);out=root/'run'
            def stage(name,command,run_root):
                if name=='evaluate':
                    p=out/'pixel_scratch/view/labels.bin';p.parent.mkdir(parents=True)
                    p.write_bytes(b'abandoned mmap')
                    c.st.write_json(out/'pixel_scratch/debug.json',dict(temporary=True))
                    p=out/'logs/evaluate.log';p.parent.mkdir(parents=True);p.write_text('MemoryError')
                    raise MemoryError('evaluation allocation failed')
            with (patch.object(c,'check_outputs'),
                  patch('DINOv3.MADEqual.aebad_s_support_witness_segment_p0.autodl.run',side_effect=stage),
                  contextlib.redirect_stdout(io.StringIO())):
                with self.assertRaisesRegex(MemoryError,'allocation failed'):
                    runner.main(['all','--parent-run',str(root/'parent'),'--output-root',str(out),
                                 '--cache-root',str(root/'cache')])
            with tarfile.open(root/'run_review.tar.gz','r:gz') as archive:
                self.assertEqual(set(archive.getnames()),{'run/pipeline_status.json','run/logs/evaluate.log'})
                saved=json.load(archive.extractfile('run/pipeline_status.json'))
                self.assertFalse(saved['success']);self.assertEqual(saved['stage'],'evaluate')
            self.assertTrue((out/'pixel_scratch/view/labels.bin').exists())

    def test_launcher_rejects_parent_output_before_writing_pipeline_status(self):
        from DINOv3.MADEqual.aebad_s_fixed_graph_acceleration_p0 import runner
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root=Path(temp);parent=root/'parent'
            config={k:str(root/k) for k in ('cache_root','graph_root','prediction_root','data','checkpoint','official_root','dinov3_root')}
            c.st.write_json(parent/'setup.json',dict(configuration=config))
            before=c.st.sha(parent/'setup.json')
            with self.assertRaisesRegex(ValueError,'overlaps'):
                runner.main(['all','--parent-run',str(parent),'--output-root',str(parent),'--cache-root',str(root/'new_cache')])
            self.assertFalse((parent/'pipeline_status.json').exists())
            self.assertEqual(c.st.sha(parent/'setup.json'),before)

    def test_default_preflight_does_not_import_Torch_or_create_output(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root=Path(temp);out=root/'new';cache=root/'cache'
            code=f'''import sys\nfrom DINOv3.MADEqual.aebad_s_fixed_graph_acceleration_p0.runner import main\nmain(['--parent-run',{str(root/'absent')!r},'--output-root',{str(out)!r},'--cache-root',{str(cache)!r}])\nassert 'torch' not in sys.modules\n'''
            run=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,text=True,capture_output=True)
            self.assertEqual(run.returncode,0,run.stderr)
            self.assertIn('"torch_imported": false',run.stdout)
            self.assertFalse(out.exists());self.assertFalse(cache.exists())

    def test_smoke_cannot_accept_incomplete_or_modified_checks(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            a=SimpleNamespace(run_root=Path(temp))
            c.st.write_json(a.run_root/'setup.json',dict(identity='test'))
            value=dict(identity='test',engine=search.ENGINE,verified=True,checks=[])
            value['seal']=c.st.digest(value);c.st.write_json(a.run_root/'mixed_smoke.json',value)
            with self.assertRaises(ValueError):c.require_smoke(a)


class ContinuationIntegration(unittest.TestCase):
    """Real sealed files, linking, seven-branch prediction, scores and resume.

    RGB extraction and Source loading are tiny substitutes. NN, graph search,
    distance records, original scoring and validation execute their real code.
    """
    def test_frozen_parent_reuse_predictions_and_independent_resume(self):
        import torch
        from PIL import Image
        from DINOv3.MADEqual.aebad_s_fixed_target_graph_p0 import runtime, prediction
        st=c.st
        threads=torch.get_num_threads();torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory(dir=ROOT) as temp:
                root=Path(temp);parent=root/'parent';cache=root/'old_cache';code=root/'project';data=root/'data'
                code.mkdir();(code/'original.py').write_text('# original frozen implementation\n')
                official=code/'official';official.mkdir();st.write_json(official/'config.json',{})
                dino=code/'dino';dino.mkdir();(dino/'hubconf.py').write_text('# fixture entry\n')
                (data/'train/good').mkdir(parents=True)
                a=SimpleNamespace(run_root=parent,cache_root=cache,graph_root=cache/'graph',prediction_root=cache/'prediction',
                    query_cache_root=cache/'queries',evaluation_scratch=parent/'scratch',data=data,checkpoint=root/'weight',
                    official_root=official,dinov3_root=dino,query_mode='stream',reuse_fresh_run=None,
                    device='cpu',query_chunk=4,reference_chunk=16,edge_chunk=8,witness_edge_chunk=8,
                    seed=42,evaluation_scope='primary',cpu_workers=2,limit=1)
                samples=[];supports={}
                for d in protocol.DOMAINS:
                    supports[d]=[]
                    for h in range(1,9):
                        supports[d].append(dict(domain=d,support_index=h,relative_path=f'{d}/support{h}.png'))
                    for y in (0,1):
                        relative=f'test/{"bad" if y else "good"}/{d}/q{y}.png'
                        p=data/relative;p.parent.mkdir(parents=True,exist_ok=True)
                        Image.fromarray(np.full((64,64,3),y*10,np.uint8)).save(p)
                        sample=dict(domain=d,sample_id=f'q{y}',relative_path=relative,label=y,
                                    image_height=64,image_width=64,image_sha256=st.sha(p))
                        if y:
                            p=data/relative.replace('test/','ground_truth/',1);p.parent.mkdir(parents=True,exist_ok=True)
                            mask=np.zeros((64,64),np.uint8);mask[24:32,24:32]=255;Image.fromarray(mask).save(p)
                            sample['gt_sha256']=st.sha(p)
                        samples.append(sample)
                st.write_json(parent/'fresh_catalog.json',dict(data_root=str(data),source=[],samples=samples,supports=supports))
                catalog_sha=st.sha(parent/'fresh_catalog.json')
                st.write_json(parent/'fresh_catalog.seal.json',dict(sha256=catalog_sha))
                config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items() if k!='limit'}
                config.update(code={'original.py':st.sha(code/'original.py')},software={'numpy':c.importlib.metadata.version('numpy')},host_config={},
                              external_source_code={'dinov3/hubconf.py':st.sha(dino/'hubconf.py')})
                setup=dict(configuration=config,catalog_sha256=catalog_sha,cache=str(cache));setup['identity']=st.digest(setup)
                identity=setup['identity'];st.write_json(parent/'setup.json',setup);st.write_json(parent/'run_config.json',config)
                state=dict(peak=dict(medians=[0]*4,scales=[1]*4),local=dict(medians=[0]*4,scales=[1]*4))
                state['state_sha256']=st.digest(state);st.write_json(parent/'image_calibration_state.json',state)
                st.seal(cache/'source_patch_mad.npz',identity,dict(medians=np.zeros(4,np.float32),scales=np.ones(4,np.float32)),{})
                source={l:torch.full((11,1280),3,dtype=torch.float32) for l in protocol.LAYERS}
                st.seal(cache/'source_bank.npz',identity,{f'bank_{l}':v.numpy() for l,v in source.items()},{})
                for d in protocol.DOMAINS:
                    for h,row in enumerate(supports[d],1):
                        v=np.zeros((9,1280),np.float32)
                        if h<3:v[:,0]=(-2 if h==1 else 2)
                        else:v[:,1]=h*.25
                        st.seal(cache/'supports'/d/f'{h:03d}.npz',identity,
                            dict(relative_path=np.asarray(row['relative_path']),grid=np.array([3,3]),
                                 **{f'layer_{l}':v.copy() for l in protocol.LAYERS}),{})
                    choice=dict(identity=identity,main=dict(selected=dict(config=[10,.5,.75])),paired_config=[10,.5,None],
                                plain=dict(selected=dict(config=[10,.5,None])),lambda_by_layer=[.25]*4)
                    choice['seal']=st.digest(choice);st.write_json(parent/'selection'/f'{d}.json',choice)
                    for l in protocol.LAYERS:
                        pool=dict(edges=np.array([[0,9]]),rank=np.array([1]),length=np.array([4.]),eta=np.array([.5]),
                                  witness_support=np.array([3]),witness_row=np.array([0]))
                        meta=dict(zero_length_removed=0,unavailable_scales=0,
                                  limits=[dict(k=k,q=z,limit=4.) for k in protocol.KS for z in protocol.QUANTILES])
                        st.seal(runtime.graph_path(a,d,l,0),identity,pool,meta)
                assets=[cache/'source_bank.npz',cache/'source_patch_mad.npz']
                assets += list((cache/'supports').glob('*/*.npz'))
                frozen=dict(identity=identity,assets={str(p):st.read_json(p.with_suffix('.json')) for p in assets},
                            image_calibration_state_sha256=st.sha(parent/'image_calibration_state.json'))
                frozen['seal']=st.digest(frozen);st.write_json(parent/'frozen_assets.json',frozen)
                calls=[]
                def extract(model,image,device):
                    calls.append(1);q=np.zeros((9,1280),np.float32);q[:,1]=np.asarray(image)[0,0,0]/100+.01
                    return {f'layer_{l}':q.copy() for l in protocol.LAYERS},dict(grid=[3,3],positions=np.arange(9,dtype=np.int64),processed_shape=[48,48])
                with (patch.object(runtime,'precision'),patch.object(runtime,'host',return_value=(object(),'cpu')),
                      patch.object(runtime,'source_banks',return_value=source),patch.object(prediction,'extraction',side_effect=extract),
                      patch.object(c.ip,'ROOT',code),contextlib.redirect_stdout(io.StringIO())):
                    prediction.freeze(a)
                    self.assertEqual(len(calls),1)
                    before={str(p):st.sha(p) for base in (parent,cache) for p in base.rglob('*') if p.is_file()}
                    new=c.initialize(parent,root/'output',root/'new_cache','cpu',2)
                    self.assertNotEqual(c.ip.identity(new),identity)
                    result=c.predict(new)
                    self.assertEqual(result['new_images'],5);self.assertEqual(result['reused_images'],1)
                    self.assertEqual(len(calls),6)
                    self.assertEqual(len(prediction.verified(new)),6)
                    with patch.object(runtime,'host',side_effect=AssertionError('model reloaded')):
                        c.initialize(parent,root/'output',root/'new_cache','cpu',2)
                        result=c.predict(new)
                        self.assertEqual(result['new_images'],0)
                    origins=[]
                    for sample in samples:
                        _,meta,_=c.ip.verify_prediction(new,sample)
                        origins.append(meta['acceleration']['origin'])
                    self.assertEqual(origins.count('reused_original_FP64_prediction'),1)
                    self.assertEqual(origins.count('new_mixed_precision_prediction'),5)
                    from DINOv3.MADEqual.aebad_s_fixed_target_graph_p0 import evaluation,reporting
                    for d in protocol.DOMAINS:
                        evaluation.evaluate(new,d)
                    summary=reporting.summarize(new)
                    self.assertFalse(summary['all_metrics_complete'])
                    self.assertEqual(len(summary['pending']),21)  # Never invent AUPRO.
                    self.assertEqual(before,{str(p):st.sha(p) for base in (parent,cache) for p in base.rglob('*') if p.is_file()})
                    with patch.object(c.importlib.metadata,'version',return_value='changed'):
                        with self.assertRaisesRegex(ValueError,'software differs'):
                            c.initialize(parent,root/'output',root/'new_cache','cpu',2)
                    path=new.run_root/'selection'/'background.json';value=st.read_json(path);value['lambda_by_layer'][0]=.5;st.write_json(path,value)
                    with self.assertRaisesRegex(ValueError,'frozen acceleration input changed'):
                        c.initialize(parent,root/'output',root/'new_cache','cpu',2)
        finally:
            torch.set_num_threads(threads)


if __name__=='__main__':
    unittest.main()
