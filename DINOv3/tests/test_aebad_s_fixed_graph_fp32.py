"""Actual Torch CPU mathematical path; fused CUDA/Triton remains unverified."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from DINOv3.MADEqual.aebad_s_fixed_graph_fp32_p0 import search,service,runner
from DINOv3.MADEqual.aebad_s_fixed_graph_acceleration_p0 import continuation as mixed
from DINOv3.MADEqual.aebad_s_fixed_target_graph_p0 import prediction,runtime,protocol

ROOT=Path(__file__).resolve().parents[2]


def exhaustive(q,nodes,edges,masks,union):
    edges=np.asarray(edges);masks=np.asarray(masks,bool)
    values=np.sqrt(search.direct_fp32(q[:,None],nodes[edges[:,0]][None],nodes[edges[:,1]][None]))/np.float32(1280)
    distances=np.broadcast_to(union,(len(masks),len(q))).copy();winners=np.full(distances.shape,-1,np.int64)
    for k,mask in enumerate(masks):
        actual=np.where(mask,values,np.inf);idx=actual.argmin(1);best=actual[np.arange(len(q)),idx]
        take=best<union;distances[k,take]=best[take];winners[k,take]=idx[take]
    return distances,winners


class FP32(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch=torch;cls.threads=torch.get_num_threads();torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):cls.torch.set_num_threads(cls.threads)

    def compare(self,q,nodes,edges,masks,union=None,**kwargs):
        if union is None:union=np.ones(len(q),np.float32)
        reference=exhaustive(q,nodes,edges,masks,union)
        got=search.scan_segments(q,nodes,edges,masks,union,device='torch:cpu',**kwargs)
        np.testing.assert_allclose(got[0],reference[0],atol=1e-9,rtol=1e-6)
        np.testing.assert_array_equal(got[1],reference[1])
        return got

    def test_full_dimension_against_direct_reference(self):
        for seed in range(5):
            rng=np.random.default_rng(seed);q=rng.normal(size=(7,1280)).astype(np.float32)
            nodes=rng.normal(size=(19,1280)).astype(np.float32)
            edges=np.array([(i,j) for i in range(7) for j in range(7,19)])
            masks=rng.random((3,len(edges)))<.6
            self.compare(q,nodes,edges,masks,query_chunk=3,reference_chunk=2)

    def test_common_offset_screen_uses_raw_feature_direct_projection(self):
        for offset in (1e3,1e5,1e8):
            step=np.spacing(np.float32(offset))
            nodes=np.array([[offset,offset],[offset+16*step,offset],[offset+3*step,offset+2*step]],np.float32)
            q=np.array([[offset+8*step,offset+step],[offset+2*step,offset+step]],np.float32)
            self.compare(q,nodes,[[0,1],[1,2]],[[1,1],[0,1]],query_chunk=1)

    def test_duplicate_masks_and_canonical_mapping(self):
        q=np.array([[0,.1],[0,0]],np.float32)
        nodes=np.array([[-1,0],[1,0],[-2,0],[2,0],[1,5],[1,6]],np.float32)
        edges=np.array([[4,5],[2,3],[0,1],[0,1]])
        masks=np.array([[0,0,1,1],[0,0,1,1],[0,0,0,0]],bool)
        got=self.compare(q,nodes,edges,masks)
        self.assertEqual(got[2]['distinct_masks'],2)
        self.assertEqual(got[2]['active_nodes'],2)
        self.assertEqual(got[1][0,0],2)

    def test_anchor_zero_and_no_query_Top2_restriction(self):
        q=np.array([[0,0],[0,0]],np.float32)
        nodes=np.array([[0,.5],[0,-.6],[-2,0],[2,0]],np.float32)
        got=self.compare(q,nodes,[[2,3]],[[1]],np.array([0,1],np.float32))
        self.assertEqual(got[1][0,0],-1);self.assertEqual(got[1][0,1],0)

    def test_empty_branch_preserves_union_bits_without_square_root_roundtrip(self):
        nodes=np.array([[-1,0],[1,0]],np.float32)
        q=np.array([[0,1]]*20,np.float32)
        union=np.linspace(.00001,.002,20,dtype=np.float32)
        got=search.scan_segments(q,nodes,[[0,1]],[[1],[0]],union,device='torch:cpu')
        np.testing.assert_array_equal(got[0][1],union)
        self.assertTrue((got[1][1]==-1).all())

    def test_one_wide_GEMM_and_single_edge_pass(self):
        rng=np.random.default_rng(3);nodes=rng.normal(size=(20,7)).astype(np.float32)
        q=rng.normal(size=(9,7)).astype(np.float32);edges=np.array([[i,i+1] for i in range(19)])
        got=self.compare(q,nodes,edges,np.ones((3,len(edges)),bool),query_chunk=4,reference_chunk=1)
        self.assertEqual(got[2]['endpoint_gemms'],3)
        self.assertEqual(got[2]['full_edge_passes'],1)
        self.assertEqual(got[2]['geometry_dtype'],'FP32')
        self.assertEqual(got[2]['execution'],'Torch_CPU_reference')

    def test_device_boundary_endpoint_columns_are_contiguous_FP32(self):
        nodes=np.array([[0,0],[1,0],[1,1]],np.float32)
        p=search.PreparedSearch(nodes,[[0,1],[1,2]],[[1,1]],'torch:cpu')
        self.assertTrue(p.eu.is_contiguous());self.assertTrue(p.ev.is_contiguous())
        for t in (p.x,p.centered,p.c2,p.norms):self.assertEqual(t.dtype,self.torch.float32)
        np.testing.assert_array_equal(p.eu.numpy(),p.edges[:,0].numpy())

    def test_small_residual_and_close_candidates(self):
        nodes=np.array([[-2,0],[2,0],[-2,1e-6],[2,1e-6]],np.float32)
        q=np.array([[0,4e-7],[0,6e-7],[0,0]],np.float32)
        self.compare(q,nodes,[[0,1],[2,3]],[[1,1],[1,0]],query_chunk=2)

    def test_empty_and_inactive_zero_length_edges(self):
        nodes=np.array([[0,0],[1,0]],np.float32);q=np.array([[0,1]],np.float32);u=np.ones(1,np.float32)
        got=search.scan_segments(q,nodes,[[0,0],[0,1]],[[0,1]],u,device='torch:cpu')
        self.assertEqual(got[1][0,0],1)
        empty=search.scan_segments(q,nodes,np.empty((0,2),int),np.zeros((3,0),bool),u,device='torch:cpu')
        np.testing.assert_array_equal(empty[0],np.ones((3,1),np.float32))

    def test_unrepresentable_FP32_lengths_fail_without_FP64_fallback(self):
        nodes=np.array([[-1e20,0],[1e20,0]],np.float32)
        with self.assertRaises(FloatingPointError):search.PreparedSearch(nodes,[[0,1]],[[1]],'torch:cpu')

    def test_winner_in_later_partial_tile_and_canonical_tie(self):
        q=np.array([[0,0]],np.float32)
        nodes=np.array([[-1,3],[1,3],[-1,1],[1,1]],np.float32)
        edges=np.array([[0,1]]*520+[[2,3]]*13)
        masks=np.ones((3,len(edges)),bool)
        got=self.compare(q,nodes,edges,masks)
        self.assertEqual(got[1][0,0],520)

    def test_prepared_query_reordering_preserves_output_without_state(self):
        rng=np.random.default_rng(71);nodes=rng.normal(size=(12,13)).astype(np.float32)
        q=rng.normal(size=(9,13)).astype(np.float32);e=np.array([[i,i+1] for i in range(11)])
        m=np.ones((3,len(e)),bool);u=np.ones(len(q),np.float32)
        p=search.PreparedSearch(nodes,e,m,'torch:cpu')
        first=search.scan_segments(q,nodes,e,m,u,device='torch:cpu',prepared=p,query_chunk=3)
        second=search.scan_segments(q[::-1].copy(),nodes,e,m,u,device='torch:cpu',prepared=p,query_chunk=5)
        np.testing.assert_allclose(first[0],second[0][:,::-1],atol=1e-9)
        np.testing.assert_array_equal(first[1],second[1][:,::-1])

    def test_seven_branch_UNION_SEGMENT_unchanged_graph_diagnostics_FP32(self):
        q=np.zeros((3,1280),np.float32);q[:,1]=.13
        banks={i:np.zeros((5,1280),np.float32) for i in range(1,9)}
        banks[1][:,0]=-2;banks[2][:,0]=2
        for i in range(3,9):banks[i][:,1]=.5+(i-3)
        pool=dict(edges=np.array([[0,5]]),rank=np.array([1]),length=np.array([4.]),eta=np.array([.5]),
                  witness_support=np.array([3]),witness_row=np.array([0]))
        limits={(k,z):4. for k in protocol.KS for z in protocol.QUANTILES}
        choice=dict(main=dict(selected=dict(config=[10,.5,.75])),paired_config=[10,.5,None],
                    plain=dict(selected=dict(config=[10,.5,None])),lambda_by_layer=[.25]*4)
        a=SimpleNamespace(device='cpu',query_chunk=2,edge_chunk=8,reference_chunk=16)
        source=self.torch.full((11,1280),3,dtype=self.torch.float32)
        reference=prediction.layer_branches(a,'background',7,q,source,banks,pool,limits,choice)
        new=service.layer_branches(a,'background',7,q,source,banks,pool,limits,choice)
        for b in (protocol.U,protocol.S):np.testing.assert_array_equal(new[0][b],reference[0][b])
        for b in protocol.BRANCHES:np.testing.assert_allclose(new[0][b],reference[0][b],atol=1e-7,rtol=1e-6)
        for b in (protocol.G0,protocol.W,protocol.GL):self.assertEqual(new[1]['alpha_'+b].dtype,np.float32)
        self.assertEqual(new[1]['witness_distance'].dtype,np.float32)
        self.assertIn('original_discrete_seconds',new[2]['timing'])


class Contracts(unittest.TestCase):
    def test_no_original_global_changes_and_FP64_import_rejected(self):
        old_search=mixed.search;old_initialize=mixed.initialize
        c=service.get_service()
        self.assertIs(mixed.search,old_search);self.assertIs(mixed.initialize,old_initialize)
        self.assertIs(c.predict.__globals__['search'],search)
        with self.assertRaisesRegex(ValueError,'must not import'):
            c.initialize('unused','unused','unused','cpu',1,import_completed=True)
        self.assertTrue(runner.parser().parse_args(['--parent-run','p','--output-root','o','--cache-root','c']).no_import_completed)

    def test_default_preflight_does_not_import_GPU_libraries_or_create_output(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root=Path(temp)
            script=f'''import sys\nfrom DINOv3.MADEqual.aebad_s_fixed_graph_fp32_p0.runner import main\nmain(['--parent-run',{str(root/'absent')!r},'--output-root',{str(root/'out')!r},'--cache-root',{str(root/'cache')!r}])\nassert 'torch' not in sys.modules\nassert 'triton' not in sys.modules\n'''
            result=subprocess.run([sys.executable,'-B','-c',script],cwd=ROOT,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertFalse((root/'out').exists());self.assertFalse((root/'cache').exists())

    def test_protocol_write_records_FP32_and_keeps_original_storage_global(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            path=Path(temp)/'protocol.json'
            c=service.get_service();c.st.write_json(path,dict(geometry='old',branches=list(protocol.BRANCHES)))
            value=json.loads(path.read_text())
            self.assertEqual(value['prediction_precision'],'FP32')
            self.assertFalse(value['parent_FP64_predictions_imported'])
            self.assertEqual(value['branches'],list(protocol.BRANCHES))

    def test_CUDA_kernel_module_parses_without_importing_Triton(self):
        path=Path(search.__file__).with_name('kernels.py')
        compile(path.read_text(encoding='utf-8'),str(path),'exec')


class Integration(unittest.TestCase):
    """Real NPZ/identity/linking, FP32 prediction, original maps and CPU metrics.

    Backbone output and Source load use tiny substitutes. No CUDA/Triton or
    real feature extraction is implied by this integration fixture.
    """
    def test_reused_frozen_assets_new_FP32_predictions_and_resume(self):
        import torch
        from PIL import Image
        from DINOv3.MADEqual.aebad_s_fixed_target_graph_p0 import evaluation,reporting
        st=mixed.st;c=service.get_service();threads=torch.get_num_threads();torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory(dir=ROOT) as temp:
                root=Path(temp);parent=root/'parent';cache=root/'old_cache';project=root/'project';data=root/'data'
                project.mkdir();(project/'original.py').write_text('# frozen fixture implementation\n')
                official=project/'official';official.mkdir();st.write_json(official/'config.json',{})
                dino=project/'dino';dino.mkdir();(dino/'hubconf.py').write_text('# fixture entry\n')
                (data/'train/good').mkdir(parents=True)
                a=SimpleNamespace(run_root=parent,cache_root=cache,graph_root=cache/'graphs',prediction_root=cache/'predictions',
                    query_cache_root=cache/'queries',evaluation_scratch=parent/'scratch',data=data,checkpoint=root/'weight',
                    official_root=official,dinov3_root=dino,query_mode='stream',reuse_fresh_run=None,device='cpu',
                    query_chunk=4,reference_chunk=16,edge_chunk=8,witness_edge_chunk=8,seed=42,evaluation_scope='primary',cpu_workers=2)
                samples=[];supports={}
                for d in protocol.DOMAINS:
                    supports[d]=[dict(domain=d,support_index=h,relative_path=f'{d}/support{h}.png') for h in range(1,9)]
                    for y in (0,1):
                        relative=f'test/{"bad" if y else "good"}/{d}/q{y}.png'
                        p=data/relative;p.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(np.full((64,64,3),y*10,np.uint8)).save(p)
                        s=dict(domain=d,sample_id=f'q{y}',relative_path=relative,label=y,image_height=64,image_width=64,image_sha256=st.sha(p))
                        if y:
                            p=data/relative.replace('test/','ground_truth/',1);p.parent.mkdir(parents=True,exist_ok=True)
                            mask=np.zeros((64,64),np.uint8);mask[24:32,24:32]=255;Image.fromarray(mask).save(p);s['gt_sha256']=st.sha(p)
                        samples.append(s)
                st.write_json(parent/'fresh_catalog.json',dict(data_root=str(data),source=[],samples=samples,supports=supports))
                sha=st.sha(parent/'fresh_catalog.json');st.write_json(parent/'fresh_catalog.seal.json',dict(sha256=sha))
                config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
                config.update(code={'original.py':st.sha(project/'original.py')},software={},host_config={},
                              external_source_code={'dinov3/hubconf.py':st.sha(dino/'hubconf.py')})
                setup=dict(configuration=config,catalog_sha256=sha,cache=str(cache));setup['identity']=st.digest(setup)
                identity=setup['identity'];st.write_json(parent/'setup.json',setup);st.write_json(parent/'run_config.json',config)
                state=dict(peak=dict(medians=[0]*4,scales=[1]*4),local=dict(medians=[0]*4,scales=[1]*4))
                state['state_sha256']=st.digest(state);st.write_json(parent/'image_calibration_state.json',state)
                st.seal(cache/'source_patch_mad.npz',identity,dict(medians=np.zeros(4,np.float32),scales=np.ones(4,np.float32)),{})
                source={l:torch.full((11,1280),3,dtype=torch.float32) for l in protocol.LAYERS}
                st.seal(cache/'source_bank.npz',identity,{f'bank_{l}':x.numpy() for l,x in source.items()},{})
                for d in protocol.DOMAINS:
                    for h,row in enumerate(supports[d],1):
                        x=np.zeros((9,1280),np.float32)
                        if h<3:x[:,0]=-2 if h==1 else 2
                        else:x[:,1]=h*.25
                        st.seal(cache/'supports'/d/f'{h:03d}.npz',identity,dict(relative_path=np.asarray(row['relative_path']),
                            grid=np.array([3,3]),**{f'layer_{l}':x.copy() for l in protocol.LAYERS}),{})
                    choice=dict(identity=identity,main=dict(selected=dict(config=[10,.5,.75])),paired_config=[10,.5,None],
                                plain=dict(selected=dict(config=[10,.5,None])),lambda_by_layer=[.25]*4)
                    choice['seal']=st.digest(choice);st.write_json(parent/'selection'/f'{d}.json',choice)
                    for l in protocol.LAYERS:
                        pool=dict(edges=np.array([[0,9]]),rank=np.array([1]),length=np.array([4.]),eta=np.array([.5]),
                                  witness_support=np.array([3]),witness_row=np.array([0]))
                        st.seal(runtime.graph_path(a,d,l,0),identity,pool,dict(zero_length_removed=0,unavailable_scales=0,
                            limits=[dict(k=k,q=z,limit=4.) for k in protocol.KS for z in protocol.QUANTILES]))
                assets=[cache/'source_bank.npz',cache/'source_patch_mad.npz',*list((cache/'supports').glob('*/*.npz'))]
                frozen=dict(identity=identity,assets={str(p):st.read_json(p.with_suffix('.json')) for p in assets},
                            image_calibration_state_sha256=st.sha(parent/'image_calibration_state.json'))
                frozen['seal']=st.digest(frozen);st.write_json(parent/'frozen_assets.json',frozen)
                # A bogus old prediction must not be read/imported into FP32.
                p=a.prediction_root/'background/q0.npz';p.parent.mkdir(parents=True);p.write_bytes(b'old prediction not eligible')
                st.write_json(p.with_suffix('.json'),dict(sha256='bad'))
                before={str(p):st.sha(p) for r in (parent,cache) for p in r.rglob('*') if p.is_file()}
                calls=[]
                def extract(model,image,device):
                    calls.append(1);q=np.zeros((9,1280),np.float32);q[:,1]=np.asarray(image)[0,0,0]/100+.01
                    return {f'layer_{l}':q.copy() for l in protocol.LAYERS},dict(grid=[3,3],positions=np.arange(9,dtype=np.int64),processed_shape=[48,48])
                with (patch.object(mixed.ip,'ROOT',project),patch.object(runtime,'precision'),
                      patch.object(runtime,'host',return_value=(object(),'cpu')),patch.object(runtime,'source_banks',return_value=source),
                      patch.object(prediction,'extraction',side_effect=extract),contextlib.redirect_stdout(io.StringIO())):
                    new=c.initialize(parent,root/'output',root/'new_cache','cpu',2)
                    self.assertNotEqual(c.ip.identity(new),identity)
                    result=c.predict(new,limit=1);self.assertEqual(result['new_images'],1)
                    with self.assertRaises((FileNotFoundError,ValueError)):prediction.verified(new)
                    result=c.predict(new);self.assertEqual(result['new_images'],5)
                    self.assertEqual(len(calls),6);self.assertEqual(len(prediction.verified(new)),6)
                    for s in samples:
                        _,m,_=c.ip.verify_prediction(new,s)
                        self.assertEqual(m['acceleration']['origin'],'new_FP32_prediction')
                        self.assertFalse(m['acceleration']['parent_predictions_imported'])
                    with patch.object(runtime,'host',side_effect=AssertionError('unnecessary inference')):
                        c.initialize(parent,root/'output',root/'new_cache','cpu',2)
                        self.assertEqual(c.predict(new)['new_images'],0)
                    for d in protocol.DOMAINS:evaluation.evaluate(new,d)
                    self.assertFalse(reporting.summarize(new)['all_metrics_complete'])
                self.assertEqual(before,{str(p):st.sha(p) for r in (parent,cache) for p in r.rglob('*') if p.is_file()})
                self.assertEqual(st.read_json(new.run_root/'protocol.json')['prediction_precision'],'FP32')
        finally:torch.set_num_threads(threads)


if __name__=='__main__':unittest.main()
