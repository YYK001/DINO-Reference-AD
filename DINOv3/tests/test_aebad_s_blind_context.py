"""CPU fixtures only: no CUDA initialization, real data, extraction or GPU evaluator."""
import inspect,sys,tempfile,unittest,copy,json,subprocess
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from DINOv3.MADEqual.aebad_s_blind_context_p0 import protocol as p,context as c,inputs as ip,runtime as rt,training,runner,isolation,reporting
st=ip.st


def setUpModule():(ip.ROOT/'.tmp').mkdir(exist_ok=True)


def small_dictionary():
    rng=np.random.default_rng(7);images=np.repeat(np.arange(10),6);types=(images>=2).astype(np.int64)
    keys=c.describe(rng.normal(size=(len(images),1280)).astype(np.float32),c.projection(7))
    return dict(keys=keys,values=rng.normal(size=(len(images),1280)).astype(np.float32),images=images,types=types)


def original_patch_classes():
    """Compile the actual upstream classes without constructing/importing DINO."""
    import ast,math,itertools,torch
    source=ip.ROOT/'external_baselines/SuperADD/tracks/industrial/src/industrial/model.py'
    tree=ast.parse(source.read_text(encoding='utf-8'))
    classes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ('PreProcessing','PatchedExecution')]
    namespace=dict(torch=torch,np=np,ceil=math.ceil,floor=math.floor,product=itertools.product)
    exec(compile(ast.Module(body=classes,type_ignores=[]),str(source),'exec'),namespace)
    return namespace['PreProcessing'],namespace['PatchedExecution']


def tiny_original_host():
    """Real upstream crop/stitch geometry; only the expensive backbone is fake."""
    import torch
    _,PatchedExecution=original_patch_classes()
    class FakeDino:
        def __init__(self):self.calls=[]
        def get_intermediate_layers(self,crops,n,norm=False):
            self.calls.append(crops.clone())
            pooled=torch.nn.functional.avg_pool2d(crops[:,:1],16,16).flatten(2).transpose(1,2)
            global_context=crops.mean((1,2,3))[:,None,None]
            channels=torch.arange(1280,dtype=torch.float32)[None,None]*.001
            return tuple(pooled+global_context+channels+float(l)*.01 for l in n)
    dino=FakeDino()
    return SimpleNamespace(patch_exec=PatchedExecution(64,16,16),backbone=SimpleNamespace(dino=dino),layers=p.LAYERS)


class OriginalHostGeometry(unittest.TestCase):
    def test_real_preprocessor_keeps_1890_and_uneven_sizes(self):
        import torch
        from DINOv3.relation_reliability import superadd_host as shared
        PreProcessing,PatchedExecution=original_patch_classes()
        host=SimpleNamespace(preprocessing=PreProcessing('cpu',.625),patch_exec=PatchedExecution(640,128,16))
        self.assertEqual(shared.preprocessed_spatial_shape(host,3024,3024),(1890,1890))
        wrapper=shared._MinimumSpatialPreprocessing(host.preprocessing,host)
        value=wrapper(torch.ones(1,3,1056,1040))
        self.assertEqual(tuple(value.shape[-2:]),(660,650))
        geometry=c.mask_geometry(value.shape[-2:])
        self.assertEqual(geometry,dict(grid=[41,40],effective_pixel_shape=[656,640],ignored_trailing_pixels=[4,10]))
        self.assertEqual(c.masked_input(value,0).shape,value.shape)
        self.assertFalse(torch.cuda.is_initialized())

    def test_real_axis_rois_align_global_tokens_and_exclude_tail(self):
        _,PatchedExecution=original_patch_classes();execution=PatchedExecution(640,128,16)
        for dim in (640,641,650,660,1890,1903):
            inputs,predictions,results=execution.axis_patch_split(dim);covered=np.zeros(dim//16,np.uint8)
            for (lo,hi),(p0,p1),(r0,r1) in zip(inputs,predictions,results):
                self.assertEqual(lo%16,0);self.assertEqual(hi%16,0);self.assertLessEqual(hi,(dim//16)*16)
                self.assertEqual(lo//16+p0,r0);self.assertEqual(lo//16+p1,r1);covered[r0:r1]+=1
            np.testing.assert_array_equal(covered,1)
            self.assertEqual(max(hi for _,hi in inputs),(dim//16)*16)

    def test_1890_mask_covers_full_patches_once_and_preserves_unused_border(self):
        value=np.ones((1,1,1890,1890),np.float32);count=np.zeros((1888,1888),np.uint8)
        self.assertEqual(c.mask_geometry((1890,1890))['grid'],[118,118])
        for phase in range(4):
            masked=c.masked_input(value,phase);self.assertEqual(masked.shape,value.shape)
            np.testing.assert_array_equal(masked[...,-2:,:],value[...,-2:,:])
            np.testing.assert_array_equal(masked[...,:,-2:],value[...,:,-2:])
            count+=(masked[0,0,:1888,:1888]==0)
        np.testing.assert_array_equal(count,1)

    def test_nonmultiple_numpy_Torch_masks_match_without_resize(self):
        import torch
        value=np.random.default_rng(19).normal(size=(1,3,193,211)).astype(np.float32)
        for phase in range(4):
            expected=c.masked_input(value,phase);actual=c.masked_input(torch.from_numpy(value),phase)
            np.testing.assert_array_equal(actual.numpy(),expected)
            np.testing.assert_array_equal(expected[...,192:,:],value[...,192:,:])
            np.testing.assert_array_equal(expected[...,:,208:],value[...,:,208:])
        self.assertFalse(torch.cuda.is_initialized())

    def test_actual_original_crop_stitch_and_projection_on_nonmultiple_input(self):
        import torch
        host=tiny_original_host();a=SimpleNamespace(device='cpu')
        value=torch.ones(1,3,161,179);pos=np.arange(110,dtype=np.int64)
        with patch.object(rt,'model',return_value=host):
            keys,m=rt.masked_features(a,value)
        self.assertEqual(m['grid'],[10,11]);self.assertEqual(m['effective_pixel_shape'],[160,176]);self.assertEqual(m['ignored_trailing_pixels'],[1,3])
        self.assertEqual(m['masked_full_extractions'],4);np.testing.assert_array_equal(m['positions'],pos)
        self.assertEqual(len(host.backbone.dino.calls),4)
        iy=host.patch_exec.axis_patch_split(161)[0];ix=host.patch_exec.axis_patch_split(179)[0]
        for phase,crops in enumerate(host.backbone.dino.calls):
            expected=c.masked_input(value,phase)
            expected_crops=torch.cat([expected[:,:,y0:y1,x0:x1] for y0,y1 in iy for x0,x1 in ix])
            torch.testing.assert_close(crops,expected_crops,atol=0,rtol=0)
        for l in p.LAYERS:self.assertEqual(keys[l].shape,(110,128));self.assertTrue(np.isfinite(keys[l]).all())
        self.assertFalse(torch.cuda.is_initialized())

    def test_edge_interventions_use_actual_host_and_production_synthesizer(self):
        import torch
        from DINOv3.MADEqual.aebad_s_blind_context_p0.prediction import Synthesizer
        host=tiny_original_host();a=SimpleNamespace(device='cpu');d=small_dictionary();states={l:c.initial_head(l) for l in p.LAYERS}
        value=np.random.default_rng(8).normal(size=(1,3,161,179)).astype(np.float32);pos=109
        with patch.object(rt,'model',return_value=host),patch.object(rt,'dictionary',return_value=(d,{})):
            synthesizer=Synthesizer(a,states)
            def forward(masked):
                keys,m=rt.forward_processed(a,torch.tensor(masked),np.array([pos],np.int64),context_only=True)
                return np.concatenate([keys[l] for l in p.LAYERS])
            def generate(keys):
                values=[];weights=[];indices=[]
                for j,l in enumerate(p.LAYERS):
                    v,w,ci=synthesizer.reference(l,keys[j:j+1],exclude_image=0,return_details=True)
                    values.append(v);weights.append(w);indices.append(ci)
                return np.concatenate(indices),np.concatenate(weights),np.concatenate(values)
            rows=isolation.check_interventions(value,pos,forward,generate)
        self.assertEqual(len(rows),2)
        for r in rows:
            self.assertEqual(r['range'],[80,160,160,176]);self.assertEqual(r['ignored_trailing_pixels'],[1,3]);self.assertTrue(r['reference_equal'])
        self.assertEqual(len(host.backbone.dino.calls),3)
        self.assertFalse(torch.cuda.is_initialized())


class MaskAndIsolation(unittest.TestCase):
    def test_four_phases_cover_each_patch_once_and_pixels_match(self):
        grid=(12,13);masks=np.stack([c.patch_mask(grid,i) for i in range(4)])
        np.testing.assert_array_equal(masks.sum(0),1)
        x=np.ones((1,3,192,208),np.float32)
        for phase in range(4):
            y=c.masked_input(x,phase)
            expected=np.repeat(np.repeat(masks[phase],16,0),16,1)
            np.testing.assert_array_equal(y[0,0]==0,expected)
            self.assertTrue((y[...,~expected]==1).all())
        self.assertEqual(c.phases(grid)[4,4],0);self.assertEqual(c.phases(grid)[5,4],2)
        self.assertEqual(c.phases(grid)[5,5],3)

    def test_numpy_and_actual_torch_CPU_mask_agree(self):
        import torch
        x=np.random.default_rng(1).normal(size=(1,3,192,208)).astype(np.float32)
        for phase in range(4):np.testing.assert_array_equal(c.masked_input(x,phase),c.masked_input(torch.from_numpy(x),phase).numpy())
        self.assertFalse(torch.cuda.is_initialized())

    def test_intervention_real_information_path_and_outer_information_survives(self):
        d=small_dictionary();head=c.initial_head(7);x=np.random.default_rng(3).normal(size=(1,3,192,208)).astype(np.float32)
        def forward(v):
            # Deliberately global descriptor: it would leak hidden input without pre-masking.
            raw=np.full((1,1280),float(v.mean()),np.float32);raw[0,0]=1
            return c.describe(raw,c.projection(7))
        def synth(q):
            ids=c.top_candidates(q,d,exclude_image=0);value,w=c.synthesize(q,d,ids,head);return ids,w,value
        for pos in (0,4*13+4,7*13+7,12*13-1):
            rows=isolation.check_interventions(x,pos,forward,synth)
            self.assertEqual(len(rows),2);self.assertTrue(all(r['reference_equal'] for r in rows))
        # The detector can still respond to a different original q.
        ref=np.zeros((1,1280),np.float32);u=np.ones(1,np.float32)
        a=c.detection(np.ones_like(ref),ref,u)[0];b=c.detection(np.ones_like(ref)*2,ref,u)[0]
        self.assertGreater(b[0],a[0])

    def test_isolation_rejects_an_input_bypass(self):
        x=np.ones((1,3,160,160),np.float32);counter=[0]
        def broken(masked):counter[0]+=1;return np.full((1,128),counter[0],np.float32)
        def synth(q):return np.zeros((1,1),np.int64),np.ones((1,1),np.float32),q
        with self.assertRaisesRegex(ValueError,'context depends'):isolation.check_interventions(x,0,broken,synth)

    def test_production_mask_precedes_original_host_crop_and_four_layers_share_forward(self):
        import torch
        a=SimpleNamespace(device='cpu');calls=[];grid=(12,13)
        fake_host=SimpleNamespace(patch_exec=SimpleNamespace(axis_patch_split=lambda n:([0,1],None,None)))
        def forward(a,value,selected,context_only=False):
            calls.append(value.numpy().copy());phase=len(calls)-1
            expected=np.repeat(np.repeat(c.patch_mask(grid,phase),16,0),16,1)
            self.assertTrue(np.all(value.numpy()[...,expected]==0))
            self.assertTrue(context_only)
            out={l:np.full((len(selected),128),float(l)+phase,np.float32) for l in p.LAYERS}
            return out,dict(grid=list(grid),positions=np.asarray(selected))
        x=torch.ones(1,3,192,208)
        with patch.object(rt,'model',return_value=fake_host),patch.object(rt,'forward_processed',side_effect=forward):
            keys,m=rt.masked_features(a,x)
        self.assertEqual(len(calls),4);self.assertEqual(m['host_internal_crop_forwards'],16)
        for l in p.LAYERS:self.assertEqual(keys[l].shape,(156,128))
        self.assertFalse(torch.cuda.is_initialized())


class CandidateAndMath(unittest.TestCase):
    def test_Source_selection_calibration_exclusion_and_traversal_independence(self):
        rows=[dict(relative_path=f'train/good/background/{i:04}.png',source_index=i,calibration=i%8==0) for i in range(521)]
        selected=c.select_source(rows);other=c.select_source(list(reversed(rows)))
        self.assertEqual(selected,other);self.assertEqual(len(selected),64);self.assertFalse(any(r['calibration'] for r in selected))
        a=c.select_positions(np.arange(2000),512,'image','dictionary')
        b=c.select_positions(np.arange(2000)[::-1],512,'image','dictionary');np.testing.assert_array_equal(a,b)

    def test_whole_image_exclusion_and_K7_and_K8(self):
        d=small_dictionary();q=d['keys'][:3]
        for excluded,count in ((0,12),(2,11),(-1,12)):
            ids=c.top_candidates(q,d,exclude_image=excluded)
            self.assertTrue(np.all((ids>=0).sum(1)==count))
            valid=ids>=0;self.assertFalse((d['images'][ids[valid]]==excluded).any())
        ids=c.top_candidates(q,d,exclude_image=2)
        self.assertTrue((ids[:,4]==-1).all())

    def test_context_ties_choose_physical_dictionary_order(self):
        d=small_dictionary();d['keys'][:]=0;q=np.zeros((2,128),np.float32)
        ids=c.top_candidates(q,d,exclude_image=0)
        np.testing.assert_array_equal(ids[0,:4],[6,7,8,9])

    def test_CPU_Torch_context_search_matches_and_chunking(self):
        import torch
        d=small_dictionary();q=d['keys'][::7];excluded=np.array([2]*len(q))
        cpu=c.top_candidates(q,d,excluded,chunk=2);gpu_adapter=rt.ContextIndex(d,'cpu')
        np.testing.assert_array_equal(cpu,gpu_adapter.select(q,excluded,chunk=3))
        self.assertFalse(torch.cuda.is_initialized())

    def test_convex_combination_masks_and_heads_below_100k(self):
        d=small_dictionary();q=d['keys'][:5];ids=c.top_candidates(q,d,exclude_image=2)
        v,w=c.synthesize(q,d,ids,c.initial_head(7))
        np.testing.assert_allclose(w.sum(1),1,atol=1e-7);self.assertTrue((w>=0).all());self.assertTrue((w[ids<0]==0).all())
        chosen=d['values'][np.maximum(ids,0)];lo=np.where((ids>=0)[:,:,None],chosen,np.inf).min(1);hi=np.where((ids>=0)[:,:,None],chosen,-np.inf).max(1)
        self.assertTrue((v>=lo-1e-6).all() and (v<=hi+1e-6).all())
        heads=training.make_heads('cpu');self.assertEqual(sum(x.numel() for x in heads.parameters()),32776)
        self.assertLess(p.job_plan()['parameters'],100000)

    def test_torch_head_matches_numpy_and_uses_target_only_in_loss(self):
        import torch
        d=small_dictionary();q=d['keys'][:4];ids=c.top_candidates(q,d,exclude_image=2);valid=ids>=0;ix=np.maximum(ids,0)
        expected,ww=c.synthesize(q,d,ids,c.initial_head(7));net=training.make_heads('cpu')['7']
        result,w=net(torch.tensor(q),torch.tensor(d['keys'][ix]),torch.tensor(d['types'][ix]),torch.tensor(valid),torch.tensor(d['values'][ix]))
        np.testing.assert_allclose(w.detach().numpy(),ww,atol=1e-6,rtol=1e-6)
        np.testing.assert_allclose(result.detach().numpy(),expected,atol=1e-6,rtol=1e-6)
        optimizer=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4)
        target=torch.ones_like(result);loss=((result-target)**2).mean();loss.backward();optimizer.step()
        self.assertTrue(all(torch.isfinite(x).all() for x in net.parameters()));self.assertFalse(torch.cuda.is_initialized())
        sig=inspect.signature(net.forward);self.assertNotIn('query',sig.parameters);self.assertNotIn('target',sig.parameters)

    def test_distance_units_and_exact_min_fallback(self):
        q=np.zeros((3,1280),np.float32);v=np.zeros_like(q);v[0,0]=1280;v[1,0]=2560
        d,dp=c.detection(q,v,np.array([2,.5,0],np.float32))
        np.testing.assert_array_equal(dp,[1,2,0]);np.testing.assert_array_equal(d,[1,.5,0])

    def test_image_equal_batches_are_seeded_by_step_and_shared_physical_rows(self):
        im=np.repeat(np.arange(72),[2+i%3 for i in range(72)])
        a=training.batch_rows(im,12);b=training.batch_rows(im,12)
        np.testing.assert_array_equal(a,b);self.assertEqual(len(a),128)
        self.assertTrue((im[a[:64]]<64).all());self.assertTrue((im[a[64:]]>=64).all())

    def test_CPU_optimizer_checkpoint_resumes_same_next_update(self):
        import torch
        import io
        d=small_dictionary();q=d['keys'][:4];ci=c.top_candidates(q,d,exclude_image=2);valid=ci>=0;ix=np.maximum(ci,0)
        args=[torch.tensor(q),torch.tensor(d['keys'][ix]),torch.tensor(d['types'][ix]),torch.tensor(valid),torch.tensor(d['values'][ix])]
        net=training.make_heads('cpu')['7'];opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4)
        def update(net,opt):
            opt.zero_grad(set_to_none=True);v,_=net(*args);loss=((v-1)**2).mean();loss.backward();opt.step()
        update(net,opt);buffer=io.BytesIO();torch.save(dict(heads=net.state_dict(),optimizer=opt.state_dict(),step=100,seconds=1.),buffer)
        update(net,opt);expected={k:v.detach().clone() for k,v in net.state_dict().items()}
        restored=training.make_heads('cpu')['7'];optimizer=torch.optim.AdamW(restored.parameters(),lr=1e-3,weight_decay=1e-4)
        buffer.seek(0);saved=torch.load(buffer,map_location='cpu',weights_only=True);restored.load_state_dict(saved['heads']);optimizer.load_state_dict(saved['optimizer']);update(restored,optimizer)
        for k,v in restored.state_dict().items():torch.testing.assert_close(v,expected[k],atol=0,rtol=0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_reference_API_cannot_receive_original_queries_or_GT(self):
        for fn in (c.top_candidates,c.weights,c.synthesize,rt.ContextIndex.select):
            self.assertFalse({'label','GT','union','raw_query','original_q','cls','confidence'}&set(inspect.signature(fn).parameters))
        text=inspect.getsource(training.prepare)
        self.assertNotIn('.gt(',text);self.assertNotIn('.gt(',inspect.getsource(rt.prepare_context))


class IntegrationAndBoundaries(unittest.TestCase):
    def test_cached_context_rejects_changed_image_identity(self):
        with tempfile.TemporaryDirectory(dir=ip.ROOT/'.tmp') as tmp:
            a=SimpleNamespace(cache_root=Path(tmp));row=dict(relative_path='train/good/background/a.png',image_sha256='original')
            path=ip.context_path(a,row);st.put(path,dict(c7=np.zeros((1,128),np.float32)),dict(identity='run',image=row))
            changed=dict(row,image_sha256='changed')
            self.assertEqual(path,ip.context_path(a,changed))
            with patch.object(ip,'identity',return_value='run'):
                with self.assertRaisesRegex(ValueError,'context image'):rt.context_bundle(a,changed,None,'Source')

    def test_missing_MAD_enters_specific_missing_input_list_without_GPU(self):
        a=SimpleNamespace(parent_run=Path('absent_parent'))
        with patch.object(ip,'parent',side_effect=FileNotFoundError(2,'MAD missing','absent_parent/input_distance_mad.json')):
            r=ip.preflight(a)
        self.assertEqual(r['status'],'blocked');self.assertEqual(r['missing'][0]['asset'],'required parent metadata/MAD')
        self.assertTrue(r['missing'][0]['path'].endswith('input_distance_mad.json'));self.assertFalse(r['GPU_initialized'])

    def test_new_prediction_uses_original_map_score_and_instance_kernels(self):
        from PIL import Image
        from types import FunctionType
        from DINOv3.MADEqual.aebad_s_blind_context_p0 import prediction as pred,evaluation as ev
        from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.analysis import spatial
        from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.calibration import response
        state={k:dict(medians=[0]*4,scales=[1]*4) for k in ('peak','local')};mad=dict(image=state,patch=dict(medians=[0]*4,scales=[1]*4))
        with tempfile.TemporaryDirectory(dir=ip.ROOT/'.tmp') as tmp:
            root=Path(tmp);a=SimpleNamespace(run_root=root/'run',cache_root=root/'cache',data=root/'data')
            f=a.data/'ground_truth/ablation/background/test.png';f.parent.mkdir(parents=True);(a.data/'test').mkdir()
            mask=np.zeros((64,80),np.uint8);mask[24:36,24:40]=255;Image.fromarray(mask).save(f)
            sample=dict(domain='background',sample_id='test_fixture',relative_path='test/ablation/background/test.png',label=1,image_height=64,image_width=80,gt_sha256=st.sha(f))
            rng=np.random.default_rng(42);pos=np.arange(20,dtype=np.int64)[::-1];u=rng.uniform(1,3,(4,20)).astype(np.float32);seg=u-np.float32(.1);new=u-np.float32(.03)
            maps={};scores={};meta=dict(grid=[4,5],output_shape=[16,20])
            for b,z in ((p.U,u),(p.S,seg),(p.MAIN,new)):maps[b],scores[b]=response(z,pos,meta['grid'],meta['output_shape'],state)
            small=ev.original.shared.resized_mask(mask>0,[16,20]).astype(np.uint8)
            parent_maps={(sample['sample_id'],b):dict(map_float32_sha256=ev.original.shared.array_sha(v),mask_uint8_sha256=ev.original.shared.array_sha(small),output_height=16,output_width=20) for b,v in maps.items() if b!=p.MAIN}
            parent_scores={(sample['sample_id'],b):v for b,v in scores.items() if b!=p.MAIN}
            inst=FunctionType(spatial.__code__,dict(spatial.__globals__,U=p.U,S=p.S),spatial.__name__)(sample,mask>0,{p.U:maps[p.U],p.S:maps[p.S]},(4,5))
            parent_inst={}
            for row in inst:
                ref={k:('' if v is None else str(v)) for k,v in row.items()};ref.update(small_group='True',small_group_definition='original_background_smallest_three')
                parent_inst[sample['sample_id'],str(row['instance_id']),row['branch']]=ref
            parent=SimpleNamespace(mad=mad,prediction=lambda s:({p.U:u,p.S:seg,'positions':pos},dict(metadata=meta)))
            st.put(pred.path(a,sample),{p.MAIN:new,'positions':pos},dict(identity='child',sample=sample,model_seal='model',scores=scores[p.MAIN],**meta))
            with patch.object(ip,'parent',return_value=parent),patch.object(ip,'identity',return_value='child'),patch.object(training,'trained',return_value=({},dict(seal='model'))):
                value=ev.sample_work(a,sample,parent_scores,parent_maps,parent_inst)
            self.assertTrue(all(x['verified'] for x in value[4]));self.assertEqual(len(value[3]),3)
            new_inst=[r for r in value[5] if r['branch']==p.MAIN];self.assertEqual(len(new_inst),1);self.assertTrue(new_inst[0]['small_group'])
            self.assertEqual(len(value[-1]),2);self.assertTrue(np.isfinite(value[1][p.MAIN]).all())

    def test_actual_processed_adapter_projects_on_device_without_raw_context_transfer(self):
        import torch
        a=SimpleNamespace(device='cpu');pos=np.array([0,5],np.int64);raw={l:torch.ones((2,1280))*l for l in p.LAYERS}
        pending=SimpleNamespace(grid=(2,3),positions=pos,queries=raw)
        host=SimpleNamespace(patch_exec=SimpleNamespace(axis_patch_split=lambda n:([0],None,None)))
        fake=SimpleNamespace(_extract_processed_batch=lambda *args,**kwargs:[pending])
        with patch.object(rt,'model',return_value=host),patch.dict(sys.modules,{'DINOv3.relation_reliability.superadd_host':fake}):
            q,m=rt.forward_processed(a,torch.zeros(1,3,32,48),pos,context_only=True)
        self.assertEqual(a._full_forwards,1);self.assertEqual(a._crop_forwards,1)
        for l in p.LAYERS:
            self.assertEqual(q[l].shape,(2,128));np.testing.assert_allclose(q[l],c.describe(raw[l].numpy(),c.projection(l)),atol=1e-7,rtol=1e-6)
        self.assertFalse(torch.cuda.is_initialized())

    def test_resident_synthesizer_CPU_matches_numpy(self):
        from DINOv3.MADEqual.aebad_s_blind_context_p0.prediction import Synthesizer
        import torch
        d=small_dictionary();heads={l:c.initial_head(l) for l in p.LAYERS};a=SimpleNamespace(device='cpu')
        with patch.object(rt,'dictionary',return_value=(d,{})):s=Synthesizer(a,heads)
        q=d['keys'][::8];ci=c.top_candidates(q,d);expected,_=c.synthesize(q,d,ci,heads[7])
        result=s.reference(7,q,chunk=3);np.testing.assert_allclose(result,expected,atol=1e-6,rtol=1e-6)
        self.assertFalse(torch.cuda.is_initialized())

    def test_original_scoring_and_exact_pixel_kernel(self):
        from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.calibration import z_values,response
        from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import statistics,ch1_score
        from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.evaluation import exact_pixel_result
        raw=np.random.default_rng(4).uniform(.1,2,(4,20)).astype(np.float32);pos=np.arange(20,dtype=np.int64)[::-1]
        patchstate=dict(medians=[0]*4,scales=[1]*4);state={k:dict(medians=[0]*4,scales=[1]*4) for k in ('peak','local')}
        z=z_values(raw,patchstate);m,sc=response(z,pos,[4,5],[8,10],state)
        current,peak,local=statistics(raw,pos,(4,5),(8,10),np.zeros(4),np.ones(4))
        self.assertAlmostEqual(sc['CH1'],ch1_score(peak,local,state),places=5);self.assertEqual(sc['CURRENT'],current)
        with tempfile.TemporaryDirectory(dir=ip.ROOT/'.tmp') as tmp:
            path=Path(tmp)/'pixels.json';label=np.array([0,1,0,1],np.uint8);score=np.array([.1,.8,.2,.9],np.float32);counts=dict(pixels=4,positive_pixels=2,negative_pixels=2)
            value=exact_pixel_result(path,'id','map',label,score,domain='background',branch=p.MAIN,prediction_sha='frozen',counts=counts)
            self.assertEqual(value['AUROC'],1)
            damaged=st.read_json(path);damaged['result']['AUROC']=1.5;st.write_json(path,damaged)
            with self.assertRaises(ValueError):exact_pixel_result(path,'id','map',label,score,domain='background',branch=p.MAIN,prediction_sha='frozen',counts=counts)

    def test_default_preflight_imports_no_torch_and_initializes_no_cuda(self):
        code="import sys; from DINOv3.MADEqual.aebad_s_blind_context_p0 import runner; assert 'torch' not in sys.modules; assert runner.parser().parse_args(['--parent-run','x','--parent-cache','y']).stage=='preflight'; print('NO_TORCH')"
        r=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True,cwd=ip.ROOT)
        self.assertEqual(r.returncode,0,r.stderr);self.assertIn('NO_TORCH',r.stdout)

    def test_GT_evaluation_is_blocked_until_all_predictions_freeze(self):
        from DINOv3.MADEqual.aebad_s_blind_context_p0 import evaluation as ev
        with tempfile.TemporaryDirectory(dir=ip.ROOT/'.tmp') as tmp:
            a=SimpleNamespace(run_root=Path(tmp))
            with patch.object(training,'require_isolation',return_value={}),patch.object(ip.GTData,'gt',side_effect=AssertionError('GT opened')) as gt:
                with self.assertRaises(FileNotFoundError):ev.evaluate(a)
                gt.assert_not_called()

    def test_scratch_cannot_overlap_parent_and_pack_excludes_big_files(self):
        with tempfile.TemporaryDirectory(dir=ip.ROOT/'.tmp') as tmp:
            r=Path(tmp);a=runner.normalize(runner.parser().parse_args(['--parent-run',str(r/'parent'),'--parent-cache',str(r/'oldcache'),'--run-root',str(r/'new'),'--cache-root',str(r/'cache'),'--evaluation-scratch',str(r/'scratch')]))
            a.evaluation_scratch=a.parent_run/'scratch'
            with self.assertRaises(ValueError):runner.isolation(a)
            a.evaluation_scratch=r/'scratch';a.run_root.mkdir();(a.run_root/'stray.bin').write_bytes(b'not a report');(a.run_root/'training_checkpoint.pt').write_bytes(b'large');st.write_json(a.run_root/'status.json',dict(pending=True))
            info=runner.pack(a)
            import tarfile
            with tarfile.open(info['archive']) as t:self.assertEqual(len(t.getmembers()),1)

    def test_resource_plan_fixed_and_no_auto_other_domains_or_search(self):
        plan=p.job_plan();self.assertEqual(plan['updates'],1500);self.assertEqual(plan['dictionary_images'],72)
        self.assertEqual(plan['maximum_query_masked_extractions'],1188);self.assertEqual(plan['new_AUPRO_jobs'],1)
        self.assertEqual(p.CONFIG['population'],[297,85,212]);self.assertNotIn('illumination',p.CONFIG.values())
        self.assertNotIn('shutdown',inspect.getsource(runner.all_stages))

    def test_margin_summary_keeps_NA_and_small_group(self):
        rows=[dict(branch=b,small_group=True,**{'margin_change_vs_'+p.U:x,'fixed_union_margin_change_vs_'+p.U:x}) for b in (p.S,p.MAIN) for x in (None,-.01,.02)]
        margins,mechanism=reporting.summaries(rows,[])
        for r in margins:self.assertEqual((r['computable'],r['NA'],r['damaged'],r['improved']),(2,1,1,1))
        self.assertIsNone(mechanism['G'])


class PredictionResume(unittest.TestCase):
    """Actual predict/load/verified with 297 tiny CPU cached-array fixtures.

    Only the expensive extractor and synthesizer are stand-ins. No real data,
    backbone, CUDA or GPU evaluator is called by these continuation tests.
    """
    def setUp(self):
        from DINOv3.MADEqual.aebad_s_blind_context_p0 import prediction as pred
        self.pred=pred;self.stack=ExitStack();self.addCleanup(self.stack.close)
        tmp=self.stack.enter_context(tempfile.TemporaryDirectory(dir=ip.ROOT/'.tmp'));root=Path(tmp)
        self.a=SimpleNamespace(run_root=root/'run',cache_root=root/'newcache',data=root/'absent_data',device='cpu',stage='predict')
        pos=np.arange(9,dtype=np.int64);raw=np.ones((4,9),np.float32)
        image_state={k:dict(medians=[0]*4,scales=[1]*4) for k in ('peak','local')}
        samples=[dict(domain='background',sample_id=f'test_{i:04d}',relative_path=f'test/good/background/{i}.png',label=int(i>=85),image_height=12,image_width=12) for i in range(297)]
        parent=SimpleNamespace(cache=root/'oldcache',samples=samples,mad=dict(patch=dict(medians=[0]*4,scales=[1]*4),image=image_state))
        for s in samples:
            st.put(parent.cache/'predictions/background'/s['sample_id'],dict(raw_union=raw,UNION_K8=raw,positions=pos),dict(grid=[3,3]))
        parent.prediction=lambda s:st.get(parent.cache/'predictions/background'/s['sample_id'])
        self.parent=parent
        self.stack.enter_context(patch.object(ip,'identity',return_value='fixture'))
        self.stack.enter_context(patch.object(ip,'parent',return_value=parent))
        self.stack.enter_context(patch.object(training,'require_isolation',return_value={}))
        self.stack.enter_context(patch.object(training,'trained',return_value=({},dict(seal='fixed_model'))))
        from DINOv3.MADEqual.aebad_s_support_witness_segment_p0 import fresh_inputs
        from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation import runtime as oldrt
        class FakeData:
            def __init__(self,*args):pass
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def image(self,*args):return None
        self.data=self.stack.enter_context(patch.object(fresh_inputs,'Data',side_effect=FakeData))
        self.precision=self.stack.enter_context(patch.object(oldrt,'reliable_precision'))
        self.processed=self.stack.enter_context(patch.object(rt,'processed',return_value=None))
        meta=dict(grid=[3,3],positions=pos,masked_full_extractions=4,host_internal_crop_forwards=4)
        self.masked=self.stack.enter_context(patch.object(rt,'masked_features',return_value=({l:np.zeros((9,128),np.float32) for l in p.LAYERS},meta)))
        def original(a,*args):
            a._original_calls=getattr(a,'_original_calls',0)+1
            return {**{f'q{l}':np.ones((9,1280),np.float32) for l in p.LAYERS},'positions':pos},dict(metadata=dict(grid=[3,3])),'fixture'
        self.original=self.stack.enter_context(patch.object(rt,'original',side_effect=original))
        fake_synth=SimpleNamespace(reference=lambda *args:np.zeros((9,1280),np.float32))
        self.synth=self.stack.enter_context(patch.object(pred,'Synthesizer',return_value=fake_synth))
        self.stack.enter_context(patch('builtins.print'))

    def assert_no_recomputation(self):
        for call in (self.data,self.precision,self.processed,self.masked,self.original,self.synth):call.assert_not_called()

    def reset_calls(self):
        for call in (self.data,self.precision,self.processed,self.masked,self.original,self.synth):call.reset_mock()

    def test_completed_predict_then_runner_dispatch_reuses_immutable_freeze(self):
        first=self.pred.predict(self.a);f=self.a.run_root/'predictions_frozen.json'
        before=f.read_bytes();stamp=f.stat().st_mtime_ns
        payloads={s['sample_id']:(self.pred.path(self.a,s)/(p.MAIN+'.npy')).read_bytes() for s in self.parent.samples}
        self.assertEqual((first['new_images'],first['masked_full_extractions'],first['new_original_extractions']),(297,1188,297))
        self.assertNotIn('masked_full_extractions',st.sealed_json(f))
        self.reset_calls()
        # This dispatch is also what all's predict child invokes after an earlier
        # downstream evaluation interruption; it must not construct a GPU host.
        with patch.object(ip,'validate',return_value={}),patch.object(runner,'isolation'):
            second=runner.execute(self.a)
        self.assertEqual((second['new_images'],second['reused_images'],second['masked_full_extractions'],second['new_original_extractions']),(0,297,0,0))
        self.assertEqual(second['prediction_seal'],first['prediction_seal']);self.assertEqual(f.read_bytes(),before);self.assertEqual(f.stat().st_mtime_ns,stamp)
        for s in self.parent.samples:self.assertEqual((self.pred.path(self.a,s)/(p.MAIN+'.npy')).read_bytes(),payloads[s['sample_id']])
        self.assert_no_recomputation()

    def test_partial_resume_rebuilds_only_missing_image_then_freezes_stably(self):
        first=self.pred.predict(self.a);f=self.a.run_root/'predictions_frozen.json';before=f.read_bytes()
        # Model an interruption before the global commit and one image commit.
        f.unlink();(self.pred.path(self.a,self.parent.samples[-1])/'manifest.json').unlink();self.reset_calls()
        second=self.pred.predict(self.a)
        self.assertEqual((second['new_images'],second['reused_images'],second['masked_full_extractions'],second['new_original_extractions']),(1,296,4,1))
        self.assertEqual(f.read_bytes(),before);self.assertEqual(second['prediction_seal'],first['prediction_seal'])
        self.masked.assert_called_once();self.original.assert_called_once();self.reset_calls()
        self.pred.predict(self.a);self.assert_no_recomputation()

    def test_legacy_freeze_counters_preserved_and_bad_record_still_rejected(self):
        self.pred.predict(self.a);f=self.a.run_root/'predictions_frozen.json'
        legacy=st.sealed_json(f);legacy.pop('seal');legacy.update(masked_full_extractions=1188,new_original_extractions=297);legacy['seal']=st.digest(legacy);st.write_json(f,legacy)
        before=f.read_bytes();self.reset_calls();result=self.pred.predict(self.a)
        self.assertEqual(result['prediction_seal'],legacy['seal']);self.assertEqual(f.read_bytes(),before);self.assertEqual(result['masked_full_extractions'],0)
        self.assert_no_recomputation()
        broken=copy.deepcopy(legacy);broken.pop('seal');broken['records'][0]['seal']='different';broken['seal']=st.digest(broken);st.write_json(f,broken)
        with self.assertRaisesRegex(ValueError,'frozen prediction modified'):self.pred.predict(self.a)
        self.assert_no_recomputation()


if __name__=='__main__':unittest.main()
