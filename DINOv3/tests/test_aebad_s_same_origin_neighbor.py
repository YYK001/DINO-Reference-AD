"""CPU-only formula/provenance/continuation tests. No model, CUDA or downloads."""
import csv,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from DINOv3.MADEqual.aebad_s_same_origin_neighbor_p0 import numerics as n,provenance as p,inputs as ip,pipeline as pp
from DINOv3.MADEqual.aebad_s_same_origin_neighbor_p0.protocol import A,B,C,NEW,CONFIG,LAYERS,job_plan
from DINOv3.MADEqual.aebad_s_same_origin_neighbor_p0.runner import parser,normalize,isolation,pack,enforce_time_budget,BudgetPause


def state():return {k:dict(median=0.,scale=1.) for k in ('abs','nb','rel')}


class FormulaTests(unittest.TestCase):
    def test_common_offset_relative_zero_but_absolute_nonzero(self):
        m=np.arange(8,dtype=np.float32).reshape(1,1,8);dm=np.ones((1,1,4,8),np.float32)
        q=m[:,0]+2;dq=dm[:,0]
        absolute,nb,rel=n.residuals(q,dq,m,dm)
        self.assertGreater(float(absolute[0,0]),0);self.assertGreater(float(nb[0,0]),0)
        np.testing.assert_array_equal(rel,np.zeros((1,1),np.float32))

    def test_delta_and_direct_neighbor_formulas(self):
        rng=np.random.default_rng(42);q=rng.normal(size=(3,1280)).astype(np.float32)
        qnb=rng.normal(size=(3,4,1280)).astype(np.float32);m=rng.normal(size=(3,2,1280)).astype(np.float32)
        mnb=rng.normal(size=(3,2,4,1280)).astype(np.float32)
        ab,nb,rel=n.residuals(q,qnb-q[:,None],m,mnb-m[:,:,None])
        nb0=np.mean(np.linalg.norm(qnb[:,None]-mnb,axis=-1)/1280,axis=-1,dtype=np.float32)
        rel0=np.mean(np.linalg.norm((qnb-q[:,None])[:,None]-(mnb-m[:,:,None]),axis=-1)/1280,axis=-1,dtype=np.float32)
        np.testing.assert_allclose(nb,nb0,atol=1e-7,rtol=1e-6)
        np.testing.assert_array_equal(rel,rel0)

    def test_C1_degeneracy_and_C16_not_above_C1(self):
        rng=np.random.default_rng(5);ab=rng.random((31,16),dtype=np.float32);nb=rng.random((31,16),dtype=np.float32);rel=rng.random((31,16),dtype=np.float32)
        valid=rng.random((31,16))>.5;valid[:,0]=True
        full,_=n.joint(ab,nb,rel,valid,np.ones(31,bool),np.zeros(31,np.float32),state())
        one,_=n.joint(ab[:,:1],nb[:,:1],rel[:,:1],valid[:,:1],np.ones(31,bool),np.zeros(31,np.float32),state())
        np.testing.assert_array_equal(one[B],one[C]);self.assertTrue(np.all(full[C]<=full[B]))

    def test_outer_and_invalid_top1_force_all_to_original_union(self):
        ab=np.ones((3,16),np.float32);valid=np.ones((3,16),bool);valid[1,0]=False
        uz=np.array([-4,-3,-2],np.float32)
        values,audit=n.joint(ab,ab,ab,valid,np.array([False,True,True]),uz,state())
        for b in NEW:np.testing.assert_array_equal(values[b][:2],uz[:2])
        self.assertEqual(audit['fallback'],2)

    def test_invalid_C16_candidates_excluded_without_refill(self):
        ab=np.array([[2.,0.,3.]],np.float32);valid=np.array([[True,False,True]])
        v,audit=n.joint(ab,ab,ab,valid,np.array([True]),np.array([-99],np.float32),state())
        self.assertEqual(v[C][0],2);self.assertEqual(audit['excluded_top16_candidates'],1)

    def test_z_is_not_clipped_or_recalibrated(self):
        s=state();s['abs']=dict(median=5,scale=2);s['rel']=dict(median=7,scale=3)
        x=np.array([[1.]],np.float32);v,_=n.joint(x,x,x,np.ones((1,1),bool),np.ones(1,bool),np.zeros(1,np.float32),s)
        self.assertEqual(v[B][0],-2.)

    def test_physical_permutation_and_crop_seam_not_boundary(self):
        full=np.arange(7*9*3,dtype=np.float32).reshape(7,9,3);pos=np.arange(63,dtype=np.int64)[::-1]
        np.testing.assert_array_equal(n.physical(full.reshape(63,3)[pos],pos,(7,9)),full)
        d,valid=n.deltas(full,np.array([0,3*9+4,62]));self.assertEqual(valid.tolist(),[False,True,False])
        np.testing.assert_array_equal(d[1],full[[2,4,3,3],[4,4,3,5]]-full[3,4])

    def test_blocked_top16_matches_direct_and_is_reproducible(self):
        rng=np.random.default_rng(3);q=rng.normal(size=(7,9)).astype(np.float32);m=rng.normal(size=(40,9)).astype(np.float32)
        direct=np.linalg.norm(q[:,None]-m[None],axis=-1);want=np.argsort(direct,axis=1,kind='stable')[:,:16]
        for qc,rc in ((1,3),(4,17),(7,40)):
            ds,ids=n.exact_topk_numpy(q,m,16,qc,rc);np.testing.assert_array_equal(ids,want)
            np.testing.assert_allclose(ds,np.take_along_axis(direct,want,1)/9,atol=1e-7,rtol=1e-6)
            np.testing.assert_array_equal(ids[:,0],np.argmin(direct,axis=1))

    def test_tie_rule_across_reference_blocks(self):
        q=np.zeros((2,4),np.float32);m=np.ones((35,4),np.float32)
        _,ids=n.exact_topk_numpy(q,m,16,1,3)
        np.testing.assert_array_equal(ids,np.tile(np.arange(16),(2,1)))

    def test_production_torch_CPU_top16_and_neighbor_components(self):
        import torch
        self.assertFalse(torch.cuda.is_initialized())
        rng=np.random.default_rng(3);q=rng.normal(size=(5,16)).astype(np.float32);m=rng.normal(size=(37,16)).astype(np.float32)
        dm=rng.normal(size=(37,4,16)).astype(np.float32);dq=rng.normal(size=(5,4,16)).astype(np.float32)
        search=n.Search(m,dm,np.ones(37,bool),[0,17,37],'cpu',query_chunk=3,reference_chunk=7)
        d,ids,part,pi=search.top(q);ref,ri=n.exact_topk_numpy(q,m)
        np.testing.assert_array_equal(ids,ri);np.testing.assert_allclose(d,ref,atol=1e-7,rtol=1e-6)
        _,nb,rel=n.residuals(q,dq,m[ids],dm[ids]);nb1,rel1=search.components(q,dq,ids)
        np.testing.assert_allclose(nb1,nb,atol=1e-7,rtol=1e-6);np.testing.assert_allclose(rel1,rel,atol=1e-7,rtol=1e-6)
        ties=n.Search(np.ones((35,4),np.float32),np.zeros((35,4,4),np.float32),np.ones(35,bool),[0,35],'cpu',1,3)
        np.testing.assert_array_equal(ties.top(np.zeros((1,4),np.float32))[1],np.arange(16)[None])
        self.assertFalse(torch.cuda.is_initialized())


class OriginTests(unittest.TestCase):
    def test_recovery_layout_can_read_real_PNG_via_original_Data_adapter(self):
        """Exercise the exact missing-field boundary that failed on the server."""
        import hashlib,io,zipfile
        from PIL import Image
        from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.fresh_inputs import Data
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);relative='train/good/background/fixture.png'
            image=Image.new('RGB',(1024,1024),(17,29,41));buf=io.BytesIO();image.save(buf,format='PNG');blob=buf.getvalue()
            original=dict(relative_path=relative,image_sha256=hashlib.sha256(blob).hexdigest(),image_height=1024,image_width=1024,source_index=1,calibration=False)
            rows=p.layout(dict(source=[dict(original,source_index=0,calibration=True),original]))
            self.assertEqual(len(rows),1);self.assertEqual(rows[0]['image_sha256'],original['image_sha256'])
            self.assertEqual(rows[0]['grid'],[40,40]);self.assertEqual(rows[0]['start'],0);self.assertEqual(rows[0]['stop'],1600)
            archive=root/'AeBAD.zip'
            with zipfile.ZipFile(archive,'w') as z:z.writestr('AeBAD/AeBAD_S/'+relative,blob)
            expanded=root/'expanded';path=expanded/relative;path.parent.mkdir(parents=True);path.write_bytes(blob)
            (expanded/'test').mkdir();(expanded/'ground_truth').mkdir()
            for dataset in (archive,expanded):
                with self.subTest(dataset=dataset.name),Data(dataset) as data:
                    decoded=data.image(rows[0]);self.assertEqual(decoded.size,(1024,1024));self.assertEqual(decoded.getpixel((0,0)),(17,29,41))
            self.assertEqual(original['calibration'],False);self.assertNotIn('grid',original)

    def test_original_builder_observer_preserves_sampler_values_indices_and_RNG(self):
        """Real original builder+100-round sampler, tiny synthetic CPU features.
        No real image assets, backbone, CUDA or formal bank is used.
        """
        import torch,time
        from PIL import Image
        from external_baselines.superadd_external import memory_variants as mv
        from DINOv3.MADEqual.aebad_s_same_origin_neighbor_p0 import runtime as rt
        from DINOv3.MADEqual.aebad_s_support_witness_segment_p0 import fresh_inputs
        from DINOv3.relation_reliability.superadd_host import image_tensor
        class Aug:
            min_brightness_factor=.8;max_brightness_factor=1.2
            def __call__(self,x):return x*np.random.uniform(.8,1.2)
        class Host:
            threshold_fraction=8;layers=LAYERS;max_database_size=10000;backbone=None
            def __init__(self):self.augmented_preprocessing=Aug()
            def patch_exec(self,x,backbone):
                full=np.arange(40*40*4,dtype=np.float32).reshape(1,40,40,4)/100
                return [(full+np.float32(x.mean())+np.float32(j)).astype(np.float32) for j in range(4)]
        rows=[dict(source_index=i,relative_path=f'train/{i}.png',calibration=i%8==0) for i in range(9)]
        class Data:
            def __init__(self,*a):pass
            def __enter__(self):return self
            def __exit__(self,*a):pass
            def image(self,row):return Image.fromarray(np.full((2,2,3),row['source_index']*20,np.uint8))
        class Images:
            def __len__(self):return len(rows)
            def __getitem__(self,i):return image_tensor(Data().image(rows[i]))
        layout=[dict(relative_path=r['relative_path'],source_index=r['source_index'],grid=[40,40],start=j*1600,stop=(j+1)*1600) for j,r in enumerate(rows) if not r['calibration']]
        # j above is global image index; candidate offsets must omit held-out rows.
        for i,r in enumerate(layout):r.update(start=i*1600,stop=(i+1)*1600)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=normalize(parser().parse_args(['--run-root',str(root/'run'),'--device','cpu']))
            a._started=time.perf_counter();threads=torch.get_num_threads();torch.set_num_threads(1)
            try:
                np.random.seed(42)
                original=mv.build_official_memory_spooled(Host(),Images(),device=torch.device('cpu'),scratch_dir=root/'original_spool',return_selected_indices=True,feature_transform=lambda i,l,v:v)
                rng_after=np.random.rand(5)
                np.random.seed(42)
                with patch.object(ip,'catalog',return_value=dict(source=rows)),patch.object(rt,'model',return_value=Host()),patch.object(p,'layout',return_value=layout),patch.object(fresh_inputs,'Data',Data):
                    rt.rebuild_source(a)
                actual,_=ip.st.get(a.cache_root/'source_neighbors')
                for l in LAYERS:
                    np.testing.assert_array_equal(actual[f'L{l}'],original.banks[l].numpy())
                    np.testing.assert_array_equal(actual[f'selected{l}'],original.selected_candidate_indices[l])
                np.testing.assert_array_equal(np.random.rand(5),rng_after)
                for array in actual.values():array._mmap.close()
                self.assertFalse(torch.cuda.is_initialized())
            finally:torch.set_num_threads(threads)

    def test_bank_rows_map_same_image_and_exact_neighbors(self):
        grids=[np.arange(25*2,dtype=np.float32).reshape(5,5,2),np.arange(25*2,dtype=np.float32).reshape(5,5,2)+100]
        rows=[dict(start=0,stop=25,grid=[5,5]),dict(start=25,stop=50,grid=[5,5])]
        full=np.concatenate([x.reshape(-1,2) for x in grids]);indices=np.array([25+12,12,25,24],np.int64)
        delta,valid,image,pos=p.gather_selected(full,indices,rows)
        self.assertEqual(image.tolist(),[1,0,1,0]);self.assertEqual(pos.tolist(),[12,12,0,24]);self.assertEqual(valid.tolist(),[True,True,False,False])
        np.testing.assert_array_equal(delta[0],grids[1][[1,3,2,2],[2,2,1,3]]-grids[1][2,2])
        self.assertTrue(p.check_centers(full[indices],full[indices])['verified'])
        changed=full[indices].copy();changed[0,0]+=np.float32(1e-4)
        self.assertFalse(p.check_centers(changed,full[indices])['verified'])

    def test_invalid_or_duplicate_selected_indices_rejected(self):
        for ix in (np.array([1,1]),np.array([-1]),np.array([25])):
            with self.assertRaises(ValueError):p.map_indices(ix,[dict(start=0,stop=25,grid=[5,5])])

    def test_recording_augmentation_does_not_change_RNG_or_calls(self):
        class Aug:
            min_brightness_factor=.8;max_brightness_factor=1.2
            def __call__(self,x):return x*np.random.uniform(.8,1.2)
        np.random.seed(42);plain=[Aug()(2.) for _ in range(455)];tail=np.random.rand(5)
        np.random.seed(42);record=p.AugmentationRecorder(Aug());observed=[record(2.) for _ in range(455)];tail1=np.random.rand(5)
        np.testing.assert_array_equal(plain,observed);np.testing.assert_array_equal(tail,tail1)
        np.testing.assert_array_equal(np.array([r['brightness_factor'] for r in record.records])*2,plain)
        self.assertNotEqual(record.records[0]['brightness_factor'],1.)

    def test_find_indices_through_readonly_source_import_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);first=root/'compact/source_original';older=root/'old/source_original'
            first.mkdir(parents=True);older.mkdir(parents=True)
            (first/'manifest.json').write_text(json.dumps(dict(metadata=dict(origin=str(older)))))
            (older/'manifest.json').write_text(json.dumps(dict(metadata=dict(origin='fresh_original_builder'))))
            target=older.parent/'source_bank.npz';np.savez(target,**{f'selected_{l}':np.arange(4) for l in LAYERS})
            a=SimpleNamespace(selected_indices=None,raw_cache=None,source_bank=first)
            self.assertEqual(ip.index_path(a),target)

    def test_original_split_and_K8_populations(self):
        root=Path(__file__).resolve().parents[2]
        frozen=json.loads((root/'DINOv3/MADEqual/aebad_s_reference_residual_segmentation/frozen_origins.json').read_text())
        c=dict(source=[dict(r,source_index=i,calibration=i%8==0) for i,r in enumerate(frozen['source'])],
            supports={'background':[dict(r,domain='background') for r in frozen['background_supports']]},
            samples=[dict(r,domain='background') for r in frozen['background_samples']])
        x=ip.validate_catalog(c);self.assertEqual(len(x['samples']),297);self.assertEqual(sum(r['calibration'] for r in x['source']),66)
        import copy
        bad=copy.deepcopy(c);bad['source'][0]['calibration']=False
        with self.assertRaisesRegex(ValueError,'index%8'):ip.validate_catalog(bad)
        bad=copy.deepcopy(c);bad['supports']['background'][0]['relative_path']=x['samples'][0]['relative_path']
        with self.assertRaisesRegex(ValueError,'support leaked'):ip.validate_catalog(bad)


class FeatureProviderTests(unittest.TestCase):
    def fixture(self,root,provider='provider',run='run',inline=True):
        from DINOv3.MADEqual.aebad_s_same_origin_neighbor_p0 import feature_identity as fi
        a=normalize(parser().parse_args(['--run-root',str(root/run),'--feature-root',str(root/provider)]))
        row=dict(relative_path='train/good/fixture.png',source_index=0,calibration=True,image_height=64,image_width=64,image_sha256='fixture')
        config=fi.native_extraction(a)
        path=a.feature_root/'normal_features'/ip.key(row)
        meta=dict(image=row,grid=[3,3],kind='original_complete_grid')
        if inline:meta['extraction']=config
        record=ip.st.put(path,{**{f'q{l}':np.zeros((9,1280),np.float32) for l in LAYERS},'positions':np.arange(9,dtype=np.int64)},meta)
        c=dict(source=[row],supports={'background':[]},samples=[])
        return a,row,config,path,record,c

    def change_manifest(self,path,change):
        record=ip.st.sealed_json(path/'manifest.json');record.pop('seal');change(record)
        (path/'manifest.json').unlink();return ip.st.sealed_json(path/'manifest.json',record)

    def test_same_image_shape_wrong_checkpoint_or_preprocessing_rejected(self):
        for field in ('checkpoint','preprocessing'):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as tmp:
                a,row,config,path,_,_=self.fixture(Path(tmp))
                self.assertEqual(ip.feature_ref(a,row)['path'],str(path))
                def change(r):
                    if field=='checkpoint':r['metadata']['extraction']['checkpoint_sha256']='another-checkpoint'
                    else:r['metadata']['extraction']['host_config']['resize_factor']=.5
                self.change_manifest(path,change)
                with self.assertRaisesRegex(ValueError,'extraction configuration differs'):ip.feature_ref(a,row)
                # A standalone provider cannot endorse its own changed
                # preprocessing/checkpoint as the original frozen protocol.
                bad=ip.st.sealed_json(path/'manifest.json')['metadata']['extraction']
                ip.st.sealed_json(a.feature_root/'feature_provider.json',dict(extraction=bad))
                with self.assertRaisesRegex(ValueError,'provider'):ip.feature_ref(a,row)

    def test_feature_root_change_rejected_before_old_calibration_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a,row,_,_,_,c=self.fixture(root)
            b,_,_,_,_,_=self.fixture(root,provider='other',run='run')
            with patch.object(ip,'catalog',return_value=c):saved=ip.freeze(a)
            before=(a.run_root/'inputs.json').read_bytes();ip.st.write_json(a.run_root/'calibration.json',dict(identity=saved['identity'],fixture='old completed calibration'))
            with patch.object(ip,'catalog',return_value=c),self.assertRaisesRegex(ValueError,'configuration/input paths changed'):ip.freeze(b)
            self.assertEqual(before,(a.run_root/'inputs.json').read_bytes())
            self.assertEqual(ip.st.read_json(a.run_root/'calibration.json')['fixture'],'old completed calibration')

    def test_same_root_resealed_bundle_or_provider_manifest_rejected(self):
        from DINOv3.MADEqual.aebad_s_same_origin_neighbor_p0 import feature_identity as fi
        for change_bundle in (True,False):
            with self.subTest(bundle=change_bundle),tempfile.TemporaryDirectory() as tmp:
                a,row,config,path,rec,c=self.fixture(Path(tmp))
                provider=a.feature_root/'feature_provider.json'
                export=dict(extraction=config,records={row['relative_path']:dict(path=str(path),seal=rec['seal'])})
                ip.st.sealed_json(provider,export)
                with patch.object(ip,'catalog',return_value=c):ip.freeze(a)
                if change_bundle:self.change_manifest(path,lambda r:r['metadata'].update(generation='changed'))
                else:
                    export['generation']='changed-provider-manifest';provider.unlink();ip.st.sealed_json(provider,export)
                with patch.object(ip,'catalog',return_value=c),self.assertRaisesRegex(ValueError,'provider|manifest'):ip.freeze(a)

    def test_same_config_in_new_runs_has_different_provider_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a,row,_,_,_,c=self.fixture(root)
            b,_,_,_,_,_=self.fixture(root,provider='other',run='new-run')
            with patch.object(ip,'catalog',return_value=c):left=ip.freeze(a);right=ip.freeze(b)
            self.assertNotEqual(left['identity'],right['identity'])
            self.assertEqual(left['feature_inputs']['extraction'],right['feature_inputs']['extraction'])

    def test_legacy_export_proves_features_without_inline_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,row,config,path,record,c=self.fixture(Path(tmp),inline=False)
            with self.assertRaisesRegex(ValueError,'no matching extraction/provider provenance'):ip.feature_ref(a,row)
            export=a.feature_root/'raw_assets.json'
            ip.st.sealed_json(export,dict(format='reference_residual_raw_assets_v1',extraction=config,normal_features={row['relative_path']:dict(path=str(path),seal=record['seal'])}))
            self.assertEqual(ip.feature_ref(a,row)['provider']['kind'],'export')
            with patch.object(ip,'catalog',return_value=c):saved=ip.freeze(a);self.assertEqual(ip.freeze(a),saved)

    def test_recent_parent_run_configuration_proves_legacy_feature_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a,row,config,path,_,c=self.fixture(root,inline=False)
            self.change_manifest(path,lambda r:r['metadata'].update(identity='raw-fixture'))
            rawrun=root/'recent-raw-run';a.parent_run=root/'parent';a.source_bank=root/'bank'
            record=dict(extraction=config,cache=str(a.feature_root),identity='raw-fixture')
            ip.st.sealed_json(rawrun/'inputs.json',record)
            ip.st.sealed_json(a.parent_run/'assets.json',dict(references=dict(extraction=config,parent_run=str(rawrun)),normal_features={},queries={}))
            self.assertEqual(ip.feature_ref(a,row)['provider']['kind'],'run')
            with patch.object(ip,'catalog',return_value=c):ip.freeze(a)
            record['extraction']=json.loads(json.dumps(config));record['extraction']['checkpoint_sha256']='changed'
            (rawrun/'inputs.json').unlink();ip.st.sealed_json(rawrun/'inputs.json',record)
            with patch.object(ip,'catalog',return_value=c),self.assertRaisesRegex(ValueError,'extraction configuration differs'):ip.freeze(a)

    def test_legacy_unbound_run_not_rewritten_and_own_cache_contract_checked(self):
        from DINOv3.MADEqual.aebad_s_same_origin_neighbor_p0 import runtime as rt
        with tempfile.TemporaryDirectory() as tmp:
            a,row,_,_,_,c=self.fixture(Path(tmp))
            ip.st.sealed_json(a.run_root/'inputs.json',dict(identity='legacy'))
            before=(a.run_root/'inputs.json').read_bytes()
            with patch.object(ip,'catalog',return_value=c),self.assertRaisesRegex(ValueError,'predates'):ip.freeze(a)
            self.assertEqual(before,(a.run_root/'inputs.json').read_bytes())
            (a.run_root/'inputs.json').unlink()
            with patch.object(ip,'catalog',return_value=c):saved=ip.freeze(a)
            local=a.cache_root/'features'/ip.key(row)
            ip.st.put(local,{**{f'q{l}':np.zeros((9,1280),np.float32) for l in LAYERS},'positions':np.arange(9,dtype=np.int64)},
                dict(image=row,grid=[3,3],identity=saved['identity'],extraction_identity='wrong-extractor',origin_binding=None))
            with self.assertRaisesRegex(ValueError,'cached feature extraction/provider differs'):rt.features(a,row)


class CalibrationAndExecutionTests(unittest.TestCase):
    def test_stdout_EOF_before_process_exit_does_not_spin_heartbeat(self):
        import contextlib,io,threading
        from DINOv3.MADEqual.aebad_s_same_origin_neighbor_p0.runner import wait_for_child
        # Closing stdout while the process cleans up reproduced the supervisor
        # failure mode. Use an actual CPU child, no model/CUDA.
        child=subprocess.Popen([sys.executable,'-c','import os,time; os.close(1); os.close(2); time.sleep(.12); raise SystemExit(1)'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        reader=threading.Thread(target=child.stdout.read);reader.start();reader.join()
        output=io.StringIO()
        with contextlib.redirect_stdout(output):wait_for_child(child,reader,'fixture',heartbeat_seconds=.05)
        child.stdout.close()
        self.assertEqual(child.returncode,1)
        self.assertLessEqual(output.getvalue().count('[heartbeat]'),3)

    def test_postprocessing_does_not_require_source_payload_or_old_indices(self):
        from DINOv3.MADEqual.aebad_s_same_origin_neighbor_p0 import runtime as rt
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=normalize(parser().parse_args(['aupro','--run-root',str(root/'run'),'--source-bank',str(root/'old-bank')]))
            with patch.object(ip,'catalog',return_value=dict(source=[],supports={'background':[]},samples=[])),patch.object(ip,'index_path',return_value=root/'selected.npz'):
                saved=ip.freeze(a)
            with patch.object(ip,'catalog',return_value=dict(source=[],supports={'background':[]},samples=[])),patch.object(ip,'index_path',side_effect=AssertionError('old indices are not a postprocess input')):
                self.assertEqual(ip.freeze(a),saved)
            ip.st.sealed_json(a.run_root/'calibration.json',dict(identity=saved['identity'],Source_seal='offline-source',image={}))
            with patch.object(rt,'source_bundle',side_effect=AssertionError('AUPRO does not read Source vectors')):
                self.assertEqual(pp.states(a)['Source_seal'],'offline-source')

    def test_completed_prediction_resume_never_loads_model_or_changes_freeze(self):
        from DINOv3.MADEqual.aebad_s_same_origin_neighbor_p0 import runtime as rt
        with tempfile.TemporaryDirectory() as tmp:
            a=normalize(parser().parse_args(['--run-root',str(Path(tmp)/'run')]))
            ip.st.sealed_json(a.run_root/'inputs.json',dict(identity='fixture'))
            samples=[dict(sample_id=f'test_{i:04d}') for i in range(297)]
            records=[dict(sample_id=s['sample_id'],seal='prediction-'+s['sample_id']) for s in samples]
            f=a.run_root/'predictions_frozen.json'
            ip.st.sealed_json(f,dict(identity='fixture',images=297,records=records,GT_used=False))
            original=f.read_bytes()
            with patch.object(ip,'catalog',return_value=dict(samples=samples)),patch.object(pp,'load',side_effect=lambda a,s:({},dict(seal='prediction-'+s['sample_id']))),patch.object(rt,'references',side_effect=AssertionError('must not allocate GPU references')):
                self.assertEqual(pp.predict(a)['status'],'predictions_reused')
                self.assertEqual(f.read_bytes(),original)
                bad=dict(identity='fixture',images=297,records=records[:-1],GT_used=False);f.unlink();ip.st.sealed_json(f,bad)
                with self.assertRaises(ValueError):pp.verified(a)

    def test_two_pass_calibration_writes_distinct_image_states_and_reuses_first_pass(self):
        """Small synthetic normal grids exercise production calibration plumbing;
        this is NOT real 66-image extraction or a hardware benchmark.
        """
        from DINOv3.MADEqual.aebad_s_same_origin_neighbor_p0 import runtime as rt
        import time
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=normalize(parser().parse_args(['--run-root',str(root/'run')]))
            a._started=time.perf_counter();ip.st.sealed_json(a.run_root/'inputs.json',dict(identity='fixture'))
            rows=[dict(relative_path=f'train/{i}.png',source_index=i,calibration=True,image_height=32,image_width=32) for i in range(66)]
            c=dict(source=rows,supports={'background':[]},samples=[]);calls=[]
            def feature(args,row):
                calls.append(row['source_index']);return {l:np.zeros((16,1280),np.float32) for l in LAYERS},dict(grid=[4,4],positions=np.arange(16,dtype=np.int64),output_shape=[8,8])
            def observe(q,m,searches):
                rng=np.random.default_rng(calls[-1]);result={}
                for l in LAYERS:
                    ab=np.sort(rng.uniform(.01,.5,size=(16,16)).astype(np.float32),axis=1)
                    valid=np.array([False]*5+[True,True]+[False]*2+[True,True]+[False]*5)
                    result[l]=dict(absolute=ab,nb=rng.uniform(.02,.5,size=(16,16)).astype(np.float32),rel=rng.uniform(.01,.4,size=(16,16)).astype(np.float32),
                        indices=np.tile(np.arange(16),(16,1)),valid=valid,candidate_valid=np.ones((16,16),bool))
                return result
            source=({},dict(seal='fixture-source',metadata=dict(independent_rebuild=True)))
            with patch.object(ip,'catalog',return_value=c),patch.object(rt,'source_bundle',return_value=source),patch.object(rt,'references',return_value=({},source[1])),patch.object(rt,'features',side_effect=feature),patch.object(rt,'observations',side_effect=observe):
                pp.calibrate(a);record=ip.st.sealed_json(a.run_root/'calibration.json')
                self.assertEqual(len(calls),66);self.assertEqual(record['Source_images'],66);self.assertFalse(record['Target_used']);self.assertFalse(record['GT_used'])
                self.assertEqual(set(record['image']),{A,B,C,'UNION_K8','SEGMENT_K8'})
                self.assertNotEqual(record['image'][B],record['image'][C])
                self.assertFalse(record['reuse_baselines'])
                before=(a.run_root/'calibration.json').read_bytes();pp.calibrate(a)
                self.assertEqual(len(calls),66);self.assertEqual(before,(a.run_root/'calibration.json').read_bytes())
                # Deleting only the derived calibration repeats the second pass
                # from saved observations, never the original feature/NN path.
                (a.run_root/'calibration.json').unlink();pp.calibrate(a)
                self.assertEqual(len(calls),66);self.assertEqual(before,(a.run_root/'calibration.json').read_bytes())

    def test_component_MAD_uses_top1_and_valid_population_only(self):
        rng=np.random.default_rng(6);shape=(4,400,16)
        x={name:rng.random(shape,dtype=np.float32) for name in ('abs','nb','rel')}
        x['valid']=np.ones((4,400),bool);x['candidate_valid']=np.ones(shape,bool);x['valid'][:,:100]=False
        comp,counts=pp.fit_components([x]);self.assertEqual([r['patches'] for r in counts],[400]*4+[300]*8)
        y={k:np.array(v) for k,v in x.items()};y['nb'][:,:100]=1e8;y['rel'][:,:,1:]=1e8
        other,_=pp.fit_components([y]);self.assertEqual(comp,other)
        z={k:np.array(v) for k,v in x.items()};z['valid'][:]=False
        with self.assertRaises(ValueError):pp.fit_components([z])

    def test_profiles_job_count_no_training_no_ensemble(self):
        self.assertEqual(CONFIG['evaluation_images'],297);self.assertEqual(CONFIG['Source_rows'],100000)
        self.assertEqual(job_plan()['AUPRO_jobs'],3);self.assertEqual(job_plan(False)['AUPRO_jobs'],5)
        self.assertEqual(job_plan()['training_jobs'],0)

    def test_default_preflight_imports_no_torch(self):
        command='import sys; from DINOv3.MADEqual.aebad_s_same_origin_neighbor_p0.runner import parser; assert parser().parse_args([]).stage=="preflight"; assert "torch" not in sys.modules; print("CPU_PREFLIGHT_IMPORT_OK")'
        r=subprocess.run([sys.executable,'-B','-c',command],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr);self.assertIn('CPU_PREFLIGHT_IMPORT_OK',r.stdout)

    def test_scratch_cannot_overlap_parent_and_only_owned_files_cleaned(self):
        from DINOv3.MADEqual.aebad_s_same_origin_neighbor_p0.evaluation import scratch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=normalize(parser().parse_args(['--run-root',str(root/'run'),'--parent-run',str(root/'parent')]))
            a.evaluation_scratch.mkdir();existing=a.evaluation_scratch/'someone.bin';existing.write_bytes(b'keep')
            with scratch(a,10,(A,)) as (labels,values):labels[:]=0;values[A][:]=1
            self.assertEqual(existing.read_bytes(),b'keep');self.assertEqual(list(a.evaluation_scratch.iterdir()),[existing])
            a.evaluation_scratch=a.parent_run/'scratch'
            with self.assertRaises(ValueError):isolation(a)

    def test_pack_excludes_bin_npy_and_symlinks(self):
        import tarfile
        with tempfile.TemporaryDirectory() as tmp:
            a=normalize(parser().parse_args(['--run-root',str(Path(tmp)/'run')]))
            a.run_root.mkdir();(a.run_root/'metrics.csv').write_text('ok');(a.run_root/'pixel_scratch').mkdir()
            (a.run_root/'pixel_scratch/x.bin').write_bytes(b'huge');(a.run_root/'q.npy').write_bytes(b'array')
            with tarfile.open(pack(a)['archive']) as t:self.assertEqual(t.getnames(),['run/metrics.csv'])

    def test_time_budget_pauses_at_committed_unit_not_RAM_gate(self):
        import time
        with tempfile.TemporaryDirectory() as tmp:
            a=SimpleNamespace(run_root=Path(tmp),allow_over_12h=False,_started=time.perf_counter()-3600)
            with self.assertRaises(BudgetPause):enforce_time_budget(a,'predict',5,297)
            self.assertTrue((a.run_root/'budget_pause.json').exists())
            a.allow_over_12h=True;enforce_time_budget(a,'predict',5,297)

    def test_estimate_includes_recovery_rebuild_and_unmeasured_work(self):
        from DINOv3.MADEqual.aebad_s_same_origin_neighbor_p0.reporting import estimate
        a=estimate(dict(source_case='selected_indices_replay_pending_center_check'))
        b=estimate(dict(source_case='origin_unresolved_explicit_rebuild_needed'))
        self.assertGreater(b['total_hours_range'][1],a['total_hours_range'][1]);self.assertEqual(b['disk']['official_rebuild_candidate_temporary_GiB'],120.839)
        self.assertFalse(a['measured_current_run']);self.assertEqual(len(a['stages']),5)


if __name__=='__main__':unittest.main()
