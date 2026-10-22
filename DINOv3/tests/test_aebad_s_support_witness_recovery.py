"""Real CPU recovery/audit/transfer logic; model and launcher integration mocked.

No Torch, CUDA, real image extraction, real support banks or formal experiment.
"""
import copy
import io
import json
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from DINOv3.MADEqual.aebad_s_support_witness_segment_p0 import recovery as r,storage as st,inputs as ip,portable,autodl,runner
from DINOv3.MADEqual.aebad_s_target_local_segment.geometry import geometry_layer

ROOT=Path(__file__).resolve().parents[2]
SCRATCH=ROOT/'DINOv3/MADEqual/aebad_s_support_witness_segment_p0/local_validation'


class Recovery(unittest.TestCase):
    def setUp(self):
        SCRATCH.mkdir(parents=True,exist_ok=True);self.temp=tempfile.TemporaryDirectory(dir=SCRATCH);self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()

    def fixture(self):
        banks={};rows=[]
        for i in range(1,9):
            a=np.zeros((9,1280),np.float32);a[:,0]=np.arange(9)+i/10;a[:,1]=i/7;banks[i]=a
            rows.append(dict(domain='background',support_index=i,relative_path=f'test/good/background/{i}.png',image_sha256=f'{i:064x}',
                grid=[3,3],feature_sha256=st.digest(['missing historical feature',i])))
        q=np.zeros((9,1280),np.float64);q[:,0]=np.arange(9)+.7;q[:,1]=.4
        dist=np.stack([np.linalg.norm(q[:,None]-banks[i].astype(float),axis=2).min(axis=1)/1280 for i in banks]).astype(np.float32)
        idx=np.stack([np.linalg.norm(q[:,None]-banks[i].astype(float),axis=2).argmin(axis=1) for i in banks]).astype(np.int64)
        source=np.ones(9,np.float32);g,_=geometry_layer(dist,idx,source,list(banks.values()))
        raw=dict(target_distance=np.repeat(dist[:,None],4,axis=1),target_index=np.repeat(idx[:,None],4,axis=1),
            source_distance=np.tile(source,(4,1)),segment_distance=np.tile(g['K8_SEGMENT'],(4,1)),endpoint_length=np.tile(g['endpoint_length'],(4,1)))
        cross=SimpleNamespace(identity='original_cross',required=lambda:[],missing=lambda:[])
        b=SimpleNamespace(expected={'background':copy.deepcopy(rows)},original_expected={'background':copy.deepcopy(rows)},
            bg={'capture':{'identity':'original_background'},'identity':'original_bg_parent','manifest':{'assets':{'raw_archive':'original_tar'}}},
            parent=SimpleNamespace(cross=cross,missing=lambda:[]),samples=[dict(domain='background',sample_id=f'test_{i}') for i in (1,2)],
            protected_paths=[],frozen={'original':'untouched'},paths={'data':self.root/'data'},load=lambda sample:(copy.deepcopy(raw),{},{}))
        frozen=dict(revision=r.REVISION,original=r.original_proof(b));spec=dict(identity=st.digest(frozen),frozen=frozen)
        recovery_root=r.initialize(self.root/'recovery',spec,b)
        for row in rows:
            arrays=dict(protocol_id=np.asarray(r.PROTOCOL_ID),relative_path=np.asarray(row['relative_path']),grid=np.array([3,3],np.int32),
                **{f'layer_{l}':banks[row['support_index']] for l in r.LAYERS})
            st.seal(r.feature_path(recovery_root,'background',row['support_index']),spec['identity'],arrays,r.metadata(row),compressed=False)
        return b,recovery_root,spec,raw,banks

    def test_production_geometry_all_layers_and_provenance_attachment(self):
        b,root,spec,raw,banks=self.fixture();original=copy.deepcopy(b.original_expected)
        result=r.verify(b,root);self.assertEqual(result['status'],'recovery_verified');self.assertEqual(result['images'],2)
        r.attach(b,root);self.assertEqual(b.original_expected,original)
        for old,new in zip(original['background'],b.expected['background']):
            self.assertEqual(old['feature_sha256'],new['historical_feature_sha256']);self.assertNotEqual(old['feature_sha256'],new['feature_sha256'])
        got=ip.load_layer(b.expected,b.recovered_locations,'background',31)
        np.testing.assert_array_equal(got[4],banks[4]);self.assertIn('recovered_supports',b.frozen)

    def test_old_sha_still_rejects_recovered_file(self):
        b,root,*_=self.fixture()
        with self.assertRaisesRegex(ValueError,'historical support SHA'):ip.feature_header(r.feature_path(root,'background',1),b.original_expected['background'][0])

    def test_changed_support_after_full_verification_rejected(self):
        b,root,*_=self.fixture();r.verify(b,root);p=r.feature_path(root,'background',1);p.write_bytes(p.read_bytes()+b'change')
        with self.assertRaisesRegex(ValueError,'modified/incompatible'):r.attach(b,root)

    def test_modified_partial_verification_not_reenrolled(self):
        b,root,*_=self.fixture();r.verify(b,root)
        (root/'verified_manifest.json').unlink();(root/'verified_manifest.seal.json').unlink()
        p=root/'geometry_checks/background/L7.json';rows=st.read_json(p);rows[0]['max_abs_error']=100;st.write_json(p,rows)
        with self.assertRaisesRegex(ValueError,'unsealed/modified'):r.verify(b,root)

    def test_unsealed_recovery_manifest_rejected(self):
        b,root,*_=self.fixture();r.verify(b,root);(root/'verified_manifest.seal.json').unlink()
        with self.assertRaises(FileNotFoundError):r.attach(b,root)

    def test_endpoint_lengths_detect_drift_even_if_source_masks_segment(self):
        b,root,spec,raw,banks=self.fixture();raw['source_distance'][:]=0;raw['segment_distance'][:]=0
        altered=copy.deepcopy(banks)
        for i in altered:altered[i][:,2]=i*.1
        checks=r.check_geometry(raw,7,altered)
        self.assertTrue(checks[0]['verified']);self.assertFalse(checks[1]['verified'])

    def test_bad_recovered_geometry_cannot_get_verified_manifest(self):
        b,root,spec,raw,banks=self.fixture();raw['endpoint_length']+=1
        b.load=lambda s:(copy.deepcopy(raw),{},{})
        with self.assertRaisesRegex(ValueError,'fail original'):r.verify(b,root)
        self.assertFalse((root/'verified_manifest.json').exists())

    def test_orphan_feature_not_reextracted(self):
        b,root,spec,*_=self.fixture();r.feature_path(root,'background',1).with_suffix('.json').unlink()
        with self.assertRaisesRegex(ValueError,'incomplete artifact'):r.extract(b,root,spec,{},['background'],'cuda:0')

    def test_existing_extracted_files_resume_without_loading_host(self):
        b,root,spec,*_=self.fixture();s=r.extract(b,root,spec,{},['background'],'cuda:0')
        self.assertEqual(s['reused'],8);self.assertFalse(s['model_loaded']);self.assertEqual(s['new'],0)

    def test_recovery_preflight_is_read_only_and_not_sampling_or_extracting(self):
        b,root,spec,*_=self.fixture();b.paths['data']=self.root/'data';b.parent.missing=lambda:[]
        fake=SimpleNamespace(__enter__=lambda s:s,__exit__=lambda *a:None)
        class Data:
            def __enter__(self):return self
            def __exit__(self,*a):pass
            def bytes(self,path,digest):return b'checked original RGB'
        target=self.root/'not_created'
        with patch.object(ip,'Bundle',return_value=b),patch.object(r,'specification',return_value=(spec,{})),patch.object(r,'Data',return_value=Data()),patch.object(r,'extract',side_effect=AssertionError('no extraction')):
            code=runner.main(['recover-support','--no-write','--domains','background','--checkpoint',str(self.root/'checkpoint'),
                '--recovered-supports',str(target)])
        self.assertEqual(code,0);self.assertFalse(target.exists());self.assertNotIn('torch',__import__('sys').modules)


class TransferAndLauncher(unittest.TestCase):
    def setUp(self):
        SCRATCH.mkdir(parents=True,exist_ok=True);self.temp=tempfile.TemporaryDirectory(dir=SCRATCH);self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()

    def test_real_streaming_asset_pack_restore_and_no_overwrite(self):
        source=self.root/'source';p=source/'downloads/original.npz';p.parent.mkdir(parents=True);p.write_bytes(b'original immutable input')
        archive=self.root/'assets.tar'
        with patch.object(portable,'selected_files',return_value=[p]):s=portable.pack(source,archive)
        self.assertEqual(s['files'],1);dest=self.root/'restore';portable.restore(archive,dest)
        self.assertEqual((dest/'downloads/original.npz').read_bytes(),p.read_bytes())
        self.assertEqual(portable.restore(archive,dest)['new_files'],0)
        (dest/'downloads/original.npz').write_bytes(b'edited')
        with self.assertRaisesRegex(ValueError,'existing input differs'):portable.restore(archive,dest)

    def test_transfer_rejects_tar_path_traversal(self):
        archive=self.root/'unsafe.tar';content=b'bad'
        with tarfile.open(archive,'w') as t:
            m=tarfile.TarInfo('../outside');m.size=len(content);t.addfile(m,io.BytesIO(content))
            blob=json.dumps({'files':[{'file':'../outside','bytes':3,'sha256':'x'}]}).encode()
            m=tarfile.TarInfo(portable.MANIFEST);m.size=len(blob);t.addfile(m,io.BytesIO(blob))
        with self.assertRaises(ValueError):portable.restore(archive,self.root/'restore')

    def launch(self,fail=False):
        root=self.root/'run';calls=[]
        def fake(stage,cmd,runroot):
            calls.append((stage,cmd))
            if fail and stage=='evaluate':raise RuntimeError('actual evaluation error simulated')
            if stage=='summarize':st.write_json(runroot/'experiment/status.json',dict(status='all_requested_stages_complete',all_metrics_complete=True,pending=[],numerical_failed_checks=0))
        with patch.object(autodl,'run',side_effect=fake),patch.object(autodl.subprocess,'run') as shutdown:
            rc=autodl.main(['--run-root',str(root),'--data',str(self.root/'data'),'--checkpoint',str(self.root/'checkpoint'),
                '--devices','cuda:0','cuda:1','--shutdown-on-success'])
        return root,rc,calls,shutdown

    def test_launcher_success_explicit_dual_and_pack_before_shutdown(self):
        root,rc,calls,shutdown=self.launch();self.assertEqual(rc,0);shutdown.assert_called_once_with(['shutdown','-h','now'],check=True)
        self.assertTrue((self.root/'run_review.tar.gz').is_file())
        self.assertEqual([s for s,c in calls],['environment','recover-support-preflight','recover-support','verify-support','preflight','prepare-support','support-loo','predict','evaluate','evaluate-aupro','summarize'])
        for stage,cmd in calls:
            if stage in ('recover-support','prepare-support','evaluate-aupro'):self.assertIn('cuda:1',cmd)

    def test_launcher_failure_preserves_and_never_shuts_down(self):
        root,rc,calls,shutdown=self.launch(True);self.assertEqual(rc,2);shutdown.assert_not_called()
        self.assertFalse(st.read_json(root/'pipeline_status.json')['success']);self.assertTrue((self.root/'run_FAILED_review.tar.gz').is_file())
        self.assertNotIn('evaluate-aupro',[s for s,c in calls])


if __name__=='__main__':unittest.main()
