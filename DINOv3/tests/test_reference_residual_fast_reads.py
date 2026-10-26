"""Trusted-local execution: no repeated payload scans; explicit full audit remains."""
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from DINOv3.MADEqual.aebad_s_reference_residual_segmentation import (
    storage as st,inputs as ip,protocol as pr,runner,runtime,retrieval,prediction,evaluation,training)
from DINOv3.tests.test_aebad_s_reference_residual_segmentation import Temporary,catalog


class FastReads(Temporary):
    def setUp(self):super().setUp();st.configure('fast')
    def bundle(self,name='arrays'):
        p=self.root/name;st.put(p,{'x':np.arange(40,dtype=np.float32)},dict(kind='constructed'))
        return p
    def test_default_policy_and_child_arguments(self):
        a=self.args();self.assertEqual(a.audit_mode,'fast')
        self.assertEqual(runner.common(a)[-1],'fast')
        self.assertFalse(st.full_audit())
    def test_fast_get_stores_and_hardlink_import_do_not_hash_payload(self):
        p=self.bundle();dest=self.root/'import'
        with patch.object(st,'sha',side_effect=AssertionError('unexpected payload hash')):
            for _ in range(3):
                np.testing.assert_array_equal(st.get(p)[0]['x'],np.arange(40,dtype=np.float32))
                st.Store().load(p)
            st.import_bundle(p,dest,{'x':'x'},dict(origin=str(p)))
            st.import_bundle(p,dest,{'x':'x'},dict(origin=str(p)))
            np.testing.assert_array_equal(st.get(dest)[0]['x'],st.get(p)[0]['x'])
    def test_full_audit_hashes_unchanged_file_once_across_readers_and_aliases(self):
        p=self.bundle();dest=self.root/'import';st.import_bundle(p,dest,{'x':'x'},dict(origin=str(p)))
        st.configure('full')
        with patch.object(st,'sha',wraps=st.sha) as sha:
            st.get(p);st.Store().load(p);st.get(dest);st.Store().load(dest)
        self.assertEqual(sha.call_count,1)
    def test_explicit_full_audit_detects_changed_payload_even_in_cached_reader(self):
        p=self.bundle();st.configure('full');store=st.Store();store.load(p)
        with (p/'x.npy').open('r+b') as f:f.seek(-4,2);f.write(np.float32(99).tobytes())
        with self.assertRaisesRegex(ValueError,'modified array'):store.load(p)
    def test_fast_still_rejects_wrong_shape_and_modified_metadata(self):
        p=self.bundle();st.configure('fast')
        with (p/'x.npy').open('wb') as f:np.save(f,np.zeros(20,np.float32))
        with self.assertRaisesRegex(ValueError,'layout'):st.get(p)
        p=self.bundle('metadata');r=st.read_json(p/'manifest.json');r['metadata']['kind']='changed';st.write_json(p/'manifest.json',r)
        with self.assertRaisesRegex(ValueError,'modified record'):st.get(p)
    def test_imported_full_grids_skip_full_value_scan_in_fast_mode(self):
        x={f'q{l}':np.ones((4,1280),np.float32) for l in pr.LAYERS};x['positions']=np.array([2,0,3,1])
        with patch.object(runtime.np,'isfinite',side_effect=AssertionError('unrequested full-grid scan')):
            runtime.raw_layers(x,dict(grid=[2,2]))
        x['q7']=np.ones((3,1280),np.float32)
        with self.assertRaisesRegex(ValueError,'invalid raw'):runtime.raw_layers(x,dict(grid=[2,2]))
    def test_fast_exact_keeps_distance_and_indices_without_CPU_recomputation(self):
        q=np.ones((2,1280),np.float32);bank=np.zeros((3,1280),np.float32);d=np.array([.1,.2],np.float32);i=np.array([0,1])
        with patch('DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.model.exact',return_value=(d,i)),patch.object(np.linalg,'norm',side_effect=AssertionError('CPU audit executed')):
            value,index,record=retrieval.exact_with_audit(q,bank,'cpu')
        np.testing.assert_array_equal(value,d);np.testing.assert_array_equal(index,i)
        self.assertEqual(record['status'],'not_requested')
        for distance,index in ((d,np.array([0,99])),(np.array([np.nan,.2]),i)):
            with patch('DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.model.exact',return_value=(distance,index)):
                with self.assertRaises(ValueError):retrieval.exact_with_audit(q,bank,'cpu')
    def test_real_CPU_exact_fast_and_full_return_identical_values(self):
        import torch
        torch.set_num_threads(1);rng=np.random.default_rng(18)
        q=rng.normal(size=(9,1280)).astype(np.float32);bank=rng.normal(size=(5,1280)).astype(np.float32)
        fast=retrieval.exact_with_audit(q,bank,'cpu');st.configure('full');full=retrieval.exact_with_audit(q,bank,'cpu')
        np.testing.assert_array_equal(fast[0],full[0]);np.testing.assert_array_equal(fast[1],full[1])
        self.assertEqual(full[2]['analytic_bound_failed'],0)
    def test_prediction_commit_check_does_not_rescan_all_payloads(self):
        a=self.args();c=catalog();c['samples']=[];cfg=pr.resolve()
        st.sealed_json(a.run_root/'inputs.json',dict(identity='fixture'))
        for d in pr.DOMAINS:
            samples=[dict(domain=d,sample_id=f'test_{i}',label=i) for i in (0,1)]
            c['samples'].extend(samples);cfg['populations'][d]=[2,1,1]
            st.sealed_json(a.run_root/'predictions'/(d+'.json'),dict(identity='fixture',records=[dict(domain=d,sample_id=s['sample_id'],seal='sealed-'+s['sample_id']) for s in samples]))
        st.sealed_json(a.run_root/'catalog.json',dict(catalog=c))
        with patch.object(ip,'config',return_value=cfg),patch.object(st,'get',side_effect=AssertionError('payload read')),patch.object(st.Store,'load',side_effect=AssertionError('payload scan')):
            self.assertIsInstance(prediction.verify(a),str)
        self.assertEqual(len(a._prediction_record_seals),6)
        # Metadata completeness still rejects a missing image without reading arrays.
        p=a.run_root/'predictions/background.json';p.unlink();st.sealed_json(p,dict(identity='fixture',records=[]))
        with patch.object(ip,'config',return_value=cfg):
            with self.assertRaisesRegex(ValueError,'incomplete prediction'):prediction.verify(a)
    def test_AUPRO_fast_skips_rehash_but_checks_shape_and_dtype(self):
        record=dict(output_height=3,output_width=4);response=np.ones((3,4),np.float32);mask=np.zeros((3,4),np.uint8)
        with patch.object(evaluation.original.shared,'verify_map_record',side_effect=AssertionError('full map hash')):
            evaluation.verify_map_record(record,response,mask)
            st.configure('full')
            with self.assertRaisesRegex(AssertionError,'full map hash'):evaluation.verify_map_record(record,response,mask)
        st.configure('fast')
        with self.assertRaisesRegex(ValueError,'geometry'):evaluation.verify_map_record(record,response[:2],mask)
    def test_checkpoint_read_uses_identity_schedule_without_payload_rehash(self):
        import torch
        p=self.root/'model';expected=dict(total_updates=1000,training_manifest='fixture');value=dict(updates=1,network={'x':torch.ones(1)})
        training.checkpoint(p,value,expected)
        with patch.object(st,'sha',side_effect=AssertionError('checkpoint rehashed')):
            self.assertEqual(training.load_checkpoint(p,expected,'cpu')['updates'],1)
        with self.assertRaisesRegex(ValueError,'inputs'):training.load_checkpoint(p,dict(expected,total_updates=4000),'cpu')
    def test_prepare_resume_does_not_rehash_known_backbone_weight(self):
        a=self.args();a.checkpoint.write_bytes(b'constructed weight, never loaded')
        st.write_json(a.official_root/'config.json',dict(scope='fixture'))
        a.dinov3_root.mkdir()
        st.sealed_json(a.run_root/'inputs.json',dict(extraction=dict(checkpoint_sha256='trusted-prior-identity')))
        original=st.sha
        def sha(path):
            if Path(path)==a.checkpoint:raise AssertionError('known backbone weight rehashed')
            return original(path)
        with patch.object(st,'sha',side_effect=sha):r=runtime.extraction_record(a)
        self.assertEqual(r['checkpoint_sha256'],'trusted-prior-identity')
        st.configure('full');self.assertEqual(runtime.extraction_record(a)['checkpoint_sha256'],original(a.checkpoint))
    def test_stopped_compact_inputs_remain_read_only_on_execution_only_resume(self):
        a=self.args();a.profile='compact_v1';cfg=pr.resolve();c=catalog();e=dict(scope='constructed')
        code=st.read_json(Path(pr.__file__).with_name('execution_compatibility.json'))['code']
        resolved=st.sealed_json(a.run_root/'resolved_config.json',dict(config=cfg));jobs=st.sealed_json(a.run_root/'job_plan.json',dict(plan=pr.job_plan(cfg)))
        frozen=st.sealed_json(a.run_root/'inputs.json',dict(config=cfg,identity='old-compact-identity',catalog=st.digest(c),extraction=e,cache=str(a.cache_root),query_mode=a.query_mode,code=code,
            resolved_config_seal=resolved['seal'],job_plan_seal=jobs['seal']))
        before={p:p.read_bytes() for p in a.run_root.glob('*.json')}
        self.assertEqual(ip.freeze(a,c,e),frozen)
        for p,value in before.items():self.assertEqual(p.read_bytes(),value)
        wrong=dict(frozen,code=dict(code));wrong['code'][next(iter(code))]='unknown'
        with self.assertRaisesRegex(ValueError,'implementation changed'):runner.validate_implementation(wrong)
        original=st.sha
        with patch.object(st,'sha',side_effect=lambda p:'unregistered-method-change' if Path(p).name=='model.py' else original(p)):
            with self.assertRaisesRegex(ValueError,'implementation changed'):runner.validate_implementation(frozen)


class FastIntegration(Temporary):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch=torch;torch.set_num_threads(1)
    def test_actual_compact_CPU_pipeline_with_default_fast_reads(self):
        from DINOv3.tests.test_aebad_s_reference_residual_segmentation import CPU
        st.configure('fast')
        original=st.sha
        def sha(path):
            if Path(path).suffix in ('.npy','.pt'):raise AssertionError('unexpected payload read hash')
            return original(path)  # First-write identities of new small result files.
        with patch.object(st,'sha',side_effect=sha):
            CPU.integration(self,'compact_v1')
