"""Explicit historical compatibility; synthetic CPU only, GPU launcher mocked."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from DINOv3.tests import test_aebad_s_witness_recovery_audit as fixtures
from DINOv3.MADEqual.aebad_s_support_witness_segment_p0 import compatibility as c,recovery as rec,inputs as ip,storage as st,runner,autodl


class Compatibility(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.Audit();self.fixture.setUp()
        self.bundle=self.fixture.bundle;self.base=self.fixture.fixture.root;self.source=self.base/'source_extraction';self.cert=self.base/'new_certificate'
        b=self.bundle;formal=self.base/'formal';formal.mkdir();st.write_json(formal/'memory_manifest.json',dict(host_config={'synthetic_host':True}))
        b.bg['paths']=dict(formal_dir=formal)
        frozen=dict(self.fixture.spec['frozen'],query=rec.QUERY,precision=rec.PRECISION,checkpoint_sha256=rec.CHECKPOINT_SHA256,
            host_config={'synthetic_host':True},extractor='DINOv3.ShiftADD.aebad_s_p0.memory._extract_four_layers',brightness_factor=None,
            Source_memory_loaded=False,Source_MAD_refitted=False,query_forward_calls=0)
        self.spec=dict(identity=st.digest(frozen),frozen=frozen);rec.initialize(self.source,self.spec,b)
        for row in b.original_expected['background']:
            with np.load(rec.feature_path(self.fixture.root,'background',row['support_index']),allow_pickle=False) as z:a={k:z[k] for k in z.files}
            st.seal(rec.feature_path(self.source,'background',row['support_index']),self.spec['identity'],a,rec.metadata(row))
    def tearDown(self):self.fixture.tearDown()

    def test_actual_CPU_verify_attach_and_external_locations(self):
        before={str(p):st.sha(p) for p in self.source.rglob('*') if p.is_file()}
        m=c.verify(self.bundle,self.cert,self.source,workers=2);self.assertEqual(m['images'],2);self.assertTrue(m['strict_distance_compatible'])
        rec.attach(self.bundle,self.cert)
        found,available,missing,bad,archives=ip.locate(self.bundle,[self.cert/'features'],['background'])
        self.assertEqual(len(found),8);self.assertFalse(missing);self.assertFalse((self.cert/'features').exists())
        self.assertEqual(before,{str(p):st.sha(p) for p in self.source.rglob('*') if p.is_file()})

    def test_strict_distance_failure_preserved_under_new_policy(self):
        actual=c.audit.audit_image
        def small_drift(*args):
            checks,scores=actual(*args);r=checks[0]
            r.update(strict_verified=False,failed_positions=1,max_abs_error=8e-7,mean_abs_error=8e-8,p99_abs_error=8e-7,
                max_abs_error_over_Source_MAD_scale=8e-7/float(self.bundle.scales[0]))
            return checks,scores
        # Only the historical drift summary is injected, not model/geometry/score kernels.
        with patch.object(c.audit,'audit_image',side_effect=small_drift):m=c.verify(self.bundle,self.cert,self.source,workers=2)
        self.assertFalse(m['strict_distance_compatible']);self.assertEqual(m['domain_summaries'][0]['strict_failed_image_layer_checks'],2)
        c.attach(self.bundle,self.cert);self.assertFalse(self.bundle.frozen['recovered_supports']['manifest']['strict_distance_compatible'])

    def test_modified_source_features_rejected_after_certificate(self):
        c.verify(self.bundle,self.cert,self.source);p=rec.feature_path(self.source,'background',1);p.write_bytes(p.read_bytes()+b'bad')
        with self.assertRaises(ValueError):c.attach(self.bundle,self.cert)

    def test_unsealed_intermediate_not_reenrolled(self):
        c.verify(self.bundle,self.cert,self.source)
        (self.cert/'compatibility_manifest.json').unlink();(self.cert/'compatibility_manifest.seal.json').unlink()
        (self.cert/'audits/background.seal.json').unlink()
        with self.assertRaisesRegex(ValueError,'unsealed/modified'):c.verify(self.bundle,self.cert,self.source)

    def test_range_units_completeness_and_oversized_drifts_block(self):
        c.verify(self.bundle,self.cert,self.source);original=st.read_json(self.cert/'audits/background.json')
        changes=[lambda r:r['scores'].pop(),lambda r:r['geometry'].pop(),
            lambda r:r['scores'][0].update(map_max_abs_error=1.01e-4),
            lambda r:r['scores'][0].update(CH1_recovered_geometry=r['scores'][0]['CH1_old']+2e-5,CH1_signed_difference=2e-5),
            lambda r:r['scores'][0].update(CURRENT_recovered_geometry=r['scores'][0]['CURRENT_old']+2e-5,CURRENT_signed_difference=2e-5),
            lambda r:r['scores'][0].update(map_mean_abs_error=float('nan')),
            lambda r:r['geometry'][0].update(max_abs_error_over_Source_MAD_scale=1.),
            lambda r:r['geometry'][0].update(failed_positions=1,strict_verified=True)]
        for change in changes:
            rows=copy.deepcopy(original);change(rows)
            with self.subTest(change=changes.index(change)),self.assertRaises(ValueError):c.validate_domain(self.bundle,'background',rows)

    def test_large_layer_error_cannot_hide_in_fused_map_cancellation(self):
        c.verify(self.bundle,self.cert,self.source);rows=st.read_json(self.cert/'audits/background.json');r=rows['geometry'][0]
        r.update(max_abs_error=.001,max_abs_error_over_Source_MAD_scale=.001/float(self.bundle.scales[0]),strict_verified=False,failed_positions=1)
        with self.assertRaisesRegex(ValueError,'per-layer MAD'):c.validate_domain(self.bundle,'background',rows)

    def test_modified_summary_cannot_erase_or_invent_strict_pass(self):
        c.verify(self.bundle,self.cert,self.source);p=self.cert/'compatibility_manifest.json';m=st.read_json(p);m['strict_distance_compatible']=False
        st.write_json(p,m);st.write_json(self.cert/'compatibility_manifest.seal.json',dict(sha256=st.sha(p),inputs_sha256=st.sha(self.cert/'compatibility_inputs.json')))
        with self.assertRaisesRegex(ValueError,'strict distance failure history'):c.attach(self.bundle,self.cert)

    def test_source_precision_protocol_mismatch_blocks(self):
        p=self.source/'recovery_inputs.json';spec=st.read_json(p);spec['frozen']['precision']=dict(rec.PRECISION,cuda_matmul_allow_tf32=True)
        spec['identity']=st.digest(spec['frozen']);st.write_json(p,spec)
        with self.assertRaisesRegex(ValueError,'source extraction protocol'):c.context(self.bundle,self.source)

    def test_strict_is_default_and_reuse_is_explicit(self):
        self.assertEqual(runner.parser().parse_args([]).recovery_policy,'strict')
        with self.assertRaisesRegex(ValueError,'must be used together'):
            autodl.main(['--run-root',str(self.base/'run'),'--data',str(self.base/'data'),'--checkpoint',str(self.base/'checkpoint'),
                '--reuse-recovery',str(self.source)])

    def test_reuse_launcher_skips_all_model_recovery_stages(self):
        run=self.base/'run';calls=[]
        def fake(stage,cmd,root):
            calls.append((stage,cmd))
            if stage=='summarize':st.write_json(root/'experiment/status.json',dict(status='all_requested_stages_complete',all_metrics_complete=True,pending=[],numerical_failed_checks=0))
        with patch.object(autodl,'run',side_effect=fake):
            rc=autodl.main(['--run-root',str(run),'--data',str(self.base/'data'),'--checkpoint',str(self.base/'checkpoint'),
                '--reuse-recovery',str(self.source),'--recovery-policy','historical-maps-v1'])
        self.assertEqual(rc,0);names=[s for s,cmd in calls]
        self.assertNotIn('recover-support',names);self.assertNotIn('recover-support-preflight',names)
        verify=next(cmd for s,cmd in calls if s=='verify-support');self.assertIn('--reuse-recovery',verify);self.assertIn('historical-maps-v1',verify)


if __name__=='__main__':unittest.main()
