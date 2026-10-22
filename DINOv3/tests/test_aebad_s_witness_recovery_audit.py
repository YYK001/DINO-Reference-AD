"""Small CPU drift/map checks; no real recovery, GT, Torch or CUDA."""
import copy
from pathlib import Path
import sys
import unittest
import numpy as np
from DINOv3.MADEqual import aebad_s_witness_recovery_audit as audit
from DINOv3.tests import test_aebad_s_support_witness_recovery as fixtures


class Audit(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.Recovery();self.fixture.setUp()
        self.bundle,self.root,self.spec,self.raw,self.banks=self.fixture.fixture()
        b=self.bundle;b.medians=np.zeros(4,np.float32);b.scales=np.ones(4,np.float32)
        b.state={k:dict(medians=[0.]*4,scales=[1.]*4) for k in ('peak','local')}
        self.raw['positions']=np.arange(9,dtype=np.int64);meta=dict(grid=[3,3],output_shape=[6,6],positions=self.raw['positions'])
        b.load=lambda sample:(copy.deepcopy(self.raw),dict(meta),{})
        current,ch1,response,_=audit.score_maps(dict(medians=b.medians,scales=b.scales,state=b.state),self.raw['segment_distance'],meta)
        b.references={};b.maps={}
        for s in b.samples:
            key=(s['domain'],s['sample_id'],audit.S)
            b.references[key]=dict(image_score=ch1,current_check_score=current)
            b.maps[key]=dict(map_float32_sha256=audit.array_sha(response))
    def tearDown(self):self.fixture.tearDown()

    def test_actual_geometry_maps_and_both_scores_unchanged(self):
        checks,scores=audit.audit_image(self.bundle,self.bundle.samples[0],{l:self.banks for l in audit.LAYERS})
        self.assertEqual(len(checks),8);self.assertTrue(all(c['strict_verified'] for c in checks))
        for key in ('CH1_signed_difference','CURRENT_signed_difference','map_max_abs_error'):self.assertEqual(scores[key],0)
        self.assertTrue(scores['same_cache_map_SHA_verified'])

    def test_small_drift_is_observed_without_changing_original_threshold(self):
        result=audit.difference(np.array([1.,1.000002,1.0000001]),np.ones(3))
        self.assertEqual(result['failed_positions'],1);self.assertFalse(result['strict_verified'])
        self.assertEqual((result['atol'],result['rtol']),(1e-7,1e-6))
        self.assertGreater(result['mean_signed_difference'],0)

    def test_nonfinite_or_mismatched_comparison_rejected(self):
        for actual,expected in [(np.array([np.nan]),np.ones(1)),(np.ones(2),np.ones(1))]:
            with self.assertRaises(ValueError):audit.difference(actual,expected)

    def test_same_cache_score_change_still_hard_fails(self):
        self.bundle.references['background','test_1',audit.S]['image_score']+=.001
        with self.assertRaisesRegex(ValueError,'same-cache historical score'):audit.audit_image(self.bundle,self.bundle.samples[0],{l:self.banks for l in audit.LAYERS})

    def test_same_cache_map_change_still_hard_fails(self):
        self.bundle.maps['background','test_1',audit.S]['map_float32_sha256']='edited'
        with self.assertRaisesRegex(ValueError,'same-cache frozen SEGMENT map'):audit.audit_image(self.bundle,self.bundle.samples[0],{l:self.banks for l in audit.LAYERS})

    def test_no_GT_or_label_required_and_frozen_module_not_changed(self):
        self.assertTrue(all('label' not in s for s in self.bundle.samples))
        audit.audit_image(self.bundle,self.bundle.samples[0],{l:self.banks for l in audit.LAYERS})
        self.assertNotEqual(Path(audit.__file__).parent,Path(audit.rec.__file__).parent)
        self.assertNotIn('torch',sys.modules)


if __name__=='__main__':unittest.main()
