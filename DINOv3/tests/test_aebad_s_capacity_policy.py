"""No real model/GPU execution. Real NumPy metrics and original cache identity."""
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import unittest
from unittest import mock
import numpy as np
from DINOv3.MADEqual.aebad_s_support_distance_crossdomain import inputs as ip,resume
from DINOv3.MADEqual.aebad_s_support_distance_crossdomain import execution_worker
from DINOv3.MADEqual.aebad_s_support_distance_crossdomain.resource_policy import observation_only,disable_guards
from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic import metrics


def guarded_allocate():
    if True:raise MemoryError('only the reviewed capacity check')
    raise MemoryError('actual allocation failed')


class CapacityPolicy(unittest.TestCase):
    def test_preflight_and_rejected_existing_output_stay_read_only(self):
        from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0 import runner as original
        with tempfile.TemporaryDirectory(dir=ip.ROOT) as f:
            root=Path(f);(root/'keep').write_text('untouched')
            with mock.patch.object(original,'main',return_value=0):
                self.assertEqual(execution_worker.main(['preflight','--cache-root',str(root)]),0)
            with mock.patch.object(original,'main',side_effect=FileExistsError('existing output')):
                with self.assertRaises(FileExistsError):execution_worker.main(['evaluate','--output-dir',str(root)])
            self.assertEqual([p.name for p in root.iterdir()],['keep'])
    def test_actual_failure_policy_recorded_only_in_new_output(self):
        from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0 import runner as original
        with tempfile.TemporaryDirectory(dir=ip.ROOT) as f:
            root=Path(f)/'new'
            def allocate(argv):root.mkdir();raise MemoryError('actual allocation failure')
            with mock.patch.object(original,'main',side_effect=allocate):
                with self.assertRaisesRegex(MemoryError,'actual allocation failure'):
                    execution_worker.main(['evaluate','--output-dir',str(root)])
            self.assertFalse(json.loads((root/'capacity_policy.json').read_bytes())['capacity_estimates_enforced'])
    def test_real_exact_metric_runs_with_zero_estimated_budget_and_same_values(self):
        y=np.array([0,1,0,1,1,0],np.uint8);s=np.array([.1,.2,.2,.8,.7,.2],np.float32)
        expected=metrics.exact_binary(y,s,memory_budget_bytes=1<<30)
        with observation_only() as audit:
            actual=metrics.exact_binary(y,s,memory_budget_bytes=0)
            self.assertEqual(actual,expected)
            self.assertEqual(sum(len(r['removed_guards']) for r in audit['functions']),10)
            self.assertFalse(audit['measurements_falsified'])
        with self.assertRaises(MemoryError):metrics.exact_binary(y,s,memory_budget_bytes=0)
    def test_actual_allocation_failure_propagates(self):
        y=np.array([0,1],np.uint8);s=np.array([0,1],np.float32)
        with observation_only(),mock.patch.object(np,'argsort',side_effect=MemoryError('actual allocation failed')):
            with self.assertRaisesRegex(MemoryError,'actual allocation failed'):metrics.exact_binary(y,s,memory_budget_bytes=0)
    def test_numerical_input_validation_is_not_disabled(self):
        with observation_only():
            with self.assertRaisesRegex(ValueError,'complete finite binary'):metrics.exact_binary(np.array([0,1]),np.array([0.,np.nan]),memory_budget_bytes=0)
    def test_only_reviewed_guard_removed_actual_error_preserved(self):
        replacement,audit=disable_guards(guarded_allocate,('only the reviewed capacity check',))
        with self.assertRaisesRegex(MemoryError,'actual allocation failed'):replacement()
        self.assertEqual(len(audit['removed_guards']),1)
        with self.assertRaisesRegex(ValueError,'reviewed guard list'):disable_guards(guarded_allocate,('different message',))
    def test_original_files_code_hashes_and_resource_measurements_unchanged(self):
        hashes=ip.old.code_hashes();original=metrics.exact_binary
        with mock.patch.object(ip.old.common,'available_ram',return_value=1):
            with observation_only():
                self.assertEqual(ip.old.common.available_ram(),1)
                self.assertNotIn('torch',sys.modules)
        self.assertIs(metrics.exact_binary,original)
        self.assertEqual(ip.old.code_hashes(),hashes)
    def test_resume_actual_saved_identity_without_bank_or_checkpoint_access(self):
        package=ip.ROOT/'downloads/aebad_full_crossdomain_20261021_123058_823a60_FAILED_SUMMARY.tar.gz'
        if not package.exists():self.skipTest('actual failed summary absent in code-only export')
        with tempfile.TemporaryDirectory(dir=ip.ROOT) as f:
            root=Path(f);source=root/'saved';source.mkdir()
            with tarfile.open(package) as t:
                for m in t.getmembers():
                    parts=Path(m.name).parts
                    if m.isfile() and len(parts)>1 and (parts[1] in ('cache','prediction','frozen_inputs') or parts[1]=='run_config.json'):
                        p=source.joinpath(*parts[1:]);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(t.extractfile(m).read())
            snapshot={str(p.relative_to(source)):ip.sha(p) for p in source.rglob('*') if p.is_file()}
            # Windows symlink privilege is not assumed. Only this filesystem
            # operation is replaced by tiny metadata copies; Linux links remain
            # a real-environment validation item. All real identity code runs.
            def link(path,target,**kwargs):shutil.copytree(target,path)
            out=root/'resume'
            with mock.patch.object(Path,'symlink_to',link):resume.prepare(source,out,root/'absent_model_free_data')
            ctx=resume.context(out,root/'absent_model_free_data')
            self.assertEqual(ctx['identity'],'e21f243fba926ea04c4593e1c3472c02f228c1c47d5bccbee01f75b46a45e6bb')
            self.assertEqual(snapshot,{str(p.relative_to(source)):ip.sha(p) for p in source.rglob('*') if p.is_file()})
            self.assertFalse(json.loads((out/'resume_source.json').read_bytes())['model_inference'])
            archive=resume.pack(out,False)
            with tarfile.open(archive) as t:
                self.assertFalse(any(m.issym() or m.islnk() for m in t.getmembers()))
                self.assertTrue(any(m.name.endswith('cache/cache_status.json') for m in t.getmembers()))


if __name__=='__main__':unittest.main()
