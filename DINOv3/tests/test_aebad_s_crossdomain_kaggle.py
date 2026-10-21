"""Live-parent and notebook orchestration checks, not a Kaggle/GPU experiment."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock
from DINOv3.MADEqual.aebad_s_support_distance_crossdomain import inputs as ip,kaggle,reporting
from DINOv3.MADEqual.aebad_s_support_distance_crossdomain.live_parent import LiveParent,unchanged
from DINOv3.MADEqual.aebad_s_support_distance_crossdomain import runner


def materialize(parent,root):
    for name,blob in parent.content.items():
        name=name.removeprefix(parent.prefix)
        if name.startswith('source_provenance/'):
            name=name.removeprefix('source_provenance/')
        if not name.startswith(('cache/','prediction/','frozen_inputs/','cpu/')) and name!='run_config.json':continue
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(blob)


class LiveRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=ip.ROOT/'downloads'/ip.SUMMARY_NAME
        if not path.exists():raise unittest.SkipTest('approved historical metadata absent')
        cls.parent=ip.Parent(path)
    def test_complete_live_metadata_portable_without_old_absolute_assets(self):
        with tempfile.TemporaryDirectory(dir=ip.ROOT) as f:
            root=Path(f);materialize(self.parent,root);p=LiveParent(root)
            self.assertEqual(p.origin,'new_complete_pipeline_rerun')
            self.assertEqual(p.identity,self.parent.identity)
            self.assertEqual(p.records,self.parent.records)
            self.assertNotEqual(p.source_sha256,ip.SUMMARY_SHA)
            self.assertEqual(len(p.required()),629);unchanged(p)
    def test_mutated_same_run_reference_is_not_accepted(self):
        with tempfile.TemporaryDirectory(dir=ip.ROOT) as f:
            root=Path(f);materialize(self.parent,root)
            p=root/'cpu/per_image_scores.csv';p.write_bytes(p.read_bytes()+b'\n')
            with self.assertRaisesRegex(ValueError,'SHA mismatch'):LiveParent(root)
    def test_incomplete_capture_and_partial_cpu_refused(self):
        for directory,field,value in [('cache','status','partial_cache_verified'),('cpu','images',628)]:
            with self.subTest(directory=directory),tempfile.TemporaryDirectory(dir=ip.ROOT) as f:
                root=Path(f);materialize(self.parent,root)
                p=root/directory/('cache_status.json' if directory=='cache' else 'status.json')
                data=json.loads(p.read_bytes());data[field]=value;ip.write_json(p,data)
                with self.assertRaises(ValueError):LiveParent(root)
    def test_frozen_patch_MAD_stays_pinned(self):
        with tempfile.TemporaryDirectory(dir=ip.ROOT) as f:
            root=Path(f);materialize(self.parent,root)
            (root/'frozen_inputs/formal/source_mad_state.npz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'SHA mismatch'):LiveParent(root)
    def test_no_mixing_completed_domains_from_two_runs(self):
        with tempfile.TemporaryDirectory(dir=ip.ROOT) as f:
            root=Path(f);(root/'status.json').write_text('{}')
            def result(path,domain):return dict(parent_sha256=domain,parent_origin='new_complete_pipeline_rerun')
            with mock.patch.object(reporting,'verified_result',side_effect=result):
                with self.assertRaisesRegex(ValueError,'different parent runs'):
                    reporting.summarize(None,{d:root for d in ip.DOMAINS},root)


class DiscoveryCatalog(unittest.TestCase):
    def test_code_catalog_matches_original_primary_statistics(self):
        catalog=Path(reporting.__file__).with_name('background_discovery_catalog.json')
        small=reporting.verified_catalog(catalog)
        original=ip.ROOT/'DINOv3/outputs/mad_equal/aebad_s_support_distance_diagnostic_20261021_054307'
        if not original.exists():self.skipTest('local discovery output absent')
        full=reporting.verified_result(original,'background')
        self.assertEqual(small['comparisons'],[r for r in full['comparisons'] if r['feature'] in ip.MAIN_FEATURES])
        self.assertEqual(len(small['comparisons']),24)
    def test_clone_only_reference_summarizes_without_original_output_tree(self):
        catalog=Path(reporting.__file__).with_name('background_discovery_catalog.json')
        with tempfile.TemporaryDirectory(dir=ip.ROOT) as f:
            root=Path(f);target=root/'catalog.json';target.write_bytes(catalog.read_bytes())
            rows=reporting.summarize(target,{},root)
            self.assertEqual(len(rows),36)
            self.assertEqual(sum(r['status']=='pending' for r in rows),24)
            self.assertTrue(all(r['parent_origin']=='frozen_discovery_catalog' for r in rows if r['domain']=='background'))
            target.write_bytes(target.read_bytes()+b'\n')
            with self.assertRaisesRegex(ValueError,'modified background'):reporting.verified_catalog(target)


class Orchestration(unittest.TestCase):
    def test_diagnostic_can_be_sibling_of_frozen_stages_but_cannot_overwrite_them(self):
        with tempfile.TemporaryDirectory(dir=ip.ROOT) as f:
            root=Path(f)
            args=runner.parser().parse_args(['analyze','--parent-run',str(root),'--output-dir',str(root/'diagnostic')])
            self.assertEqual(runner.fresh(args,'analyze'),root/'diagnostic')
            args.output_dir=root/'cpu'/'new_inside_old'
            with self.assertRaisesRegex(ValueError,'overlaps'):runner.fresh(args,'analyze')
    def test_never_runs_model_when_invoked_from_local_workspace(self):
        with self.assertRaisesRegex(ValueError,'Kaggle execution only'):
            kaggle.main(ip.ROOT,b'',hashlib.sha256(b'').hexdigest())
    def test_missing_inputs_list_clear_and_no_inference(self):
        with tempfile.TemporaryDirectory(dir=ip.ROOT) as f:
            with self.assertRaisesRegex(FileNotFoundError,'formal'):kaggle.inputs(Path(f))
    def test_stage_exit_and_log_are_recorded(self):
        with tempfile.TemporaryDirectory(dir=ip.ROOT) as f:
            root=Path(f)
            kaggle.command('tiny_cpu',[sys.executable,'-B','-c',"print('CPU_ONLY')"],ip.ROOT,root,None)
            self.assertEqual((root/'tiny_cpu.exitcode').read_text().strip(),'0')
            self.assertIn('CPU_ONLY',(root/'tiny_cpu.log').read_text())
            with self.assertRaisesRegex(RuntimeError,'exit_code=3'):
                kaggle.command('failed_cpu',[sys.executable,'-B','-c','raise SystemExit(3)'],ip.ROOT,root,None)
            self.assertEqual((root/'failed_cpu.exitcode').read_text().strip(),'3')
    def test_failed_run_raw_archive_keeps_distance_cache_not_support_features(self):
        with tempfile.TemporaryDirectory(dir=ip.ROOT) as f:
            root=Path(f)/'run';root.mkdir()
            for name in ('cache/raw_cache/illumination/test_1.npz','cache/support_features/illumination/001.npz',
                         'prediction/predictions/illumination/test_1.npz','frozen_inputs/formal/source_mad_state.npz'):
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fixture')
            (root/'run_config.json').write_text('{}')
            state=dict(computation_success=False);summary=kaggle.pack(root,state)
            self.assertFalse(state['experiment_success']);self.assertTrue(summary.is_file())
            with tarfile.open(state['raw_archive']) as t:names=t.getnames()
            self.assertIn('run/cache/raw_cache/illumination/test_1.npz',names)
            self.assertNotIn('run/cache/support_features/illumination/001.npz',names)
            self.assertNotIn('run/prediction/predictions/illumination/test_1.npz',names)
            with tarfile.open(summary) as t:small=t.getnames()
            self.assertIn('run/frozen_inputs/formal/source_mad_state.npz',small)
            self.assertFalse(any('/raw_cache/' in n for n in small))


if __name__=='__main__':unittest.main()
