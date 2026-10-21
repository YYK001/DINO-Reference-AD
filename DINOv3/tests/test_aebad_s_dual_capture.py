"""Actual CPU scheduling/merge paths on tiny caches; no Torch or CUDA run."""
import copy
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock
import numpy as np
from DINOv3.MADEqual.aebad_s_support_distance_crossdomain import inputs as ip,kaggle,parallel_capture as pc
from DINOv3.MADEqual.aebad_s_support_distance_crossdomain import resources,capture_worker
from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0 import runner as original_runner
from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0 import method,prediction
from DINOv3.tests.test_aebad_s_support_distance_crossdomain import fixture


def memory(gib):
    return dict(effective_available_bytes=int(gib*2**30),strict_available_bytes=int(gib*2**30),
                inactive_file_reclaim_credit_bytes=0)


def shards_fixture(root):
    contexts={};shards={}
    for d in ip.DOMAINS:
        p,s,a,m,blob,r,gt=fixture(d);contexts[d]=(p,s,a,m)
    ctx=dict(identity=contexts['view'][0].identity,frozen={'protocol':ip.old.PROTOCOL},
        catalog={'domains':{d:contexts[d][0].catalog['domains'][d] for d in ip.DOMAINS}},
        paths={'project':root/'project','cache_root':root/'cache'})
    state=copy.deepcopy(contexts['view'][0].state)
    state['state_sha256']=ip.digest(state)
    ctx.update(medians=contexts['view'][0].medians,scales=contexts['view'][0].scales,state=state)
    for d,(p,s,a,m) in contexts.items():
        shard=root/'capture_shards'/d;shard.mkdir(parents=True);shards[d]=shard
        a=dict(a,source_index=np.zeros_like(a['source_distance'],np.int64),
            target_index=np.broadcast_to(a['positions'],a['target_distance'].shape).copy(),
            alpha=np.full_like(a['source_distance'],.5),endpoint_length=np.ones_like(a['source_distance']),
            endpoint_support=np.broadcast_to(np.array([1,2],np.int8)[None,:,None],(4,2,16)).copy(),
            endpoint_index=np.broadcast_to(a['positions'],(4,2,16)).copy())
        m=copy.deepcopy(m);m['source_rows']=[100000]*4
        for support in m['supports']:support['grid']=[4,4]
        ip.old.validate_raw(ctx,s,a,m)
        relative=f'raw_cache/{d}/{s["sample_id"]}.npz';ip.old.save_npz(shard/relative,a,m)
        records=[dict(domain=d,sample_id=s['sample_id'],file=relative,sha256=ip.sha(shard/relative),action='synthetic_test_only')]
        supports=[]
        for sup in m['supports']:
            rel=f'support_features/{d}/{sup["support_index"]:03d}.npz';path=shard/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'synthetic_feature_identity')
            supports.append(dict(domain=d,file=rel,sha256=ip.sha(path),support_index=sup['support_index'],
                                 relative_path=sup['relative_path'],image_sha256=sup['image_sha256'],grid=[4,4]))
        state=dict(tool=ip.old.TOOL,identity=ctx['identity'],status='partial_cache_verified',records=records,supports=supports,sources={},invocations=[])
        ip.write_json(shard/'cache_status.json',state);ip.write_json(shard/'protocol.json',ip.old.PROTOCOL)
        ip.write_json(shard/'input_manifest.json',dict(identity=ctx['identity'],frozen=ctx['frozen']))
        ip.write_json(shard/'capture_environment.json',dict(device=d,synthetic=True))
        ip.write_csv(shard/'capture_verification.csv',[dict(domain=d,sample_id=s['sample_id'],check_kind=kind,verified=True)
                    for kind in ('SOURCE_CURRENT','UNION_CURRENT')])
    return ctx,shards


class CacheMerge(unittest.TestCase):
    def test_merged_raw_feeds_original_three_branch_prediction_and_scoring(self):
        with tempfile.TemporaryDirectory(dir=ip.ROOT) as f:
            root=Path(f);ctx,shards=shards_fixture(root);out=pc.merge(ctx,shards,root/'cache')
            state=ip.old.cache_state(ctx,out)
            for r in state['records']:
                sample=ctx['catalog']['domains'][r['domain']]['samples'][0]
                raw,meta=ip.old.load_npz(out/r['file'],r['sha256'])
                arrays,_=method.predict(raw,tuple(meta['grid']));checks=[]
                scores=prediction.scores_for(ctx,arrays,meta,sample,checks)
                self.assertEqual(len(scores),3);self.assertTrue(all(c['verified'] for c in checks))
                self.assertTrue(all(np.isfinite(s['image_score']) for s in scores))
    def test_merges_real_schema_in_catalog_order_with_same_bytes_and_no_duplicate_disk(self):
        with tempfile.TemporaryDirectory(dir=ip.ROOT) as f:
            root=Path(f);ctx,shards=shards_fixture(root)
            snapshots={d:(p/'cache_status.json').read_bytes() for d,p in shards.items()}
            out=pc.merge(ctx,shards,root/'cache');state=ip.old.cache_state(ctx,out)
            self.assertEqual(state['status'],'capture_complete')
            self.assertEqual([r['domain'] for r in state['records']],list(ip.DOMAINS))
            self.assertEqual(len(state['records']),2);self.assertEqual(len(state['supports']),16)
            for r in state['records']:
                self.assertTrue(os.path.samefile(out/r['file'],shards[r['domain']]/r['file']))
                self.assertEqual(ip.sha(out/r['file']),r['sha256'])
            self.assertEqual(snapshots,{d:(p/'cache_status.json').read_bytes() for d,p in shards.items()})
            with self.assertRaises(FileExistsError):pc.merge(ctx,shards,out)
    def test_completed_raw_does_not_require_support_feature_entities(self):
        with tempfile.TemporaryDirectory(dir=ip.ROOT) as f:
            root=Path(f);ctx,shards=shards_fixture(root)
            for path in shards.values():
                for file in (path/'support_features').glob('*/*.npz'):file.unlink()
            out=pc.merge(ctx,shards,root/'cache')
            self.assertFalse(ip.old.read_json(out/'cache_status.json')['support_feature_entities_required'])
    def test_modified_raw_is_rejected_instead_of_reenrolled(self):
        with tempfile.TemporaryDirectory(dir=ip.ROOT) as f:
            root=Path(f);ctx,shards=shards_fixture(root)
            p=next((shards['view']/'raw_cache').glob('*/*.npz'));p.write_bytes(p.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'modified'):pc.merge(ctx,shards,root/'cache')
            self.assertFalse((root/'cache').exists())
    def test_failed_and_incomplete_domain_cannot_merge(self):
        for change in ('status','records','verified'):
            with self.subTest(change=change),tempfile.TemporaryDirectory(dir=ip.ROOT) as f:
                root=Path(f);ctx,shards=shards_fixture(root);p=shards['view']/'cache_status.json'
                if change=='verified':
                    ip.write_csv(shards['view']/'capture_verification.csv',[dict(domain='view',sample_id='test_same',check_kind='SOURCE_CURRENT',verified=False)])
                else:
                    data=ip.old.read_json(p)
                    if change=='status':data['status']='capture_failed'
                    else:
                        (shards['view']/data['records'][0]['file']).unlink();data['records']=[]
                    ip.write_json(p,data)
                with self.assertRaises(ValueError):pc.merge(ctx,shards,root/'cache')
    def test_original_identity_and_numerical_history_must_match(self):
        with tempfile.TemporaryDirectory(dir=ip.ROOT) as f:
            root=Path(f);ctx,shards=shards_fixture(root);p=shards['illumination']/'input_manifest.json'
            value=ip.old.read_json(p);value['identity']='another_run';ip.write_json(p,value)
            with self.assertRaisesRegex(ValueError,'different joint'):pc.merge(ctx,shards,root/'cache')


def cpu_commands(root,fail=False):
    # Stand-ins represent only process lifetime, progress files and return code.
    # Extraction, model, precision and GPU memory are NOT validated here.
    view="""
from pathlib import Path
import json,time,sys
root=Path(sys.argv[1]);p=root/'capture_shards/view/raw_cache/view/test_first.npz'
p.parent.mkdir(parents=True);p.write_bytes(b'committed_stub')
(root/'capture_shards/view/cache_status.json').write_text(json.dumps({'status':'partial_cache_verified','records':[{'file':'raw_cache/view/test_first.npz'}]}))
time.sleep(.8)
(root/'view_finished').write_text(str(time.monotonic()))
"""
    illum="""
from pathlib import Path
import time,sys
root=Path(sys.argv[1]);assert (root/'capture_shards/view/cache_status.json').exists()
(root/'illumination_started').write_text(str(time.monotonic()))
"""+('\nraise SystemExit(4)' if fail else '')
    if fail:view=view.replace('time.sleep(.8)','time.sleep(5)')
    return {'view':[sys.executable,'-B','-c',view,str(root)],'illumination':[sys.executable,'-B','-c',illum,str(root)]}


class DomainScheduler(unittest.TestCase):
    def test_argument_overrides_use_two_devices_and_separate_caches(self):
        common=['--domain','both','--device','cuda:0','--cache-root','unused','--capture-min-free-vram-gib','14']
        commands=pc.commands(common,ip.ROOT)
        for d,cmd in commands.items():
            args=original_runner.parser().parse_args(cmd[5:])
            self.assertEqual(args.domain,d);self.assertEqual(args.device,'cuda:0')
            self.assertEqual(args.cache_root,ip.ROOT/'capture_shards'/d);self.assertEqual(args.capture_min_free_vram_gib,14)
        envs=pc.device_environments({'CUDA_VISIBLE_DEVICES':'GPU-A,GPU-B'})
        self.assertEqual(envs['view']['CUDA_VISIBLE_DEVICES'],'GPU-A')
        self.assertEqual(envs['illumination']['CUDA_VISIBLE_DEVICES'],'GPU-B')
        self.assertNotIn('OMP_NUM_THREADS',envs['view'])
        with self.assertRaises(ValueError):pc.device_environments({'CUDA_VISIBLE_DEVICES':'0'})
    def test_actual_two_cpu_subprocesses_overlap_after_stagger(self):
        with tempfile.TemporaryDirectory(dir=ip.ROOT) as f,mock.patch.object(resources,'memory_snapshot',return_value=memory(64)):
            root=Path(f);pc.run(cpu_commands(root),ip.ROOT,root,None,kaggle.command)
            self.assertLess(float((root/'illumination_started').read_text()),float((root/'view_finished').read_text()))
            self.assertEqual(ip.old.read_json(root/'parallel_capture_plan.json')['status'],'both_capture_processes_succeeded')
    def test_peer_failure_terminates_other_process_and_preserves_raw(self):
        with tempfile.TemporaryDirectory(dir=ip.ROOT) as f,mock.patch.object(resources,'memory_snapshot',return_value=memory(64)):
            root=Path(f)
            with self.assertRaisesRegex(RuntimeError,'exit_code=4'):pc.run(cpu_commands(root,True),ip.ROOT,root,None,kaggle.command)
            self.assertNotEqual((root/'capture_view.exitcode').read_text().strip(),'0')
            self.assertFalse((root/'view_finished').exists())
            self.assertTrue((root/'capture_shards/view/raw_cache/view/test_first.npz').is_file())
    def test_insufficient_ram_refuses_before_model_start(self):
        with tempfile.TemporaryDirectory(dir=ip.ROOT) as f,mock.patch.object(resources,'memory_snapshot',return_value=memory(19)):
            execute=mock.Mock()
            with self.assertRaisesRegex(MemoryError,'20 GiB'):pc.run({'view':[],'illumination':[]},ip.ROOT,Path(f),None,execute)
            execute.assert_not_called()
    def test_single_active_host_pressure_fails_without_method_downgrade(self):
        with tempfile.TemporaryDirectory(dir=ip.ROOT) as f,mock.patch.object(resources,'memory_snapshot',side_effect=[memory(64),memory(8),memory(1)]):
            root=Path(f);commands=cpu_commands(root)
            commands['view'][3]=commands['view'][3].replace('time.sleep(.8)','time.sleep(5)')
            commands['illumination'][3]+='\ntime.sleep(5)'
            with self.assertRaisesRegex(MemoryError,'less than 2 GiB'):pc.run(commands,ip.ROOT,root,None,kaggle.command)
            self.assertFalse((root/'view_finished').exists())
    def test_low_second_host_ram_finishes_first_instead_of_aborting(self):
        with tempfile.TemporaryDirectory(dir=ip.ROOT) as f,mock.patch.object(resources,'memory_snapshot',side_effect=lambda: memory(8 if (Path(f)/'capture_shards/view/cache_status.json').exists() and not (Path(f)/'view_finished').exists() else 64)):
            root=Path(f);pc.run(cpu_commands(root),ip.ROOT,root,None,kaggle.command)
            self.assertGreater(float((root/'illumination_started').read_text()),float((root/'view_finished').read_text()))
            self.assertEqual(ip.old.read_json(root/'parallel_capture_plan.json')['execution_mode'],'serial_insufficient_second_host_RAM')
    def test_parallel_pressure_pauses_then_resumes_second_after_first_exits(self):
        with tempfile.TemporaryDirectory(dir=ip.ROOT) as f:
            root=Path(f);commands=cpu_commands(root)
            commands['view'][3]=commands['view'][3].replace('time.sleep(.8)','time.sleep(2)')
            commands['illumination'][3]+="""
import os
p=Path(os.environ['DINO_CAPTURE_PAUSE_FILE'])
if not (root/'paused_once').exists():
 while not p.exists():time.sleep(.01)
 (root/'paused_once').write_text('1')
 raise SystemExit(75)
assert (root/'view_finished').exists()
"""
            calls=[memory(64),memory(64)]
            def snapshot():
                if calls:return calls.pop(0)
                return memory(1 if not (root/'paused_once').exists() else 64)
            with mock.patch.object(resources,'memory_snapshot',side_effect=snapshot):pc.run(commands,ip.ROOT,root,None,kaggle.command)
            self.assertEqual((root/'capture_illumination.exitcode').read_text().strip(),'75')
            self.assertEqual((root/'capture_illumination_serial_resume.exitcode').read_text().strip(),'0')
            self.assertGreater(float((root/'illumination_started').read_text()),float((root/'view_finished').read_text()))
            self.assertEqual(ip.old.read_json(root/'parallel_capture_plan.json')['execution_mode'],'parallel_then_serial_RAM_pressure')
    def test_progress_and_failed_pack_include_shards_without_double_count(self):
        with tempfile.TemporaryDirectory(dir=ip.ROOT) as f:
            root=Path(f)/'run';root.mkdir();p=root/'capture_shards/view/raw_cache/view/test_same.npz'
            p.parent.mkdir(parents=True);p.write_bytes(b'partial_raw')
            self.assertEqual(kaggle.capture_progress(root),1)
            state={'computation_success':False};kaggle.pack(root,state)
            with tarfile.open(state['raw_archive']) as t:self.assertIn('run/capture_shards/view/raw_cache/view/test_same.npz',t.getnames())
            dest=root/'cache/raw_cache/view/test_same.npz';dest.parent.mkdir(parents=True);os.link(p,dest)
            self.assertEqual(kaggle.capture_progress(root),1)


class WorkingSetAccounting(unittest.TestCase):
    def snapshot(self,current=22,inactive=6,files=7,anon=14,limit=30,system=900):
        with tempfile.TemporaryDirectory(dir=ip.ROOT) as f:
            root=Path(f)
            (root/'memory.max').write_text(str(int(limit*2**30)))
            (root/'memory.current').write_text(str(int(current*2**30)))
            (root/'memory.stat').write_text(f'anon {int(anon*2**30)}\nfile {int(files*2**30)}\ninactive_file {int(inactive*2**30)}\nslab_reclaimable {2**30}\n')
            return resources.memory_snapshot(root,int(system*2**30))
    def test_only_inactive_file_cache_receives_credit(self):
        s=self.snapshot()
        self.assertEqual(s['strict_available_bytes'],8*2**30)
        self.assertEqual(s['effective_available_bytes'],14*2**30)
        self.assertEqual(s['inactive_file_reclaim_credit_bytes'],6*2**30)
    def test_system_and_container_bounds_and_over_limit(self):
        self.assertEqual(self.snapshot(system=9)['effective_available_bytes'],9*2**30)
        self.assertEqual(self.snapshot(inactive=99,files=99)['effective_available_bytes'],30*2**30)
        self.assertEqual(self.snapshot(current=32,inactive=1)['effective_available_bytes'],0)
    def test_missing_stats_uses_strict_limit_not_host_memory(self):
        with tempfile.TemporaryDirectory(dir=ip.ROOT) as f:
            root=Path(f);(root/'memory.max').write_text(str(30*2**30));(root/'memory.current').write_text(str(22*2**30))
            self.assertEqual(resources.memory_snapshot(root,900*2**30)['effective_available_bytes'],8*2**30)
    def test_unlimited_or_missing_cgroup_does_not_fabricate_credit(self):
        with tempfile.TemporaryDirectory(dir=ip.ROOT) as f:
            root=Path(f)
            self.assertEqual(resources.memory_snapshot(root,7*2**30)['effective_available_bytes'],7*2**30)
            (root/'memory.max').write_text('max')
            self.assertEqual(resources.memory_snapshot(root,7*2**30)['effective_available_bytes'],7*2**30)


class WorkerCheckpoints(unittest.TestCase):
    def test_second_host_import_overhead_defers_before_host_allocation(self):
        with tempfile.TemporaryDirectory(dir=ip.ROOT) as f:
            stop=Path(f)/'pause.request'
            def invoke(argv):
                ip.old.common.available_ram();self.fail('must not reach model loading')
            with mock.patch.dict(os.environ,{'DINO_CAPTURE_PAUSE_FILE':str(stop)}),mock.patch.object(resources,'memory_snapshot',return_value=memory(11)):
                self.assertEqual(capture_worker.run([],invoke),75)
            self.assertTrue(stop.exists())
    def test_pause_preserves_environment_when_last_image_needs_cache_only_resume(self):
        with tempfile.TemporaryDirectory(dir=ip.ROOT) as f:
            root=Path(f);ctx,shards=shards_fixture(root);shard=shards['view'];stop=root/'pause.request';stop.touch()
            (shard/'capture_environment.json').unlink()
            # Metadata-only stand-in: exercises no real CUDA operation.
            cuda=SimpleNamespace(is_initialized=lambda:True,get_device_name=lambda:'synthetic GPU metadata',
                max_memory_allocated=lambda:123,max_memory_reserved=lambda:456)
            fake=SimpleNamespace(__version__='synthetic',version=SimpleNamespace(cuda=None),cuda=cuda)
            csv=shard/'capture_verification.csv';rows=ip.old.csv_file(csv)
            with mock.patch.dict(os.environ,{'DINO_CAPTURE_PAUSE_FILE':str(stop)}),mock.patch.dict(sys.modules,{'torch':fake}):
                self.assertEqual(capture_worker.run([],lambda argv:ip.old.write_csv(csv,rows)),75)
            env=ip.old.read_json(shard/'capture_environment.json')
            self.assertEqual(env['precision'],ip.old.PRECISION)
            self.assertEqual(env['peak_cuda_allocated_bytes'],123)
            self.assertIn('actual_worker_at_verified_RAM_pause',env['environment_scope'])
    def test_pause_waits_for_complete_verification_and_resumes_with_old_hashes(self):
        with tempfile.TemporaryDirectory(dir=ip.ROOT) as f:
            root=Path(f);ctx,shards=shards_fixture(root);shard=shards['view'];stop=root/'pause.request';stop.touch()
            csv=shard/'capture_verification.csv';rows=ip.old.csv_file(csv);snapshot=(shard/'cache_status.json').read_bytes()
            calls=[]
            def invoke(argv):
                ip.old.write_csv(csv,rows[:1]);calls.append('incomplete_commit_not_paused')
                ip.old.write_csv(csv,rows);self.fail('complete commit should pause')
            with mock.patch.dict(os.environ,{'DINO_CAPTURE_PAUSE_FILE':str(stop)}):
                self.assertEqual(capture_worker.run([],invoke),75)
            self.assertEqual(calls,['incomplete_commit_not_paused'])
            self.assertEqual((shard/'cache_status.json').read_bytes(),snapshot)
            ip.old.cache_state(ctx,shard)
            stop.unlink()
            with mock.patch.dict(os.environ,{'DINO_CAPTURE_PAUSE_FILE':str(stop)}):
                self.assertEqual(capture_worker.run([],lambda argv:0),0)
    def test_failed_checks_and_genuine_exceptions_are_not_converted_to_pause(self):
        with tempfile.TemporaryDirectory(dir=ip.ROOT) as f:
            root=Path(f);ctx,shards=shards_fixture(root);shard=shards['view'];stop=root/'pause.request';stop.touch()
            csv=shard/'capture_verification.csv';rows=ip.old.csv_file(csv);rows[0]['verified']=False
            def invoke(argv):
                ip.old.write_csv(csv,rows)
                raise ValueError('numerical failure')
            with mock.patch.dict(os.environ,{'DINO_CAPTURE_PAUSE_FILE':str(stop)}):
                with self.assertRaisesRegex(ValueError,'numerical failure'):capture_worker.run([],invoke)
            self.assertFalse((shard/'worker_pause.json').exists())
    def test_worker_patch_scoped_and_no_torch_import(self):
        before=ip.old.common.available_ram
        def invoke(argv):
            self.assertIsNot(ip.old.common.available_ram,before)
            with mock.patch.object(resources,'memory_snapshot',return_value=memory(64)):
                self.assertEqual(ip.old.common.available_ram(),64*2**30)
            self.assertNotIn('torch',sys.modules)
            return 0
        self.assertEqual(capture_worker.run([],invoke),0)
        self.assertIs(ip.old.common.available_ram,before)


if __name__=='__main__':unittest.main()
