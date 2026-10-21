"""Independent domain processes, then a checked, hard-linked cache union.

Only scheduling/storage changes. The original capture runner still performs
every full-image extraction, Exact search and geometry calculation unchanged.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import os
from pathlib import Path
import threading
import time
from . import inputs as ip,resources
from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.verification import audit

# The probe and this guard require 20 GiB of working-set headroom, crediting
# inactive file cache only. SECOND_RAM matches the original capture's 12 GiB.
INITIAL_RAM=20*2**30
SECOND_RAM=12*2**30
RUNNING_RAM_FLOOR=2*2**30


def commands(common,run_root):
    import sys
    worker='DINOv3.MADEqual.aebad_s_support_distance_crossdomain.capture_worker'
    return {d:[sys.executable,'-B','-u','-m',worker,'capture',*common,'--domain',d,
        '--device','cuda:0','--cache-root',str(run_root/'capture_shards'/d)]
        for d in ('view','illumination')}


def device_environments(env):
    env=dict(os.environ if env is None else env)
    visible=env.get('CUDA_VISIBLE_DEVICES')
    slots=visible.split(',') if visible else ['0','1']
    if len(slots)<2 or slots[0].strip()==slots[1].strip():raise ValueError('two distinct visible CUDA devices required')
    return {d:dict(env,CUDA_VISIBLE_DEVICES=slots[i].strip()) for i,d in enumerate(('view','illumination'))}


def committed_raw(root):
    status=root/'cache_status.json'
    if not status.is_file():return False
    state=ip.old.read_json(status)
    if state['status']=='capture_failed':raise RuntimeError('first capture domain failed; see its log')
    return any((root/ip.safe(r['file'])).is_file() for r in state['records'])


def run(commands,project,run_root,env,execute):
    """Start view on GPU0; start illumination/GPU1 after view's first capture.

    Inactive file cache receives working-set credit. Insufficient second-host
    RAM defers it; runtime pressure pauses it at a verified commit, then resumes
    serially. Genuine model/numerical errors are never converted to success.
    """
    if set(commands)!=set(ip.DOMAINS):raise ValueError('exactly the two frozen domains required')
    snapshot=resources.memory_snapshot();ram=snapshot['effective_available_bytes']
    if ram<INITIAL_RAM:raise MemoryError('capture needs 20 GiB initial effective RAM after preflight; outputs retained')
    environments=device_environments(env)
    plan=dict(devices={'view':'cuda:0','illumination':'cuda:1'},start_policy='view_first_committed_raw_then_illumination',
        child_visible_devices={d:e['CUDA_VISIBLE_DEVICES'] for d,e in environments.items()},child_device='cuda:0',
        initial_available_ram_bytes=ram,initial_memory=snapshot,
        initial_required_ram_bytes=INITIAL_RAM,second_required_ram_bytes=SECOND_RAM,
        running_available_ram_floor_bytes=RUNNING_RAM_FLOOR,
        extraction_batch_changed=False,precision_changed=False,Source_memory_rebuilt=False,
        RAM_estimate_role='working-set admission; only inactive file cache credited; not a peak guarantee',
        execution_mode='staggered_parallel',memory_observations=0)
    ip.write_json(run_root/'parallel_capture_plan.json',plan)
    cancel=threading.Event();pause=run_root/'pause_illumination.request';last_log=0
    def observe():
        nonlocal last_log
        value=resources.memory_snapshot();effective=value['effective_available_bytes']
        plan['latest_memory']=value;plan['memory_observations']+=1
        if effective<plan.get('minimum_observed_available_ram_bytes',float('inf')):
            plan['minimum_observed_available_ram_bytes']=effective;plan['minimum_memory']=value
        for name in ('cgroup_current_bytes','cgroup_anon_bytes'):
            if value.get(name) is not None:plan['peak_'+name]=max(plan.get('peak_'+name,0),value[name])
        now=time.monotonic()
        if now-last_log>=30:
            print('[capture RAM] strict=%.2f GiB inactive_file_credit=%.2f GiB effective=%.2f GiB' %
                (value['strict_available_bytes']/2**30,value['inactive_file_reclaim_credit_bytes']/2**30,effective/2**30),flush=True)
            last_log=now
        ip.write_json(run_root/'parallel_capture_plan.json',plan)
        return effective
    with ThreadPoolExecutor(max_workers=2) as pool:
        def submit(domain,label):
            worker_env=dict(environments[domain])
            if domain=='illumination':worker_env['DINO_CAPTURE_PAUSE_FILE']=str(pause)
            else:worker_env.pop('DINO_CAPTURE_PAUSE_FILE',None)
            return pool.submit(execute,label,commands[domain],project,run_root,worker_env,
                cancel_event=cancel,allowed_returncodes=(75,))
        first=submit('view','capture_view')
        def finish(active,second=None):
            paused=False
            while active:
                done,active=wait(active,timeout=1,return_when=FIRST_COMPLETED)
                for future in done:
                    rc=future.result()
                    if rc==75:
                        if future is not second or not pause.is_file():raise RuntimeError('unexpected capture pause')
                        if plan['execution_mode']=='staggered_parallel':
                            plan['execution_mode']='parallel_then_serial_child_RAM_admission'
                            print('[capture scheduling] child RAM admission deferred illumination; it will resume after view',flush=True)
                        paused=True
                ram=observe()
                if active and ram<RUNNING_RAM_FLOOR:
                    if second is not None and first in active and second in active:
                        if not pause.exists():
                            pause.write_text('pause at next verified commit\n')
                            plan['execution_mode']='parallel_then_serial_RAM_pressure'
                            print('[capture scheduling] RAM pressure: illumination will pause at a verified commit, then resume after view',flush=True)
                    elif not pause.exists():raise MemoryError('single active host has less than 2 GiB effective RAM headroom; outputs retained')
            return paused
        try:
            while not committed_raw(run_root/'capture_shards/view'):
                if first.done():
                    first.result()
                    raise ValueError('view exited without a committed capture')
                time.sleep(.25)
            if first.done():first.result()
            ram=observe();plan['second_start_available_ram_bytes']=ram
            ip.write_json(run_root/'parallel_capture_plan.json',plan)
            if ram<SECOND_RAM:
                plan['execution_mode']='serial_insufficient_second_host_RAM'
                print('[capture scheduling] second host deferred: finish view first, then illumination; unchanged math',flush=True)
                finish({first});second=submit('illumination','capture_illumination')
                finish({second})
            else:
                second=submit('illumination','capture_illumination')
                if finish({first,second},second):
                    pause.unlink()
                    second=submit('illumination','capture_illumination_serial_resume')
                    finish({second})
        except BaseException as exc:
            cancel.set()
            plan.update(status='parallel_capture_failed',error=f'{type(exc).__name__}: {exc}')
            ip.write_json(run_root/'parallel_capture_plan.json',plan)
            raise
    plan['status']='both_capture_processes_succeeded';ip.write_json(run_root/'parallel_capture_plan.json',plan)


def merge(ctx,shards,target):
    """Preserve each historical SHA, schema and audit; never approve new hashes."""
    target=Path(target).resolve();pending=target.with_name(target.name+'_merge_pending')
    if set(shards)!=set(ip.DOMAINS):raise ValueError('one shard per original domain required')
    if target.exists() or pending.exists():raise FileExistsError('fresh merge output required; partial/original shards retained')
    states={};provenance=[];checks=[]
    for domain,root in shards.items():
        root=Path(root);state=ip.old.cache_state(ctx,root)
        if state.get('status') not in ('capture_complete','partial_cache_verified'):
            raise ValueError('capture shard did not finish successfully: '+domain)
        samples=ctx['catalog']['domains'][domain]['samples']
        expected={(domain,s['sample_id']) for s in samples}
        if ({(r['domain'],r['sample_id']) for r in state['records']}!=expected
                or len(state['records'])!=len(expected)
                or {(r['domain'],r['support_index']) for r in state['supports']}!={(domain,i) for i in range(1,9)}
                or len(state['supports'])!=8 or state['sources']):
            raise ValueError('incomplete/foreign/external shard scope: '+domain)
        rows=ip.old.csv_file(root/'capture_verification.csv')
        if not rows or any(str(r['verified']).lower() not in ('true','1') or r.get('domain')!=domain for r in rows):
            raise ValueError('shard has failed/mis-scoped numerical history')
        current={(r['sample_id'],r['check_kind']) for r in rows if str(r['verified']).lower() in ('true','1')}
        if any((s['sample_id'],kind) not in current for s in samples for kind in ('SOURCE_CURRENT','UNION_CURRENT')):
            raise ValueError('shard lacks per-image Source/Union verification')
        for sample in samples:
            record=next(r for r in state['records'] if r['sample_id']==sample['sample_id'])
            arrays,meta=ip.old.load_npz(root/ip.safe(record['file']),record['sha256'])
            ip.old.validate_raw(ctx,sample,arrays,meta)
        hashes={name:ip.sha(root/name) for name in ('cache_status.json','input_manifest.json','protocol.json','capture_verification.csv','capture_environment.json')}
        provenance.append(dict(domain=domain,source_directory=str(root),metadata_sha256=hashes))
        states[domain]=state;checks.extend(rows)
    context=dict(ctx,paths={k:v for k,v in ctx['paths'].items() if k!='cache_root'})
    ip.old.new_output(context,pending,'merge-capture')
    records=[];supports=[]
    for domain in ip.DOMAINS:
        root=Path(shards[domain]);state=states[domain]
        for kind,output in (('records',records),('supports',supports)):
            keyed={r['sample_id' if kind=='records' else 'support_index']:r for r in state[kind]}
            order=[s['sample_id'] for s in ctx['catalog']['domains'][domain]['samples']] if kind=='records' else list(range(1,9))
            for key in order:
                r=keyed[key];destination=pending/ip.safe(r['file']);destination.parent.mkdir(parents=True,exist_ok=True)
                # Same filesystem required. Never silently duplicate several GB.
                # Geometry-complete raw needs no support-feature entity. Keep
                # its original provenance, without binding merge to those files.
                if kind=='records':os.link(root/ip.safe(r['file']),destination)
                output.append(dict(r))
    state=dict(tool=ip.old.TOOL,identity=ctx['identity'],status='capture_complete',records=records,supports=supports,sources={},
        invocations=[inv for d in ip.DOMAINS for inv in states[d].get('invocations',[])],
        historical_score_audit=audit(checks),merge_provenance=provenance,
        jointly_frozen_domains=list(ip.DOMAINS),GT_decoded=False,default_query_features_persisted=False,
        support_feature_entities_required=False,
        merge_storage='hard_links_original_bytes_no_copy_no_research')
    ip.write_csv(pending/'capture_verification.csv',checks);ip.write_json(pending/'cache_status.json',state)
    ip.write_json(pending/'capture_environment.json',dict(mode='two_independent_original_hosts',
        domains={d:ip.old.read_json(Path(shards[d])/'capture_environment.json') for d in ip.DOMAINS},
        device_assignment={'view':'cuda:0','illumination':'cuda:1'},shared_VRAM=False))
    ip.old.cache_state(ctx,pending)
    for p in provenance:
        if any(ip.sha(Path(p['source_directory'])/name)!=h for name,h in p['metadata_sha256'].items()):
            raise ValueError('shard metadata changed during merge; originals/pending retained')
    pending.rename(target)
    return target
