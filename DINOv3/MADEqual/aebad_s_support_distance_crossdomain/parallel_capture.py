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
from . import inputs as ip
from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.verification import audit

# The environment probe requires 24 GiB before preflight. This second guard is
# after preflight has read the immutable ~5 GiB model/bank files into page cache.
INITIAL_RAM=20*2**30
SECOND_RAM=14*2**30
RUNNING_RAM_FLOOR=2*2**30


def commands(common,run_root):
    import sys
    from .kaggle import OLD
    return {d:[sys.executable,'-B','-u','-m',OLD,'capture',*common,'--domain',d,
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

    RAM thresholds are conservative admissions, not measured peak guarantees.
    A peer failure cancels the other process before the wrapper packs outputs.
    """
    if set(commands)!=set(ip.DOMAINS):raise ValueError('exactly the two frozen domains required')
    ram=ip.base.available_ram()
    if ram<INITIAL_RAM:raise MemoryError('dual capture needs 20 GiB available host RAM after preflight; no automatic single-GPU fallback')
    environments=device_environments(env)
    plan=dict(devices={'view':'cuda:0','illumination':'cuda:1'},start_policy='view_first_committed_raw_then_illumination',
        child_visible_devices={d:e['CUDA_VISIBLE_DEVICES'] for d,e in environments.items()},child_device='cuda:0',
        initial_available_ram_bytes=ram,initial_required_ram_bytes=INITIAL_RAM,second_required_ram_bytes=SECOND_RAM,
        running_available_ram_floor_bytes=RUNNING_RAM_FLOOR,
        extraction_batch_changed=False,precision_changed=False,Source_memory_rebuilt=False,
        RAM_estimate_role='admission only; real two-host peaks remain to be measured')
    ip.write_json(run_root/'parallel_capture_plan.json',plan)
    cancel=threading.Event();futures=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        first=pool.submit(execute,'capture_view',commands['view'],project,run_root,environments['view'],cancel_event=cancel)
        futures.append(first)
        try:
            while not committed_raw(run_root/'capture_shards/view'):
                if first.done():
                    first.result()
                    raise ValueError('view exited without a committed capture')
                time.sleep(.25)
            if first.done():first.result()
            ram=ip.base.available_ram();plan['second_start_available_ram_bytes']=ram
            ip.write_json(run_root/'parallel_capture_plan.json',plan)
            if ram<SECOND_RAM:raise MemoryError('less than 14 GiB available RAM before second host; preserving first domain; no method downgrade')
            futures.append(pool.submit(execute,'capture_illumination',commands['illumination'],project,run_root,environments['illumination'],cancel_event=cancel))
            active=set(futures);minimum=ram
            while active:
                done,active=wait(active,timeout=1,return_when=FIRST_COMPLETED)
                for future in done:future.result()
                ram=ip.base.available_ram();minimum=min(minimum,ram)
                plan['minimum_observed_available_ram_bytes']=minimum
                if active and ram<RUNNING_RAM_FLOOR:
                    raise MemoryError('host RAM headroom below 2 GiB; cancelling both captures before further work')
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
