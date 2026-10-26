"""Explicit isolated engineering smoke; no formal training/evaluation completion."""
from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import datetime
from pathlib import Path
import tempfile
import threading
import time
import numpy as np
from . import inputs as ip,storage as st
from .protocol import CONFIG,DOMAINS,LAYERS


def sync(device):
    if str(device).startswith('cuda'):
        import torch
        torch.cuda.synchronize(device)


def timed(device,fn):
    sync(device);start=time.perf_counter();result=fn();sync(device)
    return result,time.perf_counter()-start


def estimate(head,assets,config=None,plan=None,inventory=None):
    """Requested remaining work only. Unknown stages stay unknown, never zero."""
    from .protocol import resolve,job_plan
    config=resolve() if config is None else config;plan=job_plan(config) if plan is None else plan
    inventory={} if inventory is None else inventory
    components={};unknown=[]
    if head.get('prepared_jobs'):
        indexed={(r['phase'],r['family']):r for r in head['prepared_jobs']}
        todo=[j for j in plan['training_jobs'] if (j['phase'],j['family'],j['seed']) not in inventory.get('completed_training_jobs',[])]
        if all((j['phase'],j['family']) in indexed and indexed[j['phase'],j['family']]['formal_BF16_verified'] for j in todo):
            components['head_training']=sum(j['updates']*indexed[j['phase'],j['family']]['seconds_per_update'] for j in todo)/3600
        else:unknown.append('requested real Dataset/BF16 job throughput')
    elif 'train_seconds_per_update' in head:
        updates=sum(j['updates'] for j in plan['training_jobs'] if (j['phase'],j['family'],j['seed']) not in inventory.get('completed_training_jobs',[]))
        components['head_compute_transfer_lower_bound']=updates*head['train_seconds_per_update']/3600
        unknown.append('real Dataset reading/reference assembly and checkpoint costs')
    else:unknown.append('formal BF16 head throughput')
    raw=inventory.get('normal_feature_forwards_remaining')
    if raw==0:components['original_normal_feature_forwards']=0.
    elif raw is not None and assets.get('backbone_seconds_per_image') is not None:
        components['original_normal_feature_forwards']=raw*assets['backbone_seconds_per_image']/3600
    else:unknown.append('missing original normal feature inventory/throughput')
    synth=inventory.get('synthetic_forwards_remaining')
    if synth==0:components['synthetic_RGB_forwards']=0.
    else:unknown.append('effective selected synthetic pool/forward throughput and sparse NN geometry')
    if assets.get('domains'):
        rows=list(assets['domains'].values());source=float(np.mean([r['warm_Source_seconds']+r['warm_query_upload_seconds'] for r in rows]))
        target=float(np.mean([r['warm_Target8_seconds'] for r in rows]))
        remaining=inventory.get('full_grid_calibration_prediction_NN_remaining')
        if remaining is not None:components['full_grid_calibration_prediction_NN']=remaining*(source+target)/3600
    stages={name:None for name in ('build_and_original_features','synthesis_and_training_cache','head_training','calibration_and_prediction','CPU_evaluation','AUPRO')}
    if 'head_training' in components:stages['head_training']=components['head_training']
    unknown += ['two-fold original sampler and candidate scratch I/O','synthesis generation/validity and cache writes',
        'CPU evaluation and original ADEval AUPRO','logit calibration/prediction heads and scoring']
    return dict(profile=config['profile'],job_plan=plan,inventory=inventory,components_hours=components,stage_hours=stages,
        measured_component_subtotal_hours=sum(components.values()),total_hours=None,pending=unknown,
        interpretation='component extrapolation only; unknown stages not zero; no unrequested controls or repeated computations',
        head_assumptions='constructed fixed inputs measure compute/transfer lower bound; prepared uses actual Dataset/prefetch; cold reads and checkpoint costs separately pending')


def head_benchmark(a,*,dataset=None,phase='source',family='REF_ST'):
    import torch
    from .model import Head
    from .training import prepare_batch,backward_batch,rate
    cfg=ip.config(a)
    cuda=str(a.device).startswith('cuda');dimension,channels=(1280,128) if cuda or dataset is not None else (16,8)
    bf16=cuda and torch.cuda.is_bf16_supported();torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(42);net=Head(dimension,channels).to(a.device);optimizer=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-4)
    fixed=None
    def batch(number):
        nonlocal fixed
        if dataset is None and fixed is not None:return fixed
        if dataset is not None:
            choices=dataset.select(42,number,source_queries_only=family=='REF_SOURCE_QUERY');samples=[dataset.load(c) for c in choices]
        else:
            gen=torch.Generator().manual_seed(number+20261026);samples=[]
            for i in range(4):
                h,w=(24,24) if cuda else (6,6);soft=np.zeros((h*4,w*4),np.float32)
                if i%2:soft[h:2*h,w:2*w]=.5
                samples.append(dict(values=[torch.randn(1,3*dimension+3,h,w,generator=gen) for _ in LAYERS],
                    soft=soft,origin=[0,0],grid=[h,w],shape=list(soft.shape),center=[4,4,h-4,w-4] if h>8 else [0,0,h,w]))
        value=(samples,prepare_batch(samples,pin=cuda))
        if dataset is None:fixed=value
        return value
    # Compare the actual production loss/backward against the previous sequential form.
    samples,packed=batch(0);old=copy.deepcopy(net);old.zero_grad();net.zero_grad()
    old_loss=backward_batch(old,prepare_batch(samples,pin=cuda,grouped=False),a.device,bf16=bf16)
    new_loss=backward_batch(net,packed,a.device,bf16=bf16)
    delta=sum(float((x.grad.float()-y.grad.float()).square().sum().item()) for x,y in zip(old.parameters(),net.parameters()))**.5
    norm=sum(float(x.grad.float().square().sum().item()) for x in old.parameters())**.5
    relative=delta/max(norm,1e-12);loss_delta=abs(float(new_loss)-float(old_loss))
    # This is an engineering gradient check, not a relaxation of frozen distance/map tolerances.
    bound=.03 if bf16 else 1e-4
    if not np.isfinite([relative,loss_delta]).all() or relative>bound or loss_delta>(.01 if bf16 else 1e-5):raise ValueError('grouped head gradient/loss comparison failed')
    del old,old_loss,new_loss,packed,samples
    rows=[];warmup=3
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(batch,0)
        for update in range(a.smoke_updates+warmup):
            start=time.perf_counter();_,groups=future.result();wait=time.perf_counter()-start
            if update+1<a.smoke_updates+warmup:future=pool.submit(batch,update+1)
            sync(a.device);start_step=time.perf_counter();optimizer.zero_grad(set_to_none=True)
            loss=backward_batch(net,groups,a.device,bf16=bf16)
            gradient=torch.nn.utils.clip_grad_norm_(net.parameters(),CONFIG['gradient_clip'])
            if not torch.isfinite(torch.stack([loss,gradient])).all():raise FloatingPointError('nonfinite smoke loss/gradient')
            for group in optimizer.param_groups:group['lr']=rate(update,cfg['source_updates' if phase=='source' else 'target_updates'])
            optimizer.step();sync(a.device);seconds=time.perf_counter()-start_step
            if update>=warmup:rows.append(dict(seconds=seconds+wait,data_wait_seconds=wait,compute_transfer_seconds=seconds))
            if update%10==0:print(f'[residual smoke head] {phase}/{family} {update}/{a.smoke_updates+warmup}',flush=True)
    # Independent temp checkpoint I/O, never a formal model/final-update marker.
    scratch=a.run_root/'smoke_checkpoint_scratch';scratch.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as tmp:
        path=Path(tmp)/'checkpoint.pt'
        _,save_seconds=timed(a.device,lambda:torch.save(dict(network=net.state_dict(),optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state()),path))
        restored,load_seconds=timed(a.device,lambda:torch.load(path,map_location=a.device,weights_only=False))
        for key,value in net.state_dict().items():torch.testing.assert_close(value,restored['network'][key],rtol=0,atol=0)
        checkpoint_bytes=path.stat().st_size
    mean=float(np.mean([r['seconds'] for r in rows]))
    return dict(phase=phase,family=family,seconds_per_update=mean,data_wait_seconds=float(np.mean([r['data_wait_seconds'] for r in rows])),
        p50=float(np.median([r['seconds'] for r in rows])),p95=float(np.quantile([r['seconds'] for r in rows],.95)),
        samples=rows,dimension=dimension,channels=channels,parameter_count=sum(x.numel() for x in net.parameters()),
        precision='BF16_head_FP32_loss' if bf16 else 'FP32_diagnostic_only' if cuda else 'CPU_FP32_prepared_diagnostic_only' if dataset is not None else 'CPU_FP32_small_constructed',
        formal_BF16_verified=bf16,grouped_loss_abs_delta=loss_delta,grouped_gradient_relative_L2=relative,
        gradient_bound=bound,checkpoint_write_seconds=save_seconds,checkpoint_load_seconds=load_seconds,checkpoint_bytes=checkpoint_bytes,
        actual_cache_loading=dataset is not None,GT_used=False)


def small_exact(a):
    from .execution import resident
    from .retrieval import exact_with_audit
    rng=np.random.default_rng(42);cuda=str(a.device).startswith('cuda')
    q=rng.normal(size=(256 if cuda else 32,1280)).astype(np.float32)
    bank=rng.normal(size=(4096 if cuda else 128,1280)).astype(np.float32);search=resident(a)
    qt,query_seconds=timed(a.device,lambda:search.query(q))
    old,old_seconds=timed(a.device,lambda:exact_with_audit(q,bank,a.device))
    new,cold=timed(a.device,lambda:exact_with_audit(q,bank,a.device,search=search,key='constructed',query=qt))
    again,warm=timed(a.device,lambda:exact_with_audit(q,bank,a.device,search=search,key='constructed',query=qt))
    _,old_warm=timed(a.device,lambda:exact_with_audit(q,bank,a.device))
    np.testing.assert_array_equal(old[0],new[0]);np.testing.assert_array_equal(old[1],new[1]);np.testing.assert_array_equal(new[0],again[0])
    return dict(scope='constructed_partial_shape_not_full_grid',query_rows=len(q),bank_rows=len(bank),dimension=1280,
        original_seconds=old_seconds,original_warm_seconds=old_warm,query_upload_seconds=query_seconds,
        cold_reference_search_seconds=cold,warm_reference_search_seconds=warm,
        timing_scope='resident searches exclude separately measured shared Q upload; original includes Q/reference transfer; no startup speedup claim',
        exact_values_and_indices_equal=True,execution=search.stats)


def assets_benchmark(a):
    from .runtime import extraction_record,compatible,host,extraction,raw_layers,reliable_precision
    from .execution import resident
    from .retrieval import exact_with_audit,target_nearest
    from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.fresh_inputs import Data
    if not a.recent_run:raise ValueError('assets smoke requires --recent-run correspondence parent')
    parent=ip.recent(a.recent_run[0])
    if parent['kind'] not in ('correspondence','reference_residual'):raise ValueError('assets smoke requires original raw asset schema')
    print('[residual smoke assets] validating parent/extraction identity; checkpoint and used payloads read once',flush=True)
    c=parent['catalog'];compatible(a,parent,c,extraction_record(a));prepared=st.sealed_json(parent['run']/'prepared.json')
    store=st.Store(100);bank,br=store.load(parent['cache']/'source_original')
    if prepared['source_seal']!=br['seal']:raise ValueError('parent original bank not frozen by prepared')
    def full(row):
        own=parent['kind']=='reference_residual'
        p=parent['cache']/('normal_features' if own else 'features')/ip.key(row)
        keys=tuple([f'q{l}' if own else f'full_L{l}' for l in LAYERS]+['positions' if own else 'full_positions'])
        x,r=store.load(p,keys)
        seal=prepared['normal_features'][row['relative_path']]['seal'] if own else prepared['features'][row['relative_path']]
        if seal!=r['seal'] or r['metadata']['image']!=row:raise ValueError('parent full feature identity differs')
        q={l:x[f'q{l}' if own else f'full_L{l}'] for l in LAYERS};g=dict(r['metadata']);g['positions']=x['positions' if own else 'full_positions']
        raw_layers({**{f'q{l}':q[l] for l in LAYERS},'positions':g['positions']},g)
        return q,g,r['seal']
    query,g,qseal=full(next(r for r in c['source'] if r['calibration']));search=resident(a);domains={}
    for domain in DOMAINS:
        print(f'[residual smoke assets] loading full-grid supports: {domain}',flush=True)
        parts={};seals=[]
        for row in c['supports'][domain]:
            q,_,seal=full(row);seals.append(seal)
            for l in LAYERS:parts.setdefault(l,[]).append(q[l])
        offsets=np.cumsum([0]+[len(x) for x in parts[LAYERS[0]]]).tolist();times=[]
        for repeat in range(2):
            source_seconds=0.;target_seconds=0.;query_seconds=0.
            for l in LAYERS:
                print(f'[residual smoke full NN] {domain}/L{l} repeat={repeat} start',flush=True)
                qt,seconds=timed(a.device,lambda:search.query(query[l]));query_seconds+=seconds;keys=[('target',seal,l) for seal in seals]
                if repeat==0 and l==7:
                    n=min(256,len(query[l]));small=query[l][:n];small_q=search.query(small)
                    old=exact_with_audit(small,bank[f'L{l}'],a.device)
                    new=exact_with_audit(small,bank[f'L{l}'],a.device,search=search,key=('source',br['seal'],l),query=small_q)
                    np.testing.assert_array_equal(old[0],new[0]);np.testing.assert_array_equal(old[1],new[1])
                    old=target_nearest(small,np.concatenate(parts[l]),offsets,a.device)
                    new=target_nearest(small,None,offsets,a.device,parts=parts[l],search=search,keys=keys,query=small_q)
                    np.testing.assert_array_equal(old[0],new[0]);np.testing.assert_array_equal(old[1],new[1]);del small_q
                _,seconds=timed(a.device,lambda:exact_with_audit(query[l],bank[f'L{l}'],a.device,search=search,key=('source',br['seal'],l),query=qt));source_seconds+=seconds
                _,seconds=timed(a.device,lambda:target_nearest(query[l],None,offsets,a.device,parts=parts[l],search=search,keys=keys,query=qt));target_seconds+=seconds
                del qt
            times.append(dict(Source_seconds=source_seconds,Target8_seconds=target_seconds,query_upload_seconds=query_seconds))
            print(f'[residual smoke full NN] {domain} repeat={repeat} {times[-1]}',flush=True)
        domains[domain]=dict(warm_Source_seconds=times[-1]['Source_seconds'],warm_Target8_seconds=times[-1]['Target8_seconds'],
            warm_query_upload_seconds=times[-1]['query_upload_seconds'],
            passes=times,query_rows=len(query[7]),bank_rows=len(bank['L7']),support_rows=offsets[-1],query_seal=qseal,support_seals=seals,
            prefix_original_vs_resident_values_and_indices_equal=True)
    result=dict(scope='real_frozen_full_grid_Source_calibration_query_NN_only_not_head_training',domains=domains,execution=dict(search.stats))
    search.cache.clear();search.bytes=0
    # Only clean Source training RGB: no calibration/test training or GT decoding.
    reliable_precision(a.device);model,_=host(a);forwards=[]
    with Data(a.data) as data:
        for row in ip.train_source(c)[:3]:
            image=data.image(row);(q,g),seconds=timed(a.device,lambda:extraction(model,image,a.device))
            raw_layers({**{f'q{l}':q[l] for l in LAYERS},'positions':np.asarray(g['positions'])},g)
            forwards.append(dict(relative_path=row['relative_path'],seconds=seconds,patches=len(q[7])))
            print(f'[residual smoke RGB] {row["relative_path"]} seconds={seconds:.3f}',flush=True)
    result.update(backbone_seconds_per_image=float(np.mean([r['seconds'] for r in forwards[1:]])),RGB_forwards=forwards,
        original_Source_seal=br['seal'],parent=str(parent['run']),parent_modified=False,GT_used=False)
    return result


def smoke(a):
    import torch,psutil
    from .runtime import reliable_precision
    reliable_precision(a.device);torch.set_num_threads(1)
    start=time.perf_counter();dest=a.run_root/'smoke'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');dest.mkdir(parents=True,exist_ok=False)
    result=dict(status='running',scope='engineering_smoke_not_formal_experiment',mode=a.smoke_mode,device=a.device,
        torch=torch.__version__,GT_used=False,formal_prediction_or_training_states_written=False,
        reference_cache_gib=a.reference_cache_gib,code={p.name:st.sha(p) for p in Path(__file__).parent.glob('*.py')})
    done=threading.Event()
    def heartbeat():
        while not done.wait(30):print(f'[heartbeat] stage=smoke mode={a.smoke_mode} seconds={time.perf_counter()-start:.0f}',flush=True)
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    try:
        cfg=ip.config(a);plan=ip.plan(a);inventory=ip.work_inventory(a)
        result.update(resolved_config=cfg,job_plan=plan)
        result['small_exact']=small_exact(a);assets={};head={}
        if a.smoke_mode=='prepared':
            from .runner import validate_runtime_inputs
            from .training_data import Dataset
            validate_runtime_inputs(a);jobs=[];shared=st.Store()
            for phase,family in dict.fromkeys((j['phase'],j['family']) for j in plan['training_jobs']):
                dataset=Dataset(a,phase,family!='NOREF_ST',store=shared)
                jobs.append(head_benchmark(a,dataset=dataset,phase=phase,family=family));del dataset
            head['prepared_jobs']=jobs
        else:
            job=head_benchmark(a);head['constructed_job']=job
            if job['formal_BF16_verified']:head['train_seconds_per_update']=job['seconds_per_update']
        if a.smoke_mode=='assets':assets=assets_benchmark(a)
        result.update(status='smoke_complete',head=head,assets=assets,time_estimate=estimate(head,assets,cfg,plan,inventory),
            pending=['real training-cache throughput: run prepared smoke after prepare-training'] if a.smoke_mode!='prepared' else [])
        return dict(status='engineering_smoke_complete_not_formal',report=str(dest/'smoke_report.json'),time_estimate=result['time_estimate'])
    except BaseException as error:
        result.update(status='smoke_failed',error=type(error).__name__+': '+str(error));raise
    finally:
        done.set();thread.join()
        result['seconds']=time.perf_counter()-start;info=psutil.Process().memory_info()
        result['process_current_rss_bytes']=info.rss;result['process_peak_rss_bytes']=getattr(info,'peak_wset',None)
        try:
            import resource
            result['process_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        except ImportError:pass
        if str(a.device).startswith('cuda'):
            result.update(GPU=torch.cuda.get_device_name(a.device),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(a.device),
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(a.device))
        st.write_json(dest/'smoke_report.json',result)
