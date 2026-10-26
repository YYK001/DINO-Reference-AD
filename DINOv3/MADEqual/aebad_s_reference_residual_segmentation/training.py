"""Image-balanced local head training with deterministic prefetch and sealed resume."""
from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path
import time
import numpy as np
from . import inputs as ip,storage as st
from .protocol import CONFIG,DOMAINS
from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.training import checkpoint,load_checkpoint


def path(a,phase,family,seed):return a.run_root/'models'/phase/family/f'seed{seed}'
def source_family(family):return 'NOREF_ST' if family=='NOREF_ST' else 'REF_ST'


def rate(update,total):
    warm=max(1,int(CONFIG['warmup_fraction']*total))
    if update<warm:return CONFIG['learning_rate']*(update+1)/warm
    return CONFIG['minimum_lr']+.5*(CONFIG['learning_rate']-CONFIG['minimum_lr'])*(1+math.cos(math.pi*(update-warm)/max(1,total-warm-1)))


def prepare_batch(samples,*,pin=False,grouped=True):
    """Exact shape groups: no padding, changed tiles or changed image weights."""
    import torch
    from .tiles import output_window
    groups={}
    for i,sample in enumerate(samples):
        key=tuple(tuple(x.shape[1:]) for x in sample['values']) if grouped else (i,)
        groups.setdefault(key,[]).append(sample)
    result=[]
    for rows in groups.values():
        values=[torch.cat([r['values'][l] for r in rows],0) for l in range(4)];labels=[]
        for row in rows:
            y,x,yy,xx=output_window(row['center'],row['grid'],row['shape'])
            labels.append(torch.from_numpy(np.array(row['soft'][y:yy,x:xx],dtype=np.float32,copy=True))[None,None])
        if pin:values=[v.pin_memory() for v in values];labels=[v.pin_memory() for v in labels]
        result.append(dict(rows=rows,values=values,labels=labels,weight=1/len(samples)))
    return result


def backward_batch(net,groups,device,*,bf16=True):
    """Batch the head only; retain separate global interpolation and loss per image."""
    import torch
    from .model import loss
    from .tiles import interpolate_center
    combined=None
    for group in groups:
        values=[v.to(device,non_blocking=True) for v in group['values']]
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=bf16 and str(device).startswith('cuda')):logits=net(values)
        losses=[]
        for i,(sample,label) in enumerate(zip(group['rows'],group['labels'])):
            interpolated,_=interpolate_center(logits[i:i+1],sample['origin'],sample['grid'],sample['shape'],sample['center'])
            losses.append(loss(interpolated.float(),label.to(device,non_blocking=True))*group['weight'])
        value=torch.stack(losses).sum();value.backward()
        combined=value.detach() if combined is None else combined+value.detach()
    return combined


def job(a,phase,family,seed,*,dataset=None,head=None,total=None,stop_after=None):
    import torch
    from .model import Head,loss
    from .tiles import interpolate_center
    from .training_data import Dataset
    from .runtime import reliable_precision
    reliable_precision(a.device);torch.backends.cudnn.allow_tf32=False;pretrain=phase=='source'
    cfg=ip.config(a);requested=ip.plan(a)['training_jobs']
    if not any((j['phase'],j['family'],j['seed'])==(phase,family,seed) for j in requested):raise ValueError('training job not requested by frozen plan')
    total=total or cfg['source_updates' if pretrain else 'target_updates']
    dataset=dataset or Dataset(a,phase,use_reference=family!='NOREF_ST')
    if dataset.use_reference!=(family!='NOREF_ST'):raise ValueError('NoRef data access contract differs')
    parent=None
    if not pretrain:parent=st.sealed_json(path(a,'source',source_family(family),seed)/'complete.json')
    expected=dict(identity=ip.identity(a),phase=phase,family=family,seed=seed,total_updates=total,
        training_manifest=dataset.manifest_seal,Source_initialization=parent['checkpoint_sha256'] if parent else None,
        precision=CONFIG['precision'],execution='shape_grouped_head_per_image_loss_v1',
        sample_schedule='PCG64(hash(protocol,seed,phase,update)); family-independent_ST_schedule')
    p=path(a,phase,family,seed)
    if (p/'complete.json').exists():
        done=st.sealed_json(p/'complete.json');record=st.sealed_json(p/'latest.json')
        if done['inputs']!=expected or done['updates']!=total or record['sha256']!=done['checkpoint_sha256']:raise ValueError('final head checkpoint differs')
        load_checkpoint(p,expected,a.device);return dict(status='training_reused',phase=phase,family=family,seed=seed)
    init=int(st.seed('residual_head_initialization',seed)%2**31);torch.random.default_generator.manual_seed(init)
    if str(a.device).startswith('cuda'):
        with torch.cuda.device(a.device):torch.cuda.manual_seed(init)
    net=(head or Head()).to(a.device);net.train()
    if parent:
        if parent['updates']!=cfg['source_updates']:raise ValueError('unfinished Source pretraining')
        old=load_checkpoint(path(a,'source',source_family(family),seed),parent['inputs'],a.device);net.load_state_dict(old['network'])
    optimizer=torch.optim.AdamW(net.parameters(),lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'])
    update=0;history=[];elapsed=0.
    if (p/'latest.json').exists():
        r=load_checkpoint(p,expected,a.device);net.load_state_dict(r['network']);optimizer.load_state_dict(r['optimizer'])
        update=r['updates'];history=r['history'];elapsed=r['seconds'];torch.set_rng_state(r['torch_rng'].cpu())
        if str(a.device).startswith('cuda'):torch.cuda.set_rng_state(r['cuda_rng'].cpu(),a.device)
    started=time.perf_counter();end=total if stop_after is None else min(total,stop_after)
    def batch(number):
        choices=dataset.select(seed,number,source_queries_only=family=='REF_SOURCE_QUERY')
        result=[dataset.load(c) for c in choices]
        return prepare_batch(result,pin=str(a.device).startswith('cuda'))
    def save():
        checkpoint(p,dict(network=net.state_dict(),optimizer=optimizer.state_dict(),updates=update,history=history,
            seconds=elapsed+time.perf_counter()-started,torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(a.device) if str(a.device).startswith('cuda') else None,
            numpy_rng=st.rng('head_sampler',seed,phase,update).bit_generator.state,
            scheduler=dict(completed_updates=update,total=total,definition='5pct_linear_then_cosine_to_1e-6')),expected)
    # CPU cache work overlaps GPU. Data selection depends on update, not thread scheduling.
    waited=0.
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(batch,update) if update<end else None
        while update<end:
            tick=time.perf_counter();samples=future.result();waited+=time.perf_counter()-tick
            if update+1<end:future=pool.submit(batch,update+1)
            optimizer.zero_grad(set_to_none=True);combined=backward_batch(net,samples,a.device)
            # Required finite error detection; avoid per-class/per-loss Python scalar reads.
            grad=torch.nn.utils.clip_grad_norm_(net.parameters(),CONFIG['gradient_clip'],error_if_nonfinite=False)
            if not torch.isfinite(torch.stack([combined,grad.detach()])).all():raise FloatingPointError('nonfinite head loss/gradient; checkpoint preserved')
            for g in optimizer.param_groups:g['lr']=rate(update,total)
            optimizer.step();update+=1
            if update%CONFIG['log_every']==0 or update==end:
                values=torch.stack([combined,grad.detach()]).cpu().tolist()
                history.append(dict(update=update,loss=values[0],gradient_norm=values[1],lr=optimizer.param_groups[0]['lr'],seconds=elapsed+time.perf_counter()-started,
                    data_wait_seconds_current_session=waited))
                print(f'[residual train] {phase}/{family}/seed{seed} {update}/{total} seconds={history[-1]["seconds"]:.1f}',flush=True)
            if update%CONFIG['checkpoint_every']==0 or update==end:save()
    if update==total:
        latest=st.sealed_json(p/'latest.json')
        st.sealed_json(p/'complete.json',dict(inputs=expected,updates=total,checkpoint_sha256=latest['sha256'],
            parameter_count=sum(v.numel() for v in net.parameters()),checkpoint_bytes=(p/latest['file']).stat().st_size,
            seconds=elapsed+time.perf_counter()-started,selected_by='fixed_final_update_no_test_metrics'))
        st.write_csv(p/'training_log.csv',history)
    return dict(status='training_complete' if update==total else 'checkpointed',updates=update,phase=phase,family=family,seed=seed)


def train(a,source=False):
    records=[]
    # One verified cache session per phase, shared across seeds/families; no repeated
    # whole-feature SHA scan for every model. NoRef has a separate no-reference loader.
    from .training_data import Dataset
    shared=st.Store()
    for phase in (('source',) if source else DOMAINS):
        jobs=[j for j in ip.plan(a)['training_jobs'] if j['phase']==phase];data={}
        for item in jobs:
            use=item['family']!='NOREF_ST'
            if use not in data:data[use]=Dataset(a,phase,use,store=shared)
            records.append(job(a,phase,item['family'],item['seed'],dataset=data[use]))
    return dict(status='requested_heads_trained',jobs=records)


def trained(a,domain,family,seed):
    from .model import Head
    p=path(a,domain,family,seed);done=st.sealed_json(p/'complete.json');latest=st.sealed_json(p/'latest.json')
    cfg=ip.config(a)
    if not any((j['phase'],j['family'],j['seed'])==(domain,family,seed) for j in ip.plan(a)['training_jobs']):raise ValueError('model not requested')
    if done['inputs']['identity']!=ip.identity(a) or done['updates']!=cfg['target_updates'] or latest['sha256']!=done['checkpoint_sha256']:
        raise ValueError('incomplete/modified final head')
    x=load_checkpoint(p,done['inputs'],a.device);net=Head().to(a.device);net.load_state_dict(x['network']);net.eval();return net,done
