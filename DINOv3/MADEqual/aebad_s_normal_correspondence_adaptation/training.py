"""Final-update normal-only training, immutable checkpoints and exact RNG resume."""
import time
from pathlib import Path
import numpy as np
from . import inputs as ip, storage as st
from .protocol import LAYERS, DOMAINS, SEEDS, CONFIG


def job_path(a,domain,seed,layer):return a.run_root/'models'/domain/f'seed{seed}'/f'L{layer}'


def load_checkpoint(path,expected,device):
    import torch
    p=Path(path);record=st.sealed_json(p/'latest.json');file=p/record['file']
    if record['inputs']!=expected or st.sha(file)!=record['sha256']:raise ValueError('checkpoint inputs/bytes changed')
    value=torch.load(file,map_location=device,weights_only=False)  # Only this run's sealed own checkpoint.
    if value['updates']!=record['updates']:raise ValueError('checkpoint update count changed')
    return value


def checkpoint(path,value,expected):
    import torch
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    old=st.read_json(path/'latest.json') if (path/'latest.json').exists() else None
    name=f"update_{value['updates']:06d}.pt";p=path/name;temp=p.with_suffix('.pending')
    torch.save(value,temp);temp.replace(p)
    record=dict(file=name,sha256=st.sha(p),updates=value['updates'],inputs=expected);record['seal']=st.digest(record)
    st.write_json(path/'latest.json',record)  # Atomic pointer commits all RNG/optimizer state together.
    if old and old['file']!=name:(path/old['file']).unlink(missing_ok=True)


class Pairs:
    def __init__(self,a,domain,kind,layer,store):
        self.store=store;self.layer=layer;self.a=a
        manifest=st.sealed_json(a.run_root/'matches'/domain/(kind+'.json'))
        if manifest['identity']!=ip.identity(a):raise ValueError('match identity differs')
        self.identity=manifest['seal'];self.rows=[r for r in manifest['records'] if any(r['anchors'])]
        if not self.rows:raise ValueError('empty eligible normal correspondence class: '+domain+'/'+kind)
        self.lookup={r['relative_path']:r for r in ip.training_rows(ip.catalog(a),None if domain=='source' else domain)}
    def batch(self,rng,n):
        row=self.rows[int(rng.integers(len(self.rows)))];available=[i for i,v in enumerate(row['anchors']) if v]
        direction=int(rng.choice(available));x,rec=self.store.load(row['path'])
        if rec['seal']!=row['seal']:raise ValueError('matching cache changed')
        left,right=(row['left'],row['right']) if direction==0 else (row['right'],row['left'])
        aa,ar=self.store.load(ip.feature_path(self.a,self.lookup[left]));bb,br=self.store.load(ip.feature_path(self.a,self.lookup[right]))
        expected=rec['metadata']
        if expected['left_seal']!=(ar if direction==0 else br)['seal'] or expected['right_seal']!=(br if direction==0 else ar)['seal']:
            raise ValueError('pair features changed after matching')
        pick=rng.integers(len(x['anchor_'+str(direction)]),size=n)
        anchor=x['anchor_'+str(direction)][pick];positive=x['positive_'+str(direction)][pick]
        weights=x['weight_'+str(direction)][pick];negative=x['negative_'+str(direction)][pick]
        values=aa[f'L{self.layer}'];other=bb[f'L{self.layer}']
        return np.array(values[anchor]),np.array(other[np.maximum(positive,0)]),np.array(weights),np.array(other[negative])


def train_job(a,domain,seed,layer,*,steps=None,hidden=None,batch_size=None,stop_after=None):
    """Private overrides exist for tiny CPU tests; CLI never changes the frozen budget."""
    import torch
    from .model import Adapter,paired_loss,lr_at
    from .runtime import reliable_precision,source_scales
    reliable_precision(a.device);store=st.Store()
    source,srec=store.load(a.cache_root/'source_original');bank=np.array(source[f'L{layer}'])
    mu,sigma=source_scales(bank);pretrain=domain=='source'
    steps=steps if steps is not None else CONFIG['source_steps' if pretrain else 'target_steps']
    hidden=hidden or CONFIG['hidden'];batch_size=batch_size or CONFIG['batch']
    classes=[Pairs(a,'source','SS',layer,store)]
    if not pretrain:classes += [Pairs(a,domain,k,layer,store) for k in ('ST','TT')]
    path=job_path(a,domain,seed,layer)
    parent=None
    if not pretrain:
        parent=st.sealed_json(job_path(a,'source',seed,layer)/'complete.json')
        if parent['updates']!=CONFIG['source_steps']:raise ValueError('complete Source pretraining required')
    expected=dict(identity=ip.identity(a),domain=domain,seed=seed,layer=layer,total_updates=steps,
        source_seal=srec['seal'],pairs=[p.identity for p in classes],hidden=hidden,batch=batch_size,
        Source_initialization=parent['checkpoint_sha256'] if parent else None)
    if (path/'complete.json').exists():
        done=st.sealed_json(path/'complete.json')
        if done['inputs']!=expected or done['updates']!=steps:raise ValueError('training configuration changed')
        load_checkpoint(path,expected,a.device)
        if st.sealed_json(path/'latest.json')['sha256']!=done['checkpoint_sha256']:raise ValueError('final checkpoint changed')
        return dict(status='training_reused',domain=domain,seed=seed,layer=layer)
    initial=int(st.seed('network',seed,layer)%(2**31))
    torch.random.default_generator.manual_seed(initial)
    if str(a.device).startswith('cuda'):
        with torch.cuda.device(a.device):torch.cuda.manual_seed(initial)
    net=Adapter(mu,sigma,hidden).to(a.device)
    if parent:
        old=load_checkpoint(job_path(a,'source',seed,layer),parent['inputs'],a.device)
        net.load_state_dict(old['network'])
    optimizer=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=1e-4)
    rng=st.rng('training',seed,domain,layer);update=0;history=[];elapsed_before=0.
    if (path/'latest.json').exists():
        state=load_checkpoint(path,expected,a.device);net.load_state_dict(state['network']);optimizer.load_state_dict(state['optimizer'])
        update=state['updates'];rng.bit_generator.state=state['numpy_rng'];torch.set_rng_state(state['torch_rng'].cpu())
        if str(a.device).startswith('cuda'):torch.cuda.set_rng_state(state['cuda_rng'].cpu(),device=a.device)
        history=state['history'];elapsed_before=state['seconds']
    # Target identity draws are uniform over all rows, not uniform over images.
    target=None
    if not pretrain:
        rows=ip.catalog(a)['supports'][domain]
        target=np.concatenate([store.load(ip.feature_path(a,r))[0][f'full_L{layer}'] for r in rows])
    tick=time.perf_counter();limit=steps if stop_after is None else min(steps,stop_after)
    def save():
        checkpoint(path,dict(network=net.state_dict(),optimizer=optimizer.state_dict(),updates=update,
            numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(a.device) if str(a.device).startswith('cuda') else None,
            scheduler=dict(completed_updates=update,total=steps,definition='linear5pct_cosine_to_1e-6'),
            history=history,seconds=elapsed_before+time.perf_counter()-tick),expected)
    while update<limit:
        optimizer.zero_grad(set_to_none=True);losses=[]
        extra=np.concatenate([bank[rng.integers(len(bank),size=512)]]) if pretrain else np.concatenate((
            bank[rng.integers(len(bank),size=256)],target[rng.integers(len(target),size=256)]))
        extra=torch.as_tensor(extra,device=a.device,dtype=torch.float32)
        for i,pairs in enumerate(classes):
            # Rotate the remainder; exactly 256 anchors/update and approximately equal classes.
            n=batch_size//len(classes)+int((i-update)%len(classes)<batch_size%len(classes))
            tensors=[torch.as_tensor(x,device=a.device,dtype=torch.float32) for x in pairs.batch(rng,n)]
            pos,neg,ident=paired_loss(net,*tensors,extra)
            loss=(pos+neg+.1*ident)/len(classes)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite normal training loss; checkpoint preserved')
            loss.backward();losses.append([float(pos.detach()),float(neg.detach()),float(ident.detach())])
        grad=torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
        for group in optimizer.param_groups:group['lr']=lr_at(update,steps)
        optimizer.step();update+=1
        if update%100==0 or update==limit:
            with torch.no_grad():
                probe=torch.as_tensor(bank[:min(256,len(bank))],device=a.device)
                transformed=net(probe)
                if not torch.isfinite(transformed).all() or transformed.var(0,unbiased=False).sum()<=0:
                    raise FloatingPointError('nonfinite/collapsed representation; previous checkpoint preserved')
            history.append(dict(update=update,losses=losses,gradient_norm=float(grad),lr=optimizer.param_groups[0]['lr'],
                seconds=elapsed_before+time.perf_counter()-tick,
                displacement_RMS=float((transformed-probe).square().mean().sqrt())))
            save();print(f'[normal train] {domain}/seed{seed}/L{layer} {update}/{steps} '
                f'seconds={history[-1]["seconds"]:.1f}',flush=True)
    if update==steps:
        record=st.sealed_json(path/'latest.json')
        st.sealed_json(path/'complete.json',dict(inputs=expected,updates=steps,checkpoint_sha256=record['sha256'],
            checkpoint_bytes=(path/record['file']).stat().st_size,parameter_count=sum(p.numel() for p in net.parameters()),
            seconds=elapsed_before+time.perf_counter()-tick,final_weights_not_metric_selected=True))
        st.write_csv(path/'training_log.csv',history)
    return dict(status='training_complete' if update==steps else 'training_checkpointed',updates=update,domain=domain,seed=seed,layer=layer)


def train(a,source=False):
    jobs=[]
    for seed in SEEDS:
        for domain in (('source',) if source else DOMAINS):
            for layer in LAYERS:jobs.append(train_job(a,domain,seed,layer))
    return dict(status='requested_training_complete',jobs=jobs)


def trained(a,domain,seed,layer,device):
    from .model import Adapter
    p=job_path(a,domain,seed,layer);done=st.sealed_json(p/'complete.json')
    if done['inputs']['identity']!=ip.identity(a) or done['updates']!=CONFIG['target_steps' if domain!='source' else 'source_steps']:
        raise ValueError('unfrozen/incomplete trained model')
    value=load_checkpoint(p,done['inputs'],device);s=value['network']
    if st.sealed_json(p/'latest.json')['sha256']!=done['checkpoint_sha256']:raise ValueError('frozen final checkpoint changed')
    net=Adapter(s['mu'].cpu().numpy(),float(s['sigma']),done['inputs']['hidden']).to(device)
    net.load_state_dict(s);net.eval();return net,done
