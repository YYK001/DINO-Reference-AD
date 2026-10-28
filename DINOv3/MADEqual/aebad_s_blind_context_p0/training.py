"""Cached whole-image-excluded candidates; four small heads, 1500 joint updates."""
import time
import numpy as np
from . import inputs as ip,context as cx,runtime as rt
from .inputs import st
from .protocol import VERSION,LAYERS,CONFIG


def require_isolation(a,trained=False):
    r=st.sealed_json(a.run_root/'isolation.json')
    if r['identity']!=ip.identity(a) or not r['passed']:raise ValueError('input isolation has not passed')
    if trained:
        saved=st.sealed_json(a.run_root/'trained.json')
        if r['model_seal']!=saved['seal']:raise ValueError('trained heads require a fresh isolation check before prediction')
    return r


def prepare(a):
    require_isolation(a);ready=st.sealed_json(a.run_root/'context_ready.json')
    if ready['identity']!=ip.identity(a):raise ValueError('dictionary configuration changed')
    from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.fresh_inputs import Data
    from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.runtime import reliable_precision,source_scales
    reliable_precision(a.device);rows=ip.dictionary_rows(a);records=[]
    for l in LAYERS:
        dest=a.cache_root/'training'/f'L{l}'
        if (dest/'manifest.json').exists():
            _,r=st.get(dest)
            if r['metadata']['identity']!=ip.identity(a) or r['metadata']['context_seal']!=ready['seal']:raise ValueError('training cache identity changed')
            records.append(r['seal']);continue
        d,dr=rt.dictionary(a,l);index=rt.ContextIndex(d,a.device);queries=[];targets=[];images=[];positions=[];ids=[]
        with Data(a.data) as data:
            for image_id,row in enumerate(rows):
                c,cr=st.get(ip.context_path(a,row),(f'c{l}','positions'));x,xr,_=rt.original(a,row,data)
                cap=1024 if 'support_index' in row else 512
                p=cx.select_positions(c['positions'],cap,row['relative_path'],'training')
                inv=np.argsort(c['positions']);qi=np.searchsorted(c['positions'][inv],p);q=np.array(c[f'c{l}'][inv[qi]],np.float32)
                candidate=index.select(q,excluded=image_id)
                if ((candidate>=0)&(d['images'][np.maximum(candidate,0)]==image_id)).any():raise ValueError('same image leaked into training candidates')
                n=np.sum(candidate>=0,axis=1)
                if not np.all(n==(11 if 'support_index' in row else 12)):raise ValueError('whole-image Source or K7 Target exclusion failed')
                queries.append(q);targets.append(np.array(x[f'q{l}'][np.argsort(x['positions'])[p]],np.float32));images.extend([image_id]*len(p));positions.extend(p);ids.append(candidate)
                print(f'[blind training cache] L{l} {image_id+1}/72',flush=True)
        # Fixed scalar feature RMS from the selected Source normal training positions only.
        q=np.concatenate(targets);im=np.asarray(images,np.int64);_,sigma=source_scales(q[im<64])
        r=st.put(dest,dict(context=np.concatenate(queries),target=q,images=im,positions=np.asarray(positions,np.int64),candidates=np.concatenate(ids)),
                 dict(identity=ip.identity(a),layer=l,dictionary_seal=dr['seal'],context_seal=ready['seal'],
                      loss_scale=float(sigma),scale_population='64 Source normal sampled original positions only',
                      query_target_role='supervision only; never candidate selection',same_image_excluded=True))
        records.append(r['seal'])
    return st.sealed_json(a.run_root/'training_ready.json',dict(identity=ip.identity(a),layers=list(LAYERS),training_seals=records,updates=1500))


def make_heads(device):
    """32776 parameters total. No backbone, global token or original q in forward."""
    import torch
    class Head(torch.nn.Module):
        def __init__(self,layer):
            super().__init__();r=cx.initial_head(layer)
            self.Wq=torch.nn.Parameter(torch.tensor(r['Wq'],device=device));self.Wk=torch.nn.Parameter(torch.tensor(r['Wk'],device=device));self.bias=torch.nn.Parameter(torch.tensor(r['bias'],device=device))
        def forward(self,context,candidate_context,types,valid,values):
            q=torch.nn.functional.normalize(context@self.Wq,dim=-1,eps=1e-12)
            k=torch.nn.functional.normalize(candidate_context@self.Wk,dim=-1,eps=1e-12)
            logits=(q[:,None]*k).sum(-1)/.1+self.bias[types]
            logits=logits.masked_fill(~valid,-torch.inf);weights=torch.softmax(logits,dim=1)
            return (weights[:,:,None]*values).sum(1),weights
    return torch.nn.ModuleDict({str(l):Head(l) for l in LAYERS})


def batch_rows(image_ids,step):
    """Image equal, Source/Target 64/64, common physical rows for all four layers."""
    rng=st.rng(VERSION,42,'batch',step);chosen=[]
    for population in (range(64),range(64,72)):
        selected=rng.choice(list(population),64,replace=True)
        for image in selected:chosen.append(rng.choice(np.flatnonzero(image_ids==image)))
    return np.asarray(chosen,np.int64)


def trained(a):
    if hasattr(a,'_trained_model'):return a._trained_model
    r=st.sealed_json(a.run_root/'trained.json')
    if r['identity']!=ip.identity(a) or r['updates']!=1500 or r['training_seal']!=st.sealed_json(a.run_root/'training_ready.json')['seal']:raise ValueError('incomplete or changed training')
    x,xr=st.get(a.run_root/'model')
    if xr['seal']!=r['model_bundle_seal']:raise ValueError('model bundle changed')
    a._trained_model=({l:{k:np.array(x[f'L{l}_{k}']) for k in ('Wq','Wk','bias')} for l in LAYERS},r)
    return a._trained_model


def train(a):
    require_isolation(a);ready=st.sealed_json(a.run_root/'training_ready.json')
    if (a.run_root/'trained.json').exists():return dict(status='last_step_model_reused',**trained(a)[1])
    import torch
    from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.runtime import reliable_precision
    reliable_precision(a.device);torch.manual_seed(42);heads=make_heads(a.device)
    optimizer=torch.optim.AdamW(heads.parameters(),lr=1e-3,weight_decay=1e-4)
    pointer=a.run_root/'training_checkpoint.json';start=0;elapsed=0.;logs=[]
    if pointer.exists():
        seal=st.sealed_json(pointer);checkpoint=a.run_root/seal['file']
        if checkpoint.parent!=a.run_root or seal['identity']!=ip.identity(a) or seal['training_seal']!=ready['seal'] or st.sha(checkpoint)!=seal['sha256']:raise ValueError('checkpoint/sampling dependencies changed')
        saved=torch.load(checkpoint,map_location=a.device,weights_only=True);heads.load_state_dict(saved['heads']);optimizer.load_state_dict(saved['optimizer']);start=saved['step'];elapsed=saved['seconds']
        if start>1500:raise ValueError('checkpoint beyond fixed schedule')
    resident={};image_ids=None;physical=None
    for l,expected in zip(LAYERS,ready['training_seals']):
        d,dr=rt.dictionary(a,l);t,tr=st.get(a.cache_root/'training'/f'L{l}')
        if tr['seal']!=expected or tr['metadata']['dictionary_seal']!=dr['seal']:raise ValueError('cached candidate/model inputs changed')
        if image_ids is not None and (not np.array_equal(t['images'],image_ids) or not np.array_equal(t['positions'],physical)):raise ValueError('four-layer physical training rows differ')
        image_ids=t['images'];physical=t['positions']
        resident[l]={k:torch.tensor(np.asarray(d[k]),device=a.device) for k in ('keys','values','types')}
        resident[l].update({k:torch.tensor(np.asarray(t[k]),device=a.device) for k in ('context','target','candidates')})
        resident[l]['scale']=tr['metadata']['loss_scale']
    tick=time.perf_counter()
    for step in range(start,1500):
        ids=torch.tensor(batch_rows(image_ids,step),device=a.device);optimizer.zero_grad(set_to_none=True);losses=[]
        for l in LAYERS:
            d=resident[l];ci=d['candidates'][ids];valid=ci>=0;ci=ci.clamp_min(0)
            synthesized,_=heads[str(l)](d['context'][ids],d['keys'][ci],d['types'][ci],valid,d['values'][ci])
            # Equal 64 Source /64 Target and equal four layers; original q only here.
            loss=((synthesized-d['target'][ids])**2).mean()/float(d['scale']**2);losses.append(loss)
        total=torch.stack(losses).mean();total.backward();optimizer.step()
        if (step+1)%100==0 or step+1==1500:
            value=float(total.detach().cpu())
            if not np.isfinite(value):raise ValueError('nonfinite training loss')
            seconds=elapsed+time.perf_counter()-tick;logs.append(dict(update=step+1,loss=value,seconds=seconds))
            checkpoint=a.run_root/f'training_checkpoint_{step+1:04d}.pt'
            tmp=checkpoint.with_suffix('.pending');torch.save(dict(heads=heads.state_dict(),optimizer=optimizer.state_dict(),step=step+1,seconds=seconds),tmp);tmp.replace(checkpoint)
            record=dict(identity=ip.identity(a),training_seal=ready['seal'],sha256=st.sha(checkpoint),file=checkpoint.name);record['seal']=st.digest(record)
            # Commit pointer last; interruption while writing a new generation leaves the last valid checkpoint intact.
            st.write_json(pointer,record)
            st.write_csv(a.run_root/f'training_session_{start}.csv',logs)
            print(f'[blind train] {step+1}/1500 loss={value:.6g} seconds={seconds:.1f}',flush=True)
    arrays={f'L{l}_{k}':v.detach().cpu().numpy() for l in LAYERS for k,v in heads[str(l)].named_parameters()}
    m=st.put(a.run_root/'model',arrays,dict(identity=ip.identity(a),training_seal=ready['seal'],updates=1500))
    return st.sealed_json(a.run_root/'trained.json',dict(identity=ip.identity(a),updates=1500,seed=42,parameters=sum(p.numel() for p in heads.parameters()),
        training_seal=ready['seal'],model_bundle_seal=m['seal'],seconds=elapsed+time.perf_counter()-tick,
        precision='FP32 head/loss; original backbone frozen; no per-step device synchronization'))
