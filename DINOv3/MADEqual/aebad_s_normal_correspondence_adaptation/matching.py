"""Frozen teacher, cross-image mutual Top3 and deterministic low-sim negatives."""
import math
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter
from . import storage as st, inputs as ip
from .protocol import CONFIG, LAYERS, DOMAINS, VERSION


def normalize(x):
    x=np.asarray(x,np.float32)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),np.float32(1e-12))


def teacher(layers,positions,grid,relative):
    """Full grid context BEFORE shared layer position sampling. Physical tie IDs sorted."""
    positions=np.asarray(positions,np.int64);n=int(np.prod(grid))
    if not np.array_equal(np.sort(positions),np.arange(n)):raise ValueError('complete physical feature grid required')
    chosen=np.sort(st.rng(VERSION,'positions',CONFIG['match_seed'],relative).choice(n,min(n,CONFIG['match_limit']),replace=False))
    inverse=np.argsort(positions);descriptors=[];global_parts=[];sampled={}
    count=uniform_filter(np.ones(grid,np.float32),size=3,mode='constant',cval=0.)
    for l in LAYERS:
        raw=np.asarray(layers[l],np.float32)[inverse];e=normalize(raw)
        context=uniform_filter(e.reshape(*grid,-1),size=(3,3,1),mode='constant',cval=0.)/count[:,:,None]
        context=normalize(context.reshape(n,-1))
        descriptors.extend((e[chosen],context[chosen]))
        global_parts.append(normalize(raw.mean(0,dtype=np.float32)))
        sampled['L'+str(l)]=raw[chosen].copy()
    sampled.update(positions=chosen,teacher=np.concatenate(descriptors,1).astype(np.float32)/np.float32(math.sqrt(8)),
        global_descriptor=normalize(np.concatenate(global_parts)))
    return sampled


def image_pairs(rows,descriptors,kind):
    """Canonical unordered image pairs; each random draw excludes all nearest picks."""
    ordered=sorted(rows,key=lambda r:r['relative_path']);pairs=set()
    source=[r for r in ordered if r.get('source_index') is not None]
    target=[r for r in ordered if r.get('support_index') is not None]
    def select(anchor,candidates,nclose,nrandom):
        names=sorted(r['relative_path'] for r in candidates if r['relative_path']!=anchor['relative_path'])
        score=np.array([np.dot(descriptors[anchor['relative_path']],descriptors[p]) for p in names])
        order=np.argsort(-score,kind='stable');close=[names[i] for i in order[:nclose]]
        remain=[p for p in names if p not in close]
        draw=st.rng(VERSION,'image_pairs',CONFIG['match_seed'],kind,anchor['relative_path']).choice(
            len(remain),min(nrandom,len(remain)),replace=False)
        for p in close+[remain[i] for i in draw]:pairs.add(tuple(sorted((anchor['relative_path'],p))))
    if kind=='SS':
        for r in source:select(r,source,2,2)
    elif kind=='ST':
        for r in target:select(r,source,16,4)
    elif kind=='TT':
        for i,r in enumerate(target):
            for other in target[i+1:]:pairs.add((r['relative_path'],other['relative_path']))
    else:raise ValueError('unknown pair class')
    return sorted(pairs)


def correspond(left,right,seed_parts,device='cpu',block=256):
    """Torch production similarities, bounded row blocks; no full matrix persistence.

    Teacher rows MUST already be sorted by physical patch index. Stable GPU sort
    supplies the tie rule. Bottom floor(N/2) is the predeclared low half.
    """
    import torch
    a=torch.as_tensor(np.array(left,copy=True),device=device,dtype=torch.float32)
    b=torch.as_tensor(np.array(right,copy=True),device=device,dtype=torch.float32)
    if min(len(a),len(b))<8:raise ValueError('too few sampled positions for four low-half negatives')
    def top(x,y):
        out=[]
        for start in range(0,len(x),block):
            sim=x[start:start+block]@y.T
            out.append(torch.argsort(sim,dim=1,descending=True,stable=True)[:,:3].cpu().numpy())
        return np.concatenate(out)
    with torch.inference_mode():
        ab,ba=top(a,b),top(b,a);results=[]
        for direction,(x,y,forward,reverse) in enumerate(((a,b,ab,ba),(b,a,ba,ab))):
            anchors=[];positives=[];weights=[];negatives=[]
            for start in range(0,len(x),block):
                sim=(x[start:start+block]@y.T).cpu().numpy()
                for j,values in enumerate(sim):
                    i=start+j;pos=[int(v) for v in forward[i] if i in reverse[v]]
                    if not pos:continue
                    low=np.argsort(values,kind='stable')[:len(values)//2]
                    low=low[~np.isin(low,pos)]
                    if len(low)<4:raise ValueError('fewer than four eligible low-half negatives')
                    neg=st.rng(VERSION,'negative',*seed_parts,direction,i).choice(low,4,replace=False)
                    w=values[pos]/np.float32(.07);w=np.exp(w-w.max()).astype(np.float32);w/=w.sum(dtype=np.float32)
                    anchors.append(i);positives.append(pos+[-1]*(3-len(pos)))
                    weights.append(w.tolist()+[0.]*(3-len(pos)));negatives.append(neg)
            results.append(dict(anchor=np.asarray(anchors,np.int64),positive=np.asarray(positives,np.int64).reshape(-1,3),
                weight=np.asarray(weights,np.float32).reshape(-1,3),negative=np.asarray(negatives,np.int64).reshape(-1,4)))
    return results


def match(a):
    from .runtime import reliable_precision
    reliable_precision(a.device)
    c=ip.catalog(a);store=st.Store();source=ip.training_rows(c)
    # One SS list/cache shared by all domains and all training seeds.
    manifests={};all_rows={r['relative_path']:r for r in source+sum(c['supports'].values(),[])}
    desc={p:store.load(ip.feature_path(a,r))[0]['global_descriptor'].copy() for p,r in all_rows.items()}
    groups=[('source','SS',source)]+[(d,k,ip.training_rows(c,d)) for d in DOMAINS for k in ('ST','TT')]
    for domain,kind,rows in groups:
        ip.validate_training(c,None if domain=='source' else domain,rows)
        records=[]
        for number,(left,right) in enumerate(image_pairs(rows,desc,kind),1):
            key=st.digest([kind,left,right])[:28];path=a.cache_root/'matches'/domain/kind/key
            lx,lr=store.load(ip.feature_path(a,all_rows[left]));rx,rr=store.load(ip.feature_path(a,all_rows[right]))
            meta=dict(identity=ip.identity(a),left=left,right=right,kind=kind,domain=domain,
                left_seal=lr['seal'],right_seal=rr['seal'],GT_used=False)
            if not (path/'manifest.json').exists():
                found=correspond(lx['teacher'],rx['teacher'],(domain,kind,left,right),a.device)
                arrays={f'{k}_{i}':v for i,direction in enumerate(found) for k,v in direction.items()}
                st.put(path,arrays,meta)
            arrays,rec=st.get(path)
            if rec['metadata']!=meta:raise ValueError('match teacher/dependencies changed')
            counts=[len(arrays['anchor_'+str(i)]) for i in (0,1)]
            records.append(dict(path=str(path),seal=rec['seal'],left=left,right=right,anchors=counts))
            if number%10==0:print(f'[normal match] {domain}/{kind} {number}',flush=True)
        # Empty image pairs are retained in manifest; sampling uses nonempty pairs only.
        st.sealed_json(a.run_root/'matches'/domain/(kind+'.json'),dict(identity=ip.identity(a),records=records,
            empty_image_pairs=sum(not any(r['anchors']) for r in records)))
        manifests[domain+'/'+kind]=len(records)
    return dict(status='normal_matches_frozen',pairs=manifests)
