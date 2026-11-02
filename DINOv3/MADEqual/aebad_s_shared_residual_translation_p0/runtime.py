"""Original full features and exact Top1; single device, lazy Torch imports."""
from pathlib import Path
import numpy as np
from . import inputs as ip, numerics as nm
from .inputs import st
from .protocol import *


def features(a,row,persist=False):
    # The original adapter checks complete extraction/provider identity, does at
    # most one full forward, and never loads a trained head or transformed bank.
    from DINOv3.MADEqual.aebad_s_joint_reference_response_p0.runtime import features as original
    return original(a,row,persist=persist)


class References:
    def __init__(self,a,d):
        from DINOv3.MADEqual.aebad_s_reference_residual_segmentation.execution import ResidentExact
        from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.runtime import reliable_precision
        reliable_precision(a.device)
        self.a=a;self.d=d;self.search=ResidentExact(a.device,int(a.reference_cache_gib*2**30))
        self.store=st.Store(12);self.source=ip.source_ref(a);self.targets=[]
        for row in ip.parent(a).supports(d):
            _,_,ref=features(a,row,persist=True)
            self.targets.append(dict(path=ref['path'],seal=ref['seal']))

    def banks(self,l):
        x,r=self.store.load(self.source['path'],(f'L{l}',))
        if r['seal']!=self.source['seal']:raise ValueError('Source changed')
        banks=[x[f'L{l}']];keys=[('Source',r['seal'],l)]
        for ref in self.targets:
            x,r=self.store.load(ref['path'],(f'q{l}',))
            if r['seal']!=ref['seal']:raise ValueError('support changed')
            banks.append(x[f'q{l}']);keys.append(('Target',r['seal'],l))
        return banks,keys

    def nearest(self,q,geometry,feature,row):
        wanted=ip.contract(self.a,self.d,row,feature,self.targets)
        ref=ip.nn_ref(self.a,self.d,row,wanted)
        if ref:
            x,r=st.get(ref['path']);return x,r,True
        if not self.a.supplement_retrieval:raise FileNotFoundError('nine-source indices absent; explicit --supplement-retrieval required')
        values={'positions':np.asarray(geometry['positions'],np.int64)}
        for l in LAYERS:
            banks,keys=self.banks(l);qt=self.search.query(q[l]);dist=[];ids=[]
            for bank,key in zip(banks,keys):
                dd,ii=self.search.exact(qt,bank,key);dist.append(dd);ids.append(ii)
            values[f'distance{l}']=np.stack(dist);values[f'index{l}']=np.stack(ids)
        path=ip.nn_path(self.a,self.d,row)
        r=st.put(path,values,dict(contract=wanted,grid=geometry['grid'],GT_used=False))
        return values,r,False


def translated_gpu(q,banks,indices,distances,search,keys,chunk):
    """Stock Torch FP32; small CPU diagnostics, no custom kernel/autocast/TF32.

    Physical order fixed BEFORE reductions. Reductions are whole-image, never
    independent block means. Gather blocks only limit peak temporary storage.
    """
    import torch
    with torch.inference_mode():
        qt=search.query(q);n,dim=q.shape
        ids=torch.tensor(indices,device=qt.device,dtype=torch.int64)
        ds=torch.tensor(distances,device=qt.device,dtype=torch.float32)
        win=ds.argmin(0);ru=torch.empty_like(qt);means=[];rnorm=[]
        zero_min=torch.full((n,),torch.inf,device=qt.device)
        def residual(s):
            bank=search.reference(banks[s],keys[s]);r=torch.empty_like(qt)
            for start in range(0,n,chunk):
                end=min(n,start+chunk);r[start:end]=qt[start:end]-bank[ids[s,start:end]]
            return r
        for s in range(len(banks)):
            r=residual(s);means.append(r.mean(0));length=torch.linalg.vector_norm(r,dim=1);rnorm.append(length.mean())
            zero_min=torch.minimum(zero_min,length/1280)
            keep=win==s;ru[keep]=r[keep]
        means=torch.stack(means);delta=ru.mean(0)
        values={A:torch.linalg.vector_norm(ru-delta,dim=1)/1280,
                B:torch.full((n,),torch.inf,device=qt.device),C:torch.full((n,),torch.inf,device=qt.device)}
        winners={B:torch.zeros_like(win),C:torch.zeros_like(win)}
        for s in range(len(banks)):
            r=residual(s)
            for b,shift in ((B,delta),(C,means[s])):
                v=torch.linalg.vector_norm(r-shift,dim=1)/1280
                keep=v<values[b];values[b]=torch.minimum(values[b],v);winners[b][keep]=s
        # A single bulk transfer for small diagnostics; no per-patch synchronize.
        rn=torch.stack(rnorm).cpu().numpy();dn=torch.linalg.vector_norm(means,dim=1).cpu().numpy()
        un=float(torch.linalg.vector_norm(ru,dim=1).mean().cpu());du=float(torch.linalg.vector_norm(delta).cpu())
        result={b:v.cpu().numpy() for b,v in values.items()}
        saved=dict(delta_U=delta.cpu().numpy(),delta_ref=means.cpu().numpy(),union_source=win.cpu().numpy().astype(np.int8),
                   zero_fixed=(torch.linalg.vector_norm(ru,dim=1)/1280).cpu().numpy(),zero_reselect=zero_min.cpu().numpy(),
                   selected_B=winners[B].cpu().numpy().astype(np.int8),selected_C=winners[C].cpu().numpy().astype(np.int8))
    if any(not np.isfinite(v).all() for v in result.values()) or np.any(result[B]>result[A]+np.float32(1e-7)):raise ValueError('invalid stock FP32 translation')
    old=distances.min(0)
    diag=dict(delta_U_norm=du,mean_union_residual_norm=un,delta_U_ratio=du/un if un else None,zero_union_denominator=un==0,
              sources=[dict(source=s,delta_norm=float(dn[s]),mean_residual_norm=float(rn[s]),ratio=float(dn[s]/rn[s]) if rn[s] else None,zero_denominator=bool(rn[s]==0)) for s in range(len(banks))],
              branches={b:dict(mean_distance_before=float(old.mean(dtype=np.float32)),mean_distance_after=float(v.mean(dtype=np.float32)),
                              mean_signed_change=float((v-old).mean(dtype=np.float32)),increased_positions=int(np.sum(v>old)),
                              source_switch_fraction=0. if b==A else float(np.mean(saved['selected_'+('B' if b==B else 'C')]!=saved['union_source']))) for b,v in result.items()})
    return result,saved,diag


def translate(a,q,banks,indices,distances,refs,keys):
    if a.device=='cpu':return nm.translate(q,banks,indices,distances,chunk=a.vector_chunk)
    return translated_gpu(q,banks,indices,distances,refs.search,keys,a.vector_chunk)
