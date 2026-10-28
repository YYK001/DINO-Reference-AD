"""Blind reference first; only then consume the original q and frozen UNION."""
from contextlib import ExitStack
import numpy as np
from . import inputs as ip,context as cx,runtime as rt,training
from .inputs import st
from .protocol import LAYERS,MAIN,U
from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.calibration import z_values,response


def path(a,s):return a.cache_root/'predictions/background'/s['sample_id']


def load(a,s):
    x,r=st.get(path(a,s));m=r['metadata'];n=int(np.prod(m['grid']))
    if m['identity']!=ip.identity(a) or m['sample']!=s or m['model_seal']!=training.trained(a)[1]['seal']:raise ValueError('prediction dependencies changed')
    ip.positions(x['positions'],n)
    if x[MAIN].shape!=(4,n) or x[MAIN].dtype!=np.float32 or not np.isfinite(x[MAIN]).all():raise ValueError('invalid full prediction')
    return x,r


class Synthesizer:
    """Resident layer keys/normal values, no original feature-dependent selection."""
    def __init__(self,a,states):
        import torch
        self.torch=torch;self.a=a;self.states=states;self.layers={}
        for l in LAYERS:
            d,r=rt.dictionary(a,l);index=rt.ContextIndex(d,a.device)
            self.layers[l]=dict(index=index,values=torch.tensor(np.asarray(d['values']),device=a.device),
                                types=torch.tensor(np.asarray(d['types']),device=a.device))

    def reference(self,layer,context,chunk=256,exclude_image=None,return_details=False):
        """Only projected masked context is accepted; q is deliberately absent."""
        torch=self.torch;d=self.layers[layer];index=d['index'];ids=index.select(context,excluded=exclude_image)
        state={k:torch.tensor(v,device=self.a.device) for k,v in self.states[layer].items()};result=[];attention=[]
        with torch.inference_mode():
            for start in range(0,len(context),chunk):
                ci=torch.tensor(ids[start:start+chunk],device=self.a.device);valid=ci>=0;ci=ci.clamp_min(0)
                q=torch.tensor(np.asarray(context[start:start+chunk]),device=self.a.device)@state['Wq']
                k=index.keys[ci]@state['Wk']
                q=torch.nn.functional.normalize(q,dim=-1,eps=1e-12);k=torch.nn.functional.normalize(k,dim=-1,eps=1e-12)
                logits=(q[:,None]*k).sum(-1)/.1+state['bias'][d['types'][ci]];logits.masked_fill_(~valid,-torch.inf)
                weights=torch.softmax(logits,1);value=(weights[:,:,None]*d['values'][ci]).sum(1)
                result.append(value.cpu().numpy())
                if return_details:attention.append(weights.cpu().numpy())
        value=np.concatenate(result)
        return (value,np.concatenate(attention),ids) if return_details else value


def predict(a):
    training.require_isolation(a,trained=True);states,trained=training.trained(a);p=ip.parent(a)
    frozen=a.run_root/'predictions_frozen.json'
    if frozen.exists():
        # Preserve the original bytes/seal, including legacy execution counters:
        # downstream CPU/AUPRO records already reference this exact commit.
        saved=verified(a)
        return dict(status='predictions_reused',prediction_seal=saved['seal'],images=saved['images'],
                    new_images=0,reused_images=saved['images'],masked_full_extractions=0,new_original_extractions=0)
    from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.fresh_inputs import Data
    synth=None;data=None;records=[];calls=0;created=0;fresh0=getattr(a,'_original_calls',0)
    # This method only creates predictions; NO GT reads, labels, score gates or instances.
    with ExitStack() as stack:
        for s in p.samples:
            dest=path(a,s)
            if not (dest/'manifest.json').exists():
                if synth is None:
                    from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.runtime import reliable_precision
                    reliable_precision(a.device);synth=Synthesizer(a,states)
                    data=stack.enter_context(Data(a.data))
                value=rt.processed(a,data.image(s));context,m=rt.masked_features(a,value);calls+=m['masked_full_extractions']
                # Commit the contextual references before any original q/UNION access.
                refs={l:synth.reference(l,context[l]) for l in LAYERS}
                old,om=p.prediction(s);x,xr,origin=rt.original(a,s,data)
                if om['metadata']['grid']!=m['grid'] or xr['metadata']['grid']!=m['grid']:raise ValueError('new and frozen physical grids differ')
                ip.positions(x['positions'],int(np.prod(m['grid'])))
                for l in LAYERS:
                    if x[f'q{l}'].dtype!=np.float32 or x[f'q{l}'].shape!=(len(x['positions']),1280):raise ValueError('original query must retain full FP32 1280D features')
                pos=m['positions'];inverse=np.argsort(x['positions']);oldinv=np.argsort(old['positions'])
                # Consume actual original raw distances. No inverse-MAD reconstruction.
                px,pr=st.get(p.cache/'predictions/background'/s['sample_id'],('raw_union','positions'))
                if pr['seal']!=om['seal']:raise ValueError('parent raw UNION identity differs')
                raw_union=np.asarray(px['raw_union'][:,np.argsort(px['positions'])[pos]],np.float32)
                new=[];audit=[]
                for j,l in enumerate(LAYERS):
                    dist,dp=cx.detection(x[f'q{l}'][inverse[pos]],refs[l],raw_union[j]);new.append(dist)
                    audit.append(dict(layer=l,effectively_reduced=int(np.sum(raw_union[j]-dist>1e-7+1e-6*np.abs(raw_union[j]))),positions=len(pos)))
                raw=np.stack(new);z=z_values(raw,p.mad['patch']);uz=np.asarray(old[U][:,oldinv[pos]])
                z[raw==raw_union]=uz[raw==raw_union]
                shape=[s['image_height']//4,s['image_width']//4];score=response(z,pos,m['grid'],shape,p.mad['image'])[1]
                st.put(dest,{MAIN:z,'positions':pos},dict(identity=ip.identity(a),sample=s,grid=m['grid'],output_shape=shape,scores=score,
                       model_seal=trained['seal'],parent_prediction_seal=om['seal'],GT_used=False,
                       reference_input='masked context only; original q loaded after references fixed',
                       masked_full_extractions=m['masked_full_extractions'],host_internal_crop_forwards=m['host_internal_crop_forwards'],diagnostics=audit))
                created+=1
            x,r=load(a,s);records.append(dict(domain='background',sample_id=s['sample_id'],seal=r['seal']))
            print(f'[blind predict] background/{s["sample_id"]} one_new_map',flush=True)
    saved=st.sealed_json(frozen,dict(identity=ip.identity(a),records=records,images=len(records),GT_used=False))
    # Invocation costs belong in the runner's per-session cost record, not in
    # immutable prediction content (which must survive no-op/partial resumes).
    return dict(status='predictions_frozen',prediction_seal=saved['seal'],images=len(records),new_images=created,
                reused_images=len(records)-created,masked_full_extractions=calls,
                new_original_extractions=getattr(a,'_original_calls',0)-fresh0)


def verified(a):
    training.require_isolation(a,trained=True);r=st.sealed_json(a.run_root/'predictions_frozen.json');samples=ip.parent(a).samples
    if r['identity']!=ip.identity(a) or r['images']!=297 or r['GT_used'] or [v['sample_id'] for v in r['records']]!=[s['sample_id'] for s in samples]:raise ValueError('full 297 predictions must freeze before GT access')
    for s,v in zip(samples,r['records']):
        _,pr=load(a,s)
        if pr['seal']!=v['seal']:raise ValueError('frozen prediction modified')
    return r
