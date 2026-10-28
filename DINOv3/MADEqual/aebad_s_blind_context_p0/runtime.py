"""Lazy original host; the input is masked globally before original stitched extraction."""
from pathlib import Path
from importlib.metadata import version
import time
import numpy as np
from . import inputs as ip,context as cx
from .inputs import st
from .protocol import LAYERS,CONFIG


def model(a):
    if not hasattr(a,'_host'):
        p=ip.parent(a);record=p.extraction
        for n,v in record['software'].items():
            if version(n)!=v:raise ValueError('reuse requires original extraction software: '+n)
        for rel,h in record['code'].items():
            if st.sha(ip.ROOT/rel)!=h:raise ValueError('original extraction implementation differs: '+rel)
        for rel,h in record['external_code'].items():
            name,tail=rel.split('/',1);root=a.dinov3_root if name=='dinov3' else a.official_root
            if st.sha(root/tail)!=h:raise ValueError('backbone/host implementation differs: '+rel)
        from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.runtime import host,reliable_precision
        reliable_precision(a.device);a._host=host(a)[0]
    return a._host


def processed(a,image):
    import torch
    from DINOv3.relation_reliability.superadd_host import image_tensor
    host=model(a)
    with torch.inference_mode():return host.preprocessing(image_tensor(image).to(a.device)[None])


def forward_processed(a,value,selected=None,context_only=False):
    """Masking precedes this call; query_masks ONLY saves transfers, never creates blindness."""
    import torch
    from DINOv3.relation_reliability.superadd_host import _extract_processed_batch
    host=model(a);gh,gw=value.shape[-2]//16,value.shape[-1]//16
    qm=None if selected is None else np.isin(np.arange(gh*gw),selected).reshape(gh,gw)
    with torch.inference_mode():
        pending=_extract_processed_batch(host,value,device=a.device,query_masks=[qm],include_identity_descriptors=False)[0]
    a._full_forwards=getattr(a,'_full_forwards',0)+1
    a._crop_forwards=getattr(a,'_crop_forwards',0)+len(host.patch_exec.axis_patch_split(value.shape[-2])[0])*len(host.patch_exec.axis_patch_split(value.shape[-1])[0])
    if tuple(pending.grid)!=(gh,gw):raise ValueError('host stitched grid is not the global processed patch grid')
    if context_only:
        if not hasattr(a,'_projections'):a._projections={l:torch.tensor(cx.projection(l),device=a.device) for l in LAYERS}
        with torch.inference_mode():
            output={l:torch.nn.functional.normalize(pending.queries[l].float()@a._projections[l],dim=-1,eps=1e-12).cpu().numpy() for l in LAYERS}
    else:output={l:pending.queries[l].detach().float().cpu().numpy() for l in LAYERS}
    return output,dict(grid=[gh,gw],positions=pending.positions.copy())


def masked_features(a,value,selected=None):
    """At most four complete extractions, each simultaneously returns four layers."""
    grid=[value.shape[-2]//16,value.shape[-1]//16];n=int(np.prod(grid))
    pos=np.arange(n,dtype=np.int64) if selected is None else np.sort(np.asarray(selected,np.int64))
    labels=cx.phases(grid).ravel()[pos];keys={l:np.empty((len(pos),128),np.float32) for l in LAYERS};calls=0;crop_calls=0
    host=model(a);ny=len(host.patch_exec.axis_patch_split(value.shape[-2])[0]);nx=len(host.patch_exec.axis_patch_split(value.shape[-1])[0])
    for phase in range(4):
        take=np.flatnonzero(labels==phase)
        if not len(take):continue
        masked=cx.masked_input(value,phase)  # BEFORE host.patch_exec sees any crop.
        q,m=forward_processed(a,masked,pos[take],context_only=True);calls+=1;crop_calls+=ny*nx
        if not np.array_equal(m['positions'],pos[take]):raise ValueError('masked row/physical position mapping differs')
        for l in LAYERS:keys[l][take]=q[l]
    return keys,dict(grid=grid,positions=pos,phase=labels,processed_shape=list(value.shape[-2:]),
                     masked_full_extractions=calls,host_internal_crop_forwards=crop_calls,
                     mask='global normalized input zeros; 5x5 disjoint blocks, clipped edges')


def original(a,row,data):
    """Original q is used as dictionary VALUE/target or AFTER synthesis, never as context."""
    path=ip.original_path(a,row)
    if (path/'manifest.json').exists():
        x,r=st.get(path)
    else:
        ref=ip.raw_reference(a,row) if row in ip.dictionary_rows(a) else ip.parent(a).assets['queries'].get('background/'+row['sample_id'])
        if ref and (Path(ref['path'])/'manifest.json').exists():
            if row in ip.dictionary_rows(a):ip.inspect_raw(ref,row)
            x,r=st.get(ref['path'],tuple(f'q{l}' for l in LAYERS)+('positions',))
            if r['seal']!=ref['seal']:raise ValueError('raw original feature changed')
            if r['metadata'].get('sample',r['metadata'].get('image'))!=row:raise ValueError('raw Q image differs')
            return x,r,'reused'
        else:
            value=processed(a,data.image(row));q,m=forward_processed(a,value)
            if row not in ip.dictionary_rows(a):
                # Streaming query: no extra 297 x four full 1280D arrays on disk.
                a._original_calls=getattr(a,'_original_calls',0)+1
                return {**{f'q{l}':q[l] for l in LAYERS},'positions':m['positions']},dict(metadata=dict(image=row,grid=m['grid'])),'fresh_streamed_not_saved'
            st.put(path,{**{f'q{l}':q[l] for l in LAYERS},'positions':m['positions']},
                   dict(image=row,kind='original_complete_grid',grid=m['grid'],output_shape=[row['image_height']//4,row['image_width']//4],extraction=ip.parent(a).extraction))
            x,r=st.get(path);a._original_calls=getattr(a,'_original_calls',0)+1
    if r['metadata']['image']!=row:raise ValueError('original feature image changed')
    return x,r,'current_cache'


def context_bundle(a,row,data,kind):
    path=ip.context_path(a,row)
    if (path/'manifest.json').exists():
        x,r=st.get(path)
        if r['metadata']['identity']!=ip.identity(a) or r['metadata']['image']!=row:raise ValueError('blind context image/implementation changed')
        return x,r
    image=data.image(row);value=processed(a,image);n=(value.shape[-2]//16)*(value.shape[-1]//16)
    chosen=cx.select_positions(np.arange(n,dtype=np.int64),512,row['relative_path'],'Source_dictionary') if kind=='Source' else None
    keys,meta=masked_features(a,value,chosen)
    st.put(path,{**{f'c{l}':keys[l] for l in LAYERS},'positions':meta.pop('positions'),'phase':meta.pop('phase')},
           dict(identity=ip.identity(a),image=row,**meta))
    return st.get(path)


class ContextIndex:
    """Keys resident once per layer; FP32 exhaustive search, first-row exact ties."""
    def __init__(self,dictionary,device):
        import torch
        self.torch=torch;self.device=device;self.dictionary=dictionary
        self.keys=torch.tensor(np.asarray(dictionary['keys']),device=device)
        self.images=torch.tensor(np.asarray(dictionary['images']),device=device)
        types=dictionary['types'];images=dictionary['images']
        groups=[np.flatnonzero(types==0)]+[np.flatnonzero((types==1)&(images==i)) for i in sorted(set(images[types==1]))]
        if len(groups)!=9:raise ValueError('eight ordered Target groups required')
        self.groups=[torch.tensor(x,device=device) for x in groups]
        self.group_keys=[self.keys[g] for g in self.groups];self.group_images=[self.images[g] for g in self.groups]

    def select(self,query_context,excluded=None,chunk=128):
        torch=self.torch;n=len(query_context);out=np.full((n,12),-1,np.int64)
        excluded=np.full(n,-1,np.int64) if excluded is None else np.broadcast_to(excluded,(n,))
        with torch.inference_mode():
            for start in range(0,n,chunk):
                q=torch.tensor(np.asarray(query_context[start:start+chunk]),device=self.device)
                ex=torch.tensor(np.asarray(excluded[start:start+chunk]),device=self.device);col=0;result=[]
                for gi,(g,k,im) in enumerate(zip(self.groups,self.group_keys,self.group_images)):
                    score=q@k.T;score.masked_fill_(im[None]==ex[:,None],-torch.inf)
                    for _ in range(4 if gi==0 else 1):
                        v,j=score.max(1);result.append(torch.where(torch.isfinite(v),g[j],-1));score.scatter_(1,j[:,None],-torch.inf);col+=1
                out[start:start+len(q)]=torch.stack(result,1).cpu().numpy()
        return out


def dictionary(a,layer):
    x,r=st.get(a.cache_root/'dictionary'/f'L{layer}')
    if r['metadata']['identity']!=ip.identity(a):raise ValueError('dictionary identity changed')
    return x,r


def prepare_context(a):
    ip.freeze(a)
    from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.fresh_inputs import Data
    rows=ip.dictionary_rows(a);records=[];times=[];context_times=[];original_times=[];calls=0;internal=0;fresh0=getattr(a,'_original_calls',0)
    limit=a.limit_normal
    measured_rows=([rows[0],rows[64]]+[r for i,r in enumerate(rows) if i not in (0,64)])[:limit] if limit else rows
    with Data(a.data) as data:
        for i,row in enumerate(measured_rows):
            tick=time.perf_counter();kind='Target' if 'support_index' in row else 'Source'
            existed=(ip.context_path(a,row)/'manifest.json').exists();cxdata,cr=context_bundle(a,row,data,kind)
            context_seconds=time.perf_counter()-tick;raw_tick=time.perf_counter()
            x,r,origin=original(a,row,data)
            original_seconds=time.perf_counter()-raw_tick
            ip.positions(x['positions'],int(np.prod(r['metadata']['grid'])))
            if r['metadata']['grid']!=cr['metadata']['grid']:raise ValueError('original/context physical grid differs')
            records.append(dict(image=row['relative_path'],context_seal=cr['seal'],original_seal=r['seal'],origin=origin))
            if not existed:
                calls+=cr['metadata']['masked_full_extractions'];internal+=cr['metadata']['host_internal_crop_forwards'];times.append(time.perf_counter()-tick)
                context_times.append(context_seconds);original_times.append(original_seconds)
            print(f'[blind context] {i+1}/{len(rows)} {kind} {row["relative_path"]} seconds={time.perf_counter()-tick:.2f}',flush=True)
    if limit and limit<72:
        # Real original RGB path + real host, never constructed-input throughput.
        value=dict(status='partial_normal_throughput_only',images=len(records),new_images_timed=len(times),seconds_per_image=float(np.mean(times)) if times else None,
                   context_seconds_per_image=float(np.mean(context_times)) if context_times else None,
                   original_feature_seconds_per_image=float(np.mean(original_times)) if original_times else None,
                   dictionary_masked_extractions=288,query_masked_extractions=1188,
                   extrapolated_remaining_context_seconds=float(np.mean(context_times))*(sum(not (ip.context_path(a,r)/'manifest.json').exists() for r in rows)+sum(not (a.cache_root/'predictions/background'/s['sample_id']/'manifest.json').exists() for s in ip.parent(a).samples)) if context_times else None,
                   head_training_seconds=None,CPU_evaluation_seconds=None,AUPRO_seconds=None,
                   original_query_extraction_seconds=None,candidate_retrieval_and_synthesis_seconds=None,
                   interpretation='real normal RGB+masked host; context timing separated from original features; no query measurements, estimate provisional',
                   masked_full_extractions=calls,host_internal_crop_forwards=internal,records=records)
        st.write_json(a.run_root/'throughput.json',value);return value
    # One layer at a time; Exact formal Source key/value mapping only at sampled Source positions.
    from DINOv3.MADEqual.aebad_s_reference_residual_segmentation.execution import ResidentExact
    from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.runtime import reliable_precision
    reliable_precision(a.device);search=ResidentExact(a.device,2*2**30);source=ip.source_reference(a);dictrecords=[]
    for l in LAYERS:
        dest=a.cache_root/'dictionary'/f'L{l}'
        if (dest/'manifest.json').exists():
            _,rec=dictionary(a,l);dictrecords.append(rec['seal']);continue
        keys=[];values=[];images=[];positions=[];types=[];bankindices=[]
        bank,_=st.get(source['path'],(f'L{l}',))
        with Data(a.data) as data:
            for image_id,row in enumerate(rows):
                c,cr=st.get(ip.context_path(a,row),(f'c{l}','positions'));x,r,_=original(a,row,data)
                inverse=np.argsort(x['positions']);pos=c['positions'];q=np.array(x[f'q{l}'][inverse[pos]],np.float32)
                is_target='support_index' in row
                if is_target:val=q;idx=np.full(len(q),-1,np.int64)
                else:
                    _,idx=search.exact(q,bank[f'L{l}'],('formal',source['seal'],l));val=np.array(bank[f'L{l}'][idx],np.float32)
                keys.append(c[f'c{l}']);values.append(val);images.extend([image_id]*len(pos));positions.extend(pos);types.extend([int(is_target)]*len(pos));bankindices.extend(idx)
        rec=st.put(dest,dict(keys=np.concatenate(keys),values=np.concatenate(values),images=np.asarray(images,np.int64),
                            positions=np.asarray(positions,np.int64),types=np.asarray(types,np.int64),source_bank_indices=np.asarray(bankindices,np.int64)),
                   dict(identity=ip.identity(a),layer=l,images=rows,inputs=records,source_seal=source['seal'],source_values='formal_bank_rows_only'))
        dictrecords.append(rec['seal']);print(f'[blind dictionary] L{l} entries={sum(map(len,keys))}',flush=True)
    done=st.sealed_json(a.run_root/'context_ready.json',dict(identity=ip.identity(a),dictionary_seals=dictrecords,records=records))
    return dict(status='normal_dictionary_ready',masked_full_extractions=calls,host_internal_crop_forwards=internal,
                new_original_extractions=getattr(a,'_original_calls',0)-fresh0,exact_statistics=search.stats,context_seal=done['seal'])
