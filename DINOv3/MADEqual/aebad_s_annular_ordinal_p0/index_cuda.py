"""GPU compact-binary CSR LSH and streamed exact integer reranking; no bucket clipping."""
import os
import json
import time
from pathlib import Path
import numpy as np
import torch
from . import inputs as ip,pixels as px,pixels_cuda as pc
from .protocol import PAIRS,VARIANTS,cuda_index_config
from .index import Store as CPUStore
from .cuda_support import TensorCache,Timers,require_device
st=ip.st
BUCKETS=1<<20

def index_root(a):return a.cache_root/'cuda_index_v1'

def hash_keys(bits,positions,chunk=262144):
    positions=torch.as_tensor(positions,device=bits.device,dtype=torch.int64)
    result=[];weights=1<<torch.arange(20,device=bits.device,dtype=torch.int32)
    for start in range(0,len(bits),chunk):
        values=bits[start:start+chunk,positions//8]
        result.append(((((values>>(positions%8))&1).to(torch.int32))*weights).sum(1))
    return torch.cat(result) if result else torch.empty(0,device=bits.device,dtype=torch.int64)

def build_csr(descriptors,positions):
    offsets=[];rows=[]
    for table in positions:
        keys=hash_keys(descriptors,table)
        rows.append(torch.argsort(keys,stable=True).to(torch.int32))
        counts=torch.bincount(keys,minlength=BUCKETS)
        offsets.append(torch.cat((torch.zeros(1,device=keys.device,dtype=torch.int64),counts.cumsum(0))).to(torch.int32))
    return torch.stack(offsets),torch.stack(rows)

def hamming(a,b):
    # Compact uint8 -> 16 integer byte popcounts, never 128 FP32 features.
    xor=torch.bitwise_xor(a,b).to(torch.int32)
    xor=xor-((xor>>1)&0x55);xor=(xor&0x33)+((xor>>2)&0x33)
    return (((xor+(xor>>4))&0x0f).sum(-1)).to(torch.int32)

def best_unique(qid,distance,rows,physical,budget):
    """Stable (distance, global row) order, one minimum per query/physical position."""
    if not len(rows):return qid,distance,rows,physical
    order=torch.argsort(distance.to(torch.int64)*(1<<32)+rows,stable=True)
    qid=qid[order];distance=distance[order];rows=rows[order];physical=physical[order]
    key=qid.to(torch.int64)*(1<<32)+physical
    order=torch.argsort(key,stable=True);key=key[order]
    first=torch.ones(len(key),device=key.device,dtype=torch.bool);first[1:]=key[1:]!=key[:-1]
    chosen=order[first];qid=qid[chosen];distance=distance[chosen];rows=rows[chosen];physical=physical[chosen]
    key=qid.to(torch.int64)*(1<<40)+distance.to(torch.int64)*(1<<32)+rows
    order=torch.argsort(key,stable=True);qid=qid[order];distance=distance[order];rows=rows[order];physical=physical[order]
    start=torch.searchsorted(qid,qid,right=False);rank=torch.arange(len(qid),device=qid.device)-start
    keep=rank<budget
    return qid[keep],distance[keep],rows[keep],physical[keep]

def probe(query,desc,offsets,order,physical,positions,row_start=0,budget=16,tile=262144,prior=None):
    """Visit every entry of every probed bucket in bounded tiles, including huge buckets."""
    q=len(query);n=len(desc);device=query.device
    if prior is None:prior=tuple(torch.empty(0,device=device,dtype=torch.int64) for _ in range(4))
    masks=torch.cat((torch.zeros(1,device=device,dtype=torch.int64),1<<torch.arange(20,device=device,dtype=torch.int64)))
    keys=torch.stack([hash_keys(query,p) for p in positions])[:,:,None]^masks
    tables=torch.arange(4,device=device)[:,None,None].expand_as(keys)
    begin=offsets[tables,keys].long()+tables*n;end=offsets[tables,keys+1].long()+tables*n
    sizes=(end-begin).reshape(-1);prefix=sizes.cumsum(0);starts=begin.reshape(-1)
    qids=torch.arange(q,device=device)[None,:,None].expand_as(keys).reshape(-1)
    total=int(prefix[-1].item()) if len(prefix) else 0  # one scheduling scalar per shard, not per bucket
    flat=order.reshape(-1)
    for lo in range(0,total,tile):
        ordinal=torch.arange(lo,min(total,lo+tile),device=device);segment=torch.searchsorted(prefix,ordinal,right=True)
        previous=torch.where(segment>0,prefix[(segment-1).clamp_min(0)],torch.zeros_like(segment))
        local=flat[starts[segment]+ordinal-previous].long();qi=qids[segment]
        d=hamming(query[qi],desc[local]).long();r=local+row_start;p=physical[local].long()
        prior=best_unique(*[torch.cat((old,new)) for old,new in zip(prior,(qi,d,r,p))],budget)
    return prior,total

def write_arrays(path,arrays,metadata):
    """Atomic uncompressed arrays, small dependency record; no repeated payload SHA pass."""
    path=Path(path);path.mkdir(parents=True,exist_ok=True);files={}
    for name,value in arrays.items():
        file=path/(name+'.npy');tmp=file.with_suffix('.pending')
        with tmp.open('wb') as f:np.save(f,value,allow_pickle=False)
        os.replace(tmp,file);files[name]=dict(shape=list(value.shape),dtype=str(value.dtype),bytes=file.stat().st_size)
    return st.sealed_json(path/'manifest.json',dict(metadata=metadata,files=files))

class Store(CPUStore):
    def __init__(self,a):
        super().__init__(a);self.device=require_device(a.device);self.timers=Timers(self.device)
        self.images=TensorCache(self.device,a.image_cache_gib*2**30,self.timers)
        self.indices=TensorCache(self.device,a.index_cache_gib*2**30,self.timers);self.load_seconds=0.
        self.shard_records={};self.metrics=dict(candidate_visits=0,unique_queries=0,original_queries=0,uncached_queries=0,query_cache_hits=0)
    def gray(self,row):
        original=self.a.cache_root/'gray'/ip.key(row)
        if (original/'manifest.json').exists():return super().gray(row)
        path=self.a.cache_root/'cuda_pixels_v1/gray'/ip.key(row)
        meta=dict(image=ip.image(row),pixel_contract=ip.pixel_contract(),producer='torch_FP32_pointwise_v1',implementation=st.sha(Path(pc.__file__)),torch=torch.__version__)
        if (path/'manifest.json').exists():
            arrays,r=st.get(path,('gray',))
            if r['metadata']!=meta:raise ValueError('GPU gray dependencies differ')
            return arrays['gray']
        if self.data is None:self.data=ip.Data(self.a.data)
        with self.timers.host('cache_read'):rgb=np.asarray(self.data.image(row))
        if rgb.shape!=(row['image_height'],row['image_width'],3):raise ValueError('encoded RGB raster dimensions differ')
        with self.timers.measure('H2D'):tensor=torch.as_tensor(rgb,device=self.device)
        with self.timers.measure('gray'):
            value=pc.grayscale(tensor)
        with self.timers.measure('D2H'):host=value.cpu().numpy()
        st.put(path,dict(gray=host),meta);self.images.tensor(ip.key(row),lambda:value);return host
    def image(self,row):return self.images.get(ip.key(row),lambda:self.gray(row))
    def describe(self,row):
        old=self.descriptor_path(row)
        if (old/'manifest.json').exists():return super().describe(row)
        path=self.a.cache_root/'cuda_pixels_v1/descriptors'/ip.key(row)
        blocks=[px.grid_spec((row['image_height'],row['image_width']),v) for v in range(9)]
        meta=dict(image=ip.image(row),pixel_contract=ip.pixel_contract(),pairs=PAIRS.tolist(),variants=list(map(list,VARIANTS)),blocks=blocks,
            producer='torch_dense_fractional_offsets_v1',implementation=st.sha(Path(pc.__file__)),torch=torch.__version__)
        if (path/'manifest.json').exists():
            r=st.sealed_json(path/'manifest.json')
            if r['metadata']!=meta:raise ValueError('GPU descriptor dependencies differ')
            return r
        gray=self.image(row);parts=[]
        for spec in blocks:
            with self.timers.measure('descriptor'):value=pc.describe_grid(gray,spec)
            with self.timers.measure('D2H'):parts.append(value.cpu().numpy())
        return st.put(path,dict(descriptors=np.concatenate(parts)),meta)
    def descriptor_file(self,row):
        path=self.descriptor_path(row)
        if not (path/'manifest.json').exists():path=self.a.cache_root/'cuda_pixels_v1/descriptors'/ip.key(row)
        return path/'descriptors.npy'
    def build(self):
        if hasattr(self,'_built_record'):return self._built_record
        root=index_root(self.a);root.mkdir(parents=True,exist_ok=True);tick=time.perf_counter();parts=[];reused=0
        for i,row in enumerate(self.rows):
            reused+=int((self.descriptor_path(row)/'manifest.json').exists() or (self.a.cache_root/'cuda_pixels_v1/descriptors'/ip.key(row)/'manifest.json').exists())
            parts.append(self.describe(row));print(f'[CUDA binary assets] {i+1}/463 preserved_or_GPU_generated',flush=True)
        inputs=dict(index=cuda_index_config(),parts=[r['seal'] for r in parts],images=[ip.image(r) for r in self.rows],
            implementation=st.sha(Path(__file__)))
        signature=st.digest(inputs);dest=root/signature[:20];self.path=dest
        commit=dest/'manifest.json'
        if commit.exists():
            record=st.sealed_json(commit);self.reference(commit,record,time.perf_counter()-tick,False);return record
        chunks=[]
        canonical={key:i for i,key in enumerate(sorted({r['instance_id'] for r in self.rows}))}
        stride=max((r['image_width']+7)//8 for r in self.rows);height=max((r['image_height']+7)//8 for r in self.rows)
        for kind,ids in [('Source',range(455))]+[(f'Target{i-455}',range(i,i+1)) for i in range(455,463)]:
            source_blocks=[];files=[];local=0
            for i in ids:
                row=self.rows[i];files.append((i,local,self.descriptor_file(row)))
                for b in parts[i]['metadata']['blocks']:
                    source_blocks.append(dict(b,ref=i,start=local,end=local+b['count']));local+=b['count']
            for begin in range(0,local,self.a.index_shard_rows):
                stop=min(local,begin+self.a.index_shard_rows);name=f'{kind}_{begin:010d}';path=dest/name
                metadata=dict(signature=signature,kind=kind,start=begin,rows=stop-begin,
                    blocks=[b for b in source_blocks if b['start']<stop and b['end']>begin])
                if (path/'manifest.json').exists():
                    r=st.sealed_json(path/'manifest.json')
                    if r['metadata']!=metadata:raise ValueError('CUDA shard metadata differs')
                else:
                    host=np.empty((stop-begin,16),np.uint8)
                    for j,(i,base,file) in enumerate(files):
                        end=files[j+1][1] if j+1<len(files) else local
                        lo=max(begin,base);hi=min(stop,end)
                        if lo<hi:host[lo-begin:hi-begin]=np.load(file,mmap_mode='r')[lo-base:hi-base]
                    with self.timers.measure('H2D'):desc=torch.as_tensor(host,device=self.device)
                    with self.timers.measure('index_prepare'):
                        offsets,order=build_csr(desc,cuda_index_config()['bit_positions'])
                    with self.timers.measure('D2H'):off_host=offsets.cpu().numpy();order_host=order.cpu().numpy()
                    with self.timers.host('index_write'):r=write_arrays(path,dict(descriptors=host,offsets=off_host,order=order_host),metadata)
                    del desc,offsets,order
                chunks.append(dict(path=str(path),seal=r['seal'],**metadata))
                print(f'[CUDA binary index] {name} rows={stop-begin} committed',flush=True)
        record=st.sealed_json(commit,dict(kind='torch_binary_CSR_LSH_v1',native_version=torch.__version__,inputs=inputs,groups=chunks,seconds=time.perf_counter()-tick,
            descriptor_bytes=sum(r['files']['descriptors']['bytes'] for r in parts),native_index_bytes=sum(
                sum(f['bytes'] for f in st.sealed_json(Path(c['path'])/'manifest.json')['files'].values()) for c in chunks),
            source_vector_copy_bytes=0,reused_descriptor_images=reused,fresh_descriptor_images=len(self.rows)-reused,
            physical_stride=stride,physical_height=height,canonical=canonical,
            timings=self.timers.collect(),all_buckets_fully_scanned=True))
        self.reference(commit,record,time.perf_counter()-tick,True);return record
    def reference(self,path,record,seconds,created):
        st.sealed_json(self.a.run_root/'index_reference.json',dict(identity=ip.identity(self.a),manifest=str(path),seal_reference=record['seal'],backend='torch_binary_CSR_LSH_v1'))
        log=self.a.run_root/'index_preparation_attempts.json'
        attempts=json.loads(log.read_text(encoding='utf-8')) if log.exists() else []
        attempts.append(dict(seconds=seconds,new_index=created,historical_build_seconds=record['seconds']))
        st.write_json(log,attempts)
        self._built_record=record
    def load(self):
        if self.backend is not None:return self.record
        tick=time.perf_counter();self.record=self.build();prepared_here=time.perf_counter()-tick;self.backend=self.record['groups']
        required=sum(sum(f['bytes'] for f in st.sealed_json(Path(g['path'])/'manifest.json')['files'].values())+g['rows']*4 for g in self.backend)
        self.residency='resident' if required<=self.indices.limit else 'streamed_LRU_no_CPU_search'
        if self.residency=='resident':
            for group in self.backend:self.group_arrays(group)
        self.load_seconds=time.perf_counter()-tick-prepared_here
        return self.record
    def group_arrays(self,group):
        path=Path(group['path'])
        if str(path) not in self.shard_records:self.shard_records[str(path)]=st.sealed_json(path/'manifest.json')
        record=self.shard_records[str(path)]
        if record['seal']!=group['seal']:raise ValueError('GPU shard reference differs')
        def load(name):
            file=path/(name+'.npy');spec=record['files'][name]
            if file.stat().st_size!=spec['bytes']:raise ValueError('GPU shard byte count differs')
            array=np.load(file,mmap_mode='r',allow_pickle=False)
            if list(array.shape)!=spec['shape'] or str(array.dtype)!=spec['dtype']:raise ValueError('GPU shard layout differs')
            return array
        values={name:self.indices.get((str(path),name),lambda name=name:load(name)) for name in ('descriptors','offsets','order')}
        values['physical']=self.indices.tensor((str(path),'physical'),lambda:self.physical(group,self.device))
        return values
    def physical(self,group,device):
        blocks=group['blocks'];ends=torch.tensor([b['end'] for b in blocks],device=device)
        ids=torch.arange(group['start'],group['start']+group['rows'],device=device);b=torch.searchsorted(ends,ids,right=True)
        values=torch.tensor([[x['start'],x['x0'],x['y0'],x['nx'],self.record['canonical'][self.rows[x['ref']]['instance_id']]] for x in blocks],device=device)
        selected=values[b];n=ids-selected[:,0];x=selected[:,1]+n%selected[:,3]*8;y=selected[:,2]+n//selected[:,3]*8
        return ((selected[:,4]*self.record['physical_height']+y//8)*self.record['physical_stride']+x//8).int()
    def lookup(self,bits):
        self.load();unique,inverse=torch.unique(bits,dim=0,return_inverse=True)
        self.metrics['original_queries']+=len(bits);self.metrics['unique_queries']+=len(unique)
        with self.timers.measure('D2H'):
            host_unique=unique.cpu().numpy();back=inverse.cpu().numpy()
        keys=[row.tobytes() for row in host_unique];answers={k:self.query_cache[k] for k in keys if k in self.query_cache}
        missing=[i for i,k in enumerate(keys) if k not in answers]
        self.metrics['query_cache_hits']+=len(keys)-len(missing);self.metrics['uncached_queries']+=len(missing)
        for k in answers:self.query_cache.move_to_end(k)
        if not missing:return [answers[keys[i]] for i in back]
        query=unique[torch.tensor(missing,device=self.device)]
        logical={};saturated=torch.zeros(len(query),device=self.device,dtype=torch.bool)
        for group in self.backend:
            kind=group['kind'];budget=16 if kind=='Source' else 4;loaded=self.group_arrays(group)
            with self.timers.measure('retrieval'):
                current,visited=probe(query,loaded['descriptors'],loaded['offsets'],loaded['order'],loaded['physical'],
                    cuda_index_config()['bit_positions'],row_start=group['start'],budget=budget+1,tile=self.a.candidate_tile,prior=logical.get(kind))
            self.metrics['candidate_visits']+=visited;logical[kind]=current
        # Only a compact <= Source16 + 8*Target4 result leaves the GPU.
        outputs=[]
        for kind,current in logical.items():
            qi,distance,rows,physical=current;budget=16 if kind=='Source' else 4
            rank=torch.arange(len(qi),device=self.device)-torch.searchsorted(qi,qi,right=False)
            saturated.scatter_(0,qi[rank>=budget],True);keep=rank<budget
            with self.timers.measure('D2H'):outputs.append((kind,torch.stack((qi[keep],distance[keep],rows[keep]),1).cpu().numpy()))
        with self.timers.measure('D2H'):flags=saturated.cpu().numpy()
        result=[[] for _ in range(len(query))]
        for kind,values in outputs:
            groups=[g for g in self.backend if g['kind']==kind];blocks={b['end']:b for g in groups for b in g['blocks']}
            blocks=[blocks[k] for k in sorted(blocks)];ends=np.array([b['end'] for b in blocks])
            for qi,distance,row in values:
                b=blocks[int(np.searchsorted(ends,row,side='right'))];n=int(row)-b['start']
                result[int(qi)].append((int(distance),b['ref'],b['x0']+(n%b['nx'])*8,b['y0']+(n//b['nx'])*8,b['version']))
        for j,i in enumerate(missing):answers[keys[i]]=(result[j],bool(flags[j]));self.query_cache[keys[i]]=answers[keys[i]]
        while len(self.query_cache)>8192:self.query_cache.popitem(last=False)
        return [answers[keys[i]] for i in back]
    def lookup_many(self,bits):return self.lookup(torch.as_tensor(bits,device=self.device))
    def close(self):self.images.clear();self.indices.clear();super().close()
