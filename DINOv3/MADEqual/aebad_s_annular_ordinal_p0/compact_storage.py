"""Lossless default-row codec and bounded asynchronous commits, no GPU dependency on import."""
import io
import os
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from . import inputs as ip
st=ip.st
CODEC='sparse_lossless_v1'
SMALL=('boxes','scores','evidence_present','agreement_fraction','disagreement_fraction','distinct_references')

def encode_tensors(arrays):
    import torch
    n=len(arrays['boxes']);result={name:arrays[name] for name in SMALL}
    mask=arrays['missing_mask_packed'].reshape(n*3,12288);flips=arrays['flip_counts'].reshape(n*3,64,64)
    refs=arrays['candidate_references'].reshape(n*3,12);trans=arrays['transforms'].reshape(n*3,12,2,3)
    rows=(mask!=255).any(1).nonzero().flatten();result.update(mask_rows=rows,mask_values=mask[rows])
    rows=(flips!=0).flatten(1).any(1).nonzero().flatten();result.update(flip_rows=rows,flip_values=flips[rows])
    rows=((refs!=-1).any(1)|(trans!=0).flatten(1).any(1)).nonzero().flatten()
    result.update(candidate_rows=rows,candidate_references_values=refs[rows],transforms_values=trans[rows])
    return result

def decode(arrays):
    n=len(arrays['boxes']);result={name:np.asarray(arrays[name]) for name in SMALL}
    mask=np.full((n*3,12288),255,np.uint8);mask[arrays['mask_rows']]=arrays['mask_values']
    flips=np.zeros((n*3,64,64),np.uint8);flips[arrays['flip_rows']]=arrays['flip_values']
    refs=np.full((n*3,12),-1,np.int16);refs[arrays['candidate_rows']]=arrays['candidate_references_values']
    transforms=np.zeros((n*3,12,2,3),np.float64);transforms[arrays['candidate_rows']]=arrays['transforms_values']
    result.update(missing_mask_packed=mask.reshape(n,3,12288),flip_counts=flips.reshape(n,3,64,64),
        candidate_references=refs.reshape(n,3,12),transforms=transforms.reshape(n,3,12,2,3))
    return result

def check_layout(arrays,boxes):
    n=len(boxes)
    specs=dict(boxes=((n,2),'int32'),scores=((n,3),'float32'),evidence_present=((n,3),'bool'),
        agreement_fraction=((n,3),'float32'),disagreement_fraction=((n,3),'float32'),distinct_references=((n,3),'int16'))
    for name,(shape,dtype) in specs.items():
        value=arrays[name]
        if value.shape!=shape or value.dtype!=np.dtype(dtype):raise ValueError('invalid sparse '+name+' layout')
    if not np.array_equal(arrays['boxes'],boxes):raise ValueError('sparse window coordinates differ')
    for prefix in ('mask','flip','candidate'):
        rows=arrays[prefix+'_rows']
        if rows.ndim!=1 or rows.dtype!=np.int64 or (len(rows) and (rows.min()<0 or rows.max()>=n*3 or np.any(np.diff(rows)<=0))):raise ValueError('invalid sparse row indices')
        values={'mask':(('mask_values',(len(rows),12288),'uint8'),),
            'flip':(('flip_values',(len(rows),64,64),'uint8'),),
            'candidate':(('candidate_references_values',(len(rows),12),'int16'),('transforms_values',(len(rows),12,2,3),'float64'))}[prefix]
        for name,shape,dtype in values:
            if arrays[name].shape!=shape or arrays[name].dtype!=np.dtype(dtype):raise ValueError('invalid sparse '+name+' layout')

def file_stamp(path):
    info=path.stat();return dict(bytes=info.st_size,mtime_ns=info.st_mtime_ns,ctime_ns=info.st_ctime_ns,inode=info.st_ino)

def validate(a,directory,record):
    path=Path(directory)/record['arrays'];stamp=file_stamp(path)
    if stamp['bytes']!=record['array_bytes']:raise ValueError('committed sparse NPZ size differs')
    proof=path.with_suffix('.validated.json');expected=dict(stamp=stamp,record=record['seal'],codec=CODEC)
    # Persisted, sealed read-validation proof avoids rereading unchanged payloads
    # at every standalone calibration/evaluation/report invocation.
    if proof.exists() and st.sealed_json(proof)['validated']==expected:return
    try:
        with np.load(path,allow_pickle=False) as z:check_layout(z,record['boxes'])
    except (OSError,ValueError,KeyError,EOFError,zipfile.BadZipFile) as e:raise ValueError('committed sparse NPZ invalid: '+str(path)) from e
    # A changed file is never accepted merely because an older proof exists.
    st.write_json(proof,dict(validated=expected,seal=st.digest(dict(validated=expected))))

def save(path,arrays):
    tmp=path.with_suffix('.pending')
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as archive:
        for name,array in arrays.items():
            with archive.open(name+'.npy','w',force_zip64=True) as member:np.lib.format.write_array(member,np.asarray(array),allow_pickle=False)
    os.replace(tmp,path)

class Writer:
    def __init__(self,depth=2):self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='annular-writer');self.slots=threading.BoundedSemaphore(depth);self.futures=[]
    def submit(self,a,directory,name,arrays,summary,started):
        import torch
        self.slots.acquire();event=None;hosts={}
        try:
            for key,value in arrays.items():
                if value.device.type=='cuda':
                    host=torch.empty_like(value,device='cpu',pin_memory=True);host.copy_(value,non_blocking=True);hosts[key]=host
                else:hosts[key]=value.detach().cpu()
            if arrays['boxes'].device.type=='cuda':event=torch.cuda.Event();event.record()
        except Exception:self.slots.release();raise
        def commit():
            try:
                if event is not None:event.synchronize()  # writer waits on this batch only
                cpu={key:value.numpy() for key,value in hosts.items()};write_tick=time.perf_counter()
                file=name.replace('.json','.npz');path=Path(directory)/file;save(path,cpu)
                record=st.sealed_json(Path(directory)/name,dict(summary,arrays=file,array_bytes=path.stat().st_size,
                    seconds=time.perf_counter()-started,codec=CODEC,write_seconds=time.perf_counter()-write_tick))
                validate(a,directory,record)
                return record
            finally:self.slots.release()
        future=self.pool.submit(commit);self.futures.append(future);return future
    def flush(self):
        for future in self.futures:future.result()
        self.futures.clear()
    def close(self):
        try:self.flush()
        finally:self.pool.shutdown(wait=True)
