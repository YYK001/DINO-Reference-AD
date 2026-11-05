"""Lazy runtime-only CUDA ownership, bounded transfers and event measurements."""
import contextlib
import time
from collections import OrderedDict
import numpy as np
import torch

def require_device(name):
    device=torch.device(name)
    if device.type!='cuda' or not torch.cuda.is_available():raise RuntimeError('CUDA backend requested but unavailable; no CPU fallback')
    if torch.cuda.device_count()!=1:raise RuntimeError('expose exactly the user-selected GPU with CUDA_VISIBLE_DEVICES')
    torch.cuda.set_device(device);return device

class Timers:
    def __init__(self,device):self.device=torch.device(device);self.events=[];self.wall={}
    @contextlib.contextmanager
    def host(self,name):
        tick=time.perf_counter()
        try:yield
        finally:self.wall[name]=self.wall.get(name,0.)+time.perf_counter()-tick
    @contextlib.contextmanager
    def measure(self,name):
        tick=time.perf_counter()
        if self.device.type=='cuda':
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
        try:yield
        finally:
            self.wall[name]=self.wall.get(name,0.)+time.perf_counter()-tick
            if self.device.type=='cuda':end.record();self.events.append((name,start,end))
    def collect(self,start=0,wall_base=None):
        result={}
        if self.events[start:]:
            self.events[-1][2].synchronize()  # once at image/stage boundary, never per operator
            for name,first,end in self.events[start:]:result[name]=result.get(name,0.)+first.elapsed_time(end)/1000.
        wall={k:v-(wall_base or {}).get(k,0.) for k,v in self.wall.items()}
        return dict(cuda_event_seconds=result,host_enqueue_or_control_seconds=wall,components_overlap_not_additive=True)

class TensorCache:
    def __init__(self,device,max_bytes,timers):
        self.device=torch.device(device);self.limit=int(max_bytes);self.timers=timers;self.values=OrderedDict();self.bytes=0
        self.stream=torch.cuda.Stream(device=self.device) if self.device.type=='cuda' else None
        self.uploads=0;self.hits=0
    def get(self,key,loader):
        if key in self.values:self.hits+=1;self.values.move_to_end(key);return self.values[key][0]
        with self.timers.host('cache_read'):
            array=np.asarray(loader());size=array.nbytes
            if key in self.values:return self.values[key][0]
            if self.stream is None:host=torch.from_numpy(array.copy())
            else:
                host=torch.empty(array.shape,dtype=torch.from_numpy(np.empty((),array.dtype)).dtype,pin_memory=True)
                host.copy_(torch.from_numpy(array.copy() if not array.flags.writeable else array))
        while self.values and self.bytes+size>self.limit:
            _,(_,used)=self.values.popitem(last=False);self.bytes-=used
        if size>self.limit:raise RuntimeError('one requested GPU cache item exceeds its configured budget; increase engineering cache size')
        with self.timers.measure('H2D'):
            if self.stream is None:value=host.to(self.device)
            else:
                # Pinned staging lifetime extends through the recorded copy event.
                with torch.cuda.stream(self.stream):value=host.to(self.device,non_blocking=True);event=torch.cuda.Event();event.record()
                torch.cuda.current_stream(self.device).wait_event(event)
                value.record_stream(torch.cuda.current_stream(self.device))
        self.values[key]=(value,size);self.bytes+=size;self.uploads+=1;return value
    def tensor(self,key,builder):
        if key in self.values:self.hits+=1;self.values.move_to_end(key);return self.values[key][0]
        value=builder();size=value.numel()*value.element_size()
        while self.values and self.bytes+size>self.limit:
            _,(_,used)=self.values.popitem(last=False);self.bytes-=used
        if size>self.limit:raise RuntimeError('one GPU cache tensor exceeds its configured engineering budget')
        self.values[key]=(value,size);self.bytes+=size;return value
    def prefetch(self,items):
        # Called once for the distinct normal IDs needed by the upcoming group.
        for key,loader in items:self.get(key,loader)
    def clear(self):self.values.clear();self.bytes=0
