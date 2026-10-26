"""Bounded resident references; unchanged host Exact FP32 cdist/chunk/ties."""
from collections import OrderedDict
import numpy as np


class ResidentExact:
    def __init__(self,device,max_bytes=4*2**30):
        self.device=device;self.max_bytes=int(max_bytes);self.cache=OrderedDict();self.bytes=0
        self.stats=dict(reference_uploads=0,reference_hits=0,reference_uploaded_bytes=0,query_uploads=0,searches=0)
    def query(self,q):
        import torch
        if isinstance(q,torch.Tensor):return q.to(device=self.device,dtype=torch.float32)
        self.stats['query_uploads']+=1
        # Own writable host storage, including when the input is read-only mmap.
        return torch.tensor(np.asarray(q),device=self.device,dtype=torch.float32)
    def reference(self,bank,key):
        import torch
        signature=(tuple(bank.shape),str(bank.dtype))
        if key in self.cache:
            tensor,old=self.cache[key]
            if old!=signature:raise ValueError('resident reference layout changed')
            self.cache.move_to_end(key);self.stats['reference_hits']+=1;return tensor
        size=int(bank.size)*np.dtype(bank.dtype).itemsize
        while self.cache and self.bytes+size>self.max_bytes:
            _,(tensor,_)=self.cache.popitem(last=False);self.bytes-=tensor.numel()*tensor.element_size()
        tensor=torch.tensor(np.asarray(bank),device=self.device,dtype=torch.float32)
        self.stats['reference_uploads']+=1;self.stats['reference_uploaded_bytes']+=size
        if size<=self.max_bytes:
            self.cache[key]=(tensor,signature);self.bytes+=tensor.numel()*tensor.element_size()
        return tensor
    def exact(self,q,bank,key):
        import torch
        from DINOv3.relation_reliability.superadd_host import _exact_top1_chunked
        with torch.inference_mode():
            query=self.query(q);reference=self.reference(bank,key)
            distance,index=_exact_top1_chunked(query,reference)
        self.stats['searches']+=1
        return distance.numpy().astype(np.float32)/1280,index.numpy().astype(np.int64)


def resident(a):return ResidentExact(a.device,int(getattr(a,'reference_cache_gib',4.)*2**30))
