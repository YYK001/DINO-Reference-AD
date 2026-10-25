"""Torch is deliberately confined to explicit compute stages."""
import math
import numpy as np
import torch
from torch import nn


class Adapter(nn.Module):
    def __init__(self,mu,sigma,hidden=512):
        super().__init__()
        self.register_buffer('mu',torch.as_tensor(np.array(mu,copy=True),dtype=torch.float32))
        self.register_buffer('sigma',torch.as_tensor(sigma,dtype=torch.float32))
        if self.mu.ndim!=1 or not torch.isfinite(self.mu).all() or not torch.isfinite(self.sigma) or self.sigma<=0:
            raise ValueError('finite positive Source normalization required')
        self.first=nn.Linear(len(mu),hidden);self.last=nn.Linear(hidden,len(mu))
        nn.init.zeros_(self.last.weight);nn.init.zeros_(self.last.bias)

    def forward(self,f,*,training_bf16=False):
        # Only Linear operations use BF16. Centering/GELU/residual/loss stay FP32.
        x=(f.float()-self.mu)/self.sigma
        with torch.autocast(f.device.type,dtype=torch.bfloat16,enabled=training_bf16 and f.is_cuda):y=self.first(x)
        y=torch.nn.functional.gelu(y.float())
        with torch.autocast(f.device.type,dtype=torch.bfloat16,enabled=training_bf16 and f.is_cuda):y=self.last(y)
        return f.float()+self.sigma*y.float()


def paired_loss(net,anchor,positive,weight,negative,extra):
    n,d=anchor.shape
    # Padded positive values carry zero weight; excluded from identity statistics.
    ha=net(anchor,training_bf16=True);hp=net(positive.reshape(-1,d),training_bf16=True).reshape_as(positive)
    hn=net(negative.reshape(-1,d),training_bf16=True).reshape_as(negative)
    denom=d*net.sigma.square()
    pos=((ha[:,None]-hp).square().sum(-1)/denom*weight).sum(-1).mean()
    dn=torch.linalg.vector_norm(ha[:,None]-hn,dim=-1)/(math.sqrt(d)*net.sigma)
    d0=torch.linalg.vector_norm(anchor[:,None]-negative,dim=-1)/(math.sqrt(d)*net.sigma)
    neg=torch.relu(.8*d0-dn).square().mean()
    raw=torch.cat((anchor,positive[weight>0],negative.reshape(-1,d),extra))
    transformed=net(raw,training_bf16=True)
    ident=(transformed-raw).square().sum(-1).mean()/denom
    return pos,neg,ident


def lr_at(update,total):
    warm=max(1,int(total*.05))
    if update<warm:return 1e-4*(update+1)/warm
    progress=(update-warm)/max(1,total-warm-1)
    return 1e-6+.5*(1e-4-1e-6)*(1+math.cos(math.pi*progress))


@torch.inference_mode()
def transform(net,x,device,chunk=4096):
    out=np.empty(x.shape,np.float32)
    for i in range(0,len(x),chunk):
        value=net(torch.as_tensor(np.array(x[i:i+chunk],copy=True),device=device,dtype=torch.float32))
        if not torch.isfinite(value).all():raise FloatingPointError('nonfinite transformed representation')
        out[i:i+chunk]=value.cpu().numpy()
    return out


def exact(q,bank,device):
    from DINOv3.relation_reliability.superadd_host import _exact_top1_chunked
    with torch.inference_mode():
        def tensor(x):
            if isinstance(x,torch.Tensor):return x.to(device=device,dtype=torch.float32)
            a=np.asarray(x)
            return (torch.as_tensor if a.flags.writeable else torch.tensor)(a,device=device,dtype=torch.float32)
        query=tensor(q);reference=tensor(bank)
        distance,index=_exact_top1_chunked(query,reference)
    return distance.numpy().astype(np.float32)/1280,index.numpy().astype(np.int64)
