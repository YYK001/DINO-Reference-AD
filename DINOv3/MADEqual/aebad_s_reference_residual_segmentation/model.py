"""Point-channel normalization and finite local receptive field, no global statistics."""
import torch
from torch import nn
import torch.nn.functional as F


class PointNorm(nn.Module):
    def __init__(self,channels):super().__init__();self.norm=nn.LayerNorm(channels)
    def forward(self,x):return self.norm(x.permute(0,2,3,1).float()).permute(0,3,1,2)


class Block(nn.Module):
    def __init__(self,channels,dilation):
        super().__init__();self.norm=PointNorm(channels);self.conv=nn.Conv2d(channels,channels,3,padding=dilation,dilation=dilation)
    def forward(self,x):return x+self.conv(F.gelu(self.norm(x)))


class Head(nn.Module):
    def __init__(self,dimension=1280,channels=128):
        super().__init__();self.branches=nn.ModuleList([nn.Sequential(nn.Conv2d(3*dimension+3,channels,1),
            PointNorm(channels),nn.GELU(),Block(channels,1),Block(channels,2),nn.Conv2d(channels,1,1)) for _ in range(4)])
    def forward(self,values):return torch.cat([branch(x) for branch,x in zip(self.branches,values)],1).float()


def inputs(query,mu,sigma,*,source=None,target=None,distances=None,patch=None,use_reference=True):
    """NoRef never accesses source/target/distances, even poisoned placeholders."""
    q=query.float();normalized=(q-mu[:,None,None])/sigma
    if not use_reference:return torch.cat((normalized,torch.zeros_like(q),torch.zeros_like(q),
        q.new_zeros((3,*q.shape[-2:]))),0)
    rs=(q-source.float())/sigma;rt=(q-target.float())/sigma
    ds,dt=distances.float();raw=torch.stack((ds,dt,torch.minimum(ds,dt)))
    z=(raw-patch[0])/patch[1]
    return torch.cat((normalized,rs,rt,z),0)


def balanced_bce(logit,mask):
    # Soft positive and negative areas separately normalized per image.
    axes=tuple(range(1,mask.ndim));pos=mask.sum(axes);neg=(1-mask).sum(axes)
    positive=(F.softplus(-logit)*mask).sum(axes)/pos.clamp_min(1e-12)
    negative=(F.softplus(logit)*(1-mask)).sum(axes)/neg.clamp_min(1e-12)
    return (torch.where(pos>0,positive,positive*0)+torch.where(neg>0,negative,negative*0)).mean()


def loss(logits,mask):
    layers=torch.stack([balanced_bce(logits[:,j:j+1],mask) for j in range(4)]).mean()
    fused=logits.mean(1,keepdim=True);bce=balanced_bce(fused,mask)
    axes=tuple(range(1,mask.ndim));probability=torch.sigmoid(fused);area=mask.sum(axes)
    dice=1-(2*(probability*mask).sum(axes)+1e-8)/(probability.sum(axes)+area+1e-8)
    dice=torch.where(area>0,dice,dice*0)
    return layers+bce+.5*dice.mean()
