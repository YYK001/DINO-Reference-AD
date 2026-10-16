"""Twelve fixed cases, four branches and only the two requested comparisons."""
from .inputs import BRANCHES,NEW
import numpy as np


def plot_case(image,mask,maps,path):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(2,5,figsize=(20,8),constrained_layout=True)
    for ax in axes.ravel():ax.set_axis_off()
    axes[0,0].imshow(image);axes[0,0].set_title('Original RGB; semantics unreviewed')
    if mask is not None:axes[1,0].imshow(mask,cmap='gray')
    axes[1,0].set_title('Original GT; no size threshold')
    lo=min(float(v.min()) for v in maps.values());hi=max(float(v.max()) for v in maps.values())
    for ax,b in zip(axes[0,1:],BRANCHES,strict=True):
        im=ax.imshow(maps[b],cmap='viridis',vmin=lo,vmax=hi)
        y,x=np.unravel_index(np.argmax(maps[b]),maps[b].shape);ax.plot(x,y,'r+')
        ax.set_title(f'{b}\nmax={maps[b][y,x]:.6g}')
    differences={ref:maps[NEW]-maps[ref] for ref in ('UNION_K8','SPARC_SOURCE_K8')}
    bound=max(1e-12,max(float(np.abs(v).max()) for v in differences.values()))
    for ax,(ref,value) in zip(axes[1,1:3],differences.items(),strict=True):
        dm=ax.imshow(value,cmap='coolwarm',vmin=-bound,vmax=bound);ax.set_title(f'{NEW} minus {ref} response')
    axes[1,3].text(0,.5,'SPARC pair: same frozen MAD.\nUNION comparison: different MAD scales.\nResponse difference is not raw distance reduction.')
    axes[1,4].text(0,.5,'Parent results preserved.\nBackground is a development domain.\nNo threshold/parameter selection.')
    fig.colorbar(im,ax=list(axes[0,1:]),shrink=.7,label='Formal response; branch-specific MAD')
    fig.colorbar(dm,ax=list(axes[1,1:3]),shrink=.7,label='Response difference')
    path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=110);plt.close(fig)
