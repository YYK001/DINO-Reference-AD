"""Fixed cases; own-MAD responses explicitly distinguished from raw distances."""
from pathlib import Path
import numpy as np
from .projection import BRANCHES


def plot_case(image,mask,maps,path:Path):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    lo=min(float(m.min()) for m in maps.values());hi=max(float(m.max()) for m in maps.values())
    diffs={b:maps[b]-maps['UNION_K8'] for b in ('SOURCE','SPARC_SOURCE_K8')}
    bound=max(1e-12,max(float(np.abs(d).max()) for d in diffs.values()))
    fig,axes=plt.subplots(2,4,figsize=(16,8),constrained_layout=True)
    for ax in axes.ravel(): ax.set_axis_off()
    axes[0,0].imshow(image);axes[0,0].set_title('Original RGB; semantics unreviewed')
    if mask is not None: axes[1,0].imshow(mask,cmap='gray')
    axes[1,0].set_title('Original GT; no threshold selection')
    for ax,b in zip(axes[0,1:],BRANCHES,strict=True):
        im=ax.imshow(maps[b],cmap='viridis',vmin=lo,vmax=hi)
        y,x=np.unravel_index(np.argmax(maps[b]),maps[b].shape);ax.plot(x,y,'r+')
        ax.set_title(f'{b}\nmax={maps[b][y,x]:.6g}; own MAD')
    for ax,(b,d) in zip(axes[1,1:3],diffs.items(),strict=True):
        dm=ax.imshow(d,cmap='coolwarm',vmin=-bound,vmax=bound);ax.set_title(b+' minus UNION_K8 response')
    axes[1,3].text(0,.5,'SPARC uses refitted Source MAD.\nResponse difference is not raw distance change.\nBackground is a development domain.')
    fig.colorbar(im,ax=list(axes[0,1:]),shrink=.7,label='Formal response; different branch MAD scales')
    fig.colorbar(dm,ax=list(axes[1,1:3]),shrink=.7,label='Response difference, not distance reduction')
    path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=110);plt.close(fig)
