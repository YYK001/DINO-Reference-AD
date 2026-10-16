"""Original fixed cases, five shared-scale maps and three signed differences."""
import numpy as np
from .inputs import BRANCHES, BRANCH, REFERENCES


def plot_case(image,mask,maps,path):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(2,6,figsize=(24,8),constrained_layout=True)
    for ax in axes.ravel(): ax.set_axis_off()
    axes[0,0].imshow(image);axes[0,0].set_title('Original RGB')
    if mask is not None: axes[1,0].imshow(mask,cmap='gray')
    axes[1,0].set_title('GT after predictions frozen')
    low=min(float(v.min()) for v in maps.values());high=max(float(v.max()) for v in maps.values())
    for ax,b in zip(axes[0,1:],BRANCHES,strict=True):
        color=ax.imshow(maps[b],cmap='viridis',vmin=low,vmax=high)
        y,x=np.unravel_index(np.argmax(maps[b]),maps[b].shape);ax.plot(x,y,'r+')
        ax.set_title(b+'\nmax='+f'{maps[b][y,x]:.6g}')
    bound=max(1e-12,max(float(np.abs(maps[BRANCH]-maps[r]).max()) for r in REFERENCES))
    for ax,r in zip(axes[1,1:4],REFERENCES,strict=True):
        delta=ax.imshow(maps[BRANCH]-maps[r],cmap='coolwarm',vmin=-bound,vmax=bound)
        ax.set_title('Projection minus '+r+'\nSource MAD response')
    axes[1,4].text(0,.5,'Shared original Source MAD.\nSigned differences, no clipping.\nNo parameter selection from cases.')
    axes[1,5].text(0,.5,'Historical four metric rows reused.\nOnly Projection newly evaluated.\nDevelopment-domain evidence only.')
    fig.colorbar(color,ax=list(axes[0,1:]),shrink=.7);fig.colorbar(delta,ax=list(axes[1,1:4]),shrink=.7)
    path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=110);plt.close(fig)
