"""Fixed twelve cases; shared map scale and a separate shared difference scale."""
from pathlib import Path
import numpy as np
from .geometry import BRANCHES


def plot_case(image, mask, maps: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    lo=min(float(m.min()) for m in maps.values());hi=max(float(m.max()) for m in maps.values())
    diff={b:maps[b]-maps[BRANCHES[0]] for b in BRANCHES[1:]}
    bound=max(1e-12,max(float(np.abs(m).max()) for m in diff.values()))
    fig,axes=plt.subplots(2,4,figsize=(16,8),constrained_layout=True)
    for ax in axes.ravel():ax.set_axis_off()
    axes[0,0].imshow(image);axes[0,0].set_title('Query RGB; semantics unreviewed')
    if mask is not None:axes[1,0].imshow(mask,cmap='gray')
    axes[1,0].set_title('Original GT; complement is not background')
    for ax,b in zip(axes[0,1:],BRANCHES,strict=True):
        im=ax.imshow(maps[b],cmap='viridis',vmin=lo,vmax=hi)
        y,x=np.unravel_index(np.argmax(maps[b]),maps[b].shape)
        ax.plot(x,y,'r+',markersize=9);ax.set_title(f'{b}\nmax={maps[b][y,x]:.7g}, xy=({x},{y})')
    for ax,(b,d) in zip(axes[1,1:3],diff.items(),strict=True):
        dm=ax.imshow(d,cmap='coolwarm',vmin=-bound,vmax=bound);ax.set_title(b+' minus K8_UNION')
    axes[1,3].text(0,.5,'Red + marks each branch peak.\nAll branches retain original union.\nNo semantic correspondence claim.')
    fig.colorbar(im,ax=list(axes[0,1:]),shrink=.7,label='Shared Source MAD-Equal response scale')
    fig.colorbar(dm,ax=list(axes[1,1:3]),shrink=.7,label='Independent shared difference scale')
    path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=110);plt.close(fig)
