"""Exactly the original twelve cases; same MAD and no spatial smoothing."""
from pathlib import Path
import numpy as np
from .inputs import BRANCHES


def plot_case(image, mask, maps, path: Path):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes = plt.subplots(2,5,figsize=(20,8),constrained_layout=True)
    for ax in axes.ravel(): ax.set_axis_off()
    axes[0,0].imshow(image); axes[0,0].set_title('Original RGB')
    if mask is not None: axes[1,0].imshow(mask,cmap='gray')
    axes[1,0].set_title('Original GT; evaluation only')
    low = min(float(m.min()) for m in maps.values()); high = max(float(m.max()) for m in maps.values())
    for axis,branch in zip(axes[0,1:],BRANCHES,strict=True):
        response = axis.imshow(maps[branch],cmap='viridis',vmin=low,vmax=high)
        y,x = np.unravel_index(np.argmax(maps[branch]),maps[branch].shape); axis.plot(x,y,'r+')
        axis.set_title(branch+'\nmax='+f'{maps[branch][y,x]:.6g}')
    pairs = [('UNION_TARGET512','UNION_K8'),('FASTREF_TARGET512_SOURCE','UNION_TARGET512'),('FASTREF_TARGET512_SOURCE','UNION_K8')]
    bound = max(1e-12,max(float(np.abs(maps[b]-maps[r]).max()) for b,r in pairs))
    for axis,(branch,reference) in zip(axes[1,1:4],pairs,strict=True):
        difference = axis.imshow(maps[branch]-maps[reference],cmap='coolwarm',vmin=-bound,vmax=bound)
        axis.set_title(branch+' minus '+reference+'\nMAD response difference')
    axes[1,4].text(0,.5,'All branches share original Source MAD.\nResponse difference is not raw distance reduction.\nNo inference from this plot selects parameters.\nBackground is a development domain.')
    fig.colorbar(response,ax=list(axes[0,1:]),shrink=.7)
    fig.colorbar(difference,ax=list(axes[1,1:4]),shrink=.7)
    path.parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,dpi=110); plt.close(fig)
