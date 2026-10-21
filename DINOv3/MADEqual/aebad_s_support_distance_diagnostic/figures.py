"""Fixed diagnostic CDFs and three historical cases; no selection by results."""
from __future__ import annotations
import numpy as np
from .inputs import Dataset
from .profiles import LAYERS


def case(ctx,figure,path):
    from matplotlib import pyplot as plt
    s=figure['sample']
    with Dataset(ctx) as data:
        image=data.image(s['relative_path'],ctx['capture']['image_sha256'][s['relative_path']])
    maps=figure['maps'];fig,axes=plt.subplots(2,3,figsize=(12,8),layout='constrained')
    axes[0,0].imshow(image);axes[0,0].set_title(s['sample_id']+' RGB')
    axes[0,1].imshow(figure['gt'],cmap='gray');axes[0,1].set_title('Original GT (normal: empty)')
    n=figure['native'];selection=np.zeros(n.size);selection[figure['selected_physical']]=1
    axes[0,2].imshow(selection.reshape(n.shape),cmap='gray');axes[0,2].set_title('Selected native patches')
    low=min(float(v.min()) for v in maps.values());high=max(float(v.max()) for v in maps.values())
    for ax,b in zip(axes[1,:2],('UNION','SEGMENT')):
        im=ax.imshow(maps[b],vmin=low,vmax=high,cmap='viridis');ax.set_title(b)
        fig.colorbar(im,ax=ax,shrink=.7)
        for r in figure['anchors']:
            for region,color in (('instance','red'),('ring','white')):
                x,y=r.get('UNION_anchor_'+region+'_x'),r.get('UNION_anchor_'+region+'_y')
                if x is not None:ax.plot(x,y,'+',color=color,markersize=6)
    im=axes[1,2].imshow(maps['UNION']-maps['SEGMENT'],cmap='coolwarm')
    axes[1,2].set_title('UNION - SEGMENT response');fig.colorbar(im,ax=axes[1,2],shrink=.7)
    for ax in axes.ravel():ax.set_axis_off()
    fig.savefig(path,dpi=120);plt.close(fig)


def cdfs(plots,directory):
    from matplotlib import pyplot as plt
    for feature in ('gap8','effective_support_count'):
        fig,axes=plt.subplots(2,4,figsize=(15,6),layout='constrained')
        for j,l in enumerate(LAYERS):
            p=plots.get(l)
            for k,mode in enumerate(('base_weights','matched_weights')):
                ax=axes[k,j];ax.set_title(f'L{l} '+('uncontrolled effective' if k==0 else 'matched b/delta'))
                if p:
                    x=p['data'][feature];y=p['data']['group'];weights=p[mode]
                    for group,label in ((0,'normal high response'),(1,'GT covered')):
                        keep=(y==group)&np.isfinite(x)&(weights>0)
                        if keep.any():
                            order=np.argsort(x[keep],kind='stable');v=x[keep][order];w=weights[keep][order]
                            ax.step(v,np.cumsum(w)/w.sum(),where='post',label=label)
                ax.set_xlabel(feature);ax.set_ylim(0,1);ax.grid(alpha=.2)
                if ax.lines:ax.legend(fontsize=7)
        fig.savefig(directory/(feature+'_weighted_CDF.png'),dpi=120);plt.close(fig)
