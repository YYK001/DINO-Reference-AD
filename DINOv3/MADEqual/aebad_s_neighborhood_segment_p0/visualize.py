"""Fixed twelve archived cases only. RGB never feeds prediction or scoring."""
from __future__ import annotations

import numpy as np

from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic.numerics import reconstruct
from . import inputs as ip
from .analysis import Dataset, U, S, P, G
from .prediction import fresh, verified_predictions, load_prediction, original_distances, protect_dependencies


def visualize(ctx,prediction_dir,output):
    pred,records=verified_predictions(ctx,prediction_dir)
    protect_dependencies(output,pred)
    out=fresh(ctx,output,'visualize')
    import os
    os.environ['MPLCONFIGDIR']=str(out/'_matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    samples={s['sample_id']:s for s in ctx['samples']};written=[]
    with ip.Archive(ctx['paths']['raw_archive']) as a, Dataset(ctx) as data:
        for case in ctx['capture']['selected_cases']:
            sample=samples[case['sample_id']];sid=sample['sample_id']
            raw,meta,segments=ip.load_image(ctx,a,sample)
            saved=load_prediction(ctx,pred,records[sid],raw,meta,segments)
            u,s=original_distances(raw,meta,segments)
            distances={U:u,S:s,P:saved['pair_distance'],G:saved['neighbor_distance']}
            maps={b:reconstruct(d,raw['positions'],tuple(meta['grid']),tuple(meta['output_shape']),ctx['medians'],ctx['scales'])[1]['FUSED'] for b,d in distances.items()}
            rgb=data.image(sample['relative_path'],ctx['capture']['image_sha256'][sample['relative_path']])
            gt=data.gt(sample)
            # Common linear display range only; it never normalizes saved responses.
            lo=min(float(m.min()) for m in maps.values());hi=max(float(m.max()) for m in maps.values())
            fig,axes=plt.subplots(4,4,figsize=(18,16),constrained_layout=True)
            axes[0,0].imshow(rgb);axes[0,0].set_title(sid+' RGB (display only)')
            axes[0,1].imshow(gt,cmap='gray',vmin=0,vmax=1);axes[0,1].set_title('Original GT (evaluation only)')
            axes[0,2].imshow(saved['pair_gate'].mean(0),vmin=0,vmax=1,cmap='viridis');axes[0,2].set_title('Pair acceptance: four-layer fraction')
            axes[0,3].imshow(saved['geom_gate'].mean(0),vmin=0,vmax=1,cmap='viridis');axes[0,3].set_title('Geometry acceptance: four-layer fraction')
            for ax,(branch,response) in zip(axes[1],maps.items(),strict=True):
                rendered=ax.imshow(response,cmap='inferno',vmin=lo,vmax=hi);ax.set_title(branch);fig.colorbar(rendered,ax=ax,shrink=.6)
            for off,layer in enumerate((7,15,23,31)):
                axes[2,off].imshow(saved['pair_gate'][off],vmin=0,vmax=1,cmap='gray');axes[2,off].set_title(f'L{layer} pair gate (native grid)')
                axes[3,off].imshow(saved['geom_gate'][off],vmin=0,vmax=1,cmap='gray');axes[3,off].set_title(f'L{layer} geometry gate (native grid)')
            for ax in axes.flat: ax.axis('off')
            file=sid+'.png';fig.savefig(out/file,dpi=100);plt.close(fig);written.append(file)
    verified_predictions(ctx,pred)
    if ip.context(ctx['paths'])['identity']!=ctx['identity']: raise ValueError('protected inputs changed during visualization')
    ip.write_json(out/'status.json',dict(tool=ip.TOOL,stage='visualize',status='fixed_cases_visualized',
                                        identity=ctx['identity'],case_ids=[c['sample_id'] for c in ctx['capture']['selected_cases']],
                                        files={n:ip.sha(out/n) for n in written},RGB_used_only_for_display=True))
    return out
