"""Original GT instances/rings and all interpolation contributors, CPU only."""
from __future__ import annotations
import numpy as np
from DINOv3.MADEqual.aebad_s_union_diagnostics.spatial import (
    components,instance_regions,interpolation_stencil,
)
from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic.numerics import adaptive_max,check,LAYERS
from .inputs import peak_position
from .profiles import FEATURES,summary_at_stencil


def scalar_profile(profile,offset):
    return {k:(v[offset].item() if isinstance(v[offset],np.generic) else v[offset]) for k,v in profile.items()}


def instance_analysis(ctx,sample,gt,maps,raw,meta,profiles,checks,write_link,write_stencil):
    sid=sample['sample_id'];grid=tuple(meta['grid']);shape=tuple(meta['output_shape'])
    labelled,descriptors=components(gt)
    rows=[];inverse=np.empty(len(raw['positions']),np.int64)
    inverse[raw['positions']]=np.arange(len(inverse))
    gt_union=np.zeros(grid,bool)
    for desc in descriptors:
        iid=int(desc['instance_id']);key=(sid,iid);component=labelled==iid
        covered=adaptive_max(component,grid);gt_union|=covered
        small=key in ctx['history']['small3']
        for physical in np.flatnonzero(covered):
            write_link(dict(sample_id=sid,instance_id=iid,physical_patch=int(physical),
                patch_y=int(physical//grid[1]),patch_x=int(physical%grid[1]),layers='7,15,23,31',
                small_original_three=small,area_original_pixels=desc['area_original_pixels']))
        inside,ring=instance_regions(component,gt,shape)
        regions=dict(instance=inside,ring=ring)
        status='mapped_instance_disappeared' if not inside.any() else 'empty_ring' if not ring.any() else 'available'
        row=dict(sample_id=sid,relative_path=sample['relative_path'],instance_id=iid,**{k:v for k,v in desc.items() if k!='instance_id'},
            patch_coverage=int(covered.sum()),instance_area_output_pixels=int(inside.sum()),ring_area_output_pixels=int(ring.sum()),
            small_original_three=small,response_status=status,ring_original_pixels=16,ring_excludes_all_GT=True)
        anchors={};own={};values={}
        for name,mask in regions.items():
            anchor=ctx['history']['peaks'].get((sid,name,iid));anchors[name]=anchor
            if mask.any():
                if anchor is None or not mask[anchor]:raise ValueError('historical UNION anchor missing or outside region')
                check(checks,sid,'historical_UNION_peak_is_max',maps['UNION'][anchor],maps['UNION'][mask].max(),instance_id=iid,region=name)
            elif anchor is not None:raise ValueError('historical anchor exists in empty region')
            for branch in ('UNION','SEGMENT'):
                p=peak_position(maps[branch],mask);own[branch,name]=p
                values[branch,name]=float(maps[branch][p]) if p is not None else None
                row[f'{branch}_own_{name}_y']=p[0] if p else None
                row[f'{branch}_own_{name}_x']=p[1] if p else None
                row[f'{branch}_own_{name}_response']=values[branch,name]
                row[f'{branch}_fixed_{name}_response']=float(maps[branch][anchor]) if anchor is not None else None
            row[f'UNION_anchor_{name}_y']=anchor[0] if anchor else None
            row[f'UNION_anchor_{name}_x']=anchor[1] if anchor else None
        for branch,history_branch in (('UNION','K8_UNION'),('SEGMENT','K8_SEGMENT')):
            available=all(values[branch,r] is not None for r in regions)
            margin=values[branch,'instance']-values[branch,'ring'] if available else None
            fixed=row[f'{branch}_fixed_instance_response']-row[f'{branch}_fixed_ring_response'] if status=='available' else None
            row[branch+'_own_margin']=margin;row[branch+'_fixed_margin']=fixed
            history=ctx['history']['instances'][sid,iid,history_branch]
            for field in ('area_original_pixels','patch_coverage','instance_area_output_pixels','ring_area_output_pixels'):
                check(checks,sid,'historical_'+field,row[field],int(history[field]),atol=0,instance_id=iid,branch=branch)
            for field,actual in (('instance_max',values[branch,'instance']),('ring_max',values[branch,'ring']),('margin',margin)):
                if actual is None:
                    if history[field] not in ('',None):raise ValueError('historical NA/instance computability changed')
                else:
                    check(checks,sid,'historical_'+field,actual,float(history[field]),instance_id=iid,branch=branch)
                    optional=ctx['history'].get('optional_instances',{}).get((sid,iid,history_branch))
                    if optional is not None:
                        check(checks,sid,'optional_neighbor_'+field,actual,float(optional[field]),instance_id=iid,branch=branch)
        if status=='available':
            di=row['UNION_fixed_instance_response']-row['SEGMENT_fixed_instance_response']
            dr=row['UNION_fixed_ring_response']-row['SEGMENT_fixed_ring_response']
            fixed_change=row['SEGMENT_fixed_margin']-row['UNION_fixed_margin']
            row.update(inside_response_reduction=di,ring_response_reduction=dr,fixed_margin_change=fixed_change,
                own_margin_change=row['SEGMENT_own_margin']-row['UNION_own_margin'],
                decomposition_ring_minus_inside=dr-di,
                fixed_change_group='inside_reduction_larger' if fixed_change < -1e-6 else 'ring_reduction_larger' if fixed_change>1e-6 else 'close',
                own_margin_group='decreased' if row['SEGMENT_own_margin']<row['UNION_own_margin']-1e-6 else 'not_decreased')
            check(checks,sid,'fixed_margin_decomposition',fixed_change,dr-di,atol=1e-5,instance_id=iid)
        else:
            row.update(fixed_margin_change=None,own_margin_change=None,fixed_change_group=status,own_margin_group=status)
        # Four roles preserve every participating patch, even when two roles coincide.
        for branch in ('UNION','SEGMENT'):
            for region in regions:
                pixel=anchors[region] if branch=='UNION' else own[branch,region]
                if pixel is None:continue
                stencil=interpolation_stencil(pixel,grid,shape)
                weights=np.asarray([w for _,_,w in stencil],np.float64)
                offsets=np.asarray([inverse[y*grid[1]+x] for y,x,_ in stencil],np.int64)
                if not np.isclose(weights.sum(),1,atol=1e-12):raise ValueError('incomplete interpolation stencil')
                responses={name:0. for name in ('UNION','SEGMENT')}
                for j,l in enumerate(LAYERS):
                    profile=profiles[j]
                    for role,key in (('UNION','union'),('SEGMENT','segment')):
                        z=((profile[key]-ctx['medians'][j])/ctx['scales'][j]).astype(np.float32)
                        responses[role]+=.25*float(np.dot(weights,z[offsets]))
                    if branch=='UNION':
                        summary=summary_at_stencil(profile,offsets,weights)
                        row.update({f'fixed_{region}_L{l}_{k}':v for k,v in summary.items()})
                    for (y,x,w),offset in zip(stencil,offsets):
                        write_stencil(dict(sample_id=sid,instance_id=iid,peak_of_branch=branch,region=region,
                            output_y=pixel[0],output_x=pixel[1],layer=l,physical_patch=y*grid[1]+x,
                            patch_y=y,patch_x=x,interpolation_weight=w,layer_fusion_weight=.25,
                            own_margin_group=row['own_margin_group'],fixed_change_group=row['fixed_change_group'],
                            **scalar_profile(profile,offset)))
                for role in responses:
                    check(checks,sid,'stencil_reconstruct_'+role,responses[role],float(maps[role][pixel]),
                          instance_id=iid,peak_of_branch=branch,region=region)
        rows.append(row)
    check(checks,sid,'GT_union_of_instance_patch_coverage',gt_union,adaptive_max(gt,grid),atol=0)
    return rows
