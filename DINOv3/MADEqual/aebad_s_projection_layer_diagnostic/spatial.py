"""Reuse GT instances/rings; new same-position decomposition, never new masks."""
from __future__ import annotations
import numpy as np
from .numerics import LAYERS,RESPONSES,adaptive_max,check
from DINOv3.MADEqual.aebad_s_union_diagnostics.spatial import components,instance_regions


def decompose(u,p,anchors,inside,ring):
    xi,xr=anchors
    if not inside[xi] or not ring[xr]:raise ValueError('fixed original UNION peak outside its instance/ring')
    contributions={l:float((float(p[l][xi])-float(u[l][xi])-float(p[l][xr])+float(u[l][xr]))/4) for l in LAYERS}
    e_in=float(p['FUSED'][inside].max())-float(p['FUSED'][xi])
    e_ring=float(p['FUSED'][ring].max())-float(p['FUSED'][xr])
    margin_delta=float(p['FUSED'][inside].max())-float(p['FUSED'][ring].max())-float(u['FUSED'][inside].max())+float(u['FUSED'][ring].max())
    result=dict(delta_margin=margin_delta,fixed_position_change=sum(contributions.values()),e_in=e_in,e_ring=e_ring,
        reconstructed_delta=sum(contributions.values())+e_in-e_ring,
        identity_error=margin_delta-(sum(contributions.values())+e_in-e_ring))
    return contributions,result


def analyze_instances(sample,maps,mask,grid,historical,anchors,small_ids):
    if not sample['label']:return [],[],[],[]
    labels,descriptors=components(mask)
    if not descriptors:raise ValueError('empty anomaly GT cannot complete diagnosis')
    rows=[];contributions=[];movement=[];checks=[];sid=sample['sample_id']
    for desc in descriptors:
        i=desc['instance_id'];component=labels==i
        inside,ring=instance_regions(component,mask,maps['U']['FUSED'].shape)
        coverage=int(adaptive_max(component,grid).sum())
        status='mapped_instance_disappeared' if not inside.any() else ('empty_ring' if not ring.any() else 'available')
        for branch in ('U','P'):
            for response in RESPONSES:
                value=maps[branch][response]
                im=float(value[inside].max()) if inside.any() else None
                rm=float(value[ring].max()) if ring.any() else None
                def peak(region):
                    if not region.any():return (None,None)
                    return tuple(int(v) for v in np.unravel_index(np.argmax(np.where(region,value,-np.inf)),value.shape))
                yi,xi=peak(inside);yr,xr=peak(ring)
                row=dict(sample_id=sid,relative_path=sample['relative_path'],**desc,branch=branch,response=response,
                    patch_coverage=coverage,instance_area_output_pixels=int(inside.sum()),ring_area_output_pixels=int(ring.sum()),
                    instance_max=im,ring_max=rm,margin=im-rm if im is not None and rm is not None else None,
                    own_instance_peak_y=yi,own_instance_peak_x=xi,own_ring_peak_y=yr,own_ring_peak_x=xr,
                    response_status=status,original_smallest_three_rule=(sid,i) in small_ids,
                    ring_radius_original_pixels=16,ring_metric='Chebyshev',ring_excludes_all_GT=True,
                    unit='original_Source_MAD_response',peak_tie_rule='first_row_major')
                rows.append(row)
                if response=='FUSED':
                    old=historical[branch][sid,i]
                    if old['response_status']!=status:raise ValueError('historical instance mapping status changed')
                    for key in ('instance_max','ring_max','margin'):
                        a,b=row[key],old[key]
                        if (a is None)!=(b==''):raise ValueError('historical missing response changed')
                        if a is not None:check(checks,sid,'historical_instance_'+key,a,float(b),branch=branch,instance_id=i)
                    if int(old['patch_coverage'])!=coverage:raise ValueError('adaptive patch GT coverage changed')
        if status!='available':
            for l in LAYERS:contributions.append(dict(sample_id=sid,instance_id=i,layer=l,c_l=None,response_status=status,original_smallest_three_rule=(sid,i) in small_ids))
            movement.append(dict(sample_id=sid,instance_id=i,response_status=status,delta_margin=None,e_in=None,e_ring=None,original_smallest_three_rule=(sid,i) in small_ids))
            continue
        frozen=anchors[sid,i]
        positions=tuple((int(frozen[role]['reference_peak_y']),int(frozen[role]['reference_peak_x'])) for role in ('instance','ring'))
        for pos,region,role in zip(positions,(inside,ring),('instance','ring'),strict=True):
            if not region[pos]:raise ValueError('archived fixed peak is outside original region')
            check(checks,sid,'original_fixed_UNION_peak_response',maps['U']['FUSED'][pos],float(frozen[role]['score_at_fixed_reference_peak']),instance_id=i,region=role)
            check(checks,sid,'original_fixed_peak_still_at_max',maps['U']['FUSED'][pos],maps['U']['FUSED'][region].max(),instance_id=i,region=role)
        c,move=decompose(maps['U'],maps['P'],positions,inside,ring)
        xi,xr=positions
        for l in LAYERS:
            contributions.append(dict(sample_id=sid,instance_id=i,layer=l,c_l=c[l],response_status='available',
                original_smallest_three_rule=(sid,i) in small_ids,x_in_y=xi[0],x_in_x=xi[1],x_ring_y=xr[0],x_ring_x=xr[1],
                P_minus_U_at_instance=float(maps['P'][l][xi])-float(maps['U'][l][xi]),
                P_minus_U_at_ring=float(maps['P'][l][xr])-float(maps['U'][l][xr]),unit='one_quarter_Source_MAD_response',direction='P_minus_U'))
        movement.append(dict(sample_id=sid,instance_id=i,response_status='available',**move,
            fixed_UNION_instance_peak_y=xi[0],fixed_UNION_instance_peak_x=xi[1],fixed_UNION_ring_peak_y=xr[0],fixed_UNION_ring_peak_x=xr[1]))
        check(checks,sid,'margin_same_position_plus_peak_movement',move['delta_margin'],move['reconstructed_delta'],instance_id=i)
    by={(r['instance_id'],r['response'],r['branch']):r for r in rows}
    for row in rows:
        u=by[row['instance_id'],row['response'],'U'];v=row['margin'];base=u['margin']
        row['margin_change_vs_U']=v-base if v is not None and base is not None else None
    return rows,contributions,movement,checks
