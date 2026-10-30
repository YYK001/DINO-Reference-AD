"""Descriptive contributions only. Formal maps/scores use the old frozen kernel."""
import math
import numpy as np
from DINOv3.MADEqual.aebad_s_target_loo_layer_weight_p0 import numerics as kernel
from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.calibration import from_z
from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic.numerics import bilinear
from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import image_metrics
from DINOv3.MADEqual.aebad_s_union_diagnostics.spatial import components,instance_regions
from .protocol import *


def responses(z,positions,grid,shape,weights):
    images={};scores={}
    for w in WEIGHTS:images[w],scores[w]=kernel.weighted(z,positions,grid,shape,weights[w])
    _,p,q=from_z(z,positions,grid,shape)
    return images,scores,np.stack([np.quantile(z,.99,axis=1,method='linear'),p,q],axis=1)


def per_layer_maps(z,positions,grid,shape):
    physical=np.empty_like(z);physical[:,positions]=z
    return np.stack([bilinear(x,shape) for x in physical.reshape(4,*grid)])


def checked_residual(actual,parts,kind):
    residual=np.asarray(actual,np.float64)-np.asarray(parts,np.float64)
    error=float(np.max(np.abs(residual),initial=0))
    if not np.isfinite(residual).all() or error>SCORE_ATOL:raise ValueError(kind+' decomposition differs; no tolerance relaxation')
    return error


def pair_analysis(samples,scores,stats,weights):
    y=np.asarray([r['label'] for r in samples]);normal=np.flatnonzero(y==0);anomaly=np.flatnonzero(y==1)
    if not len(normal) or not len(anomaly):raise ValueError('both image classes required')
    # A-major/N-minor order; exact signs of the frozen scalar scores, no tie epsilon.
    ai=np.repeat(anomaly,len(normal));ni=np.tile(normal,len(anomaly));summaries=[];contributions=[];compact={};verifications=[]
    for new,old in COMPARISONS:
        key=new+'_minus_'+old;old_scores=np.asarray([s[old][READOUTS[0]] for s in scores],np.float64);new_scores=np.asarray([s[new][READOUTS[0]] for s in scores],np.float64)
        before=old_scores[ai]-old_scores[ni];after=new_scores[ai]-new_scores[ni]
        state0=np.sign(before).astype(np.int8);state1=np.sign(after).astype(np.int8)
        dw=np.asarray(weights[new],np.float32).astype(np.float64)-np.asarray(weights[old],np.float32).astype(np.float64)
        differences=np.asarray(stats[ai,:,1:3],np.float64)-np.asarray(stats[ni,:,1:3],np.float64)
        parts=.5*differences*dw[None,:,1:3]
        error=checked_residual(after-before,parts.sum(axis=(1,2)),key)
        for label,s in ((old,old_scores),(new,new_scores)):
            h=s[ai]-s[ni];pair_auroc=float(((h>0)+.5*(h==0)).mean());official=image_metrics(y,s)['image_AUROC']
            if abs(pair_auroc-official)>4*np.finfo(np.float64).eps:raise ValueError('pair correctness and official Image AUROC disagree')
            verifications.append(dict(comparison=key,check='exact_pair_AUROC_'+label,actual=pair_auroc,expected=official,max_abs_error=abs(pair_auroc-official),verified=True))
        compact[key]=dict(anomaly_index=ai.astype(np.int16),normal_index=ni.astype(np.int16),old_gap=before,new_gap=after,old_state=state0,new_state=state1,layer_pq_contributions=parts,rounding_residual=after-before-parts.sum(axis=(1,2)))
        for a in (-1,0,1):
            for b in (-1,0,1):summaries.append(dict(comparison=key,old_state=a,new_state=b,pair_count=int(((state0==a)&(state1==b)).sum()),total_pairs=len(ai),state_definition='-1=wrong;0=exact_tie;1=correct',compact_key=key))
        for group,take in [('all',np.ones(len(ai),bool)),('repaired',(state0==-1)&(state1==1)),('damaged',(state0==1)&(state1==-1))]:
            for j,l in enumerate(LAYERS):
                for k,name in enumerate(('p','q')):contributions.append(dict(comparison=key,group=group,layer=l,component=name,pairs=int(take.sum()),contribution_sum=float(parts[take,j,k].sum()) if take.any() else None,contribution_mean=float(parts[take,j,k].mean()) if take.any() else None,interpretation='score_gap_contribution_not_additive_AUROC_or_AP'))
        for population,ids in (('normal',normal),('anomaly',anomaly)):
            change=new_scores[ids]-old_scores[ids]
            summaries.append(dict(comparison=key,population=population,images=len(ids),mean_old_score=float(old_scores[ids].mean()),mean_new_score=float(new_scores[ids].mean()),mean_score_change=float(change.mean()),median_score_change=float(np.median(change)),compact_key=key))
        verifications.append(dict(comparison=key,check='linear_PL_gap_sum',max_abs_error=error,atol=SCORE_ATOL,verified=True))
    return summaries,contributions,compact,verifications


def peak(image,mask=None):
    if mask is None:return tuple(int(x) for x in np.unravel_index(np.argmax(image),image.shape))
    ids=np.flatnonzero(mask)
    if not len(ids):return None
    return tuple(int(x) for x in np.unravel_index(ids[np.argmax(image.ravel()[ids])],image.shape))


def normal_tail(images):
    source=images['WS'];anchor=peak(source);flat=source.ravel();n=max(1,math.ceil(.01*flat.size))
    ids=np.argsort(-flat,kind='stable')[:n]  # Ascending physical flat index breaks equal scores.
    rows=[]
    for new,old in COMPARISONS:
        a,b=images[old],images[new];pa,pb=peak(a),peak(b)
        rows.append(dict(comparison=new+'_minus_'+old,reference='same_memory_WS',anchor_origin='derived_in_this_audit',anchor_y=anchor[0],anchor_x=anchor[1],top_pixels=n,
            old_at_WS_peak=float(a[anchor]),new_at_WS_peak=float(b[anchor]),fixed_peak_change=float(b[anchor])-float(a[anchor]),
            old_WS_top1_mean=float(a.ravel()[ids].mean(dtype=np.float64)),new_WS_top1_mean=float(b.ravel()[ids].mean(dtype=np.float64)),
            fixed_top1_mean_change=float((b.ravel()[ids].astype(np.float64)-a.ravel()[ids]).mean()),
            old_own_peak=float(a[pa]),new_own_peak=float(b[pb]),own_peak_change=float(b[pb])-float(a[pa]),old_own_y=pa[0],old_own_x=pa[1],new_own_y=pb[0],new_own_x=pb[1],
            own_peak_moved=pa!=pb,new_peak_moved_from_WS=pb!=anchor,interpretation='normal_high_response_not_threshold_FPR'))
    return rows


def margin_group(value):return 'NA' if value is None else 'improved' if value>MARGIN_ATOL else 'decreased' if value < -MARGIN_ATOL else 'unchanged'


def instance_changes(images,layer_maps,gt,weights):
    labels,descriptors=components(gt);rows=[];checks=[]
    for descriptor in descriptors:
        inside,ring=instance_regions(labels==descriptor['instance_id'],gt,images['WS'].shape)
        anchors=(peak(images['WS'],inside),peak(images['WS'],ring))
        status='instance_empty_at_evaluation_grid' if anchors[0] is None else 'ring_empty_at_evaluation_grid' if anchors[1] is None else 'available'
        own={w:(peak(image,inside),peak(image,ring)) for w,image in images.items()}
        for new,old in COMPARISONS:
            row=dict(descriptor,comparison=new+'_minus_'+old,status=status,anchor_origin='derived_in_this_audit_from_same_memory_WS',instance_output_pixels=int(inside.sum()),ring_output_pixels=int(ring.sum()),ring_radius_original_pixels=16,ring_excludes_all_GT=True)
            for w in (old,new):
                image=images[w];pi,pr=own[w]
                row[w+'_own_inside']=float(image[pi]) if pi is not None else None;row[w+'_own_ring']=float(image[pr]) if pr is not None else None
                row[w+'_own_margin']=row[w+'_own_inside']-row[w+'_own_ring'] if pi is not None and pr is not None else None
                row[w+'_inside_peak']=pi;row[w+'_ring_peak']=pr
            for name,pixel in zip(('inside','ring'),anchors):
                row['WS_anchor_'+name]=pixel
                row['fixed_'+name+'_response_change']=float(images[new][pixel])-float(images[old][pixel]) if pixel is not None else None
                row[name+'_peak_moved']=own[new][0 if name=='inside' else 1]!=own[old][0 if name=='inside' else 1]
            if status=='available':
                pin,pr=anchors
                before=float(images[old][pin])-float(images[old][pr]);after=float(images[new][pin])-float(images[new][pr]);change=after-before
                dw=np.asarray(weights[new],np.float32)[:,0].astype(np.float64)-np.asarray(weights[old],np.float32)[:,0].astype(np.float64)
                parts=dw*(layer_maps[:,pin[0],pin[1]].astype(np.float64)-layer_maps[:,pr[0],pr[1]].astype(np.float64))
                error=checked_residual(change,parts.sum(),'fixed_instance_margin')
                row.update(old_fixed_margin=before,new_fixed_margin=after,fixed_margin_change=change,own_margin_change=row[new+'_own_margin']-row[old+'_own_margin'],rounding_residual=change-float(parts.sum()),**{f'contribution_L{l}':float(parts[j]) for j,l in enumerate(LAYERS)})
                checks.append(dict(instance_id=descriptor['instance_id'],comparison=row['comparison'],check='fixed_margin_layer_sum',max_abs_error=error,atol=SCORE_ATOL,verified=True))
            else:row.update(old_fixed_margin=None,new_fixed_margin=None,fixed_margin_change=None,own_margin_change=None,rounding_residual=None,**{f'contribution_L{l}':None for l in LAYERS})
            row.update(fixed_margin_group=margin_group(row['fixed_margin_change']),own_margin_group=margin_group(row['own_margin_change']),margin_atol=MARGIN_ATOL,interpretation='margin_damage_not_missed_detection_no_native_distance_RN_RA')
            rows.append(row)
    return rows,checks


def distribution(values,prefix):
    x=np.asarray(values,np.float64)
    return {prefix+'_n':len(x),**{prefix+'_'+k:float(v) if len(x) else None for k,v in dict(mean=x.mean() if len(x) else np.nan,median=np.median(x) if len(x) else np.nan,p95=np.quantile(x,.95) if len(x) else np.nan,p99=np.quantile(x,.99) if len(x) else np.nan).items()}}
