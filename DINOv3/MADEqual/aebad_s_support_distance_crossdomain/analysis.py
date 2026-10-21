"""Domain adapters around the unchanged background statistical/spatial kernels."""
from __future__ import annotations
import numpy as np
from DINOv3.MADEqual.aebad_s_support_distance_diagnostic import profiles as bp,spatial as bs
from DINOv3.MADEqual.aebad_s_support_distance_diagnostic.analysis import STAT_FIELDS,instance_summaries
from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import statistics,ch1_score
from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic.numerics import reconstruct,adaptive_max,check
from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.verification import historical_score
from . import inputs as ip


def scoped(row,domain):
    row=dict(row)
    if 'small_original_three' in row:row['smallest_area_quartile']=row.pop('small_original_three')
    return dict(domain=domain,**row) if 'domain' not in row else row


def spatial_context(parent,sample,gt,maps):
    domain,sid=sample['domain'],sample['sample_id'];instances={};peaks={};small=set();origins={}
    labelled=None
    for key,ref in parent.instances.items():
        d,s,i,b=key
        if (d,s)!=(domain,sid) or b not in ('UNION_K8_CH1','SEGMENT_K8_CH1'):continue
        instances[s,i,'K8_UNION' if b=='UNION_K8_CH1' else 'K8_SEGMENT']=ref
        if str(ref.get('smallest_area_quartile','False')).lower()=='true':small.add((s,i))
        if b!='UNION_K8_CH1':continue
        for region in ('instance','ring'):
            y,x=ref.get('union_anchor_'+region+'_y'),ref.get('union_anchor_'+region+'_x')
            if y not in (None,'') and x not in (None,''):
                peaks[s,region,i]=(int(y),int(x));origins[i,region]='parent_CPU_saved_coordinate'
            elif 'union_anchor_'+region+'_y' not in ref:
                if labelled is None:labelled,_=bs.components(gt)
                inside,ring=bs.instance_regions(labelled==i,gt,maps['UNION'].shape)
                position=bs.peak_position(maps['UNION'],inside if region=='instance' else ring)
                if position is not None:peaks[s,region,i]=position
                origins[i,region]='derived_this_analysis_from_verified_UNION_first_row_major'
            else:origins[i,region]='parent_CPU_NA'
    # This legacy key is internal only. Its value is the original domain quartile,
    # never background's list of three instances. Every exported column is renamed.
    ctx=dict(medians=parent.medians,scales=parent.scales,history=dict(instances=instances,peaks=peaks,small3=small))
    return ctx,origins


def one_image(parent,sample,record,blob,data_path,image_id):
    raw,meta=ip.decode(blob,parent,sample,record);domain=sample['domain'];sid=sample['sample_id'];checks=[]
    union=np.minimum(raw['source_distance'],raw['target_distance'].min(0));segment=raw['segment_distance']
    maps={};native=None
    for branch,values in (('SOURCE',raw['source_distance']),('UNION',union),('SEGMENT',segment)):
        current,peak,local=statistics(values,raw['positions'],tuple(meta['grid']),tuple(meta['output_shape']),parent.medians,parent.scales)
        if branch=='SOURCE':
            check(checks,sid,'same_cache_SOURCE_CURRENT',current,parent.source_scores[domain,sid],domain=domain,check_class='same_cache_reconstruction')
            historical_score(checks,sample,'earlier_formal_SOURCE_CURRENT',current,sample['source_score'])
        else:
            ref=parent.scores[domain,sid,branch+'_K8_CH1']
            check(checks,sid,'same_cache_'+branch+'_CURRENT',current,float(ref['CURRENT_check_score']),domain=domain,check_class='same_cache_reconstruction')
            check(checks,sid,'same_cache_'+branch+'_CH1',ch1_score(peak,local,parent.state),float(ref['image_score']),domain=domain,check_class='same_cache_reconstruction')
            if branch=='UNION':historical_score(checks,sample,'earlier_formal_UNION_CURRENT',current,sample['union_score'])
            z,m,_=reconstruct(values,raw['positions'],tuple(meta['grid']),tuple(meta['output_shape']),parent.medians,parent.scales)
            maps[branch]=m['FUSED']
            if branch=='UNION':native=z.mean(axis=0)
    for r in checks:
        if r.get('historical_score_check'):r['check_class']='earlier_formal_compatibility_strict_and_accepted_separate'
    if any(not r['verified'] for r in checks):return dict(domain=domain,checks=checks,failed=True)
    with ip.Data(data_path) as data:gt=data.gt(sample)
    physical=bp.high_response(native) if not sample['label'] else np.flatnonzero(adaptive_max(gt,tuple(meta['grid'])))
    inverse=np.empty(len(raw['positions']),np.int64);inverse[raw['positions']]=np.arange(len(inverse));offsets=inverse[physical]
    profiles=[];columns=[];selected=[]
    group=int(sample['label']);count=ip.COUNTS[domain][group]
    for j,l in enumerate(ip.LAYERS):
        p=bp.native(raw['source_distance'][j],raw['target_distance'][:,j],segment[j],parent.medians[j],parent.scales[j])
        for k in ('alpha','endpoint_length'):p[k]=raw[k][j] if k in raw else np.full(len(inverse),np.nan)
        for k in ('endpoint_support','endpoint_index'):
            for t in (0,1):p[k+'_'+str(t+1)]=raw[k][j,t] if k in raw else np.full(len(inverse),np.nan)
        profiles.append(p)
        col={k:np.asarray(p[k])[offsets].copy() for k in (*bp.FEATURES,'b','delta','effective_reduction')}
        col.update(group=np.full(len(offsets),group,np.int8),image_id=np.full(len(offsets),image_id,np.int32));columns.append(col)
        for physical_index,offset in zip(physical,offsets):
            selected.append(dict(domain=domain,sample_id=sid,image_id=image_id,relative_path=sample['relative_path'],layer=l,
                physical_patch=int(physical_index),compact_offset=int(offset),group='GT_covered' if group else 'normal_high_response',
                unconditional_image_balanced_weight=1/(count*len(offsets)),
                geometry_origin='saved_capture' if 'alpha' in raw else 'not_saved_no_reconstruction',
                effective_support_status='available' if np.isfinite(p['effective_support_count'][offset]) else 'no_positive_target_improvement',
                **bs.scalar_profile(p,offset)))
    instances=[];links=[];stencils=[]
    if group:
        ctx,origins=spatial_context(parent,sample,gt,maps)
        instances=bs.instance_analysis(ctx,sample,gt,maps,raw,meta,profiles,checks,links.append,stencils.append)
        for r in instances:
            r['UNION_instance_coordinate_origin']=origins[r['instance_id'],'instance']
            r['UNION_ring_coordinate_origin']=origins[r['instance_id'],'ring']
            for branch in ('UNION','SEGMENT'):
                for region in ('instance','ring'):
                    p=(r.get(f'{branch}_own_{region}_y'),r.get(f'{branch}_own_{region}_x'))
                    q=(r.get(f'UNION_anchor_{region}_y'),r.get(f'UNION_anchor_{region}_x'))
                    r[f'{branch}_{region}_own_peak_moved_from_UNION_anchor']=p!=q if p[0] is not None and q[0] is not None else None
        # Verify the historical fixed-position scalar as well, not only own maxima.
        for r in instances:
            for branch,b in (('UNION','UNION_K8_CH1'),('SEGMENT','SEGMENT_K8_CH1')):
                ref=parent.instances[domain,sid,r['instance_id'],b]
                for new,old in (('fixed_instance_response','fixed_union_instance_response'),('fixed_ring_response','fixed_union_ring_response'),('fixed_margin','fixed_union_margin')):
                    v=r[branch+'_'+new];expected=ref.get(old)
                    if v is None:
                        if expected not in (None,''):raise ValueError('historical fixed margin NA changed')
                    elif expected not in (None,''):
                        check(checks,sid,'same_cache_'+branch+'_'+old,v,float(expected),instance_id=r['instance_id'])
    for r in checks:
        r['domain']=domain;r.setdefault('check_class','same_cache_reconstruction')
    return dict(domain=domain,sample_id=sid,checks=checks,failed=any(not r['verified'] for r in checks),
                selected=selected,columns=columns,instances=instances,links=links,stencils=stencils)


def summarize_instances(rows,domain):
    # Pass a private compatibility view to the exact old summary kernel.
    internal=[dict(r,small_original_three=r.get('smallest_area_quartile',r.get('small_original_three',False))) for r in rows]
    return [dict(domain=domain,small_definition='original_domain_smallest_area_quartile',**r) for r in instance_summaries(internal)]
