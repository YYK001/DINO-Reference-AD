"""One full Q, shared original Exact/anchor; six distances and one frozen MAD."""
from pathlib import Path
import numpy as np
from . import inputs as ip,runtime as rt,numerics
from .inputs import st
from .protocol import CONFIG,DOMAINS,LAYERS,BRANCHES,U,S,MAIN,TARGET,SOURCE,UNIFORM
from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.calibration import z_values,response
from DINOv3.MADEqual.aebad_s_target_local_segment.geometry import geometry_layer


def prediction_path(a,s):return a.cache_root/'predictions'/s['domain']/s['sample_id']


def calibration_states(a,domain):
    state=st.sealed_json(a.run_root/'input_distance_mad.json')
    return {b:dict(state) for b in BRANCHES}  # SAME patch + image state for all distances.


def validate_prediction(a,s,x,r,states):
    m=r['metadata'];n=int(np.prod(m['grid']))
    if m['identity']!=ip.identity(a) or m['sample']!=s or m['calibration_seal']!=st.sealed_json(a.run_root/'input_distance_mad.json')['seal']:raise ValueError('prediction identity/calibration differs')
    if x['positions'].dtype!=np.int64 or not np.array_equal(np.sort(x['positions']),np.arange(n)):raise ValueError('incomplete physical mapping')
    for b in BRANCHES:
        if x[b].shape!=(4,n) or x[b].dtype!=np.float32 or not np.isfinite(x[b]).all():raise ValueError('invalid predicted layer distances/responses')
    expected=getattr(a,'_prediction_seals',{}).get((s['domain'],s['sample_id']))
    if expected is not None and expected!=r['seal']:raise ValueError('frozen prediction record changed')
    return m


def verify(a):
    c=ip.catalog(a);expected={};seals=[]
    for d in DOMAINS:
        rows=[s for s in c['samples'] if s['domain']==d];counts=[len(rows),sum(s['label']==0 for s in rows),sum(s['label']==1 for s in rows)]
        if counts!=CONFIG['populations'][d]:raise ValueError('full 926 evaluation population required')
        r=st.sealed_json(a.run_root/'predictions'/(d+'.json'))
        if r['identity']!=ip.identity(a) or [(v['domain'],v['sample_id']) for v in r['records']]!=[(d,s['sample_id']) for s in rows]:raise ValueError('incomplete prediction freeze; GT refused')
        expected.update({(d,v['sample_id']):v['seal'] for v in r['records']});seals.append(r['seal'])
    a._prediction_seals=expected;return st.digest(seals)


def residual_anchor(q,source,parts,nn):
    si,ti=nn['si'],nn['ti'];n=len(q);offsets=np.cumsum([0]+[len(x) for x in parts]);target=nn['dt']<nn['ds']
    if si.shape!=(n,) or ti.shape!=(n,) or (si<0).any() or (si>=len(source)).any() or (ti<0).any() or (ti>=offsets[-1]).any():raise ValueError('invalid original NN identity')
    anchor=np.array(source[si],np.float32)
    for bank,lo,hi in zip(parts,offsets[:-1],offsets[1:]):
        use=target&(ti>=lo)&(ti<hi);anchor[use]=bank[ti[use]-lo]
    return np.asarray(q,np.float32)-anchor,target  # Source wins exact distance ties.


def parent_prediction(a,s):
    record=ip.assets(a)['prediction_parents'].get(s['domain']+'/'+s['sample_id'])
    if record is None:return None
    x,r=st.get(record['path']);m=r['metadata']
    if m['identity']!=record['identity'] or m['sample']!=s:raise ValueError('parent prediction image identity differs')
    return x,r,record


def compare(values,expected,kind,sample,layer,checks):
    error=float(np.max(np.abs(np.asarray(values,np.float64)-expected),initial=0))
    ok=bool(np.allclose(values,expected,atol=CONFIG['distance_atol'],rtol=CONFIG['distance_rtol']))
    checks.append(dict(domain=sample['domain'],sample_id=sample['sample_id'],layer=layer,kind=kind,max_abs_error=error,
        atol=CONFIG['distance_atol'],rtol=CONFIG['distance_rtol'],verified=ok))
    if not ok:raise ValueError('parent '+kind+' reproduction failed; no tolerance relaxation')


def predict(a):
    from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.fresh_inputs import Data
    from DINOv3.MADEqual.aebad_s_reference_residual_segmentation.runtime import extraction
    refs=rt.References(a);c=refs.c;states=calibration_states(a,'background');patch=states[U]['patch'];model=None;calls=0;reused=0;created=0
    for d in DOMAINS:
        records=[];subspaces={l:st.get(a.cache_root/'subspaces'/d/f'L{l}') for l in LAYERS}
        clusters={l:st.get(a.cache_root/'clusters'/f'L{l}')[0] for l in LAYERS}
        for l,(_,r) in subspaces.items():
            if r['metadata']['identity']!=ip.identity(a):raise ValueError('different fitted subspaces')
            completed=st.sealed_json(a.run_root/'fit'/d/f'L{l}.json')
            if completed['identity']!=ip.identity(a) or completed['subspace_seal']!=r['seal']:raise ValueError('incomplete/uncommitted subspace fit')
        with Data(a.data) as data:
            for sample in [s for s in c['samples'] if s['domain']==d]:
                path=prediction_path(a,sample)
                if not (path/'manifest.json').exists():
                    prior=rt.raw_query(a,sample)
                    if prior:q,geometry=prior;reused+=1
                    else:
                        if model is None:model=rt.extractor(a)
                        q,geometry=extraction(model,data.image(sample),a.device);calls+=1
                    positions=np.asarray(geometry['positions'],np.int64);old=parent_prediction(a,sample);checks=[];diagnostics=[];raw={b:[] for b in BRANCHES};etas=[];source_ids=[];target_ids=[];group_ids=[];anchor_sides=[]
                    if old and (old[1]['metadata']['grid']!=geometry['grid'] or not np.array_equal(old[0]['positions'],positions)):raise ValueError('parent query physical geometry differs')
                    for j,l in enumerate(LAYERS):
                        nn,parts,bank_ids=refs.formal(q[l],d,l);source,sseal=refs.source('source',l)
                        if old:
                            m=old[1]['metadata'];bound=m['NN_record']['metadata']['inputs']['banks'][j]
                            if bound!=bank_ids:raise ValueError('parent NN bank identity differs; cannot reuse old indices/distances')
                            for k in ('ds','dt'):compare(nn[k],old[0][f'{k}{l}'],k,sample,l,checks)
                            # Parent Exact identity/output is frozen, including any duplicate-row ties.
                            for k in ('ds','dt','si','ti'):nn[k]=np.asarray(old[0][f'{k}{l}'])
                        du=np.minimum(nn['ds'],nn['dt']);offsets=np.cumsum([0]+[len(x) for x in parts])
                        segment,_=geometry_layer(nn['perd'],nn['peri']-offsets[:-1,None],nn['ds'],parts);ds=segment['K8_SEGMENT']
                        if old:
                            old_u=old[0][U][j]*np.float32(patch['scales'][j])+np.float32(patch['medians'][j]);old_s=old[0][S][j]*np.float32(patch['scales'][j])+np.float32(patch['medians'][j])
                            compare(du,old_u,'UNION_distance',sample,l,checks);compare(ds,old_s,'SEGMENT_distance',sample,l,checks)
                        residual,side=residual_anchor(q[l],source,parts,nn);groups=clusters[l]['source_groups'][nn['si']];sub=subspaces[l][0];layer_eta=[]
                        raw[U].append(du);raw[S].append(ds)
                        for bi,b in enumerate((MAIN,TARGET,SOURCE)):
                            value,eta,audit=numerics.correct(du,residual,groups,sub['bases'][bi],sub['ranks'],CONFIG['strength'])
                            raw[b].append(value);layer_eta.append(eta);diagnostics.append(dict(layer=l,branch=b,**audit))
                        uniform,audit=numerics.matched_uniform(du,raw[MAIN][-1]);raw[UNIFORM].append(uniform);diagnostics.append(dict(layer=l,branch=UNIFORM,**audit))
                        etas.append(np.stack(layer_eta));source_ids.append(nn['si']);target_ids.append(nn['ti']);group_ids.append(groups);anchor_sides.append(side)
                    values={b:np.stack(raw[b]).astype(np.float32) for b in BRANCHES};z={b:z_values(v,patch) for b,v in values.items()}
                    if old:
                        z[U]=np.array(old[0][U]);z[S]=np.array(old[0][S])
                        for b in (MAIN,TARGET,SOURCE,UNIFORM):z[b][values[b]==values[U]]=z[U][values[b]==values[U]]
                    shape=[sample['image_height']//4,sample['image_width']//4];scores={b:response(z[b],positions,geometry['grid'],shape,states[b]['image'])[1] for b in BRANCHES}
                    if old:
                        from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic.numerics import check
                        for b in (U,S):
                            for rule in ('CH1','CURRENT'):check(checks,sample['sample_id'],'parent_same_cache_'+rule,scores[b][rule],old[1]['metadata']['scores'][b][rule],domain=d,branch=b)
                        if not all(r['verified'] for r in checks):raise ValueError('parent baseline score reproduction failed')
                    arrays={**z,'positions':positions,'raw_union':values[U],'raw_segment':values[S],'raw_main':values[MAIN],
                        'eta':np.stack(etas,axis=1),'groups':np.stack(group_ids),'source_indices':np.stack(source_ids),'target_indices':np.stack(target_ids),'anchor_is_Target':np.stack(anchor_sides)}
                    st.put(path,arrays,dict(identity=ip.identity(a),sample=sample,grid=geometry['grid'],output_shape=shape,scores=scores,
                        calibration_seal=st.sealed_json(a.run_root/'input_distance_mad.json')['seal'],checks=checks,diagnostics=diagnostics,
                        parent=old[2] if old else None,GT_used=False,baseline_distance_source='parent_frozen_Exact' if old else 'current_original_Exact'));created+=1
                x,r=st.get(path);validate_prediction(a,sample,x,r,states);records.append(dict(domain=d,sample_id=sample['sample_id'],seal=r['seal']))
                print(f'[SACP predict] {d}/{sample["sample_id"]} six_branches_same_Q',flush=True)
        st.sealed_json(a.run_root/'predictions'/(d+'.json'),dict(identity=ip.identity(a),records=records))
    seal=verify(a);st.sealed_json(a.run_root/'predictions_frozen.json',dict(identity=ip.identity(a),images=len(c['samples']),seal_of_manifests=seal,GT_used=False))
    return dict(status='predictions_frozen',new_images=created,query_forward_calls=calls,reused_raw_Q=reused,
        searches=refs.searches,execution=refs.search.stats,training=False)
