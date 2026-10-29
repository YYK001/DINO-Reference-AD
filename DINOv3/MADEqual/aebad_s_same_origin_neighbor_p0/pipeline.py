"""Two-pass Source-only calibration and one-query/three-method prediction."""
import numpy as np
from . import inputs as ip,runtime as rt,numerics as nm
from .inputs import st
from .protocol import LAYERS,U,S,A,B,C,NEW
from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.calibration import response,z_values
from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import fit_image_mad
from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic.numerics import check


def baseline_provider(a):
    if a.baseline_mode=='fresh':return None
    _,r=rt.source_bundle(a)
    if r['metadata']['independent_rebuild']:return None
    p=ip.parent(a)
    if p is None:return None
    if r['metadata']['original_bank_seal']!=p.assets['references']['source']['seal']:return None
    try:
        for s in p.samples:p.prediction(s,False)
    except FileNotFoundError:return None
    return p


def fit_components(records):
    """Original MAD floor/float operation order; correspondence is absolute Top1."""
    from DINOv3.relation_reliability.calibration import fit_calibrators
    result={str(l):{} for l in LAYERS};counts=[]
    for name in ('abs','nb','rel'):
        pooled={}
        for j,l in enumerate(LAYERS):
            values=[]
            for x in records:
                good=np.ones(x['abs'].shape[1],bool) if name=='abs' else x['valid'][j]&x['candidate_valid'][j,:,0]
                values.append(np.array(x[name][j,good,0],np.float32))
            pooled[l]=np.concatenate(values)
            counts.append(dict(layer=l,component=name,patches=len(pooled[l]),calibration_images=len(records),correspondence='absolute_Source_Top1'))
        _,mad=fit_calibrators(pooled)
        for j,l in enumerate(LAYERS):result[str(l)][name]=dict(median=float(mad.medians[j]),scale=float(mad.scales[j]))
    return result,counts


def image_state(records):
    state={}
    for name in ('peak','local'):
        med,scale=fit_image_mad([x[name] for x in records]);state[name]=dict(medians=med.tolist(),scales=scale.tolist())
    return state


def raw_record(a,row,searches,i):
    path=a.cache_root/'calibration_observations'/f'{i:03d}'
    if not (path/'manifest.json').exists():
        q,m=rt.features(a,row);observed=rt.observations(q,m,searches)
        values={name:np.stack([observed[l][name] for l in LAYERS]) for name in ('absolute','nb','rel','indices','valid','candidate_valid')}
        values['abs']=values.pop('absolute');values['positions']=m['positions']
        _,source=rt.source_bundle(a)
        st.put(path,values,dict(identity=ip.identity(a),image=row,grid=m['grid'],output_shape=m['output_shape'],Source_seal=source['seal']))
    x,r=st.get(path)
    if r['metadata']['identity']!=ip.identity(a) or r['metadata']['image']!=row or r['metadata']['Source_seal']!=rt.source_bundle(a)[1]['seal']:raise ValueError('calibration cache input differs')
    return x,r['metadata']


def responses(observed,components,patch):
    output={b:[] for b in NEW};audits=[]
    for j,l in enumerate(LAYERS):
        v=observed[l];du=v['absolute'][:,0];uz=(du-np.float32(patch['medians'][j]))/np.float32(patch['scales'][j])
        values,audit=nm.joint(v['absolute'],v['nb'],v['rel'],v['candidate_valid'],v['valid'],uz,components[str(l)] if str(l) in components else components[l])
        for b in NEW:output[b].append(values[b])
        audits.append(dict(layer=l,**audit))
    return {b:np.stack(v) for b,v in output.items()},audits


def calibrate(a):
    dest=a.run_root/'calibration.json'
    if dest.exists():
        r=st.sealed_json(dest)
        if r['identity']!=ip.identity(a) or r['Source_seal']!=rt.source_bundle(a)[1]['seal']:raise ValueError('calibration bank changed')
        return dict(status='calibration_reused')
    searches,sr=rt.references(a,source_only=True);rows=[r for r in ip.catalog(a)['source'] if r['calibration']]
    records=[];metas=[]
    import time
    a._unit_started=time.perf_counter()
    for i,row in enumerate(rows,1):
        x,m=raw_record(a,row,searches,i);records.append(x);metas.append(m)
        print(f'[same-origin calibration components] {i}/66 Source only',flush=True)
        budget(a,'calibrate',i,66)
    components,counts=fit_components(records);st.write_csv(a.run_root/'calibration_counts.csv',counts)
    candidate_patch=dict(medians=[components[str(l)]['abs']['median'] for l in LAYERS],scales=[components[str(l)]['abs']['scale'] for l in LAYERS])
    p=baseline_provider(a);patch=p.mad['patch'] if p else candidate_patch
    # Independent banks get independent complete original baseline calibration.
    statistics={b:[] for b in (U,*NEW)};boundaries=[]
    from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.calibration import from_z
    for i,(x,m) in enumerate(zip(records,metas),1):
        obs={l:dict(absolute=x['abs'][j],nb=x['nb'][j],rel=x['rel'][j],valid=x['valid'][j],candidate_valid=x['candidate_valid'][j]) for j,l in enumerate(LAYERS)}
        zs,audits=responses(obs,components,patch);zs[U]=z_values(x['abs'][:,:,0],patch)
        boundaries.extend(dict(sample_id=m['image']['relative_path'],stage='calibration',**v) for v in audits)
        for b,z in zs.items():
            _,peak,local=from_z(z,x['positions'],m['grid'],m['output_shape']);statistics[b].append(dict(peak=peak,local=local))
    images={b:image_state(v) for b,v in statistics.items()}
    if p:images[U]=p.mad['image']
    images[S]=images[U]
    value=dict(identity=ip.identity(a),Source_seal=sr['seal'],components=components,component_counts=counts,
        baseline_patch=patch,image=images,reuse_baselines=p is not None,baseline_parent=p.identity if p else None,
        component_absolute_patch=candidate_patch,Source_images=66,Target_used=False,GT_used=False,
        raw_records=[st.sealed_json(a.cache_root/'calibration_observations'/f'{i:03d}'/'manifest.json')['seal'] for i in range(1,67)])
    st.sealed_json(dest,value);st.write_csv(a.run_root/'calibration_boundary_statistics.csv',boundaries)
    return dict(status='independent_components_and_branch_image_MAD_complete',images=66,reuse_baselines=p is not None)


def states(a):
    if hasattr(a,'_calibration'):return a._calibration
    r=st.sealed_json(a.run_root/'calibration.json')
    if r['identity']!=ip.identity(a):raise ValueError('calibration inputs changed')
    if getattr(a,'stage',None) not in ('evaluate','aupro','summarize','pack') and r['Source_seal']!=rt.source_bundle(a)[1]['seal']:raise ValueError('calibration Source changed')
    a._calibration=r
    return r


def path(a,s):return a.cache_root/'predictions'/s['sample_id']


def load(a,s):
    x,r=st.get(path(a,s));m=r['metadata']
    if m['identity']!=ip.identity(a) or m['sample']!=s or m['calibration_seal']!=states(a)['seal']:raise ValueError('prediction inputs changed')
    n=int(np.prod(m['grid']))
    if not np.array_equal(np.sort(x['positions']),np.arange(n)):raise ValueError('prediction physical mapping differs')
    for b in (U,S,*NEW):
        if x[b].shape!=(4,n) or x[b].dtype!=np.float32 or not np.isfinite(x[b]).all():raise ValueError('invalid joint standardized response')
    return x,r


def verified(a):
    r=st.sealed_json(a.run_root/'predictions_frozen.json');wanted=ip.catalog(a)['samples']
    if r['identity']!=ip.identity(a) or r['images']!=297 or r['GT_used'] or [x['sample_id'] for x in r['records']]!=[s['sample_id'] for s in wanted]:raise ValueError('all 297 predictions must be frozen before evaluation')
    for s,rec in zip(wanted,r['records']):
        _,p=load(a,s)
        if p['seal']!=rec['seal']:raise ValueError('prediction changed after freeze')
    return r


def baseline_distances(q,searches):
    """Legacy Exact endpoints/geometry, not the new method's tiled Top16."""
    from DINOv3.MADEqual.aebad_s_target_local_segment.geometry import geometry_layer
    raw_u=[];raw_s=[]
    for l in LAYERS:
        search=searches[l];distances,indices=search.baseline_top(q[l])
        raw_u.append(distances.min(axis=0))
        seg,_=geometry_layer(distances[1:],indices[1:],distances[0],search.cpu_parts[1:])
        raw_s.append(seg['K8_SEGMENT'])
    return {U:np.stack(raw_u),S:np.stack(raw_s)}


def check_baselines(rows,sample,raw,old,patch):
    """Parent's distance tolerances; image scores have their separate check."""
    for b in (U,S):
        wanted=old[b]*np.asarray(patch['scales'],np.float32)[:,None]+np.asarray(patch['medians'],np.float32)[:,None]
        for j,l in enumerate(LAYERS):
            check(rows,sample['sample_id'],'original_Exact_distance_'+b,raw[b][j],wanted[j],
                  atol=1e-7,rtol=1e-6,branch=b,domain=sample['domain'],layer=l)


def predict(a):
    if (a.run_root/'predictions_frozen.json').exists():verified(a);return dict(status='predictions_reused',query_forward_calls=0)
    searches,sr=rt.references(a);source,_=rt.source_bundle(a);cal=states(a);p=baseline_provider(a) if cal['reuse_baselines'] else None
    if cal['reuse_baselines'] and p is None:raise ValueError('declared parent baselines unavailable')
    records=[];samples=ip.catalog(a)['samples'];checks=[]
    import time
    a._unit_started=time.perf_counter()
    for i,s in enumerate(samples,1):
        destination=path(a,s)
        if not (destination/'manifest.json').exists():
            q,m=rt.features(a,s);obs=rt.observations(q,m,searches)
            zs,audits=responses(obs,cal['components'],cal['baseline_patch']);raw=baseline_distances(q,searches)
            zs[U]=z_values(raw[U],cal['baseline_patch']);zs[S]=z_values(raw[S],cal['baseline_patch'])
            localchecks=[]
            if p:
                old,record=p.prediction(s)
                if m['grid']!=record['metadata']['grid'] or not np.array_equal(m['positions'],old['positions']):raise ValueError('parent baseline physical geometry differs')
                check_baselines(localchecks,s,raw,old,cal['baseline_patch'])
                if not all(r['verified'] for r in localchecks):
                    st.write_csv(a.run_root/'failed_baseline_reuse.csv',localchecks)
                    np.savez_compressed(a.run_root/'failed_baseline_distances.npz',
                        union=raw[U],segment=raw[S],parent_union=old[U],parent_segment=old[S],
                        medians=np.asarray(cal['baseline_patch']['medians'],np.float32),
                        scales=np.asarray(cal['baseline_patch']['scales'],np.float32),positions=m['positions'])
                    raise ValueError('original Exact baseline reconstruction differs; preserved per-layer raw-distance audit in failed_baseline_reuse.csv; inputs and completed calibration retained')
                # Preserve exact inherited fallback response, including original FP32 roundoff.
                for j,l in enumerate(LAYERS):
                    fallback=~(obs[l]['valid']&obs[l]['candidate_valid'][:,0])
                    for b in NEW:zs[b][j,fallback]=old[U][j,fallback]
                zs[U]=np.array(old[U]);zs[S]=np.array(old[S])
            else:
                for j,l in enumerate(LAYERS):
                    fallback=~(obs[l]['valid']&obs[l]['candidate_valid'][:,0])
                    for b in NEW:zs[b][j,fallback]=zs[U][j,fallback]
            scores={b:response(z,m['positions'],m['grid'],m['output_shape'],cal['image'][b])[1] for b,z in zs.items()}
            if p:
                for b in (U,S):
                    for rule in ('CH1','CURRENT'):check(localchecks,s['sample_id'],'parent_score_'+rule,scores[b][rule],record['metadata']['scores'][b][rule],branch=b,domain='background')
                if not all(r['verified'] for r in localchecks):raise ValueError('same-bank baseline scoring differs')
            st.put(destination,dict(zs,positions=m['positions']),dict(identity=ip.identity(a),sample=s,grid=m['grid'],output_shape=m['output_shape'],
                calibration_seal=cal['seal'],Source_seal=sr['seal'],scores=scores,boundaries=audits,checks=localchecks,GT_used=False))
        _,r=load(a,s);records.append(dict(sample_id=s['sample_id'],seal=r['seal']));checks.extend(r['metadata']['checks'])
        print(f'[same-origin predict] {i}/297 {s["sample_id"]} five_maps_same_Q',flush=True)
        budget(a,'predict',i,297)
    st.sealed_json(a.run_root/'predictions_frozen.json',dict(identity=ip.identity(a),images=297,records=records,GT_used=False))
    st.write_csv(a.run_root/'baseline_numerical_verification.csv',checks)
    return dict(status='predictions_frozen',images=297,query_forward_calls=getattr(a,'_new_forwards',0),training=False)


def budget(a,stage,done,total):
    # The ONLY admission/pause rule: the user-requested twelve-hour experiment budget.
    from .runner import enforce_time_budget
    enforce_time_budget(a,stage,done,total)
