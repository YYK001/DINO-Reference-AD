import time
from pathlib import Path
import numpy as np
from . import inputs as ip,numerics as nm
from .inputs import st
from .protocol import LAYERS,NEW,U,S
from .composition import image_scores


def guard(a,stage,done,total):
    costs=[st.read_json(p) for p in (a.run_root/'costs').glob('*.json')]
    for peer in getattr(a,'budget_peer_run',[]):costs.extend(st.read_json(p) for p in (peer/'costs').glob('*.json'))
    previous=sum(r['seconds'] for r in costs)
    elapsed=time.perf_counter()-a._started
    estimate=previous+elapsed*total/max(done,1)+600
    if not a.allow_over_12h and (previous+elapsed>43200 or (done>=min(5,total) and estimate>43200)):
        st.write_json(a.run_root/'budget_pause.json',dict(stage=stage,done=done,total=total,projected_seconds=estimate,status='paused_12h_budget'))
        raise RuntimeError('paused at committed unit: estimated >12h; user approval required for --allow-over-12h')


def normal_features(a,row):
    own=a.cache_root/'normal_features'/ip.old.key(row)
    if (own/'manifest.json').exists():
        x,r=st.get(own)
        if r['metadata']['image']!=row or r['metadata']['extraction']!=ip.parent(a).extraction:raise ValueError('normal feature configuration differs')
        return {l:x[f'q{l}'] for l in LAYERS},dict(grid=r['metadata']['grid'],positions=x['positions'])
    try:
        ref=ip.feature_ref(a,row);x,r=st.get(ref['path'],tuple(f'q{l}' for l in LAYERS)+('positions',))
        if r['seal']!=ref['seal']:raise ValueError('normal feature provenance changed')
        return {l:x[f'q{l}'] for l in LAYERS},dict(grid=r['metadata']['grid'],positions=x['positions'])
    except FileNotFoundError:
        if not a.extract_missing:raise
    # Only this explicitly requested path imports/constructs the frozen host.
    from DINOv3.MADEqual.aebad_s_same_origin_neighbor_p0 import feature_identity as fi
    from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.runtime import host,extraction
    fi.compatible(fi.native_extraction(a),ip.parent(a).extraction)
    if not hasattr(a,'_model'):a._model=host(a)[0]
    from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.fresh_inputs import Data
    with Data(a.data) as data:q,m=extraction(a._model,data.image(row),a.device)
    st.put(own,{**{f'q{l}':q[l] for l in LAYERS},'positions':m['positions']},
        dict(image=row,grid=m['grid'],extraction=ip.parent(a).extraction))
    a.feature_forwards=getattr(a,'feature_forwards',0)+1
    return q,m


def prepare_normal_stats(a):
    ip.freeze(a)
    if (a.run_root/'normal_statistics.json').exists():return ip.normal_statistics(a)
    p=ip.parent(a);records=[];source=None;supports=None
    tasks=ip.units(a)
    if getattr(a,'source_stats_run',None):
        shared=ip.source_statistics(a);records.extend(shared['records'])
        st.sealed_json(a.run_root/'source_reuse.json',dict(domain=ip.domain(a),identity=ip.identity(a),provider_path=str(a.source_stats_run),provider_statistics_seal=shared['seal'],provider_inputs_seal=shared['inputs_seal'],Source_images=66,provider_records_unchanged=True))
    for done,(kind,i,row) in enumerate(tasks,1):
        dest=ip.normal_path(a,kind,i);expected=ip.normal_identity(a,kind,i,row)
        if not (dest/'manifest.json').exists():
            if not a.supplement_retrieval:raise FileNotFoundError(f'{dest}: normal distance absent; explicit --supplement-retrieval required; no automatic model/retrieval')
            import torch
            from DINOv3.relation_reliability.superadd_host import _exact_top1_chunked
            from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.runtime import reliable_precision
            reliable_precision(a.device)
            if source is None:
                ref=p.assets['references']['source'];source,sr=st.get(ref['path'],tuple(f'L{l}' for l in LAYERS))
                if sr['seal']!=ref['seal']:raise ValueError('original bank differs; no reconstruction')
            q,m=normal_features(a,row);raw=[]
            if kind=='Target' and supports is None:supports=[normal_features(a,r) for r in p.supports]
            for l in LAYERS:
                with torch.inference_mode():
                    query=torch.tensor(q[l],device=a.device,dtype=torch.float32)
                    bank=torch.tensor(source[f'L{l}'],device=a.device,dtype=torch.float32)
                    d,_=_exact_top1_chunked(query,bank);del bank
                    if kind=='Target':
                        nm.reference_rows(p.supports,i)  # Assert whole-image exclusion before loading candidates.
                        for j,(target,_) in enumerate(supports):
                            if j==i:continue
                            bank=torch.tensor(target[l],device=a.device,dtype=torch.float32)
                            dt,_=_exact_top1_chunked(query,bank);d=torch.minimum(d,dt);del bank
                    raw.append(d.numpy()/np.float32(1280))
            st.put(dest,dict(raw=np.stack(raw),positions=m['positions']),dict(expected,grid=m['grid'],output_shape=[row['image_height']//4,row['image_width']//4]))
        x,r=st.get(dest,('raw','positions'))
        if any(r['metadata'].get(k)!=v for k,v in expected.items()):raise ValueError('normal distance provenance differs')
        m=r['metadata'];value=nm.normal_statistics(x['raw'],x['positions'],m['grid'],m['output_shape'],p.mad['patch'])
        records.append(dict(**ip.tags(a),kind=kind,index=i,image=row['relative_path'],distance_seal=r['seal'],statistics=value.tolist()))
        print(f'[layer weights normal] {done}/{len(tasks)} {kind} {row["relative_path"]}',flush=True)
        guard(a,'prepare-normal-stats',done,len(tasks))
    table=[]
    for r in records:
        for j,l in enumerate(LAYERS):table.append(dict(domain=ip.domain(a),kind=r['kind'],image=r['image'],index=r['index'],layer=l,**dict(zip(('u','p','q'),r['statistics'][j]))))
    st.write_csv(a.run_root/'normal_statistics.csv',table)
    return st.sealed_json(a.run_root/'normal_statistics.json',dict(**ip.tags(a),identity=ip.identity(a),records=records,quantile_method='numpy.quantile_linear',numpy_version=np.__version__))


def fit_weights(a):
    r=st.sealed_json(a.run_root/'normal_statistics.json')
    if r['identity']!=ip.identity(a):raise ValueError('normal statistics inputs changed')
    source=[x['statistics'] for x in r['records'] if x['kind']=='Source'];target=[x['statistics'] for x in r['records'] if x['kind']=='Target']
    anchor,ss,tt=nm.moments(source,target);w={};rows=[]
    for branch,gamma in zip(NEW,(0.,.5)):
        value,scale,floor=nm.weights(ss,tt,gamma);w[branch]=value.tolist()
        for l,layer in enumerate(LAYERS):
            for j,stat in enumerate(('u','p','q')):
                rows.append(dict(domain=ip.domain(a),branch=branch,gamma=gamma,layer=layer,statistic=stat,Source_anchor=anchor[l,j],Source_scale=np.sqrt(ss[l,j]),Target_LOO_scale=np.sqrt(tt[l,j]),mixed_scale=scale[l,j],floor_triggered=bool(floor[l,j]),weight=value[l,j]))
    if getattr(a,'source_stats_run',None):
        from .protocol import SOURCE
        expected=ip.source_statistics(a)['weights'][SOURCE]
        if not np.array_equal(np.asarray(w[SOURCE]),np.asarray(expected)):raise ValueError('shared Source-only weights differ across domains')
    st.write_csv(a.run_root/'weights.csv',rows)
    return st.sealed_json(a.run_root/'weights.json',dict(**ip.tags(a),identity=ip.identity(a),normal_statistics_seal=r['seal'],weights=w,GT_used=False))


def prediction(a,row):
    x,r=st.get(ip.prediction_path(a,row))
    w=st.sealed_json(a.run_root/'weights.json');p=ip.parent(a)
    if r['metadata'].get('domain',ip.domain(a))!=ip.domain(a) or r['metadata']['sample']!=row or r['metadata']['identity']!=ip.identity(a) or r['metadata']['weights_seal']!=w['seal'] or r['metadata']['parent_seal']!=p.seals[row['sample_id']]:raise ValueError('prediction dependencies changed')
    return x,r


def predict(a):
    p=ip.parent(a);w=st.sealed_json(a.run_root/'weights.json')
    if w['identity']!=ip.identity(a):raise ValueError('weight inputs differ')
    from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.calibration import response
    records=[]
    for i,row in enumerate(p.samples,1):
        dest=ip.prediction_path(a,row)
        if not (dest/'manifest.json').exists():
            old,r=p.prediction(row);m=r['metadata'];arrays={};scores={}
            for b in (U,S):
                _,s=response(old[b],old['positions'],m['grid'],m['output_shape'],p.mad['image'])
                if any(abs(s[k]-m['scores'][b][k])>1e-5 for k in ('CH1','CURRENT')):raise ValueError('original baseline scores differ; no tolerance relaxation')
            for b in NEW:
                arrays[b],scores[b]=nm.weighted(old[U],old['positions'],m['grid'],m['output_shape'],w['weights'][b])
            st.put(dest,arrays,dict(**ip.tags(a),identity=ip.identity(a),weights_seal=w['seal'],parent_seal=r['seal'],sample=row,scores=scores,**(dict(combination_image_scores=image_scores(m['scores'][S])) if ip.domain(a)!='background' else {})))
        _,r=prediction(a,row);records.append(dict(**ip.tags(a),sample_id=row['sample_id'],seal=r['seal']))
        if i%25==0 or i==ip.count(a):print(f'[layer weights replay] {i}/{ip.count(a)} no feature forward or retrieval',flush=True)
        guard(a,'predict',i,ip.count(a))
    return st.sealed_json(a.run_root/'predictions_frozen.json',dict(**ip.tags(a),identity=ip.identity(a),weights_seal=w['seal'],records=records,images=ip.count(a),GT_used=False))


def verified(a):
    frozen=st.sealed_json(a.run_root/'predictions_frozen.json')
    if ip.domain(a)!='background' and (frozen.get('domain')!=ip.domain(a) or frozen['images']!=ip.count(a) or len(frozen['records'])!=ip.count(a) or any(r['domain']!=ip.domain(a) for r in frozen['records'])):raise ValueError('prediction domain/count differs')
    if frozen['identity']!=ip.identity(a) or frozen['weights_seal']!=st.sealed_json(a.run_root/'weights.json')['seal'] or [x['sample_id'] for x in frozen['records']]!=[s['sample_id'] for s in ip.parent(a).samples]:raise ValueError('incomplete prediction freeze')
    for row,ref in zip(ip.parent(a).samples,frozen['records']):
        if prediction(a,row)[1]['seal']!=ref['seal']:raise ValueError('modified frozen prediction')
    return frozen
