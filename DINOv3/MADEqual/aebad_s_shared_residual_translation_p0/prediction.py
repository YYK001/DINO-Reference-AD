"""Shared original retrieval, immutable per-image output, freeze BEFORE GT."""
import time
from pathlib import Path
import numpy as np
from . import inputs as ip, runtime as rt, numerics as nm
from .inputs import st
from .protocol import *
from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.calibration import z_values, response
from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.evaluation import shared


def failure(a,d,row,checks):
    st.write_csv(a.run_root/'failed_baseline_checks.csv',[dict(r,domain=d,sample_id=row['sample_id']) for r in checks])
    failed=', '.join(r['check'] for r in checks if not r['verified'])
    raise ValueError('baseline compatibility failed ('+failed+'); see failed_baseline_checks.csv; no relaxation or baseline stitching')


def baseline_checks(a,d,row,raw,positions,grid,shape,zero_raw=None):
    p=ip.parent(a);x,meta,ref=ip.baseline(a,d,row);checks=[];oldorder=nm.order(x['positions'],grid)
    actual=z_values(raw,p.inputs['mad']['patch']);expected=x[U][:,oldorder]
    if not np.isfinite(actual).all() or actual.shape!=expected.shape:raise ValueError('baseline z layout differs')
    e=float(np.abs(actual-expected).max());checks.append(dict(check='original_Exact_UNION_z',max_abs_error=e,atol=1e-5,verified=e<=1e-5))
    old=p.record(d,row)
    for b in (U,S,SOURCE):
        image,scores=(nm.weighted(x[U],x['positions'],grid,shape,p.weights[d]) if b==SOURCE else
                      response(x[b],x['positions'],grid,shape,p.inputs['mad']['image']))
        err=max(abs(v-old['scores'][b][k]) for k,v in scores.items())
        ok=shared.array_sha(image)==old['map_hashes'][b]
        checks.append(dict(check='frozen_reference_map_score',branch=b,max_abs_error=err,atol=1e-5,map_sha_equal=ok,verified=ok and err<=1e-5))
    # Independently verify the zero-translation scoring chain, without accepting
    # nearby but numerically different historical maps as interchangeable.
    im,sc=nm.score(raw,positions,grid,shape,p.inputs['mad']['patch'],p.inputs['mad']['image'],p.weights[d])
    err=abs(sc['CH1']-old['scores'][U]['CH1']);maperr=float(np.abs(im-nm.weighted(x[U],x['positions'],grid,shape,p.weights[d])[0]).max())
    checks.append(dict(check='zero_translation_chain',max_abs_error=max(err,maperr),atol=1e-5,verified=max(err,maperr)<=1e-5))
    # This is the genuine zero-translation check: use the candidate's DIRECT
    # residual norm, not the original MM distances passed as raw above. Preserve
    # the existing score/map tolerance; incompatible reuse still stops here.
    if zero_raw is not None:
        reference,_=nm.weighted(x[U],x['positions'],grid,shape,p.weights[d])
        for name,value in zero_raw.items():
            zi,zs=nm.score(value,positions,grid,shape,p.inputs['mad']['patch'],p.inputs['mad']['image'],p.weights[d])
            se=abs(zs['CH1']-old['scores'][U]['CH1']);me=float(np.abs(zi-reference).max())
            checks.append(dict(check='zero_direct_vector_score_map',variant=name,max_abs_error=max(se,me),
                               CH1_abs_error=se,map_max_abs_error=me,atol=1e-5,verified=max(se,me)<=1e-5))
    if not all(r['verified'] for r in checks):failure(a,d,row,checks)
    return checks,ref


def verify(a,d,row,check_inputs=False):
    if check_inputs:ip.frozen_backend(a,compute=True)
    rec=st.sealed_json(ip.record_path(a,d,row))
    if any(rec.get(k)!=v for k,v in ip.signature(a,d).items()) or rec['image']!=ip.image(row) or rec['GT_used']:raise ValueError('prediction input identity differs')
    x,r=st.get(ip.pred_path(a,d,row))
    if r['seal']!=rec['cache_seal'] or r['metadata']['contract']!=rec['contract'] or r['metadata'].get('prediction_backend')!=rec['prediction_backend'] or not all(v['verified'] for v in rec['checks']):raise ValueError('prediction bundle/checks/backend differ')
    n=len(x['positions']);nm.order(x['positions'],r['metadata']['grid'])
    if any(x[b].shape!=(4,n) or x[b].dtype!=np.float32 or not np.isfinite(x[b]).all() or (x[b]<0).any() for b in MAPS):raise ValueError('invalid frozen candidate distances')
    if check_inputs:
        ref=ip.feature_ref(a,row);old=rec['contract']['query']
        if ref is None:
            if old.get('kind')!='original_full_forward':raise ValueError('original query feature provider disappeared')
        elif ref!=old:raise ValueError('query feature provider changed on prediction resume')
        if ip.source_ref(a)['seal']!=rec['contract']['source']:raise ValueError('Source provider changed')
        for rr in rec['contract']['reference_features']:
            if st.sealed_json(Path(rr['path'])/'manifest.json')['seal']!=rr['seal']:raise ValueError('support feature provider changed')
    return rec,x,r


def one(a,d,row,refs):
    ip.frozen_backend(a,compute=True)
    if ip.record_path(a,d,row).exists():
        verify(a,d,row,check_inputs=True);return False
    tick=time.perf_counter();p=ip.parent(a)
    forwards_before=getattr(a,'feature_forwards',0);searches_before=refs.search.stats['searches']
    q,geometry,feature=rt.features(a,row,persist=False)
    features_done=time.perf_counter()
    pos=np.asarray(geometry['positions'],np.int64);order=nm.order(pos,geometry['grid']);n=len(pos)
    nn,nnrec,reused=refs.nearest(q,geometry,feature,row)
    nearest_done=time.perf_counter()
    nnorder=nm.order(nn['positions'],geometry['grid']);physical=np.arange(n,dtype=np.int64)
    raw={b:[] for b in MAPS};compact=dict(positions=physical);diagnostics=[];original=[]
    zeros={name:[] for name in ('zero_fixed','zero_reselect')};arithmetic=[]
    for l in LAYERS:
        qq=np.array(q[l][order],np.float32);banks,keys=refs.banks(l)
        indices=np.asarray(nn[f'index{l}'][:,nnorder],np.int64);distances=np.asarray(nn[f'distance{l}'][:,nnorder],np.float32)
        if qq.shape!=(n,1280):raise ValueError('four-layer complete feature grid required')
        if not np.isfinite(qq).all() or not np.isfinite(distances).all() or (distances<0).any():raise ValueError('invalid feature/Exact values')
        if any((indices[s]<0).any() or (indices[s]>=len(bank)).any() for s,bank in enumerate(banks)):raise ValueError('invalid nearest memory row')
        values,small,diag=rt.translate(a,qq,banks,indices,distances,refs,keys)
        # torch.cdist(use_mm) and norm(q-m) have different FP32 cancellation.
        # Compare like-for-like in smoke; record cross-path drift descriptively.
        # Actual zero-correction CH1/map compatibility is checked below unchanged.
        exact_union=distances.min(0);delta=np.abs(small['zero_fixed']-exact_union)
        arithmetic.append(dict(domain=d,sample_id=row['sample_id'],layer=l,comparison='direct_residual_vs_original_cdist_MM',
            max_abs_error=float(delta.max()),mean_abs_error=float(delta.mean(dtype=np.float32)),
            positions_over_original_raw_tolerance=int(np.sum(delta>1e-7+1e-6*np.abs(exact_union))),
            diagnostic_only=True,atol_reference=1e-7,rtol_reference=1e-6))
        for name in zeros:zeros[name].append(small[name])
        for b in MAPS:raw[b].append(values[b])
        for k,v in small.items():compact[f'{k}{l}']=v
        diagnostics.append(dict(layer=l,**diag));original.append(exact_union)
    shape=[row['image_height']//4,row['image_width']//4]
    translation_done=time.perf_counter()
    st.write_csv(a.run_root/d/'arithmetic_audit'/(row['sample_id']+'.csv'),arithmetic)
    checks,baseline=baseline_checks(a,d,row,np.stack(original),physical,geometry['grid'],shape,
                                   zero_raw={name:np.stack(value) for name,value in zeros.items()})
    scores={};hashes={}
    for b in MAPS:
        compact[b]=np.stack(raw[b]);im,sc=nm.score(compact[b],physical,geometry['grid'],shape,p.inputs['mad']['patch'],p.inputs['mad']['image'],p.weights[d])
        sc['SEGMENT_CH1']=p.record(d,row)['scores'][S]['CH1'];scores[b]=sc;hashes[b]=shared.array_sha(im)
    dep=ip.contract(a,d,row,feature,refs.targets)
    saved=st.put(ip.pred_path(a,d,row),compact,dict(contract=dep,prediction_backend=ip.frozen_backend(a,compute=True),
        grid=geometry['grid'],output_shape=shape,nearest_seal=nnrec['seal'],GT_used=False))
    # Repairing an interruption between bundle commit and JSON commit is safe:
    # compare the old bundle to the newly derived distances, do not re-enroll it.
    old,_=st.get(ip.pred_path(a,d,row),MAPS)
    if any(not np.array_equal(old[b],compact[b]) for b in MAPS):raise ValueError('unrecorded committed prediction differs')
    st.write_json(a.run_root/d/'timings'/(row['sample_id']+'.json'),dict(seconds=time.perf_counter()-tick,nearest_reused=reused,
        image_key=d+'/'+row['sample_id'],full_path=True,feature_forward=getattr(a,'feature_forwards',0)-forwards_before,
        Exact_searches=refs.search.stats['searches']-searches_before,
        phases=dict(feature_IO_or_forward=features_done-tick,Exact_IO_or_search=nearest_done-features_done,
                    vector_gather_translation=translation_done-nearest_done,baseline_scoring_and_write=time.perf_counter()-translation_done)))
    st.sealed_json(ip.record_path(a,d,row),dict(**ip.signature(a,d),image=ip.image(row),sample_id=row['sample_id'],contract=dep,
        cache_seal=saved['seal'],nearest_seal=nnrec['seal'],baseline=baseline,scores=scores,map_hashes=hashes,checks=checks,
        arithmetic_audit=arithmetic,diagnostics=diagnostics,GT_used=False))
    print(f'[shared residual] {d}/{row["sample_id"]} three_maps seconds={time.perf_counter()-tick:.3f}',flush=True)
    return True


def freeze_predictions(a):
    p=ip.parent(a);records=[]
    for d in DOMAINS:
        for row in p.samples(d):
            r=st.sealed_json(ip.record_path(a,d,row))
            if r['identity']!=ip.signature(a,d)['identity'] or r['image']!=ip.image(row) or r['GT_used']:raise ValueError('incomplete pre-GT freeze')
            records.append(dict(domain=d,sample_id=row['sample_id'],seal=r['seal']))
    return st.sealed_json(a.run_root/'predictions_frozen.json',dict(identity=ip.signature(a,DOMAINS[0])['identity'],records=records,images=902,GT_used=False))


def global_freeze(a):
    # Validate list contents on every new stage, then keep a stage-local snapshot.
    if not hasattr(a,'_prediction_freeze'):a._prediction_freeze=freeze_predictions(a)
    r=st.sealed_json(a.run_root/'predictions_frozen.json')
    if r!=a._prediction_freeze:raise ValueError('prediction freeze changed within stage')
    return r


def predict(a):
    ip.frozen_backend(a,compute=True)
    new=reused=0;stats=[]
    for d in DOMAINS:
        refs=None
        for row in ip.parent(a).samples(d):
            if ip.record_path(a,d,row).exists():verify(a,d,row,check_inputs=True);reused+=1
            else:
                if refs is None:refs=rt.References(a,d)
                new+=one(a,d,row,refs)
            from .smoke import observe
            observe(a)
        if refs:stats.append(dict(domain=d,**refs.search.stats))
    freeze_predictions(a)
    return dict(status='predictions_complete',new_images=new,reused_images=reused,feature_forwards=getattr(a,'feature_forwards',0),execution=stats)


def replay(a,d,row,branch):
    rec,x,meta=verify(a,d,row);p=ip.parent(a)
    im,sc=nm.score(x[branch],x['positions'],meta['metadata']['grid'],meta['metadata']['output_shape'],p.inputs['mad']['patch'],p.inputs['mad']['image'],p.weights[d])
    if shared.array_sha(im)!=rec['map_hashes'][branch] or any(sc[k]!=rec['scores'][branch][k] for k in sc):raise ValueError('map/score replay differs from pre-GT identity')
    return im,rec['scores'][branch],meta['metadata']
