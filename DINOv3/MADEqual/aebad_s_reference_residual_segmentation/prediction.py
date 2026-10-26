"""Freeze all fourteen predictions before any test GT decode."""
import numpy as np
from pathlib import Path
from . import inputs as ip,storage as st,calibration as cal,inference
from .protocol import DOMAINS,LAYERS,U,S,member,ensemble
from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.prediction import verify as _verify
from types import FunctionType

def prediction_path(a,s):return a.cache_root/'predictions'/s['domain']/s['sample_id']


def calibration_states(a,domain):
    states={b:cal.state(a,domain,b) for b in ip.plan(a)['branches']}
    if any(v['identity']!=ip.identity(a) for v in states.values()):raise ValueError('different logit calibration run')
    return states


def validate_prediction(a,s,x,r,states):
    m=r['metadata'];n=int(np.prod(m['grid']))
    if m['identity']!=ip.identity(a) or m['sample']!=s or m['calibration_seals']!={b:v['seal'] for b,v in states.items()}:raise ValueError('prediction identity differs')
    expected=getattr(a,'_prediction_record_seals',{}).get((s['domain'],s['sample_id']))
    if expected is not None and expected!=r['seal']:raise ValueError('prediction metadata differs from frozen domain manifest')
    if not np.array_equal(np.sort(x['positions']),np.arange(n)):raise ValueError('physical position mapping differs')
    for b in ip.plan(a)['branches']:
        if x[b].shape!=(4,n) or x[b].dtype!=np.float32 or not np.isfinite(x[b]).all():raise ValueError('invalid complete calibrated output')
    return m


def verify(a):
    # Reuse exact global population/freeze checks, with independent globals.
    import sys
    if st.full_audit():
        return FunctionType(_verify.__code__,dict(_verify.__globals__,ip=ip,st=st,DOMAINS=DOMAINS,CONFIG=ip.config(a),
            calibration_states=calibration_states,prediction_path=prediction_path,validate_prediction=validate_prediction),_verify.__name__)(a)
    # Domain manifests already commit exactly the validated per-image predictions.
    # Check population/commit metadata here; read each payload only when consumed.
    c=ip.catalog(a);cfg=ip.config(a);all_records=[];expected={}
    for domain in DOMAINS:
        rows=[s for s in c['samples'] if s['domain']==domain]
        if [len(rows),sum(s['label']==0 for s in rows),sum(s['label']==1 for s in rows)]!=cfg['populations'][domain]:raise ValueError('incomplete frozen evaluation population')
        if len({s['sample_id'] for s in rows})!=len(rows):raise ValueError('duplicate domain sample ID')
        manifest=st.sealed_json(a.run_root/'predictions'/(domain+'.json'))
        if manifest['identity']!=ip.identity(a) or [(r['domain'],r['sample_id']) for r in manifest['records']]!=[(domain,s['sample_id']) for s in rows]:raise ValueError('incomplete prediction population')
        expected.update({(domain,r['sample_id']):r['seal'] for r in manifest['records']});all_records.append(manifest['seal'])
    a._prediction_record_seals=expected
    return st.digest(all_records)


def predict(a):
    from .runtime import host,extraction,reliable_precision,existing_query
    from .training import trained
    from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.fresh_inputs import Data
    reliable_precision(a.device);c=ip.catalog(a);model=None;calls=0;created=0;reused_q=0;refs=inference.References(a)
    cfg=ip.config(a);plan=ip.plan(a)
    for domain in DOMAINS:
        states=calibration_states(a,domain);samples=[s for s in c['samples'] if s['domain']==domain];records=[]
        pending=[s for s in samples if not (prediction_path(a,s)/'manifest.json').exists()]
        nets={}
        if pending:
            for family in cfg['active_families']:
                for seed in cfg['training_seeds']:nets[member(family,seed)]=trained(a,domain,family,seed)[0]
        with Data(Path(c['data_root'])) as data:
            for s in samples:
                p=prediction_path(a,s)
                if not (p/'manifest.json').exists():
                    qp=a.cache_root/'queries'/domain/s['sample_id']
                    if a.query_mode=='disk' and (qp/'manifest.json').exists():
                        x,qr=st.get(qp);q={l:x[f'q{l}'] for l in LAYERS};geometry=dict(grid=qr['metadata']['grid'],positions=x['positions'])
                        if qr['metadata']['sample']!=s or qr['metadata']['identity']!=ip.identity(a):raise ValueError('raw Q identity differs')
                    else:
                        prior=existing_query(a,s)
                        if prior:q,geometry,qorigin=prior;reused_q+=1
                        else:
                            if model is None:model,_=host(a)
                            q,geometry=extraction(model,data.image(s),a.device);calls+=1
                        if a.query_mode=='disk':st.put(qp,{**{f'q{l}':q[l] for l in LAYERS},'positions':np.asarray(geometry['positions'],np.int64)},
                            dict(identity=ip.identity(a),sample=s,grid=geometry['grid']))
                    nn,nr=inference.context(a,s,q,geometry,refs)
                    raw,origin=inference.baseline(a,s,nn,geometry,refs)
                    values={use:inference.head_inputs(a,s,q,geometry,nn if use else None,refs,use_reference=use) for use in {not b.startswith('NOREF_') for b in nets}}
                    for b,net in nets.items():raw[b]=cal.z_values(inference.logits(net,values[not b.startswith('NOREF_')],geometry['positions']),states[b]['patch'])
                    for family in cfg['active_families'] if cfg['enable_ensemble'] else []:raw[ensemble(family)]=cal.mean_member_z(raw,family,cfg['training_seeds'])
                    raw['positions']=np.asarray(geometry['positions'],np.int64)
                    # Retain exact original NN evidence, not duplicate residual vectors.
                    raw.update({k:v for k,v in nn.items() if k.startswith(('ds','dt','si','ti'))})
                    shape=[s['image_height']//4,s['image_width']//4]
                    scores={b:cal.response(raw[b],raw['positions'],geometry['grid'],shape,states[b]['image'])[1] for b in plan['branches']}
                    st.put(p,raw,dict(identity=ip.identity(a),sample=s,grid=geometry['grid'],output_shape=shape,scores=scores,
                        calibration_seals={b:v['seal'] for b,v in states.items()},baseline_origin=origin,NN_record=nr,
                        original_distances_unchanged=True,raw_logits_not_probabilities=True,GT_used=False));created+=1
                x,r=st.get(p);validate_prediction(a,s,x,r,states);records.append(dict(domain=domain,sample_id=s['sample_id'],seal=r['seal']))
                print(f'[residual predict] {domain}/{s["sample_id"]} branches={len(plan["branches"])} same_Q',flush=True)
        st.sealed_json(a.run_root/'predictions'/(domain+'.json'),dict(identity=ip.identity(a),records=records,status='domain_predictions_frozen'))
    seal=verify(a);st.sealed_json(a.run_root/'predictions_frozen.json',dict(identity=ip.identity(a),images=len(c['samples']),seal_of_manifests=seal,GT_used=False))
    return dict(status='all_predictions_frozen',new_images=created,query_forward_calls=calls,reused_full_raw_queries=reused_q)
