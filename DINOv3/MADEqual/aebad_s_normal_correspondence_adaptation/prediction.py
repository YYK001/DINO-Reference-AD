"""One frozen host forward per image, shared by all six branches, GT-free."""
from pathlib import Path
import numpy as np
from . import storage as st, inputs as ip
from .protocol import DOMAINS,LAYERS,SEEDS,BRANCHES,MEMBERS,U,S,ENS,CONFIG
from . import calibration as cal


def prediction_path(a,s):return a.cache_root/'predictions'/s['domain']/s['sample_id']


def calibration_states(a,domain):
    states={U:cal.state(a,'source',U)};states[S]=states[U]
    states.update({b:cal.state(a,domain,b) for b in (*MEMBERS,ENS)})
    if any(s['identity']!=ip.identity(a) for s in states.values()):raise ValueError('calibration/model run mismatch')
    return states


def validate_prediction(a,s,x,r,states):
    m=r['metadata']
    if m['identity']!=ip.identity(a) or m['sample']!=s or m['calibration_seals']!={b:v['seal'] for b,v in states.items()}:
        raise ValueError('prediction sample/model/calibration identity differs')
    n=int(np.prod(m['grid']));p=x['positions']
    if p.dtype!=np.int64 or not np.array_equal(np.sort(p),np.arange(n)):raise ValueError('physical query positions differ')
    for b in BRANCHES:
        if x[b].shape!=(4,n) or x[b].dtype!=np.float32 or not np.isfinite(x[b]).all():raise ValueError('invalid frozen z: '+b)
    return m


def verify(a):
    c=ip.catalog(a);store=st.Store();all_records=[]
    for domain in DOMAINS:
        rows=[r for r in c['samples'] if r['domain']==domain]
        if [len(rows),sum(r['label']==0 for r in rows),sum(r['label']==1 for r in rows)]!=CONFIG['populations'][domain]:
            raise ValueError('incomplete frozen evaluation population: '+domain)
        if len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('duplicate domain sample ID')
    for domain in DOMAINS:
        manifest=st.sealed_json(a.run_root/'predictions'/(domain+'.json'));states=calibration_states(a,domain)
        expected=[s for s in c['samples'] if s['domain']==domain]
        if manifest['identity']!=ip.identity(a) or [(r['domain'],r['sample_id']) for r in manifest['records']]!=[(domain,s['sample_id']) for s in expected]:
            raise ValueError('incomplete prediction population; GT evaluation refused')
        for s,r in zip(expected,manifest['records']):
            x,record=store.load(prediction_path(a,s))
            if record['seal']!=r['seal']:raise ValueError('frozen prediction modified')
            validate_prediction(a,s,x,record,states)
        all_records.append(manifest['seal'])
    return st.digest(all_records)


def predict(a):
    import torch
    from .runtime import host,extraction,reliable_precision
    from .training import trained
    from .model import transform,exact
    from DINOv3.MADEqual.aebad_s_fixed_target_graph_p0.runtime import original_nearest
    from DINOv3.MADEqual.aebad_s_target_local_segment.geometry import geometry_layer
    from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.fresh_inputs import Data
    reliable_precision(a.device);c=ip.catalog(a);store=st.Store();source,_=store.load(a.cache_root/'source_original')
    source_device=None;model=None;calls=0;new=0
    for domain in DOMAINS:
        samples=[s for s in c['samples'] if s['domain']==domain];states=calibration_states(a,domain)
        manifest_path=a.run_root/'predictions'/(domain+'.json')
        if manifest_path.exists():
            # Final global verification below still verifies completed payloads once per stage.
            continue
        supports=[store.load(ip.feature_path(a,r))[0] for r in c['supports'][domain]]
        pending=[s for s in samples if not (prediction_path(a,s)/'manifest.json').exists()]
        nets={};memories={}
        if pending:
            if source_device is None:source_device={l:torch.as_tensor(np.array(source[f'L{l}']),device=a.device) for l in LAYERS}
            for seed in SEEDS:
                for l in LAYERS:
                    nets[seed,l]=trained(a,domain,seed,l,a.device)[0]
                    x,_=store.load(cal.memory_path(a,domain,seed)/f'L{l}')
                    # ~13 GiB for three complete domain memories, loaded ONCE in CPU RAM.
                    # Only one transformed model/layer is transferred to GPU per search.
                    memories[seed,l]={k:np.array(x[k]) for k in ('source','target')}
        records=[]
        with Data(Path(c['data_root'])) as data:
            for sample in samples:
                path=prediction_path(a,sample)
                if not (path/'manifest.json').exists():
                    qpath=a.cache_root/'queries'/domain/sample['sample_id']
                    if a.query_mode=='disk' and (qpath/'manifest.json').exists():
                        qx,qr=store.load(qpath)
                        if qr['metadata']['sample']!=sample or qr['metadata']['identity']!=ip.identity(a):raise ValueError('query extraction identity differs')
                        q={l:qx['L'+str(l)] for l in LAYERS};m=qr['metadata']['geometry'];m['positions']=np.asarray(m['positions'],np.int64)
                    else:
                        if model is None:model,_=host(a)
                        q,m=extraction(model,data.image(sample),a.device);calls+=1
                        if a.query_mode=='disk':st.put(qpath,{f'L{l}':q[l] for l in LAYERS},
                            dict(identity=ip.identity(a),sample=sample,geometry={**m,'positions':m['positions'].tolist()}))
                    raw={b:[] for b in (U,S,*MEMBERS)}
                    for l in LAYERS:
                        banks={i+1:x[f'full_L{l}'] for i,x in enumerate(supports)}
                        sd,td,ti=original_nearest(q[l],source_device[l],banks)
                        geo,_=geometry_layer(td,ti,sd,list(banks.values()))
                        raw[U].append(geo['K8_UNION']);raw[S].append(geo['K8_SEGMENT'])
                        for seed,b in zip(SEEDS,MEMBERS):
                            hq=transform(nets[seed,l],q[l],a.device)
                            memory=memories[seed,l]
                            ds,_=exact(hq,memory['source'],a.device);dt,_=exact(hq,memory['target'],a.device)
                            raw[b].append(np.minimum(ds,dt))
                    arrays={b:cal.z_values(np.stack(values),states[b]['patch']) for b,values in raw.items()}
                    arrays[ENS]=np.stack([arrays[b] for b in MEMBERS]).mean(0,dtype=np.float32)
                    arrays['positions']=np.asarray(m['positions'],np.int64)
                    shape=[sample['image_height']//4,sample['image_width']//4]
                    scores={b:cal.response(arrays[b],arrays['positions'],m['grid'],shape,states[b]['image'])[1] for b in BRANCHES}
                    metadata=dict(identity=ip.identity(a),sample=sample,grid=m['grid'],output_shape=shape,scores=scores,
                        calibration_seals={b:v['seal'] for b,v in states.items()},GT_used=False,
                        values='independently_calibrated_z; not raw distances; ensemble has NO second patch MAD')
                    st.put(path,arrays,metadata);new+=1
                x,r=store.load(path);validate_prediction(a,sample,x,r,states)
                records.append(dict(domain=domain,sample_id=sample['sample_id'],seal=r['seal']))
                print(f'[normal query] {domain}/{sample["sample_id"]} six_branches_same_Q',flush=True)
        st.sealed_json(manifest_path,dict(identity=ip.identity(a),records=records,status='domain_predictions_frozen'))
        del memories,nets,supports
    seal=verify(a)
    st.sealed_json(a.run_root/'predictions_frozen.json',dict(identity=ip.identity(a),seal_of_manifests=seal,images=len(c['samples']),GT_used=False))
    return dict(status='all_predictions_frozen',images=len(c['samples']),new_images=new,query_forward_calls=calls)
