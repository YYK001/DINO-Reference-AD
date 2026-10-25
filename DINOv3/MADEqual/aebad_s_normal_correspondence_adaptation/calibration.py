"""Each representation gets Source-only patch MAD; ensemble fits image MAD on mean z."""
import numpy as np
from . import storage as st, inputs as ip
from .protocol import DOMAINS, LAYERS, SEEDS, U, MEMBERS, ENS
from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import statistics,fit_image_mad,ch1_score
from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic.numerics import bilinear


def memory_path(a,domain,seed):return a.cache_root/'transformed'/domain/f'seed{seed}'


def transform_memory(a):
    from .model import transform
    from .training import trained
    from .runtime import reliable_precision
    reliable_precision(a.device);store=st.Store();source,sr=store.load(a.cache_root/'source_original');c=ip.catalog(a)
    ledger=[]
    for domain in DOMAINS:
        supports=[store.load(ip.feature_path(a,r)) for r in c['supports'][domain]]
        for seed in SEEDS:
            for l in LAYERS:
                path=memory_path(a,domain,seed)/f'L{l}'
                net,done=trained(a,domain,seed,l,a.device)
                meta=dict(identity=ip.identity(a),model=done['checkpoint_sha256'],source_seal=sr['seal'],
                    support_seals=[r['seal'] for _,r in supports],source_row_count=len(source[f'L{l}']),
                    target_row_counts=[len(x[f'full_L{l}']) for x,_ in supports],
                    source_row_mapping='identity: original bank row i -> transformed row i',
                    target_row_mapping='manifest support order; within each support original compact row and full_positions')
                if not (path/'manifest.json').exists():
                    values=dict(source=transform(net,source[f'L{l}'],a.device),
                        target=np.concatenate([transform(net,x[f'full_L{l}'],a.device) for x,_ in supports]))
                    st.put(path,values,meta)
                _,record=st.get(path)
                if record['metadata']!=meta:raise ValueError('transformed memory/model inputs differ')
                ledger.append(dict(domain=domain,seed=seed,layer=l,seal=record['seal'],**meta))
                print(f'[normal memory] {domain}/seed{seed}/L{l} all original rows transformed',flush=True)
    st.sealed_json(a.run_root/'memory_mapping.json',dict(identity=ip.identity(a),records=ledger))
    return dict(status='all_memory_rows_transformed',jobs=len(ledger))


def z_values(raw,state):
    return ((np.asarray(raw,np.float32)-np.asarray(state['medians'],np.float32)[:,None])/
        np.asarray(state['scales'],np.float32)[:,None]).astype(np.float32)


def from_z(z,positions,grid,shape):
    """No second patch MAD. Same native FP32 mean, interpolation, valid3x3 stats."""
    from scipy.ndimage import uniform_filter
    physical=np.empty_like(z);physical[:,positions]=z
    physical=physical.reshape(4,*grid)
    layer_maps=[bilinear(x,shape) for x in physical]
    response=bilinear(physical.mean(0,dtype=np.float32),shape)
    peak=np.asarray([m.max() for m in layer_maps],np.float32)
    local=np.asarray([((uniform_filter(x,size=3,mode='constant',cval=0.)*9)[1:-1,1:-1]/9).max() for x in physical],np.float32)
    return response,peak,local


def fit_patch(raws):
    from DINOv3.relation_reliability.calibration import fit_calibrators
    _,mad=fit_calibrators({l:np.concatenate([r[j].ravel() for r in raws]) for j,l in enumerate(LAYERS)})
    return dict(medians=mad.medians.tolist(),scales=mad.scales.tolist())


def fit_image(records,patch=None):
    peaks=[];locals_=[]
    for x,m in records:
        z=x if patch is None else z_values(x,patch)
        _,peak,local=from_z(z,np.asarray(m['positions'],np.int64),m['grid'],m['output_shape'])
        peaks.append(peak);locals_.append(local)
    state={}
    for name,values in [('peak',peaks),('local',locals_)]:
        median,scale=fit_image_mad(values);state[name]=dict(medians=median.tolist(),scales=scale.tolist())
    return state


def state(a,domain,branch):
    return st.sealed_json(a.run_root/'calibration'/domain/(branch+'.json'))


def calibrate(a):
    import torch
    from .model import exact,transform
    from .training import trained
    from .runtime import reliable_precision
    reliable_precision(a.device);c=ip.catalog(a);store=st.Store();source,sr=store.load(a.cache_root/'source_original')
    rows=[r for r in c['source'] if r['calibration']]
    features=[ip.feature_path(a,r) for r in rows]
    jobs=[('source',U,None)]+[(d,b,s) for d in DOMAINS for b,s in zip(MEMBERS,SEEDS)]
    for domain,branch,seed in jobs:
        destination=a.run_root/'calibration'/domain/(branch+'.json')
        if destination.exists():
            saved=state(a,domain,branch)
            if saved['identity']!=ip.identity(a):raise ValueError('calibration inputs changed')
            for rec in saved['distance_records']:
                _,r=store.load(rec['path'])
                if r['seal']!=rec['seal']:raise ValueError('calibration distances changed')
            continue
        values=[];models=[]
        # One model's four Source banks resident at a time, not all nine.
        for l in LAYERS:
            if seed is None:values.append(torch.as_tensor(np.array(source[f'L{l}']),device=a.device));models.append(None)
            else:
                net,_=trained(a,domain,seed,l,a.device);models.append(net)
                bank,_=store.load(memory_path(a,domain,seed)/f'L{l}')
                values.append(torch.as_tensor(np.array(bank['source']),device=a.device))
        raw_records=[];raws=[];image_records=[]
        for i,(feature,row) in enumerate(zip(features,rows),1):
            x,rec=store.load(feature)
            path=a.cache_root/'calibration_distances'/domain/branch/f'{i:03d}'
            meta=dict(identity=ip.identity(a),feature_seal=rec['seal'],domain=domain,branch=branch,
                positions=x['full_positions'].tolist(),grid=rec['metadata']['grid'],output_shape=rec['metadata']['output_shape'],
                reference='transformed_Source_only' if seed else 'original_Source_only',relative_path=row['relative_path'])
            if not (path/'manifest.json').exists():
                ds=[]
                for l,net,bank in zip(LAYERS,models,values):
                    q=x[f'full_L{l}'] if net is None else transform(net,x[f'full_L{l}'],a.device)
                    distance,_=exact(q,bank,a.device);ds.append(distance)
                st.put(path,dict(raw=np.stack(ds)),meta)
            z,r=store.load(path)
            if r['metadata']!=meta:raise ValueError('calibration geometry/source differs')
            raw_records.append(dict(path=str(path),seal=r['seal']));raws.append(z['raw']);image_records.append((z['raw'],meta))
            print(f'[normal calibrate] {domain}/{branch} {i}/{len(rows)}',flush=True)
        patch=fit_patch(raws);image=fit_image(image_records,patch)
        st.sealed_json(destination,dict(identity=ip.identity(a),patch=patch,image=image,distance_records=raw_records,
            calibration_images=66,scope='original_Source_only_no_Target',GT_used=False))
        del values,models
    for domain in DOMAINS:
        states=[state(a,domain,b) for b in MEMBERS];records=[]
        for i in range(len(rows)):
            zs=[];meta=None
            for s in states:
                x,r=store.load(s['distance_records'][i]['path']);zs.append(z_values(x['raw'],s['patch']));meta=r['metadata']
            records.append((np.stack(zs).mean(0,dtype=np.float32),meta))
        ensemble=dict(identity=ip.identity(a),image=fit_image(records),patch=None,
            member_seals=[s['seal'] for s in states],combination='mean_of_independently_patch_calibrated_z_no_second_patch_MAD',
            calibration_images=66,scope='original_Source_only_no_Target',GT_used=False)
        st.sealed_json(a.run_root/'calibration'/domain/(ENS+'.json'),ensemble)
    return dict(status='all_independent_Source_calibrations_complete')


def response(z,positions,grid,shape,image_state):
    m,peak,local=from_z(z,positions,grid,shape)
    return m,dict(CH1=ch1_score(peak,local,image_state),CURRENT=float(m.max()))
