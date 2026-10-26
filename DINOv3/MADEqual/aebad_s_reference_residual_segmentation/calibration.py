"""Logit calibration under domain K8, distinct from old Source-only distance MAD."""
import numpy as np
from . import inputs as ip,storage as st,inference
from .protocol import DOMAINS,LAYERS,U,S,member,ensemble
from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.calibration import (
    z_values,from_z,fit_image,response)


def state(a,domain,branch):return st.sealed_json(a.run_root/'calibration'/domain/(branch+'.json'))


def fit_logits(values):
    from .protocol import CONFIG
    med=[];scales=[];audit=[]
    for j,l in enumerate(LAYERS):
        x=np.concatenate([v[j].ravel() for v in values]).astype(np.float32)
        if not np.isfinite(x).all():raise FloatingPointError('nonfinite Source-calibration logits')
        median=float(np.median(x));scale=1.4826*float(np.median(np.abs(x-median)))
        minimum=max(CONFIG['logit_minimum_scale'],CONFIG['logit_minimum_relative_scale']*max(1.,abs(median)))
        if not np.isfinite(scale) or scale<minimum:raise ValueError(f'near-constant logit output L{l}: MAD={scale}, minimum={minimum}; no tiny-denominator rescue')
        med.append(median);scales.append(scale);audit.append(dict(layer=l,median=median,scale=scale,minimum=minimum,range=float(np.ptp(x))))
    return dict(medians=np.asarray(med,np.float32).tolist(),scales=np.asarray(scales,np.float32).tolist()),audit


def image_state(records,patch=None):
    value=fit_image(records,patch)
    for k in ('peak','local'):
        if min(value[k]['scales'])<1e-6:raise ValueError('near-constant logit-derived image calibration: '+k)
    return value


def mean_member_z(values,family,seeds=None):
    """Input is already calibrated member z; no patch-MAD fit or application."""
    seeds=sorted(int(k.rsplit('SEED',1)[1]) for k in values if k.startswith(family+'_SEED')) if seeds is None else seeds
    return np.stack([values[member(family,s)] for s in seeds]).mean(0,dtype=np.float32)


def calibrate(a):
    from .training import trained
    from .retrieval import References
    from .runtime import reliable_precision
    reliable_precision(a.device);c=ip.catalog(a);refs=inference.References(a);rows=[r for r in c['source'] if r['calibration']]
    cfg=ip.config(a);plan=ip.plan(a);families=cfg['active_families'];seeds=cfg['training_seeds']
    baseline=st.sealed_json(a.run_root/'input_distance_mad.json')
    for domain in DOMAINS:
        for b in (U,S):st.sealed_json(a.run_root/'calibration'/domain/(b+'.json'),dict(identity=ip.identity(a),patch=baseline['patch'],image=baseline['image'],scope=baseline['scope']))
        nets={};models={}
        for family in families:
            for seed in seeds:
                net,done=trained(a,domain,family,seed);nets[member(family,seed)]=net;models[member(family,seed)]=done['checkpoint_sha256']
        records=[]
        for i,row0 in enumerate(rows,1):
            row=dict(row0,domain=domain);x,fr=st.get(ip.normal_path(a,row0));m=fr['metadata'];geometry=dict(grid=m['grid'],positions=x['positions'])
            q={l:x[f'q{l}'] for l in LAYERS};p=a.cache_root/'logit_calibration'/domain/f'{i:03d}'
            if not (p/'manifest.json').exists():
                nn,nr=inference.context(a,row,q,geometry,refs,cache_path=a.cache_root/'calibration_NN'/domain/f'{i:03d}')
                needs={not b.startswith('NOREF_') for b in nets}
                values={use:inference.head_inputs(a,row,q,geometry,nn if use else None,refs,use_reference=use) for use in needs}
                raw={b:inference.logits(net,values[not b.startswith('NOREF_')],x['positions']) for b,net in nets.items()}
                st.put(p,raw,dict(identity=ip.identity(a),feature_seal=fr['seal'],models=models,NN_seal=nr['seal'],grid=m['grid'],
                    positions=x['positions'].tolist(),output_shape=m['output_shape'],origin=row0['relative_path'],scope='Source_calibration_under_domain_formal_Source_plus_K8'))
            raw,r=st.get(p)
            if r['metadata']['models']!=models or r['metadata']['feature_seal']!=fr['seal']:raise ValueError('calibration model/feature identity changed')
            records.append(dict(path=str(p),seal=r['seal']));print(f'[residual logit calibration] {domain} {i}/66 heads={len(nets)}',flush=True)
        for b in plan['members']:
            values=[st.get(r['path'],(b,))[0][b] for r in records];patch,audit=fit_logits(values)
            image=image_state([(x,st.get(r['path'],(b,))[1]['metadata']) for x,r in zip(values,records)],patch)
            st.sealed_json(a.run_root/'calibration'/domain/(b+'.json'),dict(identity=ip.identity(a),patch=patch,image=image,
                model_sha256=models[b],records=records,logit_audit=audit,scope='Source_calibration_under_domain_formal_Source_plus_K8',sigmoid_applied=False))
        for family in families if cfg['enable_ensemble'] else []:
            members=[state(a,domain,member(family,s)) for s in seeds];combined=[]
            for rec in records:
                raw,r=st.get(rec['path']);z=mean_member_z({member(family,s):z_values(raw[member(family,s)],v['patch']) for s,v in zip(seeds,members)},family,seeds)
                combined.append((z,r['metadata']))
            st.sealed_json(a.run_root/'calibration'/domain/(ensemble(family)+'.json'),dict(identity=ip.identity(a),patch=None,
                image=image_state(combined),members=[v['seal'] for v in members],scope='mean_member_z_domain_conditioned_Source_calibration_no_second_patch_MAD'))
    return dict(status='all_domain_conditioned_logit_calibrations_complete')
