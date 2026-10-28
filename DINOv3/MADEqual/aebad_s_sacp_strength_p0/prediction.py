"""Three new calibrated maps replayed from frozen raw_union and eta only."""
import numpy as np
from . import inputs as ip,method
from .inputs import st
from .protocol import CONFIG,DOMAINS,LAYERS,BRANCHES,MAIN,UNIFORM,HALF,U
from .boundary import COUNTERS


def prediction_path(a,s):return a.cache_root/'predictions'/s['domain']/s['sample_id']
def calibration_states(a,domain):
    state=st.sealed_json(a.run_root/'input_distance_mad.json');return {b:state for b in BRANCHES}


def validate_prediction(a,s,x,r,states):
    m=r['metadata'];p=ip.parent(a);n=int(np.prod(m['grid']))
    if m['identity']!=ip.identity(a) or m['sample']!=s or m['parent_seal']!=p.records[s['domain'],s['sample_id']] or m['calibration_seal']!=st.sealed_json(a.run_root/'input_distance_mad.json')['seal'] or not m['replayed_half_verified']:raise ValueError('extension prediction identity differs')
    if x['positions'].dtype!=np.int64 or not np.array_equal(np.sort(x['positions']),np.arange(n)):raise ValueError('extension physical positions incomplete')
    if m['output_shape']!=[s['image_height']//4,s['image_width']//4]:raise ValueError('extension map geometry differs')
    for b in BRANCHES:
        if x[b].shape!=(4,n) or x[b].dtype!=np.float32 or not np.isfinite(x[b]).all():raise ValueError('invalid new branch z')
    expected=getattr(a,'_prediction_seals',{}).get((s['domain'],s['sample_id']))
    if expected is not None and expected!=r['seal']:raise ValueError('frozen extension prediction changed')
    return m


def verify(a,complete=True):
    c=ip.catalog(a);expected={};seals=[]
    for d in DOMAINS:
        samples=[s for s in c['samples'] if s['domain']==d]
        if [len(samples),sum(s['label']==0 for s in samples),sum(s['label']==1 for s in samples)]!=CONFIG['populations'][d]:raise ValueError('full extension population required')
        r=st.sealed_json(a.run_root/'predictions'/(d+'.json'))
        if r['identity']!=ip.identity(a) or [(x['domain'],x['sample_id']) for x in r['records']]!=[(d,s['sample_id']) for s in samples]:raise ValueError('incomplete replay freeze; GT refused')
        expected.update({(d,x['sample_id']):x['seal'] for x in r['records']});seals.append(r['seal'])
    digest=st.digest(seals)
    if complete:
        frozen=st.sealed_json(a.run_root/'predictions_frozen.json')
        if frozen['identity']!=ip.identity(a) or frozen['images']!=len(c['samples']) or frozen['seal_of_manifests']!=digest or frozen['GT_used']:raise ValueError('global replay commit incomplete; GT refused')
        for name,h in frozen['replay_files'].items():
            if st.sha(a.run_root/name)!=h:raise ValueError('replay verification/reduction table changed')
    a._prediction_seals=expected;return digest


def reduction_rows(sample,arrays,raw,half,audits):
    result=[]
    for b in BRANCHES:
        for j,l in enumerate(LAYERS):
            u=arrays['raw_union'][j].astype(np.float64);reduction=u-raw[b][j];relative=np.divide(reduction,u,out=np.zeros_like(u),where=u>0)
            result.append(dict(domain=sample['domain'],sample_id=sample['sample_id'],layer=l,branch=b,reference=U,
                total_native_reduction=float(reduction.sum()),half_total_native_reduction=float((u-half[b][j]).sum()),
                mean_native_reduction=float(reduction.mean()),relative_reduction_mean=float(relative.mean()),
                **{f'relative_reduction_p{int(q*100):02d}':float(np.quantile(relative,q)) for q in (0,.05,.5,.95,1)},
                zero_union_positions=int(np.sum(u==0)),positions=len(u),uniform_coefficient=audits[j]['coefficient'] if b==UNIFORM else None,
                main_total_reduction=audits[j]['main_total_reduction'],uniform_total_reduction=audits[j]['uniform_total_reduction'],
                uniform_matching_absolute_error=audits[j]['absolute_error'],uniform_FP32_sum_roundoff_bound=audits[j]['FP32_sum_roundoff_bound'],
                distance_unit='native_Euclidean_over_1280_once',relative_zero_union_rule='0; counted separately'))
    return result


def replay(a):
    ip.freeze(a);ip.validate(a);p=ip.parent(a);state=calibration_states(a,'background')[MAIN];created=0;reused=0;all_checks=[];reductions=[]
    for d in DOMAINS:
        records=[]
        for s in [s for s in p.catalog['samples'] if s['domain']==d]:
            path=prediction_path(a,s)
            if (path/'manifest.json').exists():
                parent_record=st.sealed_json(p.path(s)/'manifest.json')
                if parent_record['seal']!=p.records[d,s['sample_id']]:raise ValueError('resumed parent prediction manifest changed')
                x,r=st.get(path);reused+=1
            else:
                old,parent_record=p.load(s);m=parent_record['metadata'];checks,half=method.replay_half(old,m,state)
                raw,audit=method.values(old,1.)
                z={b:method.calibrated(raw[b],old['raw_union'],old[U],state['patch']) for b in BRANCHES}
                scores={b:method.response(z[b],old['positions'],m['grid'],m['output_shape'],state['image'])[1] for b in BRANCHES}
                decrease=reduction_rows(s,old,raw,half,audit)
                r=st.put(path,dict(z,positions=np.array(old['positions'])),dict(identity=ip.identity(a),sample=s,grid=m['grid'],output_shape=m['output_shape'],
                    parent_identity=p.identity,parent_seal=parent_record['seal'],parent_relative_path='predictions/'+d+'/'+s['sample_id'],
                    calibration_seal=state['seal'],scores=scores,checks=checks,reductions=decrease,replayed_half_verified=True,GT_used=False))
                x,_=st.get(path);created+=1
            validate_prediction(a,s,x,r,{b:state for b in BRANCHES});records.append(dict(domain=d,sample_id=s['sample_id'],seal=r['seal']))
            all_checks.extend(r['metadata']['checks']);reductions.extend(r['metadata']['reductions'])
            print(f'[SACP L1 replay] {d}/{s["sample_id"]} three_new_maps_cached_eta',flush=True)
        st.sealed_json(a.run_root/'predictions'/(d+'.json'),dict(identity=ip.identity(a),records=records))
    seal=verify(a,complete=False)
    st.write_csv(a.run_root/'replay_verification.csv',all_checks);st.write_csv(a.run_root/'native_reductions.csv',reductions)
    st.sealed_json(a.run_root/'predictions_frozen.json',dict(identity=ip.identity(a),images=len(p.catalog['samples']),seal_of_manifests=seal,GT_used=False,
        replay_files={name:st.sha(a.run_root/name) for name in ('replay_verification.csv','native_reductions.csv')}))
    verify(a)
    return dict(status='replay_frozen',new_images=created,reused_images=reused,parent_half_replay_checks=len(all_checks),
        numerical_failed_checks=sum(not r['verified'] for r in all_checks),Torch_imported='torch' in __import__('sys').modules,**COUNTERS)
