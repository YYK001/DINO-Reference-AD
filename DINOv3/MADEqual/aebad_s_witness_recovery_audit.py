"""CPU-only drift observation; deliberately outside the frozen witness module.

Never changes recovery seals, declares historical checks passed, runs a model,
uses GT, or starts the experiment. Reuses the original geometry/maps/scores.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
import numpy as np
from DINOv3.MADEqual.aebad_s_support_witness_segment_p0 import inputs as ip,recovery as rec,storage as st
from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.method import LAYERS,S
from DINOv3.MADEqual.aebad_s_target_local_segment.geometry import geometry_layer,DIST_ATOL,DIST_RTOL
from DINOv3.MADEqual.aebad_s_neighborhood_segment_p0.analysis import score_maps,array_sha


def difference(actual,expected):
    a,b=np.asarray(actual,np.float64),np.asarray(expected,np.float64)
    if a.shape!=b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('invalid drift comparison')
    signed=a-b;err=np.abs(signed);bad=err>DIST_ATOL+DIST_RTOL*np.abs(b)
    return dict(positions=a.size,failed_positions=int(bad.sum()),strict_verified=not bool(bad.any()),
        max_abs_error=float(err.max()),mean_abs_error=float(err.mean()),p99_abs_error=float(np.quantile(err,.99)),
        mean_signed_difference=float(signed.mean()),atol=DIST_ATOL,rtol=DIST_RTOL)


def audit_image(bundle,sample,banks):
    raw,meta,origin=bundle.load(sample);checks=[];new=[];d,sid=sample['domain'],sample['sample_id']
    for j,l in enumerate(LAYERS):
        result,_=geometry_layer(raw['target_distance'][:,j],raw['target_index'][:,j],raw['source_distance'][j],
            [banks[l][i] for i in range(1,9)])
        new.append(result['K8_SEGMENT'])
        for kind,key in [('segment_distance','K8_SEGMENT'),('endpoint_length','endpoint_length')]:
            row=dict(domain=d,sample_id=sid,layer=l,check_kind=kind,**difference(result[key],raw[kind][j]))
            if kind=='segment_distance':row['max_abs_error_over_Source_MAD_scale']=row['max_abs_error']/float(bundle.scales[j])
            checks.append(row)
    context=dict(medians=bundle.medians,scales=bundle.scales,state=bundle.state);meta=dict(meta,positions=raw['positions'])
    old_current,old_ch1,old_map,_=score_maps(context,raw['segment_distance'],meta)
    new_current,new_ch1,new_map,_=score_maps(context,np.stack(new),meta)
    reference=bundle.references[d,sid,S];current=reference.get('CURRENT_check_score',reference.get('current_check_score'))
    if abs(old_ch1-float(reference['image_score']))>1e-5 or abs(old_current-float(current))>1e-5:
        raise ValueError('same-cache historical score reconstruction changed: '+d+'/'+sid)
    if array_sha(old_map)!=bundle.maps[d,sid,S]['map_float32_sha256']:
        raise ValueError('same-cache frozen SEGMENT map SHA differs: '+d+'/'+sid)
    delta=new_map.astype(np.float64)-old_map
    scores=dict(domain=d,sample_id=sid,CH1_old=old_ch1,CH1_recovered_geometry=new_ch1,CH1_signed_difference=new_ch1-old_ch1,
        CURRENT_old=old_current,CURRENT_recovered_geometry=new_current,CURRENT_signed_difference=new_current-old_current,
        map_max_abs_error=float(np.abs(delta).max()),map_mean_abs_error=float(np.abs(delta).mean()),map_mean_signed_difference=float(delta.mean()),
        old_peak_position=int(np.argmax(old_map)),recovered_geometry_peak_position=int(np.argmax(new_map)),
        same_cache_map_SHA_verified=True,same_cache_score_atol_1e_minus5_verified=True)
    return checks,scores


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',required=True,type=Path)
    p.add_argument('--data',required=True,type=Path);p.add_argument('--cpu-workers',type=int,default=8)
    p.add_argument('--domains',nargs='+',choices=ip.DOMAINS,default=['illumination','view'])
    a=p.parse_args(argv)
    if not 1<=a.cpu_workers<=8:raise ValueError('cpu-workers must be 1..8')
    tick=time.perf_counter();paths=ip.parent.defaults();paths['data']=a.data
    bundle=ip.Bundle(paths);root=a.run_root.resolve()/'recovered_supports';spec=st.read_json(root/'recovery_inputs.json')
    if spec['identity']!=st.digest(spec['frozen']) or spec['frozen']['original']!=rec.original_proof(bundle):raise ValueError('original recovery identity mismatch')
    records,locations,entries=rec.recovered_locations(bundle,root,spec)
    out=a.run_root.resolve()/('drift_audit_'+time.strftime('%Y%m%d_%H%M%S'));out.mkdir(exist_ok=False)
    st.write_json(out/'input_manifest.json',dict(recovery_inputs_sha256=st.sha(root/'recovery_inputs.json'),files=entries,
        original=rec.original_proof(bundle),audit_source_sha256=st.sha(Path(__file__)),domains=a.domains,GT_used=False,model_inference=False))
    errors=[];all_checks=[];all_scores=[]
    for d in dict.fromkeys(a.domains):
        banks={l:{} for l in LAYERS}
        # Hold one domain (about2.13GiB), read each complete NPZ once. No Q.
        for row in bundle.original_expected[d]:
            with np.load(locations[d,row['support_index']],allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
            rec.validate_arrays(arrays,row)
            for l in LAYERS:banks[l][row['support_index']]=arrays[f'layer_{l}']
        samples=[s for s in bundle.samples if s['domain']==d]
        def work(sample):
            try:return audit_image(bundle,sample,banks),None
            except Exception as exc:return None,dict(domain=d,sample_id=sample['sample_id'],error=f'{type(exc).__name__}: {exc}')
        with ThreadPoolExecutor(max_workers=a.cpu_workers) as pool:
            for i,(result,error) in enumerate(pool.map(work,samples)):
                if error:errors.append(error)
                else:
                    checks,scores=result;all_checks.extend(checks);all_scores.append(scores)
                if (i+1)%25==0 or i+1==len(samples):print(f'[CPU recovery drift] {d} {i+1}/{len(samples)} errors={len(errors)}',flush=True)
        st.write_csv(out/(d+'_geometry.csv'),[r for r in all_checks if r['domain']==d])
        st.write_csv(out/(d+'_scores.csv'),[r for r in all_scores if r['domain']==d]);del banks,arrays
    groups=defaultdict(list)
    for row in all_checks:groups[row['domain'],row['layer'],row['check_kind']].append(row)
    summary=[]
    for (d,l,kind),rows in sorted(groups.items()):
        result=dict(domain=d,layer=l,check_kind=kind,images=len(rows),strict_failed_images=sum(not r['strict_verified'] for r in rows),
            failed_positions=sum(r['failed_positions'] for r in rows),positions=sum(r['positions'] for r in rows),max_abs_error=max(r['max_abs_error'] for r in rows))
        if kind=='segment_distance':result['max_abs_error_over_Source_MAD_scale']=max(r['max_abs_error_over_Source_MAD_scale'] for r in rows)
        summary.append(result)
    st.write_csv(out/'geometry_summary.csv',summary)
    score_summary=[]
    for d in dict.fromkeys(a.domains):
        rows=[r for r in all_scores if r['domain']==d]
        if not rows:continue
        score_summary.append(dict(domain=d,images=len(rows),map_max_abs_error=max(r['map_max_abs_error'] for r in rows),
            CH1_max_abs_change=max(abs(r['CH1_signed_difference']) for r in rows),
            CURRENT_max_abs_change=max(abs(r['CURRENT_signed_difference']) for r in rows),
            CH1_changes_over_historical_1e4=sum(abs(r['CH1_signed_difference'])>1e-4 for r in rows),
            CURRENT_changes_over_historical_1e4=sum(abs(r['CURRENT_signed_difference'])>1e-4 for r in rows),
            peak_position_changes=sum(r['old_peak_position']!=r['recovered_geometry_peak_position'] for r in rows)))
    st.write_csv(out/'score_summary.csv',score_summary)
    state=dict(status='audit_complete_not_experiment_approval' if not errors else 'audit_incomplete',
        errors=errors,geometry_summary=summary,score_summary=score_summary,seconds=time.perf_counter()-tick,
        strict_failures_preserved=True,formal_acceptance_policy_changed=False,GT_used=False,torch_imported='torch' in sys.modules)
    if state['torch_imported']:raise RuntimeError('CPU drift audit unexpectedly imported Torch')
    st.write_json(out/'status.json',state);print(json.dumps(state,ensure_ascii=False,indent=2));print('AUDIT_OUTPUT='+str(out),flush=True)
    return 0 if not errors else 2


if __name__=='__main__':raise SystemExit(main())
