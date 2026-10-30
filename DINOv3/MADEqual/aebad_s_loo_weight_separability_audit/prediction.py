"""Freeze ALL crossed responses before reading GT or doing separability audits."""
import time
import numpy as np
from . import inputs as ip,numerics as nm
from .inputs import st
from .protocol import *
from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.evaluation import shared


def budget(a,done=1,total=1):
    previous=sum(st.read_json(p)['seconds'] for p in a.output_root.glob('costs/*.json'))
    elapsed=time.perf_counter()-a._started
    estimate=st.read_json(a.output_root/'time_estimate.json') if (a.output_root/'time_estimate.json').exists() else {}
    projected=previous+elapsed*total/max(done,1)
    if a.stage in ('smoke-gt','predict'):projected=max(projected,estimate.get('full_seconds') or 0)
    if not a.allow_over_12h and ((previous+elapsed)>43200 or (done>=min(5,total) and projected>43200)):
        st.write_json(a.output_root/'budget_pause.json',dict(stage=a.stage,done=done,total=total,projected_seconds=projected,status='paused_12h_budget'))
        raise RuntimeError('projected work exceeds12h; committed units retained; explicit approval required')


def load_maps(a,d,m,row,only=None):
    p=ip.parent(a);z,pos,ref=ip.load(a,d,m,row);weights={w:p.weights[d,w] for w in WEIGHTS}
    if only:
        image,score=nm.kernel.weighted(z,pos,ref['grid'],ref['output_shape'],weights[only]);return {only:image},{only:score},None,z,pos,ref
    images,scores,stats=nm.responses(z,pos,ref['grid'],ref['output_shape'],weights)
    return images,scores,stats,z,pos,ref


def freeze(a):
    p=ip.parent(a);units=[];checks=[]
    for d in a.domains:
        for m in a.memory_seeds:
            sig=ip.signature(a,d,m);records=[]
            for i,row in enumerate(p.samples(d),1):
                dest=ip.record_path(a,d,m,row)
                if not dest.exists():
                    maps,scores,stats,z,pos,ref=load_maps(a,d,m,row);old=p.record(d,m,row);diagonal_checks=[]
                    for w,b in (('WS',SOURCE),(diagonal(m),TARGET)):
                        actual=shared.array_sha(maps[w]);expected=old['map_hashes'][b]
                        if actual!=expected:raise ValueError('original Source/diagonal map reproduction failed')
                        for rule in READOUTS:
                            delta=abs(scores[w][rule]-old['scores'][b][rule])
                            if delta>SCORE_ATOL:raise ValueError('original diagonal score reproduction failed')
                            item=dict(domain=d,memory_seed=m,sample_id=row['sample_id'],weight_source=w,check='diagonal_score_'+rule,max_abs_error=delta,atol=SCORE_ATOL,verified=True)
                            checks.append(item);diagonal_checks.append(item)
                        # Ranking uses the actual historical frozen scores, including exact ties.
                        scores[w]=dict(old['scores'][b])
                    r=st.sealed_json(dest,dict(**sig,image=row,response=ref,scores=scores,layer_statistics=stats.tolist(),map_hashes={w:shared.array_sha(image) for w,image in maps.items()},diagonal_verification=diagonal_checks,GT_used=False))
                r=st.sealed_json(dest)
                if any(r.get(k)!=v for k,v in sig.items()) or r['image']!=row or r['GT_used']:raise ValueError('cross prediction identity differs')
                records.append(dict(sample_id=row['sample_id'],seal=r['seal']))
                if i%25==0 or i==len(p.samples(d)):print(f'[cross freeze CPU] {d}/Z{m} {i}/{len(p.samples(d))} no_GT',flush=True)
                budget(a,i,len(p.samples(d)))
            result=st.sealed_json(ip.unit(a,d,m)/'responses_frozen.json',dict(**sig,records=records,images=len(records),GT_used=False))
            units.append(result['seal'])
    st.write_csv(a.output_root/'prediction_verification_current_session.csv',checks)
    return global_freeze(a,missing_ok=True)


def global_freeze(a,missing_ok=False):
    p=ip.parent(a);records=[];missing=[]
    for d in DOMAINS:
        for m in SEEDS:
            dest=ip.unit(a,d,m)/'responses_frozen.json'
            if not dest.exists():missing.append(f'{d}/Z{m}');continue
            r=st.sealed_json(dest);sig=ip.signature(a,d,m)
            if any(r.get(k)!=v for k,v in sig.items()) or [x['sample_id'] for x in r['records']]!=[s['sample_id'] for s in p.samples(d)] or r['GT_used']:raise ValueError('wrong/incomplete response group')
            for row,ref in zip(p.samples(d),r['records']):
                rec=st.sealed_json(ip.record_path(a,d,m,row))
                if rec['seal']!=ref['seal'] or rec['image']!=row or any(rec.get(k)!=v for k,v in sig.items()):raise ValueError('frozen crossed response modified')
            records.append(dict(domain=d,memory_seed=m,seal=r['seal']))
    if missing:
        if missing_ok:return dict(status='pending',pending=missing)
        raise ValueError('GT analysis requires ALL six crossed response sets frozen first: '+','.join(missing))
    return st.sealed_json(a.output_root/'responses_frozen.json',dict(identity=st.sealed_json(a.output_root/'inputs.json')['identity'],records=records,GT_used=False,images=sum(len(p.samples(d)) for d in DOMAINS)*len(SEEDS)))
