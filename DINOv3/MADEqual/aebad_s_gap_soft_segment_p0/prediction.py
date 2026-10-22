"""Atomic per-image CPU prediction records and immutable resumable freeze."""
from pathlib import Path
import json
import os
import time
import numpy as np
from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import statistics,ch1_score
from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic.numerics import check
from DINOv3.MADEqual.aebad_s_neighborhood_segment_p0.prediction import bounded_images
from . import inputs as ip
from .method import NEW,BASES,BRANCHES,compute


def score(bundle,values,positions,meta):
    current,peak,local=statistics(values,positions,tuple(meta['grid']),tuple(meta['output_shape']),bundle.medians,bundle.scales)
    return dict(CH1=ch1_score(peak,local,bundle.state),CURRENT=current)


def one(bundle,sample):
    raw,meta,origin=bundle.load(sample)
    new,reduction,_,_=compute(raw['source_distance'],raw['target_distance'],raw['segment_distance'],bundle.medians,bundle.scales)
    values=dict(zip(BASES,(raw['union_distance'],raw['segment_distance'],raw['pair_distance'])))|new
    scores=[];checks=[];d,sid=sample['domain'],sample['sample_id']
    for b in BRANCHES:
        sc=score(bundle,values[b],raw['positions'],meta)
        if b in BASES:
            ref=bundle.references[d,sid,b]
            check(checks,sid,'same_parent_CH1',sc['CH1'],float(ref['image_score']),domain=d,branch=b)
            current=ref.get('CURRENT_check_score',ref.get('current_check_score'))
            if current not in ('',None):check(checks,sid,'same_parent_CURRENT',sc['CURRENT'],float(current),domain=d,branch=b)
        scores.append(dict(domain=d,sample_id=sid,relative_path=sample['relative_path'],branch=b,**sc,
            Source_image_state_sha256=bundle.state['state_sha256']))
    if not all(r['verified'] for r in checks):raise ValueError('same-parent scores changed; no tolerance relaxation')
    metadata=dict(identity=bundle.identity,domain=d,sample_id=sid,relative_path=sample['relative_path'],
        image_sha256=sample['image_sha256'],supports=bundle.supports[d],origin=origin,grid=meta['grid'],output_shape=meta['output_shape'],
        distance_unit='euclidean_over_1280_once',GT_used=False,labels_used=False)
    return dict(positions=raw['positions'],**new),metadata,scores,[dict(domain=d,sample_id=sid,**r) for r in reduction],checks


def load_record(out,record,identity):
    record_path=out/ip.bg.safe(record['record'])
    if ip.sha(record_path)!=record['record_sha256']:raise ValueError('prediction record changed after commit')
    data=ip.read_json(record_path);p=out/ip.bg.safe(data['file'])
    if data['identity']!=identity or ip.sha(p)!=data['sha256']:raise ValueError('prediction NPZ modified; do not re-enroll its hash')
    with np.load(p,allow_pickle=False) as f:
        metadata=json.loads(str(f['metadata'].item()));a={k:f[k].copy() for k in f.files if k!='metadata'}
    if metadata!=data['metadata'] or metadata['identity']!=identity:raise ValueError('prediction identity changed')
    n=int(np.prod(metadata['grid']));positions=a['positions']
    if positions.dtype!=np.int64 or not np.array_equal(np.sort(positions),np.arange(n)):raise ValueError('invalid prediction positions')
    for b in NEW:
        if a[b].shape!=(4,n) or a[b].dtype!=np.float32 or not np.isfinite(a[b]).all() or (a[b]<0).any():
            raise ValueError('invalid new prediction distance')
    return a,metadata,data


def predict(bundle,out,workers=2,limit=None):
    started=time.perf_counter();out=ip.initialize(out,bundle);path=out/'prediction_checkpoint.json'
    ledger=ip.read_json(path) if path.exists() else dict(identity=bundle.identity,records=[])
    if ledger['identity']!=bundle.identity:raise ValueError('checkpoint identity differs')
    records=ledger['records'];by_key={}
    for r in records:
        _,meta,_=load_record(out,r,bundle.identity);key=meta['domain'],meta['sample_id']
        if key in by_key:raise ValueError('duplicate prediction record')
        by_key[key]=r
    samples={(s['domain'],s['sample_id']):s for s in bundle.samples}
    if not set(by_key)<=set(samples):raise ValueError('unexpected prediction scope')
    # Uncommitted files are not silently trusted/replaced after an interrupted write.
    files={p.relative_to(out).as_posix() for p in (out/'predictions').rglob('*.npz')}
    expected={ip.read_json(out/r['record'])['file'] for r in records}
    if files!=expected:raise ValueError('uncommitted/missing prediction file; preserve it and use a fresh output if needed')
    pending=[s for s in bundle.samples if (s['domain'],s['sample_id']) not in by_key]
    if limit is not None:pending=pending[:limit]
    reused=len(records)
    def work(s):return one(bundle,s)
    for arrays,meta,scores,reduction,checks in bounded_images(pending,work,workers):
        relative=f"predictions/{meta['domain']}/{meta['sample_id']}.npz";p=out/relative;p.parent.mkdir(parents=True,exist_ok=True)
        temporary=p.with_suffix('.pending')
        with temporary.open('xb') as f:np.savez_compressed(f,metadata=np.asarray(json.dumps(meta,sort_keys=True)),**arrays)
        if p.exists():raise FileExistsError('prediction exists without trusted commit')
        temporary.replace(p)
        record_path=f"records/{meta['domain']}/{meta['sample_id']}.json";rp=out/record_path;rp.parent.mkdir(parents=True,exist_ok=True)
        if rp.exists():raise FileExistsError('record already exists')
        ip.write_json(rp,dict(identity=bundle.identity,file=relative,sha256=ip.sha(p),metadata=meta,scores=scores,reduction=reduction,checks=checks))
        records.append(dict(record=record_path,record_sha256=ip.sha(rp)))
        ip.write_json(path,dict(identity=bundle.identity,records=records))
        if len(records)%25==0 or len(records)==len(bundle.samples):print(f'[GAP predict] {len(records)}/926 {meta["domain"]}/{meta["sample_id"]}',flush=True)
    all_rows=[ip.read_json(out/r['record']) for r in records]
    for f,key in [('prediction_scores.csv','scores'),('reduction_summary.csv','reduction'),('prediction_numerical_verification.csv','checks')]:
        ip.write_csv(out/f,[x for r in all_rows for x in r[key]])
    complete=len(records)==len(bundle.samples)
    if complete:
        manifest=dict(identity=bundle.identity,images=926,records=records,GT_used=False,labels_used=False,
            files={f:ip.sha(out/f) for f in ('prediction_scores.csv','reduction_summary.csv','prediction_numerical_verification.csv','protocol.json','input_manifest.json')})
        if (out/'prediction_manifest.json').exists():
            if ip.read_json(out/'prediction_manifest.json')!=manifest:raise ValueError('frozen prediction manifest changed')
        else:ip.write_json(out/'prediction_manifest.json',manifest)
    state=dict(status='predictions_frozen' if complete else 'partial_predictions',identity=bundle.identity,images=len(records),
        expected_images=926,GT_used=False,labels_used=False,torch_imported='torch' in __import__('sys').modules,
        reused_images=reused,new_images=len(records)-reused,seconds=time.perf_counter()-started)
    if complete:state['prediction_manifest_sha256']=ip.sha(out/'prediction_manifest.json')
    ip.write_json(out/'prediction_status.json',state)
    ip.write_json(out/'status.json',dict(state,stage='predict',all_metrics_complete=False))
    return state


def verified(bundle,out):
    s=ip.read_json(out/'prediction_status.json');m=ip.read_json(out/'prediction_manifest.json')
    if (s['status']!='predictions_frozen' or s['images']!=926 or m['images']!=926 or m['identity']!=bundle.identity
        or s['prediction_manifest_sha256']!=ip.sha(out/'prediction_manifest.json') or m['GT_used'] or m['labels_used']):
        raise ValueError('complete frozen 926-image prediction required before GT evaluation')
    for f,h in m['files'].items():
        if ip.sha(out/ip.bg.safe(f))!=h:raise ValueError('prediction sidecar changed: '+f)
    records={}
    for r in m['records']:
        _,meta,_=load_record(out,r,bundle.identity);key=meta['domain'],meta['sample_id']
        if key in records:raise ValueError('duplicate frozen sample')
        records[key]=r
    if set(records)!={(s['domain'],s['sample_id']) for s in bundle.samples}:raise ValueError('prediction scope differs')
    return m,records
