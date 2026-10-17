"""Freeze complete per-shift predictions before any GT decoding."""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
from . import inputs as ip,calibration,runtime
from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.method import predict as kernel,BRANCHES
from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.inputs import save_npz,load_npz
from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.verification import audit
from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import statistics,ch1_score
from DINOv3.MADEqual.aebad_s_neighborhood_segment_p0.prediction import bounded_images

SOURCE='SOURCE_CH1'
FIELDS=dict(zip(BRANCHES,('union_distance','segment_distance','pair_distance')))


def samples(ctx,group):
    return ctx['cat']['source_evaluation'] if group=='source' else ctx['cat']['shifts'][group]['samples']


def score(ctx,arrays,meta,state,source=False):
    values=[(SOURCE,'source_distance'),*([] if source else list(FIELDS.items()))];rows=[]
    for branch,key in values:
        current,peak,local=statistics(arrays[key],arrays['positions'],tuple(meta['grid']),tuple(meta['output_shape']),ctx['medians'],ctx['scales'])
        rows.append(dict(category=ctx['category'],shift=meta['shift'],sample_id=meta['sample_id'],branch=branch,
            image_score=ch1_score(peak,local,state),CURRENT_score=current,image_calibration_sha256=state['state_sha256'],
            role='Source_evaluation_reference' if source else 'Source_on_Target_auxiliary' if branch==SOURCE else 'Target_main'))
    return rows


def run(ctx,group,workers=4):
    started=time.perf_counter();state=calibration.load_state(ctx);raw_state=ip.ledger(ctx)
    selected=samples(ctx,group);kind='source' if group=='source' else 'target'
    records={r['sample_id']:r for r in raw_state['records'] if r['kind']==kind and r['shift']==group}
    if set(records)!={s['sample_id'] for s in selected}:raise ValueError('complete selected retrieval caches required before prediction freeze')
    out=ctx['root']/'predictions'/group
    if (out/'complete.json').is_file():verify(ctx,group);return out
    if out.exists():
        progress=ip.read_json(out/'progress.json')
        if progress['identity']!=ctx['identity'] or progress['image_calibration_sha256']!=state['state_sha256']:raise ValueError('partial prediction freeze differs')
        prediction=progress['records'];checks=progress['checks'];scored=progress['scores'];statistics_rows=progress['statistics']
        if len({r['sample_id'] for r in prediction})!=len(prediction) or not {r['sample_id'] for r in prediction}.issubset(records):raise ValueError('partial prediction scope differs')
        if {p.relative_to(out).as_posix() for p in (out/'arrays').rglob('*') if p.is_file()}!={r['file'] for r in prediction}:raise ValueError('orphan/missing partial prediction; preserve it')
        for r in prediction:
            if ip.sha(out/ip.safe(r['file']))!=r['sha256'] or r['capture_sha256']!=records[r['sample_id']]['sha256']:raise ValueError('partial prediction cache changed')
    else:
        out.mkdir(parents=True);checks=[];scored=[];statistics_rows=[];prediction=[]
        ip.write_json(out/'progress.json',dict(identity=ctx['identity'],image_calibration_sha256=state['state_sha256'],records=[],checks=[],scores=[],statistics=[]))
    completed={r['sample_id'] for r in prediction}
    def one(sample):
        r=records[sample['sample_id']];raw,meta=ip.load_cache(ctx,r,sample);local=[]
        if group=='source':
            runtime.validate_source(raw,meta,sample);arrays=raw;stats=[]
        else:
            runtime.verify_target(ctx,sample,raw,meta,local);arrays,stats=kernel(raw,tuple(meta['grid']))
            arrays.update({k:raw[k] for k in ('endpoint_support','endpoint_index','alpha','endpoint_length')})
        meta=dict(meta,identity=ctx['identity'],category=ctx['category'],shift=group,sample_id=sample['sample_id'],
            image_sha256=sample['image_sha256'],source_capture_sha256=r['sha256'],image_calibration_sha256=state['state_sha256'])
        scores=score(ctx,arrays,meta,state,source=group=='source')
        return sample,arrays,meta,r,local,scores,[dict(category=ctx['category'],domain=group,shift=group,sample_id=sample['sample_id'],**s) for s in stats]
    try:
        for sample,arrays,meta,raw_record,local,sc,st in bounded_images([s for s in selected if s['sample_id'] not in completed],one,workers):
            name='arrays/'+ip.key(sample)+'.npz';save_npz(out/name,arrays,meta)
            r=dict(sample_id=sample['sample_id'],file=name,sha256=ip.sha(out/name),capture_sha256=raw_record['sha256'])
            prediction.append(r);checks.extend(local);scored.extend(sc);statistics_rows.extend(st)
            ip.write_json(out/'progress.json',dict(identity=ctx['identity'],image_calibration_sha256=state['state_sha256'],records=prediction,checks=checks,scores=scored,statistics=statistics_rows))
            print(f'[Robust predict] {ctx["category"]} {group} {len(prediction)}/{len(selected)}',flush=True)
        if any(not r['verified'] and r.get('acceptance_role')!='historical_distance_audit_only' for r in checks):raise ValueError('prediction numerical check failed')
        prediction.sort(key=lambda r:[s['sample_id'] for s in selected].index(r['sample_id']))
        for n,rows in (('scores.csv',scored),('gate_statistics.csv',statistics_rows),('numerical_verification.csv',checks)):ip.write_csv(out/n,rows)
        ip.write_json(out/'historical_score_audit.json',audit(checks))
        manifest=dict(tool=ip.TOOL,identity=ctx['identity'],category=ctx['category'],shift=group,images=len(selected),records=prediction,
            image_calibration_sha256=state['state_sha256'],GT_used=False,labels_used_by_method=False,
            files={n:ip.sha(out/n) for n in ('scores.csv','gate_statistics.csv','numerical_verification.csv','historical_score_audit.json')})
        ip.write_json(out/'manifest.json',manifest)
        ip.write_json(out/'complete.json',dict(status='predictions_frozen',identity=ctx['identity'],images=len(selected),
            manifest_sha256=ip.sha(out/'manifest.json'),seconds=time.perf_counter()-started))
    except Exception as exc:
        ip.write_json(out/'failure.json',dict(error=f'{type(exc).__name__}: {exc}',checks=checks));raise
    return out


def verify(ctx,group):
    root=ctx['root']/'predictions'/group;status=ip.read_json(root/'complete.json');m=ip.read_json(root/'manifest.json')
    state=calibration.load_state(ctx);selected=samples(ctx,group)
    if (status['status']!='predictions_frozen' or status['identity']!=ctx['identity'] or m['identity']!=ctx['identity']
            or status['manifest_sha256']!=ip.sha(root/'manifest.json') or m['image_calibration_sha256']!=state['state_sha256']
            or m['category']!=ctx['category'] or m['shift']!=group or m['GT_used'] is not False
            or m['labels_used_by_method'] is not False or m['images']!=len(selected) or status['images']!=len(selected)
            or [r['sample_id'] for r in m['records']]!=[s['sample_id'] for s in selected]):raise ValueError('prediction incomplete/changed/wrong category or calibration')
    for name,h in m['files'].items():
        if ip.sha(root/ip.safe(name))!=h:raise ValueError('prediction sidecar modified')
    actual={p.relative_to(root).as_posix() for p in (root/'arrays').rglob('*') if p.is_file()}
    if actual!={r['file'] for r in m['records']}:raise ValueError('prediction arrays missing/orphaned')
    for r in m['records']:
        if ip.sha(root/ip.safe(r['file']))!=r['sha256']:raise ValueError('prediction array modified')
    return root,m,selected


def load(ctx,root,record,sample,state):
    arrays,meta=load_npz(Path(root)/ip.safe(record['file']),record['sha256'])
    if (meta['identity']!=ctx['identity'] or meta['sample_id']!=sample['sample_id'] or meta['image_sha256']!=sample['image_sha256']
            or meta['category']!=ctx['category'] or meta['shift']!=sample['shift'] or meta['image_calibration_sha256']!=state['state_sha256']
            or meta['source_capture_sha256']!=record['capture_sha256']):raise ValueError('frozen prediction header changed')
    runtime.validate_source(arrays,meta,sample)
    if sample['shift']!='source':
        from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.method import pair_gate
        n=len(arrays['positions'])
        for key in FIELDS.values():
            a=arrays[key]
            if a.dtype!=np.float32 or a.shape!=(4,n) or not np.isfinite(a).all() or np.any(a<0):raise ValueError('invalid frozen branch distance')
        for off in range(4):
            gate,votes,count=pair_gate(arrays['endpoint_support'][off],arrays['positions'],tuple(meta['grid']))
            for k,v in (('pair_gate',gate),('pair_votes',votes),('neighbor_count',count)):
                if not np.array_equal(arrays[k][off],v):raise ValueError('frozen PAIR rule changed')
            expected=np.where(gate.ravel()[arrays['positions']],arrays['segment_distance'][off],arrays['union_distance'][off])
            if not np.array_equal(expected,arrays['pair_distance'][off]):raise ValueError('frozen PAIR fallback changed')
        for a,b in (('segment_distance','pair_distance'),('pair_distance','union_distance'),('union_distance','source_distance')):
            if np.any(arrays[a]>arrays[b]+1e-7+1e-6*np.abs(arrays[b])):raise ValueError('frozen distance containment failed')
    return arrays,meta
