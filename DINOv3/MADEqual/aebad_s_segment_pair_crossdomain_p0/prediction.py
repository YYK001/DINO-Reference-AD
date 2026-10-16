"""CPU-only prediction freeze, independent of Source/model/support files."""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
from . import inputs as ip
from .method import predict as kernel,BRANCHES,LAYERS,U,S,P,pair_gate
from .capture import load_cached,verify_scores
from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import statistics,ch1_score
from DINOv3.MADEqual.aebad_s_neighborhood_segment_p0.prediction import bounded_images,protect_dependencies


def samples_for(ctx,domains):
    if not domains or len(set(domains))!=len(domains) or any(d not in ip.DOMAINS for d in domains):raise ValueError('invalid frozen domain scope')
    return [s for d in ip.DOMAINS if d in domains for s in ctx['catalog']['domains'][d]['samples']]


def scores_for(ctx,arrays,meta,sample,checks):
    rows=[]
    # Scores are kept separate from labels until evaluation. sample supplies
    # only immutable formal identities/current references, never gate inputs.
    verify_scores(ctx,sample,arrays['source_distance'],arrays['union_distance'],arrays['positions'],meta,checks)
    for branch,key in ((U,'union_distance'),(S,'segment_distance'),(P,'pair_distance')):
        current,peak,local=statistics(arrays[key],arrays['positions'],tuple(meta['grid']),tuple(meta['output_shape']),ctx['medians'],ctx['scales'])
        score=ch1_score(peak,local,ctx['state'])
        if not np.isfinite(score):raise ValueError('nonfinite frozen CH1 score')
        rows.append(dict(domain=sample['domain'],sample_id=sample['sample_id'],relative_path=sample['relative_path'],branch=branch,
                         image_score=score,CURRENT_check_score=current,image_calibration_state_sha256=ctx['state']['state_sha256']))
    return rows


def run(ctx,output,domains,workers=4):
    started=time.perf_counter();state=ip.cache_state(ctx);samples=samples_for(ctx,domains)
    records={(r['domain'],r['sample_id']):r for r in state['records']}
    if any((s['domain'],s['sample_id']) not in records for s in samples):raise ValueError('selected domains need complete verified retrieval/geometry first')
    out=ip.new_output(ctx,output,'predict');checks=[];scores=[];stats=[];prediction=[]
    def one(sample):
        r=records[(sample['domain'],sample['sample_id'])]
        raw,meta=load_cached(ctx,ctx['paths']['cache_root'],r,sample)
        arrays,statistics=kernel(raw,tuple(meta['grid']))
        arrays.update({k:raw[k].copy() for k in ('endpoint_support','endpoint_index','alpha','endpoint_length')})
        m=dict(tool=ip.TOOL,format='frozen_segment_pair_prediction_v1',identity=ctx['identity'],domain=sample['domain'],
            sample_id=sample['sample_id'],relative_path=sample['relative_path'],image_sha256=sample['image_sha256'],
            grid=meta['grid'],output_shape=meta['output_shape'],source_capture_sha256=r['sha256'],supports=meta['supports'],
            geometry_source='original_geometry_layer_real_support_endpoint_features',distance_unit=ip.PROTOCOL['distance_unit'])
        local=[]
        try:image_scores=scores_for(ctx,arrays,m,sample,local)
        except Exception:
            failure=out/'failures';failure.mkdir(exist_ok=True)
            ip.write_json(failure/f'{sample["domain"]}_{sample["sample_id"]}.json',dict(checks=local,source_capture_sha256=r['sha256']))
            raise
        path=f'predictions/{sample["domain"]}/{sample["sample_id"]}.npz';ip.save_npz(out/path,arrays,m)
        record=dict(domain=sample['domain'],sample_id=sample['sample_id'],file=path,sha256=ip.sha(out/path))
        return record,local,image_scores,[dict(domain=sample['domain'],sample_id=sample['sample_id'],**row) for row in statistics]
    for record,local,sc,st in bounded_images(samples,one,workers):
        prediction.append(record);checks.extend(local);scores.extend(sc);stats.extend(st)
        print(f'[crossdomain predict] {record["domain"]} {record["sample_id"]}',flush=True)
    ip.write_csv(out/'numerical_verification.csv',checks);ip.write_csv(out/'prediction_scores.csv',scores);ip.write_csv(out/'gate_statistics.csv',stats)
    if not checks or not all(r['verified'] for r in checks):raise ValueError('numerical reproduction did not pass')
    if ip.context(ctx['paths'])['identity']!=ctx['identity']:raise ValueError('joint freeze changed during prediction')
    ip.cache_state(ctx) # verify original per-image history, not newly approved hashes
    manifest=dict(tool=ip.TOOL,identity=ctx['identity'],domains=list(domains),records=prediction,images=len(samples),
                  files={n:ip.sha(out/n) for n in ('protocol.json','input_manifest.json','prediction_scores.csv','gate_statistics.csv','numerical_verification.csv')},
                  GT_used=False,labels_used_by_method=False,geometry_verified=True,
                  historical_distance_verification=dict(available=ctx.get('distance_references') is not None,
                    verified_checks=sum(r['check_kind'].startswith('original_') for r in checks),
                    missing_images=sum(not any(r['check_kind']=='original_Source_distance' and r['domain']==s['domain'] and r['sample_id']==s['sample_id'] for r in checks) for s in samples)))
    ip.write_json(out/'prediction_manifest.json',manifest)
    from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.evaluation import peak_rss
    ip.write_json(out/'timing.json',dict(seconds=time.perf_counter()-started,cpu_workers=workers,model_inference=False,feature_extraction=False,
                                       process_peak_rss_bytes=peak_rss(),new_output_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),environment=ip.common.environment()))
    ip.write_json(out/'status.json',dict(tool=ip.TOOL,status='predictions_frozen',identity=ctx['identity'],domains=list(domains),images=len(samples),
                                       prediction_manifest_sha256=ip.sha(out/'prediction_manifest.json'),GT_used=False,Source_Union_CURRENT_verified=True))
    return out


def verified(ctx,directory):
    root=Path(directory).resolve();status=ip.read_json(root/'status.json');manifest=ip.read_json(root/'prediction_manifest.json')
    if (status['status']!='predictions_frozen' or status['identity']!=ctx['identity']
            or status['prediction_manifest_sha256']!=ip.sha(root/'prediction_manifest.json') or manifest['identity']!=ctx['identity']
            or manifest['GT_used'] is not False or manifest['labels_used_by_method'] is not False):raise ValueError('prediction is incomplete, modified or incompatible')
    samples=samples_for(ctx,manifest['domains'])
    if ([(r['domain'],r['sample_id']) for r in manifest['records']]!=[(s['domain'],s['sample_id']) for s in samples]
            or manifest['images']!=len(samples) or status['images']!=len(samples)):raise ValueError('prediction set incomplete or reordered')
    if ip.read_json(root/'input_manifest.json')!=dict(identity=ctx['identity'],frozen=ctx['frozen']):raise ValueError('joint freeze mismatch')
    for name,h in manifest['files'].items():
        if ip.sha(root/ip.safe(name))!=h:raise ValueError('prediction sidecar changed')
    expected={r['file'] for r in manifest['records']}
    if {p.relative_to(root).as_posix() for p in (root/'predictions').rglob('*.npz')}!=expected:raise ValueError('unexpected/missing prediction files')
    for r in manifest['records']:
        if ip.sha(root/ip.safe(r['file']))!=r['sha256']:raise ValueError('prediction modified after freeze')
    return root,manifest,samples


def load(ctx,root,record,sample):
    arrays,meta=ip.load_npz(Path(root)/ip.safe(record['file']),record['sha256'])
    if (meta['identity']!=ctx['identity'] or meta['format']!='frozen_segment_pair_prediction_v1'
            or meta['domain']!=sample['domain'] or meta['sample_id']!=sample['sample_id']
            or meta['relative_path']!=sample['relative_path'] or meta['image_sha256']!=sample['image_sha256']
            or meta['distance_unit']!=ip.PROTOCOL['distance_unit']):raise ValueError('prediction metadata mismatch')
    grid=tuple(meta['grid']);positions=arrays['positions'];n=int(np.prod(grid))
    from .method import scatter
    for key in ('source_distance','union_distance','segment_distance','pair_distance'):
        a=arrays[key]
        if a.dtype!=np.float32 or a.shape!=(4,n) or not np.isfinite(a).all() or (a<0).any():raise ValueError('invalid prediction distance')
        scatter(a,positions,grid)
    for off in range(4):
        gate,votes,count=pair_gate(arrays['endpoint_support'][off],positions,grid)
        if any(not np.array_equal(arrays[key][off],value) for key,value in (('pair_gate',gate),('pair_votes',votes),('neighbor_count',count))):
            raise ValueError('frozen pair gate differs from physical-neighborhood rule')
        expected=np.where(gate.ravel()[positions],arrays['segment_distance'][off],arrays['union_distance'][off])
        if not np.array_equal(expected,arrays['pair_distance'][off]):raise ValueError('pair fallback differs from complete Union')
    for lower,upper in ((arrays['segment_distance'],arrays['pair_distance']),(arrays['pair_distance'],arrays['union_distance']),
                        (arrays['union_distance'],arrays['source_distance'])):
        if np.any(lower>upper+1e-7+1e-6*np.abs(upper)):raise ValueError('frozen prediction containment failed')
    if meta['output_shape']!=[sample['image_height']//4,sample['image_width']//4]:raise ValueError('output shape changed')
    return arrays,meta
