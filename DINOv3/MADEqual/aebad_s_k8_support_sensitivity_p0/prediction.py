"""CPU prediction freeze. Three groups share the same Source and query geometry."""
from __future__ import annotations

import time
import numpy as np
from . import capture, inputs as ip, scheduler
from .selection import SEEDS
from .storage import load_npz, npz_bytes, json_bytes
from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.method import BRANCHES, U, S, P
from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import statistics, ch1_score
from DINOv3.MADEqual.aebad_s_neighborhood_segment_p0.prediction import bounded_images

FIELDS = {U:'union_distance', S:'segment_distance', P:'pair_distance'}


def score(ctx, arrays, meta):
    rows = []
    for branch, field in FIELDS.items():
        current, peak, local = statistics(arrays[field], arrays['positions'], tuple(meta['grid']),
                                         tuple(meta['output_shape']), ctx['medians'], ctx['scales'])
        rows.append(dict(branch=branch, image_score=ch1_score(peak, local, ctx['state']),
                         CURRENT_audit=current, image_state_sha256=ctx['assets']['image_state_sha256']))
    return rows


def run(ctx, domains, workers=4):
    ledger = ctx['ledger']; ledger.verify(); started = time.perf_counter()
    samples = [s for s in ctx['plan']['samples'] if s['domain'] in domains]
    if any(not ledger.get('capture/'+capture.sample_key(s)) for s in samples):
        raise ValueError('selected domain capture incomplete; no partial domain prediction')
    pending = [s for s in samples if not ledger.get('prediction/'+capture.sample_key(s))]

    def one(sample):
        tick = time.perf_counter(); raw, meta, record = capture.load_raw(ctx, sample)
        checks = capture.audit_scores(ctx, sample, raw, meta)
        results = scheduler.predictions(raw, meta['mapping'], meta['grid'])
        arrays = {'positions':raw['positions'].copy()}; scores = []; gate_stats = []
        for seed, (a, stats) in results.items():
            arrays.update({f'seed{seed}_{k}':v for k,v in a.items() if k!='positions'})
            arrays[f'seed{seed}_endpoint_support'] = raw[f'seed{seed}_endpoint_support']
            arrays[f'seed{seed}_endpoint_index'] = raw[f'seed{seed}_endpoint_index']
            scores.extend(dict(support_seed=seed, **r) for r in score(ctx, a, meta))
            gate_stats.extend(dict(support_seed=seed, **r) for r in stats)
        metadata = dict(meta, stage='predictions_frozen', capture_sha256=record['files'][record['file']],
                        prediction_seconds=time.perf_counter()-tick, GT_used=False)
        name = 'predictions/'+capture.sample_key(sample)+'.npz'
        payload = dict(file=name, scores=scores, checks=checks, gate_statistics=gate_stats,
                       timing=dict(prediction=metadata['prediction_seconds']), status='predictions_frozen')
        return sample, payload, {name:npz_bytes(arrays, metadata)}

    for sample, payload, files in bounded_images(pending, one, workers):
        ledger.commit('prediction/'+capture.sample_key(sample), payload, files)
        print('[K8 sensitivity predict] '+capture.sample_key(sample), flush=True)
    for domain in domains:
        selected = [s for s in samples if s['domain']==domain]
        key = 'prediction_freeze/'+domain
        records = [ledger.get('prediction/'+capture.sample_key(s)) for s in selected]
        if not selected or any(not r or r['status']!='predictions_frozen' for r in records):
            raise ValueError('cannot freeze incomplete predictions')
        if not ledger.get(key):
            snapshot = dict(identity=ctx['identity'], domain=domain, images=len(selected), support_seeds=list(SEEDS),
                            records={s['sample_id']:r['files'] for s,r in zip(selected,records)}, GT_used=False)
            ledger.commit(key, dict(status='predictions_frozen', images=len(selected)),
                          {f'freezes/{domain}.json':json_bytes(snapshot)})
    ledger.verify()
    return dict(new_images=len(pending), reused_images=len(samples)-len(pending), seconds=time.perf_counter()-started,
                model_inference=False)


def require_frozen(ctx, domain):
    row = ctx['ledger'].get('prediction_freeze/'+domain)
    selected = [s for s in ctx['plan']['samples'] if s['domain']==domain]
    if not row or row['status']!='predictions_frozen' or row['images']!=len(selected):
        raise ValueError('all common evaluation predictions must be frozen before GT/evaluation')
    for s in selected:
        if not ctx['ledger'].get('prediction/'+capture.sample_key(s)):
            raise ValueError('frozen prediction set incomplete')
    return selected


def load(ctx, sample, seed):
    ledger = ctx['ledger']; record = ledger.get('prediction/'+capture.sample_key(sample))
    if record is None: raise FileNotFoundError('missing prediction')
    arrays, meta = load_npz(ledger.root/record['file'], record['files'][record['file']])
    if (meta.get('identity') != ctx['identity'] or meta.get('sample_id') != sample['sample_id']
            or meta.get('relative_path') != sample['relative_path'] or meta.get('image_sha256') != sample['image_sha256']
            or meta.get('stage') != 'predictions_frozen' or meta.get('GT_used') is not False
            or meta.get('output_shape') != [sample['image_height']//4,sample['image_width']//4]):
        raise ValueError('prediction identity/geometry mismatch')
    prefix = f'seed{seed}_'
    a = {k[len(prefix):]:v for k,v in arrays.items() if k.startswith(prefix)}
    a['positions'] = arrays['positions']
    from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.method import pair_gate, scatter
    for key in ('source_distance', 'union_distance', 'segment_distance', 'pair_distance'):
        v = a[key]
        if v.dtype!=np.float32 or v.shape!=(4,len(a['positions'])) or not np.isfinite(v).all() or (v<0).any():
            raise ValueError('invalid predicted distances')
        scatter(v, a['positions'], tuple(meta['grid']))
    for off in range(4):
        g,v,n = pair_gate(a['endpoint_support'][off], a['positions'], tuple(meta['grid']))
        if any(not np.array_equal(a[k][off], actual) for k,actual in (
                ('pair_gate',g),('pair_votes',v),('neighbor_count',n))):
            raise ValueError('saved PAIR differs from original neighborhood kernel')
        if not np.array_equal(a['pair_distance'][off],np.where(g.ravel()[a['positions']],a['segment_distance'][off],a['union_distance'][off])):
            raise ValueError('saved PAIR distance differs from its gate')
    for lower,upper in (('segment_distance','pair_distance'),('pair_distance','union_distance'),('union_distance','source_distance')):
        if np.any(a[lower]>a[upper]+1e-7+1e-6*np.abs(a[upper])):
            raise ValueError('distance containment failed')
    actual = score(ctx,a,meta)
    frozen = [r for r in record['scores'] if r['support_seed']==seed]
    if len(frozen)!=3: raise ValueError('prediction score set incomplete')
    for r in actual:
        old = next(s for s in frozen if s['branch']==r['branch'])
        if abs(r['image_score']-old['image_score'])>1e-5 or abs(r['CURRENT_audit']-old['CURRENT_audit'])>1e-5:
            raise ValueError('scoring backend changed after prediction freeze')
    return a, meta, actual
