"""Layer-wise cached query processing, then per-domain immutable prediction freeze."""
import time
from pathlib import Path
import numpy as np
from . import inputs as ip,storage as st,supports
from .method import LAYERS,BASES,NEW,BRANCHES,reduction
from .execution import bounded_images
from DINOv3.MADEqual.aebad_s_gap_soft_segment_p0.prediction import score
from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic.numerics import check


def selected(bundle,domain):return [s for s in bundle.samples if s['domain']==domain]
def layer_path(out,d,l,sid):return Path(out)/'distance_layers'/d/f'L{l}'/(sid+'.npz')
def image_path(out,d,sid):return Path(out)/'predictions'/d/(sid+'.npz')


def predict_layer(bundle,out,locations,domain,layer,limit=None,*,device='cpu',witness_batch=256):
    tick=time.perf_counter();out=st.initialize(out,bundle);samples=selected(bundle,domain)
    idx,_,ni=supports.load_tables(bundle,out,domain,layer)
    banks=ip.load_layer(bundle.expected,locations,domain,layer);off=LAYERS.index(layer);new=0;reused=0
    from .batch_witness import Engine
    with Engine(banks,idx,device=device,batch_size=witness_batch,
                path=out/'connections_batched'/domain/f'L{layer}',identity=bundle.identity,neighbor_identity=ni) as relations:
        for sample in samples:
            p=layer_path(out,domain,layer,sample['sample_id']);record=st.restore(p,bundle.identity)
            if record:
                if (record['metadata']['neighbors'],record['metadata']['domain'],record['metadata']['layer'],record['metadata']['sample_id'])!=(ni,domain,layer,sample['sample_id']):raise ValueError('cached layer scope changed')
                reused+=1;continue
            if limit is not None and new>=limit:continue
            raw,meta,origin=bundle.load(sample)
            g,counts=supports.candidate_gates(raw['target_distance'][:,off],raw['target_index'][:,off],relations)
            distances,red=reduction(raw['union_distance'][off],raw['segment_distance'][off],g)
            metadata=dict(domain=domain,layer=layer,sample_id=sample['sample_id'],relative_path=sample['relative_path'],
                neighbors=ni,origin=origin,grid=meta['grid'],output_shape=meta['output_shape'],counts=counts,reduction=red,
                GT_used=False,labels_used=False)
            st.seal(p,bundle.identity,dict(positions=raw['positions'],gate=g,**distances),metadata);new+=1
            if new%10==0 or new+reused==len(samples):
                print(f'[witness cached queries] {domain}/L{layer} device={device} new={new} reused={reused}'
                    f' unique_edges={relations.new} cache_hits={relations.reused}'
                    f' seconds={time.perf_counter()-tick:.1f}',flush=True)
        state=dict(identity=bundle.identity,domain=domain,layer=layer,new_images=new,reused_images=reused,
            status='layer_complete' if new+reused==len(samples) else 'partial_layer',seconds=time.perf_counter()-tick,
            connection_seconds=relations.seconds,new_unique_connections=relations.new,reused_connections=relations.reused,
            scalar_boundary_fallback_edges=relations.scalar_fallback_edges,witness_device=device,
            connection_timing=relations.timing(),
            neighbors=ni,GT_used=False,labels_used=False)
    st.write_json(out/'distance_layers'/domain/f'L{layer}'/'status.json',state);return state


def load_record(bundle,out,domain,sid):
    p=image_path(out,domain,sid);a,r=st.arrays(p,bundle.identity);m=r['metadata']
    if (m['domain'],m['sample_id'])!=(domain,sid) or m['GT_used'] or m['labels_used']:raise ValueError('frozen prediction scope/GT contract changed')
    n=int(np.prod(m['grid']))
    if a['positions'].dtype!=np.int64 or a['positions'].shape!=(n,) or not np.array_equal(np.sort(a['positions']),np.arange(n)):raise ValueError('invalid full physical grid')
    for b in NEW:
        if a[b].shape!=(4,n) or a[b].dtype!=np.float32 or not np.isfinite(a[b]).all() or (a[b]<0).any():raise ValueError('invalid new prediction distance')
    return a,m,r


def freeze_image(bundle,out,domain,sample):
    sid=sample['sample_id'];p=image_path(out,domain,sid)
    paths=[layer_path(out,domain,l,sid) for l in LAYERS]
    # Each layer is loaded/hashed once per freeze task.
    layer_data=[st.arrays(path,bundle.identity) if path.exists() else (None,st.restore(path,bundle.identity)) for path in paths]
    if any(r is None for _,r in layer_data):return None
    layer_records=[r for _,r in layer_data]
    raw,meta,origin=bundle.load(sample);meta=dict(meta,origin=origin)
    records_sha={str(l):st.sha(path.with_suffix('.json')) for l,path in zip(LAYERS,paths)}
    previous=st.restore(p,bundle.identity)
    if previous:
        a,m,r=load_record(bundle,out,domain,sid)
        if m['origin']!=origin or m['layer_records']!=records_sha:raise ValueError('frozen parent/layer dependencies changed')
    else:
        arrays={b:[] for b in NEW}
        for x,rec in layer_data:
            if not np.array_equal(x['positions'],raw['positions']) or rec['metadata']['origin']!=origin:raise ValueError('layer positions or original cache changed')
            for b in NEW:arrays[b].append(x[b])
        arrays={b:np.stack(v) for b,v in arrays.items()};arrays['positions']=raw['positions']
        values={BASES[0]:raw['union_distance'],BASES[1]:raw['segment_distance'],**{b:arrays[b] for b in NEW}}
        sc=[];ck=[]
        for b in BRANCHES:
            scores=score(bundle,values[b],raw['positions'],meta)
            if b in BASES:
                ref=bundle.references[domain,sid,b]
                check(ck,sid,'same_parent_CH1',scores['CH1'],float(ref['image_score']),domain=domain,branch=b)
                old=ref.get('CURRENT_check_score',ref.get('current_check_score'))
                if old not in ('',None):check(ck,sid,'same_parent_CURRENT',scores['CURRENT'],float(old),domain=domain,branch=b)
            sc.append(dict(domain=domain,sample_id=sid,relative_path=sample['relative_path'],branch=b,**scores))
        if any(not v['verified'] for v in ck):raise ValueError('same-parent score reproduction failed')
        m=dict(domain=domain,sample_id=sid,relative_path=sample['relative_path'],grid=meta['grid'],output_shape=meta['output_shape'],
            origin=origin,layer_records=records_sha,scores=sc,checks=ck,GT_used=False,labels_used=False)
        r=st.seal(p,bundle.identity,arrays,m)
    return sample,r,layer_records


def freeze_domain(bundle,out,domain,workers=1):
    out=Path(out);samples=selected(bundle,domain);records=[];all_scores=[];checks=[];reductions=[];connections=[]
    existing=out/'prediction_manifests'/(domain+'.json')
    if existing.exists() and st.read_json(existing)['status']=='predictions_frozen':
        # Audit tables are also frozen; do not silently regenerate a modified
        # historical table and thereby erase evidence of the modification.
        verified(bundle,out,domain)
    for item in bounded_images(samples,lambda s:freeze_image(bundle,out,domain,s),workers):
        if item is None:continue
        sample,r,layer_records=item;sid=sample['sample_id'];p=image_path(out,domain,sid)
        records.append(dict(sample_id=sid,file=p.relative_to(out).as_posix(),sha256=r['sha256'],record_sha256=st.sha(p.with_suffix('.json'))))
        all_scores.extend(r['metadata']['scores']);checks.extend(r['metadata']['checks'])
        for l,rec in zip(LAYERS,layer_records):
            reductions.append(dict(domain=domain,sample_id=sid,layer=l,**rec['metadata']['reduction']))
            connections.append(dict(domain=domain,sample_id=sid,layer=l,**rec['metadata']['counts']))
        if len(records)%25==0 or len(records)==len(samples):print(f'[witness freeze] {domain} {len(records)}/{len(samples)} workers={workers}',flush=True)
    complete=len(records)==len(samples)
    tables=out/'prediction_tables'/domain
    for f,rows in [('scores.csv',all_scores),('numerical_verification.csv',checks),('reduction_summary.csv',reductions),('connection_summary.csv',connections)]:
        st.write_csv(tables/f,rows)
    state=dict(identity=bundle.identity,domain=domain,status='predictions_frozen' if complete else 'partial_predictions',
        images=len(records),expected_images=len(samples),records=records,GT_used=False,labels_used=False,
        files={p.relative_to(out).as_posix():st.sha(p) for p in tables.glob('*.csv')})
    path=out/'prediction_manifests'/(domain+'.json')
    if path.exists() and st.read_json(path)['status']=='predictions_frozen' and st.read_json(path)!=state:raise ValueError('frozen domain manifest changed')
    st.write_json(path,state)
    return state


def verified(bundle,out,domain):
    out=Path(out);path=out/'prediction_manifests'/(domain+'.json');m=st.read_json(path)
    samples=selected(bundle,domain)
    if m['identity']!=bundle.identity or m['status']!='predictions_frozen' or m['images']!=len(samples) or m['GT_used'] or m['labels_used']:raise ValueError('complete frozen domain required before GT evaluation')
    if len(m['records'])!=len(samples) or {v['sample_id'] for v in m['records']}!={s['sample_id'] for s in samples}:raise ValueError('domain image list differs')
    for f,h in m['files'].items():
        if st.sha(out/f)!=h:raise ValueError('frozen prediction audit table modified')
    for rec in m['records']:
        path=out/rec['file']
        if st.sha(path)!=rec['sha256'] or st.sha(path.with_suffix('.json'))!=rec['record_sha256']:raise ValueError('frozen prediction modified')
        load_record(bundle,out,domain,rec['sample_id'])
    return m
