"""Explicit support retrieval, lazy unique-edge proofs and whole-image LOO."""
from collections import Counter
import json
import sqlite3
import time
from pathlib import Path
import numpy as np
from . import inputs as ip,storage as st
from .method import LAYERS,edge_key,witness,reduction,W,C
from .neighbors import topk,torch_topk
from DINOv3.MADEqual.aebad_s_target_local_segment.geometry import geometry_layer


def pair_path(out,domain,layer,a,c):return Path(out)/'neighbors'/domain/f'L{layer}'/f'{a:02d}_to_{c:02d}.npz'


def configure(out,device,qchunk,rchunk):
    config=dict(k=5,distance='raw unnormalized Euclidean',dtype='float64',
        implementation='SciPy direct Euclidean cdist' if device=='cpu' else 'Torch direct FP64 cdist compute_mode=donot_use_mm_for_euclid_dist',
        tie='distance ascending then original reference row ascending',query_chunk=qchunk,reference_chunk=rchunk,
        numerical_backend='cpu' if device=='cpu' else 'cuda',fallback=False)
    p=Path(out)/'neighbor_config.json'
    if p.exists():
        if st.read_json(p)!=config:raise ValueError('frozen neighbor configuration differs; use a new output')
    else:st.write_json(p,config)
    return config


def prepare(bundle,out,locations,domain,layer,*,device='cpu',query_chunk=64,reference_chunk=1024):
    tick=time.perf_counter();out=st.initialize(out,bundle)
    config=st.read_json(out/'neighbor_config.json')
    expected_impl='cpu' if device=='cpu' else 'cuda'
    if (config['numerical_backend'],config['query_chunk'],config['reference_chunk'])!=(expected_impl,query_chunk,reference_chunk):raise ValueError('neighbor worker configuration differs')
    if (out/'neighbors'/domain/f'L{layer}'/'status.json').exists():
        # Check the PREVIOUS completed record before any resume can rewrite it.
        load_tables(bundle,out,domain,layer)
        saved=st.read_json(out/'neighbors'/domain/f'L{layer}'/'status.json')
        return dict(saved,new_pairs=0,seconds=time.perf_counter()-tick)
    banks=ip.load_layer(bundle.expected,locations,domain,layer);proof=[];computed=0
    if min(map(len,banks.values()))<5:raise ValueError('every other support must contain at least five original rows; no k reduction')
    device_banks=None
    if device!='cpu':
        import torch
        dev=torch.device(device);torch.cuda.set_device(dev)
        device_banks={c:torch.as_tensor(bank,dtype=torch.float64,device=dev) for c,bank in banks.items()}
    for a in sorted(banks):
        for c in sorted(banks):
            if a==c:continue
            meta=dict(domain=domain,layer=layer,query_support=a,reference_support=c,
                query_feature_sha256=bundle.expected[domain][a-1]['feature_sha256'],
                reference_feature_sha256=bundle.expected[domain][c-1]['feature_sha256'],config=config)
            path=pair_path(out,domain,layer,a,c);record=st.restore(path,bundle.identity,meta)
            if record is None:
                def progress(done,total):
                    if done==total or done%(query_chunk*16)==0:print(f'[witness NN progress] {domain}/L{layer} {a}->{c} {done}/{total}',flush=True)
                if device_banks is None:
                    idx,d=topk(banks[a],banks[c],device=device,query_chunk=query_chunk,reference_chunk=reference_chunk,progress=progress)
                else:
                    idx,d=torch_topk(device_banks[a],device_banks[c],device=device,query_chunk=query_chunk,reference_chunk=reference_chunk,progress=progress)
                record=st.seal(path,bundle.identity,dict(indices=idx,distances=d),meta);computed+=1
            proof.append(dict(file=path.relative_to(out).as_posix(),sha256=record['sha256'],record_sha256=st.sha(path.with_suffix('.json'))))
            print(f'[witness neighbors] {domain}/L{layer} {a}->{c} verified',flush=True)
    state=dict(status='support_neighbors_complete',identity=bundle.identity,domain=domain,layer=layer,
        pairs=proof,new_pairs=computed,seconds=time.perf_counter()-tick,device=device,model_inference=False,feature_extraction=False,
        config_sha256=st.sha(out/'neighbor_config.json'))
    st.write_json(out/'neighbors'/domain/f'L{layer}'/'status.json',state)
    return state


def load_tables(bundle,out,domain,layer):
    out=Path(out);status=st.read_json(out/'neighbors'/domain/f'L{layer}'/'status.json')
    if status['status']!='support_neighbors_complete' or status['identity']!=bundle.identity or len(status['pairs'])!=56:
        raise ValueError('complete original 8x7 directed neighbor tables required')
    if status['config_sha256']!=st.sha(out/'neighbor_config.json'):raise ValueError('neighbor build config changed')
    indices={};distances={}
    for a in range(1,9):
        for c in range(1,9):
            if a==c:continue
            path=pair_path(out,domain,layer,a,c)
            sealed=next(x for x in status['pairs'] if x['file']==path.relative_to(out).as_posix())
            if st.sha(path.with_suffix('.json'))!=sealed['record_sha256']:raise ValueError('neighbor historical record modified')
            arrays,record=st.arrays(path,bundle.identity)
            if record['sha256']!=sealed['sha256']:raise ValueError('neighbor file identity changed')
            n=int(np.prod(bundle.expected[domain][a-1]['grid']));m=int(np.prod(bundle.expected[domain][c-1]['grid']))
            idx,d=arrays['indices'],arrays['distances']
            if idx.shape!=(n,5) or idx.dtype!=np.int64 or d.shape!=idx.shape or d.dtype!=np.float64 or not np.isfinite(d).all() or (d<0).any() or (idx<0).any() or (idx>=m).any():raise ValueError('invalid original-row neighbor table')
            if np.any(d[:,1:]<d[:,:-1]) or np.any((d[:,1:]==d[:,:-1])&(idx[:,1:]<=idx[:,:-1])):raise ValueError('neighbor tie/order mismatch')
            if np.any(np.diff(np.sort(idx,axis=1),axis=1)==0):raise ValueError('duplicate neighbor row')
            indices[a,c],distances[a,c]=idx,d
    return indices,distances,st.digest(status['pairs'])


class Connections:
    """One DB per domain/layer; namespace also excludes the WHOLE held-out image."""
    def __init__(self,path,identity,banks,neighbors,neighbor_identity):
        self.banks,self.neighbors=banks,neighbors;self.seconds=0.;self.new=0;self.reused=0
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(path)
        self.db.execute('CREATE TABLE IF NOT EXISTS metadata (id INTEGER PRIMARY KEY,value TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS edges (key TEXT PRIMARY KEY,payload TEXT NOT NULL,sha256 TEXT NOT NULL)')
        meta=json.dumps(dict(identity=identity,neighbors=neighbor_identity),sort_keys=True)
        row=self.db.execute('SELECT value FROM metadata WHERE id=1').fetchone()
        if row and row[0]!=meta:self.db.close();raise ValueError('connection cache support/input identity differs')
        if not row:self.db.execute('INSERT INTO metadata VALUES(1,?)',(meta,));self.db.commit()

    def get(self,edge,excluded=()):
        canonical=edge_key(*edge);excluded=tuple(sorted(set(map(int,excluded))))
        if set(canonical[::2])&set(excluded):raise ValueError('held-out endpoint forbidden')
        key=json.dumps([excluded,canonical],separators=(',',':'))
        old=self.db.execute('SELECT payload,sha256 FROM edges WHERE key=?',(key,)).fetchone()
        if old:
            proof=json.loads(old[0])
            if st.digest(proof)!=old[1] or proof['edge']!=list(canonical) or proof['excluded']!=list(excluded):raise ValueError('connection proof modified')
            self.reused+=1;return proof
        tick=time.perf_counter();proof=witness(self.banks,self.neighbors,canonical,excluded)
        self.seconds+=time.perf_counter()-tick;self.new+=1
        self.db.execute('INSERT INTO edges VALUES(?,?,?)',(key,json.dumps(proof,sort_keys=True,allow_nan=False),st.digest(proof)))
        if self.new%256==0:self.db.commit()
        return proof

    def close(self):self.db.commit();self.db.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()


def candidate_gates(target,indices,relations,excluded=(),support_ids=None):
    """Stable Top2 across per-image Top1 only; compact query order is preserved."""
    t,idx=np.asarray(target),np.asarray(indices);ids=np.asarray(support_ids if support_ids is not None else range(1,len(t)+1))
    if t.ndim!=2 or idx.shape!=t.shape or len(ids)!=len(t) or len(set(ids))!=len(ids):raise ValueError('per-image Top1 tables required')
    if np.any(np.isin(ids,list(excluded))):raise ValueError('held-out image is still a reference')
    # Sort support rows first, so distance ties are in original manifest order.
    order=np.argsort(ids,kind='stable');t,idx,ids=t[order],idx[order],ids[order]
    if len(t)<2:raise ValueError('at least two retained supports required')
    chosen=np.argsort(t,axis=0,kind='stable')[:2];gate=np.empty(t.shape[1],bool);proofs={};counts=Counter()
    if hasattr(relations,'get_many'):
        from .batch_witness import EDGE_REASONS
        columns=np.arange(t.shape[1]);a,b=chosen
        edges=np.stack((ids[a],idx[a,columns],ids[b],idx[b,columns]),axis=1)
        masks,reasons,proof=relations.get_many(edges,excluded)
        gate=masks!=0;unique_masks=proof['source_mask']
        import hashlib
        identity=hashlib.sha256()
        identity.update(np.asarray(sorted(excluded),np.int64).tobytes())
        for k in ('edges','source_mask','edge_reason'):identity.update(proof[k].tobytes())
        return gate,dict(candidate_positions=len(gate),accepted_positions=int(gate.sum()),
            unique_candidate_connections=len(unique_masks),unique_accepted_connections=int((unique_masks!=0).sum()),
            effective_witness_sources=int(np.unpackbits(unique_masks[:,None],axis=1).sum()),
            accepted_position_fraction=float(gate.mean()) if len(gate) else None,
            unique_connection_acceptance_fraction=float((unique_masks!=0).mean()) if len(unique_masks) else None,
            source_count_aggregation='sum over unique candidate connections, not independent query positions',
            rejection_reasons={EDGE_REASONS[int(k)]:int(v) for k,v in zip(*np.unique(reasons,return_counts=True))},
            connection_proof_identity=identity.hexdigest(),connection_proof_format='batched_array_v1')
    for k,(a,b) in enumerate(chosen.T):
        key=edge_key(ids[a],idx[a,k],ids[b],idx[b,k])
        if key not in proofs:proofs[key]=relations.get(key,excluded)
        proof=proofs[key];gate[k]=proof['accepted'];counts[proof['reason']]+=1
    rows=list(proofs.values())
    return gate,dict(candidate_positions=len(gate),accepted_positions=int(gate.sum()),unique_candidate_connections=len(rows),
        unique_accepted_connections=sum(p['accepted'] for p in rows),effective_witness_sources=sum(len(p['sources']) for p in rows),
        accepted_position_fraction=float(gate.mean()) if len(gate) else None,
        unique_connection_acceptance_fraction=sum(p['accepted'] for p in rows)/len(rows) if rows else None,
        source_count_aggregation='sum over unique candidate connections, not independent query positions',
        rejection_reasons=dict(counts),connection_proof_identity=st.digest(sorted((list(k),st.digest(v)) for k,v in proofs.items())))


def loo_one(banks,indices,distances,heldout,relations):
    retained=sorted(set(banks)-{heldout})
    t=np.stack([distances[heldout,c][:,0] for c in retained]).astype(np.float64)/1280
    idx=np.stack([indices[heldout,c][:,0] for c in retained]);t=t.astype(np.float32)
    gate,counts=candidate_gates(t,idx,relations,excluded=(heldout,),support_ids=retained)
    discrete=t.min(0)
    geometry,diag=geometry_layer(t,idx,discrete,[banks[c] for c in retained])
    seg=geometry['K8_SEGMENT'];new,reduction_row=reduction(discrete,seg,gate)
    values=dict(TARGET7_DISCRETE=discrete,TARGET7_SEGMENT=seg,TARGET7_WITNESS=new[W],TARGET7_MATCHED_UNIFORM=new[C])
    rows=[]
    for name,value in values.items():
        delta=discrete.astype(np.float64)-value
        rows.append(dict(branch=name,scope='TARGET_ONLY_LOO_COVERAGE',heldout_support=heldout,
            reference_supports=','.join(map(str,retained)),witness_excluded_support=heldout,patches=len(value),
            mean=float(value.mean(dtype=np.float64)),p95=float(np.quantile(value,.95)),p99=float(np.quantile(value,.99)),max=float(value.max()),
            reduction_total=float(delta.sum(dtype=np.float64)),reduction_mean=float(delta.mean()),
            effective_reduction_fraction=float(np.mean(delta>1e-7+1e-6*np.abs(discrete))),
            **counts,degenerate_connections=diag['degenerate_count'],beta=reduction_row['beta']))
    return rows,diag


def support_loo(bundle,out,locations,domain,layer,*,device='cpu',witness_batch=256):
    tick=time.perf_counter();out=st.initialize(out,bundle);banks=ip.load_layer(bundle.expected,locations,domain,layer)
    idx,dist,ni=load_tables(bundle,out,domain,layer);path=out/'support_loo'/domain/f'L{layer}.json'
    if path.exists():
        saved=st.read_json(path)
        validate_loo(saved,bundle.identity,domain,layer,ni)
        return saved
    rows=[]
    from .batch_witness import Engine
    with Engine(banks,idx,device=device,batch_size=witness_batch,
                path=out/'connections_batched'/domain/f'L{layer}',identity=bundle.identity,neighbor_identity=ni) as relations:
        for h in sorted(banks):
            local,diag=loo_one(banks,idx,dist,h,relations)
            rows.extend(dict(domain=domain,layer=layer,**v) for v in local)
            print(f'[witness whole-image LOO] {domain}/L{layer} heldout={h}',flush=True)
        state=dict(identity=bundle.identity,status='target_only_support_LOO_complete',domain=domain,layer=layer,
            neighbors=ni,rows=rows,rows_sha256=st.digest(rows),seconds=time.perf_counter()-tick,
            connection_seconds=relations.seconds,scalar_boundary_fallback_edges=relations.scalar_fallback_edges,
            connection_timing=relations.timing(),
            witness_device=device,model_inference=False,feature_extraction=False)
    st.write_json(path,state);return state


def validate_loo(saved,identity,domain,layer,neighbor_identity):
    expected={(h,b) for h in range(1,9) for b in ('TARGET7_DISCRETE','TARGET7_SEGMENT','TARGET7_WITNESS','TARGET7_MATCHED_UNIFORM')}
    if (saved['identity']!=identity or saved['neighbors']!=neighbor_identity or saved['status']!='target_only_support_LOO_complete'
        or saved['domain']!=domain or saved['layer']!=layer or len(saved['rows'])!=32
        or {(v['heldout_support'],v['branch']) for v in saved['rows']}!=expected):raise ValueError('LOO identity/scope/completeness differs')
    if st.digest(saved['rows'])!=saved['rows_sha256']:raise ValueError('LOO metrics modified')
    for row in saved['rows']:
        if (row['scope']!='TARGET_ONLY_LOO_COVERAGE' or row['witness_excluded_support']!=row['heldout_support']
            or set(map(int,row['reference_supports'].split(',')))!=set(range(1,9))-{row['heldout_support']}):raise ValueError('LOO whole-image exclusion differs')
