"""Original feature/Exact adapters, bounded resident references, no training imports."""
from pathlib import Path
import numpy as np
from . import inputs as ip
from .protocol import CONFIG,LAYERS,DOMAINS,VERSION
from .inputs import st
from . import clustering,numerics
from DINOv3.MADEqual.aebad_s_reference_residual_segmentation.execution import ResidentExact
from DINOv3.MADEqual.aebad_s_reference_residual_segmentation.retrieval import target_nearest


def bundle(ref,keys=None):
    x,r=st.get(ref['path'],keys)
    if r['seal']!=ref['seal']:raise ValueError('frozen original asset manifest changed: '+ref['path'])
    return x,r


class References:
    def __init__(self,a):
        self.a=a;self.assets=ip.assets(a);self.c=ip.catalog(a);self.search=ResidentExact(a.device,int(a.reference_cache_gib*2**30))
        self.store=st.Store(80);self.searches=0;self.cached_Source_NN=0;self.precision_ready=False
    def precision(self):
        if not self.precision_ready:
            from DINOv3.MADEqual.aebad_s_reference_residual_segmentation.runtime import reliable_precision
            reliable_precision(self.a.device);self.precision_ready=True
    def load(self,ref,keys):
        x,r=self.store.load(ref['path'],keys)
        if r['seal']!=ref['seal']:raise ValueError('original reference changed')
        return x,r
    def source(self,name,layer):
        ref=self.assets['references'][name];x,r=self.load(ref,(f'L{layer}',));return x[f'L{layer}'],r['seal']
    def normal(self,row,layer):
        ref=self.assets['normal_features'][row['relative_path']];x,r=self.load(ref,(f'q{layer}','positions'))
        return x[f'q{layer}'],x['positions'],r
    def parts(self,domain,layer):
        parts=[];seals=[];offsets=[0]
        for row in self.c['supports'][domain]:
            x,_,r=self.normal(row,layer);parts.append(x);seals.append(r['seal']);offsets.append(offsets[-1]+len(x))
        return parts,seals,offsets
    def exact(self,q,name,layer):
        self.precision()
        bank,seal=self.source(name,layer);value=self.search.exact(q,bank,(name,seal,layer));self.searches+=1;return value
    def formal(self,q,domain,layer):
        self.precision()
        qt=self.search.query(q);source,sseal=self.source('source',layer)
        ds,si=self.search.exact(qt,source,('source',sseal,layer));self.searches+=1
        parts,seals,offsets=self.parts(domain,layer)
        dt,ti,audit=target_nearest(q,None,offsets,self.a.device,parts=parts,search=self.search,
            keys=[('target',s,layer) for s in seals],query=qt)
        self.searches+=len(parts)
        return dict(ds=ds,si=si,dt=dt,ti=ti,perd=audit['per_support_distances'],peri=audit['per_support_indices']),parts,dict(source_seal=sseal,target_seals=seals,target_offsets=offsets,
            target_origins=[r['relative_path'] for r in self.c['supports'][domain]])


def fit_positions(positions,relative):
    positions=np.asarray(positions,np.int64);ordered=np.sort(positions)
    if len(ordered)!=len(np.unique(ordered)):raise ValueError('duplicate physical positions')
    if len(ordered)>CONFIG['fit_patch_cap']:
        rng=st.rng(VERSION,42,'physical_fit_patch',relative)
        ordered=np.sort(rng.choice(ordered,CONFIG['fit_patch_cap'],replace=False))
    return ordered


def prepare(a):
    r=ip.inventory(a);ip.freeze(a,r);refs=References(a);new=0
    for layer in LAYERS:
        p=a.cache_root/'clusters'/f'L{layer}'
        if (p/'manifest.json').exists():
            _,meta=st.get(p)
            if meta['metadata']['identity']!=ip.identity(a):raise ValueError('cluster identity differs')
            continue
        source,seal=refs.source('source',layer)
        centers,formal,diagnostic=clustering.kmeans(source,seed=42,groups=32,cap=16384,iterations=30)
        arrays=dict(centers=centers,source_groups=formal)
        for fold in ('A','B'):
            bank,_=refs.source('fold_'+fold,layer);arrays['fold_'+fold+'_groups']=clustering.assign(bank,centers)
        st.put(p,arrays,dict(identity=ip.identity(a),layer=layer,formal_Source_seal=seal,diagnostic=diagnostic));new+=1
        print(f'[SACP Source groups] L{layer} 32 centers; full bank retained',flush=True)
    return dict(status='prepared',new_clustering_layers=new,training=False,feature_forwards=0)


def old_source_indices(a,row,reference,layer,positions,raw_seal):
    """Only Source-normal opposite-fold NN with the SAME complete query/bank."""
    assets=ip.assets(a)
    # Exact all_items normal-ID rule, without importing/running its training pipeline.
    item_id=st.digest([row['relative_path'],'normal'])[:28]
    path=Path(assets['references']['parent_cache'])/'training_queries'/item_id
    if not (path/'complete.json').exists():return None
    record=st.sealed_json(path/'complete.json')
    if record['normal_feature_seal']!=raw_seal:return None
    if not (path/'source_nn/manifest.json').exists():return None
    keys=(f'i{layer}',f'd{layer}')
    if 'positions' in st.sealed_json(path/'source_nn/manifest.json')['files']:keys+=('positions',)
    x,r=st.get(path/'source_nn',keys)
    if r['seal']!=record['source_seal'] or r['metadata']['source']!=reference:return None
    if r['metadata']['source_seals'][LAYERS.index(layer)]!=assets['references'][reference]['seal']:return None
    available=x.get('positions',np.asarray(record['positions'],np.int64));inverse={int(p):i for i,p in enumerate(available)}
    if not all(int(p) in inverse for p in positions):return None
    return np.asarray(x[f'i{layer}'][[inverse[int(p)] for p in positions]],np.int64)


def correspondence(a,refs,row,kind,domain,fold,layer,cluster,sampled):
    path=a.cache_root/'correspondences'/f'L{layer}'/fold/(domain if kind=='Target' else 'Source')/ip.key(row)
    bank_name='fold_'+fold;ref=ip.assets(a)['normal_features'][row['relative_path']]
    expected=dict(identity=ip.identity(a),image=row['relative_path'],kind=kind,domain=domain,fold=fold,layer=layer,
        raw_feature_seal=ref['seal'],bank_seal=ip.assets(a)['references'][bank_name]['seal'],cluster_seal=cluster[1]['seal'])
    if (path/'manifest.json').exists():
        _,r=st.get(path)
        if r['metadata']['inputs']!=expected:raise ValueError('correspondence inputs differ')
        return path,False
    if row['relative_path'] not in sampled:
        q,pos,r=refs.normal(row,layer);chosen=fit_positions(pos,row['relative_path']);inverse=np.argsort(pos)
        sampled[row['relative_path']]=(np.array(q[inverse[chosen]],np.float32),chosen,r['seal'])
    q,chosen,raw_seal=sampled[row['relative_path']]
    idx=old_source_indices(a,row,bank_name,layer,chosen,raw_seal) if kind=='Source' else None
    reused=idx is not None
    if idx is None:_,idx=refs.exact(q,bank_name,layer)
    else:refs.cached_Source_NN+=1
    bank,_=refs.source(bank_name,layer)
    if idx.shape!=(len(q),) or (idx<0).any() or (idx>=len(bank)).any():raise ValueError('invalid fold nearest rows')
    groups=cluster[0][bank_name+'_groups'][idx];keep=[]
    for g in np.unique(groups):
        local=np.flatnonzero(groups==g)
        if len(local)>CONFIG['per_image_group_cap']:
            local=np.sort(st.rng(VERSION,42,'group_sample',row['relative_path'],fold,layer,int(g)).choice(local,64,replace=False))
        keep.extend(local.tolist())
    keep=np.sort(keep);delta=q[keep]-np.asarray(bank[idx[keep]],np.float32)
    st.put(path,dict(delta=delta,group=groups[keep],positions=chosen[keep],nn_index=idx[keep]),dict(inputs=expected,cached_opposite_fold_NN=reused,
        physical_positions_before_group_cap=len(chosen),physical_positions_after_group_cap=len(keep)))
    return path,True


def fit(a):
    refs=References(a);c=refs.c;assets=refs.assets;controls=assets['controls'];jobs=[];new=0
    for layer in LAYERS:
        pending=[d for d in DOMAINS if not (a.run_root/'fit'/d/f'L{layer}.json').exists()]
        if not pending:continue
        cluster=st.get(a.cache_root/'clusters'/f'L{layer}');sampled={};sources=[]
        for fold in ('A','B'):
            other='B' if fold=='A' else 'A'
            for row in controls[other]:
                p,created=correspondence(a,refs,row,'Source','Source',fold,layer,cluster,sampled);new+=created;sources.append((p,row['relative_path'],fold))
        def grouped(records):
            values={g:[] for g in range(32)}
            for path,image,fold in records:
                x,_=st.get(path)
                for g in np.unique(x['group']):values[int(g)].append(dict(image=image,fold=fold,delta=np.array(x['delta'][x['group']==g])))
            return values
        source_blocks=grouped(sources)
        for domain in pending:
            targets=[]
            for fold in ('A','B'):
                for row in c['supports'][domain]:
                    p,created=correspondence(a,refs,row,'Target',domain,fold,layer,cluster,sampled);new+=created;targets.append((p,row['relative_path'],fold))
            target_blocks=grouped(targets);result=np.zeros((3,32,1280,2),np.float32);ranks=np.zeros(32,np.int16);diagnostics=[]
            for group in range(32):
                basis,diag=numerics.fit_group(source_blocks[group],target_blocks[group],st.seed(VERSION,domain,layer,group))
                result[:,group]=basis;ranks[group]=diag['directions'];diagnostics.append(dict(domain=domain,layer=layer,group=group,**diag))
            r=st.put(a.cache_root/'subspaces'/domain/f'L{layer}',dict(bases=result,ranks=ranks),dict(identity=ip.identity(a),domain=domain,layer=layer,cluster_seal=cluster[1]['seal']))
            value=st.sealed_json(a.run_root/'fit'/domain/f'L{layer}.json',dict(identity=ip.identity(a),subspace_seal=r['seal'],diagnostics=diagnostics))
            jobs.append(dict(domain=domain,layer=layer,effective_groups=int(np.sum(ranks>0)),joint_fallback_groups=int(np.sum(ranks==0))))
            print(f'[SACP fit] {domain}/L{layer} effective={np.sum(ranks>0)}/32',flush=True)
        del sampled,source_blocks
    st.write_csv(a.run_root/'fit_stage_jobs.csv',jobs)
    return dict(status='fit_complete',new_correspondences=new,jobs=jobs,Exact_searches=refs.searches,cached_Source_NN=refs.cached_Source_NN,execution=refs.search.stats,
        query_image_forward_calls=0,training=False)


def raw_query(a,sample):
    ref=ip.assets(a)['queries'].get(sample['domain']+'/'+sample['sample_id'])
    if ref is None:return None
    x,r=bundle(ref,tuple([f'q{l}' for l in LAYERS]+['positions']));m=r['metadata'];pos=x['positions'];n=int(np.prod(m['grid']))
    if not np.array_equal(np.sort(pos),np.arange(n)):raise ValueError('cached Q incomplete')
    return {l:x[f'q{l}'] for l in LAYERS},dict(grid=m['grid'],positions=pos)


def extractor(a):
    from types import SimpleNamespace
    from DINOv3.MADEqual.aebad_s_reference_residual_segmentation.runtime import host,extraction_record
    # Fast identity uses the explicitly selected parent, never loads a trained head.
    probe=SimpleNamespace(**vars(a));probe.run_root=a.run_root/'extractor_identity_only_not_written'
    probe.recent_run=[Path(ip.assets(a)['references']['parent_run'])]
    actual=extraction_record(probe)
    if actual!=ip.assets(a)['references']['extraction']:raise ValueError('fresh query extractor differs from parent host/precision/software')
    return host(a)[0]
