"""Original FP32 Exact NN, immutable original references and row provenance."""
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
import shutil
import tempfile
import numpy as np
from . import storage as st,inputs as ip
from .protocol import LAYERS,CONFIG


class References:
    def __init__(self,a):
        self.a=a;self.c=ip.catalog(a);self.store=st.Store(64);self.cache=OrderedDict();self._search=None
        self.rows={r['relative_path']:r for r in self.c['source']+sum(self.c['supports'].values(),[])}
    def source(self,name,layer):
        p=self.a.cache_root/('source_original' if name=='formal' else 'source_folds/'+name)
        x,r=self.store.load(p,(f'L{layer}',));return x[f'L{layer}'],r['seal']
    @property
    def search(self):
        if self._search is None:
            from .execution import resident
            self._search=resident(self.a)
        return self._search
    def parts(self,origins,layer):
        arrays=[];seals=[];offsets=[0]
        for path in origins:
            x,r=self.store.load(ip.normal_path(self.a,self.rows[path]),(f'q{layer}',))
            arrays.append(x[f'q{layer}']);seals.append(r['seal']);offsets.append(offsets[-1]+len(arrays[-1]))
        return arrays,seals,offsets
    def target(self,origins,layer):
        k=(tuple(origins),layer)
        if k in self.cache:self.cache.move_to_end(k);return self.cache[k]
        arrays,seals,offsets=self.parts(origins,layer)
        value=(np.concatenate(arrays),seals,offsets);self.cache[k]=value
        while len(self.cache)>4:self.cache.popitem(last=False)
        return value
    def nearest_source(self,q,name,layer,query):
        bank,seal=self.source(name,layer)
        return (*exact_with_audit(q,bank,self.a.device,search=self.search,key=('source',seal,layer),query=query),seal)
    def nearest_target(self,q,origins,layer,query):
        parts,seals,offsets=self.parts(origins,layer)
        result=target_nearest(q,None,offsets,self.a.device,parts=parts,search=self.search,
            keys=[('target',seal,layer) for seal in seals],query=query)
        return (*result,seals,offsets)
    def gather(self,source_name,origins,layer,source_indices,target_indices):
        """Fetch only tile rows, never concatenate seven/eight full support grids."""
        source,sseal=self.source(source_name,layer);parts,seals,offsets=self.parts(origins,layer)
        if (source_indices<0).any() or (source_indices>=len(source)).any():raise ValueError('invalid tile Source index')
        if (target_indices<0).any() or (target_indices>=offsets[-1]).any():raise ValueError('invalid tile Target index')
        target=np.empty((len(target_indices),source.shape[1]),np.float32)
        for bank,lo,hi in zip(parts,offsets[:-1],offsets[1:]):
            use=(target_indices>=lo)&(target_indices<hi);target[use]=bank[target_indices[use]-lo]
        return np.array(source[source_indices]),target,dict(source_seal=sseal,target_seals=seals,target_offsets=offsets,target_origins=origins)
    def tensors(self,source_name,origins,layer):
        source,sseal=self.source(source_name,layer);target,tseals,offsets=self.target(origins,layer)
        return source,target,dict(source_seal=sseal,target_seals=tseals,target_offsets=offsets,target_origins=origins)


def exact_with_audit(q,bank,device,*,search=None,key=None,query=None):
    from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.model import exact
    distance,index=exact(q,bank,device) if search is None else search.exact(q if query is None else query,bank,key)
    if index.shape!=(len(q),) or (index<0).any() or (index>=len(bank)).any():raise ValueError('invalid exact reference row')
    maximum=0.;strict_failed=0;bound_failed=0
    # Direct original vectors audit; do not silently replace original cdist values.
    gamma=np.float64(16*q.shape[1]*np.finfo(np.float32).eps)
    for i in range(0,len(q),2048):
        raw=np.asarray(q[i:i+2048],np.float32);ref=np.asarray(bank[index[i:i+2048]],np.float32)
        direct=np.linalg.norm(raw-ref,axis=1).astype(np.float32)/1280
        observed=distance[i:i+len(raw)];delta=np.abs(direct-observed);maximum=max(maximum,float(delta.max(initial=0)))
        strict_failed+=int((delta>1e-7+1e-6*np.abs(observed)).sum())
        bound=gamma*(np.linalg.norm(raw,axis=1)+np.linalg.norm(ref,axis=1))**2/(1280**2)
        bound_failed+=int((np.abs(direct.astype(float)**2-observed.astype(float)**2)>bound+1e-14).sum())
    if not np.isfinite(distance).all() or bound_failed:raise ValueError('NN row/vector audit exceeds FP32 cdist roundoff bound')
    return distance,index,dict(max_abs_direct_residual_error=maximum,strict_1e7_1e6_failed_positions=strict_failed,
        analytic_bound_failed=bound_failed,criterion='squared_error <= 16*D*eps32*(norm(q)+norm(reference))^2/1280^2 + 1e-14',
        unit='Euclidean/1280 once',stored_distance='original_FP32_cdist_not_replaced')


def target_nearest(q,bank,offsets,device,*,parts=None,search=None,keys=None,query=None):
    """Same per-support cdist path and manifest-order ties as original K8."""
    values=[];indices=[];audits=[]
    for j,(lo,hi) in enumerate(zip(offsets[:-1],offsets[1:])):
        part=bank[lo:hi] if parts is None else parts[j]
        d,idx,diag=exact_with_audit(q,part,device,search=search,key=None if keys is None else keys[j],query=query)
        values.append(d);indices.append(idx+lo);audits.append(diag)
    d=np.stack(values);ii=np.stack(indices);which=d.argmin(0);cols=np.arange(len(q))
    return d[which,cols],ii[which,cols],dict(support_audits=audits,per_support_distances=d,per_support_indices=ii)


def references_for_tile(refs,policy,layer,indices):
    source,target,meta=refs.tensors(policy['source'],policy['target_origins'],layer)
    return np.asarray(source[indices[0]],np.float32),np.asarray(target[indices[1]],np.float32),meta


@contextmanager
def sampling_matrix(spool,root,fold,layer):
    """Sequential HDD read once; repeated sampler reads use ONE layer on SSD."""
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'fold_{fold}_L{layer}_',dir=root) as own:
        path=Path(own)/'candidate.f32';shutil.copyfile(spool.path,path)
        if path.stat().st_size!=spool.rows*spool.columns*4:raise ValueError('staged candidate byte count differs')
        matrix=np.memmap(path,mode='r',dtype=np.float32,shape=(spool.rows,spool.columns))
        try:yield matrix
        finally:matrix._mmap.close()


def build_source_folds(a):
    """New training banks; original augmentation, distance sampler and 100 rounds.

    Deliberately no threshold_fraction split here: input is ALREADY the original
    455 training-only images. Dropping every eighth of a fold would be wrong.
    """
    import torch
    import tempfile
    from external_baselines.superadd_external.memory_variants import _MatrixSpool,_official_subsample_values_with_indices
    from DINOv3.relation_reliability.superadd_host import image_tensor
    from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.fresh_inputs import Data
    from .runtime import host
    c=ip.catalog(a);groups=ip.folds(c);model=None;calls=0;resources=[]
    def record_size(fold,metadata):
        rows=metadata['candidate_ledger'][-1]['stop'];layer=rows*1280*4
        resources.append(dict(fold=fold,candidate_rows_per_layer=rows,disk_spool_file_bytes=layer*4,
            maximum_SSD_staged_file_bytes=layer,SSD_root=str(a.sampling_scratch),cleanup='only owned temporary directories'))
    for fold in ('A','B'):
        rows=[r for r in ip.train_source(c) if groups[r['relative_path']]==fold];path=a.cache_root/'source_folds'/('fold_'+fold)
        fold_seed=int(st.seed('Source_training_bank',fold)%2**31)
        expected=dict(identity=ip.identity(a),kind='training_only_opposite_fold',fold=fold,random_seed=fold_seed,
            origins=[r['relative_path'] for r in rows],origin_indices=[r['source_index'] for r in rows],
            sampler='original_100_round_distance_based_normalize_false',budget=CONFIG['bank_rows'],augmentation='original_brightness_0.8_1.2')
        if (path/'manifest.json').exists():
            _,r=st.get(path)
            if any(r['metadata'].get(k)!=v for k,v in expected.items()):raise ValueError('training Source fold identity changed')
            record_size(fold,r['metadata'])
            continue
        from . import reuse
        if reuse.fold(a,'fold_'+fold,expected):
            record_size(fold,st.sealed_json(path/'manifest.json')['metadata']);continue
        if model is None:model,_=host(a)
        import random
        seed=fold_seed;random.seed(seed);np.random.seed(seed)
        torch.random.default_generator.manual_seed(seed)
        if str(a.device).startswith('cuda'):
            with torch.cuda.device(a.device):torch.cuda.manual_seed(seed)
        scratch=a.cache_root/'fold_candidate_scratch';scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='fold_'+fold+'_',dir=scratch) as temp:
            spools={l:_MatrixSpool(Path(temp)/f'L{l}.f32') for l in LAYERS};ledger=[];start=0
            with Data(Path(c['data_root'])) as data,torch.inference_mode():
                for number,row in enumerate(rows,1):
                    tensor=image_tensor(data.image(row)).to(a.device)[None]
                    result=model.patch_exec(model.augmented_preprocessing(tensor),model.backbone);n=None
                    for l,value in zip(LAYERS,result):
                        raw=np.asarray(value,np.float32)
                        if raw.shape[-1]!=1280 or not np.isfinite(raw).all():raise ValueError('invalid augmented Source features')
                        q=raw.reshape(-1,1280)
                        if n is not None and n!=len(q):raise ValueError('augmented Source layer geometries differ')
                        n=len(q);spools[l].append(q)
                    if len(result)!=4:raise ValueError('four original augmented Source layers required')
                    ledger.append(dict(source_index=row['source_index'],relative_path=row['relative_path'],grid=list(raw.shape[-3:-1]),start=start,stop=start+n));start+=n;calls+=1
                    if number%10==0:print(f'[residual Source fold] {fold} {number}/{len(rows)} augmented',flush=True)
            arrays={}
            for l in LAYERS:
                with sampling_matrix(spools[l],a.sampling_scratch,fold,l) as candidates:
                    if len(candidates)<CONFIG['bank_rows']:raise ValueError('insufficient genuine candidates for 100K fold bank')
                    print(f'[residual Source sampling] fold={fold}/L{l} staged_bytes={candidates.nbytes} path={candidates.filename}',flush=True)
                    selected,indices=_official_subsample_values_with_indices(candidates,CONFIG['bank_rows'],torch.device(a.device),seed=None,iterations=100)
                    arrays[f'L{l}']=np.array(selected,copy=True);arrays[f'candidate{l}']=indices
                    starts=np.asarray([r['start'] for r in ledger],np.int64);owners=np.searchsorted(starts,indices,side='right')-1
                    arrays[f'origin{l}']=np.asarray([ledger[j]['source_index'] for j in owners],np.int32)
                    arrays[f'patch{l}']=indices-starts[owners]
            st.put(path,arrays,dict(expected,candidate_ledger=ledger));record_size(fold,dict(candidate_ledger=ledger))
    st.write_csv(a.run_root/'source_fold_resources.csv',resources)
    return dict(new_augmented_Source_forwards=calls,training_only=True,formal_bank_replaced=False,candidate_file_sizes=resources)
