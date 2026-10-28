"""Original full-grid extraction, augmented origin recovery, and explicit rebuild."""
from pathlib import Path
from types import FunctionType
import time
import numpy as np
from . import inputs as ip,provenance as pv,numerics as nm
from .inputs import st
from .protocol import LAYERS


def model(a):
    if not hasattr(a,'_model'):
        from . import feature_identity as fi
        fi.compatible(fi.native_extraction(a),fi.expected(a))
        from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.runtime import host,reliable_precision
        reliable_precision(a.device);a._model=host(a)[0]
    return a._model


def features(a,row,persist=False):
    """Single extraction, all four layers; queries are not persisted by default."""
    path=a.cache_root/'features'/ip.key(row)
    if (path/'manifest.json').exists():
        x,r=st.get(path)
        if r['metadata']['image']!=row:raise ValueError('feature image identity differs')
        from . import feature_identity as fi
        frozen=fi.small_record(a,a.run_root/'inputs.json')
        wanted=frozen['feature_inputs']
        if (r['metadata'].get('identity')!=frozen['identity'] or r['metadata'].get('extraction_identity')!=st.digest(wanted['extraction'])
                or r['metadata'].get('origin_binding')!=wanted['references'].get(row['relative_path'])):
            raise ValueError('cached feature extraction/provider differs from frozen input')
        return {l:x[f'q{l}'] for l in LAYERS},dict(r['metadata'],positions=x['positions'])
    ref=ip.feature_ref(a,row)
    if ref:
        x,r=st.get(ref['path'],(*ref['names'],ref['positions']))
        if r['seal']!=ref['seal']:raise ValueError('feature record changed')
        pos=np.asarray(x[ref['positions']],np.int64)
        q={l:x[name] for l,name in zip(LAYERS,ref['names'])};meta=dict(grid=ref['grid'],positions=pos)
    else:
        from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.fresh_inputs import Data
        from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.runtime import extraction
        with Data(a.data) as data:q,meta=extraction(model(a),data.image(row),a.device)
        a._new_forwards=getattr(a,'_new_forwards',0)+1
    meta=dict(meta,output_shape=[row['image_height']//4,row['image_width']//4])
    # Inspect the complete physical mapping once, without full payload SHA.
    if not np.array_equal(np.sort(meta['positions']),np.arange(np.prod(meta['grid']))):raise ValueError('raw feature grid incomplete')
    if persist:
        from . import feature_identity as fi
        st.put(path,{**{f'q{l}':q[l] for l in LAYERS},'positions':meta['positions']},
            dict(image=row,grid=meta['grid'],output_shape=meta['output_shape'],kind='original_complete_grid',origin=ref['path'] if ref else 'original_host',
                identity=ip.identity(a),extraction_identity=st.digest(fi.expected(a)),origin_binding=fi.binding(ref)))
    return q,meta


def source_path(a):
    # Only completed exact neighborhood bundles are accepted as external assets.
    return a.source_neighborhood or a.cache_root/'source_neighbors'


def source_bundle(a):
    if hasattr(a,'_source_bundle'):return a._source_bundle
    x,r=st.get(source_path(a))
    if r['metadata'].get('format')!='augmented_Source_neighbors_v1' or not r['metadata'].get('centers_verified'):raise ValueError('Source neighborhood provenance not verified')
    if r['metadata']['catalog_source']!=ip.catalog(a)['source']:raise ValueError('Source order/image identities differ')
    for l in LAYERS:
        if x[f'L{l}'].shape!=(100000,1280) or x[f'delta{l}'].shape!=(100000,4,1280) or x[f'valid{l}'].shape!=(100000,):raise ValueError('complete 100K Source and four directions required')
        image=np.asarray(x[f'image{l}']);pos=np.asarray(x[f'position{l}']);selected=np.asarray(x[f'selected{l}'])
        mapped,local=pv.map_indices(selected,r['metadata']['origins'])
        if not np.array_equal(mapped,image) or not np.array_equal(local,pos):raise ValueError('bank row/source coordinate mapping differs')
    a._source_bundle=(x,r)
    return x,r


def prepare_source(a):
    ip.freeze(a)
    if (source_path(a)/'manifest.json').exists():
        _,r=source_bundle(a);return dict(status='exact_Source_neighbors_reused',seal=r['seal'],new_forwards=0)
    if a.source_neighborhood:raise FileNotFoundError(a.source_neighborhood/'manifest.json')
    return rebuild_source(a) if a.rebuild_source else recover_source(a)


def recover_source(a):
    b=ip.bank_path(a);ix=ip.index_path(a)
    if b is None or ix is None:raise ValueError('cannot restore provenance; locate selected indices or explicitly choose --rebuild-source; no automatic rebuild')
    bank,br=st.get(b,tuple(f'L{l}' for l in LAYERS));rows=pv.layout(ip.catalog(a));indices={}
    with np.load(ix,allow_pickle=False) as z:
        for l in LAYERS:
            indices[l]=np.array(z[f'selected_{l}'],np.int64)
    # Real augmented centers are checked below against the current formal bank;
    # reading another 2GiB copy of the bank from this NPZ adds no origin evidence.
    import hashlib
    index_identity=hashlib.sha256(b''.join(indices[l].tobytes() for l in LAYERS)).hexdigest()
    origins={l:pv.map_indices(indices[l],rows) for l in LAYERS}
    delta={l:np.zeros((len(indices[l]),4,1280),np.float32) for l in LAYERS}
    valid={l:np.zeros(len(indices[l]),bool) for l in LAYERS};checks=[]
    # Original host constructor starts the frozen seed42 stream. Clone its exact
    # post-construction state, not context_reference.noaug and not guessed factors.
    all_recorded=bool(a.augmented_grids) and all((a.augmented_grids/f"source_{r['source_index']:03d}"/'manifest.json').exists() for r in rows)
    m=None if all_recorded else model(a)
    if m is not None and (m.augmented_preprocessing.min_brightness_factor,m.augmented_preprocessing.max_brightness_factor)!=(.8,1.2):raise ValueError('original Source brightness augmentation must remain0.8..1.2')
    clone=np.random.RandomState()
    if m is not None:clone.set_state(np.random.get_state())
    states=[]
    for row in rows:
        states.append(clone.get_state())
        row['brightness_factor']=float(clone.uniform(m.augmented_preprocessing.min_brightness_factor,m.augmented_preprocessing.max_brightness_factor)) if m is not None else None
        row['augmentation']='original_clip_brightness_then_bicubic_then_ImageNet'
        if a.augmented_grids:
            recorded=a.augmented_grids/f"source_{row['source_index']:03d}"/'manifest.json'
            if recorded.exists():
                actual=st.sealed_json(recorded)['metadata']
                for k in ('relative_path','source_index','grid'):
                    if actual[k]!=row[k]:raise ValueError('recorded augmented origin differs: '+k)
                # A real recorded augmentation takes precedence over replay guesses.
                row['brightness_factor']=actual['brightness_factor']
    import torch
    from DINOv3.relation_reliability.superadd_host import image_tensor
    from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.fresh_inputs import Data
    from contextlib import nullcontext
    with (nullcontext(None) if all_recorded else Data(a.data)) as data,torch.inference_mode():
        for number,row in enumerate(rows):
            part=a.cache_root/'source_recovery_parts'/f"source_{row['source_index']:03d}"
            partmeta=dict(Source_bank_seal=br['seal'],selected_index_identity=index_identity,origin=row)
            if (part/'manifest.json').exists():
                saved_part,part_record=st.get(part)
                if part_record['metadata']!=partmeta:raise ValueError('recovered origin checkpoint differs')
                for l in LAYERS:
                    take=np.flatnonzero(origins[l][0]==number)
                    delta[l][take]=saved_part[f'delta{l}'];valid[l][take]=saved_part[f'valid{l}']
                    checks.append(dict(layer=l,source_index=row['source_index'],rows=len(take),verified=True,max_abs_error=0.,atol=0.,rtol=0.))
                continue
            saved=None
            if a.augmented_grids:
                path=a.augmented_grids/f"source_{row['source_index']:03d}"
                if (path/'manifest.json').exists():
                    saved,rec=st.get(path);md=rec['metadata']
                    for key in ('relative_path','source_index','grid','brightness_factor'):
                        if md[key]!=row[key]:raise ValueError('saved augmented grid identity differs: '+key)
            if saved is None:
                np.random.set_state(states[number])
                processed=m.augmented_preprocessing(image_tensor(data.image(row)).to(a.device)[None])
                grids=m.patch_exec(processed,m.backbone)
                a._new_forwards=getattr(a,'_new_forwards',0)+1
            else:grids=[saved[f'L{l}'] for l in LAYERS]
            part_arrays={}
            for l,v in zip(LAYERS,grids):
                full=np.asarray(v,np.float32).reshape(*row['grid'],1280);images,pos=origins[l];take=np.flatnonzero(images==number)
                check=pv.check_centers(full.reshape(-1,1280)[pos[take]],bank[f'L{l}'][take])
                checks.append(dict(layer=l,source_index=row['source_index'],rows=len(take),**check))
                if not check['verified']:
                    st.write_csv(a.run_root/'source_center_verification.csv',checks)
                    raise ValueError('augmented replay centers differ; origin not recovered; explicit independent rebuild available (no approximate matching)')
                delta[l][take],valid[l][take]=nm.deltas(full,pos[take])
                part_arrays[f'delta{l}']=delta[l][take];part_arrays[f'valid{l}']=valid[l][take]
            st.put(part,part_arrays,partmeta)
            if (number+1)%10==0 or number+1==455:print(f'[same-origin Source recovery] {number+1}/455 augmented grids; exact centers',flush=True)
            from .runner import enforce_time_budget
            enforce_time_budget(a,'prepare-source',number+1,455)
    arrays={}
    for l in LAYERS:
        images,pos=origins[l];arrays.update({f'L{l}':bank[f'L{l}'],f'delta{l}':delta[l],f'valid{l}':valid[l],f'image{l}':images,f'position{l}':pos,f'selected{l}':indices[l]})
        widths=np.asarray([rows[i]['grid'][1] for i in images],np.int64)
        arrays.update({f'row{l}':pos//widths,f'column{l}':pos%widths,f'bank_row{l}':np.arange(len(pos),dtype=np.int64)})
    st.put(source_path(a),arrays,dict(format='augmented_Source_neighbors_v1',centers_verified=True,catalog_source=ip.catalog(a)['source'],
        original_bank_seal=br['seal'],independent_rebuild=False,index_file=str(ix),origins=rows,origin='exact_selected_index_rng_replay',tolerance=0.))
    st.write_csv(a.run_root/'source_center_verification.csv',checks)
    return dict(status='Source_origin_recovered',augmented_images=455,independent_rebuild=False)


def rebuild_source(a):
    """Same original builder/calls/sampler. Observe selected rows before spool deletion.

    The function's globals are bound locally; no parent module monkeypatch or
    source change. No extra NumPy RNG draw or sampler invocation is introduced.
    """
    import torch
    from external_baselines.superadd_external import memory_variants as original
    from DINOv3.relation_reliability.superadd_host import image_tensor
    from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.fresh_inputs import Data
    c=ip.catalog(a);rows=pv.layout(c);m=model(a);recorder=pv.AugmentationRecorder(m.augmented_preprocessing)
    if (recorder.min_brightness_factor,recorder.max_brightness_factor)!=(.8,1.2):raise ValueError('original Source brightness augmentation must remain0.8..1.2')
    m.augmented_preprocessing=recorder;selected={};cursor=[0]
    class Images:
        def __len__(self):return len(c['source'])
        def __getitem__(self,i):
            if i>=len(self):raise IndexError(i)
            with Data(a.data) as data:return image_tensor(data.image(c['source'][i]))
    def observe(index,layer,v):
        if layer==LAYERS[0]:
            row=rows[cursor[0]]
            if index!=row['source_index'] or list(v.shape[-3:-1])!=row['grid']:raise ValueError('actual builder grid/order differs')
            row.update(recorder.records[-1]);cursor[0]+=1
            if cursor[0]%10==0:print(f'[same-origin explicit Source build] {cursor[0]}/455',flush=True)
        return v
    sampler=original._official_subsample_values_with_indices
    def selected_observer(features,*args,**kwargs):
        values,indices=sampler(features,*args,**kwargs)
        if not np.array_equal(values,features[indices]):raise ValueError('original sampler selected-center identity differs')
        layer=LAYERS[len(selected)]
        delta,valid,image,position=pv.gather_selected(features,indices,rows)
        selected[layer]=dict(delta=delta,valid=valid,image=image,position=position,selected=indices)
        return values,indices
    import inspect
    # The public builder is torch.inference_mode-decorated. Bind the actual
    # implementation, not the decorator's closure (which would call the old
    # globals and bypass the selected-row observer).
    builder=inspect.unwrap(original.build_official_memory_spooled)
    bound=FunctionType(builder.__code__,dict(builder.__globals__,_official_subsample_values_with_indices=selected_observer),builder.__name__,builder.__defaults__,builder.__closure__)
    bound.__kwdefaults__=builder.__kwdefaults__
    try:
        with torch.inference_mode():result=bound(m,Images(),device=torch.device(a.device),scratch_dir=a.build_scratch,
            return_selected_indices=True,storage_backend='disk',feature_transform=observe)
    finally:m.augmented_preprocessing=recorder.delegate
    arrays={};differences=[];previous=ip.bank_path(a)
    old=st.get(previous,tuple(f'L{l}' for l in LAYERS))[0] if previous and (previous/'manifest.json').exists() else None
    for l in LAYERS:
        arrays[f'L{l}']=result.banks[l].cpu().numpy()
        if old is not None:differences.append(dict(layer=l,**pv.check_centers(arrays[f'L{l}'],old[f'L{l}'])))
        for name,value in selected[l].items():arrays[f'{name}{l}']=value
        positions=selected[l]['position'];widths=np.asarray([rows[i]['grid'][1] for i in selected[l]['image']],np.int64)
        arrays.update({f'row{l}':positions//widths,f'column{l}':positions%widths,f'bank_row{l}':np.arange(len(positions),dtype=np.int64)})
    st.put(source_path(a),arrays,dict(format='augmented_Source_neighbors_v1',centers_verified=True,
        catalog_source=c['source'],original_bank_seal=None,independent_rebuild=True,origins=rows,
        origin='independent_original_builder_selected_spool_rows',builder=result.metadata,tolerance=0.))
    st.write_json(a.run_root/'Source_rebuild_audit.json',dict(independent_rebuild=True,shared_by_all_branches=True,baseline_recomputed=True,old_bank_comparison=differences,
        sampler_unchanged=True,augmentation_draw_order_unchanged=True,selected_indices_preserved=True))
    from .runner import enforce_time_budget
    enforce_time_budget(a,'prepare-source',1,1)
    return dict(status='Source_rebuilt_with_provenance',independent_rebuild=True,baseline_reuse=False,
        candidate_peak_bytes=result.metadata['candidate_scratch_peak_bytes'])


def prepare_target(a):
    c=ip.catalog(a);records=[]
    for row in c['supports']['background']:
        path=a.cache_root/'target_neighbors'/f"{row['support_index']:02d}"
        if not (path/'manifest.json').exists():
            q,m=features(a,row);arrays={}
            for l in LAYERS:
                full=nm.physical(q[l],m['positions'],m['grid']);centers=full.reshape(-1,1280);pos=np.arange(len(centers),dtype=np.int64)
                delta,valid=nm.deltas(full,pos);arrays.update({f'L{l}':centers,f'delta{l}':delta,f'valid{l}':valid})
            st.put(path,arrays,dict(image=row,grid=m['grid'],order='physical row-major'))
        _,r=st.get(path)
        if r['metadata']['image']!=row:raise ValueError('Target support order/identity differs')
        records.append(dict(support_index=row['support_index'],seal=r['seal']))
    return st.sealed_json(a.run_root/'target_ready.json',dict(identity=ip.identity(a),records=records))


def references(a,source_only=False):
    """Four layers resident for all images, avoiding repeated HDD reads/uploads."""
    source,sr=source_bundle(a);targets=[]
    if not source_only:
        for row in ip.catalog(a)['supports']['background']:
            x,r=st.get(a.cache_root/'target_neighbors'/f"{row['support_index']:02d}")
            if r['metadata']['image']!=row:raise ValueError('Target provenance differs')
            targets.append(x)
    searches={}
    for l in LAYERS:
        parts=[source]+targets;offsets=np.cumsum([0]+[len(x[f'L{l}']) for x in parts]).tolist()
        searches[l]=nm.Search(np.concatenate([x[f'L{l}'] for x in parts]),np.concatenate([x[f'delta{l}'] for x in parts]),
            np.concatenate([x[f'valid{l}'] for x in parts]),offsets,a.device,a.query_chunk,a.reference_chunk)
        searches[l].cpu_parts=[x[f'L{l}'] for x in parts]
    return searches,sr


def observations(q,meta,searches):
    out={}
    for l in LAYERS:
        full=nm.physical(q[l],meta['positions'],meta['grid']);dq,valid=nm.deltas(full,meta['positions'])
        search=searches[l];absolute,ids,part,part_ids=search.top(q[l]);nb,rel=search.components(q[l],dq,ids)
        out[l]=dict(absolute=absolute,indices=ids,nb=nb,rel=rel,valid=valid,
            candidate_valid=search.valid[search.torch.tensor(ids,device=a_device(search))].cpu().numpy(),
            per_distance=part,per_index=part_ids)
    return out


def a_device(search):return search.device
