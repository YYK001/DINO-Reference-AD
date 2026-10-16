"""Reuse frozen projection/Source bank; prepare-target is NumPy only."""
from __future__ import annotations
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image

from .inputs import TOOL,NEW,replay,parent_inputs,verify_files
from DINOv3.MADEqual.aebad_s_sparc_p0 import runtime as parent_runtime,analysis as parent_analysis
from DINOv3.MADEqual.aebad_s_sparc_p0.projection import LAYERS,DIMENSION,load_projection,physical_positions
from DINOv3.MADEqual.aebad_s_union_support_loo.runner import audit


def ordered_parallel(function,items,workers):
    values=list(items)
    if workers<1:raise ValueError('CPU workers must be positive')
    if workers==1:return [function(v) for v in values]
    with ThreadPoolExecutor(max_workers=min(workers,len(values))) as executor:
        return list(executor.map(function,values))


def projections_for(data):
    root=Path(data['paths']['parent']);grid=data['frozen']['grid']
    return {l:load_projection(root/'projection'/f'L{l}',grid) for l in LAYERS}


def evaluation_device(data,device_name):
    """Select CUDA/precision for ADEval without constructing any backbone."""
    import torch
    device=torch.device(device_name)
    if device.type!='cuda' or not torch.cuda.is_available():raise RuntimeError('formal ADEval requires CUDA')
    device=torch.device('cuda',torch.cuda.current_device() if device.index is None else device.index)
    torch.cuda.set_device(device)
    precision=data['parent']['frozen']['precision']
    torch.set_float32_matmul_precision(precision['float32_matmul_precision'])
    torch.backends.cuda.matmul.allow_tf32=precision['cuda_matmul_allow_tf32']
    torch.backends.cudnn.allow_tf32=precision['cudnn_allow_tf32']
    torch.cuda.reset_peak_memory_stats(device)
    return device


def corrected_support(path,layer,projection,grid):
    with np.load(path,allow_pickle=False) as f:
        if f['grid'].tolist()!=list(grid):raise ValueError('support physical grid differs')
        value=f[f'layer_{layer}'].copy()
    positions=np.arange(int(np.prod(grid)),dtype=np.int64)
    corrected=projection.correct(value,positions,grid)
    # Verify actual float32 results, not an analytic shortcut replacing them.
    means=np.asarray(projection.mu);rounded=means.astype(np.float32)
    error=float(np.max(np.abs(corrected.astype(np.float64)-means)))
    rounded_error=float(np.max(np.abs(corrected.astype(np.float64)-rounded.astype(np.float64))))
    verified=bool(np.allclose(corrected,rounded,atol=1e-7,rtol=1e-6))
    if not verified:raise ValueError('corrected support does not collapse to frozen same-position mean')
    return corrected,dict(max_abs_error_vs_float64_mu=error,max_abs_error_vs_float32_mu=rounded_error,
                          verified=verified,atol=1e-7,rtol=1e-6,formal_memory='each_original_support_corrected_separately')


def prepare_target(data):
    root=Path(data['paths']['output']);start=time.perf_counter();grid=tuple(data['frozen']['grid'])
    verify_files(data['required_files']);rows=[];artifacts=[]
    n=int(np.prod(grid));workers=data['frozen']['cpu_workers'];supports=data['frozen']['supports']
    if parent_inputs.free_space(root)<data['resources']['prepare_disk_required_bytes']:
        raise RuntimeError('disk no longer sufficient for new Target/raw assets')
    ram=parent_inputs.available_ram()
    if ram is not None and ram<data['resources']['minimum_available_ram_bytes']:raise RuntimeError('insufficient available RAM')
    projections=projections_for(data)
    correction_seconds=0.;writing_seconds=0.
    try:
        for layer in LAYERS:
            t=time.perf_counter()
            results=ordered_parallel(lambda s:corrected_support(Path(s['path']),layer,projections[layer],grid),supports,workers)
            correction_seconds+=time.perf_counter()-t;t=time.perf_counter()
            directory=root/'target_memory';directory.mkdir(exist_ok=True)
            file=directory/f'L{layer}.npy'
            if file.exists():raise ValueError('Target memory already exists; no re-approval')
            memory=np.lib.format.open_memmap(file,mode='w+',dtype=np.float32,shape=(8*n,DIMENSION))
            for index,(corrected,stats) in enumerate(results):
                memory[index*n:(index+1)*n]=corrected
                rows.append(dict(layer=layer,support_index=index+1,relative_path=supports[index]['relative_path'],
                    original_feature_sha256=supports[index]['feature_sha256'],**stats))
            memory.flush();memory._mmap.close();del memory,results
            artifacts.append(file);writing_seconds+=time.perf_counter()-t
            print(f'[SPARC UNION Target] L{layer} corrected all 8 supports',flush=True)
        # Mapping is exact concatenation order; retains every original support row.
        mapping=root/'target_memory'/'support_mapping.npz'
        parent_runtime.save_npz(mapping,support_index=np.repeat(np.arange(1,9,dtype=np.int16),n),
            physical_position=np.tile(np.arange(n,dtype=np.int64),8),
            patch_y=np.tile(np.arange(n,dtype=np.int32)//grid[1],8),
            patch_x=np.tile(np.arange(n,dtype=np.int32)%grid[1],8))
        artifacts.append(mapping)
        replay.write_csv(root/'support_collapse_verification.csv',rows);artifacts.append(root/'support_collapse_verification.csv')
        replay.atomic_json(root/'target_provenance.json',dict(tool=TOOL,identity=data['identity'],
            parent_identity=data['frozen']['parent_identity'],supports=supports,grid=list(grid),layers=list(LAYERS),
            rows_per_layer=8*n,compressed=False,deduplicated=False,
            tie_note='Top-1 support ID among duplicate/near-duplicate references is not a unique semantic contribution'))
        artifacts.append(root/'target_provenance.json');verify_files(data['required_files'])
        state=dict(tool=TOOL,status='target_prepared',identity=data['identity'],
            inputs_sha256=replay.sha256(root/'inputs.json'),artifacts=parent_inputs.file_records(root,artifacts),
            parent_identity=data['frozen']['parent_identity'],backbone_loaded=False,Source_extracted=False,
            Source_bank_rebuilt=False,projection_fitted=False,MAD_fitted=False,GT_decoded=False,
            cpu_workers=workers,seconds=time.perf_counter()-start,
            timing=dict(support_read_and_correction=correction_seconds,target_memory_writing=writing_seconds))
        replay.atomic_json(root/'prepare_target_status.json',state);return state
    finally:
        for projection in projections.values():projection.close()


def exact_pair(corrected,source,target,device,*,timing=None):
    """One already-corrected query tensor, two unchanged Exact calls; /1280 once."""
    import torch
    from DINOv3.relation_reliability.superadd_host import _exact_top1_chunked
    if (corrected.ndim!=2 or corrected.shape[1]!=DIMENSION or corrected.dtype!=np.float32
            or not np.isfinite(corrected).all()):raise ValueError('Exact input must be finite already-corrected float32 [N,1280]')
    q=torch.from_numpy(corrected).to(device)
    results=[]
    for name,bank in (('source_verification_retrieval',source),('target_retrieval',target)):
        t=time.perf_counter();dist,indices=_exact_top1_chunked(q,bank)
        d=dist.numpy().astype(np.float32)/float(DIMENSION);idx=indices.numpy().astype(np.int64)
        if (d.shape!=(len(corrected),) or idx.shape!=d.shape or not np.isfinite(d).all()
                or (d<0).any() or (idx<0).any() or (idx>=bank.shape[0]).any()):
            raise ValueError('invalid Exact distance/index result')
        if timing is not None:timing[name]+=time.perf_counter()-t
        results.append((d,idx))
    (ds,si),(dt,ti)=results
    return ds,dt,np.minimum(ds,dt),si,ti


def observe(model,image,device,projections,targets,workers,timing):
    import torch
    from DINOv3.relation_reliability.superadd_host import image_tensor,_extract_processed_batch
    with torch.inference_mode():
        t=time.perf_counter();processed=model.preprocessing(image_tensor(image).to(device)[None])
        pending=_extract_processed_batch(model,processed,device=device,query_masks=[None],include_identity_descriptors=False)
        if len(pending)!=1:raise ValueError('one complete query extraction required')
        item=pending[0];grid=tuple(item.grid);positions=physical_positions(item.positions,grid)
        torch.cuda.synchronize(device);timing['query_backbone']+=time.perf_counter()-t
        t=time.perf_counter()
        queries={l:item.queries[l].detach().float().cpu().numpy() for l in LAYERS}
        corrected=ordered_parallel(lambda l:projections[l].correct(queries[l],positions,grid),LAYERS,workers)
        timing['query_correction']+=time.perf_counter()-t
        values=[exact_pair(q,model.prototype_embeddings[l],targets[l],device,timing=timing)
                for l,q in zip(LAYERS,corrected,strict=True)]
        return dict(source_distance=np.stack([v[0] for v in values]),target_distance=np.stack([v[1] for v in values]),
            union_distance=np.stack([v[2] for v in values]),source_index=np.stack([v[3] for v in values]),
            target_index=np.stack([v[4] for v in values]),positions=positions.copy(),grid=np.asarray(grid,np.int32),
            processed_shape=np.asarray(processed.shape[-2:],np.int32))


def validate_arrays(arrays,grid):
    n=int(np.prod(grid));physical_positions(arrays['positions'],grid)
    if arrays['grid'].tolist()!=list(grid):raise ValueError('new raw grid mismatch')
    for key in ('source_distance','target_distance','union_distance'):
        x=arrays[key]
        if x.shape!=(4,n) or x.dtype!=np.float32 or not np.isfinite(x).all() or (x<0).any():raise ValueError(f'invalid {key}')
    if not np.array_equal(arrays['union_distance'],np.minimum(arrays['source_distance'],arrays['target_distance'])):
        raise ValueError('union must be patch-wise min of this same query Source and Target')
    for key,bound in (('source_index',100000),('target_index',8*n)):
        x=arrays[key]
        if x.shape!=(4,n) or x.dtype!=np.int64 or (x<0).any() or (x>=bound).any():raise ValueError(f'invalid {key}')
    if arrays['processed_shape'].shape!=(2,) or arrays['processed_shape'].dtype!=np.int32 or (arrays['processed_shape']<=0).any():
        raise ValueError('invalid processed shape')


def source_checks(sample,observed,historical,source_map,history_score,rows):
    okay=audit(rows,sample['sample_id'],'historical_SPARC_Source_full_grid',observed['source_distance'],historical)
    okay=audit(rows,sample['sample_id'],'historical_SPARC_Source_image_score',float(source_map.max()),history_score,score=True) and okay
    if not okay:raise ValueError(f"historical Source distance/score reproduction failed: {sample['sample_id']}; tolerances unchanged")


def infer(data,prepared):
    import torch
    from .inputs import history_tables
    from .analysis import load_parent,backends_for,maps_for
    from DINOv3.MADEqual.aebad_s_target_support_trace.analysis import image_geometry
    root=Path(data['paths']['output']);base=Path(data['paths']['parent']);start=time.perf_counter();timing=defaultdict(float)
    if (root/'infer_status.json').exists() or (root/'raw').exists():raise ValueError('infer refuses existing/partial caches')
    verify_files(data['required_files']);parent_inputs.verify_artifacts(root,prepared['artifacts'])
    model,device=parent_runtime.load_host(data['parent'],data['device'])
    parent_runtime.load_memory(model,base/'sparc_source_memory.pt',data['frozen']['parent_identity'],device)
    projections=projections_for(data);targets={};checks=[];records=[];per_image=[]
    cap,cap_state=parent_inputs.read_capture(Path(data['paths']['capture']))
    old={r['sample_id']:r for r in cap_state['records']}
    old_infer=json.loads((base/'infer_status.json').read_text(encoding='utf-8'));parent_raw={r['sample_id']:r for r in old_infer['records']}
    history,_=history_tables(base,data['parent']['samples']);backends=backends_for(data)
    try:
        for l in LAYERS:
            value=np.load(root/'target_memory'/f'L{l}.npy',mmap_mode='r',allow_pickle=False)
            if value.dtype!=np.float32 or value.shape!=(8*int(np.prod(data['frozen']['grid'])),DIMENSION):raise ValueError('corrected Target shape/dtype differs')
            targets[l]=torch.from_numpy(np.array(value,copy=True)).to(device);value._mmap.close()
        for i,s in enumerate(data['parent']['samples']):
            t=time.perf_counter();before=dict(timing);image_path=Path(data['paths']['data'])/s['relative_path']
            raws,positions,meta=load_parent(data,s,cap,old[s['sample_id']],parent_raw[s['sample_id']])
            if replay.sha256(image_path)!=meta['image_sha256']:raise ValueError('query RGB bytes differ from parent')
            with Image.open(image_path) as image:
                observed=observe(model,image.convert('RGB'),device,projections,targets,data['frozen']['cpu_workers'],timing)
                geometry=image_geometry(model,image.height,image.width,tuple(int(v) for v in observed['grid']),
                                        tuple(int(v) for v in observed['processed_shape']))
                shape=[image.height//4,image.width//4]
            validate_arrays(observed,tuple(data['frozen']['grid']))
            if (geometry!=meta['geometry'] or shape!=meta['output_shape'] or not np.array_equal(positions,observed['positions'])
                    or observed['processed_shape'].tolist()!=[geometry['processed_height'],geometry['processed_width']]):
                raise ValueError('query complete stitching/processed geometry differs from historical parent')
            current=maps_for({'SPARC_SOURCE_K8':observed['source_distance'],NEW:observed['union_distance']},positions,meta,backends)
            source_checks(s,observed,raws['SPARC_SOURCE_K8'],current['SPARC_SOURCE_K8'],
                          float(history['SPARC_SOURCE_K8'][s['sample_id']]['image_score']),checks)
            mono=bool(np.all(current[NEW]<=current['SPARC_SOURCE_K8']+1e-5))
            checks.append(dict(sample_id=s['sample_id'],check_kind='same_query_same_MAD_map_monotonicity',verified=mono,score_atol=1e-5))
            if not mono:raise ValueError('same-MAD map monotonicity failed')
            metadata=dict(tool=TOOL,identity=data['identity'],parent_identity=data['frozen']['parent_identity'],
                sample_id=s['sample_id'],relative_path=s['relative_path'],image_sha256=meta['image_sha256'],grid=meta['grid'],
                output_shape=shape,geometry=geometry,parent_raw_sha256=parent_raw[s['sample_id']]['sha256'],
                target_prepare_sha256=replay.sha256(root/'prepare_target_status.json'),branch=NEW,distance_divisor=1280,
                historical_source_reproduced=True,query_forward_calls=1,query_corrections_per_layer=1,
                target_support_index_meaning='index//grid_cells+1; ties do not establish unique support semantics')
            file=root/'raw'/f"{s['sample_id']}.npz"
            parent_runtime.save_npz(file,metadata=np.asarray(json.dumps(metadata,sort_keys=True)),**observed)
            records.append(dict(sample_id=s['sample_id'],**parent_inputs.file_records(root,[file])[0]))
            per_image.append(dict(sample_id=s['sample_id'],**{k:timing[k]-before.get(k,0.) for k in
                ('query_backbone','query_correction','source_verification_retrieval','target_retrieval')},
                total_seconds=time.perf_counter()-t))
            print(f'[SPARC UNION infer] {i+1}/297 {s["sample_id"]}',flush=True)
        verify_files(data['required_files']);parent_inputs.verify_artifacts(root,prepared['artifacts']);parent_inputs.verify_artifacts(root,records)
        state=dict(tool=TOOL,status='predictions_frozen',identity=data['identity'],parent_identity=data['frozen']['parent_identity'],
            predicted_images=297,records=records,inputs_sha256=replay.sha256(root/'inputs.json'),
            prepare_target_sha256=replay.sha256(root/'prepare_target_status.json'),historical_source_reproduced=True,
            query_forward_calls=297,Source_builder_called=False,projection_fitted=False,MAD_fitted=False,GT_decoded=False,
            seconds=time.perf_counter()-start,timing=dict(timing))
        replay.write_csv(root/'infer_verification.csv',checks);replay.write_csv(root/'timing_per_image.csv',per_image)
        state['verification_sha256']=replay.sha256(root/'infer_verification.csv')
        replay.atomic_json(root/'infer_status.json',state);return state
    finally:
        # On failure retain diagnostics, but never create the frozen barrier.
        if not (root/'infer_status.json').exists():
            replay.write_csv(root/'infer_verification.csv',checks);replay.write_csv(root/'timing_per_image.csv',per_image)
        for p in projections.values():p.close()
