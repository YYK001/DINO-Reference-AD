"""Explicit CUDA stages only. Imports of the real host remain lazy."""
from __future__ import annotations

import json
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image

from .inputs import replay, TOOL, file_records, verify_files, free_space, available_ram
from .projection import LAYERS,DIMENSION,fit_layer,load_projection,physical_positions


def save_npz(path, **values):
    if path.exists() or path.with_suffix('.npz.tmp').exists(): raise ValueError(f'output already exists: {path}')
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.npz.tmp')
    with tmp.open('xb') as f: np.savez_compressed(f,**values)
    tmp.replace(path)


def load_host(inputs,device_name):
    import torch
    from unittest.mock import patch
    from DINOv3.relation_reliability.superadd_host import install_import_paths,construct_exact_host
    from DINOv3.MADEqual.aebad_s_target_fewshot.runner import _seed_everything
    paths={k:Path(v) for k,v in inputs['paths'].items()}
    device=torch.device(device_name)
    if device.type!='cuda' or not torch.cuda.is_available(): raise RuntimeError('explicit prepare/infer requires CUDA')
    device=torch.device('cuda',torch.cuda.current_device() if device.index is None else device.index)
    # The shared constructor uses bare 'cuda', including preprocessing tensors.
    torch.cuda.set_device(device)
    _seed_everything()
    precision=inputs['frozen']['precision']
    torch.set_float32_matmul_precision(precision['float32_matmul_precision'])
    torch.backends.cuda.matmul.allow_tf32=precision['cuda_matmul_allow_tf32']
    torch.backends.cudnn.allow_tf32=precision['cudnn_allow_tf32']
    torch.cuda.reset_peak_memory_stats(device)
    install_import_paths(paths['project'],paths['official'],paths['dinov3'])
    import industrial.model as official_model
    with patch.object(official_model,'get_weights_path',lambda name:paths['checkpoint']):
        model,config=construct_exact_host(paths['official'],paths['dinov3'],paths['checkpoint'],replay.CHECKPOINT_SHA256)
    model.backbone.dino.eval()
    model.backbone.dino.requires_grad_(False)
    if (config!=inputs['frozen']['host_config'] or list(model.layers)!=list(LAYERS)
            or model.preprocessing.resize_factor!=.625 or model.preprocessing.min_brightness_factor!=1
            or model.preprocessing.max_brightness_factor!=1 or model.evaluation_downscale!=4
            or model.patch_exec.model_patch_size!=16 or model.backbone.dino.training
            or any(p.requires_grad for p in model.backbone.dino.parameters())):
        raise ValueError('constructed host differs from frozen extraction/scoring protocol')
    return model,device


def correct_embedding(projection,embedding):
    if embedding.ndim!=4 or embedding.shape[0]!=1 or embedding.shape[-1]!=DIMENSION:
        raise ValueError('builder callback needs one complete native stitched [1,H,W,1280] map')
    grid=tuple(embedding.shape[1:3]);p=np.arange(int(np.prod(grid)),dtype=np.int64)
    return projection.correct(embedding.reshape(-1,DIMENSION),p,grid).reshape(embedding.shape)


class LazySourceTensors:
    """The builder gets all 521 ordered items and loads ONLY its 455 selected RGBs."""
    def __init__(self,records): self.records=records
    def __len__(self): return len(self.records)
    def __getitem__(self,index):
        if not 0<=index<len(self): raise IndexError(index)
        from DINOv3.relation_reliability.superadd_host import image_tensor
        with Image.open(self.records[index].path) as image: return image_tensor(image.convert('RGB'))


class TimedPatchExec:
    def __init__(self,delegate,timing): self.delegate=delegate;self.timing=timing
    def __getattr__(self,name): return getattr(self.delegate,name)
    def __call__(self,*args,**kwargs):
        start=time.perf_counter();result=self.delegate(*args,**kwargs)
        self.timing['source_feature_extraction']+=time.perf_counter()-start
        return result


def bank_provenance(selected_indices,entries):
    """Map actual sampler candidate indices to the Source image and native cell."""
    starts=np.array([r['candidate_start'] for r in entries],np.int64)
    idx=np.asarray(selected_indices,np.int64)
    if idx.ndim!=1 or (idx<0).any() or (idx>=entries[-1]['candidate_stop']).any():
        raise ValueError('selected bank candidate index out of bounds')
    where=np.searchsorted(starts,idx,side='right')-1
    source=np.array([r['source_index'] for r in entries],np.int64)[where]
    if (source%8==0).any(): raise ValueError('calibration entered bank candidates')
    return dict(candidate_index=idx,source_index=source,physical_position=idx-starts[where])


def exact_corrected_layer(query,positions,grid,projection,bank,device,*,timing=None):
    import torch
    from DINOv3.relation_reliability.superadd_host import _exact_top1_chunked
    start=time.perf_counter()
    corrected=projection.correct(query,positions,grid)
    tensor=torch.from_numpy(corrected).to(device)
    if timing is not None:
        torch.cuda.synchronize(device)
        timing['query_correction']+=time.perf_counter()-start
    start=time.perf_counter()
    distances,indices=_exact_top1_chunked(tensor,bank)
    if timing is not None: timing['query_retrieval']+=time.perf_counter()-start
    # This is the ONLY feature-dimension distance division in this path.
    d=distances.numpy().astype(np.float32)/float(DIMENSION)
    index=indices.numpy().astype(np.int64)
    if (d.shape!=(len(positions),) or index.shape!=d.shape or not np.isfinite(d).all()
            or (d<0).any() or (index<0).any() or (index>=bank.shape[0]).any()):
        raise ValueError('invalid corrected Source Exact 1-NN result')
    return d,index


def observe_corrected(model,image,device,projections,timing):
    import torch
    from DINOv3.relation_reliability.superadd_host import image_tensor,_extract_processed_batch
    with torch.inference_mode():
        start=time.perf_counter()
        tensor=image_tensor(image).to(device)[None];processed=model.preprocessing(tensor)
        pending=_extract_processed_batch(model,processed,device=device,query_masks=[None],include_identity_descriptors=False)
        if len(pending)!=1: raise ValueError('one complete query required')
        item=pending[0];grid=tuple(item.grid);positions=physical_positions(item.positions,grid)
        torch.cuda.synchronize(device)
        timing['query_backbone']+=time.perf_counter()-start
        raw=[];indices=[]
        for layer in LAYERS:
            start=time.perf_counter()
            q=item.queries[layer].detach().float().cpu().numpy()
            timing['query_correction']+=time.perf_counter()-start
            d,index=exact_corrected_layer(q,positions,grid,projections[layer],model.prototype_embeddings[layer],device,timing=timing)
            raw.append(d);indices.append(index)
            del q
        return dict(distance=np.stack(raw),nearest_index=np.stack(indices),positions=positions.copy(),
                    grid=np.array(grid,np.int32),processed_shape=np.array(processed.shape[-2:],np.int32))


def prepare(inputs,device_name):
    import torch
    from DINOv3.MADEqual.robustad_target_fewshot_p0.support import extract_or_load_support
    from external_baselines.superadd_external.memory_variants import build_official_memory_spooled
    from DINOv3.relation_reliability.calibration import fit_calibrators
    from .inputs import source_records
    from DINOv3.relation_reliability.datasets import Record
    from DINOv3.MADEqual.aebad_s_target_support_trace.analysis import image_geometry
    from DINOv3.relation_reliability.superadd_host import preprocessed_spatial_shape
    start_all=time.perf_counter();timing=defaultdict(float)
    paths={k:Path(v) for k,v in inputs['paths'].items()};out=paths['output'];identity=inputs['identity']
    needed=inputs['resource_estimate']['prepare_disk_required_bytes']
    if free_space(out)<needed: raise RuntimeError('disk no longer sufficient; original candidates/budget retained')
    ram=available_ram()
    if ram is None and inputs['resource_estimate']['candidate_resident_bytes']:
        raise RuntimeError('mixed candidate storage requires a known available RAM budget')
    if ram is not None and ram<inputs['resource_estimate']['minimum_ram_headroom_bytes']:
        raise RuntimeError('available RAM no longer sufficient; no precision/budget reduction')
    model,device=load_host(inputs,device_name)
    # Restore the host's Source RNG state after support work. No SVD or cache
    # operation is allowed to alter the original builder's brightness/sampler RNG.
    source_numpy_state=np.random.get_state()
    grid=tuple(inputs['frozen']['grid']);support_paths=[];support_proof=[]
    start=time.perf_counter()
    for row,feature in zip(inputs['frozen']['supports'],inputs['support_features'],strict=True):
        path=Path(feature['path'])
        if feature['exists']:
            if replay.sha256(path)!=row['feature_sha256']: raise ValueError('original support modified')
            origin='original_replay_feature'
        else:
            path=out/'supplemented_support'/f"{row['support_index']:02d}.npz"
            record=Record(path=str(paths['data']/row['relative_path']),label=0,mask_path=None,
                          category='AeBAD-S',dataset='aebad_s',domain='background',shift='good')
            with torch.inference_mode():
                extract_or_load_support(path,record,dataset_root=paths['data'],model=model,device=device,protocol_id=replay.PROTOCOL_ID)
            origin='prepare_only_frozen_image_extraction'
        replay.check_support_header(path,row['relative_path']);support_paths.append(path)
        support_proof.append(dict(**row,path=str(path),current_sha256=replay.sha256(path),origin=origin,
                                  frozen_checkpoint_sha256=replay.CHECKPOINT_SHA256))
    timing['support_processing']+=time.perf_counter()-start
    replay.atomic_json(out/'support_provenance.json',dict(tool=TOOL,identity=identity,supports=support_proof))
    ranks=[];projections={}
    for layer in LAYERS:
        start=time.perf_counter();banks=[]
        for path in support_paths:
            with np.load(path,allow_pickle=False) as f:
                if tuple(f['grid'])!=grid: raise ValueError('support native grid mismatch')
                banks.append(f[f'layer_{layer}'].copy())
        timing['support_processing']+=time.perf_counter()-start
        start=time.perf_counter()
        stats=fit_layer(banks,grid,out/'projection'/f'L{layer}')
        timing['projection_fit']+=time.perf_counter()-start
        ranks.append(dict(layer=layer,**stats));del banks
        projections[layer]=load_projection(out/'projection'/f'L{layer}',grid)
        print(f'[SPARC prepare] projection L{layer}',flush=True)
    replay.atomic_json(out/'projection_state.json',dict(tool=TOOL,identity=identity,layers=ranks))
    replay.write_csv(out/'projection_verification.csv',[dict(layer=r['layer'],verified=True,
        check_kind='orthogonality_and_support_collapse',rank_rule=r['rank_rule'],
        max_orthogonality_error=r['max_orthogonality_error'],max_support_collapse_abs_error=r['max_support_collapse_abs_error']) for r in ranks])
    records=source_records(paths['data'])
    replay.write_csv(out/'source_manifest.csv',inputs['frozen']['source_rows'])
    entries={l:[] for l in LAYERS};counts={l:0 for l in LAYERS}
    def transform(source_index,layer,embedding):
        if source_index%8==0: raise ValueError('builder invoked calibration transform')
        row=inputs['frozen']['source_rows'][source_index]
        processed_shape=preprocessed_spatial_shape(model,row['image_height'],row['image_width'])
        geometry=image_geometry(model,row['image_height'],row['image_width'],tuple(embedding.shape[1:3]),processed_shape)
        reference=inputs['frozen']['supports'][0]['geometry']
        if any(geometry[k]!=reference[k] for k in ('processed_height','processed_width','axes','model_patch_size')):
            raise ValueError('Source stitching/processed shape incompatible with position-specific support projector')
        t=time.perf_counter();corrected=correct_embedding(projections[layer],embedding)
        timing['source_correction']+=time.perf_counter()-t
        n=int(np.prod(embedding.shape[1:3]))
        entries[layer].append(dict(layer=layer,source_index=source_index,
            relative_path=inputs['frozen']['source_rows'][source_index]['relative_path'],grid=list(grid),
            candidate_start=counts[layer],candidate_stop=counts[layer]+n))
        counts[layer]+=n
        if layer==LAYERS[-1]: print(f'[SPARC prepare] Source {source_index+1}/521 memory-fit',flush=True)
        return corrected
    original_exec=model.patch_exec;model.patch_exec=TimedPatchExec(original_exec,timing)
    np.random.set_state(source_numpy_state)
    resources=inputs['resource_estimate']
    storage={k:resources[k] for k in ('candidate_storage','ram_layers','disk_layers')}
    if storage!=inputs['frozen']['candidate_storage']: raise ValueError('candidate storage differs from frozen plan')
    ram=available_ram()
    if resources['candidate_resident_bytes'] and (ram is None or ram<resources['minimum_ram_headroom_bytes']):
        raise RuntimeError('available RAM no longer sufficient for mixed candidates and working headroom')
    if free_space(out)<resources['candidate_scratch_bytes']+(4<<30):
        raise RuntimeError('disk no longer sufficient for candidate scratch and headroom')
    t=time.perf_counter()
    try:
        with torch.inference_mode():
            build=build_official_memory_spooled(model,LazySourceTensors(records),device=device,
                scratch_dir=out/'memory_scratch',return_selected_indices=True,storage_backend=resources['candidate_storage'],
                ram_layers=resources['ram_layers'],
                expected_candidate_rows=inputs['resource_estimate']['candidate_rows_per_layer'],
                estimated_peak_bytes=inputs['resource_estimate']['candidate_float32_bytes'],feature_transform=transform)
    finally: model.patch_exec=original_exec
    timing['memory_build_inclusive']+=time.perf_counter()-t
    timing['memory_build_excluding_backbone_and_correction']=max(0.,timing['memory_build_inclusive']-timing['source_feature_extraction']-timing['source_correction'])
    if build.selected_candidate_indices is None: raise ValueError('actual bank source sidecar required')
    if any(len(entries[l])!=455 or counts[l]!=inputs['resource_estimate']['candidate_rows_per_layer'] for l in LAYERS):
        raise ValueError('actual complete candidate scope differs from preflight')
    model.prototype_embeddings=build.banks
    for l in LAYERS:
        if tuple(build.banks[l].shape)!=(100000,DIMENSION): raise ValueError('SPARC Source bank budget changed')
        save_npz(out/'bank_provenance'/f'L{l}.npz',**bank_provenance(build.selected_candidate_indices[l],entries[l]))
    replay.atomic_json(out/'candidate_sources.json',dict(tool=TOOL,identity=identity,layers=entries,
                                                      target_memory_appended=False,builder_metadata=build.metadata))
    t=time.perf_counter()
    torch.save(dict(tool=TOOL,identity=identity,source_count=521,target_memory_appended=False,
                    **{f'bank_{l}':build.banks[l].detach().cpu() for l in LAYERS}),out/'sparc_source_memory.pt')
    timing['memory_saving']+=time.perf_counter()-t
    del build
    calibration=[r for i,r in enumerate(records) if i%8==0];pooled={l:[] for l in LAYERS}
    mad_timing=defaultdict(float)
    t=time.perf_counter()
    for i,record in enumerate(calibration):
        with Image.open(record.path) as image: observed=observe_corrected(model,image.convert('RGB'),device,projections,mad_timing)
        save_npz(out/'source_calibration'/f'{i:03d}.npz',identity=np.asarray(identity),
                 relative_path=np.asarray(Path(record.path).relative_to(paths['data']).as_posix()),**observed)
        for offset,l in enumerate(LAYERS): pooled[l].append(observed['distance'][offset])
        print(f'[SPARC prepare] corrected calibration {i+1}/66',flush=True)
    _,mad=fit_calibrators({l:np.concatenate(pooled[l]) for l in LAYERS})
    replay.write_csv(out/'calibration_stats.csv',[dict(layer=l,source_calibration_images=66,
        patch_count=sum(v.size for v in pooled[l]),median=float(mad.medians[o]),mad_scale_1p4826=float(mad.scales[o]),
        calibration_query_corrected=True,reference='corrected_Source_only') for o,l in enumerate(LAYERS)])
    save_npz(out/'sparc_source_mad_state.npz',protocol_id=np.asarray(replay.PROTOCOL_ID),identity=np.asarray(identity),
             branch=np.asarray('SPARC_SOURCE_K8'),medians=np.asarray(mad.medians,np.float32),scales=np.asarray(mad.scales,np.float32))
    replay.check_mad(out/'sparc_source_mad_state.npz')
    timing['source_MAD_fit_inclusive']+=time.perf_counter()-t
    timing.update({'calibration_'+k:v for k,v in mad_timing.items()})
    for projection in projections.values(): projection.close()
    verify_files(inputs['required_files'])
    artifacts=[p for p in out.rglob('*') if p.is_file() and p.name not in ('inputs.json','preflight.json','prepare_status.json','status.json')]
    state=dict(tool=TOOL,status='prepare_complete',identity=identity,inputs_sha256=replay.sha256(out/'inputs.json'),
               artifacts=file_records(out,artifacts),builder_input_count=521,memory_fit_count=455,calibration_count=66,
               target_memory_appended=False,timing=dict(timing),seconds=time.perf_counter()-start_all)
    replay.atomic_json(out/'prepare_status.json',state)
    return state


def load_memory(model,path,identity,device):
    import torch
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if (payload.get('tool')!=TOOL or payload['identity']!=identity or payload['source_count']!=521
            or payload['target_memory_appended'] is not False): raise ValueError('invalid SPARC Source memory identity')
    banks={}
    for l in LAYERS:
        b=payload[f'bank_{l}']
        if b.dtype!=torch.float32 or tuple(b.shape)!=(100000,DIMENSION) or not torch.isfinite(b).all():
            raise ValueError('SPARC Source memory budget/dtype/finiteness changed')
        banks[l]=b.to(device)
    model.prototype_embeddings=banks


def infer(inputs,prepare_state,device_name):
    paths={k:Path(v) for k,v in inputs['paths'].items()};out=paths['output']
    if (out/'infer_status.json').exists() or (out/'sparc_raw').exists(): raise ValueError('infer refuses existing/partial predictions')
    start=time.perf_counter();timing=defaultdict(float)
    model,device=load_host(inputs,device_name)
    load_memory(model,out/'sparc_source_memory.pt',inputs['identity'],device)
    projections={l:load_projection(out/'projection'/f'L{l}',inputs['frozen']['grid']) for l in LAYERS}
    cap=json.loads((paths['capture']/'inputs.json').read_text(encoding='utf-8'))
    from DINOv3.MADEqual.aebad_s_target_support_trace.analysis import image_geometry
    old_state=json.loads((paths['capture']/'status.json').read_text(encoding='utf-8'))
    old={r['sample_id']:r for r in old_state['records']};records=[];per_image=[]
    for i,s in enumerate(inputs['samples']):
        image_started=time.perf_counter();before=dict(timing)
        with Image.open(paths['data']/s['relative_path']) as image:
            observed=observe_corrected(model,image.convert('RGB'),device,projections,timing)
            shape=(image.height//4,image.width//4)
            geometry=image_geometry(model,image.height,image.width,tuple(int(v) for v in observed['grid']),
                                    tuple(int(v) for v in observed['processed_shape']))
        with np.load(paths['capture']/old[s['sample_id']]['file'],allow_pickle=False) as f:
            metadata=json.loads(str(f['metadata'].item()));p=f['positions']
        if (metadata['grid']!=observed['grid'].tolist() or metadata['output_shape']!=list(shape)
                or metadata['geometry']!=geometry
                or not np.array_equal(p,observed['positions'])):
            raise ValueError('SPARC query full geometry/positions differs from verified original capture')
        meta=dict(tool=TOOL,identity=inputs['identity'],sample_id=s['sample_id'],relative_path=s['relative_path'],
            image_sha256=cap['image_sha256'][s['relative_path']],grid=observed['grid'].tolist(),output_shape=list(shape),
            baseline_raw_sha256=old[s['sample_id']]['sha256'],geometry=geometry,branch='SPARC_SOURCE_K8',distance_divisor=1280)
        file=out/'sparc_raw'/f"{s['sample_id']}.npz"
        save_npz(file,metadata=np.asarray(json.dumps(meta,sort_keys=True)),**observed)
        records.append(dict(sample_id=s['sample_id'],**file_records(out,[file])[0]))
        per_image.append(dict(sample_id=s['sample_id'],
            **{k:timing[k]-before.get(k,0.) for k in ('query_backbone','query_correction','query_retrieval')},
            total_including_image_read_and_raw_save=time.perf_counter()-image_started))
        print(f'[SPARC infer] {i+1}/297 {s["sample_id"]}',flush=True)
    verify_files(inputs['required_files'])
    from .inputs import verify_artifacts
    verify_artifacts(out,prepare_state['artifacts']);verify_artifacts(out,records)
    for projection in projections.values(): projection.close()
    replay.write_csv(out/'timing_per_image.csv',per_image)
    state=dict(tool=TOOL,status='predictions_frozen',identity=inputs['identity'],predicted_images=297,records=records,
        prepare_status_sha256=replay.sha256(out/'prepare_status.json'),inputs_sha256=replay.sha256(out/'inputs.json'),
        GT_decoded=False,timing=dict(timing),seconds=time.perf_counter()-start)
    replay.atomic_json(out/'infer_status.json',state)
    return state
