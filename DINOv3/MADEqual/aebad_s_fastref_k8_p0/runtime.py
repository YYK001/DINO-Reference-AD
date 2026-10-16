"""Explicit cached-support preparation and original full-query GPU inference."""
from __future__ import annotations
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
from . import inputs, refinement, sampling
from DINOv3.MADEqual.aebad_s_sparc_p0 import runtime as host_runtime
from DINOv3.MADEqual.aebad_s_union_support_loo.analysis import load_raw, branch_maps
from DINOv3.MADEqual.aebad_s_union_support_loo.runner import audit, formal_baselines


def select_prototypes(bank, device, *, rng):
    """Explicit finite density adaptation; returns original indices and audit."""
    cfg = inputs.SAMPLING
    if bank.dtype != np.float32 or bank.ndim != 2 or bank.shape[1] != 1280 or not np.isfinite(bank).all():
        raise ValueError('original finite float32 Target pool required')
    admission = inputs.sampling_compatibility(len(bank))
    if admission['error']: raise ValueError(admission['error'])
    indices, report = sampling.sample_indices(bank,cfg['quota_per_layer'],device,rng=rng,
        rounds=cfg['rounds'],neighbors=cfg['knn_neighbors'])
    indices = np.asarray(indices)
    if (indices.dtype != np.int64 or indices.shape != (512,) or np.any(indices < 0)
        or np.any(indices >= len(bank)) or np.any(np.diff(indices) <= 0)):
        raise ValueError('density sampler must return 512 ordered unique original Target indices; no fallback')
    return indices.copy(),report


def provenance_rows(layer, indices, supports):
    sizes = [int(np.prod(s['grid'])) for s in supports]; boundaries = np.cumsum([0,*sizes])
    rows = []
    for prototype_index, selected in enumerate(indices):
        support = int(np.searchsorted(boundaries[1:],selected,side='right'))
        if support >= len(supports) or selected < 0: raise ValueError('prototype outside original supports')
        local = int(selected-boundaries[support]); row = supports[support]
        rows.append(dict(layer=layer,prototype_index=prototype_index,selected_target_flat_index=int(selected),
            support_index=row['support_index'],support_id=Path(row['relative_path']).stem,relative_path=row['relative_path'],physical_patch_index=local,
            patch_y=local//row['grid'][1],patch_x=local%row['grid'][1],original_feature_sha256=row['feature_sha256']))
    return rows


def load_target_layer(data, layer):
    """Serialized frozen paths are strings; the shared reader expects Path."""
    paths = {key:Path(value) for key,value in data['paths'].items()}
    return inputs.shared.load_support_layer(paths,data['capture']['supports'],layer)


def prepare(data, device_name):
    candidate_count = sum(int(np.prod(s['grid'])) for s in data['capture']['supports'])
    admission = inputs.sampling_compatibility(candidate_count)
    if admission['error']: raise ValueError(admission['error'])
    import torch
    root = Path(data['paths']['output']); started = time.perf_counter(); timings = defaultdict(float)
    inputs.verify_files(data['required_files'])
    device = torch.device(device_name)
    if device.type != 'cuda' or not torch.cuda.is_available(): raise RuntimeError('explicit exact-kNN preparation requires CUDA; no backbone is loaded')
    device = torch.device('cuda',torch.cuda.current_device() if device.index is None else device.index)
    torch.cuda.set_device(device); torch.set_float32_matmul_precision(data['frozen']['precision']['float32_matmul_precision'])
    torch.backends.cuda.matmul.allow_tf32=data['frozen']['precision']['cuda_matmul_allow_tf32']
    torch.backends.cudnn.allow_tf32=data['frozen']['precision']['cudnn_allow_tf32']; torch.cuda.reset_peak_memory_stats(device)
    picked = {}; indices = {}; rows = []; sampling_diagnostics=[]
    if data['frozen'].get('sampling') != inputs.SAMPLING: raise ValueError('frozen v3 sampling configuration required')
    rng = np.random.RandomState(inputs.SAMPLING['seed'])
    for layer in inputs.LAYERS:
        t = time.perf_counter()
        banks = load_target_layer(data,layer)
        bank = np.concatenate(banks); timings['normal_support_loading'] += time.perf_counter()-t
        t = time.perf_counter()
        with torch.inference_mode(): index,diagnostic = select_prototypes(bank,device,rng=rng)
        picked[layer] = np.ascontiguousarray(bank[index].copy()); indices[layer] = index
        timings['normal_target_bounded_density_sampling'] += time.perf_counter()-t
        sampling_diagnostics.append(dict(layer=layer,identity=data['identity'],sampling=dict(inputs.SAMPLING),**diagnostic))
        inputs.replay.atomic_json(root/'sampling_diagnostics.json',sampling_diagnostics)
        for row,selection in zip(provenance_rows(layer,index,data['capture']['supports']),diagnostic['selected'],strict=True):
            rows.append(dict(row,sampling_revision=inputs.SAMPLING['revision'],sampling_entry=inputs.SAMPLING['entry'],
                sampling_seed=inputs.SAMPLING['seed'],sampling_config=json.dumps(inputs.SAMPLING,sort_keys=True),input_identity=data['identity'],
                density_group=selection['group'],density_neighbors=selection['neighbors'],density_radius=selection['radius'],
                density_count=selection['density_count'],density_weight=selection['weight'],weighted_priority=selection['priority']))
        inputs.replay.write_csv(root/'prototype_provenance.csv',rows)
        del bank,banks; print(f'[FastRef prepare] L{layer} selected 512 original Target patches',flush=True)
    t = time.perf_counter()
    with ThreadPoolExecutor(max_workers=data['frozen']['config']['cpu_workers']) as executor:
        constants = list(executor.map(lambda layer:refinement.make_constants(picked[layer]),inputs.LAYERS))
    timings['normal_prototype_solver_constants'] += time.perf_counter()-t
    files = [root/'sampling_diagnostics.json',root/'prototype_provenance.csv']; diagnostics = []; prototype_ids = {}
    for layer,constant in zip(inputs.LAYERS,constants,strict=True):
        path = root/'prototypes'/f'L{layer}.npz'
        metadata = inputs.prototype_metadata(data,layer)
        prototype_ids[str(layer)] = inputs.prototype_identity(metadata,constant['prototypes'],indices[layer])
        host_runtime.save_npz(path,prototypes=constant['prototypes'],factor=constant['factor'],
            epsilon=np.asarray(constant['epsilon'],np.float64),indices=indices[layer],
            metadata=np.asarray(json.dumps(metadata,sort_keys=True)),prototype_identity=np.asarray(prototype_ids[str(layer)]),
            diagnostics=np.asarray(json.dumps(constant['diagnostics'],sort_keys=True)))
        files.append(path); diagnostics.append(dict(layer=layer,verified=True,**constant['diagnostics']))
    inputs.replay.atomic_json(root/'prototype_numerical_diagnostics.json',diagnostics); files.append(root/'prototype_numerical_diagnostics.json')
    inputs.verify_files(data['required_files'])
    state = dict(tool=inputs.TOOL,status='prototypes_prepared',identity=data['identity'],
        inputs_sha256=inputs.replay.sha256(root/'inputs.json'),artifacts=inputs.file_records(root,files),
        backbone_loaded=False,Source_memory_rebuilt=False,MAD_fitted=False,GT_decoded=False,
        sampling=dict(inputs.SAMPLING),prototype_ids=prototype_ids,
        selector_sha256=inputs.replay.sha256(Path(data['paths']['selector'])),seconds=time.perf_counter()-started,timing=dict(timings))
    inputs.replay.atomic_json(root/'prepare_status.json',state); return state


def load_constants(data, prepared):
    inputs.verify_sampling(data,prepared)
    root = Path(data['paths']['output'])
    answer = {}
    for layer in inputs.LAYERS:
        with np.load(Path(root)/'prototypes'/f'L{layer}.npz',allow_pickle=False) as f:
            p = f['prototypes'].copy(); factor = f['factor'].copy(); epsilon = float(f['epsilon'].item())
            diag = json.loads(str(f['diagnostics'].item()))
            indices = f['indices'].copy(); metadata = json.loads(str(f['metadata'].item()))
            identity = str(f['prototype_identity'].item())
        count = sum(int(np.prod(s['grid'])) for s in data['capture']['supports'])
        if (metadata != inputs.prototype_metadata(data,layer) or indices.dtype != np.int64 or indices.shape != (512,)
            or np.any(indices < 0) or np.any(indices >= count) or np.any(np.diff(indices) <= 0)
            or identity != inputs.prototype_identity(metadata,p,indices) or prepared.get('prototype_ids',{}).get(str(layer)) != identity):
            raise ValueError('prepared prototype sampling/input identity or original index order changed')
        if p.shape != (512,1280) or p.dtype != np.float32 or factor.shape != p.shape or factor.dtype != np.float64 or not np.isfinite(p).all() or not np.isfinite(factor).all() or not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError('prepared prototype/solver arrays invalid')
        answer[layer] = dict(prototypes=p,scaled=p.astype(np.float64)/1280,factor=factor,epsilon=epsilon,diagnostics=diag,
            prototype_identity=identity,indices=indices)
    return answer


def maps_for(raws, positions, meta, backend):
    n = int(np.prod(meta['grid'])); p = np.asarray(positions)
    if p.dtype != np.int64 or p.shape != (n,) or not np.array_equal(np.sort(p),np.arange(n)):
        raise ValueError('complete bijective physical patch positions required')
    scattered = {}
    for branch,raw in raws.items():
        if raw.shape != (4,n) or raw.dtype != np.float32 or not np.isfinite(raw).all() or np.any(raw < 0): raise ValueError('invalid host distance arrays')
        physical = np.empty_like(raw); physical[:,p] = raw
        scattered[branch] = physical.reshape(4,*meta['grid'])
    return branch_maps(scattered,tuple(meta['output_shape']),backend)


def query_forward(model, image, device, timing):
    import torch
    from DINOv3.relation_reliability.superadd_host import image_tensor, _extract_processed_batch
    t = time.perf_counter()
    with torch.inference_mode():
        processed = model.preprocessing(image_tensor(image).to(device)[None])
        pending = _extract_processed_batch(model,processed,device=device,query_masks=[None],include_identity_descriptors=False)
        if len(pending) != 1: raise ValueError('one original complete-grid query forward required')
        item = pending[0]
        queries = {l:np.ascontiguousarray(item.queries[l].detach().float().cpu().numpy()) for l in inputs.LAYERS}
    torch.cuda.synchronize(device); timing['backbone_and_query_transfer'] += time.perf_counter()-t
    return queries,np.asarray(item.positions,np.int64),tuple(int(v) for v in item.grid),tuple(int(v) for v in processed.shape[-2:])


def host_nn(q, bank, *, compact=False):
    from DINOv3.relation_reliability.superadd_host import _exact_top1_chunked
    kwargs = dict(max_matrix_bytes=refinement.CONFIG['new_nn_matrix_bytes']) if compact else {}
    distance,index = _exact_top1_chunked(q,bank,**kwargs)
    # Kernel returns Euclidean, never squared/cosine; divide exactly once.
    values = distance.numpy().astype(np.float32)/1280.; indices = index.numpy().astype(np.int64)
    if values.shape != (q.shape[0],) or indices.shape != values.shape or not np.isfinite(values).all() or np.any(values < 0) or np.any(indices < 0) or np.any(indices >= bank.shape[0]):
        raise ValueError('invalid original Exact Euclidean distance/index result')
    return values,indices


def expected_checks(samples):
    kinds = ('Source_full_grid','Target8_full_grid','Source_image_score','UNION_image_score','prototype_subset')
    return {(s['sample_id'],kind,'','') for s in samples for kind in kinds} | {
        (s['sample_id'],'FastRef_solver',str(layer),str(outer)) for s in samples for layer in inputs.LAYERS for outer in (1,2)}


def inference(data, prepared, device_name, *, smoke=False):
    import torch
    from DINOv3.MADEqual.aebad_s_target_fewshot.runner import _load_memory
    from DINOv3.MADEqual.aebad_s_union_diagnostics.spatial import MapBackend
    from DINOv3.MADEqual.aebad_s_target_support_trace.analysis import image_geometry
    root = Path(data['paths']['output']); mode = 'smoke' if smoke else 'infer'; raw_dir = 'smoke_raw' if smoke else 'raw'
    if (root/raw_dir).exists() or (root/f'{mode}_status.json').exists(): raise ValueError('no repeat/partial prediction re-approval; use a new output')
    samples = data['capture']['selected_cases'] if smoke else data['capture']['samples']
    if not smoke: verify_stage(data,prepared,smoke=True)
    inputs.verify_files(data['required_files']); inputs.verify_artifacts(root,prepared['artifacts'])
    host_data = dict(paths=data['paths'],frozen=dict(host_config=data['frozen']['host_config'],precision=data['frozen']['precision']))
    model,device = host_runtime.load_host(host_data,device_name)
    if not _load_memory(Path(data['paths']['memory']),model,source_count=521,device=device): raise ValueError('original Source bank missing; no rebuilding')
    constants = load_constants(data,prepared); backend = MapBackend(Path(data['paths']['mad']))
    formal = formal_baselines(Path(data['paths']['results']),data['capture']['samples'])
    supports = {}
    for layer in inputs.LAYERS:
        banks = load_target_layer(data,layer)
        provenance = provenance_rows(layer,constants[layer]['indices'],data['capture']['supports'])
        selected_rows = np.stack([banks[int(r['support_index'])-1][r['physical_patch_index']] for r in provenance])
        if not np.array_equal(selected_rows,constants[layer]['prototypes']):
            raise ValueError('prepared prototypes differ from original Target rows; no re-sampling')
        supports[layer] = [torch.from_numpy(b).to(device) for b in banks]
        del banks,selected_rows
    prototypes = {l:torch.from_numpy(constants[l]['prototypes']).to(device) for l in inputs.LAYERS}
    old = {r['sample_id']:r for r in data['capture_status']['records']}
    started = time.perf_counter(); timing = defaultdict(float); checks = []; records = []; numerical = []
    try:
        for number,sample in enumerate(samples,1):
            sid = sample['sample_id']; historical,meta = load_raw(Path(data['paths']['capture'])/old[sid]['file'],
                identity=data['capture']['identity'],sample=sample,checksum=old[sid]['sha256'])
            image_path = Path(data['paths']['data'])/sample['relative_path']
            if inputs.replay.sha256(image_path) != meta['image_sha256']: raise ValueError('original query RGB modified')
            with Image.open(image_path) as image:
                queries,positions,grid,processed = query_forward(model,image.convert('RGB'),device,timing)
                geometry = image_geometry(model,image.height,image.width,grid,processed)
            if list(grid) != meta['grid'] or geometry != meta['geometry'] or not np.array_equal(positions,historical['positions']):
                raise ValueError('complete original query geometry/positions differ')
            source = []; target = []; prototype = []; prototype_indices = []
            q_gpu = {l:torch.from_numpy(queries[l]).to(device) for l in inputs.LAYERS}
            with torch.inference_mode():
                t = time.perf_counter()
                for layer in inputs.LAYERS:
                    source.append(host_nn(q_gpu[layer],model.prototype_embeddings[layer])[0])
                    target.append([host_nn(q_gpu[layer],bank)[0] for bank in supports[layer]])
                timing['historical_Source_and_Target8_verification_search'] += time.perf_counter()-t
                t = time.perf_counter()
                for layer in inputs.LAYERS:
                    distance,index = host_nn(q_gpu[layer],prototypes[layer],compact=True)
                    prototype.append(distance); prototype_indices.append(index)
                timing['prototype512_search'] += time.perf_counter()-t
            source = np.stack(source); target = np.stack(target,axis=1); prototype = np.stack(prototype)
            union = np.minimum(source,target.min(axis=0)); proto_union = np.minimum(source,prototype)
            old_maps = maps_for(dict(SOURCE=source,UNION_K8=union),positions,meta,backend)
            okay = audit(checks,sid,'Source_full_grid',source,historical['source_distance'])
            okay = audit(checks,sid,'Target8_full_grid',target,historical['target_distance']) and okay
            for branch,key in [('SOURCE','Source'),('UNION_K8','K8')]:
                okay = audit(checks,sid,'Source_image_score' if key=='Source' else 'UNION_image_score',
                    old_maps[branch].max(),formal[key][sid]['image_score'],score=True) and okay
            subset = bool(np.all(prototype+1e-7+1e-6*target.min(axis=0) >= target.min(axis=0)))
            checks.append(dict(sample_id=sid,check_kind='prototype_subset',verified=subset))
            if not okay or not subset: raise ValueError('historical baseline/subset verification failed; tolerances unchanged')
            t = time.perf_counter()
            with ThreadPoolExecutor(max_workers=data['frozen']['config']['cpu_workers']) as executor:
                refined = list(executor.map(lambda l:refinement.refine(queries[l],constants[l]),inputs.LAYERS))
            timing['full_query_FastRef_CPU'] += time.perf_counter()-t
            refined_distances = []; refined_indices = []; t = time.perf_counter()
            with torch.inference_mode():
                for layer,(vectors,diag) in zip(inputs.LAYERS,refined,strict=True):
                    distance,index = host_nn(q_gpu[layer],torch.from_numpy(vectors).to(device),compact=True)
                    refined_distances.append(distance); refined_indices.append(index)
                    numerical.append(dict(sample_id=sid,layer=layer,**diag))
                    for step in diag['outer']:
                        checks.append(dict(sample_id=sid,layer=layer,outer=step['outer_iteration'],check_kind='FastRef_solver',verified=True,**step))
            timing['query_to_all_refined_NN'] += time.perf_counter()-t
            dref = np.stack(refined_distances)
            arrays = dict(source_distance=source,target_full_distance=target.min(axis=0),prototype_distance=prototype,
                refined_distance=dref,prototype_union_distance=proto_union,fastref_union_distance=np.minimum(source,dref),
                prototype_index=np.stack(prototype_indices),refined_index=np.stack(refined_indices),positions=positions,
                grid=np.asarray(grid,np.int32),processed_shape=np.asarray(processed,np.int32))
            metadata = dict(tool=inputs.TOOL,identity=data['identity'],sample_id=sid,relative_path=sample['relative_path'],
                image_sha256=meta['image_sha256'],grid=list(grid),geometry=geometry,output_shape=meta['output_shape'],
                historical_raw_sha256=old[sid]['sha256'],distance_divisor=1280,query_forward_calls=1,
                prepare_sha256=inputs.replay.sha256(root/'prepare_status.json'),refined_rows=len(positions),
                query_features_persisted=False,refined_vectors_persisted=False,GT_decoded=False)
            file = root/raw_dir/f'{sid}.npz'; host_runtime.save_npz(file,metadata=np.asarray(json.dumps(metadata,sort_keys=True)),**arrays)
            records.append(dict(sample_id=sid,**inputs.file_records(root,[file])[0]))
            del queries,q_gpu,refined
            print(f'[FastRef {mode}] {number}/{len(samples)} {sid}',flush=True)
        inputs.verify_files(data['required_files']); inputs.verify_artifacts(root,prepared['artifacts']); inputs.verify_artifacts(root,records)
        state = dict(tool=inputs.TOOL,status='smoke_verified' if smoke else 'predictions_frozen',identity=data['identity'],
            images=len(records),records=records,query_forward_calls=len(records),GT_decoded=False,
            inputs_sha256=inputs.replay.sha256(root/'inputs.json'),prepare_sha256=inputs.replay.sha256(root/'prepare_status.json'),
            seconds=time.perf_counter()-started,timing=dict(timing))
        inputs.replay.write_csv(root/f'{mode}_numerical_verification.csv',checks)
        inputs.replay.atomic_json(root/f'{mode}_solver_diagnostics.json',numerical)
        state['verification_sha256']=inputs.replay.sha256(root/f'{mode}_numerical_verification.csv')
        state['solver_diagnostics_sha256']=inputs.replay.sha256(root/f'{mode}_solver_diagnostics.json')
        inputs.replay.atomic_json(root/f'{mode}_status.json',state); return state
    finally:
        inputs.replay.write_csv(root/f'{mode}_numerical_verification.csv',checks)
        inputs.replay.atomic_json(root/f'{mode}_solver_diagnostics.json',numerical)


def verify_stage(data,prepared,*,smoke=False):
    inputs.verify_sampling(data,prepared)
    root = Path(data['paths']['output']); mode = 'smoke' if smoke else 'infer'
    state = json.loads((root/f'{mode}_status.json').read_text())
    samples = data['capture']['selected_cases'] if smoke else data['capture']['samples']
    checks = inputs.replay.read_csv(root/f'{mode}_numerical_verification.csv')
    keys = {(r['sample_id'],r['check_kind'],r.get('layer',''),r.get('outer','')) for r in checks}
    if (state['tool'] != inputs.TOOL or state['identity'] != data['identity']
        or state['status'] != ('smoke_verified' if smoke else 'predictions_frozen') or state['images'] != (12 if smoke else 297)
        or len(samples) != (12 if smoke else 297) or len(state['records']) != len(samples)
        or [r['sample_id'] for r in state['records']] != [s['sample_id'] for s in samples]
        or state['inputs_sha256'] != inputs.replay.sha256(root/'inputs.json')
        or state['prepare_sha256'] != inputs.replay.sha256(root/'prepare_status.json')
        or state['verification_sha256'] != inputs.replay.sha256(root/f'{mode}_numerical_verification.csv')
        or state['solver_diagnostics_sha256'] != inputs.replay.sha256(root/f'{mode}_solver_diagnostics.json')
        or keys != expected_checks(samples) or len(checks) != len(keys) or any(r['verified'].lower() != 'true' for r in checks)):
        raise ValueError('complete successful frozen numerical/baseline checks required')
    for record in state['records']:
        if record['file'] != f"{'smoke_raw' if smoke else 'raw'}/{record['sample_id']}.npz": raise ValueError('unsafe prediction path')
    diagnostics = json.loads((root/f'{mode}_solver_diagnostics.json').read_text())
    wanted = {(s['sample_id'],l) for s in samples for l in inputs.LAYERS}
    if len(diagnostics) != len(wanted) or {(d['sample_id'],d['layer']) for d in diagnostics} != wanted:
        raise ValueError('complete per-query/layer solver diagnostics required')
    cfg = data['frozen']['config']
    for diag in diagnostics:
        if (diag['gram_rank'] != 512 or not np.isfinite(diag['gram_condition']) or diag['gram_condition'] > cfg['maximum_gram_condition']
            or not np.isfinite(diag['solve_relative_residual']) or diag['solve_relative_residual'] > cfg['solve_relative_tolerance']
            or len(diag['outer']) != cfg['outer_iterations']): raise ValueError('prototype solver diagnostics failed')
        for step in diag['outer']:
            values = [step[k] for k in ('row_max_relative_error','column_max_relative_error','total_mass_error','update_normal_equation_relative_residual')]
            if (not np.isfinite(values).all() or max(values[:2]) > cfg['marginal_relative_tolerance']
                or values[2] > cfg['total_mass_tolerance'] or values[3] > cfg['update_relative_tolerance']
                or step['nonfinite_entries'] or step['transport_rescaled_by_m'] or step['lambda_value'] != cfg['lambda_value']):
                raise ValueError('full-query OT/update diagnostics failed')
    inputs.verify_files(data['required_files']); inputs.verify_artifacts(root,prepared['artifacts']); inputs.verify_artifacts(root,state['records'])
    return state
