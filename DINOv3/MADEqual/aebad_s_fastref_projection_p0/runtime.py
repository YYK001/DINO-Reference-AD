"""One original whole-grid extraction per image, projection and original Exact NN."""
from __future__ import annotations
import json
import time
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
from . import inputs
from .projection import project, difference
from DINOv3.MADEqual.aebad_s_fastref_k8_p0 import runtime as original
from DINOv3.MADEqual.aebad_s_fastref_k8_p0.analysis import load_prediction
from DINOv3.MADEqual.aebad_s_union_support_loo.analysis import load_raw
from DINOv3.MADEqual.aebad_s_union_support_loo.runner import audit


def parent_raw(p, sample, record, prepared):
    old = next(r for r in p['capture_status']['records'] if r['sample_id']==sample['sample_id'])
    historical,meta = load_raw(Path(p['paths']['capture'])/old['file'],
                              identity=p['capture']['identity'],sample=sample,checksum=old['sha256'])
    arrays = load_prediction(p,sample,record,old,meta,historical)
    return arrays,meta


def expected_checks(samples):
    return {(s['sample_id'],k,'') for s in samples for k in
            ('Source_full_grid','prototype_full_grid','Source_image_score','prototype_image_score')} | {
            (s['sample_id'],'projection_normal_equation',str(l)) for s in samples for l in inputs.LAYERS}


def load_raw_new(data,sample,record,parent_arrays,meta,*,smoke=False):
    directory = 'smoke_raw' if smoke else 'raw'
    file = Path(data['paths']['output'])/record['file']
    if record['file']!=f"{directory}/{sample['sample_id']}.npz" or inputs.replay.sha256(file)!=record['sha256']:
        raise ValueError('modified/unsafe projection raw')
    with np.load(file,allow_pickle=False) as f:
        m=json.loads(str(f['metadata'].item())); arrays={k:f[k].copy() for k in f.files if k!='metadata'}
    if (m['tool']!=inputs.TOOL or m['identity']!=data['identity'] or m['sample_id']!=sample['sample_id']
            or m['relative_path']!=sample['relative_path'] or m['image_sha256']!=meta['image_sha256']
            or m['parent_identity']!=inputs.PARENT_ID or m['prototype_ids']!=data['frozen']['prototype_ids']
            or m['geometry']!=meta['geometry'] or m['grid']!=meta['grid'] or m['output_shape']!=meta['output_shape']
            or m['distance_divisor']!=1280 or m['query_forward_calls']!=1 or m['OT_called']
            or m['GT_decoded'] or m['query_features_persisted'] or m['projection_vectors_persisted']
            or m['parent_raw_sha256']!=inputs.replay.sha256(Path(data['paths']['parent'])/'raw'/f"{sample['sample_id']}.npz")
            or not np.array_equal(arrays['positions'],parent_arrays['positions'])
            or arrays['grid'].tolist()!=meta['grid']
            or not np.array_equal(arrays['processed_shape'],parent_arrays['processed_shape'])):
        raise ValueError('projection raw identity/geometry/path semantics changed')
    n=int(np.prod(meta['grid']))
    for key in ('source_distance','prototype_distance','projection_distance','projection_union_distance'):
        a=arrays[key]
        if a.shape!=(4,n) or a.dtype!=np.float32 or not np.isfinite(a).all() or np.any(a<0):
            raise ValueError('invalid projection distance')
    for key in ('self_projection_residual','self_cast_projection_residual'):
        a=arrays[key]
        if a.shape!=(4,n) or a.dtype!=np.float64 or not np.isfinite(a).all() or np.any(a<0):
            raise ValueError('invalid independent residual')
    index=arrays['projection_index']
    if index.shape!=(4,n) or index.dtype!=np.int64 or np.any(index<0) or np.any(index>=n):
        raise ValueError('invalid all-projection nearest index')
    if not np.array_equal(arrays['projection_union_distance'],np.minimum(arrays['source_distance'],arrays['projection_distance'])):
        raise ValueError('Target-only distance confused with Source union')
    return arrays


def verify_stage(data,p,prepared,parent_state,*,smoke=False):
    root=Path(data['paths']['output']); mode='smoke' if smoke else 'infer'
    state=inputs.read_json(root/f'{mode}_status.json')
    samples=p['capture']['selected_cases'] if smoke else p['capture']['samples']
    checks=inputs.replay.read_csv(root/f'{mode}_verification.csv')
    keys={(r['sample_id'],r['check_kind'],r.get('layer','')) for r in checks}
    if (state['tool']!=inputs.TOOL or state['identity']!=data['identity']
            or state['status']!=('smoke_verified' if smoke else 'predictions_frozen')
            or state['images']!=(12 if smoke else 297) or len(state['records'])!=len(samples)
            or [r['sample_id'] for r in state['records']]!=[s['sample_id'] for s in samples]
            or state['query_forward_calls']!=len(samples) or state['OT_called'] or state['GT_decoded']
            or state['inputs_sha256']!=inputs.replay.sha256(root/'inputs.json')
            or state['verification_sha256']!=inputs.replay.sha256(root/f'{mode}_verification.csv')
            or state['diagnostics_sha256']!=inputs.replay.sha256(root/f'{mode}_projection_diagnostics.json')
            or keys!=expected_checks(samples) or len(keys)!=len(checks)
            or any(r['verified'].lower()!='true' for r in checks)):
        raise ValueError('complete frozen projection predictions and numerical checks required')
    diag=inputs.read_json(root/f'{mode}_projection_diagnostics.json')
    wanted={(s['sample_id'],l) for s in samples for l in inputs.LAYERS}
    if len(diag)!=len(wanted) or {(d['sample_id'],d['layer']) for d in diag}!=wanted:
        raise ValueError('complete per-image/layer projection diagnostics required')
    for d in diag:
        if (d['OT_called'] or d['query_rows']!=d['projection_rows']
                or d['prototype_identity']!=data['frozen']['prototype_ids'][str(d['layer'])]
                or not np.isfinite(d['normal_equation_relative_residual'])
                or d['normal_equation_relative_residual']>1e-8):
            raise ValueError('projection diagnostics failed')
    inputs.parent.verify_files(data['required_files'])
    inputs.parent.verify_artifacts(root,state['records'])
    old={r['sample_id']:r for r in parent_state['records']}
    per_layer={(d['sample_id'],d['layer']):d for d in diag}
    for sample,record in zip(samples,state['records'],strict=True):
        arrays,meta=parent_raw(p,sample,old[sample['sample_id']],prepared)
        if any(per_layer[sample['sample_id'],l]['query_rows']!=int(np.prod(meta['grid'])) for l in inputs.LAYERS):
            raise ValueError('projection diagnostic query row count changed')
        load_raw_new(data,sample,record,arrays,meta,smoke=smoke)
    return state


def inference(data,p,prepared,parent_state,device_name,*,smoke=False):
    # CUDA and the host are imported only behind an explicit computational stage.
    import torch
    from DINOv3.MADEqual.aebad_s_target_fewshot.runner import _load_memory
    from DINOv3.MADEqual.aebad_s_union_diagnostics.spatial import MapBackend
    from DINOv3.MADEqual.aebad_s_target_support_trace.analysis import image_geometry
    root=Path(data['paths']['output']); mode='smoke' if smoke else 'infer'
    directory='smoke_raw' if smoke else 'raw'
    if (root/directory).exists() or (root/f'{mode}_status.json').exists():
        raise ValueError('no repeat or partial-cache re-approval; use new output')
    if not smoke: verify_stage(data,p,prepared,parent_state,smoke=True)
    inputs.parent.verify_files(data['required_files'])
    host=dict(paths=p['paths'],frozen=dict(host_config=p['frozen']['host_config'],precision=p['frozen']['precision']))
    model,device=original.host_runtime.load_host(host,device_name)
    if not _load_memory(Path(p['paths']['memory']),model,source_count=521,device=device):
        raise ValueError('original Source memory required; no rebuilding')
    constants=original.load_constants(p,prepared)
    banks={l:torch.from_numpy(constants[l]['prototypes']).to(device) for l in inputs.LAYERS}
    backend=MapBackend(Path(p['paths']['mad']))
    scores=inputs.historical_scores(data['paths']['parent'])
    samples=p['capture']['selected_cases'] if smoke else p['capture']['samples']
    old={r['sample_id']:r for r in parent_state['records']}
    checks=[]; diagnostics=[]; records=[]; timing=defaultdict(float); started=time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=data['frozen']['cpu_workers']) as executor:
            for number,sample in enumerate(samples,1):
                sid=sample['sample_id']; cached,meta=parent_raw(p,sample,old[sid],prepared)
                file=Path(p['paths']['data'])/sample['relative_path']
                if inputs.replay.sha256(file)!=meta['image_sha256']: raise ValueError('query RGB changed')
                with Image.open(file) as image:
                    queries,positions,grid,processed=original.query_forward(model,image.convert('RGB'),device,timing)
                    geometry=image_geometry(model,image.height,image.width,grid,processed)
                if list(grid)!=meta['grid'] or geometry!=meta['geometry'] or not np.array_equal(positions,cached['positions']):
                    raise ValueError('original query geometry/positions changed')
                qgpu={l:torch.from_numpy(queries[l]).to(device) for l in inputs.LAYERS}
                source=[]; proto=[]; t=time.perf_counter()
                with torch.inference_mode():
                    for l in inputs.LAYERS:
                        source.append(original.host_nn(qgpu[l],model.prototype_embeddings[l])[0])
                        proto.append(original.host_nn(qgpu[l],banks[l],compact=True)[0])
                torch.cuda.synchronize(device)
                timing['Source_and_prototype_baseline_search']+=time.perf_counter()-t
                source=np.stack(source); proto=np.stack(proto)
                okay=audit(checks,sid,'Source_full_grid',source,cached['source_distance'])
                okay=audit(checks,sid,'prototype_full_grid',proto,cached['prototype_distance']) and okay
                baseline=original.maps_for(dict(SOURCE=source,UNION_TARGET512=np.minimum(source,proto)),positions,meta,backend)
                for branch,kind in [('SOURCE','Source_image_score'),('UNION_TARGET512','prototype_image_score')]:
                    okay=audit(checks,sid,kind,baseline[branch].max(),scores[branch,sid],score=True) and okay
                if not okay: raise ValueError('Source/P query baseline reproduction failed; original tolerances unchanged')
                t=time.perf_counter()
                projected=list(executor.map(lambda l:project(queries[l],constants[l]),inputs.LAYERS))
                timing['initial_projection_and_independent_residual_CPU']+=time.perf_counter()-t
                distances=[]; indices=[]; residuals=[]; cast=[]; t=time.perf_counter()
                with torch.inference_mode():
                    for l,(vectors,residual,cast_residual,diag) in zip(inputs.LAYERS,projected,strict=True):
                        d,index=original.host_nn(qgpu[l],torch.from_numpy(vectors).to(device),compact=True)
                        distances.append(d); indices.append(index); residuals.append(residual); cast.append(cast_residual)
                        diag.update(sample_id=sid,layer=l,
                                    nearest_index_differs_from_self=int(np.count_nonzero(index!=np.arange(len(index)))),
                                    exact_minus_ideal_self_residual=difference(d,residual),
                                    exact_minus_float32_self_residual=difference(d,cast_residual),
                                    difference_note='saved float64 solve, float32 reference cast, original Torch cdist; diagnostic only')
                        diagnostics.append(diag)
                        checks.append(dict(sample_id=sid,layer=l,check_kind='projection_normal_equation',verified=True,
                                           relative_residual=diag['normal_equation_relative_residual']))
                torch.cuda.synchronize(device)
                timing['query_to_all_projection_ExactNN']+=time.perf_counter()-t
                distance=np.stack(distances)
                arrays=dict(source_distance=source,prototype_distance=proto,projection_distance=distance,
                            projection_union_distance=np.minimum(source,distance),projection_index=np.stack(indices),
                            self_projection_residual=np.stack(residuals),self_cast_projection_residual=np.stack(cast),
                            positions=positions,grid=np.asarray(grid,np.int32),processed_shape=np.asarray(processed,np.int32))
                metadata=dict(tool=inputs.TOOL,identity=data['identity'],parent_identity=inputs.PARENT_ID,
                              prototype_ids=data['frozen']['prototype_ids'],sample_id=sid,relative_path=sample['relative_path'],
                              image_sha256=meta['image_sha256'],geometry=geometry,grid=list(grid),output_shape=meta['output_shape'],
                              parent_raw_sha256=old[sid]['sha256'],distance_divisor=1280,query_forward_calls=1,
                              OT_called=False,GT_decoded=False,query_features_persisted=False,projection_vectors_persisted=False)
                file=root/directory/f'{sid}.npz'
                original.host_runtime.save_npz(file,metadata=np.asarray(json.dumps(metadata,sort_keys=True)),**arrays)
                records.append(dict(sample_id=sid,**inputs.parent.file_records(root,[file])[0]))
                del queries,qgpu,projected
                print(f'[Projection {mode}] {number}/{len(samples)} {sid}',flush=True)
        inputs.parent.verify_files(data['required_files'])
        state=dict(tool=inputs.TOOL,identity=data['identity'],status='smoke_verified' if smoke else 'predictions_frozen',
                   images=len(records),records=records,query_forward_calls=len(records),OT_called=False,GT_decoded=False,
                   inputs_sha256=inputs.replay.sha256(root/'inputs.json'),seconds=time.perf_counter()-started,timing=dict(timing))
    finally:
        inputs.replay.write_csv(root/f'{mode}_verification.csv',checks)
        inputs.replay.atomic_json(root/f'{mode}_projection_diagnostics.json',diagnostics)
    state['verification_sha256']=inputs.replay.sha256(root/f'{mode}_verification.csv')
    state['diagnostics_sha256']=inputs.replay.sha256(root/f'{mode}_projection_diagnostics.json')
    inputs.replay.atomic_json(root/f'{mode}_status.json',state)
    try:
        verify_stage(data,p,prepared,parent_state,smoke=smoke)
    except Exception:
        # A late integrity/semantic failure must not leave a successful status.
        state['status']='stage_failed'
        inputs.replay.atomic_json(root/f'{mode}_status.json',state)
        raise
    return state
