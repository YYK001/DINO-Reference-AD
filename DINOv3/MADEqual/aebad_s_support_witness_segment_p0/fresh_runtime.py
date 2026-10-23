"""Explicit single-GPU reconstruction using the actual original production paths."""
import gc
from pathlib import Path

import numpy as np

from . import storage as st
from . import fresh_inputs as fi
from .method import LAYERS, BASES
from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import statistics, fit_image_mad, ch1_score


def host(args):
    from DINOv3.MADEqual.aebad_s_sparc_p0.runtime import load_host
    from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.inputs import PRECISION
    project = fi.ROOT
    official = project / 'external_baselines/SuperADD'
    return load_host(dict(paths=dict(project=str(project), official=str(official),
        dinov3=str(project/'external_baselines/dinov3'), checkpoint=str(args.checkpoint)),
        frozen=dict(precision=PRECISION, host_config=st.read_json(official/'config.json'))), args.device)


def catalog(args):
    return st.read_json(args.run_root/'fresh_catalog.json')


def source(args):
    """455 selected RGBs are loaded lazily; original augmentation/sampler unchanged."""
    identity = fi.setup_identity(args.run_root)
    path = args.cache_root/'source_bank.npz'
    if st.restore(path, identity):
        return dict(status='source_bank_reused')
    import torch
    from external_baselines.superadd_external.memory_variants import build_official_memory_spooled
    from DINOv3.relation_reliability.superadd_host import image_tensor
    c = catalog(args)
    model, device = host(args)
    class Images:
        def __len__(self): return len(c['source'])
        def __getitem__(self, index):
            if not 0 <= index < len(self): raise IndexError(index)
            with fi.Data(Path(c['data_root'])) as data:
                return image_tensor(data.image(c['source'][index]))
    count = [0]
    def unchanged(index, layer, values):
        if layer == LAYERS[-1]:
            count[0] += 1
            if count[0] % 10 == 0 or count[0] == 455:
                print(f'[witness fresh Source] {count[0]}/455 extracted', flush=True)
        return values
    # Explicit disk spool avoids retaining the full 455 RGB tensors or the
    # ~121 GiB four-layer candidates in RAM. No memory/disk admission gate.
    with torch.inference_mode():
        result = build_official_memory_spooled(model, Images(), device=device,
            scratch_dir=args.cache_root/'source_candidates', storage_backend='disk',
            return_selected_indices=True, feature_transform=unchanged)
    arrays = {f'bank_{l}':result.banks[l].detach().cpu().numpy() for l in LAYERS}
    arrays.update({f'selected_{l}':result.selected_candidate_indices[l] for l in LAYERS})
    if any(arrays[f'bank_{l}'].shape != (100000,1280) for l in LAYERS):
        raise ValueError('original Source bank budget/feature dimension differs')
    record = st.seal(path, identity, arrays, dict(builder=result.metadata,
        ordered_source_images=521, candidates_images=455, calibration_images=66,
        preprocessing='unchanged original augmented_preprocessing',
        sampler='unchanged original distance based, 100 rounds, normalize=False'), compressed=False)
    return dict(status='source_bank_rebuilt', sha256=record['sha256'],source_forward_calls=455)


def load_source(args, device):
    import torch
    a, _ = st.arrays(args.cache_root/'source_bank.npz', fi.setup_identity(args.run_root))
    if any(a[f'bank_{l}'].shape!=(100000,1280) or a[f'bank_{l}'].dtype!=np.float32 for l in LAYERS):
        raise ValueError('fresh Source bank shape/dtype differs')
    return {l:torch.from_numpy(a[f'bank_{l}']).to(device) for l in LAYERS}


def calibration(args):
    identity = fi.setup_identity(args.run_root)
    c = catalog(args)
    rows = [r for r in c['source'] if r['calibration']]
    model, device = None, None
    calibration_forward_calls = 0
    from DINOv3.relation_reliability.calibration import fit_calibrators
    raw_values, metas = [], []
    for i, row in enumerate(rows, 1):
        path = args.cache_root/'calibration'/f'{i:03d}.npz'
        if not st.restore(path, identity):
            if model is None:
                model, device = host(args)
                model.prototype_embeddings = load_source(args, device)
            from DINOv3.relation_reliability.superadd_host import observe_pil
            with fi.Data(Path(c['data_root'])) as data:
                observed = observe_pil(model, data.image(row), device=device)
            calibration_forward_calls += 1
            raw = observed.raw_distances.astype(np.float32)
            st.seal(path, identity, dict(raw=raw), dict(relative_path=row['relative_path'],
                grid=list(raw.shape[1:]), output_shape=[row['image_height']//4,row['image_width']//4]))
        arrays, record = st.arrays(path, identity)
        if record['metadata']['relative_path'] != row['relative_path']:
            raise ValueError('calibration order changed')
        raw_values.append(arrays['raw']); metas.append(record['metadata'])
        print(f'[witness fresh calibration] {i}/66', flush=True)
    pooled = {l:np.concatenate([r[off].ravel() for r in raw_values]) for off,l in enumerate(LAYERS)}
    _, mad = fit_calibrators(pooled)
    path = args.cache_root/'source_patch_mad.npz'
    if st.restore(path, identity):
        previous, _ = st.arrays(path, identity)
        if not np.array_equal(previous['medians'],mad.medians) or not np.array_equal(previous['scales'],mad.scales):
            raise ValueError('Source patch calibration changed on resume')
    else:
        st.seal(path,identity,dict(medians=mad.medians,scales=mad.scales),dict(images=66))
    image_rows, peaks, locals_ = [], [], []
    for row, raw, meta in zip(rows, raw_values, metas):
        _, peak, local = statistics(raw.reshape(4,-1),np.arange(raw[0].size,dtype=np.int64),
            tuple(meta['grid']),tuple(meta['output_shape']),mad.medians,mad.scales)
        peaks.append(peak); locals_.append(local)
        image_rows.append(dict(relative_path=row['relative_path'],
            **{f'peak_L{l}':float(peak[j]) for j,l in enumerate(LAYERS)},
            **{f'local_L{l}':float(local[j]) for j,l in enumerate(LAYERS)}))
    state = dict(origin='fresh_Source_only_66', layers=list(LAYERS), calibration_images=66,
                 definition='0.5*calibrated_interpolated_peak_mean4+0.5*calibrated_native_valid3x3_mean4')
    for name, values in (('peak',peaks),('local',locals_)):
        median, scale = fit_image_mad(values)
        state[name] = dict(medians=median.tolist(),scales=scale.tolist())
    state['state_sha256'] = st.digest(state)
    destination = args.run_root/'image_calibration_state.json'
    if destination.exists() and st.read_json(destination) != state:
        raise ValueError('fresh image MAD changed on resume')
    st.write_json(destination,state)
    st.write_csv(args.run_root/'calibration_per_image.csv',image_rows)
    return dict(status='Source_patch_and_image_calibration_complete', images=66,
                calibration_forward_calls=calibration_forward_calls)


def baseline_scores(raw, meta, medians, scales, state):
    scores = {}
    for b, values in zip(BASES,(raw['union_distance'],raw['segment_distance'])):
        current, peak, local = statistics(values,raw['positions'],tuple(meta['grid']),
                                          tuple(meta['output_shape']),medians,scales)
        scores[b] = dict(CH1=ch1_score(peak,local,state),CURRENT=current)
    return scores


def capture(args):
    """Same full-image extraction and Source/eight-support NN calls as parent."""
    if (args.run_root/'capture_manifest.json').exists():
        bundle = fi.Bundle(args.run_root)
        for sample in bundle.samples:
            bundle.load(sample)
        return dict(status='capture_complete',images=len(bundle.samples),query_forward_calls=0,
                    support_forward_calls=0,reused_query_caches=len(bundle.samples),reused_support_features=24)
    identity = fi.setup_identity(args.run_root)
    c = catalog(args)
    patch, _ = st.arrays(args.cache_root/'source_patch_mad.npz',identity)
    state = st.read_json(args.run_root/'image_calibration_state.json')
    model, device = host(args)
    model.prototype_embeddings = load_source(args,device)
    import torch
    from DINOv3.MADEqual.aebad_s_target_fewshot.replay12 import PROTOCOL_ID
    from DINOv3.MADEqual.robustad_target_fewshot_p0.scoring import observe_fewshot_pil
    from DINOv3.ShiftADD.aebad_s_p0.memory import _extract_four_layers
    from DINOv3.MADEqual.aebad_s_union_support_loo.analysis import CaptureLog, FORMAT, validate_arrays
    from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.method import add_geometry, scatter
    records = []
    counters = dict(query_forward_calls=0,support_forward_calls=0,reused_query_caches=0,reused_support_features=0)
    with fi.Data(Path(c['data_root'])) as data:
        for d in fi.DOMAINS:
            observed = None
            features, support_meta = [], []
            for row in c['supports'][d]:
                path = args.cache_root/'supports'/d/f'{row["support_index"]:03d}.npz'
                if not st.restore(path,identity):
                    layers, grid = _extract_four_layers(model,data.image(row),device=device,brightness_factor=None)
                    counters['support_forward_calls'] += 1
                    st.seal(path,identity,dict(protocol_id=np.asarray(PROTOCOL_ID),relative_path=np.asarray(row['relative_path']),
                        grid=np.asarray(grid,np.int32),**{f'layer_{l}':layers[l] for l in LAYERS}),
                        dict(relative_path=row['relative_path'],grid=list(grid)),compressed=False)
                else:
                    counters['reused_support_features'] += 1
                arrays, record = st.arrays(path,identity)
                grid = arrays['grid'].tolist()
                if str(arrays['relative_path'].item()) != row['relative_path']:
                    raise ValueError('support image order changed')
                features.append({l:arrays[f'layer_{l}'] for l in LAYERS})
                support_meta.append(dict(row,grid=grid,feature_sha256=record['sha256']))
                print(f'[witness fresh support] {d}/{row["support_index"]}',flush=True)
            target = {l:[torch.from_numpy(f[l]).to(device) for f in features] for l in LAYERS}
            # geometry_layer uses [layer][support] original rows, as add_geometry expects.
            geometry_features = {l:[f[l] for f in features] for l in LAYERS}
            selected = [s for s in c['samples'] if s['domain']==d]
            for i,sample in enumerate(selected,1):
                file = f'capture/{d}/{sample["sample_id"]}.npz'
                path = args.cache_root/file
                if not st.restore(path,identity):
                    log = CaptureLog()
                    observed = observe_fewshot_pil(model,data.image(sample),official_banks=model.prototype_embeddings,
                        source_support=None,target_support=target,device=device,
                        nearest_observer=log.nearest,geometry_observer=log.geometry)
                    counters['query_forward_calls'] += 1
                    raw = log.arrays()
                    if not np.array_equal(scatter(raw['source_distance'],raw['positions'],log.grid),observed.official_raw):
                        raise ValueError('same-call Source capture mismatch')
                    union = np.minimum(raw['source_distance'],raw['target_distance'].min(0))
                    if not np.array_equal(scatter(union,raw['positions'],log.grid),observed.target_raw[8]):
                        raise ValueError('same-call K8 capture mismatch')
                    raw, diagnostics = add_geometry(raw,geometry_features)
                    raw['union_distance'] = union
                    meta = dict(format=FORMAT,domain=d,sample_id=sample['sample_id'],relative_path=sample['relative_path'],
                        image_sha256=sample['image_sha256'],grid=list(log.grid),output_shape=[sample['image_height']//4,sample['image_width']//4],
                        source_rows=[100000]*4,supports=support_meta,distance_unit='euclidean_over_1280_once',
                        geometry_diagnostics=diagnostics,GT_used=False,query_forward_calls=1)
                    validate_arrays(raw,meta)
                    meta['scores'] = baseline_scores(raw,meta,patch['medians'],patch['scales'],state)
                    st.seal(path,identity,raw,meta)
                else:
                    counters['reused_query_caches'] += 1
                arrays, record = st.arrays(path,identity)
                meta = record['metadata']
                validate_arrays(arrays,meta)
                records.append(dict(domain=d,sample_id=sample['sample_id'],file=file,sha256=record['sha256'],
                    record_sha256=st.sha(path.with_suffix('.json')),scores=meta['scores']))
                print(f'[witness fresh capture] {d} {i}/{len(selected)} {sample["sample_id"]}',flush=True)
            del features,target,geometry_features,arrays,observed
            gc.collect(); torch.cuda.empty_cache()
    if len(records)!=926:
        raise ValueError('incomplete fresh capture')
    value = dict(identity=identity,status='capture_complete',records=records,GT_used=False,
                 query_images=926,support_images=24,query_features_persisted=False)
    path = args.run_root/'capture_manifest.json'
    if path.exists():
        if st.sha(path)!=st.read_json(args.run_root/'capture_manifest.seal.json')['sha256'] or st.read_json(path)!=value:
            raise ValueError('prior complete capture manifest modified/different')
    else:
        st.write_json(path,value)
        st.write_json(args.run_root/'capture_manifest.seal.json',dict(sha256=st.sha(path)))
    return dict(status='capture_complete',images=926,**counters)
