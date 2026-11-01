"""Fixed normal-image throughput, with every surrogate and unknown explicit."""
import math
import time
import numpy as np
from . import inputs as ip, runtime as rt, numerics as nm
from .inputs import st
from .protocol import *


def estimate(plan, measurements):
    """No fabricated zero for unmeasured phases; work already cached is excluded."""
    phases = {}
    for phase, units in plan.items():
        rate = measurements.get(phase)
        phases[phase] = dict(pending_units=units, seconds_per_unit=rate,
                            seconds=0. if units == 0 else None if rate is None else float(units*rate))
    unknown = [k for k, v in phases.items() if v['seconds'] is None]
    total = None if unknown else sum(v['seconds'] for v in phases.values())
    return dict(status='fixed_normal_asset_smoke' if not unknown else 'pending_unmeasured_costs',
                remaining_seconds=total, planning_seconds_range=None if total is None else [total*.5, total*3],
                unknown_phases=unknown, phases=phases, over_12h_requires_consent=total is None or total > 43200,
                interpretation='throughput extrapolation, not completion guarantee; surrogate CPU labels/AUPRO masks are timing only')


def retrieval_timing(reference, source=True):
    """Separate cold resident uploads from steady per-query Exact throughput."""
    q=reference.parts[0]
    tick=time.perf_counter();owned={l:np.array(q[l],copy=True) for l in LAYERS};feature_read=time.perf_counter()-tick
    def source_call():
        for l in LAYERS:
            query=reference.search.query(owned[l])
            reference.search.exact(query,reference.source['L'+str(l)],('Source',reference.source_seal,l))
    cold_source=hot_source=None
    if source:
        reference.load_source();tick=time.perf_counter();source_call();cold_source=time.perf_counter()-tick
    tick=time.perf_counter();reference.target_only(owned);cold_target=time.perf_counter()-tick
    if source:
        tick=time.perf_counter();source_call();hot_source=time.perf_counter()-tick
    tick=time.perf_counter();reference.target_only(owned);hot_target=time.perf_counter()-tick
    return dict(Source_four_layers_seconds=hot_source,Target_eight_supports_seconds=hot_target,
                Source_initial_upload_seconds=None if not source else max(0.,cold_source-hot_source),
                Target_group_initial_upload_seconds=max(0.,cold_target-hot_target),full_feature_read_seconds=feature_read,
                cold_Source_seconds=cold_source,cold_Target_seconds=cold_target,
                benchmark_repetition=True,normal_support_only=True,units='one_complete_normal_grid',
                note='initial uploads counted per resident bank/group, not multiplied by query count; normal feature read may be warm OS cache',search_stats=reference.search.stats)


def run(a):
    p = ip.parent(a);ip.preflight(a)
    from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.calibration import z_values
    from DINOv3.MADEqual.aebad_s_support_witness_segment_p0 import evaluation as backend
    measurements = [];all_images = [];all_native = [];read_times = [];kernel_times = [];io_times = [];backend_checks=[];distance_io=[];source_io=[]
    out = a.run_root/'smoke';out.mkdir(parents=True, exist_ok=True)
    for d in DOMAINS:
        for seed in SEEDS:
            refs = rt.normal_model(a, d, seed)
            for j in (0, 1):  # Fixed normal support order, no metric-based selection.
                tick = time.perf_counter();x, r = ip.inspect_normal(a, d, seed, j);grid = r['metadata']['grid']
                zs = nm.physical(z_values(x['Source_distance'], p.inputs['mad']['patch']), x['positions'], grid)
                zt = nm.physical(z_values(x['Target7_distance'], p.inputs['mad']['patch']), x['positions'], grid)
                read_times.append(time.perf_counter()-tick)
                tick = time.perf_counter()
                # This repeated normal K7 vector benchmarks tensor dimensions only.
                # It is NOT a formal query with eight distinct B_j, never evaluated.
                native = nm.score_native(zs, np.repeat(zt[None], 8, axis=0), refs, np.asarray(p.weights[d, 'WS'])[:, 0],
                          device=a.device, query_chunk=a.query_chunk, reference_chunk=a.reference_chunk)
                kernel_times.append(time.perf_counter()-tick)
                if a.device != 'cpu' and j == 0:
                    small_refs=[(rs[:16],tr[:16]) for rs,tr in refs]
                    small_s=zs[:,:16];small_t=np.repeat(zt[None,:,:16],8,axis=0)
                    cpu=nm.score_native(small_s,small_t,small_refs,np.asarray(p.weights[d,'WS'])[:,0],device='cpu')
                    gpu=nm.score_native(small_s,small_t,small_refs,np.asarray(p.weights[d,'WS'])[:,0],device=a.device)
                    for b in MAPS:
                        ok=bool(np.allclose(cpu[b],gpu[b],atol=1e-6,rtol=1e-6))
                        backend_checks.append(dict(domain=d,support_seed=seed,branch=b,check='small_actual_normal_CPU_vs_stock_CUDA',
                                                   max_abs_error=float(np.abs(cpu[b].astype(float)-gpu[b]).max()),atol=1e-6,rtol=1e-6,verified=ok))
                        if not ok:raise ValueError('stock CUDA tail kernel differs from small CPU reference; no tolerance relaxation')
                row = p.common['domains'][d]['supports'][str(seed)][j]
                images = nm.maps(native, grid, [row['image_height']//4, row['image_width']//4])
                tick = time.perf_counter()
                # A distinct timing payload per invocation; never masquerades as prediction.
                folder = out/('payload_'+str(time.time_ns()));st.put(folder, native, dict(timing_only=True, formal_prediction=False))
                io_times.append(time.perf_counter()-tick)
                if j==0:
                    timing='timing_only_'+str(time.time_ns())
                    tick=time.perf_counter()
                    st.put(out/('distance_'+timing),dict(Source_distance=x['Source_distance'],target_distance=np.repeat(x['Target7_distance'][:,None],8,axis=1),positions=x['positions']),
                           dict(timing_only=True,contract=ip.split_contract(a,d,seed,row),formal_prediction=False))
                    distance_io.append(time.perf_counter()-tick)
                    tick=time.perf_counter()
                    st.put(out/('source_'+timing),dict(Source_distance=x['Source_distance'],positions=x['positions']),
                           dict(timing_only=True,contract=ip.source_contract(a,row),formal_prediction=False))
                    source_io.append(time.perf_counter()-tick)
                all_images.append(images[JOINT]);all_native.append(native)
                measurements.append(dict(domain=d, support_seed=seed, support_index=j, sample=row['relative_path'], query_positions=len(zs.T),
                                         read_seconds=read_times[-1], four_maps_seconds=kernel_times[-1], IO_seconds=io_times[-1],
                                         input='actual normal Source/K7 responses; same K7 repeated eight ways for throughput only', GT_used=False))
    # Exact CPU implementation, but generated labels to obey GT-before-freeze rule.
    # These scores are explicitly discarded; only time and NlogN scaling are used.
    images = all_images[:2];values = np.concatenate([x.ravel() for x in images]).astype(np.float32)
    labels = (np.arange(len(values)) % 17 == 0).astype(np.uint8)
    tick = time.perf_counter()
    with backend.observation_only():
        backend.pixels.exact_binary(labels, values, memory_budget_bytes=0)
    cpu_seconds = time.perf_counter()-tick
    required = ip.audit.parent_reporting.rows(a.run_root/'missing_inputs.csv')
    work = ip.audit.parent_reporting.rows(a.run_root/'necessary_work.csv')
    missing_entries=[r for r in required if r.get('role')=='query_split_distances']
    missing_split=len(missing_entries)
    source_queries = {(r['domain'], r['sample_id']) for r in work if r.get('Source_search_required') == 'True'}
    forwards = {(r['domain'], r['sample_id']) for r in work if r.get('role') == 'query_features'}
    query_images={(r['domain'],r['sample_id']) for r in missing_entries}
    target_groups={(r['domain'],r['support_seed']) for r in missing_entries}
    retrieval = None;forward = None;decode=None;model_startup = 0.;au_seconds = None
    if a.benchmark_retrieval:
        tick=time.perf_counter();reference = rt.References(a, DOMAINS[0], SEEDS[0]);construct=time.perf_counter()-tick
        retrieval=retrieval_timing(reference,source=bool(source_queries))
        retrieval['reference_metadata_prepare_seconds']=construct
        del reference
    if a.benchmark_forward:
        from DINOv3.MADEqual.aebad_s_same_origin_neighbor_p0 import feature_identity as fi
        from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.runtime import host, extraction, reliable_precision
        from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.fresh_inputs import Data
        fi.compatible(fi.native_extraction(a), p.inputs['extraction']);reliable_precision(a.device)
        tick = time.perf_counter();model = host(a)[0];model_startup = time.perf_counter()-tick
        row = p.common['domains'][DOMAINS[0]]['supports']['42'][0]
        with Data(a.data) as data:
            tick=time.perf_counter();image = data.image(row);decode=time.perf_counter()-tick
            tick = time.perf_counter();extraction(model, image, a.device);forward = time.perf_counter()-tick
        del model
    if a.benchmark_aupro:
        from importlib.metadata import version
        if version('adeval') != '1.1.0':raise ValueError('original ADEval1.1.0 required')
        import torch
        from DINOv3.relation_reliability.official_metrics import official_aupro_adeval
        torch.cuda.set_device(torch.device(a.device));masks = []
        for image in images:
            mask = np.zeros(image.shape, np.uint8);h, w = mask.shape;mask[h//3:h//2, w//3:w//2] = 1;masks.append(mask)
        tick = time.perf_counter();official_aupro_adeval(masks, images, nstrips=200);au_seconds = time.perf_counter()-tick
    pending_predict = sum(not ip.record_path(a, d, seed, row).exists() for d in DOMAINS for seed in SEEDS for row in p.samples(d))
    pending_cpu = sum(not (ip.unit(a, d, seed)/'evaluation'/b/'status.json').exists() for d in DOMAINS for seed in SEEDS for b in MAPS)
    pending_au = sum(not (ip.unit(a, d, seed)/'aupro'/(b+'.json')).exists() for d in DOMAINS for seed in SEEDS for b in MAPS)
    mean_pixels = np.mean([sum((r['image_height']//4)*(r['image_width']//4) for r in p.samples(d)) for d in DOMAINS])
    pixels_per_smoke = len(values);cpu_job = cpu_seconds*(mean_pixels/pixels_per_smoke)*math.log2(mean_pixels)/math.log2(pixels_per_smoke)
    au_job = None if au_seconds is None else au_seconds*np.mean(list(COUNTS[d][0] for d in DOMAINS))/len(images)
    plan = dict(query_forwards=len(forwards), Source_retrieval=len(source_queries), Target_retrieval=missing_split,
                Source_resident_startup=int(bool(source_queries)),Target_resident_startup=len(target_groups),
                cached_query_feature_IO=len(query_images-forwards),raw_image_IO=len(forwards),split_distance_IO=missing_split,
                shared_Source_IO=sum(not (ip.source_path(a,d,next(r for r in p.samples(d) if r['sample_id']==sid))/'manifest.json').exists() for d,sid in query_images),
                four_native_maps=pending_predict, exact_CPU_jobs=pending_cpu, AUPRO_jobs=pending_au, native_IO=pending_predict,
                model_startup=1 if forwards else 0, fixed_case_plots=12, final_pack=1)
    tick = time.perf_counter()
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5), layout='constrained')
    for ax in axes:ax.imshow(images[0], cmap='magma');ax.axis('off')
    fig.savefig(out/'timing_case.png', dpi=100);plt.close(fig)
    case_seconds = time.perf_counter()-tick
    # A small actual review-shaped bundle measures pack throughput; no old arrays.
    import tarfile
    tick = time.perf_counter()
    with tarfile.open(out/'timing_payload.tar.gz', 'w:gz', compresslevel=1) as archive:
        for path in sorted(out.glob('payload_*'))[:2]:archive.add(path, arcname=path.name)
    sample_pack = time.perf_counter()-tick
    rates = dict(query_forwards=forward, Source_retrieval=None if retrieval is None else retrieval['Source_four_layers_seconds'],
                 Target_retrieval=None if retrieval is None else retrieval['Target_eight_supports_seconds'],
                 Source_resident_startup=None if retrieval is None else retrieval['Source_initial_upload_seconds'],
                 Target_resident_startup=None if retrieval is None else retrieval['Target_group_initial_upload_seconds']+retrieval['reference_metadata_prepare_seconds'],
                 cached_query_feature_IO=None if retrieval is None else retrieval['full_feature_read_seconds'],raw_image_IO=decode,
                 split_distance_IO=float(np.mean(distance_io)),shared_Source_IO=float(np.mean(source_io)),
                 four_native_maps=float(np.mean(read_times)+np.mean(kernel_times)), exact_CPU_jobs=cpu_job, AUPRO_jobs=au_job,
                 native_IO=float(np.mean(io_times)), model_startup=model_startup if a.benchmark_forward else None, fixed_case_plots=case_seconds,
                 final_pack=sample_pack*12)  # Declared 12-case report payload planning multiplier.
    result = estimate(plan, rates)
    st.write_csv(out/'numerical_backend_checks.csv',backend_checks)
    result.update(scope='fixed_normal_asset_smoke', rows=measurements, retrieval=retrieval, forward_seconds=forward,image_decode_seconds=decode, model_startup_seconds=model_startup,
                  numerical_backend_checks=backend_checks,
                  CPU_sort=dict(seconds=cpu_seconds, sample_pixels=len(values), labels='constructed periodic labels; throughput only, discarded metrics'),
                  AUPRO=dict(sample_seconds=au_seconds, masks='constructed rectangle; throughput only, discarded metric', nstrips=200, FPR_max=.3),
                  plan=plan, formal_metrics=False, new_disk_prediction_bytes=int(sum(v.nbytes for v in all_native[0].values())*pending_predict),
                  limitations=['normal grid throughput extrapolated to test images', 'CPU class distribution and AUPRO component count use surrogate masks',
                               'pack factor assumes 12 small frozen cases; other report overhead unmeasured', 'not a runtime guarantee'])
    st.write_json(a.run_root/'time_estimate.json', result);st.write_json(out/'smoke_report.json', result)
    return dict(status=result['status'], remaining_seconds=result['remaining_seconds'], unknown_phases=result['unknown_phases'], GT_used=False)
