"""Explicit original host extraction. Never imported by default preflight."""
from __future__ import annotations

import io
import shutil
import time
from types import SimpleNamespace
import numpy as np
from . import inputs as ip, scheduler
from .selection import support_pool, DOMAINS
from .storage import npz_bytes, json_bytes, load_npz, sha
from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.verification import historical_score
from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import statistics


def sample_key(sample): return sample['domain']+'/'+sample['sample_id']


def audit_scores(ctx, sample, arrays, meta):
    """Historical Union applies ONLY to seed42. Internal precision stays separate."""
    union = np.minimum(arrays['source_distance'], arrays['target_distance'][meta['mapping']['42']].min(0))
    rows = []
    for kind, value, field in (('SOURCE_CURRENT', arrays['source_distance'], 'source_score'),
                               ('UNION_CURRENT', union, 'union_score')):
        current = statistics(value, arrays['positions'], tuple(meta['grid']), tuple(meta['output_shape']),
                             ctx['medians'], ctx['scales'])[0]
        historical_score(rows, sample, kind, current, sample[field])
    if not all(r['verified'] for r in rows):
        error = ValueError('seed42 historical CURRENT exceeds fixed compatibility atol1e-4')
        error.checks = rows
        raise error
    return rows


def load_raw(ctx, sample):
    ledger = ctx['ledger']; r = ledger.get('capture/'+sample_key(sample))
    if r is None: raise FileNotFoundError('missing shared query capture: '+sample_key(sample))
    arrays, meta = load_npz(ledger.root/r['file'], r['files'][r['file']])
    scheduler.validate_shared(ctx, sample, arrays, meta)
    return arrays, meta, r


def _support_name(sample): return sample['domain']+'/'+sample['sample_id']


def run(ctx, domains, device_name='cuda:0', minimum_vram_gib=20):
    ledger = ctx['ledger']; ledger.verify()
    if any(k.startswith('rejected_capture/') and k.split('/')[1] in domains for k in ledger.state['records']):
        raise ValueError('previous capture failed numerical audit; rejected evidence retained, no re-approval')
    pending = [s for s in ctx['plan']['samples'] if s['domain'] in domains
               and not ledger.get('capture/'+sample_key(s))]
    if not pending:
        return dict(new_query_extractions=0,new_support_extractions=0,reused_support_features=0,
                    Source_Exact_calls=0,Target_Exact_calls=0,
                    complete_query_reuses=sum(s['domain'] in domains for s in ctx['plan']['samples']))
    paths = ctx['paths']
    # Estimate the remaining uncompressed products, NOT a promise of compression.
    # Reserve compact predictions too so extraction cannot consume their space.
    support_bytes=0
    for domain in domains:
        if not any(s['domain']==domain for s in pending):continue
        for s in support_pool(ctx['plan'],domain)[0]:
            if not ledger.get('support/'+_support_name(s)):
                n=(int(s['image_height']*.625)//16)*(int(s['image_width']*.625)//16)
                support_bytes+=n*4*1280*4
    query_patches=sum((int(s['image_height']*.625)//16)*(int(s['image_width']*.625)//16) for s in pending)
    required_disk=support_bytes+query_patches*(4*12+24*4*12+3*4*54+3*4*48)+(4<<30)
    if shutil.disk_usage(ledger.root).free<required_disk:
        raise OSError(f'capture needs conservative {required_disk} free bytes for remaining full supports/raw/predictions; no compression assumption')
    for role in ('memory', 'checkpoint'):
        if not paths[role].is_file() or sha(paths[role]) != ctx['assets'][role+'_sha256']:
            raise FileNotFoundError('original '+role+' missing/mismatched; rebuilding forbidden')
    for role, filename in (('dinov3', 'hubconf.py'), ('official', 'config.json')):
        if not (paths[role]/filename).is_file(): raise FileNotFoundError(str(paths[role]/filename))
    # Only explicit capture crosses this boundary.
    import torch
    from DINOv3.relation_reliability.superadd_host import _extract_processed_batch, image_tensor, _exact_top1_chunked
    from DINOv3.MADEqual.aebad_s_target_fewshot.runner import _load_memory
    from DINOv3.MADEqual.aebad_s_sparc_p0.runtime import load_host
    from DINOv3.MADEqual.robustad_target_fewshot_p0.support import _extract_four_layers
    device = torch.device(device_name)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('explicit capture needs CUDA; no CPU/precision/batch fallback')
    device = torch.device('cuda', torch.cuda.current_device() if device.index is None else device.index)
    torch.cuda.set_device(device)
    if torch.cuda.mem_get_info(device)[0] < minimum_vram_gib*2**30:
        raise MemoryError('free VRAM below explicit admission; batch/precision unchanged')
    if ip.old.common.available_ram() < 12*2**30:
        raise MemoryError('capture requires 12 GiB available RAM')
    model, device = load_host(dict(paths={k:str(v) for k,v in paths.items()}, frozen=dict(
        precision=ip.old.PRECISION, host_config=ip.read_json(paths['formal_dir']/'memory_manifest.json')['host_config'])), str(device))
    holder = SimpleNamespace()
    if not _load_memory(paths['memory'], holder, source_count=521, device=device):
        raise FileNotFoundError('original Source memory missing')
    counts = dict(new_query_extractions=0, new_support_extractions=0, reused_support_features=0,
                  Source_Exact_calls=0, Target_Exact_calls=0,
                  required_free_disk_bytes=required_disk,
                  complete_query_reuses=sum(s['domain'] in domains for s in ctx['plan']['samples'])-len(pending))
    started = time.perf_counter()
    with ip.Data(paths['data']) as data, torch.inference_mode():
        for domain in domains:
            selected = [s for s in pending if s['domain']==domain]
            if not selected: continue
            pool, mapping = support_pool(ctx['plan'], domain); support_meta = []
            for sample in pool:
                key = 'support/'+_support_name(sample); record = ledger.get(key)
                if record is None:
                    tick = time.perf_counter()
                    features, grid = _extract_four_layers(model, data.image(sample), device=device, brightness_factor=None)
                    files = {}; prefix = 'support_features/'+_support_name(sample)
                    for layer in ip.LAYERS:
                        stream = io.BytesIO(); np.save(stream, features[layer], allow_pickle=False)
                        files[f'{prefix}/L{layer}.npy'] = stream.getvalue()
                    metadata = dict(relative_path=sample['relative_path'], image_sha256=sample['image_sha256'],
                        grid=[int(v) for v in grid], identity=ctx['identity'], seconds=time.perf_counter()-tick,
                        extractor='original_support._extract_four_layers_brightness_None', layers=list(ip.LAYERS))
                    files[prefix+'/metadata.json'] = json_bytes(metadata)
                    ledger.commit(key, metadata, files)
                    counts['new_support_extractions'] += 1
                    del features, files
                    print(f'[K8 sensitivity support] {domain} {sample["sample_id"]}', flush=True)
                else: counts['reused_support_features'] += 1
                r = ledger.get(key)
                support_meta.append(dict(relative_path=sample['relative_path'], image_sha256=sample['image_sha256'],
                    grid=r['grid'], feature_files=r['files']))

            def load_feature(i, layer):
                s = pool[i]
                value = np.load(ledger.root/f'support_features/{_support_name(s)}/L{layer}.npy', mmap_mode='r', allow_pickle=False)
                if value.dtype != np.float32 or value.shape != (int(np.prod(support_meta[i]['grid'])), 1280):
                    raise ValueError('support feature shape/precision differs')
                return value

            def exact(query, bank):
                d, index = _exact_top1_chunked(query, bank)
                return d.numpy(), index.numpy()

            for sample in selected:
                tick = time.perf_counter()
                pending_query,processed_shape = scheduler.extract_query(model,data.image(sample),device,
                                                                       image_tensor,_extract_processed_batch)
                torch.cuda.synchronize(device)  # truthful extraction vs retrieval timing
                extraction_seconds = time.perf_counter()-tick; tick = time.perf_counter()
                source = []; source_idx = []; target = []; target_idx = []
                for layer in ip.LAYERS:
                    def bank(i):
                        return torch.from_numpy(np.array(load_feature(i, layer), copy=True)).to(device)
                    ds, ix, dt, it = scheduler.scan_layer(pending_query.queries[layer], holder.prototype_embeddings[layer],
                                                       len(pool), bank, exact)
                    source.append(ds); source_idx.append(ix); target.append(dt); target_idx.append(it)
                    counts['Source_Exact_calls'] += 1; counts['Target_Exact_calls'] += len(pool)
                raw = dict(positions=pending_query.positions.copy(), source_distance=np.stack(source),
                    source_index=np.stack(source_idx), target_distance=np.stack(target, axis=1),
                    target_index=np.stack(target_idx, axis=1))
                grid = [int(v) for v in pending_query.grid]
                pending_query.queries.clear(); del pending_query
                retrieval_seconds = time.perf_counter()-tick; tick = time.perf_counter()
                raw, diagnostics = scheduler.attach_geometry(raw, mapping, load_feature)
                meta = dict(identity=ctx['identity'], domain=domain, sample_id=sample['sample_id'],
                    relative_path=sample['relative_path'], image_sha256=sample['image_sha256'],
                    grid=grid, processed_shape=processed_shape,
                    output_shape=[sample['image_height']//4, sample['image_width']//4], supports=support_meta,
                    mapping=mapping, distance_unit=ip.PROTOCOL['distance_unit'], geometry_diagnostics=diagnostics,
                    timing=dict(query_extraction=extraction_seconds, Exact_search=retrieval_seconds,
                                geometry=time.perf_counter()-tick), query_extraction_calls=1,
                    Source_Exact_calls=4, Target_Exact_calls=4*len(pool), GT_used=False)
                scheduler.validate_shared(ctx, sample, raw, meta)
                try:
                    checks = audit_scores(ctx, sample, raw, meta)
                except ValueError as error:
                    name = 'rejected_capture/'+sample_key(sample)+'.npz'
                    ledger.commit('rejected_capture/'+sample_key(sample),
                        dict(status='historical_audit_failed', checks=getattr(error,'checks',[])),
                        {name:npz_bytes(raw,dict(meta,status='rejected_not_trusted'))})
                    raise
                name = 'capture/'+sample_key(sample)+'.npz'
                ledger.commit('capture/'+sample_key(sample), dict(file=name, checks=checks, timing=meta['timing']),
                              {name:npz_bytes(raw, meta)})
                counts['new_query_extractions'] += 1
                print(f'[K8 sensitivity capture] {sample_key(sample)} one_query_three_support_sets', flush=True)
                del raw
    # Input files are rechecked against the ORIGINAL recorded hashes.
    ledger.verify()
    counts.update(seconds=time.perf_counter()-started, CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                  CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    return counts
