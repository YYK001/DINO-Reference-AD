"""Explicit reconstruction of missing Source banks. No Torch at import time.

The original builder and its RNG stream are used without changing sampling.
Reconstructed banks have their own provenance; old distances are not adopted.
"""
from __future__ import annotations
import gc
import platform
import shutil
import sys
import threading
import time
from pathlib import Path
import numpy as np
from PIL import Image
from . import inputs as ip

FORMAT = 'robustad_source_reconstruction_v1'
ENTRIES = 100000
DIMENSION = 1280
BUILDER = 'DINOv3.relation_reliability.superadd_host.build_exact_official_memory'


def directory(paths, category):
    if paths.get('rebuilt_source_root') is None:
        raise ValueError('explicit --rebuilt-source-root required; original Source tree is read-only')
    root = Path(paths['rebuilt_source_root']).resolve()
    for key in ('source_root', 'data', 'official', 'dinov3', 'checkpoint',
                'formal_metalparts', 'formal_pcb', 'output_root'):
        other = Path(paths[key]).resolve()
        if root == other or root in other.parents or other in root.parents:
            raise ValueError('rebuilt Source root overlaps input or experiment output: ' + key)
    if root.drive and root.drive.upper() != 'G:':
        raise ValueError('local reconstruction outputs must stay on G:')
    return root / 'categories/robustad' / category


def plan(ctx):
    """Frozen full ordered input; the ORIGINAL builder performs mod8 exclusion."""
    from .runner import estimated_grid
    samples = ctx['cat']['train']
    if (len(samples) != 180 or [s['source_index'] for s in samples] != list(range(180))
            or any(s['label'] or s['source_role'] != ('calibration' if i % 8 == 0 else 'memory_fit')
                   for i, s in enumerate(samples))
            or [s['sample_id'] for s in samples if s['source_index'] % 8 == 0]
            != [s['sample_id'] for s in ctx['cat']['calibration']]):
        raise ValueError('original ordered 180/157/23 Source split differs')
    old = ip.read_json(ctx['source'] / 'memory/memory_provenance.json')['official_builder']
    if (old['sampler'] != 'upstream_subsampling_distance_based_fast'
            or old['sampler_iterations'] != 100 or old['memory_budget_per_layer'] != ENTRIES
            or old['sampler_distance_backend'] != 'exact_query_chunked_torch_cdist_topk'
            or old['sampler_distance_matrix_max_bytes'] != 1 << 30):
        raise ValueError('historical sampler configuration differs')
    blocks = []; offset = 0
    for s in samples:
        if s['source_index'] % 8 == 0:
            continue
        grid = estimated_grid(s); count = int(np.prod(grid))
        blocks.append(dict(sample_id=s['sample_id'], source_index=s['source_index'],
                           grid=list(grid), start=offset, stop=offset + count))
        offset += count
    if any(old['layer_audits'][str(l)]['candidate_count'] != offset
           or old['layer_audits'][str(l)]['memory_entries'] != ENTRIES for l in ip.LAYERS):
        raise ValueError('estimated candidate geometry differs from original provenance')
    return dict(format=FORMAT, category=ctx['category'], catalog_sha256=ip.CATALOG_SHA,
                code=ip.code_hashes(), builder=BUILDER, seed=42, precision=ip.PRECISION,
                checkpoint_sha256=ip.CHECKPOINT_SHA, host_config=ctx['catalog']['host_config'],
                source_metadata_sha256=ctx['cat']['source_sha256'], train=samples,
                split='builder_receives_all180_and_excludes_index_mod8_eq0',
                candidate_blocks=blocks, candidate_rows=offset, entries_per_layer=ENTRIES,
                dimension=DIMENSION, layers=list(ip.LAYERS), sampler=old['sampler'],
                sampler_iterations=100, sampler_distance_backend=old['sampler_distance_backend'],
                sampler_distance_matrix_max_bytes=1 << 30,
                historical_bank_byte_sha256=None, original_patch_MAD_refitted=False)


def resources(p):
    candidates = p['candidate_rows'] * 4 * DIMENSION * 4
    images = sum(s['image_height'] * s['image_width'] * 3 * 4 for s in p['train'])
    # All tensors are retained by the ORIGINAL builder, not a lazy replacement.
    return dict(candidate_float32_bytes=candidates, original_train_tensor_bytes=images,
                new_bank_bytes=4 * ENTRIES * DIMENSION * 4,
                minimum_available_ram_bytes=candidates + images + (20 << 30),
                minimum_disk_free_bytes=(4 * ENTRIES * DIMENSION * 4) + (4 << 30),
                storage='original_auto_RAM_or_disk; admission reserves full RAM plus work space',
                interpretation='conservative estimates, sequential categories; not measured peaks')


def validate_arrays(banks, indices, candidate_rows, *, entries=ENTRIES, dimension=DIMENSION):
    """Validate actual builder output; smaller dimensions only used by unit fixtures."""
    if set(banks) != set(ip.LAYERS) or set(indices) != set(ip.LAYERS):
        raise ValueError('reconstructed layers differ')
    for layer in ip.LAYERS:
        a = banks[layer]; ix = indices[layer]
        if a.dtype != np.float32 or a.shape != (entries, dimension) or not np.isfinite(a).all():
            raise ValueError('invalid reconstructed Source bank')
        if (ix.dtype != np.int64 or ix.shape != (entries,) or np.unique(ix).size != entries
                or np.any(ix < 0) or np.any(ix >= candidate_rows)):
            raise ValueError('invalid selected original candidate indices')


def _legacy_plan_compatible(prior, current):
    """Narrow 96d4981 bank reuse after fixing calibration ONLY, never raw reuse.

    Older plans hashed the entire tool, including unrelated observation code.
    Require the known producer, identical non-code inputs and all other code,
    and unchanged generation ASTs. Do not rewrite the producer's manifest.
    """
    import ast
    import hashlib
    if prior == current:
        return True
    if ({k:v for k,v in prior.items() if k!='code'}
            != {k:v for k,v in current.items() if k!='code'}):
        return False
    old=prior.get('code',{}); new=current.get('code',{})
    if ip.digest(old)!='48fc11a1af3db200900249338945457e9fe9ebff9f4b12d2976643c3005346a1':
        return False
    prefix='DINOv3/MADEqual/robustad_segment_pair_crossdataset_p0/'
    changed={prefix+'runtime.py',prefix+'rebuild.py'}
    if set(old)!=set(new) or any(old[k]!=new[k] for k in old if k not in changed):
        return False
    contracts=(('runtime.py',{'source_observation'},
                '9b2aff23503c53138aebe6a167c7983cc824cf3b5f1ff9e2ec6ab815cf7b3364'),
               ('rebuild.py',{'verify','snapshot','_legacy_plan_compatible'},
                'c1de8ed14f5d42e44f42789e86d33bc185f7112bd77434eb93143fcf39f401a0'))
    for filename,skip,expected in contracts:
        tree=ast.parse(Path(__file__).with_name(filename).read_text(encoding='utf-8-sig'))
        tree.body=[n for n in tree.body if not isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) or n.name not in skip]
        # Python3.12 adds empty type_params fields absent from Python3.11.
        # Remove only that empty schema addition, never nonempty declarations.
        for node in ast.walk(tree):
            if getattr(node,'type_params',None)==[]:
                node._fields=tuple(k for k in node._fields if k!='type_params')
        if hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()!=expected:
            return False
    return True


def verify(paths, category, expected_plan=None):
    root = directory(paths, category); manifest = ip.read_json(root / 'rebuild_status.json')
    if (manifest.get('status') != 'bank_reconstructed'
            or manifest.get('format') != FORMAT or manifest.get('category') != category
            or manifest.get('bank_origin') != 'explicit_reconstruction_original_builder'
            or ip.digest(manifest['plan']) != manifest.get('plan_identity')
            or (expected_plan is not None and not _legacy_plan_compatible(manifest['plan'],expected_plan))):
        raise ValueError('reconstruction identity incomplete or changed; use a new reconstruction root')
    if manifest['files'].keys() != {'memory/official_banks.npz', 'memory/selected_candidate_indices.npz'}:
        raise ValueError('incomplete reconstruction file registry')
    for name, checksum in manifest['files'].items():
        if ip.sha(root / name) != checksum:
            raise ValueError('recorded reconstructed Source file changed; preserve history')
    return manifest


def snapshot(paths, category, expected_plan=None):
    root = directory(paths, category)
    if not root.exists():
        return dict(bank_origin='explicit_reconstruction_pending', rebuild_manifest_sha256=None)
    manifest=verify(paths, category, expected_plan)
    return dict(bank_origin='explicit_reconstruction_original_builder',
                rebuild_manifest_sha256=ip.sha(root / 'rebuild_status.json'),
                generation_compatibility='96d4981_unchanged_bank_generation' if expected_plan is not None and manifest['plan']!=expected_plan else 'exact_plan')


def ordered_records(ctx):
    """The real production Record class, no pre-exclusion or re-sorting."""
    from DINOv3.relation_reliability.datasets import Record
    records = []
    for s in ctx['cat']['train']:
        path = ctx['paths']['data'] / ip.safe(s['relative_path'])
        if ip.sha(path) != s['image_sha256']:
            raise ValueError('Source training image bytes changed: ' + str(path))
        with Image.open(path) as im:
            if im.size != (s['image_width'], s['image_height']):
                raise ValueError('Source training image dimensions changed')
        row = dict(ctx['cat']['legacy_records'][s['relative_path']])
        row['path'] = str(path); row['mask_path'] = None
        records.append(Record(**row))
    return records


def storage_admission(original_choice, scratch, candidate_bytes, **kwargs):
    """Preserve original auto decision; reject unsafe disk fallback before writes."""
    decision = original_choice(candidate_bytes, **kwargs)
    if decision['backend'] != 'ram':
        parent = Path(scratch)
        while not parent.exists():
            parent = parent.parent
        required = int(candidate_bytes) + 4 * ENTRIES * DIMENSION * 4 + (4 << 30)
        if shutil.disk_usage(parent).free < required:
            raise MemoryError('original auto selected disk but full candidate scratch will not fit; no reduced bank')
    return decision


def original_build(ctx, records, device_name, min_free_gib, scratch):
    """Only this explicit stage can call a builder; imports are deliberately lazy."""
    import torch
    from unittest.mock import patch
    from .runtime import load_model
    from external_baselines.superadd_external import memory_variants
    from DINOv3.relation_reliability.superadd_host import build_exact_official_memory
    model = None; build = None
    try:
        # load_model seeds42 BEFORE construction as the original v6 runner did.
        # Do NOT reseed after construction: that would alter sampler choices.
        model, device = load_model(ctx, device_name, min_free_gib)
        if (model.threshold_fraction != 8 or model.max_database_size != ENTRIES
                or list(model.layers) != list(ip.LAYERS)):
            raise ValueError('constructed Source builder configuration differs')
        original_choice = memory_variants.choose_candidate_storage_backend
        def checked(candidate_bytes, **kwargs):
            return storage_admission(original_choice, scratch, candidate_bytes, **kwargs)
        with torch.inference_mode(), patch.object(memory_variants, 'choose_candidate_storage_backend', checked):
            build = build_exact_official_memory(model, records, device=device, scratch_dir=scratch)
        arrays = {l: build.banks[l].detach().float().cpu().numpy() for l in ip.LAYERS}
        indices = {l: np.asarray(build.selected_candidate_indices[l]).copy() for l in ip.LAYERS}
        peaks = dict(peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                     peak_cuda_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
                     device=str(device), device_name=torch.cuda.get_device_name(device),
                     cuda_version=torch.version.cuda)
        return arrays, indices, build.metadata, peaks
    finally:
        model = None; build = None
        gc.collect(); torch.cuda.empty_cache()


def run(ctx, device_name, min_free_gib=20):
    started = time.perf_counter(); p = plan(ctx)
    root = directory(ctx['paths'], ctx['category'])
    if root.exists():
        return verify(ctx['paths'], ctx['category'], p)
    estimate = resources(p)
    disk = root.parent
    while not disk.exists():
        disk = disk.parent
    if (ip.available_ram() < estimate['minimum_available_ram_bytes']
            or shutil.disk_usage(disk).free < estimate['minimum_disk_free_bytes']):
        raise MemoryError('insufficient RAM/disk for original complete Source reconstruction; no reduced bank')
    records = ordered_records(ctx)
    root.mkdir(parents=True)
    ip.write_json(root / 'rebuild_plan.json', p)
    ip.write_json(root / 'rebuild_status.json', dict(format=FORMAT, category=ctx['category'],
                  status='rebuilding', plan_identity=ip.digest(p), plan=p))
    stopped = threading.Event()
    def heartbeat():
        while not stopped.wait(30):
            print(f'[Source rebuild heartbeat] {ctx["category"]} elapsed={time.perf_counter()-started:.0f}s; original builder/save in progress', flush=True)
    watcher = threading.Thread(target=heartbeat, daemon=True); watcher.start()
    try:
        print(f'[Source rebuild] {ctx["category"]}: original180 / memory157 / calibration23; 100K per layer', flush=True)
        banks, indices, metadata, peaks = original_build(ctx, records, device_name, min_free_gib, root / 'scratch')
        validate_arrays(banks, indices, p['candidate_rows'])
        for layer in ip.LAYERS:
            if metadata['layer_audits'][str(layer)]['candidate_count'] != p['candidate_rows']:
                raise ValueError('actual candidate geometry differs from original provenance')
        with np.load(ctx['source'] / 'memory/anchor_sidecars.npz', allow_pickle=False) as old:
            index_audit = {str(l): dict(exact_order_equal=bool(np.array_equal(indices[l], old[f'physical_{l}'])),
                same_position_count=int(np.count_nonzero(indices[l] == old[f'physical_{l}']))) for l in ip.LAYERS}
        memory = root / 'memory'; memory.mkdir()
        files = {}
        for name, values in (('official_banks.npz', banks), ('selected_candidate_indices.npz', indices)):
            path = memory / name; tmp = path.with_suffix('.npz.tmp')
            with tmp.open('xb') as f:
                np.savez(f, **{str(l): values[l] for l in ip.LAYERS})
            tmp.replace(path); files[path.relative_to(root).as_posix()] = ip.sha(path)
        from importlib.metadata import version, PackageNotFoundError
        versions = {}
        for name in ('numpy', 'torch', 'torchvision', 'Pillow'):
            try: versions[name] = version(name)
            except PackageNotFoundError: versions[name] = None
        from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.evaluation import peak_rss
        result = dict(format=FORMAT, category=ctx['category'], status='bank_reconstructed',
            bank_origin='explicit_reconstruction_original_builder', plan_identity=ip.digest(p), plan=p,
            files=files, builder_metadata=metadata, historical_selected_index_audit=index_audit,
            historical_bank_byte_identity_proven=False, original_patch_MAD_refitted=False,
            Source_memory_rebuilt=True,
            actual_output_file_bytes={name:int((root/name).stat().st_size) for name in files},
            reproduction_status='23_calibration_statistics_and_Target_CURRENT_scores_pending',
            python=sys.version, platform=platform.platform(), software=versions,
            seconds=time.perf_counter()-started, peak_rss_bytes=peak_rss(), **peaks)
        ip.write_json(root / 'rebuild_status.json', result)
        print(f'[Source rebuild] {ctx["category"]}: bank_reconstructed; reproduction checks pending', flush=True)
        return result
    except Exception as exc:
        ip.write_json(root / 'rebuild_failure.json', dict(status='rebuild_failed', error=f'{type(exc).__name__}: {exc}',
                                                       seconds=time.perf_counter()-started))
        raise
    finally:
        stopped.set(); watcher.join()
