"""Explicit compact-distance supplementation, using the parent's Exact engine.

No Source building, Source66 extraction, or test GT is implemented here.
"""
from pathlib import Path
import numpy as np
from . import inputs as ip, numerics as nm
from .inputs import st
from .protocol import *
from DINOv3.MADEqual.aebad_s_target_loo_support_retest_p0.runtime import References as OriginalReferences


def features(a, row, persist=False):
    ref = ip.feature_ref(a, row)
    if ref:
        x, r = st.get(ref['path'], tuple('q'+str(l) for l in LAYERS)+('positions',))
        return {l: x['q'+str(l)] for l in LAYERS}, dict(grid=r['metadata']['grid'], positions=x['positions']), ref
    if not a.extract_missing:
        raise FileNotFoundError('complete features absent; --extract-missing is explicit permission, not enabled: '+row['relative_path'])
    from DINOv3.MADEqual.aebad_s_same_origin_neighbor_p0 import feature_identity as fi
    from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.runtime import host, extraction, reliable_precision
    from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.fresh_inputs import Data
    fi.compatible(fi.native_extraction(a), ip.parent(a).inputs['extraction'])
    reliable_precision(a.device)
    if not hasattr(a, '_model'):
        a._model = host(a)[0]
    with Data(a.data) as data:
        q, meta = extraction(a._model, data.image(row), a.device)
    a.feature_forwards = getattr(a, 'feature_forwards', 0)+1
    ref = dict(kind='original_full_forward', extraction=ip.parent(a).inputs['extraction'], image=ip.image_contract(row))
    if persist:
        path = a.cache_root/'normal_features'/ip.previous.layer_inputs.old.key(row)
        r = st.put(path, {**{'q'+str(l): q[l] for l in LAYERS}, 'positions': meta['positions']},
                   dict(image=row, extraction=ip.parent(a).inputs['extraction'], kind='original_complete_grid', grid=meta['grid']))
        ref = dict(path=str(path), seal=r['seal'])
    return q, meta, ref


class References(OriginalReferences):
    """Only initialization is adapted; original distances() performs normal K7.

    The original full-image exclusion, FP32 Top1 and /1280 are reused verbatim.
    Query Target-only supplementation skips Source if a scientific match exists.
    """
    def __init__(self, a, d, seed, search=None):
        from DINOv3.MADEqual.aebad_s_reference_residual_segmentation.execution import ResidentExact
        from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.runtime import reliable_precision
        reliable_precision(a.device)
        self.a = a;self.supports = ip.parent(a).common['domains'][d]['supports'][str(seed)]
        self.search = search if search is not None else ResidentExact(a.device, 8*2**30)
        self.source = {};self.source_seal = ip.parent(a).inputs['source']['seal']
        self.parts = [];self.support_seals = [];self.geometry = [];self.origins = []
        for row in self.supports:
            q, meta, ref = features(a, row, persist=True)
            self.parts.append(q);self.support_seals.append(st.digest(ref));self.geometry.append(meta);self.origins.append(ref)

    def load_source(self):
        if not self.source:
            path = self.a.source_bank or Path(ip.parent(self.a).inputs['source']['path'])
            self.source, record = st.get(path, tuple('L'+str(l) for l in LAYERS))
            if record['seal'] != self.source_seal:raise ValueError('exact parent Source required; no alternative or rebuild')

    def distances(self, q, heldout=None):
        self.load_source()
        return super().distances(q, heldout=heldout)

    def target_only(self, q):
        values = []
        for l in LAYERS:
            query = self.search.query(q[l])  # Upload a layer once, reuse for all eight supports.
            values.append(np.stack([self.search.exact(query, bank[l], ('Target', seal, l))[0]
                                    for bank, seal in zip(self.parts, self.support_seals)]))
        return np.stack(values)


def normal_one(a, d, seed, j, refs):
    row = refs.supports[j]
    try:
        return ip.inspect_normal(a, d, seed, j)
    except FileNotFoundError:
        pass
    nn = refs.distances(refs.parts[j], heldout=j)  # ORIGINAL whole-image K7 exclusion.
    for l in LAYERS:
        if nn[l]['support_indices'] != heldout_indices(j):
            raise ValueError('normal LOO retained self or reordered supports')
    ds = np.stack([nn[l]['ds'] for l in LAYERS]);dt = np.stack([nn[l]['dt'].min(0) for l in LAYERS])
    meta = refs.geometry[j]
    r = st.put(ip.normal_path(a, d, seed, j), dict(Source_distance=ds, Target7_distance=dt, positions=meta['positions']),
               dict(contract=ip.split_contract(a, d, seed, row, j), grid=meta['grid'], feature_origin=refs.origins[j], GT_used=False))
    return ip.inspect_normal(a, d, seed, j)


def capture(a, only_normals=False):
    if not a.supplement_retrieval:
        raise ValueError('supplementation requires --supplement-retrieval; default preflight does not compute')
    p = ip.parent(a);completed = [];query_count = 0;shared_search = None
    for d in a.domains:
        refs = {}
        def reference(seed):
            nonlocal shared_search
            if seed not in refs:
                refs[seed] = References(a, d, seed, shared_search)
                shared_search = refs[seed].search
            return refs[seed]
        for seed in a.support_seeds:
            for j in range(8):
                try:
                    _, r = ip.inspect_normal(a, d, seed, j)
                except FileNotFoundError:
                    _, r = normal_one(a, d, seed, j, reference(seed))
                completed.append(dict(domain=d, support_seed=seed, support_index=j, seal=r['seal']))
                print(f'[joint normal K7] {d}/seed{seed}/{j+1} separate_responses', flush=True)
                from .budget import observe
                observe(a)
        if only_normals:
            continue
        for row in p.samples(d):
            missing = []
            for seed in a.support_seeds:
                try:
                    ip.inspect_split(a, d, seed, row)
                except FileNotFoundError:
                    missing.append(seed)
            if not missing:
                continue
            q, meta, feature_origin = features(a, row, persist=False)  # ONE shared forward, at most.
            pos = np.asarray(meta['positions'], np.int64);grid = meta['grid']
            try:
                sx, sr, source_origin = ip.inspect_source(a, d, row)
                if sr['metadata']['grid'] != grid:
                    raise ValueError('shared Source grid differs')
                physical = nm.physical(sx['Source_distance'], sx['positions'], grid)
                ds = physical[:, pos]
            except FileNotFoundError:
                ref = reference(missing[0]);ref.load_source();values = []
                for l in LAYERS:
                    query = ref.search.query(q[l]);values.append(ref.search.exact(query, ref.source['L'+str(l)], ('Source', ref.source_seal, l))[0])
                ds = np.stack(values);source_origin = dict(kind='one_original_Source_search', feature_origin=feature_origin)
            # An existing compatible shared Source has already been inspected.
            # Keep its ORIGINAL provenance on resume; do not replace the old
            # origin with a self-reference and trigger an immutable conflict.
            if not (ip.source_path(a, d, row)/'manifest.json').exists():
                st.put(ip.source_path(a, d, row), dict(Source_distance=ds, positions=pos),
                       dict(contract=ip.source_contract(a, row), grid=grid, origin=source_origin, GT_used=False))
            for seed in missing:
                dt = reference(seed).target_only(q)
                r = st.put(ip.query_path(a, d, seed, row), dict(Source_distance=ds, target_distance=dt, positions=pos),
                           dict(contract=ip.split_contract(a, d, seed, row), grid=grid,
                                output_shape=[row['image_height']//4, row['image_width']//4],
                                feature_origin=feature_origin, Source_origin=source_origin, GT_used=False))
                query_count += 1
                print(f'[joint split capture] {d}/seed{seed}/{row["sample_id"]} all_K7_shared', flush=True)
            from .budget import observe
            observe(a)
    a.exact_execution = shared_search.stats if shared_search else {}
    return dict(status='normal_responses_complete' if only_normals else 'requested_separate_responses_complete', normal_records=completed,
                new_query_settings=query_count, feature_forwards=getattr(a, 'feature_forwards', 0), execution=a.exact_execution)


def build_references(a):
    p = ip.parent(a);records = [];sampling = []
    from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.calibration import z_values
    for d in a.domains:
        for seed in a.support_seeds:
            for j, row in enumerate(p.common['domains'][d]['supports'][str(seed)]):
                x, r = ip.inspect_normal(a, d, seed, j);grid = r['metadata']['grid']
                ids = nm.sample_positions(row['relative_path'], grid)
                rs = nm.physical(z_values(x['Source_distance'], p.inputs['mad']['patch']), x['positions'], grid)[:, ids].T
                rt = nm.physical(z_values(x['Target7_distance'], p.inputs['mad']['patch']), x['positions'], grid)[:, ids].T
                metadata = dict(**ip.signature(a, d, seed), support_index=j, image=ip.image_contract(row),
                                response_seal=r['seal'], contract=r['metadata']['contract'], grid=grid, GT_used=False)
                path = a.cache_root/'normal_model'/d/f'seed{seed}'/f'{j:02d}'
                saved = st.put(path, dict(Source=rs, Target=rt, positions=ids), metadata)
                records.append(dict(domain=d, support_seed=seed, support_index=j, seal=saved['seal'], positions=len(ids),
                                    path=str(path), response_seal=r['seal']))
                sampling.extend(dict(domain=d, support_seed=seed, support_index=j, relative_path=row['relative_path'], physical_index=int(i),
                                     patch_y=int(i)//grid[1], patch_x=int(i)%grid[1]) for i in ids)
    # Freeze only complete six-setting normal reference sets, never test-derived.
    if len(records) == 48:
        st.write_csv(a.run_root/'normal_sampling.csv', sampling)
        return st.sealed_json(a.run_root/'normal_model.json', dict(identity=st.sealed_json(a.run_root/'inputs.json')['identity'], records=records, GT_used=False))
    # A requested subset must not truncate an already complete global sampling
    # manifest. Keep partial preparation explicitly separate from the full list.
    st.write_csv(a.run_root/'normal_sampling_requested_subset.csv', sampling)
    if (a.run_root/'normal_model.json').exists():
        for d in a.domains:
            for seed in a.support_seeds:normal_model(a,d,seed)
        return dict(status='requested_normal_model_subset_complete',global_model_unchanged=True,records=records)
    return dict(status='partial_normal_model', records=records)


def normal_model(a, d, seed):
    record = st.sealed_json(a.run_root/'normal_model.json');frozen = st.sealed_json(a.run_root/'inputs.json')
    if record['identity'] != frozen['identity'] or len(record['records']) != 48:
        raise ValueError('complete fixed normal reference freeze required')
    selected = [r for r in record['records'] if r['domain'] == d and r['support_seed'] == seed]
    if [r['support_index'] for r in selected] != list(range(8)):
        raise ValueError('normal support order differs')
    result = []
    for ref in selected:
        x, r = st.get(ref['path'])
        if r['seal'] != ref['seal'] or r['metadata']['contract'] != ip.split_contract(a, d, seed, ip.parent(a).common['domains'][d]['supports'][str(seed)][ref['support_index']], ref['support_index']):
            raise ValueError('normal sampling/response dependencies differ')
        result.append((x['Source'], x['Target']))
    return result
