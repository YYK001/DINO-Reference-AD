"""Multi-seed manifest adapter. Execute the ORIGINAL pure CRC key, not a copy."""
from __future__ import annotations

import ast
import itertools
import zlib
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

DOMAINS = ('background', 'illumination', 'view')
SEEDS = (42, 43, 44)
EXPECTED = {'background': (93, 212, 24, 281),
            'illumination': (75, 198, 24, 249), 'view': (92, 280, 24, 348)}


@lru_cache(maxsize=1)
def original_key():
    # The original module eagerly imports Torch through SupportFeatures. Compile
    # just its two pure definitions unchanged, without executing module imports.
    # This is the production selection path, NOT a NumPy/mock replacement.
    source = Path(__file__).parents[1] / 'aebad_s_target_fewshot/support.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    names = ('relative_path', 'stable_support_key')
    definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if tuple(n.name for n in definitions) != names:
        raise ValueError('original pure support key API changed; audit required')
    namespace = dict(Path=Path, zlib=zlib)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                             *definitions], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), namespace)
    return namespace['stable_support_key']


def select(samples, formal_supports, *, root):
    """All samples use the original global sample_id, including support images."""
    groups = {}; common = []; excluded = []; candidates = []; overlap = []
    if len({s['relative_path'] for s in samples}) != len(samples):
        raise ValueError('duplicate physical image path')
    if len({s['sample_id'] for s in samples}) != len(samples):
        raise ValueError('duplicate original sample identity')
    for domain in DOMAINS:
        rows = [s for s in samples if s['domain'] == domain]
        normals = [s for s in rows if s['label'] == 0]
        candidates.extend(normals)
        inherited = sorted((r for r in formal_supports if r['domain'] == domain
                            and r['branch'] == 'target_matched'), key=lambda r: int(r['support_index']))
        if [int(r['support_index']) for r in inherited] != list(range(1, 9)):
            raise ValueError('formal seed42 support order incomplete/duplicated')
        lookup = {s['relative_path']: s for s in normals}
        groups[domain] = {}
        for seed in SEEDS:
            if seed == 42:
                chosen = [lookup[r['relative_path']] for r in inherited]
            else:
                chosen = sorted(normals, key=lambda s: original_key()(
                    SimpleNamespace(path=Path(root)/s['relative_path']), dataset_root=Path(root),
                    domain=domain, seed=seed))[:8]
            if len(chosen) != 8 or len({s['relative_path'] for s in chosen}) != 8:
                raise ValueError('K8 requires eight distinct confirmed normal images')
            groups[domain][str(seed)] = [dict(s, support_seed=seed, support_index=i)
                                        for i, s in enumerate(chosen, 1)]
        union = {s['relative_path'] for v in groups[domain].values() for s in v}
        heldout = [s for s in rows if s['relative_path'] not in union]
        actual = (len(normals), sum(s['label'] for s in rows), len(union), len(heldout))
        if actual != EXPECTED[domain]:
            raise ValueError(f'{domain}: actual normal/anomaly/support-union/heldout {actual}, expected {EXPECTED[domain]}')
        common.extend(heldout)
        for s in rows:
            if s['relative_path'] in union:
                excluded.append(dict(s, reason='support_in_any_of_three_sets',
                    excluded_by_seeds=[seed for seed in SEEDS if any(
                        r['relative_path'] == s['relative_path'] for r in groups[domain][str(seed)])]))
        for a, b in itertools.combinations(SEEDS, 2):
            x, y = ({s['relative_path'] for s in groups[domain][str(seed)]} for seed in (a, b))
            overlap.append(dict(domain=domain, seed_a=a, seed_b=b, intersection=len(x & y),
                                union=len(x | y), Jaccard=len(x & y)/len(x | y)))
    return dict(groups=groups, samples=common, excluded=excluded, candidates=candidates, overlap=overlap)


def support_pool(plan, domain):
    """Global image pool in first occurrence order; local mappings stay 1..8."""
    pool = []; mapping = {}
    for seed in SEEDS:
        indices = []
        for row in plan['groups'][domain][str(seed)]:
            index = next((i for i, s in enumerate(pool) if s['relative_path'] == row['relative_path']), None)
            if index is None:
                index = len(pool); pool.append(row)
            indices.append(index)
        mapping[str(seed)] = indices
    return pool, mapping
