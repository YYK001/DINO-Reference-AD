"""Fresh inputs, explicit new-workstation reuse, and seed/scope identities."""
import json
import os
from pathlib import Path
import shutil
import numpy as np
from DINOv3.MADEqual.aebad_s_support_witness_segment_p0 import storage as st
from DINOv3.MADEqual.aebad_s_support_witness_segment_p0 import fresh_inputs as parent
from .protocol import DOMAINS, REVISION, PROTOCOL, LAYERS

ROOT=Path(__file__).resolve().parents[3]
Data=parent.Data
CASE_IDS=parent.CASE_IDS


def merge_rows(path,rows,keys):
    """Derived summaries accumulate completed jobs; never drop other domains/folds."""
    import csv
    path=Path(path);combined={}
    def key(row):return tuple(str(row[k]) for k in keys)
    if path.exists():
        with path.open(encoding='utf-8-sig',newline='') as f:
            combined={key(r):r for r in csv.DictReader(f)}
    for row in rows:combined[key(row)]=row
    st.write_csv(path,[combined[k] for k in sorted(combined)])


def seeded_catalog(base,seed=42,scope='primary'):
    from DINOv3.MADEqual.aebad_s_k8_support_sensitivity_p0.selection import original_key
    from types import SimpleNamespace
    if seed not in (42,43,44) or scope not in ('primary','common') or (scope=='primary' and seed!=42):
        raise ValueError('seed42 primary or explicit three-set common protocol required')
    all_rows=sorted(base['samples']+sum(base['supports'].values(),[]),key=lambda r:r['relative_path'])
    if len({r['relative_path'] for r in all_rows})!=len(all_rows): raise ValueError('duplicate catalog image')
    groups={}; samples=[]; exclusions=[]
    for d in DOMAINS:
        normal=[r for r in all_rows if r['domain']==d and not r['label']]
        groups[d]={}
        for s in (42,43,44):
            chosen=sorted(normal,key=lambda r:original_key()(SimpleNamespace(path=Path(r['relative_path'])),
                dataset_root=Path('.'),domain=d,seed=s))[:8]
            groups[d][s]=[dict(r,support_index=i+1) for i,r in enumerate(chosen)]
        if [r['relative_path'] for r in groups[d][42]] != [r['relative_path'] for r in base['supports'][d]]:
            raise ValueError('original CRC32 seed42 support order changed')
        excluded={r['relative_path'] for r in (sum(groups[d].values(),[]) if scope=='common' else groups[d][seed])}
        rows=[r for r in all_rows if r['domain']==d and r['relative_path'] not in excluded]
        expected={'background':297,'illumination':265,'view':364} if scope=='primary' else {'background':281,'illumination':249,'view':348}
        if len(rows)!=expected[d]: raise ValueError('frozen evaluation population mismatch: '+d)
        samples.extend(rows);exclusions.extend(sorted(excluded))
    return dict(base,revision=REVISION,seed=seed,evaluation_scope=scope,
        supports={d:groups[d][seed] for d in DOMAINS},samples=samples,
        support_groups={d:{str(s):v for s,v in g.items()} for d,g in groups.items()},
        excluded_relative_paths=exclusions)


def prepare(a):
    catalog_path=a.run_root/'fresh_catalog.json'
    if catalog_path.exists():
        c=st.read_json(catalog_path)
        if st.sha(catalog_path)!=st.read_json(a.run_root/'fresh_catalog.seal.json')['sha256']:
            raise ValueError('catalog modified')
        return c
    # Parent performs exact original RGB indexing/extraction into an independent base catalog.
    base=parent.prepare_data(a.run_root/'base_catalog',a.data,a.cache_root)
    c=seeded_catalog(base,a.seed,a.evaluation_scope)
    st.write_json(catalog_path,c);st.write_json(a.run_root/'fresh_catalog.seal.json',dict(sha256=st.sha(catalog_path)))
    return c


def identity(a): return st.read_json(a.run_root/'setup.json')['identity']
def catalog(a): return st.read_json(a.run_root/'fresh_catalog.json')


def link_reused(a,old_path,destination,old_identity):
    """A new record references original sealed bytes; never rewrite the old record."""
    old_path=Path(old_path);r=st.restore(old_path,old_identity)
    if r is None: return False
    if st.restore(destination,identity(a)): return True
    destination.parent.mkdir(parents=True,exist_ok=True)
    # Symlink, not a copy or in-place relabel of the old experiment.
    destination.symlink_to(old_path.resolve())
    st.write_json(destination.with_suffix('.json'),dict(identity=identity(a),sha256=r['sha256'],
        metadata=dict(r['metadata'],reuse_source=dict(path=str(old_path.resolve()),identity=old_identity,
            original_record_sha256=st.sha(old_path.with_suffix('.json'))))))
    return True


def reuse_context(a):
    if a.reuse_fresh_run is None: return None
    old=a.reuse_fresh_run.resolve();old_setup=st.read_json(old/'setup.json');previous=st.read_json(old/'fresh_catalog.json')
    current=catalog(a)
    # The supported reuse source is the explicitly named new-workstation fresh run.
    if previous.get('revision') not in (parent.REVISION,REVISION) or previous['source']!=current['source']:
        raise ValueError('reuse run does not have the same frozen Source image identities/order')
    configuration=old_setup['configuration'];old_code=configuration['code']
    relatives=set(('DINOv3/relation_reliability/superadd_host.py',
                     'external_baselines/superadd_external/memory_variants.py',
                     'external_baselines/superadd_external/run.py',
                     'DINOv3/MADEqual/aebad_s_support_witness_segment_p0/fresh_runtime.py',
                     'DINOv3/ShiftADD/aebad_s_p0/memory.py'))
    # Old WITNESS stores external source hashes in code; this independent tool
    # stores configurable roots in external_source_code. Verify both formats,
    # rather than requiring a key that the new format never writes.
    if previous['revision']==parent.REVISION:
        relatives.add('external_baselines/SuperADD/config.json')
    elif configuration.get('host_config')!=st.read_json(a.official_root/'config.json'):
        raise ValueError('reuse host configuration differs')
    relatives.update(p for p in old_code if p.startswith(('external_baselines/dinov3/',
        'external_baselines/SuperADD/tracks/industrial/src/industrial/')))
    for relative in sorted(relatives):
        actual=(a.dinov3_root/relative.removeprefix('external_baselines/dinov3/')
            if relative.startswith('external_baselines/dinov3/') else
            a.official_root/relative.removeprefix('external_baselines/SuperADD/')
            if relative.startswith('external_baselines/SuperADD/') else ROOT/relative)
        if old_code.get(relative)!=st.sha(actual): raise ValueError('reuse host/source code differs: '+relative)
    if previous['revision']==REVISION:
        external=configuration.get('external_source_code',{})
        actual_external={}
        for name,root in (('dinov3',a.dinov3_root),('SuperADD',a.official_root)):
            for path in sorted(root.rglob('*.py')):
                actual_external[name+'/'+path.relative_to(root).as_posix()]=st.sha(path)
        if not external or external!=actual_external:
            raise ValueError('reuse external host/source code differs')
    from importlib.metadata import version
    for name in ('numpy','scipy','torch','torchvision'):
        if configuration['software'].get(name)!=version(name):
            raise ValueError('reuse extraction/calibration software differs: '+name)
    if st.read_json(a.official_root/'config.json')!=st.read_json(ROOT/'external_baselines/SuperADD/config.json'):
        raise ValueError('reuse host configuration differs')
    return old,Path(old_setup['cache']),old_setup['identity'],previous


def freeze_assets(a):
    paths=[a.cache_root/'source_bank.npz',a.cache_root/'source_patch_mad.npz']
    paths += [a.cache_root/'supports'/d/f'{r["support_index"]:03d}.npz'
              for d in DOMAINS for r in catalog(a)['supports'][d]]
    if not all(p.exists() and p.with_suffix('.json').exists() for p in paths):return
    assets={str(p):st.read_json(p.with_suffix('.json')) for p in paths}
    value=dict(identity=identity(a),assets=assets,image_calibration_state_sha256=st.sha(a.run_root/'image_calibration_state.json'))
    value['seal']=st.digest(value);path=a.run_root/'frozen_assets.json'
    if path.exists() and st.read_json(path)!=value:raise ValueError('frozen Source/MAD/support assets changed')
    st.write_json(path,value)


def estimate(a):
    """Actual header dimensions, no sampling, model, CUDA or capacity admission."""
    c=catalog(a);rows=[];table_bytes=0;query_bytes=0
    for d in getattr(a,'domains',DOMAINS):
        sizes=[];grids=[]
        for r in c['supports'][d]:
            with np.load(a.cache_root/'supports'/d/f"{r['support_index']:03d}.npz",allow_pickle=False) as z:
                g=z['grid'].tolist();sizes.append(int(np.prod(g)));grids.append(g)
        pair=sum(n*min(100,m)*16 for i,n in enumerate(sizes) for j,m in enumerate(sizes) if i!=j)*4
        table_bytes+=pair
        sample_count=sum(s['domain']==d for s in c['samples'])
        # All actual support grids agree in AeBAD-S. If not, do not pretend the
        # support shape is an exact query shape estimate.
        qbytes=sample_count*sizes[0]*1280*4*4 if len(set(map(tuple,grids)))==1 else None
        if qbytes is not None:query_bytes+=qbytes
        rows.append(dict(domain=d,support_patch_counts=sizes,stored_pair_Top100_bytes_four_layers=pair,
            maximum_mutual_edges_per_layer=sum(sizes)*100//2,
            query_cache_bytes_using_support_grid=qbytes if a.query_mode=='disk' else 0,
            query_grid_estimate_not_observed=True,
            endpoint_block_bytes=a.query_chunk*sum(sizes)*8,
            witness_geometry_block_bytes_estimate=a.witness_edge_chunk*a.reference_chunk*8*16))
    result=dict(domains=rows,execution_domains=list(getattr(a,'domains',DOMAINS)),neighbor_tables_bytes=table_bytes,
        query_cache_bytes_if_support_grid=query_bytes if a.query_mode=='disk' else 0,
        actual_graph_cost='emitted per domain/layer/fold BEFORE exhaustive witness search',
        approximation=False,capacity_admission_checks=False)
    st.write_json(a.run_root/'cost_estimate.json',result);return result


def patch_state(a):
    raw,_=st.arrays(a.cache_root/'source_patch_mad.npz',identity(a))
    state=st.read_json(a.run_root/'image_calibration_state.json')
    if state['state_sha256']!=st.digest({k:v for k,v in state.items() if k!='state_sha256'}):
        raise ValueError('image MAD modified')
    return raw['medians'],raw['scales'],state


def load_banks(a,d,l,retained=None):
    c=catalog(a);banks={}
    for row in c['supports'][d]:
        sid=row['support_index']
        if retained is not None and sid not in retained: continue
        p=a.cache_root/'supports'/d/f'{sid:03d}.npz'
        r=st.restore(p,identity(a))
        if r is None: raise FileNotFoundError(p)
        with np.load(p,allow_pickle=False) as z:
            if str(z['relative_path'].item())!=row['relative_path']: raise ValueError('support image order mismatch')
            value=z[f'layer_{l}'].copy()
            if value.dtype!=np.float32 or value.shape!=(int(np.prod(z['grid'])),1280) or not np.isfinite(value).all():
                raise ValueError('invalid original support features')
        banks[sid]=value
    return banks


def verify_prediction(a,sample):
    p=a.prediction_root/sample['domain']/(sample['sample_id']+'.npz')
    arrays,record=st.arrays(p,identity(a));m=record['metadata']
    if any(m[k]!=sample[k] for k in ('domain','sample_id','relative_path','image_sha256')):
        raise ValueError('prediction image identity differs')
    if m['selection_sha256']!=st.sha(a.run_root/'selection'/f"{sample['domain']}.json"):
        raise ValueError('normal selection changed after prediction')
    if m['source_mad_state_sha256']!=st.sha(a.cache_root/'source_patch_mad.npz') or m['image_mad_state_sha256']!=st.sha(a.run_root/'image_calibration_state.json'):
        raise ValueError('Source calibration changed after prediction freeze')
    return arrays,m,record
