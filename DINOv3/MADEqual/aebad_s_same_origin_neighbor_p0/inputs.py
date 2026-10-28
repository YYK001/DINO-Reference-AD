"""Metadata/header inventory. Default preflight never imports Torch or decodes GT."""
from pathlib import Path
import sys
import numpy as np
from .protocol import CONFIG,LAYERS,job_plan
from DINOv3.MADEqual.aebad_s_reference_residual_segmentation import storage as st
from DINOv3.MADEqual.aebad_s_reference_residual_segmentation.inputs import key
from DINOv3.MADEqual.aebad_s_sacp_strength_p0.inputs import read_csv,GTData

ROOT=Path(__file__).resolve().parents[3]


def validate_catalog(c):
    src=c['source'];ss=c['supports']['background'];samples=[r for r in c['samples'] if r['domain']=='background']
    if len(src)!=521 or [r['relative_path'] for r in src]!=sorted(r['relative_path'] for r in src):raise ValueError('ordered 521 Source images required')
    if any(r['source_index']!=i or r['calibration']!=(i%8==0) for i,r in enumerate(src)):raise ValueError('original index%8 calibration split required')
    if [len(samples),sum(r['label']==0 for r in samples),sum(r['label']==1 for r in samples)]!=[297,85,212]:raise ValueError('original full background population required')
    if [r['support_index'] for r in ss]!=list(range(1,9)) or any(r['label'] for r in ss):raise ValueError('ordered normal K8 required')
    if {r['relative_path'] for r in ss}&{r['relative_path'] for r in samples}:raise ValueError('support leaked into evaluation')
    from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.fresh_inputs import support_key
    normals=ss+[r for r in samples if r['label']==0]
    if [r['relative_path'] for r in ss]!=[r['relative_path'] for r in sorted(normals,key=lambda r:support_key(r['relative_path'],'background'))[:8]]:raise ValueError('original seed42 support order differs')
    return dict(source=src,supports={'background':ss},samples=samples)


def catalog(a):
    if not hasattr(a,'_catalog'):
        path=a.catalog or (a.parent_run/'catalog.json' if a.parent_run else None)
        if path is None:raise ValueError('--catalog or --parent-run is required (metadata only)')
        value=st.read_json(path);a._catalog=validate_catalog(value.get('catalog',value))
    return a._catalog


def parent(a):
    if not hasattr(a,'_parent'):
        if a.parent_run is None or a.parent_cache is None:return None
        from DINOv3.MADEqual.aebad_s_blind_context_p0.inputs import Parent
        a._parent=Parent(a)
    return a._parent


def bank_path(a):
    if a.source_bank:return a.source_bank
    if a.raw_cache and (a.raw_cache/'source_original/manifest.json').exists():return a.raw_cache/'source_original'
    if a.parent_run:
        refs=st.read_json(a.parent_run/'assets.json')['references']
        return Path(refs['source']['path'])
    return None


def index_path(a):
    if a.selected_indices:return a.selected_indices
    paths=[]
    if a.raw_cache:paths.extend([a.raw_cache/'source_bank.npz',a.raw_cache/'selected_indices.npz'])
    b=bank_path(a)
    seen=set()
    # Follow actual read-only import provenance; do not search entire disks or
    # assume an intermediate import discarded the original selected-index NPZ.
    while b and b not in seen:
        seen.add(b);paths.extend([b.parent/'source_bank.npz',b/'selected_indices.npz'])
        if not (b/'manifest.json').exists():break
        m=st.read_json(b/'manifest.json')['metadata'];origin=m.get('origin','')
        if origin.endswith('.npz'):paths.append(Path(origin));break
        b=Path(origin) if origin and (Path(origin)/'manifest.json').exists() else None
    for p in paths:
        if p.exists():
            with np.load(p,allow_pickle=False) as z:
                if all(f'selected_{l}' in z.files for l in LAYERS):return p
    return None


def feature_ref(a,row):
    candidates=[]
    for root in (a.raw_cache,a.feature_root):
        if root:
            candidates.extend([root/'normal_features'/key(row),root/'features'/key(row),root/'original_features'/key(row),
                root/'queries'/row.get('domain','source')/row.get('sample_id','none')])
    if a.parent_run:
        if not hasattr(a,'_parent_assets'):a._parent_assets=st.read_json(a.parent_run/'assets.json')
        assets=a._parent_assets
        for r in (assets.get('normal_features',{}).get(row['relative_path']),assets.get('queries',{}).get(row.get('domain','')+'/'+row.get('sample_id',''))):
            if r:candidates.append(Path(r['path']))
    for p in candidates:
        if not (p/'manifest.json').exists():continue
        r=st.sealed_json(p/'manifest.json');m=r['metadata'];who=m.get('image',m.get('sample'))
        if who!=row:raise ValueError('feature identity differs: '+str(p))
        if 'transformed' in m.get('kind','') or 'adapted' in m.get('kind',''):raise ValueError('adapted features cannot replace original grids')
        names=tuple(f'q{l}' for l in LAYERS)
        if not all(n in r['files'] for n in names):names=tuple(f'full_L{l}' for l in LAYERS)
        pos='positions' if 'positions' in r['files'] else 'full_positions'
        if not all(n in r['files'] for n in (*names,pos)):continue
        n=int(np.prod(m['grid']))
        for name in names:
            h=np.load(p/(name+'.npy'),mmap_mode='r',allow_pickle=False)
            if h.shape!=(n,1280) or h.dtype!=np.float32:raise ValueError('sampled/adapted features cannot replace complete original grid')
        from . import feature_identity as fi
        ref=dict(path=str(p.resolve()),seal=r['seal'],names=list(names),positions=pos,grid=m['grid'],provider=fi.proof(a,p,row,r))
        frozen=a.run_root/'inputs.json'
        if frozen.exists():
            saved=fi.small_record(a,frozen)
            if 'feature_inputs' not in saved:raise ValueError('run predates feature-provider binding; use a new run')
            before=saved['feature_inputs']['references'].get(row['relative_path'])
            if fi.binding(ref)!=before:raise ValueError('feature provider/manifest differs from frozen image input: '+row['relative_path'])
        return ref
    return None


def identity(a):return st.sealed_json(a.run_root/'inputs.json')['identity']


def preflight(a):
    r=dict(status='blocked',reusable=[],missing=[],unconfirmed=[],errors=[],config=CONFIG,job_plan=job_plan(a.baseline_mode!='fresh' and not a.rebuild_source),
        torch_imported='torch' in sys.modules,GPU_initialized=False,full_payload_hash_scan=False,scope='this filesystem only; server paths unavailable locally are not proof of remote loss')
    try:
        c=catalog(a)
        from .feature_identity import expected
        r['feature_extraction_identity']=st.digest(expected(a))
        r['feature_root']=str(a.feature_root);r['feature_manifest']=str(getattr(a,'feature_manifest',None))
        b=bank_path(a);ix=index_path(a);r['source_bank']=str(b) if b else None;r['selected_indices']=str(ix) if ix else None
        if a.source_neighborhood and (a.source_neighborhood/'manifest.json').exists():
            r['source_case']='exact_recorded_neighborhood';r['reusable'].append(dict(asset='recorded Source centers/deltas/provenance',path=str(a.source_neighborhood)))
        elif b and (b/'manifest.json').exists():
            xb,br=st.get(b,tuple(f'L{l}' for l in LAYERS))
            if any(xb[f'L{l}'].shape!=(100000,1280) or xb[f'L{l}'].dtype!=np.float32 for l in LAYERS):raise ValueError('formal 100K/1280 FP32 Source required')
            r['reusable'].append(dict(asset='original Source bank',path=str(b),seal=br['seal']))
            r['source_case']='selected_indices_replay_pending_center_check' if ix else 'origin_unresolved_explicit_rebuild_needed'
            if ix:r['reusable'].append(dict(asset='selected indices (not full provenance)',path=str(ix)))
            else:r['missing'].append(dict(asset='selected candidate indices or exact provenance',path=str(a.selected_indices or (a.raw_cache or Path('.'))/'source_bank.npz'),action='locate first; explicit --rebuild-source only if unresolved'))
        else:
            r['source_case']='bank_unavailable';r['missing'].append(dict(asset='original Source bank payload',path=str(b),action='locate or explicitly rebuild'))
        if ix is None and r['source_case']!='exact_recorded_neighborhood' and not any(v['asset'].startswith('selected candidate') for v in r['missing']):
            r['missing'].append(dict(asset='selected candidate indices / actual augmented origins',path=str(a.selected_indices or (a.raw_cache or Path('.'))/'source_bank.npz'),action='inspect retained source_bank.npz and import provenance chain on the asset machine'))
        if a.rebuild_source:
            r['source_case']='explicit_independent_rebuild'
            r['missing']=[v for v in r['missing'] if not v['asset'].startswith(('original Source bank','selected candidate'))]
            r['new_work']=['explicit original Source reconstruction with provenance; all baselines and Source calibration recomputed']
        else:r['new_work']=['exact origin/neighborhood recovery if not already recorded; three new methods and their independent Source calibration']
        r['reusable'].append(dict(asset='frozen 521/455/66 order, background297 and seed42 K8 metadata',path=str(a.catalog or a.parent_run/'catalog.json')))
        rows=[x for x in c['source'] if x['calibration']]+c['supports']['background']+c['samples'];refs=[];fresh=[]
        for row in rows:
            ref=feature_ref(a,row)
            (refs if ref else fresh).append(dict(relative_path=row['relative_path'],**(ref or {})))
        r['feature_references']=refs;r['features_to_extract']=fresh
        source_full=bool(a.augmented_grids) and all((a.augmented_grids/f"source_{x['source_index']:03d}"/'manifest.json').exists() for x in c['source'] if not x['calibration'])
        requires_host=bool(fresh) or a.rebuild_source or (r['source_case']!='exact_recorded_neighborhood' and not source_full)
        required=[('AeBAD RGB/GT',a.data)]
        if requires_host:required += [('checkpoint',a.checkpoint),('host config',a.official_root/'config.json'),('DINOv3',a.dinov3_root/'hubconf.py')]
        for name,p in required:
            if not p.exists():r['missing'].append(dict(asset=name,path=str(p),action='required for missing augmented/full features'))
        r['counts']=dict(Source=521,construction=455,calibration=66,supports=8,evaluation=297,normal=85,anomaly=212)
        r['requires_backbone']=requires_host
        r['unconfirmed']=['actual augmented Source RNG/order until exact centers verified','current-device Top16/neighbor throughput','parent baseline reuse only after bank and map/score checks']
        r['baseline_policy']='same Source seal plus compatible frozen maps/scores: reuse; any independent bank: recompute UNION/SEGMENT and their calibration/metrics'
        r['status']='inventory_incomplete' if r['missing'] else 'inventory_ready_runtime_proof_pending'
    except (OSError,ValueError,KeyError) as e:r['errors'].append(type(e).__name__+': '+str(e))
    from importlib.metadata import version,PackageNotFoundError
    r['dependency_versions_without_import']={}
    for name in ('numpy','scipy','Pillow','scikit-learn','psutil','torch','torchvision','timm','adeval'):
        try:r['dependency_versions_without_import'][name]=version(name)
        except PackageNotFoundError:r['dependency_versions_without_import'][name]=None
    from .reporting import estimate
    r['time_and_disk_estimate']=estimate(r)
    return r


def freeze(a):
    c=catalog(a);path=a.run_root/'inputs.json'
    if path.exists():
        saved=st.sealed_json(path)
        if 'feature_inputs' not in saved:raise ValueError('run predates feature-provider binding; use a new run; old manifests remain unchanged')
        expected=dict(config=CONFIG,catalog=c,cache=str(a.cache_root),data=str(a.data),rebuild=bool(a.rebuild_source),
            source_neighborhood=str(a.source_neighborhood),augmented_grids=str(a.augmented_grids),baseline_mode=a.baseline_mode,
            parent_run=str(a.parent_run),parent_cache=str(a.parent_cache),raw_cache=str(a.raw_cache),
            feature_root=str(a.feature_root),feature_manifest=str(getattr(a,'feature_manifest',None)))
        if any(saved[k]!=v for k,v in expected.items()):raise ValueError('run configuration/input paths changed; use a new run')
        for arg,keyname in ((a.selected_indices,'selected_indices'),(a.source_bank,'source_bank')):
            if arg is not None and saved[keyname]!=str(arg):raise ValueError('explicit Source provider changed')
        # A committed prediction/evaluation does not need an old index NPZ or
        # the original Source builder files to still be online.
        from .feature_identity import verify_saved
        verify_saved(a,saved)
        return saved
    value=dict(config=CONFIG,catalog=c,cache=str(a.cache_root),data=str(a.data),
        source_bank=str(bank_path(a)),selected_indices=str(index_path(a)),rebuild=bool(a.rebuild_source),
        source_neighborhood=str(a.source_neighborhood),augmented_grids=str(a.augmented_grids),baseline_mode=a.baseline_mode,
        parent_run=str(a.parent_run),parent_cache=str(a.parent_cache),raw_cache=str(a.raw_cache),
        feature_root=str(a.feature_root),feature_manifest=str(getattr(a,'feature_manifest',None)))
    from .feature_identity import inventory
    value['feature_inputs']=inventory(a,c)
    value['identity']=st.digest(value)
    record=st.sealed_json(path,value);st.sealed_json(a.run_root/'protocol.json',CONFIG)
    st.write_csv(a.run_root/'evaluation_images.csv',c['samples'])
    return record
