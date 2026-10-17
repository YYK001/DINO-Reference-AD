"""Read-only discovery, original identities and durable per-category ledgers."""
from __future__ import annotations
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import numpy as np
from PIL import Image
from DINOv3.relation_reliability.datasets import robustad_full_records, split_source_normal
from DINOv3.relation_reliability.protocol import stable_json_hash as digest
from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.method import BRANCHES, LAYERS

TOOL = 'robustad_segment_pair_crossdataset_v1'
CATEGORIES = ('MetalParts', 'PCB')
COUNTS = {'MetalParts': {'background_1': (52,40), 'background_2': (52,40),
    'lighting': (112,80), 'position': (112,80), 'rotation': (112,80), 'scale': (112,80)},
    'PCB': {'lighting': (112,90), 'white_balancing': (112,60), 'rotation': (72,60),
            'position': (72,60), 'shadow': (72,60)}}
CHECKPOINT_SHA = '7c1da9a54b3bdb333f5ebc42e404b7f19b1b5bed504877623c9dc87397f41488'
CATALOG_SHA = '479c4df67c7a768aa6340ea3a845b0f3fc1f93d9961e879810d336f51d44bb25'
PRECISION = dict(float32_matmul_precision='highest', cuda_matmul_allow_tf32=False, cudnn_allow_tf32=True)
PROTOCOL = dict(tool=TOOL, categories=list(CATEGORIES), branches=list(BRANCHES), seed=42,
    layers=list(LAYERS), dimension=1280, distance_unit='euclidean_over_1280_once',
    source_split='original_metadata_order_index_mod8_eq0', calibration_images=23,
    score='frozen_category_Source_CH1_peak_local_half_half', radius=1, exclude_center=True,
    pair_votes='ceil(actual_neighbors/2)', no_neighbors='UNION', endpoint_tie='manifest_support_order',
    distance_atol=1e-7, distance_rtol=1e-6, score_atol=1e-5, historical_score_atol=1e-4,
    historical_score_policy='cross_environment_fixed_atol_v1', score_rtol=0,
    historical_distance_role='audit_only_original_Source_observation_when_available',
    localization='Source_patch_MAD_equal4_original_bilinear', nstrips=200, FPR_max=.3,
    macro='shift_equal_within_category_then_category_equal', ARD='min(0,target_metric-source_metric)',
    Source_evaluation='source_only_no_Target_support', rgb_refinement=False)
FORMAL_FILES = ('complete.json', 'sample_scores.csv', 'support_manifest.csv')
SOURCE_FILES = ('complete.json', 'calibration/calibration_state.npz', 'memory/memory_provenance.json',
                'memory/anchor_sidecars.npz')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()


def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');tmp.replace(path)


def write_csv(path,rows):
    rows=list(rows);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
    tmp.replace(path)


def safe(value):
    p=PurePosixPath(str(value))
    if p.is_absolute() or '..' in p.parts or '\\' in str(value) or not p.parts:raise ValueError('unsafe relative path')
    return p.as_posix()


def canonical(value,category):
    parts=PurePosixPath(str(value).replace('\\','/')).parts
    if parts.count(category)!=1:raise ValueError('historical path has ambiguous category: '+str(value))
    return safe('/'.join(parts[parts.index(category):]))


def defaults(project):
    p=Path(project)
    return dict(project=p,data=p/'datasets/RobustAD',
        formal_metalparts=p/'server_results/robustad_target_fewshot_metalparts_seed42_p0_v1_formal_results',
        formal_pcb=p/'server_results/robustad_target_fewshot_pcb_piledbags_seed42_results_20260819/DINO-NVS/DINOv3/outputs/mad_equal/robustad_target_fewshot_pcb_seed42_p0_v1',
        source_root=p/'server_results/relation_reliability_seed42_v6_no_banks/outputs/relation_reliability/seed42_v6',
        official=p/'external_baselines/SuperADD',dinov3=p/'external_baselines/dinov3',
        checkpoint=p/'external_baselines/SuperADD/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth',
        output_root=p/'DINOv3/outputs/mad_equal/robustad_segment_pair_crossdataset_seed42_v1')


def build_catalog(paths):
    """Development-only freeze from actual old manifests and original data order."""
    root=Path(paths['data']).resolve();train,evaluation=robustad_full_records(root,CATEGORIES)
    old=read_csv(Path(paths['source_root'])/'dataset_manifest.csv')
    result=dict(categories={},host_config=read_json(Path(paths['official'])/'config.json'),
                shared_source_files={n:sha(Path(paths['source_root'])/n) for n in ('dataset_manifest.csv','protocol.json')})
    for category in CATEGORIES:
        formal=Path(paths['formal_'+category.lower()]);src=Path(paths['source_root'])/'categories/robustad'/category
        complete=read_json(formal/'complete.json');sc=read_json(src/'complete.json')
        if complete['category']!=category or complete['status']!='complete' or complete['cache_mode']!='complete_external_bundle':raise ValueError('wrong formal experiment')
        if sc['category']!=category or sc['status']!='complete' or sc['seed']!=42:raise ValueError('wrong Source baseline')
        mem,cal=split_source_normal(train[category],8)
        if (len(train[category]),len(mem),len(cal))!=(180,157,23):raise ValueError('Source split differs')
        for role,rr in (('memory_fit_source_normal',mem),('reliability_calibration_source_normal',cal)):
            prior=[canonical(r['path'],category) for r in old if r['dataset']=='robustad' and r['category']==category and r['protocol_role']==role]
            if prior!=[Path(r.path).relative_to(root).as_posix() for r in rr]:raise ValueError('ordered original Source split differs')
        def sample(record):
            rel=Path(record.path).relative_to(root).as_posix()
            with Image.open(record.path) as im: width,height=im.size
            s=dict(category=category,shift=record.shift,domain=record.shift,sample_id=rel,relative_path=rel,
                   label=int(record.label),image_height=height,image_width=width,image_sha256=sha(record.path))
            if record.mask_path:
                s.update(mask_path=Path(record.mask_path).resolve().relative_to(root).as_posix(),gt_sha256=sha(record.mask_path))
            return s
        source_eval=[r for r in evaluation[category] if r.domain=='source']
        prior=[(canonical(r['path'],category),int(r['label'])) for r in old if r['dataset']=='robustad' and r['category']==category and r['protocol_role']=='evaluation' and r['domain']=='source']
        if prior!=[(Path(r.path).relative_to(root).as_posix(),r.label) for r in source_eval]:raise ValueError('Source evaluation identities differ')
        if (sum(r.label==0 for r in source_eval),sum(r.label==1 for r in source_eval))!=(120,51 if category=='MetalParts' else 75):raise ValueError('Source evaluation count differs')
        lookup={Path(r.path).relative_to(root).as_posix():r for r in evaluation[category]}
        scores=read_csv(formal/'sample_scores.csv');supports=read_csv(formal/'support_manifest.csv');shifts={}
        for shift,counts in COUNTS[category].items():
            selected=[r for r in supports if r['branch']=='target_matched' and r['shift']==shift]
            if [int(r['support_index']) for r in selected]!=list(range(1,9)):raise ValueError('support order differs')
            sup=[]
            for r in selected:
                if r.get('category',category)!=category:raise ValueError('support category mismatch')
                s=sample(lookup[r['relative_path']]);s['support_index']=int(r['support_index'])
                if s['label'] or s['shift']!=shift:raise ValueError('invalid support')
                sup.append(s)
            baseline=[r for r in scores if r['shift']==shift and r['method']=='MAD_EQUAL']
            union={r['sample_id']:r for r in scores if r['shift']==shift and r['method']=='TARGET_MATCHED_K8_MAD'}
            samples=[]
            for r in baseline:
                s=sample(lookup[r['sample_id']]);u=union.pop(r['sample_id'])
                if r.get('category',category)!=category or s['label']!=int(r['label']) or s['label']!=int(u['label']):raise ValueError('formal image identity differs')
                s.update(source_score=float(r['image_score']),union_score=float(u['image_score']));samples.append(s)
            if union or (sum(s['label']==0 for s in samples),sum(s['label']==1 for s in samples))!=counts:raise ValueError('heldout counts differ')
            held=[s['sample_id'] for s in samples];support=[s['sample_id'] for s in sup]
            actual={k for k,r in lookup.items() if r.domain=='target' and r.shift==shift}
            if len(set(held))!=len(held) or set(held)&set(support) or set(held)|set(support)!=actual:raise ValueError('heldout/support identities differ')
            shifts[shift]=dict(samples=samples,supports=sup,normal_count=counts[0],anomaly_count=counts[1])
        train_rows=[sample(r) for r in train[category]]
        for i,s in enumerate(train_rows):s.update(source_index=i,source_role='calibration' if i%8==0 else 'memory_fit')
        for i,s in enumerate(s for s in train_rows if s['source_role']=='calibration'):s['calibration_index']=i+1
        legacy={canonical(r['path'],category):{k:(int(v) if k=='label' else None if k=='mask_path' and not v else v) for k,v in r.items() if k!='protocol_role'}
                for r in old if r['dataset']=='robustad' and r['category']==category}
        result['categories'][category]=dict(protocol_id=complete['protocol_id'],source_complete=sc,shifts=shifts,legacy_records=legacy,
            train=train_rows,calibration=[s for s in train_rows if s['source_role']=='calibration'],source_evaluation=[sample(r) for r in source_eval],
            formal_sha256={n:sha(formal/n) for n in FORMAL_FILES},source_sha256={n:sha(src/n) for n in SOURCE_FILES},
            historical_bank_sha256=None,bank_origin='original_completed_source_v6_tree_initial_byte_enrollment_required')
    return result


def code_hashes():
    root=Path(__file__).resolve().parents[3]
    shared=('DINOv3/MADEqual/aebad_s_segment_pair_crossdomain_p0/method.py',
        'DINOv3/MADEqual/aebad_s_segment_pair_crossdomain_p0/analysis.py',
        'DINOv3/MADEqual/aebad_s_segment_pair_crossdomain_p0/verification.py',
        'DINOv3/MADEqual/aebad_s_segment_pair_crossdomain_p0/inputs.py',
        'DINOv3/MADEqual/aebad_s_neighborhood_segment_p0/gates.py',
        'DINOv3/MADEqual/aebad_s_neighborhood_segment_p0/analysis.py',
        'DINOv3/MADEqual/aebad_s_neighborhood_segment_p0/prediction.py',
        'DINOv3/MADEqual/aebad_s_union_diagnostics/spatial.py',
        'DINOv3/MADEqual/aebad_s_sparc_p0/runtime.py',
        'DINOv3/MADEqual/aebad_s_target_local_segment/geometry.py',
        'DINOv3/MADEqual/aebad_s_ch1_scoring_inheritance/scoring.py',
        'DINOv3/MADEqual/aebad_s_projection_layer_diagnostic/numerics.py',
        'DINOv3/MADEqual/aebad_s_projection_layer_diagnostic/metrics.py',
        'DINOv3/MADEqual/robustad_target_fewshot_p0/scoring.py',
        'DINOv3/MADEqual/robustad_target_fewshot_p0/support.py',
        'DINOv3/ShiftADD/aebad_s_p0/memory.py',
        'DINOv3/relation_reliability/superadd_host.py','DINOv3/relation_reliability/datasets.py',
        'DINOv3/relation_reliability/official_metrics.py','DINOv3/relation_oracle_cache/extract.py',
        'external_baselines/superadd_external/memory_variants.py',
        'external_baselines/SuperADD/tracks/industrial/src/industrial/nearest_neighbor.py',
        'DINOv3/MADEqual/image_granularity_calibration/scoring.py','DINOv3/MADEqual/connected_region/scoring.py',
        'external_baselines/superadd_external/dataset_adapter.py')
    return {p.relative_to(root).as_posix():sha(p) for p in [*sorted(Path(__file__).parent.glob('*.py')),*[root/s for s in shared]]}


def context(paths,category,*,enroll=False):
    if category not in CATEGORIES:raise ValueError('only MetalParts and PCB are in scope')
    paths={k:Path(v).resolve() for k,v in paths.items() if v is not None}
    cp=Path(__file__).with_name('frozen_catalog.json')
    if sha(cp)!=CATALOG_SHA:raise ValueError('frozen catalog changed')
    catalog=read_json(cp);cat=catalog['categories'][category];formal=paths['formal_'+category.lower()]
    src=paths['source_root']/'categories/robustad'/category;root=paths['output_root']/category
    for n,h in cat['formal_sha256'].items():
        if sha(formal/n)!=h:raise ValueError('formal input changed: '+str(formal/n))
    for n,h in cat['source_sha256'].items():
        if sha(src/n)!=h:raise ValueError('original Source input changed: '+str(src/n))
    for n,h in catalog['shared_source_files'].items():
        if sha(paths['source_root']/n)!=h:raise ValueError('Source manifest/protocol changed')
    with np.load(src/'calibration/calibration_state.npz',allow_pickle=False) as f:
        med=f['mad_medians'].copy();scales=f['mad_scales'].copy()
    if med.dtype!=np.float32 or scales.dtype!=np.float32 or med.shape!=(4,) or scales.shape!=(4,) or not np.isfinite(med).all() or not np.isfinite(scales).all() or np.any(scales<=0):raise ValueError('invalid original category patch MAD')
    base=dict(tool=TOOL,category=category,catalog_sha256=CATALOG_SHA,protocol=PROTOCOL,code=code_hashes())
    saved=read_json(root/'inputs.json') if (root/'inputs.json').is_file() else None
    bank=src/'memory/official_banks.npz'
    bank_provenance=None
    if 'rebuilt_source_root' in paths:
        from . import rebuild
        bank=rebuild.directory(paths,category)/'memory/official_banks.npz'
        provisional=dict(paths=paths,category=category,cat=cat,catalog=catalog,source=src)
        bank_provenance=rebuild.snapshot(paths,category,rebuild.plan(provisional))
        if enroll and bank_provenance['bank_origin']=='explicit_reconstruction_pending':
            raise FileNotFoundError('explicit rebuild-source must complete before enrollment: '+str(bank))
    if saved:
        if saved['frozen']['base']!=base or digest(saved['frozen'])!=saved['identity']:raise ValueError('frozen code/protocol/category changed; new output root required')
        bank_hash=saved['frozen']['bank_sha256']
    else:
        bank_hash=sha(bank) if enroll and bank.is_file() else None
        if enroll and bank_hash is None:raise FileNotFoundError('Source bank required for initial enrollment; use explicit rebuild-source for a separate reconstruction: '+str(bank))
    bank_size=saved['frozen']['bank_size_bytes'] if saved else (bank.stat().st_size if bank.is_file() else None)
    frozen=dict(base=base,bank_sha256=bank_hash,bank_size_bytes=bank_size)
    if bank_provenance is not None:frozen['bank_provenance']=bank_provenance
    ctx=dict(paths=paths,category=category,cat=cat,catalog=catalog,root=root,source=src,bank=bank,
        frozen=frozen,identity=digest(frozen),medians=med,scales=scales)
    ctx['extraction_identity']=digest(dict(category=category,checkpoint_sha256=CHECKPOINT_SHA,config=catalog['host_config'],precision=PRECISION,
        host_code={k:v for k,v in base['code'].items() if k.endswith(('superadd_host.py','support.py','robustad_target_fewshot_p0/scoring.py'))}))
    if saved and saved['identity']!=ctx['identity']:raise ValueError('input freeze changed')
    if enroll and not saved:
        for key,p in paths.items():
            if key not in ('project','output_root') and (root==p or root in p.parents or p in root.parents):raise ValueError('output overlaps input')
        if root.drive and root.drive.upper()!='G:':raise ValueError('local outputs must stay on G:')
        if root.exists():raise FileExistsError('unregistered output directory: '+str(root))
        root.mkdir(parents=True)
        write_json(root/'inputs.json',dict(identity=ctx['identity'],frozen=frozen,paths={k:str(v) for k,v in paths.items()},
            bank_attestation=bank_provenance or 'first enrollment from original source tree; historical bank byte hash absent in local archive'))
        write_json(root/'protocol.json',PROTOCOL)
    return ctx


def selected(ctx,shift=None):
    names=list(ctx['cat']['shifts']) if not shift or shift=='all' else [shift]
    if any(s not in ctx['cat']['shifts'] for s in names):raise ValueError('shift not in this category')
    return names


def key(sample):return hashlib.sha256(sample['relative_path'].encode()).hexdigest()[:24]


def image(ctx,sample):
    path=ctx['paths']['data']/safe(sample['relative_path'])
    if sha(path)!=sample['image_sha256']:raise ValueError('image bytes changed: '+str(path))
    with Image.open(path) as im:
        if im.size!=(sample['image_width'],sample['image_height']):raise ValueError('image dimensions changed')
        return im.convert('RGB')


def gt(ctx,sample):
    if sample['label']==0:return np.zeros((sample['image_height'],sample['image_width']),bool)
    path=ctx['paths']['data']/safe(sample['mask_path'])
    if sha(path)!=sample['gt_sha256']:raise ValueError('GT bytes changed')
    with Image.open(path) as im:mask=np.asarray(im.convert('L'))>0
    if mask.shape!=(sample['image_height'],sample['image_width']) or not mask.any():raise ValueError('missing/empty required GT')
    return mask


def ledger(ctx):
    p=ctx['root']/'cache_status.json'
    state=read_json(p) if p.is_file() else dict(tool=TOOL,identity=ctx['identity'],records=[])
    if state['identity']!=ctx['identity']:raise ValueError('cache identity differs')
    seen=set()
    for r in state['records']:
        k=(r['kind'],r['shift'],r['sample_id'])
        if k in seen or sha(ctx['root']/safe(r['file']))!=r['sha256']:raise ValueError('recorded cache changed/duplicated; preserve history')
        seen.add(k)
    actual={p.relative_to(ctx['root']).as_posix() for sub in ('raw_cache','support_features','source_calibration_raw','source_evaluation_raw') for p in (ctx['root']/sub).rglob('*') if p.is_file()}
    if actual!={r['file'] for r in state['records']}:raise ValueError('orphan, missing or unfinished cache; do not reapprove')
    return state


def save_cache(ctx,state,kind,sample,arrays,meta):
    index=sample.get('support_index',sample.get('calibration_index'))
    name=(f'support_features/{sample["shift"]}/{index:03d}.npz' if kind=='support' else
          f'source_calibration_raw/{index:03d}.npz' if kind=='calibration' else
          f'source_evaluation_raw/{key(sample)}.npz' if kind=='source' else f'raw_cache/{sample["shift"]}/{key(sample)}.npz')
    path=ctx['root']/name;tmp=path.with_suffix('.npz.tmp')
    if path.exists() or tmp.exists():raise ValueError('cannot overwrite or adopt unrecorded cache')
    path.parent.mkdir(parents=True,exist_ok=True)
    header=dict(meta,identity=ctx['identity'],category=ctx['category'],shift=sample['shift'],sample_id=sample['sample_id'],relative_path=sample['relative_path'],image_sha256=sample['image_sha256'],
                layers=list(LAYERS),dtype='float32',distance_unit=PROTOCOL['distance_unit'],extraction_identity=ctx['extraction_identity'])
    with tmp.open('xb') as f:np.savez_compressed(f,metadata=np.asarray(json.dumps(header,sort_keys=True)),**arrays)
    tmp.replace(path)
    r=dict(kind=kind,category=ctx['category'],shift=sample['shift'],sample_id=sample['sample_id'],file=name,sha256=sha(path))
    state['records'].append(r);write_json(ctx['root']/'cache_status.json',state)
    return r


def load_cache(ctx,record,sample):
    path=ctx['root']/safe(record['file'])
    if sha(path)!=record['sha256']:raise ValueError('cache changed since its original record')
    with np.load(path,allow_pickle=False) as f:
        meta=json.loads(str(f['metadata'].item()));a={k:f[k].copy() for k in f.files if k!='metadata'}
    for k,v in dict(identity=ctx['identity'],category=ctx['category'],shift=sample['shift'],sample_id=sample['sample_id'],relative_path=sample['relative_path'],image_sha256=sample['image_sha256'],layers=list(LAYERS),dtype='float32',distance_unit=PROTOCOL['distance_unit'],extraction_identity=ctx['extraction_identity']).items():
        if meta.get(k)!=v:raise ValueError('cache header/provenance differs: '+k)
    return a,meta


def available_ram():
    """Credit clean inactive file cache, excluding shmem/dirty/writeback/mapped pages."""
    import psutil
    available=int(psutil.virtual_memory().available);c=Path('/sys/fs/cgroup')
    try:
        limit=(c/'memory.max').read_text().strip()
        if limit!='max':
            current=int((c/'memory.current').read_text());stats={k:int(v) for k,v in (line.split() for line in (c/'memory.stat').read_text().splitlines())}
            reserve=sum(stats.get(k,0) for k in ('shmem','file_dirty','file_writeback','file_mapped'))
            credit=max(0,min(stats.get('inactive_file',0),stats.get('file',0)-reserve))
            available=min(available,max(0,int(limit)-current)+credit)
    except (OSError,ValueError):pass
    return available
