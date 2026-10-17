"""Joint two-domain identity and honest input inventory. Never imports Torch."""
from __future__ import annotations
import hashlib
import io
import json
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from DINOv3.MADEqual.aebad_s_neighborhood_segment_p0 import inputs as common
from DINOv3.MADEqual.aebad_s_union_support_loo.analysis import validate_arrays,FORMAT
from .method import LAYERS,BRANCHES,validate_geometry

TOOL='aebad_s_segment_pair_crossdomain_v1'
DOMAINS=('illumination','view')
CATALOG_SHA='3a0082fa6c75e4b04b4c0c4b7be27287afb7422157b3357c79eeafbb133af77c'
PROTOCOL=dict(tool=TOOL,domains=list(DOMAINS),branches=list(BRANCHES),layers=list(LAYERS),seed=42,
    distance_unit='euclidean_over_1280_once',distance_atol=1e-7,distance_rtol=1e-6,score_atol=1e-5,score_rtol=0,
    historical_score_policy='cross_environment_fixed_atol_v1',historical_score_atol=1e-4,
    candidate='stable_top2_different_supports_manifest_order',geometry='original_geometry_layer_bounded_distances',
    pair_radius=1,pair_votes='ceil(actual_neighbors/2)',exclude_center=True,padding=False,no_neighbors='full_union',
    support_coordinate_constraint=False,normalization=None,image_score='frozen_Source_CH1_peak_local_0.5_0.5',
    localization='Source_patch_MAD_equal4_original_bilinear',R_scale='float32_Source_scale_plus_1e-8',
    instance_connectivity=8,ring_radius_original_pixels=16,ring_metric='Chebyshev_excludes_all_GT',
    small_instances='smallest_ceil_N_over_4_original_area_tie_sample_id_instance_id',
    nstrips=200,aupro_FPR_max=.3,domain_macro='illumination_view_unweighted_background_excluded')
QUERY=dict(brightness_factor=1,resize_factor=.625,patch_size=640,patch_overlap=128,model_patch_size=16,
           evaluation_downscale=4,dino_training=False,trainable_parameters=0)
PRECISION=dict(float32_matmul_precision='highest',cuda_matmul_allow_tf32=False,cudnn_allow_tf32=True)
sha=common.sha;digest=common.digest;read_json=common.json_file;write_json=common.write_json;write_csv=common.write_csv
safe=common.safe


def csv_file(path):return common.read_csv(Path(path).read_bytes())


def defaults(root):
    root=Path(root)
    return dict(project=root,formal_dir=root/'server_archives/aebad_s_target_fewshot_seed42_v1_formal_results/aebad_s_target_fewshot_seed42_v1',
        image_state=root/'DINOv3/outputs/mad_equal/aebad_s_ch1_scoring_local_20261016_034133/image_calibration_state.json',
        data=root/'datasets/AeBAD.zip',memory=root/'DINOv3/outputs/mad_equal/aebad_s_target_fewshot_seed42_v1/official_memory.pt',
        official=root/'external_baselines/SuperADD',checkpoint=root/'external_baselines/SuperADD/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth',
        dinov3=root/'external_baselines/dinov3',cache_root=root/'DINOv3/outputs/mad_equal/aebad_s_segment_pair_crossdomain_cache_v1')


def code_hashes():
    root=Path(__file__).resolve().parents[3]
    files=list(Path(__file__).parent.glob('*.py'))
    files += [root/p for p in (
        'DINOv3/MADEqual/aebad_s_target_local_segment/geometry.py',
        'DINOv3/MADEqual/aebad_s_neighborhood_segment_p0/gates.py',
        'DINOv3/MADEqual/aebad_s_neighborhood_segment_p0/analysis.py',
        'DINOv3/MADEqual/aebad_s_neighborhood_segment_p0/prediction.py',
        'DINOv3/MADEqual/aebad_s_neighborhood_segment_p0/inputs.py',
        'DINOv3/MADEqual/aebad_s_ch1_scoring_inheritance/scoring.py',
        'DINOv3/MADEqual/aebad_s_projection_layer_diagnostic/numerics.py',
        'DINOv3/MADEqual/aebad_s_projection_layer_diagnostic/metrics.py',
        'DINOv3/MADEqual/aebad_s_union_diagnostics/spatial.py',
        'DINOv3/MADEqual/aebad_s_union_support_loo/analysis.py',
        'DINOv3/MADEqual/robustad_target_fewshot_p0/scoring.py',
        'DINOv3/MADEqual/robustad_target_fewshot_p0/support.py',
        'DINOv3/MADEqual/aebad_s_target_fewshot/runner.py',
        'DINOv3/MADEqual/aebad_s_sparc_p0/runtime.py',
        'DINOv3/relation_reliability/superadd_host.py',
        'DINOv3/relation_reliability/official_metrics.py')]
    return {p.relative_to(root).as_posix():sha(p) for p in sorted(files)}


class Data:
    """PNG byte identity and headers at preflight; GT decoding only at evaluate."""
    def __init__(self,path):
        self.path=Path(path);self.zip=zipfile.ZipFile(path) if self.path.is_file() else None
        self.root=None if self.zip else common.dataset_root(path)
        if self.zip:
            self.names={n[len('AeBAD/AeBAD_S/'):] for n in self.zip.namelist() if n.startswith('AeBAD/AeBAD_S/')}
        else:self.names={p.relative_to(self.root).as_posix() for p in (self.root/'test').rglob('*') if p.is_file()}
    def __enter__(self):return self
    def __exit__(self,*args):
        if self.zip:self.zip.close()
    def bytes(self,relative,expected):
        relative=safe(relative)
        data=self.zip.read('AeBAD/AeBAD_S/'+relative) if self.zip else (self.root/relative).read_bytes()
        if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('raster changed: '+relative)
        return data
    def image(self,sample):
        with Image.open(io.BytesIO(self.bytes(sample['relative_path'],sample['image_sha256']))) as im:return im.convert('RGB')
    def gt(self,sample):
        if not sample['label']:return np.zeros((sample['image_height'],sample['image_width']),bool)
        rel=sample['relative_path'].replace('test/','ground_truth/',1)
        with Image.open(io.BytesIO(self.bytes(rel,sample['gt_sha256']))) as im:mask=np.asarray(im.convert('L'))>0
        if mask.shape!=(sample['image_height'],sample['image_width']) or not mask.any():raise ValueError('required GT empty or mismatched')
        return mask


def context(paths,verify_data=True,verify_gt=False):
    paths={k:Path(v).resolve() for k,v in paths.items()}
    catalog_path=Path(__file__).with_name('frozen_catalog.json')
    if sha(catalog_path)!=CATALOG_SHA:raise ValueError('joint frozen catalog modified')
    catalog=read_json(catalog_path)
    if tuple(catalog['domains'])!=DOMAINS:raise ValueError('both domains must freeze together')
    for name,checksum in catalog['formal_sha256'].items():
        if sha(paths['formal_dir']/name)!=checksum:raise ValueError('formal asset changed: '+name)
    if sha(paths['image_state'])!=catalog['image_state_sha256']:raise ValueError('frozen Source image MAD changed')
    state=read_json(paths['image_state'])
    if digest({k:v for k,v in state.items() if k!='state_sha256'})!=state['state_sha256']:
        raise ValueError('image MAD internal identity invalid')
    if state['patch_mad_frozen_sha256']!=catalog['formal_sha256']['source_mad_state.npz']:raise ValueError('patch and image MAD inconsistent')
    with np.load(paths['formal_dir']/'source_mad_state.npz',allow_pickle=False) as f:
        medians=f['medians'].copy();scales=f['scales'].copy()
    formal=csv_file(paths['formal_dir']/'sample_scores.csv');supports=csv_file(paths['formal_dir']/'support_manifest.csv')
    for domain in DOMAINS:
        d=catalog['domains'][domain]
        for method,key in (('MAD_EQUAL','source_score'),('TARGET_MATCHED_K8_MAD','union_score')):
            rows=[r for r in formal if r['domain']==domain and r['method']==method]
            if [(r['sample_id'],r['relative_path'],int(r['label']),float(r['image_score'])) for r in rows]!=[
                (s['sample_id'],s['relative_path'],s['label'],s[key]) for s in d['samples']]:
                raise ValueError('formal evaluation set or sample identity differs: '+domain)
        ss=[r for r in supports if r['domain']==domain and r['branch']=='target_matched']
        if [(int(r['support_index']),r['relative_path'],int(r['label'])) for r in ss]!=[
            (s['support_index'],s['relative_path'],0) for s in d['supports']]:raise ValueError('original K8 order differs')
        selected={s['relative_path'] for s in d['supports']};held={s['relative_path'] for s in d['samples']}
        if len(selected)!=8 or selected&held or len(held)!=len(d['samples']):raise ValueError('support exclusions/duplicates')
        if (sum(s['label']==0 for s in d['samples']),sum(s['label']==1 for s in d['samples']))!=(d['normal_count'],d['anomaly_count']):
            raise ValueError('domain counts differ from catalog')
    if verify_data:
        with Data(paths['data']) as data:
            for domain,d in catalog['domains'].items():
                actual={n for n in data.names if n.startswith('test/') and len(n.split('/'))==4 and n.split('/')[2]==domain
                        and n.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif','.tiff')) and not Path(n).name.startswith('._') and '__MACOSX' not in n}
                if actual!={s['relative_path'] for s in d['samples']+d['supports']}:raise ValueError('dataset/CSV full heldout+support set mismatch: '+domain)
                for s in d['samples']+d['supports']:
                    b=data.bytes(s['relative_path'],s['image_sha256'])
                    with Image.open(io.BytesIO(b)) as im:
                        if (im.height,im.width)!=(s['image_height'],s['image_width']):raise ValueError('image dimensions changed')
                    if verify_gt and s.get('label')==1:data.bytes(s['relative_path'].replace('test/','ground_truth/',1),s['gt_sha256'])
    extraction=dict(query=QUERY,precision=PRECISION,memory_sha256=catalog['memory_sha256'],
                    checkpoint_sha256=catalog['checkpoint_sha256'],layers=list(LAYERS),dimension=1280,
                    source_code={k:v for k,v in code_hashes().items() if k.endswith(('superadd_host.py','support.py','robustad_target_fewshot_p0/scoring.py'))})
    frozen=dict(protocol=PROTOCOL,catalog_sha256=CATALOG_SHA,code=code_hashes(),extraction=extraction)
    ctx=dict(paths=paths,catalog=catalog,state=state,medians=medians,scales=scales,
             frozen=frozen,extraction_identity=digest(extraction),distance_references=None,
             data_verification=dict(image_set_bytes_and_headers=verify_data,GT_bytes=verify_data and verify_gt,GT_decoded=False))
    if 'distance_reference_root' in paths:
        from .references import manifest
        ctx['distance_references']=manifest(ctx,paths['distance_reference_root'])
        frozen['distance_reference_manifest_sha256']=sha(paths['distance_reference_root']/'manifest.json')
    ctx['identity']=digest(frozen)
    return ctx


def new_output(ctx,path,stage):
    path=Path(path).resolve()
    if path.exists():raise ValueError('new output directory required; no overwriting')
    if path.drive and path.drive.upper()!='G:':raise ValueError('Windows outputs/scratch must stay on G:')
    for role,source in ctx['paths'].items():
        if role=='project':continue
        if path==source or path in source.parents or source in path.parents:raise ValueError('output overlaps input')
    path.mkdir(parents=True)
    write_json(path/'protocol.json',PROTOCOL)
    write_json(path/'input_manifest.json',dict(identity=ctx['identity'],frozen=ctx['frozen']))
    write_json(path/'status.json',dict(tool=TOOL,stage=stage,status='running',identity=ctx['identity']))
    return path


def save_npz(path,arrays,meta):
    path=Path(path)
    if path.exists() or path.with_suffix('.tmp').exists():raise ValueError('cache already exists or orphaned temporary file')
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    with tmp.open('xb') as f:np.savez_compressed(f,metadata=np.asarray(json.dumps(meta,sort_keys=True)),**arrays)
    tmp.replace(path)


def load_npz(path,checksum):
    if sha(path)!=checksum:raise ValueError('cache changed since original record: '+str(path))
    with np.load(path,allow_pickle=False) as f:
        meta=json.loads(str(f['metadata'].item()));arrays={k:f[k].copy() for k in f.files if k!='metadata'}
    return arrays,meta


def validate_raw(ctx,sample,arrays,meta,require_geometry=True):
    if (meta['format']!=FORMAT or meta['tool']!=TOOL or meta['identity']!=ctx['identity']
            or meta['domain']!=sample['domain'] or meta['sample_id']!=sample['sample_id']
            or meta['relative_path']!=sample['relative_path'] or meta['image_sha256']!=sample['image_sha256']
            or meta['distance_unit']!=PROTOCOL['distance_unit']):raise ValueError('crossdomain cache identity mismatch')
    validate_arrays(arrays,meta)
    expected=ctx['catalog']['domains'][sample['domain']]['supports']
    if [(r['support_index'],r['relative_path'],r['image_sha256']) for r in meta['supports']]!=[
        (r['support_index'],r['relative_path'],r['image_sha256']) for r in expected]:raise ValueError('cache support identities/order changed')
    if meta['output_shape']!=[sample['image_height']//4,sample['image_width']//4]:raise ValueError('evaluation geometry changed')
    if require_geometry:validate_geometry(arrays)


def cache_state(ctx,root=None):
    root=Path(root or ctx['paths']['cache_root'])
    if not root.exists():return dict(tool=TOOL,identity=ctx['identity'],records=[],supports=[],sources={})
    frozen=read_json(root/'input_manifest.json');state=read_json(root/'cache_status.json')
    if frozen!=dict(identity=ctx['identity'],frozen=ctx['frozen']) or state['identity']!=ctx['identity']:
        raise ValueError('cache was frozen under different joint method/input/code; no re-approval')
    for kind,key in (('records','sample_id'),('supports','support_index')):
        seen=set()
        for record in state[kind]:
            pair=(record['domain'],record[key])
            if pair in seen or pair[0] not in DOMAINS:raise ValueError('duplicate/out-of-scope cache history')
            seen.add(pair)
            rows=ctx['catalog']['domains'][pair[0]]['samples' if kind=='records' else 'supports']
            sample=next((s for s in rows if s[key]==pair[1]),None)
            if sample is None:raise ValueError('cache record outside frozen domain set')
            expected_file=f'raw_cache/{pair[0]}/{pair[1]}.npz' if kind=='records' else f'support_features/{pair[0]}/{pair[1]:03d}.npz'
            if record['file']!=expected_file:raise ValueError('cache history file is outside its canonical domain scope')
            if kind=='supports' and (record['relative_path']!=sample['relative_path'] or record['image_sha256']!=sample['image_sha256']):
                raise ValueError('support history identity/order mismatch')
    for record in state['records']+state['supports']:
        path=record_path(root,record)
        if not path.exists() and record in state['supports']:continue # geometry-complete raw does not require feature files
        if sha(path)!=record['sha256']:raise ValueError('cache/support modified after prior record')
    expected={r['file'] for r in state['records']+state['supports'] if r.get('storage')!='external'}
    present={p.relative_to(root).as_posix() for sub in ('raw_cache','support_features') for p in (root/sub).rglob('*.npz')}
    unfinished=[p for sub in ('raw_cache','support_features') for p in (root/sub).rglob('*.tmp')]
    if unfinished or present-expected or any(not (root/safe(r['file'])).is_file() for r in state['records']):
        raise ValueError('orphan/missing raw files; history retained, no automatic approval')
    return state


def record_path(root,record):
    return Path(record['external_path']) if record.get('storage')=='external' else Path(root)/safe(record['file'])


def support_arrays(path,sample):
    with np.load(path,allow_pickle=False) as f:
        if str(f['protocol_id'].item())!='mad_equal_aebad_s_target_fewshot_seed42_v1' or str(f['relative_path'].item())!=sample['relative_path']:
            raise ValueError('original support header identity mismatch')
        grid=tuple(int(v) for v in f['grid']);layers={l:f[f'layer_{l}'].copy() for l in LAYERS}
    for a in layers.values():
        if a.dtype!=np.float32 or a.shape!=(int(np.prod(grid)),1280) or not np.isfinite(a).all():raise ValueError('invalid full original support features')
    return grid,layers


def external_manifest(ctx,root,kind):
    if root is None:return None
    root=Path(root);data=read_json(root/'manifest.json')
    if (data['format']!=kind or data['status']!='verified' or data['extraction_identity']!=ctx['extraction_identity']):
        raise ValueError('unverified or incompatible '+kind)
    seen=set()
    for r in data['records']:
        key=(r['domain'],r.get('sample_id',r.get('support_index')))
        if key in seen or r['domain'] not in DOMAINS:raise ValueError('external cache scope/duplicates')
        seen.add(key)
        domain=ctx['catalog']['domains'][r['domain']]
        sample=next(s for s in domain['samples' if kind=='original_query_features_v1' else 'supports']
                    if s.get('sample_id',s.get('support_index'))==key[1])
        if (r['relative_path']!=sample['relative_path'] or r['image_sha256']!=sample['image_sha256']
                or sha(root/safe(r['file']))!=r['sha256']):raise ValueError('external feature identity/hash mismatch')
    return data


def inventory(ctx,query_features=None,support_features=None):
    report=dict(tool=TOOL,joint_identity=ctx['identity'],extraction_identity=ctx['extraction_identity'],query=QUERY,precision=PRECISION,
                protocol=PROTOCOL,domains={},assets=[],identity_mismatches=[],data_verification=ctx.get('data_verification'),
                Torch_imported=False,model_loaded=False,GT_decoded=False)
    try:state=cache_state(ctx)
    except Exception as error:state=dict(records=[],supports=[]);report['identity_mismatches'].append(str(error))
    sources={}
    for name,path,kind in (('query',query_features,'original_query_features_v1'),('support',support_features,'original_support_features_v1')):
        try:sources[name]=external_manifest(ctx,path,kind)
        except Exception as error:sources[name]=None;report['identity_mismatches'].append(str(error))
    for domain,d in ctx['catalog']['domains'].items():
        records={r['sample_id']:r for r in state['records'] if r['domain']==domain}
        verified={}
        for sample in d['samples']:
            if sample['sample_id'] in records:
                try:
                    r=records[sample['sample_id']];a,m=load_npz(ctx['paths']['cache_root']/safe(r['file']),r['sha256']);validate_raw(ctx,sample,a,m)
                    verified[sample['sample_id']]=r
                except Exception as error:report['identity_mismatches'].append(f'{domain}/{sample["sample_id"]}: {error}')
        records=verified
        q={r['sample_id'] for r in (sources['query'] or {}).get('records',[]) if r['domain']==domain}
        support_keys={r['support_index'] for r in state['supports'] if r['domain']==domain and record_path(ctx['paths']['cache_root'],r).is_file()}
        support_keys|={r['support_index'] for r in (sources['support'] or {}).get('records',[]) if r['domain']==domain}
        missing=[s['sample_id'] for s in d['samples'] if s['sample_id'] not in records]
        report['domains'][domain]=dict(normal_count=d['normal_count'],anomaly_count=d['anomaly_count'],heldout_images=len(d['samples']),
            supports=d['supports'],heldout_set_sha256=digest([(s['sample_id'],s['relative_path'],s['label']) for s in d['samples']]),
            reusable_complete_cache_images=len(records),reusable_query_feature_images=len(set(missing)&q),
            missing_query_extraction_images=len(set(missing)-q),reusable_support_features=len(support_keys),
            missing_support_indices=sorted(set(range(1,9))-support_keys),
            reuse_priority='A_complete_capture_then_B_verified_query_features_then_C_original_extraction')
    for key,checksum in (('memory',ctx['catalog']['memory_sha256']),('checkpoint',ctx['catalog']['checkpoint_sha256'])):
        path=ctx['paths'][key];exists=path.is_file();status='missing'
        if exists:
            status='reusable' if sha(path)==checksum else 'identity_mismatch'
            if status=='identity_mismatch':report['identity_mismatches'].append(key+' hash mismatch')
        report['assets'].append(dict(input=key,path=str(path),status=status,expected_sha256=checksum))
    for key,required in (('dinov3','hubconf.py'),('official','config.json')):
        path=ctx['paths'][key];report['assets'].append(dict(input=key,path=str(path),status='reusable' if (path/required).is_file() else 'missing'))
    report['assets'].extend([dict(input='formal_metadata_and_patch_MAD',path=str(ctx['paths']['formal_dir']),status='reusable',
        sha256=ctx['catalog']['formal_sha256']),dict(input='Source_image_MAD',path=str(ctx['paths']['image_state']),status='reusable',
        sha256=ctx['catalog']['image_state_sha256']),dict(input='original_images_and_GT',path=str(ctx['paths']['data']),status='reusable',
        verification=ctx.get('data_verification'))])
    report['status']='inputs_indexed_capture_required' if any(v['missing_query_extraction_images'] or v['reusable_query_feature_images'] for v in report['domains'].values()) else 'cache_ready'
    if report['identity_mismatches']:report['status']='blocked_identity_mismatch'
    return report
