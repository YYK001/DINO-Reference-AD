"""Normal-only origins, stable folds, recent raw asset schemas and read-only preflight."""
from pathlib import Path
from importlib.metadata import version,PackageNotFoundError
from . import storage as st
from .protocol import CONFIG,DOMAINS,LAYERS,VERSION
from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation import inputs as previous
from DINOv3.MADEqual.aebad_s_support_witness_segment_p0 import fresh_inputs as fi

ROOT=Path(__file__).resolve().parents[3]
def catalog(a):return st.sealed_json(a.run_root/'catalog.json')['catalog']
def identity(a):return st.sealed_json(a.run_root/'inputs.json')['identity']
def key(row):return st.digest(row['relative_path'])[:24]
def normal_path(a,row):return a.cache_root/'normal_features'/key(row)
def item_path(a,item):return a.cache_root/'training_queries'/item['item_id']
def train_source(c):return [r for r in c['source'] if not r['calibration']]
def allowed(c,domain):return train_source(c)+(c['supports'][domain] if domain in DOMAINS else [])


def validate_source(c):
    rows=c['source']
    if len(rows)!=521 or [r['relative_path'] for r in rows]!=sorted(r['relative_path'] for r in rows):raise ValueError('original sorted 521 Source required')
    if len({r['relative_path'] for r in rows})!=521 or any(r['source_index']!=i or r['calibration']!=(i%8==0) or r.get('label',0)!=0 for i,r in enumerate(rows)):
        raise ValueError('Source identity/index/mod8 calibration split differs')


def validate_frozen_origins(c):
    validate_source(c);frozen=st.read_json(Path(__file__).with_name('frozen_origins.json'))
    source_fields=('relative_path','image_sha256','image_height','image_width')
    def equal(rows,original,fields):return [tuple(r[k] for k in fields) for r in rows]==[tuple(r[k] for k in fields) for r in original]
    if not equal(c['source'],frozen['source'],source_fields):raise ValueError('original Source image identities differ')
    if not equal(c['supports']['background'],frozen['background_supports'],(*source_fields,'sample_id','support_index','label')):raise ValueError('original background K8 differs')
    if not equal([r for r in c['samples'] if r['domain']=='background'],frozen['background_samples'],(*source_fields,'sample_id','label')):raise ValueError('original full background population differs')
    gt_fields=('domain','sample_id','relative_path','label','gt_sha256')
    if [[r.get(k) for k in gt_fields] for r in c['samples']]!=[[r[k] for k in gt_fields] for r in frozen['evaluation_GT']]:raise ValueError('original evaluation GT identities differ')


def folds(c):
    rows=train_source(c)
    return {r['relative_path']:'AB'[i%2] for i,r in enumerate(rows)}


def policy(c,row,phase):
    """All synthetic/augmented versions inherit ORIGINAL origin exclusion."""
    if row.get('calibration'):raise ValueError('calibration origin forbidden in training')
    bypath={r['relative_path']:r for r in allowed(c,phase)}
    if row['relative_path'] not in bypath:raise ValueError('test/foreign-domain training origin')
    if row.get('source_index') is not None:
        group=folds(c);opposite='B' if group[row['relative_path']]=='A' else 'A'
        source_rows=[r['relative_path'] for r in train_source(c) if group[r['relative_path']]==opposite]
        if phase=='source':
            count=7+st.seed(VERSION,'sim_count',row['relative_path'])%2
            ordered=sorted(source_rows,key=lambda p:st.seed(VERSION,'sim_support',row['relative_path'],p))
            if len(ordered)<count:raise ValueError('insufficient cross-fold simulated supports')
            targets=ordered[:count]
        else:targets=[r['relative_path'] for r in c['supports'][phase]]
        return dict(source='fold_'+opposite,source_origins=source_rows,target_origins=targets,origin=row['relative_path'],phase=phase)
    if phase=='source' or row['domain']!=phase:raise ValueError('Target origin outside its own domain')
    return dict(source='formal',source_origins=[r['relative_path'] for r in train_source(c)],
        target_origins=[r['relative_path'] for r in c['supports'][phase] if r['relative_path']!=row['relative_path']],
        origin=row['relative_path'],phase=phase)


def recent(path):
    p=Path(path)
    if (p/'inputs.json').exists() and (p/'catalog.json').exists():
        value=st.sealed_json(p/'inputs.json');c=st.sealed_json(p/'catalog.json')['catalog']
        if value['configuration']['version']!='normal_correspondence_residual_v1':raise ValueError('not an original correspondence raw asset schema')
        if value['catalog']!=st.digest(c):raise ValueError('recent catalog differs')
        fi.validate_catalog(c)
        validate_frozen_origins(c)
        return dict(kind='correspondence',run=p,cache=Path(value['cache_root']),catalog=c,identity=value['identity'],inputs=value)
    r=previous.recent(p)
    return dict(kind='legacy',run=p,cache=r['cache'],catalog=r['catalog'],identity=r['setup']['identity'],legacy=r)


def raw_manifest(path):
    r=st.sealed_json(path)
    if r['format']!='reference_residual_raw_assets_v1':raise ValueError('unsupported raw asset manifest')
    fi.validate_catalog(r['catalog'])
    validate_frozen_origins(r['catalog'])
    return r


def preflight(a):
    missing=[];errors=[];assets=[]
    for name,p in [('data',a.data),('checkpoint',a.checkpoint),('host',a.official_root/'config.json'),('backbone',a.dinov3_root/'hubconf.py')]:
        if not p.exists():missing.append(dict(asset=name,path=str(p)))
    for path in a.recent_run:
        try:
            r=recent(path);base=r['cache'];c=r['catalog']
            assets.append(dict(run=str(path),status='found_metadata; compatibility_and_payload_checked_in_prepare',kind=r['kind'],
                original_bank_present=(base/('source_original/manifest.json' if r['kind']=='correspondence' else 'source_bank.npz')).exists()))
            if r['kind']=='correspondence':
                counts=dict(Source_training_complete=0,Source_calibration_complete=0,Target_complete=0,Source_sampled_only=0)
                for row in c['source']+sum(c['supports'].values(),[]):
                    p=base/'features'/key(row)/'manifest.json'
                    if not p.exists():continue
                    rr=st.read_json(p);full=all(f'full_L{l}' in rr['files'] for l in LAYERS)
                    if full:counts['Source_calibration_complete' if row.get('calibration') else 'Target_complete' if row.get('support_index') else 'Source_training_complete']+=1
                    elif row.get('source_index') is not None:counts['Source_sampled_only']+=1
                assets.append(dict(run=str(path),**counts,query_mode=r['inputs']['query_mode'],
                    original_query_cache='eligible only if full raw Q exists; z/distance/index are not features'))
        except (ValueError,OSError,KeyError) as e:errors.append(str(path)+': '+str(e))
    for path in a.asset_manifest:
        try:assets.append(dict(raw_manifest=str(path),status='found',bank=str(raw_manifest(path)['source_bank'])))
        except (OSError,ValueError,KeyError) as e:errors.append(str(e))
    dependencies={}
    for name in ('numpy','scipy','Pillow','torch','torchvision','timm','omegaconf','opencv-python','scikit-learn','psutil','matplotlib','adeval'):
        try:dependencies[name]=version(name)
        except PackageNotFoundError:dependencies[name]=None
    absent=[n for n,v in dependencies.items() if v is None]
    if dependencies['adeval'] not in (None,'1.1.0'):errors.append('ADEval==1.1.0 required')
    return dict(status='blocked' if missing or errors or absent else 'inventory_ready_not_runtime_verified',
        missing_inputs=missing,errors=errors,missing_dependencies=absent,dependency_versions=dependencies,assets=assets,
        config=CONFIG,torch_imported='torch' in __import__('sys').modules,
        new_computation=['two opposite-Source-fold original-sampler 100K training banks',
            '3640 Source + 2304 Target fixed RGB synthetic slots, each effective RGB forwarded ONCE',
            'original full-grid feature retrieval and halo tiles shared by all seeds/controls',
            'six Source-pretraining + 27 domain-training heads; independent logit calibration'],
        may_repeat=['455 full Source normal grids: old 2048 samples insufficient',
            '926 raw Q when old stream captures lack vectors'],
        scale_estimates=dict(assumption='13924 cells and 1280 FP32 channels, not measured; no resource admission',
            synthetic_slots=5944,normal_training_images=479,full_grid_bytes_per_image=13924*1280*4*4,
            all_training_full_grid_bytes=6423*13924*1280*4*4,
            maximum_tile_bytes_four_layers=(CONFIG['tile_center']+2*CONFIG['halo'])**2*1280*4*4,
            tile_count='measured per actual mask; positive-covering tiles plus fixed normal tiles',
            optional_full_926_query_bytes=926*13924*1280*4*4))


def freeze(a,c,extraction):
    fi.validate_catalog(c);validate_frozen_origins(c);st.sealed_json(a.run_root/'catalog.json',dict(catalog=c))
    st.sealed_json(a.run_root/'protocol.json',dict(CONFIG))
    shared=['DINOv3/relation_reliability/superadd_host.py','DINOv3/relation_reliability/calibration.py','DINOv3/relation_reliability/official_metrics.py',
        'external_baselines/superadd_external/memory_variants.py','DINOv3/ShiftADD/aebad_s_p0/memory.py',
        'DINOv3/MADEqual/aebad_s_normal_correspondence_adaptation/calibration.py','DINOv3/MADEqual/aebad_s_normal_correspondence_adaptation/model.py',
        'DINOv3/MADEqual/aebad_s_normal_correspondence_adaptation/evaluation.py','DINOv3/MADEqual/aebad_s_normal_correspondence_adaptation/training.py',
        'DINOv3/MADEqual/aebad_s_support_witness_segment_p0/evaluation.py','DINOv3/MADEqual/aebad_s_projection_layer_diagnostic/metrics.py',
        'DINOv3/MADEqual/aebad_s_projection_layer_diagnostic/numerics.py','DINOv3/MADEqual/aebad_s_ch1_scoring_inheritance/scoring.py',
        'DINOv3/MADEqual/aebad_s_segment_pair_crossdomain_p0/analysis.py','DINOv3/MADEqual/aebad_s_target_local_segment/geometry.py']
    code={p.relative_to(ROOT).as_posix():st.sha(p) for p in list(Path(__file__).parent.glob('*.py'))+[Path(__file__).with_name('frozen_origins.json')]+[ROOT/p for p in shared]}
    value=dict(config=CONFIG,catalog=st.digest(c),cache=str(a.cache_root),query_mode=a.query_mode,
        extraction=extraction,code=code)
    value['identity']=st.digest(value);return st.sealed_json(a.run_root/'inputs.json',value)
