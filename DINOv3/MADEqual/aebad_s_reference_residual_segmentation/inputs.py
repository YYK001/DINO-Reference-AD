"""Normal-only origins, stable folds, recent raw asset schemas and read-only preflight."""
from pathlib import Path
from importlib.metadata import version,PackageNotFoundError
from . import storage as st
from .protocol import CONFIG,DOMAINS,LAYERS,VERSION
from . import protocol
from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation import inputs as previous
from DINOv3.MADEqual.aebad_s_support_witness_segment_p0 import fresh_inputs as fi

ROOT=Path(__file__).resolve().parents[3]
def catalog(a):return st.sealed_json(a.run_root/'catalog.json')['catalog']
def identity(a):return st.sealed_json(a.run_root/'inputs.json')['identity']
def legacy(a):
    p=a.run_root/'inputs.json'
    if not p.exists():return False
    value=st.sealed_json(p)
    return value.get('config')==CONFIG and 'resolved_config_seal' not in value


def config(a):
    requested=getattr(a,'profile',protocol.DEFAULT_PROFILE);p=a.run_root/'resolved_config.json'
    if p.exists():
        value=st.sealed_json(p)['config']
        if value['profile']!=requested:raise ValueError('frozen profile differs; new output directory required')
        if value!=protocol.resolve(requested):raise ValueError('fixed profile configuration differs from implementation')
        return value
    old=a.run_root/'inputs.json'
    if old.exists():
        value=st.sealed_json(old)
        if 'resolved_config_seal' in value:raise ValueError('missing frozen resolved_config.json')
        if value.get('config')==CONFIG:
            if requested!='full_v1':raise ValueError('legacy full_v1 requires explicit --profile full_v1; never migrated to compact')
            known=st.read_json(Path(__file__).with_name('legacy_full_v1_compatibility.json'))['code']
            prefix='DINOv3/MADEqual/aebad_s_reference_residual_segmentation/'
            saved={Path(k).name:v for k,v in value.get('code',{}).items() if k.startswith(prefix) and k.endswith('.py')}
            if saved!=known:raise ValueError('legacy full_v1 code provenance unknown; original implementation required')
            result=protocol.resolve('full_v1');result['training_nn_layout']='complete_grid_v1';return result
    return protocol.resolve(requested)


def plan(a):
    cfg=config(a);expected=protocol.job_plan(cfg);p=a.run_root/'job_plan.json'
    if p.exists():
        saved=st.sealed_json(p)['plan']
        if saved!=expected:raise ValueError('frozen job plan differs')
        return saved
    if (a.run_root/'resolved_config.json').exists() and (a.run_root/'inputs.json').exists():raise ValueError('missing frozen job_plan.json')
    return expected
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
        if value.get('config',{}).get('version')==VERSION:
            if value['catalog']!=st.digest(c):raise ValueError('recent residual catalog differs')
            fi.validate_catalog(c);validate_frozen_origins(c)
            return dict(kind='reference_residual',run=p,cache=Path(value['cache']),catalog=c,identity=value['identity'],inputs=value)
        if value.get('configuration',{}).get('version')!='normal_correspondence_residual_v1':raise ValueError('not an original raw asset schema')
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


def work_inventory(a):
    cfg=config(a);jobs=plan(a);result=dict(planned_slots=jobs['planned_synthetic_slots'],requested_updates=jobs['planned_updates'],completed_training_jobs=[])
    for j in jobs['training_jobs']:
        p=a.run_root/'models'/j['phase']/j['family']/f"seed{j['seed']}"/'complete.json'
        if p.exists():
            r=st.sealed_json(p)
            if r['inputs']['identity']==identity(a) and r['updates']==j['updates']:result['completed_training_jobs'].append((j['phase'],j['family'],j['seed']))
    if not (a.run_root/'catalog.json').exists():return result
    c=catalog(a);normal=c['source']+sum(c['supports'].values(),[])
    result['normal_feature_forwards_remaining']=sum(not (normal_path(a,r)/'manifest.json').exists() for r in normal)
    synth=a.run_root/'synthetic_manifest.json'
    if synth.exists():
        rows=[r for r in st.sealed_json(synth)['items'] if r['status']=='valid'];result['effective_synthetic_samples']=len(rows)
        result['synthetic_forwards_remaining']=sum(not (a.cache_root/'training_queries'/r['item_id']/'complete.json').exists() for r in rows)
    remaining=0
    for d in DOMAINS:
        remaining+=sum(not (a.cache_root/'calibration_NN'/d/f'{i:03d}'/'manifest.json').exists() for i in range(1,sum(r['calibration'] for r in c['source'])+1))
        remaining+=sum(not (a.cache_root/'predictions'/d/r['sample_id']/'manifest.json').exists() for r in c['samples'] if r['domain']==d)
    result['full_grid_calibration_prediction_NN_remaining']=remaining
    result['inventory_scope']='committed current-run metadata, not unverified parent candidates'
    return result


def preflight(a):
    missing=[];errors=[];assets=[]
    try:cfg=config(a);jobs=plan(a)
    except (ValueError,OSError,KeyError) as e:return dict(status='blocked',errors=[str(e)],missing_inputs=[],torch_imported='torch' in __import__('sys').modules)
    for name,p in [('data',a.data),('checkpoint',a.checkpoint),('host',a.official_root/'config.json'),('backbone',a.dinov3_root/'hubconf.py')]:
        if not p.exists():missing.append(dict(asset=name,path=str(p)))
    for path in a.recent_run:
        try:
            r=recent(path);base=r['cache'];c=r['catalog']
            assets.append(dict(run=str(path),status='found_metadata; compatibility_and_payload_checked_in_prepare',kind=r['kind'],
                original_bank_present=(base/('source_original/manifest.json' if r['kind'] in ('correspondence','reference_residual') else 'source_bank.npz')).exists()))
            if r['kind']=='reference_residual':
                from .synthesis import manifest
                selected=manifest(c,cfg)
                assets.append(dict(run=str(path),normal_features='reuse compatible original complete grids by content/extraction identity',
                    complete_normal_candidates=sum((base/'normal_features'/key(row)/'manifest.json').exists() for row in c['source']+sum(c['supports'].values(),[])),
                    fold_banks={f:(base/'source_folds'/('fold_'+f)/'manifest.json').exists() for f in ('A','B')},
                    selected_synthesis='inspect selected original IDs only, including interrupted per-item pool',
                    selected_RGB_candidates=sum((base/'synthetic_rgb'/v['item_id']/'manifest.json').exists() for v in selected),
                    selected_training_candidates=sum((base/'training_queries'/v['item_id']/'complete.json').exists() for v in selected),
                    tiles_and_NN='conditional on RGB/label/geometry/reference dependencies; full NN may be subsetted',
                    models='not reused across different schedules/pools',status='payloads_and_compatibility_not_verified_in_preflight'))
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
        config=cfg,job_plan=jobs,torch_imported='torch' in __import__('sys').modules,
        new_computation=['two opposite-Source-fold original-sampler 100K training banks',
            f"{jobs['planned_synthetic_slots']} planned selected RGB slots; compatible cached slots/grids/NN imported, no hidden fill",
            'training tile+halo union Exact; full normal grids when actually covered; full-grid calibration/test unchanged',
            f"{jobs['source_jobs']} Source + {jobs['target_jobs']} target heads; {jobs['planned_updates']} updates; requested logit calibration only"],
        may_repeat=['455 full Source normal grids: old 2048 samples insufficient',
            '926 raw Q when old stream captures lack vectors'],
        reuse_decisions=dict(reuse='only compatibility-verified originals/selected RGB/labels/tiles/NN; see assets candidates',
            new='requested missing synthetic items, NN positions and training jobs only',
            repeat='missing full original grids and stream-mode test Q; never inferred from distances',
            unconfirmed='candidate existence does not prove payload/extraction/rule compatibility; verified at prepare/first consumption'),
        scale_estimates=dict(assumption='13924 cells and 1280 FP32 channels, not measured; no resource admission',
            synthetic_slots=jobs['planned_synthetic_slots'],normal_training_images=cfg['Source_training']+len(DOMAINS)*8,full_grid_bytes_per_image=13924*1280*4*4,
            all_training_full_grid_bytes=(jobs['planned_synthetic_slots']+cfg['Source_training']+len(DOMAINS)*8)*13924*1280*4*4,
            maximum_tile_bytes_four_layers=(CONFIG['tile_center']+2*CONFIG['halo'])**2*1280*4*4,
            tile_count='measured per actual mask; positive-covering tiles plus fixed normal tiles',
            optional_full_926_query_bytes=926*13924*1280*4*4))


def freeze(a,c,extraction):
    cfg=config(a);jobs=protocol.job_plan(cfg)
    if legacy(a):
        old=st.sealed_json(a.run_root/'inputs.json')
        if old['catalog']!=st.digest(c) or old['extraction']!=extraction:raise ValueError('legacy full_v1 inputs changed; no migration')
        return old
    fi.validate_catalog(c);validate_frozen_origins(c);st.sealed_json(a.run_root/'catalog.json',dict(catalog=c))
    st.sealed_json(a.run_root/'protocol.json',dict(cfg))
    resolved=st.sealed_json(a.run_root/'resolved_config.json',dict(config=cfg))
    planned=st.sealed_json(a.run_root/'job_plan.json',dict(plan=jobs))
    shared=['DINOv3/relation_reliability/superadd_host.py','DINOv3/relation_reliability/calibration.py','DINOv3/relation_reliability/official_metrics.py',
        'external_baselines/superadd_external/memory_variants.py','DINOv3/ShiftADD/aebad_s_p0/memory.py',
        'DINOv3/MADEqual/aebad_s_normal_correspondence_adaptation/calibration.py','DINOv3/MADEqual/aebad_s_normal_correspondence_adaptation/model.py',
        'DINOv3/MADEqual/aebad_s_normal_correspondence_adaptation/evaluation.py','DINOv3/MADEqual/aebad_s_normal_correspondence_adaptation/training.py',
        'DINOv3/MADEqual/aebad_s_normal_correspondence_adaptation/storage.py','DINOv3/MADEqual/aebad_s_support_witness_segment_p0/storage.py',
        'DINOv3/MADEqual/aebad_s_support_witness_segment_p0/evaluation.py','DINOv3/MADEqual/aebad_s_projection_layer_diagnostic/metrics.py',
        'DINOv3/MADEqual/aebad_s_projection_layer_diagnostic/numerics.py','DINOv3/MADEqual/aebad_s_ch1_scoring_inheritance/scoring.py',
        'DINOv3/MADEqual/aebad_s_segment_pair_crossdomain_p0/analysis.py','DINOv3/MADEqual/aebad_s_target_local_segment/geometry.py']
    code={p.relative_to(ROOT).as_posix():st.sha(p) for p in list(Path(__file__).parent.glob('*.py'))+[Path(__file__).with_name('frozen_origins.json'),Path(__file__).with_name('legacy_full_v1_compatibility.json')]+[ROOT/p for p in shared]}
    from .reuse import rules
    value=dict(config=cfg,catalog=st.digest(c),cache=str(a.cache_root),query_mode=a.query_mode,
        extraction=extraction,code=code,resolved_config_seal=resolved['seal'],job_plan_seal=planned['seal'],artifact_rules=rules(cfg))
    value['identity']=st.digest(value);return st.sealed_json(a.run_root/'inputs.json',value)
