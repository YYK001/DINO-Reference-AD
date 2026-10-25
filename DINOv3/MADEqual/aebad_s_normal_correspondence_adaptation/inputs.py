"""Only explicitly supplied new-workstation runs are eligible for asset reuse."""
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
import numpy as np
from . import storage as st
from .protocol import CONFIG, DOMAINS, LAYERS
from DINOv3.MADEqual.aebad_s_support_witness_segment_p0 import fresh_inputs as fi

ROOT=Path(__file__).resolve().parents[3]


def recent(path):
    """Inspect named run only; no old-server discovery, graph or Triton import."""
    p=Path(path);setup=st.read_json(p/'setup.json')
    cat=st.read_json(p/'fresh_catalog.json')
    conf=setup['configuration']
    expected=setup.get('catalog_sha256') or st.read_json(p/'fresh_catalog.seal.json')['sha256']
    if not expected or st.sha(p/'fresh_catalog.json')!=expected:raise ValueError('recent catalog identity differs')
    fi.validate_catalog(cat)
    accelerated=conf.get('engine')=='fp32_single_edge_pass_fused_refinement_v1'
    if setup.get('identity')!=st.digest(conf if accelerated else {k:v for k,v in setup.items() if k!='identity'}):
        raise ValueError('recent setup seal differs')
    cache=Path(setup.get('cache') or conf['cache_root'])
    origin=conf
    if accelerated:
        parent=Path(conf['parent'])/'setup.json'
        if conf['parent_records'].get(str(parent))!=st.sha(parent):raise ValueError('recent original extraction setup changed')
        origin=st.read_json(parent)['configuration']
    elif conf.get('revision') not in ('witness_fresh_rgb_seed42_v1','fixed_target_graph_v1'):
        # Original graph revisions are checked via their explicit protocol below.
        if 'fixed_target_graph' not in str(conf.get('protocol',{}).get('tool','')):
            raise ValueError('only migration-era fresh WITNESS/fixed-graph formats supported')
    # FP32 child configuration is inherited; its own cache is recorded in setup/cache.
    return dict(run=p,cache=cache,setup=setup,catalog=cat,configuration=origin)


def preflight(a):
    missing=[];errors=[];inventory=[]
    for name,p in [('data',a.data),('checkpoint',a.checkpoint),('official',a.official_root/'config.json'),
                   ('dinov3',a.dinov3_root/'hubconf.py')]:
        if not p.exists():missing.append(dict(asset=name,path=str(p)))
    for p in a.recent_run:
        try:
            r=recent(p)
            for name,loc in [('source_bank',r['cache']/'source_bank.npz'),
                ('original_patch_MAD',r['cache']/'source_patch_mad.npz'),
                ('original_image_MAD',p/'image_calibration_state.json')]:
                inventory.append(dict(run=str(p),asset=name,path=str(loc),present=loc.is_file(),
                    validation='existence only; payload SHA and extraction identity checked in prepare'))
            inventory.append(dict(run=str(p),asset='support_features',present=sum(
                (r['cache']/'supports'/d/f'{i:03d}.npz').is_file() for d in DOMAINS for i in range(1,9)),expected=24))
        except (OSError,ValueError,KeyError) as e:errors.append(str(p)+': '+str(e))
    versions={}
    for name in ('numpy','scipy','torch','torchvision','adeval','timm','omegaconf','Pillow','scikit-learn','psutil','matplotlib'):
        try:versions[name]=version(name)
        except PackageNotFoundError:versions[name]=None
    missing_dependencies=[name for name,v in versions.items() if v is None]
    if versions.get('adeval') not in (None,'1.1.0'):errors.append('official evaluation requires ADEval==1.1.0')
    return dict(status='blocked' if missing or errors or missing_dependencies else 'inventory_ready_not_runtime_verified',
        missing_inputs=missing,errors=errors,recent_assets=inventory,configuration=CONFIG,
        missing_dependencies=missing_dependencies,
        dependency_versions=versions,torch_imported=__import__('sys').modules.get('torch') is not None,
        repeated_work=['455 unaugmented Source grids -> sampled training rows plus full-grid teacher context',
            '66 complete Source calibration feature grids (distances cannot recover features)',
            '926 query forwards in stream mode unless compatible complete Q supplied by disk cache'],
        reused_when_compatible=['one original Source bank','24 complete support features'],
        original_MAD_policy='refit on this run Source calibration features/common original bank; historical values are not mandatory',
        never_run=['fixed graphs','WITNESS','SPARC','FastRef'],capacity_admission=False,
        note='Only migration-era WITNESS and fixed-graph runs supplied via --recent-run; fresh needs no historical assets.')


def catalog(a): return st.sealed_json(a.run_root/'catalog.json')['catalog']
def identity(a): return st.sealed_json(a.run_root/'inputs.json')['identity']


def image_key(row): return st.digest(row['relative_path'])[:24]
def feature_path(a,row):return a.cache_root/'features'/image_key(row)
def training_rows(c,domain=None):
    rows=[r for r in c['source'] if not r['calibration']]
    return rows+(c['supports'][domain] if domain is not None else [])


def validate_training(c,domain,rows):
    allowed={r['relative_path'] for r in training_rows(c,domain)}
    if any(r['relative_path'] not in allowed for r in rows):raise ValueError('calibration/test/other-domain image in training')
    if len({r['relative_path'] for r in rows})!=len(rows):raise ValueError('duplicate training image')


def freeze(a,c):
    fi.validate_catalog(c)
    st.sealed_json(a.run_root/'catalog.json',dict(catalog=c))
    files=list(Path(__file__).parent.glob('*.py'))
    shared=['DINOv3/ShiftADD/aebad_s_p0/memory.py','DINOv3/relation_reliability/superadd_host.py',
        'external_baselines/superadd_external/memory_variants.py',
        'external_baselines/superadd_external/run.py',
        'DINOv3/relation_reliability/calibration.py','DINOv3/relation_reliability/official_metrics.py',
        'DINOv3/MADEqual/aebad_s_target_local_segment/geometry.py',
        'DINOv3/MADEqual/aebad_s_segment_pair_crossdomain_p0/analysis.py',
        'DINOv3/MADEqual/aebad_s_support_witness_segment_p0/evaluation.py',
        'DINOv3/MADEqual/aebad_s_projection_layer_diagnostic/metrics.py',
        'DINOv3/MADEqual/aebad_s_ch1_scoring_inheritance/scoring.py',
        'DINOv3/MADEqual/aebad_s_projection_layer_diagnostic/numerics.py']
    value=dict(configuration=CONFIG,catalog=st.digest(c),query_mode=a.query_mode,cache_root=str(a.cache_root),
        checkpoint_sha256=st.sha(a.checkpoint),host_config=st.read_json(a.official_root/'config.json'),
        code={str(p.relative_to(ROOT)):st.sha(p) for p in files+[ROOT/p for p in shared]},
        external_code={name+'/'+p.relative_to(root).as_posix():st.sha(p)
            for name,root in [('dinov3',a.dinov3_root),('host',a.official_root)] for p in sorted(root.rglob('*.py'))},
        software={n:version(n) for n in ('numpy','scipy','torch','torchvision','Pillow','scikit-learn','adeval','timm','omegaconf')})
    value['identity']=st.digest(value)
    st.sealed_json(a.run_root/'inputs.json',value)
    return value
