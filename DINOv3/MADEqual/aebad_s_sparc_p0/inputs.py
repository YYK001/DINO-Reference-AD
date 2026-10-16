"""Read-only identities and resource checks, available without Torch."""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image

from DINOv3.MADEqual.aebad_s_target_fewshot import replay12 as replay
from DINOv3.MADEqual.aebad_s_union_support_loo.inputs import digest, read_capture, verify_files
from DINOv3.MADEqual.aebad_s_union_support_loo.runner import formal_baselines
from DINOv3.MADEqual.aebad_s_target_local_segment.inputs import support_manifest_matches, gt_relative
from DINOv3.MADEqual.aebad_s_full_p0.dataset import find_aebad_s_root, _images
from DINOv3.relation_reliability.datasets import Record, split_source_normal
from DINOv3.relation_reliability.protocol import validate_superadd_config
from .projection import LAYERS, DIMENSION, RANK_RULE

TOOL = 'aebad_s_sparc_same_host_v1'
CAPTURE_NAME = 'aebad_s_union_support_loo_B_20261007_230639'


def paths_for(args):
    project=Path(args.project_root).resolve(); base=project/'DINOv3/outputs/mad_equal'
    results=Path(args.results_dir or base/'aebad_s_target_fewshot_seed42_v1').resolve()
    official=Path(args.official_root or project/'external_baselines/SuperADD').resolve()
    return dict(project=project,results=results,official=official,
        capture=Path(args.capture_dir or base/CAPTURE_NAME).resolve(),
        support=Path(args.support_cache_dir or base/'aebad_s_union_background_replay12_seed42_v1/support_features/target_matched/background').resolve(),
        data=Path(args.data).resolve(),dinov3=Path(args.dinov3_root).resolve(),
        checkpoint=Path(args.checkpoint or official/'weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth').resolve(),
        output=Path(args.output_dir or base/'aebad_s_sparc_background_seed42_v1').resolve())


def guard_output(paths, *, allow_prepared=False):
    out=paths['output']
    for key,root in paths.items():
        if key in ('output','project'): continue
        if out==root or out.is_relative_to(root) or root.is_relative_to(out):
            raise ValueError(f'output overlaps protected {key}: {root}')
    if not out.exists(): return
    if not out.is_dir(): raise ValueError('output must be a directory')
    if allow_prepared:
        if json.loads((out/'inputs.json').read_text(encoding='utf-8')).get('tool')!=TOOL:
            raise ValueError('existing output belongs to another tool')
    else:
        if any(p.name!='preflight.json' for p in out.iterdir()):
            raise ValueError('prepare requires a new directory; no overwrite/resume of partial output')
        if (out/'preflight.json').exists() and json.loads((out/'preflight.json').read_text(encoding='utf-8')).get('tool')!=TOOL:
            raise ValueError('foreign preflight directory')


def estimated_grid(height, width):
    """Header-only resource estimate of the pinned host size rule, not preprocessing."""
    h,w=int(height*.625),int(width*.625)
    if min(h,w)<=0: raise ValueError('empty resized dimensions')
    if min(h,w)<640:
        s=640/min(h,w); h,w=max(640,math.ceil(h*s/16)*16),max(640,math.ceil(w*s/16)*16)
    return h//16,w//16


def source_records(root):
    records=[Record(path=str(p),label=0,mask_path=None,category='AeBAD-S',dataset='aebad_s',
                    domain='source',shift='train_good',role='train',mask_capability='not_evaluation')
             for p in _images(root/'train/good')]
    memory,calibration=split_source_normal(records,threshold_fraction=8)
    if (len(records),len(memory),len(calibration))!=(521,455,66):
        raise ValueError('Source ordered split must be 521/455/66; builder receives all 521')
    return records


def resource_estimate(candidate_rows, grid, missing_supports=0, candidate_storage='mixed'):
    if candidate_storage not in ('mixed','disk'): raise ValueError('unsupported candidate storage')
    n=int(np.prod(grid))
    candidate=int(candidate_rows)*DIMENSION*4*4
    projection=n*DIMENSION*8*8*4+n*4
    bank=100000*DIMENSION*4*4
    # New per-image distances/indices/positions + raw calibration, no old bank copies.
    raw=(297*(4*4+4*8+8)+66*4*4)*n
    support=missing_supports*n*DIMENSION*4*4
    steady=projection+bank+raw+support
    resident=candidate//2 if candidate_storage=='mixed' else 0
    scratch=candidate-resident
    ram_layers=list(LAYERS[:2]) if resident else []
    return dict(candidate_rows_per_layer=candidate_rows,candidate_float32_bytes=candidate,
                candidate_storage=candidate_storage,ram_layers=ram_layers,
                disk_layers=[l for l in LAYERS if l not in ram_layers],
                candidate_resident_bytes=resident,candidate_scratch_bytes=scratch,
                projection_float64_bytes=projection,new_memory_bytes=bank,new_raw_estimate_bytes=raw,
                support_supplement_estimate_bytes=support,steady_estimate_bytes=steady,
                prepare_disk_required_bytes=scratch+steady+(4<<30),disk_headroom_bytes=4<<30,
                minimum_ram_headroom_bytes=resident+projection+(16<<30) if resident else 5<<30,
                ram_working_headroom_bytes=16<<30 if resident else 5<<30,
                interpretation='estimates, not observed peaks; unchanged full candidates, float32 and sampler')


def free_space(path):
    p=path
    while not p.exists(): p=p.parent
    return int(shutil.disk_usage(p).free)


def available_ram():
    candidates=[]
    try:
        import psutil
        candidates.append(int(psutil.virtual_memory().available))
    except ImportError: pass
    try:
        limit=Path('/sys/fs/cgroup/memory.max').read_text().strip()
        used=int(Path('/sys/fs/cgroup/memory.current').read_text().strip())
        if limit!='max': candidates.append(max(0,int(limit)-used))
    except (OSError,ValueError): pass
    return min(candidates) if candidates else None


def preflight(args):
    paths=paths_for(args); guard_output(paths,allow_prepared=args.mode in ('infer','evaluate'))
    check=dict(tool=TOOL,mode=args.mode,domain='background',development_domain=True,seed=42,K=8,
        paths={k:str(v) for k,v in paths.items()},errors=[],required_files=[],support_features=[],
        model_loaded=False,feature_extraction=False,torch_imported=False,
        source_rows=[],samples=[],supports=[],missing_assets=[],resource_estimate=None)
    def require(kind,path,expected=None):
        if not path.is_file():
            check['missing_assets'].append(dict(kind=kind,path=str(path)))
            check['errors'].append(f'missing {kind}: {path}'); return
        h=replay.sha256(path)
        if expected and h!=expected: raise ValueError(f'input hash changed: {kind}')
        check['required_files'].append(dict(kind=kind,path=str(path),sha256=h))
    cap=None
    try:
        cap,state=read_capture(paths['capture'])
        if (cap['scope']!='background' or len(cap['samples'])!=297
                or Counter(s['label'] for s in cap['samples'])!={0:85,1:212}
                or [(s['sample_id'],s['relative_path']) for s in cap['selected_cases']]!=list(replay.CASES)
                or cap['frozen']['protocol_id']!=replay.PROTOCOL_ID):
            raise ValueError('verified full background B scope/protocol/twelve cases required')
        expected_protocol=dict(brightness_factor=1,resize_factor=.625,patch_size=640,patch_overlap=128,
            model_patch_size=16,evaluation_downscale=4,dino_training=False,trainable_parameters=0)
        if (cap['frozen']['query_protocol']!=expected_protocol
                or [Path(s['relative_path']).name for s in cap['supports']]!=list(replay.TARGET_NAMES)):
            raise ValueError('frozen query protocol or original eight support names differ')
        precision=cap['frozen']['precision']
        if precision['float32_matmul_precision']!='highest' or precision['cuda_matmul_allow_tf32'] is not False:
            raise ValueError('original stable Exact precision required')
        check['expected_precision']=precision
        check.update(samples=cap['samples'],supports=cap['supports'],capture_inputs=cap,capture_status=state)
        for key in ('inputs.json','status.json','capture_verification.csv'): require('capture_'+key,paths['capture']/key)
        for r in state['records']: require('baseline_raw_'+r['sample_id'],paths['capture']/r['file'],r['sha256'])
    except (OSError,ValueError,KeyError,RuntimeError) as e:
        check['errors'].append(f'capture rejected: {e}')
        check['missing_assets'].append(dict(kind='complete_verified_B_capture',path=str(paths['capture'])))
    for key in ('sample_scores.csv','support_manifest.csv','complete.json','memory_manifest.json','source_mad_state.npz'):
        try: require(key,paths['results']/key,cap['frozen']['formal_sha256'][key] if cap else None)
        except ValueError as e: check['errors'].append(str(e))
    try:
        if cap:
            formal_baselines(paths['results'],check['samples'])
            support_manifest_matches(check['supports'],paths['results']/'support_manifest.csv')
            if {s['relative_path'] for s in check['samples']} & {s['relative_path'] for s in check['supports']}:
                raise ValueError('Target supports leaked into heldout')
            manifest=json.loads((paths['results']/'memory_manifest.json').read_text(encoding='utf-8'))
            if (manifest['builder_input_count'],manifest['memory_fit_count'],manifest['calibration_count'],manifest['threshold_fraction'])!=(521,455,66,8):
                raise ValueError('original Source memory manifest split differs')
            grids={tuple(s['grid']) for s in cap['supports']}
            if len(grids)!=1: raise ValueError('support native grids differ')
            check['grid']=list(next(iter(grids)))
            for support,original in zip(cap['supports'],cap['frozen']['supports'],strict=True):
                if any(support[k]!=original[k] for k in ('support_index','relative_path','feature_sha256','image_sha256')):
                    raise ValueError('support identity differs from frozen original provenance')
            for r in state['records']:
                with np.load(paths['capture']/r['file'],allow_pickle=False) as f:
                    meta=json.loads(str(f['metadata'].item()))
                if meta['grid']!=check['grid']: raise ValueError('query/support full stitched grids differ')
                if meta['source_rows']!=[100000]*4: raise ValueError('baseline Source bank budget differs')
                geometry=meta['geometry'];reference=cap['supports'][0]['geometry']
                if any(geometry[k]!=reference[k] for k in ('processed_height','processed_width','axes','model_patch_size')):
                    raise ValueError('query/support stitching or processed shape differs')
            for s in cap['supports']:
                if any(s['geometry'][k]!=reference[k] for k in ('processed_height','processed_width','axes','model_patch_size')):
                    raise ValueError('support stitching differs')
            replay.check_mad(paths['results']/'source_mad_state.npz')
        for row in check['supports']:
            path=paths['support']/f"{row['support_index']:02d}.npz"
            info=dict(support_index=row['support_index'],relative_path=row['relative_path'],path=str(path),
                      exists=path.is_file(),action='reuse_original' if path.is_file() else 'extract_only_in_prepare')
            if path.is_file():
                require('support_'+str(row['support_index']),path,row['feature_sha256'])
                replay.check_support_header(path,row['relative_path'])
                with np.load(path,allow_pickle=False) as f:
                    if f['grid'].tolist()!=row['grid']: raise ValueError('support grid changed')
            check['support_features'].append(info)
    except (OSError,ValueError,KeyError,RuntimeError) as e: check['errors'].append(f'formal/support rejected: {e}')
    if cap is None:
        for j,name in enumerate(replay.TARGET_NAMES,1):
            path=paths['support']/f'{j:02d}.npz'
            check['support_features'].append(dict(support_index=j,relative_path='test/good/background/'+name,
                path=str(path),exists=path.is_file(),identity_verified=False,
                action='identity_unverified_without_B_capture' if path.is_file() else 'extract_only_in_prepare_after_B_capture_validation'))
            if not path.is_file(): check['missing_assets'].append(dict(kind=f'support_feature_{j}',path=str(path)))
    try:
        root=find_aebad_s_root(paths['data']);paths['data']=root;check['paths']['data']=str(root)
        guard_output(paths,allow_prepared=args.mode in ('infer','evaluate'))
        records=source_records(root)
        for i,r in enumerate(records):
            p=Path(r.path);require(f'source_image_{i:03d}',p)
            with Image.open(p) as image: height,width=image.height,image.width;grid=estimated_grid(height,width)
            if 'grid' in check and list(grid)!=check['grid']: raise ValueError(f'Source grid incompatible: {p}')
            check['source_rows'].append(dict(source_index=i,relative_path=p.relative_to(root).as_posix(),
                role='calibration' if i%8==0 else 'memory_fit',grid=list(grid),image_height=height,image_width=width))
        if cap:
            for s in check['samples']:
                rel=replay.safe_relative(s['relative_path']);require('query_'+s['sample_id'],root/rel,cap['image_sha256'][rel])
                if s['label']:
                    gt=gt_relative(rel);require('GT_'+s['sample_id'],root/gt,cap['gt_sha256'][gt])  # bytes only
            for s in check['supports']:
                rel=replay.safe_relative(s['relative_path']);require('support_RGB_'+str(s['support_index']),root/rel,s['image_sha256'])
    except (OSError,ValueError,KeyError,RuntimeError) as e:
        check['errors'].append(f'data rejected: {e}')
        check['missing_assets'].append(dict(kind='compatible_Source_and_background_data',path=str(paths['data'])))
    required=[('official_config',paths['official']/'config.json'),
              ('official_model',paths['official']/'tracks/industrial/src/industrial/model.py'),
              ('dinov3_hub',paths['dinov3']/'hubconf.py'),('checkpoint',paths['checkpoint'])]
    required += [('code_'+rel,paths['project']/rel) for rel in (
        'external_baselines/superadd_external/memory_variants.py',
        'external_baselines/SuperADD/tracks/industrial/src/industrial/nearest_neighbor.py',
        'DINOv3/MADEqual/aebad_s_target_fewshot/runner.py',
        'DINOv3/relation_reliability/superadd_host.py','DINOv3/relation_oracle_cache/extract.py',
        'DINOv3/MADEqual/aebad_s_full_p0/runner.py','DINOv3/relation_reliability/official_metrics.py')]
    required += [('code_sparc_'+p.name,p) for p in Path(__file__).parent.glob('*.py')]
    for kind,path in required:
        try:
            expected=replay.CHECKPOINT_SHA256 if kind=='checkpoint' else None
            if cap:
                if kind in ('official_config','official_model','dinov3_hub'):
                    expected=cap['frozen']['input_code_sha256'][kind]
                for key,rel in (('code_runner','DINOv3/MADEqual/aebad_s_target_fewshot/runner.py'),
                                ('code_host','DINOv3/relation_reliability/superadd_host.py'),
                                ('code_mad','DINOv3/relation_oracle_cache/extract.py')):
                    if kind=='code_'+rel: expected=cap['frozen']['input_code_sha256'][key]
            require(kind,path,expected)
        except (ValueError,KeyError) as e: check['errors'].append(str(e))
    try:
        config=json.loads((paths['official']/'config.json').read_text(encoding='utf-8'))
        validate_superadd_config(config)
        if config['brightness_augmentation']!=[.8,1.2] or config['evaluation_downscale']!=4 or config['subsampling_iterations']!=100:
            raise ValueError('frozen host brightness/score/sampler configuration changed')
        check['host_config']=config
    except (OSError,ValueError,KeyError,RuntimeError) as e: check['errors'].append(f'host config rejected: {e}')
    if 'grid' in check and len(check['source_rows'])==521:
        rows=sum(int(np.prod(s['grid'])) for s in check['source_rows'] if s['role']=='memory_fit')
        resources=resource_estimate(rows,check['grid'],sum(not r['exists'] for r in check['support_features']),
                                    getattr(args,'candidate_storage','mixed'))
        resources['disk_free_bytes']=free_space(paths['output']);check['resource_estimate']=resources
        resources['ram_available_bytes']=available_ram()
        if args.mode in ('preflight','prepare') and resources['disk_free_bytes']<resources['prepare_disk_required_bytes']:
            check['errors'].append('insufficient disk for original full-size candidate spool; no budget/resolution/precision reduction')
        if args.mode in ('preflight','prepare') and resources['candidate_resident_bytes'] and resources['ram_available_bytes'] is None:
            check['errors'].append('mixed candidate storage requires a known available RAM budget')
        if args.mode in ('preflight','prepare') and resources['ram_available_bytes'] is not None and resources['ram_available_bytes']<resources['minimum_ram_headroom_bytes']:
            check['errors'].append('insufficient available RAM for bounded support/candidate processing')
    check['status']='blocked' if check['errors'] else 'filesystem_preflight_passed'
    return check


def freeze_check(check):
    frozen=dict(protocol_id=replay.PROTOCOL_ID,development_domain=True,seed=42,K=8,
        grid=check['grid'],rank_rule=RANK_RULE,capture_identity=check['capture_inputs']['identity'],
        source_rows=check['source_rows'],supports=check['supports'],host_config=check['host_config'],
        precision=check['expected_precision'],
        candidate_storage={k:check['resource_estimate'][k] for k in ('candidate_storage','ram_layers','disk_layers')},
        files=[{k:r[k] for k in ('kind','sha256')} for r in check['required_files']])
    return dict(tool=TOOL,identity=digest(frozen),frozen=frozen,paths=check['paths'],
                samples=check['samples'],required_files=check['required_files'],
                support_features=check['support_features'],resource_estimate=check['resource_estimate'])


def file_records(root, files):
    return [dict(file=p.relative_to(root).as_posix(),sha256=replay.sha256(p),bytes=p.stat().st_size) for p in files]


def verify_artifacts(root, records):
    for r in records:
        rel=replay.safe_relative(r['file']);path=root/rel
        if not path.is_file() or replay.sha256(path)!=r['sha256']:
            raise ValueError(f'frozen artifact missing/modified: {rel}; no re-approval')


def read_prepared(root):
    inputs=json.loads((root/'inputs.json').read_text(encoding='utf-8'))
    state=json.loads((root/'prepare_status.json').read_text(encoding='utf-8'))
    if (inputs.get('tool')!=TOOL or inputs['identity']!=digest(inputs['frozen'])
            or state.get('status')!='prepare_complete' or state.get('tool')!=TOOL
            or state['identity']!=inputs['identity'] or state['inputs_sha256']!=replay.sha256(root/'inputs.json')):
        raise ValueError('prepare incomplete or validation record modified')
    verify_files(inputs['required_files']);verify_artifacts(root,state['artifacts'])
    return inputs,state
