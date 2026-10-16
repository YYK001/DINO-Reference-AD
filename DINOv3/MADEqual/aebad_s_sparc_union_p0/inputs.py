"""Read-only frozen parent checks. No Torch, extraction, fitting or fallback."""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
import numpy as np

from DINOv3.MADEqual.aebad_s_sparc_p0 import inputs as parent_inputs
from DINOv3.MADEqual.aebad_s_sparc_p0.projection import LAYERS,DIMENSION,load_projection
from DINOv3.MADEqual.aebad_s_union_support_loo.inputs import digest,verify_files

replay=parent_inputs.replay
TOOL='aebad_s_sparc_union_k8_v1'
NEW='SPARC_UNION_K8'
BRANCHES=('SOURCE','UNION_K8','SPARC_SOURCE_K8',NEW)
BASE_NAME='aebad_s_sparc_background_mixed_20261009_095419'
PARENT_IDENTITY='a5d86dcae44db59652170e54304d0d11e91e5fd4013ecf84129e63c936734a88'
# Approved bytes from the user's downloaded parent review package, not hashes
# freshly re-approved from a possibly modified server result.
PARENT_RECORDS={
 'inputs.json':'f61bb4769d15a3d1fc6d6e6b43899a1529117e2a2abd84bc5e9d3d332cee20d0',
 'prepare_status.json':'28537c987c76394f8ad2f7a21ed3eab779567d1b3fb91807be6dd510f7d27bba',
 'infer_status.json':'62719431aea0a2afe8ba28769dbd70ace0e38747bf85d54c22707b4f79a3ff66',
 'evaluation/status.json':'e2fd20a89170d2a7c10acc5e902d71c925560404d55430a080170cc9720c0c58',
 'evaluation/metrics.csv':'8dec4c0c7c28f4cbc548a0568f27b748b2dfa1a581bec245b6babfe844908f94',
 'evaluation/per_image_scores.csv':'66938d07426077b0f16617c22ff1260e6e226075a4b430946c01e584b0e6dfd8',
 'evaluation/numerical_verification.csv':'c67a0f1054f707713c35ddaf5df87d84a1ddbfcaf257b68ebb3612547463893c',
 'projection_state.json':'6852718ee0443f6df31ab8228a47c903fc703816a2ff92866f21aec29861614e',
 'support_provenance.json':'6e02133f8527c809e0ed3651bbba12c8dcefa28831339ebc97e5204c7794eba1',
}


def paths_for(args):
    project=Path(args.project_root).resolve();base=project/'DINOv3/outputs/mad_equal'
    return dict(project=project,parent=Path(args.sparc_base_dir or base/BASE_NAME).resolve(),
                output=Path(args.output_dir or base/'aebad_s_sparc_union_k8_seed42_v1').resolve())


def guard_output(paths,*,prepared=False):
    out=Path(paths['output'])
    for key,value in paths.items():
        if key in ('project','output'):continue
        root=Path(value)
        if out==root or out.is_relative_to(root) or root.is_relative_to(out):
            raise ValueError(f'output overlaps protected {key}: {root}')
    if out.exists() and (not out.is_dir() or (not prepared and any(p.name!='preflight.json' for p in out.iterdir()))):
        raise ValueError('use a new output; no resume, overwrite or automatic parent prepare')


def resources(grid,workers):
    n=int(np.prod(grid));target=8*n*DIMENSION*4*4
    raw=297*n*(3*4*4+2*4*8+8)
    return dict(corrected_target_bytes=target,new_raw_estimate_bytes=raw,
                prepare_disk_required_bytes=target+raw+(2<<30),
                minimum_available_ram_bytes=8<<30,source_candidate_scratch_bytes=0,
                cpu_workers=workers,interpretation='estimates; no Source extraction/bank/MAD/SVD costs')


def validate_scope(parent):
    frozen=parent['frozen'];samples=parent['samples'];supports=frozen['supports']
    if (frozen.get('protocol_id')!=replay.PROTOCOL_ID or frozen.get('seed')!=42 or frozen.get('K')!=8
            or frozen.get('grid')!=[118,118] or tuple(frozen['host_config']['layers'])!=LAYERS
            or frozen['host_config']['patch_size']!=640 or frozen['host_config']['patch_overlap']!=128
            or frozen['host_config']['evaluation_downscale']!=4
            or frozen['precision']['float32_matmul_precision']!='highest'
            or frozen['precision']['cuda_matmul_allow_tf32'] is not False):
        raise ValueError('parent extraction/precision is not the frozen background protocol')
    if (len(samples)!=297 or len({s['sample_id'] for s in samples})!=297
            or len({s['relative_path'] for s in samples})!=297
            or Counter(s['label'] for s in samples)!=Counter({0:85,1:212})
            or len(supports)!=8 or [s['support_index'] for s in supports]!=list(range(1,9))
            or {s['relative_path'] for s in supports}&{s['relative_path'] for s in samples}):
        raise ValueError('support order/count or heldout exclusion differs')
    if not set(replay.CASES).issubset({(s['sample_id'],s['relative_path']) for s in samples}):
        raise ValueError('fixed twelve cases missing')


def history_tables(root,samples):
    scores=replay.read_csv(root/'evaluation/per_image_scores.csv')
    by={b:{} for b in BRANCHES[:3]};expected={s['sample_id']:s for s in samples}
    for r in scores:
        b=r['branch'];sid=r['sample_id']
        if b not in by or sid not in expected or sid in by[b]:raise ValueError('historical score membership/duplicate')
        if r['relative_path']!=expected[sid]['relative_path'] or int(r['label'])!=expected[sid]['label']:
            raise ValueError('historical score identity/label changed')
        if not np.isfinite(float(r['image_score'])):raise ValueError('nonfinite historical score')
        by[b][sid]=r
    if any(set(g)!=set(expected) for g in by.values()):raise ValueError('all three complete historical score sets required')
    metrics=replay.read_csv(root/'evaluation/metrics.csv')
    if len(metrics)!=3 or {r['branch'] for r in metrics}!=set(BRANCHES[:3]):raise ValueError('three historical metric rows required')
    for r in metrics:
        if int(r['image_count'])!=297 or any(not np.isfinite(float(r[k])) or not 0<=float(r[k])<=1
                for k in ('image_AUROC','image_AP','pixel_AUROC','pixel_AP','AUPRO')):
            raise ValueError('invalid historical metrics')
    return by,metrics


def preflight(args):
    paths=paths_for(args);guard_output(paths,prepared=args.mode in ('infer','evaluate'))
    root=paths['parent'];check=dict(tool=TOOL,status='blocked',errors=[],missing_assets=[],
        paths={k:str(v) for k,v in paths.items()},required_files=[],parent=None,
        resources=resources((118,118),args.cpu_workers),resource_shape_source='fixed task grid; parent not yet validated',
        model_loaded=False,feature_extraction=False,source_builder_called=False,MAD_fit_called=False)
    def require(kind,path,expected=None):
        path=Path(path)
        if not path.is_file():
            check['missing_assets'].append(dict(kind=kind,path=str(path)))
            check['errors'].append(f'missing {kind}: {path}');return
        actual=replay.sha256(path)
        if expected is not None and actual!=expected:raise ValueError(f'historical hash mismatch: {kind}: {path}')
        check['required_files'].append(dict(kind=kind,path=str(path),sha256=actual,bytes=path.stat().st_size))
    for rel,sha in PARENT_RECORDS.items():
        try:require('parent_'+rel,root/rel,sha)
        except ValueError as e:check['errors'].append(str(e))
    expected_assets=['sparc_source_memory.pt','sparc_source_mad_state.npz',
        *[f'projection/L{l}/{name}.npy' for l in LAYERS for name in ('mu','basis','rank')]]
    parent=prepare=infer=None
    try:
        parent=json.loads((root/'inputs.json').read_text(encoding='utf-8'))
        prepare=json.loads((root/'prepare_status.json').read_text(encoding='utf-8'))
        infer=json.loads((root/'infer_status.json').read_text(encoding='utf-8'))
        state=json.loads((root/'evaluation/status.json').read_text(encoding='utf-8'))
        if (parent.get('tool')!=parent_inputs.TOOL or parent['identity']!=PARENT_IDENTITY
                or digest(parent['frozen'])!=parent['identity']
                or prepare.get('tool')!=parent_inputs.TOOL or prepare['status']!='prepare_complete'
                or prepare['identity']!=parent['identity'] or prepare['inputs_sha256']!=replay.sha256(root/'inputs.json')
                or infer.get('tool')!=parent_inputs.TOOL or infer['status']!='predictions_frozen'
                or infer['identity']!=parent['identity'] or infer['predicted_images']!=297
                or infer['inputs_sha256']!=prepare['inputs_sha256']
                or infer['prepare_status_sha256']!=replay.sha256(root/'prepare_status.json')
                or state['identity']!=parent['identity'] or state['status']!='analysis_complete'
                or state['analyzed_images']!=297 or state['numerical_failed_checks']!=0
                or not state['pixel_metrics_complete'] or not state['formal_baseline_reproduced']):
            raise ValueError('parent is not the approved complete frozen experiment')
        validate_scope(parent);history_tables(root,parent['samples'])
        checks=replay.read_csv(root/'evaluation/numerical_verification.csv')
        if not checks or any(r['verified'].lower()!='true' for r in checks):raise ValueError('parent verification failed')
        if [r['sample_id'] for r in infer['records']]!=[s['sample_id'] for s in parent['samples']]:
            raise ValueError('parent prediction scope/order mismatch')
        check['parent']=parent
        check['paths'].update({k:v for k,v in parent['paths'].items() if k not in ('project','output')})
        guard_output(check['paths'],prepared=args.mode in ('infer','evaluate'))
        # Do not inherit irrelevant Source RGB/calibration cache existence gates.
        # Every asset we read is checked against its original parent record.
        for record in parent['required_files']:
            if record['kind'].startswith('source_RGB_'):continue
            if record['kind'].startswith('source_') and record['kind']!='source_mad_state.npz':continue
            try:require(record['kind'],record['path'],record['sha256'])
            except ValueError as e:check['errors'].append(str(e))
        artifacts={r['file']:r for r in prepare['artifacts']}
        for rel in expected_assets:
            if rel not in artifacts:raise ValueError(f'parent has no historical artifact record: {rel}')
            try:require('parent_asset_'+rel,root/rel,artifacts[rel]['sha256'])
            except ValueError as e:check['errors'].append(str(e))
        for r in infer['records']:
            if r['file']!=f"sparc_raw/{r['sample_id']}.npz":raise ValueError('unsafe parent raw path')
            try:require('parent_raw_'+r['sample_id'],root/r['file'],r['sha256'])
            except ValueError as e:check['errors'].append(str(e))
        proof=json.loads((root/'support_provenance.json').read_text(encoding='utf-8'))
        if proof['identity']!=parent['identity']:raise ValueError('support provenance identity differs')
        if len(proof['supports'])!=8:raise ValueError('eight original support provenance records required')
        check['supports']=proof['supports']
        for s,old in zip(proof['supports'],parent['frozen']['supports'],strict=True):
            if any(s[k]!=old[k] for k in ('support_index','relative_path','feature_sha256','image_sha256','grid','geometry')):
                raise ValueError('parent support provenance/order differs')
            require('original_support_'+str(s['support_index']),s['path'],old['feature_sha256'])
            if Path(s['path']).is_file():replay.check_support_header(Path(s['path']),s['relative_path'])
        check['resources']=resources(parent['frozen']['grid'],args.cpu_workers)
        check['resource_shape_source']='validated parent grid'
        check['resources']['disk_free_bytes']=parent_inputs.free_space(paths['output'])
        check['resources']['ram_available_bytes']=parent_inputs.available_ram()
        if args.mode in ('preflight','prepare-target'):
            if check['resources']['disk_free_bytes']<check['resources']['prepare_disk_required_bytes']:
                check['errors'].append('insufficient disk for corrected K8 Target and new predictions')
            ram=check['resources']['ram_available_bytes']
            if ram is not None and ram<check['resources']['minimum_available_ram_bytes']:
                check['errors'].append('insufficient available RAM for bounded support/query processing')
    except (OSError,ValueError,KeyError,TypeError,RuntimeError) as e:check['errors'].append(f'parent rejected: {e}')
    if parent is None or prepare is None:
        for rel in expected_assets:
            if not (root/rel).is_file():check['missing_assets'].append(dict(kind='parent_asset',path=str(root/rel)))
        if not (root/'sparc_raw').is_dir():check['missing_assets'].append(dict(kind='297_parent_SPARC_raw',path=str(root/'sparc_raw')))
    # The new implementation and shared dependencies are frozen before work.
    for path in Path(__file__).parent.glob('*.py'):require('code_new_'+path.name,path)
    for rel in ('DINOv3/MADEqual/aebad_s_union_support_loo/analysis.py',
                'DINOv3/MADEqual/aebad_s_union_support_loo/runner.py',
                'DINOv3/MADEqual/aebad_s_union_support_loo/inputs.py',
                'DINOv3/MADEqual/aebad_s_union_diagnostics/spatial.py',
                'DINOv3/MADEqual/aebad_s_target_support_trace/analysis.py'):
        require('code_shared_'+rel,paths['project']/rel)
    check['status']='blocked' if check['errors'] else 'filesystem_preflight_passed'
    return check


def freeze(check):
    if check['errors'] or check['parent'] is None:raise ValueError('successful preflight required')
    parent=check['parent']
    frozen=dict(parent_identity=parent['identity'],seed=42,K=8,grid=parent['frozen']['grid'],
        supports=check['supports'],cpu_workers=check['resources']['cpu_workers'],
        files=[{k:r[k] for k in ('kind','path','sha256')} for r in check['required_files']])
    return dict(tool=TOOL,identity=digest(frozen),frozen=frozen,parent=parent,
                paths=check['paths'],required_files=check['required_files'],resources=check['resources'])


def read_prepared(root):
    root=Path(root);data=json.loads((root/'inputs.json').read_text(encoding='utf-8'))
    state=json.loads((root/'prepare_target_status.json').read_text(encoding='utf-8'))
    if (data.get('tool')!=TOOL or data['identity']!=digest(data['frozen']) or state.get('tool')!=TOOL
            or state['status']!='target_prepared' or state['identity']!=data['identity']
            or state['inputs_sha256']!=replay.sha256(root/'inputs.json')):
        raise ValueError('Target preparation incomplete or history modified')
    verify_files(data['required_files']);parent_inputs.verify_artifacts(root,state['artifacts'])
    return data,state
