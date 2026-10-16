"""Read-only admission and identities of the approved, completed parent."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
from DINOv3.MADEqual.aebad_s_fastref_k8_p0 import inputs as parent
from DINOv3.MADEqual.aebad_s_fastref_k8_p0 import runtime as parent_runtime

replay = parent.replay
TOOL = 'aebad_s_fastref_projection_target512_source_v1'
BRANCH = 'PROJECTION_TARGET512_SOURCE'
BRANCHES = (*parent.BRANCHES, BRANCH)
REFERENCES = ('FASTREF_TARGET512_SOURCE','UNION_TARGET512','UNION_K8')
LAYERS = parent.LAYERS
PARENT_NAME = 'aebad_s_fastref_target512_source_bounded_density_v3_20261013_211140'
PARENT_ID = '53fb970afcf50b7fd821bf438fe2eb9ddeaafcc5efec82a68314d57d121e0390'
# Read from the approved downloaded package, not re-authorized from current files.
APPROVED = {
    'inputs.json':'25ae7701d6cf1045c76a128a272c3f5eeddc7b6c984d510faaeb6f305e1e732b',
    'prepare_status.json':'56cf6b873420f0598a14f9aa6f49b825a181aa7bf1ce42a1033d6157224447b9',
    'infer_status.json':'35fd1ded07ec7ad51da77c3c0126bcfe4be00b6adee0db664d2949da29f21ed9',
    'evaluation/status.json':'512cece383ba8e2de96f5fd65ffb8dd02d81f3ef63b9497938f6446a6bd48b24',
    'evaluation/metrics.csv':'d33fb1356ac5a9f8985dc85094d80a20992afc1efb494e366c39dc355360af30',
    'evaluation/per_image_scores.csv':'a44c1b59acce980d0a7769a9f56d0998d49fd4a55b4e825869f89e510e7994af',
    'evaluation/defect_instances.csv':'c3cd4f0f97501ef9750c6ae669cd059dbe6b4b7b50849c9fdc4b0df6db3a0856',
    'cost_prepare.json':'b37a50f762c9f982347fbec9bfd37bc4b5d52fba63c276bbcf9c0f96e8efe944',
    'cost_smoke.json':'a5c4292aca49c6ddc137c488ec6867c4bad21fb61d0a319ea5d95bf497eef64c',
    'cost_infer.json':'b1bc1291df39660c8d8a8aa7139202f9f2c29f2205ab5ddbeba10227f004e224',
    'cost_evaluate.json':'6c04dbd414b71a1e2dbd84e423a305ef1a6551e77c7abc88edf54060fada98b0',
}
CONFIG = dict(branch=BRANCH, prototype_count=512, dimension=1280,
              divisor=1280, OT=False, reference_search='parent_all_projection_ExactNN_64MiB',
              residual_search='independent_float64_subtraction_diagnostic_only')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def parent_records(root):
    """Approved records plus hashes already bound into the pinned parent statuses."""
    root = Path(root)
    records = [dict(kind='parent_'+name,path=str(root/name),sha256=sha) for name,sha in APPROVED.items()]
    for name,field in [('prepare_status.json','artifacts'),('infer_status.json','records')]:
        state = read_json(root/name)
        for r in state[field]:
            path = (root/r['file']).resolve()
            if not path.is_relative_to(root.resolve()): raise ValueError('unsafe parent artifact path')
            records.append(dict(kind='parent_artifact',path=str(path),sha256=r['sha256']))
    return records


def guard(parent_dir, output, *, existing=False):
    p,o = Path(parent_dir).resolve(),Path(output).resolve()
    if o == p or o.is_relative_to(p) or p.is_relative_to(o):
        raise ValueError('parent and new output must be separate')
    if not existing and o.exists() and (not o.is_dir() or any(o.iterdir())):
        raise ValueError('new empty output required; no overwrite/resume')


def check_scope(data):
    samples = data['capture']['samples']; selected = data['capture']['selected_cases']
    supports = data['capture']['supports']
    ids = [s['sample_id'] for s in samples]
    if (len(ids)!=297 or len(set(ids))!=297 or sum(int(s['label'])==0 for s in samples)!=85
            or sum(int(s['label'])==1 for s in samples)!=212
            or len(supports)!=8 or [s['support_index'] for s in supports]!=list(range(1,9))
            or {s['relative_path'] for s in samples} & {s['relative_path'] for s in supports}
            or [s['sample_id'] for s in selected]!=[i for i,_ in replay.CASES]
            or not set(s['sample_id'] for s in selected).issubset(ids)):
        raise ValueError('original supports/order, exclusion or 297/85/212/12 scope changed')


def read_parent(root):
    root = Path(root).resolve()
    parent.verify_files(parent_records(root))
    data,prepared = parent.read_prepared(root)
    if data['identity'] != PARENT_ID or Path(data['paths']['output']).resolve()!=root:
        raise ValueError('approved parent identity or original frozen output path changed')
    check_scope(data)
    state = parent_runtime.verify_stage(data,prepared)
    evaluated = read_json(root/'evaluation/status.json')
    if (evaluated['status']!='analysis_complete' or evaluated['identity']!=PARENT_ID
            or evaluated['analyzed_images']!=297 or evaluated['selected_figures']!=12
            or not evaluated['pixel_metrics_complete'] or evaluated['numerical_failed_checks']
            or evaluated['missing_gt_ids'] or not evaluated['protected_inputs_unchanged']):
        raise ValueError('completed parent evaluation required')
    return data,prepared,state


def preflight(args):
    root = Path(args.parent_dir).resolve(); output = Path(args.output_dir).resolve()
    guard(root,output)
    report = dict(tool=TOOL,status='blocked',errors=[],missing_assets=[],parent=str(root),output=str(output),
                  config=CONFIG,model_loaded=False,feature_extraction=False,sampling=False,OT=False,
                  query_features='no trusted Q in the approved package; fresh original-host extraction required',
                  torch_imported='torch' in sys.modules)
    for relative in APPROVED:
        file = root/relative
        if not file.is_file(): report['missing_assets'].append(str(file))
        elif replay.sha256(file)!=APPROVED[relative]: report['errors'].append('modified parent: '+relative)
    for layer in LAYERS:
        file=root/'prototypes'/f'L{layer}.npz'
        if not file.is_file(): report['missing_assets'].append(str(file))
    if not (root/'inputs.json').is_file():
        project=Path(__file__).resolve().parents[3]
        base=project/'DINOv3/outputs/mad_equal/aebad_s_target_fewshot_seed42_v1'
        report['assets'] = dict(memory=str(base/'official_memory.pt'),mad=str(base/'source_mad_state.npz'),
            checkpoint=str(project/'external_baselines/SuperADD/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth'),
            dinov3=str(project.parent/'dinov3'),data=str(project.parent/'datasets/AeBAD/AeBAD_S'))
        report['asset_path_note']='expected defaults only; exact query paths/hashes require parent inputs.json'
        for path in report['assets'].values():
            if not Path(path).exists(): report['missing_assets'].append(path)
    if (root/'inputs.json').is_file():
        try:
            data = read_json(root/'inputs.json')
            for row in data['required_files']:
                if not Path(row['path']).is_file(): report['missing_assets'].append(row['path'])
            for row in parent_records(root):
                if not Path(row['path']).is_file(): report['missing_assets'].append(row['path'])
            report['assets'] = {k:data['paths'][k] for k in ('memory','mad','checkpoint','dinov3','data')}
        except (KeyError,ValueError,OSError) as error: report['errors'].append(str(error))
    report['missing_assets'] = sorted(set(report['missing_assets']))
    if report['missing_assets']: report['errors'].append('required parent/model/MAD/image assets missing')
    if not report['errors']:
        try:
            data,prepared,_ = read_parent(root)
            parent_runtime.load_constants(data,prepared)
            report['prototype_ids'] = prepared['prototype_ids']
            report['required_disk_bytes'] = 2 << 30
            report['minimum_ram_bytes'] = 16 << 30
            if parent.free_space(output)<report['required_disk_bytes']: report['errors'].append('less than 2 GiB disk headroom')
            if parent.available_ram()<report['minimum_ram_bytes']: report['errors'].append('less than 16 GiB available RAM')
        except Exception as error: report['errors'].append(str(error))
    if not report['errors']: report['status']='filesystem_preflight_passed'
    return report


def freeze(args, parent_data, prepared):
    root = Path(args.parent_dir).resolve(); output = Path(args.output_dir).resolve()
    paths = dict(parent_data['paths'], output=str(output), parent=str(root))
    # Also reject output overlapping any protected asset (including Source/data).
    parent.guard_output(paths)
    records = parent_records(root)+parent_data['required_files']
    for file in sorted(Path(__file__).parent.glob('*.py')):
        records.append(dict(kind='projection_code',path=str(file.resolve()),sha256=replay.sha256(file)))
    frozen = dict(config=CONFIG,cpu_workers=args.cpu_workers,parent_identity=PARENT_ID,
                  prototype_ids=prepared['prototype_ids'],files=records,paths=paths)
    return dict(tool=TOOL,identity=parent.digest(frozen),frozen=frozen,paths=paths,required_files=records)


def read_frozen(output):
    data = read_json(Path(output)/'inputs.json')
    if (data['tool']!=TOOL or data['identity']!=parent.digest(data['frozen'])
            or data['frozen']['config']!=CONFIG or data['paths']!=data['frozen']['paths']
            or data['required_files']!=data['frozen']['files'] or data['frozen']['parent_identity']!=PARENT_ID
            or Path(data['paths']['output']).resolve()!=Path(output).resolve()):
        raise ValueError('projection frozen inputs changed')
    guard(data['paths']['parent'],output,existing=True)
    parent.guard_output(data['paths'],prepared=True)
    parent.verify_files(data['required_files'])
    p,prepared,state = read_parent(data['paths']['parent'])
    if data['frozen']['prototype_ids']!=prepared['prototype_ids']:
        raise ValueError('parent prototypes/order changed')
    return data,p,prepared,state


def historical_scores(root):
    rows = replay.read_csv(Path(root)/'evaluation/per_image_scores.csv')
    result = {(r['branch'],r['sample_id']):float(r['image_score']) for r in rows}
    if len(rows)!=1188 or len(result)!=1188 or not np.isfinite(list(result.values())).all():
        raise ValueError('four complete finite historical score rows per image required')
    return result


def historical_metrics(root):
    rows = replay.read_csv(Path(root)/'evaluation/metrics.csv')
    if len(rows)!=4 or {r['branch'] for r in rows}!=set(parent.BRANCHES):
        raise ValueError('four historical metric rows required')
    for row in rows:
        row.update(image_count=int(row['image_count']),result_origin='reused_parent_result',
                   source_file=str(Path(root)/'evaluation/metrics.csv'),source_sha256=APPROVED['evaluation/metrics.csv'])
        for key in parent.METRIC_KEYS: row[key]=float(row[key])
        if row['image_count']!=297 or any(not np.isfinite(row[k]) or not 0<=row[k]<=1 for k in parent.METRIC_KEYS):
            raise ValueError('invalid parent metric')
    return rows
