"""Frozen original K8 identities and read-only preflight; no Torch imports."""
from __future__ import annotations
import csv
import hashlib
import json
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .refinement import CONFIG
from . import sampling
from DINOv3.MADEqual.aebad_s_target_local_segment import inputs as shared
from DINOv3.MADEqual.aebad_s_union_support_loo.inputs import digest, verify_files
from DINOv3.MADEqual.aebad_s_sparc_p0.inputs import available_ram, free_space, file_records, verify_artifacts

replay = shared.replay
TOOL = 'aebad_s_fastref_target512_source_k8_v3_bounded_density'
LAYERS = (7, 15, 23, 31)
BRANCHES = ('SOURCE', 'UNION_K8', 'UNION_TARGET512', 'FASTREF_TARGET512_SOURCE')
NEW = BRANCHES[2:]
SAMPLING = dict(sampling.CONFIG)
CAPTURE_ID = '4fc53367a32843d12d60c9c7a3a6932f2af694af7dbfab63cf115de57e71a0c1'
CAPTURE_RECORDS = {
    'inputs.json': 'dbc394cc5e5e09ceb1cb4b1e37cbafbb5a4361319bcf55c0176ec5e99496ea84',
    'status.json': '9fcde0e0182c53f3baa939db4469e048d857166d5ec490e67b78a7ccda4c032e',
    'capture_verification.csv': 'd248cbe71175730dad05c98879c43aebbe179295aabfb83fb4b19af2d205a12f'}
# Full precision values also preserved in the user's approved parent review.
# Only the historical SOURCE/UNION rows are used; no SPARC features/dependencies.
HISTORICAL_METRICS = {
    'SOURCE': (.6690344062153163, .7789265453196502, .90546232309725, .07720519793396138, .9003327227786695),
    'UNION_K8': (.7962264150943397, .8845433394350746, .9104479516033466, .08726311488683787, .9001193187710007)}
METRIC_KEYS = ('image_AUROC', 'image_AP', 'pixel_AUROC', 'pixel_AP', 'AUPRO')


def sampling_compatibility(candidate_count):
    """Static dimension/count admission: no sampling, RNG or Torch."""
    result = sampling.admission(candidate_count,SAMPLING['quota_per_layer'],CONFIG['dimension'])
    if not result['error']:
        rows = (int(candidate_count)+result['effective_rounds']-1)//result['effective_rounds']
        result.update(max_group_rows=rows,max_knn_neighbors=min(SAMPLING['knn_neighbors'],rows),
            max_group_distance_matrix_estimate_bytes=min(rows*rows*4,SAMPLING['distance_matrix_max_bytes']))
    return result


def paths_for(args):
    project = Path(args.project_root).resolve()
    supplied = SimpleNamespace(**vars(args))
    supplied.data = args.data or str(project.parent/'datasets/AeBAD/AeBAD_S')
    supplied.output_dir = args.output_dir or str(project/'DINOv3/outputs/mad_equal/aebad_s_fastref_target512_source_bounded_density_seed42_v3')
    paths = shared.paths_for(supplied)
    paths.update(memory=Path(args.memory or paths['results']/'official_memory.pt').resolve(),
        official=project/'external_baselines/SuperADD',
        checkpoint=Path(args.checkpoint or project/'external_baselines/SuperADD/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth').resolve(),
        dinov3=Path(args.dinov3_root or project.parent/'dinov3').resolve(),
        selector=project/'DINOv3/MADEqual/aebad_s_fastref_k8_p0/sampling.py',
        selector_backend=project/'external_baselines/superadd_external/memory_variants.py',
        selector_upstream=project/'external_baselines/SuperADD/tracks/industrial/src/industrial/nearest_neighbor.py')
    return supplied, paths


def guard_output(paths, *, prepared=False):
    output = Path(paths['output'])
    for key, value in paths.items():
        if key in ('project', 'output'): continue
        protected = Path(value)
        if output == protected or output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError('output overlaps protected input: '+key)
    if not prepared and output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('new empty output required; no overwrite/resume/re-approval')


def original_metrics(results):
    rows = replay.read_csv(Path(results)/'domain_metrics.csv'); answer = []
    for branch, method in zip(BRANCHES[:2], replay.METHODS, strict=True):
        matches = [r for r in rows if r['domain'] == 'background' and r['method'] == method]
        if len(matches) != 1: raise ValueError('unique historical background metric row required')
        row = matches[0]
        values = tuple(float(row[k]) for k in ('image_AUROC', 'image_AUPR', 'pixel_AUROC', 'pixel_AUPR', 'AUPRO'))
        if int(row['image_count']) != 297 or any(not np.isfinite(v) or abs(v-e)>1e-12 for v,e in zip(values,HISTORICAL_METRICS[branch],strict=True)):
            raise ValueError('historical formal metric values differ from approved result')
        answer.append(dict(branch=branch,image_count=297,**dict(zip(METRIC_KEYS,values,strict=True))))
    return answer


def bundle_evidence(project):
    file = Path(project)/'downloads/aebad_s_segment_looB_raw_cache_20261013.tar'
    if not file.is_file(): return dict(present=False, contains_query_features=False)
    with tarfile.open(file) as archive:
        prefix = shared.CAPTURE_NAME+'/'
        matched = all(hashlib.sha256(archive.extractfile(prefix+relative).read()).hexdigest() == checksum
                      for relative,checksum in CAPTURE_RECORDS.items())
        names = archive.getnames()
        return dict(present=True,path=str(file),approved_capture_records_match=matched,
            loo_raw_files=sum(n.startswith(prefix+'raw_cache/') and n.endswith('.npz') for n in names),
            segment_raw_files=sum('/raw_cache/' in n and n.endswith('.npz') and not n.startswith(prefix) for n in names),
            contains_query_features=False, arrays_and_raw_hashes='not checked by this archive inventory; no extraction')


def preflight(args):
    supplied, paths = paths_for(args); guard_output(paths)
    check = shared.preflight(supplied)
    check.update(tool=TOOL,paths={k:str(v) for k,v in paths.items()},config=dict(CONFIG,cpu_workers=args.cpu_workers),sampling=dict(SAMPLING),
        missing_assets=[],model_loaded=False,Source_builder_called=False,MAD_fit_called=False,
        query_features='fresh original full-grid extraction only in explicit smoke/infer; distance/index caches are not Q')
    # shared preflight resolves test/GT root; preserve that resolved root.
    if 'data rejected' not in '\n'.join(check['errors']):
        paths['data'] = shared.data_root(paths['data']); check['paths']['data'] = str(paths['data'])
    cap = check.get('capture_inputs')
    def require(kind, path, checksum=None):
        path = Path(path)
        if not path.is_file():
            check['errors'].append('missing '+kind+': '+str(path)); return
        actual = replay.sha256(path)
        if checksum is not None and actual != checksum: check['errors'].append('historical hash mismatch: '+kind); return
        check['required_files'].append(dict(kind=kind,path=str(path),sha256=actual))
    for relative, checksum in CAPTURE_RECORDS.items(): require('approved_capture_'+relative,paths['capture']/relative,checksum)
    require('Source_memory',paths['memory'],cap['frozen']['memory_sha256'] if cap else '1b9e00c85fd6be35ba365d378b1da7fb2a673400319a256bb62408fe6e001717')
    require('checkpoint',paths['checkpoint'],replay.CHECKPOINT_SHA256)
    require('original_formal_metrics',paths['results']/'domain_metrics.csv')
    expected = cap['frozen']['input_code_sha256'] if cap else {}
    for kind,path in [('official_config',paths['official']/'config.json'),('official_model',paths['official']/'tracks/industrial/src/industrial/model.py'),('dinov3_hub',paths['dinov3']/'hubconf.py')]:
        require(kind,path,expected.get(kind))
    require('prototype_selector',paths['selector'])
    require('prototype_selector_backend',paths['selector_backend'])
    require('prototype_selector_upstream',paths['selector_upstream'])
    candidate_count = sum(int(np.prod(s['grid'])) for s in cap['supports']) if cap else 8*118*118
    check['sampling_compatibility'] = sampling_compatibility(candidate_count)
    check['sampling_compatibility']['shape_source'] = 'validated frozen support grids' if cap else 'expected frozen K8 118x118 grids; actual inputs missing'
    if check['sampling_compatibility']['error']: check['errors'].append(check['sampling_compatibility']['error'])
    for path in Path(__file__).parent.glob('*.py'): require('code_new_'+path.name,path)
    require('implementation_audit',Path(__file__).parent/'implementation_audit.md')
    for relative in ('DINOv3/MADEqual/aebad_s_sparc_p0/runtime.py','DINOv3/MADEqual/aebad_s_sparc_p0/analysis.py',
        'DINOv3/MADEqual/aebad_s_union_support_loo/analysis.py','DINOv3/MADEqual/aebad_s_union_support_loo/runner.py',
        'DINOv3/MADEqual/aebad_s_target_support_trace/analysis.py','DINOv3/MADEqual/aebad_s_full_p0/runner.py'):
        require('code_shared_'+relative,paths['project']/relative)
    try:
        if cap is not None:
            if cap['identity'] != CAPTURE_ID or cap['frozen']['precision']['cuda_matmul_allow_tf32'] is not False:
                raise ValueError('approved original K8 capture identity/precision required')
            for sample in (*cap['samples'],*cap['supports']):
                require('original_RGB_'+sample['relative_path'],paths['data']/sample['relative_path'],cap['image_sha256'][sample['relative_path']])
        if (paths['results']/'domain_metrics.csv').is_file(): check['historical_metrics'] = original_metrics(paths['results'])
        if (paths['official']/'config.json').is_file(): check['host_config'] = json.loads((paths['official']/'config.json').read_text())
        check['local_archive_evidence'] = bundle_evidence(paths['project'])
        if check['local_archive_evidence'].get('approved_capture_records_match') is False: raise ValueError('local bundle capture records modified')
    except (OSError,ValueError,KeyError,tarfile.TarError) as error: check['errors'].append(str(error))
    n = 118*118
    check['resources'] = dict(prototype_and_solver_estimate_bytes=4*512*1280*12,
        new_raw_estimate_bytes=297*n*168, prepare_disk_required_bytes=297*n*168+(2<<30)+4*512*1280*12,
        minimum_available_ram_bytes=16<<30,disk_free_bytes=free_space(paths['output']),ram_available_bytes=available_ram(),
        Source_candidate_scratch_bytes=0,query_features_persisted=False,interpretation='estimates, not observed peaks; OT globally solved on CPU, 4 layers at most')
    if check['resources']['disk_free_bytes'] < check['resources']['prepare_disk_required_bytes']: check['errors'].append('insufficient new-output disk space')
    ram = check['resources']['ram_available_bytes']
    if ram is not None and ram < check['resources']['minimum_available_ram_bytes']: check['errors'].append('insufficient available container RAM')
    check['missing_assets'] = [e for e in check['errors'] if e.startswith('missing ') or 'capture rejected' in e or 'data rejected' in e]
    check['status'] = 'blocked' if check['errors'] else 'filesystem_preflight_passed'
    check['torch_imported'] = 'torch' in sys.modules
    return check


def freeze(check):
    if check['errors']: raise ValueError('successful read-only preflight required')
    frozen = dict(config=check['config'],sampling=check['sampling'],capture_identity=CAPTURE_ID,host_config=check['host_config'],
        precision=check['capture_inputs']['frozen']['precision'],files=check['required_files'])
    return dict(tool=TOOL,identity=digest(frozen),frozen=frozen,paths=check['paths'],required_files=check['required_files'],
        capture=check['capture_inputs'],capture_status=check['capture_status'],historical_metrics=check['historical_metrics'],resources=check['resources'])


def verify_sampling(data, state):
    if (data.get('tool') != TOOL or data['frozen'].get('sampling') != SAMPLING
        or state.get('sampling') != SAMPLING):
        raise ValueError('bounded inverse-density sampler v3 required; old PatchCore/v2 or changed sampling artifacts rejected')


def prototype_metadata(data, layer):
    return dict(tool=TOOL,identity=data['identity'],layer=int(layer),sampling=dict(SAMPLING),
        source_code_sha256={key:replay.sha256(Path(data['paths'][key])) for key in ('selector','selector_backend','selector_upstream')},
        capture_identity=data['capture']['identity'],supports=[
            {key:s[key] for key in ('support_index','relative_path','grid','feature_sha256','image_sha256')}
            for s in data['capture']['supports']],shared_branches=list(NEW))


def prototype_identity(metadata, prototypes, indices):
    return digest(dict(metadata=metadata,
        prototypes_sha256=hashlib.sha256(np.ascontiguousarray(prototypes).tobytes()).hexdigest(),
        indices_sha256=hashlib.sha256(np.ascontiguousarray(indices).tobytes()).hexdigest()))


def read_prepared(output):
    root = Path(output); data = json.loads((root/'inputs.json').read_text())
    state = json.loads((root/'prepare_status.json').read_text())
    verify_sampling(data,state)
    if (data['tool'] != TOOL or data['identity'] != digest(data['frozen']) or data['frozen']['capture_identity'] != CAPTURE_ID
        or data['frozen']['config'] != dict(CONFIG,cpu_workers=data['frozen']['config']['cpu_workers'])
        or state['status'] != 'prototypes_prepared' or state['identity'] != data['identity']
        or state['inputs_sha256'] != replay.sha256(root/'inputs.json')):
        raise ValueError('prepared inputs/identity/config changed or incomplete')
    guard_output(data['paths'],prepared=True); verify_files(data['required_files']); verify_artifacts(root,state['artifacts'])
    capture = Path(data['paths']['capture'])
    if data['capture'] != json.loads((capture/'inputs.json').read_text()) or data['capture_status'] != json.loads((capture/'status.json').read_text()):
        raise ValueError('copied capture metadata differs from approved protected original')
    from .runtime import load_constants
    load_constants(data,state)
    return data,state
