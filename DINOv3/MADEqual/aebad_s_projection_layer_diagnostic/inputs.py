"""Archive-to-local input mapping. Never follow recorded server asset paths."""
from __future__ import annotations
import csv
import hashlib
import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
import numpy as np
from PIL import Image
from .numerics import LAYERS
from .metrics import resource_bytes
from DINOv3.MADEqual.aebad_s_target_fewshot import replay12 as replay
from DINOv3.MADEqual.aebad_s_sparc_p0.inputs import available_ram,free_space

TOOL='aebad_s_projection_layer_diagnostic_v1'
PARENT='aebad_s_fastref_target512_source_bounded_density_v3_20261013_211140'
PROJECTION='aebad_s_fastref_projection_target512_source_20261014_121828'
PARENT_ID='53fb970afcf50b7fd821bf438fe2eb9ddeaafcc5efec82a68314d57d121e0390'
PROJECTION_ID='dd24ad2f504683c5175720e9b2f5b335b93178fef842ba5172ca977e639e09c8'
PINS={'parent':'bf981354d76560216ea25c30ee7fd02e21d3bf817bc03cc94218383bcd876191',
      'projection':'1b339c53bdeb7c52c21c75eca0d9c26395a6fe696c22cc8b4e2db3286fbe2eff',
      'mad':'65df0a748d918f20e5c85f2dbb73845a2fac2801d8d62a91e3bd0b7bb37e1a13'}
PREFIX='AeBAD/AeBAD_S/'


def digest(value):
    # Same serialization as the historical identity function, without host imports.
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def paths_for(args):
    root=Path(args.project_root).resolve()
    return dict(parent=Path(args.parent_archive or root/'downloads'/f'{PARENT}_results_20261014.tar.gz').resolve(),
        projection=Path(args.projection_archive or root/'downloads'/f'{PROJECTION}_results_20261014.tar.gz').resolve(),
        mad=Path(args.mad or root/'server_archives/aebad_s_target_fewshot_seed42_v1_formal_results/aebad_s_target_fewshot_seed42_v1/source_mad_state.npz').resolve(),
        data=Path(args.data_zip or root/'datasets/AeBAD.zip').resolve(),
        output=Path(args.output_dir or root/'DINOv3/outputs/mad_equal/aebad_s_projection_layer_diagnostic_v1').resolve())


def guard(paths):
    out=paths['output']
    for key,p in paths.items():
        if key=='output':continue
        if out==p or out.is_relative_to(p) or p.is_relative_to(out):raise ValueError('output overlaps protected input: '+key)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):raise ValueError('new empty output directory required; no overwrite/resume')


def read_bundle(path,root):
    names=('inputs.json','infer_status.json','prepare_status.json','evaluation/status.json',
        'evaluation/per_image_scores.csv','evaluation/metrics.csv','evaluation/defect_instances.csv',
        'evaluation/fixed_reference_instance_peaks.csv','evaluation/instance_comparisons.csv')
    data={};raw={};hashes={};seen=set()
    with tarfile.open(path,'r|gz') as archive:
        for item in archive:
            if not item.name.startswith(root+'/'):continue
            relative=item.name[len(root)+1:]
            selected=relative in names or (relative.startswith('raw/') and relative.endswith('.npz'))
            if not selected:continue
            if relative in seen or not item.isfile():raise ValueError('duplicate/nonregular selected archive entry: '+relative)
            seen.add(relative);blob=archive.extractfile(item).read();hashes[relative]=hashlib.sha256(blob).hexdigest()
            if relative.endswith('.json'):data[relative]=json.loads(blob)
            elif relative.endswith('.csv'):data[relative]=list(csv.DictReader(io.StringIO(blob.decode('utf-8-sig'))))
            else:
                with np.load(io.BytesIO(blob),allow_pickle=False) as f:
                    fields=('source_distance','target_full_distance') if root==PARENT else ('source_distance','projection_distance','projection_union_distance')
                    raw[relative]=dict(metadata=json.loads(str(f['metadata'].item())),
                        **{k:f[k].copy() for k in (*fields,'positions','grid','processed_shape')})
    for required in ('inputs.json','infer_status.json','evaluation/status.json','evaluation/per_image_scores.csv','evaluation/metrics.csv','evaluation/defect_instances.csv'):
        if required not in data:raise ValueError('archive missing '+required)
    state=data['infer_status.json'];identity=PARENT_ID if root==PARENT else PROJECTION_ID
    inp=data['inputs.json']
    if (inp['identity']!=identity or inp['identity']!=digest(inp['frozen']) or state['identity']!=identity
            or state['status']!='predictions_frozen' or state['images']!=297
            or state['inputs_sha256']!=hashes['inputs.json'] or len(state['records'])!=297
            or data['evaluation/status.json']['status']!='analysis_complete'):
        raise ValueError('approved complete archive identities/status required')
    wanted={f"raw/{r['sample_id']}.npz" for r in state['records']}
    if set(raw)!=wanted or len(wanted)!=297:raise ValueError('exactly 297 formal raw files required; smoke_raw excluded')
    for record in state['records']:
        if record['file']!=f"raw/{record['sample_id']}.npz" or hashes[record['file']]!=record['sha256']:
            raise ValueError('raw historical hash/path mismatch')
    return dict(data=data,raw=raw,hashes=hashes)


def validate_raw(raw,sample,identity,image_sha):
    m=raw['metadata'];grid=m['grid'];shape=m['output_shape'];p=raw['positions'];n=int(np.prod(grid))
    if (m['identity']!=identity or m['sample_id']!=sample['sample_id'] or m['relative_path']!=sample['relative_path']
            or m['image_sha256']!=image_sha or m['distance_divisor']!=1280
            or len(grid)!=2 or min(grid)<=0 or len(shape)!=2 or min(shape)<=0
            or raw['grid'].tolist()!=grid or p.dtype!=np.int64 or p.shape!=(n,)
            or not np.array_equal(np.sort(p),np.arange(n))
            or raw['processed_shape'].tolist()!=[m['geometry']['processed_height'],m['geometry']['processed_width']]
            or m['geometry']['original_height']!=sample['image_height'] or m['geometry']['original_width']!=sample['image_width']):
        raise ValueError('raw sample, complete geometry or physical positions invalid: '+sample['sample_id'])
    for k,a in raw.items():
        if k.endswith('_distance') and (a.shape!=(4,n) or a.dtype!=np.float32 or not np.isfinite(a).all() or np.any(a<0)):
            raise ValueError('invalid finite nonnegative float32 layer distances: '+k)


def historical_rows(bundle,branch,name):
    return [r for r in bundle['data']['evaluation/'+name+'.csv'] if r['branch']==branch]


def validate_pair(u,p,sample,image_sha,parent_raw_sha,prototype_ids):
    validate_raw(u,sample,PARENT_ID,image_sha);validate_raw(p,sample,PROJECTION_ID,image_sha)
    if (u['metadata']['geometry']!=p['metadata']['geometry'] or u['metadata']['output_shape']!=p['metadata']['output_shape']
            or u['metadata']['grid']!=p['metadata']['grid'] or not np.array_equal(u['positions'],p['positions'])
            or p['metadata']['parent_raw_sha256']!=parent_raw_sha or p['metadata']['prototype_ids']!=prototype_ids
            or p['metadata']['OT_called'] or not np.allclose(p['source_distance'],u['source_distance'],atol=1e-7,rtol=1e-6)
            or not np.array_equal(p['projection_union_distance'],np.minimum(p['source_distance'],p['projection_distance']))):
        raise ValueError('paired physical geometry, Source or Target/union semantics mismatch: '+sample['sample_id'])


def load_inputs(paths,*,verify_zip=True):
    checksums={key:sha(path) for key,path in paths.items() if key!='output'}
    for key in PINS:
        if checksums[key]!=PINS[key]:raise ValueError('approved input hash differs: '+key)
    parent=read_bundle(paths['parent'],PARENT);projection=read_bundle(paths['projection'],PROJECTION)
    original=parent['data']['inputs.json'];new=projection['data']['inputs.json'];cap=original['capture'];samples=cap['samples'];supports=cap['supports']
    ids=[s['sample_id'] for s in samples]
    if (len(ids)!=297 or len(set(ids))!=297 or sum(s['label']==0 for s in samples)!=85 or sum(s['label']==1 for s in samples)!=212
            or [r['sample_id'] for r in parent['data']['infer_status.json']['records']]!=ids
            or [r['sample_id'] for r in projection['data']['infer_status.json']['records']]!=ids
            or [s['sample_id'] for s in cap['selected_cases']]!=[i for i,_ in replay.CASES]
            or len(supports)!=8 or [s['support_index'] for s in supports]!=list(range(1,9))
            or [Path(s['relative_path']).name for s in supports]!=list(replay.TARGET_NAMES)
            or {s['relative_path'] for s in samples}&{s['relative_path'] for s in supports}
            or original['frozen']['config']['dimension']!=1280 or original['frozen']['config']['prototype_count']!=512
            or new['frozen']['parent_identity']!=PARENT_ID
            or new['frozen']['prototype_ids']!=parent['data']['prepare_status.json']['prototype_ids']):
        raise ValueError('frozen sample/layer/support/prototype identity mismatch')
    with np.load(paths['mad'],allow_pickle=False) as f:
        if str(f['protocol_id'].item())!=replay.PROTOCOL_ID:raise ValueError('MAD protocol mismatch')
        medians=f['medians'].copy();scales=f['scales'].copy()
    if (medians.shape!=(4,) or scales.shape!=(4,) or medians.dtype!=np.float32 or scales.dtype!=np.float32
            or not np.isfinite(medians).all() or not np.isfinite(scales).all() or np.any(scales<=0)):
        raise ValueError('original float32 MAD invalid')
    for recorded in (original['required_files'],new['required_files']):
        matches=[r for r in recorded if PurePosixPath(r['path']).name=='source_mad_state.npz']
        if len(matches)!=1 or matches[0]['sha256']!=checksums['mad']:raise ValueError('mapped MAD differs from archive identity')
    cache={};pixel_count=0;patch_count=0
    for sample in samples:
        sid=sample['sample_id'];name=f'raw/{sid}.npz';u=parent['raw'][name];p=projection['raw'][name]
        image_sha=cap['image_sha256'][sample['relative_path']]
        validate_pair(u,p,sample,image_sha,parent['hashes'][name],new['frozen']['prototype_ids'])
        cache[sid]=dict(U=np.minimum(u['source_distance'],u['target_full_distance']),P=p['projection_union_distance'],
                       positions=u['positions'],metadata=u['metadata'])
        pixel_count+=int(np.prod(u['metadata']['output_shape']));patch_count+=len(u['positions'])
    del parent['raw'],projection['raw']
    zip_checked=0
    with zipfile.ZipFile(paths['data']) as z:
        seen={};
        for item in z.infolist():seen[item.filename]=seen.get(item.filename,0)+1
        required={s['relative_path']:cap['image_sha256'][s['relative_path']] for s in [*samples,*supports]}
        required.update({s['relative_path'].replace('test/','ground_truth/',1):cap['gt_sha256'][s['relative_path'].replace('test/','ground_truth/',1)] for s in samples if s['label']})
        for relative,checksum in required.items():
            path=PREFIX+relative
            if PurePosixPath(relative).is_absolute() or '..' in PurePosixPath(relative).parts or seen.get(path)!=1:
                raise ValueError('missing/unsafe/duplicate manifest ZIP entry: '+relative)
            if verify_zip:
                h=hashlib.sha256()
                with z.open(path) as f:
                    for block in iter(lambda:f.read(1<<20),b''):h.update(block)
                if h.hexdigest()!=checksum:raise ValueError('ZIP manifest image/GT modified: '+relative)
                zip_checked+=1
    return dict(samples=samples,supports=supports,cache=cache,medians=medians,scales=scales,
                parent=parent,projection=projection,checksums=checksums,pixel_count=pixel_count,patch_count=patch_count,
                zip_checked=zip_checked,paths=paths)


def resource_estimate(context,workers):
    n=context['pixel_count'];resident=sum(a.nbytes for c in context['cache'].values() for a in (c['U'],c['P'],c['positions']))
    return dict(full_evaluation_pixels=n,formal_images=297,patches=context['patch_count'],
        exact_sort_estimate_bytes=resource_bytes(n),input_cache_bytes=resident,
        required_ram_estimate_bytes=resource_bytes(n)+resident+workers*(256<<20),
        scratch_bytes=41*n,required_disk_estimate_bytes=41*n+(1<<30),
        interpretation='conservative estimates, not observed peaks; ten float32 maps and one uint8 GT vector; one exact sort at a time')


def preflight(args):
    paths=paths_for(args);guard(paths)
    report=dict(tool=TOOL,status='blocked',errors=[],missing_inputs=[str(v) for k,v in paths.items() if k!='output' and not v.is_file()],
                paths={k:str(v) for k,v in paths.items()},torch_imported='torch' in sys.modules,
                model_loaded=False,feature_extraction=False,retrieval=False,OT=False,archive_extracted=False,analysis_executed=False)
    context=None
    if report['missing_inputs']:report['errors'].append('required local archive/ZIP/MAD missing')
    else:
        try:
            context=load_inputs(paths)
            resources=resource_estimate(context,args.cpu_workers);report.update(input_checks_passed=True,
                sample_counts=dict(total=297,normal=85,anomaly=212,supports=8),formal_raw_files=594,zip_checked=context['zip_checked'],resources=resources,
                input_sha256=context['checksums'],layers=list(LAYERS),pixel_metrics='exact CPU full pixels, no AUPRO',
                memory_budget_bytes=int(args.memory_budget_gib*2**30),available_ram_bytes=available_ram(),disk_free_bytes=free_space(paths['output']))
            if resources['exact_sort_estimate_bytes']>report['memory_budget_bytes']:report['errors'].append('exact sort exceeds configured memory budget')
            if report['available_ram_bytes'] is None:report['errors'].append('available RAM could not be determined')
            elif report['available_ram_bytes']<resources['required_ram_estimate_bytes']:report['errors'].append('insufficient available RAM for exact CPU metrics')
            if report['disk_free_bytes']<resources['required_disk_estimate_bytes']:report['errors'].append('insufficient disk for exact pixel scratch')
        except Exception as error:report['errors'].append(f'{type(error).__name__}: {error}')
    if not report['errors']:report['status']='inputs_and_resources_verified'
    return report,context


class DatasetZip:
    def __init__(self,path):self.archive=zipfile.ZipFile(path)
    def image(self,relative):
        with self.archive.open(PREFIX+relative) as f:
            with Image.open(f) as image:return image.convert('L').copy()
    def mask(self,sample):
        if not sample['label']:return None
        return np.asarray(self.image(sample['relative_path'].replace('test/','ground_truth/',1)),np.uint8)>0
    def close(self):self.archive.close()
