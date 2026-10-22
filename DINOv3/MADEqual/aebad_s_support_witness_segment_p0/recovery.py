"""Explicit original-host support recovery. Historical SHA records stay immutable.

Only extract() imports the host/Torch. verify() uses original cached distances
and endpoint indices/lengths on CPU; it cannot establish equality of unsaved Q.
"""
import copy
import platform
import sys
import time
from pathlib import Path
import numpy as np
from . import storage as st
from .method import LAYERS
from DINOv3.MADEqual.aebad_s_target_local_segment.geometry import geometry_layer,DIST_ATOL,DIST_RTOL
from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.inputs import QUERY,PRECISION,Data
from DINOv3.MADEqual.aebad_s_target_fewshot.replay12 import CHECKPOINT_SHA256,PROTOCOL_ID
from DINOv3.MADEqual.aebad_s_neighborhood_segment_p0.inputs import safe

REVISION='original_unaugmented_support_recovery_v1'


def original_proof(bundle):
    return dict(original_supports=getattr(bundle,'original_expected',bundle.expected),
        background_capture=bundle.bg['capture']['identity'],background_segment=bundle.bg['identity'],
        cross_capture=bundle.parent.cross.identity,
        query_captures=bundle.parent.cross.required(),
        background_raw_archive=bundle.bg['manifest']['assets']['raw_archive'])


def feature_path(root,d,i):return Path(root)/'features'/d/f'{i:03d}.npz'


def specification(bundle,project,checkpoint,dinov3):
    project,checkpoint,dinov3=map(lambda p:Path(p).resolve(),(project,checkpoint,dinov3))
    if st.sha(checkpoint)!=CHECKPOINT_SHA256:raise ValueError('recovery checkpoint identity differs')
    official=project/'external_baselines/SuperADD'
    host=st.read_json(bundle.bg['paths']['formal_dir']/'memory_manifest.json')['host_config']
    files={}
    for label,root in [('host',official),('dinov3',dinov3)]:
        for p in sorted(root.rglob('*.py')):
            if '.git' not in p.parts:files[label+'/'+p.relative_to(root).as_posix()]=st.sha(p)
        if not any(k.startswith(label+'/') for k in files):raise FileNotFoundError('missing original host code: '+str(root))
    for rel in ('DINOv3/ShiftADD/aebad_s_p0/memory.py','DINOv3/relation_reliability/superadd_host.py',
                'DINOv3/MADEqual/aebad_s_sparc_p0/runtime.py'):
        files[rel]=st.sha(project/rel)
    frozen=dict(revision=REVISION,original=original_proof(bundle),query=QUERY,precision=PRECISION,
        checkpoint_sha256=CHECKPOINT_SHA256,host_config=host,extraction_code=files,
        recovery_code={p.name:st.sha(p) for p in Path(__file__).parent.glob('*.py')},
        extractor='DINOv3.ShiftADD.aebad_s_p0.memory._extract_four_layers',brightness_factor=None,
        transforms='original deterministic preprocessing, stitched full image, original batch/FP32',
        Source_memory_loaded=False,Source_MAD_refitted=False,query_forward_calls=0)
    return dict(identity=st.digest(frozen),frozen=frozen),dict(project=str(project),official=str(official),checkpoint=str(checkpoint),dinov3=str(dinov3))


def initialize(root,spec,bundle):
    root=Path(root).resolve()
    for p in bundle.protected_paths:
        p=Path(p).resolve()
        if root==p or root in p.parents or p in root.parents:raise ValueError('recovery overlaps protected original input')
    if root.drive and root.drive.upper()!='G:':raise ValueError('Windows recovery must stay on G:')
    path=root/'recovery_inputs.json'
    if root.exists():
        if not path.is_file() or st.read_json(path)!=spec:raise ValueError('recovery directory identity differs; use new directory')
    else:root.mkdir(parents=True);st.write_json(path,spec)
    return root


def metadata(row):
    return dict(domain=row['domain'],support_index=row['support_index'],relative_path=row['relative_path'],
        image_sha256=row['image_sha256'],grid=row['grid'],historical_feature_sha256=row['feature_sha256'],
        action='explicit_original_host_reextraction_not_original_file')


def validate_arrays(arrays,row):
    if (str(arrays['protocol_id'].item())!=PROTOCOL_ID or str(arrays['relative_path'].item())!=row['relative_path']
        or arrays['grid'].tolist()!=row['grid']):raise ValueError('recovered support header differs')
    for l in LAYERS:
        a=arrays[f'layer_{l}']
        if a.dtype!=np.float32 or a.shape!=(int(np.prod(row['grid'])),1280) or not np.isfinite(a).all():
            raise ValueError('recovered support must contain complete finite original FP32 rows')


def extract(bundle,root,spec,paths,domains,device_name):
    root=initialize(root,spec,bundle);started=time.perf_counter();pending=[];reused=0
    for d in domains:
        for row in bundle.original_expected[d]:
            p=feature_path(root,d,row['support_index']);record=st.restore(p,spec['identity'],metadata(row))
            if record:
                with np.load(p,allow_pickle=False) as z:validate_arrays(z,row)
                reused+=1
            else:pending.append(row)
    if not pending:return dict(status='supports_extracted',new=0,reused=reused,model_loaded=False)
    # No bank construction, sampling, MAD fitting, query extraction or capacity admission.
    from DINOv3.MADEqual.aebad_s_sparc_p0.runtime import load_host
    from DINOv3.ShiftADD.aebad_s_p0.memory import _extract_four_layers
    import torch
    model,device=load_host(dict(paths=paths,frozen=spec['frozen']),device_name)
    actual=dict(brightness_factor=model.preprocessing.min_brightness_factor,resize_factor=model.preprocessing.resize_factor,
        patch_size=model.patch_exec.patch_size,patch_overlap=model.patch_exec.patch_overlap,model_patch_size=model.patch_exec.model_patch_size,
        evaluation_downscale=model.evaluation_downscale,dino_training=model.backbone.dino.training,
        trainable_parameters=sum(p.requires_grad for p in model.backbone.dino.parameters()))
    if actual!=QUERY or model.preprocessing.max_brightness_factor!=1:raise ValueError('recovery host query settings differ')
    with Data(bundle.paths['data']) as dataset,torch.inference_mode():
        for row in pending:
            tick=time.perf_counter()
            layers,grid=_extract_four_layers(model,dataset.image(row),device=device,brightness_factor=None)
            arrays=dict(protocol_id=np.asarray(PROTOCOL_ID),relative_path=np.asarray(row['relative_path']),grid=np.asarray(grid,np.int32),
                **{f'layer_{l}':layers[l] for l in LAYERS})
            validate_arrays(arrays,row)
            st.seal(feature_path(root,row['domain'],row['support_index']),spec['identity'],arrays,metadata(row),compressed=False)
            print(f'[witness recovery] {row["domain"]}/{row["support_index"]:03d} original_full_support seconds={time.perf_counter()-tick:.2f}',flush=True)
    import psutil
    if sys.platform=='win32':peak=psutil.Process().memory_info().peak_wset
    else:
        import resource
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    result=dict(status='supports_extracted',new=len(pending),reused=reused,seconds=time.perf_counter()-started,
        python=sys.version,platform=platform.platform(),torch=torch.__version__,cuda=torch.version.cuda,GPU=torch.cuda.get_device_name(device),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device),
        process_peak_rss_bytes=int(peak),process_current_rss_bytes=psutil.Process().memory_info().rss,
        query_forward_calls=0,support_forward_calls=len(pending),Source_memory_loaded=False)
    st.write_json(root/'cost_records'/('extract_'+'_'.join(domains)+'.json'),result);return result


def check_geometry(raw,layer,banks):
    off=LAYERS.index(layer)
    result,_=geometry_layer(raw['target_distance'][:,off],raw['target_index'][:,off],raw['source_distance'][off],
        [banks[i] for i in range(1,9)])
    checks=[]
    for key,actual,expected in [('segment_distance',result['K8_SEGMENT'],raw['segment_distance'][off]),
                               ('endpoint_length',result['endpoint_length'],raw['endpoint_length'][off])]:
        error=np.abs(actual.astype(np.float64)-expected.astype(np.float64));bound=DIST_ATOL+DIST_RTOL*np.abs(expected.astype(np.float64))
        checks.append(dict(check_kind=key,verified=bool(np.all(error<=bound)),max_abs_error=float(error.max(initial=0)),
            failed_positions=int(np.count_nonzero(error>bound)),atol=DIST_ATOL,rtol=DIST_RTOL))
    return checks


def recovered_locations(bundle,root,spec):
    """Each newly extracted file remains independently sealed before verification."""
    records=copy.deepcopy(bundle.original_expected);locations={};entries=[]
    for d,rows in records.items():
        for row in rows:
            p=feature_path(root,d,row['support_index']);r=st.restore(p,spec['identity'],metadata(row))
            if r is None:raise FileNotFoundError('recover this fixed support first: '+str(p))
            locations[d,row['support_index']]=str(p)
            entries.append(dict(domain=d,support_index=row['support_index'],file=p.relative_to(root).as_posix(),sha256=r['sha256'],
                record_sha256=st.sha(p.with_suffix('.json')),historical_feature_sha256=row['feature_sha256']))
            row['historical_feature_sha256']=row['feature_sha256'];row['feature_sha256']=r['sha256']
    return records,locations,entries


def verify(bundle,root):
    """Verify ALL original 926 grids before the new support relation is allowed."""
    from .inputs import load_layer
    root=Path(root).resolve();spec=st.read_json(root/'recovery_inputs.json')
    if spec['identity']!=st.digest(spec['frozen']) or spec['frozen']['original']!=original_proof(bundle):raise ValueError('recovery original input identity differs')
    if (root/'verified_manifest.json').exists():
        attach(bundle,root);return dict(status='recovery_verified',reused=True)
    records,locations,entries=recovered_locations(bundle,root,spec);tables=[];started=time.perf_counter()
    for d in records:
        for l in LAYERS:
            path=root/'geometry_checks'/d/f'L{l}.json';scope=dict(recovery_identity=spec['identity'],domain=d,layer=l,files=entries)
            seal=path.with_suffix('.seal.json')
            if path.exists() or seal.exists():
                if not path.is_file() or not seal.is_file() or st.read_json(seal)!=dict(scope=scope,sha256=st.sha(path)):
                    raise ValueError('unsealed/modified recovery verification preserved')
                rows=st.read_json(path)
            else:
                banks=load_layer(records,locations,d,l);rows=[]
                for i,sample in enumerate(s for s in bundle.samples if s['domain']==d):
                    raw,_,_=bundle.load(sample)
                    rows.extend(dict(domain=d,layer=l,sample_id=sample['sample_id'],**c) for c in check_geometry(raw,l,banks))
                    if (i+1)%50==0:print(f'[witness recovery audit] {d}/L{l} {i+1} cached images',flush=True)
                del banks
                st.write_json(path,rows);st.write_json(seal,dict(scope=scope,sha256=st.sha(path)))
            expected={(s['sample_id'],kind) for s in bundle.samples if s['domain']==d for kind in ('segment_distance','endpoint_length')}
            if len(rows)!=len(expected) or {(r['sample_id'],r['check_kind']) for r in rows}!=expected or any(r['verified'] is not True for r in rows):
                raise ValueError('recovered features fail original cached SEGMENT/endpoint checks; preserved '+str(path))
            tables.extend([path,seal])
    verification={p.relative_to(root).as_posix():st.sha(p) for p in tables}
    manifest=dict(status='recovery_verified',revision=REVISION,recovery_identity=spec['identity'],original=original_proof(bundle),files=entries,
        verification=verification,images=len(bundle.samples),layers=list(LAYERS),atol=DIST_ATOL,rtol=DIST_RTOL,
        scope='all original cached SEGMENT distances and endpoint lengths; no Q or query-NN equivalence claim',
        limitations='original NPZ byte equality and query-to-bank nearest-neighbor equivalence cannot be established without original Q',
        Source_memory_loaded=False,query_forward_calls=0,seconds=time.perf_counter()-started)
    st.write_json(root/'verified_manifest.json',manifest)
    st.write_json(root/'verified_manifest.seal.json',dict(sha256=st.sha(root/'verified_manifest.json'),recovery_inputs_sha256=st.sha(root/'recovery_inputs.json')))
    return manifest


def attach(bundle,root):
    root=Path(root).resolve();p=root/'verified_manifest.json';seal=st.read_json(root/'verified_manifest.seal.json')
    if seal!=dict(sha256=st.sha(p),recovery_inputs_sha256=st.sha(root/'recovery_inputs.json')):raise ValueError('modified recovered support manifest')
    spec=st.read_json(root/'recovery_inputs.json');m=st.read_json(p)
    if (spec['identity']!=st.digest(spec['frozen']) or spec['frozen']['original']!=original_proof(bundle)
        or m['original']!=original_proof(bundle) or m['recovery_identity']!=spec['identity'] or m['revision']!=REVISION
        or m['status']!='recovery_verified' or m['images']!=len(bundle.samples) or m['layers']!=list(LAYERS)
        or (m['atol'],m['rtol'])!=(DIST_ATOL,DIST_RTOL)):raise ValueError('unverified/incompatible recovery provenance')
    for f,h in m['verification'].items():
        if st.sha(root/safe(f))!=h:raise ValueError('modified recovery geometry checks')
    records,locations,entries=recovered_locations(bundle,root,spec)
    if entries!=m['files']:raise ValueError('recovered support file/record changed after verification')
    bundle.expected=records;bundle.recovered_locations=locations
    bundle.frozen['recovered_supports']=dict(manifest_sha256=st.sha(p),manifest=m,recovery_inputs=spec)
    bundle.protected_paths.append(root)
