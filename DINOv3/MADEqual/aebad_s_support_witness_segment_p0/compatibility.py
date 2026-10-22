"""Explicit historical recovery policy; strict original distance tests stay failed.

The old extraction directory is READ ONLY. New attestations belong to a fresh
certificate directory and a new experiment identity. No existing unsealed audit
is trusted or enrolled: maps/scores are recomputed from frozen inputs on CPU.
"""
from concurrent.futures import ThreadPoolExecutor
from numbers import Real
import math
from pathlib import Path
import time
from . import recovery as rec,storage as st
from .method import LAYERS
from DINOv3.MADEqual import aebad_s_witness_recovery_audit as audit

POLICY=dict(revision='historical-maps-v1',same_cache_score_atol=1e-5,recovered_geometry_score_atol=1e-5,
    fused_Source_MAD_map_atol=1e-4,per_layer_equal_weight_MAD_contribution_atol=1e-4,
    original_distance_atol=1e-7,original_distance_rtol=1e-6,original_distance_failures='reported_not_relabelled_passed',
    query_distances='original unchanged',GT_used=False,method_parameters_changed=False,
    interpretation='engineering recovery compatibility; not original bank bytes or query-NN equivalence')


def context(bundle,source):
    source=Path(source).resolve();p=source/'recovery_inputs.json';spec=st.read_json(p);f=spec['frozen']
    if (spec['identity']!=st.digest(f) or f['original']!=rec.original_proof(bundle) or f['revision']!=rec.REVISION
        or f.get('query')!=rec.QUERY or f.get('precision')!=rec.PRECISION or f.get('checkpoint_sha256')!=rec.CHECKPOINT_SHA256
        or f.get('extractor')!='DINOv3.ShiftADD.aebad_s_p0.memory._extract_four_layers' or f.get('brightness_factor') is not None
        or f.get('Source_memory_loaded') is not False or f.get('Source_MAD_refitted') is not False or f.get('query_forward_calls')!=0):
        raise ValueError('source extraction protocol/identity differs')
    host=st.read_json(bundle.bg['paths']['formal_dir']/'memory_manifest.json')['host_config']
    if f['host_config']!=host:raise ValueError('source recovered host config differs from frozen parent')
    records,locations,entries=rec.recovered_locations(bundle,source,spec)
    inputs=dict(policy=POLICY,original=rec.original_proof(bundle),source=str(source),source_inputs_sha256=st.sha(p),
        source_identity=spec['identity'],files=entries,audit_code_sha256=st.sha(Path(audit.__file__)),
        validation_code_sha256=st.sha(Path(__file__)))
    return inputs,spec,records,locations


def finite(value):return isinstance(value,Real) and not isinstance(value,bool) and math.isfinite(value)


def validate_domain(bundle,domain,rows):
    samples=[s for s in bundle.samples if s['domain']==domain];ids={s['sample_id'] for s in samples}
    checks,scores=rows['geometry'],rows['scores']
    expected={(sid,l,k) for sid in ids for l in LAYERS for k in ('segment_distance','endpoint_length')}
    if (len(checks)!=len(expected) or {(r['sample_id'],r['layer'],r['check_kind']) for r in checks}!=expected
        or len(scores)!=len(ids) or {r['sample_id'] for r in scores}!=ids):raise ValueError('incomplete/duplicate compatibility audit')
    count=0;max_layer=0.
    for r in checks:
        if (r['domain']!=domain or (r['atol'],r['rtol'])!=(1e-7,1e-6)
            or type(r['positions']) is not int or r['positions']<=0 or type(r['failed_positions']) is not int
            or not 0<=r['failed_positions']<=r['positions'] or type(r['strict_verified']) is not bool
            or r['strict_verified']!=(r['failed_positions']==0)):
            raise ValueError('invalid original strict distance audit')
        for k in ('max_abs_error','mean_abs_error','p99_abs_error','mean_signed_difference'):
            if not finite(r[k]):raise ValueError('nonfinite compatibility geometry statistic')
        if not 0<=r['mean_abs_error']<=r['max_abs_error'] or not 0<=r['p99_abs_error']<=r['max_abs_error']:raise ValueError('invalid drift range')
        count+=not r['strict_verified']
        if r['check_kind']=='segment_distance':
            expected_ratio=r['max_abs_error']/float(bundle.scales[LAYERS.index(r['layer'])])
            if abs(r['max_abs_error_over_Source_MAD_scale']-expected_ratio)>1e-12*(1+abs(expected_ratio)):
                raise ValueError('recovery MAD drift units differ from frozen Source scale')
            value=r['max_abs_error_over_Source_MAD_scale']/len(LAYERS)
            if not finite(value) or not 0<=value<=POLICY['per_layer_equal_weight_MAD_contribution_atol']:
                raise ValueError('recovered per-layer MAD contribution exceeds explicit compatibility policy')
            max_layer=max(max_layer,value)
    for r in scores:
        if (r['domain']!=domain or r['same_cache_map_SHA_verified'] is not True
            or r['same_cache_score_atol_1e_minus5_verified'] is not True):raise ValueError('original map/score not reproduced')
        for k in ('CH1_old','CH1_recovered_geometry','CH1_signed_difference','CURRENT_old','CURRENT_recovered_geometry',
                  'CURRENT_signed_difference','map_max_abs_error','map_mean_abs_error','map_mean_signed_difference'):
            if not finite(r[k]):raise ValueError('nonfinite recovery map/score')
        for k in ('CH1','CURRENT'):
            change=r[k+'_recovered_geometry']-r[k+'_old']
            if abs(change-r[k+'_signed_difference'])>1e-12 or abs(change)>POLICY['recovered_geometry_score_atol']:
                raise ValueError('recovered geometry image score exceeds original 1e-5 score tolerance')
        if not 0<=r['map_mean_abs_error']<=r['map_max_abs_error']<=POLICY['fused_Source_MAD_map_atol']:
            raise ValueError('recovered fused map exceeds explicit historical compatibility policy')
    return dict(domain=domain,images=len(ids),strict_failed_image_layer_checks=count,
        strict_distance_compatible=count==0,max_layer_MAD_contribution_error=max_layer,
        map_max_abs_error=max(r['map_max_abs_error'] for r in scores),
        CH1_max_abs_change=max(abs(r['CH1_signed_difference']) for r in scores),
        CURRENT_max_abs_change=max(abs(r['CURRENT_signed_difference']) for r in scores),
        peak_position_changes=sum(r['old_peak_position']!=r['recovered_geometry_peak_position'] for r in scores))


def verify(bundle,root,source,workers=8):
    tick=time.perf_counter();root=Path(root).resolve();source=Path(source).resolve()
    for p in [*bundle.protected_paths,source]:
        p=Path(p).resolve()
        if root==p or root in p.parents or p in root.parents:raise ValueError('new certificate overlaps immutable source input')
    if root.drive and root.drive.upper()!='G:':raise ValueError('Windows certificate must stay on G:')
    inputs,spec,records,locations=context(bundle,source);identity=st.digest(inputs);ipath=root/'compatibility_inputs.json'
    if root.exists():
        if not ipath.is_file() or st.read_json(ipath)!=dict(identity=identity,frozen=inputs):raise ValueError('certificate input/code identity differs')
    else:root.mkdir(parents=True);st.write_json(ipath,dict(identity=identity,frozen=inputs))
    if (root/'compatibility_manifest.json').exists():attach(bundle,root);return st.read_json(root/'compatibility_manifest.json')
    summaries=[];files={}
    for d in bundle.original_expected:
        path=root/'audits'/(d+'.json');seal=path.with_suffix('.seal.json')
        if path.exists() or seal.exists():
            if not path.is_file() or not seal.is_file() or st.read_json(seal)!=dict(identity=identity,domain=d,sha256=st.sha(path)):
                raise ValueError('unsealed/modified compatibility audit preserved')
            rows=st.read_json(path)
        else:
            banks=audit.load_domain_banks(bundle,locations,d);checks=[];scores=[]
            samples=[s for s in bundle.samples if s['domain']==d]
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for i,(a,b) in enumerate(pool.map(lambda s:audit.audit_image(bundle,s,banks),samples)):
                    checks.extend(a);scores.append(b)
                    if (i+1)%25==0 or i+1==len(samples):print(f'[recovery compatibility] {d} {i+1}/{len(samples)} cached CPU images',flush=True)
            del banks;rows=dict(geometry=checks,scores=scores)
            st.write_json(path,rows);st.write_json(seal,dict(identity=identity,domain=d,sha256=st.sha(path)))
        summaries.append(validate_domain(bundle,d,rows))
        files[path.relative_to(root).as_posix()]=st.sha(path);files[seal.relative_to(root).as_posix()]=st.sha(seal)
    manifest=dict(status='historical_map_score_compatible',recovery_identity=spec['identity'],certificate_identity=identity,
        original=rec.original_proof(bundle),files=inputs['files'],verification=files,policy=POLICY,domain_summaries=summaries,
        strict_distance_compatible=all(r['strict_distance_compatible'] for r in summaries),images=len(bundle.samples),layers=list(LAYERS),
        seconds=time.perf_counter()-tick,query_forward_calls=0,support_forward_calls=0,GT_used=False,
        limitation='Original query-NN equivalence and witness gate stability versus lost original features remain unverified')
    st.write_json(root/'compatibility_manifest.json',manifest)
    st.write_json(root/'compatibility_manifest.seal.json',dict(sha256=st.sha(root/'compatibility_manifest.json'),inputs_sha256=st.sha(ipath)))
    return manifest


def attach(bundle,root):
    root=Path(root).resolve();p=root/'compatibility_manifest.json';ipath=root/'compatibility_inputs.json'
    if st.read_json(root/'compatibility_manifest.seal.json')!=dict(sha256=st.sha(p),inputs_sha256=st.sha(ipath)):
        raise ValueError('modified compatibility certificate')
    old=st.read_json(ipath);source=Path(old['frozen']['source']);inputs,spec,records,locations=context(bundle,source)
    if old!=dict(identity=st.digest(inputs),frozen=inputs):raise ValueError('historical compatibility provenance/code changed')
    m=st.read_json(p)
    if (m['status']!='historical_map_score_compatible' or m['certificate_identity']!=old['identity'] or m['policy']!=POLICY
        or m['files']!=inputs['files'] or m['original']!=rec.original_proof(bundle) or m['recovery_identity']!=spec['identity']
        or m['images']!=len(bundle.samples) or m['layers']!=list(LAYERS)):raise ValueError('incompatible certificate scope/policy')
    expected={f'audits/{d}{suffix}' for d in bundle.original_expected for suffix in ('.json','.seal.json')}
    if set(m['verification'])!=expected:raise ValueError('incomplete compatibility audit files')
    for f,h in m['verification'].items():
        if st.sha(root/rec.safe(f))!=h:raise ValueError('modified compatibility audit')
    summaries=[]
    for d in bundle.original_expected:
        path=root/'audits'/(d+'.json')
        if st.read_json(path.with_suffix('.seal.json'))!=dict(identity=old['identity'],domain=d,sha256=st.sha(path)):
            raise ValueError('compatibility domain seal differs')
        summaries.append(validate_domain(bundle,d,st.read_json(path)))
    if summaries!=m['domain_summaries'] or m['strict_distance_compatible']!=all(r['strict_distance_compatible'] for r in summaries):
        raise ValueError('strict distance failure history altered')
    bundle.expected=records;bundle.recovered_locations=locations
    bundle.frozen['recovered_supports']=dict(manifest_sha256=st.sha(p),manifest=m,recovery_inputs=spec,compatibility_inputs=old)
    bundle.protected_paths.extend([root,source])
