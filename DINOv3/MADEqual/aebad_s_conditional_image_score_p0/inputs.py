"""Dependency-based small-statistic imports and read-only inventory; no Torch."""
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
from .protocol import *
from . import numerics as nm
from DINOv3.MADEqual.aebad_s_joint_reference_response_p0 import inputs as jp
from DINOv3.MADEqual.aebad_s_shared_residual_translation_p0.inputs import prediction_backend

st=jp.st
ROOT=jp.ROOT
parent=jp.parent
image=jp.image_contract


def supports(a,d,seed):return parent(a).common['domains'][d]['supports'][str(seed)]
def samples(a,d):return parent(a).samples(d)
def key(row):return st.digest(row['relative_path'])[:24]
def unit(a,d,seed):return a.run_root/d/f'seed{seed}'


def normal_units(a):
    p=parent(a)
    for d in DOMAINS:
        for seed in SEEDS:
            for row in p.inputs['Source_calibration']:yield d,seed,'Source',row,None
            for j,row in enumerate(supports(a,d,seed)):yield d,seed,'Target',row,j


def contract(a,d,seed,kind,row,heldout=None):
    if kind not in ('Source','Target','query'):raise ValueError('explicit observation kind required')
    group=supports(a,d,seed)
    if not hasattr(a,'_source_images'):a._source_images={r['relative_path']:image(r) for r in parent(a).inputs['Source_calibration']}
    if kind=='Source' and a._source_images.get(row['relative_path'])!=image(row):raise ValueError('Source66 only')
    if kind=='Target':
        if heldout not in range(8) or image(row)!=image(group[heldout]):raise ValueError('current normal group heldout required')
    elif heldout is not None:raise ValueError('heldout only for Target LOO')
    refs=[image(r) for j,r in enumerate(group) if j!=heldout]
    if not hasattr(a,'_condition_dependencies'):
        p=parent(a)
        a._condition_dependencies=dict(source=p.inputs['source']['seal'],extraction=st.digest(p.inputs['extraction']),
            patch_MAD=st.digest(p.inputs['mad']['patch']),image_MAD=st.digest(p.inputs['mad']['image']),
            geometry=st.sha(ROOT/'DINOv3/MADEqual/aebad_s_target_local_segment/geometry.py'))
    return dict(schema='conditional_BP_v1',domain=d,support_seed=seed,kind=kind,query=image(row),
        **a._condition_dependencies,layers=list(LAYERS),dimension=1280,
        references=refs,K=len(refs),heldout=heldout,branch='SEGMENT_K8',
        statistic=CONFIG['B'],score=CONFIG['P'])


def retrieval_contract(c):
    return {**{k:v for k,v in c.items() if k not in ('schema','patch_MAD','image_MAD','statistic','score')},
        'schema':'conditional_original_Exact_inputs_v1','distance':'original_MM_Exact_FP32_euclidean_divide1280_once'}


def record_path(a,c):return unit(a,c['domain'],c['support_seed'])/'observations'/c['kind']/(st.digest(c['query'])[:24]+'.json')


def export_record(a,name,c):
    path=getattr(a,name,None)
    if not path:return None
    if not hasattr(a,'_exports'):a._exports={}
    if name not in a._exports:a._exports[name]=st.sealed_json(path)
    r=a._exports[name]
    if r.get('schema')!='conditional_exports_v1':raise ValueError('not a conditional-statistic export; old UNION/u statistics are incompatible')
    result=r['records'].get(st.digest(c))
    if result is not None and result.get('contract')!=c:raise ValueError('export contract mismatch')
    return result


def bp(a,c):
    path=record_path(a,c)
    r=st.sealed_json(path) if path.exists() else export_record(a,'statistics_provider',c)
    if r is None:return None
    if r['contract']!=c or r.get('GT_used') is not False:raise ValueError('B/P response/reference identity mismatch')
    nm.validate_observation(r,parent(a).inputs['mad'])
    return r


def response(a,c):
    ref=export_record(a,'response_provider',c)
    if ref is None and c['kind']=='query':
        row=next(r for r in samples(a,c['domain']) if image(r)==c['query'])
        try:
            found=jp.audit.inspect(jp.old_args(a),c['domain'],c['support_seed'],row)
        except FileNotFoundError:return None
        r=st.sealed_json(Path(found['path'])/'manifest.json')
        if 'SEGMENT_K8' not in r['files']:return None
        return dict(path=found['path'],seal=r['seal'],field='SEGMENT_K8',encoding='standardized_z',
                    contract=c,origin='frozen_two_K8_query_response')
    if ref is not None:
        r=st.sealed_json(Path(ref['path'])/'manifest.json')
        meta=r['metadata'];native=meta.get('contract')==c and meta.get('encoding')==ref['encoding']
        # Read-only bridge for original SEGMENT producers whose complete actual
        # dependencies predate this tool. Never rewrite their manifests/identity.
        dep=meta.get('dependencies',{})
        legacy=False
        if dep and ref['field'] in ('SEGMENT_K8','raw_segment'):
            wanted_encoding='standardized_z' if ref['field']=='SEGMENT_K8' else 'raw_distance'
            refs=dep.get('reference_images',dep.get('supports',[]))
            geometry=dep.get('geometry',{})
            legacy=(isinstance(geometry,dict) and isinstance(refs,list) and ref['encoding']==wanted_encoding and
                meta.get('encoding',wanted_encoding)==wanted_encoding and meta.get('branch','SEGMENT_K8') in ('SEGMENT_K8','K8_SEGMENT') and
                dep.get('domain',c['domain'])==c['domain'] and dep.get('support_seed',c['support_seed'])==c['support_seed'] and
                dep.get('Source_seed',42)==42 and dep.get('heldout',c['heldout'])==c['heldout'] and dep.get('source')==c['source'] and
                dep.get('extraction')==parent(a).inputs['extraction'] and geometry.get('segment_geometry')==c['geometry'] and
                geometry.get('distance')=='original_Exact_FP32_euclidean_divide_1280_once' and
                isinstance(dep.get('image'),dict) and image(dep['image'])==c['query'] and
                all(isinstance(v,dict) for v in refs) and [image(v) for v in refs]==c['references'] and
                (wanted_encoding=='raw_distance' or dep.get('patch')==parent(a).inputs['mad']['patch']))
        if r['seal']!=ref['seal'] or not (native or legacy):
            raise ValueError('native SEGMENT response actual dependencies differ')
        if ref['field'] not in r['files']:raise ValueError('declared SEGMENT field absent')
    return ref


def retrieval(a,c):
    rc=retrieval_contract(c);ref=export_record(a,'retrieval_provider',rc)
    if ref is not None:
        r=st.sealed_json(Path(ref['path'])/'manifest.json')
        if r['seal']!=ref['seal'] or r['metadata'].get('contract')!=rc:raise ValueError('retrieval provider actual dependencies differ')
    return ref


def feature_ref(a,row):
    # Same proof/header functions as the original adapter, with stage-local
    # authority caching. Do not parse/hash a large SACP assets JSON per image.
    from DINOv3.MADEqual.aebad_s_same_origin_neighbor_p0 import feature_identity as fi
    p=parent(a);paths=[];committed={};k=jp.previous.layer_inputs.old.key(row)
    for root in (a.cache_root,a.support_feature_root,a.memory43_root,a.raw_cache):
        if root:paths.extend((root/'normal_features'/k,root/k))
    if a.feature_manifest:
        export=fi.small_record(a,a.feature_manifest);fi.compatible(export.get('extraction'),p.inputs['extraction'])
        ref=export.get('normal_features',export.get('records',{})).get(row['relative_path'])
        if ref:paths.insert(0,Path(ref['path']));committed[Path(ref['path']).resolve()]=ref['seal']
    assets_path=a.sacp_run/'assets.json'
    if assets_path.exists():
        assets=fi.small_record(a,assets_path)
        if assets['references']['source']['seal']!=p.inputs['source']['seal']:raise ValueError('feature authority Source differs')
        fi.compatible(assets['references']['extraction'],p.inputs['extraction'])
        ref=assets.get('normal_features',{}).get(row['relative_path']) or assets.get('queries',{}).get(row.get('domain','')+'/'+row.get('sample_id',''))
        if ref:paths.append(Path(ref['path']));committed[Path(ref['path']).resolve()]=ref['seal']
    for path in dict.fromkeys(paths):
        if not (path/'manifest.json').exists():continue
        r=st.sealed_json(path/'manifest.json');m=r['metadata']
        if image(m.get('image',m.get('sample',{})))!=image(row):raise ValueError('feature image identity differs')
        if path.resolve() in committed and committed[path.resolve()]!=r['seal']:raise ValueError('feature provider manifest commitment differs')
        if 'extraction' in m:
            fi.compatible(m['extraction'],p.inputs['extraction']);proof=dict(kind='complete_inline',extraction_identity=st.digest(m['extraction']))
        else:
            if not hasattr(a,'_feature_bridge'):a._feature_bridge=SimpleNamespace(run_root=a.cache_root/'_unused_feature_probe',parent_run=a.sacp_run,
                raw_cache=a.raw_cache,feature_root=a.support_feature_root,feature_manifest=a.feature_manifest)
            proof=fi.proof(a._feature_bridge,path,row,r)
            if proof['extraction_identity']!=st.digest(p.inputs['extraction']):raise ValueError('feature extraction/provider differs')
        n=int(np.prod(m['grid']))
        for l in LAYERS:jp.previous.layer_inputs.old.header(path,'q'+str(l),(n,1280),np.float32)
        pos=jp.previous.layer_inputs.old.header(path,'positions',(n,),np.int64)
        jp.nm.positions_check(pos,m['grid'])
        return dict(path=str(path),seal=r['seal'],grid=m['grid'],proof=proof)
    return None


def bank(a):
    if hasattr(a,'_original_bank'):return a._original_bank
    p=parent(a);path=a.source_bank or Path(p.inputs['source']['path'])
    r=st.sealed_json(path/'manifest.json')
    if r['seal']!=p.inputs['source']['seal']:raise ValueError('original Source bank required; no rebuild')
    for l in LAYERS:jp.previous.layer_inputs.old.header(path,f'L{l}',(100000,1280),np.float32)
    a._original_bank=dict(path=str(path),seal=r['seal']);return a._original_bank


def freeze(a):
    p=parent(a)
    backend=prediction_backend(a)
    path=a.run_root/'inputs.json'
    if path.exists() and a.stage in ('fit','predict','evaluate','summarize'):backend=st.sealed_json(path)['backend']
    value=dict(config=CONFIG,parent_identity=p.identity,common_manifest=p.common['seal'],source=p.inputs['source'],
        mad=p.inputs['mad'],extraction=p.inputs['extraction'],Source_calibration=[image(r) for r in p.inputs['Source_calibration']],
        supports={f'{d}/{s}':[image(r) for r in supports(a,d,s)] for d in DOMAINS for s in SEEDS},
        backend=backend,routes={k:str(getattr(a,k)) if getattr(a,k,None) else None for k in (
            'parent_run','cache_root','memory42_root','memory43_root','raw_cache','sacp_run','support_feature_root',
            'source_bank','feature_manifest','statistics_provider','response_provider','retrieval_provider')},
        exports={k:st.sealed_json(getattr(a,k))['seal'] for k in ('feature_manifest','statistics_provider','response_provider','retrieval_provider') if getattr(a,k,None)},
        code={name:st.sha(Path(__file__).with_name(name)) for name in ('protocol.py','numerics.py','inputs.py','runtime.py','pipeline.py')})
    value['identity']=st.digest(value)
    r=st.sealed_json(path,value);st.sealed_json(a.run_root/'protocol.json',CONFIG);st.write_json(a.run_root/'job_plan.json',plan())
    return r


def signature(a):return st.sealed_json(a.run_root/'inputs.json')['identity']


def preflight(a):
    available=[];missing=[];work=[];feature_seen=set();bank_needed=False
    try:
        p=parent(a);freeze(a)
        available.append(dict(role='frozen_lists_MAD_CH1_scores_localization_identity',path=str(a.parent_run),action='read_only'))
        for d,seed,kind,row,j in [*normal_units(a),*((d,s,'query',r,None) for d in DOMAINS for r in samples(a,d) for s in SEEDS)]:
            c=contract(a,d,seed,kind,row,j);base=dict(domain=d,support_seed=seed,kind=kind,image=row['relative_path'])
            if bp(a,c):available.append(dict(base,role='compatible_BP',action='reuse_small_statistics'));continue
            ref=response(a,c)
            if ref:available.append(dict(base,role='native_SEGMENT_response',path=ref['path'],action='CPU_statistics_no_forward'));continue
            rr=retrieval(a,c)
            if rr:available.append(dict(base,role='per_support_distance_index',path=rr['path'],action='original_geometry_then_statistics'))
            else:
                bank_needed=True
                missing.append(dict(base,role='SEGMENT_response_or_BP',K=c['K'],reason='Source-only/UNION/u99 normal tables cannot substitute'))
                work.append(dict(base,role='original_Exact_and_SEGMENT',K=c['K']))
            # Retrieval reconstruction needs the selected Target vectors, not a model.
            needed=[r for r in supports(a,d,seed) if image(r) in c['references']]
            if not rr:needed=[row,*needed]
            for r in needed:
                if r['relative_path'] in feature_seen:continue
                feature_seen.add(r['relative_path']);fr=feature_ref(a,r)
                (available if fr else missing).append(dict(role='complete_features',image=r['relative_path'],path=fr['path'] if fr else '',
                    action='reuse' if fr else 'restore_or_explicit_extract_missing'))
                if not fr:work.append(dict(role='unique_missing_forward',image=r['relative_path']))
        try:available.append(dict(role='Source_bank',**bank(a),needed_for_supplement=bank_needed))
        except FileNotFoundError:
            (missing if bank_needed else available).append(dict(role='Source_bank',needed_for_supplement=bank_needed,
                path=str(a.source_bank or Path(p.inputs['source']['path'])),action='restore_no_rebuild' if bank_needed else 'not_required_with_complete_statistics'))
        if any(r['role']=='unique_missing_forward' for r in work):
            for role,path in (('checkpoint',a.checkpoint),('official_config',a.official_root/'config.json'),('backbone_code',a.dinov3_root/'hubconf.py')):
                (available if path is not None and path.is_file() else missing).append(dict(role=role,path=str(path) if path else '',
                    required_only_if_extract_missing=True,alternative='restore complete features; no model assets needed then'))
            (available if a.data.exists() else missing).append(dict(role='RGB_dataset_for_optional_extraction',path=str(a.data),
                required_only_if_extract_missing=True,identity_checked_by_original_loader=True))
        # GT is optional descriptive area only; inventory never opens a mask.
        available.append(dict(role='optional_GT_area',path=str(a.data),present=a.data.exists(),decoded=False,blocks_core_metrics=False))
        result=dict(status='missing_inputs' if missing else 'inputs_available',plan=plan(),missing_records=len(missing),
            missing_by_role=dict(__import__('collections').Counter(r['role'] for r in missing)),torch_imported='torch' in sys.modules)
    except (OSError,KeyError,ValueError) as e:
        missing.append(dict(role='input_error',reason=str(e)));result=dict(status='blocked',error=str(e),torch_imported='torch' in sys.modules)
    st.write_csv(a.run_root/'asset_inventory.csv',available);st.write_csv(a.run_root/'missing_inputs.csv',missing)
    st.write_csv(a.run_root/'necessary_work.csv',work);st.write_json(a.run_root/'preflight.json',result)
    if not (a.run_root/'time_estimate.json').exists():st.write_json(a.run_root/'time_estimate.json',dict(status='unmeasured',plan=plan(),
        stages={k:None for k in ('cache_statistics','normal_forwards','normal_Exact_geometry_CH1','query_forwards','query_Exact_geometry_CH1','write_evaluate_pack')},
        unique_missing_forwards=len([r for r in work if r['role']=='unique_missing_forward']),
        repetition='normal K8 SEGMENT and K7 SEGMENT are new work; missing test SEGMENT may repeat historical work; original features/Source search shared',
        over_12h_requires_consent=True,new_pixel_jobs=0,new_AUPRO_jobs=0))
    return result
