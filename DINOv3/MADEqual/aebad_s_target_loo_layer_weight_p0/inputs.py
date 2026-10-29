from pathlib import Path
import numpy as np
from DINOv3.MADEqual.aebad_s_blind_context_p0 import inputs as old
from .protocol import CONFIG,LAYERS
st=old.st
ROOT=Path(__file__).resolve().parents[3]


def parent(a):
    if not hasattr(a,'_parent'):
        if a.parent_run is None or a.parent_cache is None:raise FileNotFoundError('supply --parent-run and --parent-cache for one consistent completed SACP parent')
        a._parent=old.Parent(a)
    return a._parent


def identity(a):return st.sealed_json(a.run_root/'inputs.json')['identity']


def freeze(a):
    p=parent(a)
    value=dict(config=CONFIG,parent_identity=p.identity,parent_run=str(a.parent_run),parent_cache=str(a.parent_cache),
        raw_run=str(a.raw_run) if a.raw_run else None,raw_cache=str(a.raw_cache) if a.raw_cache else None,
        data=str(a.data),parent_manifest=p.manifest['seal'],mad=p.mad['seal'],source=p.assets['references']['source']['seal'],
        supports=p.supports,source_calibration=[r for r in p.catalog['source'] if r['calibration']],extraction=p.extraction)
    value['identity']=st.digest(value)
    return st.sealed_json(a.run_root/'inputs.json',value)


def units(a):
    p=parent(a)
    return [('Source',i,r) for i,r in enumerate(p.catalog['source']) if r['calibration']]+[('Target',i,r) for i,r in enumerate(p.supports)]


def normal_path(a,kind,i):return a.cache_root/'normal_distances'/f'{kind}_{i:03d}'


def normal_identity(a,kind,i,row):
    p=parent(a)
    refs=[] if kind=='Source' else [r for j,r in enumerate(p.supports) if j!=i]
    return dict(identity=identity(a),kind=kind,image=row,source=p.assets['references']['source']['seal'],
        reference_images=[r['relative_path'] for r in refs],extraction=p.extraction)


def feature_ref(a,row):
    ref=old.raw_reference(a,row)
    if ref is None:raise FileNotFoundError('complete normal feature: '+row['relative_path'])
    old.inspect_raw(ref,row)
    return ref


def preflight(a):
    missing=[];reuse=[];unconfirmed=[]
    try:
        p=parent(a)
        for kind,i,row in units(a):
            path=normal_path(a,kind,i)
            if (path/'manifest.json').exists():
                _,r=st.get(path,('raw','positions'))
                if any(r['metadata'].get(k)!=v for k,v in normal_identity(a,kind,i,row).items()):raise ValueError('normal distance provenance differs')
                reuse.append(dict(kind=kind,path=str(path)))
            else:
                missing.append(dict(kind=kind,role='normal_distance',sample=row['relative_path'],path=str(path)))
                try:ref=feature_ref(a,row);reuse.append(dict(kind=kind,role='normal_feature',path=ref['path']))
                except FileNotFoundError as e:missing.append(dict(kind=kind,role='normal_feature',sample=row['relative_path'],path=str(e)))
        for row in p.samples:
            try:p.prediction(row,False)
            except FileNotFoundError:missing.append(dict(kind='evaluation',role='four_layer_parent_prediction',sample=row['sample_id'],path=str(p.cache/'predictions/background'/row['sample_id'])))
        bank=Path(p.assets['references']['source']['path'])
        if any(r.get('role')=='normal_distance' for r in missing) and not (bank/'manifest.json').exists():missing.append(dict(kind='Source',role='bank_for_supplement_only',path=str(bank)))
        if not a.data.exists():missing.append(dict(kind='evaluation',role='GT',path=str(a.data)))
        unconfirmed.append(dict(kind='legacy_assets',reason='old Source NPZ / evaluation leave-one-support-out / scalar LOO scores are not admitted without this parent bank/extraction/K7 image provenance'))
        state='inputs_available' if not missing else 'missing_inputs'
    except (OSError,ValueError,KeyError) as e:state='blocked';missing.append(dict(role='parent_metadata',path=str(e)))
    result=dict(status=state,missing=missing,reusable=reuse,unconfirmed=unconfirmed,config=CONFIG,
        estimate=dict(total_hours_range=[.15,1.5],conditional_on='existing original bank, 74 normal full features, 297 parent layer predictions and GT; no extraction or bank reconstruction',
        normal_retrieval='74 images x four layers, Source-only 66 and whole-image K7 LOO 8; 0.05-0.5h planning range, not measured',
        missing_feature_supplement='unknown until original host throughput and missing image count are known; explicit extract flag only',
        fitting_and_replay='0.02-0.2h',CPU_and_two_AUPRO='0.03-0.3h',remaining_IO='0.05-0.5h',
        evidence='same workstation prior three-map evaluation 72.5s plus AUPRO47.6s; no throughput measurement of this new replay; >12h pauses',
        unknown_if_assets_missing=True),torch_imported=__import__('sys').modules.get('torch') is not None)
    a.run_root.mkdir(parents=True,exist_ok=True)
    st.write_json(a.run_root/'preflight.json',result);st.write_csv(a.run_root/'missing_inputs.csv',missing)
    st.write_csv(a.run_root/'asset_inventory.csv',reuse+unconfirmed)
    return result
