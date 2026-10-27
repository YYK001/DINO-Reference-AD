"""Read-only recent assets, metadata/shape proofs; never rebuild missing banks."""
from pathlib import Path
import zlib
from importlib.metadata import version,PackageNotFoundError
import numpy as np
from .protocol import CONFIG,DOMAINS,LAYERS,job_plan
from DINOv3.MADEqual.aebad_s_reference_residual_segmentation import storage as st
from DINOv3.MADEqual.aebad_s_reference_residual_segmentation import inputs as old

ROOT=Path(__file__).resolve().parents[3]
def key(row):return old.key(row)
def identity(a):return st.sealed_json(a.run_root/'inputs.json')['identity']
def catalog(a):return st.sealed_json(a.run_root/'catalog.json')['catalog']
def config(a):return CONFIG
def plan(a):return job_plan()
def assets(a):return st.sealed_json(a.run_root/'assets.json')


def controls(c):
    folds=old.folds(c);result={}
    for fold in ('A','B'):
        candidates=[r for r in old.train_source(c) if folds[r['relative_path']]==fold]
        candidates=sorted(candidates,key=lambda r:(zlib.crc32(f"42|SACP_SOURCE_CONTROL|{fold}|{r['relative_path']}".encode())&0xffffffff,r['relative_path']))
        if len(candidates)<8:raise ValueError('insufficient Source fold controls')
        result[fold]=candidates[:8]
    return result


def record(path,fields,kind=None):
    path=Path(path);r=st.sealed_json(path/'manifest.json')
    if kind and r['metadata'].get('kind')!=kind:raise ValueError('non-original asset kind: '+str(path))
    for name,shape in fields.items():
        value=r['files'][name]
        if value['shape']!=list(shape) or value['dtype']!='float32':raise ValueError('wrong original bank/feature shape: '+str(path/name))
        if not (path/(name+'.npy')).is_file():raise FileNotFoundError(path/(name+'.npy'))
        header=np.load(path/(name+'.npy'),mmap_mode='r',allow_pickle=False)
        if header.shape!=tuple(shape) or header.dtype!=np.float32:raise ValueError('actual NPY header differs from manifest')
    return dict(path=str(path),seal=r['seal'],files=r['files'],metadata=r['metadata'])


def same_catalog(left,right):
    if any(left[k]!=right[k] for k in ('source','supports','samples')):raise ValueError('recent runs use different image/support populations')


def inventory(a):
    missing=[];providers=[];errors=[]
    for path in a.recent_run:
        try:
            p=old.recent(path)
            if p['kind']!='reference_residual':raise ValueError('use a recent original reference-residual run, not adapted features')
            expected=dict(bank_rows=100000,sampler_rounds=100,source_bank_augmentation='original_brightness_0.8_1.2',dimension=1280,layers=list(LAYERS))
            if any(p['inputs']['config'].get(k)!=v for k,v in expected.items()):raise ValueError('original Source feature/memory configuration differs')
            if providers:same_catalog(providers[0]['catalog'],p['catalog'])
            providers.append(p)
        except (OSError,ValueError,KeyError) as e:errors.append(dict(path=str(path),reason=str(e)))
    if not providers:return dict(status='blocked',missing=[dict(asset='recent reference-residual run inputs/catalog',path=str(p)) for p in a.recent_run] or [dict(asset='--recent-run',path='required')],errors=errors,config=CONFIG,job_plan=job_plan())
    c=providers[0]['catalog'];selected=controls(c);references={};normal={}
    for p in providers:
        path=p['cache']/'source_original'
        if not (path/'manifest.json').exists() or not (p['run']/'input_distance_mad.json').exists():continue
        try:
            r=record(path,{f'L{l}':(100000,1280) for l in LAYERS},'original_source')
            prepared=st.sealed_json(p['run']/'prepared.json')
            if prepared['source_seal']!=r['seal']:raise ValueError('Source not committed by parent prepare')
            mad=st.sealed_json(p['run']/'input_distance_mad.json')
            if len(mad['patch']['medians'])!=4 or len(mad['patch']['scales'])!=4 or min(mad['patch']['scales'])<=0:raise ValueError('invalid original patch MAD')
            for name in ('peak','local'):
                if len(mad['image'][name]['scales'])!=4 or min(mad['image'][name]['scales'])<=0:raise ValueError('invalid CH1 image MAD')
            references.update(source=r,mad=mad,parent_run=str(p['run']),parent_cache=str(p['cache']),parent_identity=p['identity'],extraction=p['inputs']['extraction'])
            break
        except (OSError,ValueError,KeyError) as e:errors.append(dict(asset='source/MAD',path=str(path),reason=str(e)))
    if 'source' not in references:missing.append(dict(asset='original Source bank + its Source patch/image MAD',path=str(providers[0]['cache']/'source_original'),also_required_parent_MAD=str(providers[0]['run']/'input_distance_mad.json')))
    for fold in ('A','B'):
        expected=[r['relative_path'] for r in old.train_source(c) if old.folds(c)[r['relative_path']]==fold]
        for p in providers:
            path=p['cache']/'source_folds'/('fold_'+fold)
            if not (path/'manifest.json').exists():continue
            try:
                r=record(path,{f'L{l}':(100000,1280) for l in LAYERS},'training_only_opposite_fold');m=r['metadata']
                if m['fold']!=fold or m['origins']!=expected or m['sampler']!='original_100_round_distance_based_normalize_false' or m['augmentation']!='original_brightness_0.8_1.2':raise ValueError('Source fold origin/sampler/exclusion protocol differs')
                references['fold_'+fold]=r;break
            except (OSError,ValueError,KeyError) as e:errors.append(dict(asset='fold_'+fold,path=str(path),reason=str(e)))
        if 'fold_'+fold not in references:missing.append(dict(asset='fold_'+fold,path=str(providers[0]['cache']/'source_folds'/('fold_'+fold)),policy='blocking; never rebuild or change control protocol'))
    needed=sum(selected.values(),[])+sum(c['supports'].values(),[])
    for row in needed:
        for p in providers:
            path=p['cache']/'normal_features'/key(row)
            if not (path/'manifest.json').exists():continue
            try:
                r0=st.sealed_json(path/'manifest.json');grid=r0['metadata']['grid'];n=int(np.prod(grid))
                r=record(path,{f'q{l}':(n,1280) for l in LAYERS},'original_complete_grid')
                if r['metadata']['image']!=row:raise ValueError('normal/support identity differs')
                prepared=st.sealed_json(p['run']/'prepared.json')
                if prepared['normal_features'][row['relative_path']]['seal']!=r['seal']:raise ValueError('feature not in committed parent preparation')
                pos=np.load(path/'positions.npy',mmap_mode='r',allow_pickle=False)
                if pos.dtype!=np.int64 or not np.array_equal(np.sort(pos),np.arange(n)):raise ValueError('full physical positions required')
                normal[row['relative_path']]=r;break
            except (OSError,ValueError,KeyError) as e:errors.append(dict(asset='original_complete_grid',path=str(path),reason=str(e)))
        if row['relative_path'] not in normal:missing.append(dict(asset='original_complete_grid',path=str(providers[0]['cache']/'normal_features'/key(row)),image=row['relative_path']))
    queries={};prediction_parents={}
    for sample in c['samples']:
        token=sample['domain']+'/'+sample['sample_id']
        for p in providers:
            query=p['cache']/'queries'/sample['domain']/sample['sample_id']
            if (query/'manifest.json').exists():
                r=st.sealed_json(query/'manifest.json');m=r['metadata'];n=int(np.prod(m['grid']))
                if m['identity']!=p['identity'] or m['sample']!=sample:raise ValueError('cached raw Q identity differs')
                q=record(query,{f'q{l}':(n,1280) for l in LAYERS})
                pos=np.load(query/'positions.npy',mmap_mode='r',allow_pickle=False)
                if pos.dtype!=np.int64 or not np.array_equal(np.sort(pos),np.arange(n)):raise ValueError('cached raw Q physical positions incomplete')
                q['parent_identity']=p['identity'];queries[token]=q;break
        for p in providers:
            prediction=p['cache']/'predictions'/sample['domain']/sample['sample_id']
            if (prediction/'manifest.json').exists():prediction_parents[token]=dict(path=str(prediction),parent=str(p['run']),identity=p['identity']);break
    count=len(c['samples'])-len(queries)
    for name,path in (('data',a.data),):
        if not Path(path).exists():missing.append(dict(asset=name,path=str(path)))
    if count:
        for name,path in (('checkpoint',a.checkpoint),('official_host',a.official_root/'config.json'),('dinov3',a.dinov3_root/'hubconf.py')):
            if not Path(path).exists():missing.append(dict(asset=name,path=str(path),reason=f'{count} full raw Q missing; fresh original-host forward needed'))
    # All providers must share the original extraction dependencies; no software guess.
    if 'extraction' in references:
        for p in providers:
            if p['inputs']['extraction']!=references['extraction']:raise ValueError('provider original extraction settings differ')
    return dict(status='blocked' if missing or errors else 'inventory_ready_not_runtime_verified',missing=missing,errors=errors,
        config=CONFIG,job_plan=job_plan(),catalog=c,controls=selected,references=references,normal_features=normal,queries=queries,
        prediction_parents=prediction_parents,query_forwards_required=count,query_features_reusable=len(queries),
        parent_query_modes={str(p['run']):p['inputs'].get('query_mode','not_recorded') for p in providers},
        expected_map_software=providers[0]['inputs']['extraction']['software'],
        new_work=['Source-only clustering, two-fold Target/Source residual statistics, 4 distance controls, CPU/AUPRO evaluation'],
        repeated_work=[dict(task='original full query forward',images=count,reason='stream-mode distances/indices/logits are not raw Q')],
        unconfirmed=['parent baseline numerical/map compatibility; NN reuse needs matching query and bank identities',
            'report archive presence does not prove remote payload availability'],
        audit='trusted manifest identity/shape/dtype/positions; no bulk SHA; formal baseline numeric checks use unchanged tolerances')


def preflight(a):
    try:r=inventory(a)
    except (OSError,ValueError,KeyError) as e:r=dict(status='blocked',missing=[],errors=[str(e)],config=CONFIG,job_plan=job_plan())
    versions={}
    required=['numpy','scipy','Pillow','scikit-learn','psutil','torch','adeval']
    if r.get('query_forwards_required',926):required+=['torchvision','timm','omegaconf','einops']
    for name in required:
        try:versions[name]=version(name)
        except PackageNotFoundError:versions[name]=None
    r.update(dependency_versions=versions,torch_imported='torch' in __import__('sys').modules,model_loaded=False,training=False,full_payload_hash_scan=False,
        filesystem_scope='current machine only; unavailable server absolute paths do not prove remote assets missing')
    r['missing_dependencies']=[n for n,v in versions.items() if v is None]
    r['map_software_compatibility']={name:dict(expected=value,installed=versions.get(name),matches=value==versions.get(name))
        for name,value in r.pop('expected_map_software',{}).items() if name in ('numpy','scipy','Pillow')}
    if any(not v['matches'] for v in r['map_software_compatibility'].values()):r['errors'].append('frozen map software differs; use parent environment, do not install/upgrade automatically')
    if versions.get('adeval') not in (None,'1.1.0'):r['errors'].append('ADEval 1.1.0 required')
    if r['missing_dependencies'] or r['errors']:r['status']='blocked'
    if 'catalog' in r:
        c=r.pop('catalog');r['counts']={d:CONFIG['populations'][d] for d in DOMAINS}
        r['Source_counts']=dict(training=455,calibration=66,controls_per_fold=8)
        r['assets_reusable']={k:dict(path=v.get('path'),seal=v.get('seal')) for k,v in r.pop('references').items() if isinstance(v,dict) and 'path' in v}
        r['normal_complete_fit_images_reusable']=len(r.pop('normal_features'));r.pop('queries');r['parent_prediction_candidates']=len(r.pop('prediction_parents'))
    return r


def freeze(a,r):
    if r['status']!='inventory_ready_not_runtime_verified':raise ValueError('missing raw assets; see preflight; no rebuild')
    payload={k:r[k] for k in ('references','normal_features','queries','prediction_parents','controls')}
    c=dict(r['catalog'],data_root=str(a.data));st.sealed_json(a.run_root/'catalog.json',dict(catalog=c))
    ar=st.sealed_json(a.run_root/'assets.json',payload)
    code={p.name:st.sha(p) for p in Path(__file__).parent.glob('*.py')}
    value=dict(config=CONFIG,job_plan=job_plan(),assets_seal=ar['seal'],catalog=st.digest(c),cache=str(a.cache_root),code=code)
    value['identity']=st.digest(value)
    st.sealed_json(a.run_root/'inputs.json',value)
    st.sealed_json(a.run_root/'protocol.json',CONFIG);st.sealed_json(a.run_root/'job_plan.json',job_plan())
    mad=r['references']['mad'];st.sealed_json(a.run_root/'input_distance_mad.json',dict(identity=value['identity'],patch=mad['patch'],image=mad['image'],scope=mad['scope'],origin=r['references']['parent_run']))
    st.write_csv(a.run_root/'source_control_images.csv',[dict(fold=f,reference_fold='B' if f=='A' else 'A',relative_path=q['relative_path'],source_index=q['source_index']) for f,rows in r['controls'].items() for q in rows])
    return value
