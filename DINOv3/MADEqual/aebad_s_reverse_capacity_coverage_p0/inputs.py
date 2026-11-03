"""Known-asset inventory and read-only adapters; no model or payload hash scan."""
import csv
import sys
from pathlib import Path
from importlib.metadata import version,PackageNotFoundError
import numpy as np
from .protocol import *
from DINOv3.MADEqual.aebad_s_conditional_image_score_p0 import inputs as previous
from DINOv3.MADEqual.aebad_s_same_origin_neighbor_p0 import feature_identity as fi
st=previous.st
ROOT=previous.ROOT
image=previous.image


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def small(a,path):return fi.small_record(a,path)


class Assets:
    def __init__(self,a):
        self.parent=previous.parent(a);self.conditional=small(a,a.conditional_run/'inputs.json')
        if self.conditional['parent_identity']!=self.parent.identity or self.conditional['common_manifest']!=self.parent.common['seal']:
            raise ValueError('conditional P provider uses another population or Source')
        self.extraction=self.parent.inputs['extraction'];self.source=small(a,a.raw_run/'catalog.json')['catalog']['source']
        from DINOv3.MADEqual.aebad_s_reference_residual_segmentation.inputs import validate_source
        validate_source(dict(source=self.source))
        if [image(r) for r in self.source if r['calibration']]!=[image(r) for r in self.parent.inputs['Source_calibration']]:
            raise ValueError('RMS Source455/Source66 membership differs')
        raw_inputs=small(a,a.raw_run/'inputs.json')
        if raw_inputs['catalog']!=st.digest(small(a,a.raw_run/'catalog.json')['catalog']):raise ValueError('raw catalog commitment differs')
        fi.compatible(raw_inputs['extraction'],self.extraction)
        self.train=[r for r in self.source if not r['calibration']];self.calibration=[r for r in self.source if r['calibration']]
        if len(self.train)!=455:raise ValueError('455 unaugmented Source images required for RMS')
        self.query={};self.normal={};self.verification=[];self.feature_index={}
        complete=small(a,a.conditional_run/'evaluation_complete.json')
        frozen=small(a,a.conditional_run/'predictions_frozen.json')
        if complete['identity']!=self.conditional['identity'] or complete['prediction_seal']!=frozen['seal']:
            raise ValueError('completed conditional score provider required')
        committed_scores={}
        for ref in frozen['records']:
            record=small(a,a.conditional_run/ref['domain']/f"seed{ref['support_seed']}"/'scores.json')
            if record['seal']!=ref['seal']:raise ValueError('frozen P score table changed')
            for item in record['records']:committed_scores[item['domain'],item['support_seed'],item['sample_id']]=item
        # Compact score CSV identity is verified once, not features or bank bytes.
        scores=csv_rows(a.conditional_run/'per_image_scores.csv')
        for row in scores:
            d,s,sid=row['domain'],int(row['support_seed']),row['sample_id'];sample=next(r for r in self.parent.samples(d) if r['sample_id']==sid)
            old=self.parent.record(d,s,sample);P=float(row['SEGMENT_CH1']);err=abs(P-float(old['scores']['SEGMENT_K8']['CH1']))
            committed=committed_scores[d,s,sid]
            if P!=committed['SEGMENT_CH1'] or row['map_hash']!=committed['map_hash'] or row['parent_prediction']!=committed['parent_prediction']:
                raise ValueError('P CSV differs from frozen per-setting scores')
            if not np.isfinite(P) or err>1e-5 or row['map_hash']!=old['map_hashes']['UNION_SOURCE_WEIGHT']:
                raise ValueError('original SEGMENT P/map identity differs; original 1e-5 unchanged')
            key=d,s,sid
            if key in self.query:raise ValueError('duplicate parent P score')
            self.query[key]=dict(P=P,origin=old['seal'],map_hash=row['map_hash'])
            self.verification.append(dict(domain=d,support_seed=s,sample_id=sid,CH1_abs_error=err,atol=1e-5,verified=True))
        expected={(d,s,r['sample_id']) for d in DOMAINS for s in SEEDS for r in self.parent.samples(d)}
        if set(self.query)!=expected:raise ValueError('exact 1804 frozen P instances required')
        normals=small(a,a.conditional_run/'normal_statistics.json')
        if normals['identity']!=self.conditional['identity'] or normals['GT_used']:raise ValueError('normal statistic identity/GT mismatch')
        for ref in normals['records']:
            if ref['kind']!='Source':continue  # No Target LOO needed by this method.
            d,s=ref['domain'],ref['support_seed'];path=Path(ref['path'])
            if not path.is_file():path=a.conditional_run/d/f'seed{s}'/'observations/Source'/path.name
            obs=small(a,path);c=obs['contract']
            wanted=[image(r) for r in self.supports(d,s)]
            if (obs['seal']!=ref['seal'] or obs['GT_used'] or c['kind']!='Source' or c['K']!=8 or c['heldout'] is not None
                or c['domain']!=d or c['support_seed']!=s or c['references']!=wanted or c['query']!=ref['image']
                or c['source']!=self.parent.inputs['source']['seal'] or c['extraction']!=st.digest(self.extraction)
                or c['patch_MAD']!=st.digest(self.parent.inputs['mad']['patch']) or c['image_MAD']!=st.digest(self.parent.inputs['mad']['image'])):
                raise ValueError('Source66 P is not current complete K8 SEGMENT calibration')
            key=d,s,ref['image']['relative_path']
            if key in self.normal:raise ValueError('duplicate Source calibration score')
            self.normal[key]=dict(P=float(obs['P']),origin=obs['seal'])
            origin=obs.get('origin',{})
            if origin.get('kind')=='new_original_Exact_SEGMENT':
                feature_refs=[(c['query'],origin.get('query_feature')),*zip(c['references'],origin.get('reference_features',[]))]
                for im,feature in feature_refs:
                    if feature and feature.get('path') and feature.get('seal') and feature.get('proof',{}).get('extraction_identity')==st.digest(self.extraction):
                        self.feature_index.setdefault(im['relative_path'],feature)
        expected={(d,s,r['relative_path']) for d in DOMAINS for s in SEEDS for r in self.calibration}
        if set(self.normal)!=expected or not all(np.isfinite(r['P']) for r in self.normal.values()):raise ValueError('396 Source normal P records required')
        self.identity=st.digest(dict(parent=self.parent.identity,conditional=self.conditional['seal'],
            query=self.query_as_list(),normal=[(list(k),v) for k,v in sorted(self.normal.items())],
            train=[image(r) for r in self.train],extraction=self.extraction))

    def query_as_list(self):return [(list(k),v) for k,v in sorted(self.query.items())]
    def supports(self,d,s):return self.parent.common['domains'][d]['supports'][str(s)]
    def samples(self,d):return self.parent.samples(d)
    def P(self,d,s,kind,row):return self.normal[d,s,row['relative_path']] if kind=='Source' else self.query[d,s,row['sample_id']]
    def region_rows(self):
        result={r['relative_path']:r for r in self.calibration}
        for d in DOMAINS:
            for s in SEEDS:
                for r in self.supports(d,s):result[r['relative_path']]=r
            for r in self.samples(d):result[r['relative_path']]=r
        return sorted(result.values(),key=lambda r:r['relative_path'])


def assets(a):
    if not hasattr(a,'_coverage_assets'):a._coverage_assets=Assets(a)
    return a._coverage_assets


def key(row):return st.digest(image(row))[:24]


def authorities(a):
    if hasattr(a,'_coverage_authorities'):return a._coverage_authorities
    x=assets(a)
    out=[dict(run=a.conditional_run,cache=a.conditional_run,identity=None,index=x.feature_index,
        seal=x.conditional['seal'],observed_complete_features=True)]
    # Both migrated experiments remain useful; no AutoDL/Kaggle dependency.
    for run in dict.fromkeys([a.raw_run,*a.asset_runs]):
        if not (run/'inputs.json').is_file():continue
        r=small(a,run/'inputs.json');ex=r.get('extraction')
        if ex is None and (run/'assets.json').exists():ex=small(a,run/'assets.json')['references']['extraction']
        if ex!=assets(a).extraction:continue
        cache=Path(r.get('cache',r.get('cache_root','')))
        index={}
        for name in ('prepared.json','raw_assets.json','assets.json'):
            if not (run/name).exists():continue
            record=small(a,run/name)
            index.update(record.get('normal_features',{}));index.update(record.get('queries',{}))
            if name=='prepared.json' and 'features' in record:
                index.update({k:dict(path=str(cache/'features'/previous.jp.previous.layer_inputs.old.key(dict(relative_path=k))),seal=v)
                    for k,v in record['features'].items()})
        out.append(dict(run=run,cache=cache,identity=r['identity'],index=index,seal=r['seal']))
    if a.feature_manifest:
        r=small(a,a.feature_manifest);fi.compatible(r['extraction'],assets(a).extraction)
        out.insert(0,dict(run=a.feature_manifest.parent,cache=a.feature_manifest.parent,identity=None,
            index={**r.get('records',{}),**r.get('normal_features',{}),**r.get('queries',{})},seal=r['seal']))
    a._coverage_authorities=out;return out


def candidates(a,row):
    seen=set();oldkey=previous.jp.previous.layer_inputs.old.key(row)
    for auth in authorities(a):
        ref=auth['index'].get(row['relative_path']) or auth['index'].get(row.get('domain','')+'/'+row.get('sample_id',''))
        paths=[(Path(ref['path']),ref)] if ref else []
        paths.extend(((auth['cache']/'normal_features'/oldkey,None),(auth['cache']/'queries'/row.get('domain','')/row.get('sample_id','_'),None)))
        for path,ref in paths:
            if str(path) not in seen:seen.add(str(path));yield path,ref,auth
    for root in [a.cache_root,*a.feature_roots]:
        for path in (root/'normal_features'/oldkey,root/oldkey,root/'queries'/row.get('domain','')/row.get('sample_id','_')):
            if str(path) not in seen:seen.add(str(path));yield path,None,None


def feature_ref(a,row):
    wanted=assets(a).extraction
    for path,ref,auth in candidates(a,row):
        if not (path/'manifest.json').is_file():continue
        record=small(a,path/'manifest.json');m=record['metadata'];actual=m.get('image',m.get('sample',{}))
        if not actual or image(actual)!=image(row):raise ValueError('complete feature image identity differs: '+str(path))
        names=[f'q{l}' for l in LAYERS]
        if not all(n in record['files'] for n in names):
            names=[f'L{l}' for l in LAYERS]
        if not all(n in record['files'] for n in names):names=[f'full_L{l}' for l in LAYERS]
        if not all(n in record['files'] for n in names):continue  # Distance grids never stand in for features.
        if ref and ref['seal']!=record['seal']:raise ValueError('feature differs from original provider commitment')
        if 'extraction' in m:fi.compatible(m['extraction'],wanted)
        elif auth is None or (not auth.get('observed_complete_features') and m.get('identity')!=auth['identity']):
            raise ValueError('feature extraction authority absent: '+str(path))
        if m.get('kind','original_complete_grid')!='original_complete_grid':raise ValueError('augmented/adapted feature population rejected')
        geometry=m.get('geometry',m);grid=geometry['grid'];n=int(np.prod(grid))
        for name in names:
            x=np.load(path/(name+'.npy'),mmap_mode='r',allow_pickle=False)
            if x.shape!=(n,1280) or x.dtype!=np.float32:raise ValueError('not complete 1280D FP32 features: '+str(path))
        position_name='full_positions' if names[0].startswith('full_') else 'positions'
        if position_name in record['files']:pos=np.load(path/(position_name+'.npy'),mmap_mode='r',allow_pickle=False)
        else:pos=np.asarray(geometry.get('positions',[]),np.int64)
        if pos.shape!=(n,) or not np.array_equal(np.sort(pos),np.arange(n)):raise ValueError('physical grid incomplete/overlapping')
        return dict(path=str(path),seal=record['seal'],names=names,grid=grid,
            positions_file=position_name if position_name in record['files'] else None,positions=None if position_name in record['files'] else pos.tolist(),
            extraction=st.digest(wanted),authority=auth['seal'] if auth else 'inline_full_extraction')
    return None


def load_features(a,row):
    ref=feature_ref(a,row)
    if ref is None:raise FileNotFoundError('complete original feature absent: '+row['relative_path'])
    keys=tuple(ref['names'])+((ref['positions_file'],) if ref['positions_file'] else ())
    data,record=st.get(ref['path'],keys)
    if ref['seal']!=record['seal']:raise ValueError('feature manifest changed')
    pos=data[ref['positions_file']] if ref['positions_file'] else np.array(ref['positions'],np.int64)
    return [data[name] for name in ref['names']],pos,ref


def rms_contract(a):
    x=assets(a)
    return dict(definition=CONFIG['RMS'],images=[image(r) for r in x.train],extraction=x.extraction,layers=list(LAYERS),dimension=1280)


def rms_record(a):
    path=a.rms_stat or a.cache_root/'source_RMS.json'
    if not path.is_file():return None
    r=small(a,path)
    if r['contract']!=rms_contract(a) or len(r['records'])!=455:raise ValueError('RMS is not 455 unaugmented complete Source images')
    if [v['dependencies']['image'] for v in r['records']]!=r['contract']['images'] or any(
        v['patches']<=0 or v['dependencies']['definition']!=CONFIG['RMS'] or v['dependencies']['extraction']!=st.digest(r['contract']['extraction'])
        for v in r['records']):raise ValueError('RMS reductions mixed calibration/Target/augmented or different feature populations')
    from .numerics import rms_from_sums
    expected=rms_from_sums(np.sum([v['squared_sums'] for v in r['records']],axis=0),sum(v['patches'] for v in r['records']))
    if not np.array_equal(expected,np.array(r['RMS'])):raise ValueError('RMS scalar reduction differs')
    return r


def freeze(a):
    x=assets(a);routes={k:str(getattr(a,k)) for k in ('parent_run','conditional_run','raw_run','cache_root')}
    config=dict(protocol=CONFIG,asset_identity=x.identity,routes=routes,asset_runs=[str(v) for v in a.asset_runs],
        feature_roots=[str(v) for v in a.feature_roots],feature_manifest=small(a,a.feature_manifest)['seal'] if a.feature_manifest else None,
        rms_stat=small(a,a.rms_stat)['seal'] if a.rms_stat else None,
        backend=dict(distance_device=a.device,numpy=version('numpy'),scipy=version('scipy'),
            torch=version('torch') if a.device!='cpu' else None),
        code={p.name:st.sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))})
    saved=a.run_root/'inputs.json'
    if saved.is_file() and a.stage in ('evaluate','summarize','pack'):config['backend']=small(a,saved)['backend']
    config['identity']=st.digest(config)
    r=st.sealed_json(saved,config);st.sealed_json(a.run_root/'protocol.json',CONFIG)
    st.write_json(a.run_root/'job_plan.json',plan());return r


def signature(a):return small(a,a.run_root/'inputs.json')['identity']


def region_dependencies(a,row,rms,feature_seal):
    return dict(image=image(row),feature_seal=feature_seal,extraction=st.digest(assets(a).extraction),RMS=rms['seal'],
        region_definition={k:CONFIG[k] for k in ('regions','adjacency','merge','ties','mean_arithmetic','distance')},
        implementation=st.sha(Path(__file__).with_name('regions.py')))


def region_feature_ref(a,row):
    try:return feature_ref(a,row)
    except (FileNotFoundError,PermissionError):return None  # Unavailable raw input does not invalidate a sealed derived region.


def cached_region(a,row,rms,ref=None):
    """Identical preflight/resume validation; original feature files are optional.

    New regions embed their feature origin. Earlier region manifests already bind
    image/extraction/feature seal/RMS/definition; inspect those read-only without
    inventing an extraction or rewriting their identity.
    """
    path=a.cache_root/'regions'/key(row);file=path/'manifest.json'
    if not file.is_file():return None
    r=small(a,file);meta=r['metadata'];dep=meta['dependencies']
    if dep!=region_dependencies(a,row,rms,dep['feature_seal']) or meta.get('GT_used') is not False:
        raise ValueError('region actual inputs changed')
    grid=np.load(path/'grid.npy',allow_pickle=False)
    if grid.shape!=(2,) or grid.dtype.kind not in 'iu' or np.any(grid<=0):raise ValueError('cached region physical grid differs')
    grid_values=grid.tolist()
    origin=meta.get('origin')
    if origin is not None:
        if (origin.get('schema')!='region_feature_origin_v1' or origin.get('image')!=image(row)
            or origin.get('feature_seal')!=dep['feature_seal'] or origin.get('extraction')!=dep['extraction']
            or origin.get('grid')!=grid_values or origin.get('kind') not in ('existing_complete_features','original_forward_streamed')):
            raise ValueError('region feature origin differs')
    if ref is not None and (ref['seal']!=dep['feature_seal'] or ref['extraction']!=dep['extraction'] or list(ref['grid'])!=grid_values):
        raise ValueError('region actual inputs changed')
    # Only small headers/positions metadata are read; no raw features or payload hash scan.
    shapes=dict(means=(4,CONFIG['regions'],1280),labels=(int(np.prod(grid)),),
        counts=(CONFIG['regions'],),area=(CONFIG['regions'],),roots=(CONFIG['regions'],),grid=(2,))
    for name,shape in shapes.items():
        value=np.load(path/(name+'.npy'),mmap_mode='r',allow_pickle=False);spec=r['files'][name]
        if value.shape!=shape or list(value.shape)!=spec['shape'] or str(value.dtype)!=spec['dtype']:
            raise ValueError('cached region array layout differs')
    return r


def preflight(a):
    inventory=[];missing=[];work=[];x=assets(a);freeze(a)
    for role,count,path in [('E_SEGMENT_CH1',1804,a.conditional_run/'per_image_scores.csv'),
        ('F_Source66_K8_CH1',396,a.conditional_run/'normal_statistics.json'),('G_lists_supports_extraction',1,a.parent_run/'common_manifest.json')]:
        inventory.append(dict(role=role,count=count,status='local_file_exists',path=str(path),action='read_only_reuse'))
    rms=rms_record(a)
    inventory.append(dict(role='A_Source_RMS',count=455,status='local_file_exists' if rms else 'not_yet_computed',
        path=str(a.rms_stat or a.cache_root/'source_RMS.json'),action='reuse' if rms else 'stream_original_complete_Source455_once'))
    populations=[('A_Source455_features',x.train if rms is None else []),('B_Source66_features',x.calibration),
        ('C_Target48_features',[r for d in DOMAINS for s in SEEDS for r in x.supports(d,s)]),
        ('D_query902_features',[r for d in DOMAINS for r in x.samples(d)])]
    seen=set()
    for role,pop in populations:
        for row in pop:
            if row['relative_path'] in seen:continue
            seen.add(row['relative_path']);ref=region_feature_ref(a,row);known=[p.as_posix() for p,_,_ in candidates(a,row)]
            region_file=a.cache_root/'regions'/key(row)/'manifest.json'
            if region_file.is_file() and rms is not None:
                cached_region(a,row,rms,ref)
                inventory.append(dict(role=role,image=row['relative_path'],status='available_derived_regions',path=str(region_file),action='reuse_regions_no_forward'))
                continue
            record=dict(role=role,image=row['relative_path'],domain=row.get('domain','Source'),
                status='local_file_exists' if ref else 'historically_indexed_server_unconfirmed' if any(ref0 is not None for _,ref0,_ in candidates(a,row)) else 'missing_or_server_unconfirmed',
                path=ref['path'] if ref else '',candidate_paths=' | '.join(known),
                action='reuse_complete_features' if ref else 'confirm_server_paths_then_explicit_extract_only_missing')
            inventory.append(record)
            if ref is None:missing.append(record);work.append(dict(role='missing_complete_feature',image=row['relative_path']))
    for role,path in [('RGB_dataset',a.data),('checkpoint_for_explicit_missing_only',a.checkpoint),('backbone_for_explicit_missing_only',a.dinov3_root/'hubconf.py')]:
        inventory.append(dict(role=role,path=str(path),status='local_file_exists' if path.is_file() else 'not_local',action='not_required_when_complete_features_present'))
    rows=x.region_rows()
    todo_regions=sum(not (a.cache_root/'regions'/key(row)/'manifest.json').is_file() for row in rows)
    todo_pairs=sum(not (a.run_root/'pairs'/key(row)/(key(t)+'.json')).is_file() for d in DOMAINS for s in SEEDS
        for row in [*x.calibration,*x.samples(d)] for t in x.supports(d,s))
    work += [dict(role='new_Source_RMS',count=0 if rms else 455),dict(role='new_unique_region_partitions',count=todo_regions),dict(role='new_LP_jobs',count=todo_pairs),
        dict(role='Source_rebuild',count=0),dict(role='repeat_SEGMENT_or_Source_search',count=0),dict(role='new_pixel_or_AUPRO',count=0)]
    versions={}
    for name in ('numpy','scipy','Pillow','torch','torchvision','psutil'):
        try:versions[name]=version(name)
        except PackageNotFoundError:versions[name]=None
    result=dict(status='missing_inputs' if missing else 'inputs_available',missing_feature_images=len(missing),plan=plan(),
        versions_without_import=versions,torch_imported='torch' in sys.modules,
        server_scope='historical paths only; server not contacted',large_payload_hash_scans=False)
    st.write_csv(a.run_root/'asset_inventory.csv',inventory);st.write_csv(a.run_root/'missing_inputs.csv',missing)
    from collections import Counter
    counts=Counter((r['role'],r['status']) for r in inventory)
    st.write_csv(a.run_root/'asset_summary.csv',[dict(role=role,status=status,records=count) for (role,status),count in sorted(counts.items())])
    st.write_csv(a.run_root/'necessary_work.csv',work);st.write_csv(a.run_root/'numerical_verification.csv',x.verification)
    st.write_json(a.run_root/'preflight.json',result)
    if not (a.run_root/'cost_estimate.json').is_file():st.write_json(a.run_root/'cost_estimate.json',dict(
        status='unmeasured',plan=plan(),known_pending_seconds=0,unknown_is_not_zero=True,
        unknown_stages=['missing_feature_forwards','Source_RMS','region_agglomeration','distance_matrices','sparse_LP','IO_image_evaluation_pack'],
        stages={k:None for k in ('missing_feature_forwards','Source_RMS','region_agglomeration','distance_matrices','sparse_LP','IO_image_evaluation_pack')},
        over_12h_requires_consent=True))
    if not (a.run_root/'status.json').exists():st.write_json(a.run_root/'status.json',dict(status='pending',pending=['real_asset_smoke_and_formal_evaluation'],primary=PRIMARY))
    return result
