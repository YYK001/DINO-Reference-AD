"""Read-only recent SACP metadata and original raw assets; never rebuild banks."""
from pathlib import Path
from importlib.metadata import version,PackageNotFoundError
import numpy as np
from .protocol import CONFIG,LAYERS,U,S,COMBINED,job_plan
from .context import select_source
from DINOv3.MADEqual.aebad_s_reference_residual_segmentation import storage as st
from DINOv3.MADEqual.aebad_s_reference_residual_segmentation.inputs import key
from DINOv3.MADEqual.aebad_s_sacp_strength_p0.inputs import read_csv,GTData

ROOT=Path(__file__).resolve().parents[3]


class Parent:
    """Only background prediction entities are required, never SACP eta/subspaces."""
    def __init__(self,a):
        self.run=a.parent_run;self.cache=a.parent_cache
        self.inputs=st.sealed_json(self.run/'inputs.json');self.identity=self.inputs['identity']
        if any(self.inputs['config'].get(k)!=v for k,v in dict(layers=list(LAYERS),dimension=1280,Source_rows=100000,seed=42).items()):raise ValueError('frozen backbone/memory/seed differs')
        self.assets=st.sealed_json(self.run/'assets.json')
        if self.assets['seal']!=self.inputs['assets_seal']:raise ValueError('parent asset metadata differs')
        self.extraction=self.assets['references']['extraction']
        self.catalog_record=st.sealed_json(self.run/'catalog.json');self.catalog=self.catalog_record['catalog']
        if st.digest(self.catalog)!=self.inputs['catalog']:raise ValueError('parent catalog changed')
        source=self.catalog['source']
        if len(source)!=521 or sum(r['calibration'] for r in source)!=66:raise ValueError('original 521/66 split required')
        if [r['relative_path'] for r in source]!=sorted(r['relative_path'] for r in source) or any(r['source_index']!=i or r['calibration']!=(i%8==0) for i,r in enumerate(source)):raise ValueError('Source order/mod8 calibration differs')
        self.samples=[r for r in self.catalog['samples'] if r['domain']=='background'];self.supports=self.catalog['supports']['background']
        if [len(self.samples),sum(r['label']==0 for r in self.samples),sum(r['label']==1 for r in self.samples)]!=[297,85,212]:raise ValueError('original 297 evaluation images required')
        if [r['support_index'] for r in self.supports]!=list(range(1,9)) or any(r['label'] for r in self.supports):raise ValueError('ordered K8 normals required')
        if {r['relative_path'] for r in self.samples}&{r['relative_path'] for r in self.supports}:raise ValueError('support included in evaluation')
        from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.fresh_inputs import support_key
        all_normal=self.supports+[r for r in self.samples if r['label']==0]
        if [r['relative_path'] for r in self.supports]!=[r['relative_path'] for r in sorted(all_normal,key=lambda r:support_key(r['relative_path'],'background'))[:8]]:raise ValueError('original seed42 support rule differs')
        state=st.read_json(self.run/'status.json')
        if state['identity']!=self.identity or state['status']!='all_five_metrics_complete' or state['pending'] or state['numerical_failed_checks']:raise ValueError('completed parent required')
        self.mad=st.sealed_json(self.run/'input_distance_mad.json')
        if self.mad['identity']!=self.identity:raise ValueError('parent MAD identity differs')
        if any(self.mad[k]!=self.assets['references']['mad'][k] for k in ('patch','image')):raise ValueError('Source bank provider and MAD differ')
        for v in (self.mad['patch'],self.mad['image']['peak'],self.mad['image']['local']):
            if len(v['medians'])!=4 or len(v['scales'])!=4 or not np.isfinite(v['medians']).all() or not np.isfinite(v['scales']).all() or min(v['scales'])<=0:raise ValueError('invalid frozen MAD')
        self.manifest=st.sealed_json(self.run/'predictions/background.json')
        if self.manifest['identity']!=self.identity or [(r['domain'],r['sample_id']) for r in self.manifest['records']]!=[(r['domain'],r['sample_id']) for r in self.samples]:raise ValueError('incomplete parent background freeze')
        self.seals={r['sample_id']:r['seal'] for r in self.manifest['records']}
        self.cpu=st.sealed_json(self.run/'evaluation/background/status.json')
        if self.cpu['identity']!=self.identity or self.cpu['prediction_seal']!=self.manifest['seal'] or self.cpu['images']!=297 or self.cpu['numerical_failed_checks']:raise ValueError('parent CPU freeze differs')
        self._tables={};self.metrics=[r for r in read_csv(self.run/'metrics.csv') if r['domain']=='background' and r['branch'] in (U,S,COMBINED)]
        if len(self.metrics)!=6:raise ValueError('six historical score/branch rows required')
        if {(r['branch'],r['scoring']) for r in self.metrics}!={(b,s) for b in (U,S,COMBINED) for s in ('CH1','CURRENT')}:raise ValueError('historical score rows differ')
        for r in self.metrics:
            if any(not np.isfinite(float(r[k])) or not 0<=float(r[k])<=1 for k in ('image_AUROC','image_AP','pixel_AUROC','pixel_AP','AUPRO')):raise ValueError('invalid historical metric')
        cpumetrics=self.run/'evaluation/background/metrics.csv'
        if st.sha(cpumetrics)!=self.cpu['files']['metrics.csv']:raise ValueError('historical CPU metric table changed')
        lookup={(r['branch'],r['scoring']):r for r in read_csv(cpumetrics)}
        for r in self.metrics:
            for k in ('image_AUROC','image_AP','pixel_AUROC','pixel_AP'):
                b=(S if k.startswith('image_') else U) if r['branch']==COMBINED else r['branch']
                if float(r[k])!=float(lookup[b,r['scoring']][k]):raise ValueError('root/historical CPU metrics differ')
            if r['branch']==COMBINED and float(r['AUPRO'])!=float(next(x for x in self.metrics if x['branch']==U and x['scoring']==r['scoring'])['AUPRO']):raise ValueError('fixed combination must use UNION AUPRO')
        for b in (U,S):
            v=st.read_json(self.run/'aupro/background'/(b+'.json'))
            if v.get('seal')!=st.digest({k:x for k,x in v.items() if k!='seal'}) or v['identity']!=self.identity or v['verified_images']!=297 or v['backend']!='original_ADEval_1.1.0_CUDA' or v['nstrips']!=200 or v['FPR_max']!=.3:raise ValueError('parent AUPRO definition differs')
            if any(float(r['AUPRO'])!=v['AUPRO'] for r in self.metrics if r['branch']==b):raise ValueError('parent AUPRO metric mismatch')

    def table(self,name):
        if name not in self._tables:
            p=self.run/'evaluation/background'/name
            if st.sha(p)!=self.cpu['files'][name]:raise ValueError('modified parent small result table: '+name)
            self._tables[name]=read_csv(p)
        return self._tables[name]

    def prediction(self,s,load=True):
        path=self.cache/'predictions/background'/s['sample_id'];m=st.sealed_json(path/'manifest.json')
        meta=m['metadata']
        if m['seal']!=self.seals[s['sample_id']] or meta['identity']!=self.identity or meta['sample']!=s or meta['calibration_seal']!=self.mad['seal']:raise ValueError('baseline prediction identity differs')
        n=int(np.prod(meta['grid']))
        for b in (U,S):header(path,b,(4,n),np.float32)
        header(path,'raw_union',(4,n),np.float32)
        header(path,'positions',(n,),np.int64)
        if meta['output_shape']!=[s['image_height']//4,s['image_width']//4]:raise ValueError('evaluation resize differs')
        if load:
            x,_=st.get(path,(U,S,'positions'));positions(x['positions'],n);return x,m
        return path


def header(path,name,shape,dtype):
    x=np.load(Path(path)/(name+'.npy'),mmap_mode='r',allow_pickle=False)
    if x.shape!=tuple(shape) or x.dtype!=dtype:raise ValueError('wrong array header: '+str(Path(path)/name))
    return x


def positions(p,n):
    if p.dtype!=np.int64 or not np.array_equal(np.sort(p),np.arange(n)):raise ValueError('complete physical grid required')


def parent(a):
    if not hasattr(a,'_parent'):a._parent=Parent(a)
    return a._parent


def dictionary_rows(a):
    if not hasattr(a,'_dictionary_rows'):a._dictionary_rows=select_source(parent(a).catalog['source'])+parent(a).supports
    return a._dictionary_rows


def identity(a):
    # Inputs/models are immutable within a stage; validate once at stage entry.
    if not hasattr(a,'_identity'):a._identity=st.sealed_json(a.run_root/'inputs.json')['identity']
    return a._identity
def context_path(a,row):return a.cache_root/'contexts'/key(row)
def original_path(a,row):return a.cache_root/'original_features'/key(row)


def raw_reference(a,row):
    """Reference the raw preparation when available; absent original grids may be extracted later."""
    p=parent(a);rawrun=a.raw_run or Path(p.assets['references']['parent_run'])
    rawcache=a.raw_cache or Path(p.assets['references']['parent_cache'])
    if (rawrun/'prepared.json').exists():
        inputs=st.sealed_json(rawrun/'inputs.json')
        if inputs['extraction']!=p.extraction:raise ValueError('raw provider extraction configuration differs')
        prepared=st.sealed_json(rawrun/'prepared.json');ref=prepared['normal_features'].get(row['relative_path'])
        if ref:return dict(path=str(rawcache/'normal_features'/key(row)),seal=ref['seal'])
    return p.assets['normal_features'].get(row['relative_path'])


def inspect_raw(ref,row):
    path=Path(ref['path']);r=st.sealed_json(path/'manifest.json');m=r['metadata'];n=int(np.prod(m['grid']))
    if r['seal']!=ref['seal'] or m['image']!=row or m.get('kind')!='original_complete_grid':raise ValueError('non-original or mismatched normal feature')
    for l in LAYERS:header(path,f'q{l}',(n,1280),np.float32)
    positions(header(path,'positions',(n,),np.int64),n)
    return ref


def source_reference(a):
    p=parent(a);ref=p.assets['references']['source'];path=a.source_bank or Path(ref['path'])
    r=st.sealed_json(path/'manifest.json')
    if r['seal']!=ref['seal'] or r['metadata'].get('kind')!='original_source':raise ValueError('formal Source identity differs; no bank reconstruction')
    for l in LAYERS:header(path,f'L{l}',(100000,1280),np.float32)
    return dict(path=str(path),seal=r['seal'])


def preflight(a):
    result=dict(status='blocked',missing=[],errors=[],reusable=[],new_work=[],unconfirmed=[],config=CONFIG,job_plan=job_plan())
    try:
        p=parent(a);rows=dictionary_rows(a)
        try:result['source_bank']=source_reference(a);result['reusable'].append('formal original Source bank and frozen Source patch/CH1 MAD')
        except (OSError,ValueError,KeyError) as e:result['missing'].append(dict(asset='formal Source bank',path=str(a.source_bank or p.assets['references']['source']['path']),reason=str(e),policy='no automatic rebuild'))
        raw=[];fresh=[]
        for row in rows:
            ref=raw_reference(a,row)
            if ref and (Path(ref['path'])/'manifest.json').exists():
                inspect_raw(ref,row);raw.append(dict(image=row['relative_path'],**ref))
            else:fresh.append(row['relative_path'])
        result['reusable_normal_features']=raw;result['original_forwards_needed']=fresh
        reusable_queries=[];query_forwards=[]
        for s in p.samples:
            if (a.cache_root/'predictions/background'/s['sample_id']/'manifest.json').exists():continue
            ref=p.assets['queries'].get('background/'+s['sample_id'])
            if ref and (Path(ref['path'])/'manifest.json').exists():reusable_queries.append(dict(sample_id=s['sample_id'],**ref))
            else:query_forwards.append(dict(sample_id=s['sample_id'],relative_path=s['relative_path']))
        result['reusable_original_queries']=reusable_queries;result['query_original_forwards_needed']=query_forwards
        missing_predictions=[]
        for s in p.samples:
            try:p.prediction(s,False)
            except FileNotFoundError as e:missing_predictions.append(dict(asset='parent UNION/SEGMENT prediction',sample_id=s['sample_id'],path=str(e.filename or p.cache/'predictions/background'/s['sample_id'])))
        result['missing'].extend(missing_predictions)
        # Metadata only; no RGB or GT decoding, no weight/content SHA scan.
        from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.fresh_inputs import Data
        if not a.data.exists():result['missing'].append(dict(asset='AeBAD RGB+GT',path=str(a.data)))
        else:
            with Data(a.data) as data:
                import zipfile
                names=set(data.zip.namelist()) if data.zip else None
                for row in rows+p.samples:
                    rel=row['relative_path']
                    exists=('AeBAD/AeBAD_S/'+rel in names) if names is not None else (data.root/rel).is_file()
                    if not exists:result['missing'].append(dict(asset='RGB',path=rel))
                for rel in GTData.missing(data,p.samples):result['missing'].append(dict(asset='GT',path=rel))
        for name,path in (('checkpoint',a.checkpoint),('official host',a.official_root/'config.json'),('DINOv3',a.dinov3_root/'hubconf.py')):
            if not path.exists():result['missing'].append(dict(asset=name,path=str(path),reason='new masked extraction always required'))
        result['dictionary_images']=[dict(r,role='Target' if 'support_index' in r else 'Source') for r in rows]
        result['new_work']=['72 x4 masked extractions for dictionary; four layers per extraction',
                            'whole-image-excluded training candidates; 1500 updates total',
                            '297 x4 masked query extractions; original query forwards if absent',
                            'one new exact CPU pixel evaluation and one original ADEval AUPRO']
        result['unconfirmed']=['real extraction throughput, input isolation and CUDA numerical behavior',
                               'local missing remote paths do not establish server asset loss']
        result['reusable'].append('background baseline metrics, ordered K8, 455/66 split, U/S maps and instance definitions')
    except (OSError,ValueError,KeyError) as e:
        result['errors'].append(type(e).__name__+': '+str(e))
        if isinstance(e,OSError):result['missing'].append(dict(asset='required parent metadata/MAD',path=str(e.filename or a.parent_run),reason=str(e),policy='no automatic reconstruction'))
    deps={}
    for name in ('numpy','scipy','Pillow','scikit-learn','psutil','torch','torchvision','timm','adeval'):
        try:deps[name]=version(name)
        except PackageNotFoundError:deps[name]=None
    result.update(dependencies=deps,torch_imported='torch' in __import__('sys').modules,GPU_initialized=False,full_payload_hash_scan=False)
    if hasattr(a,'_parent'):
        result['extraction_software_compatibility']={n:dict(expected=v,installed=deps.get(n),matches=deps.get(n)==v) for n,v in a._parent.extraction['software'].items()}
        result['runtime_environment_note']='masked extraction refuses incompatible software; no automatic installation; original ADEval1.1.0 required for AUPRO'
    result['status']='blocked' if result['missing'] or result['errors'] else 'inventory_ready_runtime_isolation_pending'
    return result


def freeze(a):
    p=parent(a);inventory=preflight(a)
    if inventory['status']=='blocked':raise ValueError('missing inputs; run preflight; no fallback bank/MAD build')
    refs=source_reference(a);rows=dictionary_rows(a)
    record=dict(config=CONFIG,job_plan=job_plan(),parent_identity=p.identity,parent_inputs_seal=p.inputs['seal'],
                parent_background_seal=p.manifest['seal'],source=refs,mad_seal=p.mad['seal'],extraction=p.extraction,
                dictionary_images=rows,evaluation=p.samples,supports=p.supports,
                cache=str(a.cache_root),data=str(a.data),parent_run=str(a.parent_run),parent_cache=str(a.parent_cache),
                code={f.name:st.sha(f) for f in Path(__file__).parent.glob('*.py')})
    record['identity']=st.digest(record);st.sealed_json(a.run_root/'inputs.json',record)
    st.sealed_json(a.run_root/'protocol.json',CONFIG);st.sealed_json(a.run_root/'job_plan.json',job_plan())
    st.write_csv(a.run_root/'training_images.csv',inventory['dictionary_images']);st.write_csv(a.run_root/'evaluation_images.csv',p.samples)
    st.write_json(a.run_root/'asset_inventory.json',inventory)
    return record


def validate(a):
    r=st.sealed_json(a.run_root/'inputs.json');p=parent(a)
    if r['config']!=CONFIG or r['parent_identity']!=p.identity or r['source']!=source_reference(a) or r['mad_seal']!=p.mad['seal'] or r['cache']!=str(a.cache_root) or r['data']!=str(a.data):raise ValueError('frozen inputs changed')
    if r['parent_background_seal']!=p.manifest['seal'] or r['dictionary_images']!=dictionary_rows(a):raise ValueError('population changed')
    for name,value in r['code'].items():
        if st.sha(Path(__file__).with_name(name))!=value:raise ValueError('implementation changed since preparation')
    return r
