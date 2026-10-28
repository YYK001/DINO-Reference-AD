"""Relocatable frozen prediction input; no checkpoint, Q, bank or fit dependency."""
from pathlib import Path
from importlib.metadata import version,PackageNotFoundError
import csv,zipfile
import numpy as np
from .protocol import CONFIG,DOMAINS,LAYERS,U,S,HALF_MAIN,HALF_TARGET,HALF_UNIFORM,ETA_AXES,PARENT_PREDICTION_SHA,PARENT_CONFIG,HISTORY,job_plan
from DINOv3.MADEqual.aebad_s_reference_residual_segmentation import storage as st
from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.inputs import Data as OriginalData,safe

ROOT=Path(__file__).resolve().parents[3]
FIELDS=('raw_union','raw_main','eta','positions',U,S,HALF_MAIN,HALF_TARGET,HALF_UNIFORM)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


class GTData:
    """Reuse the original GT reader/byte identity; GT-only roots are allowed."""
    bytes=OriginalData.bytes
    gt=OriginalData.gt
    def __init__(self,path):
        path=Path(path);self.zip=zipfile.ZipFile(path) if path.is_file() else None
        if self.zip:self.root=None
        else:
            options=[path,path.parent if path.name=='ground_truth' else path/'AeBAD_S',path/'AeBAD/AeBAD_S']
            self.root=next((p for p in options if (p/'ground_truth').is_dir()),None)
            if self.root is None:raise FileNotFoundError('GT root must contain ground_truth: '+str(path))
    def __enter__(self):return self
    def __exit__(self,*args):
        if self.zip:self.zip.close()
    def missing(self,samples):
        names=set(self.zip.namelist()) if self.zip else None;result=[]
        for s in samples:
            if not s['label']:continue
            rel=safe(s['relative_path'].replace('test/','ground_truth/',1))
            if ('AeBAD/AeBAD_S/'+rel not in names) if self.zip else not (self.root/rel).is_file():result.append(rel)
        return result


class Parent:
    def __init__(self,a):
        self.run=a.parent_run;self.cache=a.parent_cache
        if self.run is None or self.cache is None:raise ValueError('explicit --parent-run and --parent-cache are required')
        self.inputs=st.sealed_json(self.run/'inputs.json');self.identity=self.inputs['identity']
        protocol=st.sealed_json(self.run/'protocol.json');self.protocol={k:v for k,v in protocol.items() if k!='seal'}
        if self.protocol!=self.inputs['config'] or self.protocol!={**PARENT_CONFIG,'populations':CONFIG['populations']}:raise ValueError('parent is not the frozen four-layer lambda=.5 SACP')
        if self.inputs['code'].get('prediction.py')!=PARENT_PREDICTION_SHA:raise ValueError('unverified parent eta-axis schema; no identity rewriting')
        self.catalog_record=st.sealed_json(self.run/'catalog.json');self.catalog=self.catalog_record['catalog']
        if self.inputs['catalog']!=st.digest(self.catalog):raise ValueError('parent catalog changed')
        self.mad=st.sealed_json(self.run/'input_distance_mad.json')
        if self.mad['identity']!=self.identity:raise ValueError('parent calibration identity differs')
        for state in (self.mad['patch'],self.mad['image']['peak'],self.mad['image']['local']):
            if len(state['medians'])!=4 or len(state['scales'])!=4 or not np.isfinite(state['medians']).all() or not np.isfinite(state['scales']).all() or min(state['scales'])<=0:raise ValueError('invalid frozen patch/image MAD')
        assets=st.sealed_json(self.run/'assets.json')
        if assets['seal']!=self.inputs['assets_seal']:raise ValueError('parent metadata seal differs')
        self.software=assets['references']['extraction']['software']  # Metadata only: never follow the bank/Q paths.
        self.status=st.read_json(self.run/'status.json')
        if self.status['identity']!=self.identity or self.status['status']!='all_five_metrics_complete' or self.status['pending'] or self.status['numerical_failed_checks']:raise ValueError('parent results incomplete')
        self.manifests={};self.records={};self._tables={};self.history_files={}
        for d in DOMAINS:
            samples=[s for s in self.catalog['samples'] if s['domain']==d]
            if [len(samples),sum(s['label']==0 for s in samples),sum(s['label']==1 for s in samples)]!=CONFIG['populations'][d]:raise ValueError('full parent population differs')
            support=self.catalog['supports'][d]
            if len(support)!=8 or [s['support_index'] for s in support]!=list(range(1,9)) or any(s['label'] for s in support) or {s['relative_path'] for s in samples}&{s['relative_path'] for s in support}:raise ValueError('ordered K8/support exclusion differs')
            m=st.sealed_json(self.run/'predictions'/(d+'.json'));self.manifests[d]=m
            if m['identity']!=self.identity or [(r['domain'],r['sample_id']) for r in m['records']]!=[(d,s['sample_id']) for s in samples] or len({s['sample_id'] for s in samples})!=len(samples):raise ValueError('parent prediction freeze incomplete')
            self.records.update({(d,r['sample_id']):r['seal'] for r in m['records']})
            state=st.sealed_json(self.run/'evaluation'/d/'status.json')
            if state['identity']!=self.identity or state['prediction_seal']!=m['seal'] or state['images']!=len(samples) or state['numerical_failed_checks']:raise ValueError('parent CPU results incomplete')
            for name in ('per_image_scores.csv','defect_instances.csv','map_manifest.csv'):
                path=self.run/'evaluation'/d/name
                if st.sha(path)!=state['files'][name]:raise ValueError('parent small result table changed: '+str(path))
                self.history_files['evaluation/'+d+'/'+name]=state['files'][name]
        frozen=st.sealed_json(self.run/'predictions_frozen.json')
        if frozen['identity']!=self.identity or frozen['images']!=len(self.catalog['samples']) or frozen['seal_of_manifests']!=st.digest([self.manifests[d]['seal'] for d in DOMAINS]) or frozen['GT_used']:raise ValueError('parent global prediction freeze differs')
        self.metrics=read_csv(self.run/'metrics.csv');self.history_files['metrics.csv']=st.sha(self.run/'metrics.csv')
        for name in ('mechanism_summary.csv','margin_summary.csv'):self.history_files[name]=st.sha(self.run/name)
        expected_metrics={(d,b,s) for d in (*DOMAINS,'macro_3domains') for b in HISTORY for s in ('CH1','CURRENT')}
        if len(self.metrics)!=len(expected_metrics) or {(r['domain'],r['branch'],r['scoring']) for r in self.metrics}!=expected_metrics:raise ValueError('parent metric population incomplete')
        for r in self.metrics:
            if any(not np.isfinite(float(r[k])) or not 0<=float(r[k])<=1 for k in ('image_AUROC','image_AP','pixel_AUROC','pixel_AP','AUPRO')):raise ValueError('invalid parent metric')
        index={(r['domain'],r['branch'],r['scoring']):r for r in self.metrics}
        for d in DOMAINS:
            path=self.run/'evaluation'/d/'metrics.csv';state=st.sealed_json(path.parent/'status.json')
            if st.sha(path)!=state['files']['metrics.csv']:raise ValueError('parent metric table changed')
            self.history_files['evaluation/'+d+'/metrics.csv']=state['files']['metrics.csv']
            for r in read_csv(path):
                if any(float(r[k])!=float(index[d,r['branch'],r['scoring']][k]) for k in ('image_AUROC','image_AP','pixel_AUROC','pixel_AP')):raise ValueError('parent aggregate/domain metrics differ')
            for b in HISTORY:
                if b==HISTORY[-1]:continue  # Fixed combination has no separate AUPRO job.
                v=st.read_json(self.run/'aupro'/d/(b+'.json'))
                if v['seal']!=st.digest({k:x for k,x in v.items() if k!='seal'}) or v['identity']!=self.identity or v['verified_images']!=CONFIG['populations'][d][0] or v['backend']!='original_ADEval_1.1.0_CUDA' or v['nstrips']!=200 or v['FPR_max']!=.3:raise ValueError('parent AUPRO metadata incomplete')
                if any(float(index[d,b,s]['AUPRO'])!=v['AUPRO'] for s in ('CH1','CURRENT')):raise ValueError('parent AUPRO/metrics differ')

    def table(self,name,instance=False):
        if name not in self._tables:
            values={}
            for d in DOMAINS:
                for r in read_csv(self.run/'evaluation'/d/name):
                    key=(d,r['sample_id'],r['instance_id'],r['branch']) if instance else (d,r['sample_id'],r['branch'])
                    if key in values:raise ValueError('duplicate parent table identity')
                    values[key]=r
            self._tables[name]=values
        return self._tables[name]
    @property
    def scores(self):return self.table('per_image_scores.csv')
    @property
    def maps(self):return self.table('map_manifest.csv')
    @property
    def instances(self):return self.table('defect_instances.csv',True)
    @property
    def mechanisms(self):return read_csv(self.run/'mechanism_summary.csv')

    def path(self,s):return self.cache/'predictions'/safe(s['domain'])/safe(s['sample_id'])
    def inspect(self,s,positions=True):
        path=self.path(s);r=st.sealed_json(path/'manifest.json');m=r['metadata'];n=int(np.prod(m['grid']))
        if r['seal']!=self.records[s['domain'],s['sample_id']] or m['identity']!=self.identity or m['sample']!=s or m['calibration_seal']!=self.mad['seal']:raise ValueError('parent prediction/sample/calibration identity differs: '+str(path))
        if len(m['grid'])!=2 or min(m['grid'])<1 or m['output_shape']!=[s['image_height']//4,s['image_width']//4]:raise ValueError('parent map geometry differs')
        for name in FIELDS:
            spec=r['files'][name];shape=(n,) if name=='positions' else (3,4,n) if name=='eta' else (4,n);dtype='int64' if name=='positions' else 'float32'
            x=np.load(path/(name+'.npy'),mmap_mode='r',allow_pickle=False)
            if spec['shape']!=list(shape) or spec['dtype']!=dtype or x.shape!=shape or str(x.dtype)!=dtype:raise ValueError('parent required array header differs: '+str(path/name))
        if positions:
            pos=np.load(path/'positions.npy',mmap_mode='r',allow_pickle=False)
            if not np.array_equal(np.sort(pos),np.arange(n)):raise ValueError('parent positions incomplete or duplicated')
        return r
    def load(self,s,keys=FIELDS):
        r=self.inspect(s,positions=False);x,_=st.get(self.path(s),keys)
        if not np.array_equal(np.sort(x['positions']),np.arange(len(x['positions']))):raise ValueError('parent positions changed')
        return x,r


def parent(a):
    if not hasattr(a,'_parent'):a._parent=Parent(a)
    return a._parent
def identity(a):return st.sealed_json(a.run_root/'inputs.json')['identity']
def catalog(a):return st.sealed_json(a.run_root/'catalog.json')['catalog']


def preflight(a):
    missing=[];errors=[];required=[]
    try:
        p=parent(a)
        for s in p.catalog['samples']:
            path=p.path(s);required.append(dict(domain=s['domain'],sample_id=s['sample_id'],manifest=str(path/'manifest.json'),arrays=';'.join(name+'.npy' for name in FIELDS)))
            if not (path/'manifest.json').is_file():missing.append(dict(domain=s['domain'],sample_id=s['sample_id'],asset='parent_prediction_bundle',path=str(path/'manifest.json'),required_arrays=required[-1]['arrays']));continue
            for name in FIELDS:
                if not (path/(name+'.npy')).is_file():missing.append(dict(domain=s['domain'],sample_id=s['sample_id'],asset=name,path=str(path/(name+'.npy'))))
            if all((path/(name+'.npy')).is_file() for name in FIELDS):p.inspect(s)
        with GTData(a.data) as data:
            missing.extend(dict(asset='GT',path=rel) for rel in data.missing(p.catalog['samples']))
        expected={n:v for n,v in p.software.items() if n in ('numpy','scipy','Pillow')}
    except (OSError,ValueError,KeyError) as e:
        errors.append(str(e));expected={}
        if isinstance(e,OSError):missing.append(dict(asset='parent_metadata_or_GT',path=str(e.filename or a.data)))
    deps={}
    for n in ('numpy','scipy','Pillow','scikit-learn','psutil','torch','adeval'):
        try:deps[n]=version(n)
        except PackageNotFoundError:deps[n]=None
    mismatch={n:dict(expected=v,installed=deps[n]) for n,v in expected.items() if v!=deps[n]}
    if mismatch:errors.append('frozen map software differs; use parent environment, no automatic installation')
    absent=[n for n in ('numpy','scipy','Pillow','scikit-learn','psutil') if deps[n] is None]
    result=dict(status='blocked' if missing or errors or absent else 'inventory_ready_not_runtime_verified',missing=missing,errors=errors,missing_CPU_dependencies=absent,
        required_cache_bundles=len(required),required_cache_files=required,eta_axes=list(ETA_AXES),consumed_eta_axes=[0,1],config=CONFIG,job_plan=job_plan(),
        dependency_versions=deps,map_software_mismatch=mismatch,AUPRO_backend_available=deps['adeval']=='1.1.0' and deps['torch'] is not None,
        torch_imported='torch' in __import__('sys').modules,full_payload_hash_scan=False,forwards_or_searches_or_fit_allowed=False,
        audit='small metadata identities; required NPY headers/positions; eta finite/range and half replay checked when consumed; no bulk payload SHA',
        filesystem_scope='current machine only; missing remote absolute paths locally do not prove server assets missing')
    return result


def freeze(a):
    p=parent(a);code={f.name:st.sha(f) for f in Path(__file__).parent.glob('*.py')}
    record=dict(config=CONFIG,job_plan=job_plan(),parent_identity=p.identity,parent_inputs_seal=p.inputs['seal'],parent_protocol_seal=st.sealed_json(p.run/'protocol.json')['seal'],parent_catalog_seal=p.catalog_record['seal'],
        parent_mad_seal=p.mad['seal'],parent_prediction_manifests={d:r['seal'] for d,r in p.manifests.items()},historical_files=p.history_files,
        parent_run=str(p.run),parent_cache=str(p.cache),cache=str(a.cache_root),data=str(a.data),code=code)
    record['identity']=st.digest(record);st.sealed_json(a.run_root/'inputs.json',record)
    st.sealed_json(a.run_root/'protocol.json',CONFIG);st.sealed_json(a.run_root/'job_plan.json',job_plan())
    st.sealed_json(a.run_root/'parent_protocol.json',dict(protocol=p.protocol,parent_identity=p.identity,parent_protocol_seal=record['parent_protocol_seal']))
    st.sealed_json(a.run_root/'catalog.json',dict(catalog=dict(p.catalog,data_root=str(a.data))))
    st.sealed_json(a.run_root/'input_distance_mad.json',dict(identity=record['identity'],patch=p.mad['patch'],image=p.mad['image'],scope=p.mad['scope'],parent_identity=p.identity,parent_seal=p.mad['seal']))
    return record


def validate(a):
    r=st.sealed_json(a.run_root/'inputs.json');p=parent(a)
    if (r['config']!=CONFIG or r['job_plan']!=job_plan() or r['parent_identity']!=p.identity or r['parent_inputs_seal']!=p.inputs['seal'] or r['parent_catalog_seal']!=p.catalog_record['seal']
            or r['parent_mad_seal']!=p.mad['seal'] or r['historical_files']!=p.history_files or r['cache']!=str(a.cache_root) or r['data']!=str(a.data)
            or r['parent_run']!=str(p.run) or r['parent_cache']!=str(p.cache)):raise ValueError('extension/parent configuration changed')
    if r['parent_prediction_manifests']!={d:v['seal'] for d,v in p.manifests.items()}:raise ValueError('parent prediction manifests changed')
    for name,h in r['code'].items():
        if st.sha(Path(__file__).with_name(name))!=h:raise ValueError('strength implementation changed since replay')
    if any(version(n)!=p.software[n] for n in ('numpy','scipy','Pillow')):raise ValueError('frozen map software changed')
    return r
