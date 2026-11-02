"""Seed42-only parent adapter and metadata inventory. Never imports Torch."""
import csv
import os
import sys
from functools import lru_cache
from importlib.metadata import version,PackageNotFoundError
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .protocol import *
from . import numerics as nm
from DINOv3.MADEqual.aebad_s_reference_residual_segmentation import storage as st
from DINOv3.MADEqual.aebad_s_sacp_strength_p0.inputs import GTData
from DINOv3.MADEqual.aebad_s_joint_reference_response_p0 import inputs as previous

ROOT=Path(__file__).resolve().parents[3]


def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def image(row):
    return {k:row[k] for k in ('relative_path','image_sha256','image_height','image_width')}


class Parent:
    """Read common membership of both seeds; only seed42 scientific products.

    No seed43 feature, normal-statistic, weight or prediction file is opened.
    Source u weights are read as frozen constants; no Target-LOO is executed.
    """
    def __init__(self,path):
        self.root=Path(path);self.inputs=st.sealed_json(self.root/'inputs.json')
        self.common=st.sealed_json(self.root/'common_manifest.json')
        if self.inputs['Source_seed']!=42 or self.inputs['common_manifest']!=self.common['seal'] or self.common['total_common_images']!=902:
            raise ValueError('original Source42/common902 protocol required')
        expected=dict(K=8,Source_rows=100000,dimension=1280,layers=list(LAYERS),Source_seed=42,Source_calibration=66)
        if any(self.inputs['protocol'].get(k)!=v for k,v in expected.items()):raise ValueError('parent backbone/memory/calibration protocol differs')
        if self.inputs['identity']!=st.digest({k:v for k,v in self.inputs.items() if k not in ('identity','seal')}):raise ValueError('parent input identity differs')
        if len(self.inputs['Source_calibration'])!=66:raise ValueError('original Source66 required')
        common_csv=rows(self.root/'common_evaluation_manifest.csv')
        support_csv=rows(self.root/'support_manifest.csv');excluded=rows(self.root/'excluded_images.csv')
        summary=rows(self.root/'summary/domain_seed_metrics.csv')
        if len(common_csv)!=902:raise ValueError('common CSV population differs')
        expected_common=[r for d in DOMAINS for r in self.common['domains'][d]['common']]
        if common_csv!=[{k:str(r.get(k,'')) for k in common_csv[0]} for r in expected_common]:raise ValueError('CSV/JSON common image identity/order differs')
        excluded_paths={r['relative_path'] for r in excluded}
        self.weights={};self.records={};self.maps={};self.metrics={};self.freezes={}
        for d in DOMAINS:
            group=self.common['domains'][d];samples=group['common'];supports=group['supports']['42']
            if (len(samples),sum(r['label']==0 for r in samples),sum(r['label']==1 for r in samples))!=COUNTS[d]:raise ValueError('common domain population differs')
            if len({r['sample_id'] for r in samples})!=len(samples) or len(supports)!=8 or any(r['label'] or r['domain']!=d for r in supports):raise ValueError('unique samples/ordered normal K8 required')
            if {r['relative_path'] for r in samples}&excluded_paths:raise ValueError('excluded support returned to evaluation')
            excluded_expected={r['relative_path'] for r in group['supports']['42']+group['supports']['43']}
            if excluded_expected!={r['relative_path'] for r in excluded if r['domain']==d}:raise ValueError('support-union exclusion CSV differs')
            sc=[r for r in support_csv if r['domain']==d and r['support_seed']=='42']
            if [r['relative_path'] for r in sc]!=[r['relative_path'] for r in supports] or [r['group_index'] for r in sc]!=list(map(str,range(1,9))):raise ValueError('seed42 support order differs')
            for r,s in zip(sc,supports):
                if any(str(s[k])!=r[k] for k in ('sample_id','image_sha256','image_height','image_width','label')):raise ValueError('support CSV identity differs')
            unit=self.root/d/'seed42';status=st.read_json(unit/'status.json')
            if status['status']!='common_five_metrics_complete' or status['pending']:raise ValueError('completed seed42 parent required')
            w=st.sealed_json(unit/'weights.json');self.weights[d]=w['weights'][SOURCE]
            # Validate u/p/q normalization through the original map adapter.
            nm.weighted(np.zeros((4,9),np.float32),np.arange(9,dtype=np.int64),[3,3],[3,3],self.weights[d])
            if d!=DOMAINS[0] and self.weights[d]!=self.weights[DOMAINS[0]]:raise ValueError('shared Source weights differ across domains')
            freeze=st.sealed_json(unit/'predictions_frozen.json');self.freezes[d]=freeze
            if [r['sample_id'] for r in freeze['records']]!=[r['sample_id'] for r in samples] or freeze['weights_seal']!=w['seal']:raise ValueError('seed42 prediction list differs')
            for row,ref in zip(samples,freeze['records']):
                r=st.sealed_json(unit/'predictions'/(row['sample_id']+'.json'))
                if r['seal']!=ref['seal'] or r['image']!=row or r['support_seed']!=42 or r['weights_seal']!=w['seal']:raise ValueError('parent seed42 record differs')
                self.records[d,row['sample_id']]=r
            cpu=st.sealed_json(unit/'evaluation/status.json')
            if cpu['prediction_seal']!=freeze['seal'] or cpu['images']!=len(samples):raise ValueError('parent CPU identity differs')
            for name in ('map_manifest.csv','metrics.csv'):
                if st.sha(unit/'evaluation'/name)!=cpu['files'][name]:raise ValueError('modified parent small CPU table')
            self.maps[d]={(r['sample_id'],r['branch']):r for r in rows(unit/'evaluation/map_manifest.csv')}
            from DINOv3.MADEqual.aebad_s_target_loo_support_retest_p0.reporting import final_metrics
            actual,pending=final_metrics(SimpleNamespace(run_root=self.root,domain=d,support_seed=42))
            if pending:raise ValueError('parent baseline AUPRO pending')
            self.metrics[d]=[r for r in actual if r['branch'] in (U,S,'SEGMENT_SCORE_UNION_MAP',SOURCE,'SEGMENT_SCORE_SOURCE_WEIGHT_MAP')]
            bykey={(r['branch'],r['scoring']):r for r in self.metrics[d]}
            selected=[r for r in summary if r['domain']==d and r['support_seed']=='42' and (r['branch'],r['scoring']) in bykey]
            if len(selected)!=len(bykey) or any(float(r[k])!=float(bykey[r['branch'],r['scoring']][k]) for r in selected for k in METRICS):raise ValueError('parent summary metric differs')
        for state in (self.inputs['mad']['patch'],self.inputs['mad']['image']['peak'],self.inputs['mad']['image']['local']):
            if len(state['medians'])!=4 or len(state['scales'])!=4 or not np.isfinite(state['medians']).all() or not np.isfinite(state['scales']).all() or min(state['scales'])<=0:raise ValueError('invalid original MAD')
        self.identity=st.digest([self.inputs['seal'],self.common['seal'],[self.freezes[d]['seal'] for d in DOMAINS]])

    def samples(self,d):return self.common['domains'][d]['common']
    def supports(self,d):return self.common['domains'][d]['supports']['42']
    def record(self,d,row):return self.records[d,row['sample_id']]


def parent(a):
    if not hasattr(a,'_joint_parent'):a._joint_parent=Parent(a.parent_run)
    return a._joint_parent


@lru_cache(maxsize=None)
def package_version(name):
    try:return version(name)
    except PackageNotFoundError:return None


def prediction_backend(a):
    """Planned arithmetic only; metadata lookup never imports Torch/CUDA."""
    cuda=a.device.startswith('cuda:')
    return dict(version='prediction_backend_v1',engine='stock_torch_cuda' if cuda else 'numpy_cpu',
                device=a.device,visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES') if cuda else None,
                dtype='float32',numpy=package_version('numpy'),torch=package_version('torch'),
                autocast=False,matmul_precision='highest',matmul_allow_tf32=False)


def frozen_backend(a,*,compute=False):
    saved=st.sealed_json(a.run_root/'inputs.json').get('prediction_backend')
    if saved is None:raise ValueError('run lacks prediction backend binding; use a NEW run/cache, never rewrite old identities')
    if compute and saved!=prediction_backend(a):
        raise ValueError('prediction backend differs from frozen run; use a NEW run/cache and new backend smoke')
    return saved


def signature(a,d):
    return dict(identity=st.sealed_json(a.run_root/'inputs.json')['identity'],domain=d,support_seed=42,
                prediction_backend=frozen_backend(a))
def pred_path(a,d,row):return a.cache_root/'predictions'/d/row['sample_id']
def record_path(a,d,row):return a.run_root/d/'predictions'/(row['sample_id']+'.json')


def baseline(a,d,row):
    p=parent(a);old=p.record(d,row)['baseline_origin']
    if old['kind']!='read_only_seed42_parent':raise ValueError('unexpected baseline provider')
    root=a.memory42_root or Path(p.inputs['parent_cache'])/'predictions'
    path=root/d/row['sample_id'];r=st.sealed_json(path/'manifest.json')
    if r['seal']!=old['manifest_seal'] or r['metadata']['sample']!=row or r['metadata']['calibration_seal']!=p.inputs['mad']['seal']:raise ValueError('baseline image/MAD/manifest differs')
    x,r=st.get(path,(U,S,'positions'));nm.order(x['positions'],r['metadata']['grid'])
    n=len(x['positions'])
    if any(x[k].shape!=(4,n) or x[k].dtype!=np.float32 or not np.isfinite(x[k]).all() for k in (U,S)):raise ValueError('baseline layer layout differs')
    if r['metadata']['output_shape']!=[row['image_height']//4,row['image_width']//4]:raise ValueError('baseline output geometry differs')
    return x,r,dict(path=str(path),seal=r['seal'])


def feature_ref(a,row):
    # Thin original full-feature adapter; explicitly remove the seed43 lookup.
    a.memory43_root=None
    return previous.feature_ref(a,row)


def source_ref(a):
    p=parent(a);path=a.source_bank or Path(p.inputs['source']['path'])
    r=st.sealed_json(path/'manifest.json')
    if r['seal']!=p.inputs['source']['seal']:raise ValueError('original augmented Source bank required; no rebuild/replacement')
    for l in LAYERS:
        spec=r['files']['L'+str(l)]
        if spec['shape']!=[100000,1280] or spec['dtype']!='float32' or not (path/f'L{l}.npy').exists():raise ValueError('full original Source100K FP32 required')
    return dict(path=str(path),seal=r['seal'])


def contract(a,d,row,feature,refs):
    p=parent(a)
    return dict(image=image(row),extraction=p.inputs['extraction'],query=feature,
                source=p.inputs['source']['seal'],supports=[image(s) for s in p.supports(d)],
                reference_features=refs,layers=list(LAYERS),dimension=1280,
                exact='original_Exact_FP32_Top1_divide1280_once',support_seed=42)


def nn_path(a,d,row):
    return a.cache_root/'nearest'/d/row['sample_id']


def nn_ref(a,d,row,wanted):
    path=nn_path(a,d,row);expected=None
    if not (path/'manifest.json').exists() and a.nn_provider:
        provider=st.sealed_json(a.nn_provider);ref=provider['records'].get(d+'/'+row['sample_id'])
        if ref:path=Path(ref['path']);expected=ref['seal']
    if not (path/'manifest.json').exists():return None
    r=st.sealed_json(path/'manifest.json')
    if (expected and r['seal']!=expected) or r['metadata']['contract']!=wanted:raise ValueError('nearest index lacks same query/Source/support/extraction provenance')
    keys=['positions']+[f'{k}{l}' for l in LAYERS for k in ('distance','index')]
    x,_=st.get(path,keys);n=len(x['positions']);nm.order(x['positions'],r['metadata']['grid'])
    for l in LAYERS:
        if x[f'distance{l}'].shape!=(9,n) or x[f'index{l}'].shape!=(9,n) or x[f'distance{l}'].dtype!=np.float32 or x[f'index{l}'].dtype!=np.int64:raise ValueError('nine-source distance/index layout differs')
    return dict(path=str(path),seal=r['seal'])


def freeze(a):
    p=parent(a)
    backend=prediction_backend(a)
    if (a.run_root/'inputs.json').exists():
        # CPU evaluation and CUDA AUPRO replay the already frozen maps. Their
        # execution device does not change the producer's arithmetic identity.
        backend=frozen_backend(a,compute=a.stage not in ('evaluate','aupro','summarize'))
    value=dict(config=CONFIG,parent_identity=p.identity,parent_inputs=p.inputs['seal'],common_manifest=p.common['seal'],
               source=p.inputs['source'],extraction=p.inputs['extraction'],mad=p.inputs['mad'],weights=p.weights,
               supports={d:p.supports(d) for d in DOMAINS},
               routes={k:str(getattr(a,k)) if getattr(a,k,None) else None for k in
                       ('parent_run','sacp_run','raw_cache','memory42_root','source_bank','support_feature_root','feature_manifest','nn_provider','data')},
               cache_root=str(a.cache_root),prediction_backend=backend,
               numerical_code={n:st.sha(Path(__file__).with_name(n)) for n in ('protocol.py','numerics.py','runtime.py','prediction.py')})
    if a.feature_manifest:value['feature_provider']=st.sealed_json(a.feature_manifest)['seal']
    if a.nn_provider:value['nn_provider']=st.sealed_json(a.nn_provider)['seal']
    value['identity']=st.digest(value)
    r=st.sealed_json(a.run_root/'inputs.json',value);st.sealed_json(a.run_root/'protocol.json',CONFIG)
    st.write_json(a.run_root/'job_plan.json',job_plan());return r


def preflight(a):
    available=[];missing=[];unknown=[];work=[]
    try:
        p=parent(a);freeze(a)
        available.append(dict(role='frozen_seed42_metadata_scores_metrics_MAD_Source_weights',path=str(a.parent_run),images=902))
        try:bank=source_ref(a);available.append(dict(role='Source_memory_vectors',**bank))
        except FileNotFoundError:
            bank=None;missing.append(dict(role='Source_memory_vectors',path=str(a.source_bank or Path(p.inputs['source']['path'])),action='restore exact augmented bank; no automatic rebuild'))
        for d in DOMAINS:
            refs=[]
            for row in p.supports(d):
                ref=feature_ref(a,row);refs.append(ref)
                (available if ref else missing).append(dict(domain=d,sample_id=row['sample_id'],role='seed42_Target_memory_full_features',path=ref['path'] if ref else row['relative_path']))
                if not ref:work.append(dict(domain=d,sample_id=row['sample_id'],role='support_forward',count=1,reason='restore complete support features first, otherwise explicit original forward once'))
            for row in p.samples(d):
                ref=feature_ref(a,row)
                (available if ref else missing).append(dict(domain=d,sample_id=row['sample_id'],role='complete_query_features',path=ref['path'] if ref else row['relative_path']))
                if not ref:work.append(dict(domain=d,sample_id=row['sample_id'],role='query_forward',count=1,reason='full query vectors absent; maximum once after explicit authorization'))
                try:
                    _,_,br=baseline(a,d,row);available.append(dict(domain=d,sample_id=row['sample_id'],role='scalar_baseline_only',**br))
                except FileNotFoundError as e:missing.append(dict(domain=d,sample_id=row['sample_id'],role='frozen_UNION_SEGMENT_layers',path=str(e),action='restore; do not silently reevaluate or stitch'))
                nn=None
                if ref and bank and all(refs):nn=nn_ref(a,d,row,contract(a,d,row,ref,[dict(path=v['path'],seal=v['seal']) for v in refs]))
                if nn:available.append(dict(domain=d,sample_id=row['sample_id'],role='reliable_nine_source_NN',**nn))
                else:
                    missing.append(dict(domain=d,sample_id=row['sample_id'],role='reliable_nine_source_NN',path=str(nn_path(a,d,row)),action='36 original Exact searches shared by all branches'))
                    work.append(dict(domain=d,sample_id=row['sample_id'],role='Exact_retrieval',count=36,reason='scalar distances or Source/aggregate Target index cannot recover nine anchors'))
        try:
            with GTData(a.data) as data:
                absent=data.missing([r for d in DOMAINS for r in p.samples(d)])
                missing.extend(dict(role='GT',path=v) for v in absent)
            if not absent:available.append(dict(role='GT_path_index',path=str(a.data),decoded=False))
        except (OSError,ValueError) as e:missing.append(dict(role='GT_dataset',path=str(a.data),reason=str(e)))
        if not (a.sacp_run/'assets.json').exists():unknown.append(dict(role='server_feature_authority',path=str(a.sacp_run/'assets.json'),status='server_confirmation_required'))
        result=dict(status='missing_inputs' if missing else 'inputs_available',plan=job_plan(),missing_records=len(missing),
                    missing_query_forwards=sum(r['count'] for r in work if r['role']=='query_forward'),
                    necessary_Exact_searches=sum(r['count'] for r in work if r['role']=='Exact_retrieval'),torch_imported='torch' in sys.modules)
    except (OSError,ValueError,KeyError) as e:
        result=dict(status='blocked',error=str(e),torch_imported='torch' in sys.modules);missing.append(dict(role='input_metadata',reason=str(e)))
    st.write_csv(a.run_root/'asset_inventory.csv',available+unknown);st.write_csv(a.run_root/'missing_inputs.csv',missing)
    st.write_csv(a.run_root/'necessary_repetition.csv',work);st.write_json(a.run_root/'preflight.json',result)
    if not (a.run_root/'time_estimate.json').exists():
        st.write_json(a.run_root/'time_estimate.json',dict(status='unmeasured_pending_authorized_full_image_smoke',total_seconds=None,
            stages={k:None for k in ('feature_IO_or_forward','original_Exact','vector_gather_translation_and_write','CPU_exact','AUPRO','report_pack')},
            cached_case='full query vectors + identical bank/supports + reliable indices: no model/Exact',
            missing_case='maximum 902 query forwards + 32472 Exact calls; support forwards separately counted if needed',
            RAM_GiB_planning_range=[6,18],VRAM_GiB_planning_range=[6,20],
            new_compact_cache_GiB_planning_range=[8,11],CPU_scratch_GiB_planning_range=[3,12],
            optional_persisted_full_query_features='about 239.6 GiB for 902 x 13924 x 1280 x 4 x FP32; disabled by default',
            resource_ranges_are_unverified_not_admission_checks=True,over_12h_requires_explicit_consent=True))
    return result
