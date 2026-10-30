"""Read-only recent metadata and scalar response bundles, never model assets."""
from pathlib import Path
from types import SimpleNamespace
import shutil
import numpy as np
from DINOv3.MADEqual.aebad_s_target_loo_support_retest_p0 import inputs as old,reporting as parent_reporting
from DINOv3.MADEqual.aebad_s_sacp_strength_p0.inputs import GTData
from .protocol import *
st=old.st
ROOT=old.ROOT


class Parent:
    def __init__(self,path):
        self.root=Path(path);self.inputs=st.sealed_json(self.root/'inputs.json');self.common=st.sealed_json(self.root/'common_manifest.json')
        if self.inputs['identity']!=st.digest({k:v for k,v in self.inputs.items() if k not in ('identity','seal')}):raise ValueError('parent dependency identity differs')
        if self.inputs['protocol']!=old.CONFIG or self.inputs['Source_seed']!=42 or self.inputs['common_manifest']!=self.common['seal']:raise ValueError('not the frozen two-K8 parent')
        status=st.read_json(self.root/'summary/status.json')
        if status['status']!='two_K8_common_validation_complete' or status['pending'] or status['complete_AUPRO_jobs']!=24 or status.get('numerical_failed_checks')!=0:raise ValueError('completed two-K8 parent required')
        self.settings={};self.records={};self.weights={};self.normal={};self.maps={};source=None;source_stats=None;self.support_overlaps={}
        for d in DOMAINS:
            group=self.common['domains'][d];samples=group['common'];support_sets=[{r['relative_path'] for r in group['supports'][str(s)]} for s in SEEDS]
            if (len(samples),sum(r['label']==0 for r in samples),sum(r['label']==1 for r in samples))!=COUNTS[d] or len({r['sample_id'] for r in samples})!=len(samples):raise ValueError('902 common population differs')
            if any(len(s)!=8 for s in support_sets) or set(r['relative_path'] for r in samples)&set.union(*support_sets):raise ValueError('support union included in evaluation')
            self.support_overlaps[d]=len(support_sets[0]&support_sets[1])
            for memory in SEEDS:
                a=SimpleNamespace(run_root=self.root,domain=d,support_seed=memory);metrics,w=parent_reporting.read_complete(a);path=self.root/d/f'seed{memory}'
                if source is None:source=w['weights'][SOURCE]
                if source!=w['weights'][SOURCE]:raise ValueError('shared Source weights changed')
                self.weights[d,f'W{memory}']=w['weights'][TARGET];self.weights[d,'WS']=source
                normal=st.sealed_json(path/'normal_statistics.json');self.normal[d,memory]=normal
                s=[r for r in normal['records'] if r['kind']=='Source'];t=[r for r in normal['records'] if r['kind']=='Target']
                if normal['seal']!=w['normal_statistics_seal'] or len(s)!=66 or len(t)!=8 or [r['image'] for r in t]!=[r['relative_path'] for r in group['supports'][str(memory)]]:raise ValueError('frozen normal-statistic inputs differ')
                if source_stats is None:source_stats=s
                if s!=source_stats:raise ValueError('Source66 statistics differ across settings')
                self.settings[d,memory]=dict(weights=w,metrics=metrics,normal=normal,weight_rows=parent_reporting.rows(path/'weights.csv'))
                weight_rows=self.settings[d,memory]['weight_rows']
                if len(weight_rows)!=24 or {(r['branch'],int(r['layer']),r['statistic']) for r in weight_rows}!={(b,l,k) for b in (SOURCE,TARGET) for l in LAYERS for k in ('u','p','q')}:raise ValueError('frozen normal-scale/weight table incomplete')
                from DINOv3.MADEqual.aebad_s_target_loo_layer_weight_p0.numerics import moments
                _,ss,tt=moments([r['statistics'] for r in s],[r['statistics'] for r in t])
                for item in self.settings[d,memory]['weight_rows']:
                    j=LAYERS.index(int(item['layer']));k=('u','p','q').index(item['statistic']);b=item['branch']
                    if b not in (SOURCE,TARGET) or float(item['weight'])!=w['weights'][b][j][k] or float(item['Source_scale'])!=float(np.sqrt(ss[j,k])) or float(item['Target_LOO_scale'])!=float(np.sqrt(tt[j,k])):raise ValueError('normal-scale/weight CSV differs from frozen statistics')
                    mixed=max(float(np.sqrt(ss[j,k] if b==SOURCE else .5*ss[j,k]+.5*tt[j,k])),1e-8)
                    if float(item['mixed_scale'])!=mixed:raise ValueError('frozen mixed-scale record differs')
                frozen=st.sealed_json(path/'predictions_frozen.json');records=[]
                for sample,ref in zip(samples,frozen['records']):
                    r=st.sealed_json(path/'predictions'/(sample['sample_id']+'.json'))
                    if r['seal']!=ref['seal'] or r['image']!=sample or r['weights_seal']!=w['seal']:raise ValueError('parent per-image record differs')
                    self.records[d,memory,sample['sample_id']]=r;records.append(r['seal'])
                self.maps[d,memory]={(r['sample_id'],r['branch']):r for r in parent_reporting.rows(path/'evaluation/map_manifest.csv')}
        self.identity=st.digest(dict(parent_inputs=self.inputs['seal'],common=self.common['seal'],weights={f'{d}/{s}':r['weights']['seal'] for (d,s),r in self.settings.items()}))

    def samples(self,d):return self.common['domains'][d]['common']

    def record(self,d,m,row):return self.records[d,m,row['sample_id']]

    def original(self,d,m,row):
        origin=self.record(d,m,row)['baseline_origin']
        if m==42:
            if origin['kind']!='read_only_seed42_parent':raise ValueError('unexpected seed42 response origin')
            return dict(path=str(Path(self.inputs['parent_cache'])/'predictions'/d/row['sample_id']),seal=origin['manifest_seal'])
        if origin['kind'] not in ('new_original_Exact_same_group','own_compact_baseline','compatible_legacy_reference'):raise ValueError('unknown seed43 compact response origin')
        return dict(path=origin['path'],seal=origin['seal'])

    def known_metrics(self,d,m,w):
        b=SOURCE if w=='WS' else TARGET
        if w not in ('WS',diagonal(m)):raise ValueError('crossed metric cannot be imported')
        return [dict(r) for r in self.settings[d,m]['metrics'] if r['branch']==b]


def parent(a):
    if not hasattr(a,'_parent'):a._parent=Parent(a.parent_run)
    return a._parent


def paths(a,d,m,row):
    p=parent(a);origin=p.original(d,m,row)
    candidates=[]
    if a.response_root:candidates.append(a.response_root/d/f'memory{m}'/row['sample_id'])
    base=a.memory42_root if m==42 else a.memory43_root
    if base:candidates.append(base/d/row['sample_id'] if m==42 else base/d/'seed43/baselines'/row['sample_id'])
    candidates.append(Path(origin['path']))
    return list(dict.fromkeys(candidates)),origin


def inspect(a,d,m,row):
    candidates,origin=paths(a,d,m,row)
    path=next((x for x in candidates if (x/'manifest.json').is_file()),None)
    if path is None:raise FileNotFoundError('; '.join(str(x) for x in candidates))
    r=st.sealed_json(path/'manifest.json');meta=r['metadata'];p=parent(a)
    if r['seal']!=origin['seal']:raise ValueError('scalar response manifest differs from original prediction')
    if m==42:
        if meta['sample']!=row or meta['calibration_seal']!=p.inputs['mad']['seal']:raise ValueError('seed42 image/MAD differs')
    else:
        dep=meta['dependencies'];expected=dict(domain=d,support_seed=m,Source_seed=42,source=p.inputs['source']['seal'],supports=p.common['domains'][d]['supports'][str(m)],extraction=p.inputs['extraction'],patch=p.inputs['mad']['patch'],image=row)
        if any(dep.get(k)!=v for k,v in expected.items()):raise ValueError('seed43 Source/support/feature/MAD dependencies differ')
    grid=meta['grid'];n=int(np.prod(grid));shape=[row['image_height']//4,row['image_width']//4]
    if len(grid)!=2 or min(grid)<3 or meta['output_shape']!=shape:raise ValueError('original geometry/valid3x3 differs')
    old.layer_inputs.old.header(path,U,(4,n),np.float32)
    pos=old.layer_inputs.old.header(path,'positions',(n,),np.int64);old.layer_inputs.old.positions(pos,n)
    return dict(path=str(path),seal=r['seal'],origin=origin,grid=grid,output_shape=shape,scalar_bytes=r['files'][U]['bytes']+r['files']['positions']['bytes'])


def load(a,d,m,row):
    ref=inspect(a,d,m,row);x,r=st.get(ref['path'],(U,'positions'))
    if r['seal']!=ref['seal'] or not np.isfinite(x[U]).all():raise ValueError('nonfinite/changed UNION z')
    return x[U],x['positions'],ref


def unit(a,d,m):return a.output_root/d/f'memory{m}'
def record_path(a,d,m,row):return unit(a,d,m)/'predictions'/(row['sample_id']+'.json')


def freeze(a):
    p=parent(a);plan=[]
    for d in DOMAINS:
        for m in SEEDS:
            for w in WEIGHTS:plan.append(dict(domain=d,memory_seed=m,weight_source=w,weight_seed=None if w=='WS' else int(w[1:]),new_map=w==crossed(m),diagnostic_only=w==crossed(m),Target_normal_information_images=8 if w in ('WS',diagonal(m)) else 16-p.support_overlaps[d],shared_Source_statistics_images=66,weight_matrix=p.weights[d,w]))
    value=dict(protocol=CONFIG,parent_identity=p.identity,parent_inputs_seal=p.inputs['seal'],common_manifest_seal=p.common['seal'],plan=plan,
        response_root=str(a.response_root) if a.response_root else None,memory42_root=str(a.memory42_root) if a.memory42_root else None,memory43_root=str(a.memory43_root) if a.memory43_root else None,data=str(a.data))
    value['identity']=st.digest(value);r=st.sealed_json(a.output_root/'inputs.json',value);st.sealed_json(a.output_root/'protocol.json',CONFIG);st.write_csv(a.output_root/'cross_plan.csv',plan)
    return r


def signature(a,d,m):
    r=st.sealed_json(a.output_root/'inputs.json')
    if r['protocol']!=CONFIG or r['parent_identity']!=parent(a).identity:raise ValueError('audit parent/config changed')
    return dict(identity=r['identity'],domain=d,memory_seed=m,weight_source=crossed(m),weight_seed=85-m,diagnostic_only=True)


def preflight(a):
    available=[];missing=[];required=[]
    try:
        p=parent(a);freeze(a)
        for d in DOMAINS:
            for m in SEEDS:
                available.append(dict(domain=d,memory_seed=m,role='frozen_weights_diagonal_metrics_and_scores',images=len(p.samples(d)),path=str(p.root/d/f'seed{m}')))
                for row in p.samples(d):
                    candidates,origin=paths(a,d,m,row)
                    entry=dict(domain=d,memory_seed=m,sample_id=row['sample_id'],role='standardized_UNION_four_layers_and_positions',original_path=origin['path'],expected_manifest_seal=origin['seal'],files='manifest.json;UNION_K8.npy;positions.npy')
                    required.append(entry)
                    try:ref=inspect(a,d,m,row);available.append(dict(entry,resolved_path=ref['path'],bytes=ref['scalar_bytes']))
                    except FileNotFoundError as e:missing.append(dict(entry,searched_paths=str(e),recovery='copy scalar responses only; no feature extraction or Exact authorized'))
        try:
            with GTData(a.data) as data:
                missing.extend(dict(role='GT',relative_path=s) for s in data.missing(sum([p.samples(d) for d in DOMAINS],[])))
                available.append(dict(role='GT_index',path=str(a.data),decoded=False))
        except (OSError,ValueError) as e:missing.append(dict(role='GT_dataset',path=str(a.data),reason=str(e)))
        result=dict(status='missing_inputs' if missing else 'inputs_available',unique_images=sum(len(p.samples(d)) for d in DOMAINS),memory_image_records=len(required),new_cross_maps=6,new_AUPRO=6,known_metrics_reused_cells=12,missing_records=len(missing),torch_imported='torch' in __import__('sys').modules)
    except (OSError,ValueError,KeyError) as e:result=dict(status='blocked',error=str(e),torch_imported='torch' in __import__('sys').modules)
    a.output_root.mkdir(parents=True,exist_ok=True)
    st.write_csv(a.output_root/'asset_inventory.csv',available);st.write_csv(a.output_root/'missing_inputs.csv',missing);st.write_csv(a.output_root/'required_response_files.csv',required);st.write_json(a.output_root/'preflight.json',result)
    historical_AUPRO=[st.read_json(parent(a).root/d/f'seed{m}'/'aupro'/(b+'.json'))['seconds'] for d in DOMAINS for m in SEEDS for b in (SOURCE,TARGET)] if result['status']!='blocked' else []
    estimate=dict(status='pending_fixed_sample_smoke',full_seconds=None,new_feature_forwards=0,new_Exact_searches=0,new_weight_fit_calls=0,new_training=0,
        work_plan=dict(response_image_records=1804,new_cross_maps=6,new_formal_CPU_jobs=6,new_AUPRO_jobs=6,old_full_pixel_AUPRO_jobs_recomputed=0,pair_comparison_units=18,common_images=902),
        stages=dict(read_reweight_and_freeze_seconds=None,normal_tails_and_instances_seconds=None,pair_and_layer_analysis_seconds=None,exact_CPU_seconds=None,AUPRO_new_cross_seconds=None),
        previous_same_population_AUPRO_job_seconds=historical_AUPRO,previous_AUPRO_is_reference_not_new_job_benchmark=True,
        response_transfer_GiB_planning_range=[.5,1.],response_bytes_confirmed=sum(r.get('bytes',0) for r in available),new_output_GiB_planning_range=[.05,.5],evaluation_scratch_GiB_planning_range=[1,8],
        conditions='requires scalar responses/positions and GT; full CPU and AUPRO estimate after no-GT smoke, prediction freeze and GT smoke; no fixed few-minute promise',over_12h_requires_approval=True)
    if not (a.output_root/'time_estimate.json').exists():st.write_json(a.output_root/'time_estimate.json',estimate)
    return result


def export_responses(a):
    p=parent(a);records=[]
    if not a.export_root:raise ValueError('--export-root required')
    for d in DOMAINS:
        for m in SEEDS:
            for row in p.samples(d):
                ref=inspect(a,d,m,row);source=Path(ref['path']);dest=a.export_root/d/f'memory{m}'/row['sample_id']
                if dest.resolve()==source.resolve() or dest.resolve().is_relative_to(source.resolve()) or source.resolve().is_relative_to(dest.resolve()):raise ValueError('export must be separate from original response')
                dest.mkdir(parents=True,exist_ok=True)
                for name in ('manifest.json',U+'.npy','positions.npy'):
                    target=dest/name
                    if target.exists():
                        if target.stat().st_size!=(source/name).stat().st_size:raise ValueError('existing export differs; choose new directory')
                    else:shutil.copyfile(source/name,target)
                if st.sealed_json(dest/'manifest.json')['seal']!=ref['seal']:raise ValueError('existing exported manifest differs')
                n=int(np.prod(ref['grid']));old.layer_inputs.old.header(dest,U,(4,n),np.float32);old.layer_inputs.old.positions(old.layer_inputs.old.header(dest,'positions',(n,),np.int64),n)
                records.append(dict(domain=d,memory_seed=m,sample_id=row['sample_id'],source=ref,relative=dest.relative_to(a.export_root).as_posix()))
    return st.sealed_json(a.export_root/'response_export.json',dict(parent_identity=p.identity,records=records,original_manifests_unchanged=True,only_standardized_scalar_responses=True))
