"""Adapters to original exact CPU/ADEval/spatial kernels, after full freeze."""
from contextlib import contextmanager
from pathlib import Path
from types import FunctionType,SimpleNamespace
import tempfile,time,errno
import numpy as np
from . import inputs as ip,prediction as pred
from .inputs import st
from .protocol import DOMAINS,LAYERS,BRANCHES,U,S,MAIN,TARGET,SOURCE
from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation import evaluation as previous
from DINOv3.MADEqual.aebad_s_reference_residual_segmentation import calibration as oldcal
from DINOv3.MADEqual.aebad_s_support_witness_segment_p0 import evaluation as original
from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.execution import bounded_images
from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import image_metrics
read_csv=previous.read_csv


def bind(fn):
    return FunctionType(fn.__code__,dict(fn.__globals__,ip=ip,pred=pred,st=st,cal=SimpleNamespace(response=oldcal.response),
        BRANCHES=BRANCHES,DOMAINS=DOMAINS,U=U,S=S,restored=restored,original=original),fn.__name__,fn.__defaults__)


def restored(a,domain):
    path=a.run_root/'evaluation'/domain/'status.json'
    if not path.exists():return None
    r=st.sealed_json(path)
    if r['identity']!=ip.identity(a) or r['prediction_seal']!=st.sealed_json(a.run_root/'predictions'/(domain+'.json'))['seal']:raise ValueError('CPU state inputs changed')
    if r['images']!=sum(s['domain']==domain for s in ip.catalog(a)['samples']) or r['numerical_failed_checks']:raise ValueError('incomplete CPU result')
    verify_tables(path.parent,r['files'])
    return r


def verify_tables(root,files):
    """Small result tables only; never walk/hash feature or bank payloads."""
    for name,expected in files.items():
        path=root/name
        if st.sha(path)!=expected:raise ValueError('completed result table changed: '+str(path))


@contextmanager
def scratch(a,domain):
    from .runner import isolation
    isolation(a);a.evaluation_scratch.mkdir(parents=True,exist_ok=True)
    job=Path(tempfile.mkdtemp(prefix=domain+'_',dir=a.evaluation_scratch));created=[];maps=[]
    def allocate(name,dtype,count):
        p=job/(name+'.bin')
        if p.parent!=job:raise ValueError('invalid scratch name')
        with p.open('x+b') as f:
            created.append(p);x=np.memmap(f,mode='w+',dtype=dtype,shape=(count,))
        maps.append(x);return x
    try:yield allocate
    finally:
        for x in maps:x._mmap.close()
        for p in created:p.unlink(missing_ok=True)
        try:job.rmdir()
        except OSError as e:
            if e.errno not in (errno.ENOTEMPTY,errno.EEXIST):raise


def parent_proof(a,domain,records,branch):
    if branch not in (U,S):return None
    root=Path(ip.assets(a)['references']['parent_run']);path=root/'evaluation'/domain/'status.json'
    if not path.exists():return None
    state=st.sealed_json(path)
    if state['identity']!=ip.assets(a)['references']['parent_identity']:raise ValueError('parent evaluation identity differs')
    if state['images']!=len(records) or state['numerical_failed_checks']:return None
    needed=('map_manifest.csv',branch+'_exact_pixels.json')
    if any(name not in state['files'] for name in needed):return None
    verify_tables(path.parent,{name:state['files'][name] for name in needed})
    old=[r for r in read_csv(path.parent/'map_manifest.csv') if r['branch']==branch]
    fields=('domain','sample_id','branch','output_height','output_width','map_float32_sha256','mask_uint8_sha256')
    if len(old)!=len(records):return None
    if [[str(r[k]) for k in fields] for r in records]!=[[str(r[k]) for k in fields] for r in old]:raise ValueError('parent baseline map/GT/population differs; no silent historical replacement')
    return dict(root=str(root),state=state,records=old)


def sample_work(a,s,states):
    from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic.numerics import adaptive_max
    from DINOv3.MADEqual.aebad_s_union_diagnostics.spatial import components,instance_regions
    from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.analysis import spatial
    from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic.numerics import check
    from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.fresh_inputs import Data
    x,r=st.get(pred.prediction_path(a,s));m=pred.validate_prediction(a,s,x,r,states);maps={};scores=[];checks=list(m.get('checks',[]))
    for b in BRANCHES:
        response,sc=oldcal.response(x[b],x['positions'],m['grid'],m['output_shape'],states[b]['image']);maps[b]=response
        for rule,v in sc.items():check(checks,s['sample_id'],'same_cache_'+rule,v,m['scores'][b][rule],domain=s['domain'],branch=b)
        scores.append(dict(domain=s['domain'],sample_id=s['sample_id'],label=s['label'],branch=b,**sc))
    if not all(r['verified'] for r in checks):raise ValueError('same-cache scoring reproduction failed')
    with Data(a.data) as data:gt=data.gt(s)
    mask=original.shared.resized_mask(gt,m['output_shape']).astype(np.uint8)
    instances=FunctionType(spatial.__code__,dict(spatial.__globals__,U=U,S=S),spatial.__name__)(s,gt,maps,tuple(m['grid'])) if s['label'] else []
    records=[dict(domain=s['domain'],sample_id=s['sample_id'],branch=b,output_height=v.shape[0],output_width=v.shape[1],
        map_float32_sha256=original.shared.array_sha(v),mask_uint8_sha256=original.shared.array_sha(mask)) for b,v in maps.items()]
    value=(s,maps,mask,scores,checks,instances,records)
    # Actual final maps determine margins; eta is supplementary descriptive data.
    changes=[];eta_rows=[]
    regions={'normal_all_pixels':np.ones_like(mask,bool)} if not s['label'] else {'anomaly_GT_pixels':mask.astype(bool),'anomaly_normal_pixels':~mask.astype(bool)}
    for b in BRANCHES:
        delta=maps[U].astype(np.float64)-maps[b]
        for group,region in regions.items():changes.append(dict(domain=s['domain'],sample_id=s['sample_id'],branch=b,reference=U,group=group,pixels=int(region.sum()),
            mean_signed_response_reduction=float(delta[region].mean()) if region.any() else None,units='fused_original_Source_patch_MAD_response'))
    if s['label']:
        native_regions={'GT_cover':adaptive_max(gt,tuple(m['grid']))};labelled,descriptors=components(gt);ring=np.zeros(tuple(m['grid']),bool)
        for obj in descriptors:ring|=instance_regions(labelled==obj['instance_id'],gt,tuple(m['grid']))[1]
        native_regions['ring']=ring
    else:
        n=len(x['positions']);order=np.lexsort((x['positions'],-np.mean(x[U],axis=0,dtype=np.float32)))[:max(1,int(np.ceil(.01*n)))];region=np.zeros(n,bool);region[x['positions'][order]]=True
        native_regions={'normal_hard_top1pct':region.reshape(m['grid'])}
    for j,l in enumerate(LAYERS):
        for bi,b in enumerate((MAIN,TARGET,SOURCE)):
            for name,region in native_regions.items():
                values=x['eta'][bi,j][region.ravel()[x['positions']]]
                eta_rows.append(dict(domain=s['domain'],sample_id=s['sample_id'],layer=l,branch=b,region=name,patches=len(values),
                    mean=float(np.mean(values)) if len(values) else None,median=float(np.median(values)) if len(values) else None,
                    p95=float(np.quantile(values,.95)) if len(values) else None,status='available' if len(values) else 'NA_empty_mapped_region'))
    return (*value,changes,eta_rows)


def evaluate_domain(a,domain):
    if restored(a,domain):return dict(status='CPU_reused',domain=domain)
    from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.fresh_evaluation import mark_small
    path=a.run_root/'evaluation'/domain;path.mkdir(parents=True,exist_ok=True);samples=[s for s in ip.catalog(a)['samples'] if s['domain']==domain]
    count=sum((s['image_height']//4)*(s['image_width']//4) for s in samples);states=pred.calibration_states(a,domain)
    scores=[];checks=[];instances=[];records=[];changes=[];etas=[];offset=0;started=time.perf_counter();proofs={}
    with scratch(a,domain) as allocate:
        labels=allocate('labels',np.uint8,count);arrays={b:allocate(b,np.float32,count) for b in BRANCHES}
        for i,value in enumerate(bounded_images(samples,lambda s:sample_work(a,s,states),a.cpu_workers),1):
            s,maps,mask,sc,ck,inst,mr,change,eta=value;n=mask.size;labels[offset:offset+n]=mask.ravel()
            for b in BRANCHES:arrays[b][offset:offset+n]=maps[b].ravel()
            offset+=n;scores.extend(sc);checks.extend(ck);instances.extend(inst);records.extend(mr);changes.extend(change);etas.extend(eta)
            if i%25==0 or i==len(samples):print(f'[SACP CPU maps] {domain} {i}/{len(samples)}',flush=True)
        if offset!=count:raise ValueError('incomplete pixel population')
        metrics=[];positive=int(labels.sum(dtype=np.int64));counts=dict(pixels=count,positive_pixels=positive,negative_pixels=count-positive)
        ps=st.sealed_json(a.run_root/'predictions'/(domain+'.json'))['seal']
        for b in BRANCHES:
            local=[r for r in records if r['branch']==b];proof=parent_proof(a,domain,local,b)
            if proof:
                p=Path(proof['root'])/'evaluation'/domain/(b+'_exact_pixels.json');pixel=st.read_json(p)['result'];original.validate_exact_pixels(pixel,counts)
                proofs[b]=proof['root'];origin='parent_exact_map_GT_and_population_identity'
            else:
                pixel=original.exact_pixel_result(path/(b+'_exact_pixels.json'),ip.identity(a),st.digest(local),labels,arrays[b],domain=domain,branch=b,prediction_sha=ps,counts=counts);origin='this_run_exact_all_pixels_CPU'
            rows=[r for r in scores if r['branch']==b]
            for rule in ('CH1','CURRENT'):metrics.append(dict(domain=domain,branch=b,scoring=rule,image_count=len(rows),
                **image_metrics([r['label'] for r in rows],[r[rule] for r in rows]),pixel_AUROC=pixel['AUROC'],pixel_AP=pixel['AP'],AUPRO=None,pixel_metric_origin=origin))
    mark_small(instances,domain)
    files={'metrics.csv':metrics,'per_image_scores.csv':scores,'numerical_verification.csv':checks,'defect_instances.csv':instances,
        'map_manifest.csv':records,'response_changes.csv':changes,'eta_regions.csv':etas}
    for name,rows in files.items():st.write_csv(path/name,rows)
    return st.sealed_json(path/'status.json',dict(identity=ip.identity(a),prediction_seal=ps,domain=domain,images=len(samples),
        numerical_failed_checks=sum(not r['verified'] for r in checks),seconds=time.perf_counter()-started,baseline_reuse=proofs,files={name:st.sha(path/name) for name in files}))


def aupro(a,domain,b):
    from DINOv3.MADEqual.aebad_s_fixed_target_graph_p0.evaluation import validate_aupro
    if not restored(a,domain):raise ValueError('CPU evaluation must complete first')
    rows=[r for r in read_csv(a.run_root/'evaluation'/domain/'map_manifest.csv') if r['branch']==b]
    expected=dict(identity=ip.identity(a),domain=domain,branch=b,map_identity=st.digest(rows),nstrips=200,FPR_max=.3)
    dest=a.run_root/'aupro'/domain/(b+'.json')
    if dest.exists():r=st.read_json(dest);validate_aupro(r,expected);return dict(r,runtime_reused=True)
    proof=parent_proof(a,domain,rows,b)
    if proof:
        p=Path(proof['root'])/'aupro'/domain/(b+'.json')
        if p.exists():
            old=st.read_json(p);validate_aupro(old,dict(expected,identity=proof['state']['identity'],map_identity=st.digest(proof['records'])))
            value=dict(expected,AUPRO=old['AUPRO'],verified_images=len(rows),seconds=0.,backend=old['backend'],origin='parent_exact_map_GT_population',parent=str(p));value['seal']=st.digest(value)
            st.write_json(dest,value);return dict(value,runtime_reused=True)
    return dict(bind(previous.aupro)(a,domain,b),runtime_reused=False)


def evaluate(a,only_aupro=False):
    pred.verify(a)
    return [aupro(a,d,b) for d in DOMAINS for b in BRANCHES] if only_aupro else [evaluate_domain(a,d) for d in DOMAINS]
