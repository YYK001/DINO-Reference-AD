"""Independent orchestration; unchanged exact metrics, map/GT and instance kernels."""
from contextlib import contextmanager
from pathlib import Path
from types import FunctionType
import errno
import tempfile
import time
import numpy as np
from . import storage as st,inputs as ip,prediction as pred,calibration as cal
from .protocol import DOMAINS,BRANCHES,U,S
from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation import evaluation as previous
from DINOv3.MADEqual.aebad_s_support_witness_segment_p0 import evaluation as original
from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.execution import bounded_images
from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import image_metrics

read_csv=previous.read_csv


def bind(function):
    # Bind immutable math/orchestration with OUR names, never mutate old globals.
    return FunctionType(function.__code__,dict(function.__globals__,st=st,ip=ip,pred=pred,cal=cal,
        BRANCHES=BRANCHES,DOMAINS=DOMAINS,U=U,S=S,restored=restored),function.__name__,function.__defaults__)


def restored(a,domain):
    value=bind(previous.restored)(a,domain)
    if value and (value['domain']!=domain or value['images']!=len([s for s in ip.catalog(a)['samples'] if s['domain']==domain])
            or value['numerical_failed_checks']!=0):raise ValueError('incomplete CPU domain state')
    return value


def sample_work(a,s,states):return bind(previous.sample_work)(a,s,states)


@contextmanager
def scratch_arrays(a,domain):
    from .runner import check_isolation
    check_isolation(a);root=Path(a.evaluation_scratch);root.mkdir(parents=True,exist_ok=True)
    job=Path(tempfile.mkdtemp(prefix='residual_'+domain+'_',dir=root));created=[];maps=[]
    def allocate(name,dtype,count):
        p=job/(name+'.bin')
        if p.parent!=job:raise ValueError('invalid scratch filename')
        with p.open('x+b') as file:
            created.append(p);value=np.memmap(file,mode='w+',dtype=dtype,shape=(count,))
        maps.append(value);return value
    try:yield allocate
    finally:
        for x in maps:x._mmap.close()
        maps.clear()
        for p in created:p.unlink(missing_ok=True)
        try:job.rmdir()
        except OSError as e:
            if e.errno not in (errno.ENOTEMPTY,errno.EEXIST):raise


def figure(maps,mask,sample,path):
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(4,4,figsize=(16,16),layout='constrained')
    low=min(float(v.min()) for v in maps.values());high=max(float(v.max()) for v in maps.values())
    for ax in axes.flat:ax.axis('off')
    for ax,(name,value) in zip(axes.flat,[*maps.items(),('GT',mask)]):
        ax.imshow(value,cmap='gray' if name=='GT' else 'magma',vmin=0 if name=='GT' else low,vmax=1 if name=='GT' else high)
        ax.set_title(name,fontsize=7)
    fig.suptitle(sample['domain']+'/'+sample['sample_id']);path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=100);plt.close(fig)


def parent_proof(a,domain,records,branch):
    """Exact reconstructed maps AND GT, full population and sealed parent results."""
    p=a.run_root/'baseline_parent.json'
    if branch not in (U,S) or not p.exists():return None
    root=Path(st.sealed_json(p)['run']);status_path=root/'evaluation'/domain/'status.json'
    if not status_path.exists():return None
    value=st.sealed_json(status_path)
    if value['numerical_failed_checks'] or value['images']!=len(records):raise ValueError('parent CPU result incomplete')
    for name,sha in value['files'].items():
        if st.sha(status_path.parent/name)!=sha:raise ValueError('parent CPU result modified: '+name)
    old=[r for r in read_csv(status_path.parent/'map_manifest.csv') if r['branch']==branch]
    fields=('domain','sample_id','branch','output_height','output_width','map_float32_sha256','mask_uint8_sha256')
    if [[str(r[k]) for k in fields] for r in records]!=[[str(r[k]) for k in fields] for r in old]:return None
    return dict(root=str(root),status=value,records=old)


def evaluate_domain(a,domain):
    from .runner import check_isolation
    from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.fresh_evaluation import mark_small
    check_isolation(a)
    if restored(a,domain):return dict(status='CPU_reused',domain=domain)
    samples=[s for s in ip.catalog(a)['samples'] if s['domain']==domain];states=pred.calibration_states(a,domain)
    count=sum((s['image_height']//4)*(s['image_width']//4) for s in samples)
    path=a.run_root/'evaluation'/domain;path.mkdir(parents=True,exist_ok=True)
    scores=[];checks=[];instances=[];records=[];offset=0;started=time.perf_counter()
    cases={s['sample_id'] for label in (0,1) for s in [r for r in samples if r['label']==label][:2]}
    with scratch_arrays(a,domain) as allocate:
        arrays={'labels':allocate('labels',np.uint8,count)}
        arrays.update({b:allocate(b,np.float32,count) for b in BRANCHES})
        for i,(s,maps,mask,sc,ck,inst,mr) in enumerate(bounded_images(samples,lambda s:sample_work(a,s,states),a.cpu_workers),1):
            n=mask.size;arrays['labels'][offset:offset+n]=mask.ravel()
            for b in BRANCHES:arrays[b][offset:offset+n]=maps[b].ravel()
            offset+=n;scores.extend(sc);checks.extend(ck);instances.extend(inst);records.extend(mr)
            if s['sample_id'] in cases:figure(maps,mask,s,path/'figures'/(s['sample_id']+'.png'))
            if i%25==0 or i==len(samples):print(f'[residual CPU maps] {domain} {i}/{len(samples)}',flush=True)
        if offset!=count:raise ValueError('incomplete exact pixel population')
        labels=arrays['labels'];positive=int(labels.sum(dtype=np.int64));metrics=[];proofs={}
        counts=dict(pixels=count,positive_pixels=positive,negative_pixels=count-positive)
        ps=st.sealed_json(a.run_root/'predictions'/(domain+'.json'))['seal']
        for b in BRANCHES:
            local_maps=[r for r in records if r['branch']==b];proof=parent_proof(a,domain,local_maps,b)
            if proof:
                p=Path(proof['root'])/'evaluation'/domain/(b+'_exact_pixels.json')
                pixel=st.read_json(p)['result'];original.validate_exact_pixels(pixel,counts)
                origin='reused_parent_exact_map_GT_population_SHA';proofs[b]=dict(parent=proof['root'],status_seal=proof['status']['seal'])
            else:
                pixel=original.exact_pixel_result(path/(b+'_exact_pixels.json'),ip.identity(a),st.digest(local_maps),labels,arrays[b],
                    domain=domain,branch=b,prediction_sha=ps,counts=counts);origin='this_run_exact_all_pixels_CPU'
            local=[r for r in scores if r['branch']==b]
            for rule in ('CH1','CURRENT'):
                metrics.append(dict(domain=domain,branch=b,scoring=rule,image_count=len(local),
                    **image_metrics([r['label'] for r in local],[r[rule] for r in local]),pixel_AUROC=pixel['AUROC'],pixel_AP=pixel['AP'],
                    AUPRO=None,AUPRO_status='pending',pixel_metric_origin=origin))
        mark_small(instances,domain)
        for name,rows in {'metrics.csv':metrics,'per_image_scores.csv':scores,'numerical_verification.csv':checks,
                'defect_instances.csv':instances,'map_manifest.csv':records}.items():st.write_csv(path/name,rows)
        return st.sealed_json(path/'status.json',dict(identity=ip.identity(a),prediction_seal=ps,status='CPU_complete_AUPRO_pending',
            images=len(samples),domain=domain,seconds=time.perf_counter()-started,numerical_failed_checks=sum(not r['verified'] for r in checks),
            baseline_reuse=proofs,files={p.name:st.sha(p) for p in path.iterdir() if p.is_file() and p.name!='status.json'}))


def aupro(a,domain,b):
    from DINOv3.MADEqual.aebad_s_fixed_target_graph_p0.evaluation import validate_aupro
    if b not in BRANCHES or not restored(a,domain):raise ValueError('complete CPU maps required')
    rows=[r for r in read_csv(a.run_root/'evaluation'/domain/'map_manifest.csv') if r['branch']==b]
    expected=dict(identity=ip.identity(a),domain=domain,branch=b,map_identity=st.digest(rows),nstrips=200,FPR_max=.3)
    destination=a.run_root/'aupro'/domain/(b+'.json')
    if destination.exists():
        r=st.read_json(destination);validate_aupro(r,expected);return dict(status='AUPRO_reused',**r)
    proof=parent_proof(a,domain,rows,b)
    if proof:
        parent=Path(proof['root']);p=parent/'aupro'/domain/(b+'.json')
        if p.exists():
            old=st.read_json(p);validate_aupro(old,dict(identity=proof['status']['identity'],domain=domain,branch=b,
                map_identity=st.digest(proof['records']),nstrips=200,FPR_max=.3))
            value=dict(expected,AUPRO=old['AUPRO'],verified_images=len(rows),seconds=0.,backend=old['backend'],
                origin='exact_parent_map_GT_population',parent=str(p),parent_sha256=st.sha(p))
            value['seal']=st.digest(value);st.write_json(destination,value);return value
    return bind(previous.aupro)(a,domain,b)


def evaluate(a,only_aupro=False):
    pred.verify(a)  # No GT decode until every one of the 926 predictions is frozen.
    jobs=[]
    for d in DOMAINS:
        if only_aupro:
            for b in BRANCHES:jobs.append(aupro(a,d,b))
        else:jobs.append(evaluate_domain(a,d))
    return dict(status='requested_AUPRO_complete' if only_aupro else 'CPU_complete_AUPRO_pending',jobs=jobs)
