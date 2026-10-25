"""Exact existing CPU metrics and official ADEval; no redefinition of AUPRO."""
import csv
from contextlib import contextmanager
import errno
import gc
import tempfile
import time
from pathlib import Path
import numpy as np
from . import storage as st,inputs as ip,prediction as pred,calibration as cal
from .protocol import DOMAINS,BRANCHES,U,S,ENS
from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import image_metrics
from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic.numerics import check
from DINOv3.MADEqual.aebad_s_support_witness_segment_p0 import evaluation as original
from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.execution import bounded_images


def read_csv(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def restored(a,domain):
    path=a.run_root/'evaluation'/domain/'status.json'
    if not path.exists():return None
    saved=st.sealed_json(path)
    if saved['identity']!=ip.identity(a) or saved['prediction_seal']!=st.sealed_json(a.run_root/'predictions'/(domain+'.json'))['seal']:
        raise ValueError('CPU evaluation inputs changed')
    for name,sha in saved['files'].items():
        if st.sha(path.parent/name)!=sha:raise ValueError('modified CPU result: '+name)
    return saved


def sample_work(a,s,states):
    from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.fresh_inputs import Data
    from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.analysis import spatial,U as oldU,S as oldS
    x,r=st.get(pred.prediction_path(a,s));m=pred.validate_prediction(a,s,x,r,states)
    scores=[];checks=[];maps={}
    for b in BRANCHES:
        response,sc=cal.response(x[b],x['positions'],m['grid'],m['output_shape'],states[b]['image'])
        maps[b]=response
        for rule,v in sc.items():check(checks,s['sample_id'],'same_cache_'+rule,v,m['scores'][b][rule],domain=s['domain'],branch=b)
        scores.append(dict(domain=s['domain'],sample_id=s['sample_id'],label=s['label'],branch=b,**sc))
    if not all(r['verified'] for r in checks):raise ValueError('same-cache prediction score reproduction failed')
    with Data(Path(ip.catalog(a)['data_root'])) as data:gt=data.gt(s)
    mask=original.shared.resized_mask(gt,m['output_shape']).astype(np.uint8)
    rename={U:oldU,S:oldS};back={v:k for k,v in rename.items()}
    instances=spatial(s,gt,{rename.get(b,b):v for b,v in maps.items()},tuple(m['grid'])) if s['label'] else []
    for row in instances:row['branch']=back.get(row['branch'],row['branch'])
    records=[dict(domain=s['domain'],sample_id=s['sample_id'],branch=b,output_height=v.shape[0],output_width=v.shape[1],
        map_float32_sha256=original.shared.array_sha(v),mask_uint8_sha256=original.shared.array_sha(mask)) for b,v in maps.items()]
    return s,maps,mask,scores,checks,instances,records


def figure(maps,mask,sample,path):
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(16,8),layout='constrained')
    low=min(float(v.min()) for v in maps.values());high=max(float(v.max()) for v in maps.values())
    for ax,(name,value) in zip(axes.flat,list(maps.items())+[('GT',mask)]):
        ax.imshow(value,cmap='magma' if name!='GT' else 'gray',vmin=low if name!='GT' else 0,vmax=high if name!='GT' else 1)
        ax.set_title(name,fontsize=8);ax.axis('off')
    axes.flat[-1].axis('off');fig.suptitle(sample['domain']+'/'+sample['sample_id'])
    path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=100);plt.close(fig)


@contextmanager
def scratch_arrays(a,domain):
    """Exclusive per-call workspace. Never truncate or glob-delete preexisting files."""
    from .runner import check_isolation
    check_isolation(a)
    root=Path(a.evaluation_scratch).resolve();root.mkdir(parents=True,exist_ok=True)
    job=Path(tempfile.mkdtemp(prefix='normal_eval_'+domain+'_',dir=root))
    created=[];maps=[]
    def allocate(name,dtype,count):
        path=job/(name+'.bin')
        if path.parent!=job:raise ValueError('invalid scratch filename')
        with path.open('x+b') as file:
            created.append(path)  # Also owned if memmap construction fails.
            value=np.memmap(file,mode='w+',dtype=dtype,shape=(count,))
        maps.append(value)
        return value
    try:
        yield allocate
    finally:
        for value in maps:value._mmap.close()
        maps.clear()
        for path in created:path.unlink(missing_ok=True)
        try:job.rmdir()
        except OSError as error:
            # Preserve any unowned file, including one placed here during an error.
            if error.errno not in (errno.ENOTEMPTY,errno.EEXIST):raise


def evaluate_domain(a,domain):
    from .runner import check_isolation
    check_isolation(a)
    if restored(a,domain):return dict(status='CPU_reused',domain=domain)
    from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.fresh_evaluation import mark_small
    samples=[s for s in ip.catalog(a)['samples'] if s['domain']==domain];states=pred.calibration_states(a,domain)
    count=sum((s['image_height']//4)*(s['image_width']//4) for s in samples)
    path=a.run_root/'evaluation'/domain;path.mkdir(parents=True,exist_ok=True)
    arrays={};scores=[];checks=[];instances=[];records=[];offset=0;started=time.perf_counter()
    # Fixed first two normal and first two abnormal IDs per domain, never selected by outcome.
    cases={s['sample_id'] for label in (0,1) for s in [r for r in samples if r['label']==label][:2]}
    with scratch_arrays(a,domain) as allocate:
        arrays['labels']=allocate('labels',np.uint8,count)
        for b in BRANCHES:arrays[b]=allocate(b,np.float32,count)
        for i,(s,maps,mask,sc,ck,inst,mr) in enumerate(bounded_images(samples,lambda s:sample_work(a,s,states),a.cpu_workers),1):
            n=mask.size;arrays['labels'][offset:offset+n]=mask.ravel()
            for b in BRANCHES:arrays[b][offset:offset+n]=maps[b].ravel()
            offset+=n;scores.extend(sc);checks.extend(ck);instances.extend(inst);records.extend(mr)
            if s['sample_id'] in cases:figure(maps,mask,s,path/'figures'/(s['sample_id']+'.png'))
            if i%25==0 or i==len(samples):print(f'[normal CPU maps] {domain} {i}/{len(samples)}',flush=True)
        if offset!=count:raise ValueError('incomplete exact pixel population')
        labels=arrays['labels'];positive=int(labels.sum(dtype=np.int64));metrics=[]
        counts=dict(pixels=count,positive_pixels=positive,negative_pixels=count-positive)
        prediction_seal=st.sealed_json(a.run_root/'predictions'/(domain+'.json'))['seal']
        for b in BRANCHES:
            map_identity=st.digest([r for r in records if r['branch']==b])
            pixel=original.exact_pixel_result(path/(b+'_exact_pixels.json'),ip.identity(a),map_identity,labels,arrays[b],
                domain=domain,branch=b,prediction_sha=prediction_seal,counts=counts)
            local=[r for r in scores if r['branch']==b]
            for rule in ('CH1','CURRENT'):
                metrics.append(dict(domain=domain,branch=b,scoring=rule,image_count=len(local),
                    **image_metrics([r['label'] for r in local],[r[rule] for r in local]),
                    pixel_AUROC=pixel['AUROC'],pixel_AP=pixel['AP'],AUPRO=None,AUPRO_status='pending',
                    pixel_metric_origin='this_run_exact_all_pixels_CPU'))
        mark_small(instances,domain)
        tables={'metrics.csv':metrics,'per_image_scores.csv':scores,'numerical_verification.csv':checks,
            'defect_instances.csv':instances,'map_manifest.csv':records}
        for name,rows in tables.items():st.write_csv(path/name,rows)
        return st.sealed_json(path/'status.json',dict(identity=ip.identity(a),prediction_seal=prediction_seal,
            status='CPU_complete_AUPRO_pending',images=len(samples),domain=domain,seconds=time.perf_counter()-started,
            numerical_failed_checks=sum(not c['verified'] for c in checks),
            files={p.name:st.sha(p) for p in path.iterdir() if p.is_file() and p.name!='status.json'}))


def aupro(a,domain,b):
    from importlib.metadata import version
    import torch
    from DINOv3.relation_reliability.official_metrics import official_aupro_adeval
    from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.fresh_inputs import Data
    from DINOv3.MADEqual.aebad_s_fixed_target_graph_p0.evaluation import validate_aupro
    if version('adeval')!='1.1.0':raise ValueError('original ADEval1.1.0 required')
    if b not in BRANCHES or not restored(a,domain):raise ValueError('complete CPU maps required')
    rows=[r for r in read_csv(a.run_root/'evaluation'/domain/'map_manifest.csv') if r['branch']==b]
    expected=dict(identity=ip.identity(a),domain=domain,branch=b,map_identity=st.digest(rows),nstrips=200,FPR_max=.3)
    destination=a.run_root/'aupro'/domain/(b+'.json')
    if destination.exists():
        r=st.read_json(destination);validate_aupro(r,expected);return dict(status='AUPRO_reused',**r)
    states=pred.calibration_states(a,domain);samples=[s for s in ip.catalog(a)['samples'] if s['domain']==domain]
    if [r['sample_id'] for r in rows]!=[s['sample_id'] for s in samples]:raise ValueError('incomplete AUPRO population')
    maps=[];masks=[];started=time.perf_counter()
    with Data(Path(ip.catalog(a)['data_root'])) as data:
        for s,record in zip(samples,rows):
            x,r=st.get(pred.prediction_path(a,s));m=pred.validate_prediction(a,s,x,r,states)
            response,_=cal.response(x[b],x['positions'],m['grid'],m['output_shape'],states[b]['image'])
            mask=original.shared.resized_mask(data.gt(s),m['output_shape']).astype(np.uint8)
            original.shared.verify_map_record(record,response,mask);maps.append(response);masks.append(mask)
    torch.cuda.set_device(torch.device(a.device));torch.cuda.reset_peak_memory_stats()
    value=float(official_aupro_adeval(masks,maps,nstrips=200))
    r=dict(expected,AUPRO=value,verified_images=len(maps),seconds=time.perf_counter()-started,
        backend='original_ADEval_1.1.0_CUDA',peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    r['seal']=st.digest(r);validate_aupro(r,expected);st.write_json(destination,r);return r


def evaluate(a,only_aupro=False):
    pred.verify(a)  # All 926 predictions before FIRST GT decode, including AUPRO resume.
    jobs=[]
    for d in DOMAINS:
        if only_aupro:
            for b in BRANCHES:jobs.append(aupro(a,d,b))
        else:jobs.append(evaluate_domain(a,d))
    return dict(status='requested_AUPRO_complete' if only_aupro else 'CPU_complete_AUPRO_pending',jobs=jobs)
