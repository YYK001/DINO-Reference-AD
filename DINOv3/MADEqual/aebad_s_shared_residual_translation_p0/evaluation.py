"""Nine exact CPU and nine original ADEval jobs; three readouts share each map."""
from contextlib import ExitStack,contextmanager
from pathlib import Path
import tempfile
import time
import numpy as np
from . import inputs as ip,prediction as pred
from .inputs import st,GTData
from .protocol import *
from DINOv3.MADEqual.aebad_s_support_witness_segment_p0 import evaluation as backend
from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import image_metrics
from DINOv3.MADEqual.aebad_s_fixed_target_graph_p0.evaluation import validate_aupro


def destination(a,d,b):return a.run_root/d/'evaluation'/b


def completed(a,d,b):
    path=destination(a,d,b)/'status.json'
    if not path.exists():return None
    r=st.sealed_json(path)
    if any(r.get(k)!=v for k,v in ip.signature(a,d).items()) or r['branch']!=b or r['prediction_seal']!=pred.global_freeze(a)['seal'] or r['images']!=len(ip.parent(a).samples(d)):
        raise ValueError('completed CPU identity differs')
    for name,expected in r['files'].items():
        if st.sha(path.parent/name)!=expected:raise ValueError('modified small CPU result: '+name)
    # The original intermediate seal prevents re-enrolling edited results after
    # interruption; its range and pixel-count checks are also retained here.
    result=st.read_json(path.parent/'exact_pixels.json')['result']
    backend.validate_exact_pixels(result,r['counts'])
    return r


@contextmanager
def scratch(a,d,b,count):
    root=a.evaluation_scratch/d/b;root.mkdir(parents=True,exist_ok=True);arrays=[]
    with tempfile.TemporaryDirectory(prefix='translation_',dir=root) as tmp:
        try:
            for name,dtype in (('labels',np.uint8),('scores',np.float32)):
                arrays.append(np.memmap(Path(tmp)/(name+'.bin'),mode='w+',dtype=dtype,shape=(count,)))
            yield arrays
        finally:
            for x in arrays:x._mmap.close()


def mask(a,data,d,row,shape):
    gt=backend.shared.resized_mask(data.gt(row),shape).astype(np.uint8)
    expected=ip.parent(a).maps[d][row['sample_id'],U]
    if backend.shared.array_sha(gt)!=expected['mask_uint8_sha256']:raise ValueError('parent GT/resize differs')
    return gt


def evaluate(a):
    frozen=pred.global_freeze(a);jobs=[]
    with ExitStack() as stack:
        data=None
        for d in DOMAINS:
            for b in MAPS:
                if completed(a,d,b):jobs.append(dict(domain=d,branch=b,reused=True));continue
                if data is None:data=stack.enter_context(GTData(a.data))
                samples=ip.parent(a).samples(d);n=sum((r['image_height']//4)*(r['image_width']//4) for r in samples)
                out=destination(a,d,b);records=[];scores=[];offset=0;tick=time.perf_counter()
                with scratch(a,d,b,n) as (labels,values):
                    for i,row in enumerate(samples,1):
                        im,sc,meta=pred.replay(a,d,row,b);gt=mask(a,data,d,row,meta['output_shape'])
                        if im.shape!=gt.shape or not np.isfinite(im).all():raise ValueError('invalid evaluation map')
                        values[offset:offset+gt.size]=im.ravel();labels[offset:offset+gt.size]=gt.ravel();offset+=gt.size
                        records.append(dict(sample_id=row['sample_id'],branch=b,output_height=gt.shape[0],output_width=gt.shape[1],
                            map_float32_sha256=backend.shared.array_sha(im),mask_uint8_sha256=backend.shared.array_sha(gt)))
                        scores.append(dict(domain=d,sample_id=row['sample_id'],label=row['label'],**sc))
                        if i%50==0 or i==len(samples):print(f'[residual exact CPU] {d}/{b} {i}/{len(samples)}',flush=True)
                    if offset!=n:raise ValueError('incomplete pixel population')
                    pos=int(labels.sum(dtype=np.int64));counts=dict(pixels=n,positive_pixels=pos,negative_pixels=n-pos)
                    pixel=backend.exact_pixel_result(out/'exact_pixels.json',ip.signature(a,d)['identity'],st.digest(records),labels,values,
                        domain=d,branch=b,prediction_sha=frozen['seal'],counts=counts)
                    metrics=[dict(domain=d,branch=b,scoring=rule,image_reference=S if rule=='SEGMENT_CH1' else b,
                                  image_count=len(samples),**image_metrics([r['label'] for r in scores],[r[rule] for r in scores]),
                                  pixel_AUROC=pixel['AUROC'],pixel_AP=pixel['AP'],AUPRO=None,
                                  image_metrics_inherited=rule=='SEGMENT_CH1',metric_origin='new_exact_common902') for rule in READOUTS]
                st.write_csv(out/'metrics.csv',metrics);st.write_csv(out/'map_manifest.csv',records);st.write_csv(out/'per_image_scores.csv',scores)
                names=('metrics.csv','map_manifest.csv','per_image_scores.csv','exact_pixels.json','exact_pixels.seal.json')
                jobs.append(st.sealed_json(out/'status.json',dict(**ip.signature(a,d),branch=b,images=len(samples),counts=counts,
                    prediction_seal=frozen['seal'],status='CPU_complete_AUPRO_pending',seconds=time.perf_counter()-tick,
                    files={k:st.sha(out/k) for k in names})))
                from .smoke import observe
                observe(a)
    return dict(status='requested_CPU_complete',jobs=len(jobs))


def au_expected(a,d,b):
    records=ip.rows(destination(a,d,b)/'map_manifest.csv')
    return dict(**ip.signature(a,d),branch=b,map_identity=st.digest(records),nstrips=200,FPR_max=.3),records


def au_completed(a,d,b):
    path=a.run_root/d/'aupro'/(b+'.json')
    if not path.exists():return None
    expected,_=au_expected(a,d,b);r=st.read_json(path);validate_aupro(r,expected)
    if r['verified_images']!=len(ip.parent(a).samples(d)):raise ValueError('partial AUPRO')
    return r


def aupro(a):
    pred.global_freeze(a);done=[]
    with ExitStack() as stack:
        data=None
        for d in DOMAINS:
            for b in MAPS:
                if not completed(a,d,b):raise ValueError('exact CPU completion required')
                r=au_completed(a,d,b)
                if r is None:
                    from importlib.metadata import version
                    if version('adeval')!='1.1.0':raise ValueError('original ADEval1.1.0 required; do not install/replace implicitly')
                    import torch
                    from DINOv3.relation_reliability.official_metrics import official_aupro_adeval
                    if not a.device.startswith('cuda:'):raise ValueError('original ADEval explicit CUDA stage required')
                    expected,records=au_expected(a,d,b);images=[];masks=[];tick=time.perf_counter()
                    if data is None:data=stack.enter_context(GTData(a.data))
                    for row,ref in zip(ip.parent(a).samples(d),records):
                        im,_,meta=pred.replay(a,d,row,b);gt=mask(a,data,d,row,meta['output_shape'])
                        backend.shared.verify_map_record(ref,im,gt);images.append(im);masks.append(gt)
                    torch.cuda.set_device(torch.device(a.device));torch.cuda.reset_peak_memory_stats()
                    r=dict(expected,AUPRO=float(official_aupro_adeval(masks,images,nstrips=200)),verified_images=len(images),
                           backend='original_ADEval_1.1.0_CUDA',seconds=time.perf_counter()-tick,
                           peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
                    r['seal']=st.digest(r);validate_aupro(r,expected);st.write_json(a.run_root/d/'aupro'/(b+'.json'),r)
                done.append(r)
                from .smoke import observe
                observe(a)
    return dict(status='requested_AUPRO_complete',jobs=len(done))
