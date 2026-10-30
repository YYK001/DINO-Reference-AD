"""Only six NEW cross maps get full pixel sorting and original AUPRO."""
from contextlib import contextmanager,ExitStack
from pathlib import Path
import tempfile,time
import numpy as np
from . import inputs as ip,prediction as pred
from .inputs import st,GTData
from .protocol import *
from DINOv3.MADEqual.aebad_s_support_witness_segment_p0 import evaluation as backend
from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import image_metrics
from DINOv3.MADEqual.aebad_s_fixed_target_graph_p0.evaluation import validate_aupro


def mask_record(data,row,shape):
    original=data.gt(row);return original,backend.shared.resized_mask(original,shape).astype(np.uint8)


def completed(a,d,m,kind):
    path=ip.unit(a,d,m)/kind/'status.json'
    if not path.exists():return None
    r=st.sealed_json(path);sig=ip.signature(a,d,m)
    if any(r.get(k)!=v for k,v in sig.items()) or r['response_freeze']!=st.sealed_json(a.output_root/'responses_frozen.json')['seal']:raise ValueError('audit stage input changed')
    for name,h in r['files'].items():
        if st.sha(path.parent/name)!=h:raise ValueError('audit small result modified: '+name)
    for name,seal in r.get('compact_pair_bundles',{}).items():
        _,bundle=st.get(path.parent/'pair_records'/name)
        if bundle['seal']!=seal:raise ValueError('compact pair bundle changed')
    return r


@contextmanager
def scratch(a,d,m,n):
    root=a.evaluation_scratch/d/f'memory{m}';root.mkdir(parents=True,exist_ok=True);arrays=[]
    with tempfile.TemporaryDirectory(prefix='cross_',dir=root) as temp:
        try:
            for name,dtype in [('labels',np.uint8),('scores',np.float32)]:arrays.append(np.memmap(Path(temp)/(name+'.bin'),mode='w+',dtype=dtype,shape=(n,)))
            yield arrays
        finally:
            for x in arrays:x._mmap.close()


def replay(a,d,m,row,w):
    images,scores,_,z,pos,ref=pred.load_maps(a,d,m,row,only=w);f=st.sealed_json(ip.record_path(a,d,m,row))
    if ref['seal']!=f['response']['seal'] or backend.shared.array_sha(images[w])!=f['map_hashes'][w]:raise ValueError('crossed map differs from no-GT freeze')
    if any(abs(scores[w][k]-f['scores'][w][k])>SCORE_ATOL for k in READOUTS):raise ValueError('crossed score differs from no-GT freeze')
    return images[w],f['scores'][w],ref


def evaluate(a):
    freeze=pred.global_freeze(a);p=ip.parent(a);results=[]
    with ExitStack() as stack:
        data=None
        for d in a.domains:
            for m in a.memory_seeds:
                if completed(a,d,m,'evaluation'):results.append(dict(domain=d,memory_seed=m,status='CPU_reused'));continue
                if data is None:data=stack.enter_context(GTData(a.data))
                sig=ip.signature(a,d,m);out=ip.unit(a,d,m)/'evaluation';w=crossed(m);samples=p.samples(d)
                n=sum((r['image_height']//4)*(r['image_width']//4) for r in samples);offset=0;records=[];scores=[]
                with scratch(a,d,m,n) as (labels,values):
                    for i,row in enumerate(samples,1):
                        image,sc,ref=replay(a,d,m,row,w);_,mask=mask_record(data,row,ref['output_shape'])
                        old=p.maps[d,m][row['sample_id'],SOURCE]
                        if backend.shared.array_sha(mask)!=old['mask_uint8_sha256']:raise ValueError('common GT/resize differs from parent')
                        if image.shape!=mask.shape or not np.isfinite(image).all():raise ValueError('invalid crossed evaluation map')
                        values[offset:offset+mask.size]=image.ravel();labels[offset:offset+mask.size]=mask.ravel();offset+=mask.size
                        records.append(dict(**sig,sample_id=row['sample_id'],branch=branch(m,w),map_float32_sha256=backend.shared.array_sha(image),mask_uint8_sha256=backend.shared.array_sha(mask),output_height=mask.shape[0],output_width=mask.shape[1]))
                        scores.append(dict(**sig,sample_id=row['sample_id'],label=row['label'],**sc))
                        if i%25==0 or i==len(samples):print(f'[cross exact CPU maps] {d}/Z{m} {i}/{len(samples)}',flush=True)
                        pred.budget(a,i,len(samples))
                    if offset!=n:raise ValueError('incomplete exact pixel population')
                    positive=int(labels.sum(dtype=np.int64));counts=dict(pixels=n,positive_pixels=positive,negative_pixels=n-positive)
                    pixel=backend.exact_pixel_result(out/'cross_exact_pixels.json',sig['identity'],st.digest(records),labels,values,domain=d,branch=branch(m,w),prediction_sha=freeze['seal'],counts=counts)
                    metric=[dict(**sig,branch=branch(m,w),scoring=rule,image_count=len(samples),**image_metrics([r['label'] for r in scores],[r[rule] for r in scores]),pixel_AUROC=pixel['AUROC'],pixel_AP=pixel['AP'],AUPRO=None,metric_origin='new_cross_only_exact_CPU') for rule in READOUTS]
                st.write_csv(out/'metrics.csv',metric);st.write_csv(out/'per_image_scores.csv',scores);st.write_csv(out/'map_manifest.csv',records)
                files=('metrics.csv','per_image_scores.csv','map_manifest.csv','cross_exact_pixels.json','cross_exact_pixels.seal.json')
                results.append(st.sealed_json(out/'status.json',dict(**sig,status='CPU_complete_AUPRO_pending',response_freeze=freeze['seal'],images=len(samples),files={f:st.sha(out/f) for f in files})))
    return dict(status='requested_cross_CPU_complete',results=results)


def aupro(a):
    pred.global_freeze(a);p=ip.parent(a);results=[]
    with ExitStack() as stack:
        data=None
        for d in a.domains:
            for m in a.memory_seeds:
                if not completed(a,d,m,'evaluation'):raise ValueError('cross CPU evaluation must complete before AUPRO')
                records=ip.parent_reporting.rows(ip.unit(a,d,m)/'evaluation/map_manifest.csv');w=crossed(m)
                expected=dict(**ip.signature(a,d,m),branch=branch(m,w),map_identity=st.digest(records),nstrips=200,FPR_max=.3)
                dest=ip.unit(a,d,m)/'aupro.json'
                if dest.exists():r=st.read_json(dest);validate_aupro(r,expected)
                else:
                    if data is None:data=stack.enter_context(GTData(a.data))
                    from importlib.metadata import version
                    if version('adeval')!='1.1.0':raise ValueError('original ADEval1.1.0 required; no dependency installation')
                    import torch
                    from DINOv3.relation_reliability.official_metrics import official_aupro_adeval
                    images=[];masks=[];tick=time.perf_counter()
                    for row,record in zip(p.samples(d),records):
                        image,sc,ref=replay(a,d,m,row,w);_,mask=mask_record(data,row,ref['output_shape']);backend.shared.verify_map_record(record,image,mask);images.append(image);masks.append(mask)
                    torch.cuda.set_device(torch.device(a.device));torch.cuda.reset_peak_memory_stats()
                    r=dict(expected,AUPRO=float(official_aupro_adeval(masks,images,nstrips=200)),backend='original_ADEval_1.1.0_CUDA',verified_images=len(images),seconds=time.perf_counter()-tick,
                        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
                    r['seal']=st.digest(r);validate_aupro(r,expected);st.write_json(dest,r)
                    for key in ('peak_cuda_allocated_bytes','peak_cuda_reserved_bytes'):setattr(a,key,max(getattr(a,key,0),r[key]))
                if r['verified_images']!=len(p.samples(d)):raise ValueError('incomplete crossed AUPRO')
                results.append(r);pred.budget(a,len(results),len(a.domains)*len(a.memory_seeds))
    return dict(status='requested_new_cross_AUPRO_complete',jobs=len(results))
