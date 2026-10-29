"""Original exact all-pixel CPU and explicit original ADEval, two new maps only."""
from contextlib import contextmanager
from pathlib import Path
import tempfile,time
import numpy as np
from . import inputs as ip,pipeline as pp
from .inputs import st
from .protocol import NEW,READOUTS,U,S
from DINOv3.MADEqual.aebad_s_support_witness_segment_p0 import evaluation as old
from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import image_metrics


@contextmanager
def scratch(a,count):
    a.evaluation_scratch.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='layer_weights_',dir=a.evaluation_scratch) as tmp:
        arrays=[]
        try:
            for name,dtype in [('labels',np.uint8)]+[(b,np.float32) for b in NEW]:
                arrays.append(np.memmap(Path(tmp)/(name+'.bin'),mode='w+',dtype=dtype,shape=(count,)))
            yield arrays[0],dict(zip(NEW,arrays[1:]))
        finally:
            for v in arrays:v._mmap.close()


def restored(a):
    p=a.run_root/'evaluation/status.json'
    if not p.exists():return None
    r=st.sealed_json(p)
    if r['identity']!=ip.identity(a) or r['prediction_seal']!=pp.verified(a)['seal']:raise ValueError('CPU inputs changed')
    for name,h in r['files'].items():
        if st.sha(p.parent/name)!=h:raise ValueError('modified small evaluation result: '+name)
    return r


def mask(a,row,shape):
    with ip.old.GTData(a.data) as data:return old.shared.resized_mask(data.gt(row),shape).astype(np.uint8)


def evaluate(a):
    frozen=pp.verified(a)
    if restored(a):return dict(status='CPU_reused')
    p=ip.parent(a);out=a.run_root/'evaluation';out.mkdir(parents=True,exist_ok=True)
    parent_maps={(r['sample_id'],r['branch']):r for r in p.table('map_manifest.csv')}
    count=sum((r['image_height']//4)*(r['image_width']//4) for r in p.samples)
    from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.calibration import response
    scores=[];records=[];checks=[];offset=0
    with scratch(a,count) as (labels,values):
        for i,row in enumerate(p.samples,1):
            x,r=pp.prediction(a,row);baseline,m=p.prediction(row);meta=m['metadata'];gt=mask(a,row,meta['output_shape'])
            for b in (U,S):
                image,sc=response(baseline[b],baseline['positions'],meta['grid'],meta['output_shape'],p.mad['image'])
                old.shared.verify_map_record(parent_maps[row['sample_id'],b],image,gt)
                if any(abs(sc[k]-meta['scores'][b][k])>1e-5 for k in ('CH1','CURRENT')):raise ValueError('parent score mismatch')
                checks.append(dict(sample_id=row['sample_id'],branch=b,verified=True,map_GT='exact_parent_hash',score_atol=1e-5))
            n=gt.size;labels[offset:offset+n]=gt.ravel()
            for b in NEW:
                if x[b].shape!=gt.shape or x[b].dtype!=np.float32 or not np.isfinite(x[b]).all():raise ValueError('invalid weighted map')
                values[b][offset:offset+n]=x[b].ravel()
                scores.append(dict(sample_id=row['sample_id'],branch=b,label=row['label'],**r['metadata']['scores'][b]))
                records.append(dict(domain='background',sample_id=row['sample_id'],branch=b,output_height=gt.shape[0],output_width=gt.shape[1],map_float32_sha256=old.shared.array_sha(x[b]),mask_uint8_sha256=old.shared.array_sha(gt)))
            offset+=n
            if i%25==0:print(f'[layer weights CPU] {i}/297',flush=True)
            pp.guard(a,'evaluate',i,297)
        if offset!=count:raise ValueError('incomplete exact pixel population')
        positive=int(labels.sum(dtype=np.int64));counts=dict(pixels=count,positive_pixels=positive,negative_pixels=count-positive)
        metrics=[]
        for b in NEW:
            maps=[r for r in records if r['branch']==b]
            pix=old.exact_pixel_result(out/(b+'_exact_pixels.json'),ip.identity(a),st.digest(maps),labels,values[b],domain='background',branch=b,prediction_sha=frozen['seal'],counts=counts)
            ss=[r for r in scores if r['branch']==b]
            for rule in READOUTS:metrics.append(dict(domain='background',branch=b,scoring=rule,image_count=297,**image_metrics([r['label'] for r in ss],[r[rule] for r in ss]),pixel_AUROC=pix['AUROC'],pixel_AP=pix['AP'],AUPRO=None,metric_origin='new_weighted_map_exact_CPU'))
    metrics.extend(dict(r,metric_origin='parent_map_GT_scores_population_verified') for r in p.metrics)
    tables={'metrics.csv':metrics,'per_image_scores.csv':scores,'map_manifest.csv':records,'numerical_verification.csv':checks}
    for name,rows in tables.items():st.write_csv(out/name,rows)
    names=list(tables)+[b+s for b in NEW for s in ('_exact_pixels.json','_exact_pixels.seal.json')]
    return st.sealed_json(out/'status.json',dict(identity=ip.identity(a),prediction_seal=frozen['seal'],images=297,files={name:st.sha(out/name) for name in names},numerical_failed_checks=0))


def aupro(a):
    pp.verified(a)
    if not restored(a):raise ValueError('complete CPU evaluation first')
    from importlib.metadata import version
    if version('adeval')!='1.1.0':raise ValueError('ADEval1.1.0 required; no installs or approximations')
    from DINOv3.MADEqual.aebad_s_fixed_target_graph_p0.evaluation import validate_aupro
    from DINOv3.relation_reliability.official_metrics import official_aupro_adeval
    import torch
    result=[];allmaps=ip.old.read_csv(a.run_root/'evaluation/map_manifest.csv')
    for b in NEW:
        records=[r for r in allmaps if r['branch']==b]
        if [r['sample_id'] for r in records]!=[s['sample_id'] for s in ip.parent(a).samples]:raise ValueError('AUPRO population differs')
        expected=dict(identity=ip.identity(a),domain='background',branch=b,map_identity=st.digest(records),nstrips=200,FPR_max=.3)
        dest=a.run_root/'aupro'/(b+'.json')
        if dest.exists():v=st.read_json(dest);validate_aupro(v,expected)
        else:
            maps=[];masks=[];tick=time.perf_counter()
            for row,record in zip(ip.parent(a).samples,records):
                x,_=pp.prediction(a,row);gt=mask(a,row,x[b].shape)
                old.shared.verify_map_record(record,x[b],gt);maps.append(x[b]);masks.append(gt)
            torch.cuda.set_device(torch.device(a.device));torch.cuda.reset_peak_memory_stats()
            v=dict(expected,AUPRO=float(official_aupro_adeval(masks,maps,nstrips=200)),verified_images=297,seconds=time.perf_counter()-tick,backend='original_ADEval_1.1.0_CUDA')
            v['seal']=st.digest(v);validate_aupro(v,expected);st.write_json(dest,v)
        if v['verified_images']!=297:raise ValueError('partial AUPRO')
        result.append(v);pp.guard(a,'aupro',len(result),2)
    return dict(status='requested_AUPRO_complete',jobs=len(result))
