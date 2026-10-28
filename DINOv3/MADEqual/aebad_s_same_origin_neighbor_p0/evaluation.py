"""Existing exact CPU metrics, original ADEval, and original instance geometry."""
from pathlib import Path
from contextlib import contextmanager
from types import FunctionType
import tempfile,time
import numpy as np
from . import inputs as ip,pipeline as pp
from .inputs import st,GTData,read_csv
from .protocol import U,S,NEW,MAPS,COMBINED
from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.calibration import response
from DINOv3.MADEqual.aebad_s_support_witness_segment_p0 import evaluation as old
from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import image_metrics
from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic.numerics import check


def restored(a):
    p=a.run_root/'evaluation/status.json'
    if not p.exists():return None
    r=st.sealed_json(p)
    if r['identity']!=ip.identity(a) or r['prediction_seal']!=st.sealed_json(a.run_root/'predictions_frozen.json')['seal'] or r['images']!=297 or r['numerical_failed_checks']:raise ValueError('CPU completion input differs')
    for name,h in r['files'].items():
        if st.sha(p.parent/name)!=h:raise ValueError('completed small CPU result changed: '+name)
    return r


@contextmanager
def scratch(a,count,branches):
    from .runner import isolation
    isolation(a);a.evaluation_scratch.mkdir(parents=True,exist_ok=True)
    root=Path(tempfile.mkdtemp(prefix='same_origin_',dir=a.evaluation_scratch));created=[];opened=[]
    def alloc(name,dtype):
        f=root/(name+'.bin')
        with f.open('x+b') as h:
            created.append(f);v=np.memmap(h,mode='w+',dtype=dtype,shape=(count,))
        opened.append(v);return v
    try:yield alloc('labels',np.uint8),{b:alloc(b,np.float32) for b in branches}
    finally:
        for v in opened:v._mmap.close()
        for f in created:f.unlink(missing_ok=True)
        if not any(root.iterdir()):root.rmdir()


def work(a,s):
    x,r=pp.load(a,s);m=r['metadata'];cal=pp.states(a);maps={};scores=[];checks=[]
    for b in MAPS:
        maps[b],sc=response(x[b],x['positions'],m['grid'],m['output_shape'],cal['image'][b])
        for rule in ('CH1','CURRENT'):check(checks,s['sample_id'],'same_cache_'+rule,sc[rule],m['scores'][b][rule],branch=b,domain='background')
        scores.append(dict(domain='background',sample_id=s['sample_id'],label=s['label'],branch=b,**sc))
    if not all(v['verified'] for v in checks):raise ValueError('frozen map/score reproduction differs')
    with GTData(a.data) as data:gt=data.gt(s)
    mask=old.shared.resized_mask(gt,m['output_shape']).astype(np.uint8)
    records=[dict(domain='background',sample_id=s['sample_id'],branch=b,output_height=v.shape[0],output_width=v.shape[1],
        map_float32_sha256=old.shared.array_sha(v),mask_uint8_sha256=old.shared.array_sha(mask)) for b,v in maps.items()]
    instances=[]
    if s['label']:
        from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.analysis import spatial
        instances=FunctionType(spatial.__code__,dict(spatial.__globals__,U=U,S=S),spatial.__name__)(s,gt,maps,tuple(m['grid']))
    return maps,mask,scores,checks,records,instances


def evaluate(a):
    frozen=pp.verified(a)
    if restored(a):return dict(status='CPU_reused')
    cal=pp.states(a);parent=ip.parent(a) if cal['reuse_baselines'] else None
    active=NEW if parent else MAPS;samples=ip.catalog(a)['samples']
    parent_maps={(r['sample_id'],r['branch']):r for r in parent.table('map_manifest.csv')} if parent else {}
    scores=[];checks=[];records=[];instances=[];offset=0
    count=sum((r['image_height']//4)*(r['image_width']//4) for r in samples);out=a.run_root/'evaluation';out.mkdir(parents=True,exist_ok=True)
    from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.execution import bounded_images
    with scratch(a,count,active) as (labels,values):
        for i,(s,result) in enumerate(bounded_images(samples,lambda s:(s,work(a,s)),a.cpu_workers),1):
            maps,mask,sc,ck,mr,inst=result
            if parent:
                for b in (U,S):old.shared.verify_map_record(parent_maps[s['sample_id'],b],maps[b],mask)
            n=mask.size;labels[offset:offset+n]=mask.ravel()
            for b in active:values[b][offset:offset+n]=maps[b].ravel()
            offset+=n;scores.extend(sc);checks.extend(ck);records.extend(mr);instances.extend(inst)
            # Fixed historical examples only; no selection based on new results.
            from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.fresh_inputs import CASE_IDS
            if s['sample_id'] in (*CASE_IDS[:2],*CASE_IDS[-2:]):
                from matplotlib import pyplot as plt
                fig,axes=plt.subplots(1,5,figsize=(16,3));lo=min(v.min() for v in maps.values());hi=max(v.max() for v in maps.values())
                for ax,(b,v) in zip(axes,maps.items()):
                    ax.imshow(v,cmap='magma',vmin=lo,vmax=hi);ax.set_title(b,fontsize=7);ax.axis('off')
                    if mask.any() and not mask.all():ax.contour(mask,levels=[.5],colors='cyan',linewidths=.4)
                (out/'figures').mkdir(exist_ok=True);fig.savefig(out/'figures'/(s['sample_id']+'.png'),dpi=100);plt.close(fig)
            if i%25==0 or i==297:print(f'[same-origin exact CPU maps] {i}/297',flush=True)
            pp.budget(a,'evaluate',i,297)
        if offset!=count:raise ValueError('full pixel population incomplete')
        positive=int(labels.sum(dtype=np.int64));counts=dict(pixels=count,positive_pixels=positive,negative_pixels=count-positive)
        metrics=[]
        for b in active:
            local=[r for r in records if r['branch']==b]
            pix=old.exact_pixel_result(out/(b+'_exact_pixels.json'),ip.identity(a),st.digest(local),labels,values[b],domain='background',branch=b,prediction_sha=frozen['seal'],counts=counts)
            sc=[r for r in scores if r['branch']==b]
            for rule in ('CH1','CURRENT'):metrics.append(dict(domain='background',branch=b,scoring=rule,image_count=297,
                **image_metrics([r['label'] for r in sc],[r[rule] for r in sc]),pixel_AUROC=pix['AUROC'],pixel_AP=pix['AP'],AUPRO=None,metric_origin='current_exact_all_pixels'))
    if parent:
        for r in parent.metrics:
            if r['branch'] in (U,S):metrics.append(dict(r,metric_origin='parent_exact_map_GT_and_score_verified'))
    # Supplemental instance margins retain NA; changed response scales forbid a
    # claim that their difference is native distance shrinkage or missed defects.
    tables={'metrics.csv':metrics,'per_image_scores.csv':scores,'numerical_verification.csv':checks,'map_manifest.csv':records,'defect_instances.csv':instances}
    for name,rows in tables.items():st.write_csv(out/name,rows)
    names=list(tables)+[name for b in active for name in (b+'_exact_pixels.json',b+'_exact_pixels.seal.json')]
    return st.sealed_json(out/'status.json',dict(identity=ip.identity(a),prediction_seal=frozen['seal'],images=297,new_pixel_jobs=len(active),
        numerical_failed_checks=sum(not r['verified'] for r in checks),files={name:st.sha(out/name) for name in names}))


def aupro(a):
    pp.verified(a)
    if not restored(a):raise ValueError('complete CPU evaluation first')
    from importlib.metadata import version
    if version('adeval')!='1.1.0':raise ValueError('original ADEval1.1.0 required; no automatic installation')
    from DINOv3.MADEqual.aebad_s_fixed_target_graph_p0.evaluation import validate_aupro
    import torch
    from DINOv3.relation_reliability.official_metrics import official_aupro_adeval
    cal=pp.states(a);active=NEW if cal['reuse_baselines'] else MAPS;result=[]
    for b in active:
        rows=[r for r in read_csv(a.run_root/'evaluation/map_manifest.csv') if r['branch']==b]
        expected=dict(identity=ip.identity(a),domain='background',branch=b,map_identity=st.digest(rows),nstrips=200,FPR_max=.3)
        p=a.run_root/'aupro'/(b+'.json')
        if p.exists():
            v=st.read_json(p);validate_aupro(v,expected)
            if v['verified_images']!=297:raise ValueError('partial AUPRO job')
            result.append(v);continue
        maps=[];masks=[];tick=time.perf_counter();samples=ip.catalog(a)['samples']
        if [r['sample_id'] for r in rows]!=[s['sample_id'] for s in samples]:raise ValueError('AUPRO population differs')
        with GTData(a.data) as data:
            for s,record in zip(samples,rows):
                x,r=pp.load(a,s);m=r['metadata'];v=response(x[b],x['positions'],m['grid'],m['output_shape'],cal['image'][b])[0]
                mask=old.shared.resized_mask(data.gt(s),m['output_shape']).astype(np.uint8)
                old.shared.verify_map_record(record,v,mask);maps.append(v);masks.append(mask)
        torch.cuda.set_device(torch.device(a.device));torch.cuda.reset_peak_memory_stats()
        v=dict(expected,AUPRO=float(official_aupro_adeval(masks,maps,nstrips=200)),verified_images=297,seconds=time.perf_counter()-tick,
            backend='original_ADEval_1.1.0_CUDA',peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        v['seal']=st.digest(v);validate_aupro(v,expected);st.write_json(p,v);result.append(v)
        pp.budget(a,'aupro',len(result),len(active))
    return dict(status='requested_AUPRO_complete',jobs=len(result))
