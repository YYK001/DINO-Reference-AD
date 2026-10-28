"""Only the new map enters original exact pixel/ADEval; baselines read-only."""
from pathlib import Path
from types import FunctionType
from contextlib import contextmanager
import tempfile,time
import numpy as np
from . import inputs as ip,prediction as pred
from .inputs import st,GTData
from .protocol import U,S,MAIN,COMBINED
from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation.calibration import response
from DINOv3.MADEqual.aebad_s_support_witness_segment_p0 import evaluation as original
from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import image_metrics
from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic.numerics import check


def restored(a):
    path=a.run_root/'evaluation/status.json'
    if not path.exists():return None
    r=st.sealed_json(path)
    if r['identity']!=ip.identity(a) or r['prediction_seal']!=st.sealed_json(a.run_root/'predictions_frozen.json')['seal'] or r['images']!=297 or r['numerical_failed_checks']:raise ValueError('incomplete CPU evaluation')
    for name,h in r['files'].items():
        if st.sha(path.parent/name)!=h:raise ValueError('modified small CPU result: '+name)
    return r


@contextmanager
def scratch(a,count):
    from .runner import isolation
    isolation(a);a.evaluation_scratch.mkdir(parents=True,exist_ok=True)
    job=Path(tempfile.mkdtemp(prefix='blind_background_',dir=a.evaluation_scratch));created=[];opened=[]
    def allocate(name,dtype):
        f=job/(name+'.bin')
        with f.open('x+b') as h:
            created.append(f);x=np.memmap(h,mode='w+',dtype=dtype,shape=(count,))
        opened.append(x);return x
    try:yield allocate('labels',np.uint8),allocate('new_map',np.float32)
    finally:
        for x in opened:x._mmap.close()
        for f in created:f.unlink(missing_ok=True)
        if not any(job.iterdir()):job.rmdir()


def sample_work(a,s,parent_scores,parent_maps,parent_instances):
    p=ip.parent(a);x,r=pred.load(a,s);m=r['metadata'];old,om=p.prediction(s);maps={};scores=[];checks=[]
    if m['grid']!=om['metadata']['grid'] or m['output_shape']!=om['metadata']['output_shape']:raise ValueError('baseline map geometry changed')
    for b,z,pos in ((U,old[U],old['positions']),(S,old[S],old['positions']),(MAIN,x[MAIN],x['positions'])):
        maps[b],sc=response(z,pos,m['grid'],m['output_shape'],p.mad['image'])
        expected=m['scores'] if b==MAIN else parent_scores[s['sample_id'],b]
        for rule in ('CH1','CURRENT'):check(checks,s['sample_id'],'same_cache_'+rule,sc[rule],float(expected[rule]),branch=b,domain='background')
        scores.append(dict(domain='background',sample_id=s['sample_id'],label=s['label'],relative_path=s['relative_path'],branch=b,**sc))
    if not all(v['verified'] for v in checks):raise ValueError('frozen same-cache scoring mismatch')
    with GTData(a.data) as data:gt=data.gt(s)
    mask=original.shared.resized_mask(gt,m['output_shape']).astype(np.uint8)
    for b in (U,S):original.shared.verify_map_record(parent_maps[s['sample_id'],b],maps[b],mask)
    records=[dict(domain='background',sample_id=s['sample_id'],branch=b,output_height=v.shape[0],output_width=v.shape[1],
                  map_float32_sha256=original.shared.array_sha(v),mask_uint8_sha256=original.shared.array_sha(mask)) for b,v in maps.items()]
    instances=[]
    if s['label']:
        from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.analysis import spatial
        rows=FunctionType(spatial.__code__,dict(spatial.__globals__,U=U,S=S),spatial.__name__)(s,gt,maps,tuple(m['grid']))
        for row in rows:
            ref=parent_instances[s['sample_id'],str(row['instance_id']),U]
            row['small_group']=ref['small_group'].lower()=='true';row['small_group_definition']=ref['small_group_definition']
            if row['response_status']!=ref['response_status']:raise ValueError('instance NA status differs')
            if row['branch'] in (U,S):
                baseline=parent_instances[s['sample_id'],str(row['instance_id']),row['branch']]
                for field in ('margin','fixed_union_margin','instance_max','ring_max'):
                    if row[field] is None:
                        if baseline[field]!='':raise ValueError('instance became NA')
                    elif baseline[field]=='':raise ValueError('instance NA disappeared')
                    else:check(checks,s['sample_id'],'same_parent_'+field,row[field],float(baseline[field]),branch=row['branch'],domain='background')
            instances.append(row)
    if not all(v['verified'] for v in checks):raise ValueError('baseline instance scalar reconstruction differs')
    delta=maps[U].astype(np.float64)-maps[MAIN]
    regions={'normal_all_pixels':np.ones_like(mask,bool)} if not s['label'] else {'anomaly_GT_pixels':mask.astype(bool),'anomaly_normal_pixels':~mask.astype(bool)}
    changes=[dict(domain='background',sample_id=s['sample_id'],branch=MAIN,reference=U,group=name,pixels=int(region.sum()),
                  mean_signed_response_reduction=float(delta[region].mean()) if region.any() else None) for name,region in regions.items()]
    return s,maps,mask,scores,checks,instances,records,changes


def evaluate(a):
    pred.verified(a)
    if restored(a):return dict(status='CPU_reused')
    p=ip.parent(a);scores0={(r['sample_id'],r['branch']):r for r in p.table('per_image_scores.csv')}
    maps0={(r['sample_id'],r['branch']):r for r in p.table('map_manifest.csv')}
    inst0={(r['sample_id'],r['instance_id'],r['branch']):r for r in p.table('defect_instances.csv')}
    samples=p.samples;count=sum((s['image_height']//4)*(s['image_width']//4) for s in samples)
    scores=[];checks=[];instances=[];records=[];changes=[];offset=0;out=a.run_root/'evaluation';out.mkdir(parents=True,exist_ok=True)
    from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.execution import bounded_images
    with scratch(a,count) as (labels,values):
        for i,(s,maps,mask,sc,ck,inst,mr,ch) in enumerate(bounded_images(samples,lambda s:sample_work(a,s,scores0,maps0,inst0),a.cpu_workers),1):
            n=mask.size;labels[offset:offset+n]=mask.ravel();values[offset:offset+n]=maps[MAIN].ravel();offset+=n
            scores.extend(sc);checks.extend(ck);instances.extend(inst);records.extend(mr);changes.extend(ch)
            # A few original fixed cases, not selected by method performance.
            from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.fresh_inputs import CASE_IDS
            if s['sample_id'] in (*CASE_IDS[:2],*CASE_IDS[-2:]):
                from matplotlib import pyplot as plt
                from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.fresh_inputs import Data
                fig,axes=plt.subplots(1,4,figsize=(13,3));lo=min(float(v.min()) for v in maps.values());hi=max(float(v.max()) for v in maps.values())
                with Data(a.data) as data:rgb=data.image(s);rgb.thumbnail((512,512))
                axes[0].imshow(rgb);axes[0].set_title(s['sample_id'],fontsize=8);axes[0].axis('off')
                for ax,(b,v) in zip(axes[1:],maps.items()):
                    ax.imshow(v,cmap='magma',vmin=lo,vmax=hi)
                    if mask.any() and not mask.all():ax.contour(mask,levels=[.5],colors='cyan',linewidths=.4)
                    ax.set_title(b,fontsize=8);ax.axis('off')
                f=out/'figures'/(s['sample_id']+'.png');f.parent.mkdir(exist_ok=True);fig.savefig(f,dpi=100);plt.close(fig)
            if i%25==0 or i==297:print(f'[blind CPU] {i}/297',flush=True)
        if offset!=count:raise ValueError('incomplete full pixel population')
        positive=int(labels.sum(dtype=np.int64));counts=dict(pixels=count,positive_pixels=positive,negative_pixels=count-positive)
        local=[r for r in records if r['branch']==MAIN]
        pixel=original.exact_pixel_result(out/(MAIN+'_exact_pixels.json'),ip.identity(a),st.digest(local),labels,values,
             domain='background',branch=MAIN,prediction_sha=st.sealed_json(a.run_root/'predictions_frozen.json')['seal'],counts=counts)
    metrics=[];new=[r for r in scores if r['branch']==MAIN]
    for rule in ('CH1','CURRENT'):metrics.append(dict(domain='background',branch=MAIN,scoring=rule,image_count=297,
        **image_metrics([r['label'] for r in new],[r[rule] for r in new]),pixel_AUROC=pixel['AUROC'],pixel_AP=pixel['AP'],AUPRO=None,metric_origin='new_blind_context'))
    tables={'metrics.csv':metrics,'per_image_scores.csv':scores,'numerical_verification.csv':checks,'defect_instances.csv':instances,'map_manifest.csv':records,'response_changes.csv':changes}
    for name,rows in tables.items():st.write_csv(out/name,rows)
    names=list(tables)+[MAIN+'_exact_pixels.json',MAIN+'_exact_pixels.seal.json']
    return st.sealed_json(out/'status.json',dict(identity=ip.identity(a),prediction_seal=st.sealed_json(a.run_root/'predictions_frozen.json')['seal'],
           images=297,numerical_failed_checks=sum(not r['verified'] for r in checks),files={name:st.sha(out/name) for name in names}))


def aupro(a):
    pred.verified(a)
    if not restored(a):raise ValueError('complete CPU evaluation first')
    from DINOv3.MADEqual.aebad_s_fixed_target_graph_p0.evaluation import validate_aupro
    rows=[r for r in ip.read_csv(a.run_root/'evaluation/map_manifest.csv') if r['branch']==MAIN]
    expected=dict(identity=ip.identity(a),domain='background',branch=MAIN,map_identity=st.digest(rows),nstrips=200,FPR_max=.3)
    path=a.run_root/'aupro'/f'{MAIN}.json'
    if path.exists():
        r=st.read_json(path);validate_aupro(r,expected)
        if r['verified_images']!=297:raise ValueError('incomplete AUPRO')
        return r
    from importlib.metadata import version
    if version('adeval')!='1.1.0':raise ValueError('original ADEval1.1.0 required')
    import torch
    from DINOv3.relation_reliability.official_metrics import official_aupro_adeval
    maps=[];masks=[];p=ip.parent(a);tick=time.perf_counter()
    if [r['sample_id'] for r in rows]!=[s['sample_id'] for s in p.samples]:raise ValueError('AUPRO population differs')
    with GTData(a.data) as data:
        for s,record in zip(p.samples,rows):
            x,r=pred.load(a,s);m=r['metadata'];v=response(x[MAIN],x['positions'],m['grid'],m['output_shape'],p.mad['image'])[0]
            mask=original.shared.resized_mask(data.gt(s),m['output_shape']).astype(np.uint8);original.shared.verify_map_record(record,v,mask);maps.append(v);masks.append(mask)
    torch.cuda.set_device(torch.device(a.device));torch.cuda.reset_peak_memory_stats()
    value=dict(expected,AUPRO=float(official_aupro_adeval(masks,maps,nstrips=200)),verified_images=297,seconds=time.perf_counter()-tick,backend='original_ADEval_1.1.0_CUDA',
               peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    value['seal']=st.digest(value);validate_aupro(value,expected);st.write_json(path,value);return value
