"""Only three new maps enter original exact pixels and original ADEval."""
from contextlib import contextmanager
from pathlib import Path
from types import FunctionType
import tempfile,time,errno
import numpy as np
from . import inputs as ip,prediction as pred,method
from .inputs import st,GTData
from .protocol import CONFIG,DOMAINS,BRANCHES,HALF,U,S
from DINOv3.MADEqual.aebad_s_support_witness_segment_p0 import evaluation as original
from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.execution import bounded_images
from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import image_metrics
from DINOv3.MADEqual.aebad_s_segment_pair_crossdomain_p0.analysis import spatial


def restored(a,domain):
    path=a.run_root/'evaluation'/domain/'status.json'
    if not path.exists():return None
    r=st.sealed_json(path)
    if r['identity']!=ip.identity(a) or r['prediction_seal']!=st.sealed_json(a.run_root/'predictions'/(domain+'.json'))['seal'] or r['images']!=CONFIG['populations'][domain][0] or r['numerical_failed_checks']:raise ValueError('incomplete/changed CPU evaluation')
    for name,h in r['files'].items():
        if st.sha(path.parent/name)!=h:raise ValueError('completed small CPU result table changed')
    return r


@contextmanager
def scratch(a,domain):
    from .runner import isolation
    isolation(a);a.evaluation_scratch.mkdir(parents=True,exist_ok=True);job=Path(tempfile.mkdtemp(prefix=domain+'_',dir=a.evaluation_scratch));created=[];maps=[]
    def allocate(name,dtype,count):
        if name not in ('labels',*BRANCHES):raise ValueError('only new branches may allocate pixel scratch')
        p=job/(name+'.bin')
        with p.open('x+b') as f:created.append(p);x=np.memmap(f,mode='w+',dtype=dtype,shape=(count,))
        maps.append(x);return x
    try:yield allocate
    finally:
        for x in maps:x._mmap.close()
        for p in created:p.unlink(missing_ok=True)
        try:job.rmdir()
        except OSError as e:
            if e.errno not in (errno.ENOTEMPTY,errno.EEXIST):raise


def sample_work(a,s,states):
    parent=ip.parent(a);x,r=st.get(pred.prediction_path(a,s));m=pred.validate_prediction(a,s,x,r,states)
    old,old_record=parent.load(s,(U,S,'positions'));old_m=old_record['metadata'];checks=[];maps={};scores=[]
    if old_m['grid']!=m['grid'] or old_m['output_shape']!=m['output_shape'] or not np.array_equal(x['positions'],old['positions']):raise ValueError('replay physical geometry differs')
    for b in (U,S):maps[b]=method.response(old[b],x['positions'],m['grid'],m['output_shape'],parent.mad['image'])[0]
    for b in BRANCHES:
        response,score=method.response(x[b],x['positions'],m['grid'],m['output_shape'],states[b]['image']);maps[b]=response
        for rule in ('CH1','CURRENT'):method.comparison(checks,s,'new_same_cache_'+rule,score[rule],m['scores'][b][rule],branch=b,score=True)
        scores.append(dict(domain=s['domain'],sample_id=s['sample_id'],relative_path=s['relative_path'],label=s['label'],branch=b,**score))
    with GTData(a.data) as data:gt=data.gt(s)
    mask=original.shared.resized_mask(gt,m['output_shape']).astype(np.uint8)
    for b in (U,S):original.shared.verify_map_record(parent.maps[s['domain'],s['sample_id'],b],maps[b],mask)
    records=[dict(domain=s['domain'],sample_id=s['sample_id'],branch=b,output_height=maps[b].shape[0],output_width=maps[b].shape[1],
        map_float32_sha256=original.shared.array_sha(maps[b]),mask_uint8_sha256=original.shared.array_sha(mask)) for b in BRANCHES]
    instances=[]
    if s['label']:
        calculated=FunctionType(spatial.__code__,dict(spatial.__globals__,U=U,S=S),spatial.__name__)(s,gt,maps,tuple(m['grid']))
        expected={iid for d,sid,iid,b in parent.instances if (d,sid,b)==(s['domain'],s['sample_id'],U)}
        if {str(v['instance_id']) for v in calculated}!=expected:raise ValueError('original GT instance population differs')
        for row in calculated:
            old_u=parent.instances[s['domain'],s['sample_id'],str(row['instance_id']),U]
            if row['response_status']!=old_u['response_status']:raise ValueError('original NA instance status changed')
            if row['branch'] not in BRANCHES:continue
            row.update(small_group=old_u['small_group'].lower()=='true',small_group_definition=old_u['small_group_definition'])
            ref=parent.instances[s['domain'],s['sample_id'],str(row['instance_id']),HALF[row['branch']]]
            for field in ('instance_max','ring_max','margin','fixed_union_instance_response','fixed_union_ring_response','fixed_union_margin'):
                value=ref[field];row[field+'_change_vs_'+HALF[row['branch']]]=row[field]-float(value) if row[field] is not None and value!='' else None
            instances.append(row)
    changes=[]
    regions={'normal_all_pixels':np.ones_like(mask,bool)} if not s['label'] else {'anomaly_GT_pixels':mask.astype(bool),'anomaly_normal_pixels':~mask.astype(bool)}
    for b in BRANCHES:
        delta=maps[U].astype(np.float64)-maps[b]
        for group,region in regions.items():changes.append(dict(domain=s['domain'],sample_id=s['sample_id'],branch=b,reference=U,group=group,pixels=int(region.sum()),
            mean_signed_response_reduction=float(delta[region].mean()) if region.any() else None,units='final_fused_original_Source_patch_MAD_response'))
    return s,{b:maps[b] for b in BRANCHES},mask,scores,checks,instances,records,changes


def evaluate_domain(a,domain):
    if restored(a,domain):return dict(status='CPU_reused',domain=domain)
    path=a.run_root/'evaluation'/domain;path.mkdir(parents=True,exist_ok=True);samples=[s for s in ip.catalog(a)['samples'] if s['domain']==domain]
    count=sum((s['image_height']//4)*(s['image_width']//4) for s in samples);states=pred.calibration_states(a,domain)
    scores=[];checks=[];instances=[];records=[];changes=[];offset=0;started=time.perf_counter()
    with scratch(a,domain) as allocate:
        labels=allocate('labels',np.uint8,count);arrays={b:allocate(b,np.float32,count) for b in BRANCHES}
        for i,(s,maps,mask,sc,ck,inst,mr,ch) in enumerate(bounded_images(samples,lambda s:sample_work(a,s,states),a.cpu_workers),1):
            n=mask.size;labels[offset:offset+n]=mask.ravel()
            for b in BRANCHES:arrays[b][offset:offset+n]=maps[b].ravel()
            offset+=n;scores.extend(sc);checks.extend(ck);instances.extend(inst);records.extend(mr);changes.extend(ch)
            if i%25==0 or i==len(samples):print(f'[SACP L1 CPU maps] {domain} {i}/{len(samples)}',flush=True)
        if offset!=count:raise ValueError('incomplete new pixel population')
        positive=int(labels.sum(dtype=np.int64));counts=dict(pixels=count,positive_pixels=positive,negative_pixels=count-positive);metrics=[]
        ps=st.sealed_json(a.run_root/'predictions'/(domain+'.json'))['seal']
        for b in BRANCHES:
            local=[r for r in records if r['branch']==b]
            pixel=original.exact_pixel_result(path/(b+'_exact_pixels.json'),ip.identity(a),st.digest(local),labels,arrays[b],domain=domain,branch=b,prediction_sha=ps,counts=counts)
            rows=[r for r in scores if r['branch']==b]
            for rule in ('CH1','CURRENT'):metrics.append(dict(domain=domain,branch=b,scoring=rule,image_count=len(rows),
                **image_metrics([r['label'] for r in rows],[r[rule] for r in rows]),pixel_AUROC=pixel['AUROC'],pixel_AP=pixel['AP'],AUPRO=None,
                pixel_metric_origin='this_run_new_L1_original_exact_all_pixels_CPU'))
    files={'metrics.csv':metrics,'per_image_scores.csv':scores,'numerical_verification.csv':checks,'defect_instances.csv':instances,'map_manifest.csv':records,'response_changes.csv':changes}
    for name,rows in files.items():st.write_csv(path/name,rows)
    sealed_names=list(files)+[b+suffix for b in BRANCHES for suffix in ('_exact_pixels.json','_exact_pixels.seal.json')]
    return st.sealed_json(path/'status.json',dict(identity=ip.identity(a),prediction_seal=ps,domain=domain,images=len(samples),
        numerical_failed_checks=sum(not r['verified'] for r in checks),seconds=time.perf_counter()-started,files={name:st.sha(path/name) for name in sealed_names}))


def validate_aupro(record,expected,images):
    from DINOv3.MADEqual.aebad_s_fixed_target_graph_p0.evaluation import validate_aupro as original_validate
    original_validate(record,expected)
    if record.get('verified_images')!=images:raise ValueError('incomplete new AUPRO image population')


def aupro(a,domain,b):
    if b not in BRANCHES:raise ValueError('historical branch cannot enter new AUPRO')
    if not restored(a,domain):raise ValueError('complete CPU evaluation required first')
    maps_rows=[r for r in ip.read_csv(a.run_root/'evaluation'/domain/'map_manifest.csv') if r['branch']==b]
    samples=[s for s in ip.catalog(a)['samples'] if s['domain']==domain]
    if [r['sample_id'] for r in maps_rows]!=[s['sample_id'] for s in samples]:raise ValueError('incomplete new AUPRO population')
    expected=dict(identity=ip.identity(a),domain=domain,branch=b,map_identity=st.digest(maps_rows),nstrips=200,FPR_max=.3);dest=a.run_root/'aupro'/domain/(b+'.json')
    if dest.exists():r=st.read_json(dest);validate_aupro(r,expected,len(samples));return dict(r,runtime_reused=True)
    from importlib.metadata import version
    if version('adeval')!='1.1.0':raise ValueError('original ADEval1.1.0 required; no substitute backend')
    import torch
    from DINOv3.relation_reliability.official_metrics import official_aupro_adeval
    maps=[];masks=[];states=pred.calibration_states(a,domain);started=time.perf_counter()
    with GTData(a.data) as data:
        for s,record in zip(samples,maps_rows):
            x,r=st.get(pred.prediction_path(a,s));m=pred.validate_prediction(a,s,x,r,states)
            response,_=method.response(x[b],x['positions'],m['grid'],m['output_shape'],states[b]['image']);mask=original.shared.resized_mask(data.gt(s),m['output_shape']).astype(np.uint8)
            original.shared.verify_map_record(record,response,mask);maps.append(response);masks.append(mask)
    torch.cuda.set_device(torch.device(a.device));torch.cuda.reset_peak_memory_stats()
    v=float(official_aupro_adeval(masks,maps,nstrips=200));r=dict(expected,AUPRO=v,verified_images=len(samples),seconds=time.perf_counter()-started,
        backend='original_ADEval_1.1.0_CUDA',origin='this_run_new_L1',peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    r['seal']=st.digest(r);validate_aupro(r,expected,len(samples));st.write_json(dest,r);return dict(r,runtime_reused=False)


def evaluate(a,only_aupro=False):
    pred.verify(a)
    if not only_aupro:
        parent=ip.parent(a);parent.maps;parent.instances  # After global freeze, before worker threads; load each small table once.
    return [aupro(a,d,b) for d in DOMAINS for b in BRANCHES] if only_aupro else [evaluate_domain(a,d) for d in DOMAINS]
