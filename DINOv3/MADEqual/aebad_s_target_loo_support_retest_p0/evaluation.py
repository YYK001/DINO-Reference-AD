"""Re-evaluate all four independent maps on the NEW common population."""
from contextlib import contextmanager
from pathlib import Path
import tempfile,time
import numpy as np
from . import inputs as ip,runtime as rt,pipeline as pp
from .inputs import st
from .protocol import *
from DINOv3.MADEqual.aebad_s_support_witness_segment_p0 import evaluation as backend
from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import image_metrics
from DINOv3.MADEqual.aebad_s_target_loo_layer_weight_p0.composition import fixed_combinations


def metric_branches(base,domain,seed):
    index={(r['branch'],r['scoring']):r for r in base}
    required={(b,sc) for b in (U,S) for sc in ('CH1','CURRENT')}|{(b,sc) for b in NEW for sc in READOUTS}
    if len(base)!=8 or set(index)!=required:raise ValueError('four independent maps/two readouts required')
    rows=list(base)
    for sc in ('CH1','CURRENT'):
        r=dict(index[U,sc],branch=COMBINED,image_AUROC=index[S,sc]['image_AUROC'],image_AP=index[S,sc]['image_AP'],image_reference=S,map_reference=U)
        rows.append(r)
    rows=fixed_combinations(rows,domain)
    return [dict(r,support_seed=seed,proposal_role='fixed_before_two_support_set_evaluation') for r in rows]


@contextmanager
def scratch(a,count):
    root=a.evaluation_scratch/a.domain/f'seed{a.support_seed}';root.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='common_',dir=root) as tmp:
        values=[]
        try:
            for name,dtype in [('labels',np.uint8)]+[(b,np.float32) for b in MAPS]:values.append(np.memmap(Path(tmp)/(name+'.bin'),mode='w+',dtype=dtype,shape=(count,)))
            yield values[0],dict(zip(MAPS,values[1:]))
        finally:
            for x in values:x._mmap.close()


def gt(a,row,shape):
    with ip.layer_inputs.old.GTData(a.data) as data:return backend.shared.resized_mask(data.gt(row),shape).astype(np.uint8)


def restored(a):
    dest=ip.setting(a)/'evaluation/status.json'
    if not dest.exists():return None
    r=st.sealed_json(dest);frozen=pp.verified(a)
    if any(r.get(k)!=v for k,v in ip.signature(a).items()) or r['prediction_seal']!=frozen['seal']:raise ValueError('CPU common-population/group identity differs')
    for name,sha in r['files'].items():
        if st.sha(dest.parent/name)!=sha:raise ValueError('modified small CPU output: '+name)
    return r


def evaluate(a):
    frozen=pp.verified(a)
    if restored(a):return dict(status='common_CPU_reused',**ip.signature(a))
    _,_,samples=ip.context(a);sig=ip.signature(a);out=ip.setting(a)/'evaluation';out.mkdir(parents=True,exist_ok=True)
    n=sum((r['image_height']//4)*(r['image_width']//4) for r in samples);records=[];scores=[];checks=[];offset=0;combo=[]
    with scratch(a,n) as (labels,values):
        for i,row in enumerate(samples,1):
            maps,sc,m=rt.maps(a,row);f=st.sealed_json(ip.record_path(a,row));mask=gt(a,row,m['output_shape'])
            for b in MAPS:
                if backend.shared.array_sha(maps[b])!=f['map_hashes'][b] or any(abs(v-f['scores'][b][k])>1e-5 for k,v in sc[b].items()):raise ValueError('frozen map/score replay differs')
                if not np.isfinite(maps[b]).all() or maps[b].shape!=mask.shape:raise ValueError('invalid common-population map')
                values[b][offset:offset+mask.size]=maps[b].ravel()
                records.append(dict(**sig,sample_id=row['sample_id'],branch=b,output_height=mask.shape[0],output_width=mask.shape[1],map_float32_sha256=f['map_hashes'][b],mask_uint8_sha256=backend.shared.array_sha(mask)))
                scores.append(dict(**sig,sample_id=row['sample_id'],label=row['label'],branch=b,**sc[b]))
                checks.append(dict(**sig,sample_id=row['sample_id'],branch=b,verified=True,scoring_atol=1e-5))
            for b,ref in dict(COMBINATIONS,**{COMBINED:U}).items():
                combo.append(dict(**sig,sample_id=row['sample_id'],branch=b,image_reference=S,map_reference=ref,map_float32_sha256=f['map_hashes'][ref],no_new_map=True))
                scores.append(dict(**sig,sample_id=row['sample_id'],label=row['label'],branch=b,**sc[S]))
            labels[offset:offset+mask.size]=mask.ravel();offset+=mask.size
            if i%25==0 or i==len(samples):print(f'[two K8 CPU] {a.domain}/seed{a.support_seed} {i}/{len(samples)} four_exact_maps',flush=True)
            pp.guard(a,'evaluate',i,len(samples))
        if offset!=n:raise ValueError('common pixel population incomplete')
        pos=int(labels.sum(dtype=np.int64));counts=dict(pixels=n,positive_pixels=pos,negative_pixels=n-pos);metrics=[]
        for b in MAPS:
            selected=[r for r in records if r['branch']==b]
            pixel=backend.exact_pixel_result(out/(b+'_exact_pixels.json'),sig['identity'],st.digest(selected),labels,values[b],domain=a.domain,branch=b,prediction_sha=frozen['seal'],counts=counts)
            ss=[r for r in scores if r['branch']==b]
            for rule in (('CH1','CURRENT') if b in (U,S) else READOUTS):
                metrics.append(dict(**sig,branch=b,scoring=rule,image_count=len(samples),**image_metrics([r['label'] for r in ss],[r[rule] for r in ss]),pixel_AUROC=pixel['AUROC'],pixel_AP=pixel['AP'],AUPRO=None,metric_origin='new_common_population_exact_CPU'))
    tables={'metrics.csv':metrics,'per_image_scores.csv':scores,'map_manifest.csv':records,'combination_manifest.csv':combo,'numerical_verification.csv':checks}
    for name,rows in tables.items():st.write_csv(out/name,rows)
    files=list(tables)+[b+suffix for b in MAPS for suffix in ('_exact_pixels.json','_exact_pixels.seal.json')]
    return st.sealed_json(out/'status.json',dict(**sig,images=len(samples),prediction_seal=frozen['seal'],files={name:st.sha(out/name) for name in files},independent_map_jobs=4,numerical_failed_checks=0))


def aupro(a):
    pp.verified(a)
    if not restored(a):raise ValueError('new common-population CPU evaluation required')
    from DINOv3.MADEqual.aebad_s_fixed_target_graph_p0.evaluation import validate_aupro
    records=ip.layer_inputs.old.read_csv(ip.setting(a)/'evaluation/map_manifest.csv');_,_,samples=ip.context(a);done=[]
    for b in MAPS:
        selected=[r for r in records if r['branch']==b]
        if [r['sample_id'] for r in selected]!=[r['sample_id'] for r in samples]:raise ValueError('wrong common AUPRO list')
        expected=dict(**ip.signature(a),branch=b,map_identity=st.digest(selected),nstrips=200,FPR_max=.3)
        dest=ip.setting(a)/'aupro'/(b+'.json')
        if dest.exists():r=st.read_json(dest);validate_aupro(r,expected)
        else:
            from importlib.metadata import version
            if version('adeval')!='1.1.0':raise ValueError('original ADEval1.1.0 required; no automatic install')
            import torch
            from DINOv3.relation_reliability.official_metrics import official_aupro_adeval
            images=[];masks=[];tick=time.perf_counter()
            for row,ref in zip(samples,selected):
                x,_,m=rt.maps(a,row,only=b);mask=gt(a,row,m['output_shape']);backend.shared.verify_map_record(ref,x[b],mask);images.append(x[b]);masks.append(mask)
            torch.cuda.set_device(torch.device(a.device));torch.cuda.reset_peak_memory_stats()
            r=dict(expected,AUPRO=float(official_aupro_adeval(masks,images,nstrips=200)),verified_images=len(samples),seconds=time.perf_counter()-tick,backend='original_ADEval_1.1.0_CUDA')
            r.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
            for key in ('peak_cuda_allocated_bytes','peak_cuda_reserved_bytes'):setattr(a,key,max(getattr(a,key,0),r[key]))
            r['seal']=st.digest(r);validate_aupro(r,expected);st.write_json(dest,r)
        if r['verified_images']!=len(samples):raise ValueError('partial common-population AUPRO')
        done.append(r);pp.guard(a,'aupro',len(done),4)
    return dict(status='common_AUPRO_complete',jobs=len(done),**ip.signature(a))
