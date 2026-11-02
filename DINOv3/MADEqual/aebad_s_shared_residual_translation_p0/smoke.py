"""Explicit authorized full-image path benchmark; outputs are reusable predictions."""
import time
import numpy as np
from . import inputs as ip,prediction as pred,runtime as rt,numerics as nm
from .inputs import st
from .protocol import *


def elapsed(a):
    return sum(float(st.read_json(p)['seconds']) for p in (a.run_root/'costs').glob('*.json'))+time.perf_counter()-a._started


def observe(a):
    if not a.allow_over_12h and elapsed(a)>43200:
        st.write_json(a.run_root/'budget_pause.json',dict(status='pending_12h_consent',elapsed_seconds=elapsed(a)))
        raise RuntimeError('12h consent boundary reached; committed units preserved; explicit consent required to continue')


def launch_check(a):
    backend=ip.frozen_backend(a,compute=True)
    if a.allow_over_12h:return
    report=st.read_json(a.run_root/'time_estimate.json')
    if report.get('status')!='full_image_smoke_estimate' or report.get('identity')!=ip.signature(a,DOMAINS[0])['identity']:
        raise RuntimeError('run the explicitly authorized three-domain full-image smoke first; no unmeasured launch promise')
    if report.get('prediction_backend')!=backend:raise ValueError('smoke prediction backend differs or is unbound; new backend smoke required')
    # Recompute pending counts from committed units; a resumed run must not pay
    # for completed predictions/evaluation again in the projection.
    remaining=0.
    for sample in report['samples']:
        d=sample['domain'];count=sum(not ip.record_path(a,d,r).exists() for r in ip.parent(a).samples(d))
        remaining+=sample['seconds']*count
    from . import evaluation as ev
    cpu_jobs=sum(ev.completed(a,d,b) is None for d in DOMAINS for b in MAPS)
    au_jobs=sum(not (a.run_root/d/'aupro'/(b+'.json')).exists() for d in DOMAINS for b in MAPS)
    if a.stage!='predict':
        if report['stages'].get('CPU_exact') is not None:remaining+=report['stages']['CPU_exact']*cpu_jobs/9
        if report['stages'].get('AUPRO') is not None:remaining+=report['stages']['AUPRO']*au_jobs/9
    if elapsed(a)+remaining>43200:
        raise RuntimeError('projected known cost exceeds12h; obtain explicit consent before --allow-over-12h')


def run(a):
    backend=ip.frozen_backend(a,compute=True)
    old=a.run_root/'time_estimate.json'
    if old.exists():
        r=st.read_json(old)
        if r.get('status')=='full_image_smoke_estimate' and r.get('identity')==ip.signature(a,DOMAINS[0])['identity']:
            if r.get('prediction_backend')!=backend:raise ValueError('smoke prediction backend differs or is unbound; cannot reuse')
            for sample in r['samples']:
                row=next(x for x in ip.parent(a).samples(sample['domain']) if x['sample_id']==sample['sample_id'])
                pred.verify(a,sample['domain'],row)
            return dict(r,reused_smoke=True)
    samples=[];checks=[];phases=[]
    for d in DOMAINS:
        # Reuse an already committed full-path benchmark after interruption.
        unit=a.run_root/'smoke_units'/(d+'.json')
        if unit.exists():
            sample=st.sealed_json(unit)
            if sample['identity']!=ip.signature(a,d)['identity'] or sample.get('prediction_backend')!=backend:raise ValueError('smoke inputs/backend differ')
            row=next(r for r in ip.parent(a).samples(d) if r['sample_id']==sample['sample_id'])
            pred.verify(a,d,row);samples.append(sample);continue
        # Fixed manifest order, never select by score or GT. If an image was
        # completed by an interrupted non-benchmark step, benchmark the first
        # pending complete image rather than forward the finished image again.
        population=ip.parent(a).samples(d)
        row=next((r for r in population if not ip.record_path(a,d,r).exists()),population[0])
        if ip.record_path(a,d,row).exists():
            rec,_,_=pred.verify(a,d,row)
            timing=st.read_json(a.run_root/d/'timings'/(row['sample_id']+'.json'))
            samples.append(dict(domain=d,sample_id=row['sample_id'],**timing,prediction_backend=backend,reused_prediction=True));continue
        forwards_before=getattr(a,'feature_forwards',0)
        begin=time.perf_counter();refs=rt.References(a,d);startup=time.perf_counter()-begin
        # Validate stock tensor arithmetic on actual vectors, CPU only reference;
        # no GT/label used. Actual full-image run below shares the same retrieval.
        q,geometry,feature=rt.features(a,row,persist=False)
        nearest,r,_=refs.nearest(q,geometry,feature,row)
        acquisition=time.perf_counter()-begin-startup
        # Avoid a duplicate model forward for the timed full-image prediction.
        original=rt.features
        def once(aa,rr,persist=False):
            if rr==row:return q,geometry,feature
            return original(aa,rr,persist=persist)
        order=nm.order(geometry['positions'],geometry['grid']);no=nm.order(nearest['positions'],geometry['grid'])
        for l in LAYERS:
            banks,keys=refs.banks(l);qq=np.array(q[l][order],np.float32)
            ii=np.asarray(nearest[f'index{l}'][:,no]);dd=np.asarray(nearest[f'distance{l}'][:,no])
            # Small CPU reference uses ALL physical positions for the same means;
            # a patch-subset mean would define a different method.
            expected,es,_=nm.translate(qq,banks,ii,dd,chunk=a.vector_chunk)
            actual,ts,_=rt.translate(a,qq,banks,ii,dd,refs,keys)
            left={**expected,**{k:es[k] for k in ('zero_fixed','zero_reselect')}}
            right={**actual,**{k:ts[k] for k in ('zero_fixed','zero_reselect')}}
            for b in left:
                err=float(np.abs(left[b]-right[b]).max());ok=bool(np.allclose(left[b],right[b],atol=1e-7,rtol=1e-6))
                checks.append(dict(domain=d,layer=l,branch=b,max_abs_error=err,atol=1e-7,rtol=1e-6,verified=ok))
        st.write_csv(a.run_root/'smoke_numerical_checks.csv',checks)
        if not all(v['verified'] for v in checks):raise ValueError('stock FP32 full-image translation reference failed; no relaxed tolerance')
        # Timed path includes actual feature/Exact cold cost from begin, plus
        # gather/correction/write; excludes CPU-vs-GPU validation cost explicitly.
        correction_start=time.perf_counter()
        try:
            rt.features=once;pred.one(a,d,row,refs)
        finally:rt.features=original
        correction=time.perf_counter()-correction_start
        # Cold acquisition was measured separately; only this work once. Detailed
        # reference validation remains recorded but is NOT multiplied by902.
        validation=correction_start-begin-startup-acquisition
        sample=dict(identity=ip.signature(a,d)['identity'],prediction_backend=backend,domain=d,sample_id=row['sample_id'],startup_seconds=startup,
            acquisition_seconds=acquisition,reference_validation_seconds=validation,correction_and_write_seconds=correction,
            seconds=startup+acquisition+correction,reused_prediction=False,
            feature_forwards=getattr(a,'feature_forwards',0)-forwards_before,exact_execution=dict(refs.search.stats),
            caveat='cold full-image throughput; validation not extrapolated; same completed prediction reused by predict')
        samples.append(st.sealed_json(unit,sample))
    # Preserve measured first-image costs on repeated smoke. No model/retrieval
    # rerun merely to make a new duration number.
    prediction_estimate=sum(float(r['seconds'])*sum(not ip.record_path(a,r['domain'],row).exists() for row in ip.parent(a).samples(r['domain'])) for r in samples)
    prior=st.read_json(a.run_root/'time_estimate.json') if (a.run_root/'time_estimate.json').exists() else {}
    phases=dict(feature_retrieval_translation_write=prediction_estimate,CPU_exact=None,AUPRO=None,report_pack=None)
    if a.benchmark_metrics:
        # Surrogate masks only, prior to global freeze. Formal GT is NEVER read.
        # Results discarded; only seconds retained. Same original kernels.
        from DINOv3.MADEqual.aebad_s_support_witness_segment_p0 import evaluation as backend
        from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic import metrics
        from DINOv3.MADEqual.aebad_s_support_distance_crossdomain.resource_policy import observation_only
        row=ip.parent(a).samples(DOMAINS[0])[0];im,_,_=pred.replay(a,DOMAINS[0],row,C)
        yy=np.zeros(im.shape,np.uint8);h,w=im.shape;yy[h//3:2*h//3,w//3:2*w//3]=1
        tick=time.perf_counter()
        with observation_only():metrics.exact_binary(yy.ravel(),im.ravel(),memory_budget_bytes=0)
        cpu=time.perf_counter()-tick
        # N log N sort extrapolation, explicit surrogate rather than timing tiny
        # constructed features or treating a whole domain as one image.
        phases['CPU_exact']=sum(3*cpu*(sum((r['image_height']//4)*(r['image_width']//4) for r in ip.parent(a).samples(d))/im.size)*
            (np.log2(sum((r['image_height']//4)*(r['image_width']//4) for r in ip.parent(a).samples(d)))/np.log2(im.size)) for d in DOMAINS)
        if a.device.startswith('cuda:'):
            from importlib.metadata import version
            if version('adeval')!='1.1.0':raise ValueError('ADEval1.1.0 required')
            import torch
            from DINOv3.relation_reliability.official_metrics import official_aupro_adeval
            torch.cuda.set_device(torch.device(a.device));tick=time.perf_counter()
            official_aupro_adeval([yy],[im],nstrips=200);au=time.perf_counter()-tick
            phases['AUPRO']=3*au*sum((r['image_height']//4)*(r['image_width']//4) for d in DOMAINS for r in ip.parent(a).samples(d))/im.size
    total=sum(v for v in phases.values() if v is not None)
    result=dict(status='full_image_smoke_estimate',identity=ip.signature(a,DOMAINS[0])['identity'],prediction_backend=backend,samples=samples,
        stages=phases,known_seconds=total,total_seconds=None if any(v is None for v in phases.values()) else total,
        unknown_stages=[k for k,v in phases.items() if v is None],
        interpretation='cold first-image extrapolation, conservative; CPU NlogN and AUPRO pixel scaling only if explicitly measured with synthetic masks; actual GT component count remains uncertain',
        unknown_is_not_zero=True,over_12h=total>43200,previous_unmeasured_resource_plan=prior.get('RAM_GiB_planning_range'))
    st.write_json(a.run_root/'time_estimate.json',result)
    return result
