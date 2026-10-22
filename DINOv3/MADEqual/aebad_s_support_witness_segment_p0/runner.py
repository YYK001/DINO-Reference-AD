"""Explicit, non-chained stages; default preflight is read-only and Torch-free."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from . import inputs as ip,storage as st
from .method import LAYERS,NEW

STAGES=('preflight','recover-support','verify-support','prepare-support','support-loo','predict','evaluate','evaluate-aupro','summarize')


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',nargs='?',default='preflight',choices=STAGES)
    for k,v in ip.parent.defaults().items():p.add_argument('--'+k.replace('_','-'),type=Path,default=v)
    p.add_argument('--output-dir',type=Path,default=ip.parent.ROOT/'DINOv3/outputs/mad_equal/aebad_s_support_witness_fixed_p0_v1')
    p.add_argument('--support-root',action='append',type=Path,default=[])
    p.add_argument('--recovered-supports',type=Path,help='explicit independently verified new support provenance; never changes historical SHA')
    p.add_argument('--reuse-recovery',type=Path,help='verify-support: original sealed extraction, read only')
    p.add_argument('--recovery-policy',choices=('strict','historical-maps-v1'),default='strict')
    p.add_argument('--checkpoint',type=Path,help='recover-support only; never needed by later offline stages')
    p.add_argument('--dinov3',type=Path,default=ip.parent.ROOT/'external_baselines/dinov3')
    p.add_argument('--domains',nargs='+',choices=ip.DOMAINS,default=list(ip.DOMAINS))
    p.add_argument('--layers',nargs='+',type=int,choices=LAYERS,default=list(LAYERS))
    p.add_argument('--devices',nargs='+',default=['cpu'],help='prepare-support: cpu or explicit cuda:0 [cuda:1]; AUPRO requires CUDA')
    p.add_argument('--query-chunk',type=int,default=64);p.add_argument('--reference-chunk',type=int,default=1024)
    p.add_argument('--cpu-workers',type=int,default=2);p.add_argument('--limit',type=int)
    p.add_argument('--no-write',action='store_true',help='preflight already never creates experiment outputs')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--branch',choices=NEW,help=argparse.SUPPRESS)
    return p


def parallel_jobs(a,bundle,out,tasks):
    """Independent domain/layer or domain/branch processes, one lane per GPU.

    Explicit devices are honored; no RAM/VRAM/disk estimate admission and no
    automatic switch to CPU/single card on errors. Unaffected jobs are retained.
    """
    if len(a.devices)>2 or len(set(a.devices))!=len(a.devices) or any(not d.startswith('cuda:') for d in a.devices):raise ValueError('choose one or two distinct explicit CUDA devices')
    paths=[]
    for k in ip.parent.defaults():paths+=['--'+k.replace('_','-'),str(getattr(a,k))]
    for root in a.support_root:paths+=['--support-root',str(root)]
    if a.recovered_supports:paths+=['--recovered-supports',str(a.recovered_supports)]
    if a.checkpoint:paths+=['--checkpoint',str(a.checkpoint)]
    paths+=['--dinov3',str(a.dinov3)]
    def lane(item):
        lane_index,device=item
        for d,value in tasks[lane_index::len(a.devices)]:
            extra=['--layers',str(value)] if a.stage=='prepare-support' else [] if a.stage=='recover-support' else ['--branch',value]
            cmd=[sys.executable,'-B','-u','-m',__package__+'.runner',a.stage,*paths,'--output-dir',str(out),'--domains',d,
                '--devices',device,'--worker',*extra,'--query-chunk',str(a.query_chunk),'--reference-chunk',str(a.reference_chunk)]
            log=out/'logs'/a.stage/(d+'_'+str(value)+'.log');log.parent.mkdir(parents=True,exist_ok=True)
            print(f'[witness dispatch] {a.stage} {d}/{value} {device} log={log}',flush=True)
            with log.open('a',encoding='utf-8') as f:
                proc=subprocess.run(cmd,cwd=ip.parent.ROOT,stdout=f,stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            if proc.returncode:raise RuntimeError(f'worker failed ({proc.returncode}); preserved {log}')
    with ThreadPoolExecutor(max_workers=len(a.devices)) as pool:list(pool.map(lane,enumerate(a.devices)))
    return dict(status='requested_jobs_complete',jobs=len(tasks),devices=a.devices)


def resources():
    import psutil
    info=psutil.Process().memory_info();values=info._asdict()
    if os.name=='nt':peak=values.get('peak_wset',info.rss)
    else:
        import resource
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    result=dict(process_peak_rss_bytes=int(peak),process_current_rss_bytes=info.rss,
        capacity_admission_checks=False,model_inference=False,feature_extraction=False)
    # Observe an already initialized explicit GPU worker; never initialize CUDA
    # for a CPU stage or import Torch just to collect resource information.
    torch=sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():
        result.update(torch=torch.__version__,CUDA=torch.version.cuda,GPU=torch.cuda.get_device_name(),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    return result


def main(argv=None):
    a=parser().parse_args(argv);tick=time.perf_counter();bundle=None
    try:
        if a.limit is not None and a.limit<1:raise ValueError('positive prediction smoke limit required')
        if not 1<=a.cpu_workers<=8 or a.query_chunk<1 or a.reference_chunk<1:raise ValueError('invalid execution chunk/workers')
        a.domains=list(dict.fromkeys(a.domains));a.layers=list(dict.fromkeys(a.layers))
        bundle=ip.Bundle({k:getattr(a,k) for k in ip.parent.defaults()},
            recovered_supports=None if a.stage in ('recover-support','verify-support') else a.recovered_supports)
        roots=a.support_root or [ip.parent.ROOT/'downloads',ip.parent.ROOT/'server_archives',ip.parent.ROOT/'DINOv3/outputs/mad_equal']
        if a.recovered_supports:roots=[a.recovered_supports/'features',*a.support_root]
        if a.stage in ('recover-support','verify-support'):
            from . import recovery
            if a.recovered_supports is None:raise ValueError('explicit --recovered-supports independent new directory required')
            if a.stage=='recover-support':
                if a.checkpoint is None:raise ValueError('--checkpoint required only for explicit support extraction')
                spec,host_paths=recovery.specification(bundle,ip.parent.ROOT,a.checkpoint,a.dinov3)
                if a.no_write:
                    with recovery.Data(bundle.paths['data']) as data:
                        for d in a.domains:
                            for row in bundle.expected[d]:data.bytes(row['relative_path'],row['image_sha256'])
                    missing=bundle.parent.missing()
                    if missing:raise FileNotFoundError('original query caches missing: '+json.dumps(missing))
                    print(json.dumps(dict(status='recovery_inputs_present_not_runtime_test',supports=sum(len(bundle.expected[d]) for d in a.domains),
                        recovery_identity=spec['identity'],torch_imported='torch' in sys.modules,model_loaded=False,experiment_directory_created=False)),flush=True)
                    return 0
                if a.devices==['cpu']:raise ValueError('recover-support requires explicit --devices cuda:0 [cuda:1]')
                recovery.initialize(a.recovered_supports,spec,bundle)
                if len(a.devices)>1 and not a.worker:
                    result=parallel_jobs(a,bundle,Path(a.recovered_supports),[(d,'supports') for d in a.domains])
                else:result=recovery.extract(bundle,a.recovered_supports,spec,host_paths,a.domains,a.devices[0])
            else:
                if a.no_write:raise ValueError('verify-support writes only the independent recovery provenance')
                if a.recovery_policy=='historical-maps-v1':
                    from .compatibility import verify
                    if a.reuse_recovery is None:raise ValueError('explicit --reuse-recovery required for new historical certificate')
                    result=verify(bundle,a.recovered_supports,a.reuse_recovery,a.cpu_workers)
                else:
                    if a.reuse_recovery is not None:raise ValueError('reusing old extraction requires explicit --recovery-policy historical-maps-v1')
                    result=recovery.verify(bundle,a.recovered_supports)
            # Full attestations stay in JSON; never dump all629 parent records
            # as a single console line when a recovery certificate completes.
            keys=('status','images','new','reused','seconds','strict_distance_compatible','support_forward_calls','query_forward_calls','model_loaded','jobs','devices')
            summary={k:result[k] for k in keys if k in result}
            print(json.dumps(dict(stage=a.stage,status='requested_stage_complete',output=str(a.recovered_supports),result=summary),ensure_ascii=False),flush=True)
            return 0
        if a.stage=='preflight':
            report=ip.preflight(bundle,roots,a.domains,scan=True);print(json.dumps(report,ensure_ascii=False,indent=2));return 0 if not report['missing_inputs'] else 2
        if a.no_write:raise ValueError('--no-write only applies to preflight')
        locations={}
        if a.stage in ('prepare-support','support-loo','predict'):
            locations,available,missing,bad,_=ip.locate(bundle,roots,a.domains,scan=False)
            if missing:raise FileNotFoundError('complete historical support NPZ missing: '+json.dumps(missing,ensure_ascii=False))
            bundle.protected_paths.extend(Path(v.split('::',1)[0]) for v in locations.values())
        out=st.initialize(a.output_dir,bundle)
        if a.stage=='prepare-support':
            from . import supports
            supports.configure(out,a.devices[0],a.query_chunk,a.reference_chunk)
            if a.devices!=['cpu'] and not a.worker:
                result=parallel_jobs(a,bundle,out,[(d,l) for d in a.domains for l in a.layers])
            else:
                result=[supports.prepare(bundle,out,locations,d,l,device=a.devices[0],query_chunk=a.query_chunk,reference_chunk=a.reference_chunk) for d in a.domains for l in a.layers]
        elif a.stage=='support-loo':
            from .supports import support_loo
            result=[support_loo(bundle,out,locations,d,l) for d in a.domains for l in a.layers]
        elif a.stage=='predict':
            from . import prediction
            result=[]
            for d in a.domains:
                for l in a.layers:prediction.predict_layer(bundle,out,locations,d,l,a.limit)
                result.append(prediction.freeze_domain(bundle,out,d))
        elif a.stage=='evaluate':
            from .evaluation import evaluate
            result=[evaluate(bundle,out,d,a.cpu_workers) for d in a.domains]
        elif a.stage=='evaluate-aupro':
            if a.devices==['cpu']:raise ValueError('explicit --devices cuda:0 [cuda:1] required for original AUPRO')
            if a.worker:
                from .aupro import job
                if len(a.domains)!=1 or a.branch is None or len(a.devices)!=1:raise ValueError('one worker handles one domain/branch/device')
                result=job(bundle,out,a.domains[0],a.branch,a.devices[0])
            else:result=parallel_jobs(a,bundle,out,[(d,b) for d in a.domains for b in NEW])
        else:
            from .reporting import summarize
            result=summarize(bundle,out)
        record=dict(stage=a.stage,domains=a.domains,layers=a.layers,seconds=time.perf_counter()-tick,**resources(),
            torch_imported='torch' in sys.modules,timing_scope='offline stage, not end-to-end model inference',result=result)
        key=a.stage+'_'+('_'.join(a.domains))+('_'+str(a.layers[0]) if a.worker and a.stage=='prepare-support' else '')+('_'+a.branch if a.branch else '')
        st.write_json(out/'cost_records'/(key+'.json'),record)
        # Workers write distinct cost records. The coordinator alone aggregates.
        if not a.worker:st.write_json(out/'cost.json',{p.stem:st.read_json(p) for p in (out/'cost_records').glob('*.json')})
        print(json.dumps(dict(stage=a.stage,status='requested_stage_complete',output=str(out),seconds=record['seconds']),ensure_ascii=False),flush=True)
        return 0
    except Exception as exc:
        error=dict(status='stage_failed',stage=a.stage,error=f'{type(exc).__name__}: {exc}',seconds=time.perf_counter()-tick)
        print(json.dumps(error,ensure_ascii=False),flush=True)
        # Only an already initialized OWN directory receives a failure record.
        p=Path(a.output_dir)
        if bundle is not None and (p/'input_manifest.json').exists() and st.read_json(p/'input_manifest.json').get('identity')==bundle.identity and a.stage!='preflight':
            st.write_json(p/'failures'/(a.stage+'_'+('_'.join(a.domains))+'_'+str(time.time_ns())+'.json'),error)
        return 2


if __name__=='__main__':raise SystemExit(main())
