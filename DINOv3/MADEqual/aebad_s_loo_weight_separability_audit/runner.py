"""CPU offline audit; default inventory, explicit AUPRO, no model entrypoints."""
import argparse,json,os,sys,time,subprocess,threading,tarfile
from pathlib import Path
from . import inputs as ip
from .inputs import st
from .protocol import *


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',nargs='?',default='preflight',choices=('preflight','export-responses','smoke','predict','smoke-gt','evaluate','analyze','aupro','summarize','pack','all-cpu','all'))
    p.add_argument('--parent-run',type=Path,default=ip.ROOT/'server_results/aebad_target_loo_two_K8_20261030_013433')
    p.add_argument('--output-root',type=Path,default=ip.ROOT/'DINOv3/outputs/mad_equal/aebad_s_loo_weight_separability_audit_v1')
    for name in ('response-root','memory42-root','memory43-root','evaluation-scratch','export-root'):p.add_argument('--'+name,type=Path)
    p.add_argument('--data',type=Path,default=ip.ROOT/'datasets/AeBAD.zip');p.add_argument('--device',default='cuda:0')
    p.add_argument('--domains',nargs='+',choices=DOMAINS,default=list(DOMAINS));p.add_argument('--memory-seeds',nargs='+',type=int,choices=SEEDS,default=list(SEEDS))
    p.add_argument('--cpu-workers',type=int,choices=range(1,9),default=8);p.add_argument('--allow-over-12h',action='store_true')
    return p


def normalize(a):
    for k,v in vars(a).copy().items():
        if isinstance(v,Path):setattr(a,k,v.resolve())
    a.evaluation_scratch=(a.evaluation_scratch or a.output_root.with_name(a.output_root.name+'_pixels')).resolve()
    protected=[x for x in (a.parent_run,a.response_root,a.memory42_root,a.memory43_root,a.data) if x]
    def overlap(x,y):return x==y or x.is_relative_to(y) or y.is_relative_to(x)
    if overlap(a.output_root,a.evaluation_scratch) or any(overlap(x,y) for x in (a.output_root,a.evaluation_scratch) for y in protected):raise ValueError('new output/scratch must be separate from parent/response/data inputs')
    if a.export_root and any(overlap(a.export_root,y) for y in protected):raise ValueError('export must not overlap read-only input paths')
    if len(set(a.domains))!=len(a.domains) or len(set(a.memory_seeds))!=len(a.memory_seeds):raise ValueError('duplicate requested setting')
    return a


def pack(a):
    dest=a.output_root.with_name(a.output_root.name+'_review.tar.gz')
    with tarfile.open(dest,'w:gz',compresslevel=1) as t:
        for root,dirs,files in os.walk(a.output_root,followlinks=False):
            dirs[:]=[d for d in dirs if d not in ('scratch','pixel_scratch') and not (Path(root)/d).is_symlink()]
            for name in files:
                f=Path(root)/name
                if not f.is_symlink() and (f.suffix in ('.json','.csv','.md','.txt','.log','.png') or ('pair_records' in f.parts and f.suffix=='.npy')):t.add(f,arcname=a.output_root.name+'/'+f.relative_to(a.output_root).as_posix(),recursive=False)
    return dict(archive=str(dest),bytes=dest.stat().st_size,compact_pair_arrays_included=True,large_Z_maps_and_scratch_excluded=True)


def execute(a):
    if a.stage=='preflight':return ip.preflight(a)
    if a.stage=='pack':return pack(a)
    if a.stage=='export-responses':return ip.export_responses(a)
    ip.freeze(a)
    if a.stage in ('smoke','smoke-gt'):
        from .smoke import run
        return run(a,with_gt=a.stage=='smoke-gt')
    if a.stage=='predict':
        from .prediction import freeze
        return freeze(a)
    if a.stage in ('evaluate','aupro'):
        from . import evaluation
        return getattr(evaluation,a.stage)(a)
    if a.stage=='analyze':
        from .analysis import analyze
        return analyze(a)
    from .reporting import summarize
    return summarize(a)


def costs(a,tick,status):
    import psutil
    from importlib.metadata import version,PackageNotFoundError
    mem=psutil.Process().memory_info();peak=getattr(mem,'peak_wset',mem.rss)
    try:
        import resource
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    except ImportError:pass
    record=dict(stage=a.stage,status=status,seconds=time.perf_counter()-tick,process_peak_rss_bytes=peak,feature_forwards=0,Exact_searches=0,weight_fit_calls=0,training=0)
    software={}
    for name in ('numpy','scipy','Pillow','scikit-learn','adeval'):
        try:software[name]=version(name)
        except PackageNotFoundError:software[name]=None
    record['software']=software
    torch=sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():record.update(peak_cuda_allocated_bytes=max(torch.cuda.max_memory_allocated(),getattr(a,'peak_cuda_allocated_bytes',0)),peak_cuda_reserved_bytes=max(torch.cuda.max_memory_reserved(),getattr(a,'peak_cuda_reserved_bytes',0)))
    st.write_json(a.output_root/'costs'/f'{a.stage}_{time.time_ns()}.json',record)
    rows=[st.read_json(p) for p in sorted((a.output_root/'costs').glob('*.json'))];st.write_csv(a.output_root/'costs.csv',rows)


def orchestrate(a,tokens):
    stages=['preflight','smoke','predict','smoke-gt','evaluate','analyze']+(['aupro'] if a.stage=='all' else [])+['summarize','pack']
    for stage in stages:
        log=a.output_root/'logs'/(stage+'.log');log.parent.mkdir(parents=True,exist_ok=True)
        st.write_json(a.output_root/'pipeline_status.json',dict(status='running',stage=stage));print(f'STAGE={stage} LOG={log}',flush=True)
        with log.open('a',encoding='utf-8') as out,subprocess.Popen([sys.executable,'-B','-u','-m',__package__+'.runner',stage,*tokens],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1) as child:
            for line in child.stdout:print(line,end='',flush=True);out.write(line);out.flush()
            rc=child.wait()
        if rc and not (stage=='summarize' and a.stage=='all-cpu' and rc==2):
            st.write_json(a.output_root/'pipeline_status.json',dict(status='failed',stage=stage,exit_code=rc));print(json.dumps(pack(a)));return rc
    st.write_json(a.output_root/'pipeline_status.json',dict(status='CPU_complete_AUPRO_pending' if a.stage=='all-cpu' else 'complete',stage='pack'));return 0


def main(argv=None):
    tokens=list(sys.argv[1:] if argv is None else argv);a=normalize(parser().parse_args(tokens));a._started=time.perf_counter();a.output_root.mkdir(parents=True,exist_ok=True)
    if a.stage in ('all','all-cpu'):return orchestrate(a,tokens[1:])
    stop=threading.Event();tick=time.perf_counter();status='failed'
    def heartbeat():
        while not stop.wait(30):print(f'[heartbeat] audit stage={a.stage} seconds={time.perf_counter()-tick:.0f}',flush=True)
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    try:
        r=execute(a);print(json.dumps(r,ensure_ascii=False,indent=2));status='complete'
        return 2 if r.get('status') in ('missing_inputs','blocked','pending') else 0
    finally:
        stop.set();thread.join()
        if a.stage not in ('preflight','pack','export-responses'):costs(a,tick,status)


if __name__=='__main__':raise SystemExit(main())
