"""Default CPU-only preflight. All GPU stages require explicit manual invocation."""
import argparse,json,os,sys,time,tarfile,threading,subprocess
from pathlib import Path
from . import inputs as ip
from .inputs import st
from .protocol import STAGES


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',nargs='?',default='preflight',choices=('preflight',*STAGES,'all','pack'))
    p.add_argument('--parent-run',type=Path,required=True);p.add_argument('--parent-cache',type=Path,required=True)
    p.add_argument('--raw-run',type=Path);p.add_argument('--raw-cache',type=Path);p.add_argument('--source-bank',type=Path)
    p.add_argument('--run-root',type=Path,default=ip.ROOT/'DINOv3/outputs/mad_equal/aebad_s_blind_context_seed42_v1')
    p.add_argument('--cache-root',type=Path);p.add_argument('--evaluation-scratch',type=Path);p.add_argument('--inventory-dir',type=Path)
    p.add_argument('--data',type=Path,default=ip.ROOT/'datasets/AeBAD.zip')
    p.add_argument('--checkpoint',type=Path,default=ip.ROOT/'external_baselines/SuperADD/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth')
    p.add_argument('--official-root',type=Path,default=ip.ROOT/'external_baselines/SuperADD');p.add_argument('--dinov3-root',type=Path,default=ip.ROOT/'external_baselines/dinov3')
    p.add_argument('--cpu-workers',type=int,choices=range(1,9),default=8);p.add_argument('--device',default='cuda:0')
    p.add_argument('--limit-normal',type=int,choices=range(1,73),help='prepare-context only: real normal throughput, no dictionary/train/prediction completion')
    return p


def overlap(a,b):return a==b or a.is_relative_to(b) or b.is_relative_to(a)


def isolation(a):
    protected=[x for x in (a.parent_run,a.parent_cache,a.raw_run,a.raw_cache,a.source_bank,a.data,a.checkpoint,a.official_root,a.dinov3_root) if x is not None]
    if hasattr(a,'_parent'):protected.append(Path(a._parent.assets['references']['source']['path']).resolve())
    for dest in (a.run_root,a.cache_root,a.evaluation_scratch,a.inventory_dir):
        if dest is not None and any(overlap(dest,p) for p in protected):raise ValueError('output/scratch overlaps read-only asset')
    if overlap(a.cache_root,a.run_root) or overlap(a.evaluation_scratch,a.run_root) or overlap(a.cache_root,a.evaluation_scratch):raise ValueError('new run, cache and scratch must be separate directories')


def normalize(a):
    for k,v in vars(a).copy().items():
        if isinstance(v,Path):setattr(a,k,v.resolve())
    a.cache_root=(a.cache_root or a.run_root.with_name(a.run_root.name+'_cache')).resolve()
    a.evaluation_scratch=(a.evaluation_scratch or a.run_root.with_name(a.run_root.name+'_pixels')).resolve()
    if a.limit_normal and a.stage!='prepare-context':raise ValueError('--limit-normal only measures prepare-context, not a reduced official experiment')
    isolation(a);return a


def pack(a):
    archive=a.run_root.with_name(a.run_root.name+'_review.tar.gz')
    with tarfile.open(archive,'w:gz',compresslevel=1) as t:
        for root,dirs,files in os.walk(a.run_root,followlinks=False):
            dirs[:]=[d for d in dirs if not (Path(root)/d).is_symlink() and d!='model']
            for name in files:
                f=Path(root)/name
                if f.suffix in ('.json','.csv','.md','.log','.txt','.png') and not f.is_symlink():t.add(f,arcname=a.run_root.name+'/'+f.relative_to(a.run_root).as_posix(),recursive=False)
    return dict(archive=str(archive),bytes=archive.stat().st_size,large_arrays_checkpoints_scratch_excluded=True)


def execute(a):
    if a.stage=='prepare-context':
        from .runtime import prepare_context
        return prepare_context(a)
    ip.validate(a);isolation(a)
    if a.stage=='check-isolation':
        from .isolation import check
        return check(a)
    if a.stage in ('prepare-training','train'):
        from . import training
        return training.prepare(a) if a.stage=='prepare-training' else training.train(a)
    if a.stage=='predict':
        from .prediction import predict
        return predict(a)
    if a.stage in ('evaluate','aupro'):
        from . import evaluation
        return evaluation.evaluate(a) if a.stage=='evaluate' else evaluation.aupro(a)
    from .reporting import summarize
    return summarize(a)


def cost(a,tick,result):
    import psutil
    process=psutil.Process();mem=process.memory_info()
    try:
        import resource
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    except ImportError:peak=getattr(mem,'peak_wset',mem.rss)
    gpu={};torch=sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():gpu=dict(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    st.write_json(a.run_root/'costs'/f'{a.stage}_{time.time_ns()}.json',dict(stage=a.stage,seconds=time.perf_counter()-tick,process_peak_rss_bytes=peak,
        full_four_layer_extractions=getattr(a,'_full_forwards',0),host_internal_crop_forwards=getattr(a,'_crop_forwards',0),
        new_original_extractions=getattr(a,'_original_calls',0),**gpu,result=result))


def all_stages(a):
    tick=time.perf_counter();stage='preflight';success=False;error=None
    try:
        inventory=ip.preflight(a)
        if inventory['status']=='blocked':show(a,inventory);raise ValueError('missing inputs; no bank rebuild')
        # Repeat final isolation with TRAINED heads, before ANY evaluation RGB is used.
        for stage in (*STAGES[:4],'check-isolation',*STAGES[4:]):
            log=a.run_root/'logs'/(stage+'.log');log.parent.mkdir(parents=True,exist_ok=True)
            st.write_json(a.run_root/'pipeline_status.json',dict(status='running',stage=stage,success=False))
            args=[sys.executable,'-B','-u','-m',__spec__.name,stage]
            for k,v in vars(a).items():
                if k.startswith('_') or k in ('stage','limit_normal') or v is None:continue
                args.extend(('--'+k.replace('_','-'),str(v)))
            stop=threading.Event();start=time.perf_counter()
            def heartbeat():
                while not stop.wait(30):print(f'[heartbeat] stage={stage} seconds={time.perf_counter()-start:.0f}',flush=True)
            thread=threading.Thread(target=heartbeat,daemon=True);thread.start();print(f'STAGE={stage} LOG={log}',flush=True)
            try:
                with log.open('a',encoding='utf-8') as f:
                    p=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
                    try:
                        for line in p.stdout:print(line,end='',flush=True);f.write(line);f.flush()
                        rc=p.wait()
                    except BaseException:p.terminate();p.wait();raise
                if rc:raise RuntimeError(f'{stage} failed: exit_code={rc}')
            finally:stop.set();thread.join()
        success=st.read_json(a.run_root/'status.json')['status']=='all_five_metrics_complete'
        if not success:raise ValueError('pending evaluation')
        return dict(success=True,seconds=time.perf_counter()-tick)
    except BaseException as e:error=type(e).__name__+': '+str(e);raise
    finally:
        st.write_json(a.run_root/'pipeline_status.json',dict(status='complete' if success else 'failed',stage=stage,success=success,error=error,seconds=time.perf_counter()-tick))
        print(json.dumps(pack(a)),flush=True)


def show(a,value):
    if a.inventory_dir:
        st.write_json(a.inventory_dir/'preflight.json',value);st.write_csv(a.inventory_dir/'missing_inputs.csv',value['missing'])
        st.write_csv(a.inventory_dir/'dictionary_images.csv',value.get('dictionary_images',[]))
        if hasattr(a,'_parent'):st.write_csv(a.inventory_dir/'evaluation_images.csv',a._parent.samples)
        from .protocol import CONFIG,job_plan
        st.write_json(a.inventory_dir/'fixed_config.json',CONFIG);st.write_json(a.inventory_dir/'job_plan.json',job_plan())
    out={k:v for k,v in value.items() if k not in ('dictionary_images','reusable_normal_features','original_forwards_needed','reusable_original_queries','query_original_forwards_needed')}
    out['reusable_normal_features_count']=len(value.get('reusable_normal_features',[]));out['new_original_normal_forwards']=len(value.get('original_forwards_needed',[]))
    out['reusable_original_query_count']=len(value.get('reusable_original_queries',[]));out['new_original_query_forwards']=len(value.get('query_original_forwards_needed',[]))
    out['missing_count']=len(value['missing']);out['missing']=value['missing'][:12];print(json.dumps(out,ensure_ascii=False,indent=2))


def main(argv=None):
    a=normalize(parser().parse_args(argv));st.configure('fast')
    if a.stage=='preflight':
        r=ip.preflight(a);show(a,r);return 2 if r['status']=='blocked' else 0
    if a.stage=='pack':r=pack(a)
    elif a.stage=='all':r=all_stages(a)
    else:
        tick=time.perf_counter();r=execute(a);cost(a,tick,r)
    print(json.dumps(r,ensure_ascii=False,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
