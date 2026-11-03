"""Manual preflight by default. No connections, installations or automatic GPU work."""
import argparse
import json
import sys
import subprocess
import threading
import time
from pathlib import Path
from . import inputs as ip,pipeline,reporting
st=ip.st


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',nargs='?',default='preflight',choices=('preflight','prepare','smoke','predict','evaluate','summarize','pack','all'))
    p.add_argument('--parent-run',type=Path,default=ip.ROOT/'server_results/aebad_target_loo_two_K8_20261030_013433')
    p.add_argument('--conditional-run',type=Path,default=ip.ROOT/'server_results/aebad_conditional_image_score_20261102_224335')
    p.add_argument('--raw-run',type=Path,default=ip.ROOT/'server_results/aebad_reference_residual_compact_v1_20261027_002907')
    p.add_argument('--asset-run',dest='asset_runs',type=Path,action='append',default=[])
    p.add_argument('--feature-root',dest='feature_roots',type=Path,action='append',default=[])
    for flag in ('feature-manifest','rms-stat'):p.add_argument('--'+flag,type=Path)
    p.add_argument('--run-root',type=Path,default=ip.ROOT/'DINOv3/outputs/mad_equal/aebad_reverse_capacity_preflight')
    p.add_argument('--cache-root',type=Path)
    p.add_argument('--data',type=Path,default=ip.ROOT/'datasets/AeBAD.zip')
    p.add_argument('--checkpoint',type=Path,default=ip.ROOT/'_external/dinov3_checkpoints/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth')
    p.add_argument('--official-root',type=Path,default=ip.ROOT/'external_baselines/SuperADD')
    p.add_argument('--dinov3-root',type=Path,default=ip.ROOT/'external_baselines/dinov3')
    p.add_argument('--device',default='cpu')
    p.add_argument('--cpu-workers',type=int,choices=range(1,9),default=8)
    p.add_argument('--extract-missing',action='store_true')
    p.add_argument('--allow-over-12h',action='store_true',help='use only after explicit user approval of an over-12h estimate')
    return p


def overlap(x,y):return x==y or x.is_relative_to(y) or y.is_relative_to(x)


def normalize(a):
    for k,v in vars(a).copy().items():
        if isinstance(v,Path):setattr(a,k,v.resolve())
        if isinstance(v,list) and v and isinstance(v[0],Path):setattr(a,k,[x.resolve() for x in v])
    a.cache_root=a.cache_root or a.run_root.with_name(a.run_root.name+'_cache')
    protected=[v for k,v in vars(a).items() if isinstance(v,Path) and k not in ('run_root','cache_root')]+a.asset_runs+a.feature_roots
    # Include inherited cache targets; no recursive filesystem search/deletion.
    for run in [a.raw_run,*a.asset_runs]:
        if (run/'inputs.json').exists():
            r=st.sealed_json(run/'inputs.json')
            for k in ('cache','cache_root'):
                if r.get(k):protected.append(Path(r[k]).resolve())
    if overlap(a.run_root,a.cache_root) or any(overlap(w,r) for w in (a.run_root,a.cache_root) for r in protected):
        raise ValueError('new output/cache must be disjoint from historical assets and each other')
    if a.device!='cpu' and (not a.device.startswith('cuda:') or not a.device[5:].isdigit()):raise ValueError('one explicit cpu or cuda:index device required')
    a._started=time.perf_counter();return a


class Resources:
    def __init__(self):self.peak=0;self.stop=threading.Event();self.thread=threading.Thread(target=self.sample,daemon=True)
    def sample(self):
        import psutil
        while not self.stop.is_set():
            process=psutil.Process();rss=0
            for p in [process,*process.children(recursive=True)]:
                try:rss+=p.memory_info().rss
                except psutil.Error:pass
            self.peak=max(self.peak,rss);self.stop.wait(.5)
    def __enter__(self):self.thread.start();return self
    def __exit__(self,*args):self.stop.set();self.thread.join()


def execute(a):
    if a.stage=='pack':return reporting.pack(a)
    if a.stage=='preflight':
        try:return ip.preflight(a)
        except (OSError,ValueError,KeyError) as e:
            result=dict(status='blocked',error=f'{type(e).__name__}: {e}',torch_imported='torch' in sys.modules)
            st.write_json(a.run_root/'preflight.json',result)
            st.write_csv(a.run_root/'missing_inputs.csv',[dict(role='metadata_or_identity',reason=result['error'])])
            return result
    ip.freeze(a)
    if a.stage in ('prepare','smoke','predict'):
        if a.stage=='predict':
            smoke=st.sealed_json(a.run_root/'smoke.json')
            if smoke['identity']!=ip.signature(a):raise ValueError('current backend/input real-asset smoke required')
            pipeline.budget(a,pipeline.estimate(a,smoke))
        return getattr(pipeline,a.stage)(a)
    if a.stage=='evaluate':return pipeline.evaluate(a)
    return reporting.summarize(a)


def main(argv=None):
    a=normalize(parser().parse_args(argv));a.run_root.mkdir(parents=True,exist_ok=True)
    if a.stage=='all':
        ip.freeze(a)
        if not (a.run_root/'smoke.json').is_file():raise ValueError('explicit real-asset smoke and cost estimate required before all')
        smoke=st.sealed_json(a.run_root/'smoke.json')
        if smoke['identity']!=ip.signature(a):raise ValueError('smoke input/backend changed')
        pipeline.budget(a,pipeline.estimate(a,smoke))
        common=list(sys.argv[2:] if argv is None else argv[1:])
        for stage in ('prepare','predict','evaluate','summarize','pack'):
            log=a.run_root/'logs'/(stage+'.log');log.parent.mkdir(parents=True,exist_ok=True)
            st.write_json(a.run_root/'pipeline_status.json',dict(status='running',stage=stage,success=False))
            print(f'STAGE={stage} LOG={log}',flush=True)
            with log.open('a',encoding='utf-8') as f:
                child=subprocess.Popen([sys.executable,'-B','-u','-m',__package__+'.runner',stage,*common],stdout=f,stderr=subprocess.STDOUT)
                while True:
                    try:rc=child.wait(timeout=30);break
                    except subprocess.TimeoutExpired:print(f'[heartbeat] coverage stage={stage}',flush=True)
            print(f'STAGE={stage} EXIT_CODE={rc}',flush=True)
            if rc:
                st.write_json(a.run_root/'pipeline_status.json',dict(status='stage_failed',stage=stage,success=False,exit_code=rc))
                print(json.dumps(reporting.pack(a)),flush=True);return rc
        status=st.read_json(a.run_root/'status.json');ok=not status['pending']
        st.write_json(a.run_root/'pipeline_status.json',dict(status='complete' if ok else 'pending',success=ok,stage='pack'))
        return 0 if ok else 2
    tick=time.perf_counter()
    with Resources() as resource:
        try:result=execute(a)
        except Exception as e:
            failure=dict(stage=a.stage,error=f'{type(e).__name__}: {e}')
            st.write_json(a.run_root/'stage_failure.json',failure)
            st.write_json(a.run_root/'costs'/(a.stage+'_failed_'+str(time.time_ns())+'.json'),dict(
                seconds=time.perf_counter()-tick,peak_process_tree_RSS_bytes=resource.peak,
                feature_forwards=getattr(a,'feature_forwards',0),**failure))
            raise
    record=dict(stage=a.stage,seconds=time.perf_counter()-tick,peak_process_tree_RSS_bytes=resource.peak,
        feature_forwards=getattr(a,'feature_forwards',0),torch_imported='torch' in sys.modules,result=result)
    from DINOv3.MADEqual.aebad_s_normal_correspondence_adaptation import storage as payload_storage
    record['new_array_payload_bytes']=payload_storage.NEW_ARRAY_BYTES
    if 'torch' in sys.modules:
        torch=sys.modules['torch']
        if torch.cuda.is_initialized():record.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    st.write_json(a.run_root/'costs'/(a.stage+'_'+str(time.time_ns())+'.json'),record)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return 2 if result['status'] in ('missing_inputs','pending','blocked') else 0


if __name__=='__main__':raise SystemExit(main())
