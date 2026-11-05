"""Default: metadata inventory. Explicit stages only after user authorization."""
import argparse
import contextlib
import json
import sys
import threading
import time
from pathlib import Path
from . import inputs as ip,pipeline,runtime as rt
from DINOv3.MADEqual.aebad_s_joint_reference_response_p0.runner import pack,overlap
from DINOv3.MADEqual.aebad_s_shared_residual_translation_p0.runner import cost
st=ip.st
STAGES=('preflight','check-cuda','build-index','smoke','calibrate','infer','evaluate','report','pack','all')

def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',nargs='?',default='preflight',choices=STAGES)
    defaults=dict(parent_run=ip.ROOT/'server_results/aebad_target_loo_two_K8_20261030_013433',
        conditional_run=ip.ROOT/'server_results/aebad_conditional_image_score_20261102_224335',
        raw_run=ip.ROOT/'server_results/aebad_reference_residual_compact_v1_20261027_002907',
        data=ip.ROOT/'datasets/AeBAD.zip',run_root=ip.ROOT/'DINOv3/outputs/mad_equal/aebad_annular_ordinal_cuda_preflight_v1',cache_root=None)
    for name,default in defaults.items():p.add_argument('--'+name.replace('_','-'),type=Path,default=default)
    p.add_argument('--cpu-workers',type=int,choices=range(1,9),default=8)
    p.add_argument('--backend',choices=('cuda','cpu'),default='cuda')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--window-batch',type=int,choices=range(1,65),default=64)
    p.add_argument('--index-shard-rows',type=int,default=16777216)
    p.add_argument('--candidate-tile',type=int,default=262144)
    p.add_argument('--index-cache-gib',type=float,default=24.)
    p.add_argument('--image-cache-gib',type=float,default=4.)
    p.add_argument('--geometry-groups',type=int,default=8)
    p.add_argument('--hypothesis-batch',type=int,default=16)
    p.add_argument('--writer-depth',type=int,default=2)
    p.add_argument('--prefetch-depth',type=int,default=2)
    p.add_argument('--allow-over-12h',action='store_true',help='only after explicit user consent; never changes the method')
    return p

def normalize(a):
    for name,value in vars(a).copy().items():
        if isinstance(value,Path):setattr(a,name,value.resolve())
    a.cache_root=a.cache_root or a.run_root.with_name(a.run_root.name+'_cache')
    if min(a.index_shard_rows,a.candidate_tile,a.geometry_groups,a.hypothesis_batch,a.writer_depth,a.prefetch_depth,a.index_cache_gib,a.image_cache_gib)<=0:
        raise ValueError('positive engineering batch/cache sizes required')
    if a.index_shard_rows>=2**31:raise ValueError('GPU CSR row offsets must fit int32')
    if a.backend=='cuda' and (not a.device.startswith('cuda:') or not a.device[5:].isdigit()):raise ValueError('one explicit CUDA device required')
    inputs=[a.parent_run,a.conditional_run,a.raw_run,a.data]
    # Protect actual parent assets too, including paths routed through symlinks.
    if (a.parent_run/'inputs.json').exists():
        parent=st.sealed_json(a.parent_run/'inputs.json')
        inputs += [Path(parent[k]).resolve() for k in ('parent_cache','raw_cache') if parent.get(k)]
        if parent.get('source',{}).get('path'):inputs.append(Path(parent['source']['path']).resolve())
    if overlap(a.run_root,a.cache_root) or any(overlap(out,old) for out in (a.run_root,a.cache_root) for old in inputs):raise ValueError('new output and cache must be disjoint from each other and all input assets')
    a.run_root.mkdir(parents=True,exist_ok=True);return a

def execute(a):
    if a.stage=='preflight':return ip.preflight(a)
    if a.stage=='pack':return pack(a)
    x=ip.assets(a)
    if x.missing:raise ValueError('compatible Source+K8 P missing; run preflight, restore records; no automatic baseline rebuild')
    ip.freeze(a)
    if a.stage=='check-cuda':
        if a.backend!='cuda':raise ValueError('check-cuda requires explicit CUDA backend')
        return rt.engine(a).verify()
    if a.stage=='build-index':
        if a.backend=='cuda':rt.engine(a).verify()
        r=rt.engine(a).store.build()
        return dict(status='normal_index_complete',normal_images=463,groups=len(r['groups']),
            descriptor_bytes=r['descriptor_bytes'],native_index_bytes=r['native_index_bytes'],
            reused_descriptor_images=r['reused_descriptor_images'],fresh_descriptor_images=r['fresh_descriptor_images'],
            source_vector_copy_bytes=r['source_vector_copy_bytes'],native_version=r['native_version'],seconds=r['seconds'],index_record=r['seal'])
    if a.stage=='smoke' and a.backend=='cuda':rt.engine(a).verify()
    return getattr(pipeline,a.stage)(a)

class Tee:
    def __init__(self,*streams):self.streams=streams
    def write(self,text):
        for stream in self.streams:stream.write(text);stream.flush()
    def flush(self):
        for stream in self.streams:stream.flush()

def stage(a):
    tick=time.perf_counter();done=threading.Event()
    def heartbeat():
        while not done.wait(30):print(f'[heartbeat] annular stage={a.stage} seconds={time.perf_counter()-tick:.0f}',flush=True)
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    try:
        result=execute(a);cost(a,tick,result)
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
        if result.get('status') in ('blocked','missing_inputs','pending') or result.get('valid_multi_evidence') is False:return 2
        return 0
    except Exception as e:
        cost(a,tick,dict(status='stage_failed',error=f'{type(e).__name__}: {e}'))
        st.write_json(a.run_root/'status.json',dict(status='incomplete',stage=a.stage,error=str(e),success=False));raise
    finally:done.set();thread.join()

def main(argv=None):
    a=normalize(parser().parse_args(argv));steps=('build-index','smoke','calibrate','infer','evaluate','report','pack') if a.stage=='all' else (a.stage,)
    try:
        for name in steps:
            a.stage=name;log=a.run_root/'logs'/(name+'.log');log.parent.mkdir(parents=True,exist_ok=True)
            print(f'STAGE={name} LOG={log}',flush=True)
            with log.open('a',encoding='utf-8') as f,contextlib.redirect_stdout(Tee(sys.stdout,f)),contextlib.redirect_stderr(Tee(sys.stderr,f)):
                rc=stage(a)
            print(f'STAGE={name} EXIT_CODE={rc}',flush=True)
            if rc:return rc
        return 0
    finally:
        if hasattr(a,'_engine'):a._engine.close()

if __name__=='__main__':raise SystemExit(main())
