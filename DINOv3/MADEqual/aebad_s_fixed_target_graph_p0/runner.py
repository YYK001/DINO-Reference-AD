"""Independent fresh pipeline; default read-only preflight, no resource gates."""
import argparse
import importlib.metadata
import os
from pathlib import Path
import sys
import time
import traceback
from . import inputs as ip
from .inputs import st
from .protocol import REVISION,PROTOCOL,DOMAINS,LAYERS,BRANCHES

STAGES=('preflight','prepare-data','build-source','calibrate','extract-support','prepare-graphs',
        'support-loo','select','predict','evaluate','evaluate-aupro','summarize','all')


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',nargs='?',default='preflight',choices=STAGES)
    for name in ('run-root','cache-root','data','checkpoint'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('graph-root','query-cache-root','prediction-root','evaluation-scratch','reuse-fresh-run'):
        p.add_argument('--'+name,type=Path)
    p.add_argument('--official-root',type=Path,default=ip.ROOT/'external_baselines/SuperADD')
    p.add_argument('--dinov3-root',type=Path,default=ip.ROOT/'external_baselines/dinov3')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--cpu-workers',type=int,default=8)
    p.add_argument('--seed',type=int,choices=(42,43,44),default=42)
    p.add_argument('--evaluation-scope',choices=('primary','common'))
    p.add_argument('--query-mode',choices=('disk','stream'),default='stream')
    p.add_argument('--query-chunk',type=int,default=128)
    p.add_argument('--reference-chunk',type=int,default=2048)
    p.add_argument('--edge-chunk',type=int,default=2048)
    p.add_argument('--witness-edge-chunk',type=int,default=64)
    p.add_argument('--domains',nargs='+',choices=DOMAINS,default=list(DOMAINS),help='execution jobs, not evaluation population')
    p.add_argument('--layers',nargs='+',type=int,choices=LAYERS,default=list(LAYERS),help='prepare-graphs/support-loo jobs only')
    p.add_argument('--heldouts',nargs='+',type=int,choices=range(9),default=list(range(9)),help='0=final graph, 1..8=LOO folds')
    p.add_argument('--branch',choices=BRANCHES,help='one explicit AUPRO job')
    p.add_argument('--limit',type=int,help='predict: at most this many NEW image jobs; incomplete cannot evaluate')
    p.add_argument('--no-write',action='store_true',help='preflight is always read-only')
    return p


def normalize(a):
    a.evaluation_scope=a.evaluation_scope or ('primary' if a.seed==42 else 'common')
    if a.seed!=42 and a.evaluation_scope!='common':raise ValueError('43/44 use common population')
    for name in ('run_root','cache_root','data','checkpoint','official_root','dinov3_root'):
        setattr(a,name,getattr(a,name).resolve())
    defaults=dict(graph_root=a.cache_root/'fixed_graph',query_cache_root=a.cache_root/'queries',
        prediction_root=a.cache_root/'predictions',evaluation_scratch=a.run_root/'pixel_scratch')
    for name,default in defaults.items():setattr(a,name,(getattr(a,name) or default).resolve())
    if a.reuse_fresh_run:a.reuse_fresh_run=a.reuse_fresh_run.resolve()
    if not 1<=a.cpu_workers<=8 or min(a.query_chunk,a.reference_chunk,a.edge_chunk,a.witness_edge_chunk)<1:
        raise ValueError('1..8 workers and positive blocks required')
    if not a.device.startswith('cuda:') or not a.device[5:].isdigit():raise ValueError('one explicit cuda:N; no automatic device fallback')
    if a.limit is not None and a.limit<1:raise ValueError('positive explicit image limit required')
    return a


def preflight(a):
    required=[a.data,a.checkpoint,a.official_root/'config.json',a.dinov3_root/'hubconf.py']
    if a.reuse_fresh_run:required += [a.reuse_fresh_run/'setup.json',a.reuse_fresh_run/'fresh_catalog.json']
    missing=[str(p) for p in required if not p.exists()]
    versions={}
    for name in ('numpy','scipy','Pillow','scikit-learn','psutil','torch','torchvision','timm','omegaconf','adeval'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    counts={'background':297,'illumination':265,'view':364} if a.evaluation_scope=='primary' else {'background':281,'illumination':249,'view':348}
    missing_dependencies=[name for name,v in versions.items() if v is None]
    report=dict(status='missing_inputs' if missing else 'missing_dependencies' if missing_dependencies else 'inputs_present_not_runtime_verified',missing_inputs=missing,
        missing_dependencies=missing_dependencies,
        dependencies=versions,seed=a.seed,evaluation_scope=a.evaluation_scope,images=counts,query_mode=a.query_mode,
        old_AutoDL_assets_required=False,explicit_reuse=str(a.reuse_fresh_run) if a.reuse_fresh_run else None,
        torch_imported='torch' in sys.modules,capacity_admission_checks=False,experiment_directory_created=False,
        normal_search=dict(witness_configs=36,plain_configs=9,folds=8,final_graphs='recomputed on all8'),
        estimates=dict(query_FP32_bytes_if_118x118=sum(counts.values())*4*13924*1280*4 if a.query_mode=='disk' else 0,
            maximum_mutual_edges_per_layer_if_118x118=8*13924*100//2,
            endpoint_block_bytes_if_118x118=a.query_chunk*8*13924*8,
            note='shape assumptions only; actual counts/graph work emitted before each graph search'))
    print(__import__('json').dumps(report,indent=2),flush=True)
    return 2 if missing or any(v is None for v in versions.values()) or versions['adeval']!='1.1.0' else 0


def freeze_setup(a):
    # Execution selectors are not method parameters. Paths and numerical blocks are frozen.
    selectors=('stage','no_write','domains','layers','heldouts','branch','limit')
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items() if k not in selectors}
    paths=list(Path(__file__).parent.glob('*.py'))
    paths += list((ip.ROOT/'DINOv3/MADEqual/aebad_s_support_witness_segment_p0').glob('*.py'))
    paths += [ip.ROOT/x for x in ('DINOv3/relation_reliability/superadd_host.py',
        'DINOv3/relation_reliability/official_metrics.py','DINOv3/relation_reliability/calibration.py',
        'external_baselines/superadd_external/memory_variants.py',
        'external_baselines/superadd_external/run.py','DINOv3/ShiftADD/aebad_s_p0/memory.py',
        'DINOv3/MADEqual/aebad_s_ch1_scoring_inheritance/scoring.py',
        'DINOv3/MADEqual/aebad_s_target_local_segment/geometry.py')]
    config.update(revision=REVISION,protocol=PROTOCOL,code={p.relative_to(ip.ROOT).as_posix():st.sha(p) for p in paths},
        host_config=st.read_json(a.official_root/'config.json'),physical_GPU=os.environ.get('CUDA_VISIBLE_DEVICES'))
    for name in ('numpy','scipy','Pillow','scikit-learn','torch','torchvision','adeval'):
        try:config.setdefault('software',{})[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:config.setdefault('software',{})[name]=None
    config['external_source_code']={}
    for name,root in (('dinov3',a.dinov3_root),('SuperADD',a.official_root)):
        for p in sorted(root.rglob('*.py')):
            config['external_source_code'][name+'/'+p.relative_to(root).as_posix()]=st.sha(p)
    # JSON round trip normalizes tuples before comparison.
    config=__import__('json').loads(__import__('json').dumps(config))
    path=a.run_root/'run_config.json'
    if path.exists() and st.read_json(path)!=config:raise ValueError('configuration/code changed; keep original output and use new run')
    st.write_json(path,config)
    return config


def prepare(a):
    c=ip.prepare(a);config=freeze_setup(a)
    value=dict(configuration=config,catalog_sha256=st.sha(a.run_root/'fresh_catalog.json'),cache=str(a.cache_root))
    value['identity']=st.digest(value);p=a.run_root/'setup.json'
    if p.exists() and st.read_json(p)!=value:raise ValueError('frozen inputs changed')
    st.write_json(p,value);st.write_json(a.run_root/'protocol.json',dict(PROTOCOL,seed=a.seed,
        evaluation_scope=a.evaluation_scope,query_mode=a.query_mode,Source_brightness=[.8,1.2]))
    st.write_csv(a.run_root/'support_manifest.csv',[dict(r,domain=d) for d,g in c['supports'].items() for r in g])
    st.write_csv(a.run_root/'evaluation_manifest.csv',c['samples'])
    return dict(status='inputs_frozen',images=len(c['samples']),Source_images=521,calibration_images=66)


def run_stage(a):
    if a.stage=='prepare-data':return prepare(a)
    from . import runtime as rt
    if a.stage in ('build-source','calibrate','extract-support','prepare-graphs','support-loo','select'):
        return {'build-source':rt.build_source,'calibrate':rt.calibrate,'extract-support':rt.extract_support,
            'prepare-graphs':rt.prepare_graphs,'support-loo':rt.support_loo,'select':rt.select}[a.stage](a)
    if a.stage=='predict':
        from .prediction import freeze
        return freeze(a)
    from . import evaluation as ev
    if a.stage=='evaluate':
        from .prediction import verified
        verified(a)
        return [ev.evaluate(a,d,global_checked=True) for d in a.domains]
    if a.stage=='evaluate-aupro':
        from .prediction import verified
        verified(a)
        return [ev.aupro(a,d,b) for d in a.domains for b in ([a.branch] if a.branch else BRANCHES)]
    from .reporting import summarize
    return summarize(a)


def common_args(a):
    result=[]
    for k,v in vars(a).items():
        if k in ('stage','no_write') or v is None:continue
        result+=['--'+k.replace('_','-'),*([str(x) for x in v] if isinstance(v,(list,tuple)) else [str(v)])]
    return result


def storage_observation(a):
    """Post-stage metadata only, no content hashing or capacity admission.

    Logical sizes include linked assets; overlapping roots are non-additive.
    These are retained snapshots, never advertised as peak scratch usage.
    """
    rows=[]
    for name in ('run_root','cache_root','graph_root','query_cache_root','prediction_root','evaluation_scratch'):
        root=getattr(a,name);size=files=0
        for current,dirs,names in os.walk(root,followlinks=False):
            for filename in names:
                path=Path(current)/filename
                size+=path.stat().st_size;files+=1
        rows.append(dict(root=name,path=str(root),files=files,logical_retained_bytes=size))
    return dict(roots=rows,scope='post-stage retained snapshot; linked/overlapping roots are non-additive; not peak disk',
        capacity_admission_checks=False)


def main(argv=None):
    a=normalize(parser().parse_args(argv))
    if a.stage=='preflight':return preflight(a)
    if a.no_write:raise ValueError('--no-write is only valid for preflight')
    destinations=[a.run_root,a.cache_root,a.graph_root,a.query_cache_root,a.prediction_root,a.evaluation_scratch]
    for p in (a.data,a.checkpoint,a.official_root,a.dinov3_root,Path(__file__).parent):
        if any(x==p or x in p.parents or p in x.parents for x in destinations):raise ValueError('output overlaps protected input/code')
    if a.reuse_fresh_run and any(x==a.reuse_fresh_run or a.reuse_fresh_run in x.parents for x in destinations):
        raise ValueError('new output must not modify reuse source')
    if a.stage!='prepare-data':freeze_setup(a)
    if a.stage!='all':
        start=time.perf_counter()
        try:
            result=run_stage(a)
            from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.runner import resources
            resource=resources();resource['model_inference']=a.stage in ('build-source','calibrate','extract-support','predict')
            if isinstance(result,dict):
                calls=sum(result.get(k,0) for k in ('source_forward_calls','calibration_forward_calls','support_forward_calls','query_forward_calls'))
                resource.update(model_forward_calls=calls,model_inference=calls>0,feature_extraction=calls>0)
            record=dict(stage=a.stage,status='complete',seconds=time.perf_counter()-start,resources=resource,result=result,
                storage=storage_observation(a))
            st.write_json(a.run_root/'cost_records'/f'{a.stage}_{time.time_ns()}.json',record)
            print(__import__('json').dumps(dict(stage=a.stage,status='complete',seconds=record['seconds'])),flush=True)
            return 2 if a.stage=='summarize' and not result['all_metrics_complete'] else 0
        except BaseException as exc:
            traceback.print_exc();st.write_json(a.run_root/'failures'/f'{a.stage}.json',dict(error=f'{type(exc).__name__}: {exc}'))
            return 2
    from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.autodl import run,ENVIRONMENT
    start=time.perf_counter();value=dict(status='running',success=False);archive=None
    try:
        value['stage']='environment';st.write_json(a.run_root/'pipeline_status.json',value)
        run('environment',[sys.executable,'-B','-u','-c',ENVIRONMENT],a.run_root)
        for stage in STAGES[1:-1]:
            value['stage']=stage;st.write_json(a.run_root/'pipeline_status.json',value)
            run(stage,[sys.executable,'-B','-u','-m',__package__+'.runner',stage,*common_args(a)],a.run_root)
        value.update(status='all_requested_metrics_complete',success=True)
    except BaseException as exc:
        traceback.print_exc();value.update(status='failed_outputs_retained',success=False,error=f'{type(exc).__name__}: {exc}')
    finally:
        value['seconds']=time.perf_counter()-start;st.write_json(a.run_root/'pipeline_status.json',value)
        st.write_json(a.run_root/'cost.json',{p.stem:st.read_json(p) for p in (a.run_root/'cost_records').glob('*.json')})
        try:archive=pack_review(a,value['success']);print('REVIEW_ARCHIVE='+str(archive),flush=True)
        except BaseException as exc:
            traceback.print_exc();value.update(success=False,pack_error=str(exc));st.write_json(a.run_root/'pipeline_status.json',value)
        code=0 if value['success'] and archive else 2
        Path(str(a.run_root)+'.exitcode').write_text(str(code)+'\n')
        print(f'PIPELINE_EXIT_CODE={code}\nRUN_ROOT={a.run_root}\nELAPSED_SECONDS={value["seconds"]:.1f}',flush=True)
    return code


def pack_review(a,success):
    """Only small deliverables; configured caches/scratch are excluded even inside run."""
    import tarfile
    suffix='_review' if success else '_FAILED_review'
    path=a.run_root.with_name(a.run_root.name+suffix+'.tar.gz')
    if path.exists():path=a.run_root.with_name(a.run_root.name+suffix+'_'+str(time.time_ns())+'.tar.gz')
    excluded={p.resolve() for p in (a.cache_root,a.graph_root,a.prediction_root,a.query_cache_root,a.evaluation_scratch)}
    with tarfile.open(path,'x:gz',compresslevel=1,dereference=False) as archive:
        for current,dirs,files in os.walk(a.run_root,followlinks=False):
            current=Path(current)
            dirs[:]=[d for d in dirs if (current/d).resolve() not in excluded and not (current/d).is_symlink()]
            for name in sorted(files):
                p=current/name
                if p.suffix in ('.json','.csv','.md','.log','.exitcode','.png') and not p.is_symlink():
                    archive.add(p,arcname=a.run_root.name+'/'+p.relative_to(a.run_root).as_posix(),recursive=False)
    return path


if __name__=='__main__':raise SystemExit(main())
