"""Independent from-RGB run: no previous results, Source bank or distance caches."""
import argparse
import os
from pathlib import Path
import sys
import time
import traceback

from . import storage as st
from . import fresh_inputs as fi
from .method import LAYERS, BRANCHES, PROTOCOL

STAGES = ('preflight','prepare-data','build-source','calibrate','capture','prepare-support',
          'support-loo','predict','evaluate','evaluate-aupro','summarize','all')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',nargs='?',default='preflight',choices=STAGES)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--cache-root',type=Path,required=True,help='own run-specific HDD directory; no old cache needed')
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--device',default='cuda:0',help='one logical CUDA device; restrict physical GPU with CUDA_VISIBLE_DEVICES')
    p.add_argument('--cpu-workers',type=int,default=8)
    p.add_argument('--query-chunk',type=int,default=256)
    p.add_argument('--reference-chunk',type=int,default=2048)
    p.add_argument('--witness-batch',type=int,default=256)
    return p


def configuration(a):
    from importlib.metadata import version, PackageNotFoundError
    files = list(Path(__file__).parent.glob('*.py'))
    files += [fi.ROOT / x for x in (
        'DINOv3/relation_reliability/superadd_host.py',
        'DINOv3/relation_reliability/calibration.py',
        'DINOv3/relation_reliability/official_metrics.py',
        'DINOv3/MADEqual/aebad_s_segment_pair_crossdomain_p0/method.py',
        'DINOv3/MADEqual/aebad_s_segment_pair_crossdomain_p0/frozen_catalog.json',
        'DINOv3/MADEqual/aebad_s_ch1_scoring_inheritance/scoring.py',
        'DINOv3/MADEqual/aebad_s_target_local_segment/geometry.py',
        'DINOv3/MADEqual/robustad_target_fewshot_p0/scoring.py',
        'DINOv3/MADEqual/aebad_s_sparc_p0/runtime.py',
        'DINOv3/ShiftADD/aebad_s_p0/memory.py',
        'DINOv3/MADEqual/aebad_s_projection_layer_diagnostic/numerics.py',
        'DINOv3/MADEqual/aebad_s_projection_layer_diagnostic/metrics.py',
        'external_baselines/superadd_external/run.py',
        'external_baselines/superadd_external/memory_variants.py',
        'external_baselines/SuperADD/config.json')]
    files += list((fi.ROOT/'external_baselines/dinov3').rglob('*.py'))
    files += list((fi.ROOT/'external_baselines/SuperADD/tracks/industrial/src/industrial').rglob('*.py'))
    software={}
    for name in ('numpy','scipy','Pillow','scikit-learn','torch','torchvision','adeval'):
        try: software[name]=version(name)
        except PackageNotFoundError: software[name]=None
    return dict(revision=fi.REVISION,data=str(a.data.resolve()),checkpoint=str(a.checkpoint.resolve()),
        cache=str(a.cache_root.resolve()),device=a.device,cpu_workers=a.cpu_workers,
        query_chunk=a.query_chunk,reference_chunk=a.reference_chunk,witness_batch=a.witness_batch,
        physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),protocol=PROTOCOL,
        code={p.relative_to(fi.ROOT).as_posix():st.sha(p) for p in files},software=software,
        capacity_admission_checks=False)


def setup(a):
    c = fi.prepare_data(a.run_root,a.data,a.cache_root)
    value = dict(configuration=configuration(a),catalog_sha256=st.sha(a.run_root/'fresh_catalog.json'),
                 cache=str(a.cache_root.resolve()))
    value['identity'] = st.digest(value)
    p = a.run_root/'setup.json'
    if p.exists() and st.read_json(p)!=value: raise ValueError('new-run inputs/code changed; use a new run')
    st.write_json(p,value)
    st.write_json(a.run_root/'protocol.json',dict(PROTOCOL,run_mode=fi.REVISION,
        Source_builder='original SuperADD, ordered521, exclude index%8==0, 100K/layer, 100 sampler rounds',
        Source_augmentation=st.read_json(fi.ROOT/'external_baselines/SuperADD/config.json')['brightness_augmentation'],
        query_and_support_brightness=1,historical_metrics_reused=False))
    return dict(status='fresh_inputs_frozen',source_images=len(c['source']),evaluation_images=len(c['samples']),supports=24)


def current_config(a):
    config = configuration(a)
    p = a.run_root/'run_config.json'
    if p.exists():
        if st.read_json(p)!=config: raise ValueError('run settings/code differ; original run preserved')
    else: st.write_json(p,config)


def run_stage(a):
    if a.stage=='prepare-data': return setup(a)
    if a.stage in ('build-source','calibrate','capture'):
        from . import fresh_runtime as runtime
        return {'build-source':runtime.source,'calibrate':runtime.calibration,'capture':runtime.capture}[a.stage](a)
    bundle = fi.Bundle(a.run_root)
    from . import supports, prediction
    out = st.initialize(a.run_root/'experiment',bundle)
    connections = out/'connections_batched'
    if not connections.exists():
        target = a.cache_root/'connections_batched'
        target.mkdir(parents=True,exist_ok=True)
        connections.symlink_to(target,target_is_directory=True)
    if a.stage=='prepare-support':
        supports.configure(out,a.device,a.query_chunk,a.reference_chunk)
        return [supports.prepare(bundle,out,bundle.locations,d,l,device=a.device,
            query_chunk=a.query_chunk,reference_chunk=a.reference_chunk) for d in fi.DOMAINS for l in LAYERS]
    if a.stage=='support-loo':
        return [supports.support_loo(bundle,out,bundle.locations,d,l,device=a.device,
            witness_batch=a.witness_batch) for d in fi.DOMAINS for l in LAYERS]
    if a.stage=='predict':
        for d in fi.DOMAINS:
            for l in LAYERS:
                prediction.predict_layer(bundle,out,bundle.locations,d,l,device=a.device,witness_batch=a.witness_batch)
        return [prediction.freeze_domain(bundle,out,d,workers=a.cpu_workers) for d in fi.DOMAINS]
    from . import fresh_evaluation as ev
    if a.stage=='evaluate': return [ev.evaluate(bundle,out,d,a.cpu_workers) for d in fi.DOMAINS]
    if a.stage=='evaluate-aupro': return [ev.aupro(bundle,out,d,b,a.device) for d in fi.DOMAINS for b in BRANCHES]
    return ev.summarize(bundle,out)


def main(argv=None):
    a = parser().parse_args(argv)
    a.run_root,a.cache_root,a.data,a.checkpoint = [p.resolve() for p in (a.run_root,a.cache_root,a.data,a.checkpoint)]
    if not 1<=a.cpu_workers<=8 or min(a.query_chunk,a.reference_chunk,a.witness_batch)<1:
        raise ValueError('1..8 CPU workers and positive execution blocks required')
    if not a.device.startswith('cuda:') or not a.device[5:].isdigit(): raise ValueError('one explicit CUDA device required')
    if a.stage=='preflight':
        import importlib.metadata
        required = [a.data,a.checkpoint,fi.ROOT/'external_baselines/SuperADD/config.json',fi.ROOT/'external_baselines/dinov3/hubconf.py']
        missing = [str(p) for p in required if not p.exists()]
        deps={}
        for name in ('numpy','scipy','Pillow','scikit-learn','psutil','torch','torchvision','timm','omegaconf','matplotlib','adeval'):
            try: deps[name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError: deps[name]=None
        report=dict(status='fresh_inputs_present_not_runtime_test' if not missing else 'missing_inputs',
            missing_inputs=missing,dependencies=deps,old_caches_required=False,old_Source_memory_required=False,
            original_RGB_Source_required=521,evaluation_images=926,support_images=24,torch_imported='torch' in sys.modules,
            capacity_admission_checks=False,experiment_directory_created=False)
        print(__import__('json').dumps(report,indent=2),flush=True)
        return 2 if missing or any(v is None for v in deps.values()) or deps['adeval']!='1.1.0' else 0
    # Own run directories only; no output may contain or be contained by RGB/weights/code.
    for p in (a.data,a.checkpoint,fi.ROOT):
        for destination in (a.run_root,a.cache_root):
            if destination==p or destination in p.parents or p in destination.parents:
                raise ValueError('new-run output overlaps input/code: '+str(destination))
    current_config(a)
    if a.stage!='all':
        tick=time.perf_counter()
        try:
            result=run_stage(a)
            from .runner import resources
            record=dict(stage=a.stage,status='stage_complete',seconds=time.perf_counter()-tick,resources=resources())
            # The common resource observer describes offline stages; override the
            # actual semantics explicitly for the three fresh model stages.
            calls=sum(result.get(k,0) for k in ('source_forward_calls','calibration_forward_calls',
                'query_forward_calls','support_forward_calls')) if isinstance(result,dict) else 0
            record['resources'].update(model_inference=calls>0,feature_extraction=calls>0,model_forward_calls=calls)
            if isinstance(result,dict): record['result']=result
            st.write_json(a.run_root/'cost_records'/(a.stage+'_'+str(time.time_ns())+'.json'),record)
            if a.stage=='summarize' and result['status']!='all_requested_stages_complete': return 2
            print(__import__('json').dumps(dict(stage=a.stage,status='stage_complete',seconds=record['seconds'])),flush=True)
            return 0
        except Exception as exc:
            traceback.print_exc()
            st.write_json(a.run_root/'failures'/(a.stage+'.json'),dict(stage=a.stage,error=f'{type(exc).__name__}: {exc}'))
            return 2
    from .autodl import run, pack, ENVIRONMENT
    started=time.perf_counter(); state=dict(status='running',success=False,mode=fi.REVISION)
    archive=None
    common=['--run-root',str(a.run_root),'--cache-root',str(a.cache_root),'--data',str(a.data),'--checkpoint',str(a.checkpoint),
        '--device',a.device,'--cpu-workers',str(a.cpu_workers),'--query-chunk',str(a.query_chunk),
        '--reference-chunk',str(a.reference_chunk),'--witness-batch',str(a.witness_batch)]
    try:
        state['stage']='environment'; st.write_json(a.run_root/'pipeline_status.json',state)
        run('environment',[sys.executable,'-B','-u','-c',ENVIRONMENT],a.run_root)
        for stage in STAGES[1:-1]:
            state['stage']=stage; st.write_json(a.run_root/'pipeline_status.json',state)
            run(stage,[sys.executable,'-B','-u','-m',__package__+'.fresh',stage,*common],a.run_root)
        final=st.read_json(a.run_root/'experiment/status.json')
        if (final['status']!='all_requested_stages_complete' or not final['all_metrics_complete']
            or final['pending'] or final['numerical_failed_checks']!=0):
            raise ValueError('fresh run incomplete; outputs retained')
        state.update(status='all_requested_stages_complete',success=True)
    except BaseException as exc:
        traceback.print_exc(); state.update(status='failed_outputs_retained',success=False,error=f'{type(exc).__name__}: {exc}')
    finally:
        state['seconds']=time.perf_counter()-started
        st.write_json(a.run_root/'pipeline_status.json',state)
        st.write_json(a.run_root/'cost.json',{p.stem:st.read_json(p) for p in (a.run_root/'cost_records').glob('*.json')})
        try:
            archive=pack(a.run_root,state['success']); print('REVIEW_ARCHIVE='+str(archive),flush=True)
        except BaseException as exc:
            traceback.print_exc(); state.update(success=False,pack_error=f'{type(exc).__name__}: {exc}')
            st.write_json(a.run_root/'pipeline_status.json',state)
        rc=0 if state['success'] and archive else 2
        Path(str(a.run_root)+'.exitcode').write_text(str(rc)+'\n')
        print(f'PIPELINE_EXIT_CODE={rc}\nRUN_ROOT={a.run_root}\nELAPSED_SECONDS={state["seconds"]:.1f}',flush=True)
    return rc


if __name__=='__main__': raise SystemExit(main())
