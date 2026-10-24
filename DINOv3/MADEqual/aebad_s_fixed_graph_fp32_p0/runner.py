"""Explicit FP32 continuation, including fused CUDA dependency and benchmark."""
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import numpy as np
from DINOv3.MADEqual.aebad_s_fixed_graph_acceleration_p0 import runner as original
from . import search,service

STAGES=('preflight','prepare','smoke','bench','predict','evaluate','evaluate-aupro','summarize','pack','all')


def parser():
    # The parser reads STAGES from its private globals, never modifies the old
    # module's choices. FP64 prediction import is permanently disabled here.
    p=service.bind_namespace(original,STAGES=STAGES).parser()
    p.set_defaults(no_import_completed=True)
    return p


def preflight(args):
    c=service.get_service();missing=[]
    try:
        a,files=c.snapshot(args.parent_run)
        files += [a.data,a.checkpoint,a.official_root/'config.json',a.dinov3_root/'hubconf.py']
        missing=[str(p) for p in files if not p.exists()]
    except (KeyError,OSError,ValueError) as exc:missing.append(str(exc))
    try:triton_version=importlib.metadata.version('triton')
    except importlib.metadata.PackageNotFoundError:triton_version=None
    value=dict(engine=search.ENGINE,status='missing_inputs' if missing else
               'missing_CUDA_kernel_dependency' if triton_version is None else 'inputs_present_not_runtime_verified',
               missing_inputs=missing,triton_version=triton_version,geometry_dtype='FP32',
               CUDA_backend='Triton_fused_single_edge_pass',old_predictions_imported=False,
               reused='Source, MAD, supports, final graphs, normal-LOO selection',
               torch_imported='torch' in sys.modules,output_created=False,capacity_admission_checks=False)
    print(json.dumps(value,indent=2),flush=True)
    return 2 if missing or triton_version is None else 0


def benchmark(args):
    c=service.get_service();c.check_outputs(args.parent_run,args.output_root,args.cache_root,args.device,args.cpu_workers)
    a=c.initialize(args.parent_run,args.output_root,args.cache_root,args.device,args.cpu_workers,import_completed=False)
    c.require_smoke(a)
    started=time.perf_counter();result=c.predict(a,limit=args.limit or 3)
    prediction_seconds=time.perf_counter()-started
    previous=c.parent_args(args.parent_run);comparisons=[];missing=[]
    for sample in c.ip.catalog(a)['samples']:
        path=a.prediction_root/sample['domain']/(sample['sample_id']+'.npz')
        if not path.is_file() or not path.with_suffix('.json').is_file():continue
        old_path=previous.prediction_root/sample['domain']/(sample['sample_id']+'.npz')
        if not old_path.is_file() or not old_path.with_suffix('.json').is_file():
            missing.append(dict(domain=sample['domain'],sample_id=sample['sample_id']));continue
        new,meta,_=c.ip.verify_prediction(a,sample);old,old_meta,_=c.ip.verify_prediction(previous,sample)
        if not np.array_equal(new['positions'],old['positions']):raise ValueError('benchmark physical positions changed')
        score={r['branch']:r for r in meta['scores']};old_score={r['branch']:r for r in old_meta['scores']}
        for b in c.protocol.BRANCHES:
            row=dict(domain=sample['domain'],sample_id=sample['sample_id'],branch=b,
                     max_distance_abs_error=float(np.abs(new[b]-old[b]).max(initial=0)),
                     max_CH1_abs_error=abs(score[b]['CH1']-old_score[b]['CH1']),
                     max_CURRENT_abs_error=abs(score[b]['CURRENT']-old_score[b]['CURRENT']),
                     interpretation='FP32 minus parent FP64 numerical audit; GT unused')
            if b in (c.protocol.U,c.protocol.S):
                row['unchanged_baseline_verified']=bool(np.allclose(new[b],old[b],atol=1e-7,rtol=1e-6)
                    and row['max_CH1_abs_error']<=1e-5 and row['max_CURRENT_abs_error']<=1e-5)
            comparisons.append(row)
    c.st.write_csv(a.run_root/'FP32_full_image_numerical_audit.csv',comparisons)
    if any(r.get('unchanged_baseline_verified') is False for r in comparisons):
        raise ValueError('unchanged UNION/SEGMENT baseline failed original tolerances; audit retained')
    from DINOv3.MADEqual.aebad_s_support_witness_segment_p0.runner import resources
    observed=resources();calls=result.get('query_forward_calls',0)
    observed.update(model_forward_calls=calls,model_inference=calls>0,feature_extraction=calls>0)
    record=dict(engine=search.ENGINE,identity=c.ip.identity(a),result=result,seconds=prediction_seconds,resources=observed,
                per_new_image_seconds_including_startup=prediction_seconds/result['new_images'] if result['new_images'] else None,
                historical_audit_rows=len(comparisons),historical_reference_missing=missing,GT_used=False,
                scope='full new images, includes startup and validation; subsequent all reuses these FP32 predictions')
    c.st.write_json(a.run_root/'FP32_benchmarks'/f'{time.time_ns()}.json',record)
    c.st.write_json(a.run_root/'FP32_benchmark.json',record)
    print(json.dumps(record),flush=True)
    return 0


def main(argv=None):
    args=parser().parse_args(argv)
    if args.stage=='bench':
        for name in ('parent_run','output_root','cache_root'):setattr(args,name,getattr(args,name).resolve())
        if not 1<=args.cpu_workers<=8 or args.smoke_query_count<1 or (args.limit is not None and args.limit<1):raise ValueError('positive limits, 1..8 workers')
        if not args.device.startswith('cuda:') or not args.device[5:].isdigit():raise ValueError('one explicit cuda:N')
        return benchmark(args)
    bound=service.bind_namespace(original,c=service.get_service(),search=search,parser=parser,preflight=preflight,
                                 STAGES=STAGES,__package__=__package__)
    code=bound.main(argv)
    if args.stage=='summarize' and code==0:
        path=args.output_root.resolve()/'objective_report.md'
        with path.open('a',encoding='utf-8') as f:
            f.write('\n\n本轮图预测使用FP32单遍融合内核，未导入旧FP64逐图预测；'
                    'Source/MAD、支持特征、最终图及正常LOO选择复用父实验冻结资产。'
                    '数值核对及实际耗时请另读mixed_smoke.json、FP32_benchmark.json及acceleration_costs。\n')
    return code


if __name__=='__main__':raise SystemExit(main())
