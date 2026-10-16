"""Default read-only-capable preflight; explicit cached geometry and full analysis."""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from DINOv3.MADEqual.aebad_s_union_support_loo.analysis import load_raw
from DINOv3.MADEqual.aebad_s_union_support_loo.runner import audit,formal_baselines
from DINOv3.MADEqual.aebad_s_union_support_loo.inputs import digest
from .inputs import TOOL,preflight,verify_files,load_support_layer,gt_relative,replay
from .geometry import BRANCHES,LAYERS,geometry_layer,NumericalError,scatter,DIST_ATOL,DIST_RTOL,GAMMA
from .analysis import (RAW_FORMAT,save_layer,compact_raws,maps_from_compact,spatial,ranking,
                       mechanism,mechanism_summary,interpolation_rows)


def runtime_map_backend(mad: Path):
    from DINOv3.MADEqual.aebad_s_union_diagnostics.spatial import MapBackend
    return MapBackend(mad)


def runtime_full_protocol():
    from DINOv3.MADEqual.aebad_s_full_p0 import runner
    return runner


def runtime_pixel_backend():
    from DINOv3.relation_reliability.official_metrics import validate_official_metric_backend
    validate_official_metric_backend()
    import torch
    torch.cuda.reset_peak_memory_stats()


def memory_observation(*, cuda_evaluation_started: bool = False) -> dict:
    out=dict(process_peak_rss_bytes=None,evaluation_peak_cuda_allocated_bytes=None,
             evaluation_peak_cuda_reserved_bytes=None)
    try:
        import resource
        out['process_peak_rss_bytes']=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024)
    except ImportError:
        try:
            import psutil
            out['process_peak_rss_bytes']=getattr(psutil.Process().memory_info(),'peak_wset',None)
        except ImportError: pass
    if cuda_evaluation_started and 'torch' in sys.modules:
        torch=sys.modules['torch']
        if torch.cuda.is_available():
            out.update(evaluation_peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                       evaluation_peak_cuda_reserved_bytes=int(torch.cuda.max_memory_reserved()))
    return out


def aggregate_interpolation(rows: list[dict]) -> list[dict]:
    """Pooled counts/histograms; fractions also expose image-equal aggregation."""
    summaries=[]
    groups=defaultdict(list)
    for r in rows:groups[(r['layer'],r['label'],r['region'])].append(r)
    for (layer,label,region),group in groups.items():
        distributions=[r for r in group if r['statistic']=='interpolation_distribution']
        count=sum(r['patch_count'] for r in distributions)
        if not count:continue
        row=dict(sample_id='__aggregate__',layer=layer,label=label,region=region,
                 statistic='interpolation_distribution',patch_count=count,image_count=len(distributions),
                 aggregation='pooled_patch_counts_and_image_equal_reduction_fractions',
                 alpha_min=min(r['alpha_min'] for r in distributions),alpha_max=max(r['alpha_max'] for r in distributions),
                 alpha_mean=sum(r['alpha_mean']*r['patch_count'] for r in distributions)/count,
                 endpoint_degenerate_count=sum(r['endpoint_degenerate_count'] for r in distributions),
                 correction_count=sum(r['correction_count'] for r in distributions),
                 maximum_squared_correction=max(r['maximum_squared_correction'] for r in distributions))
        row.update({f'alpha_bin_{i}_count':sum(r[f'alpha_bin_{i}_count'] for r in distributions) for i in range(10)})
        for key in ('midpoint_lower_fraction','segment_lower_fraction'):
            row[key]=sum(r[key]*r['patch_count'] for r in distributions)/count
            row[key+'_image_equal']=float(np.mean([r[key] for r in distributions]))
        summaries.append(row)
        pairs=defaultdict(list)
        for r in group:
            if r['statistic']=='ordered_endpoint_pair':pairs[(r['support_1'],r['support_2'])].append(r)
        for (s1,s2),pair in pairs.items():
            n=sum(r['count'] for r in pair)
            summaries.append(dict(sample_id='__aggregate__',layer=layer,label=label,region=region,
                statistic='ordered_endpoint_pair',support_1=s1,support_2=s2,count=n,patch_count=count,
                fraction=n/count,image_equal_fraction=sum(r['fraction'] for r in pair)/len(distributions),
                image_count=len(distributions)))
    return summaries


def write_report(output: Path,status: dict,metrics: list[dict],summaries: list[dict]) -> None:
    lines=['# Frozen Target K8 bounded local references','',
        f"Status: {status['status']}; images: {status.get('analyzed_images',0)}/297.",
        f"Predictions frozen before GT: {status.get('predictions_frozen',False)}.",
        f"Formal K8 reproduced: {status.get('formal_baseline_reproduced',False)}.",
        f"Full pixel evaluation complete: {status.get('pixel_metrics_complete',False)}.",'',
        '| branch | Image AUROC | Image AP | Pixel AUROC | Pixel AP | AUPRO |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in metrics:
        lines.append('| '+r['branch']+' | '+' | '.join(str(r.get(k,'')) for k in ('image_AUROC','image_AP','pixel_AUROC','pixel_AP','AUPRO'))+' |')
    lines += ['', '## Confirmed numerical facts',
        'Distances/indices come from verified B capture. Each layer selects the nearest candidates from two different supports; distances/support order break ties.',
        'All three branches retain original union. No query feature, inference, extraction, memory rebuild or MAD refit.',
        'Pixel evaluation uses original full-protocol masks/maps and ADEval CUDA (200 thresholds); no resize/crop protocol substitution.',
        '', '## Intervention changes', 'See ranking_summary.csv, defect_instances.csv and mechanism_summary.csv. Mechanism reference is K8_UNION, not Source.',
        '', '## Manual review', 'Semantics unreviewed; GT complement is not automatically background.',
        '', '## Pending explanations', 'A bounded combination of normal endpoints is not guaranteed normal or defect-preserving. No best branch/support selection from GT.',
        '', '## Unverified or unavailable',
        'Not full-bank exact segment search; not same-physical-patch correspondence; within-support uniqueness and historical unique indices are unknown.',
        'FP64 arithmetic does not recover precision lost in FP32 cached distances. No new-model direct-query verification or end-to-end inference-speed measurement.',
        f"Missing/incomplete spatial IDs: {status.get('incomplete_spatial_ids',[])}.",
        f"Pixel backend error: {status.get('pixel_backend_error','')}.",
        f"Failure: {status.get('error','')}."]
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def analyze(args,check: dict) -> int:
    paths={k:Path(v) for k,v in check['paths'].items()};output=paths['output']
    cap=check['capture_inputs'];old_records={r['sample_id']:r for r in check['capture_status']['records']}
    started=time.perf_counter();timing=defaultdict(float)
    tables={k:[] for k in ('numerical_verification','per_image_scores','defect_instances','peak_positions',
                          'mechanism_per_image','interpolation_summary')}
    metrics=[];summaries=[];new_records=[]
    status=dict(tool=TOOL,stage='analyze',scope='background',status='geometry_running',predictions_frozen=False,
        geometry_layer_images=0,analyzed_images=0,branches=list(BRANCHES),model_inference=False,
        feature_extraction=False,incomplete_spatial_ids=[],pixel_metrics_complete=False,
        pixel_evaluation_started=False,formal_baseline_reproduced=False)
    frozen=dict(capture_identity=cap['identity'],capture_inputs_sha256=replay.sha256(paths['capture']/'inputs.json'),
        capture_status_sha256=replay.sha256(paths['capture']/'status.json'),
        supports=cap['supports'],formal_sha256=cap['frozen']['formal_sha256'],
        geometry=dict(dimension=1280,distance_atol=DIST_ATOL,distance_rtol=DIST_RTOL,gamma64=GAMMA,
                      degeneration='exact_endpoint_equality_c_squared_zero',endpoint_chunk=1024),
        code_sha256={r['kind']:r['sha256'] for r in check['required_files'] if 'code_' in r['kind']})
    identity=digest(frozen);status['identity']=identity
    replay.atomic_json(output/'inputs.json',dict(tool=TOOL,identity=identity,frozen=frozen,
        upstream_paths={k:str(v) for k,v in paths.items() if k!='output'},samples=check['samples'],required_files=check['required_files']))
    replay.atomic_json(output/'status.json',status)

    def read_original(sample):
        t=time.perf_counter();r=old_records[sample['sample_id']]
        values=load_raw(paths['capture']/r['file'],identity=cap['identity'],sample=sample,checksum=r['sha256'])
        timing['cache_reading']+=time.perf_counter()-t
        return values

    def read_new(sample,arrays):
        t=time.perf_counter()
        values=compact_raws(output,record_map,identity,old_records[sample['sample_id']],sample,arrays)
        timing['cache_reading']+=time.perf_counter()-t
        return values

    def map_values(raws,arrays,meta,backend):
        t=time.perf_counter();maps=maps_from_compact(raws,arrays,meta,backend)
        timing['maps']+=time.perf_counter()-t
        if any(not np.isfinite(m).all() or m.shape!=tuple(meta['output_shape']) for m in maps.values()):
            raise NumericalError('original map backend returned invalid map',{})
        return maps

    try:
        verify_files(check['required_files'])
        for offset,layer in enumerate(LAYERS):
            t=time.perf_counter();banks=load_support_layer(paths,cap['supports'],layer)
            timing['support_features_reading']+=time.perf_counter()-t
            for index,sample in enumerate(check['samples']):
                arrays,meta=read_original(sample)
                t=time.perf_counter()
                try:
                    values,diag=geometry_layer(arrays['target_distance'][:,offset],arrays['target_index'][:,offset],
                                               arrays['source_distance'][offset],banks)
                except NumericalError as error:
                    tables['numerical_verification'].append(dict(sample_id=sample['sample_id'],layer=layer,
                        check_kind='bounded_geometry',verified=False,reason=str(error),**error.diagnostics))
                    raise
                timing['geometry']+=time.perf_counter()-t
                tables['numerical_verification'].append(dict(sample_id=sample['sample_id'],layer=layer,
                    check_kind='bounded_geometry',verified=True,**diag))
                expected=np.minimum(arrays['source_distance'][offset],arrays['target_distance'][:,offset].min(axis=0))
                audit(tables['numerical_verification'],sample['sample_id'],'original_K8_distance',values['K8_UNION'],expected,layer=layer)
                relative=f"raw_cache/{sample['sample_id']}/L{layer}.npz"
                t=time.perf_counter()
                save_layer(output/relative,values,arrays['positions'],dict(format=RAW_FORMAT,identity=identity,
                    capture_raw_sha256=old_records[sample['sample_id']]['sha256'],sample_id=sample['sample_id'],
                    relative_path=sample['relative_path'],layer=layer,grid=meta['grid'],output_shape=meta['output_shape']))
                new_records.append(dict(sample_id=sample['sample_id'],layer=layer,file=relative,sha256=replay.sha256(output/relative)))
                timing['prediction_writing']+=time.perf_counter()-t
                status['geometry_layer_images']+=1
                if index%25==0 or index==len(check['samples'])-1:print(f"[LOCAL geometry] L{layer} {index+1}/297",flush=True)
            del banks;gc.collect()
            replay.write_csv(output/'numerical_verification.csv',tables['numerical_verification'])
            replay.atomic_json(output/'status.json',status)
        verify_files(check['required_files'])
        replay.atomic_json(output/'predictions.json',dict(tool=TOOL,identity=identity,records=new_records,images=297,layers=list(LAYERS)))
        status.update(predictions_frozen=True,status='maps_and_analysis_running')
        replay.atomic_json(output/'status.json',status)
        record_map={(r['sample_id'],r['layer']):r for r in new_records}
        backend=runtime_map_backend(paths['mad'])
        evaluator=runtime_full_protocol()
        formal=formal_baselines(paths['results'],check['samples'])
        selected={r['sample_id'] for r in cap['selected_cases']}
        full_masks=[]
        try:
            runtime_pixel_backend();pixel_available=True
        except (ImportError,RuntimeError,OSError) as error:
            pixel_available=False;status['pixel_backend_error']=f'{type(error).__name__}: {error}'
        # Prediction distances are now frozen for ALL images. Only here decode GT.
        for index,sample in enumerate(check['samples']):
            arrays,meta=read_original(sample);raws,trace=read_new(sample,arrays)
            maps=map_values(raws,arrays,meta,backend)
            audit(tables['numerical_verification'],sample['sample_id'],'formal_K8_image_score',
                maps[BRANCHES[0]].max(),formal['K8'][sample['sample_id']]['image_score'],score=True)
            t=time.perf_counter()
            source=backend.runner._method_map({'official_raw':scatter(arrays['source_distance'],arrays['positions'],tuple(meta['grid']))},
                                              'MAD_EQUAL',backend.mad,tuple(meta['output_shape']))
            timing['maps']+=time.perf_counter()-t
            audit(tables['numerical_verification'],sample['sample_id'],'formal_Source_image_score',
                source.max(),formal['Source'][sample['sample_id']]['image_score'],score=True)
            del source
            for b in BRANCHES:
                y,x=np.unravel_index(np.argmax(maps[b]),maps[b].shape)
                tables['per_image_scores'].append(dict(sample_id=sample['sample_id'],relative_path=sample['relative_path'],
                    label=sample['label'],branch=b,image_score=float(maps[b].max()),
                    score_change_vs_K8=float(maps[b].max()-maps[BRANCHES[0]].max()),full_peak_y=int(y),full_peak_x=int(x)))
            mask=None;patch_mask=None
            t=time.perf_counter()
            record=SimpleNamespace(path=str(paths['data']/sample['relative_path']),label=sample['label'],
                mask_path=str(paths['data']/gt_relative(sample['relative_path'])) if sample['label'] else None)
            # _load_mask provides the exact full-protocol nearest GT resize.
            full_mask=evaluator._load_mask(record,tuple(meta['output_shape']))
            if sample['label']:
                with Image.open(record.mask_path) as image:mask=np.asarray(image.convert('L'))>0
                if mask.shape!=(sample['image_height'],sample['image_width']):raise ValueError('GT/query geometry changed')
                if not mask.any():
                    status['incomplete_spatial_ids'].append(sample['sample_id'])
                else:patch_mask=backend.patch_mask(Image.fromarray(mask.astype(np.uint8)*255),tuple(meta['grid']))
            full_masks.append(full_mask)
            timing['GT_reading']+=time.perf_counter()-t
            if not sample['label'] or (mask is not None and mask.any()):
                ir,pr=spatial(sample,maps,mask,tuple(meta['grid']),backend)
                tables['defect_instances'].extend(ir);tables['peak_positions'].extend(pr)
            tables['mechanism_per_image'].extend(mechanism(sample,raws,np.asarray(backend.mad.scales),arrays['positions'],patch_mask))
            tables['interpolation_summary'].extend(interpolation_rows(sample,trace,raws,arrays['positions'],patch_mask))
            if sample['sample_id'] in selected:
                t=time.perf_counter()
                from .visualization import plot_case
                with Image.open(paths['data']/sample['relative_path']) as image:
                    plot_case(image.convert('RGB'),mask,maps,output/'figures'/f"{sample['sample_id']}.png")
                timing['figures']+=time.perf_counter()-t
            status['analyzed_images']+=1
            if index%25==0 or index==296:print(f"[LOCAL analyze] {index+1}/297",flush=True)
            del maps
        t=time.perf_counter()
        top10=[s['sample_id'] for s in cap['selected_cases'] if not s['label']]
        summaries,detail=ranking(tables['per_image_scores'],top10)
        details={(r['sample_id'],r['branch']):r for r in detail}
        for r in tables['per_image_scores']:r.update(details[(r['sample_id'],r['branch'])])
        from sklearn.metrics import roc_auc_score,average_precision_score
        baseline=next(r for r in summaries if r['branch']==BRANCHES[0])
        reference=formal['K8'];ids=sorted(reference)
        labels=[reference[i]['label'] for i in ids];scores=[reference[i]['image_score'] for i in ids]
        for key,expected in (('image_AUROC',roc_auc_score(labels,scores)),('image_AP',average_precision_score(labels,scores))):
            audit(tables['numerical_verification'],'background','formal_K8_summary',baseline[key],expected,score=True,metric=key)
        status['formal_baseline_reproduced']=all(r['verified'] for r in tables['numerical_verification'] if r['check_kind'].startswith('formal_'))
        timing['metrics']+=time.perf_counter()-t
        mechanism_rows=mechanism_summary(tables['mechanism_per_image'])
        replay.write_csv(output/'mechanism_summary.csv',mechanism_rows)
        tables['interpolation_summary'].extend(aggregate_interpolation(tables['interpolation_summary']))
        labels=[s['label'] for s in check['samples']]
        for branch in BRANCHES:
            r=next(r for r in summaries if r['branch']==branch)
            row=dict(branch=branch,image_count=297,image_AUROC=r['image_AUROC'],image_AP=r['image_AP'],
                pixel_AUROC=None,pixel_AP=None,AUPRO=None,pixel_status='not_executed',
                pixel_protocol='original full-protocol',aupro_backend='ADEval CUDA nstrips=200 FPR<=0.3')
            if pixel_available and status['formal_baseline_reproduced'] and not status['incomplete_spatial_ids']:
                print('[LOCAL metrics] '+branch,flush=True)
                # Only ONE branch's full maps is held at a time.
                branch_maps=[]
                for sample in check['samples']:
                    arrays,meta=read_original(sample);raws,_=read_new(sample,arrays)
                    t=time.perf_counter()
                    physical=scatter(raws[branch],arrays['positions'],tuple(meta['grid']))
                    branch_maps.append(backend.runner._method_map({'official_raw':physical},'MAD_EQUAL',backend.mad,tuple(meta['output_shape'])))
                    timing['maps']+=time.perf_counter()-t
                t=time.perf_counter()
                try:
                    status['pixel_evaluation_started']=True
                    values=evaluator._evaluate(labels,full_masks,branch_maps)
                    row.update(pixel_AUROC=values['pixel_AUROC'],pixel_AP=values['pixel_AUPR'],AUPRO=values['AUPRO'],pixel_status='complete')
                    for key,value_key in (('image_AUROC','image_AUROC'),('image_AP','image_AUPR')):
                        audit(tables['numerical_verification'],'background','full_protocol_image_metric',values[value_key],r[key],score=True,branch=branch,metric=key)
                except (ImportError,RuntimeError,ValueError) as error:
                    pixel_available=False;status['pixel_backend_error']=f'{type(error).__name__}: {error}'
                    row['pixel_status']='failed'
                timing['metrics']+=time.perf_counter()-t
                del branch_maps;gc.collect()
            metrics.append(row)
        status['pixel_metrics_complete']=all(r['pixel_status']=='complete' for r in metrics)
        verify_files(check['required_files'])
        for r in new_records:
            if replay.sha256(output/r['file'])!=r['sha256']:raise ValueError('new raw changed during evaluation')
        status.update(protected_inputs_unchanged=True,numerical_failed_checks=sum(not r['verified'] for r in tables['numerical_verification']),
            selected_figures=len(selected),GT_used_only_after_predictions_frozen=True)
        passed=(status['pixel_metrics_complete'] and status['formal_baseline_reproduced']
                and status['analyzed_images']==297 and not status['incomplete_spatial_ids'] and not status['numerical_failed_checks']
                and all(r['anomaly_image_count']==212 and r['normal_image_count']==85 for r in mechanism_rows))
        status['status']='analysis_complete' if passed else 'partial_evidence'
        rc=0 if passed else 3
    except (NumericalError,ImportError,OSError,ValueError,RuntimeError) as error:
        status.update(status='partial_evidence',error=f'{type(error).__name__}: {error}')
        rc=3
    finally:
        for name,rows in tables.items():replay.write_csv(output/f'{name}.csv',rows)
        replay.write_csv(output/'metrics.csv',metrics);replay.write_csv(output/'ranking_summary.csv',summaries)
        timing['total_analyze']=time.perf_counter()-started
        observations=memory_observation(cuda_evaluation_started=status['pixel_evaluation_started'])
        replay.write_csv(output/'timing.csv',[dict(stage=k,seconds=v,scope='offline_cached_analysis_only',**observations) for k,v in timing.items()])
        status.update(timing_scope='offline_cached_analysis_only',
            numerical_failed_checks=sum(not r['verified'] for r in tables['numerical_verification']),**observations)
        replay.atomic_json(output/'status.json',status)
        write_report(output,status,metrics,summaries)
    return rc


def build_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('preflight','analyze'),nargs='?',default='preflight')
    parser.add_argument('--project-root',default=str(Path(__file__).resolve().parents[3]))
    for key in ('capture-dir','results-dir','support-cache-dir','mad','output-dir'):parser.add_argument('--'+key)
    parser.add_argument('--data',default='/root/autodl-tmp/datasets/AeBAD/AeBAD_S')
    parser.add_argument('--no-write',action='store_true',help='preflight only: no files or directories are created')
    return parser


def main(argv=None):
    args=build_parser().parse_args(argv)
    if args.no_write and args.mode!='preflight':raise ValueError('--no-write is preflight only')
    try:check=preflight(args)
    except (OSError,ValueError) as error:
        print(f'PREFLIGHT_BLOCKED: {error}',file=sys.stderr);return 2
    if not args.no_write:
        output=Path(check['paths']['output']);output.mkdir(parents=True,exist_ok=True)
        replay.atomic_json(output/'preflight.json',check)
    print(json.dumps({k:v for k,v in check.items() if k not in ('capture_inputs','capture_status','required_files','samples','supports')},ensure_ascii=False,indent=2),flush=True)
    if check['errors']:return 2
    if args.mode=='preflight':return 0
    try:return analyze(args,check)
    except Exception as error:
        output=Path(check['paths']['output'])
        path=output/'status.json';state=json.loads(path.read_text()) if path.exists() else dict(tool=TOOL)
        state.update(status='analysis_failed',error=f'{type(error).__name__}: {error}')
        replay.atomic_json(path,state);print(state['error'],file=sys.stderr);return 1


if __name__=='__main__':raise SystemExit(main())
