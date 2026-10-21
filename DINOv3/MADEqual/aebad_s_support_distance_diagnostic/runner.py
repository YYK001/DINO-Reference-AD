"""Read-only default preflight. Explicit analyze only; all computation is CPU."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import numpy as np
from . import inputs as ip
from .profiles import FEATURES,LAYERS
from .conditional import VERSION,BOOTSTRAPS,BOOTSTRAP_SEED,describe

PROTOCOL=dict(tool=VERSION,scope='AeBAD-S/background_original_seed42',images=297,normal=85,anomaly=212,
    supports=8,layers=list(LAYERS),distance_unit='existing_euclidean_over_1280_once_no_further_division',
    normal_selection='ceil(1_percent_native_UNION_fused); ties_physical_flat_index',
    defect_selection='original_GT_adaptive_max_union_deduplicated_patches',
    patch_MAD='original_float32_subtract_median_divide_scale_no_new_fit',
    distance_audit_scale='original_float32_scale_plus_1e-8; float64_signed_difference',
    effective_reduction='U-S > 1e-7+1e-6*abs(U)',source_ties='abs(s-t)<=1e-7+1e-6*abs(s)',
    controls=['b','delta'],quantile_grid=[5,5],repeated_edges='merge',
    base_weights='within_group_equal_image_then_equal_selected_patch; effective subset_rebalances_images',
    boundaries='pooled_eligible_equal_group_mass_image_balanced_inverse_weighted_CDF',
    matched_weights='common_cell_mass=min(group_probabilities); normalize_common_mass; keep_base_within_cell',
    bootstrap_unit='whole_image_stratified_by_group; eligible_population',bootstrap_replicates=BOOTSTRAPS,
    bootstrap_seed=BOOTSTRAP_SEED,bootstrap_rng='NumPy_PCG64; stable_protocol_layer_SHA256',
    bootstrap_matching='frozen_edges_recompute_common_cell_masses_each_draw',CI='percentile_2.5_97.5',
    diagnostic_AUROC_positive='GT_covered; no direction flipping',features=list(FEATURES),
    ring_original_pixels=16,ring_metric='Chebyshev_exclude_all_GT',margin_classification_atol=1e-6,
    historical_score_atol=1e-5,historical_score_rtol=0,stencil='original_all_nonzero_bilinear_weights',
    figures_cases=['test_0977','test_0023','test_0545'],figures_features=['gap8','effective_support_count'],
    residual_control_SMD_flag=.1,residual_flag_role='description_only_not_a_gate',
    new_detector=False,Target_MAD_fit=False,model_inference=False,pixel_metric_evaluation=False)


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',nargs='?',default='preflight',choices=('preflight','analyze'))
    for k,v in ip.defaults().items():p.add_argument('--'+k.replace('_','-'),type=Path,default=v)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--cpu-workers',type=int,choices=(1,2,3,4),default=2)
    return p


def paths(args):return {k:getattr(args,k) for k in ip.defaults()}


def environment():
    result={}
    for n in ('numpy','scipy','Pillow','psutil','matplotlib'):
        try:result[n]=importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:result[n]=None
    return result


def preflight(args):
    started=time.perf_counter();p=paths(args)
    report=dict(tool=VERSION,status='blocked',errors=[],missing_inputs=[],expected_images=297,
        expected_capture_files=297,expected_segment_files=1188,dependencies=environment(),
        model_required=False,CH1_required=False,Source_bank_required=False,GPU_required=False,
        formal_analysis_executed=False)
    required=[p[k] for k in ('raw_archive','review_archive','formal_dir','data')]
    report['missing_inputs']=[str(f) for f in required if not f.exists()]
    try:
        if report['missing_inputs']:raise FileNotFoundError('see missing_inputs')
        if any(v is None for v in report['dependencies'].values()):raise ImportError('see dependencies')
        ctx=ip.context(p);count=0
        with ip.Archive(p['raw_archive'],ctx['raw_index']) as a:
            for sample in ctx['samples']:
                raw,meta,segments=ip.load_image(ctx,a,sample)
                u,s=ip.original_distances(raw,meta,segments)
                if not np.array_equal(u,np.minimum(raw['source_distance'],raw['target_distance'].min(axis=0))):raise ValueError('UNION reconstruction mismatch')
                count+=1
        rasters=ip.verify_data(ctx)
        disk_target=(args.output_dir or ip.ROOT/'DINOv3/outputs/mad_equal').resolve()
        while not disk_target.exists():disk_target=disk_target.parent
        disk_free=shutil.disk_usage(disk_target).free
        report.update(verified_capture_files=count,verified_segment_files=4*count,verified_required_rasters=rasters,
            supports=ctx['capture']['supports'],normal=85,anomaly=212,original_instances=388,
            assets=ctx['assets'],optional_history_used=bool(ctx['assets']['optional_history']),
            resources=dict(available_ram_bytes=ip.available_ram(),free_disk_bytes=disk_free,estimated_RAM_bytes=1536*2**20,
                recommended_RAM_headroom_bytes=2*2**30,estimated_disk_headroom_bytes=2*2**30,
                cpu_workers=args.cpu_workers,bootstrap_replicates=BOOTSTRAPS,
                scope='estimates; bounded per-image caches, no full-pixel sort, no extracted raw copies'))
        if ip.available_ram()<768*2**20:raise MemoryError('less than 768 MiB available RAM')
        if disk_free<2*2**30:raise OSError('less than 2 GiB output disk headroom')
        report['status']='inputs_verified_not_a_formal_analysis'
    except Exception as exc:report['errors'].append(f'{type(exc).__name__}: {exc}')
    report['torch_imported']='torch' in sys.modules
    if report['torch_imported']:
        report['status']='blocked';report['errors'].append('Torch imported: CPU isolation violated')
    report['seconds']=time.perf_counter()-started
    return report


def resource_record(start):
    import psutil
    m=psutil.Process().memory_info();peak=getattr(m,'peak_wset',None)
    if peak is None:
        import resource
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    return dict(seconds=time.perf_counter()-start,process_peak_rss_bytes=peak,torch_imported='torch' in sys.modules)


def analyze(args):
    from . import analysis as an,figures
    from DINOv3.MADEqual.aebad_s_neighborhood_segment_p0.prediction import bounded_images
    started=time.perf_counter();ctx=ip.context(paths(args))
    ip.verify_data(ctx)
    if ip.available_ram()<768*2**20:raise MemoryError('insufficient RAM headroom')
    output=args.output_dir or ip.ROOT/'DINOv3/outputs/mad_equal'/('aebad_s_support_distance_diagnostic_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out=ip.protect(ctx,output)
    parent=out.parent
    while not parent.exists():parent=parent.parent
    if shutil.disk_usage(parent).free<2*2**30:raise OSError('less than 2 GiB output headroom')
    out.mkdir(parents=True);(out/'figures').mkdir()
    ip.write_json(out/'protocol.json',PROTOCOL)
    frozen=dict(protocol=PROTOCOL,assets=ctx['assets'],capture_identity=ip.CAP_ID,segment_identity=ip.SEG_ID,
        source_patch_medians=ctx['medians'].tolist(),source_patch_scales=ctx['scales'].tolist(),
        samples=ctx['samples'],supports=ctx['capture']['supports'],GT_sha256=ctx['capture']['gt_sha256'],
        code={p.name:ip.sha(p) for p in Path(__file__).parent.glob('*.py')},
        shared_code={n:ip.sha(Path(__file__).parents[1]/n) for n in (
            'aebad_s_neighborhood_segment_p0/inputs.py','aebad_s_neighborhood_segment_p0/prediction.py',
            'aebad_s_neighborhood_segment_p0/analysis.py','aebad_s_projection_layer_diagnostic/numerics.py',
            'aebad_s_union_diagnostics/spatial.py','aebad_s_union_support_loo/analysis.py')},environment=environment())
    ip.write_json(out/'input_manifest.json',dict(identity=ip.digest(frozen),frozen=frozen,paths={k:str(v) for k,v in ctx['paths'].items()}))
    status=dict(tool=VERSION,status='running',images=0,model_inference=False,GPU_used=False,Target_MAD_fitted=False)
    ip.write_json(out/'status.json',status)
    checks=[];instances=[];chunks={l:[] for l in LAYERS};timing={};plots={}
    previous_mpl=os.environ.get('MPLCONFIGDIR')
    try:
        # Every Matplotlib cache and temporary file stays under the new G: output.
        with tempfile.TemporaryDirectory(prefix='mpl_',dir=out) as mpl:
            os.environ['MPLCONFIGDIR']=mpl
            import matplotlib
            matplotlib.use('Agg')
            tick=time.perf_counter()
            with an.GzipRows(out/'patch_profiles.csv.gz') as profiles,an.GzipRows(out/'patch_instance_links.csv.gz') as links,an.GzipRows(out/'peak_stencil_profiles.csv.gz') as stencils:
                def work(pair):
                    i,s=pair;return an.one_image(ctx,s,i)
                for result in bounded_images(list(enumerate(ctx['samples'])),work,args.cpu_workers):
                    checks.extend(result['checks'])
                    if result['failed']:
                        ip.write_csv(out/'numerical_verification.csv',checks)
                        raise ValueError('historical map/instance/stencil reproduction failed; no tolerance relaxation')
                    for r in result['selected']:profiles.write(r)
                    for r in result['links']:links.write(r)
                    for r in result['stencils']:stencils.write(r)
                    for j,l in enumerate(LAYERS):chunks[l].append(result['columns'][j])
                    instances.extend(result['instances']);status['images']+=1
                    if result['figure'] is not None:figures.case(ctx,result['figure'],out/'figures'/(result['sample_id']+'.png'))
                    if status['images']%25==0:print(f'[support distance profiles] {status["images"]}/297',flush=True)
            timing['native_profiles_and_instances_seconds']=time.perf_counter()-tick
            expected={(r['sample_id'],int(r['instance_id'])) for r in ctx['history']['defect_instances'] if r['branch']=='K8_UNION'}
            actual=[(r['sample_id'],r['instance_id']) for r in instances]
            if len(actual)!=388 or len(set(actual))!=len(actual) or set(actual)!=expected:raise ValueError('incomplete original instance coverage')
            ip.write_csv(out/'instance_profile_analysis.csv',instances)
            summary=an.instance_summaries(instances);ip.write_csv(out/'instance_profile_summary.csv',summary)
            combined={l:{k:np.concatenate([c[k] for c in chunks[l]]) for k in an.STAT_FIELDS} for l in LAYERS};chunks.clear()
            tick=time.perf_counter();profile_summary=[];comparisons=[];coverage=[];bins={}
            def compare(layer):
                return layer,describe(combined[layer],layer,progress=lambda n:print(f'[support distance bootstrap] L{layer} {n}/200 images resamples',flush=True))
            with ThreadPoolExecutor(max_workers=min(args.cpu_workers,4)) as pool:
                for layer,value in pool.map(compare,LAYERS):
                    a,b,c,d,p=value;profile_summary.extend(a);comparisons.extend(b);coverage.extend(c);bins[str(layer)]=d;plots[layer]=p
                    print(f'[support distance conditional] L{layer} 200 image bootstraps complete',flush=True)
            timing['conditional_bootstrap_seconds']=time.perf_counter()-tick
            ip.write_csv(out/'profile_summary.csv',profile_summary);ip.write_csv(out/'conditional_comparisons.csv',comparisons)
            ip.write_csv(out/'matching_coverage.csv',coverage);ip.write_json(out/'matching_bins.json',bins)
            ip.write_csv(out/'image_index.csv',[dict(image_id=i,**s) for i,s in enumerate(ctx['samples'])])
            tick=time.perf_counter();figures.cdfs(plots,out/'figures');an.make_report(out,comparisons,coverage,instances,summary)
            timing['figures_and_report_seconds']=time.perf_counter()-tick
        ip.write_csv(out/'numerical_verification.csv',checks)
        if status['images']!=297 or not checks or any(not r['verified'] for r in checks) or 'torch' in sys.modules:
            raise ValueError('incomplete or non-CPU diagnosis')
        # Recheck immutable metadata, without rebuilding anything.
        for n,h in ctx['assets']['formal'].items():
            if ip.sha(ctx['paths']['formal_dir']/n)!=h:raise ValueError('formal input changed during analysis')
        if ip.sha(ctx['paths']['raw_archive'])!=ip.ARCHIVE_SHA or ip.sha(ctx['paths']['review_archive'])!=ip.REVIEW_SHA:
            raise ValueError('parent archive changed during analysis')
        status.update(status='diagnostic_complete',instances=388,numerical_failed_checks=0,torch_imported=False,
            conditional_unavailable_rows=sum(r.get('status')!='available' for r in comparisons),
            interpretation='complete_protocol_not_a_claim_of_group_separability')
    except Exception as exc:
        ip.write_csv(out/'numerical_verification.csv',checks)
        status.update(status='diagnostic_failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if previous_mpl is None:os.environ.pop('MPLCONFIGDIR',None)
        else:os.environ['MPLCONFIGDIR']=previous_mpl
        timing.update(resource_record(started),cpu_workers=args.cpu_workers)
        ip.write_json(out/'timing.json',timing)
        status.update(resource_record(started))
        if status['status']=='diagnostic_complete':
            status['files']={p.relative_to(out).as_posix():ip.sha(p) for p in out.rglob('*') if p.is_file() and p.name!='status.json'}
        status['output_bytes_before_status']=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
        ip.write_json(out/'status.json',status)
    print(json.dumps(dict(status=status['status'],output=str(out),seconds=status['seconds']),ensure_ascii=False),flush=True)


def main(argv=None):
    args=parser().parse_args(argv)
    if args.stage=='preflight':
        r=preflight(args);print(json.dumps(r,ensure_ascii=False,indent=2),flush=True)
        return 0 if r['status']=='inputs_verified_not_a_formal_analysis' else 2
    try:analyze(args);return 0
    except Exception as exc:
        print(json.dumps(dict(tool=VERSION,status='diagnostic_failed',error=f'{type(exc).__name__}: {exc}'),ensure_ascii=False),flush=True);return 2


if __name__=='__main__':raise SystemExit(main())
