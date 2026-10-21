"""Default inventory; explicit CPU analyze and pending-aware summarize."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone,timedelta
import json
import os
import shutil
from pathlib import Path
import tempfile
import sys
import time
import numpy as np
from . import inputs as ip,discovery,reporting
from DINOv3.MADEqual.aebad_s_support_distance_diagnostic import conditional as kernel
from DINOv3.MADEqual.aebad_s_support_distance_diagnostic.runner import PROTOCOL as BASE_PROTOCOL,resource_record
from DINOv3.MADEqual.aebad_s_support_distance_diagnostic.analysis import GzipRows,STAT_FIELDS

VERSION='support_distance_crossdomain_v1'


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',nargs='?',default='preflight',choices=('preflight','analyze','summarize'))
    p.add_argument('--summary',type=Path,default=ip.ROOT/'downloads'/ip.SUMMARY_NAME)
    p.add_argument('--parent-run',type=Path,help='explicit NEW completed pipeline root: cache/prediction/cpu/frozen_inputs; keeps its own hashes')
    p.add_argument('--data',type=Path,default=ip.ROOT/'datasets/AeBAD.zip')
    p.add_argument('--background',type=Path,default=ip.ROOT/'DINOv3/outputs/mad_equal/aebad_s_support_distance_diagnostic_20261021_054307')
    p.add_argument('--search-root',action='append',type=Path)
    p.add_argument('--cache-root',action='append',type=Path,default=[])
    p.add_argument('--raw-archive',action='append',type=Path,default=[])
    p.add_argument('--search-index',type=Path)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--results-root',type=Path,help='analyze root containing illumination/status.json and view/status.json')
    p.add_argument('--cpu-workers',type=int,choices=(1,2,3,4),default=2)
    return p


def fresh(args,stage):
    stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    out=(args.output_dir or ip.ROOT/'DINOv3/outputs/mad_equal'/f'aebad_s_support_distance_crossdomain_{stage}_{stamp}').resolve()
    if out.exists():raise FileExistsError('fresh output required; no overwriting')
    protected=[args.summary.resolve(),args.data.resolve(),args.background.resolve(),
        *[p.resolve() for p in args.cache_root],*[p.resolve() for p in args.raw_archive]]
    if args.results_root:protected.append(args.results_root.resolve())
    if args.parent_run:
        # A fresh diagnostic child is allowed beside these frozen stages; it
        # must never overlap the stages that LiveParent actually reads.
        protected.extend((args.parent_run/name).resolve() for name in
                         ('cache','prediction','cpu','frozen_inputs','run_config.json'))
    if any(out==p or out.is_relative_to(p) or p.is_relative_to(out) for p in protected):raise ValueError('output overlaps protected inputs')
    if out.drive and out.drive.upper()!='G:':raise ValueError('Windows output/cache must remain on G:')
    out.mkdir(parents=True);return out


def verified_sources(found,parent):
    candidates=[];success={};errors={}
    samples={(s['domain'],s['sample_id']):s for d in ip.DOMAINS for s in parent.catalog['domains'][d]['samples']}
    for row in found:
        for source in row['candidates']:
            candidates.append({k:v for k,v in row.items() if k not in ('candidates','storage','path','member')}|source)
    for record,blob in discovery.blobs(candidates):
        key=record['domain'],record['sample_id']
        try:
            ip.decode(blob,parent,samples[key],parent.records[key])
            success.setdefault(key,{k:v for k,v in record.items() if k!='candidates'})
        except (ValueError,KeyError) as exc:errors.setdefault(key,[]).append(f'{record["path"]}::{record["member"]}: {exc}')
    invalid=[]
    for row in found:
        key=row['domain'],row['sample_id']
        if key not in success:
            invalid.append({k:v for k,v in row.items() if k!='candidates'}|dict(status='untrusted_or_incomplete_capture',reason='; '.join(errors.get(key,[]))))
    return list(success.values()),invalid


def inventory(args,out):
    started=time.perf_counter()
    if args.parent_run:
        from .live_parent import LiveParent
        parent=LiveParent(args.parent_run)
    else:parent=ip.Parent(args.summary)
    required=parent.required();ip.write_csv(out/'required_cache_files.csv',required)
    roots=args.search_root or ([] if args.parent_run else [p for p in (ip.ROOT/'downloads',ip.ROOT/'server_archives',ip.ROOT/'server_results',ip.ROOT/'DINOv3/outputs') if p.exists()])
    previous=ip.old.read_json(args.search_index) if args.search_index else None
    cache_roots=args.cache_root+([parent.cache_root] if args.parent_run else [])
    found,missing,index=discovery.discover(required,roots,cache_roots,args.raw_archive,previous)
    verified,invalid=verified_sources(found,parent);missing+=invalid
    ip.write_json(out/'archive_index.json',index);ip.write_csv(out/'found_cache_files.csv',verified)
    gt_count=0
    if not args.data.exists():missing.append(dict(domain='both',status='missing_GT_asset',relative_path=str(args.data),reason='data ZIP/root required for GT mapping'))
    else:
        try:
            gt_count,gt_missing=ip.verify_data(parent,args.data);missing+=gt_missing
        except (OSError,KeyError,ValueError) as exc:
            missing.append(dict(domain='both',status='missing_GT_asset',relative_path=str(args.data),reason=str(exc)))
    ip.write_csv(out/'missing_inputs.csv',missing or [dict(status='none_missing')])
    ip.write_json(out/'parent_historical_audit.json',parent.historical_audit)
    frozen=dict(summary_sha256=parent.source_sha256,parent_origin=parent.origin,original_cache_identity=parent.identity,
        required_cache_files=required,verified_sources=verified,
        source_patch_MAD_sha256=parent.catalog['formal_sha256']['source_mad_state.npz'],
        data=str(args.data.resolve()),code={p.name:ip.sha(p) for p in Path(__file__).parent.glob('*.py')})
    ip.write_json(out/'input_manifest.json',dict(identity=ip.digest(frozen),frozen=frozen,
        purpose='input_inventory_only_not_formal_analysis'))
    state=dict(tool=VERSION,stage='preflight',status='inputs_ready' if not missing else 'missing_inputs',
        required_cache_files=629,verified_cache_files=len(verified),missing_cache_files=629-len(verified),verified_GT_files=gt_count,
        domains={d:dict(normal=ip.COUNTS[d][0],anomaly=ip.COUNTS[d][1],images=sum(ip.COUNTS[d])) for d in ip.DOMAINS},
        parent_identity=parent.identity,parent_summary_sha256=parent.source_sha256,parent_origin=parent.origin,
        parent_historical_audit=parent.historical_audit,formal_analysis_executed=False,
        GPU_required=False,Torch_required=False,model_required=False,memory_entity_required=False,
        torch_imported='torch' in sys.modules,seconds=time.perf_counter()-started,
        original_cache_root_recorded_not_accessed=parent.run_config['paths']['cache_root'],
        search_roots=[str(p) for p in roots],resources=dict(available_RAM_bytes=ip.base.available_ram(),
            estimated_analysis_headroom_RAM_bytes=2*2**30,estimated_output_headroom_disk_bytes=3*2**30))
    if state['torch_imported']:raise RuntimeError('Torch import forbidden')
    ip.write_json(out/'status.json',state)
    return parent,verified,state


def protocol(domain):
    result=dict(BASE_PROTOCOL)
    result.update(tool=VERSION,scope='AeBAD-S/'+domain,images=sum(ip.COUNTS[domain]),normal=ip.COUNTS[domain][0],anomaly=ip.COUNTS[domain][1],
        main_features=list(ip.MAIN_FEATURES),auxiliary_features=[k for k in kernel.FEATURES if k not in ip.MAIN_FEATURES],
        figures_cases=[],figures_features=list(ip.MAIN_FEATURES),small_instances='original_domain_smallest_area_quartile',
        historical_score_atol=1e-4,historical_strict_atol=1e-5,numerical_reconstruction_atol=1e-5,
        bin_edges='independently_generated_with_unchanged_background_rule_per_domain',
        identity_policy='old_product_identity_separate_from_new_analysis_code')
    return result


def analyze(args,out,parent,found):
    from . import analysis as an
    from collections import deque
    started=time.perf_counter();state={};checks={d:[] for d in ip.DOMAINS};instances={d:[] for d in ip.DOMAINS}
    chunks={d:{l:[] for l in ip.LAYERS} for d in ip.DOMAINS}
    base=ip.ROOT/'DINOv3/MADEqual/aebad_s_support_distance_diagnostic'
    kernels={n:ip.sha(base/n) for n in ('profiles.py','conditional.py')}
    shared=[base/n for n in ('profiles.py','conditional.py','spatial.py','analysis.py','inputs.py')]
    shared += [ip.ROOT/'DINOv3/MADEqual'/p for p in (
        'aebad_s_ch1_scoring_inheritance/scoring.py','aebad_s_projection_layer_diagnostic/numerics.py',
        'aebad_s_union_diagnostics/spatial.py','aebad_s_segment_pair_crossdomain_p0/verification.py',
        'aebad_s_segment_pair_crossdomain_p0/inputs.py','aebad_s_segment_pair_crossdomain_p0/method.py')]
    if ip.base.available_ram()<2*2**30 or shutil.disk_usage(out).free<3*2**30:
        raise MemoryError('CPU analysis requires 2 GiB available RAM and 3 GiB disk headroom')
    sample_map={(s['domain'],s['sample_id']):(i,s) for d in ip.DOMAINS for i,s in enumerate(parent.catalog['domains'][d]['samples'])}
    previous=os.environ.get('MPLCONFIGDIR')
    try:
        with tempfile.TemporaryDirectory(dir=out,prefix='mpl_') as config,ExitStack() as stack:
            os.environ['MPLCONFIGDIR']=config
            import matplotlib
            matplotlib.use('Agg')
            writers={}
            for d in ip.DOMAINS:
                root=out/d;root.mkdir();(root/'figures').mkdir()
                ip.write_json(root/'protocol.json',protocol(d))
                frozen=dict(domain=d,protocol=protocol(d),summary_sha256=getattr(parent,'source_sha256',ip.SUMMARY_SHA),
                    parent_origin=getattr(parent,'origin','historical_summary_archive'),original_cache_identity=parent.identity,
                    statistical_kernels=kernels,samples=parent.catalog['domains'][d]['samples'],supports=parent.catalog['domains'][d]['supports'],
                    shared_CPU_code={p.relative_to(ip.ROOT).as_posix():ip.sha(p) for p in shared},
                    source_patch_MAD_sha256=parent.catalog['formal_sha256']['source_mad_state.npz'],
                    raw_records=[r for r in parent.required() if r['domain']==d],
                    code={p.name:ip.sha(p) for p in Path(__file__).parent.glob('*.py')})
                ip.write_json(root/'input_manifest.json',dict(identity=ip.digest(frozen),frozen=frozen))
                state[d]=dict(status='running',domain=d,images=0,original_cache_identity=parent.identity)
                writers[d]={n:stack.enter_context(GzipRows(root/n)) for n in ('patch_profiles.csv.gz','patch_instance_links.csv.gz','peak_stencil_profiles.csv.gz')}
            def work(item):
                r,blob=item;i,s=sample_map[r['domain'],r['sample_id']]
                return an.one_image(parent,s,parent.records[r['domain'],r['sample_id']],blob,args.data,i)
            with ThreadPoolExecutor(max_workers=args.cpu_workers) as pool:
                stream=iter(discovery.blobs(found));pending=deque()
                for _ in range(args.cpu_workers):
                    item=next(stream,None)
                    if item is not None:pending.append(pool.submit(work,item))
                while pending:
                    r=pending.popleft().result();d=r['domain'];checks[d].extend(r['checks'])
                    if r['failed']:raise ValueError('same-cache reconstruction or original compatibility failed; no tolerance relaxation')
                    for x in r['selected']:writers[d]['patch_profiles.csv.gz'].write(x)
                    for x in r['links']:writers[d]['patch_instance_links.csv.gz'].write(an.scoped(x,d))
                    for x in r['stencils']:writers[d]['peak_stencil_profiles.csv.gz'].write(an.scoped(x,d))
                    instances[d].extend(an.scoped(x,d) for x in r['instances'])
                    for j,l in enumerate(ip.LAYERS):chunks[d][l].append(r['columns'][j])
                    state[d]['images']+=1
                    if state[d]['images']%25==0:print(f'[crossdomain profiles] {d} {state[d]["images"]}/{sum(ip.COUNTS[d])}',flush=True)
                    item=next(stream,None)
                    if item is not None:pending.append(pool.submit(work,item))
            stack.close()
            for d in ip.DOMAINS:
                root=out/d;expected={(s,i) for dd,s,i,b in parent.instances if dd==d and b=='UNION_K8_CH1'}
                actual={(r['sample_id'],r['instance_id']) for r in instances[d]}
                if state[d]['images']!=sum(ip.COUNTS[d]) or actual!=expected or len(instances[d])!=len(expected):raise ValueError('domain image/instance coverage incomplete')
                ip.write_csv(root/'instance_profile_analysis.csv',instances[d]);ip.write_csv(root/'instance_profile_summary.csv',an.summarize_instances(instances[d],d))
                ip.write_json(root/'parent_historical_audit.json',parent.historical_audit)
                summary=[];comparisons=[];coverage=[];bins={};plots={}
                for l in ip.LAYERS:
                    data={k:np.concatenate([c[k] for c in chunks[d][l]]) for k in STAT_FIELDS}
                    a,b,c,edges,p=kernel.describe(data,l,progress=lambda n:print(f'[crossdomain bootstrap] {d} L{l} {n}/200',flush=True))
                    summary.extend(dict(domain=d,**r) for r in a);comparisons.extend(dict(domain=d,feature_role='primary' if r['feature'] in ip.MAIN_FEATURES else 'auxiliary',**r) for r in b)
                    coverage.extend(dict(domain=d,**r) for r in c);bins[str(l)]=edges;plots[l]=p
                ip.write_csv(root/'profile_summary.csv',summary);ip.write_csv(root/'conditional_comparisons.csv',comparisons)
                ip.write_csv(root/'matching_coverage.csv',coverage);ip.write_json(root/'matching_bins.json',bins)
                ip.write_csv(root/'image_index.csv',[dict(image_id=i,**s) for i,s in enumerate(parent.catalog['domains'][d]['samples'])])
                reporting.plots(plots,root/'figures',d);reporting.domain_report(root,d,comparisons,coverage,instances[d],parent.historical_audit)
                if 'torch' in sys.modules:raise RuntimeError('Torch import forbidden')
                state[d].update(status='diagnostic_complete',instances=len(instances[d]),numerical_failed_checks=0,torch_imported='torch' in sys.modules)
            from .live_parent import unchanged
            unchanged(parent)
    except Exception as exc:
        for d in ip.DOMAINS:
            if d in state:state[d].update(status='diagnostic_failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if previous is None:os.environ.pop('MPLCONFIGDIR',None)
        else:os.environ['MPLCONFIGDIR']=previous
        for d in ip.DOMAINS:
            if not (out/d).exists():continue
            root=out/d;ip.write_csv(root/'numerical_verification.csv',checks[d])
            state[d].update(resource_record(started),GPU_used=False,model_inference=False,pixel_metrics_evaluated=False,
                numerical_failed_checks=sum(not r['verified'] for r in checks[d]),
                resource_scope='shared two-domain analysis process, not independent per-domain peaks',
                output_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()))
            if state[d]['status']=='diagnostic_complete':state[d]['files']={p.relative_to(root).as_posix():ip.sha(p) for p in root.rglob('*') if p.is_file() and p.name!='status.json'}
            ip.write_json(root/'status.json',state[d])
    reporting.summarize(args.background,{d:out/d for d in ip.DOMAINS},out)


def main(argv=None):
    args=parser().parse_args(argv);out=fresh(args,args.stage)
    try:
        if args.stage=='summarize':
            reporting.summarize(args.background,{d:args.results_root/d for d in ip.DOMAINS} if args.results_root else {},out)
        else:
            parent,found,status=inventory(args,out)
            if args.stage=='analyze':
                if status['status']!='inputs_ready':raise FileNotFoundError('raw inputs incomplete; see missing_inputs.csv; no capture rebuild')
                analyze(args,out,parent,found)
            else:
                print(json.dumps(dict(output=str(out),**status),ensure_ascii=False),flush=True)
                return 0 if status['status']=='inputs_ready' else 2
        print(json.dumps(dict(output=str(out),stage=args.stage,status=ip.old.read_json(out/'status.json')['status']),ensure_ascii=False),flush=True)
        return 0
    except Exception as exc:
        ip.write_json(out/'failure.json',dict(stage=args.stage,error=f'{type(exc).__name__}: {exc}',torch_imported='torch' in sys.modules))
        if args.stage!='preflight':ip.write_json(out/'status.json',dict(stage=args.stage,status='diagnostic_failed',error=f'{type(exc).__name__}: {exc}'))
        print(json.dumps(dict(output=str(out),status='blocked',error=f'{type(exc).__name__}: {exc}'),ensure_ascii=False),flush=True)
        return 2


if __name__=='__main__':raise SystemExit(main())
