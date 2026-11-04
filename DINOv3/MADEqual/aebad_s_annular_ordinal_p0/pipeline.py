"""One fixed population, one calibration and a gate before full computation."""
import time
import numpy as np
from . import inputs as ip,runtime as rt
from .protocol import BASE,FUSIONS,E_NAMES,BRANCHES,CONFIG
st=ip.st

def fit_calibration(P,E):
    P=np.asarray(P,np.float64);E=np.asarray(E,np.float64)
    if P.shape!=(66,) or E.shape!=(66,3) or not np.isfinite(P).all() or not np.isfinite(E).all():raise ValueError('complete finite Source66 calibration required')
    mu=float(P.mean());sigma=float(P.std(ddof=0));floor=64*np.finfo(np.float64).eps*max(1.,float(abs(P).max()))
    if sigma<=floor:raise ValueError('original P scale numerically degenerate; restore compatible inputs')
    raw=float(E[:,2].std(ddof=0));fallback=raw<=1e-6
    return dict(mu_P=mu,sigma_P=sigma,sigma_P_floor=floor,sigma_E_raw=raw,sigma_E=1. if fallback else raw,
        E_scale_degenerate=fallback,ddof=0,shared_E_scale='E_ST_MULTI',E_centering=False,source_images=66)

def scores(P,E,calibration):
    E=np.asarray(E,np.float64)
    if E.shape!=(3,) or not np.isfinite(E).all() or not np.isfinite(P):raise ValueError('finite three-branch evidence required')
    base=(P-calibration['mu_P'])/calibration['sigma_P']
    result={BASE:float(P)}
    result.update(zip(FUSIONS,(base+E/calibration['sigma_E']).tolist()));result.update(zip(E_NAMES,E.tolist()))
    return result

def estimate(a):
    x=ip.assets(a);rows=x.rows('calibration')+x.rows('evaluation')
    smoke=st.sealed_json(a.run_root/'smoke.json');measured=smoke['images']
    measured_windows=sum(r['windows'] for r in measured);measured_seconds=sum(r['seconds'] for r in measured)
    unfinished=[r for r in rows if rt.completed(a,r) is None]
    pending_windows=sum(len(ip.px.windows((r['image_height'],r['image_width']))) for r in unfinished)
    index=st.sealed_json(a.cache_root/'index/manifest.json')
    seconds_per_window=measured_seconds/measured_windows if measured_windows else None
    pending=seconds_per_window*pending_windows if seconds_per_window is not None else None
    done=[r for r in rows if rt.completed(a,r) is not None]
    already=sum(rt.completed(a,r)['seconds'] for r in done)
    result=dict(status='partial_real_asset_estimate',Source_index_images=455,Target_index_images=8,
        calibration_images=66,evaluation_images=289,measured_full_images=len(measured),measured_windows=measured_windows,
        index_build_seconds=index['seconds'],cold_index_load_seconds=smoke['cold_index_load_seconds'],
        measured_image_seconds=measured_seconds,seconds_per_window=seconds_per_window,pending_images=len(unfinished),
        pending_windows=pending_windows,already_completed_image_seconds=already,pending_pixel_seconds=pending,
        projected_pixel_total_seconds=None if pending is None else index['seconds']+smoke['cold_index_load_seconds']+already+pending,
        measured_compressed_bytes_per_window=sum(r['bytes'] for r in measured)/max(1,measured_windows),
        pending_compressed_bytes=sum(r['bytes'] for r in measured)/max(1,measured_windows)*pending_windows,
        unknown=['CPU_image_evaluation_and_report_IO','cross_image_matching_and_compression_variability'],
        unknown_is_not_zero=True,scope='fixed four complete normal calibration images; no all-population speed guarantee',
        reused_completed_images_not_counted_pending=True)
    st.write_json(a.run_root/'cost_estimate.json',result);return result

def guard(a):
    smoke=st.sealed_json(a.run_root/'smoke.json')
    if smoke['identity']!=ip.identity(a) or not smoke['valid_multi_evidence']:raise RuntimeError('fixed normal smoke has no effective multi-reference evidence; inspect matching counters, no full run')
    result=estimate(a);total=result['projected_pixel_total_seconds']
    if total is None:raise RuntimeError('full-image throughput unknown; no unmeasured work treated as zero')
    if total>43200 and not a.allow_over_12h:
        st.write_json(a.run_root/'budget_pause.json',dict(status='pending_explicit_12h_consent',estimate=result))
        raise RuntimeError('projected known work exceeds 12h; explicit user consent required, fixed assets retained')
    return result

def smoke_status(a,saved):
    projection=estimate(a)
    if not saved['valid_multi_evidence']:
        return dict(saved,status='pending',reason='four normal images have no effective multi-reference evidence; inspect matching counters')
    total=projection['projected_pixel_total_seconds']
    if total is None or (total>43200 and not a.allow_over_12h):
        st.write_json(a.run_root/'budget_pause.json',dict(status='pending_explicit_12h_consent' if total is not None else 'unknown_throughput',estimate=projection))
        return dict(saved,status='pending',reason='review full-image timing before formal computation')
    return saved

def smoke(a):
    path=a.run_root/'smoke.json'
    if path.exists():
        old=st.sealed_json(path)
        if old['identity']!=ip.identity(a):raise ValueError('smoke dependencies differ')
        return smoke_status(a,old)
    engine=rt.engine(a);engine.store.load();load=engine.store.load_seconds
    rows=sorted(ip.assets(a).rows('calibration'),key=lambda r:r['sample_id'])[:4]
    measured=[engine.process(row,case=j==0) for j,row in enumerate(rows)]
    value=dict(identity=ip.identity(a),status='normal_smoke_complete',cold_index_load_seconds=load,
        images=measured,valid_multi_evidence=any(r['valid_windows'][2]>0 for r in measured),
        nonzero_E_is_not_required=True,scope='four full Source calibration images, no labels/metrics, reused by calibration')
    saved=st.sealed_json(path,value)
    if not value['valid_multi_evidence']:st.write_json(a.run_root/'status.json',dict(status='smoke_no_multi_evidence',pending=['scientific review; do not loosen matching rules']))
    return smoke_status(a,saved)

def calibrate(a):
    guard(a);path=a.run_root/'calibration.json'
    if path.exists():
        old=st.sealed_json(path)
        if old['identity']!=ip.identity(a):raise ValueError('calibration dependencies differ')
        return old
    x=ip.assets(a);rows=x.rows('calibration');records=[rt.engine(a).process(row) for row in rows]
    P=[x.P[r['relative_path']] for r in rows];E=[[r['E'][k] for k in E_NAMES] for r in records]
    model=fit_calibration(P,E)
    st.write_csv(a.run_root/'calibration_stats.csv',[model])
    st.write_csv(a.run_root/'calibration_evidence.csv',[dict(sample_id=row['sample_id'],P=p,**record['E'],multi_valid_windows=record['valid_windows'][2]) for row,p,record in zip(rows,P,records)])
    return st.sealed_json(path,dict(identity=ip.identity(a),model=model,records=[r['seal'] for r in records]))

def infer(a):
    guard(a);path=a.run_root/'scores_frozen.json'
    if path.exists():
        old=st.sealed_json(path)
        if old['identity']!=ip.identity(a):raise ValueError('frozen scores dependencies differ')
        return old
    x=ip.assets(a);calibration=st.sealed_json(a.run_root/'calibration.json')
    if calibration['identity']!=ip.identity(a):raise ValueError('calibration identity differs')
    records=[];summary=[]
    for j,row in enumerate(x.rows('evaluation')):
        r=rt.engine(a).process(row,case=j<2);P=x.P[row['relative_path']]
        records.append(dict(image=ip.image(row),sample_id=row['sample_id'],P=P,
            standardized_P=(P-calibration['model']['mu_P'])/calibration['model']['sigma_P'],
            scores=scores(P,[r['E'][k] for k in E_NAMES],calibration['model']),evidence_record=r['seal']))
        summary.append(dict(sample_id=row['sample_id'],windows=r['windows'],**r['counters'],
            **{f'{name}_{i}':v for name in ('valid_windows','agreement_fraction','disagreement_fraction','mean_distinct_references','evidence_absent','reference_used_windows') for i,v in enumerate(r[name])},P=P,**r['E']))
        print(f'[annular image] {j+1}/289 {row["sample_id"]} three_evidences_one_outer_search',flush=True)
    # No label or GT field enters the frozen score artifact.
    st.write_csv(a.run_root/'per_image_scores.csv',[dict(sample_id=r['sample_id'],P=r['P'],standardized_P=r['standardized_P'],**r['scores']) for r in records])
    st.write_csv(a.run_root/'matching_summary.csv',summary)
    return st.sealed_json(path,dict(identity=ip.identity(a),calibration=calibration['seal'],records=records,GT_used=False))

def evaluate(a):
    from DINOv3.MADEqual.aebad_s_ch1_scoring_inheritance.scoring import image_metrics
    from DINOv3.MADEqual.aebad_s_conditional_image_score_p0.numerics import ranking
    frozen=st.sealed_json(a.run_root/'scores_frozen.json');x=ip.assets(a)
    if frozen['identity']!=ip.identity(a) or len(frozen['records'])!=289 or any(r['image']!=ip.image(row) for r,row in zip(frozen['records'],x.evaluation)):raise ValueError('freeze all 289 scores before label access')
    labels=[r['label'] for r in x.evaluation];original=[r['P'] for r in frozen['records']]
    baseline=image_metrics(labels,original)
    parent=[r for r in x.metrics if r['branch']=='SEGMENT_K8' and r['scoring']=='CH1']
    if len(parent)!=1 or any(abs(baseline[k]-float(parent[0][k]))>1e-12 for k in baseline):raise ValueError('original SEGMENT CH1 metrics differ; no tolerance change')
    rows=[];deltas=[];pairs=[]
    for branch in BRANCHES:
        values=[r['scores'][branch] for r in frozen['records']];metric=image_metrics(labels,values)
        rows.append(dict(branch=branch,images=289,**metric,role='primary' if branch==CONFIG['primary'] else 'fixed_control_or_auxiliary'))
        deltas.append(dict(branch=branch,**{k:metric[k]-baseline[k] for k in metric}))
        if branch!=BASE:pairs.append(dict(branch=branch,**ranking(labels,original,values)))
    coverage=[]
    for label,name in ((0,'normal'),(1,'anomaly')):
        records=[rt.completed(a,row) for row in x.rows('evaluation') if next(v['label'] for v in x.evaluation if v['relative_path']==row['relative_path'])==label]
        coverage.append(dict(group=name,images=len(records),multi_images_with_evidence=sum(r['valid_windows'][2]>0 for r in records),
            multi_valid_windows=sum(r['valid_windows'][2] for r in records),windows=sum(r['windows'] for r in records)))
    st.write_csv(a.run_root/'metrics.csv',rows);st.write_csv(a.run_root/'fixed_deltas.csv',deltas)
    st.write_csv(a.run_root/'ranking_pair_changes.csv',pairs);st.write_csv(a.run_root/'coverage_after_score_freeze.csv',coverage)
    return st.sealed_json(a.run_root/'evaluation.json',dict(identity=ip.identity(a),scores=frozen['seal'],metrics=rows,
        delta=deltas,ranking_pairs=pairs,coverage=coverage,baseline_exact=True,new_pixel_jobs=0,new_AUPRO_jobs=0))

def report(a):
    from .reporting import report as write
    return write(a)
