"""Reuse frozen LOO/SEGMENT validators, without CH1 or model prerequisites."""
from __future__ import annotations
import hashlib
import json
import zipfile
from pathlib import Path
import numpy as np
from DINOv3.MADEqual.aebad_s_neighborhood_segment_p0 import inputs as old
from DINOv3.MADEqual.aebad_s_neighborhood_segment_p0.inputs import (
    Archive,load_image,sha,digest,safe,read_csv,write_json,write_csv,available_ram,
    SROOT,CROOT,CAP_ID,SEG_ID,ARCHIVE_SHA,REVIEW_SHA,
)
from DINOv3.MADEqual.aebad_s_neighborhood_segment_p0.analysis import Dataset,peak_position
from DINOv3.MADEqual.aebad_s_neighborhood_segment_p0.prediction import original_distances
from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic.numerics import LAYERS

ROOT=Path(__file__).resolve().parents[3]
NEIGHBOR_MANIFEST_SHA='9af84042a3950eacee9abfd3a1be8a26716121a2b15ee5eec7e864514d6c3b92'


def defaults():
    p=old.defaults(ROOT)
    p.pop('ch1_dir')
    p['history_dir']=ROOT/'DINOv3/outputs/mad_equal/aebad_s_neighborhood_segment_local_20261016_134948_cpu'
    return p


def context(paths):
    paths={k:Path(v).resolve() for k,v in paths.items()}
    for k,h in (('raw_archive',ARCHIVE_SHA),('review_archive',REVIEW_SHA)):
        if sha(paths[k])!=h:raise ValueError('wrong/modified frozen '+k)
    with Archive(paths['raw_archive']) as a:
        capture=a.json(CROOT+'/inputs.json');cs=a.json(CROOT+'/status.json')
        segment=a.json(SROOT+'/inputs.json');ps=a.json(SROOT+'/predictions.json');ss=a.json(SROOT+'/status.json')
        if (capture['identity']!=CAP_ID or digest(capture['frozen'])!=CAP_ID
                or segment['identity']!=SEG_ID or digest(segment['frozen'])!=SEG_ID
                or cs['status']!='capture_verified' or cs['failed_checks']
                or ss['status']!='analysis_complete' or ss['numerical_failed_checks']
                or not cs['protected_inputs_unchanged'] or not ss['protected_inputs_unchanged']
                or hashlib.sha256(a.bytes(CROOT+'/inputs.json')).hexdigest()!=cs['inputs_sha256']
                or segment['frozen']['capture_inputs_sha256']!=cs['inputs_sha256']
                or segment['samples']!=capture['samples'] or segment['frozen']['supports']!=capture['supports']):
            raise ValueError('unverified/mismatched frozen capture and SEGMENT')
        with Archive(paths['review_archive']) as r:
            for n in ('inputs.json','predictions.json','status.json','numerical_verification.csv'):
                if a.bytes(SROOT+'/'+n)!=r.bytes(SROOT+'/'+n):raise ValueError('review and raw cache differ: '+n)
            history={n:r.csv(SROOT+'/'+n+'.csv') for n in ('per_image_scores','defect_instances','peak_positions')}
        checks=a.csv(CROOT+'/capture_verification.csv')+a.csv(SROOT+'/numerical_verification.csv')
        if not checks or any(r['verified'].lower() not in ('true','1') for r in checks):raise ValueError('historical verification failed')
        index=a.members.copy()
    formal={}
    for n,h in capture['frozen']['formal_sha256'].items():
        if sha(paths['formal_dir']/safe(n))!=h:raise ValueError('modified formal input: '+n)
        formal[n]=h
    with np.load(paths['formal_dir']/'source_mad_state.npz',allow_pickle=False) as f:
        medians,scales=f['medians'].copy(),f['scales'].copy()
    if (medians.dtype!=np.float32 or scales.dtype!=np.float32 or medians.shape!=(4,) or scales.shape!=(4,)
            or not np.isfinite(medians).all() or not np.isfinite(scales).all() or np.any(scales<=0)):
        raise ValueError('invalid original Source patch MAD')
    samples=capture['samples'];supports=capture['supports']
    if (len(samples)!=297 or len({s['sample_id'] for s in samples})!=297
            or sum(s['label']==0 for s in samples)!=85 or sum(s['label']==1 for s in samples)!=212
            or len(supports)!=8 or [s['support_index'] for s in supports]!=list(range(1,9))
            or len({s['relative_path'] for s in supports})!=8
            or {s['relative_path'] for s in samples}&{s['relative_path'] for s in supports}
            or capture['distance_unit']!='euclidean_over_1280_once'):
        raise ValueError('scope/support order/unit differs')
    captures={r['sample_id']:r for r in cs['records']}
    segments={(r['sample_id'],r['layer']):r for r in ps['records']}
    if (len(cs['records'])!=297 or set(captures)!={s['sample_id'] for s in samples}
            or len(ps['records'])!=1188 or set(segments)!={(s['sample_id'],l) for s in samples for l in LAYERS}):
        raise ValueError('incomplete frozen manifests')
    lookup={(r['sample_id'],r['method']):r for r in read_csv((paths['formal_dir']/'sample_scores.csv').read_bytes()) if r['domain']=='background'}
    for s in samples:
        for method,key in (('MAD_EQUAL','source_score'),('TARGET_MATCHED_K8_MAD','union_score')):
            r=lookup[s['sample_id'],method]
            if r['relative_path']!=s['relative_path'] or int(r['label'])!=s['label'] or float(r['image_score'])!=s[key]:
                raise ValueError('formal image identity or score differs')
    support_rows=[r for r in read_csv((paths['formal_dir']/'support_manifest.csv').read_bytes())
                  if r['domain']=='background' and r['branch']=='target_matched']
    # Original manifest contains multiple shot prefixes; compare the frozen K8 rows.
    if support_rows and 'shot' in support_rows[0]:support_rows=[r for r in support_rows if int(r['shot'])==8]
    selected={(int(r['support_index']),r['relative_path']) for r in support_rows}
    if selected!={(s['support_index'],s['relative_path']) for s in supports}:raise ValueError('formal K8 support identity differs')
    history['scores']={(r['sample_id'],r['branch']):float(r['image_score']) for r in history['per_image_scores']}
    history['instances']={(r['sample_id'],int(r['instance_id']),r['branch']):r for r in history['defect_instances']}
    history['peaks']={(r['sample_id'],r['region'],int(r['instance_id'] or 0)):(int(r['output_y']),int(r['output_x']))
        for r in history['peak_positions'] if r['peak_of_branch']=='K8_UNION' and r['measured_branch']=='K8_UNION'
        and r.get('output_y') and r.get('output_x')}
    history['fixed10']={s['sample_id'] for s in capture['selected_cases'] if s['label']==0}
    originals=[r for r in history['defect_instances'] if r['branch']=='K8_UNION']
    history['small3']={(r['sample_id'],int(r['instance_id'])) for r in sorted(originals,
        key=lambda r:(int(r['area_original_pixels']),r['sample_id'],int(r['instance_id'])))[:3]}
    if len(history['fixed10'])!=10 or len(originals)!=388:raise ValueError('historical case/instance scope differs')
    optional={}
    hd=paths['history_dir']
    if hd.is_dir():
        if sha(hd/'analysis_manifest.json')!=NEIGHBOR_MANIFEST_SHA:raise ValueError('modified optional historical instance reference')
        m=old.json_file(hd/'analysis_manifest.json')
        for n in ('defect_instances.csv','peak_positions.csv','input_manifest.json'):
            if sha(hd/n)!=m['files'][n]:raise ValueError('modified optional reference: '+n)
            optional[n]=m['files'][n]
        rows=read_csv((hd/'defect_instances.csv').read_bytes())
        history['optional_instances']={(r['sample_id'],int(r['instance_id']),
             'K8_UNION' if r['branch']=='UNION_K8_CH1' else 'K8_SEGMENT'):r for r in rows if r['branch'] in ('UNION_K8_CH1','SEGMENT_K8_CH1')}
    return dict(paths=paths,capture=capture,captures=captures,segments=segments,raw_index=index,
        samples=samples,medians=medians,scales=scales,history=history,
        assets=dict(raw_archive=ARCHIVE_SHA,review_archive=REVIEW_SHA,formal=formal,optional_history=optional))


def verify_data(ctx):
    # Verify required GT bytes/headers and fixed illustrative RGB bytes, not all train data.
    paths=ctx['paths'];capture=ctx['capture']
    if paths['data'].is_file():
        with zipfile.ZipFile(paths['data']) as z:
            if len(z.namelist())!=len(set(z.namelist())):raise ValueError('duplicate dataset ZIP paths')
    needed=dict(capture['gt_sha256'])
    needed.update({s['relative_path']:capture['image_sha256'][s['relative_path']] for s in capture['selected_cases']})
    shapes={s['relative_path']:(s['image_width'],s['image_height']) for s in capture['samples']}
    shapes.update({s['relative_path'].replace('test/','ground_truth/',1):(s['image_width'],s['image_height'])
                   for s in capture['samples'] if s['label']})
    with Dataset(ctx) as data:
        for name,h in needed.items():
            img=data.image(name,h)
            if img.size!=shapes[name]:raise ValueError('frozen background raster geometry differs: '+name)
    return len(needed)


def protect(ctx,output):
    p=Path(output).resolve()
    if p.exists():raise FileExistsError('new output directory required; no overwrite/resume')
    if p.drive and p.drive.upper()!='G:':raise ValueError('local output and scratch must remain on G:')
    for v in ctx['paths'].values():
        if p==v or p in v.parents or v in p.parents:raise ValueError('output overlaps protected input')
    return p
