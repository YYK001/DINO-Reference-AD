"""Optional original few-shot distances; cumulative K8 is a Union, not Target Top1."""
from pathlib import Path
import numpy as np
from . import inputs as ip
from .method import scatter
from DINOv3.MADEqual.aebad_s_projection_layer_diagnostic.numerics import check


def manifest(ctx,directory):
    root=Path(directory);m=ip.read_json(root/'manifest.json')
    if (m['format']!='formal_fewshot_distance_reference_v1' or m['status']!='verified'
            or m['protocol_id']!='mad_equal_aebad_s_target_fewshot_seed42_v1'
            or m['cache_revision']!='fewshot_raw_v1' or m['memory_sha256']!=ctx['catalog']['memory_sha256']
            or m['sample_scores_sha256']!=ctx['catalog']['formal_sha256']['sample_scores.csv']
            or m['query']!=ip.QUERY or m['precision']!=ip.PRECISION):
        raise ValueError('original distance provenance incompatible')
    seen=set()
    for r in m['records']:
        key=(r['domain'],r['sample_id'])
        if key in seen or r['domain'] not in ip.DOMAINS:raise ValueError('reference scope/duplicates')
        seen.add(key)
        s=next(s for s in ctx['catalog']['domains'][r['domain']]['samples'] if s['sample_id']==r['sample_id'])
        if (r['relative_path']!=s['relative_path'] or r['image_sha256']!=s['image_sha256']
                or ip.sha(root/ip.safe(r['file']))!=r['sha256']):raise ValueError('original distance identity/hash mismatch')
    return m


def verify(ctx,sample,source,union,positions,meta,checks):
    m=ctx.get('distance_references')
    if m is None:return False
    r=next((r for r in m['records'] if (r['domain'],r['sample_id'])==(sample['domain'],sample['sample_id'])),None)
    if r is None:return False
    path=ctx['paths']['distance_reference_root']/ip.safe(r['file'])
    if ip.sha(path)!=r['sha256']:raise ValueError('original reference changed since enrollment')
    with np.load(path,allow_pickle=False) as f:
        if (str(f['protocol_id'].item())!=m['protocol_id'] or str(f['cache_revision'].item())!=m['cache_revision']
                or str(f['relative_path'].item())!=sample['relative_path']
                or list(f['output_shape'])!=meta['output_shape']):raise ValueError('original distance header mismatch')
        for name,actual,field in (('original_Source_distance',source,'official_raw'),('original_Union_K8_distance',union,'target_raw_8')):
            expected=f[field]
            if expected.dtype!=np.float32:raise ValueError('original distance dtype changed')
            check(checks,sample['sample_id'],name,scatter(actual,positions,tuple(meta['grid'])),expected,
                  atol=1e-7,rtol=1e-6,domain=sample['domain'],reference_sha256=r['sha256'])
    if not all(r['verified'] for r in checks[-2:]):raise ValueError('original raw-distance reproduction failed; no tolerance relaxation')
    return True
