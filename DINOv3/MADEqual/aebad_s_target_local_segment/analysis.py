"""Three-branch bookkeeping, using original maps, spatial regions and metrics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from DINOv3.MADEqual.aebad_s_union_support_loo.analysis import branch_maps
from DINOv3.MADEqual.aebad_s_union_diagnostics.spatial import components, instance_regions
from .geometry import BRANCHES, LAYERS, scatter
from .inputs import replay

RAW_FORMAT = 'bounded_target_reference_layer_v1'


def save_layer(path: Path, values: dict, positions: np.ndarray, metadata: dict) -> None:
    if path.exists() or path.with_suffix('.npz.tmp').exists():
        raise ValueError('refusing existing or partial new-distance cache')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.npz.tmp')
    with temporary.open('xb') as f:
        np.savez_compressed(f, metadata=np.asarray(json.dumps(metadata,sort_keys=True)), positions=positions,
            midpoint_distance=values['K8_MIDPOINT'], segment_distance=values['K8_SEGMENT'],
            alpha=values['alpha'], endpoint_length=values['endpoint_length'],
            endpoint_support=values['endpoint_support'], endpoint_index=values['endpoint_index'],
            squared_correction=values['squared_correction'])
    temporary.replace(path)


def load_layer(root: Path, record: dict, identity: str, capture_record: dict, sample: dict,
               layer: int, arrays: dict) -> dict:
    if record['file'] != f"raw_cache/{sample['sample_id']}/L{layer}.npz":
        raise ValueError('unsafe new raw path')
    path = root / record['file']
    if replay.sha256(path) != record['sha256']:
        raise ValueError('new distances changed after geometry; no re-approval')
    with np.load(path,allow_pickle=False) as f:
        meta = json.loads(str(f['metadata'].item()))
        values = {k:f[k].copy() for k in ('midpoint_distance','segment_distance','alpha',
                  'endpoint_length','endpoint_support','endpoint_index','positions','squared_correction')}
    if (meta['format'] != RAW_FORMAT or meta['identity'] != identity
            or meta['capture_raw_sha256'] != capture_record['sha256']
            or meta['sample_id'] != sample['sample_id'] or meta['relative_path'] != sample['relative_path']
            or meta['layer'] != layer or not np.array_equal(values['positions'],arrays['positions'])):
        raise ValueError('new distance provenance/mapping changed')
    count = len(arrays['positions'])
    for key in ('midpoint_distance','segment_distance'):
        v = values[key]
        if v.shape != (count,) or v.dtype != np.float32 or not np.isfinite(v).all() or (v<0).any():
            raise ValueError('invalid new layer distances')
    for key in ('alpha','endpoint_length'):
        if values[key].shape != (count,) or not np.isfinite(values[key]).all() or (values[key]<0).any():
            raise ValueError('invalid interpolation trace')
    if (values['alpha']>1).any() or values['squared_correction'].shape != (2,count):
        raise ValueError('invalid alpha/correction trace')
    return values


def compact_raws(root: Path, records: dict, identity: str, capture_record: dict, sample: dict,
                 arrays: dict) -> tuple[dict, list[dict]]:
    union = np.minimum(arrays['source_distance'],arrays['target_distance'].min(axis=0))
    layer_values = [load_layer(root, records[(sample['sample_id'],l)],identity,capture_record,sample,l,arrays)
                    for l in LAYERS]
    values = dict(K8_UNION=union, K8_MIDPOINT=np.stack([v['midpoint_distance'] for v in layer_values]),
                  K8_SEGMENT=np.stack([v['segment_distance'] for v in layer_values]))
    if not np.all(values['K8_SEGMENT'] <= values['K8_MIDPOINT']+1e-7+1e-6*union):
        raise ValueError('saved segment/midpoint monotonicity failed')
    if not np.all(values['K8_MIDPOINT'] <= union):
        raise ValueError('saved midpoint no longer retains union')
    return values, layer_values


def maps_from_compact(raws: dict, arrays: dict, meta: dict, backend) -> dict:
    physical = {name:scatter(value,arrays['positions'],tuple(meta['grid'])) for name,value in raws.items()}
    return branch_maps(physical,tuple(meta['output_shape']),backend)


def spatial(sample: dict, maps: dict, mask: np.ndarray | None, grid: tuple[int,int], backend) -> tuple[list,list]:
    rows, peaks = [], []
    regions = [('full','',np.ones(maps[BRANCHES[0]].shape,bool))]
    if sample['label']:
        if mask is None: raise ValueError('GT required for anomaly spatial analysis')
        labels, descriptors = components(mask)
        if not descriptors: raise ValueError('empty anomaly GT')
        for d in descriptors:
            component = labels == d['instance_id']
            inside, ring = instance_regions(component,mask,maps[BRANCHES[0]].shape)
            coverage = int(backend.patch_mask(Image.fromarray(component.astype(np.uint8)*255),grid).sum())
            maxima = {b:(float(m[inside].max()) if inside.any() else None,
                          float(m[ring].max()) if ring.any() else None) for b,m in maps.items()}
            bi,br = maxima[BRANCHES[0]]
            bm = bi-br if bi is not None and br is not None else None
            for branch,(im,rm) in maxima.items():
                margin = im-rm if im is not None and rm is not None else None
                rows.append(dict(sample_id=sample['sample_id'],relative_path=sample['relative_path'],**d,
                    branch=branch,patch_coverage=coverage,instance_area_output_pixels=int(inside.sum()),
                    ring_area_output_pixels=int(ring.sum()),ring_radius_original_pixels=16,
                    ring_metric='Chebyshev',ring_excludes_all_GT=True,instance_max=im,ring_max=rm,margin=margin,
                    instance_max_change_vs_K8=im-bi if im is not None and bi is not None else None,
                    ring_max_change_vs_K8=rm-br if rm is not None and br is not None else None,
                    margin_change_vs_K8=margin-bm if margin is not None and bm is not None else None,
                    response_status='mapped_instance_disappeared' if not inside.any() else
                                    ('empty_ring' if not ring.any() else 'available'),
                    annotation_semantics='unreviewed_no_size_threshold'))
            regions.extend([('instance',d['instance_id'],inside),('ring',d['instance_id'],ring)])
    for role,instance_id,region in regions:
        if not region.any(): continue
        for origin in BRANCHES:
            flat = int(np.argmax(np.where(region,maps[origin],-np.inf)))
            y,x = divmod(flat,region.shape[1])
            for measured in BRANCHES:
                peaks.append(dict(sample_id=sample['sample_id'],region=role,instance_id=instance_id,
                    peak_of_branch=origin,measured_branch=measured,output_y=y,output_x=x,
                    score=float(maps[measured][y,x]),change_vs_K8=float(maps[measured][y,x]-maps[BRANCHES[0]][y,x]),
                    fixed_K8_reference=origin==BRANCHES[0],peak_tie_rule='first_row_major_maximum'))
    return rows,peaks


def ranking(per_image: list[dict], fixed_top10: list[str]) -> tuple[list,list]:
    from sklearn.metrics import roc_auc_score,average_precision_score
    groups = {b:{r['sample_id']:r for r in per_image if r['branch']==b} for b in BRANCHES}
    ids = sorted(groups[BRANCHES[0]])
    if any(set(g)!=set(ids) or len(g)!=len(ids) for g in groups.values()):
        raise ValueError('three branches must have the identical sample set')
    labels = np.array([groups[BRANCHES[0]][i]['label'] for i in ids])
    normal,anomaly = labels==0,labels==1
    def errors(scores):
        n,a = scores[normal,None],scores[None,anomaly]
        return (n>a).astype(float)+.5*(n==a)
    baseline = errors(np.array([groups[BRANCHES[0]][i]['image_score'] for i in ids]))
    summaries,detail = [],[]
    for branch in BRANCHES:
        scores = np.array([groups[branch][i]['image_score'] for i in ids])
        err = errors(scores)
        improve,worse = err<baseline,err>baseline
        for idx,i in enumerate(ids):
            subset = err[list(np.flatnonzero(normal)).index(idx)] if normal[idx] else err[:,list(np.flatnonzero(anomaly)).index(idx)]
            before = baseline[list(np.flatnonzero(normal)).index(idx)] if normal[idx] else baseline[:,list(np.flatnonzero(anomaly)).index(idx)]
            detail.append(dict(sample_id=i,branch=branch,weighted_pair_errors=float(subset.sum()),
                improved_pairs_vs_K8=int((subset<before).sum()),worsened_pairs_vs_K8=int((subset>before).sum()),
                unchanged_pairs_vs_K8=int((subset==before).sum()),fixed_K8_top10=i in fixed_top10))
        summaries.append(dict(branch=branch,image_count=len(ids),normal_count=int(normal.sum()),anomaly_count=int(anomaly.sum()),
            image_AUROC=float(roc_auc_score(labels,scores)),image_AP=float(average_precision_score(labels,scores)),
            weighted_pair_errors=float(err.sum()),pair_count=int(err.size),pair_ties=int((err==.5).sum()),
            improved_pairs_vs_K8=int(improve.sum()),worsened_pairs_vs_K8=int(worse.sum()),
            unchanged_pairs_vs_K8=int((err==baseline).sum()),tie_error_weight=.5,
            normal_score_p90=float(np.percentile(scores[normal],90)),normal_score_p95=float(np.percentile(scores[normal],95)),
            normal_score_max=float(scores[normal].max()),
            fixed_top10_mean_score_change=float(np.mean([groups[branch][i]['score_change_vs_K8'] for i in fixed_top10]))))
    return summaries,detail


def mechanism(sample: dict, raws: dict, scales: np.ndarray, positions: np.ndarray,
              patch_mask: np.ndarray | None) -> list[dict]:
    rows = []
    selected = np.ones(len(positions),bool) if not sample['label'] else (
        patch_mask.reshape(-1)[positions] if patch_mask is not None else np.zeros(len(positions),bool))
    for b in BRANCHES:
        delta = (raws[BRANCHES[0]].astype(np.float64)-raws[b])/(scales[:,None]+1e-8)
        for offset,layer in enumerate(LAYERS):
            rows.append(dict(sample_id=sample['sample_id'],label=sample['label'],branch=b,layer=layer,
                reference='K8_UNION',R_value=float(delta[offset,selected].mean()) if selected.any() else None,
                valid_region='normal_all_patches' if not sample['label'] else
                             ('gt_defect_patches' if selected.any() else 'empty_defect_patch_mask'),
                selected_patch_count=int(selected.sum()),aggregation='per_image_mean'))
    return rows


def mechanism_summary(rows: list[dict]) -> list[dict]:
    result = []
    for branch in BRANCHES:
        layers = []
        for layer in LAYERS:
            group = [r for r in rows if r['branch']==branch and r['layer']==layer]
            normal = [r['R_value'] for r in group if not r['label'] and r['R_value'] is not None]
            anomaly = [r['R_value'] for r in group if r['label'] and r['R_value'] is not None]
            rn,ra = (float(np.mean(v)) if v else None for v in (normal,anomaly))
            row = dict(branch=branch,layer=layer,reference='K8_UNION',R_N=rn,R_A=ra,
                G=rn-ra if rn is not None and ra is not None else None,
                normal_image_count=len(normal),anomaly_image_count=len(anomaly),aggregation='image_equal')
            result.append(row);layers.append(row)
        result.append(dict(branch=branch,layer='four_layer_mean',reference='K8_UNION',
            **{k:float(np.mean([r[k] for r in layers])) if all(r[k] is not None for r in layers) else None for k in ('R_N','R_A','G')},
            normal_image_count=min(r['normal_image_count'] for r in layers),
            anomaly_image_count=min(r['anomaly_image_count'] for r in layers),aggregation='equal_four_layer_image_equal'))
    return result


def interpolation_rows(sample: dict, values: list[dict], raws: dict,
                       positions: np.ndarray, patch_mask: np.ndarray | None) -> list[dict]:
    """GT grouping is called ONLY after every image's geometry is frozen."""
    rows=[]
    masks=[('all_prediction_patches',np.ones(len(positions),bool))]
    if sample['label']:
        if patch_mask is not None: masks.append(('gt_defect_patches',patch_mask.reshape(-1)[positions]))
    else: masks.append(('normal_all_patches',np.ones(len(positions),bool)))
    for offset,(layer,v) in enumerate(zip(LAYERS,values,strict=True)):
        for region,mask in masks:
            if not mask.any():continue
            alpha=v['alpha'][mask];length=v['endpoint_length'][mask]
            count=int(mask.sum())
            histogram,_=np.histogram(alpha,bins=np.linspace(0,1,11))
            pairs=v['endpoint_support'][:,mask].T
            unique,counts=np.unique(pairs,axis=0,return_counts=True)
            for pair,n in zip(unique,counts,strict=True):
                rows.append(dict(sample_id=sample['sample_id'],label=sample['label'],layer=layer,region=region,
                    statistic='ordered_endpoint_pair',support_1=int(pair[0]),support_2=int(pair[1]),
                    count=int(n),patch_count=count,fraction=float(n/count)))
            row=dict(sample_id=sample['sample_id'],label=sample['label'],layer=layer,region=region,
                statistic='interpolation_distribution',patch_count=count,alpha_min=float(alpha.min()),
                alpha_max=float(alpha.max()),alpha_mean=float(alpha.mean()),
                alpha_p10=float(np.percentile(alpha,10)),alpha_p50=float(np.percentile(alpha,50)),
                alpha_p90=float(np.percentile(alpha,90)),endpoint_degenerate_count=int((length==0).sum()),
                correction_count=int((v['squared_correction'][:,mask]>0).sum()),
                maximum_squared_correction=float(v['squared_correction'][:,mask].max()),
                midpoint_lower_fraction=float((raws[BRANCHES[1]][offset,mask]<raws[BRANCHES[0]][offset,mask]).mean()),
                segment_lower_fraction=float((raws[BRANCHES[2]][offset,mask]<raws[BRANCHES[0]][offset,mask]).mean()))
            row.update({f'alpha_bin_{i}_count':int(n) for i,n in enumerate(histogram)})
            rows.append(row)
    return rows
