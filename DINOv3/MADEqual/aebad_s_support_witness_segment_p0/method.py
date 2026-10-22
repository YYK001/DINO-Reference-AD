"""Label-free, support-defined geometry and matched native reduction."""
import numpy as np

LAYERS = (7, 15, 23, 31)
U, S, W, C = BRANCHES = ('UNION_K8', 'SEGMENT_K8', 'WITNESS_SEGMENT_K8', 'MATCHED_UNIFORM_WITNESS')
BASES, NEW = BRANCHES[:2], BRANCHES[2:]
PROTOCOL = dict(tool='support_witness_segment_fixed_p0_v1', branches=list(BRANCHES),
    layers=list(LAYERS), dimension=1280, seed=42, supports=8,
    populations=dict(background=[297,85,212], illumination=[265,67,198], view=[364,84,280]),
    neighbors_per_other_image=5, alpha_interval=[.25,.75], maximum_rho=.5,
    minimum_third_images=1, candidate='original stable Top2 of per-support Top1; not global segment NN',
    geometry='FP64 direct differences; no epsilon in witness denominators',
    distance_unit='already_euclidean_over_1280_once', segment_atol=1e-7, segment_rtol=1e-6,
    score_atol=1e-5, witness='fixed support relation evaluated lazily; no materialized complete graph',
    uniform='per image/layer FP64 total native reduction matched; FP32 ULP audit',
    localization='original Source patch MAD; native four-layer FP32 mean then original FP32 bilinear',
    image_score='frozen CH1 primary; CURRENT supplement', nstrips=200, FPR_max=.3,
    GT_used=False, labels_used=False, parameter_search=False)


def edge_key(a, i, b, j):
    a,i,b,j=map(int,(a,i,b,j))
    if a==b or min(a,b)<1 or min(i,j)<0: raise ValueError('two distinct support IDs and valid feature rows required')
    return tuple(x for pair in sorted(((a,i),(b,j))) for x in pair)


def witness(banks, neighbors, edge, excluded=()):
    """Only banks, neighbor rows, endpoints and optional held-out IDs are inputs.

    Each third image records its finite candidate set and a representative point,
    plus all passing points. No query feature, coordinate, response, label or GT.
    """
    a,i,b,j=edge_key(*edge); excluded=set(map(int,excluded))
    if a in excluded or b in excluded: raise ValueError('held-out image used as endpoint')
    u=np.asarray(banks[a][i],np.float64);v=np.asarray(banks[b][j],np.float64)
    result=dict(edge=[a,i,b,j],excluded=sorted(excluded),accepted=False,sources=[],third_images=[])
    if not np.isfinite(u).all() or not np.isfinite(v).all():return dict(result,reason='nonfinite_endpoint')
    direction=v-u;c2=float(direction@direction)
    if c2==0:return dict(result,reason='zero_length_connection')
    if not np.isfinite(c2):return dict(result,reason='nonfinite_endpoint_geometry')
    for c in sorted(set(banks)-{a,b}-excluded):
        candidates=np.union1d(neighbors[a,c][i],neighbors[b,c][j]).astype(np.int64)
        points=[];counts={};passing=[]
        for idx in candidates:
            w=np.asarray(banks[c][idx],np.float64);point=dict(patch_index=int(idx),alpha=None,rho=None)
            if not np.isfinite(w).all():reason='nonfinite_witness'
            else:
                alpha=float((w-u)@direction/c2);point['alpha']=alpha if np.isfinite(alpha) else None
                if not np.isfinite(alpha): reason='nonfinite_alpha'
                elif not .25<=alpha<=.75:reason='outside_internal_interval'
                else:
                    base=float(min(np.linalg.norm(w-u),np.linalg.norm(w-v)))
                    if base==0:reason='zero_endpoint_denominator'
                    elif not np.isfinite(base):reason='nonfinite_denominator'
                    else:
                        rho=float(np.linalg.norm(w-(u+alpha*direction))/base);point['rho']=rho if np.isfinite(rho) else None
                        reason='accepted' if np.isfinite(rho) and rho<=.5 else 'nonfinite_rho' if not np.isfinite(rho) else 'too_far_from_segment'
            point['reason']=reason;counts[reason]=counts.get(reason,0)+1;points.append(point)
            if reason=='accepted':passing.append(point)
        # Full candidate IDs permit recomputation; rejected candidates do not
        # duplicate 1280-D vectors. Representative is diagnostic, not selection.
        representative=min(points,key=lambda p:(p['rho'] is None,p['rho'] if p['rho'] is not None else float('inf'),p['patch_index'])) if points else None
        result['third_images'].append(dict(support_index=c,candidates=candidates.tolist(),counts=counts,
            representative=representative,passing_points=passing,accepted=bool(passing)))
        if passing:result['sources'].append(c)
    result.update(accepted=bool(result['sources']),reason='accepted' if result['sources'] else 'no_qualified_third_image',
                  witness_source_count=len(result['sources']))
    return result


def reduction(union, segment, gate):
    """Same negative-reduction treatment, sums and rounding bound as GAP P0."""
    u,s=np.asarray(union),np.asarray(segment);g=np.asarray(gate)
    if u.dtype!=np.float32 or s.dtype!=np.float32 or u.shape!=s.shape or g.shape!=u.shape or g.dtype!=bool:
        raise ValueError('FP32 native distances and matching boolean gate required')
    if any(not np.isfinite(x).all() or (x<0).any() for x in (u,s)):raise ValueError('invalid native distance')
    signed=u.astype(np.float64)-s
    if np.any(s>u+1e-7+1e-6*np.abs(u)):raise ValueError('SEGMENT exceeds UNION; no clipping of significant error')
    delta=np.maximum(signed,0);total=float(delta.sum(dtype=np.float64));requested=float((g*delta).sum(dtype=np.float64))
    beta=requested/total if total else 0.
    main=(u.astype(np.float64)-g*delta).astype(np.float32)
    uniform=(u.astype(np.float64)-beta*delta).astype(np.float32)
    amounts=[float((u.astype(np.float64)-x).sum(dtype=np.float64)) for x in (main,uniform)]
    bound=float(np.spacing(u).astype(np.float64).sum())+1e-12
    if abs(amounts[0]-amounts[1])>bound:raise FloatingPointError('uniform total outside FP32 rounding bound')
    return {W:main,C:uniform},dict(patches=u.size,accepted_positions=int(g.sum()),
        effective_candidates=int((signed>1e-7+1e-6*np.abs(u)).sum()),
        effective_accepted_positions=int((g&(signed>1e-7+1e-6*np.abs(u))).sum()),
        tolerated_negative_reductions=int((signed<0).sum()),minimum_signed_reduction=float(signed.min(initial=0)),
        original_positive_reduction_total=total,requested_main_reduction_total=requested,beta=beta,
        main_actual_reduction_total=amounts[0],uniform_actual_reduction_total=amounts[1],
        uniform_matching_abs_error=abs(amounts[0]-amounts[1]),uniform_matching_rounding_bound=bound,
        matching_scope='all native patches of this image/layer only')
