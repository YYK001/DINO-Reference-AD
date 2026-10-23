"""Predeclared search and branch semantics, independent of test outcomes."""
from itertools import product

REVISION = 'aebad_fixed_target_graph_v1'
LAYERS = (7, 15, 23, 31)
DOMAINS = ('background', 'illumination', 'view')
KS, QUANTILES, ETAS = (10, 30, 100), (.50, .75, .90), (.25, .50, .75, 1.)
U, S, G0, W, GL, ND, ID = BRANCHES = (
    'UNION_K8', 'SEGMENT_K8', 'LOCAL_GRAPH_PAIRED', 'WITNESS_GRAPH',
    'LOCAL_GRAPH_LOO', 'NORMAL_MATCHED_DAMP', 'IMAGE_MATCHED_DAMP')
CONFIGS = tuple((k, q, eta) for k, q, eta in product(KS, QUANTILES, ETAS)) + tuple(
    (k, q, None) for k, q in product(KS, QUANTILES))
PROTOCOL = dict(revision=REVISION, branches=BRANCHES, layers=LAYERS, dimension=1280,
    k=KS, length_quantiles=QUANTILES, eta_thresholds=ETAS, alpha_interval=(.2,.8),
    graph='global_other_image_topk_mutual; canonical original support,row tie',
    witness='exhaustive third-image points; h / median cross-image nearest distances',
    selection='Source+Target7+graph; top-two CH1 mean; 95% best gain then complexity',
    selection_score_atol=1e-5, selection_score_rtol=0.,
    distance_unit='Euclidean / 1280 once', geometry='FP64 bounded GEMM + direct-vector refinement',
    localization='Source patch MAD; native FP32 equal4 then original FP32 bilinear',
    image_scoring=('CH1','CURRENT'), nstrips=200, FPR_max=.3,
    capacity_admission_checks=False, default_seed=42)


def config_id(c):
    k, q, eta = c
    return f'k{k}_q{q:.2f}_' + ('plain' if eta is None else f'eta{eta:.2f}')


def validate_configs(configs):
    for c in configs:
        if tuple(c) not in CONFIGS: raise ValueError('configuration outside predeclared search')
