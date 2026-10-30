from DINOv3.MADEqual.aebad_s_target_loo_support_retest_p0.protocol import DOMAINS,SEEDS,LAYERS,U,SOURCE,TARGET,READOUTS
WEIGHTS=('WS','W42','W43')
COMPARISONS=(('W42','WS'),('W43','WS'),('W43','W42'))
COUNTS={'background':(289,77,212),'illumination':(257,59,198),'view':(356,76,280)}
MARGIN_ATOL=1e-6
SCORE_ATOL=1e-5
CONFIG=dict(version='loo_weight_cross_separability_audit_v1',domains=list(DOMAINS),memory_seeds=list(SEEDS),weight_sources=list(WEIGHTS),Source_seed=42,
    weights='frozen_entire_u_p_q_matrix_no_refit_or_renormalization',response='original_standardized_UNION_no_second_MAD',
    weighting_kernel='aebad_s_target_loo_layer_weight_p0.numerics.weighted',
    primary='PEAK_LOCAL_WEIGHT',supplement='WEIGHTED_MAP_MAX',new_maps=6,new_AUPRO_jobs=6,
    diagnostic_only=True,cross_information='K8_memory_plus_other_K8_normal_statistics_16_distinct_normals_when_disjoint',
    image_ties='exact_frozen_scores_no_epsilon',normal_top_fraction=.01,top_ties='descending_value_then_ascending_flat_pixel_index',
    instance_connectivity=8,ring_radius_original_pixels=16,ring_metric='Chebyshev_excludes_all_GT',margin_atol=MARGIN_ATOL,score_atol=SCORE_ATOL,
    map_precision='original_FP32',decomposition_precision='small_FP64_with_effective_FP32_weights_roundoff_reported',
    GT='evaluation_only_after_all_responses_frozen',nstrips=200,FPR_max=.3,training=False,feature_forwards=0,Exact_searches=0,weight_fit_calls=0,parameter_search=False,duration_budget_hours=12)

def crossed(memory):return 'W43' if memory==42 else 'W42'
def diagonal(memory):return f'W{memory}'
def branch(memory,weight):return f'Z{memory}_X_{weight}'
