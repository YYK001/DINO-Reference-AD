from DINOv3.MADEqual.aebad_s_target_loo_support_retest_p0.protocol import DOMAINS, SEEDS, LAYERS, U, S, SOURCE, COMBINED
from DINOv3.MADEqual.aebad_s_target_loo_layer_weight_p0.composition import METRICS

COUNTS = {'background': (289, 77, 212), 'illumination': (257, 59, 198), 'view': (356, 76, 280)}
JOINT = 'JOINT_ST8_CF7'
INDEPENDENT = 'INDEPENDENT_ST8_CF7'
UNION4 = 'JOINT_U4_CF7'
CAPACITY = 'UNION_CF7_SOURCE_WEIGHT'
MAPS = (JOINT, INDEPENDENT, UNION4, CAPACITY)
READOUTS = ('SEGMENT_CH1', 'MAP_MAX')
COMPARISONS = (UNION4, INDEPENDENT, CAPACITY, SOURCE)
CONFIG = dict(version='joint_reference_response_v1', domains=list(DOMAINS), support_seeds=list(SEEDS),
              Source_seed=42, layers=list(LAYERS), Source_rows=100000, K=8, reference_K=7,
              normal_positions_per_image=512, normal_sampling='sha256_path_physical_coordinate_rank_v1',
              bandwidth=1.0, source_target_scale='same_original_Source_patch_MAD_no_second_fit',
              kernel='sigmoid((normal_response-query_response)/h)', aggregation='equal_support_probability_mean_before_negative_log',
              divisors={JOINT: 8, INDEPENDENT: 8, UNION4: 4}, maps=list(MAPS),
              readouts=list(READOUTS), new_map_jobs=24, AUPRO_jobs=24, nstrips=200, FPR_max=.3,
              response_precision='FP32_logsigmoid_logsumexp', training=False, parameter_search=False,
              GT='only_after_global_prediction_freeze', duration_budget_hours=12)


def heldout_indices(index):
    if index not in range(8):
        raise ValueError('group-local support index 0..7 required')
    return [j for j in range(8) if j != index]


def work_plan():
    return dict(unique_evaluation_images=902, repeated_query_settings=1804, settings=6,
                normal_support_images=48, normal_references=48, new_map_jobs=24,
                new_AUPRO_jobs=24, source_rebuilds=0, Source_calibration_forwards=0,
                maximum_new_query_forwards=902, training_jobs=0, parameter_search=False)
