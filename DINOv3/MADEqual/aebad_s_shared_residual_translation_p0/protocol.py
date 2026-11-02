DOMAINS = ('background', 'illumination', 'view')
LAYERS = (7, 15, 23, 31)
COUNTS = {'background': (289, 77, 212), 'illumination': (257, 59, 198), 'view': (356, 76, 280)}
U, S, SOURCE = 'UNION_K8', 'SEGMENT_K8', 'UNION_SOURCE_WEIGHT'
A, B, C = 'MEAN_U_FIXED', 'MEAN_U_RESELECT', 'MEAN_REF_RESELECT'
MAPS = (A, B, C)
READOUTS = ('CH1', 'WEIGHTED_MAP_MAX', 'SEGMENT_CH1')
METRICS = ('image_AUROC', 'image_AP', 'pixel_AUROC', 'pixel_AP', 'AUPRO')
CONFIG = dict(version='shared_residual_translation_v1', primary=C, domains=list(DOMAINS),
              support_seed=42, Source_seed=42, K=8, Source_train=455, Source_calibration=66,
              Source_rows=100000, dimension=1280, layers=list(LAYERS), images=902,
              mean='all_unique_physical_positions_equal_weight_including_scored_position',
              anchors='one_original_Exact_FP32_Top1_per_source_then_fixed_nine_candidates',
              translation_strength=1, translation_units='original_feature_units', distance_divisor=1280,
              precision='FP32_vectors_residuals_means_distances_original_Exact',
              calibration='frozen_Source_patch_and_image_MAD_no_fit',
              localization='frozen_Source_u_weights_native_FP32_then_original_interpolation',
              readouts=list(READOUTS), new_AUPRO_jobs=9, nstrips=200, FPR_max=.3,
              training=False, parameter_search=False, duration_budget_hours=12,
              GT='evaluation_only_after_all_predictions_frozen', blind_spot=False)


def job_plan():
    return dict(primary=C, images=902, settings=3, support_images=24, support_seed=42,
                localization_jobs=9, AUPRO_jobs=9, image_readouts_per_map=3,
                maximum_query_forwards=902, exact_searches_per_missing_image=36,
                Source_rebuilds=0, calibration_refits=0, training_jobs=0,
                controls_not_requested=['seed43', 'Target_LOO', 'shifted_full_bank_retrieval'])
