DOMAINS=('background','illumination','view')
SEEDS=(42,43)
LAYERS=(7,15,23,31)
COUNTS={'background':(289,77,212),'illumination':(257,59,198),'view':(356,76,280)}
BASE='SEGMENT_CH1'
SOURCE='SOURCE_CONDITIONAL'
TARGET='TARGET_CONDITIONAL'
BRANCHES=(BASE,SOURCE,TARGET)
MAP='UNION_SOURCE_WEIGHT'
CONFIG=dict(version='conditional_SEGMENT_image_score_v1',domains=list(DOMAINS),support_seeds=list(SEEDS),
    Source_seed=42,Source_training=455,Source_calibration=66,Source_rows=100000,K=8,layers=list(LAYERS),dimension=1280,
    B='mean_of_four_native_SEGMENT_z_medians_FP64',P='original_SEGMENT_CH1_Source_patch_image_MAD',
    fit='nonnegative_Source_OLS_and_Target_centered_ridge_to_Source',ddof=0,ridge=1,
    degenerate='sigma_BS_le_64eps64_max_1_abs_BS_returns_original_P',primary=TARGET,
    map_reference=MAP,pixel_metrics='read_only_parent_inheritance',new_pixel_jobs=0,new_AUPRO_jobs=0,
    target_proxy='whole_image_K7_LOO_for_formal_K8_no_distribution_equivalence_claim',
    parameter_search=False,training=False,Source_rebuild=False,duration_budget_hours=12)


def plan():
    return dict(settings=6,unique_evaluation_images=902,image_support_instances=1804,
        Source_statistics=396,Target_LOO_statistics=48,unique_Source_calibration_images=66,
        maximum_unique_support_images=48,normal_statistics=444,image_branches=3,image_evaluation_cells=18,
        new_pixel_jobs=0,new_AUPRO_jobs=0,Source_rebuilds=0,training_jobs=0,
        independent_data_repetitions=False)
