"""One fixed experiment. Tolerances are frozen before any real scoring."""
DOMAINS=('background','illumination','view')
SEEDS=(42,43)
LAYERS=(7,15,23,31)
COUNTS=dict(background=289,illumination=257,view=356)
BASE='SEGMENT_CH1'
SHARED='FREE_SHARED_SCALE'
OWN='FREE_OWN_SCALE'
PRIMARY='CAPACITY_FUSION'
CAP='CAPACITY_ONLY'
FREE='FREE_ONLY'
BRANCHES=(BASE,SHARED,OWN,PRIMARY,CAP,FREE)
CONFIG=dict(version='reverse_capacity_coverage_v1',domains=list(DOMAINS),support_seeds=list(SEEDS),
    Source_seed=42,Source_training=455,Source_calibration=66,layers=list(LAYERS),dimension=1280,
    regions=256,adjacency='physical_grid_4_neighbor',merge='minimum_current_mean_descriptor_distance',
    ties='distance_then_minimum_physical_index_pair',mean_arithmetic='FP32_weighted_mean',
    distance='mean_four_Euclidean_RMS_scaled_layers_div_sqrt1280_direct_FP32',
    RMS='455_unaugmented_complete_original_grids_per_channel_RMS_FP64_sum_no_centering',
    objective='min_W_t_max_template_region_average_cost',solver='scipy_linprog_highs_ds_sparse_W',
    solver_primal_tolerance=1e-10,solver_dual_tolerance=1e-9,
    row_relative_tolerance=1e-7,capacity_relative_tolerance=1e-6,
    objective_tolerance=1e-7,negative_W_tolerance=1e-10,
    distance_backend_check_atol=1e-5,
    mass_sum_tolerance=1e-12,scale_threshold='64eps64_times_max_1_maxabs',ddof=0,
    primary=PRIMARY,branches=list(BRANCHES),fusion='fixed_equal_sum',
    template_selection='min_of_8_whole_templates_separately_per_branch',
    map_reference='UNION_SOURCE_WEIGHT',new_pixel_jobs=0,new_AUPRO_jobs=0,
    Source_rebuild=False,training=False,parameter_search=False,duration_budget_hours=12)


def plan():
    return dict(settings=6,unique_evaluation_images=902,image_support_instances=1804,
        Source_calibration_instances=396,unique_calibration_images=66,support_images=48,
        region_images=1016,Source_RMS_images=455,query_template_pairs=14432,
        calibration_template_pairs=3168,LP_jobs=17600,regions=256,branches=list(BRANCHES),
        new_pixel_jobs=0,new_AUPRO_jobs=0,training_jobs=0,Source_rebuilds=0,
        independent_data_repetitions=False)
