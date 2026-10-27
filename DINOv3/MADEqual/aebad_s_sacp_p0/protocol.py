"""One fixed configuration; labels/metrics never select parameters."""
VERSION='sacp_source_control_v1'
LAYERS=(7,15,23,31)
DOMAINS=('background','illumination','view')
U,S='UNION_K8','SEGMENT_K8'
MAIN,TARGET,SOURCE,UNIFORM='SACP_CONTRAST','SACP_TARGET_DIR','SACP_SOURCE_DIR','SACP_MATCHED_UNIFORM'
NEW=(MAIN,TARGET,SOURCE,UNIFORM)
BRANCHES=(U,S,*NEW)
COMBINED='SEGMENT_SCORE_UNION_MAP'
CONFIG=dict(version=VERSION,layers=list(LAYERS),dimension=1280,Source_rows=100000,seed=42,
    clusters=32,cluster_bank_cap=16384,kmeans_initializations=1,kmeans_iterations=30,
    kmeans_initialization='PCG64_unique_row_indices',empty_cluster='retain_previous_center',
    fit_patch_cap=4096,per_image_group_cap=64,min_images_per_class_fold=4,
    Source_controls_per_fold=8,common_rank=32,directions=2,shrinkage=.1,strength=.5,
    svd_oversampling=8,svd_power_iterations=2,rank_rtol=1e-6,
    eta_roundoff_tolerance=1e-5,distance_atol=1e-7,distance_rtol=1e-6,score_atol=1e-5,
    populations=dict(background=[297,85,212],illumination=[265,67,198],view=[364,84,280]),
    distance_unit='FP32_parent_Exact_Euclidean_over_1280_once',anchor_ties='Source_then_Target_manifest_order',
    grouping='formal_Source_centers; query_group_from_formal_Source_nearest_row',
    statistics='uncentered; equal_classes_then_equal_folds_then_equal_images_then_equal_rows',
    uniform_arithmetic='FP64 total/Delta; FP32(1-a) times original FP32 UNION; absolute sum-roundoff <= eps32*sum_UNION',
    nstrips=200,FPR_max=.3,scoring='frozen_Source_CH1; CURRENT_supplement',
    TF32=False,training=False,full_payload_hash_scan=False)


def job_plan():
    return dict(version=VERSION,fit_jobs=[dict(domain=d,layer=l) for d in DOMAINS for l in LAYERS],
        clustering_jobs=list(LAYERS),prediction_images=926,branches=list(BRANCHES),primary=MAIN,
        Source_control_images=16,Target_support_images=24,training_jobs=0,
        localization_jobs=[dict(domain=d,branch=b) for d in DOMAINS for b in BRANCHES],
        maximum_new_AUPRO_jobs_with_compatible_baselines=12,logical_AUPRO_jobs=18,
        expected_metric_rows=56,combined=COMBINED,parameter_search=False)
