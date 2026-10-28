"""One fixed experiment; execution chunks are not method parameters."""
VERSION = 'same_origin_neighbor_background_v1'
LAYERS = (7, 15, 23, 31)
U, S = 'UNION_K8', 'SEGMENT_K8'
COMBINED = 'SEGMENT_IMAGE_UNION_MAP'
A, B, C = 'ABS_NEIGHBOR_C1', 'REL_NEIGHBOR_C1', 'REL_NEIGHBOR_C16'
NEW = (A, B, C)
MAPS = (U, S, *NEW)
BRANCHES = (U, S, COMBINED, *NEW)
STAGES = ('prepare-source', 'prepare-target', 'calibrate', 'predict', 'evaluate', 'aupro', 'summarize')
CONFIG = dict(version=VERSION, domain='background', seed=42, layers=list(LAYERS), dimension=1280,
    Source_images=521, Source_construction=455, Source_calibration=66, Source_rows=100000,
    Target_supports=8, evaluation_images=297, normal_images=85, anomaly_images=212,
    C=16, weights=[.5,.5], directions=['up','down','left','right'],
    distance='Euclidean/1280_once', precision='FP32_exact_no_feature_normalization',
    boundary='query outer ring OR absolute Top1 has incomplete neighbors: all three use UNION z; C16 filters original Top16 without refill',
    ties='distance_then_global_bank_row; Source_first_then_manifest_support_order_then_physical_patch',
    component_MAD='Source_only_absolute_Top1; abs all patches; nb/rel only complete four-neighbor pairs',
    image_MAD='separate per new branch on original Source-only 66 full images; CH1 primary CURRENT supplemental',
    fusion='native_float32_equal4_then_original_float32_bilinear', nstrips=200, FPR_max=.3,
    training=False, parameter_search=False, duration_budget_hours=12)


def job_plan(reuse_baselines=True):
    return dict(domain='background', evaluation_images=297, calibration_images=66,
        new_methods=list(NEW), map_jobs=3 if reuse_baselines else 5,
        AUPRO_jobs=3 if reuse_baselines else 5, image_score_rules=['CH1','CURRENT'],
        combined_map_jobs=0, training_jobs=0, seeds=[42], full_Source_rebuild='explicit_only')
