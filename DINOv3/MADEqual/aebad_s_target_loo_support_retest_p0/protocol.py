from DINOv3.MADEqual.aebad_s_target_loo_layer_weight_p0.protocol import LAYERS,NEW,U,S,SOURCE,TARGET,COMBINED,COMBINATIONS,READOUTS
DOMAINS=('background','illumination','view')
SEEDS=(42,43)
MAPS=(U,S,*NEW)
STAGES=('prepare-normal-stats','fit-weights','predict','evaluate','aupro','summarize')
CONFIG=dict(version='target_loo_two_support_sets_v1',domains=list(DOMAINS),support_seeds=list(SEEDS),Source_seed=42,K=8,
    layers=list(LAYERS),dimension=1280,Source_rows=100000,Source_calibration=66,
    evaluation='full_test_minus_union_T42_T43_per_domain',gamma=[0.,.5],quantile=.99,quantile_method='linear',scale_floor=1e-8,
    weighting_kernel='aebad_s_target_loo_layer_weight_p0.numerics',normal_tail='positive_part_then_square_Source_median_anchor',
    precision='original_FP32_Exact_and_maps_small_CPU_statistics_FP64',nstrips=200,FPR_max=.3,
    combinations={b:dict(image_reference=S,map_reference=m,scorings=['CH1','CURRENT']) for b,m in COMBINATIONS.items()},
    training=False,parameter_search=False,duration_budget_hours=12)
