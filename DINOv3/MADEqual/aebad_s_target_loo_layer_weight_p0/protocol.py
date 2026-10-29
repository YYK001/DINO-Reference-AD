LAYERS=(7,15,23,31)
U='UNION_K8'
S='SEGMENT_K8'
from DINOv3.MADEqual.aebad_s_blind_context_p0.protocol import COMBINED
SOURCE='UNION_SOURCE_WEIGHT'
TARGET='UNION_TARGET_LOO_WEIGHT'
NEW=(SOURCE,TARGET)
READOUTS=('PEAK_LOCAL_WEIGHT','WEIGHTED_MAP_MAX')
COUNTS={'background':(297,85,212),'illumination':(265,67,198),'view':(364,84,280)}
SOURCE_COMBO='SEGMENT_SCORE_SOURCE_WEIGHT_MAP'
TARGET_COMBO='SEGMENT_SCORE_TARGET_LOO_MAP'
COMBINATIONS={SOURCE_COMBO:SOURCE,TARGET_COMBO:TARGET}
STAGES=('prepare-normal-stats','fit-weights','predict','evaluate','aupro','summarize')
CONFIG=dict(version='target_loo_positive_tail_v1',domain='background',seed=42,layers=list(LAYERS),
    gammas=[0.,.5],quantile=.99,quantile_method='linear',scale_floor=1e-8,statistics=['u','p','q'],
    Source_calibration=66,Target_supports=8,evaluation_images=297,normal_images=85,anomaly_images=212,
    distance='original_Exact_Euclidean/1280_once',patch_MAD='frozen_Source',
    Target_reference='Source_plus_K7_exclude_entire_heldout_image',nstrips=200,FPR_max=.3,
    training=False,parameter_search=False,duration_budget_hours=12)


def config(domain):
    if domain=='background':return dict(CONFIG)  # Legacy run identities remain intact.
    n,normal,anomaly=COUNTS[domain]
    return dict(CONFIG,domain=domain,evaluation_images=n,normal_images=normal,anomaly_images=anomaly,
        extension_version='fixed_crossdomain_task_division_v1',
        combinations={b:dict(image_reference=S,map_reference=m,scorings=['CH1','CURRENT']) for b,m in COMBINATIONS.items()})
