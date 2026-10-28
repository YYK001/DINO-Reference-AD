"""One extension, two inherited eta axes, three new maps; no parameter search."""
from copy import deepcopy
from DINOv3.MADEqual.aebad_s_sacp_p0.protocol import CONFIG as PARENT_CONFIG,DOMAINS,LAYERS,U,S,MAIN as HALF_MAIN,TARGET as HALF_TARGET,UNIFORM as HALF_UNIFORM,SOURCE as HALF_SOURCE,COMBINED

MAIN,TARGET,UNIFORM='SACP_CONTRAST_L1','SACP_TARGET_DIR_L1','SACP_MATCHED_UNIFORM_L1'
BRANCHES=(MAIN,TARGET,UNIFORM)
HALF={MAIN:HALF_MAIN,TARGET:HALF_TARGET,UNIFORM:HALF_UNIFORM}
HISTORY=(U,S,HALF_MAIN,HALF_TARGET,HALF_SOURCE,HALF_UNIFORM,COMBINED)
MAIN_TABLE=(U,S,HALF_MAIN,HALF_TARGET,HALF_UNIFORM,*BRANCHES,COMBINED)
ALL_TABLE=(*MAIN_TABLE,HALF_SOURCE)
PARENT_PREDICTION_SHA='dcd4d2a3eb5867181b84e56c87abe3346572538eac27565f93b5e6175c7063d0'
ETA_AXES=(HALF_MAIN,HALF_TARGET,HALF_SOURCE)
CONFIG=dict(version='sacp_fixed_strength_l1_v1',parent_version=PARENT_CONFIG['version'],parent_strength=.5,strength=1.,
    layers=list(LAYERS),eta_axes=list(ETA_AXES),consumed_eta_axes=[0,1],populations=deepcopy(PARENT_CONFIG['populations']),
    distance_atol=PARENT_CONFIG['distance_atol'],distance_rtol=PARENT_CONFIG['distance_rtol'],score_atol=PARENT_CONFIG['score_atol'],
    margin_tolerance=1e-6,nstrips=200,FPR_max=.3,scoring='frozen_parent_Source_CH1; CURRENT_supplement',
    full_payload_hash_scan=False,parameter_search=False)


def job_plan():
    return dict(version=CONFIG['version'],primary=MAIN,prediction_images=sum(v[0] for v in CONFIG['populations'].values()),
        new_branches=list(BRANCHES),CPU_pixel_jobs=[dict(domain=d,branch=b) for d in DOMAINS for b in BRANCHES],
        AUPRO_jobs=[dict(domain=d,branch=b) for d in DOMAINS for b in BRANCHES],historical_branches=list(HISTORY),
        main_table_branches=list(MAIN_TABLE),appendix_only_branches=[HALF_SOURCE],
        new_metric_rows=len(BRANCHES)*(len(DOMAINS)+1)*2,main_metric_rows=len(MAIN_TABLE)*(len(DOMAINS)+1)*2,
        complete_metric_rows=len(ALL_TABLE)*(len(DOMAINS)+1)*2,
        query_forward_calls=0,exact_search_calls=0,subspace_fit_calls=0,training_updates=0)
