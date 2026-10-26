"""Fixed method constants plus per-run profiles/plans. No mutable global selection."""
from copy import deepcopy
VERSION='reference_residual_segmentation_v1'
SYNTHESIS_VERSION=VERSION  # Preserve original RGB/donor RNG across run profiles.
DEFAULT_PROFILE='compact_v1'
LAYERS=(7,15,23,31)
DOMAINS=('background','illumination','view')
SEEDS=(42,43,44)
FAMILIES=('REF_ST','NOREF_ST','REF_SOURCE_QUERY')
U,S='UNION_K8','SEGMENT_K8'
def member(family,seed):return f'{family}_SEED{seed}'
def ensemble(family):return family+'_ENSEMBLE'
MEMBERS=tuple(member(f,s) for f in FAMILIES for s in SEEDS)
ENSEMBLES=tuple(ensemble(f) for f in FAMILIES)
BRANCHES=(U,S,*MEMBERS,*ENSEMBLES)
PRIMARY=ensemble('REF_ST')
COMBINED='SEGMENT_SCORE_UNION_MAP'
CONFIG=dict(version=VERSION,layers=list(LAYERS),dimension=1280,support_seed=42,
    training_seeds=list(SEEDS),families=list(FAMILIES),Source_images=521,Source_training=455,Source_calibration=66,
    populations=dict(background=[297,85,212],illumination=[265,67,198],view=[364,84,280]),
    bank_rows=100000,sampler_rounds=100,source_bank_augmentation='original_brightness_0.8_1.2',
    fold='ordered_455_alternating_A_B',sim_supports='opposite_fold_fixed_7_or_8_by_origin',
    synthesis_seed=20261026,source_versions=8,target_versions=96,
    synthesis_cycle=['poisson','poisson','texture','scratch'],retry_limit=4,
    scales=dict(small=[.01,.025],medium=[.04,.10],large=[.12,.25]),
    scratch_width_pixels=[1,3],poisson_mode='NORMAL_CLONE',texture_alpha=[.55,.95],
    change_minimum=2/255,soft_change_scale=.10,minimum_changed_pixels=1,
    foreground=dict(method='Source_PC1_center_sign_Otsu_largest_component',iterations=12,
        bank_stride=8,minimum_area=.05,maximum_area=.85,minimum_mask_overlap=.8),
    vector_normalization='original_Source_channel_mean_and_scalar_RMS',distance_divisor=1280,
    channels=128,dilations=[1,2],halo=4,tile_center=16,tile_normal_probability=.5,
    source_updates=2000,target_updates=12000,effective_batch_images=4,
    query_balance='normal_synthetic_1_1; target_stage_REF_ST_NOREF_ST_Source_Target_1_1',
    learning_rate=3e-4,weight_decay=1e-4,warmup_fraction=.05,minimum_lr=1e-6,
    gradient_clip=1.,log_every=100,checkpoint_every=200,
    loss='mean_layer_balanced_BCE+balanced_BCE_mean_logit+0.5_soft_Dice_mean_logit',
    precision='raw_and_NN_and_residual_FP32; head_training_BF16_CUDA; loss_FP32; no_TF32',
    logit_minimum_scale=1e-6,logit_minimum_relative_scale=1e-5,
    nstrips=200,FPR_max=.3,output_downscale=4)


def resolve(profile=DEFAULT_PROFILE):
    """CONFIG/legacy names above describe old full_v1, never select an active run."""
    if profile not in ('compact_v1','full_v1'):raise ValueError('unknown profile: '+str(profile))
    value=deepcopy(CONFIG)
    value.update(profile=profile,run_config_version='reference_residual_profiles_v1',synthesis_algorithm_version=SYNTHESIS_VERSION,
        active_families=list(FAMILIES),enable_ensemble=True,primary=PRIMARY,training_nn_layout='physical_tile_union_v1')
    if profile=='compact_v1':
        value.update(training_seeds=[42],families=['REF_ST'],active_families=['REF_ST'],source_versions=2,target_versions=16,
            source_updates=1000,target_updates=4000,enable_ensemble=False,primary=member('REF_ST',42))
    return value


def job_plan(config):
    families=config['active_families'];seeds=config['training_seeds'];jobs=[]
    source_families=list(dict.fromkeys('NOREF_ST' if f=='NOREF_ST' else 'REF_ST' for f in families))
    for phase in ('source',*DOMAINS):
        for family in source_families if phase=='source' else families:
            for seed in seeds:jobs.append(dict(phase=phase,family=family,seed=seed,
                updates=config['source_updates' if phase=='source' else 'target_updates']))
    members=[member(f,s) for f in families for s in seeds]
    ensembles=[ensemble(f) for f in families] if config['enable_ensemble'] else []
    branches=[U,S,*members,*ensembles]
    slots=config['Source_training']*config['source_versions']+len(DOMAINS)*8*config['target_versions']
    return dict(profile=config['profile'],training_jobs=jobs,source_jobs=sum(j['phase']=='source' for j in jobs),
        target_jobs=sum(j['phase']!='source' for j in jobs),planned_updates=sum(j['updates'] for j in jobs),
        training_seeds=seeds,active_families=families,members=members,ensembles=ensembles,branches=branches,primary=config['primary'],
        planned_synthetic_slots=slots,domains=list(DOMAINS),evaluation_images=sum(v[0] for v in config['populations'].values()),
        localization_jobs=[dict(domain=d,branch=b) for d in DOMAINS for b in branches],
        expected_metric_rows=(len(branches)+1)*2*(len(DOMAINS)+1),combined=COMBINED,
        not_requested=[dict(branch=b,status='not_requested') for b in (*MEMBERS,*ENSEMBLES) if b not in branches])
