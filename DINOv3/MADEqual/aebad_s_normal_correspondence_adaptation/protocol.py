"""One predeclared configuration, independent of test labels and metrics."""
VERSION = 'normal_correspondence_residual_v1'
LAYERS = (7, 15, 23, 31)
DOMAINS = ('background', 'illumination', 'view')
SEEDS = (42, 43, 44)
U, S = 'UNION_K8', 'SEGMENT_K8'
MEMBERS = tuple('ADAPT_SEED'+str(s) for s in SEEDS)
ENS = 'ADAPT_ENSEMBLE'
BRANCHES = (U, S, *MEMBERS, ENS)
COMBINED = 'SEGMENT_SCORE_UNION_MAP'
CONFIG = dict(version=VERSION, layers=list(LAYERS), dimension=1280, hidden=512,
    support_seed=42, training_seeds=list(SEEDS), match_seed=42, match_limit=2048,match_block=256,
    SS_nearest=2, SS_random=2, ST_nearest=16, ST_random=4, mutual_k=3,
    positive_temperature=.07, negative_count=4, negative_fraction=.5,
    context='valid_3x3_including_center', teacher='mean4_point_half_context_half',
    source_steps=2000, target_steps=10000, batch=256, id_extra_per_population=256,
    learning_rate=1e-4, minimum_lr=1e-6, weight_decay=1e-4, warmup_fraction=.05,
    gradient_clip=1., loss='Lpos+Lneg+0.1*Lid', negative_margin=.8,
    output_dimension=1280, distance_divisor=1280, query_modes=['stream','disk'],
    extraction='original_FP32_host_640_128_resize_0.625_brightness1',
    training_autocast='BF16_linear_only_CUDA', inference='FP32_no_autocast_no_TF32',
    checkpoint='final_update_only', nstrips=200, FPR_max=.3,
    populations=dict(background=[297,85,212],illumination=[265,67,198],view=[364,84,280]))
