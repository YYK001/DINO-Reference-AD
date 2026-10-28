"""One configuration; no test-dependent configuration selection."""
VERSION = 'blind_context_block5_v1'
LAYERS = (7, 15, 23, 31)
U, S = 'UNION_K8', 'SEGMENT_K8'
MAIN = 'BLIND_CONTEXT_UNION'
COMBINED = 'SEGMENT_SCORE_UNION_MAP'
BRANCHES = (U, S, COMBINED, MAIN)
CONFIG = dict(version=VERSION, domain='background', seed=42, layers=list(LAYERS),
              dimension=1280, source_bank_rows=100000, source_images=64,
              block_patches=5, patch_pixels=16, phases=4, context_dimension=128,
              attention_dimension=32, temperature=0.1, source_topk=4,
              source_dictionary_positions=512, source_training_positions=512,
              target_training_positions=1024, updates=1500,
              batch_Source=64, batch_Target=64, lr=1e-3, weight_decay=1e-4,
              population=[297, 85, 212], distance_unit='Euclidean /1280 once',
              scoring='frozen_Source_CH1; CURRENT_supplement', nstrips=200, FPR_max=.3,
              precision='original_backbone; FP32 Exact and attention; no TF32 change',
              exclusion='whole dictionary query image; Target K7 in training, K8 in test')
STAGES = ('prepare-context', 'check-isolation', 'prepare-training', 'train',
          'predict', 'evaluate', 'aupro', 'summarize')


def job_plan():
    return dict(config=VERSION, training_jobs=1, layer_heads=4, parameters=4*(2*128*32+2),
                updates=1500, dictionary_images=72, evaluation_images=297,
                maximum_dictionary_masked_extractions=72*4,
                maximum_query_masked_extractions=297*4,
                new_localization_jobs=1, new_AUPRO_jobs=1,
                reused_baselines=[U, S, COMBINED], branches=list(BRANCHES),
                parameter_search=False, calibration_refit=False)
