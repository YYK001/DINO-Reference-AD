# AeBAD-S Synthetic Few-shot P0

Protocol: `mad_equal_aebad_s_synthetic_fewshot_seed42_v1`.

This is a minimal control for the completed AeBAD-S target-normal few-shot run. It asks whether eight distinct source-normal images, each transformed once by the existing Broad6 Compose2 implementation, can replace eight real target-normal support images under the same Exact support-scan protocol.

The method name `SYNTH_BROAD6_K{1,2,4,8}_MAD` means **GNL-inspired Broad6 Compose2 synthetic support**. It is not an implementation of official GNL.

## Reused frozen state

The explicit `--base-run` must be the completed `mad_equal_aebad_s_target_fewshot_seed42_v1` formal directory. The runner reads its:

- exact SOURCE_EXTRA identities and the 32 excluded target-support identities;
- 1,607-image held-out membership;
- per-image `official_raw` tensors and output shapes;
- source-only MAD calibration statistics;
- completed MAD, SOURCE_EXTRA and TARGET_MATCHED metrics;
- existing mechanism and MMR-compatible results.

It does not load, rebuild, or scan `official_memory.pt`. It does not refit MAD.

## New computation

For each of the eight frozen SOURCE_EXTRA identities:

1. load the original source-normal RGB image;
2. call the existing `broad6_compose2(image, relative_path, seed=42)` once;
3. use deterministic Exact SuperADD preprocessing with no Official brightness;
4. extract the four full patch-feature layers once.

For each held-out query, the runner performs one DINO forward, calls the existing `_support_prefix_minima` over the eight synthetic image banks, and computes all K values by cumulative minimum. The saved base `official_raw` is then combined by elementwise minimum. All domains share the same synthetic pool.

## Formal run

```bash
BASE_RUN=/root/autodl-tmp/DINO-NVS/DINOv3/outputs/mad_equal/aebad_s_target_fewshot_seed42_v1 \
bash DINOv3/MADEqual/aebad_s_synthetic_fewshot/scripts/run_aebad_s_synthetic_fewshot_seed42.sh
```

The script requires `BASE_RUN` explicitly and writes by default to:

`DINOv3/outputs/mad_equal/aebad_s_synthetic_fewshot_seed42_v1`

The run requires CUDA for DINO and the frozen ADEval AUPRO backend.

