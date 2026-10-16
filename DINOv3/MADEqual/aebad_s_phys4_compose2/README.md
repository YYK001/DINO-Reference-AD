# AeBAD-S Phys4 Compose2 + MAD-Equal

Protocol: `mad_equal_aebad_s_phys4_compose2_seed42_v1`.

Phys4 contains Brightness, Gamma, WhiteBalance, and IlluminationField. Each of
the 455 memory-fit images receives one ordered pair of distinct transforms.
All transforms run sequentially in float32 RGB `[0,1]`; each operation clips
to `[0,1]`, and the resulting CHW float32 tensor enters the existing
deterministic Official preprocessing.

The experiment independently builds four 100K Phys4 banks using the existing
four-layer SuperADD feature extraction and Official sampler. The 66 original,
unaugmented calibration images fit a new MAD state. Target images remain
unaugmented. Existing Official, MAD-Equal, and Broad6 results are read from
explicit completed output directories and are not rerun.

```bash
BASELINE_OUTPUT=/root/autodl-tmp/DINO-NVS/DINOv3/outputs/mad_equal/aebad_s_full_p0_seed42_v1 \
BROAD6_OUTPUT=/root/autodl-tmp/DINO-NVS/DINOv3/outputs/mad_equal/aebad_s_broad6_compose2_seed42_v1 \
bash DINOv3/MADEqual/aebad_s_phys4_compose2/scripts/run_aebad_s_phys4_compose2_seed42.sh
```

After the Phys4 formal output is complete, include it in the existing
MMR-compatible post-evaluation without rerunning DINO or Exact 1-NN:

```bash
BASELINE_OUTPUT=/root/autodl-tmp/DINO-NVS/DINOv3/outputs/mad_equal/aebad_s_full_p0_seed42_v1 \
BROAD6_OUTPUT=/root/autodl-tmp/DINO-NVS/DINOv3/outputs/mad_equal/aebad_s_broad6_compose2_seed42_v1 \
PHYS4_OUTPUT=/root/autodl-tmp/DINO-NVS/DINOv3/outputs/mad_equal/aebad_s_phys4_compose2_seed42_v1 \
bash DINOv3/MADEqual/aebad_s_broad6_compose2/scripts/run_mmr_compatible_eval_seed42.sh
```

That evaluator remains output-side post-evaluation only; it is not a full MMR
input-protocol reproduction.
