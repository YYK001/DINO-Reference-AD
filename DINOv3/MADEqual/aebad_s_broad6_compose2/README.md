# AeBAD-S Broad6 Compose2 + MAD-Equal

Protocol: `mad_equal_aebad_s_broad6_compose2_seed42_v1`.

This is a narrow memory-augmentation experiment on the frozen AeBAD-S Exact
SuperADD host. It reads the completed Official + MAD-Equal Full P0 result from
an explicit `--baseline-output`; it never rebuilds that baseline.

## Frozen data flow

- AeBAD-S `train/good`: 521 images, split by the existing
  `split_source_normal(..., threshold_fraction=8)` into 455 memory-fit and 66
  calibration images.
- Every memory-fit image produces exactly one augmented image. Two distinct
  operations are drawn, in order, from AutoContrast, Gamma, Contrast,
  WhiteBalance, JPEG, and IlluminationField.
- Assignment and atomic parameters reproduce `C12_BroadCompose2`: CRC32 of
  `42|tag|normalized_relative_path`, with independent assignment and
  per-operation namespaces.
- The augmented RGB image goes through the existing deterministic SuperADD
  resize/normalization and overlapping four-layer DINO forward. Official
  brightness is not called, and clean images are not unioned with augmented
  candidates.
- One upstream brightness RNG draw per memory-fit image is consumed and
  discarded, preserving the Official sampler RNG state without modifying the
  Broad6 image or calling the Official brightness preprocessing.
- The existing Official sampler produces one `(100000, 1280)` bank per layer.
- The 66 unaugmented calibration images query the new bank; the existing
  `fit_calibrators` fits the new MAD state. Target images are unaugmented and
  the existing `mad_equal` generates the final patch map.
- Ground truth is parsed only after all 1639 target raw-distance caches and the
  prediction barrier have been written.

Outputs include `augmentation_manifest.csv`, `broad6_memory.pt`,
`memory_manifest.json`, `calibration_stats.csv`, `dataset_manifest.csv`,
`sample_scores.csv`, `group_metrics.csv`, `summary.csv`,
`prediction_barrier.json`, `objective_report.md`, and `complete.json`.

## Formal invocation

```bash
BASELINE_OUTPUT=/root/autodl-tmp/DINO-NVS/DINOv3/outputs/mad_equal/aebad_s_full_p0_seed42_v1 \
bash DINOv3/MADEqual/aebad_s_broad6_compose2/scripts/run_aebad_s_broad6_compose2_seed42.sh
```

The baseline directory must contain a completed Full P0 result and its 66
calibration plus 1639 test raw caches. CUDA and the project ADEval backend are
required for the formal run.
