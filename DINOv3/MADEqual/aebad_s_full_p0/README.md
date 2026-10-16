# AeBAD-S Exact SuperADD + MAD-Equal Full P0

Protocol ID: `mad_equal_aebad_s_full_p0_seed42_v1`.

This directory is a thin AeBAD-S adapter and runner around the existing Exact
SuperADD implementation. It does not contain another MAD or MAD-Equal formula.

## Frozen execution

- DINOv3 ViT-H+/16, layers 7/15/23/31
- Exact Official SuperADD memory, built once from all 521 sorted `train/good`
  records; the Official builder performs the frozen `index % 8` exclusion
- the four independent Official layer banks are built sequentially through the
  existing Exact Official builder, bounding candidate storage to one layer;
  brightness draws and sampler RNG state retain the original four-layer order
- source split: 455 memory images and 66 calibration images
- 1639 test images: same 689, background 305, illumination 273, view 372
- one `observe_pil` call per image produces the raw four-layer 1-NN tensor
- Official and every calibrated/diagnostic method share that tensor and memory
- MAD parameters come from `fit_calibrators`; MAD-Equal calls the existing
  `DINOv3.relation_oracle_cache.extract.mad_equal`
- AUPRO uses the project ADEval CUDA backend with `nstrips=200`

The runner caches uncompressed per-image raw-distance NPZ files and the Official
bank so an interrupted run can resume without recomputing completed images. The
cache performs ordinary protocol/path/shape checks only; it does not repeatedly
hash dataset files.

## Run

```bash
cd /root/autodl-tmp/DINO-NVS
export AEBAD_ROOT=/root/autodl-tmp/datasets/AeBAD
bash DINOv3/MADEqual/aebad_s_full_p0/scripts/run_aebad_s_full_seed42.sh
```

Override `PROJECT_ROOT`, `PYTHON`, `OFFICIAL_ROOT`, `DINOV3_ROOT`, `CHECKPOINT`,
`OUTPUT_DIR`, or `DEVICE` through environment variables when needed.

## Outputs

The main outputs are `calibration_stats.csv`, `sample_scores.csv`,
`group_metrics.csv`, `summary.csv`, `objective_report.md`, and `complete.json`.
`summary.csv` contains an unweighted four-domain macro and a shifted macro over
background/illumination/view only. `complete.json` is written last.
