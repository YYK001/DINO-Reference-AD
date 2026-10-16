# RobustAD Target-normal Few-shot + MAD-Equal P0

This is a minimal RobustAD/Seed42 cross-category confirmation for
**target-normal few-shot, condition-aware adaptation**. It is not an
unsupervised method. The frozen categories are MetalParts, PCB and PiledBags.

The experiment keeps the frozen Exact SuperADD Official100 banks and the
source-only MAD calibration unchanged. For each of the six target shifts it
selects a stable eight-normal support pool, evaluates every method on the same
held-out set after removing all eight support images, and retains every anomaly.
The K=1/2/4/8 banks are nested prefixes. A 1/2/4/8-image source-calibration
prefix is the equal-capacity control.

Support images use the existing deterministic four-layer feature extractor and
retain every stitched patch. An evaluation image has one four-layer query
forward. Existing Exact 1-NN scans Official100, each source support image and
each matched-target support image; cumulative minima provide every K without
repeating the query forward or rebuilding/coresetting Official100.

All calibrated maps call the existing source-fitted `fit_calibrators` and
`mad_equal`. No target support is used to refit MAD. MetalParts and PCB use
pixel masks and project ADEval CUDA with `nstrips=200`. PiledBags is frozen as
image-only: its masks are never opened and its Pixel AUROC, Pixel AUPR and
AUPRO fields remain unavailable.

Target shifts are category-specific: MetalParts has six; PCB uses lighting,
white balancing, rotation, position and shadow; PiledBags uses background box
color, lighting, position/rotation, scale and shadow. Every category keeps the
same 180 source normals, 157/23 split, stable category-bound support ordering,
fixed eight-image exclusion, and nested K=1/2/4/8 prefixes.

Optional `--source-output-dir` and `--relation-cache-root` form one external
baseline bundle. It is used only when Official banks, source MAD and all needed
baseline raw files are present. Otherwise the runner passes all 180 ordered
source normals to the existing Official builder, which internally applies the
157/23 `index % 8` split, then fits source MAD from the 23 calibration normals.

Main outputs are `support_manifest.csv`, `support_bank_stats.csv`,
`sample_scores.csv`, `group_metrics.csv`, `target_shift_macro.csv`,
`method_differences.csv`, `mechanism_per_image.csv`, `mechanism_by_shift_layer.csv`,
`mechanism_macro.csv`, `timing.csv`, `complete.json`, and
`objective_report.md`.

New runs compute normal and defect support-hit rates inside each image first,
then average images equally. PiledBags reports only normal-side `R_N` and normal
hit rate; it does not treat all anomaly patches as a defect region. The already
completed MetalParts result is not rewritten, and its legacy patch-pooled hit
rates are excluded from the three-category summary.

Run PCB and PiledBags separately with the corresponding scripts, or run both
sequentially with `run_remaining_categories_seed42.sh`. After all three result
directories exist, `summarize_three_categories.py` reads only their final
CSV/JSON files. It averages target shifts within category and categories
equally: image metrics use three categories, while pixel metrics/AUPRO use PCB
and MetalParts only. It reports every frozen K and does not select a best K.

Example summary invocation:

```bash
python -m DINOv3.MADEqual.robustad_target_fewshot_p0.summarize_three_categories \
  --metalparts /path/to/robustad_target_fewshot_metalparts_seed42_p0_v1 \
  --pcb /path/to/robustad_target_fewshot_pcb_seed42_p0_v1 \
  --piledbags /path/to/robustad_target_fewshot_piledbags_seed42_p0_v1 \
  --output-dir /path/to/robustad_target_fewshot_three_category_seed42_summary
```

The runner requires CUDA for Exact DINO inference. Pixel-capable categories
also require the frozen ADEval backend.
