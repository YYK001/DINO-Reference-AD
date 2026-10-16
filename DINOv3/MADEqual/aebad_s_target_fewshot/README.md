# AeBAD-S Target-normal Few-shot + Exact SuperADD + MAD-Equal

Protocol: `mad_equal_aebad_s_target_fewshot_seed42_v1`.

The experiment rebuilds Exact Official memory from all 521 ordered AeBAD-S
source-normal images. The existing builder applies Official brightness to the
455 memory-fit images and produces one 100K bank for each of layers
7/15/23/31. The saved `official_memory.pt` is reusable by later support
experiments. Reserve at least about 150 GiB of temporary disk space for the
approximately 121 GiB candidate scratch peak.

One source-only MAD state is fitted from the 66 unaugmented calibration
images. Four target domains each contribute a deterministic nested pool of
eight real target-normal support images. Those 32 images are excluded from
every method and every K, leaving the same 1,607 held-out images for all ten
methods. Source-extra uses eight distinct images from the source calibration
set as a capacity-matched control.

Support images are unaugmented and retain all patch features. Query DINO
features are extracted once per held-out image; Exact Official, source-extra,
and matched-target distances are scanned from that shared query tensor.

Run:

```bash
bash DINOv3/MADEqual/aebad_s_target_fewshot/scripts/run_aebad_s_target_fewshot_seed42.sh
```

The MMR-compatible numbers are output-side post-evaluation only, not a full
MMR input-protocol reproduction.
