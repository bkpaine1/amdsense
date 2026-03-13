# Amdsense Round 3 Report — Confirmation, Ablation & Failure Boundaries
**Generated**: 2026-03-12 12:24
**Hardware**: AMD Ryzen AI MAX+ 395 / Radeon 8060S (gfx1151)

## Layer 2: Confirmation Runs
*Same recipe, 5 runs. Establishes variance band.*

| Metric | Mean | Min | Max | Spread |
|--------|------|-----|-----|--------|
| val_bpb | 1.258382 | 1.256062 | 1.263012 | 0.006950 |
| MFU % | 34.88 | 34.75 | 35.23 | 0.48 |
| tok/sec | 61,499 | 61,116 | 62,097 | 981 |

**Variance band**: 0.006950 — any improvement smaller than this is noise.

## Layer 3: Ablation Runs
*Revert each optimization to default. Measures individual contribution.*

Best recipe baseline: **1.262674**

| Change | Reverted To | val_bpb | Delta | Impact |
|--------|-------------|---------|-------|--------|
| Revert HEAD_DIM 64→128 | 128 | 1.276670 | +0.013996 (+1.11%) | CRITICAL |
| Revert WARMDOWN_RATIO 0.7→0.3 | 0.3 | 1.276278 | +0.013604 (+1.08%) | CRITICAL |
| Revert WEIGHT_DECAY 0.12→0.2 | 0.2 | 1.265526 | +0.002852 (+0.23%) | minor |
| Revert EMBEDDING_LR 0.8→1.0 | 1.0 | 1.264362 | +0.001688 (+0.13%) | minor |
| Revert MATRIX_LR 0.07→0.08 | 0.08 | 1.262130 | -0.000544 (-0.04%) | neutral/better |
| Revert WARMDOWN to 0.5 | 0.5 | 1.261673 | -0.001001 (-0.08%) | neutral/better |
| Revert SCALAR_LR 0.6→0.5 | 0.5 | 1.261010 | -0.001664 (-0.13%) | neutral/better |
| Revert ADAM_BETAS to (0.8, 0.95) | (0.8, 0.95) | 1.260915 | -0.001759 (-0.14%) | neutral/better |
| Revert FINAL_LR_FRAC 0.07→0.0 | 0.0 | 1.259924 | -0.002750 (-0.22%) | neutral/better |
| Revert UNEMBEDDING_LR 0.012→0.008 | 0.008 | 1.256625 | -0.006049 (-0.48%) | neutral/better |

## Layer 4: Failure Boundary Map
*Systematically test where training breaks.*

| Configuration | Status | val_bpb | Notes |
|--------------|--------|---------|-------|
| batch 2^14 device_batch 8 | OK | 1.265170 | TOTAL_BATCH_SIZE=2**14, DEVICE_BATCH_SIZE=8 |
| batch 2^14 device_batch 16 | CRASH | — | NaN/crash |
| batch 2^13 device_batch 8 | CRASH | — | NaN/crash |
| depth 10 | OK | 1.315949 | DEPTH=10 |
| depth 12 | CRASH | — | timeout |
| depth 16 | CRASH | — | timeout |
| head_dim 32 | CRASH | — | NaN/crash |
| head_dim 256 | OK | 1.298855 | HEAD_DIM=256 |
| matrix LR 0.15 (high) | OK | 1.273962 | MATRIX_LR=0.15 |
| matrix LR 0.20 (very high) | CRASH | — | NaN/crash |
| weight decay 0.0 (none) | OK | 1.259818 | WEIGHT_DECAY=0.0 |
| weight decay 0.5 (heavy) | OK | 1.272274 | WEIGHT_DECAY=0.5 |
| warmdown 0.9 | OK | 1.263648 | WARMDOWN_RATIO=0.9 |
| warmdown 0.95 | OK | 1.261976 | WARMDOWN_RATIO=0.95 |
| embed LR 2.0 (high) | OK | 1.262742 | EMBEDDING_LR=2.0 |
| embed LR 0.1 (low) | OK | 1.277754 | EMBEDDING_LR=0.1 |
| aspect 128 (wide) | CRASH | — | timeout |
| aspect 32 (narrow) | OK | 1.226687 | ASPECT_RATIO=32 |

---
*Round 3: Confirmation + Ablation + Failure Boundaries*
*Next: Layer 5 (cross-hardware comparison on RunPod 4070)*