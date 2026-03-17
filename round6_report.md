# Amdsense Round 6 Report — The Architecture Hunt
**Generated**: 2026-03-17 04:19
**Hardware**: AMD Ryzen AI MAX+ 395 / Radeon 8060S (gfx1151)
**Previous best (R5)**: val_bpb 1.2153
**Round 6 best**: val_bpb 1.207993
**Improvement over R5**: 0.60%

## Best Recipe (Round 6)
```python
ASPECT_RATIO = 40
HEAD_DIM = 64
WINDOW_PATTERN = "SLSLSL"
TOTAL_BATCH_SIZE = 2**15
EMBEDDING_LR = 0.8
UNEMBEDDING_LR = 0.008
MATRIX_LR = 0.05
SCALAR_LR = 0.7
WEIGHT_DECAY = 0.08
ADAM_BETAS = (0.75, 0.97)
WARMUP_RATIO = 0.0
WARMDOWN_RATIO = 0.75
FINAL_LR_FRAC = 0.07
DEPTH = 8
DEVICE_BATCH_SIZE = 16
KV_HEAD_DIVISOR = 6
X0_LAMBDA_INIT = 0.2
RESID_LAMBDA_INIT = 0.8
```

## Phase 1: Window Patterns
| Config | val_bpb | VRAM (MB) | Params (M) | tok/sec | Status |
|--------|---------|-----------|------------|---------|--------|
| {'window_pattern': '"SSSSSL"'} | nan | 3946 | 25.6 | 112,565 | CRASH |
| {'window_pattern': '"SSSSSSL"'} | 1.215246 | 3946 | 25.6 | 109,374 | ok |
| {'window_pattern': '"SSSL"'} | nan | 3946 | 25.6 | 112,064 | CRASH |
| {'window_pattern': '"SSL"'} | nan | 3946 | 25.6 | 112,604 | CRASH |
| {'window_pattern': '"SSSSLL"'} | 1.215442 | 3946 | 25.6 | 109,698 | ok |
| {'window_pattern': '"SLSLSL"'} | 1.215067 | 3946 | 25.6 | 109,197 | ok |
| {'window_pattern': '"SSSLSL"'} | nan | 3946 | 25.6 | 112,513 | CRASH |
| {'window_pattern': '"SSSSLSL"'} | nan | 3946 | 25.6 | 112,633 | CRASH |
| {'window_pattern': '"SSSSSSSSL"'} | nan | 3946 | 25.6 | 112,230 | CRASH |
| {'window_pattern': '"LLLLLL"'} | 1.215255 | 3946 | 25.6 | 109,998 | ok |

## Phase 2: Grouped Query Attention
| Config | val_bpb | VRAM (MB) | Params (M) | tok/sec | Status |
|--------|---------|-----------|------------|---------|--------|
| {'kv_head_divisor': 1} | nan | 3946 | 25.6 | 112,700 | CRASH |
| {'kv_head_divisor': 2} | 0.000000 | 0 | 0.0 | 0 | CRASH |
| {'kv_head_divisor': 3} | 1.232705 | 3690 | 15.9 | 114,526 | ok |
| {'kv_head_divisor': 4} | 1.232898 | 3690 | 15.9 | 114,491 | ok |
| {'kv_head_divisor': 6} | 1.232556 | 3690 | 15.9 | 114,065 | ok |

## Phase 3: MLP Ratio
| Config | val_bpb | VRAM (MB) | Params (M) | tok/sec | Status |
|--------|---------|-----------|------------|---------|--------|
| {'mlp_ratio': 2} | 1.223923 | 3275 | 22.3 | 120,316 | ok |
| {'mlp_ratio': 3} | 1.224263 | 3611 | 23.9 | 114,510 | ok |
| {'mlp_ratio': 4} | 1.214795 | 3946 | 25.6 | 109,708 | ok |
| {'mlp_ratio': 5} | 1.215493 | 4282 | 27.2 | 105,266 | ok |
| {'mlp_ratio': 6} | 1.219801 | 4619 | 28.8 | 98,810 | ok |
| {'mlp_ratio': 8} | nan | 5292 | 32.1 | 94,285 | CRASH |

## Phase 4: Residual Scaling
| Config | val_bpb | VRAM (MB) | Params (M) | tok/sec | Status |
|--------|---------|-----------|------------|---------|--------|
| {'x0_lambda_init': 0.0} | 1.218688 | 3946 | 25.6 | 109,785 | ok |
| {'x0_lambda_init': 0.05} | 1.220773 | 3946 | 25.6 | 110,041 | ok |
| {'x0_lambda_init': 0.1} | nan | 3946 | 25.6 | 112,737 | CRASH |
| {'x0_lambda_init': 0.15} | 1.219390 | 3946 | 25.6 | 109,804 | ok |
| {'x0_lambda_init': 0.2} | 1.217259 | 3946 | 25.6 | 109,539 | ok |
| {'x0_lambda_init': 0.3} | nan | 3946 | 25.6 | 112,570 | CRASH |
| {'x0_lambda_init': 0.5} | 1.219625 | 3946 | 25.6 | 109,468 | ok |
| {'resid_lambda_init': 0.5} | 1.222847 | 3946 | 25.6 | 109,790 | ok |
| {'resid_lambda_init': 0.7} | nan | 3946 | 25.6 | 112,806 | CRASH |
| {'resid_lambda_init': 0.8} | 1.216515 | 3946 | 25.6 | 109,428 | ok |
| {'resid_lambda_init': 0.9} | 1.220708 | 3946 | 25.6 | 109,579 | ok |
| {'resid_lambda_init': 1.0} | 1.217705 | 3946 | 25.6 | 109,718 | ok |
| {'resid_lambda_init': 1.1} | 1.220460 | 3946 | 25.6 | 110,094 | ok |
| {'resid_lambda_init': 1.2} | 1.220355 | 3946 | 25.6 | 109,789 | ok |
| {'resid_lambda_init': 1.5} | nan | 3946 | 25.6 | 112,921 | CRASH |

## Phase 5: Depth + LR Co-optimization
| Config | val_bpb | VRAM (MB) | Params (M) | tok/sec | Status |
|--------|---------|-----------|------------|---------|--------|
| {'depth': 10, 'matrix_lr': 0.02} | 1.414141 | 3650 | 49.8 | 61,920 | ok |
| {'depth': 10, 'matrix_lr': 0.03} | 1.381519 | 3650 | 49.8 | 62,100 | ok |
| {'depth': 10, 'matrix_lr': 0.04} | 1.375854 | 3650 | 49.8 | 62,176 | ok |
| {'depth': 10, 'matrix_lr': 0.05} | 1.371187 | 3650 | 49.8 | 62,101 | ok |
| {'depth': 10, 'matrix_lr': 0.06} | 1.369660 | 3650 | 49.8 | 62,304 | ok |
| {'depth': 10, 'matrix_lr': 0.08} | 1.381151 | 3650 | 49.8 | 62,255 | ok |
| {'depth': 12, 'matrix_lr': 0.02} | 1.548094 | 2143 | 46.4 | 62,145 | ok |
| {'depth': 12, 'matrix_lr': 0.03} | 1.480964 | 2143 | 46.4 | 62,149 | ok |
| {'depth': 12, 'matrix_lr': 0.04} | 1.484010 | 2143 | 46.4 | 62,329 | ok |
| {'depth': 6, 'aspect_ratio': 48} | 1.207993 | 3123 | 20.5 | 140,180 | ok |
| {'depth': 6, 'aspect_ratio': 56} | 1.210262 | 3652 | 26.3 | 119,682 | ok |
| {'depth': 6, 'aspect_ratio': 64} | 1.210580 | 3652 | 26.3 | 119,833 | ok |

## Phase 6: Softcap
| Config | val_bpb | VRAM (MB) | Params (M) | tok/sec | Status |
|--------|---------|-----------|------------|---------|--------|
| {'softcap': 5} | 1.275782 | 3946 | 25.6 | 110,216 | ok |
| {'softcap': 10} | 1.219361 | 3946 | 25.6 | 109,462 | ok |
| {'softcap': 12} | 1.214972 | 3946 | 25.6 | 109,987 | ok |
| {'softcap': 15} | 1.214892 | 3946 | 25.6 | 109,921 | ok |
| {'softcap': 20} | 1.215706 | 3946 | 25.6 | 110,045 | ok |
| {'softcap': 30} | nan | 3946 | 25.6 | 112,558 | CRASH |
| {'softcap': 50} | 1.220410 | 3946 | 25.6 | 110,071 | ok |
| {'softcap': 100} | 1.220821 | 3946 | 25.6 | 110,353 | ok |

## Phase 7: The Big Push
| Config | val_bpb | VRAM (MB) | Params (M) | tok/sec | Status |
|--------|---------|-----------|------------|---------|--------|
| {'config': 'all_combined', 'overrides': {'WINDOW_PATTERN': '"SLSLSL"', 'KV_HEAD_DIVISOR': '6', 'X0_LAMBDA_INIT': '0.2', 'RESID_LAMBDA_INIT': '0.8'}} | nan | 3946 | 25.6 | 112,633 | CRASH |
| {'config': 'r5_baseline'} | 1.214160 | 3946 | 25.6 | 110,476 | ok |
| {'config': 'scaled_2**16'} | 1.225270 | 7617 | 25.6 | 112,711 | ok |
| {'config': 'scaled_2**17'} | 1.273092 | 14960 | 25.6 | 113,139 | ok |
| {'config': 'wide_ar48'} | 1.325955 | 2593 | 33.0 | 91,901 | ok |
| {'config': 'wide_ar56'} | 1.337957 | 3006 | 41.3 | 76,246 | ok |

---
*Round 6: The Architecture Hunt — no upstream, no NVIDIA, our code, our ideas*
*AMD Radeon 8060S on GMKTEC EVO X2 ($1,999) — making waves*