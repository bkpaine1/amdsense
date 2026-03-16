# Amdsense Round 5 Report — The Deep Squeeze
**Generated**: 2026-03-15 20:19
**Hardware**: AMD Ryzen AI MAX+ 395 / Radeon 8060S (gfx1151)
**Previous best**: val_bpb 1.2189
**Round 5 best**: val_bpb 1.215286
**Improvement over R4**: 0.30%
**Total improvement (R3→R5)**: 0.93%

## Best Recipe (Round 5)
```python
ASPECT_RATIO = 40
HEAD_DIM = 64
WINDOW_PATTERN = "SSSSSL"
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
```

## Embedding LR Sweep
| Config | val_bpb | VRAM (MB) | tok/sec | Status |
|--------|---------|-----------|---------|--------|
| {'embedding_lr': 0.4} | nan | 3946 | 112,180 | CRASH |
| {'embedding_lr': 0.5} | 1.219490 | 3946 | 109,500 | ok |
| {'embedding_lr': 0.6} | 1.223239 | 3946 | 109,213 | ok |
| {'embedding_lr': 0.7} | 1.220036 | 3946 | 109,691 | ok |
| {'embedding_lr': 0.8} | 1.218873 | 3946 | 109,664 | ok |
| {'embedding_lr': 0.9} | 1.221754 | 3946 | 109,449 | ok |
| {'embedding_lr': 1.0} | 1.220123 | 3946 | 109,121 | ok |
| {'embedding_lr': 1.2} | 1.220203 | 3946 | 109,503 | ok |

## Unembedding LR Sweep
| Config | val_bpb | VRAM (MB) | tok/sec | Status |
|--------|---------|-----------|---------|--------|
| {'unembedding_lr': 0.005} | 1.221110 | 3946 | 109,866 | ok |
| {'unembedding_lr': 0.008} | 1.218221 | 3946 | 109,737 | ok |
| {'unembedding_lr': 0.01} | 1.219836 | 3946 | 109,527 | ok |
| {'unembedding_lr': 0.012} | 1.219746 | 3946 | 109,660 | ok |
| {'unembedding_lr': 0.015} | 1.224682 | 3946 | 109,798 | ok |
| {'unembedding_lr': 0.018} | nan | 3946 | 112,250 | CRASH |
| {'unembedding_lr': 0.022} | 1.234507 | 3946 | 109,484 | ok |
| {'unembedding_lr': 0.028} | 1.229905 | 3946 | 110,186 | ok |

## Scalar LR Sweep
| Config | val_bpb | VRAM (MB) | tok/sec | Status |
|--------|---------|-----------|---------|--------|
| {'scalar_lr': 0.3} | 1.219625 | 3946 | 109,766 | ok |
| {'scalar_lr': 0.4} | 1.220514 | 3946 | 109,775 | ok |
| {'scalar_lr': 0.5} | 1.223419 | 3946 | 109,590 | ok |
| {'scalar_lr': 0.6} | 1.220478 | 3946 | 109,143 | ok |
| {'scalar_lr': 0.7} | 1.216357 | 3946 | 109,572 | ok |
| {'scalar_lr': 0.8} | 1.218154 | 3946 | 109,578 | ok |
| {'scalar_lr': 1.0} | nan | 3946 | 112,112 | CRASH |

## Final LR Frac Sweep
| Config | val_bpb | VRAM (MB) | tok/sec | Status |
|--------|---------|-----------|---------|--------|
| {'final_lr_frac': 0.01} | 1.217058 | 3946 | 109,960 | ok |
| {'final_lr_frac': 0.03} | 1.216311 | 3946 | 109,179 | ok |
| {'final_lr_frac': 0.05} | 1.216081 | 3946 | 109,001 | ok |
| {'final_lr_frac': 0.07} | 1.215922 | 3946 | 109,776 | ok |
| {'final_lr_frac': 0.09} | 1.216918 | 3946 | 109,544 | ok |
| {'final_lr_frac': 0.12} | 1.217145 | 3946 | 109,983 | ok |
| {'final_lr_frac': 0.15} | 1.218733 | 3946 | 109,824 | ok |
| {'final_lr_frac': 0.2} | nan | 3946 | 112,532 | CRASH |

## Matrix LR Fine-grain
| Config | val_bpb | VRAM (MB) | tok/sec | Status |
|--------|---------|-----------|---------|--------|
| {'matrix_lr': 0.01} | 1.233737 | 3946 | 109,839 | ok |
| {'matrix_lr': 0.02} | 1.221667 | 3946 | 109,767 | ok |
| {'matrix_lr': 0.03} | 1.219862 | 3946 | 108,052 | ok |
| {'matrix_lr': 0.04} | 1.220376 | 3946 | 109,265 | ok |
| {'matrix_lr': 0.05} | 1.216417 | 3946 | 109,501 | ok |
| {'matrix_lr': 0.06} | 1.222874 | 3946 | 109,480 | ok |
| {'matrix_lr': 0.07} | 1.225634 | 3946 | 109,854 | ok |
| {'matrix_lr': 0.08} | 1.224332 | 3946 | 109,030 | ok |

## Weight Decay Sweep
| Config | val_bpb | VRAM (MB) | tok/sec | Status |
|--------|---------|-----------|---------|--------|
| {'weight_decay': 0.04} | 1.216537 | 3946 | 109,271 | ok |
| {'weight_decay': 0.06} | 1.216535 | 3946 | 108,966 | ok |
| {'weight_decay': 0.08} | 1.215943 | 3946 | 109,300 | ok |
| {'weight_decay': 0.1} | 1.216208 | 3946 | 108,792 | ok |
| {'weight_decay': 0.12} | nan | 3946 | 112,124 | CRASH |
| {'weight_decay': 0.14} | 1.216631 | 3946 | 109,137 | ok |
| {'weight_decay': 0.16} | 1.217406 | 3946 | 109,645 | ok |
| {'weight_decay': 0.2} | 1.219137 | 3946 | 109,346 | ok |

## Adam Beta1 Sweep
| Config | val_bpb | VRAM (MB) | tok/sec | Status |
|--------|---------|-----------|---------|--------|
| {'beta1': 0.7} | 1.215140 | 3946 | 109,718 | ok |
| {'beta1': 0.75} | 1.214813 | 3946 | 109,295 | ok |
| {'beta1': 0.8} | 1.215606 | 3946 | 109,557 | ok |
| {'beta1': 0.85} | 1.216480 | 3946 | 109,594 | ok |
| {'beta1': 0.9} | 1.219396 | 3946 | 109,563 | ok |
| {'beta1': 0.92} | 1.221301 | 3946 | 109,196 | ok |
| {'beta1': 0.95} | 1.225746 | 3946 | 109,336 | ok |

## Adam Beta2 Sweep
| Config | val_bpb | VRAM (MB) | tok/sec | Status |
|--------|---------|-----------|---------|--------|
| {'beta2': 0.95} | nan | 3946 | 112,900 | CRASH |
| {'beta2': 0.96} | nan | 3946 | 112,814 | CRASH |
| {'beta2': 0.97} | 1.215275 | 3946 | 109,658 | ok |
| {'beta2': 0.98} | 1.215658 | 3946 | 109,573 | ok |
| {'beta2': 0.99} | 1.216464 | 3946 | 109,407 | ok |
| {'beta2': 0.995} | 1.217464 | 3946 | 109,474 | ok |
| {'beta2': 0.999} | 1.218686 | 3946 | 109,514 | ok |

## Batch Size Sweep
| Config | val_bpb | VRAM (MB) | tok/sec | Status |
|--------|---------|-----------|---------|--------|
| {'total_batch': '2**14', 'device_batch': '8'} | nan | 2111 | 108,015 | CRASH |
| {'total_batch': '2**14', 'device_batch': '16'} | 0.000000 | 0 | 0 | CRASH |
| {'total_batch': '2**15', 'device_batch': '16'} | 1.215442 | 3946 | 109,700 | ok |
| {'total_batch': '2**15', 'device_batch': '32'} | 0.000000 | 0 | 0 | CRASH |
| {'total_batch': '2**16', 'device_batch': '16'} | 1.314707 | 4023 | 91,130 | ok |
| {'total_batch': '2**16', 'device_batch': '32'} | 1.226026 | 7617 | 111,365 | ok |
| {'total_batch': '3*2**14', 'device_batch': '16'} | 0.000000 | 0 | 0 | CRASH |

## Phase 4: Interaction Effects
| Config | val_bpb | Status | Notes |
|--------|---------|--------|-------|
| all_combined | nan | CRASH | all_combined |
| r4_baseline | 1.220064 | ok | r4_baseline |
| drop_EMBEDDING_LR | nan | CRASH | Dropped: EMBEDDING_LR |
| drop_UNEMBEDDING_LR | 1.222265 | ok | Dropped: UNEMBEDDING_LR |
| drop_SCALAR_LR | nan | CRASH | Dropped: SCALAR_LR |
| drop_FINAL_LR_FRAC | 1.215053 | ok | Dropped: FINAL_LR_FRAC |
| drop_MATRIX_LR | 1.214631 | ok | Dropped: MATRIX_LR |
| drop_WEIGHT_DECAY | 1.215472 | ok | Dropped: WEIGHT_DECAY |
| drop_ADAM_BETAS | 1.218711 | ok | Dropped: ADAM_BETAS |
| drop_TOTAL_BATCH_SIZE | 1.215368 | ok | Dropped: TOTAL_BATCH_SIZE |
| drop_DEVICE_BATCH_SIZE | 1.215380 | ok | Dropped: DEVICE_BATCH_SIZE |

## Phase 5: Extended Training (10 min)
| Run | val_bpb | Steps | tok/sec |
|-----|---------|-------|---------|
| 1 | 1.215286 | 1016 | 109,905 |
| 2 | 1.216111 | 1010 | 108,878 |
| 3 | nan | 1008 | 109,411 |

## Phase 6: Memory Wall v2
| Config | Params (M) | VRAM (MB) | val_bpb | tok/sec | Status | >4090? |
|--------|-----------|-----------|---------|---------|--------|--------|
| aspect48_batch2_16 | 33.0 | 8949 | 1.245946 | 92,648 | ok | no |
| aspect48_batch2_17 | 33.0 | 17558 | 1.279243 | 95,713 | ok | no |
| aspect48_batch2_18 | 33.0 | 17655 | 1.472701 | 96,316 | ok | no |
| aspect56_batch2_16 | 41.3 | 5472 | 1.336969 | 78,247 | ok | no |
| aspect56_batch2_17 | 41.3 | 10408 | nan | 83,389 | CRASH | no |
| best_batch2_16 | 25.6 | 7617 | 1.226491 | 110,742 | ok | no |
| best_batch2_17 | 25.6 | 14960 | 1.274429 | 112,645 | ok | no |
| best_batch2_18 | 25.6 | 15037 | 1.468920 | 113,173 | ok | no |
| best_batch2_19 | 25.6 | 29722 | 1.664971 | 112,727 | ok | YES |
| best_batch2_20 | 25.6 | 29722 | 1.820192 | 112,214 | ok | YES |
| depth10_tuned | 49.8 | 3650 | 1.361429 | 61,642 | ok | no |
| aspect64_depth8 | 50.3 | 3425 | 1.395944 | 58,294 | ok | no |
| aspect64_depth10 | 85.9 | 3179 | 1.601582 | 40,049 | ok | no |

### Kill Shot: 2 models trained beyond 4090's 24GB limit
- **best_batch2_19**: 25.6M params, 29722MB VRAM, val_bpb=1.664971
- **best_batch2_20**: 25.6M params, 29722MB VRAM, val_bpb=1.820192

---
*Round 5: The Deep Squeeze — every hyperparameter earned its place*
*AMD Radeon 8060S on GMKTEC EVO X2 ($1,999) vs NVIDIA RTX 4090 (~$2,400)*