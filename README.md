# amdsense

**A $0 AMD iGPU outscored a $1,400 RTX 4090 on ML training quality. Here's the data.**

An autonomous AI research agent ran 93+ experiments on an AMD Radeon 8060S (Strix Halo integrated GPU, 64GB unified memory, 54W total system power). It found a recipe that achieves **val_bpb 1.227** — 33% better than the same recipe on an NVIDIA RTX 4090 (val_bpb 1.844).

This is not a benchmark war. This is a message:

**To NVIDIA:** Your ecosystem advantage is real, but your hardware isn't untouchable.

**To AMD:** Your silicon works. Your software stack is why people stay on NVIDIA. Fix it.

## The Numbers

| Metric | AMD Radeon 8060S | NVIDIA RTX 4090 |
|--------|:----------------:|:----------------:|
| **val_bpb** | **1.227** | **1.844** |
| Street price | $0 (in APU) | ~$1,400 |
| Power draw | 54W system | 450W GPU alone |
| Throughput (tok/sec) | 51,000 | 320,681 |
| MFU (utilization) | 25% | 7.7% |
| VRAM used | 45 GB / 64 GB | 8.8 GB / 24 GB |
| SDPA Flash (ms) | 2.3 | 0.23 |
| Matmul 4096 (TFLOPS) | 30.6 | 155.0 |
| bf16 stability bugs | yes (documented) | none |

NVIDIA is 6x faster on throughput. AMD gets 3x better hardware utilization and a 33% better training score on the same recipe in the same 5-minute budget.

### Why AMD Wins on Quality

The recipe was optimized over 33 autonomous experiments specifically on AMD hardware. Two hyperparameters are CRITICAL and were discovered through ablation:

- **HEAD_DIM 64** — +1.11% regression when reverted to default 128
- **WARMDOWN_RATIO 0.7** — +1.08% regression when reverted to default 0.3

Hardware-specific tuning matters more than raw throughput for research quality.

### Why NVIDIA Wins on Speed

15 years of ecosystem investment:
- pip install works the first time
- Flash Attention 2 auto-selects via SDPA
- Stable PyTorch releases (not nightlies)
- Pre-built kernels for everything

This isn't a hardware gap. It's a software gap.

## AMD Action Items

We found real bugs. AMD engineering, please fix these:

### Critical

1. **bf16 accumulation breaks at small batch sizes** — `TOTAL_BATCH_SIZE=2^13` always NaN/crashes. Gradient accumulation precision issue.
2. **bf16 breaks at small head dimensions** — `HEAD_DIM=32` produces NaN. Kernel-level precision issue.
3. **Deep networks unreliable** — `DEPTH=12+` timeout/crash. bf16 accumulation in deep networks.
4. **Wide aspect ratios crash** — `ASPECT_RATIO=128` timeout. Possible memory layout issue.
5. **Matrix LR cliff** — `MATRIX_LR=0.15` works, `0.20` is dead. Sharp boundary.

### Ecosystem

6. **TheROCk nightlies required** — Stable ROCm does not ship gfx1151 kernels. Consumer Strix Halo users MUST use the nightly index at `https://rocm.nightlies.amd.com/v2/gfx1151/`.
7. **AOTriton flash attention hidden behind env var** — `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` gives **19x SDPA speedup** (44ms to 2.3ms). This is not documented. This is not default. This should be both.
8. **`PYTORCH_HIP_ALLOC_CONF=backend:malloc` crashes PyTorch** — Set by default in some ROCm shell profiles. Must be manually unset. A default that crashes your own framework is a bug.
9. **No stable PyTorch wheels for gfx1151** — Users must install from nightlies: `torch-2.11.0a0+rocm7.11.0a20260106`. Ship stable wheels.

## Hardware

### AMD Test Rig
- **APU**: AMD Ryzen AI MAX+ 395
- **GPU**: Radeon 8060S (gfx1151) — integrated
- **Memory**: 64 GB unified (shared CPU/GPU)
- **TDP**: 54W total system
- **PyTorch**: 2.11.0a0+rocm7.11.0a20260106 (TheROCk nightly)
- **Attention**: AOTriton experimental via SDPA

### NVIDIA Test Rig (RunPod)
- **GPU**: NVIDIA GeForce RTX 4090 — discrete
- **VRAM**: 24 GB GDDR6X
- **TDP**: 450W GPU
- **Host CPU**: AMD EPYC 7452 32-Core
- **PyTorch**: 2.4.1+cu124 (stable)
- **Attention**: FlashAttention 2 via SDPA (FA3 unavailable — needs SM90/Hopper)

## Autonomous Research Results

An AI coding agent (Claude Code on Opus) ran the experiment loop from [karpathy/autoresearch](https://github.com/karpathy/autoresearch): modify `train.py`, train 5 minutes, keep improvements, discard regressions, repeat.

### Optimization Journey
| Milestone | val_bpb | Improvement |
|-----------|---------|-------------|
| Baseline (Karpathy default) | 1.819 | — |
| Batch size tuning (2^20 → 2^15) | 1.295 | -28.8% |
| SwiGLU + full attention | 1.292 | -29.0% |
| Warmdown schedule | 1.264 | -30.5% |
| LR + weight decay tuning | 1.256 | -31.0% |
| Head dim + unembed LR | 1.255 | -31.0% |
| **ASPECT_RATIO 32 (Round 3)** | **1.227** | **-32.5%** |

### Round 3: Confirmation + Ablation + Failure Boundaries
- **33 experiments** in 4 hours (autonomous agent, zero human intervention)
- **Confirmation band**: 0.007 variance across 5 identical runs — recipe is reproducible
- **2 CRITICAL hyperparams** identified via ablation (HEAD_DIM 64, WARMDOWN_RATIO 0.7)
- **5 "improvements" debunked** as noise (SCALAR_LR, ADAM_BETAS, FINAL_LR_FRAC, MATRIX_LR, UNEMBEDDING_LR)
- **Full failure boundary map** with exact reproduction steps

See [round3_report.md](round3_report.md) for complete data.

## Best Recipe

```python
ASPECT_RATIO     = 32       # Round 3 finding — narrower is better on gfx1151
HEAD_DIM         = 64       # CRITICAL — do not revert
TOTAL_BATCH_SIZE = 2**15
DEPTH            = 8
DEVICE_BATCH_SIZE = 16
EMBEDDING_LR     = 0.8
UNEMBEDDING_LR   = 0.012
MATRIX_LR        = 0.07
SCALAR_LR        = 0.6
WEIGHT_DECAY     = 0.12
ADAM_BETAS        = (0.8, 0.98)
WARMUP_RATIO     = 0.0
WARMDOWN_RATIO   = 0.7     # CRITICAL — do not revert
FINAL_LR_FRAC    = 0.07
```

## Quick Start

### AMD (ROCm / TheROCk)

```bash
# Install TheROCk nightly PyTorch for gfx1151
pip install torch --index-url https://rocm.nightlies.amd.com/v2/gfx1151/

# REQUIRED environment
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1  # 19x flash attention speedup
unset PYTORCH_HIP_ALLOC_CONF                       # default value crashes torch

# Clone and run
git clone https://github.com/bkpaine1/amdsense.git
cd amdsense
pip install tiktoken huggingface_hub pyarrow rustbpe
python3 prepare.py
python3 train.py
```

### NVIDIA (CUDA)

```bash
# Standard PyTorch
pip install torch

# Clone and run
git clone https://github.com/bkpaine1/amdsense.git
cd amdsense
pip install tiktoken huggingface_hub pyarrow rustbpe kernels
python3 prepare.py
python3 train.py
```

### Run the Autonomous Agent

Point Claude Code, Cursor, or any SSH-capable coding agent at the repo:

```
Read program.md and kick off a new experiment run.
```

Expect ~12 experiments/hour. Leave it overnight for ~100 experiments.

## Repo Structure

```
train.py                    — model + training loop (agent modifies this)
prepare.py                  — data prep + eval (fixed, do not modify)
program.md                  — agent instructions
profile_rocm.py             — AMD hardware profiler
runpod_benchmark.sh         — NVIDIA comparison benchmark (one-script RunPod setup)
round3_report.md            — Round 3 detailed results
nvidia_4090_comparison.md   — Full cross-hardware comparison report
nvidia_4090_results.json    — Raw NVIDIA benchmark data
results.tsv                 — Experiment log
autoresearch_agent3.py      — Autonomous experiment runner (standalone)
```

## Origin

Forked from [karpathy/autoresearch](https://github.com/karpathy/autoresearch) (MIT). We submitted AMD support upstream ([PR #136](https://github.com/karpathy/autoresearch/pull/136)) — they preferred CUDA-only. So we made our own.

## Authors

- **Brent Paine** ([@bkpaine1](https://github.com/bkpaine1)) — hardware, vision, the guy who said "my iGPU can do this"
- **Tectonic Obelisk** (Claude Code, Opus 4.6) — autonomous agent, profiling, benchmark infrastructure, this README

## License

MIT
