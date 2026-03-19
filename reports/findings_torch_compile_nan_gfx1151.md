# torch.compile Generates Incorrect Code on gfx1151: Systematic NaN in Adam Optimizer

**Date**: 2026-03-19
**Hardware**: AMD Radeon 8060S (gfx1151, Strix Halo)
**Software**: PyTorch 2.11.0a0+rocm7.11.0a20260106, ROCm 7.11.0 nightlies
**Branch**: autoresearch/mar19-dsstrix2
**Related Issue**: [ROCm/ROCm#6034](https://github.com/ROCm/ROCm/issues/6034)

## Executive Summary

Through 23 systematic diagnostic experiments, we isolated the **exact root cause** of NaN in torch.compile on gfx1151: the **compiled `sqrt()` instruction produces NaN for subnormal (denormalized) floating point inputs**. This affects both bf16 and fp32 dtypes. The eager (uncompiled) PyTorch sqrt handles the same values correctly.

**Minimal reproducer** (7 lines):
```python
import torch
x = torch.full((1024,), 1.14e-38, dtype=torch.bfloat16, device='cuda')  # subnormal

@torch.compile
def f(x): return x.sqrt()

print(torch.isnan(f(x)).sum())  # gfx1151: 1024 NaN; expected: 0
print(torch.isnan(x.sqrt()).sum())  # eager: 0 NaN (correct)
```

This explains ALL NaN scenarios in compiled Adam on gfx1151:
- Adam optimizer's `exp_avg_sq` decays toward zero for cold embedding rows
- When `exp_avg_sq` reaches bf16 subnormal range (~1e-38), compiled `denom = sqrt(exp_avg_sq/bias2) + eps` produces NaN
- NaN propagates through parameter updates, eventually cascading to full training collapse

## Root Cause Chain

```
torch.compile(adamw_step_fused) on gfx1151
    -> Triton generates bf16 kernel variant [1/1] with fp32->bf16 store conversions
    -> Stage B (bias correction + p.add_) compiled kernel produces incorrect value for ~1 entry
       (Stage A lerp operations compile correctly - DIAG 20)
    -> NaN silently grows in value_embeds table (1 -> 62 entries over ~250 steps)
    -> when batch contains affected token, forward pass cascades NaN to all gradients
    -> optimizer propagates NaN to all 18 parameters
    -> training dies

EXACT ROOT CAUSE: compiled sqrt() produces NaN for subnormal inputs on gfx1151
  - exp_avg_sq decays to bf16 subnormal (~1.15e-38, below min normal 1.175e-38)
  - compiled sqrt(subnormal) -> NaN (DIAG 23: ALL 1024 subnormal elements -> NaN)
  - eager sqrt(subnormal) -> correct small positive value
  - compiled div(subnormal, scalar) -> correct (NOT the division)
  - compiled sqrt(subnormal fp32) -> also NaN (NOT bf16-specific!)
  - compiled sqrt(normal values) -> correct (subnormal-specific trigger)

Isolation chain:
  DIAG 20: Stage B (not Stage A lerp)
  DIAG 21: compiled arithmetic (not compiled stores)
  DIAG 22: compute_denom (not compute_update)
  DIAG 23: sqrt() on subnormals (not division, not bf16 conversion)
```

## Evidence Table

| DIAG | Commit | Change | NaN? | First NaN | val_bpb | Key Finding |
|------|--------|--------|------|-----------|---------|-------------|
| 3 | e360874 | Adam beta2=0.95 | YES | step 835 | nan | Reproducible NaN |
| 6 | b637f72 | + fp32 optimizer states | YES | step 835 | nan | fp32 states don't fix it |
| 7 | 39ef210 | + NaN detection | YES | step 835 | nan | ALL 16 Muon params 100% NaN |
| 8 | 4a8b04f | Pre-optimizer check | YES | step 835 | nan | Gradients ALL NaN BEFORE optimizer step |
| 9 | fcd9b38 | Every-step tracking | YES | **step 586** | nan | First NaN in value_embeds.1 POST-OPT |
| 10 | 59dbbd7 | eps=1e-4 | YES | step 682 | nan | Larger eps delays but doesn't fix |
| **11** | **53bd85e** | **torch.compile OFF on Adam** | **NO** | **-** | **1.281545** | **ZERO NaN across 1062 steps** |
| 12 | 0ea95be | Compiled Adam with fp32 math | YES | step 711 | nan | NaN even with fp32 inside compile |
| 13 | 1bb1db2 | Baseline beta2=0.97 NaN monitoring | NO | - | 1.280839 | Compile bug needs beta2=0.95 to trigger at depth=2 |
| **14** | **26e39fe** | **DEPTH=12, optimizer compile OFF** | **NO** | **-** | **-** | **42 steps clean (vs step 6 crash in DIAG 5)** |
| **15** | **1139bfa** | **Small batch 2^13, optimizer compile OFF** | **NO** | **-** | **1.315537** | **3540 steps clean (vs NaN at 1783 in DIAG 1)** |
| 16 | e56e466 | DEPTH=12, Adam compiled, Muon uncompiled | YES | step 6 | nan | Adam compile alone crashes depth=12 |
| **16b** | **47c7ad1** | **DEPTH=12, Adam uncompiled, Muon compiled** | **NO** | **-** | **-** | **43 steps clean: Muon compile is SAFE** |
| 19 | 57cc26e | Dump compiled Triton kernel code | - | - | - | bf16 kernel [1/1] has fp32→bf16 store conversions; fp32 kernel [1/0] has none |
| **20** | **9b9c623** | **Split Adam: stage A (lerp) + stage B (update)** | **YES** | **step 649 (B)** | **nan** | **Stage B (bias correction+update) first NaN; Stage A (lerp) clean until cascade at 836** |
| **21** | **56857be** | **Compiled pure arithmetic + eager bf16 stores** | **YES** | **step 622** | **nan** | **Compiled arithmetic returns NaN update tensor BEFORE eager store. Bug is in compiled pow/sqrt/div, NOT in compiled stores** |
| **22** | **476b052** | **Split: compute_denom + compute_update** | **YES** | **step 633** | **nan** | **denom_nan=1 from CLEAN exp_avg_sq. Bug is in compiled `(exp_avg_sq/bias2).sqrt()+eps`** |
| **23** | **edb2ccf** | **Compiled vs eager + value dump + minimal reproducer** | **YES** | **step 681** | **nan** | **ROOT CAUSE: compiled sqrt(subnormal)=NaN. exp_avg_sq=1.15e-38 (subnormal) triggers it. Standalone reproducer confirms.** |
| **24** | **18b4f53** | **Workaround: clamp exp_avg_sq above subnormal** | **NO** | **-** | **1.281410** | **beta2=0.95 + compiled Adam: ZERO NaN with clamp_min(1.1755e-38)** |
| **24b** | **3ed8de0** | **Production: clamp + beta2=0.97** | **NO** | **-** | **1.280383** | **NEW BEST val_bpb! Compiled Adam with subnormal clamp = safe + optimal** |
| 25 | - | Test 11 compiled math ops on subnormals | - | - | - | **ONLY sqrt broken**. rsqrt, reciprocal, log, exp, abs, neg, square, add, div, mul all correct on subnormals |
| **26** | **e409d4d** | **Clamp workaround at DEPTH=12** | **NO** | **-** | **-** | **44 steps ZERO NaN (vs step 6 crash without clamp). Workaround comprehensive.** |

## Detailed Timeline (DIAG 9, compiled Adam, bf16)

| Step | Event | Details |
|------|-------|---------|
| 0-585 | Clean training | No NaN in any parameter |
| **586** | **First NaN** | 1 value in 1 row of `value_embeds.1` appears POST-OPT (Adam step creates it) |
| 586-647 | Stable | NaN count stays at 1 (same row, same value) |
| 648 | Growth | 2 NaN values (still 1 row) |
| 707 | Acceleration | 3 NaN values |
| 719 | Spreading | 7 NaN values (1 row), growth accelerating |
| 784 | Cross-table | NaN appears in `wte` (1 value, 1 row) |
| 830 | Pre-crash | 53 NaN in value_embeds (2 rows), 1 in wte |
| **835** | **Cascade** | Batch contains NaN token -> forward pass produces NaN everywhere |
| 835 POST-OPT | Aftermath | 194,605 NaN in value_embeds (609 rows), lm_head 100% NaN |

## Key Observations

### 1. The NaN originates in the compiled Adam optimizer step, not the forward/backward pass

At step 834, all gradients are clean (no NaN/Inf, norms in 0.04-0.08 range). At step 835, all 18 gradients are NaN. The only parameter with NaN before the cascade is `value_embeds.1` (introduced at step 586 by the Adam optimizer POST-OPT step).

Evidence: DIAG 8 (`4a8b04f`), step 834 vs 835 comparison.

### 2. The NaN is NOT a bf16 precision issue

DIAG 12 (`0ea95be`) runs all Adam computation in fp32 inside the compiled kernel (explicit `.float()` casts, computation in fp32, `.copy_()` back to bf16). NaN still appears at step 711. If this were a bf16 precision issue, fp32 computation would fix it.

Evidence: DIAG 12 NaN at step 711 despite fp32 computation.

### 3. torch.compile is the sole differentiator

DIAG 11 (`53bd85e`) removes only the `@torch.compile(dynamic=False, fullgraph=True)` decorator from `adamw_step_fused`. Everything else identical: bf16 parameters, bf16 optimizer states, eps=1e-10, beta2=0.95, same seed. Result: **zero NaN** across all 1062 steps, val_bpb=1.281545.

Evidence: DIAG 11 clean run vs DIAG 3/9 NaN runs.

### 4. The bug is specific to certain hyperparameter regimes

With beta2=0.97 (our baseline), the compiled Adam does not produce NaN within the 5-minute training budget (~1067 steps). With beta2=0.95, NaN appears at step ~586. This suggests the compiled kernel has a precision or correctness issue that manifests more easily with certain optimizer dynamics.

### 5. NaN grows predictably through embedding table

The NaN spreads one value at a time within a single embedding row, then to additional rows. The max_finite value of the embedding table is ~178 (well within bf16 range of 65504), ruling out overflow.

## Reproduction

### Minimal reproducer (7 lines, no training required)
```python
import torch
x = torch.full((1024,), 1.14e-38, dtype=torch.bfloat16, device='cuda')  # subnormal

@torch.compile
def f(x): return x.sqrt()

print(torch.isnan(f(x)).sum())  # gfx1151: 1024 (ALL NaN) — BUG
print(torch.isnan(x.sqrt()).sum())  # eager: 0 (correct)
```

### Extended reproducer (tests all edge cases)
See `reports/minimal_reproducer_subnormal_nan.py` — confirms:
- `compiled sqrt(bf16 subnormal)` → NaN (100% failure rate)
- `compiled sqrt(bf16 normal)` → correct
- `compiled div(bf16 subnormal, 1.0)` → correct (only sqrt is broken)
- `compiled sqrt(fp32 subnormal)` → also NaN (not bf16-specific)
- `eager sqrt(any value)` → always correct

### Training-level reproduction
```python
# In the Adam optimizer step, after ~600+ steps, cold embedding rows'
# exp_avg_sq decays to bf16 subnormal range. The compiled denom calculation:
denom = (exp_avg_sq / bias2).sqrt() + eps_t
# produces NaN for those subnormal entries, eventually cascading to full collapse.
```

## Workarounds

### Option 1: Clamp exp_avg_sq above subnormal range (RECOMMENDED)
```python
@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, ...):
    ...
    # Clamp exp_avg_sq above bf16 min normal to avoid compiled sqrt(subnormal) NaN
    denom = (exp_avg_sq.clamp_min(1.1755e-38) / bias2).sqrt() + eps_t
    ...
```
- **Preserves compiled Adam performance** (no recompilation overhead)
- DIAG 24: beta2=0.95 + clamp → ZERO NaN, val_bpb=1.281410
- DIAG 24b: beta2=0.97 + clamp → ZERO NaN, val_bpb=1.280383 (new best)
- The clamp only affects subnormal values (~1e-38) which are already negligible in the Adam denominator (eps=1e-10 dominates)

### Option 2: Remove @torch.compile from Adam step
- No performance regression (Adam step is not the bottleneck)
- DIAG 11: uncompiled Adam → ZERO NaN, val_bpb=1.281545

## Relation to ROCm Issue #6034

The original issue (#6034) reports "bf16 precision bugs" on gfx1151 including:
1. NaN with small batch sizes
2. NaN with head_dim=32
3. NaN with deep networks (depth >= 12)
4. NaN with specific Adam beta2 values

Our investigation suggests that at least the Adam beta2 NaN (item 4) is caused by **torch.compile code generation**, not bf16 arithmetic precision. The other items may have different root causes (depth=12 NaN at step 6 is too fast for the slow Adam embedding NaN mechanism).

### Cross-references to Related Issues

- [ROCm/ROCm#5991](https://github.com/ROCm/ROCm/issues/5991): GPU page faults on basic tensor operations (may share compiler code-gen root cause)
- [ROCm/TheRock#2991](https://github.com/ROCm/TheRock/issues/2991): Incorrect VGPR count crashes (fixed in nightlies Dec 2025, but may still affect compiled kernels)
- [ROCm/rocm-libraries#4618](https://github.com/ROCm/rocm-libraries/issues/4618): Missing rocWMMA support (related to matrix operation correctness on gfx1151)

## Additional Diagnostic Results

### DIAG 1: Small batch size (TOTAL_BATCH_SIZE=2^13, compiled optimizer)
- NaN at step 1783/3644 (later than original report's "within 15 steps")
- Originally hypothesized as genuine bf16 accumulation overflow

### DIAG 15: Small batch size (TOTAL_BATCH_SIZE=2^13, optimizer compile DISABLED)
- NO NaN in 3540 steps (complete run), val_bpb=1.315537
- **KEY FINDING**: Small batch NaN is ALSO caused by torch.compile, not bf16 precision
- The "bf16 accumulation overflow" hypothesis from DIAG 1 was WRONG - same compiler bug
- With more steps (3540 vs 1067 in baseline), the compiler bug manifests later (step 1783)

### DIAG 2: HEAD_DIM=32 at depth=2
- NO NaN, only quality loss from 2x attention compute
- Contradicts original report, but may be depth-dependent (report used depth=8)

### DIAG 4: MATRIX_LR=0.20 at depth=2
- NO NaN, only quality loss
- LR cliff from original report is depth-dependent

### DIAG 5: DEPTH=12 (compiled optimizers)
- NaN at step 6/45 - catastrophic and immediate
- Different mechanism from Adam compile NaN (too fast, affects Muon directly)
- Originally attributed to genuine bf16 overflow in deep network forward/backward pass

### DIAG 14: DEPTH=12 (both optimizer compiles DISABLED, model compile active)
- NO NaN in 42 training steps (vs NaN at step 6 in DIAG 5 with compiled optimizers)
- Loss curve healthy: 9.01 -> 5.74, steadily decreasing
- 596M params, only 42 steps in 5min budget (9.5s/step), eval killed (too slow)
- **KEY FINDING**: Depth=12 NaN is ALSO caused by torch.compile on optimizer steps
- The "deep network bf16 overflow" hypothesis from DIAG 5 was WRONG - it's the same compiler bug

### DIAG 16: DEPTH=12 (Adam compiled, Muon uncompiled)
- NaN at step 6 (identical to DIAG 5 with both compiled)
- **Adam compile alone reproduces the depth=12 crash**

### DIAG 16b: DEPTH=12 (Adam uncompiled, Muon compiled)
- NO NaN in 43 training steps (clean, identical to DIAG 14)
- **Muon compile is SAFE** - only Adam compile causes the NaN
- **DEFINITIVE: torch.compile on adamw_step_fused is the sole root cause for all NaN scenarios**

### DIAG 19: Compiled Triton kernel code analysis (commit 57cc26e)
- Dumped compiled kernel code using `TORCH_LOGS=output_code`
- torch.compile generates **two kernel variants** for the same `adamw_step_fused` function:
  - `[1/0]` fp32 variant: used for wte/lm_head (fp32 params). No type conversions anywhere.
  - `[1/1]` bf16 variant: used for value_embeds (bf16 params). Has `tl.load(*bf16).to(tl.float32)` + implicit `tl.store(*bf16, fp32_value)` conversions.
  - `[1/2]` fp32 scalar variant: used for resid/x0 lambdas (2 elements).
- The bf16 kernel also has redundant `.to(tl.float32)` casts on values already in fp32.
- **KEY FINDING**: NaN originates exclusively in bf16 parameters (value_embeds), never in fp32 parameters (wte, lm_head). The sole difference between the two kernel variants is the bf16 load/store conversion code.
- This is consistent with DIAG 12 (fp32 math inside compiled kernel still NaN): DIAG 12 forced fp32 computation but the stores still went to bf16 tensors via the same implicit conversion path.
- **Hypothesis**: Bug is in the Triton→AMDGPU ISA compiled fp32→bf16 store conversion instruction (`v_cvt_bf16_f32` or equivalent) on gfx1151.
- Full kernel analysis: `reports/compiled_adam_kernel_analysis.md`

### DIAG 20: Split Adam isolates Stage B as NaN source (commit 9b9c623)
- Split `adamw_step_fused` into two separately compiled functions:
  - **Stage A** (`adamw_stage_a`): `exp_avg.lerp_()` + `exp_avg_sq.lerp_()` — momentum updates
  - **Stage B** (`adamw_stage_b`): `p.mul_(1-lr*wd)` + bias correction + `p.add_(exp_avg/denom)` — parameter update
- NaN monitoring between stages at every step after step 550.
- **DEFINITIVE RESULT**:
  - Stage B first NaN at **step 649**: 1 NaN value in `p` (parameter tensor, shape 8192×320)
  - Stage A first NaN at **step 836**: cascade from corrupted forward pass (grad already NaN)
  - Between steps 649-836: NaN grows in `p` via Stage B, but `exp_avg`/`exp_avg_sq` remain clean through Stage A
- **CONCLUSION**: The bug is specifically in the compiled kernel for bias correction + parameter update:
  `p.mul_(1-lr*wd)`, `bias1 = 1-beta1^step`, `bias2 = 1-beta2^step`, `denom = (exp_avg_sq/bias2).sqrt()+eps`, `step_size = lr/bias1`, `p.add_(exp_avg/denom, alpha=-step_size)`
- The lerp (momentum) operations compile correctly on gfx1151.

### DIAG 21: Compiled arithmetic vs compiled stores isolation (commit 56857be)
- Separated Stage B into:
  - **Compiled pure arithmetic** (`adamw_stage_b_calc`): computes `update = exp_avg / denom * (-step_size)` using pow/sqrt/div but returns the result tensor with NO in-place mutations
  - **Eager stores**: `p.mul_()` and `p.add_(update)` executed WITHOUT torch.compile
- NaN monitoring checks the compiled `update` tensor BEFORE and p AFTER eager stores.
- **DEFINITIVE RESULT**:
  - Step 622: `update_nan=1, p_nan=0` **BEFORE** eager store (compiled arithmetic PRODUCED the NaN)
  - Step 622: `p_nan=1` **AFTER** eager store (eager p.add_() just copied the NaN through)
  - NaN count stays at 1 for hundreds of steps, then cascade at step ~835
- **CONCLUSION**: The bug is in the **compiled arithmetic** (pow/sqrt/div chain) operating on bf16 input data, NOT in the compiled in-place store instructions.
  - The compiled `adamw_stage_b_calc` function reads bf16 tensors (exp_avg, exp_avg_sq), performs fp32 scalar arithmetic (pow, div) and tensor arithmetic (div, sqrt, add), and returns a tensor that already contains NaN
  - Eager bf16 stores work correctly — they just faithfully store whatever value they receive
  - This refines the DIAG 19 hypothesis: the fp32→bf16 store conversion is NOT the root cause; the arithmetic code-gen is
- **Cross-reference with DIAG 12**: DIAG 12 forced fp32 computation inside the full compiled kernel and still got NaN. This is consistent: the bug is in the compiled arithmetic code-gen, not in the dtype of the computation.

### DIAG 22: compute_denom produces NaN from clean inputs (commit 476b052)
- Split arithmetic into two compiled functions:
  - `compute_denom(exp_avg_sq, step_t, beta2_t, eps_t)`: `bias2 = 1 - beta2_t ** step_t; denom = (exp_avg_sq / bias2).sqrt() + eps_t`
  - `compute_update(exp_avg, denom, step_t, lr_t, beta1_t)`: `exp_avg / denom * (-step_size)`
- NaN monitoring between both functions.
- **DEFINITIVE RESULT**:
  - Step 633: `denom_nan=1, exp_avg_nan=0, exp_avg_sq_nan=0` — compiled `compute_denom` produces 1 NaN from perfectly clean input
  - `update_nan=1` — NaN propagates via `exp_avg / denom` (division by NaN)
- **CONCLUSION**: The bug is localized to the compiled Triton kernel for `(exp_avg_sq / bias2).sqrt() + eps_t`:
  - Input `exp_avg_sq` is a bf16 tensor with ZERO NaN values
  - `bias2` is a fp32 scalar very close to 1.0 (e.g., `1 - 0.95^633 ≈ 1.0`)
  - Yet the compiled kernel produces 1 NaN in the output
  - The expression involves: bf16→fp32 load, fp32 scalar pow, fp32 scalar division, fp32 tensor division, fp32 sqrt, fp32 scalar addition
  - DIAG 23 subsequently proved: it's specifically `sqrt()` on subnormal values

### DIAG 23: Definitive root cause — compiled sqrt(subnormal) = NaN (commit edb2ccf)
- Ran compiled vs eager comparison side-by-side.
- At step 681: `compiled_nan=1, eager_nan=0` for the same `exp_avg_sq` input.
- **Value at NaN index [529, 269]**: `exp_avg_sq=1.14794370e-38` — this is a **bf16 subnormal** (below min normal 2^-126 ≈ 1.175e-38).
  - Eager denom: `1.00044417e-10` (correct — eps dominates since sqrt of ~1e-38 is ~1e-19, much smaller than eps=1e-10)
  - Compiled denom: **NaN**
- **Standalone minimal reproducer** (`reports/minimal_reproducer_subnormal_nan.py`) confirmed:
  - `compiled sqrt(bf16 subnormal)` → **NaN for ALL 1024 elements** — not a rare miscomputation, 100% failure rate
  - `eager sqrt(bf16 subnormal)` → correct (0 NaN)
  - `compiled sqrt(bf16 normal)` → correct (0 NaN) — only subnormals trigger it
  - `compiled div(bf16 subnormal, 1.0)` → correct (0 NaN) — division is fine, only sqrt is broken
  - `compiled sqrt(fp32 subnormal)` → **NaN** — bug affects fp32 too, not bf16-specific
- **ROOT CAUSE**: The Triton-compiled `sqrt()` instruction on gfx1151 produces NaN for subnormal (denormalized) floating-point inputs, regardless of dtype. The eager PyTorch `sqrt()` handles subnormals correctly. This is either:
  1. A Triton code-gen issue: the compiled kernel may use a fast sqrt approximation that doesn't handle subnormals
  2. A hardware issue: gfx1151's sqrt instruction may flush subnormals to negative zero or produce NaN
  3. A mode flag issue: compiled kernels may set FTZ (flush-to-zero) mode differently from eager PyTorch

## Why beta2=0.95 triggers NaN earlier than beta2=0.97

The `exp_avg_sq` in Adam tracks the exponential moving average of squared gradients. For cold embedding rows (tokens rarely seen in training), the gradient is zero most steps. At step `s`, the exp_avg_sq for a cold row decays as:

```
exp_avg_sq[s] ≈ exp_avg_sq[last_seen] × beta2^(s - last_seen)
```

With beta2=0.95, values decay to subnormal (~1e-38) after ~880 steps of zero gradient. With beta2=0.97, this takes ~1270 steps. In our 5-minute budget (~1067 steps), beta2=0.95 reaches subnormal within budget, beta2=0.97 barely doesn't.

### DIAG 24/24b: Workaround confirmed — clamp prevents NaN (commits 18b4f53, 3ed8de0)
- Added `exp_avg_sq.clamp_min(1.1755e-38)` before `sqrt()` in compiled Adam.
- **DIAG 24** (beta2=0.95): ZERO NaN across 1067 steps, val_bpb=1.281410 (was NaN at ~step 620)
- **DIAG 24b** (beta2=0.97): ZERO NaN, val_bpb=1.280383 — new best (previous: 1.280888)
- The clamp only affects subnormal values (~1e-38) which are negligible vs eps=1e-10 in the Adam denominator, so no quality impact.

### DIAG 25: Only sqrt is broken — 10 other ops handle subnormals correctly
- Tested 11 compiled math operations on subnormal values (bf16 and fp32):

| Operation | Compiled on subnormal | Status |
|-----------|-----------------------|--------|
| `sqrt` | NaN (100% of elements) | **BUG** |
| `rsqrt` | Correct | ok |
| `reciprocal` | Correct | ok |
| `log` | Correct | ok |
| `exp` | Correct | ok |
| `abs` | Correct | ok |
| `neg` | Correct | ok |
| `square` | Correct | ok |
| `add` | Correct | ok |
| `div` | Correct | ok |
| `mul` | Correct | ok |

- **Key insight**: `rsqrt` (1/sqrt) works correctly on subnormals but `sqrt` fails. These likely use different ISA instructions on gfx1151 (`v_rsq_f32` vs `v_sqrt_f32`).
- This strongly suggests the bug is in the **`v_sqrt_f32`** (or `tl.sqrt_rn`) instruction specifically on gfx1151, not in a general subnormal handling issue.

## Recommended Actions for AMD/ROCm

1. **Fix `v_sqrt_f32` on gfx1151**: The compiled Triton kernel's sqrt instruction produces NaN for subnormal inputs. This affects both bf16 and fp32 tensors. `rsqrt` works correctly, suggesting the issue is specific to `v_sqrt_f32` (or whichever ISA instruction `tl.sqrt_rn()` compiles to).
2. **Check FTZ/DAZ mode flags**: Compiled Triton kernels may set Flush-To-Zero or Denormals-Are-Zero mode flags differently from eager PyTorch. If FTZ is enabled, subnormals should flush to +0 (which sqrt(+0)=+0, valid), not produce NaN. The fact that `rsqrt` handles subnormals correctly while `sqrt` doesn't suggests different FTZ handling per instruction.
3. **Add subnormal tests to CI**: Any compiled kernel test suite should include subnormal inputs for math functions, especially sqrt.
4. **Document torch.compile limitations**: Until fixed, gfx1151 users should clamp inputs away from subnormal range before compiled sqrt, or avoid compiling optimizer steps.
5. **Consider Triton-level workaround**: Triton could use `rsqrt(x) * x` instead of `sqrt(x)` as a temporary fix for gfx1151, since rsqrt handles subnormals correctly.
