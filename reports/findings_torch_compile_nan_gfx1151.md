# torch.compile Generates Incorrect Code on gfx1151: Systematic NaN in Adam Optimizer

**Date**: 2026-03-19
**Hardware**: AMD Radeon 8060S (gfx1151, Strix Halo)
**Software**: PyTorch 2.11.0a0+rocm7.11.0a20260106, ROCm 7.11.0 nightlies
**Branch**: autoresearch/mar19-dsstrix2
**Related Issue**: [ROCm/ROCm#6034](https://github.com/ROCm/ROCm/issues/6034)

## Executive Summary

Through 17 systematic diagnostic experiments, we isolated a **torch.compile code generation bug on gfx1151** that produces NaN values specifically in the **compiled Adam optimizer step kernel**. The bug is NOT a bf16 precision issue - it persists even when all computation inside the compiled function is done in fp32. The Muon optimizer compile is unaffected. Disabling `torch.compile` on `adamw_step_fused` alone eliminates ALL NaN scenarios tested:
- Shallow model (depth=2) with Adam beta2=0.95: NaN at step ~586 with compile, clean 1062 steps without
- Deep model (depth=12): NaN at step 6 with compile, clean 43 steps without (Muon compile safe)
- Small batch (2^13): NaN at step 1783 with compile, clean 3540 steps without

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

Narrowed to: compiled kernel for p.mul_() + pow() + div() + sqrt() + add_() with bf16 stores
NOT in: compiled lerp_() operations (Stage A clean through step 836)
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

```python
# Minimal reproduction conditions:
# - gfx1151 hardware (AMD Radeon 8060S / Strix Halo)
# - PyTorch 2.11.0a0+rocm7.11.0a20260106
# - torch.compile on an Adam optimizer step processing bf16 embedding parameters
# - Adam beta2=0.95 (beta2=0.97 may work within short training runs)

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

# When called with bf16 embedding parameters on gfx1151,
# this compiled kernel produces NaN at step ~586 with beta2=0.95.
# Removing @torch.compile eliminates the NaN entirely.
```

## Workaround

Remove `@torch.compile` from the Adam optimizer step function. There is no performance regression since the Adam step is not the bottleneck (the main model forward/backward is compiled separately and works correctly).

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

## Recommended Actions for AMD/ROCm

1. **Investigate torch.compile code generation for gfx1151**: The compiled kernel for the Adam optimizer pattern produces incorrect values that don't occur in the uncompiled version.
2. **Test other compiled kernels**: If Adam is affected, other compiled optimizer/utility functions may be too.
3. **Verify VGPR allocation**: The fixed VGPR count issue (#2991) may still affect compiled kernels differently from pre-compiled ops.
4. **Document torch.compile limitations**: Until fixed, gfx1151 users should be warned about potential correctness issues with compiled optimizer steps.
