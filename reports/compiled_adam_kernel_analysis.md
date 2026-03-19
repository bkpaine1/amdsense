# Compiled Adam Kernel Analysis for gfx1151 NaN Bug

**Date**: 2026-03-19
**Source**: DIAG 19 (commit 57cc26e)
**Command**: `TORCH_LOGS=output_code python train.py`

## Summary

The `torch.compile(adamw_step_fused)` generates a single fused Triton kernel
(`triton_poi_fused_add_copy__div_lerp_mul_neg_pow_rsub_sqrt_0`) with **two dtype
variants** for the same embedding shape (8192 x 320 = 2,621,440 elements):

| Graph ID | Tensor dtype | Used for | NaN observed? |
|----------|-------------|----------|--------------|
| `[1/0]` | `*fp32` | wte, lm_head (fp32 params) | Not first; NaN cascades from value_embeds |
| `[1/1]` | `*bf16` | value_embeds (bf16 params) | **YES - first NaN at step 586** |
| `[1/2]` | `*fp32` | Per-layer scalars (2 elements) | No |

## Key Observation: bf16 variant has suspicious conversion pattern

The bf16 kernel [1/1] loads bf16 values and converts to fp32 for computation,
then stores fp32 results back to bf16 pointers:

```python
# Load bf16 -> explicit fp32 conversion
tmp8 = tl.load(in_ptr1 + (x0), None).to(tl.float32)   # grad (bf16 -> fp32)
tmp10 = tl.load(in_ptr2 + (x0), None).to(tl.float32)   # exp_avg (bf16 -> fp32)
tmp25 = tl.load(in_ptr4 + (x0), None).to(tl.float32)   # exp_avg_sq (bf16 -> fp32)
tmp42 = tl.load(in_ptr7 + (x0), None).to(tl.float32)   # p (bf16 -> fp32)

# Redundant fp32->fp32 casts (already fp32 from load conversion above)
tmp9 = tmp8.to(tl.float32)    # REDUNDANT
tmp11 = tmp10.to(tl.float32)  # REDUNDANT
tmp16 = tmp15.to(tl.float32)  # REDUNDANT - new exp_avg already fp32
tmp31 = tmp30.to(tl.float32)  # REDUNDANT - new exp_avg_sq already fp32

# Store fp32 -> implicit bf16 conversion
tl.store(out_ptr4 + (x0), tmp16, None)  # out_ptr4 is *bf16 - IMPLICIT CONVERSION
tl.store(out_ptr5 + (x0), tmp31, None)  # out_ptr5 is *bf16 - IMPLICIT CONVERSION
tl.store(out_ptr6 + (x0), tmp55, None)  # out_ptr6 is *bf16 - IMPLICIT CONVERSION
```

The fp32 kernel [1/0] has **NO conversion operations** - all loads and stores are fp32:

```python
# Direct fp32 loads - no conversion needed
tmp8 = tl.load(in_ptr1 + (x0), None)   # grad (fp32 direct)
tmp9 = tl.load(in_ptr2 + (x0), None)   # exp_avg (fp32 direct)
tmp21 = tl.load(in_ptr4 + (x0), None)  # exp_avg_sq (fp32 direct)
tmp34 = tl.load(in_ptr7 + (x0), None)  # p (fp32 direct)

# Direct fp32 stores - no conversion needed
tl.store(out_ptr4 + (x0), tmp13, None)  # fp32 -> fp32
tl.store(out_ptr5 + (x0), tmp25, None)  # fp32 -> fp32
tl.store(out_ptr6 + (x0), tmp45, None)  # fp32 -> fp32
```

## Hypothesis: Bug in fp32->bf16 store conversion on gfx1151

The NaN appears **only in bf16 parameters** (value_embeds), never originating from
fp32 parameters (wte, lm_head). The sole difference between the fp32 and bf16 kernel
variants is the load/store conversion operations.

This is consistent with DIAG 12 (fp32 math inside compiled kernel still produces NaN):
- DIAG 12 forced fp32 COMPUTATION but the stores still went to bf16 tensors
- If the bug is in the fp32->bf16 store conversion, fp32 computation wouldn't help

**Suspected code-gen issue**: The Triton compiler's fp32->bf16 store conversion
instruction (`v_cvt_bf16_f32` or similar) may produce incorrect results for specific
values on gfx1151. This would explain:
- NaN in bf16 parameters but not fp32 parameters
- NaN even with fp32 computation inside compiled kernel
- No NaN when the function is uncompiled (PyTorch's eager bf16 conversion is correct)

## Full bf16 Kernel Code [1/1]

Cleaned from TORCH_LOGS output (removed log prefix):

```python
@triton_heuristics.pointwise(
    size_hints={'x': 4194304},
    triton_meta={
        'signature': {
            'in_ptr0': 'fp32',   # beta1_t (scalar)
            'in_ptr1': '*bf16',  # grad
            'in_ptr2': '*bf16',  # exp_avg
            'in_ptr3': 'fp32',   # beta2_t (scalar)
            'in_ptr4': '*bf16',  # exp_avg_sq
            'in_ptr5': 'fp32',   # step_t (scalar)
            'in_ptr6': 'fp32',   # eps_t (scalar)
            'in_ptr7': '*bf16',  # p (parameter)
            'in_ptr8': 'fp32',   # lr_t (scalar)
            'in_ptr9': 'fp32',   # wd_t (scalar)
            'out_ptr4': '*bf16', # new exp_avg (aliased with in_ptr2)
            'out_ptr5': '*bf16', # new exp_avg_sq (aliased with in_ptr4)
            'out_ptr6': '*bf16', # new p (aliased with in_ptr7)
            'xnumel': 'i32',
            'XBLOCK': 'constexpr'
        },
        'device': DeviceProperties(type='hip', index=0, cc='gfx1151',
                                    multi_processor_count=20, warp_size=32),
        'enable_fp_fusion': True
    },
    mutated_arg_names=['in_ptr2', 'in_ptr4', 'in_ptr7',
                       'out_ptr4', 'out_ptr5', 'out_ptr6']
)
@triton.jit
def triton_poi_fused_add_copy__div_lerp_mul_neg_pow_rsub_sqrt_0(
    in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6,
    in_ptr7, in_ptr8, in_ptr9, out_ptr4, out_ptr5, out_ptr6,
    xnumel, XBLOCK: tl.constexpr
):
    xnumel = 2621440  # 8192 * 320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex

    # Load scalars
    tmp0 = in_ptr0      # beta1_t
    tmp17 = in_ptr3      # beta2_t
    tmp32 = in_ptr5      # step_t
    tmp38 = in_ptr6      # eps_t
    tmp43 = in_ptr8      # lr_t
    tmp44 = in_ptr9      # wd_t

    # Load bf16 tensors -> convert to fp32
    tmp8 = tl.load(in_ptr1 + (x0), None).to(tl.float32)   # grad
    tmp10 = tl.load(in_ptr2 + (x0), None).to(tl.float32)  # exp_avg
    tmp25 = tl.load(in_ptr4 + (x0), None).to(tl.float32)  # exp_avg_sq
    tmp42 = tl.load(in_ptr7 + (x0), None).to(tl.float32)  # p

    # ---- exp_avg.lerp_(grad, 1 - beta1_t) ----
    tmp1 = 1.0
    tmp2 = tmp1 - tmp0          # 1 - beta1 = weight
    tmp3 = tl_math.abs(tmp2)
    tmp4 = 0.5
    tmp5 = tmp3 >= tmp4         # |weight| >= 0.5?
    tmp6 = tmp2 - tmp1          # weight - 1
    tmp7 = tl.where(tmp5, tmp6, tmp2)  # adjusted weight
    tmp9 = tmp8.to(tl.float32)         # REDUNDANT: grad already fp32
    tmp11 = tmp10.to(tl.float32)       # REDUNDANT: exp_avg already fp32
    tmp12 = tmp9 - tmp11               # grad - exp_avg
    tmp13 = tmp7 * tmp12               # weight * diff
    tmp14 = tl.where(tmp5, tmp9, tmp11)  # base value
    tmp15 = tmp13 + tmp14              # new exp_avg (fp32)
    tmp16 = tmp15.to(tl.float32)       # REDUNDANT: already fp32

    # ---- exp_avg_sq.lerp_(grad^2, 1 - beta2_t) ----
    tmp18 = tmp1 - tmp17        # 1 - beta2
    tmp19 = tl_math.abs(tmp18)
    tmp20 = tmp19 >= tmp4       # |weight| >= 0.5?
    tmp21 = tmp18 - tmp1        # weight - 1
    tmp22 = tl.where(tmp20, tmp21, tmp18)
    tmp23 = tmp8 * tmp8         # grad^2 (bf16 * bf16 -> fp32 result)
    tmp24 = tmp23.to(tl.float32)  # REDUNDANT
    tmp26 = tmp25.to(tl.float32)  # REDUNDANT: exp_avg_sq already fp32
    tmp27 = tmp24 - tmp26       # grad^2 - exp_avg_sq
    tmp28 = tmp22 * tmp27       # weight * diff
    tmp29 = tl.where(tmp20, tmp24, tmp26)
    tmp30 = tmp28 + tmp29       # new exp_avg_sq (fp32)
    tmp31 = tmp30.to(tl.float32)  # REDUNDANT: already fp32

    # ---- Bias correction and update ----
    tmp33 = libdevice.pow(tmp17, tmp32)   # beta2^step
    tmp34 = tmp1 - tmp33                  # bias2
    tmp35 = tmp34.to(tl.float32)          # REDUNDANT
    tmp36 = (tmp31 / tmp35)               # exp_avg_sq_corrected
    tmp37 = tl.sqrt_rn(tmp36)             # sqrt
    tmp39 = tmp38.to(tl.float32)          # REDUNDANT: eps already fp32 scalar
    tmp40 = tmp37 + tmp39                 # denom = sqrt + eps
    tmp41 = (tmp16 / tmp40)               # exp_avg / denom

    # ---- Weight decay ----
    tmp45 = tmp43 * tmp44                 # lr * wd
    tmp46 = tmp1 - tmp45                  # 1 - lr*wd
    tmp47 = tmp46.to(tl.float32)          # REDUNDANT
    tmp48 = tmp42 * tmp47                 # p * (1 - lr*wd)

    # ---- Step size ----
    tmp49 = libdevice.pow(tmp0, tmp32)    # beta1^step
    tmp50 = tmp1 - tmp49                  # bias1
    tmp51 = (tmp43 / tmp50)               # lr / bias1 = step_size
    tmp52 = -tmp51                        # -step_size
    tmp53 = tmp52.to(tl.float32)          # REDUNDANT
    tmp54 = tmp41 * tmp53                 # update = ratio * (-step_size)

    # ---- Final parameter update ----
    tmp55 = tmp48 + tmp54                 # p_new = p*(1-lr*wd) + update

    # Store fp32 -> bf16 (IMPLICIT CONVERSION on gfx1151)
    tl.store(out_ptr4 + (x0), tmp16, None)   # exp_avg
    tl.store(out_ptr5 + (x0), tmp31, None)   # exp_avg_sq
    tl.store(out_ptr6 + (x0), tmp55, None)   # p
```

## Full fp32 Kernel Code [1/0]

```python
# Same heuristics, but signature uses *fp32 for all tensor pointers
# Key difference: NO .to(tl.float32) conversions anywhere

@triton.jit
def triton_poi_fused_add_copy__div_lerp_mul_neg_pow_rsub_sqrt_0(
    in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6,
    in_ptr7, in_ptr8, in_ptr9, out_ptr4, out_ptr5, out_ptr6,
    xnumel, XBLOCK: tl.constexpr
):
    xnumel = 2621440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = in_ptr0
    tmp8 = tl.load(in_ptr1 + (x0), None)    # grad (fp32 direct)
    tmp9 = tl.load(in_ptr2 + (x0), None)    # exp_avg (fp32 direct)
    tmp14 = in_ptr3
    tmp21 = tl.load(in_ptr4 + (x0), None)   # exp_avg_sq (fp32 direct)
    tmp26 = in_ptr5
    tmp31 = in_ptr6
    tmp34 = tl.load(in_ptr7 + (x0), None)   # p (fp32 direct)
    tmp35 = in_ptr8
    tmp36 = in_ptr9
    # ... identical math, no type conversions ...
    tl.store(out_ptr4 + (x0), tmp13, None)   # fp32 direct store
    tl.store(out_ptr5 + (x0), tmp25, None)   # fp32 direct store
    tl.store(out_ptr6 + (x0), tmp45, None)   # fp32 direct store
```

## Diagnostic Implication

The Triton IR (Python-level kernel code) appears mathematically correct for both
variants. The bug must be in one of:

1. **Triton -> AMDGPU ISA compilation for bf16 load/store on gfx1151**
   - `tl.load(*bf16).to(tl.float32)` might use a faulty bf16->fp32 instruction
   - `tl.store(*bf16, fp32_value)` might use a faulty fp32->bf16 instruction
   - The redundant `.to(tl.float32)` casts might confuse the compiler

2. **FMA fusion (`enable_fp_fusion: True`) with bf16 conversions**
   - Fused multiply-add across bf16->fp32 boundaries might produce incorrect results

3. **VGPR allocation for bf16 values on gfx1151**
   - Related to TheRock#2991 (incorrect VGPR count, fixed Dec 2025)
   - But may still affect compiled Triton kernels differently

## Next Steps

- DIAG 20: Split into separate compiled functions per operation to isolate which
  op (lerp, pow, sqrt, store) triggers the NaN
- Test with `enable_fp_fusion: False` (requires patching Inductor meta)
- Dump AMDGPU ISA (assembly) for the bf16 variant to inspect actual instructions
