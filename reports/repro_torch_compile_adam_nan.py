"""
Reproducer: torch.compile on AdamW step produces NaN on gfx1151

Hardware: AMD Radeon 8060S (gfx1151, Strix Halo)
Software: PyTorch 2.11.0a0+rocm7.11.0a20260106 (ROCm 7.11.0 nightlies)

NOTE: A simplified model alone does not trigger the bug. The NaN requires
realistic training dynamics with non-uniform gradient statistics across
embedding rows (common vs. rare tokens). Use the full training script
(train.py) with these changes to reproduce:

  1. Set ADAM_BETAS = (0.65, 0.95)   # beta2=0.95 triggers NaN faster
  2. Run: python train.py
  3. NaN appears at step ~586 in value_embeds, cascades at step ~835

To confirm it's the compile bug:
  1. Comment out @torch.compile on adamw_step_fused
  2. Run: python train.py
  3. Training completes 1062 steps with ZERO NaN

Evidence (17 systematic experiments on gfx1151):
  - Compiled Adam + beta2=0.95: NaN at step 586 (DIAG 9)
  - Uncompiled Adam + beta2=0.95: 1062 steps clean (DIAG 11)
  - Compiled Adam + fp32 math: NaN at step 711 (DIAG 12, not bf16)
  - Compiled Adam + depth=12: NaN at step 6 (DIAG 5/16)
  - Uncompiled Adam + depth=12: 43 steps clean (DIAG 14/16b)
  - Compiled Adam + batch=2^13: NaN at step 1783 (DIAG 1)
  - Uncompiled Adam + batch=2^13: 3540 steps clean (DIAG 15)
  - Compiled Muon + uncompiled Adam: always clean (DIAG 16b)

The bug is specific to torch.compile code generation for the Adam step
pattern on gfx1151. The Muon optimizer compile and model forward/backward
compile work correctly.

See reports/findings_torch_compile_nan_gfx1151.md for full analysis.
"""

# The Adam step function that triggers the bug when compiled on gfx1151:
#
# @torch.compile(dynamic=False, fullgraph=True)  # <- THIS CAUSES NaN
# def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t,
#                      beta1_t, beta2_t, eps_t, wd_t):
#     p.mul_(1 - lr_t * wd_t)
#     exp_avg.lerp_(grad, 1 - beta1_t)
#     exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
#     bias1 = 1 - beta1_t ** step_t
#     bias2 = 1 - beta2_t ** step_t
#     denom = (exp_avg_sq / bias2).sqrt() + eps_t
#     step_size = lr_t / bias1
#     p.add_(exp_avg / denom, alpha=-step_size)
#
# When called with:
#   - bf16 embedding parameters (8192 x 320)
#   - bf16 optimizer states
#   - beta2=0.95, eps=1e-10
#   - On gfx1151 hardware
#
# The compiled kernel produces NaN in specific embedding rows starting
# around step 500-700. Removing @torch.compile eliminates the NaN entirely.
