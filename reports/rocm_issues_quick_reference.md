# ROCm gfx1151 Issues - Quick Reference

**Generated**: 2026-03-19

## Critical Issues (OPEN)

| Issue | Title | Key Problem | Workaround |
|-------|-------|-------------|------------|
| [#6034](https://github.com/ROCm/ROCm/issues/6034) | Strix Halo gfx1151: 93 ML experiments, 5 critical bf16 bugs | bf16 crashes at small batch/head_dim=32, depth≥12 instability, 19x AOTriton speedup undocumented | Use `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`, avoid small batches/head_dim=32 with bf16 |
| [#5991](https://github.com/ROCm/ROCm/issues/5991) | GPU page fault - basic tensor operations hang | GFX Hub page faults on simple PyTorch ops | `amdgpu.cwsr_enable=0` kernel flag |
| [#6022](https://github.com/ROCm/ROCm/issues/6022) | WSL2 librocdxg fails to map Dedicated VRAM | ROCm pool limited by .wslconfig, not GPU VRAM | Use native Linux/Windows instead of WSL2 |
| [#5339](https://github.com/ROCm/ROCm/issues/5339) | Confusing rocm support for gfx1151 | Conflicting docs about gfx1151 support status | Use TheRock repository for bleeding-edge support |
| [#3128](https://github.com/ROCm/TheRock/issues/3128) | Better performance with 1GB than 4GB memory | Counter-intuitive perf degradation | Test different dedicated memory allocations |
| [#4618](https://github.com/ROCm/rocm-libraries/issues/4618) | Missing rocWMMA support for gfx1151 | Compilation fails with flash attention | Build rocWMMA from source; ROCm 7.11 packages missing rocwmma.hpp |

## Recently Resolved Issues (CLOSED)

| Issue | Title | Resolution | Status |
|-------|-------|------------|--------|
| [#5724](https://github.com/ROCm/ROCm/issues/5724) | MES 0x83 firmware causing GPU hang | Firmware reverted upstream; use amdgpu-dkms-firmware + cwsr_enable=0 | CLOSED |
| [#5890](https://github.com/ROCm/ROCm/issues/5890) | Page fault under rocm7.2 | Use Ubuntu OEM kernel with `--no-dkms` install | CLOSED |
| [#2991](https://github.com/ROCm/TheRock/issues/2991) | Incorrect VGPR count crashes | Fixed in nightlies Dec 19, 2025; ROCm 7.2/7.11; OEM kernel 1018+ | CLOSED |
| [#5824](https://github.com/ROCm/ROCm/issues/5824) | Memory access fault Strix Halo | linux-firmware-20260110-1 resolved | CLOSED |
| [#5853](https://github.com/ROCm/ROCm/issues/5853) | Segfault torch nightly VRAM access | Use TheRock builds instead of PyTorch nightlies | CLOSED |
| [#4809](https://github.com/ROCm/ROCm/issues/4809) | WSL2 not detect 8060S | Radeon 8060S APUs not supported on ROCm WSL2 | CLOSED |
| [#3874](https://github.com/ROCm/hip/issues/3874) | Windows H2D memcpy crash | Fixed in later builds | CLOSED |
| [#5696](https://github.com/ROCm/ROCm/issues/5696) | ROCm reports wrong GFX version | User error: HSA_OVERRIDE_GFX_VERSION env var | CLOSED |
| [#2229](https://github.com/ROCm/TheRock/issues/2229) | AOTriton cross-attention fails gfx1103 | Fixed in torch-2.10.0a0+rocm7.11.0a20251211 | CLOSED |

## Essential Workarounds

### Kernel Configuration
```bash
# Add to kernel boot parameters
amdgpu.cwsr_enable=0
```

### Installation Method
```bash
# Use OEM kernel, not DKMS
amdgpu-install -y --usecase=rocm --no-dkms
```

### Firmware
```bash
# Stay on older firmware (MES 0x80)
# Use amdgpu-dkms-firmware package instead of upstream linux-firmware
```

### PyTorch Source
```bash
# Use TheRock repository builds instead of PyTorch official nightlies
# More reliable gfx1151 support
```

### Performance Optimization
```bash
# Enable undocumented 19x attention speedup
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
```

## System Requirements

### Working Configurations

**Linux (Best Support)**:
- Ubuntu 24.04 with OEM kernel 1018+
- linux-firmware-20260110-1 or later
- ROCm 7.2 or 7.11 (from TheRock for latest)
- PyTorch from TheRock nightlies
- Kernel flag: `amdgpu.cwsr_enable=0`

**Windows (Limited)**:
- ROCm 6.4.4 preview or TheRock 7.x
- Native Windows only (WSL2 not supported for 8060S)
- Some memory copy issues in earlier builds

**WSL2**:
- **NOT SUPPORTED** for Radeon 8060S APUs
- VRAM mapping broken
- Use native Linux or Windows instead

## Known Precision Issues (from #6034, corrected by DIAG 1-18)

**IMPORTANT**: Our 17 systematic diagnostic experiments (DIAG 1-18) proved that most "bf16 precision bugs" from #6034 are actually **torch.compile code generation bugs on gfx1151**. See `findings_torch_compile_nan_gfx1151.md` for full evidence.

| Issue | Original Hypothesis | Actual Root Cause (per our DIAGs) | Workaround |
|-------|-------------------|----------------------------------|------------|
| Small batch NaN | bf16 accumulation overflow | **torch.compile on adamw_step_fused** (DIAG 1 vs 15) | Remove @torch.compile from Adam step |
| Deep network NaN (depth≥12) | bf16 overflow in deep forward pass | **torch.compile on adamw_step_fused** (DIAG 5 vs 14, 16 vs 16b) | Remove @torch.compile from Adam step |
| Adam beta2=0.95 NaN | bf16 precision in optimizer | **torch.compile on adamw_step_fused** (DIAG 9 vs 11) | Remove @torch.compile from Adam step |
| head_dim=32 quality loss | bf16 NaN | No NaN at depth=2; only 2x attention compute cost (DIAG 2) | Use head_dim=64 |
| Learning rate cliff | bf16 precision boundary | Depth-dependent, not reproduced at depth=2 (DIAG 4) | Tune LR per depth |

## Performance Notes

### Confirmed Working
- **DEPTH=2**: Optimal for 5-min training budget on 8060S
- **MLP expansion=6x**: Optimal at depth=2
- **Softcap=12**: Optimal at depth=2
- **VE gate channels=16**: Optimal at depth=2
- **q*1.45 attention temp**: Optimal at depth=2
- **AOTriton enabled**: 19x speedup (44ms → 2.3ms per attention call)

### Known Broken
- **torch.compile on adamw_step_fused**: Produces NaN on gfx1151 (sole root cause of ALL NaN scenarios)
- **SDPA window_size**: Ignored on ROCm SDPA fallback
- **Gradient clipping**: Hurts performance with Muon (already normalizes)
- **Label smoothing**: Terrible for LM pretraining
- **Adam beta2=0.95**: Slightly worse than 0.97 (1.281917 vs 1.280888), not a NaN issue with uncompiled Adam

## Missing/Unknown

Issues from #6034 not found in separate GitHub issues:
1. **torch.compile code-gen bug on gfx1151** producing NaN in Adam optimizer (our primary finding - no existing issue)
2. SDPA ignoring window_size parameter
3. amd-smi reporting N/A on gfx1151
4. TheROCk Linux nightly builds stopping

Note: "bf16 precision bugs" (small batch NaN, deep network NaN, Adam beta2 NaN) have been **reclassified** as torch.compile code-gen bugs based on our 17 systematic experiments. See `findings_torch_compile_nan_gfx1151.md`.

## Recommended Action Items

1. **Enable AOTriton**: Add `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` to environment
2. **Update Firmware**: Ensure linux-firmware-20260110-1 or later
3. **Use OEM Kernel**: Ubuntu OEM kernel 1018+ for proper VGPR count
4. **Switch to TheRock**: Use TheRock PyTorch builds instead of official nightlies
5. **Add Kernel Flag**: `amdgpu.cwsr_enable=0` for stability
6. **Avoid bf16 Edge Cases**: Don't use small batches, head_dim=32, or depth≥12 with bf16
7. **Test Depth=2**: Optimal depth for 5-min training budget based on experiments

## Links

- [ROCm/ROCm Issues](https://github.com/ROCm/ROCm/issues)
- [ROCm/TheRock Issues](https://github.com/ROCm/TheRock/issues)
- [ROCm Installation Docs](https://rocm.docs.amd.com/projects/install-on-linux/)
- [PyTorch ROCm Installation](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html)
