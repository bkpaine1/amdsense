# ROCm GitHub Issues Related to gfx1151/Strix Halo/Radeon 8060S

Research compiled: 2026-03-19

This document catalogs ROCm GitHub issues related to the problems documented in issue #6034, focusing on gfx1151 (Strix Halo/Radeon 8060S) hardware and associated software stack issues.

---

## Executive Summary

**Total Issues Found**: 17 directly related issues across ROCm repositories

**Primary Categories**:
1. **Firmware and Memory Issues** (6 issues) - Page faults, memory access violations, CWSR problems
2. **Precision and Stability** (3 issues) - bf16 issues, deep network instability, optimizer precision
3. **Performance and Optimization** (4 issues) - AOTriton, SDPA, flash attention, memory allocator
4. **Platform Support** (4 issues) - WSL2, Linux detection, documentation gaps

**Status Overview**:
- Open: 7 issues
- Closed/Completed: 10 issues

**Critical Finding**: Many "closed" issues represent workarounds rather than true fixes, with users dependent on specific firmware versions, kernel flags (cwsr_enable=0), or nightly builds.

---

## 1. Primary Reference Issue

### Issue #6034: Strix Halo gfx1151: 93 ML experiments, 5 critical bf16 bugs, AOTriton 19x speedup undocumented

**URL**: https://github.com/ROCm/ROCm/issues/6034

**Status**: OPEN

**Author**: bkpaine1

**Date**: March 13, 2026

**Labels**: status: triage

**Assigned to**: tcgu-amd (Tim Gu)

**Summary**:
Comprehensive ML research report documenting 250+ autonomous experiments on Strix Halo revealing critical software gaps despite exceptional hardware performance.

**Key Findings**:
- **Performance Achievement**: val_bpb 1.2080 (~34.5% better than RTX 4090 on equivalent workloads)
- **Critical bf16 Bugs**:
  - bf16 accumulation crashes at batch sizes ≤2^13
  - NaN failures with head dimension=32
  - Network instability at depth ≥12
  - Crashes with wide aspect ratios (128)
  - Sharp precision boundary at specific learning rates

- **Undocumented AOTriton Speedup**: `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` yields 19x speedup (44ms → 2.3ms per attention call) but remains undocumented

- **Ecosystem Issues**:
  - Consumer GPU support requires nightly builds vs stable wheels
  - Default shell configuration can crash PyTorch initialization
  - Documentation gaps for gfx1151 kernel support

**Relation to Other Issues**: This is the primary meta-issue documenting systematic research that connects precision bugs, performance optimizations, and ecosystem gaps.

---

## 2. Firmware and Memory Issues

### Issue #5991: GPU page fault on gfx1151 (Radeon 8060S) — basic tensor operations hang

**URL**: https://github.com/ROCm/ROCm/issues/5991

**Status**: OPEN

**Author**: naveenjafer

**Date**: February 24, 2026

**Labels**: status: assessed

**Assigned to**: amd-nicknick

**Summary**:
Basic PyTorch GPU operations cause indefinite hangs with GFX Hub page faults.

**Problem**:
- Simple operations like `torch.randn(1000, 1000, device="cuda:0")` trigger page fault and hang
- Kernel logs show "GFX Hub page fault...walker, mapping, and permission errors"
- Device detected correctly but compute dispatch fails

**Environment**:
- PyTorch: 2.11.0a0+rocm7.11.0a20260106 (nightly)
- GPU: AMD Radeon 8060S (Device ID 0x1586)
- OS: Ubuntu 24.04.4 LTS, Kernel: 6.17.0-1011-oem

**Workaround**: Apply `cwsr_enable=0` boot flag until kernel 6.18-rc6+ available

**Relation to #6034**: Documents the underlying page fault issues that may contribute to training instability at various depths/configurations.

---

### Issue #5824: Memory Access Fault on gfx1151 (Strix Halo)

**URL**: https://github.com/ROCm/ROCm/issues/5824

**Status**: CLOSED (Completed)

**Author**: adamskrodzki

**Date**: December 29, 2025

**Labels**: status: triage

**Summary**:
Critical GPU memory failures on basic PyTorch operations.

**Problem**:
- "Memory access fault by GPU node-1...Page not present or supervisor privilege"
- Immediate crashes on GPU tensor operations

**Hardware**:
- CPU: AMD Ryzen AI Max+ 395 (Strix Halo)
- GPU: AMD Radeon 8060S (gfx1151)
- RAM: 128GB unified (32GB CPU/96GB GPU)

**Software**:
- OS: Nobara Linux 43 (Fedora-based)
- Kernel: 6.17.8-200.nobara.fc43.x86_64
- PyTorch: 2.9.1+rocm6.3

**Resolution**: Updated linux-firmware-20260110-1 package resolved primary issues, though some segfaults persisted

**Relation to #6034**: Demonstrates firmware-related memory access issues that could contribute to bf16 precision bugs and deep network instability.

---

### Issue #5724: amdgpu firmware (MES 0x83) causing GPU Hang / Memory access fault w/ Strix Halo

**URL**: https://github.com/ROCm/ROCm/issues/5724

**Status**: CLOSED (Completed)

**Author**: ianbmacdonald

**Date**: November 29, 2025

**Labels**: status: assessed

**Assigned to**: amd-nicknick (Nick Kuo)

**Summary**:
New amdgpu firmware (MES 0x83) introduced memory access faults in previously functional scenarios.

**Problem**:
- "Memory access fault by GPU node-1...Page not present or supervisor privilege"
- Reproducible using AMD ROCm vLLM serving IBM Granite hybrid models
- Kernel page faults with gfxhub errors and permission violations
- 54 comments indicating widespread community impact

**Workarounds**:
- Use amdgpu-dkms-firmware package (maintains MES 0x80)
- Apply kernel flag: `amdgpu.cwsr_enable=0`
- Avoid upstream linux-firmware packages

**Environment**:
- OS: Ubuntu 24.04.3, Debian 13
- GPU: Strix Halo (gfx1151)
- ROCm: 7.1.1

**Resolution**: Firmware subsequently reverted upstream

**Relation to #6034**: Firmware instability could explain training crashes at specific batch sizes/configurations.

---

### Issue #5890: amdgpu pagefault under rocm7.2 on gfx1151

**URL**: https://github.com/ROCm/ROCm/issues/5890

**Status**: CLOSED (Completed)

**Author**: kellrott

**Date**: January 23, 2026

**Labels**: status: triage

**Assigned to**: amd-nicknick

**Summary**:
ROCm 7.2 upgrade caused system hangs when copying tensors to device.

**Problem**:
- Previously working with ROCm 7.1
- ROCm 7.2 causes torch operations to freeze
- dmesg error: "[gfxhub] page fault...GCVM_L2_PROTECTION_FAULT_STATUS...PERMISSION_FAULTS: 0x3"

**Reproduction**:
```python
import torch
torch.cuda.is_available()  # Returns True
torch.randn(4096, 4096, device='cuda', dtype=torch.float32)  # Freezes
```

**Resolution**: Using Ubuntu OEM kernel drivers instead of amdgpu-install DKMS resolved the problem. Recommended: `amdgpu-install -y --usecase=rocm --no-dkms`

**Hardware**: AMD Ryzen AI Max+ 395 with Radeon 8060S

**Relation to #6034**: Page fault issues during tensor operations could explain NaN failures and crashes at specific batch sizes.

---

### Issue #2991 (TheRock): Incorrect VGPR count causing crashes in ROCm 6.x/7.0.x/7.1.x/7.9/7.10 on Linux

**URL**: https://github.com/ROCm/TheRock/issues/2991

**Status**: CLOSED (Completed)

**Author**: fjankovi

**Date**: January 19, 2026

**Summary**:
ROCr runtime contained incorrect VGPR count values for gfx1151 causing crashes.

**Key Details**:
- **User-Mode Fix**: Available in nightly builds since December 19, 2025; planned for ROCm 7.2 and 7.11
- **Kernel-Mode Fix**: Ubuntu OEM kernel 1018 and newer; not in Ubuntu 24.04 generic kernel

**Addressed Components**:
- rocm-systems Pull Request #2200 (CWSR and control stack exposure)
- Linux kernel patch for exporting stack size parameters to userspace

**Affected Systems**: Linux, any CPU, gfx1151 GPU, ROCm 6.x through 7.10

**Relation to #6034**: Incorrect VGPR counts could cause instability in deep networks (depth ≥12) and explain crashes at specific model configurations.

---

### Issue #5853: Strix Halo (gfx1151) gives segfault on any VRAM access with torch nightly package

**URL**: https://github.com/ROCm/ROCm/issues/5853

**Status**: CLOSED (Completed)

**Author**: chaserhkj

**Date**: January 15, 2026

**Labels**: status: triage

**Assigned to**: lucbruni-amd

**Summary**:
Segmentation faults on GPU memory operations with PyTorch nightly ROCm 7.1 builds.

**Problem**:
- "Any ROCm VRAM access" triggers segfault
- Same operations succeed with TheRock repository builds
- Suggests discrepancy between PyTorch nightly and specialized gfx1151 packages

**System**:
- CPU: AMD RYZEN AI MAX+ 395
- GPU: Radeon 8060S (gfx1151)
- OS: Arch Linux
- ROCm: 7.1.0

**Debug Info**: "numa_node_id is out range" warnings suggest memory management configuration issues

**Relation to #6034**: Demonstrates ecosystem fragmentation where nightly builds lack proper gfx1151 support, forcing reliance on specialized builds.

---

## 3. Platform-Specific Issues

### Issue #6022: [Strix Halo / 8060S] librocdxg fails to map Dedicated VRAM in WSL2

**URL**: https://github.com/ROCm/ROCm/issues/6022

**Status**: OPEN

**Author**: sandynz

**Date**: March 7, 2026

**Labels**: status: triage

**Assigned to**: benrichard-amd

**Summary**:
WSL2 fails to map dedicated 96GB VRAM; ROCm pool limited by .wslconfig memory setting.

**Problem**:
- Native Windows successfully uses 41.5GB of 96GB dedicated VRAM
- WSL2 ROCm pool size tied to .wslconfig memory, not BIOS UMA allocation
- Attempts beyond WSL2 VM limit trigger system paging and CPU throttling (85°C)
- rocminfo reports "IOMMU Support: None" despite BIOS IOMMU enabled

**Expected**: System should recognize full 96GB dedicated VRAM as primary LOCAL_SEGMENT

**Relation to #6034**: WSL2 memory mapping issues could contribute to crashes at larger batch sizes or model sizes requiring more VRAM.

---

### Issue #4809: WSL2 not detect Radeon 8060S

**URL**: https://github.com/ROCm/ROCm/issues/4809

**Status**: CLOSED (Completed)

**Author**: yaoman3

**Date**: May 27, 2025

**Summary**:
AMD Radeon 8060S not recognized after installing WSL2 drivers.

**Problem**:
- Only CPU detected via `lspci` or `glxinfo`
- Following official installation procedures
- WSL2 version 2.4.13.0, Windows 11, AMDGPU driver 25.5.1

**Resolution**: AMD clarified "Radeon 8060S and other APUs are not yet supported by ROCm on WSL" - directed to GPU Support Matrix

**Relation to #6034**: Documents lack of WSL2 support for consumer APUs, limiting development/testing platforms.

---

### Issue #3874 (HIP): HIP Windows (gfx1151) hard crash in H2D memcpy via torch.Tensor.to("cuda")

**URL**: https://github.com/ROCm/hip/issues/3874

**Status**: CLOSED (Completed)

**Author**: ajc9988

**Date**: October 22, 2025

**Summary**:
Critical crash on Windows during host-to-device memory transfers with PyTorch.

**Problem**:
- Access violation in `amdhip64_6.dll` during `Tensor.to("cuda")`
- Device initialization and allocation succeed
- Exception Code: 0xC0000005 in `c10::hip::memcpy_and_sync()`

**Hardware**:
- AMD Ryzen AI Max+ 395 with Radeon 8060S (gfx1151)
- Windows 11 Pro (Build 26100)

**Software**:
- PyTorch 2.8.0a0 (ROCm preview)
- ROCm/HIP 6.4.50101
- Driver version 32.0.22001.14007

**Reproduction**: `torch.empty()` works, but `tensor.to("cuda")` crashes

**Relation to #6034**: Windows memory copy failures could explain platform-specific training issues and limit development workflows.

---

### Issue #5339: Confusing rocm support for gfx1151

**URL**: https://github.com/ROCm/ROCm/issues/5339

**Status**: OPEN

**Author**: VantorreWannes

**Date**: September 16, 2025

**Labels**: Under Investigation, status: assessed

**Assigned to**: harkgill-amd

**Summary**:
Conflicting documentation about gfx1151 support across ROCm versions.

**Confusion Points**:
1. Some sources indicate gfx1151 support in ROCm 6
2. ROCm 7 announcements mention "Ryzen AI MAX products"
3. Official docs lack gfx1151 in compatibility matrices/release notes

**Clarification from AMD**:
- ROCm 6.4.4: preview-level PyTorch support for gfx1151 (Windows/Linux)
- Full stable ROCm 7.x support: still in development
- TheRock repository: bleeding-edge ROCm 7 wheels for early adopters

**Relation to #6034**: Documentation gaps explain ecosystem confusion and difficulty determining supported configurations.

---

## 4. Performance and Optimization Issues

### Issue #2229 (TheRock): comfyui can not use --pytorch-cross-attention with aotriton on gfx1103

**URL**: https://github.com/ROCm/TheRock/issues/2229

**Status**: CLOSED (Completed)

**Author**: LuXuxue

**Date**: November 20, 2025

**Labels**: status: triage

**Summary**:
AOTriton cross-attention fails on gfx1103 despite experimental flag enabled.

**Problem**:
- `--pytorch-cross-attention` flag causes HIP error "invalid argument" during kernel decompression
- `--use-quad-cross-attention` works properly
- `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` enabled
- Attention passes PyTorch validation but fails at runtime

**Hardware**:
- GPU: Radeon 780M Graphics (gfx1103)
- ROCm: 7.10
- OS: Windows 11 24H2

**Resolution**: AMD indicated work-in-progress for gfx1103; resolved in torch-2.10.0a0+rocm7.11.0a20251211

**Relation to #6034**: Demonstrates AOTriton experimental flag issues across gfx11 architecture family; similar problems may affect gfx1151's 19x speedup feature.

---

### Issue #3128 (TheRock): Gfx1151 achieves better inference performance with 1 GB than 4 GB dedicated memory

**URL**: https://github.com/ROCm/TheRock/issues/3128

**Status**: OPEN

**Author**: menglcai

**Date**: January 28, 2026

**Summary**:
Counterintuitive performance degradation with increased dedicated memory allocation.

**Performance Results**:
- 4GB allocation: ~6.3-6.4 seconds per SD3 pipeline round
- 1GB allocation: ~5.8-5.9 seconds per pipeline round

**Hardware**:
- Device: Radeon 8060S (AMD RYZEN AI MAX+ 395W)
- OS: Windows 26200.7623
- ROCm: TheRock 2.9.1+rocm7.11.0a20260114

**Reproducibility**: Verified by adjusting Adrenalin Performance→Tuning dedicated memory settings

**Relation to #6034**: Memory allocator issues (PYTORCH_HIP_ALLOC_CONF) may explain crashes at specific batch sizes and performance cliffs.

---

### Issue #4618 (rocm-libraries): Missing support for Radeon 8060s (Ryzen AI Max+ 395) gfx1151

**URL**: https://github.com/ROCm/rocm-libraries/issues/4618

**Status**: OPEN (reopened February 17, 2026)

**Author**: Zerout

**Date**: October 16, 2025

**Summary**:
rocWMMA support for gfx1151 broken in practice despite claimed fixes.

**Problem**:
- Initially reported missing gfx1151 support
- Marked resolved when gfx1151 added to rocWMMA
- Reopened when users found compilation failures with flash attention
- Compiling with `-DGGML_HIP_ROCWMMA_FATTN=ON -DAMDGPU_TARGETS=gfx1151` fails
- Static assertion errors: "Unsupported architecture"

**Technical Issue**: HIP clang compiler treats `_Float16` and `half` as distinct types on gfx1151, causing type mismatches

**Relation to #6034**: rocWMMA issues may contribute to SDPA/flash attention performance problems and force fallback to slower paths.

---

### Issue #3397 (TheRock): Native 7.11 packages are missing rocWMMA

**URL**: https://github.com/ROCm/TheRock/issues/3397

**Status**: Not fetched (referenced in search results)

**Summary**: ROCm 7.11 packages don't include rocWMMA for gfx1151

**Problem**: When installing amdrocm-core-dev7.11-gfx1151, rocwmma.hpp missing from installation

**Relation to #6034**: Missing libraries force users to build from source, contributing to ecosystem fragmentation.

---

## 5. Precision and Stability Issues

### Issue #6034: bf16 bugs (CORRECTED by our diagnostic experiments)

See Section 1 for complete details.

**Original Hypotheses from #6034**:
1. **Small Batch Crash**: bf16 accumulation crashes at batch sizes ≤2^13
2. **Head Dimension NaN**: NaN failures with head_dim=32
3. **Deep Network Instability**: Network instability at depth ≥12
4. **Aspect Ratio Crash**: Crashes with wide aspect ratios (128)
5. **Learning Rate Cliff**: Sharp precision boundary at specific learning rates

**CORRECTED by our 17 diagnostic experiments (DIAG 1-18)**:
- Items 1, 3 are **NOT bf16 precision bugs** - they are `torch.compile` code-gen bugs on gfx1151
- Item 2 (head_dim=32) shows no NaN at depth=2, only quality loss from 2x attention compute
- Item 4 (aspect ratio) and Item 5 (LR cliff) are depth-dependent, not reproduced at depth=2
- **Root cause**: `torch.compile(adamw_step_fused)` generates incorrect kernel code on gfx1151
- **Evidence**: Removing `@torch.compile` from Adam step eliminates ALL NaN across all tested configs
- See `findings_torch_compile_nan_gfx1151.md` for full evidence table

**Actual Root Cause**: `torch.compile` code generation for gfx1151, NOT:
- ~~Firmware memory access issues~~ (page faults are a separate problem)
- ~~VGPR count errors~~ (fixed in Dec 2025 nightlies, our nightlies are Jan 2026)
- ~~MFMA denormal flushing~~ (NaN persists even with fp32 inside compiled kernel)

---

## 6. Ecosystem and Documentation Issues

### Issue #5696: ROCm 7.1.0 incorrectly reports 8060S as GFX1201 when system uses RDNA4 eGPU

**URL**: https://github.com/ROCm/ROCm/issues/5696

**Status**: CLOSED (Completed)

**Author**: Soddentrough

**Date**: November 26, 2025

**Labels**: status: triage

**Summary**:
Configuration mismatch where ROCm misidentified GPU architecture.

**System**:
- AMD RYZEN AI MAX+ 395 with Radeon 8060S (integrated)
- R9700 Pro (external GPU via OcuLink)

**Problem**:
- Kernel driver correctly detected gfx_v12_0 and gfx_v11_0
- ROCm's `rocminfo` showed both as "gfx1201"

**Resolution**: User error - environment variable forcing override:
`export HSA_OVERRIDE_GFX_VERSION=12.0.1`

Removing override restored proper identification: "gfx1201" and "gfx1151"

**Relation to #6034**: Demonstrates potential for configuration errors and environment variable interactions affecting GPU detection.

---

## 7. Summary of Key Patterns

### Common Root Causes

1. **Firmware Issues**:
   - MES firmware 0x83 causing page faults
   - Requires cwsr_enable=0 workaround
   - linux-firmware-20260110-1 partially resolves

2. **Kernel/Driver Mismatch**:
   - Ubuntu OEM kernel 1018+ required for proper VGPR count
   - Generic kernels lack gfx1151 fixes
   - DKMS vs no-DKMS installation differences

3. **ROCm Version Fragmentation**:
   - Nightly builds required for consumer GPUs
   - TheRock repository more reliable than PyTorch nightlies
   - Version-specific bugs (7.1 works, 7.2 breaks, 7.11 fixes)

4. **Documentation Gaps**:
   - AOTriton experimental flag undocumented
   - gfx1151 support status unclear
   - GPU Support Matrix incomplete

### Workarounds in Use

1. **Kernel Flags**: `amdgpu.cwsr_enable=0`
2. **Firmware**: Stay on MES 0x80 via amdgpu-dkms-firmware
3. **Kernel**: Use Ubuntu OEM kernel, not generic
4. **Installation**: `amdgpu-install -y --usecase=rocm --no-dkms`
5. **PyTorch**: Use TheRock builds instead of official nightlies
6. **Environment**: `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` for 19x speedup

### Outstanding Issues

1. **torch.compile code-gen bug on gfx1151**: No existing issue - our primary novel finding (17 experiments proving compiled Adam step generates NaN)
2. **SDPA Window Size**: No issues confirming window_size ignored on SDPA fallback
3. **amd-smi N/A Reporting**: No issues found for amd-smi reporting N/A on gfx1151
4. **TheROCk Linux Nightly Stopping**: No issues documenting Linux nightly builds stopping

Note: "bf16 precision bugs" and "Adam optimizer bf16" previously listed here have been **reclassified** as torch.compile code-gen bugs.

---

## 8. Recommendations for Further Investigation

### Issues to Create

Based on our diagnostic findings not yet represented in ROCm issue tracker:

1. **torch.compile code-gen bug on gfx1151**: Compiled Adam optimizer step produces NaN; 17 experiments isolating the cause (HIGH PRIORITY - novel finding with full evidence chain)
2. **SDPA Window Size Ignored**: Document that WINDOW_PATTERN has no effect on SDPA fallback
3. **amd-smi gfx1151 Support**: Document amd-smi reporting N/A for metrics on gfx1151

Note: Original items "bf16 Small Batch NaN", "Head Dimension 32 Crash", and "Adam bf16 Precision" have been reclassified as manifestations of the torch.compile bug (item 1).

### Cross-References Needed

Issues #6034 should reference:
- #5991 (page faults)
- #5724 (firmware issues)
- #2991 (VGPR count)
- #6022 (WSL2 VRAM mapping)
- #4618 (rocWMMA support)
- #3128 (memory allocator performance)

---

## 9. Sources

### ROCm/ROCm Repository
- [Issues · ROCm/ROCm](https://github.com/ROCm/ROCm/issues)
- [Issue #6034: Strix Halo gfx1151 ML experiments](https://github.com/ROCm/ROCm/issues/6034)
- [Issue #5991: GPU page fault basic tensor operations](https://github.com/ROCm/ROCm/issues/5991)
- [Issue #5824: Memory Access Fault](https://github.com/ROCm/ROCm/issues/5824)
- [Issue #5724: amdgpu firmware MES 0x83](https://github.com/ROCm/ROCm/issues/5724)
- [Issue #5890: amdgpu pagefault rocm7.2](https://github.com/ROCm/ROCm/issues/5890)
- [Issue #5853: Segfault torch nightly](https://github.com/ROCm/ROCm/issues/5853)
- [Issue #6022: WSL2 VRAM mapping](https://github.com/ROCm/ROCm/issues/6022)
- [Issue #4809: WSL2 not detect 8060S](https://github.com/ROCm/ROCm/issues/4809)
- [Issue #5696: ROCm 7.1.0 incorrectly reports GFX1201](https://github.com/ROCm/ROCm/issues/5696)
- [Issue #5339: Confusing rocm support documentation](https://github.com/ROCm/ROCm/issues/5339)

### ROCm/TheRock Repository
- [Issue #2991: Incorrect VGPR count](https://github.com/ROCm/TheRock/issues/2991)
- [Issue #2229: comfyui pytorch-cross-attention](https://github.com/ROCm/TheRock/issues/2229)
- [Issue #3128: Better performance with less memory](https://github.com/ROCm/TheRock/issues/3128)
- [Issue #3397: Missing rocWMMA](https://github.com/ROCm/TheRock/issues/3397)

### ROCm/hip Repository
- [Issue #3874: HIP Windows H2D memcpy crash](https://github.com/ROCm/hip/issues/3874)

### ROCm/rocm-libraries Repository
- [Issue #4618: Missing gfx1151 support](https://github.com/ROCm/rocm-libraries/issues/4618)

### ROCm/aotriton Repository
- [Releases · ROCm/aotriton](https://github.com/ROCm/aotriton/releases)
- [GitHub - ROCm/aotriton](https://github.com/ROCm/aotriton)

### ROCm Documentation and Blogs
- [PyTorch on ROCm installation](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html)
- [Optimizing FP4 Mixed-Precision Inference with Petit](https://rocm.blogs.amd.com/artificial-intelligence/fp4-mixed-precision/README.html)
- [Empowering Developers PyTorch Ecosystem](https://rocm.blogs.amd.com/artificial-intelligence/pytorch-amd-gpus/README.html)
- [ROCm 7 Software](https://www.amd.com/en/products/software/rocm/whats-new.html)

### Related Projects
- [AMD Strix Halo ROCm Guide - Ollama](https://github.com/ollama/ollama/issues/14855)
- [GitHub Petit Kernel](https://github.com/causalflow-ai/petit-kernel)

---

## Appendix: Search Queries Used

Successful queries that yielded results:
1. `site:github.com/ROCm 8060S Radeon issues`
2. `ROCm firmware gfx1151 page fault cwsr_enable`
3. `ROCm TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL github`
4. `ROCm nightly builds PyTorch consumer GPU 2026`
5. `site:github.com/ROCm/ROCm-libraries gfx1151 rocWMMA support`

Queries with limited/no results:
1. `site:github.com/ROCm/ROCm Adam optimizer bf16 precision`
2. `site:github.com/ROCm PyTorch SDPA fallback window_size`
3. `site:github.com/ROCm/amd-smi gfx1151`
4. `site:github.com/ROCm TheROCk Linux nightly stopped`
5. `site:github.com/ROCm batch size NaN training`
6. `site:github.com/ROCm head dimension attention crash 32`

This suggests some specific technical issues from #6034 have not yet been independently reported or may be unique findings from the systematic ML experimentation.
