# amd-smi reports ALL N/A on Strix Halo gfx1151 — kernel exposes data, user-space tools are blind

## Summary

`amd-smi` v26.2.1 on ROCm 7.2.0 reports **every monitoring metric as N/A** on the Radeon 8060S (gfx1151, Strix Halo). Power, temperature, clocks, utilization, fan speed, PCIe, power caps, thermal limits — all blind. The kernel driver (`amdgpu`) exposes the data correctly through sysfs/hwmon. `amd-smi` cannot read any of it.

This is AMD's flagship AI-branded consumer APU. It says "AI" in the product name. We're running sustained ML training workloads on it. We cannot monitor GPU health through official tools.

## Hardware
- **APU**: AMD Ryzen AI MAX+ 395
- **GPU**: Radeon 8060S (gfx1151) — integrated
- **Memory**: 128 GB unified (64 GB VRAM allocation)
- **System**: GMKTEC EVO X2

## Software
- **Kernel**: 6.18.1
- **ROCm**: 7.2.0
- **amd-smi**: 26.2.1
- **rocm-smi**: 4.0.0
- **PyTorch**: 2.11.0a0+rocm7.11.0a20260106 (TheROCk nightly)

## The Problem

During sustained ML training (5-10 minute pretraining loops), the GPU runs at 87-91°C and 120W. We have experienced **thermal reboots** under heavy load. We need to monitor thermals programmatically to build safety cutoffs.

### amd-smi monitor output (COMPLETELY BLIND):
```
GPU  XCP  POWER  PWR_CAP   GPU_T   MEM_T   GFX_CLK   GFX%   MEM%  MEM_CLOCK
  0    0    N/A      N/A     N/A     N/A       N/A    N/A    N/A        N/A
```

### amd-smi metric output (COMPLETELY BLIND):
```
USAGE: N/A
POWER:
    SOCKET_POWER: N/A
    GFX_VOLTAGE: N/A
    THROTTLE_STATUS: N/A
CLOCK: N/A
TEMPERATURE:
    EDGE: 91 °C        <-- this ONE field works
    HOTSPOT: N/A
    MEM: N/A
FAN:
    SPEED: N/A
    RPM: N/A
```

### amd-smi static LIMIT section (ALL N/A):
```
LIMIT:
    PPT0:
        MAX_POWER_LIMIT: N/A
        MIN_POWER_LIMIT: N/A
        SOCKET_POWER_LIMIT: N/A
    SLOWDOWN_EDGE_TEMPERATURE: N/A
    SHUTDOWN_EDGE_TEMPERATURE: N/A
```

### Meanwhile, sysfs/hwmon exposes the data correctly:
```
power1_average:  119095000 µW  (119.0W)    ← amd-smi says N/A
temp1_input:     89000 m°C     (89.0°C)    ← amd-smi says N/A (monitor)
freq1_input:     2787000000 Hz (2787MHz)   ← amd-smi says N/A
gpu_busy_percent: 100                       ← amd-smi says N/A
pp_dpm_sclk:     0: 600Mhz / 1: 2787Mhz* / 2: 2900Mhz  ← amd-smi says N/A
```

### rocm-smi gets PARTIAL data:
```
Temp: 91.0°C  Power: 120.019W  SCLK: N/A  Fan: 0%  PwrCap: N/A  GPU%: 100%
```
`rocm-smi` reads temp, power, and GPU% from sysfs directly, but SCLK and power cap return N/A (likely falling back to the amdsmi library for those fields).

## Root Cause

The `amdsmi` library that `amd-smi` uses does not have support for gfx1151 APU monitoring interfaces. The library appears to expect discrete GPU SMU interfaces and doesn't know how to read the APU's power management data, even though the kernel's `amdgpu` driver exposes it through standard hwmon/sysfs paths.

`rocm-smi` has partial coverage because it reads some fields directly from sysfs (temp, power via hwmon) but delegates others to amdsmi (clocks, power cap, fan).

## Impact

1. **Safety**: We cannot build programmatic thermal safeguards for ML training. We've had thermal reboots at 90°C+ during sustained workloads. Without `amd-smi` working, users must scrape raw sysfs paths — which most ML practitioners won't do.

2. **Ecosystem**: NVIDIA's `nvidia-smi` works perfectly on every GPU they sell. This is table stakes. Users evaluating AMD for ML see `amd-smi` return all N/A and conclude the hardware isn't supported. We've published benchmark data showing AMD Strix Halo beating RTX 4090 on training quality (val_bpb 1.219 vs 1.844) — but we can't tell users "just use amd-smi to monitor your training" because it doesn't work.

3. **TheROCk nightlies**: The nightly PyTorch wheels for gfx1151 work excellently. But they ship with the same broken monitoring tools. Native kernel support for ML training + completely blind monitoring = a dangerous combination.

## Reproduction

```bash
# On any Strix Halo / gfx1151 system with ROCm 7.2.0:
amd-smi monitor -p -t -u -m
# Expected: Power, temp, clocks, utilization
# Actual: ALL N/A

# Proof the data exists:
cat /sys/class/drm/card*/device/hwmon/hwmon*/power1_average
cat /sys/class/drm/card*/device/hwmon/hwmon*/temp1_input
cat /sys/class/drm/card*/device/hwmon/hwmon*/freq1_input
cat /sys/class/drm/card*/device/gpu_busy_percent
# All return valid data
```

Full diagnostic script and output: https://github.com/bkpaine1/amdsense

## What We're Asking For

1. **amd-smi gfx1151 APU support** — read power, temp, clocks, utilization, and power caps from hwmon/sysfs on Strix Halo APUs
2. **Power cap exposure** — `power1_cap` is not exposed in sysfs either. Users need to know their thermal envelope.
3. **Thermal limit exposure** — slowdown and shutdown temperatures should be queryable so ML frameworks can implement safety cutoffs

## Context

We've filed 5 bf16 precision bugs separately (#6034). This issue is specifically about the monitoring gap. We're running 100+ autonomous ML experiments on this hardware and publishing all data publicly. AMD has a real competitive advantage with unified memory APUs for ML — 128GB addressable VRAM vs NVIDIA's 24GB hard limit on consumer cards. But the tooling needs to match.

Related: Linux 7.1 is adding NPU power reporting via `DRM_IOCTL_AMDXDNA_GET_INFO` for Ryzen AI. The GPU monitoring side needs the same attention.

## Why This Matters Beyond One Bug

Every ML engineer who tries AMD for the first time runs `amd-smi` and sees all N/A. Most of them go back to NVIDIA and never return. That's not a lost hardware sale — that's a lost evangelist.

Enterprise GPU procurement doesn't start with an RFP. It starts with an engineer who trained a model on Radeon at home and brings that experience to a data center conversation. NVIDIA understands this — they're actively lending engineers to open-source AI projects (OpenClaw, etc.) because every developer who builds on CUDA at their desk becomes an advocate for NVIDIA in the procurement meeting.

AMD has silicon that beats the RTX 4090 on training quality at lower cost. The hardware argument is already won. But broken tooling on consumer/prosumer platforms ensures nobody gets far enough to make that argument at work. BlackBerry had 50% enterprise market share when the iPhone launched. They lost it not because IT departments switched, but because executives experienced something better personally and pushed it into corporate.

`amd-smi` returning all N/A on a Ryzen AI MAX+ system is the equivalent of shipping a phone with no email client and wondering why enterprise won't adopt. The data is right there in the kernel. Read it.
