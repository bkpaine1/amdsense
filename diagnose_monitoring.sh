#!/usr/bin/env bash
# diagnose_monitoring.sh — Collect evidence that amd-smi is blind on gfx1151
# while the data exists in sysfs. Zero GPU load. Pure diagnostic.
#
# Usage: bash diagnose_monitoring.sh > monitoring_bug_report.txt 2>&1

set -euo pipefail

echo "=============================================="
echo "gfx1151 Monitoring Bug Report"
echo "Generated: $(date -Iseconds)"
echo "Kernel: $(uname -r)"
echo "=============================================="

echo ""
echo "=== SYSTEM INFO ==="
echo "CPU: $(lscpu | grep 'Model name' | sed 's/.*: *//')"
echo "GPU: $(cat /sys/class/drm/card*/device/product_name 2>/dev/null || echo 'N/A')"
echo "GPU Device ID: $(cat /sys/class/drm/card*/device/device 2>/dev/null || echo 'N/A')"
echo "GPU Target: $(cat /sys/class/drm/card*/device/gfx_target_version 2>/dev/null || echo 'check amd-smi')"
echo "ROCm version: $(cat /opt/rocm/.info/version 2>/dev/null || echo 'N/A')"
echo "amd-smi version: $(amd-smi version 2>/dev/null || echo 'N/A')"
echo "rocm-smi version: $(rocm-smi --version 2>/dev/null | head -1 || echo 'N/A')"

echo ""
echo "=== amd-smi monitor (EXPECTED: all N/A on gfx1151) ==="
amd-smi monitor -p -t -u -m 2>/dev/null || echo "amd-smi monitor FAILED"

echo ""
echo "=== amd-smi static — LIMIT section (EXPECTED: all N/A) ==="
amd-smi static 2>/dev/null | grep -A 20 "LIMIT:" || echo "N/A"

echo ""
echo "=== amd-smi metric (EXPECTED: all N/A) ==="
amd-smi metric 2>/dev/null | head -60 || echo "amd-smi metric FAILED"

echo ""
echo "=== rocm-smi (PARTIAL — gets some data) ==="
rocm-smi 2>/dev/null

echo ""
echo "=== rocm-smi detailed ==="
rocm-smi --showclocks --showpower --showtemp --showmeminfo all --showpids 2>/dev/null

echo ""
echo "=== SYSFS RAW DATA (PROOF the data exists) ==="

echo ""
echo "--- Power (hwmon) ---"
for f in /sys/class/drm/card*/device/hwmon/hwmon*; do
    echo "hwmon path: $f"
    for attr in power1_average power1_cap power1_cap_max power1_cap_min; do
        val=$(cat "$f/$attr" 2>/dev/null || echo "NOT_EXPOSED")
        if [[ "$val" != "NOT_EXPOSED" ]]; then
            echo "  $attr: ${val} µW ($(echo "scale=1; $val/1000000" | bc)W)"
        else
            echo "  $attr: NOT EXPOSED"
        fi
    done
done

echo ""
echo "--- Temperature (hwmon) ---"
for f in /sys/class/drm/card*/device/hwmon/hwmon*; do
    for attr in temp1_input temp1_crit temp1_crit_hyst temp2_input temp3_input; do
        val=$(cat "$f/$attr" 2>/dev/null || echo "NOT_EXPOSED")
        if [[ "$val" != "NOT_EXPOSED" ]]; then
            echo "  $attr: ${val} m°C ($(echo "scale=1; $val/1000" | bc)°C)"
        else
            echo "  $attr: NOT EXPOSED"
        fi
    done
done

echo ""
echo "--- Clock frequencies ---"
echo "  sclk levels:"
cat /sys/class/drm/card*/device/pp_dpm_sclk 2>/dev/null || echo "  NOT EXPOSED"
echo "  mclk levels:"
cat /sys/class/drm/card*/device/pp_dpm_mclk 2>/dev/null || echo "  NOT EXPOSED"
echo "  Current freq (hwmon):"
for f in /sys/class/drm/card*/device/hwmon/hwmon*/freq*_input; do
    val=$(cat "$f" 2>/dev/null || echo "N/A")
    echo "    $f: $val Hz ($(echo "scale=0; $val/1000000" | bc)MHz)"
done 2>/dev/null

echo ""
echo "--- GPU utilization ---"
echo "  gpu_busy_percent: $(cat /sys/class/drm/card*/device/gpu_busy_percent 2>/dev/null || echo 'NOT EXPOSED')"
echo "  mem_busy_percent: $(cat /sys/class/drm/card*/device/mem_busy_percent 2>/dev/null || echo 'NOT EXPOSED')"

echo ""
echo "--- VRAM ---"
echo "  mem_info_vram_total: $(cat /sys/class/drm/card*/device/mem_info_vram_total 2>/dev/null || echo 'NOT EXPOSED')"
echo "  mem_info_vram_used:  $(cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null || echo 'NOT EXPOSED')"

echo ""
echo "--- Power profile / performance level ---"
echo "  power_dpm_force_performance_level: $(cat /sys/class/drm/card*/device/power_dpm_force_performance_level 2>/dev/null || echo 'NOT EXPOSED')"
echo "  pp_power_profile_mode:"
cat /sys/class/drm/card*/device/pp_power_profile_mode 2>/dev/null || echo "  NOT EXPOSED"

echo ""
echo "--- Fan ---"
for f in /sys/class/drm/card*/device/hwmon/hwmon*; do
    echo "  pwm1: $(cat "$f/pwm1" 2>/dev/null || echo 'NOT EXPOSED')"
    echo "  fan1_input: $(cat "$f/fan1_input" 2>/dev/null || echo 'NOT EXPOSED')"
done

echo ""
echo "--- PCIe ---"
echo "  current_link_speed: $(cat /sys/class/drm/card*/device/current_link_speed 2>/dev/null || echo 'NOT EXPOSED')"
echo "  current_link_width: $(cat /sys/class/drm/card*/device/current_link_width 2>/dev/null || echo 'NOT EXPOSED')"

echo ""
echo "=============================================="
echo "SUMMARY"
echo "=============================================="
echo ""
echo "amd-smi reports ALL N/A for power, temperature, clocks, utilization, and limits."
echo "rocm-smi gets PARTIAL data (temp, power, clocks) but misses power cap, fan, PCIe."
echo "The kernel driver (amdgpu via sysfs/hwmon) exposes the data correctly."
echo ""
echo "Root cause hypothesis: amd-smi uses a different interface (likely amdsmi library)"
echo "that doesn't have gfx1151 APU support. rocm-smi uses sysfs directly for some"
echo "fields but falls back to amdsmi for others, explaining partial coverage."
echo ""
echo "Impact: Users cannot monitor GPU health during ML training without raw sysfs reads."
echo "This is especially dangerous on APUs where thermal throttling and reboots occur"
echo "under sustained ML workloads (we've experienced this at 87-90°C)."
echo ""
echo "Related: Linux 7.1 adds NPU power reporting (DRM_IOCTL_AMDXDNA_GET_INFO) for"
echo "Ryzen AI. The GPU side needs the same attention."
