#!/bin/bash
# =============================================================================
# RunPod NVIDIA Benchmark — One script, paste and walk away.
# Produces profile_report.md comparable to our AMD gfx1151 results.
#
# Usage: SSH into RunPod 4070 pod, then:
#   curl -sL https://raw.githubusercontent.com/YOUR_REPO/runpod_benchmark.sh | bash
#   OR just paste this whole thing into the terminal.
#
# Requirements: PyTorch template pod with CUDA. That's it.
# =============================================================================

set -e
echo "=== Jensen's Hardware Betrayal Script v1.0 ==="
echo "Starting at $(date)"
echo ""

# --- Setup workspace ---
WORK_DIR="$HOME/amdsense_benchmark"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# --- Install deps ---
echo "[1/6] Installing dependencies..."
pip install -q tiktoken huggingface_hub requests numpy 2>/dev/null || true

# --- Clone autoresearch for the data prep ---
if [ ! -d "autoresearch" ]; then
    echo "[2/6] Cloning autoresearch..."
    git clone --depth 1 https://github.com/karpathy/autoresearch.git
fi

# --- Write our best recipe train.py ---
echo "[3/6] Writing best AMD recipe (adapted for CUDA)..."
# We copy the original train.py from autoresearch as base, then apply our hyperparams
cp autoresearch/train.py train_original.py 2>/dev/null || true

# --- Write the profiler ---
echo "[4/6] Writing profiler..."
cat > profile_nvidia.py << 'PROFILER_EOF'
#!/usr/bin/env python3
"""
NVIDIA Profiling — identical methodology to AMD profile_rocm.py.
Run on RunPod to produce comparable data.
"""
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import subprocess
import time
import re
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

REPORT_FILE = Path("profile_report_nvidia.md")
RESULTS_FILE = Path("benchmark_results.json")


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def get_hw_info():
    info = {}
    info["gpu_name"] = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    info["gpu_vram_gb"] = round(props.total_mem / 1024**3, 1)
    info["gpu_compute_capability"] = f"{props.major}.{props.minor}"
    info["gpu_multiprocessors"] = props.multi_processor_count

    # CUDA version
    info["cuda_version"] = torch.version.cuda or "N/A"
    info["torch_version"] = torch.__version__
    info["cudnn_version"] = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "N/A"

    # Check FA3 availability
    info["flash_attn_3"] = False
    try:
        from kernels import get_kernel
        cap = torch.cuda.get_device_capability()
        repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
        fa3 = get_kernel(repo)
        info["flash_attn_3"] = True
    except Exception:
        pass

    # CPU
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":")[1].strip()
                    break
    except Exception:
        info["cpu"] = "unknown"

    # RAM
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    info["ram_gb"] = round(kb / 1024**2, 1)
                    break
    except Exception:
        info["ram_gb"] = "unknown"

    # nvidia-smi
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=clocks.max.sm,clocks.max.mem,power.limit",
                                "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        parts = result.stdout.strip().split(",")
        if len(parts) >= 3:
            info["max_sm_clock"] = parts[0].strip()
            info["max_mem_clock"] = parts[1].strip()
            info["power_limit"] = parts[2].strip()
    except Exception:
        pass

    return info


def benchmark_sdpa():
    """Compare SDPA backends — same test as AMD profile."""
    log("Benchmarking SDPA variants...")
    device = torch.device("cuda")
    results = {}

    batch, heads, seq_len, head_dim = 16, 8, 1024, 64
    q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)

    # Warmup
    for _ in range(10):
        _ = F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()

    n_iters = 100

    # Default (auto-select best)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iters):
        _ = F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    results["sdpa_current"] = round((time.time() - t0) / n_iters * 1000, 3)

    # Flash attention
    try:
        with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.FLASH_ATTENTION]):
            for _ in range(5):
                _ = F.scaled_dot_product_attention(q, k, v)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(n_iters):
                _ = F.scaled_dot_product_attention(q, k, v)
            torch.cuda.synchronize()
            results["sdpa_flash"] = round((time.time() - t0) / n_iters * 1000, 3)
    except Exception as e:
        results["sdpa_flash"] = f"unavailable: {str(e)[:50]}"

    # Efficient
    try:
        with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION]):
            for _ in range(5):
                _ = F.scaled_dot_product_attention(q, k, v)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(n_iters):
                _ = F.scaled_dot_product_attention(q, k, v)
            torch.cuda.synchronize()
            results["sdpa_efficient"] = round((time.time() - t0) / n_iters * 1000, 3)
    except Exception as e:
        results["sdpa_efficient"] = f"unavailable: {str(e)[:50]}"

    # Math fallback
    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_iters):
            _ = F.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
        results["sdpa_math_fallback"] = round((time.time() - t0) / n_iters * 1000, 3)

    # Head dim sweep
    for hd in [64, 128, 256]:
        q2 = torch.randn(batch, heads, seq_len, hd, device=device, dtype=torch.bfloat16)
        k2 = torch.randn(batch, heads, seq_len, hd, device=device, dtype=torch.bfloat16)
        v2 = torch.randn(batch, heads, seq_len, hd, device=device, dtype=torch.bfloat16)
        for _ in range(5):
            _ = F.scaled_dot_product_attention(q2, k2, v2)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(50):
            _ = F.scaled_dot_product_attention(q2, k2, v2)
        torch.cuda.synchronize()
        results[f"sdpa_head{hd}"] = round((time.time() - t0) / 50 * 1000, 3)

    log(f"SDPA results: {results}")
    return results


def benchmark_matmul():
    """Matmul throughput — same test as AMD profile."""
    log("Benchmarking matmul throughput...")
    device = torch.device("cuda")
    results = {}

    # Detect peak TFLOPS for this GPU
    gpu_name = torch.cuda.get_device_name(0).lower()
    # Rough bf16 peak TFLOPS for common GPUs
    peak_tflops = 49.6  # default
    if "4090" in gpu_name:
        peak_tflops = 165.2
    elif "4080" in gpu_name:
        peak_tflops = 97.5
    elif "4070 ti" in gpu_name:
        peak_tflops = 81.6
    elif "4070" in gpu_name:
        peak_tflops = 59.1
    elif "3090" in gpu_name:
        peak_tflops = 71.0
    elif "3080" in gpu_name:
        peak_tflops = 47.0
    elif "a100" in gpu_name:
        peak_tflops = 312.0
    elif "h100" in gpu_name:
        peak_tflops = 989.5

    results["gpu_peak_tflops_bf16"] = peak_tflops

    for size in [1024, 2048, 4096]:
        a = torch.randn(size, size, device=device, dtype=torch.bfloat16)
        b = torch.randn(size, size, device=device, dtype=torch.bfloat16)
        for _ in range(5):
            _ = torch.matmul(a, b)
        torch.cuda.synchronize()
        t0 = time.time()
        n = 50
        for _ in range(n):
            _ = torch.matmul(a, b)
        torch.cuda.synchronize()
        ms = (time.time() - t0) / n * 1000
        flops = 2 * size**3
        tflops = flops / (ms / 1000) / 1e12
        pct = tflops / peak_tflops * 100
        results[f"matmul_{size}x{size}"] = {"ms": round(ms, 3), "tflops_bf16": round(tflops, 2), "pct_peak": round(pct, 1)}

    log(f"Matmul results: {results}")
    return results


def test_nan_boundaries():
    """bf16 precision tests — same as AMD profile."""
    log("Testing bf16 numerical stability...")
    device = torch.device("cuda")
    results = {}

    tests = [
        ("large_values", lambda: torch.tensor([65000.0], device=device, dtype=torch.bfloat16) * torch.tensor([2.0], device=device, dtype=torch.bfloat16)),
        ("small_gradients", lambda: torch.tensor([1e-8], device=device, dtype=torch.bfloat16) / torch.tensor([1e8], device=device, dtype=torch.bfloat16)),
        ("exp_overflow", lambda: torch.exp(torch.tensor([100.0], device=device, dtype=torch.bfloat16))),
        ("log_underflow", lambda: torch.log(torch.tensor([1e-38], device=device, dtype=torch.bfloat16))),
        ("softmax_large", lambda: F.softmax(torch.randn(1, 1024, device=device, dtype=torch.bfloat16) * 100, dim=-1)),
        ("layernorm_tiny", lambda: F.layer_norm(torch.randn(16, 512, device=device, dtype=torch.bfloat16) * 1e-7, [512])),
    ]

    for name, test_fn in tests:
        try:
            result = test_fn()
            has_nan = torch.isnan(result).any().item()
            has_inf = torch.isinf(result).any().item()
            results[name] = {"nan": has_nan, "inf": has_inf, "status": "nan" if has_nan else "inf" if has_inf else "ok"}
        except Exception as e:
            results[name] = {"status": f"error: {str(e)[:80]}"}

    x = torch.randn(32768, 512, device=device, dtype=torch.bfloat16)
    for op_name, op_fn in [("sum", torch.sum), ("mean", torch.mean), ("var", lambda t: torch.var(t, dim=-1))]:
        try:
            result = op_fn(x)
            has_nan = torch.isnan(result).any().item()
            results[f"reduce_{op_name}_32k"] = {"nan": has_nan, "status": "nan" if has_nan else "ok"}
        except Exception as e:
            results[f"reduce_{op_name}_32k"] = {"status": f"error: {str(e)[:80]}"}

    return results


def run_training_benchmark():
    """Run the autoresearch training with our best hyperparams."""
    log("Running training benchmark with best AMD recipe...")

    # Check if train.py exists (from autoresearch clone)
    train_py = Path("autoresearch/train.py")
    if not train_py.exists():
        log("ERROR: autoresearch/train.py not found")
        return {}

    # We need to patch our hyperparams into their train.py
    content = train_py.read_text()

    # Apply our best recipe hyperparams via sed-like replacement
    patches = {
        "ASPECT_RATIO": "64",
        "HEAD_DIM": "64",
        "TOTAL_BATCH_SIZE": "2**15",
        "EMBEDDING_LR": "0.8",
        "UNEMBEDDING_LR": "0.012",
        "MATRIX_LR": "0.07",
        "SCALAR_LR": "0.6",
        "WEIGHT_DECAY": "0.12",
        "WARMUP_RATIO": "0.0",
        "WARMDOWN_RATIO": "0.7",
        "FINAL_LR_FRAC": "0.07",
        "DEPTH": "8",
        "DEVICE_BATCH_SIZE": "16",
    }

    for param, val in patches.items():
        # Try with comment
        pattern = rf"^({param}\s*=\s*)(.+?)(\s*#.*)$"
        match = re.search(pattern, content, re.MULTILINE)
        if match:
            old_line = match.group(0)
            new_line = f"{match.group(1)}{val}{match.group(3)}"
            content = content.replace(old_line, new_line, 1)
            continue
        # Try without comment
        pattern2 = rf"^({param}\s*=\s*)(.+)$"
        match2 = re.search(pattern2, content, re.MULTILINE)
        if match2:
            old_line = match2.group(0)
            new_line = f"{match2.group(1)}{val}"
            content = content.replace(old_line, new_line, 1)

    # Also patch ADAM_BETAS
    content = re.sub(r"ADAM_BETAS\s*=\s*\(.+?\)", "ADAM_BETAS = (0.8, 0.98)", content)

    patched_py = Path("train_patched.py")
    patched_py.write_text(content)

    # Run it
    t0 = time.time()
    try:
        result = subprocess.run(
            ["python3", str(patched_py)],
            capture_output=True, text=True, timeout=600,
            cwd=str(Path.cwd()),
        )
        wall_time = time.time() - t0
        output = result.stdout + result.stderr

        metrics = {"wall_time": round(wall_time, 1)}
        for line in output.splitlines():
            for key in ["val_bpb", "training_seconds", "total_seconds", "peak_vram_mb",
                         "mfu_percent", "total_tokens_M", "num_steps", "num_params_M"]:
                if line.strip().startswith(f"{key}:"):
                    try:
                        metrics[key] = float(line.split(":")[1].strip())
                    except ValueError:
                        pass

        tok_matches = re.findall(r"tok/sec:\s*([\d,]+)", output)
        if tok_matches:
            metrics["tok_per_sec"] = int(tok_matches[-1].replace(",", ""))

        # Save full output for debugging
        Path("training_output.log").write_text(output)
        return metrics

    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    except Exception as e:
        return {"error": str(e)}


def generate_report(hw_info, training, sdpa, matmul, nan_results):
    """Generate NVIDIA comparison report."""
    r = []
    r.append("# NVIDIA Profiling Report — RunPod Benchmark")
    r.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    r.append(f"**Purpose**: Cross-hardware comparison with AMD Radeon 8060S (gfx1151)")
    r.append("")

    r.append("## Hardware & Environment")
    r.append(f"- **GPU**: {hw_info['gpu_name']}")
    r.append(f"- **VRAM**: {hw_info['gpu_vram_gb']} GB")
    r.append(f"- **Compute Capability**: {hw_info['gpu_compute_capability']}")
    r.append(f"- **SMs**: {hw_info['gpu_multiprocessors']}")
    r.append(f"- **CUDA**: {hw_info['cuda_version']}")
    r.append(f"- **cuDNN**: {hw_info['cudnn_version']}")
    r.append(f"- **PyTorch**: {hw_info['torch_version']}")
    r.append(f"- **FlashAttention 3**: {'available' if hw_info.get('flash_attn_3') else 'not available'}")
    r.append(f"- **CPU**: {hw_info.get('cpu', 'N/A')}")
    r.append(f"- **RAM**: {hw_info.get('ram_gb', 'N/A')} GB")
    if "max_sm_clock" in hw_info:
        r.append(f"- **Max SM Clock**: {hw_info['max_sm_clock']}")
        r.append(f"- **Max Mem Clock**: {hw_info['max_mem_clock']}")
        r.append(f"- **Power Limit**: {hw_info['power_limit']}")
    r.append("")

    r.append("## Training Results (Same Recipe as AMD)")
    if training and "val_bpb" in training:
        r.append(f"- **val_bpb**: {training['val_bpb']}")
        r.append(f"- **MFU**: {training.get('mfu_percent', 'N/A')}%")
        r.append(f"- **Tokens/sec**: {training.get('tok_per_sec', 'N/A'):,}")
        r.append(f"- **Steps**: {int(training.get('num_steps', 0))}")
        r.append(f"- **Peak VRAM**: {training.get('peak_vram_mb', 0):.0f} MB")
        r.append(f"- **Training time**: {training.get('training_seconds', 'N/A')}s")
        r.append(f"- **Wall time**: {training.get('wall_time', 'N/A')}s")
        startup = training.get('wall_time', 0) - training.get('training_seconds', 0)
        r.append(f"- **Startup overhead**: {startup:.1f}s")
    elif training:
        r.append(f"- **Error**: {training.get('error', 'unknown')}")
    r.append("")

    r.append("## SDPA Benchmarks")
    r.append("Shape: batch=16, heads=8, seq=1024")
    r.append("")
    if sdpa:
        math_ms = sdpa.get("sdpa_math_fallback", 1)
        r.append("| Backend | Time (ms) | vs Math |")
        r.append("|---------|-----------|---------|")
        for key in ["sdpa_current", "sdpa_flash", "sdpa_efficient", "sdpa_math_fallback"]:
            val = sdpa.get(key, "N/A")
            if isinstance(val, (int, float)):
                speedup = f"{math_ms/val:.1f}x" if val > 0 else "N/A"
                r.append(f"| {key} | {val} | {speedup} |")
            else:
                r.append(f"| {key} | {val} | — |")
        r.append("")
        r.append("### Head Dimension Sweep")
        r.append("| Head Dim | Time (ms) |")
        r.append("|----------|-----------|")
        for key in ["sdpa_head64", "sdpa_head128", "sdpa_head256"]:
            r.append(f"| {key.replace('sdpa_head','')} | {sdpa.get(key, 'N/A')} |")
    r.append("")

    r.append("## Matmul Throughput (bf16)")
    if matmul:
        peak = matmul.get("gpu_peak_tflops_bf16", "?")
        r.append(f"Theoretical peak: {peak} TFLOPS")
        r.append("")
        r.append("| Size | Time (ms) | TFLOPS | % Peak |")
        r.append("|------|-----------|--------|--------|")
        for key in ["matmul_1024x1024", "matmul_2048x2048", "matmul_4096x4096"]:
            data = matmul.get(key, {})
            if isinstance(data, dict):
                r.append(f"| {key.replace('matmul_','')} | {data.get('ms','N/A')} | {data.get('tflops_bf16','N/A')} | {data.get('pct_peak','N/A')}% |")
    r.append("")

    r.append("## bf16 Numerical Stability")
    r.append("| Test | Result |")
    r.append("|------|--------|")
    if nan_results:
        for name, data in nan_results.items():
            status = data.get("status", "unknown")
            r.append(f"| {name} | {status} |")
    r.append("")

    r.append("---")
    r.append(f"*NVIDIA comparison benchmark for amdsense project*")
    r.append(f"*GPU: {hw_info['gpu_name']}*")

    report_text = "\n".join(r)
    REPORT_FILE.write_text(report_text)

    # Also save raw JSON
    all_data = {
        "hardware": hw_info,
        "training": training,
        "sdpa": sdpa,
        "matmul": matmul,
        "nan": nan_results,
    }
    RESULTS_FILE.write_text(json.dumps(all_data, indent=2, default=str))

    log(f"Report: {REPORT_FILE}")
    log(f"Raw data: {RESULTS_FILE}")


def main():
    log("=== NVIDIA Benchmark for AMD Comparison ===")

    hw_info = get_hw_info()
    log(f"GPU: {hw_info['gpu_name']}, VRAM: {hw_info['gpu_vram_gb']}GB, CUDA: {hw_info['cuda_version']}")

    training = run_training_benchmark()
    sdpa = benchmark_sdpa()
    matmul = benchmark_matmul()
    nan_results = test_nan_boundaries()

    generate_report(hw_info, training, sdpa, matmul, nan_results)

    log("=== NVIDIA Benchmark Complete ===")
    log("Download these files:")
    log(f"  - {REPORT_FILE}")
    log(f"  - {RESULTS_FILE}")
    log(f"  - training_output.log")
    print("\n\nDONE. Copy the files above back to your local machine.")


if __name__ == "__main__":
    main()
PROFILER_EOF

# --- Write the one-liner runner ---
echo "[5/6] Preparing runner..."
cat > run_benchmark.sh << 'RUNNER_EOF'
#!/bin/bash
cd ~/amdsense_benchmark
python3 profile_nvidia.py 2>&1 | tee benchmark.log
echo ""
echo "=== FILES TO DOWNLOAD ==="
echo "  ~/amdsense_benchmark/profile_report_nvidia.md"
echo "  ~/amdsense_benchmark/benchmark_results.json"
echo "  ~/amdsense_benchmark/training_output.log"
echo "  ~/amdsense_benchmark/benchmark.log"
echo ""
echo "scp them back or cat them into your terminal."
RUNNER_EOF
chmod +x run_benchmark.sh

echo "[6/6] Setup complete!"
echo ""
echo "=== READY ==="
echo "Run:  cd ~/amdsense_benchmark && bash run_benchmark.sh"
echo "Then download the report files when done (~15 min)."
echo ""
echo "Or just run it now:"
echo "  bash run_benchmark.sh"
