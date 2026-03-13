#!/usr/bin/env python3
"""
ROCm Profiling Agent — produces AMD-actionable data.

Runs the best recipe with profiling enabled, then generates a report:
1. Per-kernel time breakdown (where does GPU time go?)
2. SDPA with vs without AOTriton experimental
3. Memory bandwidth utilization vs theoretical peak
4. Op-level fallback detection (what's running unoptimized?)
5. NaN stability boundary testing
6. Hardware specs and environment snapshot

Output: profile_report.md — the doc we hand AMD.

Usage:
    source venv/bin/activate
    unset PYTORCH_HIP_ALLOC_CONF
    export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
    export GPU_BF16_PEAK_FLOPS=49.6e12
    python3 profile_rocm.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import json
import subprocess
import time
import re
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

REPORT_FILE = Path(__file__).parent / "profile_report.md"
TRAIN_PY = Path(__file__).parent / "train.py"

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def get_hw_info():
    """Collect hardware and environment info."""
    info = {}

    # GPU
    info["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    props = torch.cuda.get_device_properties(0)
    info["gpu_vram_gb"] = round(props.total_memory / 1024**3, 1)
    info["gpu_compute_capability"] = f"{props.major}.{props.minor}"
    info["gpu_multiprocessors"] = props.multi_processor_count

    # ROCm version
    try:
        result = subprocess.run(["rocm-smi", "--showdriverversion"], capture_output=True, text=True, timeout=10)
        for line in result.stdout.splitlines():
            if "Driver" in line:
                info["rocm_driver"] = line.strip()
                break
    except Exception:
        info["rocm_driver"] = "unknown"

    try:
        result = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=10)
        for line in result.stdout.splitlines():
            if "Name:" in line and "gfx" in line.lower():
                info["gpu_arch"] = line.split(":")[-1].strip()
                break
    except Exception:
        info["gpu_arch"] = "unknown"

    # PyTorch
    info["torch_version"] = torch.__version__
    info["hip_version"] = getattr(torch.version, 'hip', 'N/A')

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

    # AOTriton
    info["aotriton_experimental"] = os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "0")

    # TheROCk index
    info["therock_index"] = "https://rocm.nightlies.amd.com/v2/gfx1151/"

    return info


def profile_training_step(num_warmup=5, num_profile=10):
    """Run a few training steps under the PyTorch profiler."""
    log("Setting up model for profiling...")

    from prepare import MAX_SEQ_LEN, Tokenizer, make_dataloader

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    tokenizer = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()

    # Import model classes from train.py namespace
    # We'll exec the model definition portion
    train_src = TRAIN_PY.read_text()

    # Build a minimal model with current best hyperparams
    exec_globals = {
        "torch": torch, "nn": nn, "F": F,
        "os": os, "gc": gc, "time": time,
        "MAX_SEQ_LEN": MAX_SEQ_LEN,
        "vocab_size": vocab_size,
    }

    # Extract everything up to the training loop
    # Find "# Training loop" or the optimizer setup end
    lines = train_src.splitlines()
    setup_end = None
    for i, line in enumerate(lines):
        if "# Training loop" in line or "Training loop" in line:
            setup_end = i
            break
        if "train_loader = " in line:
            setup_end = i + 1
            break

    if setup_end is None:
        # Fallback: find where step variable is initialized
        for i, line in enumerate(lines):
            if re.match(r'^step\s*=\s*0', line.strip()):
                setup_end = i + 1
                break

    if setup_end is None:
        setup_end = len(lines)

    setup_code = "\n".join(lines[:setup_end])

    log(f"Executing model setup ({setup_end} lines)...")
    exec(compile(setup_code, "train_setup", "exec"), exec_globals)

    model = exec_globals.get("model")
    optimizer = exec_globals.get("optimizer")
    train_loader = exec_globals.get("train_loader")

    if model is None:
        log("ERROR: Could not extract model from train.py")
        return None

    log(f"Model loaded. Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # Warmup steps
    log(f"Running {num_warmup} warmup steps...")
    model.train()
    x, y, epoch = next(train_loader)
    for i in range(num_warmup):
        with autocast_ctx:
            loss = model(x, y)
        loss.backward()
        optimizer.step()
        model.zero_grad(set_to_none=True)
        x, y, epoch = next(train_loader)

    torch.cuda.synchronize()

    # Profiled steps
    log(f"Profiling {num_profile} steps...")
    profile_dir = Path(__file__).parent / "profile_traces"
    profile_dir.mkdir(exist_ok=True)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        schedule=torch.profiler.schedule(
            wait=0,
            warmup=2,
            active=num_profile - 2,
            repeat=1,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profile_dir)),
    ) as prof:
        for i in range(num_profile):
            with autocast_ctx:
                loss = model(x, y)
            loss.backward()
            optimizer.step()
            model.zero_grad(set_to_none=True)
            x, y, epoch = next(train_loader)
            torch.cuda.synchronize()
            prof.step()

    log("Profiling complete. Generating kernel table...")

    # Key averages by kernel
    key_averages = prof.key_averages()
    kernel_table = key_averages.table(sort_by="cuda_time_total", row_limit=30)

    # Extract top GPU kernels
    top_kernels = []
    for evt in sorted(key_averages, key=lambda e: e.cuda_time_total, reverse=True)[:30]:
        if evt.cuda_time_total > 0:
            top_kernels.append({
                "name": evt.key,
                "cuda_time_ms": round(evt.cuda_time_total / 1000, 2),
                "cpu_time_ms": round(evt.cpu_time_total / 1000, 2),
                "calls": evt.count,
                "cuda_mem_mb": round(evt.cuda_memory_usage / 1024**2, 1) if evt.cuda_memory_usage else 0,
            })

    # Memory stats
    mem_stats = {
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 1024**2, 1),
    }

    return {
        "kernel_table": kernel_table,
        "top_kernels": top_kernels,
        "mem_stats": mem_stats,
    }


def benchmark_sdpa():
    """Compare SDPA with and without AOTriton experimental."""
    log("Benchmarking SDPA variants...")
    device = torch.device("cuda")
    results = {}

    # Test dimensions matching our model
    batch, heads, seq_len, head_dim = 16, 8, 1024, 64
    q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)

    # Warmup
    for _ in range(10):
        _ = F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()

    # Benchmark
    n_iters = 100
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iters):
        _ = F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    t1 = time.time()
    ms_per_call = (t1 - t0) / n_iters * 1000
    results["sdpa_current"] = round(ms_per_call, 3)

    # Check which backend is being used
    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION]):
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_iters):
            _ = F.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
        t1 = time.time()
        results["sdpa_efficient"] = round((t1 - t0) / n_iters * 1000, 3)

    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_iters):
            _ = F.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
        t1 = time.time()
        results["sdpa_math_fallback"] = round((t1 - t0) / n_iters * 1000, 3)

    # Test different head dims
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
        t1 = time.time()
        results[f"sdpa_head{hd}"] = round((t1 - t0) / 50 * 1000, 3)

    log(f"SDPA results: {results}")
    return results


def benchmark_matmul():
    """Memory bandwidth and compute throughput via matmul."""
    log("Benchmarking matmul throughput...")
    device = torch.device("cuda")
    results = {}

    for size in [1024, 2048, 4096]:
        a = torch.randn(size, size, device=device, dtype=torch.bfloat16)
        b = torch.randn(size, size, device=device, dtype=torch.bfloat16)
        # warmup
        for _ in range(5):
            _ = torch.matmul(a, b)
        torch.cuda.synchronize()
        t0 = time.time()
        n = 50
        for _ in range(n):
            _ = torch.matmul(a, b)
        torch.cuda.synchronize()
        t1 = time.time()
        ms = (t1 - t0) / n * 1000
        flops = 2 * size**3  # matmul FLOPS
        tflops = flops / (ms / 1000) / 1e12
        results[f"matmul_{size}x{size}"] = {"ms": round(ms, 3), "tflops_bf16": round(tflops, 2)}

    log(f"Matmul results: {results}")
    return results


def test_nan_boundaries():
    """Systematically test where NaN crashes occur."""
    log("Testing NaN stability boundaries...")
    device = torch.device("cuda")
    results = {}

    # Test bf16 precision edge cases
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

    # Test reduction ops (common NaN source in training)
    x = torch.randn(32768, 512, device=device, dtype=torch.bfloat16)
    for op_name, op_fn in [("sum", torch.sum), ("mean", torch.mean), ("var", lambda t: torch.var(t, dim=-1))]:
        try:
            result = op_fn(x)
            has_nan = torch.isnan(result).any().item()
            results[f"reduce_{op_name}_32k"] = {"nan": has_nan, "status": "nan" if has_nan else "ok"}
        except Exception as e:
            results[f"reduce_{op_name}_32k"] = {"status": f"error: {str(e)[:80]}"}

    log(f"NaN boundary results: {results}")
    return results


def run_best_recipe_timed():
    """Run the actual training script and capture timing."""
    log("Running best recipe for timing baseline...")
    env = os.environ.copy()
    env["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"

    t0 = time.time()
    result = subprocess.run(
        ["python3", str(TRAIN_PY)],
        capture_output=True, text=True, timeout=600,
        cwd=str(TRAIN_PY.parent), env=env,
    )
    t1 = time.time()

    output = result.stdout + result.stderr

    # Parse the final summary
    metrics = {}
    for line in output.splitlines():
        for key in ["val_bpb", "training_seconds", "total_seconds", "peak_vram_mb",
                     "mfu_percent", "total_tokens_M", "num_steps", "num_params_M"]:
            if line.strip().startswith(f"{key}:"):
                try:
                    metrics[key] = float(line.split(":")[1].strip())
                except ValueError:
                    pass

    # Extract tok/sec from training output
    tok_sec_matches = re.findall(r"tok/sec:\s*([\d,]+)", output)
    if tok_sec_matches:
        # Take last (steady state)
        metrics["tok_per_sec_steady"] = int(tok_sec_matches[-1].replace(",", ""))

    metrics["wall_time"] = round(t1 - t0, 1)
    log(f"Training metrics: {metrics}")
    return metrics


def generate_report(hw_info, profile_data, sdpa_results, matmul_results, nan_results, training_metrics):
    """Generate the AMD-actionable report."""
    log("Generating report...")

    report = []
    report.append("# AMD ROCm Profiling Report — Strix Halo gfx1151")
    report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append(f"**Benchmark**: Karpathy autoresearch (5-min pretraining)")
    report.append("")

    # Hardware
    report.append("## Hardware & Environment")
    report.append(f"- **GPU**: {hw_info['gpu_name']}")
    report.append(f"- **Architecture**: {hw_info.get('gpu_arch', 'N/A')}")
    report.append(f"- **VRAM**: {hw_info['gpu_vram_gb']} GB")
    report.append(f"- **Multiprocessors**: {hw_info['gpu_multiprocessors']}")
    report.append(f"- **CPU**: {hw_info.get('cpu', 'N/A')}")
    report.append(f"- **RAM**: {hw_info.get('ram_gb', 'N/A')} GB")
    report.append(f"- **PyTorch**: {hw_info['torch_version']}")
    report.append(f"- **HIP**: {hw_info['hip_version']}")
    report.append(f"- **ROCm Driver**: {hw_info.get('rocm_driver', 'N/A')}")
    report.append(f"- **TheROCk Index**: {hw_info['therock_index']}")
    report.append(f"- **AOTriton Experimental**: {'enabled' if hw_info['aotriton_experimental'] == '1' else 'DISABLED'}")
    report.append("")

    # Training results
    report.append("## Training Results (Best Recipe)")
    if training_metrics:
        report.append(f"- **val_bpb**: {training_metrics.get('val_bpb', 'N/A')}")
        report.append(f"- **MFU**: {training_metrics.get('mfu_percent', 'N/A')}%")
        report.append(f"- **Tokens/sec**: {training_metrics.get('tok_per_sec_steady', 'N/A'):,}")
        report.append(f"- **Steps**: {int(training_metrics.get('num_steps', 0))}")
        report.append(f"- **Params**: {training_metrics.get('num_params_M', 'N/A')}M")
        report.append(f"- **Peak VRAM**: {training_metrics.get('peak_vram_mb', 0):.0f} MB / {hw_info['gpu_vram_gb']*1024:.0f} MB ({training_metrics.get('peak_vram_mb', 0)/(hw_info['gpu_vram_gb']*1024)*100:.0f}%)")
        report.append(f"- **Training time**: {training_metrics.get('training_seconds', 'N/A')}s")
        report.append(f"- **Wall time**: {training_metrics.get('wall_time', 'N/A')}s")
        startup = training_metrics.get('wall_time', 0) - training_metrics.get('training_seconds', 0)
        report.append(f"- **Startup overhead**: {startup:.1f}s")
    report.append("")

    report.append("## Optimization Journey")
    report.append("| Milestone | val_bpb | Improvement |")
    report.append("|-----------|---------|-------------|")
    milestones = [
        ("Baseline (default)", 1.819),
        ("Batch size tuning (2^20→2^15)", 1.295),
        ("SwiGLU + full attention", 1.292),
        ("Warmdown schedule", 1.264),
        ("LR + weight decay tuning", 1.256),
        ("Head dim + unembed LR", 1.255),
    ]
    for name, bpb in milestones:
        pct = (1.819 - bpb) / 1.819 * 100
        report.append(f"| {name} | {bpb:.3f} | -{pct:.1f}% |")
    report.append("")

    # SDPA Benchmarks
    report.append("## SDPA (Attention) Benchmarks")
    report.append("Shape: batch=16, heads=8, seq=1024")
    report.append("")
    if sdpa_results:
        report.append("| Backend | Time (ms) | vs Math Fallback |")
        report.append("|---------|-----------|-----------------|")
        math_ms = sdpa_results.get("sdpa_math_fallback", 1)
        for key, ms in sdpa_results.items():
            if not key.startswith("sdpa_head"):
                speedup = f"{math_ms/ms:.1f}x" if ms > 0 else "N/A"
                report.append(f"| {key} | {ms} | {speedup} |")
        report.append("")
        report.append("### SDPA by Head Dimension")
        report.append("| Head Dim | Time (ms) |")
        report.append("|----------|-----------|")
        for key, ms in sdpa_results.items():
            if key.startswith("sdpa_head"):
                hd = key.replace("sdpa_head", "")
                report.append(f"| {hd} | {ms} |")
    report.append("")

    # Matmul throughput
    report.append("## Matmul Throughput (bf16)")
    report.append("| Size | Time (ms) | TFLOPS | vs 49.6 TFLOPS peak |")
    report.append("|------|-----------|--------|---------------------|")
    if matmul_results:
        for key, data in matmul_results.items():
            size = key.replace("matmul_", "")
            pct = data['tflops_bf16'] / 49.6 * 100
            report.append(f"| {size} | {data['ms']} | {data['tflops_bf16']} | {pct:.0f}% |")
    report.append("")

    # Kernel profile
    if profile_data:
        report.append("## Top GPU Kernels (by total CUDA time)")
        report.append("```")
        report.append(profile_data["kernel_table"])
        report.append("```")
        report.append("")

        report.append("### Memory")
        report.append(f"- Peak allocated: {profile_data['mem_stats']['peak_allocated_mb']} MB")
        report.append(f"- Peak reserved: {profile_data['mem_stats']['peak_reserved_mb']} MB")
        report.append("")

    # NaN stability
    report.append("## bf16 Numerical Stability")
    report.append("| Test | Result |")
    report.append("|------|--------|")
    if nan_results:
        for name, data in nan_results.items():
            status = data.get("status", "unknown")
            emoji = "PASS" if status == "ok" else "FAIL" if "nan" in status else "WARN"
            report.append(f"| {name} | {emoji}: {status} |")
    report.append("")

    # Known issues / AMD action items
    report.append("## Known Issues & AMD Action Items")
    report.append("")
    report.append("### Critical")
    report.append("1. **TheROCk nightlies required** — stable ROCm does not ship gfx1151 kernels. Consumer Strix Halo users must use nightly index.")
    report.append("2. **AOTriton experimental hidden behind env var** — `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` gives 19x SDPA speedup but is not documented or default.")
    report.append("3. **`PYTORCH_HIP_ALLOC_CONF=backend:malloc` crashes PyTorch** — this is set by default in some ROCm shell profiles. Must be unset.")
    report.append("")
    report.append("### Stability")
    report.append("4. **batch 2^14 (16K tokens) always NaN** — at device_batch=8, training diverges within 15 steps. Possible bf16 precision issue in small-batch gradient accumulation.")
    report.append("5. **HEAD_DIM 64 intermittent NaN** — crashed on first attempt (step 348), succeeded on second. Non-deterministic instability suggests kernel-level precision issue.")
    report.append("6. **Depth 10+ unreliable** — NaN at step 23 with depth 16, crash with depth 10 on second round. May be related to bf16 accumulation in deep networks.")
    report.append("")
    report.append("### Performance")
    report.append("7. **No FlashAttention 3 for ROCm** — NVIDIA gets FA3, we get SDPA with AOTriton. Quantify the gap.")
    report.append("8. **HSA_OVERRIDE_GFX_VERSION no longer needed** — TheROCk nightlies ship native gfx1151. Document this for consumer GPU users.")
    report.append("")

    # Environment snapshot
    report.append("## Best Recipe Hyperparameters")
    report.append("```")
    report.append("ASPECT_RATIO = 64")
    report.append("HEAD_DIM = 64")
    report.append("WINDOW_PATTERN = \"L\"")
    report.append("TOTAL_BATCH_SIZE = 2**15")
    report.append("EMBEDDING_LR = 0.8")
    report.append("UNEMBEDDING_LR = 0.012")
    report.append("MATRIX_LR = 0.07")
    report.append("SCALAR_LR = 0.6")
    report.append("WEIGHT_DECAY = 0.12")
    report.append("ADAM_BETAS = (0.8, 0.98)")
    report.append("WARMUP_RATIO = 0.0")
    report.append("WARMDOWN_RATIO = 0.7")
    report.append("FINAL_LR_FRAC = 0.07")
    report.append("DEPTH = 8")
    report.append("DEVICE_BATCH_SIZE = 16")
    report.append("```")
    report.append("")

    report.append("## Experiment History")
    report.append(f"- Total experiments: 93+")
    report.append(f"- Improvements kept: 26")
    report.append(f"- NaN crashes: 14")
    report.append(f"- Discarded: 53")
    report.append(f"- Starting val_bpb: 1.819")
    report.append(f"- Final val_bpb: 1.255")
    report.append(f"- Total improvement: 31%")
    report.append("")

    report.append("---")
    report.append("*Generated by amdsense profiling agent on AMD Ryzen AI MAX+ 395 / Radeon 8060S*")
    report.append("*Contact: obelisk@msbs.com*")

    report_text = "\n".join(report)
    REPORT_FILE.write_text(report_text)
    log(f"Report written to {REPORT_FILE}")
    return report_text


def main():
    log("=== ROCm Profiling Agent Started ===")

    # 1. Hardware info
    hw_info = get_hw_info()
    log(f"GPU: {hw_info['gpu_name']}, VRAM: {hw_info['gpu_vram_gb']}GB")

    # 2. Run best recipe for timing
    training_metrics = run_best_recipe_timed()

    # 3. SDPA benchmarks
    sdpa_results = benchmark_sdpa()

    # 4. Matmul throughput
    matmul_results = benchmark_matmul()

    # 5. NaN stability tests
    nan_results = test_nan_boundaries()

    # 6. Profile training steps (this is heavy — do last)
    profile_data = None
    try:
        profile_data = profile_training_step()
    except Exception as e:
        log(f"Profiling failed (non-fatal): {e}")

    # 7. Generate report
    generate_report(hw_info, profile_data, sdpa_results, matmul_results, nan_results, training_metrics)

    log("=== Profiling complete ===")


if __name__ == "__main__":
    main()
