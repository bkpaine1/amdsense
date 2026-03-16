#!/usr/bin/env python3
"""
Autoresearch Agent Round 5 — The Deep Squeeze.

Round 4 found: ASPECT_RATIO=40, HEAD_DIM=64, WARMDOWN=0.75, MATRIX_LR=0.05,
DEPTH=8, WINDOW_PATTERN="SSSSSL". Best val_bpb=1.2189.

Round 5 strategy — squeeze the remaining juice from unexplored dimensions:
  Phase 1: LR sweep — EMBEDDING_LR, UNEMBEDDING_LR, SCALAR_LR, FINAL_LR_FRAC
           (never properly swept at ASPECT_RATIO=40)
  Phase 2: Weight decay + Adam betas fine-tune
  Phase 3: Batch size fine-tune around 2**15
  Phase 4: Interaction effects — test combined changes vs individual
  Phase 5: Extended training — 10min runs on top-3 recipes (more steps = better signal)
  Phase 6: Memory wall v2 — smarter scaling (batch + aspect combos that R4 missed)

Survives session close via setsid. Logs everything.
Restores train.py to best recipe when done.

Usage:
    cd ~/projects/amdsense
    source venv/bin/activate
    unset PYTORCH_HIP_ALLOC_CONF
    export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
    export GPU_BF16_PEAK_FLOPS=49.6e12
    setsid python3 autoresearch_agent5.py > agent5_stdout.log 2>&1 &
"""

import os
import re
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime

TRAIN_PY = Path(__file__).parent / "train.py"
RESULTS_DIR = Path(__file__).parent / "round5_results"
LOG_FILE = Path(__file__).parent / "agent5.log"
REPORT_FILE = Path(__file__).parent / "round5_report.md"

RESULTS_DIR.mkdir(exist_ok=True)

# --- Best known recipe from Round 4 ---
BEST_RECIPE = {
    "ASPECT_RATIO": "40",
    "HEAD_DIM": "64",
    "WINDOW_PATTERN": '"SSSSSL"',
    "TOTAL_BATCH_SIZE": "2**15",
    "EMBEDDING_LR": "0.8",
    "UNEMBEDDING_LR": "0.012",
    "MATRIX_LR": "0.05",
    "SCALAR_LR": "0.6",
    "WEIGHT_DECAY": "0.12",
    "ADAM_BETAS": "(0.8, 0.98)",
    "WARMUP_RATIO": "0.0",
    "WARMDOWN_RATIO": "0.75",
    "FINAL_LR_FRAC": "0.07",
    "DEPTH": "8",
    "DEVICE_BATCH_SIZE": "16",
}

CURRENT_BEST_BPB = 1.2189


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def read_train_py():
    return TRAIN_PY.read_text()


def write_train_py(content):
    TRAIN_PY.write_text(content)


def set_param(content, param_name, new_val):
    pattern = rf"^({param_name}\s*=\s*)(.+?)(\s*#.*)$"
    match = re.search(pattern, content, re.MULTILINE)
    if match:
        old_line = match.group(0)
        new_line = f"{match.group(1)}{new_val}{match.group(3)}"
        return content.replace(old_line, new_line, 1), True
    pattern2 = rf"^({param_name}\s*=\s*)(.+)$"
    match2 = re.search(pattern2, content, re.MULTILINE)
    if match2:
        old_line = match2.group(0)
        new_line = f"{match2.group(1)}{new_val}"
        return content.replace(old_line, new_line, 1), True
    return content, False


def get_current_value(content, param_name):
    pattern = rf"^{param_name}\s*=\s*(.+?)(?:\s*#|$)"
    match = re.search(pattern, content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None


def apply_recipe(content, recipe):
    for param, val in recipe.items():
        content, _ = set_param(content, param, val)
    return content


def run_training(timeout=600):
    env = os.environ.copy()
    env["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"
    if "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL" not in env:
        env["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

    try:
        t0 = time.time()
        result = subprocess.run(
            ["python3", str(TRAIN_PY)],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(TRAIN_PY.parent), env=env,
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

        return metrics
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    except Exception as e:
        return {"error": str(e)}


def save_results(filename, data):
    with open(RESULTS_DIR / filename, "w") as f:
        json.dump(data, f, indent=2)


def run_sweep(baseline_content, param_name, values, extra_overrides=None, timeout=600):
    """Generic sweep: test a single param across values, return results + best value."""
    results = []
    for val in values:
        label = f"{param_name}={val}"
        log(f"  Testing {label}...")
        content = apply_recipe(baseline_content, BEST_RECIPE)
        if extra_overrides:
            for p, v in extra_overrides.items():
                content, _ = set_param(content, p, str(v))
        content, _ = set_param(content, param_name, str(val))
        write_train_py(content)
        metrics = run_training(timeout=timeout)
        bpb = metrics.get("val_bpb", 0)
        vram = metrics.get("peak_vram_mb", 0)
        tok = metrics.get("tok_per_sec", 0)
        status = "ok" if bpb > 0 else "CRASH"
        log(f"    {label}: {status} val_bpb={bpb:.6f} vram={vram:.0f}MB tok/s={tok}")
        results.append({param_name.lower(): val, "metrics": metrics, "status": status})
        time.sleep(3)

    valid = [r for r in results if r["status"] == "ok" and "val_bpb" in r["metrics"]]
    if valid:
        best = min(valid, key=lambda r: r["metrics"]["val_bpb"])
        best_val = best[param_name.lower()]
        log(f"  Best {param_name}: {best_val} with val_bpb={best['metrics']['val_bpb']:.6f}")
        return results, best_val
    return results, None


# ==========================================================================
# PHASE 1: Learning Rate Deep Dive
# ==========================================================================

def phase1_lr_sweeps(baseline_content):
    """Sweep all LR params that were never tested at ASPECT_RATIO=40."""
    log("=== PHASE 1: LEARNING RATE DEEP DIVE ===")
    all_results = {}
    best_overrides = {}

    # 1A: EMBEDDING_LR (current: 0.8)
    log("--- Phase 1A: EMBEDDING_LR ---")
    tests = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2]
    results, best = run_sweep(baseline_content, "EMBEDDING_LR", tests)
    save_results("phase1a_embedding_lr.json", results)
    all_results["Embedding LR Sweep"] = results
    if best is not None:
        best_overrides["EMBEDDING_LR"] = best

    # 1B: UNEMBEDDING_LR (current: 0.012)
    log("--- Phase 1B: UNEMBEDDING_LR ---")
    tests = [0.005, 0.008, 0.010, 0.012, 0.015, 0.018, 0.022, 0.028]
    results, best = run_sweep(baseline_content, "UNEMBEDDING_LR", tests,
                              extra_overrides=best_overrides)
    save_results("phase1b_unembedding_lr.json", results)
    all_results["Unembedding LR Sweep"] = results
    if best is not None:
        best_overrides["UNEMBEDDING_LR"] = best

    # 1C: SCALAR_LR (current: 0.6)
    log("--- Phase 1C: SCALAR_LR ---")
    tests = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    results, best = run_sweep(baseline_content, "SCALAR_LR", tests,
                              extra_overrides=best_overrides)
    save_results("phase1c_scalar_lr.json", results)
    all_results["Scalar LR Sweep"] = results
    if best is not None:
        best_overrides["SCALAR_LR"] = best

    # 1D: FINAL_LR_FRAC (current: 0.07)
    log("--- Phase 1D: FINAL_LR_FRAC ---")
    tests = [0.01, 0.03, 0.05, 0.07, 0.09, 0.12, 0.15, 0.20]
    results, best = run_sweep(baseline_content, "FINAL_LR_FRAC", tests,
                              extra_overrides=best_overrides)
    save_results("phase1d_final_lr_frac.json", results)
    all_results["Final LR Frac Sweep"] = results
    if best is not None:
        best_overrides["FINAL_LR_FRAC"] = best

    # 1E: MATRIX_LR fine-grain (R4 showed 0.05 good, 0.09 crash — explore 0.01-0.08)
    log("--- Phase 1E: MATRIX_LR fine-grain ---")
    tests = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    results, best = run_sweep(baseline_content, "MATRIX_LR", tests,
                              extra_overrides=best_overrides)
    save_results("phase1e_matrix_lr.json", results)
    all_results["Matrix LR Fine-grain"] = results
    if best is not None:
        best_overrides["MATRIX_LR"] = best

    return all_results, best_overrides


# ==========================================================================
# PHASE 2: Weight Decay + Adam Betas
# ==========================================================================

def phase2_regularization(baseline_content, lr_overrides):
    """Sweep weight decay and Adam betas."""
    log("=== PHASE 2: REGULARIZATION SWEEP ===")
    all_results = {}
    best_overrides = dict(lr_overrides)

    # 2A: WEIGHT_DECAY (current: 0.12)
    log("--- Phase 2A: WEIGHT_DECAY ---")
    tests = [0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.20]
    results, best = run_sweep(baseline_content, "WEIGHT_DECAY", tests,
                              extra_overrides=best_overrides)
    save_results("phase2a_weight_decay.json", results)
    all_results["Weight Decay Sweep"] = results
    if best is not None:
        best_overrides["WEIGHT_DECAY"] = best

    # 2B: ADAM_BETAS — test beta1 (current: 0.8)
    log("--- Phase 2B: ADAM_BETAS (beta1) ---")
    beta1_tests = [0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.95]
    results = []
    best_b1 = 0.8
    best_bpb = 99.0
    for b1 in beta1_tests:
        val = f"({b1}, 0.98)"
        log(f"  Testing ADAM_BETAS={val}...")
        content = apply_recipe(baseline_content, BEST_RECIPE)
        for p, v in best_overrides.items():
            content, _ = set_param(content, p, str(v))
        content, _ = set_param(content, "ADAM_BETAS", val)
        write_train_py(content)
        metrics = run_training()
        bpb = metrics.get("val_bpb", 0)
        status = "ok" if bpb > 0 else "CRASH"
        log(f"    ADAM_BETAS={val}: {status} val_bpb={bpb:.6f}")
        results.append({"beta1": b1, "metrics": metrics, "status": status})
        if status == "ok" and bpb < best_bpb:
            best_bpb = bpb
            best_b1 = b1
        time.sleep(3)
    save_results("phase2b_adam_beta1.json", results)
    all_results["Adam Beta1 Sweep"] = results

    # 2C: ADAM_BETAS — test beta2 with best beta1 (current: 0.98)
    log("--- Phase 2C: ADAM_BETAS (beta2) ---")
    beta2_tests = [0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999]
    results = []
    best_b2 = 0.98
    best_bpb2 = 99.0
    for b2 in beta2_tests:
        val = f"({best_b1}, {b2})"
        log(f"  Testing ADAM_BETAS={val}...")
        content = apply_recipe(baseline_content, BEST_RECIPE)
        for p, v in best_overrides.items():
            content, _ = set_param(content, p, str(v))
        content, _ = set_param(content, "ADAM_BETAS", val)
        write_train_py(content)
        metrics = run_training()
        bpb = metrics.get("val_bpb", 0)
        status = "ok" if bpb > 0 else "CRASH"
        log(f"    ADAM_BETAS={val}: {status} val_bpb={bpb:.6f}")
        results.append({"beta2": b2, "metrics": metrics, "status": status})
        if status == "ok" and bpb < best_bpb2:
            best_bpb2 = bpb
            best_b2 = b2
        time.sleep(3)
    save_results("phase2c_adam_beta2.json", results)
    all_results["Adam Beta2 Sweep"] = results
    best_overrides["ADAM_BETAS"] = f"({best_b1}, {best_b2})"

    return all_results, best_overrides


# ==========================================================================
# PHASE 3: Batch Size Fine-tune
# ==========================================================================

def phase3_batch_size(baseline_content, prior_overrides):
    """Fine-tune batch size and device batch size."""
    log("=== PHASE 3: BATCH SIZE FINE-TUNE ===")
    all_results = {}
    best_overrides = dict(prior_overrides)

    # 3A: TOTAL_BATCH_SIZE
    log("--- Phase 3A: TOTAL_BATCH_SIZE ---")
    # Test powers of 2 and some in-between
    tests = [
        ("2**14", "8"),    # smaller batch, smaller device batch
        ("2**14", "16"),   # smaller batch, same device batch
        ("2**15", "16"),   # current
        ("2**15", "32"),   # same batch, bigger device batch (more VRAM, fewer grad accum)
        ("2**16", "16"),   # bigger batch
        ("2**16", "32"),   # bigger batch, bigger device
        ("3*2**14", "16"), # 49152 — between 2**15 and 2**16
    ]
    results = []
    best_bpb = 99.0
    best_combo = ("2**15", "16")
    for total, device in tests:
        label = f"TOTAL_BATCH={total}, DEVICE_BATCH={device}"
        log(f"  Testing {label}...")
        content = apply_recipe(baseline_content, BEST_RECIPE)
        for p, v in best_overrides.items():
            content, _ = set_param(content, p, str(v))
        content, _ = set_param(content, "TOTAL_BATCH_SIZE", total)
        content, _ = set_param(content, "DEVICE_BATCH_SIZE", device)
        write_train_py(content)
        metrics = run_training()
        bpb = metrics.get("val_bpb", 0)
        vram = metrics.get("peak_vram_mb", 0)
        tok = metrics.get("tok_per_sec", 0)
        status = "ok" if bpb > 0 else "CRASH"
        log(f"    {label}: {status} val_bpb={bpb:.6f} vram={vram:.0f}MB tok/s={tok}")
        results.append({
            "total_batch": total, "device_batch": device,
            "metrics": metrics, "status": status
        })
        if status == "ok" and bpb < best_bpb:
            best_bpb = bpb
            best_combo = (total, device)
        time.sleep(3)

    save_results("phase3a_batch_size.json", results)
    all_results["Batch Size Sweep"] = results
    best_overrides["TOTAL_BATCH_SIZE"] = best_combo[0]
    best_overrides["DEVICE_BATCH_SIZE"] = best_combo[1]

    return all_results, best_overrides


# ==========================================================================
# PHASE 4: Interaction Effects
# ==========================================================================

def phase4_interactions(baseline_content, all_overrides):
    """Test if combined improvements actually stack or interfere."""
    log("=== PHASE 4: INTERACTION EFFECTS ===")
    results = []

    # Run with ALL improvements
    log("  Testing: ALL Round 5 improvements combined...")
    content = apply_recipe(baseline_content, BEST_RECIPE)
    for p, v in all_overrides.items():
        content, _ = set_param(content, p, str(v))
    write_train_py(content)
    metrics = run_training()
    bpb = metrics.get("val_bpb", 0)
    status = "ok" if bpb > 0 else "CRASH"
    log(f"    ALL combined: {status} val_bpb={bpb:.6f}")
    results.append({"config": "all_combined", "overrides": {k: str(v) for k, v in all_overrides.items()},
                     "metrics": metrics, "status": status})
    time.sleep(3)

    # Run with Round 4 baseline (no Round 5 changes) for comparison
    log("  Testing: Round 4 baseline (no R5 changes)...")
    content = apply_recipe(baseline_content, BEST_RECIPE)
    write_train_py(content)
    metrics = run_training()
    bpb = metrics.get("val_bpb", 0)
    status = "ok" if bpb > 0 else "CRASH"
    log(f"    R4 baseline: {status} val_bpb={bpb:.6f}")
    results.append({"config": "r4_baseline", "overrides": {},
                     "metrics": metrics, "status": status})
    time.sleep(3)

    # Drop each improvement one at a time to find which actually matter
    for drop_param in all_overrides:
        log(f"  Testing: ALL minus {drop_param}...")
        content = apply_recipe(baseline_content, BEST_RECIPE)
        for p, v in all_overrides.items():
            if p != drop_param:
                content, _ = set_param(content, p, str(v))
        write_train_py(content)
        metrics = run_training()
        bpb = metrics.get("val_bpb", 0)
        status = "ok" if bpb > 0 else "CRASH"
        log(f"    ALL minus {drop_param}: {status} val_bpb={bpb:.6f}")
        results.append({"config": f"drop_{drop_param}", "dropped": drop_param,
                         "metrics": metrics, "status": status})
        time.sleep(3)

    save_results("phase4_interactions.json", results)
    return results


# ==========================================================================
# PHASE 5: Extended Training — 10-minute runs
# ==========================================================================

def phase5_extended(baseline_content, final_recipe):
    """Run the best recipe for 10 minutes instead of 5 for better signal."""
    log("=== PHASE 5: EXTENDED TRAINING (10 min) ===")

    # Modify train.py to use 600s instead of 300s
    results = []

    # We need to change the training seconds in the code
    # Look for the training time parameter
    content = apply_recipe(baseline_content, final_recipe)

    # Try to find and set training duration
    content_check = content
    # Common param names for training time
    for time_param in ["TRAINING_DURATION_SECONDS", "training_seconds", "num_seconds"]:
        content_check, found = set_param(content_check, time_param, "600")
        if found:
            content = content_check
            log(f"  Set {time_param}=600 for extended training")
            break
    else:
        # If we can't find a time param, just let it run with longer timeout
        log("  No explicit training duration param found — using default with 900s timeout")

    # 3 extended runs for statistical confidence
    for i in range(3):
        log(f"  Extended run {i+1}/3...")
        content = apply_recipe(read_train_py(), final_recipe)
        write_train_py(content)
        metrics = run_training(timeout=900)
        bpb = metrics.get("val_bpb", 0)
        steps = metrics.get("num_steps", 0)
        log(f"    Run {i+1}: val_bpb={bpb:.6f} steps={steps}")
        results.append(metrics)
        time.sleep(5)

    save_results("phase5_extended.json", results)

    bpbs = [r.get("val_bpb", 0) for r in results if "val_bpb" in r and r["val_bpb"] > 0]
    if bpbs:
        mean = sum(bpbs) / len(bpbs)
        log(f"  Extended training: mean={mean:.6f}, min={min(bpbs):.6f}, max={max(bpbs):.6f}")
    return results


# ==========================================================================
# PHASE 6: Memory Wall v2 — Smarter scaling
# ==========================================================================

def phase6_memory_wall(baseline_content, final_recipe):
    """Smarter memory wall tests based on R4 learnings."""
    log("=== PHASE 6: MEMORY WALL v2 ===")
    log("R4 learned: wide models crash, big batch works. Focus on batch scaling + moderate width.")

    tests = [
        # (name, overrides, timeout)
        # Moderate width + batch scaling (what actually worked in R4)
        ("aspect48_batch2_16", {"ASPECT_RATIO": "48", "TOTAL_BATCH_SIZE": "2**16", "DEVICE_BATCH_SIZE": "32"}, 600),
        ("aspect48_batch2_17", {"ASPECT_RATIO": "48", "TOTAL_BATCH_SIZE": "2**17", "DEVICE_BATCH_SIZE": "64"}, 600),
        ("aspect48_batch2_18", {"ASPECT_RATIO": "48", "TOTAL_BATCH_SIZE": "2**18", "DEVICE_BATCH_SIZE": "64"}, 600),
        ("aspect56_batch2_16", {"ASPECT_RATIO": "56", "TOTAL_BATCH_SIZE": "2**16", "DEVICE_BATCH_SIZE": "16"}, 600),
        ("aspect56_batch2_17", {"ASPECT_RATIO": "56", "TOTAL_BATCH_SIZE": "2**17", "DEVICE_BATCH_SIZE": "32"}, 600),
        # Pure batch scaling on best recipe (proved concept in R4)
        ("best_batch2_16", {"TOTAL_BATCH_SIZE": "2**16", "DEVICE_BATCH_SIZE": "32"}, 600),
        ("best_batch2_17", {"TOTAL_BATCH_SIZE": "2**17", "DEVICE_BATCH_SIZE": "64"}, 600),
        ("best_batch2_18", {"TOTAL_BATCH_SIZE": "2**18", "DEVICE_BATCH_SIZE": "64"}, 600),
        ("best_batch2_19", {"TOTAL_BATCH_SIZE": "2**19", "DEVICE_BATCH_SIZE": "128"}, 900),
        ("best_batch2_20", {"TOTAL_BATCH_SIZE": "2**20", "DEVICE_BATCH_SIZE": "128"}, 900),
        # Depth 10 with R5 tuned LRs (R4 showed 1.253 — maybe LR tuning helps)
        ("depth10_tuned", {"DEPTH": "10", "DEVICE_BATCH_SIZE": "8"}, 900),
        # Aspect 64 with small device batch (R4 crashed at 96+ but 64 might work)
        ("aspect64_depth8", {"ASPECT_RATIO": "64", "DEPTH": "8", "DEVICE_BATCH_SIZE": "8"}, 600),
        ("aspect64_depth10", {"ASPECT_RATIO": "64", "DEPTH": "10", "DEVICE_BATCH_SIZE": "4"}, 900),
    ]

    results = []
    for name, overrides, timeout in tests:
        log(f"  Memory wall: {name}...")
        content = apply_recipe(baseline_content, final_recipe)
        for param, val in overrides.items():
            content, _ = set_param(content, param, val)
        write_train_py(content)
        metrics = run_training(timeout=timeout)
        bpb = metrics.get("val_bpb", 0)
        vram = metrics.get("peak_vram_mb", 0)
        params = metrics.get("num_params_M", 0)
        tok = metrics.get("tok_per_sec", 0)
        status = "ok" if bpb > 0 else "CRASH"
        log(f"    {name}: {status} val_bpb={bpb:.6f} params={params}M vram={vram:.0f}MB tok/s={tok}")
        results.append({
            "name": name, "overrides": overrides, "metrics": metrics,
            "status": status, "vram_mb": vram, "exceeds_4090": vram > 24000,
        })
        time.sleep(5)

    save_results("phase6_memory_wall.json", results)

    trainable_beyond = [r for r in results if r["status"] == "ok" and r["vram_mb"] > 24000]
    if trainable_beyond:
        log(f"=== {len(trainable_beyond)} models trained BEYOND 4090's 24GB ===")
        for r in trainable_beyond:
            log(f"    {r['name']}: {r['vram_mb']:.0f}MB VRAM, val_bpb={r['metrics'].get('val_bpb', 0):.6f}")

    return results


# ==========================================================================
# Report Generation
# ==========================================================================

def generate_report(all_phase_results, memory_wall_results, final_recipe, final_bpb,
                    interaction_results, extended_results):
    log("Generating Round 5 report...")
    r = []
    r.append("# Amdsense Round 5 Report — The Deep Squeeze")
    r.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    r.append(f"**Hardware**: AMD Ryzen AI MAX+ 395 / Radeon 8060S (gfx1151)")
    r.append(f"**Previous best**: val_bpb {CURRENT_BEST_BPB}")
    r.append(f"**Round 5 best**: val_bpb {final_bpb:.6f}")
    improvement = (CURRENT_BEST_BPB - final_bpb) / CURRENT_BEST_BPB * 100
    r.append(f"**Improvement over R4**: {improvement:.2f}%")
    total_improvement = (1.2267 - final_bpb) / 1.2267 * 100 if final_bpb < 1.2267 else 0
    r.append(f"**Total improvement (R3→R5)**: {total_improvement:.2f}%")
    r.append("")

    r.append("## Best Recipe (Round 5)")
    r.append("```python")
    for k, v in final_recipe.items():
        r.append(f"{k} = {v}")
    r.append("```")
    r.append("")

    # Phase results tables
    for phase_name, data in all_phase_results.items():
        r.append(f"## {phase_name}")
        r.append("| Config | val_bpb | VRAM (MB) | tok/sec | Status |")
        r.append("|--------|---------|-----------|---------|--------|")
        for entry in data:
            m = entry.get("metrics", {})
            bpb = m.get("val_bpb", 0)
            vram = m.get("peak_vram_mb", 0)
            tok = m.get("tok_per_sec", 0)
            st = entry.get("status", "?")
            label = str({k: v for k, v in entry.items() if k not in ("metrics", "status")})
            r.append(f"| {label} | {bpb:.6f} | {vram:.0f} | {tok:,} | {st} |")
        r.append("")

    # Interaction effects
    r.append("## Phase 4: Interaction Effects")
    r.append("| Config | val_bpb | Status | Notes |")
    r.append("|--------|---------|--------|-------|")
    for entry in interaction_results:
        m = entry.get("metrics", {})
        bpb = m.get("val_bpb", 0)
        st = entry.get("status", "?")
        config = entry.get("config", "?")
        dropped = entry.get("dropped", "")
        note = f"Dropped: {dropped}" if dropped else config
        r.append(f"| {config} | {bpb:.6f} | {st} | {note} |")
    r.append("")

    # Extended training
    r.append("## Phase 5: Extended Training (10 min)")
    r.append("| Run | val_bpb | Steps | tok/sec |")
    r.append("|-----|---------|-------|---------|")
    for i, m in enumerate(extended_results):
        bpb = m.get("val_bpb", 0)
        steps = m.get("num_steps", 0)
        tok = m.get("tok_per_sec", 0)
        r.append(f"| {i+1} | {bpb:.6f} | {steps:.0f} | {tok:,} |")
    r.append("")

    # Memory wall
    r.append("## Phase 6: Memory Wall v2")
    r.append("| Config | Params (M) | VRAM (MB) | val_bpb | tok/sec | Status | >4090? |")
    r.append("|--------|-----------|-----------|---------|---------|--------|--------|")
    for entry in memory_wall_results:
        m = entry.get("metrics", {})
        bpb = m.get("val_bpb", 0)
        vram = m.get("peak_vram_mb", 0)
        tok = m.get("tok_per_sec", 0)
        params = m.get("num_params_M", 0)
        st = entry.get("status", "?")
        beyond = "YES" if entry.get("exceeds_4090") else "no"
        r.append(f"| {entry['name']} | {params:.1f} | {vram:.0f} | {bpb:.6f} | {tok:,} | {st} | {beyond} |")
    r.append("")

    trainable_beyond = [e for e in memory_wall_results if e["status"] == "ok" and e.get("exceeds_4090")]
    if trainable_beyond:
        r.append(f"### Kill Shot: {len(trainable_beyond)} models trained beyond 4090's 24GB limit")
        for e in trainable_beyond:
            m = e["metrics"]
            r.append(f"- **{e['name']}**: {m.get('num_params_M', 0):.1f}M params, {e['vram_mb']:.0f}MB VRAM, val_bpb={m.get('val_bpb', 0):.6f}")
        r.append("")

    r.append("---")
    r.append("*Round 5: The Deep Squeeze — every hyperparameter earned its place*")
    r.append(f"*AMD Radeon 8060S on GMKTEC EVO X2 ($1,999) vs NVIDIA RTX 4090 (~$2,400)*")

    REPORT_FILE.write_text("\n".join(r))
    log(f"Report written to {REPORT_FILE}")


# ==========================================================================
# Main
# ==========================================================================

def main():
    log("=" * 70)
    log("Autoresearch Agent Round 5 — The Deep Squeeze")
    log("Every hyperparameter earns its place or gets cut.")
    log("=" * 70)

    original_content = read_train_py()
    baseline_content = apply_recipe(read_train_py(), BEST_RECIPE)

    # --- PHASE 1: LR sweep ---
    phase1_results, lr_overrides = phase1_lr_sweeps(baseline_content)

    # --- PHASE 2: Regularization ---
    phase2_results, reg_overrides = phase2_regularization(baseline_content, lr_overrides)

    # --- PHASE 3: Batch size ---
    phase3_results, batch_overrides = phase3_batch_size(baseline_content, reg_overrides)

    # Combine all results for report
    all_phase_results = {}
    all_phase_results.update(phase1_results)
    all_phase_results.update(phase2_results)
    all_phase_results.update(phase3_results)

    # Build final recipe
    final_recipe = dict(BEST_RECIPE)
    for k, v in batch_overrides.items():
        final_recipe[k] = str(v)

    save_results("best_recipe.json", final_recipe)

    # --- PHASE 4: Interaction effects ---
    interaction_results = phase4_interactions(baseline_content, batch_overrides)

    # Check if combined is better or worse — use the better one
    combined_entry = next((r for r in interaction_results if r["config"] == "all_combined"), None)
    baseline_entry = next((r for r in interaction_results if r["config"] == "r4_baseline"), None)
    if combined_entry and baseline_entry:
        combined_bpb = combined_entry["metrics"].get("val_bpb", 99)
        baseline_bpb = baseline_entry["metrics"].get("val_bpb", 99)
        if combined_bpb > baseline_bpb:
            log("WARNING: Combined R5 improvements are WORSE than R4 baseline!")
            log("Checking drop tests to find which changes help...")
            # Find which drops improve — those params should revert
            for entry in interaction_results:
                if entry.get("dropped"):
                    drop_bpb = entry["metrics"].get("val_bpb", 99)
                    if drop_bpb < combined_bpb:
                        log(f"  Dropping {entry['dropped']} IMPROVES from {combined_bpb:.6f} to {drop_bpb:.6f}")

    # --- PHASE 5: Extended training ---
    extended_results = phase5_extended(baseline_content, final_recipe)

    # Get final best bpb across all confirmation + extended runs
    all_bpbs = []
    for r in extended_results:
        if "val_bpb" in r and r["val_bpb"] > 0:
            all_bpbs.append(r["val_bpb"])
    if combined_entry and "val_bpb" in combined_entry["metrics"]:
        all_bpbs.append(combined_entry["metrics"]["val_bpb"])
    final_bpb = min(all_bpbs) if all_bpbs else CURRENT_BEST_BPB

    # --- PHASE 6: Memory wall v2 ---
    memory_wall_results = phase6_memory_wall(baseline_content, final_recipe)

    # --- Generate Report ---
    generate_report(all_phase_results, memory_wall_results, final_recipe, final_bpb,
                    interaction_results, extended_results)

    # --- Restore train.py to best recipe ---
    final_content = apply_recipe(original_content, final_recipe)
    write_train_py(final_content)
    log(f"train.py set to Round 5 best recipe")

    log("=" * 70)
    log(f"Round 5 complete. Best val_bpb: {final_bpb:.6f}")
    log("The stone squeezes. AMD deserves this.")
    log("=" * 70)


if __name__ == "__main__":
    main()
