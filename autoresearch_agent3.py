#!/usr/bin/env python3
"""
Autoresearch Agent Round 3 — Confirmation + Ablation.
Beth's framework layers 2 & 3.

Layer 2: Run best recipe 5 times to establish variance band.
Layer 3: Ablate each change from best recipe one at a time to prove WHY each helps.

This produces AMD-grade evidence, not one-off miracles.

Usage:
    source venv/bin/activate
    unset PYTORCH_HIP_ALLOC_CONF
    export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
    export GPU_BF16_PEAK_FLOPS=49.6e12
    python3 autoresearch_agent3.py
"""

import os
import re
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime

TRAIN_PY = Path(__file__).parent / "train.py"
RESULTS_DIR = Path(__file__).parent / "round3_results"
LOG_FILE = Path(__file__).parent / "agent3.log"
REPORT_FILE = Path(__file__).parent / "round3_report.md"

RESULTS_DIR.mkdir(exist_ok=True)


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
    """Set a param to a new value, preserving comment."""
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


def run_training():
    """Run train.py and extract all metrics."""
    env = os.environ.copy()
    env["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"
    if "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL" not in env:
        env["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

    try:
        t0 = time.time()
        result = subprocess.run(
            ["python3", str(TRAIN_PY)],
            capture_output=True, text=True, timeout=600,
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


def run_confirmation(baseline_content, n_runs=5):
    """Layer 2: Run best recipe N times, measure variance."""
    log(f"=== LAYER 2: CONFIRMATION ({n_runs} runs) ===")
    results = []

    for i in range(n_runs):
        log(f"Confirmation run {i+1}/{n_runs}...")
        write_train_py(baseline_content)
        metrics = run_training()
        bpb = metrics.get("val_bpb", 0)
        mfu = metrics.get("mfu_percent", 0)
        tok = metrics.get("tok_per_sec", 0)
        log(f"  Run {i+1}: val_bpb={bpb:.6f}, MFU={mfu:.2f}%, tok/s={tok}")
        results.append(metrics)
        time.sleep(3)

    # Stats
    bpbs = [r["val_bpb"] for r in results if "val_bpb" in r]
    if bpbs:
        mean_bpb = sum(bpbs) / len(bpbs)
        min_bpb = min(bpbs)
        max_bpb = max(bpbs)
        spread = max_bpb - min_bpb
        log(f"Confirmation: mean={mean_bpb:.6f}, min={min_bpb:.6f}, max={max_bpb:.6f}, spread={spread:.6f}")

    # Save
    with open(RESULTS_DIR / "confirmation.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def run_ablations(baseline_content):
    """Layer 3: Remove each optimization one at a time from the best recipe.
    Proves which changes actually matter by reverting to the default value."""
    log("=== LAYER 3: ABLATION RUNS ===")

    # What we changed from defaults and what the defaults were
    # Format: (name, param, default_value, description)
    ablations = [
        ("Revert HEAD_DIM 64→128", "HEAD_DIM", "128", "head dim back to default 128"),
        ("Revert EMBEDDING_LR 0.8→1.0", "EMBEDDING_LR", "1.0", "embedding LR back to default"),
        ("Revert UNEMBEDDING_LR 0.012→0.008", "UNEMBEDDING_LR", "0.008", "unembed LR back to default"),
        ("Revert MATRIX_LR 0.07→0.08", "MATRIX_LR", "0.08", "matrix LR back to original"),
        ("Revert SCALAR_LR 0.6→0.5", "SCALAR_LR", "0.5", "scalar LR back to default"),
        ("Revert WEIGHT_DECAY 0.12→0.2", "WEIGHT_DECAY", "0.2", "weight decay back to default"),
        ("Revert ADAM_BETAS to (0.8, 0.95)", "ADAM_BETAS", "(0.8, 0.95)", "adam betas back to default"),
        ("Revert WARMDOWN_RATIO 0.7→0.3", "WARMDOWN_RATIO", "0.3", "warmdown back to original default"),
        ("Revert FINAL_LR_FRAC 0.07→0.0", "FINAL_LR_FRAC", "0.0", "final LR frac back to zero"),
        ("Revert WARMDOWN to 0.5", "WARMDOWN_RATIO", "0.5", "warmdown to pre-optimization"),
    ]

    # First, run baseline once for comparison
    log("Running best recipe as ablation baseline...")
    write_train_py(baseline_content)
    baseline_metrics = run_training()
    baseline_bpb = baseline_metrics.get("val_bpb", 0)
    log(f"Ablation baseline: val_bpb={baseline_bpb:.6f}")

    results = {"baseline": baseline_metrics, "ablations": []}

    for name, param, default_val, desc in ablations:
        log(f"Ablation: {name}...")
        content = baseline_content
        content, ok = set_param(content, param, default_val)
        if not ok:
            log(f"  SKIP: couldn't set {param}={default_val}")
            continue

        write_train_py(content)
        metrics = run_training()
        bpb = metrics.get("val_bpb", 0)

        if bpb > 0:
            delta = bpb - baseline_bpb
            pct = delta / baseline_bpb * 100
            impact = "CRITICAL" if pct > 1.0 else "significant" if pct > 0.3 else "minor" if pct > 0 else "neutral/better"
            log(f"  {name}: val_bpb={bpb:.6f} (delta={delta:+.6f}, {pct:+.2f}%) — {impact}")
        else:
            impact = "CRASH"
            delta = 0
            pct = 0
            log(f"  {name}: CRASHED — this optimization prevents instability")

        results["ablations"].append({
            "name": name,
            "param": param,
            "reverted_to": default_val,
            "description": desc,
            "metrics": metrics,
            "delta_bpb": round(delta, 6) if bpb > 0 else None,
            "delta_pct": round(pct, 2) if bpb > 0 else None,
            "impact": impact,
        })

        # Restore baseline
        write_train_py(baseline_content)
        time.sleep(3)

    # Save
    with open(RESULTS_DIR / "ablations.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def run_failure_boundaries(baseline_content):
    """Layer 4 (partial): Systematically map NaN crash boundaries."""
    log("=== LAYER 4: FAILURE BOUNDARY MAPPING ===")

    # Test specific configurations known to be unstable
    boundary_tests = [
        ("batch 2^14 device_batch 8", {"TOTAL_BATCH_SIZE": "2**14", "DEVICE_BATCH_SIZE": "8"}),
        ("batch 2^14 device_batch 16", {"TOTAL_BATCH_SIZE": "2**14", "DEVICE_BATCH_SIZE": "16"}),
        ("batch 2^13 device_batch 8", {"TOTAL_BATCH_SIZE": "2**13", "DEVICE_BATCH_SIZE": "8"}),
        ("depth 10", {"DEPTH": "10"}),
        ("depth 12", {"DEPTH": "12"}),
        ("depth 16", {"DEPTH": "16"}),
        ("head_dim 32", {"HEAD_DIM": "32"}),
        ("head_dim 256", {"HEAD_DIM": "256"}),
        ("matrix LR 0.15 (high)", {"MATRIX_LR": "0.15"}),
        ("matrix LR 0.20 (very high)", {"MATRIX_LR": "0.20"}),
        ("weight decay 0.0 (none)", {"WEIGHT_DECAY": "0.0"}),
        ("weight decay 0.5 (heavy)", {"WEIGHT_DECAY": "0.5"}),
        ("warmdown 0.9", {"WARMDOWN_RATIO": "0.9"}),
        ("warmdown 0.95", {"WARMDOWN_RATIO": "0.95"}),
        ("embed LR 2.0 (high)", {"EMBEDDING_LR": "2.0"}),
        ("embed LR 0.1 (low)", {"EMBEDDING_LR": "0.1"}),
        ("aspect 128 (wide)", {"ASPECT_RATIO": "128"}),
        ("aspect 32 (narrow)", {"ASPECT_RATIO": "32"}),
    ]

    results = []
    for name, changes in boundary_tests:
        log(f"Boundary test: {name}...")
        content = baseline_content
        ok = True
        for param, val in changes.items():
            content, success = set_param(content, param, val)
            if not success:
                ok = False
                break

        if not ok:
            log(f"  SKIP: couldn't apply changes")
            continue

        write_train_py(content)
        metrics = run_training()
        bpb = metrics.get("val_bpb", 0)
        status = "ok" if bpb > 0 else "CRASH"
        log(f"  {name}: {status} (val_bpb={bpb:.6f})")

        results.append({
            "name": name,
            "changes": changes,
            "metrics": metrics,
            "status": status,
        })

        write_train_py(baseline_content)
        time.sleep(3)

    with open(RESULTS_DIR / "failure_boundaries.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def generate_round3_report(confirmation, ablations, boundaries):
    """Generate the round 3 report."""
    log("Generating round 3 report...")
    r = []
    r.append("# Amdsense Round 3 Report — Confirmation, Ablation & Failure Boundaries")
    r.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    r.append(f"**Hardware**: AMD Ryzen AI MAX+ 395 / Radeon 8060S (gfx1151)")
    r.append("")

    # Confirmation
    r.append("## Layer 2: Confirmation Runs")
    r.append("*Same recipe, 5 runs. Establishes variance band.*")
    r.append("")
    bpbs = [c.get("val_bpb", 0) for c in confirmation if "val_bpb" in c]
    mfus = [c.get("mfu_percent", 0) for c in confirmation if "mfu_percent" in c]
    toks = [c.get("tok_per_sec", 0) for c in confirmation if "tok_per_sec" in c]

    if bpbs:
        mean_bpb = sum(bpbs) / len(bpbs)
        r.append(f"| Metric | Mean | Min | Max | Spread |")
        r.append(f"|--------|------|-----|-----|--------|")
        r.append(f"| val_bpb | {mean_bpb:.6f} | {min(bpbs):.6f} | {max(bpbs):.6f} | {max(bpbs)-min(bpbs):.6f} |")
        if mfus:
            r.append(f"| MFU % | {sum(mfus)/len(mfus):.2f} | {min(mfus):.2f} | {max(mfus):.2f} | {max(mfus)-min(mfus):.2f} |")
        if toks:
            r.append(f"| tok/sec | {sum(toks)//len(toks):,} | {min(toks):,} | {max(toks):,} | {max(toks)-min(toks):,} |")
        r.append("")
        r.append(f"**Variance band**: {max(bpbs)-min(bpbs):.6f} — any improvement smaller than this is noise.")
    r.append("")

    # Ablation
    r.append("## Layer 3: Ablation Runs")
    r.append("*Revert each optimization to default. Measures individual contribution.*")
    r.append("")

    if ablations and "ablations" in ablations:
        baseline_bpb = ablations["baseline"].get("val_bpb", 0)
        r.append(f"Best recipe baseline: **{baseline_bpb:.6f}**")
        r.append("")
        r.append("| Change | Reverted To | val_bpb | Delta | Impact |")
        r.append("|--------|-------------|---------|-------|--------|")

        # Sort by impact
        sorted_abl = sorted(ablations["ablations"],
                           key=lambda x: x.get("delta_bpb", 0) if x.get("delta_bpb") is not None else 999,
                           reverse=True)
        for a in sorted_abl:
            bpb = a["metrics"].get("val_bpb", 0)
            if a["delta_bpb"] is not None:
                r.append(f"| {a['name']} | {a['reverted_to']} | {bpb:.6f} | {a['delta_bpb']:+.6f} ({a['delta_pct']:+.2f}%) | {a['impact']} |")
            else:
                r.append(f"| {a['name']} | {a['reverted_to']} | CRASH | — | CRITICAL (stability) |")
    r.append("")

    # Failure boundaries
    r.append("## Layer 4: Failure Boundary Map")
    r.append("*Systematically test where training breaks.*")
    r.append("")
    r.append("| Configuration | Status | val_bpb | Notes |")
    r.append("|--------------|--------|---------|-------|")
    if boundaries:
        for b in boundaries:
            bpb = b["metrics"].get("val_bpb", 0)
            status = b["status"]
            changes_str = ", ".join(f"{k}={v}" for k, v in b["changes"].items())
            if status == "ok":
                r.append(f"| {b['name']} | OK | {bpb:.6f} | {changes_str} |")
            else:
                error = b["metrics"].get("error", "NaN/crash")
                r.append(f"| {b['name']} | CRASH | — | {error} |")
    r.append("")

    r.append("---")
    r.append("*Round 3: Confirmation + Ablation + Failure Boundaries*")
    r.append("*Next: Layer 5 (cross-hardware comparison on RunPod 4070)*")

    report_text = "\n".join(r)
    REPORT_FILE.write_text(report_text)
    log(f"Report written to {REPORT_FILE}")


def main():
    log("=== Autoresearch Agent Round 3 ===")
    log("Beth's framework: Layer 2 (confirmation) + Layer 3 (ablation) + Layer 4 (failure boundaries)")

    baseline_content = read_train_py()

    # Log current best params
    for p in ["HEAD_DIM", "EMBEDDING_LR", "UNEMBEDDING_LR", "MATRIX_LR", "SCALAR_LR",
              "WEIGHT_DECAY", "ADAM_BETAS", "WARMDOWN_RATIO", "FINAL_LR_FRAC", "DEPTH",
              "TOTAL_BATCH_SIZE", "ASPECT_RATIO", "DEVICE_BATCH_SIZE"]:
        log(f"  {p} = {get_current_value(baseline_content, p)}")

    # Layer 2: Confirmation
    confirmation = run_confirmation(baseline_content, n_runs=5)

    # Layer 3: Ablation
    ablations = run_ablations(baseline_content)

    # Layer 4: Failure boundaries
    boundaries = run_failure_boundaries(baseline_content)

    # Generate report
    generate_round3_report(confirmation, ablations, boundaries)

    # Restore best recipe
    write_train_py(baseline_content)

    log("=== Round 3 complete ===")


if __name__ == "__main__":
    main()
