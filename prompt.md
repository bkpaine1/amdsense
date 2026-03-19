# Strix Halo Autonomous Research & Diagnostics

This is a multi-workstream prompt for autonomous operation on an AMD Strix Halo (gfx1151) system. Read this file, then execute all workstreams. If you crash, re-read this file and resume from where you left off using results.tsv and findings.md as state.

## Workstream 1: Autoresearch Experiment Loop

Read `program.md` for full instructions. Key points:

- **Branch**: Use `autoresearch/<tag>` (create fresh from master)
- **Goal**: Minimize val_bpb by modifying only `train.py`
- **Loop**: Edit → commit → `uv run train.py > run.log 2>&1` → check results → keep/discard → repeat
- **Budget**: 5 min per experiment, ~12 experiments/hour
- **Never stop**: Run indefinitely until manually interrupted
- **Track**: Log every experiment to `results.tsv` (tab-separated)
- **Crash recovery**: If you crash mid-loop, read `results.tsv` to find your last experiment, check `git log` for current state, and continue

## Workstream 2: ROCm Source Diagnostics

The ROCm source code is cloned at `~/proj/ROCm`. Use it to root-cause known Strix Halo (gfx1151) bugs:

1. **bf16 accumulation at small batch sizes** — TOTAL_BATCH_SIZE=2^13 always NaN/crash
2. **bf16 NaN at small head dimensions** — HEAD_DIM=32 produces NaN
3. **Deep network instability** — DEPTH=12+ timeout/crash
4. **Wide aspect ratio crash** — ASPECT_RATIO=128 timeout
5. **Matrix LR cliff** — MATRIX_LR=0.15 works, 0.20 is dead
6. **Adam beta2 < 0.97 NaN** — beta2=0.95/0.96 crash on bf16
7. **bf16 degradation over extended training** — NaN after ~1000 steps in 10-min runs

For each issue:
- Find the relevant ROCm/HIP/AOTriton/MIOpen source paths
- Identify the bf16 accumulation or precision code
- Diff between nightly releases to find relevant changes
- Write detailed findings with file paths, line numbers, and root cause analysis

Output: `findings.md` with structured sections per bug.

## Workstream 3: Stress & Fuzz Testing

Systematically test every ROCm feature relevant to Strix Halo ML training:

- **bf16 matmul**: Various sizes (powers of 2, non-powers of 2, tall/wide/square)
- **SDPA/AOTriton attention**: HEAD_DIM sweep (16,32,48,64,96,128), seq lengths, batch sizes
- **Gradient accumulation**: Test accumulation precision at various step counts
- **Optimizer precision**: Adam/AdamW with various beta values, test for NaN
- **Memory allocation**: Large tensor allocation, fragmentation, OOM boundaries
- **torch.compile**: Compiled vs eager correctness comparison
- **Mixed precision**: bf16/fp16/fp32 interactions, autocast edge cases

Record all results with pass/fail status and any anomalies found.

## Crash Recovery Protocol

All workstreams maintain persistent state:
- **Experiments**: `results.tsv` + git history on experiment branch
- **Diagnostics**: `findings.md` (append-only, each section dated)
- **Tests**: Results logged to stdout/findings.md

On restart:
1. Read `results.tsv` — find last experiment number and best val_bpb
2. Read `findings.md` — find last completed diagnostic section
3. Check `git log --oneline -20` — verify branch state
4. Resume the appropriate workstream

## Parallelism

Use sub-agents for:
- ROCm source code exploration (multiple agents for different subsystems)
- Stress test execution (while experiments run on GPU)
- Root cause analysis across different codebases

The experiment loop must own the GPU exclusively — do not run stress tests while an experiment is active.
