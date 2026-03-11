# amdsense

Autonomous AI research on AMD hardware. Originally forked from [karpathy/autoresearch](https://github.com/karpathy/autoresearch), now an independent AMD-focused project.

An AI agent experiments with neural network training code autonomously — modifying architecture, hyperparameters, and optimization, training for 5-minute intervals, keeping improvements and discarding regressions. You wake up to a log of experiments and a better model.

**What's different here:** Native AMD ROCm support. No NVIDIA required. The training script auto-detects your GPU and uses PyTorch's native SDPA when Flash Attention 3 isn't available — zero performance compromise on AMD hardware.

## Quick start

**Requirements:** A single AMD GPU (ROCm) or NVIDIA GPU, Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install uv project manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 4. Run a single training experiment (~5 min)
uv run train.py
```

### AMD ROCm setup

Install [PyTorch for ROCm](https://pytorch.org/get-started/locally/) and run as normal. The SDPA fallback activates automatically when FA3 is unavailable.

For accurate MFU (Model FLOPs Utilization) reporting, set your GPU's peak BF16 FLOPS:

```bash
# Example for Radeon 8060S (Strix Halo)
export GPU_BF16_PEAK_FLOPS=100e12

# Example for MI300X
export GPU_BF16_PEAK_FLOPS=1307e12
```

The script auto-detects AMD devices and defaults to MI300X FLOPS if the env var isn't set.

## How it works

Three files that matter:

- **`prepare.py`** — data prep + runtime utilities (fixed, not modified)
- **`train.py`** — model, optimizer, training loop (agent modifies this)
- **`program.md`** — agent instructions (human modifies this)

Training runs for a **fixed 5-minute time budget** per experiment. The metric is **val_bpb** (validation bits per byte) — lower is better. Expect ~12 experiments/hour, ~100 overnight.

## Running the agent

Point your coding agent (Claude Code, Codex, etc.) at this repo and prompt:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

## AMD-specific changes from upstream

1. **SDPA fallback**: FA3 import wrapped in try/except. When unavailable, PyTorch's `scaled_dot_product_attention` handles attention with proper GQA support via `repeat_interleave`.

2. **GPU auto-detection**: Detects AMD devices by checking for "gfx", "amd", or "radeon" in device name. Sets appropriate peak FLOPS for MFU reporting.

3. **Zero NVIDIA breakage**: FA3 path is unchanged. NVIDIA users see identical behavior. AMD path only activates when FA3 is absent.

## Tested on

- AMD Radeon 8060S (Strix Halo, gfx1151) with ROCm 7.2
- SDPA verified with standard and GQA tensor configurations

## Smaller hardware tips

If running on consumer GPUs or APUs with limited VRAM:

1. Use [TinyStories dataset](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean) for lower entropy data
2. Lower `vocab_size` (8192 → 4096 or 2048)
3. Decrease `MAX_SEQ_LEN` in `prepare.py` (down to 256 if needed)
4. Lower `DEPTH` in `train.py` (default 8, try 4)
5. Use `WINDOW_PATTERN = "L"` (avoid banded attention overhead)
6. Lower `TOTAL_BATCH_SIZE` (try `2**14`)

## Origin

Based on [karpathy/autoresearch](https://github.com/karpathy/autoresearch) (MIT). We submitted AMD support upstream ([PR #136](https://github.com/karpathy/autoresearch/pull/136)) — they preferred to keep the parent repo CUDA-only. So we made our own.

## License

MIT
