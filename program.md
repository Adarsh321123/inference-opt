# LLM Inference Optimization via Autoresearch

An autonomous experiment loop that discovers optimal inference optimization strategies for LLMs,
accumulating transferable knowledge across models.

## Overview

You are an autonomous researcher optimizing LLM inference efficiency. Your goal: find the
optimization strategy (quantization, compression, configuration) that maximizes the **efficiency
score** — the best quality-speed-memory tradeoff. You run experiments in a loop, keep what works,
discard what doesn't, and **accumulate knowledge** that transfers to the next model.

The key insight: after optimizing Model 1, you should be FASTER at optimizing Model 2 because
of what you learned. By Model 3, you should converge on good strategies almost immediately.

## Setup

To set up a new experiment run:

1. **Read all in-scope files** for full context:
   - This `program.md` — your instructions
   - `knowledge.md` — accumulated insights from prior experiments (READ THIS CAREFULLY)
   - `evaluate.py` — the evaluation harness (do not modify)
   - `optimize.py` — the file you modify

2. **Install dependencies**: Run `uv sync` to install all packages from `pyproject.toml`.
   If `uv` is not installed: `curl -LsSf https://astral.sh/uv/install.sh | sh`

3. **Check GPU**: Run `nvidia-smi` to confirm GPU availability and VRAM.

4. **Run baseline**: Execute `uv run evaluate.py --baseline` to establish the FP16 baseline
   for the current model. This measures unoptimized quality, speed, and memory. The baseline
   numbers are written to `baseline.json` and used to compute the efficiency score.

5. **Initialize results.tsv** with just the header row. The baseline will be the first entry.

6. **Confirm and go**: Verify setup looks good, then begin the experiment loop.

## The Files

```
evaluate.py     — evaluation harness: loads model, applies optimization, benchmarks (DO NOT MODIFY)
optimize.py     — your optimization strategy (YOU MODIFY THIS)
knowledge.md    — accumulated insights (YOU UPDATE THIS every ~10 experiments)
results.tsv     — experiment log (append after each experiment, do not commit to git)
baseline.json   — FP16 baseline numbers (written by evaluate.py --baseline)
```

## The Fitness Function

The efficiency score combines quality, speed, and memory:

```
efficiency_score = quality_retained * speedup * memory_reduction

where:
  quality_retained = benchmark_score / baseline_score    (capped at 1.0)
  speedup          = tokens_per_sec / baseline_tps
  memory_reduction = baseline_vram_gb / peak_vram_gb

if quality_retained < 0.85:
  efficiency_score = 0    # hard quality floor — we don't want garbage models
```

**Interpretation**: An optimization that retains 95% quality, runs 2x faster, and uses half the
memory scores: 0.95 * 2.0 * 2.0 = 3.8 (vs baseline of 1.0). Higher is better.

## Evaluation Details

`evaluate.py` does the following (you don't need to understand the implementation, just the interface):

1. **Loads the base model** (FP16, from HuggingFace cache)
2. **Calls `optimize_model()`** from your `optimize.py` — this is the function you write
3. **Quality benchmark**: Runs a fast evaluation suite (subset of standard benchmarks, ~2 min)
4. **Speed test**: Generates tokens and measures throughput (tokens/sec)
5. **Memory measurement**: Records peak VRAM usage

Output format:
```
---
efficiency_score: 3.8000
quality_retained: 0.9500
speedup:          2.0000
memory_reduction: 2.0000
benchmark_score:  0.6800
baseline_score:   0.7158
tokens_per_sec:   85.3
baseline_tps:     42.7
peak_vram_gb:     8.2
baseline_vram_gb: 16.4
```

## What You Modify: optimize.py

`optimize.py` must export a function with this signature:

```python
def optimize_model(model_name: str, device: str = "cuda") -> tuple[Any, Any]:
    """
    Load and optimize a model for efficient inference.

    Args:
        model_name: HuggingFace model identifier (e.g., "meta-llama/Meta-Llama-3-8B")
        device: Target device

    Returns:
        (model, tokenizer) — the optimized model ready for inference
    """
```

Everything inside this function is fair game. You can:
- Use any quantization library (GPTQ, AWQ, bitsandbytes, HQQ, AQLM, custom)
- Set any bit-width, group size, or configuration
- Apply layer-wise mixed precision (different bits for different layers)
- Apply mathematical transformations before/after quantization
- Combine multiple techniques (quantize + prune + distill)
- Write entirely custom quantization logic
- Try things that aren't in any paper

**The only constraint**: the function must return a working (model, tokenizer) pair that can
generate text. If it crashes or produces garbage, that's a failed experiment.

## Experimentation

Each experiment targets a **single model** at a time. The model sequence:

1. **Model 1**: `meta-llama/Meta-Llama-3-8B` — establish methodology, build initial knowledge
2. **Model 2**: `mistralai/Mistral-7B-v0.3` — test knowledge transfer, refine strategies
3. **Model 3**: `microsoft/Phi-3-small-8k-instruct` — validate transfer to different architecture

For each model, run the baseline first, then iterate.

**Time budget**: Each experiment (optimization + evaluation) should complete in ~5 minutes.
If an approach takes longer than 15 minutes, kill it and log as timeout.

**Speed tips for fast iteration**:
- Use **HQQ** for rapid exploration: quantizes 7B in under 1 minute, no calibration data needed.
  Install via `pip install hqq`. Supports 1-8 bit, per-layer mixed precision.
- Use **bitsandbytes** NF4 for instant quantization (quantizes at load time, zero extra time).
- Save **GPTQ** and **AWQ** for validating promising strategies (they take 10-45 min to quantize
  but may produce better quality or faster inference via specialized kernels).
- Perplexity eval on WikiText-2 takes under 1 minute. This is your primary quality signal.
- Target: **8-15 experiments per hour** during fast exploration phases.

**Baselines to beat (Llama 3 8B, WikiText-2 perplexity, lower = better)**:
- FP16 baseline: 6.23
- bitsandbytes NF4: ~6.30 (best quality, slowest inference ~23 tok/s)
- AWQ 4-bit: ~6.35 (good quality, fast with Marlin kernel)
- GPTQ 4-bit: ~6.41 (fast inference ~42-52 tok/s via ExLlama kernel)
- GGUF Q4_K_M: ~6.38-6.41 (good ecosystem support, ~30-45 tok/s)

LOOP FOREVER (for each model):

1. **Read knowledge.md** — especially the General Principles and Transferable Techniques sections.
   On a new model, start with techniques that worked on previous models.
2. **Plan your experiment** — Based on accumulated knowledge, decide what to try.
   Prioritize: Transferable Techniques > Hypotheses to Test > Novel ideas.
3. **Modify optimize.py** with your experimental approach.
4. **git commit** — preserve the state.
5. **Run**: `uv run evaluate.py > run.log 2>&1`
6. **Read results**: `grep "^efficiency_score:\|^quality_retained:\|^speedup:\|^memory_reduction:" run.log`
7. **Handle failures**:
   - If grep is empty: the run crashed. `tail -n 50 run.log` to read the error.
   - Fix simple bugs (typos, imports) and re-run. If fundamentally broken, discard and move on.
8. **Log to results.tsv** (tab-separated):
   ```
   commit	model	efficiency_score	quality_retained	speedup	memory_reduction	peak_vram_gb	status	description
   ```
9. **Keep or discard**:
   - If efficiency_score improved → keep the commit (advance the branch)
   - If efficiency_score is equal or worse → `git reset --hard` to previous best
10. **Every ~10 experiments, UPDATE knowledge.md**:
    - What worked? What failed? What patterns emerged?
    - Abstract transferable insights under "General Principles"
    - Add model-specific findings under the model's section
    - Update "Hypotheses to Test" with new ideas
    - Move confirmed techniques to "Transferable Techniques"
    - Move confirmed failures to "Failed Approaches"

## Knowledge Accumulation (CRITICAL)

This is what makes this different from standard autoresearch. After every ~10 experiments,
you MUST update knowledge.md. This is not optional — it is the core of the methodology.

**What to write:**
- Concrete, specific findings: "GPTQ with group_size=64 on Llama 3 8B scores 0.95 quality
  but group_size=32 only improves to 0.955 while being 15% slower — not worth it"
- Abstracted principles: "For GQA architectures, the KV projection layers are most sensitive
  to quantization — keep them at higher precision"
- Failed approaches with reasons: "SqueezeLLM crashed on 3090 due to memory during
  sensitivity analysis — not viable for 24GB GPUs"
- Hypotheses generated from observations: "Both Llama and Mistral showed better results
  with asymmetric quantization — this might be a general principle"

**What NOT to write:**
- Vague observations: "quantization helps" (too obvious)
- Exact numbers without context: "score was 3.2" (not useful for transfer)
- Implementation details: "set bits=4 in line 23" (not transferable)

**When switching models**: Before starting a new model, review the ENTIRE knowledge.md.
Explicitly plan which techniques to try first based on accumulated knowledge. Your first
experiment on Model 2 should use the best technique from Model 1 — don't start from scratch.

## Constraints

**What you CAN do:**
- Modify `optimize.py` — everything is fair game
- Use any installed quantization library
- Try novel approaches not in any paper
- Combine techniques creatively
- Read papers or documentation for inspiration

**What you CANNOT do:**
- Modify `evaluate.py` — it is the ground truth metric
- Install new packages (use only what's in the environment)
- Modify the base model files on disk
- Pre-compute or cache benchmark answers
- Use more than 24GB VRAM (hard constraint — 3090)

**Simplicity criterion** (same as original autoresearch): All else being equal, simpler is
better. A tiny efficiency_score gain from ugly complexity is not worth it. Removing complexity
while maintaining score is a great outcome.

## NEVER STOP

Once the experiment loop has begun, do NOT pause to ask if you should continue. The human
might be asleep. You are autonomous. If you run out of ideas:

1. Re-read knowledge.md for patterns you haven't exploited
2. Try combining two techniques that each helped independently
3. Try the opposite of something that failed (if high bits failed, try very low bits)
4. Try a mathematical transformation nobody has published
5. Re-read the quantization library documentation for options you missed

The loop runs until the human interrupts you, period.

When you finish ~50+ experiments on the current model and feel you've plateaued, move to the
next model in the sequence. Update knowledge.md BEFORE switching. On the new model, start by
applying the best known techniques from previous models.

## Success Criteria

This experiment succeeds if:
1. The best efficiency_score on Model 3 exceeds the best on Model 1
2. Convergence on Model 3 is measurably faster than on Model 1 (fewer experiments to reach 90% of best score)
3. knowledge.md contains genuine, transferable insights (not just a log of what happened)
4. At least one technique discovered beats off-the-shelf GPTQ/AWQ defaults
