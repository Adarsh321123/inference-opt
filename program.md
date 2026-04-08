# inference-opt

This is an experiment to have the LLM optimize inference efficiency autonomously.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr7`). The branch `inference-opt/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b inference-opt/<tag>` from current main.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `program.md` — these instructions.
   - `knowledge.md` — accumulated insights from prior runs. Read this carefully.
   - `evaluate.py` — fixed evaluation harness, dataloader, metrics. Do not modify.
   - `optimize.py` — the file you modify. Optimization strategy, quantization, compression.
4. **Install dependencies**: Run `uv sync`.
5. **Run baseline**: `uv run evaluate.py --baseline --model <model_name>` to establish the FP16 baseline. This writes `baseline.json`.
6. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. You launch it simply as: `uv run evaluate.py --model <model_name>`.

**What you CAN do:**
- Modify `optimize.py` — this is the only file you edit. Everything is fair game: quantization method, bit-width, group size, layer-wise precision, mathematical transformations, calibration strategy, combining techniques, inventing new approaches.

**What you CANNOT do:**
- Modify `evaluate.py`. It is read-only. It contains the fixed evaluation metrics.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the base model weights on disk.

**The goal is simple: get the highest efficiency_score.** The efficiency score combines quality retention, inference speedup, and memory reduction vs the FP16 baseline:

```
efficiency_score = quality_retained * speedup * memory_reduction

quality_retained = baseline_perplexity / optimized_perplexity  (capped at 1.0)
speedup          = tokens_per_sec / baseline_tps
memory_reduction = baseline_vram_gb / peak_vram_gb

If quality_retained < 0.85, efficiency_score = 0.
```

Higher is better. A baseline FP16 model scores 1.0. A good 4-bit quantization might score 3-4.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing complexity while maintaining score is a great outcome.

**The first run**: Your very first run should always be to establish the baseline with the default optimize.py, so you will run it as is.

**Model sequence**: Optimize models in this order:
1. `meta-llama/Meta-Llama-3-8B`
2. `mistralai/Mistral-7B-v0.3`
3. `microsoft/Phi-3-small-8k-instruct`

When switching models, run a new baseline first.

## Output format

Once the script finishes it prints a summary like this:

```
---
efficiency_score: 3.8000
quality_retained: 0.9500
speedup:          2.0000
memory_reduction: 2.0000
perplexity:       6.5600
baseline_ppl:     6.2300
tokens_per_sec:   85.3
baseline_tps:     42.7
peak_vram_gb:     8.2
baseline_vram_gb: 16.4
```

You can extract the key metric from the log file:

```
grep "^efficiency_score:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 7 columns:

```
commit	model	efficiency_score	quality_retained	speedup	memory_reduction	status	description
```

Example:

```
commit	model	efficiency_score	quality_retained	speedup	memory_reduction	status	description
a1b2c3d	llama3-8b	1.0000	1.0000	1.0000	1.0000	keep	baseline FP16
b2c3d4e	llama3-8b	3.2100	0.9500	1.8000	1.8800	keep	bitsandbytes NF4 double quant
c3d4e5f	llama3-8b	2.9000	0.9200	1.7000	1.8500	discard	GPTQ 4-bit group128
d4e5f6g	llama3-8b	0.0000	0.0000	0.0000	0.0000	crash	HQQ 2-bit (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `inference-opt/apr7`).

LOOP FOREVER:

1. Read knowledge.md for accumulated insights. On a new model, start with what worked before.
2. Tune `optimize.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run evaluate.py --model <model_name> > run.log 2>&1`
5. Read out the results: `grep "^efficiency_score:\|^quality_retained:\|^peak_vram_gb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked)
8. If efficiency_score improved (higher), you "advance" the branch, keeping the git commit
9. If efficiency_score is equal or worse, you git reset back to where you started

**Knowledge accumulation**: Every ~10 experiments, update `knowledge.md` with what you learned. Write concrete, transferable insights — what worked, what failed, what patterns you noticed. When switching to a new model, update knowledge.md BEFORE switching. This is critical: your goal is to converge faster on each subsequent model because of accumulated knowledge.

**Timeout**: Each experiment should take ~5 minutes total. If a run exceeds 15 minutes, kill it and treat it as a failure.

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix, fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status, and move on.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human might be asleep. You are autonomous. If you run out of ideas, think harder — re-read knowledge.md, try combining previous near-misses, try more radical approaches. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
