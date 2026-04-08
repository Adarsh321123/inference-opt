# inference-opt

The idea: give an AI agent an LLM and let it experiment with optimization strategies autonomously overnight. It modifies the optimization code, evaluates quality/speed/memory, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a more efficient model. The core idea is that you're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the `program.md` Markdown file that provides context to the AI agent. Unlike standard autoresearch, this repo also accumulates knowledge in `knowledge.md` across models — the agent should get faster at optimizing each subsequent model.

## How it works

The repo is deliberately kept small and only really has four files that matter:

- **`evaluate.py`** — fixed evaluation harness: loads a model, applies the optimization from `optimize.py`, measures perplexity (quality), tokens/sec (speed), and peak VRAM (memory). Computes an efficiency score. Not modified.
- **`optimize.py`** — the single file the agent edits. Contains the `optimize_model()` function that loads and optimizes a model. Everything is fair game: quantization method, bit-width, group size, layer-wise precision, mathematical transformations, combining techniques. **This file is edited and iterated on by the agent**.
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.
- **`knowledge.md`** — accumulated insights from prior experiments. The agent reads this at the start of every session and updates it every ~10 experiments. **This file is edited by the agent and grows over time**.

The metric is **efficiency_score** — a combination of quality retention, speedup, and memory reduction vs FP16 baseline. Higher is better.

## Quick start

**Requirements:** A single NVIDIA GPU (tested on 3090), Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Run FP16 baseline for a model (~5 min)
uv run evaluate.py --baseline --model meta-llama/Meta-Llama-3-8B

# 4. Run the default optimization to verify setup works
uv run evaluate.py --model meta-llama/Meta-Llama-3-8B
```

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo, then prompt something like:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

## Project structure

```
evaluate.py     — evaluation harness: quality + speed + memory (do not modify)
optimize.py     — optimization strategy (agent modifies this)
knowledge.md    — accumulated insights (agent updates this)
program.md      — agent instructions
pyproject.toml  — dependencies
```

## Design choices

- **Single file to modify.** The agent only touches `optimize.py`. This keeps the scope manageable and diffs reviewable.
- **Knowledge accumulation.** Unlike standard autoresearch, the agent updates `knowledge.md` with transferable insights. When switching to a new model, it reads accumulated knowledge and should converge faster.
- **Model sequence.** The agent optimizes models in order: Llama 3 8B → Mistral 7B → Phi-3. The goal is to demonstrate that accumulated knowledge transfers across architectures.
- **Self-contained.** No external dependencies beyond PyTorch and quantization libraries. One GPU, one file, one metric.

## License

MIT
