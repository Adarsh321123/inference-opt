# Knowledge Base: LLM Inference Optimization

This file accumulates insights from optimization experiments across models.
It is READ at the start of every session and UPDATED every ~10 experiments.
The goal: by the Nth model, converge on optimal strategies faster because of what was learned on models 1 through N-1.

## General Principles
<!-- Insights that apply across all models. Updated as patterns emerge. -->

- Post-training quantization (PTQ) methods like GPTQ and AWQ are the standard baseline
- 4-bit quantization typically retains 95-98% of FP16 quality for 7B+ models
- Smaller group sizes (e.g., 64 vs 128) improve quality at cost of slightly slower inference
- Attention layers tend to be more sensitive to quantization than MLP layers
- Calibration data matters: GPTQ degrades 2.3-4.9 ppl under domain shift, AWQ only 0.5-0.6
- Weight-only quantization (W4A16) is better than W4A4 for generative tasks on consumer GPUs

## Known Baselines (Llama 3 8B, WikiText-2 perplexity)

| Method | Perplexity | Delta | VRAM | Speed (tok/s) | Quant Time |
|--------|-----------|-------|------|---------------|------------|
| FP16 | 6.233 | -- | ~16 GB | ~25-30 | -- |
| bitsandbytes NF4 | ~6.30 | +1.1% | ~8.2 GB | ~23 | instant |
| AWQ 4-bit | ~6.35 | +1.9% | ~9.6 GB | ~40 | 10-15 min |
| GGUF Q4_K_M (imatrix) | 6.383 | +2.4% | ~5.0 GB | ~30-45 | 2-5 min |
| GPTQ 4-bit | ~6.41 | +2.8% | ~8.7 GB | ~42-52 | 25-45 min |
| HQQ 4-bit | comparable | ~2-3% | ~8 GB | varies | <1 min |
| EXL2 4bpw | ~6.4 | ~2.7% | ~8 GB | ~56-80 | 15-30 min |

**Key observation**: bitsandbytes has best quality but worst speed. GPTQ/EXL2 have best speed but worst quality. The gap between best quality (6.30) and best speed (~6.41) at 4-bit is only 0.11 perplexity points. Can we close this gap or beat both?

## Model-Specific Findings

### Llama 3 8B
<!-- Will be populated after first run -->

### Mistral 7B
<!-- Will be populated after second run -->

### Phi-3 Small
<!-- Will be populated after third run -->

## Failed Approaches
<!-- What didn't work and why. Critical for avoiding repeated mistakes. -->

## Hypotheses to Test
<!-- Promising ideas that haven't been tested yet. Prioritize these. -->

- Does layer-wise mixed precision (higher bits for attention, lower for MLP) beat uniform quantization?
- Can mathematical transformations (rotations, projections) before quantization improve quality?
- Does quantizing the first and last layers at higher precision disproportionately help?
- Are there optimal calibration data compositions (code vs text vs math)?

## Transferable Techniques
<!-- Approaches that worked on one model and should be tried first on the next model. -->
