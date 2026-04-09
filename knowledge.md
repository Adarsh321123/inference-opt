# knowledge

Accumulated insights from 125+ experiments across 3 rounds. Read this before every session.

## Rounds 1-3 Summary

Best scores achieved with library-level optimization (API calls only):
- Llama 3.1 8B: score 2.7 (torchao Int4 HQQ + prompt_lookup=64)
- Mistral 7B: score 3.9 (torchao Int4 HQQ + prompt_lookup=256, no cuBLASLt)

These represent the ceiling of configuring existing libraries. Round 4 shifts to from-scratch quantization math to break through this ceiling.

## Key Findings (transferable)

- torchao Int4WeightOnly with use_hqq=True is the best quantization backend (faster than bnb NF4)
- prompt_lookup_num_tokens is the single biggest inference speedup (amortizes dequant overhead)
- cuBLASLt is model-dependent: helps Llama, hurts Mistral by 15%
- Streaming CPU→GPU quantization keeps peak VRAM low
- bfloat16 compute dtype is required for torchao tinygemm kernels
- Mistral retains quality better under 4-bit than Llama (0.97 vs 0.90)
- Quality retention at 4-bit varies 85-97% across methods — the MATH of quantization matters

## Round 4: From-Scratch Quantization

optimize.py now contains a full quantization pipeline:
1. Load model bf16
2. Collect calibration data and activation statistics
3. Apply custom weight transformations (YOUR CREATIVE SPACE)
4. Quantize with torchao int4 kernels

The `transform_weights_for_quantization()` function is where breakthroughs happen.
The goal: discover a transformation that produces better quality at 4-bit than any existing method.

IMPORTANT: The default baseline uses `use_hqq=True`, which is already a good quantizer (not naive RTN). Try `use_hqq=False` early — transformations will show a bigger effect against the weaker RTN baseline. Then test if the best transformation also helps on top of HQQ.

Known techniques to try and beat:
- AWQ: scales important channels by activation magnitude (alpha=0.5 weighting)
- GPTQ: uses Hessian to find optimal rounding direction per weight
- QuIP#: Hadamard rotation makes weight distribution more uniform
- SqueezeLLM: separates outlier weights into sparse matrix

For cross-layer compensation (needed for AWQ-style scaling): if you scale layer i's input weights by s, you must divide layer i-1's output weights by s. The function has access to prev_layer and next_layer for this.

## Round 4: Weight Transformation Experiments (20 experiments on Llama 3.1 8B)

### Key Findings

**HQQ quality ceiling**: HQQ int4 gs=128 gives quality_retained=0.9021 (ppl 11.52) for Llama, consistently across all experiments. This is the algorithm's quality ceiling and cannot be improved through weight preprocessing.

**Weight transforms hurt HQQ quality**: Every weight transformation tried before HQQ made quality worse:
- AWQ-style channel scaling (s ∈ [0.5, 2.0]): quality 0.894 → WORSE
- 3σ per-group outlier clipping: quality 0.037 → CATASTROPHIC
- 6σ per-group outlier clipping: quality 0.854 → still hurts
- Post-quantization bias correction: quality 0.876 → WORSE

**Why transforms hurt HQQ**: HQQ already adapts its quantization grid to the weight distribution. Modifying the distribution before HQQ changes what HQQ is optimizing for, often making its job harder.

**Outlier weights are sacred**: Even clipping 0.0025% of weights (6σ) increases perplexity from 11.52 to 12.17. Outlier weights in transformers carry disproportionate information — they MUST be preserved.

**RTN has no speedup**: use_hqq=False (RTN) with version=1 gives 37.4 tps vs 37.0 baseline (1.01x). HQQ gives 48.1 tps (1.30x). The RTN code path in torchao 0.15.0 doesn't engage fast tinygemm kernels properly.

**Mixed precision kills speedup**: Keeping even 2 of 32 layers in bf16 drops speedup from 1.30x to 1.03x.

**group_size=64 kills speedup**: Better quality (0.92 vs 0.90) but speedup drops from 1.30x to 0.98x.

**torch.compile doesn't help**: No speedup improvement. Warmup overhead hurts short generation tasks.

**lm_head quantization VRAM spike**: Quantizing the [128256, 4096] lm_head with HQQ creates massive temporary buffers (peak 17.36 GB), negating any savings.

**Alternative quantization backends**:
- FP4 (e2m1): 27x SLOWER, quality below floor
- W4A8 (Int8DynAct+Int4Weight): 1.8x slower, quality 0.70
- GemliteUIntX: not installed
- FPXWeightOnly: only useful for fp6+, uses too much memory

### Best Score: Llama 3.1 8B
Score ~2.6 (varies 2.5-2.8 from speedup noise): HQQ int4 gs=128 + prompt_lookup=64
- quality_retained: 0.9021 (fixed)
- speedup: 1.25-1.40 (noise)
- memory_reduction: 2.23 (fixed)

### Mistral Experiments (10+ experiments)

**AWQ improves quality but kills speedup via prompt_lookup**:
- AWQ scaling (calibration-based): quality 0.969 vs 0.959, BUT speedup drops 1.72→1.12
- AWQ scaling (weight-only): quality 0.969 vs 0.959, BUT speedup drops 1.65→1.03
- Root cause: AWQ changes the model's output distribution, reducing n-gram match rate in prompt_lookup speculative decoding
- Raw kernel speed is IDENTICAL with/without AWQ (23.3 vs 23.6 tps without prompt_lookup)

**Critical insight: int4 is SLOWER than FP16 at batch=1**:
- Without prompt_lookup: int4 HQQ gets 23.3 tps, FP16 baseline gets 36.8 tps (0.63x!)
- Dequantization overhead dominates at batch=1 autoregressive generation
- prompt_lookup enables batched verification, which makes int4 2.5x faster (23→60 tps)
- Therefore: prompt_lookup is not just a speedup optimization — it's ESSENTIAL for int4 to beat FP16

**Disabling cuBLASLt no longer helps Mistral** (might be version-dependent)

**Tied embed/lm_head weights destroy quality** (ppl 59K — they're trained separately)

### Best Score: Mistral 7B
Score ~4.0 (varies 3.6-4.1 from speedup noise): HQQ int4 gs=128 + prompt_lookup=256
- quality_retained: 0.9590 (fixed)
- speedup: 1.50-1.72 (noise)
- memory_reduction: 2.48 (fixed)

### Phi-3-small: BLOCKED
- Requires pytest which isn't in dependencies
- Would need pyproject.toml update

### Transferable Insights (All Rounds)
1. Don't try weight transforms with HQQ — they interfere with HQQ's optimization
2. Don't clip weight outliers — they're critical for model quality
3. RTN in torchao 0.15.0 is broken for speedup — always use HQQ
4. Keep the pipeline simple: load → stream layers → HQQ quantize → prompt_lookup
5. The only way to improve quality is a fundamentally different quantization algorithm (GPTQ, etc.)
6. prompt_lookup is ESSENTIAL — without it, int4 is slower than FP16 at batch=1
7. Anything that changes model output distribution (AWQ, weight transforms) reduces prompt_lookup effectiveness
8. lm_head quantization with HQQ causes massive VRAM spikes — avoid
9. The simplest pipeline consistently gives the best results
10. Post-quantization fine-tuning blocked: aten::_weight_int4pack_mm has no backward implementation
11. Quantized KV cache (HQQ 4-bit) doesn't help for short sequences (<128 tokens)
12. inner_k_tiles in TensorCoreTiledLayout (2/4/8) makes no measurable difference
13. Removing per-layer gc.collect()/empty_cache() is safe and saves ~5s optimization time
14. Score noise is significant: Llama ±11%, Mistral ±14% — due to speedup variance in baseline
