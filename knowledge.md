# knowledge

Accumulated insights from 170+ experiments across 4 rounds. Read this before every session.

## Rounds 1-3: Library-Level Optimization

Best scores with existing library configurations:
- Llama 3.1 8B: score 2.7 (torchao Int4 HQQ gs=128 + prompt_lookup=64)
- Mistral 7B: score 3.9 (torchao Int4 HQQ gs=128 + prompt_lookup=256)

## Round 4: Weight Transformations (FAILED)

Tried 40+ experiments applying math transformations before HQQ quantization. ALL made things worse:
- AWQ-style channel scaling: improved quality 0.959→0.969 but killed prompt_lookup speedup (1.72→1.12x). Net WORSE.
- Outlier clipping (3σ, 6σ): destroyed quality. Outlier weights carry critical information.
- Bias correction: hurt quality.
- Mixed precision (keep some layers bf16): killed speedup.

**Why transforms failed:** HQQ already optimizes its quantization grid for the weight distribution. Changing the distribution before HQQ makes its job harder, not easier.

**Key discovery:** AWQ changes the model's output distribution, reducing n-gram match rate in prompt_lookup speculative decoding. Any transform that changes output distribution hurts prompt_lookup.

**Key discovery:** Int4 is SLOWER than FP16 at batch=1 (23 vs 37 tok/s). prompt_lookup is essential — it enables batched verification that makes int4 2.5x faster (23→60 tok/s).

## Round 4 Conclusions

- Don't transform weights before HQQ — it interferes with HQQ's optimization
- Don't clip outliers — they're critical
- RTN (use_hqq=False) in torchao 0.15.0 doesn't engage fast tinygemm kernels
- group_size=64 improves quality but kills speedup
- torch.compile doesn't help for short generation
- lm_head quantization causes massive VRAM spikes
- Quality is deterministic: Llama 0.9021, Mistral 0.9590 with HQQ int4 gs=128

## Round 5: Replace HQQ's Quantization Math

The lesson from round 4: don't work AROUND HQQ, REPLACE it.

torchao's quantize_() with use_hqq=True calls HQQ internally to compute int4 values, then packs them into TensorCoreTiledLayout for fast tinygemm inference. The approach:

1. Read torchao's source to understand how HQQ is called internally
2. Write a custom quantization function that computes BETTER int4 values
3. Pack those values into the same torchao int4 format
4. Get fast inference from the same tinygemm kernels

CRITICAL from round 4: RTN (use_hqq=False) does NOT engage fast tinygemm kernels.
RTN gives 1.01x speedup vs HQQ's 1.30x. So we MUST keep use_hqq=True.

Approach: monkey-patch HQQ's internal rounding function with custom math.
1. Read torchao's source: `python -c "import torchao; print(torchao.__file__)"`
2. Trace how Int4WeightOnlyConfig(use_hqq=True) triggers HQQ rounding
3. Find the specific function that does rounding/scale computation
4. Replace it with custom logic (GPTQ-style Hessian rounding, etc.)
5. Call quantize_() normally — gets HQQ's fast kernel path + custom math

The calibration collects Hessians (H = X^T X) stored in _layer_hessians global.
The monkey-patched function can read these for GPTQ-style optimal rounding.

Quality bars to beat: Llama 0.9021, Mistral 0.9590 (HQQ int4 gs=128).
