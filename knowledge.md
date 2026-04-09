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

The agent should explore torchao's internals: look at how quantize_() calls HQQ, what interface it expects, and how to inject custom quantization logic. The calibration infrastructure from round 4 is available (activation stats collection).

Techniques to implement:
- GPTQ: use Hessian (X^T X) to find optimal rounding direction per weight. ~100 lines of core math.
- Optimal brain quantization: second-order error minimization
- Custom rounding: instead of round-to-nearest, use calibration data to pick the rounding direction that minimizes output error
- Novel combinations

Quality bars to beat: Llama 0.9021, Mistral 0.9590 (HQQ int4 gs=128).
