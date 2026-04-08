# knowledge

Accumulated insights from optimization experiments. Read this before every session.
Update every ~10 experiments with what you learned.

## Environment Constraints (Critical)

- **Triton is broken** — gcc cannot compile triton CUDA utils. This rules out:
  - `torch.compile()` (any mode)
  - AWQ inference (uses triton for dequantization kernels)
  - Any custom triton kernels
- **auto-gptq is incompatible** with this transformers version (`no_init_weights` import error)
- **HQQ not supported** — transformers says "hqq is not available yet and will be supported soon"
- **torchao not installed** — not in pyproject.toml
- **Only bitsandbytes works** for quantization. All optimization must work within bnb constraints.

## What Works (Llama 3.1 8B, RTX 3090)

Best config (score 1.71):
- `load_in_4bit=True` with `bnb_4bit_quant_type="nf4"`
- `bnb_4bit_compute_dtype=torch.bfloat16` — **much faster than float16** on Ampere
- `bnb_4bit_use_double_quant=False` — double quant adds ~33% speed penalty
- `bnb_4bit_quant_storage=torch.uint8` — slight improvement
- `torch.backends.cuda.preferred_blas_library("cublaslt")` — small speedup for batch-1
- `attn_implementation="sdpa"` — default but explicit is fine

Baseline numbers (FP16):
- Perplexity: 10.3939
- Speed: 38.3 tok/s
- VRAM: 15.64 GB

NF4 best numbers:
- Perplexity: 11.11 (quality_retained 0.936)
- Speed: 28.2 tok/s (0.74x baseline)
- VRAM: 6.33 GB (2.47x reduction)

## What Doesn't Work

- **bnb 8-bit**: 2x slower than 4-bit (9 tok/s). Do not use.
- **Double quantization**: Saves 0.33 GB VRAM but costs 33% speed. Net negative.
- **FP4 quant type**: Worse quality (0.878 vs 0.935) with same speed. NF4 is strictly better.
- **Static KV cache**: `model.generation_config.cache_implementation = "static"` — actually HURTS speed by ~10%.
- **Speculative decoding**: With Llama-3.2-1B draft model, both models suffer from bnb overhead. Net slower.
- **Layer pruning**: Even removing 4 layers (12.5%) destroys quality (ppl 25 vs 11). LLMs are surprisingly sensitive to layer removal without calibration/finetuning.
- **2:4 structured sparsity**: Naive magnitude pruning destroys quality (ppl 1442). Requires calibration-aware pruning (SparseGPT/Wanda) which we can't implement without triton.
- **Removing accelerate dispatch hooks**: No measurable effect.
- **TF32 flags**: No effect on bfloat16 compute.
- **Disabling math SDP**: Slightly hurts, probably forces suboptimal attention path.

## Key Insight: bnb Speed Ceiling

The fundamental bottleneck is bitsandbytes' dequantization overhead. NF4 should theoretically give ~2.5x speedup (due to 2.5x less memory to read), but bnb's per-block dequantization adds enough overhead to make it 0.74x instead. The only way to break this ceiling is with fused dequant-matmul kernels (AWQ GEMM, GPTQ exllama/marlin), which all require triton or incompatible packages.

## Strategy for New Models

1. Start with the proven config: NF4 bf16 no-double-quant uint8-storage cuBLASLt sdpa
2. Run baseline first, then run this config
3. Quality retained is model-dependent — if it drops below 0.85, no amount of speed/memory improvement helps
4. Focus experiments on model-specific tweaks rather than re-discovering environmental constraints
