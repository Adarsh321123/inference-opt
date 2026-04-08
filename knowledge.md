# knowledge

Accumulated insights from optimization experiments. Read this before every session.
Update every ~10 experiments with what you learned.

## Round 1 (apr8, RTX 3090)

### Best Configuration (bitsandbytes-only, score ~2.0-3.2)

NF4 bf16 compute, no double quant, uint8 storage, cuBLASLt, prompt_lookup=40.

### Key Discovery: prompt_lookup_num_tokens

prompt_lookup_num_tokens=40 was the single biggest win. Uses n-gram matching on generated tokens as draft sequences, verified in parallel. Amortizes bnb dequantization overhead across multiple tokens per forward pass. Sweet spot is ~40. Below 20 too conservative, above 60 diminishing returns. On Mistral it beat FP16 speed while at 4-bit.

- Llama 3.1 8B: 1.71 → 2.02 (18% improvement)
- Mistral 7B: 2.07 → 3.16 (53% improvement)

### What Works

- bnb_4bit_compute_dtype=torch.bfloat16 — much faster than float16 on Ampere
- bnb_4bit_use_double_quant=False — double quant adds ~33% speed penalty
- bnb_4bit_quant_storage=torch.uint8 — slight improvement
- torch.backends.cuda.preferred_blas_library("cublaslt") — small speedup for batch-1

### What Doesn't Work

- bnb 8-bit: 2x slower than 4-bit
- Double quantization: saves 0.33 GB but costs 33% speed
- FP4 quant type: worse quality (0.878 vs 0.935) with same speed
- Static KV cache: hurts speed by ~10%
- Speculative decoding with assistant model: both models suffer from bnb overhead
- Layer pruning: even 4 layers (12.5%) destroys quality without calibration-aware methods
- 2:4 structured sparsity: naive pruning destroys quality
- TF32/cudnn flags: no effect on bfloat16 compute

### Model-Specific

- Llama 3.1 8B: FP16 ppl 10.39, 38.3 tok/s, 15.64 GB. Best NF4: ppl 11.11, score 2.02
- Mistral 7B: FP16 ppl 8.89, 41.3 tok/s, 13.78 GB. Best NF4: ppl 9.16, score 3.16
- Mistral retains quality better under NF4 than Llama (0.971 vs 0.936)
- Phi-3 small blocked in round 1 (missing tiktoken, now fixed)

### Round 1 Limitations

Only bitsandbytes worked. GPTQ, AWQ, HQQ, torch.compile all failed due to env issues (triton broken, version incompatibilities). Round 2 has triton, auto-gptq, autoawq, optimum, tiktoken added to deps.

### Hypotheses for Round 2

- GPTQ/AWQ should give much faster inference (42-80 tok/s) than bnb (~23-38 tok/s)
- prompt_lookup on top of GPTQ/AWQ could produce scores of 5-8x
- Phi-3 small should be testable now with tiktoken installed
- Try EXL2-style mixed bitrate if possible via optimum
