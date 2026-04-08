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
- **tiktoken not installed** — Phi-3-small-8k-instruct cannot be used (requires tiktoken)
- **Only bitsandbytes works** for quantization. All optimization must work within bnb constraints.

## Best Configuration (Score ~2.0-3.2 depending on model)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

torch.backends.cuda.preferred_blas_library("cublaslt")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_storage=torch.uint8,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
)

model.generation_config.prompt_lookup_num_tokens = 40
```

### Key insight: prompt_lookup_num_tokens is the biggest win

**prompt_lookup_num_tokens=40** was the single most impactful optimization. It uses n-gram matching on previously generated tokens to create draft token sequences, then verifies them in parallel. This effectively amortizes the bnb dequantization overhead across multiple tokens per forward pass.

Results:
- Llama 3.1 8B: 1.71 → 2.02 (18% improvement)
- Mistral 7B v0.3: 2.07 → 2.81-3.16 (36-53% improvement)
- Sweet spot is ~40 tokens. Below 20 is too conservative, above 60 has diminishing returns.

### Why it works

bnb NF4 is memory-bandwidth limited during batch-1 generation. Each token requires reading all model weights. With prompt_lookup, the model processes 2-5 tokens per forward pass (when n-gram matches succeed), effectively 2-5x throughput for those steps. The verification of N draft tokens costs only marginally more than generating 1 token (parallel forward pass).

## What Works

- `bnb_4bit_compute_dtype=torch.bfloat16` — **much faster than float16** on Ampere
- `bnb_4bit_use_double_quant=False` — double quant adds ~33% speed penalty even with prompt_lookup
- `bnb_4bit_quant_storage=torch.uint8` — slight improvement
- `torch.backends.cuda.preferred_blas_library("cublaslt")` — small speedup for batch-1
- Default attention (no explicit `attn_implementation`) — slightly better for Mistral (sliding window)

## Model-Specific Results

### Llama 3.1 8B (RTX 3090)
- FP16 baseline: ppl 10.39, 38.3-39.2 tok/s, 15.64 GB
- Best NF4: ppl 11.11, 34.2 tok/s, 6.33 GB → **score 2.02**
- Quality retained: 0.936

### Mistral 7B v0.3 (RTX 3090)  
- FP16 baseline: ppl 8.89, 36.0-41.3 tok/s, 13.78 GB
- Best NF4: ppl 9.16, 38.0-38.8 tok/s, 4.47 GB → **score 2.81-3.16**
- Quality retained: 0.971 (better than Llama — NF4 is kinder to Mistral)
- Mistral benefits more from prompt_lookup (can exceed FP16 speed!)

### Phi-3 small 8k instruct
- **BLOCKED**: requires tiktoken package (not in deps)

## What Doesn't Work

- **bnb 8-bit**: 2x slower than 4-bit. Do not use.
- **Double quantization**: Saves 0.33 GB but costs 33% speed. Not worth it even with prompt_lookup.
- **FP4 quant type**: Worse quality (0.878 vs 0.935) with same speed.
- **Static KV cache**: Hurts speed by ~10%.
- **Speculative decoding (assistant model)**: Both models suffer from bnb overhead. Use prompt_lookup instead.
- **Layer pruning**: Even 4 layers (12.5%) destroys quality. Requires calibration-aware methods.
- **2:4 structured sparsity**: Naive pruning destroys quality. Needs SparseGPT/Wanda (not implementable without triton).
- **Removing accelerate dispatch hooks**: No measurable effect.
- **TF32/cudnn flags**: No effect on bfloat16 compute.
- **Disabling math SDP**: Slightly hurts — default is better.

## Strategy for Future Sessions

1. Start with the best config above (NF4 bf16 no-double-quant uint8-storage cuBLASLt prompt_lookup=40)
2. If a new quantization library becomes available (triton fixed, torchao installed), it could break through the bnb speed ceiling
3. The prompt_lookup sweet spot is model-dependent — test 20/40/60 for new architectures
4. Quality retention varies by model — Mistral is more NF4-friendly than Llama
