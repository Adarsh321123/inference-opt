# knowledge

Accumulated insights from optimization experiments. Read this before every session.
Update every ~10 experiments with what you learned.

## Environment Constraints (Critical)

- **Triton works** (after `apt-get install python3-dev`). Enables torch.compile.
- **torch.compile works** — default mode is best.
- **auto-gptq broken** — fixable import issues but unfixable RoPE shape mismatch with transformers 4.51+.
- **autoawq works** for quantization but `awq_ext` CUDA kernels missing → 2x slower than bnb.
- **transformers AWQ/GPTQ loading** requires `gptqmodel` (not installed).
- **Phi-3-small** needs `einops`. **Phi-3-mini** has `rope_scaling` config incompatibility.
- **FP8** not supported on RTX 3090 (need 8.9+).
- **Only bitsandbytes works** for quantization.

## Best Configuration (Score ~2.2-2.9 depending on model)

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
    attn_implementation="eager",  # faster for batch-1 on Llama
)

model = torch.compile(model, mode="default")  # saves ~0.35 GB VRAM
model.generation_config.prompt_lookup_num_tokens = 40
```

### Key insights

1. **prompt_lookup_num_tokens=40** — biggest win. Sweet spot is 40.
2. **eager attention > SDPA** for Llama batch-1 (~1-2 tok/s). Equivalent for Mistral.
3. **torch.compile mode="default"** — saves ~0.35 GB VRAM (~7% score boost).
4. bnb NF4 dequantization is the fundamental bottleneck (~60% of time).
5. **CompileConfig** (generation-level compile) adds no benefit on top of model-level compile.
6. **model.forward-only compile** is catastrophically slow — always compile the full model.

## Model-Specific Results

### Llama 3.1 8B (RTX 3090)
- FP16 baseline: ppl 10.39, 37-39 tok/s, 15.64 GB
- Best: ppl ~11.1, 33-35 tok/s, 5.98 GB → **score 2.1-2.3** (median ~2.20)
- Quality retained: 0.936

### Mistral 7B v0.3 (RTX 3090)
- FP16 baseline: ppl 8.89, 36-41 tok/s (high variance!), 13.78 GB
- Best: ppl ~9.16, 35-38 tok/s, 4.47 GB → **score 2.8-2.9**
- Quality retained: 0.971. Baseline variance → score range 2.7-3.2.

### Phi-3: BLOCKED (einops/rope_scaling issues)

## What Doesn't Work (exhaustive list from 45+ experiments)

**Quantization alternatives:**
- AWQ without awq_ext: 2x slower (17.9 tok/s)
- GPTQ auto_gptq: RoPE shape mismatch
- bnb 8-bit: 2x slower, double quant: 27% speed penalty
- FP4 quant type: worse quality (0.878 vs 0.935)
- FP16 no quant: fast (44 tok/s) but memory_reduction≈1.0

**torch.compile variations:**
- reduce-overhead: CUDA graphs break with variable shapes
- max-autotune: more memory
- fullgraph=True: worse performance
- aot_eager: no benefit
- forward-only: catastrophic (12 tok/s)
- dynamic=True: no improvement
- CompileConfig (generation-level): no benefit over model compile

**Other:**
- TF32/bf16 matmul precision, suppress_errors: no effect
- Static KV cache, device_map changes, pre-warmup: no effect
- Inductor tuning: marginal
- Layer pruning: quality destruction
- Speculative decoding with assistant model: bnb overhead
- bf16_reduced_precision_reduction: no effect

## Strategy for Future Sessions

1. Start with best config above
2. If gptqmodel/torchao installed → try immediately
3. If autoawq-kernels installed → AWQ becomes viable
4. Baseline speed has ±15% variance — run multiple baselines
5. **All bnb optimizations are at ceiling** — need new quantization library to break through
