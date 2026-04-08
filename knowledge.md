# knowledge

Accumulated insights from optimization experiments (55+ experiments across 2 rounds).

## Environment Constraints

- **Triton works** (after `apt-get install python3-dev`). Enables torch.compile.
- **torch.compile works** — default mode best. Helps Llama (memory), neutral for Mistral.
- **auto-gptq broken** — RoPE shape mismatch with transformers 4.51+.
- **autoawq** — works but `awq_ext` missing → 2x slower than bnb.
- **Phi-3** blocked (einops/rope_scaling).
- **Only bitsandbytes works** for quantization.

## Best Configuration

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_storage=torch.uint8,
)

is_llama = "llama" in model_name.lower()

# cuBLASLt: +5.6% Llama, -15% Mistral. MUST be conditional.
if is_llama:
    torch.backends.cuda.preferred_blas_library("cublaslt")

attn_kwargs = {"attn_implementation": "eager"} if is_llama else {}

model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=quantization_config,
    device_map="auto", trust_remote_code=True, **attn_kwargs,
)

if is_llama:
    model = torch.compile(model, mode="default")  # saves 0.35 GB on Llama only

model.generation_config.prompt_lookup_num_tokens = 40
```

### Key insights (ranked)

1. **prompt_lookup=40** — biggest single win, amortizes dequant cost
2. **cuBLASLt is model-dependent**: +5.6% Llama, **-15% Mistral** (biggest round 2 finding)
3. **eager attention** > SDPA for Llama batch-1. Neutral for Mistral.
4. **torch.compile** saves 0.35 GB on Llama, no effect on Mistral
5. NF4 can **exceed FP16 speed** on Mistral (40 tok/s > 36 baseline) thanks to smaller model + prompt_lookup

## Results

### Llama 3.1 8B: score 2.1-2.3 (median 2.20)
- Config: cuBLASLt + eager + compile + prompt_lookup=40
- 33-35 tok/s, 5.98 GB, quality 0.936

### Mistral 7B v0.3: score 3.0-3.3 (median 3.25)
- Config: NO cuBLASLt, no compile, prompt_lookup=40
- 37-40 tok/s, 4.47 GB, quality 0.971
- NF4 exceeds FP16 speed!

### Phi-3: BLOCKED

## What Doesn't Work (55+ experiments)

AWQ (no kernels), GPTQ (RoPE mismatch), bnb 8-bit, FP4, double quant, layer pruning,
all compile variants except default, CompileConfig, forward-only compile,
TF32, static cache, device_map changes, dynamo configs, pre-warmup,
max_matching_ngram≠2, FP16 no quant (memory_reduction≈1.0)

## Future Strategy

1. Start with model-adaptive config above
2. If gptqmodel/torchao installed → try immediately (2-3x potential)
3. All bnb optimizations at ceiling
