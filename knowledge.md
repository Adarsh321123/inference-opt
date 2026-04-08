# knowledge

Accumulated insights from optimization experiments. Read this before every session.
Update every ~10 experiments with what you learned.

## Environment (Round 3 — UPDATED)

- **auto-gptq and autoawq are DEAD** (archived 2025). Removed from deps.
- **torchao 0.15.0** installed — PyTorch-native Int4WeightOnly quantization. Composable with torch.compile.
- **hqq** installed — calibration-free quantization (used internally by torchao's `use_hqq=True`).
- **einops** installed — Phi-3 should now work (but Phi-3 also needs `pytest` which is not installed).
- **Triton works**, torch.compile works.
- **bitsandbytes still works** as fallback. Best bnb config is in rounds 1-2 below.
- **GPTQ/AWQ loading broken**: optimum's GPTQConfig needs `gptqmodel` (not installed). Pre-quantized HF models can't be loaded.
- **HqqConfig through transformers broken**: "not available yet" error in current transformers version.

## Best Configuration (Round 3 — Score 2.6-4.1)

```python
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def optimize_model(model_name, device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # cuBLASLt: helps Llama, HURTS Mistral with torchao int4 kernels
    if "mistral" not in model_name.lower():
        torch.backends.cuda.preferred_blas_library("cublaslt")

    # Load bf16 on CPU to avoid GPU memory spike
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cpu",
        trust_remote_code=True, low_cpu_mem_usage=True,
    )

    from torchao.quantization import quantize_, Int4WeightOnlyConfig
    config = Int4WeightOnlyConfig(group_size=128, use_hqq=True, version=1)

    # Move non-quantizable layers to GPU first
    model.model.embed_tokens.to("cuda:0")
    model.model.norm.to("cuda:0")
    model.model.rotary_emb.to("cuda:0")
    model.lm_head.to("cuda:0")

    # Stream each layer: CPU → GPU, quantize, free bf16
    for layer in model.model.layers:
        layer.to("cuda:0")
        quantize_(layer, config)
        gc.collect()
        torch.cuda.empty_cache()

    # Model-specific prompt_lookup — Mistral benefits from much higher values
    if "mistral" in model_name.lower():
        model.generation_config.prompt_lookup_num_tokens = 256
    else:
        model.generation_config.prompt_lookup_num_tokens = 40

    return model, tokenizer
```

## Key Insights from Round 3

### torchao Int4WeightOnly is the breakthrough (2x better than bnb for speed)

The `torchao.quantization.quantize_()` function with `Int4WeightOnlyConfig(group_size=128, use_hqq=True, version=1)` gives dramatically faster inference than bitsandbytes NF4:
- **Llama 3.1 8B**: 48 tok/s (1.37x speedup) vs bnb's 33 tok/s (0.87x)
- **Mistral 7B**: 59 tok/s (1.58x speedup) vs bnb's 38 tok/s (1.0x)

This is because torchao uses tinygemm tensor-core-tiled INT4 matmul kernels, which are much faster than bnb's NF4 dequantize→matmul path for batch-1 generation.

### Streaming quantization is critical for memory

Loading the model bf16 on GPU then quantizing peaks at 16+ GB. Loading bf16 on CPU and streaming layers one at a time to GPU + quantize keeps peak at ~5.5-7 GB:

```
Peak VRAM with model.to("cuda"): 13.45 GB
Peak VRAM with layer-by-layer:   5.55 GB (Mistral), 7.01 GB (Llama)
```

The model.to("cuda") bulk transfer temporarily doubles GPU memory. Layer-by-layer avoids this.

### cuBLASLt interaction with torchao

- cuBLASLt **helps** Llama with torchao int4 kernels (48.2 vs 48.1 tok/s — minimal)
- cuBLASLt **HURTS** Mistral with torchao int4 (52.8 vs 60.9 tok/s — 15% penalty!)
- This is the opposite of bnb, where cuBLASLt helped slightly for both models.
- The torchao tinygemm kernels have their own optimized GEMM path; cuBLASLt interferes.

### prompt_lookup scales MUCH higher with torchao

With bnb NF4, prompt_lookup sweet spot was 40 (diminishing returns above 60). With torchao int4:
- **Llama**: 40 is still optimal (higher values slightly worse)
- **Mistral**: 256 is optimal (60→100→128→200→256 kept improving, up to 61.8 tok/s at 256; 512 was too high)

This is because torchao's faster forward pass means the overhead of verifying more tokens is relatively lower. Mistral benefits more than Llama from this effect.

### HQQ quantization quality vs optimization

- `use_hqq=True` gives better quality than RTN (`use_hqq=False`): 0.902 vs 0.905 quality for Llama
- HQQ's proximal optimization uses CUDA internally — can't avoid GPU allocation during quantization
- If you do HQQ quantization on CPU with `device='cpu'`, it's ~100x slower (hours vs seconds)
- The HQQ no-optimize path (skip proximal optimization) gives worse quality AND speed

### Additional findings from round 3

- **Eager attention hurts with torchao**: SDPA (default) is better. Opposite of bnb where eager helped Llama.
- **group_size must be power of 2**: group_size=96 silently fails to compress (tinygemm constraint).
- **group_size=32**: Better quality (0.935) but much slower (40.9 tok/s) and more memory (7.69 GB). Not worth it.
- **Mixed precision (MLP int4, attention bf16)**: Better quality (0.928) and speed (50.2) but much worse memory (8.83 GB). Full int4 wins overall.
- **torch.compile on torchao**: Neutral — tinygemm already optimized.
- **gc.collect() in quantize loop**: Helps slightly — keep it.

### What doesn't work with torchao

- **Int4WeightOnlyConfig version=2**: Requires `fbgemm-gpu-genai >= 1.2.0` (not installed)
- **Int4WeightOnlyConfig version=1 via TorchAoConfig**: Loads through transformers but inference is 17+ minutes (tinygemm extremely slow through HF integration)
- **Int8DynamicActivationInt4WeightConfig**: 10 tok/s — extremely slow
- **Int8WeightOnlyConfig**: Only 1.7x memory reduction, slower than fp16
- **torch.compile on torchao int4**: No significant benefit — tinygemm kernels already optimized
- **flash_attention_2**: Not installed
- **Quantizing lm_head**: Causes 17+ GB peak VRAM spike (HQQ temporary allocations on large weight)
- **FPXWeightOnlyConfig (FP4)**: Untested end-to-end (slow to quantize)
- **Int4DynamicActivationInt4WeightConfig**: Destroys quality (0.03 retention) and very slow (8.6 tok/s)
- **Static KV cache**: Destroys quality with quantized models (0.77 quality_retained)
- **Eager attention**: Worse than SDPA for torchao int4 (opposite of bnb)

## Model-Specific Results (Round 3)

### Llama 3.1 8B (RTX 3090)
- FP16 baseline: ppl 10.39, 35-38 tok/s, 15.64 GB
- **torchao Int4 HQQ**: ppl 11.52, 47-49 tok/s, 7.01 GB → **score 2.6-2.8**
- vs bnb NF4: ppl 11.11, 33 tok/s, 6.33 GB → **score 2.01**
- Improvement: **30-40% better** than bnb

### Mistral 7B v0.3 (RTX 3090)
- FP16 baseline: ppl 8.89, 37.3 tok/s, 13.78 GB
- **torchao Int4 HQQ**: ppl 9.27, 57-62 tok/s, 5.55 GB → **score 3.6-4.1**
- vs bnb NF4: ppl 9.16, 38 tok/s, 4.47 GB → **score 2.81-3.25**
- Improvement: **25-60% better** than bnb (large variance from baseline tps noise)

### Phi-3 small 8k instruct
- **BLOCKED**: requires pytest package (not in deps), in addition to tiktoken

## Strategy for Future Sessions

1. Start with the torchao streaming approach above (not bnb)
2. The `version=1` tinygemm path is deprecated — if fbgemm-gpu-genai becomes available, version=2 should be much faster
3. If loading a new model architecture, check that it has `model.model.layers` (might be `model.layers` or similar)
4. cuBLASLt behavior is model-specific — always test with and without
5. prompt_lookup scales higher with torchao than bnb — test 40/80/128 for new models
6. Quality retention is worse than bnb (0.90 vs 0.94 for Llama) — if quality floor (0.85) is a concern, try group_size=64

## What Works (consolidated)

- `torchao Int4WeightOnlyConfig(group_size=128, use_hqq=True, version=1)` — BEST quantization
- Stream layers CPU→GPU one at a time during quantization
- `prompt_lookup_num_tokens` — 40 for Llama, 256 for Mistral
- `torch_dtype=torch.bfloat16` for torchao (required by tinygemm kernels)
- cuBLASLt for Llama only (not Mistral)
- Default attention (no explicit attn_implementation)
