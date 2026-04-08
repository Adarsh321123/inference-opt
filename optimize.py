"""
LLM Inference Optimization — Weight Transformation + HQQ Pipeline
=================================================================
THIS IS THE FILE THE AGENT MODIFIES. Everything is fair game.

Pipeline:
1. Load model bf16 on CPU
2. Apply weight transformations (per-group outlier clipping)
3. Stream layers CPU→GPU, quantize with torchao int4 HQQ
4. Return optimized model
"""

import gc
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.backends.cudnn.benchmark = True

# ============================================================
# QUANTIZATION HYPERPARAMETERS
# ============================================================
GROUP_SIZE = 128


# ============================================================
# WEIGHT TRANSFORMATION — THE AGENT'S CREATIVE SPACE
# ============================================================

def clip_outliers_per_group(weight, clip_sigma=3.0):
    """
    Clip weight outliers within each quantization group.
    This reduces the dynamic range per group, allowing the quantizer
    to allocate more precision to the majority of weights.

    weight: [out_features, in_features]
    Groups are along the in_features dimension, size GROUP_SIZE.
    """
    out_f, in_f = weight.shape
    # Pad in_features to multiple of GROUP_SIZE
    n_groups = (in_f + GROUP_SIZE - 1) // GROUP_SIZE
    padded = n_groups * GROUP_SIZE

    if padded != in_f:
        return weight  # Don't clip if padding would be needed

    w = weight.float()
    # Reshape to [out_features, n_groups, group_size]
    w_grouped = w.reshape(out_f, n_groups, GROUP_SIZE)

    # Per-group statistics
    g_mean = w_grouped.mean(dim=2, keepdim=True)
    g_std = w_grouped.std(dim=2, keepdim=True).clamp(min=1e-8)

    # Clip to clip_sigma standard deviations within each group
    w_clipped = w_grouped.clamp(g_mean - clip_sigma * g_std, g_mean + clip_sigma * g_std)

    return w_clipped.reshape(out_f, in_f).to(weight.dtype)


def transform_weights_for_quantization(model):
    """
    Apply weight transformations before quantization.
    Currently: per-group outlier clipping to reduce quantization range.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.weight.data = clip_outliers_per_group(module.weight.data)


# ============================================================
# MAIN PIPELINE
# ============================================================

def optimize_model(model_name: str, device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load bf16 on CPU
    print("Loading model bf16 on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # --- Weight transformation ---
    print("Transforming weights...")
    transform_weights_for_quantization(model)

    # --- Quantize with torchao int4 HQQ ---
    print("Quantizing...")
    from torchao.quantization import quantize_, Int4WeightOnlyConfig
    config = Int4WeightOnlyConfig(group_size=GROUP_SIZE, use_hqq=True, version=1)

    model.model.embed_tokens.to(device)
    model.model.norm.to(device)
    model.model.rotary_emb.to(device)
    model.lm_head.to(device)

    for layer in model.model.layers:
        layer.to(device)
        quantize_(layer, config)
        gc.collect()
        torch.cuda.empty_cache()

    # Prompt lookup for speculative decoding
    is_llama = "llama" in model_name.lower()
    model.generation_config.prompt_lookup_num_tokens = 64 if is_llama else 256

    print("Done.")
    return model, tokenizer
