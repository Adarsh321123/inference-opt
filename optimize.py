"""
LLM Inference Optimization — HQQ Int4 + Targeted Outlier Clipping
==================================================================
THIS IS THE FILE THE AGENT MODIFIES. Everything is fair game.
"""

import gc
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.backends.cudnn.benchmark = True

GROUP_SIZE = 128


def clip_group_outliers(model, clip_sigma=6.0):
    """
    Clip extreme weight outliers within each quantization group.
    At 6σ, only affects ~0.1% of groups (those with extreme outliers).
    Most groups have max at 3-4σ and are completely unaffected.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            w = module.weight.data
            out_f, in_f = w.shape
            n_groups = in_f // GROUP_SIZE
            if n_groups == 0 or n_groups * GROUP_SIZE != in_f:
                continue

            w_float = w.float()
            w_grouped = w_float.reshape(out_f, n_groups, GROUP_SIZE)

            g_mean = w_grouped.mean(dim=2, keepdim=True)
            g_std = w_grouped.std(dim=2, keepdim=True).clamp(min=1e-8)

            lo = g_mean - clip_sigma * g_std
            hi = g_mean + clip_sigma * g_std
            w_clipped = w_grouped.clamp(lo, hi)

            module.weight.data = w_clipped.reshape(out_f, in_f).to(w.dtype)


def optimize_model(model_name: str, device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print("Loading model bf16 on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # Clip extreme weight outliers before quantization
    print("Clipping outliers...")
    clip_group_outliers(model, clip_sigma=6.0)

    # Quantize with torchao int4 HQQ
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
