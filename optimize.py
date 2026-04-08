"""
LLM Inference Optimization — HQQ Int4 + torch.compile
======================================================
THIS IS THE FILE THE AGENT MODIFIES. Everything is fair game.

Pipeline:
1. Load model bf16 on CPU
2. Stream layers CPU→GPU, quantize with torchao int4 HQQ
3. torch.compile for kernel fusion speedup
4. Return optimized model
"""

import gc
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.backends.cudnn.benchmark = True

GROUP_SIZE = 128


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

    # torch.compile for kernel fusion
    model = torch.compile(model, mode="reduce-overhead")

    # Prompt lookup for speculative decoding
    is_llama = "llama" in model_name.lower()
    model.generation_config.prompt_lookup_num_tokens = 64 if is_llama else 256

    print("Done.")
    return model, tokenizer
