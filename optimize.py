"""
LLM Inference Optimization — HQQ Int4 + Weight-Only AWQ
========================================================
THIS IS THE FILE THE AGENT MODIFIES. Everything is fair game.
"""

import gc
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.backends.cudnn.benchmark = True

GROUP_SIZE = 128


def awq_scale_layer_weight_only(layer, alpha=0.5):
    """
    AWQ-style channel scaling using only weight statistics (no calibration).
    Uses per-column weight magnitude as importance proxy.
    """
    attn = layer.self_attn

    # Attention: scale input_layernorm ↔ q/k/v
    qkv = [attn.q_proj, attn.k_proj, attn.v_proj]
    w_max = torch.zeros(qkv[0].weight.shape[1], dtype=torch.float32)
    for lin in qkv:
        w_max = torch.max(w_max, lin.weight.data.float().abs().max(dim=0).values)
    w_max.clamp_(min=1e-8)

    s = (w_max / w_max.mean()).pow(alpha)
    s.clamp_(min=0.8, max=1.25)

    dtype = layer.input_layernorm.weight.dtype
    s_dev = s.to(dtype=dtype)
    layer.input_layernorm.weight.data.div_(s_dev)
    for lin in qkv:
        lin.weight.data.mul_(s_dev.unsqueeze(0).to(dtype=lin.weight.dtype))

    # MLP: scale post_attention_layernorm ↔ gate/up
    mlp = layer.mlp
    gate_up = [mlp.gate_proj, mlp.up_proj]
    w_max = torch.zeros(gate_up[0].weight.shape[1], dtype=torch.float32)
    for lin in gate_up:
        w_max = torch.max(w_max, lin.weight.data.float().abs().max(dim=0).values)
    w_max.clamp_(min=1e-8)

    s = (w_max / w_max.mean()).pow(alpha)
    s.clamp_(min=0.8, max=1.25)

    dtype = layer.post_attention_layernorm.weight.dtype
    s_dev = s.to(dtype=dtype)
    layer.post_attention_layernorm.weight.data.div_(s_dev)
    for lin in gate_up:
        lin.weight.data.mul_(s_dev.unsqueeze(0).to(dtype=lin.weight.dtype))


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

    # Weight-only AWQ scaling (no calibration needed)
    print("Applying AWQ scaling...")
    for layer in model.model.layers:
        awq_scale_layer_weight_only(layer)

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
