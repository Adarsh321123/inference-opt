"""
LLM Inference Optimization — HQQ Int4 + AWQ scaling
=====================================================
THIS IS THE FILE THE AGENT MODIFIES. Everything is fair game.
"""

import gc
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.backends.cudnn.benchmark = True

GROUP_SIZE = 128


def load_calibration_data(tokenizer, n_samples=32, seq_len=512):
    """Load calibration samples from WikiText-2."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in ds["text"] if len(t.strip()) > 200][:n_samples * 2]
    encodings = []
    for text in texts:
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
        if tokens.input_ids.shape[1] >= 64:
            encodings.append(tokens.input_ids)
        if len(encodings) >= n_samples:
            break
    return encodings


def collect_activation_stats(model, calib_data, device="cuda:0"):
    """Collect per-channel activation absmax for each linear layer."""
    stats = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, inp, out):
            x = inp[0].detach().float()
            flat = x.reshape(-1, x.shape[-1])
            if name not in stats:
                stats[name] = torch.zeros(flat.shape[-1], device=flat.device)
            stats[name] = torch.max(stats[name], flat.abs().max(dim=0).values)
        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for input_ids in calib_data:
            try:
                model(input_ids.to(device))
            except Exception:
                continue

    for h in hooks:
        h.remove()

    return stats


def awq_scale_layer(layer, layer_idx, act_stats, alpha=0.5):
    """
    AWQ-style channel scaling for one transformer layer.
    Scale important channels UP in weight, compensate in LayerNorm.
    """
    attn = layer.self_attn

    # --- Attention: q/k/v share input from input_layernorm ---
    qkv = [attn.q_proj, attn.k_proj, attn.v_proj]
    q_key = f"model.layers.{layer_idx}.self_attn.q_proj"
    if q_key in act_stats:
        a_scale = act_stats[q_key].cpu().float().clamp(min=1e-8)
        w_max = torch.zeros_like(a_scale)
        for lin in qkv:
            w_max = torch.max(w_max, lin.weight.data.float().abs().max(dim=0).values)
        w_max.clamp_(min=1e-8)

        # AWQ: s = (a^alpha * w^(1-alpha)) / mean, clamped conservatively
        s = a_scale.pow(alpha) * w_max.pow(1 - alpha)
        s = s / s.mean().clamp(min=1e-8)
        s.clamp_(min=0.8, max=1.25)  # Very conservative

        dtype = layer.input_layernorm.weight.dtype
        s_dev = s.to(dtype=dtype)
        layer.input_layernorm.weight.data.div_(s_dev)
        for lin in qkv:
            lin.weight.data.mul_(s_dev.unsqueeze(0).to(dtype=lin.weight.dtype))

    # --- MLP: gate/up share input from post_attention_layernorm ---
    mlp = layer.mlp
    gate_up = [mlp.gate_proj, mlp.up_proj]
    g_key = f"model.layers.{layer_idx}.mlp.gate_proj"
    if g_key in act_stats:
        a_scale = act_stats[g_key].cpu().float().clamp(min=1e-8)
        w_max = torch.zeros_like(a_scale)
        for lin in gate_up:
            w_max = torch.max(w_max, lin.weight.data.float().abs().max(dim=0).values)
        w_max.clamp_(min=1e-8)

        s = a_scale.pow(alpha) * w_max.pow(1 - alpha)
        s = s / s.mean().clamp(min=1e-8)
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

    # Calibration
    print("Calibrating...")
    calib_data = load_calibration_data(tokenizer)
    model.to(device)
    act_stats = collect_activation_stats(model, calib_data, device)
    model.to("cpu")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # AWQ-style weight transformation
    print("Applying AWQ scaling...")
    for i, layer in enumerate(model.model.layers):
        awq_scale_layer(layer, i, act_stats, alpha=0.5)

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
