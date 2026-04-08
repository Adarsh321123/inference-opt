"""
LLM Inference Optimization — HQQ Int4 + Bias Correction
========================================================
THIS IS THE FILE THE AGENT MODIFIES. Everything is fair game.

Pipeline:
1. Load model bf16 on CPU
2. Calibrate to get per-layer activation means
3. Stream layers CPU→GPU, quantize with HQQ int4
4. Add bias correction to compensate for mean quantization error
5. Return optimized model
"""

import gc
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.backends.cudnn.benchmark = True

GROUP_SIZE = 128
CALIBRATION_SEQLEN = 512


def load_calibration_data(tokenizer, n_samples=32, seq_len=CALIBRATION_SEQLEN):
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


def collect_activation_means(model, calib_data, device="cuda:0"):
    """Collect per-channel mean activations for each linear layer."""
    means = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, inp, out):
            x = inp[0].detach().float()
            flat = x.reshape(-1, x.shape[-1])
            if name not in means:
                means[name] = {"sum": torch.zeros(flat.shape[-1], device=flat.device), "count": 0}
            means[name]["sum"] += flat.sum(dim=0)
            means[name]["count"] += flat.shape[0]
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

    result = {}
    for name, d in means.items():
        result[name] = d["sum"] / max(d["count"], 1)

    return result


def apply_bias_correction(layer, layer_idx, original_weights, act_means):
    """
    After quantization, add bias to compensate for mean quantization error.

    For each linear: bias = -(Q(W) - W) @ E[x] = -quant_error @ mean_activation
    """
    for proj_name, linear in [
        ("self_attn.q_proj", layer.self_attn.q_proj),
        ("self_attn.k_proj", layer.self_attn.k_proj),
        ("self_attn.v_proj", layer.self_attn.v_proj),
        ("self_attn.o_proj", layer.self_attn.o_proj),
        ("mlp.gate_proj", layer.mlp.gate_proj),
        ("mlp.up_proj", layer.mlp.up_proj),
        ("mlp.down_proj", layer.mlp.down_proj),
    ]:
        full_name = f"model.layers.{layer_idx}.{proj_name}"
        if full_name not in act_means or proj_name not in original_weights:
            continue

        mean_act = act_means[full_name].to(linear.weight.device)
        w_orig = original_weights[proj_name].to(linear.weight.device).float()

        # Dequantize to get Q(W)
        w_quant = linear.weight.dequantize().float()

        # Quantization error per output channel
        # error_per_output = (Q(W) - W) @ mean_act → shape [out_features]
        quant_error = w_quant - w_orig
        bias_correction = -(quant_error @ mean_act)

        # Add bias (creates it if doesn't exist)
        if linear.bias is not None:
            linear.bias.data += bias_correction.to(linear.bias.dtype)
        else:
            linear.bias = nn.Parameter(bias_correction.to(torch.bfloat16))


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

    # Calibration: collect activation means
    print("Calibrating...")
    calib_data = load_calibration_data(tokenizer)
    model.to(device)
    act_means = collect_activation_means(model, calib_data, device)
    model.to("cpu")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Quantize with torchao int4 HQQ + bias correction
    print("Quantizing with bias correction...")
    from torchao.quantization import quantize_, Int4WeightOnlyConfig
    config = Int4WeightOnlyConfig(group_size=GROUP_SIZE, use_hqq=True, version=1)

    model.model.embed_tokens.to(device)
    model.model.norm.to(device)
    model.model.rotary_emb.to(device)
    model.lm_head.to(device)

    for i, layer in enumerate(model.model.layers):
        # Save original bf16 weights before quantization
        original_weights = {}
        for name, mod in layer.named_modules():
            if isinstance(mod, nn.Linear):
                original_weights[name] = mod.weight.data.clone()

        # Move to GPU and quantize
        layer.to(device)
        quantize_(layer, config)

        # Apply bias correction
        apply_bias_correction(layer, i, original_weights, act_means)

        del original_weights
        gc.collect()
        torch.cuda.empty_cache()

    # Prompt lookup for speculative decoding
    is_llama = "llama" in model_name.lower()
    model.generation_config.prompt_lookup_num_tokens = 64 if is_llama else 256

    print("Done.")
    return model, tokenizer
