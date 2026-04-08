"""
LLM Inference Optimization — From-Scratch Quantization Pipeline
================================================================
THIS IS THE FILE THE AGENT MODIFIES. Everything is fair game.

Pipeline:
1. Load model bf16 on CPU
2. Collect calibration data → activation statistics per layer
3. Transform weights using custom math (THE AGENT'S CREATIVE SPACE)
4. Quantize with torchao int4 (fast tinygemm kernels for inference)
5. Return optimized model

The agent should focus on transform_weights_for_quantization() and
compute_channel_importance() to discover novel techniques that produce
better quality at 4-bit than existing methods (AWQ, GPTQ, QuIP#).
"""

import gc
import math
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# CALIBRATION
# ============================================================

def load_calibration_data(tokenizer, n_samples=128, seq_len=512):
    """Load calibration samples from WikiText-2."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in ds["text"] if len(t.strip()) > 200][:n_samples]
    encodings = []
    for text in texts:
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
        if tokens.input_ids.shape[1] >= 64:
            encodings.append(tokens.input_ids)
    return encodings


def collect_activation_stats(model, calib_data, device="cuda:0"):
    """
    Run calibration data through the model and collect per-linear-layer
    activation statistics: mean, std, max per input channel.
    """
    stats = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, inp, out):
            x = inp[0].detach().float()
            flat = x.reshape(-1, x.shape[-1])
            if name not in stats:
                stats[name] = {
                    "sum": torch.zeros(flat.shape[-1], device=flat.device),
                    "sq_sum": torch.zeros(flat.shape[-1], device=flat.device),
                    "absmax": torch.zeros(flat.shape[-1], device=flat.device),
                    "count": 0,
                }
            s = stats[name]
            s["sum"] += flat.sum(dim=0)
            s["sq_sum"] += (flat ** 2).sum(dim=0)
            s["absmax"] = torch.max(s["absmax"], flat.abs().max(dim=0).values)
            s["count"] += flat.shape[0]
        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for input_ids in calib_data[:32]:
            try:
                model(input_ids.to(device))
            except Exception:
                continue

    for h in hooks:
        h.remove()

    # Finalize stats
    for name in stats:
        s = stats[name]
        s["mean"] = s["sum"] / max(s["count"], 1)
        s["std"] = torch.sqrt(s["sq_sum"] / max(s["count"], 1) - s["mean"] ** 2).clamp(min=0)
        del s["sum"], s["sq_sum"]

    return stats


# ============================================================
# WEIGHT ANALYSIS
# ============================================================

def compute_channel_importance(weight, act_stats=None, alpha=0.5):
    """
    Compute per-input-channel importance scores.

    The agent can modify this to implement novel importance metrics.
    Channels with high importance should be protected during quantization
    (e.g., by scaling them up before quantization).

    Args:
        weight: [out_features, in_features] weight matrix
        act_stats: dict with 'absmax', 'mean', 'std' per channel
        alpha: balance between activation and weight importance (0=weight only, 1=activation only)

    Returns:
        importance: [in_features] tensor of per-channel importance scores
    """
    w_scale = weight.abs().max(dim=0).values.clamp(min=1e-8)

    if act_stats is not None and "absmax" in act_stats:
        a_scale = act_stats["absmax"].to(weight.device).clamp(min=1e-8)
        importance = (a_scale ** alpha) * (w_scale ** (1 - alpha))
    else:
        importance = w_scale

    return importance


# ============================================================
# WEIGHT TRANSFORMATION — THE AGENT'S CREATIVE SPACE
# ============================================================

def transform_weights_for_quantization(layer, layer_idx, act_stats):
    """
    Apply mathematical transformation to layer weights before quantization.

    THIS IS THE PRIMARY FUNCTION THE AGENT SHOULD MODIFY.

    The baseline does nothing (naive round-to-nearest quantization).
    The agent should replace this with techniques that make weights
    more quantization-friendly, such as:

    - Channel scaling: scale important channels up to protect them,
      compensate by scaling down in the next layer
    - Outlier clipping: clip extreme weight values to reduce quantization range
    - Rotation: apply orthogonal transformation (e.g., Hadamard) to spread
      information more uniformly across channels
    - Zero-centering: shift weight distributions to be symmetric around zero
    - Custom math: anything that reduces quantization error

    Args:
        layer: a transformer decoder layer (has self_attn, mlp sub-modules)
        layer_idx: integer index of this layer
        act_stats: dict mapping sub-module names to activation statistics
    """
    # === BASELINE: No transformation ===
    # This means torchao will do naive round-to-nearest quantization.
    # The agent should replace this with something better.
    #
    # Example: AWQ-style channel scaling
    # for name, module in layer.named_modules():
    #     if isinstance(module, nn.Linear):
    #         w = module.weight.data.float()
    #         stats = act_stats.get(f"model.layers.{layer_idx}.{name}", None)
    #         importance = compute_channel_importance(w, stats, alpha=0.5)
    #         scale = (importance / importance.mean()).sqrt().clamp(0.1, 10.0)
    #         module.weight.data = (w * scale.unsqueeze(0)).to(module.weight.dtype)
    #
    # Example: Outlier clipping at 3 sigma
    # for name, module in layer.named_modules():
    #     if isinstance(module, nn.Linear):
    #         w = module.weight.data
    #         threshold = w.abs().mean() + 3 * w.std()
    #         module.weight.data = w.clamp(-threshold, threshold)

    pass


# ============================================================
# QUANTIZATION
# ============================================================

def get_quant_config():
    """
    Return the torchao quantization config.
    Agent can modify group_size, use_hqq, or replace entirely.
    """
    from torchao.quantization import Int4WeightOnlyConfig
    return Int4WeightOnlyConfig(group_size=128, use_hqq=True, version=1)


# ============================================================
# MAIN PIPELINE
# ============================================================

def optimize_model(model_name: str, device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load bf16 on CPU to avoid GPU memory spike during quantization
    print("Loading model bf16 on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # --- Calibration ---
    print("Collecting activation statistics...")
    calib_data = load_calibration_data(tokenizer, n_samples=128, seq_len=512)
    model.to(device)
    act_stats = collect_activation_stats(model, calib_data, device)
    model.to("cpu")
    torch.cuda.empty_cache()

    # --- Weight transformation (agent's creative space) ---
    print("Transforming weights...")
    for i, layer in enumerate(model.model.layers):
        transform_weights_for_quantization(layer, i, act_stats)

    # --- Quantize with torchao int4 ---
    print("Quantizing...")
    from torchao.quantization import quantize_
    quant_config = get_quant_config()

    # Move non-quantizable layers to GPU
    model.model.embed_tokens.to(device)
    model.model.norm.to(device)
    model.model.rotary_emb.to(device)
    model.lm_head.to(device)

    # Stream each layer: CPU → GPU, quantize on GPU
    for i, layer in enumerate(model.model.layers):
        layer.to(device)
        quantize_(layer, quant_config)
        gc.collect()
        torch.cuda.empty_cache()

    # --- Inference config ---
    is_llama = "llama" in model_name.lower()
    model.generation_config.prompt_lookup_num_tokens = 64 if is_llama else 256

    print("Done.")
    return model, tokenizer
