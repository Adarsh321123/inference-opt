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

The agent modifies how weights are TRANSFORMED before quantization.
This is where AWQ, QuIP#, and SmoothQuant innovate — the math that
makes weights more quantization-friendly. torchao handles the actual
int4 packing and inference kernels.
"""

import gc
import math
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# QUANTIZATION HYPERPARAMETERS
# ============================================================
BITS = 4
GROUP_SIZE = 128
CALIBRATION_SAMPLES = 128
CALIBRATION_SEQ_LEN = 512


# ============================================================
# CALIBRATION
# ============================================================

def load_calibration_data(tokenizer, n_samples=CALIBRATION_SAMPLES, seq_len=CALIBRATION_SEQ_LEN):
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
    activation statistics: mean, std, absmax per input channel.
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
    Per-input-channel importance scores. Used to decide which channels
    need protection during quantization.

    Agent can modify: change alpha, use std instead of absmax,
    add Hessian information, use different norms, etc.
    """
    w_scale = weight.abs().max(dim=0).values.clamp(min=1e-8)

    if act_stats is not None and "absmax" in act_stats:
        a_scale = act_stats["absmax"].to(weight.device).clamp(min=1e-8)
        importance = (a_scale ** alpha) * (w_scale ** (1 - alpha))
    else:
        importance = w_scale

    return importance


def compute_weight_sensitivity(weight, act_stats=None):
    """
    Per-weight sensitivity: how much does quantizing this weight affect output?
    Returns [out_features, in_features] tensor.

    Agent can replace with Hessian diagonal, Fisher information, etc.
    """
    w_mag = weight.abs()
    if act_stats is not None and "absmax" in act_stats:
        a_mag = act_stats["absmax"].to(weight.device).clamp(min=1e-8)
        return w_mag * a_mag.unsqueeze(0)
    return w_mag


# ============================================================
# WEIGHT TRANSFORMATION — THE AGENT'S CREATIVE SPACE
# ============================================================

def transform_weights_for_quantization(model, act_stats):
    """
    Apply mathematical transformations to model weights before quantization.
    Has access to ALL layers for cross-layer compensation.

    THIS IS THE FUNCTION THE AGENT SHOULD MODIFY.

    Baseline: no transformation (naive round-to-nearest via torchao).

    Techniques to explore:
    - AWQ-style channel scaling: scale important channels up in layer N,
      compensate by scaling down in layer N-1's output projection.
      Key: importance = (activation_mag ** alpha) * (weight_mag ** (1-alpha))
    - Outlier clipping: clip extreme weight values to reduce quantization range.
      Trade small accuracy loss on outliers for better precision on the bulk.
    - Hadamard rotation (QuIP#): multiply W by orthogonal Hadamard matrix
      to spread information uniformly. Requires compensating in adjacent layers.
    - SmoothQuant: migrate quantization difficulty from activations to weights
      by per-channel scaling: W_new = W * diag(s), X_new = X / diag(s)
    - Per-channel zero-centering: shift each channel to be symmetric around zero
      for better symmetric quantization.
    - Novel combinations of the above.

    The agent has access to:
    - model.model.layers[i] — all transformer layers
    - act_stats — per-layer activation statistics from calibration
    - compute_channel_importance() — importance scoring
    - compute_weight_sensitivity() — per-weight sensitivity
    """
    layers = model.model.layers

    for i, layer in enumerate(layers):
        prev_layer = layers[i - 1] if i > 0 else None
        next_layer = layers[i + 1] if i < len(layers) - 1 else None

        # === BASELINE: No transformation ===
        # The agent replaces this with real math.
        pass


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

    # --- Calibration ---
    print("Collecting activation statistics...")
    calib_data = load_calibration_data(tokenizer)
    model.to(device)
    act_stats = collect_activation_stats(model, calib_data, device)
    model.to("cpu")
    torch.cuda.empty_cache()

    # --- Weight transformation (the agent's creative space) ---
    print("Transforming weights...")
    transform_weights_for_quantization(model, act_stats)

    # --- Quantize with torchao int4 ---
    print("Quantizing...")
    from torchao.quantization import quantize_, Int4WeightOnlyConfig
    quant_config = Int4WeightOnlyConfig(group_size=GROUP_SIZE, use_hqq=True, version=1)

    model.model.embed_tokens.to(device)
    model.model.norm.to(device)
    model.model.rotary_emb.to(device)
    model.lm_head.to(device)

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
