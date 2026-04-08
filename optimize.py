"""
LLM Inference Optimization — From-Scratch Quantization Pipeline
================================================================
THIS IS THE FILE THE AGENT MODIFIES. Everything is fair game.

Pipeline:
1. Load model bf16 on CPU
2. Collect calibration data → activation statistics per layer
3. Transform weights using custom math (CREATIVE SPACE #1)
4. Quantize weights using custom math (CREATIVE SPACE #2)
5. Pack quantized weights for fast inference
6. Return optimized model

The agent modifies the quantization MATH: how weights are analyzed,
transformed, scaled, grouped, and rounded. The goal is to discover
techniques that produce better quality at 4-bit than existing methods.
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
    need higher quantization precision.

    Agent can modify the importance metric: change alpha, use std instead
    of absmax, add Hessian information, etc.
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
    Estimate how sensitive the layer output is to quantization error in each weight.
    Higher sensitivity = this weight needs more precise quantization.

    Agent can replace this with Hessian diagonal, Fisher information,
    or any other sensitivity metric.
    """
    # Simple sensitivity: weight magnitude * activation magnitude
    w_mag = weight.abs()
    if act_stats is not None and "absmax" in act_stats:
        a_mag = act_stats["absmax"].to(weight.device).clamp(min=1e-8)
        return w_mag * a_mag.unsqueeze(0)  # [out, in]
    return w_mag


# ============================================================
# WEIGHT TRANSFORMATION — CREATIVE SPACE #1
# ============================================================

def transform_weights_for_quantization(model, act_stats):
    """
    Apply mathematical transformations to model weights before quantization.
    The agent has access to ALL layers and can do cross-layer compensation.

    THIS IS CREATIVE SPACE #1.

    Baseline: no transformation.

    Techniques to explore:
    - AWQ-style channel scaling: scale important channels up in layer N,
      scale down corresponding outputs in layer N-1 to compensate
    - Outlier clipping: clip extreme values to reduce quantization range
    - Hadamard rotation: rotate weight matrices for more uniform distribution
    - SmoothQuant: balance quantization difficulty between weights and activations
    - Per-channel zero-centering
    - Novel combinations
    """
    layers = model.model.layers

    for i, layer in enumerate(layers):
        prev_layer = layers[i - 1] if i > 0 else None
        next_layer = layers[i + 1] if i < len(layers) - 1 else None

        # === BASELINE: No transformation ===
        # The agent replaces this with real math.
        pass


# ============================================================
# QUANTIZATION MATH — CREATIVE SPACE #2
# ============================================================

def compute_group_scales(weight, group_size=GROUP_SIZE, bits=BITS):
    """
    Compute per-group quantization scale and zero-point.

    Agent can modify:
    - Symmetric vs asymmetric quantization
    - Scale computation (min-max, percentile, MSE-optimal)
    - Group boundaries (uniform, adaptive based on weight distribution)
    """
    w = weight.float()
    out_features, in_features = w.shape

    # Pad if needed
    if in_features % group_size != 0:
        pad = group_size - (in_features % group_size)
        w = nn.functional.pad(w, (0, pad))

    # Reshape into groups
    w_grouped = w.reshape(-1, group_size)  # [n_groups, group_size]

    # Symmetric min-max scaling
    n_levels = 2 ** bits
    half = n_levels // 2
    absmax = w_grouped.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scale = absmax / half

    return scale, w_grouped


def quantize_weights(weight, group_size=GROUP_SIZE, bits=BITS):
    """
    Quantize a weight matrix to N-bit integers with per-group scaling.
    Returns the DEQUANTIZED weight (simulated quantization).

    Agent can modify:
    - Rounding: nearest, stochastic, GPTQ-style optimal rounding
    - Clipping: clip outliers before computing scale
    - Non-uniform quantization: log-scale, learned grid, k-means
    - Error feedback: propagate rounding error to subsequent weights
    """
    w = weight.float()
    orig_shape = w.shape
    out_features, in_features = orig_shape

    n_levels = 2 ** bits
    half = n_levels // 2

    # Pad if needed
    padded = False
    if in_features % group_size != 0:
        pad = group_size - (in_features % group_size)
        w = nn.functional.pad(w, (0, pad))
        padded = True

    w_grouped = w.reshape(-1, group_size)

    # === Scale computation ===
    # Baseline: symmetric min-max
    absmax = w_grouped.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scale = absmax / half

    # === Quantize ===
    # Baseline: round to nearest
    w_int = torch.round(w_grouped / scale).clamp(-half, half - 1)

    # === Dequantize ===
    w_deq = w_int * scale

    # Reshape back
    w_deq = w_deq.reshape(w.shape)
    if padded:
        w_deq = w_deq[:, :in_features]

    return w_deq.to(weight.dtype)


def quantize_layer_custom(layer, act_stats=None, layer_idx=0):
    """
    Apply custom quantization to all linear layers in a transformer layer.
    Uses quantize_weights() which the agent can modify.
    """
    for name, module in layer.named_modules():
        if isinstance(module, nn.Linear):
            w = module.weight.data
            stats_key = f"model.layers.{layer_idx}.{name}"
            stats = act_stats.get(stats_key) if act_stats else None

            # Quantize → dequantize (simulated quantization)
            w_quantized = quantize_weights(w, GROUP_SIZE, BITS)
            module.weight.data = w_quantized


# ============================================================
# QUANTIZATION BACKEND SELECTION
# ============================================================

USE_TORCHAO = True  # True = torchao int4 kernels (fast). False = simulated quantization (flexible).


def quantize_layer(layer, quant_config, act_stats=None, layer_idx=0):
    """
    Quantize a layer. Two modes:

    USE_TORCHAO=True: Use torchao int4 kernels for fast inference.
      The agent's transform_weights_for_quantization() runs BEFORE this,
      and torchao handles the actual int4 packing.

    USE_TORCHAO=False: Use custom quantize_weights() for full control.
      Simulated quantization — model stays bf16 but weights have int4 precision.
      Slower inference, but the agent controls every aspect of the math.

    The agent can switch between modes to test quality (simulated) vs speed (torchao).
    """
    if USE_TORCHAO:
        from torchao.quantization import quantize_
        quantize_(layer, quant_config)
    else:
        quantize_layer_custom(layer, act_stats, layer_idx)


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

    # --- Weight transformation (creative space #1) ---
    print("Transforming weights...")
    transform_weights_for_quantization(model, act_stats)

    # --- Quantization (creative space #2) ---
    print("Quantizing...")
    if USE_TORCHAO:
        from torchao.quantization import Int4WeightOnlyConfig
        quant_config = Int4WeightOnlyConfig(group_size=GROUP_SIZE, use_hqq=True, version=1)
    else:
        quant_config = None

    # Move non-quantizable layers to GPU
    model.model.embed_tokens.to(device)
    model.model.norm.to(device)
    model.model.rotary_emb.to(device)
    model.lm_head.to(device)

    # Stream each layer: CPU → GPU, quantize
    for i, layer in enumerate(model.model.layers):
        layer.to(device)
        quantize_layer(layer, quant_config, act_stats, i)
        gc.collect()
        torch.cuda.empty_cache()

    # --- Inference config ---
    is_llama = "llama" in model_name.lower()
    model.generation_config.prompt_lookup_num_tokens = 64 if is_llama else 256

    print("Done.")
    return model, tokenizer
