"""
LLM Inference Optimization — Custom Quantization Pipeline
==========================================================
THIS IS THE FILE THE AGENT MODIFIES. Everything is fair game.

Round 5 approach: REPLACE HQQ's quantization math, not work around it.

torchao's quantize_() pipeline:
1. HQQ/RTN computes int4 values for each weight group
2. Weights are packed into TensorCoreTiledLayout
3. tinygemm kernels do fast int4 inference

We replace step 1 with custom math. Steps 2-3 stay the same.
The agent writes better quantization functions that produce higher-quality
int4 values than HQQ, using calibration data and novel algorithms.

Two modes:
- USE_CUSTOM_QUANTIZATION=True: custom math → dequantize → torchao RTN repack
  (RTN on already-quantized weights preserves custom rounding decisions)
- USE_CUSTOM_QUANTIZATION=False: plain HQQ (baseline to beat)
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
USE_CUSTOM_QUANTIZATION = False  # Agent sets True when ready


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
    Run calibration data and collect per-linear-layer stats:
    absmax, mean, std per channel, and Hessian H = X^T X for GPTQ.
    """
    stats = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, inp, out):
            x = inp[0].detach().float()
            flat = x.reshape(-1, x.shape[-1])
            if name not in stats:
                n = flat.shape[-1]
                stats[name] = {
                    "absmax": torch.zeros(n, device=flat.device),
                    "sq_sum": torch.zeros(n, device=flat.device),
                    "H": torch.zeros(n, n, device=flat.device),
                    "count": 0,
                }
            s = stats[name]
            s["absmax"] = torch.max(s["absmax"], flat.abs().max(dim=0).values)
            s["sq_sum"] += (flat ** 2).sum(dim=0)
            s["H"] += flat.T @ flat
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
        cnt = max(s["count"], 1)
        s["channel_var"] = s["sq_sum"] / cnt
        s["H"] = s["H"] / cnt
        del s["sq_sum"]

    return stats


# ============================================================
# CUSTOM QUANTIZATION — THE AGENT'S CREATIVE SPACE
# ============================================================

def quantize_weight_matrix(weight, hessian=None, bits=BITS, group_size=GROUP_SIZE):
    """
    Quantize a weight matrix [out_features, in_features] to N-bit.
    Returns dequantized weight (simulated quantization).
    All ops are VECTORIZED — no Python loops over groups.

    THIS IS THE CORE FUNCTION THE AGENT SHOULD MODIFY.

    Baseline: symmetric min-max with round-to-nearest (RTN).

    The agent can implement:
    - Asymmetric quantization: different range for pos/neg
    - Percentile clipping: ignore top 0.1% when computing scale
    - MSE-optimal scale: binary search for scale minimizing MSE
    - GPTQ-style rounding: use Hessian diagonal to pick rounding
      direction that minimizes output error per weight
    - Error feedback across groups
    - Non-uniform quantization grids
    - Novel approaches combining the above
    """
    w = weight.float()
    out_feat, in_feat = w.shape
    n_levels = 2 ** bits
    half = n_levels // 2

    # Trim to multiple of group_size
    n_groups_per_row = in_feat // group_size
    usable = n_groups_per_row * group_size
    w_main = w[:, :usable]
    w_remainder = w[:, usable:] if usable < in_feat else None

    # Reshape: [out_feat, n_groups_per_row, group_size]
    w_grouped = w_main.reshape(out_feat, n_groups_per_row, group_size)

    # === SCALE COMPUTATION (vectorized over all groups) ===
    # Baseline: symmetric min-max
    absmax = w_grouped.abs().amax(dim=2, keepdim=True).clamp(min=1e-8)
    scale = absmax / half

    # === ROUNDING (vectorized) ===
    # Baseline: round to nearest
    w_scaled = w_grouped / scale
    w_int = torch.round(w_scaled).clamp(-half, half - 1)

    # === DEQUANTIZE ===
    w_deq = (w_int * scale).reshape(out_feat, usable)

    # Handle remainder (not quantized — kept at full precision)
    if w_remainder is not None:
        w_deq = torch.cat([w_deq, w_remainder], dim=1)

    return w_deq.to(weight.dtype)


def apply_custom_quantization(layer, layer_idx, act_stats):
    """
    Apply custom quantization to all linear modules in a layer.
    Replaces weights with dequantized versions (custom rounding decisions).
    """
    for name, module in layer.named_modules():
        if isinstance(module, nn.Linear):
            stats_key = f"model.layers.{layer_idx}.{name}"
            layer_stats = act_stats.get(stats_key)
            hessian = layer_stats["H"] if layer_stats and "H" in layer_stats else None

            w_deq = quantize_weight_matrix(
                module.weight.data, hessian, BITS, GROUP_SIZE
            )
            module.weight.data = w_deq


# ============================================================
# MAIN PIPELINE
# ============================================================

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

    # --- Calibration (only for custom quantization) ---
    act_stats = {}
    if USE_CUSTOM_QUANTIZATION:
        print("Collecting activation statistics + Hessians...")
        calib_data = load_calibration_data(tokenizer)
        model.to(device)
        act_stats = collect_activation_stats(model, calib_data, device)
        model.to("cpu")
        torch.cuda.empty_cache()

    # --- Quantization ---
    print("Quantizing...")
    from torchao.quantization import quantize_, Int4WeightOnlyConfig

    # When custom=True: use RTN to repack (preserves custom rounding decisions)
    # When custom=False: use HQQ (baseline to beat)
    use_hqq = not USE_CUSTOM_QUANTIZATION
    config = Int4WeightOnlyConfig(group_size=GROUP_SIZE, use_hqq=use_hqq, version=1)

    model.model.embed_tokens.to(device)
    model.model.norm.to(device)
    if hasattr(model.model, 'rotary_emb'):
        model.model.rotary_emb.to(device)
    model.lm_head.to(device)

    for i, layer in enumerate(model.model.layers):
        layer.to(device)

        if USE_CUSTOM_QUANTIZATION:
            # Custom quantize → dequantize (replaces weights with optimally-rounded bf16)
            apply_custom_quantization(layer, i, act_stats)

        # torchao packs into int4 format for fast tinygemm inference
        quantize_(layer, config)
        gc.collect()
        torch.cuda.empty_cache()

    # --- Inference config ---
    is_llama = "llama" in model_name.lower()
    model.generation_config.prompt_lookup_num_tokens = 64 if is_llama else 256

    print("Done.")
    return model, tokenizer
