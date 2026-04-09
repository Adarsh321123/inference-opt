"""
LLM Inference Optimization — From-Scratch Quantization
=======================================================
THIS IS THE FILE THE AGENT MODIFIES. Everything is fair game.

This file implements int4 quantization and inference FROM SCRATCH:
- Custom quantization math (scale, rounding, grouping)
- Custom Triton kernel for dequantize + matmul
- Custom nn.Linear replacement
- Calibration-aware quantization using activation statistics

No torchao, no bitsandbytes, no library quantization calls.
The agent controls every line of the algorithm.
"""

import gc
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# HYPERPARAMETERS
# ============================================================
BITS = 4
GROUP_SIZE = 128
CALIBRATION_SAMPLES = 128
CALIBRATION_SEQ_LEN = 512


# ============================================================
# TRITON KERNEL: int4 dequantize + matrix-vector product
# ============================================================

@triton.jit
def int4_matvec_kernel(
    # Pointers
    output_ptr, input_ptr, weight_packed_ptr, scales_ptr, zeros_ptr,
    # Dimensions
    M,  # out_features
    K,  # in_features
    groups_per_row,  # K // GROUP_SIZE
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Compute y = W @ x where W is stored as packed int4 with per-group scales.

    Weight packing: two int4 values packed per uint8 byte.
    weight_packed shape: [M, K // 2]
    scales shape: [M, groups_per_row]
    zeros shape: [M, groups_per_row]
    input shape: [K]
    output shape: [M]
    """
    row_idx = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row_idx < M

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_idx < K

        # Load input activations
        x = tl.load(input_ptr + k_idx, mask=k_mask, other=0.0).to(tl.float32)

        # Load packed int4 weights (2 values per byte)
        byte_idx = k_idx // 2
        packed_offset = row_idx[:, None] * (K // 2) + byte_idx[None, :]
        packed = tl.load(weight_packed_ptr + packed_offset, mask=row_mask[:, None] & k_mask[None, :], other=0)

        # Unpack: even indices = low nibble, odd indices = high nibble
        is_high = (k_idx % 2 == 1)
        unpacked = tl.where(is_high[None, :], (packed >> 4) & 0xF, packed & 0xF)

        # Convert from unsigned [0, 15] to signed [-8, 7]
        w_int = (unpacked.to(tl.int8) - 8).to(tl.float32)

        # Load per-group scale and zero
        group_idx = k_idx // BLOCK_K  # approximate group index
        group_id = k_start // GROUP_SIZE
        scale_offset = row_idx * groups_per_row + group_id
        s = tl.load(scales_ptr + scale_offset, mask=row_mask, other=1.0).to(tl.float32)
        z = tl.load(zeros_ptr + scale_offset, mask=row_mask, other=0.0).to(tl.float32)

        # Dequantize: w_float = (w_int - zero) * scale
        w_float = (w_int - z[:, None]) * s[:, None]

        # Dot product accumulation
        acc += tl.sum(w_float * x[None, :], axis=1)

    tl.store(output_ptr + row_idx, acc, mask=row_mask)


# ============================================================
# WEIGHT PACKING
# ============================================================

def pack_int4(w_int4):
    """Pack int4 tensor into uint8 (2 values per byte). Input range [-8, 7]."""
    w_uint = (w_int4 + 8).to(torch.uint8)  # shift to [0, 15]
    assert w_uint.shape[-1] % 2 == 0, "in_features must be even"
    low = w_uint[..., 0::2]
    high = w_uint[..., 1::2]
    return low | (high << 4)


# ============================================================
# QUANTIZATION MATH — THE AGENT'S CREATIVE SPACE
# ============================================================

def compute_scales(weight, group_size=GROUP_SIZE, bits=BITS):
    """
    Compute per-group quantization scales and zero points.

    THE AGENT MODIFIES THIS to implement better scale computation:
    - Percentile clipping (ignore top 0.1%)
    - MSE-optimal scale (grid search)
    - Asymmetric quantization
    - Activation-aware scaling
    """
    w = weight.float()
    out_feat, in_feat = w.shape
    n_groups = in_feat // group_size
    n_levels = 2 ** bits

    w_grouped = w[:, :n_groups * group_size].reshape(out_feat, n_groups, group_size)

    # Baseline: symmetric min-max
    absmax = w_grouped.abs().amax(dim=2).clamp(min=1e-8)
    scales = absmax / (n_levels // 2)
    zeros = torch.zeros_like(scales)

    return scales, zeros


def quantize_weight(weight, scales, zeros, group_size=GROUP_SIZE, bits=BITS):
    """
    Quantize weight matrix using precomputed scales and zeros.
    Returns int4 values in range [-8, 7].

    THE AGENT MODIFIES THIS to implement better rounding:
    - GPTQ-style: use Hessian to pick optimal rounding direction
    - Stochastic rounding
    - Error feedback across groups
    """
    w = weight.float()
    out_feat, in_feat = w.shape
    n_groups = in_feat // group_size
    n_levels = 2 ** bits
    half = n_levels // 2

    w_grouped = w[:, :n_groups * group_size].reshape(out_feat, n_groups, group_size)

    # Baseline: round to nearest
    w_scaled = w_grouped / scales.unsqueeze(2).clamp(min=1e-8)
    w_int = torch.round(w_scaled).clamp(-half, half - 1).to(torch.int8)

    return w_int.reshape(out_feat, n_groups * group_size)


# ============================================================
# CUSTOM LINEAR LAYER
# ============================================================

class QuantizedLinear(nn.Module):
    """Drop-in replacement for nn.Linear using int4 weights + Triton kernel."""

    def __init__(self, in_features, out_features, weight_packed, scales, zeros,
                 bias=None, group_size=GROUP_SIZE):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.groups_per_row = in_features // group_size

        self.register_buffer('weight_packed', weight_packed)
        self.register_buffer('scales', scales)
        self.register_buffer('zeros', zeros)
        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.bias = None

    def forward(self, x):
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)
        batch = x_flat.shape[0]

        output = torch.empty(batch, self.out_features, device=x.device, dtype=torch.float32)

        BLOCK_M = 64
        BLOCK_K = self.group_size  # process one group at a time
        grid = ((self.out_features + BLOCK_M - 1) // BLOCK_M,)

        for b in range(batch):
            int4_matvec_kernel[grid](
                output[b], x_flat[b], self.weight_packed, self.scales, self.zeros,
                self.out_features, self.in_features, self.groups_per_row,
                BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
            )

        output = output.to(x.dtype)
        if self.bias is not None:
            output = output + self.bias

        return output.reshape(*orig_shape[:-1], self.out_features)


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
    """Collect per-linear-layer activation absmax and Hessian diagonal."""
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
                    "H_diag": torch.zeros(n, device=flat.device),
                    "count": 0,
                }
            s = stats[name]
            s["absmax"] = torch.max(s["absmax"], flat.abs().max(dim=0).values)
            s["H_diag"] += (flat ** 2).sum(dim=0)
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
        s["H_diag"] = s["H_diag"] / max(s["count"], 1)

    return stats


# ============================================================
# QUANTIZE A LINEAR MODULE
# ============================================================

def quantize_linear(module, stats=None):
    """
    Quantize an nn.Linear → QuantizedLinear.
    Agent can modify to use stats for calibration-aware quantization.
    """
    weight = module.weight.data
    bias = module.bias.data if module.bias is not None else None
    out_feat, in_feat = weight.shape

    # Ensure in_features is divisible by group_size and by 2 (for packing)
    assert in_feat % GROUP_SIZE == 0, f"in_features {in_feat} not divisible by {GROUP_SIZE}"
    assert in_feat % 2 == 0, f"in_features {in_feat} must be even for int4 packing"

    # Compute scales (agent modifies compute_scales)
    scales, zeros = compute_scales(weight, GROUP_SIZE, BITS)

    # Quantize (agent modifies quantize_weight)
    w_int4 = quantize_weight(weight, scales, zeros, GROUP_SIZE, BITS)

    # Pack into uint8
    w_packed = pack_int4(w_int4)

    return QuantizedLinear(
        in_feat, out_feat, w_packed, scales, zeros,
        bias=bias, group_size=GROUP_SIZE,
    )


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

    # --- Calibration ---
    print("Collecting activation statistics...")
    calib_data = load_calibration_data(tokenizer)
    model.to(device)
    act_stats = collect_activation_stats(model, calib_data, device)
    model.to("cpu")
    torch.cuda.empty_cache()

    # --- Quantize all linear layers ---
    print("Quantizing...")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name) if parent_name else model

            stats = act_stats.get(name)
            q_linear = quantize_linear(module, stats)
            q_linear.to(device)
            setattr(parent, child_name, q_linear)

    # Move remaining modules to GPU
    model.to(device)
    gc.collect()
    torch.cuda.empty_cache()

    # --- Inference config ---
    is_llama = "llama" in model_name.lower()
    model.generation_config.prompt_lookup_num_tokens = 64 if is_llama else 256

    print("Done.")
    return model, tokenizer
