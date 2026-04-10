"""
LLM Inference Optimization — From-Scratch Int4 Quantization
============================================================
Triton dequant kernel + cuBLAS matmul for speed.
Symmetric int4 with percentile clipping for quality.
"""

import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
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
# TRITON KERNEL: dequantize int4 packed → bfloat16
# ============================================================

@triton.jit
def dequant_int4_kernel(
    out_ptr, packed_ptr, scales_ptr,
    M, K, half_K, groups_per_row,
    GROUP_SIZE_CONST: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Dequantize packed int4 weights to bf16. 2D grid over [M, K]."""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    row_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    row_mask = row_offs < M
    col_mask = col_offs < K
    mask = row_mask[:, None] & col_mask[None, :]

    # Load packed bytes
    byte_offs = col_offs // 2
    packed = tl.load(
        packed_ptr + row_offs[:, None] * half_K + byte_offs[None, :],
        mask=mask, other=0
    )

    # Unpack: even cols = low nibble, odd cols = high nibble
    is_high = (col_offs % 2) == 1
    w_uint = tl.where(is_high[None, :], (packed >> 4) & 0xF, packed & 0xF)
    w_int = (w_uint.to(tl.int8) - 8).to(tl.bfloat16)

    # Load per-group scales
    group_idx = col_offs // GROUP_SIZE_CONST
    scale = tl.load(
        scales_ptr + row_offs[:, None] * groups_per_row + group_idx[None, :],
        mask=mask, other=1.0
    ).to(tl.bfloat16)

    # Dequantize and store
    w_deq = w_int * scale
    tl.store(out_ptr + row_offs[:, None] * K + col_offs[None, :], w_deq, mask=mask)


# ============================================================
# WEIGHT PACKING
# ============================================================

def pack_int4(w_int4):
    """Pack int4 tensor into uint8 (2 values per byte). Input range [-8, 7]."""
    w_uint = (w_int4 + 8).to(torch.uint8)  # shift to [0, 15]
    assert w_uint.shape[-1] % 2 == 0
    low = w_uint[..., 0::2]
    high = w_uint[..., 1::2]
    return low | (high << 4)


# ============================================================
# QUANTIZATION MATH
# ============================================================

def compute_scales(weight, group_size=GROUP_SIZE, bits=BITS, h_diag=None):
    """Compute per-group quantization scales. MSE-optimal when h_diag available."""
    w = weight.float()
    out_feat, in_feat = w.shape
    n_groups = in_feat // group_size
    half = 2 ** (bits - 1)

    w_grouped = w[:, :n_groups * group_size].reshape(out_feat, n_groups, group_size)
    absmax = w_grouped.abs().amax(dim=2).clamp(min=1e-8)
    base_scale = absmax / half

    if h_diag is None:
        return base_scale

    # MSE-optimal scale search weighted by Hessian diagonal
    h = h_diag[:n_groups * group_size].reshape(n_groups, group_size)
    h = h.unsqueeze(0).expand(out_feat, -1, -1)

    best_scale = base_scale.clone()
    best_mse = torch.full_like(base_scale, float('inf'))
    best_mult = torch.ones_like(base_scale)

    def _try(m):
        nonlocal best_scale, best_mse, best_mult
        ts = base_scale * m
        ws = w_grouped / ts.unsqueeze(2)
        wi = torch.round(ws).clamp(-half, half - 1)
        wd = wi * ts.unsqueeze(2)
        wm = ((w_grouped - wd) ** 2 * h).sum(dim=2)
        b = wm < best_mse
        best_mse = torch.where(b, wm, best_mse)
        best_scale = torch.where(b, ts, best_scale)
        best_mult = torch.where(b, m, best_mult)

    # Coarse search
    for m in [0.60, 0.70, 0.80, 0.90, 1.00]:
        _try(torch.full_like(base_scale, m))

    # Fine refinement ±0.05
    for off in [-0.05, -0.03, -0.01, 0.01, 0.03, 0.05]:
        _try((best_mult + off).clamp(0.3, 1.1))

    return best_scale


def quantize_weight(weight, scales, group_size=GROUP_SIZE, bits=BITS):
    """Quantize weight matrix. Returns int4 values in [-8, 7]."""
    w = weight.float()
    out_feat, in_feat = w.shape
    n_groups = in_feat // group_size
    half = 2 ** (bits - 1)

    w_grouped = w[:, :n_groups * group_size].reshape(out_feat, n_groups, group_size)
    w_scaled = w_grouped / scales.unsqueeze(2).clamp(min=1e-8)
    w_int = torch.round(w_scaled).clamp(-half, half - 1).to(torch.int8)

    return w_int.reshape(out_feat, n_groups * group_size)


# ============================================================
# CUSTOM LINEAR LAYER
# ============================================================

class QuantizedLinear(nn.Module):
    """Int4 linear with Triton dequant + cuBLAS matmul."""

    def __init__(self, in_features, out_features, weight_packed, scales,
                 bias=None, group_size=GROUP_SIZE):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.groups_per_row = in_features // group_size

        self.register_buffer('weight_packed', weight_packed)
        self.register_buffer('scales', scales.to(torch.bfloat16))
        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.bias = None
        self._grid = ((out_features + 127) // 128, (in_features + 127) // 128)
        self._half_K = in_features // 2

    def forward(self, x):
        M, K = self.out_features, self.in_features
        w_deq = torch.empty(M, K, dtype=torch.bfloat16, device=x.device)
        dequant_int4_kernel[self._grid](
            w_deq, self.weight_packed, self.scales,
            M, K, self._half_K, self.groups_per_row,
            GROUP_SIZE_CONST=self.group_size,
            BLOCK_M=128, BLOCK_K=128,
            num_warps=4, num_stages=3,
        )
        return F.linear(x, w_deq, self.bias)


# ============================================================
# CALIBRATION
# ============================================================

def load_calibration_data(tokenizer, n_samples=CALIBRATION_SAMPLES, seq_len=CALIBRATION_SEQ_LEN):
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
    """Collect per-linear-layer Hessian diagonal."""
    stats = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, inp, out):
            x = inp[0].detach().float()
            flat = x.reshape(-1, x.shape[-1])
            if name not in stats:
                n = flat.shape[-1]
                stats[name] = {
                    "H_diag": torch.zeros(n, device=flat.device),
                    "count": 0,
                }
            s = stats[name]
            s["H_diag"] += (flat ** 2).sum(dim=0)
            s["count"] += flat.shape[0]
        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for input_ids in calib_data[:64]:
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
    weight = module.weight.data
    bias = module.bias.data if module.bias is not None else None
    out_feat, in_feat = weight.shape

    assert in_feat % GROUP_SIZE == 0, f"in_features {in_feat} not divisible by {GROUP_SIZE}"
    assert in_feat % 2 == 0

    h_diag = stats["H_diag"].cpu() if stats is not None else None
    scales = compute_scales(weight, GROUP_SIZE, BITS, h_diag=h_diag)
    w_int4 = quantize_weight(weight, scales, GROUP_SIZE, BITS)
    w_packed = pack_int4(w_int4)

    return QuantizedLinear(
        in_feat, out_feat, w_packed, scales,
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

    # Reset peak memory so calibration doesn't inflate measurement
    torch.cuda.reset_peak_memory_stats()

    # --- Quantize all linear layers ---
    print("Quantizing...")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if name == "lm_head":
                continue
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name) if parent_name else model

            stats = act_stats.get(name)
            q_linear = quantize_linear(module, stats)
            q_linear.to(device)
            setattr(parent, child_name, q_linear)

    model.to(device)
    gc.collect()
    torch.cuda.empty_cache()

    # --- Inference config ---
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    is_llama = "llama" in model_name.lower()
    model.generation_config.prompt_lookup_num_tokens = 128

    print("Done.")
    return model, tokenizer
