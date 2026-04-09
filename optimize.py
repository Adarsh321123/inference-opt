"""
LLM Inference Optimization — Fused Int4 Dequant+MatVec Kernel
=============================================================
Round 6: From-scratch int4 quantization with fused Triton kernel.

For batch=1 (generation): fused kernel reads int4, dequants in registers,
computes matvec, writes output — no intermediate bf16 buffer.
For batch>1 (prefill/perplexity): dequant to bf16, use cuBLAS matmul.
"""

import gc
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
TILE_M = 8  # rows per Triton program in fused kernel
HALF_GS = GROUP_SIZE // 2  # bytes per group (64)


# ============================================================
# TRITON KERNEL: fused int4 dequant + matrix-vector product
# ============================================================
@triton.jit
def int4_matvec_fused(
    output_ptr, input_ptr, w_tiled_ptr, s_tiled_ptr,
    M, groups_per_row,
    TILE_M: tl.constexpr, HALF_GS: tl.constexpr,
):
    """Fused int4 dequant+matvec for batch=1. Tiled weight layout."""
    tile_id = tl.program_id(0)
    row_idx = tile_id * TILE_M + tl.arange(0, TILE_M)
    row_mask = row_idx < M
    acc = tl.zeros((TILE_M,), dtype=tl.float32)

    tile_size = groups_per_row * TILE_M * HALF_GS
    scale_tile_size = groups_per_row * TILE_M
    group_stride = TILE_M * HALF_GS
    tile_base = tile_id * tile_size
    stile_base = tile_id * scale_tile_size
    pairs = groups_per_row // 2

    row_in_tile = tl.arange(0, TILE_M)
    byte_off = tl.arange(0, HALF_GS)

    for pair_id in range(0, pairs):
        g0 = pair_id * 2
        g1 = g0 + 1

        # Group 0
        p0 = tl.load(w_tiled_ptr + tile_base + g0 * group_stride + row_in_tile[:, None] * HALF_GS + byte_off[None, :])
        lo0 = ((p0 & 0xF).to(tl.int8) - 8).to(tl.float32)
        hi0 = (((p0 >> 4) & 0xF).to(tl.int8) - 8).to(tl.float32)
        ks0 = g0 * HALF_GS * 2
        xe0 = tl.load(input_ptr + ks0 + tl.arange(0, HALF_GS) * 2).to(tl.float32)
        xo0 = tl.load(input_ptr + ks0 + tl.arange(0, HALF_GS) * 2 + 1).to(tl.float32)
        s0 = tl.load(s_tiled_ptr + stile_base + g0 * TILE_M + row_in_tile).to(tl.float32)

        # Group 1
        p1 = tl.load(w_tiled_ptr + tile_base + g1 * group_stride + row_in_tile[:, None] * HALF_GS + byte_off[None, :])
        lo1 = ((p1 & 0xF).to(tl.int8) - 8).to(tl.float32)
        hi1 = (((p1 >> 4) & 0xF).to(tl.int8) - 8).to(tl.float32)
        ks1 = g1 * HALF_GS * 2
        xe1 = tl.load(input_ptr + ks1 + tl.arange(0, HALF_GS) * 2).to(tl.float32)
        xo1 = tl.load(input_ptr + ks1 + tl.arange(0, HALF_GS) * 2 + 1).to(tl.float32)
        s1 = tl.load(s_tiled_ptr + stile_base + g1 * TILE_M + row_in_tile).to(tl.float32)

        d0 = tl.sum(lo0 * xe0[None, :], axis=1) + tl.sum(hi0 * xo0[None, :], axis=1)
        d1 = tl.sum(lo1 * xe1[None, :], axis=1) + tl.sum(hi1 * xo1[None, :], axis=1)
        acc += d0 * s0 + d1 * s1

    tl.store(output_ptr + row_idx, acc, mask=row_mask)


# ============================================================
# WEIGHT PACKING + TILING
# ============================================================

def pack_int4(w_int4):
    """Pack int4 tensor into uint8 (2 values per byte). Input range [-8, 7]."""
    w_uint = (w_int4 + 8).to(torch.uint8)
    return w_uint[..., 0::2] | (w_uint[..., 1::2] << 4)


def tile_packed_weights(w_packed, scales, out_features, in_features, tile_m=TILE_M):
    """Rearrange into tiled layout [n_tiles, groups, tile_m, HALF_GS]."""
    groups = in_features // GROUP_SIZE
    n_tiles = out_features // tile_m
    w_tiled = w_packed.reshape(n_tiles, tile_m, groups, HALF_GS).permute(0, 2, 1, 3).contiguous()
    s_tiled = scales.reshape(n_tiles, tile_m, groups).permute(0, 2, 1).contiguous()
    return w_tiled, s_tiled


# ============================================================
# QUANTIZATION
# ============================================================

def compute_scales(weight, group_size=GROUP_SIZE, bits=BITS):
    w = weight.float()
    out_feat, in_feat = w.shape
    n_groups = in_feat // group_size
    w_grouped = w[:, :n_groups * group_size].reshape(out_feat, n_groups, group_size)
    absmax = w_grouped.abs().amax(dim=2).clamp(min=1e-8)
    return absmax / (2 ** bits // 2)


def quantize_weight(weight, scales, group_size=GROUP_SIZE, bits=BITS):
    w = weight.float()
    out_feat, in_feat = w.shape
    n_groups = in_feat // group_size
    half = 2 ** bits // 2
    w_grouped = w[:, :n_groups * group_size].reshape(out_feat, n_groups, group_size)
    w_scaled = w_grouped / scales.unsqueeze(2).clamp(min=1e-8)
    return torch.round(w_scaled).clamp(-half, half - 1).to(torch.int8).reshape(out_feat, n_groups * group_size)


# ============================================================
# CUSTOM LINEAR LAYER
# ============================================================

class QuantizedLinear(nn.Module):
    """Drop-in nn.Linear replacement with int4 weights.
    batch=1: fused Triton matvec kernel (fast, no intermediate buffer)
    batch>1: dequant to bf16 + cuBLAS matmul (for prefill/perplexity)
    """

    def __init__(self, in_features, out_features, w_tiled, s_tiled,
                 bias=None, tile_m=TILE_M):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tile_m = tile_m
        self.groups_per_row = in_features // GROUP_SIZE

        self.register_buffer('w_tiled', w_tiled)
        self.register_buffer('s_tiled', s_tiled)
        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.bias = None

    def _dequant(self, dtype=torch.bfloat16):
        """Dequantize tiled int4 weights on-the-fly."""
        wt = self.w_tiled
        lo = ((wt & 0xF).to(torch.int8) - 8).to(dtype)
        hi = (((wt >> 4) & 0xF).to(torch.int8) - 8).to(dtype)

        w_int = torch.empty(*wt.shape[:-1], HALF_GS * 2, dtype=dtype, device=wt.device)
        w_int[..., 0::2] = lo
        w_int[..., 1::2] = hi

        w_float = w_int * self.s_tiled.unsqueeze(-1).to(dtype)
        return w_float.permute(0, 2, 1, 3).reshape(self.out_features, self.in_features)

    def forward(self, x):
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)
        batch = x_flat.shape[0]

        if batch == 1:
            # Fused kernel: fast path for generation
            output = torch.empty(1, self.out_features, device=x.device, dtype=torch.float32)
            n_tiles = self.out_features // self.tile_m
            int4_matvec_fused[(n_tiles,)](
                output[0], x_flat[0], self.w_tiled, self.s_tiled,
                self.out_features, self.groups_per_row,
                TILE_M=self.tile_m, HALF_GS=HALF_GS,
                num_warps=2,
            )
            output = output.to(x.dtype)
        else:
            # Dequant + cuBLAS: handles prefill/perplexity efficiently
            w_deq = self._dequant(x.dtype)
            output = x_flat @ w_deq.t()

        if self.bias is not None:
            output = output + self.bias

        return output.reshape(*orig_shape[:-1], self.out_features)


# ============================================================
# QUANTIZE A LINEAR MODULE
# ============================================================

def quantize_linear(module):
    weight = module.weight.data
    bias = module.bias.data if module.bias is not None else None
    out_feat, in_feat = weight.shape

    if in_feat % GROUP_SIZE != 0 or out_feat % TILE_M != 0:
        return None  # skip layers that don't fit

    scales = compute_scales(weight)
    w_int4 = quantize_weight(weight, scales)
    w_packed = pack_int4(w_int4)
    w_tiled, s_tiled = tile_packed_weights(w_packed, scales, out_feat, in_feat)

    return QuantizedLinear(in_feat, out_feat, w_tiled, s_tiled, bias=bias)


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

    # Quantize all linear layers on CPU (no GPU needed for quantization)
    print("Quantizing with fused int4 kernel...")
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        # Skip lm_head: +3.5% quality retention, minimal memory cost
        if "lm_head" in name:
            continue

        q_linear = quantize_linear(module)
        if q_linear is None:
            continue

        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, q_linear)

    # Move to GPU
    model.to(device)
    gc.collect()
    torch.cuda.empty_cache()

    # Inference config
    is_llama = "llama" in model_name.lower()
    model.generation_config.prompt_lookup_num_tokens = 64 if is_llama else 256

    print("Done.")
    return model, tokenizer
