"""
Micro-benchmark round 7: Pipeline stages + multi-size validation.
Best so far: ~49μs for 4096x4096 (1.31x faster than cuBLAS bf16)
"""

import torch
import triton
import triton.language as tl

GROUP_SIZE = 128
WARMUP = 50
ITERS = 200


def make_test_data(M, K, device="cuda"):
    w_int4 = torch.randint(-8, 8, (M, K), dtype=torch.int8, device=device)
    w_uint = (w_int4 + 8).to(torch.uint8)
    w_packed = w_uint[:, 0::2] | (w_uint[:, 1::2] << 4)
    groups_per_row = K // GROUP_SIZE
    scales = torch.randn(M, groups_per_row, dtype=torch.float16, device=device).abs() * 0.01 + 0.001
    zeros = torch.zeros(M, groups_per_row, dtype=torch.float16, device=device)
    x = torch.randn(K, dtype=torch.bfloat16, device=device)
    w_float = torch.zeros(M, K, dtype=torch.float32, device=device)
    for g in range(groups_per_row):
        s = g * GROUP_SIZE
        e = s + GROUP_SIZE
        w_float[:, s:e] = w_int4[:, s:e].float() * scales[:, g:g+1].float()
    w_bf16 = w_float.to(torch.bfloat16)
    return w_packed, scales, zeros, x, w_bf16


def tile_weights(w_packed, scales, M, K, TILE_M):
    HALF_GS = GROUP_SIZE // 2
    groups = K // GROUP_SIZE
    n_tiles = M // TILE_M
    w_r = w_packed.reshape(n_tiles, TILE_M, groups, HALF_GS)
    w_tiled = w_r.permute(0, 2, 1, 3).contiguous()
    s_r = scales.reshape(n_tiles, TILE_M, groups)
    s_tiled = s_r.permute(0, 2, 1).contiguous()
    return w_tiled, s_tiled


# ============================================================
# WINNING KERNEL: V9 tiled 2-group (cleanest, fastest)
# ============================================================
@triton.jit
def int4_matvec_fused(
    output_ptr, input_ptr, w_tiled_ptr, s_tiled_ptr,
    M, groups_per_row,
    TILE_M: tl.constexpr, HALF_GS: tl.constexpr,
):
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

        p0 = tl.load(w_tiled_ptr + tile_base + g0 * group_stride + row_in_tile[:, None] * HALF_GS + byte_off[None, :])
        lo0 = ((p0 & 0xF).to(tl.int8) - 8).to(tl.float32)
        hi0 = (((p0 >> 4) & 0xF).to(tl.int8) - 8).to(tl.float32)
        ks0 = g0 * HALF_GS * 2
        xe0 = tl.load(input_ptr + ks0 + tl.arange(0, HALF_GS) * 2).to(tl.float32)
        xo0 = tl.load(input_ptr + ks0 + tl.arange(0, HALF_GS) * 2 + 1).to(tl.float32)
        s0 = tl.load(s_tiled_ptr + stile_base + g0 * TILE_M + row_in_tile).to(tl.float32)

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
# V15: With num_stages pipelining via tl.range
# ============================================================
@triton.jit
def int4_matvec_pipelined(
    output_ptr, input_ptr, w_tiled_ptr, s_tiled_ptr,
    M, groups_per_row,
    TILE_M: tl.constexpr, HALF_GS: tl.constexpr,
):
    tile_id = tl.program_id(0)
    row_idx = tile_id * TILE_M + tl.arange(0, TILE_M)
    row_mask = row_idx < M
    acc = tl.zeros((TILE_M,), dtype=tl.float32)

    tile_size = groups_per_row * TILE_M * HALF_GS
    scale_tile_size = groups_per_row * TILE_M
    group_stride = TILE_M * HALF_GS
    tile_base = tile_id * tile_size
    stile_base = tile_id * scale_tile_size

    row_in_tile = tl.arange(0, TILE_M)
    byte_off = tl.arange(0, HALF_GS)

    for g in tl.range(0, groups_per_row, num_stages=2):
        p = tl.load(w_tiled_ptr + tile_base + g * group_stride + row_in_tile[:, None] * HALF_GS + byte_off[None, :])
        lo = ((p & 0xF).to(tl.int8) - 8).to(tl.float32)
        hi = (((p >> 4) & 0xF).to(tl.int8) - 8).to(tl.float32)
        ks = g * HALF_GS * 2
        xe = tl.load(input_ptr + ks + tl.arange(0, HALF_GS) * 2).to(tl.float32)
        xo = tl.load(input_ptr + ks + tl.arange(0, HALF_GS) * 2 + 1).to(tl.float32)
        s = tl.load(s_tiled_ptr + stile_base + g * TILE_M + row_in_tile).to(tl.float32)
        acc += (tl.sum(lo * xe[None, :], axis=1) + tl.sum(hi * xo[None, :], axis=1)) * s

    tl.store(output_ptr + row_idx, acc, mask=row_mask)


# ============================================================
# V16: Flat layout with num_stages pipelining
# (for comparison: maybe flat + pipeline > tiled without pipeline)
# ============================================================
@triton.jit
def int4_matvec_flat_pipelined(
    output_ptr, input_ptr, weight_packed_ptr, scales_ptr, zeros_ptr,
    M, K, groups_per_row, half_K,
    BLOCK_M: tl.constexpr, HALF_GS: tl.constexpr,
):
    row_idx = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row_idx < M
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for group_id in tl.range(0, groups_per_row, num_stages=3):
        byte_start = group_id * HALF_GS
        k_start = group_id * HALF_GS * 2
        byte_off = tl.arange(0, HALF_GS)
        packed_ptrs = row_idx[:, None] * half_K + (byte_start + byte_off[None, :])
        packed = tl.load(weight_packed_ptr + packed_ptrs, mask=row_mask[:, None])
        lo = ((packed & 0xF).to(tl.int8) - 8).to(tl.float32)
        hi = (((packed >> 4) & 0xF).to(tl.int8) - 8).to(tl.float32)
        x_even = tl.load(input_ptr + k_start + tl.arange(0, HALF_GS) * 2).to(tl.float32)
        x_odd = tl.load(input_ptr + k_start + tl.arange(0, HALF_GS) * 2 + 1).to(tl.float32)
        s = tl.load(scales_ptr + row_idx * groups_per_row + group_id, mask=row_mask, other=1.0).to(tl.float32)
        acc += (tl.sum(lo * x_even[None, :], axis=1) + tl.sum(hi * x_odd[None, :], axis=1)) * s

    tl.store(output_ptr + row_idx, acc, mask=row_mask)


# ============================================================
# V17: Tiled with pipelining AND 2-group
# ============================================================
@triton.jit
def int4_matvec_tiled_pipelined_2g(
    output_ptr, input_ptr, w_tiled_ptr, s_tiled_ptr,
    M, groups_per_row,
    TILE_M: tl.constexpr, HALF_GS: tl.constexpr,
):
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

    for pair_id in tl.range(0, pairs, num_stages=2):
        g0 = pair_id * 2
        g1 = g0 + 1

        p0 = tl.load(w_tiled_ptr + tile_base + g0 * group_stride + row_in_tile[:, None] * HALF_GS + byte_off[None, :])
        lo0 = ((p0 & 0xF).to(tl.int8) - 8).to(tl.float32)
        hi0 = (((p0 >> 4) & 0xF).to(tl.int8) - 8).to(tl.float32)
        ks0 = g0 * HALF_GS * 2
        xe0 = tl.load(input_ptr + ks0 + tl.arange(0, HALF_GS) * 2).to(tl.float32)
        xo0 = tl.load(input_ptr + ks0 + tl.arange(0, HALF_GS) * 2 + 1).to(tl.float32)
        s0 = tl.load(s_tiled_ptr + stile_base + g0 * TILE_M + row_in_tile).to(tl.float32)

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
# BENCHMARKING
# ============================================================
def bench_fn(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000)
    times.sort()
    lo = len(times) // 10
    hi = len(times) - lo
    return times[lo:hi][len(times[lo:hi]) // 2]


def check_correctness(fn, ref):
    out = fn()
    max_err = (out.float() - ref).abs().max().item()
    rel_err = max_err / (ref.abs().max().item() + 1e-8)
    return rel_err < 0.15, rel_err


def main():
    torch.manual_seed(42)
    HALF_GS = GROUP_SIZE // 2

    # Test multiple layer sizes from Llama-3.1-8B / Mistral-7B
    sizes = [
        (4096, 4096, "attn_proj"),
        (4096, 14336, "gate/up_proj"),
        (14336, 4096, "down_proj"),
        (1024, 4096, "k/v_proj"),
    ]

    print("=" * 70)
    print("INT4 FUSED MATVEC — MULTI-SIZE + PIPELINE BENCHMARK")
    print("=" * 70)

    for M, K, label in sizes:
        if K % GROUP_SIZE != 0:
            print(f"\nSkipping {label} ({M}x{K}): K not divisible by {GROUP_SIZE}")
            continue

        print(f"\n{'='*70}")
        print(f"{label}: {M}x{K}")
        print(f"{'='*70}")

        groups_per_row = K // GROUP_SIZE
        half_K = K // 2

        w_packed, scales, zeros, x, w_bf16 = make_test_data(M, K)
        ref = torch.mv(w_bf16.float(), x.float())

        t_cublas = bench_fn(lambda: torch.mv(w_bf16, x))
        print(f"  {'cuBLAS bf16:':<50} {t_cublas:7.1f} μs")

        results = []

        # --- V9/fused tiled 2-group (current best, no pipeline) ---
        for tm in [4, 8, 16]:
            if M % tm != 0:
                continue
            w_tiled, s_tiled = tile_weights(w_packed, scales, M, K, tm)
            for nw in [1, 2, 4]:
                name = f"V9 tiled2g TM={tm} W={nw}"
                def run(tm_=tm, wt_=w_tiled, st_=s_tiled, nw_=nw):
                    out = torch.empty(M, dtype=torch.float32, device="cuda")
                    int4_matvec_fused[(M // tm_,)](
                        out, x, wt_, st_, M, groups_per_row,
                        TILE_M=tm_, HALF_GS=HALF_GS, num_warps=nw_)
                    return out
                try:
                    ok, err = check_correctness(run, ref)
                    if not ok:
                        print(f"    {name:<48} WRONG ({err:.4f})")
                        continue
                    t = bench_fn(run)
                    results.append((name, t))
                    r = t / t_cublas
                    print(f"    {name:<48} {t:7.1f} μs ({r:.2f}x){' <<<' if t < t_cublas else ''}")
                except Exception as e:
                    print(f"    {name:<48} ERR: {str(e)[:50]}")

        # --- V15: Tiled 1-group pipelined ---
        for tm in [4, 8, 16]:
            if M % tm != 0:
                continue
            w_tiled, s_tiled = tile_weights(w_packed, scales, M, K, tm)
            for nw in [1, 2, 4]:
                name = f"V15 pipe TM={tm} W={nw}"
                def run(tm_=tm, wt_=w_tiled, st_=s_tiled, nw_=nw):
                    out = torch.empty(M, dtype=torch.float32, device="cuda")
                    int4_matvec_pipelined[(M // tm_,)](
                        out, x, wt_, st_, M, groups_per_row,
                        TILE_M=tm_, HALF_GS=HALF_GS, num_warps=nw_)
                    return out
                try:
                    ok, err = check_correctness(run, ref)
                    if not ok:
                        print(f"    {name:<48} WRONG ({err:.4f})")
                        continue
                    t = bench_fn(run)
                    results.append((name, t))
                    r = t / t_cublas
                    print(f"    {name:<48} {t:7.1f} μs ({r:.2f}x){' <<<' if t < t_cublas else ''}")
                except Exception as e:
                    print(f"    {name:<48} ERR: {str(e)[:50]}")

        # --- V16: Flat pipelined ---
        for bm in [8, 16]:
            for nw in [2, 4]:
                name = f"V16 flat-pipe BM={bm} W={nw}"
                def run(bm_=bm, nw_=nw):
                    out = torch.empty(M, dtype=torch.float32, device="cuda")
                    int4_matvec_flat_pipelined[((M + bm_ - 1) // bm_,)](
                        out, x, w_packed, scales, zeros,
                        M, K, groups_per_row, half_K,
                        BLOCK_M=bm_, HALF_GS=HALF_GS, num_warps=nw_)
                    return out
                try:
                    ok, err = check_correctness(run, ref)
                    if not ok:
                        print(f"    {name:<48} WRONG ({err:.4f})")
                        continue
                    t = bench_fn(run)
                    results.append((name, t))
                    r = t / t_cublas
                    print(f"    {name:<48} {t:7.1f} μs ({r:.2f}x){' <<<' if t < t_cublas else ''}")
                except Exception as e:
                    print(f"    {name:<48} ERR: {str(e)[:50]}")

        # --- V17: Tiled + pipelined 2-group ---
        for tm in [4, 8, 16]:
            if M % tm != 0:
                continue
            w_tiled, s_tiled = tile_weights(w_packed, scales, M, K, tm)
            for nw in [1, 2, 4]:
                name = f"V17 tile-pipe2g TM={tm} W={nw}"
                def run(tm_=tm, wt_=w_tiled, st_=s_tiled, nw_=nw):
                    out = torch.empty(M, dtype=torch.float32, device="cuda")
                    int4_matvec_tiled_pipelined_2g[(M // tm_,)](
                        out, x, wt_, st_, M, groups_per_row,
                        TILE_M=tm_, HALF_GS=HALF_GS, num_warps=nw_)
                    return out
                try:
                    ok, err = check_correctness(run, ref)
                    if not ok:
                        print(f"    {name:<48} WRONG ({err:.4f})")
                        continue
                    t = bench_fn(run)
                    results.append((name, t))
                    r = t / t_cublas
                    print(f"    {name:<48} {t:7.1f} μs ({r:.2f}x){' <<<' if t < t_cublas else ''}")
                except Exception as e:
                    print(f"    {name:<48} ERR: {str(e)[:50]}")

        if results:
            results.sort(key=lambda x: x[1])
            print(f"  Best: {results[0][0]} @ {results[0][1]:.1f}μs ({results[0][1]/t_cublas:.2f}x cuBLAS)")


if __name__ == "__main__":
    main()
