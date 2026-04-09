# knowledge

Accumulated insights from 200+ experiments across 5 rounds. Read this before every session.

## Rounds 1-3: Library Configs
Best: Llama 2.7, Mistral 3.9 (torchao HQQ int4 + prompt_lookup).
HQQ quality: Llama 0.90, Mistral 0.96. HQQ speed: 1.3x baseline.

## Round 4: Weight Transforms Before HQQ
ALL failed. HQQ already optimizes for the distribution. Transforms interfere.

## Round 5: From-Scratch Quantization
Built complete int4 system from scratch: Triton dequant kernel, Hessian-weighted
MSE-optimal scales, QuantizedLinear replacement.
- Quality MATCHES HQQ (Llama 0.90, Mistral 0.97)
- Memory BETTER than HQQ (2.61x vs 2.23x Llama, 3.33x vs 2.48x Mistral)
- Speed WORSE than HQQ (1.0x vs 1.3x Llama, 0.70x vs 1.6x Mistral)
- Final scores: Llama 2.35, Mistral 2.26 (worse than HQQ due to speed)

KEY INSIGHT: if we match HQQ speed, we WIN:
- Llama: 0.90 * 1.3 * 2.61 = 3.05 (beats HQQ's 2.7)
- Mistral: 0.97 * 1.6 * 3.33 = 5.17 (beats HQQ's 3.9 by 33%)

The quality math is solved. The memory is better. SPEED IS THE ONLY GAP.

## Round 6: Fused Kernel Optimization

### WHY ROUND 5 FAILED ON SPEED

The dequant+cuBLAS approach does TWO memory passes:
1. Triton kernel reads int4, writes bf16 to VRAM
2. cuBLAS reads bf16 from VRAM for matmul

tinygemm (HQQ's kernel) does ONE pass: read int4, dequant in registers, matmul
in registers, write output. This fused approach skips writing the full bf16 weight.

For a 4096x4096 layer at batch=1, the extra memory pass costs ~30% speed.
A fused kernel eliminates this.

### WHY THE AGENT GAVE UP TOO EARLY

In round 5, the agent tried a fused matvec kernel ONCE. It was 20x slower.
The agent discarded it and moved on. But kernel optimization needs 50-100
iterations on a FAST feedback loop, not one shot in a 5-minute pipeline.

### THE FIX: KERNEL MICRO-BENCHMARK

Write a standalone benchmark script that:
1. Creates random tensors matching a real layer (4096x4096 int4 packed, bf16 input)
2. Runs the Triton kernel 100 times
3. Reports throughput in GB/s and tok/s equivalent
4. Takes <1 second total

Use THIS as the inner loop fitness function. Iterate the kernel 100+ times
against this benchmark. Only run the full model evaluation once the kernel
is competitive with dequant+cuBLAS (>35 tok/s equivalent).

### KERNEL OPTIMIZATION TARGETS

The 3090 has 936 GB/s memory bandwidth.
For 4096x4096 int4 at batch=1: 8MB weight data → ~9μs theoretical minimum.
Current dequant+cuBLAS: ~25μs per layer.
Current fused (round 5 attempt): ~500μs (20x too slow).
Target: <15μs per layer (fused, competitive with tinygemm).

### WHAT TO ITERATE ON IN THE KERNEL

- Tiling strategy: BLOCK_M, BLOCK_K values (try 32, 64, 128, 256)
- Memory access pattern: coalesced reads of packed int4
- Shared memory: load weight tile to shared, then compute
- Warp specialization: some warps load, others compute
- Output accumulation: fp32 accumulator, cast to bf16 at end
- L2 cache: tile sizes that fit in L2 (6MB on 3090)
- Register pressure: balance between tile size and register usage

### KEY FINDING FROM ROUND 5

num_warps=4 was 32% faster than num_warps=8 for the dequant-only kernel.
Lower warp count works better for memory-bound kernels on 3090.
Start kernel iterations with num_warps=4.

## Transferable Findings (All Rounds)
- prompt_lookup: 64 for Llama, 256 for Mistral (essential for int4)
- Hessian-weighted MSE-optimal scale: best quality technique (+1.5%)
- Skip lm_head quantization: +3.5% quality, minimal memory cost
- 64 calibration samples is the sweet spot
- Outlier weights: handled by MSE-optimal scale, never clip
- Mistral retains quality better under int4 than Llama
