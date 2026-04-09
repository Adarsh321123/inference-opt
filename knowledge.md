# knowledge

Accumulated insights from 180+ experiments across 5 rounds. Read this before every session.

## Rounds 1-4 Summary

Rounds 1-3 optimized library configurations: torchao HQQ int4 + prompt_lookup.
Best scores: Llama 2.7, Mistral 3.9.

Round 4 tried weight transforms before HQQ. ALL failed — HQQ already optimizes
for the weight distribution. Transforms interfere with it.

Key lesson: calling library functions limits you to library quality.
To beat HQQ (quality 0.90 Llama, 0.96 Mistral), you must write the algorithm yourself.

## Round 5: From-Scratch Quantization Results

### Best Scores
- Llama: 1.8941 (quality 0.90, speedup 0.81, memory 2.61)
- Mistral: 1.8207 (quality 0.97, speedup 0.56, memory 3.33)

### Architecture
- Triton dequant kernel: packed int4 → bf16 (BLOCK_M=128, BLOCK_K=128)
- cuBLAS matmul via F.linear on the dequantized bf16 tensor
- Calibration: 64 samples of WikiText-2, Hessian diagonal per channel

### What Worked
1. **Hessian-weighted MSE-optimal scale search**: Grid search over scale multipliers
   (0.60-1.00), picking the one that minimizes Hessian-weighted reconstruction error.
2. **Two-stage refinement**: Coarse search (5 points) then fine search (±0.04).
3. **Skip lm_head quantization**: +3.5% quality retained. Small memory cost.
4. **BLOCK_M=128 for Triton dequant**: Sweet spot. 64 was slower, 256 much slower.
5. **Peak memory reset after calibration**: Ensures calibration doesn't inflate measurement.
6. **64 calibration samples**: Better Hessian estimates than 32.

### What Failed
1. **AWQ-style per-channel scaling**: Destroyed quality at alpha=0.5.
2. **Error diffusion rounding**: Hurt quality (columns aren't spatially correlated).
3. **GROUP_SIZE=64**: Quality +3% but speed -28%. Net score decrease.
4. **Fused dequant+matvec**: Slower than dequant+cuBLAS.
5. **BLOCK_K=256 or BLOCK_M=256**: Register pressure kills speed.
6. **Triton autotune**: No significant speed improvement over hand-tuned.
7. **Asymmetric quantization**: Hurt quality, slowed inference, 27min quant time.
8. **num_warps=8**: 32% slower than num_warps=4 for memory-bound dequant kernel.

### Speed Notes
- num_warps=4 is CRITICAL (32% faster than num_warps=8)
- Speed varies 29-38 tok/s between runs (GPU thermal state, Triton cache)
- With num_warps=4 and warm cache, speed reaches ~1.0x baseline via prompt_lookup

### Key Findings from All Rounds
- prompt_lookup essential: 64 for Llama, 256 for Mistral
- lm_head is most quality-sensitive layer
- Quality at 4-bit: 85-97%, algorithm matters
- Mistral retains quality better under 4-bit than Llama
- Phi-3-small requires pytest (not in deps, can't run)
