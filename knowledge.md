# knowledge

Accumulated insights from 190+ experiments across 5 rounds. Read this before every session.

## Rounds 1-4 Summary

Rounds 1-3 optimized library configurations: torchao HQQ int4 + prompt_lookup.
Best scores: Llama 2.7, Mistral 3.9.

Round 4 tried weight transforms before HQQ. ALL failed — HQQ already optimizes
for the weight distribution. Transforms interfere with it.

Key lesson: calling library functions limits you to library quality.
To beat HQQ (quality 0.90 Llama, 0.96 Mistral), you must write the algorithm yourself.

## Round 5: From-Scratch Quantization (37+ experiments)

### Best Scores (commit 839158a)
- Llama 3.1 8B: **2.35** (quality 0.90, speedup ~1.0x, memory 2.61x)
- Mistral 7B v0.3: **2.26** (quality 0.97, speedup 0.70x, memory 3.33x)

### Architecture
- Triton dequant kernel: packed int4 → bf16 (BLOCK_M=128, BLOCK_K=128, num_warps=4, num_stages=3)
- cuBLAS matmul via F.linear on the dequantized bf16 tensor
- Calibration: 64 samples of WikiText-2, Hessian diagonal per channel
- Skip lm_head quantization (keep in bf16)
- prompt_lookup: 64 for Llama, 256 for Mistral

### What Worked
1. **Hessian-weighted MSE-optimal scale search**: Two-stage grid search (5 coarse + 6 fine = 11 points)
   over scale multipliers 0.60-1.00. Minimizes Hessian-weighted reconstruction error per group.
   Best single quality improvement (+1.5% quality_retained).
2. **Skip lm_head quantization**: +3.5% quality retained. lm_head is quality-critical.
   Memory cost: ~170MB for Llama, ~17MB for Mistral.
3. **num_warps=4**: CRITICAL. 32% faster than num_warps=8. Lower warp count is better
   for memory-bound dequant kernel.
4. **BLOCK_M=128, BLOCK_K=128**: Sweet spot for register pressure vs launch overhead.
5. **Peak memory reset**: torch.cuda.reset_peak_memory_stats() after calibration.
6. **64 calibration samples**: Sweet spot. 32 is too noisy, 128 is no better.

### What Failed (all tested, all worse than final)
1. AWQ-style per-channel scaling (alpha=0.5): destroyed quality
2. Error diffusion rounding: hurt quality (columns aren't spatially correlated)
3. GROUP_SIZE=64: quality +3% but speed -28%. Net score decrease
4. Fused dequant+matvec Triton kernel: slower than dequant+cuBLAS
5. BLOCK_K=256 or BLOCK_M=256: register pressure kills speed
6. Asymmetric quantization: hurt quality, slowed inference, slow quantization
7. num_warps=8: 32% slower than num_warps=4
8. First transformer layer skip: quality +0.85% but memory -5%
9. 3-stage MSE refinement: marginal quality gain, speed too variable
10. 128 calibration samples: no improvement over 64, slower calibration
11. num_stages=2,3,4: all identical speed (37.8 tok/s)
12. PyTorch dequant (no Triton): 7x slower than Triton dequant
13. Original Triton matvec kernel: 20x slower than dequant+cuBLAS

### Speed Notes
- num_warps=4 is the most important speed setting
- Speed varies ±15% between runs (GPU thermal state, baseline variance)
- With num_warps=4 and prompt_lookup, Llama reaches ~1.0x baseline speed
- Mistral is slower (0.7x) because prompt_lookup=256 has lower speculation hit rate
- Without prompt_lookup, raw decode speed is ~0.37x baseline for both models

### Quality Notes
- Symmetric int4 at GROUP_SIZE=128 can reach quality 0.90 (Llama) or 0.97 (Mistral)
- This matches HQQ library quality (Llama 0.90, Mistral 0.96)
- Hessian-weighted MSE scale search is the key quality technique
- Per-weight rounding is already optimal (nearest); only cross-weight methods could improve
- lm_head is by far the most quality-sensitive layer to skip

### Key Findings from All Rounds
- prompt_lookup essential: 64 for Llama, 256 for Mistral
- Outlier weights: handled by MSE-optimal scale clipping
- Quality at 4-bit: 85-97%, algorithm matters
- Mistral retains quality better under 4-bit than Llama
- Phi-3-small requires pytest (not in deps, can't run)
- cuBLAS matmul cannot be beaten by Triton for matmul quality
