# knowledge

Accumulated insights from 260+ experiments across 7 rounds. Read this before every session.

## Score History (Llama 3.1 8B)
- Round 3 HQQ library: 2.7 (quality 0.90, speed 1.3x, memory 2.23x)
- Round 5 from-scratch: 2.35 (quality 0.90, speed 1.0x, memory 2.61x)
- Round 6 fused kernel: 1.58 (quality 0.88, speed 0.83x, memory 2.14x)
- **Round 7 prompt_lookup=128: 2.79** (quality 0.90, speed 1.18x, memory 2.61x)

## Score History (Mistral 7B v0.3)
- Round 3 HQQ library: 3.9 (quality 0.97, speed ??, memory 3.33x)
- Round 5 from-scratch: 2.25 (quality 0.97, speed 0.69x, memory 3.33x)
- Round 7: 2.20 (quality 0.97, speed 0.68x, memory 3.33x)

## Round 7 Findings

### torch.compile Is Incompatible (PyTorch 2.4.1)
- mode="reduce-overhead" + Triton dequant → >15min compile timeout
- mode="default" + Triton dequant → >15min compile timeout
- Pure PyTorch dequant (replacing Triton) → 3x slower in eager fallback
- Custom torch.library.custom_op → adds wrapper overhead, worse speed
- **Conclusion: torch.compile is a dead end on PyTorch 2.4.1 with Triton kernels**

### prompt_lookup_num_tokens Is the Key Speed Lever
- **Llama: 128 is optimal** (44.8 tok/s, 1.18x speedup). 64→1.0x, 256→same as 128
- Mistral: 128 or 256 gives same result (0.68x — dequant overhead dominates)
- Higher prompt_lookup enables more speculative tokens per generation step

### Round 5 Dequant+cuBLAS Is Still the Best Approach
- Best code: commit 839158a (round 5) — dequant Triton kernel + F.linear
- The round 6 fused matvec kernel never beat round 5's score
- Round 6 was committed on top but is strictly worse
- Key: Hessian-weighted MSE-optimal scales with calibration data

### GPU-Accelerated Quantization
- Moving weight+h_diag to GPU for compute_scales: 10x faster (30s vs 15min)
- Same quality as CPU quantization
- BUT: inflates peak VRAM slightly (GPU temp allocations after reset_peak_memory_stats)
- Best to use CPU quantization for final VRAM score

### Failed Approaches (Don't Try Again)
- Quantizing lm_head: quality drops 1%, not enough memory savings
- Int8 embeddings: 20% speed drop from dequant overhead
- GROUP_SIZE=64: quality up (0.92) but 42% speed drop
- BLOCK_M=64 BLOCK_K=256: slower than default 128/128
- device_map="auto": accelerate dispatch hooks kill speed AND memory

## Transferable Findings (All Rounds)
- **prompt_lookup: 128 for all models** (critical for speedup!)
- Hessian-weighted MSE-optimal scale: best quality technique
- Skip lm_head only: best quality/memory tradeoff
- 64 calibration samples is the sweet spot
- BLOCK_M=128, BLOCK_K=128, num_warps=4, num_stages=3 for dequant on 3090
- reset_peak_memory_stats() after calibration: essential for VRAM score
- CPU quantization for best VRAM measurement
- TF32 enabled has no measurable effect (matmuls are already bf16)
