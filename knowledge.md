# knowledge

Accumulated insights from 240+ experiments across 6 rounds. Read this before every session.

## Score History (Llama 3.1 8B)
- Round 3 HQQ library: **2.7** (quality 0.90, speed 1.3x, memory 2.23x)
- Round 5 from-scratch: 2.35 (quality 0.90, speed 1.0x, memory 2.61x)
- Round 6 fused kernel: 1.58 (quality 0.88, speed 0.83x, memory 2.14x)

Round 3's library approach STILL has the best score. From-scratch keeps losing on speed.

## Round 6 Findings

### The Kernel IS Fast — Python Overhead Is the Bottleneck
- Fused int4 matvec: 47μs per 4096x4096 layer (1.39x faster than cuBLAS bf16!)
- On large layers (14336x4096): 2.36x faster than cuBLAS
- But 224 Triton kernel launches per token × ~20μs Python dispatch = 4.5ms overhead
- End-to-end: 31.6 tok/s (0.83x baseline) despite kernel being 1.39x faster per-op

### The Fix: Eliminate Python Launch Overhead
Three approaches to try (in order of likely impact):

1. **torch.compile(model)**: Should trace the forward pass, see the Triton kernels,
   and eliminate Python dispatch. This is the #1 thing to try. Use mode="reduce-overhead"
   for CUDA graph capture, or mode="default" for general optimization.

2. **Register as custom torch op**: Use torch.library.custom_op to register the
   Triton kernel as a PyTorch operation. torch.compile can then fuse multiple calls.

3. **CUDA graphs**: Capture the 224 kernel launches in a graph and replay them.
   Requires static tensor shapes (batch=1 for generation is static).

### Round 6 Quality Note
Skipping first/last 2 transformer layers hurt quality (0.88 vs 0.90 from round 5).
Just skipping lm_head (round 5 approach) is better: 0.90 quality with less memory cost.

## Two Paths Forward

### Path A: Fix the from-scratch kernel speed (continue round 6)
The fused kernel beats cuBLAS per-operation. If Python overhead is eliminated
(torch.compile/CUDA graphs), the from-scratch approach could score:
- Llama: 0.90 * 1.39 * 2.61 = 3.26 (beats HQQ's 2.7 by 21%!)
- Mistral: 0.97 * 1.39 * 3.33 = 4.49 (beats HQQ's 3.9 by 15%!)

### Path B: Inject custom quality into HQQ's fast path
Use torchao HQQ for speed (tinygemm, 1.3x). Find a way to improve quality
beyond HQQ's 0.90 (Llama). If quality reaches 0.95:
- Llama: 0.95 * 1.3 * 2.23 = 2.76 (marginal improvement)
- Mistral: already at 0.97

Path A has bigger potential (3.26 vs 2.76) but requires solving the launch overhead.
Path B is safer but ceiling is lower.

## Transferable Findings (All Rounds)
- prompt_lookup: 64 for Llama, 256 for Mistral (essential for int4)
- Hessian-weighted MSE-optimal scale: best quality technique (+1.5%)
- Skip lm_head only (not first/last layers): best quality/memory tradeoff
- 64 calibration samples is the sweet spot
- num_warps=4 for memory-bound kernels on 3090
- TILE_M=8, HALF_GS=64, tiled weight layout for coalesced access
- Fused kernel: 47μs per layer (1.39x cuBLAS) — the kernel works, the dispatch doesn't
