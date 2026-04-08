# knowledge

Accumulated insights from 125+ experiments across 3 rounds. Read this before every session.

## Rounds 1-3 Summary

Best scores achieved with library-level optimization (API calls only):
- Llama 3.1 8B: score 2.7 (torchao Int4 HQQ + prompt_lookup=64)
- Mistral 7B: score 3.9 (torchao Int4 HQQ + prompt_lookup=256, no cuBLASLt)

These represent the ceiling of configuring existing libraries. Round 4 shifts to from-scratch quantization math to break through this ceiling.

## Key Findings (transferable)

- torchao Int4WeightOnly with use_hqq=True is the best quantization backend (faster than bnb NF4)
- prompt_lookup_num_tokens is the single biggest inference speedup (amortizes dequant overhead)
- cuBLASLt is model-dependent: helps Llama, hurts Mistral by 15%
- Streaming CPU→GPU quantization keeps peak VRAM low
- bfloat16 compute dtype is required for torchao tinygemm kernels
- Mistral retains quality better under 4-bit than Llama (0.97 vs 0.90)
- Quality retention at 4-bit varies 85-97% across methods — the MATH of quantization matters

## Round 4: From-Scratch Quantization

optimize.py now contains a full quantization pipeline:
1. Load model bf16
2. Collect calibration data and activation statistics
3. Apply custom weight transformations (YOUR CREATIVE SPACE)
4. Quantize with torchao int4 kernels

The `transform_weights_for_quantization()` function is where breakthroughs happen.
The goal: discover a transformation that produces better quality at 4-bit than any existing method.

Known techniques to try and beat:
- AWQ: scales important channels by activation magnitude (alpha=0.5 weighting)
- GPTQ: uses Hessian to find optimal rounding direction per weight
- QuIP#: Hadamard rotation makes weight distribution more uniform
- SqueezeLLM: separates outlier weights into sparse matrix
