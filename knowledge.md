# knowledge

Accumulated insights from 170+ experiments across 4 rounds. Read this before every session.

## Rounds 1-4 Summary

Rounds 1-3 optimized library configurations: torchao HQQ int4 + prompt_lookup.
Best scores: Llama 2.7, Mistral 3.9.

Round 4 tried weight transforms before HQQ. ALL failed — HQQ already optimizes
for the weight distribution. Transforms interfere with it.

Key lesson: calling library functions limits you to library quality.
To beat HQQ (quality 0.90 Llama, 0.96 Mistral), you must write the algorithm yourself.

## Round 5: From-Scratch Quantization

optimize.py now implements int4 quantization FROM SCRATCH:
- compute_scales(): per-group scale computation (agent modifies)
- quantize_weight(): rounding logic (agent modifies)
- Triton kernel: int4 dequant + matvec (agent modifies)
- QuantizedLinear: drop-in nn.Linear replacement
- Calibration: activation absmax + Hessian diagonal per channel

The baseline is naive symmetric RTN. Quality bars to beat:
- HQQ: Llama 0.9021, Mistral 0.9590
- AWQ/GPTQ (published): ~0.97 for both

## Key Findings from All Rounds

- prompt_lookup_num_tokens is essential: int4 is SLOWER than FP16 at batch=1 without it
- prompt_lookup sweet spot: 64 for Llama, 256 for Mistral
- Outlier weights are sacred — never clip them
- Quality at 4-bit varies 85-97% across methods — the MATH matters
- Mistral retains quality better under 4-bit than Llama
- Calibration data from WikiText-2 works well for general models

## Techniques to Implement and Beat

- GPTQ: use H_diag (Hessian diagonal) to pick optimal rounding direction.
  For each weight: compare round-up vs round-down error weighted by H_diag.
  This is ~10 lines of code change in quantize_weight().
- Asymmetric quantization: use actual min/max instead of symmetric absmax.
  Captures skewed distributions better.
- Percentile clipping: compute scale from 99.9th percentile instead of max.
  Reduces scale, improves precision for bulk of weights.
- MSE-optimal scale: binary search for scale that minimizes reconstruction MSE.
- Error feedback: propagate rounding error to subsequent columns (like GPTQ).
- Activation-aware scaling: weight scale by channel activation magnitude (AWQ-style).
  But implement it IN the scale computation, not as a pre-transform.
