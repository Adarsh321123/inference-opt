"""
Evaluation Harness for LLM Inference Optimization
DO NOT MODIFY THIS FILE — it is the ground truth metric.

Usage:
    python evaluate.py --baseline --model meta-llama/Meta-Llama-3.1-8B   # run FP16 baseline
    python evaluate.py --model meta-llama/Meta-Llama-3.1-8B              # evaluate optimized model
"""

import argparse
import json
import time
import math
import torch
import numpy as np
from pathlib import Path

# ============================================================
# Fixed evaluation parameters — do not change
# ============================================================
EVAL_SAMPLES = 80        # number of text samples for perplexity
MAX_SEQ_LEN = 512        # sequence length for perplexity evaluation
GEN_TOKENS = 128         # tokens to generate per prompt for speed test
GEN_PROMPTS = 8          # number of prompts for speed test
WARMUP_RUNS = 3          # warmup before speed measurement
QUALITY_FLOOR = 0.85     # minimum quality_retained before score = 0


def load_eval_data():
    """Load a fixed evaluation dataset (WikiText-2 test set)."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    # Filter to non-empty, reasonably long texts
    texts = [t for t in ds["text"] if len(t.strip()) > 200][:EVAL_SAMPLES]
    if len(texts) < 20:
        raise RuntimeError(f"Only found {len(texts)} eval texts — need at least 20")
    return texts


def measure_perplexity(model, tokenizer, texts):
    """Compute perplexity on the evaluation texts."""
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            encodings = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN
            )
            input_ids = encodings.input_ids.to(model.device)

            if input_ids.shape[1] < 2:
                continue

            outputs = model(input_ids, labels=input_ids)
            # loss is mean over tokens; multiply back by count
            n_tokens = input_ids.shape[1] - 1
            total_nll += outputs.loss.float().item() * n_tokens
            total_tokens += n_tokens

    if total_tokens == 0:
        return float("inf")

    avg_nll = total_nll / total_tokens
    perplexity = math.exp(avg_nll)
    return perplexity


def measure_speed(model, tokenizer):
    """Measure generation throughput in tokens/sec."""
    prompts = [
        "The meaning of life is",
        "In a groundbreaking study, researchers found that",
        "The future of artificial intelligence depends on",
        "Once upon a time in a land far away there lived",
        "The key to understanding quantum mechanics is",
        "Recent advances in machine learning have shown",
        "The history of mathematics demonstrates that",
        "According to the latest research in physics,",
    ][:GEN_PROMPTS]

    # Warmup — don't measure these
    for _ in range(WARMUP_RUNS):
        inputs = tokenizer(prompts[0], return_tensors="pt").to(model.device)
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=16, do_sample=False)

    torch.cuda.synchronize()

    # Timed runs
    total_new_tokens = 0
    start = time.perf_counter()
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=GEN_TOKENS, do_sample=False
            )
        total_new_tokens += outputs.shape[1] - inputs.input_ids.shape[1]

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    if elapsed == 0:
        return 0.0
    return total_new_tokens / elapsed


def measure_peak_vram_gb():
    """Get peak VRAM usage in GB."""
    return torch.cuda.max_memory_allocated() / (1024**3)


def run_baseline(model_name: str):
    """Run FP16 baseline and save results to baseline.json."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"=== FP16 BASELINE: {model_name} ===")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model (FP16)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    print("Loading eval data...")
    texts = load_eval_data()

    print(f"Measuring perplexity on {len(texts)} samples...")
    perplexity = measure_perplexity(model, tokenizer, texts)

    print("Measuring generation speed...")
    tps = measure_speed(model, tokenizer)

    vram = measure_peak_vram_gb()

    baseline = {
        "model_name": model_name,
        "perplexity": round(perplexity, 4),
        "tokens_per_sec": round(tps, 2),
        "peak_vram_gb": round(vram, 2),
    }

    with open("baseline.json", "w") as f:
        json.dump(baseline, f, indent=2)

    print(f"\n--- BASELINE ---")
    print(f"perplexity:     {perplexity:.4f}")
    print(f"tokens_per_sec: {tps:.1f}")
    print(f"peak_vram_gb:   {vram:.2f}")
    print(f"\nBaseline saved to baseline.json")


def run_evaluation(model_name: str):
    """Evaluate the optimized model and print the efficiency score."""
    # Load baseline
    if not Path("baseline.json").exists():
        print("ERROR: baseline.json not found. Run with --baseline first.")
        exit(1)

    with open("baseline.json") as f:
        baseline = json.load(f)

    # Verify model matches
    if baseline["model_name"] != model_name:
        print(f"WARNING: baseline was for {baseline['model_name']}, evaluating {model_name}")

    # Reset GPU state
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Import and run the optimization
    print(f"=== EVALUATING OPTIMIZED: {model_name} ===")
    print("Running optimize_model()...")

    from optimize import optimize_model

    opt_start = time.perf_counter()
    model, tokenizer = optimize_model(model_name)
    opt_elapsed = time.perf_counter() - opt_start
    print(f"Optimization took {opt_elapsed:.1f}s")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Evaluate quality
    print("Loading eval data...")
    texts = load_eval_data()

    print(f"Measuring perplexity on {len(texts)} samples...")
    perplexity = measure_perplexity(model, tokenizer, texts)

    print("Measuring generation speed...")
    tps = measure_speed(model, tokenizer)

    vram = measure_peak_vram_gb()

    # Compute fitness score
    quality_retained = min(1.0, baseline["perplexity"] / perplexity)
    speedup = tps / baseline["tokens_per_sec"] if baseline["tokens_per_sec"] > 0 else 0.0
    memory_reduction = baseline["peak_vram_gb"] / vram if vram > 0 else 0.0

    if quality_retained < QUALITY_FLOOR:
        efficiency_score = 0.0
    else:
        efficiency_score = quality_retained * speedup * memory_reduction

    # Print results in grep-able format
    print(f"\n---")
    print(f"efficiency_score: {efficiency_score:.4f}")
    print(f"quality_retained: {quality_retained:.4f}")
    print(f"speedup:          {speedup:.4f}")
    print(f"memory_reduction: {memory_reduction:.4f}")
    print(f"perplexity:       {perplexity:.4f}")
    print(f"baseline_ppl:     {baseline['perplexity']:.4f}")
    print(f"tokens_per_sec:   {tps:.1f}")
    print(f"baseline_tps:     {baseline['tokens_per_sec']:.1f}")
    print(f"peak_vram_gb:     {vram:.2f}")
    print(f"baseline_vram_gb: {baseline['peak_vram_gb']:.2f}")
    print(f"opt_seconds:      {opt_elapsed:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Inference Optimization Evaluator")
    parser.add_argument("--baseline", action="store_true", help="Run FP16 baseline evaluation")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B",
        help="HuggingFace model name",
    )
    args = parser.parse_args()

    if args.baseline:
        run_baseline(args.model)
    else:
        run_evaluation(args.model)
