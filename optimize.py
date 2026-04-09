"""
LLM Inference Optimization — Custom Quantization Pipeline
==========================================================
THIS IS THE FILE THE AGENT MODIFIES. Everything is fair game.

Round 5 approach: monkey-patch HQQ's internal rounding with custom math.

WHY: torchao's HQQ code path produces fast tinygemm inference (1.3x speedup).
RTN does NOT (1.01x). So we MUST use the HQQ code path. But we want custom
quantization math. Solution: find where HQQ does rounding inside torchao's
source and replace that function with our own.

STEPS FOR THE AGENT:
1. Find torchao's installed source. Start with:
   python -c "import torchao; print(torchao.__file__)"
   Then explore the quantization directory, especially:
   - How Int4WeightOnlyConfig(use_hqq=True) triggers HQQ
   - Where HQQ computes scales and rounds weights
   - What function signature to match for the monkey-patch

2. Write a custom rounding/scale function that matches HQQ's interface
   but uses better math (GPTQ-style Hessian rounding, MSE-optimal scale, etc.)

3. Monkey-patch it BEFORE calling quantize_():
   import torchao.some.internal.module as m
   m.hqq_rounding_function = my_custom_function

4. Call quantize_(layer, config) as normal — it uses HQQ's code path
   (fast kernels) but with our custom rounding decisions.

The calibration infrastructure below collects Hessians (H = X^T X) that
the custom rounding function can use for GPTQ-style optimal rounding.
"""

import gc
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# QUANTIZATION HYPERPARAMETERS
# ============================================================
BITS = 4
GROUP_SIZE = 128
CALIBRATION_SAMPLES = 128
CALIBRATION_SEQ_LEN = 512


# ============================================================
# CALIBRATION
# ============================================================

def load_calibration_data(tokenizer, n_samples=CALIBRATION_SAMPLES, seq_len=CALIBRATION_SEQ_LEN):
    """Load calibration samples from WikiText-2."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in ds["text"] if len(t.strip()) > 200][:n_samples]
    encodings = []
    for text in texts:
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
        if tokens.input_ids.shape[1] >= 64:
            encodings.append(tokens.input_ids)
    return encodings


def collect_activation_stats(model, calib_data, device="cuda:0"):
    """
    Run calibration data and collect per-linear-layer Hessians (H = X^T X)
    and activation channel statistics. The Hessian is used for GPTQ-style
    optimal rounding: the error from rounding weight w_i is weighted by H_ii.
    """
    stats = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, inp, out):
            x = inp[0].detach().float()
            flat = x.reshape(-1, x.shape[-1])
            if name not in stats:
                n = flat.shape[-1]
                stats[name] = {
                    "absmax": torch.zeros(n, device=flat.device),
                    "H": torch.zeros(n, n, device=flat.device),
                    "count": 0,
                }
            s = stats[name]
            s["absmax"] = torch.max(s["absmax"], flat.abs().max(dim=0).values)
            s["H"] += flat.T @ flat
            s["count"] += flat.shape[0]
        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for input_ids in calib_data[:32]:
            try:
                model(input_ids.to(device))
            except Exception:
                continue

    for h in hooks:
        h.remove()

    for name in stats:
        s = stats[name]
        s["H"] = s["H"] / max(s["count"], 1)

    return stats


# ============================================================
# MONKEY-PATCH HQQ — THE AGENT'S CREATIVE SPACE
# ============================================================

# Global dict: the agent stores Hessians here so the monkey-patched
# function can access them during quantize_().
_layer_hessians = {}


def patch_hqq_quantization():
    """
    Find and replace HQQ's internal rounding function with custom math.

    THE AGENT MUST IMPLEMENT THIS by:
    1. Reading torchao's source to find HQQ's rounding logic
    2. Writing a replacement that uses _layer_hessians for better rounding
    3. Monkey-patching the function before quantize_() is called

    EXAMPLE (agent should verify the actual function path):

    import torchao.prototype.hqq as hqq_module

    original_quantize = hqq_module.some_quantize_function

    def custom_quantize(weight, ...):
        # Use Hessian for GPTQ-style optimal rounding:
        # For each weight w_i, compare:
        #   error_up   = (ceil(w_i/s)*s - w_i)^2 * H_ii
        #   error_down = (floor(w_i/s)*s - w_i)^2 * H_ii
        # Pick the direction with smaller error.
        ...

    hqq_module.some_quantize_function = custom_quantize
    """
    # === BASELINE: No patch (plain HQQ) ===
    # The agent replaces this with the actual monkey-patch.
    pass


# ============================================================
# MAIN PIPELINE
# ============================================================

def optimize_model(model_name: str, device: str = "cuda"):
    global _layer_hessians

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print("Loading model bf16 on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # --- Calibration ---
    print("Collecting activation statistics + Hessians...")
    calib_data = load_calibration_data(tokenizer)
    model.to(device)
    act_stats = collect_activation_stats(model, calib_data, device)
    model.to("cpu")
    torch.cuda.empty_cache()

    # Store Hessians globally so monkey-patched function can access them
    _layer_hessians = act_stats

    # --- Monkey-patch HQQ (agent's creative space) ---
    patch_hqq_quantization()

    # --- Quantize with HQQ code path (fast tinygemm kernels) ---
    print("Quantizing...")
    from torchao.quantization import quantize_, Int4WeightOnlyConfig
    config = Int4WeightOnlyConfig(group_size=GROUP_SIZE, use_hqq=True, version=1)

    model.model.embed_tokens.to(device)
    model.model.norm.to(device)
    if hasattr(model.model, 'rotary_emb'):
        model.model.rotary_emb.to(device)
    model.lm_head.to(device)

    for i, layer in enumerate(model.model.layers):
        layer.to(device)
        quantize_(layer, config)
        gc.collect()
        torch.cuda.empty_cache()

    # --- Inference config ---
    is_llama = "llama" in model_name.lower()
    model.generation_config.prompt_lookup_num_tokens = 64 if is_llama else 256

    print("Done.")
    return model, tokenizer
