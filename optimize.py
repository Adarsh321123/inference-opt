"""
LLM Inference Optimization Strategy
====================================
THIS IS THE FILE THE AGENT MODIFIES. Everything is fair game.

The agent rewrites this file to try different optimization approaches.
The only requirement: optimize_model() must return a working (model, tokenizer) pair.
"""

import os
import torch
from transformers import AutoTokenizer

# Force cuBLASLt for small batch GEMM — can be faster for batch-1 generation
torch.backends.cuda.preferred_blas_library("cublaslt")

AWQ_CACHE_DIR = "/tmp/awq_cache"


def optimize_model(model_name: str, device: str = "cuda"):
    """
    Load and optimize a model for efficient inference.
    Uses AWQ 4-bit quantization for fast fused-kernel inference.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Build a cache path based on the model name
    safe_name = model_name.replace("/", "--")
    cache_path = os.path.join(AWQ_CACHE_DIR, safe_name + "-awq")

    if os.path.exists(cache_path):
        # Load pre-quantized AWQ model from cache
        from awq import AutoAWQForCausalLM
        model = AutoAWQForCausalLM.from_quantized(
            cache_path,
            fuse_layers=True,
            trust_remote_code=True,
        )
    else:
        # Quantize on-the-fly and cache
        from awq import AutoAWQForCausalLM
        model = AutoAWQForCausalLM.from_pretrained(
            model_name,
            safetensors=True,
            trust_remote_code=True,
        )
        quant_config = {
            "w_bit": 4,
            "q_group_size": 128,
            "zero_point": True,
            "version": "GEMM",
        }
        model.quantize(tokenizer, quant_config=quant_config)

        # Save for future runs
        os.makedirs(cache_path, exist_ok=True)
        model.save_quantized(cache_path)

        # Reload with fused layers for speed
        del model
        torch.cuda.empty_cache()
        model = AutoAWQForCausalLM.from_quantized(
            cache_path,
            fuse_layers=True,
            trust_remote_code=True,
        )

    return model, tokenizer
