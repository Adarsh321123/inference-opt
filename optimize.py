"""
LLM Inference Optimization Strategy
====================================
THIS IS THE FILE THE AGENT MODIFIES. Everything is fair game.

The agent rewrites this file to try different optimization approaches.
The only requirement: optimize_model() must return a working (model, tokenizer) pair.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def optimize_model(model_name: str, device: str = "cuda"):
    """
    Load and optimize a model for efficient inference.

    Args:
        model_name: HuggingFace model identifier
        device: Target device

    Returns:
        (model, tokenizer) — the optimized model ready for inference
    """
    # =================================================================
    # NF4 with bfloat16 compute, no double quant, static KV cache
    # bfloat16 might be faster on 3090 Ampere tensor cores
    # No double quant = less dequantization overhead
    # Static KV cache = faster generation (pre-allocated)
    # =================================================================

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    # Set static KV cache for faster generation
    model.generation_config.cache_implementation = "static"

    return model, tokenizer
