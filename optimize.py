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
        model_name: HuggingFace model identifier (e.g., "meta-llama/Meta-Llama-3-8B")
        device: Target device

    Returns:
        (model, tokenizer) — the optimized model ready for inference
    """
    # =================================================================
    # BASELINE: Simple 4-bit bitsandbytes NF4 quantization
    # The agent should iterate on this to find better strategies.
    # =================================================================

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )

    return model, tokenizer
