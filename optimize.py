"""
LLM Inference Optimization Strategy
====================================
THIS IS THE FILE THE AGENT MODIFIES. Everything is fair game.

The agent rewrites this file to try different optimization approaches.
The only requirement: optimize_model() must return a working (model, tokenizer) pair.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# NOTE: cuBLASLt helps Llama but hurts Mistral (~10% slower).
# Set conditionally in optimize_model() below.


def optimize_model(model_name: str, device: str = "cuda"):
    """
    Load and optimize a model for efficient inference.
    NF4 quantization + prompt_lookup. Model-specific attention and compile tuning.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_storage=torch.uint8,
    )

    # Model-specific tuning
    is_llama = "llama" in model_name.lower()
    attn_kwargs = {"attn_implementation": "eager"} if is_llama else {}

    # cuBLASLt helps Llama but hurts Mistral by ~10%
    if is_llama:
        torch.backends.cuda.preferred_blas_library("cublaslt")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        **attn_kwargs,
    )

    # torch.compile saves ~0.35 GB on Llama but not on Mistral
    if is_llama:
        model = torch.compile(model, mode="default")

    # Prompt lookup decoding: use n-grams from prompt as draft tokens
    # Mistral benefits from slightly larger window (50 vs 40) without cuBLASLt
    model.generation_config.prompt_lookup_num_tokens = 40 if is_llama else 50

    return model, tokenizer
