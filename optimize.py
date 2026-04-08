"""
LLM Inference Optimization Strategy
====================================
THIS IS THE FILE THE AGENT MODIFIES. Everything is fair game.

The agent rewrites this file to try different optimization approaches.
The only requirement: optimize_model() must return a working (model, tokenizer) pair.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# cuBLASLt for batch-1 (helps Llama, neutral for Mistral)
# torch.backends.cuda.preferred_blas_library("cublaslt")


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

    # Eager attention is faster for Llama batch-1, default for Mistral (sliding window)
    is_llama = "llama" in model_name.lower()
    attn_kwargs = {"attn_implementation": "eager"} if is_llama else {}

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
    model.generation_config.prompt_lookup_num_tokens = 40

    return model, tokenizer
