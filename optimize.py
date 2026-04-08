"""
LLM Inference Optimization Strategy
====================================
THIS IS THE FILE THE AGENT MODIFIES. Everything is fair game.

The agent rewrites this file to try different optimization approaches.
The only requirement: optimize_model() must return a working (model, tokenizer) pair.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Force cuBLASLt for small batch GEMM — can be faster for batch-1 generation
torch.backends.cuda.preferred_blas_library("cublaslt")


def optimize_model(model_name: str, device: str = "cuda"):
    """
    Load and optimize a model for efficient inference.
    NF4 quantization + eager attention + torch.compile + prompt_lookup.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_storage=torch.uint8,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )

    # Compile for optimized inference
    model = torch.compile(model, mode="default")

    # Prompt lookup decoding: use n-grams from prompt as draft tokens
    model.generation_config.prompt_lookup_num_tokens = 40

    return model, tokenizer
