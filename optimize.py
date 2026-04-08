"""
LLM Inference Optimization Strategy
====================================
THIS IS THE FILE THE AGENT MODIFIES. Everything is fair game.

The agent rewrites this file to try different optimization approaches.
The only requirement: optimize_model() must return a working (model, tokenizer) pair.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig
from torchao.quantization import Int4WeightOnlyConfig

# Force cuBLASLt for small batch GEMM
torch.backends.cuda.preferred_blas_library("cublaslt")


def optimize_model(model_name: str, device: str = "cuda"):
    """
    Load and optimize a model for efficient inference.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # torchao Int4WeightOnly quantization — PyTorch-native, composable with torch.compile
    quantization_config = TorchAoConfig(Int4WeightOnlyConfig(group_size=128))

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Prompt lookup decoding: biggest single win from prior rounds
    model.generation_config.prompt_lookup_num_tokens = 40

    return model, tokenizer
