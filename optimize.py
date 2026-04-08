"""
LLM Inference Optimization Strategy
====================================
THIS IS THE FILE THE AGENT MODIFIES. Everything is fair game.

The agent rewrites this file to try different optimization approaches.
The only requirement: optimize_model() must return a working (model, tokenizer) pair.
"""

import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# cuBLASLt: helps Llama, hurts Mistral with torchao int4 kernels
# Set dynamically in optimize_model based on model


def optimize_model(model_name: str, device: str = "cuda"):
    """
    Load and optimize a model for efficient inference.
    """
    # cuBLASLt helps Llama but hurts Mistral
    if "mistral" not in model_name.lower():
        torch.backends.cuda.preferred_blas_library("cublaslt")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load model in bf16 on CPU — avoid GPU memory spike
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # torchao Int4WeightOnly with HQQ quantization
    from torchao.quantization import quantize_, Int4WeightOnlyConfig
    config = Int4WeightOnlyConfig(group_size=128, use_hqq=True, version=1)

    # Move non-quantizable layers to GPU
    model.model.embed_tokens.to("cuda:0")
    model.model.norm.to("cuda:0")
    model.model.rotary_emb.to("cuda:0")
    model.lm_head.to("cuda:0")

    # Stream each transformer layer: CPU → GPU, quantize on GPU, free bf16
    for layer in model.model.layers:
        layer.to("cuda:0")
        quantize_(layer, config)
        gc.collect()
        torch.cuda.empty_cache()

    # Model-adaptive prompt lookup: Mistral benefits from higher values
    if "mistral" in model_name.lower():
        model.generation_config.prompt_lookup_num_tokens = 128
    else:
        model.generation_config.prompt_lookup_num_tokens = 40

    return model, tokenizer
