"""
LLM Inference Optimization — HQQ Int4 Pipeline
================================================
THIS IS THE FILE THE AGENT MODIFIES. Everything is fair game.

Round 4 conclusion from 35+ experiments:
- HQQ int4 gs=128 + prompt_lookup=256 is the optimal pipeline
- Weight transforms (AWQ, clipping, bias correction) hurt HQQ quality
- Post-quant fine-tuning blocked (no backward through int4pack)
- Int4 is SLOWER than FP16 at batch=1; prompt_lookup is essential
- The simplest pipeline consistently gives the best results
"""

import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.backends.cudnn.benchmark = True

GROUP_SIZE = 128


def optimize_model(model_name: str, device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load bf16 on CPU — avoid GPU memory spike
    print("Loading model bf16 on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # torchao Int4WeightOnly with HQQ quantization
    print("Quantizing...")
    from torchao.quantization import quantize_, Int4WeightOnlyConfig
    config = Int4WeightOnlyConfig(group_size=GROUP_SIZE, use_hqq=True, version=1)

    # Move non-quantizable layers to GPU
    model.model.embed_tokens.to(device)
    model.model.norm.to(device)
    if hasattr(model.model, 'rotary_emb'):
        model.model.rotary_emb.to(device)
    model.lm_head.to(device)

    # Stream each transformer layer: CPU → GPU, quantize on GPU
    for layer in model.model.layers:
        layer.to(device)
        quantize_(layer, config)
    gc.collect()
    torch.cuda.empty_cache()

    # Convert LayerNorm weights to fp32 for higher precision
    for name, module in model.named_modules():
        if 'norm' in name.lower() and hasattr(module, 'weight'):
            module.weight.data = module.weight.data.float()

    # Prompt lookup: speculative n-gram decoding
    model.generation_config.prompt_lookup_num_tokens = 256

    print("Done.")
    return model, tokenizer
