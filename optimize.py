"""
LLM Inference Optimization — HQQ Int4 + LayerNorm Fine-tuning
===============================================================
THIS IS THE FILE THE AGENT MODIFIES. Everything is fair game.

After HQQ quantization, fine-tune LayerNorm weights (the only remaining
float parameters) to compensate for quantization error. Uses calibration
data for layer-wise output matching.
"""

import gc
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.backends.cudnn.benchmark = True

GROUP_SIZE = 128
FINETUNE_STEPS = 50
FINETUNE_LR = 1e-4


def load_calibration_data(tokenizer, n_samples=8, seq_len=512):
    """Load calibration samples from WikiText-2."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in ds["text"] if len(t.strip()) > 200][:n_samples * 2]
    encodings = []
    for text in texts:
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
        if tokens.input_ids.shape[1] >= 64:
            encodings.append(tokens.input_ids)
        if len(encodings) >= n_samples:
            break
    return encodings


def finetune_layernorms(model, calib_data, device, n_steps=FINETUNE_STEPS, lr=FINETUNE_LR):
    """
    Fine-tune LayerNorm weights to minimize perplexity on calibration data.
    Only optimizes the RMSNorm/LayerNorm weights — all quantized weights frozen.
    """
    # Collect trainable parameters (only layernorm weights)
    trainable = []
    for name, param in model.named_parameters():
        if 'layernorm' in name.lower() or 'norm' in name.lower():
            param.requires_grad_(True)
            trainable.append(param)
        else:
            param.requires_grad_(False)

    if not trainable:
        print("  No trainable parameters found")
        return

    print(f"  Fine-tuning {len(trainable)} LayerNorm parameters...")
    optimizer = torch.optim.Adam(trainable, lr=lr)

    model.train()
    for step in range(n_steps):
        total_loss = 0.0
        for input_ids in calib_data[:4]:
            input_ids = input_ids.to(device)
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            total_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    for param in trainable:
        param.requires_grad_(False)

    print(f"  Final loss: {total_loss / len(calib_data[:4]):.4f}")


def optimize_model(model_name: str, device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print("Loading model bf16 on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # Quantize with torchao int4 HQQ
    print("Quantizing...")
    from torchao.quantization import quantize_, Int4WeightOnlyConfig
    config = Int4WeightOnlyConfig(group_size=GROUP_SIZE, use_hqq=True, version=1)

    model.model.embed_tokens.to(device)
    model.model.norm.to(device)
    if hasattr(model.model, 'rotary_emb'):
        model.model.rotary_emb.to(device)
    model.lm_head.to(device)

    for layer in model.model.layers:
        layer.to(device)
        quantize_(layer, config)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Fine-tune LayerNorm weights to compensate for quantization error
    print("Fine-tuning LayerNorm weights...")
    calib_data = load_calibration_data(tokenizer)
    finetune_layernorms(model, calib_data, device)
    gc.collect()
    torch.cuda.empty_cache()

    # Prompt lookup: speculative n-gram decoding
    model.generation_config.prompt_lookup_num_tokens = 256

    print("Done.")
    return model, tokenizer
