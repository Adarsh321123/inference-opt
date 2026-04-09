"""
LLM Inference Optimization — HQQ Int4 + Scale Optimization
=============================================================
THIS IS THE FILE THE AGENT MODIFIES. Everything is fair game.

After HQQ quantization, fine-tune per-group scale/zero parameters
to minimize calibration-weighted output error per linear layer.
"""

import gc
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

GROUP_SIZE = 128
SCALE_OPT_STEPS = 20
SCALE_OPT_LR = 1e-3


def load_calibration_data(tokenizer, n_samples=16, seq_len=512):
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


def optimize_scales(layer, layer_idx, calib_inputs, device, n_steps=SCALE_OPT_STEPS, lr=SCALE_OPT_LR):
    """
    Fine-tune quantization scale/zero parameters of each linear layer
    to minimize output reconstruction error on calibration data.
    """
    # Collect input activations for each linear using hooks
    linear_inputs = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, inp, out):
            x = inp[0].detach()
            if name not in linear_inputs:
                linear_inputs[name] = []
            linear_inputs[name].append(x)
        return hook_fn

    for name, module in layer.named_modules():
        if isinstance(module, nn.Linear) and hasattr(module.weight, 'tensor_impl'):
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Forward pass to collect activations
    layer.eval()
    with torch.no_grad():
        for x in calib_inputs[:4]:
            try:
                layer(x)
            except Exception:
                continue

    for h in hooks:
        h.remove()

    # For each linear, optimize scale/zero
    for name, module in layer.named_modules():
        if not isinstance(module, nn.Linear) or not hasattr(module.weight, 'tensor_impl'):
            continue
        if name not in linear_inputs or len(linear_inputs[name]) == 0:
            continue

        ti = module.weight.tensor_impl
        if not hasattr(ti, 'scale_and_zero'):
            continue

        # Get original weights (dequantize with current scale/zero)
        w_orig = module.weight.dequantize().detach().clone()

        # Concatenate calibration inputs
        cal_x = torch.cat(linear_inputs[name], dim=0)
        cal_x = cal_x.reshape(-1, cal_x.shape[-1])[:1024]  # Limit samples

        # Target output
        with torch.no_grad():
            target = (w_orig @ cal_x.T).float()

        # Optimize scale_and_zero
        sz = ti.scale_and_zero.data.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([sz], lr=lr)

        for step in range(n_steps):
            ti.scale_and_zero.data.copy_(sz)
            w_dq = module.weight.dequantize()
            output = (w_dq @ cal_x.T).float()
            loss = (output - target).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Apply optimized scale/zero
        ti.scale_and_zero.data.copy_(sz.detach())

    del linear_inputs
    gc.collect()
    torch.cuda.empty_cache()


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

    for i, layer in enumerate(model.model.layers):
        layer.to(device)
        quantize_(layer, config)
        gc.collect()
        torch.cuda.empty_cache()

    # Post-quantization scale optimization using calibration data
    print("Optimizing scales with calibration data...")
    calib_data = load_calibration_data(tokenizer)

    # Collect hidden states layer by layer
    model.eval()
    hidden_states_list = []
    with torch.no_grad():
        for input_ids in calib_data[:4]:
            input_ids = input_ids.to(device)
            h = model.model.embed_tokens(input_ids)
            hidden_states_list.append(h)

    # Optimize scales for each layer
    for i, layer in enumerate(model.model.layers):
        try:
            optimize_scales(layer, i, hidden_states_list, device)
            # Forward hidden states through this layer for next layer
            with torch.no_grad():
                new_hidden = []
                for h in hidden_states_list:
                    pos_ids = torch.arange(h.shape[1], device=device).unsqueeze(0)
                    out = layer(h, position_ids=pos_ids)
                    new_hidden.append(out[0])
                hidden_states_list = new_hidden
        except Exception as e:
            print(f"  Layer {i} scale opt failed: {e}")
            with torch.no_grad():
                new_hidden = []
                for h in hidden_states_list:
                    pos_ids = torch.arange(h.shape[1], device=device).unsqueeze(0)
                    out = layer(h, position_ids=pos_ids)
                    new_hidden.append(out[0])
                hidden_states_list = new_hidden

    del hidden_states_list
    gc.collect()
    torch.cuda.empty_cache()

    # Prompt lookup for speculative decoding
    model.generation_config.prompt_lookup_num_tokens = 256

    print("Done.")
    return model, tokenizer
