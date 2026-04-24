"""
02_activation_analysis.py -- Collect residual-stream activations on prompts.

For a given model and prompt set, runs each prompt through the model with
output_hidden_states=True, captures the hidden state at the final prompt
token at every layer, and saves to .npy.

Output shape: (n_prompts, n_layers, hidden_dim), float32.
Plus a .meta.jsonl with prompt id, category, paired_id in the same order.

Usage:
    python scripts/02_activation_analysis.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --prompts prompts/medical_harmful.json \\
        --output results/activations/llama8b__medical_harmful.npy \\
        --quantize
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--quantize", action="store_true")
    p.add_argument("--n-prompts", type=int, default=None)
    return p.parse_args()


def build_model_kwargs(quantize):
    kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16}
    if quantize:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    return kwargs


def main():
    args = parse_args()

    with open(args.prompts) as f:
        prompts = json.load(f)
    print(f"Loaded {len(prompts)} prompts from {args.prompts}")
    if args.n_prompts:
        prompts = prompts[:args.n_prompts]

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=os.environ.get("HF_TOKEN"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        token=os.environ.get("HF_TOKEN"),
        **build_model_kwargs(args.quantize),
    )
    model.eval()

    hidden_dim = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    print(f"  layers: {n_layers}, hidden_dim: {hidden_dim}")

    out_array = np.zeros((len(prompts), n_layers, hidden_dim), dtype=np.float32)
    meta_records = []

    for i, p in enumerate(tqdm(prompts, desc="collecting")):
        chat = [{"role": "user", "content": p["prompt"]}]
        text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model(**inputs, output_hidden_states=True)

        last_pos = inputs["input_ids"].shape[1] - 1
        for l in range(n_layers):
            out_array[i, l] = output.hidden_states[l + 1][0, last_pos].float().cpu().numpy()

        meta_records.append({
            "id": p["id"],
            "category": p.get("category", "unknown"),
            "paired_id": p.get("paired_id"),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, out_array)
    meta_path = args.output.with_suffix(".meta.jsonl")
    with open(meta_path, "w") as f:
        for r in meta_records:
            f.write(json.dumps(r) + "\n")

    print(f"\nSaved activations: {args.output} shape={out_array.shape}")
    print(f"Saved metadata:    {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
