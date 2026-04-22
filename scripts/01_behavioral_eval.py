import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--quantize", action="store_true",
                   help="Use bitsandbytes 4-bit quant (required for 8B on T4/A100).")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--n-prompts", type=int, default=None,
                   help="Limit for debugging.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_model_kwargs(quantize: bool) -> dict:
    kwargs: dict = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if quantize:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    return kwargs


def load_done_ids(path: Path) -> set:
    done = set()
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.prompts, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    if args.n_prompts is not None:
        prompts = prompts[: args.n_prompts]
    print(f"Loaded {len(prompts)} prompts from {args.prompts}")

    done = load_done_ids(args.output)
    if done:
        print(f"Resume: {len(done)} IDs already in {args.output}; skipping.")
    todo = [p for p in prompts if p["id"] not in done]
    if not todo:
        print("All prompts already processed.")
        return 0
    print(f"Generating {len(todo)} responses.")

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.chat_template is not None, (
        f"{args.model} has no chat template. Use an Instruct variant."
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, **build_model_kwargs(args.quantize)
    )
    model.eval()
    print(f"  layers: {len(model.model.layers)}, dtype: {next(model.parameters()).dtype}, "
          f"device: {next(model.parameters()).device}")

    with open(args.output, "a", encoding="utf-8") as fout:
        for p in tqdm(todo, desc="generating"):
            messages = [{"role": "user", "content": p["prompt"]}]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            response = tokenizer.decode(
                out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()
            record = {
                "id": p["id"],
                "prompt": p["prompt"],
                "response": response,
                "category": p.get("category", "unknown"),
                "paired_id": p.get("paired_id"),
                "model": args.model,
                "promptset": args.prompts.stem,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"\nWrote {len(todo)} responses to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())