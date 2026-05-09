import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ACT_DIR = Path("results/activations")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, default=22)
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=256)
    return p.parse_args()


def load_pairs(harmful_base, benign_base, category):
    hX = np.load(ACT_DIR / f"{harmful_base}.npy")
    bX = np.load(ACT_DIR / f"{benign_base}.npy")
    with open(ACT_DIR / f"{harmful_base}.meta.jsonl") as f:
        hm = [json.loads(l) for l in f]
    with open(ACT_DIR / f"{benign_base}.meta.jsonl") as f:
        bm = [json.loads(l) for l in f]
    b_by_id = {m["id"]: i for i, m in enumerate(bm)}
    h_idx, b_idx = [], []
    for i, m in enumerate(hm):
        if m.get("category") != category: continue
        pid = m.get("paired_id")
        if pid and pid in b_by_id:
            h_idx.append(i); b_idx.append(b_by_id[pid])
    return hX[h_idx], bX[b_idx], [hm[i]["id"] for i in h_idx]


def find_shared_dims(layer, k):
    ch_h, ch_b, ch_ids = load_pairs("llama8b__medical_harmful",
                                     "llama8b__medical_benign",
                                     "VII_participate_community_health")
    mi_h, mi_b, mi_ids = load_pairs("llama8b__general_harmful",
                                     "llama8b__general_benign",
                                     "misinformation_disinformation")
    ch_c = (ch_h - ch_b).mean(axis=0)[layer]
    mi_c = (mi_h - mi_b).mean(axis=0)[layer]
    ch_top = set(np.argsort(-np.abs(ch_c))[:k].tolist())
    mi_top = set(np.argsort(-np.abs(mi_c))[:k].tolist())
    return sorted(ch_top & mi_top), ch_ids, mi_ids


def make_hook(dim_indices, device):
    dim_tensor = torch.tensor(dim_indices, dtype=torch.long, device=device)
    def hook(module, inp, out):
        if isinstance(out, tuple):
            hidden = out[0].clone()
            hidden[..., dim_tensor] = 0
            return (hidden,) + out[1:]
        hidden = out.clone()
        hidden[..., dim_tensor] = 0
        return hidden
    return hook


def get_prompts(prompts_file, ids_subset):
    with open(prompts_file) as f:
        prompts = json.load(f)
    keep = set(ids_subset)
    return [p for p in prompts if p["id"] in keep]


def generate(model, tok, prompt, mx):
    chat = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=mx, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def run_condition(model, tok, prompts, label, mx):
    rows = []
    for p in tqdm(prompts, desc=label):
        try:
            r = generate(model, tok, p["prompt"], mx)
        except Exception as e:
            r = f"[ERROR: {e}]"
        rows.append({**p, "response": r, "condition": label})
    return rows


def main():
    args = parse_args()
    shared, ch_ids, mi_ids = find_shared_dims(args.layer, args.k)
    print(f"Layer {args.layer}: {len(shared)} shared dims (top-{args.k})")
    print(f"  Community-health: {len(ch_ids)} prompts; Misinformation: {len(mi_ids)} prompts")

    ch_prompts = get_prompts("prompts/medical_harmful.json", ch_ids)
    mi_prompts = get_prompts("prompts/general_harmful.json", mi_ids)
    all_p = ch_prompts + mi_prompts

    rng = np.random.default_rng(args.seed)
    pool = [i for i in range(4096) if i not in set(shared)]
    random_dims = sorted(rng.choice(pool, size=len(shared), replace=False).tolist())

    print("Loading meta-llama/Llama-3.1-8B-Instruct...")
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct",
                                         token=os.environ.get("HF_TOKEN"))
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        quantization_config=bnb, device_map="auto",
        token=os.environ.get("HF_TOKEN"))
    model.eval()

    rows = []
    rows += run_condition(model, tok, all_p, "baseline", args.max_new_tokens)

    print(f"Shared ablation: layer {args.layer}, {len(shared)} dims")
    h = model.model.layers[args.layer].register_forward_hook(make_hook(shared, model.device))
    try:
        rows += run_condition(model, tok, all_p, "shared", args.max_new_tokens)
    finally:
        h.remove()

    print(f"Random ablation control: layer {args.layer}, {len(random_dims)} dims")
    h = model.model.layers[args.layer].register_forward_hook(make_hook(random_dims, model.device))
    try:
        rows += run_condition(model, tok, all_p, "random", args.max_new_tokens)
    finally:
        h.remove()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in rows:
            line = json.dumps({**r, "shared_dims": shared, "random_dims": random_dims,
                               "ablation_layer": args.layer}, ensure_ascii=False)
            f.write(line + "\n")
    print(f"Saved {len(rows)} records to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
