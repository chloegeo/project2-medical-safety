"""
03e_structured_ablation.py -- Structured-baseline ablation control.

For each non-focal (medical x general) category pair, computes the pair's
top-k shared dimensions at layer 22 in base Llama, ablates those dims, and
measures refusal change on the FOCAL-pair prompts (CH + MI). Builds a
structured null distribution to compare against the focal-pair effect.
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ACT_DIR = Path("results/activations")
FOCAL_MED = "VII_participate_community_health"
FOCAL_GEN = "misinformation_disinformation"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, default=22)
    p.add_argument("--target-size", type=int, default=31)
    p.add_argument("--top-k-initial", type=int, default=100)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--limit-pairs", type=int, default=None)
    return p.parse_args()


def load_activations(harmful_base, benign_base):
    hX = np.load(ACT_DIR / f"{harmful_base}.npy")
    bX = np.load(ACT_DIR / f"{benign_base}.npy")
    with open(ACT_DIR / f"{harmful_base}.meta.jsonl") as f:
        hm = [json.loads(l) for l in f]
    with open(ACT_DIR / f"{benign_base}.meta.jsonl") as f:
        bm = [json.loads(l) for l in f]
    return hX, bX, hm, bm


def get_pair_indices(hm, bm, category):
    b_by_id = {m["id"]: i for i, m in enumerate(bm)}
    h_idx, b_idx, ids = [], [], []
    for i, m in enumerate(hm):
        if m.get("category") != category:
            continue
        pid = m.get("paired_id")
        if pid and pid in b_by_id:
            h_idx.append(i); b_idx.append(b_by_id[pid]); ids.append(m["id"])
    return h_idx, b_idx, ids


def shared_dims_for_pair(hX_med, bX_med, h_idx_med, b_idx_med,
                          hX_gen, bX_gen, h_idx_gen, b_idx_gen,
                          layer, target_size, top_k_init):
    if not h_idx_med or not h_idx_gen:
        return []
    med_c = (hX_med[h_idx_med] - bX_med[b_idx_med]).mean(axis=0)[layer]
    gen_c = (hX_gen[h_idx_gen] - bX_gen[b_idx_gen]).mean(axis=0)[layer]
    k = top_k_init
    while True:
        med_top = set(np.argsort(-np.abs(med_c))[:k].tolist())
        gen_top = set(np.argsort(-np.abs(gen_c))[:k].tolist())
        shared = sorted(med_top & gen_top)
        if len(shared) >= target_size or k >= med_c.shape[0]:
            break
        k = min(int(k * 1.5) + 1, med_c.shape[0])
    if len(shared) > target_size:
        arr = np.array(shared)
        score = np.abs(med_c[arr]) + np.abs(gen_c[arr])
        keep = arr[np.argsort(-score)[:target_size]]
        shared = sorted(keep.tolist())
    return shared


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
    for p in tqdm(prompts, desc=label, leave=False):
        try:
            r = generate(model, tok, p["prompt"], mx)
        except Exception as e:
            r = f"[ERROR: {e}]"
        rows.append({**p, "response": r, "condition": label})
    return rows


MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


def main():
    args = parse_args()
    print(f"Layer={args.layer}, target_size={args.target_size}")

    print("Loading activations...")
    hX_med, bX_med, hm_med, bm_med = load_activations("llama8b__medical_harmful",
                                                       "llama8b__medical_benign")
    hX_gen, bX_gen, hm_gen, bm_gen = load_activations("llama8b__general_harmful",
                                                       "llama8b__general_benign")

    med_cats = sorted({m["category"] for m in hm_med if m.get("category")})
    gen_cats = sorted({m["category"] for m in hm_gen if m.get("category")})
    print(f"Medical categories: {len(med_cats)}, General categories: {len(gen_cats)}")

    med_indices = {c: get_pair_indices(hm_med, bm_med, c) for c in med_cats}
    gen_indices = {c: get_pair_indices(hm_gen, bm_gen, c) for c in gen_cats}

    fmh, fmb, _ = med_indices[FOCAL_MED]
    fgh, fgb, _ = gen_indices[FOCAL_GEN]
    focal_shared = shared_dims_for_pair(hX_med, bX_med, fmh, fmb,
                                         hX_gen, bX_gen, fgh, fgb,
                                         args.layer, args.target_size, args.top_k_initial)
    print(f"Focal pair shared dims: {len(focal_shared)}")

    other_pair_dims = []
    for mc in med_cats:
        for gc in gen_cats:
            if mc == FOCAL_MED and gc == FOCAL_GEN:
                continue
            mh, mb, _ = med_indices[mc]
            gh, gb, _ = gen_indices[gc]
            if not mh or not gh:
                continue
            dims = shared_dims_for_pair(hX_med, bX_med, mh, mb,
                                         hX_gen, bX_gen, gh, gb,
                                         args.layer, args.target_size, args.top_k_initial)
            other_pair_dims.append({
                "med_cat": mc, "gen_cat": gc, "n_dims": len(dims),
                "overlap_with_focal": len(set(dims) & set(focal_shared)),
                "dims": dims,
            })

    if args.limit_pairs:
        other_pair_dims = other_pair_dims[:args.limit_pairs]
    print(f"Non-focal pairs to test: {len(other_pair_dims)}")
    sizes = [opd["n_dims"] for opd in other_pair_dims]
    print(f"  Sizes: min={min(sizes)}, max={max(sizes)}, median={int(np.median(sizes))}")

    _, _, ch_ids = med_indices[FOCAL_MED]
    _, _, mi_ids = gen_indices[FOCAL_GEN]
    ch_prompts = get_prompts("prompts/medical_harmful.json", ch_ids)
    mi_prompts = get_prompts("prompts/general_harmful.json", mi_ids)
    all_prompts = ch_prompts + mi_prompts
    print(f"Focal prompts: {len(ch_prompts)} CH + {len(mi_prompts)} MI = {len(all_prompts)}")

    print(f"\nLoading {MODEL_ID}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, token=os.environ.get("HF_TOKEN"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto",
        token=os.environ.get("HF_TOKEN"))
    model.eval()

    rows = []

    print("\n=== baseline ===")
    rows += run_condition(model, tok, all_prompts, "baseline", args.max_new_tokens)

    print(f"\n=== focal ({len(focal_shared)} dims) ===")
    h = model.model.layers[args.layer].register_forward_hook(
        make_hook(focal_shared, model.device))
    try:
        rows += run_condition(model, tok, all_prompts, "focal", args.max_new_tokens)
    finally:
        h.remove()

    for i, opd in enumerate(other_pair_dims):
        label = f"other__{opd['med_cat']}__x__{opd['gen_cat']}"
        print(f"\n=== [{i+1}/{len(other_pair_dims)}] {label} "
              f"({opd['n_dims']} dims; overlap_w_focal={opd['overlap_with_focal']}) ===")
        h = model.model.layers[args.layer].register_forward_hook(
            make_hook(opd['dims'], model.device))
        try:
            rows += run_condition(model, tok, all_prompts, label, args.max_new_tokens)
        finally:
            h.remove()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({**r, "ablation_layer": args.layer}, ensure_ascii=False) + "\n")

    meta_path = args.output.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "layer": args.layer,
            "target_size": args.target_size,
            "top_k_initial": args.top_k_initial,
            "focal_shared_dims": focal_shared,
            "n_focal_dims": len(focal_shared),
            "n_pairs": len(other_pair_dims),
            "other_pair_dims": other_pair_dims,
        }, f, indent=2)

    print(f"\nSaved {len(rows)} rows to {args.output}")
    print(f"Saved metadata to {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
