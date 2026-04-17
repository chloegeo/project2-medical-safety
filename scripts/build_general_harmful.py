"""
build_general_harmful.py -- Construct prompts/general_harmful.json from HarmBench.
 
Downloads HarmBench's text-behaviors CSV, filters to the 'standard' functional
category (core harmful behaviors with no contextual setup), stratified-samples
across semantic categories to get ~100 prompts, and writes them in the project's
JSON schema:
 
    [{"id": str, "prompt": str, "category": str, "paired_id": str}, ...]
 
The paired_id field points to the (yet-to-be-generated) benign counterpart.
IDs are assigned gh_001 through gh_N (general_harmful), with matching
paired_ids gb_001 through gb_N (general_benign).
 
Usage:
    python scripts/build_general_harmful.py
    python scripts/build_general_harmful.py --seed 123
    python scripts/build_general_harmful.py --target-n 100
 
Run once. After running, manually spot-check a handful of prompts in the
output JSON before moving on to benign counterpart generation.
"""
import argparse
import json
import sys
from pathlib import Path
 
import pandas as pd
 
HARMBENCH_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/"
    "main/data/behavior_datasets/harmbench_behaviors_text_all.csv"
)
 
 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build general_harmful.json from HarmBench standard behaviors."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--target-n",
        type=int,
        default=100,
        help="Target total number of prompts to sample.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("prompts/general_harmful.json"),
        help="Output JSON path (relative to repo root).",
    )
    return parser.parse_args()
 
 
def stratified_sample(
    df: pd.DataFrame, strata_col: str, target_n: int, seed: int
) -> pd.DataFrame:
    """Approximately-equal stratified sample across strata_col.
 
    Distributes target_n roughly equally across strata. If a stratum has fewer
    rows than its quota, takes all of it and backfills from the leftover pool.
    Final result is shuffled to avoid category grouping in the output order.
    """
    strata = sorted(df[strata_col].unique())
    n_strata = len(strata)
    quota = target_n // n_strata
    extra = target_n - quota * n_strata
 
    first_pass = []
    leftover_pool = []
    taken = 0
 
    for i, s in enumerate(strata):
        s_df = df[df[strata_col] == s]
        quota_s = quota + (1 if i < extra else 0)
        if len(s_df) <= quota_s:
            first_pass.append(s_df)
            taken += len(s_df)
        else:
            sampled = s_df.sample(n=quota_s, random_state=seed)
            first_pass.append(sampled)
            taken += quota_s
            leftover_pool.append(s_df.drop(sampled.index))
 
    result = pd.concat(first_pass, ignore_index=True)
 
    # Backfill from the leftover pool if any strata were smaller than their quota.
    if taken < target_n and leftover_pool:
        leftover = pd.concat(leftover_pool, ignore_index=True)
        backfill_n = min(target_n - taken, len(leftover))
        if backfill_n > 0:
            backfill = leftover.sample(n=backfill_n, random_state=seed)
            result = pd.concat([result, backfill], ignore_index=True)
 
    # Shuffle so the output JSON isn't grouped by category.
    return result.sample(frac=1, random_state=seed).reset_index(drop=True)
 
 
def main() -> int:
    args = parse_args()
 
    print(f"Downloading HarmBench behaviors CSV...")
    print(f"  URL: {HARMBENCH_URL}")
    df = pd.read_csv(HARMBENCH_URL)
    print(f"  Total rows loaded: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    assert "FunctionalCategory" in df.columns, (
        "Expected 'FunctionalCategory' column -- HarmBench schema may have "
        f"changed. Got: {list(df.columns)}"
    )
    assert "SemanticCategory" in df.columns, (
        "Expected 'SemanticCategory' column. Got: {list(df.columns)}"
    )
    assert "Behavior" in df.columns, (
        "Expected 'Behavior' column. Got: {list(df.columns)}"
    )
 
    # Filter to standard functional category.
    # HarmBench categorizes behaviors as: standard (plain harmful asks),
    # contextual (need a context string), or copyright. We want 'standard'
    # because it's self-contained and directly comparable to benign prompts.
    standard = df[df["FunctionalCategory"] == "standard"].copy()
    print(f"\n  Standard behaviors: {len(standard)}")
 
    cat_counts = standard["SemanticCategory"].value_counts()
    print(f"\n  Semantic category distribution (pre-sample):")
    for cat, count in cat_counts.items():
        print(f"    {cat:>32s}: {count}")
 
    if args.target_n > len(standard):
        print(
            f"\n  WARNING: target-n ({args.target_n}) > available "
            f"standard behaviors ({len(standard)}). Will sample all."
        )
        args.target_n = len(standard)
 
    sampled = stratified_sample(
        standard, "SemanticCategory", args.target_n, args.seed
    )
    print(f"\n  Sampled {len(sampled)} prompts total.")
 
    final_counts = sampled["SemanticCategory"].value_counts()
    print(f"  Final category distribution:")
    for cat, count in final_counts.items():
        print(f"    {cat:>32s}: {count}")
 
    # Build JSON entries.
    entries = []
    for i, row in sampled.iterrows():
        prompt_text = str(row["Behavior"]).strip()
        # Standard behaviors should have empty ContextString, but guard for drift.
        context = row.get("ContextString", None)
        if pd.notna(context) and str(context).strip():
            prompt_text = f"{str(context).strip()}\n\n{prompt_text}"
 
        entry = {
            "id": f"gh_{i + 1:03d}",
            "prompt": prompt_text,
            "category": str(row["SemanticCategory"]),
            "paired_id": f"gb_{i + 1:03d}",
        }
        entries.append(entry)
 
    # Save.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
 
    print(f"\nWrote {len(entries)} prompts to {args.output}")
    print(f"\nFirst 3 entries (preview):")
    for e in entries[:3]:
        preview = e["prompt"][:100] + ("..." if len(e["prompt"]) > 100 else "")
        print(f"  {e['id']} [{e['category']}]")
        print(f"    {preview}")
 
    print(
        f"\nNext step: open {args.output} in VS Code, skim 10-20 random "
        f"entries, confirm they read as harmful and match their category label. "
        f"Then move on to build_medical_harmful.py."
    )
    return 0
 
 
if __name__ == "__main__":
    sys.exit(main())
