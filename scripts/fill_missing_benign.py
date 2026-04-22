import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--domain", choices=["general", "medical"], required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    h_path = Path(f"prompts/{args.domain}_harmful.json")
    b_path = Path(f"prompts/{args.domain}_benign.json")

    harmful = json.load(open(h_path, encoding="utf-8"))
    benign = json.load(open(b_path, encoding="utf-8"))

    b_paired = {x["paired_id"] for x in benign}
    missing = [x for x in harmful if x["id"] not in b_paired]

    print(f"\n{len(missing)} missing benign counterparts for {args.domain}.\n")
    print("For each prompt: type your benign version and press Enter.")
    print("  's' + Enter = skip this one")
    print("  'q' + Enter = save and quit\n")

    for i, h in enumerate(missing):
        print("=" * 70)
        print(f"[{i+1}/{len(missing)}]  {h['id']}  category={h['category']}")
        print(f"HARMFUL: {h['prompt']}")
        print()
        try:
            response = input("BENIGN: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nInterrupted.")
            break

        if response.lower() == "q":
            break
        if response.lower() == "s" or not response:
            print("  skipped.")
            continue

        entry = {
            "id": h["paired_id"],
            "prompt": response,
            "category": h["category"],
            "paired_id": h["id"],
        }
        benign.append(entry)
        with open(b_path, "w", encoding="utf-8") as f:
            json.dump(benign, f, indent=2, ensure_ascii=False)
        print(f"  saved. total benign: {len(benign)}")

    print(f"\nDone. {b_path} has {len(benign)} entries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())