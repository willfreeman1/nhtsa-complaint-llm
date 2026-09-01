"""Print gold-eval-set rows in reviewable batches, with each model's prediction.

Used for a manual read-through of every row's assigned truth label. Model predictions
are shown because unanimous model disagreement is a strong hint that the truth label
(not the models) is wrong.

    python scripts/dump_gold_for_review.py --start 0 --count 35
"""
import argparse
import glob
import json
import sys
from pathlib import Path

# Narratives contain smart quotes and other non-cp1252 characters that crash the
# default Windows console encoding.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = Path(__file__).parent.parent / "output"
MAX_NARRATIVE = 1400


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=35)
    args = ap.parse_args()

    with open(OUT_DIR / "gold_eval_set.json") as f:
        gold = json.load(f)

    preds = {}
    for path in sorted(glob.glob(str(OUT_DIR / "model_eval_*.json"))):
        with open(path) as f:
            e = json.load(f)
        short = e["model"].replace("gpt-", "").replace("claude-", "")
        for r in e["results"]:
            preds.setdefault(r["cmplid"], {})[short] = r["pred"]

    rows = gold["rows"][args.start: args.start + args.count]
    for i, r in enumerate(rows, start=args.start):
        p = preds.get(r["cmplid"], {})
        disagree = {k: v for k, v in p.items() if v not in r["accept"]}
        print(f"--- [{i}] cmplid={r['cmplid']} | {r['year']} {r['make']} {r['model']}")
        print(f"    ACCEPT: {r['accept']}   ({r['truth_source']})")
        if disagree:
            print(f"    MODELS DISAGREE: {disagree}")
        if r["adjudication_note"]:
            print(f"    prior note: {r['adjudication_note']}")
        text = " ".join(str(r["narrative"]).split())
        if len(text) > MAX_NARRATIVE:
            text = text[:MAX_NARRATIVE] + " ...[TRUNCATED]"
        print(f"    {text}")
        print()
    print(f"(showed rows {args.start}-{args.start + len(rows) - 1} of {gold['n']})")


if __name__ == "__main__":
    main()
