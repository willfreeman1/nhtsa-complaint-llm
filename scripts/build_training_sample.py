"""Build the Rung-1 training sample: a stratified pull from data/cmpl_clean.parquet,
capped at --per-class rows per COMPDESC_LABEL class, excluding every cmplid already
used in the gold eval set (no train/test leakage).

All 31 training classes have >=4,520 real rows available even after the PROD_TYPE=='V'
scope + duplicate-merge (see output/label_schema.json), so a 2,000/class target is
achievable for every class without exhausting any class's population.

Writes:
    data/training_sample.parquet   -- full rows (gitignored, regenerate via this script)
    output/training_sample_summary.json -- small, trackable per-class counts

Usage:
    python scripts/build_training_sample.py --per-class 2000
    python scripts/build_training_sample.py --per-class 20 --out-suffix _smoketest
"""
import argparse
import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent.parent / "output"
SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    df = pd.read_parquet(DATA_DIR / "cmpl_clean.parquet")
    print(f"Loaded {len(df):,} cleaned vehicle rows")

    with open(OUT_DIR / "gold_eval_set_v2.json") as f:
        gold = json.load(f)
    # CMPLID is stored as str in cmpl_clean.parquet -- cast gold ids to match, or
    # isin() silently matches nothing and every gold row leaks straight into training.
    gold_ids = {str(row["cmplid"]) for row in gold["rows"]}
    before = len(df)
    df = df[~df["CMPLID"].astype(str).isin(gold_ids)]
    excluded = before - len(df)
    print(f"Excluded {excluded:,} rows already in gold_eval_set_v2 (no train/test leakage)")
    assert excluded == len(gold["rows"]), (
        f"expected to exclude all {len(gold['rows'])} gold rows, only matched {excluded} -- "
        "check CMPLID dtype/format before trusting the sample"
    )

    parts = []
    per_class_actual = {}
    for label, group in df.groupby("COMPDESC_LABEL"):
        n = min(args.per_class, len(group))
        parts.append(group.sample(n=n, random_state=args.seed))
        per_class_actual[label] = n
    sample = pd.concat(parts).sample(frac=1, random_state=args.seed).reset_index(drop=True)

    print(f"\nBuilt stratified sample: {len(sample):,} rows across {len(per_class_actual)} classes")
    for label, n in sorted(per_class_actual.items(), key=lambda kv: -kv[1]):
        flag = "  <-- capped by availability" if n < args.per_class else ""
        print(f"  {label:35s} {n:>6,}{flag}")

    out_path = DATA_DIR / f"training_sample{args.out_suffix}.parquet"
    sample.to_parquet(out_path, index=False)
    print(f"\nWrote {out_path}")

    summary = {
        "per_class_target": args.per_class,
        "seed": args.seed,
        "n_total": len(sample),
        "n_classes": len(per_class_actual),
        "n_excluded_gold_rows": before - len(df),
        "per_class_actual": dict(sorted(per_class_actual.items(), key=lambda kv: -kv[1])),
        "output_file": out_path.name,
    }
    summary_path = OUT_DIR / f"training_sample_summary{args.out_suffix}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
