"""Build a topup candidate sample targeting the 9 weakest rare categories (all of
RARE_CATEGORIES except COMMUNICATION -- see teacher_prompt.py) by pulling directly
from cmpl_clean.parquet's literal NHTSA-assigned COMPDESC_TOP column (an exact string
match against each rare category name), excluding any CMPLID already present in the
gold eval set or the original training_sample.parquet (no train/test leakage, no
re-labeling rows we already have).

Mirrors the pattern in build_training_sample.py: same column set, same exclusion
logic, same seed convention (random_state=42).

Usage:
    python scripts/build_training_sample_rare_topup.py
"""
import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent.parent / "output"
SEED = 42

# Per-category targets, chosen based on real corpus availability (capped at what's
# actually available so we don't oversample beyond real data). COMMUNICATION is
# deliberately excluded -- it already has 619 teacher-labeled examples in the
# existing training set, and topping it up via literal COMPDESC_TOP match (only 108
# total instances in the whole corpus) would not help.
TARGETS = {
    "FUEL SYSTEM, DIESEL": 800,
    "CHILD SEAT": 800,
    "INTERIOR LIGHTING": 800,
    "SERVICE BRAKES, ELECTRIC": 800,
    "TRACTION CONTROL SYSTEM": 800,
    "EQUIPMENT ADAPTIVE/MOBILITY": 800,
    "TRAILER HITCHES": 800,
    "HYBRID PROPULSION SYSTEM": 800,
    "FIRERELATED": 800,
}


def main():
    df = pd.read_parquet(DATA_DIR / "cmpl_clean.parquet")
    print(f"Loaded {len(df):,} cleaned vehicle rows")

    with open(OUT_DIR / "gold_eval_set_v2.json") as f:
        gold = json.load(f)
    gold_ids = {str(row["cmplid"]) for row in gold["rows"]}

    training_sample = pd.read_parquet(DATA_DIR / "training_sample.parquet")
    training_ids = set(training_sample["CMPLID"].astype(str))

    exclude_ids = gold_ids | training_ids
    print(f"Excluding {len(gold_ids):,} gold-eval CMPLIDs and {len(training_ids):,} "
          f"existing training_sample CMPLIDs ({len(exclude_ids):,} unique total)")

    before = len(df)
    df = df[~df["CMPLID"].astype(str).isin(exclude_ids)]
    print(f"Excluded {before - len(df):,} rows already used; {len(df):,} remain eligible")

    parts = []
    per_category_actual = {}
    per_category_available = {}
    for category, target in TARGETS.items():
        available = df[df["COMPDESC_TOP"] == category]
        n_available = len(available)
        n = min(target, n_available)
        per_category_available[category] = n_available
        per_category_actual[category] = n
        if n > 0:
            parts.append(available.sample(n=n, random_state=SEED))

    sample = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)

    print(f"\nBuilt rare-topup sample: {len(sample):,} rows across {len(TARGETS)} categories")
    print(f"\n{'Category':35s} {'Target':>8s} {'Available':>10s} {'Actual':>8s}")
    for category, target in TARGETS.items():
        avail = per_category_available[category]
        actual = per_category_actual[category]
        flag = "  <-- capped by availability" if actual < target else ""
        print(f"{category:35s} {target:>8,} {avail:>10,} {actual:>8,}{flag}")

    out_path = DATA_DIR / "training_sample_rare_topup.parquet"
    sample.to_parquet(out_path, index=False)
    print(f"\nWrote {out_path}")

    summary = {
        "targets": TARGETS,
        "seed": SEED,
        "n_total": len(sample),
        "n_categories": len(TARGETS),
        "n_excluded_gold_ids": len(gold_ids),
        "n_excluded_training_sample_ids": len(training_ids),
        "n_excluded_unique_total": len(exclude_ids),
        "n_excluded_rows_removed": before - len(df),
        "per_category_available": per_category_available,
        "per_category_actual": dict(sorted(per_category_actual.items(), key=lambda kv: -kv[1])),
        "output_file": out_path.name,
    }
    summary_path = OUT_DIR / "training_sample_summary_rare_topup.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
