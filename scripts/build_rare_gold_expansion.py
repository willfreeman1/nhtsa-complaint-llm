"""Pull a small, deliberately-targeted sample of real complaints for the 9
`raw_component` classes that have ZERO examples in the 498-row gold_eval_set_v2.json
(see WRITEUP.md section 4.5/6) -- CHILD SEAT, ELECTRONIC STABILITY CONTROL,
EQUIPMENT ADAPTIVE/MOBILITY, FIRERELATED, FUEL SYSTEM OTHER, FUEL/PROPULSION SYSTEM,
SERVICE BRAKES ELECTRIC, TRACTION CONTROL SYSTEM, TRAILER HITCHES.

Unlike the rest of gold_eval_set_v2.json (a random pilot sample later adjudicated),
these rows are intentionally selected by literal COMPDESC_TOP match -- this is a
disclosed, deliberate coverage sample, not a random one, same spirit as the
rare-category training top-up in build_training_sample_rare_topup.py. Excludes any
CMPLID already used in the gold eval set or ANY training data (original sample +
rare topup), so there's no train/eval leakage.

Writes output/rare_gold_expansion_candidates.json for manual adjudication (read each
narrative, decide the correct accept-list label(s) using the class definitions in
teacher_prompt.py -- NHTSA's own COMPDESC is one hypothesis, not ground truth, per the
project's established gold-set methodology) via apply_rare_gold_expansion.py.

Usage:
    python scripts/build_rare_gold_expansion.py
"""
import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent.parent / "output"
SEED = 43  # different from the rare-topup training sample's seed, just to decorrelate
PER_CATEGORY = 7

MISSING_CATEGORIES = [
    "CHILD SEAT",
    "ELECTRONIC STABILITY CONTROL",
    "EQUIPMENT ADAPTIVE/MOBILITY",
    "FIRERELATED",
    "FUEL SYSTEM, OTHER",
    "FUEL/PROPULSION SYSTEM",
    "SERVICE BRAKES, ELECTRIC",
    "TRACTION CONTROL SYSTEM",
    "TRAILER HITCHES",
]


def main():
    df = pd.read_parquet(DATA_DIR / "cmpl_clean.parquet")
    print(f"Loaded {len(df):,} cleaned vehicle rows")

    with open(OUT_DIR / "gold_eval_set_v2.json") as f:
        gold = json.load(f)
    gold_ids = {str(row["cmplid"]) for row in gold["rows"]}

    exclude_ids = set(gold_ids)
    for fname in ["training_sample.parquet", "training_sample_rare_topup.parquet"]:
        path = DATA_DIR / fname
        if path.exists():
            ids = set(pd.read_parquet(path, columns=["CMPLID"])["CMPLID"].astype(str))
            exclude_ids |= ids
            print(f"  + {len(ids):,} CMPLIDs from {fname}")
    print(f"Excluding {len(exclude_ids):,} unique CMPLIDs already used (gold + all training)")

    df = df[~df["CMPLID"].astype(str).isin(exclude_ids)]

    candidates = []
    for category in MISSING_CATEGORIES:
        pool = df[df["COMPDESC_TOP"] == category]
        n = min(PER_CATEGORY, len(pool))
        picked = pool.sample(n=n, random_state=SEED)
        print(f"{category:<32} {len(pool):>6,} available -> sampled {n}")
        for _, row in picked.iterrows():
            candidates.append({
                "cmplid": str(row["CMPLID"]),
                "make": row["MAKETXT"],
                "model": row["MODELTXT"],
                "year": row["YEARTXT"],
                "narrative": row["CDESCR"],
                "nhtsa_compdesc_raw": row["COMPDESC"],
                "nhtsa_compdesc_top": row["COMPDESC_TOP"],
                "nhtsa_crash": row["CRASH"],
                "nhtsa_fire": row["FIRE"],
                "nhtsa_injured": int(row["INJURED"]) if pd.notna(row["INJURED"]) else 0,
                "nhtsa_deaths": int(row["DEATHS"]) if pd.notna(row["DEATHS"]) else 0,
                "target_category": category,
            })

    out_path = OUT_DIR / "rare_gold_expansion_candidates.json"
    with open(out_path, "w") as f:
        json.dump({"n": len(candidates), "per_category": PER_CATEGORY, "rows": candidates}, f, indent=2)
    print(f"\nWrote {out_path} ({len(candidates)} rows across {len(MISSING_CATEGORIES)} categories)")


if __name__ == "__main__":
    main()
