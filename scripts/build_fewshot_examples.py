"""Sample real candidate narratives to use as few-shot examples in the teacher prompt.

Pulls 3 candidates per kept label (30 classes + UNKNOWN OR OTHER) with readable-length
narratives (150-450 chars), plus targeted candidates for two specific gotchas:
  - "fire routes to causal system, not a generic fire label" (B1/B2)
  - "OTHER means a known-but-rare specific system, not vagueness" (disambiguating
    OTHER from UNKNOWN OR OTHER)
Writes output/fewshot_candidates.json for manual curation into the final prompt.
"""
import json
import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent.parent / "output"
SEED = 42

FIRE_RE = re.compile(r"\b(fire|flame|caught on fire|smoke|burn(?:ed|ing)?)\b", re.I)

# Real, specific top-level COMPDESC_TOP values that got bucketed into OTHER but are
# genuinely distinct systems (not "unknown") -- used to build the disambiguation
# few-shots between OTHER and UNKNOWN OR OTHER.
OTHER_BUCKET_REAL_SYSTEMS = [
    "TRACTION CONTROL SYSTEM",
    "HYBRID PROPULSION SYSTEM",
    "TRAILER HITCHES",
    "INTERIOR LIGHTING",
    "FIRERELATED",
    "EQUIPMENT ADAPTIVE/MOBILITY",
]


def sample_readable(df, n, seed):
    pool = df[df["CDESCR"].str.len().between(150, 450)]
    if len(pool) < n:
        pool = df
    return pool.sample(n=min(n, len(pool)), random_state=seed)


def main():
    df = pd.read_parquet(
        DATA_DIR / "cmpl_clean.parquet",
        columns=["CMPLID", "MAKETXT", "MODELTXT", "YEARTXT", "CDESCR",
                 "COMPDESC_TOP", "COMPDESC_LABEL", "CRASH", "FIRE",
                 "INJURED", "DEATHS"],
    )

    with open(OUT_DIR / "label_schema.json") as f:
        schema = json.load(f)
    real_classes = [c["label"] for c in schema["classes"] if c["label"] != "OTHER"]

    def row_to_example(row):
        return {
            "cmplid": int(row["CMPLID"]),
            "make": row["MAKETXT"],
            "model": row["MODELTXT"],
            "year": row["YEARTXT"],
            "narrative": row["CDESCR"],
            "gold_COMPDESC_TOP": row["COMPDESC_TOP"],
            "gold_COMPDESC_LABEL": row["COMPDESC_LABEL"],
            "crash": row["CRASH"],
            "fire": row["FIRE"],
            "injured": row["INJURED"],
            "deaths": row["DEATHS"],
        }

    result = {"per_class": {}, "fire_routing_examples": [], "other_vs_unknown_examples": []}

    print("Sampling per-class candidates...")
    for cls in real_classes:
        sub = df[df["COMPDESC_LABEL"] == cls]
        sample = sample_readable(sub, 3, SEED)
        result["per_class"][cls] = [row_to_example(r) for _, r in sample.iterrows()]
        print(f"  {cls}: {len(sample)} candidates (pool={len(sub)})")

    print("\nSampling fire-routing candidates (mentions fire/flame/smoke but coded to a causal system)...")
    fire_mask = df["CDESCR"].str.contains(FIRE_RE, na=False)
    causal_systems = ["ENGINE AND ENGINE COOLING", "ELECTRICAL SYSTEM", "STRUCTURE", "AIR BAGS", "ENGINE"]
    fire_sub = df[fire_mask & df["COMPDESC_LABEL"].isin(causal_systems)]
    fire_sample = sample_readable(fire_sub, 6, SEED)
    result["fire_routing_examples"] = [row_to_example(r) for _, r in fire_sample.iterrows()]
    print(f"  {len(fire_sample)} candidates (pool={len(fire_sub)})")

    print("\nSampling OTHER-bucket real-system candidates (vs UNKNOWN OR OTHER)...")
    other_examples = []
    for sys_name in OTHER_BUCKET_REAL_SYSTEMS:
        sub = df[df["COMPDESC_TOP"] == sys_name]
        if len(sub) == 0:
            print(f"  {sys_name}: 0 rows, skipping")
            continue
        sample = sample_readable(sub, 1, SEED)
        other_examples.extend([row_to_example(r) for _, r in sample.iterrows()])
        print(f"  {sys_name}: {len(sample)} candidate (pool={len(sub)})")
    unknown_sub = df[df["COMPDESC_LABEL"] == "UNKNOWN OR OTHER"]
    unknown_sample = sample_readable(unknown_sub, 2, SEED)
    other_examples.extend([row_to_example(r) for _, r in unknown_sample.iterrows()])
    result["other_vs_unknown_examples"] = other_examples

    out_path = OUT_DIR / "fewshot_candidates.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
