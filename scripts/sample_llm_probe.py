"""Step 5: draw a fresh (previously-unseen) 160-example sample for the zero-shot
LLM probe. Writes a BLIND input file (narrative + make/model/year only, no gold)
and a separate GOLD file (not to be read until after predictions are made)."""
import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent.parent / "output"

N = 160
SEED = 777  # different seed from prior samples to guarantee no overlap in judged rows
LABEL_SET_MIN_COUNT = 50


def main():
    df = pd.read_parquet(DATA_DIR / "cmpl.parquet")
    df["COMPDESC_TOP"] = df["COMPDESC"].astype(str).str.split(":").str[0]
    counts = df["COMPDESC_TOP"].value_counts()
    label_set = sorted(counts[counts >= LABEL_SET_MIN_COUNT].index.tolist())

    sample = df.sample(n=N, random_state=SEED).reset_index(drop=True)
    sample["idx"] = sample.index

    blind_cols = ["idx", "CMPLID", "MAKETXT", "MODELTXT", "YEARTXT", "CDESCR"]
    gold_cols = ["idx", "CMPLID", "COMPDESC_TOP", "COMPDESC", "CRASH", "FIRE", "INJURED", "DEATHS"]

    sample[blind_cols].to_json(OUT_DIR / "step5_llm_probe_BLIND_input.json", orient="records", indent=2)
    sample[gold_cols].to_json(OUT_DIR / "step5_llm_probe_GOLD_do_not_peek.json", orient="records", indent=2)

    with open(OUT_DIR / "step5_label_set.json", "w") as f:
        json.dump(label_set, f, indent=2)

    print(f"Wrote {N} blind examples. Label set ({len(label_set)} classes):")
    print(label_set)


if __name__ == "__main__":
    main()
