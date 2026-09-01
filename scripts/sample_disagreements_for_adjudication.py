"""Pull a stratified sample of teacher/gold disagreements from the 500-row pilot
for manual read-through, joined with the actual narrative text.

Oversamples high-confidence disagreements relative to their share (these are the
most informative/concerning -- a confident wrong answer matters more than an
uncertain one) while still covering medium and low.
"""
import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent.parent / "output"
SEED = 42

TARGET_PER_BUCKET = {"high": 15, "medium": 20, "low": 5}


def main():
    with open(OUT_DIR / "teacher_labels_anthropic_500.json") as f:
        pilot = json.load(f)["results"]

    disagreements = [r for r in pilot if r["pred_component_bucketed"] != r["gold_COMPDESC_LABEL"]]

    df = pd.DataFrame(disagreements)
    df["confidence"] = df["raw_response"].apply(lambda x: x["confidence"])

    sampled = []
    for bucket, n in TARGET_PER_BUCKET.items():
        sub = df[df["confidence"] == bucket]
        take = min(n, len(sub))
        sampled.append(sub.sample(n=take, random_state=SEED))
    sample_df = pd.concat(sampled).reset_index(drop=True)

    cmplids = [str(r["cmplid"]) for r in sample_df.to_dict("records")]
    narratives = pd.read_parquet(DATA_DIR / "cmpl.parquet", columns=["CMPLID", "CDESCR", "MAKETXT", "MODELTXT", "YEARTXT"])
    narratives = narratives[narratives["CMPLID"].isin(cmplids)].set_index("CMPLID")

    out = []
    for r in sample_df.to_dict("records"):
        cid = str(r["cmplid"])
        out.append({
            "cmplid": r["cmplid"],
            "make": narratives.loc[cid, "MAKETXT"],
            "model": narratives.loc[cid, "MODELTXT"],
            "year": narratives.loc[cid, "YEARTXT"],
            "narrative": narratives.loc[cid, "CDESCR"],
            "gold": r["gold_COMPDESC_LABEL"],
            "teacher_pred": r["pred_component_bucketed"],
            "teacher_raw_component": r["raw_response"]["component"],
            "confidence": r["raw_response"]["confidence"],
            "rationale": r["raw_response"]["rationale"],
        })

    with open(OUT_DIR / "disagreements_sample_for_adjudication.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"Sampled {len(out)} disagreements ({dict(sample_df['confidence'].value_counts())}) "
          f"out of {len(disagreements)} total.")
    print(f"Wrote {OUT_DIR / 'disagreements_sample_for_adjudication.json'}")


if __name__ == "__main__":
    main()
