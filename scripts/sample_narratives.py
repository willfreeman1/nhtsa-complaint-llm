"""Draw random samples for (a) narrative-substantiveness read (n=30) and
(b) narrative-vs-coded-field qualitative agreement check (n=50), plus a
supplementary stratified sample oversampling rare positives (crash/fire/injury)
for extra signal on rare-field coding quality. Also runs the circularity check
(does the narrative literally contain the COMPDESC label text?).
"""
import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent.parent / "output"
OUT_DIR.mkdir(exist_ok=True)

SEED = 42
FIELDS = ["CMPLID", "MAKETXT", "MODELTXT", "YEARTXT", "COMPDESC", "CRASH", "FIRE",
          "INJURED", "DEATHS", "VEH_SPEED", "MEDICAL_ATTN", "VEHICLES_TOWED_YN", "CDESCR"]


def main():
    df = pd.read_parquet(DATA_DIR / "cmpl.parquet")

    # (a) 30 pure random narratives, for length/substantiveness read
    s30 = df.sample(n=30, random_state=SEED)[FIELDS].reset_index(drop=True)
    s30.to_json(OUT_DIR / "sample_30_narratives.json", orient="records", indent=2)

    # (b) 50 pure random complaints, for narrative-vs-coded-field agreement check
    s50 = df.sample(n=50, random_state=SEED + 1)[FIELDS].reset_index(drop=True)
    s50.to_json(OUT_DIR / "sample_50_qualitative.json", orient="records", indent=2)

    # (c) supplementary: 20 stratified toward rare positives (crash/fire/injury/death)
    rare_mask = (df["CRASH"] == "Y") | (df["FIRE"] == "Y") | (df["INJURED"].fillna(0) > 0) | (df["DEATHS"].fillna(0) > 0)
    s20_rare = df[rare_mask].sample(n=20, random_state=SEED + 2)[FIELDS].reset_index(drop=True)
    s20_rare.to_json(OUT_DIR / "sample_20_rare_positive.json", orient="records", indent=2)

    # (d) Circularity check: does CDESCR literally contain the COMPDESC label text (any segment)?
    def contains_label(row):
        cdescr = str(row["CDESCR"]).upper()
        segments = str(row["COMPDESC"]).upper().split(":")
        hits = [seg for seg in segments if len(seg) > 3 and seg in cdescr]
        return hits

    df["_label_hits"] = df.apply(contains_label, axis=1)
    df["_any_label_hit"] = df["_label_hits"].apply(lambda x: len(x) > 0)
    circularity_rate = df["_any_label_hit"].mean() * 100
    # also check just the top-level category segment specifically
    top_level = df["COMPDESC"].astype(str).str.split(":").str[0]
    top_hit = [
        (tl.upper() in str(cd).upper()) if len(tl) > 3 else False
        for tl, cd in zip(top_level, df["CDESCR"])
    ]
    top_level_circularity_rate = (pd.Series(top_hit).mean()) * 100

    circ_report = {
        "any_segment_of_compdesc_literally_in_cdescr_pct": float(circularity_rate),
        "top_level_compdesc_literally_in_cdescr_pct": float(top_level_circularity_rate),
    }
    with open(OUT_DIR / "circularity_check.json", "w") as f:
        json.dump(circ_report, f, indent=2)
    print(json.dumps(circ_report, indent=2))

    print("\nWrote sample files to output/:")
    for fn in ["sample_30_narratives.json", "sample_50_qualitative.json", "sample_20_rare_positive.json", "circularity_check.json"]:
        print(" -", fn)


if __name__ == "__main__":
    main()
