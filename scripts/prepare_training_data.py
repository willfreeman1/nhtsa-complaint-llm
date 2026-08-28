"""Rung-1 data prep: turn the raw parquet into a clean, labeled dataset ready for
teacher-model prompting and downstream fine-tuning.

Implements the fixes called out in output/NHTSA_CODING_GOTCHAS.md and
output/FINAL_REPORT.md section 4:
  1. Scope to PROD_TYPE == 'V' (vehicles) — drops the ~0.7% of child-seat rows where
     COMPDESC is literal free text instead of a controlled vocabulary (gotcha B8).
  2. Merge exact-duplicate-meaning top-level COMPDESC strings that are legacy/typo
     artifacts, not genuinely distinct categories (gotcha B3):
       - "ELECTRONIC STABILITY CONTROL (ESC)" -> "ELECTRONIC STABILITY CONTROL"
       - "COMMUNICATIONS" -> "COMMUNICATION"
     (Genuinely-confusable-but-distinct pairs like ENGINE vs POWER TRAIN are NOT
     merged — that ambiguity is real signal loss in the narrative, not a data bug,
     and collapsing them would throw away cases where the text *does* disambiguate.)
  3. Build the final ~30-35 class label list (COMPDESC_LABEL) covering ~99% of
     volume, bucketing the long tail into OTHER.
  4. Flag rows whose ODINO (incident id) is shared by >1 row (gotcha A1) — for those,
     CRASH/FIRE/INJURED/DEATHS may describe a sibling row's incident rather than this
     row's own CDESCR text, which is a known noise source for the "also extracted"
     binary/count fields (not for COMPDESC, which is legitimately one label per row).
"""
import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent.parent / "output"

MERGE_MAP = {
    "ELECTRONIC STABILITY CONTROL (ESC)": "ELECTRONIC STABILITY CONTROL",
    "COMMUNICATIONS": "COMMUNICATION",
}

# Coverage target for the "real" label set before bucketing the tail to OTHER.
TARGET_CUMULATIVE_COVERAGE = 0.99
MAX_CLASSES = 35


def main():
    print("Loading data/cmpl.parquet ...")
    df = pd.read_parquet(DATA_DIR / "cmpl.parquet")
    n_total = len(df)
    print(f"  {n_total:,} rows loaded")

    # --- 1. Scope to vehicles ---
    df = df[df["PROD_TYPE"] == "V"].copy()
    n_vehicles = len(df)
    print(f"  {n_vehicles:,} rows after PROD_TYPE=='V' filter "
          f"({n_vehicles / n_total:.1%} of total)")

    # --- 2. Top-level COMPDESC + merge known duplicate artifacts ---
    df["COMPDESC_TOP"] = df["COMPDESC"].astype(str).str.split(":").str[0].str.strip()
    df["COMPDESC_TOP"] = df["COMPDESC_TOP"].replace(MERGE_MAP)

    counts = df["COMPDESC_TOP"].value_counts()
    print(f"\n  {len(counts)} distinct top-level COMPDESC values after merge "
          f"(vs. 55 raw, before PROD_TYPE filter + merge)")

    # --- 3. Build final label list: top classes covering target coverage, capped ---
    cum = counts.cumsum() / counts.sum()
    n_keep = min(MAX_CLASSES, int((cum < TARGET_CUMULATIVE_COVERAGE).sum()) + 1)
    keep_classes = counts.index[:n_keep].tolist()
    coverage = cum.iloc[n_keep - 1]
    print(f"  Keeping top {n_keep} classes -> {coverage:.2%} coverage, rest -> OTHER")

    df["COMPDESC_LABEL"] = df["COMPDESC_TOP"].where(
        df["COMPDESC_TOP"].isin(keep_classes), "OTHER"
    )

    label_counts = df["COMPDESC_LABEL"].value_counts()
    label_schema = [
        {
            "label": label,
            "count": int(count),
            "pct_of_vehicle_rows": round(float(count) / n_vehicles, 6),
        }
        for label, count in label_counts.items()
    ]
    with open(OUT_DIR / "label_schema.json", "w") as f:
        json.dump(
            {
                "n_classes_incl_other": len(label_schema),
                "n_real_classes": n_keep,
                "coverage_before_other": round(float(coverage), 6),
                "merge_map_applied": MERGE_MAP,
                "classes": label_schema,
            },
            f,
            indent=2,
        )
    print(f"  Wrote {OUT_DIR / 'label_schema.json'}")

    # --- 4. ODINO multi-row flag (gotcha A1) ---
    odino_group_size = df.groupby("ODINO")["CMPLID"].transform("size")
    df["odino_group_size"] = odino_group_size
    df["odino_multi_row"] = odino_group_size > 1
    pct_multi = df["odino_multi_row"].mean()
    print(f"\n  {pct_multi:.1%} of vehicle rows share their ODINO with >=1 other row "
          f"(CRASH/FIRE/INJURED/DEATHS noise risk for these, per gotcha A1)")

    # --- Save cleaned dataset ---
    keep_cols = [
        "CMPLID", "ODINO", "MAKETXT", "MODELTXT", "YEARTXT", "CDESCR",
        "COMPDESC", "COMPDESC_TOP", "COMPDESC_LABEL",
        "CRASH", "FIRE", "INJURED", "DEATHS", "MEDICAL_ATTN", "VEHICLES_TOWED_YN",
        "odino_group_size", "odino_multi_row",
    ]
    out_path = DATA_DIR / "cmpl_clean.parquet"
    df[keep_cols].to_parquet(out_path, index=False)
    print(f"\n  Wrote {out_path} ({len(df):,} rows, {len(keep_cols)} columns)")

    summary = {
        "n_total_raw": n_total,
        "n_vehicle_rows": n_vehicles,
        "n_classes_incl_other": len(label_schema),
        "n_real_classes": n_keep,
        "coverage_before_other": round(float(coverage), 6),
        "pct_odino_multi_row": round(float(pct_multi), 6),
    }
    with open(OUT_DIR / "prepare_training_data_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Wrote {OUT_DIR / 'prepare_training_data_summary.json'}")
    print("\nSummary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
