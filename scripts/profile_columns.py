"""Step 2: profile candidate target fields + CDESCR narrative usability."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent.parent / "output"
OUT_DIR.mkdir(exist_ok=True)

pd.set_option("display.width", 140)
pd.set_option("display.max_rows", 60)


def main():
    df = pd.read_parquet(DATA_DIR / "cmpl.parquet")
    n = len(df)
    report = {"n_rows": n}
    print(f"Total rows: {n:,}")

    # ---- Narrative usability (CDESCR) ----
    cdescr = df["CDESCR"].fillna("")
    lengths = cdescr.str.len()
    report["cdescr_length"] = {
        "pct_blank": float((lengths == 0).mean() * 100),
        "pct_lt_100": float((lengths < 100).mean() * 100),
        "pct_lt_200": float((lengths < 200).mean() * 100),
        "pct_ge_200": float((lengths >= 200).mean() * 100),
        "mean": float(lengths.mean()),
        "median": float(lengths.median()),
        "p10": float(lengths.quantile(0.10)),
        "p90": float(lengths.quantile(0.90)),
        "max": int(lengths.max()),
    }
    print("\n--- CDESCR length distribution ---")
    print(json.dumps(report["cdescr_length"], indent=2))

    # common boilerplate marker NHTSA uses when narrative was redacted/confidential
    boilerplate_markers = [
        "TO WHOM IT MAY CONCERN",
        "SEE ATTACHED",
        "PLEASE SEE ATTACHED",
        "*TR",  # closing/handling codes used by NHTSA staff, often appended
        "CONSUMER STATED",
    ]
    marker_hits = {}
    for m in boilerplate_markers:
        marker_hits[m] = int(cdescr.str.contains(m, case=False, regex=False, na=False).sum())
    report["boilerplate_marker_hits"] = marker_hits
    print("\n--- Boilerplate marker hit counts ---")
    print(json.dumps(marker_hits, indent=2))

    # ---- Label completeness ----
    targets = ["COMPDESC", "CRASH", "FIRE", "INJURED", "DEATHS", "VEH_SPEED", "MEDICAL_ATTN", "VEHICLES_TOWED_YN"]
    completeness = {}
    for col in targets:
        s = df[col]
        if s.dtype == object:
            populated = s.astype(str).str.strip().ne("").mean() * 100
        else:
            populated = s.notna().mean() * 100
        completeness[col] = float(populated)
    report["label_completeness_pct"] = completeness
    print("\n--- Label completeness (% populated/non-blank) ---")
    print(json.dumps(completeness, indent=2))

    # Y/N field value distributions
    yn_dist = {}
    for col in ["CRASH", "FIRE", "MEDICAL_ATTN", "VEHICLES_TOWED_YN"]:
        yn_dist[col] = df[col].value_counts(dropna=False).to_dict()
    report["yn_value_counts"] = {k: {str(kk): int(vv) for kk, vv in v.items()} for k, v in yn_dist.items()}
    print("\n--- Y/N field value counts ---")
    for k, v in report["yn_value_counts"].items():
        print(k, v)

    # INJURED / DEATHS defaulting to 0
    report["injured_zero_pct"] = float((df["INJURED"].fillna(0) == 0).mean() * 100)
    report["deaths_zero_pct"] = float((df["DEATHS"].fillna(0) == 0).mean() * 100)
    report["injured_gt0_pct"] = float((df["INJURED"].fillna(0) > 0).mean() * 100)
    report["deaths_gt0_pct"] = float((df["DEATHS"].fillna(0) > 0).mean() * 100)
    print(f"\nINJURED==0: {report['injured_zero_pct']:.2f}%  DEATHS==0: {report['deaths_zero_pct']:.2f}%")
    print(f"INJURED>0: {report['injured_gt0_pct']:.3f}%  DEATHS>0: {report['deaths_gt0_pct']:.3f}%")

    # VEH_SPEED completeness / distribution
    report["veh_speed_nonnull_pct"] = float(df["VEH_SPEED"].notna().mean() * 100)
    report["veh_speed_nonzero_pct"] = float((df["VEH_SPEED"].fillna(0) > 0).mean() * 100)

    # ---- COMPDESC cardinality ----
    compdesc = df["COMPDESC"].astype(str)
    n_distinct_full = compdesc.nunique()
    top_level = compdesc.str.split(":").str[0]
    n_distinct_top = top_level.nunique()
    report["compdesc_cardinality"] = {
        "n_distinct_full_string": int(n_distinct_full),
        "n_distinct_top_level_category": int(n_distinct_top),
    }
    print(f"\nCOMPDESC distinct full strings: {n_distinct_full:,}")
    print(f"COMPDESC distinct TOP-LEVEL categories (before first ':'): {n_distinct_top:,}")

    top_level_counts = top_level.value_counts()
    top_level_counts.to_csv(OUT_DIR / "compdesc_top_level_counts.csv")
    print("\n--- Top-level COMPDESC categories (top 30) ---")
    print(top_level_counts.head(30))

    # cumulative coverage of top-level categories
    cum_pct = (top_level_counts.cumsum() / n * 100)
    n_cats_for_90 = int((cum_pct < 90).sum() + 1)
    report["compdesc_cardinality"]["n_top_level_cats_for_90pct_coverage"] = n_cats_for_90
    print(f"\n# top-level categories needed for 90% coverage: {n_cats_for_90} (of {n_distinct_top})")

    full_counts = compdesc.value_counts()
    full_counts.head(200).to_csv(OUT_DIR / "compdesc_full_string_top200_counts.csv")
    cum_pct_full = (full_counts.cumsum() / n * 100)
    n_full_for_90 = int((cum_pct_full < 90).sum() + 1)
    report["compdesc_cardinality"]["n_full_strings_for_90pct_coverage"] = n_full_for_90
    print(f"# full granular COMPDESC strings needed for 90% coverage: {n_full_for_90} (of {n_distinct_full:,})")

    # ---- Class balance ----
    report["class_balance"] = {
        "crash_Y_pct": float((df["CRASH"] == "Y").mean() * 100),
        "fire_Y_pct": float((df["FIRE"] == "Y").mean() * 100),
        "any_injury_pct": float((df["INJURED"].fillna(0) > 0).mean() * 100),
        "any_death_pct": float((df["DEATHS"].fillna(0) > 0).mean() * 100),
    }
    print("\n--- Base rates ---")
    print(json.dumps(report["class_balance"], indent=2))

    with open(OUT_DIR / "step2_profile_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {OUT_DIR / 'step2_profile_report.json'}")


if __name__ == "__main__":
    main()
