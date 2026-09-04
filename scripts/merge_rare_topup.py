"""Step 3: merge the rare-category topup teacher labels into the existing 62k
training-label set, verify no train/test-style leakage (zero duplicate cmplids), and
write a before/after summary for the 10 rare categories plus the 30 primary
categories (sanity check that the primaries are unaffected).

Usage:
    python scripts/merge_rare_topup.py
"""
import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent.parent / "output"

RARE_10 = [
    "FUEL SYSTEM, DIESEL", "CHILD SEAT", "INTERIOR LIGHTING", "SERVICE BRAKES, ELECTRIC",
    "TRACTION CONTROL SYSTEM", "EQUIPMENT ADAPTIVE/MOBILITY", "TRAILER HITCHES",
    "HYBRID PROPULSION SYSTEM", "FIRERELATED", "COMMUNICATION",
]


def main():
    full = pd.read_parquet(DATA_DIR / "training_labels_full_62k.parquet")
    topup = pd.read_parquet(DATA_DIR / "training_labels_rare_topup.parquet")
    print(f"Loaded full_62k: {len(full):,} rows; rare_topup: {len(topup):,} rows")

    assert list(full.columns) == list(topup.columns), "schema mismatch between the two label sets"

    full_ids = full["cmplid"].astype(str)
    topup_ids = topup["cmplid"].astype(str)
    dupe_ids = set(full_ids) & set(topup_ids)

    note = None
    if dupe_ids:
        n_before = len(topup)
        topup = topup[~topup_ids.isin(dupe_ids)]
        note = (
            f"Found {len(dupe_ids)} duplicate cmplid(s) across the two label sets; "
            f"dropped {n_before - len(topup)} rows from the topup half to avoid duplication."
        )
        print(f"WARNING: {note}")
    else:
        print("Confirmed zero duplicate cmplid values across the two label sets.")

    merged = pd.concat([full, topup], ignore_index=True)
    assert merged["cmplid"].astype(str).is_unique, "merged set still has duplicate cmplids"
    print(f"Merged: {len(merged):,} total rows")

    out_path = DATA_DIR / "training_labels_full_62k_plus_rare.parquet"
    merged.to_parquet(out_path, index=False)
    print(f"Wrote {out_path}")

    # Before/after for the 10 rare categories, keyed on raw_component (the teacher's
    # raw answer) since these are rare categories that bucket_component() would
    # otherwise collapse into OTHER.
    before_counts = full["raw_component"].value_counts()
    after_counts = merged["raw_component"].value_counts()
    rare_before_after = {
        cat: {"before": int(before_counts.get(cat, 0)), "after": int(after_counts.get(cat, 0))}
        for cat in RARE_10
    }

    # Primary-30 component_label distribution (component_label is the bucketed
    # 30-primary-classes-plus-OTHER training label) -- min/median/max, to show the
    # topup didn't skew the primary classes.
    top30_labels = [c for c in full["component_label"].dropna().unique() if c not in ("OTHER",)]
    # Restrict to the actual 30 primary classes via teacher_prompt for correctness.
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from teacher_prompt import TOP_30_CLASSES

    before_primary_counts = full["component_label"].value_counts()
    after_primary_counts = merged["component_label"].value_counts()
    primary_before = [int(before_primary_counts.get(c, 0)) for c in TOP_30_CLASSES]
    primary_after = [int(after_primary_counts.get(c, 0)) for c in TOP_30_CLASSES]

    summary = {
        "n_total_merged": len(merged),
        "n_full_62k": len(full),
        "n_topup_added": len(merged) - len(full),
        "duplicate_cmplids_found": len(dupe_ids),
        "duplicate_handling_note": note,
        "rare_categories_before_after": rare_before_after,
        "primary_30_classes_before": {
            "min": min(primary_before), "median": sorted(primary_before)[len(primary_before)//2],
            "max": max(primary_before),
        },
        "primary_30_classes_after": {
            "min": min(primary_after), "median": sorted(primary_after)[len(primary_after)//2],
            "max": max(primary_after),
        },
        "output_parquet": out_path.name,
    }
    summary_path = OUT_DIR / "training_labels_summary_full_62k_plus_rare.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")

    print("\n--- Rare-category before/after (raw_component match) ---")
    print(f"{'Category':32s} {'Before':>8s} {'After':>8s}")
    for cat in RARE_10:
        b, a = rare_before_after[cat]["before"], rare_before_after[cat]["after"]
        print(f"{cat:32s} {b:>8,} {a:>8,}")

    print("\n--- Primary-30 classes (component_label), for reference ---")
    print(f"Before: min={summary['primary_30_classes_before']['min']:,} "
          f"median={summary['primary_30_classes_before']['median']:,} "
          f"max={summary['primary_30_classes_before']['max']:,}")
    print(f"After:  min={summary['primary_30_classes_after']['min']:,} "
          f"median={summary['primary_30_classes_after']['median']:,} "
          f"max={summary['primary_30_classes_after']['max']:,}")


if __name__ == "__main__":
    main()
