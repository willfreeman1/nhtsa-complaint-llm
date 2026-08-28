"""Supplementary: re-score Step 5 component predictions after collapsing known
near-synonym / overlapping top-level COMPDESC categories, to separate 'model
didn't understand the narrative' from 'NHTSA's own top-level taxonomy has
redundant/overlapping buckets that are not distinguishable from prose alone'."""
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

OUT_DIR = Path(__file__).parent.parent / "output"

COLLAPSE = {
    "ENGINE": "ENGINE_GROUP",
    "ENGINE AND ENGINE COOLING": "ENGINE_GROUP",
    "POWER TRAIN": "ENGINE_GROUP",
    "FUEL/PROPULSION SYSTEM": "ENGINE_GROUP",
    "FUEL SYSTEM, GASOLINE": "ENGINE_GROUP",
    "FUEL SYSTEM, DIESEL": "ENGINE_GROUP",
    "FUEL SYSTEM, OTHER": "ENGINE_GROUP",
    "HYBRID PROPULSION SYSTEM": "ENGINE_GROUP",
    "VEHICLE SPEED CONTROL": "ENGINE_GROUP",
    "TRACTION CONTROL SYSTEM": "ENGINE_GROUP",
    "ELECTRONIC STABILITY CONTROL": "ENGINE_GROUP",
    "ELECTRONIC STABILITY CONTROL (ESC)": "ENGINE_GROUP",
    "SERVICE BRAKES": "BRAKES_GROUP",
    "SERVICE BRAKES, HYDRAULIC": "BRAKES_GROUP",
    "SERVICE BRAKES, AIR": "BRAKES_GROUP",
    "SERVICE BRAKES, ELECTRIC": "BRAKES_GROUP",
    "PARKING BRAKE": "BRAKES_GROUP",
    "STEERING": "STEER_SUSP_GROUP",
    "SUSPENSION": "STEER_SUSP_GROUP",
    "STRUCTURE": "STEER_SUSP_GROUP",
    "WHEELS": "STEER_SUSP_GROUP",
    "TIRES": "STEER_SUSP_GROUP",
    "ELECTRICAL SYSTEM": "ELECTRICAL_GROUP",
    "EXTERIOR LIGHTING": "ELECTRICAL_GROUP",
    "INTERIOR LIGHTING": "ELECTRICAL_GROUP",
    "COMMUNICATION": "ELECTRICAL_GROUP",
    "VISIBILITY": "VISIBILITY_GROUP",
    "VISIBILITY/WIPER": "VISIBILITY_GROUP",
    "AIR BAGS": "RESTRAINT_GROUP",
    "SEAT BELTS": "RESTRAINT_GROUP",
    "SEATS": "RESTRAINT_GROUP",
    "BACK OVER PREVENTION": "ADAS_UNKNOWN_GROUP",
    "UNKNOWN OR OTHER": "ADAS_UNKNOWN_GROUP",
}


def collapse(label):
    return COLLAPSE.get(str(label).strip(), str(label).strip().upper())


def main():
    preds = pd.DataFrame(json.load(open(OUT_DIR / "step5_llm_predictions.json")))
    gold = pd.DataFrame(json.load(open(OUT_DIR / "step5_llm_probe_GOLD_do_not_peek.json")))
    df = preds.merge(gold, on="idx", how="inner")

    df["pred_grp"] = df["pred_component"].apply(collapse)
    df["gold_grp"] = df["COMPDESC_TOP"].apply(collapse)

    exact = (df["pred_component"].str.upper().str.strip() == df["COMPDESC_TOP"].str.upper().str.strip())
    collapsed = (df["pred_grp"] == df["gold_grp"])

    print(f"Exact top-level accuracy:     {exact.mean():.3f}")
    print(f"Collapsed-taxonomy accuracy:  {collapsed.mean():.3f}  (n={len(df)})")
    print(f"Gap explained by taxonomy overlap alone: {collapsed.mean() - exact.mean():.3f} ({(collapsed & ~exact).sum()} rows)")

    remaining_errors = df.loc[~collapsed, ["idx", "CMPLID", "pred_component", "COMPDESC_TOP"]]
    remaining_errors.to_csv(OUT_DIR / "step5_remaining_errors_after_collapse.csv", index=False)
    print(f"\n{len(remaining_errors)} rows still wrong after collapsing near-synonym categories:")
    print(remaining_errors.to_string(index=False))

    with open(OUT_DIR / "step5_collapsed_score.json", "w") as f:
        json.dump({
            "exact_accuracy": float(exact.mean()),
            "collapsed_accuracy": float(collapsed.mean()),
            "n_explained_by_taxonomy_overlap": int((collapsed & ~exact).sum()),
            "n_remaining_real_errors": int((~collapsed).sum()),
        }, f, indent=2)


if __name__ == "__main__":
    main()
