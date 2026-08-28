"""Step 5 scoring: compare zero-shot LLM predictions (produced blind, reading
only narrative + make/model/year) against NHTSA gold, plus the same TF-IDF and
keyword baselines restricted to this exact 160-row sample for an apples-to-apples
comparison."""
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

OUT_DIR = Path(__file__).parent.parent / "output"


def main():
    preds = pd.DataFrame(json.load(open(OUT_DIR / "step5_llm_predictions.json")))
    gold = pd.DataFrame(json.load(open(OUT_DIR / "step5_llm_probe_GOLD_do_not_peek.json")))
    df = preds.merge(gold, on="idx", how="inner")
    assert len(df) == len(preds) == len(gold), "row mismatch"

    report = {"n": len(df)}

    # ---- Component (top-level) ----
    comp_correct = (df["pred_component"].str.upper().str.strip() == df["COMPDESC_TOP"].str.upper().str.strip())
    report["component_exact_accuracy"] = float(comp_correct.mean())
    report["component_macro_f1"] = float(f1_score(df["COMPDESC_TOP"], df["pred_component"], average="macro"))
    report["component_n_correct"] = int(comp_correct.sum())

    mismatches = df.loc[~comp_correct, ["idx", "CMPLID", "pred_component", "COMPDESC_TOP", "COMPDESC"]]
    mismatches.to_csv(OUT_DIR / "step5_component_mismatches.csv", index=False)

    # ---- CRASH ----
    report["crash_accuracy"] = float((df["pred_crash"] == df["CRASH"]).mean())
    report["crash_f1_Y"] = float(f1_score(df["CRASH"], df["pred_crash"], pos_label="Y", zero_division=0))

    # ---- FIRE ----
    report["fire_accuracy"] = float((df["pred_fire"] == df["FIRE"]).mean())
    report["fire_f1_Y"] = float(f1_score(df["FIRE"], df["pred_fire"], pos_label="Y", zero_division=0))

    # ---- INJURED / DEATHS (exact match on count) ----
    gold_inj = df["INJURED"].fillna(0).astype(int)
    gold_deaths = df["DEATHS"].fillna(0).astype(int)
    report["injured_exact_match_pct"] = float((df["pred_injured"] == gold_inj).mean())
    report["deaths_exact_match_pct"] = float((df["pred_deaths"] == gold_deaths).mean())
    report["injured_any_vs_any_agreement"] = float(((df["pred_injured"] > 0) == (gold_inj > 0)).mean())
    report["deaths_any_vs_any_agreement"] = float(((df["pred_deaths"] > 0) == (gold_deaths > 0)).mean())

    # ---- All-fields-exact (record-level) ----
    all_exact = comp_correct & (df["pred_crash"] == df["CRASH"]) & (df["pred_fire"] == df["FIRE"]) & \
                (df["pred_injured"] == gold_inj) & (df["pred_deaths"] == gold_deaths)
    report["all_fields_exact_pct"] = float(all_exact.mean())

    print(json.dumps(report, indent=2))
    with open(OUT_DIR / "step5_llm_probe_scored.json", "w") as f:
        json.dump(report, f, indent=2)

    # base rates in this sample, for context
    print("\nSample base rates:", {
        "crash_Y": int((df["CRASH"] == "Y").sum()),
        "fire_Y": int((df["FIRE"] == "Y").sum()),
        "any_injured": int((gold_inj > 0).sum()),
        "any_deaths": int((gold_deaths > 0).sum()),
    })
    print(f"\nWrote {OUT_DIR / 'step5_llm_probe_scored.json'} and step5_component_mismatches.csv ({len(mismatches)} mismatches)")


if __name__ == "__main__":
    main()
