"""Step 4a: regex/keyword baselines for CRASH and FIRE, scored against NHTSA's
own coded fields (treated as gold, with the Step-3 caveats noted separately)."""
import json
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent.parent / "output"

FIRE_PATTERN = re.compile(r"\b(fire|flames?|smoke|burn(?:ed|ing|t)?)\b", re.IGNORECASE)
CRASH_PATTERN = re.compile(r"\b(crash(?:ed)?|collision|collided|hit|accident|wreck(?:ed)?)\b", re.IGNORECASE)


def main():
    df = __import__("pandas").read_parquet(DATA_DIR / "cmpl.parquet", columns=["CDESCR", "CRASH", "FIRE"])
    n = len(df)
    text = df["CDESCR"].fillna("")

    pred_fire = text.str.contains(FIRE_PATTERN).map({True: "Y", False: "N"})
    pred_crash = text.str.contains(CRASH_PATTERN).map({True: "Y", False: "N"})

    gold_fire = df["FIRE"]
    gold_crash = df["CRASH"]

    report = {}
    for name, gold, pred in [("FIRE", gold_fire, pred_fire), ("CRASH", gold_crash, pred_crash)]:
        acc = float((gold == pred).mean())
        f1 = f1_score(gold, pred, pos_label="Y")
        prec = precision_score(gold, pred, pos_label="Y")
        rec = recall_score(gold, pred, pos_label="Y")
        report[name] = {
            "accuracy": acc,
            "f1_pos_Y": float(f1),
            "precision_pos_Y": float(prec),
            "recall_pos_Y": float(rec),
            "base_rate_Y_pct": float((gold == "Y").mean() * 100),
            "pred_rate_Y_pct": float((pred == "Y").mean() * 100),
        }
        print(f"\n=== {name} keyword baseline (n={n:,}) ===")
        print(json.dumps(report[name], indent=2))
        print(classification_report(gold, pred, digits=3))

    with open(OUT_DIR / "step4_keyword_baseline.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {OUT_DIR / 'step4_keyword_baseline.json'}")


if __name__ == "__main__":
    main()
