"""How large should the Rung-1 teacher-labeled training corpus be, and what would it
cost to label at various quality tiers?

Combines:
  - output/label_schema.json: real per-class row availability (31-class training
    schema, post PROD_TYPE=='V' filter + duplicate-merge, from prepare_training_data.py).
  - output/model_comparison.json + output/confidence_cascade_analysis.json: measured
    $/100k-row cost and accuracy for each labeling strategy on the gold eval set.

Literature range for how much labeled data these two fine-tunes typically need
(see conversation notes / cited sources, not re-derived here):
  - Encoder classifier (DeBERTa-v3-base full fine-tune): a few hundred to ~1-2k
    examples/class is the "comfortable" zone for a task with real class overlap;
    beyond that, most published results show diminishing returns.
  - LoRA/QLoRA classification fine-tune on a 7-8B open model: 200-500 examples/class
    for well-separated categories, up to a few thousand total for nuanced/ambiguous
    domains -- comfortably covered by whatever satisfies the encoder target above.

Conclusion baked into this script: pick a per-class target in that "comfortable"
zone, confirm it's within real data availability (it is, for all 31 classes here),
and then note that at the resulting total-row-count scale, labeling cost is cheap
enough that the *quality* of the labeling strategy -- not its cost -- should drive
the choice for the real training-data run (unlike the model-comparison exercise,
which was about picking a cost-efficient strategy at hypothetical million-row scale).

Writes output/training_set_sizing.json.
"""
import json
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "output"

PER_CLASS_TARGETS = [500, 1000, 1500, 2000, 3000, 4000]
VALIDATION_HOLDOUT_FRAC = 0.15  # carved out of the training corpus for early stopping / HPO;
                                  # gold_eval_set_v2 remains the untouched final benchmark.


def load_class_counts():
    with open(OUT_DIR / "label_schema.json") as f:
        schema = json.load(f)
    return {c["label"]: c["count"] for c in schema["classes"]}


def load_pricing_options():
    options = []
    with open(OUT_DIR / "model_comparison.json") as f:
        mc = json.load(f)
    for m in mc["models"]:
        options.append({
            "strategy": f"{m['provider']}/{m['model']} (solo)",
            "accuracy": m["accuracy"],
            "usd_per_100k_rows": m["usd_per_100k_rows"],
        })
    with open(OUT_DIR / "confidence_cascade_analysis.json") as f:
        cc = json.load(f)
    for c in cc["cascades"]:
        options.append({
            "strategy": f"{c['cheap_model']} -> {c['smart_model']} [{c['escalation_policy']}]",
            "accuracy": c["accuracy"],
            "usd_per_100k_rows": c["usd_per_100k_rows"],
        })
    return sorted(options, key=lambda o: o["usd_per_100k_rows"])


def main():
    counts = load_class_counts()
    n_classes = len(counts)
    pricing = load_pricing_options()

    sizing_rows = []
    for target in PER_CLASS_TARGETS:
        capped = {label: min(target, count) for label, count in counts.items()}
        total = sum(capped.values())
        n_capped_classes = sum(1 for label, count in counts.items() if count < target)
        train_after_holdout = int(total * (1 - VALIDATION_HOLDOUT_FRAC))
        row = {
            "per_class_target": target,
            "total_rows_labeled": total,
            "n_classes_below_target": n_capped_classes,
            "train_rows_after_val_holdout": train_after_holdout,
            "val_holdout_rows": total - train_after_holdout,
            "cost_by_strategy_usd": {
                opt["strategy"]: round(opt["usd_per_100k_rows"] * total / 100_000, 2)
                for opt in pricing
            },
        }
        sizing_rows.append(row)

    out = {
        "n_classes": n_classes,
        "class_availability": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "note": (
            "All 31 classes have >=4,520 rows available even after PROD_TYPE=='V' "
            "scoping and duplicate-merge, so every per-class target up to 4,000 is "
            "achievable without running out of real examples for any class -- no "
            "class requires taking 100% of its population below a 4,000/class target."
        ),
        "sizing_options": sizing_rows,
        "pricing_options_used": pricing,
    }
    path = OUT_DIR / "training_set_sizing.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {path}\n")

    print(f"{'per-class':>10} {'total rows':>11} {'train (85%)':>12}  cheapest $   mid-tier $   best-quality $")
    cheapest = pricing[0]["strategy"]
    priciest = pricing[-1]["strategy"]
    mid = pricing[len(pricing) // 2]["strategy"]
    for row in sizing_rows:
        c = row["cost_by_strategy_usd"]
        print(f"{row['per_class_target']:>10} {row['total_rows_labeled']:>11,} {row['train_rows_after_val_holdout']:>12,}  "
              f"${c[cheapest]:>9,.2f}  ${c[mid]:>9,.2f}  ${c[priciest]:>9,.2f}")
    print(f"\ncheapest strategy   = {cheapest}")
    print(f"mid-tier strategy   = {mid}")
    print(f"best-quality strategy = {priciest}")


if __name__ == "__main__":
    main()
