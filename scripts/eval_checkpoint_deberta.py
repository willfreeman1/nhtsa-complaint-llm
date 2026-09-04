"""Standalone gold-set per-class breakdown + novel-category probe for an ALREADY
TRAINED Rung 2 DeBERTa checkpoint. Doesn't retrain -- loads the saved `final`
checkpoint and reuses train_deberta.py's evaluate_on_gold() logic exactly (so numbers
match the training-run summary), then additionally:
  - dumps a full sklearn classification_report broken out per class
  - runs the shared novel_category_probe narratives through the model and reports the
    top-3 predicted classes + softmax probability for each (qualitative check, not
    scored -- see novel_category_probe.py's docstring)

Usage:
    PYTHONPATH=scripts python3 scripts/eval_checkpoint_deberta.py \
        --checkpoint checkpoints/deberta_rung2_full_62k_plus_rare/final \
        --run-name full_62k_plus_rare
"""
import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import train_deberta as td
from novel_category_probe import NOVEL_CATEGORY_NARRATIVES

OUT_DIR = Path(__file__).parent.parent / "output"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--gold-file", default="gold_eval_set_v2.json")
    ap.add_argument("--run-name", default="full_62k_plus_rare")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint).to(device)
    model.eval()

    with open(OUT_DIR / args.gold_file) as f:
        gold = json.load(f)
    gold_rows = gold["rows"]
    td.GOLD_BY_ID = {r["cmplid"]: r for r in gold_rows}

    metrics, results = td.evaluate_on_gold(
        model, tokenizer, gold_rows, args.max_length, args.batch_size, device,
    )
    print(f"Gold: accuracy={metrics['accuracy']:.3f} macro_f1={metrics['macro_f1']:.3f} "
          f"(sanity check against the training-run summary JSON)")

    # NOTE: do NOT run primary through td.bucket_component() here -- that function
    # collapses all 9 rare categories (and any off-list answer) down to a generic
    # "OTHER" bucket, which is the 31-class scheme this script's model was NOT
    # trained/scored on (see the big comment at the top of train_deberta.py). LABELS
    # here is the full 40-class raw_component space, so primary should be used as-is.
    y_true = [
        r["pred"] if r["correct"] else td.GOLD_BY_ID[r["cmplid"]]["primary"]
        for r in results
    ]
    y_pred = [r["pred"] for r in results]
    report = classification_report(
        y_true, y_pred, labels=td.LABELS, output_dict=True, zero_division=0,
    )
    support_sorted = sorted(
        ((label, report[label]) for label in td.LABELS if label in report),
        key=lambda kv: -kv[1]["support"],
    )
    print("\nPer-class F1 (sorted by support, most-represented first):")
    for label, stats in support_sorted:
        print(f"  {label:<32} f1={stats['f1-score']:.3f}  support={int(stats['support'])}")

    # --- Novel-category probe ---
    novel_texts = [
        td.build_text(r["narrative"], r["make"], r["model"], r["year"])
        for r in NOVEL_CATEGORY_NARRATIVES
    ]
    with torch.no_grad():
        enc = tokenizer(
            novel_texts, truncation=True, padding=True, max_length=args.max_length,
            return_tensors="pt",
        ).to(device)
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        top_preds = logits.argmax(dim=-1).tolist()

    novel_results = []
    print("\nNovel-category probe (no ground truth -- qualitative check for forced fits):")
    for row, pred_id, prob_row in zip(NOVEL_CATEGORY_NARRATIVES, top_preds, probs.tolist()):
        pred_label = td.ID2LABEL[pred_id]
        top3_idx = sorted(range(len(prob_row)), key=lambda i: -prob_row[i])[:3]
        top3 = [[td.ID2LABEL[i], round(prob_row[i], 3)] for i in top3_idx]
        print(f"  [{row['cmplid']}] pred={pred_label}  top3={top3}")
        print(f"      narrative: {row['narrative'][:100]}...")
        novel_results.append({**row, "pred": pred_label, "top3": top3})

    out = {
        "run_name": args.run_name,
        "checkpoint": args.checkpoint,
        "gold_metrics": metrics,
        "per_class_report": report,
        "novel_category_results": novel_results,
    }
    out_path = OUT_DIR / f"rung2_percategory_and_novel_{args.run_name}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
