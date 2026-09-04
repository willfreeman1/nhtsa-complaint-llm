"""Rung-2: fine-tune a DeBERTa-v3 encoder classifier on the Rung-1 (teacher-labeled)
training set, then score it on the same held-out gold set used for Rung 1/3 so all
three rungs are directly comparable.

Label universe is the full 40-class `raw_component` space (30 primary + 10 rare, see
teacher_prompt.ALL_VALID_COMPONENTS) -- the SAME space Rung 3 trains on. An earlier
version of this script trained on the bucketed 31-class `component_label` instead
(30 primary + a single catch-all "OTHER" standing in for all 10 rare categories). That
gave Rung 2 a strictly easier task than Rung 1/3 (fewer, coarser classes) *and*, worse,
an inconsistent scoring advantage: the eval bucketed the gold set's accepted answers
before comparing, so a truth of e.g. "HYBRID PROPULSION SYSTEM" only required Rung 2 to
guess generic "OTHER" for credit, while Rung 1/3 had to name the exact rare category.
Training on raw_component with matching eval logic (see evaluate_on_gold below) closes
both gaps so all three rungs are solving -- and are scored on -- the identical task.
`gold_COMPDESC_LABEL` is carried through only as a reference to raw NHTSA COMPDESC,
never used as a training signal, to avoid quietly re-deriving Rung 1's own known
NHTSA-vs-teacher disagreements as ground truth.

Class weights (inverse-frequency, "balanced") correct for real-world volume skew that
survives even in the capped-at-2000/class training sample (see
NHTSA_CODING_GOTCHAS.md) -- without them, a fixed softmax head tends to under-call rare
classes it saw fewer weight-updates for. This matters more now that 10 of the 40
classes are genuinely rare in the training sample (see module docstring above).

Gold-set scoring mirrors eval_models_on_gold.py / train_llm_lora.py exactly: score
against the model's raw (un-bucketed) prediction first -- the gold `accept` list can
itself list a rare category -- falling back to the bucketed label only if the raw
prediction isn't itself accepted.

Usage:
    # Smoke test: tiny sample, 1 epoch, proves the pipeline runs end-to-end.
    python scripts/train_deberta.py --train-parquet data/training_labels_smoketest.parquet --run-name smoketest --epochs 1 --batch-size 4

    # Full run (once the cascade-labeled parquet lands), ideally on a rented GPU:
    python scripts/train_deberta.py --train-parquet data/training_labels_full_62k.parquet --run-name full_62k
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from teacher_prompt import ALL_VALID_COMPONENTS, bucket_component

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent.parent / "output"
CKPT_DIR = Path(__file__).parent.parent / "checkpoints"

LABELS = sorted(ALL_VALID_COMPONENTS)  # full 40-class raw_component space, same as Rung 3
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

# Populated in main(), read by evaluate_on_gold()'s macro-F1 fallback.
GOLD_BY_ID = {}


def build_text(narrative, make="", model="", year=""):
    """Same shape the teacher LLM saw (build_user_message in teacher_prompt.py), so
    the encoder gets access to the same make/model/year context."""
    return f"Make/Model/Year: {make} {model} {year}\nNarrative: {narrative}"


def build_texts(df):
    """Vectorized version of build_text for a whole DataFrame (row-by-row string
    formatting is too slow at 62k+ rows)."""
    return (
        "Make/Model/Year: " + df["make"].astype(str) + " " + df["model"].astype(str)
        + " " + df["year"].astype(str) + "\nNarrative: " + df["narrative"].astype(str)
    ).tolist()


class ComplaintDataset(Dataset):
    def __init__(self, texts, label_ids, tokenizer, max_length):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding="max_length", max_length=max_length,
        )
        self.label_ids = list(label_ids)

    def __len__(self):
        return len(self.label_ids)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.label_ids[idx], dtype=torch.long)
        return item


class WeightedTrainer(Trainer):
    """Standard HF Trainer, but with a class-weighted cross-entropy loss instead of
    the default unweighted one."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = (
            self.class_weights.to(device=logits.device, dtype=logits.dtype)
            if self.class_weights is not None else None
        )
        loss = nn.functional.cross_entropy(logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float((preds == labels).mean()),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
    }


def load_training_frame(path):
    df = pd.read_parquet(path)
    n_total = len(df)
    df = df[df["raw_component"].isin(ALL_VALID_COMPONENTS) & df["error"].isna()].copy()
    n_dropped = n_total - len(df)
    if n_dropped:
        print(f"Dropped {n_dropped:,}/{n_total:,} rows (parse/API errors or off-schema teacher answers)")
    return df


def split_train_val(df, val_frac, seed):
    try:
        train_df, val_df = train_test_split(
            df, test_size=val_frac, random_state=seed, stratify=df["raw_component"],
        )
    except ValueError as e:
        print(f"Stratified split failed ({e}); falling back to a plain random split "
              f"(expected on tiny smoke-test samples where classes have <2 rows).")
        train_df, val_df = train_test_split(df, test_size=val_frac, random_state=seed)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def build_class_weights(train_df):
    present = sorted(train_df["raw_component"].unique())
    present_ids = np.array([LABEL2ID[c] for c in present])
    y = train_df["raw_component"].map(LABEL2ID).values
    balanced = compute_class_weight("balanced", classes=present_ids, y=y)
    weights = np.ones(len(LABELS), dtype=np.float32)
    for cid, w in zip(present_ids, balanced):
        weights[cid] = w
    return torch.tensor(weights, dtype=torch.float32)


def evaluate_on_gold(model, tokenizer, gold_rows, max_length, batch_size, device):
    model.eval()
    texts = [build_text(r["narrative"], r["make"], r["model"], r["year"]) for r in gold_rows]
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(
                batch, truncation=True, padding=True, max_length=max_length, return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            preds.extend(logits.argmax(dim=-1).tolist())
    pred_labels = [ID2LABEL[p] for p in preds]

    results = []
    for row, pred in zip(gold_rows, pred_labels):
        # Same precedence as eval_models_on_gold.py / train_llm_lora.py: score against
        # the raw (un-bucketed) prediction first -- accept can itself list a rare
        # category -- falling back to the bucketed label only if the raw guess isn't
        # itself accepted.
        accept = row["accept"]
        pred_bucket = bucket_component(pred)
        correct = (pred in accept) or (pred_bucket in accept)
        scored_pred = pred if pred in accept else pred_bucket
        results.append({
            "cmplid": row["cmplid"], "pred": scored_pred, "raw_pred": pred, "correct": correct,
            "is_hard": row["is_hard"], "truth_source": row["truth_source"],
        })

    def acc(subset):
        return (sum(r["correct"] for r in subset) / len(subset)) if subset else None

    hard = [r for r in results if r["is_hard"]]
    easy = [r for r in results if not r["is_hard"]]
    # For macro-F1, an accepted-but-not-primary prediction counts as that label -- see
    # eval_models_on_gold.py's identical rationale (scoring ambiguous rows' second
    # valid answer as a miss would just measure coin flips).
    y_true = [
        r["pred"] if r["correct"] else bucket_component(GOLD_BY_ID[r["cmplid"]]["primary"])
        for r in results
    ]
    y_pred = [r["pred"] for r in results]

    metrics = {
        "n": len(results),
        "accuracy": acc(results),
        "accuracy_hard_subset": acc(hard),
        "accuracy_dual_agreement_subset": acc(easy),
        # NOTE: deliberately no `labels=LABELS` here. Passing the full 40-class label
        # universe forces every class absent from the 498-row gold set (9 of them, e.g.
        # TRAILER HITCHES, FIRERELATED -- genuinely rare enough to have zero gold rows)
        # to contribute a guaranteed 0 to the macro average, regardless of model
        # quality. eval_models_on_gold.py (Rung 1) and train_llm_lora.py (Rung 3) both
        # call f1_score without `labels=`, i.e. only over classes that actually appear
        # in y_true/y_pred -- matching that convention here is required for the
        # macro-F1 numbers across all three rungs to be comparable at all. An earlier
        # version of this line passed `labels=LABELS` and made Rung 2 look far weaker
        # on macro-F1 (0.642) than it actually was (0.825) purely from this scoring
        # mismatch, not a real accuracy gap -- see WRITEUP.md section 4.
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    return metrics, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-parquet", required=True,
                     help="Cascade-labeled parquet from run_batch_cascade_labeling.py")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--model-name", default="microsoft/deberta-v3-base")
    ap.add_argument("--gold-file", default="gold_eval_set_v2.json")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--no-class-weights", action="store_true")
    args = ap.parse_args()

    run_name = args.run_name or Path(args.train_parquet).stem
    out_dir = CKPT_DIR / f"deberta_rung2_{run_name}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("Device: cpu -- no GPU on this machine; run the full 62k training on a "
              "rented GPU per PROJECT_PLAN.md. This is fine for a smoke test.")
    else:
        print("Device: cuda")

    df = load_training_frame(args.train_parquet)
    print(f"Loaded {len(df):,} labeled rows across {df['raw_component'].nunique()} classes")

    train_df, val_df = split_train_val(df, args.val_frac, args.seed)
    print(f"Train/val split: {len(train_df):,} / {len(val_df):,}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID,
        dtype=torch.float32,  # keep weights fp32; Trainer's own fp16 flag handles CUDA mixed precision
    ).to(device)

    train_ds = ComplaintDataset(
        build_texts(train_df), train_df["raw_component"].map(LABEL2ID).values,
        tokenizer, args.max_length,
    )
    val_ds = ComplaintDataset(
        build_texts(val_df), val_df["raw_component"].map(LABEL2ID).values,
        tokenizer, args.max_length,
    ) if len(val_df) else None

    class_weights = None if args.no_class_weights else build_class_weights(train_df)

    steps_per_epoch = max(1, -(-len(train_ds) // args.batch_size))  # ceil div
    warmup_steps = max(1, int(0.1 * steps_per_epoch * args.epochs))

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        eval_strategy="epoch" if val_ds is not None else "no",
        save_strategy="epoch" if val_ds is not None else "no",
        save_total_limit=2,
        load_best_model_at_end=val_ds is not None,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=device == "cuda",
        report_to="none",
        seed=args.seed,
        logging_steps=25,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics if val_ds is not None else None,
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] if val_ds is not None else [],
    )

    trainer.train()

    val_metrics = trainer.evaluate() if val_ds is not None else {}
    print(f"Validation metrics: {val_metrics}")

    global GOLD_BY_ID
    with open(OUT_DIR / args.gold_file) as f:
        gold = json.load(f)
    gold_rows = gold["rows"]
    GOLD_BY_ID = {r["cmplid"]: r for r in gold_rows}

    gold_metrics, gold_results = evaluate_on_gold(
        model, tokenizer, gold_rows, args.max_length, args.eval_batch_size, device,
    )
    print(f"\n=== Rung 2 (DeBERTa, run={run_name}) on gold set (n={gold_metrics['n']}) ===")
    print(f"  accuracy overall           {gold_metrics['accuracy']:.1%}")
    print(f"  accuracy hard subset       {gold_metrics['accuracy_hard_subset']:.1%}")
    print(f"  accuracy dual-agreement    {gold_metrics['accuracy_dual_agreement_subset']:.1%}")
    print(f"  macro-F1                   {gold_metrics['macro_f1']:.3f}")

    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Saved model to {final_dir}")

    summary_path = OUT_DIR / f"rung2_deberta_eval_{run_name}.json"
    with open(summary_path, "w") as f:
        json.dump({
            "run_name": run_name,
            "model_name": args.model_name,
            "n_train": len(train_df),
            "n_val": len(val_df),
            "class_weighted": class_weights is not None,
            "hyperparams": {
                "epochs": args.epochs, "batch_size": args.batch_size,
                "lr": args.lr, "max_length": args.max_length,
            },
            "validation_metrics": val_metrics,
            "gold_metrics": gold_metrics,
            "checkpoint_dir": str(final_dir),
        }, f, indent=2)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
