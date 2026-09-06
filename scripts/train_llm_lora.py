"""Rung-3: LoRA/QLoRA fine-tune an open-weight instruct LLM to distill the cascade
teacher's labeling behavior, then score it on the same held-out gold set used for
Rungs 1/2 so all three rungs are directly comparable.

Design choices worth calling out:
- Trains on `raw_component` (the teacher's literal answer -- one of the 30 primary +
  10 rare + "UNKNOWN OR OTHER" = 41 valid strings in teacher_prompt.ALL_VALID_COMPONENTS),
  not the bucketed 31-class `component_label`. Unlike Rung 2's fixed-size softmax head,
  a generative model can legitimately output a rare category verbatim -- collapsing
  training data down to OTHER before fine-tuning would erase exactly the flexibility
  advantage Rung 3 is supposed to have over Rung 2 (see PROJECT_PLAN.md). Off-list rows
  (raw_component not in ALL_VALID_COMPONENTS -- teacher hallucination noise) are
  dropped from training rather than taught to the student.
- Scoring mirrors eval_models_on_gold.py exactly: score against the raw answer first
  (gold's `accept` list can itself contain a rare category), falling back to the
  bucketed 31-class label only if the raw answer isn't itself accepted.
- The student's system prompt drops the few-shot block that Rung 1's teacher prompt
  needs -- that's the whole point of fine-tuning: label-schema knowledge gets baked
  into the weights via many real training examples instead of a handful of in-context
  ones. This also meaningfully shrinks the prompt (cheaper/faster at inference).
- QLoRA (4-bit base weights + LoRA adapters) hand-rolled on plain
  transformers + peft + bitsandbytes -- no trl. A custom Dataset/Trainer (same pattern
  as train_deberta.py's WeightedTrainer) keeps this consistent with Rung 2 and avoids
  one more dependency on a stack that's already needed a few version-compat fixes.

Usage:
    # Smoke test on CPU with a small stand-in model, no GPU/bitsandbytes needed:
    python scripts/train_llm_lora.py --train-parquet data/training_labels_smoketest.parquet --run-name smoketest --model-name Qwen/Qwen2.5-0.5B-Instruct --no-quantization --epochs 1 --batch-size 2 --eval-limit 10

    # Full run on the rented GPU:
    python scripts/train_llm_lora.py --train-parquet data/training_labels_full_62k.parquet --run-name full_62k
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from teacher_prompt import ALL_VALID_COMPONENTS, CLASS_DEFINITIONS, RARE_CATEGORIES, build_user_message, bucket_component

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent.parent / "output"
CKPT_DIR = Path(__file__).parent.parent / "checkpoints"

# Populated in main(), read by evaluate_on_gold()'s macro-F1 fallback.
GOLD_BY_ID = {}


def build_student_system_prompt():
    """Deliberately much shorter than Rung 1's teacher prompt (build_system_prompt in
    teacher_prompt.py): that prompt's ~1,400 tokens of per-class descriptions exist
    because a zero-shot teacher has no other way to learn category boundaries. Here,
    the student sees the same fixed system prompt on every one of tens of thousands of
    training examples, so those descriptions would just be ~1,400 tokens of dead
    weight repeated on every row -- the category *boundaries* get learned from the
    labeled (narrative, category) pairs themselves during fine-tuning, not from prose
    re-read every time. We keep the exact category NAMES (so the model has a closed,
    canonical vocabulary to anchor on and doesn't invent near-duplicate spellings) but
    drop the descriptions. This is the same prompt used at eval/inference time, so
    train/eval stay consistent."""
    names = "\n".join(f"- {name}" for name in list(CLASS_DEFINITIONS) + list(RARE_CATEGORIES))
    return (
        "You are labeling NHTSA vehicle owner complaint narratives. For each complaint, "
        "read the narrative (and make/model/year if given) and return a single JSON "
        "object with these fields:\n\n"
        '- "component": the vehicle system/component at fault. Output EXACTLY one of '
        "the category names below -- never invent a new name. If nothing fits, use "
        '"UNKNOWN OR OTHER".\n'
        '- "crash": "Y" or "N" -- does THIS narrative\'s own text describe a crash/collision?\n'
        '- "fire": "Y" or "N" -- does THIS narrative\'s own text describe an actual vehicle '
        "fire (flames, something burning)? Smoke/glowing from normal friction or airbag "
        "deployment is NOT a fire.\n"
        '- "injured": integer count of people injured per THIS narrative\'s own text (0 if none).\n'
        '- "deaths": integer count of deaths per THIS narrative\'s own text (0 if none).\n\n'
        "Output ONLY the JSON object, no other text.\n\n"
        "## Valid categories\n\n" + names
    )


def build_target_json(row):
    return json.dumps({
        "component": row["raw_component"],
        "crash": row["crash"],
        "fire": row["fire"],
        "injured": int(row["injured"]),
        "deaths": int(row["deaths"]),
    })


def load_training_frame(path):
    df = pd.read_parquet(path)
    n_total = len(df)
    df = df[
        df["raw_component"].isin(ALL_VALID_COMPONENTS)
        & df["error"].isna()
        & df["crash"].isin(["Y", "N"])
        & df["fire"].isin(["Y", "N"])
        & df["injured"].notna()
        & df["deaths"].notna()
    ].copy()
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


class ChatSFTDataset(Dataset):
    """Tokenizes (system, user, assistant-JSON) triples and masks the loss on
    everything except the assistant's answer, so the model is only ever trained to
    predict the JSON, never to reproduce the (fixed) prompt it was given."""

    def __init__(self, rows, system_prompt, tokenizer, max_length):
        self.examples = []
        for row in rows:
            user_msg = build_user_message(
                narrative=row["narrative"], make=row["make"], model=row["model"], year=row["year"],
            )
            target = build_target_json(row)
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_msg}],
                add_generation_prompt=True, tokenize=False,
            )
            full_text = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt}, {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": target},
                ],
                add_generation_prompt=False, tokenize=False,
            )
            # tokenize=False + a separate tokenizer() call (rather than tokenize=True
            # directly) avoids a chat-template/tokenizer version mismatch that returns
            # a raw tokenizers.Encoding instead of a plain list of ids in some
            # transformers/tokenizers version combinations.
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
            full_ids = full_ids[:max_length]
            labels = list(full_ids)
            mask_len = min(len(prompt_ids), len(full_ids))
            for i in range(mask_len):
                labels[i] = -100
            self.examples.append((full_ids, labels))

        n_fully_masked = sum(1 for _, labels in self.examples if all(l == -100 for l in labels))
        if n_fully_masked:
            frac = n_fully_masked / len(self.examples)
            msg = (f"{n_fully_masked}/{len(self.examples)} ({frac:.0%}) examples have every token "
                   f"masked -- max_length ({max_length}) is too small for the system prompt "
                   f"(the assistant answer got fully truncated away, so these rows train on nothing).")
            if frac > 0.05:
                raise ValueError(msg)
            print(f"WARNING: {msg}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_ids, labels = self.examples[idx]
        return {"input_ids": input_ids, "labels": labels}


class PadCollator:
    """Right-pads a batch of variable-length tokenized examples. Labels pad with
    -100 so padding never contributes to the loss."""

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids, attn, labels = [], [], []
        for x in batch:
            pad_n = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [self.pad_token_id] * pad_n)
            attn.append([1] * len(x["input_ids"]) + [0] * pad_n)
            labels.append(x["labels"] + [-100] * pad_n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def parse_json_response(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"No JSON object found: {text[:200]!r}")
    return json.loads(m.group(0))


@torch.no_grad()
def evaluate_on_gold(model, tokenizer, gold_rows, system_prompt, max_new_tokens, device):
    model.eval()
    results = []
    for row in gold_rows:
        user_msg = build_user_message(
            narrative=row["narrative"], make=row["make"], model=row["model"], year=row["year"],
        )
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_msg}],
            add_generation_prompt=True, tokenize=False,
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        out = model.generate(
            prompt_ids, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        gen_text = tokenizer.decode(out[0][prompt_ids.shape[1]:], skip_special_tokens=True)
        try:
            parsed = parse_json_response(gen_text)
            raw_component = parsed.get("component", "")
        except Exception:  # noqa: BLE001
            raw_component = None

        # Same precedence as eval_models_on_gold.py: score against the raw answer
        # first (accept can itself list a rare category), bucketed 31-class label
        # as fallback only.
        accept = row["accept"]
        pred_bucket = bucket_component(raw_component) if raw_component else None
        correct = bool(raw_component) and (raw_component in accept or pred_bucket in accept)
        scored_pred = raw_component if (raw_component in accept) else pred_bucket

        results.append({
            "cmplid": row["cmplid"], "raw_component": raw_component, "pred": scored_pred,
            "correct": correct, "is_hard": row["is_hard"], "truth_source": row["truth_source"],
            "gen_text_preview": gen_text[:200],
        })

    def acc(subset):
        return (sum(r["correct"] for r in subset) / len(subset)) if subset else None

    hard = [r for r in results if r["is_hard"]]
    easy = [r for r in results if not r["is_hard"]]
    y_true = [
        r["pred"] if r["correct"] else bucket_component(GOLD_BY_ID[r["cmplid"]]["primary"])
        for r in results
    ]
    y_pred = [r["pred"] or "PARSE_FAILURE" for r in results]

    metrics = {
        "n": len(results),
        "n_parse_failures": sum(1 for r in results if r["raw_component"] is None),
        "accuracy": acc(results),
        "accuracy_hard_subset": acc(hard),
        "accuracy_dual_agreement_subset": acc(easy),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    return metrics, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-parquet", required=True,
                     help="Cascade-labeled parquet from run_batch_cascade_labeling.py")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--gold-file", default="gold_eval_set_v3.json")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum-steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-length", type=int, default=1280,
                     help="Must comfortably exceed the student system prompt (~450 tokens for the "
                          "name-only category list) plus narrative (p99 ~450 tokens) plus JSON answer, "
                          "or training examples get truncated/masked (silent -- watch for train_loss=0).")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--no-quantization", action="store_true",
                     help="Skip 4-bit bnb loading (needed for CPU smoke tests; bitsandbytes needs CUDA).")
    ap.add_argument("--eval-limit", type=int, default=None,
                     help="Score only the first N gold rows (smoke test).")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    run_name = args.run_name or Path(args.train_parquet).stem
    out_dir = CKPT_DIR / f"llm_rung3_{run_name}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_quant = (not args.no_quantization) and device == "cuda"
    print(f"Device: {device}, 4-bit quantization: {use_quant}")
    if device != "cuda":
        print("No GPU on this machine -- this is fine for a smoke test with a small "
              "stand-in model, but the real run needs the rented GPU per PROJECT_PLAN.md.")

    df = load_training_frame(args.train_parquet)
    print(f"Loaded {len(df):,} labeled rows, {df['raw_component'].nunique()} distinct raw component values")
    train_df, val_df = split_train_val(df, args.val_frac, args.seed)
    print(f"Train/val split: {len(train_df):,} / {len(val_df):,}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    system_prompt = build_student_system_prompt()
    print(f"Student system prompt: ~{len(system_prompt)//4:,} tokens (no few-shot block)")

    quant_config = None
    if use_quant:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quant_config,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    if use_quant:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    if device != "cuda":
        model = model.to(device)
    else:
        # Activations for a 7B model at any reasonable batch size are huge without
        # this -- trades ~20-30% more compute for a large cut in activation memory.
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    train_ds = ChatSFTDataset(train_df.to_dict("records"), system_prompt, tokenizer, args.max_length)
    val_ds = ChatSFTDataset(val_df.to_dict("records"), system_prompt, tokenizer, args.max_length) if len(val_df) else None
    collator = PadCollator(tokenizer.pad_token_id)

    steps_per_epoch = max(1, -(-len(train_ds) // (args.batch_size * args.grad_accum_steps)))
    warmup_steps = max(1, int(0.03 * steps_per_epoch * args.epochs))

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.lr,
        weight_decay=0.0,
        warmup_steps=warmup_steps,
        eval_strategy="epoch" if val_ds is not None else "no",
        save_strategy="epoch" if val_ds is not None else "no",
        save_total_limit=2,
        load_best_model_at_end=val_ds is not None,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=device == "cuda",
        gradient_checkpointing=False,  # enabled manually on the model above
        report_to="none",
        seed=args.seed,
        logging_steps=10,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] if val_ds is not None else [],
    )
    trainer.train()

    val_metrics = trainer.evaluate() if val_ds is not None else {}
    print(f"Validation metrics: {val_metrics}")

    global GOLD_BY_ID
    with open(OUT_DIR / args.gold_file) as f:
        gold = json.load(f)
    gold_rows = gold["rows"][: args.eval_limit] if args.eval_limit else gold["rows"]
    GOLD_BY_ID = {r["cmplid"]: r for r in gold["rows"]}

    gold_metrics, gold_results = evaluate_on_gold(
        model, tokenizer, gold_rows, system_prompt, args.max_new_tokens, device,
    )
    def fmt_pct(v):
        return f"{v:.1%}" if v is not None else "n/a"

    print(f"\n=== Rung 3 (LLM LoRA, run={run_name}) on gold set (n={gold_metrics['n']}) ===")
    print(f"  accuracy overall           {fmt_pct(gold_metrics['accuracy'])}")
    print(f"  accuracy hard subset       {fmt_pct(gold_metrics['accuracy_hard_subset'])}")
    print(f"  accuracy dual-agreement    {fmt_pct(gold_metrics['accuracy_dual_agreement_subset'])}")
    macro_f1 = gold_metrics["macro_f1"]
    print(f"  macro-F1                   {macro_f1:.3f}" if macro_f1 is not None else "  macro-F1                   n/a")
    print(f"  parse failures             {gold_metrics['n_parse_failures']}/{gold_metrics['n']}")

    final_dir = out_dir / "final"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Saved LoRA adapter to {final_dir}")

    summary_path = OUT_DIR / f"rung3_llm_eval_{run_name}.json"
    with open(summary_path, "w") as f:
        json.dump({
            "run_name": run_name,
            "model_name": args.model_name,
            "n_train": len(train_df),
            "n_val": len(val_df),
            "hyperparams": {
                "epochs": args.epochs, "batch_size": args.batch_size,
                "grad_accum_steps": args.grad_accum_steps, "lr": args.lr,
                "max_length": args.max_length, "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
            },
            "validation_metrics": val_metrics,
            "gold_metrics": gold_metrics,
            "adapter_dir": str(final_dir),
        }, f, indent=2)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
