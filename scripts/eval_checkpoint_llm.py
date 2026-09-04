"""Standalone gold-set per-class breakdown + novel-category probe for an ALREADY
TRAINED Rung 3 LoRA adapter. Doesn't retrain -- loads the base model + saved adapter
and reuses train_llm_lora.py's evaluate_on_gold() logic exactly (so numbers match the
training-run summary), then additionally:
  - dumps a full sklearn classification_report broken out per class
  - runs the shared novel_category_probe narratives through the model and reports the
    raw generated JSON for each (qualitative check, not scored -- see
    novel_category_probe.py's docstring)

Usage:
    PYTHONPATH=scripts python3 scripts/eval_checkpoint_llm.py \
        --adapter-dir checkpoints/llm_rung3_full_62k_plus_rare \
        --run-name full_62k_plus_rare
"""
import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from sklearn.metrics import classification_report
from transformers import AutoModelForCausalLM, AutoTokenizer

import train_llm_lora as tl
from novel_category_probe import NOVEL_CATEGORY_NARRATIVES

OUT_DIR = Path(__file__).parent.parent / "output"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-dir", required=True)
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--gold-file", default="gold_eval_set_v2.json")
    ap.add_argument("--run-name", default="full_62k_plus_rare")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    if device != "cuda":
        model = model.to(device)
    model.eval()

    with open(OUT_DIR / args.gold_file) as f:
        gold = json.load(f)
    gold_rows = gold["rows"]
    tl.GOLD_BY_ID = {r["cmplid"]: r for r in gold_rows}

    system_prompt = tl.build_student_system_prompt()

    metrics, results = tl.evaluate_on_gold(
        model, tokenizer, gold_rows, system_prompt, args.max_new_tokens, device,
    )
    print(f"Gold: accuracy={metrics['accuracy']:.3f} macro_f1={metrics['macro_f1']:.3f} "
          f"(sanity check against the training-run summary JSON)")

    # NOTE: do NOT run primary through tl.bucket_component() here -- that function
    # collapses all 9 rare categories (and any off-list answer) down to a generic
    # "OTHER" bucket. Rung 3 trains/scores on the full 40-class raw_component space, so
    # primary should be used as-is.
    y_true = [
        r["pred"] if r["correct"] else tl.GOLD_BY_ID[r["cmplid"]]["primary"]
        for r in results
    ]
    y_pred = [r["pred"] or "PARSE_FAILURE" for r in results]
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    support_sorted = sorted(
        ((label, stats) for label, stats in report.items() if isinstance(stats, dict)),
        key=lambda kv: -kv[1].get("support", 0),
    )
    print("\nPer-class F1 (sorted by support, most-represented first):")
    for label, stats in support_sorted:
        print(f"  {label:<32} f1={stats['f1-score']:.3f}  support={int(stats['support'])}")

    # --- Novel-category probe ---
    novel_results = []
    print("\nNovel-category probe (no ground truth -- qualitative check for forced fits):")
    for row in NOVEL_CATEGORY_NARRATIVES:
        user_msg = tl.build_user_message(
            narrative=row["narrative"], make=row["make"], model=row["model"], year=row["year"],
        )
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_msg}],
            add_generation_prompt=True, tokenize=False,
        )
        prompt_ids = tokenizer(
            prompt_text, add_special_tokens=False, return_tensors="pt",
        )["input_ids"].to(device)
        with torch.no_grad():
            out = model.generate(
                prompt_ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        gen_text = tokenizer.decode(out[0][prompt_ids.shape[1]:], skip_special_tokens=True)
        try:
            parsed = tl.parse_json_response(gen_text)
        except Exception as e:  # noqa: BLE001
            parsed = None
            print(f"  [{row['cmplid']}] PARSE FAILURE: {e}")
        print(f"  [{row['cmplid']}] pred_component={parsed.get('component') if parsed else None}")
        print(f"      narrative: {row['narrative'][:100]}...")
        print(f"      raw gen: {gen_text[:200]}")
        novel_results.append({**row, "gen_text": gen_text, "parsed": parsed})

    out = {
        "run_name": args.run_name,
        "adapter_dir": args.adapter_dir,
        "gold_metrics": metrics,
        "per_class_report": report,
        "novel_category_results": novel_results,
    }
    out_path = OUT_DIR / f"rung3_percategory_and_novel_{args.run_name}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
