"""Tiny CLI demo for the trained Rung 2 (DeBERTa-v3) checkpoint. Loads the saved
`final` checkpoint and classifies one or more complaint narratives into one of the
40 `raw_component` categories, using the exact same input formatting
(`train_deberta.build_text`) the model was trained/evaluated with everywhere else in
this project. CPU-only is fine -- no GPU or hosted API required.

Usage:
    python scripts/predict.py "The brake pedal went to the floor..." --make Toyota --model Camry --year 2019
    python scripts/predict.py --demo
"""
import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_deberta as td

CKPT_DIR = Path(__file__).resolve().parent.parent / "checkpoints" / "deberta_rung2_full_62k_plus_rare" / "final"

DEMO_NARRATIVES = [
    {
        "narrative": "The brake pedal went to the floor without stopping the vehicle. "
                     "I had to use the emergency brake to avoid a collision at a stop light.",
        "make": "Toyota", "model": "Camry", "year": "2019",
    },
    {
        "narrative": "During a minor fender bender at low speed, the driver's side airbag "
                     "deployed unexpectedly and caused facial injuries.",
        "make": "Honda", "model": "Accord", "year": "2020",
    },
    {
        "narrative": "While driving on the highway at 65 mph, the steering wheel became "
                     "extremely loose and the vehicle was difficult to control.",
        "make": "Ford", "model": "F-150", "year": "2018",
    },
]


def load_model(checkpoint_dir, device):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir).to(device)
    model.eval()
    return tokenizer, model


def predict_one(tokenizer, model, device, narrative, make, model_name, year, top_k, max_length=256):
    text = td.build_text(narrative, make, model_name, year)
    with torch.no_grad():
        enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        logits = model(**enc).logits[0]
        probs = torch.softmax(logits, dim=-1)
    top_idx = torch.argsort(probs, descending=True)[:top_k].tolist()
    return [(td.ID2LABEL[i], probs[i].item()) for i in top_idx]


def print_prediction(narrative, make, model_name, year, top_preds):
    shown = narrative if len(narrative) <= 100 else narrative[:100].rstrip() + "..."
    print(f"Narrative: {shown}")
    vehicle_bits = " ".join(b for b in (year, make, model_name) if b)
    if vehicle_bits:
        print(f"Vehicle: {vehicle_bits}")
    print()
    top_label, top_prob = top_preds[0]
    print(f"Predicted component: {top_label}  ({top_prob:.1%})")
    if len(top_preds) > 1:
        print("Other candidates:")
        for label, prob in top_preds[1:]:
            print(f"  {label:<30} {prob:.1%}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("narrative", nargs="?", default=None, help="Complaint narrative text")
    ap.add_argument("--make", default="", help="Vehicle make (optional)")
    ap.add_argument("--model", dest="model_name", default="", help="Vehicle model (optional)")
    ap.add_argument("--year", default="", help="Vehicle year (optional)")
    ap.add_argument("--checkpoint", default=str(CKPT_DIR), help="Path to a trained Rung 2 checkpoint dir")
    ap.add_argument("--top-k", type=int, default=3, help="Number of top predictions to show")
    ap.add_argument("--demo", action="store_true", help="Run 3 built-in example narratives instead")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")
    tokenizer, model = load_model(args.checkpoint, device)
    print()

    if args.demo or args.narrative is None:
        for row in DEMO_NARRATIVES:
            top_preds = predict_one(
                tokenizer, model, device, row["narrative"], row["make"], row["model"], row["year"],
                args.top_k,
            )
            print_prediction(row["narrative"], row["make"], row["model"], row["year"], top_preds)
    else:
        top_preds = predict_one(
            tokenizer, model, device, args.narrative, args.make, args.model_name, args.year, args.top_k,
        )
        print_prediction(args.narrative, args.make, args.model_name, args.year, top_preds)


if __name__ == "__main__":
    main()
