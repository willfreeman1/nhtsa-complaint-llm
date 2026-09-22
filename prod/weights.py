"""Load the live DeBERTa checkpoint. Tokenizer is the training base, not the saved files."""
from __future__ import annotations

import sys
from pathlib import Path

from prod.paths import ROOT, SCRIPTS

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_deberta as td  # noqa: E402


def _weight_file(path: Path) -> Path | None:
    for name in ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json"):
        cand = path / name
        if cand.exists():
            return cand
    return None


def load_ckpt(path: Path, device: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    weights = _weight_file(path)
    if weights is None:
        raise SystemExit(f"No model weights under {path}")
    if weights.stat().st_size < 50_000_000:
        raise SystemExit(f"Weight file too small ({weights.stat().st_size} bytes): {weights}")
    print(f"Loading weights {weights} ({weights.stat().st_size / 1e6:.1f} MB)", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    model = AutoModelForSequenceClassification.from_pretrained(
        path, local_files_only=True,
    ).to(device)
    model.eval()
    n_labels = int(getattr(model.config, "num_labels", 0) or 0)
    if n_labels != len(td.LABELS):
        raise SystemExit(f"num_labels={n_labels} expected {len(td.LABELS)}")
    return model, tokenizer
