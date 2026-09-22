"""Score new vehicle filings with the live DeBERTa. No labels. Not accuracy.

Usage (from repo root):
    python -m prod.score_incoming
    python -m prod.score_incoming --input data/incoming/new_vehicle_latest.parquet
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter

import pandas as pd

from prod.paths import DEBERTA_CKPT, INCOMING, ensure_prod_out
from prod.replay_prepare import _nhtsa_top


UNKNOWN = "UNKNOWN OR OTHER"
GADGET = (
    "FORWARD COLLISION AVOIDANCE",
    "BACK OVER PREVENTION",
    "LANE DEPARTURE",
)


def _frame_for_model(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["narrative"] = out["CDESCR"]
    out["make"] = out["MAKETXT"]
    out["model"] = out["MODELTXT"]
    out["year"] = out["YEARTXT"]
    return out


def _predict(model, tokenizer, df, batch_size, max_length, device):
    import torch
    import train_deberta as td

    texts = td.build_texts(df)
    preds: list[str] = []
    confs: list[float] = []
    alt: list[list[dict]] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            probs = torch.softmax(model(**enc).logits.float(), dim=-1)
            conf, idx = probs.max(dim=-1)
            topv, topi = probs.topk(3, dim=-1)
            preds.extend(td.ID2LABEL[int(j)] for j in idx.tolist())
            confs.extend(float(c) for c in conf.tolist())
            for row_v, row_i in zip(topv.tolist(), topi.tolist()):
                alt.append([
                    {"label": td.ID2LABEL[int(j)], "p": float(p)}
                    for p, j in zip(row_v, row_i)
                ])
            if i and i % (batch_size * 20) == 0:
                print(f"  scored {i:,}/{len(texts):,}", flush=True)
    return preds, confs, alt


def score_incoming(input_path, batch_size: int = 32, max_length: int = 256) -> dict:
    import torch
    from prod.weights import load_ckpt as _load_ckpt

    path = input_path
    if not path.exists():
        raise SystemExit(f"No incoming file at {path}. Run python -m prod.ingest pull first.")
    raw = pd.read_parquet(path)
    if raw.empty:
        summary = {
            "n": 0,
            "checkpoint": str(DEBERTA_CKPT),
            "note": "incoming file is empty",
        }
        out = ensure_prod_out() / "incoming_deberta.json"
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    df = _frame_for_model(raw)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Scoring {len(df):,} rows on {device} from {DEBERTA_CKPT}", flush=True)
    model, tokenizer = _load_ckpt(DEBERTA_CKPT, device)
    t0 = time.perf_counter()
    preds, confs, alt = _predict(model, tokenizer, df, batch_size, max_length, device)
    elapsed = time.perf_counter() - t0

    nhtsa = df["COMPDESC"].map(_nhtsa_top)
    scored = raw.copy()
    scored["pred"] = preds
    scored["confidence"] = confs
    scored["nhtsa_top"] = nhtsa.to_numpy()
    scored["top3"] = [json.dumps(a) for a in alt]
    scored_path = INCOMING / "new_vehicle_latest_scored.parquet"
    scored.to_parquet(scored_path, index=False)

    hist = Counter(preds)
    n = len(preds)
    agree = float((scored["pred"] == scored["nhtsa_top"]).mean())
    summary = {
        "schema": "incoming_deberta",
        "n": n,
        "checkpoint": str(DEBERTA_CKPT),
        "device": device,
        "ldate_min": str(raw["LDATE"].min()) if "LDATE" in raw.columns else None,
        "ldate_max": str(raw["LDATE"].max()) if "LDATE" in raw.columns else None,
        "unknown_rate": hist.get(UNKNOWN, 0) / n,
        "median_confidence": float(pd.Series(confs).median()),
        "gadget_rate": sum(p in GADGET for p in preds) / n,
        "nhtsa_field_agreement": agree,
        "nhtsa_field_agreement_is_not_accuracy": True,
        "pred_hist": dict(hist.most_common()),
        "timing": {
            "score_seconds": elapsed,
            "rows_per_second": n / elapsed if elapsed else None,
            "batch_size": batch_size,
        },
        "scored_path": str(scored_path),
    }
    out = ensure_prod_out() / "incoming_deberta.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"unknown={summary['unknown_rate']:.3f}  "
        f"median_conf={summary['median_confidence']:.3f}  "
        f"gadget={summary['gadget_rate']:.3f}  "
        f"nhtsa_agree={agree:.3f} (not accuracy)  "
        f"{summary['timing']['rows_per_second']:.2f} rows/s",
        flush=True,
    )
    print(f"Wrote {out}", flush=True)
    print(f"Wrote {scored_path}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(INCOMING / "new_vehicle_latest.parquet"))
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    score_incoming(__import__("pathlib").Path(args.input), batch_size=args.batch_size)


if __name__ == "__main__":
    main()
