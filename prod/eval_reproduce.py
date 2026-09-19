"""Reproduce published gold-v3 numbers and attach Wilson CIs, timing, memory.

Does not overwrite published eval JSON. Writes output/prod/ instead.

Usage (from repo root):
    python -m prod.eval_reproduce --model deberta
    python -m prod.eval_reproduce --model deberta --year-slice
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
import psutil

from prod.paths import DEBERTA_CKPT, GOLD_V3, ROOT, SCRIPTS, ensure_prod_out
from prod.stats import accuracy_with_ci

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_deberta as td  # noqa: E402


PUBLISHED = {
    "deberta": {
        "source": "output/rung2_percategory_and_novel_full_62k_plus_rare_v3.json",
        "accuracy": 0.822141560798548,
        "accuracy_hard_subset": 0.7326732673267327,
        "accuracy_dual_agreement_subset": 0.9314516129032258,
        "macro_f1": 0.7746955110198922,
    },
    "qwen": {
        "source": "output/rung3_percategory_and_novel_full_62k_plus_rare_v3.json",
        "accuracy": 0.8620689655172413,
        "accuracy_hard_subset": 0.7656765676567657,
        "accuracy_dual_agreement_subset": 0.9798387096774194,
        "macro_f1": 0.7993698768968066,
    },
}

YEAR_BUCKETS = (
    ("1995-2009", 1995, 2009),
    ("2010-2014", 2010, 2014),
    ("2015-2019", 2015, 2019),
    ("2020-2022", 2020, 2022),
    ("2023-2024", 2023, 2024),
    ("2025-2026", 2025, 2026),
)


def _load_gold():
    gold = json.loads(GOLD_V3.read_text(encoding="utf-8"))
    return gold["rows"]


def _attach_received_year(gold_rows):
    """Join gold CMPLID -> LDATE year from the 2026-08-28 parquet."""
    import pandas as pd

    dates = pd.read_parquet(
        ROOT / "data" / "cmpl.parquet", columns=["CMPLID", "LDATE", "DATEA"]
    )
    dates["CMPLID"] = dates["CMPLID"].astype(str)
    by_id = dates.set_index("CMPLID")
    out = []
    missing = 0
    for row in gold_rows:
        rec = by_id.loc[str(row["cmplid"])] if str(row["cmplid"]) in by_id.index else None
        if rec is None:
            missing += 1
            year = None
        else:
            ldate = str(rec["LDATE"])
            year = int(ldate[:4]) if ldate and ldate[:4].isdigit() else None
        out.append({**row, "ldate_year": year})
    return out, missing


def _bucket(year):
    if year is None:
        return "unknown"
    for name, lo, hi in YEAR_BUCKETS:
        if lo <= year <= hi:
            return name
    return "other"


def _metrics_from_results(results):
    def pack(subset, label):
        succ = sum(1 for r in subset if r["correct"])
        packed = accuracy_with_ci(succ, len(subset))
        packed["label"] = label
        return packed

    hard = [r for r in results if r["is_hard"]]
    easy = [r for r in results if not r["is_hard"]]
    return {
        "overall": pack(results, "overall"),
        "hard": pack(hard, "hard"),
        "easy": pack(easy, "easy"),
    }


def _check_published(model_key, metrics):
    pub = PUBLISHED[model_key]
    observed = metrics["accuracy"]
    ok = round(observed, 3) == round(pub["accuracy"], 3)
    return {
        "pass": ok,
        "published_overall": pub["accuracy"],
        "observed_overall": observed,
        "abs_diff": abs(observed - pub["accuracy"]),
        "threshold": "agree to 3 decimal places (THRESHOLDS.md Stage 0 reproduction)",
    }


def eval_deberta(gold_rows, batch_size, max_length):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    proc = psutil.Process()
    rss_before = proc.memory_info().rss
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t_load = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(DEBERTA_CKPT)
    model = AutoModelForSequenceClassification.from_pretrained(DEBERTA_CKPT).to(device)
    model.eval()
    load_s = time.perf_counter() - t_load
    rss_after_load = proc.memory_info().rss

    td.GOLD_BY_ID = {r["cmplid"]: r for r in gold_rows}
    t_score = time.perf_counter()
    metrics, results = td.evaluate_on_gold(
        model, tokenizer, gold_rows, max_length, batch_size, device,
    )
    score_s = time.perf_counter() - t_score
    rss_peak = proc.memory_info().rss

    n = len(gold_rows)
    return {
        "device": device,
        "checkpoint": str(DEBERTA_CKPT),
        "gold_metrics": metrics,
        "ci": _metrics_from_results(results),
        "reproduction": _check_published("deberta", metrics),
        "timing": {
            "load_seconds": load_s,
            "score_seconds": score_s,
            "rows_per_second": n / score_s if score_s else None,
            "n": n,
            "batch_size": batch_size,
        },
        "memory_bytes": {
            "rss_before": rss_before,
            "rss_after_load": rss_after_load,
            "rss_after_score": rss_peak,
            "rss_delta_load": rss_after_load - rss_before,
        },
        "results": results,
    }, model, tokenizer


def year_slice(gold_rows, results):
    dated, missing = _attach_received_year(gold_rows)
    by_cmplid = {r["cmplid"]: r for r in results}
    buckets = defaultdict(list)
    for row in dated:
        res = by_cmplid[row["cmplid"]]
        buckets[_bucket(row["ldate_year"])].append(res)
    sliced = {}
    for name, _lo, _hi in YEAR_BUCKETS:
        subset = buckets.get(name, [])
        succ = sum(1 for r in subset if r["correct"])
        sliced[name] = accuracy_with_ci(succ, len(subset))
    sliced["unknown_ldate"] = accuracy_with_ci(
        sum(1 for r in buckets.get("unknown", []) if r["correct"]),
        len(buckets.get("unknown", [])),
    )
    return {"missing_date_join": missing, "by_received_year": sliced}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["deberta"], default="deberta")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--year-slice", action="store_true")
    args = ap.parse_args()

    out_dir = ensure_prod_out()
    gold_rows = _load_gold()
    print(f"Gold v3 rows: {len(gold_rows)}")

    payload, _model, _tok = eval_deberta(gold_rows, args.batch_size, args.max_length)
    print(
        f"Device={payload['device']}  acc={payload['gold_metrics']['accuracy']:.3f}  "
        f"hard={payload['gold_metrics']['accuracy_hard_subset']:.3f}  "
        f"{payload['timing']['rows_per_second']:.2f} rows/s  "
        f"load_delta_rss={payload['memory_bytes']['rss_delta_load'] / 1e6:.0f} MB"
    )
    print(f"Reproduction pass: {payload['reproduction']['pass']}  "
          f"(diff={payload['reproduction']['abs_diff']:.6f})")
    print(
        f"Overall {payload['ci']['overall']['accuracy']:.3f}  "
        f"Wilson95 [{payload['ci']['overall']['wilson_95']['lo']:.3f}, "
        f"{payload['ci']['overall']['wilson_95']['hi']:.3f}]"
    )

    if args.year_slice:
        payload["year_slice"] = year_slice(gold_rows, payload["results"])
        print("By received year:")
        for name, stats in payload["year_slice"]["by_received_year"].items():
            if not stats["n"]:
                continue
            print(
                f"  {name}: n={stats['n']} acc={stats['accuracy']:.3f} "
                f"[{stats['wilson_95']['lo']:.3f}, {stats['wilson_95']['hi']:.3f}]"
            )

    # Per-row preds are useful later; keep them in a sidecar so the summary stays readable.
    results = payload.pop("results")
    summary_path = out_dir / "stage0_deberta_reproduce.json"
    rows_path = out_dir / "stage0_deberta_reproduce_rows.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")
    print(f"Wrote {rows_path}")


if __name__ == "__main__":
    main()
