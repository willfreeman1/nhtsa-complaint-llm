"""Build the blind current-complaints labeling queue.

Random sample of vehicle complaints received AFTER 2026-08-26, excluding
every training and legacy-gold CMPLID. Writes a blind queue (what Will sees)
and a sealed sidecar (NHTSA fields, never opened by the labeling UI).

Usage (from repo root):
    python -m prod.sample_current --source data/incoming/<extracted>.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from prod.ingest import _load_flat
from prod.paths import DATA, GOLD_V3, SCRIPTS, ensure_prod_out

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from teacher_prompt import CLASS_DEFINITIONS, RARE_CATEGORIES  # noqa: E402

LABEL_DEFINITIONS = {**CLASS_DEFINITIONS, **RARE_CATEGORIES}

CUTOFF = "20260826"
SEED = 20260919
N_RANDOM = 150
N_ADAS_OVERSAMPLE = 50
ADAS_TOP = {
    "FORWARD COLLISION AVOIDANCE",
    "BACK OVER PREVENTION",
    "LANE DEPARTURE",
}

TRAIN_PARQUETS = (
    DATA / "training_labels_full_62k_plus_rare.parquet",
    DATA / "training_labels_full_62k.parquet",
    DATA / "training_labels_rare_topup.parquet",
    DATA / "training_sample.parquet",
    DATA / "training_sample_rare_topup.parquet",
)


def _blocked_ids() -> set[str]:
    blocked = set()
    gold = json.loads(GOLD_V3.read_text(encoding="utf-8"))
    blocked.update(str(r["cmplid"]) for r in gold["rows"])
    for path in TRAIN_PARQUETS:
        if not path.exists():
            continue
        peek = pd.read_parquet(path)
        col = "cmplid" if "cmplid" in peek.columns else "CMPLID"
        if col not in peek.columns:
            continue
        blocked.update(peek[col].astype(str))
    return blocked


def _eligible(df: pd.DataFrame, blocked: set[str]) -> pd.DataFrame:
    top = df["COMPDESC"].astype(str).str.split(":").str[0].str.strip()
    out = df.copy()
    out["COMPDESC_TOP"] = top
    out = out[out["PROD_TYPE"] == "V"]
    out = out[out["LDATE"].astype(str) > CUTOFF]
    out = out[~out["CMPLID"].astype(str).isin(blocked)]
    out = out[out["CDESCR"].astype(str).str.len() >= 20]
    return out


def _row_to_blind(r: dict) -> dict:
    return {
        "cmplid": str(r["CMPLID"]),
        "make": r["MAKETXT"],
        "model": r["MODELTXT"],
        "year": r["YEARTXT"],
        "narrative": r["CDESCR"],
        "ldate": str(r["LDATE"]),
    }


def _row_to_sealed(r: dict) -> dict:
    return {
        **_row_to_blind(r),
        "nhtsa_compdesc": r["COMPDESC"],
        "nhtsa_compdesc_top": r["COMPDESC_TOP"],
        "odino": r["ODINO"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Extracted new FLAT_CMPL.txt (or 5-year txt)")
    ap.add_argument("--n-random", type=int, default=N_RANDOM)
    ap.add_argument("--n-adas", type=int, default=N_ADAS_OVERSAMPLE)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    blocked = _blocked_ids()
    print(f"Blocked IDs (train + gold v3): {len(blocked):,}")
    src = Path(args.source)
    print(f"Loading {src} ...")
    raw = _load_flat(
        src,
        columns=[
            "CMPLID", "ODINO", "PROD_TYPE", "MAKETXT", "MODELTXT", "YEARTXT",
            "COMPDESC", "CDESCR", "LDATE", "DATEA",
        ],
    )
    elig = _eligible(raw, blocked)
    print(f"Eligible vehicle rows with LDATE > {CUTOFF}: {len(elig):,}")
    if elig.empty:
        raise SystemExit(
            "No eligible rows. The new dump may not contain complaints after "
            f"{CUTOFF}, or the source path is the research copy."
        )

    n_rand = min(args.n_random, len(elig))
    random_df = elig.sample(n=n_rand, random_state=args.seed)
    remaining = elig[~elig["CMPLID"].isin(random_df["CMPLID"])]
    adas_pool = remaining[remaining["COMPDESC_TOP"].isin(ADAS_TOP)]
    n_adas = min(args.n_adas, len(adas_pool))
    adas_df = adas_pool.sample(n=n_adas, random_state=args.seed) if n_adas else adas_pool

    out_dir = ensure_prod_out()
    queue = {
        "schema": "gold_current_v1_queue",
        "blind": True,
        "cutoff_ldate_exclusive": CUTOFF,
        "seed": args.seed,
        "class_definitions": LABEL_DEFINITIONS,
        "instructions": (
            "Assign accept-list label(s) from the class definitions only. "
            "You may accept more than one label when the narrative genuinely "
            "supports more than one. Do not look up NHTSA's field or any model "
            "prediction until the first pass is finished."
        ),
        "rows": [_row_to_blind(r) for r in random_df.to_dict(orient="records")],
    }
    sealed = {
        "schema": "gold_current_v1_sealed",
        "note": "Do not open during first-pass labeling.",
        "rows": [_row_to_sealed(r) for r in random_df.to_dict(orient="records")],
    }
    adas_queue = {
        "schema": "gold_current_adas_oversample_queue",
        "blind": True,
        "separate_from_random_accuracy": True,
        "rows": [_row_to_blind(r) for r in adas_df.to_dict(orient="records")],
    }
    adas_sealed = {
        "schema": "gold_current_adas_oversample_sealed",
        "rows": [_row_to_sealed(r) for r in adas_df.to_dict(orient="records")],
    }

    files = {
        "queue": out_dir / "gold_current_v1_queue.json",
        "sealed": out_dir / "gold_current_v1_sealed.json",
        "adas_queue": out_dir / "gold_current_adas_queue.json",
        "adas_sealed": out_dir / "gold_current_adas_sealed.json",
    }
    files["queue"].write_text(json.dumps(queue, indent=2), encoding="utf-8")
    files["sealed"].write_text(json.dumps(sealed, indent=2), encoding="utf-8")
    files["adas_queue"].write_text(json.dumps(adas_queue, indent=2), encoding="utf-8")
    files["adas_sealed"].write_text(json.dumps(adas_sealed, indent=2), encoding="utf-8")

    summary = {
        "eligible": int(len(elig)),
        "random_n": int(len(random_df)),
        "adas_oversample_n": int(len(adas_df)),
        "cutoff_ldate_exclusive": CUTOFF,
        "seed": args.seed,
        "source": str(src),
        "ldate_min": str(elig["LDATE"].min()),
        "ldate_max": str(elig["LDATE"].max()),
        "blocked_n": len(blocked),
        "files": {k: str(v) for k, v in files.items()},
    }
    (out_dir / "gold_current_v1_sample_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print("Blind queue is ready. Start the labeling tool:")
    print("  python -m prod.labeling.app")


if __name__ == "__main__":
    main()
