"""Unattended pull → score → mix check → retrain decision.

Does not launch a paid GPU unless you pass --allow-retrain AND the locked
mix rule has a two-month fire. Scoring the live DeBERTa is CPU-safe.
Retrain is not: this box has no GPU.

Usage (from repo root):
    python -m prod.loop
    python -m prod.loop --skip-pull --skip-score
    docker compose run --rm check
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from prod.paths import DATA, DEBERTA_CKPT, ensure_prod_out
from prod.live_mix import LAST_RETRAIN_YM, run_live_mix


CHAMPION = {
    "model": "deberta-v3-base",
    "checkpoint": str(DEBERTA_CKPT),
    "why": (
        "Live serving champion is the full-history DeBERTa. "
        "The 2019 as-of replay model is not served. "
        "The November 2025 challenger was not promoted (Wilson overlap)."
    ),
}


def _teacher_through(ym: str) -> int:
    path = DATA / "replay" / "teacher_dated.parquet"
    if not path.exists():
        return 0
    dated = pd.read_parquet(path, columns=["ldate"])
    return int((dated["ldate"].astype(str).str[:6] <= ym).sum())


def _decide(mix: dict, allow_retrain: bool) -> dict:
    fire = mix.get("first_fire")
    if not fire:
        return {
            "retrain": False,
            "promote": False,
            "reason": (
                "No two-month fire on the locked NHTSA mix rule after "
                f"{LAST_RETRAIN_YM}. Keep the live DeBERTa."
            ),
            "launch": False,
        }

    fire_month = fire["month"]
    n_teacher = _teacher_through(fire_month)
    decision = {
        "retrain": True,
        "promote": False,
        "fire_month": fire_month,
        "previous_month": fire.get("previous_month"),
        "n_teacher_through_fire": n_teacher,
        "reason": (
            f"Mix fire on {fire.get('previous_month')} + {fire_month}. "
            f"A challenger would train on the {n_teacher:,} existing teacher "
            "labels received through that month. New pull rows have no teacher "
            "labels. Promotion still needs the Stage 3 gate "
            "(current-complaints key point-higher, legacy drop caps, no "
            "overlapping Wilson intervals). gold_current_v1 is allowed for a "
            "live gate; it was not allowed in the 2019 replay."
        ),
        "launch": False,
    }
    if not allow_retrain:
        decision["reason"] += (
            " Not launching a GPU. Pass --allow-retrain if you want that sitting."
        )
        return decision
    decision["launch"] = True
    decision["reason"] += " --allow-retrain is set; the loop will print the train command, not start a VM."
    return decision


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-pull", action="store_true")
    ap.add_argument("--skip-score", action="store_true")
    ap.add_argument("--allow-retrain", action="store_true")
    ap.add_argument("--kind", choices=["recent", "full"], default="recent")
    args = ap.parse_args()

    out_dir = ensure_prod_out()
    steps = {}

    if not args.skip_pull:
        from prod.ingest import cmd_pull

        cmd_pull(argparse.Namespace(kind=args.kind, force=False))
        steps["pull"] = "ran"
    else:
        steps["pull"] = "skipped"

    if not args.skip_score:
        from prod.score_incoming import score_incoming
        from prod.ingest import NEW_ROWS

        steps["score"] = score_incoming(Path(NEW_ROWS))
    else:
        scored = out_dir / "incoming_deberta.json"
        steps["score"] = (
            json.loads(scored.read_text(encoding="utf-8"))
            if scored.exists()
            else "skipped"
        )

    mix = run_live_mix()
    decision = _decide(mix, args.allow_retrain)
    payload = {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "champion": CHAMPION,
        "steps": {k: (v if isinstance(v, str) else {kk: v[kk] for kk in v if kk != "pred_hist"}) for k, v in steps.items()},
        "mix": {
            "threshold": mix["threshold"],
            "latest_ym": mix["latest_ym"],
            "n_over": mix["n_over"],
            "first_fire": mix["first_fire"],
            "partial_month_trips": mix["partial_month_trips"],
        },
        "decision": decision,
    }
    # Keep pred hist out of the tiny decision file; it lives on incoming_deberta.json
    path = out_dir / "loop_decision.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "champion.json").write_text(json.dumps(CHAMPION, indent=2), encoding="utf-8")
    print(json.dumps({"decision": decision, "mix": payload["mix"]}, indent=2))
    print(f"Wrote {path}")
    if decision.get("launch"):
        ym = decision["fire_month"]
        print(
            "Train command (GPU box):\n"
            f"  python -m prod.replay_promote --fire-month {ym} --skip-asof-train"
        )
        print("That is a paid sitting. This process did not start a VM.")


if __name__ == "__main__":
    main()
