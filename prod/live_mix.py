"""Live NHTSA-census mix check after the first locked retrain month.

Uses the same matched-pile rule as docs/THRESHOLDS.md: last 24 months vs the
20 years immediately before those 24 months; a new month vs that same
20-year pile; trip = mean + 2 sample SD; fire = two consecutive complete
months over the line. Windows are slid to November 2025 (the first fire).

This is NHTSA's own category field on every vehicle filing. Not model
error. Not accuracy.

Usage (from repo root):
    python -m prod.live_mix
"""
from __future__ import annotations

import json

import pandas as pd

from prod.nhtsa_mix_trigger import (
    LAST_N,
    _first_fire,
    _line,
    _walk,
    load_vehicle_filings,
    windows_at,
)
from prod.paths import ensure_prod_out
from prod.replay_prepare import _months

# First locked fire. After that sitting the windows slide here.
LAST_RETRAIN_YM = "202511"


def _month_coverage(veh: pd.DataFrame, months: list[str]) -> dict[str, dict]:
    out = {}
    for ym in months:
        days = veh.loc[veh["ym"] == ym, "ldate"].str[6:8]
        if days.empty:
            out[ym] = {"n": 0, "max_day": None, "complete": False}
            continue
        max_day = int(days.max())
        out[ym] = {
            "n": int(len(days)),
            "max_day": max_day,
            "complete": max_day >= 28,
        }
    return out


def run_live_mix() -> dict:
    veh = load_vehicle_filings()
    universe = sorted(set(veh["nhtsa_top"].astype(str)))
    win = windows_at(LAST_RETRAIN_YM)
    line = _line(veh, universe, win)

    latest = str(veh["ym"].max())
    walk_start = _months(LAST_RETRAIN_YM, latest)
    walk_months = [m for m in walk_start if m > LAST_RETRAIN_YM]
    walk = _walk(veh, universe, walk_months, line["pool_mix"], line["threshold"], f"after_{LAST_RETRAIN_YM}")
    coverage = _month_coverage(veh, walk_months)
    for row in walk:
        cov = coverage[row["month"]]
        row["complete"] = cov["complete"]
        row["max_day"] = cov["max_day"]

    complete_walk = [r for r in walk if r["complete"]]
    fire = _first_fire(complete_walk)
    watching = [r for r in walk if r["trip"] and not r["complete"]]

    summary = {
        "schema": "live_nhtsa_census_mix_v2",
        "last_retrain_ym": LAST_RETRAIN_YM,
        "rule": (
            f"Windows slid to {LAST_RETRAIN_YM}. Last {LAST_N} months vs the "
            "prior 20 years. New months vs that same pile. Trip = mean + 2 "
            "sample SD. Fire = two consecutive complete months (day 28 or later "
            "present). A partial last month can trip the line but cannot fire."
        ),
        "windows": win,
        "pool_n": line["pool_n"],
        "mean": line["mean"],
        "std": line["std"],
        "threshold": line["threshold"],
        "latest_ym": latest,
        "n_vehicle": int(len(veh)),
        "walk": walk,
        "n_over": sum(1 for r in walk if r["trip"]),
        "n_over_complete": sum(1 for r in complete_walk if r["trip"]),
        "first_fire": fire,
        "partial_month_trips": watching,
        "what_this_is": (
            "NHTSA top-level component field on every vehicle filing, "
            "including the latest pull. Not DeBERTa predictions. Not accuracy."
        ),
    }
    out = ensure_prod_out() / "live_mix.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "windows": win,
        "threshold": line["threshold"],
        "latest_ym": latest,
        "n_over": summary["n_over"],
        "first_fire": fire,
        "partial_month_trips": [
            {"month": r["month"], "distance_vs_pool": r["distance_vs_pool"], "n": r["n"]}
            for r in watching
        ],
    }, indent=2))
    print(f"Wrote {out}")
    return summary


def main():
    run_live_mix()


if __name__ == "__main__":
    main()
