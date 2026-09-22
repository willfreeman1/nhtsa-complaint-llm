"""NHTSA-census mix trigger. Every vehicle filing. Matched piles.

A new month is compared to the same twenty-year pile used to set the line.
After a retrain, both windows slide to the retrain month.

Usage (from repo root):
    python -m prod.nhtsa_mix_trigger
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from prod.paths import DATA, INCOMING, ensure_prod_out
from prod.replay_prepare import WALK_END, WALK_START, _months, _nhtsa_top


def _jensen_shannon(p, q, eps=1e-12) -> float:
    """Sqrt of Jensen–Shannon divergence, natural log. Same formula as the replay."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / np.clip(b[mask], eps, None))))

    return float(np.sqrt(max(0.0, 0.5 * kl(p, m) + 0.5 * kl(q, m))))

ASOF_YM = "201912"
LAST_N = 24
PRIOR_YEARS = 20


def add_months(ym: str, n: int) -> str:
    y, m = int(ym[:4]), int(ym[4:6])
    m += n
    while m > 12:
        m -= 12
        y += 1
    while m < 1:
        m += 12
        y -= 1
    return f"{y}{m:02d}"


def windows_at(asof_ym: str) -> dict:
    last_end = asof_ym
    last_start = add_months(asof_ym, -(LAST_N - 1))
    pool_end_ym = add_months(last_start, -1)
    pool_start_ym = add_months(last_start, -PRIOR_YEARS * 12)
    return {
        "asof_ym": asof_ym,
        "last_24_start": last_start,
        "last_24_end": last_end,
        "pool_start": f"{pool_start_ym}01",
        "pool_end": f"{pool_end_ym}31",
    }


def _mix(labels: list[str], universe: list[str]) -> np.ndarray:
    from collections import Counter

    counts = Counter(labels)
    return np.array([counts.get(lab, 0) for lab in universe], dtype=np.float64)


def _in_range(ldate: pd.Series, start: str, end: str) -> pd.Series:
    s = ldate.astype(str)
    return (s >= start) & (s <= end)


def _line(veh, universe, win: dict) -> dict:
    pool = veh[_in_range(veh["ldate"], win["pool_start"], win["pool_end"])]
    pool_mix = _mix(pool["nhtsa_top"].astype(str).tolist(), universe)
    sd_rows = []
    for ym in _months(win["last_24_start"], win["last_24_end"]):
        labs = veh[veh["ym"] == ym]["nhtsa_top"].astype(str).tolist()
        d = _jensen_shannon(_mix(labs, universe), pool_mix) if labs else None
        sd_rows.append({"month": ym, "n": len(labs), "distance_vs_pool": d})
    vals = np.array([r["distance_vs_pool"] for r in sd_rows], dtype=np.float64)
    mean = float(vals.mean())
    std = float(vals.std(ddof=1))
    return {
        "windows": win,
        "pool_n": int(len(pool)),
        "mean": mean,
        "std": std,
        "threshold": mean + 2.0 * std,
        "sd_months": sd_rows,
        "pool_mix": pool_mix,
    }


def _walk(veh, universe, months: list[str], pool_mix, threshold: float, era: str) -> list[dict]:
    out = []
    for ym in months:
        labs = veh[veh["ym"] == ym]["nhtsa_top"].astype(str).tolist()
        d = _jensen_shannon(_mix(labs, universe), pool_mix) if labs else None
        out.append({
            "month": ym,
            "n": len(labs),
            "distance_vs_pool": d,
            "trip": bool(d is not None and d > threshold),
            "era": era,
        })
    return out


def load_vehicle_filings() -> pd.DataFrame:
    """Research dump plus any later pull. Vehicle rows only."""
    cols = ["PROD_TYPE", "LDATE", "COMPDESC"]
    frames = [pd.read_parquet(DATA / "cmpl.parquet", columns=cols)]
    incoming = INCOMING / "new_vehicle_latest.parquet"
    if incoming.exists():
        extra = pd.read_parquet(incoming)
        have = [c for c in cols if c in extra.columns]
        frames.append(extra[have])
    veh = pd.concat(frames, ignore_index=True)
    veh = veh[veh["PROD_TYPE"].astype(str) == "V"].copy()
    veh["ldate"] = veh["LDATE"].astype(str)
    veh["ym"] = veh["ldate"].str[:6]
    veh["nhtsa_top"] = veh["COMPDESC"].map(_nhtsa_top)
    return veh


def _first_fire(walk: list[dict]) -> dict | None:
    for i in range(1, len(walk)):
        if walk[i - 1]["trip"] and walk[i]["trip"]:
            return {
                "month": walk[i]["month"],
                "previous_month": walk[i - 1]["month"],
                "signal": "nhtsa_census_mix",
            }
    return None


def main():
    # Historical lock used the research dump only. Do not add incoming rows here.
    cmpl = pd.read_parquet(DATA / "cmpl.parquet", columns=["PROD_TYPE", "LDATE", "COMPDESC"])
    veh = cmpl[cmpl["PROD_TYPE"] == "V"].copy()
    veh["ldate"] = veh["LDATE"].astype(str)
    veh["ym"] = veh["ldate"].str[:6]
    veh["nhtsa_top"] = veh["COMPDESC"].map(_nhtsa_top)
    universe = sorted(set(veh["nhtsa_top"].astype(str)))

    win0 = windows_at(ASOF_YM)
    line0 = _line(veh, universe, win0)
    months0 = _months(WALK_START, WALK_END)
    walk0 = _walk(veh, universe, months0, line0["pool_mix"], line0["threshold"], "asof_2019")
    fire = _first_fire(walk0)

    after = None
    if fire:
        win1 = windows_at(fire["month"])
        line1 = _line(veh, universe, win1)
        rest = [m for m in months0 if m > fire["month"]]
        walk1 = _walk(veh, universe, rest, line1["pool_mix"], line1["threshold"], f"after_{fire['month']}")
        fire2 = _first_fire(walk1)
        after = {
            "line": {k: line1[k] for k in ("windows", "pool_n", "mean", "std", "threshold", "sd_months")},
            "walk": walk1,
            "n_over": sum(1 for r in walk1 if r["trip"]),
            "next_fire": fire2,
            "note": "Windows slid to the first retrain month. Do not retrain again unless DECISIONS.md says so.",
        }
        dated = pd.read_parquet(DATA / "replay" / "teacher_dated.parquet")
        through = dated[dated["ldate"].str[:6] <= fire["month"]].copy()
        through_path = DATA / "replay" / f"asof_through_{fire['month']}.parquet"
        through.to_parquet(through_path, index=False)
        after["n_teacher_through_fire"] = int(len(through))
        after["teacher_parquet"] = str(through_path)

    def _line_public(line):
        return {k: line[k] for k in ("windows", "pool_n", "mean", "std", "threshold", "sd_months")}

    summary = {
        "schema": "nhtsa_census_mix_trigger_v2_matched_piles",
        "asof_ym": ASOF_YM,
        "rule": (
            "Last 24 months vs the 20 years immediately before those 24 months. "
            "New months vs that same 20-year pile. Trip = mean + 2 sample SD. "
            "Fire = two consecutive months. After a retrain, both windows slide."
        ),
        "asof_2019": {
            "line": _line_public(line0),
            "walk": walk0,
            "n_over": sum(1 for r in walk0 if r["trip"]),
            "first_fire": fire,
        },
        "after_first_retrain": after,
        "notes": [
            "NHTSA top-level COMPDESC on every vehicle filing. Not the 50k teacher file.",
            "Not model error.",
            "Same pile for the line and for the walk. That is the v1 lock fix.",
        ],
    }
    out = ensure_prod_out() / "replay" / "nhtsa_mix_trigger.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "asof_windows": win0,
        "asof_mean": line0["mean"],
        "asof_std": line0["std"],
        "asof_threshold": line0["threshold"],
        "asof_n_over": summary["asof_2019"]["n_over"],
        "first_fire": fire,
        "after_windows": after["line"]["windows"] if after else None,
        "after_threshold": after["line"]["threshold"] if after else None,
        "after_n_over": after["n_over"] if after else None,
        "next_fire": after["next_fire"] if after else None,
        "n_teacher_through_fire": after.get("n_teacher_through_fire") if after else None,
    }, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
