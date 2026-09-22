"""Monthly NHTSA category counts for the mix check.

The live mix rule needs every vehicle filing. The nightly job cannot ship
the 2-million-row history file, so we store one count table: month ×
top-level category, plus the last day seen in that month.

Usage (from repo root, needs data/cmpl.parquet):
    python -m prod.census
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from prod.nhtsa_mix_trigger import (
    LAST_N,
    _first_fire,
    _jensen_shannon,
    windows_at,
)
from prod.paths import DATA, ensure_prod_out
from prod.replay_prepare import _months, _nhtsa_top

CUTOFF_LDATE = "20260826"
NIGHTLY_DIR = DATA / "nightly"
CENSUS_PATH = NIGHTLY_DIR / "nhtsa_month_counts.json"


def _empty_month() -> dict:
    return {"n": 0, "max_day": 0, "counts": {}}


def month_counts_from_frame(df: pd.DataFrame) -> dict[str, dict]:
    """Aggregate vehicle rows to {yyyymm: {n, max_day, counts}}."""
    veh = df[df["PROD_TYPE"].astype(str) == "V"].copy()
    if veh.empty:
        return {}
    veh["ldate"] = veh["LDATE"].astype(str)
    veh["ym"] = veh["ldate"].str[:6]
    veh["day"] = pd.to_numeric(veh["ldate"].str[6:8], errors="coerce").fillna(0).astype(int)
    veh["nhtsa_top"] = veh["COMPDESC"].map(_nhtsa_top)
    out: dict[str, dict] = {}
    for ym, g in veh.groupby("ym", sort=True):
        counts = {k: int(v) for k, v in g["nhtsa_top"].value_counts().items()}
        out[str(ym)] = {
            "n": int(len(g)),
            "max_day": int(g["day"].max()),
            "counts": counts,
        }
    return out


def overlay_months(base: dict[str, dict], incoming: dict[str, dict]) -> dict[str, dict]:
    """Replace months present in incoming (a newer dump of those months)."""
    merged = dict(base)
    merged.update(incoming)
    return merged


def universe_from_months(months: dict[str, dict]) -> list[str]:
    names: set[str] = set()
    for rec in months.values():
        names.update(rec.get("counts", {}))
    return sorted(names)


def _mix_from_counts(counts: dict[str, int], universe: list[str]) -> np.ndarray:
    return np.array([float(counts.get(lab, 0)) for lab in universe], dtype=np.float64)


def _sum_counts(months: dict[str, dict], yms: list[str], universe: list[str]) -> tuple[np.ndarray, int]:
    total: dict[str, int] = defaultdict(int)
    n = 0
    for ym in yms:
        rec = months.get(ym) or _empty_month()
        n += int(rec.get("n") or 0)
        for k, v in (rec.get("counts") or {}).items():
            total[k] += int(v)
    return _mix_from_counts(total, universe), n


def _ym_in_window(ym: str, start_ymd: str, end_ymd: str) -> bool:
    return start_ymd[:6] <= ym <= end_ymd[:6]


def mix_from_census(months: dict[str, dict], last_retrain_ym: str = "202511") -> dict:
    """Same locked rule as prod.live_mix, using the count table instead of rows."""
    universe = universe_from_months(months)
    win = windows_at(last_retrain_ym)
    pool_yms = [ym for ym in months if _ym_in_window(ym, win["pool_start"], win["pool_end"])]
    pool_mix, pool_n = _sum_counts(months, pool_yms, universe)

    sd_rows = []
    for ym in _months(win["last_24_start"], win["last_24_end"]):
        rec = months.get(ym) or _empty_month()
        labs = _mix_from_counts(rec.get("counts") or {}, universe)
        d = _jensen_shannon(labs, pool_mix) if rec.get("n") else None
        sd_rows.append({"month": ym, "n": int(rec.get("n") or 0), "distance_vs_pool": d})
    vals = np.array([r["distance_vs_pool"] for r in sd_rows if r["distance_vs_pool"] is not None], dtype=np.float64)
    mean = float(vals.mean()) if len(vals) else 0.0
    std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    threshold = mean + 2.0 * std

    latest = max(months) if months else last_retrain_ym
    walk_months = [m for m in _months(last_retrain_ym, latest) if m > last_retrain_ym]
    walk = []
    for ym in walk_months:
        rec = months.get(ym) or _empty_month()
        n = int(rec.get("n") or 0)
        d = _jensen_shannon(_mix_from_counts(rec.get("counts") or {}, universe), pool_mix) if n else None
        max_day = int(rec.get("max_day") or 0)
        walk.append({
            "month": ym,
            "n": n,
            "distance_vs_pool": d,
            "trip": bool(d is not None and d > threshold),
            "era": f"after_{last_retrain_ym}",
            "complete": max_day >= 28,
            "max_day": max_day or None,
        })
    complete = [r for r in walk if r["complete"]]
    fire = _first_fire(complete)
    watching = [r for r in walk if r["trip"] and not r["complete"]]
    return {
        "schema": "live_nhtsa_census_mix_v2_from_counts",
        "last_retrain_ym": last_retrain_ym,
        "windows": win,
        "pool_n": int(pool_n),
        "mean": mean,
        "std": std,
        "threshold": threshold,
        "latest_ym": latest,
        "n_over": sum(1 for r in walk if r["trip"]),
        "n_over_complete": sum(1 for r in complete if r["trip"]),
        "first_fire": fire,
        "partial_month_trips": watching,
        "walk": walk,
        "what_this_is": (
            "NHTSA top-level component field on every vehicle filing, "
            "from the monthly count table. Not model predictions. Not accuracy."
        ),
    }


def load_census(path: Path | None = None) -> dict:
    p = path or CENSUS_PATH
    return json.loads(p.read_text(encoding="utf-8"))


def save_census(months: dict[str, dict], source: str, path: Path | None = None) -> Path:
    p = path or CENSUS_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "nhtsa_month_counts_v1",
        "source": source,
        "cutoff_ldate": CUTOFF_LDATE,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "universe": universe_from_months(months),
        "months": months,
    }
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


def build_from_research_parquet(cmpl_path: Path | None = None) -> Path:
    path = cmpl_path or (DATA / "cmpl.parquet")
    df = pd.read_parquet(path, columns=["PROD_TYPE", "LDATE", "COMPDESC"])
    months = month_counts_from_frame(df)
    out = save_census(months, source=str(path))
    print(f"months={len(months)} rows={sum(m['n'] for m in months.values()):,} -> {out}")
    return out


def main():
    build_from_research_parquet()
    ensure_prod_out()


if __name__ == "__main__":
    main()
