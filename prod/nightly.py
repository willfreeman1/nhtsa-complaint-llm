"""Slim daily job for GitHub Actions: pull, score new rows, mix-check, write a summary.

Does not need the 2-million-row history file or a 1.6-million fingerprint ledger.
A filing is new if NHTSA received it after the research cutoff (26 Aug 2026)
and we have not hashed it yet. The mix check uses the committed monthly
count table, with 2025–2026 months replaced from today's zip.

Usage (from repo root):
    python -m prod.nightly build-census   # once, needs data/cmpl.parquet
    python -m prod.nightly seed-seen      # once, optional, needs a prior pull
    python -m prod.nightly fetch-weights  # Actions / any machine without weights
    python -m prod.nightly run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from prod.census import (
    CENSUS_PATH,
    CUTOFF_LDATE,
    NIGHTLY_DIR,
    load_census,
    mix_from_census,
    month_counts_from_frame,
    overlay_months,
    save_census,
)
from prod.ingest import RECENT_URL, _find_txt, _load_flat, _probe, cmd_download, row_fingerprint
from prod.live_mix import LAST_RETRAIN_YM
from prod.loop import CHAMPION, _decide
from prod.paths import DEBERTA_CKPT, INCOMING, ROOT, ensure_prod_out

SEEN_PATH = NIGHTLY_DIR / "seen_after_cutoff.txt"
WEIGHTS_TAG = os.environ.get("NHTSA_WEIGHTS_RELEASE", "deberta-champion")
WEIGHT_FILES = ("model.safetensors", "config.json")
NIGHTLY_JSON = "nightly.json"


def _hash_fp(fp: str) -> str:
    return hashlib.sha256(fp.encode("utf-8")).hexdigest()


def load_seen(path: Path | None = None) -> set[str]:
    p = path or SEEN_PATH
    if not p.exists():
        return set()
    return {line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()}


def save_seen(keys: set[str], path: Path | None = None) -> Path:
    p = path or SEEN_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(sorted(keys)) + ("\n" if keys else ""), encoding="utf-8")
    return p


def cmd_build_census(_args):
    from prod.census import build_from_research_parquet

    build_from_research_parquet()


def cmd_seed_seen(_args):
    """Hash post-cutoff fingerprints from a local incoming pull, if one exists."""
    import pandas as pd

    NIGHTLY_DIR.mkdir(parents=True, exist_ok=True)
    seen = load_seen()
    incoming = INCOMING / "new_vehicle_latest.parquet"
    if not incoming.exists():
        print(f"No {incoming}; writing empty seen file with {len(seen)} hashes")
        save_seen(seen)
        return
    df = pd.read_parquet(incoming)
    if "fingerprint" not in df.columns:
        df["fingerprint"] = row_fingerprint(df)
    post = df[df["LDATE"].astype(str) > CUTOFF_LDATE]
    before = len(seen)
    seen |= {_hash_fp(str(fp)) for fp in post["fingerprint"]}
    save_seen(seen)
    print(f"seeded {len(seen) - before} hashes from {len(post)} post-cutoff rows -> {SEEN_PATH} (total {len(seen)})")


def _release_base() -> str:
    repo = os.environ.get("GITHUB_REPOSITORY", "willfreeman1/nhtsa-complaint-llm")
    return f"https://github.com/{repo}/releases/download/{WEIGHTS_TAG}"


def cmd_fetch_weights(_args):
    dest = DEBERTA_CKPT
    dest.mkdir(parents=True, exist_ok=True)
    existing = dest / "model.safetensors"
    if existing.exists() and existing.stat().st_size > 50_000_000:
        print(f"Weights already present ({existing.stat().st_size / 1e6:.1f} MB)")
        return
    dest.mkdir(parents=True, exist_ok=True)
    for name in WEIGHT_FILES:
        url = f"{_release_base()}/{name}"
        out = dest / name
        print(f"GET {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "nhtsa-complaint-llm-nightly"})
        with urllib.request.urlopen(req, timeout=600) as resp, open(out, "wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        print(f"  wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    weights = dest / "model.safetensors"
    if not weights.exists() or weights.stat().st_size < 50_000_000:
        raise SystemExit(f"Download failed or too small: {weights}")


def _download_recent(force: bool) -> tuple[dict, Path]:
    probe = _probe(RECENT_URL)
    last_mod = (probe.get("headers") or {}).get("Last-Modified")
    args = argparse.Namespace(kind="recent", force=force)
    meta, extract_dir = cmd_download(args)
    meta["last_modified"] = meta.get("last_modified") or last_mod
    return meta, extract_dir


def cmd_run(args):
    import pandas as pd

    from prod.score_incoming import score_incoming

    out_dir = ensure_prod_out()
    NIGHTLY_DIR.mkdir(parents=True, exist_ok=True)
    ran_at = datetime.now(timezone.utc).isoformat()
    seen = load_seen()
    census = load_census()
    months = census["months"]

    print("Downloading 2025–2026 NHTSA chunk...")
    meta, extract_dir = _download_recent(force=args.force)
    txt = _find_txt(extract_dir)
    keep = [
        "CMPLID", "ODINO", "PROD_TYPE", "MAKETXT", "MODELTXT", "YEARTXT",
        "COMPDESC", "CDESCR", "DATEA", "LDATE",
    ]
    print(f"Loading {txt} ...")
    fresh = _load_flat(txt, columns=keep)
    fresh["fingerprint"] = row_fingerprint(fresh)
    veh = fresh[fresh["PROD_TYPE"] == "V"].copy()
    print(f"  dump vehicle rows: {len(veh):,}")

    overlay = month_counts_from_frame(veh)
    months = overlay_months(months, overlay)
    save_census(months, source=f"research + {meta.get('url')}")

    post = veh[veh["LDATE"].astype(str) > CUTOFF_LDATE].copy()
    post["fp_hash"] = post["fingerprint"].map(_hash_fp)
    new = post[~post["fp_hash"].isin(seen)].copy()
    print(f"  post-cutoff: {len(post):,}  new hashes: {len(new):,}  known hashes: {len(seen):,}")

    INCOMING.mkdir(parents=True, exist_ok=True)
    new_path = INCOMING / "new_vehicle_latest.parquet"
    new.drop(columns=["fp_hash"], errors="ignore").to_parquet(new_path, index=False)

    score = {"n": 0, "note": "no new post-cutoff rows"}
    if len(new) and not args.skip_score:
        score = score_incoming(new_path)
    elif args.skip_score:
        score = {"n": int(len(new)), "note": "score skipped"}

    seen |= set(post["fp_hash"])
    save_seen(seen)

    mix = mix_from_census(months, last_retrain_ym=LAST_RETRAIN_YM)
    (out_dir / "live_mix.json").write_text(json.dumps(mix, indent=2), encoding="utf-8")
    decision = _decide(mix, allow_retrain=False)

    nightly = {
        "schema": "nightly_v1",
        "ran_at": ran_at,
        "host": "github-actions" if os.environ.get("GITHUB_ACTIONS") else "local",
        "cutoff_ldate": CUTOFF_LDATE,
        "champion": {
            "model": CHAMPION["model"],
            "checkpoint": "checkpoints/deberta_rung2_full_62k_plus_rare/final",
            "why": CHAMPION["why"],
        },
        "pull": {
            "url": meta.get("url"),
            "last_modified": meta.get("last_modified"),
            "dump_vehicle": int(len(veh)),
            "post_cutoff": int(len(post)),
            "new_vehicle": int(len(new)),
            "ldate_min": str(new["LDATE"].min()) if len(new) else None,
            "ldate_max": str(new["LDATE"].max()) if len(new) else None,
        },
        "score": {
            "n": score.get("n"),
            "unknown_rate": score.get("unknown_rate"),
            "median_confidence": score.get("median_confidence"),
            "gadget_rate": score.get("gadget_rate"),
            "nhtsa_field_agreement": score.get("nhtsa_field_agreement"),
            "nhtsa_field_agreement_is_not_accuracy": True,
            "note": score.get("note"),
        },
        "mix": {
            "threshold": mix.get("threshold"),
            "latest_ym": mix.get("latest_ym"),
            "n_over": mix.get("n_over"),
            "first_fire": mix.get("first_fire"),
            "partial_month_trips": [
                {"month": r["month"], "n": r["n"], "distance_vs_pool": r["distance_vs_pool"]}
                for r in (mix.get("partial_month_trips") or [])
            ],
            "what_this_is": mix.get("what_this_is"),
        },
        "decision": {
            "retrain": decision.get("retrain"),
            "promote": decision.get("promote"),
            "reason": decision.get("reason"),
        },
    }
    path = out_dir / NIGHTLY_JSON
    path.write_text(json.dumps(nightly, indent=2), encoding="utf-8")
    print(json.dumps({
        "new_vehicle": nightly["pull"]["new_vehicle"],
        "scored": nightly["score"]["n"],
        "mix_fire": nightly["mix"]["first_fire"],
        "retrain": nightly["decision"]["retrain"],
        "wrote": str(path),
    }, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("build-census").set_defaults(func=cmd_build_census)
    sub.add_parser("seed-seen").set_defaults(func=cmd_seed_seen)
    sub.add_parser("fetch-weights").set_defaults(func=cmd_fetch_weights)
    p_run = sub.add_parser("run")
    p_run.add_argument("--force", action="store_true")
    p_run.add_argument("--skip-score", action="store_true")
    p_run.set_defaults(func=cmd_run)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
