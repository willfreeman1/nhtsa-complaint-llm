"""Download the current NHTSA complaints dump and compare it to the research copy.

Usage (from repo root):
    python -m prod.ingest headers
    python -m prod.ingest download --kind recent
    python -m prod.ingest download --kind full
    python -m prod.ingest compare --new data/incoming/FLAT_CMPL_current.txt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from prod.paths import DATA, INCOMING, ROOT, ensure_prod_out

FLAT_URL = "https://static.nhtsa.gov/odi/ffdd/cmpl/FLAT_CMPL.zip"
RECENT_URL = "https://static.nhtsa.gov/odi/ffdd/cmpl/COMPLAINTS_RECEIVED_2025-2026.zip"
PAGE_URL = "https://www.nhtsa.gov/nhtsa-datasets-and-apis"

HEADERS_OF_INTEREST = (
    "Last-Modified",
    "Content-Length",
    "ETag",
    "Content-Type",
    "Date",
)


def _probe(url: str) -> dict:
    req = urllib.request.Request(url, method="HEAD")
    req.add_header("User-Agent", "nhtsa-complaint-llm-stage0/0.1")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            info = {k: resp.headers.get(k) for k in HEADERS_OF_INTEREST}
            return {"url": url, "status": resp.status, "headers": info}
    except Exception as exc:  # noqa: BLE001
        return {"url": url, "status": None, "error": str(exc)}


def cmd_headers(_args):
    out = {
        "probed_at": datetime.now(timezone.utc).isoformat(),
        "files": [_probe(FLAT_URL), _probe(RECENT_URL)],
        "page": PAGE_URL,
        "notes": (
            "NHTSA's CMPL.txt says the file is published daily and that CMPLID "
            "and component names can change between dumps. HEAD Last-Modified "
            "is the ground truth for 'how often', not the datasets page copy."
        ),
    }
    path = ensure_prod_out() / "nhtsa_headers.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"Wrote {path}")


def cmd_download(args):
    INCOMING.mkdir(parents=True, exist_ok=True)
    url = RECENT_URL if args.kind == "recent" else FLAT_URL
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    zip_name = f"{'COMPLAINTS_RECEIVED_2025-2026' if args.kind == 'recent' else 'FLAT_CMPL'}_{stamp}.zip"
    zip_path = INCOMING / zip_name
    print(f"GET {url}")
    print(f" -> {zip_path}")
    t0 = time.perf_counter()
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "nhtsa-complaint-llm-stage0/0.1")
    with urllib.request.urlopen(req, timeout=300) as resp:
        last_mod = resp.headers.get("Last-Modified")
        total = resp.headers.get("Content-Length")
        print(f"Last-Modified: {last_mod}  Content-Length: {total}")
        tmp = zip_path.with_suffix(".zip.part")
        n = 0
        with open(tmp, "wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                n += len(chunk)
                if n % (20 * 1024 * 1024) < 1024 * 1024:
                    print(f"  {n / 1e6:.0f} MB...")
        tmp.replace(zip_path)
    print(f"Downloaded {n / 1e6:.1f} MB in {time.perf_counter() - t0:.1f}s")

    extract_dir = INCOMING / zip_path.stem
    extract_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
        names = zf.namelist()
    print(f"Extracted {names} -> {extract_dir}")
    meta = {
        "url": url,
        "kind": args.kind,
        "zip_path": str(zip_path),
        "extract_dir": str(extract_dir),
        "names": names,
        "bytes": n,
        "last_modified": last_mod,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }
    (ensure_prod_out() / f"download_{args.kind}.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, indent=2))


def _load_flat(path: Path, columns=None):
    import csv

    import pandas as pd

    sys.path.insert(0, str(ROOT / "scripts"))
    from schema import COLUMNS

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=COLUMNS,
        usecols=columns or COLUMNS,
        encoding="latin-1",
        dtype=str,
        na_values=[],
        keep_default_na=False,
        quoting=csv.QUOTE_NONE,
        on_bad_lines="warn",
    )
    return df


def cmd_compare(args):
    import pandas as pd

    new_path = Path(args.new)
    print(f"Loading new dump {new_path} ...")
    t0 = time.perf_counter()
    keep = [
        "CMPLID", "ODINO", "PROD_TYPE", "MAKETXT", "MODELTXT", "YEARTXT",
        "COMPDESC", "CDESCR", "DATEA", "LDATE", "FAILDATE",
    ]
    new = _load_flat(new_path, columns=keep)
    print(f"  {len(new):,} rows in {time.perf_counter() - t0:.1f}s")

    print("Loading research copy data/cmpl.parquet ...")
    old = pd.read_parquet(
        DATA / "cmpl.parquet",
        columns=keep,
    )
    for col in keep:
        old[col] = old[col].astype(str)
        new[col] = new[col].astype(str)

    old_ids = set(old["CMPLID"])
    new_ids = set(new["CMPLID"])
    added = new_ids - old_ids
    # A 5-year chunk is not a full dump. Only treat IDs in the new file's
    # received-date window as comparable for removals/revisions.
    new_ldate_min = str(new["LDATE"].min())
    new_ldate_max = str(new["LDATE"].max())
    old_in_window = old[
        (old["LDATE"] >= new_ldate_min) & (old["LDATE"] <= new_ldate_max)
    ]
    old_window_ids = set(old_in_window["CMPLID"])
    removed = old_window_ids - new_ids
    common = old_window_ids & new_ids
    scoped_to_window = len(old_in_window) < len(old)

    old_i = old.set_index("CMPLID")
    new_i = new.set_index("CMPLID")
    # Align on common IDs only; compare fields that the brief cares about.
    common_index = old_i.index.intersection(new_i.index)
    old_c = old_i.loc[common_index]
    new_c = new_i.loc[common_index]
    changed = {}
    for field in ("COMPDESC", "CDESCR", "DATEA", "LDATE", "MAKETXT", "MODELTXT", "YEARTXT"):
        mask = old_c[field] != new_c[field]
        n_chg = int(mask.sum())
        examples = []
        if n_chg:
            sample = new_c.loc[mask, [field]].join(
                old_c.loc[mask, [field]], lsuffix="_new", rsuffix="_old"
            ).head(8)
            examples = sample.reset_index().to_dict(orient="records")
        changed[field] = {"n": n_chg, "examples": examples}

    added_df = new[new["CMPLID"].isin(added)]
    added_v = added_df[added_df["PROD_TYPE"] == "V"]
    new_v = new[new["PROD_TYPE"] == "V"]

    summary = {
        "old_n": int(len(old)),
        "new_n": int(len(new)),
        "old_n_in_new_file_ldate_window": int(len(old_in_window)),
        "compare_scoped_to_new_file_window": bool(scoped_to_window),
        "window_ldate_min": new_ldate_min,
        "window_ldate_max": new_ldate_max,
        "added": int(len(added)),
        "removed": int(len(removed)),
        "common": int(len(common)),
        "added_vehicle": int(len(added_v)),
        "new_vehicle": int(len(new_v)),
        "old_ldate_max": str(old["LDATE"].max()),
        "new_ldate_max": str(new["LDATE"].max()),
        "old_datea_max": str(old["DATEA"].max()),
        "new_datea_max": str(new["DATEA"].max()),
        "added_ldate_min": str(added_df["LDATE"].min()) if len(added_df) else None,
        "added_ldate_max": str(added_df["LDATE"].max()) if len(added_df) else None,
        "field_changes_on_common_ids": changed,
        "new_path": str(new_path),
    }
    path = ensure_prod_out() / "ingest_compare.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "field_changes_on_common_ids"}, indent=2))
    print("Field changes on common CMPLID:")
    for field, info in changed.items():
        print(f"  {field}: {info['n']:,}")
    print(f"Wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_h = sub.add_parser("headers")
    p_h.set_defaults(func=cmd_headers)

    p_d = sub.add_parser("download")
    p_d.add_argument("--kind", choices=["recent", "full"], default="recent")
    p_d.set_defaults(func=cmd_download)

    p_c = sub.add_parser("compare")
    p_c.add_argument("--new", required=True, help="Path to extracted FLAT_CMPL.txt (or 5-year txt)")
    p_c.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
