"""Re-read dump-vs-dump changes on a stable key, not CMPLID.

CMPLID is documented as updateable. The first compare showed narratives
sliding to neighboring IDs, so a CMPLID join measures remapping, not edits.
This audit fingerprints a row as (ODINO, first 400 chars of CDESCR) and
reports how often the same complaint changed COMPDESC / make / CMPLID.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from prod.ingest import _load_flat
from prod.paths import DATA, ensure_prod_out


def _fp(df: pd.DataFrame) -> pd.Series:
    narr = df["CDESCR"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip().str[:400]
    return df["ODINO"].astype(str) + "||" + narr


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    args = ap.parse_args()

    keep = ["CMPLID", "ODINO", "PROD_TYPE", "MAKETXT", "MODELTXT", "YEARTXT",
            "COMPDESC", "CDESCR", "LDATE"]
    print(f"Loading {args.new} ...")
    new = _load_flat(Path(args.new), columns=keep)
    print("Loading data/cmpl.parquet ...")
    old = pd.read_parquet(DATA / "cmpl.parquet", columns=keep)
    for c in keep:
        old[c] = old[c].astype(str)
        new[c] = new[c].astype(str)

    window = (new["LDATE"].min(), new["LDATE"].max())
    old_w = old[(old["LDATE"] >= window[0]) & (old["LDATE"] <= window[1])].copy()
    new = new.copy()
    old_w["fp"] = _fp(old_w)
    new["fp"] = _fp(new)

    old_u = old_w.drop_duplicates("fp", keep="first").set_index("fp")
    new_u = new.drop_duplicates("fp", keep="first").set_index("fp")
    common = old_u.index.intersection(new_u.index)

    def n_changed(col):
        return int((old_u.loc[common, col] != new_u.loc[common, col]).sum())

    # Same narrative/ODINO, different CMPLID = remapping, not an edit.
    cmplid_moved = int((old_u.loc[common, "CMPLID"] != new_u.loc[common, "CMPLID"]).sum())
    comp_changed = old_u.loc[common, "COMPDESC"] != new_u.loc[common, "COMPDESC"]
    n_comp = int(comp_changed.sum())
    examples = []
    if n_comp:
        sample = (
            new_u.loc[common][comp_changed][["CMPLID", "ODINO", "COMPDESC", "MAKETXT", "CDESCR"]]
            .join(old_u.loc[common][comp_changed][["CMPLID", "COMPDESC", "MAKETXT"]],
                  lsuffix="_new", rsuffix="_old")
            .head(6)
        )
        examples = sample.reset_index(drop=True).to_dict(orient="records")
        for ex in examples:
            ex["CDESCR"] = str(ex.get("CDESCR", ""))[:180]

    only_new_fp = int(len(new_u.index.difference(old_u.index)))
    only_old_fp = int(len(old_u.index.difference(new_u.index)))

    out = {
        "window": {"ldate_min": window[0], "ldate_max": window[1]},
        "old_rows_in_window": int(len(old_w)),
        "new_rows": int(len(new)),
        "common_fingerprints": int(len(common)),
        "fingerprints_only_in_new": only_new_fp,
        "fingerprints_only_in_old": only_old_fp,
        "same_complaint_cmplid_moved": cmplid_moved,
        "same_complaint_COMPDESC_changed": n_comp,
        "same_complaint_MAKETXT_changed": n_changed("MAKETXT"),
        "same_complaint_MODELTXT_changed": n_changed("MODELTXT"),
        "same_complaint_YEARTXT_changed": n_changed("YEARTXT"),
        "compdesc_change_examples": examples,
        "note": (
            "Fingerprint is ODINO + first 400 chars of whitespace-normalized CDESCR. "
            "CMPLID movement on a matching fingerprint is remapping, not a content edit."
        ),
    }
    path = ensure_prod_out() / "revision_audit.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "compdesc_change_examples"}, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
