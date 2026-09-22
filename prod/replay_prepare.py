"""Build dated teacher labels and locked replay samples. Local, no GPU, no APIs.

Usage (from repo root):
    python -m prod.replay_prepare
"""
from __future__ import annotations

import json

import pandas as pd

from prod.paths import DATA, GOLD_V3, PROD_OUT, ensure_prod_out

CUTOFF = "20191231"
SEED = 20260921
WALK_START = "202001"
WALK_END = "202608"
N_PER_MONTH = 2000
TEACHER = DATA / "training_labels_full_62k_plus_rare.parquet"
REPLAY_DIR = DATA / "replay"

MERGE_MAP = {
    "ELECTRONIC STABILITY CONTROL (ESC)": "ELECTRONIC STABILITY CONTROL",
    "COMMUNICATIONS": "COMMUNICATION",
}


def _months(start: str, end: str) -> list[str]:
    out = []
    y, m = int(start[:4]), int(start[4:6])
    y2, m2 = int(end[:4]), int(end[4:6])
    while (y, m) <= (y2, m2):
        out.append(f"{y}{m:02d}")
        m += 1
        if m == 13:
            m = 1
            y += 1
    return out


def _nhtsa_top(s) -> str:
    top = str(s).split(":")[0].strip()
    return MERGE_MAP.get(top, top)


def _gold_ids() -> set[str]:
    ids = {str(r["cmplid"]) for r in json.loads(GOLD_V3.read_text(encoding="utf-8"))["rows"]}
    cur_path = PROD_OUT / "gold_current_v1.json"
    if cur_path.exists():
        ids |= {str(r["cmplid"]) for r in json.loads(cur_path.read_text(encoding="utf-8"))["rows"]}
    return ids


def main():
    out = ensure_prod_out() / "replay"
    out.mkdir(parents=True, exist_ok=True)
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)

    gold_ids = _gold_ids()
    print(f"Blocked gold IDs: {len(gold_ids):,}")

    cmpl = pd.read_parquet(
        DATA / "cmpl.parquet",
        columns=["CMPLID", "PROD_TYPE", "LDATE", "MAKETXT", "MODELTXT", "YEARTXT", "COMPDESC", "CDESCR"],
    )
    cmpl["CMPLID"] = cmpl["CMPLID"].astype(str)
    veh = cmpl[cmpl["PROD_TYPE"] == "V"].copy()
    veh["ldate"] = veh["LDATE"].astype(str)
    veh["ym"] = veh["ldate"].str[:6]
    veh["nhtsa_top"] = veh["COMPDESC"].map(_nhtsa_top)
    print(f"Vehicle rows: {len(veh):,}")

    teacher = pd.read_parquet(TEACHER)
    teacher["cmplid"] = teacher["cmplid"].astype(str)
    dated = teacher.merge(veh[["CMPLID", "ldate"]], left_on="cmplid", right_on="CMPLID", how="left")
    n_miss = int(dated["ldate"].isna().sum())
    n_gold = int(dated["cmplid"].isin(gold_ids).sum())
    dated = dated[dated["ldate"].notna() & ~dated["cmplid"].isin(gold_ids)].copy()
    dated_path = REPLAY_DIR / "teacher_dated.parquet"
    dated.to_parquet(dated_path, index=False)
    asof = dated[dated["ldate"] <= CUTOFF].copy()
    asof_path = REPLAY_DIR / "asof_2019.parquet"
    asof.to_parquet(asof_path, index=False)
    print(
        f"Teacher dated {len(dated):,} (dropped missing_date={n_miss} gold={n_gold})  "
        f"as-of-2019 {len(asof):,}"
    )

    by_id_teacher = dated.set_index("cmplid")["raw_component"]

    months = ["2019"] + _months(WALK_START, WALK_END)
    frames = []
    summary_months = []
    for token in months:
        if token == "2019":
            pool = veh[veh["ym"].str.startswith("2019")]
            ym = "2019"
        else:
            pool = veh[veh["ym"] == token]
            ym = token
        n_pool = int(len(pool))
        n_take = min(N_PER_MONTH, n_pool)
        if n_take == 0:
            summary_months.append({"month": ym, "pool": 0, "sampled": 0})
            continue
        sample = pool.sample(n=n_take, random_state=SEED)
        rec = pd.DataFrame({
            "month": ym,
            "cmplid": sample["CMPLID"].astype(str).values,
            "ldate": sample["ldate"].values,
            "make": sample["MAKETXT"].values,
            "model": sample["MODELTXT"].values,
            "year": sample["YEARTXT"].astype(str).values,
            "narrative": sample["CDESCR"].astype(str).values,
            "nhtsa_top": sample["nhtsa_top"].values,
        })
        rec["teacher_raw"] = rec["cmplid"].map(by_id_teacher)
        frames.append(rec)
        summary_months.append({
            "month": ym,
            "pool": n_pool,
            "sampled": n_take,
            "with_teacher": int(rec["teacher_raw"].notna().sum()),
        })
        print(f"  {ym} pool={n_pool:,} sampled={n_take} teacher={int(rec['teacher_raw'].notna().sum())}")

    samples = pd.concat(frames, ignore_index=True)
    samples_path = out / "walk_samples.parquet"
    samples.to_parquet(samples_path, index=False)

    gold = json.loads(GOLD_V3.read_text(encoding="utf-8"))
    dates = veh.set_index("CMPLID")["ldate"]
    through, after, no_date = [], [], []
    for row in gold["rows"]:
        cid = str(row["cmplid"])
        ld = dates.get(cid)
        attached = {**row, "ldate": None if pd.isna(ld) else str(ld)}
        if attached["ldate"] is None:
            no_date.append(attached)
        elif attached["ldate"] <= CUTOFF:
            through.append(attached)
        else:
            after.append(attached)
    (out / "gold_v3_through_2019.json").write_text(
        json.dumps({"n": len(through), "cutoff": CUTOFF, "rows": through}, indent=2),
        encoding="utf-8",
    )
    (out / "gold_v3_after_2019.json").write_text(
        json.dumps({"n": len(after), "cutoff": CUTOFF, "rows": after}, indent=2),
        encoding="utf-8",
    )

    meta = {
        "cutoff": CUTOFF,
        "seed": SEED,
        "n_per_month": N_PER_MONTH,
        "walk_start": WALK_START,
        "walk_end": WALK_END,
        "n_teacher_dated": int(len(dated)),
        "n_asof_2019": int(len(asof)),
        "n_gold_blocked": len(gold_ids),
        "n_gold_v3_through_2019": len(through),
        "n_gold_v3_after_2019": len(after),
        "n_gold_v3_no_date": len(no_date),
        "months": summary_months,
        "files": {
            "teacher_dated": str(dated_path),
            "asof_2019": str(asof_path),
            "walk_samples": str(samples_path),
        },
    }
    (out / "prepare_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in (
        "n_asof_2019", "n_gold_v3_through_2019", "n_gold_v3_after_2019", "n_gold_v3_no_date"
    )}, indent=2))
    print(f"Wrote {out / 'prepare_summary.json'}")


if __name__ == "__main__":
    main()
