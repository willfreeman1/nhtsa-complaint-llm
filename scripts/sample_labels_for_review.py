"""Sample rows from the cascade-labeled training parquet for a manual quality
spot-check (separate from gold-set accuracy -- this looks at real production rows the
gold set never touched).

Deliberately over-samples the highest-risk subsets rather than pure random, since a
flat random sample is dominated by easy high-confidence rows that tell us little:
    - escalated (sonnet-relabeled) rows -- where haiku itself was unsure
    - low-confidence rows (whichever model produced the final answer)
    - off-list raw answers that got bucketed to OTHER
    - a smaller plain-random slice, as a baseline read on "typical" rows

Usage:
    python scripts/sample_labels_for_review.py --seed 7
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent.parent / "output"
MAX_NARRATIVE = 1200


def sample_group(df, mask, n, seed, taken):
    pool = df[mask & ~df["cmplid"].isin(taken)]
    n = min(n, len(pool))
    return pool.sample(n=n, random_state=seed) if n else pool.iloc[0:0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-parquet", default=str(DATA_DIR / "training_labels_full_62k.parquet"))
    ap.add_argument("--n-escalated", type=int, default=12)
    ap.add_argument("--n-low-conf", type=int, default=8)
    ap.add_argument("--n-off-list", type=int, default=5)
    ap.add_argument("--n-random", type=int, default=15)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-json", default=str(OUT_DIR / "label_review_sample.json"))
    args = ap.parse_args()

    df = pd.read_parquet(args.train_parquet)
    df = df[df["component_label"].notna() & df["error"].isna()].reset_index(drop=True)

    taken = set()
    groups = []

    g = sample_group(df, df["escalated"], args.n_escalated, args.seed, taken)
    taken |= set(g["cmplid"])
    groups.append(("escalated", g))

    g = sample_group(df, df["haiku_confidence"] == "low", args.n_low_conf, args.seed + 1, taken)
    taken |= set(g["cmplid"])
    groups.append(("low_confidence", g))

    g = sample_group(df, df["off_list"], args.n_off_list, args.seed + 2, taken)
    taken |= set(g["cmplid"])
    groups.append(("off_list", g))

    g = sample_group(df, pd.Series(True, index=df.index), args.n_random, args.seed + 3, taken)
    taken |= set(g["cmplid"])
    groups.append(("random", g))

    review_rows = []
    for tag, g in groups:
        for _, r in g.iterrows():
            review_rows.append({
                "sample_reason": tag,
                "cmplid": r["cmplid"], "make": r["make"], "model": r["model"], "year": r["year"],
                "narrative": r["narrative"],
                "labeling_model": r["labeling_model"],
                "escalated": bool(r["escalated"]),
                "haiku_confidence": r["haiku_confidence"],
                "raw_component": r["raw_component"],
                "component_label": r["component_label"],
                "off_list": bool(r["off_list"]),
                "crash": r["crash"], "fire": r["fire"],
                "injured": int(r["injured"]), "deaths": int(r["deaths"]),
            })

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"n": len(review_rows), "seed": args.seed, "rows": review_rows}, f, indent=2)
    print(f"Wrote {len(review_rows)} rows to {args.out_json}", file=sys.stderr)

    for tag, g in groups:
        print(f"\n===== {tag.upper()} (n={len(g)}) =====")
        for _, r in g.iterrows():
            text = " ".join(str(r["narrative"]).split())
            if len(text) > MAX_NARRATIVE:
                text = text[:MAX_NARRATIVE] + " ...[TRUNCATED]"
            print(f"--- cmplid={r['cmplid']} | {r['year']} {r['make']} {r['model']}")
            print(f"    LABEL: {r['component_label']}"
                  + (f"  (raw: {r['raw_component']})" if r["raw_component"] != r["component_label"] else "")
                  + f"  | model={r['labeling_model']} escalated={r['escalated']} "
                  f"haiku_conf={r['haiku_confidence']} off_list={r['off_list']}")
            print(f"    {text}")
            print()


if __name__ == "__main__":
    main()
