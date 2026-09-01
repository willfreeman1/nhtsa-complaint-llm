"""Print pilot teacher/gold disagreements not yet in disagreement_adjudication_n40.json.

The 500-row Anthropic pilot found 254 rows where the teacher (claude-haiku-4-5) disagreed
with NHTSA's gold COMPDESC. Only 40 of those got a careful manual read (see
disagreement_adjudication_n40.json). This dumps the other ~218 so they can get the same
treatment -- adjudicating each of them (not just the 40) is required before the gold eval
set can be called a real sample rather than a hand-picked easy subset.

    python scripts/dump_disagreements_for_adjudication.py --start 0 --count 40
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent.parent / "output"
MAX_NARRATIVE = 1400


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=40)
    args = ap.parse_args()

    pilot = json.load(open(OUT_DIR / "teacher_labels_anthropic_500.json"))["results"]
    adjudicated_ids = {r["cmplid"] for r in json.load(open(OUT_DIR / "disagreement_adjudication_n40.json"))["rows"]}
    disagreements = [r for r in pilot if r["pred_component_bucketed"] != r["gold_COMPDESC_LABEL"]]
    remaining = [r for r in disagreements if r["cmplid"] not in adjudicated_ids]

    narratives = pd.read_parquet(
        DATA_DIR / "cmpl.parquet", columns=["CMPLID", "CDESCR", "MAKETXT", "MODELTXT", "YEARTXT"]
    ).set_index("CMPLID")

    rows = remaining[args.start: args.start + args.count]
    for i, r in enumerate(rows, start=args.start):
        cid = r["cmplid"]
        meta = narratives.loc[str(cid)]
        text = " ".join(str(meta["CDESCR"]).split())
        if len(text) > MAX_NARRATIVE:
            text = text[:MAX_NARRATIVE] + " ...[TRUNCATED]"
        print(f"--- [{i}] cmplid={cid} | {meta['YEARTXT']} {meta['MAKETXT']} {meta['MODELTXT']}")
        print(f"    GOLD={r['gold_COMPDESC_LABEL']}  TEACHER={r['pred_component_bucketed']}  "
              f"(raw: {r.get('pred_component')})")
        print(f"    {text}")
        print()
    print(f"(showed {args.start}-{args.start + len(rows) - 1} of {len(remaining)} unadjudicated disagreements)")


if __name__ == "__main__":
    main()
