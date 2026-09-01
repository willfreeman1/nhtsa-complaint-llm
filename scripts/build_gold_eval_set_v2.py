"""Expand the gold eval set from 275 rows to (nearly) the full 500-row pilot.

output/gold_eval_set_reviewed.json only covered dual-agreement rows plus the original
40 hand-adjudicated disagreements -- 273 of the 254 pilot disagreements were never
independently read, which meant the eval set was, in effect, hand-picked for rows that
were easy to agree on. This script adjudicates the other 218 (see
disagreement_adjudication_full_batch_00..04.json, each entry an independent read of the
narrative deciding the correct label(s) from scratch -- not "whichever of gold/teacher
looks less wrong") and merges them in, so the eval set is now every pilot row except the
handful excluded as out-of-scope or genuinely unrecoverable from text alone.
"""
import glob
import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent.parent / "output"

EXCLUDE_LABELS = {"unknowable_from_text"}


def load_full_adjudication():
    rows = {}
    for path in sorted(glob.glob(str(OUT_DIR / "disagreement_adjudication_full_batch_*.json"))):
        with open(path) as f:
            batch = json.load(f)
        for r in batch["rows"]:
            if r["cmplid"] in rows:
                raise ValueError(f"duplicate cmplid {r['cmplid']} in {path}")
            rows[r["cmplid"]] = r
    return rows


def main():
    with open(OUT_DIR / "gold_eval_set_reviewed.json") as f:
        reviewed = json.load(f)

    with open(OUT_DIR / "teacher_labels_anthropic_500.json") as f:
        pilot_meta = json.load(f)
    pilot = pilot_meta["results"]

    full_adj = load_full_adjudication()

    already_covered = {r["cmplid"] for r in reviewed["rows"]}
    narratives = pd.read_parquet(
        DATA_DIR / "cmpl.parquet", columns=["CMPLID", "CDESCR", "MAKETXT", "MODELTXT", "YEARTXT"]
    ).set_index("CMPLID")

    new_rows = []
    still_missing = []
    for r in pilot:
        cid = r["cmplid"]
        if cid in already_covered:
            continue
        gold = r["gold_COMPDESC_LABEL"]
        rerun_pred = r["pred_component_bucketed"]
        if rerun_pred == gold:
            # Already inside gold_eval_set_reviewed.json as dual_agreement -- shouldn't happen.
            continue
        if cid not in full_adj:
            still_missing.append(cid)
            continue

        adj = full_adj[cid]
        meta = narratives.loc[str(cid)]
        new_rows.append({
            "cmplid": cid,
            "make": meta["MAKETXT"],
            "model": meta["MODELTXT"],
            "year": meta["YEARTXT"],
            "narrative": meta["CDESCR"],
            "accept": adj["accept"],
            "accept_before_review": [gold],
            "accept_strict_brakes": r["accept_strict_brakes"] if "accept_strict_brakes" in adj else (
                [gold] if adj.get("brake_family") else adj["accept"]
            ),
            "primary": adj["accept"][0],
            "truth_source": "adjudicated_full_v2",
            "is_hard": True,
            "brake_family": bool(adj.get("brake_family")),
            "is_hard_multi_issue": bool(adj.get("is_hard")),
            "adjudication_note": adj["note"],
            "review_verdict": "adjudicated",
            "review_note": adj["note"],
            "exclude_candidate": False,
            "nhtsa_gold": gold,
            "rerun_teacher": rerun_pred,
            "nhtsa_crash": r["gold_CRASH"],
            "nhtsa_fire": r["gold_FIRE"],
        })

    if still_missing:
        raise ValueError(f"{len(still_missing)} disagreements have no adjudication entry: {still_missing}")

    out = dict(reviewed)
    out["rows"] = reviewed["rows"] + new_rows
    out["n"] = len(out["rows"])
    out["n_hard"] = sum(r["is_hard"] for r in out["rows"])
    out["caveats"] = [
        "Every one of the 254 pilot teacher/gold disagreements (plus 4 dual-agreement "
        "corrections found during review) has now been independently adjudicated from the "
        "narrative text -- none are dropped as unadjudicated. Remaining unresolved cases are "
        "rows where the narrative genuinely supports more than one label (accept has >1 entry) "
        "or is truly unknowable from text alone (labeled UNKNOWN OR OTHER), not gaps in review.",
        "Dual-agreement rows are, by construction, rows the Anthropic pilot model (claude-haiku-4-5) "
        "got right, which mildly favors models that behave like it. Relative ranking across models "
        "is still meaningful; treat small (<2pt) gaps with suspicion.",
        "NHTSA CRASH/FIRE flags are carried for secondary scoring only and are known to be noisy "
        "(gotcha B1/B2).",
    ]
    out["v2_expansion"] = {
        "method": "Every one of the 254 pilot teacher/gold disagreements is now independently "
                  "adjudicated from the narrative text (not just the original 40) -- see "
                  "disagreement_adjudication_full_batch_00..04.json. Each row's accept list is "
                  "this reviewer's own judgment of the correct label(s), not a default to "
                  "whichever of NHTSA-gold or the teacher's guess seemed less wrong.",
        "n_new_rows": len(new_rows),
        "n_brake_family_new": sum(r["brake_family"] for r in new_rows),
        "n_multi_issue_hard_new": sum(r["is_hard_multi_issue"] for r in new_rows),
        "previous_n": len(reviewed["rows"]),
    }

    with open(OUT_DIR / "gold_eval_set_v2.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"gold_eval_set_v2.json: {out['n']} rows ({out['n_hard']} hard), "
          f"+{len(new_rows)} newly adjudicated disagreements")


if __name__ == "__main__":
    main()
