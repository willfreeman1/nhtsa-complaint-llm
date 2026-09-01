"""Build a curated ground-truth eval set that does NOT treat raw NHTSA COMPDESC as truth.

Motivation (see output/NHTSA_CODING_GOTCHAS.md B1-B15 and
output/disagreement_adjudication_n40.json): hand-adjudicating 40 teacher/gold
disagreements from the 500-row Anthropic pilot found only 1/40 (2.5%) were genuine
teacher errors -- the rest were NHTSA gold noise, fuzzy system boundaries, genuinely
multi-issue narratives, or labels that depend on a linked recall campaign rather than
the narrative. So "agreement with raw gold" measures label noise as much as model
quality, and is useless for ranking models.

This script assembles a set where every row's truth comes from one of two much stronger
signals:

  1. dual_agreement -- the teacher LLM and NHTSA independently landed on the same label.
     Two independent sources agreeing is strong evidence the label is actually right.
  2. adjudicated -- a human/model read of the narrative resolved the disagreement:
       gold_noise      -> the teacher's label is truth (gold was wrong)
       teacher_miss    -> gold's label is truth (teacher overreached)
       ambiguous       -> BOTH labels are accepted as correct (subtype_ambiguity and
                          multi_issue_ambiguity: the boundary itself is fuzzy, or the
                          narrative genuinely describes several problems, so scoring
                          either one as "wrong" would be measuring coin-flips)

Rows adjudicated `unknowable_from_text` are EXCLUDED entirely rather than accepted
either way: their correct label lives in NHTSA's recall database, not in CDESCR, so no
narrative-only model can recover it and including them just adds irreducible noise.

IMPORTANT provenance note. The adjudication was performed against an earlier 500-row
run that was later OVERWRITTEN by the current claude-haiku-4-5 run under the same
filename (the old output naming scheme omitted the model name -- since fixed in
run_teacher_labeling.py). Verified: the sample itself is identical (same seed, all 40
NHTSA gold labels match exactly), but the teacher's *predictions* differ on 10 of the
40 adjudicated rows, and the adjudicated run had 240 disagreements vs. 254 here. So for
adjudicated rows the teacher label is read from
disagreements_sample_for_adjudication.json -- the record of what was actually read and
judged -- NOT from the pilot file. Adjudication also takes precedence over the
dual-agreement rule, so a row judged `gold_noise` is never silently re-admitted as
"truth = gold" just because the newer model happened to agree with gold.

One exception is handled explicitly: if a row was judged `gold_noise` (adjudicated
teacher beats gold) but the current independent run *also* chose gold, that's a genuine
second opinion contradicting the verdict, so both labels are accepted rather than
trusting either. This affects 1 row.

Known selection bias, recorded in the output file's `caveats` and reported by the eval
script: only 40 of the 254 pilot disagreements have been adjudicated, so the set is
~88% rows where labeling was unambiguous, and it over-represents rows the Anthropic
pilot model got right. Absolute accuracy on this set is therefore optimistic, and it
mildly favors models that behave like the pilot teacher. The eval script mitigates this
by scoring the 33 adjudicated ("hard") rows as a separate subset -- that's where models
actually separate from each other. Adjudicating more disagreements is the way to
shrink the bias.
"""
import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent.parent / "output"

PILOT_FILE = "teacher_labels_anthropic_500.json"

# adjudication category -> where truth comes from
TRUTH_FROM = {
    "gold_noise": "teacher",
    "teacher_clean_miss": "gold",
    "subtype_ambiguity": "both",
    "multi_issue_ambiguity": "both",
    "unknowable_from_text": "exclude",
}


def main():
    with open(OUT_DIR / PILOT_FILE) as f:
        pilot_meta = json.load(f)
    pilot = pilot_meta["results"]

    with open(OUT_DIR / "disagreement_adjudication_n40.json") as f:
        adjudication = json.load(f)
    adj_by_id = {r["cmplid"]: r for r in adjudication["rows"]}

    with open(OUT_DIR / "disagreements_sample_for_adjudication.json") as f:
        disagreement_rows = json.load(f)
    dis_by_id = {r["cmplid"]: r for r in disagreement_rows}

    narratives = pd.read_parquet(
        DATA_DIR / "cmpl.parquet",
        columns=["CMPLID", "CDESCR", "MAKETXT", "MODELTXT", "YEARTXT"],
    ).set_index("CMPLID")

    rows, excluded, verdict_overrides = [], [], []
    for r in pilot:
        cid = r["cmplid"]
        gold = r["gold_COMPDESC_LABEL"]
        rerun_pred = r["pred_component_bucketed"]

        # Adjudication takes precedence: a row judged gold_noise must not be re-admitted
        # as "truth = gold" just because the (different, later) rerun agreed with gold.
        if cid in adj_by_id:
            adj = adj_by_id[cid]
            truth_from = TRUTH_FROM[adj["category"]]
            # The teacher label that was actually read and judged, not the rerun's.
            adjudicated_teacher = dis_by_id[cid]["teacher_pred"]

            if truth_from == "exclude":
                excluded.append({
                    "cmplid": cid,
                    "reason": adj["category"],
                    "note": adj["note"],
                    "nhtsa_gold": gold,
                    "adjudicated_teacher": adjudicated_teacher,
                })
                continue

            accept = {
                "teacher": [adjudicated_teacher],
                "gold": [gold],
                "both": [gold, adjudicated_teacher],
            }[truth_from]
            truth_source = f"adjudicated_{adj['category']}"

            if truth_from == "teacher" and rerun_pred == gold:
                accept = [gold, adjudicated_teacher]
                truth_source = "adjudicated_gold_noise_contested_by_rerun"
                verdict_overrides.append({
                    "cmplid": cid,
                    "note": "Judged gold_noise, but the independent rerun also chose the NHTSA "
                            "label, so both are accepted rather than trusting either verdict.",
                    "nhtsa_gold": gold,
                    "adjudicated_teacher": adjudicated_teacher,
                })

            entry = {
                "truth_source": truth_source,
                "accept": accept,
                "adjudication_note": adj["note"],
            }
        elif rerun_pred == gold:
            entry = {"truth_source": "dual_agreement", "accept": [gold], "adjudication_note": None}
        else:
            # Unadjudicated disagreement: truth genuinely unknown, can't score it.
            continue

        meta = narratives.loc[str(cid)]
        rows.append({
            "cmplid": cid,
            "make": meta["MAKETXT"],
            "model": meta["MODELTXT"],
            "year": meta["YEARTXT"],
            "narrative": meta["CDESCR"],
            "accept": entry["accept"],
            "primary": entry["accept"][0],
            "truth_source": entry["truth_source"],
            "is_hard": entry["truth_source"] != "dual_agreement",
            "adjudication_note": entry["adjudication_note"],
            "nhtsa_gold": gold,
            "rerun_teacher": rerun_pred,
            # Carried for secondary scoring only -- NHTSA's own CRASH/FIRE flags are
            # themselves noisy (gotcha B1/B2: FIRERELATED is set for non-fire events),
            # so these are reported separately and never mixed into the headline number.
            "nhtsa_crash": r["gold_CRASH"],
            "nhtsa_fire": r["gold_FIRE"],
        })

    n_unadjudicated = sum(
        1 for r in pilot
        if r["pred_component_bucketed"] != r["gold_COMPDESC_LABEL"] and r["cmplid"] not in adj_by_id
    )

    by_source = {}
    for r in rows:
        by_source[r["truth_source"]] = by_source.get(r["truth_source"], 0) + 1

    out = {
        "n": len(rows),
        "n_hard": sum(r["is_hard"] for r in rows),
        "built_from": {
            "dual_agreement_run": PILOT_FILE,
            "dual_agreement_model": pilot_meta["model"],
            "dual_agreement_n": pilot_meta["n"],
            "seed": pilot_meta["seed"],
            "adjudication": "disagreement_adjudication_n40.json",
            "adjudicated_teacher_labels_from": "disagreements_sample_for_adjudication.json",
            "provenance_warning":
                f"The adjudication was done against an earlier 500-row run since overwritten by "
                f"{pilot_meta['model']} under the same filename. Same seed/sample and all 40 gold "
                f"labels verified identical, but teacher predictions differ on 10 of 40 rows, so "
                f"adjudicated rows take their teacher label from the adjudication-time snapshot.",
        },
        "by_truth_source": by_source,
        "n_excluded_unknowable": len(excluded),
        "n_unadjudicated_disagreements_dropped": n_unadjudicated,
        "scoring": {
            "primary_metric": "accuracy = fraction of rows where prediction is in `accept`",
            "ambiguous_rows": "adjudicated_subtype_ambiguity and adjudicated_multi_issue_ambiguity "
                              "rows accept EITHER the NHTSA label or the pilot teacher's label",
            "hard_subset": "the 33 adjudicated rows -- report separately, this is where models separate",
        },
        "caveats": [
            f"Only 40 of {n_unadjudicated + 40} pilot disagreements were adjudicated, so "
            f"{n_unadjudicated} rows with genuinely unknown truth are dropped. The set is "
            "therefore skewed toward rows where labeling was unambiguous and absolute "
            "accuracy on it is optimistic.",
            "Dual-agreement rows are, by construction, rows the Anthropic pilot model got "
            "right, which mildly favors models that behave like claude-haiku-4-5. Relative "
            "ranking is still meaningful; treat small gaps with suspicion.",
            "NHTSA CRASH/FIRE flags are carried for secondary scoring only and are known to "
            "be noisy (gotcha B1/B2).",
        ],
        "excluded_unknowable_from_text": excluded,
        "verdict_contested_by_rerun": verdict_overrides,
        "rows": rows,
    }

    with open(OUT_DIR / "gold_eval_set.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"Gold eval set: {len(rows)} scoreable rows ({out['n_hard']} hard / adjudicated)")
    for src, n in sorted(by_source.items(), key=lambda kv: -kv[1]):
        print(f"  {src:<40} {n:>4}")
    print(f"  {'excluded (unknowable_from_text)':<40} {len(excluded):>4}")
    print(f"  {'dropped (unadjudicated disagreements)':<40} {n_unadjudicated:>4}")
    print(f"\nWrote {OUT_DIR / 'gold_eval_set.json'}")


if __name__ == "__main__":
    main()
