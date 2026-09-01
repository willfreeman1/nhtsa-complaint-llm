"""Apply the row-by-row review of the gold eval set and re-score saved model predictions.

Every one of the 275 rows in output/gold_eval_set.json was read against the narrative and
given a verdict in output/gold_review_batch_*.json:

    confirm  -- the assigned label is right as-is
    widen    -- a second (or third) label is equally defensible; all are accepted
    relabel  -- the assigned label was wrong and is replaced

Widening is not generosity: the eval exists to rank models, and scoring a defensible answer
as wrong measures NHTSA's coding coin-flips rather than model quality. Two systematic sources
of widening came out of the read-through:

  1. brake_family -- the prompt's own definitions of SERVICE BRAKES, SERVICE BRAKES HYDRAULIC
     and SERVICE BRAKES AIR each claim ordinary subtype-less brake complaints (AIR literally
     says it covers "rotors, calipers, ABS units"). Rows tagged brake_family accept all three.
     Because that is a judgment call, the scorer also reports a STRICT number where those rows
     require the originally-assigned label, so the effect is visible rather than baked in.
  2. multi-issue / vague narratives -- several rows either describe 3-5 unrelated failures or
     name no part at all, and UNKNOWN OR OTHER's own definition covers the latter.

Re-scoring is offline: model_eval_*.json stores each row's prediction, so no API calls are
needed to see what the review does to the rankings.
"""
import glob
import json
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "output"


def load_reviews():
    reviews, batches = {}, []
    for path in sorted(glob.glob(str(OUT_DIR / "gold_review_batch_*.json"))):
        with open(path) as f:
            batch = json.load(f)
        batches.append({"file": Path(path).name, "rows_reviewed": batch["rows_reviewed"],
                        "n": len(batch["verdicts"])})
        for v in batch["verdicts"]:
            if v["cmplid"] in reviews:
                raise ValueError(f"duplicate verdict for {v['cmplid']} in {path}")
            reviews[v["cmplid"]] = v
    return reviews, batches


def score(preds, rows, key="accept"):
    """Accuracy over rows for which we have a prediction."""
    hit = n = 0
    for r in rows:
        p = preds.get(r["cmplid"])
        if p is None:
            continue
        n += 1
        hit += p in r[key]
    return (hit / n) if n else None, n


def main():
    with open(OUT_DIR / "gold_eval_set.json") as f:
        gold = json.load(f)
    reviews, batches = load_reviews()

    missing = [r["cmplid"] for r in gold["rows"] if r["cmplid"] not in reviews]
    extra = set(reviews) - {r["cmplid"] for r in gold["rows"]}
    if missing or extra:
        raise ValueError(f"review coverage mismatch: {len(missing)} unreviewed, {len(extra)} unknown")

    counts = {"confirm": 0, "widen": 0, "relabel": 0}
    rows = []
    for r in gold["rows"]:
        v = reviews[r["cmplid"]]
        counts[v["verdict"]] += 1
        accept = v.get("accept", r["accept"])
        row = dict(r)
        row["accept_before_review"] = r["accept"]
        row["accept"] = accept
        # STRICT scoring keeps the pre-review label for brake-family rows only, so the cost of
        # that specific (contestable) decision can be read straight off the results.
        row["accept_strict_brakes"] = r["accept"] if v.get("brake_family") else accept
        row["review_verdict"] = v["verdict"]
        row["review_note"] = v.get("note")
        row["brake_family"] = bool(v.get("brake_family"))
        row["exclude_candidate"] = bool(v.get("exclude_candidate"))
        rows.append(row)

    excluded = [r for r in rows if r["exclude_candidate"]]
    kept_rows = [r for r in rows if not r["exclude_candidate"]]

    out = dict(gold)
    out["rows"] = kept_rows
    out["n"] = len(kept_rows)
    out["n_hard"] = sum(r["is_hard"] for r in kept_rows)
    out["review"] = {
        "method": "Every row was read against its narrative and judged by the class definitions "
                  "the models themselves are given in scripts/teacher_prompt.py. NHTSA's raw "
                  "COMPDESC label is one hypothesis about the truth, not the truth itself -- "
                  "where the narrative text supports a different (or additional) label, that "
                  "label is accepted instead of / alongside NHTSA's.",
        "batches": batches,
        "verdict_counts": counts,
        "n_brake_family": sum(r["brake_family"] for r in kept_rows),
        "n_removed_out_of_scope": len(excluded),
        "removed_out_of_scope": [
            {"cmplid": r["cmplid"], "nhtsa_gold": r["nhtsa_gold"], "reason": r["review_note"]}
            for r in excluded
        ],
        "revised_prior_adjudications": [
            r["cmplid"] for r in kept_rows
            if r["review_note"] and "PRIOR ADJUDICATION" in r["review_note"]
        ],
    }
    with open(OUT_DIR / "gold_eval_set_reviewed.json", "w") as f:
        json.dump(out, f, indent=2)

    keep = kept_rows
    hard = [r for r in keep if r["is_hard"]]

    results = []
    for path in sorted(glob.glob(str(OUT_DIR / "model_eval_*.json"))):
        with open(path) as f:
            e = json.load(f)
        preds = {r["cmplid"]: r["pred"] for r in e["results"]}
        before, n = score(preds, keep, "accept_before_review")
        after, _ = score(preds, keep, "accept")
        strict, _ = score(preds, keep, "accept_strict_brakes")
        hard_after, n_hard = score(preds, hard, "accept")
        results.append({
            "model": e["model"],
            "n_scored": n,
            "accuracy_before_review": before,
            "accuracy_strict_brakes": strict,
            "accuracy_after_review": after,
            "accuracy_hard_subset": hard_after,
            "n_hard": n_hard,
            "usd_per_100k_rows": e["cost"]["usd_per_100k_rows"],
        })
    results.sort(key=lambda r: -r["accuracy_after_review"])

    with open(OUT_DIR / "model_comparison_reviewed.json", "w") as f:
        json.dump({"eval_set": "gold_eval_set_reviewed.json",
                   "n_rows": len(keep),
                   "removed_out_of_scope": out["review"]["removed_out_of_scope"],
                   "review": out["review"], "models": results}, f, indent=2)

    print(f"Reviewed all {len(rows)} rows: "
          f"{counts['confirm']} confirmed, {counts['widen']} widened, {counts['relabel']} relabeled")
    print(f"  brake-family rows (accept all 3 brake subtypes): {out['review']['n_brake_family']}")
    print(f"  removed as out-of-scope (not in the 275):        {out['review']['n_removed_out_of_scope']}")
    print(f"  prior adjudications revised:                     "
          f"{len(out['review']['revised_prior_adjudications'])} "
          f"{out['review']['revised_prior_adjudications']}")

    hdr = f"{'model':<16} {'before':>8} {'strict':>8} {'after':>8} {'hard':>8} {'$/100k':>9}"
    print(f"\n=== Accuracy on the reviewed gold set (n={len(keep)}, {len(hard)} hard) ===")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        cost = f"${r['usd_per_100k_rows']:.2f}" if r["usd_per_100k_rows"] is not None else "unknown"
        print(f"{r['model']:<16} {r['accuracy_before_review']:>7.1%} {r['accuracy_strict_brakes']:>7.1%} "
              f"{r['accuracy_after_review']:>7.1%} {r['accuracy_hard_subset']:>7.1%} {cost:>9}")
    print("\nbefore = original labels | strict = review minus the brake-subtype widening | "
          "after = full review")
    print(f"Wrote {OUT_DIR / 'gold_eval_set_reviewed.json'} and "
          f"{OUT_DIR / 'model_comparison_reviewed.json'}")


if __name__ == "__main__":
    main()
