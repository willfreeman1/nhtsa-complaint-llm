"""Aggregate every output/model_eval_*.json into one quality-vs-cost comparison.

Also decomposes cost (the system prompt dominates, so the biggest lever is prompt
overhead per row, not model choice) and flags rows that EVERY evaluated model missed --
those are the best candidates for a gold-set error, i.e. places where the adjudication
may itself be wrong and should be re-read.
"""
import glob
import json
from collections import Counter
from pathlib import Path

from pricing import rates

OUT_DIR = Path(__file__).parent.parent / "output"


def _load_gold(path: Path):
    with open(path) as f:
        gold = json.load(f)
    return gold, {r["cmplid"]: r for r in gold["rows"]}


def main():
    evals = []
    for path in sorted(glob.glob(str(OUT_DIR / "model_eval_*.json"))):
        with open(path) as f:
            evals.append(json.load(f))
    if not evals:
        print("No model_eval_*.json files found.")
        return

    eval_files = [e.get("eval_set", {}).get("file") for e in evals]
    eval_files = [f for f in eval_files if f]
    if not eval_files:
        raise RuntimeError("No eval_set.file found in model_eval_*.json files")

    primary_gold_file = Counter(eval_files).most_common(1)[0][0]
    gold, gold_by_id = _load_gold(OUT_DIR / primary_gold_file)

    table = []
    for e in evals:
        m, c = e["metrics"], e["cost"]
        tok = c["tokens"]
        r = rates(e["provider"], e["model"])
        n = m["n_scored"] or 1
        prompt_cost = None
        if r is not None:
            prompt_cost = (tok["input_tokens"] * r[0] + tok["cached_input_tokens"] * r[1]) / 1e6
        table.append({
            "provider": e["provider"],
            "model": e["model"],
            "reasoning_effort": e["request_shape"].get("reasoning_effort"),
            "accuracy": m["accuracy"],
            "accuracy_hard": m["accuracy_hard_subset"],
            "macro_f1": m["macro_f1"],
            "off_list_rate": m["off_list_rate"],
            "errors": m["n_errors"],
            "usd_per_100k_rows": c["usd_per_100k_rows"],
            "input_tokens_per_row": (tok["input_tokens"] + tok["cached_input_tokens"]) / n,
            "output_tokens_per_row": tok["output_tokens"] / n,
            "reasoning_tokens_per_row": tok["reasoning_tokens"] / n,
            "pct_of_cost_from_input": (prompt_cost / c["total_usd"] * 100)
                                      if prompt_cost is not None and c["total_usd"] else None,
            "sec_per_row": e["elapsed_s"] / n,
        })
    table.sort(key=lambda r: (-(r["accuracy"] or 0)))

    # Rows nobody got right -- suspect the gold set here, not the models.
    result_ids = [set(r["cmplid"] for r in e["results"]) for e in evals]
    common_ids = set.intersection(*result_ids) if result_ids else set()
    universally_missed = []
    missing_in_primary_gold = []
    for cid in sorted(common_ids, key=str):
        hits = [next((r for r in e["results"] if r["cmplid"] == cid), None) for e in evals]
        flags = [bool(h and h.get("correct")) for h in hits]
        if not any(flags):
            g = gold_by_id.get(cid)
            if g is None:
                missing_in_primary_gold.append(cid)
                continue
            preds = {}
            for e, hit in zip(evals, hits):
                if hit:
                    preds[e["model"]] = hit["pred"]
            universally_missed.append({
                "cmplid": cid,
                "accept": g["accept"],
                "truth_source": g["truth_source"],
                "predictions": preds,
                "unanimous_alternative": len(set(preds.values())) == 1,
                "adjudication_note": g["adjudication_note"],
            })
    universally_missed.sort(key=lambda r: (not r["unanimous_alternative"], str(r["cmplid"])))

    out = {
        "eval_set": {
            "file": primary_gold_file,
            "n": gold["n"],
            "n_hard": gold["n_hard"],
            "caveats": gold.get("caveats"),
        },
        "eval_sets_seen": sorted(set(eval_files)),
        "models": table,
        "cost_note":
            "Cost is dominated by the ~7.4k-token system prompt resent on every row. "
            "Prompt caching already absorbs most of it; the remaining levers are batching "
            "several complaints per call, the Batch API (50% off), and trimming few-shot "
            "examples -- all of which move cost far more than switching between models in "
            "this tier.",
        "universally_missed_rows": universally_missed,
        "universally_missed_scope_note":
            "Computed over rows common to every model_eval file in this run.",
        "rows_missing_in_primary_gold": missing_in_primary_gold,
        "universally_missed_note":
            "Every evaluated model disagreed with the gold set on these rows. Where the "
            "models unanimously agree on the same alternative label, the gold set (or the "
            "underlying adjudication) is the more likely error and should be re-read.",
    }
    with open(OUT_DIR / "model_comparison.json", "w") as f:
        json.dump(out, f, indent=2)

    hdr = f"{'model':<16} {'effort':<8} {'acc':>7} {'hard':>7} {'macroF1':>8} {'$/100k':>9} {'in/row':>8} {'out/row':>8} {'s/row':>6}"
    print(f"\n=== Quality vs cost on curated gold set (n={gold['n']}, {gold['n_hard']} hard) ===")
    print(f"Primary eval set: {primary_gold_file}")
    if len(set(eval_files)) > 1:
        print(f"Mixed eval sets detected: {sorted(set(eval_files))}")
    print(hdr)
    print("-" * len(hdr))
    for r in table:
        cost = f"${r['usd_per_100k_rows']:.2f}" if r["usd_per_100k_rows"] is not None else "unknown"
        print(f"{r['model']:<16} {str(r['reasoning_effort']):<8} {r['accuracy']:>6.1%} "
              f"{r['accuracy_hard']:>6.1%} {r['macro_f1']:>8.3f} {cost:>9} "
              f"{r['input_tokens_per_row']:>8,.0f} {r['output_tokens_per_row']:>8,.0f} {r['sec_per_row']:>6.2f}")

    print(f"\n{len(universally_missed)} rows missed by ALL {len(evals)} models "
          f"({sum(r['unanimous_alternative'] for r in universally_missed)} with a unanimous "
          f"alternative label -> likely gold-set errors worth re-reading)")
    if missing_in_primary_gold:
        print(f"Skipped {len(missing_in_primary_gold)} common rows not found in {primary_gold_file}")
    print(f"Wrote {OUT_DIR / 'model_comparison.json'}")


if __name__ == "__main__":
    main()
