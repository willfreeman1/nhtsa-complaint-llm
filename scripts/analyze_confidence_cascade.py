"""Does self-reported confidence predict correctness, and if so, is a cheap-model-first
/ escalate-on-low-confidence cascade a viable way to cut teacher-labeling cost?

Uses the per-row `confidence` field each model already returns (see teacher_prompt.py)
and the existing 5-model gold-set eval results (output/model_eval_*.json) to:
  1. Confirm confidence is calibrated (accuracy by self-reported confidence bucket).
  2. Simulate real two-tier cascades: run a cheap model on every row, and only pay for
     a smarter model on rows where the cheap model reports low/medium confidence.
     Cost and correctness for escalated rows use each model's *actual* recorded
     token usage and outcome on that exact cmplid (not an estimate), so this is a
     faithful backtest, not a Fermi guess -- limited only by gold-set size (n=498).

Writes output/confidence_cascade_analysis.json.

Usage:
    python scripts/analyze_confidence_cascade.py
"""
import glob
import json
import math
from pathlib import Path

from pricing import cost_usd

OUT_DIR = Path(__file__).parent.parent / "output"

SMART_TARGETS = ["anthropic/claude-sonnet-5", "anthropic/claude-haiku-4-5"]
CHEAP_CANDIDATES = [
    "openai/gpt-5.4-nano",
    "openai/gpt-5-mini",
    "openai/gpt-5.4-mini",
    "anthropic/claude-haiku-4-5",
]
ESCALATION_POLICIES = {
    "low_only": ("low",),
    "medium_only": ("medium",),
    "low_and_medium": ("low", "medium"),
}


def load_models():
    models = {}
    for f in glob.glob(str(OUT_DIR / "model_eval_*.json")):
        d = json.load(open(f, encoding="utf-8"))
        key = f"{d['provider']}/{d['model']}"
        by_id = {r["cmplid"]: r for r in d["results"] if r.get("usage") is not None}
        models[key] = {"provider": d["provider"], "model": d["model"], "by_id": by_id}
    return models


def row_cost(provider, model, usage):
    return cost_usd(provider, model, usage["input_tokens"], usage["cached_input_tokens"], usage["output_tokens"]) or 0.0


def confidence_calibration(models):
    out = {}
    for key, m in models.items():
        rows = list(m["by_id"].values())
        buckets = {}
        for level in ("high", "medium", "low", None):
            sub = [r for r in rows if r.get("confidence") == level]
            if not sub:
                continue
            buckets[level or "missing"] = {
                "n": len(sub),
                "accuracy": sum(r["correct"] for r in sub) / len(sub),
                "pct_of_rows": len(sub) / len(rows),
            }
        out[key] = buckets
    return out


def solo(models, key):
    m = models[key]
    rows = list(m["by_id"].values())
    n = len(rows)
    acc = sum(r["correct"] for r in rows) / n
    cost = sum(row_cost(m["provider"], m["model"], r["usage"]) for r in rows) / n * 100_000
    return {"n": n, "accuracy": acc, "accuracy_se": math.sqrt(acc * (1 - acc) / n), "usd_per_100k_rows": cost}


def cascade(models, cheap_key, smart_key, escalate_levels):
    cheap, smart = models[cheap_key], models[smart_key]
    common_ids = set(cheap["by_id"]) & set(smart["by_id"])
    total_cost = 0.0
    n_correct = 0
    n_escalated = 0
    for cid in common_ids:
        crow = cheap["by_id"][cid]
        cost = row_cost(cheap["provider"], cheap["model"], crow["usage"])
        if crow.get("confidence") in escalate_levels:
            n_escalated += 1
            srow = smart["by_id"][cid]
            cost += row_cost(smart["provider"], smart["model"], srow["usage"])
            correct = srow["correct"]
        else:
            correct = crow["correct"]
        total_cost += cost
        n_correct += int(correct)
    n = len(common_ids)
    acc = n_correct / n
    return {
        "n": n,
        "accuracy": acc,
        "accuracy_se": math.sqrt(acc * (1 - acc) / n),
        "pct_escalated": n_escalated / n,
        "usd_per_100k_rows": total_cost / n * 100_000,
    }


def main():
    models = load_models()
    available = set(models)

    calibration = confidence_calibration(models)

    solo_baselines = {key: solo(models, key) for key in models}

    cascades = []
    skipped_pairs = []
    for cheap_key in CHEAP_CANDIDATES:
        if cheap_key not in available:
            continue
        for smart_key in SMART_TARGETS:
            if smart_key == cheap_key:
                continue
            if smart_key not in available:
                skipped_pairs.append({
                    "cheap_model": cheap_key,
                    "smart_model": smart_key,
                    "reason": "missing_eval_artifact",
                })
                continue
            for policy_name, levels in ESCALATION_POLICIES.items():
                r = cascade(models, cheap_key, smart_key, levels)
                cascades.append({
                    "cheap_model": cheap_key,
                    "smart_model": smart_key,
                    "escalation_policy": policy_name,
                    **r,
                })

    out = {
        "note": (
            "Backtest on the n=498 gold_eval_set_v2 rows using each model's *actual* "
            "recorded per-row confidence, correctness, and token usage -- escalated rows "
            "pay for both the cheap and smart model call, matching real cascade cost. "
            "Small n means single-digit-percent differences between cascades are noisy; "
            "treat the qualitative ranking as more reliable than exact numbers."
        ),
        "confidence_calibration_by_model": calibration,
        "solo_baselines": solo_baselines,
        "available_models": sorted(available),
        "skipped_pairs": skipped_pairs,
        "cascades": sorted(cascades, key=lambda c: c["usd_per_100k_rows"]),
    }
    path = OUT_DIR / "confidence_cascade_analysis.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {path}")

    print("\n=== Solo baselines (cheapest first) ===")
    for key in sorted(solo_baselines, key=lambda k: solo_baselines[k]["usd_per_100k_rows"]):
        s = solo_baselines[key]
        print(f"  {key:32s} acc={s['accuracy']*100:5.1f}%+-{s['accuracy_se']*100:.1f}  ${s['usd_per_100k_rows']:7.2f}/100k rows")

    print("\n=== Cascades (cheapest first) ===")
    for c in out["cascades"]:
        print(f"  {c['cheap_model']:26s} -> {c['smart_model']:26s} [{c['escalation_policy']:15s}] "
              f"acc={c['accuracy']*100:5.1f}%+-{c['accuracy_se']*100:.1f}  escalated={c['pct_escalated']*100:5.1f}%  "
              f"${c['usd_per_100k_rows']:7.2f}/100k rows")


if __name__ == "__main__":
    main()
