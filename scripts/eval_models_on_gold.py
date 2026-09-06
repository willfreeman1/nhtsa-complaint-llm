"""Score an LLM on the curated gold eval set (output/gold_eval_set.json) and price it.

Unlike the pilot runs, this does NOT score against raw NHTSA COMPDESC. Truth here is
either dual-agreement (teacher and NHTSA independently agreed) or hand-adjudicated, and
genuinely-ambiguous rows accept either defensible label. See build_gold_eval_set.py.

Usage:
    python scripts/eval_models_on_gold.py --provider openai --model gpt-5.4-mini
    python scripts/eval_models_on_gold.py --provider openai --model gpt-5-mini --reasoning-effort minimal

Writes output/model_eval_{provider}_{model}.json and prints a quality-vs-cost summary.
"""
import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from sklearn.metrics import f1_score

from pricing import cost_usd, rates
from run_teacher_labeling import call_anthropic, parse_json_response
from teacher_prompt import ALL_VALID_COMPONENTS, build_system_prompt, build_user_message, bucket_component

load_dotenv(override=True)

OUT_DIR = Path(__file__).parent.parent / "output"


def openai_usage(resp):
    u = resp.usage
    cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
    reasoning = getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", 0) or 0
    return {
        "input_tokens": u.prompt_tokens - cached,
        "cached_input_tokens": cached,
        "output_tokens": u.completion_tokens,
        "reasoning_tokens": reasoning,
    }


# Cheapest-first: reasoning tokens bill at the output rate, and this is closed-form
# classification with explicit rules plus few-shot examples, so buying reasoning is
# mostly buying latency and cost. Mirrors disabling thinking on the Anthropic side.
EFFORT_PREFERENCE = ["none", "minimal", "low", "medium", "high"]

SUPPORTED_VALUES_RE = re.compile(r"Supported values are:\s*(.+?)\.?$")


def cheapest_supported_effort(error_msg, current):
    """Pick the cheapest effort level the model advertises in its 400 error."""
    m = SUPPORTED_VALUES_RE.search(error_msg.replace("\n", " "))
    if not m:
        return None
    offered = {v.strip().strip("'\"") for v in m.group(1).replace(" and ", ",").split(",")}
    for level in EFFORT_PREFERENCE:
        if level in offered and level != current:
            return level
    return None


def negotiate_openai_kwargs(client, model, system_prompt, user_msg, reasoning_effort):
    """Find a request shape this model actually accepts.

    Model families disagree about temperature, max_tokens vs max_completion_tokens, and
    which reasoning_effort values exist, and they signal this with HTTP 400 rather than
    anything introspectable. Probe once up front instead of eating a 400 on every row.
    """
    kwargs = {
        "model": model,
        "response_format": {"type": "json_object"},
        "max_completion_tokens": 4096,
    }
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort

    ladder = [
        ("reasoning_effort", "AUTO"),     # effort level rejected -> cheapest one it offers
        ("reasoning_effort", None),       # reasoning_effort unsupported at all -> drop
        ("max_completion_tokens", None),  # older models want max_tokens instead
        ("response_format", None),        # no JSON mode -> rely on the prompt
    ]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]

    attempts = []
    for _ in range(len(ladder) + 1):
        try:
            resp = client.chat.completions.create(messages=messages, **kwargs)
            return kwargs, attempts
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            attempts.append(msg[:200])
            for param, replacement in ladder:
                if param not in kwargs or param not in msg:
                    continue
                if replacement == "AUTO":
                    better = cheapest_supported_effort(msg, kwargs.get("reasoning_effort"))
                    if better is None:
                        continue
                    kwargs["reasoning_effort"] = better
                elif replacement is None:
                    kwargs.pop(param)
                    if param == "max_completion_tokens":
                        kwargs["max_tokens"] = 4096
                else:
                    if kwargs[param] == replacement:
                        continue
                    kwargs[param] = replacement
                break
            else:
                raise
    raise RuntimeError(f"Could not find an accepted request shape for {model}: {attempts}")


def label_one(row, provider, client, model, system_prompt, req_kwargs, max_retries=4):
    user_msg = build_user_message(
        narrative=row["narrative"], make=row["make"], model=row["model"], year=row["year"]
    )
    last_err = None
    for attempt in range(max_retries):
        try:
            if provider == "openai":
                resp = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_msg},
                    ],
                    **req_kwargs,
                )
                text = resp.choices[0].message.content
                usage = openai_usage(resp)
                if not text:
                    raise ValueError("empty response content (reasoning may have consumed the budget)")
            else:
                text, raw_usage = call_anthropic(client, model, system_prompt, user_msg)
                usage = {
                    "input_tokens": raw_usage["input_tokens"],
                    "cached_input_tokens": raw_usage.get("cache_read_input_tokens", 0),
                    "cache_creation_input_tokens": raw_usage.get("cache_creation_input_tokens", 0),
                    "output_tokens": raw_usage["output_tokens"],
                    "reasoning_tokens": 0,
                }
            parsed = parse_json_response(text)
            raw_component = parsed.get("component", "")
            pred = bucket_component(raw_component)
            # Score against the raw (pre-bucketing) answer first: the gold "accept" list
            # can itself contain a rare category (e.g. "HYBRID PROPULSION SYSTEM"), and
            # collapsing a correct rare-category answer to "OTHER" before comparing would
            # wrongly mark it wrong. Only fall back to the bucketed 31-class label when the
            # raw answer isn't itself one of the accepted labels.
            correct = raw_component in row["accept"] or pred in row["accept"]
            scored_pred = raw_component if raw_component in row["accept"] else pred
            return {
                "cmplid": row["cmplid"],
                "pred": scored_pred,
                "raw_component": raw_component,
                "off_list": raw_component not in ALL_VALID_COMPONENTS,
                "confidence": parsed.get("confidence"),
                "pred_crash": parsed.get("crash"),
                "pred_fire": parsed.get("fire"),
                "correct": correct,
                "accept": row["accept"],
                "truth_source": row["truth_source"],
                "is_hard": row["is_hard"],
                "nhtsa_gold": row["nhtsa_gold"],
                "usage": usage,
                "error": None,
            }
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
            time.sleep(min(2 ** attempt, 20))
    return {
        "cmplid": row["cmplid"],
        "pred": None,
        "correct": False,
        "accept": row["accept"],
        "truth_source": row["truth_source"],
        "is_hard": row["is_hard"],
        "usage": None,
        "error": last_err,
    }


def score(results, gold_rows):
    scored = [r for r in results if r["pred"] is not None]
    by_id = {r["cmplid"]: r for r in gold_rows}

    def acc(subset):
        return (sum(r["correct"] for r in subset) / len(subset)) if subset else None

    # For macro-F1, a prediction inside the accepted set counts as that label -- scoring
    # an ambiguous row's second valid answer as a miss would just measure coin flips.
    y_true = [r["pred"] if r["correct"] else r["accept"][0] for r in scored]
    y_pred = [r["pred"] for r in scored]

    hard = [r for r in results if r["is_hard"]]
    easy = [r for r in results if not r["is_hard"]]

    crash_hits = fire_hits = crash_n = fire_n = 0
    for r in scored:
        g = by_id[r["cmplid"]]
        if r.get("pred_crash") in ("Y", "N"):
            crash_n += 1
            crash_hits += r["pred_crash"] == g["nhtsa_crash"]
        if r.get("pred_fire") in ("Y", "N"):
            fire_n += 1
            fire_hits += r["pred_fire"] == g["nhtsa_fire"]

    by_conf = {}
    for level in ("high", "medium", "low"):
        sub = [r for r in scored if r.get("confidence") == level]
        by_conf[level] = {"n": len(sub), "accuracy": acc(sub)}

    return {
        "n_scored": len(scored),
        "n_errors": sum(1 for r in results if r["pred"] is None),
        "accuracy": acc(results),
        "accuracy_hard_subset": acc(hard),
        "accuracy_dual_agreement_subset": acc(easy),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "off_list_rate": sum(1 for r in scored if r.get("off_list")) / len(scored) if scored else None,
        "accuracy_by_self_reported_confidence": by_conf,
        "secondary_vs_noisy_nhtsa_flags": {
            "note": "NHTSA CRASH/FIRE flags are themselves unreliable (gotcha B1/B2) -- "
                    "disagreement here is not necessarily a model error.",
            "crash_agreement": crash_hits / crash_n if crash_n else None,
            "fire_agreement": fire_hits / fire_n if fire_n else None,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    ap.add_argument("--model", required=True)
    ap.add_argument("--reasoning-effort", default="none",
                    help="OpenAI reasoning models only; falls back to the cheapest "
                         "level the model supports, or is dropped if unsupported.")
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None, help="Score only the first N rows (smoke test).")
    ap.add_argument("--gold-file", default="gold_eval_set_v3.json",
                     help="Gold set file in output/ to score against.")
    args = ap.parse_args()

    with open(OUT_DIR / args.gold_file) as f:
        gold = json.load(f)
    gold_rows = gold["rows"][: args.limit] if args.limit else gold["rows"]

    system_prompt = build_system_prompt()
    print(f"Provider={args.provider} model={args.model} rows={len(gold_rows)}")
    print(f"System prompt ~{len(system_prompt)//4:,} tokens")

    if args.provider == "openai":
        from openai import OpenAI
        client = OpenAI()
        probe_msg = build_user_message(
            narrative=gold_rows[0]["narrative"], make=gold_rows[0]["make"],
            model=gold_rows[0]["model"], year=gold_rows[0]["year"],
        )
        req_kwargs, attempts = negotiate_openai_kwargs(
            client, args.model, system_prompt, probe_msg, args.reasoning_effort
        )
        shape = {k: v for k, v in req_kwargs.items() if k != "model"}
        print(f"Accepted request shape: {shape}")
        for a in attempts:
            print(f"  (rejected: {a})")
    else:
        from anthropic import Anthropic
        client = Anthropic()
        req_kwargs = None

    t0 = time.time()
    done = threading.Lock()
    counter = {"n": 0}

    def work(row):
        r = label_one(row, args.provider, client, args.model, system_prompt, req_kwargs)
        with done:
            counter["n"] += 1
            if counter["n"] % 50 == 0 or counter["n"] == len(gold_rows):
                print(f"  {counter['n']}/{len(gold_rows)} ({time.time()-t0:.0f}s)")
        return r

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(work, gold_rows))

    elapsed = time.time() - t0
    metrics = score(results, gold_rows)

    tok = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
    for r in results:
        if r["usage"]:
            for k in tok:
                tok[k] += r["usage"].get(k, 0)

    total_cost = cost_usd(
        args.provider, args.model, tok["input_tokens"], tok["cached_input_tokens"], tok["output_tokens"]
    )
    n = metrics["n_scored"] or 1
    cost_block = {
        "rates_usd_per_1m_input_cached_output": rates(args.provider, args.model),
        "tokens": tok,
        "total_usd": total_cost,
        "usd_per_1k_rows": total_cost / n * 1000 if total_cost is not None else None,
        "usd_per_100k_rows": total_cost / n * 100_000 if total_cost is not None else None,
    }

    out = {
        "provider": args.provider,
        "model": args.model,
        "request_shape": {k: v for k, v in (req_kwargs or {}).items() if k != "model"},
        "eval_set": {"file": args.gold_file, "n": len(gold_rows), "n_hard": sum(r["is_hard"] for r in gold_rows)},
        "elapsed_s": elapsed,
        "metrics": metrics,
        "cost": cost_block,
        "results": results,
    }
    safe = args.model.replace("/", "-").replace(":", "-")
    path = OUT_DIR / f"model_eval_{args.provider}_{safe}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n=== {args.model} on curated gold set (n={len(gold_rows)}) in {elapsed:.0f}s ===")
    print(f"  accuracy overall           {metrics['accuracy']:.1%}")
    print(f"  accuracy hard subset       {metrics['accuracy_hard_subset']:.1%} "
          f"(n={sum(r['is_hard'] for r in gold_rows)})")
    print(f"  accuracy dual-agreement    {metrics['accuracy_dual_agreement_subset']:.1%}")
    print(f"  macro-F1                   {metrics['macro_f1']:.3f}")
    print(f"  off-list / errors          {metrics['off_list_rate']:.1%} / {metrics['n_errors']}")
    print(f"  tokens  in {tok['input_tokens']:,} (+{tok['cached_input_tokens']:,} cached) "
          f"out {tok['output_tokens']:,} (reasoning {tok['reasoning_tokens']:,})")
    if total_cost is None:
        print("  cost                       unknown (no verified rates for this model)")
    else:
        print(f"  cost  ${total_cost:.4f} for this run  ->  ${cost_block['usd_per_100k_rows']:.2f} / 100k rows")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
