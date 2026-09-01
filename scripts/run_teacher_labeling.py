"""Run the teacher-labeling prompt (scripts/teacher_prompt.py) against a real LLM API.

Usage:
    python scripts/run_teacher_labeling.py --provider anthropic --n 500
    python scripts/run_teacher_labeling.py --provider openai --n 500

Requires OPENAI_API_KEY / ANTHROPIC_API_KEY in a .env file at the project root
(see .env.example). Never pass real keys on the command line or commit .env.

Samples N rows at random from data/cmpl_clean.parquet (same seed across providers
so a pilot run is directly comparable model-to-model), calls the teacher prompt for
each, parses the JSON response, and writes results (raw teacher output + gold labels
side by side) to output/teacher_labels_{provider}_{n}.json.
"""
import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from teacher_prompt import ALL_VALID_COMPONENTS, build_system_prompt, build_user_message, bucket_component

load_dotenv()

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent.parent / "output"
SEED = 42

DEFAULT_MODEL = {
    "openai": "gpt-5.6-terra",
    "anthropic": "claude-sonnet-5",
}

JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_json_response(text):
    m = JSON_RE.search(text)
    if not m:
        raise ValueError(f"No JSON object found in response: {text[:200]!r}")
    return json.loads(m.group(0))


def call_openai(client, model, system_prompt, user_msg):
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    text = resp.choices[0].message.content
    usage = resp.usage
    return text, {"input_tokens": usage.prompt_tokens, "output_tokens": usage.completion_tokens}


def call_anthropic(client, model, system_prompt, user_msg):
    kwargs = dict(
        model=model,
        max_tokens=1024,
        system=[
            {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}
        ],
        messages=[{"role": "user", "content": user_msg}],
    )
    # Sonnet 5 / Opus 5 run *adaptive* thinking by default and reject non-default
    # temperature/top_p/top_k (HTTP 400). This is closed-form classification with
    # explicit rules + few-shot examples, so disable thinking rather than pay for
    # reasoning tokens we don't need. Haiku 4.5 uses *manual* extended thinking,
    # which defaults to off when the `thinking` field is omitted entirely --
    # passing type="disabled" is not a valid value for it, so skip the field.
    if "sonnet-5" in model or "opus-5" in model:
        kwargs["thinking"] = {"type": "disabled"}
    resp = client.messages.create(**kwargs)
    text = "".join(block.text for block in resp.content if block.type == "text")
    usage = resp.usage
    return text, {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0),
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0),
    }


def label_one(row, provider, client, model, system_prompt, max_retries=4):
    user_msg = build_user_message(
        narrative=row["CDESCR"], make=row["MAKETXT"], model=row["MODELTXT"], year=row["YEARTXT"]
    )
    last_err = None
    for attempt in range(max_retries):
        try:
            if provider == "openai":
                text, usage = call_openai(client, model, system_prompt, user_msg)
            else:
                text, usage = call_anthropic(client, model, system_prompt, user_msg)
            parsed = parse_json_response(text)
            return {
                "cmplid": int(row["CMPLID"]),
                "raw_response": parsed,
                "usage": usage,
                "gold_COMPDESC_LABEL": row["COMPDESC_LABEL"],
                "gold_COMPDESC_TOP": row["COMPDESC_TOP"],
                "gold_CRASH": row["CRASH"],
                "gold_FIRE": row["FIRE"],
                "gold_INJURED": None if pd.isna(row["INJURED"]) else float(row["INJURED"]),
                "gold_DEATHS": None if pd.isna(row["DEATHS"]) else float(row["DEATHS"]),
                "pred_component_bucketed": bucket_component(parsed.get("component", "")),
                "pred_component_off_list": parsed.get("component", "") not in ALL_VALID_COMPONENTS,
                "error": None,
            }
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
            time.sleep(min(2 ** attempt, 20))
    return {
        "cmplid": int(row["CMPLID"]),
        "raw_response": None,
        "usage": None,
        "gold_COMPDESC_LABEL": row["COMPDESC_LABEL"],
        "gold_COMPDESC_TOP": row["COMPDESC_TOP"],
        "gold_CRASH": row["CRASH"],
        "gold_FIRE": row["FIRE"],
        "gold_INJURED": None if pd.isna(row["INJURED"]) else float(row["INJURED"]),
        "gold_DEATHS": None if pd.isna(row["DEATHS"]) else float(row["DEATHS"]),
        "pred_component_bucketed": None,
        "error": last_err,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["openai", "anthropic"], required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    model = args.model or DEFAULT_MODEL[args.provider]

    if args.provider == "openai":
        from openai import OpenAI
        client = OpenAI()
    else:
        from anthropic import Anthropic
        client = Anthropic()

    print(f"Provider={args.provider} model={model} n={args.n}")
    system_prompt = build_system_prompt()
    print(f"System prompt: ~{len(system_prompt)//4:,} tokens (cached after first call)")

    df = pd.read_parquet(DATA_DIR / "cmpl_clean.parquet")
    sample = df.sample(n=args.n, random_state=args.seed).reset_index(drop=True)
    print(f"Sampled {len(sample):,} rows from {len(df):,} cleaned vehicle rows")

    results = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(label_one, row, args.provider, client, model, system_prompt): i
            for i, row in sample.iterrows()
        }
        done = 0
        for fut in as_completed(futures):
            results.append(fut.result())
            done += 1
            if done % 50 == 0 or done == len(sample):
                elapsed = time.time() - t0
                print(f"  {done}/{len(sample)} done ({elapsed:.0f}s elapsed)")

    n_errors = sum(1 for r in results if r["error"] is not None)
    n_off_list = sum(1 for r in results if r.get("pred_component_off_list"))
    total_input = sum(r["usage"]["input_tokens"] for r in results if r["usage"])
    total_output = sum(r["usage"]["output_tokens"] for r in results if r["usage"])
    print(f"\nDone in {time.time()-t0:.0f}s. {n_errors} errors out of {len(results)}.")
    print(f"Off-list component answers (didn't match any of the 41 valid strings): {n_off_list}/{len(results)}")
    print(f"Total tokens: {total_input:,} input, {total_output:,} output")

    safe_model = model.replace("/", "-").replace(":", "-")
    out_path = OUT_DIR / f"teacher_labels_{args.provider}_{safe_model}_{args.n}.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "provider": args.provider,
                "model": model,
                "n": args.n,
                "seed": args.seed,
                "n_errors": n_errors,
                "total_input_tokens": total_input,
                "total_output_tokens": total_output,
                "elapsed_s": time.time() - t0,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
