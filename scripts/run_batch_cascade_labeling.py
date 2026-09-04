"""Rung-1 training-data labeling via the validated cheap-first cascade, run through
Anthropic's Message Batches API (50% off both stages -- see pricing.py BATCH_DISCOUNT).

Cascade (see output/confidence_cascade_analysis.json): claude-haiku-4-5 labels every
row; rows where haiku reports "medium" self-confidence get relabeled by
claude-sonnet-5 (its answer wins).

Batches are chunked: Anthropic caps a Message Batch at 100k requests or 256 MB,
and our system prompt is ~30 KB, so a 62k-row payload would not fit in one batch.
A small canary chunk is submitted and scored first; remaining chunks only go out
if that gate passes (so we do not spend the full budget on a broken request shape).

Two-stage, resumable via a JSON state file (chunk results live in sidecar JSON so
the state file stays small):

    Stage 1: haiku on every row (canary, then remaining chunks).
    Stage 2: sonnet on "medium"-confidence rows only, also chunked.
    Finalize: merge (sonnet wins on escalated rows), cost at batch rates, write parquet.

Usage:
    python scripts/run_batch_cascade_labeling.py --sample data/training_sample.parquet --run-name full_62k
    python scripts/run_batch_cascade_labeling.py --resume output/batch_state_full_62k.json
"""
import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from anthropic import Anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from dotenv import load_dotenv

from pricing import cost_usd
from teacher_prompt import ALL_VALID_COMPONENTS, build_system_prompt, build_user_message, bucket_component

load_dotenv(override=True)

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:  # noqa: BLE001
    pass

OUT_DIR = Path(__file__).parent.parent / "output"
DATA_DIR = Path(__file__).parent.parent / "data"
CHUNK_DIR = OUT_DIR / "batch_chunks"

HAIKU_MODEL = "claude-haiku-4-5"
SONNET_MODEL = "claude-sonnet-5"
ESCALATE_ON = {"medium"}

JSON_RE = re.compile(r"\{.*\}", re.S)
POLL_INTERVAL_S = 30

# ~30 KB system prompt * 4000 requests ~= 120 MB, under the 256 MB batch cap.
DEFAULT_CHUNK_SIZE = 4000
DEFAULT_CANARY_N = 250
CANARY_MAX_ERROR_RATE = 0.05
CANARY_MIN_PARSE_RATE = 0.95


def log(msg):
    print(msg, flush=True)


def parse_json_response(text):
    m = JSON_RE.search(text)
    if not m:
        raise ValueError(f"No JSON object found in response: {text[:200]!r}")
    return json.loads(m.group(0))


def build_request(custom_id, model, system_prompt, user_msg):
    params = dict(
        model=model,
        max_tokens=1024,
        system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_msg}],
    )
    if "sonnet-5" in model or "opus-5" in model:
        params["thinking"] = {"type": "disabled"}
    return Request(custom_id=custom_id, params=MessageCreateParamsNonStreaming(**params))


def save_state(path, state):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    tmp.replace(path)


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def make_chunks(rows, canary_n, chunk_size):
    """First chunk is the canary (if canary_n > 0), then equal-sized remainder chunks."""
    chunks = []
    if canary_n > 0 and rows:
        n = min(canary_n, len(rows))
        chunks.append(("canary", rows[:n]))
        rows = rows[n:]
    for i in range(0, len(rows), chunk_size):
        chunks.append((f"chunk_{i // chunk_size:02d}", rows[i:i + chunk_size]))
    return chunks


def submit_batch(client, requests, label):
    log(f"Submitting batch '{label}': {len(requests):,} requests")
    batch = client.messages.batches.create(requests=requests)
    log(f"  batch_id={batch.id} status={batch.processing_status}")
    return batch.id


def poll_batch(client, batch_id, label):
    t0 = time.time()
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        log(f"  [{label}] {time.time()-t0:.0f}s elapsed: processing={counts.processing} "
            f"succeeded={counts.succeeded} errored={counts.errored} "
            f"canceled={counts.canceled} expired={counts.expired} status={batch.processing_status}")
        if batch.processing_status == "ended":
            return batch
        time.sleep(POLL_INTERVAL_S)


def cmplid_from_custom_id(custom_id):
    """custom_id is either the raw CMPLID or '{stage}-{CMPLID}' (stage names contain hyphens)."""
    return custom_id.rsplit("-", 1)[-1]


def retrieve_results(client, batch_id, batch_label):
    out = {}
    for item in client.messages.batches.results(batch_id):
        cmplid = cmplid_from_custom_id(item.custom_id)
        if item.result.type != "succeeded":
            out[cmplid] = {"error": f"{batch_label}:{item.result.type}"}
            continue
        msg = item.result.message
        text = "".join(b.text for b in msg.content if b.type == "text")
        usage = {
            "input_tokens": msg.usage.input_tokens,
            "output_tokens": msg.usage.output_tokens,
            "cache_read_input_tokens": getattr(msg.usage, "cache_read_input_tokens", 0) or 0,
            "cache_creation_input_tokens": getattr(msg.usage, "cache_creation_input_tokens", 0) or 0,
        }
        try:
            parsed = parse_json_response(text)
            out[cmplid] = {"parsed": parsed, "usage": usage, "error": None}
        except Exception as e:  # noqa: BLE001
            out[cmplid] = {"parsed": None, "usage": usage, "error": str(e)}
    return out


def batch_cost(usage, model_tag):
    return cost_usd(
        "anthropic", HAIKU_MODEL if model_tag == "haiku" else SONNET_MODEL,
        usage["input_tokens"],
        usage.get("cache_read_input_tokens", 0),
        usage["output_tokens"],
        batch=True,
        cache_creation_input=usage.get("cache_creation_input_tokens", 0),
    ) or 0.0


def summarize_results(results, model_tag):
    n = len(results)
    n_err = sum(1 for r in results.values() if r.get("error"))
    n_parsed = sum(1 for r in results.values() if r.get("parsed"))
    conf = Counter()
    n_off = 0
    cost = 0.0
    for r in results.values():
        if r.get("usage"):
            cost += batch_cost(r["usage"], model_tag)
        parsed = r.get("parsed")
        if not parsed:
            continue
        conf[parsed.get("confidence") or "missing"] += 1
        raw = parsed.get("component", "")
        if raw not in ALL_VALID_COMPONENTS:
            n_off += 1
    return {
        "n": n,
        "n_errors": n_err,
        "n_parsed": n_parsed,
        "parse_rate": n_parsed / n if n else 0.0,
        "error_rate": n_err / n if n else 0.0,
        "off_list_rate": n_off / n_parsed if n_parsed else None,
        "confidence": dict(conf),
        "cost_usd": round(cost, 4),
    }


def print_gate(label, summary, is_canary=False):
    conf = summary["confidence"]
    off = summary["off_list_rate"]
    log(f"  GATE [{label}] n={summary['n']} parsed={summary['n_parsed']} "
        f"errors={summary['n_errors']} parse_rate={summary['parse_rate']:.1%} "
        f"off_list={off if off is None else f'{off:.1%}'} "
        f"conf={conf} cost=${summary['cost_usd']:.4f}")
    if not is_canary:
        return True
    ok = (
        summary["error_rate"] <= CANARY_MAX_ERROR_RATE
        and summary["parse_rate"] >= CANARY_MIN_PARSE_RATE
        and summary["cost_usd"] > 0
        and summary["n_parsed"] > 0
    )
    if ok:
        log(f"CANARY GATE PASSED for {label}")
    else:
        log(f"CANARY GATE FAILED for {label} -- not submitting remaining chunks")
    return ok


def run_chunked_stage(client, state, state_path, stage_key, model, model_tag, rows,
                      system_prompt, canary_n, chunk_size, gate_canary):
    meta = state.setdefault(stage_key, {"chunks": []})
    planned = make_chunks(rows, canary_n, chunk_size)
    existing = {c["name"]: c for c in meta["chunks"]}

    all_results = {}
    for name, chunk_rows in planned:
        rec = existing.get(name)
        results_path = CHUNK_DIR / f"{state['run_name']}_{stage_key}_{name}.json"

        if rec and rec.get("retrieved"):
            log(f"Skipping already-retrieved {stage_key}/{name} ({rec['n_requests']} rows)")
            chunk_results = load_json(results_path)
        else:
            if rec and rec.get("batch_id"):
                batch_id = rec["batch_id"]
                log(f"Resuming {stage_key}/{name}: batch_id={batch_id}")
            else:
                requests = [
                    build_request(
                        str(row["CMPLID"]), model, system_prompt,
                        build_user_message(
                            narrative=row["CDESCR"], make=row["MAKETXT"],
                            model=row["MODELTXT"], year=row["YEARTXT"],
                        ),
                    )
                    for row in chunk_rows
                ]
                batch_id = submit_batch(client, requests, f"{stage_key}/{name}")
                rec = {
                    "name": name,
                    "batch_id": batch_id,
                    "n_requests": len(requests),
                    "retrieved": False,
                    "results_path": str(results_path),
                }
                meta["chunks"] = [c for c in meta["chunks"] if c["name"] != name] + [rec]
                save_state(state_path, state)

            poll_batch(client, rec["batch_id"], f"{stage_key}/{name}")
            chunk_results = retrieve_results(client, rec["batch_id"], model_tag)
            save_json(results_path, chunk_results)
            summary = summarize_results(chunk_results, model_tag)
            rec["retrieved"] = True
            rec["summary"] = summary
            rec["results_path"] = str(results_path)
            is_canary = name == "canary" and gate_canary
            rec["gate_passed"] = print_gate(f"{stage_key}/{name}", summary, is_canary=is_canary)
            meta["chunks"] = [c for c in meta["chunks"] if c["name"] != name] + [rec]
            save_state(state_path, state)
            if not rec["gate_passed"]:
                raise SystemExit(2)

        if name == "canary" and gate_canary and rec and rec.get("gate_passed") is False:
            raise SystemExit(2)

        all_results.update(chunk_results)

    return all_results


def load_stage_results(state, stage_key):
    out = {}
    for rec in state.get(stage_key, {}).get("chunks", []):
        if rec.get("retrieved") and rec.get("results_path"):
            out.update(load_json(Path(rec["results_path"])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", help="Parquet file with rows to label")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--resume", help="Path to an existing state JSON to resume from")
    ap.add_argument("--canary-n", type=int, default=DEFAULT_CANARY_N,
                    help="First haiku batch size; remaining chunks wait for this gate.")
    ap.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                    help="Max requests per subsequent Message Batch (256 MB cap).")
    args = ap.parse_args()

    client = Anthropic()
    system_prompt = build_system_prompt()
    log(f"System prompt {len(system_prompt):,} chars (~{len(system_prompt)/1024:.1f} KB)")

    if args.resume:
        state_path = Path(args.resume)
        state = load_json(state_path)
        sample = pd.read_parquet(state["sample_file"])
        run_name = state["run_name"]
        canary_n = state.get("canary_n", args.canary_n)
        chunk_size = state.get("chunk_size", args.chunk_size)
    else:
        assert args.sample, "--sample is required unless --resume is given"
        run_name = args.run_name or Path(args.sample).stem
        sample = pd.read_parquet(args.sample)
        canary_n = args.canary_n
        chunk_size = args.chunk_size
        state = {
            "run_name": run_name,
            "sample_file": args.sample,
            "n_rows": len(sample),
            "canary_n": canary_n,
            "chunk_size": chunk_size,
        }
        state_path = OUT_DIR / f"batch_state_{run_name}.json"
        save_state(state_path, state)

    rows = sample.to_dict("records")
    row_by_cmplid = {str(r["CMPLID"]): r for r in rows}
    n_haiku_chunks = len(make_chunks(rows, canary_n, chunk_size))
    log(f"Run '{run_name}': {len(rows):,} rows, canary={canary_n}, "
        f"chunk_size={chunk_size}, haiku_batches={n_haiku_chunks}. State: {state_path}")

    results1 = run_chunked_stage(
        client, state, state_path, "stage1_haiku", HAIKU_MODEL, "haiku",
        rows, system_prompt, canary_n, chunk_size, gate_canary=True,
    )
    summary1 = summarize_results(results1, "haiku")
    log(f"Stage 1 (haiku) done: {summary1['n']:,} results, {summary1['n_errors']} errors, "
        f"cost=${summary1['cost_usd']:.4f}")

    escalate_ids = [
        cmplid for cmplid, r in results1.items()
        if r.get("parsed") and r["parsed"].get("confidence") in ESCALATE_ON
    ]
    log(f"Escalating {len(escalate_ids):,}/{len(results1):,} rows "
        f"({len(escalate_ids)/len(results1):.1%}) to {SONNET_MODEL}")

    results2 = {}
    if escalate_ids:
        escalate_rows = [row_by_cmplid[cid] for cid in escalate_ids]
        # Stage 2 uses a smaller first chunk as a second gate (same request shape,
        # different model). Remaining sonnet chunks follow if it passes.
        sonnet_canary = min(50, len(escalate_rows))
        results2 = run_chunked_stage(
            client, state, state_path, "stage2_sonnet", SONNET_MODEL, "sonnet",
            escalate_rows, system_prompt, sonnet_canary, chunk_size, gate_canary=True,
        )
        summary2 = summarize_results(results2, "sonnet")
        log(f"Stage 2 (sonnet) done: {summary2['n']:,} results, {summary2['n_errors']} errors, "
            f"cost=${summary2['cost_usd']:.4f}")

    final_rows = []
    total_cost = 0.0
    for cmplid, row in row_by_cmplid.items():
        r1 = results1.get(cmplid, {})
        escalated = cmplid in results2 and not results2[cmplid].get("error")
        chosen = results2.get(cmplid, {}) if escalated else r1
        if r1.get("usage"):
            total_cost += batch_cost(r1["usage"], "haiku")
        if escalated and chosen.get("usage"):
            total_cost += batch_cost(chosen["usage"], "sonnet")

        parsed = chosen.get("parsed") if chosen else None
        raw_component = parsed.get("component", "") if parsed else None
        final_rows.append({
            "cmplid": cmplid,
            "make": row["MAKETXT"], "model": row["MODELTXT"], "year": row["YEARTXT"],
            "narrative": row["CDESCR"],
            "gold_COMPDESC_LABEL": row["COMPDESC_LABEL"],
            "labeling_model": SONNET_MODEL if escalated else HAIKU_MODEL,
            "escalated": escalated,
            "haiku_confidence": r1.get("parsed", {}).get("confidence") if r1.get("parsed") else None,
            "raw_component": raw_component,
            "component_label": bucket_component(raw_component) if raw_component else None,
            "off_list": (raw_component not in ALL_VALID_COMPONENTS) if raw_component else None,
            "crash": parsed.get("crash") if parsed else None,
            "fire": parsed.get("fire") if parsed else None,
            "injured": parsed.get("injured") if parsed else None,
            "deaths": parsed.get("deaths") if parsed else None,
            "error": (chosen or {}).get("error"),
        })

    n_failed = sum(1 for r in final_rows if r["component_label"] is None)
    log(f"=== Final: {len(final_rows):,} rows, {n_failed} failed/unparseable, "
        f"total cost ${total_cost:.4f} (batch-rate) ===")

    out_path = DATA_DIR / f"training_labels_{run_name}.parquet"
    pd.DataFrame(final_rows).to_parquet(out_path, index=False)
    log(f"Wrote {out_path}")

    summary_path = OUT_DIR / f"training_labels_summary_{run_name}.json"
    with open(summary_path, "w") as f:
        json.dump({
            "run_name": run_name,
            "n_rows": len(final_rows),
            "n_failed": n_failed,
            "n_escalated": len(escalate_ids),
            "pct_escalated": len(escalate_ids) / len(results1) if results1 else None,
            "total_cost_usd": round(total_cost, 4),
            "cost_per_100k_rows": round(total_cost / len(final_rows) * 100_000, 2) if final_rows else None,
            "output_parquet": out_path.name,
            "stage1_summary": summary1,
        }, f, indent=2)
    log(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
