"""Finish rare_topup Stage-2 Sonnet escalations via on-demand API (not Batch).

Uses already-retrieved Stage-1 Haiku results + Stage-2 Sonnet canary, then labels
the remaining medium-confidence rows concurrently with claude-sonnet-5 and writes
the same parquet/summary artifacts as run_batch_cascade_labeling.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv

from pricing import cost_usd
from run_batch_cascade_labeling import (
    ALL_VALID_COMPONENTS,
    CHUNK_DIR,
    DATA_DIR,
    ESCALATE_ON,
    HAIKU_MODEL,
    OUT_DIR,
    SONNET_MODEL,
    batch_cost,
    build_system_prompt,
    build_user_message,
    bucket_component,
    load_json,
    load_stage_results,
    parse_json_response,
    save_json,
    save_state,
    summarize_results,
)

load_dotenv(override=True)

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:  # noqa: BLE001
    pass


def log(msg):
    print(msg, flush=True)


def ondemand_cost(usage, model):
    return cost_usd(
        "anthropic",
        model,
        usage["input_tokens"],
        usage.get("cache_read_input_tokens", 0),
        usage["output_tokens"],
        batch=False,
        cache_creation_input=usage.get("cache_creation_input_tokens", 0),
    ) or 0.0


def call_sonnet(client, system_prompt, row):
    cmplid = str(row["CMPLID"])
    user_msg = build_user_message(
        narrative=row["CDESCR"],
        make=row["MAKETXT"],
        model=row["MODELTXT"],
        year=row["YEARTXT"],
    )
    kwargs = dict(
        model=SONNET_MODEL,
        max_tokens=1024,
        system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_msg}],
    )
    if "sonnet-5" in SONNET_MODEL or "opus-5" in SONNET_MODEL:
        kwargs["thinking"] = {"type": "disabled"}
    try:
        msg = client.messages.create(**kwargs)
        text = "".join(b.text for b in msg.content if b.type == "text")
        usage = {
            "input_tokens": msg.usage.input_tokens,
            "output_tokens": msg.usage.output_tokens,
            "cache_read_input_tokens": getattr(msg.usage, "cache_read_input_tokens", 0) or 0,
            "cache_creation_input_tokens": getattr(msg.usage, "cache_creation_input_tokens", 0) or 0,
        }
        try:
            parsed = parse_json_response(text)
            return cmplid, {"parsed": parsed, "usage": usage, "error": None}
        except Exception as e:  # noqa: BLE001
            return cmplid, {"parsed": None, "usage": usage, "error": str(e)}
    except Exception as e:  # noqa: BLE001
        return cmplid, {"parsed": None, "usage": None, "error": f"api:{e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="output/batch_state_rare_topup.json")
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()

    state_path = Path(args.state)
    state = load_json(state_path)
    run_name = state["run_name"]
    sample = pd.read_parquet(state["sample_file"])
    rows = sample.to_dict(orient="records")
    row_by_cmplid = {str(r["CMPLID"]): r for r in rows}

    client = Anthropic()
    system_prompt = build_system_prompt()

    results1 = load_stage_results(state, "stage1_haiku")
    results2 = load_stage_results(state, "stage2_sonnet")

    escalate_ids = [
        cmplid for cmplid, r in results1.items()
        if r.get("parsed") and r["parsed"].get("confidence") in ESCALATE_ON
    ]
    remaining_ids = [cid for cid in escalate_ids if cid not in results2]
    log(
        f"Stage1={len(results1):,} escalate={len(escalate_ids):,} "
        f"sonnet_done={len(results2):,} remaining_ondemand={len(remaining_ids):,}"
    )

    # Drop incomplete batch chunk_00 from state so we don't resume it later.
    stage2 = state.setdefault("stage2_sonnet", {"chunks": []})
    stage2["chunks"] = [
        c for c in stage2.get("chunks", [])
        if not (c.get("name") == "chunk_00" and not c.get("retrieved"))
    ]
    save_state(state_path, state)

    if remaining_ids:
        remaining_rows = [row_by_cmplid[cid] for cid in remaining_ids]
        results_path = CHUNK_DIR / f"{run_name}_stage2_sonnet_chunk_00_ondemand.json"
        out = {}
        t0 = time.time()
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [
                ex.submit(call_sonnet, client, system_prompt, row)
                for row in remaining_rows
            ]
            for fut in as_completed(futs):
                cmplid, result = fut.result()
                out[cmplid] = result
                done += 1
                if done % 25 == 0 or done == len(remaining_ids):
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed else 0
                    log(
                        f"  ondemand progress {done}/{len(remaining_ids)} "
                        f"({elapsed:.0f}s, {rate:.2f} rows/s)"
                    )
                    save_json(results_path, out)

        save_json(results_path, out)
        # Cost summary uses on-demand rates for this chunk.
        n_err = sum(1 for r in out.values() if r.get("error"))
        n_parsed = sum(1 for r in out.values() if r.get("parsed"))
        cost = 0.0
        for r in out.values():
            if r.get("usage"):
                cost += ondemand_cost(r["usage"], SONNET_MODEL)
        summary = {
            "n": len(out),
            "n_errors": n_err,
            "n_parsed": n_parsed,
            "parse_rate": n_parsed / len(out) if out else 0.0,
            "error_rate": n_err / len(out) if out else 0.0,
            "cost_usd": round(cost, 4),
            "mode": "ondemand",
        }
        rec = {
            "name": "chunk_00_ondemand",
            "batch_id": None,
            "n_requests": len(out),
            "retrieved": True,
            "results_path": str(results_path),
            "summary": summary,
            "gate_passed": True,
            "mode": "ondemand",
        }
        stage2["chunks"] = [c for c in stage2["chunks"] if c["name"] != rec["name"]] + [rec]
        save_state(state_path, state)
        results2.update(out)
        log(
            f"On-demand Sonnet done: {summary['n']:,} rows, "
            f"{summary['n_errors']} errors, cost=${summary['cost_usd']:.4f}"
        )

    # Finalize (haiku stage still batch-priced; on-demand sonnet priced at live rates).
    final_rows = []
    total_cost = 0.0
    for cmplid, row in row_by_cmplid.items():
        r1 = results1.get(cmplid, {})
        escalated = cmplid in results2 and not results2[cmplid].get("error")
        chosen = results2.get(cmplid, {}) if escalated else r1
        if r1.get("usage"):
            total_cost += batch_cost(r1["usage"], "haiku")
        if escalated and chosen.get("usage"):
            # Canary was batch; on-demand chunk is live. Detect via state chunk names.
            total_cost += ondemand_cost(chosen["usage"], SONNET_MODEL)

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

    # Correct canary rows to batch pricing (they were double-counted as on-demand above).
    canary_path = None
    for rec in stage2.get("chunks", []):
        if rec.get("name") == "canary" and rec.get("retrieved"):
            canary_path = Path(rec["results_path"])
            break
    if canary_path and canary_path.exists():
        canary = load_json(canary_path)
        for cmplid, r in canary.items():
            if not r.get("usage"):
                continue
            # subtract on-demand, add batch
            total_cost -= ondemand_cost(r["usage"], SONNET_MODEL)
            total_cost += batch_cost(r["usage"], "sonnet")

    n_failed = sum(1 for r in final_rows if r["component_label"] is None)
    out_path = DATA_DIR / f"training_labels_{run_name}.parquet"
    pd.DataFrame(final_rows).to_parquet(out_path, index=False)
    summary1 = summarize_results(results1, "haiku")
    summary_path = OUT_DIR / f"training_labels_summary_{run_name}.json"
    with open(summary_path, "w") as f:
        json.dump({
            "run_name": run_name,
            "n_rows": len(final_rows),
            "n_failed": n_failed,
            "n_escalated": len(escalate_ids),
            "pct_escalated": len(escalate_ids) / len(results1) if results1 else None,
            "total_cost_usd": round(total_cost, 4),
            "sonnet_mode": "mixed_batch_canary_plus_ondemand_remainder",
            "output_parquet": out_path.name,
            "stage1_summary": summary1,
        }, f, indent=2)
    log(f"=== Final: {len(final_rows):,} rows, {n_failed} failed, cost ${total_cost:.4f} ===")
    log(f"Wrote {out_path}")
    log(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
