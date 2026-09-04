"""One-off recovery script: the stage2_sonnet/chunk_00 batch for the rare_topup run
got forcibly canceled mid-flight (591/899 requests canceled, 308 succeeded) when the
long-running parent process was killed by the environment around the 2h47m mark.
Those 591 rows silently fell back to their original Haiku "medium confidence" answer
in the final merge (since they have no valid Sonnet result) instead of getting the
intended Sonnet re-label.

This script:
  1. Finds the 591 canceled cmplids from the saved chunk_00 results.
  2. Resubmits exactly those rows to Sonnet as a fresh batch (reusing the same
     request-building logic as run_batch_cascade_labeling.py, unmodified).
  3. Merges the new results into the saved chunk_00 results file and the state file's
     summary (best-effort accounting only -- the state file's own gate/summary fields
     are left as originally recorded for audit-trail purposes; a corrected combined
     summary is printed and saved separately).
  4. Rewrites data/training_labels_rare_topup.parquet and
     output/training_labels_summary_rare_topup.json by re-running the exact same
     finalize logic as the main script's `main()`, now with the gap-filled Sonnet
     results.

Does NOT modify run_batch_cascade_labeling.py.
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from teacher_prompt import ALL_VALID_COMPONENTS, build_system_prompt, build_user_message, bucket_component
from run_batch_cascade_labeling import (
    build_request, retrieve_results, summarize_results, print_gate, batch_cost,
    save_json, load_json, POLL_INTERVAL_S, SONNET_MODEL, HAIKU_MODEL, log,
)

load_dotenv(override=True)
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent.parent / "output"
CHUNK_DIR = OUT_DIR / "batch_chunks"
RUN_NAME = "rare_topup"

CHUNK00_PATH = CHUNK_DIR / f"{RUN_NAME}_stage2_sonnet_chunk_00.json"
STATE_PATH = OUT_DIR / f"batch_state_{RUN_NAME}.json"
CANARY_PATH = CHUNK_DIR / f"{RUN_NAME}_stage2_sonnet_canary.json"


def main():
    client = Anthropic()
    system_prompt = build_system_prompt()

    chunk00 = load_json(CHUNK00_PATH)
    canceled_ids = [cid for cid, r in chunk00.items() if r.get("error") == "sonnet:canceled"]
    log(f"Found {len(canceled_ids):,} canceled cmplids in {CHUNK00_PATH.name}")
    if not canceled_ids:
        log("Nothing to retry.")
        return

    sample = pd.read_parquet(DATA_DIR / f"training_sample_{RUN_NAME}.parquet")
    row_by_cmplid = {str(r["CMPLID"]): r for r in sample.to_dict("records")}
    missing = [cid for cid in canceled_ids if cid not in row_by_cmplid]
    if missing:
        raise RuntimeError(f"{len(missing)} canceled cmplids not found in sample parquet: {missing[:5]}")

    requests = [
        build_request(
            cid, SONNET_MODEL, system_prompt,
            build_user_message(
                narrative=row_by_cmplid[cid]["CDESCR"], make=row_by_cmplid[cid]["MAKETXT"],
                model=row_by_cmplid[cid]["MODELTXT"], year=row_by_cmplid[cid]["YEARTXT"],
            ),
        )
        for cid in canceled_ids
    ]
    log(f"Submitting retry batch: {len(requests):,} requests to {SONNET_MODEL}")
    batch = client.messages.batches.create(requests=requests)
    log(f"  batch_id={batch.id} status={batch.processing_status}")

    t0 = time.time()
    while True:
        b = client.messages.batches.retrieve(batch.id)
        counts = b.request_counts
        log(f"  {time.time()-t0:.0f}s elapsed: processing={counts.processing} "
            f"succeeded={counts.succeeded} errored={counts.errored} "
            f"canceled={counts.canceled} expired={counts.expired} status={b.processing_status}")
        if b.processing_status == "ended":
            break
        time.sleep(POLL_INTERVAL_S)

    retry_results = retrieve_results(client, batch.id, "sonnet")
    retry_summary = summarize_results(retry_results, "sonnet")
    print_gate("stage2_sonnet/retry_chunk00_gap", retry_summary, is_canary=False)

    n_fixed = sum(1 for cid in canceled_ids if retry_results.get(cid, {}).get("parsed"))
    log(f"Retry resolved {n_fixed}/{len(canceled_ids)} previously-canceled rows")

    # Merge: retry results overwrite the canceled entries in chunk00's saved results.
    merged_chunk00 = dict(chunk00)
    merged_chunk00.update(retry_results)
    save_json(CHUNK00_PATH, merged_chunk00)
    log(f"Wrote merged {CHUNK00_PATH}")

    # Recombine full stage2 results (canary + merged chunk00) and rebuild final rows,
    # mirroring run_batch_cascade_labeling.main()'s finalize logic exactly.
    canary_results = load_json(CANARY_PATH)
    results2 = dict(canary_results)
    results2.update(merged_chunk00)

    stage1_dir_files = [
        CHUNK_DIR / f"{RUN_NAME}_stage1_haiku_canary.json",
        CHUNK_DIR / f"{RUN_NAME}_stage1_haiku_chunk_00.json",
        CHUNK_DIR / f"{RUN_NAME}_stage1_haiku_chunk_01.json",
    ]
    results1 = {}
    for p in stage1_dir_files:
        results1.update(load_json(p))

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
    n_escalated_final = sum(1 for r in final_rows if r["escalated"])
    log(f"=== Final (gap-filled): {len(final_rows):,} rows, {n_failed} failed/unparseable, "
        f"{n_escalated_final} escalated to sonnet, total cost ${total_cost:.4f} (batch-rate) ===")

    out_path = DATA_DIR / f"training_labels_{RUN_NAME}.parquet"
    pd.DataFrame(final_rows).to_parquet(out_path, index=False)
    log(f"Wrote {out_path}")

    # Preserve original stage1 summary from state file for the report.
    state = load_json(STATE_PATH)
    stage1_chunks = state["stage1_haiku"]["chunks"]
    stage1_summary_combined = {
        "n": sum(c["summary"]["n"] for c in stage1_chunks),
        "n_errors": sum(c["summary"]["n_errors"] for c in stage1_chunks),
        "cost_usd": round(sum(c["summary"]["cost_usd"] for c in stage1_chunks), 4),
    }
    n_escalate_candidates = sum(
        1 for cmplid, r in results1.items()
        if r.get("parsed") and r["parsed"].get("confidence") == "medium"
    )

    summary_path = OUT_DIR / f"training_labels_summary_{RUN_NAME}.json"
    with open(summary_path, "w") as f:
        json.dump({
            "run_name": RUN_NAME,
            "n_rows": len(final_rows),
            "n_failed": n_failed,
            "n_escalated": n_escalated_final,
            "pct_escalated": n_escalated_final / len(results1) if results1 else None,
            "total_cost_usd": round(total_cost, 4),
            "cost_per_100k_rows": round(total_cost / len(final_rows) * 100_000, 2) if final_rows else None,
            "output_parquet": out_path.name,
            "stage1_summary": stage1_summary_combined,
            "recovery_note": (
                f"stage2_sonnet/chunk_00 batch was force-canceled mid-run (591/899 requests) "
                f"when the parent process was killed by the environment; {len(canceled_ids)} "
                f"canceled rows were resubmitted in a follow-up batch and {n_fixed} were "
                f"successfully re-labeled by sonnet. See retry_summary below."
            ),
            "retry_summary": retry_summary,
        }, f, indent=2)
    log(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
