# Stage 0 feasibility memo

2026-09-19. Priorities: honest numbers > unattended runs > feature breadth.

Spend: **$0** (`docs/COST_LOG.md`).

---

## 1. Machine

Intel i9-10900 (10c/20t), 128 GB RAM, **CPU-only** PyTorch 2.13. DeBERTa
weights (704 MB) and the Qwen LoRA adapter (154 MB) are on disk. No Qwen
base-model cache.

Research copy of NHTSA data: `data/FLAT_CMPL.zip` from 2026-08-28,
`LDATE`/`DATEA` max **20260826**, 2,239,860 rows.

---

## 2. Reproduction

Threshold locked first (`docs/THRESHOLDS.md`): overall accuracy must match
the published v3 JSON to 3 decimals.

| | Published | This rerun | Wilson 95% | Pass |
|---|---|---|---|---|
| DeBERTa overall | 0.822 | **0.822** (453/551) | [0.788, 0.852] | yes, diff 0 |
| DeBERTa hard | 0.733 | 0.733 (222/303) | [0.680, 0.779] | yes |
| DeBERTa easy | 0.931 | 0.931 (231/248) | [0.893, 0.957] | yes |
| DeBERTa macro-F1 | 0.775 | 0.775 | — | yes |
| DeBERTa rows/s | ~5 (writeup, 1 core) | **4.34** | — | |
| DeBERTa peak RSS | 704 MB weights | **1.27 GB** | — | |
| Qwen overall | 0.862 | *not rerun* | | deferred |

Source: `output/prod/stage0_deberta_reproduce.json`.

Qwen was deferred on purpose (`docs/DECISIONS.md`): no base-model cache, and
a 551-row CPU generate would be hours. That is a gap, not a hidden number.

---

## 3. Legacy answer key by received year

Join of `gold_eval_set_v3.json` (n=551) to `data/cmpl.parquet` on `CMPLID`.
Zero missed joins. **82 / 551 rows have `LDATE` in 2023 or later**, matching
the brief. Accuracy below is this sitting's DeBERTa rerun.

| Received year | n | Share | Acc | Wilson 95% |
|---|---|---|---|---|
| 1995–2009 | 172 | 31.2% | 0.820 | [0.756, 0.870] |
| 2010–2014 | 110 | 20.0% | 0.791 | [0.706, 0.856] |
| 2015–2019 | 138 | 25.0% | 0.855 | [0.787, 0.904] |
| 2020–2022 | 49 | 8.9% | 0.898 | [0.782, 0.956] |
| 2023–2024 | 45 | 8.2% | 0.778 | [0.637, 0.875] |
| 2025–2026 | 37 | 6.7% | 0.757 | [0.599, 0.866] |

Point estimates dip after 2022. **The intervals overlap the overall
[0.788, 0.852].** This is not a finding that the model has failed on recent
complaints. It is the reason a current-complaints key of 150–300 exists:
n=37 cannot decide a 6-point gap.

---

## 4. Ingestion facts

### How often does it update?

NHTSA's `CMPL.txt` says **daily**. The datasets HTML page still showed
2026-08-30 when fetched; that page is stale. **HEAD `Last-Modified` is the
ground truth.**

Probed 2026-09-19 20:46 UTC (`output/prod/nhtsa_headers.json`):

| File | Last-Modified | Downloaded size |
|---|---|---|
| `FLAT_CMPL.zip` | Sat, 19 Sep 2026 10:23:52 GMT | 372 MB |
| `COMPLAINTS_RECEIVED_2025-2026.zip` | Sat, 19 Sep 2026 09:28:49 GMT | 40 MB |

The file **did update today**, 22 days after the research copy. Newest
received date in the new 2025–26 chunk: **2026-09-17**.

### Lighter source than the 350 MB zip?

Yes: the newest 5-year chunk. The by-vehicle API
(`/complaints/complaintsByVehicle`) is lookup-only, not an incremental feed.

### New complaints since 2026-08-26

From the 2025–26 chunk vs the research parquet
(`output/prod/ingest_compare.json`):

| | |
|---|---|
| New dump rows (2025–26 chunk) | 195,389 |
| Research rows in that LDATE window | 187,556 |
| Added `CMPLID`s | 7,833 (7,676 vehicle) |
| Removed IDs in-window | 0 |
| Eligible for the current key (`PROD_TYPE=V`, `LDATE>20260826`, not in train/gold, narrative ≥20 chars) | **7,589** |

That is ~350 vehicle complaints/day over 22 days, in line with the brief's
~300/day.

### Do existing rows get revised?

`CMPLID` is **not a stable key**. `CMPL.txt` says it is updateable. A naive
CMPLID join looked like 38,696 component edits. That was remapping:
narratives had slid to neighboring IDs.

Rematched on `ODINO` + first 400 chars of the narrative
(`output/prod/revision_audit.json`):

| | n |
|---|---|
| Same complaint, `CMPLID` moved | 27,430 |
| Same complaint, `COMPDESC` string changed | **3** |
| Same complaint, make / year changed | 2 / 2 |
| Same complaint, model string changed | 30 |

The three `COMPDESC` changes were more-specific suffixes of the **same
top-level category** (`WHEELS` → `WHEELS:LUGS/...`, `ELECTRICAL SYSTEM` →
`...BATTERY`, `BACK OVER PREVENTION: WARNINGS` → `...CAMERA`). In this
window, top-level recodes of an already-published complaint were
effectively **zero**.

Production ingest must store a fingerprint (`ODINO` + normalized narrative)
and treat `CMPLID` as dump-local. Newest 5-year chunk is enough for new
rows; a periodic full-file pass is only needed if we start seeing real
top-level recodes.

---

## 5. Current-complaints answer key

Blind queue is built. Predictions are sealed. No model has been scored on
it.

| | |
|---|---|
| Random sample | **150**, seed `20260919` |
| Received dates | 2026-08-27 through 2026-09-17 |
| ADAS oversample | 50, **separate file**, not for the headline accuracy |
| Blocked train+gold IDs | 68,803 |
| First-pass labels | **0 / 150** |

Tool: `python -m prod.labeling.app` → http://127.0.0.1:8765

The HTTP API serves `cmplid`, make/model/year, narrative, `ldate`, and class
definitions. It does not serve NHTSA's field or any model output. First-pass
saves go to `output/prod/gold_current_v1_first_pass.jsonl`. After that pass
is finished we reveal sealed predictions and log any later changes
separately.

**This is the remaining Stage 0 human step.** Scoring DeBERTa / Qwen / the
teacher on current traffic waits on it.

---

## 6. Serving champion and hosting

`output/prod/serving_table.json`:

| Variant | Acc (legacy v3) | rows/s | Peak RSS | Monthly host |
|---|---|---|---|---|
| **DeBERTa CPU** | 0.822 [0.788, 0.852] | 4.34 | 1.27 GB | **$0–10** |
| Qwen 7B LoRA | 0.862 published, not rerun | not measured | needs ~15 GB base | GPU, over budget if always-on |

**Provisional serving champion: DeBERTa.** 300 complaints/day is about 70
seconds. The laptop-off constraint is met by a GitHub Actions cron or a
$5–10 VPS. Qwen stays the offline accuracy benchmark unless the current key
shows a gap large enough to justify GPU spend.

No outcome here kills the project. If the current key later shows a real
drop, Stage 3's first live retrain is the story. If it holds, monitoring
plus the historical replay is the story.

---

## 7. Surprises

1. The NHTSA datasets **page date is not trustworthy**; HEAD is.
2. The brief's "82 of 551 from 2023+" is exact.
3. `CMPLID` remaps between dumps. Reporting the 38k CMPLID-join
   "COMPDESC changes" as revisions would have been a false finding.
4. Real content edits in the 2025–26 window are rare and, for component,
   suffix-only.
5. `cmpl_clean.parquet` dropped `LDATE`/`DATEA`. Year slices have to read
   `cmpl.parquet`. Keep those columns in the Stage 1 clean table.
6. Published training-run JSON for DeBERTa still shows n=498 / 85.3%. The
   writeup's 82.2% is the **v3** file. Reproduction targeted v3 and matched
   it exactly.
7. Year-slice point estimates dip after 2022; the CIs do not support calling
   that a failure.

---

## 8. Checkpoint 0

| Item | Status |
|---|---|
| `docs/THRESHOLDS.md` | locked, including the Stage 3 promotion gate |
| `docs/DECISIONS.md` | written |
| `docs/COST_LOG.md` | $0 |
| Serving table | DeBERTa measured; Qwen deferred with that said |
| Current-key file + first-pass vs revised log | queue + sealed sidecar ready; **0 labels** |
| This memo | filled except current-key accuracy |

**Will:** first-pass blind labels on the 150-row queue at
http://127.0.0.1:8765 (server is running). Do not open
`output/prod/gold_current_v1_sealed.json` until that pass is done.
