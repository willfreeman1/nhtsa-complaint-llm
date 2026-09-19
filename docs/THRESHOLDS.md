# Thresholds

Every go/adjust, alert, and promotion threshold lives here **before** the run
it applies to. Changing a number after seeing the result of that run needs an
entry in `DECISIONS.md`.

Status key: **LOCKED** = applies to a run that has started or will start;
**DRAFT** = written early so Stage 2/3 are not invented under the data.

---

## Stage 0 — reproduction (LOCKED 2026-09-19, before the rerun)

Published v3 numbers to match (`output/rung2_percategory_and_novel_full_62k_plus_rare_v3.json`,
`output/rung3_percategory_and_novel_full_62k_plus_rare_v3.json`):

| Model | Overall | Hard | Easy (dual-agreement) | Macro-F1 |
|---|---|---|---|---|
| DeBERTa | 0.822 | 0.733 | 0.931 | 0.775 |
| Qwen LoRA | 0.862 | 0.766 | 0.980 | 0.799 |

**Pass:** overall accuracy agrees with the published JSON to 3 decimal places
(e.g. DeBERTa `0.822`). Hard/easy/macro-F1 are reported; a miss there is
explained, not a silent fail.

**If it fails:** stop and explain (code drift, data drift, or numeric
difference) before treating any later Stage 0 number as comparable to the
writeup.

## Stage 0 — serving comparison (LOCKED 2026-09-19)

This table decides the serving champion. It is a measurement, not a
promotion. Record accuracy (with Wilson 95% CI), rows/s, peak RSS, and
estimated monthly host cost for each variant actually measured.

Qwen full-precision on this CPU-only box is allowed as a throughput probe
(small n) rather than a full 551-row rerun if a full rerun cannot finish in
a sitting. That limitation must be stated in the memo; it does not invent a
Qwen accuracy number.

## Stage 3 — promotion gate (LOCKED 2026-09-19, before any retrain)

A challenger is promoted only if **all** of the following hold:

1. Current-complaints answer key (`gold_current_v1`, once frozen): challenger
   overall accuracy **point estimate** is higher than the champion's.
2. Legacy answer key (`gold_eval_set_v3`): challenger overall accuracy drops
   by **at most 2.0 percentage points**.
3. Legacy hard tier: challenger hard accuracy drops by **at most 3.0
   percentage points**.
4. If the two current-key Wilson 95% CIs overlap, **do not promote**. That is
   a no-clear-win, not a win. Retrain or collect more labels.

Teacher disagreement and NHTSA-field agreement are not inputs to this gate.

Rollback is moving the champion alias back. No other rule.

## Stage 2 — monitoring alerts (DRAFT, lock before first unattended week)

Baselines will be the first 14 days of champion scores on live traffic, once
those exist. Until then these are placeholders so the dashboard work has
numbers to wire, not licenses to page anyone.

| Signal | Draft trip | Why this shape |
|---|---|---|
| Predicted-category mix vs. 28-day reference | Jensen–Shannon > 0.08 | Catches a driver-assist-scale mix shift; exact cutoff to be calibrated on the historical replay |
| UNKNOWN OR OTHER rate | +5 pp vs. 28-day mean | Abstention spike is a cheap broken-input / schema-shift alarm |
| Median confidence | −0.10 vs. 28-day median | Soft failure mode before accuracy is measurable |
| Agreement with NHTSA `COMPDESC` | change of 10 pp from the post-2015 ~42% plateau, sustained 7 days | Only sharp breaks; the level itself is not accuracy |
| Weekly teacher disagreement (n≈100) | 15 pp above the first-month mean | Drift alarm, never an accuracy number |
| Live hand-check (n≈50/month) | Wilson 95% CI entirely below champion-on-current-key minus 5 pp | The only live accuracy alarm |

None of these fire email until Stage 2. They write an alert row and a
dashboard banner.
