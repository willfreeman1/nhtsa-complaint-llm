# Decisions

Format: date, decision, options, why, what would reverse it.

---

## 2026-09-19 — Production code lives in `prod/`, research `scripts/` stay intact

**Decision:** Add a `prod/` package (eval harness, ingest, labeling tool,
shared preprocess wrapper) and keep writing working notes under `docs/`.
Stage 0 artifacts go in `output/prod/`. Do not edit research scripts to
make production work, and do not overwrite published eval JSON.

**Options:** (a) extend `scripts/` in place, (b) `prod/` alongside
`scripts/`, (c) a new repo.

**Why:** Non-negotiable 9: existing research results stay reproducible. The
brief's suggested layout was `prod/` + `docker/`. Docker waits for Stage 1.

**Reverse if:** a later packaging choice (installable wheel, single `src/`)
is clearly cleaner and a mechanical move costs less than living with two
trees.

---

## 2026-09-19 — Shared preprocess is a wrapper, not a copy

**Decision:** `prod/preprocess.py` imports `train_deberta.build_text` and
`teacher_prompt` helpers. Training and future serving both go through this
module (or the same underlying functions). No forked prompt/text builder.

**Options:** (a) copy the format string into prod, (b) import the research
functions, (c) extract a third shared module and rewrite research imports.

**Why:** Non-negotiable 7, with the smallest diff. (c) is cleaner but
touches research scripts.

**Reverse if:** we extract a real shared package in Stage 1 and migrate
research imports in one commit.

---

## 2026-09-19 — Stage 0 measures DeBERTa fully; Qwen CPU is a probe unless it finishes

**Decision:** This machine is an i9-10900, 128 GB RAM, **CPU-only** PyTorch.
DeBERTa (704 MB, ~5 rows/s claimed) gets a full 551-row reproduction plus
timing/memory. Qwen 7B + LoRA has no local base-model cache; a full 551-row
CPU generate would be hours and needs a ~15 GB download. Stage 0 will
download/run Qwen only if the DeBERTa + ingest path leaves enough time;
otherwise the memo reports a timed subset and does not invent a Qwen
accuracy.

**Options:** (a) full Qwen rerun on CPU anyway, (b) rent a GPU for the
rerun, (c) probe + defer GGUF/GPU to a follow-up sitting.

**Why:** Honest numbers over feature breadth. Renting a GPU for a
reproduction we already published is not worth $25+ until serving
quantization is the question being answered.

**Reverse if:** a GPU is already sitting idle, or the current-key result
makes Qwen-vs-DeBERTa the decision that matters.

---

## 2026-09-19 — Provisional serving champion is DeBERTa

**Decision:** Serve DeBERTa. Keep Qwen as the offline benchmark. Host
unattended scoring on GitHub Actions cron or a $5–10 VPS, not a GPU box.

**Why:** DeBERTa reproduced at 0.822 [0.788, 0.852], 4.34 rows/s, 1.27 GB
RSS. 300 rows/day is about a minute. Always-on 7B GPU hosting blows the
~$30/month cap. The current key does not exist yet, so this is provisional.

**Reverse if:** the frozen current key shows DeBERTa clearly worse than a
servable Qwen variant (GGUF/Q4) by more than the promotion-gate noise, *and*
that variant fits the budget.

---

## 2026-09-19 — Ingest from the ODI flat file (and 5-year chunks), not the by-vehicle API

**Decision:** Daily source of record is
`https://static.nhtsa.gov/odi/ffdd/cmpl/FLAT_CMPL.zip` (full history) or
`COMPLAINTS_RECEIVED_YYYY-YYYY.zip` (same rows, split by received year).
The public API (`/complaints/complaintsByVehicle`) is lookup-by-vehicle
only and is not a "new since date" feed.

**Options:** (a) full zip every day, (b) newest 5-year chunk most days plus
periodic full-file revision audit, (c) scrape/API by make-model-year.

**Why:** NHTSA documents daily publication and, critically, that existing
rows (including `CMPLID` and component name) can change. (c) cannot see
revisions or a complete daily increment. (b) is the likely production
shape if the full 350 MB zip is wasted work; Stage 0 will measure that.

**Reverse if:** NHTSA adds a real incremental/change-feed API, or the
revision audit shows old rows never change after 2021 (then newest-chunk
only is enough).

---

## 2026-09-19 — Complaint identity is ODINO + narrative, not CMPLID

**Decision:** Treat `CMPLID` as a dump-local sequence number. Across dumps,
join on `ODINO` plus a whitespace-normalized narrative fingerprint. Ingest
should store both `cmplid` (as published that day) and that fingerprint.

**Why:** `CMPL.txt` says CMPLID is updateable. A naive CMPLID join of the
2026-08-28 copy vs the 2026-09-19 2025–26 chunk looked like 38k component
edits. The same rows rematched by fingerprint show **27,430 CMPLID moves**,
**3** COMPDESC string changes (all more-specific suffixes of the same
top-level category), and 2 make / 30 model / 2 year edits. Reporting the
CMPLID-join numbers as "revisions" would have been wrong.

**Reverse if:** NHTSA documents a stable surrogate key, or a later full-file
audit shows CMPLID becoming stable.

---

## 2026-09-19 — Promotion tolerances (2 pp overall / 3 pp hard / overlapping CI = no promote)

**Decision:** See `THRESHOLDS.md` Stage 3. Written before any retrain.

**Options:** (a) "any point-estimate win", (b) "statistically significant
win only", (c) point-estimate win on current + capped legacy drop +
overlapping current-key CI blocks promotion.

**Why:** Current-key n will be ~150–300. A 4-point gap is often noise
(n=150 is roughly ±8 pp). (a) over-promotes. (b) may never promote. (c)
lets a clear current-key win through if it does not sacrifice the legacy
set, and refuses coin flips.

**Reverse if:** the first current-key CI is so wide that nothing can
promote, and we choose to grow n before changing the rule — growing n is
the preferred reverse, not loosening the gate.

---

## 2026-09-19 — Current-key sample: n=150, seed 20260919, filed after 2026-08-26

**Decision:** Uniform random sample of vehicle (`PROD_TYPE='V'`) complaints
with `LDATE` (date received) **after** 2026-08-26, excluding every
`cmplid` in any training parquet or in `gold_eval_set_v3`. First-pass
target 150. Optional separate driver-assist oversample (n=50) written to
its own file, never mixed into the random-sample accuracy number.

**Why:** Random sampling is what makes the number apply to live traffic.
`LDATE` is "date complaint received by NHTSA" and matches the brief's
"filed after" better than vehicle model year. Seed is fixed so the queue
is reproducible.

**Reverse if:** the new dump has fewer than ~200 eligible rows (then take
all of them and say so).
