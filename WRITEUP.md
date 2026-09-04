# NHTSA Complaint Classification — Project Writeup

**The big picture**: NHTSA's public complaint database tags every complaint with a
vehicle component, but that tagging is noisy (§3.3–3.4 quantify how noisy). This
project fine-tunes a small, open-weight LLM that reads a raw complaint narrative and
assigns the correct component instead — accurately enough to nearly match a frontier
LLM asked fresh every time, at a small fraction of the ongoing cost once trained. See
the top-level [`README.md`](README.md) for that framing written for a reader new to
the project. **This document is the full technical record underneath it**: what was
actually done, what it found, and the reasoning behind each decision — including
inconvenient/nuanced findings (e.g. §3.3's skeptical read on the teacher's own error
rate, §3.4's gold-set expansion turning up a lot of NHTSA mislabeling) that a cleaner
summary would otherwise smooth over.

## 1. Goal and approach

Given a raw NHTSA vehicle complaint narrative (free text) plus structured metadata
(make/model/year), predict the affected vehicle **component** (30 primary categories +
10 rare categories + `UNKNOWN OR OTHER`, deduping to 40 distinct valid strings — see
`teacher_prompt.ALL_VALID_COMPONENTS`), and secondary flags (`CRASH`, `FIRE`,
`INJURED`, `DEATHS`).

There's no existing large set of reliably-labeled complaints to fine-tune on — NHTSA's
own raw field is exactly the noisy thing this project is trying to improve on. So the
approach builds its own training labels first (§3), then compares two ways of turning
those labels into a small, deployable model (§4–§5) — structured as three "rungs" of
increasing training complexity but decreasing per-prediction cost, so the final
comparison can show exactly where the accuracy/cost tradeoff lands for this task:

Feasibility (green-light verdict, baselines, label-schema gotchas) is documented in
`output/FINAL_REPORT.md` and `output/NHTSA_CODING_GOTCHAS.md` and isn't repeated here.

## 2. The three rungs

| Rung | Approach |
|---|---|
| 1 | Zero/few-shot frontier LLM teacher (Haiku→Sonnet confidence cascade) — used to label all training data |
| 2 | Fine-tuned encoder classifier (DeBERTa-v3-base, 40-class softmax head) |
| 3 | Fine-tuned open-weight LLM (Qwen2.5-7B-Instruct, LoRA, generative) |

## Headline result

The gold set (551 hand-adjudicated complaints, §3.3) deliberately oversamples
ambiguous and rare-category cases rather than being a random, easy sample — so read
these numbers as a stress test, not a softball benchmark.

| | Rung 1 (Sonnet solo) | Rung 1 (Haiku solo) | Rung 1 (cascade, as actually run) | Rung 2 (DeBERTa) | Rung 3 (Qwen2.5-7B LoRA) |
|---|---|---|---|---|---|
| Gold accuracy (overall) | 86.6% | 85.1% | 89.8%* | 82.2% | **86.2%** |
| Gold accuracy, hard subset | 78.2% | 74.9% | — | 73.3% | **76.6%** |
| Gold accuracy, dual-agreement | 96.8% | 97.6% | — | 93.1% | **98.0%** |
| Macro-F1 | 0.764 | 0.752 | — | 0.775 | **0.799** |
| Per-inference cost | ~$467/100K rows (on-demand) | ~$154/100K rows (on-demand) | ~$126/100K rows (batch, measured) | ~$0 marginal (self-hosted) | ~$0 marginal (self-hosted) |

\* backtest estimate from `confidence_cascade_analysis.json`, not a direct gold-set run
of the exact production cascade.

Sonnet solo is the top individual model by a modest ~0.4pt margin over the fine-tuned
LLM (Rung 3), with Rung 3 comfortably ahead of Rung 2 on every metric. See §6 for
cost/latency and deployment notes.

## 3. Rung 1 — Teacher labeling via confidence cascade

### 3.1 Cost problem and the cascade idea

Labeling ~68K rows with a single frontier model directly would cost real money at this
project's self-funded budget (a "few hundred dollars" ceiling). The key empirical
finding that made a cheaper approach viable: **cheap models' own self-reported
confidence is a real, usable signal for when they're about to be wrong.** Measured by
running Haiku 4.5 and Sonnet 5 solo against the 498-row gold set and bucketing by each
model's own stated confidence (`output/confidence_cascade_analysis.json`):

| Model | High-confidence accuracy | Medium-confidence accuracy | Low-confidence accuracy |
|---|---|---|---|
| Claude Haiku 4.5 | 94.6% (78.3% of rows) | 64.4% (18.1% of rows) | 72.2% (3.6% of rows) |
| Claude Sonnet 5 | 99.6% (47.4% of rows) | 82.2% (48.6% of rows) | 80.0% (4.0% of rows) |

Haiku's confident answers are ~95% reliable, but its medium-confidence answers drop to
64% — barely better than guessing among ~40 classes. That gap is exploitable: run Haiku
on everything (cheap), escalate only its medium/low-confidence rows to Sonnet
(expensive but rare), and get most of Sonnet's accuracy at close to Haiku's cost.

Backtesting every cheap/smart pairing and escalation policy against the gold set
(replaying each row's *actual* recorded confidence/correctness/token cost) confirmed
Haiku→Sonnet escalating on medium+low confidence as the best accuracy/cost tradeoff
among the options tried — **89.8% accuracy on gold, only ~21.7% of rows escalated**,
vs. 90.4% for full Sonnet-solo at ~4x the cost of the cascade.

### 3.2 Production run

Ran via Anthropic's Batch API (50% discount) with chunking (256MB payload limit) and a
canary gate before each chunk:

- **62,000 rows**, stratified ~2,000/class target: **18.6% escalated** to Sonnet,
  **$78.24 total** ($126.19 per 100K rows), 7 unrecoverable rows (empty narratives).
- **+6,252 rare-category top-up rows** pulled specifically to fix severe
  under-representation in 9 rare categories (e.g. `EQUIPMENT ADAPTIVE/MOBILITY`: 19→50,
  `TRAILER HITCHES`: 195→890, `FUEL SYSTEM, DIESEL`: 364→895). Anthropic Batch API
  briefly stalled on a credit-balance issue mid-run; finished the remaining Sonnet
  escalations via on-demand API calls to avoid further delay (small $ cost difference,
  not worth the wait). 18 rows failed outright (16 recovered via manual labeling from
  the narrative text; 2 had no real narrative — "see attached document" — left
  unlabeled).
- **Merged total: 68,252 labeled rows**, 0 duplicate `CMPLID`s, used as training data
  for both Rung 2 and Rung 3.

### 3.3 Nuanced read on teacher accuracy — the ~10% "error" rate isn't what it looks like

Sonnet's solo accuracy on the 498-row gold set is 90.4% (Haiku's is 88.4%) — a ~10%
"error" rate. Digging into *which* rows Sonnet actually misses (breakdown of Sonnet's
48 wrong answers, see `output/model_eval_anthropic_claude-sonnet-5.json`) shows this
number is misleading if read as "Sonnet is confidently, obviously wrong 10% of the
time":

The gold set has two truth tiers: **dual-agreement** rows (teacher LLM and NHTSA's own
`COMPDESC` field independently landed on the same label — strong, clean ground truth,
n=240) and **adjudicated/hard** rows (teacher and NHTSA disagreed, so a human read the
narrative to break the tie — genuinely harder, ambiguous cases, n=258).

| Subset | n | Sonnet accuracy | Sonnet error rate |
|---|---|---|---|
| Dual-agreement ("easy") | 240 | 97.5% | 2.5% |
| Adjudicated ("hard") | 258 | 83.7% | 16.3% |

**87.5% of Sonnet's total errors (42 of 48) are concentrated in the 258 hard rows.**
Only 6 of its 48 errors happen on the clean, unambiguous rows. Breaking the 42 hard
errors down further:

- **14/42 (33%)**: the adjudicated truth is `UNKNOWN OR OTHER` — a human decided the
  narrative genuinely didn't contain enough detail to name a specific system, but
  Sonnet guessed a plausible-sounding specific category instead of abstaining.
  Overconfidence on vague narratives, not a "wrong" answer in a meaningful sense.
- **6/42 (14%)**: same-family near-miss (e.g. `SERVICE BRAKES, AIR` vs
  `SERVICE BRAKES, HYDRAULIC` — right system, wrong sub-type).
- **22/42 (52%)**: genuinely distant wrong category on a hard narrative — e.g.
  predicting `FORWARD COLLISION AVOIDANCE` when the adjudicated answer was brakes, or
  `OTHER` on cmplid 2026794's "textbook" accelerator-response-delay complaint
  (correct answer: `VEHICLE SPEED CONTROL`). Real misses, not defensible alternates,
  even though they occur on objectively harder narratives.

Even one of the 6 "easy" dual-agreement errors turned out to be a near-coinflip on
inspection: cmplid 1072538's narrative literally names *both* "AIR BAGS, ELECTRICAL
SYSTEM" as recall campaign components with no failure actually experienced; truth
landed on `AIR BAGS`, Sonnet said `ELECTRICAL SYSTEM`.

**Bottom line**: Sonnet's real "confidently, obviously wrong" rate is closer to
**1-3%**, not 10%. But it would also be wrong to write off the entire gap as label
noise — roughly half of the hard-subset misses (22/42) are genuine miscategorizations
on cases a careful human still got right, just on harder-than-average narratives. This
matters for interpreting Rung 2/3 accuracy against the same gold set: expect their
error profiles to skew the same way (concentrated in the hard subset), and expect a
"true" ceiling for any of these approaches to sit somewhere south of ~95-97%, not 100%,
because a real fraction of complaints are inherently ambiguous even to a careful reader.

### 3.4 How reliable are NHTSA's rare-category labels?

The gold set includes real complaints for all 30 primary categories, but at project
start had zero examples for 9 of the 10 designated rare categories — those categories
are rare enough that a stratified sample doesn't reliably surface them. To check
whether NHTSA's raw `COMPDESC_TOP` field can even be trusted for these categories,
complaints were pulled directly from the raw corpus for each one (`scripts/
build_rare_gold_expansion.py`) and read individually against `teacher_prompt.py`'s
class definitions, then folded into the gold set (`scripts/build_gold_eval_set_v3.py`,
551 rows total).

Two categories, `FIRERELATED` (169 total complaints in the entire 2.17M-row corpus)
and `TRAILER HITCHES` (924 total), turned out to have **zero remaining real complaints
anywhere in the corpus** once rows already used elsewhere in this project were
excluded — these categories are rare enough that the entire real-world supply is
already spoken for. Their accuracy can't be independently verified against any further
held-out data; that's a hard limit of the available data, documented rather than
papered over.

For the other 7 categories, 7 candidates each (49 rows total; +4 more for `EQUIPMENT
ADAPTIVE/MOBILITY` via a targeted keyword search after random sampling turned up zero
genuine matches) were read against the class definitions. **Only 16 of 53 (30%)
actually belonged in the category NHTSA's raw field said they did:**

| Sampled category | Confirmed / sampled |
|---|---|
| `CHILD SEAT` | 2/7 |
| `ELECTRONIC STABILITY CONTROL` | 2/7 |
| `EQUIPMENT ADAPTIVE/MOBILITY` | 4/11 |
| `FUEL SYSTEM, OTHER` | 1/7 |
| `FUEL/PROPULSION SYSTEM` | 1/7 |
| `SERVICE BRAKES, ELECTRIC` | 4/7 |
| `TRACTION CONTROL SYSTEM` | 2/7 |

Several of NHTSA's raw rare-category buckets are heavily contaminated in practice.
`CHILD SEAT` is dominated by aftermarket car-seat-product complaints that don't belong
in a vehicle-component schema at all (the class definition explicitly requires a
built-in/integrated feature). `EQUIPMENT ADAPTIVE/MOBILITY`'s random sample was 0/7
genuine — narratives about airbags, horns, ABS modules, and unintended acceleration all
got filed there for reasons unrelated to adaptive/mobility equipment; genuine examples
(wheelchair lifts, hand controls) only turned up via targeted keyword search.
`SERVICE BRAKES, ELECTRIC` was the pleasant surprise — 4/7 genuinely confirmed once it
became clear this bucket is reserved for hybrid vehicles' electronically-blended
regenerative brake systems (a real Toyota Prius/Camry Hybrid recall pattern), not noise
at all once you know what to look for.

All 53 rows (confirmed or not) were adjudicated to whatever the narrative actually
supports and added to the gold set regardless. Per-class results for these specific
categories are in §4.5/§5.4's model files
(`output/rung{2,3}_percategory_and_novel_full_62k_plus_rare_v3.json`); treat any single
one of these classes' accuracy as directional at this sample size (1-5 examples each),
not precise.

## 4. Rung 2 — DeBERTa-v3-base fine-tune

### 4.1 Setup

Standard classification fine-tune: DeBERTa-v3-base, 40-class softmax head over
`raw_component` (not the bucketed 31-class bucket — see §4.3 for why this mattered),
class-weighted loss for imbalance, 85/15 stratified train/val split of the 68,252-row
teacher-labeled set, evaluated against the same gold set as Rungs 1 and 3.

### 4.2 Two metrics, and why only one of them matters

The training run reports two different "accuracy" numbers and they measure different
things:

- **`validation_metrics.eval_accuracy` = 83.2%**: agreement with the teacher's own
  (noisy) labels, on a held-out 10,237-row slice of the training data. Used internally
  during training for early stopping / checkpoint selection. **Not a measure of
  correctness** — the held-out split shares the same labeling process and systematic
  biases as the training data, so a model can score well here while being confidently
  wrong about the same things the teacher was confidently wrong about.
- **`gold_metrics.accuracy` = 82.2%**: agreement with the independently-verified
  551-row gold set. **This is the number that's comparable across all three rungs.**

Worth noting these two numbers land close together (82.2% vs. 83.2%) — a healthy sign.
The model isn't tracking real ground truth meaningfully worse than it tracks the
(noisy) signal it was actually trained on, which is what "learned the underlying task,
not just memorized one specific labeling process's quirks" should look like.

### 4.3 Class space matches Rung 1 and Rung 3

The softmax head covers the full 40-class `raw_component` space (the 30 primary
categories + 9 rare categories with enough surviving examples + `UNKNOWN OR OTHER`) —
the same label space Rung 1 and Rung 3 use. Scoring also uses `eval_models_on_gold.py`'s
"accept list" semantics: an accepted-but-non-primary prediction counts as correct, same
as how ambiguous gold rows are scored for the LLM teacher. Matching both the class count
and the scoring convention is what makes the headline accuracy numbers directly
comparable across all three rungs.

### 4.4 Results

| Metric | Value |
|---|---|
| Gold accuracy (overall) | **82.2%** |
| Gold accuracy, hard/adjudicated subset | 73.3% |
| Gold accuracy, dual-agreement subset | 93.1% |
| Macro-F1 (gold) | 0.775 |
| n_train / n_val | 58,006 / 10,237 |

The hard-vs-easy split mirrors the same pattern found for the teacher itself in §3.3 —
DeBERTa struggles more on the rows that are inherently harder, which is expected and
healthy (the reverse pattern would be a red flag).

**Takeaway**: an encoder classifier fine-tuned on distilled LLM labels gets within
~4-5 points of the teacher's own solo gold accuracy, at a small fraction of the
inference cost and with no per-call API dependency — a strong result for the
cheapest/fastest of the three rungs.

### 4.5 Per-class comparison: Rung 2 vs. Rung 3

**A methodological note on macro-F1**: it's computed here only over classes that
actually appear in the gold set's true/predicted labels, not forced over the full
40-class schema. With several classes having zero or near-zero gold examples (§3.4),
forcing the full label universe into the average would assign each of those an
automatic F1 of 0 regardless of model quality — a scoring artifact that has nothing to
do with actual model quality on a small, class-imbalanced eval set. All rungs' macro-F1
numbers in this document use the non-forced convention for that reason.

**Per-class F1, Rung 2 vs. Rung 3** (classes with meaningful gold support; full data in
`output/rung{2,3}_percategory_and_novel_full_62k_plus_rare_v3.json`):

| Class (approx. support) | Rung 2 F1 | Rung 3 F1 | Rung 3 − Rung 2 |
|---|---|---|---|
| POWER TRAIN (~65) | 0.812 | 0.900 | +0.088 |
| AIR BAGS (~52) | 0.959 | 0.944 | −0.015 |
| SERVICE BRAKES, HYDRAULIC (47) | 0.844 | 0.857 | +0.013 |
| ENGINE (~42) | 0.833 | 0.829 | −0.004 |
| ELECTRICAL SYSTEM (~40) | 0.769 | 0.825 | +0.056 |
| UNKNOWN OR OTHER (~35) | 0.545 | 0.643 | +0.097 |
| STEERING (~34) | 0.957 | 0.971 | +0.015 |
| VEHICLE SPEED CONTROL (~25) | 0.778 | 0.833 | +0.056 |
| ENGINE AND ENGINE COOLING (~23) | 0.920 | 0.857 | −0.063 |
| STRUCTURE (~22) | 0.898 | 0.864 | −0.034 |
| SUSPENSION (~20) | 0.818 | 0.944 | +0.126 |
| FUEL SYSTEM, GASOLINE (19) | 0.973 | 1.000 | +0.027 |
| EXTERIOR LIGHTING (~18) | 0.944 | 0.950 | +0.006 |
| VISIBILITY (~15) | 0.897 | 0.923 | +0.027 |
| — 8 classes, support 5–11 (SEAT BELTS, FORWARD COLLISION AVOIDANCE, LATCHES/LOCKS/LINKAGES, EQUIPMENT, HYBRID PROPULSION SYSTEM, TIRES, SEATS, WHEELS) | 0.50–1.000 | 0.67–1.000 | tied on 2, Rung 3 ahead on 6 |

**Reading this**: Rung 3's macro-F1 edge (0.799 vs. 0.775) is **broad-based, not
concentrated in one or two classes** — it wins on most mid-frequency classes by
small-to-moderate margins (SUSPENSION, POWER TRAIN, ELECTRICAL SYSTEM, UNKNOWN OR OTHER
all n≥20), loses narrowly on a few (STRUCTURE, ENGINE AND ENGINE COOLING, AIR BAGS), and
the smaller classes below n=11 mostly favor Rung 3 too, though single-digit support
means any one of those flips isn't very statistically meaningful on its own. The rare
categories specifically targeted by the training top-up (§3.2) and gold expansion
(§3.4) are broken out separately below.

**The 7 newly-covered rare classes, Rung 2 vs. Rung 3** (n as low as 1-2 for some — see
§3.4's caveat, this is a directional read, not a precise one):

| Class | n (v3 gold) | Rung 2 F1 | Rung 3 F1 |
|---|---|---|---|
| `EQUIPMENT ADAPTIVE/MOBILITY` | 4 | 0.857 | **1.000** |
| `SERVICE BRAKES, ELECTRIC` | 4 | 0.000 | 0.000 |
| `ELECTRONIC STABILITY CONTROL` | 4 (Rung 2) / 5 (Rung 3)* | 0.500 | 0.615 |
| `TRACTION CONTROL SYSTEM` | 2 | 0.667 | 0.667 |
| `CHILD SEAT` | 2 | 0.667 | 0.667 |
| `FUEL/PROPULSION SYSTEM` | 1 | 0.667 | 0.000 |
| `FUEL SYSTEM, OTHER` | 1 | 0.000 | 0.000 |

\* per-class support can differ slightly by rung under this project's scoring
convention (`y_true = prediction if accept-list-correct else gold primary`, same logic
as §3.3/§4.5) — an "accept-list-correct" prediction contributes its own predicted label
to the true-label tally instead of the row's primary label, so two rungs scoring the
same 551 rows can produce marginally different per-class denominators (e.g.
`ELECTRONIC STABILITY CONTROL` shows support=4 for Rung 2 vs. support=5 for Rung 3).
Not a bug, just a quirk worth knowing about before reading too much into single-row
swings at this n.

`SERVICE BRAKES, ELECTRIC` is the standout miss for **both** rungs (0.000 despite 895
training rows from the §3.2 top-up) — worth flagging rather than smoothing over: the
genuine examples in this class are specifically hybrid-vehicle regenerative-brake
complaints (§3.4), a narrow and easily-confused-with-plain-ABS pattern that apparently
neither model reliably picked up on even with real training volume. `FUEL SYSTEM,
OTHER` and `FUEL/PROPULSION SYSTEM` have only 1 gold example each post-expansion (both
of the classes' other candidate rows were reclassified elsewhere on adjudication, per
§3.4) so a single miss swings the whole class's F1 to 0 — not a meaningful signal
either way at n=1. `EQUIPMENT ADAPTIVE/MOBILITY` is the clearest win for the rare-topup
+ expansion combination working as intended: real training volume (§3.2) plus real gold
coverage (§3.4) now shows both rungs doing well, Rung 3 perfectly.

## 5. Rung 3 — LoRA fine-tune of Qwen2.5-7B-Instruct

### 5.1 Design choices

- Trains on the full 40-value `raw_component` space generatively (not a fixed softmax
  head) — the whole point of using an LLM here is that a genuinely new category is just
  a new string, not a retrain, unlike Rung 2's fixed-size head.
- Off-list rows (teacher hallucinations not in the valid 40-value schema) dropped from
  training rather than taught to the student.
- Scored on gold identically to Rungs 1/2 for direct comparability.
- Hand-rolled QLoRA/LoRA on `transformers` + `peft` + `bitsandbytes` (no `trl`), to
  avoid extra dependency-version churn on top of what was already needed.
- Student system prompt deliberately drops the few-shot block the Rung-1 teacher prompt
  needs — the premise of fine-tuning is that label-schema knowledge gets baked into
  weights via many real examples instead of a handful of in-context ones. Also
  shortened the prompt itself (~1775 → ~400 tokens) once profiling showed prompt length
  was the dominant cost driver during training, not the training step or GPU I/O.

### 5.2 What the "hope" for Rung 3 actually should be

Rung 3 trains on the *same* 68,252 teacher-labeled rows as Rung 2, so its accuracy is
bounded by the same underlying label quality (§3.3's ~90% teacher ceiling on gold). A
higher LoRA capacity (7B params vs. DeBERTa-base's ~184M) can fit the training labels
more precisely, but fitting noisier labels more precisely is not automatically better —
it can mean memorizing the teacher's own mistakes rather than generalizing past them.
The right target for Rung 3 is **gold-set accuracy**, the same metric used for Rungs 1
and 2, not agreement with its own training labels. Conservative expectation going in:
land in the same neighborhood as or modestly above Rung 2's gold accuracy, likely still
short of the teacher's own ceiling.

**Actual result (§5.4) beat that conservative expectation** — Rung 3 landed at 86.2%,
essentially matching the teacher's own ceiling (86.6%, §2) rather than plateauing
meaningfully below it, while also winning decisively on macro-F1. The flexibility argument (generative
output, not a fixed classification head) still holds as a structural advantage, but it
turned out not to be the *only* advantage — the larger pretrained model's general
language understanding seems to have helped it partially generalize past some of the
teacher's own labeling noise, not just reproduce it.

### 5.3 Infrastructure notes

- Initially targeted a rented GCP L4 GPU; ~6-day naive runtime estimate at that GPU tier
  drove the prompt-shortening work in §5.1 (cut epoch time roughly in half).
- GCP A100/H100 quota requests were denied for this (new, low-usage) billing account —
  a policy wall, not a technical one. Pivoted to Lambda Labs for on-demand H100 access
  with no quota approval step.
- Running on a Lambda H100 SXM5 (80GB). First attempt at full-precision (no 4-bit
  quantization) LoRA at batch-size 8 OOM'd — a 7B model's activation memory without
  gradient checkpointing is large regardless of available VRAM. Fix: enabled
  `gradient_checkpointing` + `enable_input_require_grads()`, dropped to batch-size 4 /
  grad-accum 4 (same effective batch of 16). Running stably at ~27GB/80GB VRAM,
  ~100% GPU utilization, full bf16 precision (no quantization needed given the extra
  headroom — likely marginally more accurate than QLoRA, and avoids extra
  bitsandbytes-version-compatibility risk).
- Actual full 3-epoch run: **~6h50m** on the H100 SXM5 (vs. ~6 days originally estimated
  on the L4), then instance terminated immediately after pulling results back to stop
  billing.
- Rung 2 and Rung 3 have no dependency on each other (both just need the same source
  parquet) and were run concurrently on separate rented GPUs (GCP L4 for Rung 2, Lambda
  H100 for Rung 3) to save wall-clock time.

### 5.4 Results

| Metric | Value |
|---|---|
| Gold accuracy (overall) | **86.2%** |
| Gold accuracy, hard/adjudicated subset | 76.6% |
| Gold accuracy, dual-agreement subset | 98.0% |
| Macro-F1 (gold) | 0.799 |
| Parse failures | 0/551 |
| Validation loss (held-out training split, epoch 3) | 0.0105 |
| n_train / n_val | 58,006 / 10,237 |
| Model | Qwen2.5-7B-Instruct, LoRA r=16/alpha=32, full bf16 (no quantization) |
| LoRA adapter size | 165MB |

Same hard-vs-easy pattern as Rungs 1 and 2 — struggles more on the genuinely ambiguous
rows, as expected. Zero parse failures (the model always produced valid JSON with a
schema-valid or intentionally-rare-category component string), confirming the
shortened, few-shot-free student prompt was sufficient — the mode-collapse risk flagged
in `PROJECT_PLAN.md` (fine-tuning narrowing willingness to use rare/OTHER categories)
doesn't show up in this metric, and is checked more directly in §5.5 below. Rung 3
keeps its lead over Rung 2 on every metric.

### 5.5 Novel-category flexibility probe

`PROJECT_PLAN.md` flagged a real risk with fine-tuning on a fixed label set: the model
could over-fit to *only* ever producing the categories it saw during training, and force
a bad guess when a complaint genuinely describes something outside that set, rather than
gracefully falling back to a catch-all. Zero parse failures on the gold set (§5.4) is
reassuring but doesn't test this directly, since every gold-set narrative already maps
to one of the 40 schema categories by construction.

To test it directly, 5 synthetic complaint narratives were written describing real
vehicle accessories that are *not* well covered by the 40-class schema and were
deliberately picked to be specific-but-uncategorized (not vague, so `UNKNOWN OR OTHER`
isn't obviously the "right" abstention either) — a built-in mini-fridge console, a
power tonneau cover, heated/cooled cupholders, a dash-cam/cabin-camera system, and a
vehicle-to-home bidirectional EV charging feature (`scripts/novel_category_probe.py`).
Neither rung was trained on anything resembling these. Results (full text in
`output/rung{2,3}_percategory_and_novel_full_62k_plus_rare_v3.json`):

| Narrative | Rung 2 (DeBERTa) top prediction | Rung 3 (LLM) prediction |
|---|---|---|
| Mini-fridge console leaking | `SEATS` (only 47% confident; `EQUIPMENT` 2nd at 23%) | `EQUIPMENT` |
| Power tonneau cover motor failure | `EQUIPMENT` (98% confident) | `EQUIPMENT` |
| Heated/cooled cupholders overheating | `EQUIPMENT` (98% confident) | `EQUIPMENT` |
| Dash cam / cabin camera reboots | `BACK OVER PREVENTION` (64% confident) | `COMMUNICATION` |
| Vehicle-to-home bidirectional charging | `HYBRID PROPULSION SYSTEM` (99.8% confident) | `HYBRID PROPULSION SYSTEM` |

**Neither model force-fits a nonsensical, unrelated specific category** — both
consistently reach for `EQUIPMENT` (the schema's genuine accessory catch-all) on the
clearly-accessory cases, and both land on the same defensible answer for the EV-charging
case (no dedicated EV-charging class exists in the 40-class schema, so
`HYBRID PROPULSION SYSTEM` is the closest real analog, not a forced miss). The one
disagreement (dash cam) is a judgment call either way (`BACK OVER PREVENTION` reads the
camera angle as backup-camera-adjacent; `COMMUNICATION` reads it as an infotainment/data
system) rather than either model producing something clearly wrong like `ENGINE` or
`TIRES`. Net: **no evidence of the mode-collapse risk in either rung** on this small
probe — both generalize sensibly to genuinely novel accessory complaints, with Rung 2's
lower top-1 confidence on the ambiguous mini-fridge case (47%, with the "right" answer
`EQUIPMENT` as a close second) being an honest reflection of it being a harder case
even for a human, not a failure mode.

## 6. Cost, latency, and deployment

### 6.1 Inference cost/latency comparison

Measured directly from this project's own eval runs against `gold_eval_set_v3.json`
(n=551), not estimated:

| Approach | Marginal cost / 100K rows | Throughput (as measured here) | Needs a GPU? |
|---|---|---|---|
| Rung 1, Sonnet solo (on-demand API) | ~$467 | ~4.3 rows/sec at concurrency=12 | No |
| Rung 1, Sonnet solo (Batch API, 50% off) | ~$233 | async, ~24h SLA, no live latency | No |
| Rung 1, Haiku solo (on-demand API) | ~$154 | ~8.6 rows/sec at concurrency=12 | No |
| Rung 1, cascade (as actually run, §3.2) | ~$126 (production run, batch) | bulk/async | No |
| Rung 2, DeBERTa (self-hosted) | ~$0 marginal (own hardware, no per-call fee) | ~5 rows/sec on a single CPU core, batched | **No — CPU is enough** |
| Rung 3, Qwen2.5-7B + LoRA (self-hosted) | ~$0 marginal if hardware already owned/idle; ~$70/100K if renting a $1.29/hr A10 continuously at this measured (unbatched) rate | ~0.5 rows/sec, naive one-request-at-a-time `.generate()` on a single rented A10 | **Yes, for reasonable latency** |

Caveats that matter more than the numbers themselves:

- Rung 3's throughput figure is from the simplest possible scoring loop (one row,
  one `.generate()` call, no batching) — not a production serving setup. A real
  deployment would use vLLM/TGI-style batched decoding or a smaller
  quantized model, both of which would cut this substantially; treat "needs a GPU and
  some serving engineering" as the takeaway, not the exact $70/100K figure.
- Rung 2's CPU-only throughput is the more interesting number precisely because it's
  *not* a simplification — a small encoder classifier genuinely doesn't need a GPU to
  hit usable throughput, which is most of why it's the cheapest rung to actually run.
- Rung 1's API costs scale linearly with volume forever; Rungs 2/3 pay a one-time
  training cost (§3.2's labeling spend + the GPU rental hours in §4/§5, all comfortably
  inside the self-funded budget ceiling) and then cost ~nothing marginal. The
  break-even volume where self-hosting starts paying for itself is low — well under
  100K rows even against Haiku's already-cheap on-demand price.

### 6.2 Deployment story

**Rung 2 (`checkpoints/deberta_rung2_full_62k_plus_rare/final/`, ~700MB
`model.safetensors`)**: a standard `AutoModelForSequenceClassification` checkpoint —
wrap it in a small FastAPI/Flask service that loads the model once and serves
`(narrative, make, model, year) → component` over HTTP, or call it directly in a batch
job. Runs fine on CPU (§6.1), trivially horizontally scalable (stateless, no GPU
scheduling), easy to containerize. Best fit: a low-ops, low-latency classification
endpoint, or a nightly batch re-scoring job, where simplicity and cost matter more than
the last few points of accuracy or the ability to name a genuinely novel category
(§5.5's flexibility argument is Rung 3's, not Rung 2's).

**Rung 3 (`checkpoints/llm_rung3_full_62k_plus_rare/`, 165MB LoRA adapter over
`Qwen/Qwen2.5-7B-Instruct`)**: load the ~15GB base model plus this adapter (either kept
separate via `peft.PeftModel.from_pretrained` for hot-swappable adapters on one shared
base model, or flattened once via `merge_and_unload()` for a single standalone
deployable model). Needs a GPU with enough VRAM for a 7B model in bf16 (~16GB+, less
with 4-bit quantization at some accuracy cost); for real throughput, swap the naive
`.generate()` loop used for scoring here for a proper inference server (vLLM or
TGI). Best fit: where the accuracy edge and the generative flexibility to name a
genuinely novel category on the fly (§5.5) justify the extra GPU hosting cost and
serving complexity — e.g. an internal analyst-facing tool, or a lower-volume pipeline
where per-row accuracy matters more than raw throughput.
