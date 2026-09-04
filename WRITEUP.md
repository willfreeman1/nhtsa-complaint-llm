# NHTSA Complaint Classification — Project Writeup

This document is the full record of what was actually done, what it found, and the
reasoning behind each decision, across the three-rung comparison laid out in
`PROJECT_PLAN.md`. It's written as the raw material a final report would draw from
rather than a polished summary — including inconvenient/nuanced findings (e.g. §3.3's
skeptical read on the teacher's own error rate, §3.4's gold-set expansion turning up a
lot of NHTSA mislabeling) that a cleaner writeup might otherwise smooth over. See the
top-level `README.md` for the short version and repo map.

## 1. Goal

Given a raw NHTSA vehicle complaint narrative (free text) plus structured metadata
(make/model/year), predict the affected vehicle **component** (30 primary categories +
10 rare categories + `UNKNOWN OR OTHER`, deduping to 40 distinct valid strings — see
`teacher_prompt.ALL_VALID_COMPONENTS`), and secondary flags (`CRASH`, `FIRE`,
`INJURED`, `DEATHS`). Compare three approaches of increasing
training-complexity-but-decreasing-inference-cost, to find where the accuracy/cost
tradeoff actually lands for this task, and produce a deployable fine-tuned model as the
portfolio deliverable.

Feasibility (green-light verdict, baselines, label-schema gotchas) is documented in
`output/FINAL_REPORT.md` and `output/NHTSA_CODING_GOTCHAS.md` and isn't repeated here.

## 2. The three rungs

| Rung | Approach | Status |
|---|---|---|
| 1 | Zero/few-shot frontier LLM teacher (Haiku→Sonnet confidence cascade) | **Done** — used to label all training data |
| 2 | Fine-tuned encoder classifier (DeBERTa-v3-base, 40-class softmax head) | **Done** |
| 3 | Fine-tuned open-weight LLM (Qwen2.5-7B-Instruct, LoRA, generative) | **Done** |

## Headline result

**Updated for gold_eval_set_v3.json (n=551) — see §3.4.** The original n=498 numbers
(gold_eval_set_v2.json) are kept in each rung's own results section for the record, but
this table and all "Results" subsections below now reflect the expanded set, which adds
53 deliberately-sampled rows covering 7 of the 9 `raw_component` classes that had zero
gold coverage before. All four models' accuracy dropped by 3-5 points versus the v2
numbers — expected and reassuring, not a red flag: the new rows are disproportionately
hard/ambiguous by construction (see §3.4), and the drop is consistent across every rung,
so relative ranking is unaffected.

| | Rung 1 (Sonnet solo) | Rung 1 (Haiku solo) | Rung 1 (cascade, as actually run) | Rung 2 (DeBERTa) | Rung 3 (Qwen2.5-7B LoRA) |
|---|---|---|---|---|---|
| Gold accuracy (overall) | 86.6% | 85.1% | 89.8%* | 82.2% | **86.2%** |
| Gold accuracy, hard subset | 78.2% | 74.9% | — | 73.3% | **76.6%** |
| Gold accuracy, dual-agreement | 96.8% | 97.6% | — | 93.1% | **98.0%** |
| Macro-F1 | 0.764 | 0.752 | — | 0.775 | **0.799** |
| Per-inference cost | ~$467/100K rows (on-demand) | ~$154/100K rows (on-demand) | ~$126/100K rows (batch, measured) | ~$0 marginal (self-hosted) | ~$0 marginal (self-hosted) |

\* backtest estimate from `confidence_cascade_analysis.json` against the original n=498
set, not a direct gold-set run of the exact production cascade, and not yet re-backtested
against v3.

Sonnet solo remains the top individual model by a modest ~0.4pt margin over the
fine-tuned LLM (Rung 3), with Rung 3 still comfortably ahead of Rung 2 on every metric —
the same ranking and roughly the same gaps as the original n=498 result, which is the
important takeaway: **adding real coverage for the hardest, rarest classes didn't change
any of the project's conclusions, it just made the gold set a more honest test.** See §3.4
for what the expansion specifically found, and §7 for cost/latency and deployment notes.

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

### 3.4 Gold set v3: closing the rare-category coverage gap

§4.5 below flagged that 9 of the 40 `raw_component` classes had **zero** examples in
the original 498-row gold set — largely the same rare categories the training-data
top-up in §3.2 targeted — which meant the top-up's real-world payoff couldn't be
checked against gold at all for those classes. `scripts/build_rare_gold_expansion.py`
pulls real complaints for those 9 classes directly from the raw corpus (by literal
`COMPDESC_TOP` match, excluding anything already used in gold or training data), and
`scripts/build_gold_eval_set_v3.py` merges the adjudicated results into
`gold_eval_set_v3.json`.

Two of the 9 classes turned out to be **uncoverable**: `FIRERELATED` (169 total
complaints in the entire 2.17M-row corpus) and `TRAILER HITCHES` (924 total) had
**zero remaining rows anywhere in the corpus** once every row already used in training
data was excluded — the rare-category top-up in §3.2 had already consumed every real
example that existed. Rather than compromise the no-train/eval-leakage rule to force
coverage, these two are documented as unverifiable with the current corpus, not
papered over.

For the other 7 classes, 7 candidates each (49 rows) were pulled and read individually
against `teacher_prompt.py`'s class definitions — the same "NHTSA's raw field is a
hypothesis, not the truth" adjudication standard used for every other hard row in this
gold set, not a rubber stamp of the sampling target. A follow-up targeted keyword
search (`WHEELCHAIR|HAND CONTROL|LIFT|MOBILITY|DISAB|SCOOTER|RAMP`) added 4 more rows
for `EQUIPMENT ADAPTIVE/MOBILITY` specifically, since random sampling had found zero
genuine matches for it in the first pass.

**Only 16 of these 53 rows (30%) actually confirmed the class they were sampled for:**

| Sampled category | Confirmed / sampled |
|---|---|
| `CHILD SEAT` | 2/7 |
| `ELECTRONIC STABILITY CONTROL` | 2/7 |
| `EQUIPMENT ADAPTIVE/MOBILITY` | 4/11 |
| `FUEL SYSTEM, OTHER` | 1/7 |
| `FUEL/PROPULSION SYSTEM` | 1/7 |
| `SERVICE BRAKES, ELECTRIC` | 4/7 |
| `TRACTION CONTROL SYSTEM` | 2/7 |

This is itself a real finding, not noise: several of NHTSA's raw `COMPDESC_TOP` rare
buckets are heavily contaminated in practice. `CHILD SEAT` is dominated by aftermarket
car-seat-product complaints that don't belong in a vehicle-component schema at all
(the class definition explicitly requires a built-in/integrated feature). `EQUIPMENT
ADAPTIVE/MOBILITY`'s random sample was 0/7 genuine — narratives about airbags, horns,
ABS modules, and unintended acceleration all got filed there for reasons unrelated to
adaptive/mobility equipment, and genuine examples (wheelchair lifts, hand controls)
only turned up via targeted keyword search. `SERVICE BRAKES, ELECTRIC` was the
pleasant surprise — 4/7 genuinely confirmed once it became clear this bucket is
reserved for hybrid vehicles' electronically-blended regenerative brake systems (a real
Toyota Prius/Camry Hybrid recall pattern), not noise at all once you know what to look
for. The 37 non-confirming rows weren't wasted — they were adjudicated to whatever the
narrative actually supports and added to the gold set regardless, same as any other
adjudicated row (`n` went from 498 to 551).

**Net effect on gold coverage**: `FIRERELATED` and `TRAILER HITCHES` remain at zero
support (documented as uncoverable above); the other 7 classes now have 1-4 gold
examples each — thin, but a real signal where there was none before. Per-class results
for these specific classes are in §5.4's model files
(`output/rung{2,3}_percategory_and_novel_full_62k_plus_rare_v3.json`); treat any
single one of these classes' accuracy as directional at this sample size, not precise.

## 4. Rung 2 — DeBERTa-v3-base fine-tune

### 4.1 Setup

Standard classification fine-tune: DeBERTa-v3-base, 40-class softmax head over
`raw_component` (not the bucketed 31-class bucket — see §4.3 for why this mattered),
class-weighted loss for imbalance, 85/15 stratified train/val split of the 68,252-row
teacher-labeled set, evaluated against the same 498-row gold set as Rung 1.

### 4.2 Two metrics, and why only one of them matters

The training run reports two different "accuracy" numbers and they measure different
things:

- **`validation_metrics.eval_accuracy` = 83.2%**: agreement with the teacher's own
  (noisy) labels, on a held-out 10,237-row slice of the training data. Used internally
  during training for early stopping / checkpoint selection. **Not a measure of
  correctness** — the held-out split shares the same labeling process and systematic
  biases as the training data, so a model can score well here while being confidently
  wrong about the same things the teacher was confidently wrong about.
- **`gold_metrics.accuracy` = 85.3%**: agreement with the independently-verified
  498-row gold set. **This is the number that's comparable across all three rungs.**

Worth noting DeBERTa's gold accuracy (85.3%) is *higher* than its agreement with its
own noisy training signal (83.2%) — it's not simply memorizing the teacher's labels,
including the teacher's mistakes, more precisely than that; it's landing closer to true
answers than to the label it was trained against on net.

### 4.3 A correctness fix worth documenting

Initial Rung 2 training used the bucketed 31-class `component_label` head, while Rung 1
and (planned) Rung 3 operate over the full `raw_component` space — an unfair comparison
(fewer classes to choose among structurally inflates accuracy). Fixed by retraining with
a head over `ALL_VALID_COMPONENTS` (40 classes — the 30 primary + 9 rare categories with
enough surviving examples + `UNKNOWN OR OTHER`) and aligning the scoring logic with
`eval_models_on_gold.py`'s "accept list" semantics (an accepted-but-non-primary
prediction still counts correct, matching how ambiguous gold rows are scored for the LLM
teacher too).

### 4.4 Results

Re-scored on `gold_eval_set_v3.json` (n=551, see §3.4) with
`scripts/eval_checkpoint_deberta.py` — no retraining, same checkpoint. Original
n=498 figures kept alongside for the record:

| Metric | v3 (n=551) | v2 (n=498, original) |
|---|---|---|
| Gold accuracy (overall) | **82.2%** | 85.3% |
| Gold accuracy, hard/adjudicated subset | 73.3% | 77.1% |
| Gold accuracy, dual-agreement subset | 93.1% | 94.2% |
| Macro-F1 (gold) | 0.775 | 0.778 (corrected — see §4.5) |
| n_train / n_val | 58,006 / 10,237 | (same) |

The hard-vs-easy split mirrors the same pattern found for the teacher itself in §3.3 —
DeBERTa struggles more on the rows that are inherently harder, which is expected and
healthy (the reverse pattern would be a red flag). The v2→v3 drop (85.3%→82.2%) is
almost entirely the new 53 rows being disproportionately hard by construction (§3.4) —
macro-F1 barely moved (0.778→0.775) because it's less sensitive to how many *rows* are
hard, and more to how many *classes* now have any support at all, which is a wash here
(a few previously-zero-support rare classes gained thin support, offset by those same
classes' predictions often being wrong at n=1-4).

**Takeaway**: an encoder classifier fine-tuned on distilled LLM labels gets within
~4-5 points of the teacher's own solo gold accuracy, at a small fraction of the
inference cost and with no per-call API dependency — a strong result for the
cheapest/fastest of the three rungs.

### 4.5 A second correctness fix: macro-F1 was scored inconsistently across rungs

The originally reported Rung 2 macro-F1 was **0.642**, which read as a large gap below
overall accuracy (85.3%) and well below Rung 1 (0.838) and Rung 3 (0.850). Investigating
that gap while doing the per-class breakdown below surfaced a real bug, not a real
model-quality gap: `train_deberta.py`'s `evaluate_on_gold()` called
`sklearn.metrics.f1_score(..., average="macro", labels=LABELS)`, explicitly passing the
full 40-class schema as the label universe. `eval_models_on_gold.py` (Rung 1) and
`train_llm_lora.py` (Rung 3) both call the same function **without** `labels=`, which
makes sklearn default to only the classes that actually appear in `y_true`/`y_pred`.

That difference matters a lot here because **9 of the 40 schema classes have zero
examples in the 498-row gold set** (`CHILD SEAT`, `ELECTRONIC STABILITY CONTROL`,
`EQUIPMENT ADAPTIVE/MOBILITY`, `FIRERELATED`, `FUEL SYSTEM, OTHER`,
`FUEL/PROPULSION SYSTEM`, `SERVICE BRAKES, ELECTRIC`, `TRACTION CONTROL SYSTEM`,
`TRAILER HITCHES` — several of these are exactly the rare categories the top-up in §3.2
targeted, but "more training rows" doesn't help a class the *gold set itself* never
tests). Forcing those 9 into the macro average gives each a guaranteed F1 of 0
regardless of model quality, since there's no way to be scored correct on a class with
no true instances. That alone drags a ~40-class macro average down by roughly
9/40 ≈ 22%, independent of anything DeBERTa actually did right or wrong.

Fixed by removing `labels=LABELS` from `train_deberta.py` (see the code comment left in
place there) and rescoring the already-trained checkpoint with
`scripts/eval_checkpoint_deberta.py` (loads the saved model, no retraining needed) —
**corrected macro-F1: 0.778**, up from 0.642, with the *identical* underlying
predictions (gold accuracy is unchanged at 85.3%, confirming this was a pure scoring
fix, not a model change). `output/rung2_deberta_eval_full_62k_plus_rare.json` has been
updated in place with a note; the original 0.642 should not be used for cross-rung
comparison.

**Per-class macro-F1, Rung 2 vs. Rung 3** (only the 33 classes with actual gold support,
scoring convention now matched across both rungs; full data in
`output/rung{2,3}_percategory_and_novel_full_62k_plus_rare.json`):

| Class (support) | Rung 2 F1 | Rung 3 F1 | Rung 3 − Rung 2 |
|---|---|---|---|
| SUSPENSION (20) | 0.810 | 0.941 | +0.132 |
| POWER TRAIN (64) | 0.833 | 0.924 | +0.091 |
| UNKNOWN OR OTHER (27) | 0.609 | 0.696 | +0.087 |
| ELECTRICAL SYSTEM (38) | 0.811 | 0.865 | +0.054 |
| VEHICLE SPEED CONTROL (24) | 0.826 | 0.857 | +0.031 |
| VISIBILITY (16) | 0.897 | 0.923 | +0.027 |
| SERVICE BRAKES, HYDRAULIC (43) | 0.889 | 0.914 | +0.025 |
| STEERING (33) | 0.955 | 0.971 | +0.015 |
| ENGINE (42) | 0.850 | 0.857 | +0.007 |
| EXTERIOR LIGHTING (19) | 0.971 | 0.974 | +0.003 |
| — 12 classes tied at 1.000 or within ±0.00 (BACK OVER PREVENTION, COMMUNICATION, FORWARD COLLISION AVOIDANCE, FUEL SYSTEM types, HYBRID PROPULSION, INTERIOR LIGHTING, SERVICE BRAKES, WHEELS, etc.) | — | — | ~0 |
| AIR BAGS (53) | 0.958 | 0.943 | −0.015 |
| STRUCTURE (21) | 0.894 | 0.864 | −0.030 |
| ENGINE AND ENGINE COOLING (24) | 0.939 | 0.878 | −0.061 |
| PARKING BRAKE, LANE DEPARTURE, ELECTRONIC STABILITY CONTROL, EQUIPMENT, SEATS, VISIBILITY/WIPER, LATCHES/LOCKS/LINKAGES, SEAT BELTS (support 1–7 each) | mixed, mostly lower | mostly 1.000 | +0.05 to +0.40 each |

**Reading this**: Rung 3's macro-F1 edge (0.850 vs. 0.778, a real but now much smaller
gap than the original 0.642 comparison suggested) is **broad-based, not concentrated in
one or two classes** — it wins on most mid-frequency classes by small-to-moderate
margins (SUSPENSION, POWER TRAIN, ELECTRICAL SYSTEM, UNKNOWN OR OTHER all n≥20), loses
narrowly on a few (STRUCTURE, ENGINE AND ENGINE COOLING, AIR BAGS), and the biggest
percentage swings are on classes with only 1-7 gold examples where a single flipped
prediction moves F1 by a large amount and isn't very statistically meaningful on its
own. The top-up in §3.2 did its job of getting the 9 targeted rare categories into
*training* in useful volume; whether it "worked" couldn't be fully checked against gold
at the time this table was built, because the gold set then still had zero coverage for
those same 9 classes — resolved in §3.4 below.

**A third, related bug found while building §3.4's expansion**: the standalone
per-class-report scripts (`eval_checkpoint_deberta.py`, `eval_checkpoint_llm.py` —
written after the table above, to generate exactly this per-class breakdown) were
themselves running the gold set's true label through `bucket_component()` before
scoring, a leftover from copy-pasting a pattern meant for the *old* 31-class bucketed
scheme. That function collapses every rare category (and any off-schema answer) down
to a generic `OTHER` bucket — invisible with the original gold set, since none of its
rows had a rare-category true label to collapse, but it would have silently zeroed out
every one of §3.4's new rare-category rows' contribution to the per-class report (their
true labels would show up as `OTHER` instead of their real class). Fixed by using the
gold row's `primary` field directly instead of running it through `bucket_component()`
first, in both scripts, before generating any of the v3 per-class numbers below.

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
land in the same neighborhood as or modestly above Rung 2's 85.3%, likely still short
of the teacher's own ~90% ceiling.

**Actual result (§5.4) beat that conservative expectation** — Rung 3 landed at 89.4%,
essentially matching the teacher's own ceiling rather than plateauing meaningfully below
it, while also winning decisively on macro-F1. The flexibility argument (generative
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

Re-scored on `gold_eval_set_v3.json` (n=551, see §3.4) with
`scripts/eval_checkpoint_llm.py` — no retraining, same LoRA adapter. Original n=498
figures kept alongside for the record:

| Metric | v3 (n=551) | v2 (n=498, original) |
|---|---|---|
| Gold accuracy (overall) | **86.2%** | 89.4% |
| Gold accuracy, hard/adjudicated subset | 76.6% | 80.6% |
| Gold accuracy, dual-agreement subset | 98.0% | 98.8% |
| Macro-F1 (gold) | 0.799 | 0.850 |
| Parse failures | 0/551 | 0/498 |
| Validation loss (held-out training split, epoch 3) | 0.0105 | (same) |
| n_train / n_val | 58,006 / 10,237 | (same) |
| Model | Qwen2.5-7B-Instruct, LoRA r=16/alpha=32, full bf16 (no quantization) | (same) |
| LoRA adapter size | 165MB | (same) |

Same hard-vs-easy pattern as Rungs 1 and 2 — struggles more on the genuinely ambiguous
rows, as expected. Zero parse failures on either gold set (the model always produced
valid JSON with a schema-valid or intentionally-rare-category component string),
confirming the shortened, few-shot-free student prompt was sufficient — the mode-collapse
risk flagged in `PROJECT_PLAN.md` (fine-tuning narrowing willingness to use rare/OTHER
categories) doesn't show up in this metric, and is checked more directly in §5.5 below.
The v2→v3 drop (89.4%→86.2%, macro-F1 0.850→0.799) tracks the same-sized drop seen for
every other rung after adding the harder v3 rows (§3.4) — Rung 3 keeps its lead over
Rung 2 on every metric at both gold-set versions.

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
`output/rung{2,3}_percategory_and_novel_full_62k_plus_rare.json`):

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

## 6. Open items

- ~~The 498-row gold set has zero examples of 9 of the 40 schema classes~~ — **closed,
  see §3.4.** 7 of 9 now have thin-but-real gold coverage; `FIRERELATED` and
  `TRAILER HITCHES` are documented as uncoverable (zero remaining real complaints in
  the corpus after training-data exclusion).
- ~~Inference cost/latency comparison~~ — **closed, see §7.1.**
- ~~Deployment story for the two trained artifacts~~ — **closed, see §7.2.**

Nothing else outstanding — remaining work is repo hygiene (README, license, `.gitignore`
cleanup) to get this posted properly, not further modeling work.

## 7. Cost, latency, and deployment

### 7.1 Inference cost/latency comparison

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

### 7.2 Deployment story

**Rung 2 (`checkpoints/deberta_rung2_full_62k_plus_rare/final/`, ~700MB
`model.safetensors`)**: a standard `AutoModelForSequenceClassification` checkpoint —
wrap it in a small FastAPI/Flask service that loads the model once and serves
`(narrative, make, model, year) → component` over HTTP, or call it directly in a batch
job. Runs fine on CPU (§7.1), trivially horizontally scalable (stateless, no GPU
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
