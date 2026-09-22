# NHTSA Complaint Classification — Project Writeup

**The National Highway Traffic Safety Administration's (NHTSA's) public complaint
database tags every complaint with a vehicle component, but that tagging is noisy.
This project fine-tunes small models that read a raw complaint narrative and assign
the correct component instead.** A 7-billion-parameter open model (Qwen2.5-7B),
trained once, nearly matches an expensive general-purpose model you pay per
question (a frontier model) asked fresh every time, at a small fraction of the
ongoing cost once trained. What we actually run as a service is a
smaller text classifier (DeBERTa-v3-base): on complaints filed today the two are
close enough that the gap sits inside ordinary sampling noise, and the smaller one
runs on a regular computer.

Sections 1–6 are the original research: how the training labels were built, how the
two fine-tunes compare on a 551-complaint hand-checked set mixed across 1995–2026,
and what that ~87% ceiling actually means. Sections 7–10 cover the later question:
does this hold on complaints filed right now, what do we serve, and how would we
know to retrain?


## 1. Goal and approach

Given a raw NHTSA vehicle complaint narrative (free text) plus structured metadata
(make/model/year), predict the affected vehicle **component** (40 distinct categories —
see `teacher_prompt.ALL_VALID_COMPONENTS`) and secondary flags (`CRASH`, `FIRE`,
`INJURED`, `DEATHS`).

### 1.1 Why 40 categories

NHTSA's own raw category field (`COMPDESC`) has 55 distinct top-level values, but
those don't represent 55 equally-real, equally-common failure types. Counting actual
complaints per category shows a sharp cliff: the top ~35 categories each have
thousands of examples, then volume collapses fast — several of the smallest "categories"
turn out to be data artifacts rather than real distinctions, e.g. `ELECTRONIC STABILITY
CONTROL` vs. `ELECTRONIC STABILITY CONTROL (ESC)` (335 vs. 21,286 rows for the literal
same concept, clearly a legacy-naming duplicate) or `COMMUNICATION` vs.
`COMMUNICATIONS` (a typo pair). A few others are free-text child-seat entries that
aren't vehicle-component categories at all (e.g. *"I suspect the car seat is
counterfeit"*).

After merging those duplicate/typo pairs and dropping the non-component noise, the
schema settled on **30 primary categories** (covering the overwhelming majority of
real complaint volume, including a general `UNKNOWN OR OTHER` catch-all for narratives
that genuinely don't name a specific system) plus **10 additional categories** that
are real, distinct vehicle systems — just rare in volume rather than artifacts (e.g.
`TRAILER HITCHES`, `HYBRID PROPULSION SYSTEM`, `FIRERELATED`) — for **40 total category
strings**. This fixed 40-category schema (`scripts/teacher_prompt.py`) is the label
space used consistently everywhere in this project: the frontier LLM's prompt, the
encoder classifier's output head, and the fine-tuned LLM's generation target.

There's no existing large set of reliably-labeled complaints to fine-tune on — NHTSA's
own raw field is exactly the noisy thing this project is trying to improve on. So the
approach builds its own training labels first (§2), then compares two ways of turning
those labels into a small, deployable model — a fine-tuned encoder classifier (§3) and
a fine-tuned open-weight LLM (§4) — against each other and against the frontier LLM
that generated their training labels in the first place (§5).

## 2. Building the training labels: a frontier LLM as teacher

Labeling ~68,000 complaints with a frontier model directly would cost real money at
this project's self-funded budget. Testing showed a cheaper model's own self-reported
confidence is a reliable signal for when it's about to be wrong: Claude Haiku's
confident answers were about 95% reliable, but accuracy on its medium/low-confidence
answers dropped to barely better than a coin-flip among ~40 categories. That gap is
exploitable — run the cheap model on everything, and escalate only the ~20% of rows it
wasn't confident on to a stronger model (Claude Sonnet). Backtesting confirmed this
**confidence cascade** captured about 90% accuracy on the gold set, close to what the
strong model gets solo, at roughly a quarter of the cost.

That cascade was run over ~68,000 real complaints (including extra complaints sampled
specifically for the rarest categories, which were otherwise badly under-represented)
to produce the training labels used for both fine-tunes below.

### 2.1 The teacher's ~13% "error" rate isn't quite what it looks like

Claude Sonnet's solo accuracy on the hand-checked gold set is 86.6% — a real gap, but
digging into *which* rows it actually misses shows the headline number is misleading if
read as "confidently, obviously wrong 13% of the time." The gold set has two tiers:
**dual-agreement** rows, where the teacher and NHTSA's own field independently landed
on the same label (strong, clean ground truth), and **adjudicated/hard** rows, where
they disagreed and a person read the narrative to break the tie (genuinely harder,
ambiguous cases).

| Subset | Sonnet accuracy |
|---|---|
| Dual-agreement ("easy") | 96.8% |
| Adjudicated ("hard") | 78.2% |

The vast majority of Sonnet's errors are concentrated in the hard subset, and roughly
a third of those are cases where the "correct" answer is itself `UNKNOWN OR OTHER` — a
human decided the narrative genuinely didn't contain enough detail to name a specific
system, but the model guessed a plausible-sounding category instead of abstaining.
**Bottom line**: the model's real "confidently, obviously wrong" rate is closer to
1-3%, not 13% — but it would also be wrong to write off the whole gap as label noise;
a real share of the misses are genuine miscategorizations on harder-than-average
narratives. Expect the fine-tuned models below to show the same pattern (worse on hard
rows than easy ones), and expect a realistic ceiling for any of these approaches to sit
somewhat south of 100%, since a real fraction of complaints are inherently ambiguous
even to a careful reader.

### 2.2 A gotcha worth knowing: NHTSA's rare-category labels are often wrong

Before training, the gold test set had zero real examples for 9 of the rarest
categories in the schema, so we pulled and manually read real complaints for each one
to check whether NHTSA's own raw category field can even be trusted there. **Only 30%
of the complaints NHTSA had tagged into a given rare category actually belonged in
it.** Several of these buckets are heavily contaminated in practice — e.g. most complaints filed under
`EQUIPMENT ADAPTIVE/MOBILITY` turned out to be misfiled airbag, horn, or ABS
complaints unrelated to adaptive/mobility equipment. One pleasant exception:
`SERVICE BRAKES, ELECTRIC` turned out to be legitimately reserved for hybrid vehicles'
regenerative-braking systems — once we knew what to look for, most of its sampled
complaints checked out.

Two categories, `FIRERELATED` and `TRAILER HITCHES`, turned out to have **zero real
complaints left anywhere in the entire 2.17M-row corpus** once complaints already used
elsewhere in this project were excluded — they're rare enough that the real-world
supply is already spoken for. Their accuracy can't be independently verified at all;
that's a hard limit of the available data, not something either fine-tuned model can
be faulted for.

## 3. Fine-tuning a DeBERTa encoder classifier

**Setup**: DeBERTa-v3-base with a 40-category output head matching the label space
used everywhere else in this project (so its accuracy is directly comparable to the
other two approaches), fine-tuned on 68,243 usable teacher-labeled complaints (from a
68,252-row merged pool after dropping 9 parse/off-schema rows; 85/15 train/validation
split) with class-weighted loss to handle the imbalance across categories.

Training produces two different "accuracy" numbers, and only one of them matters:
agreement with the teacher's own training labels (83.2%) isn't a real correctness
check, since the model could just be reproducing the teacher's own mistakes.
Accuracy against the independent gold set (82.2%) is the number that's actually
comparable across every approach in this project, and the one used throughout. That
the two land close together is a healthy sign — the model isn't tracking real ground
truth noticeably worse than it tracks the noisy signal it was actually trained on.

**Results**:

| Metric | Value |
|---|---|
| Gold accuracy (overall) | **82.2%** |
| Gold accuracy, hard/adjudicated subset | 73.3% |
| Gold accuracy, dual-agreement subset | 93.1% |
| Macro-F1 (gold) | 0.775 |

Same hard-vs-easy pattern as the teacher (§2.1) — struggles more on the rows that are
inherently harder, which is expected and healthy. **Takeaway**: an encoder classifier
fine-tuned on distilled LLM labels gets within a few points of the frontier model's
own accuracy, at a small fraction of the inference cost and no per-call API dependency.

## 4. Fine-tuning an open-weight LLM (Qwen2.5-7B) with LoRA

Unlike the encoder, this model generates its answer as free text rather than picking
from a fixed list of 40 — meaning a genuinely new category is just a new string, not a
retrain. It's fine-tuned (LoRA) on the same 68,243-row usable training set (same
68,252-row merged pool minus 9 dropped rows) and scored the same way as the other two
approaches for direct comparability. One practical finding:
cutting the fine-tuning prompt from ~1,775 to ~400 tokens (dropping the worked examples
the teacher needed, since fine-tuning bakes that knowledge into the model's weights
directly instead) roughly halved training time with no accuracy cost. Trained via LoRA
on a rented H100 GPU; the full run took under 7 hours.

Since this model trains on the same noisy teacher labels as the encoder, a bigger model
isn't automatically better — it could just memorize the teacher's mistakes more
precisely instead of generalizing past them. The honest expectation going in was
"somewhat better than the encoder, still short of the frontier teacher." It beat that:
**86.2% gold accuracy, essentially matching the frontier teacher's own 86.6%**, while
also winning clearly on macro-F1 — suggesting the larger model's general language
understanding helped it partially see past some of the teacher's own labeling noise,
rather than just reproducing it.

**Results**:

| Metric | Value |
|---|---|
| Gold accuracy (overall) | **86.2%** |
| Gold accuracy, hard/adjudicated subset | 76.6% |
| Gold accuracy, dual-agreement subset | 98.0% |
| Macro-F1 (gold) | 0.799 |
| Parse failures | 0/551 |

Zero parse failures — the model always produced a valid, schema-consistent answer,
confirming the shortened prompt was sufficient and there's no sign of a common
fine-tuning failure mode where a model, once heavily trained on a fixed label set,
loses the flexibility to handle anything outside it.

### 4.1 Does it still handle genuinely new categories?

To test that directly, 5 synthetic complaints were written describing real vehicle
accessories not well covered by the 40-category schema (a built-in mini-fridge
console, a power tonneau cover, heated/cooled cupholders, a dash-cam system, and a
vehicle-to-home EV charging feature) — deliberately specific rather than vague, so
`UNKNOWN OR OTHER` isn't an obvious safe answer either. Neither model was trained on
anything resembling these.

**Neither model force-fits a nonsensical category.** Both consistently reach for
`EQUIPMENT` (the schema's genuine accessory catch-all) on the clearly-accessory cases,
and both land on the same defensible answer for the EV-charging case (no dedicated
EV-charging category exists, so `HYBRID PROPULSION SYSTEM` is the closest real match,
not a forced miss). The two models disagreed on one case (a dash-cam complaint — one
called it `BACK OVER PREVENTION`, the other `COMMUNICATION`), a reasonable judgment
call either way, not a wrong answer. Net: no evidence either model narrowed itself to
only the categories it saw a lot of during training.

## 5. Results comparison

With all three approaches built, here's how they compare on the same 551-complaint
hand-checked test set — deliberately weighted toward ambiguous and rare-category
cases, so read these as a stress test, not a softball benchmark.

| | Frontier LLM (Claude Sonnet) | Frontier LLM (Claude Haiku) | Confidence cascade (as run) | DeBERTa encoder | Fine-tuned LLM (Qwen2.5-7B) |
|---|---|---|---|---|---|
| Gold accuracy (overall) | 86.6% | 85.1% | 89.8% | 82.2% | **86.2%** |
| Gold accuracy, hard subset | 78.2% | 74.9% | — | 73.3% | **76.6%** |
| Gold accuracy, dual-agreement | 96.8% | 97.6% | — | 93.1% | **98.0%** |
| Macro-F1 | 0.764 | 0.752 | — | 0.775 | **0.799** |
| Cost per 100K predictions | ~$467 (on-demand) | ~$154 (on-demand) | ~$126 (batch, measured) | ~$0 marginal | ~$0 marginal |

Note: the cascade's 89.8% is a backtest estimate from `confidence_cascade_analysis.json`,
not a live re-run of the exact production cascade against this gold set.

Claude Sonnet asked fresh remains the single most accurate option, by well under a
point over the fine-tuned Qwen model — which in turn clearly beats the DeBERTa encoder
on every metric. The fine-tuned LLM's real advantage shows up in §6's cost comparison,
not accuracy: it gets frontier-level results while running on hardware you own instead
of paying per prediction forever.

### 5.1 Where the encoder and the fine-tuned LLM differ, class by class

Across the categories with meaningful gold-set support, the fine-tuned LLM's edge in
macro-F1 (0.799 vs. 0.775) is broad-based rather than concentrated in one or two
classes — it's ahead on most mid-frequency categories (`POWER TRAIN`, `SUSPENSION`,
`ELECTRICAL SYSTEM`, `UNKNOWN OR OTHER`) by small-to-moderate margins, trails narrowly
on a few (`STRUCTURE`, `ENGINE AND ENGINE COOLING`, `AIR BAGS`), and is tied or ahead on
most of the smaller categories too, though single-digit sample sizes there make any one
result more of a directional signal than a precise measurement.

The rare categories specifically targeted by the training top-up and gold-set check
(§2.2) are worth calling out on their own:

| Category (gold examples) | DeBERTa encoder F1 | Fine-tuned LLM F1 |
|---|---|---|
| `EQUIPMENT ADAPTIVE/MOBILITY` (4) | 0.857 | **1.000** |
| `SERVICE BRAKES, ELECTRIC` (4) | 0.000 | 0.000 |
| `ELECTRONIC STABILITY CONTROL` (~5) | 0.500 | 0.615 |
| `TRACTION CONTROL SYSTEM` (2) | 0.667 | 0.667 |
| `CHILD SEAT` (2) | 0.667 | 0.667 |
| `FUEL/PROPULSION SYSTEM` (1) | 0.667 | 0.000 |
| `FUEL SYSTEM, OTHER` (1) | 0.000 | 0.000 |

`SERVICE BRAKES, ELECTRIC` is a miss for both models despite real training volume —
its genuine examples (hybrid regenerative brakes, §2.2) are a narrow pattern that's
easily confused with ordinary ABS complaints, and neither model reliably picked up on
the distinction. The two 1-example categories swing entirely on a single row each, so
they're not meaningful signal either way. `EQUIPMENT ADAPTIVE/MOBILITY` is the clearest
win: real training volume plus real gold coverage now shows both models doing well,
the fine-tuned LLM perfectly.

## 6. Cost, latency, and deployment

### 6.1 Inference cost and latency

Measured directly from this project's own runs, not estimated:

| Approach | Marginal cost / 100K predictions | Throughput (as measured here) | Needs a GPU? |
|---|---|---|---|
| Frontier LLM, Claude Sonnet (on-demand API) | ~$467 | ~4.3 rows/sec at concurrency 12 | No |
| Frontier LLM, Claude Sonnet (Batch API, 50% off) | ~$233 | async, ~24h SLA | No |
| Frontier LLM, Claude Haiku (on-demand API) | ~$154 | ~8.6 rows/sec at concurrency 12 | No |
| Confidence cascade (as run in production) | ~$126 | bulk/async, batch | No |
| DeBERTa encoder (self-hosted) | ~$0 marginal | ~5 rows/sec on a single CPU core | **No — CPU is enough** |
| Fine-tuned LLM, Qwen2.5-7B (self-hosted) | ~$0 marginal if hardware already owned; ~$70/100K if renting a GPU continuously at this unbatched rate | ~0.5 rows/sec, naive one-request-at-a-time generation | **Yes, for reasonable latency** |

A few caveats that matter more than the exact numbers: the fine-tuned LLM's throughput
here is from the simplest possible scoring loop (one request at a time, no batching) —
a real deployment would use a proper inference server (vLLM, TGI) or a quantized model,
both of which would cut this substantially, so read "needs a GPU and some serving
work" as the takeaway, not the $70/100K figure. The encoder's CPU-only throughput is
the more load-bearing number: a small encoder classifier genuinely doesn't need a GPU
to hit usable speed, which is most of why it's the cheapest option to actually run.
And structurally, the frontier-API approach's cost scales linearly with volume
forever, while the two fine-tuned models pay a one-time training cost and then cost
essentially nothing marginal — the break-even volume where self-hosting starts paying
for itself is low, well under 100K predictions even against Haiku's already-cheap price.

### 6.2 Deployment story (as the research left it)

**The DeBERTa encoder** (`checkpoints/deberta_rung2_full_62k_plus_rare/final/`, ~700MB)
is a standard `AutoModelForSequenceClassification` checkpoint — wrap it in a small
FastAPI/Flask service that loads the model once and serves
`(narrative, make, model, year) → component` over HTTP, or call it directly in a batch
job. Runs fine on CPU, trivially horizontally scalable, easy to containerize. Best fit:
a low-ops, low-latency endpoint or a nightly batch re-scoring job, where simplicity and
cost matter more than the last few points of accuracy or the ability to name a
genuinely novel category.

**The fine-tuned LLM** (`checkpoints/llm_rung3_full_62k_plus_rare/`, a 165MB LoRA
adapter over `Qwen/Qwen2.5-7B-Instruct`) needs the ~15GB base model plus this adapter
loaded together — either kept separate for hot-swappable adapters on one shared base
model, or merged once into a standalone deployable model. Needs a GPU with enough VRAM
for a 7B model (~16GB+ in bf16, less with quantization); for real throughput, swap the
simple scoring loop used here for a proper inference server. Best fit: where the
accuracy edge and the flexibility to name a genuinely novel category on the fly (§4.1)
justify the extra GPU hosting cost — e.g. an internal analyst-facing tool, or a
lower-volume pipeline where per-row accuracy matters more than raw throughput.

That was the research tradeoff: the 7-billion-parameter model is more accurate on the
551-row stress-test set, and the smaller classifier is cheaper to run. Sections 7–8
measure the same models on complaints filed *today*, and explain why the live service
calls the smaller one anyway.


## 7. Does this hold on complaints filed today?

The 551-row answer key in §2–§5 was drawn **evenly across 1995–2026**. That is a fair
stress test of rare and ambiguous cases. It is not a sample of what arrives this
month.

The mix of complaints has changed a lot. On the teacher's labels for the 62,000-row
random training sample:

| Share of complaints | 2015–19 | 2020–22 | 2025–26 |
|---|---|---|---|
| Forward collision avoidance | 0.9% | 8.5% | 11.8% |
| Back-over prevention | 0.5% | 4.1% | 13.3% |
| Lane departure | 0.3% | 3.0% | 4.1% |

NHTSA's own published field shows the same rise. Complaints about driver-assist
systems (automatic emergency braking, backup cameras, lane-keeping) went from near
zero to a large share of current volume. Meanwhile, only **82 of 551** answer-key
rows are from 2023 or later, and the driver-assist categories have **13** answer-key
examples in total. Slicing the 551-row set by the year NHTSA received the complaint
shows the smaller classifier's point estimate dipping after 2022 (about 76% on the
37 rows from 2025–26), but those year slices are tiny — a 37-row slice has a
confidence interval roughly 60% to 87%, which still overlaps the overall 82.2%.
That is not a finding that the model has failed on recent complaints. It is the
reason a separate, randomly sampled current-complaints key exists: n=37 cannot
decide a 6-point gap.

So the 86.2% in §4 is a fair score on the answer key's mix of complaints, but nobody
had measured accuracy on what arrives today. That is the question this half of the
writeup is for.

### 7.1 A second answer key, labeled without seeing any model

We drew a uniform random sample of 150 vehicle complaints NHTSA received after
26 August 2026 (the last date in the research copy of the file). Random sampling is
what makes the accuracy number apply to live traffic: an oversample of
driver-assist complaints would answer a different question. Every complaint ID that
had already been used in training or in the 551-row key was excluded, so this is
genuinely unseen text.

Three of the 150 were not about a vehicle system at all — aftermarket child seats
filed on a make/model, title fraud hung on a vehicle, an aftermarket seat after a
crash. Those rows stay in the sample so the out-of-scope rate is visible (3/150 =
2%), and they are **dropped from the accuracy denominator**. They are not labeled
`CHILD SEAT`, `EQUIPMENT`, or `UNKNOWN OR OTHER`. Stuffing them into a bucket would
lie about what the 40 categories mean. The scored set is **147** filings received
27 August through 17 September 2026. The frozen file is
`output/prod/gold_current_v1.json`.

The labels were not written by the original teacher (Claude Sonnet). Qwen was
trained on Sonnet-style labels; grading Qwen — or Sonnet — against Sonnet would be
circular, and §2.1's 86.6% already includes the same coding rules. Instead, two
independent models from different companies (Claude Opus and GPT) each read the
story and the category rulebook, without seeing each other, NHTSA's field, DeBERTa,
Qwen, or Sonnet. Where their lists of acceptable answers overlapped (ordinary
brake names treated as one family), that overlap became the label. Where they
disagreed, a person settled the fight. The person also audited a random 20 of the
agreements. Every human change is logged (row, before, after, why) in
`output/prod/gold_current_v1_revisions.jsonl`, so a first-pass score can still be
separated from a later revise. Looking up what a part *is* ("what is a control
arm?") was allowed. Asking any model, or opening NHTSA's field, for *this
complaint's category* was not — that would make the labels partly the model's.

A prediction counts as correct if it lands in that row's list of acceptable
answers, the same rule as the 551-row key. The new lists are **wider** than the
old ones: 1.80 acceptable labels per row on average, and 43% of rows accept more
than one name, versus 1.33 and 23% on the 551-row set. That matters in §7.2.

A separate 50-complaint slice, drawn from filings NHTSA had already tagged as
forward-collision, back-over, or lane-departure, was labeled the same way and
**kept in its own file**. It is a sanity check for whether the serving model is
quietly failing on driver-assist stories. It is not mixed into the 147, and it is
not a 2018-versus-2026 replay.

### 7.2 Scores on the 147, with a range around each percentage

Because 147 is a small sample, each percentage comes with a 95% confidence
interval: a range that says how precise the percentage is. If we drew
another random 147 from the same pile of recent filings, the true rate would
usually land in that range. When two models' ranges overlap, the gap between them
is too small to treat as a real win. All three models below were scored on the
same frozen 147; the 7-billion-parameter model and the teacher were not used to
build the labels.

| | Correct / 147 | Accuracy | 95% confidence interval |
|---|---|---|---|
| Frontier model (Claude Sonnet), original teacher prompt, not the panel prompt | 135 / 147 | 91.8% | 86.3% – 95.3% |
| Fine-tuned 7-billion-parameter model (existing adapter, no retrain) | 134 / 147 | 91.2% | 85.5% – 94.8% |
| Smaller classifier (DeBERTa, same checkpoint as §3) | 132 / 147 | 89.8% | 83.8% – 93.7% |
| Same smaller classifier, on the 551-row key (this sitting) | 453 / 551 | 82.2% | 78.8% – 85.2% |

The published §3 / §4 numbers reproduced exactly on the 551-row key this sitting
(DeBERTa 82.2%, Qwen 86.2%), so the research scores have not drifted.

All three current-key intervals overlap. The smaller classifier is two hits behind
the 7-billion-parameter model and three behind the teacher. That is not a reason
to pick one over the others. The 12–15 misses are the same seams we already knew
from §2.1 — `UNKNOWN OR OTHER` on the key versus a specific guess (hesitation,
stall, undiagnosed reboot), power-train versus engine versus speed control,
driver-assist family mixups, a Tesla cracked center screen (communication versus
visibility), an EV battery pack (hybrid propulsion versus electrical). There is
no new 2026 failure mode in this sample.

The 89.8% versus the old 82.2% looks like a jump. **It is not a model upgrade.**
The confidence intervals still overlap (83.8%–93.7% versus 78.8%–85.2%), and the
new key's acceptable-answer lists are wider, so more predictions count as hits
for a reason that has nothing to do with the weights. A fair reading is only:
**there is no evidence the serving model has failed on late-August /
mid-September 2026 traffic.**

On the separate 50-complaint driver-assist slice, the same smaller classifier
scored 41 / 50 (82.0%, interval 69.2%–90.2%). That range overlaps the random-147
range. The nine misses are mostly mixups inside the driver-assist family
(forward-collision versus lane-departure versus backup camera) and unknown versus
a gadget guess. n=50 is too wide to claim a drop, and this number is not mixed
into the 147.

NHTSA's own field versus this frozen key is a **drift signal, not accuracy**. The
primary category NHTSA published matched our primary on 62 / 147 (42.2%, interval
34.5%–50.3%) — the same noisy plateau this project has seen since about 2015.
The smaller classifier tracks NHTSA at about the same 43% and tracks the
hand-checked key at 90%. That is the point of the independent key: agreeing with
NHTSA would have looked like a coin flip, and would have been the wrong number
to report.

## 8. What we serve, and why

The research comparison in §5 still holds: on the 551-row stress-test set, the
7-billion-parameter model clearly beats the smaller classifier and essentially
matches the teacher. The serving decision is a different question. It asks: given
today's filings, and that the smaller model runs on a regular computer while the
larger one needs a rented graphics card, which checkpoint does the live page call?

Measured on this sitting (same checkpoints as §3–§4; current key as in §7):

| Variant | 551-row key | 147 current filings |
|---|---|---|
| **Smaller classifier (what the service actually calls)** | 82.2% (78.8%–85.2%) | **89.8% (83.8%–93.7%)** |
| 7-billion-parameter model | **86.2% (83.1%–88.8%)** | 91.2% (85.5%–94.8%) |
| Claude Sonnet, asked the same way it labeled the training data | 86.6% (published) | **91.8% (86.3%–95.3%)** |

**The live service calls the smaller classifier.** The 7-billion-parameter model's
point estimate on the current key is 1.4 points higher; the confidence intervals
overlap, so the written promotion rule in §8.1 forbids treating that as a win.
The larger model stays the offline benchmark: we can still score it on a rented
graphics card when we need the comparison, and we do not pretend it lost the
research match in §5.

The service runs only on this computer. It is not on the public internet. When
the computer is off, the classify page and the model catalog are down. A
prediction returns the category, how sure the model is, the next-best
alternatives, and which model version produced it. Training and serving format
each complaint through the same function (`prod/preprocess.py` imports the
research text builder; a test fails if those two strings ever diverge).

### 8.1 When a new model is allowed to replace the live one

A challenger is promoted only if **all** of the following hold. The numbers were
written down before any retrain (`docs/THRESHOLDS.md`), so we could not move them
after seeing the result.

1. On the current-complaints answer key (§7), the challenger's overall accuracy
   **point estimate** is higher than the live model's.
2. On the 551-row key, the challenger drops by **at most 2.0 percentage points**
   overall.
3. On the hard tier of the 551-row key, the challenger drops by **at most 3.0
   percentage points**.
4. If the two current-key 95% confidence intervals overlap, **do not promote**.
   That is a no-clear-win, not a win. Retrain or collect more labels.

Teacher disagreement and agreement with NHTSA's field are not inputs to this
rule. They are drift alarms. Rollback is moving the live alias back.

That is why the 7-billion-parameter model was not promoted after the current-key
score: rule 4 is a no. It is also why the November 2025 retrain in §9 was not
promoted.

The model catalog (MLflow, local, same computer) records three versions under
the name `deberta-complaint`:

- **`champion`** — the full-history smaller classifier from §3. This is what
  `/classify` calls.
- **`replay_asof_2019`** — trained only on complaints received through
  31 December 2019. A historical exhibit. Not served.
- **`rejected_challenger_202511`** — the November 2025 retrain from §9. The
  gate refused it. Not served.

Every reported production number traces to a recorded run in that catalog.
GitHub checks on each push confirm the shared text format and that a planted
leak of a hand-checked complaint ID into a fake training file is caught.
Hand-checked rows never enter training.

## 9. Watching new complaints, and a historical replay

NHTSA publishes an updated complaint file most days. The datasets web page's
displayed date is stale; the file's `Last-Modified` header is what to trust.
About 300–350 new vehicle complaints arrive per day. The public
by-vehicle API is lookup-only — it is not a "new since date" feed — so the
source of record is NHTSA's flat file (the newest five-year chunk most days, plus
a periodic full-file pass if we ever start seeing real category recodes of old
rows).

**NHTSA's published complaint ID is not a stable key.** The documentation says
it is updateable. A naive join of two dumps on that ID looked like tens of
thousands of category edits. That was remapping: the same story had slid to a
neighboring ID. Rematched on the incident number plus the start of the
narrative, the same window showed 27,430 ID moves, **three** category-string
changes (all more-specific suffixes of the same top-level system — wheels to
wheels/lugs, electrical to electrical/battery, and so on), and a handful of
make/model/year string edits. In that window, top-level recodes of an
already-published complaint were effectively zero. Production ingest therefore
stores a fingerprint (incident number + normalized narrative) and treats the
published complaint ID as dump-local.

`python -m prod.loop` pulls the newest chunk, scores new rows with the live
classifier, and checks the mix rule below. Scoring the live model is safe on
this computer's processor. Retraining is not: this box has no usable graphics
card for training, and the loop will not launch a paid machine unless someone
passes an explicit flag after the mix rule has already fired.

### 9.1 The mix rule (not accuracy)

We do not retrain because "the teacher disagreed with the student" or because
"we agree with NHTSA 42% of the time." Those numbers are drift signals. The
retrain trigger that is actually locked is about **NHTSA's own published
category mix on every vehicle filing** — a census, not a 2,000-row sample, and
not our model's predictions.

A new month is compared to a long earlier pile of filings (twenty years). The
"how far is unusual" line is the average distance of the last 24 months of that
window, plus two sample standard deviations. A fire is **two complete months in
a row** over that line. After a retrain, both windows slide to the retrain
month, so we are not forever comparing 2026 to 1998–2017. The formula and the
windows were written down before the walk (`docs/THRESHOLDS.md`), and an earlier
draft that compared new months to a *different* pile than the one used to set
the line was thrown out.

This is a statement about how the published file's category shares have moved.
It is not a statement that our model got worse.

### 9.2 The replay: freeze a 2019 model, watch 2020–2026 arrive

To see whether that alarm would have been useful, we ran a **historical
replay**. It is always called a replay: no model in it sees labels dated after
its as-of date. We did not use the late-2026 current-complaints key to decide a
2021 promotion — that would peek at the future.

The cutoff is the date NHTSA **received** the complaint (`LDATE`), not the model
year of the car. A 2014 car can already have cameras; 2014 *filings* were still
about 0% driver-assist categories. In this file the jump in filings is
2021–2022. End of 2019 leaves 49,379 usable teacher-labeled rows for the as-of
train and 18,873 dated 2020 or later. Every hand-checked ID was dropped. No new
teacher API calls; no 7-billion-parameter retrain in this replay.

We trained a DeBERTa on the through-2019 labels, then walked January 2020
through August 2026. The mix line starting from end-of-2019 was 0.3677 (last
24 months of 2018–2019 versus filings from 1998 through 2017). Against that
same 1998–2017 pile, the walk's first two-month fire was **October and November
2025**. We retrained on teacher labels through November 2025. The windows then
slid (last 24 months through November 2025 versus the twenty years before
that); the new line was 0.2631; there was **no second fire** through August
2026.

Promotion inside the replay used only the dated slices of the 551-row key that
were in-time for that month — 420 rows received through 2019, 131 rows received
after 2019 — with the same "overlapping confidence intervals means do not
promote" rule. The late-2026 current key was not used.

| | Through-2019 slice (n=420) | After-2019 slice (n=131) |
|---|---|---|
| 2019-only model | 346 / 420 = 82.4% (78.4%–85.7%) | 94 / 131 = 71.8% (63.5%–78.8%) |
| Retrain through November 2025 | 345 / 420 = 82.1% (78.2%–85.5%) | 106 / 131 = 80.9% (73.3%–86.7%) |

The retrain's point estimate is higher on the later slice and within a fraction
of a point on the older slice (well inside the 2-point / 3-point drop caps).
**Both pairs of confidence intervals overlap**, so the written rule refused the
swap. The live service still uses the original full-history classifier from §3,
not the 2019-only copy and not the November 2025 retrain. Those two checkpoints
live in the catalog as history so a reviewer can see the loop ran, and see that
"looks better" is not the same as "cleared the rule."

Agreeing with the teacher on 2020-and-later teacher-labeled rows during the
walk is a **proxy**, never an accuracy number. Real accuracy in the replay is
only those dated slices of the hand-checked 551-row key, and even there the
after-2019 slice is 131 rows — wide enough that a 9-point point-estimate gap
can still be a coin flip, which is exactly what the gate said.

A later live mix check, after the November 2025 windows had slid, has not
produced a second two-month fire. September 2026 was partly over the new line
in an incomplete month; incomplete months do not count.

## 10. Limitations, stated plainly

- **Small hand-checked samples.** The current-complaints key is 147 scored rows,
  not the 300 we would have liked. A 147-row 90% has a confidence interval about
  84%–94%. That is wide enough that a several-point gap between models is often
  noise, which is why overlapping intervals block a promotion. Growing the key
  is the honest way to make the gate decidable; loosening the gate is not.
- **The two answer keys are not interchangeable.** The current key's acceptable-
  answer lists are wider than the 551-row key's. Do not read 89.8% versus 82.2%
  as the model getting better.
- **The teacher cannot grade the student.** Sonnet at 91.8% and the smaller
  classifier at 89.8% on the same 147 is a useful comparison only because a
  third, independent panel built the key. Teacher–student disagreement on new
  unlabeled filings is a drift alarm, never an accuracy number.
- **Replay accuracy against the teacher is a proxy.** The promotion decision
  inside the replay used dated slices of the 551-row hand-checked key, and even
  those slices are small (131 post-2019 rows).
- **Forty categories is a ceiling.** NHTSA's raw field is noisier than 40
  names, and a few real systems still sit on fuzzy seams (steering versus
  suspension, unknown versus a specific guess). Some complaints are inherently
  ambiguous even to a careful reader (§2.1).
- **NHTSA's field is not a target.** A 42% match with NHTSA is the post-2015
  plateau, not a quality score.
- **This is not a public service.** No dashboard of defect trends, no
  "similar past complaints" evidence on each prediction, no always-on cloud
  host. The live path is a local classify page and a local catalog on one
  computer.
- **Out-of-scope filings exist.** About 2% of a random current sample was not
  a vehicle system. Those rows are excluded from the component score; they are
  not a 41st class the classifier was trained to emit.
