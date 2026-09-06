# NHTSA Complaint Classification — Project Writeup

**The National Highway Traffic Safety Administration's (NHTSA's) public complaint database tags every complaint with a
vehicle component, but that tagging is noisy. This project fine-tunes a small,
open-weight LLM that reads a raw complaint narrative and assigns the correct component
instead — accurately enough to nearly match a frontier LLM asked fresh every time, at a
small fraction of the ongoing cost once trained.**


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

### 6.2 Deployment story

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
