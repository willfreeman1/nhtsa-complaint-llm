# Automating NHTSA Vehicle Complaint Categorization

[![Nightly pull](https://github.com/willfreeman1/nhtsa-complaint-llm/actions/workflows/nightly.yml/badge.svg)](https://github.com/willfreeman1/nhtsa-complaint-llm/actions/workflows/nightly.yml)
[![CI](https://github.com/willfreeman1/nhtsa-complaint-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/willfreeman1/nhtsa-complaint-llm/actions/workflows/ci.yml)

Every year, drivers file hundreds of thousands of free-text complaints with NHTSA
(the National Highway Traffic Safety Administration — the federal agency that tracks
vehicle safety and issues recalls) — things like *"the brake pedal went to the floor
without stopping the vehicle."* For complaints to be quantified, someone has to tag
each one with which vehicle **component** it's actually about (brakes, airbags,
steering, and so on) so safety engineers can spot patterns across thousands of
complaints and catch the next recall early.

NHTSA's own published data already carries a category for each complaint, but it's
noisy — this project manually re-checked a sample against the actual complaint text and
found real mislabeling (see [`WRITEUP.md`](WRITEUP.md) §2.2). On rare categories,
NHTSA's own tag was right only about 30% of the time. Text like this is also exactly
the kind of task language models are good at, and manual review doesn't scale to
hundreds of thousands of filings.

**This project trains a small model that reads a complaint and names the vehicle
part.** A 7-billion-parameter open model (Qwen2.5-7B), trained once on labels from an
expensive general-purpose model you pay per question (a frontier model), comes within
half a point of that frontier model asked fresh every time — at effectively zero
cost per prediction afterward. What we actually
run as a service is a smaller text classifier (DeBERTa-v3-base). On complaints filed
*today*, the two are close enough that the gap sits inside ordinary sampling noise,
and the smaller one runs on a regular computer. The 7-billion-parameter model stays
the offline check.

A scheduled GitHub job pulls new NHTSA filings each day, scores the ones we have
not seen, and checks whether NHTSA's own published category mix has moved. The
latest one-page summary is [`output/prod/nightly.json`](output/prod/nightly.json).
That job does not need this computer to be on. The try-it page and the model
catalog still run only here.

The full methodology, every metric, and the error analysis behind each claim is in
[`WRITEUP.md`](WRITEUP.md).

## The result

The first table is the original research score. It uses a 551-complaint hand-checked
answer key drawn evenly across 1995–2026 — a stress test, weighted toward rare and
ambiguous cases, not a softball. A prediction counts as correct if it lands in that
row's list of acceptable answers (some complaints honestly support more than one
category).

| | Accuracy on the 551-complaint hand-checked set (1995–2026 mix) |
|---|---|
| Frontier model, asked directly, no training (the accuracy ceiling, and expensive to run at scale) | 86.6% |
| **This project's fine-tuned 7-billion-parameter model** (trained once, then cheap to run) | **86.2%** |
| Smaller fine-tuned classifier (DeBERTa), trained the same way | 82.2% |

The 7-billion-parameter model gets within half a point of the expensive frontier
model it learned from, with no ongoing bill to a hosted service. The same model,
asked with the same written rules but *before* we trained it on our labels, scored
71.5% on this set — so the training improved it by about **15 points** (71.5% →
86.2%). See [`WRITEUP.md`](WRITEUP.md) §4.0 for that before/after, §6 for the full
cost comparison, and §2.1 for an honest look at what that ~87% ceiling actually
means: a chunk of the "misses" are genuinely ambiguous complaints, not clean
model errors.

That score is fair on the answer key's mix of complaints. It is **not** a score on
what arrives this month. Driver-assist complaints (forward-collision braking, backup
cameras, lane-keeping) went from almost none of the file in the late 2010s to a large
share of current volume. Only 82 of those 551 hand-checked rows are from 2023 or
later, and the driver-assist categories have 13 examples in total. So we built a
second answer key: 150 complaints chosen at random from filings NHTSA received after
26 August 2026 (none of them were in training). Three were not about a vehicle system
at all — aftermarket child seats, title fraud — and were dropped from the score
rather than stuffed into a bucket. The remaining 147 are the number below.

Because 147 is a small sample, each percentage comes with a 95% confidence interval:
a range that says how precise the percentage is. If two models' ranges overlap, the
gap between them is too small to treat as a real win.

| | Correct on 147 current filings (received 27 Aug – 17 Sep 2026) | 95% confidence interval |
|---|---|---|
| Frontier model (Claude Sonnet), asked the same way it labeled the training data | 135 / 147 (91.8%) | 86.3% – 95.3% |
| Fine-tuned 7-billion-parameter model | 134 / 147 (91.2%) | 85.5% – 94.8% |
| **Smaller classifier (what the service actually calls)** | **132 / 147 (89.8%)** | **83.8% – 93.7%** |

All three ranges overlap. The 89.8% is also **not** a like-for-like jump over the old
82.2%: the new answer key allows more acceptable answers per row (on average 1.80
labels, versus 1.33 on the older set), so some of the point gap is how the key was
built, not a better model. A fair reading is only: **there is no evidence the serving
model has failed on late-August / mid-September 2026 traffic.** See
[`WRITEUP.md`](WRITEUP.md) §7.

## How it works

There's no large existing dataset of correctly-labeled complaints to train on —
NHTSA's own field is exactly the noisy thing we're trying to fix. So the project
builds its own training data, then asks a second question the original score could
not answer.

1. **Use a frontier model (Claude) as a labeling teacher.** Rather than pay a person
   to hand-label tens of thousands of complaints, a carefully-prompted frontier model
   reads each one and assigns a category, with a cost-saving trick (cheap model
   first, escalate only the uncertain cases to a stronger model) that kept labeling
   about 68,000 complaints under a few hundred dollars.
2. **Fine-tune two small models on those labels.** One is a 7-billion-parameter open
   model (Qwen2.5-7B), trained to reproduce the teacher's judgment — distilling a
   large, expensive model's behavior into a small, cheap one. The other is a more
   traditional text classifier (DeBERTa-v3-base): it picks from a fixed list of 40
   categories and does not generate free text. On the 551-complaint stress-test set,
   the 7-billion-parameter model clearly beat the smaller one and essentially matched
   the teacher (see the first table and [`WRITEUP.md`](WRITEUP.md) §5). That
   comparison is still the research finding.
3. **Grade everything against a small, hand-checked answer key** — not NHTSA's raw
   field, and not the teacher's own labels (that would just be checking the teacher
   agrees with itself). 551 complaints were read and categorized by hand against a
   written rulebook, specifically including examples of the rarest, most-often-
   mislabeled categories, so the reported accuracy numbers reflect *actual*
   correctness, not just agreement with a noisy source.
4. **Then ask whether that score holds on complaints filed today.** A new 147-row
   answer key was labeled without anyone seeing NHTSA's field or any of our models
   (two independent models from different companies proposed labels; a person settled
   the close calls and audited a sample of the agreements). The teacher is not
   allowed to grade the student: when the teacher and a trained model disagree, there
   is no way to tell which one is wrong, so disagreement is a drift alarm, never an
   accuracy number. Accuracy on new data comes only from this kind of independent
   check. See [`WRITEUP.md`](WRITEUP.md) §7.

The headline is simpler than the methodology: **a small model, fine-tuned once, can
do this specific job about as well as an expensive frontier model asked fresh every
time — and the even-smaller classifier is close enough on today's filings that we
run that one, because it fits on this computer.**

## Try it

`scripts/predict.py` loads the trained classifier and names the vehicle part in a
complaint — this computer's processor only, no graphics card or hosted API needed.

```bash
python scripts/predict.py --demo
# or classify your own narrative:
python scripts/predict.py "The brake pedal went to the floor..." --make Toyota --model Camry --year 2019
```

Real output from `python scripts/predict.py --demo` (about 10 seconds, including
model load):

```text
Device: cpu
Loading checkpoint: checkpoints/deberta_rung2_full_62k_plus_rare/final

Narrative: The brake pedal went to the floor without stopping the vehicle. I had to use the emergency brake to...
Vehicle: 2019 Toyota Camry

Predicted component: SERVICE BRAKES, HYDRAULIC  (96.9%)
Other candidates:
  SERVICE BRAKES                 1.4%
  SERVICE BRAKES, AIR            0.7%

Narrative: During a minor fender bender at low speed, the driver's side airbag deployed unexpectedly and caused...
Vehicle: 2020 Honda Accord

Predicted component: AIR BAGS  (99.9%)
Other candidates:
  SEAT BELTS                     0.0%
  ELECTRICAL SYSTEM              0.0%

Narrative: While driving on the highway at 65 mph, the steering wheel became extremely loose and the vehicle wa...
Vehicle: 2018 Ford F-150

Predicted component: STEERING  (99.7%)
Other candidates:
  ELECTRICAL SYSTEM              0.0%
  EQUIPMENT ADAPTIVE/MOBILITY    0.0%
```

This demo runs the smaller classifier, because that is what is practical here with
no extra setup — the same model the local service calls. See
[`WRITEUP.md`](WRITEUP.md) §6 and §8 for how the 7-billion-parameter model would be
served, and why we do not do that as the live path.

## What we run

The daily pull — new filings, scores on those new rows, and the mix check — runs
on a schedule ([nightly workflow](https://github.com/willfreeman1/nhtsa-complaint-llm/actions/workflows/nightly.yml)).
Latest result: [`output/prod/nightly.json`](output/prod/nightly.json).

The same smaller classifier also sits behind a local web service and a model
catalog. **Those two are not on the public internet. When this computer is off,
the classify page and the catalog are down.** The scheduled pull keeps going.

A prediction comes back with the category, how sure the model is, the next-best
alternatives, and which model version produced it. A new model is allowed to replace
the live one only if it is clearly better on the current-complaints answer key
*and* does not give back more than a locked amount on the older 551-row key. If the
confidence intervals overlap, that is a coin flip, not a win — we keep the live
model. That rule was written down before any retrain, so we could not move the
goalposts after seeing the numbers. See [`WRITEUP.md`](WRITEUP.md) §8.

The catalog also keeps two historical copies so a reviewer can see them. They are
**not** what the classify page uses:

- a model trained only on complaints received through the end of 2019 (a
  historical replay — see below)
- a retrain through November 2025 that looked better on later years but did not
  clear the written rule, because the error bars still overlapped

```bash
docker compose up -d postgres mlflow
python -m prod.register_models
python -m prod.api
```

- Classify (and a try-it form): http://127.0.0.1:8000/docs
- `GET /health` and `GET /model` need no complaint — they just say whether the
  service is up and which model is live
- Model catalog: http://127.0.0.1:5000 — **Models** → `deberta-complaint`. The
  name marked live is `champion` — that is the model the classify page calls. The
  2019 replay and the rejected November 2025 retrain are listed there as history.

```bash
curl http://127.0.0.1:8000/classify -H "Content-Type: application/json" -d "{\"narrative\":\"The brake pedal went to the floor without stopping the vehicle.\",\"make\":\"Toyota\",\"model\":\"Camry\",\"year\":\"2019\"}"
```

Every push to GitHub runs a small check: training and serving format a complaint
the same way (so we cannot silently score a different string than we trained on),
and a planted leak of a hand-checked complaint ID into a fake training file is
caught. Hand-checked rows are never allowed into training. See
[`WRITEUP.md`](WRITEUP.md) §8.

## Watching new complaints, and a historical replay

NHTSA publishes an updated complaint file most days — about 300 vehicle filings a
day. `python -m prod.loop` pulls the newest chunk, scores new rows with the live
classifier, and checks whether NHTSA's own category mix has moved far enough, for
long enough, that a retrain is worth considering.

A filing counts as new if its incident number plus the start of the narrative is
not already on disk. NHTSA's published complaint ID can move between dumps — the
same story can show up under a different ID — so that ID alone is not a stable
key. In one window, tens of thousands of IDs remapped; real edits to the
*category* of an already-published complaint were essentially zero (three rows
got a more-specific suffix of the same top-level system). See
[`WRITEUP.md`](WRITEUP.md) §9.

We do not retrain because "the teacher disagreed" or because "we agree with
NHTSA." Those are drift signals, not accuracy. Retrain is considered only after
**two complete months in a row** where NHTSA's own published category mix stays
unusually far from the mix in a long earlier window. The distance and the "how
far is unusual" line were locked before the walk, so we could not invent a
threshold after seeing 2021.

To see whether that alarm would have been useful, we ran a **historical replay**
(always called a replay: no model in it sees labels dated after its as-of date).
We trained a copy of the smaller classifier using only teacher labels from
complaints received through 31 December 2019 — a year when driver-assist
categories were still almost absent from *filings*, even though some cars already
had the cameras. We then walked January 2020 through August 2026. The mix alarm
fired on October and November 2025. We retrained using labels through November
2025.

That retrain looked better on hand-checked complaints received after 2019 (about
81% versus 72% for the 2019-only model) and about the same on the older ones.
The error bars still overlapped on both slices, so the written rule refused the
swap. The live service still uses the original model trained on the full
history, not the 2019-only copy and not the November 2025 retrain. After that
retrain, the mix line slides forward; there has not been a second two-month
fire through August 2026. See [`WRITEUP.md`](WRITEUP.md) §9.

## Repo layout

```
scripts/            Research pipeline — data prep, labeling, training, evaluation
prod/               Local classify service, model catalog, scheduled pull / mix job
output/             Eval results, answer keys, label schema, run summaries (JSON)
data/nightly/       Monthly NHTSA category counts and hashed ids of new filings
data/               Training data (parquet) — regenerated by scripts, not fully versioned
checkpoints/        Trained model configs/adapters (weight binaries gitignored — see below)
PROJECT_PLAN.md     Original plan/rationale for the three-way comparison
WRITEUP.md          Full writeup: methodology, every metric, error analysis, later serving work
```

### Research pipeline, in order

1. **Data prep** — `scripts/load_data.py` (raw NHTSA flat file → parquet) →
   `scripts/prepare_training_data.py` (clean, keep vehicle filings, dedupe).
2. **Label schema + first answer key** — `scripts/schema.py`,
   `output/label_schema.json`, `output/NHTSA_CODING_GOTCHAS.md`. The hand-checked
   551-row key went through several rounds of manual review before landing on
   `output/gold_eval_set_v3.json`; only the last, best-documented expansion step
   is kept as a runnable script —
   `scripts/build_rare_gold_expansion.py` (samples real complaints for
   rare categories the key was missing) + `scripts/build_gold_eval_set_v3.py`
   (adjudicates and merges them in) — see `WRITEUP.md` §2.2.
3. **Teacher labeling** — `scripts/teacher_prompt.py` (prompt and category
   definitions) + `scripts/run_batch_cascade_labeling.py` (the cheap-model-first,
   escalate-if-uncertain cascade, run as a batch job over ~62,000 complaints plus
   a targeted rare-category top-up via
   `scripts/build_training_sample_rare_topup.py`). Cost modeling in
   `scripts/pricing.py` and `scripts/analyze_confidence_cascade.py`.
4. **Teacher eval** — `scripts/eval_models_on_gold.py` scores any untrained
   frontier model against the hand-checked answer key.
5. **Small classifier fine-tune** — `scripts/train_deberta.py` (DeBERTa-v3-base;
   a graphics card is recommended for training). Eval a saved run with
   `scripts/eval_checkpoint_deberta.py`.
6. **Small language-model fine-tune** — `scripts/train_llm_lora.py`
   (Qwen2.5-7B-Instruct; a graphics card is required). Eval a saved run with
   `scripts/eval_checkpoint_llm.py`. This is the model that matched the teacher
   on the 551-row key.
7. **Novel-category flexibility probe** — `scripts/novel_category_probe.py`
   (shared synthetic narratives) checks whether either fine-tuned model still
   generalizes to accessories it saw little or no training data for — see
   `WRITEUP.md` §4.1.

Every eval above runs against the held-out hand-checked answer key the models
never saw in training — judged against a written rulebook
(`scripts/teacher_prompt.py`), not just NHTSA's raw, noisy field (see
`WRITEUP.md` §2.1–2.2 for how much that judgment actually changes the numbers).

### Later production code

- Current-complaints answer key: `prod/panel.py`, `prod/panel_prompt.py`,
  frozen file `output/prod/gold_current_v1.json` — see `WRITEUP.md` §7.
- Daily pull, score, and mix check (GitHub Actions): `python -m prod.nightly run`
  — same job locally: `python -m prod.loop`
- Local classify service: `python -m prod.api`
- Model catalog registration: `python -m prod.register_models`
- Historical replay and the November 2025 retrain: `prod/replay_run.py`,
  `prod/replay_promote.py`, `prod/nhtsa_mix_trigger.py` — see `WRITEUP.md` §9.

## Reproducing

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY / OPENAI_API_KEY as needed
python scripts/load_data.py
python scripts/prepare_training_data.py
# ... see WRITEUP.md for the full sequence + expected runtimes/costs at each stage
```

The small classifier trains in a few hours on a single L4-class graphics card;
the 7-billion-parameter fine-tune took under 7 hours on a rented H100 in this run
(and will take longer on smaller cards) — see `WRITEUP.md` §4 for the runtime
breakdown and the prompt-shortening change that got it there.

The production scoring path does **not** need a graphics card. It loads the
smaller classifier on this computer's processor. Renting a graphics card was
only for the original 7-billion-parameter train, for scoring that model on the
new answer key, and for the historical replay trains.

### On model weights

`checkpoints/` tracks each trained run's small config / tokenizer / adapter
files (so the exact settings of every run are visible in the repo) but not the
multi-hundred-MB weight binaries themselves (`*.safetensors`, `*.bin`) — those
are regenerated by re-running the training scripts above against a rented
graphics card.

## License

See [`LICENSE`](LICENSE).
