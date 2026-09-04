# Automating NHTSA Vehicle Complaint Categorization with a Fine-Tuned LLM

Every year, drivers file hundreds of thousands of free-text complaints with NHTSA
(the federal agency that tracks vehicle safety and issues recalls) — things like *"the
brake pedal went to the floor without stopping the vehicle."* For that complaint to be
useful, someone has to tag it with which vehicle **component** it's actually about
(brakes, airbags, steering, etc.) so safety engineers can spot patterns across
thousands of complaints and catch the next recall early.

NHTSA's own published data already carries a category for each complaint, but it's
noisy — this project manually re-checked a sample against the actual complaint text and
found real mislabeling (see [`WRITEUP.md`](WRITEUP.md) §2.2). Text like this is also
exactly the kind of task language models are good at and manual review doesn't scale to.

**This project fine-tunes a small, open-weight LLM (Qwen2.5-7B) that reads a complaint
narrative and outputs the correct component category — cheaply enough to run on any
complaint at scale, and about as accurately as asking a frontier model (GPT-5-class)
to do it fresh, one at a time, forever.**

## The result

| | Accuracy on a 551-complaint hand-checked test set |
|---|---|
| Frontier LLM, asked directly, no training (the accuracy ceiling, and expensive to run at scale) | 86.6% |
| **This project's fine-tuned 7B model** (trained once, runs on commodity hardware, ~free per prediction after that) | **86.2%** |
| A smaller fine-tuned classifier, trained the "traditional" way for comparison | 82.2% |

The fine-tuned model gets within half a point of the expensive frontier model it
learned from — at effectively zero cost per prediction once trained, with no ongoing
API bill. See [`WRITEUP.md`](WRITEUP.md) §6 for the full cost comparison, and §2.1 for
an honest look at what that ~87% ceiling actually means (a chunk of the "misses" are
genuinely ambiguous complaints, not clean model errors).

## How it works, briefly

There's no large existing dataset of correctly-labeled complaints to train on — NHTSA's
own field is exactly the noisy thing we're trying to fix. So the project builds its own
training data in three steps:

1. **Use a frontier LLM (Claude) as a labeling teacher.** Rather than pay a person to
   hand-label tens of thousands of complaints, a carefully-prompted frontier model
   reads each one and assigns a category, with a cost-saving trick (cheap model first,
   escalate only the uncertain cases to a stronger model) that kept labeling ~68,000
   complaints under a few hundred dollars.
2. **Fine-tune a small model on those labels.** The actual deliverable: a 7-billion-
   parameter open-weight LLM, fine-tuned (LoRA) to reproduce the teacher's judgment —
   distilling a large, expensive model's behavior into a small, cheap one. A more
   "traditional" fine-tuned classifier (DeBERTa, no generative ability) was trained the
   same way as a point of comparison, to check whether the flexibility of a generative
   model was actually buying anything (it was — see the table above and
   [`WRITEUP.md`](WRITEUP.md) §5).
3. **Grade everything against a small, hand-checked answer key** — not NHTSA's raw
   field, and not the teacher's own labels (that would just be checking the teacher
   agrees with itself). 551 complaints were read and categorized by hand against a
   written rulebook, specifically including examples of the rarest, most-often-
   mislabeled categories, so the reported accuracy numbers reflect *actual* correctness,
   not just agreement with a noisy source.

The teacher-vs-encoder-vs-generative-LLM comparison above is the interesting
methodology, but the headline is simpler: **a small model, fine-tuned once, can do this
specific job about as well as an expensive frontier model asked fresh every time — at a
fraction of the ongoing cost.** Full methodology, every metric, and the error analysis
behind each claim is in [`WRITEUP.md`](WRITEUP.md).

## Try it

`scripts/predict.py` loads the trained model and classifies a complaint narrative into
one of 40 component categories — CPU-only, no GPU or hosted API needed.

```bash
python scripts/predict.py --demo
# or classify your own narrative:
python scripts/predict.py "The brake pedal went to the floor..." --make Toyota --model Camry --year 2019
```

Real output from `python scripts/predict.py --demo` (CPU, ~10s including model load):

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

(This demo runs the smaller classifier checkpoint since it's practical on CPU with no
setup — see [`WRITEUP.md`](WRITEUP.md) §6.2 for how the fine-tuned LLM would be served.)

## Repo layout

```
scripts/            All pipeline code — data prep, labeling, training, evaluation
output/             Eval results, gold sets, label schema, run summaries (JSON)
data/               Training data (parquet) — regenerated by scripts, not fully versioned
checkpoints/        Trained model configs/adapters (weight binaries gitignored — see below)
PROJECT_PLAN.md     Original plan/rationale for the three-way comparison
WRITEUP.md          Full writeup: methodology, every metric, error analysis, decision log
```

### Pipeline, in order

1. **Data prep** — `scripts/load_data.py` (raw NHTSA flat file → parquet) →
   `scripts/prepare_training_data.py` (clean, scope to `PROD_TYPE='V'`, dedupe).
2. **Label schema + gold set** — `scripts/schema.py`, `output/label_schema.json`,
   `output/NHTSA_CODING_GOTCHAS.md`. The hand-checked answer key was built/expanded via
   `scripts/build_gold_eval_set*.py` and `scripts/build_rare_gold_expansion.py`
   (see `WRITEUP.md` §2.2 for that methodology).
3. **Teacher labeling** — `scripts/teacher_prompt.py` (prompt/schema definitions) +
   `scripts/run_batch_cascade_labeling.py` (the cheap-model-first, escalate-if-uncertain
   cascade, run as a batch job over ~62K complaints + a targeted rare-category top-up
   via `scripts/build_training_sample_rare_topup.py`). Cost modeling in
   `scripts/pricing.py` and `scripts/analyze_confidence_cascade.py`.
4. **Teacher eval** — `scripts/eval_models_on_gold.py` scores any zero-shot LLM against
   the hand-checked answer key.
5. **Small classifier fine-tune** — `scripts/train_deberta.py` (DeBERTa-v3-base, GPU
   recommended). Eval a saved checkpoint with `scripts/eval_checkpoint_deberta.py`.
6. **Small LLM fine-tune (the deliverable)** — `scripts/train_llm_lora.py`
   (Qwen2.5-7B-Instruct + LoRA/QLoRA, GPU required). Eval a saved checkpoint with
   `scripts/eval_checkpoint_llm.py`.
7. **Novel-category flexibility probe** — `scripts/novel_category_probe.py` (shared
   synthetic narratives) checks whether either fine-tuned model still generalizes to
   categories it saw little/no training data for — see `WRITEUP.md` §4.1.

Every eval above runs against the held-out hand-checked answer key the models never saw
in training — adjudicated against a written rulebook (`scripts/teacher_prompt.py`), not
just NHTSA's raw, noisy field (see `WRITEUP.md` §2.1–2.2 for how much that adjudication
actually changes the numbers).

## Reproducing

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY / OPENAI_API_KEY as needed
python scripts/load_data.py
python scripts/prepare_training_data.py
# ... see WRITEUP.md for the full sequence + expected runtimes/costs at each stage
```

The small classifier trains in a few hours on a single L4-class GPU; the fine-tuned LLM
takes roughly a day on an H100 (or longer on smaller cards) — see `WRITEUP.md` §4 for
the runtime breakdown and the prompt-shortening optimization that got it there.

### On model weights

`checkpoints/` tracks each trained run's small config/tokenizer/adapter-config/README
files (so the exact hyperparameters of every run are visible in the repo) but not the
multi-hundred-MB weight binaries themselves (`*.safetensors`, `*.bin`) — those are
regenerated by re-running the training scripts above against a rented GPU.

## License

See [`LICENSE`](LICENSE).
