# NHTSA Complaint Classifier — Production Brief

**For the coding agent.** This is both the brief (what and why) and the build
plan (order, guardrails, checkpoints). Repo:
`C:\Users\willf\git_proj\nhtsa-complaint-llm` (GitHub:
`willfreeman1/nhtsa-complaint-llm`). Copy this file into `docs/` when you
start.

You are expected to think, not just type. This brief is a strong starting
position, not a spec to execute literally. When data or results contradict
an assumption here, say so with the numbers, propose a change, and log it.
Priorities in order: **honest numbers > a system that runs unattended >
feature breadth.**

---

## 1. What exists today

A finished research project, documented in `README.md` and `WRITEUP.md`.
Read both before starting.

- **Task:** classify a vehicle complaint narrative (plus make/model/year)
  into one of **40 component categories** (`scripts/teacher_prompt.py`,
  `output/label_schema.json`).
- **Labels:** NHTSA's own component field is unreliable. On the ~500
  randomly sampled complaints in the answer key, NHTSA's label held up only
  **67%** of the time, and for rare categories about 30%. So training labels
  came from a Claude Haiku→Sonnet **confidence cascade**
  (`scripts/run_batch_cascade_labeling.py`) over ~68k complaints, roughly
  $126 per 100k complaints.
- **Models** (weights are on disk locally; binaries are git-ignored):
  - `checkpoints/llm_rung3_full_62k_plus_rare/`: Qwen2.5-7B-Instruct LoRA
    adapter (165 MB) over the base model. **86.2%** on the answer key.
  - `checkpoints/deberta_rung2_full_62k_plus_rare/final/`: DeBERTa-v3-base
    classifier (712 MB), runs on CPU (~5 rows/s on one core). **82.2%**.
  - The teacher (Sonnet) scores 86.6%.
- **Answer key:** `output/gold_eval_set_v3.json`, 551 hand-checked
  complaints. Each row has an `accept` list: a prediction is correct if it's
  in the list, and ambiguous rows accept more than one label. There's an
  easy tier (teacher and NHTSA agreed) and a hard tier (a person read the
  narrative to decide). The eval scripts are `scripts/eval_checkpoint_llm.py`
  and `scripts/eval_checkpoint_deberta.py`.
- **Data:** NHTSA flat file (`data/FLAT_CMPL.zip`, downloaded 2026-08-28,
  complaints through **2026-08-26**), cleaned by `scripts/load_data.py` and
  `scripts/prepare_training_data.py`. About **9–10k vehicle complaints a
  month (~300/day)**.

## 2. Why productionize: the finding that makes this real ML work

The training sample and answer key were drawn **evenly across 1995–2026**.
But the mix of complaints has changed a lot. Teacher labels on the 62k
random training sample:

| Share of complaints | 2015–19 | 2020–22 | 2025–26 |
|---|---|---|---|
| Forward collision avoidance | 0.9% | 8.5% | 11.8% |
| Back-over prevention | 0.5% | 4.1% | 13.3% |
| Lane departure | 0.3% | 3.0% | 4.1% |

NHTSA's own field shows the same rise. Driver-assist complaints went from
near zero to roughly 30% of current volume. Meanwhile:
- Only **82 of 551** answer-key rows are from 2023 or later.
- The driver-assist categories have **13** answer-key examples in total.
- For 2025–26, NHTSA labels 14.7% of complaints as lane departure while the
  teacher labels 4.1%. That disagreement is worth understanding.

So **the 86.2% is a fair score on the answer key's mix of complaints, but
nobody has measured accuracy on what arrives today.** The project's central
question:

> Does the model hold up on today's complaints, how would I know when it
> stops, and what do I do then?

That question gives the infrastructure a real job: the model registry,
evaluation runs, monitoring, retraining and the promotion gate all exist to
answer it.

**Resume gaps this closes:** production/MLOps (containerized serving,
scheduled pipelines, CI/CD, MLflow tracking and registry, monitoring,
retraining) and explainability (confidence, alternatives and evidence on
every prediction). It doesn't try to cover a named agent framework.

## 3. Two measurement facts to design around

1. **The teacher can't grade the student.** Sonnet scores 86.6% on the
   answer key and Qwen scores 86.2%. When they disagree, there's no way to
   tell which one is wrong. **Teacher–student disagreement is a drift alarm,
   never an accuracy number.** The earlier plan made an LLM judge the main
   quality score; that is superseded.
2. **Accuracy on new data comes only from human-checked complaints.** Small
   samples mean wide error bars. Always report accuracy with a confidence
   interval (e.g., Wilson), and size samples with that in mind (n=50 gives
   roughly ±12 points; n=300 gives roughly ±4).

---

## 4. Non-negotiables

Changing any of these needs Will's explicit sign-off and an entry in
`DECISIONS.md`.

1. **Hand-checked sets are labeled blind.** Will assigns labels without
   seeing any model's or the teacher's prediction. Only after that first
   pass are predictions revealed. Any label changed afterward is logged
   separately (row, before, after, why), and the first-pass accuracy is kept
   too. (In a previous project, 34 of 40 human verdicts matched pre-filled
   model suggestions. Showing suggestions first makes the labels partly the
   model's.)
2. **No hand-checked complaint ever enters training data.** An automated
   check (in CI) fails if any answer-key ID appears in any training set.
3. **Promotion gate, written down before the first retrain:** a challenger
   is promoted only if it (a) beats the champion on the current-complaints
   answer key and (b) doesn't drop more than a set tolerance on the legacy
   answer key (overall and hard tier). Tolerances go in `THRESHOLDS.md`
   first.
4. **Teacher disagreement and NHTSA agreement are never reported as
   accuracy.** They're drift signals.
5. **The historical replay is always called a replay** (see Stage 3). No
   model in the replay sees labels dated after its as-of date.
6. **Every reported number traces to an MLflow run.**
7. **Training and serving share preprocessing code.** One cleaning/prompt
   module is imported by both; a test checks this.
8. **No secrets or paid API calls in CI.** CI uses recorded fixtures or the
   CPU model.
9. **Existing research results stay reproducible.** Don't break the current
   scripts or change the published README numbers. The production work is
   added alongside them.

## 5. Yours to decide (propose, log, proceed)

- Repo layout (suggested: a `prod/` package plus `docker/`, leaving
  `scripts/` intact).
- Orchestration (cron, Prefect, GitHub Actions scheduled workflows, Cloud
  Run Jobs).
- **Where it runs.** Will's laptop is only on part of the time. For "runs
  unattended," a small cloud VM or scheduled cloud job is more credible.
  Propose options with monthly cost.
- Quantization method, serving runtime (llama.cpp, vLLM, transformers),
  database schema, drift statistics, dashboard framework, and where weights
  live (e.g., the Hugging Face Hub).
- Sample sizes within the ranges given here.

## 6. Working practices

- **`DECISIONS.md`**: date, decision, options, why, what would reverse it.
- **`THRESHOLDS.md`**: every go/adjust, alert and promotion threshold,
  committed **before** the run it applies to.
- **Cost logging** from the first API call or GPU rental. Ask Will before
  spending more than ~$25 in one go. Target running cost: under ~$30/month.
- Small commits with explanatory messages.

---

## Stage 0 — Feasibility spike (2–4 days)

**Goal:** answer the questions that decide the design, cheaply. Notebooks
are fine.

1. **Reproduce.** Load both models locally and rerun the existing evals on
   `gold_eval_set_v3.json`. Confirm 86.2% and 82.2% (or explain any
   difference).
2. **Serving options.** Quantize Qwen (e.g., GGUF Q8 and Q4, or merge + AWQ)
   and measure, for each variant and for DeBERTa: answer-key accuracy, rows
   per second on CPU, memory, and the GPU option's cost. This table decides
   the champion and is the first real MLflow comparison.
3. **Ingestion facts.** Download the current flat file and compare it with
   the 2026-08-28 copy:
   - How often does it update?
   - How many new complaints since 2026-08-26?
   - **Are existing rows ever revised,** especially the component field?
   - Is there a lighter source than the full 370 MB zip (e.g., NHTSA's API)?
4. **Current-complaints answer key, first pass.** Randomly sample complaints
   filed **after 2026-08-26**; none of them can have been in training. Random
   sampling is what makes the accuracy estimate apply to live traffic.
   - Build a blind labeling tool: narrative, make/model/year and the class
     definitions only.
   - Will labels **150** in the same `accept`-list format as v3, then
     predictions are revealed and disagreements re-reviewed under
     Non-negotiable 1.
   - Target is 300 by the end of Stage 1.
   - Optionally, a small separate oversample of driver-assist complaints,
     kept as its own set.
5. **Score everything on it.** DeBERTa, Qwen variants and the teacher. Also
   slice the legacy answer key by year.
6. **Feasibility memo** (`docs/feasibility_memo.md`): the tables above;
   current vs. legacy accuracy with confidence intervals; ingestion facts;
   the proposed champion and hosting plan; surprises.

**No outcome here kills the project; it sets the story:**
- Accuracy holds within the thresholds: the story is "monitoring confirmed
  stability," and retraining is shown through the replay.
- Accuracy drops: the first retrain on recent data is Stage 3's real job.
- Qwen can't be served within budget: DeBERTa becomes champion, Qwen stays
  the offline benchmark, and the write-up says why.

⛳ **Checkpoint 0.** Bring: memo, `THRESHOLDS.md`, `DECISIONS.md`, the
serving table, the current-key file with the first-pass vs. revised log,
and spend.

## Stage 1 — Foundation (week 1)

1. **Docker Compose:** Postgres, MLflow tracking server (with an artifact
   volume), the scoring job, the API, the dashboard.
2. **MLflow registry:** register the existing models as version 1 (Qwen
   adapter, the chosen quantized variant, DeBERTa). Set champion/challenger
   aliases. Record data and label versions (which teacher-label file, which
   answer-key version) with each.
3. **Evaluation harness:** one command evaluates any registered version on
   any answer-key set and logs overall/hard/easy accuracy, macro-F1,
   per-class results, confidence intervals, latency and cost to MLflow.
   Reuse the existing eval code.
4. **CI (GitHub Actions):**
   - lint and unit tests
   - the answer-key leakage check
   - the shared-preprocessing test
   - a fast evaluation on a fixture subset

   Any PR touching model, prompt or preprocessing code must show its
   evaluation change.
5. Finish the current-complaints answer key to ~300; freeze it as
   `gold_current_v1`.

⛳ **Checkpoint 1.** Bring: the MLflow UI with registered versions and
their evaluations on both keys, CI passing, the frozen current key.

## Stage 2 — Daily pipeline and monitoring (week 2)

1. **Ingestion job:** fetch new complaints since the last run, clean them
   with the shared code, store them.
2. **Scoring job:** the champion scores new rows. Store label, confidence,
   top-3 alternatives, model version and timestamp.
3. **Monitoring**, logged daily/weekly to Postgres and MLflow:
   - predicted-category mix vs. a reference window (e.g., PSI or
     Jensen–Shannon)
   - UNKNOWN OR OTHER rate
   - confidence distribution
   - agreement with NHTSA's field, as a trend (flat near 42% since 2015, so
     it only catches sharp changes)
   - weekly teacher disagreement on ~100 sampled complaints (cents per week)
4. **Monthly blind hand check** of ~50 random new complaints → a rolling
   live-accuracy estimate with confidence intervals. The tooling from
   Stage 0 makes this ~1 hour a month for Will.
5. **Alerts:** threshold crossings (from `THRESHOLDS.md`) write an alert row
   and show a dashboard banner; email is optional.

⛳ **Checkpoint 2** after ~2 weeks of unattended runs. Bring: uptime/failure
log, monitoring charts, alerts fired (or not), spend.

## Stage 3 — Retraining and the historical replay (weeks 2–3)

1. **Retraining pipeline:**
   1. Sample recent complaints.
   2. Label them with the existing teacher cascade.
   3. Exclude all answer-key IDs.
   4. Train a challenger.
   5. Evaluate it on both keys.
   6. Apply the promotion gate.
   7. Promote by moving the alias. Rollback is moving it back.

   DeBERTa retrains cheaply; Qwen LoRA needs a rented GPU (the last run took
   under 7 hours on an H100). Document the command and cost.
2. **A real modeling question, tracked as MLflow experiments:** what
   training mix fixes the recent categories without hurting old ones? Full
   history plus recent data, a recency window, or time-weighting. Compare
   on both keys and by class.
3. **Historical replay** (labeled a replay everywhere):
   1. Train an **as-of-2019 model** (DeBERTa for cost) using only existing
      teacher labels dated through 2019-12-31.
   2. Run 2020-01 → 2026-08 through the monitoring month by month. Sample,
      e.g., ~2k complaints a month to limit compute.
   3. Record when each trigger fires.
   4. At each trigger, retrain using only labels dated up to that month, and
      track the recovery.

   Accuracy during the replay is measured against teacher labels as a
   **proxy** (say so), plus answer-key rows from that period where they
   exist. Expected: the driver-assist rise trips the triggers. If it
   doesn't, that's a finding about the monitoring, and worth reporting.
4. If Stage 0 or 2 showed a real accuracy gap, run the first **live**
   retrain through the gate.

⛳ **Checkpoint 3 — before promoting any retrained model.** Bring: the
challenger vs. champion on both keys, the gate calculation, replay results,
spend.

## Stage 4 — Interface and explainability (week 3)

1. **FastAPI:**
   - `POST /classify` (narrative, make, model, year) → label, confidence,
     top-3 alternatives, model version, and **the most similar
     already-labeled complaints with their labels** as supporting evidence
     (nearest neighbors over an embedding index of the labeled pool)
   - `GET /health`
   - `GET /model` (current champion and its evaluation summary)
2. **Dashboard:**
   - classify a complaint, with the evidence above
   - monitoring charts and alerts, with live accuracy and its intervals
   - model history: versions, evaluations, promotions
   - **defect trends:** complaint counts by make/model/component over time,
     flagging unusual increases. Use a minimum count and a proper rate test,
     and account for how many series are being scanned, so it doesn't flag
     noise.

     This page is the one outsiders (auto-safety journalists, safety
     researchers) might actually use. Spikes are statistical flags, not
     defect findings, and the page says so.
3. Optional: host the dashboard publicly. Scoring stays batch.
4. Optional, only if time remains: train a retrained Qwen version to output
   the label plus the sentence from the narrative that supports it (the
   teacher cascade can supply those). Evaluate whether the quoted text is
   real and relevant.

## Stage 5 — Write-up (end of week 3)

A `PRODUCTION.md` linked from the README (leave the research section
intact):
- architecture diagram
- serving tradeoff table
- **current vs. legacy accuracy, with intervals** (the headline)
- the drift story and what the monitoring saw
- replay results
- retraining and promotion history
- monthly running cost
- limitations, stated plainly: small hand-checked samples, teacher-proxy
  accuracy in the replay, 40-class schema limits, NHTSA field noise

⛳ **Checkpoint 4 — review before publishing.**

---

## Cut order if time runs short
1. Optional Stage 4 items.
2. Public hosting.
3. Qwen retraining (keep DeBERTa retraining).
4. The replay reduced to fewer months or a smaller sample.

**Never cut:**
- blind labeling
- the current-complaints answer key
- the promotion gate
- CI with the leakage check
- confidence intervals on accuracy
- unattended daily runs
