# Project plan: NHTSA complaint classification — from feasibility to fine-tune

Feasibility verdict (see `output/FINAL_REPORT.md`): **green-light.** This doc picks up
from there and lays out the build plan, including a three-way model comparison that
became part of the plan during the feasibility discussion.

## The three-way comparison

Rather than jumping straight to "fine-tune an LLM," build up through three rungs so the
final writeup can show exactly where each increment of cost/complexity buys accuracy:

| Rung | Approach | Cost/complexity | Purpose |
|---|---|---|---|
| 1 | Zero-shot frontier LLM (properly prompted: label schema + few-shot examples per class, unlike the deliberately-blind Step 5 probe) | Cheap, no training | Establishes the ceiling a fine-tune should try to approach/beat cheaply |
| 2 | Fine-tuned encoder classifier (e.g. DeBERTa-v3-base, full backprop fine-tune, fixed softmax head over ~30 classes) | Needs GPU, but far cheaper/faster to train than an LLM | Tests how much of the LLM's advantage is "task-specific gradient adaptation to NHTSA's labels" vs. "general reasoning/world knowledge" |
| 3 | Fine-tuned LLM (LoRA/QLoRA on an open-weight model, distilling from the Rung-1 teacher's outputs, or directly on NHTSA gold) | Needs GPU (rented), the real deliverable | The actual portfolio piece — demonstrates fine-tuning/distillation skill, and should ideally beat Rung 2 while retaining more flexibility (see below) |

This was prompted by two side-experiments during feasibility: TF-IDF (58.3% acc /
0.444 macro-F1) and a frozen `all-MiniLM-L6-v2` embedding + LightGBM (55.7% acc / 0.419
macro-F1) both plateaued well below the zero-shot LLM's ~52-78% range (exact vs.
taxonomy-collapsed) — showing that *frozen* representations, however you generate them,
leave real headroom that only task-adapted (i.e. fine-tuned) models seem to close.
Rung 2 tests whether that headroom needs an LLM specifically, or just needs *any*
model that gets to see NHTSA's actual labeling decisions during training.

## Why Rung 3 (LLM) over Rung 2 (encoder) as the actual deliverable

Even if Rung 2 performs comparably to Rung 3 on today's fixed ~30-class label set, the
LLM approach has a structural flexibility advantage worth calling out explicitly in the
writeup:

- An encoder classifier ends in a fixed-size softmax head (K neurons, set at training
  time). It **cannot** emit a class it has no neuron for — adding a genuinely new
  category (e.g. a new ADAS/battery-thermal-event category NHTSA hasn't used before)
  requires expanding and retraining the head.
- A fine-tuned LLM used generatively has no such ceiling — a new category is just a new
  string, and with a well-chosen prompt (updated label list + a few examples) it can
  often produce a reasonable answer with zero retraining, falling back on the base
  model's general world knowledge.
- **Caveat to build in and test for**: aggressive fine-tuning can narrow a generative
  model's willingness to venture outside the categories it saw a lot of during training
  (a mode-collapse-toward-training-distribution risk). Mitigate with (a) light-touch
  LoRA (modest rank, moderate steps, don't over-train), and (b) deliberately including
  a handful of "doesn't cleanly fit any known category" examples in the fine-tuning set
  so the model retains permission to punt to free text / `UNKNOWN OR OTHER` rather than
  force-fitting. Test this explicitly post-training: hold out a few real or synthetic
  "novel category" cases and confirm the fine-tuned LLM doesn't blindly force-fit them.

## Build order

1. **Data prep** (extends feasibility scripts): scope to `PROD_TYPE='V'`, dedupe/flag by
   `ODINO` for CRASH/INJURED/DEATHS, collapse near-duplicate top-level `COMPDESC`
   categories per `output/NHTSA_CODING_GOTCHAS.md` §B3, settle on ~30 classes + `OTHER`.
2. **Teacher labeling (Rung 1)**: prompt a frontier LLM with the cleaned label schema +
   few-shot examples (informed by every gotcha in that doc — especially B1/B2 on fire
   mislabeling and B4-B6 on category overlap) over a large sample to generate
   high-quality pseudo-labels / distillation targets, and to re-establish the "proper"
   zero-shot ceiling number (expect meaningfully above the ad-hoc 52-78% found in the
   blind Step 5 probe).
3. **Rung 2 — encoder fine-tune**: standard classification fine-tune (DeBERTa-v3-base
   or similar), eval against the same held-out set as Rung 1/3.
4. **Rung 3 — LLM fine-tune**: LoRA/QLoRA on an open-weight model (e.g. Llama/Qwen
   7-8B class), trained on teacher-labeled data (distillation) with the novel-category
   safety examples mixed in; eval + novel-category flexibility test.
5. **Writeup**: three-way comparison table (accuracy/macro-F1 + cost + flexibility),
   explicit callback to every decision criterion and gotcha found during feasibility.

## Open decisions to revisit once Rung 1/2 numbers are in
- Exact final class list (≥30 vs. merge further)
- Whether to distill purely from the teacher LLM's labels, train on NHTSA gold directly,
  or blend both (teacher labels to fix known NHTSA quirks like the FIRERELATED trap,
  NHTSA gold to stay grounded)
- Student model size/family, chosen partly based on what's practical on rented GPU time
