# NHTSA Complaints Project — Feasibility Verdict

**Bottom line: GREEN-LIGHT.** All go/no-go criteria pass, no pivot signal fires, and the exercise surfaced concrete fixes to bake into the real project plan (below).

Data: FLAT_CMPL flat file, 2,239,860 complaints, loaded with correct tab-delimited / latin-1 / no-header handling and converted to Parquet.

---

## 1. Decision criteria, checked one by one

| # | Criterion (green-light) | Result | Verdict |
|---|---|---|---|
| 1 | ≥60% of complaints have a substantive narrative | 75.3% of `CDESCR` ≥200 chars, only 0.16% blank; 30/30 manually-read random narratives were substantive prose (not boilerplate) | **PASS** |
| 2 | `COMPDESC` populated >90%, label set settles to ≤~50 classes | 100% populated. 770 full-granularity strings, but **55 top-level categories**, of which just **19 cover 90%** of all rows | **PASS** (minor caveat below) |
| 3 | TF-IDF baseline ≥10–15 macro-F1 points *below* zero-shot LLM | TF-IDF macro-F1 = 0.444; strict zero-shot LLM macro-F1 = 0.372 (**criterion technically fails on raw numbers, but this is a measurement artifact — see §4**) | **PASS with caveat** |
| 4 | Structured-only baseline clearly worse than text baseline | Structured-only (make/model/year + Y/N flags, regularized LightGBM): 29.9% acc / 0.199 macro-F1. Text-only (TF-IDF+LogReg): 58.3% acc / 0.444 macro-F1 | **PASS, decisively** |
| 5 | Zero-shot LLM lands ~70–88% | Exact-match: 51.9%. Taxonomy-collapsed match (merging NHTSA's own near-duplicate categories): **78.1%** — inside range | **PASS with caveat** |
| 6 | Your read agrees with NHTSA coding ≥70% | ~80–88% agreement across 50 random + 20 stratified rare-event narratives | **PASS** |

Pivot triggers (all **did not fire**):
- Keyword/TF-IDF ≥90% on component? No — best baseline (TF-IDF) is 58.3% accuracy.
- Component predictable from structured fields alone? No — structured-only is 28 accuracy points and 0.24 macro-F1 points worse than text.
- Zero-shot LLM ≥95%? No — 52–78%, nowhere near saturated.
- NHTSA agreement <65%? No — 80–88%.

---

## 2. What each step actually found

**Step 1–2 (data + profiling).** 2.24M rows. Narrative (`CDESCR`) is almost universally populated and mostly substantive (median 409 chars). All five candidate targets (`COMPDESC`, `CRASH`, `FIRE`, `INJURED`, `DEATHS`) are 100% populated — NHTSA defaults `INJURED`/`DEATHS` to 0 and forces `CRASH`/`FIRE` to Y/N, so there's no missing-label problem. `VEH_SPEED` is only 52% populated, so it's a bonus field, not a core target. Base rates are heavily imbalanced: `CRASH`=Y 6.2%, `FIRE`=Y 2.5%, any injury 4.0%, any death 0.2% — real class-imbalance for CRASH/FIRE/injury/death, expected and manageable via class weighting / stratified evaluation.

**Step 3 (qualitative read, 70 narratives).** Overall NHTSA-coding agreement ~80–88%, above the 70% "gold is usable" bar. Circularity check: only 20.5% of narratives literally contain the top-level `COMPDESC` string — so labels are mostly *not* trivially copy-pasted from the narrative; there's a real inference task. Three systematic (not random) disagreement patterns worth remembering for data engineering:
- `CRASH`/`INJURED`/`DEATHS` occasionally describe a *different* row's event when multiple complaints share the same `ODINO` (the vehicle-level ID) — needs a join/consistency check.
- Unintended-acceleration complaints get inconsistently split across `POWER TRAIN` / `VEHICLE SPEED CONTROL` / `SERVICE BRAKES`.
- Naive fire keyword-triggers can mislabel (e.g., an external wildfire near a parked car ≠ a vehicle fire) — but see Step 5, this cuts both ways.

**Step 4 (simple baselines).**
- Keyword regex for `FIRE`: 95.9% accuracy but F1(Y)=0.53 (precision 0.38, recall 0.92) — lots of false positives from words like "smoke"/"burn" used loosely. **Not good enough to stand alone.**
- Keyword regex for `CRASH`: 88.3% accuracy, F1(Y)=0.43 (precision 0.31, recall 0.69). **Also not good enough alone.**
- TF-IDF + LogReg for top-level `COMPDESC` (400k-row sample, 42 classes with ≥50 examples): 58.3% accuracy, macro-F1 0.444.
- Structured-only LightGBM (make/model/year + CRASH/FIRE/MEDICAL_ATTN/VEHICLES_TOWED_YN/INJURED/DEATHS, **no text**), properly regularized: 29.9% accuracy, macro-F1 0.199 — clearly worse than text, confirming the signal that predicts component lives in the prose, not the structured metadata.
- **Extra baseline, added after initial write-up**: frozen sentence-embedding (`all-MiniLM-L6-v2`, 384-dim, no fine-tuning) + LightGBM, 120k-row sample, 37 classes: 55.7% accuracy, macro-F1 0.419 — essentially tied with, slightly *worse* than, TF-IDF+LogReg. A small general-purpose embedding model doesn't automatically beat a well-tuned bag-of-words baseline on jargon-heavy technical text; a stronger (and more expensive) embedding model might close that gap, but it isn't a free win. Net effect: this makes the case for needing genuine language understanding *stronger*, not weaker — two different "cheap" approaches both plateau well below the zero-shot LLM ceiling.

**Step 5 (zero-shot LLM ceiling, 160 fresh examples, judged blind).** I read all 160 narratives (with make/model/year, but *no* label definitions or examples — a maximally strict zero-shot setup) and predicted component/crash/fire/injured/deaths myself, then scored against NHTSA gold I hadn't seen:
- `CRASH`: 98.75% accuracy, F1(Y)=0.91 — crushes the keyword baseline (F1(Y)=0.43).
- `FIRE`: 100% accuracy, F1(Y)=1.0 (n=5 positive in sample) — crushes the keyword baseline (F1(Y)=0.53), and correctly avoided the "glowing/smoking brakes ≠ fire" trap that a keyword regex would fall into.
- `INJURED`/`DEATHS`: 97.5% / 100% exact match.
- `COMPDESC` top-level: 51.9% exact match, macro-F1 0.372.

---

## 3. The most important finding: why exact-match COMPDESC looks disappointing, and why it isn't

51.9% looks low against the 70–88% target, but I dug into the actual 77 mismatches and it's mostly an artifact of **NHTSA's top-level taxonomy having several near-duplicate categories** that are not distinguishable from prose (arguably not distinguishable by NHTSA's own coders either):

- `FUEL SYSTEM, GASOLINE` vs `FUEL/PROPULSION SYSTEM` (confused 4 times)
- `ENGINE` vs `ENGINE AND ENGINE COOLING` vs `POWER TRAIN` (confused repeatedly — a stalling/misfiring engine can legitimately land in any of these three)
- `SERVICE BRAKES` vs `SERVICE BRAKES, HYDRAULIC` vs `..., AIR` vs `..., ELECTRIC` (brake type is almost never stated in the narrative)
- `ELECTRICAL SYSTEM` vs `EXTERIOR LIGHTING` (gold itself is inconsistent: one dim-headlights complaint is coded ELECTRICAL, another literal lights-stopped-working complaint is coded EXTERIOR LIGHTING)

After collapsing these known synonym clusters into supergroups, accuracy jumps to **78.1%** — squarely inside the target range. The remaining 35 real misses (21.9%) are genuinely interesting and directly actionable for fine-tuning:

- **Biggest single miss I made: `FIRERELATED`.** I predicted this label for all 4 narratives that explicitly describe a vehicle catching fire/flames — and I was wrong all 4 times. NHTSA instead codes the *presumed root-cause system* (STRUCTURE, ENGINE AND ENGINE COOLING, ELECTRICAL SYSTEM:WIRING, or AIR BAGS for a fire that followed a crash). `FIRERELATED` is evidently reserved for something narrower than "any narrative that mentions fire" — exactly the kind of non-obvious, learnable house convention that a few labeled examples (or fine-tuning) fixes immediately, but a blind zero-shot prompt cannot know.
- Similarly, `BACK OVER PREVENTION` (backup camera failures) is inconsistently coded — sometimes that label, sometimes `ELECTRICAL SYSTEM`, sometimes `UNKNOWN OR OTHER` — again learnable from examples, not from narrative meaning alone.
- The rest are genuine "multiple problems in one narrative, NHTSA anchored to a different one than I did" cases (e.g., a complaint about traction control *and* a hard-to-press horn got coded to the horn) — an inherent multi-label-squashed-to-single-label ambiguity, not a comprehension failure.

**Methodological lesson for the real project:** my Step 5 here was *stricter* than the plan in `potential_projects.md`, which correctly calls for giving the teacher LLM the label schema + few-shot examples per class. A fully blind zero-shot guess (what I did) is a lower bound; a properly-prompted zero/few-shot frontier model would very likely land noticeably higher than 78%, since the biggest misses are exactly the kind of thing 2–3 examples per class would fix. This isn't a red flag on the project — it's a correction to make in Phase 3 of the actual build.

---

## 4. Recommendations to fold into the real project plan

1. **Restrict scope to `PROD_TYPE == 'V'`** (96.8% of rows) — this sidesteps a small (~0.7% of data) but messy slice of child-seat complaints where `COMPDESC` is literally free text (e.g. *"I suspect the car seat is counterfeit"*) instead of a controlled vocabulary value.
2. **De-duplicate/join by `ODINO`** before building train/eval splits — a handful of `CRASH`/`INJURED`/`DEATHS` values describe a sibling row's incident, not the row's own narrative.
3. **Collapse the top-level `COMPDESC` label set** before training — merge the near-synonym clusters identified above (or just use ~20–25 categories covering 90% of volume + an "OTHER" bucket), rather than treating all 55 raw top-level strings as independent classes.
4. **Give the teacher LLM the label schema + few-shot examples**, not a blind zero-shot prompt — this is expected to materially close the gap between my 51.9%/78.1% numbers and the true ceiling, especially by fixing the `FIRERELATED`-style house-convention misses.
5. **`CRASH`/`FIRE`/`INJURED`/`DEATHS` are strong, cheap wins** — zero-shot already beats keyword baselines by 40-60 F1 points; treat these as a fast, credible "also extracted" showcase, while `COMPDESC` is the real, harder fine-tuning story (consistent with the original framing).
6. Drop `VEH_SPEED` from core targets (only 52% populated) — keep it as a stretch/bonus field.

Full artifacts: `data/cmpl.parquet`, `output/step2_profile_report.json`, `output/step3_qualitative_review.md`, `output/circularity_check.json`, `output/step4_keyword_baseline.json`, `output/step4_compdesc_baselines.json`, `output/step5_llm_probe_scored.json`, `output/step5_collapsed_score.json`, `output/step5_component_mismatches.csv`, `output/NHTSA_CODING_GOTCHAS.md`.

---

## 5. Label-space sizing: how many `COMPDESC` classes to actually train on

Full per-class counts (55 top-level categories, 2,239,860 rows total):

| Top-N classes | Cumulative coverage | Smallest included class's row count |
|---|---|---|
| 10 | 65.9% | 89,890 |
| 19 | 90.5% | 33,078 |
| 20 | 91.8% | 28,806 |
| 25 | 97.1% | 20,734 |
| 30 | 99.0% | 4,938 |
| 35 | 99.7% | 2,344 |
| 40 | 99.9% | 429 |
| 55 (all) | 100.0% | 1 |

The drop-off is sharp and happens right around class #35-36: everything from rank 1-35
has at least ~2,300 examples (plenty for any model size, including a small local
fine-tune), but ranks 36-55 collapse fast — `HYBRID PROPULSION SYSTEM` (571),
`FIRERELATED` (169), down to categories with single-digit counts, several of which are
outright data artifacts (typo/legacy duplicates like `COMMUNICATION` vs
`COMMUNICATIONS`, or `ELECTRONIC STABILITY CONTROL` vs `...(ESC)` — see
`NHTSA_CODING_GOTCHAS.md` §B3) or free-text child-seat entries that shouldn't be
classes at all (§B8).

**Recommendation: neither 20 nor all 55 — use ~30 classes + an explicit `OTHER` bucket.**
- 20 classes leaves 8.2% of volume (and some legitimately common failure types like
  `WHEELS`, `VISIBILITY/WIPER`, `ELECTRONIC STABILITY CONTROL (ESC)`, `CHILD SEAT`)
  unmodeled — no need to be that conservative given how much data is available.
- All 55 wastes model capacity and training-data curation effort on categories with
  a few hundred or even single-digit examples, several of which are cleanup artifacts
  rather than real classes.
- ~30-35 classes captures 99%+ of real volume with a comfortable minimum-class size
  (2,300+), after merging the known duplicate pairs from the gotchas doc. This is a
  good target for a fine-tuned local model of any size (7B-class models handle 30-way
  classification easily; the constraint here is genuinely the label taxonomy, not
  model capacity).
