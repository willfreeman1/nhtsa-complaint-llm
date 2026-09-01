# NHTSA Coding Gotchas — for teacher-model prompt engineering

Every item below is a systematic (not random) pattern found while manually reading
narratives against NHTSA's coded fields (Step 3, 70 narratives) and while doing the
blind zero-shot probe (Step 5, 160 narratives). These are exactly the things a teacher
LLM prompt needs to either (a) explain as a labeling convention, or (b) supply few-shot
examples for, so the teacher doesn't repeat my mistakes.

---

## A. Field-level gotchas: CRASH / INJURED / DEATHS

**A1. These fields can describe a *different* row's incident, not this row's narrative.**
NHTSA assigns one row per affected *component* per incident, but `CRASH`/`INJURED`/
`DEATHS` appear to be set once at the incident level (keyed by `ODINO`) and then copied
onto every component row for that incident — even rows whose own `CDESCR` text doesn't
mention a crash or injury at all.
- Example: CMPLID 384993 (tire tread separation/blowout) is coded `CRASH=Y, INJURED=2`,
  but its narrative only describes the tire failure, no crash or injury.
- Example: CMPLID 791379 (Yamaha ATV fuel-tank design complaint) is coded `CRASH=Y` with
  a narrative that's purely a design gripe, no crash mentioned.
- **Fix for the pipeline**: de-duplicate/aggregate by `ODINO` before building train/eval
  sets for these three fields, or explicitly flag multi-row `ODINO`s as a known noise
  source rather than scoring every row independently.

## B. `COMPDESC` (component) gotchas

**B1. `FIRERELATED` is not "any narrative that describes a vehicle fire."**
This was my single biggest, most consistent error in the Step 5 probe: every one of the
4 fire narratives I read (flames from under the hood, engine catching fire while parked,
a crash followed by fire that destroyed the vehicle, etc.) was coded to the *presumed
root-cause system* instead — `STRUCTURE`, `ENGINE AND ENGINE COOLING`, `ELECTRICAL
SYSTEM:WIRING`, or `AIR BAGS` (for the crash-then-fire case). `FIRERELATED` is a tiny
category (169 of 2.24M rows total) — it is evidently reserved for something narrower
than "fire is mentioned." **Teacher-model instruction: do not default to FIRERELATED
just because the narrative mentions fire/flames — code the underlying system that
caused it, and only use FIRERELATED for [confirm exact convention via a NHTSA
data dictionary / recall database lookup before finalizing the prompt].**

**B2. "Smoke"/"glowing"/"burning smell" ≠ `FIRE=Y` and ≠ `FIRERELATED`.**
- Overheated/worn brakes can glow and smoke from friction with no actual flame
  (CMPLID from Step 5 sample: "wheel well was glowing and smoking" after riding the
  brakes hard — correctly not a fire).
- Airbag deployment produces visible "black smoke" as normal propellant/talc
  discharge, not a vehicle fire (Ford Contour case, Step 5 sample).
- An external wildfire near a broken-down vehicle, with fire retardant dropped nearby,
  got coded `FIRE=Y` even though the vehicle itself never caught fire (CMPLID 815740) —
  so NHTSA's own gold labels are not fully immune to this trap either; don't over-trust
  literal fire language in either direction.

**B3. Near-duplicate top-level categories exist — narrative alone often can't disambiguate.**
These pairs/clusters were confused repeatedly in the blind probe and likely reflect
genuine ambiguity in NHTSA's own taxonomy, not comprehension failure:
- `ENGINE` vs `ENGINE AND ENGINE COOLING` vs `POWER TRAIN` (a stalling/misfiring engine
  can legitimately land in any of the three)
- `FUEL SYSTEM, GASOLINE` vs `FUEL/PROPULSION SYSTEM` (confused 4/4 times I hit it)
- `SERVICE BRAKES` vs `SERVICE BRAKES, HYDRAULIC` vs `..., AIR` vs `..., ELECTRIC`
  (brake subtype is almost never stated in the narrative)
- `ELECTRICAL SYSTEM` vs `EXTERIOR LIGHTING` — and the gold itself is inconsistent
  here: one dim-headlights complaint is coded `ELECTRICAL SYSTEM`, another
  lights-stopped-working complaint (corrosion-damaged wiring) is coded
  `EXTERIOR LIGHTING`.
- `ELECTRONIC STABILITY CONTROL` vs `ELECTRONIC STABILITY CONTROL (ESC)` — literally
  two separate top-level strings for the same concept (335 vs 21,286 rows) — almost
  certainly a data-entry/legacy-naming artifact, should be merged before training.
- `COMMUNICATION` (110 rows) vs `COMMUNICATIONS` (1 row) — same, a typo duplicate.
- **Recommendation:** merge these clusters at the label-definition stage (both for the
  teacher prompt's label list and for training data) rather than treating all 55
  strings as independent, non-overlapping classes.

**B4. `BACK OVER PREVENTION` is applied inconsistently to backup-camera failures.**
Of 3 backup-camera-malfunction narratives in the Step 5 sample, one was coded
`BACK OVER PREVENTION` (correct guess), one was coded `ELECTRICAL SYSTEM`, and one was
coded `UNKNOWN OR OTHER`. No obvious narrative-level signal distinguishes them — this
looks like it depends on information not visible in `CDESCR` alone (e.g. whether the
issue was already tied to a known recall campaign).

**B5. Unintended-acceleration complaints split unpredictably across three categories.**
Near-identical fact patterns ("foot on brake, car revs/accelerates on its own") get
coded to `POWER TRAIN`, `VEHICLE SPEED CONTROL`, or `SERVICE BRAKES` depending on which
detail the coder anchored on. One case (CMPLID 862718, Camry) was coded
`VEHICLE SPEED CONTROL:ACCELERATOR PEDAL` even though its own narrative describes a
*brake* failure, not acceleration — likely coded using outside knowledge of the Toyota
UA investigation rather than the narrative text alone. **This failure mode has a real
human-judgment ceiling below 95%; don't expect a teacher/student model to do better
than a person would here.**

**B6. Multi-issue narratives get anchored to just one detail, not necessarily the most
prominent one.**
Many complaints describe 2+ simultaneous problems, but NHTSA assigns only one
`COMPDESC` per row. Examples: a Buick Terraza complaint mostly about traction-control
engaging unexpectedly was coded to `ELECTRICAL SYSTEM:HORN` (a minor detail mentioned
almost in passing); a Pontiac Grand Am complaint about dashboard/defroster/detached
airbag compartment was coded to the defroster, not the airbag issue. **This is an
inherent single-label simplification of sometimes multi-label reality — not a coding
error, but it puts a ceiling on any single-label classifier, including a fine-tuned
one, unless the anchoring logic can be learned from many examples.**

**B7. Granular (colon-suffixed) codes are sometimes over-specific relative to the narrative.**
E.g., CMPLID 97877 was coded down to `SERVICE BRAKES...:ANTILOCK/TRACTION
CONTROL/ELECTRONIC LIMITED SLIP` when the narrative only mentions generic hard
brake-pedal effort with no ABS-specific symptom. This is part of why the project should
use the **top-level rollup** (≤~30-35 categories, see label-space note below), not the
full 770-value granular string, as the target label set.

**B8. `COMPDESC` is not always a controlled vocabulary — child-seat complaints
(`PROD_TYPE='C'`) sometimes contain literal free text** (e.g. *"I suspect the car seat
is counterfeit"*, *"Other/I am not sure"*). This is a small slice (~0.7% of rows) but
will pollute a clean label space if not filtered. **Fix: scope the project to
`PROD_TYPE='V'` (vehicles, 96.8% of rows) to avoid this entirely.**

**B10. `SERVICE BRAKES, AIR` does not mean what its name implies.** Checked at scale
(4,520 vehicle rows): the make distribution is dominated by ordinary passenger
vehicles (Chevrolet, Ford, Dodge, GMC, Toyota, Honda, Jeep...), and only **1.7%** of
narratives in this bucket even mention "air brake" literally. In practice this
category is applied to the same kind of ordinary hydraulic/disc-brake complaints
(rotors, calipers, ABS units) as `SERVICE BRAKES` and `SERVICE BRAKES, HYDRAULIC` —
there is no narrative-text signal that distinguishes it. **Recommendation: treat as
functionally equivalent to `SERVICE BRAKES` for prompting purposes (don't expect or
reward "air brake" language), and consider merging it into `SERVICE BRAKES` at the
label-schema stage** the same way the ESC/COMMUNICATION typo-duplicates were merged —
unlike those, this isn't a spelling artifact, but it may be functionally unlearnable
as a separate class from text alone, which is the same practical problem.

**B11. `LANE DEPARTURE` also covers blind-spot monitoring.** All 3 few-shot
candidates sampled for this class were about blind-spot camera/detection failures,
not literal lane-departure-warning language — this looks like a genuine, consistent
NHTSA convention (grouping ADAS side-detection features under the same top-level
bucket as lane-keeping) rather than sampling noise. Teacher prompt should say so
explicitly rather than assuming the name is literal.

**B12. `VISIBILITY` vs `VISIBILITY/WIPER` is genuinely inconsistent in the raw gold
labels, in *both* directions — confirmed at n=40 on the teacher-vs-gold disagreement
sample.** A windshield that spontaneously cracked (no wiper involved) was coded
`VISIBILITY/WIPER`; two separate narratives about literal wiper failure ("wipers
become inoperative," "wipers stop working in heavy rain") were coded plain
`VISIBILITY`, not the wiper variant. This isn't a case where the teacher model needs a
better rule — NHTSA's own coding doesn't consistently distinguish these two strings, so
don't expect (or try to force) a clean split here.

**B13. Heater/HVAC complaints are inconsistently routed between `ENGINE AND ENGINE
COOLING` and `VISIBILITY/WIPER`.** Two narratives about a malfunctioning heater core
(one "won't release heat," one "blows only cool air, can't defrost windows") were
coded to *different* top-level categories — presumably because the second one
mentioned the defrost/defog consequence explicitly and the first one didn't — but the
underlying defect (heater core / HVAC blend-air-door hardware) is the same. Treat this
as inherent gold noise rather than a learnable convention.

**B14. `SUSPENSION` / `STEERING` / `POWER TRAIN` share a genuinely fuzzy boundary at
axles, tie rods, and wheel hubs.** Multiple narratives in the n=40 disagreement sample
landed on this exact seam: a wheel hub separating (`WHEELS` vs `SUSPENSION`), "death
wobble" fixed by tie-rod recalls (`STEERING` vs `SUSPENSION`), and a leaking front axle
(`POWER TRAIN` — our own definition explicitly lists driveline/axle — vs
`SUSPENSION`). These are physically adjacent/overlapping systems where a single part
touches both; expect real disagreement here independent of model quality.

**B15. Recall-campaign linkage really does drive some gold labels independent of the
narrative text — directly confirmed, not just inferred.** CMPLID 1981710 (Ford
EcoSport) literally states *"the failure was related to NHTSA Campaign Number:
23V905000 (Engine and Engine Cooling)"* — quoting the recall's own assigned category
by name — while the actual repair described in the same narrative ("wiring needed to
be replaced") reads as electrical. The gold label matches the recall's official
category, not the narrative's literal content. This confirms (not just plausibly
explains) several other teacher/gold disagreements in complaints that mention a recall
number with no other diagnostic detail: the correct label in those cases may live in
NHTSA's recall database, not in `CDESCR` at all, and no narrative-only model — teacher
or student — can be expected to recover it.

**B13-REVISED. The `VISIBILITY`/`EQUIPMENT`/`ENGINE AND ENGINE COOLING` heater/AC split is
*mostly* a learnable convention, not pure noise -- confirmed against raw `COMPDESC`, not
just narrative reading.** Checked at scale: HVAC complaints that explicitly mention a
defrost/defog/visibility consequence roll up to `VISIBILITY` (or the legacy
`VISIBILITY/WIPER` string -- see B12, that half is genuinely noisy); a failed heater
**core** specifically rolls to `ENGINE AND ENGINE COOLING:COOLING SYSTEM` because the
core physically sits in the coolant loop, regardless of whether defrost is mentioned;
and standalone AC-unit complaints (compressor, refrigerant, "blows warm") with no
defrost/visibility consequence roll to `EQUIPMENT:APPLIANCE:AIR CONDITIONER`. All three
readings are encoded in the class definitions now (see `teacher_prompt.py`). This
**corrects an earlier adjudication note** for CMPLID 463855 that had misread the raw
gold label as `EQUIPMENT` (checked directly against `cmpl_clean.parquet`: it's actually
`ENGINE AND ENGINE COOLING`) -- a reminder to verify against the raw `COMPDESC_LABEL`
column directly rather than trusting a prior note's summary.

**B16. Exhaust-system complaints roll up to `ENGINE AND ENGINE COOLING`, not a
separate class -- confirmed at scale (~11k rows: manifold/muffler/tailpipe, catalytic
converter, EGR valve).** There's no dedicated exhaust category in NHTSA's own
top-level rollup, so none was added to the teacher schema either; the
`ENGINE AND ENGINE COOLING` definition was extended to say so explicitly instead
(previously the definition only described the cooling system, not exhaust, which is
why 3 different models missed this on CMPLID 683472 despite the correct answer being
recoverable from NHTSA's own convention).

**B17. Recall Campaign Numbers quoted in the narrative should outrank the
complainant's own symptom-based diagnosis (extends B15 into an actionable prompt
rule).** B15 already showed the gold label sometimes matches a quoted recall
campaign's official category rather than the narrative's literal repair description.
CMPLID 1938571 (Ford F-450) is a second confirmed instance the model evaluation
surfaced: the narrative describes a failed "electronic brake control" and mentions no
electrical fault, but explicitly quotes "NHTSA Campaign Number: 22V193000 (Electrical
System)" -- and the true top-level `COMPDESC` (checked directly) is `ELECTRICAL
SYSTEM`. All three evaluated models (gpt-5-mini, gpt-5.4-mini, gpt-5.4-nano) missed
this in three different ways, none of them ELECTRICAL SYSTEM, because nothing in the
prompt told them a quoted campaign category should outweigh the symptom description.
**Fixed in `teacher_prompt.py` (new instruction rule 4).**

**B18. Crash narratives that list multiple collision-damaged parts can mislead a model
into classifying the crash damage instead of the alleged product defect.** CMPLID
452061 (Dodge Durango): the narrative lists a broken steering rack and pinion,
radiator, headlights, bumper, and transmission line as damage sustained *in the
crash*, then separately states "AIRBAGS DID NOT DEPLOYED!!" as the actual complaint
(confirmed: true `COMPDESC` is `AIR BAGS`). All three evaluated models answered
STEERING, apparently anchored on the most specific/technical-sounding damaged part
in the list rather than recognizing it as collision damage rather than the alleged
defect. **Fixed in `teacher_prompt.py` (new instruction rule 5)**: distinguish
impact-damaged parts from the defect actually being alleged (look for phrasing like
"did not deploy," "failed to activate").

**B19. Some narratives are inherently too garbled to resolve, and no prompt fix will
help.** CMPLID 1435252 (Jeep Renegade): "...THE VEHICLE STALLED WITHOUT WARNING...
DIAGNOSED AS A SAFETY PRECAUTION FOR THE VEHICLE WHEN THE OIL WAS LOW..." reads like a
mangled auto-summary that conflates a stall event with an unrelated
low-oil-safety-shutdown explanation. The true `COMPDESC` is plain `ENGINE`, but three
different models gave three different wrong answers (POWER TRAIN, ENGINE AND ENGINE
COOLING, UNKNOWN OR OTHER) on a genuinely confusing sentence. Counted as a real
human-judgment-ceiling case (same category as B5/C1), not a fixable prompt gap.

**B20. Non-vehicle product complaints (`PROD_TYPE != 'V'`) must be removed from the
dataset entirely, not mapped to OTHER/UNKNOWN OR OTHER.** CMPLID 826851 is a
standalone Graco child car seat -- make/model are `UNKNOWN`, and the complaint has
nothing to do with any vehicle system. B8 already recommended scoping the whole
project to `PROD_TYPE='V'`; this is a concrete instance of what goes wrong if that
scoping is skipped: the teacher gets marked "wrong" for correctly recognizing there's
no relevant vehicle system, when the real fix is that the row should never have been
in a vehicle-system classification set at all. **Removed from `gold_eval_set_reviewed.json`
along with CMPLID 2112094 (a telematics dump of three unrelated recall notices with no
actual complaint text, same non-answer problem).**

**B21. `SERVICE BRAKES` / `SERVICE BRAKES, HYDRAULIC` / `SERVICE BRAKES, AIR` are not
reliably distinguishable from ANYTHING available to us -- not narrative text, and not
vehicle make/model/year either.** Checked at scale across all 192,497 brake-family rows
in `cmpl_clean.parquet`: the top makes in all three buckets are the *same* ordinary
passenger-vehicle mix (Chevrolet, Ford, Dodge, GMC, Toyota, Honda, Jeep), not heavy
trucks for `AIR` as the name would suggest (consistent with B10's narrative-level
finding, extended here to the vehicle itself). More tellingly: **4,662 of 13,861
(33.6%) unique make+model+year combinations appear coded under more than one of the
three brake categories** -- e.g. the exact same vehicle shows up as plain `SERVICE
BRAKES` in one complaint and `SERVICE BRAKES, HYDRAULIC` in another, with no
identifiable reason. This isn't a narrow text-ambiguity problem the prompt can fix --
NHTSA's own coding of this subtype is functionally close to random. **Practical
consequence: the gold set now treats these three (plus `SERVICE BRAKES, ELECTRIC` when
no electric-specific detail is present) as a single interchangeable answer whenever the
narrative lacks distinguishing detail (`accept` contains all three/four) rather than
scoring a model down for landing on a different one of the three than NHTSA happened to
pick.** For any future *training* run (not just eval), the same logic argues for
collapsing these into one training class (e.g. `SERVICE BRAKES` as the umbrella) rather
than asking a model to learn a 3-4-way split that NHTSA's own human coders don't apply
consistently -- training on it as-is would just be teaching the model to memorize noise.

## D. Methodology note: what counts as "truth" when NHTSA disagrees with itself

During the manual row-by-row review of the 275-row gold eval set (2026-08-31), it
became clear that NHTSA's raw `COMPDESC` is **one hypothesis about the correct label,
not the definition of correct.** Two patterns showed this concretely:

- **NHTSA contradicts itself on the raw label, but that's a cue to read more carefully,
  not a license to accept both.** Two rows that look superficially similar at the
  `COMPDESC` level can still describe genuinely different things once you actually read
  the narratives -- CMPLID 1293783 explicitly states "FEAR OF AIRBAG DEPLOYING" (a real
  airbag-system concern, correctly `AIR BAGS`), while CMPLID 1346344 is purely "the
  dashboard cracked, I don't know what caused it" with no airbag symptom at all
  (`STRUCTURE`). Similarly, 1837120 and 1886408 explicitly describe sensor-triggered
  false braking with no obstacle (`FORWARD COLLISION AVOIDANCE`-supporting language),
  which is textually distinguishable from a bare brake-lockup complaint with no ADAS
  language. In every case checked, deciding from the text -- not defaulting to
  "NHTSA said both, so accept both" -- resolved these to a single label. Multi-label
  `accept` entries in the gold set are reserved for rows where two labels are each
  independently supported by that row's own narrative (see B10/B21 for the brake-family
  schema case, which is different: there the narrative genuinely can't distinguish the
  subtype, not "we didn't bother deciding").
- **The reviewer's own read of the narrative can override NHTSA outright**, as it did
  on 463855 (relabeled from a mis-transcribed "EQUIPMENT" to `ENGINE AND ENGINE
  COOLING`, which is also what the raw label already said once checked directly) and
  on the four rows where an earlier disagreement adjudication (see
  `disagreement_adjudication_n40.json`) turned out to have missed a detail in the
  narrative (1447285, 1779673, 463855, 1335413 -- see the `revised_prior_adjudications`
  field in `gold_eval_set_reviewed.json`).

In short: **NHTSA's label is our starting hypothesis, not our verdict.** The gold set
records what a careful reading of the narrative text supports, using NHTSA's label,
the teacher LLM's label, and NHTSA's own class definitions as evidence -- and where two
readings are both defensible (including two different NHTSA rows disagreeing with each
other), both are accepted rather than treating either as automatically right.

**B9. Circularity is low but nonzero.** Only ~20.5% of narratives literally contain the
top-level `COMPDESC` string (28.7% for any sub-segment) — so ~75-80% of the time this
is a real inference task, not a lookup task. Good news for the "does this need language
understanding" question, but also means the ~20% of cases where the narrative *does*
contain the literal component word are the easy wins, not representative of overall
difficulty.

## C. Other

**C1. Some complaints reference an attachment** ("please see attached letter") where
the visible `CDESCR` text alone is insufficient to derive the component (~1-2/50 in the
random sample). This is a small, irreducible ceiling on any narrative-only model —
worth a line in the limitations section, not a blocker.

---

## What this means for the teacher-model prompt (Phase 3)

1. Supply the **collapsed label list** (merge B3's near-duplicate clusters; ~30 classes
   covering ~99% of volume, see label-space memo), not the raw 55 top-level strings.
2. Include **2-3 few-shot examples per class**, chosen to specifically demonstrate B1
   (fire → root-cause system, not FIRERELATED) and B2 (smoke/glow ≠ fire) since these
   are the most confidently-wrong, most fixable mistakes a blind zero-shot model makes.
3. Explicitly instruct: *"if the narrative describes multiple problems, pick the one
   most central to the complainant's stated concern"* — won't fully resolve B5/B6 but
   gives the model a consistent tie-breaking rule to imitate.
4. Don't expect >90-95% agreement even from a well-prompted teacher on `COMPDESC` for
   the acceleration/braking overlap cases (B5) — that's a human-judgment ceiling, not a
   prompt-engineering failure.
5. Scope to `PROD_TYPE='V'` and dedupe by `ODINO` (A1) **before** the teacher ever sees
   the data, not as a post-hoc filter.
