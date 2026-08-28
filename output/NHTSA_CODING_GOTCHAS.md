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
