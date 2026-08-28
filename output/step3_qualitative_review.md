# Step 3 — Qualitative read: 50 random + 20 stratified rare-positive complaints

Method: read `sample_50_qualitative.json` (pure random, seed 42+1) narrative-by-narrative
against COMPDESC/CRASH/FIRE/INJURED/DEATHS, judged agreement, derivability from the
narrative text alone, and circularity (label literally echoed). Supplemented with
`sample_20_rare_positive.json` (stratified toward CRASH=Y/FIRE=Y/INJURED>0/DEATHS>0)
because rare-field base rates are too low (~2-6%) for a pure-random 50 to say much
about crash/fire/injury coding quality specifically.

## Headline numbers

- **COMPDESC agreement (50 random sample): ~44/50 (88%) clearly correct**, ~3/50 (6%)
  clearly wrong, ~3/50 (6%) borderline/defensible-either-way.
- **My overall agreement with NHTSA's coding, combined sample: ~80-88%** — comfortably
  above the >=70% "gold is usable" bar, but with three *systematic*, non-random
  disagreement patterns below (not just noise — actionable for the pipeline).
- **Circularity: only 20.5% of narratives literally contain the top-level COMPDESC
  string** (28.7% for any colon-segment). So ~75-80% of the time the label is *not* a
  literal echo of the narrative — there is a real inference task, not a lookup task.

## Disagreement pattern 1 (most important): CRASH/INJURED/DEATHS sometimes describe a
**different row's incident**, not this row's CDESCR

Found in the rare-positive sample: a Continental tire complaint (CMPLID 384993) coded
CRASH=Y, INJURED=2, but its own CDESCR text ("tread separated from side wall... tire
had blow out") mentions no crash or injury at all. Same pattern in CMPLID 791379
(Yamaha ATV, CRASH=Y, narrative is purely about a fuel-tank-boiling design gripe, no
crash mentioned). Per CMPL.txt: **ODINO can repeat across multiple CMPLID rows for the
same incident/complainant** (one row per affected component), and CRASH/INJURED/DEATHS
appear to be set at the *incident* level and copied onto every component row — even
rows whose own narrative text doesn't describe the crash.

**Implication for the project:** a model trained/evaluated naively row-by-row will be
scored against labels that are sometimes not supported by that row's own input text.
Fix: dedupe/aggregate by ODINO before building the extraction-eval set, or explicitly
note+exclude/flag multi-row ODINOs as a known noise source. This is a data-engineering
fix, not a modeling problem — good "what v1 got wrong" material either way.

## Disagreement pattern 2: overlapping categories for unintended-acceleration complaints

Near-identical fact patterns ("foot on brake, car revs and accelerates on its own")
get coded inconsistently across three different top-level COMPDESC values depending on
which detail the coder anchored on:
- CMPLID 967312 (Jeep Laredo) -> **POWER TRAIN**
- CMPLID 703792, 1867138 (Camry, Grand Marquis) -> **VEHICLE SPEED CONTROL**
- CMPLID 2041059 (Kia EV6) -> **SERVICE BRAKES**
- CMPLID 862718 (Camry) -> **VEHICLE SPEED CONTROL:ACCELERATOR PEDAL**, even though its
  own narrative describes a *brake* failure ("she applied the brakes but the vehicle
  would not stop"), not acceleration — likely coded per the broader Toyota UA
  investigation context rather than this narrative alone (outside-knowledge coding).

**Implication:** COMPDESC for this failure mode is a real judgment call even for a
human, so don't expect >95% ceiling here; also explains why per-class confusion will
concentrate in POWER TRAIN / VEHICLE SPEED CONTROL / SERVICE BRAKES — worth calling out
in the error-analysis section later.

## Disagreement pattern 3: literal keyword-triggered FIRE mislabel

CMPLID 815740 (VW Jetta TDI): fuel pump failure caused a breakdown; the car happened to
break down next to an active wildfire and had **fire retardant dropped near it** by
aircraft. FIRE is coded **Y**, but the vehicle itself never caught fire — the narrative
literally contains "fire"/"flame" only in reference to the external wildfire. This is
the exact failure mode a naive keyword classifier would also make, and it shows NHTSA's
own coding isn't immune to it. One clean counter-example (CMPLID 227817, wiring
overheating/melting, correctly FIRE=Y) and one correct FIRE=N despite two "smoke from
steering column" incidents (CMPLID 725197) show the field is *usually* right, just not
error-free.

## Other observations
- Several complaints reference an attachment ("please see attached letter") where the
  *visible* CDESCR text alone is insufficient to derive the component
  (e.g., CMPLID 1482760) — a small but real ceiling on any narrative-only model.
  ~1-2/50 in this sample.
- Granular (colon-suffixed) COMPDESC values are sometimes over-specific relative to
  what the narrative supports (e.g., CMPLID 97877 coded down to
  "ANTILOCK/TRACTION CONTROL/ELECTRONIC LIMITED SLIP" when the narrative only mentions
  generic hard brake-pedal effort, no ABS symptom) — reinforces the Step 2 decision to
  use the **top-level ~19-25 category rollup**, not the 770-value granular string, as
  the label set.
- Many complaints describe 2+ simultaneous issues (e.g., CMPLID 288704: hood latch
  recall + cruise control recall + PS leak + CV boot + multiple engine replacements);
  only one COMPDESC is coded per row. This is an inherent single-label simplification
  of a sometimes multi-label reality, not a coding error — worth a line in limitations.

## Decision-relevant conclusion

Agreement rate (~80-88%) clears the >=70% "gold usable" bar with real margin, and it
does **not** clear the <65% pivot bar. The disagreement is concentrated in
*identifiable, fixable* patterns (ODINO grouping, UA-symptom category overlap) rather
than diffuse randomness — which is actually a *better* signal than a uniform noise
rate would be, since it means the eval set can be cleaned/documented rather than
distrusted wholesale.
