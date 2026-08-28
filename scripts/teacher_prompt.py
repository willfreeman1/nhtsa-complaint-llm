"""Teacher-model prompt assembly for Rung-1 labeling.

Design notes (see output/NHTSA_CODING_GOTCHAS.md for the underlying evidence):

- The model is asked for the *specific* system name, not literally one of our 31
  training buckets. If it clearly identifies a real system outside our top-30 list
  (e.g. "TRACTION CONTROL SYSTEM", "HYBRID PROPULSION SYSTEM", "TRAILER HITCH"), it
  should say so in plain words rather than defaulting to a vague answer. Only
  "UNKNOWN OR OTHER" is used when the narrative genuinely doesn't pin down a system.
  `bucket_component()` below then maps that raw string to the final 31-class training
  label (top-30 verbatim, else OTHER) -- this is the same logic used in
  prepare_training_data.py, applied post-hoc so we don't ask the LLM to reason about
  our label-simplification choice.
- Fire/smoke language is explicitly de-fanged (gotchas B1/B2): mentioning fire does
  not imply a dedicated "fire" category (FIRERELATED is real but tiny, and isn't even
  a training class) -- route to the causal/associated system, or UNKNOWN OR OTHER if
  the cause is genuinely undetermined. Smoke/glow from friction or airbag deployment
  is explicitly called out as not necessarily FIRE=Y.
- CRASH/FIRE/INJURED/DEATHS are scoped to *this row's own narrative text only*
  (gotcha A1) -- the prompt says so explicitly, and the multi-row-ODINO noise is
  handled upstream in prepare_training_data.py (odino_multi_row flag), not by asking
  the teacher to guess about sibling rows it never sees.
- Multi-issue narratives (gotcha B6): explicit tie-break instruction to anchor on the
  complainant's most central stated concern.
"""
import json
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "output"

with open(OUT_DIR / "label_schema.json") as f:
    _SCHEMA = json.load(f)

TOP_30_CLASSES = [c["label"] for c in _SCHEMA["classes"] if c["label"] != "OTHER"]

CLASS_DEFINITIONS = {
    "ELECTRICAL SYSTEM": "Wiring, battery, ignition switch, fuses, general electrical faults; also electrical-origin fires when wiring/electrical is the identified cause.",
    "POWER TRAIN": "Transmission, driveline/axle, transfer case, engine mounts interacting with the drivetrain; EV/hybrid propulsion power-delivery issues not specific to the battery pack; general stalling/loss-of-power tied to the drivetrain.",
    "ENGINE": "Engine internals: spark plugs, valves, engine block, misfires, stalling caused by the engine itself; engine-bay fires with confirmed engine origin.",
    "AIR BAGS": "Airbag warning lights, non-deployment, inadvertent deployment, seat belt sensors tied to the airbag system, Takata-style recalls.",
    "STEERING": "Steering wheel, steering column, power steering, rack and pinion.",
    "ENGINE AND ENGINE COOLING": "Cooling system specifically: radiator, coolant leaks/overheating, water pump, cooling fan. Use over plain ENGINE when overheating/coolant is the central complaint.",
    "UNKNOWN OR OTHER": "The narrative genuinely does not identify which system is at fault -- vague drivability complaints (car \"died\"/\"shook\"/\"lost power\") with no specific part named, or a failure with no cause identified even after investigation.",
    "SERVICE BRAKES, HYDRAULIC": "Hydraulic brake system: brake lines, master cylinder, brake fluid -- the default bucket for ordinary brake complaints without ABS/air/electric-specific detail.",
    "STRUCTURE": "Body/frame/chassis structural issues: frame rust/cracking, hood/body structural failures; a fire whose origin is attributed to a structural/body component rather than engine or wiring.",
    "SUSPENSION": "Struts, shocks, control arms, ball joints, springs.",
    "VEHICLE SPEED CONTROL": "Cruise control and unintended-acceleration/accelerator-pedal issues (throttle sticking, pedal misapplication) -- speed control, not braking.",
    "SERVICE BRAKES": "General/unspecified brake complaints that don't clearly indicate hydraulic, air, or electric subtype.",
    "EXTERIOR LIGHTING": "Headlights, tail lights, turn signals, exterior bulbs/housings -- only when the lighting itself (not general electrical) is the stated problem.",
    "FUEL/PROPULSION SYSTEM": "General fuel or propulsion delivery issues not specific to a gasoline/diesel subtype.",
    "FUEL SYSTEM, GASOLINE": "Gasoline-specific fuel system (fuel pump, tank, lines) explicitly tied to a gas engine.",
    "VISIBILITY": "Windshield, mirrors, defrosting/defogging, general visibility obstruction (not wiper-specific).",
    "TIRES": "Tire tread separation, blowouts, wear, sidewall failures.",
    "SEAT BELTS": "Belt retraction, buckle, webbing, pretensioner failures.",
    "SEATS": "Seat frame, seat-back collapse, seat track/adjustment failures, seat heaters.",
    "FORWARD COLLISION AVOIDANCE": "Forward collision warning / automatic emergency braking system false alerts or failure to activate.",
    "WHEELS": "Wheel rim cracking/bending or wheel separation, distinct from the tire itself failing.",
    "VISIBILITY/WIPER": "Windshield wiper/washer system specifically.",
    "ELECTRONIC STABILITY CONTROL": "ESC/stability system malfunction or warning light.",
    "EQUIPMENT": "General aftermarket/OEM accessory equipment not covered by any other category (e.g. roof racks, tonneau covers, cargo equipment).",
    "LATCHES/LOCKS/LINKAGES": "Door/hood latches, locks, and related linkages -- the latch/lock mechanism itself, not a structural failure.",
    "BACK OVER PREVENTION": "Backup camera / rearview visibility systems specifically intended for back-over prevention.",
    "LANE DEPARTURE": "Lane departure warning / lane keep assist AND blind-spot monitoring system malfunctions -- NHTSA groups blind-spot detection under this category too, not just literal lane-departure warnings.",
    "FUEL SYSTEM, OTHER": "Fuel system issues not clearly gasoline or diesel (e.g. propane, CNG, or unspecified fuel type).",
    "PARKING BRAKE": "Parking/emergency brake mechanism specifically.",
    "SERVICE BRAKES, AIR": "Despite the name, in practice this category is used broadly for ordinary hydraulic/disc brake complaints on regular passenger vehicles (rotors, calipers, ABS units) -- only ~1.7% of narratives in this category even mention \"air brake\" literally. Do not expect the narrative to describe a heavy-truck pneumatic air-brake system; treat it as functionally similar to SERVICE BRAKES / SERVICE BRAKES, HYDRAULIC and don't over-weight the word \"air.\"",
}
assert set(CLASS_DEFINITIONS) == set(TOP_30_CLASSES), "definitions must exactly cover the 30 primary classes"

SYSTEM_INSTRUCTIONS = """You are labeling NHTSA vehicle owner complaint narratives for a machine-learning training set. For each complaint, read the narrative (and make/model/year if given) and return a single JSON object with these fields:

- "component": the specific vehicle system/component at fault, as a string.
  - If the narrative clearly matches one of the 30 primary categories listed below, output that category name EXACTLY as written.
  - If the narrative clearly points to a real, specific vehicle system that is NOT one of the 30 (e.g. traction control system, hybrid/EV battery propulsion, trailer hitch, interior lighting, infotainment/communications, child-seat hardware), name that specific system in plain words rather than forcing a poor fit into one of the 30. Do not be shy about naming something outside the list -- precision matters more than matching the list.
  - Only output "UNKNOWN OR OTHER" when the narrative genuinely does not identify which system is at fault (e.g. vague "car died"/"lost power" complaints with no specific part named).
- "crash": "Y" or "N" -- does THIS narrative's own text describe a crash/collision? Do not infer from context outside the text; if the narrative is silent on a crash, answer "N".
- "fire": "Y" or "N" -- does THIS narrative's own text describe an actual vehicle fire (flames, something burning)? Smoke/glowing from normal friction (e.g. hard-braked brakes) or airbag-deployment discharge is NOT a fire. An external fire (e.g. wildfire) near the vehicle that never ignites the vehicle itself is NOT a fire.
- "injured": integer count of people injured per THIS narrative's own text (0 if none mentioned).
- "deaths": integer count of deaths per THIS narrative's own text (0 if none mentioned).
- "confidence": "low", "medium", or "high" -- your confidence in the "component" answer.
- "rationale": one short phrase (<15 words) explaining the component choice.

Important rules:
1. Mentioning fire/flames/smoke does NOT by itself imply a special "fire" category -- identify the underlying system that caused or is most closely associated with the fire (e.g. a wiring fire -> ELECTRICAL SYSTEM; an engine-bay fire with confirmed engine origin -> ENGINE; a fire with no origin identified -> UNKNOWN OR OTHER).
2. If the narrative describes multiple problems, pick the ONE most central to the complainant's stated primary concern -- not necessarily the first-mentioned or most technical-sounding detail.
3. Base crash/fire/injured/deaths ONLY on what this specific narrative says, never on assumptions about a broader incident you're not shown.
4. Output ONLY the JSON object, no other text.

## The 30 primary categories

{class_list}

## Examples
{fewshot_block}
"""

USER_TEMPLATE = """Make/Model/Year: {make} {model} {year}
Narrative: {narrative}"""


def _format_class_list():
    return "\n".join(f"- {name}: {desc}" for name, desc in CLASS_DEFINITIONS.items())


def _format_example(ex, component_override=None):
    user = USER_TEMPLATE.format(
        make=ex["make"], model=ex["model"], year=ex["year"], narrative=ex["narrative"]
    )
    component = component_override if component_override else ex["gold_COMPDESC_TOP"]
    answer = {
        "component": component,
        "crash": ex["crash"],
        "fire": ex["fire"],
        "injured": int(ex["injured"]),
        "deaths": int(ex["deaths"]),
    }
    return f"Input:\n{user}\n\nOutput:\n{json.dumps(answer)}"


# Manual overrides for classes where the first auto-sampled candidate turned out to be
# a poor teaching example on inspection (either a multi-issue narrative anchored to a
# minor detail, or a narrative that doesn't actually match the class definition).
# cmplid -> hand-picked replacement, sourced by direct query against cmpl_clean.parquet.
CURATED_OVERRIDES = {
    "VEHICLE SPEED CONTROL": {
        "cmplid": 88495, "make": "CHEVROLET", "model": "S10", "year": "1997",
        "narrative": "CRUISE CONTROL FAILED TO DISENGAGE, VEHICLE CONTINUED TO ACCELERATE OUT OF CONTROL. WAS ABLE TO STOP VEHICLE BY PUTTING IT INTO NEUTRAL AND TURNING OFF KEY.  *AK",
        "crash": "N", "fire": "N", "injured": 0, "deaths": 0,
    },  # original auto-sample (Ford Explorer "lost power downhill") was an ambiguous
        # loss-of-power case with no cruise/accelerator signal -- bad teaching example
        # for this class even though it's a real gold label.
    "EXTERIOR LIGHTING": {
        "cmplid": 325175, "make": "UNKNOWN", "model": "UNKNOWN", "year": "2001",
        "narrative": "THIS COMPLAINT IS CONCERNING HID HEADLAMPS ON NEWER MODEL VEHICLES.  THEY ARE VERY BLINDING AND EYES TAKE CONSIDERABLY MORE TIME TO RECOVER AFTER PASSING A VEHICLE EQUIPED WITH HID.  CONSIDER BANNING THESE PLEASE.*AK",
        "crash": "N", "fire": "N", "injured": 0, "deaths": 0,
    },  # original auto-sample (Dodge Journey multi-system electrical cascade) directly
        # contradicted this class's own definition (general electrical, not lighting).
    "WHEELS": {
        "cmplid": 626198, "make": "UNKNOWN", "model": "UNKNOWN", "year": "9999",
        "narrative": "FRONT AND REAR RIMS ON MY 2005 NISSAN 350Z BROKE AND REQUIRED REPLACEMENT. A TOTAL OF 5 RIMS CRACKED ON MY CAR.  IT REQUIRED ME TO REPLACE THE NISSAN RIMS WITH AFTER MARKET RIMS.  *TR",
        "crash": "N", "fire": "N", "injured": 0, "deaths": 0,
    },  # original auto-sample (Cadillac DeVille) was mostly about an ignition-switch/key
        # problem with only a passing "wheel monitoring light" mention -- contradicts our
        # own multi-issue tie-break instruction.
    "SERVICE BRAKES, AIR": {
        "cmplid": 543268, "make": "FORD", "model": "F SERIES", "year": "2000",
        "narrative": "ABS WARNING LIGHT CAME ON.  DIAGNOSIS WAS FAULTY HYDRAULLIC CONTROL UNIT.  SINCE IT OCCURED AT 61,615 MILES AND STILL ON ORIGINAL BRAKE PADS/LININGS AND TIRES I FEEL FAILURE WAS PREMATURE.  HAPPENED SHORTLY AFTER RECAL FOR SPEED CONTOL DISTURBED BRAKE SYSTEM.  REPAIR COST FOR DIAGNOSIS AND EXTIMATE FOR REPAIR $475+ SO REPAIRS HAVE NOT YET BEEN DONE.",
        "crash": "N", "fire": "N", "injured": 0, "deaths": 0,
    },  # original auto-sample (Jeep rotor wear) was fine too, but this one at least
        # mentions ABS/hydraulic brake system explicitly, consistent with the corrected
        # class definition above (this bucket is not actually about pneumatic air brakes).
}


def build_fewshot_block():
    with open(OUT_DIR / "fewshot_candidates.json") as f:
        cands = json.load(f)

    blocks = []
    # one representative example per primary class (curated override if we have one,
    # else the first auto-sampled candidate)
    for cls in TOP_30_CLASSES:
        if cls in CURATED_OVERRIDES:
            blocks.append(_format_example(CURATED_OVERRIDES[cls], component_override=cls))
            continue
        examples = cands["per_class"].get(cls, [])
        if examples:
            blocks.append(_format_example(examples[0]))

    # fire-routing gotcha (B1/B2): fire language routed to the causal system
    for ex in cands["fire_routing_examples"][:3]:
        blocks.append(_format_example(ex))

    # OTHER vs UNKNOWN OR OTHER disambiguation
    for ex in cands["other_vs_unknown_examples"]:
        blocks.append(_format_example(ex))

    return "\n\n".join(blocks)


def build_system_prompt():
    return SYSTEM_INSTRUCTIONS.format(
        class_list=_format_class_list(),
        fewshot_block=build_fewshot_block(),
    )


def build_user_message(narrative, make="", model="", year=""):
    return USER_TEMPLATE.format(make=make, model=model, year=year, narrative=narrative)


def bucket_component(raw_component: str) -> str:
    """Map a teacher's raw component string to the final 31-class training label."""
    if raw_component in TOP_30_CLASSES or raw_component == "UNKNOWN OR OTHER":
        return raw_component
    return "OTHER"


if __name__ == "__main__":
    prompt = build_system_prompt()
    print(f"System prompt length: {len(prompt):,} chars (~{len(prompt)//4:,} tokens)")

    demo_user = build_user_message(
        narrative="MY HUSBAND WAS DRIVING AND SUDDENLY THE STEERING WHEEL LOCKED UP AND HE COULD NOT TURN. HE MANAGED TO PULL OVER SAFELY.",
        make="TOYOTA", model="CAMRY", year="2015",
    )
    preview = (
        "# Teacher prompt preview\n\n"
        "## System prompt\n\n```\n" + prompt + "\n```\n\n"
        "## Example user message (not from few-shot set)\n\n```\n" + demo_user + "\n```\n"
    )
    out_path = OUT_DIR / "teacher_prompt_preview.md"
    out_path.write_text(preview, encoding="utf-8")
    print(f"Wrote {out_path}")
