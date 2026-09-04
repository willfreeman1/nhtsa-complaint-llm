"""Expand gold_eval_set_v2.json (498 rows) with 53 deliberately-sampled rows targeting
the 7 (of 9) `raw_component` classes that had zero gold coverage and were recoverable
from the corpus without train/eval leakage (see build_rare_gold_expansion.py; the other
2 -- FIRERELATED and TRAILER HITCHES -- have literally zero real complaints left in the
corpus after training-data exclusion, and are documented as uncoverable rather than
compromised by reusing training rows).

Unlike the rest of gold_eval_set_v2.json (a random pilot sample later adjudicated), these
53 rows were selected by literal COMPDESC_TOP match (deliberate coverage sampling, not
random) and then independently adjudicated from the narrative text against the same
class definitions in teacher_prompt.py used throughout this project -- NHTSA's raw
COMPDESC field is treated as one hypothesis, not ground truth, same as every other
adjudicated row in this gold set. Only 16 of the 53 rows (30%) actually confirmed the
NHTSA-hypothesized rare category as the correct primary label; the rest were reclassified
to whatever the narrative actually supports -- itself a notable finding about how noisy
several of these rare COMPDESC_TOP buckets are in the raw corpus (see WRITEUP.md).

Usage:
    python scripts/build_gold_eval_set_v3.py
"""
import json
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "output"

# Each entry: cmplid -> (primary, accept_list, is_hard, note, target_category)
# target_category is what NHTSA's raw COMPDESC_TOP said before adjudication.
ADJUDICATIONS = {
    # --- CHILD SEAT candidates ---
    "608797": ("SEAT BELTS", ["SEAT BELTS"], True,
        "Narrative describes the vehicle's own 3-point seat belt failing to hold an "
        "infant seat base tight, not a built-in child-seat feature; SEAT BELTS fits the "
        "actual described failure better than CHILD SEAT (which per teacher_prompt.py's "
        "definition requires a built-in/integrated child-seat feature, not an aftermarket "
        "seat secured by the regular belt).", "CHILD SEAT"),
    "1516532": ("UNKNOWN OR OTHER", ["UNKNOWN OR OTHER"], True,
        "Third-party aftermarket car seat (Diono Radian RXT) strap complaint; not a "
        "vehicle system at all, and CHILD SEAT's definition explicitly excludes "
        "standalone aftermarket products.", "CHILD SEAT"),
    "677248": ("UNKNOWN OR OTHER", ["UNKNOWN OR OTHER"], True,
        "Extremely terse ('son's finger stuck in his carseat') with no detail on whether "
        "this is an integrated vehicle child-seat feature or an aftermarket seat; too "
        "vague to confidently assign a specific system.", "CHILD SEAT"),
    "461243": ("CHILD SEAT", ["CHILD SEAT"], False,
        "1998-2004 GM minivans (Oldsmobile Silhouette here) offered a factory-integrated "
        "child seat; harness arm detaching/cracking is a direct integrated-child-seat-"
        "feature failure.", "CHILD SEAT"),
    "279025": ("CHILD SEAT", ["CHILD SEAT"], False,
        "Narrative explicitly names an 'INTEGRATED CHILD SEAT' (Honda Odyssey factory "
        "integrated seat option); harness cinching malfunction is part of that "
        "integrated feature.", "CHILD SEAT"),
    "550726": ("UNKNOWN OR OTHER", ["UNKNOWN OR OTHER"], True,
        "Explicitly names a third-party product ('EDDIE BAUER 22 INFANT COSCO INC.'); "
        "aftermarket product, out of CHILD SEAT's defined scope per teacher_prompt.py.",
        "CHILD SEAT"),
    "714087": ("UNKNOWN OR OTHER", ["UNKNOWN OR OTHER"], True,
        "General safety-design question/opinion about why rear-facing infant seats lack "
        "a top tether, not a report of an actual failure of anything; no specific system "
        "failure described.", "CHILD SEAT"),

    # --- ELECTRONIC STABILITY CONTROL candidates ---
    "1016581": ("ELECTRONIC STABILITY CONTROL", ["ELECTRONIC STABILITY CONTROL", "TRACTION CONTROL SYSTEM"], True,
        "Narrative explicitly names both 'SERVICE ESC' and 'SERVICE TRACTION' warnings "
        "together; ESC named directly so kept as primary, traction control accepted as "
        "a defensible alternate.", "ELECTRONIC STABILITY CONTROL"),
    "1696183": ("ELECTRONIC STABILITY CONTROL", ["ELECTRONIC STABILITY CONTROL"], False,
        "Clean explicit match -- 'ABS light and STABILITY light' -- matches a documented "
        "2012-2015 Dodge Journey recall for this exact symptom.", "ELECTRONIC STABILITY CONTROL"),
    "1437163": ("UNKNOWN OR OTHER", ["UNKNOWN OR OTHER"], True,
        "Narrative is purely about a recurring check-engine light with no mention of "
        "ESC/stability/traction at all; NHTSA's raw COMPDESC_TOP appears mislabeled "
        "here.", "ELECTRONIC STABILITY CONTROL"),
    "1453743": ("TRACTION CONTROL SYSTEM", ["TRACTION CONTROL SYSTEM"], True,
        "Narrative explicitly names 'ADVANCED SERVICE TRACTION LIGHT' (not stability); "
        "per teacher_prompt.py's own tie-break rule ('distinct from ESC when the "
        "narrative specifically names traction control, not ESC/stability'), this is "
        "TRACTION CONTROL SYSTEM, not ESC.", "ELECTRONIC STABILITY CONTROL"),
    "1335594": ("POWER TRAIN", ["POWER TRAIN", "TRACTION CONTROL SYSTEM"], True,
        "Dominant complaint is the vehicle stalling/losing power below 30mph ('reduced "
        "engine power'); traction control light is a concurrent symptom, not the "
        "primary failure.", "ELECTRONIC STABILITY CONTROL"),
    "1523922": ("SUSPENSION", ["SUSPENSION"], True,
        "Narrative is entirely about air suspension lowering/failing in cold weather; no "
        "ESC/traction/stability mention at all -- clear NHTSA mislabel.",
        "ELECTRONIC STABILITY CONTROL"),
    "1242671": ("UNKNOWN OR OTHER", ["UNKNOWN OR OTHER"], True,
        "Extremely vague ('van just stopped, completely dead'); no specific system "
        "identifiable from the text.", "ELECTRONIC STABILITY CONTROL"),

    # --- EQUIPMENT ADAPTIVE/MOBILITY candidates (original 7, random sample) ---
    "1185717": ("ELECTRICAL SYSTEM", ["ELECTRICAL SYSTEM"], True,
        "Multiple ADAS features (blind spot, cross path, park assist, camera, compass) "
        "all failed together due to one corroded module behind the bumper; the shared "
        "electrical root cause is the best single label, not any one downstream "
        "feature.", "EQUIPMENT ADAPTIVE/MOBILITY"),
    "1208068": ("ELECTRICAL SYSTEM", ["ELECTRICAL SYSTEM", "STRUCTURE"], True,
        "Convertible-top water intrusion corroded the wire loom, disabling all "
        "electrical safety items (brake lights included); electrical failure is the "
        "described safety hazard, convertible top design flaw is the root cause.",
        "EQUIPMENT ADAPTIVE/MOBILITY"),
    "914919": ("EQUIPMENT", ["EQUIPMENT"], False,
        "Textbook case for teacher_prompt.py's EQUIPMENT definition: standalone "
        "AC/climate-control failure (compressor) with no defrost/defog/visibility "
        "consequence mentioned.", "EQUIPMENT ADAPTIVE/MOBILITY"),
    "1190333": ("EQUIPMENT", ["EQUIPMENT"], True,
        "Horn malfunction; no dedicated horn category exists in the schema, EQUIPMENT's "
        "general-accessory catch-all fits best.", "EQUIPMENT ADAPTIVE/MOBILITY"),
    "1132608": ("AIR BAGS", ["AIR BAGS"], True,
        "Narrative is purely about the airbag warning light cycling on/off; clear NHTSA "
        "mislabel.", "EQUIPMENT ADAPTIVE/MOBILITY"),
    "1209940": ("SERVICE BRAKES, HYDRAULIC", ["SERVICE BRAKES, HYDRAULIC"], True,
        "ABS control module/pump failure disabling the ABS system; standard "
        "hydraulic-brake-family issue, not adaptive/mobility equipment.",
        "EQUIPMENT ADAPTIVE/MOBILITY"),
    "744932": ("VEHICLE SPEED CONTROL", ["VEHICLE SPEED CONTROL"], True,
        "Unintended-acceleration complaint explicitly compared to cruise-control "
        "behavior, with floor mat ruled out; fits VEHICLE SPEED CONTROL, not "
        "adaptive/mobility equipment.", "EQUIPMENT ADAPTIVE/MOBILITY"),

    # --- EQUIPMENT ADAPTIVE/MOBILITY, targeted keyword-search follow-up (genuine hits) ---
    "240212": ("EQUIPMENT ADAPTIVE/MOBILITY", ["EQUIPMENT ADAPTIVE/MOBILITY"], False,
        "Genuine match found via targeted keyword search: handicap-conversion "
        "wheelchair lift installed incorrectly, described as unreliable for "
        "independent handicapped use.", "EQUIPMENT ADAPTIVE/MOBILITY"),
    "197350": ("EQUIPMENT ADAPTIVE/MOBILITY", ["EQUIPMENT ADAPTIVE/MOBILITY"], True,
        "Genuine match: factory handicap ramp lacked a safety lock and fell during a "
        "side-impact crash, injuring the driver -- multi-issue (crash + equipment "
        "defect) but the equipment failure is the root cause.", "EQUIPMENT ADAPTIVE/MOBILITY"),
    "977756": ("EQUIPMENT ADAPTIVE/MOBILITY", ["EQUIPMENT ADAPTIVE/MOBILITY"], False,
        "Genuine match: powered wheelchair ramp motor locked up, trapping the "
        "wheelchair-using occupant inside the vehicle -- textbook adaptive/mobility "
        "equipment failure.", "EQUIPMENT ADAPTIVE/MOBILITY"),
    "1159204": ("EQUIPMENT ADAPTIVE/MOBILITY", ["EQUIPMENT ADAPTIVE/MOBILITY"], False,
        "Genuine match: aftermarket hand controls (gas/brake pedal hand-control device) "
        "for a handicapped driver broke in use; teacher_prompt.py's definition "
        "explicitly lists hand controls as a covered example (unlike CHILD SEAT, this "
        "category's definition doesn't exclude aftermarket adaptive devices).",
        "EQUIPMENT ADAPTIVE/MOBILITY"),

    # --- FUEL SYSTEM, OTHER candidates ---
    "772267": ("VEHICLE SPEED CONTROL", ["VEHICLE SPEED CONTROL"], True,
        "Unintended-acceleration complaint ('bursts of energy, jolts forward'), floor "
        "mat explicitly ruled out; matches VEHICLE SPEED CONTROL, not a "
        "fuel-type-ambiguous fuel system issue.", "FUEL SYSTEM, OTHER"),
    "707562": ("FUEL SYSTEM, OTHER", ["FUEL SYSTEM, OTHER"], False,
        "RV propane/ammonia refrigerator gas leak -- a clean, literal match for the "
        "definition's own example (propane/unspecified fuel type).", "FUEL SYSTEM, OTHER"),
    "383022": ("FUEL SYSTEM, GASOLINE", ["FUEL SYSTEM, GASOLINE"], True,
        "1992 F-150 (gasoline) dual-tank fuel transfer/pressure issue; explicitly a gas "
        "vehicle's tank/lines, fits GASOLINE better than the ambiguous-fuel-type OTHER "
        "bucket.", "FUEL SYSTEM, OTHER"),
    "768442": ("FUEL SYSTEM, GASOLINE", ["FUEL SYSTEM, GASOLINE"], True,
        "Fuel gauge/sender malfunction on a standard gasoline Trailblazer; fits "
        "GASOLINE better than OTHER.", "FUEL SYSTEM, OTHER"),
    "252356": ("UNKNOWN OR OTHER", ["UNKNOWN OR OTHER"], True,
        "Confusing complaint about being unable to shift to neutral after a fuel stop; "
        "doesn't describe an actual fuel-delivery failure, root mechanism unclear from "
        "text.", "FUEL SYSTEM, OTHER"),
    "573467": ("FUEL SYSTEM, DIESEL", ["FUEL SYSTEM, DIESEL"], True,
        "Fuel injection pump replaced three times at ~45,000-mile intervals on a Ram "
        "2500 -- matches the well-documented Dodge Cummins diesel VP44 injection pump "
        "failure pattern; fits DIESEL better than the ambiguous OTHER bucket.",
        "FUEL SYSTEM, OTHER"),
    "1713": ("FUEL SYSTEM, GASOLINE", ["FUEL SYSTEM, GASOLINE"], True,
        "Simple fuel gauge malfunction on a standard gasoline Sentra; fits GASOLINE "
        "better than OTHER.", "FUEL SYSTEM, OTHER"),

    # --- FUEL/PROPULSION SYSTEM candidates ---
    "1705326": ("ENGINE", ["ENGINE", "FUEL/PROPULSION SYSTEM"], True,
        "Engine knock and hesitation under acceleration is primarily an "
        "engine-mechanical symptom.", "FUEL/PROPULSION SYSTEM"),
    "1578783": ("POWER TRAIN", ["POWER TRAIN"], True,
        "Hard-starting/cranking issue (15 key turns to catch) is the dominant, "
        "safety-relevant complaint; a separate DVD-player malfunction is also "
        "described but is not a vehicle propulsion system.", "FUEL/PROPULSION SYSTEM"),
    "1142140": ("FUEL SYSTEM, GASOLINE", ["FUEL SYSTEM, GASOLINE"], True,
        "Explicitly names the failed part (corroded fuel pump controller) on a "
        "gasoline V8; fits GASOLINE better than the generic bucket.",
        "FUEL/PROPULSION SYSTEM"),
    "1709414": ("FUEL SYSTEM, GASOLINE", ["FUEL SYSTEM, GASOLINE"], True,
        "Explicit gasoline leak from the fuel tank on a gas Corvette; fits GASOLINE "
        "cleanly.", "FUEL/PROPULSION SYSTEM"),
    "1115614": ("FUEL/PROPULSION SYSTEM", ["FUEL/PROPULSION SYSTEM", "POWER TRAIN"], True,
        "Speed sensor feeding fuel-flow interruption to the engine management system, "
        "not a specific tank/pump/line failure -- fits the generic 'not specific to a "
        "subtype' bucket as intended; genuine match.", "FUEL/PROPULSION SYSTEM"),
    "1329293": ("POWER TRAIN", ["POWER TRAIN"], True,
        "Cannot shift gears while the engine runs -- a transmission/drivetrain failure "
        "(matches a known Freestar transmission recall pattern), not a fuel system "
        "issue.", "FUEL/PROPULSION SYSTEM"),
    "1728411": ("ENGINE", ["ENGINE"], True,
        "Two-word complaint naming only 'catalytic converter' -- an exhaust/emissions "
        "component of the engine system; terse but specific enough to name ENGINE over "
        "a vaguer fuel-system bucket.", "FUEL/PROPULSION SYSTEM"),

    # --- SERVICE BRAKES, ELECTRIC candidates ---
    "822481": ("POWER TRAIN", ["POWER TRAIN"], True,
        "Multi-issue complaint dominated by repeated transmission/throttle-body "
        "failures (surging in reverse, dying, input shaft bearing replacement, "
        "transmission light); no electric brake system mentioned.",
        "SERVICE BRAKES, ELECTRIC"),
    "750966": ("SERVICE BRAKES, ELECTRIC", ["SERVICE BRAKES, ELECTRIC"], True,
        "2010 Prius brake-surge-over-bumps complaint matches Toyota's documented recall "
        "for the electronically-controlled regenerative brake system's "
        "ABS-modulation behavior -- a genuine SERVICE BRAKES, ELECTRIC case, not a "
        "plain hydraulic brake issue.", "SERVICE BRAKES, ELECTRIC"),
    "820993": ("ELECTRICAL SYSTEM", ["ELECTRICAL SYSTEM"], True,
        "Brake light switch malfunction (lights stay on when brakes aren't engaged, "
        "also disrupts cruise control) on a conventional (non-hybrid) G6; an "
        "electrical switch fault, not the vehicle's braking hardware itself.",
        "SERVICE BRAKES, ELECTRIC"),
    "760021": ("SERVICE BRAKES, ELECTRIC", ["SERVICE BRAKES, ELECTRIC"], True,
        "Same documented Prius electronically-controlled brake-blending recall pattern "
        "as cmplid 750966.", "SERVICE BRAKES, ELECTRIC"),
    "758906": ("SERVICE BRAKES, ELECTRIC", ["SERVICE BRAKES, ELECTRIC"], True,
        "Same Prius brake-blending pattern, vaguer description but same known issue.",
        "SERVICE BRAKES, ELECTRIC"),
    "758789": ("SERVICE BRAKES, ELECTRIC", ["SERVICE BRAKES, ELECTRIC"], True,
        "Camry Hybrid shares the same era's electronically-controlled brake system as "
        "the Prius cases above; brake-lag complaint fits the same recall pattern.",
        "SERVICE BRAKES, ELECTRIC"),
    "734318": ("SERVICE BRAKES, HYDRAULIC", ["SERVICE BRAKES, HYDRAULIC"], True,
        "Conventional ABS control module failure on a non-hybrid Windstar; standard "
        "hydraulic ABS issue, not an electronically-blended brake system.",
        "SERVICE BRAKES, ELECTRIC"),

    # --- TRACTION CONTROL SYSTEM candidates ---
    "786293": ("SERVICE BRAKES, HYDRAULIC", ["SERVICE BRAKES, HYDRAULIC", "TRACTION CONTROL SYSTEM"], True,
        "Actual brakes fail to engage immediately (had to slam brakes to stop "
        "mid-intersection); traction/ABS lights are concurrent symptoms of a "
        "wheel-speed-sensor fault, but the substantive safety issue is brake function "
        "itself.", "TRACTION CONTROL SYSTEM"),
    "2158233": ("SERVICE BRAKES, HYDRAULIC", ["SERVICE BRAKES, HYDRAULIC", "TRACTION CONTROL SYSTEM"], True,
        "Narrative directly cites NHTSA recall campaign 24V896000, officially titled "
        "'Service Brakes, Hydraulic' -- strong external confirmation despite the "
        "traction-control symptom described.", "TRACTION CONTROL SYSTEM"),
    "1023573": ("TRACTION CONTROL SYSTEM", ["TRACTION CONTROL SYSTEM", "ELECTRONIC STABILITY CONTROL"], True,
        "Explicitly names both 'traction controls' and 'StabiliTrak' (GM's ESC brand) "
        "together with a suspected ABS sensor; traction control named first/primarily, "
        "genuine match.", "TRACTION CONTROL SYSTEM"),
    "1075370": ("STEERING", ["STEERING", "TRACTION CONTROL SYSTEM"], True,
        "The substantive safety issue described is a loss of power steering assist "
        "('steering was extremely hard'); the traction control light is a concurrent, "
        "secondary symptom.", "TRACTION CONTROL SYSTEM"),
    "840221": ("UNKNOWN OR OTHER", ["UNKNOWN OR OTHER"], True,
        "Vague complaint about repair-cost estimates covering multiple named sensors "
        "(ABS, traction/stability, throttle) with no specific failure behavior "
        "described.", "TRACTION CONTROL SYSTEM"),
    "2117316": ("TRACTION CONTROL SYSTEM", ["TRACTION CONTROL SYSTEM", "ELECTRONIC STABILITY CONTROL"], True,
        "Ford's 'AdvanceTrac' (combined traction/stability system) light illuminated "
        "alongside ABS/Hill Start Assist/Powertrain lights; genuine match given "
        "AdvanceTrac's traction-control naming.", "TRACTION CONTROL SYSTEM"),
    "948325": ("POWER TRAIN", ["POWER TRAIN", "TRACTION CONTROL SYSTEM"], True,
        "Dominant, dangerous symptom is a sudden loss of engine power (limp-mode, "
        "engine running only at idle) nearly causing a rear-end collision; "
        "traction/stability warnings are concurrent symptoms of the underlying "
        "drivetrain fault.", "TRACTION CONTROL SYSTEM"),
}


def main():
    with open(OUT_DIR / "gold_eval_set_v2.json") as f:
        v2 = json.load(f)

    with open(OUT_DIR / "rare_gold_expansion_candidates.json") as f:
        candidates = {r["cmplid"]: r for r in json.load(f)["rows"]}

    # The 4 targeted keyword-search follow-ups aren't in rare_gold_expansion_candidates.json
    # (that file only has the original random-within-category sample); their full metadata
    # was captured directly during the search step.
    extra_rows = {
        "240212": {"make": "FORD", "model": "E-250", "year": "1999",
            "narrative": ("IT WAS A HANDICAP CONVERSION PURCHASED THROUGH COACHCRAFT, IN "
                "BROWNSVILLE PAYED FOR ON FEB 5,1999.THE ACTUAL CONVERSION DONE BY VANTAGE "
                "MOBILITY IN PHOENIX AR.WAS NOT DELIVERD TILL 6-11-99 AND HAS BEEN A CONSTANT "
                "SOURCE OF PROBLEMS.THE LIFT WAS INSTALLED WRONG AND WILL NOT WORK PROPERLY "
                "OR RELIABLY FOR ME THE HANDICAPPED TO BE INDEPENDENT, THE FUEL TANK WHEN "
                "FILLED LEAKS FUME INTO THE INTERIOR CHOKING ME."),
            "nhtsa_crash": "N", "nhtsa_fire": "N"},
        "197350": {"make": "DODGE", "model": "CARAVAN", "year": "1998",
            "narrative": ("VEHICLE WAS MANUFACTURED WITH A HANDICAP RAMP. VEHICLE WAS INVOLVED "
                "IN A SIDE IMPACT ACCIDENT WHICH CAUSED THE LIFT TO FALL AND HIT THE DRIVER IN "
                "THE JAW BREAKING IT.  THE DRIVER ALSO SUSTAINED INJURIES TO THE SHOULDER AND "
                "NECK.  THE RAMP HAS NO SAFETY LOCK FEATURE WHICH COULD HAVE PREVENTED "
                "INJURIES.  *AK  *ML  *NM"),
            "nhtsa_crash": "Y", "nhtsa_fire": "N"},
        "977756": {"make": "TOYOTA", "model": "SIENNA", "year": "2011",
            "narrative": ("THE POWERED FOLD UP RAMP AND MOTOR LOCKED UP COMPLETELY BLOCKING "
                "WHEEL CHAIR EXIT AND ENTRY ACCESS IN THE TOYOTA SIENNA 2011 BRAUN ABILITY "
                "RAMPVAN. THE ALLEGED MANUAL RAMP OPERATION FUNCTION FAILED TO WORK TRAPPING "
                "THE WHEEL CHAIR OCCUPANT INSIDE THE VEHICLE. THIS DANGEROUS DESIGN HAS BEEN "
                "MADE SINCE 2011 AND CONTINUES TO BE MADE INTO 2013. BRAUN HAS RELEASED "
                "ANOTHER IN-FLLOOR RAMP OPTION THIS YEAR."),
            "nhtsa_crash": "N", "nhtsa_fire": "N"},
        "1159204": {"make": "UNKNOWN", "model": "UNKNOWN", "year": "9999",
            "narrative": ("I PURCHASED HAND CONTROLS CALLED PEDDLEMASTER. THE DEVICE ALLOWS "
                "HANDICAPPED PERSONS TO DEPRESS THE GAS AND BRAKE PEDALS USING THEIR HANDS. "
                "AFTER USING THIS DEVICE ABOUT ONCE A MONTH FOR 6 MONTHS, A TOTAL OF 6 USES, "
                "A THIN PLASTIC PIECE SNAPPED, MAKING ME LOSE CONTROL OF THE GAS PEDAL. THIS "
                "HAPPENED ON THE FREEWAY WITH MY CHILD IN THE CAR. I HAVE NOT USED THE DEVICE "
                "AGAIN OR TRIED TO REPAIR IT."),
            "nhtsa_crash": "N", "nhtsa_fire": "N"},
    }

    new_rows = []
    for cmplid, (primary, accept, is_hard, note, target_category) in ADJUDICATIONS.items():
        if cmplid in candidates:
            c = candidates[cmplid]
            make, model, year = c["make"], c["model"], c["year"]
            narrative = c["narrative"]
            nhtsa_crash, nhtsa_fire = c["nhtsa_crash"], c["nhtsa_fire"]
        else:
            e = extra_rows[cmplid]
            make, model, year = e["make"], e["model"], e["year"]
            narrative = e["narrative"]
            nhtsa_crash, nhtsa_fire = e["nhtsa_crash"], e["nhtsa_fire"]

        new_rows.append({
            "cmplid": cmplid,
            "make": make, "model": model, "year": year,
            "narrative": narrative,
            "accept": accept,
            "primary": primary,
            "truth_source": "adjudicated_v3_rare_coverage",
            "is_hard": is_hard,
            "adjudication_note": note,
            "review_verdict": "adjudicated",
            "review_note": note,
            "nhtsa_gold": target_category,
            "nhtsa_crash": nhtsa_crash,
            "nhtsa_fire": nhtsa_fire,
            "target_category_before_review": target_category,
            "confirmed_target_category": primary == target_category,
        })

    out = dict(v2)
    out["rows"] = v2["rows"] + new_rows
    out["n"] = len(out["rows"])
    out["n_hard"] = sum(r["is_hard"] for r in out["rows"])
    out["caveats"] = v2["caveats"] + [
        "v3 adds 53 deliberately-sampled rows (not random) targeting the 9 raw_component "
        "classes that had zero coverage in v2 -- see v3_expansion below and "
        "build_gold_eval_set_v3.py / build_rare_gold_expansion.py. Only 16/53 (30%) "
        "confirmed the NHTSA-hypothesized rare category as correct; treat any single "
        "class's gold accuracy here (n as low as 1-4 for some classes) as a directional "
        "signal, not a precise estimate.",
    ]
    n_confirmed = sum(r["confirmed_target_category"] for r in new_rows)
    per_category_confirmed = {}
    for r in new_rows:
        cat = r["target_category_before_review"]
        per_category_confirmed.setdefault(cat, [0, 0])
        per_category_confirmed[cat][1] += 1
        if r["confirmed_target_category"]:
            per_category_confirmed[cat][0] += 1
    out["v3_expansion"] = {
        "method": "53 rows deliberately sampled by literal COMPDESC_TOP match against the "
                   "9 zero-coverage classes (7 of 9 had recoverable, non-training-overlapping "
                   "examples; FIRERELATED and TRAILER HITCHES had zero real complaints left "
                   "in the corpus after training-data exclusion and could not be added "
                   "without train/eval leakage). Each row independently adjudicated from the "
                   "narrative against teacher_prompt.py's class definitions, not defaulted "
                   "to NHTSA's raw label.",
        "n_new_rows": len(new_rows),
        "n_confirmed_target_category": n_confirmed,
        "confirmation_rate": n_confirmed / len(new_rows),
        "per_category_confirmed_of_sampled": {
            k: f"{v[0]}/{v[1]}" for k, v in sorted(per_category_confirmed.items())
        },
        "uncoverable_classes": ["FIRERELATED", "TRAILER HITCHES"],
        "previous_n": v2["n"],
    }

    with open(OUT_DIR / "gold_eval_set_v3.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"gold_eval_set_v3.json: {out['n']} rows ({out['n_hard']} hard), +{len(new_rows)} new")
    print(f"Confirmed target category: {n_confirmed}/{len(new_rows)}")
    for k, v in sorted(per_category_confirmed.items()):
        print(f"  {k:<32} {v[0]}/{v[1]}")


if __name__ == "__main__":
    main()
