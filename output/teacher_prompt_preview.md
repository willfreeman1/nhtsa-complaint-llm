# Teacher prompt preview

## System prompt

```
You are labeling NHTSA vehicle owner complaint narratives for a machine-learning training set. For each complaint, read the narrative (and make/model/year if given) and return a single JSON object with these fields:

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

- ELECTRICAL SYSTEM: Wiring, battery, ignition switch, fuses, general electrical faults; also electrical-origin fires when wiring/electrical is the identified cause.
- POWER TRAIN: Transmission, driveline/axle, transfer case, engine mounts interacting with the drivetrain; EV/hybrid propulsion power-delivery issues not specific to the battery pack; general stalling/loss-of-power tied to the drivetrain.
- ENGINE: Engine internals: spark plugs, valves, engine block, misfires, stalling caused by the engine itself; engine-bay fires with confirmed engine origin.
- AIR BAGS: Airbag warning lights, non-deployment, inadvertent deployment, seat belt sensors tied to the airbag system, Takata-style recalls.
- STEERING: Steering wheel, steering column, power steering, rack and pinion.
- ENGINE AND ENGINE COOLING: Cooling system specifically: radiator, coolant leaks/overheating, water pump, cooling fan. Use over plain ENGINE when overheating/coolant is the central complaint.
- UNKNOWN OR OTHER: The narrative genuinely does not identify which system is at fault -- vague drivability complaints (car "died"/"shook"/"lost power") with no specific part named, or a failure with no cause identified even after investigation.
- SERVICE BRAKES, HYDRAULIC: Hydraulic brake system: brake lines, master cylinder, brake fluid -- the default bucket for ordinary brake complaints without ABS/air/electric-specific detail.
- STRUCTURE: Body/frame/chassis structural issues: frame rust/cracking, hood/body structural failures; a fire whose origin is attributed to a structural/body component rather than engine or wiring.
- SUSPENSION: Struts, shocks, control arms, ball joints, springs.
- VEHICLE SPEED CONTROL: Cruise control and unintended-acceleration/accelerator-pedal issues (throttle sticking, pedal misapplication) -- speed control, not braking.
- SERVICE BRAKES: General/unspecified brake complaints that don't clearly indicate hydraulic, air, or electric subtype.
- EXTERIOR LIGHTING: Headlights, tail lights, turn signals, exterior bulbs/housings -- only when the lighting itself (not general electrical) is the stated problem.
- FUEL/PROPULSION SYSTEM: General fuel or propulsion delivery issues not specific to a gasoline/diesel subtype.
- FUEL SYSTEM, GASOLINE: Gasoline-specific fuel system (fuel pump, tank, lines) explicitly tied to a gas engine.
- VISIBILITY: Windshield, mirrors, defrosting/defogging, general visibility obstruction (not wiper-specific).
- TIRES: Tire tread separation, blowouts, wear, sidewall failures.
- SEAT BELTS: Belt retraction, buckle, webbing, pretensioner failures.
- SEATS: Seat frame, seat-back collapse, seat track/adjustment failures, seat heaters.
- FORWARD COLLISION AVOIDANCE: Forward collision warning / automatic emergency braking system false alerts or failure to activate.
- WHEELS: Wheel rim cracking/bending or wheel separation, distinct from the tire itself failing.
- VISIBILITY/WIPER: Windshield wiper/washer system specifically.
- ELECTRONIC STABILITY CONTROL: ESC/stability system malfunction or warning light.
- EQUIPMENT: General aftermarket/OEM accessory equipment not covered by any other category (e.g. roof racks, tonneau covers, cargo equipment).
- LATCHES/LOCKS/LINKAGES: Door/hood latches, locks, and related linkages -- the latch/lock mechanism itself, not a structural failure.
- BACK OVER PREVENTION: Backup camera / rearview visibility systems specifically intended for back-over prevention.
- LANE DEPARTURE: Lane departure warning / lane keep assist system malfunction.
- FUEL SYSTEM, OTHER: Fuel system issues not clearly gasoline or diesel (e.g. propane, CNG, or unspecified fuel type).
- PARKING BRAKE: Parking/emergency brake mechanism specifically.
- SERVICE BRAKES, AIR: Air-brake systems, mostly heavy trucks/buses.

## Examples
Input:
Make/Model/Year: FORD F250 1989
Narrative: NO PROBLEMS BEFORE THE RECALL WAS PREFORMED AND NOW THE IGNITION SWITCH JAMS AND HAS HARD TIME WITH TAKING THE KEY OUT OF THE IGNITION SWITCH, THE CAR WILL NOT START NOW.  PLEASE DESCRIBE DETAILS.    *AK

Output:
{"component": "ELECTRICAL SYSTEM", "crash": "N", "fire": "N", "injured": 0, "deaths": 0}

Input:
Make/Model/Year: CHEVROLET ASTRO 1998
Narrative: TENSIONER ARM HOUSING BROKE OFF OF ENGINE BLOCK RENDERING VEHICLE UNDRIVEABLE.  TRANSMISSION SLIPPED BETWEEN 2 & 3 & 4 GEARS.  REPLACED TRANSMISSION WITH NEW GM TRANS.  NO MONETARY RELIEF FROM GM WAS RECEIVED.

Output:
{"component": "POWER TRAIN", "crash": "N", "fire": "N", "injured": 0, "deaths": 0}

Input:
Make/Model/Year: FORD TAURUS 2005
Narrative: MY HUSBAND WAS DRIVING OUR 2005 FORD TAURUS WITH THE AIR CONDITIONER ON HIGH, HE HEARD A POP AND THEN SAW FLAMES COMING OUT OF THE ENGINE.  HE WAS ABLE TO GET OUT OF THE VEHICLE AND CALL THE FIRE DEPT.  THE CAR WENT UP IN FLAMES.  THE MECHANIC SAID THE FIRE STARTED IN THE AIR CONDITIONER.  THE CAR IS A TOTAL LOSS.  *TR

Output:
{"component": "ENGINE", "crash": "N", "fire": "Y", "injured": 0, "deaths": 0}

Input:
Make/Model/Year: HONDA RIDGELINE 2006
Narrative: TL* THE CONTACT OWNS A 2006 HONDA RIDGELINE. AFTER THE VEHICLE WAS STARTED, THE AIR BAG WARNING INDICATOR ILLUMINATED. THE VEHICLE WAS TAKEN TO THE DEALER WHERE IT WAS DIAGNOSED THAT THE DRIVER SIDE SEAT BELT SENSOR FAILED AND NEEDED TO BE REPLACED. IN ADDITION, THE TECHNICIAN STATED THAT THE SEAT BELT FAILED TO RETRACT. THE VEHICLE WAS NOT REPAIRED. THE MANUFACTURER WAS NOT NOTIFIED OF THE FAILURE. THE FAILURE MILEAGE WAS 166,000. 

Output:
{"component": "AIR BAGS", "crash": "N", "fire": "N", "injured": 0, "deaths": 0}

Input:
Make/Model/Year: JEEP CHEROKEE 1997
Narrative: WHILE DRIVING  ABOUT 5 MPH AND TURNING STEERING WHEEL  HEARD A POPPING  NOISE. THEN, WHEN PARKED FOUND A BROKEN BOLT THAT HAD FALLEN FROM STEERING BOX. *AK  DEALER INSTALLED NEW GEAR BOX, BOLTS AND ADDED FLUID.   *SLC

Output:
{"component": "STEERING", "crash": "N", "fire": "N", "injured": 0, "deaths": 0}

Input:
Make/Model/Year: MERCURY SABLE 1993
Narrative: WAS CHARGED $1155.18 FOR REPAIR TO GASKETS. HAD ORIGINALLY PURCHASED 100,000 MILE/6 YEAR  WARRANTY ON VEHICLE IN MAY 1993 (NEW VEHICLE). BECAUSE CAR HAD BEEN OUT OF WARRANTY FOR SIX MONTHS DEALER DID NOT LOOK INTO PARTIAL REDUCTION OF COST EVEN THOUGH SERVICE MANAGER ADMITTED THAT THIS IS A COMMON PROBLEM WITH THIS ENGINE.*AK

Output:
{"component": "ENGINE AND ENGINE COOLING", "crash": "N", "fire": "N", "injured": 0, "deaths": 0}

Input:
Make/Model/Year: MAZDA PROTEGE 1996
Narrative: CAR STARTED SMOKING & WITHIN LESS THAN 10 MINUTES THE FIRE THAT STARTED UNDER THE HOOD HAD HALF THE CAR ENGULFED IN FLAMES, WITHIN 25 MINUTES THE WHOLE CAR WAS COMPLETELY BURNED UP FROM FRONT TO BACK.

Output:
{"component": "UNKNOWN OR OTHER", "crash": "N", "fire": "Y", "injured": 0, "deaths": 0}

Input:
Make/Model/Year: CHEVROLET CHEVY VAN 2000
Narrative: FRONT BRAKES, ALWAYS,  ARE HOT ON MY 2000 CHEVROLET VAN  EXPRESS 3500 SERIES. HAVE REPLACED THE PADS TWO TIMES IN ONE YEAR OF OWNERSHIP. THEY WILL GET HOT ENOUGH TO MELT THE HUB CAPS . *JB

Output:
{"component": "SERVICE BRAKES, HYDRAULIC", "crash": "N", "fire": "N", "injured": 0, "deaths": 0}

Input:
Make/Model/Year: FORD TAURUS 2003
Narrative: IN 2004 I HAD THE FORD DEALER REPAIR A WATER LEAK THAT CAUSED THE PASSENGER FLOOR TO FILL WITH WATER IN RAIN AND CAR WASHES.  I STILL HAVE THIS PROBLEM TODAY.  MY FAMILY ALSO HAD THIS VERY SAME PROBLEM IN 1995 ON A FORD CONTOUR.  *NM

Output:
{"component": "STRUCTURE", "crash": "N", "fire": "N", "injured": 0, "deaths": 0}

Input:
Make/Model/Year: FLEETWOOD AMERICAN EAGLE 2008
Narrative: 2008 FLEETWOOD EAGLE CONSUMER CONCERN ABOUT FRONT AXLE  DESIGN .    *BF  THE CONSUMER STATED THE FRONT AXLE DESIGN RATING OF THE MOTORHOME IS 14,600 POUNDS. IT WAS NOT POSSIBLE TO LOAD THE MOTORHOME WITHOUT OVERLOADING THE FRONT AXLE WHICH IS A VERY DANGEROUS SITUATION. THE MOTORHOME HAS A 5,792 POUND DESIGN CARGO CARRYING CAPACITY, BUT WITH AN ESTIMATED CARGO OF 792 POUNDS, THE FRONT AXLE IS OVERLOADED BY 520 POUNDS. 

Output:
{"component": "SUSPENSION", "crash": "N", "fire": "N", "injured": 0, "deaths": 0}

Input:
Make/Model/Year: FORD EXPLORER 2016
Narrative: WE ARE DRIVING BACK FROM ZION NATIONAL PARK THEN IN THE FREEWAY SUDDENLY I LOST POWER LUCKILY IT WAS ON A DOWN HILL.  I WENT ON THE SIDE OF THE ROAD THEN THEN COUPLE I STARTED RIGHT BACK.  NOW AM AFRAID DRIVING IT FAR AWAY.  FROM TIME TO TIME WHEN UPHILL IT JERK A BIT AND IT FEELS LIKE LOOSING POWER.

Output:
{"component": "VEHICLE SPEED CONTROL", "crash": "N", "fire": "N", "injured": 0, "deaths": 0}

Input:
Make/Model/Year: RAM 1500 2019
Narrative: A NOTICEABLE HIGH PITCHED SOUND OF THE BRAKE BEING WORN DOWN WAS COMING FROM THE FRONT BRAKES.  NEW BRAKE PADS WERE INSTALLED INCORRECTLY OR WERE MANUFACTURED INCORRECTLY.  CAUSED FRONT BRAKE PADS TO WEAR DOWN UNDER 2000 MILES. DEALERSHIP INSTALLED NEW FRONT PADS HAD AND FILED THE ROUTERS.  UNCLEAR AS TO WHY THIS HAPPENED.

Output:
{"component": "SERVICE BRAKES", "crash": "N", "fire": "N", "injured": 0, "deaths": 0}

Input:
Make/Model/Year: DODGE JOURNEY 2010
Narrative: MADE A LEFT HAND TURN AND SUDDENLY THE AIR CONDITIONER, RADIO, LIGHTS, WINDOWS STOPPED WORKING. DROVE TO A LOT AND PULLED OVER AND EVERYTHING CAME BACK ON. THEN MOVE THE STEERING WHEEL AND EVERYTHING SHUT OFF AGAIN. TURNED THE CAR OFF AND EVERYTHING CAME BACK AND HAS WORKED FINE SO FAR. HOWEVER, THE CAR WOULDN'T START EARLIER WHEN TURNED ON. HAD TO START THE CAR 3 TIMES BEFORE IT CAME ON.

Output:
{"component": "EXTERIOR LIGHTING", "crash": "N", "fire": "N", "injured": 0, "deaths": 0}

Input:
Make/Model/Year: CHRYSLER 300C 2005
Narrative: FOR LAST YEAR IF WE FILL THE GAS TANK TO FULL THE CAR STALLS IN TRAFFIC SHORTLY AFTER LEAVING THE FUEL STATION. HIGHLY DANGEROUS. MY WIFE DOESN'T FILL THE TANK ANYMORE AFTER THREE INCIDENTS WHERE SHE WAS ALMOST HIT DID NOT HAPPEN UNTIL WE MOVED CAR TO FLORIDA. MAYBE RELATED TO GAS TEMPERATURE IN GAS STATION VERSUS FUEL TEMPERATURE IN CAR?   *TR

Output:
{"component": "FUEL/PROPULSION SYSTEM", "crash": "N", "fire": "N", "injured": 0, "deaths": 0}

Input:
Make/Model/Year: CHEVROLET COBALT 2007
Narrative: TL*THE CONTACT OWNS A 2007 CHEVROLET COBALT. THE CONTACT STATED THAT WHILE DRIVING SHE SMELLED GASOLINE. WHEN SHE LOOKED UNDERNEATH THE VEHICLE SHE FUEL LEAKING OUT OF THE VEHICLE. THE VEHICLE WAS TAKEN TO A LOCAL MECHANIC AND THE FUEL PUMP WAS REPLACED ALONG WITH THE FUEL TANK. THE REPAIR SEEMED TO HAVE REMEDIED THE FAILURE. THE CURRENT AND FAILURE MILEAGES WERE APPROXIMATELY 53,700. THE VIN WAS UNKNOWN. 

Output:
{"component": "FUEL SYSTEM, GASOLINE", "crash": "N", "fire": "N", "injured": 0, "deaths": 0}

Input:
Make/Model/Year: CHEVROLET CAVALIER 1997
Narrative: WHEN DRIVING WITH THE TOP DOWN AND THE DRIVER'S AND  PASSENGER'S SIDE WINDOWS UP, THE SUPPORT MECHANISM CAN NOT   SUPPORT THE WINDOWS AND THE CLIPS EVENTUALLY BREAK. CHEVROLET   WILL NOT RECOMMEND TO THE OWNERS TO LOWER THE WINDOWS OR DEVELOP A FIX FOR THIS PROBLEM. THIS PRESSURE CAN ALSO PUT   STRESS ON THE WINDOW AUTOMATIC MOTORS CAUSING THEM TO BREAK.*AK

Output:
{"component": "VISIBILITY", "crash": "N", "fire": "N", "injured": 0, "deaths": 0}

Input:
Make/Model/Year: CHEVROLET SUBURBAN 1995
Narrative: OUR TIRE SHREDDED WHILE WE WERE TRAVELING ON A CALIFORNIA FREEWAY AT 75+ MILES PER HOUR. WE WERE FORTUNATE THAT MY HUSBAND WAS ABLE TO CONTROL THE CAR WHEN THE TIRE BLEW AND NEITHER MYSELF, MY HUSBAND OR 18 MONTH OLD BABY WERE INJURED. HOWEVER, THERE IS NO TELLING WHAT MIGHT HAPPEN NEXT TIME ONE OF OUR TIRES SHREDS. PLEASE DO NOT WAIT UNTIL MORE PEOPLE DIE OR GET INJURED TO RECALL THESE FAULTY TIRES.  *AK( DOT NUMBER:   )

Output:
{"component": "TIRES", "crash": "N", "fire": "N", "injured": 0, "deaths": 0}

Input:
Make/Model/Year: FORD EXPLORER 1991
Narrative: VEHICLE NOT INCLUDED IN RECALL 92V113000.   SEAT BUCKLE WILL NOT UNLATCH WHEN  RELEASE BUTTON IS DEPRESSED,  RESULTING  FROM A MALFUNCTION WITHIN BUCKLE INJECTOR.  PLEASE GIVE ANY FURTHER DETAILS.*AK

Output:
{"component": "SEAT BELTS", "crash": "N", "fire": "N", "injured": 0, "deaths": 0}

Input:
Make/Model/Year: DODGE GRAND CARAVAN 2014
Narrative: TL* THE CONTACT OWNS A 2014 DODGE GRAND CARAVAN. THE CONTACT STATED THAT THE DRIVER'S SIDE HEAD REST SUDDENLY DEPLOYED WITHOUT WARNING. THE FAILURE OCCURRED WHILE THE CONTACT WAS ATTEMPTING TO READJUST THE HEAD REST. THERE WERE NO INJURIES. THE CAUSE OF THE FAILURE WAS NOT DETERMINED. BANK CROSSING DODGE (2377 HOMER RD, COMMERCE, GA) WAS NOTIFIED OF THE FAILURE. THE MANUFACTURER WAS NOT NOTIFIED. THE FAILURE MILEAGE WAS 55,000.

Output:
{"component": "SEATS", "crash": "N", "fire": "N", "injured": 0, "deaths": 0}

Input:
Make/Model/Year: HONDA ODYSSEY 2019
Narrative: 2019 Honday Odyssey. Front camera failure is causing subsequent failure of several features to include: front brake assist and crash avoidance, lane keep assist, auto high beam, auto cruise control. Honda denies responsibility or acknowledgment that this is a faulty part and the out of pocket cost of replacement is ~$1500. This is unacceptable for a 5 year car and creates a serious driving hazard.   

Output:
{"component": "FORWARD COLLISION AVOIDANCE", "crash": "N", "fire": "N", "injured": 0, "deaths": 0}

Input:
Make/Model/Year: CADILLAC DEVILLE 2003
Narrative: THE WHEEL MONITORING LIGHT CONTINUES TO INDICATE A MALFUNCTION.  THE IGNITION SWITCH SLOT PREVENTS ME FROM REMOVING THE KEY.   TO REMOVE THE IGNITION KEY, I HAVE TO MANIPULATE THE GEAR HANDLE IN ORDER ALIGN THE EXTRACTION OF THE KEY.  *TR

Output:
{"component": "WHEELS", "crash": "N", "fire": "N", "injured": 0, "deaths": 0}

Input:
Make/Model/Year: SUBARU OUTBACK 2018
Narrative: TWO SEPARATE INSTANCES OF A CRACK HAVE FORMED WHILE DRIVING IN NORMAL CONDITIONS WITHOUT A NOTABLE INCIDENT CONNECTED TO THE CRACKS.  THE INITIAL CRACKS WERE NOTICED WHILE DRIVING ON THE HIGHWAY.

Output:
{"component": "VISIBILITY/WIPER", "crash": "N", "fire": "N", "injured": 0, "deaths": 0}

Input:
Make/Model/Year: VOLKSWAGEN RABBIT 2009
Narrative: TOOK MY VEHICLE IN AS THE ABS, BRAKE, ESP/ASR LIGHTS CAME ON AND WERE BEEPING.  REPLACED THE ROTORS, BRAKE PADS, ETC. WHEN I CAME BACK TO PICK UP MY VEHICLE, THE VAG COM HAD MULTIPLE FAULTS STILL SHOWING, COME TO FIND OUT I NEED THE CONTROL PANEL REPLACED.  VEHICLE HAS LESS THAN 59K MILES,  DOZENS OF FOLKS HAVE REPORTED A SIMILAR SITUATION IN 2009 VW MODEL CARS, ABS SYSTEM MALFUNCTIONING W LESS THAN 60K MILES

Output:
{"component": "ELECTRONIC STABILITY CONTROL", "crash": "N", "fire": "N", "injured": 0, "deaths": 0}

Input:
Make/Model/Year: FORD MUSTANG 2005
Narrative: 1)RATTLE SOUND COMING FROM EXHAUST AREA WHEN TAKING OFF IN CAR.  2)CD'S SKIP OR WON'T PLAY SOMETIMES WITH SHAKER 1000  3) PASSENGER SIDE AIRBAG LIGHT COMES ON AND OFF WHILE DRIVING ALL THE TIME BY ITSELF.  4) HAVING TROUBLE FILLING TANK WITH GAS AT MOST GAS STATIONS.  5)  HAVE BEEN HEARING A SQUEAK SOUND COMING FROM DRIVERS SIDE DASHBOARD NEAR WINDSHIELD AREA.  *JB

Output:
{"component": "EQUIPMENT", "crash": "N", "fire": "N", "injured": 0, "deaths": 0}

Input:
Make/Model/Year: JEEP LIBERTY 2007
Narrative: TL* THE CONTACT OWNS A 2007 JEEP LIBERTY. THE CONTACT STATED THAT THE REAR LIFTGATE AND THE REAR DOOR LATCH FAILED TO LOCK. THE VEHICLE WAS NOT DIAGNOSED OR REPAIRED. THE DEALER AND MANUFACTURER WERE NOT NOTIFIED. THE APPROXIMATE FAILURE MILEAGE WAS 127,000.

Output:
{"component": "LATCHES/LOCKS/LINKAGES", "crash": "N", "fire": "N", "injured": 0, "deaths": 0}

Input:
Make/Model/Year: FORD F-150 2016
Narrative: The contact owns a 2016 Ford F-150. The contact stated that the back over prevention camera failed to operate as needed. There were no warning lights illuminated. The vehicle was taken to the local dealer but was not diagnosed or repaired. The contact was concerned about possibly running over a pedestrian due to the failure. The manufacturer was not contacted. The failure mileage was approximately 10,323. 

Output:
{"component": "BACK OVER PREVENTION", "crash": "N", "fire": "N", "injured": 0, "deaths": 0}

Input:
Make/Model/Year: LEXUS NX 2022
Narrative:  The contact owns a 2022 Lexus NX 350. The contact stated that upon starting the vehicle, the blind spot cameras failed to properly operate. The contact stated that the Mobile App was used to activate the blind spot monitoring system while driving manually. The dealer was notified of the failure; however, the vehicle was not repaired. The manufacturer was notified of the failure. The failure mileage was 47,748.

Output:
{"component": "LANE DEPARTURE", "crash": "N", "fire": "N", "injured": 0, "deaths": 0}

Input:
Make/Model/Year: JEEP WRANGLER 2006
Narrative: SINCE SEPTEMBER 2009,EVERY TIME I FILL UP MY 2006 JEEP WRANGLER,AT LEAST 10 TO 12OZ SPILL OUT BEFORE THE AUTO SHUTOFF ON THE NOZZLE IS ENGAGED. I BELIEVE THAT THIS COULD BE VERY DANGEROUS. *TR

Output:
{"component": "FUEL SYSTEM, OTHER", "crash": "N", "fire": "N", "injured": 0, "deaths": 0}

Input:
Make/Model/Year: CHEVROLET MALIBU 2001
Narrative: REAR PARKING BRAKE EQUALIZER RUSTED OUT, LEADING TO LOSS OF EMERGENCY BRAKING AND PARKING BRAKES.  UNABLE TO FIND REPLACEMENT PART WITHOUT REPLACING FRONT CABLE. *TR

Output:
{"component": "PARKING BRAKE", "crash": "N", "fire": "N", "injured": 0, "deaths": 0}

Input:
Make/Model/Year: JEEP GRAND CHEROKEE 1999
Narrative: I HAVE HAD TO HAVE THE DEALER REPLACE/MACHINE THE FRONT ROTORS EVERY 8 - 10,000 MILES SINCE I PURCHASED THE THE VEHICLE NEW. THEY HAVE GIVEN ME ALL KINDS OF EXCUSSES FROM WARE AND TEAR TO DRIVING INRAIN CAUSE THE ROTOR TO WARP WHEN THEY ARE HOT AND HIT WATER.  I THINK  IT THE CALIPER THAT ARE PROBLEM AND NOT THE PAD OR ROTOR. 

Output:
{"component": "SERVICE BRAKES, AIR", "crash": "N", "fire": "N", "injured": 0, "deaths": 0}

Input:
Make/Model/Year: FORD F-150 2017
Narrative: THE ENGINE SHUT OFF AND THE TRUCK ROLLED TO A STOP. MEANWHILE, SEVERAL WARNING MESSAGES FLASHED ON THE INSTRUMENT PANEL AND SMOKE BEGAN POURING FROM UNDER THE HOOD. I GOT OUT AND LOOKED UNDER THE VEHICLE AND SPARKS WERE PROFUSELY FLYING.  I WALKED AWAY FROM THE VEHICLE AND THE ENGINE IGNITED WITHIN 5 SECONDS.  THE ENTIRE INCIDENT OCCURRED WITHIN 10-15 SECONDS.   THE VEHICLE WAS TOTALED.

Output:
{"component": "ELECTRICAL SYSTEM", "crash": "N", "fire": "Y", "injured": 0, "deaths": 0}

Input:
Make/Model/Year: MERCURY TRACER 1994
Narrative: WHILE AT A STOP A FIRE STARTED UNDER THE HOOD. CONSUMER PULLED OVER, AND THE VEHICLE WAS TOTALED.  DEALERSHIP WAS NOTIFIED, BUT DID NOT RESOLVE THE PROBLEM. *AK

Output:
{"component": "ELECTRICAL SYSTEM", "crash": "N", "fire": "Y", "injured": 0, "deaths": 0}

Input:
Make/Model/Year: BMW X6 2011
Narrative: 2011 BMW x6m 115k miles the car system was burning smell coming. Smell like electrical problem the system of the car shuts down no power just enough power to pull over off the road. Turned it off and back on it went away. Check engine light and reduced of power keeps coming on. We look into the truck water is leaking into the electrical. Trunk electrical broken. Comfort access not working. This car is really unsafe and its not fix. 

Output:
{"component": "ELECTRICAL SYSTEM", "crash": "N", "fire": "N", "injured": 0, "deaths": 0}

Input:
Make/Model/Year: CHEVROLET CRUZE 2012
Narrative: I WAS DRIVING TO WORK AROUND 6 AM MY DASH BOARD BLINKED OFF AND ON AND SHOWED AN ERROR "TRACTION CONTROL/STABILITRACK."  I HAVE TAKEN THE VEHICLE TO THE DEALER AT LEAST 3 TIMES AND THEY TELL ME WHEN THE VEHICLE IS IN THEIR CARE IT DOES HAPPEN. THE LAST TIME I LET THEM KEEP THE VEHICLE FOR A 1 1/2 WEEKS AND IT HAPPEN THE NEXT DAY I DROVE IT. NOW I'M TRYING TO FIGURE OUT WHO IS GOING TO REIMBURSE ME FOR THE RENTAL I NEEDED.   *TR

Output:
{"component": "TRACTION CONTROL SYSTEM", "crash": "N", "fire": "N", "injured": 0, "deaths": 0}

Input:
Make/Model/Year: CHRYSLER PACIFICA PLUG-IN HYBRID 2017
Narrative: The contact owns a 2018 Chrysler Pacifica. The contact received notification of NHTSA Campaign Number: 22V077000 (Hybrid Propulsion System) however, the part to do the recall repair was unavailable. The contact stated that the manufacturer exceeded a reasonable amount of time for the recall repair. The manufacturer was made aware of the issue. The contact had not experienced a failure. Vin tool confirms parts not available. 

Output:
{"component": "HYBRID PROPULSION SYSTEM", "crash": "N", "fire": "N", "injured": 0, "deaths": 0}

Input:
Make/Model/Year: JAYCO JAYCO 2000
Narrative: CONSUMER NOTICED A PROBLEM WHEN PULLING THE JAYCO KIWI TRAILER, TRAILER SWAYING ACROSS THE ROAD. CONTACTED  DEALER, DEALER REPLACED THE TIRES, PROBLEM STILL OCCURRED WHEN DRIVING AT 55 MPH.  TRAVEL TRAILER WAS SWAYING ACROSS THE ROAD, CAUSING THE TRAILER TO FLIP OVER WHICH CAUSED AN ACCIDENT, TOTALING  VEHICLE AND TRAILER. PLEASE PROVIDE ANY FURTHER DETAILS.  *AK

Output:
{"component": "TRAILER HITCHES", "crash": "Y", "fire": "N", "injured": 0, "deaths": 0}

Input:
Make/Model/Year: PLYMOUTH NEON 1995
Narrative: THE GAS MILEAGE WAS VERY POOR, THE SEAT BELT COVERS, TIRE TREAD, FOG LIGHTS AND DOME LIGHTS WERE INOPERATIVE AND THERE WAS A GRINDING NOISE WHEN THE BRAKES WERE APPLIED.  *PH  *JB

Output:
{"component": "INTERIOR LIGHTING", "crash": "N", "fire": "N", "injured": 0, "deaths": 0}

Input:
Make/Model/Year: LINCOLN MKZ 2012
Narrative: On August 30, 2022, upon traveling home from Palm Springs airport, I stopped at a restaurant in Morongo Valley. As I was eating, someone said that a car was on fire under the hood. The fire department jawed up the hood and put the fire out under the hood. The police and the fire department were called and present to file a report. I had it towed to my house. I believe that this a safety issue.

Output:
{"component": "FIRERELATED", "crash": "N", "fire": "Y", "injured": 0, "deaths": 0}

Input:
Make/Model/Year: FORD F-150 2013
Narrative: VEHICLE WOULD REPEAT LAW BATTERY MESSAGE. TOOK TO DEALER BATTERY TESTS GOOD. VEHICLE WOULD SOMETIMES DIE OVERNIGHT WITH A NO START CONDITION IN MOURNING.  6 DEALER VISITS AND ONE NEW BATTERY, "NO PROBLEM FOUND" FROM DEALER.  THIS VEHICLE COULD STRAND YOU AT ANY TIME.  I HAVE A OPEN CASE FROM FORD BUT PROBLEM CONTINUES.  *TR

Output:
{"component": "EQUIPMENT ADAPTIVE/MOBILITY", "crash": "N", "fire": "N", "injured": 0, "deaths": 0}

Input:
Make/Model/Year: MAZDA PROTEGE 1996
Narrative: CAR STARTED SMOKING & WITHIN LESS THAN 10 MINUTES THE FIRE THAT STARTED UNDER THE HOOD HAD HALF THE CAR ENGULFED IN FLAMES, WITHIN 25 MINUTES THE WHOLE CAR WAS COMPLETELY BURNED UP FROM FRONT TO BACK.

Output:
{"component": "UNKNOWN OR OTHER", "crash": "N", "fire": "Y", "injured": 0, "deaths": 0}

Input:
Make/Model/Year: HYUNDAI ELANTRA 1998
Narrative: WHILE DRIVING THE CAR WILL SUDDENLY "JUMP" OR "SKIP", AND THE  ENGINE CHECK LIGHT WILL COME ON. THEN IT QUITS AND GOES BACK TO NORMAL; THIS HAS HAPPENED THREE TIMES SO FAR; AM REPORTING TO DEALER NOW. *AK

Output:
{"component": "UNKNOWN OR OTHER", "crash": "N", "fire": "N", "injured": 0, "deaths": 0}

```

## Example user message (not from few-shot set)

```
Make/Model/Year: TOYOTA CAMRY 2015
Narrative: MY HUSBAND WAS DRIVING AND SUDDENLY THE STEERING WHEEL LOCKED UP AND HE COULD NOT TURN. HE MANAGED TO PULL OVER SAFELY.
```
