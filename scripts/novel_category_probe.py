"""Shared set of synthetic 'novel category' narratives for the Rung 2/3 flexibility
probe (see PROJECT_PLAN.md's mode-collapse caveat and WRITEUP.md section 6).

Each of these describes a real-sounding vehicle complaint about a specific accessory
or system that is NOT one of the 40 valid `raw_component` categories in
teacher_prompt.ALL_VALID_COMPONENTS, and isn't a vague/no-system-named complaint either
(which is what `UNKNOWN OR OTHER` is actually defined for). A model that's learned the
schema well should ideally route these to a genuinely defensible catch-all (e.g.
`EQUIPMENT` for a specific-but-uncategorized accessory) or `UNKNOWN OR OTHER`, rather
than force-fitting a specific, unrelated primary category (e.g. predicting `ENGINE` for
a broken minibar). There's no hard "correct" answer scored here -- this is a
qualitative check for forced/nonsensical answers, not an accuracy number.
"""

NOVEL_CATEGORY_NARRATIVES = [
    {
        "cmplid": "novel_1", "make": "Toyota", "model": "Sienna", "year": "2024",
        "narrative": (
            "The built-in refrigerated console box between the second-row seats stopped "
            "cooling completely and now leaks water onto the floor mats every time we "
            "drive on a warm day."
        ),
    },
    {
        "cmplid": "novel_2", "make": "Ford", "model": "F-150", "year": "2023",
        "narrative": (
            "The power-retracting tonneau cover motor on my truck bed burned out and the "
            "cover is now stuck half-open. It will not respond to the remote or the "
            "switch inside the cab."
        ),
    },
    {
        "cmplid": "novel_3", "make": "Cadillac", "model": "Escalade", "year": "2025",
        "narrative": (
            "The heated and cooled cupholders in the center console started making a "
            "burning smell and one of them got hot enough to melt the plastic insert "
            "holding a water bottle."
        ),
    },
    {
        "cmplid": "novel_4", "make": "Honda", "model": "Odyssey", "year": "2024",
        "narrative": (
            "The built-in dash cam and cabin-watch camera system randomly reboots itself "
            "several times per drive and loses the last week of recorded footage every "
            "time it does."
        ),
    },
    {
        "cmplid": "novel_5", "make": "Rivian", "model": "R1S", "year": "2024",
        "narrative": (
            "The vehicle-to-home bidirectional charging feature stopped exporting power "
            "to my house during a planned outage test, even though the app said the "
            "feature was active and the battery was at 80 percent."
        ),
    },
]
