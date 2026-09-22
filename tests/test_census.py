"""Mix check from the monthly count table, no full history file."""
from prod.census import mix_from_census, overlay_months
from prod.nightly import _hash_fp, load_seen, save_seen


def test_hash_is_stable_and_not_the_narrative():
    fp = "123||the brake pedal went to the floor"
    h = _hash_fp(fp)
    assert h == _hash_fp(fp)
    assert "brake" not in h
    assert len(h) == 64


def test_seen_roundtrip(tmp_path):
    path = tmp_path / "seen.txt"
    keys = {_hash_fp("a"), _hash_fp("b")}
    save_seen(keys, path)
    assert load_seen(path) == keys


def test_overlay_replaces_those_months_only():
    base = {
        "202401": {"n": 10, "max_day": 31, "counts": {"AIR BAGS": 10}},
        "202501": {"n": 5, "max_day": 15, "counts": {"AIR BAGS": 5}},
    }
    incoming = {"202501": {"n": 20, "max_day": 31, "counts": {"STEERING": 20}}}
    merged = overlay_months(base, incoming)
    assert merged["202401"]["n"] == 10
    assert merged["202501"]["n"] == 20
    assert merged["202501"]["counts"] == {"STEERING": 20}


def test_two_complete_months_over_the_line_is_a_fire():
    # Pool months far from the walk, last-24 all identical, then two walk
    # months that are a different pile — enough to trip mean+2sd.
    months = {}
    for ym in [
        "200312", "200401", "200402", "200403", "200404", "200405",
        "200406", "200407", "200408", "200409", "200410", "200411",
        "200412", "200501", "200502", "200503", "200504", "200505",
        "200506", "200507", "200508", "200509", "200510", "200511",
        "200512", "200601", "200602", "200603", "200604", "200605",
        "200606", "200607", "200608", "200609", "200610", "200611",
        "202312", "202401", "202402", "202403", "202404", "202405",
        "202406", "202407", "202408", "202409", "202410", "202411",
        "202412", "202501", "202502", "202503", "202504", "202505",
        "202506", "202507", "202508", "202509", "202510", "202511",
    ]:
        months[ym] = {"n": 100, "max_day": 28, "counts": {"AIR BAGS": 100}}
    months["202512"] = {"n": 100, "max_day": 28, "counts": {"STEERING": 100}}
    months["202601"] = {"n": 100, "max_day": 28, "counts": {"STEERING": 100}}
    out = mix_from_census(months, last_retrain_ym="202511")
    assert out["first_fire"] is not None
    assert out["first_fire"]["month"] == "202601"
    assert out["first_fire"]["previous_month"] == "202512"


def test_partial_month_cannot_complete_a_fire():
    months = {}
    for ym in [
        "200312", "202312", "202401", "202402", "202403", "202404",
        "202405", "202406", "202407", "202408", "202409", "202410",
        "202411", "202412", "202501", "202502", "202503", "202504",
        "202505", "202506", "202507", "202508", "202509", "202510",
        "202511",
    ]:
        months[ym] = {"n": 100, "max_day": 28, "counts": {"AIR BAGS": 100}}
    # Need a full 24-month last-24 window; fill 2023-12 through 2025-11 already.
    # One complete trip then a partial trip should not fire.
    months["202512"] = {"n": 100, "max_day": 28, "counts": {"STEERING": 100}}
    months["202601"] = {"n": 40, "max_day": 12, "counts": {"STEERING": 40}}
    out = mix_from_census(months, last_retrain_ym="202511")
    assert out["first_fire"] is None
    assert any(r["month"] == "202601" and not r["complete"] for r in out["walk"])
