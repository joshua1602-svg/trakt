"""Geographic exposure by ITL3 area (the map's data layer)."""
from __future__ import annotations

import warnings

import pandas as pd

from mi_agent_api import geo


def test_exposure_derived_from_postcode_via_lookup():
    warnings.simplefilter("ignore")
    df = pd.DataFrame({
        "current_outstanding_balance": [500_000, 300_000, 200_000, 100_000, 250_000],
        # BS1* -> Bristol; AB10 -> Aberdeen; XX99 -> unmatched.
        "property_post_code": ["BS1 4DJ", "BS16 1QY", "AB10 1AB", "BS1 5TR", "XX99 9ZZ"],
    })
    out = geo.exposure_by_itl3(df)
    assert out["available"] is True
    assert out["basis"] == "postcode_derived"
    # 4 of 5 loans resolve (one unmatched postcode).
    assert out["coveragePct"] == 80.0
    top = out["areas"][0]
    assert top["itl3_code"] == "TLK51"
    assert "Bristol" in top["itl3_name"]
    assert top["count"] == 2  # BS1 4DJ + BS1 5TR (+ BS16 trims to BS1)
    # Areas are ranked by exposure and shares sum to ~100%.
    balances = [a["balance"] for a in out["areas"]]
    assert balances == sorted(balances, reverse=True)
    assert abs(sum(a["sharePct"] for a in out["areas"]) - 100.0) < 0.1


def test_prefers_tape_itl3_field_over_postcode():
    warnings.simplefilter("ignore")
    df = pd.DataFrame({
        "current_outstanding_balance": [100.0, 100.0],
        "geographic_region_collateral_itl3": ["TLK51", "TLM50"],
        "property_post_code": ["ZZ1 1ZZ", "ZZ2 2ZZ"],  # ignored when ITL3 present
    })
    out = geo.exposure_by_itl3(df)
    assert out["available"] is True
    assert out["basis"] == "collateral"
    assert out["resolvedFromItl3Field"] == 2
    assert {a["itl3_code"] for a in out["areas"]} == {"TLK51", "TLM50"}


def test_per_area_analytics_ticket_ltv_age():
    warnings.simplefilter("ignore")
    # Two loans in area A (LTV 40/60, ages 70/80), one in area B (LTV 50, age 75).
    df = pd.DataFrame({
        "current_outstanding_balance": [100_000.0, 300_000.0, 200_000.0],
        "current_loan_to_value": [40.0, 60.0, 50.0],
        "youngest_borrower_age": [70, 80, 75],
        "geographic_region_collateral_itl3": ["TLK51", "TLK51", "TLM50"],
    })
    out = geo.exposure_by_itl3(df)
    by = {a["itl3_code"]: a for a in out["areas"]}
    a = by["TLK51"]
    assert a["count"] == 2
    assert a["avgTicket"] == 200_000.0                    # (100k+300k)/2
    # Balance-weighted LTV = (40*100k + 60*300k)/400k = 55.0
    assert a["avgLtv"] == 55.0
    assert a["avgAge"] == 75.0                             # mean(70,80)
    b = by["TLM50"]
    assert b["avgTicket"] == 200_000.0 and b["avgLtv"] == 50.0 and b["avgAge"] == 75.0


def test_per_area_analytics_omitted_when_columns_absent():
    warnings.simplefilter("ignore")
    df = pd.DataFrame({
        "current_outstanding_balance": [100.0, 200.0],
        "geographic_region_collateral_itl3": ["TLK51", "TLM50"],
    })
    out = geo.exposure_by_itl3(df)
    area = out["areas"][0]
    assert area["avgTicket"] is not None      # ticket is always derivable
    assert "avgLtv" not in area and "avgAge" not in area


def test_mixed_itl3_field_and_postcode_fallback():
    # Row 0: ITL3 field present -> used directly. Row 1: blank field -> postcode
    # fallback. Row 2: sentinel 'nan' field -> postcode fallback. Row 3: blank
    # field + unmatched postcode -> unresolved.
    warnings.simplefilter("ignore")
    df = pd.DataFrame({
        "current_outstanding_balance": [400_000, 300_000, 200_000, 100_000],
        "geographic_region_collateral_itl3": ["TLM50", "", "nan", ""],
        "property_post_code": ["ZZ9 9ZZ", "BS1 4DJ", "BS1 5TR", "XX99 9ZZ"],
    })
    out = geo.exposure_by_itl3(df)
    assert out["available"] is True
    assert out["resolvedFromItl3Field"] == 1          # only row 0
    assert out["resolvedFromPostcode"] == 2           # rows 1 and 2 (both BS1)
    assert out["coveragePct"] == 75.0                 # 3 of 4 resolved
    codes = {a["itl3_code"] for a in out["areas"]}
    assert codes == {"TLM50", "TLK51"}                # Bristol group from postcodes
    bristol = next(a for a in out["areas"] if a["itl3_code"] == "TLK51")
    assert bristol["count"] == 2                       # BS1 4DJ + BS1 5TR


def test_unavailable_when_no_geography_on_tape():
    warnings.simplefilter("ignore")
    df = pd.DataFrame({"current_outstanding_balance": [100.0, 200.0]})
    out = geo.exposure_by_itl3(df)
    assert out["available"] is False
    assert out["areas"] == []


def test_outward_code_parsing():
    assert geo._outward_code("BS1 4DJ") == "BS1"
    assert geo._outward_code("bs14dj") == "BS1"       # unspaced -> strip last 3
    assert geo._outward_code("SW1A 1AA") == "SW1A"
    assert geo._outward_code("") == ""
