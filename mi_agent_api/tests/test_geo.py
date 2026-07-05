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
