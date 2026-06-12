"""
Gate 4 geography projection regression tests.

Locks the corrected Annex 2 geography projection semantics:
  * RREL11 <- geographic_region_obligor   (NUTS3 code)
  * RREL12 <- geographic_region_classification (classification YEAR)
  * RREC6  <- geographic_region_collateral (NUTS3 code)
and proves the post-processing steps never overwrite these from readable
labels or collateral_geography, and never treat the classification year as a
region code.
"""

from __future__ import annotations

import pandas as pd

from engine.gate_4_projection.regime_projector import (
    apply_annex2_post_projection_guards,
    apply_esma_uk_geography_override,
    is_classification_year,
    rename_to_esma_codes,
)

# Minimal fields_list (canonical_name, regime_meta) for the geography fields.
_GEO_FIELDS = [
    ("geographic_region_obligor", {"code": "RREL11", "priority": "Analytics"}),
    ("geographic_region_classification", {"code": "RREL12", "priority": "Analytics"}),
    ("geographic_region_collateral", {"code": "RREC6", "priority": "Optional"}),
]


def _canonical_df():
    return pd.DataFrame({
        "geographic_region_obligor": ["TLG31", "TLF14"],
        "geographic_region_classification": ["2021", "2021"],
        "geographic_region_collateral": ["TLG31", "TLF14"],
        "collateral_geography": ["West Midlands", "East Midlands"],
    })


# --------------------------------------------------------------------------- #
# 1. Direct projection: select/rename keeps the correct sources
# --------------------------------------------------------------------------- #


def test_direct_projection_geography_codes():
    df = _canonical_df()[["geographic_region_obligor",
                          "geographic_region_classification",
                          "geographic_region_collateral"]].copy()
    out = rename_to_esma_codes(df, _GEO_FIELDS)
    assert list(out["RREL11"]) == ["TLG31", "TLF14"]
    assert list(out["RREL12"]) == ["2021", "2021"]
    assert list(out["RREC6"]) == ["TLG31", "TLF14"]


# --------------------------------------------------------------------------- #
# 2. Post-processing must not overwrite RREL11/RREL12/RREC6
# --------------------------------------------------------------------------- #


def test_post_projection_guards_do_not_touch_geography():
    # ESMA-coded frame with the mandatory fields the guards expect.
    df = pd.DataFrame({
        "RREL11": ["TLG31", "TLF14"],
        "RREL12": ["2021", "2021"],
        "RREC6": ["TLG31", "TLF14"],
        "RREL1": ["213800ABCDE123456701N202601", "213800ABCDE123456701N202601"],
        "RREL6": ["2026-01-31", "2026-01-31"],
        "RREL2": ["L1", "L2"], "RREL3": ["A1", "A2"], "RREL4": ["B1", "B2"],
        "RREL5": ["B1", "B2"], "RREC2": ["A1", "A2"], "RREC9": ["HOUS", "HOUS"],
    })
    config = {"regime_overrides": {"ESMA_Annex2": {
        "header_constants": {"RREL1": "213800ABCDE123456701N202601",
                             "RREL6": "2026-01-31"}}}}
    out, _ = apply_annex2_post_projection_guards(df.copy(), config, {})
    assert list(out["RREL11"]) == ["TLG31", "TLF14"]
    assert list(out["RREL12"]) == ["2021", "2021"]
    assert list(out["RREC6"]) == ["TLG31", "TLF14"]


def test_uk_geography_override_disabled_by_default():
    # No uk_geography config -> override is a no-op.
    df = _canonical_df()
    out, report = apply_esma_uk_geography_override(df.copy(), {}, "ESMA_Annex2")
    assert report["applied"] is False
    assert list(out["geographic_region_obligor"]) == ["TLG31", "TLF14"]
    assert list(out["geographic_region_classification"]) == ["2021", "2021"]


def test_uk_geography_override_never_clobbers_classification_year():
    # Even if misconfigured to target the classification YEAR, the override must
    # leave it intact (only region-CODE fields go to GBZZZ).
    df = _canonical_df()
    config = {
        "portfolio": {"country": "GB"},
        "regime_overrides": {"ESMA_Annex2": {"uk_geography": {
            "enabled": True,
            "override_value": "GBZZZ",
            "target_fields": ["geographic_region_obligor",
                              "geographic_region_classification",
                              "geographic_region_collateral"],
        }}},
    }
    out, report = apply_esma_uk_geography_override(df.copy(), config, "ESMA_Annex2")
    assert report["applied"] is True
    # Year field excluded from override and preserved.
    assert "geographic_region_classification" in report.get(
        "excluded_classification_year_fields", [])
    assert list(out["geographic_region_classification"]) == ["2021", "2021"]
    # Region-code fields overridden to GBZZZ (the configured behaviour).
    assert set(out["geographic_region_obligor"]) == {"GBZZZ"}
    assert set(out["geographic_region_collateral"]) == {"GBZZZ"}


# --------------------------------------------------------------------------- #
# 3. Semantic guard for RREL12
# --------------------------------------------------------------------------- #


def test_classification_year_semantic_guard():
    assert is_classification_year("2021") is True       # pass
    assert is_classification_year("2016") is True       # pass (configured year)
    assert is_classification_year("West Midlands") is False   # fail (label)
    assert is_classification_year("TLG31") is False     # fail (NUTS code)
    assert is_classification_year("GBZZZ") is False     # fail (region code)
    assert is_classification_year("ND1") is True        # valid no-data


def test_uk_override_preserves_granular_itl3():
    # The ESMA GBZZZ override targets only the regulatory region-code fields;
    # the granular ITL3 fields must survive untouched (FCA/MI drilldown).
    df = pd.DataFrame({
        "geographic_region_obligor": ["TLG31", "TLF14"],
        "geographic_region_collateral": ["TLG31", "TLF14"],
        "geographic_region_obligor_itl3": ["TLG31", "TLF14"],
        "geographic_region_collateral_itl3": ["TLG31", "TLF14"],
        "geographic_region_classification": ["2021", "2021"],
        "collateral_geography": ["West Midlands", "East Midlands"],
    })
    config = {
        "portfolio": {"country": "GB"},
        "regime_overrides": {"ESMA_Annex2": {"uk_geography": {
            "enabled": True, "override_value": "GBZZZ",
            "target_fields": ["geographic_region_obligor",
                              "geographic_region_collateral"],
        }}},
    }
    out, _ = apply_esma_uk_geography_override(df.copy(), config, "ESMA_Annex2")
    assert set(out["geographic_region_obligor"]) == {"GBZZZ"}
    assert set(out["geographic_region_collateral"]) == {"GBZZZ"}
    # ITL3 + classification + readable label untouched
    assert list(out["geographic_region_obligor_itl3"]) == ["TLG31", "TLF14"]
    assert list(out["geographic_region_collateral_itl3"]) == ["TLG31", "TLF14"]
    assert list(out["geographic_region_classification"]) == ["2021", "2021"]
    assert list(out["collateral_geography"]) == ["West Midlands", "East Midlands"]


# --------------------------------------------------------------------------- #
# 4. Delivery-time RREL12 validation (year/ND only)
# --------------------------------------------------------------------------- #


def test_delivery_rrel12_validator():
    from engine.gate_4b_delivery.annex2_delivery_normalizer import normalize_delivery

    rules = {"field_rules": {"RREL12": {
        "esma_code": "RREL12", "nd_allowed": ["ND1"],
        "validators": {"regex": "^(ND[0-9]|(19|20)[0-9]{2})$"},
    }}}

    def _ok(value):
        df = pd.DataFrame([{"RREL12": value}])
        _out, issues, _summary = normalize_delivery(df, rules)
        return not issues

    assert _ok("2021") is True
    assert _ok("2016") is True
    assert _ok("ND1") is True
    assert _ok("West Midlands") is False
    assert _ok("TLG31") is False
    assert _ok("GBZZZ") is False
