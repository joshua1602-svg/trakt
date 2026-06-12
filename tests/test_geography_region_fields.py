"""
Root-cause regression tests for the geographic region / classification fields.

ESMA Annex 2 template:
    RREL11 = Geographic Region - obligor       = NUTS3 region code
    RREL12 = Geographic Region Classification  = the YEAR of the NUTS3
             classification used for the geographic region fields ({YEAR})

So `geographic_region_classification` (RREL12) is a YEAR, not a region code.
These tests lock the corrected field-registry definitions.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = REPO_ROOT / "config" / "system" / "fields_registry.yaml"


def _fields() -> dict:
    return yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))["fields"]


def test_classification_is_year_not_nuts3():
    f = _fields()["geographic_region_classification"]
    # Corrected: it is a YEAR, not a NUTS3 region list.
    assert f["format"] == "year"
    assert f["allowed_values"] in (None, "null")
    assert f["allowed_values"] != "geographic_region_nuts3"
    assert f["format"] != "list"
    # Annex 2 code unchanged.
    assert f["regime_mapping"]["ESMA_Annex2"]["code"] == "RREL12"
    # Business meaning documented and makes the year-not-region intent explicit.
    assert "year" in (f.get("meaning") or "").lower()


def test_obligor_remains_nuts3_region_code():
    f = _fields()["geographic_region_obligor"]
    assert f["allowed_values"] == "geographic_region_nuts3"
    assert f["format"] == "list"
    assert f["regime_mapping"]["ESMA_Annex2"]["code"] == "RREL11"


def test_collateral_remains_nuts3_region_code():
    f = _fields()["geographic_region_collateral"]
    assert f["allowed_values"] == "geographic_region_nuts3"
    assert f["format"] == "list"
    assert f["regime_mapping"]["ESMA_Annex2"]["code"] == "RREC6"


def test_region_and_classification_are_distinct_concepts():
    f = _fields()
    obligor = f["geographic_region_obligor"]
    classification = f["geographic_region_classification"]
    # The region field is a NUTS3 list; the classification field is a year.
    assert obligor["format"] == "list"
    assert classification["format"] == "year"
    assert obligor["allowed_values"] != classification["allowed_values"]
