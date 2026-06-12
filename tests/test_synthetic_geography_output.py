"""
Geography data-production regression tests.

Verifies that generated synthetic canonical output and the Annex 2 delivery
mapping route geography correctly:
  * geographic_region_classification = NUTS classification YEAR (or ND1)
  * geographic_region_obligor / _collateral = NUTS3-like codes (or ND1)
  * readable region labels live in collateral_geography (analytics display)
  * RREL11 <- geographic_region_obligor, RREL12 <- geographic_region_classification
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTH_CSV = (REPO_ROOT / "synthetic_demo" / "output"
             / "SYNTHETIC_ERE_Portfolio_012026_canonical_typed.csv")
DELIVERY_RULES = REPO_ROOT / "config" / "regime" / "annex2_delivery_rules.yaml"

_NUTS_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{1,4}$")
_READABLE_REGIONS = {"london", "south east", "south west", "west midlands",
                     "east midlands", "wales", "scotland",
                     "yorkshire and the humber", "north west", "east of england"}


def _is_year_or_nd(v: str) -> bool:
    s = str(v).strip()
    return bool(re.match(r"^(19|20)\d{2}$", s)) or s.upper().startswith("ND")


def _is_nuts_or_nd(v: str) -> bool:
    s = str(v).strip()
    return bool(_NUTS_RE.match(s.upper())) or s.upper().startswith("ND")


@pytest.fixture(scope="module")
def synth():
    if not SYNTH_CSV.exists():
        pytest.skip("synthetic canonical output not present")
    return pd.read_csv(SYNTH_CSV)


# --------------------------------------------------------------------------- #
# Synthetic output
# --------------------------------------------------------------------------- #


def test_classification_is_year_or_nd_not_region_label(synth):
    vals = synth["geographic_region_classification"].dropna().astype(str)
    assert len(vals) > 0
    assert all(_is_year_or_nd(v) for v in vals.unique()), vals.unique()[:5]
    # explicitly: no readable region label leaked into the year field
    assert not any(v.strip().lower() in _READABLE_REGIONS for v in vals)


def test_regulatory_region_fields_are_nuts_or_nd(synth):
    for col in ("geographic_region_obligor", "geographic_region_collateral"):
        if col not in synth.columns:
            continue
        vals = synth[col].dropna().astype(str)
        assert all(_is_nuts_or_nd(v) for v in vals.unique()), (col, vals.unique()[:5])
        assert not any(v.strip().lower() in _READABLE_REGIONS for v in vals)


def test_readable_labels_live_in_analytics_geography(synth):
    assert "collateral_geography" in synth.columns
    vals = set(synth["collateral_geography"].dropna().astype(str).str.lower())
    assert vals & _READABLE_REGIONS, "expected readable region labels in collateral_geography"


def test_classification_source_is_configured_year(synth):
    if "geographic_region_classification_source" in synth.columns:
        srcs = set(synth["geographic_region_classification_source"].dropna().astype(str))
        assert srcs == {"configured_nuts_classification_year"} or not srcs


# --------------------------------------------------------------------------- #
# Annex 2 delivery mapping
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def delivery_rules():
    return yaml.safe_load(DELIVERY_RULES.read_text(encoding="utf-8"))


def _rule(rules: dict, code: str) -> dict:
    # rules file is a mapping of esma_code -> rule (under some top-level key)
    for v in rules.values():
        if isinstance(v, dict) and code in v and isinstance(v[code], dict):
            return v[code]
    # or flat at top level
    return rules.get(code, {})


def test_rrel11_sources_obligor(delivery_rules):
    r = _rule(delivery_rules, "RREL11")
    assert r.get("projected_source_field") == "geographic_region_obligor"


def test_rrel12_sources_classification_year(delivery_rules):
    r = _rule(delivery_rules, "RREL12")
    assert r.get("projected_source_field") == "geographic_region_classification"
    # not the old non-existent obligor-year field
    assert r.get("projected_source_field") != "geographic_region_obligor_year"
