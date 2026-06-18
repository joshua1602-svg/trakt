"""
Phase 0B — MI/M&A semantic registry & route-contract foundations.

Lightweight, dependency-free tests (yaml + stdlib only — no pandas/streamlit)
proving the Phase 0B config/registry foundations are well-formed:

  1. every new YAML config parses;
  2. route configs contain the required top-level keys;
  3. the MI semantic registry recognises the curated Phase 0B fields;
  4. no duplicate route / state / dimension keys (duplicate YAML keys rejected);
  5. no Streamlit / chart code is imported or copied into the Phase 0B files.

These are foundations only: no orchestration, snapshot layer, MI states, M&A
agent, chart migration or bucketing engine is built or exercised here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SEMANTICS_PATH = REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"

ROUTE_FILES = [
    "config/routes/mi_route.yaml",
    "config/routes/mna_route.yaml",
    "config/routes/regulatory_annex2_route.yaml",
    "config/routes/regulatory_and_mi_route.yaml",
]

MI_CONFIG_FILES = [
    "config/mi/state_library.yaml",
    "config/mi/stratification_catalogue.yaml",
    "config/mi/buckets.yaml",
    "config/mi/risk_monitor.yaml",
]

MNA_CONFIG_FILES = [
    "config/mna/diligence_scorecard.yaml",
]

ALL_NEW_CONFIGS = ROUTE_FILES + MI_CONFIG_FILES + MNA_CONFIG_FILES

# Fields the brief requires the MI semantic registry to recognise.
CURATED_PHASE0B_FIELDS = [
    "amortisation_type",
    "internal_risk_grade",
    "internal_risk_score",
    "internal_risk_stage",
    "ifrs9_stage",
    "probability_of_default",
    "pd_bucket",
    "loss_given_default",
    "lgd_bucket",
    "exposure_at_default",
    "ead_bucket",
    "portfolio_id",
    "spv_id",
    "acquired_portfolio_id",
    "acquisition_date",
    "spv_transfer_date",
    "reporting_date",
    "cut_off_date",
    "upload_timestamp",
    "pipeline_stage",
    "funded_status",
    "forecast_funding_date",
    "forecast_funding_probability",
    "forecast_funded_balance",
    "number_of_borrowers",
    "borrower_structure",
    "months_on_book",
]

REQUIRED_ROUTE_KEYS = {
    "route_id",
    "temporality",
    "requires_history",
    "allowed_states",
    "allowed_dimensions",
    "temporal_modes",
    "risk_monitor",
    "forecast",
    "exceptions_scorecard",
}


# --------------------------------------------------------------------------- #
# Duplicate-key-detecting YAML loader
# --------------------------------------------------------------------------- #


class _DuplicateKeyError(ValueError):
    pass


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that raises if a mapping has duplicate keys (which
    ``yaml.safe_load`` would otherwise silently collapse)."""


def _no_duplicates(loader, node, deep=False):  # pragma: no cover - thin shim
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise _DuplicateKeyError(f"duplicate key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicates
)


def _load_unique(path: Path) -> dict:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)


# --------------------------------------------------------------------------- #
# 1. Every new YAML config parses (and has no duplicate keys)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("rel", ALL_NEW_CONFIGS)
def test_new_config_parses(rel):
    path = REPO_ROOT / rel
    assert path.exists(), f"missing config file: {rel}"
    data = _load_unique(path)
    assert isinstance(data, dict) and data, f"{rel} did not parse to a mapping"


# --------------------------------------------------------------------------- #
# 2. Route configs contain the required top-level keys
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("rel", ROUTE_FILES)
def test_route_required_keys(rel):
    data = _load_unique(REPO_ROOT / rel)
    missing = REQUIRED_ROUTE_KEYS - set(data)
    assert not missing, f"{rel} missing required keys: {sorted(missing)}"


@pytest.mark.parametrize("rel", ROUTE_FILES)
def test_route_capability_values_are_valid(rel):
    data = _load_unique(REPO_ROOT / rel)
    assert data["temporality"] in ("recurring", "point_in_time")
    assert isinstance(data["requires_history"], bool)
    assert isinstance(data["allowed_states"], list)
    assert isinstance(data["allowed_dimensions"], list)
    assert isinstance(data["temporal_modes"], list) and data["temporal_modes"]
    for mode in data["temporal_modes"]:
        assert mode in ("single", "compare", "trend"), mode
    for cap in ("risk_monitor", "forecast", "exceptions_scorecard"):
        assert data[cap] in ("enabled", "disabled", "optional"), (cap, data[cap])
    # point-in-time routes must be single-mode only.
    if data["temporality"] == "point_in_time":
        assert data["temporal_modes"] == ["single"], rel


def test_route_ids_are_unique_and_expected():
    ids = [_load_unique(REPO_ROOT / r)["route_id"] for r in ROUTE_FILES]
    assert len(ids) == len(set(ids)), f"duplicate route_id across files: {ids}"
    assert set(ids) == {"mi", "mna", "regulatory_annex2", "regulatory_and_mi"}


# --------------------------------------------------------------------------- #
# 3. MI semantic registry recognises the curated Phase 0B fields
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def semantics():
    return _load_unique(SEMANTICS_PATH)


@pytest.mark.parametrize("field", CURATED_PHASE0B_FIELDS)
def test_registry_recognises_curated_field(semantics, field):
    fields = semantics["fields"]
    assert field in fields, f"{field} missing from MI semantic registry"
    entry = fields[field]
    assert entry["role"] in (
        "metric", "dimension", "date", "flag",
    ), f"{field} has unexpected role {entry['role']!r}"
    assert entry["business_name"], f"{field} needs a business_name"
    assert entry["business_description"], f"{field} needs a business_description"
    assert isinstance(entry["synonyms"], list) and entry["synonyms"], (
        f"{field} should have at least one synonym"
    )


def test_registry_risk_metrics_have_bucket_hints(semantics):
    fields = semantics["fields"]
    assert fields["probability_of_default"]["bucket_field"] == "pd_bucket"
    assert fields["loss_given_default"]["bucket_field"] == "lgd_bucket"
    assert fields["exposure_at_default"]["bucket_field"] == "ead_bucket"
    # The referenced bucket dimensions must themselves be registered.
    for b in ("pd_bucket", "lgd_bucket", "ead_bucket"):
        assert fields[b].get("derived") is True
        assert fields[b]["role"] == "dimension"


def test_registry_virtual_dimensions_flagged(semantics):
    fields = semantics["fields"]
    for v in ("portfolio_id", "spv_id", "acquired_portfolio_id",
              "reporting_date", "cut_off_date", "pipeline_stage",
              "funded_status", "number_of_borrowers", "months_on_book"):
        assert fields[v].get("virtual") is True, f"{v} should be virtual"


def test_registry_metadata_counts_consistent(semantics):
    m = semantics["metadata"]
    assert m["field_count"] == m["core_field_count"] + m["extended_field_count"]
    assert m["field_count"] == len(semantics["fields"])
    assert m["virtual_field_count"] >= 14


# --------------------------------------------------------------------------- #
# 4. No duplicate route / state / dimension keys
# --------------------------------------------------------------------------- #


def test_no_duplicate_keys_in_any_config():
    # _load_unique raises _DuplicateKeyError on any duplicate mapping key.
    for rel in ALL_NEW_CONFIGS + [str(SEMANTICS_PATH.relative_to(REPO_ROOT))]:
        _load_unique(REPO_ROOT / rel)


def test_state_keys_unique_and_referenced_by_routes():
    states = _load_unique(REPO_ROOT / "config/mi/state_library.yaml")["states"]
    assert len(states) == len(set(states))
    mi = _load_unique(REPO_ROOT / "config/routes/mi_route.yaml")
    # Every MI allowed_state must be defined in the state library.
    for s in mi["allowed_states"]:
        assert s in states, f"MI route references undefined state {s!r}"


def test_dimension_keys_unique_and_resolve_to_semantics():
    cat = _load_unique(
        REPO_ROOT / "config/mi/stratification_catalogue.yaml"
    )["dimensions"]
    assert len(cat) == len(set(cat))
    semantics = _load_unique(SEMANTICS_PATH)["fields"]
    for dim, spec in cat.items():
        sf = spec["semantic_field"]
        assert sf in semantics, (
            f"stratification dimension {dim!r} -> unknown semantic_field {sf!r}"
        )


def test_bucket_keys_unique_and_cover_required_buckets():
    buckets = _load_unique(REPO_ROOT / "config/mi/buckets.yaml")["buckets"]
    assert len(buckets) == len(set(buckets))
    required = {
        "ltv_bucket", "borrower_age_bucket", "youngest_borrower_age_bucket",
        "interest_rate_bucket", "pd_bucket", "lgd_bucket", "ead_bucket",
        "balance_band", "time_on_book_bucket",
    }
    assert required.issubset(set(buckets)), required - set(buckets)


# --------------------------------------------------------------------------- #
# 5. No Streamlit / chart code imported or copied into Phase 0B files
# --------------------------------------------------------------------------- #


def test_no_streamlit_or_chart_code_in_phase0b_files():
    banned = ("import streamlit", "streamlit", "import plotly",
              "plotly.", "st.plotly_chart", "go.Figure")
    for rel in ALL_NEW_CONFIGS:
        text = (REPO_ROOT / rel).read_text(encoding="utf-8").lower()
        for token in banned:
            assert token.lower() not in text, (
                f"{rel} unexpectedly references chart/Streamlit code: {token!r}"
            )
