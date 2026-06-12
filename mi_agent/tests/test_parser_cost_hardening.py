"""
Parser-correctness and LLM cost-control tests (hardening pass).

Covers:
  * explicit dimensions are honoured exactly and never silently substituted;
  * missing dataset columns fail validation cleanly (and skip LLM repair);
  * zero-cost-first deterministic handling of known prompts;
  * compact vs full catalogue size;
  * available column names (not values) sent to the LLM;
  * token/cost observability from a mocked usage response.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mi_agent.llm_query_parser import (
    _deterministic_parse,
    build_prompt,
    compact_catalogue,
    estimate_cost,
    parse_with_repair,
)
from mi_agent.mi_agent_workflow import run_mi_agent_query
from mi_agent.mi_query_validator import load_mi_semantics, validate_mi_query

REPO_ROOT = Path(__file__).resolve().parents[2]
SEMANTICS_PATH = REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"


@pytest.fixture(scope="module")
def semantics():
    return load_mi_semantics(SEMANTICS_PATH)


@pytest.fixture
def df():
    """Full dataframe WITH broker/product/age_bucket columns."""
    regions = ["North", "South", "East", "West", "Wales", "Scotland", "NI"]
    rows = []
    for i in range(48):
        rows.append({
            "loan_identifier": f"L{i:04d}",
            "current_outstanding_balance": 100_000 + i * 5_000,
            "current_loan_to_value": 0.30 + (i % 5) * 0.05,
            "youngest_borrower_age": 55 + (i % 20),
            "geographic_region_obligor": regions[i % 7],
            "broker_channel": ["Broker A", "Broker B", "Broker C"][i % 3],
            "erm_product_type": ["Lump Sum", "Drawdown"][i % 2],
            "account_status": ["Active", "Redeemed"][i % 2],
            "age_bucket": ["55-60", "60-65", "65-70", "70-75"][i % 4],
        })
    return pd.DataFrame(rows)


@pytest.fixture
def df_no_broker(df):
    """Same data but WITHOUT the broker / product columns."""
    return df.drop(columns=["broker_channel", "erm_product_type"])


# --------------------------------------------------------------------------- #
# Parser correctness — explicit dimensions, no substitution
# --------------------------------------------------------------------------- #


def test_top_n_brokers_maps_to_broker_not_status(df, semantics):
    spec, meta = _deterministic_parse("Show top 10 brokers by balance", semantics)
    assert spec.dimension == "broker_channel"
    assert spec.dimension != "account_status"
    assert spec.top_n == 10
    assert spec.aggregation == "sum"
    assert spec.chart_type == "bar"
    assert meta["explicit_dimension_requested"] is True
    assert meta["dimension_substituted"] is False
    # valid against data that HAS broker
    assert validate_mi_query(spec, semantics, available_columns=set(df.columns)).ok


def test_broker_missing_column_fails_no_substitution(df_no_broker, semantics):
    spec, _ = _deterministic_parse("Show top 10 brokers by balance", semantics)
    assert spec.dimension == "broker_channel"  # still the requested field
    vr = validate_mi_query(spec, semantics, available_columns=set(df_no_broker.columns))
    assert not vr.ok
    assert any("broker_channel" in e and "not present" in e for e in vr.errors)


def test_product_type_maps_and_fails_when_missing(df_no_broker, semantics):
    spec, _ = _deterministic_parse("Show weighted average LTV by product type", semantics)
    assert spec.dimension == "erm_product_type"
    assert spec.aggregation == "weighted_avg"
    vr = validate_mi_query(spec, semantics, available_columns=set(df_no_broker.columns))
    assert not vr.ok
    assert any("erm_product_type" in e for e in vr.errors)


def test_balance_by_region(semantics):
    spec, _ = _deterministic_parse("Show balance by region", semantics)
    # region resolves to a true region field (readable display preferred),
    # never the classification year.
    assert spec.dimension in {"collateral_geography",
                              "geographic_region_collateral",
                              "geographic_region_obligor"}
    assert spec.dimension != "geographic_region_classification"
    assert spec.chart_type == "bar"


_REGION_FIELDS = {"collateral_geography", "geographic_region_collateral",
                  "geographic_region_obligor"}


def test_heatmap_age_bucket_and_region_no_substitution(semantics):
    spec, _ = _deterministic_parse(
        "Show LTV by age bucket and region as a heatmap", semantics)
    assert spec.chart_type == "heatmap"
    assert "age_bucket" in spec.dimensions
    # region resolves to a true region field, never account_status / classification
    assert any(d in _REGION_FIELDS for d in spec.dimensions)
    assert "account_status" not in spec.dimensions
    assert "geographic_region_classification" not in spec.dimensions


def test_redemptions_by_account_status(semantics):
    spec, _ = _deterministic_parse("Show redemptions by account status", semantics)
    assert spec.dimension == "account_status"


def test_generic_concentration_picks_sensible_dimension(semantics):
    spec, meta = _deterministic_parse("Where are we most concentrated?", semantics)
    # generic question -> sensible default region field, not a failure and never
    # the classification year.
    assert spec.dimension in _REGION_FIELDS
    assert spec.dimension != "geographic_region_classification"
    assert meta["explicit_dimension_requested"] is False


def test_unknown_dimension_is_not_substituted(semantics):
    spec, _ = _deterministic_parse("Show balance by wibble", semantics)
    # 'wibble' is not a dimension -> no substitution, dimension stays None
    assert spec.dimension is None


# --------------------------------------------------------------------------- #
# Cost control
# --------------------------------------------------------------------------- #


def test_zero_cost_first_handles_known_prompt_without_llm(df, semantics):
    calls = {"n": 0}

    def mock(prompt):
        calls["n"] += 1
        return "{}"

    res = run_mi_agent_query("Show balance by region", df, semantics,
                             llm_enabled=True, parser_mode="llm",
                             zero_cost_first=True, llm_callable=mock)
    assert res["ok"]
    assert res["parser_mode_detail"] == "deterministic_zero_cost"
    assert calls["n"] == 0
    assert res["metadata"]["llm"]["calls"] == 0


def test_missing_column_skips_llm_repair(df_no_broker, semantics):
    calls = {"n": 0}

    def mock(prompt):
        calls["n"] += 1
        return "{}"

    # Explicit broker request, but broker column is missing -> controlled
    # failure with NO LLM calls.
    res = run_mi_agent_query("Show top 10 brokers by balance", df_no_broker,
                             semantics, llm_enabled=True, parser_mode="llm",
                             zero_cost_first=True, llm_callable=mock)
    assert res["ok"] is False
    assert calls["n"] == 0
    pm = res["parse_metadata"]
    assert pm["repair_skipped_reason"] == "missing_dataset_columns"
    assert pm["llm"]["calls"] == 0


def test_missing_column_skips_repair_in_pure_llm_path(df_no_broker, semantics):
    # Force LLM (zero_cost_first off); the model returns a spec referencing the
    # missing broker column -> repair must NOT run (would require substitution).
    bad = json.dumps({"intent": "chart", "chart_type": "bar",
                      "metric": "current_outstanding_balance",
                      "dimension": "broker_channel", "aggregation": "sum"})
    calls = {"n": 0}

    def mock(prompt):
        calls["n"] += 1
        return bad

    spec, meta = parse_with_repair(
        "balance by broker", semantics,
        available_columns=set(df_no_broker.columns),
        llm_enabled=True, max_attempts=2, zero_cost_first=False,
        llm_callable=mock)
    assert meta["ok"] is False
    assert meta["repair_skipped_reason"] == "missing_dataset_columns"
    assert calls["n"] == 1  # one call, no repair calls
    assert meta["llm"]["calls"] == 1


def test_available_columns_sent_but_no_values(df, semantics):
    prompt = build_prompt("balance by region", semantics,
                          available_columns=set(df.columns), catalog_mode="core")
    text = prompt["system"] + "\n" + prompt["user"]
    assert "Available dataset columns" in text
    assert "current_outstanding_balance" in text   # a column NAME
    assert "L0001" not in text                      # no loan identifiers
    assert "987654321" not in text                  # no values


def test_compact_catalogue_smaller_than_full(semantics):
    core = compact_catalogue(semantics, mode="core")
    full = compact_catalogue(semantics, mode="full")
    assert len(core) < len(full)
    # core has fewer field lines than full
    assert len(core.splitlines()) < len(full.splitlines())
    # compact prompt is materially smaller than the old full-JSON catalogue
    from mi_agent.llm_query_parser import _catalogue
    full_json = json.dumps(_catalogue(semantics))
    assert len(core) < len(full_json)


def test_estimate_cost_known_and_unknown_models():
    usage = {"input_tokens": 1000, "output_tokens": 200}
    haiku = estimate_cost("claude-haiku-4-5-20251001", usage)
    assert haiku["cost_estimate_status"] == "estimated"
    assert haiku["total_tokens"] == 1200
    assert haiku["estimated_total_cost"] > 0

    sonnet = estimate_cost("claude-sonnet-4-6", usage)
    assert sonnet["estimated_total_cost"] > haiku["estimated_total_cost"]

    unknown = estimate_cost("some-future-model", usage)
    assert unknown["cost_estimate_status"] == "unknown"
    assert unknown["total_tokens"] == 1200  # tokens still reported


def test_llm_metadata_includes_tokens_when_usage_returned(df, semantics):
    # Mock returns text + usage; workflow should surface token/cost metadata.
    good = json.dumps({"intent": "chart", "chart_type": "bar",
                       "metric": "current_outstanding_balance",
                       "dimension": "geographic_region_obligor",
                       "aggregation": "sum"})

    def mock(prompt):
        return {"text": good, "usage": {"input_tokens": 800, "output_tokens": 120}}

    res = run_mi_agent_query("balance by region", df, semantics,
                             llm_enabled=True, parser_mode="llm",
                             model="claude-haiku-4-5-20251001",
                             zero_cost_first=False, llm_callable=mock)
    llm = res["metadata"]["llm"]
    assert llm["calls"] == 1
    assert llm["input_tokens"] == 800
    assert llm["output_tokens"] == 120
    assert llm["total_tokens"] == 920
    assert llm["estimated_total_cost"] > 0
    assert llm["cost_estimate_status"] == "estimated"
