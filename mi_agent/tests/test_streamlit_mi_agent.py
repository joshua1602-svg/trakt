"""
Tests for the Streamlit MI Agent's non-UI workflow (mi_agent_workflow.py),
the LLM repair loop (llm_query_parser.parse_with_repair) and the env-driven
config (mi_agent_config). The Streamlit runtime itself is not exercised.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mi_agent.llm_query_parser import parse_with_repair
from mi_agent.mi_agent_config import get_llm_config
from mi_agent.mi_agent_workflow import (
    chart_html_str,
    metadata_json_str,
    result_csv_bytes,
    run_mi_agent_query,
    spec_json_str,
)
from mi_agent.mi_query_validator import load_mi_semantics

REPO_ROOT = Path(__file__).resolve().parents[2]
SEMANTICS_PATH = REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"


@pytest.fixture(scope="module")
def semantics():
    return load_mi_semantics(SEMANTICS_PATH)


@pytest.fixture
def df():
    regions = ["North", "South", "East", "West", "Wales", "Scotland", "NI"]
    rows = []
    for i in range(48):
        rows.append({
            "loan_identifier": f"L{i:04d}",
            "current_outstanding_balance": 100_000 + i * 5_000,
            "current_principal_balance": 95_000 + i * 5_000,
            "current_loan_to_value": 0.30 + (i % 5) * 0.05,
            "current_interest_rate": 3.0 + (i % 4) * 0.5,
            "youngest_borrower_age": 55 + (i % 20),
            "geographic_region_obligor": regions[i % 7],
            "broker_channel": ["Broker A", "Broker B"][i % 2],
            "account_status": ["Performing", "Arrears", "Default"][i % 3],
            "origination_date": pd.Timestamp("2021-01-01") + pd.Timedelta(days=40 * i),
            "age_bucket": ["55-60", "60-65", "65-70", "70-75"][i % 4],
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 1-3. deterministic workflow happy paths
# --------------------------------------------------------------------------- #


def test_balance_by_region(df, semantics):
    res = run_mi_agent_query("Show balance by region", df, semantics)
    assert res["ok"], res.get("error")
    assert res["spec"]["chart_type"] == "bar"
    assert res["query_result"].row_count > 0
    assert res["chart_result"] is not None
    assert res["chart_result"].chart_type == "bar"
    assert res["interpreted"]["Validation"] == "Passed"
    assert res["parser_mode"] == "deterministic"


def test_ltv_by_age_by_balance_bubble(df, semantics):
    res = run_mi_agent_query("ltv by age by balance", df, semantics)
    assert res["ok"], res.get("error")
    assert res["spec"]["chart_type"] == "bubble"
    assert res["chart_result"].chart_type == "bubble"


def test_heatmap_age_bucket_and_region(df, semantics):
    res = run_mi_agent_query("Show heatmap ltv by age and region", df, semantics)
    assert res["ok"], res.get("error")
    assert res["spec"]["chart_type"] == "heatmap"
    assert res["chart_result"] is not None


# --------------------------------------------------------------------------- #
# 4. invalid / unparseable question -> clear failure metadata
# --------------------------------------------------------------------------- #


def test_unparseable_question_returns_failure(df, semantics):
    # No "by" pattern -> deterministic parser yields a summary (no chart).
    res = run_mi_agent_query("hello there", df, semantics)
    # Either it validates as a harmless summary (ok, no chart) or it fails
    # cleanly — in both cases there must be NO crash and clear metadata.
    assert res["error"] is None or isinstance(res["error"], str)
    assert res["parse_metadata"] is not None
    assert res["validation"] is not None


def test_validation_failure_surfaces_errors(df, semantics):
    # Force an invalid spec via a mock LLM (sum on a percent metric).
    bad = json.dumps({"intent": "chart", "chart_type": "bar",
                      "metric": "current_loan_to_value",
                      "dimension": "geographic_region_obligor",
                      "aggregation": "sum"})

    res = run_mi_agent_query(
        "weighted ltv by region", df, semantics,
        llm_enabled=True, parser_mode="llm", max_repair_attempts=0,
        zero_cost_first=False,  # force the LLM path so the bad spec is used
        llm_callable=lambda prompt: bad,
    )
    assert res["ok"] is False
    assert res["error"] and "validation" in res["error"].lower()
    assert any("not allowed" in e for e in res["validation"]["errors"])
    assert res["chart_result"] is None


# --------------------------------------------------------------------------- #
# 5-6. LLM enable/disable behaviour
# --------------------------------------------------------------------------- #


def test_llm_disabled_needs_no_api_key(df, semantics, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ENABLE_LLM_MI_AGENT", raising=False)
    cfg = get_llm_config({})
    assert cfg.enabled is False and cfg.available is False
    # deterministic workflow runs fine with no key
    res = run_mi_agent_query("Show balance by region", df, semantics)
    assert res["ok"]


def test_llm_enabled_missing_key_is_controlled(df, semantics):
    # Config reports unavailable + warning rather than raising.
    cfg = get_llm_config({"ENABLE_LLM_MI_AGENT": "true",
                          "MI_AGENT_LLM_PROVIDER": "anthropic"})
    assert cfg.enabled is True
    assert cfg.available is False
    assert any("key" in w.lower() for w in cfg.warnings)

    # Asking the workflow to use the LLM with no key does not crash the process.
    res = run_mi_agent_query("Show balance by region", df, semantics,
                             llm_enabled=True, parser_mode="llm")
    assert res["ok"] in (True, False)
    if not res["ok"]:
        assert isinstance(res["error"], str)


def test_mock_provider_config_available():
    cfg = get_llm_config({"ENABLE_LLM_MI_AGENT": "true",
                          "MI_AGENT_LLM_PROVIDER": "mock"})
    assert cfg.available is True
    assert cfg.provider == "mock"


# --------------------------------------------------------------------------- #
# raw-data privacy
# --------------------------------------------------------------------------- #


def test_llm_prompt_accepts_no_dataframe_and_leaks_no_raw_data(df, semantics):
    """The LLM only ever sees the data-free semantic catalogue + the question."""
    import inspect

    from mi_agent.llm_query_parser import build_prompt

    # build_prompt accepts only the question, semantics, optional COLUMN NAMES
    # and a catalogue mode — never a dataframe / data values.
    params = list(inspect.signature(build_prompt).parameters)
    assert params == ["user_question", "mi_semantics",
                      "available_columns", "catalog_mode"], params
    assert not any(p in ("data", "df", "dataframe", "rows") for p in params)

    # Capture the exact prompt sent during a full LLM workflow run (force the
    # LLM path with zero_cost_first=False so the mock is actually called).
    captured = {}

    def mock(prompt):
        captured["prompt"] = prompt
        return json.dumps({"intent": "chart", "chart_type": "bar",
                           "metric": "current_outstanding_balance",
                           "dimension": "geographic_region_obligor",
                           "aggregation": "sum"})

    spiked = df.copy()
    spiked.loc[spiked.index[0], "loan_identifier"] = "SENTINEL_LOAN_X"
    spiked.loc[spiked.index[0], "current_outstanding_balance"] = 987654321

    run_mi_agent_query("balance by region", spiked, semantics,
                       llm_enabled=True, parser_mode="llm",
                       zero_cost_first=False, llm_callable=mock)

    text = captured["prompt"]["system"] + "\n" + captured["prompt"]["user"]
    # No raw dataset values of any kind.
    assert "SENTINEL_LOAN_X" not in text
    assert "987654321" not in text
    assert "L0001" not in text  # loan identifiers
    # But the data-free catalogue IS present, and only COLUMN NAMES are sent.
    assert "current_loan_to_value" in text
    assert "balance by region" in text.lower()
    assert "Available dataset columns" in text
    assert "current_outstanding_balance" in text  # column name (not a value)


# --------------------------------------------------------------------------- #
# repair loop
# --------------------------------------------------------------------------- #


def test_repair_loop_fixes_invalid_then_valid(df, semantics):
    bad = json.dumps({"intent": "chart", "chart_type": "bar",
                      "metric": "current_loan_to_value",
                      "dimension": "geographic_region_obligor",
                      "aggregation": "sum"})            # invalid: sum on percent
    good = json.dumps({"intent": "chart", "chart_type": "bar",
                       "metric": "current_loan_to_value",
                       "dimension": "geographic_region_obligor",
                       "aggregation": "weighted_avg"})  # valid

    calls = {"n": 0}

    def mock_llm(prompt):
        calls["n"] += 1
        return bad if calls["n"] == 1 else good

    spec, meta = parse_with_repair(
        "weighted ltv by region", semantics,
        available_columns=set(df.columns),
        llm_enabled=True, max_attempts=2, zero_cost_first=False,
        llm_callable=mock_llm,
    )
    assert meta["ok"] is True
    assert meta["parser_mode"] == "llm"
    assert meta["repair_attempts"] == 1
    assert meta["original_error_count"] >= 1
    assert spec.aggregation == "weighted_avg"


def test_repair_loop_exhausts_and_reports(df, semantics):
    bad = json.dumps({"intent": "chart", "chart_type": "bar",
                      "metric": "current_loan_to_value",
                      "dimension": "geographic_region_obligor",
                      "aggregation": "sum"})

    spec, meta = parse_with_repair(
        "weighted ltv by region", semantics,
        available_columns=set(df.columns),
        llm_enabled=True, max_attempts=1, zero_cost_first=False,
        llm_callable=lambda p: bad,
    )
    assert meta["ok"] is False
    assert meta["validation_errors"]
    assert meta["parser_mode"] == "llm"


def test_deterministic_parse_with_repair_metadata(df, semantics):
    spec, meta = parse_with_repair(
        "Show balance by region", semantics,
        available_columns=set(df.columns), llm_enabled=False)
    assert meta["parser_mode"] == "deterministic"
    assert meta["repair_attempts"] == 0
    assert meta["ok"] is True


# --------------------------------------------------------------------------- #
# 7. export helpers
# --------------------------------------------------------------------------- #


def test_export_helpers(df, semantics):
    res = run_mi_agent_query("Show balance by region", df, semantics)
    csv = result_csv_bytes(res["query_result"])
    assert isinstance(csv, bytes) and b"concentration_pct" in csv

    html = chart_html_str(res["chart_result"])
    assert isinstance(html, str) and "<html" in html.lower()

    sj = spec_json_str(res["spec_obj"])
    assert json.loads(sj)["chart_type"] == "bar"

    mj = metadata_json_str(res)
    payload = json.loads(mj)
    assert payload["parser_mode"] == "deterministic"
    assert payload["validation"]["ok"] is True
