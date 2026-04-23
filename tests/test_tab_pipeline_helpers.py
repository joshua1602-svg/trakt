from __future__ import annotations

from pathlib import Path

import pandas as pd

from analytics.pipeline_tab_helpers import (
    DEFAULT_EXPECTED_CONFIG_RELATIVE,
    add_pipeline_stratification_buckets,
    load_expected_funding_config_yaml,
    resolve_expected_funding_config_path,
)


def test_resolve_expected_funding_config_path_module_relative(monkeypatch):
    monkeypatch.chdir(Path("/tmp"))
    resolved = resolve_expected_funding_config_path(DEFAULT_EXPECTED_CONFIG_RELATIVE)
    assert resolved is not None
    assert resolved.name == "pipeline_expected_funding.yaml"


def test_load_expected_funding_config_returns_dict_and_path():
    cfg, path = load_expected_funding_config_yaml(DEFAULT_EXPECTED_CONFIG_RELATIVE)
    assert isinstance(cfg, dict)
    assert path is not None
    assert path.endswith("config/client/pipeline_expected_funding.yaml")


def test_pipeline_stratification_buckets_created_when_fields_present():
    df = pd.DataFrame(
        {
            "current_ltv": [35.0, 82.0, None],
            "loan_amount": [75_000, 250_000, None],
            "borrower_age": [66, 81, None],
        }
    )
    out = add_pipeline_stratification_buckets(df)

    assert "ltv_bucket" in out.columns
    assert "ticket_bucket" in out.columns
    assert "age_bucket" in out.columns
    assert out["ticket_bucket"].astype(str).isin(["50-100k", "200-300k", "Unknown"]).any()
