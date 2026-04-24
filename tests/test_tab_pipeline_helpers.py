from __future__ import annotations

from pathlib import Path

import pandas as pd

from analytics.pipeline_tab_helpers import (
    DEFAULT_EXPECTED_CONFIG_RELATIVE,
    add_pipeline_stratification_buckets,
    load_expected_funding_config_yaml,
    prepare_weekly_trend_dataset,
    resolve_expected_funding_config_path,
)
from analytics.pipeline_prep import normalize_pipeline_snapshot, PipelinePrepConfig


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


def test_pipeline_age_bucket_prefers_youngest_borrower_age():
    df = pd.DataFrame(
        {
            "youngest_borrower_age": [54, 62, None],
            "borrower_age": [70, 80, 90],
        }
    )
    out = add_pipeline_stratification_buckets(df)
    assert list(out["age_bucket"].astype(str)) == ["<55", "60-65", "Unknown"]


def test_pipeline_prep_maps_borrower_age_to_youngest():
    raw = pd.DataFrame(
        {
            "Snapshot Date": ["2026-03-01"],
            "Status": ["Application"],
            "Loan Amount": [100000],
            "Estimated Value": [200000],
            "Borrower Age": [67],
        }
    )
    out = normalize_pipeline_snapshot(raw, PipelinePrepConfig())
    assert "youngest_borrower_age" in out.columns
    assert float(out.loc[0, "youngest_borrower_age"]) == 67.0


def test_prepare_weekly_trend_dataset_preserves_chronological_and_stage_order():
    history = pd.DataFrame(
        {
            "snapshot_date": ["2026-01-08", "2026-01-01", "2026-01-08", "2026-01-01"],
            "stage": ["APPLICATION", "KFI", "KFI", "APPLICATION"],
            "count": [2, 1, 3, 4],
            "amount": [200, 100, 300, 400],
        }
    )
    trend = prepare_weekly_trend_dataset(history)
    assert list(trend["week"].cat.categories) == ["01 Jan 2026", "08 Jan 2026"]
    assert list(trend["stage"].cat.categories) == ["KFI", "APPLICATION", "OFFER", "COMPLETED"]
