"""Shared fixtures for the mi_agent_pptx test suite.

Builds a small, deterministic synthetic canonical typed tape and a fake MI Agent
run directory so tests exercise the real registry + analytics_lib code paths
without depending on a live pipeline run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mi_agent_pptx.registry_loader import RegistryLoader


@pytest.fixture(scope="session")
def registries() -> RegistryLoader:
    return RegistryLoader()


@pytest.fixture
def sample_tape() -> pd.DataFrame:
    """A small equity-release-style canonical typed tape (LTV as points)."""
    rows = []
    regions = ["TLI3", "TLI3", "TLJ2", "TLK1", "TLD3", "TLI3"]
    ages = [62, 68, 74, 81, 77, 70]
    ltv_points = [38.0, 45.0, 52.0, 61.0, 29.0, 47.0]  # whole-number points
    balances = [120000.0, 180000.0, 240000.0, 310000.0, 90000.0, 150000.0]
    rates = [7.1, 7.4, 6.9, 8.2, 7.0, 7.6]
    orig = ["2024-01-15", "2024-06-20", "2025-02-10", "2025-08-05",
            "2023-11-30", "2025-05-18"]
    for i in range(6):
        rows.append({
            "unique_identifier": f"L{i+1:04d}",
            "current_principal_balance": balances[i],
            "current_loan_to_value": ltv_points[i],
            "original_loan_to_value": ltv_points[i] / 100.0 - 0.02,
            "current_interest_rate": rates[i],
            "youngest_borrower_age": ages[i],
            "geographic_region_obligor": regions[i],
            "origination_date": orig[i],
            "data_cut_off_date": "2026-01-31",
        })
    return pd.DataFrame(rows)


@pytest.fixture
def run_dir(tmp_path, sample_tape) -> Path:
    """A fake run directory carrying a platform canonical typed tape."""
    d = tmp_path / "orun_test"
    (d / "out_platform").mkdir(parents=True)
    sample_tape.to_csv(d / "out_platform" / "platform_canonical_typed.csv",
                       index=False)
    (d / "run_state.json").write_text(json.dumps({
        "run_id": "orun_test", "client_id": "test_client", "target": "mi",
        "out_root": str(tmp_path),
    }))
    return d


@pytest.fixture
def empty_run_dir(tmp_path) -> Path:
    """A run directory with no artifacts at all."""
    d = tmp_path / "orun_empty"
    d.mkdir()
    return d


@pytest.fixture
def deck_config_path() -> Path:
    from mi_agent_pptx.registry_loader import REPO_ROOT
    return REPO_ROOT / "configs" / "pptx" / "investor_pack.yaml"
