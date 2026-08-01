"""Shared fixtures for the concentration-test suites: file-backed storage in a
tmp directory (the same seam every OCC suite uses — no Azure, no mocks), the
governed library, and a small hand-checkable funded frame."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAKT_STORAGE_BACKEND", "file")
    monkeypatch.setenv("TRAKT_LOCAL_BLOB_ROOT", str(tmp_path / "blob"))
    monkeypatch.setenv("TRAKT_RUNTIME_MODE", "test")
    return tmp_path


@pytest.fixture(scope="module")
def lib():
    from mi_agent.concentration_tests.library import load_library
    return load_library(use_cache=False)


@pytest.fixture()
def storage(env):
    from apps.blob_trigger_app.storage import open_storage
    return open_storage()


@pytest.fixture()
def store(storage):
    from mi_agent.concentration_tests.store import ConcentrationStore
    return ConcentrationStore(storage, container="operations-control")


@pytest.fixture()
def ops_store(storage):
    from operations_control.stores import OpsLayout, OpsStore
    return OpsStore(storage, OpsLayout(container="operations-control"))


@pytest.fixture()
def service(ops_store, store, lib):
    from operations_control.concentration import ConcentrationGovernanceService
    return ConcentrationGovernanceService(ops_store, store=store, lib=lib)


@pytest.fixture()
def funded_frame() -> pd.DataFrame:
    """Ten loans with hand-checkable shares (total balance 1,000,000)."""
    return pd.DataFrame({
        "loan_id": [f"L{i}" for i in range(1, 11)],
        "current_outstanding_balance": [
            100_000, 150_000, 50_000, 200_000, 100_000,
            80_000, 120_000, 60_000, 90_000, 50_000],
        "original_principal_balance": [
            120_000, 160_000, 55_000, 1_200_000, 110_000,
            90_000, 130_000, 65_000, 95_000, 60_000],
        "collateral_geography": [
            "East Anglia", "London", "South East", "East Anglia", "London",
            "Wales", "Scotland", "South East", "East Midlands", "East Anglia"],
        "original_valuation_amount": [
            300_000, 2_000_000, 90_000, 450_000, 500_000,
            250_000, 350_000, 95_000, 400_000, 280_000],
        "current_valuation_amount": [
            330_000, 2_100_000, 99_000, 470_000, 520_000,
            260_000, 365_000, 99_500, 420_000, 300_000],
        "youngest_borrower_age": [70, 54, 62, 68, 71, 80, 66, 59, 73, 86],
        "current_interest_rate": [5.0, 6.0, 4.5, 5.5, 6.5, 4.0, 5.0, 5.5, 6.0, 4.5],
        "interest_rate_type": [
            "Fixed", "Variable", "Fixed", "Variable", "Fixed",
            "Fixed", "Variable", "Fixed", "Fixed", "Fixed"],
        "borrower_identifier": [
            "B1", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B1"],
        "borrower_structure": [
            "Joint", "Joint", "Single", "Joint", "Single",
            "Single", "Joint", "Single", "Single", "Single"],
        "current_loan_to_value": [
            0.30, 0.07, 0.51, 0.43, 0.19, 0.31, 0.33, 0.60, 0.21, 0.17],
        "number_of_days_in_arrears": [0, 0, 45, 0, 0, 95, 0, 0, 0, 0],
        "origination_date": [
            "2023-05-01", "2024-02-01", "2023-08-01", "2024-06-01",
            "2022-01-01", "2023-03-01", "2024-09-01", "2022-11-01",
            "2023-12-01", "2024-01-01"],
    })
