"""mi_agent_api/tests/test_risk_limits_contract.py

Production risk-limits CONFIG CONTRACT (Part B):
  * onboarding builds + emits ``output/risk/risk_limits_config.yaml`` from the
    client's Schedule 8 doc, with self-describing source metadata;
  * a missing / unreadable doc yields a controlled not_found/failed status with
    diagnostics, NEVER placeholder limits;
  * the API reads the run config FIRST (run config -> docs -> fallback ->
    placeholder) and surfaces source_type / source_file / extraction_status /
    is_placeholder on the envelope.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent.risk_monitor import risk_limits_contract as contract
from mi_agent_api import risk_limits as rl


def _write_run(root: Path, client_id: str, run_id: str, reporting_date: str,
               n: int, london_share: float) -> None:
    rng = np.random.default_rng(abs(hash(run_id)) % (2**32))
    regions = rng.choice(["London", "South East", "Scotland", "Wales", "East"], n)
    regions[: int(n * london_share)] = "London"
    df = pd.DataFrame({
        "loan_identifier": [f"{run_id}_{i}" for i in range(n)],
        "current_outstanding_balance": rng.uniform(80_000, 250_000, n).round(2),
        "current_loan_to_value": rng.uniform(20, 55, n).round(1),
        "current_interest_rate": rng.uniform(3, 8, n).round(2),
        "youngest_borrower_age": rng.integers(62, 88, n),
        "broker_channel": rng.choice(["Alpha", "Beta", "Gamma", "Delta"], n),
        "geographic_region_obligor": regions,
        "reporting_date": [reporting_date] * n,
    })
    d = root / client_id / run_id / "output" / "central"
    d.mkdir(parents=True, exist_ok=True)
    df.to_csv(d / "18_central_lender_tape.csv", index=False)


# --------------------------------------------------------------------------- #
# build_config — schema + statuses
# --------------------------------------------------------------------------- #
def test_build_config_from_schedule8_doc(monkeypatch):
    monkeypatch.chdir(_REPO_ROOT)  # so the client_001 fixture doc resolves
    cfg = contract.build_config("client_001", extracted_at="2026-06-27T00:00:00Z")
    assert cfg["source_type"] == contract.SOURCE_SCHEDULE_8
    assert cfg["is_placeholder"] is False
    assert cfg["extraction_status"] in (contract.STATUS_SUCCESS, contract.STATUS_PARTIAL)
    assert "docs" in cfg["source_file"]
    assert cfg["limits"]
    row = next(l for l in cfg["limits"] if l["category"] == "geographic_concentration"
               and l["region_codes"] == ["London"])
    # Production schema fields are present on every limit row.
    for key in ("limit_id", "category", "display_name", "metric", "region_codes",
                "operator", "threshold", "warning_threshold", "denominator",
                "source_text", "source"):
        assert key in row
    assert row["threshold"] == 30.0
    assert row["operator"] == "max"
    assert row["warning_threshold"] == 27.0  # 90% of 30
    assert row["source"] == "Schedule 8 document"


def test_build_config_missing_doc_is_not_found_not_placeholder_limits(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no fixture, no docs
    cfg = contract.build_config("client_absent")
    assert cfg["source_type"] == contract.SOURCE_PLACEHOLDER
    assert cfg["extraction_status"] == contract.STATUS_NOT_FOUND
    assert cfg["is_placeholder"] is True
    assert cfg["limits"] == []
    assert "not found" in cfg["diagnostics"]["reason"].lower()


def test_build_config_unreadable_doc_is_failed_with_diagnostics(monkeypatch, tmp_path):
    docs = tmp_path / "tests" / "fixtures" / "client_088_mi_pack" / "docs"
    docs.mkdir(parents=True)
    (docs / "Schedule 8 Concentration.pdf").write_bytes(b"%PDF-1.7\x00bin")
    monkeypatch.chdir(tmp_path)
    cfg = contract.build_config("client_088")
    assert cfg["source_type"] == contract.SOURCE_SCHEDULE_8
    assert cfg["extraction_status"] == contract.STATUS_FAILED
    assert cfg["is_placeholder"] is False
    assert cfg["limits"] == []
    assert cfg["diagnostics"]["suffix"] == ".pdf"


def test_config_round_trips_to_internal_limits(monkeypatch):
    monkeypatch.chdir(_REPO_ROOT)
    cfg = contract.build_config("client_001")
    internal = contract.config_to_internal_limits(cfg)
    london = next(l for l in internal if l.get("region") == "London")
    assert london["direction"] == "max"
    assert london["limit_value"] == 30.0
    assert london["unit"] == "percent"
    assert london["source_snippet"]


# --------------------------------------------------------------------------- #
# emit + API precedence
# --------------------------------------------------------------------------- #
def test_emit_writes_run_config_yaml(monkeypatch, tmp_path):
    monkeypatch.chdir(_REPO_ROOT)
    root = tmp_path / "onboarding_output"
    out = contract.emit_for_run(root, "client_001", "mi_2025_11")
    path = contract.run_config_path(root, "client_001", "mi_2025_11")
    assert path.exists()
    assert out["written_to"] == str(path)
    reloaded = contract.load_config(path)
    assert reloaded["source_type"] == contract.SOURCE_SCHEDULE_8
    assert reloaded["limits"]


def test_api_reads_run_config_first(monkeypatch, tmp_path):
    warnings.simplefilter("ignore")
    monkeypatch.chdir(_REPO_ROOT)
    root = tmp_path / "onboarding_output"
    _write_run(root, "client_001", "mi_2025_11", "2025-11-30", 70, 0.45)
    contract.emit_for_run(root, "client_001", "mi_2025_11")

    out = rl.compute_risk_limits(root, "client_001", "mi_2025_11")
    assert out["sourceType"] == contract.SOURCE_SCHEDULE_8
    assert out["isPlaceholder"] is False
    assert "docs" in (out["sourceFile"] or "")
    assert out["extractionStatus"] in (contract.STATUS_SUCCESS, contract.STATUS_PARTIAL)
    assert "run config" in out["limitsSource"]
    assert out["lineage"]["runConfigPath"].endswith("risk_limits_config.yaml")
    # The London limit drives a real test against the funded actual.
    london = next(t for t in out["testsByCategory"]["geographic_concentration"]
                  if (t["region"] or "") == "London")
    assert london["limitValue"] == 30.0
    assert london["actualValue"] is not None


def test_api_placeholder_when_nothing_available(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no run config, no doc, no fallback
    out = rl.compute_risk_limits(None, "client_xyz", None)
    assert out["available"] is False
    assert out["sourceType"] == contract.SOURCE_PLACEHOLDER
    assert out["isPlaceholder"] is True
    assert out["extractionStatus"] == contract.STATUS_NOT_FOUND


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
