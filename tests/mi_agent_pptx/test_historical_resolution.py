"""Historical (multi-period) snapshot-resolution regressions for the deck.

The deck must resolve funded / pipeline / forecast / funnel history through the
SAME resolution the MI Agent dashboard uses, and populate the time-series slides
wherever enough history exists — not require the historical cuts to live inside
the current run directory. These tests cover the escalation the PR promises:

  * current run only            → time-series slides placeholder (insufficient history)
  * current run + ≥2 funded cuts → funded evolution renders
  * current run + ≥2 pipeline weeks → pipeline evolution + funnel render
  * current run + 4 pipeline weeks  → run-rate projection renders IF the endpoint would
  * missing risk limits          → risk stays a placeholder
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from mi_agent_pptx.mi_api import build_dashboard_data
from mi_agent_pptx.registry_loader import REPO_ROOT

_M2L = REPO_ROOT / "tests" / "fixtures" / "client_001_mi_pack" / "pipeline"
_WEEKS = ["2025-10-01", "2025-11-01"]


def _canonical(n: int, date: str, balance_each: float) -> pd.DataFrame:
    regions = ["TLI3", "TLJ2", "TLK1", "TLD3"]
    return pd.DataFrame([{
        "unique_identifier": f"L{i:04d}",
        "current_outstanding_balance": balance_each,
        "current_principal_balance": balance_each,
        "current_loan_to_value": 40.0 + (i % 5) * 5,
        "current_interest_rate": 7.0 + (i % 4) * 0.3,
        "youngest_borrower_age": 62 + (i % 20),
        "geographic_region_obligor": regions[i % 4],
        "origination_date": f"{2023 + (i % 3)}-06-15",
        "reporting_date": date,
        "data_cut_off_date": date,
        "source_portfolio_id": "ERE",
    } for i in range(n)])


def _run_dir(container: Path, *, reporting_date: str = "2025-11-30") -> Path:
    run = container / "orun_ere"
    (run / "out_platform").mkdir(parents=True)
    _canonical(40, reporting_date, 150000.0).to_csv(
        run / "out_platform" / "platform_canonical_typed.csv", index=False)
    (run / "run_state.json").write_text(json.dumps(
        {"run_id": "orun_ere", "client_id": "ERE", "reporting_date": reporting_date}))
    return run


def _pipeline_weeks(root: Path, weeks) -> Path:
    """Lay governed weekly M2L extracts under ``root/{date}/`` (skips if absent)."""
    for wk in weeks:
        src = _M2L / wk / f"M2L_KFI_and_Pipeline_{wk.replace('-', '_')}.csv"
        if not src.exists():
            pytest.skip(f"M2L fixture missing for {wk}")
        dst_dir = root / wk
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst_dir / src.name)
    return root


def _dated_funded_cuts(out_root: Path, cuts) -> None:
    for date, balance in cuts:
        d = out_root / "platform" / "ERE" / date
        d.mkdir(parents=True, exist_ok=True)
        _canonical(40, date, balance).to_csv(d / "platform_canonical_typed.csv", index=False)


# --------------------------------------------------------------------------- #

def test_single_period_time_series_placeholder(tmp_path):
    """One funded cut + one pipeline week → every time-series surface reports
    insufficient history; the funnel falls back to the current-week funnel."""
    run = _run_dir(tmp_path)
    proot = _pipeline_weeks(tmp_path / "pipeline", ["2025-11-01"])
    data = build_dashboard_data(run, client_id="ERE", pipeline_root=str(proot))

    ts = data.diagnostics["timeSeries"]
    assert ts["funded_evolution"]["placeholder"] is True
    assert data.diagnostics["fundedCutsFound"] == 0
    assert ts["pipeline_evolution"]["placeholder"] is True
    assert data.diagnostics["pipelineSnapshotsFound"] == 1
    # Funnel is NOT a placeholder — single-week current funnel is shown.
    assert data.pipeline                      # pipeline snapshot resolved
    assert ts["funnel"]["placeholder"] is False
    # And the reason strings say *insufficient history*, not "data unavailable".
    assert "insufficient history" in ts["funded_evolution"]["reason"]


def test_two_funded_cuts_render_funded_evolution(tmp_path):
    """Two dated funded cuts under the onboarding root (NOT inside the run dir) →
    funded evolution renders from the same assembler the dashboard uses."""
    run = _run_dir(tmp_path)
    _dated_funded_cuts(tmp_path, [("2025-09-30", 120000.0), ("2025-10-31", 135000.0)])

    data = build_dashboard_data(run, client_id="ERE", output_root=str(tmp_path))
    assert data.diagnostics["fundedCutsFound"] >= 2
    assert data.diagnostics["timeSeries"]["funded_evolution"]["placeholder"] is False
    assert len(data.funded_evolution.get("periods", [])) >= 2


def test_two_pipeline_weeks_render_evolution_and_funnel(tmp_path):
    """Two governed weekly extracts → pipeline evolution and the weekly funnel both
    render (no longer single-period)."""
    run = _run_dir(tmp_path)
    proot = _pipeline_weeks(tmp_path / "pipeline", _WEEKS)
    data = build_dashboard_data(run, client_id="ERE", pipeline_root=str(proot))

    assert data.diagnostics["pipelineSnapshotsFound"] >= 2
    assert data.diagnostics["timeSeries"]["pipeline_evolution"]["placeholder"] is False
    assert len(data.pipeline_evolution.get("periods", [])) >= 2
    assert data.funnel.get("singlePeriod") is not True


def test_four_pipeline_weeks_render_projection_if_endpoint_would(tmp_path):
    """Four weekly extracts → the run-rate projection renders IF the extrapolation
    endpoint would (it is fed the multi-week history); otherwise the diagnostic
    reflects the discovered week count rather than a hard failure."""
    run = _run_dir(tmp_path)
    proot = tmp_path / "pipeline"
    # Duplicate the two governed weeks into four dated folders.
    _pipeline_weeks(proot, _WEEKS)
    for date in ("2025-08-01", "2025-09-01"):
        (proot / date).mkdir(parents=True, exist_ok=True)
        src = _M2L / "2025-10-01" / "M2L_KFI_and_Pipeline_2025_10_01.csv"
        shutil.copy(src, proot / date / f"M2L_KFI_and_Pipeline_{date.replace('-', '_')}.csv")

    data = build_dashboard_data(run, client_id="ERE", pipeline_root=str(proot))
    assert data.diagnostics["pipelineSnapshotsFound"] >= 4
    proj = data.diagnostics["timeSeries"]["forecast_projection"]
    model = (data.extrapolation.get("completionRunRateForecast")
             or data.extrapolation.get("kfiConversionForecast") or {})
    if model.get("available"):
        assert proj["placeholder"] is False
    else:
        # Endpoint judged the history insufficient — the deck placeholders, and the
        # diagnostic reports the discovered count (wiring is exercised either way).
        assert proj["placeholder"] is True
        assert "extract" in (proj["reason"] or "")


def test_missing_risk_limits_stays_placeholder(tmp_path):
    """No Schedule 8 limits artifact → the risk slide stays a branded placeholder."""
    run = _run_dir(tmp_path)
    data = build_dashboard_data(run, client_id="ERE", output_root=str(tmp_path))
    risk = data.diagnostics["timeSeries"]["risk"]
    assert risk["placeholder"] is True
    assert "Schedule 8" in (risk["reason"] or "")
