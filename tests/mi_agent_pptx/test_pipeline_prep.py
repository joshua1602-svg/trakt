"""Tests for the config-driven pipeline-tape canonicalisation."""

from __future__ import annotations

import pandas as pd

from mi_agent_pptx.data_resolver import resolve_data
from mi_agent_pptx.pipeline_prep import canonicalise_pipeline, is_raw_pipeline


def _raw_pipeline() -> pd.DataFrame:
    return pd.DataFrame([
        {"Account Number": "A1", "Broker": "Age Partnership", "DOB App 1": "1952-03-01",
         "Loan Amount": 120000, "Estimated Value": 300000, "Product Rate": 7.1,
         "Property Region": "London", "Status": "KFI Issued",
         "Snapshot Date": "2026-01-31"},
        {"Account Number": "A2", "Broker": "Key Later Life", "DOB App 1": "1948-06-15",
         "Loan Amount": 200000, "Estimated Value": 400000, "Product Rate": 7.5,
         "Property Region": "Scotland", "Status": "Offer Issued",
         "Snapshot Date": "2026-01-31"},
        {"Account Number": "A3", "Broker": "Cornerstone", "DOB App 1": "1955-01-20",
         "Loan Amount": 90000, "Estimated Value": 250000, "Product Rate": 6.9,
         "Property Region": "Wales", "Status": "Withdrawn",
         "Snapshot Date": "2026-01-31"},
    ])


def test_detects_raw_pipeline():
    assert is_raw_pipeline(_raw_pipeline())


def test_canonicalises_core_fields():
    out = canonicalise_pipeline(_raw_pipeline(), as_of="2026-01-31")
    for col in ("current_outstanding_balance", "current_valuation_amount",
                "current_interest_rate", "broker_channel", "pipeline_stage",
                "current_loan_to_value"):
        assert col in out.columns, col
    # Loan Amount -> canonical balance, preserved values.
    assert out["current_outstanding_balance"].tolist() == [120000, 200000, 90000]


def test_stage_vocabulary_normalised():
    out = canonicalise_pipeline(_raw_pipeline(), as_of="2026-01-31")
    stages = set(out["pipeline_stage"])
    assert stages <= {"KFI Issued", "Application", "Offer Issued", "Completed",
                      "Withdrawn", "Other"}
    assert "KFI Issued" in stages and "Offer Issued" in stages


def test_forecast_inputs_derived():
    out = canonicalise_pipeline(_raw_pipeline(), as_of="2026-01-31")
    assert "completion_probability" in out.columns
    assert "weighted_expected_funded_amount" in out.columns
    assert "expected_completion_date" in out.columns
    # weighted = balance * probability, and probability in [0,1].
    assert (out["completion_probability"].between(0, 1)).all()
    w = out["weighted_expected_funded_amount"].sum()
    assert 0 < w < out["current_outstanding_balance"].sum()


def test_age_derived_from_dob():
    out = canonicalise_pipeline(_raw_pipeline(), as_of="2026-01-31")
    assert "youngest_borrower_age" in out.columns
    assert out["youngest_borrower_age"].between(60, 85).all()


def test_resolves_through_data_resolver(registries):
    out = canonicalise_pipeline(_raw_pipeline(), as_of="2026-01-31")
    rd = resolve_data(out, registries, as_of_date="2026-01-31")
    assert rd.balance_col == "current_outstanding_balance"
    assert "region" in rd.df.columns          # readable region label enriched
    assert rd.bucket_column("ltv_bucket")     # buckets materialise


def test_idempotent_on_canonical_tape():
    out = canonicalise_pipeline(_raw_pipeline(), as_of="2026-01-31")
    twice = canonicalise_pipeline(out, as_of="2026-01-31")
    assert twice["current_outstanding_balance"].tolist() == \
        out["current_outstanding_balance"].tolist()


# --------------------------------------------------------------------------- #
# Integration with the real MI Agent prep (the layer the dashboard uses) — the
# regression that a raw 18a/M2L pipeline tape and a funded tape without an
# explicit original LTV both resolve to canonical fields (only Risk Monitor
# should ever be a placeholder).
# --------------------------------------------------------------------------- #
def test_cli_prep_pipeline_uses_real_mi_agent_prep():
    from mi_agent_pptx.cli import _prep_pipeline
    out = _prep_pipeline(_raw_pipeline(), "2026-01-31")
    for col in ("current_outstanding_balance", "pipeline_stage", "broker_channel",
                "weighted_expected_funded_amount", "expected_completion_date"):
        assert col in out.columns, col
    # Real prep emits canonical UPPERCASE stage tokens.
    assert set(out["pipeline_stage"].str.upper()) <= {
        "KFI", "APPLICATION", "OFFER", "COMPLETED", "WITHDRAWN", "OTHER"}


def test_cli_prep_funded_derives_original_ltv(sample_tape):
    from mi_agent_pptx.cli import _prep_funded
    # sample_tape carries current LTV but not an explicit original LTV column.
    out = _prep_funded(sample_tape)
    assert "original_loan_to_value" in out.columns
    assert out["original_loan_to_value"].notna().any()


def test_pipeline_discovery_prefers_rich_source_over_thin_18a(tmp_path):
    """The rich governed M2L source (deep under output/pipeline/) must win over
    the thin 18a tape — the regression where pipeline charts placeholdered."""
    import shutil
    from mi_agent_pptx.artifact_loader import load_run_artifacts
    from mi_agent_pptx.cli import _resolve_pipeline_tape
    from mi_agent_pptx.registry_loader import REPO_ROOT

    run = tmp_path / "orun_disc"
    deep = run / "portfolios" / "direct_001" / "onboarding" / "mi" / "output" / "pipeline" / "2025-11-01"
    deep.mkdir(parents=True)
    fixture = (REPO_ROOT / "tests" / "fixtures" / "client_001_mi_pack" / "pipeline"
               / "2025-11-01" / "M2L_KFI_and_Pipeline_2025_11_01.csv")
    if not fixture.exists():
        import pytest
        pytest.skip("M2L pipeline fixture not present")
    shutil.copy(fixture, deep / fixture.name)
    # A thin, near-useless 18a at the run root.
    pd.DataFrame({"application_id": [1, 2], "expected_advance": [1, 2]}).to_csv(
        run / "18a_central_pipeline_tape.csv", index=False)
    (run / "run_state.json").write_text('{"run_id":"orun_disc","client_id":"ERE",'
                                        '"reporting_date":"2025-11-01"}')

    artifacts = load_run_artifacts(run)
    out = _resolve_pipeline_tape(artifacts, "2025-11-01")
    assert out is not None and not out.empty
    # Rich source → canonical pipeline fields present (thin 18a could not).
    for col in ("current_outstanding_balance", "pipeline_stage", "broker_channel"):
        assert col in out.columns, col
    assert out["pipeline_stage"].notna().any()


def test_pipeline_resolved_from_sibling_run(tmp_path):
    """Pipeline is a client-level, cross-run source: a funded run with NO M2L
    under it must still resolve the pipeline from a sibling run's source."""
    import shutil
    from mi_agent_pptx.artifact_loader import load_run_artifacts
    from mi_agent_pptx.cli import _resolve_pipeline_tape
    from mi_agent_pptx.registry_loader import REPO_ROOT

    fixture = (REPO_ROOT / "tests" / "fixtures" / "client_001_mi_pack" / "pipeline"
               / "2025-11-01" / "M2L_KFI_and_Pipeline_2025_11_01.csv")
    if not fixture.exists():
        import pytest
        pytest.skip("M2L pipeline fixture not present")

    container = tmp_path / "blob_trigger"
    funded = container / "orun_ere_funded"
    funded.mkdir(parents=True)
    (funded / "run_state.json").write_text('{"run_id":"orun_ere_funded",'
                                           '"client_id":"ERE","reporting_date":"2026-01-31"}')
    # Sibling run carries the pipeline source; the funded run has none.
    sib = (container / "orun_ere_pipe" / "portfolios" / "direct_001"
           / "output" / "pipeline" / "2025-11-01")
    sib.mkdir(parents=True)
    shutil.copy(fixture, sib / fixture.name)

    artifacts = load_run_artifacts(funded)
    # No explicit root → falls back to the run dir's parent (sibling runs).
    out = _resolve_pipeline_tape(artifacts, "2026-01-31")
    assert out is not None and not out.empty
    assert "pipeline_stage" in out.columns
    assert "current_outstanding_balance" in out.columns
