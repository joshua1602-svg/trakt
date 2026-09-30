"""The pipeline forecast runs off stage by stage, from measured history.

ERE's Pipeline Run-Off Model workbook forecasts the market-standard way:
applications convert to offers and offers to completions at pull-through
rates, each stage has a validity window after which a case has lapsed, the
chance a case completes depends on how long it has sat in its stage, and KFIs
are top of funnel rather than funding pipeline.

Trakt weighted every open case — 4,750 year-old KFIs included — at one flat
rate per stage ("ever seen completed / ever seen"), with no expiry. It now
measures the workbook's assumptions from the weekly snapshots and applies them
per case. Only the forecast changes: pipeline totals, counts and stage
breakdowns are exactly what they were.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mi_agent_api import pipeline_contract as pc
from mi_agent_api import pipeline_runoff as runoff

APP_TO_OFFER, OFFER_TO_COMPLETION = 0.65, 0.67


def _simulate(root: Path, n_cases: int = 2500, weeks: int = 40, seed: int = 7) -> None:
    """Cumulative weekly snapshots of a pipeline with known true pull-through."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-09-08")
    cases = []
    for i in range(n_cases):
        kfi = start - pd.Timedelta(days=60) + pd.Timedelta(
            days=int(rng.integers(0, 7 * weeks + 60)))
        c = {"id": f"A{i:05d}", "kfi": kfi, "amt": int(rng.integers(50_000, 400_000))}
        if rng.random() < 0.30:
            c["app"] = kfi + pd.Timedelta(days=int(rng.integers(1, 15)))
            if rng.random() < APP_TO_OFFER:
                c["offer"] = c["app"] + pd.Timedelta(days=int(rng.choice([3, 8, 14, 20])))
                if rng.random() < OFFER_TO_COMPLETION:
                    c["done"] = c["offer"] + pd.Timedelta(
                        days=int(np.clip(rng.gamma(3, 9), 1, 125)))
                else:
                    c["wd"] = c["offer"] + pd.Timedelta(
                        days=int(np.clip(rng.gamma(3, 12), 1, 125)))
            else:
                c["wd"] = c["app"] + pd.Timedelta(days=int(rng.integers(10, 30)))
        cases.append(c)

    def status(c, d):
        if c.get("done") is not None and c["done"] <= d:
            return "Completed"
        if c.get("wd") is not None and c["wd"] <= d:
            return "Withdrawn"
        if c.get("offer") is not None and c["offer"] <= d:
            return "Offer"
        if c.get("app") is not None and c["app"] <= d:
            return "Application"
        return "KFI"

    def fmt(c, key, d):
        t = c.get(key)
        return t.strftime("%d/%m/%Y") if t is not None and t <= d else ""

    for w in range(weeks):
        d = start + pd.Timedelta(days=7 * w)
        rows = [{"Account Number": c["id"], "Status": status(c, d),
                 "Loan Amount": c["amt"],
                 "KFI Submitted Date": fmt(c, "kfi", d),
                 "Application Submitted Date": fmt(c, "app", d),
                 "Offer Date": fmt(c, "offer", d),
                 "Date Funds Released": (fmt(c, "done", d)
                                         if status(c, d) == "Completed" else "")}
                for c in cases if c["kfi"] <= d]
        folder = root / "ERE" / d.strftime("%Y-%m-%d")
        folder.mkdir(parents=True)
        pd.DataFrame(rows).to_csv(folder / "pipeline_snapshot.csv", index=False)


@pytest.fixture(scope="module")
def year(tmp_path_factory):
    root = tmp_path_factory.mktemp("pipeline")
    _simulate(root)
    model = pc.build_pipeline_history(root, "ERE")
    latest = pc.weekly_extract_inventory(root, "ERE")["extracts"][-1]
    df, report = pc.load_prepared_pipeline(latest, historical_model=model)
    return root, model, latest, df, report


def test_the_pull_through_rates_are_measured(year):
    _root, model, *_ = year
    stages = model["runoff"]["stages"]
    assert stages["APPLICATION"]["pullThrough"] == pytest.approx(APP_TO_OFFER, abs=0.06)
    assert stages["OFFER"]["pullThrough"] == pytest.approx(OFFER_TO_COMPLETION, abs=0.06)
    assert all(stages[s]["sufficient"] for s in ("APPLICATION", "OFFER"))


def test_the_validity_windows_are_measured_from_history(year):
    _root, model, *_ = year
    stages = model["runoff"]["stages"]
    for st in ("KFI", "APPLICATION", "OFFER"):
        assert stages[st]["windowBasis"] == "measured"
    # Simulated dwell: KFI->App <= 14 days, App->Offer <= 20, Offer->Done <= 125.
    assert stages["KFI"]["windowDays"] <= 14
    assert stages["APPLICATION"]["windowDays"] <= 20
    assert 20 < stages["OFFER"]["windowDays"] <= 125


def test_a_kfi_is_not_forecast(year):
    *_, df, _report = year
    kfi = df[df["pipeline_stage"] == "KFI"]
    assert not kfi.empty
    assert (kfi["completion_probability"] == 0.0).all()
    assert (kfi["completion_probability_source"] == "not_forecast_kfi").all()


def test_a_case_past_its_window_has_lapsed(year):
    *_, df, _report = year
    live = df[df["pipeline_stage"].isin(["APPLICATION", "OFFER"])]
    past = live[live["pipeline_stage_dwell_days"] > live["pipeline_stage_validity_days"]]
    assert not past.empty
    assert (past["completion_probability"] == 0.0).all()
    assert past["completion_probability_source"].str.startswith("expired_").all()


def test_a_live_case_is_weighted_by_its_age_in_stage(year):
    _root, model, *_ = year
    offer = model["runoff"]["stages"]["OFFER"]
    fresh, _ = runoff.advance_from(offer, 0)
    older, _ = runoff.advance_from(offer, 28)
    assert fresh == pytest.approx(OFFER_TO_COMPLETION, abs=0.08)
    assert older < fresh
    # An application completes via an offer: its chance is the product.
    p_app, days_app = runoff.complete_from(model["runoff"], "APPLICATION", 0, (None, None))
    assert p_app == pytest.approx(APP_TO_OFFER * OFFER_TO_COMPLETION, abs=0.08)
    _p, days_offer = runoff.complete_from(model["runoff"], "OFFER", 0, (None, None))
    assert days_app > days_offer > 0


def test_live_cases_are_expected_after_the_as_of_date(year):
    _root, _model, latest, df, _report = year
    live = df[df["completion_probability_source"] == "historical_runoff"]
    assert not live.empty
    as_of = pd.Timestamp(latest["pipeline_extract_date"])
    assert (live["expected_completion_date"] > as_of).all()


def test_only_forecast_cases_have_an_expected_completion_month(year):
    *_, df, report = year
    snap = pc.compute_pipeline_snapshot(df, report, {}, client_id="ERE", run_id="x")
    months = snap["expectedCompletionBreakdown"]
    live = df[df["completion_probability_source"] == "historical_runoff"]
    assert sum(r["caseCount"] for r in months) == len(live)


def test_the_pipeline_itself_is_unchanged(year):
    """Totals, counts and stage breakdowns do not depend on the forecast."""
    root, model, latest, df, report = year
    bare, bare_report = pc.load_prepared_pipeline(latest, historical_model=None)
    assert report["row_count"] == bare_report["row_count"] == len(df)
    assert report["total_pipeline_amount"] == bare_report["total_pipeline_amount"]
    assert report["stage_counts"] == bare_report["stage_counts"]


def test_a_case_settled_before_observation_began_is_not_timed(tmp_path):
    """Left truncation: an exit before the first snapshot has no observed
    timing, so it does not enter the hazards."""
    cases = [{"offer_date": "2025-01-01", "completed_on": "2025-01-20",
              "first_seen": {"COMPLETED": "2025-09-08"}, "final_stage": "COMPLETED",
              "seen_open": False}]
    model = runoff.fit_runoff(cases, "2025-09-08", "2026-09-24")
    assert model["stages"]["OFFER"]["advanced"] == 0
    assert model["stages"]["OFFER"]["windowEvidence"] == 1


def test_thin_history_measures_no_window_and_configures_none():
    """D21 (owner decision 2026-09-30): a validity window is measured from the
    client's history or there is none — no configured window stands in."""
    model = runoff.fit_runoff([], None, None)
    for st in ("KFI", "APPLICATION", "OFFER"):
        assert model["stages"][st]["windowBasis"] == "insufficient_history"
        assert model["stages"][st]["windowDays"] is None
        assert "fallbackWindowDays" not in model["stages"][st]
    assert model["available"] is False


def test_the_forecast_view_lists_only_forecast_months(year):
    """The Forecast tab's completion-month chart, like the Pipeline page's,
    reports only the cases carrying forecast weight — no £0 past months."""
    from mi_agent_api.workspace import forecast_breakdowns
    _root, _model, latest, df, _report = year
    months = forecast_breakdowns(None, df)["byCompletionMonth"]
    assert months
    as_of_month = latest["pipeline_extract_date"][:7]
    assert all(m["month"] >= as_of_month for m in months)
    assert all(m["weightedExpectedFundedAmount"] > 0 for m in months)


def test_the_blended_conversion_is_over_the_forecast_population(year):
    """KFIs and lapsed cases carry no weight, so they are outside the
    population the blended conversion and the forward case count describe."""
    *_, df, report = year
    s = report["completion_probability_summary"]
    live = df[df["completion_probability_source"].isin(
        ["historical_runoff", "historical_stage_rate", "configured_stage_rate",
         "row_level"])]
    assert s["excluded_count"] == len(df) - len(live)
    assert s["active_gross_amount"] == pytest.approx(
        float(live["current_outstanding_balance"].sum()), abs=1)
    # Weighted over live Applications and Offers: a pull-through-sized rate,
    # not one diluted by thousands of unweighted KFIs.
    assert 0.2 < s["blended_weighted_conversion"] < 0.8


def test_the_forward_loan_count_counts_only_weighted_cases(year):
    from mi_agent_api.forecast_bridge import compute_forecast_bridge
    _root, _model, latest, df, report = year
    snap = pc.compute_pipeline_snapshot(df, report, {}, client_id="ERE", run_id="x")
    funded = pd.DataFrame({"current_outstanding_balance": [100.0, 200.0]})
    out = compute_forecast_bridge(
        client_id="ERE", run_id="x", funded_reporting_date="2026-08-31",
        funded_df=funded, pipeline_df=df, pipeline_report=report,
        pipeline_snapshot=snap,
        pipeline_source={"pipeline_as_of_date": latest["pipeline_extract_date"]})
    bridge = out["forecastBridge"]
    weighted_cases = int(df["completion_probability_source"].isin(
        ["historical_runoff", "historical_stage_rate", "configured_stage_rate",
         "row_level"]).sum())
    assert bridge["eligibleCaseCount"] == weighted_cases
    assert bridge["forecastLoanCount"] == bridge["fundedLoanCount"] + weighted_cases


def test_the_methodology_version_moved_with_the_forecast():
    """Cached responses built by the old forecast must not be served (a
    browser holding an old ETag would otherwise get a 304)."""
    from mi_agent_api import serving_cache
    assert serving_cache.METHODOLOGY_VERSION != "1"
