#!/usr/bin/env python3
"""Observed historical behaviour, checked against independently planted truth.

The fixture (``tests/history_portfolio.py``) is 18 monthly snapshots in which
the SAME loans persist, so behaviour is observable rather than asserted: a loan
goes 30 → 60 → 90 DPD and stays there, another cures, others redeem in waves.
Every expectation below is a literal from that fixture's own truth table, worked
out by hand.

The two sharpest tests here:

* ``test_prepayment_is_not_inferred_from_balance_movement`` — a book whose
  balance falls purely through scheduled amortisation must report a prepayment
  rate of zero. Inferring from balance change would report a healthy-looking
  rate for a book where nobody prepaid anything.
* ``test_every_loss_denominator_is_reported`` — three legitimate denominators
  give 0.428%, 0.380% and 0.453% on the same portfolio. Silently choosing one is
  how a book gets described as materially better or worse than it is.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analytics_lib.history import (
    UNSCHEDULED_PRINCIPAL_FIELD,
    loss_and_recovery,
    prepayment_rate,
)
from trakt_core import config_cache
from trakt_core.context import SCOPE_LOAN_READ, SCOPE_RISK_READ
from trakt_core.envelope import STATUS_BLOCKED, STATUS_SUCCESS
from trakt_core.errors import ErrorCode
from trakt_tools.execution import ToolDependencies, execute_governed_tool

from tests.history_portfolio import (
    ARREARS_30_CLOSING_PCT,
    ARREARS_30_OPENING_PCT,
    ARREARS_90_CLOSING_PCT,
    CLOSING_LOAN_COUNT,
    CURING_RETURNS_AT,
    DETERIORATING_REACHES_90_AT,
    LAST_PERIOD,
    LONDON_CLOSING_PCT,
    OPENING_LOAN_COUNT,
    PERIOD_COUNT,
    PERIODS,
    RECOVERY_RATE_ON_LOSSES,
    STRONG_VINTAGE,
    TOTAL_ALLOCATED_LOSSES,
    TOTAL_RECOVERIES,
    WEAK_VINTAGE,
    history,
    redemptions_in_period,
    snapshot_id,
)
from tests.planted_portfolio import planted_lineage_index
from tests.test_agent_governed_execution import _Datasets, _Descriptor
from tests.test_agent_loan_retrieval import (
    PORTFOLIO_A,
    PORTFOLIO_B,
    SPV_I,
    TENANT,
    _catalogue,
    _context,
    _CountingResolver,
)

BOTH = (SCOPE_LOAN_READ, SCOPE_RISK_READ)


@pytest.fixture(scope="module")
def snapshots() -> Dict[str, pd.DataFrame]:
    return history()


@pytest.fixture(autouse=True)
def _clean_cache():
    config_cache.reset()
    yield
    config_cache.reset()


def _deps(snapshots, resolver=None) -> ToolDependencies:
    return ToolDependencies(
        datasets=_Datasets(_Descriptor(snapshot_id=snapshot_id(LAST_PERIOD))),
        runtime_mode="test", catalogue=_catalogue(), output_root="/unused",
        loan_frame_resolver=resolver or _CountingResolver(snapshots[LAST_PERIOD]),
        loan_history_resolver=lambda: snapshots,
        lineage_index=planted_lineage_index(TENANT, snapshot_id(LAST_PERIOD)))


def _ok(tool: str, args: Dict[str, Any], snapshots, **kwargs):
    result = execute_governed_tool(
        tool, args, kwargs.pop("context", None) or _context(capabilities=BOTH),
        dependencies=kwargs.pop("deps", None) or _deps(snapshots))
    assert result.status == STATUS_SUCCESS, result.error and result.error.to_dict()
    return result.result


# =========================================================================== #
# The fixture is genuinely a history, not twelve unrelated portfolios
# =========================================================================== #
def test_the_same_loans_persist_across_periods(snapshots):
    """The whole premise. Unrelated portfolios would make behaviour
    unobservable — a loan has to be followable."""
    first = set(snapshots[PERIODS[0]]["loan_identifier"])
    last = set(snapshots[LAST_PERIOD]["loan_identifier"])
    assert len(first) == OPENING_LOAN_COUNT
    assert len(last) == CLOSING_LOAN_COUNT
    # The closing book is a strict subset: loans leave, none appear.
    assert last < first
    # And most of them are the same loans, not a coincidental overlap.
    assert len(first & last) / len(first) > 0.85


def test_a_single_loan_can_be_followed_through_a_delinquency_and_a_cure(snapshots):
    """The behaviour the fixture exists to make observable."""
    curing = snapshots[PERIODS[0]]
    # Any loan in the curing cohort: Scotland, since only that cohort is there.
    loan_id = curing.loc[curing["geographic_region_collateral"] == "Scotland",
                         "loan_identifier"].iloc[0]

    track = []
    for period in PERIODS:
        frame = snapshots[period]
        row = frame.loc[frame["loan_identifier"] == loan_id]
        track.append(int(row["number_of_days_in_arrears"].iloc[0]) if len(row)
                     else None)

    assert track[0] == 0
    assert max(track[:8]) >= 90, "the loan must actually reach 90 DPD"
    assert track[CURING_RETURNS_AT] == 0, "and must return to performing"


def test_the_deteriorating_cohort_reaches_90_and_stays(snapshots):
    wales = snapshots[PERIODS[DETERIORATING_REACHES_90_AT]]
    deteriorating = wales.loc[
        (wales["geographic_region_collateral"] == "Wales")
        & (wales["account_status"] != "defaulted")]
    assert (deteriorating["number_of_days_in_arrears"] >= 90).all()
    later = snapshots[LAST_PERIOD]
    still = later.loc[(later["geographic_region_collateral"] == "Wales")
                      & (later["account_status"] != "defaulted")]
    assert (still["number_of_days_in_arrears"] >= 90).all()


def test_redemptions_remove_loans_rather_than_zeroing_them(snapshots):
    """A redeemed loan is ABSENT, which is what makes the exit invisible on the
    closing tape and forces the set-difference derivation."""
    before = set(snapshots[PERIODS[3]]["loan_identifier"])
    after = set(snapshots[PERIODS[4]]["loan_identifier"])
    gone = before - after
    assert gone, "period 4 is a redemption wave"
    assert not any(snapshots[PERIODS[4]]["loan_identifier"].isin(gone))


# =========================================================================== #
# Prepayment — the methodology, and the trap it avoids
# =========================================================================== #
def test_the_observed_prepayment_rate_matches_the_planted_activity(snapshots):
    result = prepayment_rate(snapshots, periods=list(PERIODS))
    assert result.available
    # v2 since Sprint 2.5C: the denominator now nets out scheduled principal.
    assert result.method == "OBSERVED_CPR@v2"
    assert result.periods_observed == PERIOD_COUNT - 1

    expected_redemptions = sum(redemptions_in_period(i)
                               for i in range(PERIOD_COUNT))
    assert result.redemptions == pytest.approx(expected_redemptions, rel=1e-9)
    assert result.unscheduled_principal > 0


def test_prepayment_is_not_inferred_from_balance_movement(snapshots):
    """The single most important assertion in this file.

    A book that amortises on schedule and nothing else has prepaid NOTHING. A
    rate inferred from the change in total balance would report a healthy-looking
    figure anyway, and it would be wrong in a way that looks entirely plausible.
    """
    # A constant population: only loans present in EVERY period of the window,
    # so nothing exits and the only balance movement is scheduled amortisation.
    window = list(PERIODS[:6])
    persistent = set(snapshots[window[0]]["loan_identifier"])
    for period in window[1:]:
        persistent &= set(snapshots[period]["loan_identifier"])

    scheduled_only = {}
    for period in window:
        frame = snapshots[period].copy()
        frame[UNSCHEDULED_PRINCIPAL_FIELD] = 0.0
        scheduled_only[period] = frame[
            frame["loan_identifier"].isin(persistent)]

    balances = [float(scheduled_only[p]["current_principal_balance"].sum())
                for p in window]
    assert balances[-1] < balances[0], "the balance must genuinely fall"
    counts = {len(scheduled_only[p]) for p in window}
    assert len(counts) == 1, "the population must be constant for this test"

    result = prepayment_rate(scheduled_only, periods=window)
    assert result.available
    assert result.smm_pct == pytest.approx(0.0, abs=1e-9), (
        "a falling balance with no unscheduled principal is amortisation, "
        "not prepayment")


def test_a_tape_without_the_unscheduled_field_refuses_rather_than_guessing(
        snapshots):
    stripped = {p: snapshots[p].drop(columns=[UNSCHEDULED_PRINCIPAL_FIELD])
                for p in PERIODS[:4]}
    result = prepayment_rate(stripped, periods=list(PERIODS[:4]))
    assert result.available is False
    assert "A rate is NOT inferred from the change in total balance" in result.reason


def test_exits_are_classified_before_being_counted_as_redemptions(snapshots):
    """UPDATED in Sprint 2.5C. This test previously asserted that exits were
    "derived" by set difference and that the rate "counts EVERY exit" — which
    was the Sprint 2.5B behaviour and was methodologically wrong.

    A disappearance is not a redemption. A loan can leave a tape because it
    defaulted and was written off, reached maturity, was sold, or because the
    extract broke. Counting all of them as voluntary prepayment inflates the
    rate most in exactly the case where it matters — a book shedding defaulted
    loans. Exits are now classified, and only evidenced redemptions count.
    """
    result = prepayment_rate(snapshots, periods=list(PERIODS))
    classified = [p for p in result.per_period
                  if p["redemption_source"].startswith("classified")]
    assert classified
    assert any("a disappearance is not a redemption" in note.lower()
               for note in result.notes)

    # Every exit in this fixture carries a redemption flag, so none is unknown.
    for period in result.per_period:
        exits = period.get("exits") or {}
        assert exits.get("unknown_exit", {}).get("balance", 0.0) == 0.0


def test_the_annualised_form_is_labelled_as_an_annualisation(snapshots):
    result = prepayment_rate(snapshots, periods=list(PERIODS), annualise=True)
    assert result.cpr_pct is not None and result.smm_pct is not None
    assert result.cpr_pct > result.smm_pct
    assert any("not a forecast" in note for note in result.notes)

    unannualised = prepayment_rate(snapshots, periods=list(PERIODS),
                                   annualise=False)
    assert unannualised.cpr_pct is None
    assert unannualised.smm_pct == pytest.approx(result.smm_pct)


def test_one_period_cannot_produce_a_rate(snapshots):
    result = prepayment_rate({PERIODS[0]: snapshots[PERIODS[0]]},
                             periods=[PERIODS[0]])
    assert result.available is False
    assert "at least two consecutive snapshots" in result.reason


# =========================================================================== #
# Loss and recovery
# =========================================================================== #
def test_losses_and_recoveries_match_the_planted_amounts(snapshots):
    """Rewritten in Sprint 2.5C. The two assertions removed here pinned a
    methodology the regime contradicts: ``recovery_rate_on_losses_pct`` divided
    recoveries by a loss figure that RREL73 already states net of them, and
    ``net_loss`` deducted them from it a second time. Both are withdrawn at
    OBSERVED_LOSS@v2 — see ``analytics_lib.history._loss_severity``."""
    outcome = loss_and_recovery(snapshots, periods=list(PERIODS))
    assert outcome["available"]
    assert outcome["cumulative_losses"] == pytest.approx(TOTAL_ALLOCATED_LOSSES)
    assert outcome["cumulative_recoveries"] == pytest.approx(TOTAL_RECOVERIES)
    assert outcome["recoveries_against_residual_loss_pct"] == pytest.approx(
        RECOVERY_RATE_ON_LOSSES)
    assert "net_loss" not in outcome


def test_every_loss_denominator_is_reported(snapshots):
    """Several legitimate definitions exist and they give materially different
    answers on the same book. Choosing one silently is how a portfolio gets
    described as better or worse than it is."""
    outcome = loss_and_recovery(snapshots, periods=list(PERIODS))
    rates = {
        "original": outcome["loss_rate_on_original_pct"],
        "opening": outcome["loss_rate_on_opening_pct"],
        "current": outcome["loss_rate_on_current_pct"],
    }
    assert all(v is not None for v in rates.values())
    assert len(set(round(v, 4) for v in rates.values())) == 3, (
        "the fixture is built so the three denominators genuinely differ")
    # And a denominator that shrinks produces a HIGHER rate, which is the trap.
    assert rates["current"] > rates["original"]


def test_a_tape_without_the_loss_field_refuses_rather_than_inferring(snapshots):
    stripped = {p: snapshots[p].drop(columns=["allocated_losses"])
                for p in PERIODS[:3]}
    outcome = loss_and_recovery(stripped, periods=list(PERIODS[:3]))
    assert outcome["available"] is False
    assert "No loss figure is inferred from balance movement" in outcome["reason"]


def test_cumulative_fields_are_differenced_not_summed(snapshots):
    """allocated_losses is CUMULATIVE: summing it across 18 snapshots would
    count the same loss eighteen times."""
    outcome = loss_and_recovery(snapshots, periods=list(PERIODS))
    naive_sum = sum(float(snapshots[p]["allocated_losses"].sum())
                    for p in PERIODS)
    assert naive_sum > outcome["cumulative_losses"] * 5
    assert outcome["cumulative_losses"] == pytest.approx(TOTAL_ALLOCATED_LOSSES)


# =========================================================================== #
# The tools
# =========================================================================== #
def test_portfolio_history_reproduces_the_planted_series(snapshots):
    payload = _ok("portfolio_history",
                  {"resource": PORTFOLIO_A.key,
                   "measures": ["arrears_30_plus_pct", "arrears_90_plus_pct",
                                "largest_region_share_pct", "loan_count"]},
                  snapshots)
    assert payload["period_count"] == PERIOD_COUNT

    arrears = payload["series"]["arrears_30_plus_pct"]
    assert arrears["first"] == pytest.approx(ARREARS_30_OPENING_PCT, abs=0.01)
    assert arrears["last"] == pytest.approx(ARREARS_30_CLOSING_PCT, abs=0.01)
    assert arrears["direction"] == "rising"

    assert payload["series"]["arrears_90_plus_pct"]["last"] == pytest.approx(
        ARREARS_90_CLOSING_PCT, abs=0.01)
    assert payload["series"]["loan_count"]["first"] == OPENING_LOAN_COUNT
    assert payload["series"]["loan_count"]["last"] == CLOSING_LOAN_COUNT


def test_concentration_drift_is_observed_rather_than_authored(snapshots):
    """No loan's region ever changes. The share moves because other loans leave,
    which is the only honest way to plant a drift."""
    regions = {p: set(snapshots[p]["geographic_region_collateral"].unique())
               for p in (PERIODS[0], LAST_PERIOD)}
    assert regions[PERIODS[0]] == regions[LAST_PERIOD]

    payload = _ok("portfolio_history",
                  {"resource": PORTFOLIO_A.key,
                   "measures": ["largest_region_share_pct"]}, snapshots)
    series = payload["series"]["largest_region_share_pct"]
    assert series["last"] == pytest.approx(LONDON_CLOSING_PCT, abs=0.05)
    assert series["direction"] == "rising"
    assert series["change"] > 2.0


def test_the_history_costs_one_pass_per_snapshot_not_per_measure(snapshots):
    """Eight measures over eighteen periods is eighteen frame walks, not 144."""
    payload = _ok("portfolio_history",
                  {"resource": PORTFOLIO_A.key,
                   "measures": ["total_balance", "loan_count", "wa_ltv",
                                "arrears_30_plus_pct", "arrears_60_plus_pct",
                                "arrears_90_plus_pct", "high_ltv_share_pct",
                                "largest_region_share_pct"]}, snapshots)
    assert payload["_telemetry"]["counters"]["snapshots_read"] == PERIOD_COUNT


def test_an_unknown_measure_is_refused_with_the_available_ones(snapshots):
    result = execute_governed_tool(
        "portfolio_history",
        {"resource": PORTFOLIO_A.key, "measures": ["make_something_up"]},
        _context(capabilities=BOTH), dependencies=_deps(snapshots))
    assert result.error_code == ErrorCode.TOOL_INPUT_INVALID
    assert "no expression language" in result.error.message
    assert "available" in (result.error.details or {})


def test_the_window_is_bounded_and_narrowable(snapshots):
    payload = _ok("portfolio_history",
                  {"resource": PORTFOLIO_A.key, "months": 6,
                   "measures": ["loan_count"]}, snapshots)
    assert payload["period_count"] == 6
    assert payload["periods"] == list(PERIODS[-6:])


def test_transition_analysis_reuses_the_risk_monitor_implementation(snapshots):
    """Not reimplemented: a transition an agent reads and one the Risk Monitor
    shows are the same computation."""
    payload = _ok("transition_analysis",
                  {"resource": PORTFOLIO_A.key, "dimension": "account_status",
                   "from_period": PERIODS[0], "to_period": PERIODS[8]},
                  snapshots)
    assert payload["calculation_source"] == (
        "mi_agent.risk_monitor.migration.migration_matrix")

    movements = {(r["from_value"], r["to_value"]): r
                 for r in payload["transitions"]}
    # 40 deteriorating + 10 defaulting went performing -> arrears by period 8.
    changed = movements[("performing", "arrears")]
    assert changed["loan_count"] == 50
    # 16 loans had redeemed by period 8 (waves at 4, 6 and 8: 5 + 5 + 6).
    exited = movements[("performing", None)]
    assert exited["loan_count"] == 16
    assert exited["movement_type"] == "exited"


def test_cohort_comparison_finds_the_weak_vintage_without_being_told(snapshots):
    """The fixture never labels a vintage as weak. The comparison discovers it."""
    payload = _ok("cohort_comparison",
                  {"resource": PORTFOLIO_A.key,
                   "cohort_field": "portfolio_cohort",
                   "cohort_value": str(WEAK_VINTAGE),
                   "measures": ["arrears_30_plus_pct", "wa_ltv"]}, snapshots)

    arrears = payload["measures"]["arrears_30_plus_pct"]
    assert arrears["cohort"] > arrears["remainder"]
    assert arrears["difference"] > 10.0
    # The strong vintage genuinely has no arrears, which is the sharpest
    # available evidence the comparison reads the right population.
    assert arrears["remainder"] == pytest.approx(0.0, abs=1e-9)


def test_a_cohort_figure_is_never_returned_without_its_comparator(snapshots):
    payload = _ok("cohort_comparison",
                  {"resource": PORTFOLIO_A.key,
                   "cohort_field": "current_loan_to_value", "above": 70,
                   "measures": ["arrears_30_plus_pct"]}, snapshots)
    measure = payload["measures"]["arrears_30_plus_pct"]
    assert "cohort" in measure and "remainder" in measure
    assert payload["cohort"]["loan_count"] and payload["remainder"]["loan_count"]
    assert "only interpretable against the rest" in payload["note"]


def test_an_empty_cohort_is_not_reported_as_good_performance(snapshots):
    payload = _ok("cohort_comparison",
                  {"resource": PORTFOLIO_A.key,
                   "cohort_field": "portfolio_cohort", "cohort_value": "1999",
                   "measures": ["arrears_30_plus_pct"]}, snapshots)
    assert payload["cohort"]["loan_count"] == 0
    assert any("not evidence the cohort performs well" in w
               for w in payload["warnings"])


def test_prepayment_and_loss_tools_report_their_methodology(snapshots):
    prepayment = _ok("prepayment_analysis", {"resource": PORTFOLIO_A.key},
                     snapshots)
    # UPDATED in Sprint 2.5C. The method identifier moved v1 -> v2 and the note
    # changed because the DENOMINATOR changed: external verification established
    # that the market-standard SMM divides by the beginning balance LESS
    # scheduled principal, not by the full beginning balance. The old assertion
    # pinned the superseded methodology.
    assert prepayment["method"] == "OBSERVED_CPR@v2"
    assert any("scheduled principal" in n.lower() for n in prepayment["notes"])
    assert any("none of them is prepayment" in n for n in prepayment["notes"])

    losses = _ok("loss_analysis", {"resource": PORTFOLIO_A.key}, snapshots)
    # v2: the net-of-recoveries outputs were withdrawn once RREL73's own
    # wording showed they deducted recoveries twice.
    assert losses["method"] == "OBSERVED_LOSS@v2"
    assert any("different question" in n for n in losses["notes"])


def test_no_history_is_reported_as_absent_not_as_stability(snapshots):
    deps = ToolDependencies(
        datasets=_Datasets(_Descriptor(snapshot_id="s1")), runtime_mode="test",
        catalogue=_catalogue(), output_root="/unused",
        loan_frame_resolver=_CountingResolver(snapshots[LAST_PERIOD]),
        loan_history_resolver=None)
    result = execute_governed_tool("portfolio_history",
                                   {"resource": PORTFOLIO_A.key},
                                   _context(capabilities=BOTH), dependencies=deps)
    assert result.status != STATUS_SUCCESS
    assert "not a finding of stability" in result.error.message


# =========================================================================== #
# Governance holds across the historical surface
# =========================================================================== #
HISTORY_TOOLS = (
    ("portfolio_history", {}),
    ("prepayment_analysis", {}),
    ("loss_analysis", {}),
    ("cohort_comparison", {"cohort_field": "portfolio_cohort",
                           "cohort_value": "2024"}),
    ("transition_analysis", {}),
)


@pytest.mark.parametrize("tool,extra", HISTORY_TOOLS)
def test_an_ungranted_resource_is_refused(tool, extra, snapshots):
    result = execute_governed_tool(tool, {"resource": PORTFOLIO_B.key, **extra},
                                   _context(capabilities=BOTH),
                                   dependencies=_deps(snapshots))
    assert result.status == STATUS_BLOCKED, tool
    assert result.error_code == ErrorCode.RESOURCE_NOT_AUTHORISED, tool


@pytest.mark.parametrize("tool,extra", HISTORY_TOOLS)
def test_every_history_tool_states_its_scope_and_cost(tool, extra, snapshots):
    payload = _ok(tool, {"resource": PORTFOLIO_A.key, **extra}, snapshots)
    assert payload["scope"]["resource"] == PORTFOLIO_A.key
    assert payload["_telemetry"]["tool"] == tool


def test_the_history_tools_need_only_the_aggregate_capability():
    """None returns a per-loan series, so none needs loan:read."""
    from trakt_tools.registry import get

    for tool, _extra in HISTORY_TOOLS:
        assert get(tool).required_capability == SCOPE_RISK_READ, tool


def test_no_history_tool_holds_its_own_calculation():
    """Structural: the calculations live in analytics_lib so React and MI Query
    reach them too. A securitisation-specific analytics layer is what this
    sprint must not create."""
    import inspect

    from trakt_tools.handlers import history as module

    source = inspect.getsource(module)
    assert "from analytics_lib.history import" in source
    assert "migration_matrix" in source
    for forbidden in ("def _smm", "def _cpr", "def _loss_rate", "** 12"):
        assert forbidden not in source, (
            f"trakt_tools.handlers.history computes with {forbidden!r}")


# =========================================================================== #
# Cost
# =========================================================================== #
def test_a_full_history_sweep_is_cheap_enough_to_ask_for_routinely(snapshots,
                                                                    capsys):
    """The whole point of bounding the surface: if a trend is expensive, a
    reviewer stops asking for one."""
    args = {"resource": PORTFOLIO_A.key,
            "measures": ["total_balance", "loan_count", "wa_ltv",
                         "arrears_30_plus_pct", "arrears_90_plus_pct",
                         "high_ltv_share_pct", "largest_region_share_pct"]}
    _ok("portfolio_history", args, snapshots)          # warm

    started = time.perf_counter()
    payload = _ok("portfolio_history", args, snapshots)
    elapsed_ms = (time.perf_counter() - started) * 1000

    rows = sum(len(snapshots[p]) for p in PERIODS)
    with capsys.disabled():
        print(f"\n    {PERIOD_COUNT} periods x 7 measures over {rows:,} rows"
              f"\n    elapsed          : {elapsed_ms:8.0f} ms"
              f"\n    snapshots read   : "
              f"{payload['_telemetry']['counters']['snapshots_read']}")

    assert payload["_telemetry"]["counters"]["snapshots_read"] == PERIOD_COUNT
    assert elapsed_ms < 5_000
