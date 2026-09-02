"""Red-team: reporting periods constructed to make the briefing lie.

Every test here corresponds to a statement the Teams card DID make, in a
realistic month, before the fix beside it. They are regressions in the strict
sense: each one fails on the commit before this file.

The standard being held is not "the arithmetic is right". It is that a board
member, a warehouse funder or a rating agency reading only the card cannot form
a materially incorrect view of the portfolio.

Evidence class C (purpose-built adversarial fixtures) — labelled as such. The
frames are small because each isolates one economic shape; the same code is
exercised over pipeline-produced canonical in
``test_funded_composition_on_real_canonical.py``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mi_agent_api import evolution as ev
from mi_agent_api import funded_composition as fc
from trakt_notifications import portfolio_update, risk_review, sources
from trakt_notifications.contract import UPDATE_FUNDED

from .conftest import TENANT

B, L, P = "current_outstanding_balance", "loan_identifier", "source_portfolio_id"
LTV, PTYPE, LABEL = "current_loan_to_value", "source_portfolio_type", "source_portfolio_label"

CLEAR_CONCENTRATION = {
    "available": True, "reportingDate": "2026-07-31",
    "source": "approved_configuration", "tests": [], "emergingRisks": [],
    "states": {"available": True}, "lineage": {},
}


def frame(rows) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _wa_ltv_points(df):
    if LTV not in df.columns or not len(df):
        return None
    v, w = pd.to_numeric(df[LTV], errors="coerce"), pd.to_numeric(df[B], errors="coerce")
    m = v.notna() & w.notna()
    if not m.any() or float(w[m].sum()) <= 0:
        return None
    return round(float((v[m] * w[m]).sum() / w[m].sum()) * 100, 2)


def _movement(cur, pri):
    """A governed period movement consistent with the two frames."""
    cb, pb = float(cur[B].sum()), float(pri[B].sum())
    cl, pl = _wa_ltv_points(cur), _wa_ltv_points(pri)
    return {
        "available": True, "currentReportingDate": "2026-07-31",
        "priorReportingDate": "2026-06-30",
        "current": {"funded_balance": round(cb, 2), "loan_count": len(cur),
                    "wa_ltv_points": cl},
        "prior": {"funded_balance": round(pb, 2), "loan_count": len(pri),
                  "wa_ltv_points": pl},
        "delta": {"funded_balance": round(cb - pb, 2),
                  "loan_count": len(cur) - len(pri),
                  "wa_ltv_points": (round(cl - pl, 2)
                                    if cl is not None and pl is not None else None)},
        "regionContributions": [], "cohortMovements": [], "primaryRegion": None,
    }


@pytest.fixture
def period(monkeypatch):
    """Drive the production resolver over two stated frames.

    Stubs only at the governed SERVICE boundary — ``funded_frames``,
    ``period_movement``, ``compute_concentration_tests``. The resolver, the
    generators, the selector, the message contract and both card builders are
    the real ones.
    """
    from mi_agent_api import concentration_tests_api as conc_mod
    from mi_agent_api import movement_summary as ms

    state = {"current": None, "prior": None, "concentration": CLEAR_CONCENTRATION}

    def _frames(*a, **k):
        return [
            {"run_id": "mi_2026_06", "reporting_date": "2026-06-30",
             "df": state["prior"], "source": "/p"},
            {"run_id": "mi_2026_07", "reporting_date": "2026-07-31",
             "df": state["current"], "source": "/c"},
        ]

    monkeypatch.setattr(ev, "funded_frames", _frames)
    monkeypatch.setattr(ms, "period_movement",
                        lambda *a, **k: _movement(state["current"], state["prior"]))
    monkeypatch.setattr(conc_mod, "compute_concentration_tests",
                        lambda *a, **k: state["concentration"])

    def _run(current, prior, concentration=None):
        state["current"], state["prior"] = current, prior
        if concentration is not None:
            state["concentration"] = concentration
        inputs = sources.resolve(
            tenant_id=TENANT, portfolio_id=TENANT, portfolio_context="total",
            pipeline_root="/p", output_root="/f",
            want_pipeline=False, want_funded=True)
        return (inputs,
                portfolio_update.build(inputs, update_type=UPDATE_FUNDED),
                risk_review.build(inputs))

    return _run


def texts(message) -> str:
    return " ".join(i.text for i in message.items)


# =========================================================================== #
# Scenario 6 / 5 — an acquisition must not mask the incumbent book
# =========================================================================== #
MASK_PRIOR = frame([
    {L: "A1", B: 50_000_000.0, P: "alp_origination", LTV: 0.30, LABEL: "Direct"},
    {L: "A2", B: 50_000_000.0, P: "alp_origination", LTV: 0.30, LABEL: "Direct"}])
MASK_CURRENT = frame([
    {L: "A1", B: 50_000_000.0, P: "alp_origination", LTV: 0.38, LABEL: "Direct"},
    {L: "A2", B: 50_000_000.0, P: "alp_origination", LTV: 0.38, LABEL: "Direct"},
    {L: "B1", B: 100_000_000.0, P: "newco_book", LTV: 0.20, LABEL: "Portfolio B",
     PTYPE: "acquired"}])


def test_an_acquisition_cannot_hide_a_deteriorating_incumbent_book(period):
    """The card said "LTV moved from 30.0% to 29.0%" — an improvement.

    The incumbent book's LTV had risen 30% to 38%. The improvement was entirely
    the arriving book. A reader given only the combined figure concludes credit
    quality improved in a month it deteriorated by eight points.
    """
    _inputs, update, _risk = period(MASK_CURRENT, MASK_PRIOR)
    body = texts(update)

    assert "from 30.0% to 38.0% (+8.0pp)" in body
    assert "Excluding portfolios added this period" in body
    # The combined figure is not suppressed — it is just not the whole sentence.
    assert "combined book moved from 30.0% to 29.0% (-1.0pp)" in body
    assert "opposite direction to the underlying book's" in body


def test_the_underlying_movement_leads_the_combined_one(period):
    inputs, _u, _r = period(MASK_CURRENT, MASK_PRIOR)
    ltv = next(i for i in inputs.funded_insights()
               if i["insight_type"] == "FUNDED_LTV_MOVEMENT")

    assert ltv["metrics"]["underlying_change_pp"] == 8.0
    assert ltv["metrics"]["change_pp"] == -1.0
    assert ltv["headline"].startswith("Underlying weighted-average LTV")


def test_a_masked_deterioration_reaches_the_risk_review(period):
    """It is the finding a reader is least able to reach on their own."""
    _i, _u, risk = period(MASK_CURRENT, MASK_PRIOR)
    assert risk.severity == "attention"
    assert "underlying weighted-average LTV" in risk.headline
    assert risk_review.CLEAR_STATEMENT not in [i.text for i in risk.items]


def test_a_combined_only_ltv_move_stays_an_observation(period):
    """No addition, so no underlying/combined distinction and no risk finding."""
    prior = frame([{L: "A1", B: 100_000_000.0, P: "alp_origination", LTV: 0.30}])
    current = frame([{L: "A1", B: 100_000_000.0, P: "alp_origination", LTV: 0.32}])
    inputs, _u, risk = period(current, prior)

    ltv = next(i for i in inputs.funded_insights()
               if i["insight_type"] == "FUNDED_LTV_MOVEMENT")
    assert ltv["severity"] == "info"
    assert "underlying_change_pp" not in ltv["metrics"]
    assert risk.severity == "clear"


def test_a_material_underlying_move_survives_an_immaterial_combined_one(period):
    """The gate must not be the combined figure alone.

    An arriving book can hold the combined move under the threshold while the
    incumbent book moves a long way; gating on combined would report the month
    as having no LTV movement at all.
    """
    prior = frame([{L: "A1", B: 100_000_000.0, P: "alp", LTV: 0.30}])
    current = frame([
        {L: "A1", B: 100_000_000.0, P: "alp", LTV: 0.34},
        {L: "B1", B: 100_000_000.0, P: "newco", LTV: 0.26, PTYPE: "acquired",
         LABEL: "Portfolio B"}])

    inputs, _u, _r = period(current, prior)
    ltv = next(i for i in inputs.funded_insights()
               if i["insight_type"] == "FUNDED_LTV_MOVEMENT")
    # Combined 30.0 -> 30.0 (0.0pp, immaterial); underlying +4.0pp (material).
    assert ltv["metrics"]["change_pp"] == 0.0
    assert ltv["metrics"]["underlying_change_pp"] == 4.0


def test_mix_is_measured_on_the_population_present_in_both_periods(period):
    """A mix share computed across an arriving book describes the arrival."""
    inputs, _u, _r = period(MASK_CURRENT, MASK_PRIOR)
    for insight in inputs.funded_insights():
        if insight["insight_type"] == "FUNDED_MIX_SHIFT":
            assert insight["metrics"]["population"] == "underlying"


# =========================================================================== #
# Scenario 5 — an addition larger than the net movement
# =========================================================================== #
OFFSET_PRIOR = frame([
    {L: "A1", B: 60_000_000.0, P: "alp", LTV: 0.30},
    {L: "A2", B: 40_000_000.0, P: "alp", LTV: 0.30}])
OFFSET_CURRENT = frame([
    {L: "A1", B: 55_000_000.0, P: "alp", LTV: 0.42},
    {L: "B1", B: 50_000_000.0, P: "newco", LTV: 0.25, PTYPE: "acquired",
     LABEL: "Portfolio B"}])


def test_an_addition_larger_than_the_movement_is_not_stated_as_a_share(period):
    """The card said "£50.0m of the £5.0m movement". That is not a sentence.

    An acquisition alongside heavy redemptions is the most ordinary shape an
    acquisition month takes, and it gives a share of 1000%.
    """
    _i, update, _r = period(OFFSET_CURRENT, OFFSET_PRIOR)
    body = texts(update)

    assert "of the £5.0m movement" not in body
    assert "added £50.0m, against a net movement of +£5.0m" in body
    assert "smaller than the addition because £40.0m of redemptions and exits" in body


def test_the_share_is_withheld_rather_than_formatted_as_nonsense():
    decomposition = fc.decompose(OFFSET_CURRENT, OFFSET_PRIOR)
    lead = fc.dominant_addition(decomposition)

    assert lead["exceeds_movement"] is True
    assert lead["share_of_movement"] is None      # nothing to format
    assert lead["share_of_closing_balance"] == pytest.approx(0.4762, abs=1e-3)


def test_a_normal_acquisition_still_carries_its_share():
    """The guard must not remove the share where it does mean something."""
    lead = fc.dominant_addition(fc.decompose(MASK_CURRENT, MASK_PRIOR))
    assert lead["exceeds_movement"] is False
    assert lead["share_of_movement"] == 1.0


# =========================================================================== #
# Scenarios 12 / 18 — the card contradicting its own summary
# =========================================================================== #
@pytest.mark.parametrize("current, prior, label", [
    (frame([{L: "T1", B: 220_000.0, P: "alp", LTV: 0.30}]),
     frame([{L: "T1", B: 220_000.0, P: "alp", LTV: 0.30},
            {L: "T2", B: 180_000.0, P: "alp", LTV: 0.30}]),
     "a tiny book losing 45% to one redemption"),
    (frame([{L: "P1", B: 100_000_000.0, P: "alp", LTV: 0.30},
            {L: "P3", B: 40_000_000.0, P: "alp", LTV: 0.30}]),
     frame([{L: "P1", B: 100_000_000.0, P: "alp", LTV: 0.30}]),
     "a book growing 40%"),
])
def test_a_material_month_never_says_nothing_material_happened(period, current,
                                                               prior, label):
    """The card printed both, one line apart.

    A month whose only material finding is the headline movement produces no
    bullet for it — the lead sentence already says it — and the card then
    declared nothing material had happened directly underneath a lead sentence
    reporting -45%.
    """
    inputs, update, _r = period(current, prior)

    assert inputs.funded_insights(), label
    body = texts(update)
    assert "No material developments were identified" not in body, label


def test_a_genuinely_quiet_month_still_says_so(period):
    """The fix must not remove the statement where it is true."""
    prior = frame([{L: "A1", B: 100_000_000.0, P: "alp", LTV: 0.30}])
    current = frame([{L: "A1", B: 100_100_000.0, P: "alp", LTV: 0.30}])

    inputs, update, _r = period(current, prior)
    assert inputs.funded_insights() == []
    assert "No material developments were identified in the funded book " \
           "this period." in texts(update)


# =========================================================================== #
# The null-drop in the governed grouping
# =========================================================================== #
def test_a_null_dimension_value_is_bucketed_not_dropped():
    """``_group_balance`` dropped null rows from the numerator AND denominator.

    ``astype(str)`` leaves a real NaN as NaN rather than the string "nan", so
    the Unknown mask never fired and groupby discarded those rows. On a £140m
    book with £90m of unset product it returned Lump Sum as 100% of a £50m
    total — a published share of 100% where the truth was 35.7%.

    A column read from a CSV with blank cells is exactly that shape.
    """
    import io

    df = pd.read_csv(io.StringIO(
        "loan_identifier,current_outstanding_balance,erm_product_type\n"
        "L1,50000000,Lump Sum\nL2,50000000,\nL3,40000000,\n"))

    groups = ev._group_balance(df, "erm_product_type")

    assert sum(groups.values()) == pytest.approx(140_000_000.0)
    assert groups["Unknown / Missing"] == pytest.approx(90_000_000.0)
    assert groups["Lump Sum"] / sum(groups.values()) == pytest.approx(0.357, abs=1e-3)


def test_an_empty_string_and_a_null_reach_the_same_bucket():
    df = pd.DataFrame([
        {B: 10.0, "d": "A"}, {B: 10.0, "d": ""}, {B: 10.0, "d": None},
        {B: 10.0, "d": "  "}])
    groups = ev._group_balance(df, "d")

    assert groups == {"A": 10.0, "Unknown / Missing": 30.0}
    assert sum(groups.values()) == 40.0


def test_the_bridge_reconciliation_property_holds_over_nulls():
    """The bridge claims per-category deltas sum exactly to the net change.

    Dropped rows were in the net change and in no category, so the claim was
    false wherever a dimension had nulls.
    """
    prior = pd.DataFrame([{B: 100.0, "d": "A"}, {B: 100.0, "d": None}])
    current = pd.DataFrame([{B: 100.0, "d": "A"}, {B: 150.0, "d": None}])

    pri_g, cur_g = ev._group_balance(prior, "d"), ev._group_balance(current, "d")
    net = float(current[B].sum()) - float(prior[B].sum())
    per_category = sum(cur_g.get(k, 0.0) - pri_g.get(k, 0.0)
                       for k in set(cur_g) | set(pri_g))

    assert per_category == pytest.approx(net)


# =========================================================================== #
# Scenario 21 — a busy month must rank, not dump
# =========================================================================== #
def test_a_busy_month_is_ranked_and_capped(period):
    warning = {
        **CLEAR_CONCENTRATION,
        "tests": [{"testId": "ldn", "displayName": "London exposure",
                   "status": "warning", "priorStatus": "pass",
                   "statusTransition": "pass -> warning", "deteriorated": True,
                   "currentValue": 0.31, "threshold": 0.30, "utilization": 0.97,
                   "headroom": 0.9, "unit": "percent",
                   "reportingDate": "2026-07-31"}]}
    prior = frame([
        {L: "A1", B: 40_000_000.0, P: "alp", LTV: 0.30},
        {L: "A2", B: 30_000_000.0, P: "alp", LTV: 0.30},
        {L: "A3", B: 30_000_000.0, P: "alp", LTV: 0.30}])
    current = frame([
        {L: "A1", B: 40_000_000.0, P: "alp", LTV: 0.36},
        {L: "A2", B: 30_000_000.0, P: "alp", LTV: 0.36},
        {L: "A4", B: 25_000_000.0, P: "alp", LTV: 0.36},
        {L: "B1", B: 60_000_000.0, P: "newco", LTV: 0.22, PTYPE: "acquired",
         LABEL: "Portfolio B"}])

    _i, update, risk = period(current, prior, concentration=warning)

    # Ranked: the limit crossing outranks the acquisition, which outranks the
    # underlying LTV move, which outranks the underlying balance move.
    body = [i.text for i in update.items]
    assert "London exposure deteriorated" in body[1]
    assert "acquisition of Portfolio B" in body[2]
    assert "balance-weighted current LTV" in body[3]

    # Within the card budget without needing to truncate: six governed insights
    # were produced and the five that fit are the five most material. A reader
    # gets the month in five bullets, not a reproduction of the MI.
    assert len(update.items) == 5
    assert risk.severity == "attention"

    # And nothing immaterial reached it: every bullet after the loan count is a
    # governed insight, not a statistic included because it was available.
    assert not any("was unchanged" in t for t in body)
