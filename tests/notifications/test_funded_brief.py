"""The monthly funded review has a materiality layer, and states what it hid.

Before this the funded card printed its figures unconditionally: a £4k month and
a £24m month produced the same three bullets, with no threshold behind them and
no record of anything suppressed. These tests hold the funded generators to the
same discipline the weekly ones already keep — gate on configured thresholds,
record an explicit omission for everything below them, and never let a quiet
month look like an unexamined one.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mi_agent_api import insight_config as cfg
from mi_agent_api import insight_engine as engine
from mi_agent_api import insight_generators_funded as fgen
from mi_agent_api.insight_contract import (
    FUNDED_COMPOSITION, FUNDED_LTV_MOVEMENT, FUNDED_MIX_SHIFT, FUNDED_MOVEMENT,
    OMITTED_IMMATERIAL, RISK_LIMIT_TRANSITION, SEVERITY_ATTENTION,
    SEVERITY_CONCERN, TYPE_PRIORITY, UNDERLYING_BOOK_MOVEMENT,
)

CTX = {"tenant_id": "T", "portfolio_id": "portfolio_alpha",
       "portfolio_context": "total", "run_id": "mi_2026_07",
       "as_of_date": "2026-07-31", "comparison_date": "2026-06-30"}


def _movement(*, balance=184_000_000.0, opening=180_000_000.0,
              ltv=30.2, prior_ltv=30.0, loans=980, prior_loans=960) -> dict:
    return {
        "available": True,
        "currentReportingDate": "2026-07-31",
        "priorReportingDate": "2026-06-30",
        "current": {"funded_balance": balance, "loan_count": loans,
                    "wa_ltv_points": ltv},
        "prior": {"funded_balance": opening, "loan_count": prior_loans,
                  "wa_ltv_points": prior_ltv},
        "delta": {"funded_balance": round(balance - opening, 2),
                  "loan_count": loans - prior_loans,
                  "wa_ltv_points": round(ltv - prior_ltv, 2)},
        "regionContributions": [], "cohortMovements": [],
    }


# --------------------------------------------------------------------------- #
# Headline movement — the gate that did not exist
# --------------------------------------------------------------------------- #
def test_a_material_month_is_reported():
    insights, omissions = fgen.funded_movement(CTX, _movement())
    assert not omissions
    assert insights[0].metrics["change"] == 4_000_000.0
    # 4m on 180m ≈ 2.22%, above the 1.0% gate.
    assert insights[0].metrics["change_pct"] == pytest.approx(2.22, abs=0.01)


def test_an_immaterial_month_is_suppressed_and_says_so():
    """Below threshold produces an omission, never silence.

    Silence is what "we did not look" sounds like, and the whole reason the
    weekly brief records omissions is that a reader cannot tell the two apart.
    """
    quiet = _movement(balance=180_500_000.0, opening=180_000_000.0)
    insights, omissions = fgen.funded_movement(CTX, quiet)

    assert insights == []
    assert len(omissions) == 1
    assert omissions[0].category == OMITTED_IMMATERIAL
    assert "below the 1.0% materiality threshold" in omissions[0].reason


def test_the_threshold_comes_from_configuration_not_from_code(tmp_path,
                                                              monkeypatch):
    """A deployment can tune the gate without a code change."""
    path = tmp_path / "insights.yaml"
    path.write_text("insights:\n  funded_movement:\n    min_change_pct: 5.0\n",
                    encoding="utf-8")
    monkeypatch.setenv("TRAKT_MI_INSIGHTS_CONFIG", str(path))
    cfg.reset_cache()
    try:
        # 2.22% is material by default and immaterial at a 5% gate.
        insights, omissions = fgen.funded_movement(CTX, _movement())
        assert insights == []
        assert omissions[0].category == OMITTED_IMMATERIAL
    finally:
        cfg.reset_cache()


def test_the_headline_carries_rate_and_age_without_gating_on_them():
    """Carried for the agent to drill on; not what the card leads with."""
    movement = _movement()
    movement["delta"]["wa_interest_rate"] = 0.0012
    movement["delta"]["avg_borrower_age"] = -0.4

    insight = fgen.funded_movement(CTX, movement)[0][0]
    assert insight.metrics["wa_interest_rate_change"] == 0.0012
    assert insight.metrics["avg_borrower_age_change"] == -0.4


# --------------------------------------------------------------------------- #
# Composition
# --------------------------------------------------------------------------- #
def _decomposition(**over) -> dict:
    base = {
        "available": True,
        "opening_balance": 112_000_000.0,
        "closing_balance": 184_000_000.0,
        "movement": 72_000_000.0,
        "components": {
            "portfolio_additions": 68_000_000.0, "portfolio_disposals": 0.0,
            "organic_new_lending": 3_000_000.0, "exits": 0.0,
            "existing_book_movement": 1_000_000.0,
        },
        "portfolio_additions": [{
            "source_portfolio_id": "acquired_portfolio_beta",
            "label": "Portfolio B", "portfolio_type": "acquired",
            "balance": 68_000_000.0, "loan_count": 2}],
        "portfolio_disposals": [],
        "continuing_portfolio_ids": ["portfolio_alpha"],
        "reconciliation": {"reconciles": True, "residual": 0.0, "basis": "…"},
        "unavailable": {},
        "currentReportingDate": "2026-07-31",
        "priorReportingDate": "2026-06-30",
    }
    base.update(over)
    return base


def test_an_acquisition_month_leads_with_the_acquisition():
    insights, omissions = fgen.funded_composition(CTX, _decomposition())

    assert not omissions
    insight = insights[0]
    assert insight.severity == SEVERITY_ATTENTION
    assert "Portfolio B" in insight.headline
    assert "the acquisition of Portfolio B" in insight.summary
    assert insight.metrics["portfolio_additions"] == 68_000_000.0
    assert insight.discriminator == "acquired_portfolio_beta"


def test_an_unclassified_addition_is_not_described_as_an_acquisition():
    """Identity did not say it was bought, so the review does not say so."""
    decomposition = _decomposition()
    decomposition["portfolio_additions"][0].update(
        portfolio_type="unclassified", label="Book 17")

    insight = fgen.funded_composition(CTX, decomposition)[0][0]
    assert "acquisition" not in insight.summary.lower()
    assert "the addition of the source portfolio Book 17" in insight.summary


def test_a_single_component_month_defers_to_the_headline():
    """One material component is the headline plus a word; a card would be noise."""
    decomposition = _decomposition(
        movement=50.0, opening_balance=300.0, closing_balance=350.0,
        components={"portfolio_additions": 0.0, "portfolio_disposals": 0.0,
                    "organic_new_lending": 50.0, "exits": 0.0,
                    "existing_book_movement": 0.0},
        portfolio_additions=[])

    insights, omissions = fgen.funded_composition(CTX, decomposition)
    assert insights == []
    assert omissions[0].category == OMITTED_IMMATERIAL


# --------------------------------------------------------------------------- #
# Underlying book
# --------------------------------------------------------------------------- #
def test_the_underlying_book_is_reported_only_when_something_was_added():
    underlying = {
        "available": True, "opening_balance": 112_000_000.0,
        "closing_balance": 116_000_000.0, "movement": 4_000_000.0,
        "components": {"exits": -500_000.0, "organic_new_lending": 3_000_000.0},
        "currentReportingDate": "2026-07-31", "priorReportingDate": "2026-06-30",
    }
    insights, _ = fgen.underlying_book(CTX, _decomposition(), underlying)

    insight = insights[0]
    assert "+3.6%" in insight.summary
    assert "net of £500k of redemptions and exits" in insight.summary
    assert insight.metrics["movement"] == 4_000_000.0


def test_no_addition_means_no_underlying_card():
    decomposition = _decomposition(portfolio_additions=[])
    insights, omissions = fgen.underlying_book(CTX, decomposition, None)
    assert insights == []
    assert omissions[0].category == OMITTED_IMMATERIAL
    assert "the underlying book is the whole book" in omissions[0].reason


# --------------------------------------------------------------------------- #
# Mix
# --------------------------------------------------------------------------- #
def _shift(dimension, category, cur, pri, label=None) -> dict:
    return {"dimension": dimension, "dimension_label": label or dimension,
            "category": category, "current_share_pct": cur,
            "prior_share_pct": pri, "share_change_pp": round(cur - pri, 2),
            "current_balance": 0.0, "prior_balance": 0.0, "source_dates": {}}


def test_only_material_mix_moves_are_reported():
    shifts = [
        _shift("product", "Lump Sum", 61.0, 55.0, "Product"),      # +6.0pp
        _shift("borrower_type", "joint", 40.5, 40.0, "Borrower structure"),  # +0.5
    ]
    insights, omissions = fgen.mix_shift(CTX, shifts)

    assert [i.metrics["dimension"] for i in insights] == ["product"]
    assert not omissions          # something was kept, so no blanket omission
    assert "Lump Sum moved from 55.0% to 61.0%" in insights[0].summary


def test_a_month_with_no_material_mix_move_records_one_omission():
    shifts = [_shift("product", "Lump Sum", 55.2, 55.0, "Product"),
              _shift("vintage_year", "2024", 12.1, 12.0, "Origination vintage")]
    insights, omissions = fgen.mix_shift(CTX, shifts)

    assert insights == []
    assert len(omissions) == 1
    assert "2 checked" in omissions[0].reason


def test_borrower_and_vintage_dimensions_are_in_the_governed_mix_set():
    """The dimensions a monthly review is expected to cover, by name."""
    dimensions = {d for d, _label in engine.FUNDED_MIX_DIMENSIONS}
    assert {"product", "geographic_region_obligor", "ltv_bucket", "age_bucket",
            "borrower_type", "vintage_year", "source_portfolio_id"} == dimensions


def test_mix_shares_are_grouped_by_the_same_function_the_bridge_uses():
    """One grouping, so a mix share and a bridge contribution cannot diverge."""
    current = pd.DataFrame([
        {"current_outstanding_balance": 61.0, "erm_product_type": "Lump Sum"},
        {"current_outstanding_balance": 39.0, "erm_product_type": "Drawdown"}])
    prior = pd.DataFrame([
        {"current_outstanding_balance": 55.0, "erm_product_type": "Lump Sum"},
        {"current_outstanding_balance": 45.0, "erm_product_type": "Drawdown"}])

    shifts = engine._mix_shifts(current, prior, source_dates={})
    product = next(s for s in shifts if s["dimension"] == "product")
    assert product["column"] == "erm_product_type"
    assert abs(product["share_change_pp"]) == pytest.approx(6.0)


def test_a_tied_mix_move_resolves_the_same_way_on_every_run():
    """On a two-band dimension the moves are exact complements.

    Both are equally true, so which one is reported is arbitrary — but it must
    be the SAME arbitrary choice every time, because the selector downstream is
    deterministic and a non-deterministic input would undo that. Ties resolve by
    name ascending, the rule ``rank_contributors`` already uses.
    """
    current = pd.DataFrame([
        {"current_outstanding_balance": 61.0, "erm_product_type": "Lump Sum"},
        {"current_outstanding_balance": 39.0, "erm_product_type": "Drawdown"}])
    prior = pd.DataFrame([
        {"current_outstanding_balance": 55.0, "erm_product_type": "Lump Sum"},
        {"current_outstanding_balance": 45.0, "erm_product_type": "Drawdown"}])

    chosen = {engine._mix_shifts(current, prior, source_dates={})[0]["category"]
              for _ in range(5)}
    assert chosen == {"Drawdown"}


# --------------------------------------------------------------------------- #
# LTV
# --------------------------------------------------------------------------- #
def test_ltv_gate_is_tighter_than_the_weekly_pipeline_gate():
    # 0.2pp: below the funded 0.5pp gate.
    _, omissions = fgen.ltv_movement(CTX, _movement(ltv=30.2, prior_ltv=30.0))
    assert omissions[0].category == OMITTED_IMMATERIAL

    insights, _ = fgen.ltv_movement(CTX, _movement(ltv=30.9, prior_ltv=30.0))
    assert insights[0].metrics["change_pp"] == pytest.approx(0.9)


# --------------------------------------------------------------------------- #
# Risk-limit transitions — governed status, no second framework
# --------------------------------------------------------------------------- #
def _test_row(**over) -> dict:
    row = {"testId": "region_ldn", "displayName": "London exposure",
           "status": "warning", "priorStatus": "pass",
           "statusTransition": "pass -> warning", "deteriorated": True,
           "currentValue": 0.284, "priorValue": 0.262, "threshold": 0.30,
           "utilization": 0.947, "headroom": 1.6, "unit": "percent",
           "reportingDate": "2026-07-31", "priorReportingDate": "2026-06-30"}
    row.update(over)
    return row


def _concentration(tests) -> dict:
    return {"available": True, "tests": tests, "emergingRisks": [],
            "lineage": {"configurationVersion": "v3"}}


def test_a_deterioration_is_reported_at_the_severity_of_the_new_status():
    insights, _ = fgen.risk_limit_transitions(
        CTX, _concentration([_test_row()]))
    assert insights[0].severity == SEVERITY_ATTENTION
    assert insights[0].headline == "London exposure: pass → warning"

    breach, _ = fgen.risk_limit_transitions(
        CTX, _concentration([_test_row(status="breach",
                                       priorStatus="warning",
                                       statusTransition="warning -> breach")]))
    assert breach[0].severity == SEVERITY_CONCERN


def test_a_resolved_breach_is_reported_too():
    """A review that only delivers bad news teaches its reader to distrust silence."""
    insights, _ = fgen.risk_limit_transitions(CTX, _concentration([
        _test_row(status="pass", priorStatus="breach",
                  statusTransition="breach -> pass", deteriorated=False)]))
    assert insights[0].headline == "London exposure: breach → pass"
    assert "improved" in insights[0].summary


def test_improvements_can_be_switched_off_by_configuration(tmp_path, monkeypatch):
    path = tmp_path / "insights.yaml"
    path.write_text("insights:\n  risk_limit_transition:\n"
                    "    report_improvements: false\n", encoding="utf-8")
    monkeypatch.setenv("TRAKT_MI_INSIGHTS_CONFIG", str(path))
    cfg.reset_cache()
    try:
        insights, omissions = fgen.risk_limit_transitions(CTX, _concentration([
            _test_row(status="pass", priorStatus="breach",
                      statusTransition="breach -> pass", deteriorated=False)]))
        assert insights == []
        assert omissions[0].category == OMITTED_IMMATERIAL
    finally:
        cfg.reset_cache()


def test_a_test_that_did_not_move_produces_no_transition():
    insights, omissions = fgen.risk_limit_transitions(CTX, _concentration([
        _test_row(statusTransition=None, priorStatus="warning")]))
    assert insights == []
    assert "No governed concentration test changed status" in omissions[0].reason


def test_the_transition_generator_defines_no_limit_of_its_own():
    """It reads the approved configuration's verdict; it never re-grades one."""
    insight = fgen.risk_limit_transitions(CTX, _concentration([_test_row()]))[0][0]
    assert insight.metrics["threshold"] == 0.30
    assert insight.methodology["lineage"] == {"configurationVersion": "v3"}
    assert "no limit, status or threshold is defined here" in \
        insight.methodology["owner"]


def test_unavailable_concentration_does_not_read_as_no_transitions():
    insights, omissions = fgen.risk_limit_transitions(
        CTX, {"available": False, "reason": "no approved configuration"})
    assert insights == []
    assert omissions[0].category != OMITTED_IMMATERIAL
    assert "no approved configuration" in omissions[0].reason


# --------------------------------------------------------------------------- #
# Ordering and selection
# --------------------------------------------------------------------------- #
def test_a_limit_crossing_outranks_everything_else_in_the_month():
    """Priority order, stated as the contract rather than discovered by running."""
    assert TYPE_PRIORITY[RISK_LIMIT_TRANSITION] > TYPE_PRIORITY[FUNDED_COMPOSITION]
    assert TYPE_PRIORITY[FUNDED_COMPOSITION] > TYPE_PRIORITY[FUNDED_MOVEMENT]
    assert TYPE_PRIORITY[FUNDED_MOVEMENT] > TYPE_PRIORITY[UNDERLYING_BOOK_MOVEMENT]
    assert TYPE_PRIORITY[FUNDED_LTV_MOVEMENT] > TYPE_PRIORITY[FUNDED_MIX_SHIFT]


def test_the_monthly_brief_allows_more_than_the_weekly_one():
    """A month can cross several limits; reporting three of four is the omission
    a reader most needs not to have."""
    weekly, monthly = cfg.brief_limits(), cfg.funded_brief_limits()
    assert monthly["max_insights"] > weekly["max_insights"]
    assert monthly["max_per_type"][RISK_LIMIT_TRANSITION] == 4


def test_selection_shares_one_ordering_rule_with_the_weekly_brief():
    """Two limit sets, one order. The order is what must never diverge."""
    concentration = _concentration([
        _test_row(testId=f"t{i}", displayName=f"Test {i}") for i in range(6)])
    insights, _ = fgen.risk_limit_transitions(CTX, concentration)
    kept, capped = engine.select_funded(insights)

    assert len(kept) == 4                    # the configured per-type cap
    assert capped and capped[0].category == "capped"
    assert kept == sorted(kept, key=engine.rank_key)


# --------------------------------------------------------------------------- #
# The card takes its observations from the gated set
# --------------------------------------------------------------------------- #
def test_the_card_leads_with_what_the_materiality_layer_ranked_first():
    """Card observations are the governed insight summaries, in ranked order."""
    from trakt_notifications import portfolio_update
    from trakt_notifications.contract import UPDATE_FUNDED

    from .conftest import funded_inputs

    inputs = funded_inputs()
    inputs.concentration = _concentration([_test_row()])
    from .conftest import funded_brief_for
    inputs.funded_brief = funded_brief_for(inputs)

    message = portfolio_update.build(inputs, update_type=UPDATE_FUNDED)
    texts = [i.text for i in message.items]

    # The limit crossing outranks everything else the month produced.
    assert any("London exposure deteriorated from pass to warning" in t
               for t in texts)
    # And it is carried with its governed insight id, not re-worded.
    assert message.insight_ids


def test_a_month_below_every_threshold_says_so_rather_than_going_quiet():
    """The card must distinguish "nothing material" from "nothing ran"."""
    from trakt_notifications import portfolio_update
    from trakt_notifications.contract import UPDATE_FUNDED

    from .conftest import funded_brief_for, funded_inputs

    inputs = funded_inputs()
    # 0.28% movement: below the 1.0% gate. LTV 0.1pp: below the 0.5pp gate.
    inputs.funded_movement = _movement(
        balance=1_400_000_000.0, opening=1_396_000_000.0,
        ltv=31.1, prior_ltv=31.0)
    inputs.funded_movement["cohortMovements"] = [
        {"id": "portfolio_alpha", "label": "Alpha", "delta": 4_000_000.0}]
    inputs.funded_brief = funded_brief_for(inputs)

    message = portfolio_update.build(inputs, update_type=UPDATE_FUNDED)
    texts = [i.text for i in message.items]

    assert "No material developments were identified in the funded book " \
           "this period." in texts
    # The ungated attribution still runs, so the card is not empty.
    assert any("Largest book contribution" in t for t in texts)


def test_an_unavailable_monthly_brief_is_stated_not_silently_omitted():
    from trakt_notifications import portfolio_update
    from trakt_notifications.contract import UPDATE_FUNDED
    from trakt_notifications.sources import CAP_MONTHLY_BRIEF

    from .conftest import funded_inputs

    inputs = funded_inputs()
    inputs.funded_brief = None
    inputs.unavailable[CAP_MONTHLY_BRIEF] = "no governed funded run was available"

    message = portfolio_update.build(inputs, update_type=UPDATE_FUNDED)
    disclosure = next(i for i in message.items if i.metric == "monthly_brief")
    assert disclosure.unavailable is True
    assert "no governed funded run was available" in disclosure.text
