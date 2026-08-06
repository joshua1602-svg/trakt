"""Message generation: what the two cards say, and what they must never say."""

from __future__ import annotations

import pytest

from trakt_notifications import generate, portfolio_update, risk_review
from trakt_notifications.config import (
    LEVEL_APPROVED_ACTION, LEVEL_CONSIDERATION, LEVEL_OBSERVATION,
)
from trakt_notifications.contract import (
    MESSAGE_PORTFOLIO_UPDATE, MESSAGE_RISK_REVIEW, SEVERITY_CLEAR,
    SEVERITY_CONCERN, UPDATE_COMBINED, UPDATE_FUNDED, UPDATE_PIPELINE, validate,
)
from trakt_notifications import recommendation

from .conftest import (
    concentration_risk, contributors_block, funded_inputs, pipeline_inputs,
)


def _texts(message) -> str:
    return " | ".join(i.text for i in message.items)


# --------------------------------------------------------------------------- #
# Portfolio Update
# --------------------------------------------------------------------------- #
def test_pipeline_update_reports_levels_and_five_week_comparisons():
    message = portfolio_update.build(pipeline_inputs(),
                                     update_type=UPDATE_PIPELINE)
    assert message.headline == "Weekly Pipeline Update — 7 August"
    assert message.summary == "Pipeline contains 812 cases totalling £503.4m."
    body = _texts(message)
    assert "Case count is 9% above the five-week average." in body
    assert "Completions totalled 41 cases and £18.7m" in body
    assert "14% below the five-week average" in body
    assert "Weighted expected funding is £236.2m." in body


def test_pipeline_update_quotes_the_governed_contributors():
    message = portfolio_update.build(pipeline_inputs(),
                                     update_type=UPDATE_PIPELINE)
    assert "led by Broker A and the South East" in _texts(message)
    # The bullet is traceable to the insight it was taken from.
    assert "insight-pm-1" in message.insight_ids


def test_below_average_reads_as_below():
    inputs = pipeline_inputs()
    inputs.pipeline_evolution["fiveWeekAverage"]["differencePct"][
        "pipeline_case_count"] = -12.0
    message = portfolio_update.build(inputs, update_type=UPDATE_PIPELINE)
    assert "Case count is 12% below the five-week average." in _texts(message)


def test_no_prior_history_omits_the_comparison_with_a_reason():
    """The specific failure this guards: substituting a different baseline."""
    inputs = pipeline_inputs()
    inputs.pipeline_evolution = {"fiveWeekAverage": {"available": False}}
    message = portfolio_update.build(inputs, update_type=UPDATE_PIPELINE)
    body = _texts(message)
    # No pipeline comparison is invented from a different baseline, and the
    # reason names the pipeline comparison specifically — the completions
    # comparison comes from the funnel and is still legitimately available.
    assert "Case count is" not in body
    assert "Pipeline five-week comparisons are unavailable" in body
    assert "no prior weekly history has been observed" in body


def test_a_short_history_says_how_many_weeks_it_had():
    inputs = pipeline_inputs()
    inputs.pipeline_evolution["fiveWeekAverage"]["weeksObserved"] = 2
    message = portfolio_update.build(inputs, update_type=UPDATE_PIPELINE)
    assert "Pipeline five-week comparisons are based on 2 observed weeks" in _texts(message)


def test_funded_update_reports_balance_count_ltv_and_cohorts():
    message = portfolio_update.build(funded_inputs(), update_type=UPDATE_FUNDED)
    assert message.headline == "Monthly Funded Update — 31 July"
    assert "Funded balance is £1.42bn" in (message.summary or "")
    body = _texts(message)
    assert "Loan count is 4,820 (+59 on the month)." in body
    assert "Largest book contribution: Direct at +£18.4m." in body
    assert "Weighted-average LTV is 31.4%, +0.30pp on the month." in body


def test_combined_update_keeps_the_two_dates_apart():
    """A combined card must never imply the datasets share a reporting date."""
    inputs = pipeline_inputs()
    funded = funded_inputs()
    inputs.funded_movement = funded.funded_movement
    inputs.source_dates.update(funded.source_dates)

    message = portfolio_update.build(inputs, update_type=UPDATE_COMBINED)
    assert "pipeline 7 August" in message.headline
    assert "funded 31 July" in message.headline
    assert "as at 7 August" in message.summary
    assert "as at 31 July" in message.summary


def test_update_respects_the_item_cap_and_says_it_truncated():
    message = portfolio_update.build(pipeline_inputs(),
                                     update_type=UPDATE_PIPELINE, max_items=3)
    assert len(message.items) == 3
    assert "further observations are in Trakt" in message.items[-1].text


def test_conversion_is_omitted_when_the_governed_window_is_insufficient():
    inputs = pipeline_inputs()
    inputs.funnel["summary"]["COMPLETED"]["conversion"]["sufficient"] = False
    message = portfolio_update.build(inputs, update_type=UPDATE_PIPELINE)
    assert "conversion" not in _texts(message).lower()


def test_unavailable_funded_movement_is_stated_not_dropped():
    inputs = funded_inputs()
    inputs.funded_movement = {"available": False,
                              "reason": "only one funded period is available"}
    message = portfolio_update.build(inputs, update_type=UPDATE_FUNDED)
    assert "Funded movement is unavailable" in _texts(message)
    assert any(i.unavailable for i in message.items)


# --------------------------------------------------------------------------- #
# Risk Review — the clear state
# --------------------------------------------------------------------------- #
def test_no_material_risk_sends_the_strong_clear_statement():
    message = risk_review.build(pipeline_inputs())
    assert message.severity == SEVERITY_CLEAR
    assert message.headline == "Risk Review"
    body = _texts(message)
    assert "No material portfolio risks were identified from this update." in body
    assert "No current concentration breaches." in body
    assert "No forecast breaches within the governed horizon." in body


def test_an_unavailable_check_weakens_the_clear_statement():
    """The distinction the product depends on: a claim about the portfolio
    versus a claim about the checks."""
    inputs = pipeline_inputs()
    inputs.unavailable["concentration tests"] = "the service did not respond"
    inputs.concentration = None

    message = risk_review.build(inputs)
    body = _texts(message)
    assert "No material portfolio risks were identified from this update." not in body
    assert ("No material risks were identified from the checks that were "
            "available.") in body
    assert "concentration tests" in body
    assert message.unavailable_checks == ["concentration tests"]


def test_a_clear_message_never_claims_a_check_that_did_not_run():
    inputs = pipeline_inputs()
    inputs.concentration = None
    inputs.unavailable["concentration tests"] = "unavailable"
    body = _texts(risk_review.build(inputs))
    assert "No current concentration breaches." not in body


# --------------------------------------------------------------------------- #
# Risk Review — findings
# --------------------------------------------------------------------------- #
def test_concentration_breach_leads_the_message():
    inputs = pipeline_inputs()
    inputs.concentration = concentration_risk()
    message = risk_review.build(inputs)
    assert message.severity == SEVERITY_CONCERN
    assert message.headline == "Risk Review — South West exposure"
    assert "expected to breach" in message.items[0].text
    assert "crossing around 2026-09" in message.items[0].text


def test_a_stress_only_breach_is_labelled_as_a_scenario():
    inputs = pipeline_inputs()
    snapshot = concentration_risk(expected_breach=False)
    snapshot["emergingRisks"] = [{
        "category": "full_pipeline_only_breach", "rank": 5,
        "testId": "region_sw", "displayName": "South West exposure",
        "statement": ("South West exposure breaches only if ALL pipeline "
                      "converts (23.5% → 27.1% vs 25.0%) — a maximum-exposure "
                      "scenario, not an expected outcome.")}]
    inputs.concentration = snapshot
    body = _texts(risk_review.build(inputs))
    assert "a maximum-exposure scenario, not an expected outcome" in body


def test_risk_items_are_capped_and_the_remainder_is_declared():
    inputs = pipeline_inputs()
    snapshot = concentration_risk()
    snapshot["emergingRisks"] = [
        {"category": "expected_breach", "rank": 2, "testId": f"t{n}",
         "displayName": f"Test {n}", "statement": f"Statement {n}."}
        for n in range(6)]
    inputs.concentration = snapshot
    message = risk_review.build(inputs, max_items=3)
    assert sum(1 for i in message.items if i.metric == "expected_breach") == 3
    assert "3 further risk items are shown in Trakt." in _texts(message)


def test_insight_findings_only_count_when_graded_above_informational():
    inputs = pipeline_inputs()
    inputs.brief["insights"].append({
        "insight_type": "DATA_QUALITY", "insight_id": "dq-1",
        "severity": "info", "headline": "No material data-quality issues",
        "summary": "Everything within threshold."})
    assert risk_review.build(inputs).severity == SEVERITY_CLEAR

    inputs.brief["insights"][-1]["severity"] = "concern"
    inputs.brief["insights"][-1]["headline"] = "2 data-quality thresholds breached"
    message = risk_review.build(inputs)
    assert message.severity == SEVERITY_CONCERN
    assert "dq-1" in message.insight_ids


def test_data_quality_outranks_a_forecast_concentration_warning():
    """Data quality governs whether the rest of the update can be believed."""
    inputs = pipeline_inputs()
    snapshot = concentration_risk()
    snapshot["emergingRisks"][0]["rank"] = 3
    snapshot["emergingRisks"][0]["category"] = "expected_warning_low_headroom"
    inputs.concentration = snapshot
    inputs.brief["insights"].append({
        "insight_type": "DATA_QUALITY", "insight_id": "dq-1",
        "severity": "concern", "headline": "2 data-quality thresholds breached",
        "summary": "Dimensions are unstable."})
    message = risk_review.build(inputs)
    assert "data-quality" in message.headline


def test_deterministic_wording_across_repeated_builds():
    inputs = pipeline_inputs()
    inputs.concentration = concentration_risk()
    inputs.contributors = {"region_sw": contributors_block()}
    first = risk_review.build(inputs).to_dict()
    second = risk_review.build(inputs).to_dict()
    assert first == second


# --------------------------------------------------------------------------- #
# Broker / region decision support and the action-language policy
# --------------------------------------------------------------------------- #
def test_contributor_evidence_names_the_broker_region_intersection():
    inputs = pipeline_inputs()
    inputs.concentration = concentration_risk()
    inputs.contributors = {"region_sw": contributors_block()}
    body = _texts(risk_review.build(inputs))
    assert ("£8.2m of probability-weighted expected completions from Broker A "
            "in the South West") in body


def test_default_level_is_consideration_not_instruction():
    resolved = recommendation.resolve(contributors_block(),
                                      level=LEVEL_CONSIDERATION)
    assert resolved["level"] == LEVEL_CONSIDERATION
    assert ("Consider reviewing completion timing for Broker A's South West "
            "pipeline.") in resolved["recommendation"]
    assert "reduce projected utilisation" in resolved["recommendation"]


def test_observation_level_states_the_contributor_and_stops():
    resolved = recommendation.resolve(contributors_block(),
                                      level=LEVEL_OBSERVATION)
    assert resolved["level"] == LEVEL_OBSERVATION
    assert resolved["recommendation"] == (
        "Broker A in the South West is the largest projected contributor.")


def test_approved_action_without_a_client_rule_degrades_to_consideration():
    """Configuration alone must not be able to escalate the wording."""
    resolved = recommendation.resolve(contributors_block(),
                                      level=LEVEL_APPROVED_ACTION,
                                      action_rule=None)
    assert resolved["level"] == LEVEL_CONSIDERATION


def test_approved_action_uses_the_client_rule_verbatim():
    rule = {"statement": "Pause new South West submissions from Broker A."}
    resolved = recommendation.resolve(contributors_block(),
                                      level=LEVEL_APPROVED_ACTION,
                                      action_rule=rule)
    assert resolved["level"] == LEVEL_APPROVED_ACTION
    assert resolved["recommendation"] == rule["statement"]


@pytest.mark.parametrize("level", [LEVEL_OBSERVATION, LEVEL_CONSIDERATION])
def test_generated_wording_carries_no_operational_instruction(level):
    resolved = recommendation.resolve(contributors_block(), level=level)
    lowered = (resolved["recommendation"] or "").lower()
    for phrase in recommendation.FORBIDDEN_PHRASES:
        assert phrase not in lowered


def test_no_contributor_evidence_produces_no_recommendation():
    resolved = recommendation.resolve({"available": False},
                                      level=LEVEL_CONSIDERATION)
    assert resolved["recommendation"] is None
    assert resolved["evidence"] is None


# --------------------------------------------------------------------------- #
# Batch assembly
# --------------------------------------------------------------------------- #
def test_every_batch_carries_exactly_the_two_messages(enabled_config):
    batch = generate.build(pipeline_inputs(), update_type=UPDATE_PIPELINE)
    assert validate(batch) == []
    assert {m.message_type for m in batch.messages} == {
        MESSAGE_PORTFOLIO_UPDATE, MESSAGE_RISK_REVIEW}
    assert [m.sequence for m in
            sorted(batch.messages, key=lambda m: m.sequence)] == [1, 2]


def test_batch_identity_is_stable_across_regeneration(enabled_config):
    first = generate.build(pipeline_inputs(), update_type=UPDATE_PIPELINE,
                           approved_run_ids=["run-1"],
                           generated_at="2026-08-07T09:00:00+00:00")
    second = generate.build(pipeline_inputs(), update_type=UPDATE_PIPELINE,
                            approved_run_ids=["run-1"],
                            generated_at="2026-08-07T18:30:00+00:00")
    assert first.notification_batch_id == second.notification_batch_id
    assert first.reporting_key == second.reporting_key


def test_a_different_run_for_the_same_dates_shares_only_the_reporting_key():
    first = generate.build(pipeline_inputs(), update_type=UPDATE_PIPELINE,
                           approved_run_ids=["run-1"])
    second = generate.build(pipeline_inputs(), update_type=UPDATE_PIPELINE,
                            approved_run_ids=["run-2"])
    assert first.reporting_key == second.reporting_key
    assert first.notification_batch_id != second.notification_batch_id


def test_failed_risk_generation_never_produces_a_clear_state(monkeypatch,
                                                             enabled_config):
    def explode(*_args, **_kwargs):
        raise RuntimeError("concentration evaluation failed")

    monkeypatch.setattr(risk_review, "build", explode)
    batch = generate.build(pipeline_inputs(), update_type=UPDATE_PIPELINE)

    update = batch.message(MESSAGE_PORTFOLIO_UPDATE)
    risk = batch.message(MESSAGE_RISK_REVIEW)
    # The operational message still goes out.
    assert "812 cases" in (update.summary or "")
    # The risk message says so, in terms, rather than reading as clear.
    assert risk.severity != SEVERITY_CLEAR
    assert "could not be completed" in _texts(risk)
    assert "not a statement that no risks exist" in _texts(risk)


def test_update_type_is_derived_from_the_approved_datasets():
    assert generate.update_type_for(["pipeline"]) == UPDATE_PIPELINE
    assert generate.update_type_for(["funded"]) == UPDATE_FUNDED
    assert generate.update_type_for(["pipeline", "funded"]) == UPDATE_COMBINED


def test_correction_labelling_is_explicit(enabled_config):
    batch = generate.build(pipeline_inputs(), update_type=UPDATE_PIPELINE)
    batch.corrects_batch_id = "earlier-batch"
    generate.label_corrections(batch)
    assert batch.message(MESSAGE_PORTFOLIO_UPDATE).headline.startswith(
        "Corrected Weekly Pipeline Update")
    assert batch.message(MESSAGE_RISK_REVIEW).headline.startswith(
        "Corrected Risk Review")
    assert "following a corrected data upload" in _texts(
        batch.message(MESSAGE_RISK_REVIEW))
