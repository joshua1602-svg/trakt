#!/usr/bin/env python3
"""The Portfolio Review Agent's boundaries: mandate, arithmetic, brevity.

Three controls are tested here, and all three exist because a sentence in a
system prompt failed to enforce them against a real model:

* **scope** — readiness and regulatory tools are not offered, and a call naming
  one is refused before execution;
* **arithmetic** — a figure no governed result contains does not reach a reader;
* **brevity** — the card is a ranked selection inside a word budget.

The most important test in this file is
``test_the_recorded_one_point_eight_eight_million_leak_is_caught``. It replays
the exact sentence a real model published against the exact governed payloads
that session produced, and asserts the gate refuses it. Every other test here
describes a rule; that one reproduces a defect.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import trakt_tools.handlers  # noqa: F401 - populates the registry
from portfolio_review import brief, mandate, numeric_gate
from portfolio_review.brief import _finding_words
from portfolio_review.numeric_gate import GovernedIndex
from portfolio_review.session import MIScopedSession


# --------------------------------------------------------------------------- #
# The mandate
# --------------------------------------------------------------------------- #
def test_every_registered_tool_is_classified():
    """The allow-list is total, so a new tool cannot arrive unnoticed.

    This is the test that keeps the mandate honest over time. Registering a
    tool in Trakt without deciding whether the Portfolio Review Agent may call
    it breaks the build here rather than silently widening the agent.
    """
    assert mandate.audit_registry() == {
        "unclassified": [], "missing": [], "excluded_but_absent": []}


def test_the_readiness_surface_is_excluded_in_full():
    """Every tool in the securitisation readiness module is out of mandate."""
    from trakt_tools.handlers import readiness as readiness_module

    for tool in ("readiness_framework", "readiness_metrics",
                 "regulatory_readiness", "evaluate_rule_packs",
                 "valuation_age_profile"):
        assert tool not in mandate.ALLOWED
        assert tool in mandate.EXCLUDED_NAMES
    assert readiness_module.__doc__.lstrip().startswith(
        "trakt_tools.handlers.readiness — the securitisation readiness surface")


def test_approved_client_limits_stay_in_scope():
    """`evaluate_covenants` reads the client's own approved configuration.

    It is the one covenant tool in scope, and it is in scope precisely because
    it reports the Risk Limits workspace's approved tests rather than an
    illustrative rulebook. Excluding it would leave the agent unable to answer
    the risk half of its own mandate.
    """
    assert "evaluate_covenants" in mandate.ALLOWED
    assert "evaluate_rule_packs" not in mandate.ALLOWED


def test_the_offered_surface_contains_no_excluded_tool():
    names = {schema["name"] for schema in mandate.tool_schemas()}
    assert names == set(mandate.ALLOWED)
    assert not (names & mandate.EXCLUDED_NAMES)


def test_the_prompt_and_the_allow_list_agree_about_readiness():
    """The prompt says regulatory analysis is out; the surface makes it true."""
    from portfolio_review.objective import SYSTEM_PROMPT

    assert "ESMA Annex 2" in SYSTEM_PROMPT
    assert "regulatory_readiness" not in {s["name"]
                                          for s in mandate.tool_schemas()}


# --------------------------------------------------------------------------- #
# Physical enforcement
# --------------------------------------------------------------------------- #
class _Session:
    """A session that would happily run anything asked of it."""

    resource = "client2/portfolio/total"

    def __init__(self):
        self.executed: List[str] = []

    def call(self, tool: str, arguments=None) -> Dict[str, Any]:
        self.executed.append(tool)
        return {"available": True, "value": 42.0}

    def transcript(self):
        return []


def test_an_out_of_mandate_call_never_reaches_execution():
    """Not merely unused — unavailable. The wrapped session is never asked."""
    inner = _Session()
    scoped = MIScopedSession(inner)

    payload = scoped.call("regulatory_readiness", {})

    assert payload["refused"] is True
    assert payload["error_code"] == "OUT_OF_MANDATE"
    assert inner.executed == []


def test_the_refusal_names_the_owning_agent_and_forbids_reporting_it():
    """An out-of-mandate refusal must not become a `could_not_assess` entry.

    A gap in a review means Trakt could not answer something the agent needed.
    A question belonging to another agent is not that, and a review that lists
    it as a gap reintroduces the scope leak in the shape of an apology.
    """
    message = MIScopedSession(_Session()).call("evaluate_rule_packs", {})["message"]

    assert "Securitisation Readiness Agent" in message
    assert "do not list it as a gap" in message.lower()


def test_an_unknown_tool_is_refused_rather_than_forwarded():
    inner = _Session()
    assert MIScopedSession(inner).call("nonexistent_tool", {})["refused"] is True
    assert inner.executed == []


def test_allowed_calls_pass_through_and_are_indexed():
    inner = _Session()
    scoped = MIScopedSession(inner)

    scoped.call("portfolio_summary", {})

    assert inner.executed == ["portfolio_summary"]
    assert 42.0 in set(scoped.index.values())
    assert scoped.out_of_mandate_calls() == []


# --------------------------------------------------------------------------- #
# The arithmetic gate
# --------------------------------------------------------------------------- #
def _index(payloads: Dict[str, Any]) -> GovernedIndex:
    index = GovernedIndex()
    for tool, payload in payloads.items():
        index.absorb(tool, payload)
    return index


def _review(**over: Any) -> Dict[str, Any]:
    base = {
        "period_verdict": "ROUTINE_PERIOD",
        "headline": "The book grew modestly.",
        "summary": "Nothing material changed.",
        "findings": [],
    }
    base.update(over)
    return base


def test_the_recorded_one_point_eight_eight_million_leak_is_caught():
    """The exact defect the real-model red-team recorded, replayed.

    A real model, given real canonical, published:

        "Highest LTV loans are ORIGINATION-0043 at 70.99% (£954k) and
         SPV1-0022 at 70.79% (£926k). Combined they are £1.88m."

    Both loan balances were governed. Their sum was not — the model added them
    and said so. The governed payload below carries the two balances at the
    precision `rank_loans` returned them, and nothing else, exactly as that
    session's did.
    """
    index = _index({"rank_loans": {"loans": [
        {"loan_identifier": "ORIGINATION-0043",
         "current_outstanding_balance": 954513.89, "current_loan_to_value": 70.99},
        {"loan_identifier": "SPV1-0022",
         "current_outstanding_balance": 926460.77, "current_loan_to_value": 70.79},
    ]}})
    review = _review(findings=[{
        "title": "Two loans above 70% LTV",
        "observation": ("Highest LTV loans are ORIGINATION-0043 at 70.99% "
                        "(£954k) and SPV1-0022 at 70.79% (£926k). Combined "
                        "they are £1.88m."),
        "why_it_matters": "Size and leverage coincide.",
        "severity": "low",
    }])

    result = numeric_gate.apply(review, index)

    assert result.status == numeric_gate.DEGRADED
    assert result.review["findings"] == []
    assert [c.stated for c in result.unsupported] == ["1.88"]
    assert result.dropped_findings


def test_the_component_balances_themselves_are_accepted():
    """The gate must not reject a figure merely because it was rounded.

    £954k for 954,513.89 is presentation. £1.88m for a sum is derivation. If the
    gate could not tell them apart it would be unusable, so this pins the half
    that must pass.
    """
    index = _index({"rank_loans": {"balance": 954513.89}})
    result = numeric_gate.apply(
        _review(findings=[{"title": "Largest exposure",
                           "observation": "The largest loan is £954k.",
                           "why_it_matters": "Concentration.",
                           "severity": "low"}]), index)

    assert result.status == numeric_gate.PUBLISHABLE
    assert len(result.review["findings"]) == 1


def test_a_wrong_rounding_is_not_accepted():
    index = _index({"rank_loans": {"balance": 954513.89}})
    result = numeric_gate.apply(
        _review(findings=[{"title": "Largest exposure",
                           "observation": "The largest loan is £974k.",
                           "why_it_matters": "Concentration.",
                           "severity": "low"}]), index)

    assert result.status == numeric_gate.DEGRADED
    assert result.review["findings"] == []


@pytest.mark.parametrize("stated,governed,expect_ok", [
    ("£11.97m", 11974544.28, True),      # presentation
    ("£11.98m", 11974544.28, False),     # wrong at the stated precision
    ("£12m", 11974544.28, True),         # coarser, still correct
    ("93%", 0.9337, True),               # governed share, stated as percent
    ("93.37%", 0.9337, True),
    ("32%", 0.9337, False),              # a different quantity entirely
])
def test_scaling_is_allowed_and_arithmetic_is_not(stated, governed, expect_ok):
    index = _index({"funded_composition": {"value": governed}})
    result = numeric_gate.apply(
        _review(findings=[{"title": "t", "observation": f"It is {stated}.",
                           "why_it_matters": "w", "severity": "low"}]), index)

    assert bool(result.review["findings"]) is expect_ok


def test_an_unsupported_figure_in_the_headline_blocks_the_whole_review():
    """There is no honest way to publish a headline nobody can source."""
    index = _index({"period_change": {"movement": 554485.91}})
    result = numeric_gate.apply(
        _review(headline="The book grew £7.51m this period."), index)

    assert result.status == numeric_gate.BLOCKED
    assert result.review is None
    assert not result.publishable
    assert result.reasons


def test_one_bad_finding_does_not_withhold_the_good_ones():
    index = _index({"period_change": {"movement": 554485.91}})
    result = numeric_gate.apply(_review(findings=[
        {"title": "Movement", "observation": "The book moved £554,485.91.",
         "why_it_matters": "Growth.", "severity": "low"},
        {"title": "Derived", "observation": "That is 4.2% annualised.",
         "why_it_matters": "Trend.", "severity": "low"},
    ]), index)

    assert result.status == numeric_gate.DEGRADED
    assert [f["title"] for f in result.review["findings"]] == ["Movement"]
    assert len(result.dropped_findings) == 1


def test_dropped_findings_are_recorded_never_silently_removed():
    index = _index({"period_change": {"movement": 1.0}})
    result = numeric_gate.apply(_review(findings=[
        {"title": "Invented", "observation": "Exposure is £4.44m.",
         "why_it_matters": "w", "severity": "high"}]), index)

    assert result.dropped_findings[0]["finding"]["title"] == "Invented"
    assert "not a governed value" in result.dropped_findings[0]["reason"]


def test_dates_codes_and_bucket_terms_are_not_treated_as_measurements():
    """A field code is a name and "90 days" is a bucket. Neither is a figure.

    Without this the ledger fills with noise and a real unsupported number is
    harder to see, which is a way of failing quietly.
    """
    index = _index({"portfolio_summary": {"loans": 118}})
    result = numeric_gate.apply(_review(findings=[{
        "title": "Clean",
        "observation": ("As at 2026-06-30, RREC17/18/19 aside, 0 loans are "
                        "over 90 days past due across all 118 loans."),
        "why_it_matters": "Performance is clean.", "severity": "low"}]), index)

    assert result.status == numeric_gate.PUBLISHABLE


def test_a_governed_number_inside_a_string_still_counts():
    """Trakt renders figures into warnings and labels; those are governed too."""
    index = _index({"evaluate_covenants": {
        "warning": "Utilisation is 12.7% against a 15.0% limit."}})
    result = numeric_gate.apply(_review(findings=[{
        "title": "Utilisation", "observation": "Utilisation is 12.7%.",
        "why_it_matters": "Approaching the approved limit.",
        "severity": "medium"}]), index)

    assert result.status == numeric_gate.PUBLISHABLE


def test_the_ledger_names_the_governed_field_behind_each_figure():
    """§16's audit table, produced by the gate rather than reconstructed."""
    index = _index({"funded_composition": {"movement": 12825098.04}})
    result = numeric_gate.apply(_review(findings=[{
        "title": "Movement", "observation": "The book moved £12.83m.",
        "why_it_matters": "Material.", "severity": "high"}]), index)

    row = next(r for r in result.ledger() if r["output_number"] == "12.83m")
    assert row["governed_source_tool"] == "funded_composition"
    assert row["source_field"] == "movement"
    assert row["exact_match"] == "yes"


def test_a_review_with_no_figures_at_all_is_publishable():
    """A quiet period is a real answer and states nothing to verify."""
    result = numeric_gate.apply(
        _review(headline="Nothing material changed this period."),
        GovernedIndex())

    assert result.status == numeric_gate.PUBLISHABLE


# --------------------------------------------------------------------------- #
# Brevity
# --------------------------------------------------------------------------- #
def _finding(i: int, words: int = 40) -> Dict[str, Any]:
    return {"title": f"Finding {i}", "observation": "word " * words,
            "why_it_matters": "word " * words, "severity": "low"}


def test_the_card_keeps_at_most_five_findings():
    card = brief.render(_review(findings=[_finding(i, 3) for i in range(9)]))

    assert len(card.findings) == brief.MAX_FINDINGS
    assert len(card.withheld) == 4
    assert any("top 5 of 9" in n for n in card.notes)


def test_the_card_drops_from_the_bottom_of_the_ranking_until_it_fits():
    """Selection, not truncation: whole findings go, sentences never do."""
    card = brief.render(_review(findings=[_finding(i, 90) for i in range(5)]))

    assert card.word_count <= brief.SOFT_CARD_WORDS
    assert card.findings[0]["title"] == "Finding 0"
    assert all(f["observation"].strip() for f in card.findings)


def test_a_thousand_word_review_becomes_a_card():
    """The measured failure: the ungated runs were 744–1,011 words."""
    long_review = _review(
        headline="A headline.", summary="word " * 80,
        findings=[_finding(i, 70) for i in range(8)],
        could_not_assess=[{"check": f"Check {i}", "reason": "word " * 40,
                           "implication": "word " * 40} for i in range(5)])
    assert sum(_finding_words(f) for f in long_review["findings"]) > 1000

    card = brief.render(long_review)

    assert card.word_count <= brief.SOFT_CARD_WORDS
    assert len(card.gaps) <= brief.MAX_GAPS
    assert all(isinstance(g, str) for g in card.gaps)


def test_going_over_the_guide_is_a_note_not_a_failure():
    """Length alone must never withhold a useful briefing.

    A summary longer than selection can trim leaves the card over the guide.
    That is reported, and the card is still produced with its findings intact —
    the commercial requirement is a readable briefing, not a word count.
    """
    card = brief.render(_review(summary="word " * 400,
                                findings=[_finding(0, 20)]))

    assert card.word_count > brief.SOFT_CARD_WORDS
    assert any("a note, not a defect" in n for n in card.notes)
    assert len(card.findings) == 1
    assert not brief.quality_flags(card)


# --------------------------------------------------------------------------- #
# Quality — what actually makes a briefing bad
# --------------------------------------------------------------------------- #
def _card(**over):
    return brief.render(_review(**over))


def test_a_clean_briefing_raises_no_quality_flag():
    card = _card(findings=[
        {"title": "Movement", "observation": "The book grew £554k.",
         "why_it_matters": "Ordinary origination.", "severity": "low"},
        {"title": "Mix", "observation": "London rose to 22.6% of balance.",
         "why_it_matters": "Worth watching against the regional limit.",
         "severity": "medium"}])

    assert brief.quality_flags(card) == []


def test_two_findings_that_repeat_each_other_are_flagged():
    same = ("The funded book increased by five hundred and fifty four thousand "
            "pounds through new lending across three loans this period")
    card = _card(findings=[
        {"title": "Growth", "observation": same,
         "why_it_matters": "a", "severity": "low"},
        {"title": "New lending", "observation": same + " overall",
         "why_it_matters": "b", "severity": "low"}])

    assert any("repeat each other" in f for f in brief.quality_flags(card))


def test_duplicate_titles_are_flagged():
    card = _card(findings=[
        {"title": "Concentration", "observation": "London is 22.6%.",
         "why_it_matters": "a", "severity": "low"},
        {"title": "Concentration", "observation": "The top ten are 20.1%.",
         "why_it_matters": "b", "severity": "low"}])

    assert "duplicate finding titles" in brief.quality_flags(card)


def test_methodology_exposition_is_flagged():
    card = _card(findings=[{
        "title": "LTV", "observation": "Weighted LTV is 45.9%.",
        "why_it_matters": "It is calculated as the balance-weighted mean.",
        "severity": "low"}])

    assert any("explains method" in f for f in brief.quality_flags(card))


def test_raw_tool_output_is_flagged():
    card = _card(findings=[{
        "title": "Composition", "observation": "components: {'exits': None}",
        "why_it_matters": "a", "severity": "low"}])

    assert "carries raw tool output" in brief.quality_flags(card)


def test_claims_hedged_away_are_flagged():
    card = _card(findings=[{
        "title": "Concentration",
        "observation": "London may be rising and could potentially matter.",
        "why_it_matters": ("This might suggest a trend, though it appears to "
                           "be unclear and arguably cannot be determined."),
        "severity": "low"}])

    assert any("hedging" in f for f in brief.quality_flags(card))


def test_word_count_alone_is_never_a_quality_flag():
    """The explicit commercial instruction: length is not a defect."""
    card = _card(findings=[{
        "title": "A long but useful finding",
        "observation": " ".join(f"governed figure {i}" for i in range(90)),
        "why_it_matters": " ".join(f"reason {i}" for i in range(90)),
        "severity": "high"}])

    assert card.word_count > brief.SOFT_CARD_WORDS
    assert brief.quality_flags(card) == []


def test_the_gaps_are_named_not_argued():
    card = brief.render(_review(could_not_assess=[
        {"check": "Valuation age", "reason": "word " * 50,
         "implication": "word " * 50}]))

    assert card.gaps == ["Valuation age"]


def test_an_over_long_headline_is_reported_rather_than_cut():
    card = brief.render(_review(headline="word " * 60))

    assert any("headline is 60 words" in n for n in card.notes)
    assert len(card.headline.split()) == 60


def test_a_short_review_passes_through_untouched():
    review = _review(findings=[_finding(0, 8)])
    card = brief.render(review)

    assert card.withheld == []
    assert card.notes == []
    assert len(card.findings) == 1
