"""Two production-hardening guarantees, end to end through ``POST /mi/query``.

1. **A request is parsed exactly once.** Routing and execution used to parse
   independently — double the parse cost, and the routing decision could be taken
   on a different spec from the one executed.
2. **An unresolved measure is never substituted.** "the unicorn ratio by region"
   used to answer as *balance by region* with ``ok:true`` — the most damaging
   failure mode an MI agent has, because the answer is confident, plausible and
   about a different question.

Both fail on the pre-refactor code.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mi_agent_api.app import app

client = TestClient(app)


def ask(question: str, **body):
    response = client.post("/mi/query", json={"question": question, **body})
    assert response.status_code == 200, response.text
    return response.json()


# --------------------------------------------------------------------------- #
# 1. Parsed once
# --------------------------------------------------------------------------- #
@pytest.fixture()
def parse_counter(monkeypatch):
    """Count calls to the ONE parser entry point the chat path uses."""
    from mi_agent import llm_query_parser

    calls = []
    original = llm_query_parser.parse_with_repair

    def _counting(question, semantics, *a, **kw):
        calls.append(question)
        return original(question, semantics, *a, **kw)

    monkeypatch.setattr(llm_query_parser, "parse_with_repair", _counting)
    return calls


@pytest.mark.parametrize("question", [
    "what is the total current balance?",          # point-in-time KPI
    "show balance by region",                      # point-in-time grouped
    "where is the largest geographic concentration?",  # routed (geo)
    "are we within our risk limits?",              # routed (risk)
    "show balance over time",                      # evolution recogniser
    "portfolio summary",                           # composite recogniser
])
def test_a_request_is_parsed_exactly_once(parse_counter, question):
    ask(question)
    assert len(parse_counter) == 1, (
        f"{question!r} was parsed {len(parse_counter)} times: {parse_counter}")


def test_the_routed_spec_is_the_executed_spec(parse_counter):
    """One parse means one spec object — routing cannot decide on one spec while
    the executor runs another."""
    ask("show balance by region")
    assert len(parse_counter) == 1


def test_drill_through_filters_are_merged_once(parse_counter):
    body = ask("show balance by region",
               filters={"current_loan_to_value": {"op": "gt", "value": 0}})
    assert len(parse_counter) == 1
    applied = [w for w in body["warnings"] if "drill-through filters applied" in w]
    assert len(applied) == 1, f"filter merge disclosed {len(applied)} times"


def test_the_workflow_accepts_a_pre_parsed_question():
    """The seam itself: a caller may supply the parse, and the workflow must not
    re-derive it."""
    from mi_agent.mi_agent_workflow import run_mi_agent_query
    from mi_agent.parsed_question import ParsedQuestion
    from mi_agent_api.data_source import get_dataframe, semantics_path
    from mi_agent.mi_query_validator import load_mi_semantics

    df = get_dataframe()
    semantics = load_mi_semantics(semantics_path())
    parsed = ParsedQuestion.parse("show balance by region", semantics,
                                  available_columns=set(df.columns))
    result = run_mi_agent_query("show balance by region", df, str(semantics_path()),
                                parsed=parsed)
    assert result["ok"] is True
    assert result["spec_obj"] is parsed.spec


def test_parsing_without_a_supplied_parse_still_works():
    """Streamlit, the harness and calibration call the workflow directly."""
    from mi_agent.mi_agent_workflow import run_mi_agent_query
    from mi_agent_api.data_source import get_dataframe, semantics_path

    result = run_mi_agent_query("show balance by region", get_dataframe(),
                                str(semantics_path()))
    assert result["ok"] is True


# --------------------------------------------------------------------------- #
# 2. No silent metric substitution
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("question,term", [
    ("show me the unicorn ratio by region", "unicorn"),
    ("show the widget index by broker", "widget"),
    ("show the frobnicator by product type", "frobnicator"),
])
def test_an_unknown_measure_is_refused_not_substituted(question, term):
    body = ask(question)
    assert body["ok"] is False, "an unknown measure must never answer ok:true"
    # The refusal names the term, so the user can see WHAT was not understood.
    assert term in (body["error"] or "").lower()
    # And it must not have silently answered about balance instead.
    spec = body.get("spec") or {}
    assert spec.get("metric") in (None, ""), (
        f"a substitute metric {spec.get('metric')!r} was used")
    assert not [a for a in body.get("artifacts") or [] if a.get("type") == "chart"]


def test_the_refusal_preserves_the_governed_response_contract():
    body = ask("show me the unicorn ratio by region")
    for key in ("ok", "error", "question", "answer", "spec", "validation",
                "artifacts", "warnings", "metadata", "governance"):
        assert key in body, f"governed contract lost key {key!r}"
    assert body["validation"]["ok"] is False
    assert any("unresolved_metric" in e for e in body["validation"]["errors"])
    # Traceable: classified as a capability outcome, not a crash.
    assert body["governance"]["status"] in ("error", "blocked")
    assert body["governance"]["error"]["code"] in (
        "UNSUPPORTED_QUESTION", "AMBIGUOUS_QUESTION")


def test_a_named_portfolio_does_not_rescue_an_unknown_measure():
    """Mentioning a portfolio scope keeps a question on the summary path. That
    must not let an unresolvable measure through as a balance answer."""
    body = ask("the unicorn ratio for the acquired book by region")
    assert body["ok"] is False
    assert "unicorn" in (body["error"] or "").lower()


@pytest.mark.parametrize("question", [
    # Analytical FRAMING with no measure named: defaulting to balance is
    # long-standing governed behaviour, not a substitution — nothing was named
    # to substitute for.
    "concentration by region",
    "regions with the most loans",
    "breakdown by region",
    "give me a breakdown of balance by region",
    "show the split by region",
    # Genuinely named measures must keep working.
    "show balance by region",
    "show weighted average LTV by region",
    "show balance by age bucket",
    "what is the average property value?",
    "show top 5 regions by balance",
    "show balance by region and age bucket",
    "show loans with LTV above 50%",
    "how many loans are there?",
])
def test_supported_questions_still_answer(question):
    body = ask(question)
    assert body["ok"] is True, f"{question!r} regressed: {body.get('error')}"


def test_an_unavailable_dimension_is_refused_not_substituted():
    """P0 contract change, deliberate and asserted here rather than dropped.

    ``coverage by borrower type`` previously answered ``ok:true`` on this
    dataset: ``borrower_type`` is absent, so the parser reached for
    ``amortisation_type`` and the user received a confident single-bucket
    breakdown of a dimension they never named. That is the DIMENSION form of the
    measure substitution this module already forbids, and the P0 launch
    invariant rules it out explicitly — an unavailable requested field is never
    silently replaced by an available one.

    The question is still parsed and still routed; what changed is that the
    substitution now fails closed instead of being presented as the answer.
    """
    body = ask("coverage by borrower type")
    assert body["ok"] is False
    error = (body.get("error") or "").lower()
    assert "borrower type" in error
    # The substitute is NAMED, so the refusal is actionable rather than opaque.
    assert "amortisation" in error
    assert "not returned" in error or "not substituted" in error


def test_an_unknown_measure_with_no_dimension_is_still_refused():
    """The pre-existing unmapped path already covered this; it must stay."""
    body = ask("what is the unicorn ratio?")
    assert body["ok"] is False


def test_a_governed_measure_absent_from_the_dataset_fails_cleanly():
    """The complement of the residue rule, and the reason it is scoped to
    UNRESOLVED terms only.

    "spread" IS governed vocabulary — a registry synonym for the interest-rate
    margin. The parser resolves it correctly and validation then fails because
    the column is absent from this pack. That is the pre-existing fail-closed
    behaviour and must not be re-routed through the residue path: the measure
    was understood, it simply is not reported here.
    """
    body = ask("show spread by product type")
    assert body["ok"] is False
    spec = body.get("spec") or {}
    # Resolved to its OWN field — not swapped for balance.
    assert spec.get("metric") == "current_interest_rate_margin"
    assert body["validation"]["ok"] is False
