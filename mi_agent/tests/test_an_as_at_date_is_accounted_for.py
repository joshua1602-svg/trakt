"""A point-in-time date the answer did not honour must not produce a figure.

MEASURED, on the fully governed three-period book (2026-04-30 / 05-31 / 06-30):

    What is the balance as at 31 December 1999?   -> £1.96BN, ok, 11,035 loans
    What is the balance as at 30 April 1975?      -> £1.96BN, ok
    How many loans were there at 31 December 1999? -> 11,035 loans, ok

every one of them receipted "as at 30 June 2026" — a date the reader did not
ask for. The as-at span is dropped between the question and the plan, and the
answer is then labelled with a DIFFERENT date, which is worse than dropping it
silently: the receipt asserts the very thing that did not happen.

WHY IT LOOKED LIKE IT WORKED. "as at 31 December 2024" refuses on that book, so
the capability appears to fail closed. It does not. `2024` is a value of
`vintage_year`, so the CATEGORICAL owner claims it and the coverage gate refuses
it as an unaccounted *category*. Move the year outside the book's vintage range
— 1999, 1975 — and the refusal disappears with it. The guard was a coincidence,
and its message ("you asked about 2024") describes a category, not a date.

THE HOLE, stated exactly: `completeness.stated_concepts` consults the axis,
scope, value, dataset, measure, forecast-target and facet owners. No owner is
asked what PERIOD the sentence named, so a point-in-time span enters no ledger,
and a gate that refuses on unaccounted concepts has nothing to refuse.

The repair is the same shape as the estate's other accounting repairs: ONE owner
answers "what period did this sentence name" — `period_change.recognition`,
which already owns `since`, `between`, `from…to` and `versus` — and the ledger
records what it says. Nothing here resolves a date to a snapshot; that is the
executor's job when point-in-time lands. Until then an as-at question is
UNACCOUNTED, which is a refusal, which is correct.
"""

from __future__ import annotations

import os
import warnings

import pytest

os.environ.setdefault("TRAKT_RUNTIME_MODE", "development")
warnings.simplefilter("ignore")


# --------------------------------------------------------------------------- #
# The owner: what period did this sentence name?
# --------------------------------------------------------------------------- #
#: Deliberately spans years INSIDE and OUTSIDE any plausible vintage range, so a
#: test cannot pass because a year happened to collide with a categorical value.
AS_AT_SENTENCES = (
    ("What is the balance as at 31 December 1999?", "31 december 1999"),
    ("What is the balance as at 31 December 2024?", "31 december 2024"),
    ("What was the balance at 31 March 2026?", "31 march 2026"),
    ("What is the balance as at 30 April 1975?", "30 april 1975"),
    ("What is the balance as of 2026-06-30?", "2026-06-30"),
    ("How many loans were there at 31 December 1999?", "31 december 1999"),
    ("What is the balance as at June 2026?", "june 2026"),
)

#: As-at FRAMING with nothing a governed date owner can resolve. The restriction
#: is still material — the reader asked for one moment and not another — so it
#: must be reported as unresolved rather than dropped.
AS_AT_WITHOUT_A_RESOLVABLE_DATE = (
    "What is the balance as at the 45th of Octember?",
    "What is the balance as at the end of the reign?",
)

#: Sentences that name NO point in time. The owner must stay silent on these, or
#: every ordinary question becomes a temporal refusal.
NOT_AS_AT = (
    "What is the total balance?",
    "How many loans are there?",
    "Total balance by region",
    "Show balance over time",
    "How has the balance changed since last month?",
    "What is the total balance in Scotland?",
    "Compare balance over time",
    "Total balance by month",
    "What is the average loan size?",
    "Balance over time for joint borrowers",
)


class TestThePeriodOwnerRecognisesAPointInTime:
    def test_an_as_at_sentence_names_its_date(self) -> None:
        from mi_agent.period_change import recognition

        for sentence, expected in AS_AT_SENTENCES:
            request = recognition.as_at_request(sentence)
            assert request is not None, f"{sentence!r} names a point in time"
            assert request.token is not None, f"{sentence!r} carries a date"
            assert request.token.lower() == expected, (
                f"{sentence!r} -> {request.token!r}, expected {expected!r}")

    def test_as_at_framing_without_a_date_is_still_reported(self) -> None:
        from mi_agent.period_change import recognition

        for sentence in AS_AT_WITHOUT_A_RESOLVABLE_DATE:
            request = recognition.as_at_request(sentence)
            assert request is not None, (
                f"{sentence!r} asks for one moment; the owner must say so even "
                "though it cannot resolve which")
            assert request.token is None

    def test_a_sentence_naming_no_moment_is_not_claimed(self) -> None:
        from mi_agent.period_change import recognition

        for sentence in NOT_AS_AT:
            assert recognition.as_at_request(sentence) is None, (
                f"{sentence!r} names no point in time and must not be claimed")


# --------------------------------------------------------------------------- #
# The ledger: the concept has to be IN it
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def semantics():
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent_api.app import semantics_path

    return load_mi_semantics(semantics_path())


class TestTheLedgerRecordsThePeriod:
    def test_an_as_at_date_is_a_stated_concept(self, semantics) -> None:
        from question_interpretation.completeness import stated_concepts

        for sentence, _token in AS_AT_SENTENCES:
            kinds = {c.kind for c in stated_concepts(sentence, semantics)}
            assert "temporal" in kinds, (
                f"{sentence!r} states a point in time and no owner recorded it; "
                f"the ledger holds only {sorted(kinds)}")

    def test_an_ordinary_question_states_no_period(self, semantics) -> None:
        from question_interpretation.completeness import stated_concepts

        for sentence in NOT_AS_AT:
            kinds = {c.kind for c in stated_concepts(sentence, semantics)}
            assert "temporal" not in kinds, (
                f"{sentence!r} names no point in time, but a temporal concept "
                "was raised — every ordinary question would now refuse")


# --------------------------------------------------------------------------- #
# End to end: no confident figure over the wrong period
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def ask():
    from demo_platform import config as cfg

    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    os.environ["MI_AGENT_AUTH_ENABLED"] = "false"
    from fastapi.testclient import TestClient

    from mi_agent_api.app import app

    client = TestClient(app)
    cache = {}

    def _ask(question: str) -> dict:
        if question not in cache:
            cache[question] = client.post(
                "/mi/query",
                json={"question": question, "portfolioId": cfg.CLIENT_ID}).json()
        return cache[question]

    return _ask


class TestNoFigureOverAPeriodThatWasNotHonoured:
    @pytest.mark.parametrize("sentence", [s for s, _ in AS_AT_SENTENCES]
                             + list(AS_AT_WITHOUT_A_RESOLVABLE_DATE))
    def test_an_as_at_question_does_not_answer(self, ask, sentence) -> None:
        envelope = ask(sentence)
        assert not envelope.get("ok"), (
            f"{sentence!r} was ANSWERED. No point-in-time capability exists, so "
            f"the figure is the CURRENT book under a question about another "
            f"moment: {envelope.get('answer')}")

    @pytest.mark.parametrize("sentence", NOT_AS_AT)
    def test_an_ordinary_question_is_unaffected(self, ask, sentence) -> None:
        """The reverse failure: this repair must not refuse ordinary questions."""
        from due_diligence.evidence.mi_api_certification.broad import refusal_class

        envelope = ask(sentence)
        if envelope.get("ok"):
            return
        assert refusal_class(envelope) != "SEMANTIC_UNRESOLVED", (
            f"{sentence!r} names no period yet was refused as unresolved: "
            f"{envelope.get('answer')}")
