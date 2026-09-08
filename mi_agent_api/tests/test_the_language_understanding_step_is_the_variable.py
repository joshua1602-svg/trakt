#!/usr/bin/env python3
"""Why one question answered in one replay and refused in the next.

THE OBSERVATION. "How many cases left KFI in the last week" ANSWERED in one
115-question replay and, on byte-identical deployed code, came back as *"I could
not complete the language-understanding step for this question"* in the next.
For a client-facing assessment that is worse than a steady failure: a reader can
work around a refusal they can predict, and cannot reason about this one at all.

THE DIAGNOSIS, reproduced deterministically below. Nothing in the deterministic
reading varies — the same question parses to the same contract and reaches the
same route every time, with the arm off (`test_the_deterministic_reading_is_
stable`). The variable is ONE outbound model call.

`concept_merge_arm.apply` asks the model to propose concepts in registered
vocabulary. Any exception from that call — rate limit, overload, timeout,
exhausted credit, a reply that will not parse — is caught and reported as
``status: proposal_unavailable``. `mi_service._enforce_model_availability` then
turns an otherwise-successful envelope into the language-understanding refusal,
DELIBERATELY: an arm that was switched on and did not answer may not be allowed
to silently narrow the question. That rule is right. Its consequence is that the
arm's availability decides the outcome of a question the deterministic path
answers perfectly well, and availability is not a property of the code.

Two things follow, and both are pinned below.

  1. THE SAME QUESTION, THE SAME CODE, TWO OUTCOMES — selected by whether one
     call returned. That is the whole non-determinism; there is nothing else to
     find in the parse.

  2. THE RECORD CANNOT TELL THIS FROM A BROKEN CALCULATION.
     `_classify_analytical_failure` recognises the coverage gate's marker, the
     capability boundary and an unmapped question; the availability refusal sets
     none of them, so it falls through to ``CALCULATION_FAILED`` — category
     ``capability``, ``retryable: false`` — while the sentence the reader is
     shown ends "Please try again". `mi_query_telemetry` and
     `migration_phase0/replay_probe` both count that code as an ERROR, so a
     transient model outage is recorded as the system having broken. This is
     exactly the mislabel the coverage gate was given ``semanticCoverageRefused``
     to escape on 2026-09-03, and this path never got the same treatment.

WHAT IS NOT CHANGED HERE. Nothing. This file is the diagnosis, and
`test_the_record_calls_an_unavailable_model_a_failed_calculation` pins the gap
as it stands so it cannot be lost — closing it means changing that test WITH the
fix, which needs a governed decision this session did not have: a new error code
is part of the external contract (`trakt_core.errors` says so), and no existing
code means "an upstream model was unavailable; ask again".

A SECOND SOURCE, NAMED SO IT IS NOT CONFUSED WITH THIS ONE. The free-form parser
arm (`MI_AGENT_LLM_PARSER`) fails DIFFERENTLY: it falls back to the deterministic
reading and answers, publishing ``parser_used:
deterministic_fallback_after_llm_failure``. That varies the ANSWER rather than
the outcome, and it carries no such sentence — so the observed refusal is this
path and not that one.
"""

from __future__ import annotations

import os
import unittest

from mi_agent import llm_query_parser as LQ
from mi_agent_api import concept_merge_arm as ARM
from mi_agent_api import mi_service
from mi_agent_api.tests.test_stage_movement_query import ask

QUESTION = "How many cases left KFI in the last week?"
ROUTE = "pipeline_stage_movement"

#: The reader-facing sentence `_enforce_model_availability` publishes.
LANGUAGE_UNDERSTANDING = "I could not complete the language-understanding step"

#: A well-formed reply that proposes nothing. Availability, not content, is the
#: variable under test — so the AVAILABLE arm must change nothing about the
#: answer, and this is the reply that guarantees it.
NO_PROPOSALS = '{"concepts": []}'


class _Arm:
    """The arm switched on, with its one outbound call replaced.

    The call is the only thing that varies between the two runs below: same
    process, same code, same question, same fixture.
    """

    def __init__(self, call):
        self._call = call
        self._saved = {}

    def __enter__(self):
        self._saved = {k: os.environ.get(k) for k in
                       ("MI_AGENT_CONCEPT_MERGE", "ANTHROPIC_API_KEY")}
        os.environ["MI_AGENT_CONCEPT_MERGE"] = "on"
        os.environ["ANTHROPIC_API_KEY"] = "sk-not-used-the-call-is-replaced"
        self._original = LQ._call_llm
        LQ._call_llm = self._call
        return self

    def __exit__(self, *exc):
        LQ._call_llm = self._original
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        return False


def _unavailable(*_a, **_k):
    """What the deployment did on the failing replay: the call did not return."""
    raise RuntimeError("overloaded_error: the model is temporarily overloaded")


def _available(*_a, **_k):
    return NO_PROPOSALS, {}, False


class TestTheDeterministicReadingIsNotTheVariable(unittest.TestCase):

    def test_the_deterministic_reading_is_stable(self):
        """Five identical asks with the arm off. If the parse were the variable
        it would show here, and it does not."""
        seen = {(e.get("ok"), (e.get("metadata") or {}).get("route"),
                 e.get("answer")) for e in (ask(QUESTION) for _ in range(5))}
        self.assertEqual(len(seen), 1, seen)
        ok, route, answer = seen.pop()
        self.assertTrue(ok)
        self.assertEqual(route, ROUTE)
        self.assertIn("KFI", answer or "")


class TestOneOutboundCallDecidesTheOutcome(unittest.TestCase):

    def test_an_unavailable_call_refuses_a_question_that_answers(self):
        with _Arm(_unavailable):
            envelope = ask(QUESTION)
        self.assertFalse(envelope.get("ok"))
        self.assertIn(LANGUAGE_UNDERSTANDING, envelope.get("error") or "")
        evidence = (envelope.get("metadata") or {}).get("conceptMerge") or {}
        self.assertEqual(evidence.get("status"), ARM.PROPOSAL_UNAVAILABLE)
        # The route CLAIMED and ANSWERED it; the refusal is stamped on top.
        self.assertEqual((envelope.get("metadata") or {}).get("route"), ROUTE)

    def test_the_same_question_answers_when_the_call_returns(self):
        with _Arm(_available):
            envelope = ask(QUESTION)
        self.assertTrue(envelope.get("ok"), envelope.get("error"))
        self.assertEqual((envelope.get("metadata") or {}).get("route"), ROUTE)
        self.assertEqual(
            ((envelope.get("metadata") or {}).get("conceptMerge") or {}).get("status"),
            "no_change")

    def test_the_two_runs_differ_in_nothing_but_availability(self):
        """The replay's two outcomes, side by side, in one process."""
        with _Arm(_unavailable):
            refused = ask(QUESTION)
        with _Arm(_available):
            answered = ask(QUESTION)
        self.assertNotEqual(bool(refused.get("ok")), bool(answered.get("ok")))
        self.assertEqual(answered.get("answer"), ask(QUESTION).get("answer"))


class TestWhatTheRecordSays(unittest.TestCase):
    """The finding an operator has to live with, pinned as it stands."""

    def test_the_record_now_calls_an_unavailable_model_what_it_is(self):
        """THE GAP THIS TEST WAS LEFT TO HOLD, closed 2026-09-04.

        It used to assert ``CALCULATION_FAILED`` as a FINDING rather than a
        desideratum: that code is what `mi_query_telemetry` and
        `migration_phase0/replay_probe` both count as an ERROR, so every one of
        these refusals was recorded as the system having broken — and
        ``retryable: false`` contradicted the sentence the reader is shown,
        which ends "Please try again". Its docstring said closing the gap meant
        changing it with the fix. This is that change.

        `SEMANTIC_MODEL_UNAVAILABLE` is INFRASTRUCTURE and retryable, so an
        operator counting broken calculations is no longer counting model
        outages, and an autonomous caller can tell that waiting may help. The
        REFUSAL is unchanged: availability still fails closed, because the
        estate has no completeness proof independent of the deterministic parse.
        """
        from trakt_core.errors import ErrorCode, is_retryable

        with _Arm(_unavailable):
            envelope = ask(QUESTION)
        error = ((envelope.get("governance") or {}).get("error")) or {}
        self.assertEqual(error.get("code"), ErrorCode.SEMANTIC_MODEL_UNAVAILABLE)
        self.assertTrue(is_retryable(error.get("code")))
        self.assertNotEqual(error.get("code"), "CALCULATION_FAILED")
        # And it is still a refusal, with no figure.
        self.assertFalse(envelope.get("ok"))
        self.assertFalse(envelope.get("artifacts"))
        self.assertEqual(error.get("category"), "infrastructure")
        self.assertIs(error.get("retryable"), True)
        # The sentence and the contract now agree: it says try again, and the
        # record says the caller may.
        self.assertIn("try again", str(error.get("message") or "").lower())

    def test_the_classifier_reads_the_gate_s_marker_and_not_the_evidence(self):
        """Where the gap WAS. It named the fix's obvious home: the coverage gate
        marks its own decline and is classified by that marker, and this one
        published none. It does now — and the distinction that remains is worth
        keeping.

        A CLASSIFICATION FOLLOWS A DECISION, NOT AN OBSERVATION. The arm's
        evidence block records what happened to the model; the GATE decides
        whether that cost the answer. An envelope carrying `conceptMerge:
        proposal_unavailable` where no gate refused is an answer that stood
        despite an unavailable arm, and calling that a failed request would be
        as wrong as the code this replaced.
        """
        from trakt_core.errors import ErrorCode

        # The gate refused: its marker is present, and it classifies.
        refused = {"ok": False, "error": "…", "controlledRefusal": True,
                   "metadata": {"modelUnavailableRefused": True,
                                "conceptMerge": {
                                    "status": ARM.PROPOSAL_UNAVAILABLE}}}
        self.assertEqual(mi_service._classify_analytical_failure(refused),
                         ErrorCode.SEMANTIC_MODEL_UNAVAILABLE)
        # The evidence ALONE is not a verdict about the request.
        observed = {"ok": False, "error": "…", "controlledRefusal": True,
                    "metadata": {"conceptMerge": {
                        "status": ARM.PROPOSAL_UNAVAILABLE}}}
        self.assertNotEqual(mi_service._classify_analytical_failure(observed),
                            ErrorCode.SEMANTIC_MODEL_UNAVAILABLE)
        # The precedent, one branch below it in the same function.
        self.assertEqual(
            mi_service._classify_analytical_failure(
                {"metadata": {"semanticCoverageRefused": True}}),
            "UNSUPPORTED_QUESTION")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
