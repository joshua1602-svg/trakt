#!/usr/bin/env python3
"""A model that did not answer is not a calculation that broke.

THE EVIDENCE, replay_after_409 row 6:

    How many cases left KFI in the last week
    ANSWERED -> ERROR, error_code CALCULATION_FAILED, retryable false
    "I could not complete the language-understanding step ... Please try again."
    modelUsage: None   parserMode: None

Three separate contract failures in one row.

1. THE CODE IS WRONG. Nothing miscalculated. The concept-merge arm could not
   reach its model, the estate refused rather than answer from a partial
   reading, and that refusal was labelled a broken calculation and marked
   NOT retryable — while its own sentence told the reader to try again.

2. THE SAME CODE COVERS A DIFFERENT EVENT. "Has pipeline progression improved
   month on month?" is a governed SEMANTIC refusal — the series is weekly and
   the question asked for months — and it also reports CALCULATION_FAILED.
   The coverage guard already has the right shape for this
   (`semanticCoverageRefused` -> UNSUPPORTED_QUESTION); the facet guard did not.

3. THE EVENT IS INVISIBLE. `modelUsage` is stamped on the point-in-time path
   only, so a ROUTED question carries no model telemetry at all — and an arm
   that FAILED produces no usage to report even there. The one row whose
   outcome the model decided is the one row that cannot show it.

None of this changes whether a reader gets an answer. Availability still
refuses, fail-closed, exactly as `_refuse_when_model_unavailable` decided: the
estate has no completeness proof independent of the deterministic parse, so
standing down on unavailability could silently broaden a question.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent_api import mi_service as MS
from trakt_core.errors import ErrorCategory, ErrorCode, category_for, is_retryable


def _unavailable_envelope():
    """An envelope as `_refuse_when_model_unavailable` leaves it."""
    return {
        "ok": False,
        "error": MS._AVAILABILITY_REFUSAL,
        "answer": MS._AVAILABILITY_REFUSAL,
        "controlledRefusal": True,
        "metadata": {"modelUnavailableRefused": True,
                     "conceptMerge": {"status": "proposal_unavailable",
                                      "detail": "APIStatusError: overloaded"}},
    }


def _guard_refused_envelope():
    """An envelope as the facet guard leaves a semantic refusal."""
    return {
        "ok": False,
        "error": "I understood that you asked for month, but that could not be "
                 "applied to the calculation.",
        "controlledRefusal": True,
        "metadata": {"semanticGuardRefused": True},
    }


# ------------------------------------------------ 1. availability is its own #
def test_the_code_exists_and_is_retryable():
    assert hasattr(ErrorCode, "SEMANTIC_MODEL_UNAVAILABLE")
    code = ErrorCode.SEMANTIC_MODEL_UNAVAILABLE
    assert is_retryable(code) is True
    assert category_for(code) == ErrorCategory.INFRASTRUCTURE


def test_an_unavailable_model_is_not_a_calculation_failure():
    assert (MS._classify_analytical_failure(_unavailable_envelope())
            == ErrorCode.SEMANTIC_MODEL_UNAVAILABLE)


def test_it_is_distinguishable_from_every_capability_refusal():
    """The point of the code: an operator counting broken calculations must not
    be counting model outages, and vice versa."""
    assert (category_for(ErrorCode.SEMANTIC_MODEL_UNAVAILABLE)
            != category_for(ErrorCode.CALCULATION_FAILED))
    assert (category_for(ErrorCode.SEMANTIC_MODEL_UNAVAILABLE)
            != category_for(ErrorCode.UNSUPPORTED_QUESTION))


def test_the_caller_contract_does_not_change():
    """A governed MI answer is HTTP 200 with the verdict inside the envelope.
    An outage must not become a transport failure a caller handles elsewhere."""
    from trakt_core.errors import http_status_for

    assert (http_status_for(ErrorCode.SEMANTIC_MODEL_UNAVAILABLE)
            == http_status_for(ErrorCode.CALCULATION_FAILED))


# ------------------------------------------- 2. a semantic refusal is refused #
def test_a_semantic_guard_refusal_is_not_a_calculation_failure():
    assert (MS._classify_analytical_failure(_guard_refused_envelope())
            == ErrorCode.UNSUPPORTED_QUESTION)


def test_it_agrees_with_the_coverage_guard_that_already_had_this_right():
    coverage = {"ok": False, "metadata": {"semanticCoverageRefused": True}}
    assert (MS._classify_analytical_failure(coverage)
            == MS._classify_analytical_failure(_guard_refused_envelope()))


def test_a_genuine_calculation_failure_is_still_one():
    """The classifier must not become a machine for never saying so."""
    broken = {"ok": False, "metadata": {},
              "validation": {"errors": ["divide by zero in weighted average"]}}
    assert (MS._classify_analytical_failure(broken)
            == ErrorCode.CALCULATION_FAILED)


def test_availability_outranks_the_semantic_marker():
    """Both markers present: the model never ran, so nothing downstream of it
    can be the reason."""
    envelope = _unavailable_envelope()
    envelope["metadata"]["semanticGuardRefused"] = True
    assert (MS._classify_analytical_failure(envelope)
            == ErrorCode.SEMANTIC_MODEL_UNAVAILABLE)


# ------------------------------------------------- 3. the attempt is recorded #
def test_availability_is_carried_separately_from_usage():
    block = MS._model_availability({"status": "proposal_unavailable",
                                    "detail": "APIStatusError: overloaded"})
    assert block["concept_merge_attempted"] is True
    assert block["concept_merge_status"] == "proposal_unavailable"
    assert block["retryable"] is True
    assert block["failure_class"]


def test_a_successful_call_records_the_attempt_too():
    block = MS._model_availability({"status": "applied", "model": "claude-opus-5"})
    assert block["concept_merge_attempted"] is True
    assert block["concept_merge_status"] == "applied"
    assert block["retryable"] is False
    assert block["failure_class"] is None


def test_an_arm_that_was_switched_off_says_so_rather_than_going_quiet():
    block = MS._model_availability(None)
    assert block["concept_merge_attempted"] is False
    assert block["concept_merge_status"] is None


def test_no_provider_message_or_secret_reaches_the_block():
    """`detail` carries a raw provider string. It is CLASSIFIED, never
    forwarded: a stack trace or a key in an upstream message must not travel
    into telemetry a probe writes to disk."""
    block = MS._model_availability({
        "status": "proposal_unavailable",
        "detail": "AuthenticationError: invalid x-api-key sk-ant-SECRET"})
    assert "sk-ant" not in repr(block)
    assert "SECRET" not in repr(block)
    assert block["failure_class"] == "authentication"


def test_the_failure_classes_are_the_ones_the_brief_named():
    for detail, expected in (("APIStatusError: overloaded_error", "overloaded"),
                             ("APITimeoutError: timed out", "timeout"),
                             ("JSONDecodeError: Expecting value", "malformed"),
                             ("AuthenticationError: bad key", "authentication"),
                             ("ValueError: something else", "unknown")):
        block = MS._model_availability({"status": "proposal_unavailable",
                                        "detail": detail})
        assert block["failure_class"] == expected, detail


# ------------------------------- 5. the contract holds under every failure #
class _ArmFails:
    """The concept-merge arm on, with its outbound call raising `exc`."""

    def __init__(self, exc):
        self._exc = exc

    def __enter__(self):
        import os
        from mi_agent import llm_query_parser as LQ

        self._saved = {k: os.environ.get(k) for k in
                       ("MI_AGENT_CONCEPT_MERGE", "ANTHROPIC_API_KEY")}
        os.environ["MI_AGENT_CONCEPT_MERGE"] = "on"
        os.environ["ANTHROPIC_API_KEY"] = "sk-not-used-the-call-is-replaced"
        self._original = LQ._call_llm

        def _boom(*a, **k):
            raise self._exc
        LQ._call_llm = _boom
        return self

    def __exit__(self, *exc):
        import os
        from mi_agent import llm_query_parser as LQ

        LQ._call_llm = self._original
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        return False


#: The failure modes the brief named, each as the provider raises it.
_MODES = [
    ("overload", RuntimeError("APIStatusError: overloaded_error (529)")),
    ("timeout", RuntimeError("APITimeoutError: request timed out")),
    ("malformed", ValueError("JSONDecodeError: Expecting value: line 1")),
]

QUESTION = "How many cases left KFI in the last week"


def _ask_with(exc):
    import sys
    sys.argv = ["pytest"]
    from mi_agent_api.tests.test_stage_movement_query import ask

    with _ArmFails(exc):
        return ask(QUESTION)


def test_the_outcome_contract_is_stable_under_every_failure_mode():
    """THE STABILITY ACCEPTANCE. Whatever the dependency does, the estate says
    the same three things: it did not answer, it is retryable, and it is not a
    broken calculation."""
    for name, exc in _MODES:
        envelope = _ask_with(exc)
        meta = envelope.get("metadata") or {}
        assert envelope.get("ok") is False, name
        assert meta.get("modelUnavailableRefused") is True, name
        assert (MS._classify_analytical_failure(envelope)
                == ErrorCode.SEMANTIC_MODEL_UNAVAILABLE), name
        assert is_retryable(ErrorCode.SEMANTIC_MODEL_UNAVAILABLE), name


def test_the_attempt_is_visible_in_every_failure_mode():
    """The gap that hid the KFI regression: a routed question with no model
    telemetry at all."""
    for name, exc in _MODES:
        meta = (_ask_with(exc).get("metadata") or {})
        block = meta.get("modelAvailability")
        assert block, name
        assert block["concept_merge_attempted"] is True, name
        assert block["concept_merge_status"] == "proposal_unavailable", name
        assert block["retryable"] is True, name


def test_it_never_silently_broadens_the_deterministic_answer():
    """THE RULE THAT MUST NOT BEND. Standing down to the deterministic parse on
    unavailability could answer a wider question than the reader asked; the
    estate has no completeness proof independent of that parse. So every mode
    refuses, and none returns a figure."""
    for name, exc in _MODES:
        envelope = _ask_with(exc)
        assert envelope.get("ok") is False, name
        assert not envelope.get("artifacts"), name
        assert "try again" in str(envelope.get("answer") or "").lower(), name


def test_a_working_model_still_answers_the_same_question():
    """The pair. If this refuses too, the tests above prove nothing."""
    import os
    import sys
    sys.argv = ["pytest"]
    from mi_agent_api.tests.test_stage_movement_query import ask

    saved = os.environ.get("MI_AGENT_CONCEPT_MERGE")
    os.environ["MI_AGENT_CONCEPT_MERGE"] = "off"
    try:
        envelope = ask(QUESTION)
    finally:
        if saved is None:
            os.environ.pop("MI_AGENT_CONCEPT_MERGE", None)
        else:
            os.environ["MI_AGENT_CONCEPT_MERGE"] = saved
    assert envelope.get("ok") is True, envelope.get("error")
