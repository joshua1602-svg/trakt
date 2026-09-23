"""One `TypeError` cost a delivery a night, four stages downstream.

The live pack's onboarding summary::

    "status": "FAILED",
    "error": "TypeError: Messages.create() got an unexpected keyword argument
              'temperature'"

`anthropic` 1.x removed ``temperature`` from ``messages.create()`` — and
`requirements.txt` pins `anthropic>=0.40.0` with no upper bound, so the App
Service installed an SDK newer than the code. The call raises before any HTTP
request is made. Onboarding records a failed run rather than raising, so nothing
stopped: artefacts 01–09 were never written, the central tape then found an
empty inventory, and the operator was told

    "No file in this delivery was opened … The delivery appears to hold no
     files."

which is true, and is four stages away from the fault.

THE LESSON WAS ALREADY LEARNED AND NOT SHARED. `mi_agent.llm_query_parser`
hit this exact `TypeError` on 166 acceptance questions, worked out that it is
two questions — does this SDK expose the parameter, does this model accept it —
and wrote a runtime check. Three other callers went on passing `temperature=`.
That is the same shape as every other defect on this delivery: one fact held in
two places, and only one of them right.

So these assert the MECHANISM, which a model release cannot invalidate and a
stale allowlist cannot satisfy.
"""

from __future__ import annotations

import inspect

import pytest

from trakt_core import llm_sampling


class _NoSamplingSDK:
    """`anthropic` 1.x: no `temperature` parameter on `messages.create`."""

    class messages:  # noqa: N801 - mirrors the SDK's own shape
        @staticmethod
        def create(*, model, max_tokens, system, messages):
            raise AssertionError("not called in this test")


class _SamplingSDK:
    class messages:  # noqa: N801
        @staticmethod
        def create(*, model, max_tokens, system, messages, temperature=None):
            raise AssertionError("not called in this test")


class _OpaqueSDK:
    class messages:  # noqa: N801
        @staticmethod
        def create(**kwargs):
            raise AssertionError("not called in this test")


class TestTheSdkIsAskedRatherThanAssumed:

    def test_an_sdk_without_the_parameter_is_sent_none(self):
        for model in ("claude-opus-5", "claude-haiku-4-5-20251001",
                      "claude-sonnet-4-6", "some-model-shipped-tomorrow"):
            assert llm_sampling.sampling_for(_NoSamplingSDK(), model) == {}

    def test_an_sdk_with_the_parameter_is_asked_for_determinism(self):
        assert llm_sampling.sampling_for(
            _SamplingSDK(), "claude-sonnet-4-6") == {"temperature": 0.0}

    def test_an_sdk_taking_arbitrary_kwargs_defers_to_the_model(self):
        assert "temperature" in llm_sampling.sdk_sampling_parameters(
            _OpaqueSDK())

    def test_an_unreadable_signature_defers_to_the_model(self):
        class _Weird:
            messages = object()
        assert "temperature" in llm_sampling.sdk_sampling_parameters(_Weird())

    def test_a_model_the_api_already_refused_is_remembered(self):
        assert llm_sampling.sampling_for(
            _SamplingSDK(), "claude-opus-9",
            rejected={"claude-opus-9"}) == {}
        assert llm_sampling.sampling_for(
            _SamplingSDK(), "claude-opus-8",
            rejected={"claude-opus-9"}) == {"temperature": 0.0}


class TestBothShapesOfOneRejectionAreRecognised:
    """A `TypeError` from the SDK and a 400 from the API are one fact told two
    ways."""

    @pytest.mark.parametrize("exc", [
        TypeError("Messages.create() got an unexpected keyword argument "
                  "'temperature'"),
        RuntimeError("Error code: 400 - temperature is not supported"),
        RuntimeError("invalid_request_error: top_p is unsupported"),
    ])
    def test_it_is_read_as_a_sampling_rejection(self, exc):
        assert llm_sampling.is_sampling_rejection(exc)

    def test_an_unrelated_failure_is_not(self):
        """Downgrading on everything would hide a real outage behind a retry."""
        assert not llm_sampling.is_sampling_rejection(
            RuntimeError("Error code: 529 - overloaded"))
        assert not llm_sampling.is_sampling_rejection(
            RuntimeError("credit balance is too low"))


class TestEveryCallerAsks:
    """The defect was not that one call was wrong. It was that the answer lived
    in one module and the other callers did not consult it."""

    CALLERS = (
        "engine.onboarding_agent.llm_mapping_reviewer",
        "engine.gate_1_alignment.llm_mapper_agent",
        "engine.enum_agent.enum_mapping_agent",
        "mi_agent.llm_query_parser",
    )

    @pytest.mark.parametrize("module", CALLERS)
    def test_no_caller_hardcodes_the_parameter_into_a_request(self, module):
        import importlib
        src = inspect.getsource(importlib.import_module(module))
        # The kwarg may be READ from configuration and spread into a request
        # the runtime accepts; what must not survive is it being written
        # straight into a `messages.create(...)` call.
        for line in src.splitlines():
            stripped = line.strip()
            if stripped.startswith("temperature=") and "self.temperature" not in stripped:
                pytest.fail(f"{module}: hardcoded sampling kwarg: {stripped}")

    @pytest.mark.parametrize("module", CALLERS)
    def test_every_caller_reaches_the_one_shared_answer(self, module):
        import importlib
        src = inspect.getsource(importlib.import_module(module))
        assert "llm_sampling" in src, (
            f"{module} decides for itself whether sampling can be sent")


class TestTheParserKeepsItsOwnMemory:
    """`mi_agent` learns from a rejection at runtime; delegating the mechanism
    must not cost it that."""

    def test_its_remembered_rejections_still_apply(self, monkeypatch):
        from mi_agent import llm_query_parser as parser
        monkeypatch.setattr(parser, "_SAMPLING_REJECTED", {"claude-opus-9"})
        assert parser._sampling_for(_SamplingSDK(), "claude-opus-9") == {}
        assert parser._sampling_for(
            _SamplingSDK(), "claude-opus-8") == {"temperature": 0.0}
