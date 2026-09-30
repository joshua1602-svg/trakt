"""The pipeline's case history is fetched only by a plan that reads it.

It is built from every weekly extract, and until 2026-09-30 the production seam
fetched it — listing the extracts and copying the cached model — for EVERY
governed attempt, a funded question included. The seam now hands the canary a
provider, and the canary resolves it in the one branch that reads it.
"""
from __future__ import annotations

import inspect

from mi_agent import plan_serving_canary as canary
from tests.interpretation_v2 import test_funded_breadth as funded
from tests.interpretation_v2 import test_stage_conversion as stage
from tests.interpretation_v2.test_funded_breadth import book, semantics  # noqa: F401
from tests.interpretation_v2.test_stage_conversion import history  # noqa: F401


class _Provider:
    def __init__(self, model=None, fail=False):
        self.model, self.fail, self.calls = model, fail, 0

    def __call__(self):
        self.calls += 1
        if self.fail:
            raise OSError("extract listing failed")
        return self.model


def test_a_stage_plan_fetches_it_once(monkeypatch, history):
    provider = _Provider(history)
    payload = stage._served(stage._intent("expected_completion_date"),
                            monkeypatch, provider)
    assert payload["answer"].startswith("Expected completion date: ")
    assert provider.calls == 1


def test_a_funded_plan_never_fetches_it(monkeypatch, book, semantics):
    provider = _Provider(fail=True)
    real_serve = canary.serve
    monkeypatch.setattr(canary, "serve", lambda **kw: real_serve(
        **kw, pipeline_history=provider))
    payload, record, _ = funded._served(
        funded._intent(operation="point_in_time"), monkeypatch, book, semantics)
    assert payload is not None, record.get("execution")
    assert provider.calls == 0


def test_a_provider_that_fails_is_an_unresolved_input_not_a_failed_request():
    assert canary._history(_Provider(fail=True)) is None
    assert canary._history({"a": 1}) == {"a": 1}
    assert canary._history(None) is None


def test_the_seam_hands_over_a_provider_not_the_model():
    from mi_agent_api import mi_service
    source = inspect.getsource(mi_service)
    start = source.index("def _pipeline_inputs(")
    body = source[start:source.index("def _governed_serving_attempt(", start)]
    assert "lambda: ds_mod._pipeline_history(cid)" in body
    assert 'found["pipeline_history"] = ds_mod._pipeline_history(' not in body
