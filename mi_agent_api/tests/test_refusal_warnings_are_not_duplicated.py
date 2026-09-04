"""A refused answer says each thing it says exactly once.

Reported defect: a pipeline question the coverage gate refused rendered THREE
lines in the chat's warnings box, for what is genuinely one gap:

  1. "Scope not narrowed: the governed weekly pipeline extract carries no
     source-portfolio provenance..." — pipeline_summary's own, route-specific
     disclosure (chat_routing._pipeline_scope_disclosure).
  2. "Scope not narrowed: this routed answer is computed across the whole
     platform book..." — the generic dispatcher-level disclosure
     (chat_routing._disclose_lens_scope), re-deriving the SAME fact the route
     had just disclosed, in different words.
  3. "I understood that you asked about pipeline, but I could not confirm it
     was applied to this calculation..." — the coverage refusal
     (mi_service._enforce_semantic_coverage), duplicated into `warnings` even
     though it is already `answer` and `error` — the exact text the chat's
     red bubble already renders.

Two independent fixes, pinned here against the real functions rather than a
mock of their combined effect:

  * `_enforce_semantic_coverage` no longer appends the refusal message to
    `warnings` — it is already `answer`/`error`.
  * `_disclose_lens_scope` stands down when a "Scope not narrowed:" sentence
    is already present, rather than appending a second one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent_api import chat_routing as CR  # noqa: E402
from mi_agent_api import mi_service as MS  # noqa: E402

QUESTION = "Summarise the current pipeline."

_ROUTE_DISCLOSURE = (
    "Scope not narrowed: the governed weekly pipeline extract carries no "
    "source-portfolio provenance, so this position is the whole platform "
    "pipeline and is NOT narrowed to a selected book.")


class _FakeLens:
    label = "Direct"
    filters = {"source_portfolio_type": "direct"}


def _routed_envelope(*, declared: bool):
    """An envelope shaped exactly as pipeline_summary emits it, then passed
    through the same dispatcher-level lens disclosure every route gets."""
    warnings = [_ROUTE_DISCLOSURE] if declared else []
    env = CR._envelope(
        ok=True, question=QUESTION,
        answer="At the weekly extract the pipeline holds ...",
        spec={}, artifacts=[], route="pipeline_summary",
        lens_applied=False, reconciliation=None, warnings=warnings,
    )
    orig = CR._resolve_lens
    CR._resolve_lens = lambda q, sl: _FakeLens()
    try:
        return CR._disclose_lens_scope(env, QUESTION, source_lens="direct_001")
    finally:
        CR._resolve_lens = orig


def _gate(env):
    """The service's own stamp-then-enforce sequence, undeclared so it refuses."""
    from mi_agent_api.datasets import load_mi_semantics, semantics_path
    env.setdefault("metadata", {})
    MS._stamp_semantic_coverage(env, question=QUESTION,
                                semantics=load_mi_semantics(semantics_path()),
                                frame=None)
    return MS._enforce_semantic_coverage(env)


def test_a_route_that_already_disclosed_its_scope_is_not_disclosed_again():
    env = _routed_envelope(declared=True)
    scope_warnings = [w for w in env["warnings"] if w.startswith("Scope not narrowed:")]
    assert len(scope_warnings) == 1, env["warnings"]
    assert scope_warnings[0] == _ROUTE_DISCLOSURE


def test_a_route_with_no_disclosure_of_its_own_still_gets_the_generic_one():
    """The fallback this generic mechanism exists for is UNCHANGED — it only
    stands down once something has already said the scope is not narrowed."""
    env = _routed_envelope(declared=False)
    scope_warnings = [w for w in env["warnings"] if w.startswith("Scope not narrowed:")]
    assert len(scope_warnings) == 1, env["warnings"]
    assert "NOT Direct-only" in scope_warnings[0]


def test_the_refused_answer_is_not_also_a_warning():
    env = _routed_envelope(declared=True)
    gated = _gate(env)
    assert gated["ok"] is False
    assert "could not confirm it was applied" in gated["answer"]
    assert not any("could not confirm it was applied" in w for w in gated["warnings"]), (
        gated["warnings"])


def test_the_refused_pipeline_question_shows_exactly_one_warning():
    """THE SCREENSHOT, END TO END. One route-level scope disclosure survives;
    the generic re-disclosure and the duplicated refusal text do not."""
    env = _routed_envelope(declared=True)
    gated = _gate(env)
    assert gated["ok"] is False
    assert gated["warnings"] == [_ROUTE_DISCLOSURE], gated["warnings"]


def test_a_successful_answer_keeps_its_scope_disclosure():
    """Coverage enforcement only touches refusals — a warning on a SUCCESSFUL
    answer must not be removed by either fix."""
    env = _routed_envelope(declared=True)
    assert env["warnings"] == [_ROUTE_DISCLOSURE]
    # ok stays True with nothing unaccounted (no coverage stamp run here);
    # the disclosure must survive untouched.
    assert env["ok"] is True
