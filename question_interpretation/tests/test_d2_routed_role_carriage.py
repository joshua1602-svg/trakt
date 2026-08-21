"""D2 — the role decision arriving on the ROUTED path, and what that could break.

`_guard_routed_answer` now consumes `dimension_role` instead of letting
`requested_dimension_terms` assert "axis" for every dimension the question
names. That arrival is a new producer of `KIND_POPULATION` on a path that had
exactly one, and the standing rule from `7c46f81` is that **consolidating a
decision can create a new reader of it**. Two failure modes follow, both of them
recurrences, and both are asserted here rather than reasoned about:

  DUPLICATE   the ledger and the owner raise the same governed population, one
              is stamped applied and the other left lost, and the answer refuses
              itself. Live for ten minutes in `7c46f81`.
  UNSTAMPED   a population that is NOT a duplicate is pulled out before
              `reconcile_routed_facets` and never handed to `reconcile_population`,
              so it keeps its LOST default whatever execution did. That is
              `e35a01b`'s shape — a reclassification into a kind with no receiver
              — arriving on this path.

None of the 693 corpus questions reaches this: the deterministic parser resolves
no categorical filter, so no corpus question puts a named dimension in
`spec.filters` (asserted in `test_dimension_role_owner.py`). The parse is
therefore constructed. That is the point — a surface that cannot reach a
construct is evidence about coverage before it is evidence about the product.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mi_agent import execution_receipt as R
from mi_agent.mi_query_validator import load_mi_semantics
from mi_agent import mi_calibration as CAL
from mi_agent_api import mi_service


_REPO_ROOT = Path(__file__).resolve().parents[2]
_TAPE = (_REPO_ROOT / "demo_platform" / "workspace" / "store" / "processed"
         / "platform" / "alderbridge" / "2026-06-30"
         / "platform_canonical_typed.csv")


@pytest.fixture(scope="module")
def semantics():
    return load_mi_semantics(
        _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml")


@pytest.fixture(scope="module")
def frame():
    if not _TAPE.exists():
        pytest.skip("calibration book not built")
    return CAL.default_bank_frame(_TAPE)


class _Spec:
    def __init__(self, filters=None, dimensions=None):
        self.filters = dict(filters or {})
        self.dimensions = list(dimensions or [])
        self.dimension = None


class _Parsed:
    def __init__(self, spec):
        self.spec = spec


def _routed(spec_filters=None, applied=None, answer="Regional exposure."):
    """A minimal geo_exposure envelope with a declared narrowing ledger."""
    return {
        "ok": True,
        "answer": answer,
        "artifacts": [{"type": "table", "title": "Funded exposure by ITL3 area",
                       "rows": [{"area": "Westminster", "balance": 1.0}]}],
        "spec": {"filters": dict(spec_filters or {})},
        "metadata": {"route": "geo_exposure",
                     "populationApplied": {"applied": list(applied or [])}},
    }


def _facets(envelope):
    return [(f.get("kind"), f.get("field"), f.get("status"))
            for f in ((envelope.get("executionSummary") or {}).get("facets") or [])]


QUESTION = "Show funded exposure by region where account status is active"


def test_the_owners_answer_reaches_the_routed_receipt(semantics, frame):
    """The consolidation itself. A dimension the parser positively slotted as a
    FILTER is recorded as a population, not asserted to be a breakdown.

    Before this commit the routed path had no reader that could tell the
    difference, so `account_status` reached the receipt as
    `grouping_dimension` and the routed reconciler stamped it.
    """
    out = mi_service._guard_routed_answer(
        _routed(applied=["account_status"]), question=QUESTION,
        route="geo_exposure", semantics=semantics, frame=frame,
        parsed=_Parsed(_Spec(filters={"account_status": "Active"})))
    facets = _facets(out)
    assert ("row_population", "account_status", "applied") in facets
    assert not [f for f in facets
                if f[0] == "grouping_dimension" and f[1] == "account_status"]


def test_a_population_the_owner_raises_is_STAMPED_not_left_lost(semantics, frame):
    """The e35a01b shape, on this path.

    The route's own spec carries NO filter, so the ledger raises nothing and the
    owner's population is the only one. If it were appended after reconciliation
    it would keep its LOST default, and a lost population blocks — turning a
    correct answer into a refusal.
    """
    out = mi_service._guard_routed_answer(
        _routed(spec_filters={}, applied=["account_status"]),
        question=QUESTION, route="geo_exposure", semantics=semantics,
        frame=frame,
        parsed=_Parsed(_Spec(filters={"account_status": "Active"})))
    populations = [f for f in _facets(out) if f[0] == "row_population"]
    assert populations == [("row_population", "account_status", "applied")]
    assert out.get("ok") is True
    assert not out.get("controlledRefusal")


def test_the_owner_and_the_ledger_do_not_raise_it_twice(semantics, frame):
    """The 7c46f81 shape. Both name the same governed population; the receipt
    must carry ONE claim, not one applied and one lost."""
    out = mi_service._guard_routed_answer(
        _routed(spec_filters={"account_status": "Active"},
                applied=["account_status"]),
        question=QUESTION, route="geo_exposure", semantics=semantics,
        frame=frame,
        parsed=_Parsed(_Spec(filters={"account_status": "Active"})))
    populations = [f for f in _facets(out) if f[0] == "row_population"]
    assert len(populations) == 1, populations
    assert populations[0][2] == "applied"
    assert out.get("ok") is True


def test_specs_that_disagree_still_produce_one_stamped_claim(semantics, frame):
    """The dedupe is on (kind, field, label), and the two raisers read DIFFERENT
    specs — the owner reads the parse, the ledger reads the route's spec. Where
    they carry different values for the same field the labels differ, so both
    claims stand, and the test is that neither reaches the receipt unstamped."""
    out = mi_service._guard_routed_answer(
        _routed(spec_filters={"account_status": "Redeemed"},
                applied=["account_status"]),
        question=QUESTION, route="geo_exposure", semantics=semantics,
        frame=frame,
        parsed=_Parsed(_Spec(filters={"account_status": "Active"})))
    populations = [f for f in _facets(out) if f[0] == "row_population"]
    assert populations, "the population claims vanished"
    assert all(status in ("applied", "lost", "unavailable")
               for _k, _f, status in populations)
    assert all(status != "lost" for _k, _f, status in populations), (
        "a population reached the receipt unstamped: %s" % (populations,))


def test_an_unresolved_role_leaves_the_routed_facet_alone(semantics, frame):
    """The can-fail. The routed path settles the ROLE and not its CONSEQUENCE:
    measured on the 15 corpus questions that route and diverge, asking instead
    of answering would turn ten correct answers into clarifications.

    A change that made the routed path clarify on an unresolved role would
    satisfy every test above and fail here.
    """
    out = mi_service._guard_routed_answer(
        _routed(), question="show me interesting regions",
        route="geo_exposure", semantics=semantics, frame=frame,
        parsed=_Parsed(_Spec()))
    kinds = {kind for kind, _f, _s in _facets(out)}
    assert "unresolved_role" not in kinds
    assert not out.get("clarificationRequested")
