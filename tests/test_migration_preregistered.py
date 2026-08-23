"""Pre-registered migration expectations — DECLARED FAILING.

Every test in this module asserts behaviour the product does NOT have today.
They are committed BEFORE the work so that no expectation can be shaped by what
the implementation turns out to do — the same discipline
``clause_splitting_phase1`` used for its probes.

All are ``xfail(strict=True)``. That is deliberate and load-bearing:

* the estate stays green, so these do not block unrelated work;
* the expectation is recorded in executable form rather than in prose;
* **if one starts passing, the suite FAILS.** A capability cannot arrive
  silently. Whoever makes it pass must come here, delete the marker, and say in
  the commit which authorised migration step did it.

Two groups:

``TestT3Acceptance``
    The byte-identical bar the ``evolution`` conversion (Phase 5) must clear.
    The migration currently has no bar for T3; this is it. Landing T3 before
    Phase 5 was ruled against (docs/mi_t3_now_versus_migration.md) — these tests
    do not authorise it, they define what "done" means when it is authorised.

``TestArityIndependentDisclosure``
    The governance prerequisite from Objective 3A. The thin-sample and
    denominator disclosures are guarded by ``len(group_cols) == 1``
    (mi_agent/mi_query_executor.py::_execute_grouped), so a two-dimension
    grouped answer omits both. This is a LIVE defect on shipped shapes, recorded
    as known-open in the Phase 0 baseline. No policy VALUE is asserted here —
    ``LOW_GROUP_COUNT`` is read from the module, never restated.
"""
from __future__ import annotations

import os
import warnings

import pytest

pytestmark = pytest.mark.usefixtures("_migration_book")

_BALANCE = "current_outstanding_balance"


@pytest.fixture(scope="module")
def _migration_book():
    """The governed demo book, in the mode the measurement surfaces use."""
    warnings.simplefilter("ignore")
    os.environ.setdefault("TRAKT_RUNTIME_MODE", "development")
    from demo_platform import config as cfg
    before = dict(os.environ)
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    try:
        yield cfg.CLIENT_ID
    finally:
        os.environ.clear()
        os.environ.update(before)


def _ask(question: str):
    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query
    from trakt_core.context import ExecutionContext
    ctx = ExecutionContext.for_internal(os.environ["MI_AGENT_CLIENT_ID"])
    return execute_governed_mi_query(MiQueryRequest(question=question), ctx).result or {}


def _rows(result):
    for artifact in result.get("artifacts") or []:
        rows = artifact.get("rows") or []
        if rows:
            return rows
    return []


def _distinct(rows, predicate):
    return {str(r[k]) for r in rows for k in r if predicate(k) and r.get(k) is not None}


class TestT3Acceptance:
    """The bar the `evolution` conversion must clear. Not authorisation to build it."""

    @pytest.mark.xfail(strict=True, reason="T3 is refused today; Phase 5 bar")
    def test_t3_artifact_carries_both_a_period_axis_and_a_region_breakdown(self):
        result = _ask("balance over time by region")
        assert result.get("ok") is True, result.get("answer")
        rows = _rows(result)
        periods = _distinct(rows, lambda k: "period" in k.lower())
        regions = _distinct(rows, lambda k: "geog" in k.lower() or "region" in k.lower())
        # The rule the time-series surface enforces: a whole-book series returned
        # for a segmented request is ABSENT, not partial. Both limbs or nothing.
        assert len(periods) > 1, f"no time axis: {periods}"
        assert len(regions) > 1, f"no region breakdown: {regions}"

    @pytest.mark.xfail(strict=True, reason="T3 is refused today; Phase 5 bar")
    def test_t3_reconciles_to_the_shipped_t1_series_period_by_period(self):
        """The migration bar is byte-identical economics, not a plausible shape."""
        from collections import defaultdict

        t3 = _rows(_ask("balance over time by region"))
        t1 = _rows(_ask("balance over time"))
        assert t1 and t3

        def value_of(row):
            for key in row:
                if "balance" in key.lower() or key.lower() == "value":
                    return float(row[key] or 0.0)
            raise AssertionError(f"no measure column in {sorted(row)}")

        def period_of(row):
            for key in row:
                if "period" in key.lower():
                    return str(row[key])
            raise AssertionError(f"no period column in {sorted(row)}")

        composed = defaultdict(float)
        for row in t3:
            composed[period_of(row)] += value_of(row)
        shipped = {period_of(r): value_of(r) for r in t1}
        assert set(composed) == set(shipped)
        for period, total in shipped.items():
            assert abs(composed[period] - total) < 0.005, (
                f"{period}: composed {composed[period]:,.2f} != shipped {total:,.2f}")

    @pytest.mark.xfail(strict=True, reason="T3 is refused today; Phase 5 bar")
    def test_t3_declares_the_dimension_it_grouped_by(self):
        """A route that declares nothing proves nothing — `grouping_proven`'s bar."""
        from mi_agent import execution_receipt as receipt

        result = _ask("balance over time by region")
        declared = receipt.declared_group_fields(result, result.get("route"))
        assert declared, "the answer declares no grouping axis"
        from mi_agent.mi_query_validator import load_mi_semantics
        from mi_agent_api.data_source import semantics_path
        semantics = load_mi_semantics(semantics_path())
        requested = receipt.requested_dimension_terms("balance over time by region",
                                                      semantics)
        assert requested
        key, term, alts = requested[0]
        facet = receipt.RequestedFacet(kind=receipt.KIND_GROUPING, label=term,
                                       field_key=key, alt_keys=alts)
        assert receipt.grouping_proven(facet, declared,
                                       semantics.get("fields", {})), (
            f"declared {sorted(declared)} does not satisfy {facet.satisfied_by()}")


class TestArityIndependentDisclosure:
    """Objective 3A. The policy VALUE is never asserted — only that it applies."""

    def _grouped(self, question):
        result = _ask(question)
        assert result.get("ok") is True, result.get("answer")
        return result, _rows(result)

    def test_arity_one_attaches_the_denominator_today(self):
        """The control. If this ever fails, the baseline moved, not the target."""
        _result, rows = self._grouped("Show me balance by LTV band")
        assert rows and "loan_count" in rows[0], sorted(rows[0])

    @pytest.mark.xfail(strict=True,
                       reason="len(group_cols) == 1 guard; known-open baseline defect")
    def test_arity_two_attaches_the_denominator(self):
        _result, rows = self._grouped("Show me balance by LTV band and ticket size")
        assert rows and "loan_count" in rows[0], sorted(rows[0])

    @pytest.mark.xfail(strict=True,
                       reason="len(group_cols) == 1 guard; known-open baseline defect")
    def test_arity_two_discloses_thin_leaf_groups(self):
        """Same governed policy, applied to every leaf group at any arity."""
        from mi_agent.mi_query_executor import LOW_GROUP_COUNT

        result, rows = self._grouped(
            "What is the average borrower age by region and LTV band?")
        thin = [r for r in rows if int(r.get("loan_count") or 0) < LOW_GROUP_COUNT]
        assert thin, "fixture no longer has a thin leaf group at arity 2"
        warnings_out = list(result.get("warnings") or [])
        warnings_out += list((result.get("metadata") or {}).get("warnings") or [])
        assert any("thin sample" in str(w) for w in warnings_out), (
            f"{len(thin)} thin leaf group(s) and no disclosure: {warnings_out}")

    def test_arity_one_discloses_thin_groups_today(self):
        """The control for the test above."""
        result, _rows_ = self._grouped("What is the average borrower age by region?")
        warnings_out = list(result.get("warnings") or [])
        warnings_out += list((result.get("metadata") or {}).get("warnings") or [])
        assert any("thin sample" in str(w) for w in warnings_out), warnings_out
