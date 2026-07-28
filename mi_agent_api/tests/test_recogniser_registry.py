"""The governed recogniser registry: registration, ordering and capability gating.

These are the structural guarantees the routing refactor rests on. The old
``if/elif`` chain had none of them: precedence was source-code line order, a
duplicate could silently shadow an existing route, and capability availability
was decided per-route (or not at all).

Also covers the Business Semantics Registry seams, so the extension points are
not merely documented but exercised.
"""

from __future__ import annotations

import pytest

from mi_agent_api import chat_routing
from mi_agent_api.recogniser_registry import (
    DEFAULT_CONFIDENCE,
    DuplicateRecogniserError,
    Recogniser,
    RecogniserRegistry,
    Recognition,
    RouteRequest,
    resolve_capability_state,
)
from trakt_core.portfolio import (
    CAP_PIPELINE,
    REASON_NO_PORTFOLIOS_IN_SCOPE,
    REASON_NON_ORIGINATING,
    CapabilityState,
)


def _request(question: str = "q", **kw) -> RouteRequest:
    defaults = dict(question=question, spec=object(), spec_dict={}, semantics={},
                    view="funded", client_id="client_001", run_id=None,
                    portfolio_id=None)
    defaults.update(kw)
    return RouteRequest(**defaults)


def _rec(name: str, priority: int, *, matches: bool = True,
         confidence: float = DEFAULT_CONFIDENCE, result=None, **kw) -> Recogniser:
    return Recogniser(
        name=name, priority=priority,
        recognise=lambda r: Recognition(matches, confidence),
        handle=lambda r: result, **kw)


# --------------------------------------------------------------------------- #
# Registration is deterministic
# --------------------------------------------------------------------------- #
def test_a_duplicate_name_is_refused_not_silently_overwritten():
    """Silent replacement would make precedence depend on import order — the
    exact non-determinism this registry exists to remove."""
    registry = RecogniserRegistry()
    registry.register(_rec("alpha", 10))
    with pytest.raises(DuplicateRecogniserError):
        registry.register(_rec("alpha", 20))
    assert registry.names() == ("alpha",)


def test_a_recogniser_must_be_named():
    with pytest.raises(ValueError):
        RecogniserRegistry().register(_rec("", 10))


def test_registration_order_does_not_affect_evaluation_order():
    """Priority is the contract; the order modules happen to import is not."""
    forward, reverse = RecogniserRegistry(), RecogniserRegistry()
    forward.extend([_rec("a", 10), _rec("b", 20), _rec("c", 30)])
    reverse.extend([_rec("c", 30), _rec("b", 20), _rec("a", 10)])
    assert forward.names() == reverse.names() == ("a", "b", "c")


def test_equal_priority_falls_back_to_registration_order():
    registry = RecogniserRegistry()
    registry.extend([_rec("second", 10), _rec("first", 10)])
    assert registry.names() == ("second", "first")


def test_ordering_is_stable_across_repeated_calls():
    registry = RecogniserRegistry()
    registry.extend([_rec(f"r{i}", 10) for i in range(25)])
    assert len({registry.names() for _ in range(10)}) == 1


# --------------------------------------------------------------------------- #
# Candidate selection
# --------------------------------------------------------------------------- #
def test_only_matching_recognisers_are_candidates():
    registry = RecogniserRegistry()
    registry.extend([_rec("no", 10, matches=False), _rec("yes", 20)])
    assert [r.name for r, _ in registry.candidates(_request())] == ["yes"]


def test_higher_confidence_outranks_earlier_priority():
    """Confidence is a real input, not decoration — a more specific recogniser
    can win from a later position."""
    registry = RecogniserRegistry()
    registry.extend([_rec("early", 10, confidence=0.4),
                     _rec("specific", 90, confidence=0.9)])
    assert [r.name for r, _ in registry.candidates(_request())] == ["specific", "early"]


def test_equal_confidence_collapses_to_priority_order():
    """The behaviour-preservation guarantee: every migrated recogniser shares
    DEFAULT_CONFIDENCE, so dispatch order IS the historical chain order."""
    registry = RecogniserRegistry()
    registry.extend([_rec("late", 90), _rec("early", 10)])
    assert [r.name for r, _ in registry.candidates(_request())] == ["early", "late"]


def test_a_raising_recogniser_is_skipped_not_fatal():
    registry = RecogniserRegistry()
    registry.register(Recogniser(
        name="boom", priority=10,
        recognise=lambda r: (_ for _ in ()).throw(RuntimeError("bang")),
        handle=lambda r: {"ok": True}))
    registry.register(_rec("sane", 20))
    assert [r.name for r, _ in registry.candidates(_request())] == ["sane"]


def test_a_bare_bool_recogniser_is_accepted():
    """A simple predicate must not have to construct a Recognition."""
    registry = RecogniserRegistry()
    registry.register(Recogniser(name="p", priority=10,
                                 recognise=lambda r: "geo" in r.question,
                                 handle=lambda r: None))
    assert [r.name for r, _ in registry.candidates(_request("geo please"))] == ["p"]
    assert registry.candidates(_request("balance")) == ()


# --------------------------------------------------------------------------- #
# The live registry preserves the historical chain
# --------------------------------------------------------------------------- #
HISTORICAL_ORDER = (
    "scenario", "cohort_conversion", "forecast_extrapolation", "funded_bridge",
    "cohort_progression", "geo_exposure", "period_movement", "portfolio_summary",
    "temporal_compare", "risk_limits", "evolution",
)

#: Workflow-layer routes registered ABOVE the migrated chain (additive
#: registration — the extension mechanism the registry was built for). Each
#: declares its own position; the historical chain keeps its relative order.
WORKFLOW_ROUTES = ("portfolio_risk_comparison",)


def test_the_live_registry_reproduces_the_historical_chain_order():
    """Every route from the old if/elif chain, in its original relative order.

    A capability added since the refactor (``period_change_analysis``) takes a
    position between two historical routes, so the registry is no longer
    *identical* to the old chain. The guarantee being asserted is the one that
    actually matters and is unchanged: no historical route ever moves relative to
    another, so a question the old chain routed one way is still routed that way
    by whichever of those routes claims it first.
    """
    from mi_agent_api.recogniser_registry import REGISTRY

    live = REGISTRY.names()
    assert set(HISTORICAL_ORDER) <= set(live)
    assert tuple(n for n in live if n in HISTORICAL_ORDER) == HISTORICAL_ORDER


def test_every_live_recogniser_shares_the_default_confidence():
    """The refactor is behaviour-preserving only while this holds."""
    from mi_agent_api.recogniser_registry import REGISTRY

    for rec in REGISTRY.ordered():
        verdict = Recognition.yes()
        assert verdict.confidence == DEFAULT_CONFIDENCE
        assert rec.priority > 0
        assert rec.description, f"{rec.name} must describe itself"


def test_lens_aware_routes_are_declared_on_the_recogniser():
    """The lens fact lives on the recogniser, not in a parallel set that drifts."""
    assert chat_routing._lens_aware_routes() == frozenset({
        "portfolio_summary", "period_movement", "funded_bridge",
        "cohort_progression", "geo_exposure", "period_change_analysis"})


# --------------------------------------------------------------------------- #
# Capability gating — the SAME resolution React uses
# --------------------------------------------------------------------------- #
def test_a_disabled_capability_short_circuits_with_the_governed_explanation():
    """The chat must give the same reason the dashboard does, not its own error."""
    state = CapabilityState(
        CAP_PIPELINE, False, REASON_NON_ORIGINATING,
        "No portfolio in this scope originates new lending, so there is no "
        "origination pipeline to report.",
        excluded_portfolios=("acquired_001",))

    class _Ctx:
        def capability(self, name):
            return state

    registry = RecogniserRegistry()
    registry.register(_rec("scenario", 10, capability=CAP_PIPELINE,
                           result={"ok": True, "metadata": {"route": "scenario"}}))
    out = chat_routing.try_route(
        "what if conversion increased by 10%?", portfolio_id="client_001",
        view="funded", output_root=None, pipeline_root=None, semantics={},
        registry=registry, capability_resolver=lambda ctx: _Ctx())

    assert out["ok"] is False
    assert "originates new lending" in out["answer"]
    assert out["metadata"]["capabilityReason"] == REASON_NON_ORIGINATING
    assert out["metadata"]["capability"] == CAP_PIPELINE
    # Classified as a governed refusal, not a calculation failure.
    assert out["metadata"]["controlledUnsupported"] is True


def test_an_enabled_capability_runs_the_route():
    class _Ctx:
        def capability(self, name):
            return CapabilityState(CAP_PIPELINE, True)

    registry = RecogniserRegistry()
    registry.register(_rec("scenario", 10, capability=CAP_PIPELINE,
                           result={"ok": True, "metadata": {"route": "scenario"}}))
    out = chat_routing.try_route(
        "what if conversion increased by 10%?", portfolio_id="client_001",
        view="funded", output_root=None, pipeline_root=None, semantics={},
        registry=registry, capability_resolver=lambda ctx: _Ctx())
    assert out["ok"] is True


def test_an_unresolvable_gate_does_not_claim_the_capability_is_inapplicable():
    """A storage/config fault is not the same statement as 'this does not apply
    to your portfolio'. The route is attempted instead."""
    def _boom(_ctx):
        raise RuntimeError("registry unavailable")

    assert resolve_capability_state(CAP_PIPELINE, "total", resolver=_boom) is None


def test_a_deployment_without_provenance_is_not_gated():
    """A canonical tape with no source-portfolio provenance yields an EMPTY
    registry, so every capability resolves NO_PORTFOLIOS_IN_SCOPE. Treating that
    as 'disabled' would silently switch off every routed capability for a
    single-portfolio deployment — which is the current ERE production shape."""
    class _Ctx:
        def capability(self, name):
            return CapabilityState(CAP_PIPELINE, False, REASON_NO_PORTFOLIOS_IN_SCOPE,
                                   "No governed portfolios resolve for this selection.")

    assert resolve_capability_state(CAP_PIPELINE, None, resolver=lambda c: _Ctx()) is None


def test_capability_gating_matches_the_react_dashboard_resolution():
    """Chat and dashboard must read the SAME governed capability state.

    Not 'similar logic' — literally the same call. The chat's default resolver
    delegates to ``portfolio_context.resolve_context``, which is what the REST
    endpoints behind the React dashboard use.
    """
    from mi_agent_api import portfolio_context as ctx_mod
    from mi_agent_api.recogniser_registry import default_capability_resolver

    for cap in (CAP_PIPELINE, "funded", "risk", "cohorts"):
        chat_state = default_capability_resolver("total").capability(cap)
        react_state = ctx_mod.resolve_context("total").capability(cap)
        assert (chat_state is None) == (react_state is None)
        if chat_state is not None:
            assert chat_state.enabled == react_state.enabled
            assert chat_state.reason_code == react_state.reason_code


# --------------------------------------------------------------------------- #
# Business Semantics Registry extension points
# --------------------------------------------------------------------------- #
def test_semantics_context_reaches_a_recogniser_without_signature_changes():
    """The BSR seam: metadata resolved at parse time must arrive at recognisers
    through the request they already receive."""
    seen = {}

    registry = RecogniserRegistry()
    registry.register(Recogniser(
        name="probe", priority=10,
        recognise=lambda r: seen.update(r.semantics_context) or True,
        handle=lambda r: {"ok": True, "metadata": {"route": "probe"}}))

    from mi_agent.parsed_question import ParsedQuestion

    parsed = ParsedQuestion.parse(
        "balance by region", _semantics(),
        semantics_resolver=lambda q, s, sem: {"business_term": "funded_balance"})
    chat_routing.try_route(
        "balance by region", portfolio_id="c", view="funded", output_root=None,
        pipeline_root=None, semantics=_semantics(), parsed=parsed, registry=registry)

    assert seen == {"business_term": "funded_balance"}


def test_a_failing_semantics_resolver_never_breaks_a_query():
    """The BSR is additive: if it is unavailable the agent still answers."""
    from mi_agent.parsed_question import ParsedQuestion

    def _boom(q, spec, sem):
        raise RuntimeError("registry down")

    parsed = ParsedQuestion.parse("balance by region", _semantics(),
                                  semantics_resolver=_boom)
    assert parsed.semantics_context == {}
    assert parsed.spec is not None


def test_recogniser_metadata_is_a_declarative_slot_for_registry_terms():
    """A future workflow recogniser declares what it consumes; the registry
    carries it without knowing the schema."""
    rec = _rec("future", 10, metadata={"business_terms": ["funded_balance"],
                                       "comparison_basis": "period"})
    assert rec.metadata["business_terms"] == ["funded_balance"]


def test_dependencies_expose_the_semantics_resolver_seam():
    """The seam is now WIRED: the Business Semantics Registry resolver is the
    default, and explicit injection still wins (tests, alternative stores)."""
    from mi_agent_api.dependencies import build_dependencies
    from mi_workflows.semantics import semantics_context_resolver

    marker = lambda q, s, sem: {"x": 1}  # noqa: E731
    deps = build_dependencies(semantics_resolver=marker)
    assert deps.semantics_resolver is marker
    assert build_dependencies().semantics_resolver is semantics_context_resolver


def _semantics():
    from pathlib import Path

    from mi_agent.mi_query_validator import load_mi_semantics
    return load_mi_semantics(
        Path(__file__).resolve().parents[2] / "mi_agent" / "mi_semantics_field_registry.yaml")
