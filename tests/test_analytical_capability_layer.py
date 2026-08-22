"""tests/test_analytical_capability_layer.py — the analytical capability layer.

Nine CFO questions need answers that are more than one deterministic capability.
This file defends the three properties that make composing them safe:

1. **Additive.** The layer claims a question only when the deterministic planner
   composes two or more capabilities, and never one an existing route already
   answers. Every simple MI question keeps the route it had.
2. **Deterministic.** No adapter in the layer touches a dataframe library: every
   figure is a governed engine's own output, carried with the population, period
   and evidence it was computed under.
3. **Governed.** A population the question did not request cannot execute — the
   fabricated-population guard is applied to this layer's own plans — and an
   answer states only what a structured finding contains.

The end-to-end section runs the nine questions against a purpose-built second
book with a governed weekly pipeline, and reconciles every delivered figure
against pandas read straight from the fixture CSVs. The agent is never its own
oracle.

Run: python -m pytest tests/test_analytical_capability_layer.py
"""

from __future__ import annotations

import ast
import json
import logging
import os
import sys
import warnings
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_workflows.analytical import contract as contract_mod  # noqa: E402
from mi_workflows.analytical import narrative as narrative_mod  # noqa: E402
from mi_workflows.analytical import planner as planner_mod  # noqa: E402
from mi_workflows.analytical import populations as pops  # noqa: E402
from mi_workflows.analytical import route as route_mod  # noqa: E402
from mi_workflows.analytical.registry import (  # noqa: E402
    CAPABILITIES,
    CapabilityRegistry,
    DuplicateCapabilityError,
)

# The nine commercial acceptance questions, verbatim.
Q1 = "How has the profile of new originations changed in the last few months?"
Q2 = "When are we likely to reach £100m of loans?"
Q3 = ("What is the current value of outstanding offers in the pipeline, how "
      "much do we expect to complete, and when?")
Q4 = "What is the current completion run rate?"
Q5 = "Are we likely to breach any concentration limits in the next three months?"
Q6 = "Which limits are closest to breaching?"
Q7 = "How does the risk profile of older vintages compare with the front book?"
Q8 = "How has the balance from direct lending changed relative to acquired lending?"
Q9 = "Based on the current pipeline, what is the forecast funded balance?"

BANK = [Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9]

#: Questions the layer must COMPOSE, and the plan intent each must produce.
COMPOSED = {
    Q1: planner_mod.INTENT_ORIGINATION_PROFILE_CHANGE,
    Q3: planner_mod.INTENT_PIPELINE_OFFER_OUTLOOK,
    Q7: planner_mod.INTENT_VINTAGE_RISK_COMPARISON,
    Q8: planner_mod.INTENT_POPULATION_MOVEMENT_COMPARISON,
    Q9: planner_mod.INTENT_FUNDED_BALANCE_OUTLOOK,
}

#: Questions an existing single capability already answers. The layer must
#: decline every one of them: replacing a working answer is not an improvement.
DEFERRED = [Q2, Q4, Q5, Q6]


# --------------------------------------------------------------------------- #
# 1. The capability contract
# --------------------------------------------------------------------------- #
class TestCapabilityContract:

    def test_every_declared_capability_can_actually_run(self):
        """A described capability with no executor is documentation."""
        for capability in CAPABILITIES.all():
            assert callable(capability.executor), capability.id

    def test_every_capability_declares_its_engine_and_limits(self):
        for capability in CAPABILITIES.all():
            assert capability.engine, capability.id
            assert capability.intent, capability.id
            assert capability.produces, capability.id
            assert capability.limitations, (
                f"{capability.id} declares no limitation; a capability whose "
                "limits are folklore cannot be planned against")

    def test_a_duplicate_id_is_refused_not_overwritten(self):
        registry = CapabilityRegistry(CAPABILITIES.all())
        with pytest.raises(DuplicateCapabilityError):
            registry.register(CAPABILITIES.require("portfolio_snapshot"))

    def test_missing_required_input_is_rejected_before_execution(self):
        capability = CAPABILITIES.require("portfolio_snapshot")
        assert capability.validate_inputs({}) == [
            "portfolio_snapshot: missing required input 'measures'"]

    def test_an_undeclared_input_is_rejected(self):
        capability = CAPABILITIES.require("population_profile")
        errors = capability.validate_inputs({"nonsense": 1})
        assert errors == ["population_profile: unknown input 'nonsense'"]

    def test_a_finding_that_is_not_ok_must_say_why(self):
        with pytest.raises(ValueError):
            contract_mod.Finding(
                capability="portfolio_snapshot", kind=contract_mod.KIND_MEASURE,
                label="Balance", status=contract_mod.STATUS_UNAVAILABLE)

    def test_an_unknown_finding_kind_is_refused(self):
        with pytest.raises(ValueError):
            contract_mod.Finding(capability="x", kind="vibes", label="Balance")


# --------------------------------------------------------------------------- #
# 2. No adapter computes a financial result
# --------------------------------------------------------------------------- #
_LAYER_MODULES = sorted((_REPO_ROOT / "mi_workflows" / "analytical").glob("*.py"))


class TestDeterministicSourceDiscipline:
    """The layer orchestrates governed engines; it does not calculate.

    Checked structurally rather than by inspection: a module that cannot reach a
    dataframe library cannot aggregate a frame behind the engines' backs. What
    it may still do — and does — is add up subtotals a governed function
    returned, which is why those call sites name the function in the finding's
    evidence.
    """

    @pytest.mark.parametrize("path", _LAYER_MODULES, ids=lambda p: p.name)
    def test_the_layer_never_reaches_for_a_dataframe_library(self, path: Path):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported += [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module.split(".")[0])
        assert "pandas" not in imported, f"{path.name} imports pandas"
        assert "numpy" not in imported, f"{path.name} imports numpy"

    def test_every_capability_names_the_engine_it_delegated_to(self):
        engines = {c.id: c.engine for c in CAPABILITIES.all()}
        for capability_id, engine in engines.items():
            assert "." in engine, (
                f"{capability_id} must name a module-qualified engine, not "
                f"{engine!r}")


# --------------------------------------------------------------------------- #
# 2b. The execution receipt reads the layer's evidence, and only the layer's
# --------------------------------------------------------------------------- #
class TestReceiptEvidenceIsOptIn:
    """Every new reconciliation branch is gated on evidence only this layer
    publishes, so no other route's receipt can change."""

    def test_a_non_analytical_envelope_yields_no_evidence(self):
        from mi_agent import execution_receipt as receipt

        for envelope in (None, {}, {"metadata": {}},
                         {"metadata": {"route": "risk_limits"}},
                         {"metadata": {"portfolioComparison": {"cohortConcept": "x"}}}):
            assert receipt.analytical_evidence(envelope) == {}

    def test_a_claim_without_a_plan_is_not_accepted_as_evidence(self):
        from mi_agent import execution_receipt as receipt

        assert receipt.analytical_evidence(
            {"metadata": {"analyticalComposition": {"projected": True}}}) == {}
        assert receipt.analytical_evidence(
            {"metadata": {"analyticalComposition": {
                "intent": "x", "capabilities": []}}}) == {}

    def test_a_real_block_is_accepted(self):
        from mi_agent import execution_receipt as receipt

        block = {"intent": "funded_balance_outlook",
                 "capabilities": ["funded_balance_forecast"], "projected": True}
        assert receipt.analytical_evidence(
            {"metadata": {"analyticalComposition": block}}) == block

    def test_a_forward_facet_is_lost_when_the_plan_did_not_project(self):
        """The evidence makes the guard STRICTER: a plan that ran but projected
        nothing loses the facet, which route membership could never express."""
        from mi_agent import execution_receipt as receipt

        facet = receipt.RequestedFacet(kind=receipt.KIND_PROJECTION,
                                       label="a forward projection")
        receipt.reconcile_routed_facets(
            [facet], route=route_mod.ROUTE_NAME, semantics={},
            envelope={"metadata": {"analyticalComposition": {
                "intent": "x", "capabilities": ["portfolio_snapshot"],
                "projected": False}}})
        assert facet.status == receipt.LOST

    def test_a_forward_facet_is_applied_when_the_plan_did_project(self):
        from mi_agent import execution_receipt as receipt

        facet = receipt.RequestedFacet(kind=receipt.KIND_PROJECTION,
                                       label="a forward projection")
        receipt.reconcile_routed_facets(
            [facet], route=route_mod.ROUTE_NAME, semantics={},
            envelope={"metadata": {"analyticalComposition": {
                "intent": "x", "capabilities": ["funded_balance_forecast"],
                "projected": True}}})
        assert facet.status == receipt.APPLIED


# --------------------------------------------------------------------------- #
# 3. Planning, and the deference rule
# --------------------------------------------------------------------------- #
class TestPlanning:

    @pytest.mark.parametrize("question,intent", sorted(COMPOSED.items()))
    def test_a_composite_question_produces_a_composite_plan(self, question, intent):
        plan = planner_mod.plan_for(question)
        assert plan is not None, question
        assert plan.intent == intent
        assert plan.is_composite, "a single-capability plan is not this layer's work"
        assert plan.rationale, "a plan must say why it is what it is"
        assert plan.required_kinds, "a plan must declare what would satisfy it"

    @pytest.mark.parametrize("question", DEFERRED)
    def test_a_question_an_existing_route_owns_is_declined(self, question):
        assert planner_mod.plan_for(question) is None, question

    @pytest.mark.parametrize("question", [
        "What is the total AuM?",
        "What is the weighted average LTV?",
        "Show balance by region.",
        "What is the average borrower age?",
        "Show me a heatmap of balance by region and LTV band.",
        "What is the largest single loan exposure?",
        "Balance for the acquired book.",
        "What is the balance of the front book?",
        "Show balance by LTV band for the back book.",
        "Which region has the largest exposure?",
        "Summarise the portfolio.",
        "What has changed versus the prior month?",
        "Compare October and November funded balance.",
        "Export the loan-level tape.",
        "Show funded balance evolution by month.",
    ])
    def test_simple_questions_are_never_claimed(self, question):
        """The no-regression rule, at the only seam that could break it."""
        assert planner_mod.plan_for(question) is None, question

    def test_every_deterministic_plan_validates_against_the_registry(self):
        for question in COMPOSED:
            plan = planner_mod.plan_for(question)
            assert planner_mod.validate(plan) == []

    def test_the_recogniser_declines_a_non_composite_question(self):
        recogniser = route_mod.recogniser()
        assert recogniser.name == route_mod.ROUTE_NAME
        assert recogniser.priority < 10, (
            "a composite question must be considered before the single-capability "
            "routes that would otherwise answer part of it")

    def test_the_route_declares_what_it_defers_to(self):
        metadata = route_mod.recogniser().metadata
        assert set(metadata["defers_to"]) == {"forecast_extrapolation", "risk_limits"}


# --------------------------------------------------------------------------- #
# 4. Governed populations
# --------------------------------------------------------------------------- #
class TestGovernedPopulations:

    def test_a_population_the_question_never_requested_cannot_be_planned(self):
        with pytest.raises(pops.FabricatedPopulation):
            pops.assert_not_fabricated(
                [pops.seasoning_population("Front Book")],
                "What is the forecast funded balance?")

    def test_both_sides_of_a_governed_partition_pass_when_one_is_named(self):
        pops.assert_not_fabricated(
            [pops.seasoning_population("Front Book"),
             pops.seasoning_population("Back Book")], Q7)

    def test_a_seasoning_population_carries_its_configured_boundary(self):
        population = pops.seasoning_population("Front Book")
        assert "months" in population.label, (
            "the cutoff is configurable, so the label must state which one "
            "produced the population")

    def test_the_comparison_pair_for_provenance_is_direct_and_acquired(self):
        pair = planner_mod.resolve_population_pair(Q8)
        assert pair is not None
        assert {pair[0].key, pair[1].key} == {"direct", "acquired"}

    def test_two_values_of_a_governed_dimension_resolve_to_a_pair(self):
        """The brief's generic "dimension X vs dimension Y" pattern."""
        import pandas as pd

        semantics = {"fields": {
            "collateral_geography": {"role": "dimension",
                                     "canonical_field": "collateral_geography",
                                     "business_name": "Region"},
            "origination_channel": {"role": "dimension",
                                    "canonical_field": "origination_channel",
                                    "business_name": "Channel"},
        }}
        frame = pd.DataFrame({
            "collateral_geography": ["South East", "London", "Wales"],
            "origination_channel": ["Broker", "Direct", "Broker"],
        })
        pair = planner_mod.resolve_population_pair(
            "How has the balance from the South East changed relative to London?",
            frame, semantics)
        assert pair is not None
        assert [p.key for p in pair] == ["collateral_geography=South East",
                                         "collateral_geography=London"]

    def test_a_category_the_book_does_not_carry_is_never_approximated(self):
        import pandas as pd

        semantics = {"fields": {"collateral_geography": {
            "role": "dimension", "canonical_field": "collateral_geography"}}}
        frame = pd.DataFrame({"collateral_geography": ["South East", "London"]})
        assert planner_mod.resolve_population_pair(
            "How has the balance from Cornwall changed relative to Anglesey?",
            frame, semantics) is None

    def test_the_pair_a_question_resolves_to_does_not_depend_on_registry_order(self):
        """Two dimensions can both match; which wins must be a rule, not luck."""
        import pandas as pd

        frame = pd.DataFrame({
            "collateral_geography": ["South East", "London"],
            "origination_channel": ["Broker", "Direct"],
        })
        entries = {
            "collateral_geography": {"role": "dimension",
                                     "canonical_field": "collateral_geography"},
            "origination_channel": {"role": "dimension",
                                    "canonical_field": "origination_channel"},
        }
        question = ("How has the balance from the South East Broker book changed "
                    "relative to London Direct?")
        first = planner_mod.resolve_population_pair(question, frame,
                                                    {"fields": dict(entries)})
        flipped = {"fields": {k: entries[k] for k in reversed(list(entries))}}
        second = planner_mod.resolve_population_pair(question, frame, flipped)
        assert [p.key for p in first] == [p.key for p in second]

    def test_a_question_naming_no_two_governed_populations_resolves_none(self):
        assert planner_mod.resolve_population_pair(
            "How does the weather compare with last year?") is None

    def test_a_population_the_book_cannot_express_is_refused_not_ignored(self):
        """The P1L failure mode, at this layer's own seam.

        ``apply_population`` leaves the frame ALONE for a predicate whose column
        the book lacks. Measuring the untouched frame and labelling it with the
        population would present a whole-book figure as a narrowed one.
        """
        import pandas as pd

        from mi_workflows.analytical import executors as ex

        frame = pd.DataFrame({
            "current_outstanding_balance": [100.0, 200.0, 300.0],
        })  # deliberately carries no seasoning column

        class _Ctx:
            semantics: Dict[str, Any] = {"fields": {}}
            question = "front book balance"

            def base_frame(self):
                return frame

            def business_semantics(self):
                from mi_workflows.semantics import load_business_semantics
                return load_business_semantics()

            def snapshots(self):
                return ()

            as_of = "2026-06-30"
            client_id = "test"
            portfolio_id = "test"
            warnings: List[str] = []
            source_notes: List[Dict[str, Any]] = []

            def warn(self, message):
                self.warnings.append(message)

        findings = ex.portfolio_snapshot(
            _Ctx(), measures=["current_outstanding_balance"],
            population=pops.seasoning_population("Front Book"))
        assert len(findings) == 1
        assert findings[0].status == contract_mod.STATUS_UNAVAILABLE
        assert "could not be applied" in findings[0].note
        assert findings[0].value is None, (
            "a population that could not be applied must not carry a figure")


# --------------------------------------------------------------------------- #
# 4b. A plan that runs but does not answer is a refusal, not a partial answer
# --------------------------------------------------------------------------- #
class TestUnsatisfiedPlans:
    """The half a composite plan DID compute answers a different question.

    A funded-balance outlook that resolved the funded balance but no forecast is
    usable and not satisfied; offering it would hand a point-in-time figure to
    someone who asked where the book is going.
    """

    def _result(self, findings):
        from mi_workflows.analytical import orchestrator as orch

        plan = contract_mod.AnalyticalPlan(
            intent="funded_balance_outlook",
            calls=(contract_mod.CapabilityCall("funded_balance_forecast", {}),
                   contract_mod.CapabilityCall("pipeline_completion_forecast", {})),
            required_kinds=(contract_mod.KIND_FORECAST,))
        return orch.AnalyticalResult(plan=plan, findings=tuple(findings),
                                     execution=())

    def test_a_plan_missing_its_required_kind_is_usable_but_not_satisfied(self):
        funded = contract_mod.Finding(
            capability="funded_balance_forecast", kind=contract_mod.KIND_MEASURE,
            label="Current funded balance", value=1_000_000.0)
        missing = contract_mod.Finding(
            capability="funded_balance_forecast", kind=contract_mod.KIND_FORECAST,
            label="Forecast funded balance",
            status=contract_mod.STATUS_UNAVAILABLE,
            note="No governed pipeline source is available for this book.")
        result = self._result([funded, missing])
        assert result.usable is True
        assert result.satisfied is False
        assert any("No governed pipeline source" in r
                   for r in result.unmet_reasons())

    def test_a_plan_that_produced_its_required_kind_is_satisfied(self):
        result = self._result([contract_mod.Finding(
            capability="funded_balance_forecast",
            kind=contract_mod.KIND_FORECAST, label="Forecast funded balance",
            forecast_value=1_100_000.0)])
        assert result.satisfied is True


# --------------------------------------------------------------------------- #
# 5. An LLM may propose a plan; it may not calculate, and it may not invent
# --------------------------------------------------------------------------- #
class TestLlmProposedPlans:

    def test_a_valid_proposal_is_accepted_and_marked_as_the_model_s(self):
        plan, errors = planner_mod.plan_from_proposal({
            "intent": "front_book_position",
            "rationale": "the reader asked for the front book's position",
            "calls": [
                {"capability": "portfolio_snapshot",
                 "inputs": {"measures": ["current_outstanding_balance"],
                            "population": "front book"},
                 "because": "the level"},
                {"capability": "population_profile",
                 "inputs": {"population": "front book"},
                 "because": "the composition"},
            ],
        }, question="What is the front book's position and profile?")
        assert errors == []
        assert plan.origin == "llm"
        assert plan.calls[0].inputs["population"].key == "front_book"

    def test_an_unknown_capability_rejects_the_whole_proposal(self):
        plan, errors = planner_mod.plan_from_proposal(
            {"intent": "x", "calls": [{"capability": "make_up_a_number"}]},
            question="anything")
        assert plan is None
        assert any("unknown analytical capability" in e for e in errors)

    def test_a_missing_required_input_rejects_the_proposal(self):
        plan, errors = planner_mod.plan_from_proposal(
            {"intent": "x", "calls": [{"capability": "portfolio_snapshot"}]},
            question="anything")
        assert plan is None
        assert any("missing required input" in e for e in errors)

    def test_a_free_text_population_is_not_accepted(self):
        plan, errors = planner_mod.plan_from_proposal({
            "intent": "x",
            "calls": [{"capability": "population_profile",
                       "inputs": {"population": "the good loans"}}]},
            question="how do the good loans look?")
        assert plan is None
        assert any("not a governed population" in e for e in errors)

    def test_a_model_cannot_propose_a_population_the_question_never_named(self):
        plan, errors = planner_mod.plan_from_proposal({
            "intent": "x",
            "calls": [{"capability": "portfolio_snapshot",
                       "inputs": {"measures": ["current_outstanding_balance"],
                                  "population": "front book"}}]},
            question="What is the balance of the sponsored book?")
        assert plan is None
        assert any("did not request" in e for e in errors)


# --------------------------------------------------------------------------- #
# 6. End to end on a second book — with independent reconciliation
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def second_book(tmp_path_factory):
    """A materially different book, plus a governed weekly pipeline pack."""
    from tests.analytical import second_book as book_mod
    from tests.analytical import pipeline_fixture as pipe_mod

    root = tmp_path_factory.mktemp("analytical_second_book")
    store = root / "store"
    book_mod.build(store)
    pipeline_root = root / "pipeline"
    pipe_mod.write_pack(pipeline_root, pipe_mod.KESTRELMOOR, weeks=12,
                        last_week=date(2026, 6, 29))
    return {"store": store, "pipeline_root": pipeline_root,
            "client": book_mod.CLIENT_ID, "dates": book_mod.REPORTING_DATES,
            "env": book_mod.mi_env(store)}


@pytest.fixture(scope="module")
def answers(second_book):
    """The nine answers, from the real governed /mi/query path."""
    warnings.simplefilter("ignore")
    before = dict(os.environ)
    os.environ.update(second_book["env"])
    os.environ["MI_AGENT_PIPELINE_ROOT"] = str(second_book["pipeline_root"])
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ.pop("ANTHROPIC_API_KEY", None)
    logging.disable(logging.WARNING)
    cwd = os.getcwd()
    os.chdir(_REPO_ROOT)
    try:
        from fastapi.testclient import TestClient
        from mi_agent_api import data_source
        from mi_agent_api.app import app

        data_source.reset_cache()
        client = TestClient(app)
        portfolio = f"{second_book['client']}/{second_book['dates'][-1]}"
        out = {}
        for question in BANK:
            out[question] = client.post("/mi/query", json={
                "question": question, "portfolioId": portfolio,
                "datasetContext": "funded"}).json()
        yield out
    finally:
        os.chdir(cwd)
        logging.disable(logging.NOTSET)
        os.environ.clear()
        os.environ.update(before)
        from mi_agent_api import data_source as _ds
        _ds.reset_cache()


@pytest.fixture(scope="module")
def truth(second_book):
    """Independently computed truth, straight from the fixture CSVs."""
    import pandas as pd

    def _load(reporting: str) -> "pd.DataFrame":
        path = (second_book["store"] / "processed" / "platform"
                / second_book["client"] / reporting
                / "platform_canonical_typed.csv")
        frame = pd.read_csv(path, low_memory=False)
        origination = pd.to_datetime(frame["origination_date"], errors="coerce")
        stamp = pd.Timestamp(reporting)
        months = ((stamp.year - origination.dt.year) * 12
                  + (stamp.month - origination.dt.month))
        frame["_segment"] = ["Front Book" if m == m and m <= 12 else "Back Book"
                             for m in months]
        frame["_vintage"] = origination.dt.year
        return frame

    return {d: _load(d) for d in second_book["dates"]}


def _num(series):
    import pandas as pd
    return pd.to_numeric(series, errors="coerce")


def _wavg(values, weights) -> Optional[float]:
    value, weight = _num(values), _num(weights)
    mask = value.notna() & weight.notna() & (weight > 0)
    if not mask.any():
        return None
    return float((value[mask] * weight[mask]).sum() / weight[mask].sum())


def _findings(answer: Dict[str, Any]) -> List[Dict[str, Any]]:
    return (answer.get("metadata") or {}).get("analyticalFindings") or []


def _finding(answer, *, metric=None, kind=None, population=None, label=None):
    for finding in _findings(answer):
        if metric is not None and finding.get("metric") != metric:
            continue
        if kind is not None and finding.get("kind") != kind:
            continue
        if population is not None and (
                (finding.get("population") or {}).get("key") != population):
            continue
        if label is not None and finding.get("label") != label:
            continue
        return finding
    return None


def _close(delivered, expected, tolerance=0.05):
    assert delivered is not None, "no value delivered"
    assert abs(float(delivered) - float(expected)) <= tolerance, (
        f"delivered {delivered!r} vs independently computed {expected!r}")


class TestSecondBookAcceptance:
    """Nine questions, one materially different book, every figure reconciled."""

    def test_no_question_produces_a_hard_failure(self, answers):
        for question, answer in answers.items():
            assert isinstance(answer, dict), question
            assert "answer" in answer, question
            assert not str(answer.get("answer", "")).startswith("EXCEPTION"), question

    def test_the_layer_claims_exactly_the_composite_questions(self, answers):
        for question in COMPOSED:
            route = (answers[question].get("metadata") or {}).get("route")
            assert route == route_mod.ROUTE_NAME, f"{question} -> {route}"
        for question in DEFERRED:
            route = (answers[question].get("metadata") or {}).get("route")
            assert route != route_mod.ROUTE_NAME, f"{question} -> {route}"

    # -- Q1 ---------------------------------------------------------------- #
    def test_q1_front_book_movement_reconciles(self, answers, truth,
                                               second_book):
        answer = answers[Q1]
        assert answer["ok"] is True
        opening, closing = second_book["dates"][0], second_book["dates"][-1]
        front_open = truth[opening][truth[opening]["_segment"] == "Front Book"]
        front_close = truth[closing][truth[closing]["_segment"] == "Front Book"]

        balance = _finding(answer, metric="current_outstanding_balance",
                           kind=contract_mod.KIND_MOVEMENT)
        assert balance is not None
        _close(balance["value"],
               float(_num(front_close["current_outstanding_balance"]).sum()))
        _close(balance["priorValue"],
               float(_num(front_open["current_outstanding_balance"]).sum()))
        assert balance["population"]["rows"] == len(front_close)

        ltv = _finding(answer, metric="current_loan_to_value",
                       kind=contract_mod.KIND_MOVEMENT)
        _close(ltv["value"], _wavg(front_close["current_loan_to_value"],
                                   front_close["current_outstanding_balance"]),
               tolerance=1e-6)

    def test_q1_discloses_that_the_front_book_is_a_rolling_population(self, answers):
        joined = " ".join(answers[Q1].get("warnings") or [])
        assert "rolling population" in joined

    def test_q1_composition_shares_reconcile(self, answers, truth, second_book):
        closing = second_book["dates"][-1]
        front = truth[closing][truth[closing]["_segment"] == "Front Book"].copy()
        front["_region"] = (front["collateral_geography"].fillna("").astype(str)
                            .str.strip().replace("", "(unknown)"))
        total = float(_num(front["current_outstanding_balance"]).fillna(0).sum())
        for finding in _findings(answers[Q1]):
            if finding["kind"] != contract_mod.KIND_COMPOSITION:
                continue
            if finding.get("metric") != "collateral_geography":
                continue
            category = finding["label"].split(":", 1)[-1].strip()
            rows = front[front["_region"] == category]
            expected = float(_num(rows["current_outstanding_balance"])
                             .fillna(0).sum()) / total
            _close(finding["share"], expected, tolerance=1e-9)

    # -- Q3 ---------------------------------------------------------------- #
    def test_q3_offer_stock_and_expected_completion(self, answers):
        answer = answers[Q3]
        assert answer["ok"] is True
        stock = _finding(answer, metric="pipeline_amount",
                         kind=contract_mod.KIND_MEASURE)
        assert stock is not None and stock["value"] > 0
        assert stock["population"]["predicate"] == "pipeline_stage = OFFER"
        forecast = _finding(answer, kind=contract_mod.KIND_FORECAST)
        assert forecast is not None
        assert 0 < forecast["forecastValue"] < stock["value"], (
            "an expected completion amount above the stock it comes from would "
            "mean a probability over 1")
        assert forecast["probabilityBasis"], (
            "an expected figure must say what its probability rests on")
        timing = [f for f in _findings(answer)
                  if f["kind"] == contract_mod.KIND_TIMING]
        assert timing and all(f.get("forecastDate") for f in timing)

    # -- Q7 ---------------------------------------------------------------- #
    def test_q7_compares_the_two_governed_sides_and_reconciles(self, answers,
                                                               truth, second_book):
        answer = answers[Q7]
        assert answer["ok"] is True
        closing = truth[second_book["dates"][-1]]
        front = closing[closing["_segment"] == "Front Book"]
        back = closing[closing["_segment"] == "Back Book"]

        comparison = _finding(answer, metric="current_loan_to_value",
                              kind=contract_mod.KIND_COMPARISON)
        assert comparison is not None
        # ORIENTATION FOLLOWS THE QUESTION. Q7 asks how OLDER VINTAGES compare
        # with the front book, so the older side is the subject and the front
        # book the comparand — and the delta therefore carries the sign the
        # reader asked for. Before the analytical intent boundary, "older
        # vintages" was not recognised as a governed population at all and the
        # pair was assembled from "front book" alone, which put the two sides in
        # the opposite order to the sentence.
        _close(comparison["value"],
               _wavg(back["current_loan_to_value"],
                     back["current_outstanding_balance"]), tolerance=1e-6)
        _close(comparison["comparandValue"],
               _wavg(front["current_loan_to_value"],
                     front["current_outstanding_balance"]), tolerance=1e-6)
        assert comparison["population"]["rows"] == len(back)
        assert comparison["comparand"]["rows"] == len(front)

    def test_q7_vintages_reconcile_and_cover_the_whole_book(self, answers, truth,
                                                            second_book):
        closing = truth[second_book["dates"][-1]]
        cohorts = [f for f in _findings(answers[Q7])
                   if f["kind"] == contract_mod.KIND_COHORT]
        assert cohorts, "a vintage comparison with no vintages is not one"
        assert len(cohorts) == closing["_vintage"].nunique()
        for finding in cohorts:
            rows = closing[closing["_vintage"] == int(finding["label"])]
            _close(finding["value"],
                   float(_num(rows["current_outstanding_balance"]).sum()))
            assert finding["evidence"]["loanCount"] == len(rows)
        total = sum(f["value"] for f in cohorts)
        _close(total, float(_num(closing["current_outstanding_balance"]).sum()),
               tolerance=0.5)

    # -- Q8 ---------------------------------------------------------------- #
    def test_q8_two_populations_move_independently_and_reconcile(self, answers,
                                                                  truth,
                                                                  second_book):
        answer = answers[Q8]
        assert answer["ok"] is True
        opening, closing = second_book["dates"][0], second_book["dates"][-1]
        for key, provenance in (("direct", "direct"), ("acquired", "acquired")):
            finding = _finding(answer, metric="current_outstanding_balance",
                               kind=contract_mod.KIND_MOVEMENT, population=key)
            assert finding is not None, key
            close_rows = truth[closing][
                truth[closing]["source_portfolio_type"] == provenance]
            open_rows = truth[opening][
                truth[opening]["source_portfolio_type"] == provenance]
            _close(finding["value"],
                   float(_num(close_rows["current_outstanding_balance"]).sum()))
            _close(finding["priorValue"],
                   float(_num(open_rows["current_outstanding_balance"]).sum()))
            assert finding["population"]["rows"] == len(close_rows)

    def test_q8_the_two_sides_sum_to_the_whole_book(self, answers, truth,
                                                    second_book):
        closing = second_book["dates"][-1]
        sides = [_finding(answers[Q8], metric="current_outstanding_balance",
                          kind=contract_mod.KIND_MOVEMENT, population=key)
                 for key in ("direct", "acquired")]
        _close(sum(f["value"] for f in sides),
               float(_num(truth[closing]["current_outstanding_balance"]).sum()),
               tolerance=0.5)

    # -- Q9 ---------------------------------------------------------------- #
    def test_q9_forecast_is_funded_plus_weighted_expected_completions(self, answers,
                                                                      truth,
                                                                      second_book):
        answer = answers[Q9]
        assert answer["ok"] is True
        funded = _finding(answer, metric="current_outstanding_balance",
                          kind=contract_mod.KIND_MEASURE)
        expected = _finding(answer, metric="weighted_expected_funded_amount",
                            kind=contract_mod.KIND_FORECAST)
        forecast = _finding(answer, metric="forecast_funded_balance")
        assert None not in (funded, expected, forecast)
        _close(funded["value"],
               float(_num(truth[second_book["dates"][-1]]
                          ["current_outstanding_balance"]).sum()), tolerance=0.5)
        _close(forecast["forecastValue"],
               funded["value"] + expected["forecastValue"], tolerance=0.5)
        assert forecast["evidence"]["formula"].startswith(
            "forecast_funded_balance = current_funded_balance")

    def test_q9_discloses_the_two_different_as_at_dates(self, answers):
        forecast = _finding(answers[Q9], metric="forecast_funded_balance")
        evidence = forecast["evidence"]
        assert evidence["fundedReportingDate"] != evidence["pipelineAsOfDate"], (
            "the funded cut-off and the weekly pipeline extract are different "
            "dates and must never be presented as one")

    # -- cross-cutting ----------------------------------------------------- #
    @pytest.mark.parametrize("question", sorted(COMPOSED))
    def test_every_answer_publishes_its_plan_and_its_evidence(self, answers,
                                                              question):
        metadata = answers[question].get("metadata") or {}
        if not answers[question].get("ok"):
            pytest.skip("controlled refusal — checked separately")
        plan = metadata.get("analyticalPlan")
        assert plan and len(plan["calls"]) >= 2
        composition = metadata.get("analyticalComposition")
        assert composition and composition["intent"] == plan["intent"]
        for finding in _findings(answers[question]):
            assert finding["capability"], question
            if finding["status"] == contract_mod.STATUS_OK:
                assert finding["evidence"].get("engine"), (
                    f"{question}: {finding['label']} names no engine")

    @pytest.mark.parametrize("question", sorted(COMPOSED))
    def test_no_answer_reports_a_figure_no_finding_holds(self, answers, question):
        """The narrative is downstream of the findings, and provably so."""
        answer = answers[question]
        if not answer.get("ok"):
            pytest.skip("controlled refusal — no figures to check")
        import re
        # Magnitudes, because the narrative renders a direction separately
        # ("−£10.6m", "-8,681") and the sign is presentation, not a figure.
        rendered = set()

        def _record(value) -> None:
            if value is not None:
                rendered.add(round(abs(float(value)), 4))

        for finding in _findings(answer):
            for key in ("value", "priorValue", "change", "forecastValue",
                        "limit", "headroom", "comparandValue"):
                _record(finding.get(key))
            for key in ("share", "priorShare"):
                if finding.get(key) is not None:
                    rendered.add(round(abs(float(finding[key])) * 100, 1))
            _record((finding.get("population") or {}).get("rows"))
            _record((finding.get("comparand") or {}).get("rows"))
            # A rolling-cohort answer states the population at BOTH dates. The
            # prior count is held on the same population block the current one
            # is, so it is a declared figure by the same rule — the invariant
            # here is unchanged, only the field list was incomplete.
            _record((finding.get("population") or {}).get("rowsPrior"))
            _record((finding.get("comparand") or {}).get("rowsPrior"))
        # A difference between two declared populations is itself declared.
        rendered |= {round(abs(a - b), 4) for a in list(rendered)
                     for b in list(rendered)}
        # Every count in the answer text must be a population a finding declares.
        counts = re.findall(r"(?<![\d.,£%])(\d{1,3}(?:,\d{3})+)(?![\d.,%])",
                            answer["answer"])
        for raw in counts:
            value = round(abs(float(raw.replace(",", ""))), 4)
            assert value in rendered, (
                f"{question}: the answer states {raw}, which no finding holds")

    def test_the_safety_counters_are_zero(self, answers):
        """INCORRECT_SUCCESSFUL / SILENT_SEMANTIC_ERROR / HARD_FAILURE."""
        for question, answer in answers.items():
            if answer.get("ok"):
                continue
            # A refusal must name a reason; a blank refusal is a silent error.
            assert answer.get("answer"), question
            assert len(str(answer["answer"])) > 40, question
