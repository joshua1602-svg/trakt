#!/usr/bin/env python3
"""Target-state SERVING controls for the change-intelligence family.

WHAT IS UNDER TEST. The arrow between a compiled `GovernedQueryPlan` and the
deterministic owner that already computes the answer — for `material_summary`,
whose plan adapter existed but was wired to nothing, and for `attribution`, whose
plan adapter did not exist. Nothing about interpretation is under test here:
every control is built from STRUCTURED INTENT, no question is parsed, no wording
is matched and no model is called. ``LIVE_MODEL_CALLS = 0``.

WHAT IS THEREFORE PROVED, AND WHAT IS NOT. These prove that a plan carrying a
governed analytical form reaches exactly one deterministic owner, in exactly one
governed mode, over exactly the two snapshots the period resolver chose, under
exactly the scope the plan stated — and that a plan stating anything that owner
cannot honour is REFUSED rather than answered over the whole book. They do NOT
prove the interpreter assigns the right form to any particular sentence; that is
the live interpretation gate, measured separately at 17/17.

THE OWNER PARITY GATE is the last section. For every positive case the same
structured request is executed twice — once through the serving path, once by
calling the deterministic owner directly with a period request this file states
itself — and every published figure, the resolved pair, the scope and the
composed finding ids are required to be identical. A serving adapter that changed
any of them would be a second analytical owner, however thin it looked.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
import pytest

from mi_agent import plan_attribution as attribution
from mi_agent import plan_material_summary as material_summary
from mi_agent import plan_runtime_adapter as adapter
from mi_agent import plan_serving_canary as canary
from mi_agent import plan_temporal_runtime as temporal
from mi_agent.interpretation_v2.compiler import CompilerContext, DeterministicCompiler
from mi_agent.interpretation_v2.intent import parse_candidate_intent
from mi_agent.interpretation_v2.outcomes import OUTCOME_CLARIFY, OUTCOME_PLAN

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The month-on-month pair the commonest question in this family states, in the
#: compiler's own normal form: a pair with no grain and one period back.
PAIR = {"form": "relative_pair", "periods_back": 1}

#: The two governed source portfolios this fixture book records provenance for.
#: TWO, deliberately: with one, `chat_routing._apply_lens_filter` treats the book
#: as already being the scope and a Direct control would pass without narrowing
#: anything.
DIRECT_ID = "direct_001"
ACQUIRED_ID = "acquired_001"


# --------------------------------------------------------------------------- #
# the book — supplied through the EXISTING governed frame service
# --------------------------------------------------------------------------- #
def _frame(*, n: int, balance: float, ltv: float, first_loan: int = 0
           ) -> pd.DataFrame:
    ids = [DIRECT_ID if i % 2 == 0 else ACQUIRED_ID for i in range(n)]
    return pd.DataFrame({
        "loan_identifier": [f"L{i:05d}" for i in range(first_loan, first_loan + n)],
        "current_outstanding_balance": [balance / n] * n,
        "current_loan_to_value": [ltv] * n,
        "current_interest_rate": [5.5] * n,
        "interest_in_arrears": ["N"] * n,
        "account_status": ["Performing"] * n,
        "collateral_geography": ["South East"] * n,
        "source_portfolio_id": ids,
        "source_portfolio_label": [("Direct Book" if i == DIRECT_ID
                                    else "Acquired Book") for i in ids],
        "source_portfolio_type": [("direct" if i == DIRECT_ID else "acquired")
                                  for i in ids],
    })


def _book(count: int = 2):
    built = [
        {"run_id": "2026-05-31", "reporting_date": "2026-05-31",
         "df": _frame(n=100, balance=1_000_000.0, ltv=0.40),
         "source": "blob://x/2026-05-31/platform_canonical_typed.csv"},
        {"run_id": "2026-06-30", "reporting_date": "2026-06-30",
         "df": _frame(n=110, balance=1_200_000.0, ltv=0.43),
         "source": "blob://x/2026-06-30/platform_canonical_typed.csv"},
    ]
    return built[-count:]


@pytest.fixture
def frames(monkeypatch):
    """The governed funded frames, through the service the route already uses.

    `evolution.funded_frames` is the one door `period_change_route.build_snapshots`
    opens, so standing in here means the serving path reaches its data exactly as
    production does — not through a parallel loader written for a test.
    """
    built = _book()
    from mi_agent_api import evolution as evolution_mod
    monkeypatch.setattr(evolution_mod, "funded_frames",
                        lambda *a, **k: [dict(f) for f in built])
    return built


@pytest.fixture
def one_snapshot(monkeypatch):
    built = _book(count=1)
    from mi_agent_api import evolution as evolution_mod
    monkeypatch.setattr(evolution_mod, "funded_frames",
                        lambda *a, **k: [dict(f) for f in built])
    return built


@pytest.fixture
def registry(monkeypatch):
    """The governed portfolio registry, resolving a ROLE to its portfolio ids.

    THE HALF THAT MAKES A SCOPE REAL. `portfolio_lens.lens_from_term("direct")`
    produces `{source_portfolio_type: "direct"}`, and `_apply_lens_filter`
    narrows on `source_portfolio_id` — so an unresolved role lens narrows
    NOTHING, which is why it now raises instead. `contract_scope` resolves the
    role through `portfolio_context` before handing the lens on, and this stands
    in for that registry. The un-resolvable case has its own control below.
    """
    from mi_agent_api import portfolio_context as ctx_mod

    class _Scope:
        def __init__(self, ids):
            self.filters = {"source_portfolio_id": list(ids)}

    class _Resolved:
        def __init__(self, ids):
            self.scope = _Scope(ids)

    by_context = {"direct": (DIRECT_ID,), "acquired": (ACQUIRED_ID,),
                  DIRECT_ID: (DIRECT_ID,), ACQUIRED_ID: (ACQUIRED_ID,)}

    def resolve_context(context_id, **_kw):
        return _Resolved(by_context.get(str(context_id), ()))

    monkeypatch.setattr(ctx_mod, "resolve_context", resolve_context)
    return by_context


# --------------------------------------------------------------------------- #
# structured plans — no question is parsed anywhere in this file
# --------------------------------------------------------------------------- #
def _compile(change_form, **over):
    body = {"schema_version": "candidate_intent/1.0",
            "population": {"base": "funded"}}
    if change_form is not None:
        body["change_form"] = change_form
    body.update(over)
    return DeterministicCompiler(CompilerContext()).compile(
        parse_candidate_intent(body))


def _plan(change_form, **over):
    result = _compile(change_form, **over)
    assert result.outcome == OUTCOME_PLAN, [r.code for r in result.reasons]
    return result.plan.to_dict()


def _summary(**over):
    body = dict(capability="generic_analysis", operation="movement",
                measures=[], time=dict(PAIR))
    body.update(over)
    return _plan("material_summary", **body)


def _attribution(**over):
    body = dict(capability="funded_bridge", operation="bridge",
                measures=[{"concept": "funded_balance_movement"}],
                time=dict(PAIR))
    body.update(over)
    return _plan("attribution", **body)


# --------------------------------------------------------------------------- #
# the serving path, called exactly as `_attempt` calls it
# --------------------------------------------------------------------------- #
def _serve(plan, *, owner=None, client_id="client", tenant_id="tenant_a",
           output_root="blob://x"):
    """``(payload, reason, record)`` for one serving attempt.

    `_attempt_change_form` is the real dispatch helper, reached with the real
    adapter, the real owner and the real renderer. The interpreter is the one
    stage not exercised — these controls start from a compiled plan on purpose.
    """
    body: dict = {}
    resolved = owner or canary._change_form_owner(plan)
    assert resolved is not None, "no change-form adapter claimed this plan"
    payload, reason = canary._attempt_change_form(
        body, plan=plan, owner=resolved, question="(not read)",
        client_id=client_id, output_root=output_root, tenant_id=tenant_id,
        authorised_portfolio_ids=(), run_id=None,
        render_portfolio_id="client/2026-06-30", as_of=None)
    return payload, reason, body


def _receipt(record):
    return (record.get("execution") or {}).get("receipt") or {}


# --------------------------------------------------------------------------- #
# A — MATERIAL SUMMARY
# --------------------------------------------------------------------------- #

def test_A1_explicit_relative_pair_reaches_the_material_summary_runtime(frames):
    plan = _summary()
    assert canary._change_form_owner(plan) is material_summary
    payload, reason, record = _serve(plan)
    assert payload is not None, reason
    receipt = _receipt(record)
    assert receipt["change_form"] == "material_summary"
    assert receipt["capability"] == "period_movement"
    assert receipt["operation"] == "summary"
    assert receipt["mode"] == "portfolio_overview"
    assert receipt["calculation_owner"] == material_summary.CALCULATION_OWNER
    assert receipt["composition_owner"] == "mi_agent_api.insight_funded.compose"
    assert receipt["period_from"] == "2026-05-31"
    assert receipt["period_to"] == "2026-06-30"
    assert receipt["interpreted_time_present"] is True
    assert receipt["interpreted_period_form"] == "relative_pair"
    assert receipt["temporal_default_applied"] is False
    assert receipt["plan_id"] == plan["plan_id"]


def test_A2_an_explicit_current_anchor_resolves_current_vs_previous(frames):
    plan = _summary(time={"form": "current"})
    payload, reason, record = _serve(plan)
    assert payload is not None, reason
    receipt = _receipt(record)
    # THE INTERPRETED FORM IS NOT REWRITTEN. `current` stays `current`; the
    # RESOLUTION is the owner's, and it is the two adjacent governed snapshots.
    assert receipt["interpreted_time_present"] is True
    assert receipt["interpreted_period_form"] == "current"
    assert receipt["period_completed_by_form"] is True
    assert receipt["temporal_default_applied"] is False
    assert receipt["period_resolution"]["resolution_method"] == "current_vs_previous"
    assert (receipt["period_from"], receipt["period_to"]) == ("2026-05-31",
                                                             "2026-06-30")


def test_A3_absent_time_uses_the_authorised_default_and_says_so(frames):
    # NO `time` BLOCK AT ALL, which is the state under test: the reading stated
    # no temporal form and the compiler applied the form's authorised default.
    plan = _plan("material_summary", capability="generic_analysis",
                 operation="movement", measures=[])
    payload, reason, record = _serve(plan)
    assert payload is not None, reason
    receipt = _receipt(record)
    # ABSENCE IS NOT AN EXPLICIT `current`, and the receipt keeps them apart.
    assert receipt["interpreted_time_present"] is False
    assert receipt["interpreted_period_form"] is None
    assert receipt["temporal_default_applied"] is True
    assert receipt["temporal_default_method"] == "current_vs_previous"
    assert receipt["temporal_default_owner"] == "material_summary"
    assert receipt["period_resolution"]["resolution_method"] == "current_vs_previous"
    assert (receipt["period_from"], receipt["period_to"]) == ("2026-05-31",
                                                             "2026-06-30")


def test_A4_a_direct_scope_survives_end_to_end(frames, registry):
    plan = _summary(population={"base": "funded", "lens": "direct"})
    payload, reason, record = _serve(plan)
    assert payload is not None, reason
    receipt = _receipt(record)
    assert receipt["direct_acquired_scope"] == "direct"
    # THE EXECUTED SIDE, not the requested one: the portfolio ids the comparison
    # was actually narrowed to, from the workflow's own scope reference.
    assert receipt["executed_scope"]["portfolio_ids"] == [DIRECT_ID]
    assert receipt["executed_scope"]["context_id"] == "direct"
    assert receipt["executed_scope"]["tenant_id"] == "tenant_a"


def test_A4b_a_scope_the_registry_cannot_resolve_refuses_rather_than_widens(frames):
    """NO REGISTRY, NO ANSWER. The role lens then carries a type string and no
    portfolio id, `_apply_lens_filter` raises, and the legacy envelope serves.
    The failure this guards is the measured one: five snapshots in at full size
    and out at full size, a whole-book movement published under the word
    "Direct"."""
    plan = _summary(population={"base": "funded", "lens": "direct"})
    payload, reason, record = _serve(plan)
    assert payload is None
    assert reason == canary.EXECUTION_FAILED
    assert "LensNotApplied" in (record["execution"].get("error") or "")


@pytest.mark.parametrize("filters", [
    [{"concept": "account_status", "comparator": "eq", "value": "Performing"}],
    [{"concept": "collateral_geography", "comparator": "eq",
      "value": "South East"}],
])
def test_A5_an_unsupported_filter_refuses_and_never_widens(frames, filters):
    """THE OWNER APPLIES NO ROW PREDICATES, so a restriction cannot be honoured.

    `analyse_period_change` passes no `population=` to `build_snapshots`, so the
    restriction would reach neither snapshot. Refused with the slot named —
    never dropped, and never answered over the whole book.
    """
    plan = _summary(filters=filters)
    assert plan["filters"] or any(o["filters"] for o in plan["outputs"])
    eligible, why, detail = material_summary.check_eligibility(plan)
    assert not eligible
    assert why == material_summary.FILTER_NOT_SUPPORTED, detail
    payload, reason, record = _serve(plan)
    assert payload is None
    assert reason == f"{canary.INELIGIBLE}:{material_summary.FILTER_NOT_SUPPORTED}"
    assert record["execution"]["attempted"] is False


@pytest.mark.parametrize("dimension,expected", [
    ("account_status", "DIMENSION_NARROWS_THE_SUMMARY"),
    ("product_type", "DIMENSION_NARROWS_THE_SUMMARY"),
    # A GOVERNED GEOGRAPHY IS ITS OWN AXIS with its own owner, and the plan
    # carries it in `geography` rather than as a dimension. Refused under the
    # slot it actually occupies, which is what "name the slot" means.
    ("collateral_geography", "GEOGRAPHY_NOT_SUPPORTED"),
])
def test_A5b_a_named_dimension_is_a_different_question(frames, dimension,
                                                      expected):
    plan = _summary(dimensions=[dimension])
    eligible, why, detail = material_summary.check_eligibility(plan)
    assert not eligible
    assert why == getattr(material_summary, expected), detail


# --------------------------------------------------------------------------- #
# B — ATTRIBUTION
# --------------------------------------------------------------------------- #

def test_B6_explicit_relative_pair_reaches_the_existing_bridge_owner(frames):
    plan = _attribution()
    assert canary._change_form_owner(plan) is attribution
    payload, reason, record = _serve(plan)
    assert payload is not None, reason
    receipt = _receipt(record)
    assert receipt["change_form"] == "attribution"
    assert receipt["capability"] == "funded_bridge"
    assert receipt["operation"] == "bridge"
    assert receipt["calculation_owner"] == attribution.CALCULATION_OWNER
    assert receipt["composition_owner"] is None
    assert receipt["bridge_status"] == "available"
    assert receipt["bridge_reconciles"] is True
    assert (receipt["period_from"], receipt["period_to"]) == ("2026-05-31",
                                                             "2026-06-30")
    # The decomposition is published as the owner's own table, not re-derived.
    assert "Balance bridge" in [a.get("title") for a in payload["artifacts"]]


def test_B7_an_explicit_current_anchor_resolves_current_vs_previous(frames):
    plan = _attribution(time={"form": "current"})
    payload, reason, record = _serve(plan)
    assert payload is not None, reason
    receipt = _receipt(record)
    assert receipt["interpreted_period_form"] == "current"
    assert receipt["period_completed_by_form"] is True
    assert receipt["period_resolution"]["resolution_method"] == "current_vs_previous"


def test_B8_absent_time_uses_the_authorised_default_and_says_so(frames):
    plan = _plan("attribution", capability="funded_bridge", operation="bridge",
                 measures=[{"concept": "funded_balance_movement"}])
    payload, reason, record = _serve(plan)
    assert payload is not None, reason
    receipt = _receipt(record)
    assert receipt["interpreted_time_present"] is False
    assert receipt["interpreted_period_form"] is None
    assert receipt["temporal_default_applied"] is True
    assert receipt["temporal_default_method"] == "current_vs_previous"
    assert receipt["temporal_default_owner"] == "attribution"
    assert receipt["period_resolution"]["resolution_method"] == "current_vs_previous"


@pytest.mark.parametrize("role,expected", [("direct", DIRECT_ID),
                                           ("acquired", ACQUIRED_ID)])
def test_B9_a_governed_role_scope_survives_end_to_end(frames, registry, role,
                                                      expected):
    plan = _attribution(population={"base": "funded", "lens": role})
    payload, reason, record = _serve(plan)
    assert payload is not None, reason
    receipt = _receipt(record)
    assert receipt["direct_acquired_scope"] == role
    assert receipt["executed_scope"]["portfolio_ids"] == [expected]
    assert receipt["bridge_status"] == "available"


def test_B9b_attribution_across_a_dimension_is_refused_not_substituted(frames):
    """This owner decomposes by LOAN IDENTITY and groups by nothing.

    The estate's other bridge attributes across a dimension and resolves its own
    opening period; it cannot honour a window this form's temporal contract
    defaults. So a dimensioned attribution is refused with the dimension named,
    rather than answered by the decomposition the reader did not ask for.
    """
    plan = _attribution(dimensions=["account_status"])
    eligible, why, detail = attribution.check_eligibility(plan)
    assert not eligible
    assert why == attribution.DIMENSION_NOT_BRIDGEABLE, detail
    # And a governed geography, which arrives in its own slot, likewise.
    geo = _attribution(dimensions=["collateral_geography"])
    eligible, why, detail = attribution.check_eligibility(geo)
    assert not eligible
    assert why == attribution.GEOGRAPHY_NOT_SUPPORTED, detail


def test_B9c_a_measure_the_bridge_does_not_publish_is_refused():
    plan = _attribution(measures=[{"concept": "current_outstanding_balance",
                                   "statistic": "sum"}])
    eligible, why, _ = attribution.check_eligibility(plan)
    assert not eligible
    assert why == attribution.MEASURE_NOT_A_BRIDGE_TARGET


# --------------------------------------------------------------------------- #
# C — CONTROLS
# --------------------------------------------------------------------------- #

def test_C10_metric_delta_reaches_the_path_it_already_reached():
    """UNCHANGED, and that is the assertion.

    `metric_delta` is `period_movement`/`movement`. No change-form adapter claims
    it, so the dispatch added by this sprint does not see it, and it arrives at
    the perimeters it arrived at before — claimed by the temporal runtime on its
    period form and refused there as a specialist capability. That refusal is
    this form's existing behaviour; connecting it was not in this sprint, and
    refactoring it for symmetry would reorganise a working path.
    """
    plan = _plan("metric_delta", capability="generic_analysis",
                 operation="movement",
                 measures=[{"concept": "current_outstanding_balance",
                            "statistic": "sum"}], time=dict(PAIR))
    assert plan["capability"] == "period_movement"
    assert plan["operation"] == "movement"
    assert canary._change_form_owner(plan) is None
    assert temporal.claims(plan) is True
    eligible, why, _ = temporal.check_temporal_eligibility(plan)
    assert not eligible and why == adapter.CAPABILITY_NOT_GENERIC


def test_C11_level_comparison_stays_on_its_governed_temporal_path():
    plan = _plan("level_comparison", capability="generic_analysis",
                 operation="compare",
                 measures=[{"concept": "current_outstanding_balance",
                            "statistic": "sum"}], time=dict(PAIR))
    assert canary._change_form_owner(plan) is None
    assert temporal.claims(plan) is True
    eligible, why, detail = temporal.check_temporal_eligibility(plan)
    assert eligible, f"{why}: {detail}"


def test_C12_a_missing_change_form_does_not_serve():
    """The completeness perimeter refuses BEFORE a plan exists, so there is
    nothing for the dispatch to claim. Proved at both layers: the compile is a
    CLARIFY, and a plan with no form binding is claimed by no adapter."""
    result = _compile(None, capability="generic_analysis", operation="movement",
                      measures=[], time=dict(PAIR))
    assert result.outcome == OUTCOME_CLARIFY
    assert "MISSING_REQUIRED_SLOT" in [r.code for r in result.reasons]
    assert result.plan is None
    assert canary._change_form_owner({"provenance": {"compiler_bindings": {}}}) is None


@pytest.mark.parametrize("change_form,capability,operation", [
    ("attribution", "funded_bridge", "breakdown"),
    ("attribution", "funded_bridge", "series"),
    ("material_summary", "generic_analysis", "point_in_time"),
])
def test_C13_an_incompatible_form_and_operation_is_refused(change_form,
                                                          capability, operation):
    """THE GOVERNED VOCABULARY REFUSES FIRST. `CAPABILITY_OPERATIONS` states
    which shapes each capability produces, so an incompatible pair never reaches
    an adapter."""
    result = _compile(change_form, capability=capability, operation=operation,
                      measures=[], time=dict(PAIR))
    assert result.outcome != OUTCOME_PLAN
    assert result.plan is None


@pytest.mark.parametrize("operation", ["breakdown", "rank", "compare"])
def test_C13b_a_non_canonical_operation_reaching_the_adapter_fails_closed(operation):
    """A plan built without the canonicalisation seam is refused, not repaired.

    `normalise` rule 5 collapses this form's three spellings to `summary` before
    a plan exists. Re-canonicalising here would be a second place that rewrites
    an operation, and two such places can disagree.
    """
    plan = {"capability": "period_movement", "operation": operation,
            "population": {"base": "funded", "lens": "all"},
            "outputs": [{"id": "o", "measures": [], "dimensions": [],
                         "filters": []}],
            "period": {"form": "relative_pair", "periods_back": 1,
                       "stated": True},
            "comparison_kind": "none", "filters": [],
            "provenance": {"compiler_bindings": {
                "change_form": {"form": "material_summary"}}}}
    eligible, why, _ = material_summary.check_eligibility(plan)
    assert not eligible
    assert why == material_summary.OPERATION_NOT_ADMITTED


@pytest.mark.parametrize("form", ["material_summary", "attribution"])
def test_C14_one_snapshot_refuses_and_invents_no_date(one_snapshot, form):
    """A BOOK WITH ONE SNAPSHOT HAS NO PAIR TO COMPARE.

    The period resolver raises `PeriodChangeFailure(insufficient_snapshots)` —
    its own classified refusal — and the legacy envelope serves. No date is
    synthesised, no single-period answer is published as a movement, and the
    record names the owner's failure rather than a generic one.
    """
    plan = _summary() if form == "material_summary" else _attribution()
    payload, reason, record = _serve(plan)
    assert payload is None
    assert reason == canary.EXECUTION_FAILED
    error = record["execution"].get("error") or ""
    assert "PeriodChangeFailure" in error
    assert "2026-05-31" not in json.dumps(record), (
        "a period the book does not carry appears in the record")


def test_C15_a_pipeline_population_never_reaches_a_funded_comparison(frames):
    """THE POPULATION GATE RUNS FIRST, and that ordering is the control.

    Both connected forms are funded-book forms. A plan asking about the pipeline
    extract is refused by `check_population_base` before any change-form adapter
    is consulted, so a weekly pipeline question can never be answered from
    funded monthly snapshots.
    """
    plan = _summary(population={"base": "pipeline"})
    ok, why, _ = adapter.check_population_base(plan, "funded")
    assert not ok, why
    eligible, form_why, _ = material_summary.check_eligibility(plan)
    assert not eligible
    assert form_why == material_summary.POPULATION_NOT_FUNDED


# --------------------------------------------------------------------------- #
# No raw question, anywhere on the new arrow
# --------------------------------------------------------------------------- #
_NEW_MODULES = ("mi_agent/plan_attribution.py",
                "mi_agent/plan_material_summary.py")


@pytest.mark.parametrize("path", _NEW_MODULES)
def test_no_adapter_reads_a_question(path):
    """`question` is not an input to any decision on this arrow.

    Checked structurally: no function in either adapter takes a `question`
    parameter except the two that pass it straight to the envelope builder,
    where it is ECHOED into the response and read by nothing; and neither module
    imports a parser, a recogniser, a router or `re`.
    """
    tree = ast.parse((_REPO_ROOT / path).read_text())
    echoing = {"governed_envelope", "envelope"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            takes = {a.arg for a in node.args.args} | {
                a.arg for a in node.args.kwonlyargs}
            if "question" in takes:
                assert node.name in echoing, (
                    f"{path}:{node.name} takes a question and is not an "
                    f"envelope builder")
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for name in names:
            root = name.split(".")[0]
            assert root != "re", f"{path} imports the regex module"
            assert not any(banned in name for banned in (
                "llm_query_parser", "parsed_question", "recogniser",
                "question_interpretation", "chat_routing")), (
                    f"{path} imports {name}")


def test_the_dispatch_reads_the_compilers_reading_and_not_the_models():
    """A serving decision is never taken from what the model claimed.

    `claims` reads `provenance.compiler_bindings["change_form"]`. The raw claim
    lives in `provenance.intent_claims["change_form"]`, and a plan whose two
    disagree is dispatched on the compiler's.
    """
    plan = _summary()
    assert plan["provenance"]["intent_claims"]["change_form"] == "material_summary"
    tampered = json.loads(json.dumps(plan))
    tampered["provenance"]["intent_claims"]["change_form"] = "attribution"
    assert canary._change_form_owner(tampered) is material_summary


# --------------------------------------------------------------------------- #
# OWNER PARITY GATE
# --------------------------------------------------------------------------- #
def _owner_direct(*, mode, period_request, lens=None, client_id="client",
                  tenant_id="tenant_a"):
    """The deterministic owner, invoked INDEPENDENTLY of the serving path.

    The period request is stated by this file rather than taken from the
    adapter, so the comparison is between two independent expressions of the
    same structured question — not between the adapter and itself.
    """
    from mi_agent_api.period_change_route import analyse_period_change

    return analyse_period_change(
        client_id=client_id, output_root="blob://x", mode=mode,
        period_request=period_request, scope=lens, tenant_id=tenant_id,
        include_bridge=True)


def _governed_request(relative_mode="current_vs_previous"):
    from mi_agent.period_change.periods import PeriodRequest

    return PeriodRequest(relative_mode=relative_mode)


def _role_lens(role):
    from mi_agent import portfolio_lens as lens_mod
    from mi_agent_api import contract_scope

    return contract_scope._through_the_registry(lens_mod.lens_from_term(role))


def _comparable(result):
    """Everything a reader could be shown, as plain data."""
    payload = result.to_dict()
    return {
        "period": payload["period_resolution"]["resolution_method"],
        "from": payload["period_resolution"]["resolved_start_snapshot"],
        "to": payload["period_resolution"]["resolved_end_snapshot"],
        "scope": payload["period_resolution"]["portfolio_scope"],
        "metrics": payload["metric_changes"],
        "distributions": payload["distribution_changes"],
        "bridge": payload["balance_bridge"],
        "summary": payload["summary"],
    }


PARITY_CASES = [
    ("A1 material_summary / explicit pair", "material_summary", {}, None),
    ("A2 material_summary / current anchor", "material_summary",
     {"time": {"form": "current"}}, None),
    ("A3 material_summary / absent time", "material_summary", {"absent": True},
     None),
    ("A4 material_summary / Direct scope", "material_summary",
     {"population": {"base": "funded", "lens": "direct"}}, "direct"),
    ("B6 attribution / explicit pair", "attribution", {}, None),
    ("B7 attribution / current anchor", "attribution",
     {"time": {"form": "current"}}, None),
    ("B8 attribution / absent time", "attribution", {"absent": True}, None),
    ("B9 attribution / Acquired scope", "attribution",
     {"population": {"base": "funded", "lens": "acquired"}}, "acquired"),
]


def _parity_plan(form, over):
    over = dict(over)
    absent = over.pop("absent", False)
    if form == "material_summary":
        body = dict(capability="generic_analysis", operation="movement",
                    measures=[])
        if not absent:
            body["time"] = over.pop("time", dict(PAIR))
    else:
        body = dict(capability="funded_bridge", operation="bridge",
                    measures=[{"concept": "funded_balance_movement"}])
        if not absent:
            body["time"] = over.pop("time", dict(PAIR))
    body.update(over)
    return _plan(form, **body)


@pytest.mark.parametrize("label,form,over,role",
                         PARITY_CASES,
                         ids=[c[0] for c in PARITY_CASES])
def test_owner_parity(frames, registry, label, form, over, role):
    """NUMERICAL, SEMANTIC AND SCOPE DIFFERENCES MUST ALL BE ZERO.

    The serving adapter may translate a plan and shape an envelope. It may not
    change a figure, a period, a scope or a finding — and the only way to prove
    that is to run the owner twice and compare everything a reader could see.
    """
    plan = _parity_plan(form, over)
    payload, reason, record = _serve(plan)
    assert payload is not None, reason

    direct = _owner_direct(
        mode=(material_summary.WORKFLOW_MODE if form == "material_summary"
              else attribution.WORKFLOW_MODE),
        period_request=_governed_request(),
        lens=_role_lens(role) if role else None)

    receipt = _receipt(record)
    # SCOPE
    assert receipt["executed_scope"] == direct.portfolio_scope.to_dict()
    # PERIOD
    expected = _comparable(direct)
    assert receipt["period_from"] == expected["from"]["reporting_date"]
    assert receipt["period_to"] == expected["to"]["reporting_date"]
    assert receipt["period_resolution"]["resolution_method"] == expected["period"]
    assert receipt["snapshot_references"] == [expected["from"], expected["to"]]
    # NUMBERS — every figure the reader is actually shown. The served tables are
    # compared against rows the OWNER'S OWN builders produce from the
    # independently-run result, so an adapter that rounded, rescaled or
    # re-ordered a published value would be caught here rather than in a field
    # chosen for the comparison.
    from mi_agent_api import period_change_route as pcr

    titles = {a["title"]: a for a in payload["artifacts"]}
    assert titles["Metric movements"]["rows"] == pcr._metric_rows(direct)
    if pcr._distribution_rows(direct):
        assert (titles["Composition shifts"]["rows"]
                == pcr._distribution_rows(direct))
    assert titles["Balance bridge"]["rows"] == pcr._bridge_rows(direct)

    # SEMANTICS — the same structured request served twice produces the same
    # receipt, so nothing in the path depends on call order or wall-clock.
    again = _serve(plan)[2]
    assert _receipt(again) == receipt, "the serving path is not deterministic"

    if form == "attribution":
        assert receipt["bridge_status"] == expected["bridge"]["status"]
        assert receipt["bridge_reconciles"] == expected["bridge"]["reconciles"]
        assert receipt["bridge_residual"] == expected["bridge"]["residual"]
        assert receipt["bridge_balance_field"] == expected["bridge"]["balance_field"]
    else:
        # The COMPOSITION is the owner of a material summary's findings, so
        # parity is over the findings, composed independently from the
        # independently-run result.
        from mi_agent_api import insight_funded

        composed = insight_funded.compose(
            direct, tenant_id="tenant_a", portfolio_id="client",
            portfolio_context=(direct.portfolio_scope.context_id or "total"),
            limit_envelope=None, run_id=None)
        brief = payload["metadata"]["materialChangeBrief"]
        assert [i["insight_id"] for i in brief["insights"]] == [
            i["insight_id"] for i in composed["insights"]]
        assert [i["metrics"] for i in brief["insights"]] == [
            i["metrics"] for i in composed["insights"]]
        assert brief["insight_count"] == composed["insight_count"]
        assert brief["config_source"] == composed["config_source"]
        assert receipt["finding_count"] == composed["insight_count"]


def test_the_served_envelope_publishes_the_governed_plan_receipt(frames):
    plan = _summary()
    payload, reason, _record = _serve(plan)
    assert payload is not None, reason
    governed = payload["metadata"]["governedPlan"]
    assert governed["requested"]["change_form"] == "material_summary"
    assert governed["requested"]["plan_id"] == plan["plan_id"]
    assert governed["executed"]["calculation_owner"] == (
        material_summary.CALCULATION_OWNER)
    assert payload["metadata"]["route"] == "governed_plan_material_summary"
    assert payload["metadata"]["parserMode"] == "governed_plan"
