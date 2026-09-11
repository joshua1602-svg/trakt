"""Pipeline executes from a GovernedQueryPlan, and only Pipeline does.

Every control the slice requires, offline. No model call, no network: the
intents are constructed or replayed, and the data is the committed pipeline
fixtures — the October extract and the five-week history — so the figures are
the Pipeline owners' own and nothing here writes a portfolio value down.
"""

from __future__ import annotations

import glob
import json

import pytest

from mi_agent import plan_pipeline_runtime as pipeline_rt
from mi_agent import plan_runtime_adapter as adapter
from mi_agent import plan_temporal_runtime as temporal
from mi_agent.interpretation_v2.compiler import CompilerContext, DeterministicCompiler
from mi_agent.interpretation_v2.intent import parse_candidate_intent

_EXTRACT = sorted(glob.glob("tests/fixtures/client_001_mi_pack/pipeline/*/*"))[0]
_HISTORY_ROOT = "tests/fixtures/pipeline_history_5w"
_CLIENT = "client_001"
_SOURCE = {"source_file": _EXTRACT, "pipeline_as_of_date": "2025-10-01"}


def _code_of(module) -> str:
    """The module's EXECUTABLE text: docstrings and comments stripped.

    A module that documents what it must not reach for names those seams in
    prose, and a guard that greps the raw file then fails on its own
    explanation. Measured here twice before this existed. Comments are dropped
    by the tokeniser and docstrings by walking the AST, so what is left is what
    actually runs.
    """
    import ast
    import io
    import inspect
    import tokenize

    source = inspect.getsource(module)
    tree = ast.parse(source)
    doc_lines = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if (isinstance(first, ast.Expr)
                    and isinstance(getattr(first, "value", None), ast.Constant)
                    and isinstance(first.value.value, str)):
                doc_lines.update(range(first.lineno,
                                       (first.end_lineno or first.lineno) + 1))
    kept = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            continue
        if token.start[0] in doc_lines:
            continue
        kept.append(token.string)
    return " ".join(kept)


def _plan(**over):
    body = {"schema_version": "candidate_intent/1.0", "capability": "pipeline",
            "operation": "summary", "population": {"base": "pipeline"},
            "measures": [{"concept": "pipeline_amount"}],
            "time": {"form": "current"}}
    body.update(over)
    result = DeterministicCompiler(CompilerContext()).compile(
        parse_candidate_intent(body))
    assert result.plan is not None, (str(result.outcome),
                                     [r.code for r in result.reasons])
    return result.plan.to_dict()


def _funded_plan(**over):
    body = {"schema_version": "candidate_intent/1.0",
            "capability": "generic_analysis", "operation": "point_in_time",
            "population": {"base": "funded"},
            "measures": [{"concept": "current_outstanding_balance"}],
            "time": {"form": "current"}}
    body.update(over)
    return DeterministicCompiler(CompilerContext()).compile(
        parse_candidate_intent(body)).plan.to_dict()


# --------------------------------------------------------------------------- #
# A — current pipeline, the slice 1 shapes
# --------------------------------------------------------------------------- #

def test_C1_pipeline_balance_executes_from_the_plan():
    plan = _plan()
    assert pipeline_rt.claims(plan)
    assert pipeline_rt.check_eligibility(plan)[0] is True
    out = pipeline_rt.execute_current(plan, source=_SOURCE)
    assert out.ok, out.detail
    # The figure is the Pipeline owner's own, read back from the same report.
    from mi_agent_api.pipeline_contract import load_prepared_pipeline
    _frame, report = load_prepared_pipeline(_SOURCE)
    assert out.value == pytest.approx(float(report["total_pipeline_amount"]))
    assert out.receipt["capability"] == "pipeline"
    assert out.receipt["population_base"] == "pipeline"
    assert out.receipt["measure_concept"] == "pipeline_amount"
    assert out.receipt["result_shape"] == "scalar"
    assert out.receipt["dataset"]["identity"] == "governed_pipeline_extract"


def test_C2_pipeline_case_count_executes_from_the_plan():
    plan = _plan(operation="point_in_time", measures=[{"concept": "loan"}])
    out = pipeline_rt.execute_current(plan, source=_SOURCE)
    assert out.ok, out.detail
    from mi_agent_api.pipeline_contract import load_prepared_pipeline
    _frame, report = load_prepared_pipeline(_SOURCE)
    assert out.value == pytest.approx(float(report["row_count"]))
    assert out.receipt["measure_kind"] == "count"


def test_C3_pipeline_by_stage_executes_and_the_receipt_proves_the_axis():
    plan = _plan(operation="breakdown", measures=[{"concept": "loan"}],
                 dimensions=["pipeline_stage"])
    out = pipeline_rt.execute_current(plan, source=_SOURCE)
    assert out.ok, out.detail
    from mi_agent_api.pipeline_contract import load_prepared_pipeline
    _frame, report = load_prepared_pipeline(_SOURCE)
    assert {c["pipeline_stage"]: c["value"] for c in out.cells} == {
        str(k): float(v) for k, v in report["stage_counts"].items()}
    assert out.receipt["group_field_keys"] == ["pipeline_stage"]
    assert out.receipt["result_shape"] == "grouped"


# --------------------------------------------------------------------------- #
# B — temporal pipeline, the slice 2 shapes, on WEEKLY history
# --------------------------------------------------------------------------- #

def test_T1_pipeline_evolution_by_stage_uses_the_weekly_owner():
    plan = _plan(operation="breakdown", measures=[{"concept": "pipeline_amount"}],
                 dimensions=["pipeline_stage"],
                 time={"form": "series", "grain": "weekly"})
    assert pipeline_rt.is_temporal(plan) is True
    out = pipeline_rt.execute_temporal(plan, root=_HISTORY_ROOT,
                                       client_id=_CLIENT)
    assert out.ok, out.detail
    from mi_agent_api import evolution as evolution_mod
    series = evolution_mod.pipeline_evolution(_HISTORY_ROOT, _CLIENT, None)
    assert len(out.cells) == len(series["byStage"])
    assert out.receipt["group_field_keys"] == ["pipeline_stage"]
    assert out.receipt["grain"] == "weekly"
    assert out.receipt["temporal_basis"] == "governed_weekly_pipeline_extracts"
    assert out.receipt["dataset"]["identity"] == "governed_weekly_pipeline_extracts"


def test_T2_simple_weekly_pipeline_series_uses_the_weekly_owner():
    """The closest already-supported simple temporal shape, and it exists.

    `evolution.pipeline_evolution` publishes `periods[].metrics.pipeline_amount`
    and `pipeline_case_count` per governed extract, so a weekly balance or count
    series is an EXISTING capability and no new one was built for it.
    """
    plan = _plan(operation="series", measures=[{"concept": "pipeline_amount"}],
                 time={"form": "series", "grain": "weekly"})
    out = pipeline_rt.execute_temporal(plan, root=_HISTORY_ROOT, client_id=_CLIENT)
    assert out.ok, out.detail
    from mi_agent_api import evolution as evolution_mod
    series = evolution_mod.pipeline_evolution(_HISTORY_ROOT, _CLIENT, None)
    assert [c["value"] for c in out.cells] == [
        p["metrics"]["pipeline_amount"] for p in series["periods"]]
    assert out.receipt["selected_periods"] == [
        p["week"] for p in series["periods"]]


def test_the_funded_snapshot_store_is_never_consulted_for_pipeline():
    """Pipeline time is weekly extracts; funded time is monthly snapshots."""
    source = _code_of(pipeline_rt)
    for forbidden in ("SnapshotStore", "snapshot_store", "plan_temporal_runtime",
                      "select_between", "resolve_range"):
        assert forbidden not in source, (
            f"the pipeline runtime reaches for {forbidden!r}, which belongs to "
            f"the funded monthly catalogue")


# --------------------------------------------------------------------------- #
# C — funded controls
# --------------------------------------------------------------------------- #

def test_F1_a_funded_current_plan_is_untouched_by_the_pipeline_runtime():
    plan = _funded_plan()
    assert pipeline_rt.claims(plan) is False
    assert adapter.check_eligibility(plan)[0] is True
    assert adapter.check_population_base(plan, "funded")[0] is True


def test_F2_a_funded_temporal_plan_is_untouched_by_the_pipeline_runtime():
    plan = _funded_plan(operation="series",
                        time={"form": "series", "grain": "monthly"})
    assert pipeline_rt.claims(plan) is False
    assert temporal.claims(plan) is True


# --------------------------------------------------------------------------- #
# D — wrong-runtime controls
# --------------------------------------------------------------------------- #

def test_S1_a_pipeline_plan_is_refused_by_the_funded_runtime():
    ok, why, _ = adapter.check_population_base(_plan(), "funded")
    assert (ok, why) == (False, adapter.POPULATION_NOT_EXECUTABLE)


def test_S2_a_funded_plan_is_refused_by_the_pipeline_runtime():
    ok, why, _ = pipeline_rt.check_eligibility(_funded_plan())
    assert (ok, why) == (False, pipeline_rt.CAPABILITY_NOT_PIPELINE)
    # And a pipeline-capability plan that asks for funded rows is refused too.
    ok, why, _ = pipeline_rt.check_eligibility(
        _plan(population={"base": "funded"}))
    assert (ok, why) == (False, pipeline_rt.POPULATION_NOT_PIPELINE)


def test_S3_an_unproven_pipeline_population_is_refused():
    for undeclared in (None, "", "   "):
        ok, why, _ = adapter.check_population_base(
            _plan(), undeclared, executable=pipeline_rt.EXECUTABLE_POPULATIONS)
        assert (ok, why) == (False, adapter.EXECUTED_POPULATION_UNPROVEN)


def test_S4_a_pipeline_plan_whose_runtime_claims_funded_is_refused():
    ok, why, _ = adapter.check_population_base(
        _plan(), "funded", executable=pipeline_rt.EXECUTABLE_POPULATIONS)
    assert (ok, why) == (False, adapter.POPULATION_BASE_MISMATCH)


def test_S5_a_stage_request_whose_receipt_omits_the_grouping_is_unaccounted():
    """The coverage ledger, not the runtime, is what catches a dropped axis."""
    from mi_agent_api.mi_service import _governed_plan_coverage

    plan = _plan(operation="breakdown", measures=[{"concept": "loan"}],
                 dimensions=["pipeline_stage"])
    requested = adapter.requested_semantics(plan)
    envelope = {"metadata": {"parserMode": "governed_plan", "governedPlan": {
        "requested": requested,
        "executed": {"capability": "pipeline", "population_base": "pipeline",
                     "measure_concept": "loan", "group_field_keys": []}}}}
    ledger = _governed_plan_coverage(envelope)
    assert [e["field"] for e in ledger["unaccounted"]] == ["pipeline_stage"]


def test_S6_a_pipeline_answer_claiming_a_different_capability_is_unaccounted():
    from mi_agent_api.mi_service import _governed_plan_coverage

    requested = adapter.requested_semantics(_plan())
    envelope = {"metadata": {"parserMode": "governed_plan", "governedPlan": {
        "requested": requested,
        "executed": {"capability": "generic_analysis",
                     "population_base": "pipeline",
                     "measure_concept": "pipeline_amount",
                     "group_field_keys": []}}}}
    ledger = _governed_plan_coverage(envelope)
    assert "capability" in {e["field"] for e in ledger["unaccounted"]}


def test_a_substituted_measure_is_unaccounted():
    from mi_agent_api.mi_service import _governed_plan_coverage

    requested = adapter.requested_semantics(_plan())      # asks for the amount
    envelope = {"metadata": {"parserMode": "governed_plan", "governedPlan": {
        "requested": requested,
        "executed": {"capability": "pipeline", "population_base": "pipeline",
                     "measure_concept": "loan",        # served the count
                     "group_field_keys": []}}}}
    ledger = _governed_plan_coverage(envelope)
    assert "measure" in {e["field"] for e in ledger["unaccounted"]}


def test_a_fully_proved_pipeline_answer_is_accounted():
    from mi_agent_api.mi_service import _governed_plan_coverage

    plan = _plan(operation="breakdown", measures=[{"concept": "loan"}],
                 dimensions=["pipeline_stage"])
    out = pipeline_rt.execute_current(plan, source=_SOURCE)
    envelope = {"metadata": {"parserMode": "governed_plan", "governedPlan": {
        "requested": adapter.requested_semantics(plan),
        "executed": out.receipt}}}
    ledger = _governed_plan_coverage(envelope)
    assert ledger["unaccounted"] == [], ledger


# --------------------------------------------------------------------------- #
# E — the specialist boundary
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("capability,operation,measure", [
    ("borrowing_base", "headroom", "borrowing_base"),
    ("forecast", "forecast_projection", "pipeline_amount"),
    ("funded_bridge", "bridge", "funded_balance_movement"),
    ("portfolio_summary", "summary", "portfolio_overview"),
])
def test_S7_S8_no_other_specialist_capability_enters_the_pipeline_runtime(
        capability, operation, measure):
    body = {"schema_version": "candidate_intent/1.0", "capability": capability,
            "operation": operation, "population": {"base": "pipeline"},
            "measures": [{"concept": measure}], "time": {"form": "current"}}
    result = DeterministicCompiler(CompilerContext()).compile(
        parse_candidate_intent(body))
    plan = result.plan.to_dict() if result.plan is not None else dict(body)
    assert pipeline_rt.claims(plan) is False
    ok, why, _ = pipeline_rt.check_eligibility(plan)
    assert (ok, why) == (False, pipeline_rt.CAPABILITY_NOT_PIPELINE)


def test_only_the_pipeline_capability_is_migrated():
    assert pipeline_rt.CAPABILITY == "pipeline"
    assert pipeline_rt.EXECUTABLE_POPULATIONS == frozenset({"pipeline"})
    assert adapter.EXECUTABLE_POPULATIONS == frozenset({"funded"})


# --------------------------------------------------------------------------- #
# the invariant the whole tier rests on
# --------------------------------------------------------------------------- #

def test_the_pipeline_runtime_never_reads_the_question():
    source = _code_of(pipeline_rt)
    for forbidden in ("ParsedQuestion", "parse_with_repair", "resolve_dataset",
                      "try_route", "llm_query_parser", "chat_routing",
                      "recognise"):
        assert forbidden not in source, (
            f"the pipeline runtime reaches for {forbidden!r}; the plan is "
            f"authoritative once it exists")


def test_unsupported_pipeline_shapes_refuse_rather_than_approximate():
    # a filtered pipeline question
    filtered = _plan(filters=[{"concept": "erm_product_type", "comparator": "eq",
                               "value": "drawdown"}])
    assert pipeline_rt.check_eligibility(filtered)[1] == (
        pipeline_rt.FILTERS_NOT_SUPPORTED)
    # a current AMOUNT by stage, which no existing owner computes
    amount_by_stage = _plan(operation="breakdown",
                            measures=[{"concept": "pipeline_amount"}],
                            dimensions=["pipeline_stage"])
    assert pipeline_rt.check_eligibility(amount_by_stage)[1] == (
        pipeline_rt.MEASURE_NOT_SUPPORTED)
    # a period form the pipeline estate has no owner for
    ranged = _plan(time={"form": "range", "labels": ["May 2026", "June 2026"]})
    assert pipeline_rt.check_eligibility(ranged)[1] == (
        pipeline_rt.PERIOD_NOT_SUPPORTED)


# --------------------------------------------------------------------------- #
# end to end, through the real serve() — the dispatch, not just the runtime
# --------------------------------------------------------------------------- #

class _Scripted:
    """An interpreter that returns one recorded payload. No model call."""

    def __init__(self, payload):
        self._payload = payload

    def interpret(self, question, **_kwargs):
        from mi_agent.interpretation_v2.opus_interpreter import InterpretationOutcome
        return InterpretationOutcome(
            question=question,
            intent=parse_candidate_intent(dict(self._payload)),
            model_id="claude-opus-5", raw_payload=dict(self._payload))


def _served(payload, monkeypatch, **serve_over):
    import os

    from mi_agent import plan_serving_canary as canary
    from mi_agent import plan_shadow_evidence as evidence
    from mi_agent import plan_shadow_wiring as wiring

    monkeypatch.setenv("MI_AGENT_PLAN_SERVE", "canary")
    monkeypatch.setenv("MI_AGENT_PLAN_SERVE_PRINCIPALS", "canary-principal")
    wiring.set_interpreter_factory(lambda: _Scripted(payload))
    written = []
    monkeypatch.setattr(evidence, "write", lambda body: written.append(body))

    class Principal:
        actor_id = "canary-principal"

    try:
        kwargs = {"question": "q", "context": Principal(), "client_id": _CLIENT,
                  "run_id": None, "legacy_result": {"ok": True}, "frame": None,
                  "semantics": {}, "view": "funded",
                  "portfolio_id": f"{_CLIENT}/2025-10-01",
                  "render_portfolio_id": f"{_CLIENT}/2025-10-01", "as_of": None,
                  "pipeline_source": _SOURCE, "pipeline_root": _HISTORY_ROOT,
                  "pipeline_client_id": _CLIENT, "pipeline_history": None}
        kwargs.update(serve_over)
        payload_out = canary.serve(**kwargs)
    finally:
        wiring.set_interpreter_factory(None)
    return payload_out, (written[-1] if written else {})


_PIPELINE_INTENT = {
    "schema_version": "candidate_intent/1.0", "capability": "pipeline",
    "operation": "summary", "population": {"base": "pipeline"},
    "measures": [{"concept": "pipeline_amount"}], "time": {"form": "current"}}


def test_serve_dispatches_a_pipeline_plan_to_the_pipeline_runtime(monkeypatch):
    """The whole seam: plan -> dispatch -> existing owner -> NEW envelope.

    Note `view="funded"` above. The legacy router's dataset choice is DELIBERATELY
    wrong here, and the answer is still the pipeline's — which is the point of the
    slice: once the plan exists it owns the dataset decision, and the sentence the
    legacy router read does not.
    """
    payload, record = _served(_PIPELINE_INTENT, monkeypatch)
    assert payload is not None, record.get("execution")
    assert record["serving"]["decision"] == "NEW"
    assert record["execution"]["runtime"] == "pipeline_current"
    receipt = record["execution"]["receipt"]
    assert receipt["capability"] == "pipeline"
    assert receipt["population_base"] == "pipeline"

    from mi_agent_api.pipeline_contract import load_prepared_pipeline
    _frame, report = load_prepared_pipeline(_SOURCE)
    figure = payload["artifacts"][0]["kpis"][0]["rawValue"]
    assert figure == pytest.approx(float(report["total_pipeline_amount"]))

    # and the served envelope reconciles through the governed coverage owner
    from mi_agent_api.mi_service import _governed_plan_coverage
    assert _governed_plan_coverage(payload)["unaccounted"] == []


def test_serve_dispatches_a_temporal_pipeline_plan_to_the_weekly_owner(monkeypatch):
    payload, record = _served(
        dict(_PIPELINE_INTENT, operation="breakdown",
             dimensions=["pipeline_stage"],
             time={"form": "series", "grain": "weekly"}), monkeypatch)
    assert payload is not None, record.get("execution")
    assert record["serving"]["decision"] == "NEW"
    assert record["execution"]["runtime"] == "pipeline_temporal"
    receipt = record["execution"]["receipt"]
    assert receipt["grain"] == "weekly"
    assert receipt["group_field_keys"] == ["pipeline_stage"]
    assert len(receipt["selected_periods"]) == 5
    from mi_agent_api.mi_service import _governed_plan_coverage
    assert _governed_plan_coverage(payload)["unaccounted"] == []


def test_serve_refuses_a_pipeline_plan_when_no_pipeline_source_exists(monkeypatch):
    """Fail closed, and never onto the funded frame that WAS supplied."""
    payload, record = _served(_PIPELINE_INTENT, monkeypatch,
                              pipeline_source=None)
    assert payload is None
    assert record["execution"]["attempted"] is False
    assert "SOURCE_UNAVAILABLE" in record["execution"]["why_not"]


def test_serve_still_refuses_an_unmigrated_specialist_capability(monkeypatch):
    """borrowing_base falls through to the funded perimeter, exactly as before."""
    payload, record = _served(
        {"schema_version": "candidate_intent/1.0", "capability": "borrowing_base",
         "operation": "headroom", "population": {"base": "funded"},
         "measures": [{"concept": "borrowing_base_headroom"}],
         "time": {"form": "current"}}, monkeypatch)
    assert payload is None
    assert record["eligibility"]["reason"] == adapter.CAPABILITY_NOT_GENERIC
