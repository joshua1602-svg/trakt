"""Pipeline stage movement executes from a GovernedQueryPlan.

The oracle is the legacy route, on its own fixture. For every stage-movement
question the certification run recorded, the plan the compiler made from the
model's recorded intent is translated by this runtime into the owner's reading,
and that reading must be the one the legacy route built by reading the
sentence; the governed answer must then be the legacy answer, word for word.
No model call: the intents are the recorded ones, and the data is
`tests/fixtures/pipeline_transition_2w`, the fixture the legacy tests use.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from mi_agent import plan_runtime_registry as registry
from mi_agent import plan_stage_movement_runtime as stage_rt
from mi_agent.interpretation_v2.compiler import CompilerContext, DeterministicCompiler
from mi_agent.interpretation_v2.intent import parse_candidate_intent
from tests.interpretation_v2.test_specialist_runtime_pipeline import (
    _Scripted, _code_of)

_REPO = Path(__file__).resolve().parents[2]
_FIXTURE = str(_REPO / "tests" / "fixtures" / "pipeline_transition_2w")
_CLIENT = "client_001"
_RECORDS = (_REPO / "due_diligence" / "evidence"
            / "mi_135_release_certification_v3" / "raw_records.json")


def _recorded():
    """`[(question_id, question, intent payload)]` for every recorded SM case."""
    rows = []
    for record in json.loads(_RECORDS.read_text())["records"]:
        if not str(record.get("question_id", "")).startswith("SM"):
            continue
        payload = json.loads(json.dumps(
            record["record"]["interpretation"]["candidate_intent"]))
        payload.pop("provenance", None)
        payload.get("time", {}).pop("stated", None)
        rows.append((record["question_id"], record["question"], payload))
    return rows


_CASES = _recorded()
#: The one recorded reading this runtime refuses: "What number of offers
#: reached Completion?" came back as ARRIVALS with an ORIGIN stage. A new
#: arrival has no origin, so the plan asks for two different things; legacy
#: answered it as a transition and was adjudicated PARTIALLY_CORRECT.
_REFUSED = {"SM03C": stage_rt.FILTERS_NOT_SUPPORTED}


def _plan(payload):
    result = DeterministicCompiler(CompilerContext()).compile(
        parse_candidate_intent(dict(payload)))
    assert result.plan is not None, [r.code for r in result.reasons]
    return result.plan.to_dict()


@pytest.fixture(scope="module")
def payload():
    from mi_agent_api import movement_detail
    return movement_detail.resolve_stage_transition_detail(_FIXTURE, _CLIENT)


def _money(value):
    from mi_agent_api import currency
    return currency.format_money(value, suffixes=("bn", "m", "k"))


def test_the_bank_carries_the_stage_movement_family():
    assert len(_CASES) == 27


def test_the_runtime_is_registered_above_the_funded_gate():
    assert stage_rt in registry.POPULATION_OWNING_RUNTIMES
    assert stage_rt.EXECUTABLE_POPULATIONS == frozenset({"pipeline"})
    assert registry.FUNDED_GATE_POPULATIONS == frozenset({"funded"})


@pytest.mark.parametrize("qid, question, intent", _CASES,
                         ids=[c[0] for c in _CASES])
def test_each_recorded_plan_reads_as_the_legacy_route_reads_the_sentence(
        qid, question, intent):
    from mi_agent_api import stage_movement_query as legacy

    plan = _plan(intent)
    ok, why, detail = stage_rt.check_eligibility(plan)
    if qid in _REFUSED:
        assert (ok, why) == (False, _REFUSED[qid]), detail
        return
    assert ok, f"{qid}: {why}: {detail}"
    assert stage_rt.reading_for(plan) == legacy.read(question).to_dict()


@pytest.mark.parametrize("qid, question, intent",
                         [c for c in _CASES if c[0] not in _REFUSED],
                         ids=[c[0] for c in _CASES if c[0] not in _REFUSED])
def test_each_governed_answer_is_the_legacy_answer(qid, question, intent,
                                                   payload):
    from mi_agent_api import stage_movement_query as legacy

    outcome = stage_rt.execute(_plan(intent), root=_FIXTURE, client_id=_CLIENT)
    assert outcome.ok, outcome.detail
    expected, rows, refusal = legacy.compose(legacy.read(question), payload,
                                             money=_money)
    assert refusal is None
    assert outcome.answer == expected
    assert outcome.rows == [dict(r) for r in rows]


def test_the_arrival_with_an_origin_names_why_it_is_refused():
    intent = dict(next(c for c in _CASES if c[0] == "SM03C")[2])
    detail = stage_rt.check_eligibility(_plan(intent))[2]
    assert "a new arrival has no origin stage" in detail


def test_the_receipt_proves_the_stages_it_was_selected_on():
    intent = next(c for c in _CASES if c[0] == "SM01A")[2]
    receipt = stage_rt.execute(_plan(intent), root=_FIXTURE,
                               client_id=_CLIENT).receipt
    assert receipt["applied_predicates"] == [
        {"field": "origin_stage", "canonical_field": "origin_stage", "op": "eq",
         "values": ["KFI"]},
        {"field": "destination_stage", "canonical_field": "destination_stage",
         "op": "eq", "values": ["APPLICATION"]}]
    assert receipt["dataset"]["as_of_date"] and receipt["dataset"]["comparison_date"]


@pytest.mark.parametrize("over, reason", [
    ({"population": {"base": "funded"}}, stage_rt.POPULATION_NOT_PIPELINE),
    ({"time": {"form": "relative_pair", "grain": "weekly", "periods_back": 1}},
     stage_rt.PERIOD_NOT_SUPPORTED),
    ({"operation": "breakdown"}, stage_rt.OPERATION_NOT_SUPPORTED),
    ({"filters": [{"concept": "origin_stage", "comparator": "eq",
                   "value": "KFI"}]}, stage_rt.FILTERS_NOT_SUPPORTED),
])
def test_unsupported_shapes_refuse_by_name(over, reason):
    intent = dict(next(c for c in _CASES if c[0] == "SM01A")[2])
    intent.update(over)
    result = DeterministicCompiler(CompilerContext()).compile(
        parse_candidate_intent(intent))
    plan = result.plan.to_dict() if result.plan is not None else intent
    assert stage_rt.check_eligibility(plan)[1] == reason


def test_the_runtime_never_reads_the_question():
    code = _code_of(stage_rt)
    for forbidden in (".read(", "recognise", "names_a_stage_movement",
                      "chat_routing", "ParsedQuestion", "llm_query_parser",
                      "question"):
        assert forbidden not in code, forbidden


def test_the_runtime_computes_nothing():
    """No numeric operator anywhere: the only `+` and `|` in the module join
    tuples, sets and strings, and every figure is the owner's."""
    tree = ast.parse(Path(stage_rt.__file__).read_text())
    numeric = (ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
    arithmetic = [ast.unparse(n) for n in ast.walk(tree)
                  if isinstance(n, ast.BinOp) and isinstance(n.op, numeric)]
    assert arithmetic == [], arithmetic


# --------------------------------------------------------------------------- #
# through serve()
# --------------------------------------------------------------------------- #

def _served(intent, monkeypatch):
    from mi_agent import plan_serving_canary as canary
    from mi_agent import plan_shadow_evidence as evidence
    from mi_agent import plan_shadow_wiring as wiring

    monkeypatch.setenv("MI_AGENT_PLAN_SERVE", "canary")
    monkeypatch.setenv("MI_AGENT_PLAN_SERVE_PRINCIPALS", "canary-principal")
    wiring.set_interpreter_factory(lambda: _Scripted(intent))
    written = []
    monkeypatch.setattr(evidence, "write", lambda body: written.append(body))

    class Principal:
        actor_id = "canary-principal"

    try:
        out = canary.serve(question="q", context=Principal(), client_id=_CLIENT,
                           run_id=None, legacy_result={"ok": True}, frame=None,
                           semantics={}, view="funded", pipeline_root=_FIXTURE,
                           pipeline_client_id=_CLIENT)
    finally:
        wiring.set_interpreter_factory(None)
    return out, (written[-1] if written else {})


@pytest.mark.parametrize("qid", ["SM01A", "SM02A", "SM05A", "SM06A", "SM07A",
                                 "SM08A", "SM09B"])
def test_serve_answers_and_the_coverage_owner_accounts_for_it(qid, monkeypatch,
                                                              payload):
    from mi_agent_api import stage_movement_query as legacy
    from mi_agent_api.mi_service import _governed_plan_coverage

    question, intent = next((c[1], c[2]) for c in _CASES if c[0] == qid)
    served, record = _served(intent, monkeypatch)
    assert served is not None, record.get("execution")
    assert record["execution"]["runtime"] == "stage_movement"
    expected, _rows, _ = legacy.compose(legacy.read(question), payload,
                                        money=_money)
    assert served["answer"] == expected
    assert payload["as_of_date"] in served["answer"]
    assert _governed_plan_coverage(served)["unaccounted"] == []
