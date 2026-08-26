"""The live ranked-movement answer is derived from the governed receipt.

WHY THESE ARE MUTATION TESTS AND NOT ASSERTIONS ON THE ANSWER. Asserting that
the prose names the right dimension proves only that the route agrees with
itself; it passed just as well when the prose, the table and
`metadata.rankedMovement` were three independent derivations of the same facts.
The discriminating question is whether the published answer MOVES when the
evidence record moves. So each test perturbs exactly one fact in the
`MovementReceipt` the route built, and requires the live response to change.

A test here that fails is not a formatting difference: it means some part of the
response is still reading a semantic fact from somewhere other than the receipt.
"""
from __future__ import annotations

import builtins
import dataclasses
import json
import shutil
from pathlib import Path

import pytest

QUESTION = ("Which two geographic region obligors added the most balance "
            "since last month?")
PORTFOLIO = "client_001/mi_2026_06"
AS_OF = "2026-06-30"


@pytest.fixture(scope="module")
def fixture_root(tmp_path_factory):
    """Two governed funded snapshots, exactly as the canary builds them."""
    from migration_phase0.compound_canary import _write_run
    from migration_phase0.route_ownership_period_change import funded_runs
    root = tmp_path_factory.mktemp("mr") / "onboarding_output"
    for run_id, reporting_date, rows, scale in funded_runs(2):
        _write_run(root, run_id, reporting_date, rows, scale)
    return root


@pytest.fixture(scope="module")
def client(fixture_root, monkeypatch_module=None):
    import os
    os.environ["MI_AGENT_ONBOARDING_OUTPUT_ROOT"] = str(fixture_root)
    os.environ["MI_AGENT_AUTH_ENABLED"] = "false"
    from fastapi.testclient import TestClient
    from mi_agent_api.app import app
    return TestClient(app)


def _ask(client, question=QUESTION):
    return client.post("/mi/query", json={"question": question,
                                          "portfolioId": PORTFOLIO,
                                          "asOfDate": AS_OF}).json()


def _evidence(response):
    """The parts of the response a reader is answered with."""
    meta = response.get("metadata") or {}
    return {
        "answer": response.get("answer"),
        "rankedMovement": meta.get("rankedMovement"),
        "movementReceipt": meta.get("movementReceipt"),
        "artifacts": [{"title": a.get("title"), "description": a.get("description"),
                       "rows": a.get("rows")}
                      for a in (response.get("artifacts") or [])],
    }


def _with_mutated_receipt(monkeypatch, mutate):
    """Route the next query through a receipt with exactly one fact changed."""
    import mi_agent_api.period_change_route as pcr
    original = pcr.movement_receipt_for

    def patched(result, intent, ranking):
        return mutate(original(result, intent, ranking))

    monkeypatch.setattr(pcr, "movement_receipt_for", patched)


# --------------------------------------------------------------------------- #
# The delivered case this is measured on
# --------------------------------------------------------------------------- #
def test_the_case_is_actually_delivered(client):
    """Non-vacuity. A mutation control over a refusal proves nothing."""
    r = _ask(client)
    assert r.get("ok") is True, r.get("answer")
    meta = r["metadata"]
    assert (meta.get("rankedMovement") or {}).get("applied") is True
    receipt = meta.get("movementReceipt")
    assert receipt, "the live response carries no movement receipt"
    assert receipt["schema"] == "movement_receipt/1"
    assert len(receipt["elements"]) == 2
    assert [e["rank"] for e in receipt["elements"]] == [1, 2]


def test_the_receipt_is_complete_and_chronological(client):
    r = _ask(client)
    receipt = r["metadata"]["movementReceipt"]
    assert receipt["startPeriod"] <= receipt["endPeriod"]
    for name in ("measure", "groupingDimension", "startPeriod", "endPeriod",
                 "rankingBasis", "rankingDirection"):
        assert receipt[name], name
    for element in receipt["elements"]:
        assert round(element["endValue"] - element["startValue"], 2) == round(
            element["absoluteMovement"], 2), element


def test_every_published_fact_agrees_with_the_receipt(client):
    """One record, three renderings — checked against each other, not asserted."""
    r = _ask(client)
    meta = r["metadata"]
    receipt, ranked = meta["movementReceipt"], meta["rankedMovement"]
    assert ranked["canonicalField"] == receipt["groupingDimension"]
    assert ranked["displayName"] == receipt["groupingDisplayName"]
    assert ranked["basis"] == receipt["rankingBasis"]
    assert ranked["direction"] == receipt["rankingDirection"]
    assert ranked["openingPeriod"] == receipt["startPeriod"]
    assert ranked["closingPeriod"] == receipt["endPeriod"]
    assert ranked["topN"] == receipt["orderingLimit"]
    assert ranked["categoriesAnalysed"] == receipt["groupsAnalysed"]
    table = next(a for a in r["artifacts"]
                 if str(a.get("title", "")).startswith("Ranked movement"))
    assert receipt["groupingDisplayName"] in table["title"]
    assert [row["rank"] for row in table["rows"]] == \
        [e["rank"] for e in receipt["elements"]]
    assert [row["category"] for row in table["rows"]] == \
        [e["groupValue"] for e in receipt["elements"]]
    assert receipt["elements"][0]["groupValue"] in r["answer"]


# --------------------------------------------------------------------------- #
# Mutation controls — each must MOVE the live response
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,mutate", [
    ("period order",
     lambda rc: dataclasses.replace(rc, start_period=rc.end_period,
                                    end_period=rc.start_period)),
    ("movement value",
     lambda rc: dataclasses.replace(rc, elements=(
         dataclasses.replace(rc.elements[0],
                             absolute_movement=-rc.elements[0].absolute_movement),
     ) + rc.elements[1:])),
    ("ranking position",
     lambda rc: dataclasses.replace(rc, elements=(
         dataclasses.replace(rc.elements[1], rank=1),
         dataclasses.replace(rc.elements[0], rank=2)))),
    ("ranking direction",
     lambda rc: dataclasses.replace(rc, ranking_direction="decrease")),
    ("grouping field",
     lambda rc: dataclasses.replace(rc, grouping_dimension="broker_channel",
                                    grouping_display_name="Broker Channel")),
    ("predicate evidence",
     lambda rc: dataclasses.replace(rc, population=dataclasses.replace(
         rc.population, predicates=(("current_loan_to_value", "gt", 50.0),),
         row_counts=(1, 1)))),
])
def test_mutating_the_receipt_moves_the_live_response(client, monkeypatch,
                                                      name, mutate):
    baseline = _evidence(_ask(client))
    _with_mutated_receipt(monkeypatch, mutate)
    mutated = _evidence(_ask(client))
    assert json.dumps(mutated, sort_keys=True, default=str) != \
        json.dumps(baseline, sort_keys=True, default=str), \
        f"mutating {name} left the live response unchanged"


def test_the_response_is_restored_once_the_mutation_is_undone(client):
    """The controls above must not have left the route in a mutated state."""
    r = _ask(client)
    assert r["metadata"]["movementReceipt"]["rankingDirection"] == "increase"
    assert r["metadata"]["rankedMovement"]["canonicalField"] == \
        "geographic_region_obligor"


# --------------------------------------------------------------------------- #
# Named mutations, checked where they must land rather than merely "somewhere"
# --------------------------------------------------------------------------- #
def test_period_order_reaches_prose_table_and_metadata(client, monkeypatch):
    before = _ask(client)
    _with_mutated_receipt(monkeypatch, lambda rc: dataclasses.replace(
        rc, start_period=rc.end_period, end_period=rc.start_period))
    after = _ask(client)
    assert after["answer"] != before["answer"]
    assert after["metadata"]["rankedMovement"]["openingPeriod"] == \
        before["metadata"]["rankedMovement"]["closingPeriod"]
    tb = next(a for a in before["artifacts"]
              if str(a.get("title", "")).startswith("Ranked movement"))
    ta = next(a for a in after["artifacts"]
              if str(a.get("title", "")).startswith("Ranked movement"))
    assert ta["description"] != tb["description"]


def test_a_mutated_grouping_field_is_caught_and_refused_not_rendered(
        client, monkeypatch):
    """The strongest outcome available, and the one that actually happens.

    Pointing the receipt at a dimension the reader did not ask for does not
    merely retitle the table: the estate's own disclosure guard reads the
    published evidence, sees that the answer ranks Broker Channel while the
    question asked for geographic region, and REFUSES the whole answer rather
    than substituting a different breakdown. That is only possible because the
    guard and the renderers are reading the same record.
    """
    before = _ask(client)
    assert before["ok"] is True
    assert "Broker Channel" not in before["answer"]
    _with_mutated_receipt(monkeypatch, lambda rc: dataclasses.replace(
        rc, grouping_dimension="broker_channel",
        grouping_display_name="Broker Channel"))
    after = _ask(client)
    assert after["ok"] is False, "a substituted dimension was rendered as an answer"
    assert "Broker Channel" in after["answer"]
    assert "not substituted" in after["answer"]
    assert not [a for a in (after["artifacts"] or [])
                if str(a.get("title", "")).startswith("Ranked movement")]


def test_ranking_position_reaches_the_prose_lead_and_the_table(client, monkeypatch):
    before = _ask(client)
    _with_mutated_receipt(monkeypatch, lambda rc: dataclasses.replace(
        rc, elements=(dataclasses.replace(rc.elements[1], rank=1),
                      dataclasses.replace(rc.elements[0], rank=2))))
    after = _ask(client)
    tb = next(a for a in before["artifacts"]
              if str(a.get("title", "")).startswith("Ranked movement"))
    ta = next(a for a in after["artifacts"]
              if str(a.get("title", "")).startswith("Ranked movement"))
    assert [r["category"] for r in ta["rows"]] == \
        list(reversed([r["category"] for r in tb["rows"]]))
    assert after["answer"] != before["answer"]


def test_predicate_evidence_is_published_and_moves(client, monkeypatch):
    before = _ask(client)
    assert before["metadata"]["movementReceipt"]["population"]["predicates"] == []
    assert before["metadata"]["movementReceipt"]["population"]["narrowed"] is False
    _with_mutated_receipt(monkeypatch, lambda rc: dataclasses.replace(
        rc, population=dataclasses.replace(
            rc.population, predicates=(("current_loan_to_value", "gt", 50.0),),
            row_counts=(1, 1))))
    after = _ask(client)
    population = after["metadata"]["movementReceipt"]["population"]
    assert population["predicates"] == [["current_loan_to_value", "gt", 50.0]]
    assert population["narrowed"] is True


# --------------------------------------------------------------------------- #
# Structural — narration cannot reach a second source, because it has none
# --------------------------------------------------------------------------- #
def _function(name):
    import ast
    src = Path("mi_agent_api/period_change_route.py").read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} is not defined")


@pytest.mark.parametrize("name", ["build_rank_answer", "_rank_rows", "_render"])
def test_the_renderers_take_no_ranking_outcome(name):
    """A parameter they do not receive is a source they cannot read."""
    import ast
    node = _function(name)
    params = {a.arg for a in node.args.args} | {a.arg for a in node.args.kwonlyargs}
    assert "ranking" not in params, f"{name} still takes the ranking outcome"


@pytest.mark.parametrize("name", ["build_rank_answer", "_rank_rows"])
def test_the_ranked_renderers_read_neither_the_question_nor_the_result(name):
    import ast
    node = _function(name)
    params = {a.arg for a in node.args.args} | {a.arg for a in node.args.kwonlyargs}
    assert "question" not in params and "result" not in params, name
    names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
    assert not ({"question", "result", "spec", "spec_dict"} & names), \
        f"{name} reads {sorted({'question', 'result', 'spec', 'spec_dict'} & names)}"


def test_the_route_never_publishes_a_second_receipt_type():
    """One evidence type. `rankedMovement` is a projection, not a rival."""
    import ast
    src = Path("mi_agent_api/period_change_route.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and "movement_receipt" in (node.module or ""):
            imported |= {a.name for a in node.names}
    assert "build_movement_receipt" in imported
    render = _function("_render")
    # Every value assigned into the rankedMovement dict must come off `receipt`.
    for node in ast.walk(render):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Subscript)
                and isinstance(t.slice, ast.Constant)
                and t.slice.value == "rankedMovement" for t in node.targets):
            names = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
            # `receipt` plus the comprehension variables. Builtins are not
            # sources of semantics; anything else named here would be.
            allowed = {"receipt", "e", "c", "r"} | set(dir(builtins))
            assert names <= allowed, \
                f"rankedMovement is built from {sorted(names - allowed)}, " \
                f"not the receipt alone"
            break
    else:
        raise AssertionError("no rankedMovement assignment found in _render")
