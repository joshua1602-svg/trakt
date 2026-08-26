"""A route that fails AFTER claiming a question must not let another answer it.

THE DEFECT THESE PIN. `chat_routing`'s dispatch loop caught every exception out
of a handler, logged a warning and `continue`d to the next candidate. A fault
injected after `period_change_analysis` had already run the governed
period-change analysis therefore produced a `temporal_compare` refusal about
ranking — a different analysis, answering as though it were the answer, with no
receipt and no trace that the claimed route had ever run.

THE BOUNDARY IS THE REGISTRY'S OWN, not a list of route names:

    recognise()  pre-claim   may fail open; a raise is skipped in `candidates`
    handle()     post-claim  declines by RETURNING None; a raise is a failure

So these tests must establish a distinction, not just an outcome. Every
deliberate fault proves `alternate route execution count == 0`, and the
pre-claim control proves the opposite — that a handler returning None still lets
the next candidate run and produce the identical answer.
"""
from __future__ import annotations

import contextlib

import pytest

RANKED = ("Which two geographic region obligors added the most balance "
          "since last month?")
MOVEMENT = "What has changed since last month?"
PORTFOLIO = "client_001/mi_2026_06"
AS_OF = "2026-06-30"


@pytest.fixture(scope="module")
def fixture_root(tmp_path_factory):
    from migration_phase0.compound_canary import _write_run
    from migration_phase0.route_ownership_period_change import funded_runs
    root = tmp_path_factory.mktemp("fc") / "onboarding_output"
    for run_id, reporting_date, rows, scale in funded_runs(2):
        _write_run(root, run_id, reporting_date, rows, scale)
    return root


@pytest.fixture(scope="module")
def client(fixture_root):
    import os
    os.environ["MI_AGENT_ONBOARDING_OUTPUT_ROOT"] = str(fixture_root)
    os.environ["MI_AGENT_AUTH_ENABLED"] = "false"
    from fastapi.testclient import TestClient
    from mi_agent_api.app import app
    return TestClient(app)


class InjectedExecutionFault(RuntimeError):
    """Deliberate. Never raised by production code."""


@contextlib.contextmanager
def watching_handlers():
    """Record which handlers are ENTERED, in order. Restores on exit.

    `Recogniser` is a frozen dataclass, so this goes through
    `object.__setattr__` and puts the originals back unconditionally — an
    instrument that leaves the registry wrapped would corrupt every test that
    ran after it.
    """
    from mi_agent_api.recogniser_registry import REGISTRY
    entered: list = []
    originals = {}
    for rec in REGISTRY.ordered():
        originals[rec.name] = rec.handle

        def handle(request, _name=rec.name, _fn=rec.handle):
            entered.append(_name)
            return _fn(request)

        object.__setattr__(rec, "handle", handle)
    try:
        yield entered
    finally:
        for rec in REGISTRY.ordered():
            if rec.name in originals:
                object.__setattr__(rec, "handle", originals[rec.name])


def _ask(client, question=RANKED):
    return client.post("/mi/query", json={"question": question,
                                          "portfolioId": PORTFOLIO,
                                          "asOfDate": AS_OF}).json()


def _route(response):
    return (response.get("metadata") or {}).get("route")


# --------------------------------------------------------------------------- #
# The route-substitution detector, used by every fault test below
# --------------------------------------------------------------------------- #
def substitution_evidence(entered, response, claimed):
    """Structural evidence, not prose comparison.

    The invariant this feeds is narrow on purpose: a route legitimately
    declining and another answering is NOT a substitution, so the check fires
    only when the claim boundary was crossed and the final answer came from
    somewhere else.
    """
    meta = response.get("metadata") or {}
    return {
        "claimed_route": claimed,
        "handlers_entered": list(entered),
        "alternate_executions": [n for n in entered if n != claimed],
        "final_route": meta.get("route"),
        "claim_boundary_crossed": bool(meta.get("claimBoundaryCrossed")),
        "execution_failure": bool(meta.get("executionFailure")),
    }


def assert_no_substitution(evidence):
    if evidence["claim_boundary_crossed"] and \
            evidence["claimed_route"] != evidence["final_route"]:
        raise AssertionError(
            f"route substitution: {evidence['claimed_route']} failed after "
            f"claiming and {evidence['final_route']} answered — {evidence}")


# --------------------------------------------------------------------------- #
# F1 — the defect that exposed this
# --------------------------------------------------------------------------- #
def test_f1_ranked_movement_receipt_failure_fails_closed(client, monkeypatch):
    import mi_agent_api.period_change_route as pcr

    baseline = _ask(client)
    assert baseline["ok"] is True and _route(baseline) == "period_change_analysis"

    fired = {"n": 0}

    def faulting(result, intent, ranking):
        fired["n"] += 1
        raise InjectedExecutionFault("after the governed analysis ran")

    monkeypatch.setattr(pcr, "movement_receipt_for", faulting)
    with watching_handlers() as entered:
        response = _ask(client)

    evidence = substitution_evidence(entered, response, "period_change_analysis")
    assert fired["n"] == 1, "the fault did not execute"
    assert evidence["claim_boundary_crossed"] is True
    assert evidence["alternate_executions"] == [], evidence
    assert "temporal_compare" not in entered
    assert evidence["final_route"] == "period_change_analysis"
    assert response["ok"] is False
    assert response["metadata"]["executionFailure"] is True
    assert_no_substitution(evidence)


# --------------------------------------------------------------------------- #
# F2 — a different claimed route. The control must be generic.
# --------------------------------------------------------------------------- #
def test_f2_a_different_route_fails_closed_the_same_way(client, monkeypatch):
    from mi_agent_api import chat_routing

    baseline = _ask(client, MOVEMENT)
    assert baseline["ok"] is True and _route(baseline) == "period_movement"

    fired = {"n": 0}
    original = chat_routing._plan.period_movement

    def faulting(*args, **kwargs):
        # EXECUTION FIRST, THEN THE FAULT. Raising instead of the analysis would
        # not prove the boundary was crossed; running it and then failing does.
        original(*args, **kwargs)
        fired["n"] += 1
        raise InjectedExecutionFault("after the movement analysis ran")

    monkeypatch.setattr(chat_routing._plan, "period_movement", faulting)
    with watching_handlers() as entered:
        response = _ask(client, MOVEMENT)

    evidence = substitution_evidence(entered, response, "period_movement")
    assert fired["n"] >= 1, "the fault did not execute"
    assert evidence["claim_boundary_crossed"] is True
    assert evidence["alternate_executions"] == [], evidence
    assert evidence["final_route"] == "period_movement"
    assert response["ok"] is False
    assert response["metadata"]["executionFailure"] is True
    assert_no_substitution(evidence)


# --------------------------------------------------------------------------- #
# F3 — the key negative control. A DECLINE is not a failure.
# --------------------------------------------------------------------------- #
def test_f3_a_recogniser_that_declines_still_falls_through(client):
    """`handle` returning None is the documented decline, and must still work."""
    from mi_agent_api.recogniser_registry import REGISTRY

    baseline = _ask(client)
    assert _route(baseline) == "period_change_analysis"

    first = REGISTRY.get("period_change_analysis")
    original = first.handle
    declined = {"n": 0}

    def declining(request):
        declined["n"] += 1
        return None  # not mine — before any execution

    try:
        object.__setattr__(first, "handle", declining)
        with watching_handlers() as entered:
            response = _ask(client)
    finally:
        object.__setattr__(first, "handle", original)

    assert declined["n"] == 1
    evidence = substitution_evidence(entered, response, "period_change_analysis")
    assert evidence["claim_boundary_crossed"] is False, evidence
    assert len(evidence["alternate_executions"]) > 0, \
        "a declining route blocked the next candidate — routing was altered"
    assert response["metadata"].get("executionFailure") is None
    assert _route(response) != "period_change_analysis"
    assert_no_substitution(evidence)


def test_f3b_the_fallthrough_answer_is_identical_to_the_baseline(client):
    """What the next route says must not change because the first declined."""
    from mi_agent_api.recogniser_registry import REGISTRY

    first = REGISTRY.get("period_change_analysis")
    original = first.handle
    try:
        object.__setattr__(first, "handle", lambda request: None)
        fell_through = _ask(client)
    finally:
        object.__setattr__(first, "handle", original)

    # The same question with the first route removed from contention entirely.
    assert fell_through["ok"] is False
    assert _route(fell_through) == "temporal_compare"
    assert fell_through["metadata"].get("executionFailure") is None
    assert "not substituted" in (fell_through.get("answer") or "")


# --------------------------------------------------------------------------- #
# F4 — a governed refusal is not an execution failure
# --------------------------------------------------------------------------- #
def test_f4_a_governed_refusal_is_unchanged(client):
    """The no-implicit-period ruling. A refusal must stay a refusal."""
    response = _ask(client, "Which region grew the most?")
    assert response["ok"] is False
    meta = response["metadata"]
    assert meta.get("executionFailure") is None, \
        "a governed refusal was converted into an internal execution failure"
    assert meta.get("claimBoundaryCrossed") is None
    assert "names no period to compare over" in (response.get("answer") or "")
    assert "I have not chosen one for you" in (response.get("answer") or "")
    assert meta.get("route") == "period_change"
    assert response.get("artifacts") == []


# --------------------------------------------------------------------------- #
# F5 — no fault. Nothing moves.
# --------------------------------------------------------------------------- #
def test_f5_a_normal_delivered_answer_is_unchanged(client):
    response = _ask(client)
    meta = response["metadata"]
    assert response["ok"] is True
    assert meta["route"] == "period_change_analysis"
    assert meta.get("executionFailure") is None
    assert meta["rankedMovement"]["applied"] is True
    assert meta["movementReceipt"]["schema"] == "movement_receipt/1"
    assert [e["rank"] for e in meta["movementReceipt"]["elements"]] == [1, 2]
    assert any(str(a.get("title", "")).startswith("Ranked movement")
               for a in response["artifacts"])


# --------------------------------------------------------------------------- #
# The control must not leak internals, and must stay route-agnostic
# --------------------------------------------------------------------------- #
def test_the_failure_answer_carries_no_internal_detail(client, monkeypatch):
    import mi_agent_api.period_change_route as pcr

    def faulting(result, intent, ranking):
        raise InjectedExecutionFault("SECRET-INTERNAL-abc123 /home/user/trakt")

    monkeypatch.setattr(pcr, "movement_receipt_for", faulting)
    response = _ask(client)
    published = repr(response)
    for leak in ("InjectedExecutionFault", "SECRET-INTERNAL-abc123",
                 "Traceback", "/home/user/trakt"):
        assert leak not in published, f"{leak!r} reached the caller"


def test_the_control_names_no_route():
    """A route-specific exception list would make this fix local to C7."""
    import ast
    from pathlib import Path
    src = Path("mi_agent_api/chat_routing.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    target = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "_execution_failure_envelope")
    literals = {n.value for n in ast.walk(target)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    for name in ("period_change_analysis", "temporal_compare", "period_change",
                 "period_movement", "geo_exposure", "evolution"):
        assert not any(name in text for text in literals), \
            f"the control special-cases {name}"


# --------------------------------------------------------------------------- #
# The detector must not read a signal that only exists once the fix is in
# --------------------------------------------------------------------------- #
def test_the_substitution_detector_derives_its_boundary_from_execution():
    """A control whose signal appears only after the fix cannot detect the bug.

    The first cut of `route_substitution_detector` took `claim_boundary_crossed`
    from `metadata.claimBoundaryCrossed` — published only by the fix. Against
    the defective tree the flag was absent, the invariant read False, and the
    detector reported "SUBSTITUTIONS 0" and exited 0 over a run in which
    `temporal_compare` had answered after `period_change_analysis` failed.
    """
    import ast
    from pathlib import Path
    src = Path("migration_phase0/route_substitution_detector.py").read_text(
        encoding="utf-8")
    tree = ast.parse(src)
    run = next(n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "run")
    boundary = None
    for node in ast.walk(run):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and \
                        key.value == "claim_boundary_crossed":
                    boundary = value
    assert boundary is not None, "the detector records no claim boundary"
    literals = {n.value for n in ast.walk(boundary)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert "claimBoundaryCrossed" not in literals, (
        "the detector's boundary signal is the flag the fix publishes; it must "
        "be derived from the run (the fault executing inside the claimed route)")
    names = {n.id for n in ast.walk(boundary) if isinstance(n, ast.Name)}
    assert {"fired", "entered", "claimed"} & names, \
        "the boundary is not derived from the detector's own execution evidence"
