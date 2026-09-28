"""Seventeen reruns, and the same eleven questions every time.

ERE's workflow was rerun seventeen times. A rerun resumes the orchestrator and
never repeats a completed step, so Gate 1 — the step that asks the mapping
questions — never ran again, however the operator's answers changed. And had
it run, nothing would have closed the questions it stopped asking: decisions
were only ever added, so an old question about a column since set aside would
have stayed open, holding the run in review.

Two rules now:

* the columns set aside are part of the digest that says Gate 1's inputs
  changed, so a rerun after they change runs Gate 1 again;
* when a stage finishes a pass, an open question it did not raise again is
  closed as superseded — and raised again later, it opens again. A stage that
  failed or was blocked closes nothing: it stopped, it did not change its mind.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from operations_control.contracts import DEC_OPEN, DEC_SUPERSEDED
from operations_control.engine import OpsEngine

ST_DONE, ST_REVIEW, ST_BLOCKED = "completed", "needs_review", "blocked"


class _Store:
    def __init__(self, docs: List[Dict[str, Any]]):
        self.docs = {d["decision_id"]: d for d in docs}
        self.audit: List[str] = []

    def list_decisions(self, client_id, status=None, workflow_id=None):
        return [dict(d) for d in self.docs.values()
                if (status is None or d["status"] == status)
                and (workflow_id is None or d["workflow_id"] == workflow_id)]

    def load_decision(self, client_id, decision_id):
        d = self.docs.get(decision_id)
        return dict(d) if d else None

    def save_decision(self, client_id, doc):
        self.docs[doc["decision_id"]] = dict(doc)

    def append_audit(self, client_id, action, **_):
        self.audit.append(action)


def _open(decision_id, stage="onboarding"):
    return {"decision_id": decision_id, "status": DEC_OPEN, "stage": stage,
            "workflow_id": "wf_1"}


def _raised(decision_id):
    return SimpleNamespace(decision_id=decision_id,
                           to_dict=lambda: {"decision_id": decision_id})


def _sync(store, raised, *, status=ST_REVIEW, stage="onboarding"):
    engine = SimpleNamespace(store=store)
    run = SimpleNamespace(client_id="ERE", workflow_id="wf_1")
    gar = SimpleNamespace(stage=stage, status=status,
                          decisions_required=[_raised(d) for d in raised])
    OpsEngine._sync_decisions(engine, run, gar)


class TestAQuestionNoLongerAskedIsClosed:

    def test_the_set_aside_duplicate_question_is_closed(self):
        store = _Store([_open("priority:current_valuation_amount"),
                        _open("priority:erm_product_type")])
        _sync(store, ["priority:erm_product_type"])
        assert store.docs["priority:current_valuation_amount"]["status"] == \
            DEC_SUPERSEDED
        assert store.docs["priority:erm_product_type"]["status"] == DEC_OPEN
        assert "decision_superseded" in store.audit

    def test_it_says_why(self):
        store = _Store([_open("q1")])
        _sync(store, [], status=ST_DONE)
        assert "No longer asked" in store.docs["q1"]["resolution_reason"]

    @pytest.mark.parametrize("status", [ST_BLOCKED, "failed"])
    def test_a_stage_that_stopped_closes_nothing(self, status):
        store = _Store([_open("q1")])
        _sync(store, [], status=status)
        assert store.docs["q1"]["status"] == DEC_OPEN

    def test_another_stage_s_questions_are_left_alone(self):
        store = _Store([_open("q1", stage="validation")])
        _sync(store, [], status=ST_DONE, stage="onboarding")
        assert store.docs["q1"]["status"] == DEC_OPEN

    def test_raised_again_it_opens_again(self):
        store = _Store([dict(_open("q1"), status=DEC_SUPERSEDED)])
        _sync(store, ["q1"])
        assert store.docs["q1"]["status"] == DEC_OPEN

    def test_an_answered_question_stays_answered(self):
        store = _Store([dict(_open("q1"), status="approved")])
        _sync(store, ["q1"])
        assert store.docs["q1"]["status"] == "approved"


class TestARerunAfterASetAsideRunsGateOneAgain:

    def _engine(self, set_aside, tmp_path):
        return SimpleNamespace(
            _set_aside_columns=lambda _run: set_aside,
            _approved_decisions_path=lambda _run: None,
            _staging_dir=lambda _run: tmp_path)

    def test_set_asides_change_the_digest(self, tmp_path):
        run = SimpleNamespace()
        before = OpsEngine._approved_decisions_digest(
            self._engine([], tmp_path), run)
        after = OpsEngine._approved_decisions_digest(
            self._engine([("P&I.xlsx", "Current Interest Rate")], tmp_path), run)
        assert before == "" and after != ""

    def test_the_same_set_asides_give_the_same_digest(self, tmp_path):
        pairs = [("P&I.xlsx", "Current Interest Rate")]
        run = SimpleNamespace()
        assert OpsEngine._approved_decisions_digest(
            self._engine(pairs, tmp_path), run) == \
            OpsEngine._approved_decisions_digest(self._engine(list(pairs),
                                                              tmp_path), run)
