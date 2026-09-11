#!/usr/bin/env python3
"""The two production defects the first Slice 2 canary exposed, pinned shut.

Measured live on 9ab14b34, six questions through the real `/mi/query`:

  DEFECT 1 — THE LEGACY ROUTER PREEMPTED THE CANARY. S2-P1, S2-P4 and S2-P5
  produced no evidence record at all, because `plan_serving_canary.serve` was
  never called for them. `_run_analysis` ran the canary at ONE site, at the foot
  of the point-in-time branch, and the legacy chat router returns before it — so
  every shape the router claimed (trend, evolution, period comparison: precisely
  the temporal ones) could never reach the governed path, whatever the flag said.

  DEFECT 2 — THE GOVERNED COVERAGE OWNER COULD NOT READ A SERIES. S2-P3 WAS
  served: decision NEW, three production snapshots, the `erm_product_type`
  filter in every execution receipt. The caller still received
  UNSUPPORTED_QUESTION with no rows, because `_governed_plan_coverage` read
  `executed["applied_predicates"]` at the TOP LEVEL and a temporal answer keeps
  its receipts PER SNAPSHOT — so every requested predicate and axis read as
  unaccounted and the gate refused a correctly executed series.

  This is NOT the raw-text reread it first looked like. `render` stamps
  `parser_mode="governed_plan"`, so `completeness.coverage_report` is never
  reached on this path; the governed reconciler ran and could not see evidence
  that was one level down. That is why the repair is inside it rather than a
  bypass of it.

NO MODEL CALL ANYWHERE IN THIS FILE. The temporal cases compile a frozen
CandidateIntent with the real `DeterministicCompiler` and execute it against the
real fixture store, so the plan and the receipts are the product's own.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from mi_agent import plan_serving_canary as canary
from mi_agent import plan_temporal_runtime as temporal
from mi_agent.interpretation_v2.compiler import DeterministicCompiler
from mi_agent.interpretation_v2.intent import parse_candidate_intent
from mi_agent.mi_query_validator import load_mi_semantics
from mi_agent.tests import temporal_snapshot_fixture as fixture
from mi_agent_api import mi_service
from mi_agent_api.app import app

_REGISTRY = (Path(__file__).resolve().parents[2] / "mi_agent"
             / "mi_semantics_field_registry.yaml")

#: The three live questions the legacy router claimed, verbatim.
S2_P1 = "What was funded balance each month?"
S2_P4 = "Show loan count by LTV bucket each month."
S2_P5 = "What was funded balance this month versus last month?"

#: What the legacy router returns for them today — the envelope that used to
#: become the answer before the governed path was ever offered the request.
def routed_envelope():
    return {"ok": True, "answer": "the legacy routed answer", "artifacts": [],
            "assumptions": [], "warnings": [], "diagnostics": [],
            "sourceNotes": [], "metadata": {"route": "evolution"}}


#: What the governed path returns when it claims one.
#
# IT CARRIES THE GOVERNED BLOCK, and that is not decoration. `render` stamps
# `parserMode` and `governedPlan` on every served answer, and they are what make
# `_stamp_semantic_coverage` reconcile two governed objects instead of re-reading
# the sentence. Written without them, this stub took the LEGACY coverage branch
# and was refused for concepts it had never claimed to carry — a faithful
# demonstration of why the block exists, and the reason it is spelled out here
# rather than left out for brevity.
def governed_envelope():
    return {"ok": True, "answer": "the governed temporal answer",
            "artifacts": [], "assumptions": [], "warnings": [],
            "diagnostics": [], "sourceNotes": [],
            "metadata": {"servedFrom": "NEW", "parserMode": "governed_plan",
                         "governedPlan": {
                             "requested": {"filters": [], "dimensions": []},
                             "executed": {"snapshots": [
                                 {"applied_predicates": [],
                                  "group_field_keys": []}]}}}}


class _Routed:
    """The legacy router claims this question; the canary is configured or not.

    Patches the two seams and nothing else: what the router returns, and whether
    the principal is allow-listed. The serving-order decision under test is the
    product's own.
    """

    def __init__(self, *, handles=True, served=governed_envelope):
        self.handles, self._served = handles, served
        self.calls = []

    def _serve(self, **kwargs):
        self.calls.append(kwargs)
        return None if self._served is None else self._served()

    def __enter__(self):
        self._patches = [
            mock.patch.object(mi_service.chat_routing_mod, "try_route",
                              return_value=routed_envelope()),
            mock.patch.object(canary, "handles", return_value=self.handles),
            mock.patch.object(canary, "serve", side_effect=self._serve),
        ]
        for patch in self._patches:
            patch.start()
        return self

    def __exit__(self, *exc):
        for patch in self._patches:
            patch.stop()
        return False


def ask(question):
    with TestClient(app, raise_server_exceptions=False) as client:
        return client.post("/mi/query", json={"question": question}).json()


# --------------------------------------------------------------------------- #
# DEFECT 1 — the governed path gets the first opportunity
# --------------------------------------------------------------------------- #
class TestTheGovernedPathIsOfferedARoutedQuestion(unittest.TestCase):
    """Controls 1-3: the three questions that never reached `serve` at all."""

    def _case(self, question):
        with _Routed() as cfg:
            body = ask(question)
        self.assertEqual(len(cfg.calls), 1,
                         "the governed path was not offered this request")
        self.assertEqual(cfg.calls[0]["question"], question)
        # The LEGACY envelope is still computed and handed over as the fallback:
        # precedence means the governed result may replace it, not that the
        # legacy answer is skipped.
        # IDENTIFIED BY ITS ROUTE, not by its prose. The post-routing guards
        # run before this attempt and may rewrite `answer` — S2-P4's does,
        # refusing the routed envelope outright — and the fallback handed over
        # is deliberately the GUARDED envelope, which is the one that would have
        # been served. `metadata.route` is what proves it is that envelope and
        # not a fresh one.
        self.assertEqual(
            (cfg.calls[0]["legacy_result"].get("metadata") or {}).get("route"),
            "evolution")
        self.assertEqual(body.get("answer"), "the governed temporal answer",
                         "the routed envelope was still authoritative")
        return body

    def test_1_S2_P1_reaches_the_governed_path(self):
        self._case(S2_P1)

    def test_2_S2_P4_reaches_the_governed_path(self):
        self._case(S2_P4)

    def test_3_S2_P5_reaches_the_governed_path(self):
        self._case(S2_P5)

    def test_the_catalogue_is_supplied_on_the_routed_branch_too(self):
        """A temporal plan cannot resolve a snapshot without one."""
        with _Routed() as cfg:
            ask(S2_P1)
        self.assertIn("snapshot_store", cfg.calls[0])
        self.assertEqual(cfg.calls[0]["snapshot_route"], "funded")
        self.assertTrue(cfg.calls[0]["snapshot_client_id"])


class TestLegacyBehaviourIsUnchanged(unittest.TestCase):
    """Controls 7-8: everyone else, and a request the governed path declines."""

    def test_7_a_non_canary_principal_never_reaches_the_governed_path(self):
        with _Routed(handles=False) as cfg:
            body = ask(S2_P1)
        self.assertEqual(cfg.calls, [], "serve ran for a non-canary principal")
        self.assertNotEqual(body.get("answer"), "the governed temporal answer")
        self.assertEqual((body.get("metadata") or {}).get("route"), "evolution")

    def test_8_a_declined_request_keeps_the_routed_answer(self):
        """Ineligible, unresolvable, or any failure: `serve` returns None."""
        with _Routed(served=None) as cfg:
            body = ask(S2_P1)
        self.assertEqual(len(cfg.calls), 1, "the attempt was not made")
        self.assertNotEqual(body.get("answer"), "the governed temporal answer",
                            "a declined attempt still served the governed path")
        self.assertEqual((body.get("metadata") or {}).get("route"), "evolution")

    def test_the_routed_answer_is_identical_with_and_without_the_canary(self):
        with _Routed(handles=False):
            without = ask(S2_P1)
        with _Routed(served=None):
            declined = ask(S2_P1)
        for key in ("ok", "answer", "artifacts"):
            self.assertEqual(without.get(key), declined.get(key), key)


# --------------------------------------------------------------------------- #
# DEFECT 2 — the coverage owner reads per-snapshot receipts
# --------------------------------------------------------------------------- #
BASE_INTENT = {
    "schema_version": "candidate_intent/1.0",
    "capability": "generic_analysis", "operation": "series",
    "population": {"base": "funded", "lens": "all", "seasoning": "any"},
    "measures": [{"concept": "loan", "statistic": "count"}],
    "dimensions": [],
    "filters": [{"concept": "erm_product_type", "comparator": "eq",
                 "value": "drawdown"}],
    "geography": {"requested": False}, "comparison": {"kind": "none"},
    "time": {"form": "series", "grain": "monthly"},
}


class TemporalCoverage(unittest.TestCase):
    """Controls 4-6 and 10, on evidence the real temporal runtime produced."""

    @classmethod
    def setUpClass(cls):
        cls.semantics = load_mi_semantics(str(_REGISTRY))
        cls.history = fixture.default_history()
        cls._tmp = tempfile.TemporaryDirectory()
        cls.store = fixture.build_store(Path(cls._tmp.name) / "snaps",
                                        cls.history)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def outcome(self, **overrides):
        payload = dict(BASE_INTENT)
        payload.update(overrides)
        compiled = DeterministicCompiler().compile(parse_candidate_intent(payload))
        self.assertTrue(compiled.is_plan,
                        f"the frozen intent did not compile: {compiled.codes()}")
        return temporal.execute_temporal_plan(
            compiled.plan.to_dict(), store=self.store,
            client_id=fixture.CLIENT_ID, semantics=self.semantics,
            route=fixture.ROUTE)

    def envelope_for(self, outcome):
        """The envelope `render` builds, in the part the coverage owner reads."""
        return {"ok": True,
                "artifacts": [{"rows": [{"reporting_date": p.reporting_date,
                                         "loan_count": p.value}
                                        for p in outcome.points]}],
                "metadata": {
                    "parserMode": "governed_plan",
                    "governedPlan": {
                        "requested": dict(outcome.requested),
                        "executed": temporal.served_evidence(outcome)}}}

    def through_the_gate(self, envelope):
        mi_service._stamp_semantic_coverage(
            envelope, question="(never read on this path)",
            semantics=self.semantics, frame=None, geography=None)
        ledger = (envelope.get("metadata") or {}).get("semanticCoverage") or {}
        return mi_service._enforce_semantic_coverage(envelope), ledger

    # -- 4: the live case ---------------------------------------------------- #
    def test_4_a_filtered_series_keeps_its_rows(self):
        outcome = self.outcome()
        self.assertTrue(outcome.executed and len(outcome.points) > 1)
        envelope = self.envelope_for(outcome)
        served_rows = len(envelope["artifacts"][0]["rows"])
        out, ledger = self.through_the_gate(envelope)

        self.assertTrue(out["ok"], f"the series was refused: {out.get('error')}")
        self.assertIsNone((out.get("metadata") or {}).get(
            "semanticCoverageRefused"))
        self.assertEqual(ledger.get("unaccounted"), [])
        self.assertEqual([c["field"] for c in ledger["concepts"]],
                         ["erm_product_type"])
        self.assertEqual({c["owner"] for c in ledger["concepts"]},
                         {"governed_plan + execution_receipt"})
        self.assertEqual(len(out["artifacts"][0]["rows"]), served_rows,
                         "the gate stripped the temporal rows")

    def test_the_evidence_really_is_per_snapshot(self):
        """The shape the top-level read could not see. If this ever moves to the
        top level the repair is still correct, but this test says which world we
        are in."""
        executed = temporal.served_evidence(self.outcome())
        self.assertNotIn("applied_predicates", executed)
        self.assertTrue(executed["snapshots"])
        for snapshot in executed["snapshots"]:
            self.assertTrue(snapshot["applied_predicates"])

    def test_a_top_level_only_read_would_have_refused(self):
        """The defect itself, stated as an assertion rather than a memory."""
        executed = temporal.served_evidence(self.outcome())
        self.assertEqual(list(executed.get("applied_predicates") or ()), [],
                         "a top-level read finds no predicate — which is why "
                         "every requested filter read as unaccounted")

    # -- 5: one snapshot omits the predicate --------------------------------- #
    def test_5_a_snapshot_missing_the_predicate_refuses(self):
        envelope = self.envelope_for(self.outcome())
        snapshots = envelope["metadata"]["governedPlan"]["executed"]["snapshots"]
        snapshots[-1]["applied_predicates"] = []
        out, ledger = self.through_the_gate(envelope)
        self.assertFalse(out["ok"], "a series with an unfiltered period served")
        self.assertTrue(out["metadata"]["semanticCoverageRefused"])
        self.assertEqual([c["field"] for c in ledger["unaccounted"]],
                         ["erm_product_type"])
        self.assertEqual(out["artifacts"], [])

    # -- 6: one snapshot contradicts the predicate --------------------------- #
    def test_6_a_snapshot_proving_a_different_value_refuses(self):
        envelope = self.envelope_for(self.outcome())
        snapshots = envelope["metadata"]["governedPlan"]["executed"]["snapshots"]
        for predicate in snapshots[0]["applied_predicates"]:
            predicate["values"] = ["lump sum"]       # the plan asked for drawdown
        out, ledger = self.through_the_gate(envelope)
        self.assertFalse(out["ok"], "a contradicted predicate was answered over")
        self.assertEqual([c["field"] for c in ledger["unaccounted"]],
                         ["erm_product_type"])

    def test_6b_a_snapshot_proving_a_different_direction_refuses(self):
        envelope = self.envelope_for(self.outcome())
        snapshots = envelope["metadata"]["governedPlan"]["executed"]["snapshots"]
        for predicate in snapshots[0]["applied_predicates"]:
            predicate["op"] = "ne"
        out, _ = self.through_the_gate(envelope)
        self.assertFalse(out["ok"], "a reversed comparator was answered over")

    def test_no_receipt_at_all_refuses(self):
        """Fail-closed: an empty receipt list proves nothing and must not pass
        vacuously, which is what `all()` over an empty sequence would do."""
        envelope = self.envelope_for(self.outcome())
        envelope["metadata"]["governedPlan"]["executed"]["snapshots"] = []
        out, _ = self.through_the_gate(envelope)
        self.assertFalse(out["ok"], "an answer with no receipt was served")

    # -- a grouped series: the axis must hold on every period ---------------- #
    def test_a_grouped_series_proves_its_axis_on_every_period(self):
        outcome = self.outcome(
            filters=[], dimensions=["ltv_bucket"])
        if not outcome.executed:                       # axis not in this registry
            self.skipTest(f"grouped series unavailable: {outcome.reason}")
        envelope = self.envelope_for(outcome)
        out, ledger = self.through_the_gate(envelope)
        self.assertTrue(out["ok"], f"refused: {out.get('error')}")
        self.assertEqual(ledger.get("unaccounted"), [])

        envelope = self.envelope_for(outcome)
        envelope["metadata"]["governedPlan"]["executed"][
            "snapshots"][-1]["group_field_keys"] = []
        out, _ = self.through_the_gate(envelope)
        self.assertFalse(out["ok"], "a period grouped on nothing was served")

    # -- 10: the overlong window still fails closed -------------------------- #
    def test_10_an_unavailable_window_still_fails_closed(self):
        outcome = self.outcome(
            time={"form": "series", "grain": "monthly", "periods_back": 120})
        self.assertFalse(outcome.executed,
                         "a 120-month window executed against 8 snapshots")
        self.assertTrue(outcome.reason, "no reason was recorded for the refusal")
        self.assertEqual(list(outcome.points), [],
                         "the window was shortened to what was available")


# --------------------------------------------------------------------------- #
# a slice 1 answer is untouched by the coverage repair
# --------------------------------------------------------------------------- #
class TestSliceOneShapeIsUnchanged(unittest.TestCase):
    """Control 9's unit half: one execution is the one-receipt case of the rule.

    The end-to-end slice 1 proof is `mi_agent/tests/test_plan_serving_canary.py`,
    which is run unchanged.
    """

    def test_a_top_level_receipt_is_still_read(self):
        executed = {"applied_predicates": [
            {"canonical_field": "current_loan_to_value", "op": "gt",
             "values": [50]}], "group_field_keys": []}
        self.assertEqual(mi_service._execution_receipts(executed), [executed])

    def test_a_slice_one_envelope_still_resolves(self):
        envelope = {"ok": True, "artifacts": [{"rows": []}], "metadata": {
            "parserMode": "governed_plan",
            "governedPlan": {
                "requested": {"filters": [
                    {"field": "current_loan_to_value", "comparator": "gt",
                     "value": 50}], "dimensions": []},
                "executed": {"applied_predicates": [
                    {"canonical_field": "current_loan_to_value", "op": "gt",
                     "values": [0.5]}], "group_field_keys": []}}}}
        ledger = mi_service._governed_plan_coverage(envelope)
        self.assertEqual(ledger["unaccounted"], [],
                         "the executor's own percent rescaling stopped proving "
                         "a slice 1 threshold")

    def test_a_legacy_envelope_is_still_not_a_governed_one(self):
        self.assertIsNone(mi_service._governed_plan_coverage(
            {"ok": True, "metadata": {"parserMode": "legacy"}}))


# --------------------------------------------------------------------------- #
# the architectural assertion: no raw text decides anything after the plan
# --------------------------------------------------------------------------- #
class TestNoRawTextRereadAfterThePlan(unittest.TestCase):
    """RAW_TEXT_TEMPORAL_REREADS = 0 and RAW_TEXT_FILTER_REREADS = 0.

    Stated as an executable assertion rather than a claim. `coverage_report` is
    the raw-text owner — its requested side is `stated_concepts(question)` — and
    for a governed answer it must never run: the requested side is the plan's own
    transcription and the executed side the deterministic receipts.

    This is what the repair PRESERVED. It would have been just as easy to fix
    S2-P3 by exempting temporal answers from the gate, and that would have
    traded a false refusal for an unchecked answer.
    """

    def _governed(self, executed):
        return {"ok": True, "artifacts": [], "metadata": {
            "parserMode": "governed_plan",
            "governedPlan": {"requested": {"filters": [], "dimensions": []},
                             "executed": executed}}}

    def _stamp_without_raw_text(self, envelope):
        from question_interpretation import completeness
        with mock.patch.object(completeness, "coverage_report",
                               side_effect=AssertionError(
                                   "the sentence was re-read after the plan "
                                   "existed")) as raw:
            mi_service._stamp_semantic_coverage(
                envelope, question="how many drawdown loans each month?",
                semantics={}, frame=None, geography=None)
        return raw

    def test_a_temporal_answer_is_never_re_read_from_the_sentence(self):
        raw = self._stamp_without_raw_text(self._governed(
            {"snapshots": [{"applied_predicates": [], "group_field_keys": []}]}))
        self.assertEqual(raw.call_count, 0)

    def test_a_slice_one_answer_is_never_re_read_either(self):
        raw = self._stamp_without_raw_text(self._governed(
            {"applied_predicates": [], "group_field_keys": []}))
        self.assertEqual(raw.call_count, 0)

    def test_the_repair_did_not_exempt_temporal_answers_from_the_gate(self):
        """The gate still consumes the ledger, and the ledger still refuses."""
        envelope = self._governed({"snapshots": [
            {"applied_predicates": [], "group_field_keys": []}]})
        envelope["metadata"]["governedPlan"]["requested"]["filters"] = [
            {"field": "erm_product_type", "comparator": "eq",
             "value": "drawdown"}]
        mi_service._stamp_semantic_coverage(
            envelope, question="(never read on this path)", semantics={},
            frame=None, geography=None)
        ledger = envelope["metadata"]["semanticCoverage"]
        self.assertEqual([c["field"] for c in ledger["unaccounted"]],
                         ["erm_product_type"])
        self.assertFalse(mi_service._enforce_semantic_coverage(envelope)["ok"],
                         "a predicate no receipt proves was answered over")


# --------------------------------------------------------------------------- #
# SLICE 3 — the governed role scope, through the same coverage owner
# --------------------------------------------------------------------------- #
class TestTheRequestedScopeMustReachExecution(unittest.TestCase):
    """Requested scope = the plan. Executed scope = the receipts. Nothing else.

    `reconcile_receipt` is deliberately the STRUCTURAL half — it asks whether a
    predicate on the field ran, not which value it compared — so the claim that
    a receipt proving `direct` cannot answer a question asking `acquired` is
    owned here, by the same requested-vs-executed reconciliation that adjudicates
    every other governed predicate. Slice 3 adds no second mechanism for it; it
    only has to put the scope on the requested side, which `plan_predicates`
    does.
    """

    ROLE_FIELD = "source_portfolio_type"

    def _requested(self, role="acquired"):
        return {"filters": [{"field": self.ROLE_FIELD, "comparator": "eq",
                             "value": role}], "dimensions": []}

    def _gate(self, executed, role="acquired"):
        envelope = {"ok": True, "artifacts": [{"rows": []}], "metadata": {
            "parserMode": "governed_plan",
            "governedPlan": {"requested": self._requested(role),
                             "executed": executed}}}
        mi_service._stamp_semantic_coverage(
            envelope, question="(never read on this path)", semantics={},
            frame=None, geography=None)
        ledger = envelope["metadata"]["semanticCoverage"]
        return mi_service._enforce_semantic_coverage(envelope), ledger

    def _receipt(self, role):
        return [{"canonical_field": self.ROLE_FIELD, "op": "eq",
                 "values": [role]}]

    def test_the_role_that_ran_is_the_role_that_was_asked(self):
        out, ledger = self._gate({"applied_predicates": self._receipt("acquired"),
                                  "group_field_keys": []})
        self.assertTrue(out["ok"], out.get("error"))
        self.assertEqual(ledger["unaccounted"], [])

    def test_a_receipt_proving_a_different_role_is_refused(self):
        """Answering the DIRECT book under a question about the ACQUIRED one is
        the worst outcome this axis can produce: plausible, wrong, and silent."""
        out, ledger = self._gate({"applied_predicates": self._receipt("direct"),
                                  "group_field_keys": []})
        self.assertFalse(out["ok"], "a different book was served")
        self.assertEqual([c["field"] for c in ledger["unaccounted"]],
                         [self.ROLE_FIELD])

    def test_a_receipt_with_no_scope_at_all_is_refused(self):
        out, ledger = self._gate({"applied_predicates": [],
                                  "group_field_keys": []})
        self.assertFalse(out["ok"], "an unscoped total was served as a role")
        self.assertEqual([c["field"] for c in ledger["unaccounted"]],
                         [self.ROLE_FIELD])

    def test_one_snapshot_losing_the_scope_refuses_the_whole_series(self):
        """Slice 2's per-snapshot rule, doing Slice 3's work unchanged."""
        out, _ = self._gate({"snapshots": [
            {"applied_predicates": self._receipt("acquired"),
             "group_field_keys": []},
            {"applied_predicates": self._receipt("acquired"),
             "group_field_keys": []},
            {"applied_predicates": [], "group_field_keys": []}]})
        self.assertFalse(out["ok"], "a period without the scope was served")

    def test_a_series_scoped_in_every_snapshot_is_served(self):
        out, ledger = self._gate({"snapshots": [
            {"applied_predicates": self._receipt("acquired"),
             "group_field_keys": []} for _ in range(3)]})
        self.assertTrue(out["ok"], out.get("error"))
        self.assertEqual(ledger["unaccounted"], [])


if __name__ == "__main__":
    unittest.main()
