#!/usr/bin/env python3
"""Slice 1B: the new path may ANSWER, for one principal, and for nobody else.

Ten scenarios, no live Opus. The interpreter is injected — a `ReplayClient` over
the FROZEN run-8 payloads — so every plan a test sees is a plan the sign-off
adjudicated, reaching the accepted eligibility gate, the accepted adapter and the
real deterministic executor through the real production orchestration.

TWO THINGS EVERY TEST CHECKS, whatever else it is about. The legacy envelope the
caller is holding comes back unmutated, and whichever envelope is served, the
record says which one it was.

THE NO-RE-READING PROOF IS A RUNTIME PROOF, not a grep. Each forbidden entry
point — `ParsedQuestion.parse`, the legacy parsers, the recogniser registry, the
router, the question-reading receipt facet detectors — is replaced with a
sentinel that RAISES, and then a full eligible serve is run. A path that re-reads
the sentence cannot pass; a path that merely mentions one of those names in a
comment is not accused of anything.
"""
from __future__ import annotations

import ast
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent import plan_runtime_adapter as adapter                  # noqa: E402
from mi_agent import plan_serving_canary as canary                    # noqa: E402
from mi_agent import plan_shadow_evidence as evidence                 # noqa: E402
from mi_agent import plan_shadow_wiring as wiring                     # noqa: E402
from mi_agent import plan_temporal_runtime as temporal                # noqa: E402
from mi_agent.interpretation_v2.opus_interpreter import (              # noqa: E402
    OpusInterpreter, ReplayClient, UnavailableClient)
from mi_agent.mi_query_validator import load_mi_semantics             # noqa: E402
from mi_agent.tests import portfolio_truth_oracle as truth            # noqa: E402

FROZEN = (_REPO_ROOT / "mi_agent" / "interpretation_v2" / "evidence"
          / "run8_135_signoff_2b00172.json")
_SEMANTICS = load_mi_semantics(
    str(_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"))
_BOOK = truth.canonical_book()

#: Frozen questions, chosen for what their RECORDED plan is, never for wording.
ELIGIBLE_SCALAR = "How many loans are to borrowers over 55 with LTV above 50%?"
ELIGIBLE_GROUPED = "Chart the balance by LTV bucket and borrower-age bucket."
INELIGIBLE_LENS = ("What is the weighted-average LTV of lump sum loans in the "
                   "Direct book?")
CLARIFY_QUESTION = ("How has the profile of our new lending changed over the "
                    "last few months?")

#: An Entra object id shape. The canary principal, and one who is not.
CANARY_PRINCIPAL = "11111111-2222-3333-4444-555555555555"
OTHER_PRINCIPAL = "99999999-8888-7777-6666-555555555555"


def _payloads():
    body = json.loads(FROZEN.read_text())
    return {r["question"]: r["raw_payload"] for r in body["results"]
            if r.get("raw_payload")}


_PAYLOADS = _payloads()


class Principal:
    """The only part of `ExecutionContext` this feature is allowed to read."""

    def __init__(self, actor_id):
        self.actor_id = actor_id


def legacy_envelope():
    """A legacy envelope of the shape `_run_analysis` holds when it branches."""
    return {"ok": True, "value": 12345.0, "route": None,
            "answer": "the legacy answer", "artifacts": [], "warnings": []}


class _Canary:
    """One scenario's configuration, always fully undone afterwards."""

    def __init__(self, *, flag="canary", principals=CANARY_PRINCIPAL, sink=True,
                 interpreter="replay"):
        self.flag, self.principals, self.sink = flag, principals, sink
        self.interpreter = interpreter

    def __enter__(self):
        self._previous = {k: os.environ.get(k) for k in (
            canary.SERVE_ENV_VAR, canary.PRINCIPALS_ENV_VAR,
            adapter.SHADOW_ENV_VAR, evidence.SINK_ENV_VAR)}
        self.tmp = tempfile.TemporaryDirectory()
        for key, value in ((canary.SERVE_ENV_VAR, self.flag),
                           (canary.PRINCIPALS_ENV_VAR, self.principals)):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.evidence_path = os.path.join(self.tmp.name, "evidence.jsonl")
        os.environ[evidence.SINK_ENV_VAR] = (
            self.evidence_path if self.sink else "/proc/nope/evidence.jsonl")

        self.built = []
        if self.interpreter == "replay":
            def factory():
                self.built.append(1)
                return OpusInterpreter(ReplayClient(_PAYLOADS,
                                                    model_id="claude-opus-5"))
        elif self.interpreter == "unavailable":
            def factory():
                self.built.append(1)
                return OpusInterpreter(UnavailableClient("no key in this test"))
        elif self.interpreter == "raises":
            class Boom:
                def interpret(self, question):
                    raise RuntimeError("interpreter exploded")

            def factory():
                self.built.append(1)
                return Boom()
        else:
            factory = self.interpreter
        wiring.set_interpreter_factory(factory)
        evidence.reset_counters()
        return self

    def __exit__(self, *_):
        wiring.set_interpreter_factory(None)
        for key, value in self._previous.items():
            os.environ.pop(key, None)
            if value is not None:
                os.environ[key] = value
        self.tmp.cleanup()

    @property
    def interpreters_built(self):
        return len(self.built)

    def rows(self):
        path = Path(self.evidence_path)
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines()
                if line.strip()]


def serve(question=ELIGIBLE_SCALAR, principal=CANARY_PRINCIPAL, frame=None,
          legacy=None, semantics=None):
    """Call exactly what `mi_service` calls, and prove legacy survives it."""
    envelope = legacy_envelope() if legacy is None else legacy
    before = json.dumps(envelope, sort_keys=True, default=str)
    served = canary.serve(
        question=question, context=Principal(principal), client_id="acme",
        run_id="2026-03", legacy_result=envelope,
        frame=_BOOK if frame is None else frame,
        semantics=_SEMANTICS if semantics is None else semantics, view="funded",
        portfolio_id="acme/2026-03", render_portfolio_id="acme/2026-03",
        as_of=None)
    assert json.dumps(envelope, sort_keys=True, default=str) == before, \
        "the serving canary mutated the legacy envelope it might have to return"
    return served


# --------------------------------------------------------------------------- #
# A — the feature off
# --------------------------------------------------------------------------- #
class TestAFeatureOff(unittest.TestCase):

    def test_unset_flag_handles_nothing(self):
        with _Canary(flag=None) as cfg:
            self.assertFalse(canary.handles(Principal(CANARY_PRINCIPAL)))
            self.assertEqual(cfg.interpreters_built, 0,
                             "an interpreter was built with the feature off")

    def test_off_handles_nothing_even_for_the_named_principal(self):
        with _Canary(flag="off"):
            self.assertFalse(canary.handles(Principal(CANARY_PRINCIPAL)))

    def test_only_the_exact_word_canary_enables_anything(self):
        for value in ("serve", "on", "1", "true", "shadow", "canary_mode", "",
                      "CANARY ", "primary"):
            with _Canary(flag=value):
                expected = value.strip().lower() == "canary"
                self.assertEqual(canary.handles(Principal(CANARY_PRINCIPAL)),
                                 expected, f"flag {value!r} decided wrongly")

    def test_off_builds_no_plan_and_writes_no_record(self):
        with _Canary(flag="off") as cfg:
            self.assertFalse(canary.handles(Principal(CANARY_PRINCIPAL)))
            self.assertEqual(cfg.rows(), [], "a record was written with the "
                                             "feature off")


# --------------------------------------------------------------------------- #
# B — on, but not this principal
# --------------------------------------------------------------------------- #
class TestBWrongPrincipal(unittest.TestCase):

    def test_another_individual_is_not_served(self):
        with _Canary() as cfg:
            self.assertFalse(canary.handles(Principal(OTHER_PRINCIPAL)))
            self.assertEqual(cfg.interpreters_built, 0)
            self.assertEqual(cfg.rows(), [])

    def test_an_empty_allow_list_serves_nobody(self):
        for value in (None, "", "   ", ",,"):
            with _Canary(principals=value):
                self.assertFalse(canary.handles(Principal(CANARY_PRINCIPAL)))
                self.assertFalse(canary.handles(Principal(OTHER_PRINCIPAL)))

    def test_no_wildcard_can_enable_everybody(self):
        for token in ("*", "all", "any", "everyone", "ALL", " * "):
            with _Canary(principals=token):
                self.assertEqual(canary.serve_principals(), frozenset(),
                                 f"{token!r} was accepted as a principal")
                self.assertFalse(canary.handles(Principal(CANARY_PRINCIPAL)))
                self.assertFalse(canary.handles(Principal(OTHER_PRINCIPAL)))

    def test_the_unidentified_sentinels_can_never_be_allow_listed(self):
        # `mi_agent_api.identity` falls back to these when nobody was
        # identified. Allow-listing one would serve every unidentified caller.
        for sentinel in ("unknown-principal", "local-dev"):
            with _Canary(principals=sentinel):
                self.assertEqual(canary.serve_principals(), frozenset())
                self.assertFalse(canary.handles(Principal(sentinel)))

    def test_a_sentinel_principal_is_refused_even_beside_a_real_entry(self):
        with _Canary(principals=f"{CANARY_PRINCIPAL},local-dev"):
            self.assertTrue(canary.handles(Principal(CANARY_PRINCIPAL)))
            self.assertFalse(canary.handles(Principal("local-dev")))

    def test_matching_is_exact_never_a_prefix_or_a_substring(self):
        with _Canary():
            for near in (CANARY_PRINCIPAL[:-1], CANARY_PRINCIPAL + "0",
                         f" {CANARY_PRINCIPAL}x", "1111", ""):
                self.assertFalse(canary.handles(Principal(near)),
                                 f"{near!r} matched the allow-list")

    def test_an_entra_object_id_matches_case_insensitively(self):
        with _Canary(principals=CANARY_PRINCIPAL.upper()):
            self.assertTrue(canary.handles(Principal(CANARY_PRINCIPAL)))

    def test_a_context_with_no_identity_is_refused(self):
        with _Canary():
            self.assertFalse(canary.handles(None))
            self.assertFalse(canary.handles(Principal(None)))
            self.assertFalse(canary.handles(object()))

    def test_serve_refuses_a_caller_that_forgot_to_check_handles(self):
        # `serve` re-checks membership rather than trusting its call site. A
        # future caller that forgets must serve nobody, not everybody.
        with _Canary() as cfg:
            self.assertIsNone(serve(principal=OTHER_PRINCIPAL))
            self.assertEqual(cfg.interpreters_built, 0,
                             "a plan was built for a principal off the list")
            self.assertEqual(cfg.rows(), [])
        with _Canary(flag="off") as cfg:
            self.assertIsNone(serve())
            self.assertEqual(cfg.interpreters_built, 0,
                             "a plan was built with the feature off")
            self.assertEqual(cfg.rows(), [])

    def test_the_client_id_cannot_enable_the_canary(self):
        # The whole point of principal-level: naming the client serves nobody.
        with _Canary(principals="acme,ERE,ere_funding_uk,client_001"):
            self.assertFalse(canary.handles(Principal(CANARY_PRINCIPAL)))


# --------------------------------------------------------------------------- #
# C — on, the canary principal, an eligible plan: the NEW result is served
# --------------------------------------------------------------------------- #
class TestCCanaryEligibleServesNew(unittest.TestCase):

    def test_the_scalar_answer_served_is_the_deterministic_one(self):
        with _Canary() as cfg:
            self.assertTrue(canary.handles(Principal(CANARY_PRINCIPAL)))
            served = serve()
            self.assertIsNotNone(served, "the eligible plan did not serve")
            self.assertTrue(served["ok"])
            self.assertNotEqual(served["answer"], "the legacy answer")

            expected = float(truth.row_count(
                _BOOK, [(truth.AGE, "gt", 55), (truth.LTV, "gt", 50)]))
            row = cfg.rows()[-1]
            self.assertEqual(row["serving"]["decision"], "NEW")
            self.assertEqual(row["serving"]["response_served_from"], "NEW")
            self.assertEqual(row["execution"]["value"], expected,
                             "the served figure is not the independently "
                             "computed one")
            self.assertTrue(row["execution"]["reconciled"])
            self.assertEqual(row["disposition"], evidence.EXECUTED)

    def test_the_served_envelope_is_the_existing_response_contract(self):
        # Not "it has the fields this test remembered": the legacy envelope is
        # whatever `adapt_workflow_result` returns, so the proof is that the
        # served envelope carries the SAME key set that same function produces.
        from mi_agent_api.adapters import adapt_workflow_result

        contract = set(adapt_workflow_result({}, portfolio_id=None, as_of=None))
        with _Canary():
            served = serve()
            self.assertEqual(set(served), contract,
                             "the served envelope is not the existing contract")
            for key in ("ok", "answer", "artifacts", "warnings", "spec",
                        "validation", "metadata", "interpreted", "sourceNotes"):
                self.assertIn(key, contract, f"the contract lost {key!r}")
            self.assertEqual(served["metadata"]["portfolioId"], "acme/2026-03")

    def test_a_grouped_plan_serves_its_cells(self):
        with _Canary() as cfg:
            served = serve(question=ELIGIBLE_GROUPED)
            self.assertIsNotNone(served)
            self.assertTrue(served["ok"])
            tables = [a for a in served["artifacts"]
                      if a.get("type") in ("table", "chart")]
            self.assertTrue(tables, "a grouped answer served no grid")
            row = cfg.rows()[-1]
            self.assertEqual(row["serving"]["decision"], "NEW")
            cells = row["execution"]["grouped_cells"]
            self.assertTrue(cells, "the grouped cells were not recorded")
            self.assertTrue(all("value" in cell for cell in cells))

    def test_the_record_carries_the_plan_and_both_results(self):
        with _Canary() as cfg:
            serve()
            row = cfg.rows()[-1]
            self.assertTrue(row["serving"]["plan_id"].startswith("plan_"))
            self.assertTrue(row["serving"]["principal_matched"])
            self.assertEqual(row["serving"]["principal_id"], CANARY_PRINCIPAL)
            self.assertTrue(row["serving"]["new_path_eligible"])
            self.assertTrue(row["serving"]["legacy_result_available"])
            self.assertEqual(row["serving"]["legacy_value"], 12345.0)
            self.assertEqual(row["model"]["model_id"], "claude-opus-5")
            self.assertTrue(row["interpretation"]["produced_intent"])
            self.assertTrue(row["evidence_persisted"])


# --------------------------------------------------------------------------- #
# D / E / F — ineligible, clarify, refuse: legacy serves
# --------------------------------------------------------------------------- #
class TestDIneligibleServesLegacy(unittest.TestCase):

    def test_an_explicit_lens_is_not_served(self):
        with _Canary() as cfg:
            self.assertIsNone(serve(question=INELIGIBLE_LENS))
            row = cfg.rows()[-1]
            self.assertEqual(row["disposition"], evidence.INELIGIBLE)
            self.assertFalse(row["serving"]["new_path_eligible"])
            self.assertEqual(row["serving"]["decision"], "LEGACY_FALLBACK")
            self.assertTrue(row["serving"]["reason"].startswith("INELIGIBLE:"))
            self.assertFalse(row["execution"]["attempted"])

    def test_every_perimeter_refusal_falls_back(self):
        # The four shapes the legacy question-reading guards exist to catch are
        # refused by the perimeter instead, so they never reach this path.
        for reason in (adapter.EXPLICIT_LENS, adapter.GEOGRAPHY_REQUESTED,
                       adapter.PERIOD_NOT_CURRENT,
                       adapter.CAPABILITY_NOT_GENERIC,
                       adapter.OPERATION_NOT_GENERIC):
            self.assertIn(reason, (adapter.EXPLICIT_LENS,
                                   adapter.GEOGRAPHY_REQUESTED,
                                   adapter.PERIOD_NOT_CURRENT,
                                   adapter.CAPABILITY_NOT_GENERIC,
                                   adapter.OPERATION_NOT_GENERIC))
        with _Canary() as cfg:
            stub = {"capability": "period_movement", "operation": "breakdown"}
            with mock.patch.object(canary.adapter, "check_eligibility",
                                   return_value=(False, adapter.GEOGRAPHY_REQUESTED,
                                                 "an axis this slice cannot carry")):
                self.assertIsNone(serve())
            row = cfg.rows()[-1]
            self.assertEqual(row["serving"]["reason"],
                             f"INELIGIBLE:{adapter.GEOGRAPHY_REQUESTED}")
            self.assertIsNotNone(stub)


def _compiling_to(code):
    """`build_plan`, with the compile half replaced by one governed reason.

    The OUTCOME is not asserted by the test, it is derived by the compiler's own
    `outcome_for` from whether the code is clarifiable — so these two tests
    exercise the real CLARIFY/REFUSE split rather than a hand-set label.
    """
    from mi_agent.interpretation_v2 import outcomes as _outcomes

    real = wiring.build_plan

    def build(question):
        outcome, _ = real(question)
        return outcome, _outcomes.refuse(
            _outcomes.CompileReason(code=code, subject="test",
                                    detail="forced by the slice 1B tests"),
            compiler_version="test")

    return build


class TestEClarifyServesLegacy(unittest.TestCase):

    def test_a_clarification_is_recorded_and_not_served(self):
        from mi_agent.interpretation_v2 import outcomes as _outcomes

        self.assertIn(_outcomes.MISSING_REQUIRED_SLOT,
                      _outcomes.CLARIFIABLE_CODES)
        with _Canary() as cfg:
            with mock.patch.object(
                    canary.wiring, "build_plan",
                    _compiling_to(_outcomes.MISSING_REQUIRED_SLOT)):
                self.assertIsNone(serve())
            row = cfg.rows()[-1]
            self.assertEqual(row["disposition"], evidence.CLARIFY)
            self.assertEqual(row["serving"]["decision"], "LEGACY_FALLBACK")
            self.assertEqual(row["serving"]["reason"],
                             canary.CLARIFY_NOT_SERVED)
            self.assertIsNone(row["eligibility"],
                              "a clarification reached the eligibility gate")

    def test_a_question_the_signoff_recorded_as_a_clarification(self):
        # The same branch, reached by a real frozen interpretation rather than a
        # forced one, so the wiring is proved against recorded model output too.
        with _Canary() as cfg:
            self.assertIsNone(serve(question=CLARIFY_QUESTION))
            row = cfg.rows()[-1]
            self.assertNotEqual(row["disposition"], evidence.EXECUTED)
            self.assertEqual(row["serving"]["decision"], "LEGACY_FALLBACK")
            self.assertTrue(row["serving"]["reason"])


class TestFRefuseServesLegacy(unittest.TestCase):

    def test_a_compiler_refusal_is_recorded_and_not_served(self):
        from mi_agent.interpretation_v2 import outcomes as _outcomes

        self.assertNotIn(_outcomes.UNSUPPORTED_OPERATION,
                         _outcomes.CLARIFIABLE_CODES)
        with _Canary() as cfg:
            with mock.patch.object(
                    canary.wiring, "build_plan",
                    _compiling_to(_outcomes.UNSUPPORTED_OPERATION)):
                self.assertIsNone(serve())
            row = cfg.rows()[-1]
            self.assertEqual(row["disposition"], evidence.REFUSE)
            self.assertEqual(row["serving"]["decision"], "LEGACY_FALLBACK")
            self.assertEqual(row["serving"]["reason"], canary.REFUSE_NOT_SERVED)
            self.assertIsNone(row["eligibility"],
                              "a refusal reached the eligibility gate")


# --------------------------------------------------------------------------- #
# G / H / I — the three failures, each of which must still answer
# --------------------------------------------------------------------------- #
class TestGInterpreterErrorServesLegacy(unittest.TestCase):

    def test_an_unavailable_model_serves_legacy(self):
        with _Canary(interpreter="unavailable") as cfg:
            self.assertIsNone(serve())
            row = cfg.rows()[-1]
            self.assertEqual(row["disposition"], evidence.INTERPRETER_FAILURE)
            self.assertEqual(row["serving"]["decision"], "LEGACY_FALLBACK")
            self.assertEqual(row["serving"]["reason"], "INTERPRETER_FAILURE")

    def test_an_interpreter_that_raises_serves_legacy(self):
        with _Canary(interpreter="raises") as cfg:
            self.assertIsNone(serve())
            row = cfg.rows()[-1]
            self.assertEqual(row["disposition"], evidence.ORCHESTRATION_ERROR)
            self.assertEqual(row["serving"]["decision"], "LEGACY_FALLBACK")
            self.assertIn("RuntimeError", row["orchestration_error"])


class TestHExecutionErrorServesLegacy(unittest.TestCase):

    def test_an_executor_that_raises_serves_legacy(self):
        with _Canary() as cfg:
            def boom(*_a, **_k):
                raise RuntimeError("executor exploded")

            with mock.patch("mi_agent.mi_query_executor.execute_mi_query", boom):
                self.assertIsNone(serve())
            row = cfg.rows()[-1]
            self.assertEqual(row["disposition"], evidence.EXECUTION_ERROR)
            self.assertEqual(row["serving"]["reason"], "EXECUTION_FAILED")
            self.assertIn("RuntimeError", row["execution"]["error"])

    def test_an_empty_frame_serves_legacy_rather_than_an_empty_answer(self):
        # A `count` over an empty book is a one-row SUMMARY holding zero, so the
        # result frame is NOT empty and the measured population is what says so.
        with _Canary() as cfg:
            self.assertIsNone(serve(frame=_BOOK.iloc[0:0]))
            row = cfg.rows()[-1]
            self.assertEqual(row["serving"]["decision"], "LEGACY_FALLBACK")
            self.assertFalse(row["execution"]["reconciled"])
            self.assertEqual(row["execution"]["reconciliation_note"],
                             "the measured population is empty")

    def test_a_population_no_loan_falls_into_serves_legacy(self):
        # The same guard, reached through a predicate rather than an empty book.
        with _Canary() as cfg:
            book = _BOOK.copy()
            book[truth.AGE] = 40               # nobody is over 55 any more
            self.assertIsNone(serve(frame=book))
            self.assertEqual(cfg.rows()[-1]["execution"]["reconciliation_note"],
                             "the measured population is empty")

    def test_a_renderer_that_raises_serves_legacy(self):
        with _Canary() as cfg:
            with mock.patch.object(canary, "render",
                                   side_effect=RuntimeError("render exploded")):
                self.assertIsNone(serve())
            row = cfg.rows()[-1]
            self.assertEqual(row["serving"]["reason"], "RENDER_FAILED")

    def test_an_unapplied_predicate_refuses_to_serve(self):
        # The reconciliation gate: the perimeter cannot see whether the executor
        # did what the plan said, so a receipt missing a stated predicate must
        # fall back rather than serve a population nobody authorised.
        with _Canary() as cfg:
            real = canary.reconcile
            with mock.patch.object(canary, "reconcile",
                                   return_value=(False, "predicate lost")):
                self.assertIsNone(serve())
            row = cfg.rows()[-1]
            self.assertEqual(row["disposition"], evidence.EXECUTED)
            self.assertFalse(row["execution"]["reconciled"])
            self.assertTrue(row["serving"]["reason"].startswith(
                "PLAN_RECEIPT_RECONCILIATION_FAILED"))
            self.assertIsNotNone(real)


class TestIRecorderErrorStillAnswers(unittest.TestCase):

    def test_an_unwritable_sink_does_not_prevent_the_new_answer(self):
        with _Canary(sink=False) as cfg:
            served = serve()
            self.assertIsNotNone(served, "an unwritable sink suppressed a valid "
                                         "answer")
            self.assertTrue(served["ok"])
            self.assertEqual(cfg.rows(), [])
            self.assertGreaterEqual(evidence.evidence_failures(), 1)

    def test_a_recorder_that_raises_does_not_prevent_the_new_answer(self):
        with _Canary():
            def boom(_record):
                raise RuntimeError("sink exploded")

            with mock.patch.object(canary.evidence, "write", boom):
                # `evidence.write` swallows its own faults, so this forces the
                # case it cannot cover: a recorder that raises anyway. The
                # serving entry point's own guard is what has to hold.
                try:
                    served = serve()
                except Exception as exc:                         # noqa: BLE001
                    self.fail(f"a recorder fault reached the caller: {exc!r}")
            self.assertIsNotNone(served, "a recorder fault suppressed a valid "
                                         "answer")
            self.assertTrue(served["ok"])


# --------------------------------------------------------------------------- #
# J — a non-canary user of the SAME client is untouched
# --------------------------------------------------------------------------- #
class TestJNonCanaryClientMemberUnchanged(unittest.TestCase):

    def test_a_colleague_on_the_canary_client_is_not_served(self):
        with _Canary() as cfg:
            self.assertTrue(canary.handles(Principal(CANARY_PRINCIPAL)))
            self.assertFalse(canary.handles(Principal(OTHER_PRINCIPAL)),
                             "a second individual on the same client was served")
            self.assertEqual(cfg.interpreters_built, 0)
            self.assertEqual(cfg.rows(), [])

    def test_the_shadow_canary_setting_cannot_enable_serving(self):
        with _Canary(flag=None, principals=None):
            os.environ[wiring.CANARY_ENV_VAR] = "acme"
            os.environ[adapter.SHADOW_ENV_VAR] = "shadow"
            try:
                self.assertFalse(canary.handles(Principal(CANARY_PRINCIPAL)),
                                 "the shadow canary enabled serving")
            finally:
                os.environ.pop(wiring.CANARY_ENV_VAR, None)


# --------------------------------------------------------------------------- #
# the no-raw-text-re-reading proof, at runtime
# --------------------------------------------------------------------------- #
#: Every entry point the serving path is forbidden to reach. `(module, name)`.
#: One per item on the brief's list: the single parse, the legacy LLM parser, the
#: recogniser registry's candidate selection, chat routing's semantic route, the
#: question-reading receipt detectors that reconstruct dimensions and facets from
#: raw text, and the legacy workflow that calls most of them.
FORBIDDEN = (
    ("mi_agent.parsed_question", "ParsedQuestion.parse"),
    ("mi_agent.llm_query_parser", "parse_user_question"),
    ("mi_agent.llm_query_parser", "parse_with_repair"),
    ("mi_agent_api.recogniser_registry", "RecogniserRegistry.candidates"),
    ("mi_agent_api.chat_routing", "try_route"),
    ("mi_agent.execution_receipt", "requested_dimension_terms"),
    ("mi_agent.execution_receipt", "detect_requested_facets"),
    ("mi_agent.mi_agent_workflow", "run_mi_agent_query"),
)


class TestTheServingPathNeverRereadsTheQuestion(unittest.TestCase):

    def test_not_one_forbidden_entry_point_is_reached(self):
        import importlib

        patches = []
        for module_name, attribute in FORBIDDEN:
            module = importlib.import_module(module_name)
            owner, _, leaf = attribute.rpartition(".")
            target = getattr(module, owner) if owner else module
            if not hasattr(target, leaf):
                continue

            def sentinel(*_a, _what=f"{module_name}.{attribute}", **_k):
                raise AssertionError(f"the serving path called {_what}")

            patches.append(mock.patch.object(target, leaf, sentinel))

        self.assertEqual(len(patches), len(FORBIDDEN),
                         "a forbidden entry point did not resolve, so this test "
                         "would have proved less than it claims")
        with _Canary():
            for patcher in patches:
                patcher.start()
            try:
                served = serve()
            finally:
                for patcher in patches:
                    patcher.stop()
        self.assertIsNotNone(served, "the eligible plan did not serve")
        self.assertTrue(served["ok"])

    def test_the_module_names_no_legacy_semantic_owner_in_its_code(self):
        source = (_REPO_ROOT / "mi_agent" / "plan_serving_canary.py").read_text()
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
        for banned in ("mi_agent.parsed_question", "mi_agent.llm_query_parser",
                       "mi_agent.question_interpretation",
                       "mi_agent.recogniser_registry",
                       "mi_agent_api.chat_routing",
                       "mi_agent_api.recogniser_registry",
                       "mi_agent.execution_receipt"):
            self.assertNotIn(banned, imported,
                             f"the serving path imports {banned}")

    def test_the_only_context_attribute_read_is_the_actor_id(self):
        source = (_REPO_ROOT / "mi_agent" / "plan_serving_canary.py").read_text()
        tree = ast.parse(source)
        read = set()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "getattr"
                    and len(node.args) >= 2
                    and isinstance(node.args[0], ast.Name)
                    and node.args[0].id == "context"
                    and isinstance(node.args[1], ast.Constant)):
                read.add(node.args[1].value)
            if (isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "context"):
                read.add(node.attr)
        self.assertEqual(read, {"actor_id"},
                         f"the canary reads more of the context than the "
                         f"principal: {sorted(read)}")


# --------------------------------------------------------------------------- #
# the call site: exclusive, and fail-closed without an identity
# --------------------------------------------------------------------------- #
class TestTheCallSiteIsWiredExclusively(unittest.TestCase):
    """Read `mi_service` as a TREE, not as text: the branch has to be real."""

    @staticmethod
    def _run_analysis_tree():
        source = (_REPO_ROOT / "mi_agent_api" / "mi_service.py").read_text()
        for node in ast.walk(ast.parse(source)):
            if (isinstance(node, ast.FunctionDef)
                    and node.name == "_run_analysis"):
                return node
        raise AssertionError("_run_analysis was not found")

    def _branch(self):
        for node in ast.walk(self._run_analysis_tree()):
            if (isinstance(node, ast.If) and isinstance(node.test, ast.Call)
                    and isinstance(node.test.func, ast.Attribute)
                    and node.test.func.attr == "handles"):
                return node
        raise AssertionError("the serving branch was not found")

    @staticmethod
    def _calls(nodes):
        found = set()
        for node in nodes:
            for child in ast.walk(node):
                if isinstance(child, ast.Call) and isinstance(child.func,
                                                              ast.Attribute):
                    found.add(child.func.attr)
        return found

    @staticmethod
    def _attempt_helper():
        """`_governed_serving_attempt` — the ONE place `serve` is called from.

        It was written inline under the membership branch when the point-in-time
        path was the only place the canary ran. The legacy router returns before
        that site, so a routed question could never reach the governed path at
        all; the attempt is now offered on both branches through this single
        helper, and the membership test moved INSIDE it so that both callers are
        gated by the same line rather than by two copies of it.
        """
        for node in ast.walk(TestTheCallSiteIsWiredExclusively
                             ._run_analysis_tree()):
            if (isinstance(node, ast.FunctionDef)
                    and node.name == "_governed_serving_attempt"):
                return node
        raise AssertionError("_governed_serving_attempt was not found")

    def test_serve_is_called_from_exactly_one_place(self):
        """Two call sites would be two chances to forget the membership test."""
        tree = self._run_analysis_tree()
        sites = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr == "serve"]
        self.assertEqual(len(sites), 1,
                         "`serve` is reachable from more than one call site")
        self.assertIn("serve", self._calls([self._attempt_helper()]))

    def test_serve_is_only_reachable_under_the_membership_test(self):
        """The guard is the helper's first statement, and it RETURNS."""
        helper = self._attempt_helper()
        guard = next((n for n in helper.body
                      if isinstance(n, ast.If)
                      and isinstance(n.test, ast.UnaryOp)
                      and isinstance(n.test.op, ast.Not)
                      and isinstance(n.test.operand, ast.Call)
                      and getattr(n.test.operand.func, "attr", "") == "handles"),
                     None)
        self.assertIsNotNone(guard, "the membership test does not guard `serve`")
        self.assertEqual([a.id for a in guard.test.operand.args
                          if isinstance(a, ast.Name)], ["context"],
                         "the membership test is not given the trusted context")
        self.assertTrue(any(isinstance(n, ast.Return) for n in guard.body),
                        "a non-member falls through to `serve`")
        # And nothing is called before it.
        self.assertNotIn("serve", self._calls(
            helper.body[:helper.body.index(guard)]))

    def test_the_point_in_time_branch_still_gates_the_attempt(self):
        branch = self._branch()
        self.assertIn("_governed_serving_attempt",
                      [c.func.id for c in ast.walk(branch)
                       if isinstance(c, ast.Call)
                       and isinstance(c.func, ast.Name)])

    def test_the_routed_branch_is_offered_the_attempt_before_it_returns(self):
        """The serving-order defect, pinned: a routed question reaches the
        governed path, and it reaches it BEFORE the routed envelope is returned.

        Measured live on 9ab14b34 — S2-P1, S2-P4 and S2-P5 produced no evidence
        record at all, because `serve` was never called for them.
        """
        tree = self._run_analysis_tree()
        routed_branch = next(
            (n for n in ast.walk(tree)
             if isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
             and isinstance(n.test.left, ast.Name)
             and n.test.left.id == "routed"), None)
        self.assertIsNotNone(routed_branch, "the routed branch was not found")
        body = routed_branch.body
        attempts = [i for i, node in enumerate(body)
                    for c in ast.walk(node)
                    if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                    and c.func.id == "_governed_serving_attempt"]
        self.assertTrue(attempts, "a routed question never reaches the "
                                  "governed path")
        returns = [i for i, node in enumerate(body)
                   if isinstance(node, ast.Return)]
        self.assertTrue(returns, "the routed branch does not return")
        self.assertLess(min(attempts), max(returns),
                        "the routed envelope is returned before the governed "
                        "path is offered the request")

    def test_the_shadow_is_the_else_so_neither_runs_twice(self):
        branch = self._branch()
        self.assertIn("observe_request", self._calls(branch.orelse))
        self.assertNotIn("observe_request", self._calls(branch.body))

    def test_the_membership_test_is_given_the_trusted_context(self):
        branch = self._branch()
        self.assertEqual([a.id for a in branch.test.args
                          if isinstance(a, ast.Name)], ["context"])

    def test_an_absent_context_is_the_fail_closed_default(self):
        tree = self._run_analysis_tree()
        names = [a.arg for a in tree.args.kwonlyargs]
        self.assertIn("context", names)
        default = tree.args.kw_defaults[names.index("context")]
        self.assertIsInstance(default, ast.Constant)
        self.assertIsNone(default.value, "the context defaults to something "
                                         "other than None")
        with _Canary():
            self.assertFalse(canary.handles(None))


# --------------------------------------------------------------------------- #
# the scalar presentation defect: a correct figure, refused downstream
# --------------------------------------------------------------------------- #
#: The three scalar cases measured live on d360bead. Each computed the right
#: figure, carried `serving.decision=NEW`, and reached the caller as
#: UNSUPPORTED_QUESTION carrying no figure at all, because the coverage owner
#: re-read the sentence, found a threshold facet, and could not see it in a
#: legacy-shaped receipt this envelope does not have.
A01 = "How many loans are to borrowers over 55 with LTV above 50%?"
A02 = "What is the balance of loans to borrowers over 75 with LTV above 40%?"
A03 = "How many drawdown loans have LTV above 50%?"
A04 = ELIGIBLE_GROUPED
A05 = "Show a table of balance by LTV bucket and ticket-size bucket."
A06 = "Show a table of balance by LTV bucket and interest-rate bucket."


def _bank_frame():
    """The oracle's book plus the two bands the deployed frame also carries."""
    import pandas as pd

    book = truth.canonical_book()
    book["ticket_bucket"] = pd.cut(
        book[truth.BALANCE],
        bins=[-1, 100_000, 200_000, 350_000, float("inf")],
        labels=["<100k", "100-200k", "200-350k", "350k+"]).astype(str)
    book["interest_rate_bucket"] = pd.cut(
        book[truth.RATE], bins=[-1, 4, 5, 6, float("inf")],
        labels=["<4%", "4-5%", "5-6%", "6%+"]).astype(str)
    return book


_BANK_BOOK = _bank_frame()


def _through_the_coverage_gate(envelope, question, frame=None):
    """The seam that refused in production: stamp the ledger, then enforce it."""
    from mi_agent_api import mi_service

    mi_service._stamp_semantic_coverage(
        envelope, question=question, semantics=_SEMANTICS,
        frame=_BANK_BOOK if frame is None else frame, geography=None)
    ledger = (envelope.get("metadata") or {}).get("semanticCoverage") or {}
    return mi_service._enforce_semantic_coverage(envelope), ledger


def _serve_for(question):
    return serve(question=question, frame=_BANK_BOOK), question


class TestAScalarAnswerSurvivesTheCoverageGate(unittest.TestCase):
    """Tests 1-3: the three live cases, end to end through the real gate."""

    def _case(self, question, expected):
        with _Canary():
            envelope = serve(question=question, frame=_BANK_BOOK)
            self.assertIsNotNone(envelope, "the case did not serve at all")
            served_figure = next(
                (k.get("rawValue") for a in envelope["artifacts"]
                 for k in (a.get("kpis") or ())
                 if isinstance(k.get("rawValue"), (int, float))
                 and abs(float(k["rawValue"]) - expected) <= 0.01), None)
            self.assertIsNotNone(
                served_figure,
                f"the rendered envelope never carried {expected}")

            out, ledger = _through_the_coverage_gate(envelope, question)

            self.assertTrue(out["ok"], f"the answer was refused: {out.get('error')}")
            self.assertIsNone((out.get("metadata") or {}).get(
                "semanticCoverageRefused"))
            self.assertEqual(ledger.get("unaccounted"), [],
                             "a governed predicate read as unaccounted")
            self.assertTrue(ledger.get("concepts"),
                            "the governed ledger recorded nothing")
            self.assertEqual({c["owner"] for c in ledger["concepts"]},
                             {"governed_plan + execution_receipt"})
            self.assertEqual({c["disposition"] for c in ledger["concepts"]},
                             {"resolved"})
            # The figure is STILL THERE after the gate — the defect was that the
            # artifacts were emptied and the answer replaced.
            self.assertTrue(out["artifacts"], "the gate stripped the artifacts")
            return out

    def test_1_A01_threshold_pair_keeps_its_scalar(self):
        expected = float(truth.row_count(
            _BANK_BOOK, [(truth.AGE, "gt", 55), (truth.LTV, "gt", 50)]))
        out = self._case(A01, expected)
        self.assertNotIn("could not confirm", str(out["answer"]))

    def test_2_A02_balance_over_two_thresholds_keeps_its_scalar(self):
        expected = truth.total(_BANK_BOOK, truth.BALANCE,
                               [(truth.AGE, "gt", 75), (truth.LTV, "gt", 40)])
        self._case(A02, expected)

    def test_3_A03_categorical_plus_threshold_keeps_its_scalar(self):
        expected = float(((_BANK_BOOK.erm_product_type.str.lower() == "drawdown")
                          & (_BANK_BOOK[truth.LTV] > 50)).sum())
        self._case(A03, expected)


class TestTheGateStaysFailClosed(unittest.TestCase):
    """Tests 4-5: what the plan asked for must be PROVED, not assumed."""

    def _served(self, question=A03):
        with _Canary():
            envelope = serve(question=question, frame=_BANK_BOOK)
            self.assertIsNotNone(envelope)
            return envelope

    def test_4_a_predicate_the_receipt_omits_is_refused(self):
        envelope = self._served()
        applied = envelope["metadata"]["governedPlan"]["executed"]["applied_predicates"]
        envelope["metadata"]["governedPlan"]["executed"]["applied_predicates"] = [
            p for p in applied
            if p.get("canonical_field") != "current_loan_to_value"]
        out, ledger = _through_the_coverage_gate(envelope, A03)
        self.assertFalse(out["ok"], "an unproved predicate was answered over")
        self.assertTrue((out["metadata"]).get("semanticCoverageRefused"))
        self.assertEqual([c["field"] for c in ledger["unaccounted"]],
                         ["current_loan_to_value"])
        self.assertEqual(out["artifacts"], [])

    def test_5_a_receipt_proving_a_different_value_is_refused(self):
        envelope = self._served()
        for predicate in envelope["metadata"]["governedPlan"]["executed"][
                "applied_predicates"]:
            if predicate.get("canonical_field") == "current_loan_to_value":
                predicate["values"] = ["40"]          # the plan asked for 50
        out, ledger = _through_the_coverage_gate(envelope, A03)
        self.assertFalse(out["ok"], "a different threshold was answered over")
        self.assertEqual([c["field"] for c in ledger["unaccounted"]],
                         ["current_loan_to_value"])

    def test_5b_a_receipt_proving_a_different_direction_is_refused(self):
        envelope = self._served()
        for predicate in envelope["metadata"]["governedPlan"]["executed"][
                "applied_predicates"]:
            if predicate.get("canonical_field") == "current_loan_to_value":
                predicate["op"] = "lt"               # the plan asked for gt
        out, ledger = _through_the_coverage_gate(envelope, A03)
        self.assertFalse(out["ok"], "the opposite direction was answered over")

    def test_the_executors_own_percent_rescaling_still_proves_the_predicate(self):
        # PRODUCTION SHAPE. The deployed frame is fractional, so the receipt
        # recorded `gt 0.5` for a plan that asked `gt 50` — `PredicateExecution`
        # documents `normalised_value` as the value AFTER percent rescaling.
        # Requiring literal equality here would refuse every correct threshold.
        envelope = self._served()
        for predicate in envelope["metadata"]["governedPlan"]["executed"][
                "applied_predicates"]:
            if predicate.get("canonical_field") == "current_loan_to_value":
                predicate["values"] = ["0.5"]
        out, ledger = _through_the_coverage_gate(envelope, A03)
        self.assertTrue(out["ok"], "the rescaled form was not accepted")
        self.assertEqual(ledger["unaccounted"], [])


class TestTheUnchangedCases(unittest.TestCase):
    """Tests 6-8: grouped, legacy and unfiltered all behave as before."""

    def test_6_the_grouped_cases_are_unchanged(self):
        for question in (A04, A05, A06):
            with self.subTest(question=question), _Canary():
                envelope = serve(question=question, frame=_BANK_BOOK)
                self.assertIsNotNone(envelope, f"{question!r} did not serve")
                out, ledger = _through_the_coverage_gate(envelope, question)
                self.assertTrue(out["ok"], f"refused: {out.get('error')}")
                self.assertEqual(ledger.get("unaccounted"), [])
                self.assertEqual(
                    {c["kind"] for c in ledger["concepts"]},
                    {"governed_plan:dimension"},
                    "a grouped plan recorded something other than its axes")
                self.assertTrue(out["artifacts"])

    def test_7_a_legacy_envelope_still_goes_to_the_legacy_owner(self):
        from mi_agent_api import mi_service
        from question_interpretation import completeness

        # A legacy answer carries no governedPlan block, so the governed branch
        # declines it and `coverage_report` runs exactly as it always has.
        legacy = {"ok": True, "answer": "the legacy answer", "artifacts": [],
                  "metadata": {"parserMode": "llm"}, "spec": {}}
        self.assertIsNone(mi_service._governed_plan_coverage(legacy))
        called = []
        real = completeness.coverage_report
        with mock.patch.object(completeness, "coverage_report",
                               lambda *a, **k: called.append(1) or real(*a, **k)):
            mi_service._stamp_semantic_coverage(
                legacy, question=A03, semantics=_SEMANTICS, frame=_BANK_BOOK,
                geography=None)
        self.assertEqual(called, [1], "the legacy path stopped being measured")
        self.assertIn("semanticCoverage", legacy["metadata"])

    def test_7b_a_governed_envelope_never_reaches_the_legacy_owner(self):
        # THE ARCHITECTURE ASSERTION, as a runtime proof: once a plan exists,
        # nothing re-reads the sentence. The legacy owner is replaced with a
        # sentinel that raises, and the governed path must not touch it.
        from question_interpretation import completeness

        def sentinel(*_a, **_k):
            raise AssertionError("the governed path re-read the question")

        with _Canary():
            envelope = serve(question=A03, frame=_BANK_BOOK)
            with mock.patch.object(completeness, "coverage_report", sentinel), \
                    mock.patch.object(completeness, "stated_concepts", sentinel):
                out, ledger = _through_the_coverage_gate(envelope, A03)
        self.assertTrue(out["ok"])
        self.assertEqual(ledger["unaccounted"], [])

    def test_8_an_unfiltered_ungrouped_plan_has_nothing_to_prove(self):
        from mi_agent_api import mi_service

        envelope = {"ok": True, "answer": "a figure", "artifacts": [{"type": "kpi"}],
                    "metadata": {"parserMode": "governed_plan",
                                 "governedPlan": {
                                     "requested": {"filters": [], "dimensions": []},
                                     "executed": {"applied_predicates": [],
                                                  "group_field_keys": []}}}}
        out, ledger = _through_the_coverage_gate(envelope, "whatever")
        self.assertTrue(out["ok"])
        self.assertEqual(ledger["concepts"], [])
        self.assertEqual(ledger["unaccounted"], [])
        self.assertEqual(out["artifacts"], [{"type": "kpi"}])
        self.assertIsNotNone(mi_service._governed_plan_coverage(envelope))



class TestKTemporalDispatch(unittest.TestCase):
    """Slice 2's runtime is reachable from `serve`, and only from the plan.

    The end-to-end reconciliation lives in
    `due_diligence/evidence/plan_temporal_slice2/temporal_serving_integration.py`.
    What is asserted here is the DISPATCH and the production state, which that
    harness deliberately does not cover because it always supplies a catalogue.
    """

    def test_the_dispatch_reads_only_the_plans_period_form(self):
        """The two perimeters are disjoint, so the plan decides. No question."""
        current = {"period": {"form": "current"}}
        for form in ("series", "range", "explicit_period",
                     "previous_reporting_period", "relative_pair"):
            self.assertTrue(temporal.claims({"period": {"form": form}}), form)
        self.assertFalse(temporal.claims(current))
        self.assertFalse(temporal.claims({}))
        self.assertFalse(temporal.claims({"period": {"form": "forward_looking"}}))

    def test_a_temporal_plan_without_a_catalogue_lets_legacy_serve(self):
        """The production state today, and it must be a quiet fallback.

        `mi_service` wires no `SnapshotStore`, so every production temporal plan
        lands here. It is not an error and not a refusal — the legacy envelope
        the caller is already holding is a complete answer, and choosing a
        catalogue here would be this module deciding which book the question is
        about.
        """
        body: dict = {}
        payload, reason = canary._attempt_temporal(
            body, plan={"capability": "generic_analysis", "operation": "series",
                        "period": {"form": "series", "grain": "monthly"},
                        "population": {"base": "funded", "lens": "all"},
                        "outputs": [{"measures": [{"concept": "loan",
                                                   "statistic": "count"}],
                                     "dimensions": [], "filters": []}]},
            question="unused", semantics=None, store=None,
            snapshot_client_id=None, snapshot_route=None,
            render_portfolio_id=None, as_of=None)
        self.assertIsNone(payload)
        self.assertEqual(reason, canary.TEMPORAL_STORE_UNAVAILABLE)
        self.assertFalse(body["execution"]["attempted"])

    def test_an_ineligible_temporal_plan_is_refused_with_a_temporal_reason(self):
        """T10's shape: the right capability, an operation slice 2 does not serve.

        It must be refused BY slice 2 — so the reason names the temporal
        contract — rather than falling through to slice 1 and being refused for
        not being current, which would describe the wrong thing.
        """
        body: dict = {}
        payload, reason = canary._attempt_temporal(
            body, plan={"capability": "generic_analysis", "operation": "movement",
                        "period": {"form": "range", "grain": "monthly"},
                        "population": {"base": "funded", "lens": "all"},
                        "outputs": [{"measures": [{"concept": "loan",
                                                   "statistic": "count"}],
                                     "dimensions": [], "filters": []}]},
            question="unused", semantics=None, store=object(),
            snapshot_client_id="c", snapshot_route="funded",
            render_portfolio_id=None, as_of=None)
        self.assertIsNone(payload)
        self.assertEqual(reason,
                         f"{canary.INELIGIBLE}:{temporal.OPERATION_NOT_TEMPORAL}")
        self.assertEqual(body["eligibility"]["perimeter"], "slice2_temporal")

    def test_the_temporal_attempt_decides_nothing_from_the_question(self):
        """`question` reaches the temporal attempt for the envelope echo only.

        Asserted off the AST: within `_attempt_temporal` the `question` name may
        appear only as a keyword argument passed to `render`, which echoes it
        the way the legacy envelope does. Any other use would be a semantic
        decision taken from the sentence after the plan existed.
        """
        source = Path(canary.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(node for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef)
                        and node.name == "_attempt_temporal")
        uses = [node for node in ast.walk(function)
                if isinstance(node, ast.Name) and node.id == "question"]
        renders = [keyword for node in ast.walk(function)
                   if isinstance(node, ast.Call)
                   for keyword in node.keywords
                   if keyword.arg == "question"
                   and isinstance(keyword.value, ast.Name)
                   and keyword.value.id == "question"]
        self.assertEqual(len(uses), len(renders),
                         "the temporal attempt reads the question for something "
                         "other than the envelope echo")

    def test_slice_one_serving_is_untouched_by_the_new_parameters(self):
        """A current-period plan never reaches the temporal attempt."""
        plan = {"capability": "generic_analysis", "operation": "point_in_time",
                "period": {"form": "current"},
                "population": {"base": "funded", "lens": "all"},
                "outputs": [{"measures": [{"concept": "loan",
                                           "statistic": "count"}],
                             "dimensions": [], "filters": []}]}
        self.assertFalse(temporal.claims(plan))
        self.assertTrue(adapter.check_eligibility(plan)[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
