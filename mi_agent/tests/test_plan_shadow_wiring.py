#!/usr/bin/env python3
"""The wiring: it runs only where it is allowed to, and never on the answer.

Nine scenarios, no live Opus. The interpreter is injected — a `ReplayClient` over
the FROZEN run-8 payloads, so the plan a test sees is the plan the sign-off
adjudicated, built through the real production orchestration rather than a
hand-made dict.

The thing every one of them checks, whatever else they are about, is that the
served envelope comes back byte-identical.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent import plan_runtime_adapter as adapter                  # noqa: E402
from mi_agent import plan_shadow_evidence as evidence                 # noqa: E402
from mi_agent import plan_shadow_wiring as wiring                    # noqa: E402
from mi_agent.interpretation_v2.opus_interpreter import (             # noqa: E402
    OpusInterpreter, ReplayClient, UnavailableClient)
from mi_agent.mi_query_validator import load_mi_semantics            # noqa: E402
from mi_agent.tests import portfolio_truth_oracle as truth           # noqa: E402

FROZEN = (_REPO_ROOT / "mi_agent" / "interpretation_v2" / "evidence"
          / "run8_135_signoff_2b00172.json")
_SEMANTICS = load_mi_semantics(
    str(_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"))
_BOOK = truth.canonical_book()

#: Frozen questions, chosen for what their recorded plan is, not for wording.
ELIGIBLE_SCALAR = "How many loans are to borrowers over 55 with LTV above 50%?"
ELIGIBLE_GROUPED = "Chart the balance by LTV bucket and borrower-age bucket."
INELIGIBLE_LENS = "What is the weighted-average LTV of lump sum loans in the Direct book?"
CLARIFY_QUESTION = ("How has the profile of our new lending changed over the "
                    "last few months?")

CANARY_CLIENT = "acme"
OTHER_CLIENT = "beta"


def _code_of(source: str) -> str:
    """Source with docstrings and comments stripped, so prose cannot fail a test.

    The discipline tests below assert that certain NAMES do not appear in the new
    path. Written against raw source they also matched the prose explaining WHY
    those names must not appear — and `from ...outcomes import refuse` contains
    the substring "import re", which failed the no-regex check on nothing.
    """
    import io
    import tokenize
    kept: list = []
    previous_type = tokenize.INDENT
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT:
                continue
            if token.type == tokenize.STRING and previous_type in (
                    tokenize.INDENT, tokenize.NEWLINE, tokenize.NL):
                continue                                  # a docstring
            kept.append(token.string)
            if token.type not in (tokenize.NL, tokenize.NEWLINE):
                previous_type = token.type
            else:
                previous_type = token.type
    except (tokenize.TokenError, IndentationError):
        return source
    return "\n".join(kept)


def _frozen_payloads():
    body = json.loads(FROZEN.read_text())
    return {r["question"]: r["raw_payload"] for r in body["results"]
            if r.get("raw_payload")}


_PAYLOADS = _frozen_payloads()


def served():
    """A legacy envelope of the shape `_run_analysis` hands over."""
    return {"ok": True, "value": float(truth.row_count(
        _BOOK, [(truth.AGE, "gt", 55), (truth.LTV, "gt", 50)])),
        "route": None, "answer": "the legacy answer", "warnings": []}


class _Shadow:
    """One scenario's configuration, always fully undone afterwards."""

    def __init__(self, *, flag="shadow", clients=CANARY_CLIENT, sink=True,
                 interpreter="replay", max_in_flight=None):
        self.flag, self.clients, self.sink = flag, clients, sink
        self.interpreter, self.max_in_flight = interpreter, max_in_flight

    def __enter__(self):
        self._previous = {k: os.environ.get(k) for k in (
            adapter.SHADOW_ENV_VAR, adapter.LEDGER_ENV_VAR,
            wiring.CANARY_ENV_VAR, wiring.MAX_IN_FLIGHT_ENV_VAR,
            evidence.SINK_ENV_VAR)}
        self.tmp = tempfile.TemporaryDirectory()
        if self.flag is None:
            os.environ.pop(adapter.SHADOW_ENV_VAR, None)
        else:
            os.environ[adapter.SHADOW_ENV_VAR] = self.flag
        if self.clients is None:
            os.environ.pop(wiring.CANARY_ENV_VAR, None)
        else:
            os.environ[wiring.CANARY_ENV_VAR] = self.clients
        if self.max_in_flight is not None:
            os.environ[wiring.MAX_IN_FLIGHT_ENV_VAR] = str(self.max_in_flight)
        self.evidence_path = os.path.join(self.tmp.name, "evidence.jsonl")
        self.ledger_path = os.path.join(self.tmp.name, "ledger.jsonl")
        if self.sink:
            os.environ[evidence.SINK_ENV_VAR] = self.evidence_path
            os.environ[adapter.LEDGER_ENV_VAR] = self.ledger_path
        else:
            os.environ[evidence.SINK_ENV_VAR] = "/proc/nope/evidence.jsonl"

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
        wiring.set_dispatch(wiring._dispatch_inline)
        evidence.reset_counters()
        return self

    def __exit__(self, *_):
        wiring.set_interpreter_factory(None)
        wiring.set_dispatch(None)
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

    def ledger_rows(self):
        path = Path(self.ledger_path)
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines()
                if line.strip()]


def run(cfg, question=ELIGIBLE_SCALAR, client_id=CANARY_CLIENT, frame=None):
    """Call exactly what `mi_service` calls, and prove the answer survives it."""
    envelope = served()
    before = json.dumps(envelope, sort_keys=True)
    record = wiring.observe_request(
        question=question, client_id=client_id, run_id="2026-03",
        result=envelope, frame=_BOOK if frame is None else frame,
        semantics=_SEMANTICS, view="funded",
        portfolio_id=f"{client_id}/2026-03")
    assert json.dumps(envelope, sort_keys=True) == before, \
        "the shadow mutated the served envelope"
    return record


# --------------------------------------------------------------------------- #
# A — flag off
# --------------------------------------------------------------------------- #
class TestAFlagOff(unittest.TestCase):

    def test_unset_flag_does_absolutely_nothing(self):
        with _Shadow(flag=None) as cfg:
            self.assertIsNone(run(cfg))
            self.assertEqual(cfg.interpreters_built, 0, "an interpreter was built")
            self.assertEqual(cfg.rows(), [])
            self.assertEqual(cfg.ledger_rows(), [])

    def test_off_is_off_for_every_near_miss_value(self):
        for value in ("off", "on", "1", "true", "serve", "shadow_mode", ""):
            with _Shadow(flag=value) as cfg:
                if value == "shadow":
                    continue
                self.assertIsNone(run(cfg), f"{value!r} started a shadow")
                self.assertEqual(cfg.interpreters_built, 0)


# --------------------------------------------------------------------------- #
# B — flag on, outside the canary
# --------------------------------------------------------------------------- #
class TestBOutsideCanary(unittest.TestCase):

    def test_another_client_is_not_shadowed(self):
        with _Shadow() as cfg:
            record = run(cfg, client_id=OTHER_CLIENT)
            self.assertEqual(record["disposition"], evidence.OUTSIDE_CANARY)
            self.assertFalse(record["evidence_persisted"])
            self.assertEqual(cfg.interpreters_built, 0, "a model would have been called")
            self.assertEqual(cfg.rows(), [], "an evidence row was written")
            self.assertEqual(cfg.ledger_rows(), [])

    def test_an_unset_canary_list_shadows_nobody(self):
        with _Shadow(clients=None) as cfg:
            record = run(cfg)
            self.assertEqual(record["disposition"], evidence.OUTSIDE_CANARY)
            self.assertEqual(cfg.interpreters_built, 0)

    def test_a_wildcard_cannot_open_the_canary(self):
        for attempt in ("*", "all", "any", "everyone", " * , all "):
            with _Shadow(clients=attempt) as cfg:
                self.assertEqual(run(cfg)["disposition"], evidence.OUTSIDE_CANARY,
                                 f"{attempt!r} widened the canary")
                self.assertEqual(cfg.interpreters_built, 0)

    def test_the_canary_list_is_trimmed_and_case_insensitive(self):
        with _Shadow(clients=" ACME , beta ") as cfg:
            self.assertEqual(wiring.canary_clients(), {"acme", "beta"})
            self.assertTrue(wiring.in_canary("Acme"))
            self.assertFalse(wiring.in_canary("gamma"))
            self.assertFalse(wiring.in_canary(None))

    def test_the_canary_decision_cannot_read_the_question(self):
        """Scoping a rollout by what was asked would tie coverage to semantics."""
        import inspect
        for fn in (wiring.in_canary, wiring.canary_clients):
            code = _code_of(inspect.getsource(fn))
            for forbidden in ("question", "text", "semantic", "answer"):
                self.assertNotIn(forbidden, code.lower(),
                                 f"{fn.__name__} reads {forbidden}")


# --------------------------------------------------------------------------- #
# C — canary, eligible
# --------------------------------------------------------------------------- #
class TestCCanaryEligible(unittest.TestCase):

    def test_a_frozen_plan_is_built_through_production_orchestration(self):
        with _Shadow() as cfg:
            record = run(cfg)
        self.assertEqual(record["disposition"], evidence.EXECUTED)
        self.assertEqual(cfg.interpreters_built, 1)
        self.assertEqual(record["model"]["model_id"], "claude-opus-5")
        self.assertEqual(record["compiler"]["outcome"], "PLAN")
        self.assertTrue(record["eligibility"]["eligible"],
                        record["eligibility"]["reason"])
        self.assertEqual(record["execution"]["requested_semantics"]["statistic"],
                         "count")

    def test_the_figure_matches_the_independent_oracle(self):
        with _Shadow() as cfg:
            record = run(cfg)
        expected = float(truth.row_count(
            _BOOK, [(truth.AGE, "gt", 55), (truth.LTV, "gt", 50)]))
        self.assertAlmostEqual(record["execution"]["value"], expected, places=2)

    def test_the_plan_id_is_the_one_the_signoff_recorded(self):
        frozen = json.loads(FROZEN.read_text())
        recorded = next(r["plan_id"] for r in frozen["results"]
                        if r["question"] == ELIGIBLE_SCALAR)
        with _Shadow() as cfg:
            record = run(cfg)
        self.assertEqual(record["compiler"]["plan_id"], recorded)

    def test_a_grouped_plan_records_every_cell(self):
        with _Shadow() as cfg:
            record = run(cfg, question=ELIGIBLE_GROUPED)
        execution = record["execution"]
        self.assertEqual(record["disposition"], evidence.EXECUTED)
        self.assertTrue(execution["cell_capture_reran_the_adapters_own_spec"])
        cells = execution["grouped_cells"]
        expected = truth.grouped(_BOOK, ["ltv_bucket", "age_bucket"],
                                 column=truth.BALANCE, how="sum")
        self.assertEqual(len(cells), len(expected))
        produced = {(c["ltv_bucket"], c["age_bucket"]): c["value"] for c in cells}
        for key, value in expected.items():
            self.assertAlmostEqual(produced[key], value, places=2,
                                   msg=f"cell {key}")

    def test_the_evidence_row_is_complete_enough_to_adjudicate_offline(self):
        with _Shadow() as cfg:
            run(cfg)
            rows = cfg.rows()
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertTrue(row["model"]["raw_payload"], "no raw model response")
        self.assertTrue(row["interpretation"]["candidate_intent"])
        self.assertTrue(row["compiler"]["plan"])
        plan = row["compiler"]["plan"]
        self.assertTrue(plan["provenance"]["intent_claims"])
        self.assertTrue(plan["provenance"]["compiler_bindings"])
        self.assertIn("normalisation", plan["provenance"]["compiler_bindings"])
        self.assertTrue(row["eligibility"])
        self.assertTrue(row["execution"]["bound_spec"])
        self.assertTrue(row["execution"]["receipt"]["applied_predicates"])
        self.assertIsNotNone(row["legacy_control"]["served_ok"])
        self.assertEqual(row["request"]["question"], ELIGIBLE_SCALAR)

    def test_the_slice1_comparison_ledger_is_still_written(self):
        with _Shadow() as cfg:
            run(cfg)
            ledger = cfg.ledger_rows()
        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger[0]["classification"],
                         adapter.EXACT_SEMANTIC_PARITY)

    def test_the_evidence_carries_no_loan_identifier(self):
        with _Shadow() as cfg:
            run(cfg, question=ELIGIBLE_GROUPED)
            blob = json.dumps(cfg.rows())
        for identifier in list(_BOOK["loan_identifier"])[:30]:
            self.assertNotIn(str(identifier), blob)


# --------------------------------------------------------------------------- #
# D — canary, ineligible
# --------------------------------------------------------------------------- #
class TestDCanaryIneligible(unittest.TestCase):

    def test_the_gate_blocks_the_adapter_and_says_why(self):
        with _Shadow() as cfg:
            record = run(cfg, question=INELIGIBLE_LENS)
        self.assertEqual(record["disposition"], evidence.INELIGIBLE)
        self.assertTrue(record["compiler"]["plan"], "a plan should still exist")
        self.assertFalse(record["eligibility"]["eligible"])
        self.assertEqual(record["eligibility"]["reason"], adapter.EXPLICIT_LENS)
        self.assertFalse(record["execution"]["attempted"])
        self.assertIn(adapter.EXPLICIT_LENS, record["execution"]["why_not"])

    def test_the_recorded_gate_perimeter_is_the_accepted_one(self):
        with _Shadow() as cfg:
            record = run(cfg, question=INELIGIBLE_LENS)
        gate = record["eligibility"]
        self.assertEqual(gate["gate_capability"], "generic_analysis")
        self.assertEqual(gate["gate_operations"], ["breakdown", "point_in_time"])
        self.assertEqual(gate["gate_period_forms"], ["current"])
        self.assertEqual(gate["gate_max_dimensions"], 2)


# --------------------------------------------------------------------------- #
# E / F — clarify and refuse
# --------------------------------------------------------------------------- #
class TestEClarify(unittest.TestCase):

    def test_a_clarify_is_recorded_and_never_executed(self):
        with _Shadow() as cfg:
            record = run(cfg, question=CLARIFY_QUESTION)
        self.assertEqual(record["disposition"], evidence.CLARIFY)
        self.assertIsNone(record["compiler"]["plan"])
        self.assertTrue(record["compiler"]["reason_codes"])
        self.assertIsNone(record["eligibility"])
        self.assertIsNone(record["execution"])
        self.assertEqual(cfg.ledger_rows(), [],
                         "a clarify must not reach the adapter")


class TestFRefuse(unittest.TestCase):

    def test_a_refuse_is_recorded_and_never_executed(self):
        """Forced through the frozen contract: a payload the parser must reject."""
        class Refusing:
            def interpret(self, question):
                from mi_agent.interpretation_v2.opus_interpreter import (
                    InterpretationOutcome)
                from mi_agent.interpretation_v2.outcomes import (
                    CompileReason, MODEL_OUTPUT_MALFORMED)
                return InterpretationOutcome(
                    question=question, intent=None,
                    reason=CompileReason(MODEL_OUTPUT_MALFORMED, "root",
                                         "not an object"),
                    model_id="claude-opus-5")

        with _Shadow(interpreter=lambda: Refusing()) as cfg:
            record = run(cfg)
        self.assertEqual(record["disposition"], evidence.INTERPRETER_FAILURE)
        self.assertIsNone(record["compiler"]["plan"])
        self.assertEqual(record["model"]["failure"]["code"],
                         "MODEL_OUTPUT_MALFORMED")
        self.assertEqual(cfg.ledger_rows(), [])


# --------------------------------------------------------------------------- #
# G / H / I — failures
# --------------------------------------------------------------------------- #
class TestGInterpreterFailure(unittest.TestCase):

    def test_an_unavailable_interpreter_is_recorded_not_raised(self):
        with _Shadow(interpreter="unavailable") as cfg:
            record = run(cfg)
            rows = cfg.rows()          # inside: the sink lives in a temp dir
        self.assertEqual(record["disposition"], evidence.INTERPRETER_FAILURE)
        self.assertTrue(record["model"]["failure"])
        self.assertEqual(len(rows), 1)

    def test_an_interpreter_that_raises_is_contained(self):
        with _Shadow(interpreter="raises") as cfg:
            record = run(cfg)
            rows = cfg.rows()
        self.assertEqual(record["disposition"], evidence.ORCHESTRATION_ERROR)
        self.assertIn("interpreter exploded", record["orchestration_error"])
        self.assertEqual(len(rows), 1,
                         "a contained failure must still leave evidence")


class TestHAdapterFailure(unittest.TestCase):

    def test_a_frame_the_executor_cannot_use_is_recorded(self):
        with _Shadow() as cfg:
            record = run(cfg, frame=object())
        self.assertEqual(record["disposition"], evidence.EXECUTION_ERROR)
        self.assertTrue(record["execution"]["error"])
        self.assertTrue(record["eligibility"]["eligible"])


class TestIRecorderFailure(unittest.TestCase):

    def test_an_unwritable_sink_does_not_break_the_shadow_or_the_answer(self):
        with _Shadow(sink=False) as cfg:
            record = run(cfg)
        self.assertEqual(record["disposition"], evidence.EXECUTED,
                         "the shadow itself should still have run")
        self.assertFalse(record["evidence_persisted"])
        self.assertGreaterEqual(evidence.evidence_failures(), 1,
                                "an evidence failure must be observable")


# --------------------------------------------------------------------------- #
# bounds, dispatch and discipline
# --------------------------------------------------------------------------- #
class TestBounds(unittest.TestCase):

    def test_a_second_concurrent_shadow_is_skipped_and_recorded(self):
        """The in-flight cap is what stops a flag meaning an unbounded bill."""
        with _Shadow(max_in_flight=1) as cfg:
            seen = {}

            def nested():
                # Called from inside the first shadow, while it is in flight.
                seen["record"] = run(cfg)
                return None

            class Reentrant:
                def interpret(self, question):
                    nested()
                    return OpusInterpreter(
                        ReplayClient(_PAYLOADS, model_id="claude-opus-5")
                    ).interpret(question)

            wiring.set_interpreter_factory(lambda: Reentrant())
            outer = run(cfg)
        self.assertEqual(outer["disposition"], evidence.EXECUTED)
        self.assertEqual(seen["record"]["disposition"],
                         evidence.SHADOW_SKIPPED_BUSY)
        self.assertEqual(seen["record"]["in_flight_limit"], 1)

    def test_the_in_flight_count_returns_to_zero(self):
        with _Shadow() as cfg:
            run(cfg)
        self.assertEqual(wiring.in_flight(), 0)

    def test_production_dispatch_is_off_the_request_path(self):
        """The default must be the background dispatcher, not the inline one."""
        wiring.set_dispatch(None)
        self.assertIs(wiring._DISPATCH, wiring._dispatch_in_background)

    def test_the_default_interpreter_is_the_frozen_one(self):
        import inspect
        source = inspect.getsource(wiring._default_interpreter)
        self.assertIn("OpusInterpreter", source)
        self.assertIn("AnthropicInterpreterClient", source)


class TestDiscipline(unittest.TestCase):

    def test_exactly_one_interpretation_attempt_per_request(self):
        calls = []

        class Counting:
            def interpret(self, question):
                calls.append(question)
                return OpusInterpreter(
                    ReplayClient(_PAYLOADS, model_id="claude-opus-5")
                ).interpret(question)

        with _Shadow(interpreter=lambda: Counting()) as cfg:
            run(cfg)
        self.assertEqual(len(calls), 1, "more than one interpretation was bought")

    def test_the_wiring_calls_no_legacy_semantic_owner(self):
        code = _code_of(
            (_REPO_ROOT / "mi_agent" / "plan_shadow_wiring.py").read_text())
        for forbidden in ("ParsedQuestion", "llm_query_parser",
                          "RecogniserRegistry", "chat_routing",
                          "portfolio_lens", "requested_dimension_terms",
                          "semantic_resolver", "mi_agent_workflow"):
            self.assertNotIn(forbidden, code,
                             f"{forbidden} has no business in the new path")

    def test_the_wiring_cannot_pattern_match(self):
        code = _code_of(
            (_REPO_ROOT / "mi_agent" / "plan_shadow_wiring.py").read_text())
        self.assertNotIn("\nre\n.", code.replace(" ", ""))
        for token in code.split("\n"):
            self.assertNotEqual(token.strip(), "re",
                                "the regex module reached the new path")

    def test_the_wiring_never_edits_an_emitted_plan(self):
        """The plan reaches the gate exactly as the compiler emitted it."""
        code = _code_of(
            (_REPO_ROOT / "mi_agent" / "plan_shadow_wiring.py").read_text())
        compact = code.replace("\n", "")
        for mutation in ("plan[", "plan.pop", "plan.update", "delplan"):
            self.assertNotIn(mutation, compact,
                             f"the plan is modified: {mutation}")

    def test_the_accepted_adapter_is_untouched_by_this_slice(self):
        import subprocess
        accepted = subprocess.run(
            ("git", "rev-parse", "6f42df67:mi_agent/plan_runtime_adapter.py"),
            cwd=_REPO_ROOT, text=True, capture_output=True).stdout.strip()
        current = subprocess.run(
            ("git", "hash-object", "mi_agent/plan_runtime_adapter.py"),
            cwd=_REPO_ROOT, text=True, capture_output=True).stdout.strip()
        self.assertEqual(accepted, current,
                         "the accepted slice 1 adapter changed")


if __name__ == "__main__":
    unittest.main()
