#!/usr/bin/env python3
"""Gates 3 and 4: off by default, and a shadow failure cannot reach a user.

The whole safety claim of slice 1 is that the old path still owns the answer. Two
things have to be true for that, and neither is self-evident from reading the
call site:

    OFF     nothing happens at all — no interpreter imported, no model called,
            no execution, no ledger row
    SHADOW  the old answer is byte-identical to what it would have been, for
            every way the new path can fail

So the failures are FORCED here rather than hoped for: an interpretation that
returns nothing, a compiler refusal, an adapter that raises, and a deterministic
executor that raises. Each must end in silence or a ledger row, never in an
exception and never in a changed answer.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent import plan_runtime_adapter as adapter              # noqa: E402
from mi_agent.mi_query_validator import load_mi_semantics         # noqa: E402
from mi_agent.tests import portfolio_truth_oracle as truth        # noqa: E402
from mi_agent.tests.test_plan_runtime_adapter import plan         # noqa: E402

_SEMANTICS = load_mi_semantics(
    str(_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"))
_BOOK = truth.canonical_book()

#: A legacy envelope of the shape `_run_analysis` hands to `observe`.
CONTROL = {"ok": True, "value": float(truth.total(_BOOK, truth.BALANCE)),
           "route": None, "warnings": []}


class _Flag:
    """Set the shadow flag for one test and always put it back."""

    def __init__(self, value):
        self.value = value

    def __enter__(self):
        self.previous = os.environ.get(adapter.SHADOW_ENV_VAR)
        if self.value is None:
            os.environ.pop(adapter.SHADOW_ENV_VAR, None)
        else:
            os.environ[adapter.SHADOW_ENV_VAR] = self.value
        return self

    def __exit__(self, *_):
        os.environ.pop(adapter.SHADOW_ENV_VAR, None)
        if self.previous is not None:
            os.environ[adapter.SHADOW_ENV_VAR] = self.previous
        adapter.set_plan_provider(None)


class TestGate3OffByDefault(unittest.TestCase):

    def test_unset_flag_does_nothing(self):
        with _Flag(None):
            called = []
            adapter.set_plan_provider(lambda: called.append(1) or plan())
            out = adapter.observe(result=CONTROL, frame=_BOOK,
                                  semantics=_SEMANTICS)
            self.assertIsNone(out)
            self.assertEqual(called, [],
                             "a plan was built with the flag off")

    def test_off_writes_no_ledger_row(self):
        import tempfile
        with _Flag("off"), tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.jsonl"
            os.environ[adapter.LEDGER_ENV_VAR] = str(ledger)
            try:
                adapter.set_plan_provider(plan)
                adapter.observe(result=CONTROL, frame=_BOOK, semantics=_SEMANTICS)
                self.assertFalse(ledger.exists(), "the ledger was written")
            finally:
                os.environ.pop(adapter.LEDGER_ENV_VAR, None)

    def test_only_the_exact_word_shadow_enables_anything(self):
        for value in ("serve", "primary", "on", "1", "true", "shadow_mode", ""):
            with _Flag(value):
                self.assertEqual(adapter.shadow_mode(), adapter.SHADOW_OFF,
                                 f"{value!r} must not enable anything")
        for value in ("shadow", "SHADOW", " Shadow "):
            with _Flag(value):
                self.assertEqual(adapter.shadow_mode(), adapter.SHADOW_ON,
                                 f"{value!r} should enable shadow")

    def test_the_call_site_imports_no_interpreter_at_module_scope(self):
        """`mi_service` must not pull interpretation_v2 in just by being imported.

        The import is inside the function on purpose: a module-level import would
        load the interpreter for every request whether the flag was on or not.
        """
        source = (_REPO_ROOT / "mi_agent_api" / "mi_service.py").read_text()
        head = source.split("def _run_analysis")[0]
        # `plan_shadow_wiring` joined this list when slice 1A made the call site
        # build plans: it is the module that can reach an interpreter now, so it
        # is the one that must not be loaded for every request regardless of flag.
        for forbidden in ("interpretation_v2", "plan_runtime_adapter",
                          "plan_shadow_wiring", "plan_shadow_evidence"):
            self.assertNotIn(forbidden, head,
                             f"{forbidden} is imported at module scope")


class TestGate4ShadowCannotServe(unittest.TestCase):
    """Every failure mode, forced, with the control answer checked afterwards."""

    def _observe_unharmed(self, provider):
        control = dict(CONTROL)
        before = dict(control)
        with _Flag("shadow"):
            adapter.set_plan_provider(provider)
            record = adapter.observe(result=control, frame=_BOOK,
                                     semantics=_SEMANTICS)
        self.assertEqual(control, before,
                         "the shadow mutated the control envelope")
        return record

    def test_interpretation_returns_nothing(self):
        self.assertIsNone(self._observe_unharmed(lambda: None))

    def test_compiler_refusal_has_no_plan(self):
        record = self._observe_unharmed(lambda: {})
        self.assertIsNotNone(record)
        self.assertEqual(record["classification"], adapter.NOT_ELIGIBLE)
        self.assertEqual(record["ineligible_reason"], adapter.NOT_A_PLAN)

    def test_the_plan_provider_itself_raises(self):
        def boom():
            raise RuntimeError("interpretation exploded")
        self.assertIsNone(self._observe_unharmed(boom))

    def test_the_deterministic_executor_raises(self):
        """A frame the executor cannot use. The error is captured, not raised."""
        control = dict(CONTROL)
        with _Flag("shadow"):
            adapter.set_plan_provider(plan)
            record = adapter.observe(result=control, frame=object(),
                                     semantics=_SEMANTICS)
        self.assertIsNotNone(record)
        self.assertEqual(record["classification"], adapter.SHADOW_EXECUTION_ERROR)
        self.assertTrue(record["new_error"])
        self.assertEqual(control, CONTROL)

    def test_a_ledger_that_cannot_be_written_is_not_an_error(self):
        with _Flag("shadow"):
            os.environ[adapter.LEDGER_ENV_VAR] = "/nonexistent-dir/ledger.jsonl"
            try:
                adapter.set_plan_provider(plan)
                record = adapter.observe(result=dict(CONTROL), frame=_BOOK,
                                         semantics=_SEMANTICS)
                self.assertIsNone(record, "an unwritable ledger must stay silent")
            finally:
                os.environ.pop(adapter.LEDGER_ENV_VAR, None)

    def test_an_eligible_shadow_reaches_parity_and_still_serves_the_control(self):
        record = self._observe_unharmed(plan)
        self.assertIsNotNone(record)
        self.assertEqual(record["classification"], adapter.EXACT_SEMANTIC_PARITY)
        self.assertAlmostEqual(record["new_value"], CONTROL["value"], places=2)

    def test_a_numerical_difference_is_reported_not_resolved(self):
        """The control is not the truth oracle, so a difference is adjudicated."""
        control = {"ok": True, "value": 1.0, "route": None}
        with _Flag("shadow"):
            adapter.set_plan_provider(plan)
            record = adapter.observe(result=control, frame=_BOOK,
                                     semantics=_SEMANTICS)
        self.assertEqual(record["classification"], adapter.NUMERICAL_DIFFERENCE)
        self.assertIn("not the truth oracle", record["note"])

    def test_the_ledger_carries_no_loan_level_data(self):
        import json
        import tempfile
        with _Flag("shadow"), tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.jsonl"
            os.environ[adapter.LEDGER_ENV_VAR] = str(ledger)
            try:
                adapter.set_plan_provider(plan)
                adapter.observe(result=dict(CONTROL), frame=_BOOK,
                                semantics=_SEMANTICS)
                rows = [json.loads(line)
                        for line in ledger.read_text().splitlines() if line.strip()]
            finally:
                os.environ.pop(adapter.LEDGER_ENV_VAR, None)
        self.assertEqual(len(rows), 1)
        blob = json.dumps(rows[0])
        # No loan identifier from the book may appear anywhere in the row.
        for identifier in list(_BOOK["loan_identifier"])[:25]:
            self.assertNotIn(str(identifier), blob)
        self.assertNotIn("borrower_name", blob)


if __name__ == "__main__":
    unittest.main()
