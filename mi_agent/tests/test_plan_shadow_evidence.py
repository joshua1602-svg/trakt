#!/usr/bin/env python3
"""The evidence contract: complete where it must be, silent where it must be.

Two failures are being designed against, and both have already happened once in
this engagement:

    AN INCOMPLETE RECORD. The live sign-off kept only a projection of each plan,
    so one case could not be settled afterwards without buying the interpretation
    again. A record here has to be enough to adjudicate offline.

    A RECORD THAT BREAKS A REQUEST. An unwritable sink, a read-only disk, a frame
    handed in by accident — none of it may raise, and none of it may be silent
    either: a failure has to be observable as a failure.
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

import pandas as pd                                                  # noqa: E402

from mi_agent import plan_shadow_evidence as evidence                # noqa: E402


class _Sink:
    """Point the recorder at a temporary directory or file, and clean up."""

    def __init__(self, name: str = "evidence.jsonl"):
        self.name = name

    def __enter__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = (os.path.join(self.tmp.name, self.name) if self.name
                     else self.tmp.name)
        self.previous = os.environ.get(evidence.SINK_ENV_VAR)
        os.environ[evidence.SINK_ENV_VAR] = self.path
        evidence.reset_counters()
        return self

    def __exit__(self, *_):
        os.environ.pop(evidence.SINK_ENV_VAR, None)
        if self.previous is not None:
            os.environ[evidence.SINK_ENV_VAR] = self.previous
        self.tmp.cleanup()

    def rows(self):
        if self.name.endswith(".jsonl"):
            return [json.loads(line)
                    for line in Path(self.path).read_text().splitlines()
                    if line.strip()]
        return [json.loads(p.read_text()) for p in sorted(Path(self.path).glob("*.json"))]


def record(**extra):
    body = evidence.new_record(correlation_id="shadow_test0001",
                               question="What is the total balance?",
                               client_id="acme", run_id="2026-03",
                               view="funded", portfolio_id="acme/2026-03")
    body.update(extra)
    return body


class TestTheSkeleton(unittest.TestCase):

    def test_every_stage_key_exists_before_any_stage_runs(self):
        """An abandoned record must say which stage it reached, not just be short."""
        body = record()
        for stage in ("model", "interpretation", "compiler", "eligibility",
                      "execution", "legacy_control"):
            self.assertIn(stage, body)
            self.assertIsNone(body[stage])
        self.assertEqual(body["schema_version"], evidence.SCHEMA_VERSION)
        self.assertEqual(body["request"]["question"],
                         "What is the total balance?")

    def test_a_correlation_id_is_opaque_and_unique(self):
        first, second = evidence.correlation_id(), evidence.correlation_id()
        self.assertNotEqual(first, second)
        self.assertTrue(first.startswith("shadow_"))
        for leak in ("acme", "funded", "balance"):
            self.assertNotIn(leak, first)

    def test_every_disposition_is_declared(self):
        for name in (evidence.OUTSIDE_CANARY, evidence.SHADOW_SKIPPED_BUSY,
                     evidence.INTERPRETER_FAILURE, evidence.CLARIFY,
                     evidence.REFUSE, evidence.INELIGIBLE, evidence.EXECUTED,
                     evidence.EXECUTION_ERROR, evidence.ORCHESTRATION_ERROR):
            self.assertIn(name, evidence.DISPOSITIONS)


class TestRedaction(unittest.TestCase):

    def test_a_credential_key_is_dropped_whatever_it_holds(self):
        clean, notes = evidence.redact({
            "request": {"question": "q", "authorization": "Bearer abc",
                        "api_key": "x", "nested": {"refresh_token": "y"}}})
        self.assertNotIn("authorization", clean["request"])
        self.assertNotIn("api_key", clean["request"])
        self.assertNotIn("refresh_token", clean["request"]["nested"])
        self.assertEqual(clean["request"]["question"], "q")
        self.assertEqual(len(notes), 3, notes)

    def test_a_secret_value_is_masked_so_the_field_still_shows(self):
        clean, notes = evidence.redact({"note": "the key is sk-ant-api03-xyz"})
        self.assertEqual(clean["note"], evidence.REDACTED)
        self.assertTrue(notes)

    def test_a_bearer_value_under_an_innocuous_key_is_still_masked(self):
        clean, _ = evidence.redact({"header": "Bearer eyJhbGciOi"})
        self.assertEqual(clean["header"], evidence.REDACTED)

    def test_a_frame_is_dropped_outright(self):
        """A DataFrame is the one shape a borrower row could arrive in."""
        frame = pd.DataFrame({"loan_identifier": ["L00001"], "balance": [1.0]})
        clean, notes = evidence.redact({"execution": {"frame": frame}})
        self.assertIsNone(clean["execution"]["frame"])
        self.assertTrue(any("frame" in n for n in notes), notes)
        self.assertNotIn("L00001", json.dumps(clean))

    def test_redaction_does_not_mutate_the_input(self):
        original = {"authorization": "Bearer abc", "keep": 1}
        evidence.redact(original)
        self.assertIn("authorization", original,
                      "the caller's structure was modified")

    def test_an_unexpected_type_is_described_not_dropped(self):
        class Odd:
            def __str__(self):
                return "odd value"
        clean, _ = evidence.redact({"thing": Odd()})
        self.assertIn("Odd", clean["thing"])


class TestWriting(unittest.TestCase):

    def test_a_jsonl_sink_appends_one_line_per_record(self):
        with _Sink() as sink:
            self.assertTrue(evidence.write(record()))
            self.assertTrue(evidence.write(record(disposition=evidence.EXECUTED)))
            rows = sink.rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["disposition"], evidence.EXECUTED)
        self.assertTrue(rows[0]["evidence_persisted"])
        self.assertEqual(evidence.evidence_written(), 2)
        self.assertEqual(evidence.evidence_failures(), 0)

    def test_a_directory_sink_writes_one_file_per_record(self):
        with _Sink(name="") as sink:
            evidence.write(record())
            rows = sink.rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["correlation_id"], "shadow_test0001")

    def test_no_sink_configured_writes_nothing_and_is_not_a_failure(self):
        previous = os.environ.pop(evidence.SINK_ENV_VAR, None)
        evidence.reset_counters()
        try:
            self.assertFalse(evidence.write(record()))
            self.assertEqual(evidence.evidence_failures(), 0,
                             "an absent sink is a configuration choice, not a fault")
            self.assertEqual(evidence.evidence_written(), 0)
        finally:
            if previous is not None:
                os.environ[evidence.SINK_ENV_VAR] = previous

    def test_an_unwritable_sink_is_counted_not_raised(self):
        previous = os.environ.get(evidence.SINK_ENV_VAR)
        os.environ[evidence.SINK_ENV_VAR] = "/proc/definitely/not/writable.jsonl"
        evidence.reset_counters()
        try:
            body = record()
            self.assertFalse(evidence.write(body))
            self.assertEqual(evidence.evidence_failures(), 1,
                             "a failure has to be observable as a failure")
            self.assertFalse(body["evidence_persisted"])
        finally:
            os.environ.pop(evidence.SINK_ENV_VAR, None)
            if previous is not None:
                os.environ[evidence.SINK_ENV_VAR] = previous

    def test_a_sink_inside_a_git_checkout_is_refused(self):
        """Production evidence in a working tree is how a secret reaches a commit."""
        previous = os.environ.get(evidence.SINK_ENV_VAR)
        os.environ[evidence.SINK_ENV_VAR] = str(_REPO_ROOT / "shadow_evidence.jsonl")
        evidence.reset_counters()
        try:
            self.assertFalse(evidence.write(record()))
            self.assertEqual(evidence.evidence_failures(), 1)
            self.assertFalse((_REPO_ROOT / "shadow_evidence.jsonl").exists())
        finally:
            os.environ.pop(evidence.SINK_ENV_VAR, None)
            if previous is not None:
                os.environ[evidence.SINK_ENV_VAR] = previous

    def test_a_credential_never_reaches_the_disk(self):
        with _Sink() as sink:
            evidence.write(record(model={"authorization": "Bearer secret-xyz",
                                         "note": "sk-ant-api03-leak"}))
            blob = Path(sink.path).read_text()
        self.assertNotIn("secret-xyz", blob)
        self.assertNotIn("sk-ant-api03-leak", blob)
        self.assertIn(evidence.REDACTED, blob)

    def test_the_written_record_says_what_was_redacted(self):
        with _Sink() as sink:
            evidence.write(record(model={"api_key": "x"}))
            rows = sink.rows()
        self.assertTrue(rows[0]["redactions"],
                        "a silent removal is worse than a noted one")


class TestGroupedCells(unittest.TestCase):

    def test_cells_are_group_keys_and_aggregates(self):
        frame = pd.DataFrame({"ltv_bucket": ["0-30%", "30-40%"],
                              "balance_sum": [10.5, 20.25]})
        cells, note = evidence.cells_of(frame, ["ltv_bucket"], "balance_sum")
        self.assertIsNone(note)
        self.assertEqual(cells, [{"ltv_bucket": "0-30%", "value": 10.5},
                                 {"ltv_bucket": "30-40%", "value": 20.25}])

    def test_two_dimensions_keep_both_keys(self):
        frame = pd.DataFrame({"a": ["x"], "b": ["y"], "v_sum": [3.0]})
        cells, _ = evidence.cells_of(frame, ["a", "b"], "v_sum")
        self.assertEqual(cells, [{"a": "x", "b": "y", "value": 3.0}])

    def test_a_missing_value_column_falls_back_and_says_nothing_silently_wrong(self):
        frame = pd.DataFrame({"a": ["x"], "other": [2.0]})
        cells, note = evidence.cells_of(frame, ["a"], "not_there")
        self.assertEqual(cells, [{"a": "x", "value": 2.0}])
        self.assertIsNone(note)

    def test_an_empty_result_is_reported_not_invented(self):
        cells, note = evidence.cells_of(pd.DataFrame(), ["a"], "v")
        self.assertEqual(cells, [])
        self.assertIn("no rows", note)

    def test_a_huge_grid_is_truncated_with_a_note(self):
        wide = pd.DataFrame({"a": [str(i) for i in range(evidence.MAX_CELLS + 10)],
                             "v_sum": [1.0] * (evidence.MAX_CELLS + 10)})
        cells, note = evidence.cells_of(wide, ["a"], "v_sum")
        self.assertEqual(len(cells), evidence.MAX_CELLS)
        self.assertIn("truncated", note)

    def test_an_unreadable_frame_is_described_not_raised(self):
        cells, note = evidence.cells_of(object(), ["a"], "v")
        self.assertEqual(cells, [])
        self.assertTrue(note)


class TestDiscipline(unittest.TestCase):

    def test_the_recorder_decides_no_semantics(self):
        """It records; it must not read a question or bind a field."""
        source = (_REPO_ROOT / "mi_agent" / "plan_shadow_evidence.py").read_text()
        body = source.split('"""', 2)[-1]
        for forbidden in ("import re", "llm_query_parser", "ParsedQuestion",
                          "RecogniserRegistry", "interpretation_v2",
                          "check_eligibility", "execute_mi_query"):
            self.assertNotIn(forbidden, body,
                             f"{forbidden} has no business in the recorder")


if __name__ == "__main__":
    unittest.main()
