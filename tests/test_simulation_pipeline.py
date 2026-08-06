#!/usr/bin/env python3
"""tests/test_simulation_pipeline.py — the framework against the real pipeline.

Integration coverage: a generated source really does travel the production
funded-ingestion path, resolve identically from three materially different
dialects, integrate into the governed snapshot store, and answer through the
deterministic MI capability, the governed limit engine and the governed MI
Agent.

The end-to-end case is deliberately ONE small smoke case (three periods, three
dialects). The full catalogue is run by ``python -m simulation.runner run-all
--profile standard``, which is not an ordinary unit-test workload.

Run: python -m pytest tests/test_simulation_pipeline.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from simulation import manifests as M  # noqa: E402
from simulation import pipeline as P  # noqa: E402
from simulation import runner as R  # noqa: E402
from simulation.assertions import canonical as A_canonical  # noqa: E402
from simulation.dialects import render  # noqa: E402
from simulation.history import canonical_frame_rows, generate_history  # noqa: E402
from simulation.models import STAGES, FailureClass  # noqa: E402

#: Kept small on purpose: three periods, three dialects, ~110 loans.
SMOKE_CASE = "bridge_clean_short_duration_v1"


def _smoke(case_id: str = SMOKE_CASE, **overrides):
    case = M.get_case(case_id)
    return replace(case, months=overrides.pop("months", 3), **overrides)


class TestSourceToCanonical(unittest.TestCase):
    """One rendered source through the REAL Gate 1-3b path."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="sim_gates_"))
        cls.case = _smoke()
        history = generate_history(cls.case)
        cls.period = history.periods[0]
        rows = canonical_frame_rows(cls.case, history, 0)
        cls.source = render("clean_csv", rows, cls.case, period_index=0,
                            reporting_date=cls.period.reporting_date,
                            out_dir=cls.tmp / "src",
                            evolution=cls.case.schema_evolution)
        cls.gate = P.run_mi_gates(
            cls.source, cls.case, reporting_date=cls.period.reporting_date,
            out_dir=cls.tmp / "out", validation_dir=cls.tmp / "val",
            master_config=R.client_config(cls.case))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_the_production_pipeline_accepted_the_source(self):
        self.assertTrue(self.gate.ok, self.gate.stderr[-2000:])
        self.assertIsNotNone(self.gate.canonical_path)

    def test_every_gate_ran(self):
        joined = "\n".join(self.gate.gate_lines)
        for gate in ("[Gate 1]", "[Transform]", "[Gate 2]", "[Gate 2.5]",
                     "[Gate 3]", "[Gate 3b]"):
            self.assertIn(gate, joined)

    def test_canonical_validation_reported_no_errors(self):
        self.assertIn("[Gate 2] Canonical validation............ OK",
                      "\n".join(self.gate.gate_lines))

    def test_business_rules_reported_no_failures(self):
        line = next(ln for ln in self.gate.gate_lines if ln.startswith("[Gate 3]"))
        self.assertIn("0 failures", line, line)

    def test_canonical_matches_the_independent_expected_truth(self):
        from simulation.reference_truth import period_truth
        frame = A_canonical.load_canonical(self.gate.canonical_path)
        truth = period_truth(self.case, self.period)
        failures = [a for a in A_canonical.assert_canonical_truth(
            frame, truth, self.case) if not a.passed]
        self.assertEqual(failures, [], [a.to_dict() for a in failures])


class TestDialectEquivalence(unittest.TestCase):
    """Three materially different sources, one canonical truth."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="sim_dialects_"))
        cls.case = _smoke()
        history = generate_history(cls.case)
        cls.date = history.periods[0].reporting_date
        rows = canonical_frame_rows(cls.case, history, 0)
        cls.canonical = {}
        for dialect in cls.case.dialects:
            source = render(dialect, rows, cls.case, period_index=0,
                            reporting_date=cls.date,
                            out_dir=cls.tmp / "src" / dialect,
                            evolution=cls.case.schema_evolution)
            gate = P.run_mi_gates(
                source, cls.case, reporting_date=cls.date,
                out_dir=cls.tmp / dialect / "out",
                validation_dir=cls.tmp / dialect / "val",
                master_config=R.client_config(cls.case))
            assert gate.ok, f"{dialect}: {gate.stderr[-1500:]}"
            cls.canonical[dialect] = gate.canonical_path

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_the_source_files_are_genuinely_different(self):
        payloads = {p.read_bytes() for p in
                    (self.tmp / "src").rglob("*") if p.is_file()}
        self.assertEqual(len(payloads), 3)

    def test_every_dialect_resolves_to_equivalent_canonical_truth(self):
        reference = A_canonical.load_canonical(self.canonical["clean_csv"])
        for dialect, path in self.canonical.items():
            if dialect == "clean_csv":
                continue
            failures = [a for a in A_canonical.assert_dialect_equivalence(
                reference, "clean_csv", A_canonical.load_canonical(path),
                dialect, date=self.date) if not a.passed]
            self.assertEqual(failures, [], [a.to_dict() for a in failures])

    def test_source_lineage_stays_distinguishable(self):
        lineage = {d: json.loads((p.parent / "field_lineage.json")
                                 .read_text(encoding="utf-8"))
                   for d, p in self.canonical.items()}
        failures = [a for a in A_canonical.assert_source_lineage_distinct(
            lineage, self.date) if not a.passed]
        self.assertEqual(failures, [], [a.to_dict() for a in failures])


class TestEndToEndCase(unittest.TestCase):
    """One case through every stage the framework verifies."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="sim_e2e_"))
        cls.result = R.run_case(_smoke(), run_root=cls.tmp, scale="small", jobs=4)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_the_case_ended_as_its_manifest_declared(self):
        failures = [s.to_dict() for s in self.result.failures]
        self.assertTrue(self.result.ok, json.dumps(failures, indent=1)[:4000])

    def test_every_stage_ran(self):
        ran = {s.stage for s in self.result.stages}
        for stage in STAGES:
            self.assertIn(stage, ran)

    def test_assertions_were_actually_made_at_every_verified_stage(self):
        for stage in self.result.stages:
            if stage.skipped or stage.stage in ("render", "ingest"):
                continue
            self.assertTrue(stage.assertions, f"{stage.stage} asserted nothing")

    def test_the_evidence_directory_has_the_required_shape(self):
        run_dir = Path(self.result.run_dir)
        for name in ("manifest.json", "generated_sources", "canonical_outputs",
                     "integrated_snapshot_evidence", "expected_truth",
                     "mi_results", "risk_results", "agent_results",
                     "regime_outputs", "assertion_results.json",
                     "run_summary.json"):
            self.assertTrue((run_dir / name).exists(), name)

    def test_the_run_summary_records_the_production_modules_used(self):
        summary = json.loads(
            (Path(self.result.run_dir) / "run_summary.json").read_text(encoding="utf-8"))
        for module in ("engine.orchestrator.trakt_run",
                       "engine.gate_1_alignment.semantic_alignment",
                       "engine.gate_2_transform.canonical_transform",
                       "engine.gate_3_validation.validate_canonical",
                       "snapshot.adapters.local_fs.LocalFsSnapshotStore"):
            self.assertIn(module, summary["production_modules"])

    def test_the_run_summary_records_the_seed_and_manifest_hash(self):
        summary = json.loads(
            (Path(self.result.run_dir) / "run_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["seed"], M.get_case(SMOKE_CASE).seed)
        self.assertTrue(summary["manifest_hash"].startswith("sha256:"))

    def test_the_agent_transcript_carries_machine_readable_evidence(self):
        transcript = json.loads(
            (Path(self.result.run_dir) / "agent_results" / "transcript.json")
            .read_text(encoding="utf-8"))
        answered = [q for q in transcript if q["ok"]]
        self.assertTrue(answered)
        self.assertTrue(any(q["evidence"] for q in answered),
                        "no answer carried a numeric figure")
        refusal = [q for q in transcript if q["id"] == "unavailable_field"]
        self.assertTrue(refusal)
        self.assertFalse(refusal[0]["evidence"],
                         "a refusal must not carry fabricated figures")


class TestGuardCasesFailAtTheRightBoundary(unittest.TestCase):
    """An intended failure must be classified, not merely 'a failure'."""

    def _run(self, case_id: str):
        tmp = Path(tempfile.mkdtemp(prefix="sim_guard_"))
        try:
            case = replace(M.get_case(case_id), months=6)
            return R.run_case(case, run_root=tmp, scale="small", jobs=3), case
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_an_unresolvable_schema_is_refused_as_such(self):
        result, case = self._run("bridge_unsupported_schema_guard_v1")
        self.assertEqual(result.observed_failure_class,
                         FailureClass.UNSUPPORTED_SOURCE_SCHEMA)
        self.assertTrue(result.ok, "the guard case did not fail as declared")

    def test_a_withheld_mandatory_column_blocks_ingestion(self):
        result, case = self._run("equity_release_missing_mandatory_guard_v1")
        self.assertEqual(result.observed_failure_class,
                         FailureClass.MISSING_MANDATORY_DATA)
        self.assertTrue(result.ok, "the guard case did not fail as declared")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
