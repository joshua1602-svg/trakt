#!/usr/bin/env python3
"""tests/test_blob_trigger_app.py

Blob-trigger routing/inference: path parsing, schema fingerprint, source-registry
inference (new vs known vs schema-drift), dataset/frequency target selection,
Orchestrator invocation (mocked) and event-manifest writing. No Azure account.

Run: python -m unittest tests.test_blob_trigger_app
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from apps.blob_trigger_app.path_parser import parse_blob_path, PathParseError
from apps.blob_trigger_app.schema_fingerprint import (
    fingerprint_from_schema, compute_schema_fingerprint)
from apps.blob_trigger_app.target_selection import select_target
from apps.blob_trigger_app.source_registry import (
    SourceRegistry, SourceRecord, STATUS_ACTIVE)
from apps.blob_trigger_app import router as R

_NOW = "2026-10-01T00:00:00+00:00"
_COLUMNS = ["loan_id", "balance", "rate", "origination_date"]


def _fp(columns=_COLUMNS, file_type="xlsx"):
    return fingerprint_from_schema(file_type=file_type, columns=columns)


class RecordingInvoker:
    """Mock Orchestrator invoker: records calls, returns a configurable result."""

    def __init__(self, status="done"):
        self.calls = []
        self.status = status

    def __call__(self, **kw):
        self.calls.append(kw)
        return {"run_id": "orun_mock", "status": self.status,
                "central_canonical_path": "/tmp/central.csv", "blockers": []}


# --------------------------------------------------------------------------- #
# 1. Path parsing
# --------------------------------------------------------------------------- #

class TestPathParsing(unittest.TestCase):

    def test_valid(self):
        p = parse_blob_path("raw/ERE/funded/monthly/direct_001/2026-09-30/loan_tape.xlsx")
        self.assertEqual(p.client_id, "ERE")
        self.assertEqual(p.dataset, "funded")
        self.assertEqual(p.frequency, "monthly")
        self.assertEqual(p.source_portfolio_id, "direct_001")
        self.assertEqual(p.reporting_period, "2026-09-30")
        self.assertEqual(p.filename, "loan_tape.xlsx")

    def test_weekly_iso_week(self):
        p = parse_blob_path("raw/ERE/pipeline/weekly/direct_001/2026-W39/pipeline_extract.xlsx")
        self.assertEqual(p.reporting_period, "2026-W39")
        self.assertEqual(p.dataset, "pipeline")

    def test_leading_container_tolerated(self):
        p = parse_blob_path("/somecontainer/raw/ERE/funded/monthly/direct_001/2026-09-30/x.csv")
        self.assertEqual(p.client_id, "ERE")

    def test_fail_closed(self):
        for bad in ("raw/ERE/funded/monthly/direct_001/loan.xlsx",            # too few
                    "raw/ERE/BADSET/monthly/direct_001/2026-09-30/x.xlsx",     # bad dataset
                    "raw/ERE/funded/yearly/direct_001/2026-09-30/x.xlsx",      # bad frequency
                    "raw/ERE/funded/monthly/direct_001/not-a-period/x.xlsx",   # bad period
                    "raw/ERE/funded/monthly/direct_001/2026-09-30/noext",      # no extension
                    "ERE/funded/monthly/direct_001/2026-09-30/x.xlsx"):        # no raw root
            with self.assertRaises(PathParseError, msg=bad):
                parse_blob_path(bad)


# --------------------------------------------------------------------------- #
# 2. Target selection (Regime never for pipeline/forecast)
# --------------------------------------------------------------------------- #

class TestTargetSelection(unittest.TestCase):

    def test_funded_monthly_regime(self):
        s = select_target("funded", "monthly", regime_required=True)
        self.assertEqual((s.target, s.run_regime), ("all", True))

    def test_funded_monthly_no_regime(self):
        s = select_target("funded", "monthly", regime_required=False)
        self.assertEqual((s.target, s.run_regime), ("mi", False))

    def test_pipeline_is_mi_only(self):
        s = select_target("pipeline", "weekly", regime_required=True)  # ignored
        self.assertEqual((s.target, s.run_regime), ("mi", False))

    def test_forecast_is_mi_only(self):
        s = select_target("forecast", "monthly", regime_required=True)
        self.assertEqual((s.target, s.run_regime), ("mi", False))


# --------------------------------------------------------------------------- #
# 3. Schema fingerprint
# --------------------------------------------------------------------------- #

class TestSchemaFingerprint(unittest.TestCase):

    def test_stable_and_sensitive(self):
        a = _fp()
        b = _fp()
        self.assertEqual(a.fingerprint, b.fingerprint)            # stable
        self.assertNotEqual(a.fingerprint, _fp(_COLUMNS + ["extra"]).fingerprint)  # added col
        self.assertNotEqual(a.fingerprint, _fp(list(reversed(_COLUMNS))).fingerprint)  # order
        self.assertNotEqual(a.fingerprint, _fp(file_type="csv").fingerprint)      # file type

    def test_from_file(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "t.csv"
            p.write_text("loan_id,balance,rate,origination_date\n1,2,3,2020-01-01\n")
            info = compute_schema_fingerprint(p)
            self.assertTrue(info.fingerprint.startswith("sha256:"))
            self.assertEqual(info.columns, _COLUMNS)
            # values don't change the key
            p.write_text("loan_id,balance,rate,origination_date\n9,9,9,2021-09-09\n")
            self.assertEqual(compute_schema_fingerprint(p).fingerprint, info.fingerprint)


# --------------------------------------------------------------------------- #
# 4-12. Routing
# --------------------------------------------------------------------------- #

class TestRouting(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.out = self._tmp.name
        self.reg_path = Path(self.out) / "source_registry.json"
        self.registry = SourceRegistry(self.reg_path)
        self.fp = _fp()

    def tearDown(self):
        self._tmp.cleanup()

    def _seed_active(self, *, pid="direct_001", dataset="funded", frequency="monthly",
                     regime_required=False, fingerprint=None):
        self.registry.upsert(SourceRecord(
            client_id="ERE", source_portfolio_id=pid, dataset=dataset, frequency=frequency,
            source_portfolio_type="direct", approved_mapping_id="m1",
            mapping_config_path="config/m1.yaml",
            expected_schema_fingerprint=(fingerprint or self.fp.fingerprint),
            regime_required=regime_required, status=STATUS_ACTIVE))

    def _run(self, blob, *, invoker, schema=None):
        return R.handle_blob_event(
            blob, registry=self.registry, out_dir=self.out,
            schema_info=schema or self.fp, orchestrator_invoker=invoker, now=_NOW)

    # ---- known source: deterministic ---------------------------------------
    def test_monthly_funded_known_source_deterministic(self):
        self._seed_active(regime_required=True)  # funded book needing ESMA
        inv = RecordingInvoker()
        m = self._run("raw/ERE/funded/monthly/direct_001/2026-09-30/loan_tape.xlsx", invoker=inv)
        self.assertTrue(m["registry_match"])
        self.assertFalse(m["requires_source_onboarding"])
        self.assertEqual(m["decision"], "deterministic")
        self.assertEqual(m["status"], "processed")
        self.assertEqual(m["selected_target"]["target"], "all")        # regime required
        self.assertTrue(m["selected_target"]["run_regime"])
        self.assertEqual(inv.calls[0]["processing_mode"], "deterministic")
        self.assertEqual(inv.calls[0]["target"], "all")
        self.assertEqual(inv.calls[0]["mapping_config_path"], "config/m1.yaml")
        # registry updated with last successful period
        rec = self.registry.lookup("ERE", "direct_001", "funded", "monthly")
        self.assertEqual(rec.last_successful_reporting_period, "2026-09-30")

    def test_weekly_pipeline_known_source_mi_only_no_regime(self):
        self._seed_active(dataset="pipeline", frequency="weekly", regime_required=True)
        inv = RecordingInvoker()
        m = self._run("raw/ERE/pipeline/weekly/direct_001/2026-W39/pipeline_extract.xlsx", invoker=inv)
        self.assertEqual(m["decision"], "deterministic")
        self.assertEqual(m["selected_target"]["target"], "mi")
        self.assertFalse(m["selected_target"]["run_regime"])           # pipeline → never regime
        self.assertEqual(inv.calls[0]["target"], "mi")
        self.assertFalse(inv.calls[0]["run_regime"])

    # ---- new acquired portfolio: source onboarding -------------------------
    def test_new_acquired_portfolio_source_onboarding(self):
        inv = RecordingInvoker(status="halted")  # onboarding halts at mapping gate
        m = self._run("raw/ERE/funded/monthly/acquired_001/2026-09-30/loan_tape.xlsx", invoker=inv)
        self.assertFalse(m["registry_match"])
        self.assertTrue(m["requires_source_onboarding"])
        self.assertEqual(m["decision"], "source_onboarding")
        self.assertEqual(m["status"], "pending_review")
        self.assertEqual(inv.calls[0]["processing_mode"], "source_onboarding")
        # a pending_review record is now tracked
        rec = self.registry.lookup("ERE", "acquired_001", "funded", "monthly")
        self.assertIsNotNone(rec)
        self.assertEqual(rec.status, "pending_review")
        self.assertEqual(rec.source_portfolio_type, "acquired")

    # ---- known source, changed schema: fail closed -------------------------
    def test_schema_drift_fails_closed(self):
        self._seed_active()
        inv = RecordingInvoker()
        drifted = _fp(_COLUMNS + ["new_mandatory_col"])
        m = self._run("raw/ERE/funded/monthly/direct_001/2026-09-30/loan_tape.xlsx",
                      invoker=inv, schema=drifted)
        self.assertTrue(m["registry_match"])
        self.assertEqual(m["decision"], "schema_drift")
        self.assertEqual(m["status"], "pending_review")
        self.assertFalse(m["orchestrator_invocation"]["invoked"])      # never processed
        self.assertEqual(len(inv.calls), 0)
        self.assertIn("schema_drift", m["error"])

    # ---- manifest written --------------------------------------------------
    def test_event_manifest_written(self):
        self._seed_active()
        m = self._run("raw/ERE/funded/monthly/direct_001/2026-09-30/loan_tape.xlsx",
                      invoker=RecordingInvoker())
        path = Path(self.out) / f"{m['event_id']}.json"
        self.assertTrue(path.exists())
        on_disk = json.loads(path.read_text())
        self.assertEqual(on_disk["status"], "processed")
        self.assertEqual(on_disk["schema_fingerprint"], self.fp.fingerprint)

    # ---- unparseable path fails closed -------------------------------------
    def test_bad_path_failed_manifest(self):
        m = self._run("raw/ERE/funded/direct_001/loan.xlsx", invoker=RecordingInvoker())
        self.assertEqual(m["status"], "failed")
        self.assertIn("path_parse_error", m["error"])

    # ---- forecast never routes to regime -----------------------------------
    def test_forecast_no_regime(self):
        self._seed_active(dataset="forecast", frequency="monthly", regime_required=True)
        inv = RecordingInvoker()
        m = self._run("raw/ERE/forecast/monthly/direct_001/2026-09-30/forecast.xlsx", invoker=inv)
        self.assertFalse(m["selected_target"]["run_regime"])
        self.assertFalse(inv.calls[0]["run_regime"])


if __name__ == "__main__":
    unittest.main()
