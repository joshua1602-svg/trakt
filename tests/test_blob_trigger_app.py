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

    def test_configurable_container(self):
        # A non-default container (e.g. raw-v2) parses when configured.
        p = parse_blob_path(
            "raw-v2/ERE/funded/monthly/direct_001/2026-09-30/loan_tape.xlsx",
            container="raw-v2")
        self.assertEqual(p.client_id, "ERE")
        self.assertEqual(p.source_portfolio_id, "direct_001")
        # Wrong container fails closed (raw-v2 path under default 'raw').
        with self.assertRaises(PathParseError):
            parse_blob_path(
                "raw-v2/ERE/funded/monthly/direct_001/2026-09-30/loan_tape.xlsx")
        # Inner-only path (no container prefix) also parses.
        p2 = parse_blob_path(
            "ERE/funded/monthly/direct_001/2026-09-30/loan_tape.xlsx", container="raw-v2")
        self.assertEqual(p2.client_id, "ERE")

    def test_router_honours_container(self):
        import tempfile
        from apps.blob_trigger_app.source_registry import SourceRegistry
        with tempfile.TemporaryDirectory() as td:
            reg = SourceRegistry(Path(td) / "r.json")
            m = R.handle_blob_event(
                "raw-v2/ERE/funded/monthly/acquired_001/2026-09-30/_READY",
                registry=reg, out_dir=td, container="raw-v2", pack_marker="_READY",
                schema_info=_fp(), orchestrator_invoker=RecordingInvoker(status="halted"),
                now=_NOW)
            self.assertEqual(m["client_id"], "ERE")
            self.assertEqual(m["decision"], "source_onboarding")  # parsed + routed

    def test_fail_closed(self):
        for bad in ("raw/ERE/funded/monthly/direct_001/loan.xlsx",            # too few
                    "raw/ERE/BADSET/monthly/direct_001/2026-09-30/x.xlsx",     # bad dataset
                    "raw/ERE/funded/yearly/direct_001/2026-09-30/x.xlsx",      # bad frequency
                    "raw/ERE/funded/monthly/direct_001/not-a-period/x.xlsx",   # bad period
                    "raw/ERE/funded/monthly/direct_001/2026-09-30/",           # empty filename
                    "raw/ERE/extra/funded/monthly/direct_001/2026-09-30/x.xlsx"):  # too many
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
        # Processing fires on the completion marker (Option A); rewrite the
        # uploaded data file to the folder's _READY marker so these decision
        # tests exercise the real routing rather than the pack-member gate.
        if "/" in blob and not blob.endswith("/_READY"):
            blob = blob.rsplit("/", 1)[0] + "/_READY"
        return R.handle_blob_event(
            blob, registry=self.registry, out_dir=self.out, pack_marker="_READY",
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


class TestPackCompletion(unittest.TestCase):
    """Option A: only the READY marker starts processing; idempotent re-fires."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.out = self._tmp.name
        self.registry = SourceRegistry(Path(self.out) / "r.json")
        self.fp = _fp()
        self.folder = "raw/ERE/funded/monthly/direct_001/2026-09-30"
        # Known source so the marker routes deterministically.
        self.registry.upsert(SourceRecord(
            client_id="ERE", source_portfolio_id="direct_001", dataset="funded",
            frequency="monthly", source_portfolio_type="direct", approved_mapping_id="m1",
            expected_schema_fingerprint=self.fp.fingerprint, status=STATUS_ACTIVE))

    def tearDown(self):
        self._tmp.cleanup()

    def _ev(self, filename, inv, schema=None):
        return R.handle_blob_event(
            f"{self.folder}/{filename}", registry=self.registry, out_dir=self.out,
            pack_marker="_READY", schema_info=schema, orchestrator_invoker=inv, now=_NOW)

    def test_data_files_do_not_start_orchestrator(self):
        inv = RecordingInvoker()
        for f in ("loan_tape.xlsx", "cashflow.xlsx", "collateral.xlsx"):
            m = self._ev(f, inv)
            self.assertEqual(m["status"], "awaiting_pack")
            self.assertFalse(m["orchestrator_invocation"]["invoked"])
        self.assertEqual(len(inv.calls), 0)          # three files, zero runs

    def test_marker_starts_once_after_pack(self):
        inv = RecordingInvoker()
        for f in ("loan_tape.xlsx", "cashflow.xlsx", "collateral.xlsx"):
            self._ev(f, inv)
        m = self._ev("_READY", inv, schema=self.fp)   # uploader writes marker LAST
        self.assertEqual(m["decision"], "deterministic")
        self.assertEqual(m["status"], "processed")
        self.assertEqual(len(inv.calls), 1)           # exactly one orchestrator run
        self.assertEqual(inv.calls[0]["processing_mode"], "deterministic")

    def test_idempotent_marker_refire_skipped(self):
        inv = RecordingInvoker()
        self._ev("_READY", inv, schema=self.fp)        # first marker → runs
        m2 = self._ev("_READY", inv, schema=self.fp)   # duplicate marker → skipped
        self.assertEqual(m2["status"], "already_processed")
        self.assertFalse(m2["orchestrator_invocation"]["invoked"])
        self.assertEqual(len(inv.calls), 1)           # still only one run

    def test_changed_pack_reruns(self):
        inv = RecordingInvoker()
        self._ev("_READY", inv, schema=self.fp)        # runs
        # new data (different fingerprint) → not idempotent-skipped. With a known
        # source this is schema_drift (fail closed), but it is NOT skipped.
        m = self._ev("_READY", inv, schema=_fp(_COLUMNS + ["extra"]))
        self.assertNotEqual(m["status"], "already_processed")
        self.assertEqual(m["decision"], "schema_drift")


# --------------------------------------------------------------------------- #
# 13. Event Grid entrypoint (subject parsing + configurable container)
# --------------------------------------------------------------------------- #

from apps.blob_trigger_app.eventgrid import classify_blob_event, parse_blob_subject


class TestEventGridEntrypoint(unittest.TestCase):

    def _subject(self, container, blob):
        return f"/blobServices/default/containers/{container}/blobs/{blob}"

    def test_raw_v2_accepted_when_configured(self):
        ref = classify_blob_event(
            self._subject("raw-v2", "ERE/funded/monthly/direct_001/2026-01-31/_READY.json"),
            "raw-v2")
        self.assertTrue(ref.accepted)
        self.assertEqual(ref.container, "raw-v2")
        self.assertEqual(ref.blob_path, "ERE/funded/monthly/direct_001/2026-01-31/_READY.json")

    def test_other_container_skipped(self):
        ref = classify_blob_event(self._subject("inbound", "x.csv"), "raw-v2")
        self.assertFalse(ref.accepted)
        ref2 = classify_blob_event(self._subject("outbound", "y.csv"), "raw-v2")
        self.assertFalse(ref2.accepted)

    def test_subject_parse(self):
        c, b = parse_blob_subject(self._subject("raw-v2", "a/b/c.csv"))
        self.assertEqual((c, b), ("raw-v2", "a/b/c.csv"))

    def test_legacy_inbound_hardcoding_removed(self):
        # The deployed root entrypoint must be an Event Grid handler with NO
        # hardcoded 'inbound' container check.
        root = (_REPO / "function_app.py").read_text()
        self.assertIn("event_grid_trigger", root)
        self.assertNotIn('container != "inbound"', root)
        self.assertIn("TRAKT_BLOB_CONTAINER", root)


# --------------------------------------------------------------------------- #
# 14. _READY.json pack triggering, completeness, force_reprocess, metadata
# --------------------------------------------------------------------------- #

class RecordingRefresher:
    def __init__(self):
        self.calls = []

    def __call__(self, **kw):
        self.calls.append(kw)
        return {"central_canonical_path": "/platform/central_canonical.csv",
                "portfolios": ["direct_001", "acquired_001"], "assembler_run_id": "asm_x"}


class TestReadyJsonAndAcquired(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.out = self._tmp.name
        self.registry = SourceRegistry(Path(self.out) / "r.json")
        self.fp = _fp()

    def tearDown(self):
        self._tmp.cleanup()

    def _seed_active(self, *, pid, dataset="funded", frequency="monthly",
                     ptype="direct", regime_required=False, fingerprint=None):
        self.registry.upsert(SourceRecord(
            client_id="ERE", source_portfolio_id=pid, dataset=dataset, frequency=frequency,
            source_portfolio_type=ptype, approved_mapping_id="m1",
            mapping_config_path="config/m1.yaml",
            expected_schema_fingerprint=(fingerprint or self.fp.fingerprint),
            regime_required=regime_required, status=STATUS_ACTIVE))

    def _ev(self, blob, inv, *, schema=None, meta=None, pack_files=None, refresher=None):
        return R.handle_blob_event(
            blob, registry=self.registry, out_dir=self.out, pack_marker="_READY.json",
            schema_info=schema, marker_metadata=meta, pack_files=pack_files,
            orchestrator_invoker=inv,
            assembler_refresher=(refresher or RecordingRefresher()), now=_NOW)

    # ---- 3-file monthly funded pack routes as ONE pack ---------------------
    def test_three_file_pack_one_run(self):
        self._seed_active(pid="direct_001")
        inv = RecordingInvoker()
        folder = "raw/ERE/funded/monthly/direct_001/2026-01-31"
        for f in ("LoanExtract.csv", "PropertyExtract.csv", "Funder.csv"):
            m = self._ev(f"{folder}/{f}", inv)
            self.assertEqual(m["status"], "awaiting_pack")
            self.assertEqual(m["event_decision"], "ignored_data_file_waiting_for_ready")
        m = self._ev(f"{folder}/_READY.json", inv, schema=self.fp)
        self.assertEqual(m["decision"], "deterministic")
        self.assertEqual(m["status"], "processed")
        self.assertEqual(m["event_decision"], "known_source_processed")
        self.assertEqual(len(inv.calls), 1)            # exactly one pack run

    # ---- missing expected files halts (pending review) ---------------------
    def test_missing_expected_files_pending_review(self):
        self._seed_active(pid="direct_001")
        inv = RecordingInvoker()
        meta = {"expected_files": ["LoanExtract.csv", "PropertyExtract.csv", "Funder.csv"]}
        m = self._ev("raw/ERE/funded/monthly/direct_001/2026-01-31/_READY.json", inv,
                     schema=self.fp, meta=meta, pack_files=["LoanExtract.csv"])  # 2 missing
        self.assertEqual(m["status"], "pending_review")
        self.assertEqual(m["event_decision"], "incomplete_pack_pending_review")
        self.assertEqual(len(inv.calls), 0)            # never started
        self.assertIn("PropertyExtract.csv", m["error"])

    def test_complete_pack_passes_completeness(self):
        self._seed_active(pid="direct_001")
        inv = RecordingInvoker()
        meta = {"expected_files": ["LoanExtract.csv", "Funder.csv"]}
        m = self._ev("raw/ERE/funded/monthly/direct_001/2026-01-31/_READY.json", inv,
                     schema=self.fp, meta=meta, pack_files=["LoanExtract.csv", "Funder.csv"])
        self.assertEqual(m["status"], "processed")

    # ---- duplicate _READY.json ignored unless force_reprocess --------------
    def test_duplicate_ready_ignored_then_force(self):
        self._seed_active(pid="direct_001")
        inv = RecordingInvoker()
        blob = "raw/ERE/funded/monthly/direct_001/2026-01-31/_READY.json"
        self._ev(blob, inv, schema=self.fp)            # first → runs
        m2 = self._ev(blob, inv, schema=self.fp)       # duplicate → ignored
        self.assertEqual(m2["status"], "already_processed")
        self.assertEqual(m2["event_decision"], "duplicate_ready_ignored")
        self.assertEqual(len(inv.calls), 1)
        # force_reprocess overrides idempotency
        m3 = self._ev(blob, inv, schema=self.fp, meta={"force_reprocess": True})
        self.assertNotEqual(m3["status"], "already_processed")
        self.assertEqual(len(inv.calls), 2)

    # ---- regime_required from marker → routes to all -----------------------
    def test_regime_required_marker_routes_all(self):
        self._seed_active(pid="direct_001", regime_required=False)  # registry says no
        inv = RecordingInvoker()
        m = self._ev("raw/ERE/funded/monthly/direct_001/2026-01-31/_READY.json", inv,
                     schema=self.fp, meta={"regime_required": True})  # marker overrides
        self.assertEqual(m["selected_target"]["target"], "all")
        self.assertTrue(m["selected_target"]["run_regime"])
        self.assertEqual(inv.calls[0]["target"], "all")

    # ---- weekly pipeline never routes to regime even if marker asks --------
    def test_pipeline_never_regime(self):
        self._seed_active(pid="direct_001", dataset="pipeline", frequency="weekly")
        inv = RecordingInvoker()
        m = self._ev("raw/ERE/pipeline/weekly/direct_001/2026-W05/_READY.json", inv,
                     schema=self.fp, meta={"regime_required": True, "target": "all"})
        self.assertEqual(m["selected_target"]["target"], "mi")
        self.assertFalse(m["selected_target"]["run_regime"])
        self.assertFalse(inv.calls[0]["run_regime"])

    # ---- acquired new source → onboarding / pending review + metadata ------
    def test_acquired_new_source_onboarding_with_metadata(self):
        inv = RecordingInvoker(status="halted")
        meta = {"acquisition_date": "2025-11-30", "seller_name": "BigBank plc"}
        m = self._ev("raw/ERE/funded/ad_hoc/acquired_001/2026-01-31/_READY.json", inv,
                     schema=self.fp, meta=meta)
        self.assertEqual(m["decision"], "source_onboarding")
        self.assertEqual(m["status"], "pending_review")
        self.assertEqual(m["event_decision"], "new_source_pending_review")
        # acquisition metadata + acquired type passed through to the orchestrator
        self.assertEqual(inv.calls[0]["acquisition_date"], "2025-11-30")
        self.assertEqual(inv.calls[0]["seller_name"], "BigBank plc")
        self.assertEqual(inv.calls[0]["source_portfolio_type"], "acquired")
        rec = self.registry.lookup("ERE", "acquired_001", "funded", "ad_hoc")
        self.assertEqual(rec.status, "pending_review")

    # ---- acquired approved known source → deterministic + assembler refresh -
    def test_acquired_known_source_deterministic_triggers_refresh(self):
        self._seed_active(pid="acquired_001", frequency="ad_hoc", ptype="acquired")
        inv = RecordingInvoker()
        ref = RecordingRefresher()
        m = self._ev("raw/ERE/funded/ad_hoc/acquired_001/2026-01-31/_READY.json", inv,
                     schema=self.fp, refresher=ref)
        self.assertEqual(m["decision"], "deterministic")
        self.assertEqual(m["status"], "processed")
        self.assertEqual(inv.calls[0]["source_portfolio_type"], "acquired")
        # Assembler refresh ran for the funded pack
        self.assertEqual(len(ref.calls), 1)
        self.assertEqual(ref.calls[0]["source_portfolio_id"], "acquired_001")
        self.assertEqual(m["central_canonical_path"], "/platform/central_canonical.csv")

    # ---- pipeline success does NOT trigger the platform assembler refresh ---
    def test_pipeline_success_no_refresh(self):
        self._seed_active(pid="direct_001", dataset="pipeline", frequency="weekly")
        inv = RecordingInvoker()
        ref = RecordingRefresher()
        m = self._ev("raw/ERE/pipeline/weekly/direct_001/2026-W05/_READY.json", inv,
                     schema=self.fp, refresher=ref)
        self.assertEqual(m["status"], "processed")
        self.assertEqual(len(ref.calls), 0)            # funded-only refresh


if __name__ == "__main__":
    unittest.main()
