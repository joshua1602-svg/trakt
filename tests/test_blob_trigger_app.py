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


# --------------------------------------------------------------------------- #
# 15. Azure-safe runtime output paths (TRAKT_TRIGGER_OUT propagation)
# --------------------------------------------------------------------------- #

import os
from unittest import mock

from apps.blob_trigger_app import runtime_paths as RP
from apps.blob_trigger_app.orchestrator_invoke import default_orchestrator_invoker


class TestRuntimePaths(unittest.TestCase):

    def test_explicit_wins(self):
        self.assertEqual(RP.resolve_output_root("/tmp/x/y"), "/tmp/x/y")

    def test_env_respected(self):
        with mock.patch.dict(os.environ, {"TRAKT_TRIGGER_OUT": "/tmp/trakt/blob_trigger"},
                             clear=False):
            self.assertEqual(RP.resolve_output_root(), "/tmp/trakt/blob_trigger")

    def test_azure_default_when_no_env(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("TRAKT_TRIGGER_OUT",)}
        env["WEBSITE_INSTANCE_ID"] = "abc123"     # simulate Azure
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertTrue(RP.running_in_azure())
            self.assertEqual(RP.resolve_output_root(), "/tmp/trakt/blob_trigger")

    def test_local_default(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("TRAKT_TRIGGER_OUT", "WEBSITE_INSTANCE_ID", "WEBSITE_SITE_NAME")}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertFalse(RP.running_in_azure())
            self.assertEqual(RP.resolve_output_root(), "out/blob_trigger")


class _FakeState:
    """Stand-in RunState capturing the out_root it was handed."""
    def __init__(self, out_root):
        self.run_id = "orun_test"
        self.status = "done"
        self.out_root = out_root
        self.central_canonical_path = str(Path(out_root) / "orun_test" / "out_platform" / "central.csv")
        self.blockers = []

    def state_path(self):
        return Path(self.out_root) / self.run_id / "run_state.json"


class TestTriggerOutPropagation(unittest.TestCase):
    """TRAKT_TRIGGER_OUT must reach orchestrator state/output paths and the
    assembler refresh roots — nothing lands under repo-root out/."""

    TRIGGER_OUT = "/tmp/trakt/blob_trigger"

    def test_invoker_forwards_out_dir_as_orchestrator_out_root(self):
        captured = {}

        def fake_run_orchestration(client_id, portfolios, *, out_root, **kw):
            captured["out_root"] = out_root
            captured["state_path"] = str(_FakeState(out_root).state_path())
            return _FakeState(out_root)

        with mock.patch("engine.orchestrator_agent.run_orchestration",
                        fake_run_orchestration):
            result = default_orchestrator_invoker(
                processing_mode="deterministic", client_id="ERE",
                source_portfolio_id="direct_001", source_portfolio_type="direct",
                dataset="funded", frequency="monthly", reporting_period="2026-01-31",
                input_path="/tmp/pack", target="mi", run_regime=False,
                mapping_config_path=None, out_dir=self.TRIGGER_OUT)
        # orchestrator state + run dir derive from the configured writable root
        self.assertEqual(captured["out_root"], self.TRIGGER_OUT)
        self.assertTrue(captured["state_path"].startswith(self.TRIGGER_OUT + "/"))
        self.assertTrue(result["central_canonical_path"].startswith(self.TRIGGER_OUT + "/"))
        self.assertFalse(captured["out_root"].startswith("out/"))

    def test_router_propagates_out_dir_to_invoker_assembler_and_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = str(Path(td) / "blob_trigger")   # stands in for TRAKT_TRIGGER_OUT
            reg = SourceRegistry(Path(out_dir) / "r.json")
            reg.upsert(SourceRecord(
                client_id="ERE", source_portfolio_id="direct_001", dataset="funded",
                frequency="monthly", source_portfolio_type="direct",
                approved_mapping_id="m1", mapping_config_path="config/m1.yaml",
                expected_schema_fingerprint=_fp().fingerprint, status=STATUS_ACTIVE))
            inv = RecordingInvoker()
            ref = RecordingRefresher()
            m = R.handle_blob_event(
                "raw/ERE/funded/monthly/direct_001/2026-01-31/_READY.json",
                registry=reg, out_dir=out_dir, pack_marker="_READY.json",
                schema_info=_fp(), orchestrator_invoker=inv,
                assembler_refresher=ref, now=_NOW)
            self.assertEqual(m["status"], "processed")
            # orchestrator told to write under the configured root
            self.assertEqual(inv.calls[0]["out_dir"], out_dir)
            # assembler accepted/platform roots derive from the same root
            self.assertTrue(ref.calls[0]["accepted_root"].startswith(out_dir))
            self.assertTrue(ref.calls[0]["platform_out_dir"].startswith(out_dir))
            # event manifest written under the configured root, not repo-root out/
            self.assertTrue((Path(out_dir) / f"{m['event_id']}.json").exists())


# --------------------------------------------------------------------------- #
# 16. Production folder structure (7-seg with source_book_type + legacy 6-seg)
# --------------------------------------------------------------------------- #

class TestSmokeBehaviourPreserved(unittest.TestCase):
    """Requirement #1: the confirmed smoke path still stops as pending_review."""

    def test_first_time_source_stops_pending_review(self):
        for blob in (
            # documented 6-seg smoke path …
            "raw-v2/ERE/funded/monthly/direct_001/2025-11-30/_READY.json",
            # … and the new 7-seg production path
            "raw-v2/ERE/direct/funded/monthly/direct_001/2025-11-30/_READY.json",
        ):
            with tempfile.TemporaryDirectory() as td:
                reg = SourceRegistry(Path(td) / "r.json")
                m = R.handle_blob_event(
                    blob, registry=reg, out_dir=td, container="raw-v2",
                    pack_marker="_READY.json", schema_info=_fp(),
                    orchestrator_invoker=RecordingInvoker(status="halted"), now=_NOW)
                self.assertEqual(m["decision"], "source_onboarding", blob)
                self.assertEqual(m["status"], "pending_review", blob)
                self.assertEqual(m["event_decision"], "new_source_pending_review", blob)


class TestFolderStructure(unittest.TestCase):

    def test_direct_funded_monthly(self):
        p = parse_blob_path(
            "raw-v2/ERE/direct/funded/monthly/direct_001/2025-11-30/LoanExtract.csv",
            container="raw-v2")
        self.assertEqual((p.client_id, p.source_book_type, p.dataset, p.frequency),
                         ("ERE", "direct", "funded", "monthly"))
        self.assertEqual(p.source_portfolio_id, "direct_001")
        self.assertFalse(p.is_legacy_path)

    def test_direct_pipeline_weekly(self):
        p = parse_blob_path(
            "raw-v2/ERE/direct/pipeline/weekly/direct_001/2025-W48/Pipe.csv",
            container="raw-v2")
        self.assertEqual((p.source_book_type, p.dataset, p.frequency, p.reporting_period),
                         ("direct", "pipeline", "weekly", "2025-W48"))

    def test_acquired_funded_monthly(self):
        p = parse_blob_path(
            "raw-v2/ERE/acquired/funded/monthly/acquired_001/2025-11-30/Loan.csv",
            container="raw-v2")
        self.assertEqual(p.source_book_type, "acquired")
        self.assertEqual(p.source_portfolio_id, "acquired_001")

    def test_acquired_ad_hoc(self):
        p = parse_blob_path(
            "raw-v2/ERE/acquired/funded/ad_hoc/acquired_001/2025-11-30/Loan.csv",
            container="raw-v2")
        self.assertEqual((p.source_book_type, p.frequency), ("acquired", "ad_hoc"))

    def test_invalid_source_book_type_rejected(self):
        with self.assertRaises(PathParseError):
            parse_blob_path(
                "raw-v2/ERE/wholesale/funded/monthly/direct_001/2025-11-30/x.csv",
                container="raw-v2")

    def test_invalid_dataset_and_frequency_rejected(self):
        for bad in (
            "raw-v2/ERE/direct/BADSET/monthly/direct_001/2025-11-30/x.csv",
            "raw-v2/ERE/direct/funded/yearly/direct_001/2025-11-30/x.csv",
            "raw-v2/ERE/direct/funded/monthly/direct_001/not-a-date/x.csv",
        ):
            with self.assertRaises(PathParseError, msg=bad):
                parse_blob_path(bad, container="raw-v2")

    def test_book_type_pid_inconsistency_rejected(self):
        with self.assertRaises(PathParseError):
            parse_blob_path(
                "raw-v2/ERE/direct/funded/monthly/acquired_001/2025-11-30/x.csv",
                container="raw-v2")

    def test_legacy_six_segment_still_supported(self):
        p = parse_blob_path("raw/ERE/funded/monthly/direct_001/2026-09-30/x.csv")
        self.assertTrue(p.is_legacy_path)
        self.assertEqual(p.source_book_type, "direct")   # derived from pid


# --------------------------------------------------------------------------- #
# 17. Storage abstraction + registry persistence
# --------------------------------------------------------------------------- #

from apps.blob_trigger_app.storage import Storage, open_storage, split_blob_uri
from apps.blob_trigger_app.layout import Layout
from apps.blob_trigger_app.persistence import ProductionPersistence
from apps.blob_trigger_app import approvals as APP


class TestStorage(unittest.TestCase):

    def test_blob_uri_roundtrip_on_filesystem(self):
        with tempfile.TemporaryDirectory() as td:
            s = Storage(local_root=td)
            uri = "blob://trakt-state/registry/source_registry.yaml"
            s.write_text(uri, "sources: []\n")
            self.assertTrue(s.exists(uri))
            self.assertEqual(s.read_text(uri), "sources: []\n")
            self.assertEqual(split_blob_uri(uri),
                             ("trakt-state", "registry/source_registry.yaml"))
            self.assertIn(uri, s.list("blob://trakt-state/registry"))

    def test_registry_read_write_via_storage(self):
        with tempfile.TemporaryDirectory() as td:
            s = Storage(local_root=td)
            uri = "blob://trakt-state/registry/source_registry.yaml"
            reg = SourceRegistry(uri, storage=s)
            reg.upsert(SourceRecord(
                client_id="ERE", source_portfolio_id="direct_001", dataset="funded",
                frequency="monthly", approved_mapping_id="m1", status=STATUS_ACTIVE,
                expected_schema_fingerprint="sha256:abc"))
            # fresh instance reads it back through storage
            reg2 = SourceRegistry(uri, storage=s)
            rec = reg2.lookup("ERE", "direct_001", "funded", "monthly")
            self.assertIsNotNone(rec)
            self.assertEqual(rec.approved_mapping_id, "m1")


# --------------------------------------------------------------------------- #
# 18. Approval workflow + persistence wiring (end-to-end on filesystem store)
# --------------------------------------------------------------------------- #

class WritingRefresher:
    """Refresher stub that writes accepted + platform files like the real one."""
    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        self.calls = []

    def __call__(self, *, client_id, source_portfolio_id, canonical_path,
                 accepted_root, platform_out_dir, target, run_regime, regime):
        self.calls.append(locals())
        acc = Path(accepted_root) / client_id / f"{source_portfolio_id}_canonical_typed.csv"
        acc.parent.mkdir(parents=True, exist_ok=True)
        acc.write_text("source_portfolio_id\n" + source_portfolio_id + "\n")
        plat = Path(platform_out_dir) / "platform_canonical_typed.csv"
        plat.parent.mkdir(parents=True, exist_ok=True)
        plat.write_text("source_portfolio_id\n" + source_portfolio_id + "\n")
        return {"central_canonical_path": str(plat),
                "portfolios": [source_portfolio_id], "assembler_run_id": "asm_t"}


def _stub_regime_runner(*, central_canonical_path, client_id, period, regime, out_dir):
    d = Path(out_dir) / client_id / period
    d.mkdir(parents=True, exist_ok=True)
    (d / "ESMA_Annex2_underlying_exposures.csv").write_text("clean_esma\n")
    (d / "provenance_companion.csv").write_text("provenance\n")
    return {"output_dir": str(d), "ok": True}


class TestApprovalAndPersistence(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.out = str(Path(self.root) / "scratch")
        self.storage = Storage(local_root=str(Path(self.root) / "blobstore"))
        self.layout = Layout()           # blob://trakt-state, processed-v2 defaults
        self.persistence = ProductionPersistence(self.storage, self.layout)
        self.registry = self.persistence.load_registry()
        self.fp = _fp()

    def tearDown(self):
        self._tmp.cleanup()

    def _marker(self, container, blob, inv, *, meta=None, refresher=None,
                regime_runner=None):
        return R.handle_blob_event(
            blob, registry=self.registry, out_dir=self.out, container=container,
            pack_marker="_READY.json", schema_info=self.fp, marker_metadata=meta,
            orchestrator_invoker=inv, assembler_refresher=(refresher or RecordingRefresher()),
            persistence=self.persistence, regime_runner=regime_runner, now=_NOW)

    def test_new_source_creates_pending_approval_artifact(self):
        inv = RecordingInvoker(status="halted")
        m = self._marker(
            "raw-v2", "raw-v2/ERE/acquired/funded/ad_hoc/acquired_001/2025-11-30/_READY.json",
            inv, meta={"acquisition_date": "2025-10-31", "seller_name": "BigBank"})
        self.assertEqual(m["event_decision"], "new_source_pending_review")
        self.assertIsNotNone(m["approval_id"])
        pend = APP.list_pending(self.storage, self.layout)
        self.assertEqual(len(pend), 1)
        art = pend[0]
        self.assertEqual(art["kind"], "new_source")
        self.assertEqual(art["schema_fingerprint"], self.fp.fingerprint)
        self.assertEqual(art["source_book_type"], "acquired")
        self.assertEqual(art["source_metadata"]["seller_name"], "BigBank")
        # event manifest persisted durably
        self.assertTrue(self.storage.exists(self.layout.event_uri(m["event_id"])))

    def test_schema_drift_creates_pending_approval_with_both_fingerprints(self):
        self.registry.upsert(SourceRecord(
            client_id="ERE", source_portfolio_id="direct_001", dataset="funded",
            frequency="monthly", source_portfolio_type="direct", approved_mapping_id="m1",
            mapping_config_path="config/m1.yaml",
            expected_schema_fingerprint="sha256:OLD", status=STATUS_ACTIVE))
        inv = RecordingInvoker()
        m = self._marker(
            "raw-v2", "raw-v2/ERE/direct/funded/monthly/direct_001/2025-11-30/_READY.json", inv)
        self.assertEqual(m["event_decision"], "schema_drift_pending_review")
        art = APP.show(self.storage, self.layout, m["approval_id"])
        self.assertEqual(art["kind"], "schema_drift")
        self.assertEqual(art["prior_schema_fingerprint"], "sha256:OLD")
        self.assertEqual(art["schema_fingerprint"], self.fp.fingerprint)
        self.assertEqual(len(inv.calls), 0)              # never processed

    def test_approve_promote_reject_and_deterministic_only_after_approval(self):
        blob = "raw-v2/ERE/acquired/funded/ad_hoc/acquired_001/2025-11-30/_READY.json"
        # 1) new source → pending, not processed
        inv = RecordingInvoker(status="halted")
        m1 = self._marker("raw-v2", blob, inv)
        self.assertEqual(m1["status"], "pending_review")
        approval_id = m1["approval_id"]
        # 2) approve + promote → active registry entry
        APP.approve(self.storage, self.layout, approval_id,
                    mapping_id="ere_acq1_v1", mapping_config_path="config/acq1.yaml")
        rec = APP.promote(self.storage, self.layout, self.registry, approval_id)
        self.assertEqual(rec.status, STATUS_ACTIVE)
        self.assertEqual(rec.expected_schema_fingerprint, self.fp.fingerprint)
        # 3) re-trigger (force, since the first pending run recorded the pack)
        inv2 = RecordingInvoker(status="done")
        ref = WritingRefresher(self.out)
        m2 = self._marker("raw-v2", blob, inv2,
                          meta={"force_reprocess": True}, refresher=ref,
                          regime_runner=_stub_regime_runner)
        self.assertEqual(m2["decision"], "deterministic")
        self.assertEqual(m2["status"], "processed")
        self.assertEqual(inv2.calls[0]["mapping_config_path"], "config/acq1.yaml")
        # accepted + platform canonical persisted durably
        self.assertTrue(self.storage.exists(self.layout.accepted_uri("ERE", "acquired_001")))
        self.assertTrue(self.storage.exists(self.layout.platform_latest_uri("ERE")))
        self.assertTrue(self.storage.exists(
            self.layout.platform_period_uri("ERE", "2025-11-30")))

    def test_reject_marks_rejected(self):
        blob = "raw-v2/ERE/acquired/funded/ad_hoc/acquired_002/2025-11-30/_READY.json"
        m = self._marker("raw-v2", blob, RecordingInvoker(status="halted"))
        APP.reject(self.storage, self.layout, m["approval_id"], reason="seller unverified")
        art = APP.show(self.storage, self.layout, m["approval_id"])
        self.assertEqual(art["status"], "rejected")
        self.assertEqual(art["reject_reason"], "seller unverified")
        # promoting a rejected approval is refused
        with self.assertRaises(ValueError):
            APP.promote(self.storage, self.layout, self.registry, m["approval_id"])

    def test_regime_outputs_persisted_for_funded_all(self):
        self.registry.upsert(SourceRecord(
            client_id="ERE", source_portfolio_id="direct_001", dataset="funded",
            frequency="monthly", source_portfolio_type="direct", approved_mapping_id="m1",
            mapping_config_path="config/m1.yaml",
            expected_schema_fingerprint=self.fp.fingerprint, regime_required=True,
            status=STATUS_ACTIVE))
        ref = WritingRefresher(self.out)
        m = self._marker(
            "raw-v2", "raw-v2/ERE/direct/funded/monthly/direct_001/2025-11-30/_READY.json",
            RecordingInvoker(status="done"), refresher=ref, regime_runner=_stub_regime_runner)
        self.assertEqual(m["selected_target"]["target"], "all")
        regime_uris = m["persisted"]["regime_uris"]
        self.assertTrue(any("ESMA_Annex2" in u for u in regime_uris))
        self.assertTrue(any("provenance_companion" in u for u in regime_uris))
        # under the regime prefix for the period
        for u in regime_uris:
            self.assertIn("processed-v2/regime/ERE/2025-11-30", u)

    def test_halted_run_manifest_carries_diagnostic_reason(self):
        # A recurring approved pack whose orchestrator run HALTS must not silently
        # report orchestrator_status=halted with a null central canonical — the
        # event manifest has to EXPLAIN the halt (stage/reason/run_state.json path).
        self.registry.upsert(SourceRecord(
            client_id="ERE", source_portfolio_id="direct_001", dataset="funded",
            frequency="monthly", source_portfolio_type="direct", approved_mapping_id="m1",
            mapping_config_path="config/m1.yaml",
            expected_schema_fingerprint=self.fp.fingerprint, status=STATUS_ACTIVE))

        class DiagnosticInvoker:
            def __call__(self, **kw):
                return {
                    "run_id": "orun_halt", "status": "halted",
                    "central_canonical_path": None,
                    "blockers": ["direct_001/transform: onboarding handoff not ready"],
                    "state_path": "/tmp/trakt/blob_trigger/orun_halt/run_state.json",
                    "diagnostics": {
                        "halt_stage": "direct_001/transform",
                        "halt_reason": "onboarding handoff not ready_for_transformation_validation",
                        "blocking_decisions": ["unresolved mapping gaps / blocking decisions"],
                        "registry_gap_count": 3,
                        "validation_errors": [],
                        "run_state_path": "/tmp/trakt/blob_trigger/orun_halt/run_state.json",
                    },
                }

        m = self._marker(
            "raw-v2", "raw-v2/ERE/direct/funded/monthly/direct_001/2025-11-30/_READY.json",
            DiagnosticInvoker())
        self.assertEqual(m["status"], "halted")
        self.assertIsNone(m["central_canonical_path"])
        diag = m["orchestrator_diagnostics"]
        self.assertEqual(diag["halt_stage"], "direct_001/transform")
        self.assertIn("ready_for_transformation_validation", diag["halt_reason"])
        self.assertEqual(diag["registry_gap_count"], 3)
        self.assertEqual(diag["run_state_path"],
                         "/tmp/trakt/blob_trigger/orun_halt/run_state.json")
        # The manifest explains WHY the central canonical is null.
        self.assertIsNotNone(diag["central_canonical_unavailable_reason"])
        self.assertIn("direct_001/transform", diag["central_canonical_unavailable_reason"])

    def test_mi_locator_reads_latest_platform_canonical(self):
        # persist a platform canonical, then resolve it through the MI locator
        self.registry.upsert(SourceRecord(
            client_id="ERE", source_portfolio_id="direct_001", dataset="funded",
            frequency="monthly", source_portfolio_type="direct", approved_mapping_id="m1",
            expected_schema_fingerprint=self.fp.fingerprint, status=STATUS_ACTIVE))
        ref = WritingRefresher(self.out)
        self._marker(
            "raw-v2", "raw-v2/ERE/direct/funded/monthly/direct_001/2025-11-30/_READY.json",
            RecordingInvoker(status="done"), refresher=ref)
        local = self.persistence.mi_latest_platform_path("ERE")
        self.assertIsNotNone(local)
        self.assertTrue(Path(local).exists())
        self.assertIn("platform_canonical_typed.csv", local)


# --------------------------------------------------------------------------- #
# 19. MI API resolution of the persisted platform canonical (env-gated)
# --------------------------------------------------------------------------- #

class TestMIPlatformResolution(unittest.TestCase):

    def test_mi_resolves_platform_uri_via_storage(self):
        from mi_agent_api import data_source as DS
        with tempfile.TemporaryDirectory() as td:
            blobroot = Path(td) / "blobstore"
            s = Storage(local_root=str(blobroot))
            layout = Layout()
            uri = layout.platform_latest_uri("ERE")
            s.write_text(uri, "source_portfolio_id\ndirect_001\n")
            env = {k: v for k, v in os.environ.items()
                   if not k.startswith("MI_AGENT_") and k != "WEBSITE_INSTANCE_ID"}
            env["TRAKT_LOCAL_BLOB_ROOT"] = str(blobroot)
            env["MI_AGENT_PLATFORM_URI"] = uri          # blob:// dir or file
            with mock.patch.dict(os.environ, env, clear=True):
                resolved = DS._resolve_platform_canonical()
            self.assertIsNotNone(resolved)
            self.assertTrue(Path(resolved).exists())


# --------------------------------------------------------------------------- #
# 20. Historical backfill (independent dated folders, idempotency, periods)
# --------------------------------------------------------------------------- #

class TestBackfill(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.out = str(Path(self.root) / "scratch")
        self.storage = Storage(local_root=str(Path(self.root) / "blobstore"))
        self.layout = Layout()
        self.persistence = ProductionPersistence(self.storage, self.layout)
        self.registry = self.persistence.load_registry()
        self.fp = _fp()
        self.registry.upsert(SourceRecord(
            client_id="ERE", source_portfolio_id="direct_001", dataset="funded",
            frequency="monthly", source_portfolio_type="direct", approved_mapping_id="m1",
            expected_schema_fingerprint=self.fp.fingerprint, status=STATUS_ACTIVE))

    def tearDown(self):
        self._tmp.cleanup()

    def test_twelve_monthly_folders_process_independently_and_keep_periods(self):
        inv = RecordingInvoker(status="done")
        months = [f"2025-{m:02d}-28" for m in range(1, 13)]
        for period in months:
            ref = WritingRefresher(self.out)
            blob = f"raw-v2/ERE/direct/funded/monthly/direct_001/{period}/_READY.json"
            m = R.handle_blob_event(
                blob, registry=self.registry, out_dir=self.out, container="raw-v2",
                pack_marker="_READY.json", schema_info=self.fp, orchestrator_invoker=inv,
                assembler_refresher=ref, persistence=self.persistence, now=_NOW)
            self.assertEqual(m["status"], "processed")
        self.assertEqual(len(inv.calls), 12)             # one run per dated folder
        # period-level platform artifacts preserved (not only latest)
        for period in months:
            self.assertTrue(self.storage.exists(
                self.layout.platform_period_uri("ERE", period)))
        self.assertTrue(self.storage.exists(self.layout.platform_latest_uri("ERE")))

    def test_duplicate_marker_idempotent(self):
        blob = "raw-v2/ERE/direct/funded/monthly/direct_001/2025-06-30/_READY.json"
        inv = RecordingInvoker(status="done")
        ref = WritingRefresher(self.out)
        common = dict(registry=self.registry, out_dir=self.out, container="raw-v2",
                      pack_marker="_READY.json", schema_info=self.fp,
                      orchestrator_invoker=inv, assembler_refresher=ref,
                      persistence=self.persistence, now=_NOW)
        R.handle_blob_event(blob, **common)
        m2 = R.handle_blob_event(blob, **common)
        self.assertEqual(m2["status"], "already_processed")
        self.assertEqual(len(inv.calls), 1)

    def test_weekly_pipeline_independent_no_regime(self):
        self.registry.upsert(SourceRecord(
            client_id="ERE", source_portfolio_id="direct_001", dataset="pipeline",
            frequency="weekly", source_portfolio_type="direct", approved_mapping_id="m1",
            expected_schema_fingerprint=self.fp.fingerprint, status=STATUS_ACTIVE))
        inv = RecordingInvoker(status="done")
        ref = WritingRefresher(self.out)
        m = R.handle_blob_event(
            "raw-v2/ERE/direct/pipeline/weekly/direct_001/2025-W48/_READY.json",
            registry=self.registry, out_dir=self.out, container="raw-v2",
            pack_marker="_READY.json", schema_info=self.fp,
            orchestrator_invoker=inv, assembler_refresher=ref,
            persistence=self.persistence, regime_runner=_stub_regime_runner, now=_NOW)
        self.assertFalse(m["selected_target"]["run_regime"])
        self.assertEqual(len(ref.calls), 0)              # pipeline → no platform refresh
        # no regime persisted
        self.assertNotIn("regime_uris", (m.get("persisted") or {}))


# --------------------------------------------------------------------------- #
# 21. Exception logging — first failing persistence op is diagnosable
# --------------------------------------------------------------------------- #

class _BoomStorage(Storage):
    """Storage whose writes always fail (simulates a Blob 403 / missing container)."""
    def exists(self, uri):
        return False
    def write_text(self, uri, text):
        raise RuntimeError("blob write denied (simulated)")


class TestExceptionLogging(unittest.TestCase):

    def test_storage_write_guard_logs_uri_and_traceback(self):
        with tempfile.TemporaryDirectory() as td:
            s = Storage(local_root=td)
            uri = "blob://processed-v2/accepted/ERE/x_canonical_typed.csv"
            with self.assertLogs("trakt.blob_trigger.storage", level="ERROR") as cm, \
                    self.assertRaises(Exception):
                s.upload_file(str(Path(td) / "missing.csv"), uri)   # source absent → fails
            blob = "\n".join(cm.output)
            self.assertIn("STORAGE WRITE FAILED", blob)
            self.assertIn(uri, blob)
            self.assertIn("Traceback", blob)

    def test_registry_save_failure_logs_uri_and_traceback(self):
        reg = SourceRegistry("blob://trakt-state/registry/source_registry.yaml",
                             storage=_BoomStorage(local_root="/tmp"))
        with self.assertLogs("trakt.blob_trigger.source_registry", level="ERROR") as cm, \
                self.assertRaises(RuntimeError):
            reg.upsert(SourceRecord(client_id="ERE", source_portfolio_id="acquired_001",
                                    dataset="funded", frequency="ad_hoc"))
        blob = "\n".join(cm.output)
        self.assertIn("REGISTRY SAVE FAILED", blob)
        self.assertIn("trakt-state/registry/source_registry.yaml", blob)
        self.assertIn("Traceback", blob)

    def test_router_wraps_uncaught_and_logs_blob_path_and_traceback(self):
        with tempfile.TemporaryDirectory() as td:
            reg = SourceRegistry("blob://trakt-state/registry/source_registry.yaml",
                                 storage=_BoomStorage(local_root=td))
            blob_path = "raw-v2/ERE/acquired/funded/ad_hoc/acquired_001/2025-11-30/_READY.json"
            with self.assertLogs("trakt.blob_trigger", level="ERROR") as cm, \
                    self.assertRaises(RuntimeError):
                R.handle_blob_event(
                    blob_path, registry=reg, out_dir=td, container="raw-v2",
                    pack_marker="_READY.json", schema_info=_fp(),
                    orchestrator_invoker=RecordingInvoker(status="halted"), now=_NOW)
            blob = "\n".join(cm.output)
            self.assertIn("BLOB-TRIGGER ROUTER FAILED", blob)
            self.assertIn(blob_path, blob)
            self.assertIn("Traceback", blob)

    def test_persistence_op_logs_uri_on_failure(self):
        persistence = ProductionPersistence(_BoomStorage(local_root="/tmp"), Layout())
        with self.assertLogs("trakt.blob_trigger.persistence", level="ERROR") as cm, \
                self.assertRaises(RuntimeError):
            persistence.persist_event_manifest({"event_id": "evt_x", "k": "v"})
        blob = "\n".join(cm.output)
        self.assertIn("PERSISTENCE FAILED", blob)
        self.assertIn("op=persist_event_manifest", blob)
        self.assertIn("trakt-state/events/evt_x.json", blob)


# --------------------------------------------------------------------------- #
# 22. Storage backend selection (Azure → BlobStorage, never read-only fs)
# --------------------------------------------------------------------------- #

from apps.blob_trigger_app.storage import (
    BlobStorage, decide_backend, open_storage as _open_storage, running_in_azure)

_FAKE_CONN = "DefaultEndpointsProtocol=https;AccountName=x;AccountKey=y==;EndpointSuffix=core.windows.net"


def _clean_env(**overrides):
    base = {k: v for k, v in os.environ.items()
            if k not in ("WEBSITE_INSTANCE_ID", "WEBSITE_SITE_NAME",
                         "TRAKT_STORAGE_BACKEND", "TRAKT_BLOB_CONNECTION",
                         "AzureWebJobsStorage", "TRAKT_LOCAL_BLOB_ROOT")}
    base.update(overrides)
    return base


class TestBackendSelection(unittest.TestCase):

    def test_azure_site_name_only_selects_blob(self):
        # WEBSITE_SITE_NAME present, WEBSITE_INSTANCE_ID absent (Linux/Flex plans):
        # the old code stayed on filesystem; now it must select Azure Blob.
        env = _clean_env(WEBSITE_SITE_NAME="trakt-func", TRAKT_BLOB_CONNECTION=_FAKE_CONN)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertTrue(running_in_azure())
            d = decide_backend()
            self.assertEqual(d["backend"], "azure_blob")
            self.assertTrue(d["connection_detected"])
            self.assertIsInstance(_open_storage(), BlobStorage)

    def test_azure_instance_id_selects_blob(self):
        env = _clean_env(WEBSITE_INSTANCE_ID="abc", TRAKT_BLOB_CONNECTION=_FAKE_CONN)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertIsInstance(_open_storage(), BlobStorage)

    def test_azure_without_connection_raises_not_filesystem(self):
        # In Azure with no connection string, refuse to silently use the
        # read-only wwwroot filesystem — fail loud.
        env = _clean_env(WEBSITE_SITE_NAME="trakt-func")
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(decide_backend()["backend"], "azure_blob")
            with self.assertRaises(ValueError):
                _open_storage()

    def test_azure_falls_back_to_azurewebjobsstorage(self):
        env = _clean_env(WEBSITE_SITE_NAME="trakt-func", AzureWebJobsStorage=_FAKE_CONN)
        with mock.patch.dict(os.environ, env, clear=True):
            d = decide_backend()
            self.assertTrue(d["connection_detected"])
            self.assertEqual(d["connection_source"], "AzureWebJobsStorage")
            self.assertIsInstance(_open_storage(), BlobStorage)

    def test_explicit_file_backend_wins_even_in_azure(self):
        env = _clean_env(WEBSITE_SITE_NAME="trakt-func", TRAKT_BLOB_CONNECTION=_FAKE_CONN,
                         TRAKT_STORAGE_BACKEND="file", TRAKT_LOCAL_BLOB_ROOT="/tmp/x")
        with mock.patch.dict(os.environ, env, clear=True):
            d = decide_backend()
            self.assertEqual(d["backend"], "filesystem")
            s = _open_storage()
            self.assertIsInstance(s, Storage)
            self.assertNotIsInstance(s, BlobStorage)

    def test_connection_present_selects_blob_outside_azure(self):
        env = _clean_env(TRAKT_BLOB_CONNECTION=_FAKE_CONN)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertFalse(running_in_azure())
            self.assertEqual(decide_backend()["backend"], "azure_blob")

    def test_local_default_is_filesystem(self):
        env = _clean_env(TRAKT_LOCAL_BLOB_ROOT="/tmp/lbr")
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertFalse(running_in_azure())
            d = decide_backend()
            self.assertEqual(d["backend"], "filesystem")
            self.assertFalse(d["connection_detected"])
            self.assertNotIsInstance(_open_storage(), BlobStorage)

    def test_open_storage_logs_selection(self):
        env = _clean_env(WEBSITE_SITE_NAME="trakt-func", TRAKT_BLOB_CONNECTION=_FAKE_CONN)
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertLogs("trakt.blob_trigger.storage", level="INFO") as cm:
                _open_storage()
            blob = "\n".join(cm.output)
            self.assertIn("STORAGE BACKEND SELECTED", blob)
            self.assertIn("backend=azure_blob", blob)
            self.assertIn("connection_detected=True", blob)


# --------------------------------------------------------------------------- #
# 23. Funded MI route → full production Orchestrator path (CLI parity)
# --------------------------------------------------------------------------- #

class TestFullPipelineRouting(unittest.TestCase):
    """The Azure funded-MI route invokes the full onboard→transform→validate→
    stamp path (full_pipeline), same as the CLI; pipeline stays lean."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.out = self._tmp.name
        self.registry = SourceRegistry(Path(self.out) / "r.json")
        self.fp = _fp()

    def tearDown(self):
        self._tmp.cleanup()

    def _seed(self, pid, dataset, frequency, ptype="direct"):
        self.registry.upsert(SourceRecord(
            client_id="ERE", source_portfolio_id=pid, dataset=dataset, frequency=frequency,
            source_portfolio_type=ptype, approved_mapping_id="m1",
            mapping_config_path="config/m1.yaml",
            expected_schema_fingerprint=self.fp.fingerprint, status=STATUS_ACTIVE))

    def _run(self, blob, inv, meta=None):
        return R.handle_blob_event(
            blob, registry=self.registry, out_dir=self.out, container="raw-v2",
            pack_marker="_READY.json", schema_info=self.fp, marker_metadata=meta,
            orchestrator_invoker=inv, assembler_refresher=RecordingRefresher(), now=_NOW)

    def test_funded_monthly_uses_full_pipeline(self):
        self._seed("direct_001", "funded", "monthly")
        inv = RecordingInvoker()
        m = self._run("raw-v2/ERE/direct/funded/monthly/direct_001/2025-11-30/_READY.json", inv)
        self.assertEqual(m["decision"], "deterministic")
        self.assertTrue(inv.calls[0]["full_pipeline"])         # Gate 2/3 pipeline
        self.assertFalse(inv.calls[0]["force_publish"])
        self.assertTrue(m["full_pipeline"])

    def test_weekly_pipeline_stays_lean(self):
        self._seed("direct_001", "pipeline", "weekly")
        inv = RecordingInvoker()
        self._run("raw-v2/ERE/direct/pipeline/weekly/direct_001/2025-W48/_READY.json", inv)
        self.assertFalse(inv.calls[0]["full_pipeline"])        # lean MI

    def test_force_publish_threaded_from_ready_json(self):
        self._seed("direct_001", "funded", "monthly")
        inv = RecordingInvoker()
        self._run("raw-v2/ERE/direct/funded/monthly/direct_001/2025-11-30/_READY.json",
                  inv, meta={"force_publish": True})
        self.assertTrue(inv.calls[0]["force_publish"])

    def test_new_funded_source_onboarding_also_full_pipeline(self):
        # New source → discovery (LLM onboarding), but STILL the full pipeline.
        inv = RecordingInvoker(status="halted")
        m = self._run("raw-v2/ERE/acquired/funded/ad_hoc/acquired_001/2025-11-30/_READY.json", inv)
        self.assertEqual(inv.calls[0]["processing_mode"], "source_onboarding")
        self.assertTrue(inv.calls[0]["full_pipeline"])
        self.assertEqual(m["status"], "pending_review")        # halt at approval

    def _capture_invoke(self, *, target, full_pipeline):
        # Capture the adapters/flags the default invoker hands to run_orchestration.
        from apps.blob_trigger_app.orchestrator_invoke import default_orchestrator_invoker
        captured = {}

        class _FakeState:
            run_id = "r"; status = "done"; central_canonical_path = None; blockers = []
            def state_path(self):
                return Path("/tmp/x")

        def fake_run(client_id, portfolios, *, adapters, full_pipeline, force_publish, **kw):
            captured.update(full_pipeline=full_pipeline, force_publish=force_publish,
                            onboarding_mode=adapters.onboarding_mode,
                            processing_mode=adapters.processing_mode)
            return _FakeState()

        with mock.patch("engine.orchestrator_agent.run_orchestration", fake_run):
            default_orchestrator_invoker(
                processing_mode="deterministic", client_id="ERE",
                source_portfolio_id="direct_001", source_portfolio_type="direct",
                dataset="funded", frequency="monthly", reporting_period="2025-11-30",
                input_path="/tmp/pack", target=target, run_regime=(target != "mi"),
                mapping_config_path="config/m1.yaml", out_dir="/tmp/out",
                full_pipeline=full_pipeline, force_publish=True)
        return captured

    def test_funded_mi_full_pipeline_uses_MI_contract_not_annex2(self):
        # THE fix: full pipeline (depth) does NOT change the contract. target=mi
        # → mi_only onboarding (MI contract), even with full_pipeline=True.
        cap = self._capture_invoke(target="mi", full_pipeline=True)
        self.assertTrue(cap["full_pipeline"])                 # Gate 2/3 run (depth)
        self.assertEqual(cap["onboarding_mode"], "mi_only")   # MI contract, NOT regulatory_mi
        self.assertTrue(cap["force_publish"])
        self.assertEqual(cap["processing_mode"], "deterministic")  # no LLM on approved

    def test_regime_target_uses_annex2_contract(self):
        cap = self._capture_invoke(target="all", full_pipeline=True)
        self.assertEqual(cap["onboarding_mode"], "regulatory_mi")   # Annex 2 / combined
        self.assertTrue(cap["full_pipeline"])

    def test_run_diagnostics_pins_first_halted_step_from_run_state(self):
        # _run_diagnostics walks a REAL RunState and pins the first halted stage,
        # its reason, registry-gap count and the run_state.json path — the data the
        # manifest surfaces to explain a null central canonical.
        from apps.blob_trigger_app.orchestrator_invoke import _run_diagnostics
        from engine.orchestrator_agent.state import (
            RunState, PortfolioState, STEP_DONE, STEP_HALTED)
        st = RunState(run_id="orun_it", client_id="ERE", target="mi",
                      out_root="/tmp/trakt/blob_trigger", created_at=_NOW)
        p = PortfolioState(source_portfolio_id="direct_001", source_portfolio_type="direct")
        onb = p.step("onboard")
        onb.status = STEP_DONE
        onb.readiness = {"registry_gap_count": 2}
        tr = p.step("transform")
        tr.status = STEP_HALTED
        tr.blockers = ["onboarding handoff not ready_for_transformation_validation — pending review"]
        st.portfolios.append(p)
        st.status = STEP_HALTED
        st.blockers = ["direct_001/transform: onboarding handoff not ready"]

        diag = _run_diagnostics(st)
        self.assertEqual(diag["halt_stage"], "direct_001/transform")
        self.assertIn("ready_for_transformation_validation", diag["halt_reason"])
        self.assertEqual(diag["blocking_decisions"], tr.blockers)
        self.assertEqual(diag["registry_gap_count"], 2)
        self.assertTrue(diag["run_state_path"].endswith("orun_it/run_state.json"))


if __name__ == "__main__":
    unittest.main()
