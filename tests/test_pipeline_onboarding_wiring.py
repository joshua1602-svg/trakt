#!/usr/bin/env python3
"""tests/test_pipeline_onboarding_wiring.py

The blob/backfill route must onboard a weekly PIPELINE pack the way the CLI does:
feed the central PIPELINE tape (18a) — not the funded mi_semantics lender-tape
contract — so it does not halt on absent funded fields, and approve/promote keys
a SEPARATE pipeline/weekly registry record without touching funded/monthly.

Covers acceptance criteria 2, 5, 6, 8, 9, 10.

Run: python -m unittest tests.test_pipeline_onboarding_wiring
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from apps.blob_trigger_app import ops as OPS
from apps.blob_trigger_app import router as R
from apps.blob_trigger_app.layout import Layout
from apps.blob_trigger_app.persistence import ProductionPersistence
from apps.blob_trigger_app.repin import repin_source
from apps.blob_trigger_app.schema_fingerprint import fingerprint_pack
from apps.blob_trigger_app.source_registry import (
    SourceRegistry, STATUS_ACTIVE)
from apps.blob_trigger_app.storage import Storage
from engine.orchestrator_agent.adapters import RealAgentAdapters, PortfolioSpec

_PIPE = ["deal_id", "amount", "stage", "status", "expected_close", "interest_rate"]


def _pipeline_res(**over):
    d = {"mode": "mi_only",
         "central_lender_tape_created": False, "central_lender_tape_path": "",
         "central_pipeline_tape_created": True,
         "central_pipeline_tape_path": "/tmp/run/18a_central_pipeline_tape.csv",
         "pipeline_count": 42, "pipeline_source_dir": "/tmp/run/output/pipeline",
         "pipeline_sources_materialised": ["M2L KFI and Pipeline.xlsx"],
         "loan_count": 0}
    d.update(over)
    return d


class TestAdapterPipelineDeliverable(unittest.TestCase):
    """Criteria 5, 6, 10 — adapter selects the pipeline tape, does not gate on the
    empty lender tape, and never invokes the funded coverage/LLM for pipeline."""

    def _run_onboard(self, dataset, res, deterministic=False):
        captured = {}
        with tempfile.TemporaryDirectory() as td:
            with mock.patch("engine.onboarding_agent.workflow.run_operator_workflow",
                            side_effect=lambda **kw: captured.update(kw) or {}), \
                 mock.patch("engine.onboarding_agent.storage_paths.resolve_run_paths",
                            return_value=object()), \
                 mock.patch("engine.onboarding_agent.central_tape_builder.build_central_tapes",
                            return_value=res), \
                 mock.patch.object(RealAgentAdapters, "_build_mi_handoff", return_value="/tmp/h.json"):
                ad = RealAgentAdapters(
                    client_name="ERE", onboarding_mode="mi_only",
                    processing_mode=("deterministic" if deterministic else "source_onboarding"),
                    dataset=dataset, enable_llm_mapping_review=True, enable_llm_advisor=True)
                spec = PortfolioSpec(source_portfolio_id="direct_001", input=str(td))
                result = ad.onboard(spec, Path(td))
        return result, captured

    def test_pipeline_returns_pipeline_tape_not_gated_on_lender(self):
        result, captured = self._run_onboard("pipeline", _pipeline_res())
        self.assertTrue(result.ok)                       # (6) does NOT halt
        self.assertFalse(result.blocking)
        self.assertEqual(result.output_path,             # (5) pipeline tape is the deliverable
                         "/tmp/run/18a_central_pipeline_tape.csv")
        self.assertEqual(result.readiness["target_contract"], "pipeline_field_contract")
        # (10) funded coverage / advisor / mapping resolver are OFF for pipeline.
        self.assertFalse(captured["enable_mapping_review"])
        self.assertFalse(captured["enable_llm_target_advisor"])
        self.assertFalse(captured["enable_llm_mapping_review"])

    def test_pipeline_with_no_pipeline_tape_reports_actionable_blocker(self):
        result, _ = self._run_onboard(
            "pipeline", _pipeline_res(central_pipeline_tape_created=False,
                                      central_pipeline_tape_path=""))
        self.assertFalse(result.ok)
        self.assertIn("central pipeline tape", result.blockers[0])

    def test_funded_still_uses_lender_tape(self):
        res = _pipeline_res(central_lender_tape_created=True,
                            central_lender_tape_path="/tmp/run/18_central_lender_tape.csv")
        result, captured = self._run_onboard("funded", res)
        self.assertTrue(result.ok)
        self.assertEqual(result.output_path, "/tmp/run/18_central_lender_tape.csv")
        self.assertEqual(result.readiness["target_contract"], "mi_semantics")


class TestOrchestratorInvokeThreadsDataset(unittest.TestCase):
    """Criterion 2 — dataset reaches RealAgentAdapters."""

    def test_dataset_pipeline_forwarded(self):
        import apps.blob_trigger_app.orchestrator_invoke as OI
        captured = {}

        class _State:
            run_id = "r"; status = "done"; central_canonical_path = "/tmp/c.csv"
            blockers = []; portfolios = []
            def state_path(self):
                return "/tmp/x/run_state.json"

        with mock.patch("engine.orchestrator_agent.adapters.RealAgentAdapters",
                        side_effect=lambda **kw: captured.update(kw) or object()), \
             mock.patch("engine.orchestrator_agent.run_orchestration", return_value=_State()), \
             mock.patch("engine.orchestrator_agent.orchestrator.onboarding_mode_for_target",
                        return_value="mi_only"), \
             mock.patch("apps.blob_trigger_app.orchestrator_invoke._generate_investor_pptx"):
            OI.default_orchestrator_invoker(
                processing_mode="source_onboarding", client_id="ERE",
                source_portfolio_id="direct_001", source_portfolio_type="direct",
                dataset="pipeline", frequency="weekly", reporting_period="2025-09-08",
                input_path="/tmp/in", target="mi", run_regime=False,
                mapping_config_path=None, out_dir="/tmp/out")
        self.assertEqual(captured.get("dataset"), "pipeline")


class TestPipelineRegistrySeparation(unittest.TestCase):
    """Criteria 8, 9 — promoting pipeline/weekly creates a separate active record
    and leaves funded/monthly active + unchanged."""

    def _ctx(self, td):
        storage = Storage(Path(td))
        layout = Layout()
        persistence = ProductionPersistence(storage, layout)
        registry = SourceRegistry("blob://trakt-state/registry/source_registry.yaml",
                                  storage=storage)
        return storage, layout, persistence, registry

    class _Inv:
        def __call__(self, **kw):
            return {"run_id": "orun", "status": "done", "central_canonical_path": None,
                    "blockers": []}

    def _route(self, registry, persistence, td, marker, pack, meta=None):
        role_schemas = R.role_schemas_for_pack(registry, marker, "raw-v2")
        schema = fingerprint_pack(pack, role_schemas=role_schemas)
        return R.handle_blob_event(
            marker, registry=registry, out_dir=td, container="raw-v2",
            pack_marker="_READY.json", schema_info=schema,
            input_dir_override=str(Path(pack[0]).parent),
            pack_files=[Path(p).name for p in pack], orchestrator_invoker=self._Inv(),
            assembler_refresher=lambda **k: {}, persistence=persistence,
            marker_metadata=meta, now="2026-01-01T00:00:00+00:00")

    def test_promote_pipeline_does_not_touch_funded(self):
        with tempfile.TemporaryDirectory() as td:
            _, layout, persistence, registry = self._ctx(td)
            # An existing ACTIVE funded/monthly record (control).
            funded_pack = [str(Path(td) / "fund" / "LoanExtract.csv")]
            (Path(td) / "fund").mkdir()
            Path(funded_pack[0]).write_text("loan_id,balance,rate\n1,2,3\n")
            repin_source(registry, client_id="ERE", source_portfolio_id="direct_001",
                         dataset="funded", frequency="monthly", data_files=funded_pack,
                         source_book_type="direct", regime_required=True)
            funded_before = registry.lookup("ERE", "direct_001", "funded", "monthly")
            funded_fp = funded_before.expected_schema_fingerprint

            # New pipeline/weekly source → pending_review → approve → promote.
            wk = Path(td) / "wk"
            wk.mkdir()
            f = wk / "PipelineExtract.csv"
            f.write_text(",".join(_PIPE) + "\n" + ",".join(["x"] * len(_PIPE)) + "\n")
            marker = "raw-v2/ERE/direct/pipeline/weekly/direct_001/2025-09-08/_READY.json"
            m = self._route(registry, persistence, td, marker, [str(f)])
            pack_key = m["pack_key"]
            approved_uri = layout.run_onboarding_uri(pack_key, "34_target_first_decisions_approved.yaml")
            persistence.storage.write_text(approved_uri, "approved: true\n")

            src = OPS.promote_pack(persistence, registry, pack_key)
            self.assertEqual(src.key, "ERE/direct_001/pipeline/weekly")   # (8) separate key
            self.assertEqual(src.status, STATUS_ACTIVE)

            # (9) funded/monthly is a DIFFERENT, still-active, unchanged record.
            funded_after = registry.lookup("ERE", "direct_001", "funded", "monthly")
            self.assertEqual(funded_after.status, STATUS_ACTIVE)
            self.assertEqual(funded_after.expected_schema_fingerprint, funded_fp)
            self.assertTrue(funded_after.regime_required)
            self.assertNotEqual(src.key, funded_after.key)

            # (10) the next identical weekly pack is deterministic (no re-onboard).
            m2 = self._route(registry, persistence, td, marker, [str(f)],
                             meta={"force_reprocess": True})
            self.assertEqual(m2["decision"], R.DECISION_DETERMINISTIC)


class TestPipelineAssemblerSkip(unittest.TestCase):
    """Criteria 1, 2, 6 — for dataset=pipeline the funded platform assembler (which
    requires loan_identifier/unique_identifier) is skipped; the run reaches done
    with the stamped pipeline tape as the canonical. Funded still assembles."""

    def _adapters(self, tape):
        from engine.orchestrator_agent.adapters import AgentAdapters, StepResult

        class _A(AgentAdapters):
            def __init__(self):
                self.assemble_called = False

            def onboard(self, spec, work_dir):
                return StepResult(ok=True, output_path=tape, readiness={}, message="onb")

            def transform(self, spec, handoff, work_dir):
                raise AssertionError("transform must not run for lean MI")

            def validate(self, spec, manifest, work_dir):
                raise AssertionError("validate must not run for lean MI")

            def stamp_provenance(self, spec, canonical, out_dir):
                return StepResult(ok=True, output_path=canonical, readiness={}, message="stamp")

            def assemble(self, stamped, out_dir, client_id, target, regime=None):
                self.assemble_called = True
                return StepResult(ok=False, blocking=True,
                                  blockers=["no loan identifier column"], message="halt")

            def route_mi(self, central):
                return StepResult(ok=True, output_path=central, readiness={}, message="route")

            def project(self, central, out_dir, regime):
                raise AssertionError("project must not run for mi target")

        return _A()

    def test_pipeline_run_reaches_done_without_funded_assembler(self):
        from engine.orchestrator_agent.orchestrator import run_orchestration
        from engine.orchestrator_agent.adapters import PortfolioSpec
        from engine.orchestrator_agent.state import STEP_DONE
        with tempfile.TemporaryDirectory() as td:
            tape = Path(td) / "18a_central_pipeline_tape.csv"
            tape.write_text("application_id,pipeline_stage,interest_rate\nA1,offer,0.05\n")
            ad = self._adapters(str(tape))
            spec = PortfolioSpec(source_portfolio_id="direct_001", input=str(td))
            state = run_orchestration(
                "ERE", [spec], target="mi", out_root=str(Path(td) / "out"),
                adapters=ad, created_at="2026-01-01T00:00:00+00:00", dataset="pipeline")
            self.assertEqual(state.status, STEP_DONE)          # (6) no halt
            self.assertFalse(ad.assemble_called)               # (2) funded assembler skipped
            self.assertEqual(state.central_canonical_path, str(tape))  # (3) pipeline tape = canonical
            self.assertEqual(state.dataset, "pipeline")


class TestPipelineIdentity(unittest.TestCase):
    """Criterion 4 — an existing application/KFI id is used; otherwise a stable
    pipeline_row_id is generated."""

    def _parsed(self):
        from apps.blob_trigger_app.path_parser import parse_blob_path
        return parse_blob_path(
            "raw-v2/ERE/direct/pipeline/weekly/direct_001/2025-09-08/x.csv", "raw-v2")

    def test_existing_application_id_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "snap.csv"
            p.write_text("application_id,pipeline_stage\nA1,offer\n")
            out = R._ensure_pipeline_identity(str(p), self._parsed())
            self.assertEqual(out, str(p))                      # unchanged — id present

    def test_missing_id_generates_stable_row_id(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "snap.csv"
            p.write_text("pipeline_stage,interest_rate\noffer,0.05\ncompleted,0.04\n")
            out = R._ensure_pipeline_identity(str(p), self._parsed())
            import csv
            with open(out) as fh:
                rows = list(csv.DictReader(fh))
            self.assertIn("pipeline_row_id", rows[0])
            self.assertTrue(rows[0]["pipeline_row_id"].endswith("|2025-09-08|0"))
            self.assertTrue(rows[1]["pipeline_row_id"].endswith("|2025-09-08|1"))


class TestPipelineFullExtractPublished(unittest.TestCase):
    """The published snapshot is the FULL raw extract (rate/value/DOB), not the thin
    18a tape — so the React pipeline view maps the rich fields and derives the
    youngest-borrower age bucket (NNEG) from the DOBs."""

    _RAW_HEADERS = ("Company,Pool,Account Number,KFI Number,Broker,DOB App 1,DOB App 2,"
                    "Loan Amount,Estimated Value,Product Rate,Property Value,Status,"
                    "Application Submitted Date,Date Funds Released")

    def _parsed(self):
        from apps.blob_trigger_app.path_parser import parse_blob_path
        return parse_blob_path(
            "raw-v2/ERE/direct/pipeline/weekly/direct_001/2025-09-08/x.csv", "raw-v2")

    def test_snapshot_from_raw_preserves_rich_headers(self):
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td) / "M2L KFI and Pipeline.csv"
            raw.write_text(self._RAW_HEADERS + "\n"
                           "ACME,P1,A100,K200,BrokerX,1950-03-01,1948-07-09,"
                           "250000,600000,0.062,590000,Offer,2025-08-01,2025-09-05\n")
            out = R._pipeline_snapshot_from_raw(str(raw), self._parsed())
            self.assertIsNotNone(out)
            import pandas as pd
            cols = list(pd.read_csv(out, nrows=0).columns)
            for c in ("Product Rate", "Property Value", "DOB App 1", "DOB App 2"):
                self.assertIn(c, cols)

    def test_persist_publishes_raw_extract_not_thin_tape(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            storage = Storage(root)
            persistence = ProductionPersistence(storage, Layout())
            parsed = self._parsed()
            # A raw KFI extract in the pack + a THIN 18a tape from the run.
            pack = root / "pack"
            pack.mkdir()
            (pack / "M2L KFI and Pipeline.csv").write_text(
                self._RAW_HEADERS + "\nACME,P1,A100,K200,BrokerX,1950-03-01,1948-07-09,"
                "250000,600000,0.062,590000,Offer,2025-08-01,2025-09-05\n")
            thin = root / "18a_central_pipeline_tape.csv"
            thin.write_text("application_id,pipeline_stage,expected_funded_amount\nA100,offer,250000\n")

            manifest: dict = {}
            R._persist_pipeline_outputs(
                persistence, manifest, parsed,
                {"central_canonical_path": str(thin)},
                input_dir=str(pack), local_input_path=str(pack / "M2L KFI and Pipeline.csv"))
            self.assertEqual(manifest.get("pipeline_snapshot_source"), "raw_extract")

            published = persistence.pipeline_latest_path("ERE")
            self.assertIsNotNone(published)
            import pandas as pd
            cols = list(pd.read_csv(published, nrows=0).columns)
            self.assertIn("Product Rate", cols)          # rich field published
            self.assertIn("DOB App 1", cols)             # DOB present → NNEG age bucket
            self.assertNotIn("expected_funded_amount", cols)  # NOT the thin tape

    def test_mi_contract_derives_youngest_age_from_dob(self):
        # The existing MI pipeline contract already turns the applicant DOBs into the
        # youngest-borrower age bucket — the DOB columns just have to reach it.
        import yaml
        cfg = yaml.safe_load((_REPO / "config/mi/pipeline_field_contract.yaml").read_text())
        yba = cfg.get("fields", cfg).get("youngest_borrower_age") if isinstance(cfg, dict) else None
        # tolerate either a top-level or fields-nested layout
        if yba is None:
            for v in (cfg.values() if isinstance(cfg, dict) else []):
                if isinstance(v, dict) and "youngest_borrower_age" in v:
                    yba = v["youngest_borrower_age"]
                    break
        self.assertIsNotNone(yba, "youngest_borrower_age must exist in the pipeline contract")
        aliases = [str(a).lower() for a in (yba.get("dob_source_aliases") or [])]
        self.assertIn("dob app 1", aliases)
        self.assertIn("dob app 2", aliases)


if __name__ == "__main__":
    unittest.main()
