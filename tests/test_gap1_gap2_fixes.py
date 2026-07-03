#!/usr/bin/env python3
"""tests/test_gap1_gap2_fixes.py

Gap 1 — a SUCCESSFUL new-source onboarding run (one that used the LLM mapping
resolver) must persist its onboarding decision inputs + resolved-mapping artefacts,
so approve-recommendations → promote → rerun can capture and reuse the resolved
mapping. Previously only HALTED runs persisted them (via diagnostics), so a
succeeding new source left onboarding_decision_inputs empty and promote had no
mapping to pin.

Gap 2 — the fingerprint / header-signature path must re-detect a real header that
sits below row 1 (the same logic onboarding applies internally), so the pinned
signature carries business column names, not Unnamed:* noise.

Run: python -m unittest tests.test_gap1_gap2_fixes
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from apps.blob_trigger_app import ops as OPS
from apps.blob_trigger_app import router as R
from apps.blob_trigger_app.layout import Layout
from apps.blob_trigger_app.persistence import ProductionPersistence
from apps.blob_trigger_app.schema_fingerprint import compute_schema_fingerprint
from apps.blob_trigger_app.source_registry import (
    SourceRegistry, SourceRecord, STATUS_ACTIVE)
from apps.blob_trigger_app.storage import Storage

_NOW = "2026-01-01T00:00:00+00:00"
_LOAN = ["loan_id", "balance", "rate", "origination_date", "maturity_date"]


def _ctx(td):
    storage = Storage(Path(td))
    layout = Layout()
    persistence = ProductionPersistence(storage, layout)
    registry = SourceRegistry("blob://trakt-state/registry/source_registry.yaml", storage=storage)
    return storage, layout, persistence, registry


class _OnboardInvoker:
    """Mock new-source onboarding run that SUCCEEDS and writes the target-first
    decisions + resolver artefacts into a project dir, exactly like the real run."""

    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.project_dir.mkdir(parents=True, exist_ok=True)
        # Minimal but present decision inputs + a resolved-mapping artefact.
        (self.project_dir / "34_target_first_decisions.yaml").write_text("decisions: []\n")
        (self.project_dir / "36_target_first_llm_recommendations.json").write_text('{"rows": []}')
        (self.project_dir / "28a_target_coverage_matrix.json").write_text('{"coverage": []}')
        (self.project_dir / "22_llm_mapping_suggestions.json").write_text('{"suggestions": []}')

    def __call__(self, **kw):
        pid = kw.get("source_portfolio_id")
        return {"run_id": "orun", "status": "done", "central_canonical_path": None,
                "blockers": [], "onboarding_project_dirs": {pid: str(self.project_dir)}}


def _route_new_source(td, registry, persistence, invoker):
    pack = Path(td) / "pack"
    pack.mkdir(exist_ok=True)
    f = pack / "PipelineExtract.csv"
    f.write_text(",".join(_LOAN) + "\n" + ",".join(["x"] * len(_LOAN)) + "\n")
    from apps.blob_trigger_app.schema_fingerprint import fingerprint_pack
    marker = "raw-v2/ERE/direct/pipeline/weekly/direct_001/2026-W02/_READY.json"
    schema = fingerprint_pack([str(f)])
    return R.handle_blob_event(
        marker, registry=registry, out_dir=td, container="raw-v2",
        pack_marker="_READY.json", schema_info=schema, input_dir_override=str(pack),
        pack_files=["PipelineExtract.csv"], orchestrator_invoker=invoker,
        persistence=persistence, now=_NOW)


class TestGap1DecisionInputsPersisted(unittest.TestCase):

    def test_successful_new_source_persists_decision_inputs(self):
        with tempfile.TemporaryDirectory() as td:
            _, _, persistence, registry = _ctx(td)
            inv = _OnboardInvoker(Path(td) / "proj")
            m = _route_new_source(td, registry, persistence, inv)
            self.assertEqual(m["status"], "pending_review")

            rec = persistence.load_run_record(m["pack_key"])
            inputs = rec.get("onboarding_decision_inputs") or {}
            # The successful run's decisions + resolved mapping are now captured.
            self.assertTrue(inputs, "onboarding_decision_inputs should be non-empty")
            self.assertIn("34_target_first_decisions.yaml", inputs)
            self.assertIn("36_target_first_llm_recommendations.json", inputs)
            self.assertIn("22_llm_mapping_suggestions.json", inputs)  # resolved mapping captured
            # approve-recommendations can now find its inputs.
            self.assertTrue(persistence.storage.exists(inputs["34_target_first_decisions.yaml"]))

    def test_promote_pins_mapping_config_not_only_fingerprint(self):
        with tempfile.TemporaryDirectory() as td:
            _, layout, persistence, registry = _ctx(td)
            inv = _OnboardInvoker(Path(td) / "proj")
            m = _route_new_source(td, registry, persistence, inv)
            pack_key = m["pack_key"]

            # Simulate approve-recommendations having produced the approved decisions
            # file (accept_target_advice output) — what promote pins as the mapping.
            approved_uri = layout.run_onboarding_uri(pack_key, "34_target_first_decisions_approved.yaml")
            persistence.storage.write_text(approved_uri, "approved: true\n")
            self.assertTrue(persistence.has_approved_decisions(pack_key))

            src = OPS.promote_pack(persistence, registry, pack_key)
            # Promote pins the MAPPING CONFIG (the approved decisions), not just a fingerprint.
            self.assertEqual(src.mapping_config_path, approved_uri)
            self.assertIsNotNone(src.approved_mapping_id)
            self.assertEqual(src.status, STATUS_ACTIVE)
            self.assertTrue(src.has_approved_mapping)  # future months run deterministically


class TestGap2HeaderRedetection(unittest.TestCase):

    def _write(self, p: Path, text: str):
        p.write_text(text)
        return p

    def test_real_header_below_row1_pins_business_columns(self):
        with tempfile.TemporaryDirectory() as td:
            # A PropertyExtract-style CSV: a title/blank band, then the real header.
            csv = self._write(Path(td) / "PropertyExtract - Omni.csv",
                              "PROPERTY EXTRACT REPORT,,,\n"
                              ",,,\n"
                              "Loan ID,Property Value,Postcode,Valuation Date\n"
                              "1,250000,AB1 2CD,2020-01-01\n"
                              "2,310000,EF3 4GH,2021-06-01\n")
            info = compute_schema_fingerprint(csv)
            # Business columns, NOT Unnamed:* noise.
            self.assertIn("Loan ID", info.columns)
            self.assertIn("Property Value", info.columns)
            self.assertFalse(any(str(c).startswith("Unnamed:") for c in info.columns),
                             f"header still misaligned: {info.columns}")

    def test_clean_header_unchanged(self):
        with tempfile.TemporaryDirectory() as td:
            csv = self._write(Path(td) / "clean.csv",
                              "loan_id,balance,rate\n1,2,3\n")
            info = compute_schema_fingerprint(csv)
            self.assertEqual(info.columns, ["loan_id", "balance", "rate"])

    def test_repin_captures_business_columns_for_messy_property_extract(self):
        with tempfile.TemporaryDirectory() as td:
            _, _, _, registry = _ctx(td)
            pack = Path(td) / "nov"
            pack.mkdir()
            (pack / "LoanExtract.csv").write_text(",".join(_LOAN) + "\n1,2,3,4,5\n")
            (pack / "PropertyExtract - Omni.csv").write_text(
                "PROPERTY EXTRACT,,,\n,,,\nLoan ID,Property Value,Postcode\n1,250000,AB1 2CD\n")
            from apps.blob_trigger_app.repin import repin_source
            repin_source(registry, client_id="ERE", source_portfolio_id="direct_001",
                         dataset="funded", frequency="monthly",
                         data_files=[str(p) for p in sorted(pack.iterdir())],
                         source_book_type="direct", regime_required=True)
            rec = registry.lookup("ERE", "direct_001", "funded", "monthly")
            prop = rec.file_role_schemas.get("property_extract", [])
            self.assertIn("Property Value", prop)
            self.assertFalse(any(str(c).startswith("Unnamed:") for c in prop))


if __name__ == "__main__":
    unittest.main()
